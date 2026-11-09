"""GPU side-spectrum outputs and exact native accumulation for foil loss VJPs."""
import torch

from ..runtime.representation import spectral_scale


SPECTRAL_STATE = ("spectral_real", "spectral_imag", "spectral_norm")
RESPONSE = ("B", "rates", "D", "admittance_scale")


def prepare_spectral_outputs(solver, raw_output):
    """Keep both physical electric traces and bulk loss coefficients on device."""
    from ...postprocess.fdtd_loss import midpoint_trace_work_conductivity
    from ..runtime.representation import spectral_phasor_factor
    from ..copper_foil import copper_sampling_provenance
    descriptors = []
    for index, entry in enumerate(getattr(solver, "_copper_foils", ())):
        if "spectral_real" not in entry:
            continue
        indices = entry["indices"]
        adjacent = []
        for offset in (-1, 1):
            side = indices.clone()
            side[:, entry["face"].axis] += offset
            adjacent.append(side)
        conduction_volume = entry["material_widths"]*entry["area"][None]
        if "material_metric_factors" in entry:
            conduction_volume = conduction_volume * entry["material_metric_factors"]
        descriptors.append(dict(sampling=copper_sampling_provenance(solver, entry),entry_index=index, name=entry["face"].name,
            component=entry["electric_name"].removesuffix("_imag"),
            imaginary=entry["imaginary"], indices=indices,
            adjacent=tuple(adjacent), volume=entry["material_widths"]*entry["area"][None],
            conduction_volume=conduction_volume,
            area=entry["area"], frequencies=entry["spectral_frequencies"], dt=solver.dt,
            phasor_factor=spectral_phasor_factor(solver), integration_dt=solver._spectral_integration_dt,
            reference_frequency=solver._spectral_reference_frequency,
            dispersion_coefficient=midpoint_trace_work_conductivity(
                entry, solver.dt, entry["spectral_frequencies"]) *
                (entry["material_widths"]*entry["area"][None])[None],
            **{key:entry[key].clone() for key in (*SPECTRAL_STATE, *RESPONSE)}))
    raw_output["copper_spectral_outputs"] = tuple(descriptors)
    from ...monitors import PowerLossMonitor
    if descriptors or any(isinstance(monitor, PowerLossMonitor) and "conduction" in monitor.channels
                          for monitor in solver.scene.monitors):
        raw_output["loss_materials"] = {name:getattr(solver, "sigma_e_"+name).clone()
                                       for name in ("Ex", "Ey", "Ez")}


def pack_spectral_outputs(raw_output, outputs):
    templates = []
    for descriptor in raw_output.get("copper_spectral_outputs", ()):
        positions = {}
        for key in (*SPECTRAL_STATE, *RESPONSE):
            positions[key] = len(outputs)
            outputs.append(descriptor[key])
        templates.append((descriptor, positions))
    materials = {}
    for component, value in raw_output.get("loss_materials", {}).items():
        materials[component] = len(outputs)
        outputs.append(value)
    return tuple(templates), materials


def rebuild_spectral_outputs(pack, output_tensors, records):
    """Ordinary PyTorch loss algebra preserves phase, norm and direct coefficients."""
    from ..copper_foil import copper_spectral_power
    materials = {name:output_tensors[index] for name,index in pack.loss_material_templates.items()}
    entries = [(descriptor, {key:output_tensors[index] for key,index in positions.items()})
               for descriptor,positions in pack.copper_spectral_templates]
    for descriptor, values in entries:
        if descriptor["imaginary"]:
            continue
        phasor = torch.complex(values["spectral_real"], values["spectral_imag"])
        partner = next((other for other, _ in entries if other["imaginary"] and
            other["name"] == descriptor["name"] and other["component"] == descriptor["component"]), None)
        if partner is not None:
            other = next(value for entry,value in entries if entry is partner)
            phasor = phasor + 1j*torch.complex(other["spectral_real"], other["spectral_imag"])
        norm = values["spectral_norm"]
        phasor = phasor * spectral_scale(norm.clamp_min(1)[:,None,None],
            phasor_factor=descriptor["phasor_factor"], integration_dt=descriptor["integration_dt"],
            reference_frequency=descriptor["reference_frequency"],
            frequency=descriptor["frequencies"][:, None, None])
        frequency, dt = descriptor["frequencies"], descriptor["dt"]
        B, rates, D, scale = (values[key] for key in RESPONSE)
        power = copper_spectral_power(phasor, frequency, dt, B, rates, D, descriptor["area"] * scale)
        record = records[descriptor["name"]]
        record["frequencies"] = frequency
        record["spectral_valid"] = norm > 0
        record["surface_power"] = record.get("surface_power", 0.) + power
        sigma = materials[descriptor["component"]]
        sides = torch.stack([sigma[tuple(side.T)].double() for side in descriptor["adjacent"]])
        trace = dict(sampling=dict(descriptor["sampling"], weight_sums=norm.clone()), component=descriptor["component"], indices=descriptor["indices"],
            electric_fields=phasor,
            conduction_coefficient=(sides*descriptor["conduction_volume"])[None].expand(len(frequency),-1,-1),
            dispersion_coefficient=descriptor["dispersion_coefficient"])
        record.setdefault("bulk_electric_traces", []).append(trace)
    return materials


def replay_copper_spectra(solver, previous, updated):
    from ..copper_foil import copper_spectral_weights
    result = {}
    for index, entry in enumerate(getattr(solver, "_copper_foils", ())):
        prefix = f"copper_foil_{index}_"
        if prefix+"spectral_step" not in previous:
            continue
        step = previous[prefix+"spectral_step"]
        weight = copper_spectral_weights(entry, step)
        angle = 2*torch.pi*entry["spectral_frequencies"]*solver.dt*(step+1)
        electric = torch.stack((updated[entry["electric_name"]][entry["index"]],
                                updated[prefix+"right_e"])).double()
        result[prefix+"spectral_real"] = previous[prefix+"spectral_real"] + (weight*torch.cos(angle))[:,None,None]*electric
        result[prefix+"spectral_imag"] = previous[prefix+"spectral_imag"] + (weight*torch.sin(angle))[:,None,None]*electric
        result[prefix+"spectral_norm"] = previous[prefix+"spectral_norm"] + weight
        result[prefix+"spectral_step"] = step+1
    return result


def seed_spectral_outputs(pack, grad_outputs, adjoint_state, grad_sigma, grad_copper):
    for descriptor, positions in pack.copper_spectral_templates:
        index = descriptor["entry_index"]
        for key in SPECTRAL_STATE:
            gradient = grad_outputs[positions[key]]
            if gradient is not None:
                adjoint_state[f"copper_foil_{index}_{key}"].add_(gradient)
        for key in RESPONSE:
            gradient = grad_outputs[positions[key]]
            if gradient is not None:
                grad_copper[index][key].add_(gradient)
    for component,index in pack.loss_material_templates.items():
        gradient = grad_outputs[index]
        if gradient is not None:
            grad_sigma[component].add_(gradient)
