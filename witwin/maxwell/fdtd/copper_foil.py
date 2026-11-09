"""Implicit passive two-sided copper-foil coupling to the Yee field update."""
from __future__ import annotations

import os

import numpy as np
import torch

from ..compiler.copper_foil import (
    compile_copper_faces, compile_copper_response, copper_edge_admittance_scale, warn_copper_edge_features,
)
from .copper_foil_ade import (
    initialize_copper_trace_ade, prepare_copper_trace_history,
    finalize_copper_trace_ade, iter_copper_trace_ade_state,
)


_FIELDS = ("Ex", "Ey", "Ez")
_MAGNETIC = ("Hx", "Hy", "Hz")


def _selector(indices):
    return tuple(indices[:, axis] for axis in range(3))


def _flat_offsets(shape, indices):
    return (indices[:, 0] * shape[1] + indices[:, 1]) * shape[2] + indices[:, 2]


def prepare_copper_magnetic_support(solver, face, indices):
    """Prepare the shared split-trace magnetic curl stencil without modal state."""
    component, normal = face.component, face.axis
    tangent = 3 - component - normal
    sign = float(np.cross(np.eye(3)[normal], np.eye(3)[component])[tangent])
    normal_axis = "xyz"[normal]
    h_low_indices, h_high_indices = indices.clone(), indices.clone()
    h_low_indices[:, normal] -= 1
    h_low_index, h_high_index = _selector(h_low_indices), _selector(h_high_indices)
    h_high_gain = getattr(solver, "c" + _MAGNETIC[tangent].lower() + "_curl")[h_high_index].double()
    h_high_gain = sign * h_high_gain * float(getattr(solver, f"inv_d{normal_axis}_h")[face.normal_index])
    normal_h = getattr(solver, _MAGNETIC[normal])
    low_indices, high_indices = indices.clone(), indices.clone()
    low_indices[:, tangent] -= 1
    # The periodic seam uses one physical node; duplicate terminal nodes are
    # handled by the native boundary synchronization and carry half dual area.
    low_valid = (low_indices[:, tangent] >= 0) & (low_indices[:, tangent] < normal_h.shape[tangent])
    high_valid = (high_indices[:, tangent] >= 0) & (high_indices[:, tangent] < normal_h.shape[tangent])
    safe_low = low_indices.clone()
    safe_high = high_indices.clone()
    safe_low[:, tangent].clamp_(0, normal_h.shape[tangent] - 1)
    safe_high[:, tangent].clamp_(0, normal_h.shape[tangent] - 1)
    gather_low, gather_high = safe_low.clone(), safe_high.clone()
    paired = solver.scene.boundary.axis_kind("xyz"[tangent]) in ("periodic", "bloch")
    if paired:
        gather_low[~low_valid, tangent] = normal_h.shape[tangent] - 1
        gather_high[~high_valid, tangent] = 0
    phase = 0.
    if solver.scene.boundary.axis_kind("xyz"[tangent]) == "bloch":
        limits = solver.scene.domain.bounds[tangent]
        phase = float(solver.resolved_bloch_wavevector[tangent]) * (limits[1] - limits[0])
    normal_curl = getattr(solver, "c" + _MAGNETIC[normal].lower() + "_curl")
    inv_tangent = getattr(solver, f"inv_d{'xyz'[tangent]}_h")
    low_gain = sign * normal_curl[_selector(safe_low)].double() * inv_tangent[safe_low[:, tangent]].double()
    high_gain = -sign * normal_curl[_selector(safe_high)].double() * inv_tangent[safe_high[:, tangent]].double()
    tangent_dual = torch.as_tensor(getattr(solver.scene, f"d{'xyz'[tangent]}_dual64"), device=solver.device)
    tangent_primal = torch.as_tensor(getattr(solver.scene, f"d{'xyz'[component]}_primal64"), device=solver.device)
    area = tangent_primal[indices[:, component]] * tangent_dual[indices[:, tangent]]
    # Boundary node traces represent half of a physical dual face; the
    # duplicated periodic endpoint likewise must not count a second area.
    tangent_nodes = len(getattr(solver.scene, f"{'xyz'[tangent]}_nodes64"))
    endpoint = (indices[:, tangent] == 0) | (indices[:, tangent] == tangent_nodes - 1)
    area = area * torch.where(endpoint, .5, 1.)
    return dict(indices=indices, index=_selector(indices), sign=sign,
        electric_name=_FIELDS[component], magnetic_name=_MAGNETIC[tangent],
        normal_magnetic_name=_MAGNETIC[normal], low_h=h_low_index, high_h=h_high_index,
        high_h_gain=h_high_gain, low_normal_h=_selector(gather_low), high_normal_h=_selector(gather_high),
        low_valid=low_valid, high_valid=high_valid,
        low_sources=torch.nonzero(low_valid, as_tuple=False).reshape(-1),
        high_sources=torch.nonzero(high_valid, as_tuple=False).reshape(-1),
        paired=paired, phase=phase,
        low_normal_offsets=_flat_offsets(normal_h.shape, safe_low)[low_valid],
        high_normal_offsets=_flat_offsets(normal_h.shape, safe_high)[high_valid],
        low_normal_gain=low_gain[low_valid], high_normal_gain=high_gain[high_valid],
        inv_tangent_dual=1 / tangent_dual[indices[:, tangent]], area=area)


def copper_contact_masks(solver, *, frequency, attachments):
    """Boolean E-edge masks of conductors and attachments that continue a foil edge.

    PEC (zero curl), conformal PEC (any edge with open fraction < 1), good
    conductors (sigma >= 100 omega eps at the highest foil band edge), thin-wire
    sample/deposition edges and, once prepared, port voltage and lumped-element
    edges. A foil edge touching any of them is not a free edge.
    """
    from .runtime.materials import _pec_edge_open_fractions
    fractions = _pec_edge_open_fractions(solver)
    masks = {}
    for name in _FIELDS:
        curl = getattr(solver, "c" + name.lower() + "_curl")
        mask = curl == 0
        if fractions is not None:
            mask = mask | (torch.as_tensor(fractions[name], device=curl.device) < 1)
        sigma = getattr(solver, "sigma_e_" + name, None)
        eps = getattr(solver, "eps_" + name, None)
        if torch.is_tensor(sigma) and torch.is_tensor(eps):
            mask = mask | (sigma >= 100 * 2 * np.pi * frequency * eps)
        masks[name] = mask.clone()
    runtime = getattr(solver, "_wire_runtime", None)
    if runtime is not None:
        for plan in (runtime.target_plan, runtime.sample_plan):
            for name, (_destination, offsets) in zip(_FIELDS, plan.entries):
                masks[name].view(-1)[offsets.to(masks[name].device)] = True
    if attachments:
        for port in getattr(solver, "_port_runtimes", ()):
            geometry = port.geometry
            name = getattr(geometry, "voltage_component", None)
            if name in masks:
                indices = geometry.voltage_indices.to(device=masks[name].device, dtype=torch.int64)
                masks[name][tuple(indices.T)] = True
        for runtime, field_name in getattr(solver, "_lumped_element_runtimes", ()):
            name = field_name.removesuffix("_imag")
            if name in masks:
                masks[name].view(-1)[runtime.linear_indices.to(masks[name].device)] = True
    return masks


def refresh_copper_trace_matrices(entry, dt):
    """Rebuild the implicit two-trace matrices from the current admittance scale."""
    scaled = entry["effective"][None] * entry["admittance_scale"][:, None, None]
    conductance = entry["trace_conductance"]
    lhs = scaled / 2 + torch.diag_embed((entry["capacity"] / dt + conductance / 2).T)
    entry["inverse"].copy_(torch.linalg.inv(lhs))
    entry["rhs_matrix"].copy_(-scaled / 2 + torch.diag_embed((entry["capacity"] / dt - conductance / 2).T))


def finalize_copper_edges(solver):
    """Re-evaluate free edges once ports and lumped elements are attached."""
    entries = [entry for entry in getattr(solver, "_copper_foils", ()) if entry["edge_geometry"] is not None]
    if not entries:
        return
    attached = any(getattr(solver, name, ()) for name in ("_port_runtimes", "_lumped_element_runtimes"))
    if not attached:
        return
    masks = copper_contact_masks(solver, frequency=max(entry["face"].material.frequency_range[1] for entry in entries),
                                 attachments=True)
    # Real and imaginary banks share one scale tensor; refresh every bank once.
    groups = {}
    for entry in entries:
        groups.setdefault(id(entry["admittance_scale"]), []).append(entry)
    for group in groups.values():
        first = group[0]
        scale, geometry = copper_edge_admittance_scale(solver.scene, first["face"], first["indices"],
                                                       solver._copper_edge_faces, contact=masks)
        if torch.equal(scale, first["admittance_scale"]):
            continue
        first["admittance_scale"].copy_(scale)
        refreshed = set()
        for entry in group:
            entry["edge_geometry"] = geometry
            if id(entry["inverse"]) not in refreshed:
                refresh_copper_trace_matrices(entry, solver.dt)
                refreshed.add(id(entry["inverse"]))


def initialize_copper_foils(solver):
    solver._copper_foils = []
    faces = compile_copper_faces(solver.scene, device=solver.device)
    if not faces:
        return
    if solver.full_aniso_enabled or solver.nonlinear_enabled:
        raise NotImplementedError("CopperFoil currently requires isotropic linear Yee fields.")
    response_cache = {}
    occupied = {}
    # Free-edge detection must see neighbour traces owned by adjacent shards.
    all_faces = (compile_copper_faces(solver.scene, device=solver.device, include_halos=True)
                 if getattr(solver.scene, "_fdtd_shard_layout", None) is not None else faces)
    solver._copper_edge_faces = all_faces
    contact = None
    for face in faces:
        material = face.material
        frequencies = (solver.source_frequency, *getattr(solver, "_requested_port_frequencies", ()))
        material._frequencies(frequencies)
        if id(material) not in response_cache:
            response_cache[id(material)] = compile_copper_response(material)
        response = response_cache[id(material)]
        component, normal = face.component, face.axis
        indices = face.indices
        electric = getattr(solver, _FIELDS[component])
        curl = getattr(solver, "c" + _FIELDS[component].lower() + "_curl")
        active = curl[_selector(indices)] != 0
        contact_count = int((~active).sum())
        # Explicit PEC contact geometry owns the contacted tangential trace.
        # Its zero electric trace needs no copper state and remains a physical
        # ideal-conductor connection, rather than a second surface owner.
        indices = indices[active]
        if not len(indices):
            continue
        offsets = _flat_offsets(electric.shape, indices)
        owned = torch.cat((occupied.get(component, offsets.new_empty(0)), offsets)).sort().values
        if bool((owned[1:] == owned[:-1]).any()):
            raise ValueError("CopperFoil interfaces have conflicting two-sided state ownership.")
        occupied[component] = owned
        index = _selector(indices)
        left_indices, right_indices = indices.clone(), indices.clone()
        left_indices[:, normal] -= 1
        right_indices[:, normal] += 1
        left_index, right_index = _selector(left_indices), _selector(right_indices)
        normal_axis = "xyz"[normal]
        primal = getattr(solver.scene, f"d{normal_axis}_primal64")
        left_width, right_width = float(primal[face.normal_index - 1]) / 2, float(primal[face.normal_index]) / 2
        eps = getattr(solver, "eps_" + _FIELDS[component])
        sigma = getattr(solver, "sigma_e_" + _FIELDS[component])
        capacity = torch.stack((eps[left_index].double() * left_width,
                                eps[right_index].double() * right_width))
        conductance = torch.stack((sigma[left_index].double() * left_width,
                                   sigma[right_index].double() * right_width))
        chart=getattr(solver.scene,"coordinates",None)
        if chart is not None:
            if normal!=1 or component not in (0,2):
                raise NotImplementedError("Mapped copper requires chart-normal u with tangential s/z traces.")
            from ..compiler.coordinates import copper_half_metric_average
            surface_u=float(solver.scene.y_nodes64[face.normal_index])
            surface_h=1-chart.curvature*surface_u
            factors=[]
            for side,width in ((-1,left_width),(1,right_width)):
                sample_u=float(solver.scene.y_nodes64[face.normal_index+side])
                sample_h=1-chart.curvature*sample_u
                sample_metric=1/sample_h if component==0 else sample_h
                factors.append(copper_half_metric_average(chart,surface_u,width,side,component)/sample_metric)
            # The physical side material is constant over each exterior half
            # dual cell. Integrate its varying metric, rather than reusing the
            # adjacent full-cell material sample at a different u coordinate.
            half_metric=torch.tensor(factors,device=solver.device,dtype=torch.float64)[:,None]
            capacity=capacity*half_metric
            conductance=conductance*half_metric
        rates = response.rates.to(device=solver.device)
        B = response.state_space.B.to(device=solver.device)
        D = response.state_space.D.to(device=solver.device)
        if chart is not None:
            alpha=1/surface_h if component==0 else surface_h
            # Covariant tangential fields and chart surface measure transform
            # the passive two-port admittance by alpha. Square-root scaling
            # preserves its state energy and physical dissipated power.
            B=B*np.sqrt(alpha)
            D=D*alpha
        denominator = 1 + solver.dt * rates / 2
        decay = (1 - solver.dt * rates / 2) / denominator
        drive = solver.dt / 2 * B / denominator[:, None]
        effective = D + B.T @ drive
        edge_count = len(indices)
        # Free foil edges carry the finite-thickness edge impedance kappa*Z;
        # every other trace keeps an exact unit scale.
        if contact is None:
            contact = copper_contact_masks(solver, frequency=max(item.material.frequency_range[1] for item in faces),
                                           attachments=False)
        admittance_scale, edge_geometry = copper_edge_admittance_scale(solver.scene, face, indices, all_faces,
                                                                       contact=contact)
        warn_copper_edge_features(solver.scene, face, indices, all_faces, edge_geometry, contact=contact)
        scaled_effective = effective[None] * admittance_scale[:, None, None]
        lhs = scaled_effective / 2
        lhs[:, 0, 0] += capacity[0] / solver.dt + conductance[0] / 2
        lhs[:, 1, 1] += capacity[1] / solver.dt + conductance[1] / 2
        inverse = torch.linalg.inv(lhs)
        rhs_matrix = -scaled_effective / 2
        rhs_matrix[:, 0, 0] += capacity[0] / solver.dt - conductance[0] / 2
        rhs_matrix[:, 1, 1] += capacity[1] / solver.dt - conductance[1] / 2
        magnetic_support = prepare_copper_magnetic_support(solver, face, indices)
        solver._copper_foils.append(dict(
            **magnetic_support, face=face, response=response,
            pec_contact_edge_count=contact_count, imaginary=False,
            left_width=left_width, right_width=right_width, capacity=capacity, conductance=conductance,
            B=B, D=D, rates=rates, decay=decay, drive=drive, inverse=inverse, rhs_matrix=rhs_matrix,
            admittance_scale=admittance_scale, edge_geometry=edge_geometry,
            effective=effective, trace_conductance=conductance,
            state=torch.zeros((len(rates), edge_count), dtype=torch.float64, device=solver.device),
            right_e=torch.zeros(edge_count, dtype=electric.dtype, device=solver.device),
            old_e=torch.zeros((2, edge_count), dtype=torch.float64, device=solver.device),
            cumulative_loss=torch.zeros((), dtype=torch.float64, device=solver.device),
        ))
        if chart is not None:
            solver._copper_foils[-1].update(material_metric_factors=half_metric,
                                           response_alpha=alpha)
        if solver.complex_fields_enabled:
            imaginary = dict(solver._copper_foils[-1])
            for name in ("electric_name", "magnetic_name", "normal_magnetic_name"):
                imaginary[name] += "_imag"
            for name in ("state", "right_e", "old_e", "cumulative_loss"):
                imaginary[name] = torch.zeros_like(imaginary[name])
            imaginary["imaginary"] = True
            solver._copper_foils.append(imaginary)

    initialize_copper_trace_ade(solver)
    if solver.device.type == "cuda" and os.environ.get("WITWIN_MAXWELL_COPPER_KERNEL", "fused") == "fused":
        from .cuda.backend import prepare_copper_electric
        for entry in solver._copper_foils:
            if not (entry["imaginary"] or entry["paired"] or entry["phase"] or entry["material_ade"]):
                prepare_copper_electric(solver, entry)


def begin_copper_step(solver):
    for entry in solver._copper_foils:
        if "fused_electric" in entry:
            from .cuda.backend import begin_copper_step as fused_begin
            fused_begin(solver, entry)
            continue
        entry["old_e"][0].copy_(getattr(solver, entry["electric_name"])[entry["index"]])
        entry["old_e"][1].copy_(entry["right_e"])


def correct_copper_magnetic_support(solver, entry, difference):
    """Apply the same magnetic stencil to owned traces or received halo traces."""
    magnetic = getattr(solver, entry["magnetic_name"])
    magnetic[entry["high_h"]] += (entry["high_h_gain"] * difference).to(magnetic.dtype)
    normal_h = getattr(solver, entry["normal_magnetic_name"])
    weighted = difference * entry["right_width"] / (entry["left_width"] + entry["right_width"])
    normal_h.view(-1).index_add_(0, entry["low_normal_offsets"],
        (entry["low_normal_gain"] * weighted.index_select(0, entry["low_sources"])).to(normal_h.dtype))
    normal_h.view(-1).index_add_(0, entry["high_normal_offsets"],
        (entry["high_normal_gain"] * weighted.index_select(0, entry["high_sources"])).to(normal_h.dtype))


def correct_copper_magnetic(solver, *, activity=1):
    for entry in solver._copper_foils:
        if "fused_electric" in entry:
            from .cuda.backend import correct_copper_magnetic as fused_magnetic
            fused_magnetic(solver, entry, activity=activity)
            continue
        correct_copper_magnetic_support(solver, entry,
            (entry["old_e"][1] - entry["old_e"][0]) * activity)


def advance_copper_electric(solver, *, activity=1):
    for entry in solver._copper_foils:
        if "fused_electric" in entry:
            from .cuda.backend import advance_copper_electric as fused_electric
            fused_electric(solver, entry, activity=activity)
            continue
        magnetic = getattr(solver, entry["magnetic_name"])
        normal_h = getattr(solver, entry["normal_magnetic_name"])
        low_normal = normal_h[entry["low_normal_h"]].double()
        high_normal = normal_h[entry["high_normal_h"]].double()
        if entry["paired"]:
            if entry["phase"]:
                import math
                companion_name = entry["normal_magnetic_name"]
                companion_name = companion_name.removesuffix("_imag") if entry["imaginary"] else companion_name + "_imag"
                companion = getattr(solver, companion_name)
                low_other, high_other = companion[entry["low_normal_h"]].double(), companion[entry["high_normal_h"]].double()
                sine = math.sin(entry["phase"]) * (-1 if entry["imaginary"] else 1)
                cosine = math.cos(entry["phase"])
                low_normal = torch.where(entry["low_valid"], low_normal, cosine * low_normal + sine * low_other)
                high_normal = torch.where(entry["high_valid"], high_normal, cosine * high_normal - sine * high_other)
        else:
            low_normal *= entry["low_valid"]
            high_normal *= entry["high_valid"]
        transverse = entry["sign"] * (high_normal - low_normal) * entry["inv_tangent_dual"]
        forcing = torch.stack((entry["sign"] * magnetic[entry["low_h"]].double() + entry["left_width"] * transverse,
                              -entry["sign"] * magnetic[entry["high_h"]].double() + entry["right_width"] * transverse))
        if entry["material_ade"]:
            material_current = prepare_copper_trace_history(entry, solver.dt)
            forcing = forcing - entry["material_widths"] * material_current
        state = entry["state"]
        history = ((entry["B"].T * ((1 + entry["decay"]) / 2)[None, :]) @ state) * entry["admittance_scale"]
        rhs = torch.einsum("mij,jm->im", entry["rhs_matrix"], entry["old_e"]) + forcing - history
        updated = torch.einsum("mij,jm->im", entry["inverse"], rhs)
        if torch.is_tensor(activity):
            updated = torch.where(activity != 0, updated, entry["old_e"])
        electric = getattr(solver, entry["electric_name"])
        electric[entry["index"]] = updated[0].to(electric.dtype)
        entry["right_e"].copy_(updated[1])
        if entry["material_ade"]:
            finalize_copper_trace_ade(entry, updated, solver.dt, activity=activity)
        midpoint_e = (updated + entry["old_e"]) / 2
        next_state = entry["decay"][:, None] * state + entry["drive"] @ (updated + entry["old_e"])
        if torch.is_tensor(activity):
            next_state = torch.where(activity != 0, next_state, state)
        midpoint_state = (next_state + state) / 2
        loss = (entry["rates"][:, None] * midpoint_state.square()).sum(0)
        loss += torch.sum(midpoint_e * (entry["D"] @ midpoint_e), 0)
        loss = loss * entry["admittance_scale"]
        entry["cumulative_loss"].add_(solver.dt * torch.sum(loss * entry["area"]) * activity)
        state.copy_(next_state)


def iter_copper_state(solver):
    for index, entry in enumerate(getattr(solver, "_copper_foils", ())):
        for name in ("state", "right_e", "cumulative_loss", "spectral_step", "spectral_norm", "spectral_real", "spectral_imag"):
            if name not in entry:
                continue
            yield f"copper_foil_{index}_{name}", entry[name]
        yield from iter_copper_trace_ade_state(index, entry)


@torch.no_grad()
def copper_sampling_provenance(solver, entry):
    """Snapshot the executed side-spectrum clock, without changing its phasor."""
    from .runtime.representation import spectral_phasor_factor, energy_representation
    return dict(energy_representation=energy_representation(solver), schema="native-copper-electric-sampling/1", dt=float(solver.dt),
        owner=entry["face"].name, component=entry["electric_name"],
        normal_axis="xyz"[entry["face"].axis], normal_index=entry["face"].normal_index,
        frequencies=entry["spectral_frequencies"].detach().cpu().clone(),
        start_steps=entry["spectral_starts"].detach().cpu().clone(),
        planned_steps=entry["spectral_end"],
        completed_steps=entry["spectral_step"].detach().cpu().clone(),
        window=entry["spectral_window"],
        weight_sums=entry["spectral_norm"].detach().cpu().clone(),
        phasor_factor=spectral_phasor_factor(solver), normalize_source=False,
        reference_frequency=("per-frequency" if solver._spectral_reference_frequency is None
                             else solver._spectral_reference_frequency),
        integration_dt=solver._spectral_integration_dt,
        electric_phase_origin_step=1, electric_time_offset_steps=0.0,
        phasor_convention="exp(-iwt)")


@torch.no_grad()
def copper_loss_report(solver):
    """Detached conductor work/state in joules, with explicit field representation.

    A complex run reports the combined two-bank algorithmic work and storage;
    those values are not instantaneous real-projected transient heat. Spectral
    power uses physical peak phasors and its ordinary one-half work factor.
    """
    result = {}
    from .runtime.representation import energy_representation, spectral_phasor_factor, spectral_scale
    for entry in getattr(solver, "_copper_foils", ()):
        record = result.setdefault(entry["face"].name, dict(dissipated_energy=0., stored_energy=0.,
            frequency_range=entry["response"].frequency_range, max_relative_model_error=entry["response"].max_relative_error,
            certified_relative_model_error_bound=entry["response"].relative_error_bound,
            pec_contact_edge_count=0, active_tangential_edge_count=0, free_edge_trace_count=0))
        record["energy_representation"] = energy_representation(solver)
        record["dissipated_energy"] += float(entry["cumulative_loss"].detach())
        record["stored_energy"] += float((entry["state"].square() * entry["area"] * entry["admittance_scale"]).sum().detach()) / 2
        if not entry["imaginary"]:
            record["pec_contact_edge_count"] += entry["pec_contact_edge_count"]
            record["active_tangential_edge_count"] += len(entry["indices"])
            record["free_edge_trace_count"] += int((entry["admittance_scale"] != 1).sum())
        if entry["imaginary"] or "spectral_real" not in entry:
            continue
        phasor = torch.complex(entry["spectral_real"], entry["spectral_imag"])
        partner = next((candidate for candidate in solver._copper_foils
            if candidate["imaginary"] and candidate["face"] is entry["face"]
            and candidate["electric_name"] == entry["electric_name"] + "_imag"), None)
        if partner is not None:
            phasor += 1j * torch.complex(partner["spectral_real"], partner["spectral_imag"])
        normalization = entry["spectral_norm"]
        phasor *= spectral_scale(normalization.clamp_min(1)[:, None, None],
                                 phasor_factor=spectral_phasor_factor(solver),
                                 integration_dt=solver._spectral_integration_dt,
                                 reference_frequency=solver._spectral_reference_frequency,
                                 frequency=entry["spectral_frequencies"][:, None, None])
        frequency = entry["spectral_frequencies"]
        power, edge_power = copper_spectral_power(phasor, frequency, solver.dt,
            entry["B"], entry["rates"], entry["D"], entry["area"] * entry["admittance_scale"], return_edge_power=True)
        record.setdefault("frequencies", frequency.detach().cpu().clone())
        record.setdefault("coordinate_frame", "mapped_chart" if getattr(solver.scene, "coordinates", None) is not None else "cartesian")
        record.setdefault("periodic_axes", tuple(axis for axis in "xyz"
            if solver.scene.boundary.axis_kind(axis) in ("periodic", "bloch")))
        record.setdefault("surface_power", torch.zeros_like(power, device="cpu"))
        record.setdefault("spectral_valid", (normalization > 0).detach().cpu().clone())
        record["surface_power"] += power.detach().cpu()
        from ..compiler.power_loss import _component_cell_bounds
        bounds = _component_cell_bounds(solver.scene, entry["electric_name"])
        indices = entry["indices"]
        lower = torch.stack([bounds[axis][0][indices[:, axis]] for axis in range(3)], dim=-1)
        upper = torch.stack([bounds[axis][1][indices[:, axis]] for axis in range(3)], dim=-1)
        normal = entry["face"].axis
        coordinate = float(getattr(solver.scene, "xyz"[normal]+"_nodes64")[entry["face"].normal_index])
        lower[:, normal] = upper[:, normal] = coordinate
        record.setdefault("surface_power_traces", []).append(dict(
            component=entry["electric_name"], indices=indices.detach().cpu().clone(),
            power=edge_power.detach().cpu().clone(), area=entry["area"].detach().cpu().clone(),
            edge_impedance_factor=(1 / entry["admittance_scale"]).detach().cpu().clone(),
            lower_bounds=lower.detach().cpu().clone(), upper_bounds=upper.detach().cpu().clone(),
            normal_axis="xyz"[normal], normal_coordinate=coordinate,
            autograd_provenance="detached spectral diagnostics; no ROI gradient claim"))
        from ..postprocess.fdtd_loss import midpoint_trace_work_conductivity
        indices = entry["indices"]
        volume = entry["material_widths"]*entry["area"][None]
        dispersion = midpoint_trace_work_conductivity(entry, solver.dt, frequency)
        record.setdefault("bulk_electric_traces", []).append(dict(
            sampling=copper_sampling_provenance(solver, entry),
            component=entry["electric_name"], indices=indices.detach().cpu().clone(),
            normal_axis="xyz"[normal], normal_coordinate=coordinate,
            normal_half_widths=entry["material_widths"].detach().cpu().clone(),
            electric_fields=phasor.detach().cpu().clone(),
            conduction_coefficient=(entry["conductance"]*entry["area"][None])[None].expand(len(frequency), -1, -1).detach().cpu().clone(),
            dispersion_coefficient=(dispersion*volume[None]).detach().cpu().clone()))

    return result


def prepare_copper_spectra(solver, time_steps, frequencies, window):
    """Allocate surface-only phasor buffers using the shared sampling schedule."""
    if not solver._copper_foils:
        return
    from .runtime.spectral import compute_spectral_start_step, resolve_spectral_window_type
    values = (float(frequencies),) if isinstance(frequencies, (int, float)) else tuple(map(float, frequencies))
    kind = resolve_spectral_window_type(solver, window)
    starts = [compute_spectral_start_step(solver, frequency, window_type=kind) for frequency in values]
    signature = (values, tuple(starts), int(time_steps), kind)
    changed = False
    for entry in solver._copper_foils:
        entry["face"].material._frequencies(values)
        if entry.get("spectral_signature") == signature:
            for name in ("spectral_step", "spectral_norm", "spectral_real", "spectral_imag"):
                entry[name].zero_()
            continue
        changed = True
        entry["spectral_signature"] = signature
        entry["spectral_frequencies"] = torch.tensor(values, device=solver.device, dtype=torch.float64)
        entry["spectral_starts"] = torch.tensor(starts, device=solver.device, dtype=torch.float64)
        entry["spectral_end"] = int(time_steps)
        entry["spectral_window"] = kind
        entry["spectral_step"] = torch.zeros((), device=solver.device, dtype=torch.float64)
        entry["spectral_norm"] = torch.zeros(len(values), device=solver.device, dtype=torch.float64)
        shape = (len(values), 2, len(entry["indices"]))
        entry["spectral_real"] = torch.zeros(shape, device=solver.device, dtype=torch.float64)
        entry["spectral_imag"] = torch.zeros(shape, device=solver.device, dtype=torch.float64)
    if changed:
        solver._copper_spectral_generation = getattr(solver, "_copper_spectral_generation", 0) + 1


def copper_spectral_power(phasor, frequencies, dt, B, rates, D, area, *, return_edge_power=False):
    """Endpoint work of the native midpoint realization, optionally per tile.

    The default total keeps its original summation order; per-tile powers can
    differ from that sum only by floating-point reduction roundoff.
    """
    if frequencies.ndim != 1 or frequencies.numel() == 0:
        raise ValueError("frequencies must have non-empty shape [F].")
    angle = torch.pi * frequencies * dt
    warped = torch.tan(angle) / (torch.pi * dt)
    if not bool(torch.all(torch.isfinite(warped))) or not bool(torch.all(warped >= 0)):
        raise ValueError("frequencies must be finite and non-negative.")
    s = -2j * torch.tan(angle) / dt
    admittance = D.to(torch.complex128)[None] + torch.einsum("ni,nj,fn->fij",
        B.to(torch.complex128), B.to(torch.complex128), 1 / (s[:, None] + rates[None]))
    power = .5 * torch.einsum("fim,fij,fjm->fm", phasor.conj(), admittance, phasor).real
    total = (power * area).sum(-1) * torch.cos(angle).square()
    if return_edge_power:
        return total, power*area*torch.cos(angle).square()[:, None]
    return total


def copper_spectral_weights(entry, step, *, activity=1):
    """The sampling window evaluated on the native side-spectrum clock."""
    start, end = entry["spectral_starts"], entry["spectral_end"]
    valid = (step >= start) & (step < end)
    fraction = (step - start) / torch.clamp(end - start, min=1)
    kind = entry["spectral_window"]
    if kind == "hanning":
        weight = .5 * (1 - torch.cos(2 * torch.pi * fraction)) * valid
    elif kind == "ramp":
        weight = torch.where(fraction < .1, .5 * (1 - torch.cos(torch.pi * fraction / .1)), 1.) * valid
    else:
        weight = valid.double()
    return weight * activity


def accumulate_copper_spectra(solver, *, activity=1):
    for entry in solver._copper_foils:
        if "spectral_real" not in entry:
            continue
        if "fused_electric" in entry:
            from .cuda.backend import accumulate_copper_spectra as fused_spectra
            fused_spectra(solver, entry, activity=activity)
            continue
        step = entry["spectral_step"]
        weight = copper_spectral_weights(entry, step, activity=activity)
        angle = 2 * torch.pi * entry["spectral_frequencies"] * solver.dt * (step + 1)
        electric = torch.stack((getattr(solver, entry["electric_name"])[entry["index"]], entry["right_e"])).double()
        entry["spectral_real"].add_((weight * torch.cos(angle))[:, None, None] * electric)
        entry["spectral_imag"].add_((weight * torch.sin(angle))[:, None, None] * electric)
        entry["spectral_norm"].add_(weight)
        entry["spectral_step"].add_(activity)


def validate_copper_port_coupling(solver):
    """Do not bind a one-trace Norton element to a split-trace foil edge."""
    finalize_copper_edges(solver)
    for entry in solver._copper_foils:
        component = entry["electric_name"]
        if entry["imaginary"]:
            continue
        offsets = _flat_offsets(getattr(solver, component).shape, entry["indices"])
        for term in (*getattr(solver, "_source_terms", ()), *getattr(solver, "_electric_source_terms", ())):
            if term["field_name"].removesuffix("_imag") != component:
                continue
            for key in ("patch", "cw_cos_patch", "cw_sin_patch"):
                patch = term.get(key)
                if patch is None:
                    continue
                local = entry["indices"] - torch.tensor(term["offsets"], device=solver.device)
                inside = ((local >= 0) & (local < torch.tensor(patch.shape, device=solver.device))).all(-1)
                if bool(inside.any()) and bool(torch.any(patch[tuple(local[inside].T)] != 0)):
                    raise NotImplementedError("An electric source patch overlaps a CopperFoil split-trace edge; joint source/trace elimination is required.")
        for port in getattr(solver, "_port_runtimes", ()):
            geometry = port.geometry
            if getattr(geometry, "voltage_component", None) != component:
                continue
            port_offsets = _flat_offsets(getattr(solver, component).shape, geometry.voltage_indices)
            if bool(torch.isin(port_offsets, offsets).any()):
                raise NotImplementedError(f"Port {port.port.name!r} overlaps CopperFoil {entry['face'].name!r}; split-trace port elimination is required.")
        for runtime, field_name in getattr(solver, "_lumped_element_runtimes", ()):
            if field_name == component and bool(torch.isin(runtime.linear_indices, offsets).any()):
                raise NotImplementedError("A lumped element overlaps a CopperFoil split-trace edge.")
