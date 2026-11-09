"""Discrete bulk electric work and split-trace loss on native Yee volumes."""
from __future__ import annotations

from dataclasses import replace
import math

import torch

from ..constants import EPSILON_0
from ..fdtd.runtime.materials import pole_component_weight


def native_pole_work_conductivity(kind, decay, drive, dt, frequencies, restoring=0.):
    """Positive real same-step E-midpoint/J work transfer of native bulk ADE."""
    angle = 2 * torch.pi * frequencies.reshape(-1, *([1] * drive.ndim)) * dt
    sin_half_sq = torch.sin(angle / 2).square()
    if kind == "debye":
        denominator = (1-decay)**2 + 4*decay*sin_half_sq
        gain = torch.sin(angle).square() / (dt * denominator.clamp_min(torch.finfo(torch.float64).tiny))
    elif kind == "drude":
        denominator = (1-decay)**2 + 4*decay*sin_half_sq
        gain = (1-decay)*torch.cos(angle/2).square() / denominator.clamp_min(torch.finfo(torch.float64).tiny)
    else:
        real = restoring*dt - 2*(1+decay)*sin_half_sq
        imaginary = (1-decay)*torch.sin(angle)
        gain = (1-decay)*torch.sin(angle).square() / (real.square()+imaginary.square()).clamp_min(torch.finfo(torch.float64).tiny)
    return drive[None] * gain


def automatic_electric_dispersion_density(scene, compiled, frequencies, fields, dt):
    """Evaluate actual native bulk pole work on the monitor's sparse E edges."""
    model = scene.compile_materials()
    output = {}
    for component in ("Ex", "Ey", "Ez"):
        mask = compiled.component_masks[component]
        coefficient = torch.zeros((len(frequencies), int(mask.sum())), device=mask.device, dtype=torch.float64)
        for kind in ("debye", "drude", "lorentz"):
            for entry in model[kind+"_poles"]:
                if entry.get("axes") is not None and component[-1].lower() not in entry["axes"]:
                    continue
                pole = entry["pole"]
                weight = pole_component_weight(entry, component)[mask].double()
                restoring = 0.
                if kind == "debye":
                    decay = (2*pole.tau-dt)/(2*pole.tau+dt)
                    drive = weight*(2*EPSILON_0*pole.delta_eps*dt/(2*pole.tau+dt))
                else:
                    gamma = 2*math.pi*pole.gamma
                    denominator = 2+gamma*dt
                    decay = (2-gamma*dt)/denominator
                    if kind == "drude":
                        drive = weight*(2*EPSILON_0*(2*math.pi*pole.plasma_frequency)**2*dt/denominator)
                    else:
                        omega_sq = (2*math.pi*pole.resonance_frequency)**2
                        restoring = 2*omega_sq*dt/denominator
                        drive = weight*(2*EPSILON_0*pole.delta_eps*omega_sq*dt/denominator)
                coefficient = coefficient + native_pole_work_conductivity(kind, decay, drive, dt, frequencies, restoring)
        output[component] = .5 * coefficient * fields[component][:, mask].abs().square()
    return output


def midpoint_trace_work_conductivity(entry, dt, frequencies):
    """Exact local midpoint material work per squared endpoint E phasor."""
    shape = (len(frequencies), *entry["old_e"].shape)
    result = entry["old_e"].new_zeros(shape)
    angle = torch.pi * frequencies[:, None, None] * dt
    omega = 2*torch.tan(angle)/dt
    cosine_sq = torch.cos(angle).square()
    for pole in entry.get("material_ade", ()):
        inverse = pole["inverse_strength"]
        strength = torch.where(inverse > 0, inverse.clamp_min(torch.finfo(inverse.dtype).tiny).reciprocal(), 0.)[None]
        if pole["kind"] == "debye":
            tau = .5*dt*(1+pole["decay"])/(1-pole["decay"])
            gain = strength*tau*omega.square()/(1+(tau*omega).square())
        else:
            gamma = 2*(1-pole["decay"])/(dt*(1+pole["decay"]))
            if pole["kind"] == "drude":
                gain = strength*gamma/(gamma**2+omega.square())
            else:
                denominator = (pole["omega_sq"]-omega.square()).square()+gamma**2*omega.square()
                gain = strength*gamma*omega.square()/denominator.clamp_min(torch.finfo(result.dtype).tiny)
        result = result + gain*cosine_sq
    return result


def apply_split_trace_densities(data, compiled, fields, records, *, correct_dispersion):
    """Replace one native volume by its two physical side integrals, once."""
    densities = {channel: dict(values) for channel, values in data.volume_density.items()}
    changed = set()
    for record in records.values():
        frequencies = record.get("frequencies")
        for trace in record.get("bulk_electric_traces", ()):
            component = trace["component"]
            mask = compiled.component_masks[component]
            indices = trace["indices"].to(mask.device)
            selected = mask[tuple(indices.T)]
            if not bool(selected.any()):
                continue
            if fields[component].requires_grad and not trace["electric_fields"].requires_grad:
                raise NotImplementedError("Split-foil spectral bulk loss requires differentiable side spectra; use the qualified conductor energy objective for foil gradients.")
            locations = []
            for frequency in data.frequencies:
                matches = torch.nonzero(frequencies.to(frequency.device) == frequency).flatten()
                if matches.numel() != 1:
                    raise ValueError("Foil side spectra do not contain the requested loss-monitor frequency.")
                locations.append(int(matches[0]))
            fi = torch.tensor(locations, device=mask.device)
            if "spectral_valid" in record and not bool(record["spectral_valid"][locations].all()):
                raise RuntimeError("Requested foil loss frequency has no valid side-spectrum samples.")
            side_fields = trace["electric_fields"].to(mask.device)[fi]
            original_left = side_fields[:, 0, selected]
            native = fields[component][(slice(None), *tuple(indices[selected].T))].to(side_fields.dtype)
            denominator = original_left.abs().square().sum(-1)
            if bool(torch.any((denominator == 0) & (side_fields[:, 1, selected].abs().square().sum(-1) > 0))):
                raise RuntimeError("An identically zero left trace cannot determine the retained right-trace spectral normalization.")
            factor = (original_left.conj()*native).sum(-1)/denominator.clamp_min(torch.finfo(torch.float64).tiny)
            mismatch = (native-original_left*factor[:, None]).abs().square().sum(-1)
            scale = native.abs().square().sum(-1).clamp_min(torch.finfo(torch.float64).tiny)
            if bool(torch.any(mismatch/scale > 1e-7)):
                raise ValueError("Electric fields do not match the retained foil side spectral schedule and normalization.")
            side_fields = side_fields*factor[:, None, None]
            flat = (indices[:, 0]*mask.shape[1]+indices[:, 1])*mask.shape[2]+indices[:, 2]
            selected_flat = torch.nonzero(mask.flatten()).flatten()
            positions = torch.searchsorted(selected_flat, flat[selected])
            volumes = compiled.component_volumes[component][positions]
            overlap = split_trace_overlap_fractions(compiled, trace, indices)
            for channel, key in (("conduction", "conduction_coefficient"), ("electric_dispersion", "dispersion_coefficient")):
                if channel not in densities or channel == "electric_dispersion" and not correct_dispersion:
                    continue
                power = .5 * (trace[key].to(mask.device)[fi] * side_fields.abs().square()*overlap[None]).sum(1)
                density = densities[channel][component].to(power.dtype).clone()
                density[:, positions] = power[:, selected]/volumes[None]
                densities[channel][component] = density
                changed.add(channel)
    powers = dict(data.channel_power)
    for channel in changed:
        powers[channel] = sum((density*data.component_volumes[component][None]).sum(-1)
                              for component, density in densities[channel].items())
    return replace(data, volume_density=densities, channel_power=powers,
                   total=torch.stack(tuple(powers.values())).sum(0))


def split_trace_overlap_fractions(compiled, trace, indices):
    """Clip each actual bulk half-volume independently at a foil plane."""
    if compiled.component_cell_bounds is None:
        return torch.ones((2, len(indices)), device=indices.device, dtype=torch.float64)
    if not all(key in trace for key in ("normal_axis", "normal_coordinate", "normal_half_widths")):
        raise ValueError("Physical-intersection bulk loss requires retained foil half-volume geometry; this result only supports native_cells.")
    normal = "xyz".index(trace["normal_axis"])
    bounds = compiled.component_cell_bounds[trace["component"]]
    window = compiled.monitor.bounds
    area_fraction = torch.ones(len(indices), device=indices.device, dtype=torch.float64)
    for axis in range(3):
        if axis == normal:
            continue
        lower, upper = [bound[indices[:, axis]] for bound in bounds[axis]]
        overlap = torch.clamp(torch.minimum(upper, upper.new_tensor(window[axis][1]))
            - torch.maximum(lower, lower.new_tensor(window[axis][0])), min=0.)
        area_fraction *= overlap/(upper-lower)
    widths = trace["normal_half_widths"].to(device=indices.device, dtype=torch.float64)
    coordinate = float(trace["normal_coordinate"])
    lower = torch.stack((coordinate-widths[0], torch.full_like(widths[1], coordinate)))
    upper = torch.stack((torch.full_like(widths[0], coordinate), coordinate+widths[1]))
    overlap = torch.clamp(torch.minimum(upper, upper.new_tensor(window[normal][1]))
        - torch.maximum(lower, lower.new_tensor(window[normal][0])), min=0.)
    return area_fraction[None]*overlap/widths


def conductor_region_power(record, bounds):
    """Detached native film work inside a box, with half-open plane ownership.

    The collapsed foil plane belongs to [lower, upper). Its physical thickness
    is not spatially resolved, so no partial through-thickness heat is inferred.
    Tangential native tiles use their exact physical intersection areas.
    """
    if len(bounds) != 3 or any(not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or hi <= lo for lo, hi in bounds):
        raise ValueError("Conductor ROI bounds must contain three finite increasing intervals.")
    traces = record.get("surface_power_traces")
    if traces is None:
        raise ValueError("Regional conductor power requires retained per-edge work and physical tile geometry.")
    if record.get("coordinate_frame") != "cartesian":
        raise NotImplementedError("Regional conductor power currently requires Cartesian surface tiles; mapped partial-tile metrics are not implemented.")
    if record.get("periodic_axes"):
        raise NotImplementedError("Regional conductor power requires nonperiodic axes; wrapped surface-tile fragments are not implemented.")
    total = torch.zeros_like(record["surface_power"])
    for trace in traces:
        normal = "xyz".index(trace["normal_axis"])
        coordinate = trace["normal_coordinate"]
        if not bounds[normal][0] <= coordinate < bounds[normal][1]:
            continue
        lower, upper = trace["lower_bounds"], trace["upper_bounds"]
        fraction = torch.ones_like(trace["area"])
        for axis in range(3):
            if axis == normal:
                continue
            width = upper[:, axis]-lower[:, axis]
            overlap = torch.clamp(torch.minimum(upper[:, axis], upper.new_tensor(bounds[axis][1]))
                - torch.maximum(lower[:, axis], lower.new_tensor(bounds[axis][0])), min=0.)
            fraction *= overlap/width
        total = total+(trace["power"]*fraction).sum(-1)
    return total
