"""Passive midpoint bulk-material states on the two foil electric traces."""
from __future__ import annotations

import torch


def initialize_copper_trace_ade(solver):
    """Eliminate local material states into a positive admittance and history.

    Each foil side has an independent polarization history and the same material
    samples as its electric capacity. Material midpoint current is
    ``J_mid = history + gain * E_mid``. The positive ``gain`` is folded into the
    two-trace Ampere solve; states advance only after its electric midpoint is
    known. Native single-trace ADE drives are cleared after all side bindings
    have been gathered, so the material tail cannot deposit them a second time.
    """
    templates = getattr(solver, "_dispersive_templates", {})
    for entry in getattr(solver, "_copper_foils", ()):
        name = entry["electric_name"].removesuffix("_imag")
        indices = entry["indices"]
        axis = entry["face"].axis
        left, right = indices.clone(), indices.clone()
        left[:, axis] -= 1
        right[:, axis] += 1
        li = tuple(left[:, coordinate] for coordinate in range(3))
        ri = tuple(right[:, coordinate] for coordinate in range(3))
        local = []
        dt = solver.dt
        for kind in ("debye", "drude", "lorentz"):
            for pole_index, template in enumerate(templates.get(name, {}).get(kind, ())):
                drive = torch.stack((template["drive"][li], template["drive"][ri])).double()
                if not bool(torch.any(drive != 0)):
                    continue
                decay = template["decay"]
                descriptor = dict(kind=kind, pole_index=pole_index, decay=decay, drive=drive,
                    current=torch.zeros_like(drive), history=torch.zeros_like(drive),
                    midpoint_current=torch.zeros_like(drive), scratch=torch.zeros_like(drive))
                active = drive != 0
                if kind == "debye":
                    descriptor["polarization"] = torch.zeros_like(drive)
                    descriptor["gain"] = drive / dt
                    strength = drive / (1 - decay)
                    tau = .5 * dt * (1 + decay) / (1 - decay)
                    inverse = torch.where(active, strength.clamp_min(torch.finfo(drive.dtype).tiny).reciprocal(), 0.)
                    descriptor.update(inverse_strength=inverse, loss_coefficient=tau * inverse)
                else:
                    restoring = template.get("restoring", 0.) if kind == "lorentz" else 0.
                    denominator = 2 + .5 * dt * restoring
                    descriptor.update(restoring=restoring, denominator=denominator, gain=drive / denominator)
                    strength = 2 * drive / (dt * (1 + decay))
                    inverse = torch.where(active, strength.clamp_min(torch.finfo(drive.dtype).tiny).reciprocal(), 0.)
                    damping = 2 * (1 - decay) / (dt * (1 + decay))
                    omega_sq = 2 * restoring / (dt * (1 + decay))
                    descriptor.update(inverse_strength=inverse, loss_coefficient=damping * inverse, omega_sq=omega_sq)
                    if kind == "lorentz":
                        descriptor["polarization"] = torch.zeros_like(drive)
                local.append(descriptor)
        entry["material_ade"] = tuple(local)
        entry["material_current"] = torch.zeros_like(entry["old_e"])
        entry["material_widths"] = torch.tensor((entry["left_width"], entry["right_width"]),
                                                device=solver.device, dtype=torch.float64)[:, None]
        entry["material_gain"] = sum((pole["gain"] for pole in local), torch.zeros_like(entry["old_e"]))
        entry["material_work"] = torch.zeros((), device=solver.device, dtype=torch.float64)
        entry["material_dissipated_energy"] = torch.zeros((), device=solver.device, dtype=torch.float64)
        if local:
            sigma = getattr(solver, "sigma_e_" + name)
            conductance = torch.stack((sigma[li].double(), sigma[ri].double())) * entry["material_widths"]
            conductance = conductance + entry["material_widths"] * entry["material_gain"]
            effective = (entry["D"] + entry["B"].T @ entry["drive"])[None] * entry["admittance_scale"][:, None, None]
            lhs = effective / 2 + torch.diag_embed((entry["capacity"] / dt + conductance / 2).T)
            entry["inverse"] = torch.linalg.inv(lhs)
            entry["trace_conductance"] = conductance
            entry["rhs_matrix"] = -effective / 2 + torch.diag_embed((entry["capacity"] / dt - conductance / 2).T)
    for entry in getattr(solver, "_copper_foils", ()):
        name = entry["electric_name"].removesuffix("_imag")
        for kind in ("debye", "drude", "lorentz"):
            for template in templates.get(name, {}).get(kind, ()):
                template["drive"][entry["index"]] = 0


def prepare_copper_trace_history(entry, dt):
    """Evaluate the known side of each eliminated midpoint material equation."""
    total = entry["material_current"]
    total.zero_()
    for pole in entry["material_ade"]:
        history = pole["history"]
        if pole["kind"] == "debye":
            torch.mul(pole["polarization"], (pole["decay"] - 1) / dt, out=history)
        else:
            torch.mul(pole["current"], 1 + pole["decay"], out=history)
            if pole["kind"] == "lorentz":
                history.add_(pole["polarization"], alpha=-pole["restoring"])
            history.div_(pole["denominator"])
        total.add_(history)
    return total


def finalize_copper_trace_ade(entry, updated, dt, *, activity=1):
    """Advance material endpoints and exact passive heat from solved E_mid."""
    midpoint_e = (updated + entry["old_e"]) / 2
    gated = isinstance(activity, torch.Tensor) or activity != 1
    active = torch.as_tensor(activity, device=updated.device, dtype=torch.bool) if gated else None
    total = entry["material_current"]
    total.zero_()
    for pole in entry["material_ade"]:
        midpoint = pole["midpoint_current"]
        torch.addcmul(pole["history"], pole["gain"], midpoint_e, out=midpoint)
        if pole["kind"] == "debye":
            pole["current"].copy_(torch.where(active, midpoint, pole["current"]) if gated else midpoint)
        else:
            scratch = pole["scratch"]
            torch.mul(midpoint, 2., out=scratch)
            scratch.sub_(pole["current"])
            pole["current"].copy_(torch.where(active, scratch, pole["current"]) if gated else scratch)
        if "polarization" in pole:
            scratch = pole["scratch"]
            torch.add(pole["polarization"], midpoint, alpha=dt, out=scratch)
            pole["polarization"].copy_(torch.where(active, scratch, pole["polarization"]) if gated else scratch)
        total.add_(midpoint)
        heat = pole["loss_coefficient"] * midpoint.square() * entry["material_widths"] * entry["area"]
        entry["material_dissipated_energy"].add_(dt * torch.sum(heat) * activity)
    work = midpoint_e * total * entry["material_widths"] * entry["area"]
    entry["material_work"].add_(dt * torch.sum(work) * activity)


def copper_material_energy(entry):
    value = entry["material_work"].new_zeros(())
    for pole in entry["material_ade"]:
        if pole["kind"] == "debye":
            density = pole["polarization"].square() * pole["inverse_strength"] / 2
        else:
            density = pole["current"].square() * pole["inverse_strength"] / 2
            if pole["kind"] == "lorentz":
                density = density + pole["omega_sq"] * pole["polarization"].square() * pole["inverse_strength"] / 2
        value = value + torch.sum(density * entry["material_widths"] * entry["area"])
    return value


def iter_copper_trace_ade_state(index, entry):
    if not entry.get("material_ade"):
        return
    for pole_index, pole in enumerate(entry["material_ade"]):
        for name in ("polarization", "current"):
            if name in pole:
                yield f"copper_foil_{index}_material_{pole_index}_{name}", pole[name]
    for name in ("material_work", "material_dissipated_energy"):
        yield f"copper_foil_{index}_{name}", entry[name]
