"""Functional two-trace foil recurrence shared by checkpoint replay and GPU VJP."""
import torch


# Differentiable per-entry foil coefficients. ``admittance_scale`` is the
# per-trace 1/kappa free-edge factor; it is exactly one away from foil edges.
COPPER_COEFFICIENTS = ("B", "rates", "D", "admittance_scale")


def _old_e(entry, state, index):
    return torch.stack((state[entry["electric_name"]][entry["index"]].double(),
                        state[f"copper_foil_{index}_right_e"].double()))


def replay_copper_magnetic(solver, magnetic_fields, state):
    output = dict(magnetic_fields)
    for index, entry in enumerate(getattr(solver, "_copper_foils", ())):
        old = _old_e(entry, state, index)
        difference = old[1] - old[0]
        name = entry["magnetic_name"]
        field = output[name].clone()
        field[entry["high_h"]] = field[entry["high_h"]] + (entry["high_h_gain"] * difference).to(field.dtype)
        output[name] = field
        name = entry["normal_magnetic_name"]
        field = output[name].clone()
        weighted = difference * entry["right_width"] / (entry["left_width"] + entry["right_width"])
        field.reshape(-1).index_add_(0, entry["low_normal_offsets"],
            (entry["low_normal_gain"] * weighted.index_select(0, entry["low_sources"])).to(field.dtype))
        field.reshape(-1).index_add_(0, entry["high_normal_offsets"],
            (entry["high_normal_gain"] * weighted.index_select(0, entry["high_sources"])).to(field.dtype))
        output[name] = field
    return output


def replay_copper_electric(solver, electric_fields, magnetic_fields, state):
    output = dict(electric_fields)
    auxiliary = {}
    for index, entry in enumerate(getattr(solver, "_copper_foils", ())):
        old = _old_e(entry, state, index)
        magnetic = magnetic_fields[entry["magnetic_name"]]
        normal = magnetic_fields[entry["normal_magnetic_name"]]
        low = normal[entry["low_normal_h"]].double()
        high = normal[entry["high_normal_h"]].double()
        if entry["paired"]:
            if entry["phase"]:
                import math
                companion_name = entry["normal_magnetic_name"]
                companion_name = companion_name.removesuffix("_imag") if entry["imaginary"] else companion_name + "_imag"
                companion = magnetic_fields[companion_name]
                sine = math.sin(entry["phase"]) * (-1 if entry["imaginary"] else 1)
                cosine = math.cos(entry["phase"])
                low = torch.where(entry["low_valid"], low, cosine * low + sine * companion[entry["low_normal_h"]].double())
                high = torch.where(entry["high_valid"], high, cosine * high - sine * companion[entry["high_normal_h"]].double())
        else:
            low = low * entry["low_valid"]
            high = high * entry["high_valid"]
        transverse = entry["sign"] * (high - low) * entry["inv_tangent_dual"]
        forcing = torch.stack((entry["sign"] * magnetic[entry["low_h"]].double() + entry["left_width"] * transverse,
                              -entry["sign"] * magnetic[entry["high_h"]].double() + entry["right_width"] * transverse))
        prefix = f"copper_foil_{index}_"
        material_history = []
        material_current = torch.zeros_like(old)
        for pole_index, descriptor in enumerate(entry.get("material_ade", ())):
            material_prefix = prefix + f"material_{pole_index}_"
            if descriptor["kind"] == "debye":
                history_current = (descriptor["decay"] - 1) / solver.dt * state[material_prefix + "polarization"]
            else:
                history_current = (1 + descriptor["decay"]) * state[material_prefix + "current"]
                if descriptor["kind"] == "lorentz":
                    history_current = history_current - descriptor["restoring"] * state[material_prefix + "polarization"]
                history_current = history_current / descriptor["denominator"]
            material_history.append(history_current)
            material_current = material_current + history_current
        if entry.get("material_ade"):
            forcing = forcing - entry["material_widths"] * material_current
        previous_state = state[prefix + "state"]
        history = ((entry["B"].T * ((1 + entry["decay"]) / 2)[None, :]) @ previous_state) * entry["admittance_scale"]
        rhs = torch.einsum("mij,jm->im", entry["rhs_matrix"], old) + forcing - history
        updated = torch.einsum("mij,jm->im", entry["inverse"], rhs)
        name = entry["electric_name"]
        field = output[name].clone()
        field[entry["index"]] = updated[0].to(field.dtype)
        output[name] = field
        next_state = entry["decay"][:, None] * previous_state + entry["drive"] @ (updated + old)
        midpoint_state = (next_state + previous_state) / 2
        midpoint_e = (updated + old) / 2
        if entry.get("material_ade"):
            material_current = torch.zeros_like(old)
            material_heat = old.new_zeros(())
            for pole_index, (descriptor, history_current) in enumerate(zip(entry["material_ade"], material_history)):
                material_prefix = prefix + f"material_{pole_index}_"
                current = history_current + descriptor["gain"] * midpoint_e
                material_current = material_current + current
                auxiliary[material_prefix + "current"] = current if descriptor["kind"] == "debye" else 2 * current - state[material_prefix + "current"]
                if descriptor["kind"] != "drude":
                    auxiliary[material_prefix + "polarization"] = state[material_prefix + "polarization"] + solver.dt * current
                material_heat = material_heat + torch.sum(descriptor["loss_coefficient"] * current.square() * entry["material_widths"] * entry["area"])
            auxiliary[prefix + "material_work"] = state[prefix + "material_work"] + solver.dt * torch.sum(
                midpoint_e * material_current * entry["material_widths"] * entry["area"])
            auxiliary[prefix + "material_dissipated_energy"] = state[prefix + "material_dissipated_energy"] + solver.dt * material_heat
        loss = (entry["rates"][:, None] * midpoint_state.square()).sum(0)
        loss = loss + torch.sum(midpoint_e * (entry["D"] @ midpoint_e), 0)
        loss = loss * entry["admittance_scale"]
        auxiliary[prefix + "state"] = next_state
        auxiliary[prefix + "right_e"] = updated[1].to(field.dtype)
        auxiliary[prefix + "cumulative_loss"] = state[prefix + "cumulative_loss"] + solver.dt * torch.sum(loss * entry["area"])
    return output, auxiliary


def differentiable_copper_entries(solver, epsilons, sigmas):
    """Rebuild the exact midpoint matrix from differentiable GPU coefficients."""
    entries, leaves = [], []
    for frozen in getattr(solver, "_copper_foils", ()):
        entry = dict(frozen)
        B, rates, D, scale = (frozen[key].detach().requires_grad_(True) for key in COPPER_COEFFICIENTS)
        leaves.append((B, rates, D, scale))
        indices = entry["indices"]
        left, right = indices.clone(), indices.clone()
        left[:, entry["face"].axis] -= 1
        right[:, entry["face"].axis] += 1
        li = tuple(left[:, axis] for axis in range(3))
        ri = tuple(right[:, axis] for axis in range(3))
        name = entry["electric_name"].removesuffix("_imag")
        capacity = torch.stack((epsilons[name][li].double() * entry["left_width"],
                                epsilons[name][ri].double() * entry["right_width"]))
        conductance = torch.stack((sigmas[name][li].double() * entry["left_width"],
                                   sigmas[name][ri].double() * entry["right_width"]))
        if "material_metric_factors" in entry:
            capacity = capacity * entry["material_metric_factors"]
            conductance = conductance * entry["material_metric_factors"]
        if entry.get("material_ade"):
            conductance = conductance + entry["material_gain"] * entry["material_widths"]
        denominator = 1 + solver.dt * rates / 2
        decay = (1 - solver.dt * rates / 2) / denominator
        drive = solver.dt / 2 * B / denominator[:, None]
        effective = (D + B.T @ drive)[None] * scale[:, None, None]
        diagonal = torch.diag_embed((capacity / solver.dt + conductance / 2).T)
        lhs = effective / 2 + diagonal
        rhs_matrix = -effective / 2 + torch.diag_embed((capacity / solver.dt - conductance / 2).T)
        entry.update(B=B, rates=rates, D=D, admittance_scale=scale, decay=decay, drive=drive,
                     inverse=torch.linalg.inv(lhs), rhs_matrix=rhs_matrix)
        entries.append(entry)
    return entries, leaves
