"""Deterministic dynamic dielectric-breakdown runtime for the FDTD step loop.

Design (all GPU-resident, no per-step host synchronization):

* A breakdown-capable *node* set is compiled once from the scene (the material
  grid ``(Nx, Ny, Nz)``). Each capable node carries the field-duration/latching
  state machine: ``state`` (intact/conducting), a contiguous-exceedance
  ``timer``, and a latching ``trigger_step``.
* ``|E|`` at a node uses energy-consistent cell-center colocation: for each Yee
  component the squared values of the two edges meeting the node are averaged,
  and the three component averages are summed (the same squared-field averaging
  the FDTD electric energy density uses).
* On trigger the node conductivity ramps linearly from its base value toward
  ``post_breakdown_conductivity`` over ``ramp_time``. The effective per-edge
  conductivity is scattered onto a precompiled capable-*edge* index set and the
  semi-implicit lossy decay/curl coefficients are rewritten in place there,
  reusing the compiler's node->edge arithmetic averaging. Intact capable edges
  are rewritten with their exact stored base coefficients, so a run in which no
  cell ever triggers stays bitwise identical to the same scene without the
  descriptor.
* Breakdown dissipation ``integral(sigma_breakdown * |E|^2 dV dt)`` is
  accumulated on the same Yee edges that carry the conduction term and scattered
  back to nodes, keeping a dedicated energy channel separate from ordinary
  static-conductivity loss.
* Events are stored in a bounded preallocated per-node buffer whose capacity is
  the breakdown-capable node count -- the exact upper bound for the latching
  model (one event per cell), so overflow cannot occur. The overflow guard is
  retained defensively and host transfer happens only at run end.

Scenes with no breakdown material set ``solver.breakdown_enabled = False`` and
take the existing code path with zero added machinery.
"""

from __future__ import annotations

import warnings

import torch

from ...breakdown import (
    BREAKDOWN_STATE_CONDUCTING,
    BREAKDOWN_STATE_INTACT,
    BreakdownEvent,
    BreakdownResultData,
)
from ...compiler.breakdown import compile_breakdown_layout, scene_has_breakdown
from ..boundary.common import has_complex_fields


# Above this semi-implicit loss ratio a = 0.5*sigma*dt/eps the decay factor
# (1-a)/(1+a) approaches -1: the cell is effectively a poorly-resolved PEC-like
# reflector. The update stays unconditionally stable, but the ramp/target is too
# aggressive for the time step, so we warn (never silently hide it).
_LOSS_RATIO_WARN_THRESHOLD = 10.0


def _flat_index(coords, shape):
    """Row-major flat index of integer coordinate columns for a 3D shape."""
    i, j, k = coords
    _, ny, nz = shape
    return (i * ny + j) * nz + k


def _capable_edge_sets(solver, layout):
    """Build the capable-edge index tensors and their static reconstruction data.

    For each electric component we collect the Yee edges that touch at least one
    breakdown-capable node, together with: the flat edge index into the field
    tensor, the compact capable-node indices of the two nodes the compiler
    averages onto that edge (``-1`` -> a padded zero), the edge permittivity, the
    static base edge conductivity, the exact stored base decay/curl coefficients,
    the recovered external factor (PML decay times PEC open fraction), and the
    edge control volume.
    """
    device = solver.device
    node_mask = layout.node_mask
    nx, ny, nz = layout.grid_shape

    # Compact node index for every capable node (row-major scan order == global
    # cell-id order), -1 elsewhere.
    node_compact = torch.full(layout.grid_shape, -1, dtype=torch.int64, device=device)
    node_compact[node_mask] = torch.arange(
        int(node_mask.sum().item()), dtype=torch.int64, device=device
    )

    dt = float(solver.dt)
    edge_sets = {}
    component_specs = (
        ("Ex", solver.eps_Ex, solver.sigma_e_Ex, solver.cex_decay, solver.cex_curl),
        ("Ey", solver.eps_Ey, solver.sigma_e_Ey, solver.cey_decay, solver.cey_curl),
        ("Ez", solver.eps_Ez, solver.sigma_e_Ez, solver.cez_decay, solver.cez_curl),
    )
    # Edge control volumes: primal along the edge axis, dual on the transverse axes
    # (matches compiler.power_loss._component_geometry).
    scene = solver.scene
    dx_primal = torch.as_tensor(scene.dx_primal64, device=device, dtype=torch.float64)
    dy_primal = torch.as_tensor(scene.dy_primal64, device=device, dtype=torch.float64)
    dz_primal = torch.as_tensor(scene.dz_primal64, device=device, dtype=torch.float64)
    dx_dual = torch.as_tensor(scene.dx_dual64, device=device, dtype=torch.float64)
    dy_dual = torch.as_tensor(scene.dy_dual64, device=device, dtype=torch.float64)
    dz_dual = torch.as_tensor(scene.dz_dual64, device=device, dtype=torch.float64)

    for name, eps, sigma_base, decay, curl in component_specs:
        shape = eps.shape
        if name == "Ex":
            cap = node_mask[:-1, :, :] | node_mask[1:, :, :]
        elif name == "Ey":
            cap = node_mask[:, :-1, :] | node_mask[:, 1:, :]
        else:
            cap = node_mask[:, :, :-1] | node_mask[:, :, 1:]
        idx = cap.nonzero(as_tuple=False)
        if idx.numel() == 0:
            edge_sets[name] = None
            continue
        i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
        if name == "Ex":
            left = node_compact[i, j, k]
            right = node_compact[i + 1, j, k]
            vol = (dx_primal[i] * dy_dual[j] * dz_dual[k]).to(torch.float32)
        elif name == "Ey":
            left = node_compact[i, j, k]
            right = node_compact[i, j + 1, k]
            vol = (dx_dual[i] * dy_primal[j] * dz_dual[k]).to(torch.float32)
        else:
            left = node_compact[i, j, k]
            right = node_compact[i, j, k + 1]
            vol = (dx_dual[i] * dy_dual[j] * dz_primal[k]).to(torch.float32)

        edge_flat = _flat_index((i, j, k), shape)
        eps_edge = eps.reshape(-1)[edge_flat].clone()
        sigma_edge = sigma_base.reshape(-1)[edge_flat].clone()
        base_decay = decay.reshape(-1)[edge_flat].clone()
        base_curl = curl.reshape(-1)[edge_flat].clone()
        # Recover the external multiplicative factor (PML decay * PEC open fraction)
        # from the stored base curl: curl_base = (dt/eps/(1+a0)) * ext, with
        # a0 = 0.5*sigma_base*dt/eps. ext == 1 when neither PML nor PEC touches the edge.
        a0 = 0.5 * sigma_edge * dt / eps_edge
        ext = base_curl * (1.0 + a0) * eps_edge / dt

        node_total = int(node_mask.sum().item())
        # Static node-gather indices: -1 (non-capable) -> the padded trailing zero.
        left_pad = torch.where(left < 0, torch.full_like(left, node_total), left)
        right_pad = torch.where(right < 0, torch.full_like(right, node_total), right)
        # Static per-node dissipation scatter selections (energy split among capable
        # endpoints so the per-node channel sums to the edge total).
        n_cap = (left >= 0).to(torch.float32) + (right >= 0).to(torch.float32)
        n_cap = torch.clamp(n_cap, min=1.0)
        left_sel = (left >= 0).nonzero(as_tuple=False).squeeze(1)
        right_sel = (right >= 0).nonzero(as_tuple=False).squeeze(1)
        edge_sets[name] = {
            "edge_flat": edge_flat.contiguous(),
            "left_pad": left_pad.contiguous(),
            "right_pad": right_pad.contiguous(),
            "eps": eps_edge.contiguous(),
            "sigma_base": sigma_edge.contiguous(),
            "base_decay": base_decay.contiguous(),
            "base_curl": base_curl.contiguous(),
            "ext": ext.contiguous(),
            "volume": vol.contiguous(),
            "n_cap": n_cap.contiguous(),
            "left_sel": left_sel.contiguous(),
            "left_nodes": left[left_sel].contiguous(),
            "right_sel": right_sel.contiguous(),
            "right_nodes": right[right_sel].contiguous(),
        }
    return edge_sets


def _node_colocation_indices(solver, layout):
    """Precompute the two edge flat indices per component that colocate |E|^2 onto
    each capable node, with boundary nodes clamped so the two indices coincide."""
    device = solver.device
    node_mask = layout.node_mask
    nx, ny, nz = layout.grid_shape
    idx = node_mask.nonzero(as_tuple=False)
    i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]

    ex_shape = solver.eps_Ex.shape  # (Nx-1, Ny, Nz)
    ey_shape = solver.eps_Ey.shape  # (Nx, Ny-1, Nz)
    ez_shape = solver.eps_Ez.shape  # (Nx, Ny, Nz-1)

    ex_lo = torch.clamp(i - 1, 0, ex_shape[0] - 1)
    ex_hi = torch.clamp(i, 0, ex_shape[0] - 1)
    ey_lo = torch.clamp(j - 1, 0, ey_shape[1] - 1)
    ey_hi = torch.clamp(j, 0, ey_shape[1] - 1)
    ez_lo = torch.clamp(k - 1, 0, ez_shape[2] - 1)
    ez_hi = torch.clamp(k, 0, ez_shape[2] - 1)

    return {
        "ex_lo": _flat_index((ex_lo, j, k), ex_shape).contiguous(),
        "ex_hi": _flat_index((ex_hi, j, k), ex_shape).contiguous(),
        "ey_lo": _flat_index((i, ey_lo, k), ey_shape).contiguous(),
        "ey_hi": _flat_index((i, ey_hi, k), ey_shape).contiguous(),
        "ez_lo": _flat_index((i, j, ez_lo), ez_shape).contiguous(),
        "ez_hi": _flat_index((i, j, ez_hi), ez_shape).contiguous(),
    }


def initialize_breakdown_runtime(solver):
    """Compile the breakdown state machine, or mark the scene breakdown-free.

    Must run after ``build_update_coefficients`` so the base decay/curl tensors
    exist to seed the exact reconstruction. Rejects the feature combinations
    whose coefficient representation the breakdown scatter does not model.
    """
    solver.breakdown_enabled = False
    solver.breakdown_runtime = None
    # Zero-impact-when-unused: a cheap structure pre-scan gates ALL breakdown
    # allocation. A scene whose materials carry no breakdown descriptor never
    # calls compile_breakdown_layout (which would otherwise allocate seven
    # full-grid (Nx,Ny,Nz) tensors on every FDTD prepare), so the breakdown
    # machinery costs nothing on non-breakdown scenes.
    if not scene_has_breakdown(solver.scene):
        return
    layout = compile_breakdown_layout(solver.scene)
    if not layout.has_breakdown or int(layout.node_mask.sum().item()) == 0:
        return

    if has_complex_fields(solver):
        raise NotImplementedError(
            "Dielectric breakdown requires the real-valued FDTD update; a Bloch-periodic "
            "run carries complex phase-shifted fields, for which the real conductivity "
            "scatter is undefined. Use a real-field boundary."
        )
    for flag, feature in (
        ("dispersive_enabled", "dispersive media"),
        ("nonlinear_enabled", "instantaneous nonlinear media"),
        ("full_aniso_enabled", "full anisotropic media"),
        ("modulation_enabled", "time-modulated media"),
    ):
        if bool(getattr(solver, flag, False)):
            raise NotImplementedError(
                f"Dielectric breakdown does not yet compose with {feature} in the same "
                "scene: they share the per-edge E-update coefficient representation that "
                "the breakdown conductivity scatter rewrites."
            )
    if getattr(solver, "gyromagnetic_enabled", False):
        raise NotImplementedError(
            "Dielectric breakdown does not compose with gyromagnetic ferrite media in the "
            "same scene: the ferrite forward advances a coupled magnetization ADE whose "
            "energy accounting the breakdown dissipation channel does not model."
        )

    device = solver.device
    dt = float(solver.dt)
    node_mask = layout.node_mask
    node_count = int(node_mask.sum().item())
    idx = node_mask.nonzero(as_tuple=False)
    i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]

    model = solver._compiled_material_model
    eps_r_node = model["eps_r"]
    sigma_node = model["sigma_e"]
    base_sigma = sigma_node[i, j, k].to(torch.float32).contiguous()
    eps_abs = (eps_r_node[i, j, k].to(torch.float32) * float(solver.eps0)).contiguous()

    critical = layout.critical_field[i, j, k].contiguous()
    minimum_duration = layout.minimum_duration[i, j, k].contiguous()
    post_sigma = layout.post_conductivity[i, j, k].contiguous()
    ramp_explicit = layout.ramp_time_explicit[i, j, k]
    ramp_steps = layout.ramp_steps[i, j, k].to(torch.float32)
    ramp_time = torch.where(ramp_explicit > 0.0, ramp_explicit, ramp_steps * dt).contiguous()
    material_id = layout.material_id[i, j, k].contiguous()

    # Global (row-major) linear node ids and cell-center positions.
    nx, ny, nz = layout.grid_shape
    cell_index = ((i * ny + j) * nz + k).contiguous()
    x = torch.as_tensor(solver.scene.x, device=device, dtype=torch.float32)
    y = torch.as_tensor(solver.scene.y, device=device, dtype=torch.float32)
    z = torch.as_tensor(solver.scene.z, device=device, dtype=torch.float32)
    position = torch.stack((x[i], y[j], z[k]), dim=1).contiguous()

    dx_dual = torch.as_tensor(solver.scene.dx_dual64, device=device, dtype=torch.float64)
    dy_dual = torch.as_tensor(solver.scene.dy_dual64, device=device, dtype=torch.float64)
    dz_dual = torch.as_tensor(solver.scene.dz_dual64, device=device, dtype=torch.float64)
    node_volume = (dx_dual[i] * dy_dual[j] * dz_dual[k]).to(torch.float32).contiguous()

    # Prepare-time accuracy check on the target conductivity (unconditionally
    # stable, but a very large sigma*dt/eps makes the cell a poorly-resolved
    # PEC-like reflector).
    loss_ratio = 0.5 * post_sigma * dt / eps_abs
    max_ratio = float(loss_ratio.max().item())
    if max_ratio > _LOSS_RATIO_WARN_THRESHOLD:
        warnings.warn(
            "DielectricBreakdown post-breakdown conductivity gives a semi-implicit loss "
            f"ratio 0.5*sigma*dt/eps up to {max_ratio:.3g} (> {_LOSS_RATIO_WARN_THRESHOLD}); "
            "the conducting cell approaches a poorly time-resolved PEC-like reflector. "
            "Reduce post_breakdown_conductivity, refine the time step, or lengthen ramp_time.",
            RuntimeWarning,
            stacklevel=2,
        )

    coloc = _node_colocation_indices(solver, layout)
    edge_sets = _capable_edge_sets(solver, layout)

    runtime = {
        "layout": layout,
        "node_count": node_count,
        "capacity": node_count,
        "dt": dt,
        # Static per-node parameters.
        "critical": critical,
        "minimum_duration": minimum_duration,
        "post_sigma": post_sigma,
        "base_sigma": base_sigma,
        "eps_abs": eps_abs,
        "ramp_time": ramp_time,
        "material_id": material_id,
        "cell_index": cell_index,
        "position": position,
        "node_volume": node_volume,
        # Dynamic per-node state.
        "state": torch.zeros(node_count, dtype=torch.int8, device=device),
        "timer": torch.zeros(node_count, dtype=torch.float32, device=device),
        "trigger_step": torch.full((node_count,), -1, dtype=torch.int64, device=device),
        "field_before": torch.zeros(node_count, dtype=torch.float32, device=device),
        "deposited_at_trigger": torch.zeros(node_count, dtype=torch.float32, device=device),
        "dissipated_energy": torch.zeros(node_count, dtype=torch.float32, device=device),
        "colocation": coloc,
        "edge_sets": edge_sets,
        "total_dissipated": torch.zeros((), dtype=torch.float64, device=device),
    }
    solver.breakdown_enabled = True
    solver.breakdown_runtime = runtime

    # The compressed CPML electric kernel reads a cached per-coefficient uniform
    # value; the breakdown scatter makes these tensors non-uniform mid-run, so
    # invalidate the cache to force the full per-edge tensor read.
    uniformity = getattr(solver, "_coefficient_uniformity", None)
    if isinstance(uniformity, dict):
        for key in ("cex_decay", "cex_curl", "cey_decay", "cey_curl", "cez_decay", "cez_curl"):
            uniformity[key] = None


def _node_field_magnitude(solver, runtime):
    coloc = runtime["colocation"]
    ex = solver.Ex.reshape(-1)
    ey = solver.Ey.reshape(-1)
    ez = solver.Ez.reshape(-1)
    e2 = (
        0.5 * (ex[coloc["ex_lo"]].square() + ex[coloc["ex_hi"]].square())
        + 0.5 * (ey[coloc["ey_lo"]].square() + ey[coloc["ey_hi"]].square())
        + 0.5 * (ez[coloc["ez_lo"]].square() + ez[coloc["ez_hi"]].square())
    )
    return torch.sqrt(torch.clamp(e2, min=0.0)), e2


def _node_extra_conductivity(runtime, step):
    """Per-node breakdown-induced extra conductivity (linear ramp, >= 0)."""
    state = runtime["state"]
    conducting = state == BREAKDOWN_STATE_CONDUCTING
    elapsed = (float(step) - runtime["trigger_step"].to(torch.float32)) * runtime["dt"]
    frac = torch.clamp(elapsed / runtime["ramp_time"], 0.0, 1.0)
    target = torch.clamp(runtime["post_sigma"] - runtime["base_sigma"], min=0.0)
    extra = torch.where(conducting, frac * target, torch.zeros_like(target))
    return torch.clamp(extra, min=0.0)


def _scatter_edge_coefficients(solver, runtime, node_extra):
    """Rewrite the capable-edge decay/curl coefficients from the current node
    conductivity, and accumulate the breakdown dissipation on those edges."""
    dt = runtime["dt"]
    # Pad with a trailing zero so a -1 (non-capable) neighbor contributes no extra.
    extra_padded = torch.cat(
        (node_extra, torch.zeros(1, dtype=node_extra.dtype, device=node_extra.device))
    )
    edge_sets = runtime["edge_sets"]
    step_dissipation = torch.zeros((), dtype=torch.float64, device=solver.device)
    for name, decay_tensor, curl_tensor, field in (
        ("Ex", solver.cex_decay, solver.cex_curl, solver.Ex),
        ("Ey", solver.cey_decay, solver.cey_curl, solver.Ey),
        ("Ez", solver.cez_decay, solver.cez_curl, solver.Ez),
    ):
        data = edge_sets.get(name)
        if data is None:
            continue
        edge_extra = 0.5 * (extra_padded[data["left_pad"]] + extra_padded[data["right_pad"]])
        eps = data["eps"]
        sigma_edge = data["sigma_base"] + edge_extra
        a = 0.5 * sigma_edge * dt / eps
        decay_new = ((1.0 - a) / (1.0 + a)) * data["ext"]
        curl_new = (dt / eps / (1.0 + a)) * data["ext"]
        active = edge_extra > 0.0
        final_decay = torch.where(active, decay_new, data["base_decay"])
        final_curl = torch.where(active, curl_new, data["base_curl"])
        decay_tensor.reshape(-1).index_copy_(0, data["edge_flat"], final_decay)
        curl_tensor.reshape(-1).index_copy_(0, data["edge_flat"], final_curl)

        # Breakdown dissipation on this edge: sigma_breakdown * E^2 * volume * dt.
        e_edge = field.reshape(-1)[data["edge_flat"]]
        energy_edge = edge_extra * e_edge.square() * data["volume"] * dt
        step_dissipation = step_dissipation + energy_edge.to(torch.float64).sum()

        # Split each edge's energy equally among its capable node endpoints so the
        # per-node channel sums to the edge total (static scatter selections).
        share = energy_edge / data["n_cap"]
        dissip = runtime["dissipated_energy"]
        dissip.index_add_(0, data["left_nodes"], share[data["left_sel"]])
        dissip.index_add_(0, data["right_nodes"], share[data["right_sel"]])
    runtime["total_dissipated"] += step_dissipation


def advance_breakdown_state(solver, step):
    """One post-E-update breakdown step: colocation, state machine, conductivity
    scatter, and energy accumulation. All GPU-resident, no host synchronization."""
    runtime = solver.breakdown_runtime
    if runtime is None:
        return

    magnitude, e2 = _node_field_magnitude(solver, runtime)
    state = runtime["state"]
    intact = state == BREAKDOWN_STATE_INTACT
    exceeding = magnitude >= runtime["critical"]

    # Contiguous-exceedance timer: increment while exceeding, reset otherwise;
    # frozen once conducting.
    timer = runtime["timer"]
    incremented = timer + runtime["dt"]
    new_timer = torch.where(
        intact,
        torch.where(exceeding, incremented, torch.zeros_like(timer)),
        timer,
    )
    runtime["timer"] = new_timer

    trigger = intact & exceeding & (new_timer >= runtime["minimum_duration"])
    runtime["state"] = torch.where(
        trigger, torch.full_like(state, BREAKDOWN_STATE_CONDUCTING), state
    )
    runtime["trigger_step"] = torch.where(
        trigger, torch.full_like(runtime["trigger_step"], int(step)), runtime["trigger_step"]
    )
    runtime["field_before"] = torch.where(
        trigger, magnitude, runtime["field_before"]
    )
    deposited = 0.5 * runtime["eps_abs"] * e2 * runtime["node_volume"]
    runtime["deposited_at_trigger"] = torch.where(
        trigger, deposited, runtime["deposited_at_trigger"]
    )

    node_extra = _node_extra_conductivity(runtime, step)
    _scatter_edge_coefficients(solver, runtime, node_extra)


def finalize_breakdown_data(solver) -> BreakdownResultData | None:
    """Transfer the breakdown state to host and assemble the typed result."""
    runtime = getattr(solver, "breakdown_runtime", None)
    if runtime is None:
        return None

    node_count = runtime["node_count"]
    capacity = int(runtime["capacity"])
    trigger_step = runtime["trigger_step"]
    triggered_mask = trigger_step >= 0
    triggered_total = int(triggered_mask.sum().item())
    if triggered_total > capacity:
        raise RuntimeError(
            f"Breakdown event buffer overflow: {triggered_total} events exceed the "
            f"preallocated capacity {capacity}. Increase the event buffer capacity."
        )

    layout = runtime["layout"]
    nx, ny, nz = layout.grid_shape

    # Per-node final-state mask on the material grid.
    final_state = torch.zeros(layout.grid_shape, dtype=torch.int8, device=solver.device)
    idx = layout.node_mask.nonzero(as_tuple=False)
    final_state[idx[:, 0], idx[:, 1], idx[:, 2]] = runtime["state"]

    # Per-node cumulative breakdown dissipation on the material grid.
    dissipated = torch.zeros(layout.grid_shape, dtype=torch.float32, device=solver.device)
    dissipated[idx[:, 0], idx[:, 1], idx[:, 2]] = runtime["dissipated_energy"]

    # Assemble events sorted by (step, cell_index). The capable-node array is
    # already in row-major (cell-index) order, so ordering by step is enough to
    # get the global (step, cell_index) order.
    if triggered_total > 0:
        sel = triggered_mask.nonzero(as_tuple=False).squeeze(1)
        steps = trigger_step[sel]
        cells = runtime["cell_index"][sel]
        order = torch.argsort(steps * (nx * ny * nz) + cells)
        sel = sel[order]
        steps_h = trigger_step[sel].tolist()
        cells_h = runtime["cell_index"][sel].tolist()
        material_h = runtime["material_id"][sel].tolist()
        field_h = runtime["field_before"][sel].tolist()
        deposited_h = runtime["deposited_at_trigger"][sel].tolist()
        position_h = runtime["position"][sel].tolist()
        dt = runtime["dt"]
        events = tuple(
            BreakdownEvent(
                step=int(steps_h[n]),
                time=float(steps_h[n]) * dt,
                cell_index=int(cells_h[n]),
                position=(float(position_h[n][0]), float(position_h[n][1]), float(position_h[n][2])),
                material_id=int(material_h[n]),
                field_before=float(field_h[n]),
                state_before=BREAKDOWN_STATE_INTACT,
                state_after=BREAKDOWN_STATE_CONDUCTING,
                deposited_energy_at_trigger=float(deposited_h[n]),
            )
            for n in range(triggered_total)
        )
    else:
        events = ()

    total = float(runtime["total_dissipated"].item())
    return BreakdownResultData(
        events=events,
        final_state=final_state,
        dissipated_energy=dissipated,
        total_dissipated_energy=total,
        grid_shape=layout.grid_shape,
    )


__all__ = [
    "initialize_breakdown_runtime",
    "advance_breakdown_state",
    "finalize_breakdown_data",
]
