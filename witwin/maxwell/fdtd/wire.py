from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from ..thin_wire import WireData
from .wire_lossy import (
    LossySegmentModel,
    build_lossy_segment_model,
    resolve_lossy_band,
)


@dataclass(frozen=True)
class WireComponentPlan:
    """Per-component scatter/gather indices resolved once at prepare time.

    The component code of every sampling and deposition entry is compile-time
    topology. Resolving it per step with ``torch.nonzero`` would force a
    device-to-host synchronization on every reverse step, so the plan is built
    once and reused. ``entries`` holds one ``(destination, flat_offset)`` index
    pair per electric component, in Ex/Ey/Ez order.
    """

    entries: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    count: int


@dataclass
class WireRuntime:
    """Compressed device state for one compiled thin-wire network."""

    network: Any
    monitors: tuple[Any, ...]
    current: torch.Tensor
    charge: torch.Tensor
    emf: torch.Tensor
    coefficients: dict[str, torch.Tensor]
    sample_plan: WireComponentPlan
    target_plan: WireComponentPlan
    sample_segments: torch.Tensor
    monitor_state: list[dict[str, Any]]
    cfl_limit: float
    wire_cfl_limit: float
    maxwell_cfl_limit: float
    dt_adjusted: bool
    state_bytes: int
    lossy_model: LossySegmentModel | None = None
    ade_state: torch.Tensor | None = None


@dataclass(frozen=True)
class WireReverseResult:
    """Sparse transpose of one lossless wire-network leapfrog step."""

    pre_current: torch.Tensor
    pre_charge: torch.Tensor
    field_adjoint: dict[str, torch.Tensor]
    grad_inductance: torch.Tensor
    grad_node_capacitance: torch.Tensor
    grad_weights: torch.Tensor
    grad_eps: dict[str, torch.Tensor]


def _as_runtime_tensor(value, *, device, dtype=None) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.to(device=device, dtype=dtype, copy=False).contiguous()


def _component_target_mass(solver, component: int, offsets: torch.Tensor) -> torch.Tensor:
    """Return the diagonal electric energy mass for flattened Yee-edge offsets."""

    if component == 0:
        ny, nz = solver.Ny, solver.Nz
        i = offsets // (ny * nz)
        j = (offsets // nz) % ny
        k = offsets % nz
        return (
            solver.eps_Ex.reshape(-1).index_select(0, offsets)
            / solver.inv_dx_h.index_select(0, i)
            / solver.inv_dy_e.index_select(0, j)
            / solver.inv_dz_e.index_select(0, k)
        )
    if component == 1:
        ny, nz = solver.Ny - 1, solver.Nz
        i = offsets // (ny * nz)
        j = (offsets // nz) % ny
        k = offsets % nz
        return (
            solver.eps_Ey.reshape(-1).index_select(0, offsets)
            / solver.inv_dx_e.index_select(0, i)
            / solver.inv_dy_h.index_select(0, j)
            / solver.inv_dz_e.index_select(0, k)
        )
    if component == 2:
        ny, nz = solver.Ny, solver.Nz - 1
        i = offsets // (ny * nz)
        j = (offsets // nz) % ny
        k = offsets % nz
        return (
            solver.eps_Ez.reshape(-1).index_select(0, offsets)
            / solver.inv_dx_e.index_select(0, i)
            / solver.inv_dy_e.index_select(0, j)
            / solver.inv_dz_h.index_select(0, k)
        )
    raise ValueError(f"Unknown electric-field component code {component}.")


def _target_masses(solver, components: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    masses = torch.empty(offsets.numel(), device=solver.device, dtype=solver.Ex.dtype)
    for component in range(3):
        indices = torch.nonzero(components == component, as_tuple=False).reshape(-1)
        if indices.numel() == 0:
            continue
        component_offsets = offsets.index_select(0, indices)
        masses.index_copy_(0, indices, _component_target_mass(solver, component, component_offsets))
    if not bool(torch.all(torch.isfinite(masses) & (masses > 0.0))):
        raise ValueError("Thin-wire coupling encountered a non-positive or non-finite electric energy mass.")
    return masses


def _group_indices(offsets: torch.Tensor) -> torch.Tensor:
    counts = offsets[1:] - offsets[:-1]
    return torch.arange(counts.numel(), device=offsets.device, dtype=torch.int64).repeat_interleave(counts)


def _segmented_sum(values: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Deterministic CSR reduction with the stored contribution order."""

    prefix = torch.cat((values.new_zeros((1,)), torch.cumsum(values, dim=0)))
    return prefix.index_select(0, offsets[1:]) - prefix.index_select(0, offsets[:-1])


def _component_plan(components: torch.Tensor, offsets: torch.Tensor) -> WireComponentPlan:
    """Resolve the per-component index plan for one sampling or deposition list."""

    entries = []
    for component in range(3):
        selected = torch.nonzero(components == component, as_tuple=False).reshape(-1)
        entries.append((selected.contiguous(), offsets.index_select(0, selected).contiguous()))
    return WireComponentPlan(entries=tuple(entries), count=int(offsets.numel()))


def _field_vector(
    fields: dict[str, torch.Tensor],
    plan: WireComponentPlan,
) -> torch.Tensor:
    values = fields["Ex"].new_empty(plan.count)
    for (selected, gather), name in zip(plan.entries, ("Ex", "Ey", "Ez")):
        if selected.numel() == 0:
            continue
        values.index_copy_(
            0,
            selected,
            fields[name].reshape(-1).index_select(0, gather),
        )
    return values


def replay_wire_state(
    solver,
    state: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay ``I^(n+1/2), q^(n+1)`` from one frozen checkpoint state."""

    runtime = getattr(solver, "_wire_runtime", None)
    if runtime is None:
        raise RuntimeError("Wire replay requires an initialized wire runtime.")
    if runtime.lossy_model is not None:
        raise NotImplementedError(
            "Reverse/adjoint replay of a finite-conductor thin wire is not yet "
            "implemented: the lossy current recurrence carries auxiliary ADE state "
            "that the reverse pass must transpose (conductivity adjoint, B3). Run a "
            "PEC wire for differentiable workflows until the lossy reverse lands."
        )
    coeff = runtime.coefficients
    sampled = _field_vector(
        {name: state[name] for name in ("Ex", "Ey", "Ez")},
        runtime.sample_plan,
    )
    emf = _segmented_sum(
        coeff["weights"] * sampled,
        coeff["segment_offsets"],
    )
    charge = state["wire_charge"]
    potential = torch.where(
        coeff["grounded"],
        torch.zeros_like(charge),
        charge / coeff["node_capacitance"],
    )
    current = state["wire_current"] + float(solver.dt) * (
        emf
        + potential.index_select(0, coeff["tail"])
        - potential.index_select(0, coeff["head"])
    ) / coeff["inductance"]
    incidence = coeff["node_signs"].to(dtype=current.dtype) * current.index_select(
        0, coeff["node_segments"]
    )
    flow = _segmented_sum(incidence, coeff["node_offsets"])
    next_charge = torch.where(
        coeff["grounded"],
        torch.zeros_like(charge),
        charge - float(solver.dt) * flow,
    )
    return current, next_charge


def deposit_replayed_wire_current(
    solver,
    electric_fields: dict[str, torch.Tensor],
    current: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Apply the sorted reciprocal current deposition without mutating replay state."""

    runtime = getattr(solver, "_wire_runtime", None)
    if runtime is None:
        return electric_fields
    coeff = runtime.coefficients
    contributions = coeff["contribution_scales"] * current.index_select(
        0, coeff["contribution_segments"]
    )
    deposited = _segmented_sum(contributions, coeff["edge_group_offsets"])
    result = dict(electric_fields)
    for (selected, target_offsets), name in zip(
        runtime.target_plan.entries, ("Ex", "Ey", "Ez")
    ):
        if selected.numel() == 0:
            continue
        flat = electric_fields[name].reshape(-1).clone()
        flat.index_copy_(
            0,
            target_offsets,
            flat.index_select(0, target_offsets) - deposited.index_select(0, selected),
        )
        result[name] = flat.reshape_as(electric_fields[name])
    return result


def reverse_wire_step(
    solver,
    forward_state: dict[str, torch.Tensor],
    post_step_adjoint: dict[str, torch.Tensor],
    *,
    eps_by_field: dict[str, torch.Tensor],
) -> WireReverseResult:
    """Apply the exact sparse transpose of wire sampling, recurrence, and deposition."""

    runtime = getattr(solver, "_wire_runtime", None)
    if runtime is None:
        raise RuntimeError("Wire reverse requires an initialized wire runtime.")
    coeff = runtime.coefficients
    dt = float(solver.dt)
    current0 = forward_state["wire_current"]
    charge0 = forward_state["wire_charge"]
    current1, _charge1 = replay_wire_state(solver, forward_state)

    sample_target_adjoint = _field_vector(
        {name: post_step_adjoint[name] for name in ("Ex", "Ey", "Ez")},
        runtime.sample_plan,
    )
    target_adjoint = _field_vector(
        {name: post_step_adjoint[name] for name in ("Ex", "Ey", "Ez")},
        runtime.target_plan,
    )
    contribution_segments = coeff["contribution_segments"]
    contribution_current = current1.index_select(0, contribution_segments)
    sample_segments = runtime.sample_segments
    current_adjoint = post_step_adjoint["wire_current"] - _segmented_sum(
        coeff["sample_deposition_scales"] * sample_target_adjoint,
        coeff["segment_offsets"],
    )

    charge_adjoint = torch.where(
        coeff["grounded"],
        torch.zeros_like(post_step_adjoint["wire_charge"]),
        post_step_adjoint["wire_charge"],
    )
    tail_charge_adjoint = charge_adjoint.index_select(0, coeff["tail"])
    head_charge_adjoint = charge_adjoint.index_select(0, coeff["head"])
    current_adjoint = current_adjoint - dt * (
        tail_charge_adjoint - head_charge_adjoint
    )

    scaled_current_adjoint = dt * current_adjoint / coeff["inductance"]
    node_potential_adjoint = _segmented_sum(
        coeff["node_signs"].to(dtype=scaled_current_adjoint.dtype)
        * scaled_current_adjoint.index_select(0, coeff["node_segments"]),
        coeff["node_offsets"],
    )
    nongrounded = ~coeff["grounded"]
    pre_charge = charge_adjoint + torch.where(
        nongrounded,
        node_potential_adjoint / coeff["node_capacitance"],
        torch.zeros_like(node_potential_adjoint),
    )
    grad_node_capacitance = torch.where(
        nongrounded,
        -charge0
        * node_potential_adjoint
        / (coeff["node_capacitance"] * coeff["node_capacitance"]),
        torch.zeros_like(charge0),
    )
    grad_inductance = -(
        current1 - current0
    ) * current_adjoint / coeff["inductance"]
    sampled_field = _field_vector(
        {name: forward_state[name] for name in ("Ex", "Ey", "Ez")},
        runtime.sample_plan,
    )
    grad_weights = (
        sampled_field * scaled_current_adjoint.index_select(0, sample_segments)
        - dt
        * sample_target_adjoint
        * current1.index_select(0, sample_segments)
        / coeff["sample_masses"]
    )

    contribution_sample_adjoint = coeff["contribution_weights"] * (
        scaled_current_adjoint.index_select(0, contribution_segments)
    )
    target_sample_adjoint = _segmented_sum(
        contribution_sample_adjoint,
        coeff["edge_group_offsets"],
    )
    field_adjoint = {
        name: torch.zeros_like(forward_state[name]) for name in ("Ex", "Ey", "Ez")
    }
    for (selected, target_offsets), name in zip(
        runtime.target_plan.entries, ("Ex", "Ey", "Ez")
    ):
        if selected.numel() == 0:
            continue
        flat = field_adjoint[name].reshape(-1)
        flat.index_copy_(
            0,
            target_offsets,
            target_sample_adjoint.index_select(0, selected),
        )

    deposited = _segmented_sum(
        coeff["contribution_scales"] * contribution_current,
        coeff["edge_group_offsets"],
    )
    grad_eps = {
        name: torch.zeros_like(eps_by_field[name]) for name in ("Ex", "Ey", "Ez")
    }
    for (selected, offsets), name in zip(
        runtime.target_plan.entries, ("Ex", "Ey", "Ez")
    ):
        if selected.numel() == 0:
            continue
        eps_values = eps_by_field[name].reshape(-1).index_select(0, offsets)
        values = (
            target_adjoint.index_select(0, selected)
            * deposited.index_select(0, selected)
            / eps_values
        )
        grad_eps[name].reshape(-1).index_copy_(0, offsets, values)

    return WireReverseResult(
        pre_current=current_adjoint,
        pre_charge=pre_charge,
        field_adjoint=field_adjoint,
        grad_inductance=grad_inductance,
        grad_node_capacitance=grad_node_capacitance,
        grad_weights=grad_weights,
        grad_eps=grad_eps,
    )


def _require_topology(condition: torch.Tensor | bool, message: str) -> None:
    if not bool(condition):
        raise ValueError(f"Invalid compiled thin-wire topology: {message}.")


def _validate_csr(offsets: torch.Tensor, *, groups: int, entries: int, name: str) -> None:
    _require_topology(offsets.ndim == 1 and offsets.numel() == groups + 1, f"{name} shape")
    _require_topology(offsets[0] == 0, f"{name} must start at zero")
    _require_topology(offsets[-1] == entries, f"{name} terminal offset")
    _require_topology(torch.all(offsets[1:] >= offsets[:-1]), f"{name} must be monotone")


def _validate_compiled_topology(network) -> None:
    """Validate all content assumptions of the synchronization-free CUDA ops once."""

    segment_count = int(network.inductance.numel())
    node_count = int(network.node_capacitance.numel())
    sample_count = int(network.weights.numel())
    fragment_count = int(network.fragment_lengths.numel())
    target_count = int(network.target_offsets.numel())
    contribution_count = int(network.contribution_weights.numel())
    _require_topology(segment_count > 0 and node_count > 0, "network must not be empty")
    _require_topology(
        network.edge_components.numel() == sample_count
        and network.edge_offsets.numel() == sample_count,
        "sampling arrays must have equal lengths",
    )
    _validate_csr(
        network.segment_offsets,
        groups=segment_count,
        entries=sample_count,
        name="segment_offsets",
    )
    _require_topology(
        torch.all(network.segment_offsets[1:] > network.segment_offsets[:-1]),
        "every segment must have a sampling entry",
    )
    _validate_csr(
        network.fragment_offsets,
        groups=fragment_count,
        entries=sample_count,
        name="fragment_offsets",
    )
    _require_topology(
        network.fragment_segment_ids.shape == (fragment_count,)
        and network.fragment_cell_indices.shape == (fragment_count, 3),
        "fragment metadata shapes",
    )
    _require_topology(
        torch.all(network.fragment_lengths > 0.0),
        "fragment lengths must be positive",
    )
    _require_topology(
        torch.all(
            (network.fragment_segment_ids >= 0)
            & (network.fragment_segment_ids < segment_count)
        ),
        "fragment segment indices must be in bounds",
    )
    fragment_counts = torch.bincount(
        network.fragment_segment_ids,
        minlength=segment_count,
    )
    fragment_boundaries = torch.cat(
        (
            fragment_counts.new_zeros((1,)),
            torch.cumsum(fragment_counts, dim=0),
        )
    )
    _require_topology(
        torch.all(fragment_counts > 0)
        and torch.equal(
            network.segment_offsets,
            network.fragment_offsets.index_select(0, fragment_boundaries),
        ),
        "fragment rows must be contiguous within physical segments",
    )
    _require_topology(
        torch.all((network.edge_components >= 0) & (network.edge_components < 3)),
        "edge component codes must be in [0, 2]",
    )
    _require_topology(
        torch.all((network.target_components >= 0) & (network.target_components < 3)),
        "target component codes must be in [0, 2]",
    )
    for component, shape in enumerate(network.field_shapes):
        field_size = math.prod(int(value) for value in shape)
        sample_mask = network.edge_components == component
        target_mask = network.target_components == component
        _require_topology(
            torch.all(
                (~sample_mask)
                | ((network.edge_offsets >= 0) & (network.edge_offsets < field_size))
            ),
            f"sampling offsets for component {component} must be in bounds",
        )
        _require_topology(
            torch.all(
                (~target_mask)
                | ((network.target_offsets >= 0) & (network.target_offsets < field_size))
            ),
            f"target offsets for component {component} must be in bounds",
        )

    _require_topology(
        network.target_components.numel() == target_count,
        "target arrays must have equal lengths",
    )
    _require_topology(
        network.contribution_segments.numel() == contribution_count,
        "deposition contribution arrays must have equal lengths",
    )
    _validate_csr(
        network.edge_group_offsets,
        groups=target_count,
        entries=contribution_count,
        name="edge_group_offsets",
    )
    _require_topology(
        torch.all(network.edge_group_offsets[1:] > network.edge_group_offsets[:-1]),
        "every deposit target must have a contribution",
    )
    _require_topology(
        torch.all(
            (network.contribution_segments >= 0)
            & (network.contribution_segments < segment_count)
        ),
        "deposition segment indices must be in bounds",
    )
    largest_field = max(math.prod(int(value) for value in shape) for shape in network.field_shapes)
    target_keys = network.target_components.to(torch.int64) * largest_field + network.target_offsets
    _require_topology(
        torch.unique(target_keys).numel() == target_count,
        "deposit targets must be unique",
    )

    _require_topology(
        network.tail.numel() == segment_count and network.head.numel() == segment_count,
        "segment endpoint arrays must match the segment count",
    )
    _require_topology(
        torch.all(
            (network.tail >= 0)
            & (network.tail < node_count)
            & (network.head >= 0)
            & (network.head < node_count)
            & (network.tail != network.head)
        ),
        "segment endpoint indices must be distinct and in bounds",
    )
    incidence_count = int(network.node_segments.numel())
    _require_topology(
        network.node_signs.numel() == incidence_count,
        "node incidence arrays must have equal lengths",
    )
    _validate_csr(
        network.node_offsets,
        groups=node_count,
        entries=incidence_count,
        name="node_offsets",
    )
    _require_topology(
        torch.all((network.node_segments >= 0) & (network.node_segments < segment_count)),
        "node incidence segment indices must be in bounds",
    )
    _require_topology(
        torch.all((network.node_signs == 1) | (network.node_signs == -1)),
        "node incidence signs must be +1 or -1",
    )
    _require_topology(
        torch.all(torch.bincount(network.node_segments, minlength=segment_count) == 2),
        "every segment must have exactly two node incidences",
    )
    signed_incidence = torch.zeros(
        segment_count, device=network.node_signs.device, dtype=torch.int32
    )
    signed_incidence.index_add_(0, network.node_segments, network.node_signs)
    _require_topology(
        torch.all(signed_incidence == 0),
        "every segment must have one tail and one head incidence",
    )
    node_groups = _group_indices(network.node_offsets)
    expected_nodes = torch.where(
        network.node_signs == 1,
        network.tail.index_select(0, network.node_segments),
        network.head.index_select(0, network.node_segments),
    )
    _require_topology(
        torch.equal(node_groups, expected_nodes),
        "node CSR signs must match tail/head incidence",
    )

    wire_count = len(network.wire_names)
    wire_node_count = int(network.wire_node_indices.numel())
    _validate_csr(
        network.wire_node_offsets,
        groups=wire_count,
        entries=wire_node_count,
        name="wire_node_offsets",
    )
    _require_topology(
        torch.all(network.wire_node_offsets[1:] > network.wire_node_offsets[:-1]),
        "every wire must own at least one node",
    )
    _require_topology(
        torch.all(
            (network.wire_node_indices >= 0) & (network.wire_node_indices < node_count)
        ),
        "wire node indices must be in bounds",
    )
    _require_topology(
        network.node_wire_ids.numel() == node_count
        and torch.all(
            (network.node_wire_ids >= 0) & (network.node_wire_ids < wire_count)
        ),
        "node owner IDs must match the node count and reference a wire",
    )
    _validate_csr(
        network.wire_segment_offsets,
        groups=wire_count,
        entries=segment_count,
        name="wire_segment_offsets",
    )
    expected_wire_ids = _group_indices(network.wire_segment_offsets)
    _require_topology(
        network.segment_wire_ids.numel() == segment_count
        and torch.equal(network.segment_wire_ids, expected_wire_ids),
        "segment wire IDs must match wire_segment_offsets",
    )
    _require_topology(
        network.segment_source_ids.numel() == segment_count
        and torch.all(network.segment_source_ids >= 0),
        "source segment IDs must match the segment count and be non-negative",
    )
    expected_node_owners = torch.full(
        (node_count,), wire_count, device=network.node_wire_ids.device, dtype=torch.int64
    )
    for wire_id in range(wire_count):
        node_start = int(network.wire_node_offsets[wire_id])
        node_end = int(network.wire_node_offsets[wire_id + 1])
        segment_start = int(network.wire_segment_offsets[wire_id])
        segment_end = int(network.wire_segment_offsets[wire_id + 1])
        members = network.wire_node_indices[node_start:node_end]
        endpoints = torch.cat(
            (network.tail[segment_start:segment_end], network.head[segment_start:segment_end])
        )
        _require_topology(
            torch.all(torch.isin(endpoints, members)),
            f"wire {wire_id} segment endpoints must belong to its node membership",
        )
        unique_members = torch.unique(members, sorted=True)
        expected_members = torch.unique(endpoints, sorted=True)
        _require_topology(
            unique_members.numel() == members.numel()
            and torch.equal(unique_members, expected_members),
            f"wire {wire_id} node membership must exactly match its unique segment endpoints",
        )
        owner_candidates = expected_node_owners.index_select(0, members)
        expected_node_owners.index_copy_(
            0,
            members,
            torch.minimum(owner_candidates, torch.full_like(owner_candidates, wire_id)),
        )
    _require_topology(
        torch.equal(network.node_wire_ids, expected_node_owners),
        "node owner IDs must be the minimum wire ID in each node membership",
    )

    _require_topology(
        network.open_endpoints.numel() == node_count
        and network.grounded.numel() == node_count,
        "endpoint flags must match the node count",
    )
    _require_topology(
        torch.all(~(network.open_endpoints & network.grounded)),
        "a node cannot be both open and grounded",
    )
    degrees = network.node_offsets[1:] - network.node_offsets[:-1]
    _require_topology(
        torch.all(degrees[network.open_endpoints | network.grounded] == 1),
        "open and grounded endpoints must have degree one",
    )
    junction_count = len(network.junction_names)
    _require_topology(
        len(set(network.junction_names)) == junction_count,
        "junction names must be unique",
    )
    _require_topology(
        network.junction_node_ids.numel() == junction_count,
        "junction IDs must match junction names",
    )
    _require_topology(
        torch.all(
            (network.junction_node_ids >= 0) & (network.junction_node_ids < node_count)
        )
        and torch.unique(network.junction_node_ids).numel() == junction_count,
        "junction node IDs must be unique and in bounds",
    )
    _require_topology(
        torch.all(degrees.index_select(0, network.junction_node_ids) >= 2),
        "named junctions must have degree at least two",
    )
    branch_nodes = torch.nonzero(degrees > 2, as_tuple=False).reshape(-1)
    _require_topology(
        torch.all(torch.isin(branch_nodes, network.junction_node_ids)),
        "every branch node must be a named junction",
    )
    _require_topology(
        torch.all(torch.isfinite(network.weights))
        and torch.all(torch.isfinite(network.contribution_weights)),
        "coupling weights must be finite",
    )
    _require_topology(
        torch.all(torch.isfinite(network.inductance) & (network.inductance > 0.0)),
        "segment inductance must be finite and positive",
    )
    _require_topology(
        torch.all(
            torch.isfinite(network.node_capacitance) & (network.node_capacitance > 0.0)
        ),
        "node capacitance must be finite and positive",
    )


def _wire_cfl_limit(
    *,
    inductance: torch.Tensor,
    node_capacitance: torch.Tensor,
    grounded: torch.Tensor,
    node_offsets: torch.Tensor,
    node_segments: torch.Tensor,
    edge_group_offsets: torch.Tensor,
    contribution_segments: torch.Tensor,
    contribution_weights: torch.Tensor,
    target_masses: torch.Tensor,
) -> float:
    """Gershgorin-safe bound for the coupled field/charge wire operator.

    The calculation uses only O(nodes + segments + coupling entries) temporary
    storage. It therefore preserves the compressed-state contract even for long
    wires and shared field edges.

    The row bound is accumulated in float64 regardless of the runtime field
    dtype. This is a prepare-time cost only, and it keeps the reported limit from
    depending on float32 rounding: the Gershgorin bound is exactly tight against
    the leapfrog eigenvalue limit on uniform-coefficient wire topologies, so a
    float32 accumulation could round it to either side of that equality.
    """

    inductance = inductance.to(torch.float64)
    node_capacitance = node_capacitance.to(torch.float64)
    contribution_weights = contribution_weights.to(torch.float64)
    target_masses = target_masses.to(torch.float64)

    segment_count = inductance.numel()
    inv_sqrt_l = torch.rsqrt(inductance)
    row_bound = torch.zeros(segment_count, device=inductance.device, dtype=inductance.dtype)

    edge_groups = _group_indices(edge_group_offsets)
    normalized_weights = contribution_weights.abs() * inv_sqrt_l.index_select(
        0, contribution_segments
    )
    edge_sums = torch.zeros_like(target_masses)
    edge_sums.index_add_(0, edge_groups, normalized_weights)
    row_bound.index_add_(
        0,
        contribution_segments,
        normalized_weights
        * edge_sums.index_select(0, edge_groups)
        / target_masses.index_select(0, edge_groups),
    )

    node_groups = _group_indices(node_offsets)
    normalized_incidence = inv_sqrt_l.index_select(0, node_segments)
    node_sums = torch.zeros_like(node_capacitance)
    node_sums.index_add_(0, node_groups, normalized_incidence)
    inverse_capacitance = torch.where(
        grounded,
        torch.zeros_like(node_capacitance),
        node_capacitance.reciprocal(),
    )
    row_bound.index_add_(
        0,
        node_segments,
        normalized_incidence
        * node_sums.index_select(0, node_groups)
        * inverse_capacitance.index_select(0, node_groups),
    )

    lambda_upper = float(row_bound.max().item())
    if not math.isfinite(lambda_upper) or lambda_upper <= 0.0:
        raise ValueError("Thin-wire coupled CFL operator has a non-positive or non-finite bound.")
    return 2.0 / math.sqrt(lambda_upper)


def _reject_unsupported_composition(solver) -> None:
    if solver.scene.boundary.uses_kind("bloch"):
        raise NotImplementedError(
            "Thin-wire FDTD does not yet support Bloch-periodic phase topology."
        )
    if getattr(solver, "full_aniso_enabled", False):
        raise NotImplementedError(
            "Thin-wire FDTD does not yet support full off-diagonal anisotropic electric updates."
        )
    if getattr(solver, "conductive_enabled", False):
        raise NotImplementedError(
            "Thin-wire FDTD does not yet support a conductive background electric update."
        )
    if getattr(solver, "dispersive_enabled", False):
        raise NotImplementedError("Thin-wire FDTD does not yet support dispersive background media.")
    if getattr(solver, "nonlinear_enabled", False) or getattr(solver, "modulation_enabled", False):
        raise NotImplementedError(
            "Thin-wire FDTD does not yet support nonlinear or time-modulated background media."
        )
    material_model = getattr(solver, "_compiled_material_model", None)
    if getattr(solver, "surface_impedance_enabled", False) or (
        material_model is not None and material_model.get("surface_impedance")
    ):
        raise NotImplementedError(
            "Thin-wire FDTD cannot share conductor ownership with a surface-impedance "
            "boundary until wire loss and SIBC power accounting are coupled."
        )


def _maxwell_cfl_limit(solver) -> float:
    return 1.0 / (
        float(solver.c)
        * math.sqrt(
            1.0 / float(solver.min_dx) ** 2
            + 1.0 / float(solver.min_dy) ** 2
            + 1.0 / float(solver.min_dz) ** 2
        )
    )


def initialize_wire_runtime(solver) -> WireRuntime | None:
    wires = tuple(getattr(solver.scene, "thin_wires", ()))
    if not wires:
        solver._wire_runtime = None
        return None

    _reject_unsupported_composition(solver)
    network = solver.scene.compile_thin_wires()
    _validate_compiled_topology(network)
    monitors = tuple(solver.scene.compile_wire_monitors())
    device = solver.device
    dtype = solver.Ex.dtype

    coefficients = {
        "segment_offsets": _as_runtime_tensor(network.segment_offsets, device=device, dtype=torch.int64),
        "edge_components": _as_runtime_tensor(network.edge_components, device=device, dtype=torch.int32),
        "edge_offsets": _as_runtime_tensor(network.edge_offsets, device=device, dtype=torch.int64),
        "weights": _as_runtime_tensor(network.weights, device=device, dtype=dtype),
        "tail": _as_runtime_tensor(network.tail, device=device, dtype=torch.int64),
        "head": _as_runtime_tensor(network.head, device=device, dtype=torch.int64),
        "inductance": _as_runtime_tensor(network.inductance, device=device, dtype=dtype),
        "node_capacitance": _as_runtime_tensor(network.node_capacitance, device=device, dtype=dtype),
        "grounded": _as_runtime_tensor(network.grounded, device=device, dtype=torch.bool),
        "node_offsets": _as_runtime_tensor(network.node_offsets, device=device, dtype=torch.int64),
        "node_segments": _as_runtime_tensor(network.node_segments, device=device, dtype=torch.int64),
        "node_signs": _as_runtime_tensor(network.node_signs, device=device, dtype=torch.int32),
        "edge_group_offsets": _as_runtime_tensor(
            network.edge_group_offsets, device=device, dtype=torch.int64
        ),
        "target_components": _as_runtime_tensor(
            network.target_components, device=device, dtype=torch.int32
        ),
        "target_offsets": _as_runtime_tensor(network.target_offsets, device=device, dtype=torch.int64),
        "contribution_segments": _as_runtime_tensor(
            network.contribution_segments, device=device, dtype=torch.int64
        ),
        "contribution_weights": _as_runtime_tensor(
            network.contribution_weights, device=device, dtype=dtype
        ),
    }
    target_masses = _target_masses(
        solver,
        coefficients["target_components"],
        coefficients["target_offsets"],
    )
    wire_cfl_limit = _wire_cfl_limit(
        inductance=coefficients["inductance"],
        node_capacitance=coefficients["node_capacitance"],
        grounded=coefficients["grounded"],
        node_offsets=coefficients["node_offsets"],
        node_segments=coefficients["node_segments"],
        edge_group_offsets=coefficients["edge_group_offsets"],
        contribution_segments=coefficients["contribution_segments"],
        contribution_weights=coefficients["contribution_weights"],
        target_masses=target_masses,
    )
    maxwell_cfl_limit = _maxwell_cfl_limit(solver)
    cfl_limit = 1.0 / math.sqrt(
        maxwell_cfl_limit**-2 + wire_cfl_limit**-2
    )
    dt_adjusted = float(solver.dt) >= cfl_limit
    if dt_adjusted:
        # The Maxwell curl and reciprocal wire coupling act in the same
        # leapfrog step. Adding their positive-semidefinite spectral bounds is
        # conservative and avoids treating two individually stable limits as
        # a stable composition. The remaining one-percent margin also keeps
        # float32 coefficient rounding away from the equality boundary.
        solver.dt = 0.99 * cfl_limit

    edge_groups = _group_indices(coefficients["edge_group_offsets"])
    coefficients["contribution_scales"] = (
        float(solver.dt)
        * coefficients["contribution_weights"]
        / target_masses.index_select(0, edge_groups)
    ).contiguous()
    sample_masses = _target_masses(
        solver,
        coefficients["edge_components"],
        coefficients["edge_offsets"],
    )
    coefficients["sample_masses"] = sample_masses
    coefficients["sample_deposition_scales"] = (
        float(solver.dt) * coefficients["weights"] / sample_masses
    ).contiguous()

    segment_count = int(coefficients["inductance"].numel())
    node_count = int(coefficients["node_capacitance"].numel())
    current = torch.zeros(segment_count, device=device, dtype=dtype)
    charge = torch.zeros(node_count, device=device, dtype=dtype)
    emf = torch.zeros(segment_count, device=device, dtype=dtype)

    # Finite-conductor segments consume a passive series-impedance ADE in the
    # current recurrence (fdtd/wire_lossy.py). Build it with the FINAL dt (after
    # the joint Maxwell/wire CFL adjustment above) so the discrete companion and
    # its stability certificate use the timestep the run actually steps. The
    # fitting band is derived from the monitored frequencies (fail-closed when a
    # lossy wire carries no frequency to fit at).
    lossy_model = None
    ade_state = None
    if network.metadata.get("has_finite_conductor"):
        lossy_model = build_lossy_segment_model(
            inductance=coefficients["inductance"],
            radius=network.radius.to(device=device, dtype=dtype),
            length=network.length.to(device=device, dtype=dtype),
            segment_wire_ids=network.segment_wire_ids,
            metadata=network.metadata,
            band=resolve_lossy_band(_lossy_band_frequencies(solver, monitors)),
            dt=float(solver.dt),
        )
        if lossy_model is not None:
            ade_state = lossy_model.initial_state()

    state_bytes = sum(value.numel() * value.element_size() for value in (current, charge, emf))
    if ade_state is not None:
        state_bytes += ade_state.numel() * ade_state.element_size()
    runtime = WireRuntime(
        network=network,
        monitors=monitors,
        current=current,
        charge=charge,
        emf=emf,
        coefficients=coefficients,
        sample_plan=_component_plan(
            coefficients["edge_components"], coefficients["edge_offsets"]
        ),
        target_plan=_component_plan(
            coefficients["target_components"], coefficients["target_offsets"]
        ),
        sample_segments=_group_indices(coefficients["segment_offsets"]),
        monitor_state=[],
        cfl_limit=cfl_limit,
        wire_cfl_limit=wire_cfl_limit,
        maxwell_cfl_limit=maxwell_cfl_limit,
        dt_adjusted=dt_adjusted,
        state_bytes=state_bytes,
        lossy_model=lossy_model,
        ade_state=ade_state,
    )
    solver._wire_runtime = runtime
    return runtime


def _lossy_band_frequencies(solver, monitors) -> tuple[float, ...]:
    """Collect the frequencies that set a lossy wire's fitting band.

    Uses the wire monitors' frequencies (the quantities read out) and anchors to
    the source frequency so a lossy wire always fits where the run cares.
    """

    frequencies: list[float] = []
    for monitor in monitors:
        frequencies.extend(float(value) for value in getattr(monitor, "frequencies", ()))
    source_frequency = getattr(solver, "source_frequency", None)
    if source_frequency is not None:
        frequencies.append(float(source_frequency))
    return tuple(frequencies)


def _update_wire_state_lossy(solver, runtime) -> None:
    """Advance I/q for a network with finite-conductor segments (torch path).

    Mirrors the CUDA ``updateWireState1D`` leapfrog ordering (current update with
    the pre-step charge, then charge update with the new current) but replaces the
    current update with the passive series-impedance companion. PEC segments in
    the same network keep G=0 (the lossless leapfrog). The pure-PEC path never
    reaches here, so it stays byte-identical.
    """

    coeff = runtime.coefficients
    model = runtime.lossy_model
    charge = runtime.charge
    potential = torch.where(
        coeff["grounded"],
        torch.zeros_like(charge),
        charge / coeff["node_capacitance"],
    )
    drive = runtime.emf + potential.index_select(0, coeff["tail"]) - potential.index_select(
        0, coeff["head"]
    )
    current_next, ade_next = model.advance_current(
        runtime.current, runtime.ade_state, drive
    )
    incidence = coeff["node_signs"].to(dtype=current_next.dtype) * current_next.index_select(
        0, coeff["node_segments"]
    )
    flow = _segmented_sum(incidence, coeff["node_offsets"])
    charge_next = torch.where(
        coeff["grounded"],
        torch.zeros_like(charge),
        charge - float(solver.dt) * flow,
    )
    # In-place writes preserve the runtime tensors referenced by monitors/checkpoints.
    runtime.current.copy_(current_next)
    runtime.charge.copy_(charge_next)
    runtime.ade_state.copy_(ade_next)


def sample_and_update_wire(solver) -> None:
    runtime = solver._wire_runtime
    if runtime is None:
        return
    coeff = runtime.coefficients
    solver.fdtd_module.sampleWireEmf3D(
        Ex=solver.Ex,
        Ey=solver.Ey,
        Ez=solver.Ez,
        segmentOffsets=coeff["segment_offsets"],
        edgeComponents=coeff["edge_components"],
        edgeOffsets=coeff["edge_offsets"],
        weights=coeff["weights"],
        emf=runtime.emf,
    ).launchRaw()
    if runtime.lossy_model is not None:
        _update_wire_state_lossy(solver, runtime)
        return
    solver.fdtd_module.updateWireState1D(
        emf=runtime.emf,
        tail=coeff["tail"],
        head=coeff["head"],
        inductance=coeff["inductance"],
        nodeCapacitance=coeff["node_capacitance"],
        grounded=coeff["grounded"],
        nodeOffsets=coeff["node_offsets"],
        nodeSegments=coeff["node_segments"],
        nodeSigns=coeff["node_signs"],
        dt=float(solver.dt),
        current=runtime.current,
        charge=runtime.charge,
    ).launchRaw()


def deposit_wire_current(solver) -> None:
    runtime = solver._wire_runtime
    if runtime is None:
        return
    coeff = runtime.coefficients
    solver.fdtd_module.depositWireCurrent3D(
        Ex=solver.Ex,
        Ey=solver.Ey,
        Ez=solver.Ez,
        edgeGroupOffsets=coeff["edge_group_offsets"],
        targetComponents=coeff["target_components"],
        targetOffsets=coeff["target_offsets"],
        contributionSegments=coeff["contribution_segments"],
        contributionScales=coeff["contribution_scales"],
        current=runtime.current,
    ).launchRaw()


def _monitor_entry(solver, frequency: float, time_steps: int) -> dict[str, Any]:
    omega_dt = 2.0 * math.pi * float(frequency) * float(solver.dt)
    return {
        "frequency": float(frequency),
        "start_step": solver._compute_spectral_start_step(
            float(frequency), window_type=solver.observer_window_type
        ),
        "end_step": int(time_steps),
        "sample_count": 0,
        "window_normalization": 0.0,
        "phase_cos": 1.0,
        "phase_sin": 0.0,
        "step_cos": math.cos(omega_dt),
        "step_sin": math.sin(omega_dt),
        "half_cos": math.cos(0.5 * omega_dt),
        "half_sin": math.sin(0.5 * omega_dt),
    }


def prepare_wire_monitors(solver, time_steps: int, window_type: str) -> None:
    runtime = solver._wire_runtime
    if runtime is None:
        return
    solver.observer_window_type = solver._resolve_spectral_window_type(window_type)
    runtime.monitor_state = []
    for monitor in runtime.monitors:
        quantities = set(monitor.quantities)
        segment_indices = _as_runtime_tensor(
            monitor.segment_indices, device=solver.device, dtype=torch.int32
        )
        node_indices = _as_runtime_tensor(monitor.node_indices, device=solver.device, dtype=torch.int32)
        frequencies = tuple(float(value) for value in monitor.frequencies)
        state = {
            "compiled": monitor,
            "entries": [_monitor_entry(solver, frequency, time_steps) for frequency in frequencies],
            "frequencies": torch.tensor(frequencies, device=solver.device, dtype=solver.Ex.dtype),
            "segment_indices": segment_indices,
            "segment_zeros": torch.zeros_like(segment_indices),
            "node_indices": node_indices,
            "node_zeros": torch.zeros_like(node_indices),
            "current_real": (
                torch.zeros(
                    (len(frequencies), segment_indices.numel()),
                    device=solver.device,
                    dtype=solver.Ex.dtype,
                )
                if "current" in quantities
                else None
            ),
            "current_imag": (
                torch.zeros(
                    (len(frequencies), segment_indices.numel()),
                    device=solver.device,
                    dtype=solver.Ex.dtype,
                )
                if "current" in quantities
                else None
            ),
            "charge_real": (
                torch.zeros(
                    (len(frequencies), node_indices.numel()),
                    device=solver.device,
                    dtype=solver.Ex.dtype,
                )
                if "charge" in quantities
                else None
            ),
            "charge_imag": (
                torch.zeros(
                    (len(frequencies), node_indices.numel()),
                    device=solver.device,
                    dtype=solver.Ex.dtype,
                )
                if "charge" in quantities
                else None
            ),
        }
        runtime.monitor_state.append(state)


def _accumulate_vector(
    solver,
    field: torch.Tensor,
    indices: torch.Tensor,
    zeros: torch.Tensor,
    real: torch.Tensor,
    imag: torch.Tensor,
    *,
    weighted_cos: float,
    weighted_sin: float,
) -> None:
    solver.fdtd_module.accumulatePointObservers3D(
        field=field.reshape(-1, 1, 1),
        pointI=indices,
        pointJ=zeros,
        pointK=zeros,
        realAccum=real,
        imagAccum=imag,
        weightedCos=weighted_cos,
        weightedSin=weighted_sin,
    ).launchRaw()


def accumulate_wire_monitors(solver, n: int) -> None:
    runtime = solver._wire_runtime
    if runtime is None or not runtime.monitor_state:
        return
    for state in runtime.monitor_state:
        for frequency_index, entry in enumerate(state["entries"]):
            phase_cos = entry["phase_cos"]
            phase_sin = entry["phase_sin"]
            if entry["start_step"] <= n < entry["end_step"]:
                window_weight = solver._compute_window_weight(
                    n,
                    start_step=entry["start_step"],
                    end_step=entry["end_step"],
                    window_type=solver.observer_window_type,
                )
                current_cos = phase_cos * entry["half_cos"] - phase_sin * entry["half_sin"]
                current_sin = phase_sin * entry["half_cos"] + phase_cos * entry["half_sin"]
                charge_cos = phase_cos * entry["step_cos"] - phase_sin * entry["step_sin"]
                charge_sin = phase_sin * entry["step_cos"] + phase_cos * entry["step_sin"]
                if state["current_real"] is not None:
                    _accumulate_vector(
                        solver,
                        runtime.current,
                        state["segment_indices"],
                        state["segment_zeros"],
                        state["current_real"][frequency_index],
                        state["current_imag"][frequency_index],
                        weighted_cos=window_weight * current_cos,
                        weighted_sin=window_weight * current_sin,
                    )
                if state["charge_real"] is not None:
                    _accumulate_vector(
                        solver,
                        runtime.charge,
                        state["node_indices"],
                        state["node_zeros"],
                        state["charge_real"][frequency_index],
                        state["charge_imag"][frequency_index],
                        weighted_cos=window_weight * charge_cos,
                        weighted_sin=window_weight * charge_sin,
                    )
                entry["window_normalization"] += window_weight
                entry["sample_count"] += 1
            entry["phase_cos"] = phase_cos * entry["step_cos"] - phase_sin * entry["step_sin"]
            entry["phase_sin"] = phase_sin * entry["step_cos"] + phase_cos * entry["step_sin"]


def complete_wire_monitor_normalization(solver, time_steps: int) -> None:
    runtime = solver._wire_runtime
    if runtime is None:
        return
    for state in runtime.monitor_state:
        for entry in state["entries"]:
            total = int(time_steps) - int(entry["start_step"])
            if total <= 0:
                entry["window_normalization"] = 0.0
            elif solver.observer_window_type == "none":
                entry["window_normalization"] = float(total)
            elif solver.observer_window_type == "hanning":
                entry["window_normalization"] = sum(
                    0.5 * (1.0 - math.cos(2.0 * math.pi * position / total))
                    for position in range(total)
                )
            else:
                entry["window_normalization"] = sum(
                    0.5 * (1.0 - math.cos(math.pi * (position / total) / 0.1))
                    if position / total < 0.1
                    else 1.0
                    for position in range(total)
                )


def finalize_wire_data(solver) -> dict[str, WireData]:
    runtime = solver._wire_runtime
    if runtime is None:
        return {}
    results = {}
    network = runtime.network
    for state in runtime.monitor_state:
        scales = torch.tensor(
            [
                2.0 / entry["window_normalization"]
                if entry["window_normalization"] > 0.0
                else 0.0
                for entry in state["entries"]
            ],
            device=solver.device,
            dtype=solver.Ex.dtype,
        ).unsqueeze(1)
        monitor = state["compiled"]
        current = (
            scales * torch.complex(state["current_real"], state["current_imag"])
            if state["current_real"] is not None
            else None
        )
        charge = (
            scales * torch.complex(state["charge_real"], state["charge_imag"])
            if state["charge_real"] is not None
            else None
        )
        segment_indices = monitor.segment_indices.to(device=network.segment_ids.device, dtype=torch.int64)
        ohmic_loss = None
        if "ohmic_loss" in monitor.quantities:
            # Time-averaged ohmic dissipation per monitored segment at each
            # frequency: P = 0.5 * Re(Z_series(f)) * |I(f)|^2 with the per-segment
            # series AC resistance Re(Z'(f)) * length the passive companion realizes
            # (R_dc + Re(excess_fit)); PEC segments contribute zero. This is the
            # cycle-averaged dissipation the recurrence removes (the instantaneous
            # companion dissipation integrates to it), so the monitored channel
            # closes the wire energy budget.
            if current is None:
                raise ValueError(
                    "A wire ohmic_loss monitor requires the current quantity."
                )
            if runtime.lossy_model is None:
                ohmic_loss = torch.zeros(
                    (len(state["entries"]), segment_indices.numel()),
                    device=solver.device,
                    dtype=solver.Ex.dtype,
                )
            else:
                frequencies = torch.tensor(
                    [entry["frequency"] for entry in state["entries"]],
                    dtype=torch.float64,
                )
                ac_resistance = runtime.lossy_model.ac_resistance_per_length(
                    frequencies
                ).index_select(
                    1, segment_indices.to(device=solver.device)
                )
                seg_length = network.length.index_select(0, segment_indices).to(
                    device=solver.device, dtype=solver.Ex.dtype
                )
                ohmic_loss = (
                    0.5
                    * ac_resistance
                    * seg_length.reshape(1, -1)
                    * current.abs().square()
                )
        node_indices = monitor.node_indices.to(device=network.node_ids.device, dtype=torch.int64)
        metadata = {
            "quantities": tuple(monitor.quantities),
            "segment_ids": network.segment_ids.index_select(0, segment_indices),
            "node_ids": network.node_ids.index_select(0, node_indices),
            "segment_positions": 0.5
            * (
                network.node_positions.index_select(
                    0, network.tail.index_select(0, segment_indices)
                )
                + network.node_positions.index_select(
                    0, network.head.index_select(0, segment_indices)
                )
            ),
            "node_positions": network.node_positions.index_select(0, node_indices),
            "phasor_convention": "peak phasor with exp(-i*omega*t) time dependence",
            "wire_cfl_limit": runtime.cfl_limit,
            "uncoupled_wire_cfl_limit": runtime.wire_cfl_limit,
            "maxwell_cfl_limit": runtime.maxwell_cfl_limit,
            "time_step_adjusted": runtime.dt_adjusted,
            "validity": dict(network.metadata.get("validity", {})),
        }
        results[monitor.name] = WireData(
            monitor_name=monitor.name,
            wire_name=monitor.wire_name,
            frequencies=state["frequencies"],
            current=current,
            charge=charge,
            ohmic_loss=ohmic_loss,
            metadata=metadata,
        )
    return results


__all__ = [
    "WireRuntime",
    "accumulate_wire_monitors",
    "complete_wire_monitor_normalization",
    "deposit_wire_current",
    "finalize_wire_data",
    "initialize_wire_runtime",
    "prepare_wire_monitors",
    "sample_and_update_wire",
]
