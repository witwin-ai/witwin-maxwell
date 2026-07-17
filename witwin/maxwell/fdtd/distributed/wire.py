from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math
from typing import Any

import torch

from ...compiler.thin_wire import CompiledWireMonitor, CompiledWireNetwork
from ...fdtd_parallel import FDTDPartitionPlan
from ...thin_wire import WireData
from ..wire import (
    WireComponentPlan,
    WireRuntime,
    _component_plan,
    _group_indices,
    _target_masses,
    accumulate_wire_monitors,
    complete_wire_monitor_normalization,
    finalize_wire_data,
    prepare_wire_monitors,
)


_COMPONENTS = ("Ex", "Ey", "Ez")


def _decode_edge(component: int, offset: int, field_shapes) -> tuple[int, int, int]:
    """Decode a compiled flat Yee-edge offset into its ``(i, j, k)`` grid index."""

    shape = field_shapes[component]
    _sx, sy, sz = (int(value) for value in shape)
    i = offset // (sy * sz)
    j = (offset // sz) % sy
    k = offset % sz
    return int(i), int(j), int(k)


def _local_flat(component_layout, index: tuple[int, int, int]) -> int:
    local = component_layout.global_to_local(index)
    _lx, ly, lz = (int(value) for value in component_layout.local_shape)
    return (int(local[0]) * ly + int(local[1])) * lz + int(local[2])


@dataclass(frozen=True)
class _ShardSamplePlan:
    """Cell-local sampling entries owned by one shard, grouped by physical segment."""

    rank: int
    segment_offsets: torch.Tensor  # int64, [num_segments + 1]
    edge_components: torch.Tensor  # int32, [E_s]
    edge_offsets: torch.Tensor  # int64, [E_s] local flat
    weights: torch.Tensor  # float, [E_s]


@dataclass(frozen=True)
class _ShardDepositPlan:
    """Deposition targets owned by one shard, with their contribution lists."""

    rank: int
    edge_group_offsets: torch.Tensor  # int64, [T_s + 1]
    target_components: torch.Tensor  # int32, [T_s]
    target_offsets: torch.Tensor  # int64, [T_s] local flat
    contribution_segments: torch.Tensor  # int64, [C_s] global segment ids
    contribution_weights: torch.Tensor  # float, [C_s]


@dataclass(frozen=True)
class DistributedWirePlan:
    """Deterministic ownership plan for one distributed thin-wire network.

    The whole compressed recurrence (``I``/``q`` and the wire monitors) is owned
    by a single ``owner_rank`` shard chosen from a stable global edge id, so the
    coupled node-charge continuity solve never needs cross-owner communication.
    The field coupling is distributed: every sampling/deposition edge is applied
    by the shard that owns that Yee edge, and per-segment EMF partials are reduced
    to the owner while ``I`` is broadcast back for deposition.
    """

    owner_rank: int
    num_segments: int
    num_nodes: int
    owner_reference_edge: tuple[int, int, int, int]
    sample_plans: tuple[_ShardSamplePlan, ...]
    deposit_plans: tuple[_ShardDepositPlan, ...]

    @property
    def cross_shard_segment_count(self) -> int:
        """Physical segments whose sampling entries span more than one shard."""

        touched: dict[int, set[int]] = {}
        for plan in self.sample_plans:
            counts = plan.segment_offsets[1:] - plan.segment_offsets[:-1]
            for segment in torch.nonzero(counts, as_tuple=False).reshape(-1).tolist():
                touched.setdefault(int(segment), set()).add(plan.rank)
        return sum(1 for ranks in touched.values() if len(ranks) > 1)


def compile_distributed_wire_plan(
    network: CompiledWireNetwork,
    partition_plan: FDTDPartitionPlan,
) -> DistributedWirePlan:
    """Resolve deterministic wire fragment/state ownership without touching CUDA.

    Pure topology: it decodes each compiled global Yee-edge offset, assigns it to
    the shard that owns that edge, remaps it to that shard's local flat index, and
    groups the results back into per-shard sampling and deposition plans. The owner
    of the compressed ``I``/``q`` state is the shard owning the lexicographically
    smallest ``(i, j, k, component)`` sampling edge, which is independent of wire
    declaration order because the compiler emits a name-sorted, deterministic edge
    list.
    """

    layouts = partition_plan.shard_layouts
    field_shapes = network.field_shapes

    edge_components = network.edge_components.detach().cpu().to(torch.int64).tolist()
    edge_offsets = network.edge_offsets.detach().cpu().tolist()
    weights = network.weights.detach().cpu()
    segment_offsets = network.segment_offsets.detach().cpu().tolist()
    num_segments = len(segment_offsets) - 1
    num_nodes = int(network.node_capacitance.numel())

    target_components = network.target_components.detach().cpu().to(torch.int64).tolist()
    target_offsets = network.target_offsets.detach().cpu().tolist()
    edge_group_offsets = network.edge_group_offsets.detach().cpu().tolist()
    contribution_segments = network.contribution_segments.detach().cpu()
    contribution_weights = network.contribution_weights.detach().cpu()

    def _owner(component: int, index: tuple[int, int, int]) -> int:
        return partition_plan.owner_of_component_x(_COMPONENTS[component], index[0])

    # Owner selection: minimum (i, j, k, component) sampling edge.
    best_key: tuple[int, int, int, int] | None = None
    best_edge: tuple[int, tuple[int, int, int]] | None = None
    for entry in range(len(edge_components)):
        component = edge_components[entry]
        index = _decode_edge(component, int(edge_offsets[entry]), field_shapes)
        key = (index[0], index[1], index[2], component)
        if best_key is None or key < best_key:
            best_key = key
            best_edge = (component, index)
    if best_edge is None:
        raise ValueError("Distributed thin-wire network has no sampling edges.")
    owner_rank = _owner(*best_edge)
    owner_reference_edge = best_key

    # Sampling plans: iterate segments in stored order so each shard's per-segment
    # partial EMF sums the same entries in the same order the single-GPU kernel
    # would, keeping the reduction bit-stable up to the cross-shard partial split.
    shard_sample: dict[int, dict[str, list]] = {}
    for segment in range(num_segments):
        begin = int(segment_offsets[segment])
        end = int(segment_offsets[segment + 1])
        per_shard_counts: dict[int, int] = {}
        for entry in range(begin, end):
            component = edge_components[entry]
            index = _decode_edge(component, int(edge_offsets[entry]), field_shapes)
            rank = _owner(component, index)
            local_offset = _local_flat(layouts[rank].component(_COMPONENTS[component]), index)
            bucket = shard_sample.setdefault(
                rank,
                {"segment_counts": [0] * num_segments, "components": [], "offsets": [], "weights": []},
            )
            bucket["components"].append(component)
            bucket["offsets"].append(local_offset)
            bucket["weights"].append(weights[entry])
            per_shard_counts[rank] = per_shard_counts.get(rank, 0) + 1
        for rank, count in per_shard_counts.items():
            shard_sample[rank]["segment_counts"][segment] = count

    sample_plans = []
    for rank in sorted(shard_sample):
        bucket = shard_sample[rank]
        counts = torch.tensor(bucket["segment_counts"], dtype=torch.int64)
        offsets = torch.zeros(num_segments + 1, dtype=torch.int64)
        offsets[1:] = torch.cumsum(counts, dim=0)
        sample_plans.append(
            _ShardSamplePlan(
                rank=rank,
                segment_offsets=offsets,
                edge_components=torch.tensor(bucket["components"], dtype=torch.int32),
                edge_offsets=torch.tensor(bucket["offsets"], dtype=torch.int64),
                weights=torch.stack(bucket["weights"]) if bucket["weights"] else weights.new_zeros((0,)),
            )
        )

    # Deposition plans: every deposit target is owned by exactly one shard, so the
    # per-edge contribution order (and therefore the deposited value) is identical
    # to the single-GPU kernel for that edge.
    shard_deposit: dict[int, dict[str, list]] = {}
    num_targets = len(target_components)
    for target in range(num_targets):
        component = target_components[target]
        index = _decode_edge(component, int(target_offsets[target]), field_shapes)
        rank = _owner(component, index)
        local_offset = _local_flat(layouts[rank].component(_COMPONENTS[component]), index)
        begin = int(edge_group_offsets[target])
        end = int(edge_group_offsets[target + 1])
        bucket = shard_deposit.setdefault(
            rank,
            {
                "group_counts": [],
                "components": [],
                "offsets": [],
                "segments": [],
                "weights": [],
            },
        )
        bucket["group_counts"].append(end - begin)
        bucket["components"].append(component)
        bucket["offsets"].append(local_offset)
        bucket["segments"].append(contribution_segments[begin:end])
        bucket["weights"].append(contribution_weights[begin:end])

    deposit_plans = []
    for rank in sorted(shard_deposit):
        bucket = shard_deposit[rank]
        group_counts = torch.tensor(bucket["group_counts"], dtype=torch.int64)
        group_offsets = torch.zeros(group_counts.numel() + 1, dtype=torch.int64)
        group_offsets[1:] = torch.cumsum(group_counts, dim=0)
        deposit_plans.append(
            _ShardDepositPlan(
                rank=rank,
                edge_group_offsets=group_offsets,
                target_components=torch.tensor(bucket["components"], dtype=torch.int32),
                target_offsets=torch.tensor(bucket["offsets"], dtype=torch.int64),
                contribution_segments=torch.cat(bucket["segments"]).to(torch.int64),
                contribution_weights=torch.cat(bucket["weights"]),
            )
        )

    return DistributedWirePlan(
        owner_rank=owner_rank,
        num_segments=num_segments,
        num_nodes=num_nodes,
        owner_reference_edge=owner_reference_edge,
        sample_plans=tuple(sample_plans),
        deposit_plans=tuple(deposit_plans),
    )


def move_network(network: CompiledWireNetwork, device: torch.device) -> CompiledWireNetwork:
    """Move every tensor field of a compiled wire network to ``device``."""

    updates = {}
    for field_info in fields(network):
        value = getattr(network, field_info.name)
        if isinstance(value, torch.Tensor):
            updates[field_info.name] = value.to(device=device)
    return replace(network, **updates)


def move_wire_monitor(monitor: CompiledWireMonitor, device: torch.device) -> CompiledWireMonitor:
    return replace(
        monitor,
        node_indices=monitor.node_indices.to(device=device),
        segment_indices=monitor.segment_indices.to(device=device),
    )


def _move_wire_data(data: WireData, device: torch.device) -> WireData:
    def _move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device=device)
        if isinstance(value, dict):
            return {key: _move(item) for key, item in value.items()}
        return value

    return replace(
        data,
        frequencies=data.frequencies.to(device=device),
        current=None if data.current is None else data.current.to(device=device),
        charge=None if data.charge is None else data.charge.to(device=device),
        ohmic_loss=None if data.ohmic_loss is None else data.ohmic_loss.to(device=device),
        metadata={key: _move(value) for key, value in dict(data.metadata).items()},
    )


@dataclass
class _ShardSampleRuntime:
    rank: int
    shard: Any
    segment_offsets: torch.Tensor
    edge_components: torch.Tensor
    edge_offsets: torch.Tensor
    weights: torch.Tensor
    emf_buffer: torch.Tensor
    sample_done: torch.cuda.Event
    emf_read: torch.cuda.Event


@dataclass
class _ShardDepositRuntime:
    rank: int
    shard: Any
    edge_group_offsets: torch.Tensor
    target_components: torch.Tensor
    target_offsets: torch.Tensor
    contribution_segments: torch.Tensor
    contribution_scales: torch.Tensor
    current_buffer: torch.Tensor | None
    current_read: torch.cuda.Event | None


class DistributedWireRuntime:
    """GPU-resident distributed forward for one compiled thin-wire network.

    Sampling and deposition run on the shard that owns each Yee edge; per-segment
    EMF partials are gathered to the state owner with a deterministic rank-ordered
    reduction; the compressed ``I``/``q`` recurrence and every wire monitor are
    advanced only on the owner shard. Reverse/gradient parity is out of scope: a
    trainable wire under multi-GPU is rejected before this runtime is built.
    """

    def __init__(
        self,
        *,
        plan: DistributedWirePlan,
        shards,
        owner_solver,
        owner_runtime: WireRuntime,
        sample_runtimes: tuple[_ShardSampleRuntime, ...],
        deposit_runtimes: tuple[_ShardDepositRuntime, ...],
        dt: float,
    ) -> None:
        self.plan = plan
        self.shards = tuple(shards)
        self.owner_shard = self.shards[plan.owner_rank]
        self.owner_solver = owner_solver
        self.owner_runtime = owner_runtime
        self.sample_runtimes = sample_runtimes
        self.deposit_runtimes = deposit_runtimes
        self.dt = float(dt)
        self.emf_accum = owner_runtime.emf
        self.current = owner_runtime.current
        self.charge = owner_runtime.charge
        self.coefficients = owner_runtime.coefficients
        self._solve_done = torch.cuda.Event()

    @classmethod
    def prepare(
        cls,
        *,
        network: CompiledWireNetwork,
        monitors: tuple[CompiledWireMonitor, ...],
        partition_plan: FDTDPartitionPlan,
        shards,
        dt: float,
        cfl_metadata: dict[str, Any],
    ) -> "DistributedWireRuntime | None":
        if int(network.segment_count) == 0:
            return None
        plan = compile_distributed_wire_plan(network, partition_plan)
        shards = tuple(shards)
        owner_shard = shards[plan.owner_rank]
        owner_device = owner_shard.device
        dtype = owner_shard.solver.Ex.dtype

        owner_network = move_network(network, owner_device)
        owner_monitors = tuple(move_wire_monitor(monitor, owner_device) for monitor in monitors)

        def _owner_tensor(value, target_dtype):
            return value.to(device=owner_device, dtype=target_dtype).contiguous()

        coefficients = {
            "tail": _owner_tensor(network.tail, torch.int64),
            "head": _owner_tensor(network.head, torch.int64),
            "inductance": _owner_tensor(network.inductance, dtype),
            "node_capacitance": _owner_tensor(network.node_capacitance, dtype),
            "grounded": _owner_tensor(network.grounded, torch.bool),
            "node_offsets": _owner_tensor(network.node_offsets, torch.int64),
            "node_segments": _owner_tensor(network.node_segments, torch.int64),
            "node_signs": _owner_tensor(network.node_signs, torch.int32),
            "segment_offsets": _owner_tensor(network.segment_offsets, torch.int64),
            "edge_components": _owner_tensor(network.edge_components, torch.int32),
            "edge_offsets": _owner_tensor(network.edge_offsets, torch.int64),
            "weights": _owner_tensor(network.weights, dtype),
            "target_components": _owner_tensor(network.target_components, torch.int32),
            "target_offsets": _owner_tensor(network.target_offsets, torch.int64),
        }

        current = torch.zeros(plan.num_segments, device=owner_device, dtype=dtype)
        charge = torch.zeros(plan.num_nodes, device=owner_device, dtype=dtype)
        emf_accum = torch.zeros(plan.num_segments, device=owner_device, dtype=dtype)
        state_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in (current, charge, emf_accum)
        )

        owner_runtime = WireRuntime(
            network=owner_network,
            monitors=owner_monitors,
            current=current,
            charge=charge,
            emf=emf_accum,
            coefficients=coefficients,
            sample_plan=_component_plan(
                coefficients["edge_components"], coefficients["edge_offsets"]
            ),
            target_plan=_component_plan(
                coefficients["target_components"], coefficients["target_offsets"]
            ),
            sample_segments=_group_indices(coefficients["segment_offsets"]),
            monitor_state=[],
            cfl_limit=float(cfl_metadata["cfl_limit"]),
            wire_cfl_limit=float(cfl_metadata["wire_cfl_limit"]),
            maxwell_cfl_limit=float(cfl_metadata["maxwell_cfl_limit"]),
            dt_adjusted=bool(cfl_metadata["dt_adjusted"]),
            state_bytes=state_bytes,
        )
        owner_shard.solver._wire_runtime = owner_runtime

        sample_runtimes = []
        for sample_plan in plan.sample_plans:
            shard = shards[sample_plan.rank]
            device = shard.device
            sample_runtimes.append(
                _ShardSampleRuntime(
                    rank=sample_plan.rank,
                    shard=shard,
                    segment_offsets=sample_plan.segment_offsets.to(device=device),
                    edge_components=sample_plan.edge_components.to(device=device),
                    edge_offsets=sample_plan.edge_offsets.to(device=device),
                    weights=sample_plan.weights.to(device=device, dtype=dtype),
                    emf_buffer=torch.zeros(plan.num_segments, device=device, dtype=dtype),
                    sample_done=torch.cuda.Event(),
                    emf_read=torch.cuda.Event(),
                )
            )

        deposit_runtimes = []
        for deposit_plan in plan.deposit_plans:
            shard = shards[deposit_plan.rank]
            device = shard.device
            components = deposit_plan.target_components.to(device=device)
            offsets = deposit_plan.target_offsets.to(device=device)
            group_offsets = deposit_plan.edge_group_offsets.to(device=device)
            contribution_weights = deposit_plan.contribution_weights.to(device=device, dtype=dtype)
            with torch.cuda.device(device):
                masses = _target_masses(shard.solver, components, offsets)
                edge_groups = _group_indices(group_offsets)
                contribution_scales = (
                    float(dt) * contribution_weights / masses.index_select(0, edge_groups)
                ).contiguous()
            remote = deposit_plan.rank != plan.owner_rank
            deposit_runtimes.append(
                _ShardDepositRuntime(
                    rank=deposit_plan.rank,
                    shard=shard,
                    edge_group_offsets=group_offsets,
                    target_components=components,
                    target_offsets=offsets,
                    contribution_segments=deposit_plan.contribution_segments.to(device=device),
                    contribution_scales=contribution_scales,
                    current_buffer=(
                        torch.zeros(plan.num_segments, device=device, dtype=dtype)
                        if remote
                        else None
                    ),
                    current_read=torch.cuda.Event() if remote else None,
                )
            )

        return cls(
            plan=plan,
            shards=shards,
            owner_solver=owner_shard.solver,
            owner_runtime=owner_runtime,
            sample_runtimes=tuple(sample_runtimes),
            deposit_runtimes=tuple(deposit_runtimes),
            dt=dt,
        )

    def apply_sample_and_update(self) -> None:
        """Sample the pre-update E field, reduce EMF to the owner, advance I/q."""

        for sample in self.sample_runtimes:
            shard = sample.shard
            with torch.cuda.device(shard.device), torch.cuda.stream(shard.compute_stream):
                if sample.rank != self.plan.owner_rank:
                    # Do not overwrite the send buffer before the owner read it.
                    shard.compute_stream.wait_event(sample.emf_read)
                shard.solver.fdtd_module.sampleWireEmf3D(
                    Ex=shard.solver.Ex,
                    Ey=shard.solver.Ey,
                    Ez=shard.solver.Ez,
                    segmentOffsets=sample.segment_offsets,
                    edgeComponents=sample.edge_components,
                    edgeOffsets=sample.edge_offsets,
                    weights=sample.weights,
                    emf=sample.emf_buffer,
                ).launchRaw()
                sample.sample_done.record(shard.compute_stream)

        owner = self.owner_shard
        with torch.cuda.device(owner.device), torch.cuda.stream(owner.compute_stream):
            for deposit in self.deposit_runtimes:
                if deposit.current_read is not None:
                    # Protect current from being overwritten before last step's read.
                    owner.compute_stream.wait_event(deposit.current_read)
            self.emf_accum.zero_()
            for sample in self.sample_runtimes:
                if sample.rank == self.plan.owner_rank:
                    self.emf_accum.add_(sample.emf_buffer)
                else:
                    owner.compute_stream.wait_event(sample.sample_done)
                    self.emf_accum.add_(sample.emf_buffer.to(owner.device, non_blocking=True))
                    sample.emf_read.record(owner.compute_stream)
            coeff = self.coefficients
            owner.solver.fdtd_module.updateWireState1D(
                emf=self.emf_accum,
                tail=coeff["tail"],
                head=coeff["head"],
                inductance=coeff["inductance"],
                nodeCapacitance=coeff["node_capacitance"],
                grounded=coeff["grounded"],
                nodeOffsets=coeff["node_offsets"],
                nodeSegments=coeff["node_segments"],
                nodeSigns=coeff["node_signs"],
                dt=float(self.dt),
                current=self.current,
                charge=self.charge,
            ).launchRaw()
            self._solve_done.record(owner.compute_stream)

    def apply_deposit(self) -> None:
        """Broadcast the advanced current and deposit onto every owned E edge."""

        for deposit in self.deposit_runtimes:
            shard = deposit.shard
            with torch.cuda.device(shard.device), torch.cuda.stream(shard.compute_stream):
                if deposit.rank == self.plan.owner_rank:
                    current = self.current
                else:
                    shard.compute_stream.wait_event(self._solve_done)
                    deposit.current_buffer.copy_(self.current, non_blocking=True)
                    deposit.current_read.record(shard.compute_stream)
                    current = deposit.current_buffer
                shard.solver.fdtd_module.depositWireCurrent3D(
                    Ex=shard.solver.Ex,
                    Ey=shard.solver.Ey,
                    Ez=shard.solver.Ez,
                    edgeGroupOffsets=deposit.edge_group_offsets,
                    targetComponents=deposit.target_components,
                    targetOffsets=deposit.target_offsets,
                    contributionSegments=deposit.contribution_segments,
                    contributionScales=deposit.contribution_scales,
                    current=current,
                ).launchRaw()

    def prepare_outputs(self, *, time_steps: int, window_type: str) -> None:
        owner = self.owner_shard
        with torch.cuda.device(owner.device), torch.cuda.stream(owner.compute_stream):
            prepare_wire_monitors(self.owner_solver, int(time_steps), window_type)

    def accumulate_monitors(self, n: int) -> None:
        owner = self.owner_shard
        with torch.cuda.device(owner.device), torch.cuda.stream(owner.compute_stream):
            accumulate_wire_monitors(self.owner_solver, n)

    def complete_normalization(self, time_steps: int) -> None:
        complete_wire_monitor_normalization(self.owner_solver, int(time_steps))

    def finalize(self, result_device: torch.device) -> dict[str, WireData]:
        owner = self.owner_shard
        with torch.cuda.device(owner.device):
            data = finalize_wire_data(self.owner_solver)
        output = {}
        for name, value in data.items():
            metadata = dict(value.metadata)
            metadata["distributed_wire_owner_rank"] = self.plan.owner_rank
            output[name] = _move_wire_data(replace(value, metadata=metadata), result_device)
        return output

    def checkpoint_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "wire_current": self.current.detach().clone(),
            "wire_charge": self.charge.detach().clone(),
        }

    @property
    def state_bytes(self) -> int:
        return int(self.owner_runtime.state_bytes)

    def stats(self, *, steps_run: int) -> dict[str, Any]:
        element_size = self.owner_shard.solver.Ex.element_size()
        remote_samples = sum(
            1 for sample in self.sample_runtimes if sample.rank != self.plan.owner_rank
        )
        remote_deposits = sum(
            1 for deposit in self.deposit_runtimes if deposit.rank != self.plan.owner_rank
        )
        emf_scalars = sum(
            int((sample.segment_offsets[1:] - sample.segment_offsets[:-1] > 0).sum())
            for sample in self.sample_runtimes
            if sample.rank != self.plan.owner_rank
        )
        current_scalars = self.plan.num_segments * remote_deposits
        scalars_per_step = emf_scalars + current_scalars
        return {
            "enabled": True,
            "owner_rank": self.plan.owner_rank,
            "owner_device": str(self.owner_shard.device),
            "owner_reference_edge": self.plan.owner_reference_edge,
            "segment_count": self.plan.num_segments,
            "node_count": self.plan.num_nodes,
            "cross_shard_segment_count": self.plan.cross_shard_segment_count,
            "sample_shard_count": len(self.sample_runtimes),
            "deposit_shard_count": len(self.deposit_runtimes),
            "remote_sample_shard_count": remote_samples,
            "remote_deposit_shard_count": remote_deposits,
            "reduction_order": "sorted_by_rank_deterministic",
            "state_bytes": self.state_bytes,
            "communication_bytes_per_step": scalars_per_step * element_size,
            "communication_bytes_total": scalars_per_step * element_size * int(steps_run),
        }


__all__ = [
    "DistributedWirePlan",
    "DistributedWireRuntime",
    "compile_distributed_wire_plan",
    "move_network",
    "move_wire_monitor",
]
