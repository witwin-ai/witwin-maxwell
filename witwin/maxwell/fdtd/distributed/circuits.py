from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

from ...compiler.ports import CompiledPortGeometry
from ...fdtd_parallel import FDTDPartitionPlan, FDTDShardLayout
from ...ports import TerminalPort, _resolve_terminal_port
from ..circuits import (
    finalize_circuit_data,
    prepare_circuit_runtimes,
    prepare_circuit_time_series,
)
from ..lumped import FieldPortCoupling, prepare_field_port_coupling
from ..ports import (
    PreparedPortRuntime,
    _edge_control_volume,
    _open_declared_pec_terminal_edges,
    _validate_local_update_coefficient,
    _validate_supported_field_coupling,
    accumulate_port_observers,
    finalize_port_data,
    prepare_port_spectral_accumulators,
)
from ._shared import local_port_geometry, move_port_data


@dataclass(frozen=True)
class DistributedBoundPortPlan:
    """One circuit-bound port with a single owning Yee shard."""

    port_name: str
    owner_rank: int
    voltage_component: str
    minimum_global_index: tuple[int, int, int]
    local_voltage_indices: tuple[tuple[int, int, int], ...]
    geometry: CompiledPortGeometry


@dataclass(frozen=True)
class DistributedCircuitPlan:
    """Immutable ownership plan for one distributed circuit graph."""

    circuit_name: str
    owner_rank: int
    owner_reference_port: str
    ports: tuple[DistributedBoundPortPlan, ...]

    @property
    def port_owners(self) -> dict[str, int]:
        return {port.port_name: port.owner_rank for port in self.ports}


def _owner_for_index(
    layouts: tuple[FDTDShardLayout, ...],
    component: str,
    index: tuple[int, int, int],
) -> int:
    owners = tuple(
        layout.rank for layout in layouts if layout.component(component).owns(index)
    )
    if len(owners) != 1:
        raise RuntimeError(
            f"Distributed {component} index {index!r} must have exactly one owner; "
            f"found ranks {owners}."
        )
    return owners[0]


def compile_distributed_circuit_plan(
    scene,
    partition_plan: FDTDPartitionPlan,
    *,
    geometries: tuple[CompiledPortGeometry, ...] | None = None,
) -> DistributedCircuitPlan | None:
    """Resolve deterministic circuit/port owners without touching CUDA runtime state.

    A bound port must remain wholly within one x slab.  This keeps the hot-path
    communication contract at exactly one voltage scalar and one current scalar
    per remote bound port, independent of the port's number of Yee edges.
    """

    circuits = tuple(getattr(scene, "circuits", ()))
    if not circuits:
        return None
    if len(circuits) != 1:
        raise ValueError(
            "Distributed circuit-coupled FDTD currently supports exactly one circuit."
        )
    circuit = circuits[0]
    if not circuit.bindings:
        raise ValueError("A distributed circuit must bind at least one FDTD port.")

    compiled = tuple(
        scene.compile_ports(device="cpu") if geometries is None else geometries
    )
    geometry_by_name = {geometry.port_name: geometry for geometry in compiled}
    binding_names = tuple(binding.port_name for binding in circuit.bindings)
    if len(set(binding_names)) != len(binding_names):
        raise ValueError(
            f"Circuit {circuit.name!r} binds the same FDTD port more than once."
        )
    missing = tuple(name for name in binding_names if name not in geometry_by_name)
    if missing:
        raise ValueError(
            f"Circuit {circuit.name!r} references ports without lumped/terminal geometry: "
            f"{missing}."
        )

    layouts = partition_plan.shard_layouts
    occupied_edges: dict[tuple[str, tuple[int, int, int]], str] = {}
    port_plans = []
    for binding in circuit.bindings:
        geometry = geometry_by_name[binding.port_name]
        global_indices = tuple(
            tuple(int(value) for value in index)
            for index in geometry.voltage_indices.detach().cpu().tolist()
        )
        if not global_indices:
            raise ValueError(
                f"Circuit-bound port {binding.port_name!r} has no voltage edges."
            )
        owners = tuple(
            _owner_for_index(layouts, geometry.voltage_component, index)
            for index in global_indices
        )
        unique_owners = tuple(sorted(set(owners)))
        if len(unique_owners) != 1:
            raise ValueError(
                f"Circuit-bound port {binding.port_name!r} crosses the multi-GPU x "
                f"partition and spans ranks {unique_owners}. Move the port off the split "
                "or change the partition; one port must have one scalar owner so circuit "
                "communication remains O(bound ports)."
            )
        owner_rank = unique_owners[0]
        component_layout = layouts[owner_rank].component(geometry.voltage_component)
        local_indices = tuple(
            component_layout.global_to_local(index) for index in global_indices
        )
        for global_index in global_indices:
            key = (geometry.voltage_component, global_index)
            previous = occupied_edges.get(key)
            if previous is not None:
                raise ValueError(
                    f"Circuit-bound ports {previous!r} and {binding.port_name!r} overlap "
                    f"on {geometry.voltage_component} edge {global_index}."
                )
            occupied_edges[key] = binding.port_name
        port_plans.append(
            DistributedBoundPortPlan(
                port_name=binding.port_name,
                owner_rank=owner_rank,
                voltage_component=geometry.voltage_component,
                minimum_global_index=min(global_indices),
                local_voltage_indices=local_indices,
                geometry=geometry,
            )
        )

    component_order = {"Ex": 0, "Ey": 1, "Ez": 2}
    minimum_port = min(
        port_plans,
        key=lambda port: (
            *port.minimum_global_index,
            component_order[port.voltage_component],
        ),
    )
    return DistributedCircuitPlan(
        circuit_name=circuit.name,
        owner_rank=minimum_port.owner_rank,
        owner_reference_port=minimum_port.port_name,
        ports=tuple(port_plans),
    )


def _owner_proxy_field(
    local_field: FieldPortCoupling,
    *,
    owner_device: torch.device,
) -> FieldPortCoupling:
    dtype = local_field.field_dtype
    empty_indices = torch.empty((0,), device=owner_device, dtype=torch.int64)
    empty_values = torch.empty((0,), device=owner_device, dtype=dtype)
    zeros = tuple(torch.zeros((), device=owner_device, dtype=dtype) for _ in range(7))
    return FieldPortCoupling(
        port_name=local_field.port_name,
        field_shape=local_field.field_shape,
        field_dtype=dtype,
        linear_indices=empty_indices,
        voltage_weights=empty_values,
        injection=empty_values.clone(),
        dt=local_field.dt.to(device=owner_device),
        coupling_impedance=local_field.coupling_impedance.to(device=owner_device),
        discrete_port_impedance=local_field.discrete_port_impedance.to(
            device=owner_device
        ),
        edge_buffer=empty_values.clone(),
        correction_buffer=empty_values.clone(),
        last_voltage_before=zeros[0],
        coupling_voltage=zeros[6],
        last_voltage=zeros[1],
        last_voltage_after=zeros[2],
        last_current=zeros[3],
        last_port_work=zeros[4],
        last_field_energy_change=zeros[5],
    )


@dataclass
class _OwnerCurrentReuseState:
    """Host-side state machine for one asynchronously read owner scalar."""

    copy_pending: bool = False

    def begin_owner_write(self) -> bool:
        needs_acknowledgement = self.copy_pending
        self.copy_pending = False
        return needs_acknowledgement

    def finish_remote_read_schedule(self) -> None:
        if self.copy_pending:
            raise RuntimeError(
                "Owner current scalar cannot be reused before its remote copy acknowledgement."
            )
        self.copy_pending = True


@dataclass
class _DistributedPortRuntime:
    plan: DistributedBoundPortPlan
    shard: Any
    local_geometry: CompiledPortGeometry
    local_field: FieldPortCoupling
    owner_field: FieldPortCoupling
    owner_voltage: torch.Tensor
    local_current: torch.Tensor
    voltage_ready: torch.cuda.Event | None
    voltage_received: torch.cuda.Event | None
    current_copied: torch.cuda.Event | None
    current_received: torch.cuda.Event | None
    owner_current_reuse: _OwnerCurrentReuseState


class DistributedCircuitRuntime:
    """GPU-resident scalar gather/owner-solve/scatter coordinator."""

    def __init__(
        self,
        *,
        plan: DistributedCircuitPlan,
        circuit,
        shards,
        port_runtimes: tuple[_DistributedPortRuntime, ...],
        owner_port_runtimes: tuple[PreparedPortRuntime, ...],
        circuit_runtime,
    ) -> None:
        self.plan = plan
        self.circuit = circuit
        self.shards = tuple(shards)
        self.port_runtimes = port_runtimes
        self.owner_shard = self.shards[plan.owner_rank]
        self.owner_solver = self.owner_shard.solver
        self.owner_port_runtimes = owner_port_runtimes
        self.circuit_runtime = circuit_runtime
        self.solve_ready = torch.cuda.Event()

    @classmethod
    def prepare(
        cls,
        *,
        prepared_scene,
        partition_plan: FDTDPartitionPlan,
        shards,
        frequency: float,
    ) -> "DistributedCircuitRuntime | None":
        plan = compile_distributed_circuit_plan(prepared_scene, partition_plan)
        if plan is None:
            return None
        circuits = tuple(prepared_scene.circuits)
        circuit = circuits[0]
        owner_shard = tuple(shards)[plan.owner_rank]
        owner_device = owner_shard.device
        ports_by_name = {}
        for port in prepared_scene.ports:
            resolved = (
                _resolve_terminal_port(
                    port,
                    prepared_scene.structures,
                    prepared_scene.domain.bounds,
                )
                if isinstance(port, TerminalPort)
                else port
            )
            ports_by_name[resolved.name] = resolved

        for shard in shards:
            _validate_supported_field_coupling(shard.solver)

        distributed_ports = []
        proxy_runtimes = []
        proxy_fields = {}
        for port_plan in plan.ports:
            shard = tuple(shards)[port_plan.owner_rank]
            port = ports_by_name[port_plan.port_name]
            local_geometry = local_port_geometry(port_plan, device=shard.device)
            with torch.cuda.device(shard.device):
                _open_declared_pec_terminal_edges(
                    shard.solver,
                    local_geometry,
                    port,
                )
                local_field = prepare_field_port_coupling(
                    local_geometry,
                    dt=shard.solver.dt,
                    eps_edge=getattr(
                        shard.solver,
                        f"eps_{local_geometry.voltage_component}",
                    ),
                    yee_control_volume=_edge_control_volume(
                        shard.solver,
                        local_geometry.voltage_component,
                    ),
                )
                _validate_local_update_coefficient(
                    shard.solver,
                    local_field,
                    local_geometry.voltage_component,
                )

            with torch.cuda.device(owner_device):
                owner_field = _owner_proxy_field(
                    local_field,
                    owner_device=owner_device,
                )
                owner_voltage = torch.empty(
                    (), device=owner_device, dtype=owner_shard.solver.Ex.dtype
                )
                proxy_runtime = PreparedPortRuntime(
                    port=port,
                    geometry=port_plan.geometry,
                    frequencies=torch.as_tensor(
                        (float(frequency),),
                        device=owner_device,
                        dtype=torch.float64,
                    ),
                    field_name=port_plan.voltage_component,
                    yee_control_volume=None,
                    lumped=None,
                    wire_provider=None,
                    excitation=None,
                    source_kind=None,
                    source_frequency=0.0,
                    source_fwidth=0.0,
                    source_phase=0.0,
                    source_delay=0.0,
                    source_amplitude=torch.zeros(
                        (),
                        device=owner_device,
                        dtype=(
                            torch.complex64
                            if owner_shard.solver.Ex.dtype == torch.float32
                            else torch.complex128
                        ),
                    ),
                    drive_buffer=torch.zeros(
                        (), device=owner_device, dtype=owner_shard.solver.Ex.dtype
                    ),
                    electric_time=torch.as_tensor(
                        owner_shard.solver.dt,
                        device=owner_device,
                        dtype=owner_shard.solver.Ex.dtype,
                    ),
                    magnetic_time=torch.as_tensor(
                        0.5 * owner_shard.solver.dt,
                        device=owner_device,
                        dtype=owner_shard.solver.Ex.dtype,
                    ),
                )
            remote = port_plan.owner_rank != plan.owner_rank
            distributed_ports.append(
                _DistributedPortRuntime(
                    plan=port_plan,
                    shard=shard,
                    local_geometry=local_geometry,
                    local_field=local_field,
                    owner_field=owner_field,
                    owner_voltage=owner_voltage,
                    local_current=torch.empty(
                        (), device=shard.device, dtype=shard.solver.Ex.dtype
                    ),
                    voltage_ready=torch.cuda.Event() if remote else None,
                    voltage_received=torch.cuda.Event() if remote else None,
                    current_copied=torch.cuda.Event() if remote else None,
                    current_received=torch.cuda.Event() if remote else None,
                    owner_current_reuse=_OwnerCurrentReuseState(),
                )
            )
            proxy_runtimes.append(proxy_runtime)
            proxy_fields[port_plan.port_name] = owner_field

        owner_shard.solver._port_runtimes = tuple(proxy_runtimes)
        circuit_runtimes = prepare_circuit_runtimes(
            owner_shard.solver,
            owner_shard.solver._port_runtimes,
            circuits=(circuit,),
            field_couplings=proxy_fields,
        )
        if len(circuit_runtimes) != 1:
            raise RuntimeError("Distributed circuit preparation did not create one owner runtime.")
        return cls(
            plan=plan,
            circuit=circuit,
            shards=shards,
            port_runtimes=tuple(distributed_ports),
            owner_port_runtimes=tuple(proxy_runtimes),
            circuit_runtime=circuit_runtimes[0],
        )

    @property
    def remote_port_count(self) -> int:
        return sum(
            runtime.plan.owner_rank != self.plan.owner_rank
            for runtime in self.port_runtimes
        )

    @property
    def same_shard_port_count(self) -> int:
        return len(self.port_runtimes) - self.remote_port_count

    @property
    def communication_bytes_per_step(self) -> int:
        return 2 * self.remote_port_count * self.owner_shard.solver.Ex.element_size()

    def prepare_outputs(
        self,
        *,
        time_steps: int,
        frequencies,
        window_type: str,
    ) -> None:
        if isinstance(frequencies, (float, int)):
            frequencies = (float(frequencies),)
        frequency_tensor = torch.as_tensor(
            tuple(float(value) for value in frequencies),
            device=self.owner_shard.device,
            dtype=torch.float64,
        )
        for runtime in self.owner_port_runtimes:
            runtime.frequencies = frequency_tensor
        with torch.cuda.device(self.owner_shard.device), torch.cuda.stream(
            self.owner_shard.compute_stream
        ):
            prepare_port_spectral_accumulators(
                self.owner_solver,
                int(time_steps),
                window_type,
            )
            prepare_circuit_time_series(self.owner_solver, int(time_steps))

    @staticmethod
    def _sample_voltage(runtime: _DistributedPortRuntime) -> None:
        field_tensor = getattr(runtime.shard.solver, runtime.plan.voltage_component)
        field = runtime.local_field
        torch.index_select(
            field_tensor.reshape(-1),
            0,
            field.linear_indices,
            out=field.edge_buffer,
        )
        torch.mul(field.edge_buffer, field.voltage_weights, out=field.edge_buffer)
        torch.sum(field.edge_buffer, dim=0, out=field.last_voltage_before)

    @staticmethod
    def _apply_current(
        runtime: _DistributedPortRuntime,
        current: torch.Tensor,
    ) -> None:
        field = runtime.local_field
        field_tensor = getattr(runtime.shard.solver, runtime.plan.voltage_component)
        torch.mul(field.injection, current, out=field.correction_buffer)
        field_tensor.reshape(-1).index_add_(
            0,
            field.linear_indices,
            field.correction_buffer,
            alpha=-1.0,
        )

    def apply(self) -> None:
        for runtime in self.port_runtimes:
            shard = runtime.shard
            with torch.cuda.device(shard.device), torch.cuda.stream(shard.compute_stream):
                self._sample_voltage(runtime)
                if runtime.voltage_ready is not None:
                    runtime.voltage_ready.record(shard.compute_stream)

        owner = self.owner_shard
        for runtime in self.port_runtimes:
            if runtime.voltage_ready is None:
                continue
            with torch.cuda.device(owner.device), torch.cuda.stream(
                owner.communication_stream
            ):
                owner.communication_stream.wait_event(runtime.voltage_ready)
                runtime.owner_voltage.copy_(
                    runtime.local_field.last_voltage_before,
                    non_blocking=True,
                )
                runtime.voltage_received.record(owner.communication_stream)

        with torch.cuda.device(owner.device), torch.cuda.stream(owner.compute_stream):
            free_voltages = []
            for runtime in self.port_runtimes:
                if runtime.owner_current_reuse.begin_owner_write():
                    owner.compute_stream.wait_event(runtime.current_copied)
                if runtime.voltage_received is None:
                    free_voltages.append(runtime.local_field.last_voltage_before)
                else:
                    owner.compute_stream.wait_event(runtime.voltage_received)
                    free_voltages.append(runtime.owner_voltage)
            currents = self.circuit_runtime.apply_external(
                tuple(free_voltages),
                apply_field_currents=False,
            )
            accumulate_port_observers(self.owner_solver)
            for runtime, current in zip(self.port_runtimes, currents):
                if runtime.plan.owner_rank == self.plan.owner_rank:
                    self._apply_current(runtime, current)
            self.solve_ready.record(owner.compute_stream)

        for runtime, current in zip(self.port_runtimes, currents):
            if runtime.plan.owner_rank == self.plan.owner_rank:
                continue
            shard = runtime.shard
            with torch.cuda.device(shard.device), torch.cuda.stream(
                shard.communication_stream
            ):
                shard.communication_stream.wait_event(self.solve_ready)
                runtime.local_current.copy_(current, non_blocking=True)
                runtime.current_copied.record(shard.communication_stream)
                self._apply_current(runtime, runtime.local_current)
                runtime.current_received.record(shard.communication_stream)
                runtime.owner_current_reuse.finish_remote_read_schedule()
            with torch.cuda.device(shard.device), torch.cuda.stream(shard.compute_stream):
                shard.compute_stream.wait_event(runtime.current_received)

    def checkpoint_tensors(self) -> dict[str, torch.Tensor]:
        return self.circuit_runtime.checkpoint_tensors()

    def finalize(self, result_device: torch.device):
        ports = finalize_port_data(self.owner_solver)
        port_output = {}
        for name, data in ports.items():
            metadata = dict(data.metadata)
            metadata.update(
                {
                    "distributed_circuit_owner_rank": self.plan.owner_rank,
                    "distributed_port_owner_rank": self.plan.port_owners[name],
                }
            )
            port_output[name] = move_port_data(
                data,
                result_device,
                metadata=metadata,
            )

        circuits = finalize_circuit_data(self.owner_solver)
        circuit_output = {}
        for name, data in circuits.items():
            moved = data.to(result_device)
            circuit_output[name] = replace(
                moved,
                diagnostics={
                    **moved.diagnostics,
                    "distributed_circuit_owner_rank": self.plan.owner_rank,
                    "distributed_port_owner_ranks": self.plan.port_owners,
                },
            )
        return port_output, circuit_output

    def stats(self, *, steps_run: int) -> dict[str, Any]:
        bytes_per_step = self.communication_bytes_per_step
        return {
            "enabled": True,
            "circuit_name": self.plan.circuit_name,
            "circuit_owner_rank": self.plan.owner_rank,
            "circuit_owner_device": str(self.owner_shard.device),
            "circuit_owner_reference_port": self.plan.owner_reference_port,
            "port_owner_ranks": self.plan.port_owners,
            "bound_port_count": len(self.port_runtimes),
            "remote_port_count": self.remote_port_count,
            "same_shard_fast_path_count": self.same_shard_port_count,
            "scalar_transfers_per_step": 2 * self.remote_port_count,
            "owner_copy_acknowledgements_per_step": self.remote_port_count,
            "communication_bytes_per_step": bytes_per_step,
            "communication_bytes_total": bytes_per_step * int(steps_run),
            "communication_order": "O(bound_ports)",
            "checkpoint_owner_rank": self.plan.owner_rank,
            "checkpoint_scope": "circuit_state_only",
        }


__all__ = [
    "DistributedBoundPortPlan",
    "DistributedCircuitPlan",
    "DistributedCircuitRuntime",
    "compile_distributed_circuit_plan",
]
