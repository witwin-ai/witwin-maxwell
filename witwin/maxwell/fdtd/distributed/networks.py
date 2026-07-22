from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

import torch

from ...compiler.ports import CompiledPortGeometry
from ...fdtd_parallel import FDTDPartitionPlan, FDTDShardLayout
from ...network import EmbeddedNetworkData
from ...ports import TerminalPort, _resolve_terminal_port
from ..lumped import LumpedRuntime, prepare_lumped_runtime
from ..networks import (
    advance_network_external,
    finalize_embedded_networks,
    prepare_network_runtimes,
)
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
from .circuits import _OwnerCurrentReuseState, _owner_for_index


@dataclass(frozen=True)
class DistributedNetworkPortPlan:
    """One network-connected port with a single owning Yee shard."""

    port_name: str
    owner_rank: int
    voltage_component: str
    minimum_global_index: tuple[int, int, int]
    local_voltage_indices: tuple[tuple[int, int, int], ...]
    geometry: CompiledPortGeometry


@dataclass(frozen=True)
class DistributedNetworkPlan:
    """Immutable ownership plan for one distributed embedded network."""

    network_name: str
    owner_rank: int
    owner_reference_port: str
    ports: tuple[DistributedNetworkPortPlan, ...]

    @property
    def port_owners(self) -> dict[str, int]:
        return {port.port_name: port.owner_rank for port in self.ports}


def compile_distributed_network_plan(
    scene,
    partition_plan: FDTDPartitionPlan,
    *,
    geometries: tuple[CompiledPortGeometry, ...] | None = None,
) -> DistributedNetworkPlan | None:
    """Resolve deterministic network/port owners without touching CUDA state.

    Each connected terminal must remain wholly within one x slab. This bounds
    the hot-path communication contract at exactly one voltage scalar and one
    current scalar per remote connected port, independent of the port's number
    of Yee edges, so network communication stays O(connected ports) instead of
    O(coupled edges).
    """

    networks = tuple(getattr(scene, "networks", ()))
    if not networks:
        return None
    if len(networks) != 1:
        raise ValueError(
            "Distributed network-coupled FDTD currently supports exactly one network."
        )
    block = networks[0]
    connection_names = tuple(block.connected_port_names)
    if not connection_names:
        raise ValueError("A distributed network must connect at least one FDTD port.")

    compiled = tuple(
        scene.compile_ports(device="cpu") if geometries is None else geometries
    )
    geometry_by_name = {geometry.port_name: geometry for geometry in compiled}
    if len(set(connection_names)) != len(connection_names):
        raise ValueError(
            f"Network {block.name!r} connects the same FDTD port more than once."
        )
    missing = tuple(name for name in connection_names if name not in geometry_by_name)
    if missing:
        raise ValueError(
            f"Network {block.name!r} references ports without lumped/terminal geometry: "
            f"{missing}."
        )

    layouts = partition_plan.shard_layouts
    occupied_edges: dict[tuple[str, tuple[int, int, int]], str] = {}
    port_plans = []
    # Iterate in the network's connection order so port_plans[i] aligns with the
    # owner network runtime's branch_current[i] and free_voltage[i].
    for port_name in connection_names:
        geometry = geometry_by_name[port_name]
        global_indices = tuple(
            tuple(int(value) for value in index)
            for index in geometry.voltage_indices.detach().cpu().tolist()
        )
        if not global_indices:
            raise ValueError(
                f"Network-connected port {port_name!r} has no voltage edges."
            )
        owners = tuple(
            _owner_for_index(layouts, geometry.voltage_component, index)
            for index in global_indices
        )
        unique_owners = tuple(sorted(set(owners)))
        if len(unique_owners) != 1:
            raise ValueError(
                f"Network-connected port {port_name!r} crosses the multi-GPU x "
                f"partition and spans ranks {unique_owners}. Move the port off the split "
                "or change the partition; one port must have one scalar owner so network "
                "communication remains O(connected ports)."
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
                    f"Network-connected ports {previous!r} and {port_name!r} overlap "
                    f"on {geometry.voltage_component} edge {global_index}."
                )
            occupied_edges[key] = port_name
        port_plans.append(
            DistributedNetworkPortPlan(
                port_name=port_name,
                owner_rank=owner_rank,
                voltage_component=geometry.voltage_component,
                minimum_global_index=min(global_indices),
                local_voltage_indices=local_indices,
                geometry=geometry,
            )
        )

    # Deterministic owner: the shard holding the connected port whose reference
    # point has the smallest (global index, component) key. This tiebreak is
    # identical to the circuit owner rule so a scene combining both places them
    # consistently.
    component_order = {"Ex": 0, "Ey": 1, "Ez": 2}
    minimum_port = min(
        port_plans,
        key=lambda port: (
            *port.minimum_global_index,
            component_order[port.voltage_component],
        ),
    )
    return DistributedNetworkPlan(
        network_name=block.name,
        owner_rank=minimum_port.owner_rank,
        owner_reference_port=minimum_port.port_name,
        ports=tuple(port_plans),
    )


def _owner_proxy_lumped(
    local_lumped: LumpedRuntime,
    *,
    owner_device: torch.device,
) -> LumpedRuntime:
    """Rebuild a connected port's lumped coupling on the owner with no edges.

    The owner solves the implicit network loop but never touches a field, so the
    proxy carries zero Yee edges. Its ``discrete_port_impedance`` is the real
    per-shard value moved to the owner device: the owner stacks these into the
    ``I + D Z`` loop denominator, and each connected port's impedance is computed
    from its own shard's permittivity before being gathered here.
    """

    dtype = local_lumped.field_dtype
    empty_values = torch.empty((0,), device=owner_device, dtype=dtype)
    moved: dict[str, Any] = {}
    for descriptor in fields(local_lumped):
        value = getattr(local_lumped, descriptor.name)
        moved[descriptor.name] = (
            value.to(device=owner_device) if isinstance(value, torch.Tensor) else value
        )
    moved["linear_indices"] = torch.empty((0,), device=owner_device, dtype=torch.int64)
    moved["voltage_weights"] = empty_values
    moved["injection"] = empty_values.clone()
    moved["edge_buffer"] = empty_values.clone()
    moved["correction_buffer"] = empty_values.clone()
    return LumpedRuntime(**moved)


def _move_embedded_network_data(
    data: EmbeddedNetworkData,
    device: torch.device,
    *,
    metadata,
) -> EmbeddedNetworkData:
    return replace(
        data,
        frequencies=data.frequencies.to(device=device),
        voltage=data.voltage.to(device=device),
        current=data.current.to(device=device),
        port_power=data.port_power.to(device=device),
        absorbed_power=data.absorbed_power.to(device=device),
        generated_power=data.generated_power.to(device=device),
        state_norm=data.state_norm.to(device=device),
        metadata=metadata,
    )


@dataclass
class _DistributedNetworkPortRuntime:
    plan: DistributedNetworkPortPlan
    shard: Any
    local_lumped: LumpedRuntime
    owner_voltage: torch.Tensor
    local_current: torch.Tensor
    voltage_ready: torch.cuda.Event | None
    voltage_received: torch.cuda.Event | None
    current_copied: torch.cuda.Event | None
    current_received: torch.cuda.Event | None
    owner_current_reuse: _OwnerCurrentReuseState


class DistributedNetworkRuntime:
    """GPU-resident scalar gather / owner-solve / scatter coordinator for networks."""

    def __init__(
        self,
        *,
        plan: DistributedNetworkPlan,
        block,
        shards,
        port_runtimes: tuple[_DistributedNetworkPortRuntime, ...],
        owner_port_runtimes: tuple[PreparedPortRuntime, ...],
        network_runtime,
    ) -> None:
        self.plan = plan
        self.block = block
        self.shards = tuple(shards)
        self.port_runtimes = port_runtimes
        self.owner_shard = self.shards[plan.owner_rank]
        self.owner_solver = self.owner_shard.solver
        self.owner_port_runtimes = owner_port_runtimes
        self.network_runtime = network_runtime
        self.solve_ready = torch.cuda.Event()

    @classmethod
    def prepare(
        cls,
        *,
        prepared_scene,
        partition_plan: FDTDPartitionPlan,
        shards,
        frequency: float,
        requested_frequencies: tuple[float, ...] = (),
    ) -> "DistributedNetworkRuntime | None":
        plan = compile_distributed_network_plan(prepared_scene, partition_plan)
        if plan is None:
            return None
        block = tuple(prepared_scene.networks)[0]
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

        # Compile the network once from the global scene; the owner shard's local
        # scene carries no networks, so the compiled block is injected below.
        compiled_networks = prepared_scene.compile_networks(
            dt=owner_shard.solver.dt,
            device=owner_device,
        )
        if len(compiled_networks) != 1:
            raise RuntimeError(
                "Distributed network preparation expects exactly one compiled network."
            )
        compiled = compiled_networks[0]
        if compiled.delay is not None:
            # The delayed-core path carries per-port bidirectional ring state that
            # has only been validated on the single-device runtime. Reject it on
            # the distributed path rather than run an unqualified numerical mode.
            raise ValueError(
                "Multi-GPU FDTD does not yet support embedded networks with an explicit "
                f"per-port delay; network {compiled.name!r} declares delay_seconds."
            )

        distributed_ports = []
        proxy_runtimes = []
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
                local_lumped = prepare_lumped_runtime(
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
                    resistance=0.0,
                )
                _validate_local_update_coefficient(
                    shard.solver,
                    local_lumped,
                    local_geometry.voltage_component,
                )

            with torch.cuda.device(owner_device):
                owner_lumped = _owner_proxy_lumped(
                    local_lumped,
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
                    lumped=owner_lumped,
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
                    embedded_network_name=block.name,
                )
            remote = port_plan.owner_rank != plan.owner_rank
            distributed_ports.append(
                _DistributedNetworkPortRuntime(
                    plan=port_plan,
                    shard=shard,
                    local_lumped=local_lumped,
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

        # Build the single owner-resident network runtime. Its per-port proxy
        # lumped views wire ``last_voltage_midpoint``/``last_branch_current`` onto
        # the owner network buffers, so the owner alone accumulates every port's
        # V/I DFT and the network state advances on one device.
        owner_shard.solver._port_runtimes = tuple(proxy_runtimes)
        # Propagate the requested output frequencies to the owner shard solver so
        # ``prepare_network_runtimes`` enforces the fitted-band 'reject' contract
        # against them, exactly as the single-device runtime does. Without this the
        # owner shard has no requested frequencies and out-of-band requests silently
        # pass on every multi-GPU network run.
        owner_shard.solver._requested_port_frequencies = tuple(
            float(value) for value in requested_frequencies
        )
        with torch.cuda.device(owner_device):
            network_runtimes = prepare_network_runtimes(
                owner_shard.solver,
                compiled_networks=(compiled,),
            )
        if len(network_runtimes) != 1:
            raise RuntimeError(
                "Distributed network preparation did not create one owner runtime."
            )
        return cls(
            plan=plan,
            block=block,
            shards=shards,
            port_runtimes=tuple(distributed_ports),
            owner_port_runtimes=tuple(proxy_runtimes),
            network_runtime=network_runtimes[0],
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

    @staticmethod
    def _sample_voltage(runtime: _DistributedNetworkPortRuntime) -> None:
        field_tensor = getattr(runtime.shard.solver, runtime.plan.voltage_component)
        lumped = runtime.local_lumped
        torch.index_select(
            field_tensor.reshape(-1),
            0,
            lumped.linear_indices,
            out=lumped.edge_buffer,
        )
        torch.mul(lumped.edge_buffer, lumped.voltage_weights, out=lumped.edge_buffer)
        torch.sum(lumped.edge_buffer, dim=0, out=lumped.last_voltage_before)

    @staticmethod
    def _apply_current(
        runtime: _DistributedNetworkPortRuntime,
        current: torch.Tensor,
    ) -> None:
        lumped = runtime.local_lumped
        field_tensor = getattr(runtime.shard.solver, runtime.plan.voltage_component)
        torch.mul(lumped.injection, current, out=lumped.correction_buffer)
        field_tensor.reshape(-1).index_add_(
            0,
            lumped.linear_indices,
            lumped.correction_buffer,
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
                    runtime.local_lumped.last_voltage_before,
                    non_blocking=True,
                )
                runtime.voltage_received.record(owner.communication_stream)

        with torch.cuda.device(owner.device), torch.cuda.stream(owner.compute_stream):
            free_voltages = []
            for runtime in self.port_runtimes:
                if runtime.owner_current_reuse.begin_owner_write():
                    owner.compute_stream.wait_event(runtime.current_copied)
                if runtime.voltage_received is None:
                    free_voltages.append(runtime.local_lumped.last_voltage_before)
                else:
                    owner.compute_stream.wait_event(runtime.voltage_received)
                    free_voltages.append(runtime.owner_voltage)
            currents = advance_network_external(
                self.network_runtime,
                tuple(free_voltages),
            )
            accumulate_port_observers(self.owner_solver)
            for index, runtime in enumerate(self.port_runtimes):
                if runtime.plan.owner_rank == self.plan.owner_rank:
                    self._apply_current(runtime, currents[index])
            self.solve_ready.record(owner.compute_stream)

        for index, runtime in enumerate(self.port_runtimes):
            if runtime.plan.owner_rank == self.plan.owner_rank:
                continue
            shard = runtime.shard
            with torch.cuda.device(shard.device), torch.cuda.stream(
                shard.communication_stream
            ):
                shard.communication_stream.wait_event(self.solve_ready)
                runtime.local_current.copy_(currents[index], non_blocking=True)
                runtime.current_copied.record(shard.communication_stream)
                self._apply_current(runtime, runtime.local_current)
                runtime.current_received.record(shard.communication_stream)
                runtime.owner_current_reuse.finish_remote_read_schedule()
            with torch.cuda.device(shard.device), torch.cuda.stream(shard.compute_stream):
                shard.compute_stream.wait_event(runtime.current_received)

    def checkpoint_tensors(self) -> dict[str, torch.Tensor]:
        # Slice U2: the trapezoidal interface carries the previous step's
        # post-step port voltage, so it is dynamic owner-GPU state alongside the
        # state-space vector -- a resume that dropped it would restart the
        # coupling from a zero half-step and desynchronize the network trace.
        return {
            f"network_state_{self.plan.network_name}": self.network_runtime.state,
            f"network_carried_voltage_{self.plan.network_name}": (
                self.network_runtime.carried_voltage
            ),
        }

    def finalize(self, result_device: torch.device):
        ports = finalize_port_data(self.owner_solver)
        port_output = {}
        for name, data in ports.items():
            metadata = dict(data.metadata)
            metadata.update(
                {
                    "distributed_network_owner_rank": self.plan.owner_rank,
                    "distributed_port_owner_rank": self.plan.port_owners[name],
                }
            )
            port_output[name] = move_port_data(
                data,
                result_device,
                metadata=metadata,
            )

        networks = finalize_embedded_networks(self.owner_solver, ports)
        network_output = {}
        for name, data in networks.items():
            metadata = dict(data.metadata)
            metadata.update(
                {
                    "distributed_network_owner_rank": self.plan.owner_rank,
                    "distributed_port_owner_ranks": self.plan.port_owners,
                }
            )
            network_output[name] = _move_embedded_network_data(
                data,
                result_device,
                metadata=metadata,
            )
        return port_output, network_output

    def stats(self, *, steps_run: int) -> dict[str, Any]:
        bytes_per_step = self.communication_bytes_per_step
        return {
            "enabled": True,
            "network_name": self.plan.network_name,
            "network_owner_rank": self.plan.owner_rank,
            "network_owner_device": str(self.owner_shard.device),
            "network_owner_reference_port": self.plan.owner_reference_port,
            "port_owner_ranks": self.plan.port_owners,
            "connected_port_count": len(self.port_runtimes),
            "remote_port_count": self.remote_port_count,
            "same_shard_fast_path_count": self.same_shard_port_count,
            "scalar_transfers_per_step": 2 * self.remote_port_count,
            "owner_copy_acknowledgements_per_step": self.remote_port_count,
            "communication_bytes_per_step": bytes_per_step,
            "communication_bytes_total": bytes_per_step * int(steps_run),
            "communication_order": "O(connected_ports)",
            "checkpoint_owner_rank": self.plan.owner_rank,
            "checkpoint_scope": "network_state_only",
        }


__all__ = [
    "DistributedNetworkPlan",
    "DistributedNetworkPortPlan",
    "DistributedNetworkRuntime",
    "compile_distributed_network_plan",
]
