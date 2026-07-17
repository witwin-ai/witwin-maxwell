from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ...monitors import (
    ClosedSurfaceMonitor,
    DiffractionMonitor,
    DipoleEmissionMonitor,
    FieldTimeMonitor,
    FinitePlaneMonitor,
    FluxTimeMonitor,
    MediumMonitor,
    ModeMonitor,
    PermittivityMonitor,
    PlaneMonitor,
    PointMonitor,
)
from ...scene import Domain, GridSpec, Scene, prepare_scene
from ...ports import LumpedPort, ModePort, TerminalPort
from ...sources import PointDipole, UniformCurrentSource
from ...fdtd_parallel import FDTDParallelConfig, FDTDPartitionPlan, FDTDShardLayout
from ..excitation import (
    inject_electric_surface_source_terms,
    inject_magnetic_surface_source_terms,
)
from ..observers import _merge_frequency_lists
from ..solver import FDTD
from ..runtime import stepping
from .counts import reduce_sample_counts
from .capacity import (
    local_dft_working_set_bytes,
    require_gather_capacity,
    require_local_dft_capacity,
)
from .circuits import DistributedCircuitRuntime
from .networks import DistributedNetworkRuntime
from .frequency_counts import reduce_frequency_sample_counts
from .monitor_merge import merge_sharded_monitor_payloads
from .persistence import export_distributed_field_shards
from .sources import crop_solver_source_terms_to_owned_x
from .source_corrections import correct_ideal_point_ex_control_volume
from .transport import CudaP2PHaloTransport


_CELL_COMPONENTS = frozenset(("Ex", "Hy", "Hz"))
_NODE_COMPONENTS = frozenset(("Ey", "Ez", "Hx"))


@dataclass
class FDTDShard:
    rank: int
    device: torch.device
    layout: FDTDShardLayout
    solver: FDTD
    compute_stream: torch.cuda.Stream
    communication_stream: torch.cuda.Stream
    electric_ready: torch.cuda.Event
    electric_received: torch.cuda.Event
    magnetic_ready: torch.cuda.Event
    magnetic_received: torch.cuda.Event
    halo_hy_low: torch.Tensor | None
    halo_hz_low: torch.Tensor | None
    peak_memory_bytes: int = 0

    @property
    def is_first(self) -> bool:
        return self.rank == 0

    @property
    def is_last(self) -> bool:
        return self.layout.owns_physical_face("x", "high")


def _physical_axis_nodes(prepared_scene, axis: str) -> np.ndarray:
    nodes = np.asarray(getattr(prepared_scene, f"{axis}_nodes64"), dtype=np.float64)
    low = int(prepared_scene.pml_thickness_for_face(axis, "low"))
    high = int(prepared_scene.pml_thickness_for_face(axis, "high"))
    stop = nodes.size - high if high else nodes.size
    return np.array(nodes[low:stop], copy=True)


def _owns_global_x(layout: FDTDShardLayout, prepared_scene, x: float) -> bool:
    nodes = np.asarray(prepared_scene.x_nodes64, dtype=np.float64)
    value = float(x)
    if value <= float(nodes[0]):
        cell = 0
    elif value >= float(nodes[-1]):
        cell = nodes.size - 2
    else:
        cell = int(np.searchsorted(nodes, value, side="right")) - 1
    return int(layout.global_cell_owned.start) <= cell < int(layout.global_cell_owned.stop)


def _point_source_position(scene: Scene, source_name: str):
    for source in scene.sources:
        if getattr(source, "name", None) == source_name and hasattr(source, "position"):
            return source.position
    return None


def _plane_position(monitor: PlaneMonitor | FinitePlaneMonitor | ModeMonitor) -> float:
    if isinstance(monitor, PlaneMonitor):
        return float(monitor.position)
    return float(monitor.plane_position)


def _matches_x_node(nodes: np.ndarray, node_index: int, value: float) -> bool:
    coordinate = float(nodes[node_index])
    spacings = []
    if node_index > 0:
        spacings.append(abs(coordinate - float(nodes[node_index - 1])))
    if node_index + 1 < nodes.size:
        spacings.append(abs(float(nodes[node_index + 1]) - coordinate))
    scale = max(
        abs(coordinate),
        abs(float(value)),
        max(spacings, default=0.0),
        np.finfo(np.float64).tiny,
    )
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    return abs(float(value) - coordinate) <= tolerance


def _local_monitors(scene: Scene, layout: FDTDShardLayout, global_prepared):
    selected = []
    for monitor in scene.monitors:
        if isinstance(monitor, (PermittivityMonitor, MediumMonitor)):
            # Material monitors are resolved from compiled tensors, not accumulated.
            selected.append(monitor)
            continue
        if isinstance(monitor, PointMonitor):
            if _owns_global_x(layout, global_prepared, monitor.position[0]):
                selected.append(monitor)
            continue
        if isinstance(monitor, FieldTimeMonitor) and monitor.region_kind == "point":
            if _owns_global_x(layout, global_prepared, monitor.position[0]):
                selected.append(monitor)
            continue
        if isinstance(monitor, DipoleEmissionMonitor):
            position = _point_source_position(scene, monitor.source_name)
            if position is not None and _owns_global_x(layout, global_prepared, position[0]):
                selected.append(monitor)
            continue
        if isinstance(monitor, (PlaneMonitor, FinitePlaneMonitor, ModeMonitor)):
            if monitor.axis != "x" or _owns_global_x(
                layout,
                global_prepared,
                _plane_position(monitor),
            ):
                selected.append(monitor)
            continue
        raise ValueError(
            f"Multi-GPU FDTD does not support {type(monitor).__name__}."
        )
    return selected


def _build_local_scene(
    logical_scene: Scene,
    global_prepared,
    layout: FDTDShardLayout,
    physical_x_nodes: np.ndarray,
) -> Scene:
    begin = int(layout.physical_cell_begin)
    end = int(layout.physical_cell_end)
    storage_begin = begin - (1 if layout.rank > 0 else 0)
    local_x = np.array(physical_x_nodes[storage_begin : end + 1], copy=True)
    local_y = _physical_axis_nodes(global_prepared, "y")
    local_z = _physical_axis_nodes(global_prepared, "z")

    global_boundary = logical_scene.boundary
    local_boundary = global_boundary.with_faces(
        x_low=global_boundary.face_kind("x", "low") if layout.rank == 0 else "none",
        x_high=(
            global_boundary.face_kind("x", "high")
            if layout.owns_physical_face("x", "high")
            else "none"
        ),
    )
    local_domain = Domain(
        bounds=(
            (float(local_x[0]), float(local_x[-1])),
            (float(local_y[0]), float(local_y[-1])),
            (float(local_z[0]), float(local_z[-1])),
        )
    )
    local_grid = GridSpec.custom(local_x, local_y, local_z)
    return logical_scene.clone(
        domain=local_domain,
        grid=local_grid,
        boundary=local_boundary,
        monitors=_local_monitors(logical_scene, layout, global_prepared),
        ports=(),
        circuits=(),
        networks=(),
        device=str(layout.device),
        symmetry=(None, logical_scene.symmetry[1], logical_scene.symmetry[2]),
    )


def _bounded_x_kwargs(layout: FDTDShardLayout, component: str, x_slice: slice) -> dict[str, int]:
    component_layout = layout.component(component)
    return {
        "localXBegin": int(x_slice.start),
        "localXEnd": int(x_slice.stop),
        "globalXOffset": int(component_layout.global_origin[0]),
        "globalXExtent": int(component_layout.global_shape[0]),
    }


def _launch_magnetic_hx(solver, layout: FDTDShardLayout, x_slice: slice) -> None:
    if x_slice.stop <= x_slice.start:
        return
    solver.fdtd_module.updateMagneticFieldHxStandardBounded3D(
        Hx=solver.Hx,
        Ey=solver.Ey,
        Ez=solver.Ez,
        HxDecay=solver.chx_decay,
        HxCurl=solver.chx_curl,
        invDy=solver.inv_dy_h,
        invDz=solver.inv_dz_h,
        **_bounded_x_kwargs(layout, "Hx", x_slice),
    ).launchRaw()


def _launch_magnetic_hy_hz(solver, layout: FDTDShardLayout, x_slice: slice) -> None:
    if x_slice.stop <= x_slice.start:
        return
    solver.fdtd_module.updateMagneticFieldHyStandardBounded3D(
        Hy=solver.Hy,
        Ex=solver.Ex,
        Ez=solver.Ez,
        HyDecay=solver.chy_decay,
        HyCurl=solver.chy_curl,
        invDx=solver.inv_dx_h,
        invDz=solver.inv_dz_h,
        **_bounded_x_kwargs(layout, "Hy", x_slice),
    ).launchRaw()
    solver.fdtd_module.updateMagneticFieldHzStandardBounded3D(
        Hz=solver.Hz,
        Ex=solver.Ex,
        Ey=solver.Ey,
        HzDecay=solver.chz_decay,
        HzCurl=solver.chz_curl,
        invDx=solver.inv_dx_h,
        invDy=solver.inv_dy_h,
        **_bounded_x_kwargs(layout, "Hz", x_slice),
    ).launchRaw()


def _electric_coefficients(solver):
    return stepping._electric_decay_tensors(solver), stepping._electric_curl_tensors(solver)


def _launch_electric_ex(solver, layout: FDTDShardLayout, x_slice: slice) -> None:
    if x_slice.stop <= x_slice.start:
        return
    decay, curl = _electric_coefficients(solver)
    solver.fdtd_module.updateElectricFieldExStandardBounded3D(
        Ex=solver.Ex,
        Hy=solver.Hy,
        Hz=solver.Hz,
        ExDecay=decay[0],
        ExCurl=curl[0],
        invDy=solver.inv_dy_e,
        invDz=solver.inv_dz_e,
        yLowBoundaryMode=solver.boundary_y_low_code,
        yHighBoundaryMode=solver.boundary_y_high_code,
        zLowBoundaryMode=solver.boundary_z_low_code,
        zHighBoundaryMode=solver.boundary_z_high_code,
        **_bounded_x_kwargs(layout, "Ex", x_slice),
    ).launchRaw()


def _launch_electric_ey_ez(
    solver,
    layout: FDTDShardLayout,
    node_slice: slice,
    *,
    x_low_mode: int,
    x_high_mode: int,
) -> None:
    if node_slice.stop <= node_slice.start:
        return
    decay, curl = _electric_coefficients(solver)
    solver.fdtd_module.updateElectricFieldEyStandardBounded3D(
        Ey=solver.Ey,
        Hx=solver.Hx,
        Hz=solver.Hz,
        EyDecay=decay[1],
        EyCurl=curl[1],
        invDx=solver.inv_dx_e,
        invDz=solver.inv_dz_e,
        xLowBoundaryMode=x_low_mode,
        xHighBoundaryMode=x_high_mode,
        zLowBoundaryMode=solver.boundary_z_low_code,
        zHighBoundaryMode=solver.boundary_z_high_code,
        **_bounded_x_kwargs(layout, "Ey", node_slice),
    ).launchRaw()
    solver.fdtd_module.updateElectricFieldEzStandardBounded3D(
        Ez=solver.Ez,
        Hx=solver.Hx,
        Hy=solver.Hy,
        EzDecay=decay[2],
        EzCurl=curl[2],
        invDx=solver.inv_dx_e,
        invDy=solver.inv_dy_e,
        xLowBoundaryMode=x_low_mode,
        xHighBoundaryMode=x_high_mode,
        yLowBoundaryMode=solver.boundary_y_low_code,
        yHighBoundaryMode=solver.boundary_y_high_code,
        **_bounded_x_kwargs(layout, "Ez", node_slice),
    ).launchRaw()


class DistributedFDTD:
    """Single-process x-slab coordinator that duck-types the prepared FDTD solver."""

    def __init__(
        self,
        scene: Scene,
        *,
        frequency: float,
        parallel: FDTDParallelConfig,
        absorber_type: str = "cpml",
        cpml_config: dict[str, Any] | None = None,
    ):
        if not isinstance(parallel, FDTDParallelConfig):
            raise TypeError("parallel must be an FDTDParallelConfig instance.")
        if parallel.transport == "nccl":
            raise RuntimeError(
                "NCCL transport is reserved by the API but is not available in the same-process runtime."
            )
        self.logical_scene = scene
        self.scene = prepare_scene(scene)
        self.frequency = float(frequency)
        # Requested output/port frequencies enforced against an embedded network's
        # fitted band. Defaults to the time-stepping frequency; the public
        # Simulation path overwrites this with the full requested set before
        # init_field so the owner shard rejects out-of-band requests identically
        # to the single-device runtime.
        self._requested_port_frequencies: tuple[float, ...] = (self.frequency,)
        self.parallel = parallel
        self.devices = tuple(torch.device(device) for device in parallel.devices)
        self.device = torch.device(parallel.result_device)
        self.absorber_type = str(absorber_type)
        self.cpml_config = dict(cpml_config or {})
        requested_cpml_mode = str(self.cpml_config.get("memory_mode", "auto")).strip().lower()
        if requested_cpml_mode not in {"auto", "dense", "slab"}:
            raise ValueError(
                "cpml_config['memory_mode'] must be one of 'auto', 'dense', or 'slab'."
            )
        self._cpml_memory_mode_requested = requested_cpml_mode
        self._cpml_memory_mode = "none"
        self._cpml_allocated_memory_bytes = 0
        self._cpml_dense_memory_bytes = 0
        self._cpml_slab_memory_bytes = 0
        self.c = 299792458.0
        self.Nx, self.Ny, self.Nz = self.scene.Nx, self.scene.Ny, self.scene.Nz
        self.dt = None
        self.shards: tuple[FDTDShard, ...] = ()
        self.partition_plan: FDTDPartitionPlan | None = None
        self.shard_layouts: tuple[FDTDShardLayout, ...] = ()
        self.transport = CudaP2PHaloTransport(self.devices)
        self.last_solve_elapsed_s = None
        self._cuda_graph_active = False
        self._tail_graph_active = False
        self._gather_preflight: dict[str, int] = {}
        self._dft_preflight: dict[str, dict[str, int]] = {}
        self._peak_memory_including_gather: dict[str, int] = {}
        self._shutoff_triggered = False
        self._shutoff_step = None
        self._shutoff_peak = None
        self._parallel_stats: dict[str, Any] = {}
        self._observer_frequencies: tuple[float, ...] = ()
        self._distributed_circuit: DistributedCircuitRuntime | None = None
        self._distributed_network: DistributedNetworkRuntime | None = None
        self._network_cuda_graph_active = False
        self._initialized = False
        self._validate_static_capabilities()

    def _validate_static_capabilities(self) -> None:
        boundary = self.logical_scene.boundary
        if boundary.uses_kind("bloch"):
            raise ValueError(
                "Multi-GPU Bloch fields require complex halo exchange, which is not enabled yet."
            )
        if boundary.axis_kind("x") == "periodic":
            raise ValueError(
                "Multi-GPU x decomposition does not yet support periodic x ring exchange."
            )
        if self.logical_scene.symmetry[0] is not None:
            raise ValueError("Multi-GPU x decomposition does not support x-axis symmetry yet.")
        if self.logical_scene.material_regions:
            raise ValueError(
                "Multi-GPU material density regions require distributed density slicing and are "
                "currently limited to the single-GPU adjoint path."
            )
        if self.logical_scene.networks and self.logical_scene.circuits:
            # Both distributed feedback runtimes claim owner-resident proxy port
            # runtimes on their owner shard's solver. Supporting both in one scene
            # would require merging those proxy sets; that combination is a
            # separate capability, so reject it explicitly rather than let the
            # second preparation silently clobber the first.
            raise ValueError(
                "Multi-GPU FDTD does not yet support a scene with both embedded networks "
                "and lumped circuits; run them in separate simulations."
            )
        if len(self.logical_scene.networks) > 1:
            raise ValueError(
                "Multi-GPU FDTD embedded-network coupling currently supports exactly one "
                "network per scene."
            )
        unsupported_ports = tuple(
            port
            for port in self.logical_scene.ports
            if not isinstance(port, (LumpedPort, TerminalPort))
        )
        if unsupported_ports:
            names = tuple(port.name for port in unsupported_ports)
            if any(isinstance(port, ModePort) for port in unsupported_ports):
                raise ValueError(
                    "Multi-GPU FDTD mode ports require distributed modal plane tiling; "
                    f"unsupported ports: {names}."
                )
            raise ValueError(f"Multi-GPU FDTD does not support port types for {names}.")
        bound_port_names = {
            binding.port_name
            for circuit in self.logical_scene.circuits
            for binding in circuit.bindings
        }
        bound_port_names.update(
            port_name
            for network in self.logical_scene.networks
            for port_name in network.connected_port_names
        )
        unbound_ports = tuple(
            port.name
            for port in self.logical_scene.ports
            if port.name not in bound_port_names
        )
        if unbound_ports:
            raise ValueError(
                "Multi-GPU lumped/terminal ports must be bound to the distributed "
                f"circuit or network owner; unbound ports: {unbound_ports}."
            )
        for monitor in self.logical_scene.monitors:
            if isinstance(monitor, ClosedSurfaceMonitor):
                raise ValueError(
                    "Multi-GPU FDTD does not yet support closed-surface monitor assembly."
                )
            if isinstance(monitor, DiffractionMonitor):
                raise ValueError(
                    "Multi-GPU FDTD does not yet support distributed diffraction monitors."
                )
            if isinstance(monitor, FluxTimeMonitor):
                raise ValueError(
                    "Multi-GPU FDTD does not yet support time-domain flux plane monitors."
                )
            if isinstance(monitor, FieldTimeMonitor) and monitor.region_kind != "point":
                raise ValueError(
                    "Multi-GPU FDTD currently supports FieldTimeMonitor only for point regions."
                )

        prospective_plan = FDTDPartitionPlan(
            global_shape=(self.Nx, self.Ny, self.Nz),
            devices=self.devices,
            decomposition_axis="x",
            halo_width=1,
            low_pml_cells=int(self.scene.pml_thickness_for_face("x", "low")),
            high_pml_cells=int(self.scene.pml_thickness_for_face("x", "high")),
        )
        internal_x_nodes = tuple(
            int(layout.global_cell_owned.stop)
            for layout in prospective_plan.shard_layouts[:-1]
        )
        x_nodes = np.asarray(self.scene.x_nodes64, dtype=np.float64)
        for monitor in self.logical_scene.monitors:
            if (
                isinstance(monitor, (PlaneMonitor, FinitePlaneMonitor))
                and monitor.axis == "x"
                and "Ex" in monitor.fields
                and not bool(monitor.compute_flux)
                and any(
                    _matches_x_node(x_nodes, node_index, _plane_position(monitor))
                    for node_index in internal_x_nodes
                )
            ):
                raise ValueError(
                    f"Multi-GPU x-normal monitor {monitor.name!r} requests Ex exactly on an "
                    "internal x partition node. The owning shard does not have a current "
                    "monitor-only low Ex halo. Move the plane off the split, remove Ex, or "
                    "use a tangential-field FluxMonitor/ModeMonitor."
                )
        if any(
            bool(getattr(getattr(structure, "material", None), "is_lossy_metal", False))
            for structure in self.logical_scene.structures
        ):
            raise ValueError("Multi-GPU FDTD does not yet support distributed SIBC ownership.")
        for monitor in self.logical_scene.resolved_monitors():
            if isinstance(monitor, (PermittivityMonitor, MediumMonitor)):
                raise ValueError(
                    "Multi-GPU material monitors require sharded material compilation; "
                    "full-domain material gathering is intentionally disabled."
                )
            if isinstance(monitor, DipoleEmissionMonitor):
                matches = tuple(
                    source
                    for source in self.logical_scene.sources
                    if getattr(source, "name", None) == monitor.source_name
                )
                if len(matches) != 1 or not isinstance(matches[0], PointDipole):
                    raise ValueError(
                        f"DipoleEmissionMonitor {monitor.name!r} requires exactly one named "
                        f"PointDipole source {monitor.source_name!r}."
                    )
        for source in self.logical_scene.sources:
            if not isinstance(source, (PointDipole, UniformCurrentSource)):
                raise ValueError(
                    f"Multi-GPU FDTD currently supports PointDipole and UniformCurrentSource; "
                    f"got {type(source).__name__}."
                )
            if isinstance(source, PointDipole) and source.profile != "ideal":
                raise ValueError(
                    "Multi-GPU PointDipole currently requires profile='ideal' so source "
                    "normalization has exactly one interface owner."
                )

    def _validate_hardware(self) -> None:
        self.transport.preflight()
        if self.device.index is None:
            raise RuntimeError("Multi-GPU result_device must be an indexed CUDA device.")
        for device in self.devices:
            if device == self.device:
                continue
            if device.index is None or not torch.cuda.can_device_access_peer(
                self.device.index, device.index
            ):
                raise RuntimeError(
                    f"Direct CUDA peer access is unavailable from {self.device} to {device}; "
                    "result gathering/reduction cannot silently stage through host memory."
                )
            if not torch.cuda.can_device_access_peer(device.index, self.device.index):
                raise RuntimeError(
                    f"Direct CUDA peer access is unavailable from {device} to {self.device}; "
                    "result gathering/reduction cannot silently stage through host memory."
                )
        properties = [torch.cuda.get_device_properties(device) for device in self.devices]
        signatures = {(prop.name, prop.major, prop.minor) for prop in properties}
        if len(signatures) != 1:
            raise RuntimeError(
                "Multi-GPU FDTD requires homogeneous devices with the same model and compute capability."
            )

    def init_field(self) -> None:
        if self._initialized:
            return
        self._validate_hardware()
        timing_scene = prepare_scene(
            self.logical_scene.clone(ports=(), circuits=(), networks=())
        )
        reference_solver = FDTD(
            timing_scene,
            frequency=self.frequency,
            absorber_type=self.absorber_type,
            cpml_config=self.cpml_config,
        )
        self.dt = float(reference_solver.dt)
        del reference_solver
        low_pml = int(self.scene.pml_thickness_for_face("x", "low"))
        high_pml = int(self.scene.pml_thickness_for_face("x", "high"))
        self.partition_plan = FDTDPartitionPlan(
            global_shape=(self.Nx, self.Ny, self.Nz),
            devices=self.devices,
            decomposition_axis="x",
            halo_width=1,
            low_pml_cells=low_pml,
            high_pml_cells=high_pml,
        )
        self.shard_layouts = self.partition_plan.shard_layouts
        physical_x = _physical_axis_nodes(self.scene, "x")

        shards = []
        for layout in self.shard_layouts:
            device = torch.device(layout.device)
            with torch.cuda.device(device):
                torch.cuda.reset_peak_memory_stats(device)
                local_scene = _build_local_scene(
                    self.logical_scene,
                    self.scene,
                    layout,
                    physical_x,
                )
                local_solver = FDTD(
                    local_scene,
                    frequency=self.frequency,
                    absorber_type=self.absorber_type,
                    cpml_config=self.cpml_config,
                )
                local_solver.dt = self.dt
                local_solver.init_field()
                if local_solver.nonlinear_enabled:
                    raise ValueError(
                        "Multi-GPU nonlinear media require additional collocation halos and "
                        "bounded nonlinear kernels."
                    )
                if local_solver.full_aniso_enabled:
                    raise ValueError(
                        "Multi-GPU full off-diagonal anisotropy requires additional H/curl halos."
                    )
                crop_solver_source_terms_to_owned_x(local_solver, layout)
                correct_ideal_point_ex_control_volume(local_solver, layout, self.scene)
                compute_stream = torch.cuda.Stream(device=device)
                communication_stream = torch.cuda.Stream(device=device, priority=-1)
                halo_hy = local_solver.Hy[0] if layout.rank > 0 else None
                compute_stream.wait_stream(torch.cuda.current_stream(device))
                halo_hz = local_solver.Hz[0] if layout.rank > 0 else None
                shard = FDTDShard(
                    rank=layout.rank,
                    device=device,
                    layout=layout,
                    solver=local_solver,
                    compute_stream=compute_stream,
                    communication_stream=communication_stream,
                    electric_ready=torch.cuda.Event(),
                    electric_received=torch.cuda.Event(),
                    magnetic_ready=torch.cuda.Event(),
                    magnetic_received=torch.cuda.Event(),
                    halo_hy_low=halo_hy,
                    halo_hz_low=halo_hz,
                )
                self._validate_local_layout(shard)
                shards.append(shard)

        self.shards = tuple(shards)
        dts = tuple(float(shard.solver.dt) for shard in self.shards)
        if not all(np.isclose(self.dt, dt, rtol=0.0, atol=1e-18) for dt in dts):
            raise RuntimeError(f"Shard-local time steps disagree: {dts}.")
        modes = tuple(
            getattr(shard.solver, "_cpml_memory_mode", "none")
            for shard in self.shards
        )
        active_modes = {mode for mode in modes if mode != "none"}
        self._cpml_memory_mode = (
            "none"
            if not active_modes
            else next(iter(active_modes)) if len(active_modes) == 1 else "mixed"
        )

        def _sum_shard_bytes(attribute: str) -> int:
            return sum(
                int(getattr(shard.solver, attribute, 0))
                for shard in self.shards
            )

        self._cpml_allocated_memory_bytes = _sum_shard_bytes("_cpml_allocated_memory_bytes")
        self._cpml_dense_memory_bytes = _sum_shard_bytes("_cpml_dense_memory_bytes")
        self._cpml_slab_memory_bytes = _sum_shard_bytes("_cpml_slab_memory_bytes")
        self._distributed_circuit = DistributedCircuitRuntime.prepare(
            prepared_scene=self.scene,
            partition_plan=self.partition_plan,
            shards=self.shards,
            frequency=self.frequency,
        )
        self._distributed_network = DistributedNetworkRuntime.prepare(
            prepared_scene=self.scene,
            partition_plan=self.partition_plan,
            shards=self.shards,
            frequency=self.frequency,
            requested_frequencies=self._requested_port_frequencies,
        )
        self._initialized = True

    @staticmethod
    def _validate_local_layout(shard: FDTDShard) -> None:
        layout = shard.layout
        solver = shard.solver
        cell_owned = layout.storage_cell_owned
        node_owned = layout.storage_node_owned
        if cell_owned.stop > solver.Nx - 1 or node_owned.stop > solver.Nx:
            raise RuntimeError(
                f"Rank {shard.rank} padded storage is smaller than its declared owned slices."
            )
        expected_low_pad = 0 if shard.rank == 0 else 1
        if cell_owned.start != expected_low_pad or node_owned.start != expected_low_pad:
            raise RuntimeError(
                f"Rank {shard.rank} has invalid padded low ownership: "
                f"cell={cell_owned}, node={node_owned}."
            )

    def _prepare_outputs(self, time_steps, dft_frequency, dft_window, full_field_dft):
        spectral_enabled = any(shard.solver.observers for shard in self.shards)
        default_frequencies = dft_frequency if dft_frequency is not None else (self.frequency,)
        monitor_frequencies = tuple(
            getattr(monitor, "frequencies", None)
            for monitor in self.logical_scene.resolved_monitors()
            if not isinstance(monitor, FieldTimeMonitor)
        )
        self._observer_frequencies = (
            _merge_frequency_lists(default_frequencies, *monitor_frequencies) if spectral_enabled else ()
        )
        for shard in self.shards:
            solver = shard.solver
            with torch.cuda.device(shard.device), torch.cuda.stream(shard.compute_stream):
                if dft_frequency is not None and full_field_dft:
                    solver.enable_dft(dft_frequency, window_type=dft_window, end_step=time_steps)
                else:
                    solver.dft_enabled = False
                    solver._dft_entries = []
                    solver._sync_dft_legacy_state()
                observer_frequency = dft_frequency if dft_frequency is not None else solver.source_frequency
                if solver.observers:
                    solver._prepare_observers(observer_frequency, dft_window, time_steps)
                if solver.time_observers:
                    solver._prepare_time_observers(time_steps)
                solver._shutoff_triggered = False
                solver._shutoff_step = None
        if self._distributed_circuit is not None:
            self._distributed_circuit.prepare_outputs(
                time_steps=int(time_steps),
                frequencies=default_frequencies,
                window_type=dft_window,
            )
        if self._distributed_network is not None:
            self._distributed_network.prepare_outputs(
                time_steps=int(time_steps),
                frequencies=default_frequencies,
                window_type=dft_window,
            )

    def _overlap_active(self) -> bool:
        if not self.parallel.overlap:
            return False
        return all(
            not shard.solver.uses_cpml
            and not shard.solver.complex_fields_enabled
            and not getattr(shard.solver, "modulation_enabled", False)
            for shard in self.shards
        )

    def _advance_magnetic_overlapped(self) -> None:
        self.transport.exchange_electric(self.shards)
        for shard in self.shards:
            solver = shard.solver
            cs = shard.layout.storage_cell_owned
            ns = shard.layout.storage_node_owned
            with torch.cuda.device(shard.device), torch.cuda.stream(shard.compute_stream):
                _launch_magnetic_hx(solver, shard.layout, ns)
                if shard.is_last:
                    _launch_magnetic_hy_hz(solver, shard.layout, cs)
                    continue
                interior = slice(cs.start, max(cs.start, cs.stop - 1))
                _launch_magnetic_hy_hz(solver, shard.layout, interior)
                shard.compute_stream.wait_event(shard.electric_received)
                _launch_magnetic_hy_hz(
                    solver, shard.layout, slice(cs.stop - 1, cs.stop)
                )

    def _advance_electric_overlapped(self) -> None:
        self.transport.exchange_magnetic(self.shards)
        for shard in self.shards:
            solver = shard.solver
            cs = shard.layout.storage_cell_owned
            ns = shard.layout.storage_node_owned
            with torch.cuda.device(shard.device), torch.cuda.stream(shard.compute_stream):
                _launch_electric_ex(solver, shard.layout, cs)
                if shard.is_first:
                    _launch_electric_ey_ez(
                        solver,
                        shard.layout,
                        ns,
                        x_low_mode=solver.boundary_x_low_code,
                        x_high_mode=solver.boundary_x_high_code,
                    )
                    continue
                interior = slice(min(ns.start + 1, ns.stop), ns.stop)
                _launch_electric_ey_ez(
                    solver,
                    shard.layout,
                    interior,
                    x_low_mode=solver.boundary_x_low_code,
                    x_high_mode=solver.boundary_x_high_code,
                )
                shard.compute_stream.wait_event(shard.magnetic_received)
                boundary = slice(ns.start, min(ns.start + 1, ns.stop))
                _launch_electric_ey_ez(
                    solver,
                    shard.layout,
                    boundary,
                    x_low_mode=solver.boundary_x_low_code,
                    x_high_mode=solver.boundary_x_high_code,
                )

    def _advance_magnetic_serialized(self) -> None:
        self.transport.exchange_electric(self.shards)
        for shard in self.shards:
            with torch.cuda.device(shard.device), torch.cuda.stream(shard.compute_stream):
                if not shard.is_last:
                    shard.compute_stream.wait_event(shard.electric_received)
                solver = shard.solver
                if not solver.uses_cpml and not solver.complex_fields_enabled:
                    _launch_magnetic_hx(
                        solver, shard.layout, shard.layout.storage_node_owned
                    )
                    _launch_magnetic_hy_hz(
                        solver, shard.layout, shard.layout.storage_cell_owned
                    )
                else:
                    stepping.update_magnetic_fields(
                        solver,
                        solver.Hx, solver.Hy, solver.Hz,
                        solver.Ex, solver.Ey, solver.Ez,
                    )

    def _advance_electric_serialized(self, time_value: float) -> None:
        self.transport.exchange_magnetic(self.shards)
        for shard in self.shards:
            with torch.cuda.device(shard.device), torch.cuda.stream(shard.compute_stream):
                if not shard.is_first:
                    shard.compute_stream.wait_event(shard.magnetic_received)
                solver = shard.solver
                if (
                    not solver.uses_cpml
                    and not solver.complex_fields_enabled
                    and not solver.modulation_enabled
                ):
                    _launch_electric_ex(
                        solver, shard.layout, shard.layout.storage_cell_owned
                    )
                    _launch_electric_ey_ez(
                        solver,
                        shard.layout,
                        shard.layout.storage_node_owned,
                        x_low_mode=solver.boundary_x_low_code,
                        x_high_mode=solver.boundary_x_high_code,
                    )
                else:
                    stepping.update_electric_fields(
                        solver,
                        solver.Ex,
                        solver.Ey,
                        solver.Ez,
                        solver.Hx,
                        solver.Hy,
                        solver.Hz,
                        time_value=time_value,
                    )

    def _advance_one_step(self, n: int, *, overlap_active: bool) -> None:
        time_value = n * self.dt
        for shard in self.shards:
            solver = shard.solver
            with torch.cuda.device(shard.device), torch.cuda.stream(shard.compute_stream):
                if solver.modulation_enabled:
                    solver.fdtd_module.advanceModulationTime3D(
                        ModulationTime=solver._modulation_time,
                        dt=solver.dt,
                    ).launchRaw()
                solver._advance_magnetic_dispersive_state()

        if overlap_active:
            self._advance_magnetic_overlapped()
        else:
            self._advance_magnetic_serialized()

        for shard in self.shards:
            solver = shard.solver
            with torch.cuda.device(shard.device), torch.cuda.stream(shard.compute_stream):
                if solver._magnetic_source_terms:
                    inject_magnetic_surface_source_terms(solver, time_value=time_value)
                solver._apply_magnetic_dispersive_corrections()
                solver._advance_dispersive_state()
                if solver.nonlinear_enabled:
                    solver._update_nonlinear_electric_coefficients()
                stepping.capture_aniso_conduction_currents(solver)

        if overlap_active:
            self._advance_electric_overlapped()
        else:
            self._advance_electric_serialized(time_value)

        for shard in self.shards:
            solver = shard.solver
            with torch.cuda.device(shard.device), torch.cuda.stream(shard.compute_stream):
                if solver.full_aniso_enabled:
                    stepping.apply_full_aniso_corrections(solver)
                    stepping.apply_full_aniso_conduction(solver)
                if getattr(solver, "_sibc", None) is not None:
                    raise RuntimeError("Distributed SIBC surface ownership is not enabled.")
                if solver._electric_source_terms:
                    inject_electric_surface_source_terms(
                        solver, time_value=time_value + 0.5 * float(solver.dt)
                    )
                if solver._source_terms:
                    solver.add_source(time_value=time_value)

        if self._distributed_circuit is not None:
            self._distributed_circuit.apply()
        if self._distributed_network is not None:
            self._distributed_network.apply()

        for shard in self.shards:
            solver = shard.solver
            with torch.cuda.device(shard.device), torch.cuda.stream(shard.compute_stream):
                solver._apply_dispersive_corrections()
                stepping.enforce_pec_boundaries(solver)
                stepping.apply_mur_boundaries(solver)
                solver.accumulate_dft(n)
                solver.accumulate_observers(n)
                solver.accumulate_time_observers(n)

    def _owned_electric_energy(self, shard: FDTDShard) -> torch.Tensor:
        solver = shard.solver
        cs = shard.layout.storage_cell_owned
        ns = shard.layout.storage_node_owned
        return (
            (solver.eps_Ex[cs] * solver.Ex[cs] * solver.Ex[cs]).sum()
            + (solver.eps_Ey[ns] * solver.Ey[ns] * solver.Ey[ns]).sum()
            + (solver.eps_Ez[ns] * solver.Ez[ns] * solver.Ez[ns]).sum()
        )

    def _global_shutoff_energy(self) -> torch.Tensor:
        local_energies = []
        for shard in self.shards:
            with torch.cuda.device(shard.device), torch.cuda.stream(shard.compute_stream):
                local = self._owned_electric_energy(shard)
                shard.electric_ready.record(shard.compute_stream)
            local_energies.append(local)
        with torch.cuda.device(self.device):
            result_stream = torch.cuda.current_stream(self.device)
            total = torch.zeros((), device=self.device, dtype=torch.float32)
            for shard, local in zip(self.shards, local_energies):
                result_stream.wait_event(shard.electric_ready)
                total.add_(local.to(self.device, non_blocking=True))
            return total

    def _synchronize_all(self) -> None:
        for shard in self.shards:
            shard.compute_stream.synchronize()
            shard.communication_stream.synchronize()

    def _gather_component(self, component: str, local_values: tuple[torch.Tensor, ...]) -> torch.Tensor:
        is_cell = component.capitalize() in _CELL_COMPONENTS
        global_x = self.Nx - 1 if is_cell else self.Nx
        sample = local_values[0]
        x_axis = sample.ndim - 3
        shape = list(sample.shape)
        shape[x_axis] = global_x
        destination = torch.empty(tuple(shape), device=self.device, dtype=sample.dtype)
        for shard, value in zip(self.shards, local_values):
            local_slice = (
                shard.layout.storage_cell_owned if is_cell else shard.layout.storage_node_owned
            )
            global_slice = (
                shard.layout.global_cell_owned if is_cell else shard.layout.global_node_owned
            )
            src_index = [slice(None)] * value.ndim
            dst_index = [slice(None)] * destination.ndim
            src_index[x_axis] = local_slice
            dst_index[x_axis] = global_slice
            destination[tuple(dst_index)].copy_(value[tuple(src_index)], non_blocking=True)
        return destination

    def _collect_output(self) -> dict[str, Any] | None:
        local_fields: dict[str, list[torch.Tensor]] = {}
        shard_monitor_payloads: list[tuple[int, dict[str, Any]]] = []
        frequency_metadata: tuple[float, ...] | None = None
        for shard in self.shards:
            solver = shard.solver
            shard_monitors: dict[str, Any] = {}
            if solver.dft_enabled:
                local = solver.get_frequency_solution(all_frequencies=True)
                metadata = local.get("frequencies")
                if metadata is not None:
                    if isinstance(metadata, torch.Tensor):
                        values = tuple(
                            float(value) for value in metadata.detach().cpu().tolist()
                        )
                    else:
                        values = tuple(float(value) for value in metadata)
                    if frequency_metadata is None:
                        frequency_metadata = values
                    elif frequency_metadata != values:
                        raise RuntimeError(
                            "Shard-local DFT frequency metadata is inconsistent."
                        )
                for name, tensor in local.items():
                    if name not in {"Ex", "Ey", "Ez"}:
                        continue
                    local_fields.setdefault(name, []).append(tensor)
            for enabled, getter in (
                (solver.observers_enabled, solver.get_observer_results),
                (solver.time_observers_enabled, solver.get_time_observer_results),
            ):
                if not enabled:
                    continue
                for name, payload in getter().items():
                    if name in shard_monitors:
                        raise RuntimeError(
                            f"Monitor {name!r} appears in multiple observer groups on shard "
                            f"{shard.rank}."
                        )
                    shard_monitors[name] = payload
            shard_monitor_payloads.append((shard.rank, shard_monitors))

        monitors = merge_sharded_monitor_payloads(
            (monitor.name for monitor in self.logical_scene.resolved_monitors()),
            shard_monitor_payloads,
            result_device=self.device,
            shard_layouts=self.shard_layouts,
            physical_bounds=self.logical_scene.domain.bounds,
        )

        output: dict[str, Any] = {}
        if self.parallel.gather_fields:
            if local_fields:
                for name, values in local_fields.items():
                    output[name] = self._gather_component(name, tuple(values))
            else:
                for name in ("Ex", "Ey", "Ez"):
                    values = tuple(getattr(shard.solver, name) for shard in self.shards)
                    output[name] = self._gather_component(name, values)
        if frequency_metadata is not None:
            output["frequencies"] = frequency_metadata
        if monitors:
            output["observers"] = monitors
        if self._distributed_circuit is not None:
            ports, circuits = self._distributed_circuit.finalize(self.device)
            if ports:
                output["ports"] = ports
            if circuits:
                output["circuits"] = circuits
        if self._distributed_network is not None:
            ports, networks = self._distributed_network.finalize(self.device)
            if ports:
                output.setdefault("ports", {}).update(ports)
            if networks:
                output["embedded_networks"] = networks
        return output or None

    def solve(
        self,
        time_steps: int,
        dft_frequency: float = None,
        enable_plot: bool = False,
        dft_window: str = "hanning",
        full_field_dft: bool = True,
        normalize_source: bool = False,
        shutoff: float = 0.0,
        shutoff_check_interval: int = 100,
        use_cuda_graph: bool = False,
        resume_from=None,
        stop_step: int | None = None,
    ):
        if resume_from is not None or stop_step is not None:
            raise ValueError(
                "Distributed FDTD checkpoint replay is not available; circuit state is "
                "owned and checkpointable on one GPU, but field-shard replay is deferred."
            )
        if not self._initialized:
            self.init_field()
        self._gather_preflight = {}
        self._dft_preflight = {}
        for shard in self.shards:
            local_shape = (
                shard.solver.Nx,
                shard.solver.Ny,
                shard.solver.Nz,
            )
            pending_dft = local_dft_working_set_bytes(
                local_shape,
                dft_frequency=dft_frequency,
                full_field_dft=full_field_dft,
            )
            if shard.device == self.device and self.parallel.gather_fields:
                preflight = require_gather_capacity(
                    self.device,
                    (self.Nx, self.Ny, self.Nz),
                    dft_frequency=dft_frequency,
                    full_field_dft=full_field_dft,
                    pending_local_dft_bytes=pending_dft,
                )
                self._gather_preflight = dict(preflight)
            else:
                preflight = require_local_dft_capacity(
                    shard.device,
                    local_shape,
                    dft_frequency=dft_frequency,
                    full_field_dft=full_field_dft,
                )
            self._dft_preflight[str(shard.device)] = dict(preflight)
        if enable_plot:
            raise ValueError("Multi-GPU FDTD plotting requires an explicit gathered result slice.")
        if use_cuda_graph:
            raise ValueError("Multi-GPU FDTD does not capture peer communication in CUDA Graphs.")
        if normalize_source and len(self.logical_scene.sources) != 1:
            raise ValueError(
                "Multi-GPU source normalization requires exactly one logical source."
            )
        for shard in self.shards:
            shard.solver._normalize_source = bool(normalize_source)

        self._prepare_outputs(time_steps, dft_frequency, dft_window, full_field_dft)
        shutoff_min_step = max(
            stepping._compute_shutoff_min_step(shard.solver, int(shutoff_check_interval))
            for shard in self.shards
        )
        overlap_active = self._overlap_active()
        self._shutoff_triggered = False
        self._shutoff_step = None
        self._shutoff_peak = torch.zeros((), device=self.device, dtype=torch.float32)
        self._synchronize_all()
        start = time.perf_counter()

        for n in range(int(time_steps)):
            self._advance_one_step(n, overlap_active=overlap_active)
            if shutoff > 0.0 and (n + 1) % int(shutoff_check_interval) == 0:
                energy = self._global_shutoff_energy()
                self._shutoff_peak = torch.maximum(self._shutoff_peak, energy)
                if bool(
                    n >= shutoff_min_step
                    and ((self._shutoff_peak > 0.0) & (energy < float(shutoff) * self._shutoff_peak)).item()
                ):
                    self._shutoff_triggered = True
                    self._shutoff_step = n
                    break

        self._synchronize_all()
        self.last_solve_elapsed_s = time.perf_counter() - start
        for shard in self.shards:
            solver = shard.solver
            solver.last_solve_elapsed_s = self.last_solve_elapsed_s
            solver._shutoff_triggered = self._shutoff_triggered
            solver._shutoff_step = self._shutoff_step
            if self._shutoff_triggered:
                with torch.cuda.device(shard.device):
                    stepping._complete_spectral_normalization(solver, int(time_steps))
            if solver.dft_enabled:
                solver._sync_dft_legacy_state()
            if solver.observers_enabled:
                solver._sync_observer_legacy_state()
            shard.peak_memory_bytes = int(torch.cuda.max_memory_allocated(shard.device))

        output = self._collect_output()
        for device in self.devices:
            torch.cuda.synchronize(device)
        self._peak_memory_including_gather = {
            str(device): int(torch.cuda.max_memory_allocated(device))
            for device in self.devices
        }
        self._parallel_stats = self._build_parallel_stats(time_steps, overlap_active)
        return output

    def _build_parallel_stats(self, time_steps: int, overlap_active: bool) -> dict[str, Any]:
        steps_run = (self._shutoff_step + 1) if self._shutoff_triggered else int(time_steps)
        halo_bytes_per_step = 0
        for left, right in zip(self.shards[:-1], self.shards[1:]):
            halo_bytes_per_step += (
                left.solver.Ey[-1].numel()
                + left.solver.Ez[-1].numel()
                + right.solver.Hy[0].numel()
                + right.solver.Hz[0].numel()
            ) * left.solver.Ex.element_size()
        partitions = tuple(
            {
                "rank": shard.rank,
                "device": str(shard.device),
                "physical_cells": (
                    shard.layout.physical_cell_begin,
                    shard.layout.physical_cell_end,
                ),
                "global_cells": (
                    shard.layout.global_cell_owned.start,
                    shard.layout.global_cell_owned.stop,
                ),
                "global_nodes": (
                    shard.layout.global_node_owned.start,
                    shard.layout.global_node_owned.stop,
                ),
                "peak_memory_bytes": shard.peak_memory_bytes,
            }
            for shard in self.shards
        )
        stats = {
            "devices": tuple(str(device) for device in self.devices),
            "decomposition_axis": "x",
            "transport": self.transport.name,
            "topology": self.transport.topology,
            "overlap_requested": bool(self.parallel.overlap),
            "overlap_active": bool(overlap_active),
            "gather_fields": bool(self.parallel.gather_fields),
            "result_device": str(self.device),
            "gather_preflight": dict(self._gather_preflight),
            "dft_preflight": {device: dict(stats) for device, stats in self._dft_preflight.items()},
            "partitions": partitions,
            "halo_bytes_per_step": int(halo_bytes_per_step),
            "halo_bytes_total": int(halo_bytes_per_step * steps_run),
            "peak_memory_bytes": {
                str(shard.device): shard.peak_memory_bytes for shard in self.shards
            },
            "peak_memory_bytes_including_gather": dict(self._peak_memory_including_gather),
            "wall_time_s": self.last_solve_elapsed_s,
            "compute_time_s": None,
            "communication_time_s": None,
            "exposed_communication_time_s": None,
            "timing_note": (
                "Phase timings require an external CUDA profiler; per-step synchronization "
                "is intentionally not inserted into the production solve loop."
            ),
        }
        if self._distributed_circuit is not None:
            stats["circuit"] = self._distributed_circuit.stats(steps_run=steps_run)
        if self._distributed_network is not None:
            stats["network"] = self._distributed_network.stats(steps_run=steps_run)
        return stats

    @property
    def parallel_stats(self) -> dict[str, Any]:
        return dict(self._parallel_stats)

    def export_field_shards(self):
        """Export owned electric-field shards without a global gather."""

        return export_distributed_field_shards(self)

    def circuit_checkpoint_tensors(self) -> dict[str, torch.Tensor]:
        """Return live owner-GPU circuit history tensors without gathering fields."""

        if self._distributed_circuit is None:
            return {}
        return self._distributed_circuit.checkpoint_tensors()

    def network_checkpoint_tensors(self) -> dict[str, torch.Tensor]:
        """Return live owner-GPU embedded-network state without gathering fields."""

        if self._distributed_network is None:
            return {}
        return self._distributed_network.checkpoint_tensors()

    @property
    def dft_sample_counts(self):
        return reduce_sample_counts(self.shards, "dft_sample_counts")

    @property
    def observer_frequencies(self):
        return tuple(self._observer_frequencies)

    @property
    def observer_sample_counts(self):
        return reduce_frequency_sample_counts(self.shards, self._observer_frequencies)

    @property
    def active_absorber_type(self):
        if not self.shards:
            return "none"
        active = {
            shard.solver.active_absorber_type for shard in self.shards
        } - {"none"}
        if not active:
            return "none"
        if len(active) != 1:
            raise RuntimeError(f"Shard-local absorber types disagree: {sorted(active)}.")
        return next(iter(active))

    @property
    def boundary_kind(self):
        return self.logical_scene.boundary.kind
