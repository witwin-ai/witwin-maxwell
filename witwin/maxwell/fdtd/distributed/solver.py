from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import torch
from witwin.core import Box

from ...monitors import (
    ClosedSurfaceMonitor,
    DiffractionMonitor,
    DipoleEmissionMonitor,
    FieldTimeMonitor,
    FinitePlaneMonitor,
    FluxTimeMonitor,
    MediumMonitor,
    PermittivityMonitor,
    PlaneMonitor,
    WireMonitor,
)
from ...scene import Scene, prepare_scene
from ...ports import LumpedPort, ModePort, TerminalPort
from ...sources import PointDipole, UniformCurrentSource
from ...fdtd_parallel import FDTDParallelConfig, FDTDPartitionPlan, FDTDShardLayout
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
from .nccl_transport import NcclHaloTransport
from .shard_engine import ShardEngine, _plane_position
from .transport import CudaP2PHaloTransport
from .wire import DistributedWireRuntime, move_wire_monitor


def _unsupported_distributed_trainable_tensors(scene: Scene) -> tuple[torch.Tensor, ...]:
    """Grad-requiring scene leaves the distributed adjoint bridge cannot yet handle.

    The distributed joint-solve adjoint bridge differentiates Box material-region
    densities (the density texture rasterizes per shard by physical position, and
    the reverse pass gathers the per-shard grad_eps owned slices into the logical
    scene before running the existing single-GPU material pullback). Every other
    trainable channel -- structure geometry, material perturbation tensors, circuit
    parameters, and RF/port parameters -- has no verified distributed reverse core,
    so a scene carrying one is rejected here as defense in depth even if constructed
    directly, independent of the public ``Simulation`` capability validator.

    Density is intentionally excluded: it is the one supported trainable channel, so
    rejecting it would block the very workflow the bridge exists to run.
    """

    # Imported lazily to avoid an import cycle: ``simulation`` pulls in the
    # distributed backend on demand, so the collectors cannot be a module-level
    # import here.
    from ...simulation import (
        _scene_trainable_circuit_parameters,
        _scene_trainable_geometry_parameters,
        _scene_trainable_material_parameters,
        _trainable_rf_parameters,
    )

    return (
        _scene_trainable_geometry_parameters(scene)
        + _scene_trainable_material_parameters(scene)
        + _scene_trainable_circuit_parameters(scene)
        + _trainable_rf_parameters(scene)
    )


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
        allow_adjoint: bool = False,
    ):
        if not isinstance(parallel, FDTDParallelConfig):
            raise TypeError("parallel must be an FDTDParallelConfig instance.")
        # Internal flag: set only by the verified NCCL adjoint driver so it may
        # construct a per-rank solver for a trainable point-monitor / full-field
        # objective without tripping the forward-only NCCL monitor / trainable
        # fences. Every other capability fence stays active. Never exposed on the
        # public Simulation path -- a plain NCCL forward run keeps allow_adjoint
        # False and the fences reject a trainable/monitor scene as before.
        self._allow_adjoint = bool(allow_adjoint)
        self._nccl = parallel.transport == "nccl"
        if self._nccl and not all(
            os.environ.get(key) for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK")
        ):
            # torch 2.13 binds one device per rank, so NCCL requires a
            # one-process-per-GPU torchrun launch. Without the launcher
            # environment the single-process coordinator cannot drive the
            # per-rank shape, and it must fail closed rather than silently build
            # the in-process CUDA P2P transport (a different execution).
            raise RuntimeError(
                "NCCL transport requires a one-process-per-GPU torchrun launch; "
                "RANK/WORLD_SIZE/LOCAL_RANK are not set. Launch with "
                "`torchrun --nproc-per-node=<gpus> ...`, or use transport='cuda_p2p' "
                "(or 'auto') for the in-process multi-GPU runtime."
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
        self.shards: tuple[ShardEngine, ...] = ()
        self.partition_plan: FDTDPartitionPlan | None = None
        self.shard_layouts: tuple[FDTDShardLayout, ...] = ()
        if self._nccl:
            # One-process-per-GPU: this process owns exactly one rank/device. The
            # transport reads RANK/WORLD_SIZE/LOCAL_RANK and validates the world
            # size against the configured device count.
            self.transport = NcclHaloTransport.from_env(
                expected_world_size=len(self.devices),
                timeout_s=float(parallel.timeout_s),
            )
            self.rank = int(self.transport.rank)
            self.world_size = int(self.transport.world_size)
            self._result_root = self.rank == 0
        else:
            self.transport = CudaP2PHaloTransport(self.devices, result_device=self.device)
            self.rank = 0
            self.world_size = 1
            self._result_root = True
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
        self._distributed_wire: DistributedWireRuntime | None = None
        self._wire_network_cpu = None
        self._wire_monitors_cpu: tuple = ()
        self._wire_cfl_metadata: dict[str, Any] = {}
        self._network_cuda_graph_active = False
        self._initialized = False
        self._validate_static_capabilities()

    def _validate_static_capabilities(self) -> None:
        if self._nccl:
            self._validate_nccl_capabilities()
        # Defense in depth: the public Simulation entry validates trainable+parallel
        # per capability, but the distributed solver must also fail closed if
        # constructed directly with an unsupported trainable channel. Trainable Box
        # densities are supported by the distributed joint-solve adjoint bridge;
        # every other trainable channel would otherwise run a forward-only solve and
        # silently drop its gradient.
        if _unsupported_distributed_trainable_tensors(self.logical_scene):
            raise ValueError(
                "Multi-GPU FDTD adjoint supports trainable Box material-region densities "
                "only; trainable geometry, material perturbation, circuit, and RF/port "
                "parameters have no distributed reverse core yet."
            )
        # Defense in depth for the one supported trainable channel: trainable Box
        # densities reverse only through the open/PEC standard core or the CPML
        # absorbing update. The legacy graded-sigma absorbers ("pml"/"absorber")
        # have no verified distributed reverse (require_distributed_adjoint_support
        # rejects them at reverse time), so a trainable-density scene resolving to a
        # graded absorber must fail closed here at construction rather than run a
        # forward that a later backward cannot service. The active absorber only
        # engages when the boundary declares a PML kind (see fdtd/boundary/runtime).
        from ...simulation import _scene_trainable_density_parameters

        if _scene_trainable_density_parameters(self.logical_scene):
            resolved_absorber = (
                str(self.absorber_type).lower()
                if self.logical_scene.boundary.uses_kind("pml")
                else "none"
            )
            if resolved_absorber in ("pml", "absorber"):
                raise ValueError(
                    "Multi-GPU FDTD adjoint supports trainable Box densities on open/PEC "
                    f"or CPML absorbing boundaries only; the graded-sigma absorber="
                    f"{resolved_absorber!r} has no verified distributed reverse core. "
                    "Use absorber='cpml' (or 'stablepml') for an absorbing trainable "
                    "distributed run."
                )
        from ...compiler.breakdown import scene_has_breakdown

        if scene_has_breakdown(self.logical_scene):
            raise ValueError(
                "Multi-GPU FDTD does not yet support deterministic dielectric breakdown: "
                "the per-cell state machine, conductivity scatter, and event log require an "
                "owner-write halo contract and deterministic cross-shard event ordering that "
                "the distributed runtime does not implement yet."
            )
        boundary = self.logical_scene.boundary
        if getattr(self.logical_scene, "thin_wires", ()):
            self._validate_distributed_wire_support()
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
        for region in self.logical_scene.material_regions:
            # Box density regions rasterize per shard automatically: each local
            # scene keeps the same MaterialRegion, and the density texture is
            # sampled by physical position (grid_sample against the region's global
            # center/size), so a shard's local grid selects its own sub-window with
            # no distributed density slicing. Non-Box geometries have no such
            # position-parameterized rasterization path yet.
            if not isinstance(region.geometry, Box):
                raise ValueError(
                    "Multi-GPU material density regions currently support Box geometry only; "
                    f"got {type(region.geometry).__name__}."
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
            or bool(getattr(getattr(structure, "material", None), "is_surface_impedance", False))
            for structure in self.logical_scene.structures
        ):
            raise ValueError(
                "Multi-GPU FDTD does not yet support distributed surface-impedance ownership."
            )
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

    def _validate_nccl_capabilities(self) -> None:
        """Fail-closed envelope for the one-process-per-GPU NCCL forward path.

        The NCCL coordinator drives the standard forward field solve and gathers
        the full-field DFT output to rank 0 via sized point-to-point. Per-monitor
        payload assembly across ranks (an object gather), the owner-resident
        circuit/network/wire runtimes, and the distributed adjoint reverse are not
        wired over NCCL yet, so a scene requesting any of them fails closed here
        rather than silently dropping output or running a partial coupling. The
        in-process ``transport="cuda_p2p"`` runtime covers all of them today.
        """

        from ...monitors import FieldTimeMonitor, PointMonitor
        from ...simulation import _scene_trainable_density_parameters

        # The verified NCCL adjoint driver (allow_adjoint=True) seeds a separable
        # local objective per rank: a point monitor is owned by exactly one rank and
        # a full-field DFT is gathered slab-wise, so neither needs the cross-rank
        # per-monitor payload gather the forward path lacks. Point-region monitors
        # are therefore admitted on the adjoint path; every non-point (tiled) monitor
        # still needs the unimplemented cotangent scatter and stays rejected.
        if self.logical_scene.monitors:
            non_point = [
                monitor
                for monitor in self.logical_scene.monitors
                if not (
                    self._allow_adjoint
                    and (
                        isinstance(monitor, PointMonitor)
                        or (
                            isinstance(monitor, FieldTimeMonitor)
                            and monitor.region_kind == "point"
                        )
                    )
                )
            ]
            if non_point:
                raise ValueError(
                    "Multi-GPU NCCL forward currently gathers full-field DFT output only; "
                    "per-monitor payload gather across ranks is not wired yet. Remove monitors, "
                    "use gather_fields output, or run transport='cuda_p2p'."
                )
        if (
            self.logical_scene.circuits
            or self.logical_scene.networks
            or getattr(self.logical_scene, "thin_wires", ())
            or self.logical_scene.ports
        ):
            raise ValueError(
                "Multi-GPU NCCL forward does not drive the owner-resident circuit/network/"
                "wire/port runtimes yet; run transport='cuda_p2p' for coupled scenes."
            )
        if _scene_trainable_density_parameters(self.logical_scene) and not self._allow_adjoint:
            raise ValueError(
                "Multi-GPU NCCL adjoint (trainable density) is not wired yet; run the "
                "trainable scene with transport='cuda_p2p'."
            )

    def _validate_distributed_wire_support(self) -> None:
        """Fail-closed gate for the distributed thin-wire forward envelope.

        The distributed forward supports a PEC thin-wire network on the shared
        real standard path with a non-absorbing boundary. Everything the verified
        forward does not cover is rejected here with a precise message rather than
        silently running a partial coupling:

        - a trainable wire (or any trainable scene) is rejected because the Phase 7
          distributed adjoint bridge has no wire reverse channel -- the compressed
          I/q recurrence is never checkpointed or replayed across shards;
        - a distributed-CPML boundary is rejected because wire+CPML ownership on the
          split is unverified;
        - a distributed Mur absorbing boundary is likewise rejected: the wire+Mur
          edge coupling on the split is undocumented and unverified, so it fails
          closed rather than silently running a partial absorbing coupling;
        - mixing a wire with an embedded network or a lumped circuit is rejected
          because both would claim owner-resident coordination state.
        """

        from ...simulation import (
            _scene_trainable_density_parameters,
            _scene_trainable_wire_parameters,
        )

        scene = self.logical_scene
        for wire in getattr(scene, "thin_wires", ()):
            conductor = getattr(wire, "conductor", None)
            if conductor is not None and getattr(conductor, "kind", "pec") == "finite":
                raise NotImplementedError(
                    "Multi-GPU ThinWire with a finite-conductivity (lossy) conductor is "
                    "not supported: the distributed owner runtime builds only the "
                    "lossless PEC current update and never constructs the passive "
                    "series-impedance ADE companion, so a lossy wire would silently run "
                    "as PEC across shards. The owner-reduction contract does not carry "
                    "the per-segment ADE loss state. Run a lossy thin wire on the "
                    "single-GPU FDTD path."
                )
        if (
            _scene_trainable_wire_parameters(scene)
            or _scene_trainable_density_parameters(scene)
            or _unsupported_distributed_trainable_tensors(scene)
        ):
            raise ValueError(
                "Multi-GPU ThinWire supports the forward solve only; a trainable scene "
                "carrying a ThinWire has no distributed wire reverse (the Phase 7 adjoint "
                "bridge does not checkpoint or replay wire I/q state). Run forward-only or "
                "use the single-GPU adjoint path."
            )
        if scene.boundary.uses_kind("pml"):
            raise NotImplementedError(
                "Multi-GPU ThinWire with a distributed CPML boundary has no verified "
                "wire-edge/PML ownership across the x split; use a non-absorbing boundary or "
                "run single-device FDTD."
            )
        if scene.boundary.uses_kind("mur"):
            raise NotImplementedError(
                "Multi-GPU ThinWire with a distributed Mur absorbing boundary has no verified "
                "wire-edge/boundary ownership across the x split; use a non-absorbing boundary or "
                "run single-device FDTD."
            )
        if scene.networks or scene.circuits:
            raise NotImplementedError(
                "Multi-GPU ThinWire and an embedded network or lumped circuit each claim "
                "owner-resident coordination state on one shard with no distributed ownership "
                "merge between them; run them in separate simulations."
            )

    def _validate_hardware(self) -> None:
        # preflight() binds the transport: for NCCL it initialises the process
        # group and verifies device homogeneity across ranks via an all_gather.
        self.transport.preflight()
        if self._nccl:
            # NCCL result assembly moves owned slabs with sized point-to-point,
            # not cudaMemcpyPeer, so the in-process peer-access matrix does not
            # apply. Homogeneity is enforced inside the transport preflight.
            return
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
        if getattr(self.logical_scene, "thin_wires", ()):
            # Build the full-domain single-GPU wire runtime once so the shared dt is
            # the exact joint Maxwell+wire CFL step used by the single-GPU solver;
            # any other derivation risks a dt mismatch that would break field parity.
            # The compiled network and monitors are captured to host memory and the
            # transient full-domain reference is then released.
            reference_solver.init_field()
            reference_runtime = reference_solver._wire_runtime
            cpu = torch.device("cpu")
            from .wire import move_network

            self._wire_network_cpu = move_network(reference_runtime.network, cpu)
            self._wire_monitors_cpu = tuple(
                move_wire_monitor(monitor, cpu) for monitor in reference_runtime.monitors
            )
            self._wire_cfl_metadata = {
                "cfl_limit": reference_runtime.cfl_limit,
                "wire_cfl_limit": reference_runtime.wire_cfl_limit,
                "maxwell_cfl_limit": reference_runtime.maxwell_cfl_limit,
                "dt_adjusted": reference_runtime.dt_adjusted,
            }
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

        # Every rank-local engine is built deterministically from the logical
        # scene + its partition layout. The in-process coordinator builds one
        # engine per layout; a one-process-per-GPU NCCL launch builds only its own
        # rank's engine from the identical inputs, and binds every layout to the
        # transport so rank 0 can size the field gather without a shape exchange.
        if self._nccl:
            build_layouts = (self.shard_layouts[self.rank],)
            self.transport.bind_coordinator_layouts(self.shard_layouts)
        else:
            build_layouts = self.shard_layouts
        self.shards = tuple(
            ShardEngine.build(
                self.logical_scene,
                self.scene,
                layout,
                frequency=self.frequency,
                absorber_type=self.absorber_type,
                cpml_config=self.cpml_config,
                dt=self.dt,
            )
            for layout in build_layouts
        )
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
        if not self._nccl:
            # The owner-resident circuit/network/wire runtimes span shards in one
            # process; the NCCL forward path guards them out (each rank holds a
            # single engine and cannot host the cross-shard owner state yet).
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
            if self._wire_network_cpu is not None:
                self._distributed_wire = DistributedWireRuntime.prepare(
                    network=self._wire_network_cpu,
                    monitors=self._wire_monitors_cpu,
                    partition_plan=self.partition_plan,
                    shards=self.shards,
                    dt=self.dt,
                    cfl_metadata=self._wire_cfl_metadata,
                )
        self._initialized = True

    def _prepare_outputs(self, time_steps, dft_frequency, dft_window, full_field_dft):
        spectral_enabled = any(shard.solver.observers for shard in self.shards)
        default_frequencies = dft_frequency if dft_frequency is not None else (self.frequency,)
        monitor_frequencies = tuple(
            getattr(monitor, "frequencies", None)
            for monitor in self.logical_scene.resolved_monitors()
            if not isinstance(monitor, (FieldTimeMonitor, WireMonitor))
        )
        self._observer_frequencies = (
            _merge_frequency_lists(default_frequencies, *monitor_frequencies) if spectral_enabled else ()
        )
        for shard in self.shards:
            shard.prepare_outputs(
                dft_frequency=dft_frequency,
                full_field_dft=full_field_dft,
                dft_window=dft_window,
                time_steps=time_steps,
            )
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
        if self._distributed_wire is not None:
            self._distributed_wire.prepare_outputs(
                time_steps=int(time_steps),
                window_type=dft_window,
            )

    def _overlap_active(self) -> bool:
        if self._nccl:
            # The NCCL halo exchange is a blocking batched send/recv on the
            # engine's compute stream; the coordinator runs the serialized
            # schedule (interior/boundary overlap is a follow-up on work handles).
            return False
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
        # The coordinator orchestrates one Yee step as an alternation of rank-local
        # engine phases and transport-mediated field launches: pre-magnetic
        # bookkeeping, the magnetic update (electric halo inside), magnetic
        # sources, the electric update (magnetic halo inside), electric sources,
        # and accumulation. Owner-resident circuit/network/wire runtimes are
        # driven between the phases at their fixed schedule positions.
        time_value = n * self.dt
        for shard in self.shards:
            shard.advance_pre_magnetic()

        if overlap_active:
            self._advance_magnetic_overlapped()
        else:
            self._advance_magnetic_serialized()

        for shard in self.shards:
            shard.apply_magnetic_sources_and_corrections(time_value)

        # Wire EMF is sampled from the pre-update E field and the compressed I/q
        # recurrence is advanced on the owner shard before the electric update,
        # matching the single-GPU schedule (sample -> update -> E update -> deposit).
        if self._distributed_wire is not None:
            self._distributed_wire.apply_sample_and_update()

        if overlap_active:
            self._advance_electric_overlapped()
        else:
            self._advance_electric_serialized(time_value)

        # Deposit the advanced wire current onto the owned E edges immediately after
        # the electric update, before any electric source injection, matching the
        # single-GPU field-update block ordering.
        if self._distributed_wire is not None:
            self._distributed_wire.apply_deposit()

        for shard in self.shards:
            shard.apply_electric_sources(time_value)

        if self._distributed_circuit is not None:
            self._distributed_circuit.apply()
        if self._distributed_network is not None:
            self._distributed_network.apply()

        for shard in self.shards:
            shard.accumulate(n)

        if self._distributed_wire is not None:
            self._distributed_wire.accumulate_monitors(n)

    def _synchronize_all(self) -> None:
        for shard in self.shards:
            shard.compute_stream.synchronize()
            shard.communication_stream.synchronize()

    def _gather_component(self, component: str, local_values: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self.transport.gather_component_slabs(
            self.shards,
            component,
            local_values,
            result_device=self.device,
            global_nx=self.Nx,
        )

    def _collect_output(self) -> dict[str, Any] | None:
        shard_monitor_payloads, local_fields, frequency_metadata = (
            self.transport.gather_monitor_payloads(self.shards)
        )

        monitors = merge_sharded_monitor_payloads(
            (monitor.name for monitor in self.logical_scene.resolved_monitors()),
            shard_monitor_payloads,
            result_device=self.device,
            shard_layouts=self.shard_layouts,
            physical_bounds=self.logical_scene.domain.bounds,
        )

        output: dict[str, Any] = {}
        if self.parallel.gather_fields:
            # gather_component_slabs is a collective on the NCCL path: every rank
            # contributes its owned slab and only the result root receives the
            # global tensor (None elsewhere). The in-process transport always
            # returns the tensor, so this loop is unchanged for cuda_p2p.
            if local_fields:
                for name, values in local_fields.items():
                    gathered = self._gather_component(name, tuple(values))
                    if gathered is not None:
                        output[name] = gathered
            else:
                for name in ("Ex", "Ey", "Ez"):
                    values = tuple(getattr(shard.solver, name) for shard in self.shards)
                    gathered = self._gather_component(name, values)
                    if gathered is not None:
                        output[name] = gathered
        if self._distributed_wire is not None:
            wire_monitors = self._distributed_wire.finalize(self.device)
            if wire_monitors:
                monitors = dict(monitors)
                monitors.update(wire_monitors)
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
        # On a one-process-per-GPU launch only the result root assembles output;
        # non-root ranks have already contributed their slabs to the collective.
        if not self._result_root:
            return None
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
        if self._nccl and float(shutoff) > 0.0:
            # reduce_owned_energy is a primitive-tested NCCL collective, but the
            # coordinator's shutoff bookkeeping (peak tracking on the result
            # device, cross-rank break lockstep) is not reconciled for the
            # per-rank shape yet; fail closed rather than diverge silently.
            raise ValueError(
                "Multi-GPU NCCL forward does not support field shutoff (shutoff>0) yet; "
                "run a fixed step count or use transport='cuda_p2p'."
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
                energy = self.transport.reduce_owned_energy(self.shards)
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
            shard.finalize_after_solve(
                time_steps=int(time_steps),
                elapsed=self.last_solve_elapsed_s,
                shutoff_triggered=self._shutoff_triggered,
                shutoff_step=self._shutoff_step,
            )

        if self._shutoff_triggered and self._distributed_wire is not None:
            self._distributed_wire.complete_normalization(int(time_steps))

        output = self._collect_output()
        for device in self.devices:
            torch.cuda.synchronize(device)
        self._peak_memory_including_gather = {
            str(device): int(torch.cuda.max_memory_allocated(device))
            for device in self.devices
        }
        self._parallel_stats = self._build_parallel_stats(time_steps, overlap_active)
        return output

    def teardown(self) -> None:
        """Release transport resources (NCCL process group) deterministically."""

        self.transport.teardown()

    def _build_parallel_stats(self, time_steps: int, overlap_active: bool) -> dict[str, Any]:
        steps_run = (self._shutoff_step + 1) if self._shutoff_triggered else int(time_steps)
        gathered = self.transport.gather_stats(self.shards)
        halo_bytes_per_step = gathered["halo_bytes_per_step"]
        # A rank-local transport (one-process-per-GPU NCCL) cannot see both sides
        # of any halo, so it reports ``halo_bytes_per_step`` as ``None`` and marks
        # its partitions/peak-memory snapshot as rank-local rather than global.
        # Propagate that honestly instead of coercing ``None`` to a misleading 0.
        stats_are_rank_local = bool(gathered.get("rank_local", False))
        stats = {
            "stats_rank_local": stats_are_rank_local,
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
            "partitions": gathered["partitions"],
            "halo_bytes_per_step": (
                None if halo_bytes_per_step is None else int(halo_bytes_per_step)
            ),
            "halo_bytes_total": (
                None if halo_bytes_per_step is None else int(halo_bytes_per_step * steps_run)
            ),
            "peak_memory_bytes": gathered["peak_memory_bytes"],
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
        if self._distributed_wire is not None:
            stats["thin_wire"] = self._distributed_wire.stats(steps_run=steps_run)
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

    def wire_checkpoint_tensors(self) -> dict[str, torch.Tensor]:
        """Return live owner-GPU thin-wire I/q state without gathering fields."""

        if self._distributed_wire is None:
            return {}
        return self._distributed_wire.checkpoint_tensors()

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
