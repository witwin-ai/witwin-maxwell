"""Rank-local shard engine for the distributed FDTD coordinator.

``ShardEngine`` is the rank-local half of the x-slab decomposition. It owns
everything one rank does on its own device: the deterministic local-scene build
from ``FDTDPartitionPlan`` metadata, the local :class:`FDTD` solver, its Yee
storage/streams/events/receive halos, and the rank-local per-step phases
(pre-magnetic bookkeeping, magnetic/electric source injection, DFT/observer
accumulation), plus the rank-local output aggregation (owned-energy scalar,
local monitor payloads).

The build is *pure deterministic metadata*: every rank derives its own layout,
local grid, and local solver identically from ``(logical_scene, global_prepared,
layout)`` with no cross-rank communication, so a one-process-per-GPU launcher can
construct its own engine independently. The single-process coordinator
(:class:`~witwin.maxwell.fdtd.distributed.solver.DistributedFDTD`) simply builds
one engine per shard layout.

Cross-rank work -- halo exchange, scalar reductions, field/monitor/stat gathers --
lives entirely in the transport primitives the coordinator drives, never here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ...monitors import (
    DipoleEmissionMonitor,
    FieldTimeMonitor,
    FinitePlaneMonitor,
    MediumMonitor,
    ModeMonitor,
    PermittivityMonitor,
    PlaneMonitor,
    PointMonitor,
    WireMonitor,
)
from ...scene import Domain, GridSpec, Scene
from ..excitation import (
    inject_electric_surface_source_terms,
    inject_magnetic_surface_source_terms,
)
from ..runtime import stepping
from ..solver import FDTD
from .source_corrections import correct_ideal_point_ex_control_volume
from .sources import crop_solver_source_terms_to_owned_x


# ---------------------------------------------------------------------------
# Deterministic rank-local scene-build helpers (no cross-rank state).
# ---------------------------------------------------------------------------


def _physical_axis_nodes(prepared_scene, axis: str) -> np.ndarray:
    nodes = np.asarray(getattr(prepared_scene, f"{axis}_nodes64"), dtype=np.float64)
    low = int(prepared_scene.pml_thickness_for_face(axis, "low"))
    high = int(prepared_scene.pml_thickness_for_face(axis, "high"))
    stop = nodes.size - high if high else nodes.size
    return np.array(nodes[low:stop], copy=True)


def _owns_global_x(layout, prepared_scene, x: float) -> bool:
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


def _local_monitors(scene: Scene, layout, global_prepared):
    selected = []
    for monitor in scene.monitors:
        if isinstance(monitor, WireMonitor):
            # Wire monitors are owned and finalized by DistributedWireRuntime on the
            # single state-owner shard, not accumulated per field shard.
            continue
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
    layout,
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
        # Thin wires are coupled to Yee edges spread across shards, so the
        # per-shard local solver must not build its own single-GPU wire runtime.
        # DistributedWireRuntime owns the compressed I/q state on one shard and
        # drives the distributed sampling/deposition explicitly.
        thin_wires=(),
        device=str(layout.device),
        symmetry=(None, logical_scene.symmetry[1], logical_scene.symmetry[2]),
    )


@dataclass
class ShardEngine:
    """Rank-local FDTD shard: local solver, storage, and per-step phases.

    Holds every field the halo transport and the owner-resident circuit/network/
    wire runtimes read (``solver``, ``layout``, ``device``, the compute/comm
    streams, the halo-exchange events, and the persistent magnetic receive
    halos), so those consumers duck-type an engine exactly as they did the
    previous shard record.
    """

    rank: int
    device: torch.device
    layout: object
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

    # -- construction ------------------------------------------------------

    @classmethod
    def build(
        cls,
        logical_scene: Scene,
        global_prepared,
        layout,
        *,
        frequency: float,
        absorber_type: str,
        cpml_config: dict,
        dt: float,
    ) -> "ShardEngine":
        """Deterministically build one rank's engine from partition metadata.

        Pure per-rank derivation: the local grid, boundary, monitors, and solver
        are a function of ``(logical_scene, global_prepared, layout)`` alone, so
        every rank -- in-process or one-process-per-GPU -- constructs an
        identical layout with no cross-rank communication.
        """

        device = torch.device(layout.device)
        with torch.cuda.device(device):
            torch.cuda.reset_peak_memory_stats(device)
            physical_x = _physical_axis_nodes(global_prepared, "x")
            local_scene = _build_local_scene(
                logical_scene, global_prepared, layout, physical_x
            )
            local_solver = FDTD(
                local_scene,
                frequency=frequency,
                absorber_type=absorber_type,
                cpml_config=cpml_config,
            )
            local_solver.dt = dt
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
            if getattr(local_solver, "gyromagnetic_enabled", False):
                # The shard phases never advance/apply the magnetization-ADE hooks,
                # and the rank-local build wires an independent gyromagnetic layout
                # per shard-local grid whose overlap slice drops the top plane at
                # rank seams (physical interior planes, not domain boundaries). A
                # joint solve would silently simulate a reciprocal medium with
                # uncorrected seam planes. Fail closed until seam-aware distributed
                # ferrite support exists (contract boundary 8: rejected until Phase 4).
                raise NotImplementedError(
                    "Multi-GPU distributed FDTD does not support GyromagneticFerrite media: the "
                    "shard phases do not run the magnetization-ADE hooks and the shard-local "
                    "gyromagnetic layout has no rank-seam handling, so a joint solve would silently "
                    "drop the non-reciprocal gyrotropy. Run the ferrite scene on a single device."
                )
            crop_solver_source_terms_to_owned_x(local_solver, layout)
            correct_ideal_point_ex_control_volume(local_solver, layout, global_prepared)
            compute_stream = torch.cuda.Stream(device=device)
            communication_stream = torch.cuda.Stream(device=device, priority=-1)
            halo_hy = local_solver.Hy[0] if layout.rank > 0 else None
            compute_stream.wait_stream(torch.cuda.current_stream(device))
            halo_hz = local_solver.Hz[0] if layout.rank > 0 else None
            engine = cls(
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
            engine._validate_layout()
        return engine

    def _validate_layout(self) -> None:
        layout = self.layout
        solver = self.solver
        cell_owned = layout.storage_cell_owned
        node_owned = layout.storage_node_owned
        if cell_owned.stop > solver.Nx - 1 or node_owned.stop > solver.Nx:
            raise RuntimeError(
                f"Rank {self.rank} padded storage is smaller than its declared owned slices."
            )
        expected_low_pad = 0 if self.rank == 0 else 1
        if cell_owned.start != expected_low_pad or node_owned.start != expected_low_pad:
            raise RuntimeError(
                f"Rank {self.rank} has invalid padded low ownership: "
                f"cell={cell_owned}, node={node_owned}."
            )

    # -- rank-local per-step phases ----------------------------------------

    def advance_pre_magnetic(self) -> None:
        """Modulation clock + magnetic ADE advance, before the H update."""

        solver = self.solver
        with torch.cuda.device(self.device), torch.cuda.stream(self.compute_stream):
            if solver.modulation_enabled:
                solver.fdtd_module.advanceModulationTime3D(
                    ModulationTime=solver._modulation_time,
                    dt=solver.dt,
                ).launchRaw()
            solver._advance_magnetic_dispersive_state()

    def apply_magnetic_sources_and_corrections(self, time_value: float) -> None:
        solver = self.solver
        with torch.cuda.device(self.device), torch.cuda.stream(self.compute_stream):
            if solver._magnetic_source_terms:
                inject_magnetic_surface_source_terms(solver, time_value=time_value)
            solver._apply_magnetic_dispersive_corrections()
            solver._advance_dispersive_state()
            if solver.nonlinear_enabled:
                solver._update_nonlinear_electric_coefficients()
            stepping.capture_aniso_conduction_currents(solver)

    def apply_electric_sources(self, time_value: float) -> None:
        solver = self.solver
        with torch.cuda.device(self.device), torch.cuda.stream(self.compute_stream):
            if solver.full_aniso_enabled:
                stepping.apply_full_aniso_corrections(solver)
                stepping.apply_full_aniso_conduction(solver)
            if getattr(solver, "_surface_impedance", None) is not None:
                raise RuntimeError(
                    "Distributed surface-impedance ownership is not enabled."
                )
            if solver._electric_source_terms:
                inject_electric_surface_source_terms(
                    solver, time_value=time_value + 0.5 * float(solver.dt)
                )
            if solver._source_terms:
                solver.add_source(time_value=time_value)

    def accumulate(self, n: int) -> None:
        solver = self.solver
        with torch.cuda.device(self.device), torch.cuda.stream(self.compute_stream):
            solver._apply_dispersive_corrections()
            stepping.enforce_pec_boundaries(solver)
            stepping.apply_mur_boundaries(solver)
            solver.accumulate_dft(n)
            solver.accumulate_observers(n)
            solver.accumulate_time_observers(n)

    # -- rank-local output aggregation -------------------------------------

    def owned_electric_energy(self) -> torch.Tensor:
        """Owned electric energy scalar; records ``electric_ready`` for the reduce.

        The scalar is computed on this rank's compute stream and the event is
        recorded so the transport reduction can order its cross-device read after
        the compute completes, exactly as the shutoff energy reduce required.
        """

        solver = self.solver
        cs = self.layout.storage_cell_owned
        ns = self.layout.storage_node_owned
        with torch.cuda.device(self.device), torch.cuda.stream(self.compute_stream):
            local = (
                (solver.eps_Ex[cs] * solver.Ex[cs] * solver.Ex[cs]).sum()
                + (solver.eps_Ey[ns] * solver.Ey[ns] * solver.Ey[ns]).sum()
                + (solver.eps_Ez[ns] * solver.Ez[ns] * solver.Ez[ns]).sum()
            )
            self.electric_ready.record(self.compute_stream)
        return local

    def prepare_outputs(
        self,
        *,
        dft_frequency,
        full_field_dft: bool,
        dft_window: str,
        time_steps: int,
    ) -> None:
        solver = self.solver
        with torch.cuda.device(self.device), torch.cuda.stream(self.compute_stream):
            if dft_frequency is not None and full_field_dft:
                solver.enable_dft(dft_frequency, window_type=dft_window, end_step=time_steps)
            else:
                solver.dft_enabled = False
                solver._dft_entries = []
                solver._sync_dft_legacy_state()
            observer_frequency = (
                dft_frequency if dft_frequency is not None else solver.source_frequency
            )
            if solver.observers:
                solver._prepare_observers(observer_frequency, dft_window, time_steps)
            if solver.time_observers:
                solver._prepare_time_observers(time_steps)
            solver._shutoff_triggered = False
            solver._shutoff_step = None

    def collect_local_monitor_payload(self):
        """Return this rank's ``(monitor payloads, DFT E fields, frequencies)``.

        DFT E fields are keyed by name (only the ones this shard computed); the
        transport gather concatenates them across ranks in rank order. Frequency
        metadata is returned for the transport to check cross-rank consistency.
        """

        solver = self.solver
        shard_monitors: dict = {}
        local_fields: dict = {}
        frequency_metadata = None
        if solver.dft_enabled:
            local = solver.get_frequency_solution(all_frequencies=True)
            metadata = local.get("frequencies")
            if metadata is not None:
                if isinstance(metadata, torch.Tensor):
                    frequency_metadata = tuple(
                        float(value) for value in metadata.detach().cpu().tolist()
                    )
                else:
                    frequency_metadata = tuple(float(value) for value in metadata)
            for name, tensor in local.items():
                if name in {"Ex", "Ey", "Ez"}:
                    local_fields[name] = tensor
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
                        f"{self.rank}."
                    )
                shard_monitors[name] = payload
        return shard_monitors, local_fields, frequency_metadata

    def finalize_after_solve(
        self,
        *,
        time_steps: int,
        elapsed,
        shutoff_triggered: bool,
        shutoff_step,
    ) -> None:
        solver = self.solver
        solver.last_solve_elapsed_s = elapsed
        solver._shutoff_triggered = shutoff_triggered
        solver._shutoff_step = shutoff_step
        if shutoff_triggered:
            with torch.cuda.device(self.device):
                stepping._complete_spectral_normalization(solver, int(time_steps))
        if solver.dft_enabled:
            solver._sync_dft_legacy_state()
        if solver.observers_enabled:
            solver._sync_observer_legacy_state()
        self.peak_memory_bytes = int(torch.cuda.max_memory_allocated(self.device))
