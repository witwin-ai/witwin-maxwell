"""Distributed joint-solve adjoint (Phase 7, slices S1-S3).

This module holds both the internal reverse building blocks and the public-facing
distributed gradient bridge:

* per-shard forward checkpoint capture keyed by the partition manifest,
* a distributed forward *replay* that advances every shard as two explicit
  half-steps with the forward Yee halos copied between them, reproducing the
  native distributed forward run's owned states to within floating-point
  reduction-order drift (the replay runs the torch reference update while the
  native forward runs fused CUDA kernels, so the two reduce in a different
  order; the decomposition itself adds no algorithmic error), and
* :class:`_DistributedFDTDGradientBridge` (S3), which drives the forward with
  checkpoints, then a transposed reverse step per forward step -- Phase 1 on every
  shard, the transposed magnetic halo, Phase 2, the transposed electric halo,
  Phase 3, then the per-shard source-term eps gradient -- and gathers the per-shard
  grad_eps owned slices into a global tensor so the existing single-GPU material
  pullback runs once on the logical scene.

Only the pure real *standard* (open-boundary) configuration is supported. Every
unsupported medium/boundary/coupling is rejected fail-closed by
:func:`require_distributed_adjoint_support` before any checkpoint is taken, and
tiled-monitor-seeded objectives by
:func:`require_distributed_adjoint_objective_support`, so the reverse never runs a
configuration whose transposed cores or seed routing have not been verified. The
public entry point is :func:`run_distributed_fdtd_with_gradient_bridge`, reached
from ``Simulation`` for a trainable Box-density multi-GPU scene.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..adjoint.core import (
    _accumulate_source_term_gradients,
    _build_spectral_weight_schedule,
    _checkpoint_stride,
    _forward_electric_fields_cpml,
    _forward_electric_fields_standard,
    _forward_magnetic_fields,
    _forward_magnetic_fields_cpml,
    _monitor_template_is_point,
    _prepare_forward_pack,
    _resolved_source_term_lists,
)
from ..adjoint.dispatch import _NATIVE_REVERSE_LABELS, _ReverseBackend
from ..adjoint.native import (
    reverse_cpml_phase_electric,
    reverse_cpml_phase_magnetic,
    reverse_phase1_electric_to_h,
    reverse_phase2_magnetic_to_e,
    reverse_phase3_decay,
)
from ..adjoint.profiler import _ReverseStepResult
from ..adjoint.reverse_common import (
    allocate_cpml_reverse_context,
    dynamic_electric_curls,
)
from ..adjoint.seeds import (
    _apply_seed_runtime,
    _build_output_seeds,
    _schedule_to_tensor_pack,
)
from ..checkpoint import (
    FDTDCheckpointState,
    capture_checkpoint_state,
    clone_checkpoint_tensors,
)
from ..material_pullback import pullback_material_input_gradients
from .capacity import require_gather_capacity

_ELECTRIC_NAMES = ("Ex", "Ey", "Ez")
_MAGNETIC_NAMES = ("Hx", "Hy", "Hz")
# The 12 CPML memory (psi) fields carried alongside the six Yee fields on the
# CPML distributed reverse. psi_e_* are advanced by the electric half-step and
# read by the CPML electric reverse; psi_h_* are advanced by the magnetic
# half-step. The adjoint state carries a psi cotangent for each (seeded to zero;
# monitor seeds inject only E/H), and the CPML reverse kernels assign the pre-step
# psi adjoint. No psi is exchanged across the x halo: the partition pins every
# x-CPML region to the outer shards, so internal-face psi is inactive by
# construction (see the S4 audit).
_CPML_PSI_NAMES = (
    "psi_ex_y",
    "psi_ex_z",
    "psi_ey_x",
    "psi_ey_z",
    "psi_ez_x",
    "psi_ez_y",
    "psi_hx_y",
    "psi_hx_z",
    "psi_hy_x",
    "psi_hy_z",
    "psi_hz_x",
    "psi_hz_y",
)

# Shard-solver attributes that place the shard outside the verified reverse
# class. Each maps to the reason surfaced by the guard.
#
# Absorbing boundaries are handled separately by ``active_absorber_type`` (below)
# rather than a single boolean, because the verified envelope now covers two
# boundary regimes: the open-boundary standard update (active_absorber_type ==
# "none") AND the CPML absorbing update (active_absorber_type in {"cpml",
# "stablepml"}, i.e. ``uses_cpml``). The CPML distributed reverse is gated by the
# S4 parity/finite-difference gates on an x-CPML dielectric scene and relies on
# the partition pinning every x-CPML region to the outer shards (asserted by
# :func:`_assert_x_pml_pinned_to_outer_shards`). The legacy graded-sigma absorber
# families -- "pml"/"absorber" -- have no verified distributed reverse core and
# stay rejected.
_UNSUPPORTED_SHARD_FLAGS = (
    ("complex_fields_enabled", "complex (Bloch split-field) state"),
    ("dispersive_enabled", "electric dispersive (ADE) media"),
    ("magnetic_dispersive_enabled", "magnetic dispersive (ADE) media"),
    ("nonlinear_enabled", "nonlinear media"),
    ("conductive_enabled", "static-conductive media"),
    ("full_aniso_enabled", "full off-diagonal anisotropy"),
    ("tfsf_enabled", "total-field/scattered-field injection"),
    ("modulation_enabled", "time-modulated media"),
    ("surface_impedance_enabled", "surface-impedance boundary media"),
)


def _assert_x_pml_pinned_to_outer_shards(distributed) -> None:
    """Fail closed unless every x-CPML region lives entirely on an outer shard.

    The per-shard CPML reverse is interface-correct only because the partition
    excludes x-PML from the split: rank 0 owns all low x-PML cells and the last
    rank owns all high x-PML cells, so every internal x-face carries inactive
    x-CPML (``c == 0``) and the cross-interface curl coupling flows entirely
    through the existing Yee field halos -- no psi halo is required (S4 audit,
    PART A). This is guaranteed by construction today; assert it here so a future
    partition change cannot silently break the invariant and run an unverified
    CPML reverse. The check is vacuous when there is no x-PML (``low == high ==
    0``), so it is safe to run on every distributed adjoint configuration.
    """

    plan = getattr(distributed, "partition_plan", None)
    if plan is None:
        return
    layouts = tuple(plan.shard_layouts)
    if not layouts:
        return
    low = int(getattr(plan, "low_pml_cells", 0))
    high = int(getattr(plan, "high_pml_cells", 0))
    cell_count = int(getattr(plan, "cell_count"))
    last_rank = len(layouts) - 1
    for layout in layouts:
        start = int(layout.global_cell_owned.start)
        stop = int(layout.global_cell_owned.stop)
        if low > 0 and start < low and int(layout.rank) != 0:
            raise RuntimeError(
                "Distributed CPML reverse invariant violated: low x-PML cells "
                f"[0, {low}) are owned by shard {layout.rank} (owns cells "
                f"[{start}, {stop})), not rank 0. The x-CPML region must be pinned "
                "to the outer shards for the per-shard reverse to be interface-correct."
            )
        if high > 0 and stop > cell_count - high and int(layout.rank) != last_rank:
            raise RuntimeError(
                "Distributed CPML reverse invariant violated: high x-PML cells "
                f"[{cell_count - high}, {cell_count}) are owned by shard "
                f"{layout.rank} (owns cells [{start}, {stop})), not the last rank "
                f"{last_rank}. The x-CPML region must be pinned to the outer shards."
            )


def require_distributed_adjoint_support(distributed) -> None:
    """Reject any distributed configuration outside the verified reverse class.

    Fail-closed guard: the distributed checkpoint/replay/reverse path is validated
    for the pure real standard (open/PEC-boundary) update and the CPML absorbing
    update (``uses_cpml``, i.e. ``active_absorber_type`` in {"cpml", "stablepml"}).
    The legacy graded-sigma absorbers ("pml"/"absorber"), and dispersive,
    nonlinear, conductive, anisotropic, complex/Bloch, TFSF, modulated, and any
    coupled port/circuit/network configuration are rejected here rather than
    producing an unverified gradient. On the CPML path the x-PML pinning invariant
    is asserted so the per-shard reverse stays interface-correct.
    """

    if not getattr(distributed, "_initialized", False):
        raise RuntimeError(
            "Distributed adjoint support can only be validated after init_field()."
        )
    for shard in distributed.shards:
        solver = shard.solver
        active_absorber = str(getattr(solver, "active_absorber_type", "none")).lower()
        if active_absorber not in ("none", "cpml", "stablepml"):
            raise ValueError(
                "Distributed FDTD checkpoint/replay supports only the pure real "
                "standard open-boundary update or the CPML absorbing update; shard "
                f"{shard.rank} runs a {active_absorber!r} absorbing boundary. Use "
                "open/PEC or CPML boundaries for the trainable distributed path."
            )
        for attribute, reason in _UNSUPPORTED_SHARD_FLAGS:
            if bool(getattr(solver, attribute, False)):
                raise ValueError(
                    "Distributed FDTD checkpoint/replay supports only the pure real "
                    f"standard or CPML update; shard {shard.rank} uses {reason}."
                )
        if getattr(solver, "_magnetic_source_terms", ()):
            raise ValueError(
                "Distributed FDTD checkpoint/replay does not yet support magnetic "
                f"surface source terms; shard {shard.rank} declares them."
            )
    if getattr(distributed, "_distributed_circuit", None) is not None:
        raise ValueError(
            "Distributed FDTD checkpoint/replay does not yet support embedded "
            "circuit/port/network coupling."
        )
    _assert_x_pml_pinned_to_outer_shards(distributed)


@dataclass(frozen=True)
class DistributedCheckpoint:
    """A partition-keyed snapshot of every shard's forward state at one step.

    ``partition_signature`` pins the ownership geometry the states were captured
    against so a replay cannot be started from a checkpoint that belongs to a
    different decomposition.
    """

    step: int
    partition_signature: tuple
    states: dict[int, FDTDCheckpointState]


def _partition_signature(distributed) -> tuple:
    return tuple(
        (
            int(layout.rank),
            str(layout.device),
            int(layout.global_cell_owned.start),
            int(layout.global_cell_owned.stop),
            int(layout.global_node_owned.start),
            int(layout.global_node_owned.stop),
            int(layout.storage_cell_owned.start),
            int(layout.storage_cell_owned.stop),
            int(layout.storage_node_owned.start),
            int(layout.storage_node_owned.stop),
        )
        for layout in distributed.shard_layouts
    )


def capture_distributed_checkpoint(distributed, step: int) -> DistributedCheckpoint:
    """Capture every shard's forward state on its own device.

    The per-shard capture clones exactly the persistent padded field storage
    (owned cells plus the single ghost plane), never a transient receive-halo
    staging buffer, because the magnetic receive halos are views into that same
    padded storage and are refreshed deterministically on replay.
    """

    require_distributed_adjoint_support(distributed)
    states: dict[int, FDTDCheckpointState] = {}
    for shard in distributed.shards:
        with torch.cuda.device(shard.device):
            states[shard.rank] = capture_checkpoint_state(shard.solver, step)
    return DistributedCheckpoint(
        step=int(step),
        partition_signature=_partition_signature(distributed),
        states=states,
    )


def _synchronize(devices) -> None:
    for device in devices:
        torch.cuda.synchronize(device)


def replay_distributed_segment(
    distributed,
    checkpoint: DistributedCheckpoint,
    start_step: int,
    end_step: int,
    *,
    mid_magnetic_out=None,
):
    """Replay one checkpoint segment across all shards in lockstep.

    Each step advances every shard as two explicit halves with the forward Yee
    halos copied between them, mirroring the distributed serialized forward
    schedule (electric halo -> magnetic half -> magnetic halo -> electric half).
    Returns ``{rank: [state_0, ..., state_(end-start)]}``. When ``mid_magnetic_out``
    is a dict of per-rank lists, the post-halo mid-step H of each replayed step is
    appended per rank so the reverse pass can read valid interface forward H for
    the Phase-2 eps-gradient terms.

    Owned states reproduce the native distributed forward run for the pure real
    standard configuration to within floating-point reduction-order drift, not
    bitwise: the replay runs the torch reference update while the native forward
    runs fused CUDA bounded kernels, so the two reduce in a different order. The
    replay-parity test tolerance-gates this at rtol=1e-5/atol=1e-7, ~200x above
    the measured ~5e-10 owned-state drift and ~500x below the smallest halo-bug
    error. The ghost planes carry the neighbour's owned value after each halo
    copy exactly as the forward serialized path does.
    """

    require_distributed_adjoint_support(distributed)
    if checkpoint.partition_signature != _partition_signature(distributed):
        raise RuntimeError(
            "Distributed replay checkpoint was captured against a different partition."
        )
    shards = distributed.shards
    devices = distributed.devices
    dt = float(distributed.dt)

    resolved = {}
    current = {}
    for shard in shards:
        with torch.cuda.device(shard.device):
            resolved[shard.rank] = _resolved_source_term_lists(
                shard.solver,
                shard.solver.eps_Ex,
                shard.solver.eps_Ey,
                shard.solver.eps_Ez,
            )
            current[shard.rank] = clone_checkpoint_tensors(checkpoint.states[shard.rank])

    uses_cpml = _distributed_uses_cpml(distributed)
    # The reverse reads the pre-step forward psi_e alongside E/H on the CPML
    # branch, and the CPML reverse context allocates a pre-step adjoint per
    # forward-state key, so the trajectory carries the full 18-field CPML state
    # (six Yee + twelve psi). On the standard branch it carries the six Yee fields.
    state_names = _CPML_STATE_NAMES if uses_cpml else _STANDARD_STATE_NAMES

    trajectories = {
        shard.rank: [{name: current[shard.rank][name].clone() for name in state_names}]
        for shard in shards
    }

    with torch.no_grad():
        for step_index in range(start_step, end_step):
            time_value = step_index * dt

            # (1) Electric halo, then (2) magnetic half on every shard. The CPML
            # half additionally advances psi_h (a same-x-slice recurrence, no
            # halo); the standard half advances only the fields.
            distributed.transport.forward_electric_halo(shards, current)
            _synchronize(devices)
            for shard in shards:
                with torch.cuda.device(shard.device):
                    if uses_cpml:
                        magnetic, psi_h = _forward_magnetic_fields_cpml(
                            shard.solver,
                            current[shard.rank],
                            time_value=time_value,
                            resolved_source_terms=resolved[shard.rank],
                        )
                        current[shard.rank].update(psi_h)
                    else:
                        magnetic = _forward_magnetic_fields(
                            shard.solver,
                            current[shard.rank],
                            time_value=time_value,
                            resolved_source_terms=resolved[shard.rank],
                        )
                    for name in _MAGNETIC_NAMES:
                        current[shard.rank][name] = magnetic[name]
            _synchronize(devices)

            # (3) Magnetic halo; the mid-step H is valid on owned + ghost after it.
            distributed.transport.forward_magnetic_halo(shards, current)
            _synchronize(devices)
            if mid_magnetic_out is not None:
                for shard in shards:
                    with torch.cuda.device(shard.device):
                        mid_magnetic_out.setdefault(shard.rank, []).append(
                            {name: current[shard.rank][name].clone() for name in _MAGNETIC_NAMES}
                        )

            # (4) Electric half on every shard (CPML additionally advances psi_e).
            for shard in shards:
                with torch.cuda.device(shard.device):
                    magnetic_fields = {
                        name: current[shard.rank][name] for name in _MAGNETIC_NAMES
                    }
                    if uses_cpml:
                        electric, psi_e = _forward_electric_fields_cpml(
                            shard.solver,
                            current[shard.rank],
                            magnetic_fields,
                            time_value=time_value,
                            eps_ex=shard.solver.eps_Ex,
                            eps_ey=shard.solver.eps_Ey,
                            eps_ez=shard.solver.eps_Ez,
                            resolved_source_terms=resolved[shard.rank],
                        )
                        current[shard.rank].update(psi_e)
                    else:
                        electric = _forward_electric_fields_standard(
                            shard.solver,
                            current[shard.rank],
                            magnetic_fields,
                            time_value=time_value,
                            eps_ex=shard.solver.eps_Ex,
                            eps_ey=shard.solver.eps_Ey,
                            eps_ez=shard.solver.eps_Ez,
                            resolved_source_terms=resolved[shard.rank],
                        )
                    for name in _ELECTRIC_NAMES:
                        current[shard.rank][name] = electric[name]
            _synchronize(devices)

            for shard in shards:
                trajectories[shard.rank].append(
                    {
                        name: current[shard.rank][name].clone()
                        for name in state_names
                    }
                )

    return trajectories


# ---------------------------------------------------------------------------
# Slice S3: distributed joint-solve gradient bridge (standard real path).
# ---------------------------------------------------------------------------

_STANDARD_STATE_NAMES = _ELECTRIC_NAMES + _MAGNETIC_NAMES
_CPML_STATE_NAMES = _STANDARD_STATE_NAMES + _CPML_PSI_NAMES
_CELL_COMPONENTS = frozenset(("Ex", "Hy", "Hz"))
_STANDARD_BACKEND_LABEL = _NATIVE_REVERSE_LABELS[_ReverseBackend.STANDARD]
_CPML_BACKEND_LABEL = _NATIVE_REVERSE_LABELS[_ReverseBackend.CPML]


def _reverse_state_names(distributed) -> tuple:
    """The Yee (+psi on CPML) state names the distributed reverse carries."""
    if distributed.shards and bool(getattr(distributed.shards[0].solver, "uses_cpml", False)):
        return _CPML_STATE_NAMES
    return _STANDARD_STATE_NAMES


def _distributed_uses_cpml(distributed) -> bool:
    return bool(distributed.shards) and bool(
        getattr(distributed.shards[0].solver, "uses_cpml", False)
    )


def require_distributed_adjoint_objective_support(distributed) -> None:
    """Reject objectives seeded from tiled plane/flux/mode monitors.

    Baseline distributed seed routing supports point-monitor spectra (single owner)
    and full-field DFT (owned x-slices scattered from the gathered global grad). A
    plane / flux / mode / closed-surface / diffraction monitor is stitched across
    shards in the forward, so its cotangent would have to be scattered by the same
    owned intervals -- the follow-up seed-scatter slice. Until that lands, an
    objective that reads a tiled monitor is rejected here before any reverse work.
    """

    from ...monitors import (
        ClosedSurfaceMonitor,
        DiffractionMonitor,
        FieldTimeMonitor,
        FinitePlaneMonitor,
        FluxMonitor,
        FluxTimeMonitor,
        ModeMonitor,
        PlaneMonitor,
        PointMonitor,
    )

    _TILED = (
        PlaneMonitor,
        FinitePlaneMonitor,
        ModeMonitor,
        FluxMonitor,
        FluxTimeMonitor,
        ClosedSurfaceMonitor,
        DiffractionMonitor,
    )
    for monitor in distributed.logical_scene.resolved_monitors():
        if isinstance(monitor, _TILED):
            raise ValueError(
                "Distributed FDTD adjoint objectives currently support point-monitor "
                "spectra and full-field DFT only; tiled plane/flux/mode monitor "
                f"cotangent scatter is not enabled yet ({type(monitor).__name__} "
                f"{getattr(monitor, 'name', '?')!r})."
            )
        if isinstance(monitor, FieldTimeMonitor) and monitor.region_kind != "point":
            raise ValueError(
                "Distributed FDTD adjoint objectives support point FieldTimeMonitor "
                f"regions only; {monitor.name!r} is a {monitor.region_kind} region."
            )
        if not isinstance(monitor, (PointMonitor, FieldTimeMonitor)):
            # Material monitors etc. are analytic and never seeded; the forward
            # guard already rejects the unsupported ones. Anything reaching here
            # that is neither point nor field-time is out of the seeded set.
            from ...monitors import MediumMonitor, PermittivityMonitor

            if not isinstance(monitor, (MediumMonitor, PermittivityMonitor)):
                raise ValueError(
                    "Distributed FDTD adjoint objectives support point-monitor spectra "
                    f"and full-field DFT only; {type(monitor).__name__} is unsupported."
                )


def _owned_field_slices(shard, field_name):
    is_cell = field_name in _CELL_COMPONENTS
    layout = shard.layout
    local_slice = layout.storage_cell_owned if is_cell else layout.storage_node_owned
    global_slice = layout.global_cell_owned if is_cell else layout.global_node_owned
    return local_slice, global_slice


def _scatter_field_grad_to_shard(global_grad, shard, field_name, local_reference):
    """Transpose of ``_gather_component`` for one DFT field grad.

    The gathered global field is the owned x-slices of every shard stitched into
    one tensor, so the cotangent of a shard's local (padded) DFT field is the
    global grad restricted to that shard's owned global x-interval, written at the
    matching owned local x-interval and zero elsewhere (the ghost columns carry no
    monitor cotangent).
    """
    result = torch.zeros_like(local_reference)
    local_slice, global_slice = _owned_field_slices(shard, field_name)
    x_axis = global_grad.ndim - 3
    src_index = [slice(None)] * global_grad.ndim
    dst_index = [slice(None)] * result.ndim
    src_index[x_axis] = global_slice
    dst_index[x_axis] = local_slice
    result[tuple(dst_index)] = global_grad[tuple(src_index)].to(
        device=result.device, dtype=result.dtype
    )
    return result


def _index_global_grads(global_pack, grad_outputs):
    """Index global-pack cotangents by field name and (monitor, component)."""
    field_grads: dict[str, torch.Tensor] = {}
    monitor_grads: dict[tuple[str, str], torch.Tensor] = {}
    cursor = 0
    for field_name in global_pack.field_names:
        field_grads[field_name] = grad_outputs[cursor]
        cursor += 1
    for monitor_name, template in global_pack.monitor_templates.items():
        if not _monitor_template_is_point(template):
            raise RuntimeError(
                "Distributed adjoint seed routing reached a non-point monitor template; "
                "the objective-support guard should have rejected it."
            )
        for component_name in template["fields"]:
            monitor_grads[(monitor_name, component_name)] = grad_outputs[cursor]
            cursor += 1
    if cursor != int(global_pack.port_offset):
        raise RuntimeError("Distributed adjoint grad-output layout drifted from the forward pack.")
    return field_grads, monitor_grads


class _DistributedFDTDGradientBridge:
    """Joint-solve adjoint bridge over an x-slab ``DistributedFDTD`` forward.

    The forward runs the distributed field solve while capturing per-shard
    checkpoints; the merged monitor output feeds the user objective exactly as the
    non-differentiable distributed run does. The reverse runs one transposed reverse
    step per forward step -- ``Phase 1`` on every shard, the transposed magnetic
    halo, ``Phase 2``, the transposed electric halo, ``Phase 3``, then the per-shard
    source-term eps gradient -- and finally gathers the per-shard grad_eps owned
    slices into a single global tensor so the existing single-GPU material pullback
    runs once on the logical scene.
    """

    def __init__(self, simulation):
        from ...adjoint_inputs import (
            ancestry_safe_trainable_tensors,
            material_dependent_inputs,
            scene_trainable_material_tensors,
        )

        self.simulation = simulation
        self.base_scene = simulation.scene
        graph_scene = self._material_graph_scene()
        if simulation.scene_module is not None:
            candidates = tuple(
                parameter
                for parameter in simulation.scene_module.parameters()
                if parameter.requires_grad
            )
        else:
            candidates = scene_trainable_material_tensors(self.base_scene)
        self.material_inputs = material_dependent_inputs(graph_scene, candidates)
        self.trainable_inputs = ancestry_safe_trainable_tensors(self.material_inputs)
        if not self.trainable_inputs:
            raise ValueError(
                "Distributed FDTD backward requires a trainable input that contributes to "
                "the prepared-scene material tensors (a Box MaterialRegion density)."
            )
        if any(not tensor.is_cuda for tensor in self.trainable_inputs):
            raise RuntimeError("Differentiable distributed FDTD material inputs must be CUDA tensors.")

        self._distributed = None
        self._pack = None
        self._checkpoints = ()
        self._time_steps = 0
        self._dt = 0.0
        self._eps0 = 1.0
        self._solver_stats = None
        self._raw_output = None
        self._overlap_active = False
        self._uses_cpml = False
        self._last_global_grad_eps = None

    def _material_graph_scene(self):
        if self.simulation.scene_module is not None:
            return self.simulation.scene_module.to_scene()
        self.simulation._refresh_scene()
        return self.simulation.scene

    # -- forward -----------------------------------------------------------
    def forward(self, material_inputs):
        del material_inputs
        from ..distributed import DistributedFDTD

        simulation = self.simulation
        config = simulation.config
        if float(getattr(config, "shutoff", 0.0)) > 0.0:
            # Defense in depth: the public prepare-time validator already rejects a
            # trainable+parallel shutoff run; kept here (ValueError, matching the
            # distributed guard family) in case the bridge is driven directly.
            raise ValueError(
                "Distributed FDTD adjoint does not support field shutoff (shutoff>0) on "
                "trainable runs; the reverse pass replays a fixed step count."
            )

        dft_cfg = config.spectral_sampler
        with torch.no_grad():
            simulation._refresh_scene()
            scene = simulation.scene
            distributed = DistributedFDTD(
                scene,
                frequency=simulation.frequency,
                parallel=config.parallel,
                absorber_type=config.absorber,
                cpml_config=config.cpml_config,
            )
            distributed.init_field()
            require_distributed_adjoint_support(distributed)
            require_distributed_adjoint_objective_support(distributed)

            time_steps = simulation._resolve_fdtd_time_steps(distributed, scene)
            use_full_field_dft = config.full_field_dft or len(simulation.frequencies) > 1
            dft_request = (
                simulation.frequency
                if len(simulation.frequencies) == 1
                else simulation.frequencies
            )
            normalize_source = dft_cfg.normalize_source
            if normalize_source and len(scene.sources) != 1:
                raise ValueError(
                    "Distributed source normalization requires exactly one logical source."
                )
            if use_full_field_dft and not config.parallel.gather_fields:
                raise ValueError(
                    "Distributed FDTD adjoint with full-field DFT objectives requires "
                    "gather_fields=True so the global DFT cotangent can be scattered to shards."
                )

            self._preflight_gather_capacity(distributed, dft_request, use_full_field_dft)

            raw_output, checkpoints = self._run_forward_with_checkpoints(
                distributed,
                time_steps=time_steps,
                dft_frequency=dft_request,
                dft_window=dft_cfg.window,
                full_field_dft=use_full_field_dft,
                normalize_source=normalize_source,
            )

        if raw_output is None:
            raise RuntimeError("Distributed FDTD solve did not return any output.")
        pack = _prepare_forward_pack(raw_output)
        # Expose the bridge on the distributed solver so a caller holding the Result
        # (whose ``solver`` is the DistributedFDTD) can inspect the reverse product.
        distributed._adjoint_bridge = self
        self._distributed = distributed
        self._uses_cpml = _distributed_uses_cpml(distributed)
        self._pack = pack
        self._checkpoints = checkpoints
        self._time_steps = int(time_steps)
        self._dt = float(distributed.dt)
        self._eps0 = float(distributed.shards[0].solver.eps0)
        self._raw_output = raw_output
        self._solver_stats = simulation._build_fdtd_solver_stats(
            distributed,
            time_steps=time_steps,
            use_full_field_dft=use_full_field_dft,
            dft_cfg=dft_cfg,
        )
        return pack.output_tensors

    def _preflight_gather_capacity(self, distributed, dft_frequency, full_field_dft):
        # The one new global allocation the reverse adds is the eps-shaped grad
        # gather on the result device; preflight it before the time loop starts so
        # a too-large grid fails fast rather than after a full forward + reverse.
        require_gather_capacity(
            distributed.device,
            (distributed.Nx, distributed.Ny, distributed.Nz),
            dft_frequency=dft_frequency,
            full_field_dft=bool(full_field_dft),
            pending_local_dft_bytes=0,
        )

    def _run_forward_with_checkpoints(
        self,
        distributed,
        *,
        time_steps,
        dft_frequency,
        dft_window,
        full_field_dft,
        normalize_source,
    ):
        for shard in distributed.shards:
            shard.solver._normalize_source = bool(normalize_source)
        distributed._prepare_outputs(time_steps, dft_frequency, dft_window, full_field_dft)
        overlap_active = distributed._overlap_active()
        self._overlap_active = overlap_active
        stride = _checkpoint_stride(self.simulation, time_steps)

        distributed._synchronize_all()
        checkpoints = [capture_distributed_checkpoint(distributed, 0)]
        for n in range(int(time_steps)):
            if n > 0 and n % stride == 0:
                # The step kernels of the preceding iterations are enqueued on each
                # shard's non-blocking compute/communication streams, while the
                # checkpoint clones read the padded field storage on the device
                # default stream. Enforce stream ordering before every mid-loop
                # capture so the clone never races an in-flight update kernel; the
                # cost is one sync per checkpoint stride (~sqrt(N) times total).
                distributed._synchronize_all()
                checkpoints.append(capture_distributed_checkpoint(distributed, n))
            distributed._advance_one_step(n, overlap_active=overlap_active)
        distributed._synchronize_all()

        for shard in distributed.shards:
            solver = shard.solver
            if solver.dft_enabled:
                solver._sync_dft_legacy_state()
            if solver.observers_enabled:
                solver._sync_observer_legacy_state()
        raw_output = distributed._collect_output()
        return raw_output, tuple(checkpoints)

    # -- backward ----------------------------------------------------------
    def _backward_is_noop(self, grad_outputs) -> bool:
        """Whether the reverse can be skipped entirely (single-process only).

        In one process a fully-empty cotangent means zero gradient. The per-rank
        NCCL driver overrides this to always return ``False``: an off-owner rank
        carries empty cotangents yet must still drive the collective reverse (halo
        exchanges and the grad_eps gather) in lockstep, or the ranks deadlock.
        """
        return all(grad_output is None for grad_output in grad_outputs)

    def backward(self, base_inputs, grad_outputs):
        distributed = self._distributed
        if distributed is None or self._pack is None or not self._checkpoints:
            raise RuntimeError("Distributed FDTD backward called before forward initialized the bridge.")
        if self._backward_is_noop(grad_outputs):
            return tuple(torch.zeros_like(tensor) for tensor in base_inputs)

        shards = distributed.shards
        devices = distributed.devices
        dt = float(distributed.dt)

        resolved_grads = tuple(
            torch.zeros_like(output) if grad_output is None else grad_output
            for output, grad_output in zip(self._pack.output_tensors, grad_outputs)
        )
        field_grads, monitor_grads = _index_global_grads(self._pack, resolved_grads)

        seed_runtimes = self._build_shard_seed_runtimes(shards, field_grads, monitor_grads)

        # Per-shard static reverse inputs: detached eps leaves, dynamic curls, and
        # the resolved (cropped) source-term lists -- all constant across the sweep.
        eps_by_shard = {}
        curls_by_shard = {}
        resolved_by_shard = {}
        grad_eps_accum = {}
        for shard in shards:
            solver = shard.solver
            with torch.cuda.device(shard.device):
                eps = (solver.eps_Ex, solver.eps_Ey, solver.eps_Ez)
                eps_by_shard[shard.rank] = eps
                curls_by_shard[shard.rank] = dynamic_electric_curls(
                    solver, eps_ex=eps[0], eps_ey=eps[1], eps_ez=eps[2]
                )
                resolved_by_shard[shard.rank] = _resolved_source_term_lists(solver, *eps)
                grad_eps_accum[shard.rank] = {
                    "Ex": torch.zeros_like(eps[0]),
                    "Ey": torch.zeros_like(eps[1]),
                    "Ez": torch.zeros_like(eps[2]),
                }

        state_names = _CPML_STATE_NAMES if self._uses_cpml else _STANDARD_STATE_NAMES
        adjoint_states = {
            shard.rank: {
                name: torch.zeros_like(self._checkpoints[0].states[shard.rank].tensors[name])
                for name in state_names
            }
            for shard in shards
        }
        distributed.transport.prepare_adjoint_staging(shards, adjoint_states)

        checkpoint_lookup = {checkpoint.step: checkpoint for checkpoint in self._checkpoints}
        checkpoint_steps = [checkpoint.step for checkpoint in self._checkpoints]
        segment_bounds = list(zip(checkpoint_steps, checkpoint_steps[1:] + [self._time_steps]))

        for start_step, end_step in reversed(segment_bounds):
            mid_magnetic: dict[int, list] = {}
            trajectories = replay_distributed_segment(
                distributed,
                checkpoint_lookup[start_step],
                start_step,
                end_step,
                mid_magnetic_out=mid_magnetic,
            )
            for offset in range(end_step - start_step - 1, -1, -1):
                step_index = start_step + offset
                self._reverse_one_step(
                    distributed,
                    shards,
                    devices,
                    step_index=step_index,
                    offset=offset,
                    dt=dt,
                    trajectories=trajectories,
                    mid_magnetic=mid_magnetic,
                    adjoint_states=adjoint_states,
                    seed_runtimes=seed_runtimes,
                    eps_by_shard=eps_by_shard,
                    curls_by_shard=curls_by_shard,
                    resolved_by_shard=resolved_by_shard,
                    grad_eps_accum=grad_eps_accum,
                )

        return self._material_pullback(distributed, shards, base_inputs, grad_eps_accum)

    def _scatter_field_grad(self, global_grad, shard, field_name, local_reference):
        """Restrict a global DFT field cotangent to this shard's owned slice.

        In-process the field cotangents come from the single global forward pack, so
        each shard's local cotangent is the global grad restricted to its owned
        x-interval. The per-rank NCCL driver overrides this: its cotangents are
        already local (a separable per-owned-slab objective), so no scatter runs.
        """
        return _scatter_field_grad_to_shard(global_grad, shard, field_name, local_reference)

    def _build_shard_seed_runtimes(self, shards, field_grads, monitor_grads):
        seed_runtimes = {}
        for shard in shards:
            solver = shard.solver
            with torch.cuda.device(shard.device):
                local_raw = self._shard_local_raw_output(solver)
                local_pack = _prepare_forward_pack(local_raw)
                shard_grads = []
                for field_name in local_pack.field_names:
                    global_grad = field_grads.get(field_name)
                    if global_grad is None:
                        shard_grads.append(torch.zeros_like(local_raw[field_name]))
                        continue
                    shard_grads.append(
                        self._scatter_field_grad(
                            global_grad, shard, field_name, local_raw[field_name]
                        )
                    )
                for monitor_name, template in local_pack.monitor_templates.items():
                    for component_name in template["fields"]:
                        global_grad = monitor_grads.get((monitor_name, component_name))
                        local_output = local_raw["observers"][monitor_name]["components"][
                            component_name
                        ]
                        if global_grad is None:
                            shard_grads.append(torch.zeros_like(local_output))
                        else:
                            shard_grads.append(
                                global_grad.to(device=local_output.device, dtype=local_output.dtype)
                            )
                dft_pack = _schedule_to_tensor_pack(
                    _build_spectral_weight_schedule(
                        getattr(solver, "_dft_entries", ()),
                        time_steps=self._time_steps,
                        window_type=getattr(solver, "dft_window_type", "none"),
                    ),
                    device=solver.device,
                    dtype=solver.eps_Ex.dtype,
                )
                observer_pack = _schedule_to_tensor_pack(
                    _build_spectral_weight_schedule(
                        getattr(solver, "_observer_spectral_entries", ()),
                        time_steps=self._time_steps,
                        window_type=getattr(solver, "observer_window_type", "none"),
                    ),
                    device=solver.device,
                    dtype=solver.eps_Ex.dtype,
                )
                seed_runtimes[shard.rank] = _build_output_seeds(
                    solver,
                    local_pack,
                    tuple(shard_grads),
                    dft_schedule=dft_pack,
                    observer_schedule=observer_pack,
                )
        return seed_runtimes

    @staticmethod
    def _shard_local_raw_output(solver):
        raw: dict = {}
        if solver.dft_enabled:
            local = solver.get_frequency_solution(all_frequencies=True)
            for name, tensor in local.items():
                if name in {"Ex", "Ey", "Ez"}:
                    raw[name] = tensor
        if solver.observers_enabled:
            raw["observers"] = solver.get_observer_results()
        return raw

    def _reverse_one_step(
        self,
        distributed,
        shards,
        devices,
        *,
        step_index,
        offset,
        dt,
        trajectories,
        mid_magnetic,
        adjoint_states,
        seed_runtimes,
        eps_by_shard,
        curls_by_shard,
        resolved_by_shard,
        grad_eps_accum,
    ):
        time_value = step_index * dt
        state_names = _CPML_STATE_NAMES if self._uses_cpml else _STANDARD_STATE_NAMES

        # Post-step adjoint = current accumulated adjoint plus this step's monitor
        # seed injection (owned indices only; each shard owns its point monitors and
        # the scattered slice of the full-field DFT cotangent). The seed injects into
        # E/H only; the psi adjoints (CPML branch) carry their accumulated value,
        # which is 0 on the first reverse step and grows through the psi pullbacks.
        post = {
            shard.rank: {
                name: adjoint_states[shard.rank][name].clone()
                for name in state_names
            }
            for shard in shards
        }
        for shard in shards:
            with torch.cuda.device(shard.device):
                _apply_seed_runtime(post[shard.rank], seed_runtimes[shard.rank], step_index)
        _synchronize(devices)

        # The two reverse-phase runners insert the transposed Yee halos at the same
        # seams the single-GPU cores expose: the standard core splits at Phase 1/2/3;
        # the CPML core splits at the electric/magnetic phase boundary. Each returns
        # ``{rank: (pre_step_adjoint, (gx, gy, gz), adj_h_mid)}`` plus the backend
        # label so the shared source-term/accumulation tail below is regime-agnostic.
        if self._uses_cpml:
            per_shard, backend_label = self._reverse_phases_cpml(
                distributed,
                shards,
                devices,
                offset=offset,
                trajectories=trajectories,
                mid_magnetic=mid_magnetic,
                post=post,
                eps_by_shard=eps_by_shard,
            )
        else:
            per_shard, backend_label = self._reverse_phases_standard(
                distributed,
                shards,
                devices,
                offset=offset,
                trajectories=trajectories,
                mid_magnetic=mid_magnetic,
                post=post,
                eps_by_shard=eps_by_shard,
                curls_by_shard=curls_by_shard,
            )

        # Per-shard source-term eps gradient + grad_eps accumulation, then roll the
        # adjoint state back one step.
        for shard in shards:
            pre_step_adjoint, (gx, gy, gz), adj_h_mid = per_shard[shard.rank]
            eps_ex, eps_ey, eps_ez = eps_by_shard[shard.rank]
            with torch.cuda.device(shard.device):
                step_result = _ReverseStepResult(
                    pre_step_adjoint=pre_step_adjoint,
                    grad_eps_ex=gx,
                    grad_eps_ey=gy,
                    grad_eps_ez=gz,
                    backend=backend_label,
                    magnetic_output_adjoint=adj_h_mid,
                )
                step_result = _accumulate_source_term_gradients(
                    step_result,
                    solver=shard.solver,
                    adjoint_state=post[shard.rank],
                    time_value=time_value,
                    eps_ex=eps_ex,
                    eps_ey=eps_ey,
                    eps_ez=eps_ez,
                    resolved_source_terms=resolved_by_shard[shard.rank],
                )
                accum = grad_eps_accum[shard.rank]
                accum["Ex"] = accum["Ex"] + step_result.grad_eps_ex
                accum["Ey"] = accum["Ey"] + step_result.grad_eps_ey
                accum["Ez"] = accum["Ez"] + step_result.grad_eps_ez
                adjoint_states[shard.rank] = dict(step_result.pre_step_adjoint)
        _synchronize(devices)

    def _reverse_phases_standard(
        self,
        distributed,
        shards,
        devices,
        *,
        offset,
        trajectories,
        mid_magnetic,
        post,
        eps_by_shard,
        curls_by_shard,
    ):
        """Standard (open/PEC) reverse phases with the two transposed Yee halos.

        Phase 1 (electric adjoint -> mid-H adjoint) -> transposed magnetic halo
        (Hy/Hz) -> Phase 2 (magnetic adjoint -> pre-step E adjoint + eps gradient)
        -> transposed electric halo (Ey/Ez) -> Phase 3 (magnetic decay pullback ->
        pre-step H adjoint).
        """
        adj_h_mid = {}
        for shard in shards:
            ex_curl, ey_curl, ez_curl = curls_by_shard[shard.rank]
            with torch.cuda.device(shard.device):
                adj_h_mid[shard.rank] = reverse_phase1_electric_to_h(
                    shard.solver,
                    trajectories[shard.rank][offset],
                    post[shard.rank],
                    ex_curl=ex_curl,
                    ey_curl=ey_curl,
                    ez_curl=ez_curl,
                )
        _synchronize(devices)

        distributed.transport.exchange_magnetic_adjoint(shards, adj_h_mid)
        _synchronize(devices)

        pre_electric = {}
        step_grad_eps = {}
        for shard in shards:
            ex_curl, ey_curl, ez_curl = curls_by_shard[shard.rank]
            eps_ex, eps_ey, eps_ez = eps_by_shard[shard.rank]
            mids = mid_magnetic[shard.rank][offset]
            with torch.cuda.device(shard.device):
                pre_e, gx, gy, gz = reverse_phase2_magnetic_to_e(
                    shard.solver,
                    trajectories[shard.rank][offset],
                    post[shard.rank],
                    adj_h_mid[shard.rank],
                    hx_mid=mids["Hx"],
                    hy_mid=mids["Hy"],
                    hz_mid=mids["Hz"],
                    ex_curl=ex_curl,
                    ey_curl=ey_curl,
                    ez_curl=ez_curl,
                    eps_ex=eps_ex,
                    eps_ey=eps_ey,
                    eps_ez=eps_ez,
                )
                pre_electric[shard.rank] = pre_e
                step_grad_eps[shard.rank] = (gx, gy, gz)
        _synchronize(devices)

        distributed.transport.exchange_electric_adjoint(shards, pre_electric)
        _synchronize(devices)

        pre_magnetic = {}
        for shard in shards:
            with torch.cuda.device(shard.device):
                pre_magnetic[shard.rank] = reverse_phase3_decay(
                    shard.solver, adj_h_mid[shard.rank], trajectories[shard.rank][offset]
                )
        _synchronize(devices)

        per_shard = {
            shard.rank: (
                {**pre_electric[shard.rank], **pre_magnetic[shard.rank]},
                step_grad_eps[shard.rank],
                adj_h_mid[shard.rank],
            )
            for shard in shards
        }
        return per_shard, _STANDARD_BACKEND_LABEL

    def _reverse_phases_cpml(
        self,
        distributed,
        shards,
        devices,
        *,
        offset,
        trajectories,
        mid_magnetic,
        post,
        eps_by_shard,
    ):
        """CPML reverse phases with the two transposed Yee halos.

        ``reverse_cpml_phase_electric`` (electric adjoint -> pre-step E/psi_e
        adjoint + eps gradient, folding curl(H) into the mid-step H adjoint that is
        pre-seeded with the post-step H adjoint) -> transposed magnetic halo (Hy/Hz)
        -> ``reverse_cpml_phase_magnetic`` (magnetic-decay + psi_h pullback ->
        pre-step H/psi_h adjoint, folding curl(E) into the pre-step E adjoint) ->
        transposed electric halo (Ey/Ez). No psi halo is exchanged: the x-CPML
        pinning invariant (asserted at prepare) keeps every internal-face psi
        inactive, so the cross-interface curl coupling rides the Yee field halos
        exactly as on the standard path (S4 audit).
        """
        ctx_by_rank = {}
        adj_h_mid = {}
        for shard in shards:
            forward_state = trajectories[shard.rank][offset]
            eps_ex, eps_ey, eps_ez = eps_by_shard[shard.rank]
            mids = mid_magnetic[shard.rank][offset]
            with torch.cuda.device(shard.device):
                ctx = allocate_cpml_reverse_context(
                    shard.solver,
                    forward_state,
                    post[shard.rank],
                    eps_ex=eps_ex,
                    eps_ey=eps_ey,
                    eps_ez=eps_ez,
                )
                reverse_cpml_phase_electric(
                    shard.solver,
                    forward_state,
                    post[shard.rank],
                    ctx,
                    hx_mid=mids["Hx"],
                    hy_mid=mids["Hy"],
                    hz_mid=mids["Hz"],
                    eps_ex=eps_ex,
                    eps_ey=eps_ey,
                    eps_ez=eps_ez,
                )
                ctx_by_rank[shard.rank] = ctx
                adj_h_mid[shard.rank] = ctx.magnetic_output_adjoint
        _synchronize(devices)

        distributed.transport.exchange_magnetic_adjoint(shards, adj_h_mid)
        _synchronize(devices)

        for shard in shards:
            with torch.cuda.device(shard.device):
                reverse_cpml_phase_magnetic(
                    shard.solver, post[shard.rank], ctx_by_rank[shard.rank]
                )
        _synchronize(devices)

        # The pre-step E adjoint (Ey/Ez) lives inside ctx.pre_step_adjoint; the
        # electric halo accumulates the left ghost into the right owner in place.
        pre_electric = {
            shard.rank: ctx_by_rank[shard.rank].pre_step_adjoint for shard in shards
        }
        distributed.transport.exchange_electric_adjoint(shards, pre_electric)
        _synchronize(devices)

        per_shard = {
            shard.rank: (
                ctx_by_rank[shard.rank].pre_step_adjoint,
                (
                    ctx_by_rank[shard.rank].grad_eps_ex,
                    ctx_by_rank[shard.rank].grad_eps_ey,
                    ctx_by_rank[shard.rank].grad_eps_ez,
                ),
                adj_h_mid[shard.rank],
            )
            for shard in shards
        }
        return per_shard, _CPML_BACKEND_LABEL

    def _material_pullback(self, distributed, shards, base_inputs, grad_eps_accum):
        global_grad_eps = {}
        for name in ("Ex", "Ey", "Ez"):
            local_values = tuple(grad_eps_accum[shard.rank][name] for shard in shards)
            global_grad_eps[name] = distributed._gather_component(name, local_values)
        for device in distributed.devices:
            torch.cuda.synchronize(device)
        # The distributed reverse product (gathered grad_eps) is the deterministic
        # output of S3; the assign-semantics fused reverse kernels, ordered add_
        # accumulation, and deterministic gather copies make it bitwise reproducible.
        # Stashed so a determinism test can assert on it directly, isolated from the
        # shared torch grid_sample material VJP (whose backward uses atomicAdd and is
        # not bitwise reproducible, single- and multi-GPU alike).
        self._last_global_grad_eps = {
            name: value.detach().clone() for name, value in global_grad_eps.items()
        }

        with torch.enable_grad():
            scene = self._material_graph_scene()
            return pullback_material_input_gradients(
                scene,
                inputs=base_inputs,
                grad_eps_ex=global_grad_eps["Ex"],
                grad_eps_ey=global_grad_eps["Ey"],
                grad_eps_ez=global_grad_eps["Ez"],
                eps0=self._eps0,
            )


class _DistributedFDTDMaterialGradientFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bridge, *material_inputs):
        ctx.bridge = bridge
        ctx.material_inputs = tuple(material_inputs)
        outputs = bridge.forward(tuple(material_inputs))
        return outputs if len(outputs) != 1 else outputs[0]

    @staticmethod
    def backward(ctx, *grad_outputs):
        gradients = ctx.bridge.backward(ctx.material_inputs, grad_outputs)
        return (None, *gradients)


class _NcclDistributedFDTDGradientBridge(_DistributedFDTDGradientBridge):
    """Per-rank collective reverse driver for a one-process-per-GPU NCCL launch.

    Reuses the in-process bridge's reverse math verbatim -- the transposed reverse
    cores (``_reverse_phases_standard`` / ``_reverse_phases_cpml``), this track's
    NCCL adjoint halos, the checkpoint/replay schedule, and the source-term eps
    accumulation -- and replaces the three single-process assumptions the in-process
    bridge makes:

    1. the forward output is built from a per-rank LOCAL pack (the global
       ``_collect_output`` returns ``None`` off root), so each rank has its own
       objective-facing tensors;
    2. the objective is seeded locally -- a point monitor is owned by exactly one
       rank (the others seed zero and receive adjoint only through the halos), and a
       full-field DFT is separable across owned slabs -- so no cross-rank cotangent
       scatter is needed;
    3. grad_eps is gathered slab-wise to rank 0 with the already-NCCL-capable
       component gather and the material pullback runs on rank 0 only.

    Everything cross-interface (both Yee field families, and the psi coupling that
    rides them on the CPML branch) travels through the transposed NCCL halos, whose
    discrete-transpose identity is independently proven at the transport level.
    """

    def __init__(self, simulation, *, objective):
        super().__init__(simulation)
        # objective(pack, leaf_outputs, layout) -> real scalar loss tensor built
        # from differentiable copies of the local pack output tensors. The driver
        # differentiates it to obtain the local seed cotangents.
        self._objective = objective
        self._local_cotangents: tuple[torch.Tensor, ...] | None = None
        self._local_loss = None
        self._rank = None

    def _backward_is_noop(self, grad_outputs) -> bool:
        # Every rank must drive the collective reverse in lockstep even when its
        # local seed is empty (off-owner ranks receive adjoint only via the halos),
        # so the reverse is never skipped on the per-rank NCCL path.
        return False

    def _scatter_field_grad(self, global_grad, shard, field_name, local_reference):
        # The per-rank objective already produces cotangents in this shard's local
        # padded layout (nonzero on owned indices, zero in the ghost), so they are
        # used directly -- there is no global field to restrict.
        return global_grad

    def _run_forward_capture_local(
        self,
        distributed,
        *,
        time_steps,
        dft_frequency,
        dft_window,
        full_field_dft,
        normalize_source,
    ):
        """Drive the real NCCL forward loop capturing per-rank checkpoints.

        Mirrors ``_run_forward_with_checkpoints`` but omits the global
        ``_collect_output`` -- the driver reads each rank's shard-local DFT/observer
        output directly, so the cross-rank monitor/field gather (which returns
        ``None`` off root) is never needed here.
        """

        for shard in distributed.shards:
            shard.solver._normalize_source = bool(normalize_source)
        distributed._prepare_outputs(time_steps, dft_frequency, dft_window, full_field_dft)
        overlap_active = distributed._overlap_active()
        self._overlap_active = overlap_active
        stride = _checkpoint_stride(self.simulation, time_steps)

        distributed._synchronize_all()
        checkpoints = [capture_distributed_checkpoint(distributed, 0)]
        for n in range(int(time_steps)):
            if n > 0 and n % stride == 0:
                distributed._synchronize_all()
                checkpoints.append(capture_distributed_checkpoint(distributed, n))
            distributed._advance_one_step(n, overlap_active=overlap_active)
        distributed._synchronize_all()

        for shard in distributed.shards:
            solver = shard.solver
            if solver.dft_enabled:
                solver._sync_dft_legacy_state()
            if solver.observers_enabled:
                solver._sync_observer_legacy_state()
        return tuple(checkpoints)

    def forward(self, material_inputs):
        del material_inputs
        from ..distributed import DistributedFDTD

        simulation = self.simulation
        config = simulation.config
        if float(getattr(config, "shutoff", 0.0)) > 0.0:
            raise ValueError(
                "Distributed FDTD adjoint does not support field shutoff (shutoff>0) on "
                "trainable runs; the reverse pass replays a fixed step count."
            )

        dft_cfg = config.spectral_sampler
        with torch.no_grad():
            simulation._refresh_scene()
            scene = simulation.scene
            distributed = DistributedFDTD(
                scene,
                frequency=simulation.frequency,
                parallel=config.parallel,
                absorber_type=config.absorber,
                cpml_config=config.cpml_config,
                allow_adjoint=True,
            )
            distributed.init_field()
            require_distributed_adjoint_support(distributed)
            require_distributed_adjoint_objective_support(distributed)

            time_steps = simulation._resolve_fdtd_time_steps(distributed, scene)
            use_full_field_dft = config.full_field_dft or len(simulation.frequencies) > 1
            dft_request = (
                simulation.frequency
                if len(simulation.frequencies) == 1
                else simulation.frequencies
            )
            normalize_source = dft_cfg.normalize_source
            if normalize_source and len(scene.sources) != 1:
                raise ValueError(
                    "Distributed source normalization requires exactly one logical source."
                )
            # The eps-shaped grad gather lands only on rank 0; preflight its capacity
            # there so the check runs on the device that actually allocates it.
            if distributed.rank == 0:
                self._preflight_gather_capacity(distributed, dft_request, use_full_field_dft)

            checkpoints = self._run_forward_capture_local(
                distributed,
                time_steps=time_steps,
                dft_frequency=dft_request,
                dft_window=dft_cfg.window,
                full_field_dft=use_full_field_dft,
                normalize_source=normalize_source,
            )

        local_raw = self._shard_local_raw_output(distributed.shards[0].solver)
        pack = _prepare_forward_pack(local_raw)
        distributed._adjoint_bridge = self
        self._distributed = distributed
        self._rank = int(distributed.rank)
        self._uses_cpml = _distributed_uses_cpml(distributed)
        self._pack = pack
        self._checkpoints = checkpoints
        self._time_steps = int(time_steps)
        self._dt = float(distributed.dt)
        self._eps0 = float(distributed.shards[0].solver.eps0)
        self._raw_output = local_raw
        self._solver_stats = None
        return pack.output_tensors

    def compute_local_objective(self):
        """Differentiate the local objective into per-rank seed cotangents.

        Builds differentiable leaf copies of the local pack output tensors, runs the
        objective callable, and returns ``(local_loss, cotangents)`` where the
        cotangents align to the pack output order (exactly what
        ``_build_output_seeds`` consumes). Off-owner ranks whose local pack is empty
        return a zero loss and an empty cotangent tuple; they still run the reverse
        loop and receive adjoint through the halos.
        """

        pack = self._pack
        n = int(pack.wire_offset)
        layout = self._distributed.shards[0].layout
        device = self._distributed.shards[0].device
        leaves = tuple(
            tensor.detach().clone().requires_grad_(True)
            for tensor in pack.output_tensors[:n]
        )
        with torch.enable_grad():
            loss = self._objective(pack, leaves, layout)
            if not isinstance(loss, torch.Tensor):
                loss = torch.as_tensor(float(loss), device=device)
            loss = loss.reshape(())
            if leaves:
                grads = torch.autograd.grad(loss, leaves, allow_unused=True)
                cotangents = tuple(
                    torch.zeros_like(leaf) if grad is None else grad
                    for leaf, grad in zip(leaves, grads)
                )
            else:
                cotangents = ()
        self._local_loss = loss.detach()
        self._local_cotangents = cotangents
        return self._local_loss, cotangents

    def _material_pullback(self, distributed, shards, base_inputs, grad_eps_accum):
        # grad_eps is gathered slab-wise: the NCCL component gather returns the
        # global tensor on rank 0 and None on every other rank. The single-GPU
        # material pullback then runs on rank 0 only; other ranks hold no global
        # density and return zeros so the driver contract (grads meaningful on
        # rank 0) is explicit rather than crashing on a None gather result.
        global_grad_eps = {}
        for name in ("Ex", "Ey", "Ez"):
            local_values = tuple(grad_eps_accum[shard.rank][name] for shard in shards)
            global_grad_eps[name] = distributed._gather_component(name, local_values)
        for device in distributed.devices:
            torch.cuda.synchronize(device)

        if distributed.rank != 0:
            self._last_global_grad_eps = None
            return tuple(torch.zeros_like(tensor) for tensor in base_inputs)

        self._last_global_grad_eps = {
            name: value.detach().clone() for name, value in global_grad_eps.items()
        }
        with torch.enable_grad():
            scene = self._material_graph_scene()
            return pullback_material_input_gradients(
                scene,
                inputs=base_inputs,
                grad_eps_ex=global_grad_eps["Ex"],
                grad_eps_ey=global_grad_eps["Ey"],
                grad_eps_ez=global_grad_eps["Ez"],
                eps0=self._eps0,
            )


def point_monitor_l2_objective(pack, leaves, layout):
    """Separable ``sum |spectrum|^2`` over every local point-monitor component.

    A point monitor is owned by exactly one rank, so only the owner's local pack
    carries monitor outputs; every other rank has an empty monitor set and returns a
    zero loss, contributing nothing to the world sum and seeding nothing (it receives
    adjoint solely through the halos). ``leaves`` are differentiable copies of the
    pack output tensors, ordered fields-then-monitor-components, so the monitor
    cursor starts past the (here empty) field block.
    """

    del layout
    cursor = len(pack.field_names)
    loss = None
    for _monitor_name, template in pack.monitor_templates.items():
        for _component_name in template["fields"]:
            spectrum = leaves[cursor]
            cursor += 1
            term = (spectrum.real ** 2 + spectrum.imag ** 2).sum()
            loss = term if loss is None else loss + term
    if loss is None:
        return 0.0
    return loss


def run_nccl_distributed_reverse(simulation, *, objective):
    """Run the per-rank collective NCCL forward+reverse and return the gradient.

    Returns ``(total_loss, grads, bridge)`` where ``total_loss`` is the world-summed
    objective (identical on every rank), ``grads`` is the tuple of gradients w.r.t.
    the trainable inputs (meaningful on rank 0, zeros elsewhere), and ``bridge``
    exposes ``_last_global_grad_eps`` (rank 0) for determinism checks. The objective
    is a separable local functional; the world sum reproduces the single-process
    objective because the owned slabs tile the global domain with no overlap.
    """

    bridge = _NcclDistributedFDTDGradientBridge(simulation, objective=objective)
    bridge.forward(bridge.trainable_inputs)
    local_loss, cotangents = bridge.compute_local_objective()
    total_loss = bridge._distributed.transport.allreduce_scalar(local_loss)
    grads = bridge.backward(bridge.trainable_inputs, cotangents)
    return total_loss, grads, bridge


def run_distributed_fdtd_with_gradient_bridge(simulation):
    from ...result import Result

    bridge = _DistributedFDTDGradientBridge(simulation)
    raw_outputs = _DistributedFDTDMaterialGradientFunction.apply(bridge, *bridge.trainable_inputs)
    output_tensors = raw_outputs if isinstance(raw_outputs, tuple) else (raw_outputs,)

    from ..adjoint.core import _rebuild_monitors

    pack = bridge._pack
    fields = {
        field_name.upper(): output_tensors[index]
        for index, field_name in enumerate(pack.field_names)
    }
    monitors = _rebuild_monitors(pack.monitor_templates, output_tensors, len(pack.field_names))
    raw_output = {
        field_name: fields[field_name.upper()] for field_name in pack.field_names
    }
    if monitors:
        raw_output["observers"] = monitors
    return Result(
        method="fdtd",
        scene=simulation.scene,
        prepared_scene=bridge._distributed.scene,
        frequency=simulation.frequency,
        frequencies=simulation.frequencies,
        solver=bridge._distributed,
        fields=fields,
        monitors=monitors,
        metadata=simulation.metadata,
        solver_stats=bridge._solver_stats,
        raw_output=raw_output,
    )
