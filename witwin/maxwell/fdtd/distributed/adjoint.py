"""Distributed joint-solve adjoint infrastructure (Phase 7, slices S1+S2).

This module provides the internal building blocks the distributed reverse pass is
assembled from:

* per-shard forward checkpoint capture keyed by the partition manifest, and
* a distributed forward *replay* that advances every shard as two explicit
  half-steps with the forward Yee halos copied between them, so the owned states
  it produces are bit-identical to the native distributed forward run.

Only the pure real *standard* (open-boundary) configuration is supported here.
Every unsupported medium/boundary/coupling is rejected fail-closed by
:func:`require_distributed_adjoint_support` before any checkpoint is taken, so the
replay never silently runs a configuration whose transposed reverse cores have
not been verified. The public trainable+parallel bridge is intentionally *not*
wired yet (that is slice S3); these entry points are internal.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..adjoint.core import (
    _forward_electric_fields_standard,
    _forward_magnetic_fields,
    _resolved_source_term_lists,
)
from ..checkpoint import (
    FDTDCheckpointState,
    capture_checkpoint_state,
    clone_checkpoint_tensors,
)

_ELECTRIC_NAMES = ("Ex", "Ey", "Ez")
_MAGNETIC_NAMES = ("Hx", "Hy", "Hz")

# Shard-solver attributes that place the shard outside the verified pure real
# standard reverse class. Each maps to the reason surfaced by the guard.
_UNSUPPORTED_SHARD_FLAGS = (
    ("uses_cpml", "CPML absorbing boundaries"),
    ("complex_fields_enabled", "complex (Bloch split-field) state"),
    ("dispersive_enabled", "electric dispersive (ADE) media"),
    ("magnetic_dispersive_enabled", "magnetic dispersive (ADE) media"),
    ("nonlinear_enabled", "nonlinear media"),
    ("conductive_enabled", "static-conductive media"),
    ("full_aniso_enabled", "full off-diagonal anisotropy"),
    ("tfsf_enabled", "total-field/scattered-field injection"),
    ("modulation_enabled", "time-modulated media"),
)


def require_distributed_adjoint_support(distributed) -> None:
    """Reject any distributed configuration outside the verified standard class.

    Fail-closed guard: the distributed checkpoint/replay/reverse path is only
    validated for the pure real standard (open-boundary) update. CPML, dispersive,
    nonlinear, conductive, anisotropic, complex/Bloch, TFSF, modulated, and any
    coupled port/circuit/network configuration are rejected here rather than
    producing an unverified gradient.
    """

    if not getattr(distributed, "_initialized", False):
        raise RuntimeError(
            "Distributed adjoint support can only be validated after init_field()."
        )
    for shard in distributed.shards:
        solver = shard.solver
        for attribute, reason in _UNSUPPORTED_SHARD_FLAGS:
            if bool(getattr(solver, attribute, False)):
                raise ValueError(
                    "Distributed FDTD checkpoint/replay currently supports only the "
                    f"pure real standard update; shard {shard.rank} uses {reason}."
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


def _forward_electric_halo(shards, current) -> None:
    """Forward electric halo: right owned Ey/Ez node -> left neighbour ghost node."""

    for destination, source in zip(shards[:-1], shards[1:]):
        destination_state = current[destination.rank]
        source_state = current[source.rank]
        destination_ghost = destination.layout.storage_node_owned.stop
        source_node = source.layout.storage_node_owned.start
        with torch.cuda.device(destination.device):
            destination_state["Ey"][destination_ghost].copy_(source_state["Ey"][source_node])
            destination_state["Ez"][destination_ghost].copy_(source_state["Ez"][source_node])


def _forward_magnetic_halo(shards, current) -> None:
    """Forward magnetic halo: left owned Hy/Hz cell -> right neighbour ghost cell."""

    for source, destination in zip(shards[:-1], shards[1:]):
        source_state = current[source.rank]
        destination_state = current[destination.rank]
        source_last = source.layout.storage_cell_owned.stop - 1
        with torch.cuda.device(destination.device):
            destination_state["Hy"][0].copy_(source_state["Hy"][source_last])
            destination_state["Hz"][0].copy_(source_state["Hz"][source_last])


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

    Owned states are bit-identical to the native distributed forward run for the
    pure real standard configuration (verified by the replay-parity test); the
    ghost planes carry the neighbour's owned value after each halo copy exactly as
    the forward serialized path does.
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

    trajectories = {
        shard.rank: [{name: current[shard.rank][name].clone() for name in _ELECTRIC_NAMES + _MAGNETIC_NAMES}]
        for shard in shards
    }

    with torch.no_grad():
        for step_index in range(start_step, end_step):
            time_value = step_index * dt

            # (1) Electric halo, then (2) magnetic half on every shard.
            _forward_electric_halo(shards, current)
            _synchronize(devices)
            for shard in shards:
                with torch.cuda.device(shard.device):
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
            _forward_magnetic_halo(shards, current)
            _synchronize(devices)
            if mid_magnetic_out is not None:
                for shard in shards:
                    with torch.cuda.device(shard.device):
                        mid_magnetic_out.setdefault(shard.rank, []).append(
                            {name: current[shard.rank][name].clone() for name in _MAGNETIC_NAMES}
                        )

            # (4) Electric half on every shard.
            for shard in shards:
                with torch.cuda.device(shard.device):
                    electric = _forward_electric_fields_standard(
                        shard.solver,
                        current[shard.rank],
                        {name: current[shard.rank][name] for name in _MAGNETIC_NAMES},
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
                        for name in _ELECTRIC_NAMES + _MAGNETIC_NAMES
                    }
                )

    return trajectories
