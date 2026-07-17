"""Distributed forward-replay parity (two CUDA devices).

The distributed joint-solve adjoint replays the forward field trajectory as two
explicit half-steps per shard with the forward Yee halos copied between them.
This test pins that the replayed OWNED states reproduce a native distributed
forward run exactly for the pure real standard configuration, and that the
checkpoint/replay entry points fail closed on unsupported configurations.
"""

from __future__ import annotations

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.distributed import DistributedFDTD
from witwin.maxwell.fdtd.distributed.adjoint import (
    capture_distributed_checkpoint,
    replay_distributed_segment,
    require_distributed_adjoint_support,
)
from witwin.maxwell.fdtd_parallel import FDTDParallelConfig


_FREQUENCY = 1.0e9
_CELL_COMPONENTS = frozenset(("Ex", "Hy", "Hz"))


def _parallel(devices):
    return FDTDParallelConfig(
        devices=devices,
        transport="cuda_p2p",
        gather_fields=False,
        overlap=False,
        result_device=devices[0],
    )


def _standard_scene(device):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.4, 0.4), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.none(),
        device=device,
    )
    scene.add_source(
        mw.PointDipole(
            position=(-0.15, 0.0, 0.0),
            polarization="Ez",
            profile="ideal",
            source_time=mw.GaussianPulse(frequency=_FREQUENCY, fwidth=5.0e8),
            name="drive",
        )
    )
    return scene


def _owned_slice(shard, component):
    layout = shard.layout
    return (
        layout.storage_cell_owned
        if component in _CELL_COMPONENTS
        else layout.storage_node_owned
    )


def test_distributed_replay_reproduces_forward_owned_states_exactly(
    cuda_p2p_devices, cuda_memory_cleanup
):
    steps = 40
    scene = _standard_scene("cuda:0")
    distributed = DistributedFDTD(
        scene,
        frequency=_FREQUENCY,
        parallel=_parallel(cuda_p2p_devices),
        absorber_type="none",
    )
    distributed.init_field()
    require_distributed_adjoint_support(distributed)

    # Checkpoint the (zero) initial state, run the native distributed forward, then
    # snapshot the final owned fields the forward produced on each shard.
    checkpoint = capture_distributed_checkpoint(distributed, 0)
    distributed.solve(time_steps=steps, dft_frequency=None, full_field_dft=False)
    forward = {
        shard.rank: {
            name: getattr(shard.solver, name).detach().clone()
            for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        }
        for shard in distributed.shards
    }

    trajectories = replay_distributed_segment(distributed, checkpoint, 0, steps)

    # The distributed forward runs fused CUDA bounded kernels; the replay runs the
    # torch reference update. These are not bitwise identical -- the fused kernel and
    # the torch elementwise ops reduce in a different floating-point order -- exactly
    # as the single-GPU bridge already tolerance-gates native-forward vs torch-replay
    # (tests/gradients/test_fdtd_adjoint_bridge.py
    # ::test_fdtd_gradient_bridge_checkpoint_replay_matches_forward_state). The
    # decomposition itself introduces no algorithmic error, so the gate is
    # calibrated against the measured drift rather than borrowing that looser bound:
    # on this 40-step scene the true owned-state drift is ~5e-10 absolute, while
    # disabling the electric halo yields ~8e-5 and the magnetic halo ~1e-2 of error
    # concentrated at the interface. rtol=1e-5/atol=1e-7 sits ~200x above the true
    # drift and ~500x below the smallest halo-bug error, so it passes the correct
    # decomposition and fails a broken halo (falsification-checked).
    assert len(distributed.shards) == 2  # the interface parity is the point
    nonzero_energy = 0.0
    for shard in distributed.shards:
        replayed = trajectories[shard.rank][steps]
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            owned = _owned_slice(shard, name)
            replay_owned = replayed[name][owned]
            forward_owned = forward[shard.rank][name][owned]
            nonzero_energy += float(forward_owned.abs().sum().item())
            torch.testing.assert_close(
                replay_owned,
                forward_owned,
                rtol=1e-5,
                atol=1e-7,
                msg=(
                    f"rank {shard.rank} component {name} owned-state replay mismatch: "
                    f"max abs diff {(replay_owned - forward_owned).abs().max().item():.3e}"
                ),
            )
    # Guard against a trivially-zero pass: the forward must have carried real signal.
    assert nonzero_energy > 0.0


def test_replay_requires_matching_partition(cuda_p2p_devices, cuda_memory_cleanup):
    scene = _standard_scene("cuda:0")
    distributed = DistributedFDTD(
        scene,
        frequency=_FREQUENCY,
        parallel=_parallel(cuda_p2p_devices),
        absorber_type="none",
    )
    distributed.init_field()
    checkpoint = capture_distributed_checkpoint(distributed, 0)
    mismatched = checkpoint.__class__(
        step=checkpoint.step,
        partition_signature=checkpoint.partition_signature[:1],
        states=checkpoint.states,
    )
    with pytest.raises(RuntimeError, match="different partition"):
        replay_distributed_segment(distributed, mismatched, 0, 1)


def test_distributed_solver_rejects_trainable_scene_directly():
    """Defense in depth: the distributed solver itself fails closed on a trainable
    scene, independent of the public Simulation trainable guard, because the
    distributed joint-solve adjoint bridge is not wired yet."""

    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.4, 0.4), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.none(),
        device="cpu",
    )
    density = torch.rand((2, 2, 2), requires_grad=True)
    scene.add_material_region(
        mw.MaterialRegion(
            name="design",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.2, 0.2, 0.2)),
            density=density,
            eps_bounds=(1.0, 5.0),
        )
    )
    with pytest.raises(ValueError, match="trainable"):
        DistributedFDTD(
            scene,
            frequency=_FREQUENCY,
            parallel=FDTDParallelConfig(
                devices=("cuda:0", "cuda:1"),
                transport="cuda_p2p",
                gather_fields=False,
                result_device="cuda:0",
            ),
        )


def test_checkpoint_rejects_cpml_absorber(cuda_p2p_devices, cuda_memory_cleanup):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.4, 0.4), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
        device="cuda:0",
    )
    scene.add_source(
        mw.PointDipole(
            position=(-0.15, 0.0, 0.0),
            polarization="Ez",
            profile="ideal",
            source_time=mw.GaussianPulse(frequency=_FREQUENCY, fwidth=5.0e8),
            name="drive",
        )
    )
    distributed = DistributedFDTD(
        scene,
        frequency=_FREQUENCY,
        parallel=_parallel(cuda_p2p_devices),
        absorber_type="cpml",
    )
    distributed.init_field()
    with pytest.raises(ValueError, match="pure real standard"):
        capture_distributed_checkpoint(distributed, 0)
