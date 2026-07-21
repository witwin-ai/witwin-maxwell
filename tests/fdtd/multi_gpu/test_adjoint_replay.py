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


def test_distributed_cpml_replay_reproduces_forward_owned_psi_and_fields(
    cuda_p2p_devices, cuda_memory_cleanup
):
    """CPML replay parity including the twelve psi memory fields.

    The open-boundary parity test above exercises only the six E/H fields on a
    psi-free scene, so it cannot see the CPML replay half-steps. This test runs a
    psi-active x-CPML forward and pins that the distributed replay reproduces the
    native forward's OWNED psi state as well as the fields. It is the committed
    guard on the Hy/Ey psi axis convention in the forward replay
    (``adjoint/core._forward_magnetic_fields_cpml`` /
    ``_forward_electric_fields_cpml`` and ``_step_state``): the psi keys follow
    ``fdtd/boundary/cpml._CPML_MEMORY_SPECS`` (``psi_hy_z`` = z-family,
    ``psi_hy_x`` = x-family), and a swap stores the advanced z-family psi under
    the x key. Because ``psi_hy_z`` and ``psi_hy_x`` differ by ~3 orders of
    magnitude here, that swap moves a psi field by its full scale (~6e-2), far
    above the gate -- falsification-checked by transposing the unpack order,
    which drives the compared diff to ~O(scale).

    Gate: fixed replay matches the native CPML forward to ~1e-7 relative / ~3e-8
    absolute on the significant psi families (the fused CPML kernel and the torch
    replay differ only in reduction order); rtol=1e-4/atol=1e-6 sits comfortably
    above that drift and orders of magnitude below the swap error.
    """
    from witwin.maxwell.fdtd.boundary.cpml import _CPML_MEMORY_SPECS

    steps = 80
    distributed = DistributedFDTD(
        _pml_scene(),
        frequency=_FREQUENCY,
        parallel=_parallel(cuda_p2p_devices),
        absorber_type="cpml",
    )
    distributed.init_field()
    require_distributed_adjoint_support(distributed)

    field_names = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    psi_names = tuple(_CPML_MEMORY_SPECS)

    checkpoint = capture_distributed_checkpoint(distributed, 0)
    distributed.solve(time_steps=steps, dft_frequency=None, full_field_dft=False)
    forward = {
        shard.rank: {
            name: getattr(shard.solver, name).detach().clone()
            for name in field_names + psi_names
        }
        for shard in distributed.shards
    }

    trajectories = replay_distributed_segment(distributed, checkpoint, 0, steps)

    assert len(distributed.shards) == 2
    field_energy = 0.0
    psi_energy = 0.0
    for shard in distributed.shards:
        replayed = trajectories[shard.rank][steps]
        for name in field_names:
            owned = _owned_slice(shard, name)
            field_energy += float(forward[shard.rank][name][owned].abs().sum().item())
            torch.testing.assert_close(
                replayed[name][owned],
                forward[shard.rank][name][owned],
                rtol=1e-5,
                atol=1e-5,
                msg=f"rank {shard.rank} field {name} owned-state replay mismatch",
            )
        for name in psi_names:
            parent_field = _CPML_MEMORY_SPECS[name][0]
            owned = _owned_slice(shard, parent_field)
            forward_owned = forward[shard.rank][name][owned]
            psi_energy += float(forward_owned.abs().sum().item())
            torch.testing.assert_close(
                replayed[name][owned],
                forward_owned,
                rtol=1e-4,
                atol=1e-6,
                msg=(
                    f"rank {shard.rank} psi {name} owned-state replay mismatch: "
                    f"max abs diff {(replayed[name][owned] - forward_owned).abs().max().item():.3e}"
                ),
            )

    # Non-vacuity: both the fields and the psi memory must carry real signal, so
    # the psi comparison is not passing on all-zero tensors.
    assert field_energy > 0.0
    assert psi_energy > 0.0
    # The two Hy families must be numerically distinct so a key swap is detectable.
    hy_z = max(
        float(forward[shard.rank]["psi_hy_z"].abs().max()) for shard in distributed.shards
    )
    hy_x = max(
        float(forward[shard.rank]["psi_hy_x"].abs().max()) for shard in distributed.shards
    )
    assert hy_z > 3.0 * max(hy_x, 1e-30) or hy_x > 3.0 * max(hy_z, 1e-30), (
        "psi_hy_z and psi_hy_x are comparable; a swap would be undetectable here"
    )


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


def _base_scene():
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.4, 0.4), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.none(),
        device="cpu",
    )


def _cpu_parallel():
    return FDTDParallelConfig(
        devices=("cuda:0", "cuda:1"),
        transport="cuda_p2p",
        gather_fields=False,
        result_device="cuda:0",
    )


def test_distributed_solver_accepts_trainable_box_density_directly():
    """A trainable Box material-region density is the one supported trainable channel;
    the distributed solver's capability-scoped guard must NOT reject it at
    construction (the joint-solve adjoint bridge differentiates it)."""

    scene = _base_scene()
    density = torch.rand((2, 2, 2), requires_grad=True)
    scene.add_material_region(
        mw.MaterialRegion(
            name="design",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.2, 0.2, 0.2)),
            density=density,
            eps_bounds=(1.0, 5.0),
        )
    )
    # Construction validates static capabilities before any hardware use; a
    # trainable Box density is accepted, so this must not raise.
    DistributedFDTD(scene, frequency=_FREQUENCY, parallel=_cpu_parallel())


def test_distributed_solver_rejects_unsupported_trainable_channel_directly():
    """Defense in depth: the distributed solver fails closed on an unsupported
    trainable channel (structure geometry) even when constructed directly, because
    that channel has no verified distributed reverse core."""

    scene = _base_scene()
    scene.add_structure(
        mw.Structure(
            name="dielectric",
            geometry=mw.Box(
                position=torch.zeros(3, requires_grad=True),
                size=(0.2, 0.2, 0.2),
            ),
            material=mw.Material(eps_r=3.0),
        )
    )
    with pytest.raises(ValueError, match="trainable"):
        DistributedFDTD(scene, frequency=_FREQUENCY, parallel=_cpu_parallel())


def _pml_scene():
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
    return scene


def test_checkpoint_accepts_cpml_absorber(cuda_p2p_devices, cuda_memory_cleanup):
    # The distributed CPML adjoint (S4) is now a supported capability: the
    # checkpoint/replay/reverse path accepts the CPML absorbing update, with the
    # x-CPML pinning invariant asserted. capture must succeed and produce a
    # checkpoint carrying the twelve psi fields.
    distributed = DistributedFDTD(
        _pml_scene(),
        frequency=_FREQUENCY,
        parallel=_parallel(cuda_p2p_devices),
        absorber_type="cpml",
    )
    distributed.init_field()
    checkpoint = capture_distributed_checkpoint(distributed, 0)
    for shard in distributed.shards:
        tensors = checkpoint.states[shard.rank].tensors
        assert "psi_ex_y" in tensors and "psi_hz_y" in tensors


def test_checkpoint_rejects_graded_sigma_absorber(cuda_p2p_devices, cuda_memory_cleanup):
    # The legacy graded-sigma absorbers have no verified distributed reverse core
    # and stay rejected at the checkpoint entry point.
    distributed = DistributedFDTD(
        _pml_scene(),
        frequency=_FREQUENCY,
        parallel=_parallel(cuda_p2p_devices),
        absorber_type="pml",
    )
    distributed.init_field()
    with pytest.raises(ValueError, match="CPML absorbing update"):
        capture_distributed_checkpoint(distributed, 0)
