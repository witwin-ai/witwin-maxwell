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


@pytest.mark.parametrize("cpml_axes", ("all", "x"))
def test_distributed_cpml_replay_reproduces_forward_owned_psi_and_fields(
    cuda_p2p_devices, cuda_memory_cleanup, cpml_axes, record_property
):
    """Pin all owned fields/psi with both all-face and controlled x-only CPML.

    All-face absorption exercises all twelve memory families, without assuming
    that a point source must create unequal directional maxima. The x-only
    control has exactly zero z-family conductivity and real nonzero x memory;
    it retains the strict factor-three key-swap sensitivity gate below. Dense
    memory makes inactive families genuine full-shape zero tensors, never an
    empty reduction or fabricated psi data. Original replay tolerances remain.
    """
    from witwin.maxwell.fdtd.boundary.cpml import _CPML_MEMORY_SPECS

    steps = 80
    distributed = DistributedFDTD(
        _pml_scene().clone(boundary=_replay_cpml_boundary(cpml_axes)),
        frequency=_FREQUENCY,
        parallel=_parallel(cuda_p2p_devices),
        absorber_type="cpml",
        cpml_config={"memory_mode": "dense"} if cpml_axes == "x" else {},
    )
    distributed.init_field()
    require_distributed_adjoint_support(distributed)
    record_property("requested_cpml_memory_mode",
                    distributed.cpml_config.get("memory_mode", "auto"))
    record_property("resolved_cpml_memory_modes", ",".join(
        shard.solver._cpml_memory_mode for shard in distributed.shards))

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
    # Controlled directional support, rather than accidental unequal maxima.
    if cpml_axes == "x":
        _assert_x_only_psi_support(distributed)
        hy_z = max(float(forward[shard.rank]["psi_hy_z"].abs().max())
                   for shard in distributed.shards)
        hy_x = max(float(forward[shard.rank]["psi_hy_x"].abs().max())
                   for shard in distributed.shards)
        record_property("controlled_hy_z_max", hy_z)
        record_property("controlled_hy_x_max", hy_x)
        assert hy_z == 0.0
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
    scene.subpixel = mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic")
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
        subpixel_samples=mw.SubpixelSpec(samples=2, averaging="polarized"),
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



def _replay_cpml_boundary(axes):
    if axes == "all":
        return mw.BoundarySpec.pml(num_layers=2, strength=1.0)
    if axes == "x":
        return mw.BoundarySpec.faces(default="none", x="pml", num_layers=2, strength=1.0)
    raise ValueError("Replay CPML axes are all or x")


def _assert_x_only_psi_support(distributed):
    for shard in distributed.shards:
        solver = shard.solver
        assert solver._cpml_memory_mode == "dense"
        assert solver.psi_hy_x.shape == solver.psi_hy_z.shape == solver.Hy.shape
        assert solver.psi_hy_z.numel() > 0
        assert bool((solver.cpml_c_h_z == 0).all())
        assert bool((solver.cpml_b_h_z == 1).all())
        assert bool((solver.cpml_c_h_x != 0).any())
        assert int(torch.count_nonzero(solver.psi_hy_z)) == 0


def _owned_cpml_snapshot(distributed, names):
    return {shard.rank: {name: getattr(shard.solver, name).detach().clone()
                         for name in names} for shard in distributed.shards}


def _assert_owned_cpml_parity(distributed, replayed, forward, step):
    from witwin.maxwell.fdtd.boundary.cpml import _CPML_MEMORY_SPECS
    for shard in distributed.shards:
        for name, target in forward[shard.rank].items():
            psi = name in _CPML_MEMORY_SPECS
            parent = _CPML_MEMORY_SPECS[name][0] if psi else name
            owned = _owned_slice(shard, parent)
            torch.testing.assert_close(replayed[shard.rank][step][name][owned],
                target[owned], rtol=1e-4 if psi else 1e-5,
                atol=1e-6 if psi else 1e-5)


@pytest.mark.parametrize("fault", ("hy_keys_swapped", "zero_checkpoint_carry"))
def test_distributed_cpml_replay_detects_actual_key_swap_and_zero_carry(
    cuda_p2p_devices, cuda_memory_cleanup, monkeypatch, record_property, fault
):
    """Falsify actual replay from a native nonzero CPML checkpoint.

    The auxiliary history is produced by eighty real forward steps. No memory
    values are fabricated. A clean step80->81 must match native fields and all
    twelve psi tensors before either deliberate replay defect is introduced.
    """
    import witwin.maxwell.fdtd.distributed.adjoint as replay_module
    from witwin.maxwell.fdtd.boundary.cpml import _CPML_MEMORY_SPECS
    distributed = DistributedFDTD(
        _pml_scene().clone(boundary=_replay_cpml_boundary("x")),
        frequency=_FREQUENCY, parallel=_parallel(cuda_p2p_devices),
        absorber_type="cpml", cpml_config={"memory_mode": "dense"},
    )
    distributed.init_field()
    require_distributed_adjoint_support(distributed)
    distributed.solve(time_steps=81, stop_step=80, dft_frequency=None, full_field_dft=False)
    distributed._synchronize_all()
    _assert_x_only_psi_support(distributed)
    checkpoint = capture_distributed_checkpoint(distributed, 80)
    # Capture clones are queued on the producer compute streams. Order them
    # before default-stream diagnostics and replay consumers inspect the data.
    distributed._synchronize_all()
    for shard in distributed.shards:
        torch.testing.assert_close(checkpoint.states[shard.rank].tensors["psi_hy_x"],
                                   shard.solver.psi_hy_x, rtol=0, atol=0)
    record_property("native_checkpoint_hy_x_max", max(
        float(checkpoint.states[shard.rank].tensors["psi_hy_x"].abs().max())
        for shard in distributed.shards))
    # A genuine carried term must exceed the existing absolute psi tolerance.
    carried = max(float((shard.solver.cpml_b_h_x[:, None, None] *
        checkpoint.states[shard.rank].tensors["psi_hy_x"]).abs().max())
        for shard in distributed.shards)
    record_property("native_propagated_hy_x_carry", carried)
    assert carried > 10.0e-6
    names = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz") + tuple(_CPML_MEMORY_SPECS)
    distributed._advance_one_step(80, overlap_active=False)
    distributed._synchronize_all()
    forward = _owned_cpml_snapshot(distributed, names)
    clean = replay_distributed_segment(distributed, checkpoint, 80, 81)
    _assert_owned_cpml_parity(distributed, clean, forward, 1)
    if fault == "hy_keys_swapped":
        original = replay_module._forward_magnetic_fields_cpml
        def swapped(*args, **kwargs):
            fields, psi = original(*args, **kwargs)
            psi = dict(psi)
            psi["psi_hy_x"], psi["psi_hy_z"] = psi["psi_hy_z"], psi["psi_hy_x"]
            return fields, psi
        monkeypatch.setattr(replay_module, "_forward_magnetic_fields_cpml", swapped)
    else:
        original = replay_module.clone_checkpoint_tensors
        def zero_carry(state):
            cloned = original(state)
            for name in _CPML_MEMORY_SPECS:
                cloned[name].zero_()
            return cloned
        monkeypatch.setattr(replay_module, "clone_checkpoint_tensors", zero_carry)
    corrupted = replay_distributed_segment(distributed, checkpoint, 80, 81)
    errors = []
    for shard in distributed.shards:
        owned = _owned_slice(shard, "Hy")
        difference = (corrupted[shard.rank][1]["psi_hy_x"][owned] -
                      forward[shard.rank]["psi_hy_x"][owned]).abs()
        errors.append(float(difference.max()))
    record_property("actual_mutant_hy_x_max_error", max(errors))
    assert max(errors) > 10.0e-6
    with pytest.raises(AssertionError):
        _assert_owned_cpml_parity(distributed, corrupted, forward, 1)
