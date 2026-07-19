"""Exit-gate parity + FD + determinism tests for the distributed CPML adjoint (S4).

The distributed FDTD adjoint bridge (fdtd/distributed/adjoint.py) now covers the
CPML absorbing update as well as the open-boundary standard update. On the CPML
branch the forward replay threads the twelve psi memory fields through psi-aware
magnetic/electric half-steps, and the reverse loop runs

    reverse_cpml_phase_electric  (electric adjoint -> pre-step E/psi_e + eps grad,
                                  folding curl(H) into the mid-step H adjoint)
    -> transposed magnetic halo (Hy/Hz)
    -> reverse_cpml_phase_magnetic (magnetic-decay + psi_h pullback -> pre-step
                                    H/psi_h, folding curl(E) into pre-step E)
    -> transposed electric halo (Ey/Ez)
    -> per-shard source-term eps gradient

with NO psi halo. That the two Yee field halos alone reproduce the single-GPU
gradient is the S4 audit's central prediction: the partition pins every x-CPML
region to the outer shards, so internal-face psi is inactive and the
cross-interface curl(H)/curl(E) coupling rides the field halos through the
``adj_d`` folds. These tests pin, on an x-CPML dielectric Box-density scene:

* 1-vs-2-GPU objective + gradient parity (single-GPU adjoint is the validated
  reference),
* an independent central finite-difference check of the distributed gradient on
  the on-split density texel column and each shard interior,
* the load-bearing no-psi-halo confirmation: no-op'ing either reverse field halo
  drives the 2-GPU gradient far off the single-GPU reference (so the field halos
  carry the full cross-interface coupling and no separate psi halo is needed),
* bitwise repeat-determinism of the gathered grad_eps, and
* the defensive x-CPML pinning assertion (positive + violated).

Gate calibration (measured on this scene, STEPS=50, 2x A6000, recorded in
docs/assessments/e3-distributed-adjoint-acceptance-2026-07-19.md):
  - Objective parity: bit-identical (rel 0.0); gated at the plan monitor gate
    (rtol 5e-5 / atol 5e-6).
  - Gradient parity: rel drift ~1.2e-7 (forward bit-identical, reverse differs
    only in gather/pullback reduction order), so the gate rtol=1e-4 with an atol
    floor 1e-6*max|grad| sits ~3 orders above it.
  - Distributed grad vs central FD: <=7.4e-5 relative on significant texels
    (float32 forward), so the FD gate 2e-3 sits ~1.5 orders above the FD floor.
  - No-psi-halo falsification: baseline rel 1.2e-7; no magnetic-adjoint halo rel
    0.96; no electric-adjoint halo rel 0.016 -- both >=2 orders above the gate.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from types import SimpleNamespace

import witwin.maxwell as mw
from witwin.maxwell.fdtd_parallel import FDTDParallelConfig


_FREQUENCY = 1.0e9
_STEPS = 50
_DENS_SHAPE = (5, 4, 4)
# Field-halo rel-error threshold that separates a working reverse (~1e-7) from a
# no-op'd halo (>=1e-2): comfortably above the reduction-order drift and below the
# smallest halo-bug error, so the falsification is non-vacuous.
_HALO_FALSIFICATION_MIN_REL = 1.0e-3


def _parallel(devices):
    return FDTDParallelConfig(
        devices=devices,
        transport="cuda_p2p",
        gather_fields=False,
        overlap=False,
        result_device=devices[0],
    )


def _scene(density, *, source_x, monitor_x, device):
    # 21 x-nodes (20 physical cells) + a 4-cell CPML pad on each x face -> Nx=29,
    # cell_count=28: rank 0 owns cells [0, 14) (all four low x-PML), rank 1 owns
    # [14, 28) (all four high x-PML), interface at global cell 14 == x=0.0. The
    # design region x in [-0.3, 0.3] straddles the interface entirely inside the
    # physical (non-PML) interior, so its on-split and shard-interior texels carry
    # a significant gradient and the x-CPML pinning invariant holds by construction.
    x = np.linspace(-0.6, 0.6, 21, dtype=np.float64)
    y = np.linspace(-0.2, 0.2, 9, dtype=np.float64)
    z = np.linspace(-0.2, 0.2, 9, dtype=np.float64)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.custom(x, y, z),
        boundary=mw.BoundarySpec.pml(num_layers=4),
        device=device,
    )
    scene.add_material_region(
        mw.MaterialRegion(
            name="design",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.6, 0.3, 0.3)),
            density=density,
            eps_bounds=(1.0, 6.0),
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(source_x, 0.0, 0.0),
            polarization="Ez",
            profile="ideal",
            source_time=mw.GaussianPulse(frequency=_FREQUENCY, fwidth=5.0e8),
            name="drive",
        )
    )
    scene.add_monitor(mw.PointMonitor("probe", (monitor_x, 0.0, 0.0), fields=("Ez",)))
    return scene


def _solve(density_values, *, parallel_devices, source_x, monitor_x, want_grad):
    density = density_values.clone().to("cuda:0")
    if want_grad:
        density.requires_grad_(True)
    scene = _scene(density, source_x=source_x, monitor_x=monitor_x, device="cuda:0")
    kwargs = dict(
        frequency=_FREQUENCY,
        run_time=mw.TimeConfig(time_steps=_STEPS),
        absorber="cpml",
    )
    if parallel_devices is not None:
        kwargs["parallel"] = _parallel(parallel_devices)
    result = mw.Simulation.fdtd(scene, **kwargs).run()
    spectrum = result.monitors["probe"]["Ez"]
    loss = (spectrum.real ** 2 + spectrum.imag ** 2).sum()
    if want_grad:
        loss.backward()
        return float(loss.detach().cpu()), density.grad.detach().cpu().clone()
    return float(loss.detach().cpu()), None


def _base_density():
    torch.manual_seed(1)
    return (0.4 + 0.2 * torch.rand(_DENS_SHAPE)).double()


def test_cpml_objective_and_gradient_parity_single_vs_two_gpu(
    cuda_p2p_devices, cuda_memory_cleanup
):
    base = _base_density()
    single_loss, single_grad = _solve(
        base, parallel_devices=None, source_x=-0.18, monitor_x=0.18, want_grad=True
    )
    dist_loss, dist_grad = _solve(
        base, parallel_devices=cuda_p2p_devices, source_x=-0.18, monitor_x=0.18, want_grad=True
    )

    # Non-vacuity: objective and gradient must carry real signal, and the design
    # region must actually be dielectric-active under the CPML absorber.
    assert abs(single_loss) > 0.0
    assert float(single_grad.abs().max()) > 0.0

    # Objective parity (plan monitor gate); measured bit-identical.
    torch.testing.assert_close(
        torch.tensor(dist_loss), torch.tensor(single_loss), rtol=5.0e-5, atol=5.0e-6
    )
    # Gradient parity: rtol 1e-4 with an atol floor tied to the gradient magnitude.
    atol_floor = 1.0e-6 * float(single_grad.abs().max())
    torch.testing.assert_close(dist_grad, single_grad, rtol=1.0e-4, atol=atol_floor)


def _central_fd_min_rel_error(base, texel, analytic, *, parallel_devices, source_x, monitor_x):
    best = None
    for h in (2.0e-3, 1.0e-3, 5.0e-4):
        plus = base.clone()
        plus[texel] += h
        minus = base.clone()
        minus[texel] -= h
        loss_plus, _ = _solve(
            plus, parallel_devices=parallel_devices, source_x=source_x, monitor_x=monitor_x, want_grad=False
        )
        loss_minus, _ = _solve(
            minus, parallel_devices=parallel_devices, source_x=source_x, monitor_x=monitor_x, want_grad=False
        )
        fd = (loss_plus - loss_minus) / (2.0 * h)
        rel = abs(fd - analytic) / max(abs(fd), abs(analytic), 1e-30)
        if best is None or rel < best[0]:
            best = (rel, h, fd)
    return best


def test_cpml_interface_finite_difference_split_and_interior(
    cuda_p2p_devices, cuda_memory_cleanup
):
    # Cross-shard objective (source in the left shard, monitor in the right):
    # independent FD check of the distributed CPML gradient on the on-split column
    # and each shard interior.
    base = _base_density()
    _, grad = _solve(
        base, parallel_devices=cuda_p2p_devices, source_x=-0.18, monitor_x=0.18, want_grad=True
    )
    grad_scale = float(grad.abs().max())
    assert grad_scale > 0.0

    # (a) texel column exactly on the x-split; (b) one texel interior to each shard.
    for texel in ((1, 2, 2), (2, 2, 2), (3, 2, 2)):
        analytic = float(grad[texel])
        assert abs(analytic) > 0.05 * grad_scale, (
            f"texel {texel} is insensitive (|grad|={abs(analytic):.3e}); FD would be vacuous"
        )
        rel, h, fd = _central_fd_min_rel_error(
            base, texel, analytic, parallel_devices=cuda_p2p_devices, source_x=-0.18, monitor_x=0.18
        )
        assert rel <= 2.0e-3, (
            f"texel {texel}: distributed CPML grad {analytic:.6e} vs central FD {fd:.6e} "
            f"(h={h:.0e}) rel {rel:.3e} exceeds the FD gate"
        )


def _no_op_halo(self, shards, adjoint_states):
    return None


def test_cpml_no_field_halo_falsification(
    cuda_p2p_devices, cuda_memory_cleanup, monkeypatch
):
    """Load-bearing no-psi-halo confirmation.

    With both Yee field halos active the 2-GPU CPML gradient matches single-GPU to
    reduction-order drift WITHOUT any psi halo (S4 audit). No-op'ing either reverse
    field halo must then drive the 2-GPU gradient far off the single-GPU reference,
    proving the field halos carry the full cross-interface curl(H)/curl(E) coupling
    -- including the psi-derived ``adj_d`` folds -- so a separate psi halo would be
    redundant. If a no-op'd halo left parity intact, the halo (or a missing psi
    halo) would be doing no work and the audit's prediction would be false.
    """

    from witwin.maxwell.fdtd.distributed.transport import CudaP2PHaloTransport

    base = _base_density()
    _, single_grad = _solve(
        base, parallel_devices=None, source_x=-0.18, monitor_x=0.18, want_grad=True
    )
    grad_scale = float(single_grad.abs().max())
    assert grad_scale > 0.0

    # Baseline: both halos active -> parity holds without a psi halo.
    _, base_grad = _solve(
        base, parallel_devices=cuda_p2p_devices, source_x=-0.18, monitor_x=0.18, want_grad=True
    )
    base_rel = float((base_grad - single_grad).abs().max()) / grad_scale
    assert base_rel < 1.0e-4, f"baseline CPML parity unexpectedly loose: rel {base_rel:.3e}"

    # No magnetic-adjoint halo (Hy/Hz mid-step adjoint) -> parity breaks.
    monkeypatch.setattr(
        CudaP2PHaloTransport, "exchange_magnetic_adjoint", _no_op_halo, raising=True
    )
    _, grad_no_mag = _solve(
        base, parallel_devices=cuda_p2p_devices, source_x=-0.18, monitor_x=0.18, want_grad=True
    )
    rel_no_mag = float((grad_no_mag - single_grad).abs().max()) / grad_scale
    monkeypatch.undo()
    assert rel_no_mag > _HALO_FALSIFICATION_MIN_REL, (
        f"no-op magnetic-adjoint halo left parity intact (rel {rel_no_mag:.3e}); the "
        "field halo is not load-bearing, contradicting the no-psi-halo audit"
    )

    # No electric-adjoint halo (pre-step Ey/Ez adjoint) -> parity breaks.
    monkeypatch.setattr(
        CudaP2PHaloTransport, "exchange_electric_adjoint", _no_op_halo, raising=True
    )
    _, grad_no_elec = _solve(
        base, parallel_devices=cuda_p2p_devices, source_x=-0.18, monitor_x=0.18, want_grad=True
    )
    rel_no_elec = float((grad_no_elec - single_grad).abs().max()) / grad_scale
    monkeypatch.undo()
    assert rel_no_elec > _HALO_FALSIFICATION_MIN_REL, (
        f"no-op electric-adjoint halo left parity intact (rel {rel_no_elec:.3e}); the "
        "field halo is not load-bearing, contradicting the no-psi-halo audit"
    )


def _solve_capture_grad_eps(base, *, parallel_devices, source_x, monitor_x):
    density = base.clone().to("cuda:0").requires_grad_(True)
    scene = _scene(density, source_x=source_x, monitor_x=monitor_x, device="cuda:0")
    result = mw.Simulation.fdtd(
        scene,
        frequency=_FREQUENCY,
        run_time=mw.TimeConfig(time_steps=_STEPS),
        absorber="cpml",
        parallel=_parallel(parallel_devices),
    ).run()
    spectrum = result.monitors["probe"]["Ez"]
    loss = (spectrum.real ** 2 + spectrum.imag ** 2).sum()
    loss.backward()
    grad_eps = {
        name: value.detach().clone()
        for name, value in result.solver._adjoint_bridge._last_global_grad_eps.items()
    }
    return grad_eps, density.grad.detach().cpu().clone()


def test_cpml_repeat_reverse_gradient_is_bitwise_deterministic(
    cuda_p2p_devices, cuda_memory_cleanup
):
    # The distributed CPML reverse product -- the gathered global grad_eps -- is
    # produced by assign-semantics fused CPML kernels, ordered add_ accumulation
    # (including the psi pullbacks), and deterministic gather copies, so two
    # identical backward passes must reproduce it bit-for-bit. (The final density
    # gradient additionally passes through torch's grid_sample VJP, whose backward
    # uses atomicAdd and is not bitwise reproducible; that shared step is checked
    # with a tight tolerance below.)
    base = _base_density()
    grad_eps_a, dens_grad_a = _solve_capture_grad_eps(
        base, parallel_devices=cuda_p2p_devices, source_x=-0.18, monitor_x=0.18
    )
    grad_eps_b, dens_grad_b = _solve_capture_grad_eps(
        base, parallel_devices=cuda_p2p_devices, source_x=-0.18, monitor_x=0.18
    )

    nonzero = 0.0
    for name in ("Ex", "Ey", "Ez"):
        nonzero += float(grad_eps_a[name].abs().sum().item())
        assert torch.equal(grad_eps_a[name], grad_eps_b[name]), (
            f"grad_eps[{name}] not bitwise reproducible: max abs diff "
            f"{(grad_eps_a[name] - grad_eps_b[name]).abs().max().item():.3e}"
        )
    assert nonzero > 0.0

    scale = float(dens_grad_a.abs().max())
    assert scale > 0.0
    torch.testing.assert_close(dens_grad_a, dens_grad_b, rtol=1.0e-5, atol=1.0e-9 * scale)


def test_cpml_trainable_parallel_scene_prepares(cuda_p2p_devices, cuda_memory_cleanup):
    """A supported x-CPML trainable Box-density parallel scene prepares successfully.

    Positive counterpart to the graded-sigma guard regression: prepare() runs the
    relaxed ``require_distributed_adjoint_support`` (CPML accepted, x-CPML pinning
    asserted) and must accept the supported CPML point-monitor scene without
    over-rejecting.
    """

    from witwin.maxwell.fdtd.distributed import DistributedFDTD

    density = _base_density().clone().to("cuda:0").requires_grad_(True)
    scene = _scene(density, source_x=-0.18, monitor_x=0.18, device="cuda:0")
    simulation = mw.Simulation.fdtd(
        scene,
        frequency=_FREQUENCY,
        run_time=mw.TimeConfig(time_steps=_STEPS),
        absorber="cpml",
        parallel=_parallel(cuda_p2p_devices),
    )
    prepared = simulation.prepare()
    assert isinstance(prepared.solver, DistributedFDTD)
    assert prepared.solver.active_absorber_type == "cpml"
    assert all(shard.solver.uses_cpml for shard in prepared.solver.shards)


# ---------------------------------------------------------------------------
# Defensive x-CPML pinning assertion (CPU-side unit test + falsification). No GPU
# fixture: the assertion is pure partition geometry.
# ---------------------------------------------------------------------------


def _fake_distributed(layouts, *, low, high, cell_count):
    plan = SimpleNamespace(
        shard_layouts=tuple(layouts),
        low_pml_cells=low,
        high_pml_cells=high,
        cell_count=cell_count,
    )
    return SimpleNamespace(partition_plan=plan)


def _fake_layout(rank, start, stop):
    return SimpleNamespace(rank=rank, global_cell_owned=slice(start, stop))


def test_x_pml_pinning_assertion_accepts_pinned_partition():
    from witwin.maxwell.fdtd.distributed.adjoint import (
        _assert_x_pml_pinned_to_outer_shards,
    )

    # rank 0 owns [0,14) (low PML [0,4)), rank 1 owns [14,28) (high PML [24,28)).
    distributed = _fake_distributed(
        [_fake_layout(0, 0, 14), _fake_layout(1, 14, 28)],
        low=4,
        high=4,
        cell_count=28,
    )
    _assert_x_pml_pinned_to_outer_shards(distributed)  # must not raise


def test_x_pml_pinning_assertion_rejects_interior_low_pml():
    from witwin.maxwell.fdtd.distributed.adjoint import (
        _assert_x_pml_pinned_to_outer_shards,
    )

    # Fabricate a 3-shard partition where the interior shard (rank 1) owns a low
    # x-PML cell -- exactly the invariant a future partition change could break.
    distributed = _fake_distributed(
        [_fake_layout(0, 0, 2), _fake_layout(1, 2, 16), _fake_layout(2, 16, 28)],
        low=4,
        high=4,
        cell_count=28,
    )
    with pytest.raises(RuntimeError, match="low x-PML"):
        _assert_x_pml_pinned_to_outer_shards(distributed)


def test_x_pml_pinning_assertion_rejects_non_last_high_pml():
    from witwin.maxwell.fdtd.distributed.adjoint import (
        _assert_x_pml_pinned_to_outer_shards,
    )

    # rank 0 stretches into the high x-PML band [24,28) it must not own.
    distributed = _fake_distributed(
        [_fake_layout(0, 0, 26), _fake_layout(1, 26, 28)],
        low=4,
        high=4,
        cell_count=28,
    )
    with pytest.raises(RuntimeError, match="high x-PML"):
        _assert_x_pml_pinned_to_outer_shards(distributed)
