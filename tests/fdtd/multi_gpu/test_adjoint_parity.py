"""Exit-gate parity + finite-difference tests for the distributed joint-solve adjoint.

The distributed FDTD adjoint bridge (fdtd/distributed/adjoint.py) runs the x-slab
forward with per-shard checkpoints, then a transposed reverse step per forward step
(Phase 1 -> transposed magnetic halo -> Phase 2 -> transposed electric halo ->
Phase 3 -> per-shard source-term eps grad), gathers the per-shard grad_eps owned
slices into a global tensor, and runs the existing single-GPU material pullback once
on the logical scene. These tests pin, on vacuum-standard scenes with a Box
MaterialRegion density:

* objective + parameter-gradient parity between the single-GPU adjoint and the
  2-GPU distributed adjoint (the single-GPU bridge is the independently validated
  reference), and
* an independent finite-difference check of the distributed gradient at density
  texels on the x-split and in each shard interior, with the source and the
  point-monitor objective on the interface node.

Gate calibration (recorded):
  - Objective parity uses the plan §7.2 monitor gate (rtol 5e-5 / atol 5e-6).
  - The 1-vs-2-GPU parameter-gradient relative drift measures ~1e-7 (the forward is
    bit-identical and the reverse differs only in gather/pullback reduction order),
    so the gradient gate rtol=1e-4 sits ~3 orders of magnitude above it, with an
    atol floor tied to the gradient magnitude (1e-6 * max|grad|). A broken halo /
    decomposition produces interface-localized error ~1e-2 relative (see the
    replay-parity falsification), far above the gate.
  - The distributed gradient vs central finite differences measures ~1e-5 relative
    on significant texels (float32 forward), so the FD gate 2e-3 sits ~2 orders
    above the float32 FD floor.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd_parallel import FDTDParallelConfig


_FREQUENCY = 1.0e9
_STEPS = 60
_DENS_SHAPE = (5, 4, 4)


def _parallel(devices):
    return FDTDParallelConfig(
        devices=devices,
        transport="cuda_p2p",
        gather_fields=False,
        overlap=False,
        result_device=devices[0],
    )


def _scene(density, *, source_x, monitor_x, device):
    x = np.linspace(-0.5, 0.5, 11, dtype=np.float64)
    y = np.linspace(-0.2, 0.2, 9, dtype=np.float64)
    z = np.linspace(-0.2, 0.2, 9, dtype=np.float64)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.custom(x, y, z),
        boundary=mw.BoundarySpec.none(),
        device=device,
    )
    scene.add_material_region(
        mw.MaterialRegion(
            name="design",
            # Region x in [-0.3, 0.3] over six physical cells, split 3/3 at x = 0
            # (the partition interface of the 11-node grid). The density x-dimension
            # is 5, so texel column 2 rasterizes onto the x = 0 split.
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


def _solve(density_values, *, parallel_devices, source_x, monitor_x, want_grad, checkpoint_stride=None):
    density = density_values.clone().to("cuda:0")
    if want_grad:
        density.requires_grad_(True)
    scene = _scene(density, source_x=source_x, monitor_x=monitor_x, device="cuda:0")
    kwargs = dict(frequency=_FREQUENCY, run_time=mw.TimeConfig(time_steps=_STEPS))
    if parallel_devices is not None:
        kwargs["parallel"] = _parallel(parallel_devices)
    simulation = mw.Simulation.fdtd(scene, **kwargs)
    if checkpoint_stride is not None:
        simulation.config.adjoint_checkpoint_stride = int(checkpoint_stride)
    result = simulation.run()
    spectrum = result.monitors["probe"]["Ez"]
    loss = (spectrum.real ** 2 + spectrum.imag ** 2).sum()
    if want_grad:
        loss.backward()
        return float(loss.detach().cpu()), density.grad.detach().cpu().clone()
    return float(loss.detach().cpu()), None


def _base_density():
    torch.manual_seed(1)
    return (0.4 + 0.2 * torch.rand(_DENS_SHAPE)).double()


def test_objective_and_gradient_parity_single_vs_two_gpu(cuda_p2p_devices, cuda_memory_cleanup):
    base = _base_density()
    single_loss, single_grad = _solve(
        base, parallel_devices=None, source_x=-0.3, monitor_x=0.1, want_grad=True
    )
    dist_loss, dist_grad = _solve(
        base, parallel_devices=cuda_p2p_devices, source_x=-0.3, monitor_x=0.1, want_grad=True
    )

    # Non-vacuity: the objective and the gradient must carry real signal.
    assert abs(single_loss) > 0.0
    assert float(single_grad.abs().max()) > 0.0

    # Objective parity: plan monitor gate.
    torch.testing.assert_close(
        torch.tensor(dist_loss), torch.tensor(single_loss), rtol=5.0e-5, atol=5.0e-6
    )

    # Gradient parity: rtol 1e-4 with an atol floor tied to the gradient magnitude
    # (calibration recorded in the module docstring).
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


def test_interface_finite_difference_split_and_interior(cuda_p2p_devices, cuda_memory_cleanup):
    # Field propagates left->right across the whole region so the on-split column and
    # both shard-interior columns carry a significant gradient.
    base = _base_density()
    _, grad = _solve(
        base, parallel_devices=cuda_p2p_devices, source_x=-0.3, monitor_x=0.1, want_grad=True
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
            base, texel, analytic, parallel_devices=cuda_p2p_devices, source_x=-0.3, monitor_x=0.1
        )
        assert rel <= 2.0e-3, (
            f"texel {texel}: distributed grad {analytic:.6e} vs central FD {fd:.6e} "
            f"(h={h:.0e}) rel {rel:.3e} exceeds the FD gate"
        )


def test_interface_source_and_monitor_finite_difference(cuda_p2p_devices, cuda_memory_cleanup):
    # Source and point-monitor objective both on the x = 0 interface node; FD-check
    # the on-split density texel column.
    base = _base_density()
    _, grad = _solve(
        base, parallel_devices=cuda_p2p_devices, source_x=0.0, monitor_x=0.0, want_grad=True
    )
    texel = (2, 2, 2)
    analytic = float(grad[texel])
    assert abs(analytic) > 0.05 * float(grad.abs().max())
    rel, h, fd = _central_fd_min_rel_error(
        base, texel, analytic, parallel_devices=cuda_p2p_devices, source_x=0.0, monitor_x=0.0
    )
    assert rel <= 2.0e-3, (
        f"interface texel {texel}: distributed grad {analytic:.6e} vs central FD {fd:.6e} "
        f"(h={h:.0e}) rel {rel:.3e} exceeds the FD gate"
    )


def test_gradient_is_checkpoint_stride_invariant(cuda_p2p_devices, cuda_memory_cleanup):
    # Different adjoint checkpoint strides replay different segment lengths but must
    # reconstruct the same forward trajectory (to replay-drift), so the gradient is
    # stride-invariant. stride=1 checkpoints every step (no replay); stride=_STEPS is
    # one segment replayed from step 0.
    base = _base_density()
    _, grad_fine = _solve(
        base, parallel_devices=cuda_p2p_devices, source_x=-0.3, monitor_x=0.1,
        want_grad=True, checkpoint_stride=1,
    )
    _, grad_coarse = _solve(
        base, parallel_devices=cuda_p2p_devices, source_x=-0.3, monitor_x=0.1,
        want_grad=True, checkpoint_stride=_STEPS,
    )
    assert float(grad_fine.abs().max()) > 0.0
    atol_floor = 1.0e-6 * float(grad_fine.abs().max())
    torch.testing.assert_close(grad_coarse, grad_fine, rtol=1.0e-4, atol=atol_floor)


def _solve_capture_grad_eps(base, *, parallel_devices, source_x, monitor_x):
    density = base.clone().to("cuda:0").requires_grad_(True)
    scene = _scene(density, source_x=source_x, monitor_x=monitor_x, device="cuda:0")
    result = mw.Simulation.fdtd(
        scene,
        frequency=_FREQUENCY,
        run_time=mw.TimeConfig(time_steps=_STEPS),
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


def test_repeat_reverse_gradient_is_bitwise_deterministic(cuda_p2p_devices, cuda_memory_cleanup):
    # The distributed reverse product -- the gathered global grad_eps -- is produced
    # by assign-semantics fused kernels, ordered add_ accumulation, and deterministic
    # gather copies, so two identical backward passes must reproduce it bit-for-bit.
    # (The final density gradient additionally passes through torch's grid_sample VJP,
    # whose backward uses atomicAdd and is not bitwise reproducible single- or
    # multi-GPU alike; that shared step is checked with a tight tolerance below.)
    base = _base_density()
    grad_eps_a, dens_grad_a = _solve_capture_grad_eps(
        base, parallel_devices=cuda_p2p_devices, source_x=-0.3, monitor_x=0.1
    )
    grad_eps_b, dens_grad_b = _solve_capture_grad_eps(
        base, parallel_devices=cuda_p2p_devices, source_x=-0.3, monitor_x=0.1
    )

    nonzero = 0.0
    for name in ("Ex", "Ey", "Ez"):
        nonzero += float(grad_eps_a[name].abs().sum().item())
        assert torch.equal(grad_eps_a[name], grad_eps_b[name]), (
            f"grad_eps[{name}] not bitwise reproducible: max abs diff "
            f"{(grad_eps_a[name] - grad_eps_b[name]).abs().max().item():.3e}"
        )
    assert nonzero > 0.0

    # The density gradient (grad_eps -> grid_sample VJP) reproduces to within the
    # torch atomic-add floor (~1e-7 relative), far below the parity gate.
    scale = float(dens_grad_a.abs().max())
    assert scale > 0.0
    torch.testing.assert_close(dens_grad_a, dens_grad_b, rtol=1.0e-5, atol=1.0e-9 * scale)


# ---------------------------------------------------------------------------
# Guard regressions: every unsupported trainable+parallel combination raises at
# prepare() -- CPU-side static validation, before the distributed solver is
# constructed or any shard is allocated. These run without a GPU (the ValueError
# fires before any CUDA work), so they carry no multi-GPU fixture.
# ---------------------------------------------------------------------------

_GUARD_PARALLEL = FDTDParallelConfig(
    devices=("cuda:0", "cuda:1"),
    transport="cuda_p2p",
    gather_fields=False,
    result_device="cuda:0",
)


def _guard_scene(**overrides):
    boundary = overrides.pop("boundary", mw.BoundarySpec.none())
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.uniform(0.1),
        boundary=boundary,
        device="cpu",
    )
    return scene


def _trainable_density_region():
    return mw.MaterialRegion(
        name="design",
        geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.4, 0.2, 0.2)),
        density=torch.rand((3, 3, 3), requires_grad=True),
        eps_bounds=(1.0, 4.0),
    )


def _prepare_parallel(scene, **sim_kwargs):
    simulation = mw.Simulation.fdtd(
        scene, frequency=_FREQUENCY, parallel=_GUARD_PARALLEL, **sim_kwargs
    )
    simulation.prepare()


def test_guard_trainable_geometry_rejected_at_prepare():
    scene = _guard_scene()
    scene.add_structure(
        mw.Structure(
            name="dielectric",
            geometry=mw.Box(position=torch.zeros(3, requires_grad=True), size=(0.2, 0.2, 0.2)),
            material=mw.Material(eps_r=3.0),
        )
    )
    with pytest.raises(ValueError, match="trainable geometry"):
        _prepare_parallel(scene)


def test_guard_trainable_material_perturbation_rejected_at_prepare():
    scene = _guard_scene()
    scene.add_structure(
        mw.Structure(
            name="dielectric",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.2, 0.2, 0.2)),
            material=mw.PerturbationMedium(
                mw.Material(eps_r=3.0),
                perturbation=torch.zeros((2, 2, 2), requires_grad=True),
            ),
        )
    )
    with pytest.raises(ValueError, match="trainable"):
        _prepare_parallel(scene)


def test_guard_cpml_absorber_rejected_at_prepare():
    scene = _guard_scene(boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0))
    scene.add_material_region(_trainable_density_region())
    with pytest.raises(ValueError, match="CPML"):
        _prepare_parallel(scene)


def test_guard_dispersive_medium_rejected_at_prepare():
    scene = _guard_scene()
    scene.add_material_region(_trainable_density_region())
    scene.add_structure(
        mw.Structure(
            name="lorentz",
            geometry=mw.Box(position=(0.2, 0.0, 0.0), size=(0.1, 0.1, 0.1)),
            material=mw.Material(
                eps_r=2.0,
                lorentz_poles=(
                    mw.LorentzPole(delta_eps=0.5, resonance_frequency=2.0e9, gamma=2.0e8),
                ),
            ),
        )
    )
    with pytest.raises(ValueError, match="dispersive"):
        _prepare_parallel(scene)


def test_guard_conductive_medium_rejected_at_prepare():
    scene = _guard_scene()
    scene.add_material_region(_trainable_density_region())
    scene.add_structure(
        mw.Structure(
            name="lossy",
            geometry=mw.Box(position=(0.2, 0.0, 0.0), size=(0.1, 0.1, 0.1)),
            material=mw.Material(eps_r=2.0, sigma_e=0.02),
        )
    )
    with pytest.raises(ValueError, match="conductive"):
        _prepare_parallel(scene)


def test_guard_tiled_plane_monitor_objective_rejected_at_prepare():
    scene = _guard_scene()
    scene.add_material_region(_trainable_density_region())
    scene.add_source(
        mw.PointDipole(
            position=(-0.3, 0.0, 0.0),
            polarization="Ez",
            profile="ideal",
            source_time=mw.GaussianPulse(frequency=_FREQUENCY, fwidth=5.0e8),
            name="drive",
        )
    )
    scene.add_monitor(
        mw.PlaneMonitor("plane", axis="y", position=0.0, fields=("Ez",))
    )
    with pytest.raises(ValueError, match="point-monitor spectra and"):
        _prepare_parallel(scene)


def test_guard_shutoff_rejected_at_prepare():
    scene = _guard_scene()
    scene.add_material_region(_trainable_density_region())
    with pytest.raises(ValueError, match="shutoff"):
        _prepare_parallel(scene, shutoff=1.0e-4)
