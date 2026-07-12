"""Finite-difference gradient validation for the multi-source FDTD adjoint.

The adjoint previously refused any scene with more than one source
(``FDTD adjoint currently supports at most one source per scene``). Lifting that
cap requires the reverse pass to (a) reconstruct the forward field from *every*
source during checkpoint replay and (b) drive each source's eps-gradient with
that source's own waveform / CW frequency rather than the primary source's.

These tests exercise both requirements with genuine two-source scenes:

* two ``PointDipole`` excitations at *different* frequencies (per-term source
  spectrum threading), validated per-element against central differences;
* a genuine 2-port ``ModeSource`` scene (the mode-source explicit-VJP
  bookkeeping generalized to any-source-is-a-mode);
* forward-consistency assertions that the second source actually shifts the
  solution and the gradient, so a near-linear regime cannot fake a pass;
* the multi-source ``normalize_source`` contract.
"""

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.adjoint import _FDTDGradientBridge
from tests.gradients import fdtd_adjoint_baselines as adjoint_baselines

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")

_FD_DELTA = 1.0e-2


def _abs2(z):
    if z.is_complex():
        return (z * z.conj()).real
    return z * z


def _central_difference_per_element(model, loss_fn, delta=_FD_DELTA):
    logits = model.logits
    flat = logits.detach().flatten()
    fd_grad = torch.zeros_like(flat)
    for i in range(flat.numel()):
        with torch.no_grad():
            saved = flat[i].clone()
            flat[i] = saved + delta
            logits.copy_(flat.reshape(logits.shape))
        loss_plus = loss_fn()
        with torch.no_grad():
            flat[i] = saved - delta
            logits.copy_(flat.reshape(logits.shape))
        loss_minus = loss_fn()
        fd_grad[i] = (loss_plus - loss_minus) / (2.0 * delta)
        with torch.no_grad():
            flat[i] = saved
            logits.copy_(flat.reshape(logits.shape))
    return fd_grad.reshape(logits.shape)


# ---------------------------------------------------------------------------
# Two point dipoles at different frequencies
# ---------------------------------------------------------------------------

class _TwoPointDipoleScene(mw.SceneModule):
    """Two PointDipole sources at distinct frequencies illuminating one slab."""

    def __init__(self, *, enable_second=True, shape=(2, 2, 2), init=0.0):
        super().__init__()
        self.enable_second = bool(enable_second)
        self.logits = torch.nn.Parameter(torch.full(shape, float(init), device="cuda"))

    def to_scene(self):
        density = torch.sigmoid(self.logits)
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6))),
            grid=mw.GridSpec.uniform(0.12),
            boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
            device="cuda",
        )
        scene.add_material_region(
            mw.MaterialRegion(
                name="design",
                # Half-cell offset keeps the voxel window off the node knife edge.
                geometry=mw.Box(position=(0.06, 0.06, 0.18), size=(0.24, 0.24, 0.24)),
                density=density,
                eps_bounds=(1.0, 6.0),
                mu_bounds=(1.0, 1.0),
            )
        )
        scene.add_source(
            mw.PointDipole(
                position=(0.06, 0.0, -0.18),
                polarization="Ez",
                width=0.04,
                source_time=mw.GaussianPulse(frequency=1.0e9, fwidth=0.25e9, amplitude=50.0),
            )
        )
        if self.enable_second:
            scene.add_source(
                mw.PointDipole(
                    position=(-0.06, 0.0, -0.18),
                    polarization="Ez",
                    width=0.04,
                    source_time=mw.GaussianPulse(frequency=1.4e9, fwidth=0.25e9, amplitude=50.0),
                )
            )
        scene.add_monitor(mw.PointMonitor("probe", (0.0, 0.0, 0.18), fields=("Ez",)))
        return scene


def _build_multifreq_sim(model, *, time_steps=200, freqs=(1.0e9, 1.4e9)):
    return mw.Simulation.fdtd(
        model,
        frequencies=list(freqs),
        run_time=mw.TimeConfig(time_steps=time_steps),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=True,
    )


def _two_dipole_multifreq_loss(model, *, time_steps=200, freqs=(1.0e9, 1.4e9)):
    result = _build_multifreq_sim(model, time_steps=time_steps, freqs=freqs).run()
    total = torch.zeros((), device="cuda")
    for index in range(len(freqs)):
        total = total + _abs2(result.monitor("probe", freq_index=index)["data"])
    return total


@_CUDA
def test_two_point_dipole_multifreq_gradient_matches_fd():
    """Per-element FD validation for two dipoles at different frequencies.

    Each source carries its own GaussianPulse spectrum, so a reverse pass that
    reused the primary source's spectrum for the second source would produce a
    wrong eps-gradient. The dominant design voxels match the central-difference
    gradient to well under 1e-3.
    """
    model = _TwoPointDipoleScene(enable_second=True).cuda()

    def loss_fn():
        return _two_dipole_multifreq_loss(model)

    loss = loss_fn()
    loss.backward()
    backward_grad = model.logits.grad.detach().clone()
    model.logits.grad = None

    fd_grad = _central_difference_per_element(model, loss_fn)

    scale = fd_grad.abs().max().item()
    assert scale > 0.0
    dominant = fd_grad.abs() > 0.2 * scale
    assert int(dominant.sum()) >= 2, "need at least two trainable voxels with strong gradient"
    dominant_rel_err = ((backward_grad - fd_grad).abs()[dominant] / fd_grad.abs()[dominant]).max().item()
    assert dominant_rel_err < 1.0e-3, (
        f"dominant-voxel relative error {dominant_rel_err:.3e} exceeds 1e-3"
    )
    # Near-zero voxels have poorer relative FD precision; bound them by scale.
    assert torch.allclose(backward_grad, fd_grad, rtol=2.0e-3, atol=scale * 3.0e-3), (
        f"max abs error {(backward_grad - fd_grad).abs().max().item():.3e}, scale {scale:.3e}"
    )


@_CUDA
def test_second_point_source_shifts_solution_and_gradient():
    """The second source (at a different frequency) genuinely changes physics.

    Without it, the 1.4 GHz DFT bin sees only spectral leakage from the 1 GHz
    source; with it, that bin is driven directly. Both the objective and the
    material gradient must change, ruling out a silent single-source fallback.
    """
    freqs = (1.0e9, 1.4e9)

    def transmitted_at_f1(enable_second):
        model = _TwoPointDipoleScene(enable_second=enable_second).cuda()
        result = _build_multifreq_sim(model, freqs=freqs).run()
        return abs(result.monitor("probe", freq_index=1)["data"].detach().cpu().item())

    one = transmitted_at_f1(False)
    two = transmitted_at_f1(True)
    # The 1.4 GHz drive interferes with the 1 GHz source's spectral tail at the
    # probe, so the response can rise or fall, but it must change substantially.
    assert abs(two - one) > 0.3 * max(one, two), (
        f"second source barely changes the 1.4 GHz response: |E|_1src={one:.3e}, |E|_2src={two:.3e}"
    )

    def grad_of(enable_second):
        model = _TwoPointDipoleScene(enable_second=enable_second).cuda()
        loss = _two_dipole_multifreq_loss(model, freqs=freqs)
        loss.backward()
        return model.logits.grad.detach().clone()

    grad_one = grad_of(False)
    grad_two = grad_of(True)
    rel = (grad_two - grad_one).abs().max().item() / (grad_two.abs().max().item() + 1e-30)
    assert rel > 0.1, f"second source barely changes the gradient (rel change {rel:.3e})"


@_CUDA
def test_two_point_dipole_uses_explicit_cpml_backend():
    """Multi-source scenes run through the analytic CPML reverse, not a fallback.

    This confirms the single-source cap is lifted (the bridge no longer raises)
    and that the multi-source reverse is handled by the hand-written CPML
    backend at every step rather than the torch-VJP fallback.
    """
    model = _TwoPointDipoleScene(enable_second=True).cuda()
    bridge = _FDTDGradientBridge(_build_multifreq_sim(model, time_steps=48))
    bridge.forward(tuple(bridge.material_inputs))
    profile = bridge.backward_profile()

    counts = profile["reverse_backend_counts"]
    assert counts.get(adjoint_baselines.expected_cpml_reverse_backend(), 0) == bridge._time_steps
    assert all(name.startswith("native_") for name in counts)


# ---------------------------------------------------------------------------
# Genuine 2-port mode-source scene
# ---------------------------------------------------------------------------

class _TwoPortModeScene(mw.SceneModule):
    """Waveguide driven from both ends by ModeSource excitations."""

    def __init__(self, *, enable_second=True, shape=(2, 1, 1), init=0.0):
        super().__init__()
        self.enable_second = bool(enable_second)
        self.logits = torch.nn.Parameter(torch.full(shape, float(init), device="cuda"))

    def to_scene(self):
        density = torch.sigmoid(self.logits)
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.72, 0.72), (-0.60, 0.60), (-0.60, 0.60))),
            grid=mw.GridSpec.uniform(0.12),
            boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
            device="cuda",
        )
        scene.add_structure(
            mw.Structure(
                name="waveguide",
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(1.20, 0.36, 0.36)),
                material=mw.Material(eps_r=8.0),
            )
        )
        scene.add_material_region(
            mw.MaterialRegion(
                name="design",
                geometry=mw.Box(position=(0.06, 0.0, 0.0), size=(0.24, 0.12, 0.12)),
                density=density,
                eps_bounds=(8.0, 12.0),
                mu_bounds=(1.0, 1.0),
            )
        )
        scene.add_source(
            mw.ModeSource(
                position=(-0.24, 0.0, 0.0),
                size=(0.0, 0.36, 0.36),
                polarization="Ez",
                direction="+",
                source_time=mw.CW(frequency=1.0e9, amplitude=40.0),
                name="port1",
            )
        )
        if self.enable_second:
            scene.add_source(
                mw.ModeSource(
                    position=(0.24, 0.0, 0.0),
                    size=(0.0, 0.36, 0.36),
                    polarization="Ez",
                    direction="-",
                    source_time=mw.CW(frequency=1.0e9, amplitude=40.0),
                    name="port2",
                )
            )
        scene.add_monitor(mw.PointMonitor("probe", (0.0, 0.0, 0.0), fields=("Ez",)))
        return scene


def _mode_loss(model, *, time_steps=192):
    result = mw.Simulation.fdtd(
        model,
        frequencies=[1.0e9],
        run_time=mw.TimeConfig(time_steps=time_steps),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
    ).run()
    return _abs2(result.monitor("probe")["data"])


@_CUDA
def test_two_mode_source_gradient_matches_fd():
    """Per-element FD validation for a two-port ModeSource scene.

    Exercises the mode-source explicit-VJP path generalized to more than one
    mode source: the per-source term cache and retain-graph counter must span
    both ports. CW mode sources have a coarser FD floor than pulsed dipoles, so
    the tolerance mirrors the single-mode adjoint test.
    """
    model = _TwoPortModeScene(enable_second=True).cuda()

    def loss_fn():
        return _mode_loss(model)

    loss = loss_fn()
    loss.backward()
    backward_grad = model.logits.grad.detach().clone()
    model.logits.grad = None
    assert torch.isfinite(backward_grad).all()
    assert (backward_grad != 0).all(), "every design voxel must receive a gradient from the two ports"

    fd_grad = _central_difference_per_element(model, loss_fn)

    scale = fd_grad.abs().max().item()
    assert scale > 0.0
    dominant_index = int(fd_grad.abs().flatten().argmax())
    dominant_rel_err = (
        (backward_grad.flatten()[dominant_index] - fd_grad.flatten()[dominant_index]).abs().item()
        / abs(fd_grad.flatten()[dominant_index].item())
    )
    assert dominant_rel_err < 2.0e-2, f"dominant-voxel relative error {dominant_rel_err:.3e}"
    assert torch.allclose(backward_grad, fd_grad, rtol=1.2e-1, atol=scale * 5.0e-2), (
        f"max abs error {(backward_grad - fd_grad).abs().max().item():.3e}, scale {scale:.3e}"
    )


@_CUDA
def test_two_mode_source_shifts_solution():
    """Adding the second port changes the standing-wave field at the probe."""
    one = _mode_loss(_TwoPortModeScene(enable_second=False).cuda()).item()
    two = _mode_loss(_TwoPortModeScene(enable_second=True).cuda()).item()
    assert abs(two - one) > 0.2 * max(one, two), (
        f"second port barely changes the field: one={one:.3e}, two={two:.3e}"
    )


# ---------------------------------------------------------------------------
# Multi-source normalize_source contract
# ---------------------------------------------------------------------------

class _TwoDipoleNormalizeScene(mw.SceneModule):
    def __init__(self, *, distinct_spectrum):
        super().__init__()
        self.distinct_spectrum = bool(distinct_spectrum)
        self.logits = torch.nn.Parameter(torch.zeros((2, 1, 1), device="cuda"))

    def to_scene(self):
        density = torch.sigmoid(self.logits)
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6))),
            grid=mw.GridSpec.uniform(0.12),
            boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
            device="cuda",
        )
        scene.add_material_region(
            mw.MaterialRegion(
                name="design",
                geometry=mw.Box(position=(0.06, 0.06, 0.18), size=(0.24, 0.12, 0.12)),
                density=density,
                eps_bounds=(1.0, 6.0),
                mu_bounds=(1.0, 1.0),
            )
        )
        second_frequency = 1.4e9 if self.distinct_spectrum else 1.0e9
        scene.add_source(
            mw.PointDipole(
                position=(0.06, 0.0, -0.18),
                polarization="Ez",
                width=0.04,
                source_time=mw.GaussianPulse(frequency=1.0e9, fwidth=0.25e9, amplitude=50.0),
            )
        )
        scene.add_source(
            mw.PointDipole(
                position=(-0.06, 0.0, -0.18),
                polarization="Ez",
                width=0.04,
                source_time=mw.GaussianPulse(frequency=second_frequency, fwidth=0.25e9, amplitude=50.0),
            )
        )
        scene.add_monitor(mw.PointMonitor("probe", (0.0, 0.0, 0.18), fields=("Ez",)))
        return scene


def _run_normalized(model, *, time_steps=48):
    return mw.Simulation.fdtd(
        model,
        frequencies=[1.0e9],
        run_time=mw.TimeConfig(time_steps=time_steps),
        spectral_sampler=mw.SpectralSampler(window="none", normalize_source=True),
    ).run()


@_CUDA
def test_normalize_source_shared_spectrum_runs_multi_source():
    """normalize_source is well-defined when all sources share one spectrum."""
    model = _TwoDipoleNormalizeScene(distinct_spectrum=False).cuda()
    result = _run_normalized(model)
    data = result.monitor("probe")["data"]
    loss = _abs2(data)
    loss.backward()
    assert model.logits.grad is not None
    assert torch.isfinite(model.logits.grad).all()


@_CUDA
def test_normalize_source_distinct_spectrum_raises_with_physical_reason():
    """Distinct source spectra make source normalization physically ill-defined."""
    model = _TwoDipoleNormalizeScene(distinct_spectrum=True).cuda()
    with pytest.raises(NotImplementedError, match="single incident source spectrum"):
        _run_normalized(model)


# ---------------------------------------------------------------------------
# End-to-end multi-source optimization
# ---------------------------------------------------------------------------

@_CUDA
def test_two_source_optimization_decreases():
    """A two-port objective is reduced monotonically by gradient descent.

    Drives both ModeSource ports and descends the probe intensity, exercising
    the full multi-source forward/adjoint loop under an optimizer over several
    iterations (the inverse-design use case the single-source cap blocked).
    """
    model = _TwoPortModeScene(enable_second=True, shape=(2, 1, 1), init=0.0).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.3)

    losses = []
    for _ in range(6):
        optimizer.zero_grad()
        loss = _mode_loss(model, time_steps=128)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.3e} -> {losses[-1]:.3e}"
    decreases = sum(1 for i in range(1, len(losses)) if losses[i] < losses[i - 1])
    assert decreases >= 4, f"expected mostly-monotonic descent, got {losses}"
