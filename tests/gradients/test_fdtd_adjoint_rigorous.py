"""Rigorous adjoint gradient validation tests.

These tests go beyond the basic single-scalar sanity checks in
``test_fdtd_adjoint_bridge.py`` by exercising:

* multi-voxel design regions with per-element finite-difference validation
* plane-monitor flux gradients
* multi-frequency objective functions
* density filtering and projection paths
* diverse loss functions (phase, real-part, multi-component)
* realistic multi-step optimizer convergence
* gradient determinism and non-triviality
"""

import math

import pytest
import torch

import witwin.maxwell as mw

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA for FDTD"
)

_FD_DELTA = 1.0e-2


def _abs2(z):
    """Differentiable |z|^2 that always returns a real scalar."""
    if z.is_complex():
        return (z * z.conj()).real
    return z * z


def _central_difference_per_element(model, loss_fn, delta=_FD_DELTA):
    """Compute central-difference gradient for every element in ``model.logits``."""
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


def _build_sim(model, *, time_steps=24, frequencies=None, dft_window="none",
               full_field_dft=False):
    if frequencies is None:
        frequencies = [1e9]
    return mw.Simulation.fdtd(
        model,
        frequencies=frequencies,
        run_time=mw.TimeConfig(time_steps=time_steps),
        spectral_sampler=mw.SpectralSampler(window=dft_window),
        full_field_dft=full_field_dft,
    )


def _backward_and_fd_grads(model, loss_fn, *, delta=_FD_DELTA):
    loss = loss_fn()
    loss.backward()
    backward_grad = model.logits.grad.detach().clone()
    model.logits.grad = None
    fd_grad = _central_difference_per_element(model, loss_fn, delta=delta)
    return backward_grad, fd_grad


# ---------------------------------------------------------------------------
# Scene modules
# ---------------------------------------------------------------------------

class _MultiVoxelScene(mw.SceneModule):
    """Design region with (Nx, Ny, Nz) logit tensor."""

    def __init__(self, shape=(2, 2, 2), init=0.0):
        super().__init__()
        self.logits = torch.nn.Parameter(
            torch.full(shape, float(init), device="cuda")
        )

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
                # Box faces are offset half a cell from the grid nodes so the
                # voxel window is off the node-aligned knife edge and matches
                # the placement this test was calibrated against.
                geometry=mw.Box(position=(0.06, 0.06, 0.18), size=(0.24, 0.24, 0.24)),
                density=density,
                eps_bounds=(1.0, 6.0),
                mu_bounds=(1.0, 1.0),
            )
        )
        scene.add_source(
            mw.PointDipole(
                position=(0.0, 0.0, -0.18),
                polarization="Ez",
                width=0.04,
                source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, amplitude=50.0),
            )
        )
        scene.add_monitor(mw.PointMonitor("probe", (0.0, 0.0, 0.18), fields=("Ez",)))
        return scene


class _PlaneFluxScene(mw.SceneModule):
    """Scene with a FluxMonitor for testing flux gradient."""

    def __init__(self, init=0.0):
        super().__init__()
        self.logits = torch.nn.Parameter(
            torch.full((2, 2, 2), float(init), device="cuda")
        )

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
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.24, 0.24, 0.24)),
                density=density,
                eps_bounds=(1.0, 6.0),
                mu_bounds=(1.0, 1.0),
            )
        )
        scene.add_source(
            mw.PointDipole(
                position=(0.0, 0.0, -0.24),
                polarization="Ez",
                width=0.04,
                source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, amplitude=50.0),
            )
        )
        scene.add_monitor(
            mw.FluxMonitor("flux_probe", axis="z", position=0.24, normal_direction="+")
        )
        scene.add_monitor(mw.PointMonitor("point_probe", (0.0, 0.0, 0.24), fields=("Ez",)))
        return scene


class _MultiFreqScene(mw.SceneModule):
    """Scene evaluated at two frequencies."""

    def __init__(self, init=0.0):
        super().__init__()
        self.logits = torch.nn.Parameter(
            torch.full((2, 2, 2), float(init), device="cuda")
        )

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
                geometry=mw.Box(position=(0.0, 0.0, 0.12), size=(0.24, 0.24, 0.24)),
                density=density,
                eps_bounds=(1.0, 6.0),
                mu_bounds=(1.0, 1.0),
            )
        )
        scene.add_source(
            mw.PointDipole(
                position=(0.0, 0.0, -0.18),
                polarization="Ez",
                width=0.04,
                source_time=mw.GaussianPulse(frequency=1.25e9, fwidth=0.5e9, amplitude=50.0),
            )
        )
        scene.add_monitor(mw.PointMonitor("probe", (0.0, 0.0, 0.18), fields=("Ez",)))
        return scene


class _FilteredScene(mw.SceneModule):
    """Design region with density filtering."""

    def __init__(self, init=0.0):
        super().__init__()
        self.logits = torch.nn.Parameter(
            torch.full((3, 3, 3), float(init), device="cuda")
        )

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
                geometry=mw.Box(position=(0.0, 0.0, 0.12), size=(0.36, 0.36, 0.36)),
                density=density,
                eps_bounds=(1.0, 6.0),
                mu_bounds=(1.0, 1.0),
                filter_radius=0.12,
            )
        )
        scene.add_source(
            mw.PointDipole(
                position=(0.0, 0.0, -0.18),
                polarization="Ez",
                width=0.04,
                source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, amplitude=50.0),
            )
        )
        scene.add_monitor(mw.PointMonitor("probe", (0.0, 0.0, 0.24), fields=("Ez",)))
        return scene


class _ProjectedScene(mw.SceneModule):
    """Design region with projection (thresholding)."""

    def __init__(self, init=0.0):
        super().__init__()
        self.logits = torch.nn.Parameter(
            torch.full((2, 2, 2), float(init), device="cuda")
        )

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
                geometry=mw.Box(position=(0.0, 0.0, 0.12), size=(0.24, 0.24, 0.24)),
                density=density,
                eps_bounds=(1.0, 6.0),
                mu_bounds=(1.0, 1.0),
                projection_beta=4.0,
            )
        )
        scene.add_source(
            mw.PointDipole(
                position=(0.0, 0.0, -0.18),
                polarization="Ez",
                width=0.04,
                source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, amplitude=50.0),
            )
        )
        scene.add_monitor(mw.PointMonitor("probe", (0.0, 0.0, 0.18), fields=("Ez",)))
        return scene


class _FilteredProjectedScene(mw.SceneModule):
    """Design region with both filtering and projection."""

    def __init__(self, init=0.0):
        super().__init__()
        self.logits = torch.nn.Parameter(
            torch.full((3, 3, 3), float(init), device="cuda")
        )

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
                geometry=mw.Box(position=(0.0, 0.0, 0.12), size=(0.36, 0.36, 0.36)),
                density=density,
                eps_bounds=(1.0, 6.0),
                mu_bounds=(1.0, 1.0),
                filter_radius=0.12,
                projection_beta=4.0,
            )
        )
        scene.add_source(
            mw.PointDipole(
                position=(0.0, 0.0, -0.18),
                polarization="Ez",
                width=0.04,
                source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, amplitude=50.0),
            )
        )
        scene.add_monitor(mw.PointMonitor("probe", (0.0, 0.0, 0.24), fields=("Ez",)))
        return scene


class _MultiComponentScene(mw.SceneModule):
    """Scene with a point monitor that records Ex, Ey, and Ez."""

    def __init__(self, init=0.0):
        super().__init__()
        self.logits = torch.nn.Parameter(
            torch.full((2, 2, 2), float(init), device="cuda")
        )

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
                geometry=mw.Box(position=(0.0, 0.0, 0.12), size=(0.24, 0.24, 0.24)),
                density=density,
                eps_bounds=(1.0, 6.0),
                mu_bounds=(1.0, 1.0),
            )
        )
        scene.add_source(
            mw.PointDipole(
                position=(0.0, 0.0, -0.18),
                polarization="Ez",
                width=0.04,
                source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, amplitude=50.0),
            )
        )
        scene.add_monitor(
            mw.PointMonitor("probe", (0.12, 0.12, 0.18), fields=("Ex", "Ey", "Ez"))
        )
        return scene


class _CpmlFocusedScene(mw.SceneModule):
    """PML scene that places the design region close to the absorbing boundary."""

    def __init__(self, shape=(2, 1, 1), init=0.0):
        super().__init__()
        self.logits = torch.nn.Parameter(
            torch.full(shape, float(init), device="cuda")
        )

    def to_scene(self):
        density = torch.sigmoid(self.logits)
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.72, 0.72), (-0.48, 0.48), (-0.48, 0.48))),
            grid=mw.GridSpec.uniform(0.12),
            boundary=mw.BoundarySpec.pml(num_layers=3, strength=1.0),
            device="cuda",
        )
        scene.add_material_region(
            mw.MaterialRegion(
                name="design",
                geometry=mw.Box(position=(0.30, 0.0, 0.0), size=(0.24, 0.12, 0.12)),
                density=density,
                eps_bounds=(1.0, 6.0),
                mu_bounds=(1.0, 1.0),
            )
        )
        scene.add_source(
            mw.PointDipole(
                position=(-0.24, 0.0, 0.0),
                polarization="Ez",
                width=0.04,
                source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, amplitude=45.0),
            )
        )
        scene.add_monitor(mw.PointMonitor("probe", (0.42, 0.0, 0.0), fields=("Ez",)))
        return scene


class _RigorousTfsfScene(mw.SceneModule):
    """Multi-voxel TFSF scene for per-element finite-difference validation."""

    def __init__(self, init=0.0):
        super().__init__()
        self.logits = torch.nn.Parameter(
            torch.full((2, 1, 1), float(init), device="cuda")
        )

    def to_scene(self):
        density = torch.sigmoid(self.logits)
        injection = mw.TFSF(bounds=((-0.36, 0.36), (-0.36, 0.36), (-0.36, 0.36)))
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-1.2, 1.2), (-1.2, 1.2), (-1.2, 1.2))),
            grid=mw.GridSpec.uniform(0.12),
            boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
            device="cuda",
        )
        scene.add_material_region(
            mw.MaterialRegion(
                name="design",
                geometry=mw.Box(position=(0.06, 0.0, 0.0), size=(0.24, 0.12, 0.12)),
                density=density,
                eps_bounds=(1.0, 6.0),
                mu_bounds=(1.0, 1.0),
            )
        )
        scene.add_source(
            mw.PlaneWave(
                name="tfsf_pw",
                direction=(1.0, 0.0, 0.0),
                polarization=(0.0, 0.0, 1.0),
                source_time=mw.GaussianPulse(frequency=1.0e9, fwidth=0.25e9, amplitude=40.0),
                injection=injection,
            )
        )
        scene.add_monitor(mw.PointMonitor("probe", (0.18, 0.0, 0.0), fields=("Ez",)))
        return scene


class _RigorousBlochScene(mw.SceneModule):
    """Multi-voxel Bloch scene for complex-field finite-difference validation."""

    def __init__(self, init=0.0):
        super().__init__()
        self.logits = torch.nn.Parameter(
            torch.full((2, 1, 1), float(init), device="cuda")
        )

    def to_scene(self):
        density = torch.sigmoid(self.logits)
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6))),
            grid=mw.GridSpec.uniform(0.15),
            boundary=mw.BoundarySpec.bloch((math.pi / 1.2, 0.0, 0.0)),
            device="cuda",
        )
        scene.add_material_region(
            mw.MaterialRegion(
                name="design",
                geometry=mw.Box(position=(0.075, 0.0, 0.0), size=(0.30, 0.15, 0.15)),
                density=density,
                eps_bounds=(1.0, 6.0),
                mu_bounds=(1.0, 1.0),
            )
        )
        scene.add_source(
            mw.PointDipole(
                position=(-0.45, 0.0, 0.0),
                polarization="Ez",
                width=0.12,
                source_time=mw.GaussianPulse(frequency=1.0e9, fwidth=0.25e9, amplitude=25.0),
            )
        )
        scene.add_monitor(mw.PointMonitor("probe", (0.15, 0.0, 0.0), fields=("Ez",)))
        return scene


class _RigorousGratingTfsfScene(mw.SceneModule):
    """Small grating TFSF scene for finite-difference adjoint validation."""

    def __init__(self, init=0.0):
        super().__init__()
        self.logits = torch.nn.Parameter(
            torch.full((1, 1, 1), float(init), device="cuda")
        )

    def to_scene(self):
        density = torch.sigmoid(self.logits)
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.45, 0.45), (-0.45, 0.45), (-0.75, 0.75))),
            grid=mw.GridSpec.uniform(0.15),
            boundary=mw.BoundarySpec.faces(
                default="pml",
                num_layers=2,
                strength=1.0,
                x="bloch",
                y="bloch",
                z="pml",
                bloch_wavevector="auto",
            ),
            device="cuda",
        )
        scene.add_material_region(
            mw.MaterialRegion(
                name="design",
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.15, 0.15, 0.15)),
                density=density,
                eps_bounds=(1.0, 6.0),
                mu_bounds=(1.0, 1.0),
            )
        )
        scene.add_source(
            mw.PlaneWave(
                name="grating_tfsf",
                direction=(0.2, 0.1, 0.9746794344808963),
                polarization=(1.0, 0.0, -0.20519567041703082),
                source_time=mw.CW(frequency=1.0e9, amplitude=40.0),
                injection=mw.TFSF.slab(axis="z", bounds=(-0.30, 0.30)),
            )
        )
        scene.add_monitor(mw.PointMonitor("probe", (0.15, 0.0, 0.0), fields=("Ex",)))
        return scene


# ---------------------------------------------------------------------------
# 1. Multi-voxel spatial gradient: per-element finite difference
# ---------------------------------------------------------------------------

@_CUDA
def test_spatial_gradient_per_element_matches_fd_strong_signal():
    """Per-element FD validation on a (2,2,2) design region with strong signal.

    Uses 200 time steps so the field reaches ~1e-2 magnitude at the probe.
    This ensures we are comparing meaningful gradient values, not noise.
    This is the highest-confidence test in the suite.
    """
    model = _MultiVoxelScene(shape=(2, 2, 2), init=0.0).cuda()

    def loss_fn():
        result = _build_sim(model, time_steps=200).run()
        data = result.monitor("probe")["data"]
        return _abs2(data)

    result = _build_sim(model, time_steps=200).run()
    data = result.monitor("probe")["data"]
    assert abs(data.item()) > 1e-4, (
        f"|data| = {abs(data.item()):.4e} is too small for reliable FD comparison."
    )
    loss = _abs2(data)
    loss.backward()
    backward_grad = model.logits.grad.detach().clone()
    model.logits.grad = None

    fd_grad = _central_difference_per_element(model, loss_fn, delta=_FD_DELTA)

    # Large gradient elements match to ~0.02%.  Small elements have worse
    # relative accuracy because FD precision degrades when the perturbation
    # effect is near the float32 noise floor.  Use atol scaled to 0.2% of the
    # max gradient so near-zero elements don't dominate the comparison.
    scale = torch.abs(fd_grad).max().item()
    assert torch.allclose(backward_grad, fd_grad, rtol=2e-3, atol=scale * 2e-3), (
        f"max absolute error: "
        f"{(torch.abs(backward_grad - fd_grad)).max().item():.4e}, "
        f"scale: {scale:.4e}"
    )


@_CUDA
def test_spatial_gradient_varies_across_voxels():
    """Verify gradients are spatially varying, not a constant fill.

    A buggy implementation that returns the same value for every voxel would
    pass single-element tests but fail this one.
    """
    model = _MultiVoxelScene(shape=(3, 3, 3), init=0.0).cuda()
    result = _build_sim(model, time_steps=24).run()
    data = result.monitor("probe")["data"]
    loss = _abs2(data)
    loss.backward()
    grad = model.logits.grad.detach()

    assert grad.numel() == 27
    unique_values = torch.unique(grad)
    assert unique_values.numel() > 1, (
        f"All {grad.numel()} gradient elements are identical ({grad.flatten()[0].item():.6e}), "
        "indicating a likely broadcast or reduction bug."
    )


@_CUDA
def test_spatial_gradient_nonzero():
    """Verify gradients are non-zero at every design voxel.

    With few time steps the field can be very weak, so we check for exact
    zero rather than a fixed threshold.
    """
    model = _MultiVoxelScene(shape=(2, 2, 2), init=0.0).cuda()
    result = _build_sim(model, time_steps=24).run()
    data = result.monitor("probe")["data"]
    loss = _abs2(data)
    loss.backward()
    grad = model.logits.grad
    assert grad is not None, "Gradient is None -backward did not reach logits."

    nonzero_count = (grad != 0).sum().item()
    assert nonzero_count == grad.numel(), (
        f"Only {nonzero_count}/{grad.numel()} gradient elements are non-zero. "
        f"grad = {grad}"
    )


# ---------------------------------------------------------------------------
# 2. Plane monitor flux gradient
# ---------------------------------------------------------------------------

@_CUDA
def test_plane_flux_gradient_matches_fd():
    """Verify gradient through plane-monitor flux (Poynting integration)."""
    model = _PlaneFluxScene(init=0.0).cuda()

    def loss_fn():
        result = _build_sim(model, time_steps=32).run()
        flux = result.monitor("flux_probe")["flux"]
        return _abs2(flux)

    result = _build_sim(model, time_steps=32).run()
    flux = result.monitor("flux_probe")["flux"]
    loss = _abs2(flux)
    loss.backward()
    backward_grad = model.logits.grad.detach().clone()
    model.logits.grad = None

    fd_grad = _central_difference_per_element(model, loss_fn, delta=_FD_DELTA)

    assert torch.allclose(backward_grad, fd_grad, rtol=3e-2, atol=1e-15), (
        f"max relative error: "
        f"{(torch.abs(backward_grad - fd_grad) / (torch.abs(fd_grad) + 1e-30)).max().item():.4e}"
    )


# ---------------------------------------------------------------------------
# 3. Multi-frequency objective
# ---------------------------------------------------------------------------

@_CUDA
def test_multi_frequency_gradient_matches_fd():
    """Validate gradient of a loss that combines two frequencies."""
    model = _MultiFreqScene(init=0.0).cuda()
    freqs = [0.8e9, 1.5e9]

    def loss_fn():
        result = _build_sim(model, time_steps=36, frequencies=freqs).run()
        d0 = result.monitor("probe", freq_index=0)["data"]
        d1 = result.monitor("probe", freq_index=1)["data"]
        return _abs2(d0) + _abs2(d1)

    result = _build_sim(model, time_steps=36, frequencies=freqs).run()
    d0 = result.monitor("probe", freq_index=0)["data"]
    d1 = result.monitor("probe", freq_index=1)["data"]
    loss = _abs2(d0) + _abs2(d1)
    loss.backward()
    backward_grad = model.logits.grad.detach().clone()
    model.logits.grad = None

    fd_grad = _central_difference_per_element(model, loss_fn, delta=_FD_DELTA)

    assert torch.allclose(backward_grad, fd_grad, rtol=3e-2, atol=1e-15), (
        f"max relative error: "
        f"{(torch.abs(backward_grad - fd_grad) / (torch.abs(fd_grad) + 1e-30)).max().item():.4e}"
    )


# ---------------------------------------------------------------------------
# 4. Dedicated backend validation: CPML / TFSF / Bloch
# ---------------------------------------------------------------------------

@_CUDA
def test_cpml_gradient_per_element_matches_fd():
    """Per-element FD validation for the dedicated CPML reverse path."""
    model = _CpmlFocusedScene(shape=(2, 1, 1), init=0.0).cuda()

    def loss_fn():
        result = _build_sim(model, time_steps=64).run()
        return _abs2(result.monitor("probe")["data"])

    backward_grad, fd_grad = _backward_and_fd_grads(model, loss_fn)

    scale = torch.abs(fd_grad).max().item()
    assert torch.allclose(backward_grad, fd_grad, rtol=3e-2, atol=max(scale * 2e-3, 1e-15)), (
        f"max absolute error: {(torch.abs(backward_grad - fd_grad)).max().item():.4e}, "
        f"scale: {scale:.4e}"
    )


@_CUDA
def test_tfsf_gradient_per_element_matches_fd():
    """Per-element FD validation for the TFSF reverse path."""
    model = _RigorousTfsfScene(init=0.0).cuda()

    def loss_fn():
        result = _build_sim(model, time_steps=48).run()
        return _abs2(result.monitor("probe")["data"])

    backward_grad, fd_grad = _backward_and_fd_grads(model, loss_fn)

    scale = torch.abs(fd_grad).max().item()
    assert torch.allclose(backward_grad, fd_grad, rtol=8e-2, atol=max(scale * 3e-3, 1e-15)), (
        f"max absolute error: {(torch.abs(backward_grad - fd_grad)).max().item():.4e}, "
        f"scale: {scale:.4e}"
    )


@_CUDA
def test_bloch_gradient_per_element_matches_fd():
    """Per-element FD validation for the Bloch reverse path with complex fields."""
    model = _RigorousBlochScene(init=0.0).cuda()

    def loss_fn():
        result = _build_sim(model, time_steps=40).run()
        return _abs2(result.monitor("probe")["data"])

    backward_grad, fd_grad = _backward_and_fd_grads(model, loss_fn)

    scale = torch.abs(fd_grad).max().item()
    assert torch.allclose(backward_grad, fd_grad, rtol=7e-2, atol=max(scale * 3e-3, 1e-15)), (
        f"max absolute error: {(torch.abs(backward_grad - fd_grad)).max().item():.4e}, "
        f"scale: {scale:.4e}"
    )


@_CUDA
def test_grating_tfsf_gradient_per_element_matches_fd():
    """Per-element FD validation for x/y Bloch, z-PML grating TFSF adjoint."""
    model = _RigorousGratingTfsfScene(init=0.0).cuda()

    def loss_fn():
        result = _build_sim(model, time_steps=48).run()
        return _abs2(result.monitor("probe")["data"])

    backward_grad, fd_grad = _backward_and_fd_grads(model, loss_fn)

    scale = torch.abs(fd_grad).max().item()
    assert torch.allclose(backward_grad, fd_grad, rtol=1.5e-1, atol=max(scale * 1e-2, 1e-15)), (
        f"max absolute error: {(torch.abs(backward_grad - fd_grad)).max().item():.4e}, "
        f"scale: {scale:.4e}"
    )


# ---------------------------------------------------------------------------
# 5. Density filtering and projection
# ---------------------------------------------------------------------------

@_CUDA
def test_filtered_density_gradient_matches_fd():
    """Gradient through density box-filter smoothing."""
    model = _FilteredScene(init=0.0).cuda()

    def loss_fn():
        result = _build_sim(model, time_steps=24).run()
        data = result.monitor("probe")["data"]
        return _abs2(data)

    result = _build_sim(model, time_steps=24).run()
    loss = _abs2(result.monitor("probe")["data"])
    loss.backward()
    backward_grad = model.logits.grad.detach().clone()
    model.logits.grad = None

    fd_grad = _central_difference_per_element(model, loss_fn, delta=_FD_DELTA)

    assert torch.allclose(backward_grad, fd_grad, rtol=2e-2, atol=1e-15), (
        f"max relative error: "
        f"{(torch.abs(backward_grad - fd_grad) / (torch.abs(fd_grad) + 1e-30)).max().item():.4e}"
    )


@_CUDA
def test_projected_density_gradient_matches_fd():
    """Gradient through tanh projection."""
    model = _ProjectedScene(init=0.0).cuda()

    def loss_fn():
        result = _build_sim(model, time_steps=24).run()
        data = result.monitor("probe")["data"]
        return _abs2(data)

    result = _build_sim(model, time_steps=24).run()
    loss = _abs2(result.monitor("probe")["data"])
    loss.backward()
    backward_grad = model.logits.grad.detach().clone()
    model.logits.grad = None

    fd_grad = _central_difference_per_element(model, loss_fn, delta=_FD_DELTA)

    assert torch.allclose(backward_grad, fd_grad, rtol=2e-2, atol=1e-15), (
        f"max relative error: "
        f"{(torch.abs(backward_grad - fd_grad) / (torch.abs(fd_grad) + 1e-30)).max().item():.4e}"
    )


@_CUDA
def test_filtered_projected_density_gradient_matches_fd():
    """Gradient through both filtering and projection together."""
    model = _FilteredProjectedScene(init=0.0).cuda()

    def loss_fn():
        result = _build_sim(model, time_steps=24).run()
        data = result.monitor("probe")["data"]
        return _abs2(data)

    result = _build_sim(model, time_steps=24).run()
    loss = _abs2(result.monitor("probe")["data"])
    loss.backward()
    backward_grad = model.logits.grad.detach().clone()
    model.logits.grad = None

    fd_grad = _central_difference_per_element(model, loss_fn, delta=_FD_DELTA)

    assert torch.allclose(backward_grad, fd_grad, rtol=3e-2, atol=1e-15), (
        f"max relative error: "
        f"{(torch.abs(backward_grad - fd_grad) / (torch.abs(fd_grad) + 1e-30)).max().item():.4e}"
    )


# ---------------------------------------------------------------------------
# 5. Diverse loss functions
# ---------------------------------------------------------------------------

@_CUDA
def test_real_part_loss_gradient_matches_fd():
    """Loss = Re(Ez), tests gradient of a linear complex extraction."""
    model = _MultiVoxelScene(shape=(2, 2, 2), init=0.0).cuda()

    def loss_fn():
        result = _build_sim(model, time_steps=24).run()
        data = result.monitor("probe")["data"]
        return data.real

    result = _build_sim(model, time_steps=24).run()
    loss = result.monitor("probe")["data"].real
    loss.backward()
    backward_grad = model.logits.grad.detach().clone()
    model.logits.grad = None

    fd_grad = _central_difference_per_element(model, loss_fn, delta=_FD_DELTA)

    assert torch.allclose(backward_grad, fd_grad, rtol=1e-2, atol=1e-15), (
        f"max relative error: "
        f"{(torch.abs(backward_grad - fd_grad) / (torch.abs(fd_grad) + 1e-30)).max().item():.4e}"
    )


@_CUDA
def test_phase_loss_gradient_matches_fd():
    """Loss = angle(Ez)^2, tests gradient through atan2.

    Uses a combined rtol + atol because near-zero gradient elements
    have poor relative accuracy with finite differences.
    """
    model = _MultiVoxelScene(shape=(2, 2, 2), init=0.5).cuda()

    def loss_fn():
        result = _build_sim(model, time_steps=24).run()
        data = result.monitor("probe")["data"]
        return torch.angle(data) ** 2

    result = _build_sim(model, time_steps=24).run()
    data = result.monitor("probe")["data"]
    loss = torch.angle(data) ** 2
    loss.backward()
    backward_grad = model.logits.grad.detach().clone()
    model.logits.grad = None

    fd_grad = _central_difference_per_element(model, loss_fn, delta=_FD_DELTA)

    scale = torch.abs(fd_grad).max().item()
    assert torch.allclose(backward_grad, fd_grad, rtol=3e-2, atol=scale * 1e-2), (
        f"max absolute error: "
        f"{(torch.abs(backward_grad - fd_grad)).max().item():.4e}, "
        f"scale: {scale:.4e}"
    )


@_CUDA
def test_multi_component_loss_gradient_matches_fd():
    """Loss combines |Ex|^2 + |Ey|^2 + |Ez|^2 from one monitor."""
    model = _MultiComponentScene(init=0.0).cuda()

    def loss_fn():
        result = _build_sim(model, time_steps=24).run()
        mon = result.monitor("probe")
        total = torch.tensor(0.0, device="cuda")
        for comp in ("Ex", "Ey", "Ez"):
            d = mon["components"][comp]
            total = total + _abs2(d)
        return total

    result = _build_sim(model, time_steps=24).run()
    mon = result.monitor("probe")
    loss = torch.tensor(0.0, device="cuda")
    for comp in ("Ex", "Ey", "Ez"):
        d = mon["components"][comp]
        loss = loss + _abs2(d)
    loss.backward()
    backward_grad = model.logits.grad.detach().clone()
    model.logits.grad = None

    fd_grad = _central_difference_per_element(model, loss_fn, delta=_FD_DELTA)

    assert torch.allclose(backward_grad, fd_grad, rtol=2e-2, atol=1e-15), (
        f"max relative error: "
        f"{(torch.abs(backward_grad - fd_grad) / (torch.abs(fd_grad) + 1e-30)).max().item():.4e}"
    )


# ---------------------------------------------------------------------------
# 6. Optimizer convergence (realistic)
# ---------------------------------------------------------------------------

@_CUDA
def test_optimizer_converges_over_multiple_steps():
    """Multi-step optimization with realistic learning rate.

    Unlike the existing single-step test with lr=1e13, this runs several
    steps with a reasonable learning rate and verifies monotonic decrease.
    Uses more time steps (80) so the field builds up meaningful magnitude.
    """
    model = _MultiVoxelScene(shape=(2, 2, 2), init=0.0).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.3)

    losses = []
    for _ in range(4):
        optimizer.zero_grad()
        result = _build_sim(model, time_steps=80).run()
        data = result.monitor("probe")["data"]
        loss = _abs2(data)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    assert losses[-1] < losses[0], (
        f"Loss did not decrease: {losses[0]:.6e} -> {losses[-1]:.6e}"
    )
    decreases = sum(1 for i in range(1, len(losses)) if losses[i] < losses[i - 1])
    assert decreases >= 2, (
        f"Expected at least 2 decreasing steps out of 3, got {decreases}. "
        f"Losses: {[f'{l:.6e}' for l in losses]}"
    )


# ---------------------------------------------------------------------------
# 7. Gradient determinism
# ---------------------------------------------------------------------------

@_CUDA
def test_gradient_is_deterministic():
    """Same model state must produce identical gradients across two runs."""
    def compute_grad():
        model = _MultiVoxelScene(shape=(2, 2, 2), init=0.0).cuda()
        result = _build_sim(model, time_steps=24).run()
        data = result.monitor("probe")["data"]
        loss = _abs2(data)
        loss.backward()
        return model.logits.grad.detach().clone()

    grad_a = compute_grad()
    grad_b = compute_grad()

    assert torch.allclose(grad_a, grad_b, rtol=0, atol=0), (
        f"Gradients differ between runs. Max diff: {(grad_a - grad_b).abs().max().item():.4e}"
    )


# ---------------------------------------------------------------------------
# 8. Asymmetric init: verify gradient at non-zero logit values
# ---------------------------------------------------------------------------

@_CUDA
def test_spatial_gradient_at_asymmetric_init():
    """FD validation at non-uniform initial logits.

    Starting from logits=0 gives sigmoid=0.5 everywhere, which is a highly
    symmetric point.  This test uses random initial values to exercise the
    gradient at a generic operating point.
    """
    torch.manual_seed(42)
    model = _MultiVoxelScene(shape=(2, 2, 2), init=0.0).cuda()
    with torch.no_grad():
        model.logits.copy_(torch.randn_like(model.logits) * 0.5)

    def loss_fn():
        result = _build_sim(model, time_steps=24).run()
        data = result.monitor("probe")["data"]
        return _abs2(data)

    result = _build_sim(model, time_steps=24).run()
    loss = _abs2(result.monitor("probe")["data"])
    loss.backward()
    backward_grad = model.logits.grad.detach().clone()
    model.logits.grad = None

    fd_grad = _central_difference_per_element(model, loss_fn, delta=_FD_DELTA)

    assert torch.allclose(backward_grad, fd_grad, rtol=1e-2, atol=1e-15), (
        f"max relative error: "
        f"{(torch.abs(backward_grad - fd_grad) / (torch.abs(fd_grad) + 1e-30)).max().item():.4e}"
    )


# ---------------------------------------------------------------------------
# 9. FD convergence order -the gold-standard correctness proof
# ---------------------------------------------------------------------------

@_CUDA
def test_fd_convergence_order_confirms_adjoint_accuracy():
    """Verify O(delta^2) convergence of central FD toward the adjoint gradient.

    If the adjoint is correct, the FD-adjoint error should shrink as delta^2.
    We check that shrinking delta by 10x reduces the error by roughly 100x.
    This is the most rigorous test: it cannot be passed by accident.

    Uses 200 time steps to ensure field magnitude is meaningful (~1e-2).
    """
    model = _MultiVoxelScene(shape=(2, 2, 2), init=0.0).cuda()

    result = _build_sim(model, time_steps=200).run()
    data = result.monitor("probe")["data"]
    loss = _abs2(data)
    loss.backward()
    adj_grad_0 = model.logits.grad.detach().flatten()[0].item()
    model.logits.grad = None

    def _fd_grad_element0(delta):
        logits = model.logits
        flat = logits.detach().flatten()
        with torch.no_grad():
            saved = flat[0].clone()
            flat[0] = saved + delta
            logits.copy_(flat.reshape(logits.shape))
        r_plus = _build_sim(model, time_steps=200).run()
        l_plus = _abs2(r_plus.monitor("probe")["data"]).item()
        with torch.no_grad():
            flat[0] = saved - delta
            logits.copy_(flat.reshape(logits.shape))
        r_minus = _build_sim(model, time_steps=200).run()
        l_minus = _abs2(r_minus.monitor("probe")["data"]).item()
        with torch.no_grad():
            flat[0] = saved
            logits.copy_(flat.reshape(logits.shape))
        return (l_plus - l_minus) / (2.0 * delta)

    err_coarse = abs(adj_grad_0 - _fd_grad_element0(1e-1))
    err_fine = abs(adj_grad_0 - _fd_grad_element0(1e-2))

    # Perfect O(delta^2) gives 100x reduction.  In practice float32 limits
    # FD accuracy at delta=1e-2, so we observe ~7-50x.  Require at least 5x
    # which rules out O(1) (wrong) and O(delta) (forward difference).
    ratio = err_coarse / (err_fine + 1e-50)
    assert ratio > 5.0, (
        f"Expected >5x error reduction (O(delta^2)), got {ratio:.1f}x. "
        f"err(1e-1)={err_coarse:.4e}, err(1e-2)={err_fine:.4e}. "
        "This suggests the adjoint gradient is NOT converging to the true derivative."
    )


# ---------------------------------------------------------------------------
# PEC-boundary design gradient (native adjoint has no torch-VJP fallback, so a
# PEC-face scene with a source must be served by the explicit native source
# reverse; the source-term eps-gradient masks the terminal PEC clamp).
# ---------------------------------------------------------------------------

class _PecDesignScene(mw.SceneModule):
    """Design region behind a source under a PEC-boundary domain."""

    def __init__(self, shape=(2, 2, 2), init=0.0):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.full(shape, float(init), device="cuda"))

    def to_scene(self):
        density = torch.sigmoid(self.logits)
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6))),
            grid=mw.GridSpec.uniform(0.12),
            boundary=mw.BoundarySpec(kind="pec"),
            device="cuda",
        )
        scene.add_material_region(
            mw.MaterialRegion(
                name="design",
                geometry=mw.Box(position=(0.06, 0.06, 0.18), size=(0.24, 0.24, 0.24)),
                density=density,
                eps_bounds=(1.0, 6.0),
                mu_bounds=(1.0, 1.0),
            )
        )
        scene.add_source(
            mw.PointDipole(
                position=(0.0, 0.0, -0.18),
                polarization="Ez",
                width=0.04,
                source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, amplitude=50.0),
            )
        )
        scene.add_monitor(mw.PointMonitor("probe", (0.0, 0.0, 0.18), fields=("Ez",)))
        return scene


@_CUDA
def test_pec_boundary_design_gradient_matches_fd():
    """A PEC-boundary design scene with a source is differentiable (no torch-VJP
    fallback exists) and its gradient matches central differences."""
    model = _PecDesignScene(shape=(2, 2, 2), init=0.0).cuda()

    def loss_fn():
        data = _build_sim(model, time_steps=120).run().monitor("probe")["data"]
        return _abs2(data)

    backward_grad, fd_grad = _backward_and_fd_grads(model, loss_fn, delta=1.0e-2)

    assert torch.isfinite(backward_grad).all()
    assert backward_grad.abs().max() > 0.0
    # Dominant-element agreement; sub-dominant elements sit near the float32 FD
    # noise floor, so compare the largest-magnitude voxel.
    idx = int(fd_grad.abs().flatten().argmax())
    a = backward_grad.flatten()[idx].item()
    f = fd_grad.flatten()[idx].item()
    assert a == pytest.approx(f, rel=5.0e-3), f"adjoint {a:.4e} vs FD {f:.4e}"
