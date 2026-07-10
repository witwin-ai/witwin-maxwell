"""Finite-difference gradient validation for the extended FDTD adjoint media.

Covers the media unlocked by the adjoint coverage extension, starting with
diagonal-anisotropic (``DiagonalTensor3`` epsilon) structures: an exact
transpose unit test of the per-axis material pullback plus end-to-end
finite-difference validation of scenes containing anisotropic media.
"""

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.material_pullback import pullback_material_input_gradients
from witwin.maxwell.fdtd.runtime.materials import average_node_to_component
from witwin.maxwell.scene import prepare_scene

_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA for FDTD"
)

_FD_DELTA = 1.0e-2
_EPS0 = 8.8541878128e-12


def _abs2(z):
    if z.is_complex():
        return (z * z.conj()).real
    return z * z


def _build_sim(model, *, time_steps):
    return mw.Simulation.fdtd(
        model,
        frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=time_steps),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
    )


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


def _backward_and_fd_grads(model, loss_fn, *, delta=_FD_DELTA):
    loss = loss_fn()
    loss.backward()
    backward_grad = model.logits.grad.detach().clone()
    model.logits.grad = None
    fd_grad = _central_difference_per_element(model, loss_fn, delta=delta)
    return backward_grad, fd_grad


# ---------------------------------------------------------------------------
# Diagonal anisotropic epsilon
# ---------------------------------------------------------------------------


class _AnisotropicGeometryScene(mw.SceneModule):
    """Trainable-size box filled with a diagonal-anisotropic permittivity.

    The three logits control the box extent along x/y/z, so each logit's
    gradient flows through a different mix of the per-axis eps components.
    """

    def __init__(self, init=0.0):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.full((3,), float(init), device="cuda"))

    def to_scene(self):
        size = 0.30 + 0.12 * torch.sigmoid(self.logits)
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6))),
            grid=mw.GridSpec.uniform(0.12),
            boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
            device="cuda",
            # Sub-cell geometry sensitivity needs subpixel occupancy averaging,
            # matching the trainable-geometry FDFD gradient tests.
            subpixel_samples=5,
        )
        scene.add_structure(
            mw.Structure(
                name="aniso_box",
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=size),
                material=mw.Material(epsilon_tensor=mw.DiagonalTensor3(2.0, 3.5, 5.0)),
            )
        )
        scene.add_source(
            mw.PointDipole(
                position=(0.0, 0.0, -0.30),
                polarization="Ez",
                width=0.04,
                source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, amplitude=50.0),
            )
        )
        scene.add_monitor(mw.PointMonitor("probe", (0.0, 0.0, 0.30), fields=("Ez",)))
        return scene


class _AnisotropicDensityScene(mw.SceneModule):
    """Trainable design density next to a static diagonal-anisotropic slab.

    Exercises the full adjoint chain (reverse steps + pullback) in a scene
    whose per-axis eps_Ex/Ey/Ez coefficient tensors genuinely differ.
    """

    def __init__(self, init=0.0):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.full((2, 2, 2), float(init), device="cuda"))

    def to_scene(self):
        density = torch.sigmoid(self.logits)
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6))),
            grid=mw.GridSpec.uniform(0.12),
            boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
            device="cuda",
        )
        scene.add_structure(
            mw.Structure(
                name="aniso_slab",
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.60, 0.60, 0.12)),
                material=mw.Material(epsilon_tensor=mw.DiagonalTensor3(2.0, 3.5, 5.0)),
            )
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
def test_anisotropic_pullback_is_exact_transpose_of_component_averaging():
    """The material pullback must be the exact transpose of the per-axis
    node->Yee-edge permittivity build for a diagonal-anisotropic scene."""
    torch.manual_seed(3)
    model = _AnisotropicGeometryScene(init=0.0).cuda()
    inputs = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)

    # Two independent to_scene() calls: the pullback consumes (frees) its own
    # compile graph, so the direct-autograd reference needs a fresh one.
    scene = model.to_scene()
    prepared = prepare_scene(scene)
    grad_eps_ex = torch.randn((prepared.Nx - 1, prepared.Ny, prepared.Nz), device="cuda")
    grad_eps_ey = torch.randn((prepared.Nx, prepared.Ny - 1, prepared.Nz), device="cuda")
    grad_eps_ez = torch.randn((prepared.Nx, prepared.Ny, prepared.Nz - 1), device="cuda")

    with torch.enable_grad():
        outputs = pullback_material_input_gradients(
            scene,
            inputs=inputs,
            grad_eps_ex=grad_eps_ex,
            grad_eps_ey=grad_eps_ey,
            grad_eps_ez=grad_eps_ez,
            eps0=_EPS0,
        )

        # Direct autograd of the equivalent objective: contract each Yee-edge
        # cotangent against the forward node->edge averaged absolute
        # permittivity of the matching axis component.
        reference_prepared = prepare_scene(model.to_scene())
        eps_components, _ = reference_prepared.compile_material_components()
        objective = (
            (grad_eps_ex * average_node_to_component(None, eps_components["x"] * _EPS0, "Ex")).sum()
            + (grad_eps_ey * average_node_to_component(None, eps_components["y"] * _EPS0, "Ey")).sum()
            + (grad_eps_ez * average_node_to_component(None, eps_components["z"] * _EPS0, "Ez")).sum()
        )
        expected = torch.autograd.grad(objective, inputs, allow_unused=True)

    for output, reference in zip(outputs, expected):
        reference = torch.zeros_like(output) if reference is None else reference
        assert torch.allclose(output, reference, rtol=1e-4, atol=1e-8), (
            output.tolist(),
            reference.tolist(),
        )


@_CUDA
def test_scene_with_diagonal_anisotropic_medium_gradient_matches_fd():
    """Per-element FD validation of design gradients in a scene containing a
    static diagonal-anisotropic slab (previously rejected by the bridge)."""
    model = _AnisotropicDensityScene(init=0.0).cuda()

    def loss_fn():
        result = _build_sim(model, time_steps=200).run()
        return _abs2(result.monitor("probe")["data"])

    backward_grad, fd_grad = _backward_and_fd_grads(model, loss_fn)

    assert (fd_grad != 0).any(), "FD gradient is identically zero; test setup is broken."
    scale = torch.abs(fd_grad).max().item()
    assert torch.allclose(backward_grad, fd_grad, rtol=3e-2, atol=scale * 5e-3), (
        f"backward={backward_grad.flatten().tolist()}, fd={fd_grad.flatten().tolist()}"
    )


@_CUDA
def test_anisotropic_geometry_gradient_is_axis_distinct():
    """Trainable geometry of a diagonal-anisotropic box must receive non-zero,
    axis-distinct gradients through the per-axis pullback.

    Note: FDTD trainable-geometry gradients carry a known (pre-existing,
    material-independent) finite-difference deviation, so this checks
    structure rather than tight FD agreement.
    """
    model = _AnisotropicGeometryScene(init=0.0).cuda()
    result = _build_sim(model, time_steps=400).run()
    loss = _abs2(result.monitor("probe")["data"])
    loss.backward()
    grad = model.logits.grad.detach()
    assert grad is not None
    assert (grad != 0).all()
    assert torch.unique(grad).numel() > 1
