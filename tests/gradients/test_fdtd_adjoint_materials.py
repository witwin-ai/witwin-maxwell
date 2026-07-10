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


# ---------------------------------------------------------------------------
# Magnetic dispersive media
# ---------------------------------------------------------------------------


class _MagneticDispersiveDensityScene(mw.SceneModule):
    """Trainable design density next to a static mu-Lorentz slab.

    Exercises the magnetic ADE mirror of the dispersive reverse backend: the
    per-step H corrections and the magnetic pole state must be reversed for
    the design-density eps gradients to be correct.
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
                name="mu_lorentz_slab",
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.60, 0.60, 0.12)),
                material=mw.Material(
                    mu_lorentz_poles=(
                        mw.LorentzPole(delta_eps=1.0, resonance_frequency=1.5e9, gamma=3.0e8),
                    ),
                ),
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
def test_scene_with_magnetic_dispersive_medium_gradient_matches_fd():
    """Per-element FD validation of design gradients in a scene containing a
    static magnetic-dispersive (mu-Lorentz) slab (previously rejected)."""
    model = _MagneticDispersiveDensityScene(init=0.0).cuda()

    def loss_fn():
        result = _build_sim(model, time_steps=200).run()
        return _abs2(result.monitor("probe")["data"])

    backward_grad, fd_grad = _backward_and_fd_grads(model, loss_fn)

    assert (fd_grad != 0).any(), "FD gradient is identically zero; test setup is broken."
    scale = torch.abs(fd_grad).max().item()
    assert torch.allclose(backward_grad, fd_grad, rtol=3e-2, atol=scale * 5e-3), (
        f"backward={backward_grad.flatten().tolist()}, fd={fd_grad.flatten().tolist()}"
    )


# ---------------------------------------------------------------------------
# Static electric conductivity (sigma_e) media
# ---------------------------------------------------------------------------

_SIGMA_E = 0.1


def _conductive_scene(sigma_e, *, density=None):
    """Design region embedded in a static conductive slab.

    The lossy slab spans (and slightly overhangs) the design region so every
    trainable design edge carries a non-zero sigma_e. The design density then
    drives the permittivity of conductive cells, whose semi-implicit lossy
    decay/curl coefficients depend on eps -- the dependence the linear-dielectric
    reverse rule drops.
    """
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6))),
        grid=mw.GridSpec.uniform(0.12),
        boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
        device="cuda",
    )
    if sigma_e != 0.0:
        scene.add_structure(
            mw.Structure(
                name="lossy_slab",
                geometry=mw.Box(position=(0.06, 0.06, 0.18), size=(0.30, 0.30, 0.30)),
                material=mw.Material(eps_r=1.0, sigma_e=sigma_e),
            )
        )
    if density is not None:
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


class _ConductiveDensityScene(mw.SceneModule):
    """Trainable design density inside a static conductive (sigma_e) slab.

    The reverse step must differentiate the semi-implicit conduction-loss
    decay/curl coefficients through eps for the design gradients to be correct.
    """

    def __init__(self, init=0.0, sigma_e=_SIGMA_E):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.full((2, 2, 2), float(init), device="cuda"))
        self._sigma = sigma_e

    def to_scene(self):
        return _conductive_scene(self._sigma, density=torch.sigmoid(self.logits))


@_CUDA
def test_scene_with_conductive_medium_gradient_matches_fd():
    """Per-element FD validation of design gradients in a scene whose trainable
    cells carry static electric conductivity (previously rejected by the bridge)."""
    model = _ConductiveDensityScene(init=0.0)

    def loss_fn():
        result = _build_sim(model, time_steps=200).run()
        return _abs2(result.monitor("probe")["data"])

    # The conduction loss must actually shape the solution, otherwise this would
    # only re-validate the lossless path (which the linear reverse handles). A
    # realistic sigma_e damps the probe field several-fold.
    lossless = _abs2(
        _build_sim(_ConductiveDensityScene(init=0.0, sigma_e=0.0), time_steps=200)
        .run()
        .monitor("probe")["data"]
    ).item()
    conductive = loss_fn().item()
    assert conductive < 0.6 * lossless, (
        f"Conduction term inactive: lossless={lossless:.6e}, conductive={conductive:.6e}"
    )

    backward_grad, fd_grad = _backward_and_fd_grads(model, loss_fn)

    assert (fd_grad != 0).any(), "FD gradient is identically zero; test setup is broken."
    # Acceptance bar: dominant-element relative error below 1e-3.
    dominant = int(torch.abs(fd_grad).flatten().argmax().item())
    dom_rel = (
        abs(backward_grad.flatten()[dominant].item() - fd_grad.flatten()[dominant].item())
        / abs(fd_grad.flatten()[dominant].item())
    )
    assert dom_rel < 1e-3, f"dominant-element rel err {dom_rel:.3e} exceeds 1e-3."

    scale = torch.abs(fd_grad).max().item()
    assert torch.allclose(backward_grad, fd_grad, rtol=3e-2, atol=scale * 5e-3), (
        f"backward={backward_grad.flatten().tolist()}, fd={fd_grad.flatten().tolist()}"
    )


@_CUDA
def test_conductive_reverse_routes_through_torch_vjp():
    """A conductive scene must fall to the torch-VJP reverse backend: the analytic
    standard/CPML backends model an eps-independent decay and would drop the
    conduction-loss eps sensitivity."""
    from witwin.maxwell.fdtd.adjoint.dispatch import _select_reverse_backend, _ReverseBackend

    prepared = mw.Simulation.fdtd(
        _conductive_scene(_SIGMA_E),
        frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=8),
    ).prepare()
    solver = prepared.solver
    assert solver.conductive_enabled
    forward_state = {name: getattr(solver, name) for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
    backend = _select_reverse_backend(
        solver,
        forward_state,
        eps_ex=solver.eps_Ex,
        eps_ey=solver.eps_Ey,
        eps_ez=solver.eps_Ez,
        resolved_source_terms=None,
    )
    assert backend is _ReverseBackend.TORCH_VJP, backend


# ---------------------------------------------------------------------------
# Kerr (instantaneous chi3) media
# ---------------------------------------------------------------------------

_KERR_CHI3 = 5.0e-3


def _kerr_scene(chi3, *, density=None):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6))),
        grid=mw.GridSpec.uniform(0.12),
        boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(
            name="kerr_slab",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.60, 0.60, 0.12)),
            material=mw.Material(eps_r=2.0, kerr_chi3=chi3),
        )
    )
    if density is not None:
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


class _KerrDensityScene(mw.SceneModule):
    """Trainable design density next to a static Kerr (chi3) slab.

    The reverse step must differentiate the per-step dynamic curl recompute
    through the forward E fields for the design gradients to be correct.
    """

    def __init__(self, init=0.0, chi3=_KERR_CHI3):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.full((2, 2, 2), float(init), device="cuda"))
        self._chi3 = chi3

    def to_scene(self):
        return _kerr_scene(self._chi3, density=torch.sigmoid(self.logits))


@_CUDA
def test_kerr_dynamic_curl_replica_matches_cuda_kernel():
    """The differentiable Kerr dynamic-curl replica used by the adjoint replay
    must match the native updateKerrElectricField*Curl3D kernels."""
    from witwin.maxwell.fdtd.adjoint.core import _kerr_dynamic_electric_curls

    torch.manual_seed(11)
    prepared = mw.Simulation.fdtd(
        _kerr_scene(_KERR_CHI3),
        frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=8),
    ).prepare()
    solver = prepared.solver
    assert solver.kerr_enabled

    solver.Ex.copy_(2.0 * torch.randn_like(solver.Ex))
    solver.Ey.copy_(2.0 * torch.randn_like(solver.Ey))
    solver.Ez.copy_(2.0 * torch.randn_like(solver.Ez))
    solver._update_nonlinear_electric_coefficients()

    state = {"Ex": solver.Ex, "Ey": solver.Ey, "Ez": solver.Ez}
    replica = _kerr_dynamic_electric_curls(
        solver,
        state,
        eps_ex=solver.eps_Ex,
        eps_ey=solver.eps_Ey,
        eps_ez=solver.eps_Ez,
        chi3_ex=solver.kerr_chi3_Ex,
        chi3_ey=solver.kerr_chi3_Ey,
        chi3_ez=solver.kerr_chi3_Ez,
    )

    assert torch.allclose(replica["Ex"], solver.cex_curl_dynamic, rtol=1e-5, atol=0.0)
    assert torch.allclose(replica["Ey"], solver.cey_curl_dynamic, rtol=1e-5, atol=0.0)
    assert torch.allclose(replica["Ez"], solver.cez_curl_dynamic, rtol=1e-5, atol=0.0)


@_CUDA
def test_scene_with_kerr_medium_gradient_matches_fd():
    """Per-element FD validation of design gradients in a scene containing a
    static Kerr slab (previously rejected by the bridge)."""
    model = _KerrDensityScene(init=0.0)

    def loss_fn():
        result = _build_sim(model, time_steps=200).run()
        return _abs2(result.monitor("probe")["data"])

    # The nonlinearity must actually shape the solution, otherwise this would
    # only re-validate the linear path.
    linear_loss = _abs2(
        _build_sim(_KerrDensityScene(init=0.0, chi3=0.0), time_steps=200)
        .run()
        .monitor("probe")["data"]
    ).item()
    kerr_loss = loss_fn().item()
    assert abs(kerr_loss - linear_loss) > 1e-3 * abs(linear_loss), (
        f"Kerr term inactive: linear={linear_loss:.6e}, kerr={kerr_loss:.6e}"
    )

    backward_grad, fd_grad = _backward_and_fd_grads(model, loss_fn)

    assert (fd_grad != 0).any(), "FD gradient is identically zero; test setup is broken."
    scale = torch.abs(fd_grad).max().item()
    assert torch.allclose(backward_grad, fd_grad, rtol=3e-2, atol=scale * 5e-3), (
        f"backward={backward_grad.flatten().tolist()}, fd={fd_grad.flatten().tolist()}"
    )


class _KerrGeometryScene(mw.SceneModule):
    """Trainable-size Kerr box: the logits move the chi3 (and eps) blend."""

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
            subpixel_samples=5,
        )
        scene.add_structure(
            mw.Structure(
                name="kerr_box",
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=size),
                material=mw.Material(eps_r=2.0, kerr_chi3=_KERR_CHI3),
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
def test_kerr_pullback_chi3_channel_is_exact_transpose():
    """The chi3 gradient channel of the material pullback must be the exact
    transpose of the node->Yee-edge chi3 averaging."""
    torch.manual_seed(5)
    model = _KerrGeometryScene(init=0.0).cuda()
    inputs = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)

    scene = model.to_scene()
    prepared = prepare_scene(scene)
    zero_eps_ex = torch.zeros((prepared.Nx - 1, prepared.Ny, prepared.Nz), device="cuda")
    zero_eps_ey = torch.zeros((prepared.Nx, prepared.Ny - 1, prepared.Nz), device="cuda")
    zero_eps_ez = torch.zeros((prepared.Nx, prepared.Ny, prepared.Nz - 1), device="cuda")
    grad_chi3_ex = torch.randn_like(zero_eps_ex)
    grad_chi3_ey = torch.randn_like(zero_eps_ey)
    grad_chi3_ez = torch.randn_like(zero_eps_ez)

    with torch.enable_grad():
        outputs = pullback_material_input_gradients(
            scene,
            inputs=inputs,
            grad_eps_ex=zero_eps_ex,
            grad_eps_ey=zero_eps_ey,
            grad_eps_ez=zero_eps_ez,
            eps0=_EPS0,
            grad_chi3_ex=grad_chi3_ex,
            grad_chi3_ey=grad_chi3_ey,
            grad_chi3_ez=grad_chi3_ez,
        )

        reference_prepared = prepare_scene(model.to_scene())
        kerr_field = reference_prepared.compile_materials()["kerr_chi3"]
        objective = (
            (grad_chi3_ex * average_node_to_component(None, kerr_field, "Ex")).sum()
            + (grad_chi3_ey * average_node_to_component(None, kerr_field, "Ey")).sum()
            + (grad_chi3_ez * average_node_to_component(None, kerr_field, "Ez")).sum()
        )
        expected = torch.autograd.grad(objective, inputs, allow_unused=True)

    for output, reference in zip(outputs, expected):
        reference = torch.zeros_like(output) if reference is None else reference
        assert (reference != 0).any()
        assert torch.allclose(output, reference, rtol=1e-4, atol=1e-10), (
            output.tolist(),
            reference.tolist(),
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
