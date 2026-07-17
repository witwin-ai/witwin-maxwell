"""Finite-difference gradient validation for the extended FDTD adjoint media.

Covers the media unlocked by the adjoint coverage extension, starting with
diagonal-anisotropic (``DiagonalTensor3`` epsilon) structures: an exact
transpose unit test of the per-axis material pullback plus end-to-end
finite-difference validation of scenes containing anisotropic media.
"""

import math

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
    """Trainable design region behind a static conductive barrier slab.

    A ``MaterialRegion`` is a clean dielectric: the compiler zeroes ``sigma_e``
    inside its footprint (see ``_apply_material_regions``), so a design region
    cannot itself host conductive cells. The barrier is therefore placed
    *alongside* the design, as a full-transverse lossy layer between the source
    and the design/probe. The probe field must cross the barrier, so conduction
    genuinely damps the objective, and the design cotangent is back-propagated
    through the barrier's semi-implicit conduction update -- exercising the
    native conductive reverse transpose that the linear standard/CPML rule drops.
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
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(1.2, 1.2, 0.12)),
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
            position=(0.0, 0.0, -0.30),
            polarization="Ez",
            width=0.04,
            source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, amplitude=50.0),
        )
    )
    scene.add_monitor(mw.PointMonitor("probe", (0.06, 0.06, 0.30), fields=("Ez",)))
    return scene


class _ConductiveDensityScene(mw.SceneModule):
    """Trainable design density behind a static conductive (sigma_e) barrier.

    The design cotangent is back-propagated through the lossy barrier, so the
    reverse must transpose the semi-implicit conduction update for the design
    gradients to be correct -- the sensitivity the linear reverse drops.
    """

    def __init__(self, init=0.0, sigma_e=_SIGMA_E):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.full((2, 2, 2), float(init), device="cuda"))
        self._sigma = sigma_e

    def to_scene(self):
        return _conductive_scene(self._sigma, density=torch.sigmoid(self.logits))


@_CUDA
def test_scene_with_conductive_medium_gradient_matches_fd():
    """Per-element FD validation of design gradients whose adjoint is
    back-propagated through a static conductive barrier (previously rejected by
    the bridge)."""
    model = _ConductiveDensityScene(init=0.0)

    def loss_fn():
        result = _build_sim(model, time_steps=200).run()
        return _abs2(result.monitor("probe")["data"])

    # The conduction loss must actually shape the solution, otherwise this would
    # only re-validate the lossless path (which the linear reverse handles). The
    # barrier sits between source and probe, so a realistic sigma_e damps the
    # probe field several-fold.
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
def test_conductive_reverse_routes_through_native_conductive():
    """A conductive CPML scene must route to the analytic native conductive
    reverse backend: it carries the eps sensitivity of the semi-implicit
    conduction-loss decay/curl pair that the linear standard/CPML reverse drops,
    so it no longer falls back to the torch-VJP autograd path."""
    from witwin.maxwell.fdtd.adjoint.dispatch import _select_reverse_backend, _ReverseBackend
    from witwin.maxwell.fdtd.checkpoint import checkpoint_schema

    prepared = mw.Simulation.fdtd(
        _conductive_scene(_SIGMA_E),
        frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=8),
    ).prepare()
    solver = prepared.solver
    assert solver.conductive_enabled
    # The conductive reverse composes with the CPML checkpoint layout, so the
    # backend must be selected against the full frozen state (fields + psi).
    forward_state = {name: getattr(solver, name) for name in checkpoint_schema(solver).state_names}
    backend = _select_reverse_backend(
        solver,
        forward_state,
        eps_ex=solver.eps_Ex,
        eps_ey=solver.eps_Ey,
        eps_ez=solver.eps_Ez,
        resolved_source_terms=None,
    )
    assert backend is _ReverseBackend.CONDUCTIVE, backend


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


@_CUDA
def test_kerr_reverse_routes_through_native_kerr():
    """An instantaneous Kerr (chi3) CPML scene must route to the analytic native
    Kerr reverse backend: it carries the field / chi3 sensitivity of the per-step
    dynamic curl recompute that the linear CPML reverse drops, so it no longer
    falls back to the torch-VJP autograd path."""
    from witwin.maxwell.fdtd.adjoint.dispatch import _select_reverse_backend, _ReverseBackend
    from witwin.maxwell.fdtd.checkpoint import checkpoint_schema

    prepared = mw.Simulation.fdtd(
        _kerr_scene(_KERR_CHI3),
        frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=8),
    ).prepare()
    solver = prepared.solver
    assert solver.kerr_enabled and not getattr(solver, "nonlinear_general_enabled", False)
    forward_state = {name: getattr(solver, name) for name in checkpoint_schema(solver).state_names}
    backend = _select_reverse_backend(
        solver,
        forward_state,
        eps_ex=solver.eps_Ex,
        eps_ey=solver.eps_Ey,
        eps_ez=solver.eps_Ez,
        resolved_source_terms=None,
    )
    assert backend is _ReverseBackend.KERR, backend



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


# ---------------------------------------------------------------------------
# chi2 (second-order susceptibility) and two-photon-absorption (TPA) media
#
# chi2 and TPA drive the general nonlinear coefficient kernel, which rewrites
# BOTH the semi-implicit decay and curl every step from the pre-update fields
# (eps_eff = eps + eps0*(chi2*E_own + chi3*|E|^2); sigma = sigma_static +
# tpa*|E|^2). The adjoint replay must replicate that kernel differentiably.
# ---------------------------------------------------------------------------

_CHI2 = 8.0e-3
_TPA_BETA = 40.0


def _chi2_scene(chi2, *, density=None, boundary="pml"):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6))),
        grid=mw.GridSpec.uniform(0.12),
        boundary=(
            mw.BoundarySpec.pml(num_layers=2, strength=1.0)
            if boundary == "pml"
            else mw.BoundarySpec.none()
        ),
        device="cuda",
    )
    if chi2 != 0.0:
        scene.add_structure(
            mw.Structure(
                name="chi2_slab",
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.60, 0.60, 0.12)),
                material=mw.Material(
                    eps_r=2.0,
                    nonlinearity=[mw.NonlinearSusceptibility(chi2=chi2)],
                ),
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


def _tpa_scene(beta, *, density=None):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6))),
        grid=mw.GridSpec.uniform(0.12),
        boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
        device="cuda",
    )
    if beta != 0.0:
        scene.add_structure(
            mw.Structure(
                name="tpa_slab",
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.60, 0.60, 0.12)),
                material=mw.Material(
                    eps_r=2.0, nonlinearity=[mw.TwoPhotonAbsorption(beta=beta)]
                ),
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
            source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, amplitude=80.0),
        )
    )
    scene.add_monitor(mw.PointMonitor("probe", (0.0, 0.0, 0.18), fields=("Ez",)))
    return scene


def _general_nonlinear_scene():
    """A slab carrying chi2, chi3, and TPA together so the coefficient replica
    exercises every term of the general nonlinear kernel."""
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6))),
        grid=mw.GridSpec.uniform(0.12),
        boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(
            name="general_slab",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.60, 0.60, 0.12)),
            material=mw.Material(
                eps_r=2.0,
                nonlinearity=[
                    mw.NonlinearSusceptibility(chi2=_CHI2, chi3=_KERR_CHI3),
                    mw.TwoPhotonAbsorption(beta=_TPA_BETA),
                ],
            ),
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
def test_general_nonlinear_coefficient_replica_matches_cuda_kernel():
    """The differentiable general nonlinear coefficient replica used by the
    adjoint replay must match the native updateNonlinearElectricCoefficients3D
    kernel (both the dynamic decay and dynamic curl outputs)."""
    from witwin.maxwell.fdtd.adjoint.core import _general_nonlinear_electric_coefficients

    torch.manual_seed(13)
    prepared = mw.Simulation.fdtd(
        _general_nonlinear_scene(),
        frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=8),
    ).prepare()
    solver = prepared.solver
    assert solver.nonlinear_general_enabled
    assert solver.chi2_enabled and solver.tpa_enabled and solver.kerr_enabled

    solver.Ex.copy_(3.0 * torch.randn_like(solver.Ex))
    solver.Ey.copy_(3.0 * torch.randn_like(solver.Ey))
    solver.Ez.copy_(3.0 * torch.randn_like(solver.Ez))
    solver._update_nonlinear_electric_coefficients()

    state = {"Ex": solver.Ex, "Ey": solver.Ey, "Ez": solver.Ez}
    replica = _general_nonlinear_electric_coefficients(
        solver,
        state,
        eps_ex=solver.eps_Ex,
        eps_ey=solver.eps_Ey,
        eps_ez=solver.eps_Ez,
        chi2_ex=solver.nonlinear_chi2_Ex,
        chi2_ey=solver.nonlinear_chi2_Ey,
        chi2_ez=solver.nonlinear_chi2_Ez,
        chi3_ex=solver.kerr_chi3_Ex,
        chi3_ey=solver.kerr_chi3_Ey,
        chi3_ez=solver.kerr_chi3_Ez,
        tpa_ex=solver.tpa_sigma_Ex,
        tpa_ey=solver.tpa_sigma_Ey,
        tpa_ez=solver.tpa_sigma_Ez,
    )

    for component, decay_dynamic, curl_dynamic in (
        ("Ex", solver.cex_decay_dynamic, solver.cex_curl_dynamic),
        ("Ey", solver.cey_decay_dynamic, solver.cey_curl_dynamic),
        ("Ez", solver.cez_decay_dynamic, solver.cez_curl_dynamic),
    ):
        replica_decay, replica_curl = replica[component]
        assert torch.allclose(replica_decay, decay_dynamic, rtol=1e-5, atol=1e-12), component
        assert torch.allclose(replica_curl, curl_dynamic, rtol=1e-5, atol=1e-20), component


class _Chi2DensityScene(mw.SceneModule):
    """Trainable design density next to a static chi2 slab.

    The reverse step must differentiate the general nonlinear decay/curl
    coefficients through the forward E fields for the design gradients to be
    correct (chi2 folds the signed own-component field into eps_eff)."""

    def __init__(self, init=0.0, chi2=1.0e-4, boundary="pml"):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.full((2, 2, 2), float(init), device="cuda"))
        self._chi2 = chi2
        self._boundary = boundary

    def to_scene(self):
        return _chi2_scene(
            self._chi2,
            density=torch.sigmoid(self.logits),
            boundary=self._boundary,
        )


@_CUDA
@pytest.mark.parametrize("boundary", ["pml", "none"])
def test_scene_with_chi2_medium_gradient_matches_fd(boundary):
    """Per-element FD validation of design gradients in a scene containing a
    static chi2 slab (previously rejected by the bridge)."""
    model = _Chi2DensityScene(init=0.0, boundary=boundary)

    def loss_fn():
        result = _build_sim(model, time_steps=200).run()
        return _abs2(result.monitor("probe")["data"])

    # The chi2 nonlinearity must actually shape the solution, otherwise this
    # would only re-validate the linear path.
    linear_loss = _abs2(
        _build_sim(_Chi2DensityScene(init=0.0, chi2=0.0, boundary=boundary), time_steps=200)
        .run()
        .monitor("probe")["data"]
    ).item()
    chi2_loss = loss_fn().item()
    assert abs(chi2_loss - linear_loss) > 1e-3 * abs(linear_loss), (
        f"chi2 term inactive: linear={linear_loss:.6e}, chi2={chi2_loss:.6e}"
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


class _TpaDensityScene(mw.SceneModule):
    """Trainable design density next to a static two-photon-absorption slab.

    TPA folds a field-dependent conductivity sigma = tpa_sigma*|E|^2 into the
    semi-implicit loss term; the reverse step must differentiate that decay/curl
    pair through the forward fields."""

    def __init__(self, init=0.0, beta=_TPA_BETA):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.full((2, 2, 2), float(init), device="cuda"))
        self._beta = beta

    def to_scene(self):
        return _tpa_scene(self._beta, density=torch.sigmoid(self.logits))


@_CUDA
def test_scene_with_tpa_medium_gradient_matches_fd():
    """Per-element FD validation of design gradients in a scene whose static slab
    carries two-photon absorption (previously rejected by the bridge)."""
    model = _TpaDensityScene(init=0.0)

    def loss_fn():
        result = _build_sim(model, time_steps=200).run()
        return _abs2(result.monitor("probe")["data"])

    # The intensity-dependent loss must actually damp the solution.
    lossless = _abs2(
        _build_sim(_TpaDensityScene(init=0.0, beta=0.0), time_steps=200)
        .run()
        .monitor("probe")["data"]
    ).item()
    absorbed = loss_fn().item()
    assert absorbed < 0.999 * lossless, (
        f"TPA term inactive: lossless={lossless:.6e}, absorbed={absorbed:.6e}"
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


class _Chi2GeometryScene(mw.SceneModule):
    """Trainable-size chi2 box: the logits move the chi2 (and eps) blend."""

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
                name="chi2_box",
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=size),
                material=mw.Material(
                    eps_r=2.0,
                    nonlinearity=[
                        mw.NonlinearSusceptibility(chi2=_CHI2),
                        mw.TwoPhotonAbsorption(beta=_TPA_BETA),
                    ],
                ),
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


def _pullback_transpose_reference(model, *, field_key, grad_ex_kwarg, grad_ey_kwarg, grad_ez_kwarg):
    """Shared body for the chi2 / TPA pullback exact-transpose unit tests.

    Feeds random per-edge cotangents through the material pullback and checks the
    returned design gradient equals the autograd transpose of the node->Yee-edge
    averaging of the corresponding compiled node channel."""
    inputs = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    scene = model.to_scene()
    prepared = prepare_scene(scene)
    zero_eps_ex = torch.zeros((prepared.Nx - 1, prepared.Ny, prepared.Nz), device="cuda")
    zero_eps_ey = torch.zeros((prepared.Nx, prepared.Ny - 1, prepared.Nz), device="cuda")
    zero_eps_ez = torch.zeros((prepared.Nx, prepared.Ny, prepared.Nz - 1), device="cuda")
    grad_ex = torch.randn_like(zero_eps_ex)
    grad_ey = torch.randn_like(zero_eps_ey)
    grad_ez = torch.randn_like(zero_eps_ez)

    with torch.enable_grad():
        outputs = pullback_material_input_gradients(
            scene,
            inputs=inputs,
            grad_eps_ex=zero_eps_ex,
            grad_eps_ey=zero_eps_ey,
            grad_eps_ez=zero_eps_ez,
            eps0=_EPS0,
            **{grad_ex_kwarg: grad_ex, grad_ey_kwarg: grad_ey, grad_ez_kwarg: grad_ez},
        )

        reference_prepared = prepare_scene(model.to_scene())
        node_field = reference_prepared.compile_materials()[field_key]
        objective = (
            (grad_ex * average_node_to_component(None, node_field, "Ex")).sum()
            + (grad_ey * average_node_to_component(None, node_field, "Ey")).sum()
            + (grad_ez * average_node_to_component(None, node_field, "Ez")).sum()
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
def test_general_nonlinear_pullback_chi2_channel_is_exact_transpose():
    """The chi2 gradient channel of the material pullback must be the exact
    transpose of the node->Yee-edge chi2 averaging."""
    torch.manual_seed(7)
    model = _Chi2GeometryScene(init=0.0).cuda()
    _pullback_transpose_reference(
        model,
        field_key="chi2",
        grad_ex_kwarg="grad_chi2_ex",
        grad_ey_kwarg="grad_chi2_ey",
        grad_ez_kwarg="grad_chi2_ez",
    )


@_CUDA
def test_general_nonlinear_pullback_tpa_channel_is_exact_transpose():
    """The TPA-conductivity gradient channel of the material pullback must be the
    exact transpose of the node->Yee-edge tpa_sigma averaging."""
    torch.manual_seed(8)
    model = _Chi2GeometryScene(init=0.0).cuda()
    _pullback_transpose_reference(
        model,
        field_key="tpa_sigma",
        grad_ex_kwarg="grad_tpa_ex",
        grad_ey_kwarg="grad_tpa_ey",
        grad_ez_kwarg="grad_tpa_ez",
    )


@_CUDA
def test_chi2_reverse_routes_through_native_general_nonlinear():
    from witwin.maxwell.fdtd.adjoint.dispatch import (
        _native_backend_available,
        _select_reverse_backend,
        _ReverseBackend,
    )
    from witwin.maxwell.fdtd.checkpoint import capture_checkpoint_state

    prepared = mw.Simulation.fdtd(
        _general_nonlinear_scene(),
        frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=8),
    ).prepare()
    solver = prepared.solver
    assert solver.nonlinear_general_enabled
    forward_state = capture_checkpoint_state(solver, step=0).tensors
    backend = _select_reverse_backend(
        solver,
        forward_state,
        eps_ex=solver.eps_Ex,
        eps_ey=solver.eps_Ey,
        eps_ez=solver.eps_Ez,
        resolved_source_terms=None,
    )
    assert backend is _ReverseBackend.GENERAL_NONLINEAR, backend
    assert _native_backend_available(backend, solver, forward_state)


# ---------------------------------------------------------------------------
# Full (off-diagonal) anisotropic epsilon (Tensor3x3) media
#
# Full anisotropy uses a component-coupled forward update: the base curl carries
# the diagonal inverse-permittivity entry, and a separate kernel adds the
# off-diagonal coupling E_i += dt/eps0 * inv_ij * <curlH_j>. The adjoint replay
# must replicate that off-diagonal correction differentiably so the reverse
# propagates cotangents through the coupling; the scene routes to the torch-VJP
# reverse because the analytic backends model the diagonal curl only.
# ---------------------------------------------------------------------------

# Symmetric positive-definite tensor with a strong xz/yz coupling so an
# Ez-driven, Ez-probed scene is genuinely reshaped by the off-diagonal terms.
_FULL_ANISO_OFFDIAG = 1.2


def _full_aniso_tensor(offdiag):
    return mw.Tensor3x3(
        (
            (4.0, 0.0, offdiag),
            (0.0, 4.0, offdiag),
            (offdiag, offdiag, 5.0),
        )
    )


def _full_aniso_scene(offdiag=_FULL_ANISO_OFFDIAG, *, density=None):
    """Static full-anisotropic slab between the source and an isotropic design.

    The trainable design permittivity sits behind the tensor slab, so its
    gradient depends on the field transmitted through (and the adjoint field
    reflected back across) the off-diagonal coupling.
    """
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6))),
        grid=mw.GridSpec.uniform(0.12),
        boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(
            name="full_aniso_slab",
            # Kept clear of the 2-layer PML: the off-diagonal correction is only
            # exact outside the absorber, so the slab (and its soft-SDF tail) must
            # not touch it.
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.36, 0.36, 0.12)),
            material=mw.Material(epsilon_tensor=_full_aniso_tensor(offdiag)),
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


class _FullAnisoDensityScene(mw.SceneModule):
    """Trainable design density behind a static full-anisotropic (Tensor3x3) slab."""

    def __init__(self, init=0.0, offdiag=_FULL_ANISO_OFFDIAG):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.full((2, 2, 2), float(init), device="cuda"))
        self._offdiag = offdiag

    def to_scene(self):
        return _full_aniso_scene(self._offdiag, density=torch.sigmoid(self.logits))


@_CUDA
def test_full_aniso_electric_correction_replica_matches_cuda_kernel():
    """The differentiable off-diagonal correction used by the adjoint replay must
    match the native updateElectricFieldE{x,y,z}FullAniso3D kernels.

    This is the forward-consistency guard the collocation transpose needs: a
    diagonal-dominant tensor would hide a mis-collocated off-axis curl, so the
    scene uses strong xz/yz coupling and random magnetic fields."""
    from witwin.maxwell.fdtd.adjoint.core import _full_aniso_electric_correction
    from witwin.maxwell.fdtd.runtime.stepping import apply_full_aniso_corrections

    torch.manual_seed(17)
    prepared = mw.Simulation.fdtd(
        _full_aniso_scene(),
        frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=8),
    ).prepare()
    solver = prepared.solver
    assert solver.full_aniso_enabled

    hx = torch.randn_like(solver.Hx)
    hy = torch.randn_like(solver.Hy)
    hz = torch.randn_like(solver.Hz)

    # Native kernel path: zero E, inject the random H, apply the correction in place.
    solver.Ex.zero_()
    solver.Ey.zero_()
    solver.Ez.zero_()
    solver.Hx.copy_(hx)
    solver.Hy.copy_(hy)
    solver.Hz.copy_(hz)
    apply_full_aniso_corrections(solver)
    kernels = {"Ex": solver.Ex.clone(), "Ey": solver.Ey.clone(), "Ez": solver.Ez.clone()}

    replica = _full_aniso_electric_correction(
        solver, {"Hx": hx, "Hy": hy, "Hz": hz}
    )

    # The correction must be genuinely active (the off-diagonal coupling is not a
    # no-op), otherwise a broken replica of zeros would pass trivially.
    assert kernels["Ez"].abs().max().item() > 0.0
    for name, kernel in kernels.items():
        scale = max(kernel.abs().max().item(), 1.0)
        # Tight bit-for-bit agreement (well under 1e-3): this is the exact
        # correctness check for the new off-diagonal reverse physics, and the
        # strong xz/yz coupling means a mis-collocated off-axis curl cannot hide.
        assert torch.allclose(replica[name], kernel, rtol=1e-5, atol=1e-6 * scale), name


@_CUDA
def test_full_aniso_reverse_routes_through_native_full_aniso():
    """A full-anisotropic CPML scene must route to the analytic native full-aniso
    reverse backend: it carries the off-diagonal coupling the linear standard/CPML
    reverse drops (folded into the mid-step H adjoint), so it no longer falls back
    to the torch-VJP autograd path."""
    from witwin.maxwell.fdtd.adjoint.dispatch import _select_reverse_backend, _ReverseBackend
    from witwin.maxwell.fdtd.checkpoint import checkpoint_schema

    prepared = mw.Simulation.fdtd(
        _full_aniso_scene(),
        frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=8),
    ).prepare()
    solver = prepared.solver
    assert solver.full_aniso_enabled
    # The off-diagonal reverse composes with the CPML checkpoint layout, so the
    # backend must be selected against the full frozen state (fields + psi).
    forward_state = {name: getattr(solver, name) for name in checkpoint_schema(solver).state_names}
    backend = _select_reverse_backend(
        solver,
        forward_state,
        eps_ex=solver.eps_Ex,
        eps_ey=solver.eps_Ey,
        eps_ez=solver.eps_Ez,
        resolved_source_terms=None,
    )
    assert backend is _ReverseBackend.FULL_ANISO, backend


@_CUDA
def test_full_aniso_bridge_forward_matches_reference_physics():
    """The adjoint-bridge forward must apply the off-diagonal coupling exactly
    like the plain forward, otherwise the checkpointed trajectory (and the design
    gradient built from it) would optimize the wrong physics."""
    density = torch.sigmoid(torch.zeros((2, 2, 2), device="cuda"))
    reference = _abs2(
        mw.Simulation.fdtd(
            _full_aniso_scene(density=density),
            frequencies=[1e9],
            run_time=mw.TimeConfig(time_steps=200),
            spectral_sampler=mw.SpectralSampler(window="none"),
            full_field_dft=False,
        )
        .run()
        .monitor("probe")["data"]
    ).item()

    model = _FullAnisoDensityScene(init=0.0)
    bridge = _abs2(_build_sim(model, time_steps=200).run().monitor("probe")["data"]).item()

    assert abs(bridge - reference) <= 1e-5 * abs(reference), (
        f"bridge forward diverges from reference physics: bridge={bridge:.6e}, reference={reference:.6e}"
    )


@_CUDA
def test_scene_with_full_aniso_medium_gradient_matches_fd():
    """End-to-end FD validation of design gradients behind a static
    full-anisotropic (off-diagonal Tensor3x3) slab (previously rejected by the
    bridge).

    Tolerance matches the landed diagonal-anisotropic adjoint test
    (``allclose(rtol=3e-2)``): anisotropic FDTD scenes carry a precision floor on
    end-to-end FD-gradient agreement. Here the whole electric step is replayed on
    the torch-VJP reverse, and its per-step reparametrized Jacobian differs from
    the native kernels at float level; that mismatch compounds through the reverse
    rollout of the high-index tensor cavity to ~0.5% for strong coupling. The
    exact correctness of the new off-diagonal physics is pinned separately by the
    ``..._correction_replica_matches_cuda_kernel`` forward-consistency test at
    rtol 1e-5."""
    model = _FullAnisoDensityScene(init=0.0)

    def loss_fn():
        result = _build_sim(model, time_steps=200).run()
        return _abs2(result.monitor("probe")["data"])

    # The off-diagonal coupling must actually reshape the solution, otherwise
    # this would only re-validate the diagonal-anisotropic path.
    diagonal_only = _abs2(
        _build_sim(_FullAnisoDensityScene(init=0.0, offdiag=0.0), time_steps=200)
        .run()
        .monitor("probe")["data"]
    ).item()
    coupled = loss_fn().item()
    assert abs(coupled - diagonal_only) > 1e-2 * abs(diagonal_only), (
        f"off-diagonal coupling inactive: diagonal={diagonal_only:.6e}, coupled={coupled:.6e}"
    )

    backward_grad, fd_grad = _backward_and_fd_grads(model, loss_fn)

    assert (fd_grad != 0).any(), "FD gradient is identically zero; test setup is broken."
    scale = torch.abs(fd_grad).max().item()
    assert torch.allclose(backward_grad, fd_grad, rtol=3e-2, atol=scale * 5e-3), (
        f"backward={backward_grad.flatten().tolist()}, fd={fd_grad.flatten().tolist()}"
    )


# ---------------------------------------------------------------------------
# Bloch (periodic) + electric-dispersive media
#
# Under Bloch boundaries the solver propagates complex fields: a second real
# FDTD copy carrying the imaginary (quadrature) component that the phased
# boundary wrap couples in. The electric ADE poles advance on BOTH halves -- the
# forward keeps a real and an imaginary polarization current per pole and applies
# each to its own field. The adjoint checkpoint replay must reproduce that
# imaginary ADE replica; otherwise the checkpointed trajectory (and every
# gradient built from it) drifts on the imaginary field. The scene routes to the
# torch-VJP reverse because the analytic Bloch backend rejects dispersion and the
# analytic dispersive backend rejects complex fields.
#
# The Bloch phase angle is k * L; a multiple of pi has sin = 0 and never excites
# the imaginary field, which would make the imaginary dispersion a silent no-op.
# k = pi/1.8 with L = 1.2 gives angle 2*pi/3 (sin ~ 0.87), keeping the imaginary
# field comparable to the real field so the new physics is genuinely exercised.
# ---------------------------------------------------------------------------

_BLOCH_DISP_K = math.pi / 1.8


def _bloch_dispersive_scene(delta_eps, *, density=None):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6))),
        grid=mw.GridSpec.uniform(0.12),
        boundary=mw.BoundarySpec.bloch((_BLOCH_DISP_K, 0.0, 0.0)),
        device="cuda",
    )
    # Lorentz slab perpendicular to the Bloch (x) propagation axis: the phased
    # wrap-around excites a strong imaginary field inside the dispersive region,
    # so its imaginary polarization current genuinely shapes the design gradient.
    scene.add_structure(
        mw.Structure(
            name="lorentz_slab",
            geometry=mw.Box(position=(-0.06, 0.0, 0.0), size=(0.60, 1.2, 1.2)),
            material=mw.Material.lorentz(
                eps_inf=2.0,
                delta_eps=delta_eps,
                resonance_frequency=1.1e9,
                gamma=0.2e9,
            ),
        )
    )
    if density is not None:
        scene.add_material_region(
            mw.MaterialRegion(
                name="design",
                geometry=mw.Box(position=(0.30, 0.06, 0.06), size=(0.24, 0.24, 0.24)),
                density=density,
                eps_bounds=(1.0, 6.0),
                mu_bounds=(1.0, 1.0),
            )
        )
    scene.add_source(
        mw.PointDipole(
            position=(-0.54, 0.06, 0.06),
            polarization="Ez",
            width=0.12,
            source_time=mw.GaussianPulse(frequency=1.0e9, fwidth=0.3e9, amplitude=25.0),
        )
    )
    scene.add_monitor(mw.PointMonitor("probe", (0.30, 0.06, 0.06), fields=("Ez",)))
    return scene


class _BlochDispersiveDensityScene(mw.SceneModule):
    """Trainable design density behind a static Lorentz slab under Bloch boundaries."""

    def __init__(self, init=0.0, delta_eps=3.5):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.full((2, 2, 2), float(init), device="cuda"))
        self._delta_eps = delta_eps

    def to_scene(self):
        return _bloch_dispersive_scene(self._delta_eps, density=torch.sigmoid(self.logits))


@_CUDA
def test_bloch_dispersive_reverse_selects_native_bloch_dispersive_backend():
    """A Bloch + electric-dispersive scene resolves to the dedicated analytic
    ``BLOCH_DISPERSIVE`` backend, which carries the imaginary-ADE replica the
    plain Bloch and plain dispersive backends each drop, and (on a qualifying CUDA
    scene) auto mode prefers its fused native reverse runner over the torch-VJP
    fallback."""
    from witwin.maxwell.fdtd.adjoint.dispatch import (
        _native_backend_available,
        _select_reverse_backend,
        _ReverseBackend,
    )
    from witwin.maxwell.fdtd.checkpoint import capture_checkpoint_state

    density = torch.sigmoid(torch.zeros((2, 2, 2), device="cuda"))
    prepared = mw.Simulation.fdtd(
        _bloch_dispersive_scene(3.5, density=density),
        frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=8),
    ).prepare()
    solver = prepared.solver
    assert solver.electric_dispersive_enabled and solver.complex_fields_enabled
    forward_state = capture_checkpoint_state(solver, step=0).tensors
    backend = _select_reverse_backend(
        solver,
        forward_state,
        eps_ex=solver.eps_Ex,
        eps_ey=solver.eps_Ey,
        eps_ez=solver.eps_Ez,
        resolved_source_terms=None,
    )
    assert backend is _ReverseBackend.BLOCH_DISPERSIVE, backend
    assert _native_backend_available(backend, solver, forward_state)


@_CUDA
def test_bloch_dispersive_bridge_replay_matches_forward_state():
    """Forward-consistency guard for the imaginary ADE replica.

    The checkpoint replay must reproduce the native Bloch + dispersive forward
    exactly, including the imaginary polarization current the complex-field solver
    keeps per electric pole. This is the risk the reconnaissance flagged: dropping
    that imaginary current corrupts the checkpointed trajectory the reverse rolls
    back through. Measured empirically, omitting it drifts the replayed imaginary
    field by ~46% of its magnitude, whereas carrying it matches to ~1e-13."""
    from witwin.maxwell.fdtd.adjoint.bridge import _FDTDGradientBridge
    from witwin.maxwell.fdtd.adjoint.core import _replay_segment_states
    from witwin.maxwell.fdtd.checkpoint import dispersive_state_name

    model = _BlochDispersiveDensityScene(init=0.0).cuda()
    bridge = _FDTDGradientBridge(_build_sim(model, time_steps=56))
    bridge.forward(tuple(bridge.material_inputs))
    solver = bridge._last_solver

    # The nondegenerate Bloch phase must genuinely excite the imaginary field,
    # otherwise the imaginary dispersion is a silent no-op and this proves nothing.
    real_scale = solver.Ez.abs().max().item()
    imag_scale = solver.Ez_imag.abs().max().item()
    assert imag_scale > 0.1 * real_scale, (imag_scale, real_scale)

    full_states = _replay_segment_states(solver, bridge._last_checkpoints[0], 0, bridge._time_steps)
    terminal = full_states[-1]

    field_scale = max(
        getattr(solver, name).abs().max().item()
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
                     "Ex_imag", "Ey_imag", "Ez_imag", "Hx_imag", "Hy_imag", "Hz_imag")
    )
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
                 "Ex_imag", "Ey_imag", "Ez_imag", "Hx_imag", "Hy_imag", "Hz_imag"):
        err = (terminal[name] - getattr(solver, name)).abs().max().item()
        assert err < 1e-5 * field_scale, (name, err, field_scale)

    # The imaginary ADE replica must be genuinely driven and exactly replayed.
    imag_state_seen = False
    for component_name in ("Ex", "Ey", "Ez"):
        component_templates = solver._dispersive_templates.get(component_name, {})
        for model_name in ("debye", "drude", "lorentz"):
            for index, entry in enumerate(component_templates.get(model_name, ())):
                tensor_names = ("current",) if model_name == "drude" else ("polarization", "current")
                for tensor_name in tensor_names:
                    key = dispersive_state_name(component_name, model_name, index, tensor_name) + "_imag"
                    reference = entry[f"{tensor_name}_imag"]
                    state_scale = reference.abs().max().item()
                    imag_state_seen = imag_state_seen or state_scale > 0.0
                    err = (terminal[key] - reference).abs().max().item()
                    assert err < 1e-5 * max(state_scale, 1e-30), (key, err, state_scale)
    assert imag_state_seen, "imaginary dispersive ADE never activated; Bloch phase is degenerate."


@_CUDA
def test_scene_with_bloch_dispersive_medium_gradient_matches_fd():
    """Per-element FD validation of design gradients behind a static Lorentz slab
    under Bloch boundaries (previously rejected by the bridge)."""
    model = _BlochDispersiveDensityScene(init=0.0).cuda()

    # The run must be long enough for the dispersive polarization response to
    # propagate through the slab and reach the probe. At very short horizons the
    # probe samples only the near-field transient, where the Lorentz term is a
    # sub-percent perturbation (the ADE current has not built up), so the
    # dispersion-active precondition below would be dominated by roundoff. 200
    # steps captures the propagated dispersive response (~12% objective shift).
    def loss_fn():
        result = _build_sim(model, time_steps=200).run()
        return _abs2(result.monitor("probe")["data"])

    # Dispersion must actually reshape the solution versus a nondispersive slab of
    # the same eps_inf, otherwise this would only re-validate the nondispersive
    # Bloch path (which the earlier Bloch adjoint already covers).
    nondispersive = _abs2(
        _build_sim(_BlochDispersiveDensityScene(init=0.0, delta_eps=0.0), time_steps=200)
        .run()
        .monitor("probe")["data"]
    ).item()
    dispersive = loss_fn().item()
    assert abs(dispersive - nondispersive) > 1e-2 * abs(nondispersive), (
        f"dispersion inactive: nondispersive={nondispersive:.6e}, dispersive={dispersive:.6e}"
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


class _BlochMagneticDispersiveDensityScene(mw.SceneModule):
    """Bloch scene with a static magnetic-dispersive (mu-Lorentz) slab."""

    def __init__(self, init=0.0):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.full((2, 2, 2), float(init), device="cuda"))

    def to_scene(self):
        density = torch.sigmoid(self.logits)
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6))),
            grid=mw.GridSpec.uniform(0.15),
            boundary=mw.BoundarySpec.bloch((_BLOCH_DISP_K, 0.0, 0.0)),
            device="cuda",
        )
        scene.add_structure(
            mw.Structure(
                name="mu_lorentz_slab",
                geometry=mw.Box(position=(-0.15, 0.0, 0.0), size=(0.30, 1.2, 1.2)),
                material=mw.Material(
                    mu_lorentz_poles=(
                        mw.LorentzPole(delta_eps=1.0, resonance_frequency=1.2e9, gamma=2.0e8),
                    ),
                ),
            )
        )
        scene.add_material_region(
            mw.MaterialRegion(
                name="design",
                geometry=mw.Box(position=(0.15, 0.0, 0.0), size=(0.24, 0.24, 0.24)),
                density=density,
                eps_bounds=(1.0, 6.0),
                mu_bounds=(1.0, 1.0),
            )
        )
        scene.add_source(
            mw.PointDipole(
                position=(-0.54, 0.06, 0.06),
                polarization="Ez",
                width=0.12,
                source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.3e9, amplitude=25.0),
            )
        )
        scene.add_monitor(mw.PointMonitor("probe", (0.15, 0.0, 0.0), fields=("Ez",)))
        return scene


@_CUDA
def test_bloch_magnetic_dispersive_adjoint_runs_native():
    model = _BlochMagneticDispersiveDensityScene(init=0.0).cuda()
    loss = _abs2(_build_sim(model, time_steps=16).run().monitor("probe")["data"])
    loss.backward()
    assert model.logits.grad is not None
    assert torch.isfinite(model.logits.grad).all()


# ---------------------------------------------------------------------------
# Diagonal-anisotropic + electric-dispersive media (P5.2 combination)
#
# A DiagonalTensor3 background permittivity combined with a homogeneous electric
# pole exercises both the per-axis eps_Ex/Ey/Ez coefficient layout AND the ADE
# dispersive reverse in the same slab. The design density sits behind the slab, so
# its gradient depends on the field transmitted through (forward) and the adjoint
# field replayed back across (reverse) the anisotropic-dispersive medium. The
# diagonal anisotropy carries no off-diagonal coupling, so the electric ADE reverse
# already handles the combination through the per-axis eps fields.
# ---------------------------------------------------------------------------


class _AnisoDispersiveDensityScene(mw.SceneModule):
    """Trainable design density behind a static diagonal-anisotropic Lorentz slab."""

    def __init__(self, init=0.0, delta_eps=1.0):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.full((2, 2, 2), float(init), device="cuda"))
        self._delta_eps = delta_eps

    def to_scene(self):
        density = torch.sigmoid(self.logits)
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6))),
            grid=mw.GridSpec.uniform(0.12),
            boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
            device="cuda",
        )
        poles = (
            ()
            if self._delta_eps == 0.0
            else (mw.LorentzPole(delta_eps=self._delta_eps, resonance_frequency=1.5e9, gamma=3.0e8),)
        )
        scene.add_structure(
            mw.Structure(
                name="aniso_lorentz_slab",
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.60, 0.60, 0.12)),
                material=mw.Material(
                    epsilon_tensor=mw.DiagonalTensor3(2.0, 3.5, 5.0),
                    lorentz_poles=poles,
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
def test_scene_with_diagonal_aniso_dispersive_medium_gradient_matches_fd():
    """Per-element FD validation of design gradients behind a static
    diagonal-anisotropic Lorentz slab (the P5.2 diagonal-aniso-dispersion edge).

    The reverse replay must propagate the adjoint field back through both the
    per-axis anisotropic permittivity and the electric ADE pole state; a bug in
    either would corrupt the design-density gradient."""
    model = _AnisoDispersiveDensityScene(init=0.0).cuda()

    def loss_fn():
        result = _build_sim(model, time_steps=200).run()
        return _abs2(result.monitor("probe")["data"])

    # Dispersion must actually reshape the solution versus a nondispersive slab of
    # the same anisotropic eps_inf, otherwise this would only re-validate the
    # already-covered diagonal-anisotropic path.
    nondispersive = _abs2(
        _build_sim(_AnisoDispersiveDensityScene(init=0.0, delta_eps=0.0).cuda(), time_steps=200)
        .run()
        .monitor("probe")["data"]
    ).item()
    dispersive = loss_fn().item()
    assert abs(dispersive - nondispersive) > 1e-2 * abs(nondispersive), (
        f"dispersion inactive: nondispersive={nondispersive:.6e}, dispersive={dispersive:.6e}"
    )

    backward_grad, fd_grad = _backward_and_fd_grads(model, loss_fn)

    assert (fd_grad != 0).any(), "FD gradient is identically zero; test setup is broken."
    scale = torch.abs(fd_grad).max().item()
    assert torch.allclose(backward_grad, fd_grad, rtol=3e-2, atol=scale * 5e-3), (
        f"backward={backward_grad.flatten().tolist()}, fd={fd_grad.flatten().tolist()}"
    )
