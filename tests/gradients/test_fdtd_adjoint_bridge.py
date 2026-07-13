import math
from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.checkpoint import dispersive_state_name
from witwin.maxwell.fdtd.adjoint import _FDTDGradientBridge, _replay_segment_states
from witwin.maxwell.fdtd.boundary import BOUNDARY_BLOCH, BOUNDARY_NONE, BOUNDARY_PML
from witwin.maxwell.fdtd.checkpoint import capture_checkpoint_state
from witwin.maxwell.fdtd.material_pullback import pullback_density_gradients
from witwin.maxwell.scene import prepare_scene
from tests.gradients import fdtd_adjoint_baselines as adjoint_baselines


def _build_simulation(model, *, time_steps=24):
    return mw.Simulation.fdtd(
        model,
        frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=time_steps),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
    )


@pytest.mark.parametrize(
    ("material", "message"),
    [
        (
            mw.Material(mu_tensor=mw.DiagonalTensor3(2.0, 3.0, 4.0)),
            "anisotropic magnetic \\(mu_tensor\\) media",
        ),
        (
            mw.Material(eps_r=2.0, modulation=mw.ModulationSpec(frequency=1.0e8, amplitude=0.1)),
            "time-modulated media",
        ),
    ],
)
def test_fdtd_gradient_bridge_rejects_unsupported_medium_capabilities(material, message):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.25),
        device="cpu",
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.5, 0.5, 0.5)),
            material=material,
        )
    )

    bridge = object.__new__(_FDTDGradientBridge)
    with pytest.raises(NotImplementedError, match=message):
        bridge._validate_supported_configuration(SimpleNamespace(scene=scene))


def test_fdtd_gradient_bridge_accepts_full_anisotropic_epsilon():
    """Full (off-diagonal) Tensor3x3 epsilon is now differentiable: the reverse
    step replicates the off-diagonal coupling and routes to the torch-VJP
    backend, so a static full-anisotropic structure is accepted."""
    from witwin.maxwell.fdtd.adjoint.bridge import _unsupported_adjoint_medium

    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.25),
        device="cpu",
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.5, 0.5, 0.5)),
            material=mw.Material(
                eps_r=1.0,
                epsilon_tensor=mw.Tensor3x3(((2.5, 0.3, 0.0), (0.3, 2.5, 0.0), (0.0, 0.0, 2.5))),
            ),
        )
    )

    assert _unsupported_adjoint_medium(scene) is None


def test_fdtd_gradient_bridge_rejects_trainable_geometry_on_full_anisotropic_media():
    """A trainable geometry on a full-anisotropic structure is guarded: the
    off-diagonal coupling coefficients carry no material gradient channel, so the
    geometry sensitivity would be silently dropped."""
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.25),
        device="cpu",
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(
                position=(0.0, 0.0, 0.0),
                size=torch.tensor([0.5, 0.5, 0.5], requires_grad=True),
            ),
            material=mw.Material(
                eps_r=1.0,
                epsilon_tensor=mw.Tensor3x3(((2.5, 0.3, 0.0), (0.3, 2.5, 0.0), (0.0, 0.0, 2.5))),
            ),
        )
    )

    bridge = object.__new__(_FDTDGradientBridge)
    with pytest.raises(NotImplementedError, match="trainable geometry on full \\(off-diagonal\\) anisotropic"):
        bridge._validate_supported_configuration(SimpleNamespace(scene=scene))


def test_fdtd_gradient_bridge_accepts_diagonal_anisotropic_epsilon():
    from witwin.maxwell.fdtd.adjoint.bridge import _unsupported_adjoint_medium

    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.25),
        device="cpu",
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.5, 0.5, 0.5)),
            material=mw.Material(epsilon_tensor=mw.DiagonalTensor3(2.0, 3.0, 4.0)),
        )
    )

    assert _unsupported_adjoint_medium(scene) is None


def _magnetic_dispersive_material():
    return mw.Material(
        mu_lorentz_poles=(mw.LorentzPole(delta_eps=1.0, resonance_frequency=1.0e9, gamma=1.0e8),)
    )


def test_fdtd_gradient_bridge_accepts_static_magnetic_dispersive_media():
    from witwin.maxwell.fdtd.adjoint.bridge import _unsupported_adjoint_medium

    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.25),
        device="cpu",
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.5, 0.5, 0.5)),
            material=_magnetic_dispersive_material(),
        )
    )

    assert _unsupported_adjoint_medium(scene) is None


def test_fdtd_gradient_bridge_accepts_pure_kerr_media():
    from witwin.maxwell.fdtd.adjoint.bridge import _unsupported_adjoint_medium

    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.25),
        device="cpu",
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.5, 0.5, 0.5)),
            material=mw.Material(eps_r=2.0, kerr_chi3=1.0e-10),
        )
    )

    assert _unsupported_adjoint_medium(scene) is None


@pytest.mark.parametrize(
    "material",
    [
        mw.Material(eps_r=2.0, nonlinearity=mw.NonlinearSusceptibility(chi2=1.0e-12)),
        mw.Material(eps_r=2.0, nonlinearity=mw.TwoPhotonAbsorption(beta=1.0e-12)),
        mw.Material(
            eps_r=2.0,
            nonlinearity=(
                mw.NonlinearSusceptibility(chi2=1.0e-12, chi3=1.0e-10),
                mw.TwoPhotonAbsorption(beta=1.0e-12),
            ),
        ),
    ],
)
def test_fdtd_gradient_bridge_accepts_general_nonlinear_media(material):
    """chi2 and two-photon absorption drive the general nonlinear coefficient
    kernel, which the adjoint replay now replicates differentiably."""
    from witwin.maxwell.fdtd.adjoint.bridge import _unsupported_adjoint_medium

    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.25),
        device="cpu",
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.5, 0.5, 0.5)),
            material=material,
        )
    )

    assert _unsupported_adjoint_medium(scene) is None


def test_fdtd_gradient_bridge_rejects_trainable_geometry_on_magnetic_dispersive_media():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.25),
        device="cpu",
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(
                position=(0.0, 0.0, 0.0),
                size=torch.tensor([0.5, 0.5, 0.5], requires_grad=True),
            ),
            material=_magnetic_dispersive_material(),
        )
    )

    bridge = object.__new__(_FDTDGradientBridge)
    with pytest.raises(NotImplementedError, match="trainable geometry on magnetic dispersive"):
        bridge._validate_supported_configuration(SimpleNamespace(scene=scene))


def _fake_checkpoint_solver():
    solver = SimpleNamespace(
        complex_fields_enabled=True,
        uses_cpml=True,
        tfsf_enabled=True,
        _tfsf_state={
            "auxiliary_grid": SimpleNamespace(
                electric=torch.arange(4, dtype=torch.float32),
                magnetic=torch.arange(4, dtype=torch.float32) + 10.0,
            )
        },
        dispersive_enabled=True,
        _dispersive_templates={
            "Ex": {
                "debye": [
                    {
                        "polarization": torch.full((1,), 1.0),
                        "current": torch.full((1,), 2.0),
                        "polarization_imag": torch.full((1,), 101.0),
                        "current_imag": torch.full((1,), 102.0),
                    }
                ],
                "drude": [],
                "lorentz": [],
            },
            "Ey": {
                "debye": [],
                "drude": [{"current": torch.full((1,), 3.0), "current_imag": torch.full((1,), 103.0)}],
                "lorentz": [],
            },
            "Ez": {
                "debye": [],
                "drude": [],
                "lorentz": [
                    {
                        "polarization": torch.full((1,), 4.0),
                        "current": torch.full((1,), 5.0),
                        "polarization_imag": torch.full((1,), 104.0),
                        "current_imag": torch.full((1,), 105.0),
                    }
                ],
            },
        },
    )
    for index, name in enumerate(
        (
            "Ex",
            "Ey",
            "Ez",
            "Hx",
            "Hy",
            "Hz",
            "Ex_imag",
            "Ey_imag",
            "Ez_imag",
            "Hx_imag",
            "Hy_imag",
            "Hz_imag",
            "psi_ex_y",
            "psi_ex_z",
            "psi_ey_x",
            "psi_ey_z",
            "psi_ez_x",
            "psi_ez_y",
            "psi_hx_y",
            "psi_hx_z",
            "psi_hy_x",
            "psi_hy_z",
            "psi_hz_x",
            "psi_hz_y",
            "psi_ex_y_imag",
            "psi_ex_z_imag",
            "psi_ey_x_imag",
            "psi_ey_z_imag",
            "psi_ez_x_imag",
            "psi_ez_y_imag",
            "psi_hx_y_imag",
            "psi_hx_z_imag",
            "psi_hy_x_imag",
            "psi_hy_z_imag",
            "psi_hz_x_imag",
            "psi_hz_y_imag",
        )
    ):
        setattr(solver, name, torch.full((2,), float(index + 1)))
    return solver


def _fake_standard_reverse_solver():
    ex_shape = (2, 4, 5)
    ey_shape = (3, 3, 5)
    ez_shape = (3, 4, 4)
    hx_shape = (3, 3, 4)
    hy_shape = (2, 4, 4)
    hz_shape = (2, 3, 5)
    return SimpleNamespace(
        uses_cpml=False,
        complex_fields_enabled=False,
        dispersive_enabled=False,
        tfsf_enabled=False,
        has_pec_faces=False,
        boundary_x_low_code=BOUNDARY_NONE,
        boundary_x_high_code=BOUNDARY_NONE,
        boundary_y_low_code=BOUNDARY_NONE,
        boundary_y_high_code=BOUNDARY_NONE,
        boundary_z_low_code=BOUNDARY_NONE,
        boundary_z_high_code=BOUNDARY_NONE,
        inv_dx_e=torch.full((3,), 1.25),
        inv_dy_e=torch.full((4,), 0.75),
        inv_dz_e=torch.full((5,), 1.5),
        inv_dx_h=torch.full((2,), 1.25),
        inv_dy_h=torch.full((3,), 0.75),
        inv_dz_h=torch.full((4,), 1.5),
        chx_decay=torch.full(hx_shape, 0.97),
        chy_decay=torch.full(hy_shape, 0.96),
        chz_decay=torch.full(hz_shape, 0.95),
        chx_curl=torch.full(hx_shape, 0.11),
        chy_curl=torch.full(hy_shape, 0.09),
        chz_curl=torch.full(hz_shape, 0.07),
        cex_decay=torch.full(ex_shape, 0.94),
        cey_decay=torch.full(ey_shape, 0.93),
        cez_decay=torch.full(ez_shape, 0.92),
        cex_curl=torch.full(ex_shape, 0.13),
        cey_curl=torch.full(ey_shape, 0.12),
        cez_curl=torch.full(ez_shape, 0.10),
        eps_Ex=torch.full(ex_shape, 2.0),
        eps_Ey=torch.full(ey_shape, 2.5),
        eps_Ez=torch.full(ez_shape, 3.0),
        _source_time={"kind": "cw", "kind_code": 0, "amplitude": 0.0, "frequency": 1.0, "phase": 0.0, "delay": 0.0},
        source_omega=2.0 * math.pi,
        _compiled_source=None,
        _source_terms=[],
        _electric_source_terms=[],
        _magnetic_source_terms=[],
    )


def _cpml_reverse_state_shapes():
    ex_shape = (2, 4, 5)
    ey_shape = (3, 3, 5)
    ez_shape = (3, 4, 4)
    hx_shape = (3, 3, 4)
    hy_shape = (2, 4, 4)
    hz_shape = (2, 3, 5)
    return {
        "Ex": ex_shape,
        "Ey": ey_shape,
        "Ez": ez_shape,
        "Hx": hx_shape,
        "Hy": hy_shape,
        "Hz": hz_shape,
        "psi_ex_y": ex_shape,
        "psi_ex_z": ex_shape,
        "psi_ey_x": ey_shape,
        "psi_ey_z": ey_shape,
        "psi_ez_x": ez_shape,
        "psi_ez_y": ez_shape,
        "psi_hx_y": hx_shape,
        "psi_hx_z": hx_shape,
        "psi_hy_x": hy_shape,
        "psi_hy_z": hy_shape,
        "psi_hz_x": hz_shape,
        "psi_hz_y": hz_shape,
    }


def _fake_cpml_reverse_solver():
    shapes = _cpml_reverse_state_shapes()
    ex_shape = shapes["Ex"]
    ey_shape = shapes["Ey"]
    ez_shape = shapes["Ez"]
    hx_shape = shapes["Hx"]
    hy_shape = shapes["Hy"]
    hz_shape = shapes["Hz"]
    return SimpleNamespace(
        uses_cpml=True,
        complex_fields_enabled=False,
        dispersive_enabled=False,
        tfsf_enabled=False,
        has_pec_faces=False,
        boundary_x_low_code=BOUNDARY_PML,
        boundary_x_high_code=BOUNDARY_PML,
        boundary_y_low_code=BOUNDARY_PML,
        boundary_y_high_code=BOUNDARY_PML,
        boundary_z_low_code=BOUNDARY_PML,
        boundary_z_high_code=BOUNDARY_PML,
        inv_dx_e=torch.full((3,), 1.25),
        inv_dy_e=torch.full((4,), 0.75),
        inv_dz_e=torch.full((5,), 1.5),
        inv_dx_h=torch.full((2,), 1.25),
        inv_dy_h=torch.full((3,), 0.75),
        inv_dz_h=torch.full((4,), 1.5),
        chx_decay=torch.full(hx_shape, 0.97),
        chy_decay=torch.full(hy_shape, 0.96),
        chz_decay=torch.full(hz_shape, 0.95),
        chx_curl=torch.full(hx_shape, 0.11),
        chy_curl=torch.full(hy_shape, 0.09),
        chz_curl=torch.full(hz_shape, 0.07),
        cex_decay=torch.full(ex_shape, 0.94),
        cey_decay=torch.full(ey_shape, 0.93),
        cez_decay=torch.full(ez_shape, 0.92),
        cex_curl=torch.full(ex_shape, 0.13),
        cey_curl=torch.full(ey_shape, 0.12),
        cez_curl=torch.full(ez_shape, 0.10),
        eps_Ex=torch.full(ex_shape, 2.0),
        eps_Ey=torch.full(ey_shape, 2.5),
        eps_Ez=torch.full(ez_shape, 3.0),
        cpml_inv_kappa_h_x=torch.tensor([1.07, 1.03], dtype=torch.float32),
        cpml_inv_kappa_h_y=torch.tensor([1.05, 1.02, 1.01], dtype=torch.float32),
        cpml_inv_kappa_h_z=torch.tensor([1.08, 1.04, 1.02, 1.01], dtype=torch.float32),
        cpml_b_h_x=torch.tensor([0.83, 0.81], dtype=torch.float32),
        cpml_b_h_y=torch.tensor([0.85, 0.82, 0.80], dtype=torch.float32),
        cpml_b_h_z=torch.tensor([0.84, 0.82, 0.80, 0.78], dtype=torch.float32),
        cpml_c_h_x=torch.tensor([0.12, 0.10], dtype=torch.float32),
        cpml_c_h_y=torch.tensor([0.11, 0.09, 0.08], dtype=torch.float32),
        cpml_c_h_z=torch.tensor([0.13, 0.11, 0.09, 0.07], dtype=torch.float32),
        cpml_inv_kappa_e_x=torch.tensor([1.06, 1.03, 1.01], dtype=torch.float32),
        cpml_inv_kappa_e_y=torch.tensor([1.05, 1.04, 1.02, 1.01], dtype=torch.float32),
        cpml_inv_kappa_e_z=torch.tensor([1.09, 1.06, 1.03, 1.02, 1.01], dtype=torch.float32),
        cpml_b_e_x=torch.tensor([0.86, 0.83, 0.80], dtype=torch.float32),
        cpml_b_e_y=torch.tensor([0.85, 0.83, 0.81, 0.79], dtype=torch.float32),
        cpml_b_e_z=torch.tensor([0.87, 0.84, 0.82, 0.80, 0.78], dtype=torch.float32),
        cpml_c_e_x=torch.tensor([0.08, 0.07, 0.06], dtype=torch.float32),
        cpml_c_e_y=torch.tensor([0.09, 0.08, 0.07, 0.06], dtype=torch.float32),
        cpml_c_e_z=torch.tensor([0.10, 0.09, 0.08, 0.07, 0.06], dtype=torch.float32),
        _source_time={"kind": "cw", "kind_code": 0, "amplitude": 0.0, "frequency": 1.0, "phase": 0.0, "delay": 0.0},
        source_omega=2.0 * math.pi,
        _compiled_source=None,
        _source_terms=[],
        _electric_source_terms=[],
        _magnetic_source_terms=[],
    )


def _fake_conductive_cpml_reverse_solver(sigma_e=0.35, dt=0.05):
    """CPML reverse solver carrying a static electric conductivity (sigma_e).

    The conductive reverse recomputes the semi-implicit lossy decay/curl pair and
    their eps sensitivities from the frozen ``cex_curl``/``eps_Ex``/``sigma_e_Ex``
    leaves, so this solver adds the conduction attributes on top of the linear
    CPML fake and rebakes ``c*_decay``/``c*_curl`` to be consistent with a nonzero
    ``sigma_e`` (the reverse recovers the eps-independent PML factor from
    ``cex_curl`` regardless, but keeping them consistent keeps the forward replica
    physical)."""
    base = _fake_cpml_reverse_solver()
    values = dict(vars(base))
    values["conductive_enabled"] = True
    values["nonlinear_enabled"] = False
    values["full_aniso_enabled"] = False
    values["magnetic_dispersive_enabled"] = False
    values["dt"] = dt
    for comp in ("Ex", "Ey", "Ez"):
        values[f"sigma_e_{comp}"] = torch.full_like(values[f"eps_{comp}"], float(sigma_e))
    return SimpleNamespace(**values)


def _fake_kerr_cpml_reverse_solver(chi3=0.5, dt=0.05, eps0=1.0):
    """CPML reverse solver carrying an instantaneous Kerr (chi3) nonlinearity.

    The Kerr reverse recomputes the dynamic curl ``(dt / eff) * decay`` with
    ``eff = eps + eps0 * chi3 * |E|^2`` from the frozen ``eps``/``chi3`` leaves and
    the collocated fields, so this solver adds the Kerr attributes on top of the
    linear CPML fake. ``eps0`` is set to 1.0 (an arbitrary coefficient for the fake)
    and ``chi3`` large enough that the nonlinear term is a non-negligible fraction
    of the linear permittivity, so the coefficient reverse (grad_chi3 / the |E|^2
    cotangent) is exercised rather than degenerate."""
    base = _fake_cpml_reverse_solver()
    values = dict(vars(base))
    values["kerr_enabled"] = True
    values["nonlinear_enabled"] = True
    values["nonlinear_general_enabled"] = False
    values["conductive_enabled"] = False
    values["full_aniso_enabled"] = False
    values["magnetic_dispersive_enabled"] = False
    values["dt"] = dt
    values["eps0"] = eps0
    for pos, comp in enumerate(("Ex", "Ey", "Ez")):
        values[f"kerr_chi3_{comp}"] = torch.full_like(values[f"eps_{comp}"], float(chi3) + 0.05 * pos)
    return SimpleNamespace(**values)


def _fake_full_aniso_cpml_reverse_solver():
    """CPML reverse solver carrying a full (off-diagonal) anisotropic epsilon.

    Full anisotropy keeps the linear CPML diagonal update and adds the off-diagonal
    coupling ``E_i += coeff_ij * <curlH_j>`` through six per-edge coefficient
    tensors (``c{ex,ey,ez}_aniso_{...}``). This fake adds those coefficients on top
    of the linear CPML fake with distinct nonzero values, so the off-diagonal
    reverse (its transpose folded into the mid-step H adjoint) is genuinely
    exercised rather than degenerate. The coefficients are static (no material
    gradient channel), so no extra checkpoint state is introduced."""
    base = _fake_cpml_reverse_solver()
    values = dict(vars(base))
    values["full_aniso_enabled"] = True
    values["conductive_enabled"] = False
    values["nonlinear_enabled"] = False
    values["kerr_enabled"] = False
    values["dispersive_enabled"] = False
    values["magnetic_dispersive_enabled"] = False
    values["cex_aniso_y"] = torch.full_like(values["eps_Ex"], 0.031)
    values["cex_aniso_z"] = torch.full_like(values["eps_Ex"], -0.024)
    values["cey_aniso_x"] = torch.full_like(values["eps_Ey"], 0.027)
    values["cey_aniso_z"] = torch.full_like(values["eps_Ey"], 0.019)
    values["cez_aniso_x"] = torch.full_like(values["eps_Ez"], -0.022)
    values["cez_aniso_y"] = torch.full_like(values["eps_Ez"], 0.035)
    return SimpleNamespace(**values)


def _dispersive_cpml_reverse_state_shapes():
    shapes = dict(_cpml_reverse_state_shapes())
    shapes[dispersive_state_name("Ex", "debye", 0, "polarization")] = shapes["Ex"]
    shapes[dispersive_state_name("Ex", "debye", 0, "current")] = shapes["Ex"]
    shapes[dispersive_state_name("Ey", "drude", 0, "current")] = shapes["Ey"]
    shapes[dispersive_state_name("Ez", "lorentz", 0, "polarization")] = shapes["Ez"]
    shapes[dispersive_state_name("Ez", "lorentz", 0, "current")] = shapes["Ez"]
    return shapes


def _fake_dispersive_cpml_reverse_solver():
    base = _fake_cpml_reverse_solver()
    values = dict(vars(base))
    values["dispersive_enabled"] = True
    values["dt"] = 0.05
    values["_dispersive_templates"] = {
        "Ex": {
            "inv_eps": (1.0 / base.eps_Ex).contiguous(),
            "debye": [
                {
                    "decay": 0.83,
                    "drive": torch.full(base.eps_Ex.shape, 0.021, dtype=torch.float32),
                }
            ],
            "drude": [],
            "lorentz": [],
        },
        "Ey": {
            "inv_eps": (1.0 / base.eps_Ey).contiguous(),
            "debye": [],
            "drude": [
                {
                    "decay": 0.79,
                    "drive": torch.full(base.eps_Ey.shape, 0.018, dtype=torch.float32),
                }
            ],
            "lorentz": [],
        },
        "Ez": {
            "inv_eps": (1.0 / base.eps_Ez).contiguous(),
            "debye": [],
            "drude": [],
            "lorentz": [
                {
                    "decay": 0.88,
                    "restoring": 0.14,
                    "drive": torch.full(base.eps_Ez.shape, 0.024, dtype=torch.float32),
                }
            ],
        },
    }
    return SimpleNamespace(**values)


def _dispersive_standard_reverse_state_shapes():
    shapes = _cpml_reverse_state_shapes()
    base = {name: shapes[name] for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
    base[dispersive_state_name("Ex", "debye", 0, "polarization")] = shapes["Ex"]
    base[dispersive_state_name("Ex", "debye", 0, "current")] = shapes["Ex"]
    base[dispersive_state_name("Ey", "drude", 0, "current")] = shapes["Ey"]
    base[dispersive_state_name("Ez", "lorentz", 0, "polarization")] = shapes["Ez"]
    base[dispersive_state_name("Ez", "lorentz", 0, "current")] = shapes["Ez"]
    return base


def _fake_dispersive_standard_reverse_solver():
    base = _fake_standard_reverse_solver()
    values = dict(vars(base))
    values["dispersive_enabled"] = True
    values["dt"] = 0.05
    values["_dispersive_templates"] = {
        "Ex": {
            "inv_eps": (1.0 / base.eps_Ex).contiguous(),
            "debye": [
                {
                    "decay": 0.83,
                    "drive": torch.full(base.eps_Ex.shape, 0.021, dtype=torch.float32),
                }
            ],
            "drude": [],
            "lorentz": [],
        },
        "Ey": {
            "inv_eps": (1.0 / base.eps_Ey).contiguous(),
            "debye": [],
            "drude": [
                {
                    "decay": 0.79,
                    "drive": torch.full(base.eps_Ey.shape, 0.018, dtype=torch.float32),
                }
            ],
            "lorentz": [],
        },
        "Ez": {
            "inv_eps": (1.0 / base.eps_Ez).contiguous(),
            "debye": [],
            "drude": [],
            "lorentz": [
                {
                    "decay": 0.88,
                    "restoring": 0.14,
                    "drive": torch.full(base.eps_Ez.shape, 0.024, dtype=torch.float32),
                }
            ],
        },
    }
    return SimpleNamespace(**values)


def _magnetic_dispersive_standard_reverse_state_shapes():
    shapes = dict(_cpml_reverse_state_shapes())
    base = {
        name: shapes[name]
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }
    base[dispersive_state_name("Hx", "debye", 0, "polarization")] = shapes["Hx"]
    base[dispersive_state_name("Hx", "debye", 0, "current")] = shapes["Hx"]
    base[dispersive_state_name("Hy", "drude", 0, "current")] = shapes["Hy"]
    base[dispersive_state_name("Hz", "lorentz", 0, "polarization")] = shapes["Hz"]
    base[dispersive_state_name("Hz", "lorentz", 0, "current")] = shapes["Hz"]
    return base


def _magnetic_dispersive_templates(hx_shape, hy_shape, hz_shape):
    return {
        "Hx": {
            "inv_mu": torch.full(hx_shape, 1.0 / 1.8, dtype=torch.float32),
            "debye": [
                {
                    "decay": 0.81,
                    "drive": torch.full(hx_shape, 0.017, dtype=torch.float32),
                }
            ],
            "drude": [],
            "lorentz": [],
        },
        "Hy": {
            "inv_mu": torch.full(hy_shape, 1.0 / 2.1, dtype=torch.float32),
            "debye": [],
            "drude": [
                {
                    "decay": 0.77,
                    "drive": torch.full(hy_shape, 0.02, dtype=torch.float32),
                }
            ],
            "lorentz": [],
        },
        "Hz": {
            "inv_mu": torch.full(hz_shape, 1.0 / 1.4, dtype=torch.float32),
            "debye": [],
            "drude": [],
            "lorentz": [
                {
                    "decay": 0.9,
                    "restoring": 0.12,
                    "drive": torch.full(hz_shape, 0.023, dtype=torch.float32),
                }
            ],
        },
    }


def _fake_magnetic_dispersive_standard_reverse_solver():
    base = _fake_standard_reverse_solver()
    values = dict(vars(base))
    values["dispersive_enabled"] = True
    values["magnetic_dispersive_enabled"] = True
    values["dt"] = 0.05
    values["_dispersive_templates"] = {}
    values["_magnetic_dispersive_templates"] = _magnetic_dispersive_templates(
        base.chx_decay.shape, base.chy_decay.shape, base.chz_decay.shape
    )
    return SimpleNamespace(**values)


def _magnetic_dispersive_cpml_reverse_state_shapes():
    shapes = dict(_dispersive_cpml_reverse_state_shapes())
    shapes[dispersive_state_name("Hx", "debye", 0, "polarization")] = shapes["Hx"]
    shapes[dispersive_state_name("Hx", "debye", 0, "current")] = shapes["Hx"]
    shapes[dispersive_state_name("Hy", "drude", 0, "current")] = shapes["Hy"]
    shapes[dispersive_state_name("Hz", "lorentz", 0, "polarization")] = shapes["Hz"]
    shapes[dispersive_state_name("Hz", "lorentz", 0, "current")] = shapes["Hz"]
    return shapes


def _fake_magnetic_dispersive_cpml_reverse_solver():
    base = _fake_dispersive_cpml_reverse_solver()
    values = dict(vars(base))
    values["magnetic_dispersive_enabled"] = True
    values["_magnetic_dispersive_templates"] = _magnetic_dispersive_templates(
        base.chx_decay.shape, base.chy_decay.shape, base.chz_decay.shape
    )
    return SimpleNamespace(**values)


def _tfsf_cpml_reverse_state_shapes():
    shapes = dict(_cpml_reverse_state_shapes())
    shapes["tfsf_aux_electric"] = (5,)
    shapes["tfsf_aux_magnetic"] = (4,)
    return shapes


def _fake_tfsf_cpml_reverse_solver(provider: str = "plane_wave_ref_x_ez"):
    base = _fake_cpml_reverse_solver()
    values = dict(vars(base))
    values["tfsf_enabled"] = True
    values["dt"] = 0.05
    values["_source_time"] = {
        "kind": "gaussian",
        "kind_code": 1,
        "amplitude": 1.3,
        "frequency": 0.75,
        "fwidth": 0.25,
        "phase": 0.1,
        "delay": 0.0,
    }

    if provider == "analytic_profile":
        values["_tfsf_state"] = {
            "provider": "analytic_profile",
            "bounds": ((0.0, 0.1), (0.0, 0.1), (0.0, 0.1)),
            "lower": (1, 1, 1),
            "upper": (2, 2, 2),
            "electric_terms": [
                {
                    "field_name": "Ez",
                    "offsets": (1, 1, 1),
                    "patch": torch.full((1, 1, 1), 0.14, dtype=torch.float32),
                    "grid": None,
                    "phase_real": 1.0,
                    "phase_imag": 0.0,
                    "delay_patch": torch.full((1, 1, 1), 0.01, dtype=torch.float32),
                    "activation_delay_patch": None,
                    "cw_cos_patch": None,
                    "cw_sin_patch": None,
                }
            ],
            "magnetic_terms": [
                {
                    "field_name": "Hy",
                    "offsets": (0, 1, 1),
                    "patch": torch.full((1, 1, 1), -0.09, dtype=torch.float32),
                    "grid": None,
                    "phase_real": 1.0,
                    "phase_imag": 0.0,
                    "delay_patch": torch.full((1, 1, 1), 0.02, dtype=torch.float32),
                    "activation_delay_patch": torch.zeros((1, 1, 1), dtype=torch.float32),
                    "cw_cos_patch": None,
                    "cw_sin_patch": None,
                }
            ],
        }
        return SimpleNamespace(**values)

    auxiliary_grid = SimpleNamespace(
        s_min=-0.2,
        ds=0.1,
        electric_decay=torch.tensor([0.95, 0.93, 0.91, 0.89, 0.87], dtype=torch.float32),
        electric_curl=torch.tensor([0.02, 0.03, 0.04, 0.05, 0.06], dtype=torch.float32),
        magnetic_decay=torch.tensor([0.96, 0.94, 0.92, 0.90], dtype=torch.float32),
        magnetic_curl=torch.tensor([0.03, 0.04, 0.05, 0.06], dtype=torch.float32),
        source_index=2,
        source_time=values["_source_time"],
    )
    if provider == "plane_wave_aux":
        electric_terms = [
            {
                "field_name": "Ez",
                "offsets": (1, 1, 1),
                "grid": None,
                "coeff_patch": torch.full((1, 1, 1), 0.18, dtype=torch.float32),
                "work_patch": torch.zeros((1, 1, 1), dtype=torch.float32),
                "sample_positions": torch.full((1, 1, 1), -0.05, dtype=torch.float32),
                "component_scale": 0.7,
            }
        ]
        magnetic_terms = [
            {
                "field_name": "Hy",
                "offsets": (0, 1, 1),
                "grid": None,
                "coeff_patch": torch.full((1, 1, 1), -0.11, dtype=torch.float32),
                "work_patch": torch.zeros((1, 1, 1), dtype=torch.float32),
                "sample_positions": torch.full((1, 1, 1), 0.05, dtype=torch.float32),
                "component_scale": 1.2,
            }
        ]
    else:
        electric_terms = [
            {
                "field_name": "Ez",
                "offsets": (1, 1, 1),
                "grid": None,
                "coeff_patch": torch.full((1, 1, 1), 0.18, dtype=torch.float32),
                "work_patch": torch.zeros((1, 1, 1), dtype=torch.float32),
                "sample_kind": "magnetic",
                "sample_indices": torch.tensor([1], dtype=torch.int64),
                "scalar_sample_index": 1,
                "sample_view": (1, 1, 1),
                "component_scale": 0.7,
            }
        ]
        magnetic_terms = [
            {
                "field_name": "Hy",
                "offsets": (0, 1, 1),
                "grid": None,
                "coeff_patch": torch.full((1, 1, 1), -0.11, dtype=torch.float32),
                "work_patch": torch.zeros((1, 1, 1), dtype=torch.float32),
                "sample_kind": "electric",
                "sample_indices": torch.tensor([2], dtype=torch.int64),
                "scalar_sample_index": 2,
                "sample_view": (1, 1, 1),
                "component_scale": 1.2,
            }
        ]
    values["_tfsf_state"] = {
        "provider": provider,
        "bounds": ((0.0, 0.1), (0.0, 0.1), (0.0, 0.1)),
        "lower": (1, 1, 1),
        "upper": (2, 2, 2),
        "auxiliary_grid": auxiliary_grid,
        "electric_terms": electric_terms,
        "magnetic_terms": magnetic_terms,
    }
    return SimpleNamespace(**values)


def _bloch_reverse_state_shapes():
    shapes = _cpml_reverse_state_shapes()
    return {
        "Ex": shapes["Ex"],
        "Ey": shapes["Ey"],
        "Ez": shapes["Ez"],
        "Hx": shapes["Hx"],
        "Hy": shapes["Hy"],
        "Hz": shapes["Hz"],
        "Ex_imag": shapes["Ex"],
        "Ey_imag": shapes["Ey"],
        "Ez_imag": shapes["Ez"],
        "Hx_imag": shapes["Hx"],
        "Hy_imag": shapes["Hy"],
        "Hz_imag": shapes["Hz"],
    }


def _fake_bloch_reverse_solver():
    shapes = _bloch_reverse_state_shapes()
    phase_x = 0.37
    phase_y = -0.21
    phase_z = 0.19
    return SimpleNamespace(
        uses_cpml=False,
        complex_fields_enabled=True,
        boundary_kind="bloch",
        dispersive_enabled=False,
        tfsf_enabled=False,
        has_pec_faces=False,
        boundary_x_low_code=BOUNDARY_BLOCH,
        boundary_x_high_code=BOUNDARY_BLOCH,
        boundary_y_low_code=BOUNDARY_BLOCH,
        boundary_y_high_code=BOUNDARY_BLOCH,
        boundary_z_low_code=BOUNDARY_BLOCH,
        boundary_z_high_code=BOUNDARY_BLOCH,
        boundary_phase_cos=torch.tensor(
            [math.cos(phase_x), math.cos(phase_y), math.cos(phase_z)],
            dtype=torch.float32,
        ),
        boundary_phase_sin=torch.tensor(
            [math.sin(phase_x), math.sin(phase_y), math.sin(phase_z)],
            dtype=torch.float32,
        ),
        inv_dx_e=torch.full((3,), 1.25),
        inv_dy_e=torch.full((4,), 0.75),
        inv_dz_e=torch.full((5,), 1.5),
        inv_dx_h=torch.full((2,), 1.25),
        inv_dy_h=torch.full((3,), 0.75),
        inv_dz_h=torch.full((4,), 1.5),
        chx_decay=torch.full(shapes["Hx"], 0.97),
        chy_decay=torch.full(shapes["Hy"], 0.96),
        chz_decay=torch.full(shapes["Hz"], 0.95),
        chx_curl=torch.full(shapes["Hx"], 0.11),
        chy_curl=torch.full(shapes["Hy"], 0.09),
        chz_curl=torch.full(shapes["Hz"], 0.07),
        cex_decay=torch.full(shapes["Ex"], 0.94),
        cey_decay=torch.full(shapes["Ey"], 0.93),
        cez_decay=torch.full(shapes["Ez"], 0.92),
        cex_curl=torch.full(shapes["Ex"], 0.13),
        cey_curl=torch.full(shapes["Ey"], 0.12),
        cez_curl=torch.full(shapes["Ez"], 0.10),
        eps_Ex=torch.full(shapes["Ex"], 2.0),
        eps_Ey=torch.full(shapes["Ey"], 2.5),
        eps_Ez=torch.full(shapes["Ez"], 3.0),
        _source_time={"kind": "cw", "kind_code": 0, "amplitude": 0.0, "frequency": 1.0, "phase": 0.0, "delay": 0.0},
        source_omega=2.0 * math.pi,
        _compiled_source=None,
        _source_terms=[],
        _electric_source_terms=[],
        _magnetic_source_terms=[],
    )


def _move_solver_tensors_to_cuda(solver):
    def _move(value):
        if torch.is_tensor(value):
            return value.to(device="cuda")
        if isinstance(value, SimpleNamespace):
            return SimpleNamespace(**{key: _move(item) for key, item in vars(value).items()})
        if isinstance(value, dict):
            return {key: _move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(_move(item) for item in value)
        return value

    values = {}
    for name, value in vars(solver).items():
        values[name] = _move(value)
    values["device"] = "cuda"
    return SimpleNamespace(**values)


def test_capture_checkpoint_state_freezes_schema_layout():
    state = capture_checkpoint_state(_fake_checkpoint_solver(), step=3)

    assert state.step == 3
    assert state.schema.version == 1
    assert tuple(state.tensors.keys()) == state.schema.state_names
    assert state.schema.field_names == ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    assert state.schema.complex_field_names == (
        "Ex_imag",
        "Ey_imag",
        "Ez_imag",
        "Hx_imag",
        "Hy_imag",
        "Hz_imag",
    )
    assert state.schema.cpml_state_names == (
        "psi_ex_y",
        "psi_ex_z",
        "psi_ey_x",
        "psi_ey_z",
        "psi_ez_x",
        "psi_ez_y",
        "psi_hx_y",
        "psi_hx_z",
        "psi_hy_x",
        "psi_hy_z",
        "psi_hz_x",
        "psi_hz_y",
        "psi_ex_y_imag",
        "psi_ex_z_imag",
        "psi_ey_x_imag",
        "psi_ey_z_imag",
        "psi_ez_x_imag",
        "psi_ez_y_imag",
        "psi_hx_y_imag",
        "psi_hx_z_imag",
        "psi_hy_x_imag",
        "psi_hy_z_imag",
        "psi_hz_x_imag",
        "psi_hz_y_imag",
    )
    assert torch.equal(state.tensors["psi_ex_z_imag"], torch.full((2,), 26.0))
    assert state.schema.tfsf_auxiliary_state_names == (
        "tfsf_aux_electric",
        "tfsf_aux_magnetic",
    )
    # Complex-field (Bloch) solvers carry an imaginary ADE replica per electric
    # pole; it follows all real pole names in the frozen dispersive layout.
    assert state.schema.dispersive_state_names == (
        dispersive_state_name("Ex", "debye", 0, "polarization"),
        dispersive_state_name("Ex", "debye", 0, "current"),
        dispersive_state_name("Ey", "drude", 0, "current"),
        dispersive_state_name("Ez", "lorentz", 0, "polarization"),
        dispersive_state_name("Ez", "lorentz", 0, "current"),
        dispersive_state_name("Ex", "debye", 0, "polarization") + "_imag",
        dispersive_state_name("Ex", "debye", 0, "current") + "_imag",
        dispersive_state_name("Ey", "drude", 0, "current") + "_imag",
        dispersive_state_name("Ez", "lorentz", 0, "polarization") + "_imag",
        dispersive_state_name("Ez", "lorentz", 0, "current") + "_imag",
    )
    assert torch.equal(
        state.tensors[dispersive_state_name("Ez", "lorentz", 0, "current") + "_imag"],
        torch.full((1,), 105.0),
    )



def _build_material_pullback_scene(device: str):
    density_a = torch.tensor(
        [
            [[-0.4, 0.2], [0.7, 1.1]],
            [[0.0, 0.8], [1.3, -0.2]],
        ],
        device=device,
        dtype=torch.float32,
    )
    density_b = torch.tensor(
        [
            [[0.1, 0.6]],
            [[0.9, 1.2]],
        ],
        device=device,
        dtype=torch.float32,
    )
    scene = mw.Scene(
        domain=mw.Domain(bounds=((0.0, 0.5), (0.0, 0.4), (0.0, 0.4))),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.none(),
        device=device,
    )
    scene.add_material_region(
        mw.MaterialRegion(
            name="region_a",
            geometry=mw.Box(position=(0.2, 0.15, 0.15), size=(0.3, 0.3, 0.3)),
            density=density_a,
            bounds=(-0.5, 1.5),
            eps_bounds=(1.0, 4.5),
            mu_bounds=(1.0, 1.0),
            filter_radius=0.12,
            projection_beta=5.0,
        )
    )
    scene.add_material_region(
        mw.MaterialRegion(
            name="region_b",
            geometry=mw.Box(position=(0.3, 0.15, 0.15), size=(0.2, 0.2, 0.2)),
            density=density_b,
            eps_bounds=(1.5, 5.0),
            mu_bounds=(1.0, 1.0),
            filter_radius=0.16,
            projection_beta=4.0,
        )
    )
    return scene, (density_a, density_b)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")),
    ],
)
def test_material_pullback_explicit_matches_autograd_baseline(device):
    torch.manual_seed(101)
    scene, density_tensors = _build_material_pullback_scene(device)
    prepared_scene = prepare_scene(scene)
    grad_eps_ex = torch.randn((prepared_scene.Nx - 1, prepared_scene.Ny, prepared_scene.Nz), device=device, dtype=torch.float32)
    grad_eps_ey = torch.randn((prepared_scene.Nx, prepared_scene.Ny - 1, prepared_scene.Nz), device=device, dtype=torch.float32)
    grad_eps_ez = torch.randn((prepared_scene.Nx, prepared_scene.Ny, prepared_scene.Nz - 1), device=device, dtype=torch.float32)

    actual = pullback_density_gradients(
        scene,
        density_tensors=density_tensors,
        trainable_region_indices=(0, 1),
        grad_eps_ex=grad_eps_ex,
        grad_eps_ey=grad_eps_ey,
        grad_eps_ez=grad_eps_ez,
        eps0=8.8541878128e-12,
    )
    expected = adjoint_baselines.material_pullback_autograd(
        scene,
        density_tensors=density_tensors,
        trainable_region_indices=(0, 1),
        grad_eps_ex=grad_eps_ex,
        grad_eps_ey=grad_eps_ey,
        grad_eps_ez=grad_eps_ez,
        eps0=8.8541878128e-12,
    )

    assert len(actual) == len(expected) == 2
    for actual_grad, expected_grad in zip(actual, expected):
        assert torch.allclose(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)


class _DensityPointScene(mw.SceneModule):
    def __init__(self, init=0.0):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.full((1, 1, 1), float(init), device="cuda"))

    def to_scene(self):
        density = torch.sigmoid(self.logits)
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.24, 0.24), (-0.24, 0.24), (-0.24, 0.24))),
            grid=mw.GridSpec.uniform(0.12),
            boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
            device="cuda",
        )
        scene.add_material_region(
            mw.MaterialRegion(
                name="design",
                geometry=mw.Box(position=(0.0, 0.0, 0.06), size=(0.12, 0.12, 0.12)),
                density=density,
                eps_bounds=(1.0, 6.0),
                mu_bounds=(1.0, 1.0),
            )
        )
        scene.add_source(
            mw.PointDipole(
                position=(0.0, 0.0, -0.06),
                polarization="Ez",
                width=0.04,
                source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, amplitude=50.0),
            )
        )
        scene.add_monitor(mw.PointMonitor("probe", (0.0, 0.0, 0.06), fields=("Ez",)))
        return scene


class _AnalyticBoxPointScene(mw.SceneModule):
    def __init__(self, init_x=0.10):
        super().__init__()
        self.box_x = torch.nn.Parameter(torch.tensor(float(init_x), device="cuda"))

    def to_scene(self):
        position = torch.stack(
            (
                self.box_x,
                self.box_x.new_tensor(0.0),
                self.box_x.new_tensor(0.06),
            )
        )
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.24, 0.24), (-0.24, 0.24), (-0.24, 0.24))),
            grid=mw.GridSpec.uniform(0.12),
            boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
            device="cuda",
            subpixel_samples=5,
        )
        scene.add_structure(
            mw.Structure(
                name="design_box",
                geometry=mw.Box(position=position, size=(0.18, 0.18, 0.18)),
                material=mw.Material(eps_r=20.0),
            )
        )
        scene.add_source(
            mw.PointDipole(
                position=(0.0, 0.0, -0.06),
                polarization="Ez",
                width=0.04,
                source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, amplitude=5000.0),
            )
        )
        scene.add_monitor(mw.PointMonitor("probe", (0.0, 0.0, 0.06), fields=("Ez",)))
        return scene


class _DensitySurfaceSourceScene(mw.SceneModule):
    def __init__(self, source_kind: str, init=0.0):
        super().__init__()
        self.source_kind = source_kind
        self.logits = torch.nn.Parameter(torch.full((1, 1, 1), float(init), device="cuda"))

    def _build_source(self):
        if self.source_kind == "plane_wave":
            return mw.PlaneWave(
                direction=(1.0, 0.0, 0.0),
                polarization=(0.0, 0.0, 1.0),
                source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, amplitude=40.0),
                name="pw",
            )
        if self.source_kind == "gaussian_beam":
            return mw.GaussianBeam(
                direction=(1.0, 0.0, 0.0),
                polarization=(0.0, 0.0, 1.0),
                beam_waist=0.18,
                focus=(-0.06, 0.0, 0.0),
                source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, amplitude=40.0),
                name="beam",
            )
        raise ValueError(f"Unsupported source_kind {self.source_kind!r}.")

    def to_scene(self):
        density = torch.sigmoid(self.logits)
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.48, 0.48), (-0.48, 0.48), (-0.48, 0.48))),
            # Resolve the 1 GHz soft surface source above the positive discrete
            # Poynting-power limit; 0.12 m is too coarse for this wave in 3D.
            grid=mw.GridSpec.uniform(0.08),
            boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
            device="cuda",
        )
        scene.add_material_region(
            mw.MaterialRegion(
                name="design",
                # The x faces are offset half a cell from the grid nodes so
                # the voxel window is off the node-aligned knife edge and
                # matches the placement this test was calibrated against.
                geometry=mw.Box(position=(-0.06, 0.0, 0.0), size=(0.24, 0.12, 0.12)),
                density=density,
                eps_bounds=(1.0, 6.0),
                mu_bounds=(1.0, 1.0),
            )
        )
        scene.add_source(self._build_source())
        scene.add_monitor(mw.PointMonitor("probe", (0.12, 0.0, 0.0), fields=("Ez",)))
        return scene


class _DensityFluxScene(mw.SceneModule):
    def __init__(self, init=0.0):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.full((1, 1, 1), float(init), device="cuda"))

    def to_scene(self):
        density = torch.sigmoid(self.logits)
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.24, 0.24), (-0.24, 0.24), (-0.24, 0.24))),
            grid=mw.GridSpec.uniform(0.12),
            boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
            device="cuda",
        )
        scene.add_material_region(
            mw.MaterialRegion(
                name="design",
                geometry=mw.Box(position=(0.0, 0.0, 0.06), size=(0.12, 0.12, 0.12)),
                density=density,
                eps_bounds=(1.0, 6.0),
                mu_bounds=(1.0, 1.0),
            )
        )
        scene.add_source(
            mw.PointDipole(
                position=(0.0, 0.0, -0.06),
                polarization="Ez",
                width=0.04,
                source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, amplitude=50.0),
            )
        )
        scene.add_monitor(mw.FluxMonitor("flux_probe", axis="z", position=0.06, normal_direction="+"))
        return scene


class _DensityDispersiveScene(mw.SceneModule):
    def __init__(self, medium_kind: str, init=0.0):
        super().__init__()
        self.material_kind = medium_kind
        self.logits = torch.nn.Parameter(torch.full((1, 1, 1), float(init), device="cuda"))

    def _build_medium(self):
        if self.material_kind == "debye":
            return mw.Material.debye(eps_inf=1.5, delta_eps=2.0, tau=2.0e-10)
        if self.material_kind == "drude":
            return mw.Material.drude(eps_inf=1.5, plasma_frequency=2.0e9, gamma=0.3e9)
        if self.material_kind == "lorentz":
            return mw.Material.lorentz(
                eps_inf=1.5,
                delta_eps=1.2,
                resonance_frequency=1.0e9,
                gamma=0.15e9,
            )
        raise ValueError(f"Unsupported medium_kind {self.material_kind!r}.")

    def to_scene(self):
        density = torch.sigmoid(self.logits)
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.24, 0.24), (-0.24, 0.24), (-0.24, 0.24))),
            grid=mw.GridSpec.uniform(0.12),
            boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
            device="cuda",
        )
        scene.add_structure(
            mw.Structure(
                name=f"{self.material_kind}_host",
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.24, 0.24, 0.24)),
                material=self._build_medium(),
            )
        )
        scene.add_material_region(
            mw.MaterialRegion(
                name="design",
                geometry=mw.Box(position=(0.0, 0.0, 0.06), size=(0.12, 0.12, 0.12)),
                density=density,
                eps_bounds=(1.0, 6.0),
                mu_bounds=(1.0, 1.0),
            )
        )
        scene.add_source(
            mw.PointDipole(
                position=(0.0, 0.0, -0.06),
                polarization="Ez",
                width=0.04,
                source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, amplitude=50.0),
            )
        )
        scene.add_monitor(mw.PointMonitor("probe", (0.0, 0.0, 0.06), fields=("Ez",)))
        return scene


class _DensityTFSFScene(mw.SceneModule):
    def __init__(self, source_kind: str, init=0.0):
        super().__init__()
        self.source_kind = source_kind
        self.logits = torch.nn.Parameter(torch.full((1, 1, 1), float(init), device="cuda"))

    def _build_source(self):
        injection = mw.TFSF(bounds=((-0.36, 0.36), (-0.36, 0.36), (-0.36, 0.36)))
        common = {
            "direction": (1.0, 0.0, 0.0),
            "polarization": (0.0, 0.0, 1.0),
            "source_time": mw.GaussianPulse(frequency=1.0e9, fwidth=0.25e9, amplitude=40.0),
            "injection": injection,
        }
        if self.source_kind == "plane_wave":
            return mw.PlaneWave(name="tfsf_pw", **common)
        if self.source_kind == "gaussian_beam":
            return mw.GaussianBeam(
                name="tfsf_beam",
                direction=(1.0, 0.1, 0.05),
                polarization=(0.0, 0.4472135955, -0.8944271910),
                beam_waist=0.18,
                focus=(-0.12, 0.0, 0.0),
                source_time=common["source_time"],
                injection=injection,
            )
        raise ValueError(f"Unsupported source_kind {self.source_kind!r}.")

    def to_scene(self):
        density = torch.sigmoid(self.logits)
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-1.2, 1.2), (-1.2, 1.2), (-1.2, 1.2))),
            grid=mw.GridSpec.uniform(0.12),
            boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
            device="cuda",
        )
        scene.add_material_region(
            mw.MaterialRegion(
                name="design",
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.12, 0.12, 0.12)),
                density=density,
                eps_bounds=(1.0, 6.0),
                mu_bounds=(1.0, 1.0),
            )
        )
        scene.add_source(self._build_source())
        scene.add_monitor(mw.PointMonitor("probe", (0.12, 0.0, 0.0), fields=("Ez",)))
        return scene


class _DensityBlochScene(mw.SceneModule):
    def __init__(self, init=0.0):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.full((1, 1, 1), float(init), device="cuda"))

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
                geometry=mw.Box(position=(0.15, 0.0, 0.0), size=(0.15, 0.15, 0.15)),
                density=density,
                eps_bounds=(1.0, 6.0),
                mu_bounds=(1.0, 1.0),
            )
        )
        scene.add_source(
            mw.PointDipole(
                position=(-0.55, 0.0, 0.0),
                polarization="Ez",
                width=0.12,
                source_time=mw.GaussianPulse(frequency=1.0e9, fwidth=0.25e9, amplitude=25.0),
            )
        )
        scene.add_monitor(mw.PointMonitor("probe", (0.15, 0.0, 0.0), fields=("Ez",)))
        return scene


class _DensityBlochYPmlScene(mw.SceneModule):
    """x/z Bloch + y PML: the generalized single-PML-axis mixed Bloch+CPML update
    (absorbing axis != z) exercising the general reverse replay."""

    def __init__(self, init=0.0):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.full((1, 1, 1), float(init), device="cuda"))

    def to_scene(self):
        density = torch.sigmoid(self.logits)
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.6, 0.6), (-0.75, 0.75), (-0.6, 0.6))),
            grid=mw.GridSpec.uniform(0.15),
            boundary=mw.BoundarySpec.faces(
                default="pml",
                num_layers=2,
                strength=1.0,
                x="bloch",
                y="pml",
                z="bloch",
                bloch_wavevector=(math.pi / 1.2, 0.0, math.pi / 2.4),
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
            mw.PointDipole(
                position=(0.0, -0.55, 0.0),
                polarization="Ez",
                width=0.12,
                source_time=mw.GaussianPulse(frequency=1.0e9, fwidth=0.25e9, amplitude=25.0),
            )
        )
        scene.add_monitor(mw.PointMonitor("probe", (0.0, 0.0, 0.0), fields=("Ez",)))
        return scene


def _roll_normal_from_z(triple, axis):
    """Cyclically rotate an (x, y, z)-ordered triple so the z entry (the base
    grating normal) lands on ``axis``. The permutation is a proper rotation."""
    x, y, z = triple
    return {"z": (x, y, z), "x": (z, x, y), "y": (y, z, x)}[axis]


# Under the same z->axis rotation, the base "Ex" probe component maps to the new
# component occupying the old x slot.
_GRATING_PROBE_COMPONENT = {"z": "Ex", "x": "Ey", "y": "Ez"}


class _DensityGratingTFSFScene(mw.SceneModule):
    def __init__(self, init=0.0, axis="z"):
        super().__init__()
        self.axis = axis
        self.logits = torch.nn.Parameter(torch.full((1, 1, 1), float(init), device="cuda"))

    def to_scene(self):
        axis = self.axis
        density = torch.sigmoid(self.logits)
        direction = _roll_normal_from_z((0.2, 0.1, 0.9746794344808963), axis)
        polarization = _roll_normal_from_z((1.0, 0.0, -0.20519567041703082), axis)
        domain_bounds = _roll_normal_from_z(((-0.45, 0.45), (-0.45, 0.45), (-0.75, 0.75)), axis)
        probe_position = _roll_normal_from_z((0.15, 0.0, 0.0), axis)
        boundary_kinds = {other: "bloch" for other in "xyz"}
        boundary_kinds[axis] = "pml"
        scene = mw.Scene(
            domain=mw.Domain(bounds=domain_bounds),
            grid=mw.GridSpec.uniform(0.15),
            boundary=mw.BoundarySpec.faces(
                default="pml",
                num_layers=2,
                strength=1.0,
                bloch_wavevector="auto",
                **boundary_kinds,
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
                direction=direction,
                polarization=polarization,
                source_time=mw.CW(frequency=1.0e9, amplitude=40.0),
                injection=mw.TFSF.slab(axis=axis, bounds=(-0.30, 0.30)),
            )
        )
        scene.add_monitor(
            mw.PointMonitor("probe", probe_position, fields=(_GRATING_PROBE_COMPONENT[axis],))
        )
        return scene


class _UnsupportedSceneParam(mw.SceneModule):
    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(0.05, device="cuda"))

    def to_scene(self):
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.24, 0.24), (-0.24, 0.24), (-0.24, 0.24))),
            grid=mw.GridSpec.uniform(0.12),
            boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
            device="cuda",
        )
        scene.add_source(
            mw.PointDipole(
                position=(0.0, 0.0, float(self.offset.detach().item())),
                polarization="Ez",
                width=0.04,
                source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, amplitude=20.0),
            )
        )
        scene.add_monitor(mw.PointMonitor("probe", (0.0, 0.0, 0.0), fields=("Ez",)))
        return scene


def _build_direct_geometry_scene(position: torch.Tensor):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.24, 0.24), (-0.24, 0.24), (-0.24, 0.24))),
        grid=mw.GridSpec.uniform(0.12),
        boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
        device="cuda",
        subpixel_samples=5,
    )
    scene.add_structure(
        mw.Structure(
            name="direct_box",
            geometry=mw.Box(position=position, size=(0.18, 0.18, 0.18)),
            material=mw.Material(eps_r=20.0),
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, -0.06),
            polarization="Ez",
            width=0.04,
            source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, amplitude=5000.0),
        )
    )
    scene.add_monitor(mw.PointMonitor("probe", (0.0, 0.0, 0.06), fields=("Ez",)))
    return scene


def _loss_from_point_probe(model):
    result = _build_simulation(model).run()
    data = result.monitor("probe")["data"]
    loss = torch.abs(data) ** 2
    return result, data, loss


def _geometry_target_probe(target_x: float):
    _, data, _ = _loss_from_point_probe(_AnalyticBoxPointScene(init_x=target_x).cuda())
    return data.detach()


def _loss_from_point_probe_target(model, target_data: torch.Tensor):
    result = _build_simulation(model).run()
    data = result.monitor("probe")["data"]
    loss = torch.abs(data - target_data) ** 2
    return result, data, loss


def _loss_from_surface_probe(model):
    result = _build_simulation(model, time_steps=48).run()
    data = result.monitor("probe")["data"]
    loss = torch.abs(data) ** 2
    return result, data, loss


def _loss_from_dispersive_probe(model):
    result = _build_simulation(model, time_steps=36).run()
    data = result.monitor("probe")["data"]
    loss = torch.abs(data) ** 2
    return result, data, loss


def _loss_from_tfsf_probe(model):
    result = _build_simulation(model, time_steps=48).run()
    data = result.monitor("probe")["data"]
    loss = torch.abs(data) ** 2
    return result, data, loss


def _loss_from_bloch_probe(model):
    result = _build_simulation(model, time_steps=40).run()
    data = result.monitor("probe")["data"]
    loss = torch.abs(data) ** 2
    return result, data, loss


def _loss_from_grating_tfsf_probe(model):
    result = _build_simulation(model, time_steps=48).run()
    data = result.monitor("probe")["data"]
    loss = torch.abs(data) ** 2
    return result, data, loss


def _loss_from_bloch_y_pml_probe(model):
    result = _build_simulation(model, time_steps=40).run()
    data = result.monitor("probe")["data"]
    loss = torch.abs(data) ** 2
    return result, data, loss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_gradient_bridge_backpropagates_to_scene_module_density_parameters():
    model = _DensityPointScene().cuda()

    result, data, loss = _loss_from_point_probe(model)
    assert torch.is_tensor(data)
    assert data.is_cuda
    assert data.requires_grad

    loss.backward()

    assert model.logits.grad is not None
    assert model.logits.grad.shape == model.logits.shape
    assert torch.isfinite(model.logits.grad).all()
    assert result.monitor("probe")["data"].grad_fn is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_gradient_bridge_backpropagates_to_direct_scene_geometry_parameters():
    position = torch.nn.Parameter(torch.tensor([0.10, 0.0, 0.06], device="cuda"))
    target_data = _geometry_target_probe(-0.08)

    result, data, loss = _loss_from_point_probe_target(_build_direct_geometry_scene(position), target_data)
    assert torch.is_tensor(data)
    assert data.is_cuda
    assert data.requires_grad

    loss.backward()

    assert position.grad is not None
    assert position.grad.shape == position.shape
    assert torch.isfinite(position.grad).all()
    assert abs(float(position.grad[0].item())) > 1.0e-10
    assert result.monitor("probe")["data"].grad_fn is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_gradient_bridge_backpropagates_to_scene_module_geometry_parameters():
    model = _AnalyticBoxPointScene(init_x=0.10).cuda()
    target_data = _geometry_target_probe(-0.08)

    result, data, loss = _loss_from_point_probe_target(model, target_data)
    assert torch.is_tensor(data)
    assert data.is_cuda
    assert data.requires_grad

    loss.backward()

    assert model.box_x.grad is not None
    assert torch.isfinite(model.box_x.grad).all()
    assert abs(model.box_x.grad.item()) > 1.0e-10
    assert result.monitor("probe")["data"].grad_fn is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_gradient_bridge_matches_central_difference_for_scene_module_geometry_position():
    model = _AnalyticBoxPointScene(init_x=0.10).cuda()
    target_data = _geometry_target_probe(-0.08)

    _, _, loss = _loss_from_point_probe_target(model, target_data)
    loss.backward()
    backward_grad = model.box_x.grad.detach().clone()
    assert torch.isfinite(backward_grad)

    delta = 1.0e-2
    with torch.no_grad():
        base = model.box_x.detach().clone()
        model.box_x.copy_(base + delta)
    _, _, loss_plus = _loss_from_point_probe_target(model, target_data)
    with torch.no_grad():
        model.box_x.copy_(base - delta)
    _, _, loss_minus = _loss_from_point_probe_target(model, target_data)
    with torch.no_grad():
        model.box_x.copy_(base)

    finite_difference = (loss_plus - loss_minus) / (2.0 * delta)
    assert torch.allclose(
        backward_grad,
        torch.full_like(backward_grad, finite_difference.item()),
        rtol=6e-1,
        atol=1e-10,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_gradient_bridge_optimizer_step_reduces_geometry_target_loss():
    target_x = -0.08
    target_data = _geometry_target_probe(target_x)

    model = _AnalyticBoxPointScene(init_x=0.16).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5.0e-2)
    initial_distance = abs(float(model.box_x.detach().item()) - target_x)

    initial_loss_value = None
    final_loss_value = None
    for _ in range(6):
        optimizer.zero_grad()
        _, _data, loss = _loss_from_point_probe_target(model, target_data)
        if initial_loss_value is None:
            initial_loss_value = float(loss.item())
        loss.backward()
        optimizer.step()
        final_loss_value = float(loss.item())

    assert initial_loss_value is not None
    assert final_loss_value is not None
    assert final_loss_value < initial_loss_value
    assert abs(float(model.box_x.detach().item()) - target_x) < initial_distance


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_gradient_bridge_matches_central_difference_for_scene_module_logit():
    model = _DensityPointScene(init=0.0).cuda()

    _, _, loss = _loss_from_point_probe(model)
    loss.backward()
    backward_grad = model.logits.grad.detach().clone()
    assert torch.isfinite(backward_grad).all()

    delta = 1.0e-2
    with torch.no_grad():
        base = model.logits.detach().clone()
        model.logits.copy_(base + delta)
    _, _, loss_plus = _loss_from_point_probe(model)
    with torch.no_grad():
        model.logits.copy_(base - delta)
    _, _, loss_minus = _loss_from_point_probe(model)
    with torch.no_grad():
        model.logits.copy_(base)

    finite_difference = (loss_plus - loss_minus) / (2.0 * delta)
    assert torch.allclose(
        backward_grad,
        torch.full_like(backward_grad, finite_difference.item()),
        rtol=5e-3,
        atol=1e-15,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
@pytest.mark.parametrize("source_kind", ["plane_wave", "gaussian_beam"])
def test_fdtd_gradient_bridge_matches_central_difference_for_soft_surface_source_pullback(source_kind):
    model = _DensitySurfaceSourceScene(source_kind=source_kind, init=0.0).cuda()

    _, data, loss = _loss_from_surface_probe(model)
    assert torch.is_tensor(data)
    assert data.is_cuda
    assert data.requires_grad

    loss.backward()
    backward_grad = model.logits.grad.detach().clone()

    delta = 1.0e-2
    with torch.no_grad():
        base = model.logits.detach().clone()
        model.logits.copy_(base + delta)
    _, _, loss_plus = _loss_from_surface_probe(model)
    with torch.no_grad():
        model.logits.copy_(base - delta)
    _, _, loss_minus = _loss_from_surface_probe(model)
    with torch.no_grad():
        model.logits.copy_(base)

    finite_difference = (loss_plus - loss_minus) / (2.0 * delta)
    assert torch.allclose(
        backward_grad,
        torch.full_like(backward_grad, finite_difference.item()),
        rtol=5e-2,
        atol=1e-15,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
@pytest.mark.parametrize("medium_kind", ["debye", "drude", "lorentz"])
def test_fdtd_gradient_bridge_matches_central_difference_for_dispersive_materials(medium_kind):
    model = _DensityDispersiveScene(medium_kind=medium_kind, init=0.0).cuda()

    _, data, loss = _loss_from_dispersive_probe(model)
    assert torch.is_tensor(data)
    assert data.is_cuda
    assert data.requires_grad

    loss.backward()
    backward_grad = model.logits.grad.detach().clone()
    assert torch.isfinite(backward_grad).all()

    delta = 1.0e-2
    with torch.no_grad():
        base = model.logits.detach().clone()
        model.logits.copy_(base + delta)
    _, _, loss_plus = _loss_from_dispersive_probe(model)
    with torch.no_grad():
        model.logits.copy_(base - delta)
    _, _, loss_minus = _loss_from_dispersive_probe(model)
    with torch.no_grad():
        model.logits.copy_(base)

    finite_difference = (loss_plus - loss_minus) / (2.0 * delta)
    assert torch.allclose(
        backward_grad,
        torch.full_like(backward_grad, finite_difference.item()),
        rtol=6e-2,
        atol=1e-15,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
@pytest.mark.parametrize("source_kind", ["plane_wave", "gaussian_beam"])
def test_fdtd_gradient_bridge_matches_central_difference_for_tfsf_sources(source_kind):
    model = _DensityTFSFScene(source_kind=source_kind, init=0.0).cuda()

    _, data, loss = _loss_from_tfsf_probe(model)
    assert torch.is_tensor(data)
    assert data.is_cuda
    assert data.requires_grad

    loss.backward()
    backward_grad = model.logits.grad.detach().clone()
    assert torch.isfinite(backward_grad).all()

    delta = 1.0e-2
    with torch.no_grad():
        base = model.logits.detach().clone()
        model.logits.copy_(base + delta)
    _, _, loss_plus = _loss_from_tfsf_probe(model)
    with torch.no_grad():
        model.logits.copy_(base - delta)
    _, _, loss_minus = _loss_from_tfsf_probe(model)
    with torch.no_grad():
        model.logits.copy_(base)

    finite_difference = (loss_plus - loss_minus) / (2.0 * delta)
    assert torch.allclose(
        backward_grad,
        torch.full_like(backward_grad, finite_difference.item()),
        rtol=8e-2,
        atol=1e-15,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_gradient_bridge_matches_central_difference_for_bloch_boundaries():
    model = _DensityBlochScene(init=0.0).cuda()

    _, data, loss = _loss_from_bloch_probe(model)
    assert torch.is_tensor(data)
    assert data.is_cuda
    assert data.requires_grad
    assert torch.is_complex(data)

    loss.backward()
    backward_grad = model.logits.grad.detach().clone()
    assert torch.isfinite(backward_grad).all()

    delta = 1.0e-2
    with torch.no_grad():
        base = model.logits.detach().clone()
        model.logits.copy_(base + delta)
    _, _, loss_plus = _loss_from_bloch_probe(model)
    with torch.no_grad():
        model.logits.copy_(base - delta)
    _, _, loss_minus = _loss_from_bloch_probe(model)
    with torch.no_grad():
        model.logits.copy_(base)

    finite_difference = (loss_plus - loss_minus) / (2.0 * delta)
    assert torch.allclose(
        backward_grad,
        torch.full_like(backward_grad, finite_difference.item()),
        rtol=6e-2,
        atol=1e-15,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_gradient_bridge_matches_central_difference_for_bloch_y_pml_boundaries():
    # Adjoint of the generalized single-PML-axis (y) mixed Bloch+CPML update: the
    # reverse replay runs _update_general_bloch_cpml_electric_fields.
    model = _DensityBlochYPmlScene(init=0.0).cuda()

    _, data, loss = _loss_from_bloch_y_pml_probe(model)
    assert torch.is_tensor(data)
    assert data.is_cuda
    assert data.requires_grad
    assert torch.is_complex(data)

    loss.backward()
    backward_grad = model.logits.grad.detach().clone()
    assert torch.isfinite(backward_grad).all()

    delta = 1.0e-2
    with torch.no_grad():
        base = model.logits.detach().clone()
        model.logits.copy_(base + delta)
    _, _, loss_plus = _loss_from_bloch_y_pml_probe(model)
    with torch.no_grad():
        model.logits.copy_(base - delta)
    _, _, loss_minus = _loss_from_bloch_y_pml_probe(model)
    with torch.no_grad():
        model.logits.copy_(base)

    finite_difference = (loss_plus - loss_minus) / (2.0 * delta)
    assert torch.allclose(
        backward_grad,
        torch.full_like(backward_grad, finite_difference.item()),
        rtol=8e-2,
        atol=1e-15,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_gradient_bridge_matches_central_difference_for_grating_tfsf():
    model = _DensityGratingTFSFScene(init=0.0).cuda()

    _, data, loss = _loss_from_grating_tfsf_probe(model)
    assert torch.is_tensor(data)
    assert data.is_cuda
    assert data.requires_grad
    assert torch.is_complex(data)

    loss.backward()
    backward_grad = model.logits.grad.detach().clone()
    assert torch.isfinite(backward_grad).all()

    delta = 1.0e-2
    with torch.no_grad():
        base = model.logits.detach().clone()
        model.logits.copy_(base + delta)
    _, _, loss_plus = _loss_from_grating_tfsf_probe(model)
    with torch.no_grad():
        model.logits.copy_(base - delta)
    _, _, loss_minus = _loss_from_grating_tfsf_probe(model)
    with torch.no_grad():
        model.logits.copy_(base)

    finite_difference = (loss_plus - loss_minus) / (2.0 * delta)
    assert torch.allclose(
        backward_grad,
        torch.full_like(backward_grad, finite_difference.item()),
        rtol=1.5e-1,
        atol=1e-15,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_gradient_bridge_matches_central_difference_for_grating_tfsf_x_axis():
    # The grating slab normal axis is generalized beyond z: the adjoint replays the
    # single-PML-axis (here x) mixed Bloch/CPML update and the grating surface
    # currents, and the density gradient must still match a central difference.
    model = _DensityGratingTFSFScene(init=0.0, axis="x").cuda()

    _, data, loss = _loss_from_grating_tfsf_probe(model)
    assert torch.is_complex(data)

    loss.backward()
    backward_grad = model.logits.grad.detach().clone()
    assert torch.isfinite(backward_grad).all()

    delta = 1.0e-2
    with torch.no_grad():
        base = model.logits.detach().clone()
        model.logits.copy_(base + delta)
    _, _, loss_plus = _loss_from_grating_tfsf_probe(model)
    with torch.no_grad():
        model.logits.copy_(base - delta)
    _, _, loss_minus = _loss_from_grating_tfsf_probe(model)
    with torch.no_grad():
        model.logits.copy_(base)

    finite_difference = (loss_plus - loss_minus) / (2.0 * delta)
    assert torch.allclose(
        backward_grad,
        torch.full_like(backward_grad, finite_difference.item()),
        rtol=1.5e-1,
        atol=1e-15,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_gradient_bridge_optimizer_step_reduces_loss():
    model = _DensityPointScene(init=0.0).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e13)

    optimizer.zero_grad()
    _, _, initial_loss = _loss_from_point_probe(model)
    initial_value = float(initial_loss.item())
    initial_loss.backward()
    optimizer.step()

    _, _, final_loss = _loss_from_point_probe(model)
    assert float(final_loss.item()) < initial_value


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_gradient_bridge_backward_profile_reports_expected_sections():
    model = _DensityPointScene(init=0.0).cuda()
    simulation = _build_simulation(model)
    bridge = _FDTDGradientBridge(simulation)
    bridge.forward(tuple(bridge.material_inputs))

    profile = bridge.backward_profile()

    assert profile is not None
    assert profile["timer"] in {"cuda_event", "perf_counter"}
    assert profile["steps"] == bridge._time_steps
    assert profile["segments"] == len(bridge._last_checkpoints)
    assert profile["checkpoint_stride"] == 4
    assert profile["total_wall_ms"] >= 0.0
    for name in (
        "seed_build",
        "segment_replay",
        "seed_injection",
        "state_clone",
        "step_forward",
        "step_vjp",
        "material_pullback",
    ):
        assert name in profile["sections_ms"]
        assert name in profile["counts"]
        assert name in profile["mean_section_ms"]
    assert profile["counts"]["segment_replay"] == len(bridge._last_checkpoints)
    assert profile["counts"]["seed_injection"] == bridge._time_steps
    assert profile["counts"]["state_clone"] == bridge._time_steps
    assert profile["counts"]["step_forward"] == bridge._time_steps
    assert profile["counts"]["step_vjp"] == bridge._time_steps
    assert profile["sections_ms"]["step_forward"] >= 0.0
    assert profile["sections_ms"]["step_vjp"] >= 0.0
    assert profile["material_pullback_backend"] == "autograd_material_graph"
    assert profile["seed_injection_backend"] == "device_batched"
    assert profile["seed_batch_counts"]["point"] > 0
    assert profile["reverse_backend_counts"].get(adjoint_baselines.expected_cpml_reverse_backend(), 0) == bridge._time_steps


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
@pytest.mark.parametrize("source_kind", ["plane_wave", "gaussian_beam"])
def test_fdtd_gradient_bridge_backward_profile_uses_native_cpml_for_soft_surface_sources(source_kind):
    model = _DensitySurfaceSourceScene(source_kind=source_kind, init=0.0).cuda()
    simulation = _build_simulation(model, time_steps=24)
    bridge = _FDTDGradientBridge(simulation)
    bridge.forward(tuple(bridge.material_inputs))

    profile = bridge.backward_profile()

    assert profile["seed_injection_backend"] == "device_batched"
    assert profile["seed_batch_counts"]["point"] > 0
    assert profile["reverse_backend_counts"].get(adjoint_baselines.expected_cpml_reverse_backend(), 0) == bridge._time_steps


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
@pytest.mark.parametrize("source_kind", ["plane_wave", "gaussian_beam"])
def test_fdtd_gradient_bridge_backward_profile_uses_native_tfsf_for_tfsf_sources(source_kind):
    model = _DensityTFSFScene(source_kind=source_kind, init=0.0).cuda()
    simulation = _build_simulation(model, time_steps=24)
    bridge = _FDTDGradientBridge(simulation)
    bridge.forward(tuple(bridge.material_inputs))

    profile = bridge.backward_profile()

    assert profile["seed_injection_backend"] == "device_batched"
    assert profile["seed_batch_counts"]["point"] > 0
    assert (
        profile["reverse_backend_counts"].get(adjoint_baselines.expected_tfsf_reverse_backend(), 0)
        == bridge._time_steps
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_gradient_bridge_backward_profile_uses_native_bloch_for_bloch_boundaries():
    model = _DensityBlochScene(init=0.0).cuda()
    simulation = _build_simulation(model, time_steps=40)
    bridge = _FDTDGradientBridge(simulation)
    bridge.forward(tuple(bridge.material_inputs))

    profile = bridge.backward_profile()

    assert profile["seed_injection_backend"] == "device_batched"
    assert profile["seed_batch_counts"]["point"] > 0
    assert profile["reverse_backend_counts"].get(adjoint_baselines.expected_bloch_reverse_backend(), 0) == bridge._time_steps


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_gradient_bridge_backward_profile_uses_native_grating_tfsf():
    model = _DensityGratingTFSFScene(init=0.0).cuda()
    simulation = _build_simulation(model, time_steps=32)
    bridge = _FDTDGradientBridge(simulation)
    bridge.forward(tuple(bridge.material_inputs))

    profile = bridge.backward_profile()

    assert profile["seed_injection_backend"] == "device_batched"
    assert profile["seed_batch_counts"]["point"] > 0
    assert profile["reverse_backend_counts"].get("native_grating_tfsf", 0) == bridge._time_steps


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
@pytest.mark.parametrize("axis", ["x", "y"])
def test_fdtd_gradient_bridge_backward_profile_uses_grating_tfsf_for_general_axis(axis):
    # The grating-slab reverse backend is selected for any single-PML-axis grating
    # layout, not only z, so a non-z normal axis must select its native runner.
    model = _DensityGratingTFSFScene(init=0.0, axis=axis).cuda()
    simulation = _build_simulation(model, time_steps=32)
    bridge = _FDTDGradientBridge(simulation)
    bridge.forward(tuple(bridge.material_inputs))

    profile = bridge.backward_profile()
    assert profile["reverse_backend_counts"].get("native_grating_tfsf", 0) == bridge._time_steps


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_gradient_bridge_backward_profile_uses_native_dispersive_cpml_for_dispersive_materials():
    model = _DensityDispersiveScene(medium_kind="debye", init=0.0).cuda()
    simulation = _build_simulation(model, time_steps=36)
    bridge = _FDTDGradientBridge(simulation)
    bridge.forward(tuple(bridge.material_inputs))

    profile = bridge.backward_profile()

    assert profile["seed_injection_backend"] == "device_batched"
    assert profile["seed_batch_counts"]["point"] > 0
    assert profile["reverse_backend_counts"].get(adjoint_baselines.expected_dispersive_reverse_backend(), 0) == bridge._time_steps


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_gradient_bridge_backward_profile_uses_device_batched_dense_seeds():
    model = _DensityPointScene(init=0.0).cuda()
    simulation = mw.Simulation.fdtd(
        model,
        frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=24),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=True,
    )
    bridge = _FDTDGradientBridge(simulation)
    bridge.forward(tuple(bridge.material_inputs))

    profile = bridge.backward_profile()

    assert profile["seed_injection_backend"] == "device_batched"
    assert profile["seed_batch_counts"]["dense"] > 0
    assert profile["seed_batch_counts"]["point"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_gradient_bridge_backward_profile_uses_device_batched_plane_seeds():
    model = _DensityFluxScene(init=0.0).cuda()
    simulation = _build_simulation(model, time_steps=24)
    bridge = _FDTDGradientBridge(simulation)
    bridge.forward(tuple(bridge.material_inputs))

    profile = bridge.backward_profile()

    assert profile["seed_injection_backend"] == "device_batched"
    assert profile["seed_batch_counts"]["plane"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_gradient_bridge_checkpoint_replay_matches_forward_state():
    model = _DensityPointScene(init=0.0).cuda()
    simulation = _build_simulation(model)
    bridge = _FDTDGradientBridge(simulation)
    bridge.forward(tuple(bridge.material_inputs))

    solver = bridge._last_solver
    full_states = _replay_segment_states(
        solver,
        bridge._last_checkpoints[0],
        0,
        bridge._time_steps,
    )
    for checkpoint in bridge._last_checkpoints:
        reference_state = full_states[checkpoint.step]
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            assert torch.allclose(checkpoint.tensors[name], reference_state[name], rtol=1e-4, atol=5e-5)

    terminal_state = full_states[-1]
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        assert torch.allclose(terminal_state[name], getattr(solver, name), rtol=1e-4, atol=5e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_replay_mid_magnetic_capture_matches_recompute():
    """The mid-step H the checkpoint replay captures for the reverse reference
    backend is bit-identical to the backend's own ``_forward_magnetic_fields``
    recompute, so threading it changes nothing numerically."""
    from witwin.maxwell.fdtd.adjoint.core import (
        _forward_magnetic_fields,
        _replay_can_capture_mid_magnetic,
        _resolved_source_term_lists,
    )

    model = _DensityPointScene(init=0.0).cuda()
    simulation = _build_simulation(model)
    bridge = _FDTDGradientBridge(simulation)
    bridge.forward(tuple(bridge.material_inputs))
    solver = bridge._last_solver

    assert _replay_can_capture_mid_magnetic(solver) is True

    mid_magnetic = []
    states = _replay_segment_states(
        solver,
        bridge._last_checkpoints[0],
        0,
        bridge._time_steps,
        mid_magnetic_out=mid_magnetic,
    )
    assert len(mid_magnetic) == bridge._time_steps

    resolved = _resolved_source_term_lists(solver, solver.eps_Ex, solver.eps_Ey, solver.eps_Ez)
    for step_index in range(bridge._time_steps):
        recompute = _forward_magnetic_fields(
            solver,
            states[step_index],
            time_value=step_index * solver.dt,
            resolved_source_terms=resolved,
        )
        for name in ("Hx", "Hy", "Hz"):
            assert torch.equal(mid_magnetic[step_index][name], recompute[name])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
@pytest.mark.parametrize(
    "model_factory",
    [
        lambda: _DensityTFSFScene(source_kind="plane_wave", init=0.0).cuda(),
        lambda: _DensityBlochScene(init=0.0).cuda(),
        lambda: _DensityDispersiveScene(medium_kind="debye", init=0.0).cuda(),
    ],
    ids=["tfsf", "bloch", "dispersive"],
)
def test_fdtd_replay_mid_magnetic_capture_guard_excludes_non_reference_paths(model_factory):
    """The replay only offers its mid-step H to the pure real standard / CPML
    reference backends; TFSF, complex Bloch, and dispersive configurations keep
    the reverse recompute and never populate the capture list."""
    from witwin.maxwell.fdtd.adjoint.core import _replay_can_capture_mid_magnetic

    model = model_factory()
    simulation = _build_simulation(model, time_steps=24)
    bridge = _FDTDGradientBridge(simulation)
    bridge.forward(tuple(bridge.material_inputs))
    solver = bridge._last_solver

    assert _replay_can_capture_mid_magnetic(solver) is False

    mid_magnetic = []
    _replay_segment_states(
        solver,
        bridge._last_checkpoints[0],
        0,
        bridge._time_steps,
        mid_magnetic_out=mid_magnetic,
    )
    assert mid_magnetic == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_replay_mid_magnetic_capture_leaves_gradient_bit_identical(monkeypatch):
    """End-to-end proof that consuming the captured mid-step H produces exactly
    the same material gradient as recomputing it inside the reverse backend."""
    import witwin.maxwell.fdtd.adjoint.core as adjoint_core

    model = _DensityPointScene(init=0.0).cuda()
    _, _, loss = _loss_from_point_probe(model)
    loss.backward()
    grad_with_capture = model.logits.grad.detach().clone()

    monkeypatch.setattr(adjoint_core, "_replay_can_capture_mid_magnetic", lambda solver: False)
    model_recompute = _DensityPointScene(init=0.0).cuda()
    _, _, loss_recompute = _loss_from_point_probe(model_recompute)
    loss_recompute.backward()
    grad_with_recompute = model_recompute.logits.grad.detach().clone()

    assert torch.equal(grad_with_capture, grad_with_recompute)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_gradient_bridge_checkpoint_replay_matches_tfsf_forward_state():
    model = _DensityTFSFScene(source_kind="plane_wave", init=0.0).cuda()
    simulation = _build_simulation(model, time_steps=48)
    bridge = _FDTDGradientBridge(simulation)
    bridge.forward(tuple(bridge.material_inputs))

    solver = bridge._last_solver
    full_states = _replay_segment_states(
        solver,
        bridge._last_checkpoints[0],
        0,
        bridge._time_steps,
    )
    for checkpoint in bridge._last_checkpoints:
        reference_state = full_states[checkpoint.step]
        for name, tensor in checkpoint.tensors.items():
            if name.startswith("psi_"):
                continue
            assert torch.allclose(tensor, reference_state[name], rtol=1e-5, atol=1e-5)

    terminal_state = full_states[-1]
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        assert torch.allclose(terminal_state[name], getattr(solver, name), rtol=1e-4, atol=5e-5)
    auxiliary_grid = solver._tfsf_state["auxiliary_grid"]
    assert torch.allclose(terminal_state["tfsf_aux_electric"], auxiliary_grid.electric, rtol=1e-5, atol=1e-5)
    assert torch.allclose(terminal_state["tfsf_aux_magnetic"], auxiliary_grid.magnetic, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_gradient_bridge_checkpoint_replay_matches_bloch_forward_state():
    model = _DensityBlochScene(init=0.0).cuda()
    simulation = _build_simulation(model, time_steps=40)
    bridge = _FDTDGradientBridge(simulation)
    bridge.forward(tuple(bridge.material_inputs))

    solver = bridge._last_solver
    full_states = _replay_segment_states(
        solver,
        bridge._last_checkpoints[0],
        0,
        bridge._time_steps,
    )
    for checkpoint in bridge._last_checkpoints:
        reference_state = full_states[checkpoint.step]
        for name, tensor in checkpoint.tensors.items():
            assert torch.allclose(tensor, reference_state[name], rtol=1e-5, atol=1e-5)

    terminal_state = full_states[-1]
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Ex_imag", "Ey_imag", "Ez_imag", "Hx_imag", "Hy_imag", "Hz_imag"):
        assert torch.allclose(terminal_state[name], getattr(solver, name), rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_gradient_bridge_checkpoint_replay_matches_dispersive_forward_state():
    model = _DensityDispersiveScene(medium_kind="debye", init=0.0).cuda()
    simulation = _build_simulation(model, time_steps=36)
    bridge = _FDTDGradientBridge(simulation)
    bridge.forward(tuple(bridge.material_inputs))

    solver = bridge._last_solver
    full_states = _replay_segment_states(
        solver,
        bridge._last_checkpoints[0],
        0,
        bridge._time_steps,
    )
    for checkpoint in bridge._last_checkpoints:
        reference_state = full_states[checkpoint.step]
        for name, tensor in checkpoint.tensors.items():
            if name.startswith("psi_"):
                continue
            # The native CUDA forward and differentiable Torch replay use
            # different floating-point operation order. Their small discrepancy
            # accumulates in the outer CPML over a full checkpoint segment.
            assert torch.allclose(tensor, reference_state[name], rtol=3e-2, atol=5e-4)

    terminal_state = full_states[-1]
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        assert torch.allclose(terminal_state[name], getattr(solver, name), rtol=3e-2, atol=5e-4)
    for component_name in ("Ex", "Ey", "Ez"):
        component_templates = solver._dispersive_templates.get(component_name, {})
        for model_name in ("debye", "drude", "lorentz"):
            for index, entry in enumerate(component_templates.get(model_name, ())):
                if model_name != "drude":
                    polarization_name = dispersive_state_name(component_name, model_name, index, "polarization")
                    assert torch.allclose(terminal_state[polarization_name], entry["polarization"], rtol=1e-5, atol=1e-5)
                current_name = dispersive_state_name(component_name, model_name, index, "current")
                assert torch.allclose(terminal_state[current_name], entry["current"], rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_gradient_bridge_rejects_scene_module_parameters_outside_material_graph():
    model = _UnsupportedSceneParam().cuda()

    with pytest.raises(NotImplementedError, match="prepared-scene material tensors"):
        _build_simulation(model).run()
