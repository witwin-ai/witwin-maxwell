"""Forward FDTD validation for full-anisotropic + conductive (sigma_e) media.

This is the ``aniso-sigma-e`` combination of the P5.2 material matrix: a full
(off-diagonal ``Tensor3x3``) permittivity carrying a static electric conductivity,
i.e. a *lossy anisotropic crystal*. The exact semi-implicit update folds the loss
through the per-edge inverse of the conductively-shifted tensor

    B = dt * (eps_inf + dt/2 * diag(sigma))^-1,   E^{n+1} = E^n + B . (curl H - sigma . E^n),

so the diagonal decay is 1 - sigma_i * B_ii and the off-diagonal conduction
current -B_ij * sigma_j E_j^n couples the transverse components. With an isotropic
conductivity the operator ``B`` diagonalizes in the crystal principal frame, so a
uniform field decays per principal axis at exactly the scalar semi-implicit rate
(the "diagonalized isotropic-loss reference"), and a plane wave through the slab
absorbs at the analytic complex-index rate of its principal polarization.

The tests validate (1) the uniform conduction decay against the analytic per-axis
semi-implicit factor -- including the off-diagonal-generated cross component that
would vanish if the coupled-tensor conduction were dropped -- and (2) an integrated
plane-wave slab whose transmitted amplitude reproduces exp(-k0 Im(n) L) for the
extraordinary index n = sqrt(eps_e + i sigma/(omega eps0)).
"""

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box

# Speed of light and vacuum permittivity used by the solver.
_C = 299792458.0
_EPS0 = 8.8541878128e-12
_INV_SQRT2 = 2.0**-0.5


def _rotated_uniaxial_tensor(eps_xx, eps_ordinary, eps_extraordinary):
    """Uniaxial permittivity with principal axes rotated 45 degrees about x.

    Extraordinary axis u = (y+z)/sqrt(2) (eps_extraordinary), ordinary axis
    v = (-y+z)/sqrt(2) (eps_ordinary), so eps_yy = eps_zz = (e_o + e_e)/2 and
    eps_yz = (e_e - e_o)/2. The off-diagonal yz term couples the two transverse
    components, which is exactly what the off-diagonal conduction must damp.
    """
    mean = 0.5 * (eps_ordinary + eps_extraordinary)
    delta = 0.5 * (eps_extraordinary - eps_ordinary)
    return mw.Tensor3x3(((eps_xx, 0.0, 0.0), (0.0, mean, delta), (0.0, delta, mean)))


# ---------------------------------------------------------------------------
# Construction / prepared-solver behavior
# ---------------------------------------------------------------------------


def test_lossy_anisotropic_material_constructs():
    """Full anisotropy + conductivity no longer raises at construction."""
    material = mw.Material(epsilon_tensor=_rotated_uniaxial_tensor(2.0, 2.0, 3.0), sigma_e=0.4)
    assert material.is_anisotropic
    assert float(material.sigma_e) == 0.4


def _build_uniform_solver(eps_o, eps_e, eps_xx, sigma, *, spacing=0.03):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3))),
        grid=mw.GridSpec.uniform(spacing),
        boundary=mw.BoundarySpec.none(),
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(
            geometry=Box(position=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0)),
            material=mw.Material(
                epsilon_tensor=_rotated_uniaxial_tensor(eps_xx, eps_o, eps_e), sigma_e=sigma
            ),
        )
    )
    return mw.Simulation.fdtd(scene, frequency=1.0e9).prepare().solver


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_lossy_anisotropic_prepared_solver_coefficients():
    """The conductive fold uses the shifted tensor inverse and the 1 - sigma*curl decay."""
    lossy = _build_uniform_solver(2.0, 3.0, 2.0, 0.05)
    lossless = _build_uniform_solver(2.0, 3.0, 2.0, 0.0)

    assert lossy.full_aniso_enabled and lossy.conductive_enabled
    assert lossy._aniso_cond_current is not None
    assert not lossless.conductive_enabled

    # Diagonal decay satisfies the semi-implicit identity decay = 1 - sigma * curl
    # exactly (this is what the aniso_shifted branch composes for the tensor case).
    for decay, curl, sigma in (
        (lossy.cex_decay, lossy.cex_curl, lossy.sigma_e_Ex),
        (lossy.cey_decay, lossy.cey_curl, lossy.sigma_e_Ey),
        (lossy.cez_decay, lossy.cez_curl, lossy.sigma_e_Ez),
    ):
        assert torch.allclose(decay, 1.0 - sigma * curl, atol=1e-6)
        # Conductive decay is a genuine loss (< 1) inside the crystal.
        assert float(decay.min()) < 1.0

    # The conductive shift raises the effective permittivity, so the semi-implicit
    # curl coefficient B_ii = dt/eps_eff is strictly smaller than the lossless one.
    assert float((lossy.cex_curl - lossless.cex_curl).max()) <= 0.0
    assert float((lossy.cey_curl - lossless.cey_curl).min()) < 0.0


# ---------------------------------------------------------------------------
# Physics 1: uniform conduction decay vs the analytic principal-axis factor
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_lossy_anisotropic_uniform_conduction_decay_matches_analytic():
    """A uniform field in a lossy rotated-uniaxial medium decays per principal axis.

    With H held at zero the update is pure conduction ``E^{n+1} = (I - B sigma) E^n``.
    For isotropic sigma this diagonalizes in the crystal frame, so a uniform z-field
    decays at the ordinary/extraordinary semi-implicit rates and, crucially, spins
    up a y-component through the yz off-diagonal conduction; dropping that coupling
    would leave Ey identically zero.
    """
    eps_o, eps_e, eps_xx, sigma = 2.0, 3.0, 2.0, 0.03
    steps = 60
    solver = _build_uniform_solver(eps_o, eps_e, eps_xx, sigma)
    assert solver.full_aniso_enabled and solver.conductive_enabled

    from witwin.maxwell.fdtd.runtime.stepping import (
        apply_full_aniso_conduction,
        apply_full_aniso_corrections,
        capture_aniso_conduction_currents,
        update_electric_fields,
    )

    solver.Hx.zero_()
    solver.Hy.zero_()
    solver.Hz.zero_()
    solver.Ex.zero_()
    solver.Ey.zero_()
    solver.Ez.fill_(1.0)
    for _ in range(steps):
        capture_aniso_conduction_currents(solver)
        update_electric_fields(solver, solver.Ex, solver.Ey, solver.Ez, solver.Hx, solver.Hy, solver.Hz)
        apply_full_aniso_corrections(solver)
        apply_full_aniso_conduction(solver)
    torch.cuda.synchronize()

    # Analytic per-principal-axis semi-implicit decay factor after `steps` steps.
    dt, eps0 = solver.dt, solver.eps0

    def factor(eps_axis):
        half = 0.5 * sigma * dt / (eps0 * eps_axis)
        return ((1.0 - half) / (1.0 + half)) ** steps

    fe, fo = factor(eps_e), factor(eps_o)
    # z decomposes equally onto the extraordinary (u) and ordinary (v) axes, so the
    # reconstructed transverse components are the sum/difference of the two factors.
    ez_ref = 0.5 * (fe + fo)
    ey_ref = 0.5 * (fe - fo)

    ci = tuple(s // 2 for s in solver.Ez.shape)
    cyi = tuple(s // 2 for s in solver.Ey.shape)
    ez = float(solver.Ez[ci].item())
    ey = float(solver.Ey[cyi].item())

    assert abs(ez - ez_ref) / abs(ez_ref) < 1e-3
    assert abs(ey - ey_ref) / abs(ey_ref) < 1e-3
    # The off-diagonal conduction genuinely coupled the axes (guards a silent drop).
    assert abs(ey) > 0.1 * abs(ez)


# ---------------------------------------------------------------------------
# Physics 2: integrated plane-wave slab absorbs at the analytic complex index
# ---------------------------------------------------------------------------

_SLAB = 0.18
_SPACING = 0.006
_EPS_XX = 1.6
_EPS_O = 1.6
_EPS_E = 2.4
_EXTRAORDINARY_POL = (0.0, _INV_SQRT2, _INV_SQRT2)
_FIT_LO, _FIT_HI = 0.20, 0.55


def _run_slab(frequency, sigma):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.75, 0.75), (-0.03, 0.03), (-0.03, 0.03))),
        grid=mw.GridSpec.uniform(_SPACING),
        boundary=mw.BoundarySpec.faces(default="periodic", num_layers=10, strength=1.0, x=("pml", "pml")),
        device="cuda",
        sources=[
            mw.PointDipole(
                position=(-0.5, 0.0, 0.0),
                polarization=_EXTRAORDINARY_POL,
                width=0.02,
                source_time=mw.CW(frequency=frequency, amplitude=40.0),
                name="dip",
            )
        ],
    )
    scene.add_structure(
        mw.Structure(
            geometry=Box(position=(0.0, 0.0, 0.0), size=(_SLAB, 0.2, 0.2)),
            material=mw.Material(
                epsilon_tensor=_rotated_uniaxial_tensor(_EPS_XX, _EPS_O, _EPS_E), sigma_e=sigma
            ),
        )
    )
    return mw.Simulation.fdtd(
        scene,
        frequencies=[frequency],
        run_time=mw.TimeConfig.auto(steady_cycles=12, transient_cycles=26),
        full_field_dft=True,
    ).run()


def _projected_line(result):
    ey = result.field("Ey")
    ez = result.field("Ez")
    ey = (ey["data"] if isinstance(ey, dict) else ey).detach().cpu().numpy()
    ez = (ez["data"] if isinstance(ez, dict) else ez).detach().cpu().numpy()

    def _line(arr):
        crop = arr[:, 2:-2, 2:-2] if arr.shape[1] > 4 and arr.shape[2] > 4 else arr
        return crop.reshape(crop.shape[0], -1).mean(axis=1)

    line = _EXTRAORDINARY_POL[1] * _line(ey) + _EXTRAORDINARY_POL[2] * _line(ez)
    xs = result.solver.scene.x_nodes64
    if xs.shape != line.shape:
        raise ValueError("Projected field line does not match the solver x grid.")
    return xs, line


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_lossy_anisotropic_slab_attenuation_matches_analytic():
    """The extraordinary wave through the lossy slab absorbs at exp(-k0 Im(n) L).

    Taking the transmitted-amplitude ratio against the same slab with sigma = 0
    cancels the (near common) impedance reflection, isolating the conduction
    absorption. The analytic reference is the extraordinary complex index
    n = sqrt(eps_e + i sigma/(omega eps0)) with the isotropic conductive loss.
    """
    frequency = 1.0e9
    sigma = 0.02
    lossy = _run_slab(frequency, sigma)
    lossless = _run_slab(frequency, 0.0)

    xs, lossy_line = _projected_line(lossy)
    _, lossless_line = _projected_line(lossless)
    assert np.isfinite(np.abs(lossy_line)).all()

    window = (xs >= _FIT_LO) & (xs <= _FIT_HI)
    ratio = np.abs(lossy_line[window]).mean() / np.abs(lossless_line[window]).mean()

    omega = 2.0 * np.pi * frequency
    k0 = omega / _C
    n_complex = np.sqrt(_EPS_E + 1j * sigma / (omega * _EPS0))
    analytic_ratio = np.exp(-k0 * abs(n_complex.imag) * _SLAB)

    # The absorbed fraction is unambiguous (ratio clearly below one), and it matches
    # the analytic complex-index absorption within 5%.
    assert ratio < 0.85
    assert abs(ratio - analytic_ratio) / analytic_ratio < 0.05, (
        f"measured lossy/lossless amplitude ratio {ratio:.4f} vs analytic {analytic_ratio:.4f}"
    )


# ---------------------------------------------------------------------------
# Adjoint: the combination is forward-only for now (physics-worded rejection)
# ---------------------------------------------------------------------------


def test_lossy_anisotropic_adjoint_is_rejected():
    """The reverse conduction replica has no off-diagonal tensor channel yet."""
    from witwin.maxwell.fdtd.adjoint.bridge import _unsupported_adjoint_medium

    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.pml(num_layers=2),
        device="cpu",
    )
    scene.add_structure(
        mw.Structure(
            geometry=Box(position=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.4)),
            material=mw.Material(epsilon_tensor=_rotated_uniaxial_tensor(2.0, 2.0, 3.0), sigma_e=0.5),
        )
    )
    message = _unsupported_adjoint_medium(scene)
    assert message is not None
    assert "conductivity" in message
    assert "not implemented yet" not in message
