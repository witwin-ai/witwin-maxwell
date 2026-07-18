"""Slice 0b: torch LLG reference oracle for the gyromagnetic ferrite contract.

Binds docs/reference/ferrite-physics-contract.md to code. Falsification checks
are included: flipping any single frozen convention must break a gate.
"""

from __future__ import annotations

import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd import ferrite_reference as fr

BUDGET = fr.ACCEPTANCE_BUDGET


def _params(alpha=2.0e-3, bias=(0.0, 0.0, 1.0), mu_inf=1.0):
    return fr.FerriteReferenceParameters(
        saturation_magnetization=1.40e5,
        bias_magnitude=1.75e5,
        bias_unit_vector=bias,
        gilbert_damping=alpha,
        mu_infinity=mu_inf,
        eps_r=14.5,
    )


def _in_band_omega():
    # Above the ~6.16 GHz gyromagnetic resonance, away from the pole.
    return 2.0 * math.pi * 9.0e9


def test_budget_matches_frozen_contract():
    assert BUDGET.reference_polder_rtol == 1.0e-5
    assert BUDGET.analytic_response_rel_err == 2.0e-2
    assert BUDGET.analytic_phase_err_deg == 3.0
    assert BUDGET.passive_energy_residual == 1.0e-2
    assert BUDGET.convergence_tiers == 3
    assert BUDGET.bias_reversal_symmetry_rtol == 1.0e-5


def test_recurrence_algebra_identity_in_band():
    """The closed-form discrete phasor satisfies the implicit-midpoint update."""
    params = _params()
    for fghz in (8.0, 9.0, 12.0):
        resid = fr.recurrence_identity_residual(params, 2.0 * math.pi * fghz * 1e9, 2.0e-13)
        assert resid <= BUDGET.reference_polder_rtol


def test_discrete_converges_to_continuous_three_dt_tiers():
    """>=3 dt tiers, monotone O(dt^2) convergence to the analytic Polder (<2%)."""
    params = _params()
    omega = _in_band_omega()
    mu_cont, kappa_cont = fr.continuous_polder(params, omega)
    errors = []
    for dt in (4.0e-13, 2.0e-13, 1.0e-13):
        mu_d, kappa_d = fr.discrete_polder(params, omega, dt)
        err = abs(complex(mu_d - mu_cont)) / abs(complex(mu_cont))
        errors.append(err)
    assert len(errors) >= BUDGET.convergence_tiers
    assert errors[-1] < BUDGET.analytic_response_rel_err
    # Monotone decreasing with ~4x per halving (second order).
    assert errors[0] > errors[1] > errors[2]
    assert errors[0] / errors[1] > 3.0
    assert errors[1] / errors[2] > 3.0


def test_material_polder_matches_oracle_continuous():
    """The public GyromagneticFerrite formula matches the independent oracle."""
    ferrite = mw.GyromagneticFerrite(
        eps_r=14.5, saturation_magnetization=1.40e5, bias_field=(0.0, 0.0, 1.75e5),
        gilbert_damping=2.0e-3,
    )
    params = _params()
    for fghz in (8.0, 9.0, 12.0):
        omega = 2.0 * math.pi * fghz * 1e9
        t_mat = ferrite.polder_tensor(omega)
        t_ref = fr.reference_polder_tensor(params, omega)
        rel = float((t_mat - t_ref).abs().max() / t_ref.abs().max())
        assert rel <= BUDGET.reference_polder_rtol


def test_bias_reversal_flips_kappa_not_diagonal():
    params = _params()
    params_rev = params.with_reversed_bias()
    omega = _in_band_omega()
    t = fr.reference_polder_tensor(params, omega)
    t_rev = fr.reference_polder_tensor(params_rev, omega)
    # Off-diagonal (kappa) flips exactly; diagonals do not.
    assert torch.allclose(t[0, 1], -t_rev[0, 1], rtol=BUDGET.bias_reversal_symmetry_rtol)
    assert torch.allclose(t[1, 0], -t_rev[1, 0], rtol=BUDGET.bias_reversal_symmetry_rtol)
    assert torch.allclose(t[0, 0], t_rev[0, 0], rtol=BUDGET.bias_reversal_symmetry_rtol)
    assert torch.allclose(t[2, 2], t_rev[2, 2], rtol=BUDGET.bias_reversal_symmetry_rtol)


def test_faraday_rotation_sign_flips_with_bias():
    params = _params(alpha=0.0)
    omega = _in_band_omega()
    theta = fr.faraday_rotation_angle(params, omega, 1.0e-3)
    theta_rev = fr.faraday_rotation_angle(params.with_reversed_bias(), omega, 1.0e-3)
    assert float(theta) != 0.0
    assert torch.allclose(theta, -theta_rev, rtol=BUDGET.bias_reversal_symmetry_rtol)


def test_lossless_energy_does_not_grow():
    """alpha=0: implicit-midpoint propagator conserves precession energy exactly."""
    llg = fr.LLGReference(_params(alpha=0.0), dt=2.0e-13)
    energies = llg.energy_trajectory(torch.tensor([1.0, 0.3], dtype=torch.float64), 20000)
    ratio = float(energies[-1] / energies[0])
    assert abs(ratio - 1.0) <= BUDGET.passive_energy_residual
    # Zero-growth (not merely bounded): never exceeds the initial energy.
    assert bool((energies <= energies[0] * (1.0 + 1.0e-9)).all())


def test_lossy_energy_monotone_decay():
    llg = fr.LLGReference(_params(alpha=5.0e-2), dt=2.0e-13)
    energies = llg.energy_trajectory(torch.tensor([1.0, 0.3], dtype=torch.float64), 5000)
    assert bool((energies[1:] <= energies[:-1] + 1.0e-15).all())
    assert float(energies[-1]) < float(energies[0])


def test_stepping_matches_discrete_transfer_function():
    """Actual CW time-stepping reproduces the closed-form discrete response."""
    params = _params(alpha=1.0e-2)
    omega = _in_band_omega()
    dt = 2.0e-13
    chi_xx, chi_yx = fr.LLGReference(params, dt=dt).run_cw(omega, periods=3000, settle=1500)
    chi = fr.discrete_susceptibility(params, omega, dt)
    assert abs(complex(chi_xx) - complex(chi[0, 0])) / abs(complex(chi[0, 0])) < BUDGET.analytic_response_rel_err
    assert abs(complex(chi_yx) - complex(chi[1, 0])) / abs(complex(chi[1, 0])) < BUDGET.analytic_response_rel_err


def test_circular_resonance_handedness():
    """mu_+ carries the gyromagnetic resonance at omega_0; mu_- is non-resonant."""
    params = _params(alpha=1.0e-3)
    omega0 = params.omega_0
    near = 0.999 * omega0
    mu_plus, mu_minus = fr.circular_permeabilities(params, near)
    # The resonant (+) branch has a much larger magnitude near omega_0.
    assert abs(complex(mu_plus)) > 5.0 * abs(complex(mu_minus))


# --- Falsification: flipping any single convention breaks a gate --------------


def test_falsify_gyromagnetic_sign():
    """Flipping the precession (kappa) sign breaks the bias-reversal identity vs. the material."""
    params = _params()
    omega = _in_band_omega()
    good = fr.reference_polder_tensor(params, omega)
    ferrite = mw.GyromagneticFerrite(
        eps_r=14.5, saturation_magnetization=1.40e5, bias_field=(0.0, 0.0, 1.75e5),
        gilbert_damping=2.0e-3,
    )
    t_mat = ferrite.polder_tensor(omega)
    # A wrong-sign off-diagonal reference no longer matches the material.
    wrong = good.clone()
    wrong[0, 1] = -wrong[0, 1]
    wrong[1, 0] = -wrong[1, 0]
    rel_good = float((t_mat - good).abs().max() / good.abs().max())
    rel_wrong = float((t_mat - wrong).abs().max() / wrong.abs().max())
    assert rel_good <= BUDGET.reference_polder_rtol
    assert rel_wrong > BUDGET.reference_polder_rtol


def test_falsify_damping_sign_breaks_passivity():
    """A negative-damping (gain) sign makes the propagator grow energy."""
    # The reference forbids alpha<0 at construction, so emulate the wrong sign by
    # building the state-space with a negated damping and checking energy grows.
    params = _params(alpha=5.0e-2)
    dt = 2.0e-13
    P, _ = fr.state_space_matrices(params.omega_0, params.omega_m, -params.gilbert_damping)
    eye = torch.eye(2, dtype=torch.float64)
    phi = torch.linalg.inv(eye - dt / 2 * P) @ (eye + dt / 2 * P)
    m = torch.tensor([1.0, 0.3], dtype=torch.float64)
    e0 = 0.5 * (m @ m)
    for _ in range(5000):
        m = phi @ m
    assert float(0.5 * (m @ m) / e0) > 1.0 + BUDGET.passive_energy_residual


def test_falsify_time_convention():
    """Using exp(+i omega t) (conjugate) inverts the absorptive sign near resonance."""
    params = _params(alpha=1.0e-2)
    omega = 1.02 * params.omega_0  # just above resonance
    mu, _ = fr.continuous_polder(params, omega)
    # exp(-i omega t): passive medium has positive imaginary permeability (absorption).
    assert mu.imag > 0.0
    # exp(+i omega t) would flip the imaginary sign, violating passivity.
    wrong = complex(mu).conjugate()
    assert wrong.imag < 0.0
