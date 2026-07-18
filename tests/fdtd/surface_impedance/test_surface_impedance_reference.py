"""Phase 0 acceptance budget freeze and torch reference oracle for the surface boundary.

This is the consumer test that binds ``witwin/maxwell/fdtd/surface_impedance_reference.py``
to the surface-impedance contract: it pins every frozen budget threshold, checks the
generic torch oracle reproduces the incumbent narrowband ``|Gamma|`` gate, exercises the
analytic Fresnel/Leontovich oblique reflection, and proves the discrete surface power
form is nonnegative and double-count free on a hand-built two-face corner. Without this
test the reference silently stops constraining anything.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd import surface_impedance_reference as ref
from witwin.maxwell.fdtd.surface_impedance_reference import (
    SURFACE_ACCEPTANCE_BUDGET as BUDGET,
    SurfaceAcceptanceBudget,
    SurfaceEdgeContribution,
    assemble_surface_dissipation,
    good_conductor_surface_impedance,
    leontovich_reflection,
    naive_double_counted_dissipation,
    power_balance_residual,
)

_MU0 = 4.0e-7 * np.pi
_EPS0 = 8.8541878128e-12
_SIGMA = 50.0  # matches tests/validation/physics/test_lossy_metal_sibc.py


def _incumbent_gamma_magnitude(sigma, frequency):
    """The exact analytic |Gamma| gate from the incumbent SIBC validation test."""
    omega = 2.0 * np.pi * frequency
    eta0 = np.sqrt(_MU0 / _EPS0)
    r = np.sqrt(omega * _MU0 / (2.0 * sigma))
    z_s = r + 1j * r
    return abs((z_s - eta0) / (z_s + eta0))


# --------------------------------------------------------------------------- #
# Frozen acceptance budget
# --------------------------------------------------------------------------- #


def test_surface_acceptance_budget_is_frozen():
    """Pin every Phase 0 acceptance threshold; a change must update this test."""
    assert SurfaceAcceptanceBudget() == BUDGET
    expected = {
        "fit_max_complex_error": 1.0e-3,
        "fit_min_passivity_margin": 0.0,
        "passivity_tolerance": 1.0e-9,
        "discrete_pole_radius_max": 1.0,
        "analytic_reflection_relative_error": 2.0e-2,
        "analytic_phase_error_deg": 3.0,
        "narrowband_reproduction_relative_error": 5.0e-2,
        "power_residual": 1.0e-2,
        "min_local_surface_dissipation": 0.0,
        "convergence_levels": 3,
        "gradient_relative_error": 2.0e-2,
        "gradient_absolute_floor": 1.0e-8,
        "no_sibc_runtime_regression": 1.0e-2,
    }
    actual = {f.name: getattr(BUDGET, f.name) for f in dataclasses.fields(BUDGET)}
    assert actual == expected
    # The narrowband reproduction gate must never be loosened below the incumbent 5%.
    assert BUDGET.narrowband_reproduction_relative_error == 5.0e-2


def test_surface_acceptance_budget_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        BUDGET.fit_max_complex_error = 1.0  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Analytic Fresnel / Leontovich oracle
# --------------------------------------------------------------------------- #


def test_good_conductor_surface_impedance_matches_public_media():
    """The oracle Z_s must equal LossyMetalMedium.surface_impedance (convention pin).

    The oracle uses the CODATA mu0 (as thin_wire_reference.py does) while media.py uses
    4e-7*pi, so the constants agree only to ~5e-10; the pinned quantity is the sign and
    functional form (Z_s = (1 - i) * sqrt(omega*mu/(2*sigma))), not the last digit.
    """
    metal = mw.LossyMetalMedium(conductivity=_SIGMA)
    for frequency in (1.0e9, 2.0e9, 3.0e9):
        z_ref = complex(good_conductor_surface_impedance(_SIGMA, [frequency])[0])
        z_media = metal.surface_impedance(2.0 * math.pi * frequency)
        assert z_ref.real == pytest.approx(z_media.real, rel=1e-8)
        assert z_ref.imag == pytest.approx(z_media.imag, rel=1e-8)
        assert z_ref.imag < 0.0  # e^{-i omega t}: inductive part is negative imaginary


def test_oracle_reproduces_incumbent_narrowband_reflection():
    """Generic oracle reproduces the incumbent |Gamma| gate within the frozen budget."""
    for frequency in (1.0e9, 2.0e9, 3.0e9):
        z_s = good_conductor_surface_impedance(_SIGMA, [frequency])
        gamma = leontovich_reflection(z_s)[0]
        incumbent = _incumbent_gamma_magnitude(_SIGMA, frequency)
        rel_err = abs(abs(complex(gamma)) - incumbent) / incumbent
        assert rel_err < BUDGET.narrowband_reproduction_relative_error
        # This is a self-consistent reproduction of the same formula, so it must agree
        # to solver precision -- far below the 5% gate. (Falsification: a wrong sigma
        # would move |Gamma| by percent, not 1e-12.)
        assert rel_err < 1.0e-10
        assert abs(complex(gamma)) < 1.0  # a passive good conductor absorbs some power


def test_wrong_conductivity_changes_reflection():
    """Falsification: the reproduction is not vacuous -- sigma actually matters."""
    frequency = 2.0e9
    g_ref = abs(complex(leontovich_reflection(good_conductor_surface_impedance(_SIGMA, [frequency]))[0]))
    g_other = abs(complex(leontovich_reflection(good_conductor_surface_impedance(10.0 * _SIGMA, [frequency]))[0]))
    assert abs(g_ref - g_other) > 1.0e-3


def test_oblique_te_tm_reduce_to_normal_and_stay_passive():
    z_s = good_conductor_surface_impedance(_SIGMA, [2.0e9])
    normal = complex(leontovich_reflection(z_s)[0])
    te0 = complex(leontovich_reflection(z_s, angle=0.0, polarization="te")[0])
    tm0 = complex(leontovich_reflection(z_s, angle=0.0, polarization="tm")[0])
    assert te0 == pytest.approx(normal)
    assert tm0 == pytest.approx(normal)
    # Physical trends for a good conductor: TE reflects more, TM less, as the angle grows.
    angles = [0.0, 0.3, 0.6, 1.0]
    te = [abs(complex(leontovich_reflection(z_s, angle=a, polarization="te")[0])) for a in angles]
    tm = [abs(complex(leontovich_reflection(z_s, angle=a, polarization="tm")[0])) for a in angles]
    assert all(later >= earlier - 1e-12 for earlier, later in zip(te, te[1:]))
    assert all(later <= earlier + 1e-12 for earlier, later in zip(tm, tm[1:]))
    assert all(magnitude <= 1.0 for magnitude in te + tm)


def test_power_balance_residual_closes_at_normal_and_oblique():
    """1 - |Gamma|^2 from reflection equals the surface-loss fraction (exact identity)."""
    z_s = good_conductor_surface_impedance(_SIGMA, [2.0e9])
    for polarization in ("normal", "te", "tm"):
        for angle in (0.0, 0.3, 0.7):
            residual = float(
                power_balance_residual(z_s, angle=angle, polarization=polarization)[0]
            )
            assert residual < BUDGET.power_residual
            assert residual < 1.0e-10  # analytic identity, closes to solver precision


# --------------------------------------------------------------------------- #
# Discrete surface power form: unique-owner assembly on a two-face corner
# --------------------------------------------------------------------------- #


def _corner_contributions(*, shared_resistance=(2.0, 2.0)):
    """Two perpendicular faces meeting at one shared Yee edge (id 100).

    Face 0 owns interior edges 10, 11; face 1 owns interior edges 20, 21; both claim
    the shared corner edge 100 (with possibly distinct local samples).
    """
    r0, r1 = shared_resistance
    return [
        SurfaceEdgeContribution(face_id=0, edge_id=10, surface_resistance=2.0, dual_area=0.5, tangential_field=3.0),
        SurfaceEdgeContribution(face_id=0, edge_id=11, surface_resistance=2.0, dual_area=0.5, tangential_field=3.0),
        SurfaceEdgeContribution(face_id=0, edge_id=100, surface_resistance=r0, dual_area=0.5, tangential_field=4.0),
        SurfaceEdgeContribution(face_id=1, edge_id=20, surface_resistance=2.0, dual_area=0.5, tangential_field=3.0),
        SurfaceEdgeContribution(face_id=1, edge_id=21, surface_resistance=2.0, dual_area=0.5, tangential_field=3.0),
        SurfaceEdgeContribution(face_id=1, edge_id=100, surface_resistance=r1, dual_area=0.5, tangential_field=5.0),
    ]


def test_corner_dissipation_nonnegative_and_counts_shared_edge_once():
    contributions = _corner_contributions()
    total, per_edge = assemble_surface_dissipation(contributions)
    # Five unique edges (the corner edge is not duplicated).
    assert set(per_edge) == {10, 11, 20, 21, 100}
    assert all(value >= BUDGET.min_local_surface_dissipation for value in per_edge.values())
    assert total == pytest.approx(sum(per_edge.values()))
    assert total >= BUDGET.min_local_surface_dissipation


def test_unique_owner_rule_is_active_against_double_counting():
    """Falsification: the owner assembly must differ from a naive double count."""
    contributions = _corner_contributions()
    owner_total, per_edge = assemble_surface_dissipation(contributions)
    naive_total = naive_double_counted_dissipation(contributions)
    # The naive assembly counts the shared corner edge twice (once per adjoining face);
    # the owner assembly counts it exactly once, so they cannot be equal.
    assert naive_total > owner_total
    # The shared edge is owned by the minimum global face id (face 0), so its power uses
    # face 0's local sample (|H| = 4), not face 1's (|H| = 5).
    assert per_edge[100] == pytest.approx(0.5 * 2.0 * 0.5 * 4.0**2)


def test_negative_surface_resistance_is_rejected():
    """min_local_surface_dissipation is a hard gate: a non-passive R < 0 must raise."""
    bad = [
        SurfaceEdgeContribution(face_id=0, edge_id=1, surface_resistance=-1.0, dual_area=0.5, tangential_field=1.0)
    ]
    with pytest.raises(ValueError, match="nonnegative"):
        assemble_surface_dissipation(bad)


def test_module_exposes_frozen_singleton():
    assert ref.SURFACE_ACCEPTANCE_BUDGET is BUDGET
