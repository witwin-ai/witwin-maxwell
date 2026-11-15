"""CPU qualification of the test-only thin-wire oracle against published tables.

Reference: R. W. P. King, The Theory of Linear Antennas, Harvard University Press
(1956), Sec. 30, with Omega = 2 ln(2h/a), h the half-length, a the radius, beta h
the electrical half-length and Z0 the delta-gap (slice generator) input impedance:

- Table 30.15 (p. 182): third-order resistance [R0]3 at beta h = pi/2. King (Sec. 38,
  pp. 232-233) reports that this value agrees with Hartig's measurements, while the
  second-order King-Middleton resistance is a few percent low.
- Table 30.1 (p. 168): second-order reactance X0 at beta h = pi/2, which King notes is
  accurate there (p. 183), and the first resonance (beta h)res, (R0)res.
- Tables 30.8, 30.9, 30.10 (pp. 174-176): second-order Z0 for Omega = 12.5, 15, 20 at
  beta h = 3.1 and 3.2 (full-wave region, near antiresonance).

Near antiresonance the oracle's mesh-regularized delta gap adds a mesh-dependent
shunt susceptance (exact-kernel delta-gap current has a logarithmic singularity; G.
Fikioris and C. A. Valagiannopoulos, PIER 55 (2005) 285-306), so the full-wave check
uses the conductance G0 = Re(1/Z0), which a shunt susceptance leaves unchanged.

The induced-EMF value 73.13 + j42.5 ohm is the infinitely thin (Omega -> infinity)
sinusoidal-current limit, not a finite-radius table value (King, p. 153), so it is no
acceptance target: King's third-order R0 at Omega = 20 is still 78.1 ohm.

Tolerances: R within 3 percent and X within 5 ohm at beta h = pi/2 (the spread between
the second- and third-order theories is 1-2 percent for Omega >= 12.5, and King bounds
the third-order conductance from above, p. 184); G0 within 3 percent at full wave;
(beta h)res within 1 percent and (R0)res within 2 percent. Omega < 12.5 is outside the
oracle's a/lambda <= 0.002 range at these lengths and is not asserted.

Loop citation to resolve: J. E. Storer, 'Impedance of Thin-Wire Loop Antennas',
Trans. AIEE Part I 75 (1956), 606-619, doi:10.1109/TCE.1956.6372437. No verified
table value is available, so the loop oracles stay unqualified.
"""
import math

import numpy as np
import pytest
from scipy.optimize import brentq

from tests.support.thin_wire_mom import C0, dipole, loop

SEGMENTS = 80
# Omega: ([R0]3 Table 30.15, X0 Table 30.1) at beta h = pi/2.
KING_HALF_WAVE = {12.5: (84.3, 43.0), 15.0: (81.5, 43.4), 20.0: (78.1, 43.6)}
# Omega: ((beta h)res, (R0)res), Table 30.1.
KING_RESONANCE = {12.5: (1.504, 71.0), 15.0: (1.514, 71.7), 20.0: (1.530, 72.2)}
# (Omega, beta h): Z0, Tables 30.8-30.10.
KING_FULL_WAVE = {
    (12.5, 3.1): complex(689.7, -841.4), (12.5, 3.2): complex(506.0, -787.8),
    (15.0, 3.1): complex(1312.0, -1328.0), (15.0, 3.2): complex(898.8, -1273.0),
    (20.0, 3.1): complex(3262.0, -2521.0), (20.0, 3.2): complex(1944.0, -2570.0),
}


def _king_dipole(omega, beta_h):
    """Delta-gap dipole at the table's Omega and beta h, with lambda = 1 m."""
    half_length = beta_h / (2 * math.pi)
    radius = 2 * half_length * math.exp(-omega / 2)
    return dipole(length=2 * half_length, radius=radius, frequency=C0, segments=SEGMENTS)


def test_dipole_oracle_is_marked_qualified_and_loop_is_not():
    assert dipole(length=.5, radius=1e-3, frequency=C0, segments=SEGMENTS).qualified
    assert not loop(circumference=1., radius=1e-3, frequency=C0, segments=80).qualified


@pytest.mark.parametrize("omega", sorted(KING_HALF_WAVE))
def test_half_wave_dipole_matches_king_table(omega):
    resistance, reactance = KING_HALF_WAVE[omega]
    z = _king_dipole(omega, math.pi / 2).impedance
    assert abs(z.real / resistance - 1) <= .03, (omega, z)
    assert abs(z.imag - reactance) <= 5., (omega, z)


@pytest.mark.parametrize("omega", sorted(KING_RESONANCE))
def test_first_resonance_matches_king_table(omega):
    beta_h_res, resistance = KING_RESONANCE[omega]
    beta_h = brentq(lambda value: _king_dipole(omega, value).impedance.imag, 1.40, 1.56, xtol=1e-6)
    assert abs(beta_h / beta_h_res - 1) <= .01, (omega, beta_h)
    assert abs(_king_dipole(omega, beta_h).impedance.real / resistance - 1) <= .02, omega


@pytest.mark.parametrize("omega,beta_h", sorted(KING_FULL_WAVE))
def test_full_wave_dipole_conductance_matches_king_table(omega, beta_h):
    conductance = (1 / KING_FULL_WAVE[omega, beta_h]).real
    z = _king_dipole(omega, beta_h).impedance
    assert abs((1 / z).real / conductance - 1) <= .03, (omega, beta_h, z)


@pytest.mark.parametrize("kind", ["dipole", "loop"])
def test_segment_doubling_finite_gap(kind):
    """Adjacent discretizations must change finite-gap Z by <=0.5%."""
    function = dipole if kind == "dipole" else loop
    geometry = dict(length=.47) if kind == "dipole" else dict(circumference=1.)
    coarse = function(**geometry, radius=1/800, frequency=C0, gap=.025, segments=320).impedance
    fine = function(**geometry, radius=1/800, frequency=C0, gap=.025, segments=640).impedance
    assert abs(coarse-fine)/abs(fine) <= .005


@pytest.mark.parametrize("kind", ["dipole", "loop"])
def test_finite_gap_to_mesh_regularized_delta(kind):
    """At a fixed basis count, shrinking the exact physical g approaches delta."""
    function = dipole if kind == "dipole" else loop
    geometry = dict(length=.5) if kind == "dipole" else dict(circumference=1.)
    def impedance(g):
        return function(**geometry, radius=1/800, frequency=C0, gap=g, segments=160).impedance
    delta = impedance(0)
    error = [abs(impedance(g)-delta) for g in (.025, .0125, .001, .0001)]
    assert all(a > b for a, b in zip(error, error[1:]))
    assert error[-1]/abs(delta) < .005


@pytest.mark.parametrize("radius", [1/800, 1/2000])
def test_both_spec_radii_are_inside_kernel_domain(radius):
    assert np.isfinite(dipole(length=.47, radius=radius, frequency=C0, segments=80).impedance)
    assert np.isfinite(loop(circumference=1, radius=radius, frequency=C0, segments=80).impedance)


@pytest.mark.parametrize("kwargs", [{"radius": .003}, {"gap": .5}, {"segments": 17}])
def test_reject_invalid_or_outside_model_range(kwargs):
    parameters = dict(length=.5, radius=1/800, frequency=C0)
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        dipole(**parameters)


def test_fdtd_loop_polygon_segment_doubling():
    """The feed-matched straight-chord contour also passes adjacent refinement."""
    from scripts.dev.fdtd.thin_wire_ladder.common import Run, wire_geometry, F0
    from tests.support.thin_wire_mom import loop_path
    run = Run("L1", 1., "G1", "gap")
    geometry, negative, _ = wire_geometry(run)
    coarse = loop_path(points=geometry["wire"], feed_segment=negative[1], radius=run.radius,
                       frequency=F0, segments=160).impedance
    fine = loop_path(points=geometry["wire"], feed_segment=negative[1], radius=run.radius,
                     frequency=F0, segments=320).impedance
    assert abs(coarse-fine)/abs(fine) <= .005
