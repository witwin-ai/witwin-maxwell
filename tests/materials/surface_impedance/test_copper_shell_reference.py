"""Independent full-Maxwell oracle controls, not production solver tests."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import spherical_jn, spherical_yn


_path = Path(__file__).resolve().parents[3]/"scripts/dev/fdtd/copper_shell_reference.py"
_spec = importlib.util.spec_from_file_location("copper_shell_reference", _path)
oracle = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = oracle
_spec.loader.exec_module(oracle)


@pytest.mark.parametrize("outgoing", [False, True])
def test_scaled_integer_hankel_matches_independent_scipy(outgoing):
    z = np.array((1.2+.3j, 4+2j, 30+15j))
    sign = 1 if outgoing else -1
    # A small anchor keeps the independent SciPy path measurable and avoids
    # cancellation of j +/- i*y at a large imaginary argument.
    z = z[:2] if outgoing else z
    anchor = .7+.2j
    for order in range(11):
        expected = spherical_jn(order, z)+sign*1j*spherical_yn(order, z)
        derivative = spherical_jn(order, z, derivative=True)+sign*1j*spherical_yn(order, z, derivative=True)
        f, d = oracle.scaled_hankel(order, z, anchor, outgoing=outgoing)
        scale = np.exp(-sign*1j*anchor)
        np.testing.assert_allclose(f, expected*scale, rtol=3e-12, atol=2e-13)
        np.testing.assert_allclose(d, (expected/z+derivative)*scale, rtol=3e-12, atol=2e-13)


def test_vacuum_shell_is_transparent_at_both_interfaces():
    solution = oracle.conductive_shell(.02, .004, 3e9, 0.)
    assert max(abs(solution.a).max(), abs(solution.b).max()) < 2e-14
    np.testing.assert_allclose(solution.interior_te, 1, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(solution.interior_tm, 1, rtol=2e-13, atol=2e-13)
    assert solution.boundary_residual < 2e-15
    points = np.array(((.003, .004, .005), (.002, -.006, -.003)))
    e, h = oracle.vacuum_region_fields(solution, points, region="interior")
    phase = np.exp(1j*2*np.pi*solution.frequency/oracle.C_0*points[:, 2])
    expected_e, expected_h = np.zeros_like(e), np.zeros_like(h)
    expected_e[:, 0], expected_h[:, 1] = phase, phase/np.sqrt(oracle.MU_0/oracle.EPS_0)
    np.testing.assert_allclose(e, expected_e, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(h, expected_h, rtol=2e-13, atol=2e-15)


@pytest.mark.parametrize("skin_depths", [.01, .1, 1., 3., 10., 30.])
def test_real_copper_shell_is_passive_finite_and_resolves_inner_transmission(skin_depths):
    frequency, sigma = 3e9, 5.8e7
    delta = np.sqrt(2/(2*np.pi*frequency*oracle.MU_0*sigma))
    solution = oracle.conductive_shell(.02, delta*skin_depths, frequency, sigma)
    for value in (solution.a, solution.b, solution.interior_te, solution.interior_tm,
            solution.shell_te, solution.shell_tm):
        assert np.isfinite(value).all()
    assert solution.boundary_residual < 3e-15
    power = solution.cross_sections
    assert 0 < power["absorption"] < power["extinction"]
    if skin_depths <= .1:
        assert max(abs(solution.interior_te[0]), abs(solution.interior_tm[0])) > 1e-6
    if skin_depths == 30:
        outer = solution.radius+solution.thickness/2
        orders = np.arange(1, len(solution.a)+1)
        x = 2*np.pi*frequency/oracle.C_0*outer
        j = spherical_jn(orders, x)
        h = j+1j*spherical_yn(orders, x)
        dj = j+x*spherical_jn(orders, x, derivative=True)
        dh = dj+1j*(spherical_yn(orders, x)+x*spherical_yn(orders, x, derivative=True))
        assert np.linalg.norm(solution.a-dj/dh)/np.linalg.norm(dj/dh) < 2e-4
        assert np.linalg.norm(solution.b-j/h)/np.linalg.norm(j/h) < 2e-4
        # Predeclared absolute floor: 30 skin depths is deliberately opaque.
        assert max(abs(solution.interior_te).max(), abs(solution.interior_tm).max()) < 2e-12


@pytest.mark.parametrize("skin_depths", [.01, .1, 1., 3., 10., 30.])
def test_two_independent_sheet_channels_approximate_physical_shell_with_declared_curvature_error(skin_depths):
    frequency, sigma = 3e9, 5.8e7
    omega = 2*np.pi*frequency
    delta = np.sqrt(2/(omega*oracle.MU_0*sigma))
    thickness = skin_depths*delta
    exact = oracle.conductive_shell(.02, thickness, frequency, sigma)
    gamma = np.sqrt(-1j*omega*oracle.MU_0*sigma)
    eta = np.sqrt(-1j*omega*oracle.MU_0/sigma)
    y = 2*np.tanh(gamma*thickness/2)/eta
    z = 2*eta*np.tanh(gamma*thickness/2)
    a, b, te, tm = oracle.sheet_sphere_coefficients(.02, frequency, y, z, cutoff=len(exact.a))
    # A declared 0.3% bound includes genuine physical curvature/thickness error;
    # no radius or constitutive value is fitted to the full-shell reference.
    assert np.linalg.norm(a-exact.a)/np.linalg.norm(exact.a) < .003
    assert np.linalg.norm(b-exact.b)/np.linalg.norm(exact.b) < .003
    # Interior near fields are measurable through ten skin depths. At thirty
    # depths use an absolute field floor rather than divide by opaque noise.
    from dataclasses import replace
    sheet = replace(exact, a=a, b=b, interior_te=te, interior_tm=tm)
    points = np.array(((.003, .004, .005), (.002, -.006, -.003)))
    expected_e, expected_h = oracle.vacuum_region_fields(exact, points, region="interior")
    actual_e, actual_h = oracle.vacuum_region_fields(sheet, points, region="interior")
    eta0 = np.sqrt(oracle.MU_0/oracle.EPS_0)
    for reference, actual in ((expected_e, actual_e), (expected_h*eta0, actual_h*eta0)):
        if skin_depths == 30:
            assert np.linalg.norm(actual) < 2e-12
        else:
            assert np.linalg.norm(actual-reference)/np.linalg.norm(reference) < .01
