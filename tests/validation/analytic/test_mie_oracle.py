"""CPU validation of the Mie series oracle (benchmark/analytic/mie.py)."""
import numpy as np
import pytest

import witwin.maxwell as mw
from benchmark.analytic import mie
from benchmark.sie_reference import sphere_fields


def _sphere_directions(order=40, azimuths=64):
    mu, weights = np.polynomial.legendre.leggauss(order)
    phi = np.arange(azimuths) * 2 * np.pi / azimuths
    z, p = np.meshgrid(mu, phi, indexing="ij")
    s = np.sqrt(1 - z * z)
    directions = np.stack((s * np.cos(p), s * np.sin(p), z), -1).reshape(-1, 3)
    return directions, np.repeat(weights, azimuths) * 2 * np.pi / azimuths


def test_published_bohren_huffman_case():
    # Bohren and Huffman (1983) appendix A sample: x = 5.213, m = 1.55.
    x = 2 * np.pi * 0.525 / 0.6328
    data = mie.cross_sections(x, 1.0, 1.55**2)
    geometric = np.pi * x**2
    assert data["extinction"] / geometric == pytest.approx(3.10543, abs=1e-5)
    assert data["scattering"] / geometric == pytest.approx(3.10543, abs=1e-5)
    assert data["backscatter"] / geometric == pytest.approx(2.92534, abs=1e-5)


@pytest.mark.parametrize("x", [0.1, 1.0, 5.0, 30.0, 60.0])
@pytest.mark.parametrize("eps", [2.0, 4.0, 3.1875, 4.0 + 0.5j, 2.0 + 120j])
def test_bessel_ratio_and_log_derivative_paths_agree(x, eps):
    a, b = mie.coefficients(x, eps)
    a2, b2 = mie.coefficients_log_derivative(x, eps)
    assert np.max(np.abs(a - a2)) < 1e-12
    assert np.max(np.abs(b - b2)) < 1e-12


@pytest.mark.parametrize("material", [mie.PEC, 4.0, 2.25 + 0.3j])
@pytest.mark.parametrize("x", [1.0, 25.0, 50.0])
def test_series_converges_in_cutoff(material, x):
    reference = mie.cross_sections(x, 1.0, material, cutoff=mie.default_cutoff(x) + 40)
    wiscombe = mie.cross_sections(x, 1.0, material, cutoff=mie.wiscombe_cutoff(x))
    default = mie.cross_sections(x, 1.0, material)
    for key in ("extinction", "scattering", "backscatter"):
        assert abs(default[key] / reference[key] - 1) < 1e-12
        assert abs(wiscombe[key] / reference[key] - 1) < 1e-6
    theta = np.linspace(0, np.pi, 181)
    far_ref = mie.bistatic_rcs(x, 1.0, material, theta, 0.0, cutoff=mie.default_cutoff(x) + 40)
    far = mie.bistatic_rcs(x, 1.0, material, theta, 0.0)
    assert np.max(np.abs(far - far_ref)) < 1e-10 * np.max(far_ref)


@pytest.mark.parametrize("material", [mie.PEC, 4.0, 2.25 + 0.3j, 2.0 + 120j])
@pytest.mark.parametrize("x", [0.7, 6.0])
def test_optical_theorem_and_angular_power(material, x):
    k = 2.0
    radius = x / k
    data = mie.cross_sections(radius, k, material)
    forward = mie.far_field(radius, k, material, [[0, 0, 1]])
    assert 4 * np.pi / k * forward[0, 0].imag == pytest.approx(data["extinction"], rel=1e-12)
    directions, weights = _sphere_directions(order=int(2 * x) + 40)
    power = np.sum(np.sum(np.abs(mie.far_field(radius, k, material, directions)) ** 2, -1) * weights)
    assert power == pytest.approx(data["scattering"], rel=1e-10)
    back = mie.bistatic_rcs(radius, k, material, np.pi, 0.0)
    assert back == pytest.approx(data["backscatter"], rel=1e-10)
    if material == mie.PEC or np.imag(material) == 0:
        assert abs(data["absorption"]) < 1e-12 * data["extinction"]
    else:
        assert data["absorption"] > 0


def test_pec_limits():
    for ka in (1e-3, 1e-2):
        data = mie.cross_sections(ka, 1.0, mie.PEC)
        geometric = np.pi * ka**2
        # Rayleigh PEC sphere: sigma_back = 9 pi a^2 (ka)^4, Q_sca = 10/3 (ka)^4.
        assert data["backscatter"] / geometric == pytest.approx(9 * ka**4, rel=5 * ka**2)
        assert data["scattering"] / geometric == pytest.approx(10 / 3 * ka**4, rel=5 * ka**2)
    data = mie.cross_sections(1.0, 1.0, mie.PEC)
    assert data["backscatter"] / np.pi == pytest.approx(3.6376, abs=1e-4)
    data = mie.cross_sections(200.0, 1.0, mie.PEC)
    geometric = np.pi * 200.0**2
    assert data["backscatter"] / geometric == pytest.approx(1.0, abs=0.01)
    assert data["extinction"] / geometric == pytest.approx(2.0, abs=0.02)


def test_pec_near_field_boundary_condition_and_far_limit():
    rng = np.random.default_rng(3)
    radius, k = 1.3, 2.0
    normals = rng.normal(size=(40, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    e, h = mie.near_fields(radius, k, mie.PEC, normals * radius * (1 + 1e-10))
    tangential_e = e - np.sum(e * normals, -1, keepdims=True) * normals
    assert np.max(np.abs(tangential_e)) < 1e-8
    assert np.max(np.abs(np.sum(h * normals, -1))) < 1e-8 / mie.ETA0
    distance = 1e6
    e, h = mie.near_fields(radius, k, mie.PEC, distance * normals, field_kind="scattered")
    far = mie.far_field(radius, k, mie.PEC, normals)
    np.testing.assert_allclose(distance * np.exp(-1j * k * distance) * e, far, rtol=3e-6, atol=2e-6)
    np.testing.assert_allclose(h, np.cross(normals, e) / mie.ETA0, rtol=5e-6, atol=1e-14)


def test_generic_exterior_expansion_matches_pinned_dielectric_oracle():
    rng = np.random.default_rng(5)
    radius, k, eps = 1.0, 3.0, 4.0 + 0.2j
    points = rng.normal(size=(30, 3))
    points *= (1.2 + rng.random((30, 1))) / np.linalg.norm(points, axis=1, keepdims=True)
    a, b = mie.coefficients(k * radius, eps)
    e, h = mie._exterior_scattered(radius, k, a, b, points, mie.ETA0)
    e_ref, h_ref = sphere_fields(radius, k, eps, points, field_kind="scattered")
    np.testing.assert_allclose(e, e_ref, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(h, h_ref, rtol=1e-12, atol=1e-16)


def test_dispersive_closed_forms_match_library_materials():
    f = 2.0e9
    lorentz = mw.Material.lorentz(eps_inf=2.0, delta_eps=3.0, resonance_frequency=2.0e9, gamma=0.05e9)
    assert mie.lorentz_permittivity(
        f, eps_inf=2.0, delta_eps=3.0, resonance_frequency=2.0e9, gamma=0.05e9
    ) == pytest.approx(lorentz.relative_permittivity(f), rel=1e-14)
    debye = mw.Material.debye(eps_inf=2.0, delta_eps=2.0, tau=5.0e-11)
    assert mie.debye_permittivity(f, eps_inf=2.0, delta_eps=2.0, tau=5.0e-11) == pytest.approx(
        debye.relative_permittivity(f), rel=1e-14)
    sellmeier = mw.Material.sellmeier(eps_inf=1.5, b_coefficients=(1.5,), c_coefficients=(2.5e-3,))
    assert mie.sellmeier_permittivity(
        f, eps_inf=1.5, b_coefficients=(1.5,), c_coefficients=(2.5e-3,)
    ) == pytest.approx(sellmeier.relative_permittivity(f), rel=1e-12)
    lossy = mw.Material(eps_r=3.0, sigma_e=0.05)
    assert mie.conductive_permittivity(f, eps_r=3.0, sigma_e=0.05) == pytest.approx(
        lossy.relative_permittivity(f), rel=1e-9)


@pytest.mark.parametrize("steps_per_period", [30, 90])
def test_discrete_lorentz_susceptibility_matches_the_time_stepped_update(steps_per_period):
    # Step the scalar J-P recurrence of the FDTD Lorentz kernel under a CW drive
    # and compare the steady phasor P/(eps0 E) with the closed form.
    f, f0, gamma, delta = 2.0e9, 2.0e9, 0.05e9, 3.0
    dt = 1 / (f * steps_per_period)
    w0, g = 2 * np.pi * f0, 2 * np.pi * gamma
    denominator = 2 + g * dt
    decay, restoring, drive = (2 - g * dt) / denominator, 2 * w0**2 * dt / denominator, 2 * delta * w0**2 * dt / denominator
    n = np.arange(steps_per_period * 4000)
    e = np.exp(-1j * 2 * np.pi * f * n * dt)
    p = np.zeros(n.size, dtype=complex)
    current = 0j
    for step in range(n.size - 1):
        current = decay * current - restoring * p[step] + drive * e[step]
        p[step + 1] = p[step] + dt * current
    tail = slice(-steps_per_period * 400, None)
    chi = np.vdot(e[tail], p[tail]) / np.vdot(e[tail], e[tail])
    expected = mie.lorentz_permittivity_discrete(
        f, dt, eps_inf=0.0, delta_eps=delta, resonance_frequency=f0, gamma=gamma)
    assert abs(chi - expected) < 1e-3 * abs(expected)
    continuous = mie.lorentz_permittivity(f, eps_inf=0.0, delta_eps=delta, resonance_frequency=f0, gamma=gamma)
    assert abs(expected - continuous) > 1e-3 * abs(continuous)
