"""CPU validation of the Berreman 4x4 oracle (benchmark/analytic/berreman.py)."""
import numpy as np
import pytest

from benchmark.analytic import berreman as B

F = 2.0e9
K0 = 2 * np.pi * F / B.C0


def _airy(r12, phase):
    return r12 * (1 - phase) / (1 - r12**2 * phase)


@pytest.mark.parametrize("theta", [0.0, 0.3, 0.9, 1.4])
@pytest.mark.parametrize("eps", [2.89, 4.0 + 0.4j])
def test_isotropic_slab_matches_airy_formula(theta, eps):
    n, d = np.sqrt(eps), 0.1
    response = B.solve([B.Layer(d, eps)], F, theta=theta)
    c = np.cos(theta)
    q = np.sqrt(eps - np.sin(theta) ** 2)
    phase = np.exp(2j * q * K0 * d)
    r_s = _airy((c - q) / (c + q), phase)
    r_p = _airy((eps * c - q) / (eps * c + q), phase)
    assert response.reflectance[1, 1] == pytest.approx(abs(r_s) ** 2, rel=1e-10, abs=1e-14)
    assert response.reflectance[0, 0] == pytest.approx(abs(r_p) ** 2, rel=1e-10, abs=1e-14)
    assert abs(response.r[0, 1]) < 1e-12 and abs(response.r[1, 0]) < 1e-12
    if np.imag(eps) == 0:
        np.testing.assert_allclose(response.reflectance.sum(0) + response.transmittance.sum(0), 1, atol=1e-12)


def test_rotated_uniaxial_normal_incidence_is_two_scalar_slabs():
    # In-plane tensor: each eigenpolarization sees an independent Airy slab.
    tensor = np.array([[3.0, 0.2, 0.0], [0.2, 2.5, 0.0], [0.0, 0.0, 2.0]])
    d = 0.1
    response = B.solve([B.Layer(d, tensor)], F)
    values, vectors = np.linalg.eigh(tensor[:2, :2])
    for value, vector in zip(values, vectors.T):
        n = np.sqrt(value)
        r = _airy((1 - n) / (1 + n), np.exp(2j * n * K0 * d))
        reflected = response.r @ vector
        assert np.allclose(reflected, r * vector, atol=1e-12)
    np.testing.assert_allclose(response.reflectance.sum(0) + response.transmittance.sum(0), 1, atol=1e-12)


def test_full_tensor_slab_benchmark_eigenmode_values():
    tensor = np.array([[3.0, 0.2, 0.0], [0.2, 2.5, 0.0], [0.0, 0.0, 2.0]])
    jones = np.array([0.9436283192, 0.3310069414])
    response = B.solve([B.Layer(0.1, tensor)], F)
    transmitted = response.t @ jones
    reflected = response.r @ jones
    assert np.sum(np.abs(transmitted) ** 2) == pytest.approx(0.789894, abs=2e-6)
    assert np.sum(np.abs(reflected) ** 2) == pytest.approx(0.210106, abs=2e-6)
    # The launch is an eigenpolarization: no cross-polarized output.
    assert abs(transmitted[0] * jones[1] - transmitted[1] * jones[0]) < 1e-9


@pytest.mark.parametrize("theta", [0.2, 0.7])
def test_general_tensor_energy_reciprocity_and_layer_split(theta):
    rng = np.random.default_rng(1)
    a = rng.normal(size=(3, 3))
    rotation, _ = np.linalg.qr(a)
    tensor = rotation @ np.diag([2.0, 3.0, 4.5]) @ rotation.T
    one = B.solve([B.Layer(0.08, tensor)], F, theta=theta)
    split = B.solve([B.Layer(0.03, tensor), B.Layer(0.05, tensor)], F, theta=theta)
    np.testing.assert_allclose(one.r, split.r, atol=1e-12)
    np.testing.assert_allclose(one.t, split.t, atol=1e-12)
    np.testing.assert_allclose(one.reflectance.sum(0) + one.transmittance.sum(0), 1, atol=1e-12)
    # Full tensors couple p and s even at oblique incidence.
    assert one.reflectance[0, 1] > 1e-6


def test_multilayer_quarter_wave_stack_and_substrate():
    n_h, n_l = 2.0, 1.5
    lam = B.C0 / F
    layers = []
    for _ in range(6):
        layers += [B.Layer(lam / (4 * n_h), n_h**2), B.Layer(lam / (4 * n_l), n_l**2)]
    response = B.solve(layers, F, n_out=1.5)
    # Quarter-wave mirror: R = ((1 - Y)/(1 + Y))^2, Y = (n_h/n_l)^(2N) n_out/n_in... admittance form.
    y = (n_h / n_l) ** 12 * 1.5
    assert response.reflectance[0, 0] == pytest.approx(((1 - y) / (1 + y)) ** 2, rel=1e-10)
    assert response.reflectance[0, 0] + response.transmittance[0, 0] == pytest.approx(1, abs=1e-12)


def test_fields_are_continuous_and_match_boundary_values():
    tensor = np.array([[3.0, 0.2, 0.1], [0.2, 2.5, 0.0], [0.1, 0.0, 2.0]])
    layers = [B.Layer(0.1, tensor)]
    z = np.array([-0.05 - 1e-12, -0.05 + 1e-12, 0.05 - 1e-12, 0.05 + 1e-12])
    for theta in (0.0, 0.4):
        e, h = B.fields(layers, F, z, z0=-0.05, jones=(0.8, 0.6), theta=theta)
        np.testing.assert_allclose(e[0, :2], e[1, :2], atol=1e-9)
        np.testing.assert_allclose(h[0], h[1], atol=1e-9)
        np.testing.assert_allclose(e[2, :2], e[3, :2], atol=1e-9)
        np.testing.assert_allclose(h[2], h[3], atol=1e-9)
        # Normal D is continuous: D_z(in) = (eps E)_z, D_z(out) = E_z.
        d_in = tensor[2] @ e[1]
        assert d_in == pytest.approx(e[0, 2], abs=1e-9)
    e, _ = B.fields(layers, F, np.array([-0.3, 0.3]), z0=-0.05, jones=(1.0, 0.0))
    response = B.solve(layers, F)
    assert abs(e[1, 0]) ** 2 + abs(e[1, 1]) ** 2 == pytest.approx(response.transmittance[:, 0].sum(), rel=1e-10)
