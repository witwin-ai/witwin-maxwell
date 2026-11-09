import numpy as np
from benchmark.sie_reference import sphere_coefficients, sphere_cross_sections, sphere_far_field


def test_sphere_oracle_no_contrast_and_lossless_balance():
    a, b = sphere_coefficients(2, 1)
    assert np.max(np.abs(a))+np.max(np.abs(b)) < 1e-14
    data = sphere_cross_sections(1, 2, 2.25)
    assert abs(data["absorption"]) < 1e-12
    assert sphere_cross_sections(1, 2, 2.25+.3j)["absorption"] > 0


def test_sphere_rayleigh_limit_and_complex_forward_optical_theorem():
    x = .001
    data = sphere_cross_sections(1, x, 2.25)
    expected = (8*np.pi/3)*x**4*abs((2.25-1)/(2.25+2))**2
    assert abs(data["scattering"]/expected-1) < 2e-6
    k = 2.
    data = sphere_cross_sections(1, k, 2.25+.3j)
    forward = sphere_far_field(1, k, 2.25+.3j, [[0, 0, 1]])
    assert abs(4*np.pi/k*forward[0, 0].imag-data["extinction"]) < 1e-12


def test_sphere_angular_power_matches_independent_partial_wave_sum():
    mu, weights = np.polynomial.legendre.leggauss(30)
    phi = np.arange(64)*2*np.pi/64
    z, p = np.meshgrid(mu, phi, indexing="ij")
    directions = np.stack((np.sqrt(1-z*z)*np.cos(p), np.sqrt(1-z*z)*np.sin(p), z), -1).reshape(-1, 3)
    far = sphere_far_field(1, 2, 2.25+.3j, directions)
    integral = np.sum((np.abs(far)**2).sum(-1).reshape(30, 64)*weights[:, None])*2*np.pi/64
    assert abs(integral-sphere_cross_sections(1, 2, 2.25+.3j)["scattering"]) < 1e-12


def test_near_oracle_no_contrast_axes_and_origin():
    from benchmark.sie_reference import sphere_fields
    points = np.array([[0, 0, 0], [0, 0, .3], [0, 0, -.3], [.2, .1, -.4], [1.2, .4, .1]])
    e, h = sphere_fields(1., 2., 1., points)
    ei, hi = sphere_fields(1., 2., 1., points, field_kind="incident")
    np.testing.assert_allclose(e, ei, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(h, hi, atol=2e-15, rtol=2e-13)


def test_near_oracle_far_limit_matches_complex_angular_oracle():
    from benchmark.sie_reference import sphere_fields
    directions = np.array([[.3, .4, .5], [0., 0., 1.], [0., 0., -1.], [1., 0., 0.]])
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    distance, k, eta = 1e6, 2., 376.730313668
    e, h = sphere_fields(1., k, 2.25+.3j, distance*directions, field_kind="scattered")
    far = sphere_far_field(1., k, 2.25+.3j, directions)
    np.testing.assert_allclose(distance*np.exp(-1j*k*distance)*e, far, rtol=3e-6, atol=2e-6)
    np.testing.assert_allclose(h, np.cross(directions, e)/eta, rtol=5e-6, atol=1e-14)


def test_near_oracle_maxwell_curls_and_transmission_conditions():
    from benchmark.sie_reference import sphere_fields
    k, eta, eps = 2., 376.730313668, 2.25+.3j
    points = np.array([[.2, .3, -.4], [1.2, -.3, .4]])
    delta = 1e-5
    jac_e, jac_h = [], []
    for axis in np.eye(3):
        ep, hp = sphere_fields(1., k, eps, points+delta*axis)
        em, hm = sphere_fields(1., k, eps, points-delta*axis)
        jac_e.append((ep-em)/(2*delta))
        jac_h.append((hp-hm)/(2*delta))
    curl = lambda j: np.stack((j[1][:, 2]-j[2][:, 1], j[2][:, 0]-j[0][:, 2], j[0][:, 1]-j[1][:, 0]), axis=-1)
    e, h = sphere_fields(1., k, eps, points)
    np.testing.assert_allclose(curl(jac_e), 1j*k*eta*h, rtol=2e-8, atol=1e-9)
    np.testing.assert_allclose(curl(jac_h), -1j*k/eta*np.array([eps, 1.])[:, None]*e, rtol=2e-8, atol=1e-11)
    directions = np.array([[.3, .4, .5], [-.2, .4, -.3], [0., 0., 1.]])
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    eo, ho = sphere_fields(1., k, eps, directions*(1+1e-8))
    ei, hi = sphere_fields(1., k, eps, directions*(1-1e-8))
    np.testing.assert_allclose(np.cross(directions, eo), np.cross(directions, ei), rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(np.cross(directions, ho), np.cross(directions, hi), rtol=2e-7, atol=2e-10)
    np.testing.assert_allclose(np.sum(eo*directions, -1), eps*np.sum(ei*directions, -1), rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(np.sum(ho*directions, -1), np.sum(hi*directions, -1), rtol=2e-7, atol=2e-10)


def test_internal_oracle_volume_absorption_matches_partial_wave_cross_section():
    from benchmark.sie_reference import sphere_fields
    mu, wmu = np.polynomial.legendre.leggauss(12)
    nodes, wr = np.polynomial.legendre.leggauss(12)
    radial = (nodes+1)/2
    phi = np.arange(24)*np.pi/12
    rr, z, p = np.meshgrid(radial, mu, phi, indexing="ij")
    points = np.stack((rr*np.sqrt(1-z*z)*np.cos(p), rr*np.sqrt(1-z*z)*np.sin(p), rr*z), -1).reshape(-1, 3)
    eps, k = 2.25+.3j, 2.
    e, _ = sphere_fields(1., k, eps, points)
    intensity = np.sum(np.abs(e)**2, -1).reshape(12, 12, 24)
    integral = np.sum(intensity*(radial**2*wr/2)[:, None, None]*wmu[None, :, None])*np.pi/12
    cross_section = k*eps.imag*integral
    np.testing.assert_allclose(cross_section, sphere_cross_sections(1., k, eps)["absorption"], rtol=2e-12)


def test_actual_mom_electric_and_magnetic_fields_refine_toward_independent_oracle():
    import pytest
    import torch
    import witwin.maxwell as mw
    from benchmark.sie_reference import sphere_fields
    from witwin.maxwell.mom.operator import IntegrationConfig
    if not torch.cuda.is_available():
        pytest.skip("Actual CUDA MoM is required")
    radius, frequency, epsilon = .1, 3e8, 2.25+.1j
    directions = np.array([[1., .3, .2], [-.2, 1., .4], [.3, -.5, 1.]])
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    errors = []
    for level in (0, 1):
        scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=radius), mw.Material(
            eps_r=epsilon.real, sigma_e=epsilon.imag*2*np.pi*frequency*mw.constants.EPSILON_0))], sources=[mw.PlaneWave()])
        result = mw.Simulation.mom(scene, frequency=frequency, config=mw.MoMConfig(
            subdivisions=level, integration=IntegrationConfig(order=4, singular_order=6))).run()
        assert result.stats()["converged"]
        medium = result.surface_currents.handle.exterior
        level_errors = []
        for factor, region, kind in ((.3, "interior", "total"), (1.5, "exterior", "total"), (1.5, "exterior", "scattered")):
            points = radius*factor*directions
            numerical = result.query_fields(torch.as_tensor(points, device="cuda", dtype=torch.float64), region_id=region, field_kind=kind)
            reference = sphere_fields(radius, medium.k.real, epsilon, points, eta=medium.eta.real, field_kind=kind)
            level_errors.extend(np.linalg.norm(value.detach().cpu().numpy()-expected)/np.linalg.norm(expected)
                                for value, expected in zip(numerical, reference))
        errors.append(np.array(level_errors))
    assert np.all(errors[1] < errors[0]), errors
    assert np.max(errors[1]) < .2, errors
