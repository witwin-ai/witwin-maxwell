"""Public Scene -> Simulation -> Result exterior PointDipole MoM solves (CUDA)."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.fields import FieldQueryConfig
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.physics import incident_fields
from witwin.maxwell.mom.dipole import point_dipole_fields

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')

_spec = importlib.util.spec_from_file_location(
    'dipole_sphere_series', Path(__file__).parent/'reference'/'dipole_sphere_series.py')
series = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(series)

RADIUS = 0.1
DIRECTION = np.array([0.36, -0.48, 0.8])
MOMENT = np.array([0.3, 1.0, -0.4])/np.linalg.norm([0.3, 1.0, -0.4])
EXTERIOR_PROBES = np.array([[0.2, 0.05, -0.1], [-0.15, 0.12, 0.05], [0.02, -0.25, 0.03], [0.05, 0.1, -0.17]])
INTERIOR_PROBES = np.array([[0.03, 0.02, -0.04], [-0.05, 0.01, 0.02]])


def frequency_for(ka):
    return ka*series.C0/(2*math.pi*RADIUS)


def dipole(position, frequency, moment=MOMENT, **kwargs):
    return mw.PointDipole(position=tuple(position), polarization=tuple(moment), profile='ideal',
                          source_time=mw.CW(frequency=frequency), **kwargs)


def sphere_scene(source, eps_r=4.0, sigma_e=0.0, radius=RADIUS, monitors=()):
    return mw.Scene(structures=[mw.Structure(mw.Sphere(radius=radius), mw.Material(eps_r=eps_r, sigma_e=sigma_e),
                                             name='sphere')], sources=[source], monitors=list(monitors))


def cuda(points):
    return torch.tensor(points, device='cuda', dtype=torch.float64)


def relative(actual, expected):
    return float(np.abs(actual.cpu().numpy()-expected).max()/np.abs(expected).max())


def test_cuda_incident_fields_match_host_closed_form_and_carrier():
    frequency = frequency_for(1.0)
    source = dipole((0.0, 0.0, 0.3), frequency)
    scene = sphere_scene(source)
    config = mw.MoMConfig(subdivisions=1, riesz_storage='compact_rwg_v1', integration=REFERENCE_INTEGRATION)
    prepared = mw.Simulation.mom(scene, frequency=frequency, config=config).prepare()
    handle = prepared.solver.handle
    points = cuda(EXTERIOR_PROBES)
    e, h = incident_fields(source, points, handle.exterior)
    host = point_dipole_fields(points.cpu(), source.position, tuple(MOMENT), handle.exterior)
    torch.testing.assert_close(e.cpu(), host[0], rtol=1e-13, atol=0)
    torch.testing.assert_close(h.cpu(), host[1], rtol=1e-13, atol=0)
    carrier = handle.incident_carrier.evaluate(points, handle.exterior)
    torch.testing.assert_close(carrier[0], e, rtol=0, atol=0)
    torch.testing.assert_close(carrier[1], h, rtol=0, atol=0)
    with pytest.raises(ValueError, match='singular'):
        handle.incident_carrier.evaluate(cuda([source.position]), handle.exterior)


@pytest.mark.parametrize('position, subdivisions, match', [
    ((0.0, 0.0, 0.0), 2, 'inside'),
    ((0.0, 0.0, 0.09), 2, 'inside'),
    ((0.0, 0.0, 0.15), 1, 'panel radii'),
    ((0.0, 0.0, 0.11), 2, 'panel radii'),
])
def test_public_preparation_fails_closed_for_unsupported_dipole_geometry(position, subdivisions, match):
    frequency = frequency_for(1.0)
    scene = sphere_scene(dipole(position, frequency))
    with pytest.raises(ValueError, match=match):
        mw.Simulation.mom(scene, frequency=frequency, config=mw.MoMConfig(subdivisions=subdivisions)).prepare()


def test_public_preparation_fails_closed_for_surface_vertex_dipole():
    from witwin.maxwell.mom.runtime import _compile_basis
    frequency = frequency_for(1.0)
    scene = sphere_scene(dipole((0.0, 0.0, 0.5), frequency))
    basis = _compile_basis(scene, scene.structures[0].geometry, mw.MoMConfig())
    vertex = tuple(basis.mesh.vertices[0].tolist())
    with pytest.raises(ValueError, match='on the surface'):
        mw.Simulation.mom(sphere_scene(dipole(vertex, frequency)), frequency=frequency).prepare()


@pytest.mark.parametrize('config, match', [
    (lambda: mw.MoMConfig(operator_backend='hmatrix', riesz_storage='local_rwg_v1',
                          stopping=StoppingRule(method='gmres')), 'linear dense free-space'),
    (lambda: mw.MoMConfig(operator_backend='fmm', riesz_storage='local_rwg_v1',
                          stopping=StoppingRule(method='gmres', preconditioner='none')), 'linear dense free-space'),
    (lambda: mw.MoMConfig(integration=IntegrationConfig(order=3, singular_order=4)), 'order'),
])
def test_public_preparation_fails_closed_for_unsupported_configurations(config, match):
    frequency = frequency_for(1.0)
    scene = sphere_scene(dipole((0.0, 0.0, 0.3), frequency))
    with pytest.raises(ValueError, match=match):
        mw.Simulation.mom(scene, frequency=frequency, config=config()).prepare()


def test_vacuum_bodies_scatter_nothing_and_monitors_report_the_incident_dipole():
    frequency = frequency_for(1.0)
    source = dipole((0.0, 0.05, 0.0), frequency)
    scene = mw.Scene(structures=[
        mw.Structure(mw.Sphere(radius=.07, position=(-.2, 0., 0.)), mw.Material(eps_r=1.), name='left'),
        mw.Structure(mw.Sphere(radius=.06, position=(.2, 0., 0.)), mw.Material(eps_r=1.), name='right')],
        sources=[source], monitors=[mw.PointMonitor(name='inside_right', position=(.2, 0.01, 0.), fields=('Ex', 'Ez')),
                                    mw.PointMonitor(name='outside', position=(0., -.2, .1), fields=('Ey', 'Hz'))])
    config = mw.MoMConfig(subdivisions=1, query_config=FieldQueryConfig(adaptive=False), riesz_storage='compact_rwg_v1',
                          integration=REFERENCE_INTEGRATION)
    result = mw.Simulation.mom(scene, frequency=frequency, config=config).run()
    data = result.surface_currents
    assert torch.count_nonzero(data.scattered_coefficients) == 0
    assert data.incident_carrier.dipoles and not data.incident_carrier.waves
    for name, position in (('inside_right', (.2, 0.01, 0.)), ('outside', (0., -.2, .1))):
        e, h = point_dipole_fields(cuda([position]), source.position, tuple(MOMENT), data.handle.exterior)
        values = dict(zip(('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'), torch.cat((e[0], h[0]))))
        for component in result.monitors[name]['fields']:
            torch.testing.assert_close(result.monitors[name][component], values[component], rtol=1e-12, atol=0)


def _two_body_scattered(source_position, source_moment, probe, frequency):
    scene = mw.Scene(structures=[
        mw.Structure(mw.Sphere(radius=.08, position=(-.12, 0., 0.)), mw.Material(eps_r=4., sigma_e=0.01), name='lossy'),
        mw.Structure(mw.Ellipsoid(radii=(.05, .07, .06), position=(.13, .02, 0.)), mw.Material(eps_r=2.5), name='clear')],
        sources=[dipole(source_position, frequency, moment=source_moment)])
    config = mw.MoMConfig(subdivisions=2, riesz_storage='compact_rwg_v1', integration=REFERENCE_INTEGRATION)
    result = mw.Simulation.mom(scene, frequency=frequency, config=config).run()
    e, _ = result.query_fields(cuda([probe]), region_id=0, field_kind='scattered')
    return e[0]


def test_two_body_scattered_field_is_reciprocal():
    frequency = frequency_for(0.8)
    r1, p1 = (0.0, 0.16, 0.05), np.array([0.2, 1.0, -0.3])
    r2, p2 = (0.02, -0.14, -0.12), np.array([1.0, 0.1, 0.4])
    forward = (_two_body_scattered(r1, p1, r2, frequency).cpu().numpy()*p2/np.linalg.norm(p2)).sum()
    backward = (_two_body_scattered(r2, p2, r1, frequency).cpu().numpy()*p1/np.linalg.norm(p1)).sum()
    assert abs(forward-backward) < 2e-3*abs(forward)


# Interior rules at singular_order=4 match the default 8 to four digits here
# while keeping the refinement study within a few minutes of CUDA time.
REFERENCE_INTEGRATION = IntegrationConfig(order=4, singular_order=4)


def lossy_sigma(ka, imaginary_epsilon=1.0):
    return imaginary_epsilon*2*math.pi*frequency_for(ka)*series.EPS0


def _sphere_case(ka, distance, eps_r, sigma_e, subdivisions):
    frequency = frequency_for(ka)
    position = distance*RADIUS*DIRECTION
    config = mw.MoMConfig(subdivisions=subdivisions, riesz_storage='compact_rwg_v1', integration=REFERENCE_INTEGRATION)
    result = mw.Simulation.mom(sphere_scene(dipole(position, frequency), eps_r, sigma_e), frequency=frequency,
                               config=config).run()
    assert result.solver_stats['converged']
    epsilon = eps_r+1j*sigma_e/(2*math.pi*frequency*series.EPS0)
    reference = series.DipoleSphereSeries(radius=RADIUS, frequency=frequency, epsilon_r=epsilon,
                                          position=position, moment=MOMENT, n_max=60)
    e_s, h_s = result.query_fields(cuda(EXTERIOR_PROBES), region_id=0, field_kind='scattered')
    e_i, _ = result.query_fields(cuda(INTERIOR_PROBES), region_id='interior', field_kind='total')
    ref_s, ref_h = reference.scattered(EXTERIOR_PROBES)
    ref_i, _ = reference.total(INTERIOR_PROBES)
    return max(relative(e_s, ref_s), relative(h_s, ref_h)), relative(e_i, ref_i)


@pytest.mark.parametrize('ka, distance, lossy, levels', [
    (1.0, 2.5, False, (1, 2, 3)),
    (0.5, 2.2, True, (1, 2)),
    (1.0, 1.6, True, (2, 3)),
])
def test_sphere_matches_series_reference_under_mesh_refinement(ka, distance, lossy, levels):
    sigma = lossy_sigma(ka) if lossy else 0.0
    errors = [_sphere_case(ka, distance, 4.0, sigma, level) for level in levels]
    print('dipole-sphere relative errors', ka, distance, lossy, dict(zip(levels, errors)))
    for coarse, fine in zip(errors, errors[1:]):
        # Flat-panel geometry and RWG errors are second order: about 4x per midpoint subdivision.
        assert fine[0] < coarse[0]/3 and fine[1] < coarse[1]/2.5
    bound = {2: 0.05, 3: 0.015}[levels[-1]]
    assert errors[-1][0] < bound and errors[-1][1] < bound


def test_dipole_result_roundtrip_restores_the_exact_carrier(tmp_path):
    frequency = frequency_for(1.0)
    position = 2.5*RADIUS*DIRECTION
    scene = sphere_scene(dipole(position, frequency), 4.0, lossy_sigma(1.0))
    config = mw.MoMConfig(subdivisions=1, riesz_storage='compact_rwg_v1', integration=REFERENCE_INTEGRATION)
    result = mw.Simulation.mom(scene, frequency=frequency, config=config).run()
    data = result.surface_currents
    path = tmp_path/'dipole.pt'
    result.save(path)
    restored = mw.Result.load(path, scene=scene).surface_currents
    assert restored.incident_carrier.identity == data.incident_carrier.identity
    assert torch.equal(restored.scattered_coefficients, data.scattered_coefficients)
    points = cuda([[0.0, 0.3, 0.2], [0.01, 0.02, 0.03]])
    for region, point in ((0, points[:1]), ('interior', points[1:])):
        for a, b in zip(data.query_fields(point, region, 'total'), restored.query_fields(point, region, 'total')):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    moved = sphere_scene(dipole(position+np.array([0.0, 0.0, 1e-3]), frequency), 4.0, lossy_sigma(1.0))
    with pytest.raises(ValueError):
        mw.Result.load(path, scene=moved)
