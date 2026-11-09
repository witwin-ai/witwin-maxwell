"""Candidate physical result identity, native current semantics and persistence."""
import copy
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.carrier import IncidentCarrier
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.runtime import SurfaceData

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def solve(epsilon=1.01, *, monitors=()):
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.1), mw.Material(eps_r=epsilon))],
                     sources=[mw.PlaneWave()], monitors=list(monitors))
    config = mw.MoMConfig(subdivisions=0, riesz_storage='compact_rwg_v1',
                         integration=IntegrationConfig(order=4, singular_order=6))
    return scene, mw.Simulation.mom(scene, frequency=3e8, config=config).run()


def test_native_scattered_state_and_exact_snapshot(tmp_path):
    scene, result = solve()
    data = result.surface_currents
    assert data.J_scattered.shape == data.M_scattered.shape == (30,)
    for name in ('coefficients', 'J', 'M'):
        with pytest.raises(AttributeError, match='scattered|total'):
            getattr(data, name)
    points = torch.tensor([[.2, .1, .3]], device='cuda', dtype=torch.float64)
    inside = points*.05
    path = tmp_path/'lifted.pt'
    result.save(path)
    saved = torch.load(path, weights_only=False)['surface_data']
    assert saved['schema_version'] == 3 and saved['executable'] is True
    assert saved['body_table'][0]['J_slice'] == [0, 30]
    assert 'coefficients' not in saved and 'matrix' not in saved
    restored = mw.Result.load(path, scene=scene)
    assert torch.equal(restored.surface_currents.scattered_coefficients, data.scattered_coefficients)
    assert restored.surface_currents.incident_carrier == data.incident_carrier
    for queries, region, kind in ((points, 'exterior', 'total'), (inside, 'interior', 'total')):
        for original, loaded in zip(result.query_fields(queries, region_id=region, field_kind=kind),
                                    restored.query_fields(queries, region_id=region, field_kind=kind)):
            torch.testing.assert_close(original, loaded, rtol=1e-13, atol=1e-15)
    face_ids = torch.tensor([0], device='cuda')
    face_points = data.mesh.triangles[face_ids].mean(1)
    currents = data.evaluate_total_currents(face_points, face_indices=face_ids)
    assert all(torch.isfinite(current).all() for current in currents)
    data.scattered_coefficients = data.scattered_coefficients.detach().requires_grad_()
    energy = sum(value.abs().square().sum() for value in data.evaluate_total_currents(face_points, face_indices=face_ids))
    assert torch.isfinite(torch.autograd.grad(energy, data.scattered_coefficients)[0]).all()


@pytest.mark.parametrize('mutation', ['old_schema', 'archive', 'kind', 'carrier', 'body_order', 'slices', 'bool_slice', 'medium', 'decoder', 'integration', 'stopping', 'query'])
def test_snapshot_rejects_changed_representation_before_assembly(mutation):
    scene, result = solve()
    payload = copy.deepcopy(result.surface_currents.snapshot())
    if mutation == 'old_schema': payload['schema_version'] = 1
    elif mutation == 'archive': payload['executable'] = False
    elif mutation == 'kind': payload['current_kind'] = 'total_rwg_v1'
    elif mutation == 'carrier': payload['incident_carrier']['waves'][0]['amplitude'][0] = .7
    elif mutation == 'body_order': payload['body_table'][0]['structure_index'] = 1
    elif mutation == 'slices': payload['body_table'][0]['M_slice'] = [0, 30]
    elif mutation == 'bool_slice': payload['body_table'][0]['J_slice'][0] = False
    elif mutation == 'medium': payload['body_table'][0]['interior']['epsilon_r'] = 2.+0j
    elif mutation == 'decoder': payload['decoder_version'] = 'unversioned'
    elif mutation == 'integration': payload['integration']['order'] += 1
    elif mutation == 'stopping': payload['stopping']['rtol'] *= 2
    else: payload['query_config']['relative_tolerance'] *= 2
    with pytest.raises(ValueError):
        SurfaceData.restore(payload, scene, 3e8)


def test_invisible_interface_monitors_and_source_ownership():
    monitors = [mw.PointMonitor(name='outside', position=(.2, .1, .3), fields=('Ex', 'Hy')),
                mw.PointMonitor(name='inside', position=(.01, .02, .01), fields=('Ex', 'Hy'))]
    scene, result = solve(1., monitors=monitors)
    data = result.surface_currents
    assert torch.count_nonzero(data.scattered_coefficients) == 0
    for monitor, region in zip(monitors, ('exterior', 'interior')):
        points = torch.tensor([monitor.position], device='cuda', dtype=torch.float64)
        expected = data.incident_carrier.evaluate(points, data.handle.exterior)
        actual = result.query_fields(points, region_id=region, field_kind='total')
        for a, b in zip(actual, expected): torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(result.monitors[monitor.name]['Ex'], expected[0][0, 0])
        torch.testing.assert_close(result.monitors[monitor.name]['Hy'], expected[1][0, 1])
    data.handle.incident_carrier = IncidentCarrier.compile(
        [mw.PlaneWave(direction=(1, 0, 0), polarization=(0, 0, 1))], data.handle.exterior)
    with pytest.raises(ValueError, match='solved RHS'):
        data.snapshot()
    with pytest.raises(ValueError, match='solved RHS'):
        result.far_field(torch.tensor([[0., 0., 1.]], device='cuda', dtype=torch.float64))
