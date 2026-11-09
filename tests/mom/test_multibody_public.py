"""Global physical solution routing, coherent fields and strict body persistence."""
import copy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.fields import FieldQueryConfig, far_field
from witwin.maxwell.mom.lifting_fields import decode_lifted, system_local_coefficients
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.runtime import SurfaceData

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def solve(eps=1.):
    scene = mw.Scene(structures=[
        mw.Structure(mw.Sphere(radius=.07, position=(-.2, 0., 0.)), mw.Material(eps_r=eps), name='left'),
        mw.Structure(mw.Sphere(radius=.06, position=(.2, 0., 0.)), mw.Material(eps_r=eps), name='right')],
        sources=[mw.PlaneWave()], monitors=[mw.PointMonitor(name='right_probe', position=(.2, 0., 0.), fields=('Ex',))])
    config = mw.MoMConfig(subdivisions=0, integration=IntegrationConfig(order=3, singular_order=4),
        query_config=FieldQueryConfig(adaptive=False), riesz_storage='compact_rwg_v1')
    return scene, mw.Simulation.mom(scene, frequency=3e8, config=config).run()


@pytest.fixture(scope='module')
def invisible(): return solve()


def test_two_invisible_bodies_public_routing_and_traces(invisible):
    scene, result = invisible
    data = result.surface_currents
    assert data.scattered_coefficients.shape == (120,) and torch.count_nonzero(data.scattered_coefficients) == 0
    assert [body.name for body in data.bodies] == ['left', 'right']
    assert data.bodies[1].J_slice == slice(30, 60) and data.bodies[1].M_slice == slice(90, 120)
    assert result.monitors['right_probe']['region_id'] == 2
    assert result.monitors['right_probe']['structure_name'] == 'right'
    for region, position in [('exterior', (0., .3, 0.)), ('left', (-.2, 0., 0.)), (2, (.2, 0., 0.))]:
        point = torch.tensor([position], device='cuda', dtype=torch.float64)
        actual = result.query_fields(point, region_id=region, field_kind='total')
        expected = data.incident_carrier.evaluate(point, data.handle.exterior)
        for a, b in zip(actual, expected): torch.testing.assert_close(a, b, rtol=0, atol=0)
    points = torch.cat([body.mesh.centroids[:1] for body in data.bodies])
    actual = result.query_fields(points, region_id=0, field_kind='total', trace_side='exterior')
    for a, b in zip(actual, data.incident_carrier.evaluate(points, data.handle.exterior)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match='ambiguous'):
        result.query_fields(points[:1], region_id='interior', field_kind='total')
    with pytest.raises(ValueError, match='geometry'):
        result.query_fields(points[:1], region_id='right', field_kind='total', trace_side='interior')
    with pytest.raises(ValueError, match='trace_side'):
        result.query_fields(points, region_id=0, field_kind='total')
    with pytest.raises(ValueError, match='explicit'):
        _ = data.mesh


def test_coherent_exterior_sum_and_gradients(invisible):
    _, result = invisible
    original = result.surface_currents
    x = torch.randn_like(original.scattered_coefficients).requires_grad_()
    data = SurfaceData(x, original.handle, original.config, original.scene_signature)
    points = torch.tensor([[0., .3, .2]], device='cuda', dtype=torch.float64)
    actual = data.query_fields(points, 0, 'total')
    parts = [decode_lifted(h, system_local_coefficients(data.handle, x, i), points, 0, 'scattered',
                query_config=data.config.query_config) for i, h in enumerate(data.handle.body_handles)]
    incident = data.incident_carrier.evaluate(points, data.handle.exterior)
    for a, b, c, d in zip(actual, *parts, incident): torch.testing.assert_close(a, b+c+d)
    directions = torch.tensor([[0., 0., 1.], [1., 0., 0.]], device='cuda', dtype=torch.float64)
    actual_far = data.far_field(directions)
    parts = [far_field(h, system_local_coefficients(data.handle, x, i), directions)
             for i, h in enumerate(data.handle.body_handles)]
    for a, b, c in zip(actual_far, *parts): torch.testing.assert_close(a, b+c)
    loss = sum(v.abs().square().sum() for v in (*actual, *actual_far))
    grad = torch.autograd.grad(loss, x)[0]
    assert torch.isfinite(grad).all() and bool((grad.abs() > 0).all())


def test_two_body_physical_rhs_solve_and_roundtrip(tmp_path):
    scene, result = solve(1.01)
    data = result.surface_currents
    assert torch.linalg.vector_norm(data.scattered_coefficients) > 0
    path = tmp_path/'bodies.pt'
    result.save(path)
    restored = mw.Result.load(path, scene=scene).surface_currents
    assert not hasattr(restored.handle, 'matrix')
    assert torch.equal(restored.scattered_coefficients, data.scattered_coefficients)
    for region, position in [(0, (0., .3, .2)), ('right', (.2, 0., 0.))]:
        points = torch.tensor([position], device='cuda', dtype=torch.float64)
        for a, b in zip(data.query_fields(points, region, 'total'), restored.query_fields(points, region, 'total')):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    for key, value in [('edge_offset', 0), ('region_id', True), ('M_slice', [30, 60]), ('name', 'left')]:
        payload = copy.deepcopy(data.snapshot())
        payload['body_table'][1][key] = value
        with pytest.raises(ValueError): SurfaceData.restore(payload, scene, 3e8)


def test_complete_field_adaptation_absorption_and_global_modes(invisible):
    from witwin.maxwell.mom.modes import RegularModes, build_modal_rhs, modal_response
    from witwin.maxwell.mom.linalg import StoppingRule
    _, result = invisible
    data = result.surface_currents
    points = torch.tensor([[0., .3, .2]], device='cuda', dtype=torch.float64)
    data.query_fields(points, 0, 'total', query_config=FieldQueryConfig(max_order=8))
    log = data.handle.last_field_query_log
    assert log['converged'] and log['checks'][0]['order'] == 8
    assert log['estimator'] == 'complete_multibody_field_successive_order_difference'
    assert log['global_error_certified'] is False
    with pytest.raises(ValueError, match='max_order'):
        data.query_fields(points, 0, 'total', query_config=FieldQueryConfig(max_order=4))
    rules = {body.name: (body.mesh.vertices.mean(0)[None], torch.ones(1, device='cuda', dtype=torch.float64)*1e-6)
             for body in data.bodies}
    powers = data.absorbed_powers(rules)
    assert set(powers['per_region_W']) == {1, 2} and powers['total_W'] == 0
    with pytest.raises(ValueError, match='every body'): data.absorbed_powers({'left': rules['left']})
    modes = RegularModes(1, data.handle.exterior, angular_order=4)
    system = data.handle.system
    with pytest.raises(ValueError, match='enclose'): modes.validate_enclosing_sphere(system, .15)
    assert modes.validate_enclosing_sphere(system, .4)['geometry_radius_m'] < .4
    rhs = build_modal_rhs(system, modes, order=3)
    for i, body in enumerate(system.bodies):
        local = build_modal_rhs(body.basis, modes, order=3)
        n = body.basis.n_edges
        torch.testing.assert_close(rhs[system.J_slices[i]], local[:n])
        torch.testing.assert_close(rhs[system.M_slices[i]], local[n:])
    response, log = modal_response(data.handle, modes, order=3, stopping_rule=StoppingRule(method='direct'))
    torch.testing.assert_close(data.handle.matrix@response, rhs, rtol=1e-9, atol=1e-12)
    assert log['physical_lifted_solution'] is False


def test_nonzero_trace_owns_only_its_jump_and_interior_ignores_foreign_trace(invisible):
    _, result = invisible
    original = result.surface_currents
    x = torch.randn_like(original.scattered_coefficients)
    data = SurfaceData(x, original.handle, original.config, original.scene_signature)
    point = data.bodies[0].mesh.centroids[:1]
    actual = data.query_fields(point, 0, 'scattered', trace_side='exterior')
    own = decode_lifted(data.handle.body_handles[0], system_local_coefficients(data.handle, x, 0),
                       point, 0, 'scattered', 'exterior', query_config=data.config.query_config)
    foreign = decode_lifted(data.handle.body_handles[1], system_local_coefficients(data.handle, x, 1),
                           point, 0, 'scattered', query_config=data.config.query_config)
    for a, b, c in zip(actual, own, foreign): torch.testing.assert_close(a, b+c)
    inside = data.bodies[0].mesh.vertices.mean(0)[None]
    first = data.query_fields(inside, 'left', 'total')
    changed = x.clone()
    changed[data.bodies[1].J_slice] *= 1e6
    changed[data.bodies[1].M_slice] *= -1e6
    other = SurfaceData(changed, data.handle, data.config, data.scene_signature)
    for a, b in zip(first, other.query_fields(inside, 'left', 'total')):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
