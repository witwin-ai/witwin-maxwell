"""Public magnetic multibody combinations against one original dense reference."""
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.fields import FieldQueryConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.fixture(scope='module')
def magnetic_reference():
    scene = mw.Scene(structures=[mw.Structure(
        mw.Sphere(radius=.01, position=(x, 0., 0.)), material, name=f'body{i}')
        for i, (x, material) in enumerate((
            (-.5, mw.Material(eps_r=2., mu_r=1.3, sigma_e=.001, sigma_m=.002)),
            (.5, mw.Material(eps_r=3., mu_r=1.7, sigma_e=.002, sigma_m=.003))))],
        sources=[mw.PlaneWave()])
    config = mw.MoMConfig(subdivisions=0,
        integration=IntegrationConfig(order=4, singular_order=4),
        query_config=FieldQueryConfig(adaptive=False))
    result = mw.Simulation.mom(scene, frequency=3e8, config=config).run()
    assert result.stats()['converged']
    return scene, config, result


@pytest.mark.parametrize('backend', ('hmatrix', 'h2', 'direct_h2', 'fmm'))
def test_public_magnetic_multibody_backend_gate(magnetic_reference, backend, monkeypatch, tmp_path):
    scene, dense_config, reference = magnetic_reference
    config = replace(dense_config, operator_backend=backend, riesz_storage='local_rwg_v1',
        stopping=StoppingRule(method='gmres', rtol=1e-12, atol=0., maxiter=240, restart=120,
            preconditioner='rwg_diagonal' if backend in ('direct_h2', 'fmm') else 'row_column_norm'),
        hierarchical=mw.HMatrixConfig(pivoting='adaptive') if backend in ('hmatrix', 'h2') else None,
        fmm=mw.FMMConfig(order=6, table_storage='compact') if backend == 'fmm' else None)
    import witwin.maxwell.mom.runtime as runtime
    def forbidden(*args, **kwargs):
        raise AssertionError('Accelerated combination must not assemble a dense system')
    monkeypatch.setattr(runtime, 'assemble_system', forbidden)
    result = mw.Simulation.mom(scene, frequency=3e8, config=config).run()
    stats = result.stats()
    assert stats['converged'] and stats['error_budget']['checked'], stats
    data, original = result.surface_currents, reference.surface_currents
    current, expected = data.scattered_coefficients, original.scattered_coefficients
    metrics = {'backend': backend, 'iterations': stats.get('iterations')}
    # Check each body and family separately to prevent units or a stronger body
    # from hiding a weak magnetic/electric-current error.
    n = len(current)//2
    for body in original.handle.system.bodies:
        start, count = body.edge_offset, body.basis.n_edges
        for family, offset in (('J', 0), ('M', n)):
            sl = slice(offset+start, offset+start+count)
            error = float((current[sl]-expected[sl]).norm()/expected[sl].norm())
            metrics[f'body{body.structure_index}_{family}'] = error
            assert error < 1e-8, metrics
    rhs = original.handle.build_rhs()
    residual = original.handle.apply(current)-rhs
    eta = abs(original.handle.exterior.eta)
    weight = torch.cat((torch.full((n,), eta, device='cuda', dtype=torch.float64),
                        torch.ones(n, device='cuda', dtype=torch.float64)))
    metrics['full_original_balanced_residual'] = float((weight*residual).norm()/(weight*rhs).norm())
    assert metrics['full_original_balanced_residual'] < 1e-8, metrics
    for region, point in (('exterior', (.1, .2, .3)), ('body0', (-.5, 0., 0.)), ('body1', (.5, 0., 0.))):
        points = torch.tensor([point], device='cuda', dtype=torch.float64)
        for family, actual, wanted in zip(('E', 'H'),
                result.query_fields(points, region_id=region, field_kind='total'),
                reference.query_fields(points, region_id=region, field_kind='total')):
            error = float((actual-wanted).norm()/wanted.norm())
            metrics[f'{region}_{family}'] = error
            assert error < 1e-8, metrics
    points = torch.tensor([[.1, .2, .3]], device='cuda', dtype=torch.float64)
    for family, actual, wanted in zip(('E', 'H'),
            result.query_fields(points, region_id='exterior', field_kind='scattered'),
            reference.query_fields(points, region_id='exterior', field_kind='scattered')):
        error = float((actual-wanted).norm()/wanted.norm())
        metrics[f'exterior_scattered_{family}'] = error
        assert error < 1e-8, metrics
    if backend == 'hmatrix':
        assert data.handle.metadata['far_blocks'] > 0
    if backend == 'h2':
        assert len(data.handle.actions.couplings) > 0
    if backend in ('direct_h2', 'fmm'):
        assert len(data.handle.actions.exterior.far) > 0
    path = tmp_path/'magnetic_combination.pt'
    result.save(path)
    restored = mw.Result.load(path, scene=scene)
    assert torch.equal(restored.surface_currents.scattered_coefficients, current)
    assert restored.surface_currents.snapshot()['operator_metadata']['physical_qualification'] is False
    print(metrics)
