"""Public hierarchical physical solve, exact decode restore and identity checks."""
import copy
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.fields import FieldQueryConfig
from witwin.maxwell.mom.runtime import SurfaceData

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('backend,storage', [('hmatrix', 'dense'), ('h2', 'dense'), ('direct_h2', 'dense'), ('fmm', 'dense'), ('fmm', 'compact')])
def test_public_hierarchical_solve_and_exact_restore(tmp_path, monkeypatch, backend, storage):
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.1), mw.Material(eps_r=2.))],
                     sources=[mw.PlaneWave()])
    if backend in ('h2', 'direct_h2', 'fmm'):
        scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.01, position=(x, 0., 0.)),
                         mw.Material(eps_r=2.)) for x in (-.5, .5)], sources=[mw.PlaneWave()])
    config = mw.MoMConfig(subdivisions=0, integration=IntegrationConfig(order=3 if backend in ('h2', 'direct_h2', 'fmm') else 2, singular_order=3),
        stopping=StoppingRule(method='gmres', preconditioner='rwg_diagonal' if backend in ('direct_h2', 'fmm') else 'row_column_norm', rtol=1e-12, atol=0, maxiter=240, restart=120),
        query_config=FieldQueryConfig(adaptive=False), operator_backend=backend,
        riesz_storage='local_rwg_v1', hierarchical=None if backend in ('direct_h2', 'fmm') else mw.HMatrixConfig(pivoting='adaptive'))
    if backend == 'fmm':
        config = replace(config, fmm=mw.FMMConfig(table_storage=storage))
    import witwin.maxwell.mom.runtime as runtime
    def forbidden(*args, **kwargs):
        raise AssertionError('Public H path must not use dense system assembly')
    monkeypatch.setattr(runtime, 'assemble_system', forbidden)
    if backend in ('direct_h2', 'fmm'):
        import witwin.maxwell.mom.hmatrix as hmatrix
        monkeypatch.setattr(hmatrix.HierarchicalOperator, '__init__', forbidden)
    if backend == 'fmm':
        import witwin.maxwell.mom.interpolation_kernel as interpolation
        monkeypatch.setattr(interpolation.InterpolationKernel, '__init__', forbidden)
    result = mw.Simulation.mom(scene, frequency=3e8, config=config).run()
    assert result.stats()['converged'], result.stats()
    assert result.stats()['error_budget']['checked']
    data = result.surface_currents
    payload = data.snapshot()
    assert payload['schema_version'] == (11 if backend == 'fmm' else 10 if backend == 'direct_h2' else 9 if backend == 'h2' else 8) and payload['operator_backend'] == backend
    if backend == 'h2':
        assert len(data.handle.actions.couplings) == 2
        assert payload['h2_config'] == config.h2.payload()
    if backend == 'direct_h2':
        assert len(data.handle.actions.exterior.far) == 2
        assert payload['direct_h2_config'] == config.direct_h2.payload()
    if backend == 'fmm':
        assert len(data.handle.actions.exterior.far) == 2
        assert payload['fmm_config'] == config.fmm.payload()
        assert payload['operator_metadata']['complexity_qualification'] is False
    assert payload['operator_metadata']['physical_qualification'] is False
    points = torch.tensor([[.2, .3, .4]], device='cuda', dtype=torch.float64)
    expected = result.query_fields(points, region_id='exterior', field_kind='total')
    path = tmp_path/'hierarchical.pt'
    result.save(path)
    import witwin.maxwell.mom.hierarchical_runtime as hierarchical_runtime
    monkeypatch.setattr(hierarchical_runtime.HierarchicalSystemOperator, '__init__', forbidden)
    monkeypatch.setattr(hierarchical_runtime.HierarchicalH2SystemOperator, '__init__', forbidden)
    monkeypatch.setattr(hierarchical_runtime.DirectH2SystemOperator, '__init__', forbidden)
    monkeypatch.setattr(hierarchical_runtime.MultipoleSystemOperator, '__init__', forbidden)
    restored = mw.Result.load(path, scene=scene)
    outer = torch.load(path, weights_only=False)
    outer['metadata']['operator_backend'] = 'hmatrix' if backend == 'h2' else 'h2'
    corrupted_path = tmp_path/'wrong_backend.pt'
    torch.save(outer, corrupted_path)
    with pytest.raises(ValueError, match='metadata'):
        mw.Result.load(corrupted_path, scene=scene)
    assert torch.equal(restored.surface_currents.scattered_coefficients, data.scattered_coefficients)
    for actual, value in zip(restored.query_fields(points, region_id='exterior', field_kind='total'), expected):
        torch.testing.assert_close(actual, value, rtol=0, atol=0)
    corrupted = copy.deepcopy(payload)
    if backend == 'fmm':
        corrupted['fmm_config']['parameters']['order'] = 7
    elif backend == 'direct_h2':
        corrupted['direct_h2_config']['parameters']['order'] = 7
    else:
        corrupted['hierarchical_config']['parameters']['tolerance'] = 2e-8
    with pytest.raises(ValueError, match='identity'):
        SurfaceData.restore(corrupted, scene, 3e8)
    if backend == 'h2':
        corrupted = copy.deepcopy(payload)
        corrupted['h2_config']['parameters']['solve_dual_tolerance'] = 2e-8
        with pytest.raises(ValueError, match='identity'):
            SurfaceData.restore(corrupted, scene, 3e8)
    corrupted = copy.deepcopy(payload)
    corrupted['operator_metadata']['physical_qualification'] = True
    with pytest.raises(ValueError, match='identity'):
        SurfaceData.restore(corrupted, scene, 3e8)
    for physical in (result, restored):
        handle = physical.surface_currents.handle
        for target in (handle, handle.body_handles[0]):
            integration = target.integration
            target.integration = replace(integration, order=integration.order+1)
            with pytest.raises(ValueError, match='integration configuration'):
                physical.query_fields(points, region_id='exterior', field_kind='total')
            if physical is result:
                with pytest.raises(ValueError, match='binding changed'):
                    handle.build_rhs()
            target.integration = integration
        saved = physical.surface_currents.config
        if backend == 'fmm':
            physical.surface_currents.config = replace(saved, fmm=replace(saved.fmm, order=7))
        elif backend == 'direct_h2':
            physical.surface_currents.config = replace(saved, direct_h2=replace(saved.direct_h2, order=7))
        else:
            physical.surface_currents.config = replace(saved, hierarchical=replace(saved.hierarchical, max_rank=17))
        with pytest.raises(ValueError, match='configuration'):
            physical.query_fields(points, region_id='exterior', field_kind='total')
        physical.surface_currents.config = saved
        if backend == 'h2':
            physical.surface_currents.config = replace(saved, h2=replace(saved.h2, max_rank=17))
            with pytest.raises(ValueError, match='configuration'):
                physical.query_fields(points, region_id='exterior', field_kind='total')
            physical.surface_currents.config = saved


@pytest.mark.parametrize('backend', ['hmatrix', 'h2', 'direct_h2', 'fmm'])
def test_public_hierarchical_combinations_fail_before_assembly(backend):
    with pytest.raises(ValueError, match='local_rwg_v1'):
        mw.MoMConfig(operator_backend=backend)
    with pytest.raises(ValueError, match='iterative'):
        mw.MoMConfig(operator_backend=backend, riesz_storage='local_rwg_v1')
    periodic = mw.MoMPeriodicConfig(((1., 0., 0.),))
    if backend == 'direct_h2':
        with pytest.raises(NotImplementedError, match='periodic'):
            mw.MoMConfig(operator_backend=backend, periodic=periodic,
                         riesz_storage='local_rwg_v1',
                         stopping=StoppingRule(method='gmres', preconditioner='none'))
    else:
        with pytest.raises(ValueError, match='local_rwg_v1'):
            mw.MoMConfig(operator_backend=backend, periodic=periodic)
        with pytest.raises(ValueError, match='iterative'):
            mw.MoMConfig(operator_backend=backend, periodic=periodic,
                         riesz_storage='local_rwg_v1')
        admitted = mw.MoMConfig(operator_backend=backend, periodic=periodic,
            riesz_storage='local_rwg_v1',
            stopping=StoppingRule(method='gmres', preconditioner='none'))
        assert admitted.periodic is periodic and admitted.operator_backend == backend
    with pytest.raises(ValueError, match='require operator_backend'):
        mw.MoMConfig(hierarchical=mw.HMatrixConfig())
    with pytest.raises(NotImplementedError):
        mw.MoMConfig(operator_backend='pme')
    with pytest.raises(ValueError, match='requires operator_backend=h2'):
        mw.MoMConfig(h2=mw.H2Config())
