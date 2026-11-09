"""Periodic public H/H2 solves, original residual checks and exact decode restore."""
import copy
import json
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.runtime import SurfaceData
from test_periodic_tensor_public import setup

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('dimension,tensor,backend,refinement',
    [(d, t, b, 0) for b in ('hmatrix', 'h2') for t in (False, True, 'mixed') for d in (1, 2)]
    + [(d, 'mixed', b, 2) for b in ('hmatrix', 'h2') for d in (1, 2)])
def test_periodic_hierarchical_public_dense_oracle_and_restore(
        dimension, tensor, backend, refinement, tmp_path, monkeypatch):
    scene, dense_config = setup(dimension, bool(tensor))
    if tensor == 'mixed':
        scene = mw.Scene(structures=[scene.structures[0],
            mw.Structure(scene.structures[1].geometry, mw.Material(eps_r=3., mu_r=1.3), name='body1')],
            sources=scene.sources)
    if refinement:
        from witwin.maxwell.mom.mesh import SurfaceMesh
        structures = []
        for body in scene.structures:
            vertices = torch.as_tensor(body.geometry.vertices, device=scene.device, dtype=torch.float64)
            # This separated control has storage-reducing far rank at 1e-10.
            vertices = vertices[0]+.5*(vertices-vertices[0])
            mesh = SurfaceMesh(
                vertices,
                torch.as_tensor(body.geometry.faces, device=scene.device, dtype=torch.int64)).refine(refinement)
            structures.append(mw.Structure(mw.Mesh(mesh.vertices, mesh.faces,
                recenter=False, fill_mode='solid'), body.material, name=body.name))
        scene = mw.Scene(structures=structures, sources=scene.sources)
    reference = mw.Simulation.mom(scene, frequency=3e8, config=dense_config).run()
    assert reference.stats()['converged'], reference.stats()
    if not tensor:
        relabeled_dense = reference.surface_currents.snapshot()
        relabeled_dense['schema_version'] = 17
        with pytest.raises(ValueError, match='hierarchical schema'):
            SurfaceData.restore(relabeled_dense, scene, 3e8)
    config = replace(dense_config, operator_backend=backend, riesz_storage='local_rwg_v1',
        hierarchical=mw.HMatrixConfig(leaf_size=96 if refinement else 6,
            max_rank=192 if refinement else 32, tolerance=1e-10, pivoting='adaptive'),
        h2=mw.H2Config(max_rank=128) if refinement and backend == 'h2' else None,
        stopping=StoppingRule(method='gmres', preconditioner='row_column_norm',
                              rtol=1e-12, atol=0, maxiter=480 if refinement else 240,
                              restart=480 if refinement else 120,
                              stagnation_window=480 if refinement else 20))
    def forbidden(*args, **kwargs):
        raise AssertionError('Periodic H/H2 must not assemble a global dense operator or resolve on load')
    import witwin.maxwell.mom.periodic_runtime as periodic
    import witwin.maxwell.mom.anisotropic_operator as anisotropic
    import witwin.maxwell.mom.runtime as runtime
    monkeypatch.setattr(periodic, 'assemble_periodic_system', forbidden)
    monkeypatch.setattr(anisotropic, 'assemble_generalized_system', forbidden)
    monkeypatch.setattr(runtime, 'assemble_system', forbidden)
    result = mw.Simulation.mom(scene, frequency=3e8, config=config).run()
    (tmp_path/'solve-diagnostics.json').write_text(json.dumps({
        'stats': result.stats(), 'operator': result.surface_currents.handle.metadata},
        indent=2, default=str))
    assert result.stats()['converged'], result.stats()
    assert result.stats()['error_budget']['checked']
    data = result.surface_currents
    torch.testing.assert_close(data.scattered_coefficients,
        reference.surface_currents.scattered_coefficients, rtol=2e-7, atol=1e-12)
    if refinement:
        if backend == 'h2':
            assert len(data.handle.actions.couplings) > 0
        else:
            assert data.handle.metadata['far_blocks'] > 0
    else:
        # A 12-by-12 far block needs full rank at this tolerance: the storage
        # guard must retain exact leaves instead of claiming compression.
        diagnostics = (data.handle.metadata['source_h_diagnostics'] if backend == 'h2'
                       else data.handle.metadata)
        assert diagnostics['compression_rejections']
        assert all(not item['storage_reduced']
                   for item in diagnostics['compression_rejections'])
    points = {1: torch.tensor([[-.19,.01,.01],[.81,.01,.01]], device='cuda', dtype=torch.float64),
              0: torch.tensor([[.1,.2,.3],[1.1,.2,.3]], device='cuda', dtype=torch.float64)}
    if refinement:
        points[1] = torch.tensor([[-.195,.005,.005],[.805,.005,.005]],
                                 device='cuda', dtype=torch.float64)
    saved_fields = {}
    phase = torch.exp(points[0].new_tensor(config.periodic.bloch_rad_per_m[0])*1j)
    for region, point in points.items():
        actual = result.query_fields(point, region_id=region, field_kind='total')
        expected = reference.query_fields(point, region_id=region, field_kind='total')
        saved_fields[region] = actual
        for value, oracle in zip(actual, expected):
            torch.testing.assert_close(value, oracle, rtol=2e-7, atol=1e-11)
            torch.testing.assert_close(value[1], phase*value[0], rtol=1e-10, atol=1e-12)
    payload = data.snapshot()
    assert payload['schema_version'] == (16 if tensor else 17)
    assert payload['operator_metadata']['physical_qualification'] is False
    path = tmp_path/('periodic-hierarchy-compressed.pt' if refinement else 'periodic-hierarchy.pt')
    result.save(path)
    import witwin.maxwell.mom.periodic_hierarchical_runtime as hierarchy
    from witwin.maxwell.mom.surface_h_snapshot import encode, restore
    from witwin.maxwell.mom.hmatrix import HierarchicalOperator
    from witwin.maxwell.mom.h2_operator import HierarchicalH2Operator
    factors = encode(data.handle)
    factor_path = tmp_path/'periodic-hierarchy-factors.pt'
    torch.save(factors, factor_path)
    factors = torch.load(factor_path, weights_only=False)
    fresh_kernel = config.periodic.compile(device=scene.device, dtype=config.dtype)
    fresh_provider, _, _ = hierarchy._provider(data.handle.system, data.handle.exterior,
        scene.sources, config, fresh_kernel)
    vector = torch.randn_like(data.scattered_coefficients)
    expected_actions = (data.handle.apply(vector), data.handle.adjoint(vector))
    with monkeypatch.context() as patch:
        patch.setattr(HierarchicalOperator, '__init__', forbidden)
        patch.setattr(HierarchicalH2Operator, '__init__', forbidden)
        patch.setattr(type(fresh_provider), '__call__', forbidden)
        patch.setattr(type(fresh_provider), 'regular_block', forbidden)
        patch.setattr(torch.linalg, 'solve', forbidden)
        restored_operator = restore(factors, fresh_provider, sources=scene.sources)
        for actual, expected in zip((restored_operator.apply(vector),
                                     restored_operator.adjoint(vector)), expected_actions):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    from witwin.maxwell.mom.volume_h_snapshot import _digest
    damaged_factors = copy.deepcopy(factors)
    damaged_factors['numerical']['periodic']['real_shells'] += 1
    damaged_factors['sha256'] = _digest({k:v for k,v in damaged_factors.items() if k!='sha256'})
    with pytest.raises(ValueError, match='periodic kernel mismatch'):
        restore(damaged_factors, fresh_provider, sources=scene.sources)
    with monkeypatch.context() as patch:
        patch.setattr(hierarchy.PeriodicHierarchicalSystemOperator, '__init__', forbidden)
        patch.setattr(hierarchy.PeriodicHierarchicalH2SystemOperator, '__init__', forbidden)
        patch.setattr(torch.linalg, 'solve', forbidden)
        restored = mw.Result.load(path, scene=scene)
        assert torch.equal(restored.surface_currents.scattered_coefficients, data.scattered_coefficients)
        for region, point in points.items():
            for actual, expected in zip(restored.query_fields(point, region_id=region,
                    field_kind='total'), saved_fields[region]):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for key in ('hierarchical_config', 'periodic_config'):
        damaged = copy.deepcopy(payload)
        if key == 'hierarchical_config':
            damaged[key]['parameters']['tolerance'] *= 2
        else:
            damaged[key]['real_shells'] += 1
        with pytest.raises(ValueError):
            SurfaceData.restore(damaged, scene, 3e8)
    data.handle.periodic_kernel = replace(data.handle.periodic_kernel)
    with pytest.raises(ValueError, match='binding changed|kernel changed'):
        data.handle.refresh()
