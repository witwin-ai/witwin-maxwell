"""Tensor surface/zero-volume composition; qualification follows no algebra test."""
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.linalg import StoppingRule
from test_linear_volume_state import system_and_volumes

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('backend', ['dense', 'hmatrix', 'h2'])
def test_tensor_scalar_limit_keeps_exact_surface_solution_and_archive(backend, tmp_path, monkeypatch):
    from maxwell_bench.linear_state import linear_simulation_binding, write_linear_state, read_linear_state
    import witwin.maxwell.mom.anisotropic_operator as assembly
    from witwin.maxwell.mom.hmatrix import HierarchicalOperator
    from witwin.maxwell.mom.h2_operator import HierarchicalH2Operator
    _, volumes = system_and_volumes(torch.float64)
    scene = mw.Scene(structures=[mw.Structure(mw.Mesh(v.boundary.vertices, v.boundary.faces,
        recenter=False, fill_mode='solid'), mw.Material(
            epsilon_tensor=mw.DiagonalTensor3(*([2.+i]*3)),
            mu_tensor=mw.DiagonalTensor3(*([1.2+i*.1]*3)))) for i,v in enumerate(volumes)],
        sources=[mw.PlaneWave(source_time=mw.CW(frequency=3e8))])
    config = mw.MoMConfig(integration=IntegrationConfig(order=2, singular_order=2),
        query_config=mw.FieldQueryConfig(adaptive=False), operator_backend=backend,
        riesz_storage='compact_rwg_v1' if backend=='dense' else 'local_rwg_v1',
        stopping=StoppingRule(method='direct') if backend=='dense' else
            StoppingRule(method='gmres', fallback='none', rtol=1e-12, maxiter=240, restart=120),
        dyadic_integration=mw.DyadicIntegrationConfig())
    reference = mw.Simulation.mom(scene, frequency=3e8, config=config).run()
    simulation = mw.Simulation.mom(scene, frequency=3e8,
        config=replace(config, linear_coupled=mw.LinearCoupledConfig(volumes)))
    binding = linear_simulation_binding(simulation)
    prepared = simulation.prepare()
    independent = reference
    reference = prepared.solver.surface_runtime.run()
    # Compare repeatability in the vector norm used by the diagnostic; tiny
    # individual components can have large relative cancellation errors.
    expected_coefficients = independent.surface_currents.scattered_coefficients
    repeat_error = (reference.surface_currents.scattered_coefficients-expected_coefficients).norm()
    assert float(repeat_error/expected_coefficients.norm()) <= 64*torch.finfo(config.dtype).eps
    result = prepared.run()
    assert reference.stats()['converged'] and result.stats()['converged']
    assert bool(result.surface_currents.scattered_coefficients.any())
    assert torch.equal(result.surface_currents.scattered_coefficients, reference.surface_currents.scattered_coefficients)
    points = volumes[0].vertices.new_tensor([[.2,-.1,.1]])
    expected = reference.query_fields(points, region_id=0, field_kind='total')
    for actual, field in zip(result.query_fields(points, region_id=0, field_kind='total'), expected):
        assert torch.equal(actual, field)
    payload = result.surface_currents.snapshot()
    assert payload['schema_version']==20 and payload['surface']['schema_version']==13
    assert payload['surface']['operator_backend']==backend
    assert not payload['representation']['physical_qualification']
    path = tmp_path/'result.pt'; result.save(path)
    archive = tmp_path/'archive'
    sha = write_linear_state(result, archive, expected_binding=binding,
        physical_group_id='tensor-zero-volume-control', split='pilot', source_id='pump')
    def forbidden(*args, **kwargs): raise AssertionError('Exact decode cannot assemble or solve')
    monkeypatch.setattr(assembly, 'assemble_generalized_system', forbidden)
    monkeypatch.setattr(HierarchicalOperator, '__init__', forbidden)
    monkeypatch.setattr(HierarchicalH2Operator, '__init__', forbidden)
    monkeypatch.setattr(torch.linalg, 'solve', forbidden)
    native = mw.Result.load(path, scene=scene)
    _, restored = read_linear_state(archive, expected_manifest_sha256=sha,
        expected_binding=binding, scene=scene, physical_group_id='tensor-zero-volume-control',
        split='pilot', source_id='pump')
    for recovered in (native, restored):
        assert recovered.surface_currents.config == config
        assert torch.equal(recovered.surface_currents.scattered_coefficients, result.surface_currents.scattered_coefficients)
        for actual, field in zip(recovered.query_fields(points, region_id=0, field_kind='total'), expected):
            assert torch.equal(actual, field)


def test_tensor_zero_volume_preserves_actual_primary_failure():
    from witwin.maxwell.mom.dyadic_green import DyadicConvergenceError
    _, volumes = system_and_volumes(torch.float64)
    scene = mw.Scene(structures=[mw.Structure(mw.Mesh(v.boundary.vertices, v.boundary.faces,
        recenter=False, fill_mode='solid'), mw.Material(
            epsilon_tensor=mw.DiagonalTensor3(2.,3.,4.), sigma_e=.01)) for v in volumes],
        sources=[mw.PlaneWave(source_time=mw.CW(frequency=3e8))])
    config = mw.MoMConfig(integration=IntegrationConfig(order=1,singular_order=1,pair_batch=1),
        linear_coupled=mw.LinearCoupledConfig(volumes),
        dyadic_integration=mw.DyadicIntegrationConfig(radial_order=2,azimuthal_order=2,
            max_radial_order=4,max_azimuthal_order=4,relative_tolerance=1e-12,absolute_tolerance=1e-15))
    with pytest.raises(DyadicConvergenceError):
        mw.Simulation.mom(scene,frequency=3e8,config=config).prepare()


@pytest.mark.parametrize('backend', ['dense', 'hmatrix', 'h2'])
def test_manufactured_periodic_surface_stays_bound_inside_linear_state(backend, tmp_path, monkeypatch):
    from test_periodic_dyadic_public import problem
    from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
    import witwin.maxwell.mom.periodic_dyadic_operator as assembly
    import witwin.maxwell.mom.periodic_dyadic_nearfield as selected
    from witwin.maxwell.mom.hmatrix import HierarchicalOperator
    from witwin.maxwell.mom.h2_operator import HierarchicalH2Operator
    from maxwell_bench.linear_state import linear_simulation_binding, write_linear_state, read_linear_state
    scene, config = problem(2)
    cells = torch.tensor([[0,1,2,3]], device='cuda')
    volumes = tuple(TetrahedralVolumeMesh(s.geometry.vertices, cells, region_id=i+1)
                    for i,s in enumerate(scene.structures))
    config = replace(config, operator_backend=backend,
        riesz_storage='compact_rwg_v1' if backend=='dense' else 'local_rwg_v1',
        stopping=StoppingRule(method='direct') if backend=='dense' else StoppingRule(method='gmres',fallback='none'),
        linear_coupled=mw.LinearCoupledConfig(volumes))
    def matrix(system, **kwargs):
        return torch.eye(2*system.total_edges, device=system.device, dtype=config.dtype), {'physical_qualification':False}
    def block(system,i,j,rows,cols,**kwargs):
        diagonal = (rows[:,None]==cols[None,:]).to(config.dtype)*(i==j)
        value = torch.block_diag(diagonal,diagonal)
        panel=dict(triangle_pair_counts=[0]*5,maximum_classification_pairs=0)
        return value,dict(exterior_primary=panel,local_interior=panel if i==j else None,
            nonprimary_exterior=dict(panel_pairs=0,maximum_panel_pairs=0),physical_qualification=False)
    monkeypatch.setattr(assembly,'assemble_periodic_dyadic_system',matrix)
    monkeypatch.setattr(selected,'periodic_dyadic_system_block',block)
    simulation = mw.Simulation.mom(scene,frequency=3e8,config=config)
    binding = linear_simulation_binding(simulation)
    result = simulation.run()
    assert result.stats()['converged']
    payload = result.surface_currents.snapshot()
    assert payload['surface']['schema_version']==(18 if backend=='dense' else 19)
    assert payload['representation']['kernel_id']=='checked_quasiperiodic_dyadic_regular_v1'
    assert payload['representation']['operator_backend']==backend
    points = volumes[0].vertices.new_tensor([[.1,.2,.3]])
    expected = result.query_fields(points,region_id=0,field_kind='incident')
    root = tmp_path/'state'
    sha = write_linear_state(result,root,expected_binding=binding,
        physical_group_id='manufactured-periodic',split='pilot',source_id='mode')
    def forbidden(*args,**kwargs): raise AssertionError('Nested periodic restoration must only decode')
    monkeypatch.setattr(assembly,'assemble_periodic_dyadic_system',forbidden)
    monkeypatch.setattr(selected,'periodic_dyadic_system_block',forbidden)
    monkeypatch.setattr(HierarchicalOperator,'__init__',forbidden)
    monkeypatch.setattr(HierarchicalH2Operator,'__init__',forbidden)
    monkeypatch.setattr(torch.linalg,'solve',forbidden)
    _, restored = read_linear_state(root,expected_manifest_sha256=sha,expected_binding=binding,
        scene=scene,physical_group_id='manufactured-periodic',split='pilot',source_id='mode')
    for actual,field in zip(restored.query_fields(points,region_id=0,field_kind='incident'),expected):
        assert torch.equal(actual,field)
    assert not restored.surface_currents.representation_metadata()['physical_qualification']
    raw = torch.load(root/'result.pt', weights_only=True)
    assert raw['metadata']['last_field_query']==raw['surface_data']['representation']['last_field_query']
    raw['metadata']['last_field_query'] = {'converged':False}
    damaged = tmp_path/'changed-query.pt'; torch.save(raw,damaged)
    with pytest.raises(ValueError,match='metadata'):
        mw.Result.load(damaged,scene=scene)
