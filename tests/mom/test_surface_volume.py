"""Owning-region volume RHS and zero-source reduction to existing PMCHWT."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import SurfaceMesh, RWGBasis, SurfaceSystem
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh, bind_volume_region
from witwin.maxwell.mom.runtime import SystemOperator
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.fields import FieldQueryConfig
from witwin.maxwell.mom.linalg import solve_reference
from witwin.maxwell.mom.surface_volume import build_volume_source_rhs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def problem(epsilons=(2.,2.)):
    vertices = torch.tensor([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]], device='cuda', dtype=torch.float64)*.07
    cells = torch.tensor([[0,1,2,3]], device='cuda')
    volumes = [TetrahedralVolumeMesh(vertices+shift, cells, region_id=i+1) for i,shift in enumerate((0., .3))]
    materials = [compile_medium(mw.Material(eps_r=value),3e8) for value in epsilons]
    entries = [(i, f'body{i}', RWGBasis(SurfaceMesh(v.boundary.vertices.clone(),v.boundary.faces.clone())),materials[i])
               for i,v in enumerate(volumes)]
    system = SurfaceSystem.create(entries)
    config = mw.MoMConfig(integration=IntegrationConfig(order=3,singular_order=3),riesz_storage='compact_rwg_v1')
    operator = SystemOperator(system,compile_medium(mw.Material(),3e8),[mw.PlaneWave()],config)
    return operator, [bind_volume_region(v,system) for v in volumes]


def test_heterogeneous_background_double_count_falsification():
    from witwin.maxwell.mom.lifting_fields import decode_system
    operator,bindings = problem((2.,3.))
    rhs = operator.build_rhs()
    original,receipt = solve_reference(operator,rhs)
    assert receipt['converged']
    binding = bindings[1]
    medium = operator.system.bodies[1].interior
    points = binding.volume.vertices[binding.volume.cells].mean(1)
    electric,_ = decode_system(operator,original,points,2,'total',
        query_config=FieldQueryConfig(adaptive=False))
    # Deliberately count the already represented linear contrast a second time.
    duplicate_current = -1j*medium.omega*(medium.epsilon-operator.exterior.epsilon)*electric
    extra,_ = build_volume_source_rhs(operator,[binding],[duplicate_current],order=2,
        query_config=FieldQueryConfig(relative_tolerance=1e-5,absolute_tolerance=1e-10,max_order=64))
    corrupted,receipt = solve_reference(operator,rhs+extra)
    assert receipt['converged']
    roundoff = 512*torch.finfo(points.dtype).eps
    assert (corrupted-original).norm() > roundoff*original.norm()
    assert (operator.apply(corrupted)-rhs).norm() > roundoff*rhs.norm()
    outside = points.new_tensor([[.5,.1,.1]])
    actual = decode_system(operator,corrupted,outside,0,'scattered',query_config=FieldQueryConfig(adaptive=False))
    reference = decode_system(operator,original,outside,0,'scattered',query_config=FieldQueryConfig(adaptive=False))
    assert (actual[0]-reference[0]).norm() > roundoff*reference[0].norm()


def test_zero_volume_source_reduces_exactly_and_nonzero_source_has_local_rhs_ownership():
    operator, bindings = problem()
    zero = torch.zeros(1,3,device='cuda',dtype=torch.complex128)
    settings = FieldQueryConfig(relative_tolerance=1e-5,absolute_tolerance=1e-10,max_order=64)
    addition, receipt = build_volume_source_rhs(operator,bindings,[zero,zero],order=2,query_config=settings)
    assert torch.count_nonzero(addition) == 0 and not receipt['linear_contrast_added']
    original_rhs = operator.build_rhs()
    original, _ = solve_reference(operator,original_rhs)
    coupled, _ = solve_reference(operator,original_rhs+addition)
    torch.testing.assert_close(coupled,original,rtol=0,atol=0)
    current = torch.tensor([[1+.2j,-.3j,.5]],device='cuda',dtype=torch.complex128,requires_grad=True)
    local, receipt = build_volume_source_rhs(operator,[bindings[1]],[current],order=2,query_config=settings)
    assert torch.count_nonzero(local[operator.system.J_slices[0]]) == 0
    assert torch.count_nonzero(local[operator.system.M_slices[0]]) == 0
    assert torch.linalg.vector_norm(local) > 0
    assert receipt['body_sources'][0]['region_id'] == 2
    gradient, = torch.autograd.grad(local.real.sum(),current)
    assert torch.isfinite(gradient).all() and gradient.abs().max() > 0
    with pytest.raises(ValueError,match='only one'):
        build_volume_source_rhs(operator,[bindings[0],bindings[0]],[zero,zero],order=2,query_config=settings)


def test_dense_volume_response_native_gradient_and_regional_field_composition():
    from witwin.maxwell.mom.surface_volume import DenseVolumeResponse
    from witwin.maxwell.mom.lifting_fields import decode_system
    from witwin.maxwell.mom.vie_operator import evaluate_cell_current_fields
    operator, bindings = problem()
    scale = torch.tensor(.7,device='cuda',dtype=torch.float64,requires_grad=True)
    current = torch.tensor([[1+.2j,-.3j,.5]],device='cuda',dtype=torch.complex128)*scale
    settings = FieldQueryConfig(relative_tolerance=1e-5,absolute_tolerance=1e-10,max_order=64)
    response = DenseVolumeResponse(operator,[bindings[1]],[current],order=2,query_config=settings)
    assert response.coefficients.requires_grad
    assert not response.receipt['linear_contrast_added']
    original, log = solve_reference(operator,operator.build_rhs())
    assert log['converged']
    derivative, = torch.autograd.grad(response.coefficients.real.sum(),scale,retain_graph=True)
    expected = (response.coefficients-original).real.sum()/scale
    torch.testing.assert_close(derivative,expected,rtol=1e-10,atol=1e-12)
    config = FieldQueryConfig(adaptive=False)
    for region in (0,1,2):
        point = torch.tensor([[.5,.1,.1]] if region == 0 else [[.017,.016,.015]],
                             device='cuda',dtype=torch.float64)
        if region == 2:
            point = point+.3
        actual = response.fields(point,region,query_config=config)
        surface = decode_system(operator,response.coefficients,point,region,'total',query_config=config)
        if region == 2:
            volume,_ = evaluate_cell_current_fields(bindings[1].volume,current,point,
                operator.system.bodies[1].interior,order=2,config=config)
            expected = tuple(a+b for a,b in zip(surface,volume))
        else:
            expected = surface
        for a,b in zip(actual,expected):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        gradient, = torch.autograd.grad(actual[0].real.sum(),scale,retain_graph=True)
        assert torch.isfinite(gradient) and gradient.abs() > 0
    with torch.no_grad():
        response.currents[0].add_(1)
    with pytest.raises(RuntimeError,match='mutated'):
        response.fields(point,2,query_config=config)


def test_outer_volume_testing_zero_and_real_budget_failure():
    from witwin.maxwell.mom.surface_volume import adaptive_volume_source_rhs, VolumeTestingConvergenceError
    operator, bindings = problem()
    zero = torch.zeros(1,3,device='cuda',dtype=torch.complex128,requires_grad=True)
    settings = FieldQueryConfig(relative_tolerance=1e-5,absolute_tolerance=1e-10,max_order=64)
    rhs, receipt = adaptive_volume_source_rhs(operator,[bindings[0]],[zero],order=2,
        max_order=4,query_config=settings)
    assert receipt['converged'] and receipt['final_order'] == 4
    assert torch.count_nonzero(rhs) == 0
    assert rhs.requires_grad
    current = zero.detach()+torch.tensor([[1+.2j,-.3j,.5]],device='cuda',dtype=torch.complex128)
    with pytest.raises(VolumeTestingConvergenceError) as error:
        adaptive_volume_source_rhs(operator,[bindings[0]],[current],order=2,max_order=4,
            relative_tolerance=1e-14,absolute_tolerance=1e-20,query_config=settings)
    receipt = error.value.receipt
    assert not receipt['converged'] and receipt['final_order'] == 4
    assert receipt['checks'][-1]['absolute_difference_dual_proxy'] > receipt['checks'][-1]['threshold']


def test_complete_surface_volume_field_adaptation_and_budget_receipts():
    import copy
    from dataclasses import replace
    from witwin.maxwell.mom.surface_volume import DenseVolumeResponse, CoupledFieldConvergenceError
    operator, bindings = problem()
    current = torch.tensor([[1+.2j,-.3j,.5]],device='cuda',dtype=torch.complex128,requires_grad=True)
    response = DenseVolumeResponse(operator,[bindings[1]],[current],order=2,
        query_config=FieldQueryConfig(relative_tolerance=1e-5,absolute_tolerance=1e-10,max_order=64))
    assert response.receipt['outer_testing']['converged']
    print({'outer_testing': response.receipt['outer_testing']['checks']})
    point = torch.tensor([[.317,.316,.315]],device='cuda',dtype=torch.float64)
    adaptive = FieldQueryConfig(relative_tolerance=1e-5,absolute_tolerance=1e-10,max_order=48)
    actual = response.fields(point,'body1',query_config=adaptive)
    receipt = response.last_field_receipt
    print({'complete_field': receipt['checks']})
    assert receipt['converged'] and receipt['checks'][-1]['E']['passed'] and receipt['checks'][-1]['H']['passed']
    assert not receipt['complete_field_error_certified']
    reference_operator = copy.copy(operator)
    reference_operator.body_handles = tuple(copy.copy(h) for h in operator.body_handles)
    for h in reference_operator.body_handles:
        h.integration = replace(h.integration,order=64,singular_order=64,near_max_order=66)
    from witwin.maxwell.mom.surface_volume import VolumeFieldResponse
    reference = VolumeFieldResponse(reference_operator,response.bindings,response.currents,
        response.coefficients,order=64)
    expected = reference.fields(point,2,query_config=FieldQueryConfig(adaptive=False))
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,rtol=2e-5,atol=1e-10)
    derivative, = torch.autograd.grad(actual[0].real.sum(),current)
    assert torch.isfinite(derivative).all() and derivative.abs().max() > 0
    with pytest.raises(CoupledFieldConvergenceError) as failure:
        response.fields(point,2,query_config=FieldQueryConfig(relative_tolerance=1e-14,
            absolute_tolerance=1e-20,max_order=6))
    assert failure.value.receipt['failure_reason'] == 'coupled_field_order_budget_exhausted'


def test_multibody_exterior_field_refinement_constructs_valid_high_order_config():
    from dataclasses import replace
    from witwin.maxwell.mom.lifting_fields import decode_system
    operator, _ = problem()
    operator.integration = replace(operator.integration,order=12,singular_order=12,near_max_order=16)
    for local in operator.body_handles:
        local.integration = operator.integration
    coefficients = torch.arange(operator.shape[0],device='cuda',dtype=torch.float64).to(torch.complex128)*1e-4
    points = torch.tensor([[.8,.4,.2]],device='cuda',dtype=torch.float64)
    result = decode_system(operator,coefficients,points,0,'total',query_config=FieldQueryConfig(
        relative_tolerance=1e-7,absolute_tolerance=1e-10,max_order=48))
    assert operator.last_field_query_log['converged']
    assert operator.last_field_query_log['final_order'] >= 24
    assert all(torch.isfinite(value).all() for value in result)
    assert all(local.integration.near_max_order == 16 for local in operator.body_handles)


def test_volume_source_matrix_matches_action_and_column_ownership():
    from witwin.maxwell.mom.volume_response_matrix import build_volume_source_matrix
    operator,bindings = problem()
    current = torch.tensor([[1+.2j,-.3j,.5],[.3-.1j,.2j,-.7]],device='cuda',dtype=torch.complex128,requires_grad=True)
    matrix,receipt = build_volume_source_matrix(operator,bindings,order=3,inner_order=32)
    expected,source_receipt = build_volume_source_rhs(operator,bindings,[current[:1],current[1:]],order=3,
        query_config=FieldQueryConfig(relative_tolerance=1e-6,absolute_tolerance=1e-11,max_order=64))
    actual = matrix@current.flatten()
    torch.testing.assert_close(actual,expected,rtol=2e-6,atol=1e-10)
    assert torch.count_nonzero(matrix[operator.system.J_slices[0],3:]) == 0
    assert torch.count_nonzero(matrix[operator.system.M_slices[1],:3]) == 0
    assert receipt['bodies'][1]['column_start'] == 3 and not receipt['linear_contrast_added']
    gradient, = torch.autograd.grad(actual.real.sum(),current)
    assert torch.isfinite(gradient).all() and gradient.abs().max() > 0


def test_volume_source_matrix_independent_inner_outer_column_convergence():
    from witwin.maxwell.mom.volume_response_matrix import adaptive_volume_source_matrix,VolumeMatrixConvergenceError
    operator,bindings = problem()
    matrix,receipt = adaptive_volume_source_matrix(operator,[bindings[0]])
    print({'matrix_convergence':receipt})
    assert receipt['converged'] and matrix.shape == (operator.shape[0],3)
    assert any(item['axis'] == 'inner' for item in receipt['checks'])
    assert any(item['axis'] == 'outer' for item in receipt['checks'])
    with pytest.raises(VolumeMatrixConvergenceError) as failure:
        adaptive_volume_source_matrix(operator,[bindings[0]],max_order=4,max_inner_order=4,
            relative_tolerance=1e-14,absolute_tolerance=1e-20)
    assert not failure.value.receipt['converged']


def test_prepared_actual_cell_response_matches_direct_induced_current_solve():
    from witwin.maxwell.mom.volume_response_matrix import prepare_cell_field_response
    from witwin.maxwell.mom.surface_volume import DenseVolumeResponse
    operator,bindings = problem()
    settings = FieldQueryConfig(relative_tolerance=1e-5,absolute_tolerance=1e-10,max_order=48)
    prepared = prepare_cell_field_response(operator,[bindings[0]],query_config=settings)
    current = torch.tensor([[.2,.1j,-.3]],device='cuda',dtype=torch.complex128,requires_grad=True)
    electric = prepared['background_electric']+(prepared['electric_response']@current.flatten()).reshape(1,3)
    magnetic = prepared['background_magnetic']+(prepared['magnetic_response']@current.flatten()).reshape(1,3)
    direct = DenseVolumeResponse(operator,[bindings[0]],[current],order=2,
        query_config=FieldQueryConfig(relative_tolerance=1e-5,absolute_tolerance=1e-10,max_order=64))
    reference = direct.fields(prepared['points'],1,query_config=settings)
    for actual,expected in zip((electric,magnetic),reference):
        relative = torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected)
        assert relative < 3e-4
    derivative, = torch.autograd.grad(electric.real.sum(),current)
    assert torch.isfinite(derivative).all() and derivative.abs().max() > 0
    assert prepared['receipt']['converged'] and not prepared['receipt']['physical_qualification']
    print({'prepared_field_checks':prepared['receipt']['checks']})
