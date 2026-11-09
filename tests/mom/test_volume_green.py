"""Independent volume quadrature and distributional interior self-term gates."""
import math
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
from witwin.maxwell.mom.vie_operator import cell_green_potentials, cell_current_fields
from witwin.maxwell.mom.physics import compile_medium

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def mesh():
    vertices = torch.tensor([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]], device='cuda', dtype=torch.float64)*.1
    return TetrahedralVolumeMesh(vertices, torch.tensor([[0,1,2,3]], device='cuda'), region_id=1)


def test_static_interior_hessian_contains_distributional_contact_term():
    volume = mesh()
    points = volume.vertices.mean(0, keepdim=True)
    _, _, hessian = cell_green_potentials(volume, points, 0., order=8)
    torch.testing.assert_close(hessian.diagonal(dim1=-2,dim2=-1).sum(-1), hessian.new_full((1,1),-1.), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(hessian, hessian.mT, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match='trace'):
        cell_green_potentials(volume, volume.boundary.triangles[:1].mean(1), 0.)


def test_off_cell_potentials_match_independent_direct_volume_cubature():
    volume = mesh()
    point = volume.vertices.new_tensor([[.2, -.15, .25]])
    k = 7+.4j
    actual = cell_green_potentials(volume, point, k, order=10)
    nodes, weights = volume.quadrature(10)
    delta = point[:, None, None]-nodes[None]
    r = torch.linalg.vector_norm(delta, dim=-1)
    direction = delta/r[..., None]
    g = torch.exp(1j*k*r)/(4*math.pi*r)
    gp = g*(1j*k-1/r)
    gpp = g*(-k*k-2j*k/r+2/r.square())
    identity = torch.eye(3, device='cuda', dtype=torch.complex128)
    hessian = (gpp-gp/r)[..., None, None]*direction[..., :, None]*direction[..., None, :]+(gp/r)[...,None,None]*identity
    expected = ((g*weights).sum(-1), (gp[...,None]*direction*weights[...,None]).sum(-2),
                (hessian*weights[...,None,None]).sum(-3))
    for a,e in zip(actual,expected):
        torch.testing.assert_close(a,e,rtol=2e-10,atol=1e-12)
    current = torch.tensor([[1+.2j, -.3j, .5]], device='cuda', dtype=torch.complex128, requires_grad=True)
    medium = compile_medium(mw.Material(eps_r=2.), 3e8)
    electric, magnetic = cell_current_fields(volume,current,point,medium,order=10)
    gradient, = torch.autograd.grad(electric.real.sum()+magnetic.real.sum(),current)
    assert bool(torch.isfinite(gradient).all()) and gradient.abs().max() > 0


def test_interior_volume_current_satisfies_maxwell_source_equation():
    volume = mesh()
    point = volume.vertices.mean(0, keepdim=True).requires_grad_()
    medium = compile_medium(mw.Material(eps_r=2.), 3e8)
    current = torch.tensor([[1+.2j, -.3j, .5]], device='cuda', dtype=torch.complex128)
    electric, _ = cell_current_fields(volume, current, point, medium, order=14)
    def magnetic_real(query):
        return torch.view_as_real(cell_current_fields(volume, current, query, medium, order=14)[1][0])
    jacobian = torch.autograd.functional.jacobian(magnetic_real, point)
    jacobian = torch.view_as_complex(jacobian[:, :, 0, :].permute(0,2,1).contiguous())
    curl = torch.stack((jacobian[2,1]-jacobian[1,2], jacobian[0,2]-jacobian[2,0], jacobian[1,0]-jacobian[0,1]))
    source = curl+1j*medium.omega*medium.epsilon*electric[0]
    torch.testing.assert_close(source, current[0], rtol=2e-6, atol=2e-7)


def test_adaptive_volume_fields_keep_current_gradient_and_reject_budget_exhaustion():
    from witwin.maxwell.mom.vie_operator import evaluate_cell_current_fields, VolumeFieldConvergenceError
    from witwin.maxwell.mom.fields import FieldQueryConfig
    volume = mesh()
    points = torch.cat((volume.vertices.mean(0, keepdim=True), volume.vertices.new_tensor([[.2,-.15,.25]])))
    current = torch.tensor([[1+.2j, -.3j, .5]], device='cuda', dtype=torch.complex128, requires_grad=True)
    medium = compile_medium(mw.Material(eps_r=2.), 3e8)
    (electric, magnetic), receipt = evaluate_cell_current_fields(volume, current, points, medium,
        order=3, config=FieldQueryConfig(relative_tolerance=1e-7, absolute_tolerance=1e-10, max_order=48))
    assert receipt['converged'] and not receipt['physical_qualification']
    reference = cell_current_fields(volume, current, points, medium, order=48)
    for actual, expected in zip((electric, magnetic), reference):
        torch.testing.assert_close(actual, expected, rtol=2e-7, atol=1e-10)
    gradient, = torch.autograd.grad(electric.real.sum()+magnetic.real.sum(),current)
    assert torch.isfinite(gradient).all() and gradient.abs().max() > 0
    with pytest.raises(VolumeFieldConvergenceError) as failure:
        evaluate_cell_current_fields(volume, current, points, medium, order=2,
            config=FieldQueryConfig(relative_tolerance=1e-14, absolute_tolerance=1e-18, max_order=4))
    assert failure.value.receipt['failure_reason'] == 'volume_order_budget_exhausted'
    assert failure.value.receipt['checks'] and not failure.value.receipt['converged']


def test_volume_face_principal_value_matches_two_sided_limits_and_normal_jump():
    volume = mesh()
    triangle = volume.boundary.triangles[0]
    point = triangle.mean(0, keepdim=True)
    normal = volume.boundary.normals[0]
    medium = compile_medium(mw.Material(eps_r=2.), 3e8)
    current = torch.tensor([[1+.2j, -.3j, .5]], device='cuda', dtype=torch.complex128)
    principal = cell_current_fields(volume,current,point,medium,order=28,principal_value=True)
    electric_jump = 1j/(medium.omega*medium.epsilon)*normal*(normal*current[0]).sum()
    averages, jumps = [], []
    for distance in (2e-5, 1e-5):
        outside = cell_current_fields(volume,current,point+distance*normal,medium,order=28)
        inside = cell_current_fields(volume,current,point-distance*normal,medium,order=28)
        averages.append(tuple((a+b)/2 for a,b in zip(outside,inside)))
        jumps.append(outside[0][0]-inside[0][0])
    for value, coarse, fine in zip(principal,*averages):
        limit = 2*fine-coarse
        torch.testing.assert_close(value,limit,rtol=3e-5,atol=1e-8)
    torch.testing.assert_close(2*jumps[1]-jumps[0],electric_jump,rtol=3e-5,atol=1e-7)
    with pytest.raises(ValueError,match='edge'):
        cell_current_fields(volume,current,triangle[:2].mean(0,keepdim=True),medium,principal_value=True)


def test_volume_query_batch_boundaries_preserve_order_and_mixed_near_far_fields():
    volume = mesh()
    near = volume.vertices[volume.cells].mean(1)
    offsets = torch.arange(19,device='cuda',dtype=torch.float64)[:,None]*torch.tensor(
        [[.1,.13,.07]],device='cuda',dtype=torch.float64)
    points = near+offsets
    batched = cell_green_potentials(volume,points,2.7+.2j,order=4)
    separate = [cell_green_potentials(volume,p[None],2.7+.2j,order=4) for p in points]
    for index,value in enumerate(batched):
        expected = torch.cat([part[index] for part in separate])
        torch.testing.assert_close(value,expected,rtol=2e-12,atol=1e-14)


def test_dc_polarization_has_potential_gradient_and_independent_far_dipole_limit():
    from witwin.maxwell.mom.vie_operator import cell_static_polarization_fields
    from witwin.maxwell.constants import EPSILON_0
    volume = mesh()
    polarization = torch.tensor([[1.,-.3,.5]],device='cuda',dtype=torch.float64,requires_grad=True)*1e-9
    epsilon = torch.tensor(2*EPSILON_0,device='cuda',dtype=torch.float64,requires_grad=True)
    points = volume.vertices.mean(0,keepdim=True).clone().requires_grad_()
    (potential,electric),receipt = cell_static_polarization_fields(volume,polarization,points,epsilon)
    gradient, = torch.autograd.grad(potential.sum(),points,retain_graph=True)
    torch.testing.assert_close(electric,-gradient,rtol=1e-11,atol=1e-11)
    sensitivity, = torch.autograd.grad(electric.sum(),epsilon)
    torch.testing.assert_close(sensitivity,-electric.sum()/epsilon,rtol=1e-12,atol=0)
    assert receipt['distributional_contact_included'] and not receipt['interface_conditions_solved']
    displacement = volume.vertices.new_tensor([[12.,-8.,16.]])
    far = volume.vertices.mean(0,keepdim=True)+displacement
    (_,actual),_ = cell_static_polarization_fields(volume,polarization,far,epsilon)
    dipole = (polarization*volume.volumes[:,None]).sum(0)
    distance = torch.linalg.vector_norm(displacement,dim=-1)
    direction = displacement/distance[:,None]
    expected = (3*direction*(direction*dipole).sum(-1,keepdim=True)-dipole)/(4*math.pi*epsilon*distance[:,None]**3)
    torch.testing.assert_close(actual,expected,rtol=3e-5,atol=1e-13)
    with pytest.raises(ValueError,match='positive'):
        cell_static_polarization_fields(volume,polarization,far,0.)
