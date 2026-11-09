"""Independent isolated-current solutions, temporal moments and native gradients."""
import pytest
import torch
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
from witwin.maxwell.mom.static_current import NodalCurrentBalance,StaticCurrentConvergenceError

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def mesh(dtype=torch.float64):
    vertices = torch.tensor([[.01,-.02,.03],[.11,-.02,.03],[.03,.09,.03],[.02,.01,.15]],
                            device='cuda',dtype=dtype)
    return TetrahedralVolumeMesh(vertices,torch.tensor([[0,1,2,3]],device='cuda'),region_id=1).refine(1)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_ohmic_affine_solution_gauge_and_conductivity_gradient(dtype):
    volume = mesh(dtype)
    block = NodalCurrentBalance(volume)
    gradient = volume.vertices.new_tensor([.7,-.3,.4])
    sigma = volume.vertices.new_tensor(.8,requires_grad=True)
    source = (sigma.detach()*gradient).expand(len(volume.cells),3)
    ac = source.new_zeros((0,len(volume.cells),3),dtype=torch.complex64 if dtype == torch.float32 else torch.complex128)
    tolerance = 2e-8 if dtype == torch.float32 else 1e-15
    result = block.solve(ac,conductivity=sigma,source_current=source,absolute_tolerance=tolerance)
    centroid = (volume.volumes[:,None]*volume.vertices[volume.cells].mean(1)).sum(0)/volume.volumes.sum()
    expected = (volume.vertices-centroid)@gradient
    rtol = 2e-5 if dtype == torch.float32 else 1e-11
    torch.testing.assert_close(result.potential,expected,rtol=rtol,atol=rtol*1e-2)
    torch.testing.assert_close(result.electric,-gradient.expand_as(source),rtol=rtol,atol=rtol*1e-2)
    torch.testing.assert_close(result.mean_current,torch.zeros_like(source),rtol=0,atol=rtol)
    derivative, = torch.autograd.grad(result.electric[:,0].mean(),sigma)
    torch.testing.assert_close(derivative,gradient[0]/sigma,rtol=rtol,atol=rtol)
    shifted = block.solve(ac,conductivity=sigma,source_current=source,initial_potential=expected+3.,
                          absolute_tolerance=tolerance)
    torch.testing.assert_close(shifted.potential,result.potential,rtol=rtol,atol=rtol*1e-2)
    assert not result.receipt['neutral_body_charge_enforced']
    assert not result.receipt['physical_qualification']


@pytest.mark.parametrize('conductivity',[0.,.2])
def test_two_frequency_tpa_rectification_cancels_mean_current_and_has_implicit_gradient(conductivity):
    volume = mesh()
    block = NodalCurrentBalance(volume)
    amplitude = volume.vertices.new_tensor(1.4)
    second = volume.vertices.new_tensor(.3,requires_grad=True)
    axis = volume.vertices.new_tensor([1.,0.,0.])
    ac = (torch.stack((amplitude,second))[:,None,None]*axis[None,None,:]).expand(-1,len(volume.cells),-1).to(torch.complex128)
    scale = .7
    result = block.solve(ac,conductivity=conductivity,tpa_scale=scale,absolute_tolerance=1e-15)
    # Independent scalar cubic from cos(wt) and cos(2wt), solved by bisection.
    low,high = -1.,1.
    for _ in range(80):
        mid = (low+high)/2
        residual = conductivity*mid+scale*(mid**3+1.5*(1.4**2+.3**2)*mid+.75*1.4**2*.3)
        if residual > 0:
            high = mid
        else:
            low = mid
    root = (low+high)/2
    torch.testing.assert_close(result.electric[:,0],result.electric.new_full((len(volume.cells),),root),rtol=1e-11,atol=1e-13)
    assert abs(root) > .1
    derivative, = torch.autograd.grad(result.electric[:,0].mean(),second)
    expected_derivative = -scale*(3*.3*root+.75*1.4**2)/(conductivity+scale*(3*root**2+1.5*(1.4**2+.3**2)))
    torch.testing.assert_close(derivative,derivative.new_tensor(expected_derivative),rtol=1e-10,atol=1e-12)
    phase = torch.arange(4096,device='cuda',dtype=torch.float64)*(2*torch.pi/4096)
    real_ac = torch.einsum('hn,hci->nci',torch.stack((phase.cos(),(2*phase).cos())),ac.real)
    total = result.electric[None]+real_ac
    mean = conductivity*result.electric+scale*(total.square().sum(-1)[...,None]*total).mean(0)
    torch.testing.assert_close(result.mean_current,mean,rtol=1e-9,atol=1e-13)
    assert mean.norm() < 1e-12


def test_vector_tpa_weak_flux_identity_and_singular_dielectric_rejection():
    volume = mesh()
    block = NodalCurrentBalance(volume)
    generator = torch.Generator(device='cuda').manual_seed(27)
    ac = torch.randn((2,len(volume.cells),3),device='cuda',dtype=torch.complex128,generator=generator)*.4
    result = block.solve(ac,conductivity=.05,tpa_scale=.8,absolute_tolerance=1e-15)
    phase = torch.arange(512,device='cuda',dtype=torch.float64)*(2*torch.pi/512)
    oscillation = torch.exp(-1j*torch.arange(1,3,device='cuda')[:,None]*phase)
    total = result.electric[None]+torch.einsum('hn,hci->nci',oscillation,ac).real
    expected_current = .05*result.electric+.8*(total.square().sum(-1)[...,None]*total).mean(0)
    torch.testing.assert_close(result.mean_current,expected_current,rtol=1e-11,atol=1e-13)
    # Independent face-normal integration of linear test functions; all element
    # faces are included, so internal current jumps are not silently discarded.
    boundary_residual = volume.vertices.new_zeros(len(volume.vertices))
    for local in ((1,2,3),(0,3,2),(0,1,3),(0,2,1)):
        faces = volume.cells[:,local]
        points = volume.vertices[faces]
        area_normal = torch.linalg.cross(points[:,1]-points[:,0],points[:,2]-points[:,0])/2
        integrated = (area_normal*expected_current).sum(-1)/3
        boundary_residual = boundary_residual.index_add(0,faces.flatten(),integrated[:,None].expand(-1,3).flatten())
    torch.testing.assert_close(block.weak_current(result.mean_current),boundary_residual,rtol=1e-8,atol=1e-15)
    assert boundary_residual.norm() < 1e-13
    assert abs(float((volume.volumes*(result.electric*result.mean_current).sum(-1)).sum())) < 1e-14
    with pytest.raises(StaticCurrentConvergenceError,match='singular_current_tangent'):
        block.solve(torch.zeros_like(ac))
    with pytest.raises(StaticCurrentConvergenceError,match='singular_current_tangent'):
        block.solve(torch.zeros_like(ac),tpa_scale=.8,initial_potential=volume.vertices[:,0])
    with pytest.raises(StaticCurrentConvergenceError,match='iteration_budget_exhausted'):
        block.solve(ac,tpa_scale=.8,max_iterations=0)
    volume.vertices.add_(.01)
    with pytest.raises(ValueError,match='geometry changed'):
        block.weak_current(result.mean_current)
