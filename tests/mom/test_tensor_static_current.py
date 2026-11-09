"""Manufactured isolated current balance with actual tensor conductivities."""
import pytest
import torch
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
from witwin.maxwell.mom.static_current import NodalCurrentBalance,FaceCurrentBalance,StaticCurrentConvergenceError

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def volume():
    vertices=torch.tensor([[0.,0.,0.],[.07,0.,0.],[0.,.07,0.],[0.,0.,.07]],device='cuda',dtype=torch.float64)
    return TetrahedralVolumeMesh(vertices,torch.tensor([[0,1,2,3]],device='cuda'),region_id=1)


@pytest.mark.parametrize('kind',[NodalCurrentBalance,FaceCurrentBalance])
def test_tensor_current_uniform_field_and_implicit_derivative(kind):
    mesh=volume();block=kind(mesh)
    strength=torch.tensor(.4,device='cuda',dtype=torch.float64,requires_grad=True)
    base=mesh.vertices.new_tensor([[2.,.2,.1],[.2,1.5,.3],[.1,.3,1.]])
    sigma=base+strength*torch.eye(3,device='cuda',dtype=torch.float64)
    source=mesh.vertices.new_tensor([[.1,-.2,.3]])
    ac=torch.zeros(1,1,3,device='cuda',dtype=torch.complex128)
    result=block.solve(ac,conductivity=sigma,source_current=source)
    expected=-torch.linalg.solve(sigma,source[0])
    torch.testing.assert_close(result.electric[0],expected,rtol=1e-11,atol=1e-12)
    torch.testing.assert_close(result.mean_current,torch.zeros_like(source),rtol=0,atol=1e-12)
    derivative,=torch.autograd.grad(result.electric[0].sum(),strength)
    torch.testing.assert_close(derivative,-torch.linalg.solve(sigma,expected).sum(),rtol=1e-10,atol=1e-12)
    assert result.receipt['converged'] and not result.receipt['physical_qualification']


def test_cell_tensor_conservation_tpa_and_rejection():
    mesh=volume().refine(1);block=FaceCurrentBalance(mesh)
    n=len(mesh.cells);eye=torch.eye(3,device='cuda',dtype=torch.float64)
    sigma=eye.expand(n,3,3).clone()
    sigma[:,0,0]=torch.linspace(1.,2.,n,device='cuda',dtype=torch.float64)
    sigma[:,0,1]=sigma[:,1,0]=.2
    field=mesh.vertices.new_tensor([.1,.2,-.15]).expand(n,3)
    source=-torch.einsum('tij,tj->ti',sigma,field)
    ac=torch.zeros(1,n,3,device='cuda',dtype=torch.complex128)
    solved=block.solve(ac,conductivity=sigma,source_current=source)
    torch.testing.assert_close(solved.electric,field,rtol=1e-10,atol=1e-11)
    torch.testing.assert_close(solved.mean_current,torch.zeros_like(field),rtol=0,atol=1e-11)
    # A uniform prescribed DC field plus isotropic cubic loss has a known
    # cancelling source; this checks tensor Ohmic and nonlinear current together.
    source=source-.3*field.square().sum(-1,keepdim=True)*field
    solved=block.solve(ac,conductivity=sigma,tpa_scale=.3,source_current=source)
    torch.testing.assert_close(solved.electric,field,rtol=1e-9,atol=1e-10)
    bad=sigma.clone();bad[0,0,1]+=.1
    with pytest.raises(ValueError,match='symmetric'):block.solve(ac,conductivity=bad)
    bad=sigma.clone();bad[0,2,2]=-1
    with pytest.raises(ValueError,match='positive semidefinite'):block.solve(ac,conductivity=bad)
    with pytest.raises(StaticCurrentConvergenceError,match='singular_current_tangent'):
        block.solve(ac,conductivity=torch.zeros_like(sigma))
