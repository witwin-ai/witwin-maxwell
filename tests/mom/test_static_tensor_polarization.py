"""Anisotropic volume dipole quadrature and analytic cell self-field controls."""
import pytest
import torch
from test_static_coupling import system_from_volume
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
from witwin.maxwell.mom.static_tensor_polarization import StaticTensorPolarization
from witwin.maxwell.mom.vie_operator import cell_static_polarization_fields

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def source(*,refine=0,scalar=False):
    vertices=torch.tensor([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.],[0.,0.,.1]],device='cuda',dtype=torch.float64)
    volume=TetrahedralVolumeMesh(vertices,torch.tensor([[0,1,2,3]],device='cuda'),region_id=1).refine(refine)
    _,binding=system_from_volume(volume)
    a=2*torch.eye(3,device='cuda',dtype=torch.float64) if scalar else vertices.new_tensor([[2.,.3,.1],[.3,3.,.2],[.1,.2,1.5]])
    a.requires_grad_()
    p=vertices.new_tensor([[1.,-.3,.5]]).expand(len(volume.cells),3).clone().requires_grad_()
    return StaticTensorPolarization(binding,a,p,pair_batch=17),a,p


@pytest.mark.parametrize('refine',(0,1))
def test_anisotropic_particular_matches_independent_volume_dipole_quadrature(refine):
    base,a,p=source(refine=refine)
    p=(p*torch.linspace(.7,1.4,len(p),device=p.device,dtype=p.dtype)[:,None]).detach().requires_grad_()
    response=StaticTensorPolarization(base.binding,a,p,pair_batch=17)
    point=a.new_tensor([[.17,.13,.19],[-.08,.16,.12]],requires_grad=True)
    potential,field=response.fields(point)
    nodes,weights=response.binding.volume.quadrature(12)
    displacement=point[:,None,None,:]-nodes[None]
    inverse=torch.linalg.inv(a)
    mapped=displacement@inverse.T
    radius=(displacement*mapped).sum(-1).sqrt()
    gradient=-mapped/(4*torch.pi*torch.linalg.det(a).sqrt()*radius[...,None]**3)
    expected=-(gradient*p[None,:,None,:]).sum(-1).mul(weights).sum((1,2))
    expected_gradient,=torch.autograd.grad(expected.sum(),point,create_graph=True)
    torch.testing.assert_close(potential,expected,rtol=2e-11,atol=1e-14)
    torch.testing.assert_close(field,-expected_gradient,rtol=2e-10,atol=1e-13)
    own_gradient,=torch.autograd.grad(potential.sum(),point,retain_graph=True)
    torch.testing.assert_close(field,-own_gradient,rtol=2e-11,atol=1e-13)
    da,dp=torch.autograd.grad(field.sum(),(a,p))
    assert bool(torch.isfinite(da).all() & torch.isfinite(dp).all())
    assert bool(da.abs().max()>0) and bool(dp.abs().max()>0)


def test_scalar_self_field_and_internal_face_cancellation():
    coarse,_,p=source(scalar=True)
    fine,_,_=source(refine=1,scalar=True)
    points=p.new_tensor([[.021,.019,.023],[.13,.12,.11]])
    expected,_=cell_static_polarization_fields(coarse.binding.volume,p,points,2.)
    actual=coarse.fields(points)
    for got,want in zip(actual,expected):torch.testing.assert_close(got,want,rtol=2e-11,atol=1e-12)
    for got,want in zip(fine.fields(points),actual):torch.testing.assert_close(got,want,rtol=2e-11,atol=1e-12)


def test_anisotropic_polarization_face_jump_and_owned_state():
    response,a,p=source()
    panel=response.triangles[:1];point=panel.mean(1);normal=response.normals[:1]
    value,mean=response.fields(point,principal_value=True)
    _,outside=response.fields(point+1e-7*normal)
    _,inside=response.fields(point-1e-7*normal)
    expected=(normal*p).sum(-1)
    jump=((outside-inside)@a.T*normal).sum(-1)
    torch.testing.assert_close(jump,expected,rtol=2e-5,atol=1e-7)
    torch.testing.assert_close((outside+inside)/2,mean,rtol=2e-5,atol=1e-7)
    with pytest.raises(ValueError,match='principal values'):response.fields(point)
    with pytest.raises(ValueError,match='edges'):response.fields(panel[:,0],principal_value=True)
    v,e=response.fields(a.new_empty(0,3,requires_grad=True))
    assert v.shape==(0,) and e.shape==(0,3) and e.requires_grad
    response.polarization.detach().add_(1)
    with pytest.raises(ValueError,match='changed'):response.fields(point)
