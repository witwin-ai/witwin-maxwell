"""Distributional contact, independent volume cubature and derivative controls."""
import math
import pytest
import torch
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
from witwin.maxwell.mom.static_tensor_green import StaticTensorGreen
from witwin.maxwell.mom.static_tensor_volume import cell_tensor_static_potentials
from witwin.maxwell.mom.vie_operator import cell_green_potentials

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def fixture():
    v=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],device='cuda',dtype=torch.float64)
    mesh=TetrahedralVolumeMesh(v,torch.tensor([[0,1,2,3]],device='cuda'),region_id=1)
    a=v.new_tensor([[2.,.3,.1],[.3,3.,.2],[.1,.2,1.5]])
    return mesh,a


def test_scalar_limit_self_contact_and_selected_columns():
    mesh,a=fixture();x=a.new_tensor([[.2,.2,.2],[1.5,.4,.7]])
    kernel=StaticTensorGreen(3*torch.eye(3,device=a.device,dtype=a.dtype))
    actual=cell_tensor_static_potentials(mesh,x,kernel,pair_batch=3)
    expected=cell_green_potentials(mesh,x,0)
    for value,reference in zip(actual,expected):
        torch.testing.assert_close(value,reference.real/3,rtol=2e-12,atol=1e-15)
    ids=mesh.cells.new_tensor([0,0])
    selected=cell_tensor_static_potentials(mesh,x,kernel,cell_indices=ids,pair_batch=1)
    for value,reference in zip(selected,actual):
        torch.testing.assert_close(value,reference.expand_as(value),rtol=2e-12,atol=1e-15)
    tensor=cell_tensor_static_potentials(mesh,x,StaticTensorGreen(a))
    contact=torch.einsum('ij,qcij->qc',a,tensor[2])
    torch.testing.assert_close(contact,a.new_tensor([[-1.],[0.]]),rtol=0,atol=2e-14)
    torch.testing.assert_close(tensor[2],tensor[2].transpose(-1,-2),rtol=0,atol=2e-14)
    empty=cell_tensor_static_potentials(mesh,x,StaticTensorGreen(a),cell_indices=ids[:0])
    assert [tuple(t.shape) for t in empty]==[(2,0),(2,0,3),(2,0,3,3)]


def test_physical_volume_cubature_and_native_observer_derivatives():
    mesh,a=fixture();x=a.new_tensor([[1.5,.7,1.1]])
    kernel=StaticTensorGreen(a)
    actual=cell_tensor_static_potentials(mesh,x,kernel,pair_batch=2)
    nodes,weights=mesh.quadrature(16)
    delta=x[:,None,None,:]-nodes[None]
    inverse=torch.linalg.inv(a);metric=delta@inverse
    rho2=(metric*delta).sum(-1)
    value=1/(4*math.pi*torch.linalg.det(a).sqrt()*rho2.sqrt())
    gradient=-metric*(value/rho2)[...,None]
    hessian=(3*metric[..., :,None]*metric[...,None,:]/rho2[...,None,None]-inverse)*(value/rho2)[...,None,None]
    for observed,integrand in zip(actual,(value,gradient,hessian)):
        w=weights.reshape(*weights.shape,*([1]*(integrand.ndim-3)))
        expected=(integrand*w).sum(2)
        torch.testing.assert_close(observed,expected,rtol=2e-11,atol=2e-14)
    point=x[0]
    derivative=torch.autograd.functional.jacobian(lambda p:cell_tensor_static_potentials(mesh,p[None],kernel)[0][0,0],point)
    second=torch.autograd.functional.hessian(lambda p:cell_tensor_static_potentials(mesh,p[None],kernel)[0][0,0],point)
    torch.testing.assert_close(derivative,actual[1][0,0],rtol=2e-11,atol=2e-14)
    torch.testing.assert_close(second,actual[2][0,0],rtol=2e-10,atol=2e-14)


def test_face_pv_jumps_edge_rejection_and_coefficient_gradient():
    mesh,a=fixture();kernel=StaticTensorGreen(a)
    face=a.new_tensor([[.2,.3,0.]])
    with pytest.raises(ValueError,match='explicit principal'):
        cell_tensor_static_potentials(mesh,face,kernel)
    pv=cell_tensor_static_potentials(mesh,face,kernel,principal_value=True)[2]
    offset=a.new_tensor([[0.,0.,1e-7]])
    plus=cell_tensor_static_potentials(mesh,face+offset,kernel)[2]
    minus=cell_tensor_static_potentials(mesh,face-offset,kernel)[2]
    torch.testing.assert_close(pv,.5*(plus+minus),rtol=2e-6,atol=2e-8)
    normal=a.new_tensor([0.,0.,1.])
    jump=-normal[:,None]*normal[None,:]/(normal@a@normal)
    torch.testing.assert_close((plus-minus)[0,0],jump,rtol=2e-6,atol=2e-7)
    torch.testing.assert_close(torch.einsum('ij,qcij->qc',a,pv),a.new_tensor([[-.5]]),rtol=0,atol=2e-14)
    with pytest.raises(ValueError,match='edges'):
        cell_tensor_static_potentials(mesh,a.new_tensor([[.3,0.,0.]]),kernel,principal_value=True)
    strength=a.new_tensor(.2,requires_grad=True)
    x=a.new_tensor([[.2,.2,.2]])
    def objective(t):
        c=a+t*torch.eye(3,device=a.device,dtype=a.dtype)
        return cell_tensor_static_potentials(mesh,x,StaticTensorGreen(c))[2].square().sum()
    derivative=torch.autograd.grad(objective(strength),strength)[0]
    step=1e-5;finite=(objective(strength.detach()+step)-objective(strength.detach()-step))/(2*step)
    torch.testing.assert_close(derivative,finite,rtol=2e-8,atol=1e-12)
    kernel.factor.add_(.1)
    with pytest.raises(ValueError,match='changed'):cell_tensor_static_potentials(mesh,x,kernel)


def test_rotation_covariance_and_physical_constitutive_scaling():
    mesh,a=fixture();x=a.new_tensor([[.2,.2,.2],[1.3,.6,.4]])
    original=cell_tensor_static_potentials(mesh,x,StaticTensorGreen(a),pair_batch=3)
    q,_=torch.linalg.qr(a.new_tensor([[1.,2.,3.],[0.,1.,2.],[2.,1.,1.]]))
    # Preserve tetrahedron orientation even if QR returns a reflection.
    if bool(torch.linalg.det(q)<0):q=q.clone();q[:,0]=-q[:,0]
    rotated=TetrahedralVolumeMesh(mesh.vertices@q.T,mesh.cells,region_id=1)
    coefficient=q@a@q.T;coefficient=(coefficient+coefficient.T)/2
    result=cell_tensor_static_potentials(rotated,x@q.T,StaticTensorGreen(coefficient),pair_batch=1)
    for observed,expected in zip(result,(original[0],original[1]@q.T,q@original[2]@q.T)):
        torch.testing.assert_close(observed,expected,rtol=2e-11,atol=2e-14)
    for scale in (1e-11,1e6):
        scaled=cell_tensor_static_potentials(mesh,x,StaticTensorGreen(a*scale))
        for observed,expected in zip(scaled,original):
            torch.testing.assert_close(observed*scale,expected,rtol=2e-11,atol=2e-14)
