"""Physical complex cell integrals and summed-cell convergence controls."""
import pytest
import torch
from test_static_tensor_volume import fixture
from witwin.maxwell.constants import EPSILON_0
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.static_tensor_green import StaticTensorGreen
from witwin.maxwell.mom.complex_static_tensor_green import ComplexStaticTensorGreen
from witwin.maxwell.mom.static_tensor_volume import (cell_tensor_static_potentials,
    cell_complex_static_potentials,ComplexStaticCellConvergenceError)
from witwin.maxwell.mom.dyadic_green import electrostatic_kernel
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def kernel(tensor):
    return ComplexStaticTensorGreen(TensorMedium(3e7,tensor,
        torch.eye(3,device=tensor.device,dtype=tensor.dtype)))


@pytest.mark.parametrize('factor',[2+.3j,-2+.3j])
def test_scalar_times_spd_cell_distribution_pv_and_near_face(factor):
    mesh,a=fixture();k=kernel(factor*a)
    points=a.new_tensor([[.2,.2,.2],[1.5,.4,.7],[.2,.3,0.],[.2,.3,1e-7]])
    expected=cell_tensor_static_potentials(mesh,points,StaticTensorGreen(EPSILON_0*a),principal_value=True)
    values,receipt=cell_complex_static_potentials(mesh,points,k,principal_value=True,
        relative_tolerance=1e-9,pair_batch=5)
    assert receipt['converged'] and receipt['contact_distribution_included'] and not receipt['error_certified']
    assert set(receipt['checks'][-1])=={'order','potential','gradient','hessian'}
    for actual,reference in zip(values,expected):
        torch.testing.assert_close(actual,reference.to(actual.dtype)/factor,rtol=3e-8,atol=1.)
    contact=torch.einsum('ij,qcij->qc',k.coefficient,values[2])
    torch.testing.assert_close(contact,contact.new_tensor([[-1.],[0.],[-.5],[-1.]]),rtol=0,atol=3e-9)


def test_complex_cell_independent_volume_cubature_and_native_derivatives(monkeypatch):
    mesh,a=fixture();loss=a.new_tensor([[.4,.03,-.02],[.03,.2,.01],[-.02,.01,.3]])
    k=kernel(a+1j*loss);points=a.new_tensor([[1.5,.7,1.1]])
    def forbidden(*args,**kwargs):raise AssertionError('Cells must refine their summed quantities directly')
    monkeypatch.setattr(k,'integrate_panels',forbidden)
    values,receipt=cell_complex_static_potentials(mesh,points,k,relative_tolerance=1e-10)
    nodes,weights=mesh.quadrature(16)
    direct=electrostatic_kernel((points[:,None,None]-nodes[None]).reshape(-1,3),k.medium)
    for actual,integrand in zip(values,direct):
        integrand=integrand.reshape(1,*weights.shape,*actual.shape[2:])/EPSILON_0
        w=weights.reshape(1,*weights.shape,*([1]*(actual.ndim-2)))
        reference=(integrand*w).sum(2)
        torch.testing.assert_close(actual,reference,rtol=2e-9,atol=.02)
    x=points.clone().requires_grad_()
    value,record=cell_complex_static_potentials(mesh,x,k,relative_tolerance=1e-10)
    derivative=torch.autograd.grad((EPSILON_0*value[0].real).sum(),x,create_graph=True)[0]
    torch.testing.assert_close(derivative,EPSILON_0*value[1][:,0].real,rtol=2e-8,atol=1e-12)
    hessian=torch.stack([torch.autograd.grad(derivative[0,i],x,retain_graph=True)[0][0] for i in range(3)])
    torch.testing.assert_close(hessian,EPSILON_0*value[2][0,0].real,rtol=2e-8,atol=1e-12)
    strength=a.new_tensor(.2,requires_grad=True)
    def objective(t):
        c=kernel(a+1j*t*torch.eye(3,device=a.device,dtype=a.dtype))
        return (EPSILON_0*cell_complex_static_potentials(mesh,points,c,relative_tolerance=1e-10)[0][0]).real.sum()
    derivative=torch.autograd.grad(objective(strength),strength)[0];step=1e-5
    finite=(objective(strength.detach()+step)-objective(strength.detach()-step))/(2*step)
    torch.testing.assert_close(derivative,finite,rtol=2e-7,atol=1e-12)


def test_cell_selection_repetition_rotation_scale_and_empty():
    coarse,a=fixture();mesh=coarse.refine();tensor=(1+.2j)*a;k=kernel(tensor)
    points=a.new_tensor([[1.4,.5,.7]]);indices=mesh.cells.new_tensor([3,0,3])
    selected,receipt=cell_complex_static_potentials(mesh,points,k,cell_indices=indices,relative_tolerance=1e-10,pair_batch=3)
    expected=cell_tensor_static_potentials(mesh,points,StaticTensorGreen(EPSILON_0*a),cell_indices=indices)
    for actual,reference in zip(selected,expected):
        torch.testing.assert_close(actual,reference.to(actual.dtype)/(1+.2j),rtol=3e-9,atol=.02)
        torch.testing.assert_close(actual[:,0],actual[:,2],rtol=0,atol=0)
    q,_=torch.linalg.qr(a.new_tensor([[1.,2.,3.],[0.,1.,2.],[2.,1.,1.]]));qc=q.to(tensor.dtype)
    # An improper orthogonal transform would reverse tetrahedron orientation.
    q=q*torch.linalg.det(q);qc=q.to(tensor.dtype)
    rotated=qc@tensor@qc.T;rotated=(rotated+rotated.T)/2
    scale=2.7
    transformed=TetrahedralVolumeMesh(scale*mesh.vertices@q.T,mesh.cells,region_id=1)
    other,_=cell_complex_static_potentials(transformed,scale*points@q.T,kernel(rotated),
        cell_indices=indices,relative_tolerance=1e-10,pair_batch=5)
    torch.testing.assert_close(other[0],scale**2*selected[0],rtol=3e-9,atol=.02)
    torch.testing.assert_close(other[1],scale*selected[1]@qc.T,rtol=3e-9,atol=.02)
    torch.testing.assert_close(other[2],qc@selected[2]@qc.T,rtol=3e-9,atol=.02)
    for p,ids in ((points,indices[:0]),(points[:0],indices)):
        empty,record=cell_complex_static_potentials(mesh,p,k,cell_indices=ids)
        assert record['converged'] and record['final_order']==0
        assert [tuple(v.shape) for v in empty]==[(len(p),len(ids)),(len(p),len(ids),3),(len(p),len(ids),3,3)]
        assert all(v.dtype==tensor.dtype for v in empty)


def test_cell_admission_budget_and_nonfinite_sums(monkeypatch):
    mesh,a=fixture();k=kernel((1+.2j)*a);x=a.new_tensor([[.2,.2,.2]])
    with pytest.raises(ValueError,match='explicit principal'):
        cell_complex_static_potentials(mesh,a.new_tensor([[.2,.3,0.]]),k)
    with pytest.raises(ValueError,match='edges'):
        cell_complex_static_potentials(mesh,a.new_tensor([[.2,0.,0.]]),k,principal_value=True)
    with pytest.raises(ValueError,match='indices'):
        cell_complex_static_potentials(mesh,x,k,cell_indices=mesh.cells.new_tensor([2]))
    with pytest.raises(ComplexStaticCellConvergenceError) as error:
        cell_complex_static_potentials(mesh,x,k,initial_order=2,max_order=4,relative_tolerance=1e-15)
    assert error.value.receipt['failure_reason']=='cell_order_budget_exhausted'
    assert error.value.receipt['checks'][-1]['hessian']['failed_query_cells']>0
    original=k._integrate_panel_rule
    def overflow(*args):
        value,gradient=original(*args)
        return value*1e300,gradient*1e300
    monkeypatch.setattr(k,'_integrate_panel_rule',overflow)
    with pytest.raises(ComplexStaticCellConvergenceError) as error:
        cell_complex_static_potentials(mesh,x,k,initial_order=2,max_order=4)
    assert error.value.receipt['failure_reason']=='nonfinite_cell_acceptance_metrics'
