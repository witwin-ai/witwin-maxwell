"""Far-cell cancellation and representable extreme point-Hessian controls."""
import math
import pytest
import torch
from witwin.maxwell.constants import EPSILON_0
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.complex_static_tensor_green import ComplexStaticTensorGreen
from witwin.maxwell.mom.static_tensor_volume import cell_complex_static_potentials
from witwin.maxwell.mom.dyadic_green import electrostatic_kernel

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def fixture():
    vertices=torch.tensor([[1.,1.,1.],[1.,-1.,-1.],[-1.,1.,-1.],[-1.,-1.,1.]],device='cuda',dtype=torch.float64)
    mesh=TetrahedralVolumeMesh(vertices,torch.tensor([[0,2,1,3]],device='cuda'),region_id=1)
    eye=torch.eye(3,device='cuda',dtype=torch.complex128)
    return mesh,ComplexStaticTensorGreen(TensorMedium(3e7,(2+.1j)*eye,eye))


def reference(points,kernel):
    radius=points.norm(dim=-1);unit=(points/radius[:,None]).to(kernel.coefficient.dtype)
    value=1/(4*math.pi*kernel.coefficient[0,0]*radius)
    gradient=-(value/radius)[:,None]*unit
    eye=torch.eye(3,device=points.device,dtype=unit.dtype)
    hessian=((value/radius)/radius)[:,None,None]*(3*unit[:,:,None]*unit[:,None,:]-eye)
    return value,gradient,hessian


def relative_error(actual,expected):
    scale=expected.abs().amax()
    return float(((actual-expected)/scale).norm()/(expected/scale).norm())


@pytest.mark.parametrize('distance',[1e6,1e8,1e100])
def test_far_symmetric_cell_preserves_nonzero_multipoles(distance):
    mesh,kernel=fixture();points=mesh.vertices.new_tensor([[0.,0.,distance]])
    values,receipt=cell_complex_static_potentials(mesh,points,kernel)
    expected=tuple(value*mesh.volumes[0] for value in reference(points,kernel))
    # The centered regular tetrahedron has no dipole and an isotropic second
    # moment. The scalar harmonic kernel's quadrupole contraction vanishes;
    # the monopole is an independent far reference at these separations.
    errors=[relative_error(actual[:,0],target) for actual,target in zip(values,expected)]
    print({'distance':distance,'far_cell_relative_errors':errors,'receipt':receipt},flush=True)
    assert all(bool(torch.isfinite(value).all()) for value in values)
    assert all(math.isfinite(error) for error in errors)
    assert receipt['converged'] and max(errors)<1e-8
    assert receipt['separated_rule']=='positive_tetrahedral_volume_cubature'


def test_extreme_static_point_hessian_does_not_overflow_intermediate_square():
    mesh,kernel=fixture();points=mesh.vertices.new_tensor([[0.,0.,1e100]])
    actual=tuple(value/EPSILON_0 for value in electrostatic_kernel(points,kernel.medium))
    errors=[relative_error(value,expected) for value,expected in zip(actual,reference(points,kernel))]
    print({'far_point_relative_errors':errors},flush=True)
    assert all(bool(torch.isfinite(value).all()) for value in actual)
    assert all(math.isfinite(error) for error in errors) and max(errors)<1e-13


def test_nonrepresentable_lossless_point_distance_rejects_false_zero():
    mesh,kernel=fixture();eye=torch.eye(3,device=mesh.vertices.device,dtype=torch.complex128)
    medium=TensorMedium(3e7,2*eye,eye)
    with pytest.raises(ValueError,match='geometry is not representable'):
        electrostatic_kernel(mesh.vertices.new_tensor([[0.,0.,1e200]]),medium)


def test_mixed_near_far_selection_and_far_native_derivatives():
    mesh,kernel=fixture();points=mesh.vertices.new_tensor([[.2,.1,.1],[20.,4.,5.]])
    ids=mesh.cells.new_tensor([0,0])
    values,receipt=cell_complex_static_potentials(mesh,points,kernel,cell_indices=ids,pair_batch=1)
    for value in values:torch.testing.assert_close(value[:,0],value[:,1],rtol=0,atol=0)
    contraction=torch.einsum('ij,qcij->qc',kernel.coefficient,values[2])
    torch.testing.assert_close(contraction,contraction.new_tensor([[-1.,-1.],[0.,0.]]),rtol=0,atol=2e-9)
    repeated,record=cell_complex_static_potentials(mesh,points[1:],kernel,
        cell_indices=mesh.cells.new_zeros(19),pair_batch=8)
    assert record['selected_cells']==19 and record['unique_integrated_cells']==1
    for value in repeated:torch.testing.assert_close(value,value[:,:1].expand_as(value),rtol=0,atol=0)
    x=points[1:].clone().requires_grad_()
    field,_=cell_complex_static_potentials(mesh,x,kernel,relative_tolerance=1e-10)
    derivative=torch.autograd.grad((EPSILON_0*field[0]).real.sum(),x,create_graph=True)[0]
    torch.testing.assert_close(derivative,EPSILON_0*field[1][:,0].real,rtol=2e-12,atol=1e-14)
    second=torch.stack([torch.autograd.grad(derivative[0,i],x,retain_graph=True)[0][0] for i in range(3)])
    torch.testing.assert_close(second,EPSILON_0*field[2][0,0].real,rtol=2e-12,atol=1e-14)
    t=points.new_tensor(.2,requires_grad=True);eye=torch.eye(3,device=points.device,dtype=torch.complex128)
    def objective(strength):
        k=ComplexStaticTensorGreen(TensorMedium(3e7,(2+strength+.1j)*eye,eye))
        return (EPSILON_0*cell_complex_static_potentials(mesh,x.detach(),k)[0][0]).real.sum()
    actual=torch.autograd.grad(objective(t),t)[0];step=1e-5
    expected=(objective(t.detach()+step)-objective(t.detach()-step))/(2*step)
    torch.testing.assert_close(actual,expected,rtol=2e-8,atol=1e-13)


def test_selected_native_volume_rule_matches_full_and_rejects_invalid_indices():
    mesh,_=fixture();mesh=mesh.refine();ids=mesh.cells.new_tensor([3,0,3])
    full=mesh.quadrature(4);selected=mesh.quadrature(4,cell_indices=ids)
    for actual,reference in zip(selected,full):torch.testing.assert_close(actual,reference[ids],rtol=0,atol=0)
    assert mesh.quadrature(4,cell_indices=ids[:0])[0].shape==(0,64,3)
    with pytest.raises(ValueError,match='indices'):
        mesh.quadrature(4,cell_indices=ids.new_tensor([-1]))
