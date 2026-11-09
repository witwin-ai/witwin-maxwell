"""Independent far multipoles for the real SPD static volume path."""
import math
import pytest
import torch
from witwin.maxwell.mom.static_tensor_green import StaticTensorGreen
from witwin.maxwell.mom.static_tensor_volume import cell_tensor_static_potentials
from test_complex_static_cell_far import fixture,reference,relative_error

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('distance',[1e6,1e8,1e100])
def test_real_far_cell_retains_representable_multipoles(distance):
    mesh,complex_kernel=fixture()
    kernel=StaticTensorGreen(complex_kernel.coefficient.real)
    points=mesh.vertices.new_tensor([[0.,0.,distance]])
    values=cell_tensor_static_potentials(mesh,points,kernel)
    expected=tuple(value*mesh.volumes[0] for value in reference(points,kernel))
    errors=[relative_error(actual[:,0],target) for actual,target in zip(values,expected)]
    print({'distance':distance,'real_far_cell_errors':errors},flush=True)
    assert all(bool(torch.isfinite(value).all()) for value in values)
    assert all(math.isfinite(value) for value in errors)
    assert max(errors)<1e-8


def test_real_hybrid_selection_switch_and_native_derivatives():
    from witwin.maxwell.mom.static_tensor_volume import cell_real_static_potentials,RealStaticCellConvergenceError
    from witwin.maxwell.mom.vie_operator import cell_green_potentials
    mesh,_=fixture()
    eye=torch.eye(3,device=mesh.vertices.device,dtype=mesh.vertices.dtype)
    diameter=(mesh.vertices[:,None]-mesh.vertices[None]).norm(dim=-1).max()
    points=mesh.vertices.new_tensor([[0.,0.,0.],[0.,0.,1.],[0.,0.,1.]])
    points[1:,2]=2*diameter*points.new_tensor([1-1e-6,1+1e-6])
    kernel=StaticTensorGreen(2*eye)
    actual,receipt=cell_real_static_potentials(mesh,points,kernel,cell_indices=mesh.cells.new_zeros(19),pair_batch=8)
    assert receipt['representation']=='real_spd_hybrid_cell_v1' and receipt['unique_integrated_cells']==1
    expected=cell_green_potentials(mesh,points,0.)
    for value,target in zip(actual,expected):
        torch.testing.assert_close(value,value[:,:1].expand_as(value),rtol=0,atol=0)
        torch.testing.assert_close(value[:,:1],target.real/2,rtol=1e-8,atol=1e-12)
    angle=points.new_tensor(.37)
    rotation=eye.clone();rotation[:2,:2]=torch.stack((torch.stack((angle.cos(),-angle.sin())),torch.stack((angle.sin(),angle.cos()))))
    coefficient=rotation@torch.diag(points.new_tensor([.02,1.,100.]))@rotation.T
    coefficient=(coefficient+coefficient.T)/2
    query=points.new_tensor([[40.,30.,50.]],requires_grad=True)
    strength=points.new_tensor(1.,requires_grad=True)
    field,_=cell_real_static_potentials(mesh,query,StaticTensorGreen(strength*coefficient))
    first=torch.autograd.grad(field[0].sum(),query,create_graph=True)[0]
    torch.testing.assert_close(first,field[1][:,0],rtol=2e-10,atol=1e-14)
    second=torch.stack([torch.autograd.grad(first[0,i],query,retain_graph=True)[0][0] for i in range(3)])
    torch.testing.assert_close(second,field[2][0,0],rtol=2e-10,atol=1e-14)
    derivative=torch.autograd.grad(field[0].sum(),strength)[0]
    torch.testing.assert_close(derivative,-field[0].sum(),rtol=2e-10,atol=1e-14)
    with pytest.raises(RealStaticCellConvergenceError) as failed:
        cell_real_static_potentials(mesh,query.detach(),StaticTensorGreen(coefficient),initial_order=2,max_order=3,relative_tolerance=1e-14)
    assert failed.value.receipt['failure_reason']=='cell_order_budget_exhausted'


def test_real_point_hessian_combines_physical_scales_before_products():
    eye=torch.eye(3,device='cuda',dtype=torch.float64)
    kernel=StaticTensorGreen(1e-200*eye)
    points=eye.new_tensor([[0.,0.,1e150]])
    values=kernel.evaluate_with_hessian(points)
    expected=reference(points,kernel)
    errors=[relative_error(value,target) for value,target in zip(values,expected)]
    assert all(math.isfinite(value) for value in errors)
    assert max(errors)<1e-13
