"""Regional static transmission controls, separate from HB qualification."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import SurfaceSystem,RWGBasis,sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.static_tensor_interface import StaticTensorInterfaceOperator

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def system(level=0,centers=((0.,0.,0.),)):
    return SurfaceSystem.create([(i,f'body{i}',RWGBasis(sphere_mesh(radius=.1,subdivisions=level,center=center)),
        compile_medium(mw.Material(),3e8)) for i,center in enumerate(centers)])


def tensor():
    return torch.tensor([[2.,.3,.1],[.3,3.,.2],[.1,.2,1.5]],device='cuda',dtype=torch.float64)


def test_multibody_constant_potential_and_region_ownership():
    geometry=system(centers=((-0.3,0.,0.),(.3,0.,0.)))
    a=tensor();eye=torch.eye(3,device=a.device,dtype=a.dtype)
    op=StaticTensorInterfaceOperator(geometry,eye,(a,1.3*a),pair_batch=37)
    response=op.solve_uniform(torch.zeros(3,device=a.device,dtype=a.dtype),offset=2.)
    torch.testing.assert_close(response.potential,torch.full_like(response.potential,2.),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(response.conormal,torch.zeros_like(response.conormal),rtol=0,atol=1e-12)
    assert response.receipt['relative_algebraic_residual']<1e-13
    assert not response.receipt['physical_qualification']
    size=op.face_counts[0];total=len(op.centers)
    assert bool(op.matrix[:size,total+size:].abs().max()>0)
    assert torch.count_nonzero(op.matrix[total:total+size,size:total])==0
    for region,point in ((0,[1.,.2,.1]),(1,[-.3,0.,0.]),(2,[.3,0.,0.])):
        value,field=response.fields(a.new_tensor([point]),region)
        torch.testing.assert_close(value,a.new_tensor([2.]),rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(field,torch.zeros_like(field),rtol=0,atol=1e-11)
    with pytest.raises(ValueError,match='declared region'):response.fields(a.new_tensor([[0.,0.,0.]]),1)
    with pytest.raises(ValueError,match='budget'):StaticTensorInterfaceOperator(geometry,eye,(a,a),max_matrix_bytes=1)
    op.matrix.add_(1)
    with pytest.raises(ValueError,match='changed'):response.fields(a.new_tensor([[1.,0.,0.]]),0)


def test_anisotropic_sphere_refines_to_independent_uniform_field_solution():
    a=tensor();eye=torch.eye(3,device=a.device,dtype=a.dtype)
    drive=a.new_tensor([1.,-.3,.5])
    expected=torch.linalg.solve(a+2*eye,3*drive)
    errors=[]
    for level in (0,1,2):
        op=StaticTensorInterfaceOperator(system(level),eye,(a,),pair_batch=4096)
        response=op.solve_uniform(drive)
        _,actual=response.fields(a.new_tensor([[.013,.007,.019]]),1)
        error=float(((actual[0]-expected).norm()/expected.norm()).detach());errors.append(error)
        assert response.receipt['relative_algebraic_residual']<1e-12
    print({'anisotropic_sphere_internal_relative_errors':errors,'physical_qualification':False})
    assert errors[2]<errors[1]<errors[0]
    assert errors[2]<.05


def test_uniform_drive_native_sensitivity_and_equal_region_limit():
    a=tensor().requires_grad_();drive=a.new_tensor([1.,-.3,.5],requires_grad=True)
    op=StaticTensorInterfaceOperator(system(1),a,(a,),pair_batch=257)
    response=op.solve_uniform(drive)
    point=a.new_tensor([[.003,.002,.001]],requires_grad=True)
    potential,field=response.fields(point,1)
    gradient,=torch.autograd.grad(potential.sum(),point,retain_graph=True)
    torch.testing.assert_close(field,-gradient,rtol=0,atol=0)
    assert float(((field[0]-drive).norm()/drive.norm()).detach())<.05
    field_gradient,material_gradient=torch.autograd.grad(field.sum(),(drive,a))
    assert bool(torch.isfinite(field_gradient).all() & torch.isfinite(material_gradient).all())
    assert bool(field_gradient.abs().max()>0)


def test_extreme_drive_admission_and_empty_queries():
    a=tensor();eye=torch.eye(3,device=a.device,dtype=a.dtype)
    geometry=system()
    op=StaticTensorInterfaceOperator(geometry,eye,(a,))
    response=op.solve_uniform(a.new_tensor([1e220,0.,0.]))
    assert torch.isfinite(a.new_tensor(response.receipt['relative_algebraic_residual']))
    for region in (0,1):
        value,field=response.fields(a.new_empty(0,3,requires_grad=True),region)
        assert value.shape==(0,) and field.shape==(0,3)
        assert value.requires_grad and field.requires_grad
    with pytest.raises(ValueError,match='not representable'):
        response.fields(a.new_tensor([[1e100,0.,0.]]),0)
    huge=StaticTensorInterfaceOperator(geometry,1e100*eye,(1e100*a,))
    with pytest.raises(ValueError,match='physical potential/conormal'):
        huge.solve_uniform(a.new_tensor([1e220,0.,0.]))
