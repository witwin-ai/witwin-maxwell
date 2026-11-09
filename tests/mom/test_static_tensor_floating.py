"""Floating conductive-limit interfaces, neutral charge and tensor coupling."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import SurfaceSystem,RWGBasis,sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.static_tensor_floating import StaticTensorFloatingOperator

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def system(level=0,centers=((0.,0.,0.),)):
    return SurfaceSystem.create([(i,f'body{i}',RWGBasis(sphere_mesh(radius=.1,subdivisions=level,center=c)),
        compile_medium(mw.Material(),3e8)) for i,c in enumerate(centers)])


def test_floating_neutrality_tensor_coupling_and_trace_gauge():
    eye=torch.eye(3,device='cuda',dtype=torch.float64)
    a=eye.new_tensor([[2.,.3,.1],[.3,3.,.2],[.1,.2,1.5]])
    op=StaticTensorFloatingOperator(system(centers=((-0.3,0.,0.),(.3,0.,0.))),eye,(None,a),pair_batch=37)
    drive=eye.new_tensor([1.,-.3,.5],requires_grad=True)
    response=op.solve_uniform(drive)
    n=op.face_counts[0]
    torch.testing.assert_close(response.potential[:n],response.offsets[0].expand(n),rtol=1e-12,atol=1e-12)
    assert abs(float((response.conormal[:n]*op.areas[:n]).sum().detach()))<1e-12
    assert response.receipt['algebraic_residual_norm']<1e-12
    assert not response.receipt['physical_qualification']
    shifted=op.solve_uniform(drive,boundary_potentials={1:eye.new_full((n,),2.)})
    torch.testing.assert_close(shifted.potential,response.potential,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(shifted.offsets,response.offsets-2,rtol=1e-12,atol=1e-12)
    # A nonconstant induced current trace is retained exactly by the interface.
    trace=op.centers[:n,1].clone().requires_grad_()
    induced=op.solve_uniform(drive,boundary_potentials={1:trace})
    torch.testing.assert_close(induced.potential[:n],trace+induced.offsets[0],rtol=1e-12,atol=1e-12)
    for region,point in ((0,[1.,.2,.1]),(2,[.3,0.,0.])):
        value,field=induced.fields(eye.new_tensor([point]),region)
        assert bool(torch.isfinite(value).all() & torch.isfinite(field).all())
    gradient,=torch.autograd.grad(field.sum(),trace,retain_graph=True)
    assert bool(torch.isfinite(gradient).all()) and bool(gradient.abs().max()>0)
    with pytest.raises(NotImplementedError,match='volume-current'):response.fields(eye.new_tensor([[-.3,0.,0.]]),1)
    with pytest.raises(ValueError,match='conductive-limit'):op.solve_uniform(drive,boundary_potentials={2:trace})
    with pytest.raises(ValueError,match='budget'):StaticTensorFloatingOperator(op.system,eye,(None,a),max_matrix_bytes=1)
    op.matrix.add_(1)
    with pytest.raises(ValueError,match='changed'):response.fields(eye.new_tensor([[1.,0.,0.]]),0)


def test_floating_sphere_external_analytic_refinement():
    eye=torch.eye(3,device='cuda',dtype=torch.float64);drive=eye.new_tensor([1.,-.3,.5])
    point=eye.new_tensor([[.27,.13,.09]]);radius=point.norm(dim=-1)
    dot=(point*drive).sum(-1)
    exact=drive+.1**3*(3*point*dot[:,None]/radius[:,None]**5-drive/radius[:,None]**3)
    errors=[]
    for level in (0,1,2):
        op=StaticTensorFloatingOperator(system(level),eye,(None,))
        response=op.solve_uniform(drive)
        _,field=response.fields(point,0)
        errors.append(float(((field-exact).norm()/exact.norm()).detach()))
        assert abs(float((response.conormal*op.areas).sum().detach()))<1e-12
    print({'floating_sphere_external_relative_errors':errors,'threshold':.005,
           'selected_finest_physical_pass':errors[-1]<=.005,'physical_qualification':False})
    assert errors[2]<errors[1]<errors[0]


def test_floating_extreme_finite_admission():
    eye=torch.eye(3,device='cuda',dtype=torch.float64)
    op=StaticTensorFloatingOperator(system(),eye,(None,))
    response=op.solve_uniform(eye.new_tensor([1e220,0.,0.]))
    assert torch.isfinite(eye.new_tensor(response.receipt['relative_algebraic_residual']))
    with pytest.raises(ValueError,match='not representable'):
        response.fields(eye.new_tensor([[1e100,0.,0.]]),0)
    huge=StaticTensorFloatingOperator(system(),1e100*eye,(None,))
    with pytest.raises(ValueError,match='physical conormal'):
        huge.solve_uniform(eye.new_tensor([1e220,0.,0.]))
    large=SurfaceSystem.create([(0,'large',RWGBasis(sphere_mesh(radius=100.,subdivisions=0)),
                                compile_medium(mw.Material(),3e8))])
    distant=StaticTensorFloatingOperator(large,eye,(None,))
    with pytest.raises(ValueError,match='RHS'):
        distant.solve_uniform(eye.new_tensor([1e308,0.,0.]))
