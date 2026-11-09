"""Continuous potential candidate controls; no production qualification."""
import pytest
import torch
from test_static_tensor_interface import system,tensor
from witwin.maxwell.mom.static_tensor_interface import StaticTensorInterfaceOperator

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_p1_multibody_constant_and_affine_homogeneous_fields():
    geometry=system(centers=((-0.3,0.,0.),(.3,0.,0.)))
    a=tensor();zero=a.new_zeros(3);drive=a.new_tensor([.3,-.2,.7])
    op=StaticTensorInterfaceOperator(geometry,a,(a,a),potential_basis='p1',pair_batch=257)
    assert op.matrix.shape==(80,64)
    for field in (zero,drive):
        response=op.solve_uniform(field,offset=2.)
        vertices=torch.cat([b.mesh.vertices for b in geometry.bodies])
        torch.testing.assert_close(response.potential,2-vertices@field,rtol=1e-11,atol=1e-11)
        torch.testing.assert_close(response.conormal,-op.normals@(a@field),rtol=1e-10,atol=1e-11)
        assert response.receipt['relative_algebraic_residual']<1e-12
        for region,point in ((0,[1.,.2,.1]),(1,[-.3,.01,.02]),(2,[.3,-.01,.02])):
            points=a.new_tensor([point]);value,actual=response.fields(points,region)
            torch.testing.assert_close(value,2-points@field,rtol=1e-11,atol=1e-11)
            torch.testing.assert_close(actual,field[None],rtol=1e-10,atol=1e-11)
        restored=op.response_from_traces(field,response.potential,response.conormal,offset=2.)
        torch.testing.assert_close(restored.conormal,response.conormal,rtol=0,atol=0)
    with pytest.raises(ValueError,match='budget'):
        StaticTensorInterfaceOperator(geometry,a,(a,a),potential_basis='p1',max_matrix_bytes=1)
    from witwin.maxwell.mom.static_tensor_snapshot import snapshot_static_tensor_response
    with pytest.raises(ValueError,match='P0 potential'):snapshot_static_tensor_response(response)
    op.potential_indices.add_(1)
    with pytest.raises(ValueError,match='changed'):response.assert_current()


def test_p1_native_material_drive_and_observer_gradients():
    parameter=tensor().requires_grad_();a=(parameter+parameter.T)/2
    drive=a.new_tensor([.3,-.2,.7],requires_grad=True)
    eye=torch.eye(3,device=a.device,dtype=a.dtype)
    def objective(coeff,field):
        op=StaticTensorInterfaceOperator(system(),eye,(coeff,),potential_basis='p1')
        response=op.solve_uniform(field)
        point=a.new_tensor([[.013,.007,.019]],requires_grad=True)
        value,actual=response.fields(point,1)
        derivative,=torch.autograd.grad(value.sum(),point,retain_graph=True)
        torch.testing.assert_close(actual,-derivative,rtol=0,atol=0)
        return actual.square().sum()
    value=objective(a,drive)
    ga,gd=torch.autograd.grad(value,(parameter,drive))
    direction=torch.ones_like(a)/10;step=1e-5
    fd=(objective(a.detach()+step*direction,drive.detach())-objective(a.detach()-step*direction,drive.detach()))/(2*step)
    torch.testing.assert_close((ga*direction).sum(),fd,rtol=1e-6,atol=1e-9)
    torch.testing.assert_close((gd*drive).sum(),2*value,rtol=1e-10,atol=1e-12)


def test_p1_induced_polarization_rhs_and_trace_recovery():
    from test_static_tensor_polarization import source
    polarization,coefficient,values=source(refine=1)
    binding=polarization.binding;geometry=binding.surface_system
    op=StaticTensorInterfaceOperator(geometry,coefficient,(coefficient,),potential_basis='p1',pair_batch=257)
    from witwin.maxwell.mom.static_tensor_preparation import prepare_static_tensor_cell_response
    with pytest.raises(ValueError,match='P0 potential'):
        prepare_static_tensor_cell_response(op,(binding,))
    values=(values*torch.linspace(.7,1.4,len(values),device=values.device,dtype=values.dtype)[:,None]).detach().requires_grad_()
    zero=coefficient.new_zeros(3)
    response=op.solve_uniform(zero,bindings=(binding,),polarizations=(values,))
    expected=(values[polarization.boundary_cells]*op.normals).sum(-1)
    torch.testing.assert_close(response.normal_p,expected,rtol=0,atol=0)
    # Nonuniform cell polarization contributes both internal charge and the
    # boundary conormal increment, independently of potential basis.
    rhs,_,_=op._source_rhs(zero,0.,(binding,),(values,))
    single,_=op._layers(op.interiors[0],op.centers,op.triangles,op.normals,diagonal=True)
    from witwin.maxwell.mom.static_tensor_polarization import StaticTensorPolarization
    particular=StaticTensorPolarization(binding,coefficient,values,pair_batch=257)
    charge=particular.fields(op.centers,interior_sources_only=True)[0]
    assert bool(charge.abs().max()>0)
    torch.testing.assert_close(rhs[len(op.centers):],charge+single@expected,rtol=1e-11,atol=1e-13)
    restored=op.response_from_traces(zero,response.potential,response.conormal,
                                    bindings=(binding,),polarizations=(values,))
    points=values.new_tensor([[.021,.019,.023]])
    for actual,want in zip(restored.fields(points,1),response.fields(points,1)):
        torch.testing.assert_close(actual,want,rtol=0,atol=0)
    gradient,=torch.autograd.grad(response.fields(points,1)[1].square().sum(),values)
    assert bool(torch.isfinite(gradient).all()) and bool(gradient.abs().max()>0)
