"""Two-body conductivity derivatives of normal and transposed current solves."""
import runpy
import pytest
import torch
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
from witwin.maxwell.mom.dual_static_current import DualStaticCurrentBalance
from witwin.maxwell.mom.dual_current_action import DualStaticCurrentAction,StreamedDualCurrentBalance
from witwin.maxwell.mom.dual_current_material import DualMaterialInverse
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.static_current import NodalCurrentBalance

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_dual_material_inverse_and_response_second_derivatives(monkeypatch):
    first=runpy.run_path('tests/mom/test_static_current_projector.py')['block']().volume
    second=TetrahedralVolumeMesh(1.2*first.vertices+first.vertices.new_tensor([.25,.03,-.02]),first.cells,region_id=2)
    volumes=(second,first)
    dense=DualStaticCurrentBalance(volumes,[2.,3.],[5.,7.])
    parameters=torch.tensor([2.,3.,5.,7.],device='cuda',dtype=torch.float64,requires_grad=True)
    electric=parameters[:2].repeat_interleave(12)
    magnetic=parameters[2:].repeat_interleave(12)
    rhs=torch.arange(96,device='cuda',dtype=torch.float64).cos().reshape(2,48).requires_grad_()
    eye=torch.eye(24,device='cuda',dtype=torch.float64)
    pc=dense.projector@dense.curl
    matrix=torch.cat((torch.cat((eye,electric[:,None]*pc),1),torch.cat((-magnetic[:,None]*pc,eye),1)))
    normal=torch.linalg.solve(matrix,rhs.unsqueeze(-1)).squeeze(-1)
    transposed=torch.linalg.solve(matrix.T,rhs.unsqueeze(-1)).squeeze(-1)
    source=rhs[:,:24]
    injected=torch.cat((source@dense.projector.T,torch.zeros_like(source)),dim=-1)
    response=(torch.linalg.solve(matrix,injected.unsqueeze(-1)).squeeze(-1)[:,:24]-source)/electric
    expected_loss=normal.square().sum()+.3*transposed.square().sum()+response.square().sum()
    expected_grad=torch.autograd.grad(expected_loss,(rhs,parameters),create_graph=True)
    expected_second=torch.autograd.grad(expected_grad[0].sum()+expected_grad[1].sum(),(rhs,parameters))
    def forbidden(*args,**kwargs):raise AssertionError('Dense current assembly or solve forbidden')
    monkeypatch.setattr(torch.linalg,'solve',forbidden)
    monkeypatch.setattr(torch,'block_diag',forbidden)
    monkeypatch.setattr(NodalCurrentBalance,'_matrix',forbidden)
    rule=StoppingRule(preconditioner='none',rtol=1e-13,atol=1e-24,maxiter=100,restart=24)
    action=DualStaticCurrentAction(volumes,[2.,3.],[5.,7.],projection_rule=rule,column_batch=4)
    balance=StreamedDualCurrentBalance(action,rule,column_batch=8)
    material=DualMaterialInverse(balance)
    # Rebuild scalar expansion after the reference graph has been consumed.
    electric=parameters[:2].repeat_interleave(12);magnetic=parameters[2:].repeat_interleave(12)
    actual=material.apply(rhs,electric,magnetic)
    adjoint=material.adjoint(rhs,electric,magnetic)
    field=material.electric_response(source,electric,magnetic)
    for a,b in ((actual,normal),(adjoint,transposed),(field,response)):
        torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-11)
    actual_grad=torch.autograd.grad(actual.square().sum()+.3*adjoint.square().sum()+field.square().sum(),
                                    (rhs,parameters),create_graph=True)
    for a,b in zip(actual_grad,expected_grad):torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-10)
    actual_second=torch.autograd.grad(actual_grad[0].sum()+actual_grad[1].sum(),(rhs,parameters))
    for a,b in zip(actual_second,expected_second):torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-10)
    assert material.apply(rhs[:0],electric,magnetic).shape==rhs[:0].shape
    with pytest.raises(ValueError,match='differ from admitted'):
        material.apply(rhs,electric+1e-5,magnetic)
    delayed=material.apply(rhs,electric,magnetic)
    with torch.no_grad():electric[0]+=1
    with pytest.raises(RuntimeError,match='modified by an inplace operation'):
        torch.autograd.grad(delayed.sum(),rhs)
