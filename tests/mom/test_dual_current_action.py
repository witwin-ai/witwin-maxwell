"""Coupled matrix-free DC currents against the original dense formulation."""
import runpy
import pytest
import torch
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
from witwin.maxwell.mom.dual_static_current import DualStaticCurrentBalance
from witwin.maxwell.mom.dual_current_action import DualStaticCurrentAction,FixedDualCurrentInverse,DualCurrentInverseConvergenceError
from witwin.maxwell.mom.static_current import NodalCurrentBalance
from witwin.maxwell.mom.linalg import StoppingRule

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_dual_global_actions_inverse_and_vector_gradients(monkeypatch):
    first=runpy.run_path('tests/mom/test_static_current_projector.py')['block']().volume
    second=TetrahedralVolumeMesh(1.2*first.vertices+first.vertices.new_tensor([.25,.03,-.02]),first.cells,region_id=2)
    volumes=(second,first)
    dense=DualStaticCurrentBalance(volumes,[2.,3.],[5.,7.])
    matrix=dense.matrix
    rhs=torch.arange(96,device='cuda',dtype=torch.float64).reshape(2,48).cos().requires_grad_()
    expected=rhs@matrix.T
    rhs_projection=torch.block_diag(dense.projector,dense.projector)
    solution=torch.linalg.solve(matrix,rhs.unsqueeze(-1)).squeeze(-1)
    solution_grad=torch.linalg.solve(matrix.T,(2*solution).unsqueeze(-1)).squeeze(-1)
    expected_grad=2*expected@matrix
    expected_second=2*torch.ones_like(rhs)@matrix.T@matrix
    def forbidden(*args,**kwargs):raise AssertionError('Global dense DC arrays or solves forbidden')
    monkeypatch.setattr(torch.linalg,'solve',forbidden)
    monkeypatch.setattr(torch,'block_diag',forbidden)
    monkeypatch.setattr(NodalCurrentBalance,'_matrix',forbidden)
    projection_rule=StoppingRule(preconditioner='none',rtol=1e-13,atol=1e-22,maxiter=100,restart=20)
    action=DualStaticCurrentAction(volumes,[2.,3.],[5.,7.],projection_rule=projection_rule,
        column_batch=3,target_batch=4,source_batch=3)
    actual=action.apply(rhs)
    torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-11)
    torch.testing.assert_close(action.adjoint(rhs),rhs@matrix,rtol=1e-10,atol=1e-11)
    torch.testing.assert_close(action.source_rhs(rhs),rhs@rhs_projection.T,rtol=1e-10,atol=1e-11)
    torch.testing.assert_close(action.source_rhs_adjoint(rhs),rhs@rhs_projection,rtol=1e-10,atol=1e-11)
    g,=torch.autograd.grad(actual.square().sum(),rhs,create_graph=True)
    torch.testing.assert_close(g,expected_grad,rtol=1e-9,atol=1e-10)
    gg,=torch.autograd.grad(g.sum(),rhs)
    torch.testing.assert_close(gg,expected_second,rtol=1e-9,atol=1e-10)
    inverse=FixedDualCurrentInverse(action,StoppingRule(preconditioner='none',rtol=1e-11,atol=1e-13,maxiter=80,restart=20))
    solved=inverse.apply(rhs)
    torch.testing.assert_close(solved,solution,rtol=1e-9,atol=1e-10)
    assert float((solved.detach()@matrix.T-rhs.detach()).norm())<1e-9
    grad,=torch.autograd.grad(solved.square().sum(),rhs)
    torch.testing.assert_close(grad,solution_grad,rtol=1e-9,atol=1e-10)
    assert action.apply(rhs[:0]).shape==rhs[:0].shape
    failed=FixedDualCurrentInverse(action,StoppingRule(preconditioner='none',rtol=1e-15,atol=1e-25,maxiter=1,restart=1),refinement_steps=0)
    with pytest.raises(DualCurrentInverseConvergenceError):failed.apply(rhs[0])
    assert not action.receipt['coupled_nonsingularity_certified']
    delayed=action.apply(rhs)
    action.magnetic[0]+=1
    with pytest.raises(ValueError,match='binding changed'):torch.autograd.grad(delayed.sum(),rhs)


def test_dual_action_rejects_invalid_conductivity():
    volume=runpy.run_path('tests/mom/test_static_current_projector.py')['block']().volume
    rule=StoppingRule(preconditioner='none')
    with pytest.raises(ValueError,match='positive homogeneous'):
        DualStaticCurrentAction([volume],[1.],[0.],projection_rule=rule)
    with pytest.raises(ValueError,match='positive homogeneous'):
        DualStaticCurrentAction([volume],[torch.arange(1,5,device='cuda',dtype=torch.float64)],[1.],projection_rule=rule)
    with pytest.raises(ValueError,match='one conductivity'):
        DualStaticCurrentAction([volume],[],[1.],projection_rule=rule)


def test_streamed_dual_fields_potentials_and_source_gradients(monkeypatch):
    from witwin.maxwell.mom.dual_current_action import StreamedDualCurrentBalance
    first=runpy.run_path('tests/mom/test_static_current_projector.py')['block']().volume
    second=TetrahedralVolumeMesh(1.2*first.vertices+first.vertices.new_tensor([.25,.03,-.02]),first.cells,region_id=2)
    volumes=(second,first)
    dense=DualStaticCurrentBalance(volumes,[2.,3.],[5.,7.])
    sources=torch.arange(48,device='cuda',dtype=torch.float64).cos().reshape(2,8,3).requires_grad_()
    expected=dense.solve(*sources)
    keys=('electric','magnetic','electric_current','magnetic_current')
    def loss(result):
        return sum(result[k].square().sum() for k in keys)+sum(
            p.square().sum() for k in ('electric_potentials','magnetic_potentials') for p in result[k])
    expected_grad,=torch.autograd.grad(loss(expected),sources)
    def forbidden(*args,**kwargs):raise AssertionError('Global dense DC assembly or solve forbidden')
    monkeypatch.setattr(torch.linalg,'solve',forbidden)
    monkeypatch.setattr(torch,'block_diag',forbidden)
    monkeypatch.setattr(NodalCurrentBalance,'_matrix',forbidden)
    rule=StoppingRule(preconditioner='none',rtol=1e-13,atol=1e-22,maxiter=100,restart=20)
    action=DualStaticCurrentAction(volumes,[2.,3.],[5.,7.],projection_rule=rule,
        column_batch=4,target_batch=8,source_batch=4)
    balance=StreamedDualCurrentBalance(action,rule,column_batch=8)
    actual=balance.solve(*sources)
    for key in keys:torch.testing.assert_close(actual[key],expected[key],rtol=1e-9,atol=1e-11)
    for key in ('electric_potentials','magnetic_potentials'):
        for block,p,q in zip(action.blocks,actual[key],expected[key]):
            torch.testing.assert_close(p,q,rtol=1e-9,atol=1e-11)
            assert float((p.detach()*block.nodal_volume).sum().abs())<1e-17
    grad,=torch.autograd.grad(loss(actual),sources)
    torch.testing.assert_close(grad,expected_grad,rtol=1e-9,atol=1e-11)
    packed=torch.cat((actual['electric_current'].flatten(),actual['magnetic_current'].flatten()))
    original_rhs=torch.cat([dense.projector@s.flatten() for s in sources])
    assert float((dense.matrix@packed-original_rhs).detach().norm())<=actual['receipt']['threshold']
    assert actual['receipt']['coupled_inverse_bound']['admitted']
    assert not actual['receipt']['physical_qualification']
    assert max(actual['receipt'][k] for k in ('electric_reconstruction_error','magnetic_reconstruction_error'))<1e-10
    zero=balance.solve(torch.zeros_like(sources[0]),torch.zeros_like(sources[1]))
    assert zero['receipt']['coupled_current_residual']==0
    # A loose solver rule cannot bypass the original physical-current gate.
    monkeypatch.setattr(balance.inverse,'apply',lambda rhs:torch.zeros_like(rhs))
    with pytest.raises(RuntimeError,match='original coupled current residual'):
        balance.solve(*sources)
    with pytest.raises(ValueError,match='matching finite'):
        balance.solve(sources[0,:1],sources[1])
    action.electric[0]+=1
    with pytest.raises(ValueError,match='binding changed'):balance.solve(*sources)
