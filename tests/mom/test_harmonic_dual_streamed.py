"""Streamed dual cubic closure against the existing dense root and derivatives."""
import copy
import runpy
import pytest
import torch
from witwin.maxwell.mom.harmonic_dual_current import DualHarmonicCurrentClosure,DualCurrentConvergenceError
from witwin.maxwell.mom.dual_current_response import DualNonlinearTangent
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.static_current import NodalCurrentBalance
from test_harmonic_dual_current import ac_fields

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_streamed_dual_root_tangent_and_nonlinear_gradients(monkeypatch):
    volume=runpy.run_path('tests/mom/test_static_current_projector.py')['block']().volume
    ac=ac_fields(4)
    amplitude=ac.real.new_tensor(1.,requires_grad=True)
    tpa=ac.real.new_tensor(.3,requires_grad=True)
    conductivity=ac.real.new_tensor([2.,5.],requires_grad=True)
    dense=DualHarmonicCurrentClosure((volume,),[conductivity[0]],[conductivity[1]],absolute_tolerance=1e-13)
    expected=dense.solve(amplitude*ac,tpa)
    def loss(r):
        return r['electric'].square().sum()+r['magnetic'].square().sum()+r['tpa_current'].square().sum()+sum(
            p.square().sum() for key in ('electric_potentials','magnetic_potentials') for p in r[key])
    expected_grad=torch.autograd.grad(loss(expected),(amplitude,tpa,conductivity))
    local=torch.arange(36,device='cuda',dtype=torch.float64).sin().reshape(4,3,3)*.1
    matrix=torch.eye(12,device='cuda',dtype=torch.float64)-dense.response@torch.block_diag(*local)
    def forbidden(*args,**kwargs):raise AssertionError('Dense global DC solve or assembly forbidden')
    for name in ('solve','svdvals'):monkeypatch.setattr(torch.linalg,name,forbidden)
    monkeypatch.setattr(torch,'block_diag',forbidden)
    monkeypatch.setattr(NodalCurrentBalance,'_matrix',forbidden)
    rule=StoppingRule(preconditioner='none',rtol=1e-13,atol=1e-24,maxiter=100,restart=24)
    closure=DualHarmonicCurrentClosure((volume,),[conductivity[0]],[conductivity[1]],absolute_tolerance=1e-13,
                                      streamed_rule=rule,streamed_column_batch=6)
    response=closure.response
    value=torch.arange(24,device='cuda',dtype=torch.float64).cos().reshape(2,12).requires_grad_()
    torch.testing.assert_close(response.apply(value),value@dense.response.T,rtol=1e-9,atol=1e-11)
    torch.testing.assert_close(response.adjoint(value),value@dense.response,rtol=1e-9,atol=1e-11)
    tangent=DualNonlinearTangent(response,local)
    action=tangent.apply(value)
    torch.testing.assert_close(action,value@matrix.T,rtol=1e-9,atol=1e-11)
    torch.testing.assert_close(tangent.adjoint(value),value@matrix,rtol=1e-9,atol=1e-11)
    gradient,=torch.autograd.grad(action.square().sum(),value,create_graph=True)
    torch.testing.assert_close(gradient,2*value@matrix.T@matrix,rtol=1e-9,atol=1e-11)
    second,=torch.autograd.grad(gradient.sum(),value)
    torch.testing.assert_close(second,2*torch.ones_like(value)@matrix.T@matrix,rtol=1e-9,atol=1e-11)
    assert tangent.apply(value[:0]).shape==value[:0].shape
    actual=closure.solve(amplitude*ac,tpa)
    for key in ('electric','magnetic','electric_current','magnetic_current','tpa_current'):
        torch.testing.assert_close(actual[key],expected[key],rtol=1e-8,atol=1e-11)
    actual_grad=torch.autograd.grad(loss(actual),(amplitude,tpa,conductivity),retain_graph=True)
    for a,b in zip(actual_grad,expected_grad):torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-11)
    record=actual['nonlinear_receipt']
    assert record['converged'] and record['iterations']>0
    assert [c['phase'] for c in record['tangent_checks']]==['newton']*record['iterations']+['implicit']*2
    assert all(c['bound']['admitted'] and c['solve']['converged'] for c in record['tangent_checks'])
    assert not record['physical_qualification'] and record['conductivity_derivatives']
    failed=copy.copy(closure);failed.max_iterations=0;failed._versions=failed._state()
    with pytest.raises(DualCurrentConvergenceError,match='iteration_budget_exhausted'):failed.solve(ac,.3)
    tangent.tangent[0,0,0]+=1
    with pytest.raises(ValueError,match='binding changed'):tangent.apply(value)
    with torch.no_grad():conductivity[0]+=.1
    with pytest.raises(RuntimeError,match='mutated'):closure.solve(ac,.3)
    with pytest.raises(RuntimeError,match='mutated'):torch.autograd.grad(loss(actual),conductivity)


def test_streamed_dual_rejects_inactive_policy():
    with pytest.raises(ValueError,match='Inactive'):
        DualHarmonicCurrentClosure((),[],[],streamed_column_batch=3)
