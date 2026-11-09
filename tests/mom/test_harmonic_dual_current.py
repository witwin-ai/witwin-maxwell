"""Actual cubic DC feedback, independent time averaging and implicit gradients."""
import pytest
import torch
from witwin.maxwell.mom.harmonic_dual_current import DualHarmonicCurrentClosure,DualCurrentConvergenceError
from witwin.maxwell.mom.harmonic_current import HarmonicCurrentClosure
from test_dual_static_current import volume

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def ac_fields(count):
    values=torch.arange(6*count,device='cuda',dtype=torch.float64).reshape(2,count,3)
    return torch.complex(.8*torch.cos(values*.3),.4*torch.sin(values*.7))


def test_actual_mean_cubic_feedback_and_original_constitutive_current():
    closure=DualHarmonicCurrentClosure((volume(),volume(2,.3)),[2.,3.],[5.,7.])
    ac=ac_fields(8); result=closure.solve(ac,.4)
    electric=result['electric']
    assert float(electric.norm())>1e-5 and result['nonlinear_receipt']['converged']
    angle=torch.arange(257,device='cuda',dtype=torch.float64)*2*torch.pi/257
    phase=torch.exp(-1j*angle[:,None]*torch.arange(1,3,device='cuda')[None])
    real=electric[None]+torch.einsum('sh,hti->sti',phase,ac).real
    expected=.4*(real.square().sum(-1)[...,None]*real).mean(0)
    torch.testing.assert_close(result['tpa_current'],expected,rtol=1e-11,atol=1e-13)
    sigma=closure.current.electric.reshape(8,3)
    torch.testing.assert_close(result['electric_current'],sigma*electric+expected,rtol=1e-11,atol=1e-13)
    assert not result['nonlinear_receipt']['physical_qualification']


def test_implicit_amplitude_and_tpa_gradients_match_resolved_differences():
    closure=DualHarmonicCurrentClosure((volume(),),[2.],[5.],absolute_tolerance=1e-13)
    ac=ac_fields(4); scale=ac.real.new_tensor(1.,requires_grad=True); tpa=ac.real.new_tensor(.3,requires_grad=True)
    def evaluate(a,b): return closure.electric(a*ac,b).square().sum()
    gradient=torch.autograd.grad(evaluate(scale,tpa),(scale,tpa))
    step=1e-4
    for index,actual in enumerate(gradient):
        plus=[scale.detach(),tpa.detach()]; minus=list(plus)
        plus[index]=plus[index]+step; minus[index]=minus[index]-step
        expected=(evaluate(*plus)-evaluate(*minus))/(2*step)
        torch.testing.assert_close(actual,expected,rtol=2e-6,atol=1e-9)


def test_zero_tpa_and_magnetic_conduction_limit():
    v=volume(); ac=ac_fields(4)
    closure=DualHarmonicCurrentClosure((v,),[2.],[5.])
    result=closure.solve(ac,0.)
    torch.testing.assert_close(result['electric'],torch.zeros_like(result['electric']),rtol=0,atol=0)
    expected=HarmonicCurrentClosure((v,),[2.]).electric(ac,.3)
    low=DualHarmonicCurrentClosure((v,),[2.],[1e-7]).electric(ac,.3)
    torch.testing.assert_close(low,expected,rtol=1e-6,atol=1e-9)


def test_failure_receipt_and_prepared_response_guard():
    closure=DualHarmonicCurrentClosure((volume(),),[2.],[5.],max_iterations=0)
    with pytest.raises(DualCurrentConvergenceError) as caught: closure.solve(ac_fields(4),.4)
    assert caught.value.receipt['failure_reason']=='iteration_budget_exhausted'
    assert not caught.value.receipt['converged'] and caught.value.receipt['checks']
    closure.response.add_(.1)
    with pytest.raises(RuntimeError,match='mutated'): closure.solve(ac_fields(4),.4)
