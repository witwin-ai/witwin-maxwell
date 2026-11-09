"""Residual-checked fixed HB tangent solves and RHS implicit derivatives."""
from dataclasses import replace
import runpy
import pytest
import torch
from witwin.maxwell.mom.harmonic_jacobian_action import (
    HarmonicJacobianAction,FixedHarmonicTangentInverse,HarmonicTangentConvergenceError)
from witwin.maxwell.mom.linalg import StoppingRule

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_fixed_tangent_inverse_gradients_and_failure(monkeypatch):
    model=runpy.run_path('tests/mom/test_harmonic_balance.py')['model'](harmonics=2,chi2=.2,chi3=.3)
    field=model.background.clone();field[0]=.1;field[1]+=.2j;field[2]=.05-.1j
    tangent=HarmonicJacobianAction(model,field,pump_scale=.7)
    matrix=torch.autograd.functional.jacobian(tangent._equation,tangent.vector)
    rule=StoppingRule(preconditioner='none',rtol=1e-12,atol=1e-14,maxiter=60,restart=12)
    inverse=FixedHarmonicTangentInverse(tangent,rule)
    torch.manual_seed(19)
    value=torch.randn(2,3,len(tangent.vector),device='cuda',dtype=tangent.dtype,requires_grad=True)
    expected=torch.linalg.solve(matrix,value.unsqueeze(-1)).squeeze(-1)
    transposed=torch.linalg.solve(matrix.mT,value.unsqueeze(-1)).squeeze(-1)
    grad_expected=torch.linalg.solve(matrix.mT,(2*expected).unsqueeze(-1)).squeeze(-1)
    second_expected=torch.linalg.solve(matrix.mT,(2*torch.linalg.solve(matrix,torch.ones_like(value).unsqueeze(-1))).squeeze(-1).unsqueeze(-1)).squeeze(-1)
    def forbidden(*args,**kwargs):raise AssertionError('No complete Jacobian, direct solve or SVD allowed')
    with monkeypatch.context() as patch:
        patch.setattr(torch.autograd.functional,'jacobian',forbidden)
        patch.setattr(torch.linalg,'solve',forbidden)
        patch.setattr(torch.linalg,'svdvals',forbidden)
        actual=inverse.apply(value)
        torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-11)
        torch.testing.assert_close(inverse.adjoint(value),transposed,rtol=1e-10,atol=1e-11)
        gradient,=torch.autograd.grad(actual.square().sum(),value,create_graph=True)
        torch.testing.assert_close(gradient,grad_expected,rtol=1e-10,atol=1e-11)
        second,=torch.autograd.grad(gradient.sum(),value)
        torch.testing.assert_close(second,second_expected,rtol=1e-10,atol=1e-11)
    assert inverse.receipt['last_forward']['converged']
    assert not inverse.receipt['certified_nonsingular']
    assert inverse.apply(value[:0]).shape==value[:0].shape
    torch.testing.assert_close(inverse.apply(torch.zeros_like(value)),torch.zeros_like(value),rtol=0,atol=0)
    failed=FixedHarmonicTangentInverse(tangent,replace(rule,maxiter=1),refinement_steps=0)
    with pytest.raises(HarmonicTangentConvergenceError) as error:failed.apply(value[0,0])
    assert not error.value.receipt['converged']
    assert error.value.receipt['rhs'][0]['absolute_residual']>error.value.receipt['rhs'][0]['threshold']
    delayed=inverse.apply(value)
    with torch.no_grad():tangent.vector[0]+=1
    with pytest.raises(ValueError,match='changed'):torch.autograd.grad(delayed.sum(),value)


def check_compressed_inverse(model,field):
    """Reuse actual saved preparation to avoid another nonlinear root solve."""
    tangent=HarmonicJacobianAction(model,field)
    inverse=FixedHarmonicTangentInverse(tangent,StoppingRule(
        preconditioner='none',rtol=1e-9,atol=1e-12,maxiter=80,restart=20))
    generator=torch.Generator(device=tangent.device).manual_seed(37)
    rhs=torch.randn(tangent.shape[0],device=tangent.device,dtype=tangent.dtype,generator=generator,requires_grad=True)
    tolerance=1e-9*float(torch.linalg.vector_norm(rhs.detach()))
    solution=inverse.apply(rhs)
    torch.testing.assert_close(tangent.apply(solution),rhs,rtol=0,atol=tolerance)
    transpose=inverse.adjoint(rhs)
    torch.testing.assert_close(tangent.adjoint(transpose),rhs,rtol=0,atol=tolerance)
    gradient,=torch.autograd.grad((solution*rhs.detach()).sum(),rhs)
    torch.testing.assert_close(gradient,transpose,rtol=1e-10,atol=1e-11)
    return inverse.receipt


def test_overflowed_original_threshold_is_rejected():
    model=runpy.run_path('tests/mom/test_harmonic_balance.py')['model']()
    tangent=HarmonicJacobianAction(model,model.background)
    inverse=FixedHarmonicTangentInverse(tangent,StoppingRule(preconditioner='none',rtol=1e308),refinement_steps=0)
    rhs=torch.ones(tangent.shape[0],device=tangent.device,dtype=tangent.dtype)
    with pytest.raises(HarmonicTangentConvergenceError) as error:inverse.apply(rhs)
    log=error.value.receipt
    assert not log['converged'] and log['rhs'][0]['threshold']==float('inf')
    assert log['rhs'][0]['absolute_residual']>0
