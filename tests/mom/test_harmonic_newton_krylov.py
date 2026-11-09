"""Matrix-free Newton directions retain existing final root qualification gates."""
from dataclasses import replace
import runpy
import pytest
import torch
from witwin.maxwell.mom.harmonic_balance import HarmonicBalanceConvergenceError
from witwin.maxwell.mom.linalg import StoppingRule

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_newton_krylov_root_parameter_gradient_and_direction_storage(monkeypatch):
    factory=runpy.run_path('tests/mom/test_harmonic_balance.py')['model']
    chi=torch.tensor(.3,device='cuda',dtype=torch.float64,requires_grad=True)
    model=factory(harmonics=2,chi2=.2,chi3=chi)
    expected=model.solve()
    expected_gradient,=torch.autograd.grad(expected.electric.abs().square().sum(),chi,retain_graph=True)
    original=torch.autograd.functional.jacobian
    calls=[]
    def record(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(torch.autograd.functional,'jacobian',record)
    actual=model.solve(method='newton_krylov')
    assert actual.receipt['iterations']>0 and len(calls)==2
    assert actual.receipt['implicit_root_solver']=='dense_svd_checked'
    assert not actual.receipt['fully_matrix_free']
    torch.testing.assert_close(actual.electric,expected.electric,rtol=1e-9,atol=1e-10)
    gradient,=torch.autograd.grad(actual.electric.abs().square().sum(),chi)
    torch.testing.assert_close(gradient,expected_gradient,rtol=1e-8,atol=1e-10)
    assert all(c['tangent_solve']['converged'] for c in actual.receipt['checks'][:-1])
    assert not actual.receipt['physical_qualification']


def test_newton_krylov_failure_and_singular_final_root():
    factory=runpy.run_path('tests/mom/test_harmonic_balance.py')['model']
    model=factory(harmonics=2,chi2=.2,chi3=.3)
    rule=StoppingRule(preconditioner='none',maxiter=1,rtol=1e-15,atol=1e-20)
    with pytest.raises(HarmonicBalanceConvergenceError) as error:
        model.solve(method='newton_krylov',tangent_rule=rule)
    assert error.value.receipt['failure_reason']=='newton_krylov_residual_failed'
    assert not error.value.receipt['checks'][-1]['tangent_solve']['converged']
    with pytest.raises(ValueError,match='requires newton_krylov'):model.solve(tangent_rule=rule)
    with pytest.raises(ValueError,match='requires unpreconditioned'):
        model.solve(method='newton_krylov',tangent_rule=replace(rule,preconditioner='row_norm'))
    # The cubic has a double root at E=1 with drive=2/3, chi3=-4/9.
    singular=factory(chi3=-4/9,drive=2/3)
    initial=singular.background.clone();initial[1,0,0]=1
    with pytest.raises(HarmonicBalanceConvergenceError,match='singular_implicit_jacobian'):
        singular.solve(initial,method='newton_krylov')


@pytest.mark.parametrize('method',['newton','picard','newton_krylov'])
def test_outer_threshold_overflow_rejected_before_root_acceptance(method):
    model=runpy.run_path('tests/mom/test_harmonic_balance.py')['model'](drive=2.)
    with pytest.raises(ValueError,match='HB residual threshold must be finite'):
        model.solve(method=method,relative_tolerance=1e308)


def check_compressed_newton(model,field,settings,*,implicit_solver='dense'):
    """Compare with an existing root, reusing its prepared physical operators."""
    solution=model.solve(.95*field,method='newton_krylov',pump_scale=settings.pump_scales[-1],
        implicit_solver=implicit_solver,
        relative_tolerance=settings.relative_tolerance,absolute_tolerance=settings.absolute_tolerance,
        max_iterations=12)
    torch.testing.assert_close(solution.electric,field,rtol=2e-7,atol=1e-9)
    assert solution.receipt['iterations']>0 and not solution.receipt['fully_matrix_free']
    assert all(c['tangent_solve']['converged'] for c in solution.receipt['checks'][:-1])
    return solution
