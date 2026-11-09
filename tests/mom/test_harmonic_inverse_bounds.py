"""All-column numerical inverse bounds, including nondominant HB tangents."""
import math
import runpy
import pytest
import torch
from witwin.maxwell.constants import EPSILON_0
from witwin.maxwell.mom.harmonic_balance import DenseHarmonicBalance
from witwin.maxwell.mom.harmonic_jacobian_action import (
    HarmonicJacobianAction,streamed_tangent_bounds,streamed_inverse_tangent_bounds)
from witwin.maxwell.mom.linalg import StoppingRule

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def nondominant_model(chi=.3):
    base=runpy.run_path('tests/mom/test_harmonic_balance.py')['model']()
    polarization=torch.zeros_like(base.polarization_response)
    polarization[1,0,1]=20/EPSILON_0;polarization[1,1,0]=-20/EPSILON_0
    current=torch.zeros_like(polarization)
    current[1]=polarization[1]/(-1j*2*math.pi*base.frequency)
    return DenseHarmonicBalance(base.background,polarization,current,base.cell_volumes,base.frequency,chi3=chi)


def test_nondominant_inverse_bounds_without_dense_storage(monkeypatch):
    model=nondominant_model()
    tangent=HarmonicJacobianAction(model,model.background)
    matrix=torch.autograd.functional.jacobian(tangent._equation,tangent.vector)
    singular=torch.linalg.svdvals(matrix)
    assert not streamed_tangent_bounds(tangent,column_batch=2)['admitted']
    rule=StoppingRule(preconditioner='none',rtol=1e-12,atol=1e-14,maxiter=30,restart=9)
    def forbidden(*args,**kwargs):raise AssertionError('Global Jacobian/direct solve/SVD forbidden')
    with monkeypatch.context() as patch:
        patch.setattr(torch.autograd.functional,'jacobian',forbidden)
        patch.setattr(torch.linalg,'solve',forbidden)
        patch.setattr(torch.linalg,'svdvals',forbidden)
        for batch in (1,4):
            bound=streamed_inverse_tangent_bounds(tangent,rule,column_batch=batch)
            assert bound['complete'] and bound['admitted'] and not bound['inconclusive']
            assert bound['columns_checked']==tangent.shape[0]
            assert 0<bound['sigma_min_lower']<=float(singular[-1])
            assert bound['sigma_max_upper']>=float(singular[0])
            assert bound['inverse_residual_norm_upper']<1
            assert not bound['interval_certified'] and not bound['physical_qualification']
    short=StoppingRule(preconditioner='none',rtol=1e-15,atol=1e-20,maxiter=1,restart=1)
    failed=streamed_inverse_tangent_bounds(tangent,short,column_batch=2)
    assert not failed['complete'] and not failed['admitted'] and failed['inconclusive']
    assert failed['sigma_min_lower']==0 and failed['sigma_max_upper'] is None
    assert failed['solve_failure'] and failed['columns_checked']<tangent.shape[0]


def test_singular_inverse_bounds_are_inconclusive():
    model=runpy.run_path('tests/mom/test_harmonic_balance.py')['model'](chi3=-4/9,drive=2/3)
    field=model.background.clone();field[1,0,0]=1
    tangent=HarmonicJacobianAction(model,field)
    rule=StoppingRule(preconditioner='none',rtol=1e-10,atol=1e-13,maxiter=12,restart=9)
    bound=streamed_inverse_tangent_bounds(tangent,rule,column_batch=2)
    assert not bound['admitted'] and bound['inconclusive']
    assert bound['sigma_min_lower']==0
    # Deliberately loose solves may return zero columns but cannot imply an inverse.
    loose=StoppingRule(preconditioner='none',rtol=2.,atol=2.)
    bound=streamed_inverse_tangent_bounds(tangent,loose,column_batch=2)
    assert bound['complete'] and not bound['admitted'] and bound['inconclusive']
    assert bound['inverse_residual_norm_upper']>=1 and bound['inverse_norm_upper']==0


def test_nondominant_root_and_material_gradient(monkeypatch):
    chi=torch.tensor(.3,device='cuda',dtype=torch.float64,requires_grad=True)
    model=nondominant_model(chi)
    expected=model.solve()
    gradient,=torch.autograd.grad(expected.electric.abs().square().sum(),chi,retain_graph=True)
    tangent=HarmonicJacobianAction(model,expected.electric)
    assert not streamed_tangent_bounds(tangent,column_batch=2)['admitted']
    def forbidden(*args,**kwargs):raise AssertionError('Global Jacobian/direct solve/SVD forbidden')
    monkeypatch.setattr(torch.autograd.functional,'jacobian',forbidden)
    monkeypatch.setattr(torch.linalg,'solve',forbidden)
    monkeypatch.setattr(torch.linalg,'svdvals',forbidden)
    actual=model.solve(method='newton_krylov',implicit_solver='streamed_inverse_krylov',tangent_column_batch=2)
    actual_gradient,=torch.autograd.grad(actual.electric.abs().square().sum(),chi)
    torch.testing.assert_close(actual.electric,expected.electric,rtol=1e-9,atol=1e-10)
    torch.testing.assert_close(actual_gradient,gradient,rtol=1e-8,atol=1e-10)
    assert all(check['admitted'] for check in actual.receipt['implicit_tangent_checks'])
