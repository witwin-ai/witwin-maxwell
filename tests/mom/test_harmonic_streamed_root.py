"""Complete streamed tangent checks and implicit roots without HB matrix storage."""
import runpy
import pytest
import torch
from witwin.maxwell.mom.harmonic_jacobian_action import HarmonicJacobianAction,streamed_tangent_bounds
from witwin.maxwell.mom.harmonic_balance import HarmonicBalanceConvergenceError

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('implicit',['streamed_krylov','streamed_inverse_krylov'])
def test_streamed_root_and_material_gradient_without_global_jacobian(monkeypatch,implicit):
    factory=runpy.run_path('tests/mom/test_harmonic_balance.py')['model']
    chi=torch.tensor(.03,device='cuda',dtype=torch.float64,requires_grad=True)
    model=factory(harmonics=2,chi2=.02,chi3=chi)
    expected=model.solve()
    expected_gradient,=torch.autograd.grad(expected.electric.abs().square().sum(),chi,retain_graph=True)
    tangent=HarmonicJacobianAction(model,expected.electric)
    matrix=torch.autograd.functional.jacobian(tangent._equation,tangent.vector)
    singular=torch.linalg.svdvals(matrix)
    def forbidden(*args,**kwargs):raise AssertionError('Complete HB matrix/direct solve/SVD forbidden')
    with monkeypatch.context() as patch:
        patch.setattr(torch.autograd.functional,'jacobian',forbidden)
        patch.setattr(torch.linalg,'solve',forbidden)
        patch.setattr(torch.linalg,'svdvals',forbidden)
        bounds=streamed_tangent_bounds(tangent,column_batch=2)
        assert bounds['admitted'] and bounds['columns_checked']==len(tangent.vector)
        assert bounds['sigma_min_lower']<=float(singular[-1])
        assert bounds['sigma_max_upper']>=float(singular[0])
        actual=model.solve(method='newton_krylov',implicit_solver=implicit,tangent_column_batch=2)
        gradient,=torch.autograd.grad(actual.electric.abs().square().sum(),chi)
    torch.testing.assert_close(actual.electric,expected.electric,rtol=1e-9,atol=1e-10)
    torch.testing.assert_close(gradient,expected_gradient,rtol=1e-8,atol=1e-10)
    assert not actual.receipt['global_hb_jacobian_stored']
    assert len(actual.receipt['implicit_tangent_checks'])==2
    assert not actual.receipt['fully_matrix_free']


@pytest.mark.parametrize('implicit',['streamed_krylov','streamed_inverse_krylov'])
def test_streamed_singular_root_is_not_accepted(implicit):
    model=runpy.run_path('tests/mom/test_harmonic_balance.py')['model'](chi3=-4/9,drive=2/3)
    initial=model.background.clone();initial[1,0,0]=1
    with pytest.raises(HarmonicBalanceConvergenceError,match='implicit_tangent_bound_inconclusive') as error:
        model.solve(initial,method='newton_krylov',implicit_solver=implicit)
    assert not error.value.receipt['implicit_tangent_checks'][0]['admitted']
    with pytest.raises(ValueError,match='requires newton_krylov'):
        model.solve(implicit_solver='streamed_krylov')


def test_nondominant_invertible_tangent_is_reported_inconclusive():
    import math
    from witwin.maxwell.constants import EPSILON_0
    from witwin.maxwell.mom.harmonic_balance import DenseHarmonicBalance
    base=runpy.run_path('tests/mom/test_harmonic_balance.py')['model']()
    polarization=torch.zeros_like(base.polarization_response)
    polarization[1,0,1]=20/EPSILON_0;polarization[1,1,0]=-20/EPSILON_0
    current=torch.zeros_like(polarization)
    current[1]=polarization[1]/(-1j*2*math.pi*base.frequency)
    model=DenseHarmonicBalance(base.background,polarization,current,base.cell_volumes,base.frequency,chi3=.3)
    tangent=HarmonicJacobianAction(model,model.background)
    matrix=torch.autograd.functional.jacobian(tangent._equation,tangent.vector)
    assert float(torch.linalg.svdvals(matrix)[-1])>.1
    bounds=streamed_tangent_bounds(tangent,column_batch=2)
    assert not bounds['admitted'] and bounds['inconclusive']
    assert bounds['sigma_min_lower']==0
