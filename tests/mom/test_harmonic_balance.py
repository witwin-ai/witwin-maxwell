"""Algebraic HB fixtures, not electromagnetic production references."""
import math
import pytest
import torch
from witwin.maxwell.constants import EPSILON_0
from witwin.maxwell.mom.harmonic_balance import DenseHarmonicBalance,HarmonicBalanceConvergenceError

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def model(*, chi3=0., coupling=-1., drive=1., harmonics=1, chi2=0.):
    background = torch.zeros(harmonics+1,1,3,device='cuda',dtype=torch.complex128)
    background[1,0,0] = drive
    maps = torch.eye(3,device='cuda',dtype=torch.complex128)[None].repeat(harmonics+1,1,1)*coupling/EPSILON_0
    current = torch.zeros_like(maps)
    omega = 2*math.pi*3e8*torch.arange(1,harmonics+1,device='cuda',dtype=torch.float64)
    current[1:] = maps[1:]/(-1j*omega[:,None,None])
    return DenseHarmonicBalance(background,maps,current,torch.ones(1,device='cuda',dtype=torch.float64),
        3e8,chi3=chi3,chi2=chi2)


def test_hb_linear_reduction_and_cubic_implicit_gradient():
    linear = model()
    result = linear.solve()
    torch.testing.assert_close(result.electric,linear.background,rtol=0,atol=0)
    assert result.receipt['converged'] and result.receipt['iterations'] == 0
    chi = torch.tensor(.2,device='cuda',dtype=torch.float64,requires_grad=True)
    nonlinear = model(chi3=chi)
    result = nonlinear.solve()
    amplitude = result.electric[1,0,0].real
    torch.testing.assert_close(amplitude+.75*chi*amplitude**3,amplitude.new_tensor(1.),rtol=1e-10,atol=1e-10)
    derivative, = torch.autograd.grad(amplitude,chi)
    expected = -.75*amplitude**3/(1+2.25*chi*amplitude**2)
    torch.testing.assert_close(derivative,expected,rtol=1e-8,atol=1e-10)
    assert result.receipt['omitted_polarization_l2_C_per_m2'] > 0
    assert not result.receipt['harmonic_truncation_converged'] and not result.receipt['physical_qualification']
    picard = nonlinear.solve(method='picard',max_iterations=100,damping=.5)
    torch.testing.assert_close(picard.electric,result.electric,rtol=1e-8,atol=1e-9)


def test_hb_explicit_continuation_tracks_distinct_cubic_roots():
    nonlinear = model(chi3=4/3,coupling=1.,drive=.05)
    branches = []
    for sign in (-1.,1.):
        initial = torch.zeros_like(nonlinear.background)
        initial[1,0,0] = sign*.9
        results = nonlinear.solve_ramp((.2,.5,1.),branch_id=f'root_{sign}',initial=initial)
        branches.append(results[-1].electric[1,0,0].real)
        for index,result in enumerate(results):
            assert result.receipt['continuation_step'] == index
            assert not result.receipt['stability_assessed']
    assert branches[0] < -.9 and branches[1] > .9
    # At zero pump the phase/orientation continuum has no isolated root.
    singular = torch.zeros_like(nonlinear.background)
    singular[1,0,0] = .9
    with pytest.raises(HarmonicBalanceConvergenceError) as failure:
        nonlinear.solve_ramp((0.,.5),branch_id='degenerate_zero_pump',initial=singular)
    assert failure.value.receipt['failure_reason'] in ('singular_newton_jacobian','singular_implicit_jacobian')
    assert failure.value.receipt['failed_step'] == 0


def test_hb_dc_rectification_budget_failure_and_response_contract():
    nonlinear = model(chi2=.2,drive=.2,harmonics=2)
    result = nonlinear.solve()
    assert result.electric[0,0,0].real < 0 and result.electric[2,0,0].abs() > 0
    assert torch.count_nonzero(result.electric[0].imag) == 0
    power = result.receipt['nonlinear_power']
    assert power['harmonics'] == (0,1,2)
    assert power['polarization_conservative'] and power['dissipation_parseval_closed']
    assert power['dissipation_W'] == [0.,0.,0.]
    assert not power['global_power_balance_verified']
    assert not power['harmonic_truncation_converged']
    with pytest.raises(HarmonicBalanceConvergenceError) as failure:
        nonlinear.solve(max_iterations=0)
    assert failure.value.receipt['failure_reason'] == 'iteration_budget_exhausted'
    with pytest.raises(ValueError,match='P/J'):
        DenseHarmonicBalance(nonlinear.background,nonlinear.polarization_response,
            nonlinear.current_response*2,nonlinear.cell_volumes,3e8)


def test_continuation_preserves_inner_dc_failure_receipt(monkeypatch):
    from witwin.maxwell.mom.static_current import StaticCurrentConvergenceError
    solver=model()
    original=solver.solve
    def fail_second(initial=None,**settings):
        if settings['pump_scale']==1.:
            raise StaticCurrentConvergenceError({'failure_reason':'singular_current_tangent','converged':False})
        return original(initial,**settings)
    monkeypatch.setattr(solver,'solve',fail_second)
    with pytest.raises(StaticCurrentConvergenceError) as caught:
        solver.solve_ramp((.5,1.),branch_id='dc_failure')
    receipt=caught.value.receipt
    assert receipt['failure_reason']=='singular_current_tangent'
    assert receipt['branch_id']=='dc_failure' and receipt['failed_step']==1
    assert len(receipt['completed_steps'])==1 and receipt['completed_steps'][0]['converged']
