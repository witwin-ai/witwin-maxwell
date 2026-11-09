"""Action-backed harmonics retain the real conductive DC closure contract."""
import math
import runpy
import pytest
import torch
from witwin.maxwell.mom.action_harmonic_balance import ActionHarmonicBalance
from witwin.maxwell.mom.harmonic_balance import DenseHarmonicBalance
from witwin.maxwell.mom.harmonic_current import HarmonicCurrentClosure

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_action_dc_closure_residual_gradient_and_exact_ordered_binding():
    _,bindings=runpy.run_path('tests/mom/test_surface_volume.py')['problem']((2.,3.))
    volumes=[b.volume for b in bindings]
    build=runpy.run_path('tests/mom/test_action_harmonic_balance.py')['maps']
    # Small nonlinear currents require an absolute solve floor below their RHS.
    data=[build(volumes,n*3e8,solve_atol=1e-20) for n in (1,2)];actions=[d[0] for d in data]
    original=runpy.run_path('tests/mom/test_coupled_volume_action.py')['original_matrix']
    response=[original(*d)[0] for d in data]
    zero=torch.zeros_like(response[0]);weights=torch.cat([v.volumes for v in volumes])
    background=zero.new_zeros(3,2,3);background[1]=.02;background[2]=.01+.005j
    closure=HarmonicCurrentClosure(volumes,[.001,.002])
    settings=dict(chi2=.002,chi3=.001,tpa_scale=.02,dc_current_closure=closure)
    model=ActionHarmonicBalance(background,actions,weights,3e8,**settings)
    dense=DenseHarmonicBalance(background,torch.stack((zero,*[-1j*(2*math.pi*3e8*n)*v for n,v in enumerate(response,1)])),torch.stack((zero,*response)),weights,3e8,**settings)
    electric=background.clone().requires_grad_()
    actual=model.residual(electric);expected=dense.residual(electric)
    assert bool(actual[0].abs().max()>0)
    torch.testing.assert_close(actual,expected,rtol=2e-8,atol=1e-11)
    ga,=torch.autograd.grad(actual.abs().square().sum(),electric)
    gb,=torch.autograd.grad(expected.abs().square().sum(),electric)
    torch.testing.assert_close(ga,gb,rtol=3e-8,atol=1e-10)
    assert model.dc_polarization_response is None and model.dc_current_response is None
    with pytest.raises(ValueError,match='must not also'):
        ActionHarmonicBalance(background,actions,weights,3e8,dc_polarization_response=zero,**settings)
    wrong=HarmonicCurrentClosure(volumes[::-1],[.002,.001])
    with pytest.raises(ValueError,match='ordered volume'):
        ActionHarmonicBalance(background,actions,weights,3e8,dc_current_closure=wrong)
