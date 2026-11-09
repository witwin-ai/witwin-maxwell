"""Nonlinear work checked by analytic time derivatives, not FFT replay."""
import math
import pytest
import torch
from witwin.maxwell.constants import EPSILON_0
from witwin.maxwell.mom.nonlinear import instantaneous_harmonic_sources,harmonic_power_ledger

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_harmonic_work_matches_independent_time_integral_and_native_gradients():
    field = torch.tensor([[[.2,-.1,.3],[.1,.4,-.2]],
                          [[1+.4j,.2-.3j,-.5j],[.1j,1-.2j,.3]],
                          [[.3j,.25,-.2],[.4+.1j,-.3j,.2]]],device='cuda',dtype=torch.complex128)
    chi2,chi3 = .4,.7
    scale = torch.tensor(.6,device='cuda',dtype=torch.float64,requires_grad=True)
    volumes = torch.tensor([.002,.003],device='cuda',dtype=torch.float64)
    frequency = 3e8
    sources = instantaneous_harmonic_sources(field,chi2=chi2,chi3=chi3,tpa_scale=scale)
    ledger = harmonic_power_ledger(field,sources,frequency,volumes)
    phase = torch.arange(4096,device='cuda',dtype=torch.float64)*(2*math.pi/4096)
    orders = torch.arange(1,len(field),device='cuda',dtype=torch.float64)
    oscillation = torch.exp(-1j*phase[:,None]*orders)
    time_field = field[0].real+torch.einsum('tn,nci->tci',oscillation,field[1:]).real
    derivative = torch.einsum('tn,nci->tci',oscillation,(-2j*math.pi*frequency*orders[:,None,None])*field[1:]).real
    intensity = time_field.square().sum(-1)
    pdot = EPSILON_0*(2*chi2*time_field*derivative+chi3*(
        2*(time_field*derivative).sum(-1)[...,None]*time_field+intensity[...,None]*derivative))
    dissipative = scale*intensity[...,None]*time_field
    # Compare each field harmonic's work against the full time-domain current.
    for n in range(len(field)):
        component = field[0].real.expand_as(time_field) if n == 0 else (
            torch.exp(-1j*n*phase)[:,None,None]*field[n]).real
        exchange = ((component*pdot).sum(-1).mean(0)*volumes).sum()
        loss = ((component*dissipative).sum(-1).mean(0)*volumes).sum()
        torch.testing.assert_close(ledger['polarization_exchange_W'][n],exchange,rtol=1e-12,atol=1e-18)
        torch.testing.assert_close(ledger['dissipation_W'][n],loss,rtol=1e-12,atol=1e-17)
    assert ledger['polarization_exchange_W'][1:].abs().min() > 1e-7
    assert ledger['polarization_conservative'] and ledger['dissipation_parseval_closed']
    assert ledger['passive_dissipation'] and not ledger['global_power_balance_verified']
    gradient, = torch.autograd.grad(ledger['dissipation_W'].sum(),scale)
    torch.testing.assert_close(gradient,ledger['constitutive_dissipation_W']/scale,rtol=1e-12,atol=1e-17)
    # Omitting peak-phasor 1/2 or assigning it to DC fails this nonzero-DC fixture.
    naive = ((field.conj()*sources.dissipative_current[:len(field)]).real.sum(-1)*volumes).sum()
    assert (naive-ledger['constitutive_dissipation_W']).abs() > 1e-4
