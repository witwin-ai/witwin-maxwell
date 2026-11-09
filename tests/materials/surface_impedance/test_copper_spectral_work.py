"""Independent rational-work and coefficient derivatives of spectral foil loss."""
import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.copper_foil import compile_copper_response
from witwin.maxwell.fdtd.copper_foil import copper_spectral_power


@pytest.mark.parametrize("scale", [.3, 1., 2.7])
def test_spectral_midpoint_work_matches_independent_rational_response(scale):
    response = compile_copper_response(mw.CopperFoil(thickness=2e-6,
        conductivity=5.8e7, frequency_range=(.5e9, 4e9)))
    frequencies = torch.tensor([.8e9, 1.7e9, 3.2e9], dtype=torch.float64)
    dt = 2e-12
    generator = torch.Generator().manual_seed(87)
    phasor = torch.randn((3,2,7), dtype=torch.complex128, generator=generator)
    area = torch.linspace(1e-6, 3e-6, 7, dtype=torch.float64)
    B = response.state_space.B*math.sqrt(scale)
    D = response.state_space.D*scale
    actual = copper_spectral_power(phasor, frequencies, dt, B, response.rates, D, area)
    angle = torch.pi*frequencies*dt
    admittance = response.rational.evaluate(torch.tan(angle)/(torch.pi*dt))*scale
    expected = []
    for f in range(3):
        work = 0.
        for edge in range(7):
            electric = phasor[f,:,edge]
            work += .5*torch.vdot(electric, admittance[f]@electric).real*area[edge]*torch.cos(angle[f]).square()
        expected.append(work)
    torch.testing.assert_close(actual, torch.stack(expected), rtol=2e-13, atol=1e-16)
    assert bool(torch.all(actual>0))


def test_spectral_work_retains_complex_field_and_direct_realization_derivatives():
    generator = torch.Generator().manual_seed(13)
    phasor = torch.randn((2,2,3), dtype=torch.complex128, generator=generator, requires_grad=True)
    B = torch.randn((4,2), dtype=torch.float64, generator=generator, requires_grad=True)
    rates = torch.linspace(.2, 3., 4, dtype=torch.float64, requires_grad=True)
    D = torch.tensor([[.4,.1],[.1,.3]], dtype=torch.float64, requires_grad=True)
    frequencies = torch.tensor([.3,.8], dtype=torch.float64)
    area = torch.tensor([.2,.7,.4], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda e,b,r,d:
        copper_spectral_power(e, frequencies, .03, b, r, d, area),
        (phasor,B,rates,D), eps=1e-6, atol=2e-8, rtol=2e-5)


@pytest.mark.parametrize("frequency", [-.1, float("nan"), .75/.03])
def test_spectral_work_preserves_rational_warped_frequency_validation(frequency):
    with pytest.raises(ValueError,match="finite and non-negative"):
        copper_spectral_power(torch.ones((1,2,1),dtype=torch.complex128),
            torch.tensor([frequency],dtype=torch.float64),.03,torch.ones((2,2),dtype=torch.float64),
            torch.ones(2,dtype=torch.float64),torch.eye(2,dtype=torch.float64),torch.ones(1))
