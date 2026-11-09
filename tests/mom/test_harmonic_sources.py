"""Independent trigonometric nonlinear source and peak-phasor normalization gates."""
import pytest
import torch
from witwin.maxwell.constants import EPSILON_0
from witwin.maxwell.mom.nonlinear import instantaneous_harmonic_sources

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_chi2_dc_second_harmonic_and_chi3_fundamental_third_harmonic():
    e = torch.zeros(2, 3, device='cuda', dtype=torch.complex128)
    amplitude = 2.3*torch.exp(e.new_tensor(.37j))
    e[1, 0] = amplitude
    second = instantaneous_harmonic_sources(e, chi2=.4)
    torch.testing.assert_close(second.polarization[0, 0], EPSILON_0*.4*amplitude.abs().square()/2, rtol=1e-13, atol=1e-24, check_dtype=False)
    torch.testing.assert_close(second.polarization[2, 0], EPSILON_0*.4*amplitude.square()/2, rtol=1e-13, atol=1e-24)
    third = instantaneous_harmonic_sources(e, chi3=.7)
    torch.testing.assert_close(third.polarization[1, 0], EPSILON_0*.7*.75*amplitude.abs().square()*amplitude, rtol=1e-13, atol=1e-24)
    torch.testing.assert_close(third.polarization[3, 0], EPSILON_0*.7*.25*amplitude**3, rtol=1e-13, atol=1e-24)
    assert torch.count_nonzero(second.polarization_current(3e8)[0]) == 0
    assert second.polarization[0, 0].abs() > 0


def test_vector_tpa_power_matches_parseval_and_retains_native_gradient():
    e = torch.tensor([[.2, -.1, .3], [1+.4j, .2-.3j, -.5j], [.1j, .15, -.2]], device='cuda', dtype=torch.complex128, requires_grad=True)
    scale = torch.tensor(.6, device='cuda', dtype=torch.float64, requires_grad=True)
    result = instantaneous_harmonic_sources(e, tpa_scale=scale)
    power = (e[0].conj()*result.dissipative_current[0]).sum().real
    power += .5*(e[1:].conj()*result.dissipative_current[1:len(e)]).sum().real
    torch.testing.assert_close(result.average_dissipation, power, rtol=1e-13, atol=1e-14)
    assert power > 0
    gradient, = torch.autograd.grad(result.average_dissipation, scale)
    torch.testing.assert_close(gradient, power/scale, rtol=1e-13, atol=1e-14)


def test_cubic_generated_bandwidth_is_not_folded_into_retained_harmonics():
    e = torch.zeros(4, 3, device='cuda', dtype=torch.complex128)
    e[3, 0] = 2
    result = instantaneous_harmonic_sources(e, chi3=1.)
    assert result.polarization.shape == (10, 3)
    expected = torch.zeros_like(result.polarization)
    expected[3, 0], expected[9, 0] = EPSILON_0*6, EPSILON_0*2
    torch.testing.assert_close(result.polarization, expected, rtol=1e-13, atol=1e-24)
    with pytest.raises(ValueError, match='nonnegative'):
        instantaneous_harmonic_sources(e, tpa_scale=-1.)
