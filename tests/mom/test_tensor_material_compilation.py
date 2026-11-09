"""Native public-material sampling and exact tensor sample identity."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.tensor_medium import TensorMedium

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_public_tensor_compiler_preserves_tensor_overrides_and_poles():
    import witwin.maxwell as mw
    from witwin.maxwell.constants import EPSILON_0
    from witwin.maxwell.mom.tensor_medium import compile_tensor_medium
    pole = mw.DebyePole(delta_eps=.7, tau=1e-10)
    rows = ((2., .2, 0.), (.2, 3., 0.), (0., 0., 4.))
    material = mw.Material(eps_r=17., mu_r=8., epsilon_tensor=mw.Tensor3x3(rows),
        mu_tensor=mw.DiagonalTensor3(1., 1.5, 2.), sigma_e=100.,
        sigma_e_tensor=mw.DiagonalTensor3(.01, .02, .03), debye_poles=(pole,), mu_debye_poles=(pole,))
    sample = compile_tensor_medium(material, 1e9, device='cuda')
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    expected = torch.tensor(rows, device='cuda', dtype=torch.complex128)+pole.susceptibility(sample.omega)*eye
    expected += 1j*torch.diag(torch.tensor([.01, .02, .03], device='cuda', dtype=torch.complex128))/(sample.omega*EPSILON_0)
    torch.testing.assert_close(sample.epsilon_r, expected, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(sample.mu_r, torch.diag(eye.new_tensor([1., 1.5, 2.]))+pole.susceptibility(sample.omega)*eye)


def test_public_ferrite_compiler_retains_full_bias_reversal():
    import witwin.maxwell as mw
    from witwin.maxwell.mom.tensor_medium import compile_tensor_medium
    samples = []
    for sign in (1, -1):
        material = mw.GyromagneticFerrite(eps_r=3., saturation_magnetization=1.4e5,
            bias_field=(0., 0., sign*8e4), gilbert_damping=.01, sigma_e=.01)
        sample = compile_tensor_medium(material, 2e9, device='cuda')
        assert sample.reciprocal is False
        expected = material.permeability_tensor_at_freq(torch.tensor(2e9, device='cuda', dtype=torch.float64))
        torch.testing.assert_close(sample.mu_r, expected)
        samples.append(sample)
    torch.testing.assert_close(samples[0].mu_r.mT, samples[1].mu_r)
    assert samples[0].validation['mu_r']['transpose_error'] > 0


def test_tensor_compiler_rejects_active_conductivity_and_cpu():
    import witwin.maxwell as mw
    from witwin.maxwell.mom.tensor_medium import compile_tensor_medium
    with pytest.raises(ValueError, match='conductivity'):
        compile_tensor_medium(mw.Material(sigma_e_tensor=mw.DiagonalTensor3(-.01, .02, .03)), 1e9, device='cuda')
    with pytest.raises(ValueError, match='CUDA'):
        compile_tensor_medium(mw.Material(), 1e9, device='cpu')


def test_tensor_compiler_does_not_validate_a_discarded_scalar_branch():
    import witwin.maxwell as mw
    from witwin.maxwell.mom.tensor_medium import compile_tensor_medium
    # The unused scalar base plus this pole is zero; the actual tensor is I.
    material = mw.Material(eps_r=1., epsilon_tensor=mw.DiagonalTensor3(2., 2., 2.),
                           drude_poles=(mw.DrudePole(plasma_frequency=1e9, gamma=0.),))
    sample = compile_tensor_medium(material, 1e9, device='cuda')
    torch.testing.assert_close(sample.epsilon_r, torch.eye(3, device='cuda', dtype=torch.complex128))


def test_public_polder_batch_stays_on_cuda_with_frequency_gradients():
    import witwin.maxwell as mw
    material = mw.GyromagneticFerrite(saturation_magnetization=1.4e5,
        bias_field=(2e4, 3e4, 8e4), gilbert_damping=.01)
    frequencies = torch.tensor([1.9e9, 2e9], device='cuda', dtype=torch.float64, requires_grad=True)
    sampled = material.permeability_tensor_at_freq(frequencies)
    assert sampled.device == frequencies.device and sampled.shape == (2, 3, 3)
    for i in range(2):
        torch.testing.assert_close(sampled[i], material.permeability_tensor_at_freq(frequencies[i]))
    gradient = torch.autograd.grad(sampled.abs().square().sum(), frequencies)[0]
    assert torch.isfinite(gradient).all() and bool((gradient != 0).all())


@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_tensor_sample_payload_roundtrip_is_exact_and_revalidates(dtype):
    import copy
    import json
    from witwin.maxwell.mom.tensor_medium import sampled_medium_payload, restore_tensor_medium, sampled_tensor_identity
    eye = torch.eye(3, device='cuda', dtype=dtype)
    epsilon = (2+.2j)*eye
    epsilon[0, 1], epsilon[1, 0] = .1+.3j, .1-.3j
    medium = TensorMedium(3e8, epsilon, (1+.1j)*eye, reciprocal=False)
    payload = json.loads(json.dumps(sampled_medium_payload(medium)))
    restored = restore_tensor_medium(payload, device='cuda')
    assert restored.epsilon_r.dtype == dtype
    torch.testing.assert_close(restored.epsilon_r, medium.epsilon_r, rtol=0, atol=0)
    torch.testing.assert_close(restored.mu_r, medium.mu_r, rtol=0, atol=0)
    assert sampled_tensor_identity(restored) == sampled_tensor_identity(medium)
    changed = copy.deepcopy(payload)
    changed['epsilon_r_real_imag'][0][0][1] = -1.
    with pytest.raises(ValueError, match='passive'):
        restore_tensor_medium(changed, device='cuda')
    changed = copy.deepcopy(payload)
    changed['reciprocal'] = True
    with pytest.raises(ValueError, match='reciprocal'):
        restore_tensor_medium(changed, device='cuda')
    changed = copy.deepcopy(payload)
    changed['schema_version'] = True
    with pytest.raises(ValueError, match='schema'):
        restore_tensor_medium(changed, device='cuda')
    changed = copy.deepcopy(payload)
    changed['epsilon_r_real_imag'][0][0][0] = True
    with pytest.raises(ValueError, match='numeric'):
        restore_tensor_medium(changed, device='cuda')
    with pytest.raises(ValueError, match='CUDA'):
        restore_tensor_medium(payload, device='cpu')




def test_complex_public_declarations_preserve_both_tensors_and_exact_storage():
    from witwin.maxwell.mom.tensor_medium import compile_tensor_medium, sampled_medium_payload, restore_tensor_medium
    epsilon = ((2+.3j, .1+.02j, 0), (.1+.02j, 3+.4j, 0), (0, 0, 4+.5j))
    mu = ((1.5, .1j, 0), (-.1j, 2., 0), (0, 0, 2.5))
    material = mw.Material(epsilon_tensor=mw.ComplexTensor3x3(epsilon),
                          mu_tensor=mw.ComplexTensor3x3(mu, reciprocal=False))
    sample = compile_tensor_medium(material, 3e8, device='cuda')
    assert not sample.reciprocal
    torch.testing.assert_close(sample.epsilon_r, torch.tensor(epsilon, device='cuda', dtype=torch.complex128), rtol=0, atol=0)
    torch.testing.assert_close(sample.mu_r, torch.tensor(mu, device='cuda', dtype=torch.complex128), rtol=0, atol=0)
    restored = restore_tensor_medium(sampled_medium_payload(sample), device='cuda')
    torch.testing.assert_close(restored.epsilon_r, sample.epsilon_r, rtol=0, atol=0)
    torch.testing.assert_close(restored.mu_r, sample.mu_r, rtol=0, atol=0)
    real_mu = mw.Tensor3x3(((2, .1, 0), (.1, 3, 0), (0, 0, 4)))
    full_real = compile_tensor_medium(mw.Material(mu_tensor=real_mu), 3e8, device='cuda')
    torch.testing.assert_close(full_real.mu_r, torch.tensor(real_mu.rows, device='cuda', dtype=torch.complex128), rtol=0, atol=0)


def test_each_complex_declaration_reciprocity_and_passivity_is_checked():
    from witwin.maxwell.mom.tensor_medium import compile_tensor_medium
    gyro = ((2, .1j, 0), (-.1j, 3, 0), (0, 0, 4))
    # A nonreciprocal mu must not disable an epsilon declaration's own assertion.
    material = mw.Material(epsilon_tensor=mw.ComplexTensor3x3(gyro),
        mu_tensor=mw.ComplexTensor3x3(gyro, reciprocal=False))
    with pytest.raises(ValueError, match='reciprocal'):
        compile_tensor_medium(material, 3e8, device='cuda')
    # Positive diagonal imaginary parts do not suffice: the loss matrix is indefinite.
    gain = ((2+.1j, .2j, 0), (.2j, 3+.1j, 0), (0, 0, 4+.1j))
    with pytest.raises(ValueError, match='passive'):
        compile_tensor_medium(mw.Material(mu_tensor=mw.ComplexTensor3x3(gain)), 3e8, device='cuda')
