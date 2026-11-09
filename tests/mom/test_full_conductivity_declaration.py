"""Full real conductivity declarations preserve AC/DC units and backend limits."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.constants import EPSILON_0
from witwin.maxwell.mom.static_tensor_material import compile_static_tensor_material
from witwin.maxwell.mom.tensor_medium import (
    compile_tensor_medium, sampled_medium_payload, restore_tensor_medium,
)


ROWS = ((.002, .001, 0.), (.001, .002, 0.), (0., 0., 0.))
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('rows,message', (
    (((.002, .001, 0.), (0., .002, 0.), (0., 0., 0.)), 'symmetric'),
    (((float('nan'), 0., 0.), (0., .002, 0.), (0., 0., 0.)), 'finite'),
))
def test_full_conductivity_descriptor_rejects_invalid_structure(rows, message):
    with pytest.raises(ValueError, match=message):
        mw.Material(sigma_e_tensor=mw.Tensor3x3(rows))
    with pytest.raises(TypeError, match='real tensor descriptor'):
        mw.Material(sigma_e_tensor=ROWS)


@pytest.mark.parametrize('epsilon', (None, mw.Tensor3x3(((2., .1, 0.), (.1, 3., 0.), (0., 0., 4.)))))
def test_reference_export_does_not_discard_full_conductivity(epsilon):
    from witwin.maxwell.adapters.tidy3d import _convert_material
    material = mw.Material(epsilon_tensor=epsilon, sigma_e_tensor=mw.Tensor3x3(ROWS))
    with pytest.raises(NotImplementedError, match='full sigma_e_tensor'):
        _convert_material(material, object())


def test_adjoint_conductivity_detection_handles_full_declarations():
    from witwin.maxwell.fdtd.adjoint.bridge import _material_has_conductivity
    assert _material_has_conductivity(mw.Material(sigma_e_tensor=mw.Tensor3x3(ROWS)))
    assert not _material_has_conductivity(mw.Material(sigma_e_tensor=mw.Tensor3x3(((0.,)*3,)*3)))


@cuda
@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_full_conductivity_ac_dc_power_and_exact_sample_replay(dtype):
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    # Rank-deficient PSD conductivity is legal. The tensor replaces the unused
    # scalar conductivity, and the off-diagonal SI entries remain intact.
    material = mw.Material(eps_r=2., sigma_e=50., sigma_e_tensor=mw.Tensor3x3(ROWS))
    sigma = torch.tensor(ROWS, device='cuda', dtype=dtype)
    dc = compile_static_tensor_material(material, device='cuda', dtype=dtype).electric_dc
    assert dc.kind == 'conductive_limit'
    torch.testing.assert_close(dc.conductivity_sample(), sigma, rtol=0, atol=0)
    with pytest.raises(NotImplementedError, match='conductive_limit'):
        dc.dielectric_sample()
    tol = 32 * torch.finfo(dtype).eps
    samples = []
    for frequency in (3e8, 6e8):
        sample = compile_tensor_medium(material, frequency, device='cuda', dtype=complex_dtype)
        assert sample.reciprocal and sample.epsilon_r.device.type == 'cuda'
        torch.testing.assert_close(sample.epsilon_r.real, 2 * torch.eye(3, device='cuda', dtype=dtype), rtol=0, atol=0)
        torch.testing.assert_close(sample.epsilon_r.imag * (sample.omega * EPSILON_0), sigma, rtol=tol, atol=0)
        restored = restore_tensor_medium(sampled_medium_payload(sample), device='cuda')
        torch.testing.assert_close(restored.epsilon_r, sample.epsilon_r, rtol=0, atol=0)
        samples.append(sample)
    torch.testing.assert_close(samples[0].epsilon_r.imag, 2 * samples[1].epsilon_r.imag, rtol=0, atol=0)
    field = torch.tensor([[1.+.2j, -.3+.4j, .7j]], device='cuda', dtype=complex_dtype, requires_grad=True)
    power = samples[0].dissipated_power_density(field, torch.zeros_like(field))
    expected = .5 * torch.einsum('...i,ij,...j->...', field.conj(), sigma.to(complex_dtype), field).real
    torch.testing.assert_close(power, expected, rtol=tol, atol=0)
    gradient = torch.autograd.grad(power.sum(), field)[0]
    torch.testing.assert_close(gradient, field.detach() @ sigma.to(complex_dtype), rtol=tol, atol=0)
    dc.conductivity.add_(.001)
    with pytest.raises(ValueError, match='changed'):
        dc.conductivity_sample()


@cuda
def test_full_conductivity_harmonics_preserve_poles_nonlinearity_and_identity():
    from witwin.maxwell.mom.harmonic_material import compile_tensor_harmonic_material, material_signature
    pole = mw.DrudePole(plasma_frequency=1e8, gamma=2e7)
    def declaration(rows):
        return mw.Material(eps_r=2., sigma_e=50., sigma_e_tensor=mw.Tensor3x3(rows),
            drude_poles=(pole,), nonlinearity=(mw.NonlinearSusceptibility(chi2=.002, chi3=.001),
            mw.TwoPhotonAbsorption(beta=1e-12, n0=1.5)))
    material = declaration(ROWS)
    compiled = compile_tensor_harmonic_material(material, 3e8, 2, device='cuda')
    sigma = torch.tensor(ROWS, device='cuda', dtype=torch.float64)
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    expected_dc = sigma + EPSILON_0 * pole.plasma_angular_frequency**2 / pole.gamma_angular_frequency * eye.real
    torch.testing.assert_close(compiled.electric_dc.conductivity_sample(), expected_dc, rtol=0, atol=0)
    for sample in compiled.linear_samples:
        expected = (2. + pole.susceptibility(sample.omega)) * eye + 1j * sigma / (sample.omega * EPSILON_0)
        torch.testing.assert_close(sample.epsilon_r, expected, rtol=0, atol=0)
    assert compiled.chi2 == .002 and compiled.chi3 == .001 and compiled.tpa_scale > 0
    # A signed permutation tests frame covariance without rounding a singular
    # eigenvalue negative or projecting the declared material onto a new PSD law.
    rotated_rows = ((.002, -.001, 0.), (-.001, .002, 0.), (0., 0., 0.))
    rotation = eye.real.new_tensor(((0., -1., 0.), (1., 0., 0.), (0., 0., 1.)))
    rotated = compile_tensor_harmonic_material(declaration(rotated_rows), 3e8, 2, device='cuda')
    torch.testing.assert_close(rotated.electric_dc.conductivity, rotation @ expected_dc @ rotation.T, rtol=0, atol=0)
    for left, right in zip(rotated.linear_samples, compiled.linear_samples):
        torch.testing.assert_close(left.epsilon_r, rotation.to(eye.dtype) @ right.epsilon_r @ rotation.T.to(eye.dtype), rtol=0, atol=0)
    assert material_signature(material) != material_signature(declaration(rotated_rows))


@cuda
@pytest.mark.parametrize('poles', ((), (mw.DrudePole(plasma_frequency=1e8, gamma=2e7),)))
def test_native_compilers_reject_indefinite_full_conductivity(poles):
    # The passive Drude increment exceeds the negative eigenvalue at DC; it
    # must not hide an invalid declared background from the static compiler.
    material = mw.Material(sigma_e_tensor=mw.Tensor3x3(((.001, .002, 0.), (.002, .001, 0.), (0., 0., 0.))),
                           drude_poles=poles)
    with pytest.raises(ValueError, match='conductivity'):
        compile_tensor_medium(material, 3e8, device='cuda')
    with pytest.raises(ValueError, match='conductivity'):
        compile_static_tensor_material(material, device='cuda')
