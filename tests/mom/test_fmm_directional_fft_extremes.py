"""Extreme scaling fail-closed and physical carrier adjoint/reference controls."""
import json
import pytest
import torch
from test_fmm_directional_fft import _dense_tables
from witwin.maxwell.mom.fmm_directional_fft import CartesianDirectionalFFTTranslation, directional_helmholtz_taylor
from witwin.maxwell.mom.fmm_expansion import helmholtz_taylor, multipole_moments, monomials
from witwin.maxwell.mom.fmm_fft_coefficients import CartesianCoefficientFFTTranslation

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_directional_unrepresentable_intermediate_physical_powers_reject_before_expansions(monkeypatch):
    import witwin.maxwell.mom.fmm_directional_fft as module
    def forbidden(*args, **kwargs): raise AssertionError('Rejected physical scaling reached radial expansions')
    monkeypatch.setattr(module, '_radial_constants', forbidden)
    direction = torch.tensor([1., 0., 0.], device='cuda', dtype=torch.float64)
    for radius, scale in ((1e13, 1.), (1e-13, 1e-15)):
        center = direction*radius
        with pytest.raises(ValueError, match='physical scale powers'):
            directional_helmholtz_taylor(center, direction, 1., 25)
        with pytest.raises(ValueError, match='physical scale powers'):
            CartesianDirectionalFFTTranslation(center, direction, 1., center.new_tensor(scale), 12)


def test_actual_lossy_carrier_full_current_adjoint_and_unmodulated_dense_reference(tmp_path):
    order = 6; wave = 100.+.02j
    center = torch.tensor([2000., 400., -200.], device='cuda', dtype=torch.float64)
    direction = center/center.norm()+center.new_tensor([.0001, -.0002, .0003])
    direction = direction/direction.norm()
    generator = torch.Generator(device='cuda').manual_seed(956)
    source = torch.randn((7, 3), device='cuda', dtype=torch.float64, generator=generator)
    target = torch.randn((9, 3), device='cuda', dtype=torch.float64, generator=generator)
    source = source/source.norm(dim=-1, keepdim=True)*.5
    target = target/target.norm(dim=-1, keepdim=True)*.5
    source_phase = torch.exp(-1j*wave*(source@direction))
    target_phase = torch.exp(1j*wave*(target@direction))
    source_projection = monomials(source, order).to(torch.complex128)*source_phase[:, None]
    receiver = monomials(target, order).to(torch.complex128)
    op = CartesianDirectionalFFTTranslation(center, direction, wave, center.new_tensor(.5), order)
    charges = torch.randn((5, 3, 7), device='cuda', dtype=torch.complex128, generator=generator).transpose(0, 1).requires_grad_()
    moments = charges@source_projection
    locals = op.apply(moments)
    actual = torch.einsum('tb,...fb->...tf', receiver, locals)*target_phase[:, None]
    dual = torch.randn(actual.shape, device='cuda', dtype=actual.dtype, generator=generator)
    local_dual = torch.einsum('tb,...tf->...fb', receiver.conj(), dual*target_phase.conj()[:, None])
    moment_dual = op.adjoint(local_dual)
    adjoint = moment_dual@source_projection.conj().mT
    gradient, = torch.autograd.grad(actual, charges, dual)
    gradient_error = float((gradient-adjoint).norm()/adjoint.norm())
    lhs, rhs = (actual.conj()*dual).sum(), (charges.conj()*adjoint).sum()
    dot = float(((lhs-rhs).abs()/torch.maximum(lhs.abs(), rhs.abs())).detach())
    delta = center+target[:, None]-source[None]
    radius = delta.norm(dim=-1)
    green = torch.exp(1j*wave*radius)/(4*torch.pi*radius)
    physical_gradient = green[..., None]*(1j*wave-1/radius)[..., None]*delta/radius[..., None]
    exact = torch.cat((green[..., None], physical_gradient), dim=-1)
    expected = torch.einsum('tsf,...s->...tf', exact, charges)
    expected_adjoint = torch.einsum('tsf,...tf->...s', exact.conj(), dual)
    physical_forward = float(((actual-expected).norm()/expected.norm()).detach())
    physical_adjoint = float((adjoint-expected_adjoint).norm()/expected_adjoint.norm())
    original = helmholtz_taylor(center, wave, 2*order+1)
    # This dense original polynomial comparator has no directional carrier.
    tables = _dense_tables(original, order, direction, 0.)
    identity = torch.eye(len(source), device='cuda', dtype=torch.complex128)
    original_moments = multipole_moments(source, identity, order)
    dense_locals = torch.einsum('fba,sa->sfb', tables, original_moments)
    dense_unmodulated = torch.einsum('tb,sfb->tsf', receiver, dense_locals)
    original_fft = CartesianCoefficientFFTTranslation(original, center, center.new_tensor(1/abs(wave)), order)
    fft_unmodulated = torch.einsum('tb,sfb->tsf', receiver, original_fft.apply(original_moments))
    fft_dense_error = float((fft_unmodulated-dense_unmodulated).norm()/dense_unmodulated.norm())
    dense_exact_error = float((dense_unmodulated-exact).norm()/exact.norm())
    record = dict(physical_forward=physical_forward, physical_adjoint=physical_adjoint,
        gradient=gradient_error, dot=dot, original_fft_vs_independent_dense=fft_dense_error,
        original_same_order_dense_vs_exact=dense_exact_error, source_and_receiver_phases_conjugated=True,
        wave=str(wave), electrical_radius=abs(wave)*.5, physical_qualification=False,
        full_directional_fmm=False)
    (tmp_path/'carrier.json').write_text(json.dumps(record, indent=2)+'\n'); print(record, flush=True)
    assert physical_forward < 1e-8 and physical_adjoint < 1e-8
    assert gradient_error < 1e-11 and dot < 1e-11 and fft_dense_error < 1e-11
