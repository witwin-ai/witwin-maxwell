"""True polynomial multipole/local translations against free-space kernels."""
import pytest
import torch
from witwin.maxwell.mom.fmm_expansion import (multi_indices, monomials,
    multipole_moments, shift_matrix, helmholtz_taylor, multipole_to_local)
from witwin.maxwell.mom.kernels import FREE_SPACE

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_multipole_and_local_translations_preserve_polynomials():
    generator = torch.Generator(device='cuda').manual_seed(917)
    points = torch.randn((13, 3), device='cuda', dtype=torch.float64, generator=generator)*.02
    charges = torch.randn((2, 13), device='cuda', dtype=torch.complex128, generator=generator)
    first = torch.tensor([.01, -.02, .015], device='cuda', dtype=torch.float64)
    second = first.flip(0)*.3
    order = 5
    s1, s2 = shift_matrix(first, order).to(charges.dtype), shift_matrix(second, order).to(charges.dtype)
    moments = multipole_moments(points, charges, order)
    torch.testing.assert_close(moments@s1.mT@s2.mT,
        multipole_moments(points+first+second, charges, order), rtol=2e-12, atol=1e-14)
    local = torch.randn((2, len(multi_indices(order))), device='cuda', dtype=charges.dtype, generator=generator)
    child_local = local@s2@s1
    torch.testing.assert_close(child_local@monomials(points, order).to(charges.dtype).mT,
        local@monomials(points+first+second, order).to(charges.dtype).mT, rtol=2e-12, atol=1e-14)


@pytest.mark.parametrize('wave', [0., 6.3, 6.3+.4j])
def test_helmholtz_m2l_convergence_gradient_and_exact_adjoint(wave):
    generator = torch.Generator(device='cuda').manual_seed(811)
    target = torch.randn((8, 3), device='cuda', dtype=torch.float64, generator=generator)*.01
    source = torch.randn((11, 3), device='cuda', dtype=torch.float64, generator=generator)*.01
    displacement = torch.tensor([1., .3, -.2], device='cuda', dtype=torch.float64)
    reference, reference_gradient = FREE_SPACE.evaluate(displacement+target[:, None]-source[None], wave)
    errors = []
    for order in (2, 4, 6):
        green, gradient = multipole_to_local(displacement, wave, order)
        bt, bs = (monomials(points, order).to(green.dtype) for points in (target, source))
        actual = bt@green@bs.mT
        actual_gradient = torch.einsum('ib,abc,jc->ija', bt, gradient, bs)
        error = float(torch.linalg.vector_norm(actual-reference)/torch.linalg.vector_norm(reference))
        grad_error = float(torch.linalg.vector_norm(actual_gradient-reference_gradient)/torch.linalg.vector_norm(reference_gradient))
        errors.append(max(error, grad_error))
        print('M2L wave/order/errors', wave, order, error, grad_error)
    assert errors[1] < errors[0] and errors[2] < errors[1]
    assert errors[-1] < 1e-8
    charges = torch.randn((2, 11), device='cuda', dtype=green.dtype, generator=generator, requires_grad=True)
    moments = multipole_moments(source, charges, order)
    potential = (moments@green.mT)@bt.mT
    probe = torch.randn_like(potential)
    derivative, = torch.autograd.grad(potential, charges, probe)
    torch.testing.assert_close(derivative, probe@actual.conj(), rtol=2e-12, atol=1e-14)
    jet = helmholtz_taylor(displacement, wave, 2)
    lookup = {alpha: i for i, alpha in enumerate(multi_indices(2))}
    laplacian = 2*sum(jet[lookup[alpha]] for alpha in ((2, 0, 0), (0, 2, 0), (0, 0, 2)))
    torch.testing.assert_close(laplacian, -wave**2*jet[0], rtol=1e-12, atol=1e-14)


def test_m2l_admission_rejects_before_coefficients(monkeypatch):
    import witwin.maxwell.mom.fmm_expansion as expansion
    displacement = torch.tensor([1., 0., 0.], device='cuda', dtype=torch.float64)
    def forbidden(*args, **kwargs):
        raise AssertionError('Table budget must reject before coefficient generation')
    monkeypatch.setattr(expansion, 'helmholtz_taylor', forbidden)
    with pytest.raises(MemoryError, match='before coefficient'):
        multipole_to_local(displacement, 1., 6, table_budget_bytes=1)


def test_expansion_rejects_trainable_geometry_wave_and_coincident_centers():
    displacement = torch.tensor([1., 0., 0.], device='cuda', dtype=torch.float64)
    with pytest.raises(ValueError, match='fixed scalar'):
        helmholtz_taylor(displacement, torch.tensor(1., device='cuda', requires_grad=True), 2)
    with pytest.raises(ValueError, match='finite positive distance'):
        helmholtz_taylor(displacement*0, 1., 2)
    with pytest.raises(ValueError, match='passive'):
        helmholtz_taylor(displacement, 1.-.1j, 2)
    with pytest.raises(ValueError, match='fixed real CUDA'):
        multipole_to_local(displacement.requires_grad_(), 1., 2)


@pytest.mark.parametrize('wave', [0., 6.3+.4j])
def test_batched_m2l_matches_individual_tables_and_kernel(wave, monkeypatch):
    displacement = torch.tensor([[[1., .3, -.2], [1.2, -.2, .1]],
        [[-.8, .5, .2], [.9, .2, .4]]], device='cuda', dtype=torch.float64)
    order = 4
    green, gradient = multipole_to_local(displacement, wave, order)
    assert green.shape == (2, 2, 35, 35)
    assert gradient.shape == (2, 2, 3, 35, 35)
    for i in range(2):
        for j in range(2):
            g, dg = multipole_to_local(displacement[i, j], wave, order)
            torch.testing.assert_close(green[i, j], g, rtol=3e-13, atol=1e-14)
            torch.testing.assert_close(gradient[i, j], dg, rtol=3e-13, atol=1e-14)
    reference, derivative = FREE_SPACE.evaluate(displacement, wave)
    torch.testing.assert_close(green[..., 0, 0], reference, rtol=1e-13, atol=1e-14)
    torch.testing.assert_close(gradient[..., :, 0, 0], derivative, rtol=1e-13, atol=1e-14)
    import witwin.maxwell.mom.fmm_expansion as expansion
    def forbidden(*args, **kwargs):
        raise AssertionError('Batch capacity must reject before Taylor construction')
    monkeypatch.setattr(expansion, 'helmholtz_taylor', forbidden)
    with pytest.raises(MemoryError):
        multipole_to_local(displacement, wave, order, table_budget_bytes=green.numel()*16*4-1)
