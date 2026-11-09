"""Real Ewald Taylor coefficients match independent value and gradient queries."""
import pytest
import torch
from witwin.maxwell.mom.periodic import LatticeSpec, EwaldHelmholtzKernel
from witwin.maxwell.mom.fmm_expansion import multi_indices, monomials
from witwin.maxwell.mom.fmm_ewald_real import ewald_real_taylor, ewald_primary_regular_taylor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def evaluate(coefficients, offsets, order):
    value = monomials(offsets, order).to(coefficients.dtype)@coefficients
    lookup = {a: i for i, a in enumerate(multi_indices(order))}
    gradient = []
    for axis in range(3):
        derivative = []
        for alpha in multi_indices(order-1):
            shifted = list(alpha)
            shifted[axis] += 1
            derivative.append(shifted[axis]*coefficients[lookup[tuple(shifted)]])
        gradient.append(monomials(offsets, order-1).to(coefficients.dtype)@torch.stack(derivative))
    return value, torch.stack(gradient, -1)


@pytest.mark.parametrize('wave', [2., -2., 2.+.7j])
def test_screened_radial_recurrence_and_primary_entire_series(wave):
    lattice = LatticeSpec(torch.eye(3, device='cuda', dtype=torch.float64),
                          torch.tensor([.17, .23, .11], device='cuda', dtype=torch.float64))
    kernel = EwaldHelmholtzKernel(lattice, 2., reciprocal_shells=2, quadrature_order=64)
    centers = lattice.vectors.new_tensor([[.13, .19, .23], [1.1, -.2, .3]])
    offsets = centers.new_tensor([[.003, -.002, .004], [-.004, .002, -.001], [0., 0., 0.]])
    order = 8
    coefficients = ewald_real_taylor(kernel, centers, wave, order)
    for center, coeff in zip(centers, coefficients):
        actual = evaluate(coeff, offsets, order)
        reference = kernel._real(center[None]+offsets, complex(wave), images='zero')
        for a, b in zip(actual, reference):
            torch.testing.assert_close(a, b, rtol=2e-9, atol=2e-12)
    primary = ewald_primary_regular_taylor(kernel, wave, order)
    actual = evaluate(primary, offsets, order)
    reference = kernel._regular_zero(offsets, complex(wave))
    for a, b in zip(actual, reference):
        torch.testing.assert_close(a, b, rtol=2e-10, atol=2e-12)
    with pytest.raises(MemoryError, match='before radial seeds'):
        ewald_real_taylor(kernel, centers, wave, order, coefficient_budget_bytes=1)
    with pytest.raises(ValueError, match='nonzero radius'):
        ewald_real_taylor(kernel, centers.new_zeros(3), wave, order)


@pytest.mark.parametrize('dimension', [1, 2, 3])
def test_complete_regular_origin_expansion_preserves_bloch_and_primary_split(dimension):
    from witwin.maxwell.mom.fmm_ewald_real import ewald_regular_taylor
    vectors = torch.tensor([[1., .2, .1], [.1, 1.2, .3], [.2, .1, .9]], device='cuda', dtype=torch.float64)[:dimension]
    lattice = LatticeSpec(vectors, .17*vectors.sum(0))
    kernel = EwaldHelmholtzKernel(lattice, 2., real_shells=2, reciprocal_shells=2,
        quadrature_order=64, image_batch=13)
    offsets = vectors.new_tensor([[.003, -.002, .004], [-.004, .002, -.001], [0., 0., 0.]])
    coefficients = ewald_regular_taylor(kernel, 2., 8)
    actual = evaluate(coefficients, offsets, 8)
    reference = kernel.regular_part(offsets, 2.)
    for a, b in zip(actual, reference):
        torch.testing.assert_close(a, b, rtol=3e-9, atol=3e-12)
    assert torch.isfinite(coefficients).all()
