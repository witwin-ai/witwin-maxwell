"""Taylor reciprocal primitives match the configured lossless Ewald spectral term."""
import pytest
import torch
from witwin.maxwell.mom.periodic import LatticeSpec, EwaldHelmholtzKernel
from witwin.maxwell.mom.fmm_expansion import multi_indices, monomials
from witwin.maxwell.mom.fmm_ewald_spectral import ewald_spectral_taylor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('dimension', [1, 2, 3])
@pytest.mark.parametrize('wave', [2., -2.])
def test_spectral_taylor_matches_value_gradient_and_outgoing_contour(dimension, wave):
    vectors = torch.tensor([[1., .2, .1], [.1, 1.2, .3], [.2, .1, .9]], device='cuda', dtype=torch.float64)[:dimension]
    lattice = LatticeSpec(vectors, .17*vectors.sum(0))
    kernel = EwaldHelmholtzKernel(lattice, 2., real_shells=2, reciprocal_shells=2, quadrature_order=32, image_batch=13)
    center = vectors.new_tensor([.13, .19, .23])
    order = 8
    coefficients = ewald_spectral_taylor(kernel, center, wave, order)
    offsets = vectors.new_tensor([[.003, -.002, .004], [-.004, .002, -.001], [0., 0., 0.]])
    actual = monomials(offsets, order).to(coefficients.dtype)@coefficients
    lookup = {a: i for i, a in enumerate(multi_indices(order))}
    gradient = []
    for axis in range(3):
        derivative = []
        for alpha in multi_indices(order-1):
            shifted = list(alpha)
            shifted[axis] += 1
            derivative.append(shifted[axis]*coefficients[lookup[tuple(shifted)]])
        gradient.append(monomials(offsets, order-1).to(coefficients.dtype)@torch.stack(derivative))
    gradient = torch.stack(gradient, -1)
    reference, reference_gradient = kernel._spectral(center[None]+offsets, complex(wave))
    torch.testing.assert_close(actual, reference, rtol=2e-10, atol=2e-12)
    torch.testing.assert_close(gradient, reference_gradient, rtol=2e-10, atol=2e-12)
    with pytest.raises(MemoryError, match='one reciprocal mode'):
        ewald_spectral_taylor(kernel, center, wave, order, coefficient_budget_bytes=coefficients.numel()*16)


def test_spectral_taylor_single_mode_budget_matches_batched_coefficients():
    vectors = torch.eye(3, device='cuda', dtype=torch.float64)
    lattice = LatticeSpec(vectors, vectors.new_tensor([.17, .23, .11]))
    kernel = EwaldHelmholtzKernel(lattice, 2., reciprocal_shells=2, image_batch=13)
    center = vectors.new_tensor([.13, .19, .23])
    order = 6
    reference = ewald_spectral_taylor(kernel, center, 2., order)
    actual = ewald_spectral_taylor(kernel, center, 2., order,
        coefficient_budget_bytes=2*reference.numel()*reference.element_size())
    torch.testing.assert_close(actual, reference, rtol=3e-12, atol=1e-12)
