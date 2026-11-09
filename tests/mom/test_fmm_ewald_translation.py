"""Lossless Ewald M2L removes exact-near images and preserves electromagnetic adjoints."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.periodic import LatticeSpec, EwaldHelmholtzKernel
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.fmm_ewald_translation import EwaldRootM2L
from witwin.maxwell.mom.fmm_expansion import monomials
from witwin.maxwell.mom.kernels import FREE_SPACE

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('dimension,near_radius', [(1, .02), (2, .02), (3, .02), (1, .2)])
def test_ewald_root_m2l_near_subtraction_and_fixed_current_adjoint(dimension, near_radius):
    vectors = torch.tensor([[1., .2, .1], [.1, 1.2, .3], [.2, .1, .9]], device='cuda', dtype=torch.float64)[:dimension]
    lattice = LatticeSpec(vectors, .17*vectors.sum(0))
    kernel = EwaldHelmholtzKernel(lattice, 2., real_shells=3, reciprocal_shells=2, quadrature_order=64, image_batch=13)
    medium = compile_medium(mw.Material(), 9e7)
    op = EwaldRootM2L(kernel, medium, near_radius, near_radius, order=6, row_batch=7)
    x = vectors.new_tensor([[.003, -.002, .004], [-.004, .002, -.001]])
    y = vectors.new_tensor([[-.002, .001, .003], [.001, -.003, -.002]])
    green, gradient = op._tables(0, op.count)
    tx, sy = monomials(x, 6).to(op.dtype), monomials(y, 6).to(op.dtype)
    actual = tx@green@sy.mT
    actual_gradient = torch.einsum('ib,bca,jc->ija', tx, gradient, sy)
    delta = x[:, None]-y[None]
    reference, reference_gradient = kernel.regular_part(delta, medium.k)
    for index in op.near_indices[(op.near_indices != 0).any(-1)]:
        g, dg = FREE_SPACE.evaluate(delta-lattice.translations(index), medium.k)
        reference -= lattice.phase(index)*g
        reference_gradient -= lattice.phase(index)*dg
    torch.testing.assert_close(actual, reference, rtol=3e-9, atol=3e-12)
    torch.testing.assert_close(actual_gradient, reference_gradient, rtol=3e-9, atol=3e-12)
    assert len(op.near_indices) > 1 if near_radius == .2 else len(op.near_indices) == 1
    u = torch.randn((2, 2, op.count, 4), device='cuda', dtype=op.dtype, requires_grad=True)
    v = torch.randn_like(u)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        applied = op.apply(u)
    assert not saved
    derivative, = torch.autograd.grad(applied, u, v)
    torch.testing.assert_close(derivative, op.adjoint(v), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close((applied.conj()*v).sum(), (u.conj()*op.adjoint(v)).sum(), rtol=3e-12, atol=1e-10)
    with pytest.raises(MemoryError, match='before expansions'):
        EwaldRootM2L(kernel, medium, near_radius, near_radius, order=6, retained_budget_bytes=1)
    with pytest.raises(ValueError, match='every possible near image'):
        EwaldRootM2L(kernel, medium, 10., 10.)
    op.near_indices.add_(1)
    with pytest.raises(ValueError, match='changed'):
        op.apply(u)
