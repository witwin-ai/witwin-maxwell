"""Streamed original Ewald actions preserve selected rows and current gradients."""
import runpy
import pytest
import torch
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.ewald_nearfield import EwaldSystemBlocks
from witwin.maxwell.mom.periodic import LatticeSpec, EwaldHelmholtzKernel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_sampled_ewald_face_contraction_matches_blocks_and_gradient(monkeypatch):
    original, _ = runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    sample = original.system.bodies[0].mesh.vertices
    lattice = LatticeSpec(sample.new_tensor([[1., 0., 0.]]), sample.new_tensor([.17, 0., 0.]))
    kernel = EwaldHelmholtzKernel(lattice, 3., reciprocal_shells=2, real_shells=2,
                                 quadrature_order=16)
    provider = EwaldSystemBlocks(original.system, original.exterior, original.integration, kernel)
    n = provider.system.total_edges
    rows = torch.tensor([n-1, 0, n//2], device='cuda')
    columns = torch.arange(n, device='cuda')
    block = provider(rows, columns)
    q = torch.randn((2, 3, 2*n), device='cuda', dtype=block.dtype, requires_grad=True)
    expected = q@block.T
    # The optimized action must not assemble selected smooth edge blocks.
    def forbid(*args, **kwargs):
        raise AssertionError('Smooth edge blocks must be contracted by source face')
    monkeypatch.setattr(provider, 'regular_block', forbid)
    actual = provider.sampled_action(rows, q, source_batch=3)
    torch.testing.assert_close(actual, expected, rtol=3e-11, atol=1e-13)
    probe = torch.randn_like(actual)
    grad, = torch.autograd.grad(actual, q, probe)
    torch.testing.assert_close(grad, probe@block.conj(), rtol=3e-11, atol=1e-13)
    with pytest.raises(ValueError, match='positive integer'):
        provider.sampled_action(rows, q, source_batch=0)
    with pytest.raises(ValueError, match='unique'):
        provider.sampled_action(rows[[0, 0]], q)
