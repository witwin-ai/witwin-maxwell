"""Original leaf solves, native gradients and construction budget admission."""
import runpy
import pytest
import torch
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.cluster import build_support_octree
from witwin.maxwell.mom.block_preconditioner import LeafBlockInverse

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_leaf_inverse_original_blocks_adjoint_and_budget(monkeypatch):
    original, _ = runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    provider = ScalarSystemBlocks(original.system, original.exterior, original.integration)
    tree = build_support_octree(*provider.support_boxes(), leaf_size=32)
    inverse = LeafBlockInverse(provider, tree, max_block_edges=3)
    assert max(len(ids) for ids in inverse.indices) <= 6
    q = torch.randn((2, *inverse.shape[:1]), device='cuda', dtype=inverse.dtype, requires_grad=True)
    solved = inverse.apply(q)
    n = provider.system.total_edges
    for ids in inverse.indices:
        edges = ids[:len(ids)//2]
        block = provider(edges, edges)
        torch.testing.assert_close(solved[:, ids]@block.T, q[:, ids], rtol=2e-10, atol=1e-11)
    probe = torch.randn_like(q)
    grad, = torch.autograd.grad(solved, q, probe)
    torch.testing.assert_close(grad, inverse.adjoint(probe), rtol=2e-12, atol=1e-10)
    budget = inverse.retained_bytes
    def forbidden(*args, **kwargs):
        raise AssertionError('Capacity gate must precede original block integration')
    monkeypatch.setattr(ScalarSystemBlocks, '__call__', forbidden)
    with pytest.raises(MemoryError, match='retained bytes'):
        LeafBlockInverse(provider, tree, max_block_edges=3, retained_budget_bytes=budget-1)
    inverse.factors[0].add_(1)
    with pytest.raises(ValueError, match='changed'):
        inverse.apply(q)


def test_right_inverse_krylov_recovers_physical_current():
    from witwin.maxwell.mom.fmm_operator import MultipoleOperator
    from witwin.maxwell.mom.block_preconditioner import RightPreconditionedOperator
    from witwin.maxwell.mom.linalg import solve_reference, StoppingRule
    original, _ = runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    provider = ScalarSystemBlocks(original.system, original.exterior, original.integration)
    operator = MultipoleOperator(provider, order=2, leaf_size=32, separation_ratio=.25)
    inverse = LeafBlockInverse(provider, operator.tree, max_block_edges=3)
    view = RightPreconditionedOperator(operator, inverse)
    left, right, _ = view.preconditioner_scales('rwg_diagonal')
    torch.testing.assert_close(left*right, torch.ones_like(left), rtol=1e-14, atol=1e-14)
    q = torch.randn(operator.shape[0], device='cuda', dtype=operator.dtype)
    rhs = operator.apply(q)
    y, log = solve_reference(view, rhs, StoppingRule(method='gmres', preconditioner='rwg_diagonal',
        rtol=1e-12, atol=0., maxiter=48, restart=24, stagnation_window=49))
    assert log['converged'], log
    recovered = inverse.apply(y)
    torch.testing.assert_close(recovered, q, rtol=1e-8, atol=1e-9)
    assert float((operator.apply(recovered)-rhs).norm()/rhs.norm()) < 1e-12
    z = torch.randn_like(q)
    torch.testing.assert_close(torch.vdot(view.apply(y), z), torch.vdot(y, view.adjoint(z)), rtol=1e-11, atol=1e-11)


def test_primary_factors_exclude_ewald_correction_but_keep_original_binding(monkeypatch):
    from witwin.maxwell.mom.ewald_nearfield import EwaldSystemBlocks
    from witwin.maxwell.mom.periodic import LatticeSpec, EwaldHelmholtzKernel
    original, _ = runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    sample = original.system.bodies[0].mesh.vertices
    lattice = LatticeSpec(sample.new_tensor([[1., 0., 0.]]), sample.new_zeros(3))
    kernel = EwaldHelmholtzKernel(lattice, 3., reciprocal_shells=4)
    provider = EwaldSystemBlocks(original.system, original.exterior, original.integration, kernel)
    primary = ScalarSystemBlocks(original.system, original.exterior, original.integration)
    tree = build_support_octree(*provider.support_boxes(), leaf_size=32)
    def forbidden(*args, **kwargs):
        raise AssertionError('Primary preconditioner must not integrate Ewald image corrections')
    monkeypatch.setattr(EwaldSystemBlocks, 'regular_block', forbidden)
    inverse = LeafBlockInverse(provider, tree, max_block_edges=3, factor_kernel='primary')
    assert inverse.provider is provider and inverse.receipt()['factor_kernel'] == 'primary'
    q = torch.randn(2*provider.system.total_edges, device='cuda', dtype=inverse.dtype)
    solved = inverse.apply(q)
    for ids in inverse.indices:
        edges = ids[:len(ids)//2]
        torch.testing.assert_close(primary(edges, edges)@solved[ids], q[ids], rtol=2e-10, atol=1e-11)
    inverse.factor_kernel = 'original'
    with pytest.raises(ValueError, match='changed'):
        inverse.apply(q)
