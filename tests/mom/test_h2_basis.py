"""True multilevel nesting, ancestor spaces and common up/down adjoints."""
import pytest
import torch
from witwin.maxwell.mom.cluster import build_support_octree
from witwin.maxwell.mom.h2_basis import NestedClusterBasis, NestedBasisConfig, NestedBasisFailure

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def fixture():
    g = torch.Generator(device='cuda').manual_seed(313)
    points = torch.rand((64, 3), device='cuda', dtype=torch.float64, generator=g)
    tree = build_support_octree(points-.001, points+.001, leaf_size=4)
    internal = next(i for i in range(1, tree.node_count) if bool((tree.children[i] >= 0).any()))
    root = torch.randn((128, 2), device='cuda', dtype=torch.complex128, generator=g)
    local = torch.randn((2*int(tree.stops[internal]-tree.starts[internal]), 1),
                        device='cuda', dtype=torch.complex128, generator=g)
    return tree, [(0, root), (internal, local)]


def test_nested_ancestor_spaces_transfer_identity_and_action_adjoint():
    tree, seeds = fixture()
    basis = NestedClusterBasis(tree, seeds, config=NestedBasisConfig(max_rank=8, tolerance=1e-11))
    assert int(tree.depth.max()) >= 2
    assert basis.ranks[0] == 2
    for node, seed in seeds:
        restored = basis.expand(node, basis.project(node, seed))
        torch.testing.assert_close(restored, seed, rtol=1e-10, atol=1e-11)
    for parent, children in enumerate(basis.children):
        if not children:
            continue
        q = basis.expand(parent, torch.eye(basis.ranks[parent], device='cuda', dtype=basis.dtype))
        torch.testing.assert_close(q.mH@q, torch.eye(q.shape[1], device='cuda', dtype=q.dtype),
                                   rtol=1e-11, atol=1e-11)
        for child in children:
            qc = basis.expand(child, basis.transfers[child])
            torch.testing.assert_close(q[basis.restriction(parent, child)], qc, rtol=0, atol=0)
    x = torch.randn((2, 3, 128), device='cuda', dtype=basis.dtype, requires_grad=True)
    up = basis.upward(x)
    coefficients = tuple(torch.randn_like(value) for value in up)
    down = basis.downward(coefficients)
    left = sum((a.conj()*b).sum() for a, b in zip(coefficients, up))
    right = (down.conj()*x).sum()
    defect = (left-right).abs()/(torch.linalg.vector_norm(x)*torch.linalg.vector_norm(down))
    assert float(defect.detach()) < 1e-11
    gradient, = torch.autograd.grad(left.real, x)
    torch.testing.assert_close(gradient, down, rtol=1e-12, atol=1e-12)
    receipt = basis.receipt()
    assert not receipt['expanded_internal_bases_retained']
    assert receipt['retained_bytes'] == sum(t.numel()*t.element_size() for t in basis.tensors())
    assert receipt['transfer_matrices'] == tree.node_count-1
    # Original seed mutation must not alter the owned nested representation.
    seeds[0][1].zero_()
    for old, new in zip(up, basis.upward(x)):
        torch.testing.assert_close(new, old, rtol=0, atol=0)
    next(t for t in basis.transfers if t is not None and t.numel()).add_(1)
    with pytest.raises(ValueError, match='changed'):
        basis.upward(x)


def test_nested_rank_failure_and_zero_spaces():
    tree, seeds = fixture()
    with pytest.raises(NestedBasisFailure, match='rank exceeds') as caught:
        NestedClusterBasis(tree, seeds, config=NestedBasisConfig(max_rank=1, tolerance=1e-11))
    assert caught.value.receipt['required_rank'] > 1
    empty = NestedClusterBasis(tree, [])
    x = torch.ones(128, device='cuda', dtype=torch.complex128)
    assert all(rank == 0 for rank in empty.ranks)
    assert torch.equal(empty.downward(empty.upward(x)), torch.zeros_like(x))
    with pytest.raises(ValueError, match='outside'):
        empty.project(-1, torch.empty((0, 1), device='cuda', dtype=x.dtype))


def test_nested_weighted_grams_and_diagonal_forms_match_expanded_oracle():
    tree, seeds = fixture()
    basis = NestedClusterBasis(tree, seeds, config=NestedBasisConfig(max_rank=8, tolerance=1e-11))
    weights = torch.linspace(.1, 3., 128, device='cuda', dtype=torch.float64)
    grams = basis.weighted_grams(weights)
    forms, expected = [], torch.zeros_like(weights)
    for node, rank in enumerate(basis.ranks):
        eye = torch.eye(rank, device='cuda', dtype=basis.dtype)
        q = basis.expand(node, eye)
        points = tree.permutation[tree.starts[node]:tree.stops[node]]
        ids = torch.cat((points, points+tree.size))
        torch.testing.assert_close(grams[node], q.mH@(weights[ids, None]*q), rtol=2e-12, atol=1e-12)
        factor = torch.randn((rank, rank), device='cuda', dtype=basis.dtype)
        form = factor@factor.mH
        forms.append(form)
        expected.index_add_(0, ids, ((q@form)*q.conj()).sum(1).real)
    torch.testing.assert_close(basis.diagonal_forms(forms), expected, rtol=2e-12, atol=1e-12)
    with pytest.raises(ValueError, match='nonnegative'):
        basis.weighted_grams(-weights)
