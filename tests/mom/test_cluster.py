"""Independent support enclosure and exhaustive small partition controls."""
import pytest
import torch
from witwin.maxwell.mom.cluster import build_support_octree, support_block_partition
from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def _boxes(n, seed=5):
    generator = torch.Generator(device='cuda').manual_seed(seed)
    center = torch.rand((n, 3), generator=generator, device='cuda', dtype=torch.float64)
    extent = .015*torch.rand((n, 3), generator=generator, device='cuda', dtype=torch.float64)
    return center-extent, center+extent


def test_owned_tree_encloses_full_support_and_accounts_storage():
    lower, upper = _boxes(127)
    tree = build_support_octree(lower, upper, leaf_size=7)
    assert torch.equal(tree.permutation.sort().values, torch.arange(127, device='cuda'))
    leaf_coverage = torch.zeros(127, device='cuda', dtype=torch.int64)
    for node in range(tree.node_count):
        ids = tree.permutation[tree.starts[node]:tree.stops[node]]
        torch.testing.assert_close(tree.lower[node], lower[ids].amin(0), rtol=0, atol=0)
        torch.testing.assert_close(tree.upper[node], upper[ids].amax(0), rtol=0, atol=0)
        children = tree.children[node][tree.children[node] >= 0]
        if not len(children):
            leaf_coverage[ids] += 1
        else:
            assert int(tree.starts[children[0]]) == int(tree.starts[node])
            assert int(tree.stops[children[-1]]) == int(tree.stops[node])
            assert torch.equal(tree.stops[children[:-1]], tree.starts[children[1:]])
            assert bool((tree.depth[children] == tree.depth[node]+1).all())
    assert bool((leaf_coverage == 1).all())
    receipt = tree.receipt()
    assert receipt['oversized_leaves'] == 0
    assert receipt['retained_bytes'] == sum(t.numel()*t.element_size() for t in tree.tensors())
    saved = tree.lower.clone()
    lower.zero_()
    upper.zero_()
    assert torch.equal(tree.lower, saved)
    tree.lower.add_(1)
    with pytest.raises(ValueError, match='changed'):
        tree.assert_current()


@pytest.mark.parametrize('rectangular', [False, True])
def test_streamed_partition_exactly_covers_pairs_and_excludes_touching_far_blocks(rectangular):
    lower, upper = _boxes(59)
    test = build_support_octree(lower, upper, leaf_size=3)
    source = build_support_octree(*_boxes(43, seed=11), leaf_size=4) if rectangular else test
    coverage = torch.zeros((test.size, source.size), device='cuda', dtype=torch.int64)
    far_count = 0
    for aa, bb, far in support_block_partition(test, source, pair_batch=5):
        assert len(aa) <= 5
        for a, b, admissible in zip(aa, bb, far):
            rows = test.permutation[test.starts[a]:test.stops[a]]
            cols = source.permutation[source.starts[b]:source.stops[b]]
            coverage[rows[:, None], cols[None, :]] += 1
            if bool(admissible):
                far_count += 1
                gap = torch.maximum(test.lower[a]-source.upper[b], source.lower[b]-test.upper[a]).clamp_min(0)
                assert float(torch.linalg.vector_norm(gap)) > 0
            else:
                assert bool((test.children[a] < 0).all() & (source.children[b] < 0).all())
    assert bool((coverage == 1).all())
    assert far_count > 0


def test_duplicate_centers_and_overlapping_supports_remain_exact():
    lower = torch.full((33, 3), -1., device='cuda', dtype=torch.float64)
    upper = -lower
    tree = build_support_octree(lower, upper, leaf_size=2)
    assert tree.node_count == 1
    assert tree.receipt()['oversized_leaves'] == 1
    blocks = list(support_block_partition(tree, pair_batch=1))
    assert len(blocks) == 1 and not bool(blocks[0][2].any())
    with pytest.raises(ValueError, match='ordered'):
        build_support_octree(upper, lower)
    with pytest.raises(ValueError, match='fixed geometry'):
        build_support_octree(lower.requires_grad_(), upper)


def test_actual_rwg_touching_supports_never_enter_admissible_blocks():
    basis = RWGBasis(sphere_mesh(subdivisions=0, device='cuda'))
    triangles = basis.mesh.vertices[basis.mesh.faces]
    supports = torch.cat((triangles[basis.plus_faces], triangles[basis.minus_faces]), dim=1)
    supports = torch.cat((supports, supports+supports.new_tensor([6., 0., 0.])), dim=0)
    lower, upper = supports.amin(dim=1), supports.amax(dim=1)
    tree = build_support_octree(lower, upper, leaf_size=8)
    far_count = 0
    for aa, bb, far in support_block_partition(tree, pair_batch=16):
        for a, b in zip(aa[far], bb[far]):
            rows = tree.permutation[tree.starts[a]:tree.stops[a]]
            cols = tree.permutation[tree.starts[b]:tree.stops[b]]
            gap = torch.maximum(lower[rows, None]-upper[cols], lower[cols]-upper[rows, None]).clamp_min(0)
            assert bool((torch.linalg.vector_norm(gap, dim=-1) > 0).all())
            far_count += 1
    assert far_count > 0


def test_partition_points_split_coincident_box_centers_without_shrinking_supports():
    lower=torch.zeros(8,3,device='cuda',dtype=torch.float64)
    upper=torch.ones_like(lower)
    points=torch.cartesian_prod(*(torch.tensor([.2,.8],device='cuda',dtype=torch.float64) for _ in range(3)))
    old=build_support_octree(lower,upper,leaf_size=1)
    assert old.receipt()['oversized_leaves']==1
    tree=build_support_octree(lower,upper,leaf_size=1,partition_points=points)
    assert tree.receipt()['oversized_leaves']==0
    torch.testing.assert_close(tree.lower,torch.zeros_like(tree.lower),rtol=0,atol=0)
    torch.testing.assert_close(tree.upper,torch.ones_like(tree.upper),rtol=0,atol=0)
    pairs=0
    for a,b,far in support_block_partition(tree):
        assert not bool(far.any())
        pairs+=len(a)
    assert pairs==64
    for invalid in (points.cpu(),points.float(),points[:2],points+2,points.clone().requires_grad_(),points*float('nan')):
        with pytest.raises(ValueError,match='partition points'):
            build_support_octree(lower,upper,partition_points=invalid)
