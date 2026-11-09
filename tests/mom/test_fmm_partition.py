"""Multipole support-sphere traversal covers every ordered pair once."""
import pytest
import torch
from witwin.maxwell.mom.cluster import build_support_octree, sphere_block_partition

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('rectangular', [False, True])
def test_sphere_partition_complete_nonoverlapping_and_batch_independent(rectangular):
    generator = torch.Generator(device='cuda').manual_seed(23)
    points = torch.randn((73, 3), device='cuda', dtype=torch.float64, generator=generator)
    other = torch.randn((41, 3), device='cuda', dtype=torch.float64, generator=generator)+torch.tensor([3., 0., 0.], device='cuda')
    a = build_support_octree(points-.025, points+.025, leaf_size=4)
    b = build_support_octree(other-.02, other+.02, leaf_size=3) if rectangular else a
    baseline = None
    for batch in (1, 7, 64):
        coverage = torch.zeros((a.size, b.size), device='cuda', dtype=torch.int64)
        records = set()
        flags = set()
        for aa, bb, far in sphere_block_partition(a, b, pair_batch=batch):
            assert len(aa) <= batch
            for i, j, admissible in zip(aa.tolist(), bb.tolist(), far.tolist()):
                rows = a.permutation[int(a.starts[i]):int(a.stops[i])]
                cols = b.permutation[int(b.starts[j]):int(b.stops[j])]
                coverage[rows[:, None], cols] += 1
                records.add((i, j, admissible))
                flags.add(admissible)
                if admissible:
                    centers = (a.lower[i]+a.upper[i]-b.lower[j]-b.upper[j])*.5
                    radii = torch.linalg.vector_norm((a.upper[i]-a.lower[i])*.5)+torch.linalg.vector_norm((b.upper[j]-b.lower[j])*.5)
                    assert bool(radii <= .5*torch.linalg.vector_norm(centers))
                else:
                    assert bool((a.children[i] < 0).all() & (b.children[j] < 0).all())
        assert bool((coverage == 1).all())
        assert True in flags and False in flags
        if baseline is None:
            baseline = records
        else:
            assert records == baseline


def test_coincident_supports_are_exact_and_stale_tree_rejects():
    lower = torch.zeros((9, 3), device='cuda', dtype=torch.float64)
    tree = build_support_octree(lower, lower+1, leaf_size=2)
    records = list(sphere_block_partition(tree))
    assert len(records) == 1 and not bool(records[0][2].any())
    assert len(tree.permutation) == 9
    iterator = sphere_block_partition(tree, pair_batch=1)
    next(iterator)
    tree.lower.add_(1)
    with pytest.raises(ValueError, match='changed'):
        list(sphere_block_partition(tree))


@pytest.mark.parametrize('ratio', [0., 1., float('nan'), True])
def test_sphere_partition_invalid_ratio(ratio):
    with pytest.raises(ValueError, match='strictly between'):
        list(sphere_block_partition(None, separation_ratio=ratio))
