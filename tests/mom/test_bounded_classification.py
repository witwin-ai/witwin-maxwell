"""Dense traversal equivalence and CUDA classification workspace bounds."""
import pytest
import torch
from witwin.maxwell.mom.mesh import sphere_mesh
from witwin.maxwell.mom.quadrature import classify_pairs, classify_all_pairs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('batch', [1, 31, 65536])
def test_categories_match_full_pair_classification(batch):
    mesh = sphere_mesh(radius=.1, subdivisions=1, device='cuda')
    n = len(mesh.faces)
    ids = torch.arange(n*n, device=mesh.device)
    expected = classify_pairs(mesh, ids//n, ids % n)
    actual = classify_all_pairs(mesh, batch_size=batch)
    assert actual.dtype == torch.uint8
    assert torch.equal(actual.to(expected.dtype), expected)
    assert set(actual.unique().tolist()) == {0, 1, 2, 3, 4}


@pytest.mark.parametrize('batch', [True, 0, -1, 1.5])
def test_invalid_batch_rejected(batch):
    mesh = sphere_mesh(radius=.1, subdivisions=0, device='cuda')
    with pytest.raises(ValueError, match='positive integer'):
        classify_all_pairs(mesh, batch_size=batch)


def test_classification_peak_is_bounded_by_batch_not_pair_count():
    mesh = sphere_mesh(radius=.1, frequency=6, device='cuda')
    n = len(mesh.faces)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    actual = classify_all_pairs(mesh, batch_size=1024)
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated()-before
    assert extra < actual.numel()*actual.element_size()+16*1024**2
    ids = torch.arange(n*n, device=mesh.device)
    assert torch.equal(actual.long(), classify_pairs(mesh, ids//n, ids % n))
