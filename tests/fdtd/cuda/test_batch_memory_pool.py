"""Native allocation caps remain local to a Batch and survive retained outputs."""
import gc

import pytest
import torch
from witwin.maxwell.execution.pool import BatchMemoryPool

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


def test_pool_caps_segments_without_replacing_default_allocator():
    backend = torch.cuda.memory.get_allocator_backend()
    outside = torch.zeros(16, device="cuda:0")
    budget = BatchMemoryPool("cuda:0", 8 * 1024**2, 0)
    with budget.activate():
        kept = torch.ones(1024, device="cuda:0")
        with pytest.raises(torch.cuda.OutOfMemoryError):
            torch.empty(16 * 1024**2, dtype=torch.uint8, device="cuda:0")
    assert 0 < budget.stats()["peak_charged_bytes"] <= 8 * 1024**2
    assert torch.cuda.memory.get_allocator_backend() == backend
    outside.add_(1)
    del budget
    gc.collect()
    torch.testing.assert_close(kept, torch.ones_like(kept))


def test_pool_supports_graph_capture_and_distinct_pool_ownership():
    a = BatchMemoryPool("cuda:0", 8 * 1024**2, 0)
    b = BatchMemoryPool("cuda:0", 8 * 1024**2, 0)
    with a.activate():
        first = torch.zeros(100, device="cuda:0")
        with b.activate():
            second = torch.zeros(100, device="cuda:0")
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        torch.cuda.synchronize()
        with a.capture_scope(), torch.cuda.graph(graph, stream=stream, pool=a.pool.id):
            first.add_(1)
        graph.replay()
    assert first.sum().item() == 100
    assert second.sum().item() == 0
    assert a.stats()["charged_bytes"] <= a.limit_bytes
    assert 0 < b.stats()["charged_bytes"] <= b.limit_bytes
