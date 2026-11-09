"""Variable leaf sizes, additive block rows and bounded packed exact actions."""
import pytest
import torch
from witwin.maxwell.mom.packed_near import PackedNearBlocks

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def records():
    generator = torch.Generator(device='cuda').manual_seed(113)
    nodes = {3: torch.tensor([0, 5, 2], device='cuda'),
             8: torch.tensor([3, 1], device='cuda'),
             9: torch.tensor([4, 6], device='cuda')}
    return [(a, b, nodes[a], nodes[b], torch.randn((len(nodes[a]), len(nodes[b])),
                device='cuda', dtype=torch.complex128, generator=generator))
            for a, b in ((8, 9), (3, 3), (9, 8), (3, 9), (8, 8), (9, 3))]


def test_packed_rectangular_blocks_forward_adjoint_gradient_and_storage():
    raw = records()
    packed = PackedNearBlocks(raw, 7, action_batch=2)
    dense = torch.zeros((7, 7), device='cuda', dtype=torch.complex128)
    for _, _, rows, cols, block in raw:
        dense[rows[:, None], cols] += block
    x = torch.randn((2, 3, 7), device='cuda', dtype=dense.dtype, requires_grad=True)
    y = torch.randn_like(x)
    actual = packed.apply(x)
    torch.testing.assert_close(actual, x@dense.mT, rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(packed.adjoint(y), y@dense.conj(), rtol=2e-14, atol=2e-14)
    defect = ((y.conj()*actual).sum()-(packed.adjoint(y).conj()*x).sum()).abs()
    assert float(defect.detach()) < 1e-11
    grad, = torch.autograd.grad((actual.conj()*actual).real.sum()/2, x)
    torch.testing.assert_close(grad, packed.adjoint(actual), rtol=2e-14, atol=2e-14)
    assert packed.row_nodes.tolist() == [3, 8, 9]
    assert packed.rowptr.tolist() == [0, 2, 4, 6]
    assert packed.column_nodes.tolist() == [3, 9, 8, 9, 3, 8]
    for (a, b, rows, cols, block) in raw:
        position = [(aa, bb) for aa, bb, *_ in sorted(raw, key=lambda item: item[:2])].index((a, b))
        gi, bi = packed.locations[position].tolist()
        rr, cc, values = packed.groups[gi]
        assert torch.equal(rr[bi], rows) and torch.equal(cc[bi], cols)
        assert torch.equal(values[bi], block)
    assert packed.receipt()['retained_bytes'] == sum(t.numel()*t.element_size() for t in packed.tensors())
    assert packed.receipt()['padding_bytes'] == 0
    # Packing owns its storage and does not retain aliases to assembly inputs.
    for _, _, rows, cols, block in raw:
        block.zero_()
    torch.testing.assert_close(packed.apply(x), actual, rtol=0, atol=0)
    packed.groups[0][2].add_(1)
    with pytest.raises(ValueError, match='changed'):
        packed.apply(x)


def test_packed_budget_rejects_before_value_allocation(monkeypatch):
    raw = records()
    def forbidden(*args, **kwargs):
        raise AssertionError('The failing budget must precede stacking allocation')
    monkeypatch.setattr(torch, 'stack', forbidden)
    with pytest.raises(MemoryError, match='before allocation'):
        PackedNearBlocks(raw, 7, retained_budget_bytes=1)


def test_packed_layout_contract_rejects_ambiguous_nodes():
    raw = records()
    with pytest.raises(ValueError, match='unique nonnegative'):
        PackedNearBlocks(raw+[raw[0]], 7)
    a, b, rows, cols, block = raw[0]
    with pytest.raises(ValueError, match='different coefficient maps'):
        PackedNearBlocks(raw+[(b, b, cols.flip(0), cols.flip(0), block)], 7)
    with pytest.raises(ValueError, match='finite fixed'):
        PackedNearBlocks([(a, b, rows, cols, block.requires_grad_())], 7)


def test_streamed_factory_releases_each_block_and_matches_packed_actions():
    import weakref
    raw = records()
    originals = {(a, b): value for a, b, _, _, value in raw}
    layout = [record[:4] for record in raw]
    references = []
    def factory(a, b, rows, cols):
        assert all(reference() is None for reference in references)
        value = originals[a, b].clone()
        references.append(weakref.ref(value))
        return value
    streamed = PackedNearBlocks.from_layout(layout, 7, factory,
        device='cuda', dtype=torch.complex128, action_batch=2)
    assert len(references) == len(layout) and all(reference() is None for reference in references)
    packed = PackedNearBlocks(raw, 7, action_batch=2)
    x = torch.randn((3, 7), device='cuda', dtype=torch.complex128, requires_grad=True)
    for adjoint in (False, True):
        torch.testing.assert_close(streamed.action(x, adjoint=adjoint),
            packed.action(x, adjoint=adjoint), rtol=0, atol=0)
    result = streamed.apply(x)
    gradient, = torch.autograd.grad(result.abs().square().sum()/2, x)
    torch.testing.assert_close(gradient, streamed.adjoint(result), rtol=2e-14, atol=2e-14)
    assert streamed.receipt() == packed.receipt()
    with pytest.raises(MemoryError, match='before allocation'):
        PackedNearBlocks.from_layout(layout, 7, factory, device='cuda',
            dtype=torch.complex128, retained_budget_bytes=1)
    assert len(references) == len(layout)


def test_streamed_construction_avoids_full_duplicate_value_storage():
    import gc
    width = 256
    maps = [torch.arange(i*width, (i+1)*width, device='cuda') for i in range(4)]
    layout = [(a, b, maps[a], maps[b]) for a in range(4) for b in range(4)]
    def factory(a, b, rows, cols):
        return torch.ones((len(rows), len(cols)), device='cuda', dtype=torch.complex128)
    def peak(streamed):
        gc.collect()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        if streamed:
            packed = PackedNearBlocks.from_layout(layout, 4*width, factory,
                device='cuda', dtype=torch.complex128)
        else:
            raw = [(*entry, factory(*entry)) for entry in layout]
            packed = PackedNearBlocks(raw, 4*width)
        torch.cuda.synchronize()
        observed = torch.cuda.max_memory_allocated()-baseline
        assert packed.receipt()['blocks'] == 16
        return observed
    buffered_peak, streamed_peak = peak(False), peak(True)
    value_bytes = 16*width*width*16
    print('near construction allocated peaks', buffered_peak, streamed_peak)
    assert buffered_peak-streamed_peak >= value_bytes//2
