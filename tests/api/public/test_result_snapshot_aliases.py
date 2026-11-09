"""Snapshot aliases must not multiply CUDA-to-host data or merge views."""
import zipfile

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.result import _cpu_serializable


def test_nested_snapshot_preserves_tensor_aliases_and_detaches_graph():
    source = torch.arange(6., requires_grad=True).square()
    snapshot = _cpu_serializable({'data': source, 'nested': [source, (source,)]})
    assert snapshot['data'] is snapshot['nested'][0]
    assert snapshot['data'] is snapshot['nested'][1][0]
    assert not snapshot['data'].requires_grad
    assert snapshot['data'].grad_fn is None
    torch.testing.assert_close(snapshot['data'], source, rtol=0, atol=0)
    assert source.grad_fn is not None


@pytest.mark.parametrize('kind', ('transpose', 'slice', 'conjugate', 'negative', 'dtype'))
def test_distinct_views_retain_their_values_layout_and_flags(kind):
    source = torch.complex(torch.arange(16.).reshape(4, 4), torch.ones(4, 4))
    view = {'transpose': lambda: source.T, 'slice': lambda: source[::2, 1:],
            'conjugate': source.conj, 'negative': lambda: torch._neg_view(source),
            'dtype': lambda: source.view(torch.float32)}[kind]()
    snapshot = _cpu_serializable({'source': source, 'view': view, 'repeat': view})
    assert snapshot['source'] is not snapshot['view']
    assert snapshot['view'] is snapshot['repeat']
    for key, original in [('source', source), ('view', view)]:
        actual = snapshot[key]
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
        assert actual.dtype == original.dtype
        assert actual.stride() == original.stride()
        assert actual.is_conj() == original.is_conj()
        assert actual.is_neg() == original.is_neg()


def test_equal_values_in_distinct_tensors_are_not_merged():
    first = torch.ones(3)
    second = first.clone()
    snapshot = _cpu_serializable((first, second))
    assert snapshot[0] is not snapshot[1]
    snapshot[0][0] = 2
    assert snapshot[1][0] == 1


def test_snapshot_memo_does_not_persist_across_calls():
    source = torch.ones(3)
    first = _cpu_serializable({'value': source})
    second = _cpu_serializable({'value': source})
    assert first['value'] is not second['value']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires actual CUDA transfers.')
@pytest.mark.parametrize('dtype', (torch.complex64, torch.complex128))
def test_actual_cuda_monitor_aliases_serialize_one_storage(tmp_path, dtype):
    source = torch.arange(4096., device='cuda').to(dtype).reshape(64, 64)
    scene = mw.Scene(domain=mw.Domain(bounds=((-1., 1.),) * 3),
                     grid=mw.GridSpec.uniform(.5), device='cpu')
    result = mw.Result(method='fdtd', scene=scene, frequency=1e9, fields={},
                       monitors={'plane': {'kind': 'plane', 'data': source,
                                          'components': {'Ex': {'data': source}}, 'Ex': source}})
    target = tmp_path / 'result.pt'
    result.save(target)
    payload = torch.load(target, map_location='cpu', weights_only=False)
    plane = payload['monitors']['plane']
    assert plane['data'] is plane['Ex'] is plane['components']['Ex']['data']
    torch.testing.assert_close(plane['data'], source.cpu(), rtol=0, atol=0)
    assert not plane['data'].is_cuda
    with zipfile.ZipFile(target) as archive:
        sizes = [entry.file_size for entry in archive.infolist() if '/data/' in entry.filename]
    assert sizes == [source.numel() * source.element_size()]
    restored = mw.Result.load(target, scene=scene, map_location='cpu')
    stored = restored._monitors['plane']
    assert stored['data'] is stored['Ex'] is stored['components']['Ex']['data']
    torch.testing.assert_close(stored['data'], source.cpu(), rtol=0, atol=0)
