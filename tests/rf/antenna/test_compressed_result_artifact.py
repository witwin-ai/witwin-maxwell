"""Actual public snapshot round trips and corruption/resource negatives."""
import gzip
import hashlib
import json

import pytest
import torch
import witwin.maxwell as mw

from scripts.dev.fdtd.compressed_result_artifact import (
    save_compressed_result, load_compressed_result, check_artifact_resources,
)


def result():
    values = torch.arange(60, dtype=torch.float64).reshape(3, 4, 5)
    return mw.Result(method='fdtd', scene=None, frequencies=(1e9,),
        fields={'Ex': values.to(torch.complex128) * (1 + 2j), 'Hy': values.float()},
        metadata={'physical_input_sha256': 'unchanged'}, solver_stats={'steps': 1024})


def test_real_result_preserves_dtype_shape_values_and_original_serialized_bytes(tmp_path):
    before = torch.cuda.is_initialized()
    original = result()
    path = tmp_path / 'physical.result.pt.gz'
    record = save_compressed_result(original, path, max_raw_bytes=2**20, reserve_bytes=0)
    restored = load_compressed_result(path, scene=None)
    for component, values in original.fields.items():
        assert restored.fields[component].dtype == values.dtype
        torch.testing.assert_close(restored.fields[component], values, rtol=0, atol=0)
    assert restored.frequencies == original.frequencies
    assert restored._metadata['physical_input_sha256'] == 'unchanged'
    assert restored.stats()['steps'] == 1024
    raw = gzip.decompress(path.read_bytes())
    assert hashlib.sha256(raw).hexdigest() == record['raw_sha256']
    assert len(raw) == record['raw_bytes']
    assert set(p.name for p in tmp_path.iterdir()) == {path.name, path.name + '.json'}
    assert torch.cuda.is_initialized() == before


@pytest.mark.parametrize('target', ['compressed', 'receipt_hash', 'receipt_size'])
def test_corruption_never_restores_an_unverified_result(tmp_path, target):
    path = tmp_path / 'data.pt.gz'
    save_compressed_result(result(), path, max_raw_bytes=2**20, reserve_bytes=0)
    receipt = path.with_name(path.name + '.json')
    if target == 'compressed':
        data = bytearray(path.read_bytes()); data[len(data)//2] ^= 1; path.write_bytes(data)
    else:
        record = json.loads(receipt.read_text())
        record['raw_sha256' if target == 'receipt_hash' else 'raw_bytes'] = '0'*64 if target == 'receipt_hash' else 1
        receipt.write_text(json.dumps(record))
    with pytest.raises(ValueError): load_compressed_result(path, scene=None)


@pytest.mark.parametrize('existing', ['complete', 'partial', 'receipt'])
def test_refuses_to_overwrite_any_existing_evidence(tmp_path, existing):
    path = tmp_path / 'data.pt.gz'
    actual = {'complete': path, 'partial': path.with_name(path.name + '.partial'),
              'receipt': path.with_name(path.name + '.json')}[existing]
    actual.write_bytes(b'original evidence')
    with pytest.raises(FileExistsError):
        save_compressed_result(result(), path, max_raw_bytes=2**20, reserve_bytes=0)
    assert actual.read_bytes() == b'original evidence'


def test_pre_solve_admission_rejects_broken_prior_artifact_link(tmp_path):
    path = tmp_path / 'data.pt.gz'
    path.symlink_to(tmp_path / 'missing-prior-output')
    with pytest.raises(FileExistsError):
        check_artifact_resources(path, max_raw_bytes=2**20, reserve_bytes=0)
    assert path.is_symlink()


@pytest.mark.parametrize('budget', [0, -1, 1.5, True])
def test_rejects_invalid_explicit_byte_budget_before_snapshot(tmp_path, budget):
    with pytest.raises(ValueError):
        save_compressed_result(result(), tmp_path/'data.pt.gz', max_raw_bytes=budget)
    assert not list(tmp_path.iterdir())


def test_actual_disk_floor_rejects_before_serialization(tmp_path):
    import shutil
    with pytest.raises(RuntimeError, match='free disk'):
        save_compressed_result(result(), tmp_path/'data.pt.gz', max_raw_bytes=2**20,
            reserve_bytes=shutil.disk_usage(tmp_path).free)
    assert not list(tmp_path.iterdir())


def test_actual_snapshot_budget_overrun_keeps_no_partial_disk_data(tmp_path):
    with pytest.raises(RuntimeError, match='exceeded'):
        save_compressed_result(result(), tmp_path/'data.pt.gz', max_raw_bytes=1, reserve_bytes=0)
    assert not list(tmp_path.iterdir())
