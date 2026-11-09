"""Resource evidence rejects workers from a different frozen source state."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest

path = Path(__file__).resolve().parents[2]/'scripts/dev/measure_mom_fmm_scaling.py'
spec = importlib.util.spec_from_file_location('fmm_scaling_runner', path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize('field', ['identity', 'revision', 'source_hash', 'case'])
def test_publication_rejects_mismatched_worker(field):
    record = dict(identity='id', revision='rev', source_hash='sha', case=0)
    runner.validate_worker_result(record, 'id', 'rev', 'sha', 0)
    record[field] = 'changed'
    with pytest.raises(ValueError, match='identity'):
        runner.validate_worker_result(record, 'id', 'rev', 'sha', 0)


def test_worker_discards_measurement_if_source_changes(monkeypatch, capsys, tmp_path):
    identities = iter([('id', 'rev', 'sha'), ('id2', 'rev', 'sha2')])
    observed = []
    def identity(order_manifest=None):
        assert order_manifest is None
        observed.append(order_manifest)
        return next(identities)
    monkeypatch.setattr(runner, 'source_identity', identity)
    monkeypatch.setattr(runner, 'run_case', lambda index: dict(case=index, status='measured', measurement_marker=123))
    monkeypatch.setattr(sys, 'argv', ['runner', '--worker', '--case', '0', '--output', str(tmp_path)])
    runner.main()
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'failed'
    assert 'changed during worker' in result['error']
    assert observed == [None, None]
    assert 'measurement_marker' not in result
    assert result['complexity_qualification'] is False


@pytest.mark.parametrize('field', ['revision', 'source_hash'])
def test_resume_rejects_mismatched_closed_case(tmp_path, monkeypatch, field):
    def identity(order_manifest=None):
        assert order_manifest is None
        return 'id', 'rev', 'sha'
    monkeypatch.setattr(runner, 'source_identity', identity)
    record = dict(identity='id', revision='rev', source_hash='sha', case=0, status='failed')
    record[field] = 'other'
    (tmp_path/'case-00.json').write_text(json.dumps(record))
    monkeypatch.setattr(sys, 'argv', ['runner', '--case', '0', '--output', str(tmp_path)])
    with pytest.raises(ValueError, match='identity'):
        runner.main()


@pytest.mark.parametrize('error', [1e-11, float('nan'), float('inf')])
def test_native_gate_rejects_failed_or_nonfinite_values(error):
    with pytest.raises(ValueError, match='preregistered gate'):
        runner.validate_native_actions(error, 0.)
    with pytest.raises(ValueError, match='preregistered gate'):
        runner.validate_native_actions(0., error)


def test_peak_resets_preserve_warmup_differentiable_and_gap_peaks(monkeypatch):
    import torch
    state = dict(allocated=10, reserved=20, peak_allocated=10, peak_reserved=20)
    def allocate(value):
        state.update(allocated=value, reserved=value+10,
                     peak_allocated=max(state['peak_allocated'], value),
                     peak_reserved=max(state['peak_reserved'], value+10))
    def reset():
        state.update(peak_allocated=state['allocated'], peak_reserved=state['reserved'])
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', reset)
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda: state['allocated'])
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: state['peak_allocated'])
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda: state['peak_reserved'])
    runner._WHOLE_PEAK.clear()
    runner.measured(lambda: allocate(30))
    allocate(150)  # A released gap allocation must survive the next reset.
    allocate(15)
    runner.measured(lambda: allocate(50))
    runner.measured(lambda: allocate(200))  # Differentiable-forward transient.
    runner.measured(lambda: allocate(40))
    assert runner._WHOLE_PEAK == dict(peak_allocated_bytes=200, peak_reserved_bytes=210)
    assert state['peak_allocated'] == 200
    runner.measured(lambda: allocate(20))
    assert state['peak_allocated'] == 40
    assert runner._WHOLE_PEAK == dict(peak_allocated_bytes=200, peak_reserved_bytes=210)
