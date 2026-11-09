"""Partial power evidence survives later failure without aliasing live reports."""
import json
import runpy
import pytest
import torch
from witwin.maxwell.mom import harmonic_observables as observables


def test_partial_progress_is_detached_and_survives_query_failure(monkeypatch):
    original = torch.tensor([1.], requires_grad=True)
    calls = []
    reports = []
    def balance(*args, **kwargs):
        calls.append(kwargs)
        if kwargs['volume_order'] == 3:
            raise RuntimeError('query budget exhausted')
        return {name: original * kwargs['surface_order'] for name in (
            'scattered_power_W', 'interference_power_W', 'linear_absorption_W',
            'polarization_work_W', 'nonlinear_dissipation_W')}
    monkeypatch.setattr(observables, 'harmonic_power_balance', balance)
    def progress(step):
        reports.append(step)
        value = step['report']['scattered_power_W']
        assert value.device.type == 'cpu' and not value.requires_grad
        value.fill_(999)
    with pytest.raises(RuntimeError, match='query budget'):
        observables.harmonic_power_refinement(None, None, surface_orders=(2, 4),
            volume_orders=(2, 3), query_batch_size=7, progress_callback=progress)
    assert [step['axis'] for step in reports] == ['initial', 'surface']
    assert original.item() == 1
    difference = reports[1]['report']['integration_refinement']['surface'][0]['terms']['scattered_power_W']['difference_W']
    assert difference.item() == 2
    assert all(call['query_batch_size'] == 7 for call in calls)
    assert all(call['_cache'] is calls[0]['_cache'] for call in calls)


def test_diagnostic_nonfinite_receipts_and_frequency_binding():
    runner = runpy.run_path('scripts/dev/solve_mom_dual_harmonic_power_control.py')
    encoded = runner['portable']({'error': torch.tensor([float('nan'), float('inf')]),
                                  'complex': complex(1, float('-inf'))})
    assert json.loads(json.dumps(encoded, allow_nan=False)) == encoded
    validate = runner['validate_frequency_axis']
    valid = torch.arange(3, dtype=torch.float64) * 3e8
    validate({'state': {'frequencies': valid}}, 2)
    for invalid in (valid * 2, valid.float(), valid[:2], valid + float('nan')):
        with pytest.raises(ValueError, match='frequency axis'):
            validate({'state': {'frequencies': invalid}}, 2)
