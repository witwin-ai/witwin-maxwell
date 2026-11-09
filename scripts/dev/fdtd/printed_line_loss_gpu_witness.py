"""Opt-in read-only witnesses around actual prepared CUDA loss-control runs.

Load explicitly with pytest -p. The original engine, tests and assertions are
unchanged; no stand-in solver, counter or spectral field is supplied.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from witwin.maxwell.simulation import PreparedSimulation


def _tensor(value):
    return value.detach().cpu().clone()


def _snapshot(solver):
    if not type(solver).__module__.startswith('witwin.maxwell.fdtd.'):
        raise AssertionError('GPU controls must use the actual native FDTD solver.')
    fields = {name: _tensor(getattr(solver, name)) for base in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')
        for name in (base, base + '_imag') if hasattr(solver, name)}
    coefficients = []
    for entry in solver._copper_foils:
        coefficients.append({'face': entry['face'].name, 'component': entry['electric_name'],
            'imaginary': bool(entry['imaginary']),
            **{name: _tensor(entry[name]) for name in ('B', 'rates', 'D', 'indices', 'area')},
            'spectral_step': None if 'spectral_step' not in entry else float(entry['spectral_step'].detach().cpu()),
            'spectral_end': entry.get('spectral_end'),
            'spectral_norm': None if 'spectral_norm' not in entry else _tensor(entry['spectral_norm'])})
    return {'solver_type': type(solver).__module__ + '.' + type(solver).__name__,
        'actual_dt_s': float(solver.dt), 'fields': fields, 'copper': coefficients}


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    setattr(item, 'loss_witness_' + outcome.get_result().when, outcome.get_result())


@pytest.fixture(autouse=True)
def record_actual_loss_execution(request, monkeypatch):
    if request.node.path.name != 'test_printed_line_loss_execution.py':
        yield
        return
    root = Path(os.environ['PCB_LOSS_WITNESS_ROOT'])
    directory = root / hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:16]
    directory.mkdir(parents=True, exist_ok=False)
    events = []

    for method_name in ('run_until', 'run'):
        original = getattr(PreparedSimulation, method_name)

        def observe(prepared, *arguments, _original=original, _method=method_name, **keywords):
            def save(phase):
                snapshot = _snapshot(prepared.solver)
                path = directory / f'{len(events):02d}-{_method}-{phase}.pt'
                torch.save(snapshot, path)
                events.append({'operation': _method, 'phase': phase, 'path': str(path),
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'field_max_abs': max(float(value.abs().max()) for value in snapshot['fields'].values()),
                    'actual_copper_steps': [entry['spectral_step'] for entry in snapshot['copper']],
                    'actual_copper_ends': [entry['spectral_end'] for entry in snapshot['copper']],
                    'actual_copper_entry_count': len(snapshot['copper'])})
            save('before')
            try:
                return _original(prepared, *arguments, **keywords)
            finally:
                save('after')

        monkeypatch.setattr(PreparedSimulation, method_name, observe)
    yield
    after = [event for event in events if event['phase'] == 'after']
    counters = {step for event in after for step in event['actual_copper_steps'] if step is not None}
    summary = {'nodeid': request.node.nodeid, 'events': events,
        'actual_counter_values': sorted(counters), 'snapshots_are_read_only': True,
        'actual_fields_nonzero_after_advancement': any(event['field_max_abs'] > 0 for event in after),
        'actual_copper_coefficients_captured': bool(after) and all(event['actual_copper_entry_count'] > 0 for event in after)}
    (directory / 'witness.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    if getattr(request.node, 'loss_witness_call', None) is not None and request.node.loss_witness_call.passed:
        assert counters == {512., 1024.}
        assert summary['actual_fields_nonzero_after_advancement']
        assert summary['actual_copper_coefficients_captured']
