"""Missing or failed scale points cannot silently become passing tail fits."""
import importlib.util
import json
from pathlib import Path
import pytest

path = Path(__file__).resolve().parents[2]/'scripts/dev/summarize_mom_fmm_scaling.py'
spec = importlib.util.spec_from_file_location('fmm_summary', path)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def fixture(directory):
    runner = summary.runner
    cases = [list(case) for case in runner.CASES]
    manifest = dict(identity='id', revision='rev', source_hash='sha', cases=cases,
                    plan=runner.PLAN)
    (directory/'manifest.json').write_text(json.dumps(manifest))
    for i, (family, level, repeat) in enumerate(cases):
        n = 60*4**level
        record = dict(identity='id', revision='rev', source_hash='sha', case=i, family=family,
            subdivision=level, repetition=repeat, edges=n, selected_order=6, status='measured',
            retained=dict(planned_bytes=n**2), whole_setup_actions_backward_peak_measured=True,
            native_actions_gate_passed=True, adjoint_relative_error=1e-15,
            current_gradient_relative_error=1e-15,
            whole_setup_actions_backward_allocator=dict(peak_allocated_bytes=n**2, peak_reserved_bytes=n**2))
        parameters = runner.case_parameters(family, level)
        selection = dict(version='scalar_fmm_reused_order_selection_v1', parameters=parameters,
            plan_hash=runner.plan_hash(), production_source_hash=runner.production_source_hash(),
            selected_order=6, source_kind='fresh_current_order_plan', source_path=None, source_sha256=None,
            production_qualification=False)
        if repeat > 0:
            selection = runner.reused_case_order(directory/f'case-{i-repeat:02d}.json', i, 'id', 'rev', 'sha')
        record.update(order_selection=selection, order_plan=dict(selected_order=6), frequency=parameters['frequency'],
                      **{k: parameters[k] for k in ('kernel_bound_rule','table_storage','near_storage','shift_storage')})
        for phase in ('setup', 'forward', 'adjoint', 'differentiable_forward', 'backward'):
            record[phase] = dict(seconds=n**2, peak_allocated_bytes=n**2, peak_reserved_bytes=n**2)
        record['warmups'] = [dict(forward=record['forward'].copy(), adjoint=record['adjoint'].copy())]
        (directory/f'case-{i:02d}.json').write_text(json.dumps(record))


def test_fit_uses_every_size_and_repetition(tmp_path):
    fixture(tmp_path)
    result = summary.summarize(tmp_path)
    fit = result['families']['fixed_ka']['all_point_fits']['setup_seconds']
    assert fit['exponent'] == pytest.approx(2.)
    assert fit['samples'] == 12 and fit['sizes'] == 4
    assert result['complete'] and not result['complexity_qualification']


@pytest.mark.parametrize('failure', [False, True])
def test_incomplete_series_never_fits_remaining_tail(tmp_path, failure):
    fixture(tmp_path)
    for index in range(3):
        path = tmp_path/f'case-{index:02d}.json'
        if failure:
            record = json.loads(path.read_text())
            record.update(status='failed', error='budget exceeded')
            path.write_text(json.dumps(record))
        else:
            path.unlink()
    result = summary.summarize(tmp_path)
    assert not result['complete']
    assert result['families']['fixed_ka']['all_point_fits'] is None
    assert bool(result['failed_cases']) is failure


def test_reject_mismatched_worker_source(tmp_path):
    fixture(tmp_path)
    path = tmp_path/'case-00.json'
    record = json.loads(path.read_text())
    record['source_hash'] = 'different'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='identity'):
        summary.summarize(tmp_path)


@pytest.mark.parametrize('field', ['adjoint_relative_error', 'current_gradient_relative_error'])
def test_measured_status_cannot_hide_failed_native_actions(tmp_path, field):
    fixture(tmp_path)
    path = tmp_path/'case-00.json'
    record = json.loads(path.read_text())
    record[field] = 1e-11
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='native action'):
        summary.summarize(tmp_path)


@pytest.mark.parametrize('phase', ['setup', 'differentiable_forward', 'backward', 'warmup'])
def test_whole_peak_cannot_omit_a_larger_phase(tmp_path, phase):
    fixture(tmp_path)
    path = tmp_path/'case-00.json'
    record = json.loads(path.read_text())
    target = record['warmups'][0]['forward'] if phase == 'warmup' else record[phase]
    target['peak_allocated_bytes'] *= 2
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='omits a measured phase'):
        summary.summarize(tmp_path)


def test_failed_case_requires_frozen_source_identity(tmp_path):
    fixture(tmp_path)
    path = tmp_path/'case-00.json'
    record = json.loads(path.read_text())
    record.update(status='failed', source_hash='wrong', error='budget failure')
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='identity'):
        summary.summarize(tmp_path)
