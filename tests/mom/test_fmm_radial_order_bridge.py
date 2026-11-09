"""New metadata guards consume closed receipts without numerical replay."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[2]/'scripts/dev/measure_mom_fmm_scaling.py'
    spec = importlib.util.spec_from_file_location('radial_resource_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.configure_radial_series()
    return module


@pytest.fixture
def bundle():
    # An explicit evidence path is required; no missing-evidence skip or mock
    # admission can turn this workflow into an actual resource qualification.
    return json.loads(Path(os.environ['GSP_RADIAL_ORDER_BRIDGE']).read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_new_closed_original_orders_preserve_all_physical_cases(runner, bundle, tmp_path):
    path = tmp_path/'bridge.json'
    write_json(path, bundle)
    result = runner.load_radial_order_bridge(path)['selections']
    assert len(runner.CASES) == 24 and len(result) == 8
    assert [result[f'fixed_ka:{i}']['selected_order'] for i in range(4)] == [6, 6, 6, 11]
    assert [result[f'growing_ka:{i}']['selected_order'] for i in range(4)] == [6, 6, 6, 12]
    for key, selection in result.items():
        family, level = key.split(':')
        assert selection['parameters'] == runner.case_parameters(family, int(level))
        assert selection['original_order_selection']['parameters']['table_storage'] == 'compact_shared_displacement_v1'
        assert selection['parameters']['table_storage'] == 'radial_series_shared_radius_v1'
        assert selection['original_order_selection']['production_source_hash'] != selection['production_source_hash']
        assert selection['production_qualification'] is False


@pytest.mark.parametrize('mutation', ['current_source', 'current_plan', 'terminal_live',
    'terminal_undrained', 'old_receipt_failure', 'old_plan_changed', 'case_hash',
    'missing_repetition', 'old_source_relabel', 'old_native_failure',
    'old_changed_order', 'qualification'])
def test_new_bridge_rejects_changed_identity_or_unclosed_inputs(runner, bundle, tmp_path, mutation):
    bundle = copy.deepcopy(bundle)
    if mutation in ('current_source', 'current_plan'):
        bundle['production_source_hash' if mutation == 'current_source' else 'plan_hash'] = '0'*64
    elif mutation.startswith('terminal_'):
        value = json.loads(Path(bundle['original_terminal']['path']).read_text())
        value['status' if mutation == 'terminal_live' else 'handle_drained'] = 'LIVE' if mutation == 'terminal_live' else False
        bundle['original_terminal'] = write_json(tmp_path/'terminal.json', value)
    elif mutation == 'old_receipt_failure':
        value = json.loads(Path(bundle['original_receipt']['path']).read_text())
        value['returncode'] = 1
        bundle['original_receipt'] = write_json(tmp_path/'receipt.json', value)
    elif mutation == 'old_plan_changed':
        value = json.loads(Path(bundle['original_manifest']['path']).read_text())
        value['plan']['radius'] *= 2
        bundle['original_manifest'] = write_json(tmp_path/'manifest.json', value)
    elif mutation == 'case_hash':
        bundle['records']['0']['sha256'] = '0'*64
    elif mutation == 'missing_repetition':
        del bundle['records']['23']
    elif mutation == 'qualification':
        bundle['production_qualification'] = True
    else:
        value = json.loads(Path(bundle['records']['3']['path']).read_text())
        if mutation == 'old_source_relabel':
            value['order_selection']['production_source_hash'] = runner.production_source_hash()
        elif mutation == 'old_native_failure':
            value['adjoint_relative_error'] = 1e-11
        else:
            value['selected_order'] = value['order_selection']['selected_order'] = 7
        bundle['records']['3'] = write_json(tmp_path/'case.json', value)
    path = tmp_path/'bridge.json'
    write_json(path, bundle)
    with pytest.raises(ValueError):
        runner.load_radial_order_bridge(path)


@pytest.mark.parametrize('failure', ['adjoint', 'gradient', 'nonfinite', 'success'])
def test_new_final_capture_and_failure_only_native_state(runner, tmp_path, failure):
    import torch
    value = torch.arange(4, device='cuda', dtype=torch.float64).to(torch.complex128)
    applied = value.clone()
    if failure == 'nonfinite':
        applied[0] = float('nan')
    path = tmp_path/'native-state.pt'
    runner._WHOLE_PEAK.clear()
    kwargs = dict(adjoint_error=1e-11 if failure == 'adjoint' else 0.,
                  gradient_error=1e-11 if failure == 'gradient' else 0.,
                  failure_state_path=path, context=dict(x=value, y=value))
    if failure == 'success':
        interval = runner.finish_native_interval(applied, value, value, value, **kwargs)
        assert interval['peak_allocated_bytes'] >= value.untyped_storage().nbytes()
        assert not path.exists()
        return
    with pytest.raises(ValueError) as caught:
        runner.finish_native_interval(applied, value, value, value, **kwargs)
    receipt = caught.value.receipt
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['native_state_sha256']
    state = torch.load(path, weights_only=True)
    assert state['allocator']['peak_allocated_bytes'] >= value.untyped_storage().nbytes()
    assert torch.equal(state['context']['x'], value)
    assert torch.equal(state['adjoint_value'], value)
    assert state['finite'] is (failure != 'nonfinite')
    assert receipt['production_qualification'] is False
