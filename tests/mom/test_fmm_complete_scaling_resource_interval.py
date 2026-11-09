"""One new actual CUDA measurement checks the complete scaling workflow scope."""
import importlib.util
import json
from pathlib import Path
import pytest
import torch

path = Path(__file__).resolve().parents[2]/'scripts/dev/measure_mom_fmm_scaling.py'
spec = importlib.util.spec_from_file_location('complete_scaling_resource_runner', path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_actual_shared_resource_workflow_complete_interval(tmp_path):
    # Repeat one of the smallest fixed-size cases, with reference assembly
    # disabled by its preregistered repeat index. This is a workflow/resource
    # check, never an all-size series or fresh physics qualification.
    assert runner.CASES[1] == ('fixed_ka', 0, 1)
    record = runner.run_case(1)
    (tmp_path/'resource.json').write_text(json.dumps(record, indent=2)+'\n')
    assert record['status'] == 'measured' and record['edges'] == 60
    assert record['native_actions_gate_passed']
    assert record['adjoint_relative_error'] < 1e-11
    assert record['current_gradient_relative_error'] < 1e-11
    assert record['actual_owned_retained_bytes'] == record['retained']['planned_bytes']
    assert record['table_storage'] == 'compact_shared_displacement_v1'
    assert record['near_storage'] == 'scalar_reciprocal_dk_v1'
    assert record['shift_storage'] == 'separable_v1'
    assert record['kernel_bound_rule'] == 'positive_tail_min_v1'
    assert record['whole_setup_actions_backward_peak_measured']
    assert len(record['warmups']) == 1 and record['differentiable_forward']['seconds'] > 0
    phases = [record[key] for key in ('setup', 'forward', 'adjoint', 'differentiable_forward', 'backward')]
    phases.extend(record['warmups'][0].values())
    interval = record['whole_setup_actions_backward_allocator']
    for key in ('peak_allocated_bytes', 'peak_reserved_bytes'):
        assert interval[key] >= max(phase[key] for phase in phases)
    assert record['same_N_full_dense_bytes'] == (2*60)**2*16
    assert record['accuracy'] is None and not record['all_coefficients_reference_measured']
    assert not record['whole_peak_qualified'] and not record['memory_crossover_qualified']
    assert not record['physical_qualification'] and not record['complexity_qualification']
