"""Small genuine 4 ns CPW basis: direct capture, CPU consumer and seven gates.

A unit horizon, not a 64 ns science column; gate values are reported, and
only the data-bridge contracts are asserted.
"""
from dataclasses import replace
import json
import numpy as np, pytest, torch

import scripts.dev.fdtd.printed_line_loss_bridge as bridge
from scripts.dev.fdtd.printed_line_public_execution import build_public_column, column_case, run_public_column


def compact_case():
    return replace(column_case(0), axial_half_length=.06, separation=.06,
        padding=.01, positions=(-.015, 0., .015), duration=4e-9, checkpoint_time=2e-9)


def test_direct_capture_feeds_original_seven_gates(tmp_path, request):
    if not request.config.getoption('--gpu', default=False) or not torch.cuda.is_available():
        pytest.skip('Needs explicit --gpu and CUDA; no implicit GPU execution.')
    case = compact_case()
    folders = [tmp_path / f'drive{d}' for d in range(2)]
    for drive, folder in enumerate(folders):
        report = run_public_column('cpw', case, drive, folder, graph=True, resume=True)
        realization = report['executed_Cu_realization']
        assert realization['provenance'] == 'direct_executed_capture'
        assert realization['executed_copper_operator_signature'] == report['actual_operator']['copper']
        assert report['source_identity_before'] == report['source_identity_after']
    context = build_public_column('cpw', case)
    summaries = []
    for folder in folders:
        evidence, capture, result, VI, queries = bridge.load_public_column(context, folder)
        column = bridge.consume_public_column(context, result, capture, evidence, VI, queries)
        assert column['coherent_loss']['max_relative_replay_error'] <= bridge.LOSS_REPLAY_RELATIVE
        assert column['clock']['Cu_sampling_traces'] == len(capture['basis'].realizations) > 0
        assert set(column['per_column_gate_values']) == {'source_identity_unchanged', 'source_band_coverage', 'tail_decay'}
        summaries.append(bridge.column_summary(column, VI=VI, result=result, basis=capture['basis']))
    with pytest.raises(ValueError, match='original physical lengths'):
        bridge.reduce_public_basis('cpw', summaries, cpu_scene=None, regional_schema=None, regional_bounds=None)
    # The 4 ns compact fixture is not a physical length; it is labelled 0 here
    # only to exercise the reducer on genuine captured data.
    summaries = [dict(summary, length_index=0) for summary in summaries]
    report = bridge.reduce_public_basis('cpw', summaries, cpu_scene=context['observed'],
        regional_schema=context['regional_schema'], regional_bounds=context['regional_bounds'])
    assert tuple(report['original_seven_gates']) == bridge.SEVEN_GATES
    assert report['original_seven_gates']['same_actual_port_operator']
    assert report['original_seven_gates']['coherent_solve'] and np.isfinite(report['gate_values']['state_condition'])
    assert report['gate_values']['coherent_regional_power_balance'] is not None
    print(json.dumps(dict(gates=report['original_seven_gates'], values=report['gate_values']), default=str))
    # A tampered on-disk realization is rejected before any consumer runs.
    payload = torch.load(folders[0] / bridge.CAPTURE_FILE, weights_only=False)
    torch.save(dict(payload, binding_sha256='0' * 64), folders[0] / bridge.CAPTURE_FILE)
    with pytest.raises(ValueError):
        bridge.load_public_column(context, folders[0])
