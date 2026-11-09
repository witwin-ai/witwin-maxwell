"""All FMM budget and expansion fields are versioned and validated."""
from dataclasses import replace
import pytest
import witwin.maxwell as mw


def test_fmm_configuration_strict_roundtrip_and_scope():
    config = mw.FMMConfig(order=5, table_storage='compact', m2l_row_batch=7, quadrature_order=6, leaf_size=17, separation_ratio=.3,
        table_budget_bytes=1000000, near_budget_bytes=2000000, projection_budget_bytes=3000000,
        solve_dual_tolerance=1e-9, original_sample_tolerance=2e-9)
    assert mw.FMMConfig.restore(config.payload()) == config
    assert config.solve_config().original_sample_tolerance == 2e-9
    assert config.construction()['projection_budget_bytes'] == 3000000
    for name in config.payload()['parameters']:
        payload = config.payload()
        del payload['parameters'][name]
        with pytest.raises(ValueError, match='incomplete'):
            mw.FMMConfig.restore(payload)
    with pytest.raises(ValueError, match='requires operator_backend=fmm'):
        mw.MoMConfig(fmm=config)


@pytest.mark.parametrize('change', [{'near_action_batch': 0}, {'near_action_batch': True}, {'table_storage': 'unknown'}, {'m2l_row_batch': False}, {'order': True}, {'order': -1}, {'quadrature_order': False},
    {'leaf_size': 0}, {'separation_ratio': 1.}, {'separation_ratio': float('nan')},
    {'table_budget_bytes': 0}, {'near_budget_bytes': 1.5}, {'projection_budget_bytes': False},
    {'solve_dual_tolerance': float('inf')}, {'original_sample_tolerance': -1.}])
def test_invalid_fmm_configuration(change):
    with pytest.raises(ValueError):
        replace(mw.FMMConfig(), **change)


def test_periodic_fmm_rejects_unimplemented_error_admission():
    from witwin.maxwell.mom.linalg import StoppingRule
    from witwin.maxwell.mom.interpolation_error import KernelErrorBudget
    kwargs = dict(operator_backend='fmm', riesz_storage='local_rwg_v1',
        stopping=StoppingRule(rtol=1e-9, preconditioner='rwg_diagonal'),
        periodic=mw.MoMPeriodicConfig(((1., 0., 0.),)))
    budget = KernelErrorBudget(1e-6, 1e-5)
    for fmm in (mw.FMMConfig(table_storage='dense', near_storage='full_matrix', kernel_error_budget=budget),
                mw.FMMConfig(table_storage='dense', near_storage='full_matrix', kernel_error_budget=budget, max_order=8),
                mw.FMMConfig(table_storage='dense', near_storage='full_matrix', kernel_error_budget=budget, max_order=8, rwg_absolute_budget=1e-8)):
        with pytest.raises(NotImplementedError, match='fixed order'):
            mw.MoMConfig(**kwargs, fmm=fmm)
    assert mw.MoMConfig(**kwargs, fmm=mw.FMMConfig(table_storage='dense', near_storage='full_matrix')).fmm.order == 6


def test_leaf_factor_configuration_is_solve_only_and_strict():
    config = mw.FMMConfig(leaf_factor_kernel='primary', leaf_factor_max_edges=7,
                          leaf_factor_budget_bytes=12345)
    assert mw.FMMConfig.restore(config.payload()) == config
    assert not any(name.startswith('leaf_factor_') for name in config.construction())
    for change in ({'leaf_factor_kernel': 'invalid'}, {'leaf_factor_max_edges': True},
                   {'leaf_factor_budget_bytes': 0}):
        with pytest.raises(ValueError):
            replace(config, **change)


def test_near_action_batch_roundtrip_and_construction():
    config = mw.FMMConfig(near_action_batch=256)
    assert mw.FMMConfig.restore(config.payload()) == config
    assert config.construction()['near_action_batch'] == 256


def test_compact_fmm_storage_defaults():
    config = mw.FMMConfig()
    assert (config.table_storage, config.shift_storage, config.near_storage) == (
        'compact_shared_displacement_v1', 'separable_v1', 'scalar_reciprocal_dk_v1')
    assert config.payload()['version'] == 'cartesian_fmm_shared_compact_config_v1'
    storage_fields = ('table_storage', 'shift_storage', 'near_storage',
        'near_coefficient_workspace_budget_bytes', 'near_coefficient_cache_budget_bytes')
    for table in ('dense', 'compact', 'fft_v1', 'fft_ewald_v1',
            'compact_shared_displacement_v1', 'compact_shared_signed_axis_v1',
            'radial_series_shared_radius_v1'):
        for shift in ('dense', 'separable_v1'):
            for near in ('full_matrix', 'scalar_dk_v1', 'scalar_reciprocal_dk_v1',
                    'scalar_reciprocal_streamed_dk_v1', 'scalar_reciprocal_cached_dk_v1'):
                if table == 'fft_ewald_v1' and near != 'full_matrix':
                    continue
                selected = replace(config, table_storage=table, shift_storage=shift, near_storage=near)
                payload = selected.payload()
                assert all(payload['parameters'][name] == getattr(selected, name) for name in storage_fields)
                assert mw.FMMConfig.restore(payload) == selected
                for name in storage_fields:
                    incomplete = selected.payload()
                    del incomplete['parameters'][name]
                    with pytest.raises(ValueError):
                        mw.FMMConfig.restore(incomplete)


def test_hierarchical_inner_tolerance_requires_dual_budget_margin():
    import json
    import math
    from pathlib import Path
    from witwin.maxwell.mom.linalg import StoppingRule

    manifest = json.loads((Path(__file__).resolve().parents[2] /
        'docs/reference/fdtd-capability-guards.json').read_text())
    node = 'tests/mom/test_fmm_config.py::test_hierarchical_inner_tolerance_requires_dual_budget_margin'
    guard, = (entry for entry in manifest['configuration_guards'] if node in entry['negative_tests'])
    assert guard['negative_tests'] == [node]
    for backend, name, config in (
            ('hmatrix', 'hierarchical', mw.HMatrixConfig(solve_dual_tolerance=1e-4)),
            ('h2', 'h2', mw.H2Config(solve_dual_tolerance=1e-4)),
            ('direct_h2', 'direct_h2', mw.DirectH2Config(solve_dual_tolerance=1e-4)),
            ('fmm', 'fmm', mw.FMMConfig(solve_dual_tolerance=1e-4)),
            ('fmm', 'fmm', mw.DirectionalFMMConfig(solve_dual_tolerance=1e-4))):
        options = dict(operator_backend=backend, riesz_storage='local_rwg_v1', **{name: config})
        limit = 0.1 * config.solve_dual_tolerance
        for rtol in (math.nextafter(limit, math.inf), config.solve_dual_tolerance):
            with pytest.raises(ValueError) as error:
                mw.MoMConfig(**options, stopping=StoppingRule(rtol=rtol, preconditioner='none'))
            assert str(error.value) == guard['message']
        for rtol in (limit, limit / 10):
            assert mw.MoMConfig(**options,
                stopping=StoppingRule(rtol=rtol, preconditioner='none')).stopping.rtol == rtol
    assert mw.MoMConfig().stopping.rtol == 1e-8


def test_stagnation_window_defaults_to_restart_length():
    from dataclasses import asdict
    from witwin.maxwell.mom.linalg import StoppingRule

    assert asdict(StoppingRule())['stagnation_window'] == 40
    assert asdict(StoppingRule(restart=100))['stagnation_window'] == 100
    assert StoppingRule(restart=100, stagnation_window=None).stagnation_window == 100
    assert StoppingRule(restart=100, stagnation_window=20).stagnation_window == 20
