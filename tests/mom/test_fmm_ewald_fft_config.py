"""Explicit Ewald FFT identity and rejection before expensive allocation."""
import copy
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.linalg import StoppingRule


def test_ewald_fft_canonical_identity_and_public_precision():
    # This version's identity remains fixed when unrelated free-space options
    # are added to the shared configuration dataclass.
    canonical_fields = {
        'order', 'table_storage', 'shift_storage', 'near_storage', 'kernel_bound_rule',
        'near_coefficient_workspace_budget_bytes', 'near_coefficient_cache_budget_bytes',
        'm2l_row_batch', 'near_action_batch', 'max_order', 'rwg_absolute_budget',
        'kernel_error_budget', 'quadrature_order', 'leaf_size', 'separation_ratio',
        'action_retained_budget_bytes', 'table_budget_bytes', 'near_budget_bytes',
        'projection_budget_bytes', 'solve_dual_tolerance', 'original_sample_tolerance',
        'leaf_factor_kernel', 'leaf_factor_max_edges', 'leaf_factor_budget_bytes',
    }
    omitted_defaults = {
        'far_action_batch': 1,
        'expansion_center_policy': 'support_box_midpoint_v1',
    }
    rules = ('bivariate_cauchy_v1', 'real_support_min_v1', 'positive_tail_min_v1',
        'first_wave_min_v1', 'direct_positive_tail_min_v1', 'direct_first_wave_min_v1')
    for rule in rules:
        for shift in ('dense', 'separable_v1'):
            cfg = mw.FMMConfig(near_storage='full_matrix',table_storage='fft_ewald_v1', kernel_bound_rule=rule, shift_storage=shift)
            payload = cfg.payload()
            assert payload['version'] == 'cartesian_fmm_ewald_fft_config_v1'
            assert cfg.near_storage == 'full_matrix'
            assert set(payload['parameters']) == canonical_fields
            for name, default in omitted_defaults.items():
                assert getattr(cfg, name) == default
                bad = copy.deepcopy(payload); bad['parameters'][name] = default
                with pytest.raises(ValueError): mw.FMMConfig.restore(bad)
            assert mw.FMMConfig.restore(payload) == cfg
            for version in ('cartesian_fmm_fft_config_v1', 'cartesian_fmm_config_v7',
                    'cartesian_fmm_separable_config_v1', 'cartesian_fmm_real_support_config_v1',
                    'cartesian_fmm_positive_tail_config_v1', 'cartesian_fmm_first_wave_config_v1',
                    'cartesian_fmm_direct_positive_tail_config_v1', 'cartesian_fmm_direct_first_wave_config_v1'):
                bad = copy.deepcopy(payload); bad['version'] = version
                with pytest.raises(ValueError): mw.FMMConfig.restore(bad)
            for storage in ('dense', 'compact', 'fft_v1'):
                bad = copy.deepcopy(payload); bad['parameters']['table_storage'] = storage
                with pytest.raises(ValueError): mw.FMMConfig.restore(bad)
    for order in (13,):
        with pytest.raises(ValueError, match='native order'):
            mw.FMMConfig(near_storage='full_matrix',table_storage='fft_ewald_v1', order=order)
    with pytest.raises(ValueError, match='native order'):
        mw.FMMConfig(near_storage='full_matrix',table_storage='fft_ewald_v1', max_order=13, kernel_error_budget=mw.KernelErrorBudget(1e-6, 1e-5))
    periodic = mw.MoMPeriodicConfig(((1., 0., 0.),))
    kw = dict(operator_backend='fmm', riesz_storage='local_rwg_v1',
        fmm=mw.FMMConfig(near_storage='full_matrix',table_storage='fft_ewald_v1'), stopping=StoppingRule(rtol=1e-9, preconditioner='rwg_diagonal'))
    valid = mw.MoMConfig(periodic=periodic, **kw)
    assert valid.dtype == torch.complex128
    with pytest.raises(ValueError, match='complex128'):
        mw.MoMConfig(periodic=periodic, dtype=torch.complex64, **kw)
    with pytest.raises(ValueError, match='explicit periodic'):
        mw.MoMConfig(**kw)
    with pytest.raises(NotImplementedError, match='free space'):
        mw.MoMConfig(periodic=periodic, **dict(kw, fmm=mw.FMMConfig(table_storage='fft_v1')))
    with pytest.raises(NotImplementedError, match='fixed order'):
        mw.MoMConfig(periodic=periodic, **dict(kw, fmm=replace(kw['fmm'], max_order=8,
            kernel_error_budget=mw.KernelErrorBudget(1e-6, 1e-5))))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_ewald_fft_native_binding_and_capacity_guards_precede_projections(monkeypatch):
    from test_fmm_ewald_fft_integration import _provider
    import witwin.maxwell.mom.fmm_periodic_system as module
    from witwin.maxwell.mom.fmm_periodic import PeriodicImageM2L
    from witwin.maxwell.mom.fmm_periodic_regional import PeriodicMultipoleRegionalOperator
    from witwin.maxwell.mom.fmm_periodic_order import plan_periodic_order, plan_periodic_parameters
    from witwin.maxwell.mom.fmm_periodic_capacity import plan_periodic_storage
    from witwin.maxwell.mom.fmm_capacity import MultipoleCapacityError
    from witwin.maxwell.mom.fmm_system import MultipoleSystemActions
    provider = _provider()
    kernel = provider.kernel
    def forbidden(*args, **kwargs): raise AssertionError('Rejected construction allocated projections or coefficients')
    monkeypatch.setattr(module, 'NestedRWGMultipoles', forbidden)
    cls = module.PeriodicMultipoleSystemActions
    kw = dict(ewald_kernel=kernel, table_storage='fft_ewald_v1')
    with pytest.raises(ValueError, match='float64'):
        cls(provider, kernel.lattice, order=13, **kw)
    with pytest.raises(MultipoleCapacityError):
        cls(provider, kernel.lattice, action_retained_budget_bytes=1, **kw)
    with pytest.raises(ValueError, match='matching original'):
        cls(provider, None, **kw)
    for key, value in (('interior_kernel_error_budget', mw.KernelErrorBudget(1e-6, 1e-5)),
            ('exterior_kernel_error_budget', mw.KernelErrorBudget(1e-6, 1e-5)),
            ('remote_kernel_error_budget', mw.KernelErrorBudget(1e-6, 1e-5)), ('max_order', 8), ('max_shells', 3)):
        with pytest.raises(NotImplementedError, match='Ewald'):
            cls(provider, kernel.lattice, **dict(kw, **{key: value}))
    with pytest.raises(ValueError, match='cutoff'):
        plan_periodic_storage(provider, None, kernel.lattice, order=6, shells=3,
            separation_ratio=.25, budget_bytes=1024, table_storage='fft_ewald_v1', ewald=True)
    with pytest.raises(ValueError, match='table_storage'):
        MultipoleSystemActions(provider, table_storage='fft_ewald_v1')
    for constructor, args in ((PeriodicImageM2L, (None, None, 0., 0., 2.)),
            (PeriodicMultipoleRegionalOperator, (None, None, None, None)), (cls, (None, None))):
        with pytest.raises(NotImplementedError, match='FFT'):
            constructor(*args, table_storage='fft_ewald_v1')
    for func, params in ((plan_periodic_order, dict(min_order=6, max_order=8, shells=2, exterior_budget=None, interior_budget=None)),
            (plan_periodic_parameters, dict(min_shells=1, max_shells=2)),
            (plan_periodic_storage, dict(order=6, shells=2, separation_ratio=.5, budget_bytes=1024))):
        with pytest.raises(NotImplementedError, match='FFT'):
            func(None, None, None, table_storage='fft_ewald_v1', **params)
