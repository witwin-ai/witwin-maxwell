"""Direct construction configuration must preserve every admission parameter."""
import copy
from dataclasses import replace
import pytest
import witwin.maxwell as mw


def test_direct_configuration_roundtrip_preserves_nested_budgets():
    config = mw.DirectH2Config(order=3, max_order=8, quadrature_order=12,
        leaf_size=24, admissibility=.8, kernel_budget_bytes=2**24,
        kernel_error_budget=mw.KernelErrorBudget(1e-5, 2e-4),
        rwg_absolute_budget=1e-6, solve_dual_tolerance=2e-9,
        original_sample_tolerance=3e-9)
    payload = config.payload()
    assert mw.DirectH2Config.restore(payload) == config
    assert config.construction()['kernel_error_budget'] is config.kernel_error_budget
    assert config.solve_config().original_sample_tolerance == 3e-9
    for field in config.__dataclass_fields__:
        corrupted = copy.deepcopy(payload)
        del corrupted['parameters'][field]
        with pytest.raises(ValueError, match='incomplete'):
            mw.DirectH2Config.restore(corrupted)
    payload['parameters']['kernel_error_budget']['green_absolute'] = 4e-5
    assert config.kernel_error_budget.green_absolute == 1e-5
    assert mw.DirectH2Config.restore(payload) != config


@pytest.mark.parametrize('change', [
    {'order': True}, {'order': 0}, {'max_order': 0}, {'quadrature_order': False},
    {'leaf_size': 1.5}, {'kernel_budget_bytes': -1}, {'admissibility': float('nan')},
    {'solve_dual_tolerance': float('inf')}, {'original_sample_tolerance': False},
    {'kernel_error_budget': {}}, {'max_order': 8}, {'rwg_absolute_budget': 1e-6},
    {'max_order': 2, 'kernel_error_budget': mw.KernelErrorBudget(1., 1.)},
])
def test_direct_configuration_rejects_invalid_admission(change):
    with pytest.raises((ValueError, TypeError)):
        replace(mw.DirectH2Config(), **change)
