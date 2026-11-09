"""Independent-axis convergence cannot be hidden by cancellation of errors."""
import pytest
import torch
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom import dyadic_green as dg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def medium():
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    return TensorMedium(3e8, (2+.8j)*eye, (1.3+.5j)*eye)


def test_opposite_axis_errors_cannot_falsely_cancel(monkeypatch):
    m = medium()
    def controlled(points, medium, *, radial_order, azimuthal_order, restore_static=True, mode_cache=None):
        value = medium.epsilon_r.new_zeros((len(points), 6, 6))
        value[:, 0, 0] = 1+1/radial_order-1/azimuthal_order
        return value
    monkeypatch.setattr(dg, 'integrate_spatial_green', controlled)
    p = torch.tensor([[.1, 0., 0.]], device='cuda', dtype=torch.float64)
    config = dg.DyadicIntegrationConfig(2, 2, 4, 4, relative_tolerance=1e-6)
    with pytest.raises(dg.DyadicConvergenceError) as failure:
        dg.evaluate_spatial_green(p, m, config)
    receipt = failure.value.receipt
    assert receipt['failure_reason'] == 'spectral_order_budget_exhausted'
    assert receipt['checks'][0]['radial_max_absolute_error'] == .25
    assert receipt['checks'][0]['azimuthal_max_absolute_error'] == .25
    assert receipt['converged'] is False


def test_only_failing_axis_is_refined_and_orders_are_reused(monkeypatch):
    m = medium()
    called = []
    def controlled(points, medium, *, radial_order, azimuthal_order, restore_static=True, mode_cache=None):
        called.append((radial_order, azimuthal_order))
        value = medium.epsilon_r.new_zeros((len(points), 6, 6))
        value[:, 0, 0] = 1+radial_order**-4
        return value
    monkeypatch.setattr(dg, 'integrate_spatial_green', controlled)
    p = torch.tensor([[.1, 0., 0.]], device='cuda', dtype=torch.float64)
    _, receipt = dg.evaluate_spatial_green(p, m, dg.DyadicIntegrationConfig(2, 2, 16, 4, relative_tolerance=.01))
    assert receipt['converged'] and receipt['final_orders'] == [8, 4]
    assert len(called) == len(set(called)) == 5
    assert receipt['certified_error_bound'] is False


def test_actual_spectral_budget_failure_retains_both_measured_errors():
    p = torch.tensor([[.1, -.2, .15]], device='cuda', dtype=torch.float64)
    with pytest.raises(dg.DyadicConvergenceError) as failure:
        dg.evaluate_spatial_green(p, medium(), dg.DyadicIntegrationConfig(2, 2, 4, 4, relative_tolerance=1e-12))
    receipt = failure.value.receipt
    assert receipt['failure_reason'] == 'spectral_order_budget_exhausted'
    assert receipt['checks'] and receipt['checks'][0]['radial_max_absolute_error'] > 0
    assert receipt['evaluated_orders'] == [[4, 2], [2, 4], [4, 4]]


@pytest.mark.parametrize('case', ['threshold', 'reference', 'radial', 'azimuthal', 'finite'])
def test_finite_spectra_cannot_admit_overflowed_metrics(monkeypatch, case):
    m = medium()
    def controlled(points, medium, *, radial_order, azimuthal_order, **kwargs):
        value = medium.epsilon_r.new_zeros((len(points), 6, 6))
        entry = 1e200 if case == 'reference' else 2.
        if case == 'radial' and (radial_order, azimuthal_order) == (2, 4):
            entry = 1e200
        if case == 'azimuthal' and (radial_order, azimuthal_order) == (4, 2):
            entry = 1e200
        value[:, 0, 0] = entry
        assert torch.isfinite(value).all()
        return value
    monkeypatch.setattr(dg, 'integrate_spatial_green', controlled)
    p = torch.tensor([[.1, 0., 0.]], device='cuda', dtype=torch.float64)
    config = dg.DyadicIntegrationConfig(2, 2, 4, 4,
        relative_tolerance=1e308 if case == 'threshold' else 1e-6,
        absolute_tolerance=0.)
    if case == 'finite':
        value, receipt = dg.evaluate_spatial_green(p, m, config)
        assert receipt['converged'] and receipt['checks'][0]['finite_acceptance_metrics']
        assert torch.equal(value[:, 0, 0], p.new_tensor([2.]).to(value.dtype))
    else:
        with pytest.raises(dg.DyadicConvergenceError) as failure:
            dg.evaluate_spatial_green(p, m, config)
        receipt = failure.value.receipt
        assert not receipt['converged']
        assert receipt['failure_reason'] == 'nonfinite_spectral_acceptance_metrics'
        check = receipt['checks'][0]
        assert not check['finite_acceptance_metrics']
        assert not check['radial_pass'] and not check['azimuthal_pass']
