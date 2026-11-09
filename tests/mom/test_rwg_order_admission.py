"""Whole-system ideal RWG error admission before interpolation construction."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh, SurfaceSystem
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.direct_system import DirectSystemActions
from witwin.maxwell.mom.interpolation_error import KernelErrorBudget
from witwin.maxwell.mom.interpolation_order import KernelOrderSelectionError
from witwin.maxwell.mom.rwg_interpolation_error import system_interpolation_bound

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_system_rwg_budget_drives_order_and_rejects_underintegration(monkeypatch):
    exterior = compile_medium(mw.Material(), 3e8)
    interior = compile_medium(mw.Material(eps_r=1.5, mu_r=1.1), 3e8)
    entries = [(i, str(i), RWGBasis(sphere_mesh(radius=.01, center=(x, 0., 0.), subdivisions=0, device='cuda')), interior)
               for i, x in enumerate((-.5, .5))]
    provider = ScalarSystemBlocks(SurfaceSystem.create(entries), exterior, IntegrationConfig(order=2, singular_order=3))
    loose_kernel = KernelErrorBudget(1., 10.)
    budget = 1e-6
    operator = DirectSystemActions(provider, order=2, max_order=8, leaf_size=30,
        kernel_error_budget=loose_kernel, rwg_absolute_budget=budget)
    plan = operator.order_plan
    print('RWG selected order', plan['selected_order'], 'planned bound', plan['attempts'][-1]['rwg_absolute_bound'])
    assert plan['selected_order'] > 2
    assert all(not a['rwg_absolute_budget_passed'] for a in plan['attempts'][:-1])
    assert plan['attempts'][-1]['rwg_absolute_bound'] <= budget
    actual = system_interpolation_bound(operator)
    assert actual['absolute_bound'] == pytest.approx(plan['attempts'][-1]['rwg_absolute_bound'], rel=2e-12)
    assert not actual['implemented_operator_error_certified']
    assert operator.receipt()['moment_quadrature']['exterior']['polynomial_exact_in_exact_arithmetic']
    import witwin.maxwell.mom.direct_system as module
    def forbid(*args, **kwargs):
        raise AssertionError('Failed admission must precede interpolation moment construction')
    monkeypatch.setattr(module, 'NestedRWGMoments', forbid)
    with pytest.raises(KernelOrderSelectionError):
        DirectSystemActions(provider, order=2, max_order=2, kernel_error_budget=loose_kernel, rwg_absolute_budget=budget)
    with pytest.raises(ValueError, match='polynomial-exact'):
        DirectSystemActions(provider, order=2, max_order=8, quadrature_order=1,
            kernel_error_budget=loose_kernel, rwg_absolute_budget=budget)
    with pytest.raises(ValueError, match='automatic order'):
        DirectSystemActions(provider, rwg_absolute_budget=budget)
