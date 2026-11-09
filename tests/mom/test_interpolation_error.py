"""Kernel truncation bounds, separate from quadrature/rounding qualification."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.interpolation import InterpolationBox
from witwin.maxwell.mom.interpolation_kernel import InterpolationKernel
from witwin.maxwell.mom.interpolation_error import scalar_kernel_interpolation_bound, KernelErrorBudget, KernelInterpolationError
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.kernels import FREE_SPACE

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def box(center, order):
    center = torch.tensor(center, device='cuda', dtype=torch.float64)
    return InterpolationBox(center-.01, center+.01, order)


def test_bound_dominates_independent_kernel_samples_and_refines():
    medium = compile_medium(mw.Material(eps_r=1.2, sigma_e=.0005), 3e8)
    bounds = []
    for order in (2, 4, 6):
        a, b = box((-.5, 0., 0.), order), box((.5, 0., 0.), order)
        kernel = InterpolationKernel(a, b, medium)
        receipt = kernel.truncation_bound()
        bounds.append(receipt['green_absolute_bound'])
        x = a.lower+torch.rand((101, 3), device='cuda', dtype=torch.float64)*(a.upper-a.lower)
        y = b.lower+torch.rand((101, 3), device='cuda', dtype=torch.float64)*(b.upper-b.lower)
        left, right = a.evaluate(x).to(torch.complex128), b.evaluate(y).to(torch.complex128)
        green = torch.einsum('qa,ab,qb->q', left, kernel.green, right)
        gradient = torch.einsum('qa,abc,qb->qc', left, kernel.gradient, right)
        exact, derivative = FREE_SPACE.evaluate(x-y, medium.k)
        errors = float((green-exact).abs().max()), float((gradient-derivative).abs().max())
        print(order, receipt, 'sample errors', errors)
        assert errors[0] <= receipt['green_absolute_bound']
        assert errors[1] <= receipt['gradient_component_absolute_bound']
        assert not receipt['floating_point_certified'] and not receipt['operator_error_certified']
    assert bounds[2] < bounds[1] < bounds[0]


def test_budget_rejects_before_kernel_allocation(monkeypatch):
    a, b = box((-.5, 0., 0.), 2), box((.5, 0., 0.), 2)
    medium = compile_medium(mw.Material(), 3e8)
    def forbid(*args, **kwargs):
        raise AssertionError('A failed analytic budget must reject before tables are allocated')
    monkeypatch.setattr(torch, 'empty', forbid)
    with pytest.raises(KernelInterpolationError) as caught:
        InterpolationKernel(a, b, medium, error_budget=KernelErrorBudget(1e-12, 1e-12))
    assert caught.value.receipt['green_absolute_bound'] > 1e-12


def test_collapsed_axes_and_invalid_analytic_domains():
    point = torch.tensor([0., 0., 0.], device='cuda', dtype=torch.float64)
    a = InterpolationBox(point, point, 3)
    b = InterpolationBox(point+1, point+1, 3)
    result = scalar_kernel_interpolation_bound(a, b, 2.)
    assert result['active_axes'] == 0 and result['green_absolute_bound'] == 0
    with pytest.raises(ValueError, match='strictly separated'):
        scalar_kernel_interpolation_bound(a, a, 2.)
    with pytest.raises(ValueError, match='passive'):
        scalar_kernel_interpolation_bound(a, b, 2.-1j)


def test_log_bound_preserves_scales_after_exponential_underflow():
    def point(x):
        return torch.tensor([x, 0., 0.], device='cuda', dtype=torch.float64)
    a = InterpolationBox(point(0.), point(2e-102), 1)
    b = InterpolationBox(point(1.02e-100), point(1.04e-100), 1)
    result = scalar_kernel_interpolation_bound(a, b, 8e102j)
    # Independent real-axis closed form, combining scales before exponentiation.
    r = torch.tensor([1e-100, 1.02e-100], device='cuda', dtype=torch.float64)
    log_g = -8e102*r-torch.log(4*torch.pi*r)
    g = torch.exp(log_g)
    derivative = torch.exp(log_g+torch.log(8e102+1/r))
    assert abs(float(g[0]-g[1])) > 1e-250
    assert abs(float(derivative[0]-derivative[1])) > 1e-147
    assert result['green_absolute_bound'] >= abs(float(g[0]-g[1]))
    assert result['gradient_component_absolute_bound'] >= abs(float(derivative[0]-derivative[1]))
    print('extreme bound', result)


def test_accepted_budget_and_contract_binding():
    a, b = box((-.5, 0., 0.), 6), box((.5, 0., 0.), 6)
    medium = compile_medium(mw.Material(), 3e8)
    budget = KernelErrorBudget(1e-5, 1e-4)
    kernel = InterpolationKernel(a, b, medium, error_budget=budget)
    assert kernel.truncation_bound()['green_absolute_bound'] < budget.green_absolute
    kernel.error_budget = KernelErrorBudget(1e-4, 1e-3)
    with pytest.raises(ValueError, match='changed'):
        kernel.assert_current()


def test_system_forwards_kernel_truncation_budget():
    from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh, SurfaceSystem
    from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
    from witwin.maxwell.mom.direct_system import DirectSystemActions
    from witwin.maxwell.mom.operator import IntegrationConfig
    medium = compile_medium(mw.Material(), 3e8)
    entries = [(i, str(i), RWGBasis(sphere_mesh(radius=.01, center=(x, 0., 0.), subdivisions=0, device='cuda')), medium)
               for i, x in enumerate((-.5, .5))]
    provider = ScalarSystemBlocks(SurfaceSystem.create(entries), medium, IntegrationConfig(order=2, singular_order=3))
    budget = KernelErrorBudget(1e-100, 1e-100)
    with pytest.raises(KernelInterpolationError) as caught:
        DirectSystemActions(provider, order=2, quadrature_order=2, leaf_size=30, kernel_error_budget=budget)
    assert caught.value.receipt['budget']['green_absolute'] == 1e-100
