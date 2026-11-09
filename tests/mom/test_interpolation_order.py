"""Uniform order and cumulative table admission before numerical setup."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.cluster import build_support_octree
from witwin.maxwell.mom.interpolation_error import KernelErrorBudget, scalar_box_interpolation_bound
from witwin.maxwell.mom.interpolation_order import plan_uniform_kernel_order, KernelOrderSelectionError
from witwin.maxwell.mom.physics import compile_medium

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def two_boxes():
    centers = torch.tensor([[-.5, 0., 0.], [.5, 0., 0.]], device='cuda', dtype=torch.float64)
    return build_support_octree(centers-.01, centers+.01, leaf_size=1)


def test_plan_selects_passing_order_without_nodes_and_counts_all_tables(monkeypatch):
    tree = two_boxes()
    medium = compile_medium(mw.Material(), 3e8)
    budget = KernelErrorBudget(1e-5, 1e-4)
    def forbid(*args, **kwargs):
        raise AssertionError('Planning must not allocate interpolation tensor grids')
    monkeypatch.setattr(torch, 'meshgrid', forbid)
    plan = plan_uniform_kernel_order(((tree, medium),), budget, min_order=2, max_order=8)
    selected = plan['selected_order']
    assert 2 < selected <= 8
    assert plan['kernel_table_bytes'] == 2*4*16*selected**6
    assert all(a['maximum_green_bound'] > budget.green_absolute or
               a['maximum_gradient_component_bound'] > budget.gradient_component_absolute for a in plan['attempts'][:-1])
    assert plan['attempts'][-1]['maximum_green_bound'] <= budget.green_absolute
    assert plan['attempts'][-1]['maximum_gradient_component_bound'] <= budget.gradient_component_absolute
    with pytest.raises(KernelOrderSelectionError) as failure:
        plan_uniform_kernel_order(((tree, medium),), budget, min_order=2, max_order=2)
    assert failure.value.receipt['reason'] == 'maximum_order'
    with pytest.raises(KernelOrderSelectionError) as failure:
        plan_uniform_kernel_order(((tree, medium),), budget, min_order=2, max_order=8, kernel_budget_bytes=1)
    assert failure.value.receipt['reason'] == 'kernel_table_budget'


def test_system_automatic_order_matches_plan_and_early_failure(monkeypatch):
    from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh, SurfaceSystem
    from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
    from witwin.maxwell.mom.direct_system import DirectSystemActions
    from witwin.maxwell.mom.operator import IntegrationConfig
    medium = compile_medium(mw.Material(), 3e8)
    entries = [(i, str(i), RWGBasis(sphere_mesh(radius=.01, center=(x, 0., 0.), subdivisions=0, device='cuda')), medium)
               for i, x in enumerate((-.5, .5))]
    provider = ScalarSystemBlocks(SurfaceSystem.create(entries), medium, IntegrationConfig(order=2, singular_order=3))
    budget = KernelErrorBudget(1e-5, 1e-4)
    operator = DirectSystemActions(provider, order=2, max_order=8, leaf_size=30,
                                   quadrature_order=4, kernel_error_budget=budget)
    assert operator.kernel_bytes == operator.order_plan['kernel_table_bytes']
    assert operator.exterior.test.boxes[0].order == operator.order_plan['selected_order']
    for _, _, kernel in operator.exterior.far:
        bound = kernel.truncation_bound()
        assert bound['green_absolute_bound'] <= budget.green_absolute
        assert bound['gradient_component_absolute_bound'] <= budget.gradient_component_absolute
    import witwin.maxwell.mom.direct_system as module
    def forbid(*args, **kwargs):
        raise AssertionError('Planning failure must precede moments and all operator assembly')
    monkeypatch.setattr(module, 'NestedRWGMoments', forbid)
    with pytest.raises(KernelOrderSelectionError):
        DirectSystemActions(provider, order=2, max_order=2, kernel_error_budget=budget)


def test_collapsed_axis_node_count_and_point_bound():
    centers = torch.tensor([[0., 0., 0.], [1., 0., 0.]], device='cuda', dtype=torch.float64)
    widths = torch.tensor([.01, 0., 0.], device='cuda', dtype=torch.float64)
    tree = build_support_octree(centers-widths, centers+widths, leaf_size=1)
    medium = compile_medium(mw.Material(), 3e8)
    plan = plan_uniform_kernel_order(((tree, medium),), KernelErrorBudget(1., 10.), min_order=2, max_order=2)
    assert plan['kernel_table_bytes'] == 2*4*16*2**2
    bound = scalar_box_interpolation_bound(centers[0], centers[0], centers[1], centers[1], medium.k, 7, 7)
    assert bound['active_axes'] == 0 and bound['green_absolute_bound'] == 0


def test_nonfinite_low_order_bound_does_not_abort_later_finite_order():
    from witwin.maxwell.mom.interpolation_error import KernelBoundUnavailable
    lower = torch.tensor([[0., 0., 0.], [1.002, 0., 0.]], device='cuda', dtype=torch.float64)
    upper = lower+torch.tensor([.002, 0., 0.], device='cuda', dtype=torch.float64)
    tree = build_support_octree(lower, upper, leaf_size=1)
    medium = compile_medium(mw.Material(), 50000*299792458/(2*torch.pi))
    with pytest.raises(KernelBoundUnavailable):
        scalar_box_interpolation_bound(lower[0], upper[0], lower[1], upper[1], medium.k, 1, 1)
    plan = plan_uniform_kernel_order(((tree, medium),), KernelErrorBudget(1e300, 1e300),
                                     min_order=1, max_order=220)
    assert plan['selected_order'] > 1
    assert not plan['attempts'][0]['all_bounds_finite']
    assert plan['attempts'][-1]['all_bounds_finite']
    assert plan['kernel_table_bytes'] == 2*4*16*plan['selected_order']**2
    print('overflow recovery selected order', plan['selected_order'])
