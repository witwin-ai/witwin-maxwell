"""System-wide ideal kernel order planning before numerical multipole setup."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.fmm_system import MultipoleSystemActions
from witwin.maxwell.mom.fmm_order import MultipoleOrderError

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('storage', ['dense', 'compact'])
def test_automatic_order_actual_tables_and_preallocation_rejection(monkeypatch, storage):
    exterior = compile_medium(mw.Material(), 3e8)
    interior = compile_medium(mw.Material(eps_r=2.), 3e8)
    system = SurfaceSystem.create([(i, str(i), RWGBasis(sphere_mesh(radius=.01,
        center=(x, 0., 0.), subdivisions=0, device='cuda')), interior) for i, x in enumerate((-.5, .5))])
    provider = ScalarSystemBlocks(system, exterior, IntegrationConfig(order=2, singular_order=3))
    budget = mw.KernelErrorBudget(1e-6, 1e-5)
    op = MultipoleSystemActions(provider, order=2, max_order=10, kernel_error_budget=budget, leaf_size=30, table_storage=storage)
    plan = op.order_plan
    print('selected FMM order', plan)
    assert 2 < plan['selected_order'] <= 10
    assert plan['table_bytes'] == op.table_bytes
    assert all(not attempt['kernel_pass'] for attempt in plan['attempts'][:-1])
    for region in (op.exterior, *op.interiors):
        assert region.test.order == plan['selected_order']
        for _, _, coupling in region.far:
            bound = coupling.truncation_bound()
            assert bound['green_absolute'] <= budget.green_absolute
            assert bound['gradient_component_absolute'] <= budget.gradient_component_absolute
    plan['selected_order'] = -1
    assert op.order_plan['selected_order'] > 0
    import witwin.maxwell.mom.fmm_system as construction
    def forbidden(*args, **kwargs):
        raise AssertionError('Failed order planning must precede moment construction')
    monkeypatch.setattr(construction, 'NestedRWGMultipoles', forbidden)
    with pytest.raises(MultipoleOrderError) as failure:
        MultipoleSystemActions(provider, order=2, max_order=2, kernel_error_budget=budget, leaf_size=30, table_storage=storage)
    assert failure.value.receipt['reason'] == 'maximum_order_or_unavailable_bound'
    with pytest.raises(MultipoleOrderError) as failure:
        MultipoleSystemActions(provider, order=2, max_order=10, kernel_error_budget=budget, leaf_size=30, table_budget_bytes=1, table_storage=storage)
    assert failure.value.receipt['reason'] == 'table_budget'


def test_nested_budget_config_roundtrip():
    config = mw.FMMConfig(order=2, max_order=10, kernel_error_budget=mw.KernelErrorBudget(1e-6, 1e-5))
    assert mw.FMMConfig.restore(config.payload()) == config
    with pytest.raises(ValueError, match='Automatic'):
        mw.FMMConfig(order=2, max_order=10)


def test_planner_tracks_frequency_and_unavailable_contours():
    from witwin.maxwell.mom.cluster import build_support_octree
    from witwin.maxwell.mom.fmm_order import plan_multipole_order
    lower = torch.tensor([[0., 0., 0.], [1., 0., 0.]], device='cuda', dtype=torch.float64)
    tree = build_support_octree(lower, lower+.02, leaf_size=1)
    budget = mw.KernelErrorBudget(1e-6, 1e-5)
    low = plan_multipole_order(((tree, compile_medium(mw.Material(), 3e8)),), budget, max_order=14)
    high = plan_multipole_order(((tree, compile_medium(mw.Material(), 9e8)),), budget, max_order=14)
    assert high['selected_order'] > low['selected_order']
    print('frequency-selected orders', low['selected_order'], high['selected_order'])
    # Two spheres with ratio .45 pass geometric .5, but the conservative
    # analytic contour family ends at .4 and must never report a bound pass.
    lower = torch.tensor([[0., 0., 0.], [1., 0., 0.]], device='cuda', dtype=torch.float64)
    tree = build_support_octree(lower, lower+(.45/(3**.5)), leaf_size=1)
    with pytest.raises(MultipoleOrderError) as failure:
        plan_multipole_order(((tree, compile_medium(mw.Material(), 3e8)),), budget, max_order=2)
    assert all(attempt['unavailable_bounds'] == 2 and not attempt['kernel_pass']
               for attempt in failure.value.receipt['attempts'])
