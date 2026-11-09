"""Periodic order planning retains failed candidates and respects all budgets."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.periodic import LatticeSpec
from witwin.maxwell.mom.interpolation_error import KernelErrorBudget
from witwin.maxwell.mom.fmm_periodic_system import PeriodicMultipoleSystemActions
from witwin.maxwell.mom.fmm_periodic_order import PeriodicOrderError

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def fixture():
    exterior = compile_medium(mw.Material(eps_r=1.5, sigma_e=.2), 3e8)
    interior = compile_medium(mw.Material(eps_r=2., mu_r=1.2), 3e8)
    system = SurfaceSystem.create([(i, str(i), RWGBasis(sphere_mesh(radius=.01,
        center=(x, 0., 0.), subdivisions=0, device='cuda')), interior) for i, x in enumerate((-.2, .2))])
    provider = ScalarSystemBlocks(system, exterior, IntegrationConfig(order=4, singular_order=3))
    lattice = LatticeSpec(torch.tensor([[1., 0., 0.]], device='cuda', dtype=torch.float64),
                          torch.tensor([.37, 0., 0.], device='cuda', dtype=torch.float64))
    return provider, lattice


def test_periodic_order_admits_complete_kernel_and_capacity_before_construction(monkeypatch):
    provider, lattice = fixture()
    budget = KernelErrorBudget(1e-6, 1e-5)
    kwargs = dict(order=2, max_order=12, shells=3, separation_ratio=.1, leaf_size=30,
        table_storage='compact', exterior_kernel_error_budget=budget, interior_kernel_error_budget=budget)
    op = PeriodicMultipoleSystemActions(provider, lattice, **kwargs)
    plan = op.order_plan
    print('periodic order', plan['selected_order'], [(a['order'], a['exterior_bound']['green_absolute'],
        a['exterior_bound']['gradient_component_absolute'], a['resource_pass']) for a in plan['attempts']])
    selected = plan['selected_order']
    assert selected == 11
    assert all(not attempt['exterior_pass'] for attempt in plan['attempts'][:-1])
    assert all(plan['attempts'][-1][key] for key in ('exterior_pass', 'interior_pass', 'remote_pass', 'resource_pass'))
    assert op.exterior.test.order == selected
    assert all(p.order == selected for p in op.projections)
    assert plan['capacity_plan'] == op.capacity_plan
    assert op.provider.shells == plan['shells'] == 3
    plan['selected_order'] = -1
    assert op.order_plan['selected_order'] == selected
    import witwin.maxwell.mom.fmm_periodic_system as construction
    def forbidden(*args, **kwargs):
        raise AssertionError('Order rejection must precede numerical moments')
    monkeypatch.setattr(construction, 'NestedRWGMultipoles', forbidden)
    with pytest.raises(PeriodicOrderError) as failure:
        PeriodicMultipoleSystemActions(provider, lattice, **{**kwargs, 'max_order': selected-1})
    assert failure.value.receipt['reason'] == 'maximum_order_or_unavailable_bound'
    with pytest.raises(PeriodicOrderError) as failure:
        PeriodicMultipoleSystemActions(provider, lattice, **kwargs, action_retained_budget_bytes=1)
    assert failure.value.receipt['reason'] == 'storage_budget'
    assert len(failure.value.receipt['attempts']) == 1
    op._order_plan_text = '{}'
    with pytest.raises(ValueError, match='changed'):
        op.assert_current()


def test_periodic_order_rejects_root_setup_table_deficit_before_moments(monkeypatch):
    from witwin.maxwell.mom.fmm_storage import m2l_storage_bytes
    original, lattice = fixture()
    body = original.system.bodies[0]
    provider = ScalarSystemBlocks(SurfaceSystem.create([(0, 'only', body.basis, body.interior)]),
        original.exterior, original.config)
    retained = m2l_storage_bytes(2, 8, 'compact')
    import witwin.maxwell.mom.fmm_periodic_system as construction
    def forbidden(*args, **kwargs):
        raise AssertionError('Root setup budget must reject before numerical moments')
    monkeypatch.setattr(construction, 'NestedRWGMultipoles', forbidden)
    with pytest.raises(PeriodicOrderError) as failure:
        PeriodicMultipoleSystemActions(provider, lattice, order=2, max_order=2, shells=3,
            separation_ratio=.1, leaf_size=30, table_storage='compact', table_budget_bytes=retained,
            exterior_kernel_error_budget=KernelErrorBudget(1., 1.),
            interior_kernel_error_budget=KernelErrorBudget(1., 1.))
    attempt = failure.value.receipt['attempts'][0]
    assert attempt['capacity_plan']['categories']['m2l'] == retained
    assert attempt['capacity_plan']['admitted']
    assert attempt['exterior_pass'] and attempt['interior_pass']
    assert attempt['root_setup_table_bytes'] > retained
    assert not attempt['resource_pass']
