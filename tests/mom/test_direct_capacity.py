"""Direct retained-action limits reject before moments or weak-block assembly."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.direct_system import DirectSystemActions
from witwin.maxwell.mom.direct_capacity import DirectCapacityError, direct_action_tensors, storage_bytes

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_action_storage_matches_plan_and_rejects_before_numerical_setup(monkeypatch):
    medium = compile_medium(mw.Material(), 3e8)
    body = compile_medium(mw.Material(eps_r=2.), 3e8)
    system = SurfaceSystem.create([(i, str(i), RWGBasis(sphere_mesh(radius=.01,
        center=(x, 0., 0.), subdivisions=0, device='cuda')), body) for i, x in enumerate((-.5, .5))])
    provider = ScalarSystemBlocks(system, medium, IntegrationConfig(order=2, singular_order=3))
    actions = DirectSystemActions(provider, order=3, leaf_size=30)
    plan = actions.capacity_plan
    actual = storage_bytes(direct_action_tensors(actions))
    assert plan['admitted'] and plan['planned_bytes'] == actual
    assert sum(plan['categories'].values()) == actual
    assert plan['categories']['kernels'] == actions.kernel_bytes
    assert all(value > 0 for value in plan['categories'].values())
    print('retained action storage', actual, plan['categories'])
    plan['planned_bytes'] = -1
    assert actions.capacity_plan['planned_bytes'] == actual
    import witwin.maxwell.mom.direct_system as construction
    def forbidden(*args, **kwargs):
        raise AssertionError('Capacity rejection must precede moments and weak quadrature')
    monkeypatch.setattr(construction, 'NestedRWGMoments', forbidden)
    monkeypatch.setattr(provider, 'exterior_block', forbidden)
    with pytest.raises(DirectCapacityError) as error:
        DirectSystemActions(provider, order=3, leaf_size=30, action_retained_budget_bytes=actual-1)
    assert error.value.receipt['planned_bytes'] == actual
    assert not error.value.receipt['admitted']
    assert not error.value.receipt['peak_memory_qualified']


def test_storage_accounting_counts_full_storage_once_for_aliases():
    value = torch.zeros(100, device='cuda', dtype=torch.float64)
    assert storage_bytes((value[:1], value[20:40], value)) == 800
