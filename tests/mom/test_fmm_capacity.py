"""Complete action storage plan agrees with actual distinct CUDA storage."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.fmm_system import MultipoleSystemActions
from witwin.maxwell.mom.fmm_capacity import multipole_action_tensors, storage_bytes, MultipoleCapacityError

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('storage', ['dense', 'compact'])
def test_capacity_matches_all_owned_storages_and_rejects_before_moments(monkeypatch, storage):
    medium = compile_medium(mw.Material(), 3e8)
    body_medium = compile_medium(mw.Material(eps_r=2.), 3e8)
    system = SurfaceSystem.create([(i, str(i), RWGBasis(sphere_mesh(radius=.01,
        center=(x, 0., 0.), subdivisions=0, device='cuda')), body_medium) for i, x in enumerate((-.5, .5))])
    provider = ScalarSystemBlocks(system, medium, IntegrationConfig(order=2, singular_order=3))
    op = MultipoleSystemActions(provider, order=3, leaf_size=30, table_storage=storage)
    plan = op.capacity_plan
    actual = storage_bytes(multipole_action_tensors(op))
    assert actual == plan['planned_bytes'] == sum(plan['categories'].values())
    assert all(v > 0 for v in plan['categories'].values())
    assert plan['categories']['m2l'] == op.table_bytes
    assert plan['categories']['near'] == op.near_bytes
    assert sum(plan['categories'][k] for k in ('centers_radii', 'leaf_maps', 'leaf_moments', 'shifts')) == op.projection_bytes
    print('multipole retained storage', actual, plan['categories'])
    plan['planned_bytes'] = -1
    assert op.capacity_plan['planned_bytes'] == actual
    import witwin.maxwell.mom.fmm_system as construction
    def forbidden(*args, **kwargs):
        raise AssertionError('Capacity failure must precede numerical moments')
    monkeypatch.setattr(construction, 'NestedRWGMultipoles', forbidden)
    with pytest.raises(MultipoleCapacityError) as error:
        MultipoleSystemActions(provider, order=3, leaf_size=30, action_retained_budget_bytes=actual-1, table_storage=storage)
    assert not error.value.receipt['admitted']
    assert error.value.receipt['planned_bytes'] == actual
    assert not error.value.receipt['peak_memory_qualified']
    if storage == 'dense':
        first, second = (entry[2] for entry in op.exterior.far)
        assert first.green.untyped_storage().data_ptr() == second.green.untyped_storage().data_ptr()
        assert first.gradient.untyped_storage().data_ptr() == second.gradient.untyped_storage().data_ptr()
        first.green.add_(1)
        with pytest.raises(ValueError, match='changed'):
            op.assert_current()
    else:
        first = op.exterior.far[0][2]
        assert first.green is None and first.gradient is None
        first._compact.coefficients.add_(1)
        with pytest.raises(ValueError, match='changed'):
            op.assert_current()
