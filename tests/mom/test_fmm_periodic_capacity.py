"""Periodic storage admission includes translated geometry and shared tree arrays."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.periodic import LatticeSpec
from witwin.maxwell.mom.fmm_periodic_system import PeriodicMultipoleSystemActions
from witwin.maxwell.mom.fmm_periodic_capacity import periodic_action_tensors, storage_bytes
from witwin.maxwell.mom.fmm_capacity import MultipoleCapacityError

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('storage,dimension,center', [('dense', 1, .2), ('compact', 1, .2),
                                                    ('compact', 2, .2), ('compact', 1, .02),
                                                    pytest.param('compact', 3, .2, id='compact-3d-translated')])
def test_periodic_plan_counts_distinct_storage_and_rejects_before_moments(monkeypatch, storage, dimension, center):
    exterior = compile_medium(mw.Material(eps_r=1.5, sigma_e=.02), 3e8)
    interior = compile_medium(mw.Material(eps_r=2., mu_r=1.2), 3e8)
    system = SurfaceSystem.create([(i, str(i), RWGBasis(sphere_mesh(radius=.01,
        center=(x, 0., 0.), subdivisions=0, device='cuda')), interior) for i, x in enumerate((-center, center))])
    provider = ScalarSystemBlocks(system, exterior, IntegrationConfig(order=2, singular_order=3))
    lattice = LatticeSpec(torch.tensor([[1., 0., 0.], [.2, 1.1, 0.], [.1, .2, 1.2]], device='cuda', dtype=torch.float64)[:dimension],
                          torch.tensor([.37, .13 if dimension >= 2 else 0., .07 if dimension == 3 else 0.], device='cuda', dtype=torch.float64))
    kwargs = dict(order=3, shells=1, leaf_size=30, separation_ratio=.1, table_storage=storage)
    op = PeriodicMultipoleSystemActions(provider, lattice, **kwargs)
    plan = op.capacity_plan
    actual = storage_bytes(periodic_action_tensors(op))
    assert actual == plan['planned_bytes'] == sum(plan['categories'].values())
    assert plan['categories']['m2l'] == op.table_bytes
    assert plan['categories']['near'] == op.near_bytes
    assert plan['categories']['projections'] == op.projection_bytes
    assert plan['near_images'] == len(op.exterior.regions)
    if center == .02:
        assert plan['translated_images'] == 0
        assert plan['categories']['translated_geometry'] == 0
        assert op.exterior.remote.far_count > 0
    else:
        assert plan['translated_images'] > 0
        assert plan['categories']['translated_geometry'] > 0
        source = op.exterior.regions[0].source
        assert source.tree.permutation is op.exterior.test.tree.permutation
    print(storage, dimension, center, actual, plan)
    plan['planned_bytes'] = -1
    assert op.capacity_plan['planned_bytes'] == actual
    import witwin.maxwell.mom.fmm_periodic_system as module
    def forbidden(*args, **kwargs):
        raise AssertionError('Capacity rejection must precede numerical moments')
    monkeypatch.setattr(module, 'NestedRWGMultipoles', forbidden)
    with pytest.raises(MultipoleCapacityError) as failure:
        PeriodicMultipoleSystemActions(provider, lattice, action_retained_budget_bytes=actual-1, **kwargs)
    assert failure.value.receipt['planned_bytes'] == actual
    assert not failure.value.receipt['admitted']
    assert not failure.value.receipt['peak_memory_qualified']
    op._capacity_text = '{}'
    with pytest.raises(ValueError, match='changed'):
        op.assert_current()
