"""Structural storage admission before nested H2 allocations."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from witwin.maxwell.mom.cluster import build_support_octree
from witwin.maxwell.mom.lowrank import LowRankBlock
from witwin.maxwell.mom.h2 import H2Config, NestedH2Actions
from witwin.maxwell.mom.h2_capacity import plan_h2_storage, H2CapacityError

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def fixture(dtype=torch.complex128):
    g = torch.Generator(device='cuda').manual_seed(816)
    real = torch.float64 if dtype == torch.complex128 else torch.float32
    points = torch.rand((32, 3), generator=g, device='cuda', dtype=real)
    tree = build_support_octree(points-.001, points+.001, leaf_size=3)
    left = torch.randn((64, 3), device='cuda', dtype=dtype, generator=g)
    right = torch.randn((3, 64), device='cuda', dtype=dtype, generator=g)
    return tree, [(0, 0, LowRankBlock(left, right, {}))]


def test_structural_bound_covers_actual_nested_storage_and_budget_identity():
    tree, records = fixture()
    plan = plan_h2_storage(tree, records, components=2, dtype=torch.complex128, max_rank=8, budget_bytes=2**30)
    config = H2Config(max_rank=8, retained_budget_bytes=plan['retained_bytes_bound'])
    assert H2Config.restore(config.payload()) == config
    op = NestedH2Actions(tree, records, config=config)
    receipt = op.receipt()
    assert receipt['owned_retained_bytes'] <= plan['retained_bytes_bound']
    assert receipt['capacity_plan']['budget_bytes'] == config.retained_budget_bytes
    for name, basis in (('test_basis', op.test_basis), ('trial_basis', op.trial_basis)):
        assert all(actual <= bound for actual, bound in zip(basis.ranks, plan[name]['rank_bounds']))
        assert basis.receipt()['retained_bytes'] <= plan[name]['retained_bytes_bound']
    x = torch.randn(64, device='cuda', dtype=torch.complex128)
    left, right = records[0][2].left, records[0][2].right
    ids = torch.cat((tree.permutation, tree.permutation+tree.size))
    expected = torch.empty_like(x)
    expected[ids] = (x[ids]@right.mT)@left.mT
    torch.testing.assert_close(op.apply(x), expected, rtol=1e-10, atol=1e-10)
    with pytest.raises(H2CapacityError) as caught:
        NestedH2Actions(tree, records, config=replace(config, retained_budget_bytes=config.retained_budget_bytes-1))
    assert not caught.value.receipt['admitted']
    assert caught.value.receipt['retained_bytes_bound'] == plan['retained_bytes_bound']


def test_rejects_before_scales_seeds_or_svd_are_allocated(monkeypatch):
    tree, records = fixture()
    import witwin.maxwell.mom.h2 as h2
    def forbidden(*args, **kwargs):
        raise AssertionError('Conversion allocations must follow retained admission')
    monkeypatch.setattr(torch, 'ones', forbidden)
    monkeypatch.setattr(torch.linalg, 'svd', forbidden)
    monkeypatch.setattr(h2, 'NestedClusterBasis', forbidden)
    with pytest.raises(H2CapacityError):
        NestedH2Actions(tree, records, config=H2Config(retained_budget_bytes=1))


def test_rank_zero_spaces_still_charge_maps_and_scales():
    tree, _ = fixture()
    plan = plan_h2_storage(tree, [], components=2, dtype=torch.complex128, max_rank=8, budget_bytes=2**30)
    assert plan['coupling_bytes_bound'] == 0
    assert plan['retained_bytes_bound'] == 2*64*8+2*64*8
    with pytest.raises(ValueError):
        H2Config(retained_budget_bytes=True)
    with pytest.raises(ValueError):
        plan_h2_storage(tree, [], components=2, dtype=torch.float64, max_rank=8, budget_bytes=2**30)


def test_source_h_conversion_rejects_before_input_scaling(monkeypatch):
    tree, records = fixture()
    source = SimpleNamespace(refresh=lambda: None, tree=tree, dtype=torch.complex128,
        device=tree.device, block_nodes=((0, 0),), blocks=((None, None, records[0][2]),),
        packed_near=None, provider=SimpleNamespace(system=SimpleNamespace(total_edges=tree.size),
                                                  exterior=SimpleNamespace(eta=377.)))
    def forbidden(*args, **kwargs):
        raise AssertionError('Rejected conversion must not allocate input scale arrays')
    monkeypatch.setattr(torch, 'full', forbidden)
    with pytest.raises(H2CapacityError):
        NestedH2Actions.from_hmatrix(source, H2Config(retained_budget_bytes=1))
