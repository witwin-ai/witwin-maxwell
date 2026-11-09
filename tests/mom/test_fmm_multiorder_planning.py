"""Bounded multiorder contours and shared native geometry planning."""
import hashlib
import math
from pathlib import Path
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.fmm_error import multipole_truncation_bound, multipole_truncation_bounds_many
from witwin.maxwell.mom.interpolation_error import KernelBoundUnavailable
from witwin.maxwell.mom.fmm_order import plan_multipole_order
from witwin.maxwell.mom.cluster import build_support_octree
from witwin.maxwell.mom.physics import compile_medium

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_multiorder_contours_match_unchanged_scalar_bounds(dtype):
    displacement = torch.tensor([[1., .3, -.2]]*7, device="cuda", dtype=dtype)
    test = torch.tensor([0., .01, .02, .08, .13, .18, .3], device="cuda", dtype=dtype)
    source = test.flip(0).clone()
    source[0] = source[1] = 0.
    source[-1] = .3
    orders = (0, 12, 6, 1, 25, 6)
    for wave in (0., -6.3+.4j, 1e6j):
        values = multipole_truncation_bounds_many(displacement, test, source, wave, orders)
        assert all(x.shape == (7,6) for x in values.values())
        for row in range(7):
            for column,order in enumerate(orders):
                try:
                    scalar = multipole_truncation_bound(displacement[row], test[row], source[row], wave, order)
                except KernelBoundUnavailable:
                    assert not bool(values["available"][row,column])
                    continue
                assert bool(values["available"][row,column])
                for key in ("green_absolute", "gradient_component_absolute"):
                    torch.testing.assert_close(values[key][row,column],values[key].new_tensor(scalar[key]),rtol=5e-14,atol=0.)
    print(dict(multiorder_scalar_bounds_matched=True, dtype=str(dtype), physical_qualification=False),flush=True)


def test_multiorder_workspace_and_native_integer_constraints():
    d = torch.tensor([[1.,0.,0.]],device="cuda",dtype=torch.float64)
    r = d.new_tensor([.02])
    for orders in [(),tuple(range(17)),(True,),(None,),(-1,),(2**63,),range(3)]:
        with pytest.raises(ValueError):multipole_truncation_bounds_many(d,r,r,3.,orders)
    values = multipole_truncation_bounds_many(d,r,r,3.,tuple(range(16)))
    assert all(x.shape==(1,16) for x in values.values())
    assert bool(values["available"].all())
    overflow = multipole_truncation_bounds_many(d,r,r,1e308,(0,6,12))
    assert not bool(overflow["available"].any())


def scalar_reference():
    p=Path(__file__).parent/"reference"/"fmm_order_scalar_v1.py"
    assert hashlib.sha256(p.read_bytes()).hexdigest()=="51e3bb2aae8dd13475526b43c8a6a1cd4b4000d4feed28bd09b25cab78120009"
    namespace={"__package__":"witwin.maxwell.mom"}
    exec(compile(p.read_text(),str(p),"exec"),namespace)
    return namespace["plan_multipole_order"]


def invoke(planner,*args,**kwargs):
    try:return True,planner(*args,**kwargs)
    except ValueError as e:
        if not hasattr(e,"receipt"):raise
        return False,e.receipt


def compare(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:compare(a[key],b[key])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b,strict=True):compare(x,y)
    elif type(a) is float:assert math.isclose(a,b,rel_tol=5e-14,abs_tol=0.)
    else:assert a==b


@pytest.mark.parametrize("storage",["dense","compact"])
def test_chunked_planner_matches_frozen_scalar_decisions_and_shares_geometry(monkeypatch,storage):
    import witwin.maxwell.mom.fmm_order as module
    lower=torch.tensor([[0.,0.,0.],[1.,0.,0.]],device="cuda",dtype=torch.float64)
    tree=build_support_octree(lower,lower+.02,leaf_size=1)
    medium=compile_medium(mw.Material(),3e8)
    regions=((tree,medium),(tree,medium))
    reference=scalar_reference()
    original=module.sphere_block_partition
    calls=[]
    def observed(*args,**kwargs):
        calls.append((id(args[0]),kwargs["pair_batch"]))
        yield from original(*args,**kwargs)
    monkeypatch.setattr(module,"sphere_block_partition",observed)
    for g,h,maximum,cap in [(1e-6,1e-5,10,2**60),(1e-50,1e-50,18,2**60),
                              (1e-30,1e-30,30,2**60),(1e-6,1e-5,10,1)]:
        calls.clear()
        budget=mw.KernelErrorBudget(g,h)
        args=dict(min_order=2,max_order=maximum,separation_ratio=.25,table_storage=storage,table_budget_bytes=cap)
        actual=invoke(plan_multipole_order,regions,budget,**args)
        expected=invoke(reference,regions,budget,**args)
        assert actual[0]==expected[0]
        compare(actual[1],expected[1])
        last=actual[1]["attempts"][-1]["order"]
        assert len(calls)==len(regions)*((last-2)//16+1)
        assert all(pair_batch==32 for _,pair_batch in calls)
    # Strict table precedence applies even to exact-zero point-support bounds.
    points=build_support_octree(lower,lower,leaf_size=1)
    for cap in (1,2**60):
        budget=mw.KernelErrorBudget(1.,1.)
        actual=invoke(plan_multipole_order,((points,medium),),budget,max_order=20,table_budget_bytes=cap,table_storage=storage)
        expected=invoke(reference,((points,medium),),budget,max_order=20,table_budget_bytes=cap,table_storage=storage)
        assert actual[0]==expected[0];compare(actual[1],expected[1])
    print(dict(storage=storage, scalar_decisions_matched=True, bounded_orders_per_chunk=16,
        geometry_traversals_shared=True, physical_qualification=False),flush=True)


def test_shared_traversal_rejects_tree_mutation(monkeypatch):
    import witwin.maxwell.mom.fmm_order as module
    lower=torch.tensor([[0.,0.,0.],[1.,0.,0.]],device="cuda",dtype=torch.float64)
    tree=build_support_octree(lower,lower+.02,leaf_size=1)
    original=module.sphere_block_partition
    def changed(*args,**kwargs):
        yield from original(*args,**kwargs)
        tree.lower.add_(1e-6)
    monkeypatch.setattr(module,"sphere_block_partition",changed)
    with pytest.raises(ValueError,match="changed"):
        plan_multipole_order(((tree,compile_medium(mw.Material(),3e8)),),mw.KernelErrorBudget(1.,1.),max_order=4)
