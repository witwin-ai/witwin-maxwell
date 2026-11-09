"""Scalar-reference equivalence and strict batched planning controls."""
import hashlib
import math
from pathlib import Path
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.fmm_error import multipole_truncation_bound, multipole_truncation_bounds
from witwin.maxwell.mom.interpolation_error import KernelBoundUnavailable
from witwin.maxwell.mom.fmm_order import plan_multipole_order, MultipoleOrderError
from witwin.maxwell.mom.cluster import build_support_octree
from witwin.maxwell.mom.physics import compile_medium

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batch_matches_native_scalar_bounds(dtype):
    displacement = torch.tensor([[1., .3, -.2]]*10, device="cuda", dtype=dtype)
    radius = torch.tensor([0., .01, .02, .08, .13, .18, .21, .3, .001, .003], device="cuda", dtype=dtype)
    source = radius.flip(0).clone()
    radius[0] = source[0] = 0.
    source[1] = 0.
    source[7] = .3
    displacement[-2] *= 1e-12
    radius[-2] *= 1e-12
    source[-2] *= 1e-12
    for wave in [0., 6.3, -6.3+.4j, 1e6j]:
        for order in [0, 6, 12]:
            batch = multipole_truncation_bounds(displacement, radius, source, wave, order)
            for i in range(len(radius)):
                try:
                    scalar = multipole_truncation_bound(displacement[i], radius[i], source[i], wave, order)
                except KernelBoundUnavailable:
                    assert not bool(batch["available"][i])
                    continue
                assert bool(batch["available"][i])
                for key in ["green_absolute", "gradient_component_absolute"]:
                    torch.testing.assert_close(batch[key][i], batch[key].new_tensor(scalar[key]), rtol=5e-14, atol=0.)
    # Extreme passive attenuation retains a positive floor rather than zero.
    underflow = multipole_truncation_bounds(displacement[:1], radius.new_tensor([.02]),
        radius.new_tensor([.02]), 1e6j, 6)
    assert bool(underflow["available"][0]) and float(underflow["green_absolute"][0]) > 0
    # Positive-log overflow is unavailable, not a finite zero bound.
    overflow = multipole_truncation_bounds(displacement[:1], radius.new_tensor([.02]),
        radius.new_tensor([.02]), 1e308, 6)
    assert not bool(overflow["available"][0])


def test_batch_rejects_invalid_native_inputs():
    d = torch.tensor([[1., 0., 0.]], device="cuda", dtype=torch.float64)
    r = d.new_tensor([.02])
    for bad in [d.cpu(), d.to(torch.complex128), d.clone().requires_grad_(), d*math.nan, d*0]:
        with pytest.raises(ValueError): multipole_truncation_bounds(bad, r, r, 3., 4)
    for bad in [r.cpu(), -r, r.clone().requires_grad_(), r.double().reshape(()), r*math.inf]:
        with pytest.raises(ValueError): multipole_truncation_bounds(d, bad, r, 3., 4)
    for wave, order in [(3.-1j, 4), (math.inf, 4), (3., -1), (3., True)]:
        with pytest.raises(ValueError): multipole_truncation_bounds(d, r, r, wave, order)


def scalar_planner():
    source = Path(__file__).parent/"reference"/"fmm_order_scalar_v1.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == "51e3bb2aae8dd13475526b43c8a6a1cd4b4000d4feed28bd09b25cab78120009"
    namespace = {"__package__": "witwin.maxwell.mom"}
    exec(compile(source.read_text(), str(source), "exec"), namespace)
    return namespace["plan_multipole_order"]


def run_plan(planner, *args, **kwargs):
    try:
        return True, planner(*args, **kwargs)
    except (MultipoleOrderError, ValueError) as error:
        if not hasattr(error, "receipt"):
            raise
        return False, error.receipt


@pytest.mark.parametrize("rwg", [False, True])
def test_strict_planner_matches_frozen_scalar_reference(rwg):
    # Mandatory pinned pre-batching implementation; no same-path fake oracle.
    reference = scalar_planner()
    lower = torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., .3, .1]], device="cuda", dtype=torch.float64)
    tree = build_support_octree(lower, lower+.02, leaf_size=1)
    medium = compile_medium(mw.Material(), 3e8)
    regions = ((tree, medium), (tree, medium))
    stats = torch.arange(tree.node_count*3, device="cuda", dtype=torch.float64).reshape(-1,3)*.03-10.
    kwargs = dict(min_order=2, max_order=10, table_storage="compact", separation_ratio=.25)
    if rwg: kwargs.update(rwg_statistics=(stats, stats), rwg_absolute_budget=1e100, balancing_impedance=abs(medium.eta))
    for budget, cap in [(mw.KernelErrorBudget(1e-6,1e-5), 1024**3),
                         (mw.KernelErrorBudget(1e-30,1e-30),1024**3),
                         (mw.KernelErrorBudget(1e-6,1e-5),1)]:
        actual = run_plan(plan_multipole_order, regions, budget, table_budget_bytes=cap, **kwargs)
        expected = run_plan(reference, regions, budget, table_budget_bytes=cap, **kwargs)
        assert actual[0] == expected[0]
        a,e = actual[1], expected[1]
        assert a.keys() == e.keys()
        assert a.get("selected_order") == e.get("selected_order") and a.get("reason") == e.get("reason")
        for aa,ee in zip(a["attempts"],e["attempts"],strict=True):
            for key in ["order","table_bytes","far_pairs","unavailable_bounds","kernel_pass","rwg_pass"]:
                assert aa[key] == ee[key]
            for key in ["green_absolute_max","gradient_component_absolute_max"]:
                assert math.isclose(aa[key], ee[key], rel_tol=5e-14, abs_tol=0.)
            if rwg:
                assert math.isclose(aa["rwg_bound"]["absolute_bound"],ee["rwg_bound"]["absolute_bound"],rel_tol=5e-14)
    # Unavailable contours still force rejection for every attempted order.
    wide = build_support_octree(lower[:2],lower[:2]+(.45/(3**.5)),leaf_size=1)
    for planner in [plan_multipole_order, reference]:
        accepted, receipt = run_plan(planner, ((wide,medium),), mw.KernelErrorBudget(1.,1.),max_order=2)
        assert not accepted
        assert all(x["unavailable_bounds"] == 2 and not x["kernel_pass"] for x in receipt["attempts"])
    print(dict(rwg=rwg, strict_scalar_reference_matched=True, physical_qualification=False),flush=True)


def test_bundled_scalar_oracle_without_environment_and_corruption_guard(monkeypatch):
    monkeypatch.delenv("GSP_SCALAR_FMM_PLANNER", raising=False)
    monkeypatch.delenv("GSP_SCALAR_FMM_PLANNER_SHA256", raising=False)
    reference = scalar_planner()
    source = Path(__file__).parent/"reference"/"fmm_order_scalar_v1.py"
    assert Path(reference.__code__.co_filename) == source
    assert reference is not plan_multipole_order
    assert reference.__globals__["multipole_truncation_bound"] is multipole_truncation_bound
    assert "multipole_truncation_bounds" not in reference.__globals__
    with pytest.raises(ValueError, match="regional trees"):
        reference((), mw.KernelErrorBudget(1., 1.))
    read_bytes = Path.read_bytes
    def corrupted(path):
        value = read_bytes(path)
        return value+b"corrupt" if path == source else value
    monkeypatch.setattr(Path, "read_bytes", corrupted)
    with pytest.raises(AssertionError): scalar_planner()
    print(dict(bundled_oracle_executed=True, external_environment_required=False,
        corrupted_oracle_rejected=True, physical_qualification=False), flush=True)
