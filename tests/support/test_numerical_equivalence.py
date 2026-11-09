"""CPU checks of derived per-tensor bounds, pin tiers and horizon growth."""

import json
import time

import pytest
import torch

from tests.support.numerical_equivalence import (
    assert_numerically_equivalent,
    assert_run_level_equivalent,
    classify_against_pin,
    record_pin_comparison,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128])
@pytest.mark.parametrize("level", ["setup", "step", "run"])
def test_bound_edges(dtype, level):
    ref = torch.tensor([1, 0], dtype=dtype)
    real_dtype = ref.real.dtype
    factor = (1e-4 if level == "run" else 64 * torch.finfo(torch.float32).eps) if real_dtype == torch.float32 else (
        1e-10 if level == "run" else 1e-12)
    edge = torch.tensor(factor, dtype=real_dtype)
    inside = torch.nextafter(edge, torch.zeros_like(edge))
    outside = torch.nextafter(edge, torch.full_like(edge, float("inf")))
    for delta, tier in ((inside, "T1"), (outside, "T2")):
        new = ref.clone()
        new[1] = delta
        report = classify_against_pin(ref, new, level=level)
        assert report["tier"] == tier
        assert report["tensors"][0] == dict(
            name="tensor", dtype=str(dtype), shape=[2], max_abs=delta.item(), max_ref=1.0,
            max_rel=delta.item(), bound=factor, ratio_to_bound=delta.item() / factor, tier=tier, level=level)
        if tier == "T1":
            assert assert_numerically_equivalent(ref, new, level=level) == report
        else:
            with pytest.raises(AssertionError, match=rf"field: diff=.*bound=.*level={level}"):
                assert_numerically_equivalent(ref, new, level=level, name="field")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128,
                                   torch.int32, torch.int64, torch.bool])
def test_exact_is_t0(dtype):
    ref = torch.tensor([0, 1], dtype=dtype)
    assert assert_numerically_equivalent(ref, ref.clone(), level="run")["tier"] == "T0"


@pytest.mark.parametrize("ref,new", [
    (torch.tensor([True]), torch.tensor([False])),
    (torch.tensor([2**62]), torch.tensor([2**62 + 1])),
    (torch.ones(2), torch.ones(1, 2)),
    (torch.ones(2), torch.ones(2, dtype=torch.float64)),
    ({"a": torch.ones(1)}, {"b": torch.ones(1)}),
    ([torch.ones(1)], (torch.ones(1),)),
])
def test_discrete_mismatches(ref, new):
    assert classify_against_pin(ref, new, level="run")["tier"] == "T2"
    with pytest.raises(AssertionError, match="T2"):
        assert_numerically_equivalent(ref, new, level="run")


def test_discrete_float_mismatch_is_t2():
    ref = torch.tensor([1.0])
    new = torch.nextafter(ref, torch.tensor([2.0]))
    assert assert_numerically_equivalent(ref, new, level="setup")["tier"] == "T1"
    with pytest.raises(AssertionError, match="ownership.*bound=0"):
        assert_numerically_equivalent(ref, new, level="setup", discrete=True, name="ownership")


@pytest.mark.parametrize("dtype,left,right,delta", [
    (torch.int64, 2**62, 2**62 + 1, 1),
    (torch.int64, -(2**63), 2**63 - 1, float(2**64 - 1)),
    (torch.uint64, 2**63 - 1, 2**63, 1),
    (torch.uint8, 255, 0, 255),
])
def test_integer_difference_metrics(dtype, left, right, delta):
    report = classify_against_pin(torch.tensor([left], dtype=dtype), torch.tensor([right], dtype=dtype), level="setup")
    assert report["tier"] == "T2"
    assert report["tensors"][0]["max_abs"] == delta


def test_nested_per_tensor_scales():
    ref = {"fields": [torch.ones(2), (torch.ones(1) * 1e-10,)], "empty": torch.empty(0)}
    new = {"fields": [torch.ones(2) + 1e-6, (torch.ones(1) * 1e-10 + 1e-12,)], "empty": torch.empty(0)}
    report = classify_against_pin(ref, new, level="run")
    assert {row["name"]: row["tier"] for row in report["tensors"]} == {
        "fields[0]": "T1", "fields[1][0]": "T2", "empty": "T0"}
    assert report["tier"] == "T2"
    with pytest.raises(AssertionError, match=r"fields\[1\]\[0\]"):
        assert_numerically_equivalent(ref, new, level="run")


def test_complex_uses_difference_magnitude():
    ref = torch.tensor([1, 0], dtype=torch.complex128)
    new = torch.tensor([1, 0.75e-10 + 0.75e-10j], dtype=torch.complex128)
    report = classify_against_pin(ref, new, level="run")
    assert report["tier"] == "T2"
    assert report["tensors"][0]["max_abs"] == pytest.approx(2**0.5 * 0.75e-10)


def test_zero_reference_and_nonfinite_mismatch():
    ref = torch.zeros(2)
    assert assert_numerically_equivalent(ref, ref, level="run")["tier"] == "T0"
    report = classify_against_pin(ref, ref + 1e-30, level="run")
    assert report["tier"] == "T2"
    assert report["tensors"][0]["bound"] == 0
    with pytest.raises(AssertionError):
        assert_numerically_equivalent(ref, ref + 1e-30, level="run")
    assert classify_against_pin(torch.tensor([float("inf")]), ref[:1], level="run")["tier"] == "T2"
    assert classify_against_pin(ref[:1], torch.tensor([float("nan")]), level="run")["tier"] == "T2"


def test_growth_check():
    ref = {"fields": [torch.tensor([1., 0.], dtype=torch.float64)]}
    def moved(delta):
        return {"fields": [torch.tensor([1., delta], dtype=torch.float64)]}
    report = assert_run_level_equivalent(ref, moved(1e-11), ref, moved(4e-11))
    assert report["tensors"][0]["growth"] == 4
    assert report["tensors"][0]["half"]["max_abs"] == 1e-11
    assert report["tensors"][0]["full"]["max_abs"] == 4e-11
    with pytest.raises(AssertionError, match="fields.*growth"):
        assert_run_level_equivalent(ref, moved(1e-11), ref, moved(4.01e-11))
    assert_run_level_equivalent(ref, ref, ref, ref)
    assert_run_level_equivalent(ref, ref, ref, moved(1e-10))
    with pytest.raises(AssertionError):
        assert_run_level_equivalent(ref, ref, ref, moved(1.01e-10))
    # Both horizons must meet their run bound, independently of the growth test.
    with pytest.raises(AssertionError):
        assert_run_level_equivalent(ref, moved(1.01e-10), ref, moved(1e-11))
    with pytest.raises(AssertionError):
        assert_run_level_equivalent(ref, ref, {"other": ref["fields"]}, {"other": ref["fields"]})


def test_run_histories_can_have_different_horizon_shapes():
    half, full = torch.ones(4), torch.ones(8)
    report = assert_run_level_equivalent(half, half + 1e-6, full, full + 2e-6, name="history")
    assert report["tier"] == "T1"
    assert report["tensors"][0]["name"] == "history"
    with pytest.raises(AssertionError):
        assert_run_level_equivalent(half, half + 1e-6, full, full + 2e-6, discrete=True)


@pytest.mark.parametrize("change", ["sign", "timing", "mask"])
def test_physical_behavior_changes_are_t2(change):
    ref = torch.sin(torch.arange(128, dtype=torch.float32) * 0.1)
    if change == "sign":
        new = -ref
    elif change == "timing":
        new = torch.roll(ref, 1)
    else:
        ref = ref > 0
        new = ref.clone()
        new[63] = ~new[63]
    assert classify_against_pin(ref, new, level="run")["tier"] == "T2"
    with pytest.raises(AssertionError, match="T2"):
        assert_numerically_equivalent(ref, new, level="run", name=change)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=[
    pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")])])
def test_85000_leaves(device, monkeypatch):
    # Distinct small leaves model the setup consumer's large nested state tree.
    # Leaves are views of two device tensors so fixture construction stays two
    # transfers instead of 170k tiny launches on a busy device.
    base = torch.tensor([1., 0.]).repeat(85000, 1).to(device)
    ref = {"field": {str(i): leaf for i, leaf in enumerate(base.unbind(0))}}
    new = {"field": {str(i): leaf for i, leaf in enumerate((base + 1e-6).unbind(0))}}
    mask = torch.tensor([True, False]).repeat(16, 1).to(device)
    ref["mask"] = {str(i): leaf for i, leaf in enumerate(mask.unbind(0))}
    new["mask"] = {str(i): leaf for i, leaf in enumerate(mask.clone().unbind(0))}
    transfers = []
    original = torch.Tensor.cpu
    def host_metrics(values, *args, **kwargs):
        transfers.append((str(values.device), values.shape))
        return original(values, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "cpu", host_metrics)
    # No per-leaf equality/scalar extraction may synchronize CUDA implicitly.
    def forbidden(*args, **kwargs):
        pytest.fail("per-leaf host synchronization")
    monkeypatch.setattr(torch, "equal", forbidden)
    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    report = assert_numerically_equivalent(ref, new, level="setup")
    elapsed = time.perf_counter() - start
    assert len(report["tensors"]) == 85016
    assert report["tier"] == "T1"
    assert len(transfers) == 2  # Float and bool: one transfer per device/dtype.
    assert elapsed < 5, f"85000 leaves took {elapsed:.3f}s on {device}"
    print(f"85000 leaves ({device}): {elapsed:.3f}s, {len(transfers)} metric transfers")


def test_pin_receipts_and_unset_log(monkeypatch, tmp_path):
    log = tmp_path / "tiers.jsonl"
    ref = {"field": torch.ones(1), "index": torch.tensor([1])}
    new = {"field": torch.ones(1) + 1e-6, "index": torch.tensor([1])}
    monkeypatch.delenv("WITWIN_PIN_TIER_LOG", raising=False)
    assert record_pin_comparison(ref, new, nodeid="test::node", level="run")["tier"] == "T1"
    assert not log.exists()
    monkeypatch.setenv("WITWIN_PIN_TIER_LOG", str(log))
    record_pin_comparison(ref, new, nodeid="test::node", level="run")
    with pytest.raises(AssertionError):
        record_pin_comparison(torch.ones(1), torch.ones(1) * 2, nodeid="test::bad", name="bad", level="run")
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [(row["nodeid"], row["name"], row["tier"], row["level"]) for row in rows] == [
        ("test::node", "field", "T1", "run"), ("test::node", "index", "T0", "run"),
        ("test::bad", "bad", "T2", "run")]
    assert rows[0]["max_rel"] == (new["field"] - ref["field"]).item()


def test_report_json_roundtrip():
    ref = {"field": [torch.ones(2), torch.ones(2, dtype=torch.complex128)], "indices": torch.tensor([1, 2])}
    report = assert_numerically_equivalent(ref, ref, level="step")
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert [row["name"] for row in report["tensors"]] == ["field[0]", "field[1]", "indices"]
    assert all(set(row) == {"name", "dtype", "shape", "max_abs", "max_ref", "max_rel", "bound", "ratio_to_bound", "tier", "level"}
               for row in report["tensors"])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128])
@pytest.mark.parametrize("level", ["step", "run"])
@pytest.mark.parametrize("reference", [0.0, 1e-19])
@pytest.mark.parametrize("noise,tier", [(1e-19, "T1"), (1e-5, "T2"), (0.1, "T2")])
def test_family_structural_zero(dtype, level, reference, noise, tier):
    ref = {"fields": [torch.tensor([reference], dtype=dtype), torch.ones(2, dtype=dtype)]}
    new = {"fields": [ref["fields"][0] + noise, ref["fields"][1]]}
    family = {"fields[0]": "E", "fields[1]": "E"}
    report = classify_against_pin(ref, new, level=level, family=family)
    zero, sibling = report["tensors"]
    factor = (1e-4 if level == "run" else 64 * torch.finfo(torch.float32).eps) if ref["fields"][0].real.dtype == torch.float32 else (
        1e-10 if level == "run" else 1e-12)
    assert report["tier"] == zero["tier"] == tier
    assert zero["structural_zero"] is True
    assert zero["family"] == sibling["family"] == "E"
    assert zero["family_max_ref"] == sibling["family_max_ref"] == 1.0
    assert zero["bound"] == factor * 1e-6
    assert zero["ratio_to_bound"] == zero["max_abs"] / zero["bound"]
    assert sibling["structural_zero"] is False
    assert sibling["bound"] == factor
    if tier == "T1":
        assert assert_numerically_equivalent(ref, new, level=level, family=family) == report
        if level == "run":
            run = assert_run_level_equivalent(ref, new, ref, new, family=family)
            assert run["tier"] == "T1"
            assert run["tensors"][0]["half"] == run["tensors"][0]["full"] == zero
    else:
        with pytest.raises(AssertionError, match="T2"):
            assert_numerically_equivalent(ref, new, level=level, family=family)
        if level == "run":
            with pytest.raises(AssertionError, match="T2"):
                assert_run_level_equivalent(ref, new, ref, new, family=family)


@pytest.mark.parametrize("level", ["step", "run"])
def test_family_small_physical_component_keeps_per_tensor_bound(level):
    ref = {"Ex": torch.ones(1), "Ey": torch.tensor([1e-5])}
    new = {"Ex": ref["Ex"], "Ey": 2 * ref["Ey"]}
    family = {"Ex": "E", "Ey": "E"}
    report = classify_against_pin(ref, new, level=level, family=family)
    row = report["tensors"][1]
    default = classify_against_pin(ref, new, level=level)["tensors"][1]
    assert row == dict(default, structural_zero=False, family="E", family_max_ref=1.0)
    assert row["tier"] == "T2"


@pytest.mark.parametrize("family", [{}, {"tensor": "E"}, lambda path: "E"])
@pytest.mark.parametrize("compare", [assert_numerically_equivalent, classify_against_pin])
def test_setup_rejects_family(compare, family):
    with pytest.raises(ValueError, match="family.*setup"):
        compare(torch.ones(1), torch.ones(1), level="setup", family=family)


@pytest.mark.parametrize("level", ["step", "run"])
@pytest.mark.parametrize("dtype", [torch.int64, torch.bool, torch.float32])
def test_family_discrete_stays_exact(level, dtype):
    ref = {"zero": torch.zeros(1, dtype=dtype), "scale": torch.tensor([1e12])}
    new = {"zero": torch.ones(1, dtype=dtype), "scale": ref["scale"]}
    family = {"zero": "state", "scale": "state"}
    with pytest.raises(AssertionError, match="zero: diff=1, bound=0"):
        assert_numerically_equivalent(ref, new, level=level, family=family, discrete=True)
    if dtype != torch.float32:
        row = classify_against_pin(ref, new, level=level, family=family)["tensors"][0]
        assert row["tier"] == "T2" and row["bound"] == 0
        assert row["structural_zero"] is False
    exact = assert_numerically_equivalent(ref, ref, level=level, family=family, discrete=True)
    assert exact["tier"] == "T0"
    assert all(row["structural_zero"] is False for row in exact["tensors"])
    if level == "run":
        with pytest.raises(AssertionError, match="zero: diff=1, bound=0"):
            assert_run_level_equivalent(ref, new, ref, new, family=family, discrete=True)


@pytest.mark.parametrize("level", ["step", "run"])
def test_families_use_named_paths_and_independent_reference_scales(level):
    ref = {"E": [torch.ones(1), torch.zeros(1)],
           "H": [torch.tensor([1e-15]), torch.zeros(1)],
           "unmapped": torch.zeros(1)}
    new = {"E": [ref["E"][0], torch.tensor([1e-19])],
           "H": [ref["H"][0], torch.tensor([1e-19])],
           "unmapped": torch.tensor([1e-19])}
    family = {f"state.{kind}[{i}]": kind for kind in ("E", "H") for i in range(2)}
    family["absent"] = "E"
    with pytest.raises(AssertionError, match=r"state.H\[1\].*state.unmapped"):
        assert_numerically_equivalent(ref, new, level=level, name="state", family=family)
    report = classify_against_pin({"state": ref}, {"state": new}, level=level, family=family)
    rows = {row["name"]: row for row in report["tensors"]}
    assert rows["state.E[1]"]["tier"] == "T1"
    assert rows["state.H[1]"]["tier"] == "T2"
    assert rows["state.H[1]"]["family_max_ref"] == ref["H"][0].item()
    assert rows["state.unmapped"]["structural_zero"] is False
    assert "family" not in rows["state.unmapped"]
    assert "family_max_ref" not in rows["state.unmapped"]
    assert classify_against_pin(ref, new, level=level, family={}) == dict(
        classify_against_pin(ref, new, level=level),
        tensors=[dict(row, structural_zero=False) for row in classify_against_pin(ref, new, level=level)["tensors"]])


def test_family_threshold_empty_and_all_zero():
    ref = {"scale": torch.ones(1, dtype=torch.float64),
           "edge": torch.tensor([1e-6], dtype=torch.float64), "empty": torch.empty(0, dtype=torch.float64)}
    rows = assert_numerically_equivalent(ref, ref, level="step", family=dict.fromkeys(ref, "E"))["tensors"]
    assert rows[1]["structural_zero"] is False  # The threshold is strictly less than.
    assert rows[1]["bound"] == rows[2]["bound"] == 1e-12 * 1e-6
    assert rows[2]["structural_zero"] is True
    zero = torch.zeros(1)
    row = classify_against_pin(zero, zero + 1e-19, level="run", family={"tensor": "E"})["tensors"][0]
    assert row["tier"] == "T2" and row["bound"] == 0
    assert row["family_max_ref"] == 0 and row["structural_zero"] is False


def test_run_family_scales_at_both_horizons_and_checks_growth():
    half = {"Ex": torch.ones(1), "Ey": torch.zeros(1)}
    full = {"Ex": torch.full((2,), 2.0), "Ey": torch.zeros(2)}
    family = dict.fromkeys(half, "E")
    def moved(ref, delta):
        return {"Ex": ref["Ex"], "Ey": ref["Ey"] + delta}
    report = assert_run_level_equivalent(half, moved(half, 1e-11), full, moved(full, 4e-11), family=family)
    row = report["tensors"][1]
    assert row["half"]["bound"] == 1e-10 and row["half"]["family_max_ref"] == 1
    assert row["full"]["bound"] == 2e-10 and row["full"]["family_max_ref"] == 2
    assert row["growth"] == 4
    assert_run_level_equivalent(half, moved(half, 1e-11), full, moved(full, 5e-11), family=family)
    # Half-horizon rejection must survive a full-horizon pass and a shrinking diff.
    with pytest.raises(AssertionError, match="Ey"):
        assert_run_level_equivalent(half, moved(half, 1.5e-10), full, moved(full, 1.5e-10), family=family)


@pytest.mark.parametrize("level", ["step", "run"])
@pytest.mark.parametrize("name", [None, "state"])
def test_callable_family_matches_mapping_report(level, name):
    ref = {"E": [torch.ones(1), torch.zeros(1)],
           "H": (torch.tensor([1e-15], dtype=torch.float64), torch.zeros(1, dtype=torch.float64)),
           "mask": torch.tensor([True]), "unmapped": torch.ones(1), "none": torch.ones(1)}
    new = dict(ref, E=[ref["E"][0], ref["E"][1] + 1e-19],
               H=(ref["H"][0], ref["H"][1] + 1e-35))
    prefix = f"{name}." if name else ""
    family = {f"{prefix}{kind}[{i}]": kind for kind in ("E", "H") for i in range(2)}
    family.update({f"{prefix}mask": "E", f"{prefix}none": None})
    paths = []
    def select(path):
        paths.append(path)
        return family.get(path)
    mapping = assert_numerically_equivalent(ref, new, level=level, name=name, family=family)
    assert assert_numerically_equivalent(ref, new, level=level, name=name, family=select) == mapping
    assert paths == [row["name"] for row in mapping["tensors"]]
    assert all(isinstance(row["structural_zero"], bool) for row in mapping["tensors"])
    for row in mapping["tensors"][-2:]:
        assert row["structural_zero"] is False
        assert "family" not in row and "family_max_ref" not in row
    # Pin comparisons have no name argument; preserve the same report paths.
    pinned, current = ({name: ref}, {name: new}) if name else (ref, new)
    assert classify_against_pin(pinned, current, level=level, family=family) == mapping
    assert classify_against_pin(pinned, current, level=level, family=select) == mapping
    if level == "run":
        assert assert_run_level_equivalent(ref, new, ref, new, name=name, family=select) == (
            assert_run_level_equivalent(ref, new, ref, new, name=name, family=family))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128])
@pytest.mark.parametrize("reference,half_fraction,full_fraction,passes", [
    (0.0, 0.05, 0.5, True),  # Tenfold growth remains below the structural-zero floor.
    (0.0, 0.2, 2.0, False),  # Tenfold growth exceeds the floor.
    (1e-5, 0.1, 0.5, False),  # A physical component still fails at fivefold growth.
])
def test_run_family_growth_floor(dtype, reference, half_fraction, full_fraction, passes):
    factor = 1e-4 if dtype in (torch.float32, torch.complex64) else 1e-10
    scale = max(reference, 1e-6)
    ref = {"Ex": torch.ones(1, dtype=dtype), "Ey": torch.tensor([0, reference], dtype=dtype)}
    family = dict.fromkeys(ref, "E")
    def moved(fraction):
        # Put the noise at zero so its magnitude is independent of reference rounding.
        return dict(ref, Ey=torch.tensor([fraction * factor * scale, reference], dtype=dtype))
    half, full = moved(half_fraction), moved(full_fraction)
    full_row = classify_against_pin(ref, full, level="run", family=family)["tensors"][1]
    assert full_row["structural_zero"] is (reference == 0)
    assert full_row["tier"] == ("T2" if full_fraction > 1 else "T1")
    if passes:
        report = assert_run_level_equivalent(ref, half, ref, full, family=family)
        row = report["tensors"][1]
        assert report["tier"] == row["tier"] == "T1"
        assert row["growth"] == pytest.approx(10)
        assert row["max_abs"] <= row["bound"]
    else:
        with pytest.raises(AssertionError, match="T2.*Ey.*growth"):
            assert_run_level_equivalent(ref, half, ref, full, family=family)


def test_family_reuses_batched_metrics_across_dtypes(monkeypatch):
    ref = {"Ex": torch.ones(2), "Ey": torch.zeros(1, dtype=torch.float64), "mask": torch.tensor([True])}
    new = dict(ref, Ey=ref["Ey"] + 1e-19)
    transfers = []
    original = torch.Tensor.cpu
    def host_metrics(values, *args, **kwargs):
        transfers.append(values.shape)
        return original(values, *args, **kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail("per-leaf host synchronization")
    monkeypatch.setattr(torch.Tensor, "cpu", host_metrics)
    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(torch, "equal", forbidden)
    report = assert_numerically_equivalent(ref, new, level="step", family=dict.fromkeys(ref, "E"))
    assert report["tier"] == "T1"
    assert report["tensors"][1]["family_max_ref"] == 1
    assert len(transfers) == 3  # One transfer per dtype, even across family members.
