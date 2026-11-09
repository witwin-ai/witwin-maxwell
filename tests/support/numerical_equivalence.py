"""Per-tensor path/pin bounds; repeated-run determinism remains exact.

Small leaves are packed by size before reduction. Metrics and exact-mismatch
flags stay on device until one host transfer per device/dtype group.
"""

from collections import defaultdict
import json
import math
import os

import torch

__all__ = ["assert_numerically_equivalent", "assert_run_level_equivalent",
           "classify_against_pin", "record_pin_comparison"]


def _reduce(ref, new):
    """Reduce rows, retaining exact integer comparisons before widening."""
    equal = (ref == new).all(dim=1)
    continuous = ref.is_floating_point() or ref.is_complex()
    wide = torch.complex128 if ref.is_complex() else torch.float64
    reference = ref.to(wide)
    scale = reference.abs().amax(dim=1)
    if continuous:
        delta = (new.to(wide) - reference).abs()
    else:
        left, right = ref.to(torch.int64), new.to(torch.int64)
        if ref.dtype == torch.uint64:
            left, right = left ^ -(2**63), right ^ -(2**63)
        delta = torch.where((left < 0) != (right < 0),
                            left.to(torch.float64).abs() + right.to(torch.float64).abs(),
                            (right - left).to(torch.float64).abs())
    diff = delta.amax(dim=1)
    # Equal infinities follow torch.equal semantics and have zero difference.
    diff = torch.where(equal, 0.0, diff)
    return torch.stack((diff, scale, equal.to(torch.float64)), dim=1)


def _classify(ref, new, *, level, discrete, name, family=None):
    if family is not None and level == "setup":
        raise ValueError("family is not supported at setup level")
    single, double = {"setup": (64 * torch.finfo(torch.float32).eps, 1e-12),
                      "step": (64 * torch.finfo(torch.float32).eps, 1e-12),
                      "run": (1e-4, 1e-10)}[level]
    rows = []
    groups = defaultdict(lambda: defaultdict(list))
    factors = {torch.float32: single, torch.complex64: single,
               torch.float64: double, torch.complex128: double}
    row_factors = {}

    def visit(left, right, path):
        if isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys():
            for key in left:
                visit(left[key], right[key], f"{path}.{key}" if path else str(key))
            return
        if isinstance(left, (list, tuple)) and type(left) is type(right) and len(left) == len(right):
            for i, (a, b) in enumerate(zip(left, right)):
                visit(a, b, f"{path}[{i}]")
            return
        row = dict(name=path or "tensor", dtype=str(left.dtype) if isinstance(left, torch.Tensor) else type(left).__name__,
                   shape=list(left.shape) if isinstance(left, torch.Tensor) else [],
                   max_abs=float("inf"), max_ref=0.0, max_rel=float("inf"), bound=0.0,
                   ratio_to_bound=float("inf"), tier="T2", level=level)
        index = len(rows)
        rows.append(row)
        if not (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                and left.shape == right.shape and left.dtype == right.dtype):
            return
        if family is not None:
            row_factors[index] = factors.get(left.dtype, 0.0) if not discrete else 0.0
        if not left.numel():
            row.update(max_abs=0.0, max_rel=0.0, ratio_to_bound=0.0, tier="T0")
            return
        groups[left.device, left.dtype][left.numel()].append((index, left, right))

    visit(ref, new, name or "")
    with torch.no_grad():
        for (device, dtype), sizes in groups.items():
            indices, metrics = [], []
            for size, leaves in sizes.items():
                # Bound temporary storage for large state trees; small scalar leaves
                # share reductions instead of launching kernels one leaf at a time.
                chunk = max(1, 1_048_576 // size)
                for start in range(0, len(leaves), chunk):
                    batch = leaves[start:start + chunk]
                    left = torch.stack([a.detach().reshape(-1) for _, a, _ in batch])
                    right = torch.stack([b.detach().to(device=device).reshape(-1) for _, _, b in batch])
                    metrics.append(_reduce(left, right))
                    indices.extend(index for index, _, _ in batch)
            values = torch.cat(metrics).cpu().tolist()
            continuous = dtype.is_floating_point or dtype.is_complex
            factor = factors[dtype] if continuous and not discrete else 0.0
            for index, (diff, scale, equal) in zip(indices, values):
                bound = factor * scale if factor else 0.0
                within = scale > 0 and math.isfinite(bound) and diff <= bound
                tier = "T0" if equal else "T1" if factor and within else "T2"
                rows[index].update(max_abs=diff, max_ref=scale,
                                   max_rel=diff / scale if scale else (0.0 if equal else float("inf")),
                                   bound=bound, ratio_to_bound=diff / bound if bound else (0.0 if equal else float("inf")),
                                   tier=tier)
    if family is not None:
        # Reuse the transferred metrics across all device/dtype groups.
        family_key = family if callable(family) else family.get
        keys = [family_key(row["name"]) for row in rows]
        maxima = defaultdict(float)
        for row, key in zip(rows, keys):
            if key is not None:
                maxima[key] = max(maxima[key], row["max_ref"])
        for index, (row, key) in enumerate(zip(rows, keys)):
            row["structural_zero"] = False
            if key is None:
                continue
            row.update(family=key, family_max_ref=maxima[key])
            floor = 1e-6 * row["family_max_ref"]
            factor = row_factors.get(index, 0.0)
            if factor and row["max_ref"] < floor:
                bound = factor * floor
                equal = row["tier"] == "T0"
                within = math.isfinite(bound) and row["max_abs"] <= bound
                row.update(structural_zero=True, bound=bound,
                           ratio_to_bound=row["max_abs"] / bound if bound else (0.0 if equal else float("inf")),
                           tier="T0" if equal else "T1" if within else "T2")
    return dict(tier=max((row["tier"] for row in rows), default="T0"), level=level, tensors=rows)


def classify_against_pin(pinned, current, *, level, family=None):
    """Return pin tiers; family is a path-to-key mapping or callable (None opts out)."""
    return _classify(pinned, current, level=level, discrete=False, name=None, family=family)


def _assert_report(report):
    failures = [f"{row['name']}: diff={row['max_abs']:.17g}, bound={row['bound']:.17g}, level={row['level']}"
                + (f", growth={row['growth']}" if "growth" in row else "")
                for row in report["tensors"] if row["tier"] == "T2"]
    assert not failures, "Numerical equivalence T2: " + "; ".join(failures)


def assert_numerically_equivalent(ref, new, *, level, discrete=False, name=None, family=None):
    """Assert bounds with optional path-to-family structural-zero scales.

    Families apply only at step/run levels. Unmapped tensors retain per-tensor
    bounds, and discrete tensors always require exact equality.
    family may be a mapping or a callable from report paths to family keys;
    a missing key or None leaves a path outside all families.
    """
    report = _classify(ref, new, level=level, discrete=discrete, name=name, family=family)
    _assert_report(report)
    return report


def assert_run_level_equivalent(ref_half, new_half, ref_full, new_full, *, discrete=False, name=None, family=None):
    """Check run bounds at both horizons and <=4x difference growth per tensor.

    A zero half-horizon difference allows a nonzero full difference only within
    the full run bound. Half/full values may have different shapes (histories).
    Optional family mappings or callables use report paths, with None meaning
    no family; scales are computed independently at each horizon. A full-horizon
    structural zero passes the growth check whenever its max_abs stays within
    its floored bound (factor * 1e-6 * family_max_ref), even above 4x growth.
    Noise below this floor is not a growing mode. All other rows keep the 4x rule.
    """
    half = _classify(ref_half, new_half, level="run", discrete=discrete, name=name, family=family)
    full = _classify(ref_full, new_full, level="run", discrete=discrete, name=name, family=family)
    half_rows = {row["name"]: row for row in half["tensors"]}
    full_rows = {row["name"]: row for row in full["tensors"]}
    rows = []
    for path in dict.fromkeys([*half_rows, *full_rows]):
        a, b = half_rows.get(path), full_rows.get(path)
        row = dict(b or a)
        growth = b["max_abs"] / a["max_abs"] if a and b and a["max_abs"] else (
            0.0 if a and b and b["max_abs"] == 0 else float("inf"))
        passed = a is not None and b is not None and a["dtype"] == b["dtype"] and (
            b["max_abs"] <= b["bound"] if b.get("structural_zero", False) or not a["max_abs"]
            else b["max_abs"] <= 4 * a["max_abs"])
        row.update(half=a, full=b, growth=growth,
                   tier=max(a["tier"], b["tier"]) if passed else "T2")
        rows.append(row)
    report = dict(tier=max((row["tier"] for row in rows), default="T0"), level="run", tensors=rows)
    _assert_report(report)
    return report


def record_pin_comparison(ref, new, *, nodeid, level, name=None, discrete=False):
    """Record final report rows with a nodeid after a pin's torch.equal fast path."""
    report = _classify(ref, new, level=level, discrete=discrete, name=name)
    log = os.environ.get("WITWIN_PIN_TIER_LOG")
    if log:
        with open(log, "a") as handle:
            for row in report["tensors"]:
                handle.write(json.dumps(dict(row, nodeid=nodeid)) + "\n")
    _assert_report(report)
    return report
