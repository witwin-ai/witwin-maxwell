"""Existing monitor outputs are bit-identical to the pre-change baseline.

Runs one short FDTD scene with Point, Plane, Flux, FinitePlane, Mode,
ClosedSurface, FieldTime and FluxTime monitors in fresh interpreters: once on
``git archive <baseline> witwin`` (or its staged tree) and twice on this tree (without and with the
new in-simulation monitors added to the same scene), then compares every
recorded tensor with ``torch.equal``.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.support.baseline_tree_env import probe_environment
from tests.support.pinned_baselines import baseline_tree as _baseline_tree

# Absolute pulse spectra intentionally changed from a5c7a318 to reference-period Fourier integrals,
# and at 95a3ec3f to per-frequency reference periods ((f/f_c)^k on the 2.0/2.3 GHz bins).
# Baseline re-baselined at f52aa33f for the additive-J (n+1/2) injection timing; proven equal to the old baseline plus only that hunk (<evidence-root>/coverage-e-timing-proof/).
# Re-pinned 2026-11-04 to f36cdaf1 for the observer time origin (A) (physical Yee labels: raw E/H spectra and port V/I rotate by exp(+i omega dt), derived invariants hold to FP32 rounding, all else bitwise; user-approved T2; per-key proof docs/assessments/observer-origin-repin-cpu-2026-11-03/, <evidence-root>/coverage-observer-origin/repin-proof-final/). Previous: f52aa33f.
BASELINE_COMMIT = os.environ.get("WITWIN_MONITOR_BASELINE", "f36cdaf1")
_RUNNER = Path(__file__).with_name("_bit_identity_runner.py")
# Closed surfaces became watertight on the grid after the baseline (faces snap
# to Yee nodes and to the adjacent faces' planes); the runner's off-node box is
# therefore compared only between runs of this tree.
_CHANGED_SINCE_BASELINE = ("box",)
_TREE = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for native FDTD")


def _run(tree: Path, out: Path, *extra):
    env = probe_environment(tree, baseline=tree != _TREE)
    subprocess.run([sys.executable, str(_RUNNER), str(out), *extra], check=True, env=env, cwd=str(out.parent))
    return torch.load(out, weights_only=False)


def _assert_identical(reference, candidate, *, label):
    missing = sorted(set(reference) - set(candidate) - {"__file__"})
    assert not missing, f"{label}: keys missing {missing[:10]}"
    mismatched = []
    for key, value in reference.items():
        if key == "__file__":
            continue
        other = candidate[key]
        if isinstance(value, torch.Tensor):
            if not (isinstance(other, torch.Tensor) and value.dtype == other.dtype and torch.equal(value, other)):
                mismatched.append(key)
        elif value != other and not (value != value and other != other):
            mismatched.append(key)
    assert not mismatched, f"{label}: {len(mismatched)} mismatched entries, e.g. {mismatched[:10]}"
    return sum(isinstance(value, torch.Tensor) for value in reference.values())


def test_existing_monitor_outputs_are_bit_identical_to_baseline(tmp_path):
    baseline_tree = _baseline_tree(BASELINE_COMMIT, tmp_path)

    baseline = _run(baseline_tree, tmp_path / "baseline.pt")
    branch = _run(_TREE, tmp_path / "branch.pt")
    extended = _run(_TREE, tmp_path / "extended.pt", "with-new-monitors")
    assert baseline["__file__"].startswith(str(baseline_tree))
    assert branch["__file__"].startswith(str(_TREE)) and extended["__file__"].startswith(str(_TREE))

    def unchanged(payload):
        return {key: value for key, value in payload.items()
                if key == "__file__" or key.split("/")[1].split("::")[0] not in _CHANGED_SINCE_BASELINE}

    compared = _assert_identical(unchanged(baseline), unchanged(branch), label="baseline vs branch")
    assert compared > 50
    _assert_identical(unchanged(baseline), unchanged(extended), label="baseline vs branch with new monitors")
    assert any(key.startswith("monitor/box/") for key in branch)
    _assert_identical(branch, extended, label="branch vs branch with new monitors")
