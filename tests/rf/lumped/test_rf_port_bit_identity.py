"""Existing LumpedPort, TerminalPort and WavePort numerics stay bitwise unchanged.

The coaxial-port, resistor, PEC-frame, baseband and smoothing additions are new
classes and dispatch branches. This gate runs one fixed scene script in fresh
interpreters, once on a frozen source tree (``git archive`` or its staged tree) and once on the working
tree, and compares every port voltage/current, available power, network S matrix and
final Yee field with ``torch.equal``.

The frozen source is the node-consistent staircase PEC fix (an E edge is shorted iff
both endpoints are conductor nodes). That rule legitimately changes the shorted edge
sets of the scene's half-cell-faced terminal Boxes and curved coax, so the original
pre-port base ``442491e7`` no longer applies. Re-freeze evidence: ``442491e7`` with
only that ``fdtd/runtime/materials.py`` change applied reproduces the PEC-fix tree
bitwise on all 30 payload keys, so the port additions still change no existing port
numerics under the new rule.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import torch

import witwin.maxwell as mw

from tests.support.baseline_tree_env import probe_environment
from tests.support.pinned_baselines import baseline_tree as _baseline_tree

# Per-frequency pulse reference periods rescale the c971ac8a absolute spectra by
# f/f_c for amplitudes and (f/f_c)^2 for powers. Final Yee fields are unchanged.
# Re-pinned 2026-11-04 to f36cdaf1 for the observer time origin (A) (physical Yee labels: raw E/H spectra and port V/I rotate by exp(+i omega dt), derived invariants hold to FP32 rounding, all else bitwise; user-approved T2; per-key proof docs/assessments/observer-origin-repin-cpu-2026-11-03/, <evidence-root>/coverage-observer-origin/repin-proof-final/). Previous: 95a3ec3f.
BASE_COMMIT = "f36cdaf1"
ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tests" / "support" / "rf_port_bit_identity_scene.py"

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="bit-identity gate runs the CUDA FDTD solver")


def _run(source_root: Path, output: Path) -> dict:
    env = probe_environment(source_root, baseline=source_root != ROOT)
    subprocess.run([sys.executable, str(SCRIPT), str(output)], check=True, env=env, cwd=str(output.parent))
    return torch.load(output, weights_only=False)


def test_existing_port_paths_are_bitwise_identical_to_the_frozen_source(tmp_path):
    base_root = _baseline_tree(BASE_COMMIT, tmp_path)
    base = _run(base_root, tmp_path / "base.pt")
    branch = _run(ROOT, tmp_path / "branch.pt")
    assert Path(base.pop("module")).is_relative_to(base_root)
    assert Path(branch.pop("module")).resolve() == Path(mw.__file__).resolve()
    assert base.keys() == branch.keys()
    assert any(key.startswith("direct/terminal/") for key in base)
    assert any(key.startswith("sweep/network/") for key in base)
    assert any(key.startswith("wave/") and "/field/" in key for key in base)
    mismatched = [key for key in base if not torch.equal(base[key], branch[key])]
    assert not mismatched, f"bitwise changes in existing port numerics: {mismatched}"
