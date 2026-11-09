"""The TerminalWavePort branch leaves every existing port path bitwise unchanged.

Fresh interpreters run ``tests/support/rf_port_bit_identity_terminal_wave_scene.py``
once on ``git archive`` of the pre-change base (or its staged tree) and once on the working tree. The
payload covers direct and swept LumpedPort/TerminalPort runs, the coax WavePort and a
two-mode coupled-pair WavePort sweep (port V/I, available power, network S, modal
impedances/propagation constants and final Yee fields). Exact pins pass immediately;
rounding-only mismatches use the shared run-level numerical-equivalence bounds.
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
from tests.support.numerical_equivalence import record_pin_comparison

# The old 7944434f pin predates be24cc99's intended staircase PEC edge-set
# correction: short an edge only when both endpoints are conductor nodes.
# c971ac8a also includes the intended pulse reference-period normalization;
# f915a989 first changed this probe's port spectra after the node-rule correction.
# 95a3ec3f's per-frequency reference period rescales the shared multi-frequency
# (0.4/0.6 GHz) pulsed spectra by (f/f_c)^k; single-frequency pair/wave runs are unchanged.
# Re-pinned 2026-11-04 to f36cdaf1 for the observer time origin (A) (physical Yee labels: raw E/H spectra and port V/I rotate by exp(+i omega dt), derived invariants hold to FP32 rounding, all else bitwise; user-approved T2; per-key proof docs/assessments/observer-origin-repin-cpu-2026-11-03/, <evidence-root>/coverage-observer-origin/repin-proof-final/). Previous: 95a3ec3f.
BASE_COMMIT = "f36cdaf1"
ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tests" / "support" / "rf_port_bit_identity_terminal_wave_scene.py"

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="bit-identity gate runs the CUDA FDTD solver")


def _run(source_root: Path, output: Path) -> dict:
    env = probe_environment(source_root, baseline=source_root != ROOT)
    subprocess.run([sys.executable, str(SCRIPT), str(output)], check=True, env=env, cwd=str(output.parent))
    return torch.load(output, weights_only=False)


def test_existing_port_paths_and_two_mode_waveport_are_bitwise_identical(tmp_path, request):
    base_root = _baseline_tree(BASE_COMMIT, tmp_path)
    base = _run(base_root, tmp_path / "base.pt")
    branch = _run(ROOT, tmp_path / "branch.pt")
    assert Path(base.pop("module")).is_relative_to(base_root)
    assert Path(branch.pop("module")).resolve() == Path(mw.__file__).resolve()
    assert base.keys() == branch.keys()
    assert any(key.startswith("pair/") and "/field/" in key for key in base)
    assert any(key.startswith("pair/network/") for key in base)
    assert any(key.startswith("direct/terminal/") for key in base)
    for key in base:
        if base[key].dtype == branch[key].dtype and torch.equal(base[key], branch[key]):
            continue
        record_pin_comparison(base[key], branch[key], nodeid=request.node.nodeid,
                              name=key, level="run")
