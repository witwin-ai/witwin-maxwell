"""The graph-captured port step phase leaves every port result bitwise unchanged.

Single runs with lumped, terminal, coaxial and wire-gap ports replay the port
terminations, post-source tail and port V/I DFT as one CUDA graph per step,
driven by a precomputed drive table. This gate runs one fixed scene script in
fresh interpreters, once on the frozen baseline ``f52aa33f`` (re-baselined at f52aa33f for the additive-J (n+1/2) injection timing; proven equal to the old baseline plus only that hunk (<evidence-root>/coverage-e-timing-proof/); copper fixtures stay on ``95a3ec3f``)
(``git archive`` or its staged tree) and once on the working tree, and compares every port
voltage/current (phasors and per-step histories), network matrix, monitor
payload, lumped state and final Yee field with ``torch.equal``.
The mixed circuit retains histories and ratios; its absolute spectral keys
are compared in both ongoing-only and transient-only circuit fixtures.
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

# Per-frequency pulse references rescale absolute spectra from c971ac8a;
# per-step histories and final Yee fields are unchanged.
# Re-pinned 2026-11-04 to f36cdaf1 for the observer time origin (A) (physical Yee labels: raw E/H spectra and port V/I rotate by exp(+i omega dt), derived invariants hold to FP32 rounding, all else bitwise; user-approved T2; per-key proof docs/assessments/observer-origin-repin-cpu-2026-11-03/, <evidence-root>/coverage-observer-origin/repin-proof-final/). Previous: f52aa33f.
BASE_COMMIT = "f36cdaf1"
# Re-freeze the compressed copper fixtures with the same per-frequency reference.
# Re-pinned 2026-11-04 to f36cdaf1 for the observer time origin (A) (physical Yee labels: raw E/H spectra and port V/I rotate by exp(+i omega dt), derived invariants hold to FP32 rounding, all else bitwise; user-approved T2; per-key proof docs/assessments/observer-origin-repin-cpu-2026-11-03/, <evidence-root>/coverage-observer-origin/repin-proof-final/). Previous: 95a3ec3f.
COPPER_BASE_COMMIT = "f36cdaf1"
COPPER_FIXTURES = ("copper_terminal", "resume_copper_terminal_150", "resume_copper_terminal_96")
CIRCUIT_ABSOLUTE_KEYS = (
    "monitors/probe/Ez", "monitors/probe/components/Ez", "monitors/probe/data",
    "ports/p1/voltage", "ports/p1/current", "ports/p1/available_power",
    "ports/p2/voltage", "ports/p2/current",
)
ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tests" / "support" / "port_step_graph_bit_identity_scene.py"

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="bit-identity gate runs the CUDA FDTD solver")


def _run(source_root: Path, output: Path, fixtures=()) -> dict:
    env = probe_environment(source_root, baseline=source_root != ROOT)
    subprocess.run([sys.executable, str(SCRIPT), str(output), *fixtures], check=True, env=env, cwd=str(output.parent))
    return torch.load(output, weights_only=False)


def _split_flags(payload: dict) -> dict:
    flags = {key: payload.pop(key) for key in [key for key in payload if key.rsplit("/", 1)[-1].startswith("__")]}
    return {key: bool(value) for key, value in flags.items() if key.endswith("__port_graph")}


def test_port_step_graph_is_bitwise_identical_to_the_frozen_source(tmp_path, copper_torch_kernel):
    base_root = _baseline_tree(BASE_COMMIT, tmp_path)
    base = _run(base_root, tmp_path / "base.pt")
    branch = _run(ROOT, tmp_path / "branch.pt")
    assert Path(base.pop("__module")).is_relative_to(base_root)
    assert Path(branch.pop("__module")).resolve() == Path(mw.__file__).resolve()
    copper_parent = tmp_path / "copper"
    copper_parent.mkdir()
    copper_root = _baseline_tree(COPPER_BASE_COMMIT, copper_parent)
    copper = _run(copper_root, tmp_path / "copper_base.pt", COPPER_FIXTURES)
    assert Path(copper.pop("__module")).is_relative_to(copper_root)
    base.update(copper)
    for payload in (base, branch):
        for fixture in ("circuit", "circuit_ongoing", "circuit_transient"):
            assert payload[f"{fixture}/__field_graph"]
            assert payload[f"{fixture}/__circuit_graph"]
        for key in CIRCUIT_ABSOLUTE_KEYS:
            assert f"circuit/{key}" not in payload
            for fixture in ("circuit_ongoing", "circuit_transient"):
                assert f"{fixture}/{key}" in payload
    _split_flags(base)
    graph = _split_flags(branch)
    assert base.keys() == branch.keys()
    # The gate is only meaningful if the new path actually ran.
    for fixture in ("lumped1", "lumped2_one_driven", "terminal_direct", "coax", "wire_gap", "copper_terminal",
                    "resume_copper_terminal_150", "resume_lumped_100"):
        assert any(graph[key] for key in graph if key.startswith(fixture + "/")), fixture
    for fixture in ("lumped1_nograph", "circuit", "circuit_ongoing", "circuit_transient",
                    "network_embedded", "dipole"):
        assert not any(graph[key] for key in graph if key.startswith(fixture + "/")), fixture
    assert any("/history/" in key for key in base)
    mismatched = [key for key in base
                  if base[key].dtype != branch[key].dtype or not torch.equal(base[key], branch[key])]
    assert not mismatched, f"bitwise changes in port numerics: {mismatched[:20]}"
