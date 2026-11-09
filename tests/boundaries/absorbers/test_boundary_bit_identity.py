"""Existing boundary configurations stay bit-identical to a baseline tree.

The absorbing-boundary extensions (second-order Mur, mode-matched faces,
internal absorbers, all-pass mode ABC) must not perturb any pre-existing
boundary kind. For representative configurations this test runs the same probe
in fresh interpreters against ``git archive <baseline> witwin`` (or its staged tree) and against the
current tree, then requires ``torch.equal`` on every prepared boundary/update
tensor, the Mur histories and the fields after a short run.

The resume configuration fingerprint of every configuration must also match
the baseline byte for byte, and a disk checkpoint written by the baseline tree
must resume on the current tree to the same final fields.

The baseline defaults to db3bd9b5 (f52aa33f plus the additive-J checkpoint timing metadata, so its disk checkpoint is resumable; re-baselined at f52aa33f for the additive-J (n+1/2) injection timing; proven equal to the old baseline plus only that hunk (<evidence-root>/coverage-e-timing-proof/); previously 74900e42) and can be overridden with ``WITWIN_BOUNDARY_BASELINE_REF``. The
original branch point e56b59fb is no longer a valid baseline after that merge:
master's own waveform changes alter the pulsed Bloch TFSF fields at float32
round-off and every resume fingerprint, independent of this work.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.support.baseline_tree_env import probe_environment
from tests.support.pinned_baselines import baseline_tree as _baseline_tree

_REPO = Path(__file__).resolve().parents[3]
_PROBE = Path(__file__).with_name("_boundary_state_probe.py")
# Re-pinned 2026-11-04 to f36cdaf1 for the observer time origin (A) (physical Yee labels: raw E/H spectra and port V/I rotate by exp(+i omega dt), derived invariants hold to FP32 rounding, all else bitwise; user-approved T2; per-key proof docs/assessments/observer-origin-repin-cpu-2026-11-03/, <evidence-root>/coverage-observer-origin/repin-proof-final/). Previous: db3bd9b5.
# Re-pinned 2026-11-04 to fc72f2c3 for wall-E soft-sheet wall rows (4 field movers) and the (ii) physical TFSF timing (3 field movers, 1 resume fingerprint); user-approved T2 2026-11-04; evidence <evidence-root>/coverage-corner-hy/wall-e-ab/ and <evidence-root>/coverage-tfsf-raw-phase/. Previous: f36cdaf1.
_BASELINE_REF = os.environ.get("WITWIN_BOUNDARY_BASELINE_REF", "fc72f2c3")
_CONFIGS = ("cpml_all", "cpml_pec", "bloch_xy_cpml_z", "mur_first_order", "mur_cpml_mixed", "pmc_periodic")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")


@pytest.fixture(scope="module")
def baseline_tree(tmp_path_factory):
    return _baseline_tree(_BASELINE_REF, tmp_path_factory.mktemp("boundary_baseline"))


def _run_probe(tree: Path, config: str, output: Path, *extra: str) -> dict:
    environment = probe_environment(tree, baseline=tree != _REPO)
    subprocess.run(
        [sys.executable, str(_PROBE), config, str(output), *extra],
        check=True,
        cwd=str(output.parent),
        env=environment,
    )
    payload = torch.load(output, weights_only=False)
    assert Path(payload["module_file"]).resolve().is_relative_to(tree.resolve())
    return payload


def _assert_identical(left, right, label):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), label
        assert left.dtype == right.dtype and left.shape == right.shape, label
        assert torch.equal(left, right), label
        return
    if isinstance(left, dict):
        assert set(left) == set(right), f"{label}: keys {sorted(set(left) ^ set(right))}"
        for key in left:
            _assert_identical(left[key], right[key], f"{label}.{key}")
        return
    if isinstance(left, (list, tuple)):
        assert len(left) == len(right), label
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_identical(a, b, f"{label}[{index}]")
        return
    assert left == right, f"{label}: {left!r} != {right!r}"


_PAYLOADS = {}


def _payloads(baseline_tree, tmp_path_factory, config):
    if config not in _PAYLOADS:
        directory = tmp_path_factory.mktemp(f"probe_{config}")
        _PAYLOADS[config] = (
            _run_probe(baseline_tree, config, directory / "baseline.pt"),
            _run_probe(_REPO, config, directory / "current.pt"),
        )
    return _PAYLOADS[config]


@pytest.mark.parametrize("config", _CONFIGS)
def test_existing_boundary_configuration_is_bit_identical(baseline_tree, tmp_path_factory, config):
    baseline, current = _payloads(baseline_tree, tmp_path_factory, config)
    assert baseline["prepared_state"], "probe captured no boundary tensors"
    assert baseline["fields"]["Ez"].abs().max() > 0, "probe run left the fields at zero"
    for key in ("prepared_state", "scalars", "mur", "fields", "mur_after"):
        _assert_identical(baseline[key], current[key], f"{config}.{key}")


@pytest.mark.parametrize("config", _CONFIGS)
def test_existing_boundary_resume_fingerprint_is_unchanged(baseline_tree, tmp_path_factory, config):
    baseline, current = _payloads(baseline_tree, tmp_path_factory, config)
    assert isinstance(baseline["fingerprint"], str) and len(baseline["fingerprint"]) == 64
    assert current["fingerprint"] == baseline["fingerprint"]


def test_baseline_disk_checkpoint_resumes_on_current_tree(baseline_tree, tmp_path_factory, tmp_path):
    config = "mur_cpml_mixed"
    checkpoint = tmp_path / "baseline_checkpoint.pt"
    _run_probe(baseline_tree, config, tmp_path / "saved.pt", "save", str(checkpoint))
    resumed = _run_probe(_REPO, config, tmp_path / "resumed.pt", "resume", str(checkpoint))
    baseline, _ = _payloads(baseline_tree, tmp_path_factory, config)
    _assert_identical(baseline["fields"], resumed["fields"], f"{config}.resumed_fields")
