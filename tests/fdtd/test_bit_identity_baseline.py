"""Resume fingerprints and pulsed-source fields stay bit-identical to baselines.

Two coverage merges silently changed existing behavior: a new compiled
material-model key (``pmc_occupancy``) changed every resume fingerprint, and a
regrouped float32 product in the Gaussian-pulse source-time evaluation shifted
pulsed surface-source fields at round-off. These tests run the probe in fresh
interpreters against ``git archive <ref> witwin`` (or its staged tree) and the current tree:

* fingerprints of PMC-free scenes must equal the pre-PMC baseline
  (``WITWIN_FINGERPRINT_BASELINE_REF``, default 739d1ca8);
* fields of point-dipole, soft plane-wave, Gaussian-beam, mode-source and TFSF
  (box and pulsed Bloch slab) runs must be ``torch.equal`` to the pre-waveform
  baseline (``WITWIN_FIELD_BASELINE_REF``, default bacac15f; re-baselined at bacac15f for the pulsed TFSF
  physical entry-plane incident reference, proven equal to f52aa33f plus only that hunk
  (<evidence-root>/coverage-e-timing-proof/results/tfsf_ref_gpu/); f52aa33f was the
  additive-J (n+1/2) re-baseline, previously 42960ad2). The baseline already contains the soft-sheet
  sign fix, so every config is compared directly.
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

_REPO = Path(__file__).resolve().parents[2]
_PROBE = Path(__file__).with_name("_bit_identity_probe.py")
# Re-pinned 2026-11-06 to 9d7a94fb for the TFSF auxiliary-line band-edge Part 2 (1 resume fingerprint: tfsf_plane_wave gains the tfsf_auxiliary_termination marker); T2 covered by the user's 2026-11-04 band-edge pre-approval (roadmap 11-04 23:09); evidence <evidence-root>/bitid-repin-part2-20261106/. Previous: fc72f2c3 (re-pinned 2026-11-04, user-approved T2; evidence <evidence-root>/coverage-corner-hy/wall-e-ab/ and <evidence-root>/coverage-tfsf-raw-phase/).
_FINGERPRINT_REF = os.environ.get("WITWIN_FINGERPRINT_BASELINE_REF", "9d7a94fb")
# Re-pinned 2026-11-04 to fc72f2c3 for wall-E soft-sheet wall rows (4 field movers) and the (ii) physical TFSF timing (3 field movers, 1 resume fingerprint); user-approved T2 2026-11-04; evidence <evidence-root>/coverage-corner-hy/wall-e-ab/ and <evidence-root>/coverage-tfsf-raw-phase/. Previous: bacac15f.
_FIELD_REF = os.environ.get("WITWIN_FIELD_BASELINE_REF", "fc72f2c3")
# Multi-sample subpixel scenes changed on purpose with the exact sub-sample fill
# (82a17c9e); their fingerprints are pinned to the first tree that contains it.
_SUBPIXEL_FINGERPRINT_REF = os.environ.get("WITWIN_SUBPIXEL_FINGERPRINT_BASELINE_REF", "9237642a")
_MULTI_SAMPLE_SCENES = frozenset({"subpixel_dispersive"})

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")

_TREES = {}
_PAYLOADS = {}


def _tree(ref, tmp_path_factory):
    if ref not in _TREES:
        _TREES[ref] = _baseline_tree(ref, tmp_path_factory.mktemp(f"baseline_{ref}"))
    return _TREES[ref]


def _run_probe(tree: Path, mode: str, output: Path) -> dict:
    environment = probe_environment(tree, baseline=tree != _REPO)
    subprocess.run([sys.executable, str(_PROBE), mode, str(output)], check=True, cwd=str(output.parent), env=environment)
    payload = torch.load(output, weights_only=False)
    assert Path(payload["module_file"]).resolve().is_relative_to(tree.resolve())
    return payload


def _payload(mode, ref, tmp_path_factory):
    key = (mode, ref)
    if key not in _PAYLOADS:
        directory = tmp_path_factory.mktemp(f"{mode}_{ref or 'current'}")
        tree = _REPO if ref is None else _tree(ref, tmp_path_factory)
        _PAYLOADS[key] = _run_probe(tree, mode, directory / "payload.pt")
    return _PAYLOADS[key]


def test_resume_fingerprints_match_pre_pmc_baseline(tmp_path_factory):
    baseline = dict(_payload("fingerprints", _FINGERPRINT_REF, tmp_path_factory)["fingerprints"])
    subpixel_baseline = _payload("fingerprints", _SUBPIXEL_FINGERPRINT_REF, tmp_path_factory)["fingerprints"]
    baseline.update({name: subpixel_baseline[name] for name in _MULTI_SAMPLE_SCENES})
    current = _payload("fingerprints", None, tmp_path_factory)["fingerprints"]
    assert set(current) == set(baseline) and len(current) >= 14
    mismatched = sorted(name for name in current if current[name] != baseline[name])
    assert not mismatched, mismatched


def test_pmc_scene_fingerprint_stays_distinct():
    import witwin.maxwell as mw
    from witwin.maxwell.fdtd.resume import _configuration_fingerprint

    def fingerprint(material):
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.3, 0.3),) * 3),
            grid=mw.GridSpec.uniform(0.05),
            boundary=mw.BoundarySpec.pml(num_layers=4),
            device="cuda",
        )
        scene.add_structure(mw.Structure(geometry=mw.Box(position=(0.1, 0, 0), size=(0.1, 0.1, 0.1)), material=material))
        scene.add_source(mw.PointDipole(position=(-0.1, 0, 0), polarization="Ez", width=0.05,
                                        source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.5e9)))
        scene.add_monitor(mw.PointMonitor("probe", (0.0, 0.0, 0.0), fields=("Ez",)))
        prepared = mw.Simulation.fdtd(scene, frequencies=[1e9], run_time=mw.TimeConfig(time_steps=4)).prepare()
        model = prepared.solver._compiled_material_model
        return _configuration_fingerprint(prepared.solver), model.get("pmc_occupancy")

    pmc, pmc_occupancy = fingerprint(mw.Material.pmc())
    pec, pec_pmc_occupancy = fingerprint(mw.Material.pec())
    assert pmc_occupancy is not None and pec_pmc_occupancy is None
    assert pmc != pec


_FIELD_CONFIGS = (
    "bloch_tfsf_pulse",
    "soft_plane_wave_gaussian",
    "soft_plane_wave_ricker",
    "gaussian_beam_gaussian",
    "gaussian_beam_ricker",
    "mode_source_gaussian",
    "mode_source_ricker",
    "tfsf_plane_wave_gaussian",
    "tfsf_plane_wave_ricker",
    "dipole_gaussian",
    "dipole_cw",
)


@pytest.mark.parametrize("config", _FIELD_CONFIGS)
def test_source_fields_match_pre_waveform_baseline(tmp_path_factory, config):
    baseline = _payload("fields", _FIELD_REF, tmp_path_factory)["fields"][config]
    current = _payload("fields", None, tmp_path_factory)["fields"][config]
    assert set(current) == set(baseline)
    assert any(float(tensor.abs().max()) > 0.0 for tensor in baseline.values())
    for name, tensor in baseline.items():
        assert torch.equal(current[name], tensor), f"{config}.{name}"
