"""M3 external-reference generation wiring: honest fail-closed behavior.

These tests exercise ``benchmark.rf_tidy3d_references`` WITHOUT touching the cloud.
Since the F2c adapter RF-port mapping, all five target scenes export with a genuine
drive (``sources >= 1``) and are runnable: the ``rf/rectangular_waveguide`` TE10
``ModeSource`` launch, the two ``WavePort`` port scenes (mapped to a modal launch +
receive mode monitors), and the two antenna ``LumpedPort`` feeds (mapped to an
equivalent ``UniformCurrentSource`` current injection). The tests assert that the
runnable gate is reached only by a genuine source, that ``run_cloud=False`` suppresses
the cloud without fabricating a cache, and that the runnable -> cloud -> cache branch is
wired and reachable (stubbed cloud run), so the wiring is not a vacuous always-pending
stub.

Falsification records live in
``docs/assessments/f2-rf-trio-acceptance-2026-07-21.md``.
"""

from __future__ import annotations

import pytest

import benchmark.rf_tidy3d_references as refs
from benchmark.rf_tidy3d_references import (
    GENERATED,
    PENDING,
    REFERENCE_TARGETS,
    ReferenceRecord,
    attempt_reference,
)


# Every reference target now exports with a genuine drive and is runnable.
RUNNABLE_TARGETS = tuple(REFERENCE_TARGETS)


@pytest.mark.parametrize("name", RUNNABLE_TARGETS)
def test_target_scene_exports_runnable_with_a_source(name, monkeypatch):
    """Every target exports sources>=1 and stops cleanly at the suppressed gate.

    With ``run_cloud=False`` generation reaches the runnable gate on a genuine source
    but does NOT touch the cloud (any cloud call fails the test) and does NOT fabricate
    a cache -- it records the suppression reason with ``pending-generation``.
    """
    def _forbid_cloud(*args, **kwargs):
        raise AssertionError("run_cloud=False must not reach the cloud")

    monkeypatch.setattr(refs, "_run_cloud_reference", _forbid_cloud)

    record = attempt_reference(name, run_cloud=False)

    assert record.runnable is True
    assert record.exported_sources is not None and record.exported_sources >= 1
    assert record.exported_monitors is not None and record.exported_monitors >= 1
    assert record.status == PENDING  # suppressed, not generated (no cloud)
    assert record.task_id is None
    assert record.cost_flexcredits is None
    assert record.cache is None
    assert "suppressed" in record.reason


def test_runnable_export_reaches_cloud_branch(monkeypatch):
    """FALSIFICATION: forcing the runnable gate open reaches a real cloud+cache branch.

    This proves the wiring is not an always-pending stub: with the gate open and a
    stubbed cloud run, ``attempt_reference`` reaches the generation branch, records
    the task id / cost, and writes a cache. The ONLY thing keeping the four target
    scenes pending is the source-count gate.
    """
    monkeypatch.setattr(refs, "_runnable_reason", lambda td_sim: (True, ""))

    saved = {}

    def _fake_cloud(name, td_sim, frequencies):
        return {"monitor": {"kind": "field"}}, "task-abc123", 0.0417

    def _fake_save(name, *, frequencies, monitors, cache_key=None):
        saved["name"] = name
        return f"/cache/{name}.h5"

    monkeypatch.setattr(refs, "_run_cloud_reference", _fake_cloud)
    monkeypatch.setattr(refs, "save_tidy3d_result", _fake_save)

    record = attempt_reference("rf/coax_thru", run_cloud=True)

    assert record.status == GENERATED
    assert record.task_id == "task-abc123"
    assert record.cost_flexcredits == pytest.approx(0.0417)
    assert record.cache == "/cache/rf/coax_thru.h5"
    assert saved["name"] == "rf/coax_thru"
    assert record.reason == ""


def test_cloud_failure_is_recorded_fail_closed(monkeypatch):
    """A cloud failure records pending-generation with the reason; never fabricates."""
    monkeypatch.setattr(refs, "_runnable_reason", lambda td_sim: (True, ""))

    def _boom(name, td_sim, frequencies):
        raise RuntimeError("estimated cost 9.9 FlexCredits exceeds the per-scene budget")

    fabricated = {"called": False}

    def _fake_save(*args, **kwargs):
        fabricated["called"] = True
        return "/cache/should-not-happen.h5"

    monkeypatch.setattr(refs, "_run_cloud_reference", _boom)
    monkeypatch.setattr(refs, "save_tidy3d_result", _fake_save)

    record = attempt_reference("rf/coax_thru", run_cloud=True)

    assert record.status == PENDING
    assert "cloud generation failed" in record.reason
    assert "budget" in record.reason
    assert record.cache is None
    assert fabricated["called"] is False


def test_unknown_target_is_rejected():
    record = attempt_reference("rf/nonexistent")
    assert isinstance(record, ReferenceRecord)
    assert record.status == PENDING
    assert "unknown reference target" in record.reason


def test_marker_rebuild_reconstructs_records_without_cloud(tmp_path, monkeypatch):
    """``rebuild_from_markers`` rebuilds RESULTS rows from on-disk markers only.

    It reconstructs a generated record for a scene with a marker and a pending
    placeholder for one without -- with no cloud call and no scene rebuild.
    """
    monkeypatch.setattr(refs, "CACHE_DIR", tmp_path)
    results_md = tmp_path / "RESULTS.md"
    monkeypatch.setattr(refs, "RESULTS_MD", results_md)

    generated = ReferenceRecord(
        scene="rf/coax_thru",
        status=GENERATED,
        exported_sources=1,
        exported_monitors=2,
        runnable=True,
        task_id="task-xyz",
        cost_flexcredits=0.025,
        cache=str(tmp_path / "rf" / "coax_thru.h5"),
    )
    refs.write_marker(generated)

    roundtrip = refs.load_marker("rf/coax_thru")
    assert roundtrip.status == GENERATED
    assert roundtrip.task_id == "task-xyz"
    assert roundtrip.exported_monitors == 2

    records = refs.rebuild_from_markers(["rf/coax_thru", "antenna/patch"])
    by_scene = {record.scene: record for record in records}
    assert by_scene["rf/coax_thru"].status == GENERATED
    assert by_scene["antenna/patch"].status == PENDING  # no marker on disk
    text = results_md.read_text(encoding="utf-8")
    assert "task-xyz" in text
    assert "rf/coax_thru" in text
