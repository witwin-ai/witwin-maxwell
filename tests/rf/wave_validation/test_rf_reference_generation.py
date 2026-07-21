"""M3 external-reference generation wiring: honest fail-closed behavior.

These tests exercise ``benchmark.rf_tidy3d_references`` WITHOUT touching the cloud.
They assert that the four owner-authorized target scenes export source-less (their
port / lumped-port excitation has no adapter source mapping) and therefore
fail-close at the runnable gate with ``pending-generation`` and a recorded reason,
never fabricating an ``.h5`` cache -- and that the runnable -> cloud -> cache branch
is genuinely wired and reachable (proven by monkeypatching the gate open with a
stubbed cloud run, so the wiring is not a vacuous always-pending stub).

Falsification records live in
``docs/assessments/e2-rf-scenes-acceptance-2026-07-19.md``.
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


TARGETS = tuple(REFERENCE_TARGETS)


@pytest.mark.parametrize("name", TARGETS)
def test_target_scene_exports_source_less_and_fails_closed(name, tmp_path, monkeypatch):
    """Each target exports with sources=0, so generation refuses before any cloud cost."""
    # Guard: if the gate ever proceeded to the cloud, the test must fail loudly
    # rather than spend credits.
    def _forbid_cloud(*args, **kwargs):
        raise AssertionError("cloud generation must not be reached for a source-less export")

    monkeypatch.setattr(refs, "_run_cloud_reference", _forbid_cloud)

    record = attempt_reference(name, run_cloud=False)

    assert record.status == PENDING
    assert record.runnable is False
    assert record.exported_sources == 0
    assert record.task_id is None
    assert record.cost_flexcredits is None
    assert record.cache is None
    assert "sources=0" in record.reason
    # No numerical cross-reference cache was fabricated.
    assert not refs.cache_path(name).exists()


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
