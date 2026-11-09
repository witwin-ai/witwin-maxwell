"""Metadata-only resume preflight checks; no field propagation or GPU emulation."""
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from witwin.maxwell.fdtd import resume
from witwin.maxwell.fdtd.checkpoint import FDTDCheckpointSchema, FDTDCheckpointState

from witwin.maxwell.fdtd import observation_resume


def _metadata(*, complex_fields=False, tables=True, entries=True):
    solver = SimpleNamespace(
        dft_enabled=True, complex_fields_enabled=complex_fields,
        _dft_batched_fields={}, _dft_entries=[{"frequency": 1e9}] if entries else [],
        _dft_step=torch.tensor([0], dtype=torch.int64),
        _observer_spectral_entries=[], time_observers=[], dt=1e-12,
    )
    if tables:
        solver._dft_weighted_cos_table = torch.zeros((8, 1), dtype=torch.float32)
        solver._dft_weighted_sin_table = torch.zeros((8, 1), dtype=torch.float32)
    return solver


def _payload(solver, value):
    payload = observation_resume.capture_observations(solver)
    payload["arrays"]["_dft_step"].fill_(value)
    return payload


@pytest.mark.parametrize("saved", [-1, 0, 2, 4, 8])
def test_active_counter_rejection_without_mutation(saved):
    initialized = torch.cuda.is_initialized()
    solver = _metadata()
    payload = _payload(solver, saved)
    copies, updates = [], []
    with pytest.raises(ValueError, match="counter disagrees with the execution step"):
        observation_resume.preflight_observations(solver, payload, 3, copies, updates)
    assert not copies and not updates
    assert int(solver._dft_step.item()) == 0
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("step", [0, 1, 3, 7])
def test_active_counter_must_equal_checkpoint_step_not_current_target(step):
    solver = _metadata()
    copies, updates = [], []
    observation_resume.preflight_observations(solver, _payload(solver, step), step, copies, updates)
    assert len(copies) == 1 and copies[0][0] is solver._dft_step
    assert int(copies[0][1].item()) == step and int(solver._dft_step.item()) == 0
    assert not updates


@pytest.mark.parametrize("options", [
    {"complex_fields": True}, {"tables": False}, {"entries": False},
    {"complex_fields": True, "tables": False},
])
def test_inactive_host_or_complex_counter_does_not_gain_table_step_contract(options):
    solver = _metadata(**options)
    copies, updates = [], []
    observation_resume.preflight_observations(solver, _payload(solver, -1), 3, copies, updates)
    assert len(copies) == 1 and int(copies[0][1].item()) == -1
    assert int(solver._dft_step.item()) == 0


def test_tableless_fresh_complex_has_no_counter_and_remains_supported():
    solver = _metadata(complex_fields=True, tables=False)
    del solver._dft_step
    payload = observation_resume.capture_observations(solver)
    assert "_dft_step" not in payload["arrays"]
    copies, updates = [], []
    observation_resume.preflight_observations(solver, payload, 3, copies, updates)
    assert not copies and not updates


def test_disabled_dft_remains_supported_without_counter_contract():
    solver = _metadata()
    solver.dft_enabled = False
    payload = observation_resume.capture_observations(solver)
    assert not payload["arrays"]
    observation_resume.preflight_observations(solver, payload, 3, [], [])


def test_window_sample_count_is_not_physical_execution_count():
    solver = _metadata()
    solver._dft_sample_count_values = torch.tensor([0], dtype=torch.int64)
    payload = _payload(solver, 3)
    copies, updates = [], []
    observation_resume.preflight_observations(solver, payload, 3, copies, updates)
    assert int(payload["arrays"]["_dft_sample_count_values"].item()) == 0
    assert any(target is solver._dft_step for target, _ in copies)


@pytest.mark.parametrize("bad,error", [(torch.tensor([3.0]), TypeError), (torch.tensor(3), ValueError), (torch.tensor([3, 3]), ValueError)])
def test_existing_tensor_schema_rejections_preserved(bad, error):
    solver = _metadata()
    payload = _payload(solver, 3)
    payload["arrays"]["_dft_step"] = bad
    with pytest.raises(error):
        observation_resume.preflight_observations(solver, payload, 3, [], [])
    assert int(solver._dft_step.item()) == 0


def _checkpoint(step=3, shutoff=False):
    schema_args = {field.name: () for field in fields(FDTDCheckpointSchema) if field.name != "version"}
    schema = FDTDCheckpointSchema(version=2, **schema_args)
    return resume.FDTDResumeCheckpoint(
        step=step, total_steps=8, dt=1e-12,
        physics=FDTDCheckpointState(step=step, schema=schema, tensors={}),
        signature=resume._observer_origin_signature(SimpleNamespace()), auxiliary={"shutoff_triggered": shutoff},
    )


def test_outer_restore_does_not_mutate_physics_when_observation_preflight_rejects(monkeypatch):
    solver = _metadata()
    payload = _payload(solver, -1)
    mutations = []
    monkeypatch.setattr(resume, "_solver_signature", lambda solver: resume._observer_origin_signature(solver))
    def auxiliary(state, checkpoint):
        copies, updates = [], []
        observation_resume.preflight_observations(state, payload, checkpoint.step, copies, updates)
        return copies, updates
    monkeypatch.setattr(resume, "_preflight_auxiliary", auxiliary)
    monkeypatch.setattr(resume, "restore_checkpoint_state", lambda *args: mutations.append("physics"))
    with pytest.raises(ValueError, match="counter disagrees"):
        resume.restore_resume_checkpoint(solver, _checkpoint(), total_steps=8)
    assert not mutations and int(solver._dft_step.item()) == 0


def test_existing_post_shutoff_rejection_happens_before_observation_restore(monkeypatch):
    solver = _metadata()
    calls = []
    monkeypatch.setattr(resume, "_solver_signature", lambda solver: resume._observer_origin_signature(solver))
    monkeypatch.setattr(resume, "_preflight_auxiliary", lambda *args: calls.append("auxiliary"))
    monkeypatch.setattr(resume, "restore_checkpoint_state", lambda *args: calls.append("physics"))
    with pytest.raises(ValueError, match="after field shutoff"):
        resume.restore_resume_checkpoint(solver, _checkpoint(shutoff=True), total_steps=8)
    assert not calls and int(solver._dft_step.item()) == 0
