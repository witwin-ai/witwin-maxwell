"""Integration-level event log: triggering run, determinism, no-trigger run."""

from __future__ import annotations

import pytest
import torch

from witwin.maxwell.breakdown import (
    BREAKDOWN_STATE_CONDUCTING,
    BREAKDOWN_STATE_INTACT,
    BreakdownEvent,
)

from tests.breakdown._common import build_breakdown_scene, run_breakdown
import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FDTD breakdown requires CUDA"
)


def _triggering_scene():
    # Low threshold and a strong nearby source guarantee triggers once the wave
    # reaches the box.
    return build_breakdown_scene(
        critical_field=1.0e-2,
        minimum_duration=0.0,
        post_sigma=5.0,
        source_amplitude=100.0,
        source_position=(-0.12, 0.0, 0.0),
        box_size=(0.16, 0.16, 0.16),
    )


def test_triggering_run_populates_event_log():
    result = run_breakdown(_triggering_scene(), time_steps=90)
    data = mw.breakdown.breakdown_data(result)
    assert data is not None
    assert data.triggered_count > 0
    assert len(mw.breakdown.breakdown_events(result)) == data.triggered_count
    assert data.total_dissipated_energy > 0.0

    # Every event is a typed record with a conducting transition and a global id.
    for event in mw.breakdown.breakdown_events(result):
        assert isinstance(event, BreakdownEvent)
        assert event.state_before == BREAKDOWN_STATE_INTACT
        assert event.state_after == BREAKDOWN_STATE_CONDUCTING
        assert event.step >= 0
        assert event.cell_index >= 0
        assert event.field_before >= event.field_before  # finite (not NaN)

    # Events are ordered by (step, cell_index).
    keys = [(e.step, e.cell_index) for e in mw.breakdown.breakdown_events(result)]
    assert keys == sorted(keys)

    # Final-state mask marks the triggered cells conducting, and only them.
    final_state = data.final_state
    conducting = int((final_state == BREAKDOWN_STATE_CONDUCTING).sum().item())
    assert conducting == data.triggered_count


def test_event_log_is_deterministic():
    """Two identical runs produce identical logs, masks and dissipation."""
    a = run_breakdown(_triggering_scene(), time_steps=90)
    b = run_breakdown(_triggering_scene(), time_steps=90)

    ea, eb = mw.breakdown.breakdown_events(a), mw.breakdown.breakdown_events(b)
    assert len(ea) == len(eb) and len(ea) > 0
    for x, y in zip(ea, eb):
        assert x.step == y.step
        assert x.cell_index == y.cell_index
        assert x.material_id == y.material_id
        assert x.field_before == y.field_before
        assert x.deposited_energy_at_trigger == y.deposited_energy_at_trigger

    assert torch.equal(mw.breakdown.breakdown_data(a).final_state, mw.breakdown.breakdown_data(b).final_state)
    assert torch.equal(mw.breakdown.breakdown_data(a).dissipated_energy, mw.breakdown.breakdown_data(b).dissipated_energy)
    assert mw.breakdown.breakdown_data(a).total_dissipated_energy == mw.breakdown.breakdown_data(b).total_dissipated_energy


def test_no_trigger_run_has_empty_log_and_intact_state():
    """A high threshold yields a breakdown result with zero events, all intact."""
    scene = build_breakdown_scene(
        critical_field=1.0e12,
        minimum_duration=0.0,
        source_amplitude=100.0,
    )
    result = run_breakdown(scene, time_steps=60)
    data = mw.breakdown.breakdown_data(result)
    assert data is not None  # the material is present
    assert data.triggered_count == 0
    assert mw.breakdown.breakdown_events(result) == ()
    assert torch.all(data.final_state == BREAKDOWN_STATE_INTACT)
    assert data.total_dissipated_energy == 0.0
