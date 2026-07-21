"""Unit tests for the opt-in distributed step-rate instrument.

These are pure host tests -- no GPU or NCCL launch -- so they run everywhere and
pin the two load-bearing contracts of ``StepRateInstrument``:

* **zero cost when off**: a disabled instrument never synchronizes and writes no
  artifact, so wrapping the production solve loop with it costs nothing when the
  timing env var is unset (the default);
* **machine-readable per-rank JSON when on**: an enabled instrument synchronizes
  the expected number of times, produces a schema-stable summary, and writes one
  ``step_timing_rank{rank}.json`` the exclusive-window aggregation can consume.

No wall-clock number is asserted; only structure and the zero-cost invariant.
"""

from __future__ import annotations

import json

import pytest

from witwin.maxwell.fdtd.distributed.instrumentation import StepRateInstrument


class _SyncCounter:
    """A drop-in for ``torch.cuda.synchronize`` that counts invocations."""

    def __init__(self):
        self.calls = 0
        self.devices = []

    def __call__(self, device=None):
        self.calls += 1
        self.devices.append(device)


def _run_loop(instrument, steps):
    instrument.loop_begin()
    for _ in range(steps):
        instrument.step_begin()
        # stand-in for _advance_one_step
        instrument.step_end()
    instrument.loop_end()


def test_disabled_by_default_from_env(tmp_path):
    instrument = StepRateInstrument.from_env(
        rank=0, world_size=2, device="cuda:0", env={}
    )
    assert instrument.enabled is False


def test_disabled_never_synchronizes_and_writes_nothing(tmp_path):
    counter = _SyncCounter()
    instrument = StepRateInstrument.from_env(
        rank=1,
        world_size=2,
        device="cuda:1",
        env={"WITWIN_FDTD_STEP_TIMING_DIR": str(tmp_path)},  # dir set, but NOT enabled
        sync=counter,
    )
    assert instrument.enabled is False
    _run_loop(instrument, steps=25)
    # Zero-cost-off: the disabled bracket calls must not synchronize at all.
    assert counter.calls == 0, f"disabled instrument synchronized {counter.calls} times"
    assert instrument.finalize() is None
    # No artifact is written when disabled.
    assert list(tmp_path.glob("*.json")) == []
    summary = instrument.summary()
    assert summary["enabled"] is False
    assert summary["steps"] == 0
    assert "steps_per_second" not in summary


def test_enabled_synchronizes_and_emits_per_rank_json(tmp_path):
    counter = _SyncCounter()
    steps = 12
    instrument = StepRateInstrument.from_env(
        rank=0,
        world_size=2,
        device="cuda:0",
        env={
            "WITWIN_FDTD_STEP_TIMING": "1",
            "WITWIN_FDTD_STEP_TIMING_DIR": str(tmp_path),
        },
        sync=counter,
    )
    assert instrument.enabled is True
    _run_loop(instrument, steps=steps)
    # loop_begin + (step_begin + step_end)*steps + loop_end
    assert counter.calls == 2 + 2 * steps
    assert all(device == "cuda:0" for device in counter.devices)

    path = instrument.finalize()
    assert path is not None and path.name == "step_timing_rank0.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "witwin.fdtd.step_timing/1"
    assert payload["rank"] == 0
    assert payload["world_size"] == 2
    assert payload["device"] == "cuda:0"
    assert payload["enabled"] is True
    assert payload["steps"] == steps
    for key in (
        "step_ms_mean",
        "step_ms_median",
        "step_ms_min",
        "step_ms_max",
        "step_ms_p95",
        "steps_per_second",
    ):
        assert key in payload, key
    assert payload["step_ms_min"] <= payload["step_ms_median"] <= payload["step_ms_max"]
    assert payload["steps_per_second"] is None or payload["steps_per_second"] > 0.0


def test_each_rank_writes_a_distinct_file(tmp_path):
    counter = _SyncCounter()
    env = {"WITWIN_FDTD_STEP_TIMING": "on", "WITWIN_FDTD_STEP_TIMING_DIR": str(tmp_path)}
    for rank in range(3):
        instrument = StepRateInstrument.from_env(
            rank=rank, world_size=3, device=f"cuda:{rank}", env=env, sync=counter
        )
        _run_loop(instrument, steps=4)
        instrument.finalize()
    written = sorted(p.name for p in tmp_path.glob("*.json"))
    assert written == [
        "step_timing_rank0.json",
        "step_timing_rank1.json",
        "step_timing_rank2.json",
    ]


def test_step_end_without_begin_raises(tmp_path):
    counter = _SyncCounter()
    instrument = StepRateInstrument.from_env(
        rank=0,
        world_size=2,
        device="cuda:0",
        env={"WITWIN_FDTD_STEP_TIMING": "1", "WITWIN_FDTD_STEP_TIMING_DIR": str(tmp_path)},
        sync=counter,
    )
    with pytest.raises(RuntimeError, match="without step_begin"):
        instrument.step_end()
