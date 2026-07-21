"""Opt-in per-rank step-rate instrumentation for distributed FDTD workers.

This module closes the "phase timings require an external profiler" finding for
the one-process-per-GPU NCCL shape without changing the production solve loop:
the instrument is *off by default* and is a no-op with zero extra per-step work
when disabled. When enabled through the environment it wraps a per-rank time loop
and, at the end, writes a machine-readable per-rank JSON summary the supervisor's
exclusive timing window can aggregate later.

Design contract (asserted by the unit test):

* **Zero cost when off.** With the instrument disabled, :meth:`step_begin` and
  :meth:`step_end` return immediately -- no device synchronize, no ``perf_counter``
  read, no bookkeeping. The unit test injects a counting ``sync`` callable and
  asserts it is never called across a full disabled loop, so a regression that
  synchronizes unconditionally is caught.
* **Meaningful timing when on.** A device synchronize brackets each timed step so
  the recorded wall interval reflects completed GPU work rather than kernel launch
  latency; this synchronize exists *only* on the enabled path.
* **Per-rank JSON.** :meth:`finalize` emits one ``step_timing_rank{rank}.json`` per
  rank with the step statistics and enough metadata (rank, world size, device,
  step count) to aggregate across ranks. No wall-clock number is asserted by any
  test -- only the schema and the zero-cost-off invariant are gated here.

The instrument never participates in the numerical result; it only observes.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import torch

# Truthy tokens that enable the instrument via ``WITWIN_FDTD_STEP_TIMING``.
_TRUTHY = frozenset({"1", "true", "yes", "on"})
_ENV_ENABLE = "WITWIN_FDTD_STEP_TIMING"
_ENV_OUTPUT_DIR = "WITWIN_FDTD_STEP_TIMING_DIR"


def _env_enabled(env) -> bool:
    return str(env.get(_ENV_ENABLE, "")).strip().lower() in _TRUTHY


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile of an already-sorted, non-empty sequence."""

    if not sorted_values:
        raise ValueError("percentile of an empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = fraction * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    weight = rank - lo
    return float(sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight)


@dataclass
class StepRateInstrument:
    """Env-gated per-rank step-rate recorder with a zero-cost disabled path.

    Construct one per rank. Bracket each time-loop iteration with
    :meth:`step_begin` / :meth:`step_end`, then call :meth:`finalize` once after
    the loop. When disabled the bracket calls do nothing and never synchronize.

    ``sync`` is injectable so the unit test can count synchronizations without a
    GPU; production code leaves it at the default ``torch.cuda.synchronize``.
    """

    rank: int
    world_size: int
    device: str
    enabled: bool = False
    output_dir: Path | None = None
    sync: Callable[[str], None] = field(default=torch.cuda.synchronize)
    _durations_s: list[float] = field(default_factory=list, init=False, repr=False)
    _pending_start: float | None = field(default=None, init=False, repr=False)
    _loop_start: float | None = field(default=None, init=False, repr=False)
    _loop_wall_s: float | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_env(
        cls,
        *,
        rank: int,
        world_size: int,
        device: str,
        env=None,
        sync: Callable[[str], None] | None = None,
    ) -> "StepRateInstrument":
        """Build an instrument from the environment (disabled unless opted in)."""

        source = os.environ if env is None else env
        enabled = _env_enabled(source)
        output_dir = source.get(_ENV_OUTPUT_DIR) if enabled else None
        return cls(
            rank=int(rank),
            world_size=int(world_size),
            device=str(device),
            enabled=bool(enabled),
            output_dir=Path(output_dir) if output_dir else None,
            sync=sync if sync is not None else torch.cuda.synchronize,
        )

    # -- time-loop bracket -------------------------------------------------

    def loop_begin(self) -> None:
        """Mark the start of the timed loop (no synchronize when disabled)."""

        if not self.enabled:
            return
        self.sync(self.device)
        self._loop_start = time.perf_counter()

    def step_begin(self) -> None:
        if not self.enabled:
            return
        self.sync(self.device)
        self._pending_start = time.perf_counter()

    def step_end(self) -> None:
        if not self.enabled:
            return
        self.sync(self.device)
        end = time.perf_counter()
        if self._pending_start is None:
            raise RuntimeError("StepRateInstrument.step_end called without step_begin.")
        self._durations_s.append(end - self._pending_start)
        self._pending_start = None

    def loop_end(self) -> None:
        if not self.enabled:
            return
        self.sync(self.device)
        if self._loop_start is None:
            raise RuntimeError("StepRateInstrument.loop_end called without loop_begin.")
        self._loop_wall_s = time.perf_counter() - self._loop_start

    # -- summary -----------------------------------------------------------

    def summary(self) -> dict:
        """Return the machine-readable per-rank summary dict.

        Always safe to call. When disabled it reports ``enabled: False`` and no
        step statistics; when enabled it reports per-step wall statistics in
        milliseconds plus the aggregate steps-per-second.
        """

        base = {
            "schema": "witwin.fdtd.step_timing/1",
            "rank": self.rank,
            "world_size": self.world_size,
            "device": self.device,
            "enabled": bool(self.enabled),
            "steps": len(self._durations_s),
        }
        if not self.enabled or not self._durations_s:
            return base
        ordered = sorted(self._durations_s)
        total_s = float(sum(self._durations_s))
        count = len(ordered)
        base.update(
            {
                "loop_wall_s": self._loop_wall_s,
                "step_total_s": total_s,
                "step_ms_mean": (total_s / count) * 1.0e3,
                "step_ms_median": _percentile(ordered, 0.5) * 1.0e3,
                "step_ms_min": ordered[0] * 1.0e3,
                "step_ms_max": ordered[-1] * 1.0e3,
                "step_ms_p95": _percentile(ordered, 0.95) * 1.0e3,
                "steps_per_second": (count / total_s) if total_s > 0.0 else None,
            }
        )
        return base

    def finalize(self) -> Path | None:
        """Write the per-rank JSON summary and return its path (``None`` if off).

        Disabled instruments write nothing and return ``None``. The output
        directory defaults to the current working directory and is created if
        absent so a launcher need not pre-create it.
        """

        if not self.enabled:
            return None
        directory = self.output_dir if self.output_dir is not None else Path.cwd()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"step_timing_rank{self.rank}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.summary(), handle, indent=2, sort_keys=True)
        return path


__all__ = ["StepRateInstrument"]
