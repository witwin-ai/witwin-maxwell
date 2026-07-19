"""Op-count ceilings for the RF port per-step hot path (audit step S2).

Gate class: ``perf-opcount`` (non-numerical performance gate; see
``docs/reference/gate-classification.md`` §5). Not a physical exit-gate; it
locks schedule shape, not any physical quantity.

Deterministic host/device dispatch tallies only -- no wall-clock timing is
asserted here (the timed ``< 5% / < 2%`` targets run later in an exclusive
window via the variance-aware gate in ``tests/support/perf_variance_gate.py``).

These ceilings lock in three properties the audit's §9.4 cost model depends on:

* the passive field-observation port and the SeriesRLC-terminated port issue a
  bounded per-step schedule with no host<->device sync;
* that schedule is *frequency-count independent* -- scoring one DFT bin and 181
  DFT bins launch the identical number of kernels, which is the direct evidence
  that the per-frequency accumulation is vectorized and does not dominate (so the
  audit's optional per-frequency weight-table rewrite is not the bottleneck fix);
* the marginal cost of each additional passive port is constant (linear scaling),
  which is what the ``< 2% per extra passive port`` target constrains.

Falsification (recorded 2026-07-18):
* Reverting the SeriesRLC diagnostics-off fast path (``apply_lumped_runtime``
  always taking the bookkeeping branch, i.e. the eb9258b baseline) raises the
  SeriesRLC schedule from 25 launches / 0 allocs / 3 DtoD per step to
  62 / 12 / 16, tripping every SeriesRLC ceiling below.
* Reintroducing a per-frequency Python loop in ``accumulate_precomputed`` makes
  the 181-bin launch count exceed the 1-bin count, tripping
  ``test_passive_port_launches_are_frequency_count_independent``.
Both were confirmed red under a scratch edit and reverted to green.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))

import witwin.maxwell as mw  # noqa: E402
from witwin.maxwell.fdtd import ports as mw_ports  # noqa: E402

from profile_port_hot_path import (  # noqa: E402
    _build_solver,
    _frequency_grid,
    _inventory_per_step,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Port hot-path op-count ceilings require CUDA.",
)


def _inventory(port_count: int, frequency_count: int, *, series_rlc: bool) -> dict[str, float]:
    frequencies = _frequency_grid(frequency_count)
    solver = _build_solver(
        mw, port_count=port_count, frequencies=frequencies, series_rlc=series_rlc
    )
    return _inventory_per_step(mw_ports, solver)


def test_passive_field_port_step_stays_within_op_count_ceiling():
    # Measured base schedule (HEAD, 2026-07-18): 30 launches, 5 allocs, 0 DtoD,
    # 0 host<->device transfers, 0 scalar syncs per step for one passive
    # field-observation port. Ceilings carry headroom so a genuine reduction
    # passes while a reintroduced per-step sync or allocation storm turns red.
    inv = _inventory(1, 181, series_rlc=False)
    assert inv["scalar_sync"] == 0
    assert inv["memcpy_hostside"] == 0
    assert inv["launches"] <= 40
    assert inv["allocs"] <= 8
    assert inv["memcpy_dtod"] <= 4


def test_series_rlc_port_step_stays_within_op_count_ceiling():
    # Measured base schedule (HEAD, 2026-07-18): 25 launches, 0 allocs, 3 DtoD,
    # 0 scalar syncs per step. The eb9258b baseline (no diagnostics-off fast
    # path) was 62 / 12 / 16; these ceilings sit well below that so a regression
    # to the bookkeeping branch fails. ``allocs == 0`` is the direct assertion
    # that the allocation-free fast path is active.
    inv = _inventory(1, 181, series_rlc=True)
    assert inv["scalar_sync"] == 0
    assert inv["memcpy_hostside"] == 0
    assert inv["allocs"] == 0
    assert inv["launches"] <= 32
    assert inv["memcpy_dtod"] <= 6


def test_passive_port_launches_are_frequency_count_independent():
    # The per-frequency DFT accumulation is a vectorized [F] update, so the
    # kernel-launch count must not grow with the number of scored frequencies.
    # This is the standing guard against a per-frequency Python loop and the
    # evidence that the per-frequency accumulation is not the launch bottleneck.
    single_bin = _inventory(1, 1, series_rlc=False)
    full_sweep = _inventory(1, 181, series_rlc=False)
    assert full_sweep["launches"] == single_bin["launches"]
    assert full_sweep["allocs"] == single_bin["allocs"]
    assert full_sweep["memcpy_dtod"] == single_bin["memcpy_dtod"]

    series_single = _inventory(1, 1, series_rlc=True)
    series_full = _inventory(1, 181, series_rlc=True)
    assert series_full["launches"] == series_single["launches"]
    assert series_full["allocs"] == series_single["allocs"]


def test_marginal_cost_per_additional_passive_port_is_constant():
    # Linear scaling: the launch/alloc delta from 1->2 ports must equal the
    # per-port delta from 2->4 ports. A super-linear regression (e.g. an
    # accidental all-pairs interaction between ports) breaks this equality.
    one = _inventory(1, 181, series_rlc=False)
    two = _inventory(2, 181, series_rlc=False)
    four = _inventory(4, 181, series_rlc=False)

    marginal_low = two["launches"] - one["launches"]
    marginal_high = (four["launches"] - two["launches"]) / 2.0
    assert marginal_low == pytest.approx(marginal_high)
    # Each added passive port contributes at most one field-observation port's
    # worth of launches; ceiling carries headroom over the measured 30.
    assert marginal_low <= 40
    assert (four["allocs"] - two["allocs"]) / 2.0 == pytest.approx(
        two["allocs"] - one["allocs"]
    )
