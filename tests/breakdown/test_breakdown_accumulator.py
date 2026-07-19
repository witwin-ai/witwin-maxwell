"""State-free golden tests for the non-feedback dielectric-stress accumulator.

Capability level under test: stress-only. These tests drive the accumulator
update rule directly with prescribed E(t), so the exceedance intervals, longest
contiguous duration, damage integral, occupancy weighting, and Yee colocation
are all exactly known analytically.
"""

from __future__ import annotations

import math

import pytest
import torch

from witwin.maxwell.breakdown_stress import (
    BREAKDOWN_CAPABILITY_LEVEL,
    BreakdownStressAccumulator,
    colocate_electric_magnitude,
)


def _run(sequence, *, critical_field, dt=1.0, minimum_duration=0.0, damage_exponent=None, shape=(1,)):
    acc = BreakdownStressAccumulator.allocate(
        shape=shape,
        critical_field=critical_field,
        dt=dt,
        minimum_duration=minimum_duration,
        damage_exponent=damage_exponent,
    )
    for value in sequence:
        acc.update(torch.as_tensor(value, dtype=torch.float32).reshape(shape))
    return acc


# ---------------------------------------------------------------------------
# Exceedance intervals, peak, longest contiguous duration.
# ---------------------------------------------------------------------------


def test_two_pulse_exact_exceedance_and_longest_run():
    # Ecrit = 10; exceeding samples at indices 1,2 (run of 2) and 4,5,6 (run of 3).
    seq = [5.0, 12.0, 15.0, 3.0, 20.0, 11.0, 14.0, 2.0]
    data = _run(seq, critical_field=10.0, dt=0.5).finalize(name="two_pulse")
    assert data.peak_field == pytest.approx(20.0)
    assert data.exceedance_duration == pytest.approx(5 * 0.5)  # 5 exceeding samples
    assert data.longest_exceedance_duration == pytest.approx(3 * 0.5)
    assert data.capability_level == BREAKDOWN_CAPABILITY_LEVEL


def test_longest_run_resets_between_pulses():
    # Two separate runs of length 2; longest must be 2*dt, not the total 4 samples.
    seq = [11.0, 11.0, 1.0, 11.0, 11.0]
    data = _run(seq, critical_field=10.0, dt=1.0).finalize()
    assert data.exceedance_duration == pytest.approx(4.0)
    assert data.longest_exceedance_duration == pytest.approx(2.0)


def test_falsify_longest_run_without_reset_would_overcount():
    # Documents the falsification: if the contiguous-run bookkeeping never reset,
    # the longest run would equal the total exceedance time (4.0) instead of 2.0.
    seq = [11.0, 11.0, 1.0, 11.0, 11.0]
    data = _run(seq, critical_field=10.0, dt=1.0).finalize()
    assert data.longest_exceedance_duration != data.exceedance_duration


def test_exactly_at_threshold_counts_as_exceedance():
    # Heaviside convention H(0) = 1: |E| == Ecrit is an exceedance.
    data = _run([10.0], critical_field=10.0, dt=1.0).finalize()
    assert data.exceedance_duration == pytest.approx(1.0)
    assert data.longest_exceedance_duration == pytest.approx(1.0)


def test_just_below_threshold_does_not_count():
    data = _run([9.999], critical_field=10.0, dt=1.0).finalize()
    assert data.exceedance_duration == pytest.approx(0.0)
    assert data.peak_field == pytest.approx(9.999, rel=1e-4)


# ---------------------------------------------------------------------------
# Damage integral.
# ---------------------------------------------------------------------------


def test_damage_integral_accrues_only_during_exceedance():
    seq = [12.0, 15.0, 5.0, 20.0]  # exceeding at 12,15,20 (5 is below Ecrit=10)
    data = _run(seq, critical_field=10.0, dt=2.0, damage_exponent=2.0).finalize()
    expected = 2.0 * ((1.2 ** 2) + (1.5 ** 2) + (2.0 ** 2))
    assert data.damage_volume == pytest.approx(expected, rel=1e-5)


def test_damage_disabled_without_exponent():
    data = _run([12.0], critical_field=10.0).finalize()
    assert data.damage_map is None
    assert data.damage_volume is None


# ---------------------------------------------------------------------------
# minimum_duration qualifying mask / locations.
# ---------------------------------------------------------------------------


def test_minimum_duration_gates_qualifying_locations():
    # Cell 0 sustains a 3-sample run; cell 1 only a 1-sample run.
    acc = BreakdownStressAccumulator.allocate(
        shape=(2,), critical_field=10.0, dt=1.0, minimum_duration=3.0
    )
    frames = [
        [11.0, 11.0],
        [11.0, 1.0],
        [11.0, 1.0],
    ]
    for frame in frames:
        acc.update(torch.tensor(frame, dtype=torch.float32))
    data = acc.finalize()
    assert data.qualifying_cell_count == 1
    locations = data.locations()
    assert locations["count"] == 1
    assert int(locations["indices"][0, 0]) == 0


# ---------------------------------------------------------------------------
# Occupancy-weighted partial voxels.
# ---------------------------------------------------------------------------


def test_zero_occupancy_cell_excluded_from_peak_and_exceedance():
    occ = torch.tensor([1.0, 0.0], dtype=torch.float32)
    acc = BreakdownStressAccumulator.allocate(
        shape=(2,), critical_field=10.0, dt=1.0, occupancy=occ
    )
    acc.update(torch.tensor([12.0, 999.0], dtype=torch.float32))
    data = acc.finalize()
    # The huge field in the unoccupied cell must not enter the peak.
    assert data.peak_field == pytest.approx(12.0)
    assert float(data.max_field_map[1]) == 0.0
    assert float(data.exceedance_time_map[1]) == 0.0


def test_partial_occupancy_weights_region_volume_time():
    occ = torch.tensor([1.0, 0.5], dtype=torch.float32)
    vol = torch.tensor([2.0, 4.0], dtype=torch.float32)
    acc = BreakdownStressAccumulator.allocate(
        shape=(2,), critical_field=10.0, dt=1.0, occupancy=occ, cell_volume=vol
    )
    for _ in range(3):  # 3 exceeding steps in both cells
        acc.update(torch.tensor([12.0, 12.0], dtype=torch.float32))
    data = acc.finalize()
    # exceedance_volume_time = sum(exceedance_time * occupancy * cell_volume)
    # cell0: 3 * 1.0 * 2.0 = 6 ; cell1: 3 * 0.5 * 4.0 = 6
    assert data.exceedance_volume_time == pytest.approx(12.0)


def test_falsify_occupancy_ignored_would_change_weighted_metric():
    occ = torch.tensor([0.5], dtype=torch.float32)
    vol = torch.tensor([1.0], dtype=torch.float32)
    acc = BreakdownStressAccumulator.allocate(
        shape=(1,), critical_field=10.0, dt=1.0, occupancy=occ, cell_volume=vol
    )
    acc.update(torch.tensor([12.0], dtype=torch.float32))
    data = acc.finalize()
    # With occupancy applied the metric is 0.5; ignoring occupancy would give 1.0.
    assert data.exceedance_volume_time == pytest.approx(0.5)
    assert data.exceedance_volume_time != pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Yee colocation consistency vs energy density.
# ---------------------------------------------------------------------------


def _staggered_uniform(shape_cells, ex, ey, ez):
    ncx, ncy, ncz = shape_cells
    Ex = torch.full((ncx, ncy + 1, ncz + 1), float(ex))
    Ey = torch.full((ncx + 1, ncy, ncz + 1), float(ey))
    Ez = torch.full((ncx + 1, ncy + 1, ncz), float(ez))
    return Ex, Ey, Ez


def test_colocation_uniform_field_matches_analytic_magnitude():
    Ex, Ey, Ez = _staggered_uniform((3, 4, 5), 3.0, 4.0, 12.0)
    mag = colocate_electric_magnitude(Ex, Ey, Ez)
    assert tuple(mag.shape) == (3, 4, 5)
    expected = math.sqrt(3.0 ** 2 + 4.0 ** 2 + 12.0 ** 2)
    assert torch.allclose(mag, torch.full_like(mag, expected), atol=1e-5)


def test_colocation_energy_density_consistency_on_uniform_field():
    # u_E = 0.5 * eps * |E|^2 with the same colocation reproduces the analytic
    # energy density of the uniform field exactly.
    eps = 8.854e-12
    # All three components nonzero so a dropped term in the colocation would
    # break this test, not just the companion analytic-magnitude test.
    ex, ey, ez = 3.0, 4.0, 12.0
    Ex, Ey, Ez = _staggered_uniform((2, 2, 2), ex, ey, ez)
    mag = colocate_electric_magnitude(Ex, Ey, Ez)
    energy_density = 0.5 * eps * mag.square()
    analytic = 0.5 * eps * (ex ** 2 + ey ** 2 + ez ** 2)
    # atol=0.0 keeps the comparison purely relative: with the eps-scaled energy
    # densities (~1e-10) the default absolute tolerance (1e-8) would otherwise
    # swamp the check and pass even if a colocation term were dropped.
    assert torch.allclose(
        energy_density, torch.full_like(energy_density, analytic), rtol=1e-6, atol=0.0
    )


def test_colocation_requires_consistent_staggered_shapes():
    with pytest.raises(ValueError):
        colocate_electric_magnitude(
            torch.zeros((3, 3, 3)), torch.zeros((3, 3, 3)), torch.zeros((3, 3, 3))
        )


# ---------------------------------------------------------------------------
# Construction guards.
# ---------------------------------------------------------------------------


def test_allocate_rejects_bad_inputs():
    with pytest.raises(ValueError):
        BreakdownStressAccumulator.allocate(shape=(1,), critical_field=0.0, dt=1.0)
    with pytest.raises(ValueError):
        BreakdownStressAccumulator.allocate(shape=(1,), critical_field=1.0, dt=0.0)
    with pytest.raises(ValueError):
        BreakdownStressAccumulator.allocate(shape=(1,), critical_field=1.0, dt=1.0, damage_exponent=-1.0)
    with pytest.raises(ValueError):
        BreakdownStressAccumulator.allocate(
            shape=(1,), critical_field=1.0, dt=1.0, occupancy=torch.tensor([2.0])
        )


def test_update_shape_mismatch_raises():
    acc = BreakdownStressAccumulator.allocate(shape=(2,), critical_field=1.0, dt=1.0)
    with pytest.raises(ValueError):
        acc.update(torch.zeros((3,)))
