"""Analytic, resampling, and diagnostics coverage for ESD current waveforms.

Capability level under test: stress-only standard waveform reproduction. All
tolerances compare against independent numerical quadrature of the same analytic
current, not against nominal standard table values.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy import integrate

import witwin.maxwell as mw
from witwin.maxwell.esd import ESD_CAPABILITY_LEVEL


def _numpy_current(waveform, t):
    return waveform.current(torch.as_tensor(t, dtype=torch.float64)).detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Construction and provenance.
# ---------------------------------------------------------------------------


def test_factory_requires_explicit_construction():
    with pytest.raises(TypeError):
        mw.ESDWaveform(8000.0)


def test_factory_records_default_revision_in_provenance():
    waveform = mw.ESDWaveform.iec_61000_4_2(8000.0)
    assert waveform.standard_revision == "ed2-contact"
    assert waveform.discharge == "contact"
    provenance = waveform.provenance
    assert provenance["standard"] == "IEC 61000-4-2"
    assert provenance["standard_revision"] == "ed2-contact"
    assert provenance["level_voltage"] == 8000.0
    assert provenance["capability_level"] == ESD_CAPABILITY_LEVEL
    assert provenance["n"] == pytest.approx(1.8)
    assert len(provenance["terms"]) == 2


def test_unsupported_revision_and_discharge_fail_closed():
    with pytest.raises(ValueError):
        mw.ESDWaveform.iec_61000_4_2(8000.0, standard_revision="ed9-imaginary")
    with pytest.raises(ValueError):
        mw.ESDWaveform.iec_61000_4_2(8000.0, discharge="air")
    with pytest.raises(ValueError):
        mw.ESDWaveform.iec_61000_4_2(-1.0)


# ---------------------------------------------------------------------------
# Analytic diagnostics vs independent quadrature (tight rtol).
# ---------------------------------------------------------------------------


def test_charge_and_action_match_scipy_quadrature():
    waveform = mw.ESDWaveform.iec_61000_4_2(8000.0)
    diagnostics = waveform.diagnostics()
    upper = waveform.support[1]

    charge, _ = integrate.quad(lambda t: float(_numpy_current(waveform, t)), 0.0, upper, limit=400)
    action, _ = integrate.quad(lambda t: float(_numpy_current(waveform, t)) ** 2, 0.0, upper, limit=400)

    assert diagnostics.charge == pytest.approx(charge, rel=1e-4)
    assert diagnostics.action_integral == pytest.approx(action, rel=1e-4)


def test_peak_matches_independent_dense_argmax():
    waveform = mw.ESDWaveform.iec_61000_4_2(8000.0)
    diagnostics = waveform.diagnostics()

    grid = np.linspace(0.0, 6.0e-9, 2_000_001)
    dense = _numpy_current(waveform, grid)
    reference_peak = float(dense.max())
    reference_time = float(grid[int(dense.argmax())])

    assert diagnostics.peak_current == pytest.approx(reference_peak, rel=1e-4)
    assert diagnostics.peak_time == pytest.approx(reference_time, abs=5e-11)


def test_specified_time_samples_are_exact_analytic_values():
    waveform = mw.ESDWaveform.iec_61000_4_2(8000.0)
    diagnostics = waveform.diagnostics()
    assert diagnostics.current_at_30ns == pytest.approx(float(_numpy_current(waveform, 30.0e-9)), rel=1e-9)
    assert diagnostics.current_at_60ns == pytest.approx(float(_numpy_current(waveform, 60.0e-9)), rel=1e-9)


def test_level_voltage_scales_current_linearly():
    low = mw.ESDWaveform.iec_61000_4_2(4000.0).diagnostics()
    high = mw.ESDWaveform.iec_61000_4_2(8000.0).diagnostics()
    assert high.peak_current == pytest.approx(2.0 * low.peak_current, rel=1e-9)
    assert high.charge == pytest.approx(2.0 * low.charge, rel=1e-9)
    # Action integral is quadratic in amplitude.
    assert high.action_integral == pytest.approx(4.0 * low.action_integral, rel=1e-6)


def test_contact_diagnostics_sit_in_the_iec_first_transient_band():
    # Documented sanity band (NOT a tight gate): the widely published two-Heidler
    # contact set reproduces the IEC 61000-4-2 first-transient key points to
    # within model tolerance. Peak overshoots the 30 A nominal (~14%) because the
    # two Heidler terms superpose; i(30 ns) and i(60 ns) match closely.
    diagnostics = mw.ESDWaveform.iec_61000_4_2(8000.0).diagnostics()
    assert diagnostics.current_at_30ns == pytest.approx(16.0, rel=0.15)
    assert diagnostics.current_at_60ns == pytest.approx(8.0, rel=0.15)
    assert 0.7e-9 <= diagnostics.rise_time_10_90 <= 1.0e-9


# ---------------------------------------------------------------------------
# Rise time golden / independent reconstruction.
# ---------------------------------------------------------------------------


def test_rise_time_matches_independent_crossing_reconstruction():
    waveform = mw.ESDWaveform.iec_61000_4_2(8000.0)
    diagnostics = waveform.diagnostics()

    grid = np.linspace(0.0, 6.0e-9, 4_000_001)
    dense = _numpy_current(waveform, grid)
    peak_index = int(dense.argmax())
    peak = float(dense[peak_index])

    def crossing(level):
        target = level * peak
        rising = dense[: peak_index + 1]
        idx = int(np.argmax(rising >= target))
        t0, t1 = grid[idx - 1], grid[idx]
        y0, y1 = rising[idx - 1], rising[idx]
        return t0 + (target - y0) * (t1 - t0) / (y1 - y0)

    reference = crossing(0.9) - crossing(0.1)
    assert diagnostics.rise_time_10_90 == pytest.approx(reference, rel=2e-3)


# ---------------------------------------------------------------------------
# Charge-conserving resampling and convergence.
# ---------------------------------------------------------------------------


def test_resampling_conserves_charge_across_three_dt():
    waveform = mw.ESDWaveform.iec_61000_4_2(8000.0)
    for dt in (2.0e-9, 1.0e-9, 0.5e-9):
        resampled = waveform.resample_to_grid(dt, t_end=waveform.support[1])
        # Charge is conserved exactly (to quadrature precision) at every dt.
        assert resampled.charge_ratio == pytest.approx(1.0, rel=1e-6)


def test_resampled_action_converges_as_dt_refines():
    waveform = mw.ESDWaveform.iec_61000_4_2(8000.0)
    errors = [
        waveform.resample_to_grid(dt, t_end=waveform.support[1]).aliasing_metric
        for dt in (2.0e-9, 1.0e-9, 0.5e-9)
    ]
    # Monotone decrease of the aliasing metric toward zero (~O(dt^2)).
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 5e-3
    ratio = errors[0] / errors[1]
    assert ratio > 2.0  # second-order-ish refinement


def test_resample_rejects_invalid_grid():
    waveform = mw.ESDWaveform.iec_61000_4_2(8000.0)
    with pytest.raises(ValueError):
        waveform.resample_to_grid(0.0)
    with pytest.raises(ValueError):
        waveform.resample_to_grid(1e-9, t_end=-1.0)


# ---------------------------------------------------------------------------
# MeasuredWaveform.
# ---------------------------------------------------------------------------


def test_measured_waveform_reproduces_sampled_diagnostics():
    source = mw.ESDWaveform.iec_61000_4_2(8000.0)
    times = torch.linspace(0.0, source.support[1], 20001, dtype=torch.float64)
    values = source.current(times)
    measured = mw.MeasuredWaveform(times, values, units="A", bandwidth=1e9, provenance={"probe": "unit-test"})

    analytic = source.diagnostics()
    sampled = measured.diagnostics()
    assert sampled.charge == pytest.approx(analytic.charge, rel=1e-3)
    assert sampled.peak_current == pytest.approx(analytic.peak_current, rel=2e-3)
    assert measured.provenance["bandwidth"] == 1e9
    assert measured.provenance["probe"] == "unit-test"
    assert measured.provenance["capability_level"] == ESD_CAPABILITY_LEVEL


def test_measured_waveform_validates_inputs():
    with pytest.raises(ValueError):
        mw.MeasuredWaveform([0.0], [1.0])
    with pytest.raises(ValueError):
        mw.MeasuredWaveform([0.0, 0.0, 1e-9], [1.0, 2.0, 3.0])  # non-increasing
    with pytest.raises(ValueError):
        mw.MeasuredWaveform([0.0, 1e-9], [1.0, 2.0], units="V")


def test_measured_waveform_charge_conserving_resampling():
    source = mw.ESDWaveform.iec_61000_4_2(8000.0)
    times = torch.linspace(0.0, source.support[1], 40001, dtype=torch.float64)
    measured = mw.MeasuredWaveform(times, source.current(times))
    resampled = measured.resample_to_grid(1.0e-9, t_end=source.support[1])
    assert resampled.charge_ratio == pytest.approx(1.0, rel=1e-3)
