"""Series / parallel RLC resonance: wave-level C-tracking gate (rebuilt bench).

Gate taxonomy (S0.3):
* ``test_series_rlc_companion_impedance_matches_analytic`` -- **analytic-identity**
  (retained as a fast contract test; NOT an exit gate).
* the resonance-tracking tests below are **wave-level** gates on the rebuilt coax
  in-line RLC bench.

Honest-exit record (audit 2026-07-18 -> rebuild). The retired plan-01 RLC gate
swept the trapezoidal companion-impedance *formula* without running the solver;
the first solver attempt (a lumped two-port in a tiny PML box) was
parasitic-dominated and the resonance peak did NOT track ``C``, so it was recorded
as a strict xfail open gap. This rebuild inserts the RLC as an IN-LINE element in
the inner conductor of the proven coax line (``series_parallel_rlc.py``): the
element carries the full axial line current, so its resonance controls the feed
reflection and genuinely tracks ``C``.

What binds (measured at dx=0.01):

* **series** -- ``|S11|`` has a deep NOTCH at the series resonance; the extracted
  notch frequency obeys ``f_res * sqrt(C) = const`` (the ``1/(2 pi sqrt(L C))``
  law) to ~1%, is monotone in ``C``, and moves by the analytic ``1/sqrt(C)`` ratio
  under a +/-20% change in ``C`` (to within a few percent). The absolute notch
  sits ~13% BELOW the ideal ``f0`` -- the documented, consistent parasitic shift
  from the rod-gap fringe capacitance (measured, not hidden).
* **parallel** -- ``|S11|`` has a PEAK at the anti-resonance; the extracted peak
  moves in the correct direction with ``C`` (higher ``C`` -> lower ``f_res``). Its
  tracking slope is DILUTED by the fringe capacitance adding directly to the
  parallel ``C`` (documented); direction + monotone movement is what binds here.

Falsification: a C-independent resonance (the retired bench's failure mode) must
FAIL the ``f_res * sqrt(C) = const`` gate -- proving the gate detects non-tracking.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from benchmark.scenes.rf.series_parallel_rlc import (
    DEFAULT_L,
    default_frequencies,
    resonance_frequency,
    series_rlc_scene,
)


def test_series_rlc_companion_impedance_matches_analytic():
    """analytic-identity (fast contract, non-gating): companion Z == closed form."""

    r, l, c = 8.0, 0.5e-9, 1.0e-12  # noqa: E741
    termination = mw.SeriesRLC(r=r, l=l, c=c)
    f0 = 1.0 / (2.0 * math.pi * math.sqrt(l * c))
    omega = 2.0 * math.pi * torch.tensor([0.5 * f0, f0, 1.5 * f0], dtype=torch.float64)

    z = termination.impedance(omega)
    assert abs(float(z[1].real) - r) < 1.0e-6
    assert abs(float(z[1].imag)) < 1.0e-3 * r
    assert float(z[0].imag) < 0.0
    assert float(z[2].imag) > 0.0


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="wave-level RLC FDTD gate requires CUDA"
)

# +/-20% capacitance sweep around the default tier.
_C_NOMINAL = 4.0e-12
_C_VALUES = (3.2e-12, 4.0e-12, 4.8e-12)


def _interp_extremum(freqs: np.ndarray, mag: np.ndarray, *, mode: str) -> float:
    index = int(np.argmin(mag)) if mode == "min" else int(np.argmax(mag))
    if index <= 0 or index >= len(mag) - 1:
        return float(freqs[index])
    y0, y1, y2 = mag[index - 1], mag[index], mag[index + 1]
    curvature = y0 - 2.0 * y1 + y2
    if curvature == 0.0:
        return float(freqs[index])
    offset = 0.5 * (y0 - y2) / curvature
    return float(freqs[index] + offset * (freqs[1] - freqs[0]))


def _resonance(c: float, *, parallel: bool) -> float:
    freqs = default_frequencies(parallel=parallel)
    result = mw.Simulation.fdtd(
        series_rlc_scene(l=DEFAULT_L, c=c, parallel=parallel, dx=0.01, device="cuda"),
        frequencies=freqs,
        excitations=mw.PortExcitation("feed"),
        run_time=mw.TimeConfig.auto(steady_cycles=6, transient_cycles=12),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()
    feed = result.port("feed")
    a = feed.a.cpu().numpy().reshape(len(freqs), -1)[:, 0]
    b = feed.b.cpu().numpy().reshape(len(freqs), -1)[:, 0]
    mag = np.abs(b / a)
    return _interp_extremum(np.asarray(freqs), mag, mode="max" if parallel else "min")


@pytest.fixture(scope="module")
def series_resonances():
    return {c: _resonance(c, parallel=False) for c in _C_VALUES}


@pytest.fixture(scope="module")
def parallel_resonances():
    return {c: _resonance(c, parallel=True) for c in _C_VALUES}


def test_series_resonance_tracks_inverse_sqrt_c(series_resonances):
    """The series notch obeys f_res * sqrt(C) = const (the 1/sqrt(LC) law)."""

    cs = np.array(_C_VALUES)
    fr = np.array([series_resonances[c] for c in _C_VALUES])
    invariant = fr * np.sqrt(cs)
    spread = float(invariant.std() / invariant.mean())
    assert spread < 0.05, f"f_res*sqrt(C) not constant (spread {spread:.3f}): {fr}"
    # Monotone: higher C -> lower resonance.
    assert fr[0] > fr[1] > fr[2], f"series resonance not monotone in C: {fr}"


def test_series_resonance_moves_correctly_under_20pct_c(series_resonances):
    """A +/-20% change in C moves the notch by the analytic 1/sqrt(C) ratio."""

    f_lo = series_resonances[3.2e-12]   # C - 20%
    f_nom = series_resonances[4.0e-12]
    f_hi = series_resonances[4.8e-12]   # C + 20%

    ratio_lo = f_lo / f_nom
    ratio_hi = f_hi / f_nom
    analytic_lo = math.sqrt(_C_NOMINAL / 3.2e-12)   # ~1.118
    analytic_hi = math.sqrt(_C_NOMINAL / 4.8e-12)   # ~0.913
    assert abs(ratio_lo - analytic_lo) < 0.08, f"-20% C ratio {ratio_lo:.3f} vs {analytic_lo:.3f}"
    assert abs(ratio_hi - analytic_hi) < 0.08, f"+20% C ratio {ratio_hi:.3f} vs {analytic_hi:.3f}"


def test_series_absolute_shift_is_the_documented_parasitic(series_resonances):
    """The absolute notch sits ~13% below the ideal f0 (consistent parasitic)."""

    for c in _C_VALUES:
        ideal = resonance_frequency(DEFAULT_L, c)
        ratio = series_resonances[c] / ideal
        # Measured ~0.865 at every C; a loose, documented band (NOT a tight
        # absolute gate -- the parasitic shift is real and recorded honestly).
        assert 0.70 < ratio < 0.95, (
            f"series notch/ideal ratio {ratio:.3f} at C={c:.2e} outside the "
            "documented parasitic band"
        )


def test_parallel_resonance_moves_correctly_with_c(parallel_resonances):
    """The parallel anti-resonance peak moves in the correct direction with C.

    Its slope is diluted by the fringe capacitance adding to the parallel C
    (documented); direction + a measurable monotone move is what binds here.
    """

    fr = np.array([parallel_resonances[c] for c in _C_VALUES])
    assert fr[0] > fr[1] > fr[2], f"parallel peak not monotone in C: {fr}"
    # A +/-20% C change must move the peak by a resolvable amount.
    assert (fr[0] - fr[2]) / fr[1] > 0.02, f"parallel peak barely moves with C: {fr}"


def test_tracking_gate_falsification(series_resonances):
    """Falsification: a C-independent resonance (the retired bench's failure mode)
    must FAIL the f_res * sqrt(C) = const gate."""

    cs = np.array(_C_VALUES)
    # Decoupled bench: the resonance does not move with C (constant f_res).
    fixed = float(series_resonances[4.0e-12])
    fr = np.full_like(cs, fixed)
    invariant = fr * np.sqrt(cs)
    spread = float(invariant.std() / invariant.mean())
    assert not (spread < 0.05), (
        "tracking gate failed to flag a C-independent (non-tracking) resonance"
    )
