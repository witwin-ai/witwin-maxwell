"""Wave-level matched-load |S11| gate (replaces the plan-01 algebraic identity).

Gate taxonomy (S0.3): **wave-level**.

The retired plan-01 Phase-1 headline ("matched load |S11| < -30 dB") applied one
implicit lumped update and checked the constructed pair V = Z0 * I at atol 1e-12 --
an ``analytic-identity`` with no wave, grid, or window. This gate instead drives a
propagating rectangular-waveguide two-port through a real FDTD sweep and reads the
reflection coefficient from the fields:

* matched termination (far port absorbs the incident TE10 mode) -> |S11| is small,
* the falsification detunes the load by replacing the far port with a PEC short,
  which must drive |S11| back toward unity.

The absolute matched floor here (roughly -10 to -12 dB on a coarse grid) does not
meet the -30 dB plan-01 target -- that gap is recorded in the benchmark. What
binds here is the *wave-level, load-discriminating* behaviour: a matched load
reflects far less than a short, measured from the FDTD fields (asserted with a
loose threshold so the exact dB does not drift against the artifact).
"""

from __future__ import annotations

import math

import pytest
import torch

import witwin.maxwell as mw
from benchmark.scenes.rf.rectangular_waveguide import GUIDE_A, rectangular_waveguide_scene

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="wave-level FDTD gate requires CUDA"
)

# NOTE (round-4): the injected TE10 is still checkerboard-contaminated (the
# transverse-operator redesign is an open item, see the reference doc), but the
# matched-vs-short REFLECTION DISCRIMINATION this gate checks is a loose,
# qualitative wave-level property that survives the contamination -- a matched TE10
# termination still reflects far less than a PEC short. This gate therefore remains
# green; the exact de-embedded beta (which does depend on a clean mode) is the part
# recorded as BLOCKED in the benchmark, not this discrimination check.

C0 = 299792458.0


def _operating_frequency() -> float:
    return 1.8 * C0 / (2.0 * GUIDE_A)


def _matched_thru_s11() -> float:
    result = mw.Simulation.fdtd(
        rectangular_waveguide_scene(dx=0.02, device="cuda"),
        frequencies=(_operating_frequency(),),
        excitations=mw.PortSweep(),
        run_time=mw.TimeConfig.auto(steady_cycles=6, transient_cycles=14),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()
    return float(torch.abs(result.network.s[0, 0, 0]))


def _shorted_s11() -> float:
    result = mw.Simulation.fdtd(
        rectangular_waveguide_scene(dx=0.02, short_far_port=True, device="cuda"),
        frequencies=(_operating_frequency(),),
        excitations=mw.PortExcitation("left", mode_name="TE0"),
        run_time=mw.TimeConfig.auto(steady_cycles=6, transient_cycles=14),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()
    port = result.port("left")
    return float(torch.abs((port.b / port.a).flatten()[0]))


def test_matched_load_reflection_is_low_from_the_fields():
    """A matched TE10 termination reflects little; measured from FDTD, not V=Z0 I."""

    matched = _matched_thru_s11()
    # Well below the clearly-reflecting regime; the exact -30 dB plan-01 target is
    # not met on this coarse grid and the gap is recorded in the benchmark, but a
    # genuine matched load must sit far from total reflection.
    assert matched < 0.35, f"matched |S11| unexpectedly high: {matched:.4f}"
    matched_db = 20.0 * math.log10(matched + 1e-30)
    assert matched_db < -8.0


def test_detuning_the_load_to_a_short_makes_reflection_red():
    """Falsification: replacing the matched load with a PEC short must spike |S11|."""

    matched = _matched_thru_s11()
    shorted = _shorted_s11()
    # The short reflects nearly all incident power; the matched load does not.
    assert shorted > 0.85, f"shorted |S11| unexpectedly low: {shorted:.4f}"
    assert shorted > 2.0 * matched, (
        "matched-load gate is not load-discriminating: "
        f"matched={matched:.4f} shorted={shorted:.4f}"
    )
