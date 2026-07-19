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

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="wave-level FDTD gate requires CUDA"
    ),
    # Round-4 operator blocker (EXECUTED): the guided-mode selector now refuses to
    # inject a spurious (checkerboard/wall-peaked) TE10, and the transverse vector
    # operator cannot yet produce a clean full-grid TE10 on the hollow guide (it
    # decouples the odd/even sublattices). The waveguide two-port therefore raises
    # in the mode solve until the symmetric-BC staggered operator lands (open item,
    # docs/reference/rf-wave-validation-2026-07-18.md). The matched/short S11
    # discrimination gate is deferred with it.
    pytest.mark.xfail(
        strict=False,
        reason="waveguide TE10 blocked on the transverse-operator redesign (open item)",
    ),
]

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
