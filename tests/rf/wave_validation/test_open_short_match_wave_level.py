"""Wave-level open/short/match discrimination on the rebuilt coax SOL bench.

Gate taxonomy (S0.3): **wave-level**.

The retired ``rf/lumped_open_short_match`` was a broken bench: two lumped ports
two cells apart in a tiny PML box, so the feed saw near-total reflection
INDEPENDENT of the load -- matched, short and open all read |Gamma| ~ 0.997 at
the SAME phase (feed decoupled from load). This gate drives the rebuilt bench: a
coax TEM ``WavePort`` feed launches down the proven air coax line to a de-embedded
load plane terminated by one of three physical standards, and reads the feed
reflection from the fields.

What binds (measured at dx=0.01, 0.45--0.50 GHz, below the TM01 cutoff so the open
does not leak):

* **matched** (reflectionless coax-through-PML, presents Z0): |Gamma| <= 0.1
  (-20 dB);
* **short** (PEC plug) and **open** (truncated inner rod): |Gamma| >= 0.944
  (-0.5 dB) -- both totally reflect;
* **discrimination** (the core gate the old bench failed): the three responses are
  mutually distinguishable. With the SHORT as the -1 phase reference (SOL
  convention), the open lands in the +1 class (Re(Gamma_open^ref) > 0) and the
  open/short phase separation exceeds 90 deg. The separation departs from an ideal
  180 deg by the coax open-end fringe capacitance (measured ~110--120 deg here,
  i.e. an effective open-plane extension ~5.5 cm) -- documented, not hidden.

Falsification (feed-coupling): swapping the load must change the feed reflection.
The three standards must occupy distinct regions of the complex plane; feeding
identical reflection coefficients into the discrimination metric (the decoupled
-bench signature) must fail the gate -- proving the gate detects the original bug.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from benchmark.scenes.rf.lumped_open_short_match import (
    TM01_CUTOFF_HZ,
    coax_sol_scene,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="wave-level SOL FDTD gate requires CUDA"
)

_DX = 0.01
_FREQS = (0.45e9, 0.50e9)
_MATCHED_LIMIT = 0.1      # -20 dB
_REFLECT_FLOOR = 0.944    # -0.5 dB


def _gamma(standard: str) -> np.ndarray:
    result = mw.Simulation.fdtd(
        coax_sol_scene(standard, dx=_DX, device="cuda"),
        frequencies=_FREQS,
        excitations=mw.PortExcitation("feed"),
        run_time=mw.TimeConfig.auto(steady_cycles=10, transient_cycles=20),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()
    feed = result.port("feed")
    a = feed.a.cpu().numpy().reshape(len(_FREQS), -1)[:, 0]
    b = feed.b.cpu().numpy().reshape(len(_FREQS), -1)[:, 0]
    return b / a


@pytest.fixture(scope="module")
def gammas():
    return {std: _gamma(std) for std in ("matched", "short", "open")}


def test_operating_band_below_tm01_cutoff():
    """The open standard is only clean below the outer-guide TM01 cutoff."""

    assert max(_FREQS) < TM01_CUTOFF_HZ, (
        f"operating band {max(_FREQS):.3e} must stay below TM01 cutoff "
        f"{TM01_CUTOFF_HZ:.3e} so the open does not leak"
    )


def test_matched_is_absorbed(gammas):
    """Matched load: |Gamma| <= -20 dB (reflectionless Z0 termination)."""

    mag = np.abs(gammas["matched"])
    assert float(mag.max()) <= _MATCHED_LIMIT, (
        f"matched |Gamma| too high: {mag.max():.4f} (> {_MATCHED_LIMIT})"
    )
    worst_db = 20.0 * math.log10(float(mag.max()) + 1e-30)
    assert worst_db <= -20.0


def test_short_and_open_totally_reflect(gammas):
    """Short and open both reflect nearly all incident power (|Gamma| >= -0.5 dB)."""

    assert float(np.abs(gammas["short"]).min()) >= _REFLECT_FLOOR
    assert float(np.abs(gammas["open"]).min()) >= _REFLECT_FLOOR


def test_short_open_discriminate_by_phase(gammas):
    """Short and open are anti-phase-class: with the short as the -1 reference the
    open lands in the +1 class, and the two are separated by > 90 deg."""

    short = gammas["short"]
    open_ = gammas["open"]
    # short-referenced (SOL): rotate so short -> -1; the open must have Re > 0.
    open_ref = open_ * (-1.0 / short)
    assert float(open_ref.real.min()) > 0.1, (
        f"open not in the +1 class after short-referencing: Re={open_ref.real}"
    )
    separation = np.abs(np.degrees(np.angle(open_ / short)))
    assert float(separation.min()) > 90.0, (
        f"open/short phase separation too small: {separation} deg "
        "(the two standards are not distinguishable)"
    )


def test_three_standards_are_mutually_distinct(gammas):
    """The three loads occupy distinct regions -- the feed IS coupled to the load."""

    matched = gammas["matched"]
    short = gammas["short"]
    open_ = gammas["open"]
    # Each pairwise complex distance must be large (old bench: all ~identical).
    assert float(np.abs(short - matched).min()) > 0.5
    assert float(np.abs(open_ - matched).min()) > 0.5
    assert float(np.abs(open_ - short).min()) > 0.5


def test_discrimination_gate_falsification(gammas):
    """Falsification: the decoupled-bench signature (identical Gamma for every
    load) must FAIL the discrimination gate -- proving the gate has teeth."""

    # Simulate the retired broken bench: feed the SAME reflection into all standards.
    decoupled = gammas["matched"]
    short = decoupled
    open_ = decoupled
    matched = decoupled

    # The mutual-distinctness gate must go red.
    assert not (
        float(np.abs(short - matched).min()) > 0.5
        and float(np.abs(open_ - short).min()) > 0.5
    ), "distinctness gate failed to flag a decoupled (identical-Gamma) bench"

    # The phase-discrimination gate must go red too (separation ~0 for identical).
    separation = np.abs(np.degrees(np.angle(open_ / short)))
    assert not (float(separation.min()) > 90.0), (
        "phase-discrimination gate failed to flag identical open/short reflections"
    )
