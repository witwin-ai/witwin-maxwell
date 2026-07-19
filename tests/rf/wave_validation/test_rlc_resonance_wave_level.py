"""Series-RLC resonance: wave-level status and the retained fast contract test.

Gate taxonomy (S0.3):
* ``test_series_rlc_companion_impedance_matches_analytic`` -- **analytic-identity**
  (retained as a fast contract test; NOT an exit gate).
* ``test_wave_level_rlc_resonance_open_gap`` -- documents the **wave-level** gap
  honestly (xfail), because a clean field-level RLC resonance extraction was not
  achieved this session.

Honest-exit record (audit 2026-07-18, S1.2): the plan-01 RLC gate swept the
trapezoidal companion-impedance *formula* against the analytic resonance without
running the solver. Running the solver was attempted three ways this session:

1. a lumped two-port bench (feed + RLC-terminated port) -- the load-port branch
   current peak is dominated by the feed/load near-field parasitic resonance and
   barely tracks the RLC ``C`` (the measured C(1pF)->C(2pF) peak ratio is far from
   the ideal sqrt(2); exact numbers are in the regenerated benchmark artifact
   ``docs/assessments/rf-wave-validation-2026-07-18/rf__series_parallel_rlc.json``);
   it does not isolate the RLC;
2. a circuit-bound port -- ``V/I`` at the bound port is imposed by the MNA circuit
   solve (tautological), and the port cannot also be directly excited;
3. a parallel-plate transmission line fed by a lumped port -- did not guide a
   clean TEM wave at benchmark resolution (|S11| ~ 1 for every load).

The analytic reference ``f0 = 1/(2 pi sqrt(L C))`` binds as the first-line
reference and is recorded in the benchmark; the *wave-level* RLC resonance
(reflection read from a propagating transmission structure) is an OPEN GAP,
recorded as such rather than papered over with a parasitic-dominated peak that
would pass by coincidence.
"""

from __future__ import annotations

import math

import pytest
import torch

import witwin.maxwell as mw
from benchmark.scenes.rf.series_parallel_rlc import series_rlc_scene


def test_series_rlc_companion_impedance_matches_analytic():
    """analytic-identity (fast contract, non-gating): companion Z == closed form.

    This is the plan-01 formula check, re-labelled per S0.3. It validates the
    discrete series-RLC companion impedance against the closed form; it is NOT
    the wave-level exit gate and must never be presented as one.
    """

    r, l, c = 8.0, 0.5e-9, 1.0e-12  # noqa: E741
    termination = mw.SeriesRLC(r=r, l=l, c=c)
    f0 = 1.0 / (2.0 * math.pi * math.sqrt(l * c))
    omega = 2.0 * math.pi * torch.tensor([0.5 * f0, f0, 1.5 * f0], dtype=torch.float64)

    z = termination.impedance(omega)
    # At resonance the reactance cancels and Z == R.
    assert abs(float(z[1].real) - r) < 1.0e-6
    assert abs(float(z[1].imag)) < 1.0e-3 * r
    # Below resonance the series C dominates (capacitive, Im < 0); above, L (Im > 0).
    assert float(z[0].imag) < 0.0
    assert float(z[2].imag) > 0.0


@pytest.mark.xfail(
    reason="Wave-level RLC resonance from a propagating transmission structure is an "
    "open gap (S1.2): the lumped/circuit benches are parasitic-dominated or "
    "tautological. Analytic f0 binds; recorded honestly, not faked. strict=True so a "
    "silent xpass (bench unexpectedly tracking C) fails the run and reopens the gap "
    "for real evidence rather than closing it invisibly.",
    strict=True,
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_wave_level_rlc_resonance_open_gap():
    """The lumped bench must NOT be trusted to track the RLC resonance.

    This xfail encodes the open gap: on the lumped two-port bench the measured
    resonance is parasitic-dominated and does not track ``C`` as ``1/sqrt(C)``.
    The assertion below is the behaviour a *valid* wave-level gate would require
    (doubling C lowers the measured peak by a clear margin); it is expected to
    FAIL on the current bench, documenting the gap rather than hiding it.
    """

    def measured(c):
        f0 = 1.0 / (2.0 * math.pi * math.sqrt(0.5e-9 * c))
        freqs = tuple(float(x) for x in torch.linspace(f0 * 0.45, f0 * 2.2, 45))
        result = mw.Simulation.fdtd(
            series_rlc_scene(r=8.0, l=0.5e-9, c=c, device="cuda"),
            frequencies=freqs,
            excitations=mw.PortExcitation(
                "feed",
                amplitude=1.0,
                source_impedance="matched",
                source_time=mw.GaussianPulse(frequency=f0, fwidth=1.2 * f0),
            ),
            run_time=mw.TimeConfig(time_steps=4000),
            spectral_sampler=mw.SpectralSampler(window="hanning"),
        ).run()
        current = result.port("load").current.cpu().abs().squeeze()
        return freqs[int(torch.argmax(current))]

    assert measured(1.0e-12) > 1.30 * measured(2.0e-12)
