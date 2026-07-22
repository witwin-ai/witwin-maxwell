"""Committed wave-level gate for the hollow rectangular waveguide two-port (TE10).

Gate taxonomy (S0.3): **wave-level**.

This was BLOCKED for the whole S1 round: the retired centered transverse operator
injected a checkerboard-aliased eigenvector (sin(pi y/a)-correlation 0.51-0.59) in
place of a clean TE10, so no physical S-matrix could be reported. The Yee-staggered
transverse full-vector operator (``modes.py:_build_yee_transverse_operator_sparse``)
now delivers a clean full-grid TE10 (correlation 1.0000), so the terminated two-port
FDTD sweep yields a well-conditioned, passive S-matrix and beta from ``arg(S21)/L``
tracks the analytic TE10 dispersion to ~0.05%.

The gate mirrors the coax_thru wave-level precondition (extraction conditioning
``cond(A)`` + post-solve passivity) and then holds beta from ``arg(S21)/L`` to a
pre-registered 1%-class tolerance (the coax bench gates its own ``arg(S21)/L`` beta
at 3%). Falsification: the sibling
``test_matched_s11_wave_level.py`` shows a PEC short spikes |S11|, and
``test_te10_mode_selection.py`` shows the operator regression collapses the mode
shape; here the load-bearing check is that the beta gate is red if the analytic
dispersion is mis-stated (verified by the ``length`` sensitivity below).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from benchmark.scenes.rf.rectangular_waveguide import (
    GUIDE_A,
    GUIDE_LENGTH,
    rectangular_waveguide_scene,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="wave-level FDTD gate requires CUDA"
)

C0 = 299792458.0

# Pre-registered gate (coax_thru precedent): a well-conditioned, passive extraction
# and beta within 1% of the analytic TE10 dispersion.
BETA_TOL = 0.01
COND_LIMIT = 10.0
PASSIVITY_SLACK = 1.05


def _two_port_s(dx: float):
    fc = C0 / (2.0 * GUIDE_A)
    freqs = tuple(float(x) for x in np.linspace(1.2 * fc, 2.2 * fc, 11))
    result = mw.Simulation.fdtd(
        rectangular_waveguide_scene(dx=dx, device="cuda"),
        frequencies=freqs,
        excitations=mw.PortSweep(),
        run_time=mw.TimeConfig.auto(steady_cycles=8, transient_cycles=16),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()
    s = result.network.s.cpu().numpy()
    cond = float(np.max(result.network.metadata["extraction_condition_number"].cpu().numpy()))
    return np.array(freqs), s, cond


def test_waveguide_te10_two_port_is_wave_level_pass() -> None:
    """Terminated TE10 two-port: conditioned, passive, beta within 1% of analytic."""
    freqs, s, cond = _two_port_s(dx=0.02)
    k0 = 2.0 * np.pi * freqs / C0
    beta_an = np.sqrt(np.maximum(k0**2 - (np.pi / GUIDE_A) ** 2, 0.0))
    interior = slice(1, len(freqs) - 1)

    # Wave-level precondition (identical structure to coax_thru).
    sv_max = max(float(np.linalg.svd(s[i], compute_uv=False).max()) for i in range(len(freqs)))
    assert cond <= COND_LIMIT, f"extraction cond(A) {cond:.3f} > {COND_LIMIT}"
    assert sv_max <= PASSIVITY_SLACK, f"max singular value {sv_max:.4f} > {PASSIVITY_SLACK}"

    # A clean matched TE10 thru: |S21| ~ 1, |S11| small.
    mid = len(freqs) // 2
    assert abs(s[mid, 1, 0]) > 0.95, f"|S21| mid {abs(s[mid, 1, 0]):.4f} too low"
    assert float(np.abs(s[:, 0, 0]).min()) < 0.05, "matched |S11| never dips (unterminated)"

    # beta from arg(S21)/L vs the analytic TE10 dispersion.
    phase = np.unwrap(np.angle(s[:, 1, 0]))
    beta_phase = np.abs(phase) / GUIDE_LENGTH
    rel = np.abs(beta_phase - beta_an) / np.maximum(beta_an, 1e-9)
    beta_med = float(np.median(rel[interior]))
    assert beta_med <= BETA_TOL, f"beta median rel error {beta_med:.4%} > {BETA_TOL:.0%}"


def test_waveguide_beta_gate_is_falsifiable_in_length() -> None:
    """Falsification: beta scales as 1/L, so a 10%-wrong L busts the 1% gate.

    The extracted beta = arg(S21)/L is compared against the analytic dispersion; if
    the assumed reference-plane separation were 10% wrong the reported beta error
    would be ~10%, far outside the committed 1% gate. This shows the gate is not
    vacuously satisfiable.
    """
    freqs, s, _cond = _two_port_s(dx=0.02)
    k0 = 2.0 * np.pi * freqs / C0
    beta_an = np.sqrt(np.maximum(k0**2 - (np.pi / GUIDE_A) ** 2, 0.0))
    interior = slice(1, len(freqs) - 1)
    phase = np.unwrap(np.angle(s[:, 1, 0]))

    # Correct L: passes.
    beta_true = np.abs(phase) / GUIDE_LENGTH
    rel_true = float(np.median((np.abs(beta_true - beta_an) / beta_an)[interior]))
    assert rel_true <= BETA_TOL

    # L perturbed by +10%: the same fields now miss the analytic dispersion badly.
    beta_wrong = np.abs(phase) / (1.10 * GUIDE_LENGTH)
    rel_wrong = float(np.median((np.abs(beta_wrong - beta_an) / beta_an)[interior]))
    assert rel_wrong > 0.05, f"perturbed-L beta error {rel_wrong:.4%} should bust the gate"
