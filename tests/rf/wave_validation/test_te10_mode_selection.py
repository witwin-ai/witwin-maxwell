"""Regression: guided TE10 mode selection on a closed metallic aperture (F1/F2).

Gate taxonomy (S0.3): **modal-eigensolve** (supporting; a solver-correctness
regression, not a wave-level exit gate).

These assertions gate the eigenVECTOR, not just the eigenVALUE: the retired
transverse operator returned a CHECKERBOARD-aliased eigenvector that merely SHARED
the TE10 eigenvalue (beta matched analytic TE10 to <1% while its ``sin(pi y/a)``
correlation capped at 0.51-0.59), so a value-only check passed on a physically
wrong mode.

E1b (EXECUTED). The selector now solves the closed hollow guide on the
Yee-staggered transverse full-vector operator
(``modes.py:_build_yee_transverse_operator_sparse``), which keeps each transverse
E component on its own Yee location and imposes symmetric PEC walls. The selected
TE10 ``Ez`` is now a clean full-grid ``sin(pi y/a)`` (correlation >= 0.99 across
dx tiers and at 6 fc), so these tests are real asserts. The selector still never
substitutes another mode for a genuinely absent requested index -- it raises. The
coax TEM path (separate electrostatic solve) is unchanged and green.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.waveport_sweep import resolve_waveport_run_manifest
from benchmark.scenes.rf.coax_thru import analytic_z0, coax_thru_scene
from benchmark.scenes.rf.rectangular_waveguide import GUIDE_A, rectangular_waveguide_scene

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="modal solve regression uses the CUDA scene path"
)

C0 = 299792458.0


def _te10_ez_correlation(scene, frequency: float) -> float:
    """sin(pi y/a) correlation of the selected TE10 Ez profile (full transverse grid)."""
    prepared = prepare_scene(scene)
    manifest = resolve_waveport_run_manifest(prepared, mw.PortSweep(), (frequency,))
    md = manifest.prepared_ports[0].mode_data[0][0]
    ez = md["component_profiles"]["Ez"].detach().cpu().numpy().real  # (y, z)
    y = md["coords_u"].detach().cpu().numpy()
    ref = np.outer(np.sin(math.pi * (y - y[0]) / (y[-1] - y[0])), np.ones(ez.shape[1]))
    ez_flat = ez.reshape(-1)
    ref_flat = ref.reshape(-1)
    if np.dot(ez_flat, ref_flat) < 0:
        ez_flat = -ez_flat
    denom = np.linalg.norm(ez_flat) * np.linalg.norm(ref_flat)
    return float(np.dot(ez_flat, ref_flat) / denom) if denom > 0 else 0.0


@pytest.mark.parametrize("dx", [0.05, 0.025, 0.02, 0.0125, 0.01])
def test_waveguide_te10_eigenvector_is_sin(dx: float) -> None:
    """The selected TE10 Ez profile must be sin(pi y/a) (correlation >= 0.99).

    Passing this (E1b) is the acceptance of the Yee-staggered transverse operator:
    it was pinned strict-xfail as the fingerprint of the retired sublattice-
    decoupling defect, whose full-grid sin-correlation capped at 0.51-0.59.
    """
    fc = C0 / (2.0 * GUIDE_A)
    corr = _te10_ez_correlation(rectangular_waveguide_scene(dx=dx, device="cuda"), 1.8 * fc)
    assert corr >= 0.99, f"dx={dx}: TE10 Ez sin-correlation {corr:.4f} < 0.99"


def test_waveguide_te10_high_frequency_returns_genuine_te10() -> None:
    """At dx=0.005, f=6 fc the genuine TE10 (not TE20) must be returned."""
    fc = C0 / (2.0 * GUIDE_A)
    corr = _te10_ez_correlation(rectangular_waveguide_scene(dx=0.005, device="cuda"), 6.0 * fc)
    assert corr >= 0.99, f"high-f TE10 Ez sin-correlation {corr:.4f} < 0.99"


def test_waveguide_te10_operator_returns_clean_sin() -> None:
    """The Yee-staggered operator returns a clean (non-checkerboard) TE10.

    This was previously the ``blocker`` regression asserting ``corr < 0.9`` as the
    fingerprint of the sublattice-decoupling defect. With the symmetric-BC staggered
    operator wired into the selector it now asserts the clean half-wave the fixed
    operator produces (correlation >= 0.99, checkerboard content well below the 0.35
    rejection limit).
    """
    fc = C0 / (2.0 * GUIDE_A)
    corr = _te10_ez_correlation(rectangular_waveguide_scene(dx=0.02, device="cuda"), 1.8 * fc)
    assert corr >= 0.99, f"TE10 Ez sin-correlation {corr:.4f} < 0.99 (operator regressed)"


def test_coax_tem_beta_is_k0_unchanged() -> None:
    """TEM line: beta = k0 is correct and must remain untouched by the F1 rejection."""
    frequency = 1.0e9
    k0 = 2.0 * math.pi * frequency / C0

    scene = coax_thru_scene(dx=0.005, device="cuda")
    prepared = prepare_scene(scene)
    manifest = resolve_waveport_run_manifest(prepared, mw.PortSweep(), (frequency,))
    port0 = manifest.prepared_ports[0]
    beta = float(port0.tracking.beta[0, 0].real)
    z0 = complex(port0.characteristic_impedance[0, 0]).real

    assert beta == pytest.approx(k0, rel=1.0e-9)
    assert z0 == pytest.approx(analytic_z0(), rel=0.05)
