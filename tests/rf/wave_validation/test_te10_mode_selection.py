"""Regression: guided TE10 mode selection on a closed metallic aperture (F1/F2).

Gate taxonomy (S0.3): **modal-eigensolve** (supporting; a solver-correctness
regression, not a wave-level exit gate).

Round-4 correction (EXECUTED). Ordering candidates by the largest real eigenvalue
selected the transverse null-space branch at ``beta = k0``; when that was avoided
the selector still returned a CHECKERBOARD-aliased eigenvector that merely SHARES
the TE10 eigenvalue -- its ``sin(pi y/a)``-correlation is ~0.000 while its beta
matches analytic TE10 to <1%. Asserting only the eigenVALUE (the previous test)
therefore passed on a physically wrong mode. This test asserts the eigenVECTOR.

The selector (F1) now rejects the k0 branch by an absolute transverse-uniformity
signature, enables the checkerboard/wall-peak filters on the uniform-isotropic
aperture, and NEVER substitutes a spurious eigenvector -- it raises instead. The
underlying transverse VECTOR operator, however, cannot represent a clean full-grid
TE10 on this hollow guide: the centered uniform-isotropic branch composes a
stride-two stencil that decouples the odd/even transverse sublattices, so the
half-wave ``sin(pi y/a)`` lives on ONE sublattice with the other ~0 (executed:
best recoverable full-grid sin-correlation over the whole degenerate subspace is
~0.62). Fixing that needs a symmetric-BC Yee-staggered transverse operator (see
docs/reference/rf-wave-validation-2026-07-18.md, "open items"). Until then the
waveguide eigenVECTOR assertion is an xfail; the coax TEM path (separate
electrostatic solve) is unchanged and green.
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


@pytest.mark.xfail(
    strict=True,
    reason="Transverse vector operator decouples odd/even sublattices; a clean "
    "full-grid TE10 (corr>=0.99) needs a symmetric-BC staggered operator (open item).",
)
@pytest.mark.parametrize("dx", [0.05, 0.025, 0.02, 0.0125, 0.01])
def test_waveguide_te10_eigenvector_is_sin(dx: float) -> None:
    """The selected TE10 Ez profile must be sin(pi y/a) (correlation >= 0.99)."""
    fc = C0 / (2.0 * GUIDE_A)
    corr = _te10_ez_correlation(rectangular_waveguide_scene(dx=dx, device="cuda"), 1.8 * fc)
    assert corr >= 0.99, f"dx={dx}: TE10 Ez sin-correlation {corr:.4f} < 0.99"


@pytest.mark.xfail(
    strict=True,
    reason="Same transverse-operator sublattice decoupling; also exercises the "
    "high-frequency (6 fc) branch where the old envelope threshold mis-fired.",
)
def test_waveguide_te10_high_frequency_returns_genuine_te10() -> None:
    """At dx=0.005, f=6 fc the genuine TE10 (not TE20) must be returned."""
    fc = C0 / (2.0 * GUIDE_A)
    corr = _te10_ez_correlation(rectangular_waveguide_scene(dx=0.005, device="cuda"), 6.0 * fc)
    assert corr >= 0.99, f"high-f TE10 Ez sin-correlation {corr:.4f} < 0.99"


def test_waveguide_selector_refuses_to_substitute_a_spurious_mode() -> None:
    """F1d: with no structurally-valid candidate the selector raises, never substitutes."""
    fc = C0 / (2.0 * GUIDE_A)
    with pytest.raises(ValueError, match="will not substitute a spurious eigenvector"):
        _te10_ez_correlation(rectangular_waveguide_scene(dx=0.05, device="cuda"), 1.8 * fc)


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
