"""Regression: guided TE10 mode selection on a closed metallic aperture (F5).

Gate taxonomy (S0.3): **modal-eigensolve** (supporting; a solver-correctness
regression, not a wave-level exit gate).

A hollow rectangular waveguide has a uniform-isotropic (air) mode plane bounded by
PEC walls. The centered vector operator on that plane carries a discrete
transverse null-space branch at ``beta = k0`` (``kc^2 -> 0``). Ordering candidates
purely by the largest real eigenvalue selected that spurious branch whenever the
eigensolver surfaced it (observed at dx in {0.05, 0.0125, 0.01}, while dx=0.02
happened to land on the guided mode). The selector now rejects a near-``k0``
candidate on a closed metallic aperture only when its transverse envelope is
plane-wave-like (block-averaged variation ~0), so the guided TE10 mode is
returned across every grid tier.

Physics constraint (supervisor ruling): the rejection is NOT unconditional --
``beta = k0`` is the correct TEM answer for a doubly-connected line (coax), which
is solved on the separate electrostatic path and must be unchanged.
"""

from __future__ import annotations

import math

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


def _modal_beta(scene, frequency: float) -> float:
    prepared = prepare_scene(scene)
    manifest = resolve_waveport_run_manifest(prepared, mw.PortSweep(), (frequency,))
    return float(manifest.prepared_ports[0].tracking.beta[0, 0].real)


@pytest.mark.parametrize("dx", [0.05, 0.02, 0.0125, 0.01])
def test_waveguide_te10_not_spurious_k0_branch(dx: float) -> None:
    """The guided TE10 beta is returned at every tier, never the k0 null-space branch."""
    fc = C0 / (2.0 * GUIDE_A)
    frequency = 1.8 * fc
    k0 = 2.0 * math.pi * frequency / C0
    beta_analytic = math.sqrt(k0**2 - (math.pi / GUIDE_A) ** 2)

    beta = _modal_beta(rectangular_waveguide_scene(dx=dx, device="cuda"), frequency)

    # The spurious branch sits at beta = k0; the guided TE10 at beta_analytic ~ 7.84.
    assert beta == pytest.approx(beta_analytic, rel=0.02), (
        f"dx={dx}: got beta={beta:.4f}, expected guided TE10 {beta_analytic:.4f} "
        f"(k0={k0:.4f} is the spurious null-space branch)."
    )
    # Explicit guard that the k0 branch is not what came back.
    assert beta < 0.95 * k0


def test_coax_tem_beta_is_k0_unchanged() -> None:
    """TEM line: beta = k0 is correct and must remain untouched by the F5 rejection."""
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
