"""Broadband generic surface-impedance FDTD validation (S1.2).

A generic ``SurfaceImpedanceMedium`` carrying a passive rational good-conductor surface
admittance is stepped by the per-edge Z-form ADE. These forward gates hold the runtime
to the frozen ``SurfaceAcceptanceBudget``: the broadband reflection matches the analytic
Leontovich value, the surface is genuinely dissipative (|Gamma| < 1), the result is
stable and consistent across three resolution levels, and the generic-surface field
block is CUDA-graph capturable.

The scene is a good-conductor half-space flush against the +x boundary spanning the full
transverse cross-section, driven at normal incidence by a CW point dipole; reflection is
read from the vacuum standing-wave ratio, the same reference-plane-invariant measurement
the incumbent narrowband gate uses.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box
from witwin.maxwell.media import RationalSurfaceImpedance, SurfaceImpedanceMedium
from witwin.maxwell.fdtd.surface_impedance_reference import (
    ETA_0,
    SURFACE_ACCEPTANCE_BUDGET,
    good_conductor_surface_impedance,
    leontovich_reflection,
)

_C = 299792458.0
_SIGMA = 50.0
_BAND = (0.5e9, 5.0e9)


def _good_conductor_medium(order=6):
    freqs = torch.logspace(math.log10(_BAND[0]), math.log10(_BAND[1]), 120, dtype=torch.float64)
    admittance = (1.0 / good_conductor_surface_impedance(_SIGMA, freqs)).to(torch.complex128)
    model = RationalSurfaceImpedance.fit(freqs, admittance, order=order, band=_BAND)
    return SurfaceImpedanceMedium(impedance=model, name="coating")


def _analytic_gamma(frequency):
    z_s = good_conductor_surface_impedance(_SIGMA, [frequency])
    return float(leontovich_reflection(z_s, eta=ETA_0).abs()[0])


def _run_reflection(frequency, cells_per_wavelength, cuda_graph=False):
    dx = (_C / frequency) / cells_per_wavelength
    x_lo, x_hi, surface, source_x = -0.5, 0.5, 0.1, -0.3
    window = (-0.28, 0.08)
    trans = 4.0 * dx
    scene = mw.Scene(
        domain=mw.Domain(bounds=((x_lo, x_hi), (-trans / 2, trans / 2), (-trans / 2, trans / 2))),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.faces(default="periodic", num_layers=10, strength=1.0, x=("pml", "pml")),
        device="cuda",
        sources=[
            mw.PointDipole(
                position=(source_x, 0.0, 0.0),
                polarization=(0.0, 0.0, 1.0),
                width=2.0 * dx,
                source_time=mw.CW(frequency=frequency, amplitude=40.0),
                name="pw",
            )
        ],
        structures=[
            mw.Structure(
                geometry=Box(position=((surface + x_hi) / 2.0, 0.0, 0.0), size=(x_hi - surface, 1.0, 1.0)),
                material=_good_conductor_medium(),
            )
        ],
    )
    result = mw.Simulation.fdtd(
        scene,
        frequencies=[frequency],
        run_time=mw.TimeConfig.auto(steady_cycles=16, transient_cycles=24),
        full_field_dft=True,
        cuda_graph=cuda_graph,
    ).run()
    ez = result.field("Ez")
    ez = (ez["data"] if isinstance(ez, dict) else ez).detach().cpu().numpy()
    assert np.isfinite(ez).all(), "generic surface-impedance field diverged (non-finite)"
    line = np.abs(ez).reshape(ez.shape[0], -1).mean(axis=1)
    xs = result.solver.scene.x_nodes64
    v = line[(xs >= window[0]) & (xs <= window[1])]
    gamma = float((v.max() - v.min()) / (v.max() + v.min()))
    return gamma, result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_broadband_reflection_matches_leontovich():
    budget = SURFACE_ACCEPTANCE_BUDGET.analytic_reflection_relative_error
    for frequency in (1.0e9, 2.0e9, 3.0e9):
        gamma, _ = _run_reflection(frequency, cells_per_wavelength=45)
        analytic = _analytic_gamma(frequency)
        rel = abs(gamma - analytic) / analytic
        assert rel < budget, (
            f"f={frequency/1e9:.1f} GHz: |Gamma|_generic={gamma:.4f} vs Leontovich "
            f"{analytic:.4f} (rel err {rel:.4f}, budget {budget})"
        )
        # A passive good-conductor surface absorbs a real fraction: dissipation >= 0
        # shows up as a reflection strictly below unity (min_local_surface_dissipation).
        assert gamma < 0.99


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_three_level_convergence_is_stable_and_in_budget():
    levels = (30, 45, 60)
    assert len(levels) >= SURFACE_ACCEPTANCE_BUDGET.convergence_levels
    frequency = 2.0e9
    analytic = _analytic_gamma(frequency)
    budget = SURFACE_ACCEPTANCE_BUDGET.analytic_reflection_relative_error
    gammas = []
    for cpw in levels:
        gamma, _ = _run_reflection(frequency, cells_per_wavelength=cpw)
        gammas.append(gamma)
        rel = abs(gamma - analytic) / analytic
        assert rel < budget, f"cpw={cpw}: rel err {rel:.4f} exceeds {budget}"
    # The refined solution is Cauchy-consistent: the three levels agree tightly (the
    # surface is stable under refinement, not drifting or diverging).
    assert max(gammas) - min(gammas) < budget * analytic


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_generic_surface_field_block_is_cuda_graph_capturable():
    frequency = 2.0e9
    gamma, result = _run_reflection(frequency, cells_per_wavelength=45, cuda_graph=True)
    # The per-edge ADE advance is torch-native GPU state mutation with no per-step host
    # input, so the field-update block (with the surface write inside) captures into the
    # CUDA graph rather than falling back to the eager path.
    assert getattr(result.solver, "surface_impedance_enabled", False) is True
    assert getattr(result.solver, "_cuda_graph_active", False) is True
    # The captured ADE state advance is live (replayed each step), not frozen: the
    # reflection under the graph still matches the analytic Leontovich value.
    analytic = _analytic_gamma(frequency)
    rel = abs(gamma - analytic) / analytic
    assert rel < SURFACE_ACCEPTANCE_BUDGET.analytic_reflection_relative_error
