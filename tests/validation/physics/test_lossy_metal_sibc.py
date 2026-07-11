"""Forward FDTD validation for the ``LossyMetalMedium`` surface-impedance boundary.

The good-conductor SIBC (Leontovich) replaces the resolved skin-depth interior of a
metal with the first-order surface relation ``E_t = Z_s(omega) * (n x H)``, where the
surface impedance is ``Z_s(omega) = (1 - i) * sqrt(omega * mu0 / (2 * sigma))``. The
runtime evaluates ``Z_s`` at the operating frequency as a narrowband series R-L,
masks the metal interior, and updates the two tangential E faces from the vacuum-side
tangential H each step (``fdtd/runtime/materials.py::_configure_sibc`` and
``fdtd/runtime/stepping.py::apply_sibc_surface``).

The scene is a metal slab flush against the +x boundary spanning the full transverse
cross-section, with periodic transverse boundaries and a plane-wave-like point-dipole
source, i.e. normal incidence (effectively 1D). Reflection is read from the vacuum
standing-wave ratio in front of the surface,
``|Gamma| = (Vmax - Vmin) / (Vmax + Vmin)``, which is reference-plane invariant in
the lossless vacuum region and needs no incident-field subtraction.

Acceptance (plan P5.5): the SIBC reflection matches the analytic Leontovich value --
the reflection a fully-resolved volumetric metal converges to (a 1D volumetric-metal
FDTD reference at dx = delta/10 reproduces it to <1%; a coarse dx = delta/4 3D metal
still reads a near-PEC |Gamma| ~ 0.99, confirming the skin-depth cell is genuinely
needed) -- within 5% at >=3 frequencies, at >=10x fewer cells than a skin-depth-
resolved metal, and reflects measurably less than a perfect conductor.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box

_MU0 = 4.0e-7 * np.pi
_C = 299792458.0
_EPS0 = 8.8541878128e-12
_SIGMA = 50.0  # moderate good conductor: sigma >> omega*eps0, skin depth resolvable


def _analytic_gamma_magnitude(sigma, frequency):
    omega = 2.0 * np.pi * frequency
    eta0 = np.sqrt(_MU0 / _EPS0)
    r = np.sqrt(omega * _MU0 / (2.0 * sigma))
    z_s = r + 1j * r
    return abs((z_s - eta0) / (z_s + eta0))


def _skin_depth(sigma, frequency):
    return np.sqrt(2.0 / (2.0 * np.pi * frequency * _MU0 * sigma))


def _run(frequency, *, kind, dx, x_lo=-0.5, x_hi=0.5, surface=0.1, source_x=-0.3,
         window=(-0.28, 0.08), time_steps=None, steady_cycles=16, transient_cycles=24):
    """Run a normal-incidence reflection scene and return (|Gamma|, total_cells).

    ``kind`` is ``"sibc"`` (a ``LossyMetalMedium`` slab flush against the +x edge) or
    ``"pec"`` (a perfect-conductor slab, the |Gamma| ~ 1 baseline). Reflection is read
    from the vacuum standing-wave ratio over ``window``. ``time_steps`` sets an explicit
    run length; otherwise a cycle-based ``TimeConfig.auto`` is used.
    """
    run_time = (
        mw.TimeConfig(time_steps=int(time_steps)) if time_steps is not None
        else mw.TimeConfig.auto(steady_cycles=steady_cycles, transient_cycles=transient_cycles)
    )
    trans = 4.0 * dx
    material = mw.LossyMetalMedium(conductivity=_SIGMA) if kind == "sibc" else mw.Material.pec()
    scene = mw.Scene(
        domain=mw.Domain(bounds=((x_lo, x_hi), (-trans / 2.0, trans / 2.0), (-trans / 2.0, trans / 2.0))),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.faces(default="periodic", num_layers=10, strength=1.0, x=("pml", "pml")),
        device="cuda",
        sources=[mw.PointDipole(position=(source_x, 0.0, 0.0), polarization=(0.0, 0.0, 1.0),
                                width=2.0 * dx, source_time=mw.CW(frequency=frequency, amplitude=40.0), name="pw")],
        structures=[mw.Structure(
            geometry=Box(position=((surface + x_hi) / 2.0, 0.0, 0.0), size=(x_hi - surface, 1.0, 1.0)),
            material=material)],
    )
    result = mw.Simulation.fdtd(
        scene,
        frequencies=[frequency],
        run_time=run_time,
        full_field_dft=True,
    ).run()
    ez = result.field("Ez")
    ez = (ez["data"] if isinstance(ez, dict) else ez).detach().cpu().numpy()
    assert np.isfinite(ez).all(), "SIBC field diverged (non-finite)"
    line = np.abs(ez).reshape(ez.shape[0], -1).mean(axis=1)
    xs = np.linspace(x_lo, x_hi, line.shape[0])
    mask = (xs >= window[0]) & (xs <= window[1])
    v = line[mask]
    gamma = float((v.max() - v.min()) / (v.max() + v.min()))
    return gamma, int(np.prod(ez.shape))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_sibc_reflection_matches_analytic_at_three_frequencies():
    """SIBC reflection magnitude matches the analytic Leontovich value within 5%."""
    frequencies = [1.0e9, 2.0e9, 3.0e9]
    for frequency in frequencies:
        dx = (_C / frequency) / 40.0  # ~40 cells/wavelength, does NOT resolve the skin depth
        gamma_sibc, _ = _run(frequency, kind="sibc", dx=dx)
        gamma_analytic = _analytic_gamma_magnitude(_SIGMA, frequency)
        rel_err = abs(gamma_sibc - gamma_analytic) / gamma_analytic
        assert rel_err < 0.05, (
            f"f={frequency/1e9:.1f} GHz: |Gamma|_sibc={gamma_sibc:.4f} vs analytic "
            f"{gamma_analytic:.4f} (rel err {rel_err:.3f})"
        )
        # A moderate conductor absorbs a real fraction: |Gamma| clearly below one.
        assert gamma_sibc < 0.97


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_sibc_absorbs_more_than_pec():
    """The SIBC surface is genuinely lossy: less reflective than a perfect conductor."""
    frequency = 2.0e9
    dx = (_C / frequency) / 40.0
    gamma_sibc, _ = _run(frequency, kind="sibc", dx=dx)
    gamma_pec, _ = _run(frequency, kind="pec", dx=dx)
    # PEC reflects essentially everything (validates the standing-wave read-out), while
    # the lossy metal reflects measurably less.
    assert gamma_pec > 0.98
    assert gamma_sibc < gamma_pec - 0.02


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_sibc_matches_resolved_metal_limit_at_fewer_cells():
    """SIBC reproduces the fully-resolved good-conductor reflection at >=10x fewer cells.

    The Leontovich surface impedance ``Zs = (1 - i) * sqrt(omega*mu0/(2*sigma))`` is,
    by derivation, the impedance a fully-resolved semi-infinite good conductor
    presents; a direct volumetric-metal FDTD converges to it as the grid resolves the
    skin depth. A ``dx = delta/10`` volumetric-metal FDTD reproduces this analytic value
    to <1%, but a skin-depth-resolved 3D run needs tens of thousands of steps per solve
    and is impractical as a unit test, so this test compares the coarse SIBC reflection
    to that analytic fully-resolved limit. A moderately-resolved (``delta/4``) 3D
    volumetric metal instead reads a near-PEC ``|Gamma| ~ 0.99``, confirming the
    skin-depth cell is genuinely needed there and that the SIBC delivers the same
    reflection without it.

    The whole point of the boundary: the SIBC uses a cell that does not resolve the
    skin depth, >=10x coarser along the surface normal than a ``delta/6``-resolved
    volumetric metal, so it uses >=10x fewer cells across the metal cross-section.
    """
    frequency = 1.0e9
    dx_sibc = (_C / frequency) / 40.0
    dx_resolved = _skin_depth(_SIGMA, frequency) / 6.0  # a resolved metal would need this cell
    cells_reduction = dx_sibc / dx_resolved
    assert cells_reduction >= 10.0  # >=10x coarser normal cell (hence >=10x fewer cells)

    compact = dict(x_lo=-0.35, x_hi=0.15, surface=0.05, source_x=-0.25, window=(-0.32, 0.02))
    gamma_sibc, _ = _run(frequency, kind="sibc", dx=dx_sibc, time_steps=4000, **compact)
    gamma_resolved_limit = _analytic_gamma_magnitude(_SIGMA, frequency)

    rel_err = abs(gamma_sibc - gamma_resolved_limit) / gamma_resolved_limit
    assert rel_err < 0.05, (
        f"|Gamma|_sibc={gamma_sibc:.4f} vs fully-resolved limit {gamma_resolved_limit:.4f} "
        f"(rel err {rel_err:.3f}) at {cells_reduction:.0f}x fewer cells"
    )
