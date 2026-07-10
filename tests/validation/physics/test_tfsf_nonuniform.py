"""TFSF / mode-plane injection on nonuniform (graded) Yee grids.

Before P5.3 the auxiliary-grid TFSF machinery and the 2D mode solver required
the injection region / mode plane to be *exactly* uniform (spacing spread <=
1e-6 relative), so any ``GridSpec.custom`` / ``GridSpec.auto`` grid with even a
mild grade over the region raised. The single-scalar-delta contract is now a
*bounded* contract: a perfectly uniform region returns its exact spacing
bit-for-bit, a mildly graded region is accepted when the physically-derived
error it induces stays below a stated bound (using the region-mean spacing as
the effective delta), and an over-graded region raises with the *predicted*
error quoted in the message.

The bounds:

* TFSF: the leading FDTD numerical-dispersion excess is ``(k0*d)^2/24`` for a
  vacuum wave of wavenumber ``k0 = omega/c``; the spread of that excess between
  the finest and coarsest region cell is the residual the total/scattered-field
  faces cannot cancel. Bound ``1e-3``.
* Mode plane: a centered finite-difference operator built from a single spacing
  loses one order on a graded grid, with leading relative error
  ``(d_max - d_min)/d_mean``. Bound ``1e-2``.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.excitation.modes import (
    _MODE_PLANE_SPACING_SPREAD_BOUND,
    _local_uniform_plane_spacing,
)
from witwin.maxwell.fdtd.excitation.tfsf_common import (
    _TFSF_DISPERSION_SPREAD_BOUND,
    require_locally_uniform_axis,
)


C_LIGHT = 299792458.0


# --------------------------------------------------------------------------- #
# Fast unit-level coverage of the bounded contract (no CUDA / no solver run).  #
# --------------------------------------------------------------------------- #

class _FakeScene:
    def __init__(self, dx, dy=None, dz=None):
        self.dx_primal64 = np.asarray(dx, dtype=np.float64)
        self.dy_primal64 = np.asarray(dx if dy is None else dy, dtype=np.float64)
        self.dz_primal64 = np.asarray(dx if dz is None else dz, dtype=np.float64)


class _FakeSolver:
    def __init__(self, scene, frequency):
        self.scene = scene
        self.c = C_LIGHT
        self.source_omega = 2.0 * math.pi * float(frequency)


def _geom_cells(d0, per_cell_ratio, n):
    return [d0 * per_cell_ratio ** i for i in range(n)]


def _reported_number(message):
    match = re.search(r"is ([0-9.eE+\-]+) \(min=", message)
    assert match is not None, message
    return float(match.group(1))


def test_tfsf_uniform_region_returns_exact_spacing_bit_for_bit():
    # A perfectly uniform region must return its exact cell spacing (the legacy
    # scalar), not a recomputed mean, so uniform grids stay bitwise unchanged.
    cells = [0.05] * 12
    solver = _FakeSolver(_FakeScene(cells), frequency=1.0e9)
    result = require_locally_uniform_axis(solver, "x", 0, 12, context="TFSF")
    assert result == 0.05
    assert result == float(np.asarray(cells, dtype=np.float64).min())


def test_tfsf_mild_grading_within_bound_returns_region_mean():
    # A gentle grade whose induced numerical-dispersion spread is below the bound
    # is accepted, and the region-mean spacing is used as the effective delta.
    cells = _geom_cells(0.05, 1.0004, 12)  # ~0.4% total, well within bound at 1 GHz
    window = np.asarray(cells, dtype=np.float64)
    solver = _FakeSolver(_FakeScene(cells), frequency=1.0e9)
    result = require_locally_uniform_axis(solver, "x", 0, 12, context="TFSF")
    assert result != float(window.min())  # genuinely took the graded path
    assert result == pytest.approx(float(window.mean()), rel=0.0, abs=0.0)


def test_tfsf_over_bound_region_raises_with_predicted_error():
    # A strong grade must raise, quote the actual predicted phase-velocity spread
    # (a number above the bound), and keep the matched substring used elsewhere.
    cells = _geom_cells(0.05, 1.01, 12)  # ~10% grade
    solver = _FakeSolver(_FakeScene(cells), frequency=3.0e9)
    with pytest.raises(ValueError, match="locally uniform grid spacing along axis 'x'") as excinfo:
        require_locally_uniform_axis(solver, "x", 0, 12, context="TFSF")
    reported = _reported_number(str(excinfo.value))
    assert reported > _TFSF_DISPERSION_SPREAD_BOUND
    # The reported number matches the closed-form dispersion spread of the window.
    window = np.asarray(cells, dtype=np.float64)[0:12]
    k0 = 2.0 * math.pi * 3.0e9 / C_LIGHT
    expected = k0 * k0 * (float(window.max()) ** 2 - float(window.min()) ** 2) / 24.0
    assert reported == pytest.approx(expected, rel=5e-3)


def test_tfsf_bound_relaxes_at_lower_frequency():
    # The bound is physical: the same grade that fails at high frequency (large
    # numerical dispersion) is accepted when the wave is better resolved.
    cells = _geom_cells(0.05, 1.003, 12)
    high = _FakeSolver(_FakeScene(cells), frequency=2.0e9)
    low = _FakeSolver(_FakeScene(cells), frequency=2.0e8)
    with pytest.raises(ValueError):
        require_locally_uniform_axis(high, "x", 0, 12, context="TFSF")
    accepted = require_locally_uniform_axis(low, "x", 0, 12, context="TFSF")
    assert accepted == pytest.approx(float(np.asarray(cells, dtype=np.float64).mean()))


def test_mode_plane_uniform_returns_exact_spacing_bit_for_bit():
    # Call indices (1, len-1) span the whole primal array as the aperture window.
    cells = [0.05] * 12
    result = _local_uniform_plane_spacing(_FakeScene(cells), "x", 1, 11)
    assert result == 0.05
    assert result == float(np.asarray(cells, dtype=np.float64).min())


def test_mode_plane_mild_grading_within_bound_returns_region_mean():
    cells = _geom_cells(0.05, 1.0005, 12)  # fractional spread ~0.5% < 1e-2 bound
    window = np.asarray(cells, dtype=np.float64)
    result = _local_uniform_plane_spacing(_FakeScene(cells), "x", 1, 11)
    assert result != float(window.min())  # genuinely took the graded path
    assert result == pytest.approx(float(window.mean()), rel=0.0, abs=0.0)


def test_mode_plane_over_bound_raises_with_predicted_variation():
    cells = _geom_cells(0.05, 1.01, 12)  # ~10% grade
    with pytest.raises(
        ValueError, match="locally uniform grid spacing along axis 'x'"
    ) as excinfo:
        _local_uniform_plane_spacing(_FakeScene(cells), "x", 1, 11)
    reported = _reported_number(str(excinfo.value))
    assert reported > _MODE_PLANE_SPACING_SPREAD_BOUND
    window = np.asarray(cells, dtype=np.float64)
    expected = (float(window.max()) - float(window.min())) / float(window.mean())
    assert reported == pytest.approx(expected, rel=5e-3)


# --------------------------------------------------------------------------- #
# End-to-end FDTD validation of the bound (CUDA).                             #
# --------------------------------------------------------------------------- #

def _geom_graded_nodes(lo, hi, n_cells, total_ratio):
    """``n_cells`` geometrically graded cells spanning [lo, hi] exactly."""
    g = total_ratio ** (1.0 / (n_cells - 1))
    widths = g ** np.arange(n_cells)
    widths *= (hi - lo) / widths.sum()
    nodes = np.concatenate([[lo], lo + np.cumsum(widths)]).astype(np.float64)
    nodes[-1] = hi
    return nodes


def _axis_coords_for_length(scene, axis, length):
    nodes = np.asarray(getattr(scene, f"{axis}_nodes64"), dtype=np.float64)
    half = np.asarray(getattr(scene, f"{axis}_half64"), dtype=np.float64)
    if length == len(nodes):
        return nodes
    if length == len(half):
        return half
    return np.linspace(float(nodes[0]), float(nodes[-1]), length)


def _tfsf_leakage_ratio(result, component, bounds):
    solver = result.solver
    scene = solver.scene
    raw = result.raw_output[component]
    field = torch.abs(raw).detach().cpu().numpy() if isinstance(raw, torch.Tensor) else np.abs(raw)
    xc = _axis_coords_for_length(scene, "x", field.shape[0])
    yc = _axis_coords_for_length(scene, "y", field.shape[1])
    zc = _axis_coords_for_length(scene, "z", field.shape[2])
    xx, yy, zz = np.meshgrid(xc, yc, zc, indexing="ij")
    dx = float(np.asarray(scene.dx_primal64).max())
    dy = float(np.asarray(scene.dy_primal64).max())
    dz = float(np.asarray(scene.dz_primal64).max())
    inside = (
        (xx >= bounds[0][0]) & (xx <= bounds[0][1])
        & (yy >= bounds[1][0]) & (yy <= bounds[1][1])
        & (zz >= bounds[2][0]) & (zz <= bounds[2][1])
    )
    outside = (
        (xx < bounds[0][0] - dx) | (xx > bounds[0][1] + dx)
        | (yy < bounds[1][0] - dy) | (yy > bounds[1][1] + dy)
        | (zz < bounds[2][0] - dz) | (zz > bounds[2][1] + dz)
    )
    inside_max = float(np.max(field[inside]))
    outside_max = float(np.max(field[outside]))
    return outside_max / max(inside_max, 1e-12), inside_max, outside_max


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_tfsf_graded_region_within_bound_keeps_leakage_small():
    # A genuinely graded propagation axis whose injection region stays within the
    # dispersion bound is accepted, and the total-field/scattered-field null still
    # holds: the scattered region stays near-zero (no spurious face injection).
    lo, hi = -0.96, 0.96
    zn = _geom_graded_nodes(lo, hi, 24, total_ratio=1.004)  # ~0.4% graded z axis
    xn = np.linspace(lo, hi, 25)
    bounds = ((-0.32, 0.32), (-0.32, 0.32), (-0.32, 0.32))
    scene = mw.Scene(
        domain=mw.Domain(bounds=((lo, hi), (lo, hi), (lo, hi))),
        grid=mw.GridSpec.custom(xn, xn, zn),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(0.0, 0.0, 1.0),
            polarization=(1.0, 0.0, 0.0),
            source_time=mw.CW(frequency=1.0e9, amplitude=80.0),
            injection=mw.TFSF(bounds=bounds),
            name="tfsf_graded",
        )
    )
    # The graded z region is genuinely nonuniform yet accepted.
    prepared = mw.Simulation.fdtd(scene, frequency=1.0e9).prepare()
    zc = np.asarray(prepared.solver.scene.dz_primal64)
    assert float(zc.max()) - float(zc.min()) > 1e-6

    result = mw.Simulation.fdtd(
        scene,
        frequency=1.0e9,
        run_time=mw.TimeConfig.auto(steady_cycles=6, transient_cycles=15),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=True,
    ).run()
    leakage_ratio, inside_max, outside_max = _tfsf_leakage_ratio(result, "Ex", bounds)
    assert inside_max > 0.0
    assert outside_max < inside_max
    # Within-bound grading keeps the scattered-field null small and bounded
    # (empirically leakage ~= 15 * dispersion_spread, here spread ~= 3.7e-4), far
    # below the O(1) leakage a single delta on a strongly graded region produces
    # and far above the ~1e-5 uniform-grid floor.
    assert leakage_ratio < 1e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_tfsf_strongly_graded_region_rejected_with_predicted_error():
    lo, hi = -0.5, 0.5
    nodes = np.linspace(lo, hi, 21)
    graded = nodes.copy()
    graded[10] += 0.02  # region spacings become 0.03 / 0.07
    scene = mw.Scene(
        domain=mw.Domain(bounds=((lo, hi), (lo, hi), (lo, hi))),
        grid=mw.GridSpec.custom(nodes, nodes, graded),
        boundary=mw.BoundarySpec.pml(num_layers=4),
        device="cuda",
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(0.0, 0.0, 1.0),
            polarization=(1.0, 0.0, 0.0),
            source_time=mw.GaussianPulse(frequency=3e9, fwidth=1e9, amplitude=1.0),
            injection=mw.TFSF(bounds=((-0.15, 0.15), (-0.15, 0.15), (-0.15, 0.15))),
            name="pw",
        )
    )
    with pytest.raises(
        ValueError, match="locally uniform grid spacing along axis 'z'"
    ) as excinfo:
        mw.Simulation.fdtd(scene, frequency=3e9).prepare()
    assert _reported_number(str(excinfo.value)) > _TFSF_DISPERSION_SPREAD_BOUND


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_mode_source_graded_plane_within_bound_runs():
    # A mildly graded transverse mode plane is accepted and produces a finite,
    # non-trivial injected field.
    lo, hi = -0.5, 0.5
    yn = _geom_graded_nodes(lo, hi, 20, total_ratio=1.004)  # fractional spread < 1e-2
    xn = np.linspace(lo, hi, 21)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((lo, hi), (lo, hi), (lo, hi))),
        grid=mw.GridSpec.custom(xn, yn, xn),
        boundary=mw.BoundarySpec.pml(num_layers=4),
        device="cuda",
    )
    scene.add_source(
        mw.ModeSource(
            position=(-0.2, 0.0, 0.0),
            size=(0.0, 0.4, 0.4),
            polarization="Ez",
            source_time=mw.CW(frequency=1.0e9, amplitude=1.0),
            name="mode0",
        )
    )
    solver = mw.Simulation.fdtd(scene, frequency=1e9).prepare().solver
    # The transverse (y) plane is genuinely graded yet accepted.
    yc = np.asarray(solver.scene.dy_primal64)
    assert float(yc.max()) - float(yc.min()) > 1e-6
    solver.solve(time_steps=60, dft_frequency=None, dft_window="none", full_field_dft=False)
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        assert torch.isfinite(getattr(solver, name)).all(), name
    assert float(torch.abs(solver.Ez).max()) > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_mode_source_strongly_graded_plane_rejected_with_predicted_variation():
    lo, hi = -0.5, 0.5
    nodes = np.linspace(lo, hi, 21)
    graded = nodes.copy()
    graded[10] += 0.02
    scene = mw.Scene(
        domain=mw.Domain(bounds=((lo, hi), (lo, hi), (lo, hi))),
        grid=mw.GridSpec.custom(nodes, graded, nodes),
        boundary=mw.BoundarySpec.pml(num_layers=4),
        device="cuda",
    )
    scene.add_source(
        mw.ModeSource(
            position=(-0.2, 0.0, 0.0),
            size=(0.0, 0.4, 0.4),
            polarization="Ez",
            source_time=mw.CW(frequency=1.0e9, amplitude=1.0),
            name="mode0",
        )
    )
    with pytest.raises(
        ValueError, match="locally uniform grid spacing along axis 'y'"
    ) as excinfo:
        mw.Simulation.fdtd(scene, frequency=1e9).prepare()
    assert _reported_number(str(excinfo.value)) > _MODE_PLANE_SPACING_SPREAD_BOUND
