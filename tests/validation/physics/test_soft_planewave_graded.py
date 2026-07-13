"""Soft ``PlaneWave`` numerical-dispersion phase correction on graded grids.

Before P5.3 the soft ``PlaneWave`` surface source tuned its injected wavefront's
phase velocity from the *global minimum* grid spacing (``solver.min_dx/dy/dz``).
On a uniform grid that is exact, but on a graded (nonuniform) grid the finest cell
can sit far from the launch plane, so the discrete-dispersion phase velocity used
to time-align the Yee E/H source planes -- and to phase the aperture for oblique
incidence -- was mistuned by the numerical dispersion of a cell the wavefront
never crosses.

The correction now uses the spacing local to the launch footprint
(``soft_plane_wave_region_spacing``): the launch-plane cell along the injection
axis (where the one-way E/H half-cell offset lives) and the physical-aperture mean
along each tangential axis. On a grid whose interior is uniform along an axis, that
axis returns its exact spacing, bit-for-bit the global-minimum value the correction
used before, so uniform grids are unchanged.

The unit tests below quantify the improvement without CUDA: on a 2.6x-graded
propagation axis with the source in the coarse region (~6 cells/wavelength there),
the discrete-dispersion phase-velocity error of the injected wavefront drops from
~5.3% (global-minimum spacing) to exactly 0 (local spacing, which matches the launch
cell's own numerical wavenumber bit-for-bit). The E/H half-cell timing phase error
drops from ~2.9e-2 rad to 0.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.fdtd.dispersion import solve_numerical_wavenumber
from witwin.maxwell.fdtd.excitation.spatial import (
    physical_interior_indices,
    soft_plane_wave_index,
    soft_plane_wave_region_spacing,
)


C_LIGHT = 299792458.0


# --------------------------------------------------------------------------- #
# Grid + solver helpers.                                                       #
# --------------------------------------------------------------------------- #

def _geom_graded_nodes(lo, hi, n_cells, total_ratio):
    """``n_cells`` geometrically graded cells spanning [lo, hi] exactly.

    ``total_ratio < 1`` grows the cells from ``hi`` toward ``lo`` (coarse near the
    low edge, where the forward soft source is placed).
    """
    g = total_ratio ** (1.0 / (n_cells - 1))
    widths = g ** np.arange(n_cells)
    widths *= (hi - lo) / widths.sum()
    nodes = np.concatenate([[lo], lo + np.cumsum(widths)]).astype(np.float64)
    nodes[-1] = hi
    return nodes


def _auto_dt(dx_min, dy_min, dz_min, frequency, steps_per_cycle=30):
    """Reproduce ``FDTDSolver.auto_dt`` for a CW source (dt = min(period/N, CFL))."""
    dt_resolution = (1.0 / float(frequency)) / steps_per_cycle
    dt_courant = 1.0 / (C_LIGHT * math.sqrt(1.0 / dx_min**2 + 1.0 / dy_min**2 + 1.0 / dz_min**2))
    return min(dt_resolution, dt_courant)


class _FakeSolver:
    """Minimal solver surface ``solve_numerical_wavenumber`` reads."""

    def __init__(self, scene, frequency, dt):
        self.scene = scene
        self.c = C_LIGHT
        self.source_omega = 2.0 * math.pi * float(frequency)
        self.dt = float(dt)


def _cpu_scene(x_nodes, y_nodes, z_nodes, *, num_layers=6):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((float(x_nodes[0]), float(x_nodes[-1])),
                                 (float(y_nodes[0]), float(y_nodes[-1])),
                                 (float(z_nodes[0]), float(z_nodes[-1])))),
        grid=mw.GridSpec.custom(x_nodes, y_nodes, z_nodes),
        boundary=mw.BoundarySpec.pml(num_layers=num_layers),
        device="cpu",
    )
    return prepare_scene(scene)


def _global_min_deltas(scene):
    return {
        "x": float(np.asarray(scene.dx_primal64).min()),
        "y": float(np.asarray(scene.dy_primal64).min()),
        "z": float(np.asarray(scene.dz_primal64).min()),
    }


# --------------------------------------------------------------------------- #
# Uniform-grid regression guard: correction is bit-for-bit unchanged.          #
# --------------------------------------------------------------------------- #

def test_soft_plane_uniform_spacing_is_global_min_bit_for_bit():
    # A uniform interior must return, per axis, exactly the global-minimum spacing
    # the correction used before -- so the injected phase velocity is byte-identical
    # on uniform grids and no regression is introduced.
    nodes = np.linspace(-1.0, 1.0, 41)
    scene = _cpu_scene(nodes, nodes, nodes)
    plane_index = soft_plane_wave_index(scene, "z", 1.0)
    deltas = soft_plane_wave_region_spacing(
        scene, injection_axis="z", plane_index=plane_index, direction_sign=1
    )
    old = _global_min_deltas(scene)
    assert deltas["x"] == old["x"]
    assert deltas["y"] == old["y"]
    assert deltas["z"] == old["z"]

    # And the discrete-dispersion wavenumber is identical for normal and oblique
    # directions, i.e. the whole phase correction is unchanged on a uniform grid.
    dt = _auto_dt(old["x"], old["y"], old["z"], 1.0e9)
    solver = _FakeSolver(scene, 1.0e9, dt)
    for direction in [(0.0, 0.0, 1.0), (0.3, 0.0, 0.95)]:
        k_old = solve_numerical_wavenumber(solver, direction, old)
        k_new = solve_numerical_wavenumber(solver, direction, deltas)
        assert k_new == k_old


# --------------------------------------------------------------------------- #
# Graded-grid: the injection axis uses the launch-plane cell, not global min.  #
# --------------------------------------------------------------------------- #

def test_soft_plane_graded_injection_axis_uses_launch_cell_spacing():
    lo, hi = -1.0, 1.0
    zn = _geom_graded_nodes(lo, hi, 60, total_ratio=1.0 / 3.0)  # coarse near low edge
    xn = np.linspace(lo, hi, 45)
    scene = _cpu_scene(xn, xn, zn)
    plane_index = soft_plane_wave_index(scene, "z", 1.0)
    dz = np.asarray(scene.dz_primal64)
    launch_cell = float(dz[plane_index])
    global_min = float(dz.min())

    deltas = soft_plane_wave_region_spacing(
        scene, injection_axis="z", plane_index=plane_index, direction_sign=1
    )
    # The launch plane genuinely sits in a coarse region, well away from the finest
    # cell, so the local spacing differs substantially from the global minimum.
    assert launch_cell / global_min > 2.0
    assert deltas["z"] == launch_cell
    assert deltas["z"] != global_min


def test_soft_plane_graded_reverse_direction_uses_launch_side_cell():
    # For a backward-propagating wave the launch plane sits near the high edge and
    # the cell it launches into is the one on the propagation (low) side.
    lo, hi = -1.0, 1.0
    zn = _geom_graded_nodes(lo, hi, 60, total_ratio=3.0)  # coarse near high edge
    xn = np.linspace(lo, hi, 45)
    scene = _cpu_scene(xn, xn, zn)
    plane_index = soft_plane_wave_index(scene, "z", -1.0)
    dz = np.asarray(scene.dz_primal64)
    deltas = soft_plane_wave_region_spacing(
        scene, injection_axis="z", plane_index=plane_index, direction_sign=-1
    )
    assert deltas["z"] == float(dz[plane_index - 1])


# --------------------------------------------------------------------------- #
# Graded-grid: the phase-velocity error of the injected wavefront collapses.   #
# --------------------------------------------------------------------------- #

def test_soft_plane_graded_reduces_injected_phase_velocity_error():
    lo, hi = -1.0, 1.0
    zn = _geom_graded_nodes(lo, hi, 60, total_ratio=1.0 / 3.0)
    xn = np.linspace(lo, hi, 45)
    scene = _cpu_scene(xn, xn, zn)
    frequency = 1.0e9

    plane_index = soft_plane_wave_index(scene, "z", 1.0)
    dz = np.asarray(scene.dz_primal64)
    launch_cell = float(dz[plane_index])
    old = _global_min_deltas(scene)
    new = soft_plane_wave_region_spacing(
        scene, injection_axis="z", plane_index=plane_index, direction_sign=1
    )
    dt = _auto_dt(old["x"], old["y"], old["z"], frequency)
    solver = _FakeSolver(scene, frequency, dt)

    direction = (0.0, 0.0, 1.0)
    k_old = solve_numerical_wavenumber(solver, direction, old)
    k_new = solve_numerical_wavenumber(solver, direction, new)
    # The exact local numerical wavenumber the launch cell actually supports.
    k_local = solve_numerical_wavenumber(
        solver, direction, {"x": old["x"], "y": old["y"], "z": launch_cell}
    )

    vp_local = solver.source_omega / k_local
    vp_err_old = abs(solver.source_omega / k_old - vp_local) / vp_local
    vp_err_new = abs(solver.source_omega / k_new - vp_local) / vp_local

    # New correction matches the launch cell's own numerical wavenumber bit-for-bit.
    assert k_new == k_local
    assert vp_err_new == 0.0
    # Old global-minimum correction mistuned the phase velocity by several percent.
    assert vp_err_old > 1e-2

    # E/H half-cell timing phase error at the launch plane: 0.5 * d_local * (k - k_local).
    eh_phase_err_old = 0.5 * launch_cell * (k_old - k_local)
    eh_phase_err_new = 0.5 * launch_cell * (k_new - k_local)
    assert eh_phase_err_new == 0.0
    assert abs(eh_phase_err_old) > 1e-2

    # Guard the reported magnitudes so the numbers in the change stay honest.
    assert vp_err_old == pytest.approx(5.30e-2, rel=0.1)
    assert abs(eh_phase_err_old) == pytest.approx(2.95e-2, rel=0.1)


# --------------------------------------------------------------------------- #
# Graded-grid: a graded tangential aperture uses its region-mean spacing.      #
# --------------------------------------------------------------------------- #

def test_soft_plane_graded_tangential_axis_uses_aperture_mean():
    lo, hi = -1.0, 1.0
    zn = np.linspace(lo, hi, 41)                       # uniform injection axis
    xn = _geom_graded_nodes(lo, hi, 60, total_ratio=3.0)  # graded tangential aperture
    yn = np.linspace(lo, hi, 41)
    scene = _cpu_scene(xn, yn, zn)
    plane_index = soft_plane_wave_index(scene, "z", 1.0)
    deltas = soft_plane_wave_region_spacing(
        scene, injection_axis="z", plane_index=plane_index, direction_sign=1
    )

    dx = np.asarray(scene.dx_primal64)
    lo_idx, hi_idx = physical_interior_indices(scene, "x")
    aperture_mean = float(dx[lo_idx:hi_idx].mean())

    # Uniform injection axis stays exact; graded tangential aperture uses its mean,
    # not the global minimum cell somewhere in the aperture.
    assert deltas["z"] == float(np.asarray(scene.dz_primal64).min())
    assert deltas["x"] == aperture_mean
    assert deltas["x"] != float(dx.min())


# --------------------------------------------------------------------------- #
# End-to-end FDTD (CUDA): a graded-grid soft plane wave launches cleanly.       #
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_soft_plane_graded_grid_launches_forward_wave():
    lo, hi = -1.0, 1.0
    zn = _geom_graded_nodes(lo, hi, 60, total_ratio=1.0 / 2.5)  # coarse near source
    xn = np.linspace(lo, hi, 45)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((lo, hi), (lo, hi), (lo, hi))),
        grid=mw.GridSpec.custom(xn, xn, zn),
        boundary=mw.BoundarySpec.pml(num_layers=6),
        device="cuda",
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(0.0, 0.0, 1.0),
            polarization=(1.0, 0.0, 0.0),
            source_time=mw.CW(frequency=1.0e9, amplitude=1.0),
            name="soft",
        )
    )
    # The propagation axis is genuinely graded across the injection region.
    prepared = mw.Simulation.fdtd(scene, frequency=1.0e9).prepare()
    dz = np.asarray(prepared.solver.scene.dz_primal64)
    assert float(dz.max()) - float(dz.min()) > 1e-6

    result = mw.Simulation.fdtd(
        scene,
        frequency=1.0e9,
        run_time=mw.TimeConfig.auto(steady_cycles=6, transient_cycles=15),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=True,
    ).run()

    solver = result.solver
    field = torch.abs(result.raw_output["Ex"]).detach().cpu().numpy()
    assert np.isfinite(field).all()

    plane_index = soft_plane_wave_index(solver.scene, "z", 1.0)
    z_plane = float(solver.scene.z[plane_index].item())
    z_nodes = np.asarray(solver.scene.z_nodes64)
    z_axis = z_nodes if field.shape[2] == len(z_nodes) else np.asarray(solver.scene.z_half64)
    profile = field.max(axis=(0, 1))

    forward = profile[z_axis > z_plane + 0.3]
    # A strong forward traveling wave is launched.
    assert forward.size > 0
    assert float(forward.max()) > 1.0

    # The region behind the launch plane (toward the low PML) carries far less than
    # the forward wave: the soft source is one-way on the graded grid.
    z_first_interior = float(solver.scene.z[6].item())
    backward = profile[(z_axis < z_plane - 0.02) & (z_axis > z_first_interior)]
    if backward.size > 0:
        assert float(backward.max()) < 0.25 * float(forward.max())
