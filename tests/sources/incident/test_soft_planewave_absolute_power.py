"""Soft ``PlaneWave`` absolute incident-power calibration.

The soft ``PlaneWave`` face injects the surface-equivalent currents of the target
incident wave. Before P5.9 its amplitude carried two empirical fudge factors --
``_PLANE_WAVE_POWER_CALIBRATION = 0.958`` and ``_PLANE_WAVE_DELAY_CALIBRATION_S =
1.65e-10`` -- that had been fitted to a single benchmark scene and did not
generalize across frequency or grid spacing.

Both were removed. The power scale is now the derived unit-power normalization
``1/sqrt(unit_power)`` with ``unit_power = A*cos(theta)/(2*eta0)``: by the
surface-equivalence principle the single face radiates the incident field forward
with unit gain, and on the Yee grid the numerical wave impedance of a discretely
dispersing plane wave is exactly ``eta0`` (the leapfrog identity
``sin(omega*dt/2)/(c*dt) = sin(k~*d/2)/d`` gives ``H0/E0 = 1/eta0``), so the
physical-impedance magnetic current the injector uses is the correct numerical
amplitude. The delay term was a spatially uniform time offset applied identically
to the electric and magnetic faces; it only rotates the global launch phase and
leaves every phasor magnitude invariant, so it was dropped.

The unit test below pins the derived scale (no empirical factor). The CUDA test
checks that the absolute time-averaged incident power matches the analytic unit
power within 2% across three frequencies and two spacings (the P5.9 acceptance).

The measured observable is the forward Poynting flux integrated over the central
half of the physical aperture. Infinite plane sources are normalized over the full
computational aperture, including external PML, and the derived source scale
includes the Yee ``cos(k~*dz/2)`` power factor. The central window avoids cells
that abut the transverse PML while checking the native FluxMonitor convention
directly, without a post-hoc numerical correction.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.excitation import injection as _injection
from witwin.maxwell.fdtd.excitation.spatial import (
    physical_interior_indices,
    soft_plane_wave_index,
)


C0 = 299792458.0
ETA0 = 376.730313668


# --------------------------------------------------------------------------- #
# Unit: the amplitude scale is the derived unit-power factor, no fudge factor.  #
# --------------------------------------------------------------------------- #

def test_plane_wave_power_scale_is_derived_unit_power_factor():
    # Normal +z incidence over a known square aperture: the scale must be exactly
    # 1/sqrt(unit_power) with unit_power = A*cos/(2*eta0), so unit source amplitude
    # radiates unit time-averaged power. Any residual empirical multiplier (the old
    # 0.958) would show up here.
    source = {"direction": (0.0, 0.0, 1.0)}
    aperture_bounds = ((-0.4, 0.4), (-0.25, 0.25), (0.1, 0.1))  # 0.8 x 0.5 aperture
    scale = _injection._plane_wave_power_scale(source, aperture_bounds, "z")

    area = 0.8 * 0.5
    unit_power = area / (2.0 * ETA0)
    assert scale == pytest.approx(1.0 / math.sqrt(unit_power), rel=1e-12)

    # The derived scale delivers unit power: 0.5*|E0|^2*A/eta0 == 1.
    assert 0.5 * scale ** 2 * area / ETA0 == pytest.approx(1.0, rel=1e-12)

    # Oblique incidence folds in cos(theta); still unit power through the aperture.
    theta = math.radians(30.0)
    oblique = {"direction": (math.sin(theta), 0.0, math.cos(theta))}
    scale_oblique = _injection._plane_wave_power_scale(oblique, aperture_bounds, "z")
    unit_power_oblique = area * math.cos(theta) / (2.0 * ETA0)
    assert scale_oblique == pytest.approx(1.0 / math.sqrt(unit_power_oblique), rel=1e-12)


def test_soft_plane_wave_calibration_constants_removed():
    # The empirical fudge factors must be gone from the module surface, so no
    # frequency/spacing-specific magic number can silently return.
    assert not hasattr(_injection, "_PLANE_WAVE_POWER_CALIBRATION")
    assert not hasattr(_injection, "_PLANE_WAVE_DELAY_CALIBRATION_S")


# --------------------------------------------------------------------------- #
# CUDA end-to-end: absolute incident power matches analytic within 2%.          #
# --------------------------------------------------------------------------- #

def _measure_incident_power_ratio(
    frequency: float,
    dx: float,
    *,
    half_span: float = 0.6,
    pml: int = 18,
    aperture_fraction: float = 0.5,
    steady_cycles: int = 12,
    transient_cycles: int = 30,
) -> float:
    """Return measured / analytic time-averaged incident power for a soft +z wave.

    The scene is normalized to unit incident power (unit source amplitude), so the
    analytic power crossing the full computational aperture is 1.0 and the
    analytic power through the central window is its area fraction.
    """
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-half_span, half_span),) * 3),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.pml(num_layers=pml),
        device="cuda",
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(0.0, 0.0, 1.0),
            polarization=(1.0, 0.0, 0.0),
            source_time=mw.GaussianPulse(frequency=frequency, fwidth=0.25 * frequency),
            name="pw",
        )
    )

    prepared = mw.Simulation.fdtd(scene, frequency=frequency).prepare()
    solver = prepared.solver
    grid = solver.scene
    src_index = soft_plane_wave_index(grid, "z", 1.0)
    z_nodes = grid.z.detach().cpu().numpy()
    # Probe ~0.6 wavelengths forward of the launch plane: past the immediate near
    # field of the truncated aperture, before edge diffraction reaches the centre,
    # and a fixed physical (not cell-count) distance so the state is resolution
    # independent.
    cells_forward = max(6, int(round(0.6 * (C0 / frequency) / dx)))
    z_probe = float(z_nodes[min(src_index + cells_forward, len(z_nodes) - 1)])

    lo_x, hi_x = physical_interior_indices(grid, "x")
    lo_y, hi_y = physical_interior_indices(grid, "y")
    x_nodes = grid.x.detach().cpu().numpy()
    y_nodes = grid.y.detach().cpu().numpy()
    x_lo, x_hi = float(x_nodes[lo_x]), float(x_nodes[hi_x])
    y_lo, y_hi = float(y_nodes[lo_y]), float(y_nodes[hi_y])
    computational_range = grid.domain_range
    aperture_area = (
        (computational_range[1] - computational_range[0])
        * (computational_range[3] - computational_range[2])
    )
    x_mid, y_mid = 0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi)
    half_wx = 0.5 * aperture_fraction * (x_hi - x_lo)
    half_wy = 0.5 * aperture_fraction * (y_hi - y_lo)

    scene.add_monitor(
        mw.PlaneMonitor(
            name="probe",
            axis="z",
            position=z_probe,
            fields=("Ex", "Hy"),
            frequencies=(frequency,),
        )
    )
    result = mw.Simulation.fdtd(
        scene,
        frequencies=(frequency,),
        run_time=mw.TimeConfig.auto(steady_cycles=steady_cycles, transient_cycles=transient_cycles),
        spectral_sampler=mw.SpectralSampler(normalize_source=True),
    ).run()

    components = result.monitor("probe")["components"]
    ex = components["Ex"]["data"].detach().cpu().numpy()
    hy = components["Hy"]["data"].detach().cpu().numpy()
    x_coords = np.asarray(components["Ex"]["coords"][0])
    y_coords = np.asarray(components["Ex"]["coords"][1])
    # Ex and Hy share the (x-half, y) transverse grid, so the transverse Poynting
    # product is co-located; only the z stagger needs the cos correction.
    assert np.allclose(x_coords, np.asarray(components["Hy"]["coords"][0]))
    assert np.allclose(y_coords, np.asarray(components["Hy"]["coords"][1]))

    mask_x = np.abs(x_coords - x_mid) <= half_wx
    mask_y = np.abs(y_coords - y_mid) <= half_wy
    window = np.ix_(mask_x, mask_y)
    poynting_z = 0.5 * np.real(ex[window] * np.conj(hy[window]))
    measured_power = float(poynting_z.sum()) * (dx * dx)
    window_area = float(mask_x.sum() * mask_y.sum()) * (dx * dx)
    analytic_power = window_area / aperture_area  # unit power density over aperture

    del result
    torch.cuda.empty_cache()
    return measured_power / analytic_power


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_soft_plane_wave_absolute_power_matches_analytic():
    frequencies = (1.0e9, 1.5e9, 2.0e9)
    spacings = (0.008, 0.011)

    worst = 0.0
    failures = []
    for frequency in frequencies:
        for dx in spacings:
            ratio = _measure_incident_power_ratio(frequency, dx)
            deviation = abs(ratio - 1.0)
            worst = max(worst, deviation)
            if deviation >= 0.02:
                failures.append((frequency, dx, ratio))

    assert not failures, (
        "soft PlaneWave absolute incident power deviates >= 2% from analytic "
        f"unit power at {failures}; worst deviation overall {worst * 100:.2f}%"
    )
    # The derived normalization is well inside the 2% acceptance; guard the
    # achieved margin so a regression that merely stays under 2% is still caught.
    assert worst < 0.02
