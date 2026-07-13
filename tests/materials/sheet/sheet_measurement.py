"""Shared normal-incidence sheet scattering measurement for 2D-material tests.

Runs an Ez-polarized plane wave along x onto a free-standing sheet at x = 0 and
extracts the complex sheet transmission/reflection coefficients (t, r) from the
full-field DFT. PEC walls on the z faces and PMC walls on the y faces are
exactly compatible with the Ez plane wave, so the run is a clean 1D scattering
experiment. The vacuum regions on both sides of the sheet are decomposed into
forward/backward waves at the numerical wavenumber and the sheet 2x2 scattering
relations are solved, which makes the extraction insensitive to residual PML
reflections.
"""

import numpy as np
import torch

import witwin.maxwell as mw
from witwin.core import Box

C0 = 299792458.0
Z0 = 376.730313668  # free-space impedance [ohm]


def decompose_two_waves(x, profile, lo, hi, k):
    """Least-squares (forward, backward) wave amplitudes of a complex line profile."""
    mask = (x >= lo) & (x <= hi)
    xs = x[mask]
    ys = profile[mask]
    e_plus = np.exp(1j * k * xs)
    e_minus = np.exp(-1j * k * xs)
    g00 = np.sum(np.conj(e_plus) * e_plus)
    g01 = np.sum(np.conj(e_plus) * e_minus)
    g10 = np.conj(g01)
    g11 = np.sum(np.conj(e_minus) * e_minus)
    r0 = np.sum(np.conj(e_plus) * ys)
    r1 = np.sum(np.conj(e_minus) * ys)
    det = g00 * g11 - g01 * g10
    return (g11 * r0 - g01 * r1) / det, (g00 * r1 - g10 * r0) / det


def sheet_scattering_coefficients(profile, *, x, dt, frequency, fit_gap, fit_extent):
    """Extract the sheet (t, r) from a DFT Ez line profile through a sheet at x = 0.

    Solves ``fwd_r = t*fwd_l + r*bwd_r`` and ``bwd_l = t*bwd_r + r*fwd_l`` from
    the two-wave decompositions of the windows ``(-fit_extent, -fit_gap)`` and
    ``(fit_gap, fit_extent)``.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.shape != profile.shape:
        raise ValueError("The field profile and x-coordinate array must have identical shapes.")
    spacing = np.diff(x)
    dl = float(np.mean(spacing))
    if not np.allclose(spacing, dl, rtol=1e-10, atol=1e-15):
        raise ValueError("Sheet scattering extraction requires a uniform x grid.")
    omega = 2.0 * np.pi * frequency
    k_num = 2.0 / dl * np.arcsin(np.sin(omega * dt / 2.0) * dl / (C0 * dt))
    fwd_l, bwd_l = decompose_two_waves(x, profile, -fit_extent, -fit_gap, k_num)
    fwd_r, bwd_r = decompose_two_waves(x, profile, fit_gap, fit_extent, k_num)
    det = fwd_l * fwd_l - bwd_r * bwd_r
    t = (fwd_r * fwd_l - bwd_l * bwd_r) / det
    r = (bwd_l * fwd_l - fwd_r * bwd_r) / det
    return t, r


def measure_sheet_scattering(
    material,
    frequency,
    *,
    dl,
    half_length,
    half_width,
    fit_gap,
    fit_extent,
    pml_layers=12,
    amplitude=80.0,
    steady_cycles=16,
    transient_cycles=20,
):
    """Run the normal-incidence sheet experiment and return the complex (t, r)."""
    scene = mw.Scene(
        domain=mw.Domain(
            bounds=(
                (-half_length, half_length),
                (-half_width, half_width),
                (-half_width, half_width),
            )
        ),
        grid=mw.GridSpec.uniform(dl),
        boundary=mw.BoundarySpec.faces(
            default="pml", num_layers=pml_layers, strength=1.0, y="pmc", z="pec"
        ),
        device="cuda",
        sources=[
            mw.PlaneWave(
                direction=(1.0, 0.0, 0.0),
                polarization=(0.0, 0.0, 1.0),
                source_time=mw.CW(frequency=frequency, amplitude=amplitude),
                name="pw",
            )
        ],
        structures=[
            mw.Structure(
                geometry=Box(
                    position=(0.0, 0.0, 0.0),
                    size=(0.0, 8.0 * half_width, 8.0 * half_width),
                ),
                material=material,
            )
        ],
    )
    result = mw.Simulation.fdtd(
        scene,
        frequencies=[frequency],
        run_time=mw.TimeConfig.auto(steady_cycles=steady_cycles, transient_cycles=transient_cycles),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=True,
    ).run()
    ez = result.E.z
    if ez.ndim == 4:
        ez = ez[0]
    profile = ez[:, ez.shape[1] // 2, ez.shape[2] // 2].detach().cpu().numpy()
    dt = result.solver.dt
    x = result.solver.scene.x_nodes64.copy()
    del result
    torch.cuda.empty_cache()
    return sheet_scattering_coefficients(
        profile,
        x=x,
        dt=dt,
        frequency=frequency,
        fit_gap=fit_gap,
        fit_extent=fit_extent,
    )
