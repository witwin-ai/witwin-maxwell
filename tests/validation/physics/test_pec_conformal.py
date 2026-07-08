"""Conformal PEC sub-cell accuracy: a parallel-plate cavity resonance sweep.

A 1D PEC cavity (PEC boundary at x=0, PEC-material wall at x=wall) has fundamental
resonance f = c / (2 * wall). Sweeping the wall across one grid cell, a staircase PEC
snaps the effective wall to grid nodes, so the resonance is piecewise constant with a
sharp stair-step jump. The stable partial-fill conformal PEC places the wall at the
fractional crossing, so the resonance tracks the wall position smoothly (small steps,
monotonic). This test locks in that smoothness benefit.
"""

import numpy as np
import pytest
import torch

import witwin.maxwell as mw


C0 = 299_792_458.0

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Conformal PEC cavity validation needs CUDA"
)


def _peak_frequency(signal, dt):
    signal = signal - signal.mean()
    n = len(signal)
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, dt)
    band = (freqs > 5e7) & (freqs < 1.2e9)
    idx = np.where(band)[0][np.argmax(spectrum[band])]
    a, b, c = spectrum[idx - 1], spectrum[idx], spectrum[idx + 1]
    curvature = a - 2 * b + c
    delta = 0.5 * (a - c) / curvature if curvature != 0 else 0.0
    return freqs[idx] + delta * (freqs[1] - freqs[0])


def _cavity_resonance(wall_x, pec_mode, *, grid=0.01, length=0.6, steps=12000):
    domain = mw.Domain(bounds=((0.0, length), (0.0, 0.05), (0.0, 0.05)))
    scene = mw.Scene(
        domain=domain,
        grid=mw.GridSpec.uniform(grid),
        boundary=mw.BoundarySpec.periodic().with_faces(x_low="pec", x_high="pec"),
        device="cuda",
        subpixel_samples=mw.SubpixelSpec(pec=pec_mode),
    )
    scene.add_structure(
        mw.Structure(
            name="wall",
            geometry=mw.Box(
                position=((wall_x + length) / 2.0, 0.025, 0.025),
                size=(length - wall_x, 0.2, 0.2),
            ),
            material=mw.Material.pec(),
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(wall_x * 0.5, 0.025, 0.025),
            width=grid * 2,
            polarization=(0, 0, 1),
            source_time=mw.RickerWavelet(frequency=5e8, amplitude=1.0),
            name="src",
        )
    )
    scene.add_monitor(
        mw.FieldTimeMonitor(position=(wall_x * 0.27, 0.025, 0.025), name="probe", components=("Ez",))
    )
    result = mw.Simulation.fdtd(
        scene, frequencies=(5e8,), run_time=mw.TimeConfig(time_steps=steps)
    ).run()
    signal = result.monitor("probe")["data"].detach().cpu().numpy().reshape(-1)
    return _peak_frequency(signal, result.solver.dt)


def test_conformal_pec_wall_sweep_is_smoother_than_staircase():
    walls = np.linspace(0.30, 0.31, 6)  # sweep across one grid cell
    staircase = np.array([_cavity_resonance(w, "staircase") for w in walls])
    conformal = np.array([_cavity_resonance(w, "conformal") for w in walls])

    staircase_max_step = float(np.abs(np.diff(staircase)).max())
    conformal_max_step = float(np.abs(np.diff(conformal)).max())

    # Staircase snaps: one large stair-step jump across the cell. Conformal tracks the
    # wall smoothly, so its largest adjacent-step is materially smaller.
    assert conformal_max_step < 0.6 * staircase_max_step

    # Conformal tracks the monotonic analytic trend (resonance falls as the wall recedes).
    analytic = C0 / (2.0 * walls)
    assert np.corrcoef(conformal, analytic)[0, 1] > 0.9
