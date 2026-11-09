import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.postprocess import find_resonances

C0 = 299792458.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for native FDTD")
def test_pec_cavity_te101_ringdown_matches_analytic_frequency():
    a, b, d = 0.30, 0.15, 0.20  # TE101 (Ey) is the fundamental mode for a > d > b
    spacing = 0.01
    analytic = 0.5 * C0 * math.sqrt((1.0 / a) ** 2 + (1.0 / d) ** 2)

    scene = mw.Scene(
        domain=mw.Domain(bounds=((0.0, a), (0.0, b), (0.0, d))),
        grid=mw.GridSpec.uniform(spacing),
        boundary=mw.BoundarySpec.pec(),
        device="cuda",
    )
    pulse = mw.GaussianPulse(frequency=analytic, fwidth=0.3e9)
    scene.add_source(
        mw.PointDipole(
            position=(0.37 * a, 0.5 * b, 0.41 * d),
            polarization="Ey",
            width=spacing,
            source_time=pulse,
            name="src",
        )
    )
    scene.add_monitor(
        mw.FieldTimeMonitor("probe", components=("Ey",), position=(0.62 * a, 0.5 * b, 0.57 * d), interval=2)
    )
    result = mw.Simulation.fdtd(
        scene,
        frequency=analytic,
        run_time=mw.TimeConfig(time_steps=3000),
        full_field_dft=False,
    ).run()

    payload = result.monitor("probe")
    start_time = 2.0 * pulse.delay
    # Half-step current sampling changes roundoff poles; resolve modes above the FP32 noise floor.
    modes = find_resonances(payload, (0.7e9, 1.1e9), start_time=start_time, min_relative_amplitude=1e-6)

    assert len(modes) == 1
    frequency = float(modes.frequency[0])
    assert frequency == pytest.approx(analytic, rel=1e-2)

    # The Yee-grid dispersion relation of the discrete TE101 mode pins the
    # extracted frequency much tighter than the continuum value.
    dt = float(result.solver.dt)
    spatial = (math.sin(0.5 * math.pi * spacing / a) ** 2 + math.sin(0.5 * math.pi * spacing / d) ** 2) / spacing**2
    discrete = math.asin(C0 * dt * math.sqrt(spatial)) / (math.pi * dt)
    assert frequency == pytest.approx(discrete, rel=2e-3)
    # A lossless PEC cavity rings without decay.
    assert abs(float(modes.decay_rate[0])) < 1e-4 * 2.0 * math.pi * frequency
