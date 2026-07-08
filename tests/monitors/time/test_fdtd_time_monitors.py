import numpy as np
import pytest
import torch

import witwin.maxwell as mw


def _build_scene(*, bounds=((-0.64, 0.64), (-0.64, 0.64), (-0.64, 0.64)), spacing=0.08, source_time=None):
    if source_time is None:
        source_time = mw.CW(frequency=1e9, amplitude=100.0)
    scene = mw.Scene(
        domain=mw.Domain(bounds=bounds),
        grid=mw.GridSpec.uniform(spacing),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=0.05,
            source_time=source_time,
            name="src",
        )
    )
    return scene


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for native FDTD")
def test_field_time_monitor_matches_running_dft_point():
    frequency = 1e9
    point = (0.0, 0.0, 0.16)
    scene = _build_scene()
    scene.add_monitor(mw.PointMonitor("pt_dft", point, fields=("Ez",)))
    scene.add_monitor(mw.FieldTimeMonitor("pt_time", components=("Ez",), position=point, interval=1))

    result = mw.Simulation.fdtd(
        scene,
        frequencies=[frequency],
        run_time=mw.TimeConfig.auto(steady_cycles=6, transient_cycles=15),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
    ).run()

    dft_payload = result.monitor("pt_dft")
    dft_value = complex(_to_numpy(dft_payload["data"]))

    time_payload = result.monitor("pt_time")
    assert time_payload["kind"] == "field_time"
    trace = _to_numpy(time_payload["field"])
    t = _to_numpy(time_payload["t"])
    assert trace.shape == t.shape

    solver = result.solver
    dt = float(solver.dt)
    entry = solver._observer_spectral_entries[0]
    start_step = int(entry["start_step"])
    window_normalization = float(entry["window_normalization"])

    steps = np.rint(t / dt).astype(int)
    mask = steps >= start_step
    omega = 2.0 * np.pi * frequency
    manual = (2.0 / window_normalization) * np.sum(
        trace[mask] * np.exp(1j * omega * steps[mask] * dt)
    )

    assert abs(dft_value) > 0.0
    # The manual DFT of the recorded time trace must reproduce the running-DFT
    # phasor computed on-the-fly by the point monitor.
    np.testing.assert_allclose(manual, dft_value, rtol=5e-2, atol=1e-3 * abs(dft_value))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for native FDTD")
def test_flux_time_monitor_time_average_matches_dft_flux():
    frequency = 1e9
    position = 0.16
    scene = _build_scene()
    scene.add_monitor(mw.FluxMonitor("port_flux", axis="z", position=position))
    scene.add_monitor(mw.FluxTimeMonitor("port_flux_time", axis="z", position=position, interval=1))

    result = mw.Simulation.fdtd(
        scene,
        frequencies=[frequency],
        run_time=mw.TimeConfig.auto(steady_cycles=10, transient_cycles=15),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()

    dft_flux = float(_to_numpy(result.monitor("port_flux")["flux"]))

    flux_payload = result.monitor("port_flux_time")
    assert flux_payload["kind"] == "flux_time"
    flux_series = _to_numpy(flux_payload["flux"])
    t = _to_numpy(flux_payload["t"])
    assert flux_series.shape == t.shape

    dt = float(result.solver.dt)
    steps_per_cycle = (1.0 / frequency) / dt
    # Average the instantaneous flux over the steady-state tail, spanning many
    # full periods so the 2*omega oscillation of E x H averages out.
    tail_start = int(len(flux_series) - int(round(8 * steps_per_cycle)))
    tail_start = max(tail_start, len(flux_series) // 2)
    mean_instantaneous = float(np.mean(flux_series[tail_start:]))

    assert abs(dft_flux) > 0.0
    np.testing.assert_allclose(mean_instantaneous, dft_flux, rtol=1.0e-1, atol=1.0e-2 * abs(dft_flux))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for native FDTD")
def test_time_monitor_sampling_controls():
    time_steps = 50
    start, stop, interval = 5, 40, 3
    scene = _build_scene()
    scene.add_monitor(
        mw.FieldTimeMonitor(
            "sampled",
            components=("Ez",),
            position=(0.0, 0.0, 0.0),
            start=start,
            stop=stop,
            interval=interval,
        )
    )

    result = mw.Simulation.fdtd(
        scene,
        frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=time_steps),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
    ).run()

    payload = result.monitor("sampled")
    expected_steps = [n for n in range(start, min(stop, time_steps)) if (n - start) % interval == 0]
    dt = float(result.solver.dt)

    t = _to_numpy(payload["t"])
    assert t.shape == (len(expected_steps),)
    assert _to_numpy(payload["field"]).shape == (len(expected_steps),)
    np.testing.assert_allclose(t, dt * np.asarray(expected_steps, dtype=float), rtol=1e-9, atol=1e-15)
    assert payload["start"] == start
    assert payload["stop"] == stop
    assert payload["interval"] == interval


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for native FDTD")
def test_time_monitor_result_plumbing_with_normal_flux_monitor():
    time_steps = 60
    position = 0.16
    scene = _build_scene()
    scene.add_monitor(mw.FieldTimeMonitor("field_time", components=("Ez",), position=(0.0, 0.0, 0.16), interval=2))
    scene.add_monitor(mw.FluxTimeMonitor("flux_time", axis="z", position=position, interval=2))
    scene.add_monitor(mw.FluxMonitor("flux_dft", axis="z", position=position))

    result = mw.Simulation.fdtd(
        scene,
        frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=time_steps),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
    ).run()

    monitors = result.monitors
    assert {"field_time", "flux_time", "flux_dft"} <= set(monitors)

    field_payload = result.monitor("field_time")
    assert "t" in field_payload and "field" in field_payload
    expected_samples = len([n for n in range(0, time_steps) if n % 2 == 0])
    assert _to_numpy(field_payload["t"]).shape == (expected_samples,)
    assert _to_numpy(field_payload["field"]).shape == (expected_samples,)

    flux_payload = result.monitor("flux_time")
    assert "t" in flux_payload and "flux" in flux_payload
    assert _to_numpy(flux_payload["flux"]).shape == (expected_samples,)

    dft_payload = result.monitor("flux_dft")
    assert "flux" in dft_payload
    assert np.asarray(_to_numpy(dft_payload["flux"])).shape == ()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for native FDTD")
def test_time_monitor_truncated_on_early_shutoff():
    # When opt-in auto-shutoff breaks the loop early, the time trace and its t axis
    # must be truncated to the steps actually simulated, never padded with zeros at
    # time steps that never ran.
    scene = _build_scene(
        source_time=mw.GaussianPulse(frequency=1.0e9, fwidth=0.4e9, amplitude=100.0)
    )
    time_steps = 3000
    scene.add_monitor(mw.FieldTimeMonitor("trace", components=("Ez",), position=(0.0, 0.0, 0.16), interval=1))

    result = mw.Simulation.fdtd(
        scene,
        frequencies=[1.0e9],
        run_time=mw.TimeConfig(time_steps=time_steps),
        full_field_dft=False,
        shutoff=5e-2,
        shutoff_check_interval=50,
    ).run()

    stats = result.stats()
    assert stats["shutoff_triggered"] is True
    shutoff_step = int(stats["shutoff_step"])

    payload = result.monitor("trace")
    t = _to_numpy(payload["t"])
    field = _to_numpy(payload["field"])
    dt = float(result.solver.dt)

    assert t.shape == field.shape
    # Truncated: fewer samples than the full planned budget, and no sample past the
    # step where the loop actually stopped.
    assert 0 < t.shape[0] <= shutoff_step + 1
    assert t.shape[0] < time_steps
    assert float(t[-1]) <= (shutoff_step + 1) * dt


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for native FDTD")
def test_field_time_monitor_plane_and_multicomponent():
    # Exercise the plane region path and multi-component recording.
    scene = _build_scene()
    scene.add_monitor(
        mw.FieldTimeMonitor(
            "plane_time",
            components=("Ex", "Ez"),
            position=(0.0, 0.0, 0.16),
            size=(0.8, 0.8, 0.0),
            interval=4,
        )
    )
    time_steps = 80
    result = mw.Simulation.fdtd(
        scene,
        frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=time_steps),
        full_field_dft=False,
    ).run()

    payload = result.monitor("plane_time")
    assert payload["kind"] == "field_time"
    expected_samples = len([n for n in range(0, time_steps) if n % 4 == 0])
    # Multi-component monitors expose per-component buffers but no scalar alias.
    assert set(payload["components"]) == {"Ex", "Ez"}
    assert "field" not in payload
    for name in ("Ex", "Ez"):
        buffer = _to_numpy(payload["components"][name])
        assert buffer.ndim == 3  # (num_samples, nu, nv)
        assert buffer.shape[0] == expected_samples
    assert _to_numpy(payload["t"]).shape == (expected_samples,)
