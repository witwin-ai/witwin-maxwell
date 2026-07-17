import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.ports import (
    accumulate_port_observers,
    apply_port_runtimes,
    prepare_port_spectral_accumulators,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


def _port_scene(*, termination=None):
    port = mw.LumpedPort(
        name="feed",
        positive=(0.0, 0.0, 0.005),
        negative=(0.0, 0.0, -0.005),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(
            position=(0.0, 0.0, -0.0025),
            size=(0.015, 0.015, 0.0),
        ),
        reference_impedance=50.0,
        termination=termination,
    )
    return mw.Scene(
        domain=mw.Domain(
            bounds=((-0.02, 0.02), (-0.02, 0.02), (-0.02, 0.02))
        ),
        grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.none(),
        ports=(port,),
        device="cuda",
    )


def test_one_port_fdtd_run_returns_finite_device_resident_port_data():
    simulation = mw.Simulation.fdtd(
        _port_scene(),
        frequencies=(2.5e9, 3.0e9),
        excitations=mw.PortExcitation(
            "feed",
            amplitude=1.0,
            source_impedance="matched",
            source_time=mw.GaussianPulse(frequency=2.75e9, fwidth=1.0e9),
        ),
        run_time=mw.TimeConfig(time_steps=96),
        spectral_sampler=mw.SpectralSampler(window="none"),
    )

    result = simulation.run()
    data = result.port("feed")

    assert data.frequencies.shape == (2,)
    assert data.voltage.shape == data.current.shape == (2,)
    assert data.frequencies.device.type == "cuda"
    assert data.voltage.device.type == "cuda"
    assert data.available_power is not None
    assert torch.all(torch.isfinite(data.voltage))
    assert torch.all(torch.isfinite(data.current))
    assert torch.any(torch.abs(data.voltage) > 0.0)
    assert torch.any(torch.abs(data.current) > 0.0)
    assert torch.all(data.accepted_power >= 0.0)
    assert torch.all(data.available_power >= 0.0)
    assert result.stats()["num_ports"] == 1
    assert result.solver.dft_enabled is False
    assert result.fields == {}


def test_port_only_gaussian_uses_a_full_pulse_window_by_default():
    simulation = mw.Simulation.fdtd(
        _port_scene(),
        frequency=2.75e9,
        excitations=mw.PortExcitation(
            "feed",
            source_time=mw.GaussianPulse(frequency=2.75e9, fwidth=1.0e9),
        ),
        run_time=mw.TimeConfig(time_steps=96),
    )

    data = simulation.run().port("feed")

    assert data.metadata["window"] == "none"
    assert data.metadata["dft_samples"] == 96
    assert torch.all(torch.isfinite(data.voltage))


def test_cw_port_remains_stable_for_4096_steps_at_the_base_courant_limit():
    simulation = mw.Simulation.fdtd(
        _port_scene(),
        frequency=3.0e9,
        excitations=mw.PortExcitation("feed", source_time=mw.CW(3.0e9)),
        run_time=mw.TimeConfig(time_steps=4096),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
    )
    prepared = simulation.prepare()
    solver = prepared.solver
    expected_dt = 1.0 / (
        solver.c
        * (
            1.0 / solver.min_dx**2
            + 1.0 / solver.min_dy**2
            + 1.0 / solver.min_dz**2
        )
        ** 0.5
    )
    assert solver.dt == pytest.approx(expected_dt)

    data = prepared.run().port("feed")

    assert torch.all(torch.isfinite(data.voltage))
    assert torch.all(torch.isfinite(data.current))
    assert torch.all(torch.abs(data.voltage) < 2.0)
    torch.testing.assert_close(
        data.incident_power,
        data.available_power,
        rtol=1.0e-3,
        atol=1.0e-8,
    )


def test_passive_rlc_port_prepares_device_resident_auxiliary_state():
    simulation = mw.Simulation.fdtd(
        _port_scene(termination=mw.SeriesRLC(r=25.0, l=0.5e-9, c=1.0e-12)),
        frequency=3.0e9,
        run_time=mw.TimeConfig(time_steps=8),
        spectral_sampler=mw.SpectralSampler(window="none"),
    )

    prepared = simulation.prepare()
    runtime = prepared.solver._port_runtimes[0]

    assert runtime.lumped is not None
    assert runtime.lumped.inductor_current.device.type == "cuda"
    assert runtime.lumped.capacitor_voltage.device.type == "cuda"
    assert runtime.lumped.linear_indices.device.type == "cuda"


def test_scene_lumped_element_is_compiled_into_the_same_device_runtime():
    element = mw.Capacitor(
        "matching_c",
        positive=(0.0, 0.0, 0.005),
        negative=(0.0, 0.0, -0.005),
        capacitance=torch.tensor(1.0e-12, device="cuda"),
    )
    scene = mw.Scene(
        domain=mw.Domain(
            bounds=((-0.02, 0.02), (-0.02, 0.02), (-0.02, 0.02))
        ),
        grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.none(),
        lumped_elements=(element,),
        device="cuda",
    )
    prepared = mw.Simulation.fdtd(scene, frequency=3.0e9).prepare()

    runtime, field_name = prepared.solver._lumped_element_runtimes[0]

    assert field_name == "Ez"
    assert runtime.port_name == "matching_c"
    assert runtime.capacitance.device.type == "cuda"


def test_active_port_rejects_a_simultaneous_passive_termination():
    simulation = mw.Simulation.fdtd(
        _port_scene(termination=mw.SeriesRLC(r=50.0)),
        frequency=3.0e9,
        excitations=mw.PortExcitation("feed"),
    )

    with pytest.raises(ValueError, match="cannot also declare a passive termination"):
        simulation.prepare()


def test_prepare_rejects_overlapping_port_and_lumped_element_edges():
    scene = _port_scene(termination=mw.SeriesRLC(r=50.0))
    scene.add_lumped_element(
        mw.Capacitor(
            "overlap",
            positive=(0.0, 0.0, 0.005),
            negative=(0.0, 0.0, -0.005),
            capacitance=1.0e-12,
        )
    )

    with pytest.raises(ValueError, match="overlap the same Ez Yee edge"):
        mw.Simulation.fdtd(scene, frequency=3.0e9).prepare()


def test_port_step_profiler_has_no_scalar_sync_or_host_device_copy():
    solver = mw.Simulation.fdtd(
        _port_scene(),
        frequency=3.0e9,
        excitations=mw.PortExcitation("feed", source_time=mw.CW(3.0e9)),
    ).prepare().solver
    prepare_port_spectral_accumulators(solver, 64, "none")

    for _ in range(4):
        apply_port_runtimes(solver)
        accumulate_port_observers(solver)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        acc_events=True,
    ) as profile:
        for _ in range(16):
            apply_port_runtimes(solver)
            accumulate_port_observers(solver)
    torch.cuda.synchronize()

    event_names = {event.key for event in profile.key_averages()}
    assert "aten::item" not in event_names
    assert "aten::_local_scalar_dense" not in event_names
    assert not any("Memcpy HtoD" in name or "Memcpy DtoH" in name for name in event_names)
