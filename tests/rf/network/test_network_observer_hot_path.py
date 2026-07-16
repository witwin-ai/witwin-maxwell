import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.networks import apply_network_runtimes
from witwin.maxwell.fdtd.ports import (
    accumulate_port_observers,
    prepare_port_spectral_accumulators,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _prepared_embedded_solver():
    frequencies = torch.tensor((2.5e9, 3.0e9), dtype=torch.float64)
    model = mw.StateSpaceNetwork(
        A=torch.tensor(((-1.0e12,),), dtype=torch.float64),
        B=torch.zeros((1, 1), dtype=torch.float64),
        C=torch.zeros((1, 1), dtype=torch.float64),
        D=torch.tensor(((0.02,),), dtype=torch.float64),
        representation="Y",
        port_order=("load",),
        passivity_margin=0.02,
    )
    network = mw.NetworkData.from_y(
        frequencies=frequencies,
        y=model.evaluate(frequencies),
        z0=50.0,
        port_names=("load",),
        metadata={"model_id": "observer-profile"},
    )
    block = mw.NetworkBlock(
        name="load_network",
        network=network,
        connections={"load": "feed"},
        fit=False,
        model=model,
    )
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
    )
    scene = mw.Scene(
        domain=mw.Domain(
            bounds=((-0.02, 0.02), (-0.02, 0.02), (-0.02, 0.02))
        ),
        grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.none(),
        ports=(port,),
        networks=(block,),
        device="cuda",
    )
    solver = mw.Simulation.fdtd(
        scene,
        frequencies=tuple(frequencies.tolist()),
        run_time=mw.TimeConfig(time_steps=32),
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).prepare().solver
    prepare_port_spectral_accumulators(solver, 32, "none")
    return solver


def _hot_path_pointers(solver) -> tuple[int, ...]:
    port = solver._port_runtimes[0]
    network = solver._network_runtimes[0]
    accumulator = port.accumulator
    assert accumulator is not None
    assert port.observer_current_buffer is not None
    assert port.voltage_phase_weights is not None
    assert port.current_phase_weights is not None
    return (
        accumulator._voltage_sum.data_ptr(),
        accumulator._current_sum.data_ptr(),
        accumulator._voltage_term.data_ptr(),
        accumulator._current_term.data_ptr(),
        port.observer_current_buffer.data_ptr(),
        port.voltage_phase_weights.data_ptr(),
        port.current_phase_weights.data_ptr(),
        network.state.data_ptr(),
        network.output_buffer.data_ptr(),
    )


def test_embedded_network_observer_has_no_step_allocation_or_host_transfer() -> None:
    solver = _prepared_embedded_solver()
    pointers = _hot_path_pointers(solver)

    for _ in range(4):
        apply_network_runtimes(solver)
        accumulate_port_observers(solver)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ),
        profile_memory=True,
        acc_events=True,
    ) as profile:
        for _ in range(8):
            apply_network_runtimes(solver)
            accumulate_port_observers(solver)
    torch.cuda.synchronize()

    events = profile.key_averages()
    event_names = {event.key for event in events}
    assert "aten::item" not in event_names
    assert "aten::_local_scalar_dense" not in event_names
    assert "aten::empty" not in event_names
    assert "aten::empty_strided" not in event_names
    assert not any("Memcpy HtoD" in name or "Memcpy DtoH" in name for name in event_names)
    assert not any(
        event.self_cpu_memory_usage
        or getattr(event, "self_device_memory_usage", 0)
        for event in events
    )
    assert pointers == _hot_path_pointers(solver)
