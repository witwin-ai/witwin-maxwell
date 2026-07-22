import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.networks import apply_network_runtimes
from witwin.maxwell.fdtd.ports import (
    _accumulate_embedded_port_observers_gpu,
    accumulate_port_observers,
    apply_port_runtimes,
    complete_port_observer_graph,
    make_port_observer_runner,
    prepare_port_spectral_accumulators,
)
from witwin.maxwell.fdtd.runtime.stepping import (
    _field_update_block,
    _make_full_embedded_step_runner,
    apply_mur_boundaries,
    enforce_pec_boundaries,
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


def test_embedded_network_observer_cuda_graph_matches_eager() -> None:
    eager = _prepared_embedded_solver()
    captured = _prepared_embedded_solver()
    runner = make_port_observer_runner(captured, use_cuda_graph=True)
    assert captured._port_observer_graph_active

    for _ in range(16):
        apply_network_runtimes(eager)
        apply_network_runtimes(captured)
        accumulate_port_observers(eager)
        runner()
    complete_port_observer_graph(captured, 16)
    torch.cuda.synchronize()

    eager_port = eager._port_runtimes[0]
    captured_port = captured._port_runtimes[0]
    torch.testing.assert_close(
        captured_port.accumulator._voltage_sum,
        eager_port.accumulator._voltage_sum,
    )
    torch.testing.assert_close(
        captured_port.accumulator._current_sum,
        eager_port.accumulator._current_sum,
    )
    torch.testing.assert_close(
        captured_port.accumulator._window_weight_sum,
        eager_port.accumulator._window_weight_sum,
    )
    torch.testing.assert_close(captured_port.electric_time, eager_port.electric_time)
    torch.testing.assert_close(captured_port.magnetic_time, eager_port.magnetic_time)


def test_full_embedded_step_cuda_graph_restores_and_matches_nonzero_eager() -> None:
    eager = _prepared_embedded_solver()
    captured = _prepared_embedded_solver()
    field_names = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    for index, name in enumerate(field_names, start=1):
        eager_field = getattr(eager, name)
        eager_field.fill_(index * 1.0e-5)
        getattr(captured, name).copy_(eager_field)
    for solver in (eager, captured):
        runtime = solver._network_runtimes[0]
        runtime.state.fill_(2.0e-4)
        runtime.B.fill_(0.05)
        runtime.C.fill_(0.1)

    initial = {
        name: getattr(captured, name).clone()
        for name in field_names
    }
    initial["network_state"] = captured._network_runtimes[0].state.clone()
    runner = _make_full_embedded_step_runner(captured, use_cuda_graph=True)
    assert runner is not None
    assert captured._cuda_graph_active
    assert captured._network_cuda_graph_active
    assert captured._port_observer_graph_active
    for name, value in initial.items():
        actual = (
            captured._network_runtimes[0].state
            if name == "network_state"
            else getattr(captured, name)
        )
        torch.testing.assert_close(actual, value)

    def eager_step() -> None:
        _field_update_block(eager, 0.0)
        apply_port_runtimes(eager)
        apply_network_runtimes(eager)
        eager._apply_dispersive_corrections()
        if not eager.tfsf_enabled:
            enforce_pec_boundaries(eager)
        apply_mur_boundaries(eager)
        _accumulate_embedded_port_observers_gpu(eager)

    for _ in range(8):
        eager_step()
        runner()
    complete_port_observer_graph(eager, 8)
    complete_port_observer_graph(captured, 8)
    torch.cuda.synchronize()

    for name in field_names:
        torch.testing.assert_close(getattr(captured, name), getattr(eager, name))
    eager_network = eager._network_runtimes[0]
    captured_network = captured._network_runtimes[0]
    for name in ("state", "free_voltage", "network_voltage", "branch_current"):
        torch.testing.assert_close(
            getattr(captured_network, name),
            getattr(eager_network, name),
            rtol=2.0e-5,
            atol=2.0e-7,
        )
    eager_port = eager._port_runtimes[0]
    captured_port = captured._port_runtimes[0]
    torch.testing.assert_close(
        captured_port.accumulator._voltage_sum,
        eager_port.accumulator._voltage_sum,
    )
    torch.testing.assert_close(
        captured_port.accumulator._current_sum,
        eager_port.accumulator._current_sum,
    )


def test_full_embedded_step_graph_rejects_nested_surface_state() -> None:
    solver = _prepared_embedded_solver()
    solver.surface_impedance_enabled = True
    assert _make_full_embedded_step_runner(solver, use_cuda_graph=True) is None
