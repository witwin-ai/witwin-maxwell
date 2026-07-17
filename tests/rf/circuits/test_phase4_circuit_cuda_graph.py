import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.circuits import (
    prepare_circuit_graph_runners,
    prepare_circuit_time_series,
)
from witwin.maxwell.fdtd.ports import apply_port_runtimes


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Circuit CUDA Graph execution requires CUDA.",
)


def _port():
    return mw.LumpedPort(
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


def _scene(circuit):
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.02, 0.02),) * 3),
        grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.none(),
        ports=(_port(),),
        circuits=(circuit,),
        device="cuda",
    )


def _driven_circuit(*, scheduled_switch):
    circuit = mw.Circuit("graph_load")
    node = circuit.node("node")
    circuit.add(mw.Resistor("R1", node, circuit.ground, 75.0))
    circuit.add(mw.Capacitor("C1", node, circuit.ground, 0.8e-12))
    circuit.add(
        mw.CurrentSource(
            "I1",
            circuit.ground,
            node,
            waveform=mw.SineWaveform(0.0, 0.01, 3.0e9),
        )
    )
    if scheduled_switch:
        circuit.add(
            mw.TimedSwitch(
                "S1",
                node,
                circuit.ground,
                (2.0e-11, 4.0e-11),
                on_resistance=120.0,
                off_resistance=1.0e8,
            )
        )
    circuit.bind_port("feed", positive=node, negative=circuit.ground)
    return circuit


def _run(*, cuda_graph, scheduled_switch):
    return mw.Simulation.fdtd(
        _scene(_driven_circuit(scheduled_switch=scheduled_switch)),
        frequency=3.0e9,
        run_time=mw.TimeConfig(time_steps=16),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
        cuda_graph=cuda_graph,
    ).run()


def test_no_rf_runtime_avoids_circuit_loop_timing_instrumentation():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.02, 0.02),) * 3),
        grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.none(),
        monitors=(
            mw.PointMonitor(name="probe", position=(0.0, 0.0, 0.0), fields=("Ez",)),
        ),
        device="cuda",
    )

    result = mw.Simulation.fdtd(
        scene,
        frequency=3.0e9,
        run_time=mw.TimeConfig(time_steps=4),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
        cuda_graph=False,
    ).run()

    assert result.stats()["steady_step_elapsed_s"] is None
    assert result.stats()["steady_steps"] is None
    assert result.stats()["steady_ms_per_step"] is None


@pytest.mark.parametrize("scheduled_switch", (False, True))
def test_fixed_builtin_schedules_use_circuit_graph_and_match_eager(scheduled_switch):
    eager = _run(cuda_graph=False, scheduled_switch=scheduled_switch)
    graphed = _run(cuda_graph=True, scheduled_switch=scheduled_switch)

    assert eager.stats()["steady_ms_per_step"] is not None
    assert eager.stats()["steady_steps"] == 16
    assert eager.stats()["circuit_cuda_graph_active"] is False
    assert graphed.stats()["circuit_cuda_graph_active"] is True, getattr(
        graphed.solver,
        "_circuit_graph_error",
        None,
    )
    eager_data = eager.circuit("graph_load")
    graph_data = graphed.circuit("graph_load")
    torch.testing.assert_close(graph_data.times, eager_data.times, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        graph_data.node_voltages,
        eager_data.node_voltages,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        graph_data.branch_currents,
        eager_data.branch_currents,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        graph_data.energy_balance,
        eager_data.energy_balance,
        rtol=0.0,
        atol=0.0,
    )
    for name in eager_data.device_powers:
        torch.testing.assert_close(
            graph_data.device_powers[name],
            eager_data.device_powers[name],
            rtol=0.0,
            atol=0.0,
        )
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        torch.testing.assert_close(
            getattr(graphed.solver, name),
            getattr(eager.solver, name),
            rtol=0.0,
            atol=0.0,
        )
    # Port phasors depend on the per-step integration tag, which a graph replay
    # cannot re-assign from inside the captured region. A mixed
    # backward-Euler/trapezoidal schedule must still observe identical spectra.
    assert set(eager.solver._circuit_runtimes[0].integration_keys) == {
        "backward_euler",
        "trapezoidal",
    }
    eager_port = eager.port("feed")
    graph_port = graphed.port("feed")
    torch.testing.assert_close(graph_port.voltage, eager_port.voltage, rtol=0.0, atol=0.0)
    torch.testing.assert_close(graph_port.current, eager_port.current, rtol=0.0, atol=0.0)


def test_purely_reactive_bound_circuit_has_a_zero_nonreactive_reduction():
    circuit = mw.Circuit("reactive")
    node = circuit.node("node")
    circuit.add(mw.Capacitor("C1", node, circuit.ground, 1.0e-12))
    circuit.bind_port("feed", positive=node, negative=circuit.ground)

    solver = mw.Simulation.fdtd(_scene(circuit), frequency=3.0e9).prepare().solver
    prepare_circuit_time_series(solver, 4)
    prepare_circuit_graph_runners(solver, True)
    for _ in range(4):
        apply_port_runtimes(solver)

    assert solver._circuit_graph_active is True
    assert torch.all(torch.isfinite(solver._circuit_runtimes[0].finalize().energy_balance))


def test_external_free_voltage_hook_skips_only_the_yee_scatter():
    circuit = mw.Circuit("owner")
    node = circuit.node("node")
    circuit.add(mw.Resistor("R1", node, circuit.ground, 50.0))
    circuit.bind_port("feed", positive=node, negative=circuit.ground)
    solver = mw.Simulation.fdtd(_scene(circuit), frequency=3.0e9).prepare().solver
    prepare_circuit_time_series(solver, 1)
    runtime = solver._circuit_runtimes[0]
    before = solver.Ez.clone()
    free_voltage = solver.Ez.new_tensor(0.25)

    currents = runtime.apply_external((free_voltage,))

    torch.testing.assert_close(solver.Ez, before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        runtime.ports[0].field.last_voltage_before,
        free_voltage,
    )
    torch.testing.assert_close(currents[0], runtime.ports[0].field.last_current)
    torch.testing.assert_close(runtime.node_samples[1, 1], runtime.ports[0].field.last_voltage)
    assert runtime.step_index == 1
    assert int(runtime.step_tensor) == 1


def test_vectorized_rc_lowering_matches_the_generic_device_path():
    circuit = mw.Circuit("rc_fast")
    first = circuit.node("first")
    second = circuit.node("second")
    circuit.add(mw.Resistor("R1", first, second, 40.0))
    circuit.add(mw.Capacitor("C1", first, circuit.ground, 0.7e-12))
    circuit.add(mw.Resistor("R2", second, circuit.ground, 65.0))
    circuit.add(mw.Capacitor("C2", second, circuit.ground, 1.1e-12))
    circuit.bind_port("feed", positive=first, negative=circuit.ground)
    fast_solver = mw.Simulation.fdtd(_scene(circuit), frequency=3.0e9).prepare().solver
    generic_solver = mw.Simulation.fdtd(_scene(circuit), frequency=3.0e9).prepare().solver
    prepare_circuit_time_series(fast_solver, 6)
    prepare_circuit_time_series(generic_solver, 6)
    fast = fast_solver._circuit_runtimes[0]
    generic = generic_solver._circuit_runtimes[0]
    assert fast.rc_fast_plan is not None
    generic.rc_fast_plan = None

    for value in (0.2, -0.1, 0.35, -0.25, 0.05, 0.0):
        fast.apply_external((fast.initial_solution.new_tensor(value),))
        generic.apply_external((generic.initial_solution.new_tensor(value),))

    fast_data = fast.finalize()
    generic_data = generic.finalize()
    torch.testing.assert_close(fast_data.node_voltages, generic_data.node_voltages)
    torch.testing.assert_close(fast_data.branch_currents, generic_data.branch_currents)
    torch.testing.assert_close(fast_data.energy_balance, generic_data.energy_balance)
    for name in fast_data.device_powers:
        torch.testing.assert_close(
            fast_data.device_powers[name],
            generic_data.device_powers[name],
        )
