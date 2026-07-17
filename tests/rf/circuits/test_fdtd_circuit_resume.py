import pytest
import torch

import witwin.maxwell as mw


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FDTD circuit resume requires CUDA.",
)


def _simulation(
    *,
    steps=32,
    frequency=3.0e9,
    boundary=None,
    cpml_config=None,
    include_switch=False,
    cuda_graph=False,
):
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
    circuit = mw.Circuit("pulse_interconnect")
    input_node = circuit.node("input")
    output_node = circuit.node("output")
    circuit.add(mw.Resistor("R1", input_node, output_node, 35.0))
    circuit.add(mw.Capacitor("C1", output_node, circuit.ground, 1.2e-12))
    if include_switch:
        circuit.add(
            mw.TimedSwitch(
                "S1",
                input_node,
                circuit.ground,
                (8.0e-11,),
                initially_closed=False,
                on_resistance=10.0,
                off_resistance=1.0e6,
            )
        )
    circuit.add(
        mw.CurrentSource(
            "I1",
            circuit.ground,
            input_node,
            waveform=mw.PiecewiseLinearWaveform(
                (0.0, 4.0e-11, 8.0e-11, 1.4e-10, 2.0e-10),
                (0.0, 0.0, 0.02, -0.01, 0.0),
            ),
        )
    )
    circuit.bind_port("feed", positive=output_node, negative=circuit.ground)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.02, 0.02),) * 3),
        grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.none() if boundary is None else boundary,
        ports=(port,),
        circuits=(circuit,),
        device="cuda",
    )
    return mw.Simulation.fdtd(
        scene,
        frequency=frequency,
        run_time=mw.TimeConfig(time_steps=steps),
        spectral_sampler=mw.SpectralSampler(window="none"),
        cpml_config={} if cpml_config is None else cpml_config,
        cuda_graph=cuda_graph,
    )


def _assert_result_close(actual, expected):
    for name in ("Ex", "Ey", "Ez"):
        torch.testing.assert_close(
            actual.field(name),
            expected.field(name),
            rtol=1.0e-6,
            atol=2.0e-8,
        )
    actual_port = actual.port("feed")
    expected_port = expected.port("feed")
    torch.testing.assert_close(
        actual_port.voltage,
        expected_port.voltage,
        rtol=1.0e-6,
        atol=1.0e-9,
    )
    torch.testing.assert_close(
        actual_port.current,
        expected_port.current,
        rtol=1.0e-6,
        atol=1.0e-9,
    )
    actual_circuit = actual.circuit("pulse_interconnect")
    expected_circuit = expected.circuit("pulse_interconnect")
    for actual_tensor, expected_tensor in (
        (actual_circuit.times, expected_circuit.times),
        (actual_circuit.node_voltages, expected_circuit.node_voltages),
        (actual_circuit.branch_currents, expected_circuit.branch_currents),
        (actual_circuit.energy_balance, expected_circuit.energy_balance),
        (
            actual_circuit.diagnostics["port_powers"],
            expected_circuit.diagnostics["port_powers"],
        ),
    ):
        torch.testing.assert_close(
            actual_tensor,
            expected_tensor,
            rtol=1.0e-6,
            atol=2.0e-9,
        )
    for name in expected_circuit.device_powers:
        torch.testing.assert_close(
            actual_circuit.device_powers[name],
            expected_circuit.device_powers[name],
            rtol=1.0e-6,
            atol=2.0e-9,
        )


def test_checkpoint_resume_matches_uninterrupted_result_after_pwl_breakpoint(tmp_path):
    expected = _simulation().run()
    checkpoint = _simulation().prepare().run_until(13)
    path = tmp_path / "pulse-interconnect.pt"
    checkpoint.save(path)

    restored = mw.FDTDResumeCheckpoint.load(path, map_location="cpu")
    actual = _simulation().prepare().run(resume_from=restored)

    assert restored.step == 13
    assert restored.total_steps == 32
    _assert_result_close(actual, expected)


def test_resume_rejects_a_different_total_step_schedule_before_mutating_fields():
    checkpoint = _simulation(steps=32).prepare().run_until(13)
    prepared = _simulation(steps=31).prepare()
    before = prepared.solver.Ez.clone()

    with pytest.raises(ValueError, match="planned 32 steps"):
        prepared.run(resume_from=checkpoint)

    torch.testing.assert_close(prepared.solver.Ez, before, rtol=0.0, atol=0.0)
    assert prepared.solver._port_runtimes[0].accumulator is None
    assert prepared.solver._circuit_runtimes[0].sample_times is None
    assert prepared._consumed is False


def test_resume_rejects_changed_frequency_configuration_before_mutation():
    checkpoint = _simulation(frequency=3.0e9).prepare().run_until(13)
    prepared = _simulation(frequency=2.0e9).prepare()

    with pytest.raises(ValueError, match="does not match the prepared solver layout"):
        prepared.run(resume_from=checkpoint)

    assert prepared.solver._port_runtimes[0].accumulator is None
    assert prepared.solver._circuit_runtimes[0].sample_times is None
    assert prepared._consumed is False


def test_cpml_slab_checkpoint_resume_restores_compressed_memory_exactly():
    kwargs = {
        "boundary": mw.BoundarySpec.pml(num_layers=3),
        "cpml_config": {"memory_mode": "slab"},
    }
    expected = _simulation(**kwargs).run()
    prepared = _simulation(**kwargs).prepare()
    assert prepared.solver._cpml_memory_mode == "slab"
    checkpoint = prepared.run_until(11)

    actual = _simulation(**kwargs).prepare().run(resume_from=checkpoint)

    _assert_result_close(actual, expected)


def test_resume_crosses_a_timed_switch_breakpoint_without_schedule_drift():
    expected = _simulation(include_switch=True).run()
    checkpoint = _simulation(include_switch=True).prepare().run_until(5)

    actual = _simulation(include_switch=True).prepare().run(resume_from=checkpoint)

    _assert_result_close(actual, expected)
    assert actual.circuit("pulse_interconnect").diagnostics[
        "factorization_count"
    ] > 1


def test_cuda_graph_resume_preserves_circuit_state_tensor_identity():
    kwargs = {"include_switch": True, "cuda_graph": True}
    expected = _simulation(**kwargs).run()
    checkpoint = _simulation(**kwargs).prepare().run_until(5)

    actual = _simulation(**kwargs).prepare().run(resume_from=checkpoint)

    assert actual.stats()["circuit_cuda_graph_active"] is True
    _assert_result_close(actual, expected)
