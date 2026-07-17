import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler import compile_mna_system
from witwin.maxwell.fdtd.circuits import prepare_circuit_time_series
from witwin.maxwell.fdtd.ports import apply_port_runtimes


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Circuit multi-port execution requires CUDA.",
)


def _port(name, x):
    return mw.LumpedPort(
        name=name,
        positive=(x, 0.0, 0.004),
        negative=(x, 0.0, -0.004),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(
            position=(x, 0.0, -0.002),
            size=(0.012, 0.012, 0.0),
        ),
        reference_impedance=50.0,
    )


def _two_port_scene(circuit):
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.024, 0.024),) * 3),
        grid=mw.GridSpec.uniform(0.004),
        boundary=mw.BoundarySpec.none(),
        ports=(_port("input", -0.008), _port("output", 0.008)),
        circuits=(circuit,),
        device="cuda",
    )


def _set_free_voltage(solver, port, value):
    field = getattr(solver, port.field_name).view(-1)
    weights = port.field.voltage_weights
    target = weights.new_tensor(value)
    edge_values = weights * (target / torch.dot(weights, weights))
    field.index_copy_(0, port.field.linear_indices, edge_values)


def test_two_bound_ports_solve_once_and_cross_port_vccs_preserves_polarity_and_kcl():
    input_resistance = 80.0
    output_resistance = 65.0
    transconductance = 0.012
    circuit = mw.Circuit("controlled_multiport")
    input_node = circuit.node("input_node")
    output_node = circuit.node("output_node")
    circuit.add(mw.Resistor("Rin", input_node, circuit.ground, input_resistance))
    circuit.add(mw.Resistor("Rout", output_node, circuit.ground, output_resistance))
    circuit.add(
        mw.VoltageControlledCurrentSource(
            "Gcross",
            output_node,
            circuit.ground,
            input_node,
            circuit.ground,
            transconductance,
        )
    )
    circuit.bind_port("input", positive=input_node, negative=circuit.ground)
    circuit.bind_port("output", positive=output_node, negative=circuit.ground)

    solver = mw.Simulation.fdtd(
        _two_port_scene(circuit),
        frequency=3.0e9,
    ).prepare().solver
    prepare_circuit_time_series(solver, 1)
    runtime = solver._circuit_runtimes[0]
    input_port, output_port = runtime.ports
    solver.Ez.zero_()
    _set_free_voltage(solver, input_port, 0.2)
    _set_free_voltage(solver, output_port, 0.0)

    apply_port_runtimes(solver)

    input_conductance = input_port.conductance(input_port.last_integration)
    output_conductance = output_port.conductance(output_port.last_integration)
    zero = input_conductance.new_zeros(())
    matrix = torch.stack(
        (
            torch.stack((input_conductance + 1.0 / input_resistance, zero)),
            torch.stack((zero + transconductance, output_conductance + 1.0 / output_resistance)),
        )
    )
    rhs = torch.stack((input_conductance * 0.2, zero))
    expected_voltage = torch.linalg.solve(matrix, rhs)
    actual_voltage = runtime.node_samples[1, 1:]
    torch.testing.assert_close(actual_voltage, expected_voltage, rtol=2.0e-6, atol=2.0e-9)

    branch_index = {name: index for index, name in enumerate(runtime.physical_branch_names)}
    controlled_current = runtime.branch_samples[1, branch_index["Gcross"]]
    resistor_current = runtime.branch_samples[1, branch_index["Rout"]]
    output_port_current = output_port.field.last_current
    assert bool(actual_voltage[0] > 0.0)
    assert bool(actual_voltage[1] < 0.0)
    assert bool(controlled_current > 0.0)
    torch.testing.assert_close(
        controlled_current,
        actual_voltage[0] * transconductance,
        rtol=2.0e-6,
        atol=2.0e-11,
    )
    torch.testing.assert_close(
        resistor_current + controlled_current,
        output_port_current,
        rtol=2.0e-6,
        atol=2.0e-11,
    )
    assert len(runtime.factor_cache) == 1
    assert runtime.finalize().diagnostics["bound_ports"] == ("input", "output")


def _spice_transient(source, *, dt, steps):
    circuit = mw.parse_spice(
        f"{source}\nRload out 0 1k\n.end",
        name="waveform",
    )
    system = compile_mna_system(
        circuit,
        dt=torch.tensor(dt, dtype=torch.float64),
        device="cuda",
        dtype=torch.float64,
    )
    return system.transient(steps)


def test_spice_pulse_runtime_covers_edges_periodicity_and_breakpoints():
    data = _spice_transient(
        "Vdrive out 0 PULSE(0 1 0.5 0.5 0.5 0.5 2)",
        dt=0.25,
        steps=10,
    )
    expected = data.times.new_tensor((0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0))

    torch.testing.assert_close(data.node_voltage("out"), expected, rtol=0.0, atol=1.0e-12)
    assert data.diagnostics["local_backward_euler_steps"] == (1, 2, 4, 6, 8, 10)


def test_spice_sine_runtime_covers_delay_offset_damping_and_phase():
    data = _spice_transient(
        "Vdrive out 0 SIN(0.2 1 0.5 0.5 0.1 30)",
        dt=0.25,
        steps=8,
    )
    tau = torch.clamp_min(data.times - 0.5, 0.0)
    expected = 0.2 + torch.exp(-0.1 * tau) * torch.sin(math.pi * tau + math.pi / 6.0)
    expected = torch.where(data.times >= 0.5, expected, expected.new_full((), 0.2))

    torch.testing.assert_close(data.node_voltage("out"), expected, rtol=1.0e-7, atol=1.0e-9)
    assert data.diagnostics["local_backward_euler_steps"] == (1, 2)


def test_spice_pwl_runtime_covers_knots_interpolation_and_endpoint_holds():
    data = _spice_transient(
        "Vdrive out 0 PWL(0 0 0.5 0.5 1 -0.5 1.5 0)",
        dt=0.25,
        steps=8,
    )
    expected = data.times.new_tensor((0.0, 0.25, 0.5, 0.0, -0.5, -0.25, 0.0, 0.0, 0.0))

    torch.testing.assert_close(data.node_voltage("out"), expected, rtol=0.0, atol=1.0e-12)
    assert data.diagnostics["local_backward_euler_steps"] == (1, 2, 4, 6)


def _matching_filter_column(active_port, frequencies, *, dt, steps, sample_start):
    z0 = 50.0
    cutoff = 3.0e9
    inductance = z0 / (2.0 * math.pi * cutoff)
    capacitance = 2.0 / (z0 * 2.0 * math.pi * cutoff)
    circuit = mw.Circuit(f"matching_{active_port}")
    port1 = circuit.node("port1")
    middle = circuit.node("middle")
    port2 = circuit.node("port2")
    circuit.add(mw.Resistor("Rport1", port1, circuit.ground, z0))
    circuit.add(mw.Resistor("Rport2", port2, circuit.ground, z0))
    circuit.add(mw.Inductor("L1", port1, middle, inductance))
    circuit.add(mw.Capacitor("C1", middle, circuit.ground, capacitance))
    circuit.add(mw.Inductor("L2", middle, port2, inductance))
    source_node = port1 if active_port == 0 else port2
    for index, frequency in enumerate(frequencies):
        circuit.add(
            mw.CurrentSource(
                f"I{index}",
                circuit.ground,
                source_node,
                waveform=mw.SineWaveform(0.0, 2.0 / z0, frequency),
            )
        )
    data = compile_mna_system(
        circuit,
        dt=torch.tensor(dt, dtype=torch.float64),
        device="cuda",
        dtype=torch.float64,
    ).transient(steps)

    times = data.times[sample_start:steps]
    frequencies_tensor = times.new_tensor(frequencies)
    phase = torch.exp(
        -2j * math.pi * times[:, None] * frequencies_tensor[None, :]
    )

    def phasor(values):
        samples = values[sample_start:steps, None].to(torch.complex128)
        return 2.0 * torch.sum(samples * phase, dim=0) / times.numel()

    active_name = "port1" if active_port == 0 else "port2"
    inactive_name = "port2" if active_port == 0 else "port1"
    active_voltage = phasor(data.node_voltage(active_name))
    inactive_voltage = phasor(data.node_voltage(inactive_name))
    source_phasor = torch.stack(
        tuple(
            phasor(data.branch_current(f"I{index}"))[index]
            for index in range(len(frequencies))
        )
    )
    incident_voltage = 0.5 * z0 * source_phasor
    return active_voltage / incident_voltage - 1.0, inactive_voltage / incident_voltage


def _analytic_matching_filter_s(frequencies):
    z0 = 50.0
    cutoff = 3.0e9
    inductance = z0 / (2.0 * math.pi * cutoff)
    capacitance = 2.0 / (z0 * 2.0 * math.pi * cutoff)
    omega = 2.0 * math.pi * torch.as_tensor(frequencies, dtype=torch.float64)
    series = 1j * omega * inductance
    shunt = 1j * omega * capacitance
    a = 1.0 + series * shunt
    b = 2.0 * series + series.square() * shunt
    c = shunt
    d = a
    denominator = a + b / z0 + c * z0 + d
    determinant = a * d - b * c
    result = torch.empty((len(frequencies), 2, 2), dtype=torch.complex128)
    result[:, 0, 0] = (a + b / z0 - c * z0 - d) / denominator
    result[:, 1, 0] = 2.0 / denominator
    result[:, 0, 1] = 2.0 * determinant / denominator
    result[:, 1, 1] = (-a + b / z0 - c * z0 + d) / denominator
    return result


def test_two_port_matching_filter_transient_s_matches_independent_abcd_reference():
    frequencies = (1.5e9, 3.0e9, 4.5e9)
    dt = 1.0 / (64.0 * max(frequencies))
    steps_per_base_period = round(1.0 / (frequencies[0] * dt))
    steps = 20 * steps_per_base_period
    sample_start = 10 * steps_per_base_period
    actual = torch.empty((len(frequencies), 2, 2), device="cuda", dtype=torch.complex128)
    for active_port in (0, 1):
        reflection, transmission = _matching_filter_column(
            active_port,
            frequencies,
            dt=dt,
            steps=steps,
            sample_start=sample_start,
        )
        actual[:, active_port, active_port] = reflection
        actual[:, 1 - active_port, active_port] = transmission

    reference = _analytic_matching_filter_s(frequencies).to(device="cuda")
    assert bool(torch.max(torch.abs(actual - reference)) < 0.02)


def _three_port_scene(*, circuit=None):
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.024, 0.024),) * 3),
        grid=mw.GridSpec.uniform(0.004),
        boundary=mw.BoundarySpec.none(),
        ports=(
            _port("feed", -0.008),
            _port("load1", 0.0),
            _port("load2", 0.008),
        ),
        circuits=() if circuit is None else (circuit,),
        device="cuda",
    )


def test_two_bound_port_fdtd_frequency_response_matches_loaded_em_reference():
    frequency = 2.5e9
    steps = 128
    source_time = mw.GaussianPulse(frequency=frequency, fwidth=1.0e9)
    run_time = mw.TimeConfig(time_steps=steps)
    sampler = mw.SpectralSampler(window="none")
    bare = mw.Simulation.fdtd(
        _three_port_scene(),
        frequency=frequency,
        excitations=mw.PortSweep(source_time=source_time),
        run_time=run_time,
        spectral_sampler=sampler,
    ).run()
    em_impedance = bare.network.to_z()[0]

    first_resistance = 75.0
    second_resistance = 100.0
    circuit = mw.Circuit("two_port_load")
    first = circuit.node("first")
    second = circuit.node("second")
    circuit.add(mw.Resistor("R1", first, circuit.ground, first_resistance))
    circuit.add(mw.Resistor("R2", second, circuit.ground, second_resistance))
    circuit.bind_port("load1", positive=first, negative=circuit.ground)
    circuit.bind_port("load2", positive=second, negative=circuit.ground)
    coupled = mw.Simulation.fdtd(
        _three_port_scene(circuit=circuit),
        frequency=frequency,
        excitations=mw.PortExcitation("feed", source_time=source_time),
        run_time=run_time,
        spectral_sampler=sampler,
    ).run()

    load_admittance = torch.diag(
        torch.tensor(
            (1.0 / first_resistance, 1.0 / second_resistance),
            device="cuda",
            dtype=torch.complex128,
        )
    )
    z00 = em_impedance[0, 0]
    z0l = em_impedance[0, 1:]
    zl0 = em_impedance[1:, 0]
    zll = em_impedance[1:, 1:]
    loaded_voltage = torch.linalg.solve(
        torch.eye(2, device="cuda", dtype=torch.complex128)
        + zll @ load_admittance,
        zl0,
    )
    input_impedance = z00 - z0l @ load_admittance @ loaded_voltage
    expected_s11 = (input_impedance - 50.0) / (input_impedance + 50.0)
    actual_s11 = (coupled.port("feed").b / coupled.port("feed").a)[0]

    assert coupled.circuit("two_port_load").diagnostics["bound_ports"] == (
        "load1",
        "load2",
    )
    assert bool(torch.abs(actual_s11 - expected_s11) < 0.02)


def _orientation_result(*, reversed_orientation):
    positive = (0.0, 0.0, -0.005 if reversed_orientation else 0.005)
    negative = (0.0, 0.0, 0.005 if reversed_orientation else -0.005)
    port = mw.LumpedPort(
        name="feed",
        positive=positive,
        negative=negative,
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(
            position=(0.0, 0.0, -0.0025),
            size=(0.015, 0.015, 0.0),
        ),
        reference_impedance=50.0,
    )
    circuit = mw.Circuit("orientation")
    node = circuit.node("node")
    circuit.add(mw.Resistor("R1", node, circuit.ground, 50.0))
    circuit.add(
        mw.CurrentSource(
            "I1",
            circuit.ground,
            node,
            waveform=mw.SineWaveform(0.0, 0.01, 3.0e9),
        )
    )
    circuit.bind_port(
        "feed",
        positive=circuit.ground if reversed_orientation else node,
        negative=node if reversed_orientation else circuit.ground,
    )
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.02, 0.02),) * 3),
        grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.none(),
        ports=(port,),
        circuits=(circuit,),
        device="cuda",
    )
    return mw.Simulation.fdtd(
        scene,
        frequency=3.0e9,
        run_time=mw.TimeConfig(time_steps=16),
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).run()


def test_reversing_em_and_circuit_terminals_flips_vi_but_preserves_power_and_s():
    forward = _orientation_result(reversed_orientation=False)
    reverse = _orientation_result(reversed_orientation=True)
    forward_port = forward.port("feed")
    reverse_port = reverse.port("feed")
    data = forward.circuit("orientation")
    times = data.times[1:]
    phase = torch.exp(2j * torch.pi * 3.0e9 * times)

    def phasor(values):
        return 2.0 * torch.sum(values[1:].to(torch.complex128) * phase) / times.numel()

    expected_voltage = phasor(data.node_voltage("node"))
    expected_current = phasor(
        data.branch_current("I1") - data.branch_current("R1")
    )
    expected_a = (expected_voltage + 50.0 * expected_current) / (
        2.0 * math.sqrt(2.0 * 50.0)
    )
    expected_b = (expected_voltage - 50.0 * expected_current) / (
        2.0 * math.sqrt(2.0 * 50.0)
    )

    torch.testing.assert_close(forward_port.voltage[0], expected_voltage)
    torch.testing.assert_close(forward_port.current[0], expected_current)
    torch.testing.assert_close(forward_port.a[0], expected_a)
    torch.testing.assert_close(forward_port.b[0], expected_b)
    torch.testing.assert_close(reverse_port.voltage, -forward_port.voltage)
    torch.testing.assert_close(reverse_port.current, -forward_port.current)
    torch.testing.assert_close(
        reverse_port.voltage * torch.conj(reverse_port.current),
        forward_port.voltage * torch.conj(forward_port.current),
    )
    torch.testing.assert_close(
        reverse_port.b / reverse_port.a,
        forward_port.b / forward_port.a,
    )
