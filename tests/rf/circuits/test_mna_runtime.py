import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler import compile_mna_system


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Phase 1 MNA requires CUDA.")


def _system(circuit, dt=1.0e-3):
    return compile_mna_system(
        circuit,
        dt=torch.as_tensor(dt, dtype=torch.float64),
        device="cuda",
        dtype=torch.float64,
    )


def test_dc_voltage_divider_matches_hand_solution_on_cuda_double():
    circuit = mw.Circuit("divider")
    source = circuit.node("source")
    output = circuit.node("output")
    circuit.add(mw.VoltageSource("V1", source, circuit.ground, 3.7))
    circuit.add(mw.Resistor("R1", source, output, 2.3))
    circuit.add(mw.Resistor("R2", output, circuit.ground, 4.9))

    system = _system(circuit)
    matrix, rhs = system.assemble_dc()
    solution, condition = system.solve_dc()

    expected_voltage = 3.7 * 4.9 / (2.3 + 4.9)
    expected_current = -3.7 / (2.3 + 4.9)
    assert matrix.device.type == rhs.device.type == solution.device.type == "cuda"
    assert matrix.dtype == rhs.dtype == solution.dtype == torch.float64
    torch.testing.assert_close(solution[1], torch.tensor(expected_voltage, device="cuda", dtype=torch.float64))
    torch.testing.assert_close(solution[2], torch.tensor(expected_current, device="cuda", dtype=torch.float64))
    torch.testing.assert_close(matrix @ solution, rhs, rtol=1.0e-12, atol=1.0e-14)
    assert bool(torch.isfinite(condition))


@pytest.mark.parametrize(
    ("source_type", "expected"),
    (
        (mw.VoltageControlledVoltageSource, 6.0),
        (mw.VoltageControlledCurrentSource, -4.0),
        (mw.CurrentControlledCurrentSource, 6.0),
        (mw.CurrentControlledVoltageSource, -2.5),
    ),
)
def test_controlled_source_dc_stamps_match_hand_reference(source_type, expected):
    circuit = mw.Circuit("controlled")
    control = circuit.node("control")
    output = circuit.node("output")
    circuit.add(mw.VoltageSource("V1", control, circuit.ground, 2.0 if source_type in (mw.VoltageControlledVoltageSource, mw.VoltageControlledCurrentSource) else 1.0))
    circuit.add(mw.Resistor("Rcontrol", control, circuit.ground, 2.0))
    if source_type is mw.VoltageControlledVoltageSource:
        circuit.add(source_type("E1", output, circuit.ground, control, circuit.ground, 3.0))
    elif source_type is mw.VoltageControlledCurrentSource:
        circuit.add(source_type("G1", output, circuit.ground, control, circuit.ground, 0.5))
    elif source_type is mw.CurrentControlledCurrentSource:
        circuit.add(source_type("F1", output, circuit.ground, "V1", 3.0))
    else:
        circuit.add(source_type("H1", output, circuit.ground, "V1", 5.0))
    circuit.add(mw.Resistor("Rout", output, circuit.ground, 4.0))

    solution, _ = _system(circuit).solve_dc()

    torch.testing.assert_close(
        solution[1],
        torch.tensor(expected, device="cuda", dtype=torch.float64),
        rtol=1.0e-12,
        atol=1.0e-14,
    )


def _rc_sine(integration, dt, final_time=0.4):
    tau = 0.1
    frequency = 0.7
    circuit = mw.Circuit("rc", config=mw.MNAConfig(integration=integration))
    source = circuit.node("source")
    output = circuit.node("output")
    circuit.add(
        mw.VoltageSource(
            "V1",
            source,
            circuit.ground,
            waveform=mw.SineWaveform(0.0, 1.0, frequency),
        )
    )
    circuit.add(mw.Resistor("R1", source, output, 2.0))
    circuit.add(mw.Capacitor("C1", output, circuit.ground, tau / 2.0))
    data = _system(circuit, dt=dt).transient(round(final_time / dt))
    omega = 2.0 * math.pi * frequency
    times = data.times
    reference = (
        torch.sin(omega * times)
        - omega * tau * torch.cos(omega * times)
        + omega * tau * torch.exp(-times / tau)
    ) / (1.0 + (omega * tau) ** 2)
    return data, torch.max(torch.abs(data.node_voltage("output") - reference))


def test_rc_transient_matches_analytic_reference_and_circuit_data_contract():
    data, error = _rc_sine("trapezoidal", 5.0e-4)

    assert data.times.device.type == data.node_voltages.device.type == "cuda"
    assert data.times.dtype == data.node_voltages.dtype == torch.float64
    assert data.node_names == ("0", "source", "output")
    assert data.branch_names == ("V1", "R1", "C1")
    assert torch.count_nonzero(data.node_voltage("0")) == 0
    assert float(error) < 1.0e-4
    assert data.diagnostics["transient_factorization_count"] == 2
    assert data.diagnostics["factorization_count"] == 3
    assert data.diagnostics["local_backward_euler_steps"] == (1,)
    assert float(torch.max(torch.abs(data.energy_balance))) < 2.0e-8


@pytest.mark.parametrize(
    ("integration", "minimum_order", "maximum_order"),
    (("backward_euler", 0.85, 1.15), ("trapezoidal", 1.8, 2.2)),
)
def test_rc_time_discretization_has_expected_convergence_order(integration, minimum_order, maximum_order):
    errors = []
    for dt in (0.004, 0.002, 0.001):
        _, error = _rc_sine(integration, dt)
        errors.append(float(error))
    orders = [math.log2(errors[index] / errors[index + 1]) for index in range(2)]

    assert errors[0] > errors[1] > errors[2]
    assert all(minimum_order <= order <= maximum_order for order in orders)


def test_trapezoidal_discrete_energy_residual_converges_at_second_order():
    errors = []
    for dt in (0.004, 0.002, 0.001):
        data, _ = _rc_sine("trapezoidal", dt)
        errors.append(float(torch.max(torch.abs(data.energy_balance))))
    orders = [math.log2(errors[index] / errors[index + 1]) for index in range(2)]

    assert errors[0] > errors[1] > errors[2]
    assert all(1.8 <= order <= 2.2 for order in orders)


def _unit_step(delay, duration):
    return mw.PulseWaveform(
        initial=0.0,
        pulsed=1.0,
        delay=delay,
        rise=0.0,
        fall=0.0,
        width=duration * 2.0,
        period=0.0,
    )


@pytest.mark.parametrize("network", ("rc", "rl", "rlc"))
def test_linear_step_responses_match_analytic_reference(network):
    dt = 2.0e-4 if network != "rlc" else 1.0e-4
    final_time = 0.3 if network != "rlc" else 0.15
    circuit = mw.Circuit(network)
    source = circuit.node("source")
    circuit.add(
        mw.VoltageSource(
            "V1",
            source,
            circuit.ground,
            waveform=_unit_step(0.0, final_time),
        )
    )
    if network == "rc":
        output = circuit.node("output")
        resistance = 2.0
        capacitance = 0.05
        circuit.add(mw.Resistor("R1", source, output, resistance))
        circuit.add(mw.Capacitor("C1", output, circuit.ground, capacitance))
    elif network == "rl":
        output = circuit.node("output")
        resistance = 2.0
        inductance = 0.2
        circuit.add(mw.Resistor("R1", source, output, resistance))
        circuit.add(mw.Inductor("L1", output, circuit.ground, inductance))
    else:
        intermediate = circuit.node("intermediate")
        output = circuit.node("output")
        resistance = 1.0
        inductance = 0.02
        capacitance = 0.02
        circuit.add(mw.Resistor("R1", source, intermediate, resistance))
        circuit.add(mw.Inductor("L1", intermediate, output, inductance))
        circuit.add(mw.Capacitor("C1", output, circuit.ground, capacitance))

    data = _system(circuit, dt=dt).transient(round(final_time / dt))
    active = torch.arange(data.times.numel(), device="cuda") > 0
    elapsed = data.times
    if network == "rc":
        reference = 1.0 - torch.exp(-elapsed / (resistance * capacitance))
        actual = data.node_voltage("output")
    elif network == "rl":
        reference = (1.0 - torch.exp(-elapsed * resistance / inductance)) / resistance
        actual = data.branch_current("L1")
    else:
        alpha = resistance / (2.0 * inductance)
        omega_d = math.sqrt(1.0 / (inductance * capacitance) - alpha**2)
        reference = 1.0 - torch.exp(-alpha * elapsed) * (
            torch.cos(omega_d * elapsed) + alpha / omega_d * torch.sin(omega_d * elapsed)
        )
        actual = data.node_voltage("output")
    reference = torch.where(active, reference, torch.zeros_like(reference))

    relative_linf = torch.max(torch.abs(actual - reference)) / torch.max(torch.abs(reference))
    assert float(relative_linf) < 1.0e-4


def test_transformer_transient_matches_independent_state_space_reference():
    dt = 1.0e-5
    final_time = 0.01
    first_inductance = 2.0e-3
    second_inductance = 3.0e-3
    coupling = -0.3
    first_resistance = 1.0
    second_resistance = 1.5
    mutual = coupling * math.sqrt(first_inductance * second_inductance)
    circuit = mw.Circuit("transformer_reference")
    source = circuit.node("source")
    first = circuit.node("first")
    second = circuit.node("second")
    circuit.add(
        mw.VoltageSource(
            "V1",
            source,
            circuit.ground,
            waveform=_unit_step(0.0, final_time),
        )
    )
    circuit.add(mw.Resistor("R1", source, first, first_resistance))
    circuit.add(mw.Inductor("L1", first, circuit.ground, first_inductance))
    circuit.add(mw.Resistor("R2", second, circuit.ground, second_resistance))
    circuit.add(mw.Inductor("L2", second, circuit.ground, second_inductance))
    circuit.add(mw.MutualInductor("K1", "L1", "L2", coupling))

    data = _system(circuit, dt=dt).transient(round(final_time / dt))
    inductance_matrix = np.array(
        [[first_inductance, mutual], [mutual, second_inductance]],
        dtype=np.float64,
    )
    dynamics = -np.linalg.solve(
        inductance_matrix,
        np.diag([first_resistance, second_resistance]),
    )
    forcing = np.linalg.solve(inductance_matrix, np.array([1.0, 0.0]))
    steady = -np.linalg.solve(dynamics, forcing)
    eigenvalues, eigenvectors = np.linalg.eig(dynamics)
    inverse_eigenvectors = np.linalg.inv(eigenvectors)
    elapsed = data.times.detach().cpu().numpy()
    reference = np.stack(
        [
            steady - eigenvectors @ (np.exp(eigenvalues * time) * (inverse_eigenvectors @ steady))
            for time in elapsed
        ]
    ).real
    reference[0] = 0.0
    actual = torch.stack((data.branch_current("L1"), data.branch_current("L2")), dim=1)

    np.testing.assert_allclose(
        actual.detach().cpu().numpy(),
        reference,
        rtol=0.0,
        atol=1.0e-4,
    )
    relative_linf = np.max(np.abs(actual.detach().cpu().numpy() - reference)) / np.max(
        np.abs(reference)
    )
    assert relative_linf < 1.0e-4


def test_vccs_transient_matches_analytic_reference():
    dt = 5.0e-4
    final_time = 0.4
    frequency = 0.7
    resistance = 2.0
    capacitance = 0.05
    transconductance = 0.4
    circuit = mw.Circuit("controlled_transient")
    control = circuit.node("control")
    output = circuit.node("output")
    circuit.add(
        mw.VoltageSource(
            "V1",
            control,
            circuit.ground,
            voltage=0.0,
            waveform=mw.SineWaveform(0.0, 1.0, frequency),
        )
    )
    circuit.add(
        mw.VoltageControlledCurrentSource(
            "G1",
            output,
            circuit.ground,
            control,
            circuit.ground,
            transconductance,
        )
    )
    circuit.add(mw.Resistor("R1", output, circuit.ground, resistance))
    circuit.add(mw.Capacitor("C1", output, circuit.ground, capacitance))

    data = _system(circuit, dt=dt).transient(round(final_time / dt))
    omega = 2.0 * math.pi * frequency
    decay = 1.0 / (resistance * capacitance)
    forcing = transconductance / capacitance
    reference = -forcing * (
        decay * torch.sin(omega * data.times)
        - omega * torch.cos(omega * data.times)
        + omega * torch.exp(-decay * data.times)
    ) / (decay**2 + omega**2)
    error = torch.max(torch.abs(data.node_voltage("output") - reference))

    assert float(error / torch.max(torch.abs(reference))) < 1.0e-4


def test_zero_and_constrained_initialization_create_consistent_dae_state():
    zero_circuit = mw.Circuit("zero_init", config=mw.MNAConfig(initialization="zero"))
    source = zero_circuit.node("source")
    output = zero_circuit.node("output")
    zero_circuit.add(mw.VoltageSource("V1", source, zero_circuit.ground, 1.0))
    zero_circuit.add(mw.Resistor("R1", source, output, 1.0))
    zero_circuit.add(mw.Capacitor("C1", output, zero_circuit.ground, 0.1))
    zero_data = _system(zero_circuit).transient(2)

    torch.testing.assert_close(zero_data.node_voltage("source")[0], zero_data.times.new_tensor(1.0))
    torch.testing.assert_close(zero_data.node_voltage("output")[0], zero_data.times.new_tensor(0.0))
    assert zero_data.diagnostics["initial_factorization_count"] == 1

    constrained = mw.Circuit("constrained_init")
    node = constrained.node("node")
    constrained.add(mw.Resistor("R1", node, constrained.ground, 1.0))
    constrained.add(mw.Capacitor("C1", node, constrained.ground, 0.1))
    constrained.set_initial_condition(node, 0.4)
    constrained_data = _system(constrained).transient(2)

    torch.testing.assert_close(
        constrained_data.node_voltage("node")[0],
        constrained_data.times.new_tensor(0.4),
    )
    assert constrained_data.diagnostics["initialization_energy_delta"] > 0.0


def test_dc_source_value_is_distinct_from_transient_waveform():
    circuit = mw.Circuit("dc_source")
    node = circuit.node("node")
    circuit.add(
        mw.VoltageSource(
            "V1",
            node,
            circuit.ground,
            voltage=0.0,
            waveform=_unit_step(0.0, 1.0),
        )
    )
    circuit.add(mw.Resistor("R1", node, circuit.ground, 1.0))
    system = _system(circuit)

    dc, _ = system.solve_dc()
    data = system.transient(1)

    torch.testing.assert_close(dc[0], dc.new_zeros(()))
    torch.testing.assert_close(data.node_voltage("node")[0], data.times.new_zeros(()))
    torch.testing.assert_close(data.node_voltage("node")[1], data.times.new_ones(()))


def test_timed_switch_t0_state_and_piecewise_factor_cache_are_consistent():
    circuit = mw.Circuit("switch", config=mw.MNAConfig(integration="backward_euler"))
    node = circuit.node("node")
    circuit.add(mw.VoltageSource("V1", node, circuit.ground, 1.0))
    circuit.add(
        mw.TimedSwitch(
            "S1",
            node,
            circuit.ground,
            (0.0, 0.002),
            initially_closed=False,
            on_resistance=1.0,
            off_resistance=1000.0,
        )
    )
    data = _system(circuit, dt=0.001).transient(3)

    torch.testing.assert_close(data.branch_current("S1")[0], data.times.new_tensor(1.0))
    assert data.diagnostics["transient_factorization_count"] == 2


def test_reused_system_refreshes_trainable_inductance_and_autograd_graph():
    inductance = torch.tensor(0.2, device="cuda", dtype=torch.float64, requires_grad=True)
    circuit = mw.Circuit("trainable_inductor", config=mw.MNAConfig(initialization="zero"))
    source = circuit.node("source")
    output = circuit.node("output")
    circuit.add(mw.VoltageSource("V1", source, circuit.ground, 1.0))
    circuit.add(mw.Resistor("R1", source, output, 1.0))
    circuit.add(mw.Inductor("L1", output, circuit.ground, inductance))
    system = _system(circuit, dt=0.001)

    first = system.transient(2).branch_current("L1")[-1]
    first.backward()
    first_value = first.detach()
    first_gradient = inductance.grad.detach().clone()
    with torch.no_grad():
        inductance.copy_(0.4)
    inductance.grad = None
    second = system.transient(2).branch_current("L1")[-1]
    second.backward()

    assert not torch.isclose(first_value, second.detach())
    assert bool(torch.isfinite(first_gradient))
    assert bool(torch.isfinite(inductance.grad))


def test_circuit_data_exports_physical_currents_and_migrates_all_tensors():
    circuit = mw.Circuit("data")
    source = circuit.node("source")
    output = circuit.node("output")
    circuit.add(mw.VoltageSource("V1", source, circuit.ground, 1.0))
    circuit.add(mw.Resistor("R1", source, output, 2.0))
    circuit.add(mw.Capacitor("C1", output, circuit.ground, 0.1))
    data = _system(circuit).transient(2)

    torch.testing.assert_close(data.branch_current("R1"), data.branch_current("C1"))
    migrated = data.to("cpu")
    assert migrated.diagnostics["condition"].device.type == "cpu"
    assert migrated.diagnostics["dt"].device.type == "cpu"

    with pytest.raises(ValueError, match="must be unique"):
        mw.CircuitData(
            circuit_name="duplicate",
            times=torch.arange(2, dtype=torch.float32),
            node_names=("0", "0"),
            node_voltages=torch.zeros((2, 2)),
            branch_names=(),
            branch_currents=torch.zeros((2, 0)),
        )
    with pytest.raises(ValueError, match="share one dtype"):
        mw.CircuitData(
            circuit_name="dtype",
            times=torch.arange(2, dtype=torch.float64),
            node_names=("0",),
            node_voltages=torch.zeros((2, 1), dtype=torch.float32),
            branch_names=(),
            branch_currents=torch.zeros((2, 0), dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="node_names must be unique"):
        mw.CircuitData(
            circuit_name="normalized_names",
            times=torch.arange(2, dtype=torch.float64),
            node_names=(1, "1"),
            node_voltages=torch.zeros((2, 2), dtype=torch.float64),
            branch_names=(),
            branch_currents=torch.zeros((2, 0), dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="real floating dtype"):
        mw.CircuitData(
            circuit_name="integer_times",
            times=torch.arange(2),
            node_names=("0",),
            node_voltages=torch.zeros((2, 1), dtype=torch.int64),
            branch_names=(),
            branch_currents=torch.zeros((2, 0), dtype=torch.int64),
        )


def test_reused_system_refreshes_mutated_constant_source():
    voltage = torch.tensor(1.0)
    circuit = mw.Circuit("source_refresh")
    node = circuit.node("node")
    circuit.add(mw.VoltageSource("V1", node, circuit.ground, voltage))
    circuit.add(mw.Resistor("R1", node, circuit.ground, 1.0))
    system = _system(circuit)

    system.transient(1)
    voltage.fill_(2.0)
    refreshed = system.transient(1)

    torch.testing.assert_close(refreshed.node_voltage("node"), refreshed.times.new_full((2,), 2.0))


def test_float32_switch_cache_key_matches_compiled_schedule_dtype():
    circuit = mw.Circuit("switch_rounding", config=mw.MNAConfig(integration="backward_euler"))
    node = circuit.node("node")
    circuit.add(mw.VoltageSource("V1", node, circuit.ground, 1.0))
    circuit.add(
        mw.TimedSwitch(
            "S1",
            node,
            circuit.ground,
            torch.tensor((0.30000002,), dtype=torch.float64),
            initially_closed=False,
            on_resistance=1.0,
            off_resistance=1000.0,
        )
    )
    system = compile_mna_system(
        circuit,
        dt=torch.tensor(0.1, dtype=torch.float32),
        device="cuda",
        dtype=torch.float32,
    )
    data = system.transient(3)

    residual = data.node_voltage("node")[-1] - data.branch_current("S1")[-1]
    torch.testing.assert_close(residual, residual.new_zeros(()), rtol=1.0e-5, atol=1.0e-6)


def test_fixed_mna_profile_has_no_per_step_host_transfer_growth():
    circuit = mw.Circuit("profile")
    source = circuit.node("source")
    output = circuit.node("output")
    circuit.add(mw.VoltageSource("V1", source, circuit.ground, 1.0))
    circuit.add(mw.Resistor("R1", source, output, 2.0))
    circuit.add(mw.Capacitor("C1", output, circuit.ground, 0.1))
    system = _system(circuit)
    system.transient(2)

    def profile_counts(steps):
        with torch.profiler.profile(
            activities=(
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ),
            acc_events=True,
        ) as profiler:
            system.transient(steps)
            torch.cuda.synchronize()
        parents = {}
        for event in profiler.events():
            if event.name == "aten::_local_scalar_dense":
                parent = "<none>" if event.cpu_parent is None else event.cpu_parent.name
                parents[parent] = parents.get(parent, 0) + 1
        return {event.key: event.count for event in profiler.key_averages()}, parents

    short, short_parents = profile_counts(4)
    long, long_parents = profile_counts(32)
    for event_name in ("aten::item", "aten::_local_scalar_dense", "aten::_to_copy"):
        assert long.get(event_name, 0) <= short.get(event_name, 0) + 2, (
            event_name,
            short_parents,
            long_parents,
        )


def test_mutual_inductance_matrix_and_transient_block_match_hand_stamp():
    circuit = mw.Circuit("transformer")
    source = circuit.node("source")
    first = circuit.node("first")
    second = circuit.node("second")
    circuit.add(mw.VoltageSource("V1", source, circuit.ground, 0.0))
    circuit.add(mw.Resistor("R1", source, first, 3.0))
    circuit.add(mw.Inductor("L1", first, circuit.ground, 2.0))
    circuit.add(mw.Resistor("R2", second, circuit.ground, 3.0))
    circuit.add(mw.Inductor("L2", second, circuit.ground, 8.0))
    circuit.add(mw.MutualInductor("K1", "L1", "L2", -0.25))
    system = _system(circuit, dt=0.1)

    expected = torch.tensor([[2.0, -1.0], [-1.0, 8.0]], device="cuda", dtype=torch.float64)
    torch.testing.assert_close(system.plan.inductance_matrix, expected, rtol=1.0e-12, atol=1.0e-14)
    dc, _ = system.solve_dc()
    state = system._initial_state(dc)
    sources = system._source_values(torch.tensor([0.0, 0.1], device="cuda", dtype=torch.float64))
    matrix, _ = system.assemble_transient(
        state,
        time=torch.tensor(0.1, device="cuda", dtype=torch.float64),
        source_values=sources,
        step_index=1,
    )
    l1 = system.graph.branch_index["L1"] + len(system.graph.nodes) - 1
    l2 = system.graph.branch_index["L2"] + len(system.graph.nodes) - 1
    torch.testing.assert_close(matrix[l1, l1], torch.tensor(-40.0, device="cuda", dtype=torch.float64))
    torch.testing.assert_close(matrix[l1, l2], torch.tensor(20.0, device="cuda", dtype=torch.float64))
    torch.testing.assert_close(matrix[l2, l1], torch.tensor(20.0, device="cuda", dtype=torch.float64))
    torch.testing.assert_close(matrix[l2, l2], torch.tensor(-160.0, device="cuda", dtype=torch.float64))


def test_singular_dc_and_cpu_runtime_are_explicit_errors():
    circuit = mw.Circuit("singular")
    node = circuit.node("node")
    circuit.add(mw.VoltageSource("V1", node, circuit.ground, 1.0))
    circuit.add(mw.Inductor("L1", node, circuit.ground, 1.0))

    with pytest.raises(ValueError, match="DC ideal-voltage/inductor loop.*L1"):
        _system(circuit).solve_dc()
    with pytest.raises(ValueError, match="requires a CUDA device"):
        compile_mna_system(circuit, dt=1.0e-3, device="cpu")


def test_trainable_cuda_parameter_keeps_autograd_graph():
    resistance = torch.tensor(2.0, device="cuda", dtype=torch.float64, requires_grad=True)
    circuit = mw.Circuit("gradient")
    source = circuit.node("source")
    output = circuit.node("output")
    circuit.add(mw.VoltageSource("V1", source, circuit.ground, 1.0))
    circuit.add(mw.Resistor("R1", source, output, resistance))
    circuit.add(mw.Resistor("R2", output, circuit.ground, 3.0))

    data = _system(circuit).transient(1)
    data.node_voltage("output")[-1].backward()

    expected = -3.0 / (2.0 + 3.0) ** 2
    torch.testing.assert_close(
        resistance.grad,
        torch.tensor(expected, device="cuda", dtype=torch.float64),
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_gpu_solution_matches_independent_numpy_mna_reference():
    circuit = mw.Circuit("reference")
    source = circuit.node("source")
    output = circuit.node("output")
    circuit.add(mw.VoltageSource("V1", source, circuit.ground, torch.tensor(2.7, dtype=torch.float64)))
    circuit.add(mw.Resistor("R1", source, output, torch.tensor(1.3, dtype=torch.float64)))
    circuit.add(mw.Resistor("R2", output, circuit.ground, torch.tensor(4.1, dtype=torch.float64)))
    solution, _ = _system(circuit).solve_dc()

    matrix = np.array([[1 / 1.3, -1 / 1.3, 1], [-1 / 1.3, 1 / 1.3 + 1 / 4.1, 0], [1, 0, 0]], dtype=np.float64)
    rhs = np.array([0.0, 0.0, 2.7], dtype=np.float64)
    reference = np.linalg.solve(matrix, rhs)
    np.testing.assert_allclose(solution.detach().cpu().numpy(), reference, rtol=1.0e-12, atol=1.0e-14)
