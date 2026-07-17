import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.checkpoint import capture_checkpoint_state
from witwin.maxwell.fdtd.circuits import prepare_circuit_time_series
from witwin.maxwell.fdtd.ports import apply_port_runtimes
from witwin.maxwell.fdtd.ports import (
    _edge_control_volume,
    accumulate_port_observers,
    prepare_port_spectral_accumulators,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Strong FDTD/circuit coupling requires CUDA.",
)


def _port(*, termination=None):
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
        termination=termination,
    )


def _scene(port, *, circuits=()):
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.02, 0.02),) * 3),
        grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.none(),
        ports=(port,),
        circuits=circuits,
        device="cuda",
    )


def _series_circuit(*, resistance=25.0, inductance=0.5e-9, capacitance=1.0e-12):
    circuit = mw.Circuit("load")
    input_node = circuit.node("input")
    middle = circuit.node("middle")
    output = circuit.node("output")
    circuit.add(mw.Resistor("R1", input_node, middle, resistance))
    circuit.add(mw.Inductor("L1", middle, output, inductance))
    circuit.add(mw.Capacitor("C1", output, circuit.ground, capacitance))
    circuit.bind_port("feed", positive=input_node, negative=circuit.ground)
    return circuit


def _two_port_scene(*, circuit_load: bool):
    def port(name, x, termination=None):
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
            termination=termination,
        )

    load = None
    circuits = ()
    if circuit_load:
        network = mw.Circuit("network")
        input_node = network.node("input")
        middle = network.node("middle")
        output = network.node("output")
        network.add(mw.Resistor("R1", input_node, middle, 40.0))
        network.add(mw.Inductor("L1", middle, output, 0.4e-9))
        network.add(mw.Capacitor("C1", output, network.ground, 0.8e-12))
        network.bind_port("load", positive=input_node, negative=network.ground)
        circuits = (network,)
    else:
        load = mw.SeriesRLC(r=40.0, l=0.4e-9, c=0.8e-12)
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.024, 0.024),) * 3),
        grid=mw.GridSpec.uniform(0.004),
        boundary=mw.BoundarySpec.pml(num_layers=3),
        ports=(port("feed", -0.008), port("load", 0.008, load)),
        circuits=circuits,
        device="cuda",
    )


def test_coupled_mna_matches_native_series_rlc_without_a_delayed_step():
    resistance = 25.0
    inductance = 0.5e-9
    capacitance = 1.0e-12
    native = mw.Simulation.fdtd(
        _scene(
            _port(
                termination=mw.SeriesRLC(
                    r=resistance,
                    l=inductance,
                    c=capacitance,
                )
            )
        ),
        frequency=3.0e9,
    ).prepare().solver
    coupled = mw.Simulation.fdtd(
        _scene(
            _port(),
            circuits=(
                _series_circuit(
                    resistance=resistance,
                    inductance=inductance,
                    capacitance=capacitance,
                ),
            ),
        ),
        frequency=3.0e9,
    ).prepare().solver
    samples = (0.1, -0.2, 0.3, -0.4, 0.2, 0.0, -0.1, 0.05)
    prepare_circuit_time_series(coupled, len(samples))
    native_port = native._port_runtimes[0]
    coupled_port = coupled._port_runtimes[0]
    assert native_port.yee_control_volume is None
    assert coupled_port.yee_control_volume is None

    for free_field in samples:
        native.Ez.zero_()
        coupled.Ez.zero_()
        native.Ez.view(-1).index_fill_(
            0,
            native_port.lumped.linear_indices,
            free_field,
        )
        coupled.Ez.view(-1).index_fill_(
            0,
            coupled_port.circuit_port.field.linear_indices,
            free_field,
        )
        apply_port_runtimes(native)
        apply_port_runtimes(coupled)

        torch.testing.assert_close(native.Ez, coupled.Ez, rtol=2.0e-6, atol=6.0e-8)
        torch.testing.assert_close(
            native_port.lumped.last_branch_current,
            coupled_port.circuit_port.field.last_current,
            rtol=2.0e-6,
            atol=1.0e-11,
        )
        torch.testing.assert_close(
            native_port.lumped.last_voltage_midpoint,
            coupled_port.circuit_port.field.last_voltage,
            rtol=2.0e-6,
            atol=3.0e-10,
        )

    checkpoint = capture_checkpoint_state(coupled, step=len(samples))
    circuit_runtime = coupled._circuit_runtimes[0]
    assert checkpoint.schema.circuit_state_names == (
        "circuit_0_step",
        "circuit_0_capacitor_voltage_C1",
        "circuit_0_capacitor_current_C1",
        "circuit_0_inductor_current",
        "circuit_0_inductor_voltage",
    )
    torch.testing.assert_close(
        checkpoint.tensors["circuit_0_capacitor_voltage_C1"],
        circuit_runtime.state.capacitor_voltage["C1"],
    )
    torch.testing.assert_close(
        checkpoint.tensors["circuit_0_capacitor_current_C1"],
        circuit_runtime.state.capacitor_current["C1"],
    )
    torch.testing.assert_close(
        checkpoint.tensors["circuit_0_inductor_current"],
        circuit_runtime.state.inductor_current,
    )
    torch.testing.assert_close(
        checkpoint.tensors["circuit_0_inductor_voltage"],
        circuit_runtime.state.inductor_voltage,
    )
    frozen_voltage = checkpoint.tensors["circuit_0_capacitor_voltage_C1"].clone()
    circuit_runtime.state.capacitor_voltage["C1"].add_(1.0)
    torch.testing.assert_close(
        checkpoint.tensors["circuit_0_capacitor_voltage_C1"],
        frozen_voltage,
    )


def test_passive_rlc_coupling_conserves_field_circuit_energy_and_decays():
    solver = mw.Simulation.fdtd(
        _scene(_port(), circuits=(_series_circuit(),)),
        frequency=3.0e9,
    ).prepare().solver
    steps = 2048
    prepare_circuit_time_series(solver, steps)
    port_runtime = solver._port_runtimes[0]
    circuit_runtime = solver._circuit_runtimes[0]
    control_volume = _edge_control_volume(solver, "Ez")
    solver.Ez.zero_()
    solver.Ez.view(-1).index_fill_(
        0,
        port_runtime.circuit_port.field.linear_indices,
        0.1,
    )

    def field_energy():
        return 0.5 * torch.sum(
            solver.eps_Ez
            * control_volume
            * solver.Ez.square()
        )

    initial_energy = field_energy() + circuit_runtime._stored_energy(circuit_runtime.state)
    cumulative_loss = initial_energy.new_zeros(())
    previous_energy = initial_energy
    maximum_residual = initial_energy.new_zeros(())
    for index in range(steps):
        apply_port_runtimes(solver)
        cumulative_loss = (
            cumulative_loss
            + solver.dt * circuit_runtime.device_power_samples["R1"][index + 1]
        )
        energy = field_energy() + circuit_runtime._stored_energy(circuit_runtime.state)
        maximum_residual = torch.maximum(
            maximum_residual,
            torch.abs(energy + cumulative_loss - initial_energy),
        )
        assert bool(energy <= previous_energy * (1.0 + 2.0e-6))
        previous_energy = energy

    assert bool(maximum_residual / initial_energy < 0.02)
    assert bool(previous_energy / initial_energy < 0.05)


def test_circuit_run_returns_cuda_data_and_checkpointed_companion_history():
    circuit = mw.Circuit("driven")
    node = circuit.node("input")
    circuit.add(mw.Resistor("R1", node, circuit.ground, 50.0))
    circuit.add(
        mw.CurrentSource(
            "I1",
            circuit.ground,
            node,
            waveform=mw.SineWaveform(0.0, 0.01, 3.0e9),
        )
    )
    circuit.bind_port("feed", positive=node, negative=circuit.ground)
    result = mw.Simulation.fdtd(
        _scene(_port(), circuits=(circuit,)),
        frequency=3.0e9,
        run_time=mw.TimeConfig(time_steps=16),
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).run()

    data = result.circuit("driven")
    checkpoint = capture_checkpoint_state(result.solver, step=16)

    assert data.times.shape == (17,)
    assert data.node_voltages.device.type == "cuda"
    assert data.branch_currents.device.type == "cuda"
    assert torch.all(torch.isfinite(data.node_voltages))
    assert torch.any(torch.abs(data.node_voltage("input")) > 0.0)
    assert result.stats()["num_circuits"] == 1
    assert data.diagnostics["factorization_count"] == 2
    assert checkpoint.schema.version == 2
    assert checkpoint.schema.circuit_state_names == (
        "circuit_0_step",
        "circuit_0_inductor_current",
        "circuit_0_inductor_voltage",
    )
    assert checkpoint.tensors["circuit_0_step"].device.type == "cuda"
    assert int(checkpoint.tensors["circuit_0_step"]) == 16
    port_power = data.diagnostics["port_powers"][:, 0]
    field_change = data.diagnostics["field_energy_changes"][:, 0]
    torch.testing.assert_close(
        field_change[2:],
        -result.solver.dt * port_power[2:],
        rtol=2.0e-6,
        atol=2.0e-20,
    )


def test_two_port_fdtd_s11_matches_native_series_rlc_load():
    def run(circuit_load):
        return mw.Simulation.fdtd(
            _two_port_scene(circuit_load=circuit_load),
            frequencies=(2.5e9, 3.0e9),
            excitations=mw.PortExcitation(
                "feed",
                source_time=mw.GaussianPulse(frequency=2.75e9, fwidth=1.5e9),
            ),
            run_time=mw.TimeConfig(time_steps=128),
            spectral_sampler=mw.SpectralSampler(window="none"),
        ).run()

    native = run(False).port("feed")
    coupled = run(True).port("feed")
    native_s11 = native.b / native.a
    coupled_s11 = coupled.b / coupled.a
    magnitude_error = torch.abs(torch.abs(coupled_s11) / torch.abs(native_s11) - 1.0)
    phase_error = torch.abs(torch.angle(coupled_s11 / native_s11))

    assert bool(torch.all(magnitude_error < 0.01))
    assert bool(torch.all(phase_error < torch.deg2rad(phase_error.new_tensor(2.0))))


def test_circuit_step_reuses_factors_without_scalar_sync_or_host_copy():
    circuit = mw.Circuit("profile")
    node = circuit.node("input")
    circuit.add(mw.Resistor("R1", node, circuit.ground, 50.0))
    circuit.bind_port("feed", positive=node, negative=circuit.ground)
    solver = mw.Simulation.fdtd(
        _scene(_port(), circuits=(circuit,)),
        frequency=3.0e9,
    ).prepare().solver
    prepare_circuit_time_series(solver, 80)
    prepare_port_spectral_accumulators(solver, 80, "none")
    runtime = solver._circuit_runtimes[0]

    for _ in range(4):
        apply_port_runtimes(solver)
        accumulate_port_observers(solver)
    torch.cuda.synchronize()

    def profile_steps(count):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            acc_events=True,
        ) as profile:
            for _ in range(count):
                apply_port_runtimes(solver)
                accumulate_port_observers(solver)
        torch.cuda.synchronize()
        return {event.key for event in profile.key_averages()}

    short_events = profile_steps(8)
    long_events = profile_steps(64)
    forbidden = {
        "aten::item",
        "aten::_local_scalar_dense",
    }
    assert len(runtime.factor_cache) == 1
    assert forbidden.isdisjoint(short_events)
    assert forbidden.isdisjoint(long_events)
    assert not any(
        "Memcpy HtoD" in name or "Memcpy DtoH" in name
        for name in short_events | long_events
    )


def test_circuit_bound_port_rejects_direct_excitation():
    circuit = mw.Circuit("load")
    node = circuit.node("input")
    circuit.add(mw.Resistor("R1", node, circuit.ground, 50.0))
    circuit.bind_port("feed", positive=node, negative=circuit.ground)
    scene = _scene(_port(), circuits=(circuit,))

    with pytest.raises(ValueError, match="Circuit-bound port.*PortExcitation"):
        mw.Simulation.fdtd(
            scene,
            frequency=3.0e9,
            excitations=mw.PortExcitation("feed"),
        ).prepare()

def test_nonzero_coupled_dc_current_rejects_an_inconsistent_zero_field_start():
    circuit = mw.Circuit("dc_driven")
    node = circuit.node("input")
    circuit.add(mw.VoltageSource("V1", node, circuit.ground, 1.0))
    circuit.add(mw.Resistor("R1", node, circuit.ground, 50.0))
    circuit.bind_port("feed", positive=node, negative=circuit.ground)

    with pytest.raises(ValueError, match="nonzero DC port current.*zero initial Yee field"):
        mw.Simulation.fdtd(
            _scene(_port(), circuits=(circuit,)),
            frequency=3.0e9,
        ).prepare()
