from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from tests.gradients.test_fdtd_rf_lumped_adjoint import (
    _assert_three_step_central_difference,
)
from witwin.maxwell.compiler import compile_mna_system
from witwin.maxwell.fdtd.adjoint.circuits import _branch_current_row


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Circuit-coupled FDTD gradients require CUDA.",
)


def _port(name="feed", *, x=0.0, reference_impedance=50.0):
    return mw.LumpedPort(
        name=name,
        positive=(x, 0.0, 0.005),
        negative=(x, 0.0, -0.005),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(
            position=(x, 0.0, -0.0025),
            size=(0.015, 0.015, 0.0),
        ),
        reference_impedance=reference_impedance,
    )


def _series_circuit(*, resistance, inductance, capacitance, source_amplitude):
    circuit = mw.Circuit("gradient_network")
    input_node = circuit.node("input")
    middle = circuit.node("middle")
    output = circuit.node("output")
    circuit.add(mw.Resistor("R1", input_node, middle, resistance))
    circuit.add(mw.Inductor("L1", middle, output, inductance))
    circuit.add(mw.Capacitor("C1", output, circuit.ground, capacitance))
    circuit.add(
        mw.CurrentSource(
            "I1",
            circuit.ground,
            input_node,
            waveform=mw.SineWaveform(0.0, source_amplitude, 3.0e9),
        )
    )
    circuit.bind_port("feed", positive=input_node, negative=circuit.ground)
    return circuit


def _one_port_scene(circuit, *, material_density=None, reference_impedance=50.0):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.02, 0.02),) * 3),
        grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.none(),
        ports=(_port(reference_impedance=reference_impedance),),
        circuits=(circuit,),
        device="cuda",
    )
    if material_density is not None:
        scene.add_material_region(
            mw.MaterialRegion(
                name="bound_port_design",
                geometry=mw.Box(
                    position=(0.0, 0.0, 0.0),
                    size=(0.015, 0.015, 0.015),
                ),
                density=material_density,
                eps_bounds=(1.0, 4.0),
            )
        )
    return scene


def _port_voltage_objective(scene_or_module, *, steps=16):
    result = mw.Simulation.fdtd(
        scene_or_module,
        frequency=3.0e9,
        run_time=mw.TimeConfig(time_steps=steps),
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).run()
    return result.port("feed").voltage.abs().square().sum(), result


@pytest.mark.parametrize(
    ("component", "center", "steps"),
    (
        ("resistance", 25.0, (2.5, 1.25, 0.625)),
        ("inductance", 0.5e-9, (5.0e-11, 2.5e-11, 1.25e-11)),
        ("capacitance", 1.0e-12, (1.0e-13, 5.0e-14, 2.5e-14)),
        ("source_amplitude", 0.01, (1.0e-3, 5.0e-4, 2.5e-4)),
    ),
)
def test_circuit_parameter_adjoint_matches_central_difference(
    component,
    center,
    steps,
):
    parameter = torch.tensor(center, device="cuda", requires_grad=True)
    values = {
        "resistance": 25.0,
        "inductance": 0.5e-9,
        "capacitance": 1.0e-12,
        "source_amplitude": 0.01,
    }
    values[component] = parameter
    scene = _one_port_scene(_series_circuit(**values))

    result = _assert_three_step_central_difference(
        parameter,
        lambda: _port_voltage_objective(scene),
        steps,
        max_relative_error=0.01,
    )
    assert result.port("feed").voltage.grad_fn is not None
    assert result.circuit("gradient_network").node_voltages.device.type == "cuda"


def test_float64_circuit_leaf_survives_float32_prepared_cache():
    resistance = torch.tensor(
        25.0,
        device="cuda",
        dtype=torch.float64,
        requires_grad=True,
    )
    scene = _one_port_scene(
        _series_circuit(
            resistance=resistance,
            inductance=0.5e-9,
            capacitance=1.0e-12,
            source_amplitude=0.01,
        )
    )

    result = _assert_three_step_central_difference(
        resistance,
        lambda: _port_voltage_objective(scene),
        (2.5, 1.25, 0.625),
        max_relative_error=0.01,
    )
    assert result.solver_stats["circuit_cuda_graph_active"] is False


class _BoundPortMaterialScene(mw.SceneModule):
    def __init__(self):
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(0.0, device="cuda"))

    def to_scene(self):
        circuit = _series_circuit(
            resistance=25.0,
            inductance=0.5e-9,
            capacitance=1.0e-12,
            source_amplitude=0.01,
        )
        density = torch.sigmoid(self.logit).expand(2, 2, 2)
        return _one_port_scene(circuit, material_density=density)


class _DerivedCircuitScene(mw.SceneModule):
    def __init__(self):
        super().__init__()
        self.resistance_logit = torch.nn.Parameter(torch.tensor(0.1, device="cuda"))
        self.source_logit = torch.nn.Parameter(torch.tensor(-0.2, device="cuda"))

    def to_scene(self):
        resistance = 25.0 + 5.0 * torch.tanh(self.resistance_logit)
        source_amplitude = 0.01 * (1.0 + 0.2 * torch.tanh(self.source_logit))
        return _one_port_scene(
            _series_circuit(
                resistance=resistance,
                inductance=0.5e-9,
                capacitance=1.0e-12,
                source_amplitude=source_amplitude,
            )
        )


@pytest.mark.parametrize("parameter_name", ("resistance_logit", "source_logit"))
def test_scene_module_derived_circuit_parameters_match_central_difference(
    parameter_name,
):
    model = _DerivedCircuitScene().cuda()
    parameter = getattr(model, parameter_name)

    _assert_three_step_central_difference(
        parameter,
        lambda: _port_voltage_objective(model),
        (0.08, 0.04, 0.02),
        max_relative_error=0.01,
    )


def test_bound_port_material_adjoint_matches_central_difference():
    model = _BoundPortMaterialScene().cuda()
    result = _assert_three_step_central_difference(
        model.logit,
        lambda: _port_voltage_objective(model),
        (0.08, 0.04, 0.02),
        max_relative_error=0.01,
    )
    assert result.port("feed").voltage.grad_fn is not None


class _SharedCircuitGeometryScene(mw.SceneModule):
    def __init__(self):
        super().__init__()
        self.design = torch.nn.Parameter(torch.tensor(0.0, device="cuda"))

    def to_scene(self):
        resistance = 25.0 + 4.0 * torch.tanh(self.design)
        circuit = _series_circuit(
            resistance=resistance,
            inductance=0.5e-9,
            capacitance=1.0e-12,
            source_amplitude=0.01,
        )
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.02, 0.02),) * 3),
            grid=mw.GridSpec.uniform(0.005),
            boundary=mw.BoundarySpec.none(),
            ports=(_port(),),
            circuits=(circuit,),
            device="cuda",
            subpixel_samples=3,
        )
        extent = 0.012 + 0.004 * torch.sigmoid(self.design)
        scene.add_structure(
            mw.Structure(
                name="shared_design_box",
                geometry=mw.Box(
                    position=(0.0, 0.0, 0.0),
                    size=torch.stack((extent, extent, extent)),
                ),
                material=mw.Material(eps_r=2.5),
            )
        )
        return scene


def test_shared_scene_module_circuit_and_geometry_graph_matches_central_difference():
    model = _SharedCircuitGeometryScene().cuda()

    _assert_three_step_central_difference(
        model.design,
        lambda: _port_voltage_objective(model),
        (0.08, 0.04, 0.02),
        max_relative_error=0.01,
    )


def _two_port_scene(load_resistance):
    circuit = mw.Circuit("two_port_load")
    load_node = circuit.node("load")
    middle = circuit.node("middle")
    output = circuit.node("output")
    circuit.add(mw.Resistor("R1", load_node, middle, load_resistance))
    circuit.add(mw.Inductor("L1", middle, output, 0.4e-9))
    circuit.add(mw.Capacitor("C1", output, circuit.ground, 0.8e-12))
    circuit.bind_port("load", positive=load_node, negative=circuit.ground)
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.025, 0.025),) * 3),
        grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.pml(num_layers=3),
        ports=(_port("feed", x=-0.01), _port("load", x=0.01)),
        circuits=(circuit,),
        device="cuda",
    )


def test_two_port_insertion_loss_adjoint_matches_central_difference():
    resistance = torch.tensor(40.0, device="cuda", requires_grad=True)
    scene = _two_port_scene(resistance)

    def objective():
        result = mw.Simulation.fdtd(
            scene,
            frequency=3.0e9,
            excitations=mw.PortExcitation(
                "feed",
                source_time=mw.GaussianPulse(frequency=3.0e9, fwidth=1.5e9),
            ),
            run_time=mw.TimeConfig(time_steps=48),
            spectral_sampler=mw.SpectralSampler(window="none"),
        ).run()
        transmission = result.port("load").b / result.port("feed").a
        insertion_loss_db = -20.0 * torch.log10(torch.abs(transmission).clamp_min(1.0e-12))
        return insertion_loss_db.sum(), result

    result = _assert_three_step_central_difference(
        resistance,
        objective,
        (4.0, 2.0, 1.0),
        max_relative_error=0.01,
    )
    assert result.port("load").voltage.grad_fn is not None


def test_circuit_data_objective_adjoint_matches_central_difference():
    resistance = torch.tensor(25.0, device="cuda", requires_grad=True)
    scene = _one_port_scene(
        _series_circuit(
            resistance=resistance,
            inductance=0.5e-9,
            capacitance=1.0e-12,
            source_amplitude=0.01,
        )
    )

    def objective():
        result = _port_voltage_objective(scene, steps=16)[1]
        data = result.circuit("gradient_network")
        response = data.node_voltage("output")[1:] + 25.0 * data.branch_current("R1")[1:]
        return response.square().sum(), result

    result = _assert_three_step_central_difference(
        resistance,
        objective,
        (2.5, 1.25, 0.625),
        max_relative_error=0.01,
    )
    data = result.circuit("gradient_network")
    assert data.node_voltages.grad_fn is not None
    assert data.branch_currents.grad_fn is not None


def test_all_circuit_data_power_balance_and_tensor_diagnostics_are_differentiable():
    resistance = torch.tensor(25.0, device="cuda", requires_grad=True)
    scene = _one_port_scene(
        _series_circuit(
            resistance=resistance,
            inductance=0.5e-9,
            capacitance=1.0e-12,
            source_amplitude=0.01,
        )
    )

    def objective():
        result = _port_voltage_objective(scene, steps=16)[1]
        data = result.circuit("gradient_network")
        tensor_diagnostics = {
            name: value
            for name, value in data.diagnostics.items()
            if isinstance(value, torch.Tensor)
        }
        assert set(tensor_diagnostics) == {
            "dt",
            "dc_condition",
            "last_condition",
            "port_powers",
            "field_energy_changes",
            "field_energy_change_total",
        }
        assert all(value.grad_fn is not None for value in tensor_diagnostics.values())
        assert all(value.grad_fn is not None for value in data.device_powers.values())
        assert data.energy_balance.grad_fn is not None
        power_objective = sum(
            value[1:].square().mean() for value in data.device_powers.values()
        )
        diagnostic_objective = (
            data.energy_balance[1:].square().mean()
            + tensor_diagnostics["port_powers"][1:].square().mean()
            + tensor_diagnostics["field_energy_changes"][1:].square().mean()
            + tensor_diagnostics["field_energy_change_total"][1:].square().mean()
            + 1.0e-4
            * (
                torch.log(tensor_diagnostics["dc_condition"])
                + torch.log(tensor_diagnostics["last_condition"])
            )
            + 1.0e-6 * tensor_diagnostics["dt"]
        )
        return power_objective + diagnostic_objective, result

    _assert_three_step_central_difference(
        resistance,
        objective,
        (2.5, 1.25, 0.625),
        max_relative_error=0.01,
    )


def test_current_source_branch_objective_retains_waveform_gradient():
    source_amplitude = torch.tensor(0.01, device="cuda", requires_grad=True)
    scene = _one_port_scene(
        _series_circuit(
            resistance=25.0,
            inductance=0.5e-9,
            capacitance=1.0e-12,
            source_amplitude=source_amplitude,
        )
    )

    def objective():
        result = _port_voltage_objective(scene, steps=16)[1]
        current = result.circuit("gradient_network").branch_current("I1")[1:]
        return current.square().sum(), result

    result = _assert_three_step_central_difference(
        source_amplitude,
        objective,
        (1.0e-3, 5.0e-4, 2.5e-4),
        max_relative_error=0.01,
    )
    assert result.circuit("gradient_network").branch_currents.grad_fn is not None


def test_spice_parameter_expression_uses_one_ancestry_safe_gradient_frontier():
    base_resistance = torch.tensor(12.5, device="cuda", requires_grad=True)
    netlist = """
    .param base=12.5
    R1 node 0 {2*base}
    I1 0 node SIN(0 0.01 3g)
    .end
    """

    def objective():
        # SPICE expressions follow ordinary eager PyTorch semantics. Reparse so
        # every optimization/finite-difference evaluation materializes a fresh
        # ``2 * base`` graph at the current leaf value.
        circuit = mw.parse_spice(
            netlist,
            name="expression_network",
            parameters={"base": base_resistance},
        )
        node = next(node for node in circuit.nodes if node.name == "node")
        circuit.bind_port("feed", positive=node, negative=circuit.ground)
        return _port_voltage_objective(_one_port_scene(circuit))

    _assert_three_step_central_difference(
        base_resistance,
        objective,
        (1.25, 0.625, 0.3125),
        max_relative_error=0.01,
    )
    first_gradient = base_resistance.grad.detach().clone()
    base_resistance.grad = None
    repeated_loss, _result = objective()
    repeated_loss.backward()
    torch.testing.assert_close(base_resistance.grad, first_gradient)


def test_circuit_data_t0_seed_is_an_explicit_error():
    resistance = torch.tensor(25.0, device="cuda", requires_grad=True)
    scene = _one_port_scene(
        _series_circuit(
            resistance=resistance,
            inductance=0.5e-9,
            capacitance=1.0e-12,
            source_amplitude=0.01,
        )
    )
    data = _port_voltage_objective(scene, steps=4)[1].circuit("gradient_network")

    with pytest.raises(RuntimeError, match="t=0"):
        data.node_voltages[0].sum().backward()


def test_empty_physical_branch_row_is_seed_safe():
    solution = torch.ones(1, device="cuda", requires_grad=True)
    row = _branch_current_row(
        SimpleNamespace(physical_devices=()),
        {},
        solution,
    )

    assert row.shape == (0,)
    torch.testing.assert_close(torch.zeros_like(row), row)


@pytest.mark.parametrize("unsupported", ("dc_source", "initial_condition"))
def test_zero_state_adjoint_limitations_are_explicit(unsupported):
    parameter = torch.tensor(0.0, device="cuda", requires_grad=True)
    circuit = mw.Circuit("zero_state_limit")
    node = circuit.node("node")
    circuit.add(mw.Resistor("R1", node, circuit.ground, 25.0))
    if unsupported == "dc_source":
        circuit.add(mw.CurrentSource("I1", circuit.ground, node, current=parameter))
    else:
        circuit.set_initial_condition(node, parameter)
    circuit.bind_port("feed", positive=node, negative=circuit.ground)

    with pytest.raises(NotImplementedError, match="trainable|initial"):
        _port_voltage_objective(_one_port_scene(circuit), steps=4)


@pytest.mark.parametrize("source_type", (mw.CurrentSource, mw.VoltageSource))
def test_waveform_source_trainable_dc_value_is_an_explicit_error(source_type):
    dc_value = torch.tensor(0.0, device="cuda", requires_grad=True)
    circuit = mw.Circuit("waveform_dc_limit")
    node = circuit.node("node")
    circuit.add(mw.Resistor("R1", node, circuit.ground, 25.0))
    waveform = mw.SineWaveform(0.0, 0.01, 3.0e9)
    if source_type is mw.CurrentSource:
        source = source_type(
            "I1",
            circuit.ground,
            node,
            current=dc_value,
            waveform=waveform,
        )
    else:
        source = source_type(
            "V1",
            node,
            circuit.ground,
            voltage=dc_value,
            waveform=waveform,
        )
    circuit.add(source)
    circuit.bind_port("feed", positive=node, negative=circuit.ground)

    with pytest.raises(NotImplementedError, match="trainable DC source value"):
        _port_voltage_objective(_one_port_scene(circuit), steps=4)


class _DerivedCircuitGuardScene(mw.SceneModule):
    def __init__(self, unsupported):
        super().__init__()
        self.unsupported = unsupported
        self.parameter = torch.nn.Parameter(torch.tensor(0.0, device="cuda"))

    def to_scene(self):
        derived_zero = self.parameter * 0.0
        circuit = mw.Circuit("derived_guard")
        node = circuit.node("node")
        circuit.add(mw.Resistor("R1", node, circuit.ground, 25.0))
        if self.unsupported == "dc_source":
            circuit.add(
                mw.CurrentSource(
                    "I1",
                    circuit.ground,
                    node,
                    current=derived_zero,
                    waveform=mw.SineWaveform(0.0, 0.01, 3.0e9),
                )
            )
        else:
            circuit.set_initial_condition(node, derived_zero)
        circuit.bind_port("feed", positive=node, negative=circuit.ground)
        return _one_port_scene(circuit)


@pytest.mark.parametrize("unsupported", ("dc_source", "initial_condition"))
def test_scene_module_derived_zero_state_inputs_fail_explicitly(unsupported):
    model = _DerivedCircuitGuardScene(unsupported).cuda()
    with pytest.raises(NotImplementedError, match="trainable|initial"):
        _port_voltage_objective(model, steps=4)


class _DerivedReferenceImpedanceScene(mw.SceneModule):
    def __init__(self):
        super().__init__()
        self.log_impedance = torch.nn.Parameter(torch.tensor(0.0, device="cuda"))

    def to_scene(self):
        impedance = 50.0 + 5.0 * torch.tanh(self.log_impedance)
        circuit = _series_circuit(
            resistance=25.0,
            inductance=0.5e-9,
            capacitance=1.0e-12,
            source_amplitude=0.01,
        )
        return _one_port_scene(circuit, reference_impedance=impedance)


def test_scene_module_derived_reference_impedance_fails_explicitly():
    with pytest.raises(NotImplementedError, match="trainable reference_impedance"):
        _port_voltage_objective(_DerivedReferenceImpedanceScene().cuda(), steps=4)


def _standalone_transient(circuit, *, dt, steps):
    return compile_mna_system(
        circuit,
        dt=torch.tensor(dt, dtype=torch.float64),
        device="cuda",
        dtype=torch.float64,
    ).transient(steps)


def test_rc_cutoff_gradient_matches_central_difference():
    resistance = 2.0
    center = 0.05
    cutoff_frequency = 1.0 / (2.0 * torch.pi * resistance * center)
    capacitance = torch.tensor(
        center,
        device="cuda",
        dtype=torch.float64,
        requires_grad=True,
    )

    def objective():
        circuit = mw.Circuit(
            "rc_cutoff_gradient",
            config=mw.MNAConfig(initialization="zero"),
        )
        source = circuit.node("source")
        output = circuit.node("output")
        circuit.add(
            mw.VoltageSource(
                "V1",
                source,
                circuit.ground,
                waveform=mw.SineWaveform(0.0, 1.0, cutoff_frequency),
            )
        )
        circuit.add(mw.Resistor("R1", source, output, resistance))
        circuit.add(mw.Capacitor("C1", output, circuit.ground, capacitance))
        data = _standalone_transient(circuit, dt=2.0e-3, steps=320)
        return data.node_voltage("output")[1:].square().mean(), data

    data = _assert_three_step_central_difference(
        capacitance,
        objective,
        (5.0e-3, 2.5e-3, 1.25e-3),
        max_relative_error=0.01,
    )
    assert data.node_voltages.grad_fn is not None


def test_rlc_near_resonance_gradient_matches_central_difference():
    center = 0.1
    capacitance = 0.1
    resonance_frequency = 1.0 / (2.0 * torch.pi * (center * capacitance) ** 0.5)
    inductance = torch.tensor(
        center,
        device="cuda",
        dtype=torch.float64,
        requires_grad=True,
    )

    def objective():
        circuit = mw.Circuit(
            "rlc_resonance_gradient",
            config=mw.MNAConfig(initialization="zero"),
        )
        source = circuit.node("source")
        middle = circuit.node("middle")
        output = circuit.node("output")
        circuit.add(
            mw.VoltageSource(
                "V1",
                source,
                circuit.ground,
                waveform=mw.SineWaveform(0.0, 1.0, 0.9 * resonance_frequency),
            )
        )
        circuit.add(mw.Resistor("R1", source, middle, 0.5))
        circuit.add(mw.Inductor("L1", middle, output, inductance))
        circuit.add(mw.Capacitor("C1", output, circuit.ground, capacitance))
        data = _standalone_transient(circuit, dt=2.0e-3, steps=480)
        response = data.node_voltage("output")[1:]
        return response.square().mean(), data

    data = _assert_three_step_central_difference(
        inductance,
        objective,
        (1.0e-2, 5.0e-3, 2.5e-3),
        max_relative_error=0.01,
    )
    assert data.node_voltages.grad_fn is not None
