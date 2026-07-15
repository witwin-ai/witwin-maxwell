import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.scene import BoundarySpec, Domain, GridSpec, prepare_scene


def _scene(*, ports=(), circuits=()):
    return mw.Scene(
        domain=Domain(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
        grid=GridSpec.uniform(0.25),
        boundary=BoundarySpec.none(),
        ports=ports,
        circuits=circuits,
        device="cpu",
    )


def _circuit(name="network"):
    circuit = mw.Circuit(name)
    node = circuit.node("signal")
    circuit.add(mw.Resistor("R1", node, circuit.ground, 50.0))
    return circuit


def test_circuit_nodes_are_interned_ordered_and_have_unique_ground():
    circuit = mw.Circuit("network")
    first = circuit.node("first")
    second = circuit.node("second")

    assert circuit.node("0") is circuit.ground
    assert circuit.node("first") is first
    assert circuit.node("FIRST") is first
    assert circuit.nodes == (circuit.ground, first, second)
    assert circuit.ground.is_ground


def test_shared_rlc_types_accept_circuit_nodes_without_losing_tensor_identity():
    circuit = mw.Circuit("network")
    positive = circuit.node("positive")
    resistance = torch.tensor(50.0, dtype=torch.float64, requires_grad=True)
    resistor = mw.Resistor("R1", positive, circuit.ground, resistance)

    circuit.add(resistor)

    assert circuit.devices == (resistor,)
    assert resistor.positive is positive
    assert resistor.negative is circuit.ground
    assert resistor.resistance is resistance
    assert resistor.resistance.requires_grad
    assert isinstance(resistor, mw.CircuitDevice)
    assert resistor.terminals == (positive, circuit.ground)
    assert resistor.parameters == {"resistance": resistance}
    assert resistor.initial_condition is None


def test_scene_local_rlc_coordinate_contract_is_unchanged():
    resistor = mw.Resistor("R1", (0.75, 0.5, 0.5), (0.25, 0.5, 0.5), 50.0)
    scene = _scene()

    assert scene.add_lumped_element(resistor) is scene
    assert scene.lumped_elements == [resistor]
    assert scene.compile_lumped_elements(device="cpu")[0].element_name == "R1"


def test_circuit_rejects_foreign_nodes_duplicate_devices_and_invalid_bindings():
    circuit = mw.Circuit("network")
    foreign = mw.Circuit("other").node("signal")
    signal = circuit.node("signal")

    with pytest.raises(ValueError, match="does not belong"):
        circuit.add(mw.Resistor("R1", foreign, circuit.ground, 50.0))

    circuit.add(mw.Resistor("R1", signal, circuit.ground, 50.0))
    with pytest.raises(ValueError, match="already present"):
        circuit.add(mw.Capacitor("r1", signal, circuit.ground, 1.0e-12))
    with pytest.raises(ValueError, match="distinct"):
        circuit.bind_port("feed", positive=signal, negative=signal)


def test_scene_add_clone_prepare_and_compile_preserve_circuit_order():
    first = _circuit("first")
    second = _circuit("second")
    scene = _scene(circuits=(first,))

    assert scene.add_circuit(second) is scene
    assert scene.circuits == [first, second]
    assert scene.clone().circuits == [first, second]
    assert prepare_scene(scene).circuits == [first, second]
    assert tuple(graph.circuit_name for graph in scene.compile_circuits()) == ("first", "second")


def test_scene_rejects_duplicate_or_non_circuit_entries():
    scene = _scene(circuits=(_circuit("network"),))
    with pytest.raises(ValueError, match="already present"):
        scene.add_circuit(_circuit("NETWORK"))
    with pytest.raises(TypeError, match="Circuit instances"):
        scene.add_circuit(object())


def test_public_circuit_api_is_exported_without_a_second_solver_entrypoint():
    for name in (
        "Circuit",
        "CircuitNode",
        "PortBinding",
        "MNAConfig",
        "VoltageSource",
        "CurrentSource",
        "VoltageControlledVoltageSource",
        "VoltageControlledCurrentSource",
        "CurrentControlledVoltageSource",
        "CurrentControlledCurrentSource",
        "MutualInductor",
        "TimedSwitch",
    ):
        assert name in mw.__all__
        assert getattr(mw, name).__module__ == "witwin.maxwell.circuits"
    assert not hasattr(mw, "CircuitSolver")


def test_mna_config_rejects_implicit_regularization_and_invalid_integration():
    with pytest.raises(ValueError, match="integration"):
        mw.MNAConfig(integration="gear")
    with pytest.raises(ValueError, match="regularization"):
        mw.MNAConfig(regularization="gmin")
    with pytest.raises(ValueError, match="positive integer"):
        mw.MNAConfig(dense_unknown_limit=0)


def test_all_public_devices_satisfy_the_circuit_device_protocol():
    circuit = mw.Circuit("protocol")
    first = circuit.node("first")
    second = circuit.node("second")
    devices = (
        mw.VoltageSource("V1", first, circuit.ground, 1.0),
        mw.CurrentSource("I1", first, circuit.ground, 1.0),
        mw.VoltageControlledVoltageSource("E1", first, circuit.ground, second, circuit.ground, 2.0),
        mw.VoltageControlledCurrentSource("G1", first, circuit.ground, second, circuit.ground, 2.0),
        mw.CurrentControlledVoltageSource("H1", first, circuit.ground, "V1", 2.0),
        mw.CurrentControlledCurrentSource("F1", first, circuit.ground, "V1", 2.0),
        mw.MutualInductor("K1", "L1", "L2", 0.5),
        mw.TimedSwitch("S1", first, circuit.ground, (1.0,)),
    )
    assert all(isinstance(device, mw.CircuitDevice) for device in devices)


def test_circuit_scene_is_fdtd_only_and_requires_a_bound_port_for_coupling():
    scene = _scene(circuits=(_circuit(),))
    with pytest.raises(ValueError, match=r"Simulation\.fdtd.*only"):
        mw.Simulation.fdfd(scene, frequency=1.0e9).prepare()
    with pytest.raises(NotImplementedError, match="one circuit with one bound port"):
        mw.Simulation.fdtd(scene, frequency=1.0e9).prepare()
