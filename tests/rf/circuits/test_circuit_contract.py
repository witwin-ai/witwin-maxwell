import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.scene import BoundarySpec, Domain, GridSpec, prepare_scene
from witwin.maxwell.compiler import compile_circuits, compile_lumped_elements


def _scene(*, ports=(), circuits=()):
    return mw.Scene(
        domain=Domain(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
        grid=GridSpec.uniform(0.25),
        boundary=BoundarySpec.none(),
        ports=ports,
        circuits=circuits,
    )


def _circuit(name="network"):
    circuit = mw.circuits.Circuit(name)
    node = circuit.node("signal")
    circuit.add(mw.rf.Resistor("R1", node, circuit.ground, 50.0))
    return circuit


def test_circuit_nodes_are_interned_ordered_and_have_unique_ground():
    circuit = mw.circuits.Circuit("network")
    first = circuit.node("first")
    second = circuit.node("second")

    assert circuit.node("0") is circuit.ground
    assert circuit.node("first") is first
    assert circuit.node("FIRST") is first
    assert circuit.nodes == (circuit.ground, first, second)
    assert circuit.ground.is_ground


def test_shared_rlc_types_accept_circuit_nodes_without_losing_tensor_identity():
    circuit = mw.circuits.Circuit("network")
    positive = circuit.node("positive")
    resistance = torch.tensor(50.0, dtype=torch.float64, requires_grad=True)
    resistor = mw.rf.Resistor("R1", positive, circuit.ground, resistance)

    circuit.add(resistor)

    assert circuit.devices == (resistor,)
    assert resistor.positive is positive
    assert resistor.negative is circuit.ground
    assert resistor.resistance is resistance
    assert resistor.resistance.requires_grad
    assert isinstance(resistor, mw.circuits.CircuitDevice)
    assert resistor.terminals == (positive, circuit.ground)
    assert resistor.parameters == {"resistance": resistance}
    assert resistor.initial_condition is None


def test_scene_local_rlc_coordinate_contract_is_unchanged():
    resistor = mw.rf.Resistor("R1", (0.75, 0.5, 0.5), (0.25, 0.5, 0.5), 50.0)
    scene = _scene()

    scene = scene.updated(lumped_elements=(*scene.lumped_elements, resistor))
    assert scene.lumped_elements == (resistor,)
    assert compile_lumped_elements(prepare_scene(scene, device="cpu"))[0].element_name == "R1"


def test_circuit_rejects_foreign_nodes_duplicate_devices_and_invalid_bindings():
    circuit = mw.circuits.Circuit("network")
    foreign = mw.circuits.Circuit("other").node("signal")
    signal = circuit.node("signal")

    with pytest.raises(ValueError, match="does not belong"):
        circuit.add(mw.rf.Resistor("R1", foreign, circuit.ground, 50.0))

    circuit.add(mw.rf.Resistor("R1", signal, circuit.ground, 50.0))
    with pytest.raises(ValueError, match="already present"):
        circuit.add(mw.rf.Capacitor("r1", signal, circuit.ground, 1.0e-12))
    with pytest.raises(ValueError, match="distinct"):
        circuit.bind_port("feed", positive=signal, negative=signal)


def test_scene_add_clone_prepare_and_compile_preserve_circuit_order():
    first = _circuit("first")
    second = _circuit("second")
    scene = _scene(circuits=(first,))

    scene = scene.updated(circuits=(*scene.circuits, second))
    assert scene.circuits == (first, second)
    assert scene.circuits == (first, second)
    assert prepare_scene(scene, device="cpu").scene.circuits == (first, second)
    assert tuple(graph.circuit_name for graph in compile_circuits(prepare_scene(scene, device="cpu"))) == ("first", "second")


def test_scene_rejects_duplicate_or_non_circuit_entries():
    scene = _scene(circuits=(_circuit("network"),))
    with pytest.raises(ValueError, match="already present"):
        scene = scene.updated(circuits=(*scene.circuits, _circuit("NETWORK")))
    with pytest.raises(TypeError, match="Circuit instances"):
        scene = scene.updated(circuits=(*scene.circuits, object()))


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
        assert name in mw.circuits.__all__
        assert getattr(mw.circuits, name).__module__ == "witwin.maxwell.circuits"
    assert not hasattr(mw, "CircuitSolver")


def test_mna_config_rejects_implicit_regularization_and_invalid_integration():
    with pytest.raises(ValueError, match="integration"):
        mw.circuits.MNAConfig(integration="gear")
    with pytest.raises(ValueError, match="regularization"):
        mw.circuits.MNAConfig(regularization="gmin")
    with pytest.raises(ValueError, match="positive integer"):
        mw.circuits.MNAConfig(dense_unknown_limit=0)


def test_all_public_devices_satisfy_the_circuit_device_protocol():
    circuit = mw.circuits.Circuit("protocol")
    first = circuit.node("first")
    second = circuit.node("second")
    devices = (
        mw.circuits.VoltageSource("V1", first, circuit.ground, 1.0),
        mw.circuits.CurrentSource("I1", first, circuit.ground, 1.0),
        mw.circuits.VoltageControlledVoltageSource("E1", first, circuit.ground, second, circuit.ground, 2.0),
        mw.circuits.VoltageControlledCurrentSource("G1", first, circuit.ground, second, circuit.ground, 2.0),
        mw.circuits.CurrentControlledVoltageSource("H1", first, circuit.ground, "V1", 2.0),
        mw.circuits.CurrentControlledCurrentSource("F1", first, circuit.ground, "V1", 2.0),
        mw.circuits.MutualInductor("K1", "L1", "L2", 0.5),
        mw.circuits.TimedSwitch("S1", first, circuit.ground, (1.0,)),
    )
    assert all(isinstance(device, mw.circuits.CircuitDevice) for device in devices)


def test_circuit_scene_is_fdtd_only_and_requires_a_bound_port_for_coupling():
    scene = _scene(circuits=(_circuit(),))
    with pytest.raises(ValueError, match=r"Simulation\.fdtd.*only"):
        mw.Simulation.fdfd(scene, frequencies=1.0e9, config=mw.FDFDConfig(device="cpu")).prepare()
    with pytest.raises(NotImplementedError, match="one circuit with at least one bound port"):
        mw.Simulation.fdtd(scene, frequencies=1.0e9, config=mw.FDTDConfig(device="cpu")).prepare()

    multiple = _scene(circuits=(_circuit("first"), _circuit("second")))
    with pytest.raises(NotImplementedError, match="multi-circuit execution"):
        mw.Simulation.fdtd(multiple, frequencies=1.0e9, config=mw.FDTDConfig(device="cpu")).prepare()
