import pytest

import witwin.maxwell as mw
from witwin.core import Box
from witwin.maxwell.compiler import compile_circuits
from witwin.maxwell.scene import prepare_scene


def _port(name, *, termination=None, x=0.5):
    return mw.LumpedPort(
        name,
        positive=(x, 0.5, 0.75),
        negative=(x, 0.5, 0.25),
        voltage_path=mw.AxisPath("z"),
        current_surface=Box(position=(x, 0.5, 0.5), size=(0.25, 0.25, 0.0)),
        termination=termination,
    )


def test_graph_ground_node_and_branch_order_are_deterministic():
    circuit = mw.circuits.Circuit("network")
    first = circuit.node("first")
    second = circuit.node("second")
    third = circuit.node("third")
    circuit.add(mw.circuits.VoltageSource("V1", first, circuit.ground, 1.0))
    circuit.add(mw.rf.Resistor("R1", first, second, 50.0))
    circuit.add(mw.rf.Inductor("L1", second, third, 1.0e-9))
    circuit.add(mw.circuits.VoltageControlledVoltageSource("E1", third, circuit.ground, first, circuit.ground, 2.0))

    graph = circuit.compile()

    assert tuple(node.name for node in graph.nodes) == ("0", "first", "second", "third")
    assert dict(graph.node_index) == {"0": 0, "first": 1, "second": 2, "third": 3}
    assert graph.branch_names == ("V1", "L1", "E1")
    assert dict(graph.branch_index) == {"V1": 0, "L1": 1, "E1": 2}
    assert graph.unknown_count == 6


def test_controlled_branch_and_mutual_inductor_dependencies_resolve_case_insensitively():
    circuit = mw.circuits.Circuit("network")
    p = circuit.node("p")
    q = circuit.node("q")
    r = circuit.node("r")
    circuit.add(mw.circuits.VoltageSource("Vsense", p, circuit.ground, 0.0))
    circuit.add(mw.rf.Inductor("L1", p, q, 1.0e-9))
    circuit.add(mw.rf.Inductor("L2", r, circuit.ground, 2.0e-9))
    circuit.add(mw.circuits.CurrentControlledCurrentSource("F1", q, circuit.ground, "vsense", 2.0))
    circuit.add(mw.circuits.MutualInductor("K1", "l1", "L2", 0.5))

    graph = circuit.compile()

    assert dict(graph.source_dependencies) == {
        "F1": ("Vsense",),
        "K1": ("L1", "L2"),
    }


def test_missing_control_branch_and_invalid_mutual_reference_are_precise_errors():
    circuit = mw.circuits.Circuit("network")
    node = circuit.node("node")
    circuit.add(mw.rf.Resistor("R1", node, circuit.ground, 1.0))
    circuit.add(mw.circuits.CurrentControlledCurrentSource("F1", node, circuit.ground, "missing", 1.0))
    with pytest.raises(ValueError, match="F1.*missing branch.*missing"):
        circuit.compile()

    mutual = mw.circuits.Circuit("mutual")
    node = mutual.node("node")
    mutual.add(mw.circuits.VoltageSource("V1", node, mutual.ground, 1.0))
    mutual.add(mw.circuits.MutualInductor("K1", "V1", "missing", 0.5))
    with pytest.raises(ValueError, match="K1.*non-inductor.*V1"):
        mutual.compile()


def test_floating_current_source_cutset_and_unused_ground_are_rejected():
    circuit = mw.circuits.Circuit("floating")
    first = circuit.node("first")
    second = circuit.node("second")
    circuit.add(mw.circuits.CurrentSource("I1", first, second, 1.0))
    circuit.add(mw.rf.Resistor("R1", first, second, 1.0))

    with pytest.raises(ValueError, match="no DC path.*ground"):
        circuit.compile()


def test_ideal_voltage_source_loop_is_rejected():
    circuit = mw.circuits.Circuit("loop")
    first = circuit.node("first")
    second = circuit.node("second")
    circuit.add(mw.circuits.VoltageSource("V1", first, circuit.ground, 1.0))
    circuit.add(mw.circuits.VoltageSource("V2", second, first, 1.0))
    circuit.add(mw.circuits.VoltageSource("V3", circuit.ground, second, -2.0))

    with pytest.raises(ValueError, match="voltage-source loop.*V3"):
        circuit.compile()


def test_scene_binding_accepts_only_existing_lumped_or_terminal_ports():
    circuit = mw.circuits.Circuit("bound")
    signal = circuit.node("signal")
    circuit.add(mw.rf.Resistor("R1", signal, circuit.ground, 50.0))
    circuit.bind_port("feed", positive=signal, negative=circuit.ground)
    scene = mw.Scene(circuits=(circuit,))

    with pytest.raises(ValueError, match="missing EM ports.*feed"):
        compile_circuits(prepare_scene(scene, device="cpu"))


def test_port_bindings_supply_norton_reference_and_match_names_exactly():
    circuit = mw.circuits.Circuit("two_port")
    first = circuit.node("first")
    second = circuit.node("second")
    circuit.add(mw.rf.Resistor("R1", first, second, 50.0))
    circuit.bind_port("input", positive=first, negative=circuit.ground)
    circuit.bind_port("output", positive=second, negative=circuit.ground)
    scene = mw.Scene(
        ports=(_port("input", x=0.4), _port("output", x=0.6)),
        circuits=(circuit,),
    )

    assert compile_circuits(prepare_scene(scene, device="cpu"))[0].bindings == circuit.bindings

    mismatched = mw.circuits.Circuit("mismatched")
    node = mismatched.node("node")
    mismatched.add(mw.rf.Resistor("R1", node, mismatched.ground, 50.0))
    mismatched.bind_port("INPUT", positive=node, negative=mismatched.ground)
    with pytest.raises(ValueError, match="missing EM ports.*INPUT"):
        compile_circuits(prepare_scene(mw.Scene(ports=(_port("input"),), circuits=(mismatched,)), device="cpu"))


def test_circuit_binding_rejects_an_existing_local_port_termination():
    circuit = mw.circuits.Circuit("bound")
    node = circuit.node("node")
    circuit.add(mw.rf.Resistor("R1", node, circuit.ground, 50.0))
    circuit.bind_port("feed", positive=node, negative=circuit.ground)
    scene = mw.Scene(
        ports=(_port("feed", termination=mw.rf.SeriesRLC(r=50.0)),),
        circuits=(circuit,),
    )

    with pytest.raises(ValueError, match="both a local termination and a circuit binding"):
        compile_circuits(prepare_scene(scene, device="cpu"))


def test_dense_unknown_limit_is_an_explicit_error_not_a_cpu_fallback():
    circuit = mw.circuits.Circuit("large", config=mw.circuits.MNAConfig(dense_unknown_limit=1))
    node = circuit.node("node")
    circuit.add(mw.circuits.VoltageSource("V1", node, circuit.ground, 1.0))

    with pytest.raises(ValueError, match="2 unknowns.*GPU dense limit 1"):
        circuit.compile()
