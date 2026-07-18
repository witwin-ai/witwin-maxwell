"""Contract tests for the Phase-0 nonlinear circuit device surfaces."""

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.circuit_devices import NONLINEAR_DEVICE_TYPES


def _circuit():
    circuit = mw.Circuit("nl")
    return circuit, circuit.node("a")


def test_public_exports_present():
    for name in (
        "Diode",
        "PiecewiseLinearIV",
        "PolynomialIV",
        "VoltageDependentCapacitor",
        "NonlinearSolveConfig",
        "BJT",
        "MOSFET",
    ):
        assert name in mw.__all__
        assert getattr(mw, name).__module__ == "witwin.maxwell.circuit_devices"


def test_devices_satisfy_circuit_device_protocol_and_carry_kind():
    circuit, a = _circuit()
    devices = (
        mw.Diode("d1", a, circuit.ground, saturation_current=1e-12, ideality=1.05),
        mw.PiecewiseLinearIV("w1", a, circuit.ground, voltages=[-1.0, 0.0, 1.0], currents=[-0.5, 0.0, 0.4]),
        mw.PolynomialIV("p1", a, circuit.ground, coefficients=[0.0, 1e-3, 2e-4]),
        mw.VoltageDependentCapacitor("c1", a, circuit.ground, q_coefficients=[0.0, 1e-12, 3e-13]),
    )
    for device in devices:
        assert isinstance(device, mw.CircuitDevice)
        assert device.terminals == (a, circuit.ground)
        assert device.initial_condition is None
        assert isinstance(device.parameters, dict) and device.parameters
    assert {type(device) for device in devices} == set(NONLINEAR_DEVICE_TYPES)


def test_circuit_add_admits_nonlinear_devices():
    circuit, a = _circuit()
    diode = mw.Diode("d1", a, circuit.ground, saturation_current=1e-12)
    assert circuit.add(diode) is circuit
    assert circuit.devices == (diode,)


def test_diode_parameter_validation():
    circuit, a = _circuit()
    with pytest.raises(ValueError, match="saturation_current must be positive"):
        mw.Diode("d", a, circuit.ground, saturation_current=0.0)
    with pytest.raises(ValueError, match="ideality must be positive"):
        mw.Diode("d", a, circuit.ground, saturation_current=1e-12, ideality=0.0)
    with pytest.raises(ValueError, match="series_resistance must be non-negative"):
        mw.Diode("d", a, circuit.ground, saturation_current=1e-12, series_resistance=-1.0)
    with pytest.raises(ValueError, match="temperature must be positive"):
        mw.Diode("d", a, circuit.ground, saturation_current=1e-12, temperature=0.0)


def test_piecewise_linear_requires_strictly_increasing_matched_knots():
    circuit, a = _circuit()
    with pytest.raises(ValueError, match="strictly increasing"):
        mw.PiecewiseLinearIV("w", a, circuit.ground, voltages=[0.0, 0.0, 1.0], currents=[0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="equal length"):
        mw.PiecewiseLinearIV("w", a, circuit.ground, voltages=[0.0, 1.0], currents=[0.0])
    with pytest.raises(ValueError, match="at least two knots"):
        mw.PiecewiseLinearIV("w", a, circuit.ground, voltages=[0.0], currents=[0.0])


def test_transistor_surfaces_fail_closed_until_phase_5():
    circuit, a = _circuit()
    with pytest.raises(NotImplementedError, match="Transistor device BJT is gated behind the independent Phase 5"):
        mw.BJT("q1", a, circuit.ground)
    with pytest.raises(NotImplementedError, match="Transistor device MOSFET is gated behind the independent Phase 5"):
        mw.MOSFET("m1", a, circuit.ground)


def test_diode_establishes_dc_path_but_varactor_does_not():
    # A diode conducts at DC: a diode-only network to ground compiles.
    conducting = mw.Circuit("conducting")
    node = conducting.node("a")
    conducting.add(mw.Diode("d1", node, conducting.ground, saturation_current=1e-12))
    graph = conducting.compile()
    assert graph.circuit_name == "conducting"

    # A voltage-dependent capacitor is charge-only (open at DC): a varactor-only
    # network leaves its node floating from ground, which must fail closed. This
    # falsifies the graph type-set registration -- if the varactor were wrongly
    # registered as DC-connecting this would pass.
    floating = mw.Circuit("floating")
    node = floating.node("a")
    floating.add(mw.VoltageDependentCapacitor("c1", node, floating.ground, q_coefficients=[0.0, 1e-12]))
    with pytest.raises(ValueError, match="no DC path"):
        floating.compile()


def test_standalone_mna_rejects_nonlinear_devices():
    # compile_mna_system fails closed before touching CUDA: a diode cannot be
    # assembled into the linear constant-conductance stamp, so the linear runtime
    # must reject it instead of silently dropping the device (fail-open).
    from witwin.maxwell.compiler.mna import compile_mna_system

    circuit = mw.Circuit("nl")
    node = circuit.node("a")
    circuit.add(mw.Resistor("r1", node, circuit.ground, resistance=1000.0))
    circuit.add(mw.Diode("d1", node, circuit.ground, saturation_current=1e-12))
    with pytest.raises(NotImplementedError, match="nonlinear terminal law"):
        compile_mna_system(circuit, dt=1e-12)


def test_coupled_mna_rejects_nonlinear_devices():
    from witwin.maxwell.compiler.mna import compile_coupled_mna_system

    circuit = mw.Circuit("nl")
    node = circuit.node("a")
    circuit.add(mw.Resistor("r1", node, circuit.ground, resistance=1000.0))
    circuit.add(mw.Diode("d1", node, circuit.ground, saturation_current=1e-12))
    circuit.bind_port("p1", positive=node, negative=circuit.ground)
    with pytest.raises(NotImplementedError, match="nonlinear terminal law"):
        compile_coupled_mna_system(circuit, dt=1e-12)


def test_scene_circuit_prepare_rejects_nonlinear_devices():
    # This is the exact entry the FDTD circuit prepare path uses
    # (Simulation._validate_circuit_execution -> scene.compile_circuits()).
    from types import SimpleNamespace

    from witwin.maxwell.compiler.circuits import compile_circuits

    circuit = mw.Circuit("nl")
    node = circuit.node("a")
    circuit.add(mw.Resistor("r1", node, circuit.ground, resistance=1000.0))
    circuit.add(mw.Diode("d1", node, circuit.ground, saturation_current=1e-12))
    scene = SimpleNamespace(circuits=(circuit,), ports=())
    with pytest.raises(NotImplementedError, match="nonlinear terminal law"):
        compile_circuits(scene)


def test_nonlinear_solve_config_validation():
    with pytest.raises(ValueError, match="absolute_tolerance must be positive"):
        mw.NonlinearSolveConfig(absolute_tolerance=0.0)
    with pytest.raises(ValueError, match="max_iterations must be a positive integer"):
        mw.NonlinearSolveConfig(max_iterations=0)
    with pytest.raises(ValueError, match="line_search"):
        mw.NonlinearSolveConfig(line_search="newton")
    with pytest.raises(ValueError, match="failure"):
        mw.NonlinearSolveConfig(failure="continue")
