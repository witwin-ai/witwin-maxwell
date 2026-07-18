"""Phase-0 Newton-core gates: analytic root, dual gate, deterministic failure."""

import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.nonlinear_devices import (
    NonlinearConvergenceError,
    NonlinearDeviceError,
    NonlinearMNASystem,
    _dual_gate_converged,
    compile_nonlinear_devices,
    newton_solve,
)

torch.set_default_dtype(torch.float64)

ROOT_TOL = 1.0e-10


def _diode_divider(*, source_voltage, resistance, saturation_current, ideality=1.0):
    """One-node diode-resistor divider: R (Norton from Vs) with a diode to ground."""

    circuit = mw.Circuit("div")
    node = circuit.node("a")
    diode = mw.Diode("d1", node, circuit.ground, saturation_current=saturation_current, ideality=ideality)
    compiled = compile_nonlinear_devices([diode], {"a": 0}, dtype=torch.float64, device="cpu")
    conductance = torch.tensor([[1.0 / resistance]], dtype=torch.float64)
    injection = torch.tensor([source_voltage / resistance], dtype=torch.float64)
    system = NonlinearMNASystem(1, conductance, injection, compiled)
    vte = compiled[0].parameters["thermal_voltage"].item()
    return system, vte


def _bisection_root(source_voltage, resistance, saturation_current, vte):
    """Independent monotone-bisection oracle for the divider node voltage."""

    def residual(v):
        return (v - source_voltage) / resistance + saturation_current * math.expm1(v / vte)

    lo, hi = -5.0, 5.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def test_newton_reaches_analytic_root_to_1e_10():
    system, vte = _diode_divider(source_voltage=1.0, resistance=1000.0, saturation_current=1e-12)
    solution, stats = newton_solve(system, torch.zeros(1), mw.NonlinearSolveConfig())
    root = _bisection_root(1.0, 1000.0, 1e-12, vte)
    rel_err = abs(solution.item() - root) / abs(root)
    assert stats.converged
    assert rel_err <= ROOT_TOL


def test_pnjlim_keeps_hard_forward_drive_finite_and_accurate():
    # 5 V across a small resistor drives the junction hard; without pnjlim the
    # first Newton step would overflow exp(5 / Vt). Limiting keeps it finite.
    system, vte = _diode_divider(source_voltage=5.0, resistance=10.0, saturation_current=1e-14)
    solution, stats = newton_solve(system, torch.zeros(1), mw.NonlinearSolveConfig())
    assert torch.isfinite(solution).all()
    root = _bisection_root(5.0, 10.0, 1e-14, vte)
    assert abs(solution.item() - root) / abs(root) <= ROOT_TOL


def test_newton_solves_a_small_two_node_ladder():
    # Vs -R1- (a) -R2- (b) -diode- gnd, with a diode also from (a) to gnd.
    circuit = mw.Circuit("ladder")
    a, b = circuit.node("a"), circuit.node("b")
    d_a = mw.Diode("da", a, circuit.ground, saturation_current=1e-12)
    d_b = mw.Diode("db", b, circuit.ground, saturation_current=1e-12)
    compiled = compile_nonlinear_devices([d_a, d_b], {"a": 0, "b": 1}, dtype=torch.float64, device="cpu")
    r1, r2, vs = 500.0, 800.0, 1.2
    conductance = torch.tensor([[1.0 / r1 + 1.0 / r2, -1.0 / r2], [-1.0 / r2, 1.0 / r2]], dtype=torch.float64)
    injection = torch.tensor([vs / r1, 0.0], dtype=torch.float64)
    system = NonlinearMNASystem(2, conductance, injection, compiled)
    solution, stats = newton_solve(system, torch.zeros(2), mw.NonlinearSolveConfig())
    assert stats.converged
    residual = system.true_residual(solution)
    assert torch.linalg.vector_norm(residual).item() <= stats.residual_scale * 1e-7 + 1e-10


def test_dual_gate_rejects_single_gate_false_convergence():
    config = mw.NonlinearSolveConfig()
    scale = torch.tensor(1.0)
    solution_norm = torch.tensor(1.0)
    # residual gate passes, update gate fails -> not converged.
    assert not bool(_dual_gate_converged(torch.tensor(1e-11), torch.tensor(1.0), scale, solution_norm, config))
    # update gate passes, residual gate fails -> not converged.
    assert not bool(_dual_gate_converged(torch.tensor(1.0), torch.tensor(1e-13), scale, solution_norm, config))
    # both pass -> converged.
    assert bool(_dual_gate_converged(torch.tensor(1e-11), torch.tensor(1e-13), scale, solution_norm, config))


def test_capped_iteration_raises_deterministically_with_localization():
    system, _ = _diode_divider(source_voltage=1.0, resistance=1000.0, saturation_current=1e-12)
    config = mw.NonlinearSolveConfig(max_iterations=3)
    errors = []
    for _ in range(2):
        with pytest.raises(NonlinearConvergenceError) as excinfo:
            newton_solve(system, torch.zeros(1), config)
        errors.append(excinfo.value.stats)
    first, second = errors
    assert first.iterations == 3 == second.iterations
    assert first.residual_trajectory == second.residual_trajectory
    assert first.worst_node == 0
    assert not first.converged


def test_record_and_stop_returns_unconverged_state_without_raising():
    system, _ = _diode_divider(source_voltage=1.0, resistance=1000.0, saturation_current=1e-12)
    config = mw.NonlinearSolveConfig(max_iterations=3, failure="record_and_stop")
    solution, stats = newton_solve(system, torch.zeros(1), config)
    assert not stats.converged
    assert stats.iterations == 3
    assert torch.isfinite(solution).all()


def test_rootless_system_fails_closed():
    # A big diode driven by a current source demanding -2*Is has no root
    # (i(v) >= -Is); the solver must fail closed, never return a wrong iterate.
    circuit = mw.Circuit("rootless")
    node = circuit.node("a")
    diode = mw.Diode("d", node, circuit.ground, saturation_current=1.0)
    compiled = compile_nonlinear_devices([diode], {"a": 0}, dtype=torch.float64, device="cpu")
    system = NonlinearMNASystem(1, torch.zeros(1, 1), torch.tensor([-2.0]), compiled)
    with pytest.raises((NonlinearConvergenceError, NonlinearDeviceError)):
        newton_solve(system, torch.zeros(1), mw.NonlinearSolveConfig())


def test_non_finite_device_output_is_localized():
    circuit = mw.Circuit("nan")
    node = circuit.node("a")
    diode = mw.Diode("d", node, circuit.ground, saturation_current=1e-12)
    compiled = compile_nonlinear_devices([diode], {"a": 0}, dtype=torch.float64, device="cpu")
    system = NonlinearMNASystem(1, torch.zeros(1, 1), torch.zeros(1), compiled)
    with pytest.raises(NonlinearDeviceError, match="non-finite"):
        system.true_residual(torch.tensor([float("nan")]))
