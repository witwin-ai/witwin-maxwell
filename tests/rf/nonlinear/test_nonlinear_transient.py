"""Phase-1 gates: DC I-V, rectifier golden RMS, per-step KCL, multistart, charge.

These exercise the standalone nonlinear transient path (Newton in the transient
loop, gmin/source DC continuation, PWL-kink backward-Euler breakpoints) with no
FDTD coupling. The frozen rectifier goldens under ``goldens/`` are an oversampled,
differently-integrated reference (see ``_generate_goldens.py``).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.nonlinear_devices import (
    NonlinearMNASystem,
    compile_nonlinear_devices,
    multistart_dc,
    run_nonlinear_transient,
    solve_dc_operating_point,
)

import _rectifier_fixtures as fx

GOLDEN_PATH = Path(__file__).resolve().parent / "goldens" / "rectifier_goldens.pt"


@pytest.fixture(autouse=True)
def _default_float64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _diode_divider(source_voltage, resistance, saturation_current, ideality=1.0):
    circuit = mw.Circuit("div")
    node = circuit.node("a")
    diode = mw.Diode(
        "d1", node, circuit.ground, saturation_current=saturation_current, ideality=ideality
    )
    compiled = compile_nonlinear_devices([diode], {"a": 0}, dtype=torch.float64, device="cpu")
    conductance = torch.tensor([[1.0 / resistance]], dtype=torch.float64)
    injection = torch.tensor([source_voltage / resistance], dtype=torch.float64)
    system = NonlinearMNASystem(1, conductance, injection, compiled)
    vte = compiled[0].parameters["thermal_voltage"].item()
    return system, vte


def _bisection_root(source_voltage, resistance, saturation_current, vte):
    def residual(v):
        return (v - source_voltage) / resistance + saturation_current * math.expm1(v / vte)

    lo, hi = -10.0, 10.0
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# Gate: diode I-V rel err < 1e-5 vs analytic Shockley across a bias sweep.
# --------------------------------------------------------------------------- #
def test_diode_dc_iv_matches_analytic_shockley_below_1e_5():
    resistance = 1.0e3
    saturation_current = 1.0e-12
    worst = 0.0
    for source_voltage in torch.linspace(-2.0, 2.0, 41).tolist():
        system, vte = _diode_divider(source_voltage, resistance, saturation_current)
        solution, stats = solve_dc_operating_point(
            system, system.injection.clone(), mw.NonlinearSolveConfig()
        )
        assert stats.converged
        v = solution.item()
        root = _bisection_root(source_voltage, resistance, saturation_current, vte)
        # Solved diode current vs the analytic Shockley current at the solved bias.
        solved_current = saturation_current * math.expm1(v / vte)
        analytic_current = saturation_current * math.expm1(root / vte)
        denom = max(abs(analytic_current), 1e-12)
        worst = max(worst, abs(solved_current - analytic_current) / denom)
    assert worst < 1.0e-5, worst


# --------------------------------------------------------------------------- #
# Gate: gmin continuation converges a cold hard-forward start and leaves gmin=0.
# --------------------------------------------------------------------------- #
def test_gmin_continuation_converges_hard_start_and_leaves_no_residual_gmin():
    # 8 V across 5 ohm hammers the junction; a cold Newton start from 0 with no
    # continuation would take an enormous first step. gmin continuation ramps it.
    system, vte = _diode_divider(8.0, 5.0, 1e-14)
    config = mw.NonlinearSolveConfig(gmin_start=1.0, gmin_steps=6)
    solution, stats = solve_dc_operating_point(system, system.injection.clone(), config)
    assert stats.converged
    # The returned state must satisfy the true (gmin-free) residual: the last
    # continuation rung is gmin=0, so no artificial shunt survives.
    residual = torch.linalg.vector_norm(system.true_residual(solution))
    assert residual.item() <= 1e-7 * stats.residual_scale + 1e-10
    root = _bisection_root(8.0, 5.0, 1e-14, vte)
    assert abs(solution.item() - root) / abs(root) <= 1e-10


# --------------------------------------------------------------------------- #
# Gate: rectifier normalized RMS < 1% vs frozen oversampled golden.
# --------------------------------------------------------------------------- #
def _normalized_rms(candidate, golden, mask):
    error = (candidate - golden)[mask]
    reference = golden[mask]
    return (torch.linalg.vector_norm(error) / torch.linalg.vector_norm(reference)).item()


@pytest.mark.skipif(not GOLDEN_PATH.exists(), reason="rectifier goldens not generated")
@pytest.mark.parametrize(
    "builder_name, golden_key",
    [
        ("build_half_wave_system", "half_wave_output"),
        ("build_full_wave_system", "full_wave_output"),
    ],
)
def test_rectifier_normalized_rms_below_1_percent(builder_name, golden_key):
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    system, source_injection, output_index = getattr(fx, builder_name)()
    times = fx.coarse_times()
    result = run_nonlinear_transient(
        system, times, source_injection, mw.NonlinearSolveConfig(), integration="trapezoidal"
    )
    output = result.node_voltages[:, output_index]
    # Compare over the final period (the steady ripple), where the golden and the
    # graded run are both past the DC-to-AC startup transient.
    last_period_start = times[-1] - 1.0 / fx.FREQUENCY
    mask = times >= last_period_start
    rms = _normalized_rms(output, golden[golden_key], mask)
    assert rms < 0.01, (golden_key, rms)


# --------------------------------------------------------------------------- #
# Gate: every converged transient step satisfies the KCL residual tolerance.
# --------------------------------------------------------------------------- #
def test_every_converged_transient_step_satisfies_kcl():
    system, source_injection, _ = fx.build_full_wave_system()
    times = fx.coarse_times()
    config = mw.NonlinearSolveConfig()
    result = run_nonlinear_transient(system, times, source_injection, config, integration="trapezoidal")
    # Worst per-step KCL residual over the whole run must clear the residual gate
    # for a representative current scale (~ AMPLITUDE / SOURCE_RESISTANCE).
    scale = fx.AMPLITUDE / fx.SOURCE_RESISTANCE
    tolerance = config.relative_tolerance * scale + config.absolute_tolerance
    assert float(result.kcl_residual_norm.max()) <= tolerance


# --------------------------------------------------------------------------- #
# Gate: multistart reports multistability on an N-shaped PWL device.
# --------------------------------------------------------------------------- #
def _tunnel_divider(source_voltage, resistance):
    circuit = mw.Circuit("tunnel")
    node = circuit.node("a")
    # N-shaped I-V with a negative-differential-resistance middle segment.
    voltages = torch.tensor([0.0, 0.10, 0.40, 0.60], dtype=torch.float64)
    currents = torch.tensor([0.0, 10.0e-3, 1.0e-3, 6.0e-3], dtype=torch.float64)
    device = mw.PiecewiseLinearIV("j1", node, circuit.ground, voltages=voltages, currents=currents)
    compiled = compile_nonlinear_devices([device], {"a": 0}, dtype=torch.float64, device="cpu")
    conductance = torch.tensor([[1.0 / resistance]], dtype=torch.float64)
    injection = torch.tensor([source_voltage / resistance], dtype=torch.float64)
    return NonlinearMNASystem(1, conductance, injection, compiled)


def test_multistart_reports_multiple_operating_points():
    # A load line that cuts the N-shaped curve three times: three DC operating
    # points, so a single seed can only report the branch it lands in.
    system = _tunnel_divider(source_voltage=0.80, resistance=100.0)
    seeds = [torch.tensor([v]) for v in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)]
    report = multistart_dc(system, system.injection.clone(), seeds, mw.NonlinearSolveConfig())
    assert report.num_operating_points >= 2
    # At least two distinct seeds must resolve to distinct operating points.
    resolved = {root for root in report.seed_to_root if root >= 0}
    assert len(resolved) >= 2
    # Each reported root is a genuine operating point (true KCL residual ~ 0).
    for root in report.roots:
        assert float(torch.linalg.vector_norm(system.true_residual(root))) <= 1e-9


def test_monostable_bias_reports_single_operating_point():
    # Biased in the first positive-slope segment the operating point is unique;
    # multistart must report exactly one even from scattered seeds.
    system = _tunnel_divider(source_voltage=0.03, resistance=200.0)
    seeds = [torch.tensor([v]) for v in (0.0, 0.02, 0.05, 0.1, 0.2)]
    report = multistart_dc(system, system.injection.clone(), seeds, mw.NonlinearSolveConfig())
    assert report.num_operating_points == 1


# --------------------------------------------------------------------------- #
# Gate: linear Q(V) capacitor transient == analytic RC decay (charge companion).
# --------------------------------------------------------------------------- #
def test_linear_charge_companion_matches_analytic_rc_decay():
    # A VoltageDependentCapacitor with q(v) = C v discharging through R must
    # reproduce v(t) = v0 exp(-t/RC): the exact linear-capacitor companion, which
    # pins the nonlinear charge integrator to the native reactive convention.
    capacitance = 1.0e-9
    resistance = 1.0e3
    v0 = 2.0
    circuit = mw.Circuit("rc")
    node = circuit.node("a")
    cap = mw.VoltageDependentCapacitor(
        "c1", node, circuit.ground, q_coefficients=torch.tensor([0.0, capacitance])
    )
    compiled = compile_nonlinear_devices([cap], {"a": 0}, dtype=torch.float64, device="cpu")
    conductance = torch.tensor([[1.0 / resistance]], dtype=torch.float64)
    injection = torch.zeros(1, dtype=torch.float64)
    system = NonlinearMNASystem(1, conductance, injection, compiled)

    tau = resistance * capacitance
    times = torch.linspace(0.0, 5.0 * tau, 501, dtype=torch.float64)
    result = run_nonlinear_transient(
        system,
        times,
        lambda t: injection.clone(),
        mw.NonlinearSolveConfig(),
        integration="trapezoidal",
        initial_state=torch.tensor([v0]),
    )
    analytic = v0 * torch.exp(-times / tau)
    rms = (
        torch.linalg.vector_norm(result.node_voltages[:, 0] - analytic)
        / torch.linalg.vector_norm(analytic)
    ).item()
    assert rms < 1e-4, rms


# --------------------------------------------------------------------------- #
# Gate: a PWL kink crossing inside a step triggers a backward-Euler breakpoint.
# --------------------------------------------------------------------------- #
def test_pwl_knot_crossing_triggers_backward_euler_breakpoint():
    # A ramped source sweeps a PWL device's terminal voltage across its knots; the
    # transient must localize a backward-Euler step whenever a knot is crossed.
    circuit = mw.Circuit("pwl_ramp")
    node = circuit.node("a")
    voltages = torch.tensor([-1.0, 0.0, 0.2, 0.4, 1.0], dtype=torch.float64)
    currents = torch.tensor([-1.0e-3, 0.0, 0.5e-3, 3.0e-3, 9.0e-3], dtype=torch.float64)
    device = mw.PiecewiseLinearIV("j1", node, circuit.ground, voltages=voltages, currents=currents)
    cap = mw.VoltageDependentCapacitor(
        "c1", node, circuit.ground, q_coefficients=torch.tensor([0.0, 1.0e-9])
    )
    compiled = compile_nonlinear_devices([device, cap], {"a": 0}, dtype=torch.float64, device="cpu")
    resistance = 100.0
    conductance = torch.tensor([[1.0 / resistance]], dtype=torch.float64)
    system = NonlinearMNASystem(1, torch.zeros(1, 1), torch.zeros(1), compiled)
    system.conductance = conductance

    times = torch.linspace(0.0, 1.0e-6, 201, dtype=torch.float64)

    def ramp(t):
        return torch.tensor([(1.5 * (t / 1.0e-6)) / resistance], dtype=torch.float64)

    result = run_nonlinear_transient(system, times, ramp, mw.NonlinearSolveConfig(), integration="trapezoidal")
    assert len(result.breakpoint_steps) >= 1
    assert torch.isfinite(result.node_voltages).all()
