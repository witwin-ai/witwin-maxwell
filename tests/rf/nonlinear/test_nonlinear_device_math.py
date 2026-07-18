"""Phase-0 device I/Q first-derivative accuracy gate (< 1e-6 vs complex-step)."""

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.nonlinear_devices import compile_nonlinear_devices

DERIVATIVE_TOL = 1.0e-6


def _compiled(device):
    circuit = mw.Circuit("m")
    node = circuit.node("a")
    # Rebuild the device against this circuit's nodes so terminals resolve.
    return compile_nonlinear_devices([device], {"a": 0}, dtype=torch.float64, device="cpu")[0]


def _complex_step(value_fn, v0):
    h = 1.0e-30
    vc = torch.tensor([v0 + 1j * h], dtype=torch.complex128)
    return value_fn(vc).imag.item() / h


def _rel_err(approx, exact):
    return abs(approx - exact) / max(abs(exact), 1.0)


@pytest.mark.parametrize("v0", [-0.2, 0.0, 0.3, 0.55])
def test_diode_conduction_and_charge_derivatives(v0):
    circuit = mw.Circuit("m")
    node = circuit.node("a")
    diode = mw.Diode("d1", node, circuit.ground, saturation_current=1e-12, ideality=1.03, junction_capacitance=0.4e-12)
    dev = compile_nonlinear_devices([diode], {"a": 0}, dtype=torch.float64, device="cpu")[0]

    g_cs = _complex_step(lambda v: dev.conduction(v)[0], v0)
    g_analytic = dev.conduction(torch.tensor([v0], dtype=torch.float64))[1].item()
    assert _rel_err(g_cs, g_analytic) < DERIVATIVE_TOL

    c_cs = _complex_step(lambda v: dev.charge(v)[0], v0)
    c_analytic = dev.charge(torch.tensor([v0], dtype=torch.float64))[1].item()
    assert _rel_err(c_cs, c_analytic) < DERIVATIVE_TOL


@pytest.mark.parametrize("v0", [-0.8, 0.1, 0.9])
def test_polynomial_iv_derivative(v0):
    circuit = mw.Circuit("m")
    node = circuit.node("a")
    poly = mw.PolynomialIV("p", node, circuit.ground, coefficients=[0.1, 0.5, -0.2, 0.05])
    dev = compile_nonlinear_devices([poly], {"a": 0}, dtype=torch.float64, device="cpu")[0]
    g_cs = _complex_step(lambda v: dev.conduction(v)[0], v0)
    g_analytic = dev.conduction(torch.tensor([v0], dtype=torch.float64))[1].item()
    assert _rel_err(g_cs, g_analytic) < DERIVATIVE_TOL


@pytest.mark.parametrize("v0", [-0.5, 0.5, 1.5])
def test_piecewise_linear_derivative_matches_segment_slope(v0):
    circuit = mw.Circuit("m")
    node = circuit.node("a")
    pwl = mw.PiecewiseLinearIV(
        "w", node, circuit.ground, voltages=[-1.0, 0.0, 1.0, 2.0], currents=[-0.5, 0.0, 0.3, 1.2]
    )
    dev = compile_nonlinear_devices([pwl], {"a": 0}, dtype=torch.float64, device="cpu")[0]
    # Complex step stays inside a segment (real part locates the segment).
    g_cs = _complex_step(lambda v: dev.conduction(v)[0], v0)
    g_analytic = dev.conduction(torch.tensor([v0], dtype=torch.float64))[1].item()
    assert _rel_err(g_cs, g_analytic) < DERIVATIVE_TOL
    # Central finite difference away from knots agrees too.
    step = 1e-6
    fd = (
        dev.conduction(torch.tensor([v0 + step], dtype=torch.float64))[0].item()
        - dev.conduction(torch.tensor([v0 - step], dtype=torch.float64))[0].item()
    ) / (2 * step)
    assert _rel_err(fd, g_analytic) < DERIVATIVE_TOL


@pytest.mark.parametrize("v0", [-0.4, 0.0, 0.6])
def test_varactor_charge_derivative(v0):
    circuit = mw.Circuit("m")
    node = circuit.node("a")
    varactor = mw.VoltageDependentCapacitor(
        "c", node, circuit.ground, q_coefficients=[0.0, 1e-12, 3e-13, -1e-13]
    )
    dev = compile_nonlinear_devices([varactor], {"a": 0}, dtype=torch.float64, device="cpu")[0]
    c_cs = _complex_step(lambda v: dev.charge(v)[0], v0)
    c_analytic = dev.charge(torch.tensor([v0], dtype=torch.float64))[1].item()
    assert _rel_err(c_cs, c_analytic) < DERIVATIVE_TOL
    # A charge-only device carries no conduction current.
    assert dev.conduction(torch.tensor([v0], dtype=torch.float64))[0].item() == 0.0


def test_devices_batch_by_model_signature():
    circuit = mw.Circuit("m")
    a, b = circuit.node("a"), circuit.node("b")
    devices = [
        mw.Diode("d1", a, circuit.ground, saturation_current=1e-12),
        mw.Diode("d2", b, circuit.ground, saturation_current=2e-12),
        mw.PolynomialIV("p1", a, b, coefficients=[0.0, 1e-3]),
    ]
    compiled = compile_nonlinear_devices(devices, {"a": 0, "b": 1}, dtype=torch.float64, device="cpu")
    kinds = {group.kind: group.batch_size for group in compiled}
    assert kinds == {"diode": 2, "polynomial_iv": 1}
