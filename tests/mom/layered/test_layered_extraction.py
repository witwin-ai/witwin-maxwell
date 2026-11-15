"""C2 gates: layered device-parameter extraction over NetworkData (CPU fp64).

- agreement with the script extraction on frozen F-A R6 and F-B data
  (data/extraction_reference.json, provenance inside): the formulas are the
  same, so only the NetworkData S round trip separates them (bound below);
- the nine-element pi fit against the script's bounded least-squares fit;
- exact recovery of analytic circuits;
- gradcheck of every quantity, including the implicit-function pi-fit gradient;
- bitwise repeat determinism and explicit refusals.
GPU gradients through GR1 are in test_layered_extraction_gradients.py.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from witwin.maxwell.network import NetworkData
from witwin.maxwell.mom.layered import extraction as ex

DATA = json.loads((Path(__file__).parent / "data" / "extraction_reference.json").read_text())
EPS = torch.finfo(torch.float64).eps


def network_from(entry, name):
    f = torch.tensor(entry["frequency_Hz"], dtype=torch.float64)
    matrix = torch.tensor(np.asarray(entry[name + "_real"]) + 1j * np.asarray(entry[name + "_imag"]))
    ports = tuple(f"p{i + 1}" for i in range(matrix.shape[-1]))
    build = NetworkData.from_y if name == "y" else NetworkData.from_z
    return build(frequencies=f, **{name: matrix}, z0=[50.] * len(ports), port_names=ports)


def round_trip_bound(network, source):
    """Relative error of the S round trip: kappa(I+-S) * eps per conversion, with margin 64."""
    s = network.s
    identity = torch.eye(s.shape[-1], dtype=s.dtype)
    kappa = torch.linalg.cond(identity - s).max() * torch.linalg.cond(identity + s).max()
    return 64 * float(kappa) * EPS * float(source.abs().max() / source.abs().amin(-1).amin(-1).min())


def relative(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


@pytest.mark.parametrize("case", sorted(DATA["fa"]))
def test_fa_one_port_series_and_peak_match_script(case, record_property):
    entry = DATA["fa"][case]
    network = network_from(entry, "y")
    y = torch.tensor(np.asarray(entry["y_real"]) + 1j * np.asarray(entry["y_imag"]))
    bound = round_trip_bound(network, y)
    metrics = entry["metrics"]
    values = dict(L11_H=ex.inductance(network, "p1", others="short"),
                  Rin_ohm=ex.resistance(network, "p1", others="short"),
                  Q=ex.quality_factor(network, "p1", others="short"),
                  L_H=ex.pi_model(network).series_inductance,
                  R_ohm=ex.pi_model(network).series_resistance)
    errors = {k: relative(v, metrics[k]) for k, v in values.items()}
    record_property("errors", errors)
    record_property("bound", bound)
    # The series branch is symmetrized here, -2/(Y12+Y21); the script uses -1/Y21.
    asymmetry = float(((y[:, 0, 1] - y[:, 1, 0]).abs() / y[:, 1, 0].abs()).max())
    for name, error in errors.items():
        assert error <= bound + (asymmetry if name in ("L_H", "R_ohm") else 0), name
    peak, frequency = ex.peak_quality_factor(network, "p1", others="short")
    assert abs(peak.item() / metrics["Q_peak"] - 1) <= bound
    assert frequency.item() == metrics["peak_frequency_Hz"]
    resonance = ex.self_resonance(network, "p1", others="short").item()
    if metrics["self_resonance_Hz"] is None:
        assert math.isnan(resonance)
    else:
        assert abs(resonance / metrics["self_resonance_Hz"] - 1) <= bound


@pytest.mark.parametrize("case", [c for c in sorted(DATA["fa"]) if "peak" not in c])
def test_fa_pi_fit_matches_script(case, record_property):
    entry = DATA["fa"][case]
    reference = entry["pi_fit"]
    fit = ex.fit_pi_model(network_from(entry, "y"), band=tuple(reference["fit_band_Hz"]))
    expected = np.array([reference["parameters"][name] for name in ex.PI_PARAMETERS])
    error = np.abs(fit.parameters.numpy() / expected - 1)
    record_property("max_parameter_relative_error", float(error.max()))
    record_property("iterations", fit.iterations)
    assert list(fit.active_bounds) == reference["active_bound_parameters"]
    # Two independent optimizers on the same objective; both stop at 1e-12-level
    # optimality, the Jacobian condition number is <= 2.4e6, so parameters agree to ~1e-6.
    assert error.max() <= 1e-6
    assert np.allclose(fit.relative_branch_rms.numpy(), reference["relative_branch_rms"], rtol=1e-6, atol=0)


def test_fb_windings_match_script(record_property):
    entry = DATA["fb"]
    network = network_from(entry, "z")
    z = torch.tensor(np.asarray(entry["z_real"]) + 1j * np.asarray(entry["z_imag"]))
    bound = round_trip_bound(network, z)
    first, second = ("p1", "p2"), ("p3", "p4")
    frequencies = entry["frequency_Hz"]
    worst = 0.
    for key, row in entry["low_band"].items():
        i = frequencies.index(float(key))
        values = dict(L1_H=ex.inductance(network, first, others="open")[i],
                      L2_H=ex.inductance(network, second, others="open")[i],
                      M_H=ex.mutual_inductance(network, first, second, others="open")[i],
                      k=ex.coupling_coefficient(network, first, second, others="open")[i],
                      R1_ohm=ex.resistance(network, first, others="open")[i],
                      R2_ohm=ex.resistance(network, second, others="open")[i],
                      Q1=ex.quality_factor(network, first, others="open")[i],
                      Q2=ex.quality_factor(network, second, others="open")[i])
        for name, value in values.items():
            worst = max(worst, abs(value.item() / row[name] - 1))
    for label, winding in (("1", first), ("2", second)):
        peak, frequency = ex.peak_quality_factor(network, winding, others="open")
        reference = entry["peaks"][label]
        worst = max(worst, abs(peak.item() / reference["Q_peak"] - 1),
                    abs(ex.self_resonance(network, winding, others="open").item()
                        / reference["self_resonance_Hz"] - 1))
        assert frequency.item() == reference["peak_frequency_Hz"]
    record_property("max_relative_error", worst)
    record_property("bound", bound)
    assert worst <= bound


def pi_network(ls, rs, cs, c1, g1, c2, g2, frequencies):
    """Exact two-port of series (rs - i w ls) || cs with shunt (g - i w c) arms."""
    omega = 2 * math.pi * frequencies
    series = 1 / (rs - 1j * omega * ls) - 1j * omega * cs
    arm1, arm2 = g1 - 1j * omega * c1, g2 - 1j * omega * c2
    y = torch.stack((torch.stack((series + arm1, -series), -1), torch.stack((-series, series + arm2), -1)), -2)
    return NetworkData.from_y(frequencies=frequencies, y=y, z0=[50., 50.], port_names=("a", "b"))


FREQUENCIES = torch.linspace(.2e9, 20e9, 100, dtype=torch.float64)


def test_analytic_pi_and_one_port_definitions():
    ls, rs, c1, g1, c2, g2 = 2e-9, 1.5, 50e-15, 1e-4, 70e-15, 2e-4
    network = pi_network(ls, rs, 0., c1, g1, c2, g2, FREQUENCIES)
    model = ex.pi_model(network)
    tolerance = 1e-12
    assert relative(model.series_inductance, np.full(100, ls)) <= tolerance
    assert relative(model.series_resistance, np.full(100, rs)) <= tolerance
    # Shunt = Y11 + Y12 cancels the series admittance (|Ys|/|Yarm| <= 1e3 here).
    assert relative(model.shunt_capacitance[0], np.full(100, c1)) <= 1e3 * tolerance
    assert relative(model.shunt_conductance[1], np.full(100, g2)) <= 1e3 * tolerance
    # Differential (floating common mode): series branch in parallel with the two arms in series.
    omega = 2 * math.pi * FREQUENCIES
    arms = 1 / (g1 - 1j * omega * c1) + 1 / (g2 - 1j * omega * c2)
    expected = 1 / (1 / (rs - 1j * omega * ls) + 1 / arms)
    differential = ex.terminal_impedance(network, ("a", "b"), others="short")
    assert relative(differential.real, expected.real) <= tolerance
    assert relative(differential.imag, expected.imag) <= tolerance
    assert torch.equal(differential, ex.terminal_impedance(network, ("a", "b"), others="open"))
    # One-port with b grounded: series branch parallel to arm a.
    short = 1 / (1 / (rs - 1j * omega * ls) + g1 - 1j * omega * c1)
    assert relative(ex.inductance(network, "a", others="short"), -short.imag / omega) <= tolerance
    q = ex.quality_factor(network, ("a", "b"), others="short")
    assert relative(q, -expected.imag / expected.real) <= tolerance


def test_analytic_coupled_inductors_and_t_model():
    l1, l2, m, r1, r2 = 2e-9, 1.5e-9, 1.1e-9, 1., 2.
    omega = 2 * math.pi * FREQUENCIES
    z = torch.empty((100, 2, 2), dtype=torch.complex128)
    z[:, 0, 0], z[:, 1, 1] = r1 - 1j * omega * l1, r2 - 1j * omega * l2
    z[:, 0, 1] = z[:, 1, 0] = -1j * omega * m
    network = NetworkData.from_z(frequencies=FREQUENCIES, z=z, z0=[50., 50.], port_names=("a", "b"))
    k = ex.coupling_coefficient(network, "a", "b", others="open")
    assert relative(k, np.full(100, m / math.sqrt(l1 * l2))) <= 1e-12
    assert relative(ex.mutual_inductance(network, "a", "b", others="open"), np.full(100, m)) <= 1e-12
    fit = ex.fit_t_model(network, "a", "b", others="open")
    for name, value in (("L1_H", l1), ("L2_H", l2), ("M_H", m), ("R1_ohm", r1), ("R2_ohm", r2)):
        assert abs(fit[name].item() / value - 1) <= 1e-12, name
    assert abs(fit["Rm_ohm"].item()) <= 1e-12 * r1
    assert fit["relative_branch_rms"].max().item() <= 1e-12
    model = ex.t_model(network, "a", "b", others="open")
    assert relative(model.arm_inductance[0], np.full(100, l1 - m)) <= 1e-12


def test_self_resonance_brackets_the_lc_tank():
    # Parallel tank L || C with series loss: f0 = 1/(2 pi sqrt(LC)) to first order in (R/wL)^2.
    l, c, r = 1e-9, 200e-15, .05
    frequencies = torch.linspace(1e9, 20e9, 400, dtype=torch.float64)
    omega = 2 * math.pi * frequencies
    z = 1 / (1 / (r - 1j * omega * l) - 1j * omega * c)
    network = NetworkData.from_z(frequencies=frequencies, z=z[:, None, None], z0=[50.], port_names=("p",))
    found = ex.self_resonance(network, "p", others="short").item()
    exact = math.sqrt(1 / (l * c) - (r / l) ** 2) / (2 * math.pi)
    step = (frequencies[1] - frequencies[0]).item()
    assert abs(found - exact) <= step
    peak, frequency = ex.peak_quality_factor(network, "p", others="short")
    assert frequency.item() < exact and peak.item() > 0


def test_pi_fit_recovers_exact_parameters():
    truth = torch.tensor((2.5e-9, 2., 8e-15, 60e-15, 40e-15, 400., 70e-15, 35e-15, 600.), dtype=torch.float64)
    branches = ex.substrate_pi_branches(FREQUENCIES, truth)
    series, arm1, arm2 = branches.unbind(-1)
    y = torch.stack((torch.stack((series + arm1, -series), -1), torch.stack((-series, series + arm2), -1)), -2)
    network = NetworkData.from_y(frequencies=FREQUENCIES, y=y, z0=[50., 50.], port_names=("a", "b"))
    fit = ex.fit_pi_model(network, band=(.5e9, 8e9))
    assert fit.active_bounds == ()
    assert float((fit.parameters / truth - 1).abs().max()) <= 1e-7
    model = fit.admittance(FREQUENCIES)
    assert float((model - y).abs().max() / y.abs().max()) <= 1e-9


def _synthetic(theta):
    """A noisy two-port whose data depends smoothly on theta (fit residual is nonzero)."""
    ls, rs, cs, c1, g1, c2, g2 = theta.unbind()
    omega = 2 * math.pi * FREQUENCIES
    # Frequency-dependent resistance keeps the nine-element fit residual nonzero.
    series = 1 / (rs * (1 + (FREQUENCIES / 5e9) ** .5) - 1j * omega * ls) - 1j * omega * cs
    arm1 = 1 / (1 / (-1j * omega * c1) + 1 / (g1 - 1j * omega * 30e-15))
    arm2 = 1 / (1 / (-1j * omega * c2) + 1 / (g2 - 1j * omega * 30e-15))
    y = torch.stack((torch.stack((series + arm1, -series), -1), torch.stack((-series, series + arm2), -1)), -2)
    return NetworkData.from_y(frequencies=FREQUENCIES, y=y, z0=[50., 50.], port_names=("a", "b"))


THETA = (2e-9, 1.5, 40e-15, 60e-15, 2e-3, 70e-15, 1.5e-3)


def _transformer(u):
    """Four-port of two coupled windings (a-b, c-d) with lossy, frequency-dependent coupling."""
    l1, l2, m = 2e-9 * u[0], 1.5e-9 * u[1], 1e-9 * u[2]
    omega = 2 * math.pi * FREQUENCIES
    r = 1 + u[3] * (FREQUENCIES / 1e9) ** .5
    winding = torch.stack((torch.stack((r - 1j * omega * l1, -1j * omega * m * u[4]), -1),
                           torch.stack((-1j * omega * m * u[4], r - 1j * omega * l2), -1)), -2)
    d = torch.tensor(((1., 0.), (-1., 0.), (0., 1.), (0., -1.)), dtype=torch.complex128)
    # Port-space impedance of floating windings is singular; add a small common-mode shunt.
    z = d @ winding @ d.T + 1e3 * torch.ones(4, 4, dtype=torch.complex128)
    return NetworkData.from_z(frequencies=FREQUENCIES, z=z, z0=[50.] * 4, port_names=("a", "b", "c", "d"))


def test_quantity_gradcheck():
    scale = torch.tensor(THETA, dtype=torch.float64)

    def quantities(u):
        network = _synthetic(u * scale)
        return torch.cat((ex.inductance(network, "a", others="short") * 1e9,
                          ex.quality_factor(network, ("a", "b"), others="open"),
                          ex.coupling_coefficient(_transformer(u), ("a", "b"), ("c", "d"), others="open"),
                          ex.mutual_inductance(_transformer(u), ("a", "b"), ("c", "d"), others="open") * 1e9,
                          ex.self_resonance(network, "a", others="short").reshape(1) / 1e9,
                          ex.peak_quality_factor(network, "a", others="short")[0].reshape(1),
                          ex.pi_model(network).series_inductance[:5] * 1e9))

    u = torch.ones(7, dtype=torch.float64, requires_grad=True)
    assert bool(torch.isfinite(quantities(u)).all())
    # Central differences: rounding noise eps_mach*max|f|/eps ~ 1e-8 at eps=1e-6
    # (max|f| ~ 70), truncation O(eps^2); both far below atol.
    assert torch.autograd.gradcheck(quantities, (u,), eps=1e-6, atol=1e-6, rtol=1e-5)


def test_pi_fit_implicit_gradient_against_refit():
    scale = torch.tensor(THETA, dtype=torch.float64)

    def fitted(u):
        fit = ex.fit_pi_model(_synthetic(u * scale), band=(.5e9, 8e9))
        assert fit.active_bounds == ()
        return torch.log(fit.parameters)

    u = torch.ones(7, dtype=torch.float64, requires_grad=True)
    reference = fitted(u.detach())
    assert float(fitted(u.detach()).sub(reference).abs().max()) == 0.
    # Finite differences re-run the full fit; its stopping rule (1e-15 relative
    # cost change) bounds the optimum noise near 1e-8 in log-parameters.
    assert torch.autograd.gradcheck(fitted, (u,), eps=1e-5, atol=2e-3, rtol=2e-3, nondet_tol=0.)


def test_repeat_is_bitwise():
    network = network_from(DATA["fa"]["pgs-shield-grounded1-sigma10"], "y")
    first = ex.fit_pi_model(network)
    second = ex.fit_pi_model(network)
    assert torch.equal(first.parameters, second.parameters)
    for function in (ex.inductance, ex.quality_factor, ex.self_resonance):
        assert torch.equal(function(network, ("p1", "p2"), others="short"),
                           function(network, ("p1", "p2"), others="short"))


def test_refusals():
    network = network_from(DATA["fa"]["none-shield-grounded1-sigma10"], "y")
    cases = (
        (lambda: ex.inductance(network, "p9", others="short"), "Unknown port"),
        (lambda: ex.inductance(network, 5, others="short"), "outside"),
        (lambda: ex.inductance(network, ("p1", "p1"), others="short"), "distinct"),
        (lambda: ex.inductance(network, ("p1", "p2", "p1"), others="short"), "pair"),
        (lambda: ex.inductance(network, "p1", others="ground"), "short' or 'open"),
        (lambda: ex.port_matrix(network, ["p1", "p1"], others="short"), "distinct ports"),
        (lambda: ex.port_matrix(network, ["p1"], others="short", kind="s"), "kind"),
        (lambda: ex.circuit_impedance(network, [], others="short"), "at least one terminal"),
        (lambda: ex.fit_pi_model(network, band=(30e9, 40e9)), "No sampled frequency"),
        (lambda: ex.fit_pi_model(network, band=(1e8, 1e8)), "two band samples"),
    )
    for call, message in cases:
        with pytest.raises(ValueError, match=message):
            call()
    with pytest.raises(TypeError):
        ex.inductance(network, "p1")
    single = NetworkData.from_z(frequencies=torch.tensor([1e9], dtype=torch.float64),
                                z=torch.tensor([[[1 - 1j]]]), z0=[50.], port_names=("p",))
    with pytest.raises(ValueError, match="two frequencies"):
        ex.self_resonance(single, "p", others="short")
    capacitive = NetworkData.from_z(frequencies=FREQUENCIES, z=(1j / (2 * math.pi * FREQUENCIES * 1e-12))[:, None, None]
                                    .to(torch.complex128), z0=[50.], port_names=("p",))
    with pytest.raises(ValueError, match="inductive sample"):
        ex.peak_quality_factor(capacitive, "p", others="short")
