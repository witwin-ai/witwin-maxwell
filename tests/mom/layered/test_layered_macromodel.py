"""Passive wideband macromodel (plan C3): accuracy, Hamiltonian certificate, enforcement, exports.

CPU only. Analytic networks: a lossy skin-effect transmission line (not rational, so
low-order fits violate passivity) and a DC-lossless RLC pi network (S has a singular
value of exactly 1 at DC). MoM data: the F-A adaptive dense sweep samples
(data/fixture_fa_adaptive_samples.npz, written by the R6-3 sweep probe).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.layered.macromodel import (admittance_system, enforce_passivity, enforce_rational_passivity,
                                                  hamiltonian_passivity, rational_from_realization, spice_subcircuit,
                                                  vector_fit)

DATA = Path(__file__).parent / "data"
# Dense check grid well beyond the fitted band plus a log sweep from mHz to 10 THz.
DENSE = torch.cat((torch.linspace(0, 400e9, 40001, dtype=torch.float64),
                   torch.logspace(-3, 13, 2001, dtype=torch.float64)))


def line_network(frequencies):
    """Lossy line, Z0 40 ohm, eps_eff 4, 3 mm, R' = 5 + 50 sqrt(f/GHz) ohm/m (exp(-iwt): jw -> -iw)."""
    w, c0 = 2 * torch.pi * frequencies, 299792458.0
    impedance = 5 + 50 * torch.sqrt(frequencies / 1e9) - 1j * w * 80 / c0
    admittance = 1e-3 * frequencies / 1e9 - 1j * w / (20 * c0)
    zc, gamma = torch.sqrt(impedance / admittance), torch.sqrt(impedance * admittance)
    sinh, cosh = torch.sinh(gamma * 3e-3), torch.cosh(gamma * 3e-3)
    y11, y12 = cosh / (zc * sinh), -1 / (zc * sinh)
    y = torch.stack((torch.stack((y11, y12), -1), torch.stack((y12, y11), -1)), -2)
    return mw.rf.NetworkData.from_y(frequencies=frequencies, y=y, z0=50.0, port_names=("1", "2"))


def rlc_network(frequencies):
    """Series R-L with coupling C, shunt R-C at both ports: the even mode is open at DC (|S| = 1)."""
    s = torch.complex(torch.zeros_like(frequencies), -2 * torch.pi * frequencies)
    series = 1 / (2.0 + s * 1e-9) + s * 20e-15
    shunt = 1 / (8.0 + 1 / (s * 120e-15))
    y = torch.stack((torch.stack((series + shunt, -series), -1), torch.stack((-series, series + shunt), -1)), -2)
    return mw.rf.NetworkData.from_y(frequencies=frequencies, y=y, z0=50.0, port_names=("1", "2"))


LINE = line_network(torch.linspace(1e8, 60e9, 121, dtype=torch.float64))


def max_sigma(system, frequencies=DENSE):
    return float(torch.linalg.svdvals(system.evaluate(frequencies)).amax())


@pytest.mark.parametrize("order", (6, 10))
def test_enforced_line_is_certified_and_within_bounds(order, record_property):
    model = mw.mom.passive_macromodel(LINE, order=order, tolerance=1.0)
    report = model.report
    record_property("report", str(report))
    # The unconstrained fit violates (order 6: 3.4% inside 6-56 GHz; order 10: 1e-4 near DC).
    assert not report.before.passive and report.before.violation_bands_hz
    assert report.after.passive and report.after.axis_distance > 1e-6 and report.after.direct_norm < 1
    assert report.enforcement_steps >= 1
    # Independent evidence: dense sampling from mHz to 10 THz and the shared in-band interval refinement
    # (its resolvent Lipschitz bound is too loose to certify a 1e-4 margin, so only "no violation found").
    assert max_sigma(model.system) <= 1
    band = model.system.check_passivity(LINE.frequencies)
    record_property("shared_interval_check", str(band))
    assert band.passive
    # Stated bounds: triangle inequality (exact) and enforcement change <= 2 (violation + margin).
    violation = report.before.max_singular_value - 1
    assert report.max_error <= report.fit_max_error + report.enforcement_change + 1e-12
    assert report.enforcement_change <= 2 * (violation + report.margin)


def test_hamiltonian_crossings_match_dense_sampling():
    fit = vector_fit(LINE.frequencies, LINE.s, 6, 8).to_state_space()
    report = hamiltonian_passivity(fit)
    grid = torch.linspace(0, 120e9, 240001, dtype=torch.float64)
    # Every singular value (not only the largest) crosses 1 at a Hamiltonian imaginary eigenvalue.
    excess = torch.linalg.svdvals(fit.evaluate(grid)) - 1
    flips = (excess[1:] > 0) != (excess[:-1] > 0)
    changes = torch.sort(grid[1:][None, :].expand(flips.shape[1], -1)[flips.T]).values
    crossings = torch.tensor(report.crossings_hz, dtype=torch.float64)
    assert len(crossings) == len(changes) == 10
    assert float((crossings - changes).abs().max()) <= float(grid[1] - grid[0])
    assert report.max_singular_value == pytest.approx(1 + float(excess.max()), abs=2e-4)


def test_passive_fit_needs_no_enforcement():
    model = mw.mom.passive_macromodel(LINE, order=12, tolerance=2e-3)
    assert model.report.before.passive and model.report.enforcement_steps == 0
    assert model.report.enforcement_change == 0 and model.report.max_error == model.report.fit_max_error
    assert max_sigma(model.system) <= 1


def test_order_selection_and_unreachable_tolerance():
    model = mw.mom.passive_macromodel(LINE, tolerance=2e-3)
    assert model.report.order == 10 and model.report.after.passive
    with pytest.raises(RuntimeError, match="No passive macromodel within"):
        mw.mom.passive_macromodel(LINE, tolerance=1e-6, max_order=8)
    complex_reference = mw.rf.NetworkData(frequencies=LINE.frequencies, s=LINE.s, z0=50 + 1j,
                                          port_names=LINE.port_names)
    with pytest.raises(ValueError, match="frequency-independent z0"):
        mw.mom.passive_macromodel(complex_reference, order=6)


def test_dc_lossless_network():
    network = rlc_network(torch.linspace(1e8, 40e9, 81, dtype=torch.float64))
    model = mw.mom.passive_macromodel(network, order=4, tolerance=2e-4)
    report = model.report
    # |D| = 1 (lossless even mode at infinity) is clipped to 1 - margin; DC stays lossless.
    assert report.after.passive and report.after.dc_lossless
    assert report.after.direct_norm == pytest.approx(1 - report.margin, abs=1e-12)
    assert max_sigma(model.system) <= 1 + 1e-12


def test_repeat_is_exact(tmp_path):
    # QR least squares in the fit (not the default CPU driver of the shared fit, which varies at 1e-16).
    first = mw.mom.passive_macromodel(LINE, order=6, tolerance=1.0)
    second = mw.mom.passive_macromodel(LINE, order=6, tolerance=1.0)
    for name in "ABCD":
        assert torch.equal(getattr(first.system, name), getattr(second.system, name))
    assert first.report == second.report
    assert first.to_spice(tmp_path / "a.cir").read_text() == second.to_spice(tmp_path / "b.cir").read_text()


def test_fit_matches_shared_vector_fit():
    # Same algorithm as rational.fit_rational up to the least-squares solver.
    shared = LINE.fit_rational(mw.rf.rational.RationalFitConfig(order=10, iterations=8, diagnostic_only=True),
                               representation="S")
    local = vector_fit(LINE.frequencies, LINE.s, 10, 8)
    grid = torch.linspace(0, 80e9, 801, dtype=torch.float64)
    assert float((local.evaluate(grid) - shared.evaluate(grid)).abs().max()) <= 1e-9


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_repeat_is_exact(tmp_path):
    network = line_network(torch.linspace(1e8, 60e9, 121, dtype=torch.float64, device="cuda"))
    first = mw.mom.passive_macromodel(network, order=6, tolerance=1.0)
    second = mw.mom.passive_macromodel(network, order=6, tolerance=1.0)
    for name in "ABCD":
        assert torch.equal(getattr(first.system, name), getattr(second.system, name))
    assert first.report == second.report and first.report.after.passive
    assert first.to_spice(tmp_path / "a.cir").read_text() == second.to_spice(tmp_path / "b.cir").read_text()
    # Same algorithm on both devices: CUDA and CPU agree within the fit's own conditioning.
    cpu = mw.mom.passive_macromodel(LINE, order=6, tolerance=1.0)
    assert abs(cpu.report.max_error - first.report.max_error) <= 1e-6


def test_admittance_realization_matches_network_conversion():
    model = mw.mom.passive_macromodel(LINE, order=10, tolerance=1.0)
    frequencies = torch.linspace(0, 80e9, 401, dtype=torch.float64)[1:]
    expected = model.evaluate(frequencies).to_y()
    realized = model.admittance_system().evaluate(frequencies)
    assert float((realized - expected).abs().max() / expected.abs().max()) <= 1e-12


def nodal_admittance(text, frequencies):
    """Independent read-back: stamp the C/G cards (SPICE semantics) and solve the nodal equations.

    Port nodes are driven (p_k = unit voltage, refs grounded); the state nodes are solved.
    Returns Y [F, P, P] with s = -i 2 pi f (the source data's convention).
    """
    lines = [line.split() for line in text.splitlines() if line and not line.startswith("*")]
    header = next(line for line in lines if line[0] == ".SUBCKT")
    ports = header[2:]
    cards = [line for line in lines if line[0][0] in "CG"]
    nodes = sorted({n for card in cards for n in card[1:-1] if n != "0"} - set(ports))
    index = {name: i for i, name in enumerate(ports + nodes)}
    size, known = len(index), len(ports)
    s = torch.complex(torch.zeros_like(frequencies), -2 * torch.pi * frequencies)
    matrix = torch.zeros((len(s), size, size), dtype=torch.complex128)

    def stamp(row, column, value):
        if row != "0" and column != "0":
            matrix[:, index[row], index[column]] += value

    for card in cards:
        value = float(card[-1])
        if card[0][0] == "C":
            a, b = card[1], card[2]
            for row, column, sign in ((a, a, 1), (b, b, 1), (a, b, -1), (b, a, -1)):
                stamp(row, column, sign * s * value)
        else:
            # Current value * (V(c+) - V(c-)) leaves n+ through the source and enters n-.
            plus, minus, control_plus, control_minus = card[1:5]
            for row, column, sign in ((plus, control_plus, 1), (plus, control_minus, -1),
                                      (minus, control_plus, -1), (minus, control_minus, 1)):
                stamp(row, column, sign * value)
    drive = torch.zeros((known, len(ports) // 2), dtype=torch.complex128)
    drive[0::2] = torch.eye(len(ports) // 2, dtype=torch.complex128)
    states = torch.linalg.solve(matrix[:, known:, known:], -matrix[:, known:, :known] @ drive)
    currents = matrix[:, :known, :known] @ drive + matrix[:, :known, known:] @ states
    return currents[:, 0::2]


def test_spice_netlist_reads_back(tmp_path):
    model = mw.mom.passive_macromodel(LINE, order=10, tolerance=1.0)
    path = model.to_spice(tmp_path / "line.cir", name="line_macromodel")
    text = path.read_text()
    assert text.splitlines()[3] == ".SUBCKT line_macromodel p1 p1_ref p2 p2_ref"
    assert text.rstrip().endswith(".ENDS line_macromodel")
    frequencies = torch.linspace(1e8, 60e9, 241, dtype=torch.float64)
    expected = model.evaluate(frequencies).to_y()
    readback = nodal_admittance(text, frequencies)
    assert float((readback - expected).abs().max() / expected.abs().max()) <= 1e-12


def burst(system, frequency, dt, steps):
    """Energy ratio of the bilinear model driven along the worst direction at analog ``frequency``.

    A real system answers Re(w exp(+i W t)) with Re(H(+i w) w exp(+i W t)), H(+i w) = conj S(f), whose
    leading right singular vector is conj(v) = vh[0]; the bilinear map hits w at W = (2/dt) atan(w dt/2).
    """
    _, _, vh = torch.linalg.svd(system.evaluate(torch.tensor([frequency], dtype=torch.float64))[0])
    omega = 2 / dt * np.arctan(np.pi * frequency * dt)
    t = torch.arange(steps, dtype=torch.float64) * dt
    inputs = (vh[0][None, :] * torch.exp(1j * omega * t)[:, None]).real
    return energy_ratio(system.discretize(dt), inputs)


def energy_ratio(system, inputs):
    state = torch.zeros(system.state_count, dtype=torch.float64)
    incident = reflected = 0.0
    worst = 0.0
    for value in inputs:
        state, output = system.step(state, value)
        incident += float(value @ value)
        reflected += float(output @ output)
        worst = max(worst, reflected - incident)
    return reflected / incident, worst


def test_time_domain_dissipativity():
    enforced = mw.mom.passive_macromodel(LINE, order=6, tolerance=1.0)
    fit = vector_fit(LINE.frequencies, LINE.s, 6, 8).to_state_space()
    peak = max(enforced.report.before.violation_bands_hz, key=lambda band: float(
        torch.linalg.svdvals(fit.evaluate(torch.tensor([sum(band) / 2], dtype=torch.float64))).max()))
    frequency, dt = sum(peak) / 2, 1 / (40 * 60e9)
    ratio_fit, _ = burst(fit, frequency, dt, 6000)
    ratio_passive, worst = burst(enforced.system, frequency, dt, 6000)
    # The violating fit amplifies its worst direction; the passive model dissipates at every prefix.
    assert ratio_fit > 1.01
    assert ratio_passive < 1 and worst <= 0
    generator = torch.Generator().manual_seed(7)
    noise = torch.randn((6000, 2), generator=generator, dtype=torch.float64)
    ratio_noise, worst_noise = energy_ratio(enforced.discretize(dt), noise)
    assert ratio_noise < 1 and worst_noise <= 0


def test_fa_mom_samples():
    data = np.load(DATA / "fixture_fa_adaptive_samples.npz")
    network = mw.rf.NetworkData(frequencies=torch.from_numpy(data["frequencies"]), s=torch.from_numpy(data["s"]),
                                z0=torch.from_numpy(data["z0"]), port_names=("P1", "P2"))
    model = mw.mom.passive_macromodel(network, tolerance=1e-4)
    report = model.report
    # The order-8 fit is passive in band but not below it (|S|_2 = 1 + 2.6e-6 under ~32 MHz, out of the
    # 0.1-20 GHz band): only the global Hamiltonian test sees it; one enforcement step removes it.
    assert not report.before.passive and report.before.violation_bands_hz[-1][1] < float(network.frequencies[0])
    assert report.after.passive and report.max_error <= 1e-4 and report.enforcement_steps == 1
    assert max_sigma(model.system) <= 1
    assert model.system.check_passivity(network.frequencies).passive
    # The adaptive surrogate on 100 uniform points (its own 1e-3 tolerance) and the macromodel agree
    # within the sum of the two tolerances (measured 4.4e-5).
    surrogate = torch.from_numpy(data["model100"])
    uniform = torch.from_numpy(data["uniform"])
    assert float((model.system.evaluate(uniform) - surrogate).abs().max()) <= 2e-3


def test_rational_round_trip_and_enforcement():
    model = vector_fit(LINE.frequencies, LINE.s, 6, 8)
    back = rational_from_realization(model, model.to_state_space())
    grid = torch.linspace(0, 80e9, 801, dtype=torch.float64)
    assert float((back.evaluate(grid) - model.evaluate(grid)).abs().max()) <= 1e-13
    enforced, report, steps = enforce_rational_passivity(model, LINE.frequencies, LINE.s)
    assert report.passive and steps >= 1 and torch.equal(enforced.poles, model.poles)
    assert max_sigma(enforced) <= 1
    # The adaptive sweep's in-band check (shared pole-aware interval bound) now passes too.
    assert enforced.check_passivity(grid).passive


def test_public_export_is_the_layered_function():
    """mw.mom exports the layered macromodel API itself (round-7 export; round 6 shipped it under mw.mom.layered)."""
    from witwin.maxwell.mom.layered import macromodel
    for name in ("passive_macromodel", "PassiveMacromodel", "MacromodelReport", "HamiltonianReport"):
        assert getattr(mw.mom, name) is getattr(macromodel, name) is getattr(mw.mom.layered, name)
