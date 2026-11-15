"""R6-4 (A3) TRL gates, CPU fp64: exact error-box recovery through analytic fixtures.

Fixture: lumped port discontinuity P (series then shunt, D(P) = 1), a uniform lossy feed of
physical length l1 (left) / l2 (right), the reference plane, the DUT. ``trl`` must recover the
DUT, gamma and the complex line impedance to round-off from thru, reflect (short or open; its
value unknown to ``trl``) and line standards, with unequal left/right boxes and feeds.
"""

import math

import pytest
import torch

from witwin.maxwell.mom.layered import extraction as ex
from witwin.maxwell.mom.layered.deembed import C0, abcd, deembed, impedance_from_abcd, line_abcd, open_short, trl
from witwin.maxwell.network import NetworkData

# TRL band: the 1 mm line spans 31..122 degrees over 10..40 GHz (singular at 0 and 180).
FREQUENCIES = torch.linspace(10e9, 40e9, 31, dtype=torch.float64)
EPS_EFF, LOSS, DELTA, FEEDS = 6.5, .02, 1e-3, (150e-6, 250e-6)
J = torch.diag(torch.tensor([1., -1.], dtype=torch.complex128))


def gamma():
    beta = 2 * math.pi * FREQUENCIES * math.sqrt(EPS_EFF) / C0
    return LOSS * beta - 1j * beta


def impedance():
    """Lossy-line characteristic impedance (complex)."""
    return torch.full_like(gamma(), 48.) * (1 - 1j * LOSS / 2)


def port_box(series_l, series_r, shunt_c, shunt_g):
    """Port -> series (R + L) -> shunt (G + C) -> line: D = 1."""
    omega = 2 * math.pi * FREQUENCIES
    box = torch.zeros((len(omega), 2, 2), dtype=torch.complex128)
    zs, yp = series_r - 1j * omega * series_l, shunt_g - 1j * omega * shunt_c
    box[:, 0, 0], box[:, 0, 1], box[:, 1, 0], box[:, 1, 1] = 1 + zs * yp, zs, yp, 1
    return box


def reversed_box(e):
    return J @ torch.linalg.inv(e) @ J


def boxes():
    left = port_box(.12e-9, 1.5, 25e-15, 2e-4) @ line_abcd(gamma(), impedance(), FEEDS[0])
    right = line_abcd(gamma(), impedance(), FEEDS[1]) @ reversed_box(port_box(.2e-9, .8, 40e-15, 1e-4))
    return left, right


def network(t):
    return NetworkData.from_z(frequencies=FREQUENCIES, z=impedance_from_abcd(t), z0=[50., 50.], port_names=("p1", "p2"))


def moebius(t, z):
    return (t[:, 0, 0] * z + t[:, 0, 1]) / (t[:, 1, 0] * z + t[:, 1, 1])


def reflect(left, right, load):
    z = torch.zeros((len(FREQUENCIES), 2, 2), dtype=torch.complex128)
    z[:, 0, 0], z[:, 1, 1] = moebius(left, load), moebius(J @ torch.linalg.inv(right) @ J, load)
    return NetworkData.from_z(frequencies=FREQUENCIES, z=z, z0=[50., 50.], port_names=("p1", "p2"))


def dut(inductance=1e-9):
    """Series inductor 1 nH + 1.5 ohm, 15 fF to the reference at each node."""
    omega = 2 * math.pi * FREQUENCIES
    series = 1 / (1.5 - 1j * omega * inductance)
    shunt = -1j * omega * 15e-15
    y = torch.stack((torch.stack((series + shunt, -series), -1), torch.stack((-series, series + shunt), -1)), -2)
    return abcd(torch.linalg.inv(y))


def standards(kind):
    left, right = boxes()
    omega = 2 * math.pi * FREQUENCIES
    load = (.3 - 1j * omega * 5e-12) if kind == "short" else 1 / (1e-5 - 1j * omega * 8e-15)
    return (network(left @ right), reflect(left, right, load),
            network(left @ line_abcd(gamma(), impedance(), DELTA) @ right))


def relative(a, b):
    return ((a - b).abs() / b.abs()).max().item()


@pytest.mark.parametrize("kind", ["short", "open"])
def test_trl_recovers_device_line_and_impedance(kind):
    left, right = boxes()
    calibration = trl(*standards(kind), DELTA, reflect_kind=kind, feed_lengths=FEEDS)
    assert relative(calibration["gamma"], gamma()) <= 1e-10
    assert relative(calibration["eps_eff"], torch.full_like(FREQUENCIES, EPS_EFF)) <= 1e-10
    assert relative(calibration["line_impedance"], impedance()) <= 1e-10
    assert calibration["port_model_residual"].max().item() <= 1e-10
    assert calibration["reciprocity"].max().item() <= 1e-10
    assert relative(calibration["left"], left) <= 1e-10 and relative(calibration["right"], right) <= 1e-10
    measured = network(left @ dut() @ right)
    recovered = deembed(measured, calibration["left"], calibration["right"])
    exact = impedance_from_abcd(dut())
    assert relative(recovered.to_z(), exact) <= 1e-9
    assert recovered.metadata["deembedding"] == "error_boxes"
    # A known line impedance gives the same calibration.
    given = trl(*standards(kind), DELTA, reflect_kind=kind, line_impedance=impedance())
    assert relative(given["left"], left) <= 1e-10 and given["port_model_residual"] is None


def test_trl_impedance_level_sets_the_device_scale(record_property):
    """A 2% line-impedance error scales the de-embedded device impedance by 2%: the level matters."""
    left, right = boxes()
    measured = network(left @ dut() @ right)
    wrong = trl(*standards("short"), DELTA, line_impedance=1.02 * impedance())
    inductance = ex.inductance(deembed(measured, wrong["left"], wrong["right"]), ("p1", "p2"), others="open")
    exact = ex.inductance(network(dut()), ("p1", "p2"), others="open")
    error = (inductance / exact - 1).abs()
    record_property("relative_L_error", [error.min().item(), error.max().item()])
    assert ((error - .02).abs() <= 2e-3).all()


def test_trl_gradient_through_the_device():
    """d L / d L_dut through calibration and de-embedding equals the fixture-free device derivative."""
    left, right = boxes()
    calibration = trl(*standards("short"), DELTA, feed_lengths=FEEDS)

    def derivative(fixture):
        value = torch.tensor(1e-9, dtype=torch.float64, requires_grad=True)
        if fixture:
            measured = deembed(network(left @ dut(value) @ right), calibration["left"], calibration["right"])
        else:
            measured = network(dut(value))
        inductance = ex.inductance(measured, ("p1", "p2"), others="open")
        return torch.autograd.grad(inductance.sum(), value)[0].item()

    assert abs(derivative(True) / derivative(False) - 1) <= 1e-8


def test_trl_leads_then_open_short_pads(record_property):
    """Two tiers: TRL removes ports and leads, open-short then removes the pads at the device plane.

    The open (pads only) and short (pads shorted to the reference) standards are uncoupled
    two-ports measured through the same port and lead fixture.
    """
    left, right = boxes()
    calibration = trl(*standards("short"), DELTA, feed_lengths=FEEDS)
    omega = 2 * math.pi * FREQUENCIES
    pad = torch.zeros((len(omega), 2, 2), dtype=torch.complex128)
    pad[:, 0, 0] = pad[:, 1, 1] = 1
    pad[:, 1, 0] = 2e-4 - 1j * omega * 60e-15
    y_pad = pad[:, 1, 0]
    measured = network(left @ pad @ dut() @ reversed_box(pad) @ right)
    open_, short = (reflect(left, right, load) for load in (1 / y_pad, 1 / (y_pad + 1e6)))

    def through(network_):
        return deembed(network_, calibration["left"], calibration["right"])

    pads_removed = open_short(through(measured), through(open_), through(short))
    exact = impedance_from_abcd(dut())
    error = relative(pads_removed.to_z(), exact)
    record_property("relative_error", error)
    # The 1e6 S short leaves O(|Z_dut| * 1e-6) of the device.
    assert error <= 1e-4
    assert relative(through(measured).to_z(), exact) > 1e-2


def test_trl_refusals():
    thru, reflect_, line = standards("short")
    with pytest.raises(ValueError, match="exactly one of"):
        trl(thru, reflect_, line, DELTA)
    with pytest.raises(ValueError, match="positive"):
        trl(thru, reflect_, line, 0., feed_lengths=FEEDS)
    with pytest.raises(ValueError, match="reflect_kind"):
        trl(thru, reflect_, line, DELTA, reflect_kind="load", feed_lengths=FEEDS)
    shifted = NetworkData(frequencies=FREQUENCIES * 2, s=line.s, z0=line.z0, port_names=line.port_names)
    with pytest.raises(ValueError, match="identical frequencies"):
        trl(thru, reflect_, shifted, DELTA, feed_lengths=FEEDS)
