"""A3 groundwork gates (CPU fp64): two-line line parameters and open-short de-embedding.

- line_parameters recovers gamma of an analytic lossy line exactly through
  arbitrary identical reciprocal port error boxes, including beta*delta > pi
  (branch continuity), and is differentiable;
- open_short removes analytic pad (shunt) plus distributed feed-line parasitics:
  de-embedded L and Q change by <= 0.5% over feed lengths 50..200 um, while the
  raw values move by several percent;
- open_short_standards builds the open/short Scenes from named structures.
"""

import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.layered import extraction as ex
from witwin.maxwell.mom.layered.deembed import (
    C0, abcd, impedance_from_abcd, line_abcd, line_parameters, open_short, open_short_standards)
from witwin.maxwell.network import NetworkData

FREQUENCIES = torch.linspace(.5e9, 40e9, 80, dtype=torch.float64)
EPS_EFF, LOSS = 6.5, .02


def gamma(frequencies):
    """Lossy quasi-TEM line: alpha = LOSS * beta (gamma = alpha - i beta)."""
    beta = 2 * math.pi * frequencies * math.sqrt(EPS_EFF) / C0
    return LOSS * beta - 1j * beta


def network(t, names=("p1", "p2")):
    return NetworkData.from_z(frequencies=FREQUENCIES[:t.shape[0]], z=impedance_from_abcd(t), z0=[50., 50.],
                              port_names=names)


def error_box():
    """A reciprocal port discontinuity: series 0.3 nH + 2 ohm, then shunt 40 fF + 1 mS."""
    omega = 2 * math.pi * FREQUENCIES
    series = torch.zeros((len(omega), 2, 2), dtype=torch.complex128)
    series[:, 0, 0] = series[:, 1, 1] = 1
    series[:, 0, 1] = 2 - 1j * omega * .3e-9
    shunt = torch.zeros_like(series)
    shunt[:, 0, 0] = shunt[:, 1, 1] = 1
    shunt[:, 1, 0] = 1e-3 - 1j * omega * 40e-15
    return series @ shunt


def reversed_box(e):
    j = torch.diag(torch.tensor([1., -1.], dtype=torch.complex128))
    return j @ torch.linalg.inv(e) @ j


def test_two_line_gamma_through_error_boxes():
    e = error_box()
    g = gamma(FREQUENCIES)
    z0 = torch.full_like(g, 48.)
    lines = [network(e @ line_abcd(g, z0, length) @ reversed_box(e)) for length in (1e-3, 3e-3)]
    result = line_parameters(*lines, 2e-3)
    # beta*delta reaches 2*pi*40e9*sqrt(6.5)*2e-3/c = 4.3 rad > pi: branch continuity is exercised.
    assert (-g.imag * 2e-3).max().item() > math.pi
    assert ((result["gamma"] - g).abs() / g.abs()).max().item() <= 1e-9
    assert ((result["eps_eff"] / EPS_EFF - 1).abs()).max().item() <= 1e-9
    assert torch.allclose(result["wavenumber"], 1j * g, rtol=1e-9, atol=0)
    assert result["reciprocity"].max().item() <= 1e-10
    # With an ideal (identity) error box the two-line impedance is the line impedance.
    ideal = [network(line_abcd(g, z0, length)) for length in (1e-3, 3e-3)]
    assert ((line_parameters(*ideal, 2e-3)["impedance_two_line"] - 48.).abs().max()).item() <= 1e-8


def test_two_line_gradient():
    def eps(scale):
        g = gamma(FREQUENCIES) * scale
        lines = [network(line_abcd(g, torch.full_like(g, 48.), length)) for length in (1e-3, 3e-3)]
        return line_parameters(*lines, 2e-3)["eps_eff"][:10]

    scale = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    value, = torch.autograd.grad(eps(scale).sum(), scale)
    assert abs(value.item() - 2 * EPS_EFF * 10) <= 1e-6 * 2 * EPS_EFF * 10


def fixture(dut_y, feed_length):
    """Pad (shunt 30 fF + 0.1 mS) then a 50-ohm feed line of feed_length on each side of a DUT two-port."""
    omega = 2 * math.pi * FREQUENCIES
    g = gamma(FREQUENCIES)
    z0 = torch.full_like(g, 50.)
    pad = torch.zeros((len(omega), 2, 2), dtype=torch.complex128)
    pad[:, 0, 0] = pad[:, 1, 1] = 1
    pad[:, 1, 0] = 1e-4 - 1j * omega * 30e-15
    feed = pad @ line_abcd(g, z0, feed_length)
    dut = abcd(torch.linalg.inv(dut_y))
    measured = network(feed @ dut @ reversed_box(feed))
    # One-port feed input admittances terminated open / short at the DUT reference plane.
    a, b, c, d = feed[:, 0, 0], feed[:, 0, 1], feed[:, 1, 0], feed[:, 1, 1]
    y_open, y_short = c / a, d / b
    eye = torch.eye(2, dtype=torch.complex128)

    def standard(y):
        return NetworkData.from_y(frequencies=FREQUENCIES, y=y[:, None, None] * eye, z0=[50., 50.],
                                  port_names=("p1", "p2"))
    return measured, standard(y_open), standard(y_short)


def dut_admittance():
    """Inductor two-port: series 1 nH + 1.5 ohm, 15 fF to the reference at each node."""
    omega = 2 * math.pi * FREQUENCIES
    series = 1 / (1.5 - 1j * omega * 1e-9)
    shunt = -1j * omega * 15e-15
    return torch.stack((torch.stack((series + shunt, -series), -1), torch.stack((-series, series + shunt), -1)), -2)


def test_open_short_feed_length_insensitivity(record_property):
    y = dut_admittance()
    exact = NetworkData.from_y(frequencies=FREQUENCIES, y=y, z0=[50., 50.], port_names=("p1", "p2"))
    band = FREQUENCIES <= 5e9
    reference = dict(L=ex.inductance(exact, ("p1", "p2"), others="open")[band],
                     Q=ex.quality_factor(exact, ("p1", "p2"), others="open")[band])
    worst, raw_worst = {}, {}
    for feed_length in (50e-6, 100e-6, 200e-6):
        measured, open_, short = fixture(y, feed_length)
        deembedded = open_short(measured, open_, short)
        assert deembedded.metadata["deembedding"] == "open_short"
        for name, function in (("L", ex.inductance), ("Q", ex.quality_factor)):
            value = function(deembedded, ("p1", "p2"), others="open")[band]
            raw = function(measured, ("p1", "p2"), others="open")[band]
            worst[name] = max(worst.get(name, 0.), (value / reference[name] - 1).abs().max().item())
            raw_worst[name] = max(raw_worst.get(name, 0.), (raw / reference[name] - 1).abs().max().item())
    record_property("deembedded_worst_relative", worst)
    record_property("raw_worst_relative", raw_worst)
    # Distributed feeds leave an O((beta l)^2) residue: (beta l)^2 = 7e-4 at 5 GHz, 200 um.
    assert worst["L"] <= 5e-3 and worst["Q"] <= 5e-3
    assert raw_worst["L"] > 2e-2 and raw_worst["Q"] > 2e-2


def test_open_short_refusals_and_standards():
    y = dut_admittance()
    measured, open_, short = fixture(y, 50e-6)
    renamed = NetworkData(frequencies=open_.frequencies, s=open_.s, z0=open_.z0, port_names=("a", "b"))
    with pytest.raises(ValueError, match="measured ports and frequencies"):
        open_short(measured, renamed, short)
    with pytest.raises(ValueError, match="identical frequencies"):
        line_parameters(measured, NetworkData(frequencies=FREQUENCIES * 2, s=measured.s, z0=measured.z0,
                                              port_names=measured.port_names), 1e-3)
    with pytest.raises(ValueError, match="positive"):
        line_parameters(measured, measured, 0.)
    stack = mw.mom.LayerStack((mw.mom.Layer("air", 1e-3),), mw.mom.HalfSpace(), mw.mom.HalfSpace())
    scene = mw.Scene(background=stack)
    scene = scene.updated(structures=(
        *scene.structures,
        *(mw.Structure(mw.geometry.PlanarPEC([(x, 0.), (x + 1e-4, 0.), (x + 1e-4, 1e-4), (x, 1e-4)],
                                    coordinate=0.), mw.Material.pec(), name=name) for name, x in (("dut", 0.), ("feed", 2e-4))),
    ))
    scene = scene.updated(ports=(
        *scene.ports,
        mw.LumpedPort("p", positive=(2e-4, 5e-5, 0.), negative=(1e-4, 5e-5, 0.),
                      voltage_path=mw.AxisPath("x")),
    ))
    opened, shorted = open_short_standards(scene, dut=("dut",), short_region=(1e-4, 2e-4, 0., 1e-4, 0.))
    assert [s.name for s in opened.structures] == ["feed"]
    assert [s.name for s in shorted.structures] == ["feed", "deembed_short"]
    assert [s.name for s in scene.structures] == ["dut", "feed"] and opened.ports == scene.ports
    with pytest.raises(ValueError, match="unknown device structures"):
        open_short_standards(scene, dut=("missing",), short_region=(0., 1., 0., 1., 0.))
    with pytest.raises(ValueError, match="x0 < x1"):
        open_short_standards(scene, dut=("dut",), short_region=(1., 0., 0., 1., 0.))
