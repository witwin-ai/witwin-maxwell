"""R6-4 (A3) TRL de-embedding on the layered MoM, CUDA, public Scene -> Simulation.mom -> Result path.

Fixture: microstrip on a PEC ground (h = W = 254 um, eps_r = 9.8; the C1 edge-port line), copper
sheets, full-width vertical edge ports to the ground. DUT: a 1 mm long, 50 um wide series strip
between two 254 um feeds of length l = 1, 2, 4 mm. ``trl_standards`` builds thru, open reflect and a
6 mm line from each device scene; ``trl`` (absolute level from the port model and the feed lengths)
and ``deembed`` give the DUT two-port.

Acceptance (plan A3):
- line: eps_eff within 1% and Z0 within 2% of Hammerstad-Jensen (closed-form accuracy ~0.2%, first
  order edge-cell error, as in the C1 gate), port-model residual and reciprocity small;
- de-embedded DUT series L changes by <= 0.5% over l = 1..4 mm and Q by <= 0.5% over l = 1..2 mm
  (the 4 mm / 5 GHz Q case, 1.4%, is the recorded open item, bounded at 2%), while the raw values
  move by far more; bitwise repeat.
"""

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.layered import extraction as ex
from witwin.maxwell.mom.layered.deembed import deembed, trl, trl_standards
from witwin.maxwell.mom.layered.port_system import edge_port
from tests.support.coupled_line_references import hammerstad_jensen_microstrip

# slow: 13 dense solves of up to 15 mm lines in the module fixture.
pytestmark = [pytest.mark.cuda, pytest.mark.slow,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")]

H = W = 254e-6
DUT_LENGTH, DUT_WIDTH, LINE = 1e-3, 50e-6, 6e-3
FEEDS = (1e-3, 2e-3, 4e-3)
# The 6 mm line spans 37..93 degrees over 2..5 GHz.
FREQUENCIES = (2e9, 3.5e9, 5e9)
# Explicit sheet: the recorded TRL measurements are sheet-model copper; under the A2 default "auto" the 5 um
# copper (t/delta >= 0.3 at 2-5 GHz) would select surface_admittance.
OPTIONS = dict(bridge_width_m=W, max_edge_length_m=H / 2, edge_cell_m=W / 32, conductor_model="sheet")
COPPER = mw.media.LossyMetalMedium(conductivity=5.8e7)


def strip(name, x0, x1, width):
    vertices = torch.tensor(((x0, -width / 2), (x1, -width / 2), (x1, width / 2), (x0, width / 2)), dtype=torch.float64)
    return mw.Structure(mw.geometry.PlanarPEC(vertices, normal_axis="z", coordinate=H, physical_thickness=5e-6),
                        COPPER, name=name)


def device(feed):
    stack = mw.mom.LayerStack((mw.mom.Layer("substrate", H, 9.8),), "pec", mw.mom.HalfSpace())
    scene = mw.Scene(background=stack, structures=(
                strip("feed1", 0., feed, W),
                strip("dut", feed, feed + DUT_LENGTH, DUT_WIDTH),
                strip("feed2", feed + DUT_LENGTH, 2 * feed + DUT_LENGTH, W),
            ))
    scene = scene.updated(ports=(
        *scene.ports,
        edge_port(scene, "p1", signal="feed1", side="-x", reference="pec_bottom"),
    ))
    scene = scene.updated(ports=(
        *scene.ports,
        edge_port(scene, "p2", signal="feed2", side="+x", reference="pec_bottom"),
    ))
    return scene


def solve(scene):
    config = mw.mom.MoMConfig(layered=mw.mom.LayeredSurfaceConfig(**OPTIONS))
    return mw.Simulation.mom(scene, frequencies=FREQUENCIES, config=config).run().network


def series(network):
    """Series-branch L and Q of a two-port: Z = -1/Y21."""
    z = -1 / network.to_y()[:, 1, 0]
    omega = 2 * torch.pi * network.frequencies.to(torch.float64)
    return -z.imag / omega, -z.imag / z.real


@pytest.fixture(scope="module")
def runs():
    results = {}
    for feed in FEEDS:
        scene = device(feed)
        standards = trl_standards(scene, dut=("dut",), planes=(feed, feed + DUT_LENGTH), line_length=LINE)
        assert standards["feed_lengths"] == pytest.approx((feed, feed), rel=1e-12)
        networks = {name: solve(standards[name]) for name in ("thru", "reflect", "line")}
        calibration = trl(networks["thru"], networks["reflect"], networks["line"], LINE, reflect_kind="open",
                          feed_lengths=standards["feed_lengths"])
        measured = solve(scene)
        results[feed] = dict(calibration=calibration, measured=measured,
                             deembedded=deembed(measured, calibration["left"], calibration["right"]))
    return results


def test_trl_line_parameters_against_closed_form(runs, record_property):
    expected_eps, expected_z0 = hammerstad_jensen_microstrip(1., 9.8)
    record_property("reference", (expected_eps, expected_z0))
    for feed, run in runs.items():
        c = run["calibration"]
        record_property(f"eps_eff_{feed}", c["eps_eff"].tolist())
        record_property(f"Z0_{feed}", [str(z) for z in c["line_impedance"].tolist()])
        record_property(f"port_model_residual_{feed}", c["port_model_residual"].tolist())
        record_property(f"reciprocity_{feed}", c["reciprocity"].tolist())
        assert ((c["eps_eff"] / expected_eps - 1).abs() < .01).all()
        assert ((c["line_impedance"].real / expected_z0 - 1).abs() < .02).all()
        assert (c["reciprocity"] < 1e-8).all()


def spread(values):
    """Max over frequency of (max - min) / |first| across feeds."""
    stacked = torch.stack(values)
    return ((stacked.amax(0) - stacked.amin(0)) / stacked[0].abs()).max().item()


def test_trl_deembedded_device_is_feed_length_insensitive(runs, record_property):
    """L within 0.5% over 1..4 mm feeds; Q within 0.5% over 1..2 mm feeds (plan A3).

    Measured (A6000, 2026-11-11): L spread 0.15% (raw 114%); Q spread 0.22% over 1..2 mm feeds
    at all frequencies, but the 4 mm feed at 5 GHz gives Q 106.7 against 105.2 (1.4%), i.e. a
    3 mOhm error on the 0.23 ohm de-embedded series R under 8 mm of lossy feed (not yet
    attributed: feed attenuation residual or the feed mesh differing between standards and
    device). That case is the open A3 item, recorded and bounded.
    """
    values = {feed: series(run["deembedded"]) for feed, run in runs.items()}
    raw = {feed: series(run["measured"]) for feed, run in runs.items()}
    for name, index in (("L", 0), ("Q", 1)):
        record_property(f"{name}_deembedded", [values[f][index].tolist() for f in FEEDS])
        record_property(f"{name}_raw", [raw[f][index].tolist() for f in FEEDS])
    l_spread = spread([values[f][0] for f in FEEDS])
    q_short = spread([values[f][1] for f in FEEDS[:2]])
    q_all = spread([values[f][1] for f in FEEDS])
    record_property("spreads", dict(L=l_spread, Q_1_2mm=q_short, Q_1_4mm=q_all,
                                    L_raw=spread([raw[f][0] for f in FEEDS]), Q_raw=spread([raw[f][1] for f in FEEDS])))
    assert l_spread <= 5e-3 and q_short <= 5e-3
    assert q_all <= 2e-2
    assert spread([raw[f][0] for f in FEEDS]) > 10 * l_spread and spread([raw[f][1] for f in FEEDS]) > 10 * q_all


def test_trl_deembedding_repeat_is_bitwise(runs):
    feed = FEEDS[0]
    again = solve(device(feed))
    assert torch.equal(again.s, runs[feed]["measured"].s)
