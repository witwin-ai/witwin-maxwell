"""C1 gates: layered edge ports, vertical inter-layer ports and mixed-mode pairs.

Acceptance (CUDA, public Scene -> Simulation.mom -> Result path):
- microstrip on a PEC ground with full-width vertical edge ports: eps_eff and Z0
  against Hammerstad-Jensen, and bitwise identity with the delta-gap port of the
  same width (the edge port is the delta-gap bridge spanning the aperture);
- CPW with finite grounds and horizontal edge ports to the named ground: eps_eff
  and Z0 against the conformal-mapping closed form; eps_eff against narrow
  delta-gap ports (the two-line method cancels any port error box);
- microstrip over a finite ground sheet with vertical inter-layer edge ports;
- coupled microstrip, four edge ports, PortPair/mixed_mode: odd/even eps_eff and
  impedances against Kirschning-Jansen;
- reciprocity, passivity, bitwise repeat, port metadata.
Host-only: every refusal of the port system.
"""

import json
import math

import pytest
import torch
from scipy.special import ellipk

import witwin.maxwell as mw
from witwin.maxwell.mom.layered.deembed import line_parameters
from witwin.maxwell.mom.layered.network import prepare_layered_network
from witwin.maxwell.mom.layered.port_system import PortPair, edge_port, mixed_mode
from witwin.maxwell.network import NetworkData
from tests.support.coupled_line_references import hammerstad_jensen_microstrip, kirschning_jansen_coupled_microstrip

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")
ETA0 = 376.730313668


def rectangle(x0, x1, y0, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def sheet(name, vertices, z, holes=()):
    return mw.Structure(mw.geometry.PlanarPEC(vertices, holes=holes, normal_axis="z", coordinate=z), mw.Material.pec(), name=name)


def solve(scene, frequencies, **options):
    config = mw.mom.MoMConfig(layered=mw.mom.LayeredSurfaceConfig(**options))
    return mw.Simulation.mom(scene, frequencies=frequencies, config=config).run()


def checks(network):
    y = network.to_y()
    asymmetry = (torch.linalg.matrix_norm(y - y.transpose(-1, -2)) / torch.linalg.matrix_norm(y)).max().item()
    singular = torch.linalg.matrix_norm(network.s, ord=2).max().item()
    return asymmetry, singular


# Microstrip on a PEC ground (test_layered_vias.test_vertical_port_to_ground at 100 MHz).
# 1 and 3 GHz gate the horizontal admission floor: before it, rows with a source on the
# PEC plane (denormal image-cancelled remainder) refused these accurate tables.
H, WIDTH, MS_FREQUENCIES = 254e-6, 254e-6, (100e6, 1e9, 3e9)


def microstrip(length, *, edge=True):
    stack = mw.mom.LayerStack((mw.mom.Layer("substrate", H, 9.8),), "pec", mw.mom.HalfSpace())
    scene = mw.Scene(background=stack, structures=(
                sheet("sig", rectangle(0., length, -WIDTH / 2, WIDTH / 2), H),
            ))
    scene = scene.updated(ports=(
        *scene.ports,
        *(edge_port(scene, name, signal="sig", side=side, reference="pec_bottom") if edge else
          mw.LumpedPort(name, positive=(x, 0., H), negative=(x, 0., 0.), voltage_path=mw.AxisPath("z")) for name, side, x in (("p1", "-x", 0.), ("p2", "+x", length))),
    ))
    return scene


MS_OPTIONS = dict(bridge_width_m=WIDTH, max_edge_length_m=H / 2, edge_cell_m=WIDTH / 32)


@pytest.fixture(scope="module")
def microstrip_results():
    return {length: solve(microstrip(length), MS_FREQUENCIES, **MS_OPTIONS) for length in (2e-3, 4e-3)}


@requires_cuda
@pytest.mark.cuda
def test_microstrip_edge_ports(microstrip_results, record_property):
    short, long = (microstrip_results[length].network for length in (2e-3, 4e-3))
    line = line_parameters(short, long, 2e-3)
    expected_eps, expected_z0 = hammerstad_jensen_microstrip(1., 9.8)
    eps, z0 = line["eps_eff"].tolist(), line["impedance_two_line"].real.tolist()
    record_property("eps_eff", eps)
    record_property("Z0", z0)
    record_property("reference", (expected_eps, expected_z0))
    # Same limits as the delta-gap gate at 100 MHz (closed-form accuracy ~0.2%,
    # first-order mesh error). Dispersion up to 3 GHz (f h / c = 2.5e-3) moves the
    # quasi-TEM values by far less than 0.5% from the 100 MHz trend.
    for e, z in zip(eps, z0):
        assert abs(e / expected_eps - 1) < .01
        assert abs(z / expected_z0 - 1) < .02
        assert abs(e / eps[0] - 1) < .005 and abs(z / z0[0] - 1) < .005
    for result in microstrip_results.values():
        vertical = result.solver.prepared.table.vertical
        assert vertical.z_interpolation_error.item() <= 1e-6 and vertical.rho_interpolation_error.item() <= 1e-6
    for network in (short, long):
        asymmetry, singular = checks(network)
        assert asymmetry <= 1e-10 and singular <= 1 + 1e-9


@requires_cuda
@pytest.mark.cuda
def test_microstrip_edge_port_equals_delta_gap_of_the_same_width(microstrip_results):
    # The aperture spans the full strip width = bridge_width: same bridge, same mesh, same bits.
    reference = solve(microstrip(2e-3, edge=False), MS_FREQUENCIES, **MS_OPTIONS).network
    assert torch.equal(reference.s, microstrip_results[2e-3].network.s)
    again = solve(microstrip(2e-3), MS_FREQUENCIES, **MS_OPTIONS)
    assert torch.equal(again.network.s, microstrip_results[2e-3].network.s)
    assert torch.equal(mw.mom.surface_currents(again).currents, mw.mom.surface_currents(microstrip_results[2e-3]).currents)


@requires_cuda
@pytest.mark.cuda
def test_port_metadata_records_type_reference_and_definitions(microstrip_results):
    result = microstrip_results[2e-3]
    records = result.metadata["ports"]
    assert records == result.network.metadata["ports"]
    # Network metadata stays JSON-serializable (evidence producers and fixture tests dump it).
    json.dumps(result.network.metadata)
    assert records[0]["reference_impedance_ohm"] == (50.0, 0.0)
    first = records[0]
    assert (first["name"], first["kind"], first["orientation"]) == ("p1", "edge", "vertical")
    assert (first["positive_conductor"], first["reference"], first["reference_kind"]) == ("sig", "pec_bottom", "pec_plane")
    assert first["width_m"] == WIDTH and first["length_m"] == H
    assert first["voltage_definition"].startswith("V = phi(positive) - phi(negative)")
    assert first["current_definition"].startswith("I = total RWG current crossing the bridge cut")
    narrow = solve(microstrip(2e-3, edge=False), MS_FREQUENCIES[:1], **MS_OPTIONS).metadata["ports"][0]
    assert narrow["kind"] == "delta_gap" and narrow["width_m"] == WIDTH


# CPW on a dielectric half-space (zero-thickness metal at the interface).
EPS_R, CPW_Z = 11.9, 100e-6
S, SLOT, GROUND, END = 10e-6, 10e-6, 40e-6, 10e-6
CPW_FREQUENCIES = (10e9, 20e9)
CPW_OPTIONS = dict(bridge_width_m=2e-6, max_edge_length_m=20e-6, edge_cell_m=1.25e-6)


def cpw(length, *, edge=True):
    stack = mw.mom.LayerStack((mw.mom.Layer("substrate", CPW_Z, EPS_R),), mw.mom.HalfSpace(EPS_R), mw.mom.HalfSpace())
    outer, inner = S / 2 + SLOT + GROUND, S / 2 + SLOT
    # Ground frame: side grounds joined by end bars beyond each line end.
    scene = mw.Scene(background=stack, structures=(
                sheet("gnd", rectangle(-GROUND, length + GROUND, -outer, outer), CPW_Z,
              holes=(rectangle(0., length, -inner, inner),)),
                sheet("sig", rectangle(END, length - END, -S / 2, S / 2), CPW_Z),
            ))
    scene = scene.updated(ports=(
        *scene.ports,
        *(edge_port(scene, name, signal="sig", side=side, reference="gnd") if edge else
          mw.LumpedPort(name, positive=(x, 0., CPW_Z), negative=(ground, 0., CPW_Z),
                        voltage_path=mw.AxisPath("x")) for name, side, x, ground in (("p1", "-x", END, 0.), ("p2", "+x", length - END, length))),
    ))
    return scene


def cpw_reference():
    """Finite-ground CPW on a half-space (Ghione-Naldi conformal map, quasi-static)."""
    a, b, c = S / 2, S / 2 + SLOT, S / 2 + SLOT + GROUND
    k = (a / b) * math.sqrt((1 - b * b / (c * c)) / (1 - a * a / (c * c)))
    eps_eff = (EPS_R + 1) / 2
    return eps_eff, ETA0 / (4 * math.sqrt(eps_eff)) * ellipk(1 - k * k) / ellipk(k * k)


@requires_cuda
@pytest.mark.cuda
def test_cpw_edge_ports(record_property):
    edge = [solve(cpw(length), CPW_FREQUENCIES, **CPW_OPTIONS).network for length in (.6e-3, 1.2e-3)]
    gap = [solve(cpw(length, edge=False), CPW_FREQUENCIES, **CPW_OPTIONS).network for length in (.6e-3, 1.2e-3)]
    line, narrow = line_parameters(*edge, .6e-3), line_parameters(*gap, .6e-3)
    expected_eps, expected_z0 = cpw_reference()
    record_property("eps_eff", line["eps_eff"].tolist())
    record_property("Z0", line["impedance_two_line"].real.tolist())
    record_property("delta_gap_eps_eff", narrow["eps_eff"].tolist())
    record_property("delta_gap_Z0", narrow["impedance_two_line"].real.tolist())
    record_property("reference", (expected_eps, expected_z0))
    assert ((line["eps_eff"] / expected_eps - 1).abs() < .01).all()
    # Z0 converges first order in the edge cell (coplanar-strip gate: +1.5% at 1.25 um).
    assert ((line["impedance_two_line"].real / expected_z0 - 1).abs() < .03).all()
    # The propagation constant is port-independent; only the port cut lines change the mesh.
    assert ((line["eps_eff"] / narrow["eps_eff"] - 1).abs() < .005).all()
    for network in (*edge, *gap):
        asymmetry, singular = checks(network)
        assert asymmetry <= 1e-10 and singular <= 1 + 1e-9


def finite_ground_microstrip(length):
    stack = mw.mom.LayerStack((mw.mom.Layer("substrate", H, 9.8),), mw.mom.HalfSpace(), mw.mom.HalfSpace())
    scene = mw.Scene(background=stack, structures=(
                sheet("gnd", rectangle(0., length, -5 * WIDTH, 5 * WIDTH), 0.),
                sheet("sig", rectangle(0., length, -WIDTH / 2, WIDTH / 2), H),
            ))
    scene = scene.updated(ports=(
        *scene.ports,
        *(edge_port(scene, name, signal="sig", side=side, reference="gnd") for name, side in (("p1", "-x"), ("p2", "+x"))),
    ))
    return scene


@requires_cuda
@pytest.mark.cuda
def test_vertical_interlayer_edge_ports(record_property):
    results = [solve(finite_ground_microstrip(length), (100e6, 1e9), bridge_width_m=WIDTH,
                     max_edge_length_m=H / 2, edge_cell_m=WIDTH / 16) for length in (2e-3, 4e-3)]
    line = line_parameters(results[0].network, results[1].network, 2e-3)
    expected_eps, expected_z0 = hammerstad_jensen_microstrip(1., 9.8)
    record_property("eps_eff", line["eps_eff"].tolist())
    record_property("Z0", line["impedance_two_line"].real.tolist())
    record_property("unknowns", results[1].network.metadata["unknowns"])
    record_property("reference", (expected_eps, expected_z0))
    record = results[0].metadata["ports"][0]
    assert (record["orientation"], record["reference"], record["reference_kind"]) == ("vertical", "gnd", "conductor")
    # A 10 W wide ground sheet replaces the infinite plane: its fringing and return-current
    # spreading move eps_eff/Z0 by O(1%) from the infinite-ground closed form.
    assert ((line["eps_eff"] / expected_eps - 1).abs() < .02).all()
    assert ((line["impedance_two_line"].real / expected_z0 - 1).abs() < .03).all()
    for result in results:
        asymmetry, singular = checks(result.network)
        assert asymmetry <= 1e-10 and singular <= 1 + 1e-9


SPACING = H / 2


def coupled_microstrip(length):
    stack = mw.mom.LayerStack((mw.mom.Layer("substrate", H, 9.8),), "pec", mw.mom.HalfSpace())
    scene = mw.Scene(background=stack)
    offset = (WIDTH + SPACING) / 2
    scene = scene.updated(structures=(
        *scene.structures,
        *(sheet(name, rectangle(0., length, centre - WIDTH / 2, centre + WIDTH / 2), H) for name, centre in (("a", offset), ("b", -offset))),
    ))
    scene = scene.updated(ports=(
        *scene.ports,
        *(edge_port(scene, name, signal=signal, side=side, reference="pec_bottom") for name, signal, side in (("a1", "a", "-x"), ("b1", "b", "-x"), ("a2", "a", "+x"), ("b2", "b", "+x"))),
    ))
    return scene


@requires_cuda
@pytest.mark.cuda
def test_coupled_microstrip_mixed_mode(record_property):
    pairs = (PortPair("in", "a1", "b1"), PortPair("out", "a2", "b2"))
    modes = [mixed_mode(solve(coupled_microstrip(length), MS_FREQUENCIES[:1], **MS_OPTIONS).network, pairs)
             for length in (2e-3, 4e-3)]
    assert modes[0].port_names == ("in:d", "in:c", "out:d", "out:c")
    assert modes[0].metadata["mixed_mode_pairs"][0]["differential"].startswith("Vd = Vp - Vn")
    odd = line_parameters(*modes, 2e-3, ports=("in:d", "out:d"))
    even = line_parameters(*modes, 2e-3, ports=("in:c", "out:c"))
    eps_even, eps_odd, z_even, z_odd = kirschning_jansen_coupled_microstrip(1., SPACING / H, 9.8)
    measured = dict(eps_odd=odd["eps_eff"].item(), eps_even=even["eps_eff"].item(),
                    z_odd=odd["impedance_two_line"].real.item() / 2, z_even=2 * even["impedance_two_line"].real.item())
    record_property("measured", measured)
    record_property("reference", dict(eps_odd=eps_odd, eps_even=eps_even, z_odd=z_odd, z_even=z_even))
    # Kirschning-Jansen accuracy is about 1% (eps) and 2% (Z) for these u, g; mesh adds ~1%.
    assert abs(measured["eps_odd"] / eps_odd - 1) < .02 and abs(measured["eps_even"] / eps_even - 1) < .02
    assert abs(measured["z_odd"] / z_odd - 1) < .03 and abs(measured["z_even"] / z_even - 1) < .03
    for network in modes:
        asymmetry, singular = checks(network)
        assert asymmetry <= 1e-10 and singular <= 1 + 1e-9


def test_mixed_mode_is_the_terminal_basis_change():
    from witwin.maxwell.mom.layered import extraction as ex
    generator = torch.Generator().manual_seed(3)
    a = torch.randn((3, 4, 4), generator=generator, dtype=torch.complex128)
    z = a @ a.transpose(-1, -2) + 4 * torch.eye(4)
    network = NetworkData.from_z(frequencies=torch.tensor([1e9, 2e9, 3e9], dtype=torch.float64), z=z,
                                 z0=[50.] * 4, port_names=("a1", "b1", "a2", "b2"))
    modes = mixed_mode(network, (PortPair("in", "a1", "b1"),))
    assert modes.port_names == ("in:d", "in:c", "a2", "b2")
    differential = ex.terminal_impedance(network, ("a1", "b1"), others="open")
    assert torch.allclose(modes.to_z()[:, 0, 0], differential, rtol=1e-12, atol=0)
    with pytest.raises(TypeError, match="PortPair"):
        mixed_mode(network, (("a1", "b1"),))
    with pytest.raises(ValueError, match="unique"):
        mixed_mode(network, (PortPair("in", "a1", "b1"), PortPair("in", "a2", "b2")))
    with pytest.raises(ValueError, match="distinct"):
        PortPair("x", "a1", "a1")


# Host-only refusals: every one raises before any device work.
def strips():
    return [(0., 40e-6, 0., 20e-6, 0.), (60e-6, 100e-6, 0., 20e-6, 0.)]


def port(aperture, *, positive=(60e-6, 10e-6, 0.), negative=(40e-6, 10e-6, 0.), axis="x"):
    return mw.LumpedPort("p", positive=positive, negative=negative, voltage_path=mw.AxisPath(axis),
                         voltage_aperture=aperture)


def box(position, size, rotation=None):
    return mw.Box(position=torch.tensor(position, dtype=torch.float64), size=torch.tensor(size, dtype=torch.float64),
                  **({} if rotation is None else dict(rotation=rotation)))


def prepare(rectangles, ports, stack=None, **options):
    stack = stack or mw.mom.LayerStack((mw.mom.Layer("air", 1e-3),), mw.mom.HalfSpace(), mw.mom.HalfSpace())
    return prepare_layered_network(stack, rectangles, ports, (1e9,), bridge_width=2e-6, max_edge=1e-5,
                                   device="cpu", **options)


REFUSALS = (
    ("z_extent", lambda: prepare(strips(), [port(box((50e-6, 10e-6, 0.), (20e-6, 20e-6, 1e-6)))]),
     ValueError, "zero z extent"),
    ("span", lambda: prepare(strips(), [port(box((50e-6, 10e-6, 0.), (30e-6, 20e-6, 0.)))]),
     ValueError, "span exactly the terminal path along x"),
    ("outside", lambda: prepare(strips(), [port(box((50e-6, 30e-6, 0.), (20e-6, 10e-6, 0.)))]),
     ValueError, "contain the terminal path"),
    ("uncovered", lambda: prepare(strips(), [port(box((50e-6, 10e-6, 0.), (20e-6, 30e-6, 0.)))]),
     ValueError, "not covered by metal"),
    ("interior_metal", lambda: prepare(strips() + [(48e-6, 52e-6, 14e-6, 18e-6, 0.)],
                                       [port(box((50e-6, 10e-6, 0.), (20e-6, 20e-6, 0.)))]),
     ValueError, "interior overlaps metal"),
    ("rotated", lambda: prepare(strips(), [port(mw.Box(position=(50e-6, 10e-6, 0.), size=(20e-6, 20e-6, 0.),
                                                       rotation=(0.9238795, 0., 0., 0.3826834)))]),
     NotImplementedError, "axis-aligned Box"),
    ("vertical_two_widths", lambda: prepare([(0., 40e-6, 0., 20e-6, 1e-4)], [port(
        box((0., 10e-6, 5e-5), (10e-6, 20e-6, 1e-4)), positive=(0., 10e-6, 1e-4), negative=(0., 10e-6, 0.), axis="z")],
        stack=mw.mom.LayerStack((mw.mom.Layer("sub", 1e-4, 4.),), "pec", mw.mom.HalfSpace())),
     ValueError, "zero extent along exactly one"),
    ("vertical_span", lambda: prepare([(0., 40e-6, 0., 20e-6, 1e-4)], [port(
        box((0., 10e-6, 4e-5), (0., 20e-6, 8e-5)), positive=(0., 10e-6, 1e-4), negative=(0., 10e-6, 0.), axis="z")],
        stack=mw.mom.LayerStack((mw.mom.Layer("sub", 1e-4, 4.),), "pec", mw.mom.HalfSpace())),
     ValueError, "span exactly the terminal path along z"),
    ("vertical_interior", lambda: prepare([(-40e-6, 40e-6, 0., 20e-6, 1e-4)], [port(
        box((0., 10e-6, 5e-5), (0., 20e-6, 1e-4)), positive=(0., 10e-6, 1e-4), negative=(0., 10e-6, 0.), axis="z")],
        stack=mw.mom.LayerStack((mw.mom.Layer("sub", 1e-4, 4.),), "pec", mw.mom.HalfSpace())),
     NotImplementedError, "interior attachments need junction RWGs"),
)


@pytest.mark.parametrize("call,error,message", [case[1:] for case in REFUSALS], ids=[case[0] for case in REFUSALS])
def test_port_refusals(call, error, message):
    with pytest.raises(error, match=message):
        call()


def test_edge_port_builder_refusals():
    stack = mw.mom.LayerStack((mw.mom.Layer("sub", 1e-4, 4.),), mw.mom.HalfSpace(), mw.mom.HalfSpace())
    scene = mw.Scene(background=stack, structures=(
                sheet("sig", rectangle(0., 1e-4, 0., 2e-5), 1e-4),
                sheet("narrow", rectangle(1.2e-4, 1.4e-4, 0., 1e-5), 1e-4),
                sheet("comb", [(0., 0.), (4e-5, 0.), (4e-5, 1e-5), (2e-5, 1e-5), (2e-5, 2e-5), (4e-5, 2e-5),
                       (4e-5, 3e-5), (0., 3e-5)], 0.),
            ))
    cases = (
        (dict(signal="missing", side="-x", reference="pec_bottom"), "exactly one enabled Structure named 'missing'"),
        (dict(signal="sig", side="x", reference="pec_bottom"), "side must be one of"),
        (dict(signal="sig", side="-x", reference="pec_bottom"), "needs LayerStack\\(bottom='pec'\\)"),
        (dict(signal="sig", side="+x", reference="narrow"), "no edge facing"),
        (dict(signal="sig", side="-x", reference="sig2"), "exactly one enabled Structure"),
        (dict(signal="comb", side="+x", reference="pec_bottom"), "single contiguous edge"),
    )
    for kwargs, message in cases:
        with pytest.raises(ValueError, match=message):
            edge_port(scene, "p", **kwargs)


def test_edge_port_builder_geometry():
    stack = mw.mom.LayerStack((mw.mom.Layer("sub", 1e-4, 4.),), "pec", mw.mom.HalfSpace())
    scene = mw.Scene(background=stack, structures=(
                sheet("sig", rectangle(1e-5, 1e-4, -1e-5, 1e-5), 1e-4),
                sheet("gnd", rectangle(-5e-5, 0., -4e-5, 4e-5), 1e-4),
            ))
    horizontal = edge_port(scene, "h", signal="sig", side="-x", reference="gnd")
    assert (horizontal.positive, horizontal.negative, horizontal.voltage_path.axis) == ((1e-5, 0., 1e-4), (0., 0., 1e-4), "x")
    assert horizontal.voltage_aperture.size.tolist() == [1e-5, 2e-5, 0.]
    vertical = edge_port(scene, "v", signal="sig", side="+x", reference="pec_bottom")
    assert (vertical.positive, vertical.negative, vertical.voltage_path.axis) == ((1e-4, 0., 1e-4), (1e-4, 0., 0.), "z")
    assert vertical.voltage_aperture.size.tolist() == [0., 2e-5, 1e-4]
