"""R6-4 vertical ports on closed bodies: the vertical thick-terminal (PEC block) port.

A vertical port between closed bodies on consecutive metal planes is an ideal PEC block from the
lower body's top face to the upper body's bottom face, spanning the aperture (bridge) width along
the shared edge and the thinner terminal thickness into the metal, so its outer wall is coplanar
with both terminal end faces; the delta gap is its mid-height ring. A vertical port to a stack PEC
ground stays refused (the closed junction basis has no grounded half RWGs).
Host: block geometry and refusals. CUDA: see the line gates below.
"""

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.layered.network import _block_boxes, _closed_geometry

AIR = mw.mom.LayerStack((mw.mom.Layer("air", 100.),), mw.mom.HalfSpace(), mw.mom.HalfSpace())
PEC = mw.mom.LayerStack((mw.mom.Layer("air", 100.),), "pec", mw.mom.HalfSpace())


def vertical(fixed=0., lo=-1., hi=1., za=10., zb=20.):
    """Sheet-model vertical bridge record (normal x, fixed, lo, hi, za, zb, cut, centre, sign)."""
    return (0, fixed, lo, hi, za, zb, (za + zb) / 2, (lo + hi) / 2, 1.)


def test_vertical_block_geometry():
    rectangles = [(0., 50., -5., 5., 10.), (0., 50., -1., 1., 20.)]
    materials = [(None, 2., None), (None, 1., None)]
    boxes, lifted, _ = _closed_geometry(rectangles, materials, [], [vertical()], AIR)
    (record,) = lifted
    block, axis, cut, centre, direction = record
    # Depth = thinner terminal (1); z from the lower top face (11) to the upper bottom face (19.5).
    assert block == (0., 1., -1., 1., 11., 19.5) and axis == 2 and cut == 15.25
    assert (centre, direction) == (0., 1.)
    assert _block_boxes(lifted) == [(0., 1., -1., 1., 11., 15.25), (0., 1., -1., 1., 15.25, 19.5)]
    # Metal on the -x side of a +x edge: the block extends to -x.
    rectangles = [(-50., 0., -5., 5., 10.), (-50., 0., -1., 1., 20.)]
    _, lifted, _ = _closed_geometry(rectangles, [(None, 2., None)] * 2, [], [vertical()], AIR)
    assert lifted[0][0] == (-2., 0., -1., 1., 11., 19.)


@pytest.mark.parametrize("case", ["opposite_sides", "pec_ground", "short_footprint"])
def test_vertical_block_refusals(case):
    stack = AIR
    if case == "opposite_sides":
        rectangles, record = [(-50., 0., -5., 5., 10.), (0., 50., -1., 1., 20.)], vertical()
    elif case == "pec_ground":
        stack, rectangles, record = PEC, [(0., 50., -1., 1., 20.)], vertical(za=0.)
    else:
        # The upper body is shorter along x than the block depth.
        rectangles, record = [(0., 50., -5., 5., 10.), (0., 1., -1., 1., 20.)], vertical()
    with pytest.raises(NotImplementedError, match="Closed vertical ports need terminal bodies"):
        _closed_geometry(rectangles, [(None, 2., None)] * len(rectangles), [], [record], stack)


# CUDA gates: a two-strip line (10 um signal over a 40 um ground strip, planes 10 um apart) in a
# homogeneous eps_r = 4 oxide, full-width vertical inter-layer edge ports at both ends. Every PEC
# line in a homogeneous medium is TEM: eps_eff = eps_r exactly, whatever the cross-section, and the
# two-line method cancels the port blocks. Sized for a 6 GiB cap (2 um edge cell; the 1 and 0.5 um
# probes of scripts/dev/rfic/r6_4_probes.py give 3.8e-4 and 2.6e-4).
from witwin.maxwell.mom.layered.deembed import line_parameters  # noqa: E402
from witwin.maxwell.mom.layered.port_system import edge_port  # noqa: E402

UM, EPS_R, LINE_FREQUENCIES, LENGTHS = 1e-6, 4.0, (20e9, 40e9), (200., 400.)
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")


def two_strip_line(length, model, thickness=2.):
    stack = mw.mom.LayerStack((mw.mom.Layer("oxide", 40 * UM, EPS_R),), mw.mom.HalfSpace(EPS_R), mw.mom.HalfSpace(EPS_R))
    scene = mw.Scene(background=stack)
    for name, z, width in (("gnd", 15 * UM, 40 * UM), ("sig", 25 * UM, 10 * UM)):
        vertices = torch.tensor(((0., -width / 2), (length, -width / 2), (length, width / 2), (0., width / 2)),
                                dtype=torch.float64)
        extra = {} if model == "sheet" else dict(physical_thickness=thickness * UM)
        scene = scene.updated(structures=(
            *scene.structures,
            mw.Structure(mw.geometry.PlanarPEC(vertices, normal_axis="z", coordinate=z, **extra),
                         mw.Material.pec(), name=name),
        ))
    scene = scene.updated(ports=(
        *scene.ports,
        *(edge_port(scene, name, signal="sig", side=side, reference="gnd") for name, side in (("p1", "-x"), ("p2", "+x"))),
    ))
    config = mw.mom.MoMConfig(layered=mw.mom.LayeredSurfaceConfig(
        bridge_width_m=10 * UM, max_edge_length_m=10 * UM, edge_cell_m=2 * UM, conductor_model=model))
    return mw.Simulation.mom(scene, frequencies=LINE_FREQUENCIES, config=config).run()


@pytest.fixture(scope="module")
def closed_lines():
    return {length: two_strip_line(length * UM, "closed") for length in LENGTHS}


@cuda
@pytest.mark.cuda
@pytest.mark.slow  # 131 s on an A6000 (two lines plus a repeat), 1.2 GiB peak
def test_closed_vertical_port_line(closed_lines, record_property):
    torch.cuda.reset_peak_memory_stats()
    short, long = (closed_lines[length] for length in LENGTHS)
    line = line_parameters(short.network, long.network, (LENGTHS[1] - LENGTHS[0]) * UM)
    record_property("eps_eff", line["eps_eff"].tolist())
    record_property("Z0_two_line", [str(z) for z in line["impedance_two_line"].tolist()])
    record_property("unknowns", long.network.metadata["unknowns"])
    assert ((line["eps_eff"] / EPS_R - 1).abs() < 5e-3).all()
    metadata = short.network.metadata
    assert metadata["port_attachment"] == "pec_block_end_faces"
    first = metadata["ports"][0]
    assert (first["orientation"], first["reference"], first["bridge"]) == ("vertical", "gnd", "ideal_pec_block")
    # Block: x 0..2 um (terminal thickness), y -5..5 um, z 16..24 um (gnd top to sig bottom); cut 20 um.
    expected = (0., 2., -5., 5., 16., 24.)
    assert all(abs(v / UM - e) < 1e-9 for v, e in zip(metadata["port_blocks"][0], expected))
    assert abs(first["cut_m"] / UM - 20.) < 1e-9
    for result in (short, long):
        y = result.network.to_y()
        assert ((y - y.transpose(-1, -2)).abs().max() / y.abs().max()).item() <= 1e-10
        assert torch.linalg.matrix_norm(result.network.s, ord=2).max().item() <= 1 + 1e-9
    again = two_strip_line(LENGTHS[0] * UM, "closed")
    assert torch.equal(again.network.s, short.network.s)
    assert torch.equal(mw.mom.surface_currents(again).currents, mw.mom.surface_currents(short).currents)
    record_property("peak_allocated_bytes", torch.cuda.max_memory_allocated())


@cuda
@pytest.mark.cuda
@pytest.mark.slow  # shares the two-line fixture
def test_surface_admittance_vertical_port_equals_closed_for_pec(closed_lines):
    """For PEC bodies the surface_admittance block port is the closed one, bitwise."""
    result = two_strip_line(LENGTHS[0] * UM, "surface_admittance")
    assert result.network.metadata["ports"][0]["bridge"] == "ideal_pec_block"
    assert torch.equal(result.network.s, closed_lines[LENGTHS[0]].network.s)
