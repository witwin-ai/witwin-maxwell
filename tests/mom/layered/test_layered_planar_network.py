"""B1/B2 network gates of polygon layouts through Scene -> Simulation -> Result.

Circular loop against the Grover/Kirchhoff loop, octagonal against Manhattan
spiral (modified-Wheeler reference), mesh-refinement monotonicity, determinism,
reciprocity, passivity, polygonal and multi-plane vias with KCL.
"""

import gc
import math

import pytest
import torch

import witwin.maxwell as mw
from tests.support.layered_via_references import rectangular_loop_inductance
from tests.support.rfic_planar_fixtures import (
    UM, circular_loop_inductance, gap_port, modified_wheeler, polygon_structure, regular_polygon_via_radius,
    ring_loop, spiral_scene, vacuum)

shapely = pytest.importorskip("shapely")


pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")]


@pytest.fixture(autouse=True)
def release():
    yield
    gc.collect()
    torch.cuda.empty_cache()


def config(bridge, edge, *, max_edge=20., mesher="auto"):
    # Polygon layouts are sheet-only; auto would select surface_admittance for thick copper (A2).
    return mw.mom.MoMConfig(layered=mw.mom.LayeredSurfaceConfig(bridge_width_m=bridge * UM, max_edge_length_m=max_edge * UM,
                                                        edge_cell_m=edge * UM, planar_mesher=mesher, conductor_model="sheet"))


def inductance(result):
    frequency = result.network.frequencies[0].item()
    return -result.network.to_z()[0, 0, 0].imag.item() / (2 * math.pi * frequency)


def test_circular_loop_grover_and_monotone_refinement(record_property):
    """PEC 96-gon strip loop (R=250 um, w=20 um) at 100 MHz; edge cells 6/3/2 um.

    L falls in steps when another triangle row fits across the strip (measured
    +2.72/+1.20/+0.645% and +0.637% at 1.75 um); edge cells that keep the row
    count change L by <= 0.02%. The converged value is within 1% of Grover.
    """
    reference = circular_loop_inductance(250 * UM, 20 * UM)
    values, unknowns = [], []
    for edge in (6., 3., 2.):
        scene, bridge = ring_loop()
        result = mw.Simulation.mom(scene, frequencies=(1e8,), config=config(bridge, edge)).run()
        values.append(inductance(result))
        unknowns.append(result.network.metadata["unknowns"])
        assert result.network.metadata["mesher"] == "planar_conforming_delaunay_v1"
    errors = [v / reference - 1 for v in values]
    record_property("inductance_H", values)
    record_property("unknowns", unknowns)
    record_property("grover_H", reference)
    record_property("relative", errors)
    assert unknowns == sorted(unknowns)
    assert abs(errors[-1]) <= .01
    # Monotone convergence from above with shrinking increments.
    assert values[0] > values[1] > values[2]


def test_octagonal_and_manhattan_spirals(record_property):
    """PEC one-port spirals at 100 MHz on the F-A stack.

    The Manhattan spiral through the triangle mesher agrees with the rectangle
    mesher; the octagonal/square ratio agrees with the modified-Wheeler ratio
    (leads and underpass add nearly equal partial inductance to both).
    """
    def solve(octagonal, mesher, edge=5.):
        result = mw.Simulation.mom(spiral_scene(octagonal), frequencies=(1e8,),
                                   config=config(5., edge, mesher=mesher)).run()
        return inductance(result), result.network.metadata

    square_rect, rect_meta = solve(False, "auto")
    square_tri, tri_meta = solve(False, "triangle")
    octagon, oct_meta = solve(True, "auto")
    assert "mesher" not in rect_meta and tri_meta["mesher"] == oct_meta["mesher"]
    wheeler = (modified_wheeler(200 * UM, 105 * UM, 3.5, True) / modified_wheeler(200 * UM, 105 * UM, 3.5, False))
    ratio = octagon / square_tri
    record_property("square_rectangle_H", square_rect)
    record_property("square_triangle_H", square_tri)
    record_property("octagon_H", octagon)
    record_property("ratio", ratio)
    record_property("wheeler_ratio", wheeler)
    record_property("unknowns", (rect_meta["unknowns"], tri_meta["unknowns"], oct_meta["unknowns"]))
    assert abs(square_tri / square_rect - 1) <= .01
    assert octagon < square_tri
    assert abs(ratio / wheeler - 1) <= .03
    # Octagonal refinement series (edge cells 6/5/2.5 um) decreases monotonically.
    series = [solve(True, "auto", 6.)[0], octagon, solve(True, "auto", 2.5)[0]]
    record_property("octagon_series_H", series)
    assert series[0] > series[1] > series[2]


def two_rings(lossy):
    """Concentric gapped octagon-like 64-gon rings (bottom/top gaps) with one port each."""
    scene = mw.Scene(background=vacuum())
    material = mw.media.LossyMetalMedium(conductivity=5.8e7) if lossy else mw.Material.pec()
    for name, radius, sign in (("outer", 200., -1), ("inner", 150., 1)):
        ring = shapely.Polygon([(radius * math.cos(2 * math.pi * k / 64 + math.pi / 64),
                                 radius * math.sin(2 * math.pi * k / 64 + math.pi / 64)) for k in range(64)])
        ring = ring.difference(ring.buffer(-10, join_style="mitre"))
        flat = radius * math.cos(math.pi / 64) - 5
        ring = ring.difference(shapely.box(-5, sign * flat - 10, 5, sign * flat + 10))
        scene = scene.updated(structures=(
            *scene.structures,
            polygon_structure(ring, 100., material, 1. if lossy else 0.),
        ), ports=(*scene.ports, gap_port(name, (5, sign * flat, 100.), (-5, sign * flat, 100.), "x")))
    return scene


def test_two_port_reciprocity_passivity_and_determinism(record_property):
    frequencies = (1e8, 1e9, 1e10)
    first = mw.Simulation.mom(two_rings(True), frequencies=frequencies, config=config(6., 4.)).run()
    y = first.network.to_y()
    reciprocal = ((y - y.transpose(-1, -2)).abs().amax() / y.abs().amax()).item()
    singular = torch.linalg.svdvals(first.network.s).amax().item()
    record_property("reciprocity", reciprocal)
    record_property("max_singular_s", singular)
    record_property("mesh_sha256", first.network.metadata["mesh_sha256"])
    assert reciprocal <= 1e-10
    assert singular <= 1 + 1e-9
    for _ in range(2):
        again = mw.Simulation.mom(two_rings(True), frequencies=frequencies, config=config(6., 4.)).run()
        assert again.network.metadata["mesh_sha256"] == first.network.metadata["mesh_sha256"]
        assert torch.equal(again.network.s, first.network.s)
        assert torch.equal(mw.mom.surface_currents(again).currents, mw.mom.surface_currents(first).currents)


def via_loop(case):
    """Vertical rectangular loop (length 200 um) with octagonal Prism vias.

    "polygon": consecutive planes 100/300 um. "landing": vias 100 -> 300 um pass
    a 20x20 um landing pad at 200 um. "stacked": two vias per side meet on that pad.
    "pass_through": vias 100 -> 300 um clear unrelated metal on the 200 um plane.
    """
    length, width, gap, za, zm, zb = 200., 10., 10., 100., 200., 300.
    scene = mw.Scene(background=vacuum())
    pec = mw.Material.pec()
    scene = scene.updated(structures=(
        *scene.structures,
        *(polygon_structure(shapely.box(box[0], box[2], box[1], box[3]), z, pec) for box, z in (((-width, length + width, -width / 2, width / 2), zb),
                   ((-width, length / 2 - gap / 2, -width / 2, width / 2), za),
                   ((length / 2 + gap / 2, length + width, -width / 2, width / 2), za))),
    ))
    spans = [(za, zm), (zm, zb)] if case == "stacked" else [(za, zb)]
    if case == "pass_through":
        scene = scene.updated(structures=(*scene.structures, polygon_structure(shapely.box(90, 30, 110, 50), zm, pec)))
    for x in (0., length):
        if case in ("landing", "stacked"):
            scene = scene.updated(structures=(
                *scene.structures,
                polygon_structure(shapely.box(x - 10, -10, x + 10, 10), zm, pec),
            ))
        scene = scene.updated(structures=(
            *scene.structures,
            *(mw.Structure(mw.geometry.Prism(
                  position=torch.tensor((x, 0., (z0 + z1) / 2), dtype=torch.float64) * UM,
                  radius=torch.tensor(3. * UM, dtype=torch.float64), height=torch.tensor((z1 - z0) * UM,
                  dtype=torch.float64), num_sides=8, axis="z"), pec) for z0, z1 in spans),
        ))
    scene = scene.updated(ports=(
        *scene.ports,
        gap_port("loop", (length / 2 + gap / 2, 0, za), (length / 2 - gap / 2, 0, za), "x"),
    ))
    return scene, (length, zb - za, width)


@pytest.mark.parametrize("case", ["polygon", "pass_through", "landing", "stacked"])
def test_polygonal_and_multi_plane_via_loops(case, record_property):
    """B2: octagonal via tubes, a via passing a landing pad, and stacked vias: L and KCL at 1 MHz."""
    scene, (length, height, width) = via_loop(case)
    simulation = mw.Simulation.mom(scene, frequencies=(1e6,), config=config(10., 2.))
    prepared = simulation.prepare()
    result = prepared.run()
    network, currents = result.network, mw.mom.surface_currents(result).currents
    value = inductance(result)
    # The loop reference uses the square-via form with the octagon's equivalent radius.
    radius = regular_polygon_via_radius(3. * UM, 8)
    side = radius * 4 * math.pi ** 1.5 / math.gamma(.25) ** 2
    expected = rectangular_loop_inductance(length * UM, height * UM, width * UM, side)
    layout = prepared.solver.prepared.layout
    basis, mesh = layout.basis, layout.mesh
    a, b = mesh.vertices[basis.edges[:, 0]], mesh.vertices[basis.edges[:, 1]]
    flow = mesh.centroids[basis.minus_faces] - mesh.centroids[basis.plus_faces]
    terminal = (prepared.solver.prepared.excitation[:, 0] * currents[0, :, 0]).sum()
    kcl = []
    for x in (0., length * UM):
        tube = ((a[:, 2] == b[:, 2]) & (a[:, 2] > 100 * UM) & (a[:, 2] < 300 * UM)
                & (a[:, 2] != 200 * UM) & ((a[:, :2] - a.new_tensor((x, 0.))).norm(dim=-1) <= 3 * UM * (1 + 1e-9))
                & ((b[:, :2] - a.new_tensor((x, 0.))).norm(dim=-1) <= 3 * UM * (1 + 1e-9)))
        levels = a[tube, 2].unique()
        assert len(levels) >= 3
        for z in levels:
            cut = tube & (a[:, 2] == z)
            current = (torch.sign(flow[cut, 2]) * basis.edge_lengths[cut] * currents[0, cut, 0]).sum()
            kcl.append(abs(abs((current / terminal).item()) - 1))
    junctions = network.metadata.get("junction_edges", 0)
    record_property("inductance_H", value)
    record_property("reference_H", expected)
    record_property("kcl_max", max(kcl))
    record_property("junction_edges", junctions)
    record_property("unknowns", network.metadata["unknowns"])
    assert (junctions > 0) == (case in ("landing", "stacked"))
    assert abs(value / expected - 1) <= .03
    assert max(kcl) <= 1e-8
