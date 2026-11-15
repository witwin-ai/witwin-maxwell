"""B1 host gates of the conforming planar mesher: determinism, conformity, grading, rims and refusals."""

import math

import numpy as np
import pytest

from witwin.maxwell.mom.layered.planar_mesh import (
    GATE_ASPECT, PolygonSheet, PolygonVia, check_quality, planar_triangulation)


UM = 1e-6


def rect(x0, x1, y0, y1):
    return ((x0 * UM, y0 * UM), (x1 * UM, y0 * UM), (x1 * UM, y1 * UM), (x0 * UM, y1 * UM))


def regular(radius, sides, phase=0.0, centre=(0.0, 0.0)):
    return tuple((centre[0] * UM + radius * UM * math.cos(phase + 2 * math.pi * k / sides),
                  centre[1] * UM + radius * UM * math.sin(phase + 2 * math.pi * k / sides)) for k in range(sides))


def octagon_trace():
    """45-degree routed trace with a hole and an arc-discretized ring on a second plane."""
    outer = regular(100, 8, math.pi / 8)
    inner = regular(85, 8, math.pi / 8)
    ring = (regular(60, 48), regular(50, 48))
    return [PolygonSheet((outer, inner), 1 * UM), PolygonSheet(ring, 3 * UM)]


def edge_counts(faces):
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), 1)
    return np.unique(edges, axis=0, return_counts=True)


def test_deterministic_conforming_graded_mesh(record_property):
    sheets = octagon_trace()
    first = planar_triangulation(sheets, (UM, 3 * UM), 20 * UM, h_edge=4 * UM, growth_ratio=1.5)
    second = planar_triangulation(sheets, (UM, 3 * UM), 20 * UM, h_edge=4 * UM, growth_ratio=1.5)
    assert first.report["mesh_sha256"] == second.report["mesh_sha256"]
    assert np.array_equal(first.vertices, second.vertices) and np.array_equal(first.faces, second.faces)
    check_quality(first.report)
    record_property("report", first.report)
    vertices, faces = first.vertices, first.faces
    edges, counts = edge_counts(faces)
    assert counts.max() == 2
    # No hanging node: no vertex strictly inside any mesh edge of its plane.
    a, b = vertices[edges[:, 0]], vertices[edges[:, 1]]
    for start in range(0, len(vertices), 256):
        delta = vertices[start:start + 256, None] - a[None]
        collinear = np.linalg.norm(np.cross(delta, (b - a)[None]), axis=-1) < 1e-9 * np.linalg.norm(b - a, axis=-1)
        inside = (delta * (vertices[start:start + 256, None] - b[None])).sum(-1) < -1e-24
        assert not (collinear & inside).any()
    # Boundary edges lie on polygon boundaries; the faces tile each polygon exactly.
    import shapely
    for plane, sheet in enumerate(sheets):
        polygon = shapely.Polygon(sheet.loops[0], sheet.loops[1:])
        mask = first.planes == plane
        corners = vertices[faces[mask]]
        area = 0.5 * np.abs(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])[:, 2]).sum()
        assert abs(area / polygon.area - 1) < 1e-12
        local, local_counts = edge_counts(faces[mask])
        middle = (vertices[local[local_counts == 1, 0]] + vertices[local[local_counts == 1, 1]])[:, :2] / 2
        assert shapely.distance(polygon.boundary, shapely.points(middle)).max() < 1e-15
    # Grading: every edge <= h, boundary edges <= h_edge, and quality inside the gate.
    lengths = np.linalg.norm(a - b, axis=-1)
    assert lengths.max() <= math.sqrt(2) * 20 * UM * (1 + 1e-12)
    assert lengths[counts == 1].max() <= 4 * UM * (1 + 1e-12)
    assert first.report["horizontal_min_angle_degrees"] >= 25
    assert first.report["horizontal_max_aspect_ratio"] <= GATE_ASPECT


def octagon_via(z0, z1):
    return PolygonVia(regular(4, 8, math.pi / 8), z0, z1)


@pytest.mark.parametrize("case", ["single", "landing", "stacked", "pass_through"])
def test_via_rims_weld_and_junctions(case, record_property):
    z = (UM, 3 * UM, 5 * UM)
    pad = rect(-15, 15, -15, 15)
    middle = PolygonSheet((rect(30, 60, -15, 15),), z[1]) if case == "pass_through" else PolygonSheet((pad,), z[1])
    sheets = [PolygonSheet((rect(-20, 20, -20, 20),), z[0]), middle, PolygonSheet((rect(-20, 20, -20, 20),), z[2])]
    vias = {"single": [octagon_via(z[0], z[1])], "landing": [octagon_via(z[0], z[2])],
            "stacked": [octagon_via(z[0], z[1]), octagon_via(z[1], z[2])],
            "pass_through": [octagon_via(z[0], z[2])]}[case]
    result = planar_triangulation(sheets, z, 5 * UM, h_edge=2 * UM, vias=vias)
    check_quality(result.report)
    record_property("report", result.report)
    edges, counts = edge_counts(result.faces)
    vertices = result.vertices
    rim_heights = {"single": (z[0], z[1]), "landing": (z[0], z[1], z[2]), "stacked": (z[0], z[1], z[2]),
                   "pass_through": (z[0], z[2])}[case]
    radius = np.hypot(vertices[edges, 0], vertices[edges, 1]).max(1)
    flat = vertices[edges[:, 0], 2] == vertices[edges[:, 1], 2]
    for height in rim_heights:
        rim = flat & (vertices[edges[:, 0], 2] == height) & (radius <= 4 * UM * (1 + 1e-12))
        assert rim.sum() >= 8
        # Endpoint rims weld one sheet and one wall; an intermediate landing welds two walls.
        expected = 3 if height == z[1] and case in ("landing", "stacked") else 2
        assert (counts[rim] == expected).all()
    assert counts.max() == (3 if case in ("landing", "stacked") else 2)
    if case in ("landing", "stacked"):
        assert result.report["mesh_sha256"] == planar_triangulation(
            sheets, z, 5 * UM, h_edge=2 * UM, vias=vias).report["mesh_sha256"]
    walls = result.planes < 0
    assert (result.owners[walls] >= len(sheets)).all()
    assert (np.ptp(vertices[result.faces[walls]][..., 2], axis=1) > 0).all()


def test_ports_bridges_cuts_and_owners():
    sheets = [PolygonSheet((rect(-50, -5, -5, 5),), 0.0), PolygonSheet((rect(5, 50, -5, 5),), 0.0)]
    cut = (0.0, (0.0, -2 * UM), (0.0, 2 * UM))
    result = planar_triangulation(sheets, (0.0,), 10 * UM, h_edge=2 * UM, bridges=[(-8 * UM, 8 * UM, -2 * UM, 2 * UM, 0.0)],
                                  cuts=[cut])
    vertices, faces = result.vertices, result.faces
    edges, counts = edge_counts(faces)
    on_cut = (vertices[edges[:, 0], 0] == 0) & (vertices[edges[:, 1], 0] == 0)
    assert on_cut.sum() >= 2 and (counts[on_cut] == 2).all()
    assert np.isclose(np.abs(vertices[edges[on_cut]][..., 1]).max(), 2 * UM, rtol=0, atol=0)
    centres = vertices[faces].mean(1)
    gap = np.abs(centres[:, 0]) < 5 * UM
    assert (result.owners[gap] == -1).all() and (result.owners[~gap] >= 0).all()
    assert set(result.owners[centres[:, 0] < -5 * UM]) == {0}


@pytest.mark.parametrize("case,exception,message", [
    ("acute", NotImplementedError, "quality gate"),
    ("partial_landing", NotImplementedError, "intermediate plane"),
    ("rim_on_edge", ValueError, "rim surrounded"),
    ("rim_on_bridge", NotImplementedError, "port bridges"),
    ("rim_on_material", NotImplementedError, "Via rims must not touch"),
    ("overlap_materials", ValueError, "identical materials"),
    ("touching_vias", NotImplementedError, "junction geometry"),
    ("stacked_mismatch", NotImplementedError, "identical footprints"),
    ("floor", ValueError, "mesh floor"),
    ("off_plane", ValueError, "Via endpoints"),
    ("sheet_off_plane", ValueError, "off every metal plane"),
    ("parameters", ValueError, "Require h>0"),
])
def test_planar_mesher_refusals(case, exception, message):
    z = (UM, 3 * UM, 5 * UM)
    big = rect(-20, 20, -20, 20)
    sheets = [PolygonSheet((big,), z[0]), PolygonSheet((rect(-15, 15, -15, 15),), z[1]), PolygonSheet((big,), z[2])]
    options = dict(vias=(), bridges=(), groups=None)
    h = 5 * UM
    if case == "acute":
        sheets = [PolygonSheet((((0, 0), (40 * UM, 0), (40 * UM, 4 * UM)),), z[0])]
    elif case == "partial_landing":
        sheets[1] = PolygonSheet((rect(-2, 15, -15, 15),), z[1])
        options["vias"] = [octagon_via(z[0], z[2])]
    elif case == "rim_on_edge":
        options["vias"] = [PolygonVia(rect(16, 20, 0, 4), z[0], z[1])]
    elif case == "rim_on_bridge":
        options["vias"] = [octagon_via(z[0], z[1])]
        options["bridges"] = [(-2 * UM, 2 * UM, -10 * UM, 10 * UM, z[0])]
    elif case == "rim_on_material":
        sheets[0:1] = [PolygonSheet((rect(-20, 0, -20, 20),), z[0]), PolygonSheet((rect(0, 20, -20, 20),), z[0])]
        options["groups"] = [0, 1, 2, 3]
        options["vias"] = [PolygonVia(rect(-2, 2, -2, 2), z[0], z[1])]
    elif case == "overlap_materials":
        sheets.append(PolygonSheet((rect(0, 30, 0, 30),), z[0]))
        options["groups"] = [0, 0, 0, 1]
    elif case == "touching_vias":
        options["vias"] = [PolygonVia(rect(-4, 0, -2, 2), z[0], z[1]), PolygonVia(rect(0, 4, -2, 2), z[0], z[1])]
    elif case == "stacked_mismatch":
        options["vias"] = [octagon_via(z[0], z[1]), PolygonVia(regular(4, 8), z[1], z[2])]
    elif case == "floor":
        sheets = [PolygonSheet((((0, 0), (40 * UM, 0), (40 * UM, 40 * UM), (40 * UM - 1e-12, 40 * UM), (0, 40 * UM)),),
                               z[0])]
    elif case == "off_plane":
        options["vias"] = [octagon_via(z[0], 2 * UM)]
    elif case == "sheet_off_plane":
        sheets.append(PolygonSheet((big,), 7 * UM))
    elif case == "parameters":
        h = -1.0
    with pytest.raises(exception, match=message):
        planar_triangulation(sheets, z, h, h_edge=2 * UM if h > 0 else None, **options)


def test_quality_gate_refuses_reports_outside_the_validated_region():
    good = dict(horizontal_min_angle_degrees=25.0, horizontal_max_aspect_ratio=3.0, wall_max_aspect_ratio=4.0,
                smallest_input_angle_degrees=60.0)
    check_quality(good)
    for key, value, message in (("horizontal_min_angle_degrees", 10.0, "minimum angle"),
                                ("horizontal_max_aspect_ratio", 9.0, "aspect ratio"),
                                ("wall_max_aspect_ratio", 9.0, "Via wall aspect")):
        with pytest.raises(ValueError, match=message):
            check_quality({**good, key: value})


def test_planar_compile_refusals():
    """Scene-level polygon records refuse what the planar path does not model, before any device work."""
    import torch
    import witwin.maxwell as mw
    from witwin.maxwell.mom.layered.planar import _planar_port, prepare_planar_network, sheet_record, via_record
    planes = (0.0, 3 * UM)
    with pytest.raises(NotImplementedError, match="normal_axis='z'"):
        sheet_record(mw.geometry.PlanarPEC(torch.tensor(rect(0, 1, 0, 1)), normal_axis="x"))
    with pytest.raises(NotImplementedError, match="axis-aligned Box"):
        via_record(mw.Box(position=(0, 0, 1.5 * UM), size=(1 * UM, 1 * UM, 3 * UM), rotation=(0, 0, .3)), planes)
    with pytest.raises(NotImplementedError, match="rotated about z only"):
        via_record(mw.geometry.Prism(position=(0, 0, 1.5 * UM), radius=1 * UM, height=3 * UM, num_sides=8,
                            rotation=(.3, 0, 0)), planes)
    with pytest.raises(NotImplementedError, match="Box, or z-axis Prism or ExtrudedPolygon"):
        via_record(mw.Cylinder(position=(0, 0, 1.5 * UM), radius=1 * UM, height=3 * UM), planes)
    record = via_record(mw.geometry.Prism(position=(0, 0, 1.5 * UM), radius=1 * UM, height=3 * UM, num_sides=6), planes)
    assert (record.z0, record.z1) == planes and len(record.outline) == 6
    sheets = [PolygonSheet((rect(0, 10, 0, 10),), 0.0), PolygonSheet((rect(0, 10, 0, 10),), 3 * UM)]
    vertical = mw.LumpedPort("v", positive=(5 * UM, 5 * UM, 3 * UM), negative=(5 * UM, 5 * UM, 0.0),
                             voltage_path=mw.AxisPath("z"), reference_impedance=50.0)
    with pytest.raises(NotImplementedError, match="horizontal LumpedPorts only"):
        _planar_port(vertical, sheets, 2 * UM)
    with pytest.raises(NotImplementedError, match="conductor_model='sheet' only"):
        prepare_planar_network(None, sheets, [vertical], (1e9,), bridge_width=2 * UM, max_edge=5 * UM,
                               conductor_model="closed")


def test_planar_path_edge_ports_and_port_records():
    """G4 (lifts the round-2 refusal): a voltage_aperture on the polygon path makes an aperture-wide edge bridge
    with the C1 validation rules; port records name the declaring polygon sheets."""
    import witwin.maxwell as mw
    from witwin.maxwell.mom.layered.planar import _planar_port
    from witwin.maxwell.mom.layered.port_system import port_records
    sheets = [PolygonSheet((rect(0, 10, 0, 10),), 0.0), PolygonSheet((rect(14, 24, 0, 10),), 0.0)]
    gap = dict(positive=(16 * UM, 5 * UM, 0.0), negative=(8 * UM, 5 * UM, 0.0), voltage_path=mw.AxisPath("x"),
               reference_impedance=50.0)

    def edge(position, size, **terminals):
        return mw.LumpedPort("e", **{**gap, **terminals}, voltage_aperture=mw.Box(
            position=tuple(v * UM for v in position), size=tuple(v * UM for v in size)))
    span = dict(positive=(14 * UM, 5 * UM, 0.0), negative=(10 * UM, 5 * UM, 0.0))
    geometry, cut, half = _planar_port(edge((12, 5, 0), (4, 10, 0), **span), sheets, 2 * UM)
    assert geometry[0] == (10 * UM, 14 * UM, 0.0, 10 * UM, 0.0) and half == 5 * UM
    assert geometry[3:] == (12 * UM, 5 * UM, 1.0) and cut == (0.0, (12 * UM, 0.0), (12 * UM, 10 * UM))
    with pytest.raises(ValueError, match="interior overlaps metal"):
        _planar_port(edge((12, 5, 0), (8, 10, 0)), sheets, 2 * UM)
    with pytest.raises(ValueError, match="not covered by metal"):
        _planar_port(edge((12, 5, 0), (4, 12, 0), **span), sheets, 2 * UM)
    with pytest.raises(ValueError, match="span exactly the terminal path"):
        _planar_port(edge((12, 5, 0), (6, 10, 0), **span), sheets, 2 * UM)
    with pytest.raises(ValueError, match="zero z extent"):
        _planar_port(edge((12, 5, 0), (4, 10, 1), **span), sheets, 2 * UM)
    port = mw.LumpedPort("p", **gap)
    geometry, _, half = _planar_port(port, sheets, 2 * UM)
    assert half == UM
    stack = mw.mom.LayerStack((mw.mom.Layer("sub", 100 * UM, 4.0),), "pec", mw.mom.HalfSpace())
    (record,) = port_records([port], [geometry], [half], sheets, ["ground", "signal"], stack)
    assert (record["kind"], record["orientation"]) == ("delta_gap", "horizontal")
    assert (record["positive_conductor"], record["reference"]) == ("signal", "ground")
    assert record["width_m"] == 2 * UM and record["length_m"] == pytest.approx(8 * UM)
    import json
    assert json.loads(json.dumps(record))["reference_impedance_ohm"] == [50.0, 0.0]


def test_polygon_path_names_the_rfic_extra_when_shapely_is_missing(monkeypatch):
    """shapely is the optional extra 'rfic': without it the polygon path says how to install it."""
    import sys
    import witwin.maxwell as mw
    from witwin.maxwell.mom.layered.planar import _planar_port
    from witwin.maxwell.mom.layered.port_system import _sheet_covers
    sheets = [PolygonSheet((rect(0, 10, 0, 10),), 0.0), PolygonSheet((rect(14, 24, 0, 10),), 0.0)]
    port = mw.LumpedPort("p", positive=(16 * UM, 5 * UM, 0.0), negative=(8 * UM, 5 * UM, 0.0),
                         voltage_path=mw.AxisPath("x"), reference_impedance=50.0)
    monkeypatch.setitem(sys.modules, "shapely", None)
    hint = r"pip install witwin-maxwell\[rfic\]"
    with pytest.raises(ImportError, match=hint):
        _planar_port(port, sheets, 2 * UM)
    with pytest.raises(ImportError, match=hint):
        _sheet_covers(sheets[0], (5 * UM, 5 * UM, 0.0))
    with pytest.raises(ImportError, match=hint):
        planar_triangulation(sheets[:1], (0.0,), 5 * UM)
