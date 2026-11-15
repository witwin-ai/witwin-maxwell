"""Polygon layouts (non-Manhattan sheets, polygonal and multi-plane vias) for the layered network.

The rectangle path stays the default for Manhattan inputs and is unchanged.
This path compiles each PlanarPEC to one PolygonSheet record, each via (Box, or
z-axis Prism or ExtrudedPolygon) to one PolygonVia record, meshes them with the conforming planar
mesher and reuses the shared Green tables, assembly, loop/tree and solve.
"""

import math

import numpy as np
import torch

from .planar_mesh import PolygonSheet, PolygonVia, mesh_planar_layout
from ...errors import UnsupportedFeatureError


MESHER = "planar_conforming_delaunay_v1"


def is_manhattan(loops):
    return all(a[0] == b[0] or a[1] == b[1] for loop in loops for a, b in zip(loop, loop[1:] + loop[:1]))


def is_planar_record(record):
    return isinstance(record, (PolygonSheet, PolygonVia))


def sheet_record(geometry, coordinate_maps=None):
    """One PolygonSheet per z-normal PlanarPEC, holes kept, optional x/y repair maps applied."""
    if geometry.normal_axis != "z":
        raise UnsupportedFeatureError("Layered MoM supports PlanarPEC sheets with normal_axis='z' only",
                                      feature="geometry.planar_pec", method="mom.layered")
    loops = [loop.detach().cpu().tolist() for loop in (geometry.vertices, *geometry.holes)]
    if coordinate_maps is not None:
        loops = [[[coordinate_maps[a][p[a]] for a in (0, 1)] for p in loop] for loop in loops]
    return PolygonSheet(tuple(tuple(tuple(p) for p in loop) for loop in loops), float(geometry.coordinate))


def via_record(geometry, planes, coordinate_maps=None):
    """PolygonVia of a Box, or of a z-axis Prism/ExtrudedPolygon, whose end faces snap to declared metal planes.

    End heights within 1e-6*(|z|+height) of a plane (float32 Box/Prism roundoff)
    take that plane's exact value; footprint x/y are kept (no breakpoint grid).
    An ExtrudedPolygon footprint is its vertices rotated about z and translated,
    counter-clockwise, in fp64.
    """
    from witwin.core import Box, ExtrudedPolygon, Prism
    position = geometry.position.detach().to(torch.float64).cpu().tolist()
    rotation = geometry._rotation_matrix_np()
    if isinstance(geometry, Box):
        if not np.array_equal(rotation, np.eye(3)):
            raise UnsupportedFeatureError("Layered vias require an axis-aligned Box",
                                          feature="guard:632b1bf4", method="mom.layered")
        size = geometry.size.detach().to(torch.float64).cpu().tolist()
        x0, x1, y0, y1 = (position[0] - size[0] / 2, position[0] + size[0] / 2,
                          position[1] - size[1] / 2, position[1] + size[1] / 2)
        if coordinate_maps is not None:
            x0, x1 = (coordinate_maps[0][v] for v in (x0, x1))
            y0, y1 = (coordinate_maps[1][v] for v in (y0, y1))
        outline = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
        height = size[2]
    elif isinstance(geometry, (Prism, ExtrudedPolygon)):
        if geometry.axis != "z" or not np.allclose(rotation[2], (0, 0, 1), rtol=0, atol=1e-12):
            raise UnsupportedFeatureError("Layered polygonal vias require a z-axis Prism or ExtrudedPolygon "
                                          "rotated about z only",
                                          feature="guard:f809e749", method="mom.layered")
        height = float(geometry.height.detach().to(torch.float64))
        if isinstance(geometry, Prism):
            radius = float(geometry.radius.detach().to(torch.float64))
            angles = 2 * math.pi * np.arange(geometry.num_sides) / geometry.num_sides
            local = np.stack((radius * np.cos(angles), radius * np.sin(angles)), 1)
        else:
            local = geometry.vertices.detach().to(torch.float64).cpu().numpy()
            if np.sum(local[:, 0] * np.roll(local[:, 1], -1) - np.roll(local[:, 0], -1) * local[:, 1]) < 0:
                local = local[::-1]
        outline = tuple(tuple(float(v) for v in rotation[:2, :2] @ p + position[:2]) for p in local)
    else:
        raise UnsupportedFeatureError(f"Layered vias are axis-aligned Box, or z-axis Prism or ExtrudedPolygon; got "
                                      f"{type(geometry).__name__} (declare a circular via as a Prism with num_sides "
                                      "segments)",
                                      feature="guard:9f961005", method="mom.layered")
    ends = []
    for value in (position[2] - height / 2, position[2] + height / 2):
        near = [z for z in planes if abs(z - value) <= 1e-6 * (abs(position[2]) + height)]
        ends.append(near[0] if len(near) == 1 else value)
    return PolygonVia(outline, *ends)


def _material_key(record):
    sigma, thickness, roughness = record
    return (sigma, thickness, None if roughness is None else
            (type(roughness).__name__, tuple(sorted((k, float(v)) for k, v in vars(roughness).items()))))


def _planar_edge_span(port, metal, axis, z, low, high):
    """Lateral (lo, hi) of a polygon-layout edge port: the voltage_aperture width (port_system.horizontal_span rules).

    The aperture spans exactly the terminal path along ``axis`` with zero z extent; its lateral bounds snap to
    the plane's vertex coordinates within the Box rounding scale. Both bridge ends must be covered by metal over
    the full width and the open bridge interior must be metal-free.
    """
    from .planar_mesh import optional_module
    from .port_system import _bounds, _close, _snap
    shapely = optional_module("shapely", "Layered polygon layouts")
    name, lateral = port.name, 1 - axis
    bounds, scale = _bounds(port.voltage_aperture)
    coordinates = shapely.get_coordinates(metal.boundary)[:, lateral].tolist()
    lo, hi = (_snap(v, coordinates, scale[lateral]) for v in bounds[lateral])
    if not (_close(bounds[2][0], z, scale[2]) and _close(bounds[2][1], z, scale[2])):
        raise ValueError(f"Port {name!r} voltage_aperture of a horizontal port must have zero z extent on its plane")
    if not (_close(bounds[axis][0], low, scale[axis]) and _close(bounds[axis][1], high, scale[axis])):
        raise ValueError(f"Port {name!r} voltage_aperture must span exactly the terminal path along "
                         f"{'xy'[axis]} ({low:g}..{high:g})")
    if not lo < float(port.positive[lateral]) < hi:
        raise ValueError(f"Port {name!r} voltage_aperture must contain the terminal path in its width")
    for end in (low, high):
        segment = shapely.LineString(((end, lo), (end, hi)) if axis == 0 else ((lo, end), (hi, end)))
        if not metal.covers(segment):
            raise ValueError(f"Port {name!r} edge bridge end at {'xy'[axis]}={end:g} is not covered by metal "
                             "over the full voltage_aperture width")
    gap = shapely.box(low, lo, high, hi) if axis == 0 else shapely.box(lo, low, hi, high)
    if shapely.intersection(metal, gap).area > 0:
        raise ValueError(f"Port {name!r} edge bridge interior overlaps metal; the aperture must span an open gap")
    return lo, hi


def _planar_port(port, sheets, bridge_width):
    """Horizontal bridge on polygon metal: ((bridge, z, axis, cut, centre, direction), cut line, half width).

    The bridge is bridge_width wide and centred on the terminal path (delta gap), or spans the
    voltage_aperture width (edge port, _planar_edge_span).
    """
    from .planar_mesh import optional_module
    shapely = optional_module("shapely", "Layered polygon layouts")
    if port.voltage_path is None or port.voltage_path.axis == "z":
        raise UnsupportedFeatureError(f"Port {port.name!r}: polygon layouts support horizontal LumpedPorts only; "
                                  "vertical bridge ports need Manhattan sheets",
                                      feature="port.LumpedPort", method="mom.layered")
    axis = "xy".index(port.voltage_path.axis)
    positive, negative = (tuple(float(v) for v in point) for point in (port.positive, port.negative))
    z = positive[2]
    metal = shapely.union_all([shapely.Polygon(s.loops[0], s.loops[1:]) for s in sheets if s.z == z])
    if metal.is_empty or not all(shapely.intersects_xy(metal, p[0], p[1]) for p in (positive, negative)):
        raise ValueError(f"Port {port.name!r} terminals must lie on metal of their plane")
    low, high = sorted((positive[axis], negative[axis]))
    cut = 0.5 * (low + high)
    if port.voltage_aperture is None:
        centre, half = positive[1 - axis], 0.5 * bridge_width
    else:
        lo, hi = _planar_edge_span(port, metal, axis, z, low, high)
        centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    ends = ((cut, centre - half), (cut, centre + half)) if axis == 0 else ((centre - half, cut), (centre + half, cut))
    if port.voltage_aperture is None and shapely.distance(metal, shapely.LineString(ends)) <= 0:
        raise ValueError(f"Port {port.name!r} must bridge a gap: its delta-gap cut touches metal")
    bridge = ((low, high, centre - half, centre + half, z) if axis == 0 else
              (centre - half, centre + half, low, high, z))
    direction = 1.0 if positive[axis] > negative[axis] else -1.0
    return (bridge, z, axis, cut, centre, direction), (z, *ends), half


def prepare_planar_loss(layout, materials, via_materials, frequencies):
    """Per-face owner impedance (two-face sheets, single-face bulk via walls) and local Gram rows."""
    from ..topology import INTERIOR
    from .loss import MU0, SheetLoss, sheet_impedance
    from .mesh import JunctionRWGBasis
    records = [*materials, *via_materials]
    if not any(sigma is not None for sigma, _, _ in records):
        return None
    mesh, basis = layout.mesh, layout.basis
    impedance = torch.zeros((len(frequencies), len(mesh.faces)), device=mesh.device, dtype=torch.complex128)
    owners = torch.full((len(mesh.faces),), -1, device=mesh.device, dtype=torch.int64)
    for record, (sigma, thickness, roughness) in enumerate(records):
        if sigma is None:
            continue
        owned = layout.face_owners == record
        if record < len(materials):
            value = sheet_impedance(frequencies, sigma, thickness, roughness)
        else:
            # A tube wall is one bulk-metal face, unlike a two-face thin sheet.
            delta = torch.sqrt(2 / (2 * math.pi * frequencies * MU0 * sigma))
            value = (1 - 1j) / (sigma * delta)
            if roughness is not None:
                value = value * roughness.impedance_factor(delta)
        impedance[:, owned] = value[:, None]
        owners[owned] = record
    supports = torch.stack((basis.plus_faces, basis.minus_faces), -1)
    if isinstance(basis, JunctionRWGBasis):
        blocks = basis.mass_blocks()[supports, basis.support_positions]
        return SheetLoss(blocks, basis.face_edges[supports], supports, impedance, owners)
    slots = mesh.topology.incidence_slots[mesh.topology.kind == INTERIOR]
    blocks = basis.mass_blocks()[supports, slots]
    return SheetLoss(blocks, basis.face_edges[supports], supports, impedance, owners)


def prepare_planar_network(stack, sheets, ports, frequencies, *, bridge_width, max_edge, edge_cell=None,
                           growth_ratio=1.5, sheet_materials=None, vias=(), via_materials=None, device="cuda",
                           prepared_geometry=None, conductor_model="sheet", conductor_ids=None,
                           operator_backend="dense", rectangle_names=None, stream_graded=False):
    """prepare_layered_network for PolygonSheet/PolygonVia records (same keywords and result).

    stream_graded (R4-2): None when several frequencies need a graded vertical table.
    """
    from .green import LayeredGreenTable
    from .vertical_refine import grading_metadata, mesh_interfaces
    from .lowfreq import MIN_FREQUENCY, build_loop_tree
    from .loss import sheet_material_payload
    from .network import (PreparedLayeredNetwork, _cut_vectors, _wall_nodes, available_work_memory,
                          frequency_chunk_capacity)
    from .port_system import port_records
    if conductor_model != "sheet":
        raise UnsupportedFeatureError("Polygon layouts (non-Manhattan sheets, polygonal or multi-plane vias) support "
                                  "conductor_model='sheet' only",
                                      feature="guard:a8687dcd", method="mom.layered")
    if bridge_width <= 0:
        raise ValueError("bridge_width must be positive")
    if not ports:
        raise ValueError("A layered network needs at least one LumpedPort")
    if not all(isinstance(s, PolygonSheet) for s in sheets) or not all(isinstance(v, PolygonVia) for v in vias):
        raise TypeError("Polygon layouts take PolygonSheet and PolygonVia records only")
    materials = [(None, 0.0, None)] * len(sheets) if sheet_materials is None else list(sheet_materials)
    via_materials = [(None, 0.0, None)] * len(vias) if via_materials is None else list(via_materials)
    geometry, cuts, halves = zip(*(_planar_port(port, sheets, bridge_width) for port in ports))
    bridges = [g[0] for g in geometry]
    planes = tuple(sorted({s.z for s in sheets}))
    xy = np.array([p for s in sheets for loop in s.loops for p in loop] +
                  [p for b in bridges for p in ((b[0], b[2]), (b[1], b[3]))])
    rho_max = 1.01 * math.hypot(*(xy.max(0) - xy.min(0)))
    frequencies = tuple(float(f) for f in frequencies)
    if any(not math.isfinite(f) or f < MIN_FREQUENCY for f in frequencies):
        raise ValueError(f"Layered networks require frequencies >= {MIN_FREQUENCY:g} Hz; DC is a limit")
    # Every planar edge and wall-cell diagonal is <= sqrt(2)*max_edge, so the
    # near distance (near_ratio=2 times the largest edge) is inside this extraction.
    thinnest = min((layer.thickness for layer in stack.layers), default=0.0)
    extract = max(2 * math.sqrt(2) * max_edge, thinnest)
    spans = ((min(planes), max(planes)),) if vias else ()
    table = LayeredGreenTable.build(stack, planes, frequencies, rho_max, device=device, via_spans=spans,
                                    z_nodes=8, extract_distance=extract, stream_graded=stream_graded)
    if table is None:
        return None
    if prepared_geometry is not None:
        layout = prepared_geometry.layout
    else:
        layout = mesh_planar_layout(sheets, table.planes, max_edge, groups=[_material_key(m) for m in materials],
                                    vias=vias, bridges=bridges, cuts=cuts, h_edge=edge_cell,
                                    growth_ratio=growth_ratio, interfaces=mesh_interfaces(stack, table))
    loss = prepare_planar_loss(layout, materials, via_materials, table.frequencies)
    unknowns = layout.basis.edges.shape[0]
    if operator_backend == "dense":
        frequency_chunk_capacity(unknowns, len(layout.mesh.faces), len(ports), len(frequencies),
                                 available_work_memory(layout.mesh.device), wall_nodes=_wall_nodes(table))
    metadata = {"solver": f"layered_mpie_{operator_backend}", "operator_backend": operator_backend,
                "bridge_width": bridge_width, "max_edge": max_edge, "edge_cell": edge_cell,
                "unknowns": unknowns, "faces": layout.mesh.faces.shape[0], "rho_max": rho_max,
                "extract_distance": extract, "mesh_quality": layout.quality,
                "sheets": tuple(sheet_material_payload(m) for m in materials), "port_bridges": "ideal_lossless",
                "conductor_model": conductor_model, "thickness_modelled_geometrically": False,
                "mesher": MESHER, "mesh_report": dict(layout.mesh_report),
                "mesh_sha256": layout.mesh_report["mesh_sha256"],
                "ports": port_records(ports, geometry, halves, sheets,
                                      [None] * len(sheets) if rectangle_names is None else rectangle_names, stack)}
    metadata.update(grading_metadata(table))
    if vias:
        metadata.update(vias=tuple(vias), via_impedance="single_face_Leontovich_bulk", vertical_ports=0,
                        ground_half_rwg=False, junction_edges=layout.mesh_report["junction_edges"])
    loop_tree = (build_loop_tree(layout, length_scale=max_edge, dense_setup=operator_backend == "dense")
                 if prepared_geometry is None else prepared_geometry.loop_tree)
    metadata.update(low_frequency_method="loop_tree_scaled", loop_unknowns=loop_tree.loops,
                    tree_unknowns=loop_tree.trees, components=loop_tree.components,
                    minimum_frequency_Hz=MIN_FREQUENCY, dc="positive_frequency_limit")
    excitation = (_cut_vectors(layout, table, geometry, halves) if prepared_geometry is None
                  else prepared_geometry.excitation)
    return PreparedLayeredNetwork(table, layout, excitation, tuple(ports), loss, loop_tree, metadata)
