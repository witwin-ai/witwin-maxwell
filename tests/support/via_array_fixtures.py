"""R7-1 / B2 via-array fixture: a via cluster inside a via fence between two PEC pads (coax-like).

Bottom plane z=0: a solid square pad. Top plane z=h: an inner square pad over the cluster and an outer square ring
over the fence, separated by an open gap; a delta-gap LumpedPort along x bridges the gap at y=0. Between the planes
the via field is two-dimensional, so the height difference L(h2)-L(h1) = (h2-h1) L' de-embeds the pads and the port.
"""

import math

import torch

import witwin.maxwell as mw
from tests.support.rfic_planar_fixtures import UM, gap_port, polygon_structure, vacuum
from tests.support.via_array_reference import regular_outline

INNER, GAP, OUTER = 15.0, 10.0, 70.0          # um: inner pad half-side, gap, outer half-side
FENCE_RADIUS, FENCE_COUNT = 45.0, 8
VIA_RADIUS, VIA_SIDES = 4.0, 8
PITCH = 12.0


def cluster_centres(kind):
    if kind == "array":
        return [(sx * PITCH / 2, sy * PITCH / 2) for sx in (-1, 1) for sy in (-1, 1)]
    return [(0.0, 0.0)]


def fence_centres():
    return [(FENCE_RADIUS * math.cos(2 * math.pi * k / FENCE_COUNT + math.pi / FENCE_COUNT),
             FENCE_RADIUS * math.sin(2 * math.pi * k / FENCE_COUNT + math.pi / FENCE_COUNT)) for k in range(FENCE_COUNT)]


def via_outlines(kind):
    """((outline um, net sign), ...): cluster vias +1, fence vias -1 (regular octagons, or a bar for kind='bar')."""
    if kind == "bar":
        bar = ((-10.0, -3.0), (10.0, -3.0), (10.0, 3.0), (3.0, 3.0), (3.0, 9.0), (-3.0, 9.0), (-3.0, 3.0), (-10.0, 3.0))
        cluster = [bar]
    else:
        cluster = [regular_outline(c, VIA_RADIUS, VIA_SIDES, math.pi / VIA_SIDES) for c in cluster_centres(kind)]
    fence = [regular_outline(c, VIA_RADIUS, VIA_SIDES, math.pi / VIA_SIDES) for c in fence_centres()]
    return [(o, 1) for o in cluster] + [(o, -1) for o in fence]


def coax_via_scene(height, kind="array", material=None):
    """Scene (PEC pads) with ExtrudedPolygon vias of ``material`` (PEC when None) and its port."""
    from witwin.core import ExtrudedPolygon
    shapely = __import__("shapely")
    pec = mw.Material.pec()
    material = pec if material is None else material
    ring = shapely.box(-OUTER, -OUTER, OUTER, OUTER).difference(
        shapely.box(-INNER - GAP, -INNER - GAP, INNER + GAP, INNER + GAP))
    scene = mw.Scene(background=vacuum(), structures=(
                polygon_structure(shapely.box(-OUTER, -OUTER, OUTER, OUTER), 0.0, pec),
                polygon_structure(shapely.box(-INNER, -INNER, INNER, INNER), height, pec),
                polygon_structure(ring, height, pec),
            ))
    for outline, _ in via_outlines(kind):
        vertices = torch.tensor(outline, dtype=torch.float64) * UM
        scene = scene.updated(structures=(
            *scene.structures,
            mw.Structure(ExtrudedPolygon(
                vertices, height=torch.tensor(height * UM, dtype=torch.float64),
                position=torch.tensor((0.0, 0.0, height * UM / 2), dtype=torch.float64), axis="z"), material),
        ))
    scene = scene.updated(ports=(*scene.ports, gap_port("p", (INNER, 0.0, height), (INNER + GAP, 0.0, height), "x")))
    return scene


def coax_config(edge=4.0, max_edge=20.0, bridge=4.0):
    return mw.mom.MoMConfig(layered=mw.mom.LayeredSurfaceConfig(
        bridge_width_m=bridge * UM, max_edge_length_m=max_edge * UM, edge_cell_m=edge * UM))


def via_wall_currents(prepared, currents, frequency_index, height, outlines, tolerance=1e-9 * UM):
    """Per via: signed vertical current through every interior horizontal wall-edge level, over the terminal current.

    Returns one tensor of relative currents per via (wall z levels differ between outlines).
    """
    shapely = __import__("shapely")
    layout = prepared.solver.prepared.layout
    basis, mesh = layout.basis, layout.mesh
    a, b = mesh.vertices[basis.edges[:, 0]], mesh.vertices[basis.edges[:, 1]]
    flow = mesh.centroids[basis.minus_faces] - mesh.centroids[basis.plus_faces]
    terminal = (prepared.solver.prepared.excitation[:, 0] * currents[frequency_index, :, 0]).sum()
    horizontal = (a[:, 2] == b[:, 2]) & (a[:, 2] > 0) & (a[:, 2] < height * UM)
    ah, bh = a[:, :2].cpu().numpy(), b[:, :2].cpu().numpy()
    columns = []
    for outline in outlines:
        near = shapely.buffer(shapely.Polygon([(x * UM, y * UM) for x, y in outline]).boundary, tolerance)
        inside = torch.from_numpy(shapely.covers(near, shapely.points(ah)) & shapely.covers(near, shapely.points(bh)))
        tube = horizontal & inside.to(a.device)
        levels = a[tube, 2].unique()
        columns.append(torch.stack([(torch.sign(flow[tube & (a[:, 2] == z), 2]) * basis.edge_lengths[tube & (a[:, 2] == z)]
                                     * currents[frequency_index, tube & (a[:, 2] == z), 0]).sum() / terminal
                                    for z in levels]))
    return columns


def measure_coax(height, kind, edge, frequencies, sigma=5.8e7):
    """Run one coax via scene with lossy vias; Z11 per frequency and the per-via relative wall currents."""
    scene = coax_via_scene(height, kind, mw.media.LossyMetalMedium(conductivity=sigma))
    prepared = mw.Simulation.mom(scene, frequencies=tuple(frequencies), config=coax_config(edge)).prepare()
    result = prepared.run()
    currents = mw.mom.surface_currents(result).currents
    outlines = [o for o, _ in via_outlines(kind)]
    walls = [via_wall_currents(prepared, currents, k, height, outlines) for k in range(len(frequencies))]
    return dict(z=result.network.to_z()[:, 0, 0].cpu(), walls=[[t.cpu() for t in w] for w in walls],
                unknowns=result.network.metadata["unknowns"], faces=result.network.metadata["faces"],
                s=result.network.s.cpu(), currents=currents.cpu())
