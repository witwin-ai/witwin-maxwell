"""Deterministic conforming triangle meshes of arbitrary planar polygons and via walls.

Host topology preprocessing, done once per layout (B1). Every plane is meshed by
conforming Delaunay refinement (Ruppert): constraint segments (polygon and hole
boundaries, material and bridge boundaries, port cuts, via rims) are split until
each is a Delaunay edge with an empty diametral circle, then triangles that are
too large for the graded size field or have a small angle get their circumcentre
inserted. All constraint segments live in ONE registry with a plane mask, so a
via rim split on one plane is split on every plane it touches and the vertical
walls weld to identical rim vertices. Only fp64 NumPy, the Qhull Delaunay
triangulation and GEOS predicates are used, in a fixed order, so the mesh is a
deterministic function of the input; its sha256 is recorded.

Size field: s(x) = min(h, h_edge + (growth_ratio - 1) d(x)), d the distance to
the plane's constraint linework (isotropic grading from every edge, cut and rim).
Constraint segments are <= h_edge; triangle edges are <= sqrt(2) s, the diagonal
of the rectangle mesher's s x s cell.
"""

from dataclasses import dataclass
import hashlib
import importlib
import math

import numpy as np
import torch

from ...errors import UnsupportedFeatureError


# Refinement goal and quality gates. The gates sit inside the region where the
# near-field closed-form/outer-rule integration is validated against an
# independent quadrature (tests/mom/layered/test_layered_planar_assembly.py).
TARGET_ANGLE_DEGREES = 28.0
GATE_ANGLE_DEGREES = 18.0
GATE_ASPECT = 6.0
WALL_GATE_ASPECT = GATE_ASPECT
MAX_ITERATIONS = 400
# Longest edge <= sqrt(2) * s, the diagonal of the rectangle mesher's s x s cell.
# A strict edge <= s cascades when the strip width is not a multiple of s.
SIZE_TOLERANCE = math.sqrt(2)

RFIC_INSTALL = "pip install witwin-maxwell[rfic]"


def optional_module(name, purpose):
    """Import an optional dependency of the layered RFIC path (extra ``rfic``) with an install hint."""
    try:
        return importlib.import_module(name)
    except ImportError:
        raise ImportError(f"{purpose} requires the optional dependency {name}; "
                          f"install it with: {RFIC_INSTALL}") from None


@dataclass(frozen=True)
class PolygonSheet:
    """Planar sheet record: loops[0] is the outer loop, loops[1:] are holes, all ((x, y), ...)."""

    loops: tuple
    z: float


@dataclass(frozen=True)
class PolygonVia:
    """Vertical tube record: a simple footprint loop ((x, y), ...) from plane z0 to plane z1."""

    outline: tuple
    z0: float
    z1: float


@dataclass
class PlanarTriangulation:
    """Host result: welded vertices, faces, plane id (-1 walls) and owner (-1 ideal) per face."""

    vertices: np.ndarray
    faces: np.ndarray
    planes: np.ndarray
    owners: np.ndarray
    report: dict


def _segment_lengths(a, b):
    return np.hypot(*(b - a).T)


def _polygon(loops):
    shapely = optional_module("shapely", "Layered polygon layouts")
    polygon = shapely.Polygon(loops[0], loops[1:])
    if not polygon.is_valid:
        raise ValueError("Planar layered polygons must be simple, with disjoint holes strictly inside")
    return polygon


def _lines(geometry):
    """Consecutive vertex pairs of every linear part of a (multi)line geometry."""
    shapely = optional_module("shapely", "Layered polygon layouts")
    parts = shapely.get_parts(geometry)
    pairs = []
    for part in parts:
        coordinates = shapely.get_coordinates(part)
        pairs.extend(zip(coordinates[:-1], coordinates[1:]))
    return pairs


def _axis(edges, h, h_edge, growth_ratio):
    from .mesh import _axis_points
    return _axis_points(torch.tensor(sorted(set(edges)), dtype=torch.float64), h, h_edge, growth_ratio).tolist()


class _Plane:
    """Domain, constraint linework and Steiner points of one metal plane."""

    def __init__(self, z, domain, sheets, bridges):
        shapely = optional_module("shapely", "Layered polygon layouts")
        self.z, self.domain, self.sheets, self.bridges = z, domain, sheets, bridges
        shapely.prepare(domain)
        self.steiner = np.zeros((0, 2))
        self.acute = {}


def planar_triangulation(sheets, planes, h, *, groups=None, vias=(), bridges=(), cuts=(), h_edge=None,
                         growth_ratio=1.5, interfaces=()):
    """Mesh PolygonSheet/PolygonVia records, ideal (x0,x1,y0,y1,z) bridges and (z,(x,y),(x,y)) cuts.

    ``groups`` gives a hashable material key per sheet: overlapping sheets must
    share it, and boundaries between keys become constraints. Owners index the
    sheets, then the vias (len(sheets) + i); bridge-only faces are ideal (-1).
    """
    shapely = optional_module("shapely", "Layered polygon layouts")
    from scipy.spatial import cKDTree

    if h <= 0 or (h_edge is not None and not 0 < h_edge <= h) or not 1 <= growth_ratio <= 1.5:
        raise ValueError("Require h>0, 0<h_edge<=h and 1<=growth_ratio<=1.5")
    planes = [float(z) for z in planes]
    groups = list(range(len(sheets))) if groups is None else list(groups)
    small = h if h_edge is None else h_edge
    floor = small / 64
    polygons = [_polygon(sheet.loops) for sheet in sheets]
    footprints = [_polygon((via.outline,)) for via in vias]
    for sheet in sheets:
        if sheet.z not in planes:
            raise ValueError("Polygon sheet is off every metal plane")

    # Via rims per plane: endpoints always, intermediate planes only when metal lands on the tube.
    rims = {}
    wall_spans = []
    for v, (via, footprint) in enumerate(zip(vias, footprints)):
        if not via.z0 < via.z1 or via.z0 not in planes or via.z1 not in planes:
            raise ValueError("Via endpoints must coincide with declared metal planes, z0 < z1")
        span = [z for z in planes if via.z0 <= z <= via.z1]
        for z in span:
            metal = shapely.union_all([p for p, s in zip(polygons, sheets) if s.z == z])
            ideal = shapely.union_all([shapely.box(b[0], b[2], b[1], b[3]) for b in bridges if b[4] == z])
            if not ideal.is_empty and shapely.distance(footprint, ideal) <= 0:
                raise UnsupportedFeatureError("Via footprints must not touch port bridges",
                                              feature="guard:f2afcae4", method="mom.layered")
            inside = (not metal.is_empty and metal.contains(footprint)
                      and shapely.distance(footprint.boundary, metal.boundary) > 0)
            if z in (via.z0, via.z1):
                if not inside:
                    raise ValueError("Via footprint must be inside metal with its rim surrounded by metal "
                                     "on both end planes")
            elif not inside:
                if not metal.is_empty and shapely.distance(footprint, metal) <= 0:
                    raise UnsupportedFeatureError("A via crossing an intermediate plane must clear its metal or "
                                              "land with its whole rim inside that metal",
                                                  feature="guard:26daaab6", method="mom.layered")
                continue
            rims.setdefault(z, []).append(v)
        wall_spans.append(span)
    for i in range(len(vias)):
        for j in range(i):
            a, b = vias[i], vias[j]
            if max(a.z0, b.z0) >= min(a.z1, b.z1):
                if max(a.z0, b.z0) == min(a.z1, b.z1) and shapely.distance(footprints[i], footprints[j]) <= 0 \
                        and a.outline != b.outline:
                    raise UnsupportedFeatureError("Stacked vias meeting on a plane need identical footprints",
                                                  feature="guard:f4a34121", method="mom.layered")
                continue
            if shapely.distance(footprints[i], footprints[j]) <= 0:
                raise UnsupportedFeatureError("Touching or overlapping vias require junction geometry",
                                              feature="guard:b1485243", method="mom.layered")

    # Constraint registry: segment endpoints, plane mask and rim key (-1 for plane linework).
    rim_keys = {}
    seg_a, seg_b, masks, rim_of = [], [], [], []
    plane_data = []
    for p, z in enumerate(planes):
        members = [i for i, s in enumerate(sheets) if s.z == z]
        ideal = [shapely.box(b[0], b[2], b[1], b[3]) for b in bridges if b[4] == z]
        for i in members:
            for j in members:
                if j < i and groups[i] != groups[j] and shapely.intersection(polygons[i], polygons[j]).area > 0:
                    raise ValueError("Overlapping sheets on one plane require identical materials and thicknesses")
        holes = [footprints[v] for v in rims.get(z, ())]
        union = shapely.union_all([polygons[i] for i in members] + ideal)
        domain = shapely.difference(union, shapely.union_all(holes)) if holes else union
        plane = _Plane(z, domain, members, ideal)
        plane_data.append(plane)
        if domain.is_empty:
            continue
        keys = list(dict.fromkeys(groups[i] for i in members))
        linework = [shapely.union_all([polygons[i] for i in members if groups[i] == key]).boundary
                    for key in keys]
        linework += [b.boundary for b in ideal]
        linework += [shapely.LineString(c[1:]) for c in cuts if c[0] == z]
        noded = shapely.union_all(linework)
        for v in rims.get(z, ()):
            if shapely.distance(footprints[v].boundary, noded) <= 0:
                raise UnsupportedFeatureError("Via rims must not touch sheet, material, bridge or cut boundaries",
                                              feature="guard:7ae09843", method="mom.layered")
        for a, b in _lines(noded):
            # A boundary midpoint can round to either side of its own edge.
            middle = shapely.Point((a + b) / 2)
            if shapely.distance(domain, middle) > 1e-6 * floor:
                continue
            seg_a.append(a), seg_b.append(b), rim_of.append(-1)
            masks.append(np.arange(len(planes)) == p)
        for v in rims.get(z, ()):
            key = vias[v].outline
            if key in rim_keys:
                for k in rim_keys[key]:
                    masks[k][p] = True
                continue
            loop = np.asarray(vias[v].outline, dtype=np.float64)
            rim_keys[key] = []
            for a, b in zip(loop, np.roll(loop, -1, 0)):
                rim_keys[key].append(len(seg_a))
                seg_a.append(a), seg_b.append(b), rim_of.append(v)
                masks.append(np.arange(len(planes)) == p)
    if not seg_a:
        raise ValueError("Planar layout has no metal")
    seg_a, seg_b = np.array(seg_a, dtype=np.float64), np.array(seg_b, dtype=np.float64)
    masks, rim_of = np.array(masks), np.array(rim_of, dtype=np.int64)
    lengths = _segment_lengths(seg_a, seg_b)
    if lengths.min() < floor:
        raise ValueError(f"A layout segment of {lengths.min():.3g} m is below the mesh floor {floor:.3g} m; "
                         "repair near-coincident vertices with breakpoint_tolerance_m")

    # Wall z axes; rim segments are short enough for wall cells of bounded aspect.
    wall_axes, wall_edges = [], []
    for via, span in zip(vias, wall_spans):
        wall_edges.append(span + [z for z in interfaces if via.z0 < z < via.z1])
        wall_axes.append(_axis(wall_edges[-1], h, h_edge, growth_ratio))
    rim_target = np.full(len(seg_a), small)
    for k in np.nonzero(rim_of >= 0)[0]:
        rim_target[k] = min(small, 2 * max(np.diff(wall_axes[rim_of[k]])))

    # Input vertices: acute wedges (< 60 degrees inside the domain) are exempt from
    # angle refinement and are split with concentric shells to stop ping-pong splitting.
    for p, plane in enumerate(plane_data):
        if plane.domain.is_empty:
            continue
        ends = np.concatenate((seg_a[masks[:, p]], seg_b[masks[:, p]]))
        other = np.concatenate((seg_b[masks[:, p]], seg_a[masks[:, p]]))
        order = np.lexsort((other[:, 1], other[:, 0], ends[:, 1], ends[:, 0]))
        ends, other = ends[order], other[order]
        start = 0
        while start < len(ends):
            stop = start
            while stop < len(ends) and (ends[stop] == ends[start]).all():
                stop += 1
            directions = other[start:stop] - ends[start]
            angles = np.sort(np.arctan2(directions[:, 1], directions[:, 0]))
            if len(angles) >= 2:
                gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
                reach = 1e-3 * np.hypot(*directions.T).min()
                wedge = math.pi
                for angle, gap in zip(angles, gaps):
                    middle = angle + gap / 2
                    probe = ends[start] + reach * np.array((math.cos(middle), math.sin(middle)))
                    if gap < math.pi / 3 and shapely.contains_xy(plane.domain, probe[0], probe[1]):
                        wedge = min(wedge, gap)
                if wedge < math.pi / 3:
                    plane.acute[tuple(ends[start])] = wedge
            start = stop
    acute_points = {key for plane in plane_data for key in plane.acute}
    smallest_input = min((w for plane in plane_data for w in plane.acute.values()), default=math.pi / 3)
    if math.degrees(smallest_input) < GATE_ANGLE_DEGREES:
        raise UnsupportedFeatureError(f"An input corner of {math.degrees(smallest_input):.3g} degrees is below the "
                                  f"{GATE_ANGLE_DEGREES:g}-degree mesh quality gate",
                                      feature="guard:5ecc30dd", method="mom.layered")

    def split(a, b, pieces):
        steps = np.arange(1, pieces)[:, None] / pieces
        return np.concatenate((a[None], a + (b - a) * steps, b[None]))

    # Initial uniform discretization at the edge size.
    na, nb, nm, nr = [], [], [], []
    for k in range(len(seg_a)):
        points = split(seg_a[k], seg_b[k], max(1, math.ceil(lengths[k] / rim_target[k] - 1e-9)))
        na.extend(points[:-1]), nb.extend(points[1:])
        nm.extend([masks[k]] * (len(points) - 1)), nr.extend([rim_of[k]] * (len(points) - 1))
    seg_a, seg_b, masks, rim_of = np.array(na), np.array(nb), np.array(nm), np.array(nr, dtype=np.int64)

    def split_point(a, b):
        ka, kb = tuple(a) in acute_points, tuple(b) in acute_points
        if ka == kb:
            return (a + b) / 2
        apex, other = (a, b) if ka else (b, a)
        length = math.hypot(*(other - apex))
        distance = floor * 2.0 ** round(math.log2(length / 2 / floor))
        return apex + (other - apex) * (distance / length)

    def triangulate(points):
        from scipy.spatial import Delaunay
        triangulation = Delaunay(points - points.min(0))
        if len(triangulation.coplanar):
            raise RuntimeError("Delaunay triangulation dropped coincident points")
        return triangulation.simplices.astype(np.int64)

    def plane_state(p):
        selected = np.nonzero(masks[:, p])[0]
        n = len(selected)
        stack = np.concatenate((seg_a[selected], seg_b[selected], plane_data[p].steiner))
        points, inverse = np.unique(stack, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        return selected, points, inverse[:n], inverse[n:2 * n], n

    iterations = 0
    while True:
        iterations += 1
        if iterations > MAX_ITERATIONS:
            raise RuntimeError("Planar mesh refinement did not converge")
        splits, changed = set(), False
        for p, plane in enumerate(plane_data):
            if plane.domain.is_empty:
                continue
            selected, points, ia, ib, n = plane_state(p)
            simplices = triangulate(points)
            count = len(points)
            pairs = np.sort(np.concatenate((simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [2, 0]])), 1)
            edges = np.unique(pairs[:, 0] * count + pairs[:, 1])
            low, high = np.minimum(ia, ib), np.maximum(ia, ib)
            missing = ~np.isin(low * count + high, edges)
            middle = (points[ia] + points[ib]) / 2
            half = _segment_lengths(points[ia], points[ib]) / 2
            hits = cKDTree(points).query_ball_point(middle, half * (1 - 1e-9))
            encroached = np.array([any(j != ia[k] and j != ib[k] for j in hits[k]) for k in range(n)])
            bad = missing | encroached
            if bad.any():
                if (2 * half[bad]).min() < floor:
                    raise ValueError("Conforming refinement reached the mesh floor: a layout gap or feature is "
                                     f"narrower than {floor:.3g} m")
                splits.update(selected[bad].tolist())
                continue
            simplices = _inside(points, simplices, plane.domain)
            corners = points[simplices]
            edge = corners[:, [1, 2, 0]] - corners
            length = np.hypot(edge[..., 0], edge[..., 1])
            # Angle at vertex i lies between edges i and i-1.
            cosine = -(edge * edge[:, [2, 0, 1]]).sum(-1) / (length * length[:, [2, 0, 1]])
            angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
            apex = angle.argmin(1)
            apex_points = corners[np.arange(len(corners)), apex]
            exempt = np.array([tuple(q) in plane.acute for q in apex_points], dtype=bool)
            boundary = np.concatenate((seg_a[selected], seg_b[selected]))
            distance = cKDTree(boundary).query(corners.mean(1))[0]
            size = np.minimum(h, small + (growth_ratio - 1) * distance) if h_edge is not None else np.full(len(corners), h)
            longest = length.max(1)
            refine = (((angle.min(1) < TARGET_ANGLE_DEGREES) & ~exempt) | (longest > SIZE_TOLERANCE * size)) & (longest > floor)
            if not refine.any():
                continue
            a, b, c = corners[refine, 0], corners[refine, 1], corners[refine, 2]
            d = 2 * ((a[:, 0] - c[:, 0]) * (b[:, 1] - c[:, 1]) - (b[:, 0] - c[:, 0]) * (a[:, 1] - c[:, 1]))
            ac, bc = ((a - c) ** 2).sum(1), ((b - c) ** 2).sum(1)
            centre = c + np.stack((ac * (b[:, 1] - c[:, 1]) - bc * (a[:, 1] - c[:, 1]),
                                   bc * (a[:, 0] - c[:, 0]) - ac * (b[:, 0] - c[:, 0])), 1) / d[:, None]
            radius = np.hypot(*(centre - c).T)
            priority = np.argsort(-radius / size[refine], kind="stable")
            centre, radius = centre[priority], radius[priority]
            segment_tree = cKDTree(middle)
            reach = half.max()
            near = segment_tree.query_ball_point(centre, reach * (1 + 1e-12))
            candidate_tree = cKDTree(centre)
            neighbours = candidate_tree.query_ball_point(centre, radius / 2)
            accepted = np.zeros(len(centre), dtype=bool)
            for i in range(len(centre)):
                hit = [k for k in near[i] if math.hypot(*(centre[i] - middle[k])) < half[k] * (1 - 1e-9)]
                if hit:
                    long = [k for k in hit if 2 * half[k] >= floor]
                    splits.update(selected[long].tolist())
                    continue
                if any(accepted[j] for j in neighbours[i] if j < i):
                    continue
                if not shapely.contains_xy(plane.domain, centre[i, 0], centre[i, 1]):
                    continue
                accepted[i] = True
            if accepted.any():
                plane.steiner = np.concatenate((plane.steiner, centre[accepted]))
                changed = True
        if splits:
            changed = True
            na, nb, nm, nr = [], [], [], []
            for k in range(len(seg_a)):
                if k in splits:
                    point = split_point(seg_a[k], seg_b[k])
                    na.extend((seg_a[k], point)), nb.extend((point, seg_b[k]))
                    nm.extend((masks[k], masks[k])), nr.extend((rim_of[k], rim_of[k]))
                else:
                    na.append(seg_a[k]), nb.append(seg_b[k]), nm.append(masks[k]), nr.append(rim_of[k])
            seg_a, seg_b, masks, rim_of = np.array(na), np.array(nb), np.array(nm), np.array(nr, dtype=np.int64)
        if not changed:
            break

    # Final per-plane triangles inside the domain, oriented +z, and their owners.
    vertices, faces, plane_ids, owners = [], [], [], []
    offset = 0
    for p, plane in enumerate(plane_data):
        if plane.domain.is_empty:
            continue
        _, points, _, _, _ = plane_state(p)
        simplices = _inside(points, triangulate(points), plane.domain)
        corners = points[simplices]
        cross = ((corners[:, 1, 0] - corners[:, 0, 0]) * (corners[:, 2, 1] - corners[:, 0, 1])
                 - (corners[:, 1, 1] - corners[:, 0, 1]) * (corners[:, 2, 0] - corners[:, 0, 0]))
        simplices = np.where((cross < 0)[:, None], simplices[:, [0, 2, 1]], simplices)
        centres = points[simplices].mean(1)
        owner = np.full(len(simplices), -2, dtype=np.int64)
        for i in plane.sheets:
            free = owner == -2
            owner[free & shapely.contains_xy(polygons[i], centres[:, 0], centres[:, 1])] = i
        if plane.bridges:
            bridge = shapely.union_all(plane.bridges)
            owner[(owner == -2) & shapely.intersects_xy(bridge, centres[:, 0], centres[:, 1])] = -1
        if (owner == -2).any():
            raise RuntimeError("A planar triangle has no owning sheet or bridge")
        covered = np.abs(cross).sum() / 2
        if abs(covered - plane.domain.area) > 1e-9 * plane.domain.area:
            raise RuntimeError("Planar triangles do not tile the metal domain")
        vertices.append(np.column_stack((points, np.full(len(points), plane.z))))
        faces.append(simplices + offset)
        plane_ids.append(np.full(len(simplices), p, dtype=np.int64))
        owners.append(owner)
        offset += len(points)
    # Vertical walls: every rim chain segment times every wall z interval.
    walls = 0
    for v, via in enumerate(vias):
        rim = np.nonzero(rim_of == v)[0]
        if not len(rim):
            # A stacked via with an identical footprint shares the first via's rim chain.
            rim = np.nonzero(rim_of == next(u for u in range(v) if vias[u].outline == via.outline))[0]
        # Wall cells stay within twice the shortest final rim segment, so the
        # rectangles split into triangles of bounded aspect on tall thin vias.
        cap = min(h, 2 * _segment_lengths(seg_a[rim], seg_b[rim]).min())
        z = _axis(wall_edges[v], cap, min(cap, small), growth_ratio)
        for k in rim:
            for level in range(len(z) - 1):
                quad = np.array(((*seg_a[k], z[level]), (*seg_b[k], z[level]),
                                 (*seg_b[k], z[level + 1]), (*seg_a[k], z[level + 1])))
                triangles = ((0, 1, 2), (0, 2, 3)) if (k + level) % 2 == 0 else ((0, 1, 3), (1, 2, 3))
                vertices.append(quad)
                faces.append(np.array(triangles, dtype=np.int64) + offset)
                plane_ids.append(np.full(2, -1, dtype=np.int64))
                owners.append(np.full(2, len(sheets) + v, dtype=np.int64))
                offset += 4
                walls += 2
    stacked = np.concatenate(vertices)
    welded, inverse = np.unique(stacked, axis=0, return_inverse=True)
    faces = inverse.reshape(-1)[np.concatenate(faces)]
    plane_ids, owners = np.concatenate(plane_ids), np.concatenate(owners)
    report = triangle_quality(welded, faces, plane_ids)
    report.update(iterations=iterations, wall_faces=walls, horizontal_faces=int((plane_ids >= 0).sum()),
                  smallest_input_angle_degrees=math.degrees(smallest_input),
                  mesh_sha256=mesh_hash(welded, faces, plane_ids, owners))
    return PlanarTriangulation(welded, faces, plane_ids, owners, report)


def _inside(points, simplices, domain):
    """Triangles whose centroid is strictly inside the domain, without collinear hull slivers.

    Qhull may return flat simplices of collinear convex-hull points; their centroid
    sits on a boundary segment within rounding. The final area check proves the
    remaining triangles tile the domain.
    """
    shapely = optional_module("shapely", "Layered polygon layouts")
    corners = points[simplices]
    edge = corners[:, [1, 2, 0]] - corners
    cross = edge[:, 0, 0] * edge[:, 2, 1] - edge[:, 0, 1] * edge[:, 2, 0]
    longest = (edge ** 2).sum(-1).max(1)
    centres = corners.mean(1)
    keep = (np.abs(cross) > 1e-9 * longest) & shapely.contains_xy(domain, centres[:, 0], centres[:, 1])
    return simplices[keep]


def triangle_quality(vertices, faces, planes):
    """Minimum angle (degrees) and longest-edge/altitude aspect, horizontal faces and walls separately."""
    corners = vertices[faces]
    edge = corners[:, [1, 2, 0]] - corners
    length = np.linalg.norm(edge, axis=-1)
    area2 = np.linalg.norm(np.cross(edge[:, 0], -edge[:, 2]), axis=-1)
    cosine = -(edge * edge[:, [2, 0, 1]]).sum(-1) / (length * length[:, [2, 0, 1]])
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1))).min(1)
    aspect = length.max(1) ** 2 / area2
    report = {}
    for name, mask in (("horizontal", planes >= 0), ("wall", planes < 0)):
        if mask.any():
            report[f"{name}_min_angle_degrees"] = float(angle[mask].min())
            report[f"{name}_max_aspect_ratio"] = float(aspect[mask].max())
    return report


def mesh_hash(vertices, faces, planes, owners):
    digest = hashlib.sha256()
    for array in (np.ascontiguousarray(vertices, dtype=np.float64), np.ascontiguousarray(faces, dtype=np.int64),
                  np.ascontiguousarray(planes, dtype=np.int64), np.ascontiguousarray(owners, dtype=np.int64)):
        digest.update(array.tobytes())
    return digest.hexdigest()


def check_quality(report):
    """Raise when the mesh leaves the validated near-field integration region."""
    gate_angle = min(GATE_ANGLE_DEGREES, report["smallest_input_angle_degrees"] / 2)
    if report.get("horizontal_min_angle_degrees", 90.0) < gate_angle:
        raise ValueError(f"Planar mesh minimum angle {report['horizontal_min_angle_degrees']:.3g} degrees is below "
                         f"the {gate_angle:.3g}-degree quality gate")
    if report.get("horizontal_max_aspect_ratio", 1.0) > GATE_ASPECT:
        raise ValueError(f"Planar mesh aspect ratio {report['horizontal_max_aspect_ratio']:.3g} exceeds "
                         f"the quality gate {GATE_ASPECT:g}")
    if report.get("wall_max_aspect_ratio", 1.0) > WALL_GATE_ASPECT:
        raise ValueError(f"Via wall aspect ratio {report['wall_max_aspect_ratio']:.3g} exceeds "
                         f"the quality gate {WALL_GATE_ASPECT:g}")


def mesh_planar_layout(sheets, planes, h, **options):
    """Device LayeredMesh of planar_triangulation, with face_owners and the mesh report.

    Ordinary two-face edges use the open RWG basis on an oriented surface. A
    via landing on an intermediate plane, or two stacked vias meeting there,
    makes three-face rim edges; the whole layout then uses the junction RWG
    basis, whose pairwise functions conserve current at every junction edge.
    """
    from ..open_surface import OpenRWGBasis, OpenSurfaceMesh
    from ..topology import classify_surface
    from .build import build_extension
    from .mesh import ClosedConductorMesh, JunctionRWGBasis, LayeredMesh

    result = planar_triangulation(sheets, planes.cpu().tolist(), h, **options)
    check_quality(result.report)
    device = planes.device
    points = torch.tensor(result.vertices, device=device, dtype=torch.float64)
    faces = torch.tensor(result.faces, device=device, dtype=torch.int64)
    plane_ids = torch.tensor(result.planes, device=device, dtype=torch.int64)
    topology = classify_surface(faces, len(points))
    if topology.junction_edges:
        mesh = ClosedConductorMesh.__new__(ClosedConductorMesh)
        mesh.vertices, mesh.faces, mesh.topology = points, faces, topology
        mesh._topology_version = (id(mesh.faces), mesh.faces._version)
        mesh.validation_report = None
        basis = JunctionRWGBasis(mesh)
        layout = LayeredMesh(mesh, basis, plane_ids, basis.support_slots)
    else:
        if bool((plane_ids < 0).any()):
            faces = build_extension().orient(faces, topology.incidence_faces, topology.face_edges,
                                             topology.face_edge_signs)
        # Construction guarantees disjoint cells; skip the quadratic intersection validator.
        mesh = OpenSurfaceMesh(points, faces, validate=False)
        layout = LayeredMesh(mesh, OpenRWGBasis(mesh), plane_ids)
    layout.face_owners = torch.tensor(result.owners, device=device, dtype=torch.int64)
    layout.mesh_report = {**result.report, "mesher": "planar_conforming_delaunay_v1",
                          "junction_edges": topology.junction_edges}
    return layout
