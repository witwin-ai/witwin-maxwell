"""Layer cross-sections, corner detection and gap detection for auto meshing.

Host-side float64 helpers behind ``LayerRefinementSpec(corner_finder=...)``.
A layer cross-section is the set of polygon loops cut by the layer mid-plane
from each structure: exact polygons for axis-aligned ``Box``, ``PolySlab``
(including GDS imports and ``ComplexPolySlab``) and ``PlanarPEC``; a triangle
mesh slice for any other geometry with ``to_mesh``. In-plane coordinates are
the two remaining axes in ascending order (z -> (x, y), y -> (x, z),
x -> (y, z)), matching the polygon convention of the planar geometries.

Corners are detected per structure, not on a boolean union: a corner lying
strictly inside another selected cross-section is dropped, but corners where
two structures merely touch are kept (they only add refinement).
"""

from __future__ import annotations

import math

import numpy as np

_DEDUP_REL = 1e-12
_ANGLE_SLACK = 1e-9


def inplane_axes(axis: int) -> tuple[int, int]:
    return tuple(index for index in range(3) if index != axis)


def is_metal(structure) -> bool:
    """PEC / planar PEC / lossy-metal / surface-impedance / copper-foil structures."""
    from ...copper_foil import CopperFoil
    from ...geometry.planar_pec import PlanarPEC
    from ...media import LossyMetalMedium, SurfaceImpedanceMedium

    material = structure.material
    if isinstance(structure.geometry, PlanarPEC) or bool(getattr(material, "is_pec", False)):
        return True
    return isinstance(material, (LossyMetalMedium, SurfaceImpedanceMedium, CopperFoil))


def _identity_rotation(geometry) -> bool:
    rotation = getattr(geometry, "rotation", None)
    if rotation is None:
        return True
    quaternion = np.asarray(rotation.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    return bool(quaternion.size == 4 and abs(abs(quaternion[0]) - 1.0) < 1e-7)


def _signed_area(loop: np.ndarray) -> float:
    return 0.5 * float(np.sum(loop[:, 0] * np.roll(loop[:, 1], -1) - np.roll(loop[:, 0], -1) * loop[:, 1]))


def _dedupe(loop: np.ndarray) -> np.ndarray:
    scale = max(float(np.ptp(loop[:, 0])), float(np.ptp(loop[:, 1])), 1e-300)
    keep = np.linalg.norm(loop - np.roll(loop, 1, axis=0), axis=1) > _DEDUP_REL * scale
    return loop[keep]


def _slice_mesh(vertices: np.ndarray, faces: np.ndarray, axis: int, coord: float) -> list[np.ndarray]:
    """Closed loops where the plane ``x[axis] = coord`` cuts a closed triangle mesh."""
    extent = float(np.ptp(vertices, axis=0).max()) or 1.0
    nudge = 1e-9 * extent
    distance = vertices[:, axis] - coord
    # Keep the plane off vertices so every crossing is a proper edge crossing.
    while np.any(np.abs(distance) < nudge):
        coord += nudge
        distance = vertices[:, axis] - coord
    above = distance > 0.0
    crossing = faces[above[faces].any(axis=1) & ~above[faces].all(axis=1)]
    if crossing.size == 0:
        return []
    u, v = inplane_axes(axis)
    quantum = 1e-9 * extent

    def edge_point(i, j):
        t = distance[i] / (distance[i] - distance[j])
        point = vertices[i] + t * (vertices[j] - vertices[i])
        return point[[u, v]]

    def key(point):
        return tuple(np.round(point / quantum).astype(np.int64))

    adjacency: dict = {}
    points: dict = {}
    for tri in crossing:
        ends = []
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            if above[a] != above[b]:
                point = edge_point(a, b)
                k = key(point)
                points.setdefault(k, point)
                ends.append(k)
        if len(ends) == 2 and ends[0] != ends[1]:
            adjacency.setdefault(ends[0], []).append(ends[1])
            adjacency.setdefault(ends[1], []).append(ends[0])
    loops = []
    visited = set()
    for start in adjacency:
        if start in visited:
            continue
        loop, previous, current = [start], None, start
        visited.add(start)
        while True:
            options = [n for n in adjacency[current] if n != previous]
            if not options:
                break
            nxt = options[0]
            if nxt == start:
                break
            if nxt in visited:
                break
            visited.add(nxt)
            loop.append(nxt)
            previous, current = current, nxt
        if len(loop) >= 3:
            loops.append(np.asarray([points[k] for k in loop], dtype=np.float64))
    return loops


def cross_section_loops(geometry, axis: int, coord: float, tol: float) -> list[np.ndarray]:
    """Polygon loops (in-plane float64 coordinates) of ``geometry`` on a plane."""
    from ...geometry.planar_pec import PlanarPEC
    from ...geometry.polyslab import ComplexPolySlab, PolySlab, _offset_polygon_2d
    from witwin.core import Box

    u, v = inplane_axes(axis)
    if isinstance(geometry, PlanarPEC):
        if "xyz".index(geometry.normal_axis) != axis or abs(geometry.coordinate - coord) > tol:
            return []
        return [np.asarray(loop.detach().cpu().numpy(), dtype=np.float64) for loop in (geometry.vertices, *geometry.holes)]
    if isinstance(geometry, PolySlab) and _identity_rotation(geometry) and "xyz".index(geometry.axis) == axis:
        position = np.asarray(geometry.position.detach().cpu().numpy(), dtype=np.float64)
        bounds = np.asarray(geometry.bounds.detach().cpu().numpy(), dtype=np.float64)
        local = coord - position[axis]
        if not bounds[0] - tol <= local <= bounds[1] + tol:
            return []
        vertices = np.asarray(geometry.vertices.detach().cpu().numpy(), dtype=np.float64)
        sizes = list(geometry.loop_sizes)
        loops = np.split(vertices, np.cumsum(sizes)[:-1]) if isinstance(geometry, ComplexPolySlab) else [vertices]
        tan_angle = math.tan(float(geometry.sidewall_angle.detach().cpu()))
        reference = {"bottom": bounds[0], "top": bounds[1], "middle": 0.5 * (bounds[0] + bounds[1])}[
            geometry.reference_plane
        ]
        out = []
        for loop in loops:
            loop = _dedupe(np.asarray(loop, dtype=np.float64))
            if loop.shape[0] < 3:
                continue
            if tan_angle != 0.0:
                if _signed_area(loop) < 0.0:
                    loop = loop[::-1].copy()
                loop = _offset_polygon_2d(loop, -tan_angle * (local - reference))
            out.append(loop + position[[u, v]])
        return out
    if isinstance(geometry, Box) and _identity_rotation(geometry):
        from .autogrid import _geometry_aabb

        lo, hi = _geometry_aabb(geometry)
        if not lo[axis] - tol <= coord <= hi[axis] + tol or hi[u] <= lo[u] or hi[v] <= lo[v]:
            return []
        return [np.asarray([[lo[u], lo[v]], [hi[u], lo[v]], [hi[u], hi[v]], [lo[u], hi[v]]], dtype=np.float64)]
    try:
        vertices, faces = geometry.to_mesh()
    except NotImplementedError:
        return []
    vertices = np.asarray(vertices.detach().cpu().numpy(), dtype=np.float64)
    faces = np.asarray(faces.detach().cpu().numpy(), dtype=np.int64)
    if not vertices[:, axis].min() - tol <= coord <= vertices[:, axis].max() + tol:
        return []
    return _slice_mesh(vertices, faces, axis, coord)


def _point_in_loops(point: np.ndarray, loops) -> bool:
    """Even-odd membership of ``point`` in the union of ``loops``."""
    inside = False
    x, y = float(point[0]), float(point[1])
    for loop in loops:
        x0, y0 = loop[:, 0], loop[:, 1]
        x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
        straddle = (y0 > y) != (y1 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            cross_x = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        inside ^= bool(np.count_nonzero(straddle & (x < cross_x)) % 2)
    return inside


def _distance_to_loops(point: np.ndarray, loops) -> float:
    best = math.inf
    for loop in loops:
        a = loop
        b = np.roll(loop, -1, axis=0)
        ab = b - a
        denom = np.maximum(np.sum(ab * ab, axis=1), 1e-300)
        t = np.clip(np.sum((point - a) * ab, axis=1) / denom, 0.0, 1.0)
        best = min(best, float(np.min(np.linalg.norm(a + t[:, None] * ab - point, axis=1))))
    return best


def orient_loops(loops) -> list[np.ndarray]:
    """Orient loops so the material lies on the left (outer CCW, holes CW).

    A loop is a hole when one of its vertices lies inside an odd number of the
    other loops (even-odd rule, matching the planar geometry interiors).
    """
    loops = [_dedupe(np.asarray(loop, dtype=np.float64)) for loop in loops]
    loops = [loop for loop in loops if loop.shape[0] >= 3 and _signed_area(loop) != 0.0]
    out = []
    for index, loop in enumerate(loops):
        others = [other for j, other in enumerate(loops) if j != index]
        hole = _point_in_loops(loop[0], others) if others else False
        ccw = _signed_area(loop) > 0.0
        out.append(loop if ccw != hole else loop[::-1].copy())
    return out


def _rdp_open(points: np.ndarray, tolerance: float) -> np.ndarray:
    keep = np.zeros(points.shape[0], dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, points.shape[0] - 1)]
    while stack:
        first, last = stack.pop()
        if last - first < 2:
            continue
        a, b = points[first], points[last]
        ab = b - a
        length = float(np.hypot(*ab))
        segment = points[first + 1 : last]
        if length == 0.0:
            distance = np.linalg.norm(segment - a, axis=1)
        else:
            distance = np.abs(ab[0] * (segment[:, 1] - a[1]) - ab[1] * (segment[:, 0] - a[0])) / length
        index = int(np.argmax(distance))
        if distance[index] > tolerance:
            middle = first + 1 + index
            keep[middle] = True
            stack += [(first, middle), (middle, last)]
    return points[keep]


def simplify_loop(loop: np.ndarray, tolerance: float) -> np.ndarray:
    """Closed-loop Douglas-Peucker simplification (split at the farthest vertex)."""
    if loop.shape[0] <= 3:
        return loop
    far = int(np.argmax(np.linalg.norm(loop - loop[0], axis=1)))
    first = _rdp_open(loop[: far + 1], tolerance)
    second = _rdp_open(np.vstack([loop[far:], loop[:1]]), tolerance)
    simplified = np.vstack([first[:-1], second[:-1]])
    return simplified if simplified.shape[0] >= 3 else loop


def find_corners(loop: np.ndarray, angle_threshold: float, distance_threshold: float | None = None):
    """Corners of one oriented loop (material on the left).

    Returns ``(points (K, 2), convex (K,))``. A vertex is a corner when the turn
    between its incoming and outgoing edges is at least ``angle_threshold``;
    ``convex`` marks corners where the material spans less than pi. With
    ``distance_threshold`` the loop is Douglas-Peucker simplified first and
    consecutive corners closer than the threshold are merged to their mean.
    """
    points = _dedupe(np.asarray(loop, dtype=np.float64))
    if distance_threshold is not None:
        points = simplify_loop(points, distance_threshold)
    if points.shape[0] < 3:
        return np.zeros((0, 2)), np.zeros(0, dtype=bool)
    incoming = points - np.roll(points, 1, axis=0)
    outgoing = np.roll(points, -1, axis=0) - points
    incoming /= np.linalg.norm(incoming, axis=1)[:, None]
    outgoing /= np.linalg.norm(outgoing, axis=1)[:, None]
    turn = np.arccos(np.clip(np.sum(incoming * outgoing, axis=1), -1.0, 1.0))
    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    selected = turn >= angle_threshold - _ANGLE_SLACK
    corners, convex = points[selected], cross[selected] > 0.0
    if distance_threshold is not None and corners.shape[0] > 1:
        corners, convex = _merge_close(corners, convex, distance_threshold)
    return corners, convex


def _merge_close(corners, convex, threshold):
    groups = [[0]]
    for index in range(1, corners.shape[0]):
        if np.linalg.norm(corners[index] - corners[groups[-1][-1]]) < threshold:
            groups[-1].append(index)
        else:
            groups.append([index])
    if len(groups) > 1 and np.linalg.norm(corners[groups[0][0]] - corners[groups[-1][-1]]) < threshold:
        groups[0] = groups.pop() + groups[0]
    merged = np.asarray([corners[group].mean(axis=0) for group in groups])
    flags = np.asarray([bool(convex[group[0]]) for group in groups])
    return merged, flags


def _loops_box(loops):
    if not loops:
        return None
    points = np.concatenate(loops)
    return points.min(axis=0), points.max(axis=0)


def _box_contains(box, point) -> bool:
    return box is not None and bool(np.all(point > box[0]) and np.all(point < box[1]))


def drop_covered_corners(corners, convex, owner: int, sections, tol: float):
    """Drop corners strictly inside another section (an approximate union)."""
    if corners.shape[0] == 0:
        return corners, convex
    keep = np.ones(corners.shape[0], dtype=bool)
    boxes = [_loops_box(loops) for loops in sections]
    for index, point in enumerate(corners):
        for other, loops in enumerate(sections):
            if other == owner or not loops or not _box_contains(boxes[other], point):
                continue
            if _point_in_loops(point, loops) and _distance_to_loops(point, loops) > tol:
                keep[index] = False
                break
    return corners[keep], convex[keep]


def scanline_gaps(sections, along: int, max_width: float, span: tuple[float, float]):
    """In-plane gaps between metal sections, measured along in-plane index ``along``.

    Scan lines run at midpoints between consecutive distinct vertex coordinates
    of the other in-plane index, so rectilinear gaps are measured exactly. Each
    returned ``(lo, hi)`` is bounded by metal on both sides on some scan line,
    is narrower than ``max_width``, and overlaps ``span``.
    """
    across = 1 - along
    levels = np.unique(np.concatenate([loop[:, across] for loops in sections for loop in loops])) if sections else []
    gaps = set()
    ranges = [(min(l[:, across].min() for l in loops), max(l[:, across].max() for l in loops)) for loops in sections]
    for level in 0.5 * (np.asarray(levels)[:-1] + np.asarray(levels)[1:]):
        intervals = []
        for loops, (r_lo, r_hi) in zip(sections, ranges):
            if not r_lo < level < r_hi:
                continue
            crossings = []
            for loop in loops:
                a, b = loop, np.roll(loop, -1, axis=0)
                straddle = (a[:, across] > level) != (b[:, across] > level)
                t = (level - a[straddle, across]) / (b[straddle, across] - a[straddle, across])
                crossings.extend(a[straddle, along] + t * (b[straddle, along] - a[straddle, along]))
            crossings = np.sort(np.asarray(crossings))
            intervals.extend(zip(crossings[0::2], crossings[1::2]))
        if len(intervals) < 2:
            continue
        intervals.sort()
        merged = [list(intervals[0])]
        for lo, hi in intervals[1:]:
            if lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        for (_, left), (right, _) in zip(merged[:-1], merged[1:]):
            if right - left < max_width and right > span[0] and left < span[1]:
                gaps.add((float(left), float(right)))
    return sorted(gaps)
