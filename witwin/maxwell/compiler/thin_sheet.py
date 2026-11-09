"""Conformal two-sided cut complex for planar thin sheets at arbitrary tilt.

A planar sheet (``SurfaceMesh`` whose triangles are coplanar) is embedded in
the Cartesian Yee grid without staircasing. The sheet plane ``n.x = d``
classifies every grid node as below (-1), above (+1) or exactly on (0) the
plane. The local discrete complex then contains:

* every ordinary Yee edge and face of a box around the sheet;
* for each Yee face whose sheet chord has both endpoints inside the outline,
  two sub-faces (one per side) with their own magnetic flux; partial rim
  chords leave an ordinary face connecting the air around the sheet end;
* for each Yee edge bounding such a face and pierced by the plane, two
  segments (one per side) with their own electric line integral;
* for each Yee edge lying exactly on the sheet, two side traces, and the Yee
  faces lying on the sheet keep one shared normal flux (the axis-aligned
  two-sided foil structure).

Faraday's law on every (sub-)face is the exact oriented circulation of its
boundary pieces. The tangential electric trace along a cut segment is zero for
PEC. Finite copper carries independent chord fields and two independent
sheet-face fluxes per closed cut-cell loop. These curl rows never average
sides; the passive copper two-port alone couples the chord pair.
Ampere's law uses the transpose of the same circulation matrix with diagonal
positive masses, so the semi-discrete lossless system is skew-adjoint in a
positive energy and leapfrog is stable below a computed spectral bound.

Electric masses partition every Yee dual box exactly between the two sides:
the part of a box beyond the sheet is moved to the nearest same-component
unknown on that side, so each side is covered up to the sheet. Magnetic
sub-faces keep the dual length times their own area (Dey-Mittra). A
sub-face whose curl-curl diagonal would exceed its parent Yee face's is floored in
its magnetic mass; this bounds the local curl norm (small-cut stability) at
the cost of an O(h) local error confined to the floored small cuts. Edges lying exactly on
the sheet split their dual box between the two sides by exact half-space
clipping, which reproduces the existing axis-aligned foil masses.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import warnings

import numpy as np
import torch

from ..copper_foil import CopperFoil
from ..ports import LumpedPort

SHEET_MODEL = "conformal_two_sided"
CHORD_MASS_FACTOR = 0.5
_AXES = "xyz"
# Cyclic (normal, b, c) frames with b x c = normal.
_FRAME = ((0, 1, 2), (1, 2, 0), (2, 0, 1))


def owns_thin_sheet(structure) -> bool:
    """True when the conformal thin-sheet runtime owns this structure."""
    if not bool(getattr(structure, "enabled", True)):
        return False
    geometry = structure.geometry
    if getattr(geometry, "kind", None) != "surface_mesh":
        return False
    if isinstance(structure.material, CopperFoil):
        return True
    return getattr(geometry, "sheet_model", None) == SHEET_MODEL


def thin_sheet_structures(scene):
    return tuple(structure for structure in scene.structures if owns_thin_sheet(structure))


# ---------------------------------------------------------------------------
# Sheet geometry: one or more planar facets
# ---------------------------------------------------------------------------

def _plane_value(points, facet):
    """Signed plane function n.x - d of points [P, 3] (fixed operation order)."""
    n = facet.normal
    return points[:, 0] * n[0] + points[:, 1] * n[1] + points[:, 2] * n[2] - facet.offset


def _segment_distance_2d(points, a, b):
    """Distance from points [P, 2] to segments a-b ([S, 2] each) -> [P, S]."""
    d = b - a
    length2 = np.maximum(np.sum(d * d, -1), 1e-300)
    w = points[:, None, :] - a[None]
    t = np.clip(np.sum(w * d[None], -1) / length2[None], 0.0, 1.0)
    closest = a[None] + t[..., None] * d[None]
    return np.linalg.norm(points[:, None, :] - closest, axis=-1)


@dataclass(frozen=True)
class SheetFacet:
    """A planar facet: plane n.x = d, in-plane basis and local triangles."""
    index: int
    normal: np.ndarray
    offset: float
    origin: np.ndarray
    basis: np.ndarray            # [2, 3]
    triangles: np.ndarray        # [T, 3, 2] in-plane coordinates
    rim: np.ndarray              # [R, 2, 2] boundary segments without a neighbour facet
    folds: np.ndarray            # [F, 2, 2] boundary segments shared with another facet

    def local(self, points):
        return (np.asarray(points, dtype=np.float64) - self.origin) @ self.basis.T

    def inside(self, points, tolerance):
        """Closed point-in-facet test for points of the facet plane [P, 3]."""
        local = self.local(points)
        a, b, c = self.triangles[:, 0], self.triangles[:, 1], self.triangles[:, 2]
        u, v = b - a, c - a
        w = local[:, None, :] - a[None]
        det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        alpha = (w[..., 0] * v[None, :, 1] - w[..., 1] * v[None, :, 0]) / det[None]
        beta = (u[None, :, 0] * w[..., 1] - u[None, :, 1] * w[..., 0]) / det[None]
        edge = np.sqrt(np.maximum(np.abs(det), 1e-300))[None]
        relative = tolerance / edge
        hit = (alpha >= -relative) & (beta >= -relative) & (alpha + beta <= 1 + relative)
        return hit.any(-1)

    def planar_distance(self, points):
        """In-plane distance from the projection of points [P, 3] to the facet (0 inside)."""
        local = self.local(points)
        inside = self.inside(points, 0.0)
        boundary = np.concatenate((self.rim, self.folds), 0)
        if not len(boundary):
            return np.zeros(len(local))
        distance = _segment_distance_2d(local, boundary[:, 0], boundary[:, 1]).min(-1)
        return np.where(inside, 0.0, distance)


@dataclass
class SheetSurface:
    """Oriented, piecewise-planar sheet: planar facets joined at fold lines.

    The signed level function of a point is the plane function of its nearest
    facet (distance to the closed facet polygon). Its zero set is the surface,
    continued beyond the outer rim by the boundary facets' planes, and its
    sign is consistent across folds. A single facet reproduces the planar
    complex exactly.
    """
    facets: tuple
    vertices: np.ndarray
    scale: float
    adjacency: dict              # (fa, fb) -> (point, direction) of the fold line
    convex: dict                 # (fa, fb) -> True if the - side is the intersection of both - half-spaces
    fold_segments: tuple = ()    # (fa, fb, start [3], end [3]) for every mesh edge on a fold

    def box_below_volume(self, lower, upper, tolerance):
        """Exact - side volume of a box: the facets seen by 27 samples plus those of every fold meeting the box.

        A fold crossing the box near a corner can escape all 27 samples, so the
        facets of every fold segment meeting the box are added exactly.
        """
        samples = np.stack(np.meshgrid(*[np.array((lower[ax], 0.5 * (lower[ax] + upper[ax]), upper[ax]))
                                         for ax in range(3)], indexing="ij"), -1).reshape(-1, 3)
        _, seen = self.classify(samples)
        seen = set(int(v) for v in seen) | self.folds_in_box(lower, upper, tolerance)
        return self.below_volume(lower, upper, seen)

    def folds_in_box(self, lower, upper, tolerance):
        """Facets of every fold segment that meets the closed box [lower, upper] (slab test)."""
        found = set()
        lower = np.asarray(lower, dtype=np.float64) - tolerance
        upper = np.asarray(upper, dtype=np.float64) + tolerance
        for fa, fb, start, end in self.fold_segments:
            low, high = 0.0, 1.0
            delta = end - start
            for axis in range(3):
                if abs(delta[axis]) <= 1e-300:
                    if start[axis] < lower[axis] or start[axis] > upper[axis]:
                        low, high = 1.0, 0.0
                        break
                    continue
                t0 = (lower[axis] - start[axis]) / delta[axis]
                t1 = (upper[axis] - start[axis]) / delta[axis]
                low, high = max(low, min(t0, t1)), min(high, max(t0, t1))
            if low <= high:
                found.update((fa, fb))
        return found

    @property
    def planar(self):
        return len(self.facets) == 1

    def classify(self, points, *, chunk=65536):
        """Nearest facet and its signed plane function for points [P, 3]."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if self.planar:
            return _plane_value(points, self.facets[0]), np.zeros(len(points), dtype=np.int64)
        phi = np.empty(len(points))
        index = np.empty(len(points), dtype=np.int64)
        for start in range(0, len(points), chunk):
            block = points[start:start + chunk]
            values = np.stack([_plane_value(block, f) for f in self.facets], -1)
            planar = np.stack([f.planar_distance(block) for f in self.facets], -1)
            distance = np.hypot(values, planar)
            # Among equidistant facets prefer the one the point lies farthest
            # off-plane of: a point on the extension of one facet's plane beyond
            # an exact 90-degree fold is as far from that facet (in-plane) as
            # from the other facet (off-plane), but it is not on the surface.
            tie = distance <= distance.min(-1, keepdims=True) + 1e-12 * self.scale
            nearest = np.argmax(np.where(tie, np.abs(values), -1.0), -1)
            index[start:start + len(block)] = nearest
            phi[start:start + len(block)] = values[np.arange(len(block)), nearest]
        return phi, index

    def normals(self, index):
        return np.stack([f.normal for f in self.facets])[np.asarray(index)]

    def inside(self, points, tolerance):
        """Outline test for points on (or projected onto) the surface [P, 3].

        A point is inside when its projection onto its nearest facet lies in the
        facet, or when the nearest facet boundary there is a fold (the point is
        in the wedge of two facets) rather than the outer rim.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if self.planar:
            return self.facets[0].inside(points, tolerance)
        _, nearest = self.classify(points)
        result = np.zeros(len(points), dtype=bool)
        for f in self.facets:
            select = np.flatnonzero(nearest == f.index)
            if not len(select):
                continue
            block = points[select]
            projected = block - np.outer(block @ f.normal - f.offset, f.normal)
            hit = f.inside(projected, tolerance)
            if not hit.all() and len(f.folds):
                local = f.local(projected[~hit])
                fold = _segment_distance_2d(local, f.folds[:, 0], f.folds[:, 1]).min(-1)
                rim = (_segment_distance_2d(local, f.rim[:, 0], f.rim[:, 1]).min(-1)
                       if len(f.rim) else np.full(len(local), np.inf))
                # Strictly nearer a fold than the rim: a point beyond the end of
                # a fold line (equidistant from the fold end and the rim corner)
                # lies outside the sheet.
                hit[~hit] = fold + tolerance < rim
            result[select] = hit
        return result

    def members(self, points, tolerance):
        """Facets each surface point [P, 3] lies on: its nearest facet plus fold neighbours."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        _, nearest = self.classify(points)
        result = [{int(f)} for f in nearest]
        if self.planar:
            return result
        for f in self.facets:
            close = (np.abs(_plane_value(points, f)) <= tolerance) & (f.planar_distance(points) <= tolerance)
            for i in np.flatnonzero(close):
                result[i].add(f.index)
        return result

    def project(self, points):
        """Orthogonal projection of points [P, 3] onto the plane of their nearest facet."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        phi, nearest = self.classify(points)
        return points - phi[:, None] * self.normals(nearest)

    def fold_point(self, fa, fb, axis, coordinate):
        """Point where the fold line between two facets meets the plane x_axis = coordinate."""
        key = (min(fa, fb), max(fa, fb))
        if key not in self.adjacency:
            return None
        point, direction = self.adjacency[key]
        if abs(direction[axis]) < 1e-12:
            return None
        return point + (coordinate - point[axis]) / direction[axis] * direction

    def pierce(self, start, end, tolerance):
        """Parameter t of the unique crossing of the segment start-end with the surface."""
        start, end = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
        if self.planar:
            f = self.facets[0]
            p0, p1 = float(start @ f.normal - f.offset), float(end @ f.normal - f.offset)
            return p0 / (p0 - p1)
        i0, i1 = self.classify(np.stack((start, end)))[1]
        found = []
        for index in sorted(self.neighbourhood(int(i0)) | self.neighbourhood(int(i1))):
            f = self.facets[index]
            p0, p1 = (float(v) for v in _plane_value(np.stack((start, end)), f))
            if p0 == p1 or p0 * p1 > 0:
                continue
            t = p0 / (p0 - p1)
            point = start + t * (end - start)
            # The crossing must lie on the zero set of the level function (the
            # facet itself or its continuation beyond the outer rim).
            phi, nearest = self.classify(point[None])
            if int(nearest[0]) == index or abs(phi[0]) <= tolerance:
                if not any(abs(t - other) <= 1e-12 for other in found):
                    found.append(t)
        if len(found) == 1 or (len(found) == 2 and abs(found[0] - found[1]) <= 1e-9):
            return found[0]
        if len(found) > 1:
            raise NotImplementedError("A Yee edge crosses the faceted sheet more than once; refine the grid.")
        # Crossing at a fold: bisect the continuous level function.
        lo, hi = 0.0, 1.0
        phi_lo = self.classify(start[None])[0][0]
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            value = self.classify((start + mid * (end - start))[None])[0][0]
            if (value < 0) == (phi_lo < 0):
                lo, phi_lo = mid, value
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def neighbourhood(self, index):
        """A facet and the facets sharing a fold with it."""
        return {int(index)} | {b if a == index else a for a, b in self.adjacency if index in (a, b)}

    def crossings(self, starts, ends, tolerance, exclude=()):
        """Number of distinct points where segments [E] cross the facet polygons (not their extensions)."""
        parameters = np.full((len(starts), len(self.facets)), np.nan)
        for f in self.facets:
            if f.index in exclude:
                continue
            p0 = _plane_value(starts, f)
            p1 = _plane_value(ends, f)
            select = np.flatnonzero(p0 * p1 < 0)
            if not len(select):
                continue
            t = p0[select] / (p0[select] - p1[select])
            points = starts[select] + t[:, None] * (ends[select] - starts[select])
            hit = f.inside(points, tolerance)
            parameters[select[hit], f.index] = t[hit]
        ordered = np.sort(parameters, -1)                    # NaN last
        valid = ~np.isnan(ordered)
        distinct = valid[:, :1].astype(np.int64)
        if ordered.shape[1] > 1:
            distinct = distinct + np.sum(valid[:, 1:] & (np.abs(np.diff(ordered, axis=-1)) > 1e-9), -1, keepdims=True)
        return distinct[:, 0]

    def below_volume(self, lower, upper, facet_ids):
        """Exact volume of the box [lower, upper] on the - side of the surface.

        ``facet_ids`` are the facets seen by the box. One facet uses its plane.
        Several facets must form a fold-connected group with folds of one
        kind: around convex folds the - side is the intersection of the facets'
        - half-spaces, around concave folds the complement of the intersection
        of their + half-spaces.
        """
        facet_ids = sorted(set(int(v) for v in facet_ids))
        if len(facet_ids) == 1:
            f = self.facets[facet_ids[0]]
            return _halfspace_volume(lower, upper, f.normal, f.offset)
        folds = [key for key in self.adjacency if key[0] in facet_ids and key[1] in facet_ids]
        connected = {facet_ids[0]}
        for _ in facet_ids:
            connected |= {b for a, b in folds if a in connected} | {a for a, b in folds if b in connected}
        kinds = {self.convex[key] for key in folds}
        if connected != set(facet_ids) or len(kinds) != 1:
            raise NotImplementedError("A Yee dual cell meets thin-sheet facets that are not joined by folds of one "
                                      "kind (separate sheet parts or an inflection within one cell); refine the grid.")
        facets = [self.facets[i] for i in facet_ids]
        if kinds.pop():
            return _clip_box_volume(lower, upper, [(f.normal, f.offset) for f in facets])
        total = float(np.prod(np.asarray(upper, dtype=np.float64) - np.asarray(lower, dtype=np.float64)))
        return total - _clip_box_volume(lower, upper, [(-f.normal, -f.offset) for f in facets])


def _clip_box_volume(lower, upper, planes):
    """Volume of the box [lower, upper] inside every half-space {n.x <= d}."""
    lower, upper = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    corners = np.array([[(lower, upper)[m >> k & 1][k] for k in range(3)] for m in range(8)])
    faces = [[0, 2, 6, 4], [1, 5, 7, 3], [0, 4, 5, 1], [2, 3, 7, 6], [0, 1, 3, 2], [4, 6, 7, 5]]
    polygons = [corners[face] for face in faces]
    for normal, offset in planes:
        normal = np.asarray(normal, dtype=np.float64)
        clipped, cap = [], []
        for polygon in polygons:
            values = polygon @ normal - offset
            out = []
            for i in range(len(polygon)):
                j = (i + 1) % len(polygon)
                if values[i] <= 0:
                    out.append(polygon[i])
                if (values[i] < 0 < values[j]) or (values[j] < 0 < values[i]):
                    point = polygon[i] + values[i] / (values[i] - values[j]) * (polygon[j] - polygon[i])
                    out.append(point)
                    cap.append(point)
                elif values[i] == 0:
                    cap.append(polygon[i])
            if len(out) >= 3:
                clipped.append(np.array(out))
        if len(cap) >= 3:
            cap = np.unique(np.round(np.array(cap), 15), axis=0)
            if len(cap) >= 3:
                center = cap.mean(0)
                seed = np.eye(3)[int(np.argmin(np.abs(normal)))]
                e1 = np.cross(normal, seed); e1 /= np.linalg.norm(e1)
                e2 = np.cross(normal, e1)
                angle = np.arctan2((cap - center) @ e2, (cap - center) @ e1)
                clipped.append(cap[np.argsort(angle)])
        polygons = clipped
        if not polygons:
            return 0.0
    points = np.concatenate(polygons, 0)
    center = points.mean(0)
    volume = 0.0
    for polygon in polygons:
        cross = np.zeros(3)
        for i in range(len(polygon)):
            cross += np.cross(polygon[i], polygon[(i + 1) % len(polygon)])
        area_normal = 0.5 * cross
        area = np.linalg.norm(area_normal)
        if area <= 0:
            continue
        volume += area * abs(float((center - polygon[0]) @ (area_normal / area))) / 3.0
    return volume


def sheet_surface(geometry, *, fold_limit_deg=90.0) -> SheetSurface:
    vertices, triangles = geometry.to_mesh()
    if torch.is_tensor(vertices) and vertices.requires_grad:
        raise NotImplementedError("Conformal thin sheets are forward-only; trainable SurfaceMesh vertices are not "
                                  "supported.")
    vertices = vertices.detach().cpu().double().numpy()
    triangles = triangles.detach().cpu().numpy().astype(np.int64)
    corners = vertices[triangles]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    norms = np.linalg.norm(cross, axis=-1)
    scale = float(np.linalg.norm(vertices.max(0) - vertices.min(0)))
    if np.any(norms <= 1e-14 * max(scale, 1e-300) ** 2):
        raise ValueError("Thin-sheet SurfaceMesh has degenerate triangles.")
    unit = cross / norms[:, None]
    # Edge topology: manifold and consistently oriented.
    directed = {}
    for t, tri in enumerate(triangles):
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            if (a, b) in directed:
                raise NotImplementedError("The thin-sheet SurfaceMesh is not consistently oriented or not manifold.")
            directed[(a, b)] = t
    neighbours = {}
    for (a, b), t in directed.items():
        other = directed.get((b, a))
        if other is not None:
            neighbours.setdefault(t, set()).add(other)
    # Coplanar groups (facets).
    parent = list(range(len(triangles)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for t, others in neighbours.items():
        for o in others:
            if unit[t] @ unit[o] >= 1 - 1e-10 and abs(corners[o] @ unit[t] - corners[t][0] @ unit[t]).max() <= 1e-9 * scale:
                parent[find(t)] = find(o)
    groups = {}
    for t in range(len(triangles)):
        groups.setdefault(find(t), []).append(t)
    group_list = sorted(groups.values(), key=lambda g: g[0])
    owner = np.empty(len(triangles), dtype=np.int64)
    for index, group in enumerate(group_list):
        owner[group] = index
    facets = []
    adjacency, convex = {}, {}
    fold_segments = []
    for index, group in enumerate(group_list):
        total = cross[group].sum(0)
        normal = total / np.linalg.norm(total)
        used = np.unique(triangles[group])
        offsets = vertices[used] @ normal
        offset = float(offsets.mean())
        if (unit[group] @ normal).min() < 1 - 1e-10 or np.abs(offsets - offset).max() > 1e-9 * scale:
            raise NotImplementedError("A thin-sheet facet is not planar.")
        seed = np.eye(3)[int(np.argmin(np.abs(normal)))]
        first = np.cross(normal, seed)
        first /= np.linalg.norm(first)
        second = np.cross(normal, first)
        basis = np.stack((first, second))
        origin = normal * offset
        local = (corners[group] - origin) @ basis.T
        rim, folds = [], []
        for t in group:
            tri = triangles[t]
            for k in range(3):
                a, b = int(tri[k]), int(tri[(k + 1) % 3])
                other = directed.get((b, a))
                segment = (vertices[[a, b]] - origin) @ basis.T
                if other is None:
                    rim.append(segment)
                elif owner[other] != index:
                    folds.append(segment)
                    key = (min(index, int(owner[other])), max(index, int(owner[other])))
                    if index < int(owner[other]):
                        fold_segments.append((index, int(owner[other]), vertices[a].copy(), vertices[b].copy()))
                    direction = vertices[b] - vertices[a]
                    adjacency.setdefault(key, (vertices[a].copy(), direction / np.linalg.norm(direction)))
        facets.append(SheetFacet(index, normal, offset, origin, basis, local,
                                 np.array(rim).reshape(-1, 2, 2), np.array(folds).reshape(-1, 2, 2)))
    for (fa, fb), (point, direction) in adjacency.items():
        a, b = facets[fa], facets[fb]
        if float(a.normal @ b.normal) < math.cos(math.radians(fold_limit_deg)) - 1e-12:
            raise NotImplementedError(f"Thin-sheet facets fold by more than {fold_limit_deg:g} degrees.")
        # The fold must be a straight line common to both planes.
        if abs(float(direction @ a.normal)) > 1e-9 or abs(float(direction @ b.normal)) > 1e-9:
            raise NotImplementedError("Thin-sheet facets meet along a non-straight fold.")
        centroid_b = b.origin + b.triangles.reshape(-1, 2).mean(0) @ b.basis
        convex[(fa, fb)] = bool(centroid_b @ a.normal - a.offset < 0)
    return SheetSurface(tuple(facets), vertices, scale, adjacency, convex, tuple(fold_segments))


def _halfspace_volume(lower, upper, normal, offset):
    """Exact volume of the box [lower, upper] inside {normal.x <= offset}."""
    lower = np.array(lower, dtype=np.float64)
    upper = np.array(upper, dtype=np.float64)
    normal = np.array(normal, dtype=np.float64)
    for axis in range(3):
        if normal[axis] < 0:
            lower[axis], upper[axis] = -upper[axis], -lower[axis]
            normal[axis] = -normal[axis]
    lengths = upper - lower
    rhs = offset - float(normal @ lower)
    keep = normal > 1e-7
    volume = float(np.prod(lengths[~keep]))
    if not keep.any():
        return volume if rhs >= 0 else 0.0
    n, length = normal[keep], lengths[keep]
    m = len(n)
    total = 0.0
    for mask in range(1 << m):
        shift = sum(n[i] * length[i] for i in range(m) if mask >> i & 1)
        value = rhs - shift
        if value > 0:
            total += (-1) ** bin(mask).count("1") * value ** m
    total /= math.factorial(m) * float(np.prod(n))
    return volume * min(max(total, 0.0), float(np.prod(length)))


# ---------------------------------------------------------------------------
# Local complex
# ---------------------------------------------------------------------------

@dataclass
class SheetGapFeed:
    """Compiler-only gap geometry; side volumes are ordered (-, +).

    ``chord_index`` addresses the copper complex (-1 for an unpromoted trace
    or PEC). ``trace_index`` addresses the geometry's cut/on-sheet trace order;
    ``trace_dofs`` is (-1, -1) for cut chords. Open entries have no feed term.
    """
    port_name: str
    facet: int
    direction: np.ndarray
    midpoint: np.ndarray
    length: float
    width: float
    cut_cells: np.ndarray
    chord_index: np.ndarray
    chord_face: np.ndarray
    chord_rows: np.ndarray
    chord_orientation: np.ndarray
    trace_index: np.ndarray
    trace_dofs: np.ndarray
    chord_path: list
    chord_length: np.ndarray
    chord_width: np.ndarray
    chord_volume: np.ndarray
    chord_projection: np.ndarray
    open_chords: np.ndarray


@dataclass
class ThinSheetComplex:
    """Sparse local complex in E-value / H-value units (float64, host)."""
    material: object
    is_pec: bool
    name: str
    surface: SheetSurface
    # E unknowns
    e_component: np.ndarray
    e_index: np.ndarray          # [nE, 3] global Yee index of the parent edge
    e_extra: np.ndarray          # -1 for the native slot, else extra-storage slot
    e_side: np.ndarray           # -1/+1 definite side, 0 regular/ambiguous
    e_length: np.ndarray
    e_volume: np.ndarray
    e_fixed: np.ndarray          # bool: value is held (PEC trace or PEC wall)
    e_position: np.ndarray
    e_split: np.ndarray          # bool: pierced segment or on-sheet trace pair member
    e_modified: np.ndarray       # bool: mass differs from the native Yee dual volume
    e_phi: np.ndarray            # signed level function at e_position
    e_normal: np.ndarray         # [nE, 3] normal of the nearest facet
    # H unknowns
    h_component: np.ndarray
    h_index: np.ndarray
    h_extra: np.ndarray
    h_area: np.ndarray           # effective (floored) area used in the magnetic mass
    h_true_area: np.ndarray
    h_dual: np.ndarray
    h_special: np.ndarray        # bool: row differs from the native Yee update
    # circulation matrix K[h, e] (E-value units): circ_h = sum_e K[h,e] E_e
    k_rows: np.ndarray
    k_cols: np.ndarray
    k_vals: np.ndarray
    # Exact on-sheet trace pairs; cut traces are independent chord unknowns.
    trace_dofs: np.ndarray = None
    trace_length: np.ndarray = None
    trace_width: np.ndarray = None
    # Copper chord complex. Side columns are 2*c (-) and 2*c+1 (+).
    # K_x contains own Cartesian pieces, without reconstructed cut traces.
    # K_c has len(h_area) + len(sheet_face_area) rows; sheet rows come last.
    k_x_rows: np.ndarray = None
    k_x_cols: np.ndarray = None
    k_x_vals: np.ndarray = None
    k_c_rows: np.ndarray = None
    k_c_cols: np.ndarray = None
    k_c_vals: np.ndarray = None
    chord_face: np.ndarray = None       # [T, 4] (axis, i, j, k), -1 for on-sheet edges
    chord_trace: np.ndarray = None      # original trace id, including promoted on-sheet pairs
    chord_rows: np.ndarray = None       # [T, 2] sub-face rows, -1 for on-sheet pairs
    chord_orientation: np.ndarray = None  # [T, 2] sub-face CCW signs
    chord_path: list = None             # oriented polylines, including fold points
    chord_length: np.ndarray = None
    chord_width: np.ndarray = None
    chord_tensor: np.ndarray = None     # [T, 3, 3] length-weighted tangent dyads
    chord_distance: np.ndarray = None   # [T, 2] absolute level at sub-face centroids
    chord_volume: np.ndarray = None     # [T, 2] additive masses; clipped boxes for on-sheet pairs
    chord_fixed: np.ndarray = None      # held PEC-wall chords
    sheet_face_cell: np.ndarray = None
    sheet_face_side: np.ndarray = None
    sheet_face_volume: np.ndarray = None  # clipped, pair-conserving cell volumes
    sheet_face_chords: list = None      # ordered chord ids around each loop
    sheet_face_orientations: list = None
    sheet_face_traces: list = None      # original on-sheet trace ids in each loop
    sheet_face_vertices: list = None    # CCW about the + facet normal at the vertex mean
    sheet_face_area: np.ndarray = None  # triangle-fan area (also for bent loops)
    sheet_face_area_eff: np.ndarray = None
    sheet_face_dual: np.ndarray = None
    feeds: tuple[SheetGapFeed, ...] = ()
    extra_e: int = 0
    extra_h: int = 0
    statistics: dict = field(default_factory=dict)


def _grid(scene):
    nodes = [np.asarray(getattr(scene, f"{a}_nodes64"), dtype=np.float64) for a in _AXES]
    primal = [np.diff(value) for value in nodes]
    return nodes, primal


def _dual_extent(primal, index):
    """Yee dual extent at node indices (half cell at the domain ends)."""
    index = np.asarray(index)
    lower = np.where(index > 0, primal[np.clip(index - 1, 0, len(primal) - 1)], 0.0)
    upper = np.where(index < len(primal), primal[np.clip(index, 0, len(primal) - 1)], 0.0)
    return 0.5 * (lower + upper)


def _clip_polygon_box(polygon, lower, upper):
    """Clip a convex polygon to a Cartesian box (in two or three dimensions)."""
    for axis in range(len(lower)):
        for sign, bound in ((1, lower[axis]), (-1, upper[axis])):
            output = []
            for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
                a, b = sign * (start[axis] - bound), sign * (end[axis] - bound)
                if a >= 0:
                    output.append(start)
                if (a < 0) != (b < 0):
                    output.append(start + a / (a - b) * (end - start))
            polygon = np.asarray(output).reshape(-1, len(lower))
    return polygon


def _polygon_area(polygon):
    return 0.5 * abs(np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1)
                            - polygon[:, 1] * np.roll(polygon[:, 0], -1)))


def _classify_sheet_gaps(scene, structure, surface, nodes, primal, tolerance):
    """Admit full-width rectangular gaps and their one-cell arm margins."""
    gaps = []
    for port in scene.ports:
        if not isinstance(port, LumpedPort) or port.wire_binding is not None:
            continue
        terminals = np.array((port.negative, port.positive))
        members = [set() for _ in terminals]
        for facet in surface.facets:
            on = ((np.abs(_plane_value(terminals, facet)) <= tolerance)
                  & facet.inside(terminals, tolerance))
            for i in np.flatnonzero(on):
                members[i].add(facet.index)
        if not any(members):
            continue
        label = f"Sheet gap feed {port.name!r}"
        for other in thin_sheet_structures(scene):
            if other is structure:
                continue
            for facet in sheet_surface(other.geometry).facets:
                on = ((np.abs(_plane_value(terminals, facet)) <= tolerance)
                      & facet.inside(terminals, tolerance))
                if any(on[i] and members[1 - i] for i in range(2)):
                    raise ValueError(f"{label} has terminals on different sheets; both must lie on the same sheet.")
        if not all(members):
            # An ordinary off-sheet port remains the runtime's responsibility.
            if any(np.all(np.abs(_plane_value(terminals, f)) <= tolerance) for f in surface.facets):
                raise ValueError(f"{label}: the sheet outline does not cover the gap terminals.")
            continue
        common = members[0] & members[1]
        if not common:
            raise ValueError(f"{label}: a fold crosses the gap; the gap and one-cell margins must lie on one planar facet.")
        facet = surface.facets[min(common)]
        delta = terminals[1] - terminals[0]
        length = float(np.linalg.norm(delta))
        direction = delta - (delta @ facet.normal) * facet.normal
        direction /= np.linalg.norm(direction)
        transverse = np.cross(facet.normal, direction)
        transverse /= np.linalg.norm(transverse)
        midpoint = terminals.mean(0)
        midpoint -= float(_plane_value(midpoint[None], facet)[0]) * facet.normal
        if not facet.inside(midpoint[None], tolerance)[0]:
            raise ValueError(f"{label}: the sheet outline does not cover the gap midpoint.")
        basis = np.stack((direction, transverse))
        triangles = (facet.origin + facet.triangles @ facet.basis - midpoint) @ basis.T
        boundary = np.concatenate((facet.rim, facet.folds))
        boundary = (facet.origin + boundary @ facet.basis - midpoint) @ basis.T
        crossings = []
        for start, end in boundary:
            if abs(start[0]) <= tolerance:
                crossings.append(start[1])
            if start[0] * end[0] < 0:
                crossings.append(start[1] - start[0] * (end[1] - start[1]) / (end[0] - start[0]))
        # A terminal line on a transverse strip edge has a one-sided width.
        lower_w = max((y for y in crossings if y < -tolerance), default=0.0)
        upper_w = min((y for y in crossings if y > tolerance), default=0.0)
        width = float(upper_w - lower_w)

        def cell_size(point, tangent):
            sizes = np.array([p[np.clip(np.searchsorted(n, point[a], side="right") - 1, 0, len(p) - 1)]
                              for a, (n, p) in enumerate(zip(nodes, primal))])
            return float(1 / np.linalg.norm(tangent / sizes))

        cell = cell_size(midpoint, direction)
        if length + tolerance < cell:
            raise ValueError(f"{label}: g={length:g} m is less than one chord (cell size {cell:g} m along d); "
                             "refine the grid or lengthen the gap.")
        lower = np.array((-length / 2, lower_w))
        upper = np.array((length / 2, upper_w))

        def covered(lo, hi):
            area = sum(_polygon_area(_clip_polygon_box(t, lo, hi)) for t in triangles)
            return area >= np.prod(hi - lo) - tolerance * (hi - lo).sum() * 2

        # Fold contact is forbidden over the full width, including the margins.
        margin_lower = lower.copy(); margin_upper = upper.copy()
        margin_lower[0] -= cell_size(terminals[0] - tolerance * direction, direction)
        margin_upper[0] += cell_size(terminals[1] + tolerance * direction, direction)
        folds = (facet.origin + facet.folds @ facet.basis - midpoint) @ basis.T
        for segment in folds:
            if len(_clip_polygon_box(segment, margin_lower - tolerance, margin_upper + tolerance)):
                raise ValueError(f"{label}: a fold crosses the gap or its one-cell margin; "
                                 "move the feed onto one planar facet or refine the grid.")
        if not covered(lower, upper):
            raise ValueError(f"{label}: the sheet outline does not cover the whole gap.")
        for name, lo, hi in (("negative", margin_lower, np.array((lower[0], upper_w))),
                             ("positive", np.array((upper[0], lower_w)), margin_upper)):
            if not covered(lo, hi):
                raise ValueError(f"{label}: {name} terminal is at the sheet rim; "
                                 "the conductor must continue at least one cell beyond that terminal along d.")
        transverse_cell = cell_size(midpoint, transverse)
        if width + tolerance < 2 * transverse_cell:
            warnings.warn(f"{label}: w={width:g} m is resolved by fewer than 2 cells "
                          f"(transverse cell size {transverse_cell:g} m); refine the grid.", UserWarning, stacklevel=3)
        gaps.append((port.name, facet.index, midpoint, basis, length, width, lower, upper))
    return gaps


def _compile_sheet_feeds(c, geometry, gaps, nodes, tolerance):
    """Record midpoint-owned gap chords without changing the copper complex."""
    count = int(np.sum(geometry.chord_face[:, 0] >= 0))
    paths = list(geometry.chord_path)
    lengths = list(geometry.chord_length)
    widths = list(geometry.chord_width)
    volumes = list(geometry.chord_volume)
    indices = list(range(len(paths)))
    faces = list(geometry.chord_face)
    rows = list(geometry.chord_rows)
    orientations = list(geometry.chord_orientation)
    traces = list(geometry.chord_trace)
    dofs = [([-1, -1] if t < count else geometry.trace_dofs[t - count]) for t in traces]
    promoted = set(traces)
    for offset, pair in enumerate(geometry.trace_dofs):
        trace = count + offset
        if trace in promoted:
            continue
        dof = pair[0]
        axis = geometry.e_component[dof]
        length = geometry.trace_length[trace]
        delta = np.eye(3)[axis] * length / 2
        paths.append(np.stack((geometry.e_position[dof] - delta, geometry.e_position[dof] + delta)))
        lengths.append(length); widths.append(geometry.trace_width[trace])
        volumes.append(geometry.e_volume[pair]); indices.append(-1); traces.append(trace); dofs.append(pair)
        faces.append([-1] * 4); rows.append([-1, -1]); orientations.append([0, 0])
    feeds = []
    for name, facet, midpoint, basis, length, width, lower, upper in gaps:
        selected = []
        axis = int(np.argmax(np.abs(basis[0])))
        for i, path in enumerate(paths):
            local = (path.mean(0) - midpoint) @ basis.T
            on = np.abs(_plane_value(path, c.surface.facets[facet])).max() <= tolerance
            tangential = abs((path[-1] - path[0]) @ basis[0]) > tolerance
            # A one-cell gap can end exactly at two driven chord midpoints.
            # Own the lower Cartesian midpoint only, independent of port polarity.
            along = basis[0, axis] * local[0]
            inside_d = (-length / 2 - tolerance <= along < length / 2 - tolerance if tangential
                        else lower[0] + tolerance < local[0] < upper[0] - tolerance)
            if (on and inside_d
                    and lower[1] - tolerance <= local[1] <= upper[1] + tolerance):
                selected.append(i)
        selected = np.array(selected, dtype=np.int64)
        chosen_paths = [paths[i] for i in selected]
        projection = np.array([(path[-1] - path[0]) @ basis[0] / lengths[i]
                               for i, path in zip(selected, chosen_paths)])
        # Positive-area intersections give the geometric gap cells, including
        # both incident cells when the sheet coincides with a grid face.
        rectangle = midpoint + np.array((lower, (upper[0], lower[1]), upper, (lower[0], upper[1]))) @ basis
        ranges = [range(max(0, int(np.searchsorted(n, rectangle[:, a].min() - tolerance)) - 1),
                        min(len(n) - 1, int(np.searchsorted(n, rectangle[:, a].max() + tolerance))))
                  for a, n in enumerate(nodes)]
        cells = []
        for cell in np.ndindex(*(len(r) for r in ranges)):
            cell = tuple(ranges[a][cell[a]] for a in range(3))
            lo = np.array([n[cell[a]] for a, n in enumerate(nodes)])
            hi = np.array([n[cell[a] + 1] for a, n in enumerate(nodes)])
            polygon = _clip_polygon_box(rectangle, lo, hi)
            if _polygon_area((polygon - midpoint) @ basis.T) > tolerance * np.linalg.norm(hi - lo):
                cells.append(cell)
        feeds.append(SheetGapFeed(
            port_name=name, facet=facet, direction=basis[0], midpoint=midpoint, length=length, width=width,
            cut_cells=np.array(cells, dtype=np.int64).reshape(-1, 3),
            chord_index=(np.array(indices, dtype=np.int64)[selected] if not c.is_pec
                         else np.full(len(selected), -1, dtype=np.int64)),
            chord_face=np.array(faces, dtype=np.int64).reshape(-1, 4)[selected],
            chord_rows=np.array(rows, dtype=np.int64).reshape(-1, 2)[selected],
            chord_orientation=np.array(orientations, dtype=np.int64).reshape(-1, 2)[selected],
            trace_index=np.array(traces, dtype=np.int64)[selected],
            trace_dofs=np.array(dofs, dtype=np.int64).reshape(-1, 2)[selected],
            chord_path=chosen_paths, chord_length=np.array(lengths)[selected], chord_width=np.array(widths)[selected],
            chord_volume=np.array(volumes).reshape(-1, 2)[selected], chord_projection=projection,
            open_chords=np.abs(projection) < 0.5))
    c.feeds = tuple(feeds)


def _compile_chords(c, nodes, primal, face_kind, h_id, edge_id, active_trace,
                    faces, paths, orientations, distances, x_rows, x_cols, x_vals):
    """Build the independent chord circulation and closed sheet-face loops."""
    h = max(float(p.max()) for p in primal)
    count = len(faces)
    trace_ids = list(range(count))
    lengths = list(c.trace_length[:count])
    widths = list(c.trace_width[:count])
    subrows = [[h_id[(face, side)] for side in (-1, 1)] for face in faces]
    volumes = [CHORD_MASS_FACTOR * length * width * np.maximum(distance, h / 16)
               for length, width, distance in zip(lengths, widths, distances)]
    fixed = [((face[1 + face[0]] == 0 and face_kind[(face[0], "low")] == "pec")
              or (face[1 + face[0]] == len(nodes[face[0]]) - 1
                  and face_kind[(face[0], "high")] == "pec")) for face in faces]
    cells = {}
    for chord, face in enumerate(faces):
        a, *g = face
        for shift in (0, -1):
            cell = np.array(g)
            cell[a] += shift
            if all(0 <= cell[ax] < len(primal[ax]) for ax in range(3)):
                cells.setdefault(tuple(cell), []).append(chord)

    # Exact on-sheet edges can close a cut-cell loop. Unused pairs remain on
    # the current explicit foil path; fully aligned sheets have no cut cells.
    on_sheet = {}
    for t, edge in enumerate(sorted(active_trace), start=count):
        a, *g = edge
        _, b, d = _FRAME[a]
        start = np.array([nodes[ax][g[ax]] for ax in range(3)])
        end = start.copy(); end[a] = nodes[a][g[a] + 1]
        on_sheet[t] = (edge, np.stack((start, end)))
        for sb in (0, -1):
            for sd in (0, -1):
                cell = np.array(g); cell[b] += sb; cell[d] += sd
                if tuple(cell) in cells:
                    cells[tuple(cell)].append(t)

    loops = []
    for cell, ids in cells.items():
        boundary = {i: paths[i] if i < count else on_sheet[i][1] for i in ids}
        vertices = {}
        for i, path in boundary.items():
            for point in (path[0], path[-1]):
                vertices.setdefault(tuple(point), []).append(i)
        # Open rim cells have no normal sheet flux.
        if len(ids) < 3 or any(len(neighbours) != 2 for neighbours in vertices.values()):
            continue
        ordered, signs, polygon = [], [], []
        current = ids[0]
        point = tuple(boundary[current][0])
        while current not in ordered:
            path = boundary[current]
            sign = 1 if point == tuple(path[0]) else -1
            oriented = path[::sign]
            ordered.append(current); signs.append(sign); polygon.extend(oriented[:-1])
            point = tuple(oriented[-1])
            current = next(i for i in vertices[point] if i != current)
        if len(ordered) != len(ids):
            continue
        polygon = np.array(polygon)
        center = polygon.mean(0)
        facet = int(c.surface.classify(center[None])[1][0])
        normal = c.surface.facets[facet].normal
        area_vector = np.cross(polygon - center, np.roll(polygon, -1, axis=0) - center).sum(0)
        if area_vector @ normal < 0:
            ordered = ordered[::-1]
            signs = [-s for s in signs[::-1]]
            polygon = polygon[::-1]
        loops.append((cell, ordered, signs, polygon))

    promoted = sorted({i for _, ids, _, _ in loops for i in ids if i >= count})
    chord_id = {i: i for i in range(count)}
    promoted_dofs = {}
    for t in promoted:
        edge, path = on_sheet[t]
        chord = len(paths)
        chord_id[t] = chord
        dofs = [edge_id[(edge, side)] for side in (-1, 1)]
        volume = c.e_volume[dofs]
        length, width = c.trace_length[t], c.trace_width[t]
        faces.append((-1, -1, -1, -1)); paths.append(path)
        subrows.append([-1, -1]); orientations.append([0, 0])
        trace_ids.append(t); lengths.append(length); widths.append(width)
        distances.append(volume / (length * width)); volumes.append(volume)
        fixed.append(bool(c.e_fixed[dofs].all()))
        for si, dof in enumerate(dofs):
            promoted_dofs[dof] = 2 * chord + si

    c.chord_face = np.array(faces, dtype=np.int64).reshape(-1, 4)
    c.chord_trace = np.array(trace_ids, dtype=np.int64)
    c.chord_path = paths
    c.chord_rows = np.array(subrows, dtype=np.int64).reshape(-1, 2)
    c.chord_orientation = np.array(orientations, dtype=np.int64).reshape(-1, 2)
    c.chord_length = np.array(lengths)
    c.chord_width = np.array(widths)
    tensors = []
    for path, length in zip(paths, lengths):
        delta = np.diff(path, axis=0)
        piece_length = np.linalg.norm(delta, axis=1)
        tensors.append(np.einsum("ni,nj,n->ij", delta, delta, 1 / piece_length) / length)
    c.chord_tensor = np.array(tensors).reshape(-1, 3, 3)
    c.chord_distance = np.array(distances).reshape(-1, 2)
    c.chord_volume = np.array(volumes).reshape(-1, 2)
    c.chord_fixed = np.array(fixed, dtype=bool)

    cr, cc, cv = [], [], []
    for chord in range(count):
        for si in range(2):
            cr.append(subrows[chord][si]); cc.append(2 * chord + si)
            cv.append(orientations[chord][si] * lengths[chord])
    xr, xc, xv = [], [], []
    for row, col, value in zip(x_rows, x_cols, x_vals):
        if col in promoted_dofs:
            cr.append(row); cc.append(promoted_dofs[col]); cv.append(value)
        else:
            xr.append(row); xc.append(col); xv.append(value)
    c.k_x_rows = np.array(xr, dtype=np.int64)
    c.k_x_cols = np.array(xc, dtype=np.int64)
    c.k_x_vals = np.array(xv)

    c.sheet_face_cell = np.repeat(np.array([cell for cell, _, _, _ in loops], dtype=np.int64).reshape(-1, 3), 2, axis=0)
    c.sheet_face_side = np.tile([-1, 1], len(loops))
    c.sheet_face_chords, c.sheet_face_orientations, c.sheet_face_traces = [], [], []
    c.sheet_face_vertices = []
    areas, duals, face_volumes = [], [], []
    for p, (cell, ids, signs, polygon) in enumerate(loops):
        chords = np.array([chord_id[i] for i in ids], dtype=np.int64)
        center = polygon.mean(0)
        fan = 0.5 * np.cross(polygon - center, np.roll(polygon, -1, axis=0) - center)
        area = np.linalg.norm(fan, axis=1).sum()
        pair_dual = c.chord_distance[chords].mean(0).sum()
        lower = np.array([nodes[a][cell[a]] for a in range(3)])
        upper = np.array([nodes[a][cell[a] + 1] for a in range(3)])
        total = np.prod(upper - lower)
        if c.surface.planar:
            facet = c.surface.facets[0]
            below = _halfspace_volume(lower, upper, facet.normal, facet.offset)
        else:
            below = c.surface.box_below_volume(lower, upper, 1e-7 * h)
        minus = np.clip(below, 0.25 * total, 0.75 * total)
        for si, volume in enumerate((minus, total - minus)):
            c.sheet_face_chords.append(chords)
            c.sheet_face_orientations.append(np.array(signs, dtype=np.int64))
            c.sheet_face_traces.append(np.array([i for i in ids if i >= count], dtype=np.int64))
            c.sheet_face_vertices.append(polygon)
            areas.append(area)
            duals.append(pair_dual * volume / total)
            face_volumes.append(volume)
            for chord, sign in zip(chords, signs):
                cr.append(len(c.h_area) + 2 * p + si); cc.append(2 * chord + si)
                cv.append(sign * c.chord_length[chord])
    c.sheet_face_area = np.array(areas)
    c.sheet_face_area_eff = np.maximum(c.sheet_face_area, 0.25 * h * h)
    c.sheet_face_dual = np.array(duals)
    c.sheet_face_volume = np.array(face_volumes)
    c.k_c_rows = np.array(cr, dtype=np.int64)
    c.k_c_cols = np.array(cc, dtype=np.int64)
    c.k_c_vals = np.array(cv)


def compile_thin_sheet(scene, structure, *, stiffness_ratio=2.0, minimum_area=0.02, margin=3):
    return _compile_thin_sheet(scene, structure, stiffness_ratio=stiffness_ratio,
                               minimum_area=minimum_area, margin=margin)[0]


def _compile_thin_sheet(scene, structure, *, stiffness_ratio=2.0, minimum_area=0.02, margin=3):
    """Return the complex and any material-independent PEC feed chord geometry."""
    material = structure.material
    is_pec = bool(getattr(material, "is_pec", False))
    if not is_pec and not isinstance(material, CopperFoil):
        raise NotImplementedError("Conformal thin sheets support PEC or CopperFoil materials.")
    surface = sheet_surface(structure.geometry)
    nodes, primal = _grid(scene)
    dual = [_dual_extent(p, np.arange(len(p) + 1)) for p in primal]
    shape = tuple(len(value) for value in nodes)
    spacing = min(float(p.min()) for p in primal)
    h_local = max(float(p.max()) for p in primal)
    zero_tol = 1e-9 * spacing
    region_tol = 1e-7 * spacing
    gaps = _classify_sheet_gaps(scene, structure, surface, nodes, primal, zero_tol)
    collect_chords = not is_pec or bool(gaps)
    lows, highs = surface.vertices.min(0), surface.vertices.max(0)
    lo, hi = [], []
    for axis in range(3):
        lo.append(max(0, int(np.searchsorted(nodes[axis], lows[axis], side="right")) - 1 - margin))
        hi.append(min(shape[axis] - 1, int(np.searchsorted(nodes[axis], highs[axis], side="left")) + margin))
    box_shape = tuple(hi[a] - lo[a] + 1 for a in range(3))
    coords = np.meshgrid(*(nodes[a][lo[a]:hi[a] + 1] for a in range(3)), indexing="ij")
    node_points = np.stack([value.reshape(-1) for value in coords], -1)
    phi, node_facet = surface.classify(node_points)
    phi = phi.reshape(box_shape)
    node_facet = node_facet.reshape(box_shape)
    sign = np.where(np.abs(phi) <= zero_tol, 0, np.sign(phi)).astype(np.int8)

    face_kind = {}
    for axis in range(3):
        kinds = scene.boundary._face_kinds[axis] if hasattr(scene.boundary, "_face_kinds") else None
        for side_index, side in enumerate(("low", "high")):
            value = kinds[side_index] if kinds is not None else scene.boundary.axis_kind(_AXES[axis])
            face_kind[(axis, side)] = str(getattr(value, "value", value)).lower()

    def below_volume(lower, upper):
        """Exact volume of a dual box on the sheet's -1 side."""
        if surface.planar:
            f = surface.facets[0]
            return _halfspace_volume(lower, upper, f.normal, f.offset)
        return surface.box_below_volume(lower, upper, region_tol)

    # ---- E unknowns: one native slot per box edge --------------------------
    edge_info = {}
    for a, b, c in _FRAME:
        ranges = [range(box_shape[0]), range(box_shape[1]), range(box_shape[2])]
        ranges[a] = range(box_shape[a] - 1)
        grid = np.stack(np.meshgrid(*(np.arange(len(r)) for r in ranges), indexing="ij"), -1).reshape(-1, 3)
        step = np.eye(3, dtype=np.int64)[a]
        s0 = sign[tuple(grid.T)]
        s1 = sign[tuple((grid + step).T)]
        p0 = phi[tuple(grid.T)]
        p1 = phi[tuple((grid + step).T)]
        glob = grid + np.array(lo)
        length = primal[a][glob[:, a]]
        ext = dual[b][glob[:, b]] * dual[c][glob[:, c]]
        pierced = s0 * s1 < 0
        onsheet = (s0 == 0) & (s1 == 0)
        sides = np.where(pierced | onsheet, 0, np.sign(s0 + s1))
        parameters = [None] * len(grid)
        if surface.planar:
            for index, value in zip(np.flatnonzero(pierced), p0[pierced] / (p0[pierced] - p1[pierced])):
                parameters[index] = value
        else:
            for index in np.flatnonzero(pierced):
                start = np.array([nodes[ax][glob[index, ax]] for ax in range(3)])
                end = start.copy()
                end[a] = nodes[a][glob[index, a] + 1]
                parameters[index] = surface.pierce(start, end, zero_tol)
        keys = [(a, *g) for g in glob.tolist()]
        edge_info.update(zip(keys, (dict(pierced=p, onsheet=o, side=s, t=t, s0=start, length=l, ext=e)
                                   for p, o, s, t, start, l, e in zip(
                                       pierced.tolist(), onsheet.tolist(), sides.tolist(), parameters,
                                       s0.tolist(), length.tolist(), ext.tolist()))))
        if not surface.planar:
            starts = np.stack([nodes[ax][glob[:, ax]] for ax in range(3)], -1)
            ends = starts.copy()
            ends[:, a] = nodes[a][glob[:, a] + 1]
            count = surface.crossings(starts, ends, region_tol)
            pierced_all = (s0.astype(np.int64) * s1.astype(np.int64)) < 0
            if np.any(count > 1) or np.any((count == 1) & ~pierced_all & (s0 != 0) & (s1 != 0)):
                raise NotImplementedError("A Yee edge crosses the faceted thin sheet more than once; refine the grid.")
    # Faces --------------------------------------------------------------
    face_info = {}
    for a, b, c in _FRAME:
        ranges = [box_shape[0], box_shape[1], box_shape[2]]
        ranges[b] -= 1
        ranges[c] -= 1
        grid = np.stack(np.meshgrid(*(np.arange(r) for r in ranges), indexing="ij"), -1).reshape(-1, 3)
        eb, ec = np.eye(3, dtype=np.int64)[b], np.eye(3, dtype=np.int64)[c]
        corners = [grid, grid + eb, grid + eb + ec, grid + ec]
        signs = np.stack([sign[tuple(corner.T)] for corner in corners], -1)
        glob = grid + np.array(lo)
        split = (signs > 0).any(-1) & (signs < 0).any(-1)
        onsheet = (signs == 0).all(-1)
        for g, gb, gc, sg, cut, on in zip(glob.tolist(), (glob + eb).tolist(), (glob + ec).tolist(),
                                         signs.tolist(), split.tolist(), onsheet.tolist()):
            # CCW boundary edges: (edge key, orientation)
            edges = [((b, *g), 1), ((c, *gb), 1), ((b, *gc), -1), ((c, *g), -1)]
            face_info[(a, *g)] = dict(signs=tuple(sg), edges=edges, split=cut, onsheet=on)

    # ---- activity --------------------------------------------------------
    def corner_point(a, g, m):
        _, b, c = _FRAME[a]
        offset = np.zeros(3, dtype=np.int64)
        if m in (1, 2):
            offset[b] = 1
        if m in (2, 3):
            offset[c] = 1
        point = np.array(g) + offset
        return np.array([nodes[ax][point[ax]] for ax in range(3)])

    def edge_points(key):
        a, i, j, k = key
        start = np.array([nodes[0][i], nodes[1][j], nodes[2][k]])
        end = start.copy()
        end[a] = nodes[a][(i, j, k)[a] + 1]
        return start, end

    def face_cycle(key):
        """CCW cycle of (label, point, edge_slot) with label in {-1,0,+1} or 'z' for zero points."""
        a = key[0]
        g = key[1:]
        info = face_info[key]
        cycle = []
        for m in range(4):
            label = info["signs"][m]
            point = corner_point(a, g, m)
            cycle.append(("z" if label == 0 else label, point, None))
            edge, orientation = info["edges"][m]
            einfo = edge_info[edge]
            if einfo["pierced"]:
                start, end = edge_points(edge)
                p = start + einfo["t"] * (end - start)
                cycle.append(("z", p, None))
        return cycle

    def cut_path(key, zeros):
        """Cut polyline of a split face and the facet of each piece.

        A face crossed by one facet is cut by the chord of its two zero points.
        Where a fold line pierces the face, the cut runs through the fold point,
        one piece on each facet.
        """
        if surface.planar:
            return [zeros[0], zeros[1]], [0]
        first, second = surface.members(np.stack(zeros), region_tol)
        common = first & second
        if common:
            return [zeros[0], zeros[1]], [min(common)]
        a = key[0]
        _, b, c = _FRAME[a]
        g = key[1:]
        for fa in sorted(first):
            for fb in sorted(second):
                point = surface.fold_point(fa, fb, a, nodes[a][g[a]])
                if point is None:
                    continue
                inside = all(nodes[ax][g[ax]] - region_tol <= point[ax] <= nodes[ax][g[ax] + 1] + region_tol
                             for ax in (b, c))
                if inside:
                    return [zeros[0], point, zeros[1]], [fa, fb]
        raise NotImplementedError("A Yee face is cut by two thin-sheet facets that do not meet inside it; refine the "
                                  "grid or widen the facets.")

    active_face = {}
    for key, info in face_info.items():
        if not info["split"]:
            continue
        zeros = [p for label, p, _ in face_cycle(key) if label == "z"]
        if len(zeros) != 2:
            raise NotImplementedError("A Yee face is cut more than once by the thin sheet; refine the grid.")
        path, path_facets = cut_path(key, zeros)
        if surface.inside(np.stack((path[0], path[-1])), region_tol).all():
            active_face[key] = (path, path_facets)
    active_trace = set()
    for key, info in edge_info.items():
        if info["onsheet"]:
            start, end = edge_points(key)
            if surface.inside(np.stack((start, end)), region_tol).all():
                active_trace.add(key)
    split_edges = set()
    for key in active_face:
        for edge, _ in face_info[key]["edges"]:
            if edge_info[edge]["pierced"]:
                split_edges.add(edge)

    def on_boundary(edge):
        a = edge[0]
        index = edge[1:]
        result = []
        for axis in range(3):
            if axis == a:
                continue
            if index[axis] == 0:
                result.append(face_kind[(axis, "low")])
            elif index[axis] == shape[axis] - 1:
                result.append(face_kind[(axis, "high")])
        return result

    # C-order parent edges, with each extra side inserted immediately after
    # its native slot, exactly as in the face circulation dictionaries.
    edge_keys = list(edge_info)
    edge_array = np.array(edge_keys, dtype=np.int64)
    components, indices = edge_array[:, 0], edge_array[:, 1:]
    special = np.array([key in split_edges or key in active_trace for key in edge_keys])
    extra_index = np.cumsum(special) - 1
    first = np.arange(len(edge_keys)) + np.r_[0, np.cumsum(special[:-1])]
    parents = np.repeat(np.arange(len(edge_keys)), 1 + special)
    starts = np.stack([nodes[axis][indices[:, axis]] for axis in range(3)], -1)
    ends = starts.copy()
    fixed_wall = np.zeros(len(edge_keys), dtype=bool)
    for axis in range(3):
        select = components == axis
        ends[select, axis] = nodes[axis][indices[select, axis] + 1]
        for side, coordinate in (("low", 0), ("high", shape[axis] - 1)):
            if face_kind[(axis, side)] != "pmc":
                fixed_wall |= ~select & (indices[:, axis] == coordinate)
    lengths = np.array([info["length"] for info in edge_info.values()])
    extents = np.array([info["ext"] for info in edge_info.values()])
    e_comp = components[parents]
    e_idx = indices[parents]
    e_extra = np.full(len(parents), -1, dtype=np.int64)
    e_side = np.array([info["side"] for info in edge_info.values()], dtype=np.int64)[parents]
    e_len = lengths[parents].tolist()
    e_vol = (lengths * extents)[parents].tolist()
    e_fixed = fixed_wall[parents].tolist()
    e_pos = (0.5 * (starts + ends))[parents]
    e_split_arr = special[parents]
    id_side = np.zeros(len(parents), dtype=np.int64)
    id_side[first[special]] = -1
    id_side[first[special] + 1] = 1
    edge_id = {(edge_keys[parent], side): dof for dof, (parent, side)
               in enumerate(zip(parents.tolist(), id_side.tolist()))}
    extra_e = int(special.sum())
    for native in np.flatnonzero(special):
        key = edge_keys[native]
        info = edge_info[key]
        a, i, j, k = key
        start, end = starts[native], ends[native]
        kinds = on_boundary(key)
        wall = bool(fixed_wall[native])
        if wall and any(kind not in ("pec", "pmc") for kind in kinds):
            raise NotImplementedError("Thin sheet cut edges may touch PEC or PMC domain walls only.")
        dof = first[native]
        e_extra[dof + 1] = extra_index[native]
        e_side[dof:dof + 2] = (-1, 1)
        ext = info["ext"]
        if key in split_edges:
            t = info["t"]
            first_side = info["s0"]
            pierce = start + t * (end - start)
            pieces = {first_side: (t * info["length"], 0.5 * (start + pierce)),
                      -first_side: ((1 - t) * info["length"], 0.5 * (pierce + end))}
            for side, offset in ((-1, 0), (1, 1)):
                piece_len, piece_mid = pieces[side]
                e_len[dof + offset] = piece_len
                e_vol[dof + offset] = piece_len * ext
                e_pos[dof + offset] = piece_mid
        else:
            _, b, c = _FRAME[a]
            index = (i, j, k)
            lower = start.copy()
            upper = end.copy()
            for axis in (b, c):
                n_index = index[axis]
                lower[axis] = nodes[axis][n_index] - (0.5 * primal[axis][n_index - 1] if n_index > 0 else 0.0)
                upper[axis] = nodes[axis][n_index] + (0.5 * primal[axis][n_index] if n_index < len(primal[axis]) else 0.0)
            below = below_volume(lower, upper)
            total = float(np.prod(upper - lower))
            volumes = {-1: below, 1: total - below}
            for side, offset in ((-1, 0), (1, 1)):
                e_vol[dof + offset] = volumes[side]
                e_fixed[dof + offset] = wall or is_pec
    e_modified = np.zeros(len(e_comp), dtype=bool)
    e_phi, e_facet = surface.classify(e_pos)

    # Exact per-component electric volume partition near the sheet. Every Yee
    # dual box is clipped by the plane; a pierced edge's segments keep their
    # own side's part, a regular edge keeps its own side's part, and the part
    # beyond the sheet (orphan) is given to the nearest same-component unknown
    # on that side. Each side is therefore covered exactly up to the sheet,
    # which removes the O(h) capacitive position bias of unclipped boxes.
    lowers, uppers = starts.copy(), ends.copy()
    for axis in range(3):
        select = components != axis
        index = indices[select, axis]
        p = primal[axis]
        lowers[select, axis] = nodes[axis][index] - np.where(index > 0, 0.5 * p[np.maximum(index - 1, 0)], 0.0)
        uppers[select, axis] = nodes[axis][index] + np.where(index < len(p), 0.5 * p[np.minimum(index, len(p) - 1)], 0.0)

    orphans = []
    for native, (key, info) in enumerate(edge_info.items()):
        if key in active_trace:
            continue
        if key in split_edges:
            lower, upper = lowers[native], uppers[native]
            below = below_volume(lower, upper)
            total = float(np.prod(upper - lower))
            # Volume-conserving split: each segment keeps at least a quarter of
            # its prism volume (bounds its inverse mass); the total stays the
            # parent box volume.
            minus, plus = edge_id[(key, -1)], edge_id[(key, 1)]
            floor_minus, floor_plus = 0.25 * e_vol[minus], 0.25 * e_vol[plus]
            v_minus = min(max(below, floor_minus), total - floor_plus)
            e_vol[minus], e_vol[plus] = v_minus, total - v_minus
            e_modified[minus] = e_modified[plus] = True
            continue
        dof = edge_id[(key, 0)]
        side = int(e_side[dof])
        if side == 0 or e_fixed[dof]:
            # Ambiguous rim edges and held PEC-wall edges keep their volume and
            # donate nothing.
            continue
        lower, upper = lowers[native], uppers[native]
        center = 0.5 * (lower + upper)
        if surface.planar:
            f = surface.facets[0]
            distance = float(center @ f.normal - f.offset)
            reach = 0.5 * float(np.abs(f.normal) @ (upper - lower))
            if abs(distance) >= reach:
                continue
        elif abs(float(e_phi[dof])) >= 0.5 * float(np.linalg.norm(upper - lower)):
            continue
        corners = np.array([[lower[0] if m & 1 == 0 else upper[0], lower[1] if m & 2 == 0 else upper[1],
                             lower[2] if m & 4 == 0 else upper[2]] for m in range(8)])
        inside = surface.inside(surface.project(corners), region_tol)
        if not inside.any():
            continue
        if not inside.all():
            # A dual box straddling the sheet outline is left unpartitioned:
            # beyond the rim the field is single-valued (documented O(h) rim).
            continue
        below = below_volume(lower, upper)
        total = float(np.prod(upper - lower))
        keep = below if side < 0 else total - below
        orphan = total - keep
        if orphan <= 1e-12 * total:
            continue
        e_vol[dof] = keep
        e_modified[dof] = True
        orphans.append((int(e_comp[dof]), -side, center, orphan))
    if orphans:
        from scipy.spatial import cKDTree
        receivers = {}
        for comp in range(3):
            for side in (-1, 1):
                ids = np.flatnonzero((e_side == side) & (e_comp == comp) & ~np.array(e_fixed, dtype=bool))
                receivers[(comp, side)] = (ids, cKDTree(e_pos[ids]) if len(ids) else None)
        for comp, side, center, orphan in orphans:
            ids, tree = receivers[(comp, side)]
            if tree is None:
                continue
            target = int(ids[tree.query(center)[1]])
            if not surface.planar:
                _, facet = surface.classify(center[None])
                if surface.crossings(center[None], e_pos[target][None], region_tol,
                                     exclude=surface.neighbourhood(int(facet[0])))[0]:
                    raise NotImplementedError("A thin-sheet mass partition reaches across another part of the faceted "
                                              "sheet; refine the grid or keep sheet parts further apart.")
            e_vol[target] += orphan
            e_modified[target] = True

    def edge_dof(edge, side):
        if (edge, side) in edge_id:
            return edge_id[(edge, side)]
        return edge_id[(edge, 0)]

    def e_len_value(dof):
        return e_len[dof]

    # ---- H unknowns and circulation rows ---------------------------------
    h_comp, h_idx, h_extra, h_area, h_true, h_dual, h_special = ([] for _ in range(7))
    rows, cols, vals = [], [], []
    trace_dofs, trace_len, trace_width = [], [], []
    x_rows, x_cols, x_vals = [], [], []
    chord_faces, chord_paths, chord_orientations, chord_distances = [], [], [], []
    extra_h = 0
    h_id = {}

    def add_face(key, extra, area_eff, area_true, special):
        a = key[0]
        h_id_value = len(h_comp)
        h_comp.append(a); h_idx.append(key[1:]); h_extra.append(extra)
        h_area.append(area_eff); h_true.append(area_true)
        h_dual.append(float(dual[a][key[1 + a]]))
        h_special.append(special)
        return h_id_value

    split_or_trace = split_edges | active_trace
    for key, info in face_info.items():
        a = key[0]
        _, b, c = _FRAME[a]
        g = key[1:]
        full_area = primal[b][g[b]] * primal[c][g[c]]
        if key in active_face:
            cycle = face_cycle(key)
            path, path_facets = active_face[key]
            orientations, distances = [], []
            for side, extra in ((-1, -1), (1, extra_h)):
                polygon = [(label, p) for label, p, _ in cycle if label in ("z", side)]
                # The cut closes the sub-face between its two consecutive zero
                # points, through the fold point when a fold pierces the face.
                count = len(polygon)
                for m in range(count):
                    if polygon[m][0] == "z" and polygon[(m + 1) % count][0] == "z":
                        start, end, gap = polygon[m][1], polygon[(m + 1) % count][1], m
                ordered = path if np.linalg.norm(path[0] - start) <= region_tol else path[::-1]
                pieces = [(ordered[i], ordered[i + 1]) for i in range(len(ordered) - 1)]
                polygon = polygon[:gap + 1] + [("z", p) for p in ordered[1:-1]] + polygon[gap + 1:]
                points2 = np.array([[p[b], p[c]] for _, p in polygon])
                area = 0.5 * abs(np.sum(points2[:, 0] * np.roll(points2[:, 1], -1)
                                        - np.roll(points2[:, 0], -1) * points2[:, 1]))
                entries = []
                for edge, orientation in info["edges"]:
                    einfo = edge_info[edge]
                    if einfo["pierced"]:
                        dof = edge_dof(edge, side)
                        entries.append((dof, orientation * e_len_value(dof)))
                    elif einfo["side"] == side:
                        entries.append((edge_dof(edge, 0), orientation * einfo["length"]))
                # Stiffness-matched magnetic mass: the sub-face's own curl-curl
                # diagonal may not exceed stiffness_ratio times its parent Yee
                # face's, which bounds the small-cut spectral growth without a
                # fixed geometric floor. It is measured on the sub-face's own
                # edge pieces only, identically for PEC and copper.
                own = sum(value * value / e_vol[dof] for dof, value in entries if not e_fixed[dof])
                parent = sum(edge_info[edge]["length"] / edge_info[edge]["ext"] for edge, _ in info["edges"])
                area_eff = max(area, full_area * own / (stiffness_ratio * parent), minimum_area * full_area)
                row = add_face(key, extra, area_eff, area, True)
                h_id[(key, side)] = row
                if collect_chords:
                    for dof, value in entries:
                        x_rows.append(row); x_cols.append(dof); x_vals.append(value)
                    orientations.append(1 if ordered is path else -1)
                    # Area centroid of the sub-face polygon, including its fold point.
                    q = points2 - points2[0]
                    cross = q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]
                    centroid = np.array(polygon[0][1], copy=True)
                    centroid[[b, c]] += ((q + np.roll(q, -1, axis=0)) * cross[:, None]).sum(0) / (3 * cross.sum())
                    distances.append(abs(float(surface.classify(centroid[None])[0][0])))
                for dof, value in entries:
                    rows.append(row); cols.append(dof); vals.append(value)
            extra_h += 1
            if collect_chords:
                # One trace per cut face. Across a fold it integrates the trace
                # along both pieces, with the length-weighted strip width, so a
                # vanishing fold reproduces the planar trace.
                pieces = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                lengths = [float(np.linalg.norm(end - start)) for start, end in pieces]
                widths = [float(dual[a][g[a]])
                          * math.sqrt(max(0.0, 1 - surface.facets[facet].normal[a] ** 2)) for facet in path_facets]
                trace_len.append(sum(lengths))
                trace_width.append(widths[0] if len(widths) == 1
                                   else sum(w * l for w, l in zip(widths, lengths)) / sum(lengths))
                chord_faces.append(key)
                chord_paths.append(np.array(path))
                chord_orientations.append(orientations)
                chord_distances.append(distances)
            continue
        touches = [edge in split_or_trace for edge, _ in info["edges"]]
        row = add_face(key, -1, full_area, full_area, bool(any(touches)))
        h_id[(key, 0)] = row
        face_side = int(np.sign(sum(info["signs"])))
        first_entry = len(rows)
        for edge, orientation in info["edges"]:
            einfo = edge_info[edge]
            if edge in split_edges:
                for side in (-1, 1):
                    dof = edge_id[(edge, side)]
                    rows.append(row); cols.append(dof); vals.append(orientation * e_len_value(dof))
            elif edge in active_trace:
                if info["onsheet"]:
                    lower, upper = edge_id[(edge, -1)], edge_id[(edge, 1)]
                    total = e_vol[lower] + e_vol[upper]
                    for dof in (lower, upper):
                        rows.append(row); cols.append(dof)
                        vals.append(orientation * einfo["length"] * e_vol[dof] / total)
                else:
                    if face_side == 0:
                        raise RuntimeError("A face touching a sheet trace has no side.")
                    rows.append(row); cols.append(edge_id[(edge, face_side)]); vals.append(orientation * einfo["length"])
            else:
                rows.append(row); cols.append(edge_id[(edge, 0)]); vals.append(orientation * einfo["length"])
        if collect_chords:
            x_rows.extend(rows[first_entry:]); x_cols.extend(cols[first_entry:]); x_vals.extend(vals[first_entry:])
    # On-sheet traces (axis-aligned or exactly coincident grid lines).
    reusable = None
    if collect_chords:
        for edge in sorted(active_trace):
            a = edge[0]
            index = edge[1:]
            start, end = edge_points(edge)
            # A trace on a fold line represents half a strip on each facet.
            members = sorted(surface.members((0.5 * (start + end))[None], region_tol)[0])
            width = 0.0
            for facet in members:
                normal = surface.facets[facet].normal
                for axis in range(3):
                    if axis == a:
                        continue
                    width += (float(dual[axis][index[axis]])
                              * math.sqrt(max(0.0, 1 - normal[axis] ** 2)) / len(members))
            length = edge_info[edge]["length"]
            trace_dofs.append([edge_id[(edge, side)] for side in (-1, 1)])
            trace_len.append(length)
            trace_width.append(width)

    complex_ = ThinSheetComplex(
        material=material, is_pec=is_pec, name=str(getattr(structure, "name", None) or "thin_sheet"), surface=surface,
        e_component=e_comp, e_index=np.array(e_idx, dtype=np.int64), e_extra=np.array(e_extra, dtype=np.int64),
        e_side=e_side, e_length=np.array(e_len), e_volume=np.array(e_vol), e_fixed=np.array(e_fixed, dtype=bool),
        e_position=e_pos, e_split=e_split_arr, e_modified=e_modified, e_phi=e_phi,
        e_normal=surface.normals(e_facet),
        h_component=np.array(h_comp, dtype=np.int64), h_index=np.array(h_idx, dtype=np.int64),
        h_extra=np.array(h_extra, dtype=np.int64), h_area=np.array(h_area), h_true_area=np.array(h_true),
        h_dual=np.array(h_dual), h_special=np.array(h_special, dtype=bool),
        k_rows=np.array(rows, dtype=np.int64), k_cols=np.array(cols, dtype=np.int64), k_vals=np.array(vals),
        trace_dofs=np.array(trace_dofs if not is_pec else [], dtype=np.int64).reshape(-1, 2),
        trace_length=np.array(trace_len if not is_pec else []), trace_width=np.array(trace_width if not is_pec else []),
        extra_e=extra_e, extra_h=extra_h)
    if collect_chords:
        # PEC has no runtime chord unknowns. Build its feed geometry on a
        # separate record, keeping every existing PEC array unchanged.
        geometry = complex_ if not is_pec else replace(
            complex_, trace_dofs=np.array(trace_dofs, dtype=np.int64).reshape(-1, 2),
            trace_length=np.array(trace_len), trace_width=np.array(trace_width))
        _compile_chords(geometry, nodes, primal, face_kind, h_id, edge_id, active_trace,
                        chord_faces, chord_paths, chord_orientations, chord_distances,
                        x_rows, x_cols, x_vals)
        if gaps:
            _compile_sheet_feeds(complex_, geometry, gaps, nodes, zero_tol)
            if is_pec and not active_trace:
                # Without on-sheet edges, no mass or curl depends on PEC trace
                # holds. Every trace is a cut chord, with the same index.
                geometry.feeds = tuple(replace(feed, chord_index=feed.trace_index.copy())
                                       for feed in complex_.feeds)
                reusable = geometry
    complex_.statistics = dict(
        box_lower=tuple(lo), box_upper=tuple(hi), active_split_faces=len(active_face),
        split_edges=len(split_edges), on_sheet_traces=len(active_trace), traces=len(complex_.trace_length),
        floored_subfaces=int(np.sum(complex_.h_area > complex_.h_true_area * (1 + 1e-12))),
        facets=len(surface.facets), folds=len(surface.adjacency),
        fold_faces=sum(len(path) > 2 for path, _ in active_face.values()),
        stiffness_ratio=float(stiffness_ratio), minimum_area=float(minimum_area))
    if reusable is not None:
        reusable.statistics = complex_.statistics | {"traces": len(reusable.trace_length)}
    return complex_, reusable

