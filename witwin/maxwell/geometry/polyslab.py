"""Maxwell-specific extruded polygon geometries."""

from __future__ import annotations

import numpy as np
import torch

from witwin.core import GeometryBase
from witwin.core.geometry.polygon import polygon_loops_signed_distance_2d


def _as_scalar(value, *, device=None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32).reshape(())
    return torch.tensor(float(value), dtype=torch.float32, device=device)


def _axial_split(dx, dy, dz, axis: str):
    if axis == "z":
        return dz, dx, dy
    if axis == "y":
        return dy, dx, dz
    return dx, dy, dz


def _constant_tensor(data, *, device, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, device=device)


def _faces_tensor(data, *, device) -> torch.Tensor:
    return _constant_tensor(data, device=device, dtype=torch.int64)


def _as_polygon_vertices(value, *, device=None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        vertices = value.to(device=device, dtype=torch.float32)
    else:
        vertices = torch.tensor(
            [[float(u), float(v)] for u, v in value], dtype=torch.float32, device=device
        )
    if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] != 2:
        raise ValueError("vertices must have shape (N, 2) with N >= 3.")
    return vertices


def _as_axis_bounds(value, *, device=None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        bounds = value.to(device=device, dtype=torch.float32)
    else:
        bounds = torch.tensor(
            [float(value[0]), float(value[1])], dtype=torch.float32, device=device
        )
    if bounds.shape != (2,):
        raise ValueError("bounds must contain (min, max) along the extrusion axis.")
    return bounds


def _cross_2d_scalar(origin, first, second) -> float:
    return float(
        (first[0] - origin[0]) * (second[1] - origin[1])
        - (first[1] - origin[1]) * (second[0] - origin[0])
    )


def _point_in_triangle_2d(point, a, b, c, eps=1.0e-12) -> bool:
    """Inclusive point-in-triangle test for a CCW triangle (a, b, c)."""
    return (
        _cross_2d_scalar(a, b, point) >= -eps
        and _cross_2d_scalar(b, c, point) >= -eps
        and _cross_2d_scalar(c, a, point) >= -eps
    )


def _ear_clip_faces_2d(polygon: np.ndarray) -> list[list[int]]:
    """Triangulate a simple CCW (N, 2) polygon into CCW index triangles via ear clipping."""
    remaining = list(range(polygon.shape[0]))
    faces: list[list[int]] = []
    while len(remaining) > 3:
        clipped = False
        for slot in range(len(remaining)):
            i0 = remaining[slot - 1]
            i1 = remaining[slot]
            i2 = remaining[(slot + 1) % len(remaining)]
            a, b, c = polygon[i0], polygon[i1], polygon[i2]
            if _cross_2d_scalar(a, b, c) <= 1.0e-12:
                continue
            others = (polygon[j] for j in remaining if j not in (i0, i1, i2))
            if any(_point_in_triangle_2d(p, a, b, c) for p in others):
                continue
            faces.append([i0, i1, i2])
            remaining.pop(slot)
            clipped = True
            break
        if not clipped:
            raise ValueError(
                "Failed to triangulate polygon; vertices must describe a simple polygon."
            )
    faces.append(list(remaining))
    return faces


def _offset_polygon_2d(polygon: np.ndarray, offset: float) -> np.ndarray:
    """Approximate outward miter offset of a CCW (N, 2) polygon (negative offset shrinks)."""
    if offset == 0.0:
        return polygon
    previous_edges = polygon - np.roll(polygon, 1, axis=0)
    next_edges = np.roll(polygon, -1, axis=0) - polygon

    def _outward_normals(edges: np.ndarray) -> np.ndarray:
        normals = np.stack([edges[:, 1], -edges[:, 0]], axis=1)
        return normals / np.maximum(
            np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12
        )

    normal_previous = _outward_normals(previous_edges)
    normal_next = _outward_normals(next_edges)
    denom = np.maximum(
        1.0 + np.sum(normal_previous * normal_next, axis=1, keepdims=True), 1.0e-3
    )
    return polygon + offset * (normal_previous + normal_next) / denom


class PolySlab(GeometryBase):
    """Extruded 2D polygon with optional linear sidewall taper (Tidy3D semantics).

    ``vertices`` is the (N, 2) cross-section in the plane perpendicular to ``axis``
    (plane coordinates follow ``_axial_split``: axis z -> (x, y), y -> (x, z),
    x -> (y, z)), ``bounds`` is (min, max) along the axis, and a positive
    ``sidewall_angle`` (radians) shrinks the cross-section as the axis coordinate
    increases. ``reference_plane`` selects the axis position where the drawn
    polygon is exact. The interior uses the even-odd rule, so non-convex polygons
    are supported and vertex winding is irrelevant.
    """

    kind = "poly_slab"

    def __init__(
        self,
        vertices,
        bounds,
        axis="z",
        sidewall_angle=0.0,
        reference_plane="middle",
        position=(0, 0, 0),
        rotation=None,
        *,
        device=None,
    ):
        super().__init__(position=position, rotation=rotation, device=device)
        self.vertices: torch.Tensor = _as_polygon_vertices(vertices, device=device)
        self.bounds: torch.Tensor = _as_axis_bounds(bounds, device=device)
        if isinstance(axis, int) and not isinstance(axis, bool):
            if axis not in (0, 1, 2):
                raise ValueError("integer axis must be 0, 1, or 2.")
            axis = ("x", "y", "z")[axis]
        self.axis: str = self._validate_axis(axis)
        self.sidewall_angle: torch.Tensor = _as_scalar(sidewall_angle, device=device)
        if reference_plane not in ("bottom", "middle", "top"):
            raise ValueError("reference_plane must be 'bottom', 'middle', or 'top'.")
        self.reference_plane: str = reference_plane
        self.loop_sizes: tuple[int, ...] = (int(self.vertices.shape[0]),)

    def _reference_position(self, bounds: torch.Tensor) -> torch.Tensor:
        if self.reference_plane == "bottom":
            return bounds[0]
        if self.reference_plane == "top":
            return bounds[1]
        return 0.5 * (bounds[0] + bounds[1])

    def signed_distance(self, xx, yy, zz):
        dx, dy, dz = self._local_coords(xx, yy, zz)
        axial, pu, pv = _axial_split(dx, dy, dz, self.axis)
        vertices = self.vertices.to(device=pu.device, dtype=pu.dtype)
        loops = torch.split(vertices, list(self.loop_sizes), dim=0)
        points = torch.stack([pu, pv], dim=-1).reshape(-1, 2)
        polygon_distance = polygon_loops_signed_distance_2d(points, loops).reshape(
            pu.shape
        )
        bounds = self.bounds.to(device=axial.device, dtype=axial.dtype)
        tan_angle = torch.tan(
            self.sidewall_angle.to(device=axial.device, dtype=axial.dtype)
        )
        # Signed distance minus an offset is exact morphological erosion/dilation, so
        # the taper is a z-dependent inward offset of the cross-section.
        tapered_distance = polygon_distance + tan_angle * (
            axial - self._reference_position(bounds)
        )
        axial_distance = torch.abs(axial - 0.5 * (bounds[0] + bounds[1])) - 0.5 * (
            bounds[1] - bounds[0]
        )
        outside = torch.sqrt(
            torch.clamp(tapered_distance, min=0.0) ** 2
            + torch.clamp(axial_distance, min=0.0) ** 2
        )
        inside = torch.clamp(torch.maximum(tapered_distance, axial_distance), max=0.0)
        return outside + inside

    def to_mesh(self, segments=16):
        del segments
        device = self.device
        polygon = self.vertices.detach().cpu().to(torch.float64).numpy()
        # Drop consecutive duplicate vertices (including a closed-ring wrap-around
        # repeat); coincident points would block every candidate ear below.
        keep = np.linalg.norm(polygon - np.roll(polygon, 1, axis=0), axis=1) > 1.0e-12
        polygon = polygon[keep]
        if polygon.shape[0] < 3:
            raise ValueError("vertices must contain at least 3 distinct points.")
        signed_area = 0.5 * float(
            np.sum(
                polygon[:, 0] * np.roll(polygon[:, 1], -1)
                - np.roll(polygon[:, 0], -1) * polygon[:, 1]
            )
        )
        if signed_area < 0.0:
            polygon = polygon[::-1].copy()
        bounds = self.bounds.detach().cpu().to(torch.float64).numpy()
        reference = {
            "bottom": bounds[0],
            "top": bounds[1],
            "middle": 0.5 * (bounds[0] + bounds[1]),
        }[self.reference_plane]
        tan_angle = float(
            np.tan(self.sidewall_angle.detach().cpu().to(torch.float64).item())
        )
        bottom_ring = _offset_polygon_2d(polygon, -tan_angle * (bounds[0] - reference))
        top_ring = _offset_polygon_2d(polygon, -tan_angle * (bounds[1] - reference))
        count = polygon.shape[0]
        bottom = np.concatenate([bottom_ring, np.full((count, 1), bounds[0])], axis=1)
        top = np.concatenate([top_ring, np.full((count, 1), bounds[1])], axis=1)
        vertices = _constant_tensor(
            np.concatenate([bottom, top], axis=0), device=device
        )
        faces = []
        for index in range(count):
            next_index = (index + 1) % count
            faces.extend(
                [
                    [index, next_index, count + next_index],
                    [index, count + next_index, count + index],
                ]
            )
        for i0, i1, i2 in _ear_clip_faces_2d(polygon):
            faces.append([i0, i2, i1])
            faces.append([count + i0, count + i1, count + i2])
        # Map local (u, v, axial) columns to world axes following _axial_split
        # (axis z -> (x, y), y -> (x, z), x -> (y, z)); _remap_axis_torch would
        # mirror u/v for axis "x" and disagree with signed_distance/to_mask.
        vertices = vertices[
            :, {"z": [0, 1, 2], "y": [0, 2, 1], "x": [2, 0, 1]}[self.axis]
        ]
        faces_tensor = _faces_tensor(faces, device=device)
        if self.axis == "y":
            # Odd column permutation reflects orientation; flip winding to keep
            # face normals outward.
            faces_tensor = torch.flip(faces_tensor, dims=(1,))
        vertices = self._transform_mesh_verts(vertices)
        return vertices, faces_tensor


class ComplexPolySlab(PolySlab):
    """PolySlab variant for non-simple cross-sections.

    Accepts a list of vertex loops; self-intersecting loops (e.g. bowties) and
    multiple loops are allowed. The interior is the even-odd rule over the combined
    crossing count of all loops, so loop winding is irrelevant and extra loops carve
    holes. The distance magnitude is the minimum distance over all edges of all loops.
    """

    kind = "complex_poly_slab"

    def __init__(
        self,
        loops,
        bounds,
        axis="z",
        sidewall_angle=0.0,
        reference_plane="middle",
        position=(0, 0, 0),
        rotation=None,
        *,
        device=None,
    ):
        loop_tensors = [_as_polygon_vertices(loop, device=device) for loop in loops]
        if not loop_tensors:
            raise ValueError("loops must contain at least one polygon loop.")
        super().__init__(
            torch.cat(loop_tensors, dim=0),
            bounds,
            axis=axis,
            sidewall_angle=sidewall_angle,
            reference_plane=reference_plane,
            position=position,
            rotation=rotation,
            device=device,
        )
        self.loop_sizes = tuple(int(loop.shape[0]) for loop in loop_tensors)

    def to_mesh(self, segments=16):
        raise NotImplementedError("ComplexPolySlab does not support mesh export.")
