"""Explicit, fixed-topology planar PEC supports with SI float64 coordinates."""

from __future__ import annotations

import math
import numpy as np
import torch

from witwin.core import GeometryBase
from witwin.core.geometry.polygon import polygon_loops_signed_distance_2d


class PlanarPEC(GeometryBase):
    """A zero-volume PEC polygon on a declared Yee node plane.

    ``vertices`` and ``holes`` contain SI coordinates in ascending transverse-axis
    order: z -> (x, y), y -> (x, z), x -> (y, z). Multiple structures form unions.
    ``physical_thickness`` is provenance only: this ideal PEC model has no copper
    loss or volume, and its thickness does not impose a CFL grid interval.

    Plane placement must match a grid node; off-grid preparation raises instead
    of snapping. Only wholly supported tangential Yee edges are shorted. Cut
    lateral edges stay open, without fractional, dissipative PEC suppression.
    Geometry changes are discrete events; continuous shape gradients are rejected.
    """

    kind = "planar_pec"

    def __init__(self, vertices, *, holes=(), normal_axis="z", coordinate=0.0,
                 physical_thickness=0.0, device=None):
        if normal_axis not in ("x", "y", "z"):
            raise ValueError("PlanarPEC.normal_axis must be 'x', 'y', or 'z'.")
        if any(isinstance(v, torch.Tensor) and v.requires_grad for v in (coordinate, physical_thickness)):
            raise NotImplementedError("PlanarPEC has discrete edge ownership; shape gradients are unsupported.")
        self.normal_axis = normal_axis
        self.coordinate = float(coordinate)
        self.physical_thickness = float(physical_thickness)
        if not math.isfinite(self.coordinate):
            raise ValueError("PlanarPEC.coordinate must be finite.")
        if not math.isfinite(self.physical_thickness) or self.physical_thickness < 0:
            raise ValueError("PlanarPEC.physical_thickness must be finite and nonnegative.")
        self.vertices = self._loop(vertices, device=device)
        self.holes = tuple(self._loop(loop, device=device) for loop in holes)
        self._validate_topology()
        position = [0.0, 0.0, 0.0]
        position[("x", "y", "z").index(normal_axis)] = self.coordinate
        super().__init__(position=position, device=device)

    def _validate_topology(self):
        loops = [loop.detach().cpu().numpy() for loop in (self.vertices, *self.holes)]

        def intersects(a, b, c, d):
            def cross(p, q, r):
                first, second = q - p, r - p
                return float(first[0] * second[1] - first[1] * second[0])
            def on(p, q, r):
                return cross(p, q, r) == 0 and np.all(r >= np.minimum(p, q)) and np.all(r <= np.maximum(p, q))
            ab, cd = cross(a, b, c) * cross(a, b, d), cross(c, d, a) * cross(c, d, b)
            return (ab < 0 and cd < 0) or on(a, b, c) or on(a, b, d) or on(c, d, a) or on(c, d, b)

        for i, loop in enumerate(loops):
            for j in range(len(loop)):
                if np.array_equal(loop[j], loop[(j + 1) % len(loop)]):
                    raise ValueError("PlanarPEC loops must not repeat adjacent vertices.")
                for k in range(j + 1, len(loop)):
                    if k == j + 1 or (j == 0 and k == len(loop) - 1):
                        continue
                    if intersects(loop[j], loop[(j + 1) % len(loop)], loop[k], loop[(k + 1) % len(loop)]):
                        raise ValueError("PlanarPEC requires simple polygons without self intersections.")
            for previous in loops[:i]:
                if any(intersects(a, b, c, d)
                       for a, b in zip(loop, np.roll(loop, -1, axis=0))
                       for c, d in zip(previous, np.roll(previous, -1, axis=0))):
                    raise ValueError("PlanarPEC hole boundaries must be disjoint and strictly inside the outer loop.")
        for i, hole in enumerate(self.holes):
            point = hole[:1]
            if float(polygon_loops_signed_distance_2d(point, (self.vertices,))[0]) >= 0:
                raise ValueError("PlanarPEC holes must be strictly inside the outer loop.")
            for other in self.holes[:i]:
                if (float(polygon_loops_signed_distance_2d(point, (other,))[0]) <= 0
                        or float(polygon_loops_signed_distance_2d(other[:1], (hole,))[0]) <= 0):
                    raise ValueError("Nested PlanarPEC holes are unsupported; declare the intended union explicitly.")

    @staticmethod
    def _loop(value, *, device):
        if isinstance(value, torch.Tensor) and value.requires_grad:
            raise NotImplementedError("PlanarPEC has discrete edge ownership; shape gradients are unsupported.")
        loop = torch.as_tensor(value, dtype=torch.float64, device=device)
        if loop.ndim != 2 or loop.shape[0] < 3 or loop.shape[1] != 2:
            raise ValueError("PlanarPEC polygon loops require shape (N, 2), N >= 3.")
        if not bool(torch.isfinite(loop).all()):
            raise ValueError("PlanarPEC polygon coordinates must be finite.")
        area = (loop[:, 0] * loop.roll(-1, 0)[:, 1] - loop[:, 1] * loop.roll(-1, 0)[:, 0]).sum()
        if float(area) == 0.0:
            raise ValueError("PlanarPEC polygon loops must have nonzero area.")
        return loop

    @classmethod
    def rectangle(cls, bounds, **kwargs):
        """Create a finite rectangle from two transverse ``(low, high)`` bounds."""
        (u0, u1), (v0, v1) = bounds
        if not (u0 < u1 and v0 < v1):
            raise ValueError("PlanarPEC.rectangle bounds must be strictly increasing.")
        return cls(((u0, v0), (u1, v0), (u1, v1), (u0, v1)), **kwargs)

    def footprint_distance(self, u, v):
        shape = u.shape
        points = torch.stack((u.reshape(-1), v.reshape(-1)), -1)
        loops = [loop.to(device=u.device, dtype=u.dtype) for loop in (self.vertices, *self.holes)]
        return polygon_loops_signed_distance_2d(points, loops).reshape(shape)

    def signed_distance(self, xx, yy, zz):
        coords = (xx, yy, zz)
        normal = ("x", "y", "z").index(self.normal_axis)
        u, v = (coords[i] for i in range(3) if i != normal)
        return torch.maximum(self.footprint_distance(u, v), (coords[normal] - self.coordinate).abs())

    def to_mesh(self, segments=16):
        if self.holes:
            raise NotImplementedError("PlanarPEC mesh export with holes is not implemented.")
        from .polyslab import PolySlab
        slab = PolySlab(vertices=self.vertices,
                               bounds=(self.coordinate, self.coordinate), axis=self.normal_axis,
                               device=self.device)
        return slab.to_mesh(segments=segments)
