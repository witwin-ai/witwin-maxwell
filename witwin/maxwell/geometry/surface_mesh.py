"""Fixed oriented triangle surfaces for native zero-volume PEC traces."""
from __future__ import annotations

import math

import torch
from witwin.core import GeometryBase


class SurfaceMesh(GeometryBase):
    """A true zero-volume surface with SI float64 vertices and fixed topology.

    PEC surfaces default to the native mass-orthogonal trace projection. A
    planar mesh carrying ``CopperFoil``, or a planar PEC mesh declared with
    ``sheet_model="conformal_two_sided"``, uses the conformal two-sided
    thin-sheet runtime (native CUDA or the CPU reference backend).
    ``physical_thickness`` records provenance; it does not create a volume.
    """

    kind = "surface_mesh"

    def __init__(self, vertices, triangles, *, position=(0., 0., 0.), rotation=None,
                 physical_thickness=0., sheet_model=None, device=None):
        for value in (vertices, position, rotation, physical_thickness):
            if isinstance(value, torch.Tensor) and value.requires_grad:
                raise NotImplementedError("SurfaceMesh requires fixed geometry and topology.")
        super().__init__(device=device)
        self.vertices = torch.as_tensor(vertices, dtype=torch.float64, device=device)
        topology = torch.as_tensor(triangles, device=self.vertices.device)
        if topology.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("SurfaceMesh triangle indices must have an integer dtype.")
        self.triangles = topology.to(dtype=torch.int64)
        self.position = torch.as_tensor(position, dtype=torch.float64, device=self.vertices.device)
        self.rotation = torch.as_tensor((1., 0., 0., 0.) if rotation is None else rotation,
                                        dtype=torch.float64, device=self.vertices.device)
        self.physical_thickness = float(physical_thickness)
        if sheet_model not in (None, "conformal_two_sided"):
            raise ValueError("SurfaceMesh sheet_model must be None or 'conformal_two_sided'.")
        self.sheet_model = sheet_model
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3 or len(self.vertices) < 3:
            raise ValueError("SurfaceMesh vertices require shape [N,3], N >= 3.")
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3 or len(self.triangles) < 1:
            raise ValueError("SurfaceMesh triangles require shape [T,3], T >= 1.")
        if self.position.shape != (3,) or self.rotation.shape != (4,):
            raise ValueError("SurfaceMesh position and rotation require three coordinates and a wxyz quaternion.")
        if not all(bool(torch.isfinite(value).all()) for value in (self.vertices, self.position, self.rotation)):
            raise ValueError("SurfaceMesh coordinates and transformation must be finite.")
        norm = torch.linalg.vector_norm(self.rotation)
        if not bool(norm > 0):
            raise ValueError("SurfaceMesh rotation must be nonzero.")
        self.rotation = self.rotation/norm
        if not math.isfinite(self.physical_thickness) or self.physical_thickness < 0:
            raise ValueError("SurfaceMesh physical_thickness must be finite and nonnegative.")
        self._validate_topology()

    def _validate_topology(self):
        faces = self.triangles.detach().cpu().tolist()
        if len({tuple(sorted(face)) for face in faces}) != len(faces):
            raise ValueError("SurfaceMesh contains a repeated triangle.")
        incidence, used = {}, set()
        for face_id, face in enumerate(faces):
            if len(set(face)) != 3 or min(face) < 0 or max(face) >= len(self.vertices):
                raise ValueError("SurfaceMesh triangle indices must be distinct and valid.")
            used.update(face)
            for a, b in zip(face, (*face[1:], face[0])):
                incidence.setdefault(tuple(sorted((a, b))), []).append((face_id, a < b))
        if len(used) != len(self.vertices):
            raise ValueError("SurfaceMesh contains unused vertices.")
        if any(len(entries) > 2 for entries in incidence.values()):
            raise ValueError("SurfaceMesh has a nonmanifold edge.")
        if any(len(entries) == 2 and entries[0][1] == entries[1][1] for entries in incidence.values()):
            raise ValueError("SurfaceMesh neighboring faces must have consistent orientation.")
        adjacency = [set() for _ in faces]
        for entries in incidence.values():
            if len(entries) == 2:
                a, b = entries[0][0], entries[1][0]
                adjacency[a].add(b)
                adjacency[b].add(a)
        seen, queue = set(), [0]
        while queue:
            item = queue.pop()
            if item not in seen:
                seen.add(item)
                queue.extend(adjacency[item]-seen)
        if len(seen) != len(faces):
            raise ValueError("SurfaceMesh must be one connected surface; use separate structures for separate surfaces.")
        for vertex in used:
            star = {i for i, face in enumerate(faces) if vertex in face}
            seen, queue = set(), [next(iter(star))]
            while queue:
                item = queue.pop()
                if item not in seen:
                    seen.add(item)
                    queue.extend((adjacency[item] & star)-seen)
            boundary_degree = sum(vertex in edge and len(entries) == 1 for edge, entries in incidence.items())
            if seen != star or boundary_degree not in (0, 2):
                raise ValueError("SurfaceMesh has a nonmanifold vertex link.")
        self._boundary_edges = tuple((a, b) for face in faces for a, b in zip(face, (*face[1:], face[0]))
                                     if len(incidence[tuple(sorted((a, b)))]) == 1)
        self.closed = not self._boundary_edges
        vertices = self.vertices[self.triangles]
        cross = torch.linalg.cross(vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0])
        scale = torch.linalg.vector_norm(self.vertices.amax(0)-self.vertices.amin(0))
        if bool(torch.any(torch.linalg.vector_norm(cross, dim=-1) <= 64*torch.finfo(torch.float64).eps*scale.square())):
            raise ValueError("SurfaceMesh contains a degenerate or numerically unresolved triangle.")

    def to_mesh(self, segments=16):
        del segments
        return self._transform_mesh_verts(self.vertices), self.triangles

    def signed_distance(self, xx, yy, zz):
        del xx, yy, zz
        raise NotImplementedError("SurfaceMesh is zero volume and requires native PEC cut-trace compilation; it cannot be sampled as material occupancy.")

    def classify_query_points(self, points):
        """Exact triangle contact and closed-surface winding for physical queries.

        This is not a material sampling function. Open sheets have no interior;
        their two exterior sides are distinguished by the physical query point.
        """
        vertices, faces = self.to_mesh()
        triangle = vertices.to(points.device)[faces.to(points.device)]
        point = points.to(dtype=torch.float64)[:, None]
        a, b, c = triangle[None].unbind(2)
        u, v, w = b-a, c-a, point-a
        uu, uv, vv = (u*u).sum(-1), (u*v).sum(-1), (v*v).sum(-1)
        wu, wv = (w*u).sum(-1), (w*v).sum(-1)
        denominator = uu*vv-uv.square()
        alpha, beta = (wu*vv-wv*uv)/denominator, (wv*uu-wu*uv)/denominator
        normal = torch.linalg.cross(u, v)
        plane_distance_squared = (w*normal).sum(-1).square()/normal.square().sum(-1)
        inside_face = (alpha >= 0) & (beta >= 0) & (alpha+beta <= 1)
        distance_squared = torch.where(inside_face, plane_distance_squared, torch.full_like(plane_distance_squared, float("inf")))
        for start, end in ((a, b), (b, c), (c, a)):
            edge = end-start
            fraction = ((point-start)*edge).sum(-1)/edge.square().sum(-1)
            delta = point-start-fraction.clamp(0, 1)[..., None]*edge
            distance_squared = torch.minimum(distance_squared, delta.square().sum(-1))
        scale = torch.linalg.vector_norm(vertices.amax(0)-vertices.amin(0)).to(points.device)
        contact = distance_squared.amin(-1) <= (128*torch.finfo(torch.float64).eps*scale).square()
        interior = torch.zeros(len(points), dtype=torch.bool, device=points.device)
        if self.closed:
            a, b, c = (triangle[None]-point[:, :, None]).unbind(2)
            la, lb, lc = (torch.linalg.vector_norm(value, dim=-1) for value in (a, b, c))
            numerator = (a*torch.linalg.cross(b, c)).sum(-1)
            denominator = la*lb*lc+(a*b).sum(-1)*lc+(b*c).sum(-1)*la+(c*a).sum(-1)*lb
            angle = 2*torch.atan2(numerator, denominator)
            interior = angle.sum(-1).abs() > 2*math.pi
        return interior, contact

    def planar_pec(self, scene):
        """Exact grid-aligned reduction, otherwise leave the native cut surface."""
        from .planar_pec import PlanarPEC
        vertices, _ = self.to_mesh()
        vertices = vertices.to(scene.device)
        ranges = vertices.amax(0)-vertices.amin(0)
        scale = ranges.max()
        axes = torch.nonzero(ranges <= 8*torch.finfo(torch.float64).eps*scale).flatten()
        if len(axes) != 1 or not self._boundary_edges:
            return None
        axis = int(axes[0])
        coordinate = vertices[:, axis].mean()
        if not hasattr(scene, "xyz"[axis]+"_nodes64"):
            from ..scene import prepare_scene
            scene = prepare_scene(scene)
        nodes = torch.as_tensor(getattr(scene, "xyz"[axis]+"_nodes64"), device=vertices.device, dtype=torch.float64)
        spacing = (nodes[1:]-nodes[:-1]).min()
        if not bool(torch.min((nodes-coordinate).abs()) <= 8*torch.finfo(torch.float64).eps*spacing):
            return None
        outgoing = {a: b for a, b in self._boundary_edges}
        if len(outgoing) != len(self._boundary_edges):
            raise ValueError("SurfaceMesh boundary has a nonmanifold vertex.")
        loops = []
        while outgoing:
            first = next(iter(outgoing))
            loop, current = [], first
            while current in outgoing:
                loop.append(current)
                current = outgoing.pop(current)
            if current != first:
                raise ValueError("SurfaceMesh boundary loops must close.")
            loops.append(vertices[loop][:, [i for i in range(3) if i != axis]])
        areas = [(loop[:, 0]*loop.roll(-1, 0)[:, 1]-loop[:, 1]*loop.roll(-1, 0)[:, 0]).sum().abs() for loop in loops]
        outer = max(range(len(loops)), key=lambda i: float(areas[i]))
        return PlanarPEC(loops[outer], holes=[loop for i, loop in enumerate(loops) if i != outer],
            normal_axis="xyz"[axis], coordinate=float(coordinate), physical_thickness=self.physical_thickness,
            device=vertices.device)

    @classmethod
    def bent_rectangle(cls, *, width, length, radius, segments=16, device=None, **kwargs):
        """Bend an arc-length-preserving rectangle about its longitudinal y axis."""
        if any(isinstance(value, torch.Tensor) and value.requires_grad for value in (width, length, radius)):
            raise NotImplementedError("SurfaceMesh requires fixed geometry and topology.")
        width, length, radius = float(width), float(length), float(radius)
        if not all(math.isfinite(float(value)) and float(value) > 0 for value in (width, length, radius)):
            raise ValueError("Bent surface width, length and radius must be finite and positive.")
        if width/radius >= math.pi:
            raise ValueError("Bent rectangle currently requires an opening angle below pi.")
        if isinstance(segments, bool) or not isinstance(segments, int) or segments < 1:
            raise ValueError("Bent rectangle segments must be a positive integer.")
        u = torch.linspace(-width/2, width/2, segments+1, device=device, dtype=torch.float64)
        v = torch.tensor((-length/2, length/2), device=device, dtype=torch.float64)
        angle = u/radius
        x, y = torch.meshgrid(radius*torch.sin(angle), v, indexing="ij")
        z = (2*radius*torch.sin(angle/2).square())[:, None].expand(-1, 2)
        vertices = torch.stack((x, y, z), -1).reshape(-1, 3)
        faces = [(2*i, 2*i+2, 2*i+3) for i in range(segments)]
        faces += [(2*i, 2*i+3, 2*i+1) for i in range(segments)]
        return cls(vertices, faces, device=device, **kwargs)
