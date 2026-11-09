"""CUDA surface geometry and oriented Rao--Wilton--Glisson spaces.

Coordinates are metres. The RWG functions are dimensionless, coefficients
are A/m (electric current) or V/m (magnetic current), and the L2 mass matrix
has units m². The L2 metric is a proxy, not the electromagnetic trace norm.
Only combinatorial mesh construction and connectivity checks run on host.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch


def _cuda(device):
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("MoM numerical geometry requires a CUDA device")
    return device


@dataclass(frozen=True)
class MeshValidationReport:
    vertices: int
    triangles: int
    edges: int
    connected: bool = True
    closed: bool = True
    orientable: bool = True
    self_intersections: int = 0
    coordinate_units: str = "m"


def _topology(faces, vertex_count):
    host = faces.detach().cpu().tolist()
    incidence = {}
    used = set()
    for face_id, face in enumerate(host):
        if len(set(face)) != 3 or min(face) < 0 or max(face) >= vertex_count:
            raise ValueError("Triangle indices are invalid or repeated")
        used.update(face)
        for local in range(3):
            a, b = face[(local + 1) % 3], face[(local + 2) % 3]
            key = tuple(sorted((a, b)))
            incidence.setdefault(key, []).append((face_id, local, 1 if a < b else -1))
    if len(used) != vertex_count:
        raise ValueError("Mesh contains unused vertices")
    adjacency = [set() for _ in host]
    for pair in incidence.values():
        if len(pair) != 2:
            raise ValueError("Mesh must be a closed two-manifold: each edge needs two faces")
        (a, _, sa), (b, _, sb) = pair
        if sa == sb:
            raise ValueError("Mesh faces must have consistent orientation")
        adjacency[a].add(b)
        adjacency[b].add(a)
    visited, queue = set(), [0]
    while queue:
        item = queue.pop()
        if item not in visited:
            visited.add(item)
            queue.extend(adjacency[item] - visited)
    if len(visited) != len(host):
        raise ValueError("Mesh must have one connected surface component")
    # A manifold edge incidence alone does not exclude pinched vertices.
    stars = {}
    for face_id, face in enumerate(host):
        for vertex in face:
            stars.setdefault(vertex, set()).add(face_id)
    for vertex in used:
        star = stars[vertex]
        seen, queue = set(), [next(iter(star))]
        while queue:
            item = queue.pop()
            if item not in seen:
                seen.add(item)
                queue.extend((adjacency[item] & star) - seen)
        if seen != star:
            raise ValueError("Mesh has a non-manifold vertex link")
    return host, incidence


def _has_intersections(triangles):
    """Find improper triangle intersections using CUDA batched predicates.

    Candidate pairs are pruned with a bounding-volume traversal using an
    inflated tolerance, then filtered by the original box test before the
    original exact predicates; the decision equals the all-pairs scan.
    """
    from .spatial import BoxTree, overlapping_pairs
    scale = torch.linalg.vector_norm(triangles.amax((0, 1)) - triangles.amin((0, 1)))
    tol = 128 * torch.finfo(triangles.dtype).eps
    # Normalize geometry to make predicate tolerances independent of SI scale.
    tri = (triangles - triangles[0, 0]) / scale
    lows, highs = tri.amin(1), tri.amax(1)
    tree = BoxTree(lows, highs)
    for left, right in overlapping_pairs(tree, 2*tol):
        keep = ((lows[left] <= highs[right] + tol) & (highs[left] >= lows[right] - tol)).all(-1)
        left, right = left[keep], right[keep]
        for first in range(0, len(left), 1 << 16):
            if _pair_intersections(tri[left[first:first + (1 << 16)]], tri[right[first:first + (1 << 16)]], tol):
                return True
    return False


def _pair_intersections(a, b, tol):
    """Original exact predicates for paired candidate triangles (lower index first)."""
    if not len(a):
        return False
    for source, target in ((a, b), (b, a)):
        edge1, edge2 = target[:, 1] - target[:, 0], target[:, 2] - target[:, 0]
        for k in range(3):
            origin = source[:, k]
            direction = source[:, (k + 1) % 3] - origin
            h = torch.linalg.cross(direction, edge2)
            determinant = (edge1 * h).sum(-1)
            safe = torch.where(determinant.abs() > tol, determinant, torch.ones_like(determinant))
            s = origin - target[:, 0]
            u = (s * h).sum(-1) / safe
            q = torch.linalg.cross(s, edge1)
            v = (direction * q).sum(-1) / safe
            t = (edge2 * q).sum(-1) / safe
            end = source[:, (k + 1) % 3]
            # A nonparallel segment has only one intersection with the
            # target plane. An exactly shared endpoint fixes that location
            # without the cancellation in the scalar triple product for t.
            # Only this segment is classified; other edges and coplanar
            # overlap predicates still detect improper shared-face contacts.
            exact_start = (target == origin[:, None]).all(-1).any(-1)
            exact_end = (target == end[:, None]).all(-1).any(-1)
            t = torch.where(exact_start, torch.zeros_like(t), t)
            t = torch.where(exact_end, torch.ones_like(t), t)
            hit = (determinant.abs() > tol) & (u >= -tol) & (v >= -tol) & (u + v <= 1 + tol)
            # A nonshared endpoint touching a face or edge is also an
            # invalid contact. Shared conforming mesh vertices are allowed.
            interior_hit = (t > tol) & (t < 1 - tol)
            at_start = (t >= -tol) & (t <= tol)
            at_end = (t >= 1-tol) & (t <= 1+tol)
            shared_start = ((target-origin[:, None]).abs().amax(-1) <= tol).any(-1)
            shared_end = ((target-end[:, None]).abs().amax(-1) <= tol).any(-1)
            hit &= interior_hit | (at_start & ~shared_start) | (at_end & ~shared_end)
            if bool(hit.any()):
                return True
    # Coplanar intersections: strict edge crossings or interior vertices.
    normal = torch.linalg.cross(a[:, 1] - a[:, 0], a[:, 2] - a[:, 0])
    normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
    coplanar = (((b - a[:, :1]) * normal[:, None]).sum(-1).abs() < tol).all(-1)
    if not bool(coplanar.any()):
        return False
    ac, bc, nc = a[coplanar], b[coplanar], normal[coplanar]
    def side(p, q, r):
        return (torch.linalg.cross(q - p, r - p) * nc).sum(-1)
    for i in range(3):
        p, q = ac[:, i], ac[:, (i + 1) % 3]
        for j in range(3):
            r, s = bc[:, j], bc[:, (j + 1) % 3]
            # Require both endpoints to be strictly off the other edge.
            # Product-only tests amplify shared-endpoint roundoff when the
            # other orientation is large, spuriously crossing conforming
            # coplanar children after uniform refinement.
            def opposite(first,second):
                return ((first > tol) & (second < -tol)) | ((first < -tol) & (second > tol))
            crossing = opposite(side(p,q,r),side(p,q,s))
            crossing &= opposite(side(r,s,p),side(r,s,q))
            if bool(crossing.any()):
                return True
    for source, target in ((ac, bc), (bc, ac)):
        for k in range(3):
            signs = torch.stack([side(target[:, j], target[:, (j + 1) % 3], source[:, k]) for j in range(3)], -1)
            if bool(((signs > tol).all(-1) | (signs < -tol).all(-1)).any()):
                return True
            # Coplanar T contacts have neither a strict crossing nor an
            # interior vertex. Reject vertices in another edge's interior.
            for edge in range(3):
                a, b = target[:, edge], target[:, (edge+1) % 3]
                direction = b-a
                fraction = ((source[:, k]-a)*direction).sum(-1)/direction.square().sum(-1)
                projection = a+fraction[:, None]*direction
                on_edge = (source[:, k]-projection).abs().amax(-1) <= tol
                if bool((on_edge & (fraction > tol) & (fraction < 1-tol)).any()):
                    return True
    return False


class SurfaceMesh:
    """One closed consistently oriented triangular surface, resident on CUDA."""

    def __init__(self, vertices, faces, *, validate=True):
        if not isinstance(vertices, torch.Tensor) or vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must be a CUDA tensor with shape (V, 3)")
        _cuda(vertices.device)
        if vertices.dtype not in (torch.float32, torch.float64):
            raise ValueError("Mesh coordinates require float32 or float64")
        if not isinstance(faces, torch.Tensor) or faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype != torch.long:
            raise ValueError("faces must be an int64 tensor with shape (T, 3)")
        if len(faces) < 4:
            raise ValueError("A closed surface requires at least four triangles")
        self.vertices = vertices
        self.faces = faces.to(vertices.device)
        self._host_faces, self._incidence = _topology(self.faces, len(vertices))
        self._topology_version = (id(self.faces), self.faces._version)
        self.validation_report = validate_mesh(self) if validate else None

    @property
    def triangles(self):
        return self.vertices[self.faces]

    @property
    def areas(self):
        t = self.triangles
        return torch.linalg.vector_norm(torch.linalg.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), dim=-1) / 2

    @property
    def normals(self):
        t = self.triangles
        n = torch.linalg.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        return n / torch.linalg.vector_norm(n, dim=-1, keepdim=True)

    @property
    def centroids(self):
        return self.triangles.mean(1)

    @property
    def device(self):
        return self.vertices.device

    @property
    def dtype(self):
        return self.vertices.dtype

    def refine(self, levels=1):
        """Uniform planar midpoint refinement preserving the piecewise-flat body."""
        if type(levels) is not int or levels < 0:
            raise ValueError("Surface refinement levels must be a nonnegative integer")
        mesh = SurfaceMesh(self.vertices.clone(),self.faces.clone())
        for _ in range(levels):
            faces,vertices = mesh.faces,mesh.vertices
            edges = torch.stack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]),dim=1)
            unique,inverse = torch.unique(edges.reshape(-1,2).sort(dim=-1).values,
                                          dim=0,return_inverse=True)
            midpoint_ids = inverse.reshape(-1,3)+len(vertices)
            a,b,c = faces.unbind(-1)
            ab,bc,ca = midpoint_ids.unbind(-1)
            children = torch.stack((torch.stack((a,ab,ca),-1),torch.stack((ab,b,bc),-1),
                                    torch.stack((ca,bc,c),-1),torch.stack((ab,bc,ca),-1)),dim=1)
            midpoints = (vertices[unique[:,0]]+vertices[unique[:,1]])/2
            mesh = SurfaceMesh(torch.cat((vertices,midpoints)),children.reshape(-1,3))
        return mesh

    def quadrature(self, order=3):
        """Tensor Gauss--Legendre Duffy rule; ``order`` points per axis."""
        if not isinstance(order, int) or order < 1:
            raise ValueError("quadrature order must be a positive integer")
        k = torch.arange(1, order, dtype=self.dtype, device=self.device)
        off = k / torch.sqrt(4 * k * k - 1)
        jacobi = torch.diag(off, 1) + torch.diag(off, -1) if order > 1 else torch.zeros((1, 1), dtype=self.dtype, device=self.device)
        values, vectors = torch.linalg.eigh(jacobi)
        nodes, weights = (values + 1) / 2, vectors[0].square()
        u, v = torch.meshgrid(nodes, nodes, indexing="ij")
        bary = torch.stack((1 - u, u * (1 - v), u * v), -1).reshape(-1, 3)
        w = (weights[:, None] * weights[None, :] * u).reshape(-1)
        return torch.einsum("qk,tkd->tqd", bary, self.triangles), 2 * self.areas[:, None] * w


def _has_coincident_vertices(vertices, tolerance):
    """Any distinct vertex pair within ``tolerance`` in the L-infinity norm.

    Tree candidates use an inflated tolerance; the original test decides.
    """
    from .spatial import BoxTree, overlapping_pairs
    for first, second in overlapping_pairs(BoxTree(vertices, vertices), 2*tolerance):
        close = (vertices[first]-vertices[second]).abs().amax(-1) <= tolerance
        if bool((close & (first < second)).any()):
            return True
    return False


def validate_mesh(mesh, regions=None):
    """Reject degeneracy, inconsistent topology, intersections and inward orientation."""
    if regions is not None and len(regions) != 2:
        raise ValueError("The core mesh separates exactly two homogeneous regions")
    if not bool(torch.isfinite(mesh.vertices).all()):
        raise ValueError("Mesh coordinates must be finite")
    extent = torch.linalg.vector_norm(mesh.vertices.amax(0) - mesh.vertices.amin(0))
    if bool((mesh.areas <= torch.finfo(mesh.dtype).eps * extent.square()).any()):
        raise ValueError("Mesh contains degenerate triangles")
    tolerance = 128*torch.finfo(mesh.dtype).eps*extent
    if _has_coincident_vertices(mesh.vertices, tolerance):
        raise ValueError("Mesh has geometrically coincident vertices; nonincident surface contacts are invalid")
    t = mesh.triangles - mesh.vertices.mean(0)
    volume = (t[:, 0] * torch.linalg.cross(t[:, 1], t[:, 2])).sum() / 6
    if bool(volume <= 0):
        raise ValueError("Mesh normals must point outward (positive signed volume)")
    if _has_intersections(mesh.triangles):
        raise ValueError("Mesh contains self-intersecting triangles")
    return MeshValidationReport(len(mesh.vertices), len(mesh.faces), len(mesh._incidence))


def _resolution(subdivisions, frequency):
    if frequency is not None:
        if subdivisions is not None:
            raise ValueError("Specify exactly one of subdivisions and frequency")
        if type(frequency) is not int or frequency < 1:
            raise ValueError("frequency must be a positive integer")
    else:
        subdivisions = 1 if subdivisions is None else subdivisions
        if type(subdivisions) is not int or subdivisions < 0:
            raise ValueError("subdivisions must be a nonnegative integer")
    return subdivisions, frequency


def _icosahedron(device, dtype):
    p = (1 + math.sqrt(5)) / 2
    vertices = torch.tensor([(-1,p,0),(1,p,0),(-1,-p,0),(1,-p,0),(0,-1,p),(0,1,p),(0,-1,-p),(0,1,-p),(p,0,-1),(p,0,1),(-p,0,-1),(-p,0,1)], device=device, dtype=dtype)
    vertices = vertices / torch.linalg.vector_norm(vertices, dim=-1, keepdim=True)
    faces = [(0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),(1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),(3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),(4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1)]
    return vertices, faces


def _frequency_topology(seed_faces, frequency):
    """Host integer incidence only; shared edge nodes have one global identity."""
    lookup, seed_ids, weights, faces = {}, [], [], []
    for a, b, c in seed_faces:
        local = {}
        for i in range(frequency + 1):
            for j in range(frequency + 1 - i):
                coefficients = (frequency - i - j, i, j)
                key = tuple(sorted((vertex, weight) for vertex, weight in zip((a, b, c), coefficients) if weight))
                if key not in lookup:
                    lookup[key] = len(seed_ids)
                    seed_ids.append((a, b, c))
                    weights.append(coefficients)
                local[i, j] = lookup[key]
        for i in range(frequency):
            for j in range(frequency - i):
                faces.append((local[i, j], local[i + 1, j], local[i, j + 1]))
                if i + j < frequency - 1:
                    faces.append((local[i + 1, j], local[i + 1, j + 1], local[i, j + 1]))
    return seed_ids, weights, faces


def sphere_mesh(radius=1.0, subdivisions=None, *, frequency=None, center=(0., 0., 0.), device="cuda", dtype=torch.float64):
    """Icosahedral sphere: midpoint subdivisions or explicit edge frequency n.

    Omitted resolution retains one midpoint subdivision. Frequency uses one
    planar barycentric interpolation followed by radial CUDA projection; it is
    a distinct refinement family, with 20*n**2 conforming triangles.
    """
    subdivisions, frequency = _resolution(subdivisions, frequency)
    device = _cuda(device)
    if radius <= 0 or not math.isfinite(radius):
        raise ValueError("radius must be positive and finite")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Mesh coordinates require float32 or float64")
    offset = torch.as_tensor(center, dtype=dtype, device=device)
    if offset.shape != (3,) or not bool(torch.isfinite(offset).all()):
        raise ValueError("center must contain three finite coordinates")
    vertices, faces = _icosahedron(device, dtype)
    if frequency is not None:
        seed_ids, weights, faces = _frequency_topology(faces, frequency)
        ids = torch.tensor(seed_ids, device=device, dtype=torch.long)
        bary = torch.tensor(weights, device=device, dtype=dtype) / frequency
        interpolated = (vertices[ids] * bary[:, :, None]).sum(1)
        vertices = interpolated / torch.linalg.vector_norm(interpolated, dim=-1, keepdim=True)
    for _ in range(subdivisions or 0):
        edges = sorted({tuple(sorted((face[i], face[(i + 1) % 3]))) for face in faces for i in range(3)})
        edge_ids = {edge: len(vertices) + i for i, edge in enumerate(edges)}
        ids = torch.tensor(edges, device=device)
        mid = vertices[ids].mean(1)
        vertices = torch.cat((vertices, mid / torch.linalg.vector_norm(mid, dim=-1, keepdim=True)))
        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = [edge_ids[tuple(sorted(e))] for e in ((a,b),(b,c),(c,a))]
            new_faces.extend(((a,ab,ca),(b,bc,ab),(c,ca,bc),(ab,bc,ca)))
        faces = new_faces
    vertices = radius * vertices + offset
    return SurfaceMesh(vertices, torch.tensor(faces, dtype=torch.long, device=device))


def ellipsoid_mesh(axes=(1., 1., 1.), subdivisions=None, *, frequency=None, center=(0., 0., 0.), device="cuda", dtype=torch.float64):
    mesh = sphere_mesh(subdivisions=subdivisions, frequency=frequency, device=device, dtype=dtype)
    axes = torch.as_tensor(axes, device=mesh.device, dtype=dtype)
    if axes.shape != (3,) or not bool(torch.isfinite(axes).all() & (axes > 0).all()):
        raise ValueError("ellipsoid axes must contain three positive finite lengths")
    offset = torch.as_tensor(center, device=mesh.device, dtype=dtype)
    if offset.shape != (3,) or not bool(torch.isfinite(offset).all()):
        raise ValueError("center must contain three finite coordinates")
    return SurfaceMesh(mesh.vertices * axes + offset, mesh.faces)


def deformed_sphere_mesh(radius=1., subdivisions=None, *, frequency=None, harmonics: Mapping[tuple[int, int], float] | None = None, center=(0., 0., 0.), device="cuda", dtype=torch.float64):
    """Radial real orthonormal spherical harmonics, Condon--Shortley phase.

    Positive m denotes cos(m phi), negative m denotes sin(|m| phi).
    ``r = radius * (1 + sum(coeff[l,m] * Y[l,m]))`` must remain positive.
    """
    mesh = sphere_mesh(subdivisions=subdivisions, frequency=frequency, device=device, dtype=dtype)
    deformation = _radial_harmonics(mesh.vertices, harmonics)
    if radius <= 0 or not math.isfinite(radius) or not bool(torch.isfinite(deformation).all() & (deformation > 0).all()):
        raise ValueError("deformed sphere must have positive finite radius at every mesh vertex")
    offset = torch.as_tensor(center, device=mesh.device, dtype=dtype)
    if offset.shape != (3,) or not bool(torch.isfinite(offset).all()):
        raise ValueError("center must contain three finite coordinates")
    return SurfaceMesh(radius * deformation[:, None] * mesh.vertices + offset, mesh.faces)


def _radial_harmonics(directions, harmonics):
    """Shared real orthonormal harmonic mapping, evaluated on CUDA directions."""
    x, y, z = directions.unbind(-1)
    phi = torch.atan2(y, x)
    deformation = torch.ones_like(z)
    for (ell, signed_m), coefficient in (harmonics or {}).items():
        m = abs(signed_m)
        if type(ell) is not int or type(signed_m) is not int or ell < m or ell < 0 or not math.isfinite(coefficient):
            raise ValueError("harmonics require finite coefficients and integer l >= |m|")
        pmm = torch.ones_like(z)
        for k in range(1, m + 1):
            pmm = -(2 * k - 1) * torch.sqrt((1 - z.square()).clamp_min(0)) * pmm
        if ell == m:
            plm = pmm
        else:
            older, plm = pmm, (2 * m + 1) * z * pmm
            for degree in range(m + 2, ell + 1):
                older, plm = plm, ((2 * degree - 1) * z * plm - (degree + m - 1) * older) / (degree - m)
        norm = math.sqrt((2 * ell + 1) / (4 * math.pi) * math.factorial(ell - m) / math.factorial(ell + m))
        angular = torch.ones_like(z) if m == 0 else math.sqrt(2) * (torch.cos(m * phi) if signed_m > 0 else torch.sin(m * phi))
        deformation = deformation + coefficient * norm * plm * angular
    return deformation


class RWGBasis:
    """Canonical edge order; plus face traverses the sorted edge forwards."""

    def __init__(self, mesh: SurfaceMesh):
        if (id(mesh.faces), mesh.faces._version) != mesh._topology_version:
            raise ValueError('Mesh topology changed; reconstruct the SurfaceMesh before its basis')
        self.mesh = mesh
        self._mesh_version = (id(mesh.vertices), mesh.vertices._version, id(mesh.faces), mesh.faces._version)
        keys = sorted(mesh._incidence)
        self.edges = torch.tensor(keys, dtype=torch.long, device=mesh.device)
        face_edges = [[0] * 3 for _ in mesh._host_faces]
        signs = [[0] * 3 for _ in mesh._host_faces]
        plus, minus, pv, mv = [], [], [], []
        for edge_id, key in enumerate(keys):
            pairs = sorted(mesh._incidence[key], key=lambda entry: -entry[2])
            for face, local, sign in pairs:
                face_edges[face][local], signs[face][local] = edge_id, sign
            plus.append(pairs[0][0]); minus.append(pairs[1][0])
            pv.append(mesh._host_faces[pairs[0][0]][pairs[0][1]])
            mv.append(mesh._host_faces[pairs[1][0]][pairs[1][1]])
        for name, data in (("face_edges", face_edges), ("face_signs", signs), ("plus_faces", plus), ("minus_faces", minus), ("plus_vertices", pv), ("minus_vertices", mv)):
            setattr(self, name, torch.tensor(data, dtype=torch.long, device=mesh.device))
        self.edge_lengths = torch.linalg.vector_norm(mesh.vertices[self.edges[:, 1]] - mesh.vertices[self.edges[:, 0]], dim=-1)
        self.metadata = {"basis": "RWG", "test_space": "RWG", "edge_orientation": "ascending_vertex_index", "normal_orientation": "outward", "basis_units": "1", "J_coefficient_units": "A/m", "M_coefficient_units": "V/m", "metric": "surface_L2_proxy", "metric_units": "m^2", "fixture_state": "geometry_only_no_reference_solution"}

    @property
    def n_edges(self):
        return len(self.edges)

    @property
    def divergence(self):
        result = torch.zeros((len(self.mesh.faces), self.n_edges), device=self.mesh.device, dtype=self.mesh.dtype)
        return result.scatter(1, self.face_edges, self.face_signs * self.edge_lengths[self.face_edges] / self.mesh.areas[:, None])

    def local_values(self, points):
        """Values [T,Q,local_edge,xyz] at points [T,Q,xyz]."""
        if points.ndim != 3 or points.shape[0] != len(self.mesh.faces) or points.shape[-1] != 3:
            raise ValueError("points must have shape (triangles, quadrature_points, 3)")
        factors = self.face_signs * self.edge_lengths[self.face_edges] / (2 * self.mesh.areas[:, None])
        return factors[:, None, :, None] * (points[:, :, None, :] - self.mesh.triangles[:, None])

    def evaluate(self, points):
        result = torch.zeros((*points.shape[:2], self.n_edges, 3), device=points.device, dtype=points.dtype)
        indices = self.face_edges[:, None, :, None].expand(-1, points.shape[1], -1, 3)
        return result.scatter(2, indices, self.local_values(points))

    def current(self, coefficients, points):
        if coefficients.shape[-1] != self.n_edges:
            raise ValueError("RWG coefficient count does not match the basis")
        values = self.local_values(points).to(coefficients.dtype)
        return torch.einsum("...tk,tqkd->...tqd", coefficients[..., self.face_edges], values)

    def mass_blocks(self):
        """Exact per-face L2 Gram blocks ``[triangles, 3, 3]`` on CUDA.

        Block axes follow ``face_edges``. Shared-edge contributions have not
        yet been accumulated, so this representation is linear in the number
        of triangles and can be persisted without materializing a dense
        ``n_edges`` by ``n_edges`` matrix.
        """
        points, weights = self.mesh.quadrature(2)
        values = self.local_values(points)
        return torch.einsum("tqid,tqjd,tq->tij", values, values, weights)

    def mass_apply(self, coefficients):
        """Apply the exact L2 Gram matrix from local blocks.

        The coefficient dimension is last and may have arbitrary leading
        batch dimensions. Floating-point and complex CUDA tensors are
        supported; computation follows the coefficient precision without
        constructing the dense global Gram matrix.
        """
        if coefficients.ndim == 0 or coefficients.shape[-1] != self.n_edges:
            raise ValueError("RWG coefficient count does not match the basis")
        if coefficients.device != self.mesh.device:
            raise ValueError("RWG coefficients must use the basis CUDA device")
        if not (coefficients.is_floating_point() or coefficients.is_complex()):
            raise ValueError("RWG coefficients must use a floating-point or complex dtype")
        blocks = self.mass_blocks().to(coefficients.dtype)
        local = torch.einsum(
            "tij,...tj->...ti", blocks, coefficients[..., self.face_edges]
        )
        indices = self.face_edges.reshape(-1).expand(*coefficients.shape[:-1], -1)
        result = torch.zeros_like(coefficients)
        result.scatter_add_(-1, indices, local.flatten(start_dim=-2))
        return result

    def mass_matrix(self):
        """Exact dense L2 Gram matrix (quadratic integrand) on CUDA."""
        blocks = self.mass_blocks()
        indices = self.face_edges[:, :, None] * self.n_edges + self.face_edges[:, None, :]
        result = torch.zeros(self.n_edges**2, device=self.mesh.device, dtype=self.mesh.dtype)
        # In-place accumulation: the out-of-place form materializes a second
        # E x E buffer while the dense operator is already resident.
        result.scatter_add_(0, indices.reshape(-1), blocks.reshape(-1))
        return result.reshape(self.n_edges, self.n_edges)


def _closest_source_current(basis, coefficients, points):
    """Nearest-point lift from a polyhedral source to target quadrature points."""
    triangles = basis.mesh.triangles
    a, ab, ac = triangles[:, 0], triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    d00, d01, d11 = (ab * ab).sum(-1), (ab * ac).sum(-1), (ac * ac).sum(-1)
    den = d00 * d11 - d01.square()
    outputs = []
    for query in points.reshape(-1, 3).split(256):
        delta = query[:, None] - a
        d20, d21 = (delta * ab).sum(-1), (delta * ac).sum(-1)
        v, w = (d11 * d20 - d01 * d21) / den, (d00 * d21 - d01 * d20) / den
        inside = (v >= 0) & (w >= 0) & (v + w <= 1)
        projected = a + v[..., None] * ab + w[..., None] * ac
        distance = (query[:, None] - projected).square().sum(-1)
        distance = torch.where(inside, distance, torch.full_like(distance, torch.inf))
        closest = projected
        for edge in range(3):
            p, q = triangles[:, edge], triangles[:, (edge + 1) % 3]
            direction = q - p
            alpha = (((query[:, None] - p) * direction).sum(-1) / direction.square().sum(-1)).clamp(0, 1)
            candidate = p + alpha[..., None] * direction
            candidate_distance = (query[:, None] - candidate).square().sum(-1)
            better = candidate_distance < distance
            closest = torch.where(better[..., None], candidate, closest)
            distance = torch.minimum(distance, candidate_distance)
        face = distance.argmin(-1)
        nearest = closest[torch.arange(len(query), device=query.device), face]
        factors = basis.face_signs[face] * basis.edge_lengths[basis.face_edges[face]] / (2 * basis.mesh.areas[face, None])
        values = factors[..., None] * (nearest[:, None] - triangles[face])
        outputs.append(torch.einsum("...qk,qkd->...qd", coefficients[..., basis.face_edges[face]], values.to(coefficients.dtype)))
    return torch.cat(outputs, dim=-2).reshape(*coefficients.shape[:-1], *points.shape)


def project_current(source_mesh, target_mesh, coefficients, *, quadrature_order=4):
    """L2 project a nearest-point tangential lift; return relative L2 fit error.

    The reported error measures the tangential lift's fit on target quadrature;
    it does not estimate the source or surface discretization error.
    Mesh arguments may also be prebuilt RWGBasis objects.
    """
    source = source_mesh if isinstance(source_mesh, RWGBasis) else RWGBasis(source_mesh)
    target = target_mesh if isinstance(target_mesh, RWGBasis) else RWGBasis(target_mesh)
    if source.mesh.device != target.mesh.device or coefficients.device != source.mesh.device:
        raise ValueError("Projection meshes and coefficients must share a CUDA device")
    if coefficients.shape[-1] != source.n_edges:
        raise ValueError("Source coefficient count does not match the basis")
    points, weights = target.mesh.quadrature(quadrature_order)
    lifted = _closest_source_current(source, coefficients, points)
    normal = target.mesh.normals
    lifted = lifted - (lifted * normal[:, None]).sum(-1, keepdim=True) * normal[:, None]
    values = target.local_values(points).to(coefficients.dtype)
    local_rhs = torch.einsum("tqkd,...tqd,tq->...tk", values, lifted, weights.to(coefficients.dtype))
    rhs = torch.zeros((*coefficients.shape[:-1], target.n_edges), dtype=coefficients.dtype, device=coefficients.device)
    rhs = rhs.scatter_add(-1, target.face_edges.flatten().expand(*coefficients.shape[:-1], -1), local_rhs.flatten(-2))
    mass = target.mass_matrix().to(coefficients.dtype)
    projected = torch.linalg.solve(mass, rhs.unsqueeze(-1)).squeeze(-1)
    reconstructed = target.current(projected, points)
    error = ((reconstructed - lifted).abs().square().sum(-1) * weights).sum((-2, -1)).sqrt()
    norm = (lifted.abs().square().sum(-1) * weights).sum((-2, -1)).sqrt()
    return projected, error / norm.clamp_min(torch.finfo(weights.dtype).tiny)


def _point_triangle_distance_squared(point, triangle):
    """Paired/broadcast point-triangle distance; all geometry remains on CUDA."""
    a, b, c = triangle.unbind(-2)
    ab, ac = b-a, c-a
    normal = torch.linalg.cross(ab, ac)
    nn = normal.square().sum(-1)
    offset = point-a
    projected = point-((offset*normal).sum(-1)/nn)[..., None]*normal
    v = (torch.linalg.cross(projected-a, ac)*normal).sum(-1)/nn
    w = (torch.linalg.cross(ab, projected-a)*normal).sum(-1)/nn
    inside = (v >= 0) & (w >= 0) & (v+w <= 1)
    result = torch.where(inside, (point-projected).square().sum(-1), torch.inf)
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end-start
        fraction = ((point-start)*edge).sum(-1)/edge.square().sum(-1)
        nearest = start+fraction.clamp(0, 1)[..., None]*edge
        result = torch.minimum(result, (point-nearest).square().sum(-1))
    return result.clamp_min(0)


def triangle_pair_distances(left, right):
    """Exact primitive triangle distances, including crossing edges and faces.

    Inputs are paired [P,3,3] triangles, never an implicit P-by-P expansion.
    Arithmetic uses the mesh precision; contacts within its validation tolerance
    are rejected by SurfaceSystem rather than assigned a positive gap.
    """
    if (left.device.type != 'cuda' or right.device != left.device or right.dtype != left.dtype
            or left.ndim != 3 or left.shape[1:] != (3, 3) or right.shape != left.shape):
        raise ValueError('Triangle distance pairs require matching CUDA [P,3,3] arrays')
    best = torch.full(left.shape[:1], torch.inf, device=left.device, dtype=left.dtype)
    for source, target in ((left, right), (right, left)):
        normal = torch.linalg.cross(target[:, 1]-target[:, 0], target[:, 2]-target[:, 0])
        nn = normal.square().sum(-1)
        for i in range(3):
            start, end = source[:, i], source[:, (i+1) % 3]
            best = torch.minimum(best, _point_triangle_distance_squared(start, target))
            direction = end-start
            denominator = (direction*normal).sum(-1)
            nonparallel = denominator.abs() > torch.finfo(left.dtype).tiny
            fraction = ((target[:, 0]-start)*normal).sum(-1)/torch.where(nonparallel, denominator, 1)
            crossing = start+fraction[:, None]*direction
            ab, ac = target[:, 1]-target[:, 0], target[:, 2]-target[:, 0]
            v = (torch.linalg.cross(crossing-target[:, 0], ac)*normal).sum(-1)/nn
            w = (torch.linalg.cross(ab, crossing-target[:, 0])*normal).sum(-1)/nn
            hit = nonparallel & (fraction >= 0) & (fraction <= 1) & (v >= 0) & (w >= 0) & (v+w <= 1)
            best = torch.where(hit, 0, best)
    # Edge-edge minima cover skew or coplanar crossings missed by vertex tests.
    for i in range(3):
        p, u = left[:, i], left[:, (i+1) % 3]-left[:, i]
        for j in range(3):
            q, v = right[:, j], right[:, (j+1) % 3]-right[:, j]
            r = p-q
            a, b, c = (u*u).sum(-1), (u*v).sum(-1), (v*v).sum(-1)
            d, e = (u*r).sum(-1), (v*r).sum(-1)
            cross = torch.linalg.cross(u, v)
            denominator = cross.square().sum(-1)
            valid = denominator > torch.finfo(left.dtype).tiny
            numerator = (torch.linalg.cross(-r, v)*cross).sum(-1)
            s = torch.where(valid, numerator/torch.where(valid, denominator, 1), 0).clamp(0, 1)
            t = (b*s+e)/c
            s = torch.where(t < 0, (-d/a).clamp(0, 1), s)
            s = torch.where(t > 1, ((b-d)/a).clamp(0, 1), s)
            t = t.clamp(0, 1)
            separation = r+s[:, None]*u-t[:, None]*v
            best = torch.minimum(best, separation.square().sum(-1))
    return best.clamp_min(0).sqrt()


def _triangle_tree(triangles):
    from .spatial import BoxTree
    return BoxTree(triangles.amin(1), triangles.amax(1), triangles[:, 0])


def _minimum_triangle_distance(left, right, left_tree=None, right_tree=None):
    """All-pairs minimum of ``triangle_pair_distances`` by exact branch and bound.

    Returns a 0-d tensor in the triangle precision, equal to the minimum over
    every left/right pair because the minimizing pair is never pruned.
    """
    from .spatial import minimum_item_distance
    eps = torch.finfo(left.dtype).eps
    extent = float(torch.maximum(left.abs().amax(), right.abs().amax()))
    value, _ = minimum_item_distance(left_tree or _triangle_tree(left), right_tree or _triangle_tree(right),
        lambda first, second: triangle_pair_distances(left[first], right[second]),
        relative_margin=1024*eps, absolute_margin=1024*eps*extent)
    return torch.tensor(value, dtype=left.dtype, device=left.device)


@dataclass(frozen=True)
class SurfaceBody:
    structure_index: int
    name: str | None
    basis: RWGBasis
    interior: object
    edge_offset: int
    face_offset: int
    region_id: int

    @property
    def mesh(self):
        return self.basis.mesh


@dataclass(frozen=True)
class SurfaceSystem:
    """Separate closed bodies in Scene order; geometry is never joined as one mesh."""
    bodies: tuple[SurfaceBody, ...]

    @classmethod
    def create(cls, entries):
        bodies, edges, faces = [], 0, 0
        for region, (index, name, basis, interior) in enumerate(entries, 1):
            bodies.append(SurfaceBody(index, name, basis, interior, edges, faces, region))
            edges += basis.n_edges
            faces += len(basis.mesh.faces)
        return cls(tuple(bodies))

    def __post_init__(self):
        from types import MappingProxyType
        if not isinstance(self.bodies, tuple) or not self.bodies:
            raise ValueError('SurfaceSystem requires a nonempty ordered tuple of bodies')
        edges = faces = 0
        names, indices = {}, []
        for region, body in enumerate(self.bodies, 1):
            if (type(body.structure_index) is not int or body.structure_index < 0
                    or body.structure_index in indices or (indices and body.structure_index <= indices[-1])):
                raise ValueError('Bodies must retain unique enabled Scene structure order')
            if (any(type(value) is not int for value in (body.region_id, body.edge_offset, body.face_offset))
                    or body.region_id != region or body.edge_offset != edges or body.face_offset != faces):
                raise ValueError('Body regions and offsets must be contiguous in Scene order')
            if body.name is not None:
                if not isinstance(body.name, str) or not body.name.strip() or body.name in ('exterior', 'interior') or body.name in names:
                    raise ValueError('Body aliases must be unique, nonempty and not reserved')
                names[body.name] = region
            if body.mesh.device != self.bodies[0].mesh.device or body.mesh.dtype != self.bodies[0].mesh.dtype:
                raise ValueError('Every body must share CUDA device and coordinate precision')
            version = (id(body.mesh.vertices), body.mesh.vertices._version, id(body.mesh.faces), body.mesh.faces._version)
            if body.basis._mesh_version != version:
                raise ValueError('Body basis was built for a different geometry version')
            # Revalidate actual geometry, including explicitly unvalidated meshes.
            validate_mesh(body.mesh)
            indices.append(body.structure_index)
            edges += body.basis.n_edges
            faces += len(body.mesh.faces)
        object.__setattr__(self, 'total_edges', edges)
        object.__setattr__(self, 'total_faces', faces)
        object.__setattr__(self, 'aliases', MappingProxyType(names))
        object.__setattr__(self, 'J_slices', tuple(slice(b.edge_offset, b.edge_offset+b.basis.n_edges) for b in self.bodies))
        object.__setattr__(self, 'M_slices', tuple(slice(edges+b.edge_offset, edges+b.edge_offset+b.basis.n_edges) for b in self.bodies))
        object.__setattr__(self, 'face_edges', torch.cat([b.basis.face_edges+b.edge_offset for b in self.bodies]))
        object.__setattr__(self, '_versions', tuple((id(b.mesh.vertices), b.mesh.vertices._version,
            id(b.mesh.faces), b.mesh.faces._version) for b in self.bodies))
        object.__setattr__(self, 'pair_gaps', self._validate_separation())

    @property
    def device(self): return self.bodies[0].mesh.device
    @property
    def dtype(self): return self.bodies[0].mesh.dtype

    def assert_current(self):
        current = tuple((id(b.mesh.vertices), b.mesh.vertices._version, id(b.mesh.faces), b.mesh.faces._version) for b in self.bodies)
        if current != self._versions:
            raise ValueError('Surface geometry changed after system validation; rebuild its bases and system')

    def region(self, region):
        if type(region) is int:
            if 0 <= region <= len(self.bodies): return region
        elif isinstance(region, str):
            if region == 'exterior': return 0
            if region == 'interior' and len(self.bodies) == 1: return 1
            if region in self.aliases: return self.aliases[region]
        raise ValueError('Unknown or ambiguous surface region')

    def _validate_separation(self):
        from .fields import classify_queries
        gaps = []
        trees = [_triangle_tree(body.mesh.triangles) for body in self.bodies]
        for i, left in enumerate(self.bodies):
            for j, right in enumerate(self.bodies[i+1:], i+1):
                minimum = _minimum_triangle_distance(left.mesh.triangles, right.mesh.triangles, trees[i], trees[j])
                scale = torch.maximum(left.mesh.vertices.abs().amax(), right.mesh.vertices.abs().amax())
                tolerance = 256*torch.finfo(self.dtype).eps*scale
                if not bool(torch.isfinite(minimum)) or bool(minimum <= tolerance):
                    raise ValueError('Surface bodies touch or intersect; positive disjoint separation is required')
                inside_left, _ = classify_queries(left.mesh, right.mesh.vertices[:1])
                inside_right, _ = classify_queries(right.mesh, left.mesh.vertices[:1])
                if bool(inside_left.any() | inside_right.any()):
                    raise ValueError('Nested or overlapping surface bodies are outside this formulation')
                gaps.append((left.region_id, right.region_id, float(minimum)))
        return tuple(gaps)
