"""CUDA open triangle surfaces and RWG spaces on their interior edges.

An open sheet carries one total surface current J. RWG functions exist only on
interior edges, so the current has zero normal component on every boundary
edge; boundary edges carry no unknown. The edge order and plus/minus face
convention are those of the closed ``RWGBasis``: on a closed surface both
spaces coincide function by function.

Coordinates are metres, RWG functions are dimensionless and J coefficients are
A/m. Only the zero-thickness primitives' structured triangulations are built
from host integers; geometry and validation run on CUDA.
"""
from __future__ import annotations

import math

import torch

from .mesh import RWGBasis, SurfaceMesh, _cuda, _has_intersections
from .topology import INTERIOR, classify_surface, require_open_sheet


class OpenSurfaceMesh:
    """One consistently oriented manifold triangle surface, open or closed."""

    triangles = SurfaceMesh.triangles
    areas = SurfaceMesh.areas
    normals = SurfaceMesh.normals
    centroids = SurfaceMesh.centroids
    device = SurfaceMesh.device
    dtype = SurfaceMesh.dtype
    quadrature = SurfaceMesh.quadrature

    def __init__(self, vertices, faces, *, validate=True):
        if not isinstance(vertices, torch.Tensor) or vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError('vertices must be a CUDA tensor with shape (V, 3)')
        _cuda(vertices.device)
        if vertices.dtype not in (torch.float32, torch.float64):
            raise ValueError('Mesh coordinates require float32 or float64')
        if not isinstance(faces, torch.Tensor) or faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype != torch.long:
            raise ValueError('faces must be an int64 tensor with shape (T, 3)')
        self.vertices = vertices
        self.faces = faces.to(vertices.device)
        self.topology = require_open_sheet(classify_surface(self.faces, len(vertices)))
        self._topology_version = (id(self.faces), self.faces._version)
        self.validation_report = validate_open_surface(self) if validate else None

    def refine(self, levels=1):
        """Uniform planar midpoint refinement preserving the piecewise-flat sheet."""
        if type(levels) is not int or levels < 0:
            raise ValueError('Surface refinement levels must be a nonnegative integer')
        mesh = OpenSurfaceMesh(self.vertices.clone(), self.faces.clone())
        for _ in range(levels):
            faces, vertices = mesh.faces, mesh.vertices
            edges = torch.stack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), dim=1)
            unique, inverse = torch.unique(edges.reshape(-1, 2).sort(dim=-1).values, dim=0, return_inverse=True)
            ab, bc, ca = (inverse.reshape(-1, 3)+len(vertices)).unbind(-1)
            a, b, c = faces.unbind(-1)
            children = torch.stack((torch.stack((a, ab, ca), -1), torch.stack((ab, b, bc), -1),
                                    torch.stack((ca, bc, c), -1), torch.stack((ab, bc, ca), -1)), dim=1)
            midpoints = (vertices[unique[:, 0]]+vertices[unique[:, 1]])/2
            mesh = OpenSurfaceMesh(torch.cat((vertices, midpoints)), children.reshape(-1, 3))
        return mesh


def validate_open_surface(mesh):
    """Reject nonfinite, degenerate, coincident and self-intersecting geometry."""
    if not bool(torch.isfinite(mesh.vertices).all()):
        raise ValueError('Mesh coordinates must be finite')
    extent = torch.linalg.vector_norm(mesh.vertices.amax(0)-mesh.vertices.amin(0))
    if bool((mesh.areas <= torch.finfo(mesh.dtype).eps*extent.square()).any()):
        raise ValueError('Mesh contains degenerate triangles')
    tolerance = 128*torch.finfo(mesh.dtype).eps*extent
    ids = torch.arange(len(mesh.vertices), device=mesh.device)
    for start in range(0, len(mesh.vertices), 128):
        close = (mesh.vertices[start:start+128, None]-mesh.vertices[None]).abs().amax(-1) <= tolerance
        if bool((close & (ids[start:start+128, None] < ids[None])).any()):
            raise ValueError('Mesh has geometrically coincident vertices; nonincident surface contacts are invalid')
    if _has_intersections(mesh.triangles):
        raise ValueError('Mesh contains self-intersecting triangles')
    return {'vertices': len(mesh.vertices), 'triangles': len(mesh.faces), **mesh.topology.summary(),
            'coordinate_units': 'm'}


class OpenRWGBasis:
    """RWG functions on interior edges; boundary-edge slots carry zero slope.

    ``face_edges``/``face_signs`` keep the closed-basis [T, 3] layout. A local
    slot on a boundary edge has sign 0 and an arbitrary valid index, so every
    slope-weighted local formula vanishes there. Scatters into coefficient
    space must therefore accumulate (``scatter_add``), never overwrite.

    ``boundary_half_functions=True`` builds the non-tilde space used by dual
    (Calderon) bases: every boundary edge also carries a one-face RWG oriented
    out of the sheet (``minus_faces`` is -1), so normal flux through the
    boundary is free and the intra-sheet divergence excludes boundary charge.
    """

    n_edges = RWGBasis.n_edges
    local_values = RWGBasis.local_values
    current = RWGBasis.current
    mass_blocks = RWGBasis.mass_blocks
    mass_apply = RWGBasis.mass_apply
    mass_matrix = RWGBasis.mass_matrix

    def __init__(self, mesh: OpenSurfaceMesh, *, boundary_half_functions=False):
        if not isinstance(mesh, OpenSurfaceMesh):
            raise TypeError('OpenRWGBasis requires an OpenSurfaceMesh')
        if (id(mesh.faces), mesh.faces._version) != mesh._topology_version:
            raise ValueError('Mesh topology changed; reconstruct the OpenSurfaceMesh before its basis')
        topology = mesh.topology
        self.mesh = mesh
        self._mesh_version = (id(mesh.vertices), mesh.vertices._version, id(mesh.faces), mesh.faces._version)
        interior = topology.kind == INTERIOR
        if boundary_half_functions:
            interior = torch.ones_like(interior)
        if not bool(interior.any()):
            raise ValueError('An open surface needs at least one interior edge to carry RWG current')
        basis_id = torch.cumsum(interior.to(torch.long), 0)-1
        local = basis_id[topology.face_edges]
        valid = interior[topology.face_edges]
        self.edges = topology.edges[interior]
        self.face_edges = torch.where(valid, local, 0)
        # A one-face function is its face's plus function: it flows out of the sheet.
        signs = torch.where((topology.kind != INTERIOR)[topology.face_edges], 1, topology.face_edge_signs)
        self.face_signs = torch.where(valid, signs, 0)
        faces, slots = topology.incidence_faces[interior], topology.incidence_slots[interior]
        self.plus_faces, self.minus_faces = faces[:, 0].contiguous(), faces[:, 1].contiguous()
        self.plus_vertices = mesh.faces[self.plus_faces, slots[:, 0]]
        self.minus_vertices = torch.where(self.minus_faces >= 0, mesh.faces[self.minus_faces.clamp_min(0),
                                                                           slots[:, 1].clamp_min(0)], -1)
        self.edge_lengths = torch.linalg.vector_norm(
            mesh.vertices[self.edges[:, 1]]-mesh.vertices[self.edges[:, 0]], dim=-1)
        boundary = topology.edges[topology.kind != INTERIOR]
        self.boundary_edges = boundary
        self.boundary_half_functions = boundary_half_functions
        self.metadata = {'basis': 'RWG', 'basis_id': ('open_rwg_with_boundary_half_functions_v1' if
                         boundary_half_functions else 'open_rwg_interior_edges_v1'), 'test_space': 'RWG',
                         'edge_orientation': 'ascending_vertex_index', 'boundary_condition':
                         'zero_normal_current_on_boundary_edges', 'boundary_edges': len(boundary),
                         'basis_units': '1', 'J_coefficient_units': 'A/m', 'metric': 'surface_L2_proxy',
                         'metric_units': 'm^2'}

    @property
    def divergence(self):
        """Piecewise-constant surface divergence [T, n_edges] in 1/m."""
        result = torch.zeros((len(self.mesh.faces), self.n_edges), device=self.mesh.device, dtype=self.mesh.dtype)
        values = self.face_signs*self.edge_lengths[self.face_edges]/self.mesh.areas[:, None]
        return result.scatter_add(1, self.face_edges, values)

    def evaluate(self, points):
        result = torch.zeros((*points.shape[:2], self.n_edges, 3), device=points.device, dtype=points.dtype)
        indices = self.face_edges[:, None, :, None].expand(-1, points.shape[1], -1, 3)
        return result.scatter_add(2, indices, self.local_values(points))

    def assert_current(self):
        version = (id(self.mesh.vertices), self.mesh.vertices._version, id(self.mesh.faces), self.mesh.faces._version)
        if version != self._mesh_version:
            raise ValueError('Open surface geometry changed after basis construction')


def _structured_faces(nx, ny, device):
    """Union-jack triangulation of an nx-by-ny vertex-row grid of rectangles."""
    i, j = torch.meshgrid(torch.arange(nx, device=device), torch.arange(ny, device=device), indexing='ij')
    a = i*(ny+1)+j
    b, c, d = a+ny+1, a+ny+2, a+1
    flip = ((i+j) % 2).bool()
    first = torch.where(flip[..., None], torch.stack((a, b, d), -1), torch.stack((a, b, c), -1))
    second = torch.where(flip[..., None], torch.stack((b, c, d), -1), torch.stack((a, c, d), -1))
    return torch.stack((first, second), 2).reshape(-1, 3)


def rectangular_plate_mesh(size, cells, *, device='cuda', dtype=torch.float64):
    """Plate [-a/2,a/2]x[-b/2,b/2] at z=0 with +z normal and alternating diagonals."""
    device = _cuda(device)
    (a, b), (nx, ny) = size, cells
    if not all(math.isfinite(v) and v > 0 for v in (a, b)):
        raise ValueError('Plate side lengths must be positive and finite')
    if any(type(n) is not int or n < 1 for n in (nx, ny)):
        raise ValueError('Plate cell counts must be positive integers')
    x = torch.linspace(-a/2, a/2, nx+1, device=device, dtype=dtype)
    y = torch.linspace(-b/2, b/2, ny+1, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    vertices = torch.stack((xx, yy, torch.zeros_like(xx)), -1).reshape(-1, 3)
    return OpenSurfaceMesh(vertices, _structured_faces(nx, ny, device))


def disk_mesh(radius, rings, *, device='cuda', dtype=torch.float64):
    """Disk of radius ``radius`` at z=0 with +z normal: ring i has 6i vertices.

    Consecutive rings are zipped by angle, so triangles stay close to
    equilateral; the polygonal rim inscribes the circle.
    """
    device = _cuda(device)
    if not math.isfinite(radius) or radius <= 0 or type(rings) is not int or rings < 1:
        raise ValueError('Disk radius must be positive and finite and rings a positive integer')
    points, faces, previous = [(0.0, 0.0)], [], [(0, 0.0)]
    for ring in range(1, rings+1):
        count, r = 6*ring, radius*ring/rings
        current = [(len(points)+n, 2*math.pi*n/count) for n in range(count)]
        points.extend((r*math.cos(t), r*math.sin(t)) for _, t in current)
        if ring == 1:
            faces.extend((0, current[n][0], current[(n+1) % count][0]) for n in range(count))
        else:
            inner, outer = previous+[(previous[0][0], 2*math.pi)], current+[(current[0][0], 2*math.pi)]
            p = q = 0
            while p < len(inner)-1 or q < len(outer)-1:
                advance_outer = p == len(inner)-1 or (q < len(outer)-1 and outer[q+1][1] <= inner[p+1][1])
                if advance_outer:
                    faces.append((inner[p][0], outer[q][0], outer[q+1][0]))
                    q += 1
                else:
                    faces.append((inner[p][0], outer[q][0], inner[p+1][0]))
                    p += 1
        previous = current
    vertices = torch.tensor([(x, y, 0.0) for x, y in points], device=device, dtype=dtype)
    return OpenSurfaceMesh(vertices, torch.tensor(faces, device=device, dtype=torch.long))


def _cells(length, spacing):
    """Smallest cell count with spacing <= ``spacing``, tolerating float32 primitive sizes."""
    return max(1, math.ceil(length/spacing*(1-1e-6)))


def compile_open_surface(geometry, *, device, dtype, max_edge_length_m=None):
    """Mesh a zero-thickness Box/Cylinder or adopt an explicit surface Mesh."""
    from witwin.core import Box, Cylinder, Mesh
    if isinstance(geometry, Mesh):
        if geometry.recenter or geometry.surface_thickness is not None:
            raise ValueError("MoM sheet Mesh requires fill_mode='surface', recenter=False and no surface_thickness")
        vertices = geometry._transform_mesh_verts(geometry.vertices.to(device=device, dtype=dtype)
                                                  * geometry.scale.to(device=device, dtype=dtype))
        return OpenSurfaceMesh(vertices, geometry.faces.to(device=device, dtype=torch.long))
    if max_edge_length_m is None:
        raise ValueError('Zero-thickness Box/Cylinder sheets require OpenSurfaceConfig(max_edge_length_m=...)')
    if isinstance(geometry, Box):
        size = geometry.size.to(device=device, dtype=dtype)
        normal = int((size == 0).nonzero()[0])
        axes = [axis for axis in range(3) if axis != normal]
        if normal == 1:
            axes = axes[::-1]  # keep (u, v, normal) right handed
        sides = tuple(float(size[axis]) for axis in axes)
        cells = tuple(_cells(side, max_edge_length_m) for side in sides)
        local = rectangular_plate_mesh(sides, cells, device=device, dtype=dtype)
    elif isinstance(geometry, Cylinder):
        radius = float(geometry.radius)
        local = disk_mesh(radius, _cells(radius, max_edge_length_m), device=device, dtype=dtype)
        normal = {'x': 0, 'y': 1, 'z': 2}[geometry.axis]
        axes = [(normal+1) % 3, (normal+2) % 3]
    else:
        raise ValueError('MoM sheets support zero-thickness Box/Cylinder or explicit surface Mesh geometry')
    coordinates = torch.zeros_like(local.vertices)
    coordinates[:, axes[0]], coordinates[:, axes[1]] = local.vertices[:, 0], local.vertices[:, 1]
    return OpenSurfaceMesh(geometry._transform_mesh_verts(coordinates), local.faces)
