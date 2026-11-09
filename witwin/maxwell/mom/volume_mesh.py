"""Owned CUDA tetrahedral volume geometry for one nonlinear body region."""
from dataclasses import dataclass
import torch
from .mesh import SurfaceMesh
from .quadrature import gauss_rule


@dataclass(frozen=True)
class VolumeMeshReport:
    region_id: int
    tetrahedra: int
    boundary_triangles: int
    coordinate_units: str = 'm'
    surface_operator_bound: bool = False


class TetrahedralVolumeMesh:
    """Positive tetrahedral complex with a validated closed embedded boundary.

    Connectivity validation is host combinatorics; determinants, quadrature and
    boundary geometry checks stay on CUDA. Association with an actual MoM body
    is a separate coupling obligation, not inferred from the integer region ID.
    """
    def __init__(self, vertices, cells, *, region_id):
        if type(region_id) is not int or region_id < 1:
            raise ValueError('A volume mesh must name a positive interior region')
        if (not isinstance(vertices, torch.Tensor) or vertices.device.type != 'cuda'
                or vertices.dtype not in (torch.float32, torch.float64) or vertices.ndim != 2
                or vertices.shape[1] != 3 or not bool(torch.isfinite(vertices).all())):
            raise ValueError('Volume vertices require finite real CUDA [V,3] coordinates')
        if vertices.requires_grad:
            raise NotImplementedError('Volume geometry gradients are outside the MoM contract')
        if (not isinstance(cells, torch.Tensor) or cells.device != vertices.device
                or cells.dtype != torch.int64 or cells.ndim != 2 or cells.shape[1] != 4 or not len(cells)):
            raise ValueError('Volume cells require matching CUDA int64 [T,4] indices')
        self.vertices, self.cells, self.region_id = vertices.clone(), cells.clone(), region_id
        self._region_identity = region_id
        host = cells.cpu().tolist()
        used, unique, incidence = set(), set(), {}
        local_faces = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
        for index, cell in enumerate(host):
            if len(set(cell)) != 4 or min(cell) < 0 or max(cell) >= len(vertices):
                raise ValueError('Invalid or repeated tetrahedral vertex index')
            key = tuple(sorted(cell))
            if key in unique:
                raise ValueError('Duplicate tetrahedral cell')
            unique.add(key)
            used.update(cell)
            for local in local_faces:
                face = tuple(cell[i] for i in local)
                sign = (-1)**sum(face[i] > face[j] for i in range(3) for j in range(i+1, 3))
                incidence.setdefault(tuple(sorted(face)), []).append((index, face, sign))
        if len(used) != len(vertices):
            raise ValueError('Volume mesh contains unused vertices')
        adjacency, boundary = [set() for _ in host], []
        for entries in incidence.values():
            if len(entries) == 1:
                boundary.append(entries[0][1])
            elif len(entries) == 2 and entries[0][2] != entries[1][2]:
                a, b = entries[0][0], entries[1][0]
                adjacency[a].add(b)
                adjacency[b].add(a)
            else:
                raise ValueError('Nonmanifold or inconsistently oriented tetrahedral face')
        visited, pending = set(), [0]
        while pending:
            index = pending.pop()
            if index not in visited:
                visited.add(index)
                pending.extend(adjacency[index]-visited)
        if len(visited) != len(host):
            raise ValueError('Each body volume must be face-connected')
        tetrahedra = self.vertices[self.cells]
        edges = tetrahedra[:, 1:]-tetrahedra[:, :1]
        determinants = torch.linalg.det(edges)
        scale = torch.linalg.vector_norm(edges, dim=-1).amax(-1)
        if bool((determinants <= 64*torch.finfo(vertices.dtype).eps*scale**3).any()):
            raise ValueError('Tetrahedra must have positive, resolved orientation and volume')
        self.volumes = determinants/6
        boundary_faces = torch.tensor(boundary, device=vertices.device, dtype=torch.int64)
        boundary_vertices, inverse = torch.unique(boundary_faces.flatten(), sorted=True, return_inverse=True)
        self.boundary_vertex_ids = boundary_vertices
        self.boundary = SurfaceMesh(self.vertices[boundary_vertices], inverse.reshape(-1, 3))
        self.report = VolumeMeshReport(region_id, len(cells), len(boundary))
        self._versions = tuple((id(t), t._version) for t in self._tensors())

    def _tensors(self):
        return (self.vertices, self.cells, self.volumes, self.boundary_vertex_ids,
                self.boundary.vertices, self.boundary.faces)

    def assert_current(self):
        if (type(self.region_id) is not int or self.region_id != self._region_identity
                or tuple((id(t), t._version) for t in self._tensors()) != self._versions):
            raise ValueError('Volume geometry changed after validation')

    def refine(self, levels=1):
        """Conforming eight-child midpoint refinement, in parent-major order.

        Shared edge midpoints preserve the boundary's planar four-child split.
        Each central octahedron uses its shortest opposite-midpoint diagonal;
        child orientation is resolved on CUDA before normal mesh validation.
        Geometry derivatives remain outside the volume mesh contract.
        """
        self.assert_current()
        if type(levels) is not int or levels < 0:
            raise ValueError('Volume refinement levels must be a nonnegative integer')
        mesh = TetrahedralVolumeMesh(self.vertices,self.cells,region_id=self.region_id)
        for _ in range(levels):
            vertices,cells = mesh.vertices,mesh.cells
            edge_indices = cells.new_tensor([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
            edges = cells[:,edge_indices]
            unique,inverse = torch.unique(edges.reshape(-1,2).sort(-1).values,dim=0,return_inverse=True)
            midpoint_ids = inverse.reshape(-1,6)+len(vertices)
            coordinates = torch.cat((vertices,(vertices[unique[:,0]]+vertices[unique[:,1]])/2))
            a,b,c,d = cells.unbind(-1)
            ab,ac,ad,bc,bd,cd = midpoint_ids.unbind(-1)
            corners = torch.stack((torch.stack((a,ab,ac,ad),-1),torch.stack((ab,b,bc,bd),-1),
                                   torch.stack((ac,bc,c,cd),-1),torch.stack((ad,bd,cd,d),-1)),dim=1)
            diagonals = cells.new_tensor([[0,5],[1,4],[2,3]])
            rings = cells.new_tensor([[1,2,4,3],[0,2,5,3],[0,1,5,4]])
            endpoints = coordinates[midpoint_ids[:,diagonals]]
            choice = (endpoints[:,:,0]-endpoints[:,:,1]).square().sum(-1).argmin(-1)
            selected = midpoint_ids.gather(1,diagonals[choice])
            ring = midpoint_ids.gather(1,rings[choice])
            center = torch.stack((selected[:,0,None].expand(-1,4),selected[:,1,None].expand(-1,4),
                                  ring,ring.roll(-1,dims=1)),dim=-1)
            children = torch.cat((corners,center),dim=1).reshape(-1,4)
            tetrahedra = coordinates[children]
            sides = tetrahedra[:,1:]-tetrahedra[:,:1]
            signed = (sides[:,0]*torch.linalg.cross(sides[:,1],sides[:,2])).sum(-1)
            children = torch.where((signed < 0)[:,None],children[:,[0,2,1,3]],children)
            mesh = TetrahedralVolumeMesh(coordinates,children,region_id=self.region_id)
        return mesh

    def quadrature(self, order=3, *, cell_indices=None):
        """Interior Duffy cubature returning [cells,nodes,3] and SI m^3 weights."""
        self.assert_current()
        if type(order) is not int or order < 2:
            raise ValueError('Tetrahedral quadrature requires at least two Gauss nodes per axis')
        cells,volumes=self.cells,self.volumes
        if cell_indices is not None:
            if (not isinstance(cell_indices,torch.Tensor) or cell_indices.device!=cells.device
                    or cell_indices.dtype!=torch.int64 or cell_indices.ndim!=1
                    or bool(((cell_indices<0)|(cell_indices>=len(cells))).any())):
                raise ValueError('Selected quadrature cells require matching CUDA int64 indices in range')
            cells,volumes=cells[cell_indices],volumes[cell_indices]
        nodes, weights = gauss_rule(order, device=self.vertices.device, dtype=self.vertices.dtype)
        u, v, w = torch.meshgrid(nodes, nodes, nodes, indexing='ij')
        barycentric = torch.stack((1-u, u*(1-v), u*v*(1-w), u*v*w), -1).reshape(-1, 4)
        reference_weights = (weights[:, None, None]*weights[None, :, None]*weights[None, None, :]*u**2*v).flatten()
        points = torch.einsum('qi,tij->tqj', barycentric, self.vertices[cells])
        return points, 6*volumes[:, None]*reference_weights


@dataclass(frozen=True)
class BoundVolumeRegion:
    """Exact boundary correspondence to one existing MoM body, not a VIE solve."""
    volume: TetrahedralVolumeMesh
    surface_system: object
    boundary_to_surface_vertices: torch.Tensor
    _mapping_version: tuple

    def assert_current(self):
        self.volume.assert_current()
        self.surface_system.assert_current()
        mapping = self.boundary_to_surface_vertices
        if (id(mapping), mapping._version) != self._mapping_version:
            raise ValueError('Volume-to-surface correspondence changed after binding')


def bind_volume_region(volume, surface_system):
    """Require a bijective boundary vertex/triangle match to the named body."""
    from .mesh import SurfaceSystem
    if not isinstance(volume, TetrahedralVolumeMesh) or not isinstance(surface_system, SurfaceSystem):
        raise TypeError('Volume binding requires a volume mesh and validated SurfaceSystem')
    volume.assert_current()
    surface_system.assert_current()
    if volume.region_id > len(surface_system.bodies):
        raise ValueError('Volume region is absent from the surface system')
    surface = surface_system.bodies[volume.region_id-1].mesh
    boundary = volume.boundary
    if (surface.device != boundary.device or surface.dtype != boundary.dtype
            or surface.vertices.shape != boundary.vertices.shape or surface.faces.shape != boundary.faces.shape):
        raise ValueError('Volume boundary and owning surface must have matching precision and topology size')
    extent = torch.linalg.vector_norm(surface.vertices.amax(0)-surface.vertices.amin(0))
    tolerance = 128*torch.finfo(surface.dtype).eps*extent
    matches = []
    for points in boundary.vertices.split(128):
        distance = (points[:, None]-surface.vertices[None]).abs().amax(-1)
        minimum, indices = distance.min(-1)
        if bool((minimum > tolerance).any()):
            raise ValueError('Volume boundary does not match its owning surface geometry')
        matches.append(indices)
    mapping = torch.cat(matches)
    if len(torch.unique(mapping)) != len(mapping):
        raise ValueError('Volume boundary correspondence is not bijective')
    faces = torch.cat((mapping[boundary.faces].sort(-1).values, surface.faces.sort(-1).values))
    unique, counts = torch.unique(faces, dim=0, return_counts=True)
    if len(unique) != len(surface.faces) or bool((counts != 2).any()):
        raise ValueError('Volume boundary triangulation differs from its owning surface')
    return BoundVolumeRegion(volume, surface_system, mapping, (id(mapping), mapping._version))
