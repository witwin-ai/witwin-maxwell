"""Exact nested RWG transfer for planar midpoint surface refinement."""
import torch
from .mesh import RWGBasis
from .operator import local_geometry


class RWGRefinement:
    """One uniform refinement with three sparse coefficients per child edge.

    The transfer preserves the piecewise-linear tangential field and divergence,
    without projecting the geometry or solving a mass system. It is a fixed-mesh
    current map; geometry differentiation is unsupported.
    """
    def __init__(self, coarse):
        if not isinstance(coarse, RWGBasis):
            raise TypeError('RWG refinement requires an RWGBasis')
        if coarse.mesh.vertices.requires_grad:
            raise NotImplementedError('RWG refinement requires fixed geometry')
        if coarse._mesh_version != (id(coarse.mesh.vertices), coarse.mesh.vertices._version,
                                   id(coarse.mesh.faces), coarse.mesh.faces._version):
            raise ValueError('Coarse basis geometry changed')
        self.coarse = coarse
        self.fine = RWGBasis(coarse.mesh.refine())
        fine = self.fine
        self.parent_faces = torch.arange(len(coarse.mesh.faces), device=coarse.mesh.device).repeat_interleave(4)
        parents = self.parent_faces[fine.plus_faces]
        self.columns = coarse.face_edges[parents].clone()
        midpoint = fine.mesh.vertices[fine.edges].mean(1)
        tangent = fine.mesh.vertices[fine.edges[:, 1]]-fine.mesh.vertices[fine.edges[:, 0]]
        tangent = tangent/fine.edge_lengths[:, None]
        conormal = torch.linalg.cross(tangent, fine.mesh.normals[fine.plus_faces])
        fine_value = ((midpoint-fine.mesh.vertices[fine.plus_vertices])*
                      (fine.edge_lengths/(2*fine.mesh.areas[fine.plus_faces]))[:, None])
        normalization = (fine_value*conormal).sum(-1)
        _, free, slopes = local_geometry(coarse)
        values = slopes[parents, :, None]*(midpoint[:, None]-free[parents])
        self.weights = (values*conormal[:, None]).sum(-1)/normalization[:, None]
        if not bool(torch.isfinite(self.weights).all()):
            raise ValueError('Unresolved RWG refinement transfer')
        self._binding = self._state()

    def _state(self):
        tensors = [self.columns, self.weights, self.parent_faces]
        for basis in (self.coarse, self.fine):
            tensors += [basis.mesh.vertices, basis.mesh.faces, basis.edges, basis.face_edges,
                        basis.face_signs, basis.edge_lengths, basis.plus_faces, basis.minus_faces,
                        basis.plus_vertices, basis.minus_vertices]
        return (id(self.coarse), id(self.fine), id(self.coarse.mesh), id(self.fine.mesh),
                tuple((id(t), t._version) for t in tensors))

    def assert_current(self):
        if self._state() != self._binding:
            raise ValueError('RWG refinement geometry or transfer changed')

    def _validate(self, value, size):
        self.assert_current()
        real = self.coarse.mesh.dtype
        complex_dtype = torch.complex128 if real == torch.float64 else torch.complex64
        if (not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[-1] != size
                or value.device != self.coarse.mesh.device or value.dtype not in (real, complex_dtype)
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Refinement currents must have matching finite CUDA coefficient space')

    def apply(self, value):
        self._validate(value, self.coarse.n_edges)
        return (value[..., self.columns]*self.weights).sum(-1)

    def adjoint(self, value):
        self._validate(value, self.fine.n_edges)
        weighted = (value[..., :, None]*self.weights).flatten(-2)
        indices = self.columns.flatten().expand(weighted.shape)
        return value.new_zeros((*value.shape[:-1], self.coarse.n_edges)).scatter_add(-1, indices, weighted)
