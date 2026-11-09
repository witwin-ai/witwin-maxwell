"""Geometry-only RWG vector/divergence projection for direct far kernels."""
import torch
from .interpolation import InterpolationBox
from .operator import PreparedRWGGeometry
from .nearfield import _selected_edges, _local_indices
from .quadrature import triangle_rule


def required_moment_quadrature_order(box):
    """Gauss order exact for cardinal polynomials times a linear RWG.

    Total Cartesian degree is sum(axis degrees)+1; the Duffy Jacobian adds
    one to the radial degree. Each Gauss coordinate integrates degree 2*q-1.
    """
    box.assert_current()
    degree = sum(len(axis)-1 for axis in box.coordinates)
    return (degree+4)//2


def moment_quadrature_receipt(box, order):
    required = required_moment_quadrature_order(box)
    return {'quadrature_order': order, 'required_quadrature_order': required,
            'polynomial_exact_in_exact_arithmetic': order >= required,
            'floating_point_certified': False}


class RWGInterpolationMoments:
    def __init__(self, basis, edges, box, *, quadrature_order=None, face_batch=32, geometry=None):
        if not isinstance(box, InterpolationBox):
            raise TypeError('RWG moments require an interpolation box')
        quadrature_order = required_moment_quadrature_order(box) if quadrature_order is None else quadrature_order
        if any(type(v) is not int or v < 1 for v in (quadrature_order, face_batch)):
            raise ValueError('Moment quadrature and face batch must be positive integers')
        box.assert_current()
        if basis.mesh.device != box.device or basis.mesh.dtype != box.dtype:
            raise ValueError('RWG geometry and interpolation box must share device and precision')
        mesh = basis.mesh
        if basis._mesh_version != (id(mesh.vertices), mesh.vertices._version, id(mesh.faces), mesh.faces._version):
            raise ValueError('Basis geometry changed; rebuild before RWG moment integration')
        selected, positions, faces = _selected_edges(basis, edges)
        geometry = PreparedRWGGeometry(basis) if geometry is None else geometry
        geometry.assert_current(basis)
        # Protect complete supports, not only interpolation/quadrature samples.
        box.validate_points(geometry.triangles[faces])
        self.basis, self.box, self.geometry = basis, box, geometry
        self.edges = edges.clone()
        self.quadrature_order, self.face_batch = quadrature_order, face_batch
        self.device = box.device
        self.dtype = torch.complex128 if box.dtype == torch.float64 else torch.complex64
        vector = torch.zeros((len(edges), len(box.nodes), 3), device=self.device, dtype=box.dtype)
        divergence = torch.zeros((len(edges), len(box.nodes)), device=self.device, dtype=box.dtype)
        for current in faces.split(face_batch):
            triangles = geometry.triangles[current]
            points, weights = triangle_rule(triangles, quadrature_order)
            cardinal = box.evaluate(points)
            slopes = geometry.slopes[current]
            rwg = slopes[:, None, :, None]*(points[:, :, None]-triangles[:, None])
            local_vector = torch.einsum('bqec,bqa,bq->beac', rwg, cardinal, weights)
            local_divergence = 2*slopes[:, :, None]*torch.einsum('bqa,bq->ba', cardinal, weights)[:, None, :]
            indices, valid = _local_indices(basis.face_edges[current], selected, positions)
            vector.index_add_(0, indices[valid], local_vector[valid])
            divergence.index_add_(0, indices[valid], local_divergence[valid])
        self.vector, self.divergence = vector.to(self.dtype), divergence.to(self.dtype)
        self._binding = (id(basis), id(box), id(geometry), quadrature_order, face_batch, self.device, self.dtype)
        self._versions = tuple((id(t), t._version) for t in self.tensors())

    def tensors(self):
        return self.edges, self.vector, self.divergence

    def assert_current(self):
        if ((id(self.basis), id(self.box), id(self.geometry), self.quadrature_order, self.face_batch,
             self.device, self.dtype) != self._binding
                or tuple((id(t), t._version) for t in self.tensors()) != self._versions):
            raise ValueError('RWG interpolation moments changed')
        self.box.assert_current()
        self.geometry.assert_current(self.basis)

    def receipt(self):
        self.assert_current()
        return {'version': 'rwg_vector_divergence_moments_v1', 'edges': len(self.edges),
                'interpolation_nodes': len(self.box.nodes), 'quadrature_order': self.quadrature_order,
                'maximum_face_batch': self.face_batch,
                'owned_retained_bytes': sum(t.numel()*t.element_size() for t in self.tensors()),
                'frequency_independent_geometry': True, 'physical_qualification': False,
                'moment_quadrature': moment_quadrature_receipt(self.box, self.quadrature_order)}


class SystemInterpolationMoments:
    """One global leaf, preserving edge order across disjoint body bases."""
    def __init__(self, provider, edges, box, *, quadrature_order=None, face_batch=32):
        provider.assert_current()
        provider._validate_selection(edges)
        quadrature_order = required_moment_quadrature_order(box) if quadrature_order is None else quadrature_order
        if any(type(v) is not int or v < 1 for v in (quadrature_order, face_batch)):
            raise ValueError('Moment quadrature and face batch must be positive integers')
        self.quadrature_order, self.face_batch = quadrature_order, face_batch
        self.provider, self.box = provider, box
        self.edges = edges.clone()
        self.dtype = torch.complex128 if box.dtype == torch.float64 else torch.complex64
        self.vector = torch.zeros((len(edges), len(box.nodes), 3), device=box.device, dtype=self.dtype)
        self.divergence = torch.zeros((len(edges), len(box.nodes)), device=box.device, dtype=self.dtype)
        for index, selected in provider._body_selection(edges):
            body, geometry = provider.system.bodies[index], provider.geometry[index]
            local = RWGInterpolationMoments(body.basis, edges[selected]-body.edge_offset, box,
                quadrature_order=quadrature_order, face_batch=face_batch, geometry=geometry)
            self.vector[selected] = local.vector
            self.divergence[selected] = local.divergence
        self._binding = (id(provider), id(box), self.dtype, self.quadrature_order, self.face_batch)
        self._versions = tuple((id(t), t._version) for t in self.tensors())

    def tensors(self):
        return self.edges, self.vector, self.divergence

    def assert_current(self):
        if ((id(self.provider), id(self.box), self.dtype, self.quadrature_order, self.face_batch) != self._binding
                or tuple((id(t), t._version) for t in self.tensors()) != self._versions):
            raise ValueError('System interpolation moments changed')
        self.provider.assert_current()
        self.box.assert_current()
