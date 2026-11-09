"""Leaf-integrated RWG Cartesian moments with exact polynomial tree shifts."""
import torch
from .cluster import support_spheres
from .fmm_expansion import multi_indices, monomials, shift_matrix
from .nearfield import ScalarSystemBlocks, _selected_edges, _local_indices
from .operator import PreparedRWGGeometry
from .quadrature import triangle_rule
from .fmm_shift import SeparableMultipoleShift,shift_storage_bytes


def rwg_polynomial_moments(basis, edges, center, order, *, geometry=None,
                           quadrature_order=None, face_batch=32):
    """Integrate vector/divergence RWGs against (x-center)**alpha."""
    return _rwg_weighted_moments(basis, edges, center, order, geometry=geometry,
        quadrature_order=quadrature_order, face_batch=face_batch)


def _rwg_weighted_moments(basis, edges, center, order, *, geometry=None,
                          quadrature_order=None, face_batch=32, phase=None):
    """Shared CUDA face integration; phase is a prevalidated (signed k, u)."""
    indices = multi_indices(order)
    required = (order+4)//2
    q = required if quadrature_order is None else quadrature_order
    if any(type(v) is not int or v < 1 for v in (q, face_batch)):
        raise ValueError('Multipole quadrature and face batch must be positive integers')
    mesh = basis.mesh
    if basis._mesh_version != (id(mesh.vertices), mesh.vertices._version, id(mesh.faces), mesh.faces._version):
        raise ValueError('RWG basis geometry changed before multipole integration')
    if (not isinstance(center, torch.Tensor) or center.shape != (3,) or center.device != mesh.device
            or center.dtype != mesh.dtype or center.requires_grad or not bool(torch.isfinite(center).all())):
        raise ValueError('RWG multipole center must match fixed mesh geometry')
    geometry = PreparedRWGGeometry(basis) if geometry is None else geometry
    geometry.assert_current(basis)
    selected, positions, faces = _selected_edges(basis, edges)
    accumulation_dtype = mesh.dtype if phase is None else torch.complex128
    vector = torch.zeros((len(edges), len(indices), 3), device=mesh.device, dtype=accumulation_dtype)
    divergence = torch.zeros((len(edges), len(indices)), device=mesh.device, dtype=accumulation_dtype)
    for current in faces.split(face_batch):
        triangles = geometry.triangles[current]
        points, weights = triangle_rule(triangles, q)
        polynomial = monomials(points-center, order)
        slopes = geometry.slopes[current]
        rwg = slopes[:, None, :, None]*(points[:, :, None]-triangles[:, None])
        if phase is not None:
            wave, direction = phase
            carrier = torch.exp(1j*wave*((points-center)@direction))
            if not bool(torch.isfinite(carrier).all() & (carrier.abs() > 0).all()):
                raise ValueError('Directional RWG carrier exceeds native precision')
            polynomial = polynomial*carrier[..., None]
            rwg, weights = rwg.to(accumulation_dtype), weights.to(accumulation_dtype)
        local_vector = torch.einsum('bqec,bqa,bq->beac', rwg, polynomial, weights)
        local_divergence = 2*slopes[:, :, None]*torch.einsum('bqa,bq->ba', polynomial, weights)[:, None, :]
        mapping, valid = _local_indices(basis.face_edges[current], selected, positions)
        vector.index_add_(0, mapping[valid], local_vector[valid])
        divergence.index_add_(0, mapping[valid], local_divergence[valid])
    dtype = torch.complex128 if mesh.dtype == torch.float64 else torch.complex64
    if not bool(torch.isfinite(vector).all() & torch.isfinite(divergence).all()):
        raise ValueError('RWG multipole moments exceed native precision')
    return vector.to(dtype), divergence.to(dtype)


class NestedRWGMultipoles:
    """Only leaves integrate; M2M aggregates and L2L projects local coefficients.

    A projection may span one basis or a global multi-body scalar provider.
    Downward is the Hermitian adjoint of the combined all-node upward map.
    """
    def __init__(self, basis, tree, *, order=4, quadrature_order=None, face_batch=32,
                 geometry=None, retained_budget_bytes=1024**3,shift_storage='dense'):
        count = len(multi_indices(order))
        q = (order+4)//2 if quadrature_order is None else quadrature_order
        if any(type(v) is not int or v < 1 for v in (q, face_batch, retained_budget_bytes)):
            raise ValueError('Multipole quadrature, batch and retained budget must be positive integers')
        tree.assert_current()
        global_space = isinstance(basis, ScalarSystemBlocks)
        if global_space:
            basis.assert_current()
            mesh, size = basis.system, basis.system.total_edges
            if geometry is not None:
                raise ValueError('Global multipoles share provider geometry')
            geometry = basis
        else:
            mesh, size = basis.mesh, basis.n_edges
            if basis._mesh_version != (id(mesh.vertices), mesh.vertices._version, id(mesh.faces), mesh.faces._version):
                raise ValueError('RWG basis geometry changed before multipole construction')
            geometry = PreparedRWGGeometry(basis) if geometry is None else geometry
            geometry.assert_current(basis)
        if (tree.size != size or tree.device != mesh.device or tree.lower.dtype != mesh.dtype
                or not torch.equal(tree.permutation.sort().values, torch.arange(size, device=tree.device))):
            raise ValueError('Multipole tree must cover the complete matching RWG space')
        self.basis, self.tree, self.geometry = basis, tree, geometry
        self.order, self.count, self.quadrature_order = order, count, q
        self.shift_storage=shift_storage
        self.device = tree.device
        self.dtype = torch.complex128 if mesh.dtype == torch.float64 else torch.complex64
        self.children = tuple(tuple(c for c in row if c >= 0) for row in tree.children.cpu().tolist())
        self.starts, self.stops = tuple(tree.starts.cpu().tolist()), tuple(tree.stops.cpu().tolist())
        real_bytes = tree.lower.element_size()
        expected = (4*tree.node_count*real_bytes+8*size+4*size*count*2*real_bytes
                    +shift_storage_bytes(order,tree.node_count,real_bytes,shift_storage))
        if expected > retained_budget_bytes:
            raise MemoryError('Multipole retained budget exceeded before moments and shifts')
        if global_space:
            support_lower, support_upper = basis.support_boxes()
        else:
            supports = torch.cat((geometry.triangles[basis.plus_faces], geometry.triangles[basis.minus_faces]), dim=1)
            support_lower, support_upper = supports.amin(1), supports.amax(1)
        for node in range(tree.node_count):
            edges = tree.permutation[self.starts[node]:self.stops[node]]
            if not bool((support_lower[edges] >= tree.lower[node]).all()
                        & (support_upper[edges] <= tree.upper[node]).all()):
                raise ValueError('Multipole tree must enclose complete RWG supports at every node')
        self.centers,self.radii = support_spheres(tree)
        if not bool(torch.isfinite(self.centers).all() & torch.isfinite(self.radii).all()):
            raise ValueError('Multipole support spheres exceed native precision')
        self.shift_layout=(SeparableMultipoleShift(order,device=self.device,dtype=tree.lower.dtype)
            if shift_storage=='separable_v1' and tree.node_count>1 else None)
        leaves, transfers = {}, {}
        for node, children in enumerate(self.children):
            if children:
                for child in children:
                    displacement=self.centers[child]-self.centers[node]
                    transfers[child] = (shift_matrix(displacement, order).to(self.dtype)
                        if shift_storage=='dense' else displacement)
                    if not bool(torch.isfinite(transfers[child]).all()):
                        raise ValueError('Multipole shift exceeds native precision')
            else:
                edges = tree.permutation[self.starts[node]:self.stops[node]].clone()
                if global_space:
                    vector = torch.zeros((len(edges), count, 3), device=self.device, dtype=self.dtype)
                    divergence = torch.zeros((len(edges), count), device=self.device, dtype=self.dtype)
                    for bi, selected in basis._body_selection(edges):
                        body = basis.system.bodies[bi]
                        v, d = rwg_polynomial_moments(body.basis, edges[selected]-body.edge_offset,
                            self.centers[node], order, geometry=basis.geometry[bi], quadrature_order=q, face_batch=face_batch)
                        vector[selected], divergence[selected] = v, d
                else:
                    vector, divergence = rwg_polynomial_moments(basis, edges, self.centers[node], order,
                        geometry=geometry, quadrature_order=q, face_batch=face_batch)
                leaves[node] = (edges, vector, divergence)
        self.leaves, self.transfers = leaves, transfers
        self._binding = (id(basis), id(tree), id(geometry), order, count, q, self.device, self.dtype,
                         self.children, self.starts, self.stops, tuple(leaves), tuple(transfers),
                         self.shift_storage,id(self.shift_layout))
        self._versions = tuple((id(t), t._version) for t in self.tensors())
        self._retained_bytes = expected
        if sum(t.numel()*t.element_size() for t in self.tensors()) != expected:
            raise RuntimeError('Multipole storage differs from retained plan')

    def tensors(self):
        return (self.centers, self.radii, *self.transfers.values(),
                *(() if self.shift_layout is None else self.shift_layout.tensors()),
                *(t for record in self.leaves.values() for t in record))

    def assert_current(self):
        if ((id(self.basis), id(self.tree), id(self.geometry), self.order, self.count, self.quadrature_order,
             self.device, self.dtype, self.children, self.starts, self.stops, tuple(self.leaves), tuple(self.transfers),
             self.shift_storage,id(self.shift_layout)) != self._binding
                or tuple((id(t), t._version) for t in self.tensors()) != self._versions):
            raise ValueError('RWG multipole projection changed')
        self.tree.assert_current()
        if self.shift_layout is not None:self.shift_layout.assert_current()
        if isinstance(self.basis, ScalarSystemBlocks):
            self.basis.assert_current()
        else:
            self.geometry.assert_current(self.basis)

    def _validate(self, value, shape):
        if (not isinstance(value, torch.Tensor) or value.ndim < len(shape)
                or tuple(value.shape[-len(shape):]) != shape or value.device != self.device
                or value.dtype != self.dtype or not bool(torch.isfinite(value).all())):
            raise ValueError('Multipole actions require matching finite complex CUDA values')

    def upward(self, currents):
        self.assert_current()
        self._validate(currents, (self.tree.size,))
        values = [None]*self.tree.node_count
        for node in reversed(range(self.tree.node_count)):
            if node in self.leaves:
                edges, vector, divergence = self.leaves[node]
                local = currents[..., edges]
                values[node] = torch.cat((torch.einsum('...e,eac->...ac', local, vector),
                                          (local@divergence)[..., None]), dim=-1)
            else:
                values[node] = sum((torch.einsum('pa,...ac->...pc', self.transfers[c], values[c])
                                   if self.shift_layout is None else self.shift_layout.apply(self.transfers[c],values[c]))
                                   for c in self.children[node])
        return tuple(values)

    def downward(self, coefficients):
        self.assert_current()
        if not isinstance(coefficients, (tuple, list)) or len(coefficients) != self.tree.node_count:
            raise ValueError('Multipole downward requires all node coefficients')
        for value in coefficients:
            self._validate(value, (self.count, 4))
            if value.shape != coefficients[0].shape:
                raise ValueError('Multipole node batches must agree')
        values = list(coefficients)
        result = coefficients[0].new_zeros((*coefficients[0].shape[:-2], self.tree.size))
        for node, children in enumerate(self.children):
            if children:
                for child in children:
                    values[child] = values[child]+(torch.einsum('pa,...pc->...ac', self.transfers[child].conj(), values[node])
                        if self.shift_layout is None else self.shift_layout.apply(self.transfers[child],values[node],adjoint=True))
            else:
                edges, vector, divergence = self.leaves[node]
                local = (torch.einsum('eac,...ac->...e', vector.conj(), values[node][..., :3])
                         +values[node][..., 3]@divergence.mH)
                result = result.index_add(-1, edges, local)
        return result

    def receipt(self):
        self.assert_current()
        record={'version': 'nested_rwg_cartesian_multipoles_v1', 'order': self.order,
                'coefficients_per_channel': self.count, 'nodes': self.tree.node_count,
                'integrated_leaves': len(self.leaves), 'internal_rwg_matrices_retained': False,
                'owned_retained_bytes': self._retained_bytes, 'shared_geometry_and_tree_included': False,
                'quadrature_order': self.quadrature_order, 'required_quadrature_order': (self.order+4)//2,
                'polynomial_exact_in_exact_arithmetic': self.quadrature_order >= (self.order+4)//2,
                'floating_point_certified': False, 'physical_qualification': False, 'peak_memory_qualified': False}
        if self.shift_storage!='dense':record.update(version='nested_rwg_cartesian_multipoles_separable_v1',shift_storage=self.shift_storage)
        return record
