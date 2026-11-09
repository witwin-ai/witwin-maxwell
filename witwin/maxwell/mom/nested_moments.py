"""Direct RWG interpolation moments with leaf integration and nested transfers."""
import torch
from .interpolation import InterpolationBox
from .rwg_moments import RWGInterpolationMoments, SystemInterpolationMoments, moment_quadrature_receipt
from .nearfield import ScalarSystemBlocks
from .operator import PreparedRWGGeometry


class NestedRWGMoments:
    """Single-body or global system projection, with xyz/divergence channels.

    Only leaves integrate RWGs. Internal moments are aggregated through real
    cardinal transfers. Downward is the Hermitian adjoint of the combined
    all-node upward map; it does not assemble expanded internal RWG bases.
    """
    def __init__(self, basis, tree, *, order=4, quadrature_order=None, face_batch=32, geometry=None):
        tree.assert_current()
        system = isinstance(basis, ScalarSystemBlocks)
        if system:
            basis.assert_current()
            mesh, size = basis.system, basis.system.total_edges
            if geometry is not None:
                raise ValueError('System projections share provider geometry directly')
        else:
            mesh, size = basis.mesh, basis.n_edges
            if basis._mesh_version != (id(mesh.vertices), mesh.vertices._version, id(mesh.faces), mesh.faces._version):
                raise ValueError('Basis geometry changed; rebuild before nested moments')
        if tree.size != size or tree.device != mesh.device or tree.lower.dtype != mesh.dtype:
            raise ValueError('Moment tree must match the full RWG space')
        if not torch.equal(tree.permutation.sort().values, torch.arange(tree.size, device=tree.device)):
            raise ValueError('Moment tree permutation must cover every RWG edge once')
        self.basis, self.tree = basis, tree
        self.device = tree.device
        self.dtype = torch.complex128 if mesh.dtype == torch.float64 else torch.complex64
        self.starts = tuple(tree.starts.cpu().tolist())
        self.stops = tuple(tree.stops.cpu().tolist())
        self.children = tuple(tuple(c for c in row if c >= 0) for row in tree.children.cpu().tolist())
        self.boxes = tuple(InterpolationBox(tree.lower[n], tree.upper[n], order) for n in range(tree.node_count))
        self.geometry = basis if system else (PreparedRWGGeometry(basis) if geometry is None else geometry)
        if not system:
            self.geometry.assert_current(basis)
        leaves, transfers = {}, {}
        for node, children in enumerate(self.children):
            if children:
                for child in children:
                    transfers[child] = self.boxes[node].transfer_to(self.boxes[child]).to(self.dtype)
            else:
                edges = tree.permutation[self.starts[node]:self.stops[node]]
                if system:
                    leaves[node] = SystemInterpolationMoments(basis, edges, self.boxes[node],
                        quadrature_order=quadrature_order, face_batch=face_batch)
                else:
                    leaves[node] = RWGInterpolationMoments(basis, edges, self.boxes[node],
                        quadrature_order=quadrature_order, face_batch=face_batch, geometry=self.geometry)
        self.leaves, self.transfers = leaves, transfers
        self._binding = (id(basis), id(tree), id(self.geometry), self.device, self.dtype,
                         self.starts, self.stops, self.children, tuple(map(id, self.boxes)),
                         tuple((n, id(v)) for n, v in leaves.items()))
        self._versions = tuple((n, id(v), v._version) for n, v in transfers.items())

    def assert_current(self):
        if ((id(self.basis), id(self.tree), id(self.geometry), self.device, self.dtype,
             self.starts, self.stops, self.children, tuple(map(id, self.boxes)),
             tuple((n, id(v)) for n, v in self.leaves.items())) != self._binding
                or tuple((n, id(v), v._version) for n, v in self.transfers.items()) != self._versions):
            raise ValueError('Nested RWG moments changed')
        self.tree.assert_current()
        if isinstance(self.basis, ScalarSystemBlocks):
            self.geometry.assert_current()
        else:
            self.geometry.assert_current(self.basis)
        for box in self.boxes:
            box.assert_current()
        for leaf in self.leaves.values():
            leaf.assert_current()

    def _validate(self, value, shape):
        if (not isinstance(value, torch.Tensor) or value.ndim < len(shape)
                or tuple(value.shape[-len(shape):]) != shape or value.device != self.device
                or value.dtype != self.dtype or not bool(torch.isfinite(value).all())):
            raise ValueError('Nested moments require matching finite complex CUDA values')

    def upward(self, currents):
        self.assert_current()
        self._validate(currents, (self.tree.size,))
        values = [None]*self.tree.node_count
        for node in reversed(range(self.tree.node_count)):
            if node in self.leaves:
                leaf = self.leaves[node]
                local = currents[..., leaf.edges]
                values[node] = torch.cat((torch.einsum('...e,eac->...ac', local, leaf.vector),
                                           (local@leaf.divergence)[..., None]), dim=-1)
            else:
                values[node] = sum(torch.einsum('ap,...ac->...pc', self.transfers[c], values[c])
                                   for c in self.children[node])
        return tuple(values)

    def downward(self, coefficients):
        self.assert_current()
        if not isinstance(coefficients, (tuple, list)) or len(coefficients) != self.tree.node_count:
            raise ValueError('Downward requires one coefficient array per tree node')
        batch = None
        for node, value in enumerate(coefficients):
            self._validate(value, (len(self.boxes[node].nodes), 4))
            if batch is None:
                batch = value.shape[:-2]
            elif value.shape[:-2] != batch:
                raise ValueError('Downward node batches must agree')
        values = list(coefficients)
        result = coefficients[0].new_zeros((*batch, self.tree.size))
        for node, children in enumerate(self.children):
            if children:
                for child in children:
                    values[child] = values[child]+torch.einsum('ap,...pc->...ac', self.transfers[child].conj(), values[node])
            else:
                leaf = self.leaves[node]
                local = (torch.einsum('eac,...ac->...e', leaf.vector.conj(), values[node][..., :3])
                         +values[node][..., 3]@leaf.divergence.mH)
                result = result.index_add(-1, leaf.edges, local)
        return result

    def moment_quadrature_receipt(self):
        self.assert_current()
        records = [moment_quadrature_receipt(leaf.box, leaf.quadrature_order) for leaf in self.leaves.values()]
        return {'orders': sorted({r['quadrature_order'] for r in records}),
                'required_orders': sorted({r['required_quadrature_order'] for r in records}),
                'polynomial_exact_in_exact_arithmetic': all(r['polynomial_exact_in_exact_arithmetic'] for r in records),
                'floating_point_certified': False}

    def receipt(self):
        self.assert_current()
        tensors = [t for b in self.boxes for t in b.tensors()]
        tensors += list(self.transfers.values())
        tensors += [t for leaf in self.leaves.values() for t in leaf.tensors()]
        return {'version': 'nested_rwg_interpolation_moments_v1', 'nodes': self.tree.node_count,
                'integrated_leaves': len(self.leaves), 'internal_rwg_matrices_retained': False,
                'owned_retained_bytes': sum(t.numel()*t.element_size() for t in tensors),
                'moment_quadrature': self.moment_quadrature_receipt(),
                'shared_geometry_and_tree_included': False, 'physical_qualification': False}
