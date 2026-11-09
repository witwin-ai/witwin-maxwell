"""Fixed native-CUDA leaf-block inverse for original scalar surface operators."""
import torch
from .nearfield import ScalarSystemBlocks
from .cluster import SupportOctree


class LeafBlockInverse:
    """Disjoint original principal blocks, with bounded edge count per factor.

    Oversized geometric leaves are split explicitly. LU factors represent a
    preconditioner only, never a replacement for the original physical operator.
    The explicit primary mode omits periodic image corrections from factors,
    while retaining the full original provider binding for physical admission.
    The budget covers owned factors, pivots and coefficient indices, not setup
    workspaces, provider caches or solver state.
    """
    def __init__(self, provider, tree, *, max_block_edges=32, retained_budget_bytes=256*1024**2,
                 factor_kernel='original'):
        if not isinstance(provider, ScalarSystemBlocks) or not isinstance(tree, SupportOctree):
            raise TypeError('Leaf inverse requires scalar original blocks and a support tree')
        for name, value in (('max_block_edges', max_block_edges), ('retained_budget_bytes', retained_budget_bytes)):
            if type(value) is not int or value < 1:
                raise ValueError(name+' must be a positive integer')
        if factor_kernel not in ('original', 'primary'):
            raise ValueError('factor_kernel must be original or primary')
        provider.assert_current()
        tree.assert_current()
        n = provider.system.total_edges
        if tree.device != provider.system.device or tree.size != n or not torch.equal(
                tree.permutation.sort().values, torch.arange(n, device=tree.device)):
            raise ValueError('Leaf inverse tree must cover the original coefficient space')
        leaves = (tree.children < 0).all(dim=1).nonzero().flatten()
        ranges = torch.stack((tree.starts[leaves], tree.stops[leaves]), -1).cpu().tolist()
        ranges = sorted((a, b) for start, stop in ranges
                        for a in range(start, stop, max_block_edges)
                        for b in (min(a+max_block_edges, stop),))
        if not ranges or ranges[0][0] != 0 or ranges[-1][1] != n or any(
                a != previous for (_, previous), (a, _) in zip(ranges, ranges[1:])):
            raise ValueError('Leaf ranges must partition all original edges exactly once')
        dtype = torch.complex128 if provider.system.dtype == torch.float64 else torch.complex64
        complex_bytes = torch.empty((), dtype=dtype).element_size()
        planned = sum((2*(b-a))**2*complex_bytes+2*(b-a)*(4+8) for a, b in ranges)
        if planned > retained_budget_bytes:
            raise MemoryError(f'Leaf inverse retained bytes {planned} exceed budget {retained_budget_bytes}')
        self.provider, self.tree = provider, tree
        self.factor_kernel = factor_kernel
        self.shape, self.device, self.dtype = (2*n, 2*n), tree.device, dtype
        self.indices, self.factors, self.pivots = [], [], []
        for start, stop in ranges:
            edges = tree.permutation[start:stop]
            block = (provider(edges, edges) if factor_kernel == 'original' else
                     ScalarSystemBlocks._regional_selection(provider, edges, edges, include_interiors=True))
            if block.requires_grad or not bool(torch.isfinite(block).all()):
                raise ValueError('Leaf inverse requires finite fixed original blocks')
            factor, pivot, info = torch.linalg.lu_factor_ex(block)
            if int(info) != 0 or not bool(torch.isfinite(factor).all()):
                raise ValueError('Original leaf block is singular or unresolved')
            self.indices.append(torch.cat((edges, edges+n)))
            self.factors.append(factor)
            self.pivots.append(pivot)
        self.indices, self.factors, self.pivots = tuple(self.indices), tuple(self.factors), tuple(self.pivots)
        self.retained_bytes = sum(t.numel()*t.element_size() for t in self.tensors())
        if self.retained_bytes != planned:
            raise RuntimeError('Leaf inverse retained storage differs from its plan')
        self.max_block_edges = max_block_edges
        self._binding = self._state()

    def tensors(self):
        return self.indices+self.factors+self.pivots

    def _state(self):
        return (id(self.provider), id(self.tree), self.shape, self.device, self.dtype,
                self.max_block_edges, self.retained_bytes, self.factor_kernel,
                tuple((id(t), t._version) for t in self.tensors()))

    def assert_current(self):
        if self._state() != self._binding:
            raise ValueError('Leaf inverse binding or factors changed')
        self.provider.assert_current()
        self.tree.assert_current()

    def _action(self, value, adjoint):
        self.assert_current()
        if (not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[-1] != self.shape[0]
                or value.device != self.device or value.dtype != self.dtype
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Leaf inverse requires matching finite CUDA coefficient vectors')
        flat = value.reshape(-1, self.shape[0])
        result = torch.empty_like(flat)
        for indices, factor, pivot in zip(self.indices, self.factors, self.pivots):
            solved = torch.linalg.lu_solve(factor, pivot, flat[:, indices].mT.contiguous(), adjoint=adjoint)
            result[:, indices] = solved.mT
        if not bool(torch.isfinite(result).all()):
            raise ValueError('Nonfinite leaf inverse action')
        return result.reshape(value.shape)

    def apply(self, value):
        return self._action(value, False)

    def adjoint(self, value):
        return self._action(value, True)

    def receipt(self):
        self.assert_current()
        return {'version': 'scalar_leaf_block_inverse_v2', 'factor_kernel': self.factor_kernel,
            'factor_scope': ('original_principal_blocks' if self.factor_kernel == 'original'
                             else 'primary_free_space_exterior_plus_body_interiors'), 'blocks': len(self.indices),
            'max_block_edges': self.max_block_edges, 'retained_bytes': self.retained_bytes,
            'scope': 'owned_lu_pivots_and_indices', 'peak_memory_qualified': False,
            'physical_qualification': False}


class RightPreconditionedOperator:
    """Internal A P Krylov coordinates; recover physical currents with P.apply.

    Original residual admission must use the recovered current on the underlying
    operator. The adapter does not redefine the physical operator or its identity.
    """
    supports_direct = False

    def __init__(self, operator, inverse):
        from .operator_protocol import IterativeState
        operator.refresh()
        inverse.assert_current()
        if (inverse.provider is not operator.provider or inverse.shape != operator.shape
                or inverse.device != operator.device or inverse.dtype != operator.dtype):
            raise ValueError('Right inverse must bind the same original operator space')
        self.operator, self.inverse = operator, inverse
        self.shape, self.device, self.dtype = operator.shape, operator.device, operator.dtype
        self.iterative_state = IterativeState()
        self._binding = (id(operator), id(inverse), self.shape, self.device, self.dtype)

    def refresh(self):
        if (id(self.operator), id(self.inverse), self.shape, self.device, self.dtype) != self._binding:
            raise ValueError('Right-preconditioned operator binding changed')
        self.operator.refresh()
        self.inverse.assert_current()
        if self.inverse.provider is not self.operator.provider:
            raise ValueError('Right inverse original reference changed')

    def read_access(self):
        from contextlib import contextmanager
        @contextmanager
        def access():
            self.refresh()
            with self.operator.read_access():
                yield
        return access()

    def apply(self, value):
        with self.read_access():
            return self.operator.apply(self.inverse.apply(value))

    def adjoint(self, value):
        with self.read_access():
            return self.inverse.adjoint(self.operator.adjoint(value))

    def riesz_inverse(self, residual):
        with self.read_access():
            return self.operator.riesz_inverse(residual)

    def preconditioner_scales(self, kind):
        with self.read_access():
            if kind not in ('none', 'rwg_diagonal'):
                raise ValueError('Right block inverse supports none or RWG left scaling')
            left, _, _ = self.operator.preconditioner_scales(kind)
            # P already maps equation coordinates to physical currents. Use
            # similarity scaling L (A P) L^-1, preserving an identity block.
            right = left.reciprocal()
            if not bool(torch.isfinite(right).all()):
                raise ValueError('Right block coordinate scaling exceeds native precision')
            return left, right, False

    def persistence_receipt(self):
        self.refresh()
        return {'version': 'right_original_leaf_inverse_view_v1',
                'inverse': self.inverse.receipt(), 'executable_snapshot': False}

    def operator_stats(self):
        return self.persistence_receipt()
