"""Nested CUDA cluster bases with explicit child-to-parent transfers.

This algebraic construction consumes fixed seed spaces, including restrictions
of ancestor spaces. It is not a fast kernel setup algorithm or an H2 operator
by itself. Only leaf bases and transfer matrices survive construction.
"""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class NestedBasisConfig:
    max_rank: int = 64
    tolerance: float = 1e-10

    def __post_init__(self):
        if type(self.max_rank) is not int or self.max_rank < 1:
            raise ValueError('Nested max_rank must be a positive integer')
        if type(self.tolerance) not in (int, float) or not math.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError('Nested tolerance must be positive and finite')


class NestedBasisFailure(RuntimeError):
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


class NestedClusterBasis:
    """Shared cluster spaces, with component-major coefficient ordering.

    Seeds are (node_id, matrix) pairs whose rows follow the tree permutation
    within that node, repeated once per component. Returned spaces must pass
    independent reconstruction of every supplied seed at the declared tolerance.
    No seed arrays or expanded internal-node bases are retained.
    """
    def __init__(self, tree, seeds, *, components=2, dtype=torch.complex128,
                 config=NestedBasisConfig()):
        tree.assert_current()
        if type(components) is not int or components < 1 or not isinstance(config, NestedBasisConfig):
            raise ValueError('Nested basis requires positive components and its configuration')
        if dtype not in (torch.complex64, torch.complex128):
            raise ValueError('Nested basis requires complex precision')
        self.tree, self.config, self.components = tree, config, components
        self.dtype, self.device = dtype, tree.device
        self.starts = tuple(tree.starts.cpu().tolist())
        self.stops = tuple(tree.stops.cpu().tolist())
        self.children = tuple(tuple(c for c in row if c >= 0) for row in tree.children.cpu().tolist())
        seeds = tuple(seeds)
        by_node = [[] for _ in self.starts]
        for node, value in seeds:
            if type(node) is not int or not 0 <= node < tree.node_count:
                raise ValueError('Seed node is outside the cluster tree')
            if (not isinstance(value, torch.Tensor) or value.device != self.device or value.dtype != dtype
                    or value.ndim != 2 or value.shape[0] != self.width(node) or value.requires_grad
                    or not bool(torch.isfinite(value).all())):
                raise ValueError('Seeds require finite fixed CUDA matrices in their node ordering')
            by_node[node].append(value)
        leaves, transfers, indices, ranks = {}, {}, {}, [0]*tree.node_count
        local_tolerance = config.tolerance/(4*(int(tree.depth.max())+1))

        def orthogonalize(candidate, node):
            if not candidate.shape[1] or not candidate.shape[0]:
                return candidate.new_empty((candidate.shape[0], 0))
            u, singular, _ = torch.linalg.svd(candidate, full_matrices=False)
            tails = torch.cat((singular.square().flip(0).cumsum(0).flip(0).sqrt(), singular.new_zeros(1)))
            threshold = local_tolerance*float(torch.linalg.vector_norm(candidate))
            rank = int((tails <= threshold).nonzero()[0])
            if rank > config.max_rank:
                raise NestedBasisFailure('Nested cluster rank exceeds its limit',
                    {'node': node, 'required_rank': rank, 'max_rank': config.max_rank,
                     'local_threshold': threshold, 'physical_qualification': False})
            return u[:, :rank].clone()

        def build(node, inherited):
            values = [inherited, *by_node[node]]
            candidate = torch.cat(values, dim=1)
            if not self.children[node]:
                basis = orthogonalize(candidate, node)
                leaves[node] = basis
                points = tree.permutation[self.starts[node]:self.stops[node]]
                indices[node] = torch.cat([points+k*tree.size for k in range(components)])
                ranks[node] = basis.shape[1]
                return basis.mH@candidate
            projected = []
            for child in self.children[node]:
                child_candidate = candidate[self.restriction(node, child)]
                # Inherited columns come first; child-local seeds are used to
                # construct its space but do not belong to this parent's data.
                child_projected = build(child, child_candidate)
                projected.append(child_projected[:, :candidate.shape[1]])
            stacked = torch.cat(projected, dim=0)
            parent_coordinates = orthogonalize(stacked, node)
            offset = 0
            for child in self.children[node]:
                transfers[child] = parent_coordinates[offset:offset+ranks[child]].clone()
                offset += ranks[child]
            ranks[node] = parent_coordinates.shape[1]
            return parent_coordinates.mH@stacked

        build(0, torch.empty((self.width(0), 0), device=self.device, dtype=dtype))
        self.leaf_bases = tuple(leaves.get(i) for i in range(tree.node_count))
        self.leaf_indices = tuple(indices.get(i) for i in range(tree.node_count))
        self.transfers = tuple(transfers.get(i) for i in range(tree.node_count))
        self.ranks = tuple(ranks)
        self._binding = (id(tree), config, components, dtype, self.device, self.starts, self.stops, self.children, self.ranks)
        self._versions = tuple((id(t), t._version) for t in self.tensors())
        self.seed_errors = []
        for node, seed in seeds:
            reconstructed = self.expand(node, self.project(node, seed))
            error = float(torch.linalg.vector_norm(seed-reconstructed))
            norm = float(torch.linalg.vector_norm(seed))
            receipt = {'node': node, 'absolute_error': error, 'seed_norm': norm,
                       'relative_error': error/norm if norm else (0. if error == 0 else None)}
            self.seed_errors.append(receipt)
            if not math.isfinite(error) or error > config.tolerance*norm:
                raise NestedBasisFailure('Nested seed reconstruction exceeds tolerance', receipt)
        self.seed_errors = tuple(self.seed_errors)

    def width(self, node):
        if type(node) is not int or not 0 <= node < len(self.starts):
            raise ValueError('Node is outside the nested tree')
        return self.components*(self.stops[node]-self.starts[node])

    def restriction(self, parent, child):
        width = self.stops[parent]-self.starts[parent]
        points = torch.arange(self.starts[child]-self.starts[parent], self.stops[child]-self.starts[parent], device=self.device)
        return torch.cat([points+k*width for k in range(self.components)])

    def tensors(self):
        return tuple(t for group in (self.leaf_bases, self.leaf_indices, self.transfers) for t in group if t is not None)

    def assert_current(self):
        if ((id(self.tree), self.config, self.components, self.dtype, self.device, self.starts, self.stops, self.children, self.ranks)
                != self._binding or tuple((id(t), t._version) for t in self.tensors()) != self._versions):
            raise ValueError('Nested cluster basis changed')
        self.tree.assert_current()

    def _matrix(self, node, value, rows):
        self.assert_current()
        if (type(node) is not int or not 0 <= node < len(self.ranks)
                or not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] != rows
                or value.device != self.device or value.dtype != self.dtype or not bool(torch.isfinite(value).all())):
            raise ValueError('Nested projection requires finite matching node matrices')

    def project(self, node, value):
        """Apply the implicit node basis adjoint to a matrix of local columns."""
        self._matrix(node, value, self.width(node))
        def visit(current, local):
            if not self.children[current]:
                return self.leaf_bases[current].mH@local
            result = local.new_zeros((self.ranks[current], local.shape[1]))
            for child in self.children[current]:
                result = result+self.transfers[child].mH@visit(child, local[self.restriction(current, child)])
            return result
        return visit(node, value)

    def expand(self, node, value):
        """Apply the implicit node basis without retaining its expanded matrix."""
        self.width(node)
        self._matrix(node, value, self.ranks[node])
        def visit(current, local):
            if not self.children[current]:
                return self.leaf_bases[current]@local
            result = local.new_zeros((self.width(current), local.shape[1]))
            for child in self.children[current]:
                result.index_add_(0, self.restriction(current, child), visit(child, self.transfers[child]@local))
            return result
        return visit(node, value)

    def upward(self, value):
        self.assert_current()
        if (not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[-1] != self.components*self.tree.size
                or value.device != self.device or value.dtype != self.dtype or not bool(torch.isfinite(value).all())):
            raise ValueError('Upward pass requires matching finite complex CUDA vectors')
        projected = [None]*len(self.ranks)
        for node in reversed(range(len(self.ranks))):
            if not self.children[node]:
                projected[node] = value[..., self.leaf_indices[node]]@self.leaf_bases[node].conj()
            else:
                projected[node] = sum(projected[c]@self.transfers[c].conj() for c in self.children[node])
        return tuple(projected)

    def downward(self, coefficients):
        self.assert_current()
        coefficients = tuple(coefficients)
        if len(coefficients) != len(self.ranks):
            raise ValueError('Downward coefficients must identify every node')
        if not isinstance(coefficients[0], torch.Tensor) or coefficients[0].ndim < 1:
            raise ValueError('Downward coefficients require tensors with a rank axis')
        shape = coefficients[0].shape[:-1]
        for node, value in enumerate(coefficients):
            if (not isinstance(value, torch.Tensor) or value.shape != (*shape, self.ranks[node])
                    or value.device != self.device or value.dtype != self.dtype or not bool(torch.isfinite(value).all())):
                raise ValueError('Downward coefficients must match the nested ranks and batch shape')
        values = list(coefficients)
        result = coefficients[0].new_zeros((*shape, self.components*self.tree.size))
        for node in range(len(self.ranks)):
            if not self.children[node]:
                result.index_add_(-1, self.leaf_indices[node], values[node]@self.leaf_bases[node].mT)
            else:
                for child in self.children[node]:
                    values[child] = values[child]+values[node]@self.transfers[child].mT
        return result

    def receipt(self):
        self.assert_current()
        return {'version': 'nested_cluster_basis_v1', 'ranks': list(self.ranks),
                'retained_bytes': sum(t.numel()*t.element_size() for t in self.tensors()),
                'leaf_bases': sum(t is not None for t in self.leaf_bases),
                'transfer_matrices': sum(t is not None for t in self.transfers),
                'seed_errors': list(self.seed_errors), 'expanded_internal_bases_retained': False,
                'setup_complexity_qualified': False, 'physical_qualification': False}

    def weighted_grams(self, weights):
        """Compute Q_node.H diag(weights) Q_node through upward transfers."""
        self.assert_current()
        real_dtype = torch.float64 if self.dtype == torch.complex128 else torch.float32
        if (not isinstance(weights, torch.Tensor) or weights.shape != (self.components*self.tree.size,)
                or weights.device != self.device or weights.dtype != real_dtype
                or not bool(torch.isfinite(weights).all() & (weights >= 0).all())):
            raise ValueError('Nested Gram weights must be matching finite nonnegative real vectors')
        grams = [None]*len(self.ranks)
        for node in reversed(range(len(self.ranks))):
            if not self.children[node]:
                q = self.leaf_bases[node]
                grams[node] = q.mH@(weights[self.leaf_indices[node], None]*q)
            else:
                grams[node] = sum(self.transfers[c].mH@grams[c]@self.transfers[c] for c in self.children[node])
            if not bool(torch.isfinite(grams[node]).all()):
                raise ValueError('Nested weighted Gram overflowed native precision')
        return tuple(grams)

    def diagonal_forms(self, forms):
        """Sum diagonals of embedded Q_node W_node Q_node.H by downward transfer."""
        self.assert_current()
        forms = tuple(forms)
        if len(forms) != len(self.ranks):
            raise ValueError('Nested quadratic forms must identify every node')
        for rank, value in zip(self.ranks, forms):
            if (not isinstance(value, torch.Tensor) or value.shape != (rank, rank)
                    or value.device != self.device or value.dtype != self.dtype
                    or not bool(torch.isfinite(value).all())):
                raise ValueError('Nested quadratic forms must be finite matching rank matrices')
        real_dtype = torch.float64 if self.dtype == torch.complex128 else torch.float32
        result = torch.zeros(self.components*self.tree.size, device=self.device, dtype=real_dtype)
        values = list(forms)
        for node in range(len(self.ranks)):
            if not self.children[node]:
                q = self.leaf_bases[node]
                diagonal = (q.conj()*(q@values[node])).sum(1).real
                result.index_add_(0, self.leaf_indices[node], diagonal)
            else:
                for child in self.children[node]:
                    e = self.transfers[child]
                    values[child] = values[child]+e@values[node]@e.mH
        if not bool(torch.isfinite(result).all()):
            raise ValueError('Nested diagonal form overflowed native precision')
        return result
