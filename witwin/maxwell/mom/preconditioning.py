"""User-supplied right preconditioners for the scalar MoM GMRES path.

A right preconditioner maps an RWG coefficient residual (all J test rows, then
all M test rows for PMCHWT; J test rows only for the open-surface EFIE; physical
units) to a trial-coefficient correction of the same layout. It changes only the Krylov path: the operator, its identity, the RHS and
the true-residual admission rule are unchanged. Because the action may vary
between applications (for example a learned or inexact inner solve), the solve
uses flexible right-preconditioned GMRES, which stores every preconditioned
direction and recomputes the true residual ``b - A x`` after every step.

Preparation calls ``factory.bind(context)`` once. The context exposes the
coefficient layout, the leaf partition of the operator's support tree, original
principal (near-field) blocks and per-edge geometry features, so a classical or
learned preconditioner can be built without importing backend classes.
"""
from __future__ import annotations

import time

import torch


class PreconditionerContext:
    """Read-only preparation view of one prepared scalar MoM operator.

    ``handle`` is the prepared operator. Dense handles expose principal blocks by
    slicing their assembled matrix; hierarchical/FMM handles integrate them with
    the same original block provider that their near field uses.
    """

    def __init__(self, handle, frequency):
        self.handle = handle
        self.frequency = float(frequency)
        self.system = handle.system
        self.exterior = handle.exterior
        self.shape, self.device, self.dtype = tuple(handle.shape), handle.device, handle.dtype
        self.total_edges = self.system.total_edges
        self.J_slices, self.M_slices = self.system.J_slices, self.system.M_slices
        # Unknown blocks per edge: ('J', 'M') for PMCHWT, ('J',) for the EFIE.
        self.components = tuple(getattr(handle, 'current_components', ('J', 'M')))
        self._tree = getattr(handle, 'tree', None)
        self._provider = getattr(handle, 'provider', None)
        self.block_requests = 0

    @property
    def backend(self):
        return 'dense' if hasattr(self.handle, 'matrix') else 'hierarchical'

    def media(self):
        """Exterior and per-body interior (epsilon_r, mu_r, k) as Python complex numbers; None for sheets."""
        def record(medium):
            return {name: complex(getattr(medium, name)) for name in ('epsilon_r', 'mu_r', 'k', 'eta')}
        return {'exterior': record(self.exterior),
                'bodies': [None if b.interior is None else record(b.interior) for b in self.system.bodies]}

    def support_tree(self, leaf_size=32):
        """The operator's own support octree, or one built from RWG supports for dense handles."""
        if self._tree is None:
            from .cluster import build_support_octree
            lower, upper = self.edge_support_boxes()
            self._tree = build_support_octree(lower, upper, leaf_size=leaf_size)
        return self._tree

    def leaf_partition(self, max_block_edges=32, *, leaf_size=32):
        """Disjoint edge groups covering every global edge once, each at most ``max_block_edges``."""
        if type(max_block_edges) is not int or max_block_edges < 1:
            raise ValueError('max_block_edges must be a positive integer')
        tree = self.support_tree(leaf_size)
        leaves = (tree.children < 0).all(dim=1).nonzero().flatten()
        ranges = sorted(zip(tree.starts[leaves].cpu().tolist(), tree.stops[leaves].cpu().tolist()))
        groups = tuple(tree.permutation[a:min(a+max_block_edges, stop)]
                       for start, stop in ranges for a in range(start, stop, max_block_edges))
        if sum(len(g) for g in groups) != self.total_edges:
            raise RuntimeError('Leaf partition does not cover the coefficient space exactly once')
        return groups

    def coefficient_indices(self, edges):
        """Global coefficient indices of an edge selection, one block per current component."""
        return torch.cat([edges+i*self.total_edges for i in range(len(self.components))])

    def near_field_matrix(self, pfft):
        """Original body-local PMCHWT near entries as CUDA CSR (dense/pFFT only).

        The pattern is the union of each body's exterior and interior pFFT
        near pairs. Dense operators use the supplied PFFTConfig for that
        pattern; pFFT operators use their prepared pattern. Entries include
        both media on every selected pair, with cross-body entries omitted.
        """
        from .near_preconditioner import _near_field_matrix
        return _near_field_matrix(self, pfft)

    @torch.no_grad()
    def principal_block(self, edges):
        """Original operator block A[I, I] for I = coefficient_indices(edges), shape [cm, cm]."""
        if (not isinstance(edges, torch.Tensor) or edges.dtype != torch.int64 or edges.ndim != 1
                or edges.device != self.device or not len(edges)):
            raise ValueError('Principal blocks require a nonempty CUDA int64 edge vector')
        self.block_requests += 1
        if self.backend == 'dense':
            index = self.coefficient_indices(edges)
            with self.handle.read_access():
                self.handle.refresh()
                return self.handle.matrix[index][:, index].clone()
        return self._provider(edges, edges)

    @torch.no_grad()
    def edge_support_boxes(self):
        """Axis-aligned bounds [n,3] of each RWG support (both adjacent faces)."""
        lower, upper = [], []
        for body in self.system.bodies:
            basis, triangles = body.basis, body.mesh.triangles
            plus, minus = triangles[basis.plus_faces], triangles[basis.minus_faces]
            lower.append(torch.minimum(plus.amin(1), minus.amin(1)))
            upper.append(torch.maximum(plus.amax(1), minus.amax(1)))
        return torch.cat(lower), torch.cat(upper)

    @torch.no_grad()
    def edge_features(self):
        """Per-edge geometry in global edge order (metres, unit normals), all CUDA tensors."""
        out = {name: [] for name in ('midpoint', 'tangent', 'length', 'plus_centroid', 'minus_centroid',
                                     'plus_normal', 'minus_normal', 'plus_area', 'minus_area', 'body')}
        for index, body in enumerate(self.system.bodies):
            basis, mesh = body.basis, body.mesh
            a, b = mesh.vertices[basis.edges[:, 0]], mesh.vertices[basis.edges[:, 1]]
            centroids = mesh.triangles.mean(1)
            out['midpoint'].append((a+b)/2)
            out['tangent'].append((b-a)/basis.edge_lengths[:, None])
            out['length'].append(basis.edge_lengths)
            for side, faces in (('plus', basis.plus_faces), ('minus', basis.minus_faces)):
                out[side+'_centroid'].append(centroids[faces])
                out[side+'_normal'].append(mesh.normals[faces])
                out[side+'_area'].append(mesh.areas[faces])
            out['body'].append(torch.full((basis.n_edges,), index, device=self.device, dtype=torch.int64))
        return {name: torch.cat(values) for name, values in out.items()}

    @torch.no_grad()
    def apply_operator(self, value):
        """Original A x in the solver's coefficient layout (counts as an operator action)."""
        return self.handle.apply(value)


class IdentityRightPreconditioner:
    """P = I; flexible GMRES then reproduces unpreconditioned GMRES iterates."""
    def bind(self, context):
        return _Bound(lambda residual: residual.clone(), {'version': 'identity_right_preconditioner_v1'})


class LeafBlockRightPreconditioner:
    """Block Jacobi: LU of original principal blocks over disjoint support-tree leaves.

    Each block holds all J and M rows of at most ``max_block_edges`` edges.
    ``budget_bytes`` bounds the retained LU factors before any block integration.
    """
    def __init__(self, max_block_edges=32, *, budget_bytes=1024**3, leaf_size=32):
        for name, value in (('max_block_edges', max_block_edges), ('budget_bytes', budget_bytes), ('leaf_size', leaf_size)):
            if type(value) is not int or value < 1:
                raise ValueError(name+' must be a positive integer')
        self.max_block_edges, self.budget_bytes, self.leaf_size = max_block_edges, budget_bytes, leaf_size

    @torch.no_grad()
    def bind(self, context):
        from .pfft_system import PFFTSystemOperator
        groups = context.leaf_partition(self.max_block_edges, leaf_size=self.leaf_size)
        itemsize = torch.empty((), dtype=context.dtype).element_size()
        width = len(context.components)
        planned = sum((width*len(g))**2*itemsize+width*len(g)*16 for g in groups)
        if planned > self.budget_bytes:
            raise MemoryError(f'Leaf block preconditioner needs {planned} bytes, above budget {self.budget_bytes}')
        if isinstance(context.handle, PFFTSystemOperator):
            batched = context.handle._leaf_factors(groups)
            context.block_requests += len(groups)
        else:
            indices, factors, pivots = [], [], []
            for edges in groups:
                block = context.principal_block(edges)
                factor, pivot, info = torch.linalg.lu_factor_ex(block)
                if int(info) != 0 or not bool(torch.isfinite(factor).all()):
                    raise ValueError('Singular or nonfinite principal block in leaf preconditioner')
                indices.append(context.coefficient_indices(edges))
                factors.append(factor)
                pivots.append(pivot)
            # Group equal block sizes so one batched LU solve serves each size class.
            classes = {}
            for index, factor, pivot in zip(indices, factors, pivots):
                classes.setdefault(len(index), []).append((index, factor, pivot))
            batched = tuple((torch.stack([i for i, _, _ in v]), torch.stack([f for _, f, _ in v]),
                             torch.stack([p for _, _, p in v])) for v in classes.values())

        def apply(residual):
            flat = residual.reshape(-1, residual.shape[-1])
            result = torch.empty_like(flat)
            for index, factor, pivot in batched:
                rhs = flat[:, index].permute(1, 2, 0)  # [blocks, cm, columns]
                result[:, index] = torch.linalg.lu_solve(factor, pivot, rhs).permute(2, 0, 1)
            return result.reshape(residual.shape)
        retained = sum(t.numel()*t.element_size() for group in batched for t in group)
        return _Bound(apply, {'version': 'leaf_block_jacobi_right_preconditioner_v1',
                              'blocks': len(groups), 'max_block_edges': self.max_block_edges,
                              'size_classes': len(batched), 'retained_bytes': retained,
                              'block_source': context.backend})


class FunctionRightPreconditioner:
    """Adapter for a plain callable ``factory(context) -> apply(residual) -> correction``."""
    def __init__(self, factory, *, name='user_function'):
        if not callable(factory) or not isinstance(name, str) or not name:
            raise TypeError('FunctionRightPreconditioner requires a callable factory and a name')
        self.factory, self.name = factory, name

    def bind(self, context):
        apply = self.factory(context)
        if not callable(apply):
            raise TypeError('Right preconditioner factory must return a callable')
        return _Bound(apply, {'version': 'function_right_preconditioner_v1', 'name': self.name})


class _Bound:
    def __init__(self, apply, receipt):
        self._apply, self._receipt = apply, receipt

    def apply(self, residual):
        return self._apply(residual)

    def receipt(self):
        return dict(self._receipt)


def bind_right_preconditioner(factory, handle, frequency):
    """Bind once at preparation; return (bound action, receipt including setup time)."""
    device = handle.device
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    context = PreconditionerContext(handle, frequency)
    bound = factory.bind(context)
    if not callable(getattr(bound, 'apply', None)):
        raise TypeError('A bound right preconditioner must provide apply(residual)')
    torch.cuda.synchronize(device)
    receipt = dict(bound.receipt()) if callable(getattr(bound, 'receipt', None)) else {}
    receipt.update(setup_seconds=time.perf_counter()-start, principal_block_requests=context.block_requests,
                   krylov_method='flexible_right_gmres', admission='unchanged_true_residual')
    return bound, receipt
