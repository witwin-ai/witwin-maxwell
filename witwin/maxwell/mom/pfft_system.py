"""Precorrected-FFT PMCHWT system operator for disjoint scalar bodies.

The action is the sum of one exterior region over every body on a common grid
(wavenumber k0) and one interior region per body on its own grid (k_b). Each
region applies the original D/K channel algebra

    [J rows]   [K        iωε D] [J]
    [M rows] = [-iωμ D   K    ] [M]

with D and K from ``PFFTRegion`` plus a symmetric complex128 precorrection:
for every basis pair closer than the region's precorrection radius the grid
entry is replaced by the original near-field integral computed with the batched
near path (``scalar_edge_pair_channels``). D and K are symmetric (bilinear), so
one direction per pair is stored. Admission, sampled original residuals,
Riesz metric and diagonal scaling are the shared hierarchical ones.
"""
from __future__ import annotations

from contextlib import contextmanager
import functools
import time

import torch

from .pfft_config import PFFTConfig
from .pfft_grid import PFFTProjection
from .pfft_operator import PFFTRegion
from .spatial import BoxTree, overlapping_pairs

PFFT_SYSTEM_VERSION = 'pfft_precorrected_pmchwt_system_v1'


def near_pairs(midpoints, radius):
    """Unordered basis pairs (i <= j) with centre distance below ``radius``, diagonal included."""
    half = radius/2
    tree = BoxTree(midpoints-half, midpoints+half)
    rows, cols = [], []
    for a, b in overlapping_pairs(tree, 0.):
        keep = torch.linalg.vector_norm(midpoints[a]-midpoints[b], dim=-1) < radius
        rows.append(a[keep])
        cols.append(b[keep])
    diagonal = torch.arange(len(midpoints), device=midpoints.device)
    return torch.cat([diagonal, *rows]), torch.cat([diagonal, *cols])


def body_pair_chunks(bodies, rows, cols, batch):
    """(test index, source index, chunk) of global pairs grouped by body pair, each at most ``batch``.

    ``bodies`` is a sequence whose items start with (basis, edge_offset, ...).
    """
    device = rows.device
    offsets = torch.tensor([body[1] for body in bodies], device=device)
    stops = offsets+torch.tensor([body[0].n_edges for body in bodies], device=device)
    body_rows = torch.bucketize(rows, stops, right=True)
    body_cols = torch.bucketize(cols, stops, right=True)
    for ti in range(len(bodies)):
        for si in range(len(bodies)):
            selected = ((body_rows == ti) & (body_cols == si)).nonzero().flatten()
            for chunk in selected.split(batch):
                if len(chunk):
                    yield ti, si, chunk


def pmchwt_pair_entries(bodies, medium, integration, rows, cols, *, batch):
    """Original D and K entries of global edge pairs with the batched PMCHWT near path."""
    from .nearfield import scalar_edge_pair_channels
    exact_d = torch.empty(len(rows), device=rows.device, dtype=torch.complex128)
    exact_k = torch.empty_like(exact_d)
    for ti, si, chunk in body_pair_chunks(bodies, rows, cols, batch):
        (test, test_offset, test_geometry), (source, source_offset, source_geometry) = bodies[ti], bodies[si]
        d, k = scalar_edge_pair_channels(test, source, rows[chunk]-test_offset, cols[chunk]-source_offset,
                                         medium, integration, geometry=(test_geometry, source_geometry))
        exact_d[chunk], exact_k[chunk] = d, k
    return exact_d, exact_k


class PFFTPrecorrection:
    """Exact-minus-grid D/K entries for one region's near pairs (complex128, one direction).

    ``entries(rows, cols) -> (D, K or None)`` integrates the original entries of
    global pairs; the default is the PMCHWT near path over ``bodies``.
    """

    def __init__(self, region, bodies, medium, integration, *, batch, entries=None):
        projection = region.projection
        device = projection.device
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        # Grid interactions are trusted only between stencils and supports that are well
        # separated: the radius adds twice the largest RWG support radius to (factor * h).
        self.radius = region.config.near_radius+2*float(projection.support_radius.max())
        rows, cols = near_pairs(projection.midpoints, self.radius)
        if entries is None:
            exact_d, exact_k = pmchwt_pair_entries(bodies, medium, integration, rows, cols, batch=batch)
        else:
            exact_d, exact_k = entries(rows, cols)
        grid_d, grid_k = region.grid_entries(rows, cols)
        self.rows, self.cols = rows, cols
        self.off_diagonal = (rows != cols).nonzero().flatten()
        self.delta_d = exact_d-grid_d
        self.delta_k = None if exact_k is None else exact_k-grid_k
        torch.cuda.synchronize(device)
        self.setup_seconds = time.perf_counter()-start
        self.count = region.count

    def apply(self, delta, value):
        result = torch.zeros(self.count, device=value.device, dtype=torch.complex128)
        result.index_add_(0, self.rows, delta*value[self.cols])
        off = self.off_diagonal
        result.index_add_(0, self.cols[off], delta[off]*value[self.rows[off]])
        return result

    def tensors(self):
        return tuple(t for t in (self.rows, self.cols, self.off_diagonal, self.delta_d, self.delta_k)
                     if t is not None)

    def receipt(self):
        return {'pairs_stored': len(self.rows), 'pairs_logical': 2*len(self.rows)-self.count,
                'radius_m': self.radius,
                'retained_bytes': sum(t.numel()*t.element_size() for t in self.tensors()),
                'setup_seconds': self.setup_seconds}


class PFFTRegionOperator:
    """One medium region: grid D/K plus precorrection, applied to its local J and M."""

    def __init__(self, region, precorrection, medium):
        self.region, self.precorrection, self.medium = region, precorrection, medium
        self.electric, self.magnetic = 1j*medium.omega*medium.epsilon, -1j*medium.omega*medium.mu

    def channels(self, value):
        d, k = self.region.apply_channels(value)
        d = d.to(torch.complex128)+self.precorrection.apply(self.precorrection.delta_d, value)
        k = k.to(torch.complex128)+self.precorrection.apply(self.precorrection.delta_k, value)
        return d, k

    def apply(self, current_j, current_m, *, transpose=False):
        dj, kj = self.channels(current_j)
        dm, km = self.channels(current_m)
        electric, magnetic = (self.magnetic, self.electric) if transpose else (self.electric, self.magnetic)
        return kj+electric*dm, magnetic*dj+km

    def tensors(self):
        return (*self.precorrection.tensors(), self.region.green_hat, *self.region.gradient_hat)

    def receipt(self):
        record = {**self.region.receipt(), 'precorrection': self.precorrection.receipt()}
        record['retained_bytes'] = (record['kernel_bytes']+record['projection']['retained_bytes']
                                    + record['precorrection']['retained_bytes'])
        return record


class PFFTSystemActions:
    """Exterior grid over all bodies plus one interior grid per body."""

    def __init__(self, provider, config):
        from .cluster import build_support_octree
        if not isinstance(config, PFFTConfig):
            raise TypeError('pFFT system requires PFFTConfig')
        provider.assert_current()
        self.provider, self.config = provider, config
        system = provider.system
        self.device = system.device
        self.dtype = torch.complex128
        self.shape = (2*system.total_edges,)*2
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        bodies = [(body.basis, body.edge_offset, geometry) for body, geometry in zip(system.bodies, provider.geometry)]
        integration = provider.config
        if config.exterior_grids == 'clusters':
            from .pfft_coupling import ClusteredExterior
            self.exterior = ClusteredExterior(bodies, provider.exterior, config, integration)
        else:
            bases = [b.basis for b in system.bodies]
            spec = config.grid_spec(provider.exterior, bases)
            exterior = PFFTRegion(PFFTProjection.for_bases(bases, spec), provider.exterior)
            self.exterior = PFFTRegionOperator(exterior, PFFTPrecorrection(exterior, bodies, provider.exterior,
                                                                           integration, batch=config.near_pair_batch),
                                               provider.exterior)
        self.interiors = []
        for body, geometry in zip(system.bodies, provider.geometry):
            spec = config.grid_spec(body.interior, [body.basis])
            region = PFFTRegion(PFFTProjection(body.basis, spec), body.interior)
            correction = PFFTPrecorrection(region, [(body.basis, 0, geometry)], body.interior, integration,
                                           batch=config.near_pair_batch)
            self.interiors.append(PFFTRegionOperator(region, correction, body.interior))
        self.interiors = tuple(self.interiors)
        lower, upper = provider.support_boxes()
        self.tree = build_support_octree(lower, upper, leaf_size=config.tree_leaf_size)
        torch.cuda.synchronize(self.device)
        self.setup_seconds = time.perf_counter()-start
        self._binding = self._state()

    def _state(self):
        return (id(self.provider), self.config, tuple(id(t) for r in (self.exterior, *self.interiors)
                for t in r.tensors()))

    def assert_current(self):
        if self._state() != self._binding:
            raise ValueError('pFFT system tensors or bindings changed')
        self.provider.assert_current()
        self.tree.assert_current()

    def _action(self, value, *, transpose):
        n = self.shape[0]//2
        flat = value.reshape(-1, 2*n)
        output = torch.empty_like(flat)
        system = self.provider.system
        for row, x in enumerate(flat):
            yj, ym = self.exterior.apply(x[:n], x[n:], transpose=transpose)
            yj, ym = yj.clone(), ym.clone()
            for body, region in zip(system.bodies, self.interiors):
                js, ms = system.J_slices[body.region_id-1], system.M_slices[body.region_id-1]
                bj, bm = region.apply(x[js], x[ms], transpose=transpose)
                yj[js] += bj
                ym[ms.start-n:ms.stop-n] += bm
            output[row] = torch.cat((yj, ym))
        return output.reshape(value.shape)

    def apply(self, value):
        self.assert_current()
        return self._action(value, transpose=False)

    def adjoint(self, value):
        """A^H y = conj(A^T conj(y)); D and K are symmetric, so A^T swaps the material factors."""
        self.assert_current()
        return self._action(value.conj().resolve_conj(), transpose=True).conj().resolve_conj()

    def receipt(self):
        regions = [('exterior', self.exterior), *((f'interior_{i+1}', r) for i, r in enumerate(self.interiors))]
        records = {name: r.receipt() for name, r in regions}
        retained = sum(rec['retained_bytes'] for rec in records.values())
        return {'version': PFFT_SYSTEM_VERSION, 'configuration': self.config.payload(), 'regions': records,
                'retained_bytes': retained, 'setup_seconds': self.setup_seconds,
                'physical_qualification': False, 'complexity_qualification': False}


class PFFTSystemOperator:
    """Shared surface-operator protocol around ``PFFTSystemActions`` (iterative only)."""
    supports_direct = False

    def __init__(self, provider, config):
        from .operator_protocol import IterativeState
        from .riesz import LocalRWGRiesz
        self.actions = PFFTSystemActions(provider, config)
        self.provider, self.pfft = provider, config
        self.config = config.solve_config()
        self.metric = LocalRWGRiesz(provider.system, provider.exterior.eta)
        self.tree = self.actions.tree
        self.shape, self.device, self.dtype = self.actions.shape, self.actions.device, self.actions.dtype
        self.iterative_state = IterativeState()
        self._apply_calls = self._adjoint_calls = 0
        self.metadata = {'version': PFFT_SYSTEM_VERSION, 'operator_backend': 'pfft',
                         'pfft': config.payload(), 'physical_qualification': False,
                         'original_global_residual_certified': False, 'peak_memory_qualified': False,
                         'residual_metric': 'impedance_balanced_L2_dual_proxy', 'trace_norm_certified': False}
        self._binding = (id(self.actions), id(provider), id(self.metric), id(self.tree), self.config)

    def refresh(self):
        if (id(self.actions), id(self.provider), id(self.metric), id(self.tree), self.config) != self._binding:
            raise ValueError('pFFT solver binding changed')
        self.actions.assert_current()
        self.metric.assert_current()

    @contextmanager
    def read_access(self):
        self.refresh()
        try:
            yield
        finally:
            self.refresh()

    def _vector(self, value):
        if (not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[-1] != self.shape[0]
                or value.dtype != self.dtype or value.device != self.device or not bool(torch.isfinite(value).all())):
            raise ValueError('pFFT vectors require finite matching complex128 CUDA coefficients')

    def apply(self, value):
        with self.read_access():
            self._vector(value)
            self._apply_calls += 1
            return self.actions.apply(value)

    def adjoint(self, value):
        with self.read_access():
            self._vector(value)
            self._adjoint_calls += 1
            return self.actions.adjoint(value)

    def riesz_inverse(self, value):
        with self.read_access():
            return self.metric.inverse(value)

    def preconditioner_scales(self, kind):
        from .direct_operator import DirectHierarchicalOperator
        return DirectHierarchicalOperator.preconditioner_scales(self, kind)

    def _leaf_factors(self, groups):
        """Integrate and factor equal-size leaves together; check LU status once."""
        classes = {}
        for edges in groups:
            classes.setdefault(len(edges), []).append(edges)
        batched, valid = [], []
        for same_size in classes.values():
            edges = torch.stack(same_size)
            blocks = self.provider._principal_blocks_many(edges)
            factor, pivot, info = torch.linalg.lu_factor_ex(blocks)
            valid.append((info == 0).all() & torch.isfinite(factor).all())
            indices = torch.cat((edges, edges+self.shape[0]//2), dim=1)
            batched.append((indices, factor, pivot))
        if not bool(torch.stack(valid).all()):
            raise ValueError('Singular or nonfinite principal block in leaf preconditioner')
        return tuple(batched)

    def residual_report(self, value, rhs, *, original_edges=None, source_batch=64):
        from .hmatrix import HierarchicalOperator
        return HierarchicalOperator.residual_report(self, value, rhs, original_edges=original_edges,
                                                    source_batch=source_batch)

    def solve(self, rhs, stopping=None, **kwargs):
        from .direct_operator import DirectHierarchicalOperator
        return DirectHierarchicalOperator.solve(self, rhs, stopping, **kwargs)

    def persistence_receipt(self):
        self.refresh()
        return {'version': PFFT_SYSTEM_VERSION, 'construction': self.actions.receipt(),
                'executable_snapshot': False}

    def operator_stats(self):
        self.refresh()
        return {**self.metadata, 'apply_calls': self._apply_calls, 'adjoint_calls': self._adjoint_calls,
                'construction': self.actions.receipt(), 'metric': self.metric.receipt()}


@functools.lru_cache(maxsize=1)
def _system_operator_class():
    from .hierarchical_runtime import HierarchicalSystemOperator
    from .nearfield import ScalarSystemBlocks
    from .runtime import SystemDecodeHandle

    class PFFTPhysicalSystemOperator(PFFTSystemOperator, SystemDecodeHandle):
        """Public pFFT runtime handle with the lifted RHS and decode of the other fast backends."""
        _bind_physical = HierarchicalSystemOperator._bind_physical
        _refresh_physical = HierarchicalSystemOperator._refresh_physical
        build_rhs = HierarchicalSystemOperator.build_rhs

        @functools.cached_property
        def rhs_lifting(self):
            from .pfft_lifting import PFFTLifting
            liftings = []
            for i, (body, interior) in enumerate(zip(self.system.bodies, self.actions.interiors)):
                if self.pfft.exterior_grids == 'single':
                    exterior, offset = self.actions.exterior, body.edge_offset
                else:
                    cluster = next(j for j, group in enumerate(self.actions.exterior.groups) if i in group)
                    group = self.actions.exterior.groups[cluster]
                    exterior = self.actions.exterior.regions[cluster][1]
                    offset = sum(self.system.bodies[j].basis.n_edges for j in group if j < i)
                if exterior.region.config.spacing < interior.region.config.spacing:
                    region, edges, name = exterior.region, slice(offset, offset+body.basis.n_edges), 'exterior_finer_spacing'
                else:
                    region, edges, name = interior.region, slice(0, body.basis.n_edges), 'interior'
                liftings.append(PFFTLifting(self.body_handles[i], region, edges, name))
            return tuple(liftings)

        def __init__(self, system, exterior, sources, config):
            if config.operator_backend != 'pfft':
                raise ValueError('pFFT runtime requires operator_backend=pfft')
            PFFTSystemOperator.__init__(self, ScalarSystemBlocks(system, exterior, config.integration), config.pfft)
            self._bind_physical(system, exterior, sources, config)

        def refresh(self):
            PFFTSystemOperator.refresh(self)
            self._refresh_physical()

    return PFFTPhysicalSystemOperator


def pfft_system_operator(system, exterior, sources, config):
    """Construct the public pFFT runtime handle (used by ``MoMRuntime`` dispatch)."""
    return _system_operator_class()(system, exterior, sources, config)
