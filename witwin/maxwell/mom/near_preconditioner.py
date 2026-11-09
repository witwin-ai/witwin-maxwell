"""Body-local sparse LU near-field right preconditioner for scalar PMCHWT.

The matrix contains the original entries on the union of the exterior and
interior pFFT near zones, with cross-body pairs omitted. No grid correction
alone is used as a physical matrix entry. Sparse LU uses the optional GPU
direct-solver dependency.
"""
from __future__ import annotations

import weakref

import torch

from .preconditioning import _Bound


def _body_pairs(context, pfft):
    """Unordered, body-local edge pairs in global ordering."""
    from .pfft_system import near_pairs

    n = context.total_edges
    if context.backend == 'dense':
        bases = [body.basis for body in context.system.bodies]
        midpoints, radii = [], []
        for basis in bases:
            vertices = basis.mesh.vertices
            midpoint = (vertices[basis.edges[:, 0]]+vertices[basis.edges[:, 1]])/2
            corners = torch.stack((vertices[basis.edges[:, 0]], vertices[basis.edges[:, 1]],
                                   vertices[basis.plus_vertices], vertices[basis.minus_vertices]), 1)
            midpoints.append(midpoint)
            radii.append(float(torch.linalg.vector_norm(corners-midpoint[:, None], dim=-1).max()))
        exterior_radius = pfft.grid_spec(context.exterior, bases).near_radius+2*max(radii)
        pairs = []
        for body, midpoint, radius in zip(context.system.bodies, midpoints, radii):
            interior_radius = pfft.grid_spec(body.interior, [body.basis]).near_radius+2*radius
            rows, cols = near_pairs(midpoint, max(exterior_radius, interior_radius))
            pairs.append((rows+body.edge_offset)*n+cols+body.edge_offset)
        keys = torch.cat(pairs).sort().values
    else:
        actions = context.handle.actions
        if context.handle.pfft.exterior_grids == 'clusters':
            regions = actions.exterior.regions
        else:
            regions = [(torch.arange(n, device=context.device), actions.exterior)]
        stops = torch.tensor([body.edge_offset+body.basis.n_edges for body in context.system.bodies],
                             device=context.device)
        pairs = []
        for index, region in regions:
            correction = region.precorrection
            rows, cols = index[correction.rows], index[correction.cols]
            same_body = torch.bucketize(rows, stops, right=True) == torch.bucketize(cols, stops, right=True)
            pairs.append(rows[same_body]*n+cols[same_body])
        for body, region in zip(context.system.bodies, actions.interiors):
            correction = region.precorrection
            pairs.append((correction.rows+body.edge_offset)*n+correction.cols+body.edge_offset)
        keys = torch.cat(pairs).unique(sorted=True)
    return keys//n, keys % n


@torch.no_grad()
def _near_field_matrix(context, pfft):
    """Original PMCHWT entries, mirrored by edge pair before composing J/M blocks."""
    from .nearfield import scalar_edge_pair_channels

    rows, cols = _body_pairs(context, pfft)
    off = rows != cols
    n = context.total_edges
    directed_rows, directed_cols = torch.cat((rows, cols[off])), torch.cat((cols, rows[off]))
    rr = torch.cat((directed_rows, directed_rows, directed_rows+n, directed_rows+n))
    cc = torch.cat((directed_cols, directed_cols+n, directed_cols, directed_cols+n))
    if context.backend == 'dense':
        with context.handle.read_access():
            context.handle.refresh()
            values = context.handle.matrix[rr, cc].clone()
    else:
        # Reintegrate only these pairs: pFFT retains exact-minus-grid values,
        # and reconstructing grid entries would repeat its stencil-pair work.
        provider = context.handle.provider
        k = torch.empty(len(rows), device=context.device, dtype=context.dtype)
        electric, magnetic = torch.empty_like(k), torch.empty_like(k)
        for body, geometry in zip(context.system.bodies, provider.geometry):
            selected = ((rows >= body.edge_offset) &
                        (rows < body.edge_offset+body.basis.n_edges)).nonzero().flatten()
            for chunk in selected.split(pfft.near_pair_batch):
                r, c = rows[chunk]-body.edge_offset, cols[chunk]-body.edge_offset
                kk, ee, mm = (torch.zeros(len(chunk), device=context.device, dtype=context.dtype)
                              for _ in range(3))
                for medium in (context.exterior, body.interior):
                    d, kval = scalar_edge_pair_channels(body.basis, body.basis, r, c, medium,
                                                        provider.config, geometry=(geometry, geometry))
                    kk += kval
                    ee += (1j*medium.omega*medium.epsilon)*d
                    mm += (-1j*medium.omega*medium.mu)*d
                k[chunk], electric[chunk], magnetic[chunk] = kk, ee, mm
        k, electric, magnetic = (torch.cat((v, v[off])) for v in (k, electric, magnetic))
        values = torch.cat((k, electric, magnetic, k))
    return torch.sparse_coo_tensor(torch.stack((rr, cc)), values, context.shape).coalesce().to_sparse_csr()


def _equilibrate(matrix):
    """L A R with unit row norms, then unit column norms; retain L and R."""
    coo = matrix.to_sparse_coo()
    rows, cols = coo.indices()
    values = coo.values()
    left = torch.zeros(matrix.shape[0], device=values.device, dtype=values.real.dtype)
    left.index_add_(0, rows, values.abs().square())
    left = left.rsqrt()
    values = values*left[rows]
    right = torch.zeros_like(left)
    right.index_add_(0, cols, values.abs().square())
    right = right.rsqrt()
    balanced = torch.sparse_csr_tensor(matrix.crow_indices(), matrix.col_indices(),
                                       values*right[cols], size=matrix.shape)
    return balanced, left, right


class NearFieldRightPreconditioner:
    """Sparse LU body-local PMCHWT right inverse.

    Dense operators use ``pfft`` (default PFFTConfig) to define the near zone;
    pFFT operators always use their own prepared near-pair pattern and config.
    Sparse LU factors once on GPU using the optional ``direct`` dependency,
    then applies triangular solves. It requires no host
    coefficient transfers. GPU direct allocations are outside torch's cap;
    callers must reserve sufficient device memory for the factors/workspace.
    """

    def __init__(self, *, pfft=None):
        self.pfft = pfft

    @torch.no_grad()
    def bind(self, context):
        from .pfft_config import PFFTConfig

        pfft = (self.pfft or PFFTConfig()) if context.backend == 'dense' else context.handle.pfft
        matrix = context.near_field_matrix(pfft)
        balanced, left, right = _equilibrate(matrix)
        receipt = {'version': 'near_field_right_preconditioner_v1',
                   'matrix_nnz': matrix.values().numel(), 'matrix_source': context.backend,
                   'cross_body_pairs': False, 'execution': 'cuda'}
        from nvmath.sparse.advanced import DirectSolver

        # The direct interface requires int32 CSR indices.
        balanced = torch.sparse_csr_tensor(balanced.crow_indices().to(torch.int32),
            balanced.col_indices().to(torch.int32), balanced.values(), size=balanced.shape)
        rhs = torch.empty(context.shape[0], device=context.device, dtype=context.dtype)
        solver = DirectSolver(balanced, rhs)
        plan = solver.plan()
        peak = int(plan.memory_estimates.peak_device_memory)
        free, _ = torch.cuda.mem_get_info(context.device)
        if free < peak+4*1024**3:
            solver.free()
            raise MemoryError(f'Sparse LU needs {peak} device bytes plus a 4 GiB reserve; {free} free')
        factorization = solver.factorize()
        receipt['factor_nnz'] = int(factorization.lu_nnz)
        receipt['factor_peak_bytes_estimate'] = peak

        def apply(residual):
            output = []
            for row in residual.reshape(-1, residual.shape[-1]):
                rhs.copy_(left*row)
                output.append(right*solver.solve())
            return torch.stack(output).reshape(residual.shape)

        bound = _Bound(apply, receipt)
        weakref.finalize(bound, solver.free)
        return bound
