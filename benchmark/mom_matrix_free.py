"""Experimental streamed weak PMCHWT action; no public backend or acceleration claim.

Every action recomputes the five actual panel-pair integration classes. Only
bounded pair chunks and local 3-by-3 blocks are retained. This measures the
memory/time tradeoff before any fast-multipole or hierarchical implementation.
"""
from dataclasses import asdict
import math
import time

import torch

from witwin.maxwell.constants import EPSILON_0, MU_0
from witwin.maxwell.mom.operator import (IntegrationConfig, _pair_integrals,
    _adaptive_near_integrals)
from witwin.maxwell.mom.quadrature import classify_pairs


class StreamedWeakOperator:
    """CUDA action on vectors with a final coefficient dimension, including A^H."""

    def __init__(self, basis, exterior, interior, config=IntegrationConfig(), *, pair_chunk=4096):
        if basis.mesh.device.type != 'cuda':
            raise ValueError('Streamed weak PMCHWT requires CUDA')
        if isinstance(pair_chunk, bool) or not isinstance(pair_chunk, int) or pair_chunk < 1:
            raise ValueError('pair_chunk must be a positive integer')
        self.basis, self.exterior, self.interior = basis, exterior, interior
        self.config, self.pair_chunk = config, pair_chunk
        self.device = basis.mesh.device
        self.dtype = torch.complex128 if basis.mesh.dtype == torch.float64 else torch.complex64
        self.shape = (2*basis.n_edges, 2*basis.n_edges)
        self.last_action = None

    def apply(self, value):
        return self._action(value, adjoint=False)

    def adjoint(self, value):
        return self._action(value, adjoint=True)

    def _action(self, value, *, adjoint):
        if value.ndim < 1 or value.shape[-1] != self.shape[1]:
            raise ValueError('Final dimension must match the coefficient count')
        if value.device != self.device or value.dtype != self.dtype:
            raise ValueError('Input must match the CUDA operator dtype and device')
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        ne, nt = self.basis.n_edges, len(self.basis.mesh.faces)
        x = value.reshape(-1, 2*ne)
        output = torch.zeros_like(x)
        counts = [0]*5
        near = dict(maximum_order=0, pair_evaluations=0, maximum_accepted_relative_difference=0.)
        for begin in range(0, nt*nt, self.pair_chunk):
            pairs = torch.arange(begin, min(begin+self.pair_chunk, nt*nt), device=self.device)
            rows, cols = pairs//nt, pairs%nt
            classes = classify_pairs(self.basis.mesh, rows, cols)
            for category in range(5):
                selected = torch.nonzero(classes == category).flatten()
                counts[category] += len(selected)
                batch = self.config.far_pair_batch if category == 4 else self.config.pair_batch
                for ids in selected.split(batch):
                    if len(ids) == 0:
                        continue
                    r, c = rows[ids], cols[ids]
                    ri, ci = self.basis.face_edges[r], self.basis.face_edges[c]
                    order = self.config.order if category == 4 else self.config.singular_order
                    for medium in (self.exterior, self.interior):
                        if category == 3:
                            d, k, stats = _adaptive_near_integrals(self.basis, r, c, medium, self.config)
                            for name in near:
                                near[name] = near[name]+stats[name] if name == 'pair_evaluations' else max(near[name], stats[name])
                        else:
                            d, k = _pair_integrals(self.basis, r, c, medium, order, order, category != 4, category)
                        omega = 2*math.pi*medium.frequency
                        for i, j, block in ((0, 0, k), (1, 1, k),
                                (0, 1, 1j*omega*EPSILON_0*medium.epsilon_r*d),
                                (1, 0, -1j*omega*MU_0*medium.mu_r*d)):
                            if adjoint:
                                # Global scatter swaps row/column; conjugate the complete block,
                                # including the complex constitutive coefficient, exactly once.
                                contracted = torch.einsum('plm,bpl->bpm', block.conj(), x[:, ri+i*ne])
                                output.index_add_(1, (ci+j*ne).flatten(), contracted.flatten(1))
                            else:
                                contracted = torch.einsum('plm,bpm->bpl', block, x[:, ci+j*ne])
                                output.index_add_(1, (ri+i*ne).flatten(), contracted.flatten(1))
        torch.cuda.synchronize(self.device)
        self.last_action = dict(seconds=time.perf_counter()-start, adjoint=adjoint,
            rhs_count=len(x), triangle_pair_counts=counts, near_disjoint=near,
            pair_chunk=self.pair_chunk, integration=asdict(self.config),
            stored_operator_matrix_bytes=0, stored_global_pair_list_bytes=0)
        return output.reshape(value.shape)


def experimental_gmres(operator, rhs, *, rtol=1e-9, maxiter=60):
    """Physical scaling wrapper around the shared CUDA GMRES implementation.

    The shared private action seam evaluates this streamed operator for both
    Arnoldi and every true residual. No matrix or direct fallback is provided.
    """
    from witwin.maxwell.mom.linalg import StoppingRule, _single_gmres
    ne = operator.basis.n_edges
    eta = abs(operator.exterior.eta)
    options = dict(device=rhs.device, dtype=rhs.real.dtype)
    column = torch.cat((torch.full((ne,), 1/eta, **options), torch.ones(ne, **options)))
    row = torch.cat((torch.full((ne,), eta, **options), torch.ones(ne, **options)))
    rule = StoppingRule(rtol=rtol, atol=0., maxiter=maxiter, restart=maxiter,
        stagnation_window=maxiter+1, method='gmres', preconditioner='none', fallback='none')
    start = time.perf_counter()
    result, stats = _single_gmres(operator, rhs, torch.zeros_like(rhs), rule,
        row, column, action=operator.apply)
    stats.update(seconds=time.perf_counter()-start, stored_operator_matrix_bytes=0,
        scope='Shared CUDA GMRES with streamed action and physical impedance scaling; no dense fallback')
    return result, stats
