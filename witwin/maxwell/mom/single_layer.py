"""Electric single-layer operator blocks for RWG currents (exp(-i omega t)).

For RWG trial/test functions f_m, f_n and G = exp(ikR)/(4 pi R) this module
integrates, face pair by face pair,

    vector[m, n]   = int int f_m(r) . f_n(r') G dS' dS     [m^2 * m^2/m = m^3]
    scalar_p0[T, S] = int_T int_S G dS' dS                  [m^3]

so that the weak EFIE single layer is

    <f_m, L f_n> = i*omega*mu * (vector - div^T scalar_p0 div / k^2)[m, n],

with the piecewise-constant RWG divergence div [faces, edges] in 1/m. The
vector-potential and scalar-potential parts stay separate: a fast backend can
apply either one alone, and loop/star preconditioners can scale them
independently. Touching pairs use the joint Duffy rules of ``quadrature``;
near-disjoint pairs refine regular product orders until both blocks agree to
``IntegrationConfig.near_tolerance``. Boundary-edge slots of open RWG spaces
have zero slope and are excluded from every scatter.
"""
from __future__ import annotations

import time

import torch

from .kernels import FREE_SPACE
from .operator import IntegrationConfig, classify_cross_pairs, local_geometry
from .quadrature import classify_all_pairs, touching_pair_rule, triangle_rule

SINGLE_LAYER_VERSION = 'rwg_single_layer_vector_scalar_p0_v1'


def single_layer_pair_integrals(test_basis, source_basis, rows, cols, k, order, category, *, kernel=FREE_SPACE):
    """Local blocks for face pairs: vector [B,3,3] and scalar [B] (both complex)."""
    _, free, slopes = local_geometry(test_basis)
    _, source_free, source_slopes = local_geometry(source_basis)
    if category in (0, 1, 2):
        if source_basis is not test_basis:
            raise ValueError('Different bodies cannot have topologically touching quadrature pairs')
        faces = test_basis.mesh.faces
        obs, src, weight = touching_pair_rule(free[rows], free[cols], faces[rows], faces[cols], category, order)
        g, _ = kernel.evaluate(obs-src, k)
        f = (slopes[rows, None, :, None]*(obs[:, :, None]-free[rows, None])).to(g.dtype)
        h = (slopes[cols, None, :, None]*(src[:, :, None]-free[cols, None])).to(g.dtype)
        weighted = g*weight
        return torch.einsum('bqlc,bqmc,bq->blm', f, h, weighted), weighted.sum(1)
    obs, wo = triangle_rule(free[rows], order)
    src, ws = triangle_rule(source_free[cols], order)
    g, _ = kernel.evaluate(obs[:, :, None]-src[:, None], k)
    weighted = g*(wo[:, :, None]*ws[:, None])
    f = (slopes[rows, None, :, None]*(obs[:, :, None]-free[rows, None])).to(g.dtype)
    h = (source_slopes[cols, None, :, None]*(src[:, :, None]-source_free[cols, None])).to(g.dtype)
    return torch.einsum('bqlc,bsmc,bqs->blm', f, h, weighted), weighted.sum((1, 2))


def adaptive_near_single_layer(test_basis, source_basis, rows, cols, k, config, *, kernel=FREE_SPACE):
    """Raise disjoint near-pair orders until vector and scalar blocks stabilize."""
    order = config.singular_order
    vector, scalar = single_layer_pair_integrals(test_basis, source_basis, rows, cols, k, order, 3, kernel=kernel)
    pending = torch.arange(len(rows), device=rows.device)
    evaluations, worst = len(rows), 0.0
    tiny = torch.finfo(test_basis.mesh.dtype).tiny
    while len(pending):
        order = min(order+2, config.near_max_order)
        new_vector, new_scalar = single_layer_pair_integrals(test_basis, source_basis, rows[pending], cols[pending],
                                                             k, order, 3, kernel=kernel)
        evaluations += len(pending)
        vector_error = (torch.linalg.vector_norm(new_vector-vector[pending], dim=(-2, -1))
                        / torch.linalg.vector_norm(new_vector, dim=(-2, -1)).clamp_min(tiny))
        scalar_error = (new_scalar-scalar[pending]).abs()/new_scalar.abs().clamp_min(tiny)
        relative = torch.maximum(vector_error, scalar_error)
        vector[pending], scalar[pending] = new_vector, new_scalar
        accepted = relative <= config.near_tolerance
        if bool(accepted.any()):
            worst = max(worst, float(relative[accepted].max()))
        pending = pending[~accepted]
        if len(pending) and order == config.near_max_order:
            raise RuntimeError(f'Near-panel single-layer quadrature failed tolerance {config.near_tolerance:g} '
                               f'at order {order}; maximum relative order difference {float(relative.max()):.3g}.')
    return vector, scalar, {'maximum_order': order, 'pair_evaluations': evaluations,
                            'maximum_accepted_relative_difference': worst}


def _scatter_block(target, test_basis, source_basis, rows, cols, values, row_offset, col_offset):
    signs_r, signs_c = test_basis.face_signs[rows], source_basis.face_signs[cols]
    valid = (signs_r[:, :, None] != 0) & (signs_c[:, None, :] != 0)
    ri = (test_basis.face_edges[rows, :, None]+row_offset).expand(-1, -1, 3)
    ci = (source_basis.face_edges[cols, None, :]+col_offset).expand(-1, 3, -1)
    target.index_put_((ri[valid], ci[valid]), values[valid].to(target.dtype), accumulate=True)


def assemble_single_layer(bodies, k, config=IntegrationConfig(), *, kernel=FREE_SPACE):
    """Global vector [E, E] and scalar_p0 [F, F] blocks over separately meshed bodies.

    ``bodies`` is a sequence of (basis, edge_offset, face_offset) in global
    coefficient order. Self blocks classify topological contact; cross blocks
    have near or far pairs only.
    """
    if not bodies:
        raise ValueError('Single-layer assembly requires at least one body')
    first = bodies[0][0].mesh
    if first.device.type != 'cuda':
        raise ValueError('MoM assembly requires CUDA.')
    torch.cuda.synchronize(first.device)
    started = time.perf_counter()
    dtype = torch.complex128 if first.dtype == torch.float64 else torch.complex64
    edges = sum(basis.n_edges for basis, _, _ in bodies)
    faces = sum(len(basis.mesh.faces) for basis, _, _ in bodies)
    vector = torch.zeros((edges, edges), device=first.device, dtype=dtype)
    scalar = torch.zeros((faces, faces), device=first.device, dtype=dtype)
    counts = [0]*5
    near = {'maximum_order': 0, 'pair_evaluations': 0, 'maximum_accepted_relative_difference': 0.0}
    for test, edge_r, face_r in bodies:
        for source, edge_c, face_c in bodies:
            nr, nc = len(test.mesh.faces), len(source.mesh.faces)
            if test is source:
                classes = classify_all_pairs(test.mesh)
            else:
                classes = torch.empty(nr*nc, dtype=torch.uint8, device=first.device)
                for start in range(0, nr*nc, 4096):
                    ids = torch.arange(start, min(start+4096, nr*nc), device=first.device)
                    classes[start:start+len(ids)], distance = classify_cross_pairs(test, source, ids//nc, ids % nc)
                    if not bool((distance > 0).all()):
                        raise ValueError('Single-layer cross blocks require strictly separated surfaces')
            for category in range(5):
                selected = torch.nonzero(classes == category).flatten()
                counts[category] += int(selected.numel())
                batch = config.far_pair_batch if category == 4 else config.pair_batch
                for ids in selected.split(batch):
                    # An empty selection still yields one empty chunk; cross-sheet
                    # blocks have no touching categories at all.
                    if not len(ids):
                        continue
                    rows, cols = ids//nc, ids % nc
                    if category == 3:
                        values, potential, stats = adaptive_near_single_layer(test, source, rows, cols, k, config,
                                                                              kernel=kernel)
                        near['maximum_order'] = max(near['maximum_order'], stats['maximum_order'])
                        near['pair_evaluations'] += stats['pair_evaluations']
                        near['maximum_accepted_relative_difference'] = max(
                            near['maximum_accepted_relative_difference'], stats['maximum_accepted_relative_difference'])
                    else:
                        order = config.order if category == 4 else config.singular_order
                        values, potential = single_layer_pair_integrals(test, source, rows, cols, k, order, category,
                                                                        kernel=kernel)
                    _scatter_block(vector, test, source, rows, cols, values, edge_r, edge_c)
                    scalar[rows+face_r, cols+face_c] = potential.to(dtype)
            del classes
    torch.cuda.synchronize(first.device)
    return vector, scalar, {
        'single_layer_version': SINGLE_LAYER_VERSION, 'kernel_id': kernel.kernel_id,
        'kernel_version': kernel.version, 'quadrature': vars(config), 'triangle_pair_counts': counts,
        'touching_quadrature': 'joint_duffy_self_6_edge_6_vertex_2_sectors',
        'quadrature_error_estimate': {'near_disjoint': near, 'touching_and_far': 'order_convergence_required'},
        'assembly_seconds': time.perf_counter()-started,
        'block_bytes': vector.numel()*vector.element_size()+scalar.numel()*scalar.element_size()}


def global_divergence(bodies, dtype):
    """Block-diagonal piecewise-constant divergence [F, E] in 1/m."""
    blocks = [basis.divergence for basis, _, _ in bodies]
    return torch.block_diag(*blocks).to(dtype)
