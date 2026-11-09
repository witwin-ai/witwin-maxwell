"""Calderon right preconditioner for the open-surface PEC EFIE.

``CalderonPreconditioner`` plugs into ``MoMConfig(right_preconditioner=...)``
and therefore into flexible right GMRES, whose admission rule remains the true
residual ``b - Z x``. It applies P = G^-T Z_bc G^-1, where Z_bc is the same
EFIE discretized by Buffa-Christiansen (BC) functions on the barycentric
refinement and G_mn = <f_m, n x b_n> is the RWG/rotated-BC mixed Gram. The
operator product approximates the Calderon identity, removing the O(1/h^2)
dense-discretization growth of the EFIE; measured iteration counts are also
flat down to plate sizes of 0.01 wavelength.

Open surfaces: a dual cell at a boundary vertex is an open fan. The default
``open_edge_rule='natural'`` leaves such cells uncharged and lets the BC
flux cross the sheet boundary through one-face barycentric functions, so the
dual space is the non-tilde H^-1/2(div) paired with the tilde primal space;
``'zero_flux'`` keeps uniform fan charge and zero boundary flux and is kept
for comparison (it grows with refinement). The identity T^2 = -I/4 + compact
holds only on closed surfaces, so on sheets the preconditioner is an operator
heuristic whose performance is measured, not certified.
``preconditioned_residual`` reports P r next to the true residual.

Construction of the BC coefficient table is combinatorial host work on the
mesh incidence; all quadrature, assembly and application stay on CUDA. The
dual operator is streamed over barycentric face rows and never stores the
barycentric matrix, so a fast backend can replace that one step.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict

import torch

from ..constants import MU_0
from .open_surface import OpenRWGBasis, OpenSurfaceMesh
from .operator import IntegrationConfig
from .preconditioning import _Bound
from .quadrature import classify_pairs, triangle_rule
from .single_layer import adaptive_near_single_layer, single_layer_pair_integrals

CALDERON_VERSION = 'open_efie_calderon_bc_v1'


def barycentric_refinement(mesh):
    """Six children per face: (v0,m01,c),(m01,v1,c),(v1,m12,c),(m12,v2,c),(v2,m20,c),(m20,v0,c)."""
    topology = mesh.topology
    count, edge_count = len(mesh.vertices), len(topology.edges)
    midpoints = mesh.vertices[topology.edges].mean(1)
    vertices = torch.cat((mesh.vertices, midpoints, mesh.centroids))
    v0, v1, v2 = mesh.faces.unbind(-1)
    m12, m20, m01 = (topology.face_edges+count).unbind(-1)
    c = torch.arange(len(mesh.faces), device=mesh.device)+count+edge_count
    children = torch.stack((torch.stack((v0, m01, c), -1), torch.stack((m01, v1, c), -1),
                            torch.stack((v1, m12, c), -1), torch.stack((m12, v2, c), -1),
                            torch.stack((v2, m20, c), -1), torch.stack((m20, v0, c), -1)), 1).reshape(-1, 3)
    return OpenSurfaceMesh(vertices, children, validate=False)


OPEN_EDGE_RULES = ('natural', 'zero_flux')


def _bc_coefficients(primal, bary, rule):
    """COO (bary function id, primal RWG id, value) of every BC function.

    BC function e = (a, b) carries unit flux from the dual cell of a to that of
    b across the two dual-edge halves (m_e, c_T+-), half through each. A
    closed fan holds uniform charge +-1/M on its M triangles, with zero radial
    flux across (v, m_e). An open fan at a boundary vertex follows ``rule``:

    * ``natural``: the cell is uncharged and the unit flux enters (or leaves)
      through the two boundary half-edges of the fan, half through each, so the
      dual space is the non-tilde H^-1/2(div) with free boundary normal flux;
    * ``zero_flux``: uniform charge +-1/M and zero boundary flux (tilde space).

    Radial fluxes follow from charge balance along the fan.
    """
    vertex_count = len(primal.mesh.vertices)
    children = bary.mesh.faces.cpu().tolist()
    lookup = {tuple(edge): index for index, edge in enumerate(bary.edges.cpu().tolist())}
    plus = bary.plus_faces.cpu().tolist()
    lengths = bary.edge_lengths.cpu().tolist()
    cells, radial, sharing = defaultdict(list), {}, defaultdict(list)
    for face, (a, b, c) in enumerate(children):
        vertex, midpoint = (a, b) if a < vertex_count else (b, a)
        cells[vertex].append(face)
        radial[face] = (vertex, midpoint, c)
        for other in (midpoint, c):
            sharing[(min(vertex, other), max(vertex, other))].append(face)

    def keys(face):
        vertex, midpoint, centroid = radial[face]
        return [(min(vertex, other), max(vertex, other)) for other in (midpoint, centroid)]

    def emit(key, source_face, flux, basis_id):
        index = lookup[key]
        rows.append(index), cols.append(basis_id)
        values.append(flux/lengths[index]*(1.0 if plus[index] == source_face else -1.0))

    edge_index = {tuple(e): i for i, e in enumerate(primal.mesh.topology.edges.cpu().tolist())}
    rows, cols, values = [], [], []
    for basis_id, (a, b) in enumerate(primal.edges.cpu().tolist()):
        midpoint = vertex_count+edge_index[(a, b)]
        for vertex, sign in ((a, 1.0), (b, -1.0)):
            closing = (min(vertex, midpoint), max(vertex, midpoint))
            faces = cells[vertex]
            exits = [f for f in faces if radial[f][1] == midpoint]
            ends = [f for f in faces if any(len(sharing[key]) == 1 for key in keys(f))]
            if len(exits) != 2 or len(ends) not in (0, 2):
                raise RuntimeError('BC construction requires a single manifold fan per vertex')
            blocked = None if ends else closing
            order = [ends[0] if ends else exits[0]]
            while len(order) < len(faces):
                following = [g for key in keys(order[-1]) if key != blocked for g in sharing[key]
                             if g != order[-1] and g not in order]
                if len(following) != 1:
                    raise RuntimeError('Dual cell is not a single fan')
                order.append(following[0])
            natural = bool(ends) and rule == 'natural'
            source = 0.0 if natural else sign/len(faces)
            flux = sign/2 if natural else 0.0
            if natural:
                for face in (order[0], order[-1]):
                    boundary = next(key for key in keys(face) if len(sharing[key]) == 1)
                    emit(boundary, face, -sign/2, basis_id)
            for left, right in zip(order, order[1:]):
                flux += source-(sign/2 if left in exits else 0.0)
                emit((set(keys(left)) & set(keys(right))).pop(), left, flux, basis_id)
            if sign > 0:
                for face in exits:
                    emit((min(midpoint, radial[face][2]), max(midpoint, radial[face][2])), face, 0.5, basis_id)
    return rows, cols, values


def _sparse_divergence(basis):
    """Piecewise-constant RWG divergence [T, E] as a real sparse COO matrix."""
    faces = torch.arange(len(basis.mesh.faces), device=basis.mesh.device)[:, None].expand(-1, 3)
    valid = basis.face_signs != 0
    values = (basis.face_signs*basis.edge_lengths[basis.face_edges]/basis.mesh.areas[:, None])[valid]
    return torch.sparse_coo_tensor(torch.stack((faces[valid], basis.face_edges[valid])), values,
                                   (len(basis.mesh.faces), basis.n_edges), check_invariants=True).coalesce()


class DualBasis:
    """BC functions as sparse combinations of barycentric RWG (and boundary half) functions."""

    def __init__(self, primal: OpenRWGBasis, *, open_edge_rule='natural'):
        if open_edge_rule not in OPEN_EDGE_RULES:
            raise ValueError(f'open_edge_rule must be one of {OPEN_EDGE_RULES}')
        self.primal, self.open_edge_rule = primal, open_edge_rule
        self.bary = OpenRWGBasis(barycentric_refinement(primal.mesh), boundary_half_functions=True)
        rows, cols, values = _bc_coefficients(primal, self.bary, open_edge_rule)
        device, dtype = primal.mesh.device, primal.mesh.dtype
        self.coefficients = torch.sparse_coo_tensor(
            torch.tensor([rows, cols], device=device), torch.tensor(values, device=device, dtype=dtype),
            (self.bary.n_edges, primal.n_edges), check_invariants=True).coalesce()
        # Slot table: bary face slot (t, l) -> BC coefficients; boundary slots are empty.
        slot_edges = self.bary.face_edges.reshape(-1)
        valid = self.bary.face_signs.reshape(-1) != 0
        selection = torch.sparse_coo_tensor(
            torch.stack((torch.arange(len(slot_edges), device=device)[valid], slot_edges[valid])),
            torch.ones(int(valid.sum()), device=device, dtype=dtype), (len(slot_edges), self.bary.n_edges),
            check_invariants=True)
        self.slots = _sparse_product(selection, self.coefficients)
        self.divergence = _sparse_product(_sparse_divergence(self.bary), self.coefficients)

    @property
    def n_functions(self):
        return self.primal.n_edges


def _sparse_product(left, right):
    """Sparse [m,k] @ sparse [k,n] for real COO tensors on CUDA, coalesced."""
    return torch.sparse.mm(left.coalesce(), right.coalesce()).coalesce()


def _slot_rows(sparse, start, stop, columns):
    """Dense rows [start, stop) of a coalesced COO matrix."""
    indices, values = sparse.indices(), sparse.values()
    low = torch.searchsorted(indices[0], torch.tensor(start, device=indices.device))
    high = torch.searchsorted(indices[0], torch.tensor(stop, device=indices.device))
    result = torch.zeros((stop-start, columns), device=values.device, dtype=values.dtype)
    result.index_put_((indices[0, low:high]-start, indices[1, low:high]), values[low:high], accumulate=True)
    return result


def _local_rows(basis, rows, k, config, kernel):
    """Single-layer local blocks of bary faces ``rows`` against every bary face."""
    count = len(basis.mesh.faces)
    rr = rows[:, None].expand(-1, count).reshape(-1)
    cc = torch.arange(count, device=rows.device)[None].expand(len(rows), -1).reshape(-1)
    classes = classify_pairs(basis.mesh, rr, cc)
    dtype = torch.complex128 if basis.mesh.dtype == torch.float64 else torch.complex64
    vector = torch.zeros((len(rr), 3, 3), device=rows.device, dtype=dtype)
    scalar = torch.zeros(len(rr), device=rows.device, dtype=dtype)
    for category in range(5):
        selected = (classes == category).nonzero().flatten()
        for ids in selected.split(config.far_pair_batch if category == 4 else config.pair_batch):
            if not len(ids):
                continue
            if category == 3:
                v, s, _ = adaptive_near_single_layer(basis, basis, rr[ids], cc[ids], k, config, kernel=kernel)
            else:
                order = config.order if category == 4 else config.singular_order
                v, s = single_layer_pair_integrals(basis, basis, rr[ids], cc[ids], k, order, category, kernel=kernel)
            vector[ids], scalar[ids] = v.to(dtype), s.to(dtype)
    return vector.reshape(len(rows), count, 3, 3), scalar.reshape(len(rows), count)


def assemble_dual_efie(dual, k, frequency, config, *, kernel, pair_budget=2_000_000):
    """Z_bc[m, n] = <b_m, L b_n>, streamed over barycentric face rows; never stores the bary matrix.

    Each row chunk integrates at most ``pair_budget`` face pairs against all faces.
    """
    basis, n = dual.bary, dual.n_functions
    faces = len(basis.mesh.faces)
    row_batch = max(1, pair_budget//faces)
    dtype = torch.complex128 if basis.mesh.dtype == torch.float64 else torch.complex64
    vector = torch.zeros((n, n), device=basis.mesh.device, dtype=dtype)
    scalar = torch.zeros_like(vector)
    slots_t = dual.slots.t().coalesce()
    divergence_t = dual.divergence.t().coalesce()
    for start in range(0, faces, row_batch):
        rows = torch.arange(start, min(start+row_batch, faces), device=basis.mesh.device)
        local, potential = _local_rows(basis, rows, k, config, kernel)
        block = local.permute(0, 2, 1, 3).reshape(3*len(rows), 3*faces)
        # Y = block @ slots, with the real sparse slot table applied to each part.
        y = torch.complex(torch.sparse.mm(slots_t, block.real.T.contiguous()).T,
                          torch.sparse.mm(slots_t, block.imag.T.contiguous()).T)
        left = _slot_rows(dual.slots, 3*start, 3*start+3*len(rows), n).to(dtype)
        vector.addmm_(left.T, y)
        y = torch.complex(torch.sparse.mm(divergence_t, potential.real.T.contiguous()).T,
                          torch.sparse.mm(divergence_t, potential.imag.T.contiguous()).T)
        left = _slot_rows(dual.divergence, start, start+len(rows), n).to(dtype)
        scalar.addmm_(left.T, y)
    omega_mu = 2*math.pi*frequency*MU_0
    vector.sub_(scalar/k**2).mul_(1j*omega_mu)
    return vector


def mixed_gram(dual):
    """G[m, n] = <f_m, n x b_n> over barycentric faces (exact quadratic quadrature)."""
    primal, bary = dual.primal, dual.bary
    points, weights = triangle_rule(bary.mesh.triangles, 2)
    parent = torch.arange(len(bary.mesh.faces), device=points.device)//6
    triangles = primal.mesh.triangles[parent]
    slopes = (primal.face_signs*primal.edge_lengths[primal.face_edges]/(2*primal.mesh.areas[:, None]))[parent]
    primal_values = slopes[:, None, :, None]*(points[:, :, None]-triangles[:, None])
    bary_values = bary.local_values(points)
    rotated = torch.linalg.cross(bary.mesh.normals[:, None, None].expand_as(bary_values), bary_values)
    local = torch.einsum('tqlc,tqmc,tq->tlm', primal_values, rotated, weights)
    n = primal.n_edges
    gram = torch.zeros((n, n), device=points.device, dtype=points.dtype)
    rows_valid = primal.face_signs[parent] != 0
    slots = dual.slots
    indices, values = slots.indices(), slots.values()
    face, bary_slot = indices[0]//3, indices[0] % 3
    for primal_slot in range(3):
        weight = local[face, primal_slot, bary_slot]*values
        valid = rows_valid[face, primal_slot]
        target = primal.face_edges[parent[face], primal_slot]
        gram.index_put_((target[valid], indices[1][valid]), weight[valid], accumulate=True)
    return gram


def primal_star_incidence(basis):
    """Sigma[e, T] = sign of RWG e on face T (stars are face charges)."""
    result = torch.zeros((basis.n_edges, len(basis.mesh.faces)), device=basis.mesh.device, dtype=basis.mesh.dtype)
    faces = torch.arange(len(basis.mesh.faces), device=basis.mesh.device)[:, None].expand(-1, 3)
    valid = basis.face_signs != 0
    result.index_put_((basis.face_edges[valid], faces[valid]), basis.face_signs[valid].to(result.dtype),
                      accumulate=True)
    return result


def dual_star_incidence(basis, open_edge_rule='natural'):
    """Sigma'[e, v]: intra-sheet dual-cell charge (+1 source, -1 sink) of BC function e.

    Under the natural rule a boundary-vertex cell is uncharged, so its column is zero.
    """
    result = torch.zeros((basis.n_edges, len(basis.mesh.vertices)), device=basis.mesh.device, dtype=basis.mesh.dtype)
    ids = torch.arange(basis.n_edges, device=basis.mesh.device)
    result[ids, basis.edges[:, 0]] = 1
    result[ids, basis.edges[:, 1]] = -1
    if open_edge_rule == 'natural':
        result[:, basis.boundary_edges.reshape(-1)] = 0
    return result


def _require_open_efie(context):
    if tuple(context.components) != ('J',):
        raise ValueError('Open-surface EFIE preconditioners require the single-current EFIE layout')
    if len(context.system.bodies) != 1:
        raise NotImplementedError(
            f'CalderonPreconditioner supports exactly one sheet; this scene has {len(context.system.bodies)}. '
            'Multi-sheet dual bases are not implemented; solve without it or with LeafBlockRightPreconditioner.')
    return context.system.bodies[0].basis


# Preconditioner-grade dual quadrature: the dual operator only needs to be
# spectrally close to the exact one, so near pairs stop at a looser tolerance.
DUAL_INTEGRATION = IntegrationConfig(order=3, singular_order=5, pair_batch=512, far_pair_batch=32768,
                                     near_tolerance=1e-3, near_max_order=13)


class CalderonPreconditioner:
    """Right preconditioner P = G^-T Z_bc G^-1 on Buffa-Christiansen functions.

    ``integration`` is the dual-operator quadrature (default ``DUAL_INTEGRATION``);
    it changes only the preconditioner, never the EFIE operator or its admission.
    """

    def __init__(self, *, open_edge_rule='natural', integration=DUAL_INTEGRATION, pair_budget=2_000_000):
        if open_edge_rule not in OPEN_EDGE_RULES:
            raise ValueError(f'open_edge_rule must be one of {OPEN_EDGE_RULES}')
        if type(pair_budget) is not int or pair_budget < 1:
            raise ValueError('pair_budget must be a positive integer')
        if not isinstance(integration, IntegrationConfig):
            raise TypeError('integration must be an IntegrationConfig')
        self.open_edge_rule = open_edge_rule
        self.integration, self.pair_budget = integration, pair_budget

    def bind(self, context):
        basis = _require_open_efie(context)
        handle = context.handle
        torch.cuda.synchronize(context.device)
        started = time.perf_counter()
        dual = DualBasis(basis, open_edge_rule=self.open_edge_rule)
        dual_matrix = assemble_dual_efie(dual, handle.exterior.k, handle.exterior.frequency, self.integration,
                                         kernel=handle.kernel, pair_budget=self.pair_budget).to(context.dtype)
        torch.cuda.synchronize(context.device)
        assembled = time.perf_counter()
        gram = mixed_gram(dual).to(context.dtype)
        factor, pivots, info = torch.linalg.lu_factor_ex(gram)
        if int(info) != 0:
            raise ValueError('Singular RWG/BC mixed Gram matrix')
        factor_t, pivots_t = torch.linalg.lu_factor(gram.mT.contiguous())
        singular = torch.linalg.svdvals(gram)
        torch.cuda.synchronize(context.device)
        receipt = {'version': CALDERON_VERSION, 'formulation': 'efie_open_pec', 'open_edge_rule': self.open_edge_rule,
                   'dual_integration': vars(self.integration),
                   'barycentric_faces': len(dual.bary.mesh.faces), 'barycentric_functions': dual.bary.n_edges,
                   'bc_nonzeros': int(dual.coefficients.values().numel()),
                   'mixed_gram_condition': float(singular[0]/singular[-1]),
                   'dual_assembly_seconds': assembled-started, 'setup_core_seconds': time.perf_counter()-started}

        def apply(residual):
            flat = residual.reshape(-1, residual.shape[-1]).mT
            inner = torch.linalg.lu_solve(factor, pivots, flat)
            outer = torch.linalg.lu_solve(factor_t, pivots_t, dual_matrix@inner)
            return outer.mT.reshape(residual.shape).contiguous()
        bound = _Bound(apply, receipt)
        bound.dual_matrix, bound.gram = dual_matrix, gram
        return bound


def preconditioned_residual(handle, x, b, apply):
    """True and preconditioned residual norms of one EFIE iterate.

    ``apply`` is a bound right preconditioner action. Right preconditioning
    leaves the GMRES residual equal to the true residual; P r is the
    Calderon-mapped residual, a better proxy for the current error on fine
    meshes. Both absolute and relative values are reported.
    """
    residual = b-handle.apply(x)
    mapped, mapped_b = apply(residual), apply(b)
    norm = lambda value: float(torch.linalg.vector_norm(value))
    return {'true_residual': norm(residual), 'true_relative_residual': norm(residual)/norm(b),
            'preconditioned_residual': norm(mapped), 'preconditioned_relative_residual': norm(mapped)/norm(mapped_b)}
