"""Rotated-test double-RWG PMCHWT assembly for fixed geometry.

Rows test (H,-E) against RWG, equivalent to testing the trace (J,M)
against rotated RWG. Unknowns retain physical units (A/m,V/m).
"""
from __future__ import annotations

from dataclasses import dataclass
import functools
import math
import time
import torch

from .kernels import FREE_SPACE
from .quadrature import (ANISOTROPIC_TOUCHING_ASPECT, COPLANAR_TOLERANCE, anisotropic_touching_pairs, classify_all_pairs,
                         distance_scaled_duffy_fan, duffy_fan, refine_toward_source_edges, static_potential_integrals,
                         touching_pair_rule, touching_remainder_nodes, touching_remainder_rule, triangle_rule)


@dataclass(frozen=True)
class IntegrationConfig:
    order: int = 4
    singular_order: int = 8
    pair_batch: int = 64
    far_pair_batch: int = 1024
    near_tolerance: float = 1e-5
    near_max_order: int | None = None

    def __post_init__(self):
        if self.near_max_order is None:
            object.__setattr__(self, "near_max_order", max(16, self.singular_order + 8))
        for name in ("order", "singular_order", "pair_batch", "far_pair_batch", "near_max_order"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not math.isfinite(self.near_tolerance) or self.near_tolerance <= 0:
            raise ValueError("near_tolerance must be positive and finite.")
        if self.near_max_order < self.singular_order + 2:
            raise ValueError("near_max_order must allow at least one order refinement.")


def local_geometry(basis):
    """Build local support indices, free vertices and signed RWG slopes."""
    slopes = basis.face_signs*basis.edge_lengths[basis.face_edges]/(2*basis.mesh.areas[:, None])
    return basis.face_edges, basis.mesh.triangles, slopes


class PreparedRWGGeometry:
    """Fixed CUDA geometry reused across selected blocks and quadrature orders."""
    def __init__(self, basis):
        self.basis = basis
        self._basis = basis
        self._inputs = tuple((id(t), t._version) for t in self.inputs())
        _, self.triangles, self.slopes = local_geometry(basis)
        if self.triangles.requires_grad or self.slopes.requires_grad:
            raise ValueError('Prepared RWG geometry requires fixed coordinates')
        self.centres = self.triangles.mean(1)
        self.radii = torch.linalg.vector_norm(self.triangles-self.centres[:, None], dim=-1).amax(-1)
        self._versions = tuple((id(t), t._version) for t in self.tensors())

    def inputs(self):
        b = self.basis
        return (b.mesh.vertices, b.mesh.faces, b.edges, b.face_edges, b.face_signs,
                b.edge_lengths, b.plus_faces, b.minus_faces)

    def tensors(self):
        return (self.triangles, self.slopes, self.centres, self.radii)

    def assert_current(self, basis=None):
        if self.basis is not self._basis or (basis is not None and basis is not self.basis):
            raise ValueError('Prepared RWG geometry basis binding changed')
        if (tuple((id(t), t._version) for t in self.inputs()) != self._inputs
                or tuple((id(t), t._version) for t in self.tensors()) != self._versions):
            raise ValueError('Prepared RWG geometry changed; rebuild before integration')

    def local(self, basis):
        self.assert_current(basis)
        return basis.face_edges, self.triangles, self.slopes

    def receipt(self):
        self.assert_current()
        return {'version': 'prepared_rwg_geometry_v1', 'faces': len(self.triangles),
                'retained_bytes': sum(t.numel()*t.element_size() for t in self.tensors())}


def _complex_real_matmul(left, right):
    """Complex [...,M,K] @ real [...,K,N] in real arithmetic."""
    if not left.numel() or not right.numel():
        return left@right.to(left.dtype)
    pairs = torch.view_as_real(left).transpose(-1, -2).flatten(-3, -2)
    product = (pairs@right).unflatten(-2, (-1, 2)).transpose(-1, -2)
    return torch.view_as_complex(product.contiguous())


def _real_complex_matmul(left, right):
    """Real [...,M,K] @ complex [...,K,N] in real arithmetic."""
    if not left.numel() or not right.numel():
        return left.to(right.dtype)@right
    product = left@torch.view_as_real(right).flatten(-2)
    return torch.view_as_complex(product.unflatten(-1, (-1, 2)))


def _cross_moments(moments):
    """cross(D, c) from moments P[a,...,b]=sum D_a c_b with a on dim 1."""
    p = lambda a, b: moments[:, a, ..., b]
    return torch.stack((p(1, 2)-p(2, 1), p(2, 0)-p(0, 2), p(0, 1)-p(1, 0)), -1)


def _touching_integrals(basis, rows, cols, media, inner_order, category, *, kernel=FREE_SPACE,
                        zero_planar_self=True, geometry=None):
    """Joint-Duffy D/K blocks of self, edge and vertex pairs for each medium.

    Quadrature nodes and the real weighted basis products f.h and h x f are
    shared by every medium; all media contract with them in one real-arithmetic
    product over their stacked kernel samples.
    """
    _, free, slopes = local_geometry(basis) if geometry is None else geometry[0].local(basis)
    obs, src, weight = touching_pair_rule(free[rows], free[cols], basis.mesh.faces[rows],
                                          basis.mesh.faces[cols], category, inner_order)
    delta = obs-src
    f = slopes[rows, None, :, None]*(obs[:, :, None]-free[rows, None])
    h = slopes[cols, None, :, None]*(src[:, :, None]-free[cols, None])
    divergence = (2*slopes[rows, :, None])*(2*slopes[cols, None, :])
    dot = torch.einsum("bqlc,bqmc->bqlm", f, h)*weight[:, :, None, None]
    # f_l.(grad x h_m) = grad.(h_m x f_l), weighted once for all media.
    planar = category == 0 and zero_planar_self
    triple = None if planar else torch.linalg.cross(h[:, :, None], f[:, :, :, None], dim=-1)*weight[:, :, None, None, None]
    samples = [kernel.evaluate(delta, medium.k) for medium in media]
    dtype = torch.result_type(samples[0][1], 1j)
    pairs = lambda values: torch.view_as_real(values.to(dtype))
    # [B,Q,(medium,re/im)] and [B,Q,xyz,(medium,re/im)] real kernel stacks.
    green = torch.cat([pairs(g) for g, _ in samples], -1)
    vector = torch.view_as_complex(torch.einsum("bqr,bqlm->blmr", green, dot).unflatten(-1, (len(media), 2))
                                   .movedim(-2, 0).contiguous())
    if planar:
        magnetic = torch.zeros_like(vector)
    else:
        gradient = torch.cat([pairs(grad) for _, grad in samples], -1)
        magnetic = torch.view_as_complex(torch.einsum("bqar,bqlma->blmr", gradient, triple)
                                         .unflatten(-1, (len(media), 2)).movedim(-2, 0).contiguous())
    results = []
    for index, (medium, (g, _)) in enumerate(zip(media, samples)):
        scalar = (g*weight).sum(1)[:, None, None]*divergence
        results.append((vector[index]-scalar/(medium.k**2), magnetic[index]))
    return results


def _product_integrals(basis, rows, cols, media, order, inner_order, *, kernel=FREE_SPACE, source_basis=None,
                       zero_planar_self=True, geometry=None):
    """Regular product-rule D/K blocks for each medium via node-moment products.

    Each kernel sample is contracted with the source RWG values by batched
    matrix products over source nodes; no node-pair basis tensor is formed.
    """
    source_basis = basis if source_basis is None else source_basis
    _, free, slopes = local_geometry(basis) if geometry is None else geometry[0].local(basis)
    _, source_free, source_slopes = local_geometry(source_basis) if geometry is None else geometry[1].local(source_basis)
    obs, wo = triangle_rule(free[rows], order)
    src, ws = triangle_rule(source_free[cols], inner_order)
    delta = obs[:, :, None]-src[:, None]
    weight = wo[:, :, None]*ws[:, None]
    batch, nq, ns = weight.shape
    f = slopes[rows, None, :, None]*(obs[:, :, None]-free[rows, None])
    h = (source_slopes[cols, None, :, None]*(src[:, :, None]-source_free[cols, None])).flatten(2)
    divergence = (2*slopes[rows, :, None])*(2*source_slopes[cols, None, :])
    planar_self = (rows == cols)[:, None, None] if source_basis is basis and zero_planar_self else None
    results = []
    for medium in media:
        g, grad = kernel.evaluate(delta, medium.k)
        dtype = grad.dtype
        fc = f.to(dtype)
        weighted_g = (g*weight).to(dtype)
        vector = torch.einsum("bqlc,bqmc->blm", fc, _complex_real_matmul(weighted_g, h).unflatten(-1, (3, 3)))
        scalar = weighted_g.sum((1, 2))[:, None, None]*divergence
        weighted_grad = (grad*weight[..., None]).permute(0, 3, 1, 2).reshape(batch, 3*nq, ns)
        moments = _complex_real_matmul(weighted_grad.to(dtype), h).reshape(batch, 3, nq, 3, 3)
        magnetic = torch.einsum("bqlc,bqmc->blm", fc, _cross_moments(moments))
        # On a planar self panel this principal-value block is identically zero.
        if planar_self is not None:
            magnetic = torch.where(planar_self, 0, magnetic)
        results.append((vector-scalar/(medium.k**2), magnetic))
    return results


def _pair_integrals(basis, rows, cols, medium, order, inner_order, singular, category=None, *, kernel=FREE_SPACE, source_basis=None, zero_planar_self=True, geometry=None):
    source_basis = basis if source_basis is None else source_basis
    if category in (0, 1, 2):
        if source_basis is not basis:
            raise ValueError('Different bodies cannot have topologically touching quadrature pairs')
        return _touching_integrals(basis, rows, cols, (medium,), inner_order, category, kernel=kernel,
                                   zero_planar_self=zero_planar_self, geometry=geometry)[0]
    if not singular:
        return _product_integrals(basis, rows, cols, (medium,), order, inner_order, kernel=kernel,
            source_basis=source_basis, zero_planar_self=zero_planar_self, geometry=geometry)[0]
    _, free, slopes = local_geometry(basis) if geometry is None else geometry[0].local(basis)
    _, source_free, source_slopes = local_geometry(source_basis) if geometry is None else geometry[1].local(source_basis)
    obs, wo = triangle_rule(free[rows], order)
    batch, nq = obs.shape[:2]
    src, ws = duffy_fan(obs.reshape(-1, 3), source_free[cols, None].expand(-1, nq, -1, -1).reshape(-1, 3, 3), inner_order)
    src, ws = src.reshape(batch, nq, -1, 3), ws.reshape(batch, nq, -1)
    delta = obs[:, :, None]-src
    g, grad = kernel.evaluate(delta, medium.k)
    weight = wo[:, :, None]*ws
    f = slopes[rows, None, :, None]*(obs[:, :, None]-free[rows, None])
    h = source_slopes[cols, None, None, :, None]*(src[:, :, :, None]-source_free[cols, None, None])
    weighted_g = (g*weight).to(grad.dtype)
    f, h = f.to(grad.dtype), h.to(grad.dtype)
    vector = torch.einsum("bqlc,bqsmc,bqs->blm", f, h, weighted_g)
    scalar = weighted_g.sum((1, 2))[:, None, None]*(2*slopes[rows, :, None])*(2*source_slopes[cols, None, :])
    d = vector-scalar/(medium.k**2)
    cross = torch.linalg.cross(grad[:, :, :, None], h, dim=-1)
    magnetic = torch.einsum("bqlc,bqsmc,bqs->blm", f, cross, weight.to(grad.dtype))
    if source_basis is basis and zero_planar_self:
        magnetic = torch.where((rows == cols)[:, None, None], 0, magnetic)
    return d, magnetic


# Near-disjoint pairs that the ordinary product rule cannot resolve within
# near_max_order (thin gaps, high-aspect slivers) fall back to a gap-aware
# rule: the observer triangle is subdivided toward the source edges and each
# observer node integrates the source with the distance-scaled, edge-graded
# fan (which already resolves source anisotropy and small heights). Levels are (outer Gauss order, inner fan order, split ratio); results
# of consecutive levels are compared with the same tolerance.
NEAR_FALLBACK_RULE = 'observer_anisotropic_bisection_distance_scaled_fan_v1'
NEAR_FALLBACK_LEVELS = ((4, 8, .5), (6, 10, 1.), (8, 12, 2.), (10, 14, 4.))
NEAR_FALLBACK_MAX_DEPTH = 28
NEAR_FALLBACK_NODES = 1 << 22
NEAR_FALLBACK_BATCH = 512
# Touching pairs above ANISOTROPIC_TOUCHING_ASPECT use singularity
# subtraction: the static 1/(4 pi R) part integrates the source in closed
# form at every observer node, on observer leaves graded toward the shared
# edges and vertices where those potentials are only log-singular; the
# bounded dynamic remainder uses a product rule. Levels are (outer order,
# remainder order, split ratio). Other kernels use the fan-based rule.
TOUCHING_ANISOTROPIC_RULE = 'touching_static_subtraction_graded_observer_v1'
TOUCHING_FAN_RULE = 'touching_anisotropic_bisection_distance_scaled_fan_v1'
TOUCHING_LEVELS = ((4, 6, .5), (6, 8, 1.), (8, 10, 2.), (10, 12, 4.), (10, 16, 4.), (10, 20, 4.), (10, 24, 4.))
TOUCHING_MAX_DEPTH = 20
TOUCHING_REMAINDER_NODES = 1 << 21


def _fan_near_integrals(basis, rows, cols, medium, outer_order, inner_order, ratio, *, kernel=FREE_SPACE,
                        source_basis=None, geometry=None, isotropic=False, max_level=NEAR_FALLBACK_MAX_DEPTH):
    """Gap-aware near-disjoint D/K blocks; source-side sums are reduced per observer node."""
    source_basis = basis if source_basis is None else source_basis
    _, free, slopes = local_geometry(basis) if geometry is None else geometry[0].local(basis)
    _, source_free, source_slopes = local_geometry(source_basis) if geometry is None else geometry[1].local(source_basis)
    leaves, owner, depth = refine_toward_source_edges(free[rows], source_free[cols], ratio=ratio,
                                                      max_level=max_level, isotropic=isotropic)
    dtype = torch.complex128 if free.dtype == torch.float64 else torch.complex64
    d = torch.zeros((len(rows), 3, 3), device=free.device, dtype=dtype)
    magnetic = torch.zeros_like(d)
    inner = 3*inner_order**2
    step = max(1, NEAR_FALLBACK_NODES//(outer_order**2*inner))
    for start in range(0, len(leaves), step):
        leaf, pair = leaves[start:start+step], owner[start:start+step]
        obs, wo = triangle_rule(leaf, outer_order)
        q = obs.shape[1]
        r, c = rows[pair], cols[pair]
        src, ws = distance_scaled_duffy_fan(obs.reshape(-1, 3), source_free[c, None].expand(-1, q, -1, -1).reshape(-1, 3, 3),
                                            inner_order, edge_grading=True, coplanar_tolerance=COPLANAR_TOLERANCE)
        src, ws = src.reshape(len(leaf), q, inner, 3), ws.reshape(len(leaf), q, inner)
        g, grad = kernel.evaluate(obs[:, :, None]-src, medium.k)
        gw = (g*ws).to(dtype)
        total = gw.sum(-1)
        moment = torch.einsum('bqs,bqsc->bqc', gw, src.to(dtype))
        gradient = (grad*ws[..., None]).to(dtype)
        curl_moment = torch.linalg.cross(gradient, src.to(dtype).expand_as(gradient), dim=-1).sum(-2)
        gradient = gradient.sum(-2)
        f = (slopes[r, None, :, None]*(obs[:, :, None]-free[r, None])).to(dtype)
        wo = wo.to(dtype)
        sf, ss = source_free[c].to(dtype), source_slopes[c].to(dtype)
        # h_m(y) = s_m (y - v_m): integrate the source factors first.
        vector = torch.einsum('bqlc,bqmc,bq->blm', f, moment[:, :, None]-total[..., None, None]*sf[:, None], wo)*ss[:, None, :]
        scalar = (total*wo).sum(1)[:, None, None]*(2*slopes[r, :, None].to(dtype))*(2*ss[:, None, :])
        curl = curl_moment[:, :, None]-torch.linalg.cross(gradient[:, :, None].expand(-1, -1, 3, -1), sf[:, None].expand(-1, q, -1, -1), dim=-1)
        local_k = torch.einsum('bqlc,bqmc,bq->blm', f, curl, wo)*ss[:, None, :]
        d.index_add_(0, pair, vector-scalar/(medium.k**2))
        magnetic.index_add_(0, pair, local_k)
    return d, magnetic, {'leaves': len(leaves), 'depth': depth}


def _subtracted_touching_integrals(basis, rows, cols, medium, outer_order, inner_order, ratio, *, category, kernel=FREE_SPACE,
                                   source_basis=None, geometry=None, max_level=TOUCHING_MAX_DEPTH):
    """Touching D/K blocks: closed-form static source integrals plus a product-rule remainder."""
    if kernel is not FREE_SPACE or source_basis is not None or geometry is not None:
        raise ValueError('Static subtraction is defined for one free-space basis only')
    from .lifting import _gradient_remainder
    _, free, slopes = local_geometry(basis)
    dtype = torch.complex128 if free.dtype == torch.float64 else torch.complex64
    k = medium.k
    d = torch.zeros((len(rows), 3, 3), device=free.device, dtype=dtype)
    magnetic = torch.zeros_like(d)
    # Static part on observer leaves graded toward the shared features.
    leaves, owner, depth = refine_toward_source_edges(free[rows], free[cols], ratio=ratio, max_level=max_level)
    step = max(1, NEAR_FALLBACK_NODES//(64*outer_order**2))
    for start in range(0, len(leaves), step):
        leaf, pair = leaves[start:start+step], owner[start:start+step]
        obs, wo = triangle_rule(leaf, outer_order)
        q = obs.shape[1]
        r, c = rows[pair], cols[pair]
        source = free[c]
        i0, i1, gradient = static_potential_integrals(obs.reshape(-1, 3), source[:, None].expand(-1, q, -1, -1).reshape(-1, 3, 3))
        i0, i1, gradient = i0.reshape(len(leaf), q), i1.reshape(len(leaf), q, 3), gradient.reshape(len(leaf), q, 3)
        normal = torch.linalg.cross(source[:, 1]-source[:, 0], source[:, 2]-source[:, 0])
        normal = normal/torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
        projected = obs-((obs-source[:, None, 0])*normal[:, None]).sum(-1, keepdim=True)*normal[:, None]
        f = slopes[r, None, :, None]*(obs[:, :, None]-free[r, None])            # [L,Q,l,3]
        hm = projected[:, :, None]-source[:, None]                              # x_p - q_m, [L,Q,m,3]
        weight = wo/(4*math.pi)
        # int (y - q_m)/R dy = I1 + (x_p - q_m) I0.
        vector = (torch.einsum('bqlc,bqc,bq->bl', f, i1, weight)[:, :, None]
                  + torch.einsum('bqlc,bqmc,bq->blm', f, hm, weight*i0))*slopes[c, None, :]
        scalar = (weight*i0).sum(1)[:, None, None]*(2*slopes[r, :, None])*(2*slopes[c, None, :])
        # int grad_x(1/R) x (y - q_m) dy = grad x (x - q_m).
        cross = torch.linalg.cross(gradient[:, :, None].expand(-1, -1, 3, -1), obs[:, :, None]-source[:, None], dim=-1)
        local_k = torch.einsum('bqlc,bqmc,bq->blm', f, cross, weight)*slopes[c, None, :]
        d.index_add_(0, pair, (vector-scalar/k**2).to(dtype))
        magnetic.index_add_(0, pair, local_k.to(dtype))
    # Dynamic remainder G - 1/(4 pi R) = ik/(4 pi) - k^2 R/(8 pi) + ...: bounded but
    # with a cone at R = 0, so it keeps the joint touching rule, whose
    # anisotropy error on this weak remainder is far below that on 1/R.
    step = max(1, TOUCHING_REMAINDER_NODES//touching_remainder_nodes(category, inner_order))
    for first in range(0, len(rows), step):
        r, c = rows[first:first+step], cols[first:first+step]
        obs, src, weight = touching_remainder_rule(free[r], free[c], basis.mesh.faces[r], basis.mesh.faces[c], category, inner_order)
        delta = obs-src
        distance = torch.linalg.vector_norm(delta, dim=-1)
        safe = torch.where(distance > 0, distance, torch.ones_like(distance))
        z = 1j*k*safe.to(dtype)
        g = torch.where(distance > 0, torch.expm1(z)/(4*math.pi*safe), torch.full_like(z, 1j*k/(4*math.pi)))
        grad = torch.where((distance > 0)[..., None], delta.to(dtype)*(_gradient_remainder(z)/(4*math.pi*safe**3))[..., None], 0)
        weight = weight.to(dtype)
        f = (slopes[r, None, :, None]*(obs[:, :, None]-free[r, None])).to(dtype)
        h = (slopes[c, None, :, None]*(src[:, :, None]-free[c, None])).to(dtype)
        vector = torch.einsum('bqlc,bqmc,bq->blm', f, h, g*weight)
        scalar = (g*weight).sum(1)[:, None, None]*(2*slopes[r, :, None])*(2*slopes[c, None, :])
        cross = torch.linalg.cross(grad[:, :, None].expand_as(h), h, dim=-1)
        d[first:first+step] += vector-scalar/k**2
        magnetic[first:first+step] += torch.einsum('bqlc,bqmc,bq->blm', f, cross, weight)
    return d, magnetic, {'leaves': len(leaves), 'depth': depth}


def _near_relative_difference(d, magnetic, previous_d, previous_magnetic, medium, dtype):
    d_scale = torch.linalg.vector_norm(d, dim=(-2, -1)).clamp_min(torch.finfo(dtype).tiny)
    # K and k*D share area units. Symmetry can make K exactly zero;
    # normalize against the local electric block rather than roundoff.
    k_scale = torch.maximum(torch.linalg.vector_norm(magnetic, dim=(-2, -1)), d_scale*abs(medium.k))
    d_error = torch.linalg.vector_norm(d-previous_d, dim=(-2, -1))/d_scale
    k_error = torch.linalg.vector_norm(magnetic-previous_magnetic, dim=(-2, -1))/k_scale
    return torch.maximum(d_error, k_error)


def _fallback_near_integrals(basis, rows, cols, medium, config, *, kernel, source_basis, geometry,
                             rule=NEAR_FALLBACK_RULE, planar_self=False, integrator=None, levels=None):
    """Resolve pairs with the gap-aware rule; raise only if every level disagrees.

    ``planar_self`` marks coincident planar pairs, whose principal-value K
    block is identically zero, as in the joint self rule.
    """
    pending = torch.arange(len(rows), device=rows.device)
    d = magnetic = None
    stats = {'rule': rule, 'pairs': len(rows), 'pair_evaluations': 0, 'maximum_level': 0,
             'maximum_depth': 0, 'maximum_leaves_per_call': 0, 'maximum_accepted_relative_difference': 0.0}
    integrator = _fan_near_integrals if integrator is None else integrator
    levels = NEAR_FALLBACK_LEVELS if levels is None else levels
    stats['levels'] = [list(level) for level in levels]
    for level, (outer, inner, ratio) in enumerate(levels):
        updated_d, updated_k, info = integrator(basis, rows[pending], cols[pending], medium, outer, inner, ratio,
                                                kernel=kernel, source_basis=source_basis, geometry=geometry)
        stats['pair_evaluations'] += len(pending)
        stats['maximum_depth'] = max(stats['maximum_depth'], info['depth'])
        stats['maximum_leaves_per_call'] = max(stats['maximum_leaves_per_call'], info['leaves'])
        if planar_self:
            updated_k = torch.zeros_like(updated_k)
        if d is None:
            d, magnetic = updated_d, updated_k
            continue
        relative = _near_relative_difference(updated_d, updated_k, d[pending], magnetic[pending], medium, basis.mesh.dtype)
        d[pending], magnetic[pending] = updated_d, updated_k
        accepted = relative <= config.near_tolerance
        if bool(accepted.any()):
            stats['maximum_accepted_relative_difference'] = max(stats['maximum_accepted_relative_difference'],
                                                                float(relative[accepted].max()))
            stats['maximum_level'] = level
        pending = pending[~accepted]
        if not len(pending):
            return d, magnetic, stats
    raise RuntimeError(f"Panel quadrature failed tolerance {config.near_tolerance:g} in the {rule} rule"
                       f"{' after order '+str(config.near_max_order) if rule == NEAR_FALLBACK_RULE else ''}; "
                       f"maximum relative level difference {float(relative.max()):.3g}.")


def _anisotropic_touching_integrals(basis, rows, cols, medium, config, category, *, kernel=FREE_SPACE):
    """High-aspect touching pairs: static subtraction for the free-space kernel, fan rule otherwise."""
    if kernel is FREE_SPACE:
        return _fallback_near_integrals(basis, rows, cols, medium, config, kernel=kernel, source_basis=None, geometry=None,
            rule=TOUCHING_ANISOTROPIC_RULE, planar_self=category == 0,
            integrator=functools.partial(_subtracted_touching_integrals, category=category), levels=TOUCHING_LEVELS)
    return _fallback_near_integrals(basis, rows, cols, medium, config, kernel=kernel, source_basis=None, geometry=None,
                                    rule=TOUCHING_FAN_RULE, planar_self=category == 0)


def _merge_rule_stats(container, key, record):
    """Accumulate one gap-aware rule record (near fallback or anisotropic touching) under ``key``."""
    merged = container.setdefault(key, {"rule": record["rule"], "pairs": 0, "pair_evaluations": 0, "maximum_level": 0,
        "maximum_depth": 0, "maximum_leaves_per_call": 0, "maximum_accepted_relative_difference": 0.0,
        "levels": record["levels"]})
    for name in ("pairs", "pair_evaluations"):
        merged[name] += record[name]
    for name in ("maximum_level", "maximum_depth", "maximum_leaves_per_call", "maximum_accepted_relative_difference"):
        merged[name] = max(merged[name], record[name])


def _merge_near_stats(total, stats):
    """Accumulate near statistics; the fallback record appears only when it was used."""
    total["maximum_order"] = max(total["maximum_order"], stats["maximum_order"])
    total["pair_evaluations"] += stats["pair_evaluations"]
    total["maximum_accepted_relative_difference"] = max(total["maximum_accepted_relative_difference"],
                                                        stats["maximum_accepted_relative_difference"])
    if "fallback" in stats:
        _merge_rule_stats(total, "fallback", stats["fallback"])


def _pair_budget(config):
    """Node pairs per batch: the larger configured far/singular batch size."""
    return max(config.far_pair_batch*config.order**4, config.pair_batch*config.singular_order**4)


# Joint-Duffy nodes per touching batch beyond the configured pair batch.
_TOUCHING_NODES = 1 << 20


def _touching_batch(config, category, carriers=1):
    """Touching pairs per batch: the configured pair batch or about 1M nodes.

    Joint-Duffy nodes carry 3x3 basis products (and carrier traces in the
    lifted RHS), so the workspace shrinks with more than four carriers.
    """
    nodes = (6 if category < 2 else 2)*config.singular_order**4
    return max(1, (4*max(config.pair_batch, _TOUCHING_NODES//nodes))//max(4, carriers))


def _adaptive_near_media(basis, rows, cols, media, config, *, kernel=FREE_SPACE, source_basis=None, geometry=None):
    """Refine disjoint near-panel orders until both blocks stabilize per medium.

    Every pair keeps its own successive-order acceptance for each medium.
    All pending pairs advance one order at a time and share quadrature
    geometry between media; accepted entries are frozen.
    """
    budget = _pair_budget(config)

    def evaluate(r, c, order, selected):
        per_pair = max(config.pair_batch, budget//order**4)
        parts = [_product_integrals(basis, r[ids], c[ids], tuple(media[i] for i in selected), order, order,
                                    kernel=kernel, source_basis=source_basis, geometry=geometry)
                 for ids in torch.arange(len(r), device=r.device).split(per_pair)]
        return [tuple(torch.cat([part[i][block] for part in parts]) for block in range(2)) for i in range(len(selected))]

    order = config.singular_order
    if not len(rows):
        empty = _product_integrals(basis, rows, cols, media, order, order, kernel=kernel,
                                   source_basis=source_basis, geometry=geometry)
        return empty, [{"maximum_order": order, "pair_evaluations": 0,
                        "maximum_accepted_relative_difference": 0.0} for _ in media]
    values = [list(value) for value in evaluate(rows, cols, order, range(len(media)))]
    pending = torch.ones((len(media), len(rows)), dtype=torch.bool, device=rows.device)
    evaluations = [len(rows)]*len(media)
    maximum_error = [0.0]*len(media)
    final_order = [order]*len(media)
    tiny = torch.finfo(basis.mesh.dtype).tiny
    while True:
        active = pending.any(0).nonzero().flatten()
        if not len(active):
            break
        order = min(order + 2, config.near_max_order)
        mask = pending[:, active]
        selected = [i for i, refined in enumerate(mask.any(1).tolist()) if refined]
        updated = evaluate(rows[active], cols[active], order, selected)
        worst = 0.0
        for (updated_d, updated_k), i in zip(updated, selected):
            medium, local = media[i], mask[i]
            d, magnetic = values[i][0][active], values[i][1][active]
            d_scale = torch.linalg.vector_norm(updated_d, dim=(-2, -1)).clamp_min(tiny)
            # K and k*D share area units. Symmetry can make K exactly zero;
            # normalize against the local electric block rather than roundoff.
            k_scale = torch.maximum(torch.linalg.vector_norm(updated_k, dim=(-2, -1)), d_scale*abs(medium.k))
            d_error = torch.linalg.vector_norm(updated_d-d, dim=(-2, -1))/d_scale
            k_error = torch.linalg.vector_norm(updated_k-magnetic, dim=(-2, -1))/k_scale
            relative = torch.maximum(d_error, k_error)
            values[i][0][active] = torch.where(local[:, None, None], updated_d, d)
            values[i][1][active] = torch.where(local[:, None, None], updated_k, magnetic)
            accepted = local & (relative <= config.near_tolerance)
            evaluations[i] += int(local.sum())
            maximum_error[i] = max(maximum_error[i], float(torch.where(accepted, relative, 0).max()))
            final_order[i] = order
            pending[i, active] = local & ~accepted
            worst = max(worst, float(relative[local].max()))
        if order == config.near_max_order and bool(pending.any()):
            break
    stats = [{"maximum_order": final_order[i], "pair_evaluations": evaluations[i],
              "maximum_accepted_relative_difference": maximum_error[i]} for i in range(len(media))]
    # Pairs still changing at near_max_order go to the gap-aware fallback;
    # pairs the product rule accepted keep their values unchanged.
    for i, medium in enumerate(media):
        unresolved = pending[i].nonzero().flatten()
        for ids in unresolved.split(NEAR_FALLBACK_BATCH) if len(unresolved) else ():
            d, magnetic, record = _fallback_near_integrals(basis, rows[ids], cols[ids], medium, config, kernel=kernel,
                                                           source_basis=source_basis, geometry=geometry)
            values[i][0][ids], values[i][1][ids] = d, magnetic
            _merge_rule_stats(stats[i], "fallback", record)
    return [tuple(value) for value in values], stats


def _adaptive_near_integrals(basis, rows, cols, medium, config, *, kernel=FREE_SPACE, source_basis=None, geometry=None):
    """Refine disjoint near-panel orders until both integral blocks stabilize."""
    values, stats = _adaptive_near_media(basis, rows, cols, (medium,), config, kernel=kernel,
                                         source_basis=source_basis, geometry=geometry)
    return (*values[0], stats[0])


def assemble_weak(basis, exterior, interior, config=IntegrationConfig(), *, kernel=FREE_SPACE):
    """Sum the exterior and interior scalar weak regional operators."""
    return _assemble_scalar_regions(basis, (exterior, interior), config, kernel=kernel)


def assemble_region_weak(basis, medium, config=IntegrationConfig(), *, kernel=FREE_SPACE):
    """One scalar regional contribution for composition with tensor regions."""
    return _assemble_scalar_regions(basis, (medium,), config, kernel=kernel)


def _assemble_scalar_regions(basis, media, config, *, kernel):
    """Assemble a CUDA dense matrix; report all quadrature choices and costs.

    Self, edge and vertex pairs use joint-domain Duffy transformations.
    Near-disjoint pairs adapt regular product rules; numerical convergence
    remains an explicit acceptance obligation recorded in the metadata.
    Quadrature geometry is shared by all media and their contributions are
    summed before one scatter per pair batch.
    """
    if basis.mesh.vertices.device.type != "cuda":
        raise ValueError("MoM assembly requires CUDA.")
    from ..constants import EPSILON_0, MU_0
    torch.cuda.synchronize(basis.mesh.vertices.device)
    start = time.perf_counter()
    face_edges, _, _ = local_geometry(basis)
    ne, nt = len(basis.edge_lengths), len(basis.mesh.faces)
    dtype = torch.complex128 if basis.mesh.vertices.dtype == torch.float64 else torch.complex64
    device = basis.mesh.vertices.device
    classes = classify_all_pairs(basis.mesh)
    # Classifier temporaries must be released before the dense matrix exists.
    # Scatter into the contiguous final matrix, not strided block views.
    matrix = torch.zeros((2*ne, 2*ne), device=device, dtype=dtype)
    counts = []
    near_stats = {"maximum_order": 0, "pair_evaluations": 0, "maximum_accepted_relative_difference": 0.0}
    budget = _pair_budget(config)

    def scatter(r, c, values):
        ri = face_edges[r, :, None].expand(-1, -1, 3).reshape(-1)
        ci = face_edges[c, None, :].expand(-1, 3, -1).reshape(-1)
        magnetic = sum(k for _, k in values)
        omegas = [2*math.pi*medium.frequency for medium in media]
        electric = sum(1j*omega*EPSILON_0*medium.epsilon_r*d for omega, medium, (d, _) in zip(omegas, media, values))
        dual = sum(-1j*omega*MU_0*medium.mu_r*d for omega, medium, (d, _) in zip(omegas, media, values))
        matrix.index_put_((torch.cat((ri, ri+ne, ri, ri+ne)), torch.cat((ci, ci+ne, ci+ne, ci))),
                          torch.cat([v.reshape(-1).to(dtype) for v in (magnetic, magnetic, electric, dual)]),
                          accumulate=True)

    # Touching pairs with a high-aspect triangle leave the joint Duffy rules
    # and are integrated by the anisotropy-aware rule after the main pass.
    triangles = basis.mesh.triangles
    anisotropic = []
    for category in range(5):
        selected = torch.nonzero(classes == category).flatten()
        counts.append(int(selected.numel()))
        if category < 3:
            mask = anisotropic_touching_pairs(triangles[selected//nt], triangles[selected % nt])
            if bool(mask.any()):
                anisotropic.append((category, selected[mask]))
                selected = selected[~mask]
        if category == 3:
            r, c = selected//nt, selected % nt
            values, stats = _adaptive_near_media(basis, r, c, media, config, kernel=kernel)
            for record in stats if len(r) else ():
                _merge_near_stats(near_stats, record)
            for ids in torch.arange(len(r), device=device).split(max(config.pair_batch, budget//81)):
                scatter(r[ids], c[ids], [(d[ids], k[ids]) for d, k in values])
            continue
        order = config.order if category == 4 else config.singular_order
        batch_size = (max(config.far_pair_batch, budget//order**4) if category == 4
                      else _touching_batch(config, category))
        for ids in selected.split(batch_size):
            r, c = ids//nt, ids % nt
            if category == 4:
                values = _product_integrals(basis, r, c, media, order, order, kernel=kernel)
            else:
                values = _touching_integrals(basis, r, c, media, order, category, kernel=kernel)
            scatter(r, c, values)
    touching_stats = {}
    for category, ids in anisotropic:
        for part in ids.split(NEAR_FALLBACK_BATCH):
            r, c = part//nt, part % nt
            values = []
            for medium in media:
                d, k, record = _anisotropic_touching_integrals(basis, r, c, medium, config, category, kernel=kernel)
                _merge_rule_stats(touching_stats, ('self', 'edge', 'vertex')[category], record)
                values.append((d, k))
            scatter(r, c, values)
    torch.cuda.synchronize(device)
    estimate = {"near_disjoint": near_stats, "touching_and_far": "order_convergence_required"}
    if touching_stats:
        touching_stats["aspect_threshold"] = ANISOTROPIC_TOUCHING_ASPECT
        estimate["touching_anisotropic"] = touching_stats
    return matrix, {
        "formulation_id": "pmchwt_exp_minus_iwt_total_trace_v1",
        "basis_id": "RWG", "test_space_id": "rotated_RWG",
        "unknown_order": ["J_A_per_m", "M_V_per_m"],
        "normal_convention": "outward", "row_scaling": [1.0, 1.0],
        "quadrature": vars(config), "triangle_pair_counts": counts,
        "touching_quadrature": "joint_duffy_self_6_edge_6_vertex_2_sectors",
        "quadrature_error_estimate": estimate, "reference_validated": False,
        "assembly_seconds": time.perf_counter()-start,
        "matrix_bytes": matrix.numel()*matrix.element_size(),
        "kernel_id": kernel.kernel_id, "kernel_version": kernel.version,
        "device": str(matrix.device), "dtype": str(matrix.dtype),
    }


def build_rhs(basis, exterior, sources, *, order=6):
    """Weak rotated-test right hand side [-<f,Hinc>, +<f,Einc>]."""
    from .physics import incident_fields
    edges, free, slope = local_geometry(basis)
    points, weights = triangle_rule(basis.mesh.vertices[basis.mesh.faces], order)
    dtype = torch.complex128 if points.dtype == torch.float64 else torch.complex64
    e = torch.zeros_like(points, dtype=dtype)
    h = torch.zeros_like(e)
    for source in sources:
        es, hs = incident_fields(source, points.reshape(-1, 3), exterior)
        e = e+es.reshape_as(e)
        h = h+hs.reshape_as(h)
    values = (slope[:, None, :, None]*(points[:, :, None]-free[:, None])).to(dtype)
    ne = len(basis.edge_lengths)
    b = torch.zeros(2*ne, device=points.device, dtype=dtype)
    for offset, field, sign in ((0, h, -1), (ne, e, 1)):
        local = sign*torch.einsum("tqlc,tqc,tq->tl", values, field, weights.to(dtype))
        b.index_add_(0, edges.flatten()+offset, local.flatten())
    return b


def classify_cross_pairs(test_basis, source_basis, rows, cols, *, near_ratio=2., geometry=None):
    """Different bodies have only near/far pairs, independent of local IDs."""
    from .mesh import triangle_pair_distances
    if geometry is None:
        left, right = test_basis.mesh.triangles[rows], source_basis.mesh.triangles[cols]
        radii = [torch.linalg.vector_norm(tri-tri.mean(1)[:, None], dim=-1).amax(-1) for tri in (left, right)]
    else:
        geometry[0].assert_current(test_basis)
        geometry[1].assert_current(source_basis)
        left, right = geometry[0].triangles[rows], geometry[1].triangles[cols]
        radii = [geometry[0].radii[rows], geometry[1].radii[cols]]
    distance = triangle_pair_distances(left, right)
    return torch.where(distance < near_ratio*(radii[0]+radii[1]), 3, 4).to(torch.uint8), distance


def assemble_cross(test_basis, source_basis, exterior, config=IntegrationConfig(), *, kernel=FREE_SPACE):
    """One off-diagonal PMCHWT block: exterior contribution only."""
    if test_basis is source_basis:
        raise ValueError('Cross assembly requires two separately validated bodies')
    if test_basis.mesh.device.type != 'cuda' or source_basis.mesh.device != test_basis.mesh.device or source_basis.mesh.dtype != test_basis.mesh.dtype:
        raise ValueError('Cross-body assembly requires matching CUDA geometry spaces')
    ni, nj = test_basis.n_edges, source_basis.n_edges
    fi, fj = len(test_basis.mesh.faces), len(source_basis.mesh.faces)
    dtype = torch.complex128 if test_basis.mesh.dtype == torch.float64 else torch.complex64
    matrix = torch.zeros((2*ni, 2*nj), dtype=dtype, device=test_basis.mesh.device)
    classes = torch.empty(fi*fj, dtype=torch.uint8, device=matrix.device)
    minimum = torch.tensor(torch.inf, device=matrix.device, dtype=test_basis.mesh.dtype)
    for start in range(0, fi*fj, 4096):
        stop = min(start+4096, fi*fj)
        ids = torch.arange(start, stop, device=matrix.device)
        category, distance = classify_cross_pairs(test_basis, source_basis, ids//fj, ids % fj)
        classes[start:stop] = category
        minimum = torch.minimum(minimum, distance.min())
    if not bool(minimum > 0):
        raise ValueError('Cross-body assembly requires strictly separated surfaces')
    near_stats = {'maximum_order': 0, 'pair_evaluations': 0, 'maximum_accepted_relative_difference': 0.}
    counts = [0, 0, 0, 0, 0]
    from ..constants import EPSILON_0, MU_0
    omega = 2*math.pi*exterior.frequency
    for category in (3, 4):
        selected = (classes == category).nonzero().flatten()
        counts[category] = len(selected)
        for ids in selected.split(config.pair_batch if category == 3 else config.far_pair_batch):
            rows, cols = ids//fj, ids % fj
            ri = test_basis.face_edges[rows, :, None].expand(-1, -1, 3).reshape(-1)
            ci = source_basis.face_edges[cols, None, :].expand(-1, 3, -1).reshape(-1)
            if category == 3:
                d, k, stats = _adaptive_near_integrals(test_basis, rows, cols, exterior, config,
                                                       kernel=kernel, source_basis=source_basis)
                _merge_near_stats(near_stats, stats)
            else:
                d, k = _pair_integrals(test_basis, rows, cols, exterior, config.order, config.order,
                                       False, category, kernel=kernel, source_basis=source_basis)
            for i, j, values in ((0, 0, k), (1, 1, k), (0, 1, 1j*omega*EPSILON_0*exterior.epsilon_r*d),
                                 (1, 0, -1j*omega*MU_0*exterior.mu_r*d)):
                matrix.index_put_((ri+i*ni, ci+j*nj), values.reshape(-1).to(dtype), accumulate=True)
    return matrix, {'contribution': 'exterior_only', 'triangle_pair_counts': counts,
                    'minimum_gap_m': float(minimum), 'near_quadrature': near_stats}


def assemble_system(system, exterior, config=IntegrationConfig(), *, kernel=FREE_SPACE):
    """Global [J_0,...,J_B,M_0,...,M_B] matrix with diagonal interior terms."""
    system.assert_current()
    dtype = torch.complex128 if system.dtype == torch.float64 else torch.complex64
    if len(system.bodies) == 1:
        body = system.bodies[0]
        matrix, metadata = assemble_weak(body.basis, exterior, body.interior, config, kernel=kernel)
        record = {'test_region': 1, 'source_region': 1, **metadata}
        metadata.update(body_count=1, blocks=[record], unknown_order=['all_body_J_then_all_body_M'])
        return matrix, metadata
    matrix = torch.zeros((2*system.total_edges, 2*system.total_edges), dtype=dtype, device=system.device)
    records = []
    started = time.perf_counter()
    for i, test in enumerate(system.bodies):
        for j, source in enumerate(system.bodies):
            if i == j:
                block, metadata = assemble_weak(test.basis, exterior, test.interior, config, kernel=kernel)
            else:
                block, metadata = assemble_cross(test.basis, source.basis, exterior, config, kernel=kernel)
            n, m = test.basis.n_edges, source.basis.n_edges
            for u, test_slice in enumerate((system.J_slices[i], system.M_slices[i])):
                for v, source_slice in enumerate((system.J_slices[j], system.M_slices[j])):
                    matrix[test_slice, source_slice] = block[u*n:(u+1)*n, v*m:(v+1)*m]
            records.append({'test_region': test.region_id, 'source_region': source.region_id, **metadata})
    torch.cuda.synchronize(system.device)
    return matrix, {'formulation_id': 'pmchwt_exp_minus_iwt_total_trace_v1', 'operator_backend': 'dense',
                    'kernel_id': kernel.kernel_id, 'kernel_version': kernel.version,
                    'unknown_order': ['all_body_J_then_all_body_M'], 'body_count': len(system.bodies),
                    'blocks': records, 'quadrature': vars(config), 'assembly_seconds': time.perf_counter()-started,
                    'matrix_bytes': matrix.numel()*matrix.element_size(), 'device': str(system.device), 'dtype': str(dtype)}
