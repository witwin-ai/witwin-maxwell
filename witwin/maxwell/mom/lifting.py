"""Analytic incident-trace completion for scalar PMCHWT.

The unknown is the RWG scattered Cauchy trace. The exact incident trace is a
separate, source-bound carrier, never a fitted numerical background solution.
All numerical integration and current evaluation stays on the CUDA device.
"""
from __future__ import annotations

from functools import lru_cache
import math
import time
import torch

from ..constants import EPSILON_0, MU_0
from .carrier import IncidentCarrier
from .kernels import cartesian_distance
from .operator import NEAR_FALLBACK_BATCH as NEAR_BATCH, _pair_budget, _real_complex_matmul, _touching_batch, local_geometry
from .quadrature import (ANISOTROPIC_TOUCHING_ASPECT, COPLANAR_TOLERANCE, anisotropic_touching_pairs, classify_all_pairs,
                         distance_scaled_duffy_fan, refine_toward_source_edges, static_potential_integrals,
                         touching_pair_rule, touching_remainder_nodes, touching_remainder_rule, triangle_rule)

FORMULATION_ID = 'pmchwt_exp_minus_iwt_analytic_incident_lift_v1'
CURRENT_KIND = 'analytic_incident_plus_rwg_scattered_v1'
KERNEL_DIFFERENCE_VERSION = 'helmholtz_expm1_gradient_series_branch_safe_v2'


@lru_cache(maxsize=16)
def _remainder_coefficients(device, dtype):
    """Device-resident Taylor coefficients (d-1)/d! for d=13..2."""
    return torch.tensor([(degree-1)/math.factorial(degree) for degree in range(13, 1, -1)],
                        device=device, dtype=dtype).unbind()


def _gradient_remainder(z, increment=None):
    """Stable z*exp(z)-expm1(z), whose Taylor series starts at degree two.

    ``increment`` may supply an already evaluated expm1(z).
    """
    coefficients = _remainder_coefficients(z.device, z.dtype)
    polynomial = torch.full_like(z, 13/math.factorial(14))
    for coefficient in coefficients:
        polynomial = torch.addcmul(coefficient, z, polynomial)
    increment = torch.expm1(z) if increment is None else increment
    return torch.where(abs(z) < .25, z*z*polynomial, z+(z-1)*increment)


def difference_kernels(delta, exterior, interior, *, delta_k=None):
    """Return background G/grad G and stable interior-minus-background kernels.

    Paired quadrature nodes must not coincide. For small z=i*(ki-ke)*r,
    z*exp(z)-expm1(z) is evaluated as a second-order series; merely using
    expm1 for the Green function leaves cancellation in its gradient.
    """
    radius = torch.linalg.vector_norm(delta, dim=-1)
    phase = torch.exp(1j*exterior.k*radius)
    z = 1j*(interior.k-exterior.k if delta_k is None else delta_k)*radius
    increment = torch.expm1(z)
    remainder = _gradient_remainder(z, increment)
    green = phase/(4*math.pi*radius)
    prefactor = phase[..., None]*delta/(4*math.pi*radius[..., None]**3)
    gradient = prefactor*(1j*exterior.k*radius-1)[..., None]
    gradient_difference = prefactor*(1j*exterior.k*radius*increment+remainder)[..., None]
    return green, gradient, green*increment, gradient_difference


def medium_differences(exterior, interior):
    """Construct small material contrasts before multiplying physical units."""
    de = EPSILON_0*(interior.epsilon_r-exterior.epsilon_r)
    dm = MU_0*(interior.mu_r-exterior.mu_r)
    # Rationalization preserves weak contrast but is singular for opposite
    # passive roots (e.g. matched double-negative and positive media).
    denominator = interior.k+exterior.k
    if abs(denominator) >= .5*max(abs(interior.k), abs(exterior.k)):
        dk = exterior.omega**2*(interior.mu*de+exterior.epsilon*dm)/denominator
    else:
        dk = interior.k-exterior.k
    return de, dm, dk


def weighted_green_difference(green, delta_green, coefficient_i, delta_coefficient,
                              dual_i, dual_e, delta_dual, omega):
    """Stable material-weighted vector and divergence Green differences.

    epsilon/k²=1/(omega²*mu) and mu/k²=1/(omega²*epsilon).
    In particular, nonmagnetic epsilon/k² differences vanish exactly without
    independent square-root/square/division roundoff creating a false pole.
    """
    vector = delta_coefficient*green+coefficient_i*delta_green
    inverse_i = 1/(omega**2*dual_i)
    inverse_difference = -delta_dual/(omega**2*dual_i*dual_e)
    scalar = inverse_difference*green+inverse_i*delta_green
    return vector, scalar


def incident_trace(handle, points, faces):
    """Evaluate exact analytic J/M and surface divergences on known faces.

    Internal integration interface: points have shape [B,Q,3], faces [B].
    The source plane wave is an exterior-medium Maxwell solution on every face.
    """
    electric, magnetic = handle.incident_carrier.evaluate(points, handle.exterior)
    return _traces_from_fields(handle, electric, magnetic, faces)


def _traces_from_fields(handle, electric, magnetic, faces):
    normal = handle.basis.mesh.normals[faces, None].to(electric.dtype)
    j = torch.linalg.cross(normal.expand_as(magnetic), magnetic)
    m = -torch.linalg.cross(normal.expand_as(electric), electric)
    exterior = handle.exterior
    from .tensor_medium import TensorMedium
    if isinstance(exterior, TensorMedium):
        div_j = 1j*exterior.omega*(normal*(electric@exterior.epsilon.T)).sum(-1)
        div_m = 1j*exterior.omega*(normal*(magnetic@exterior.mu.T)).sum(-1)
    else:
        div_j = 1j*exterior.omega*exterior.epsilon*(normal*electric).sum(-1)
        div_m = 1j*exterior.omega*exterior.mu*(normal*magnetic).sum(-1)
    return j, m, div_j, div_m


# Face-node carrier traces kept for one product-rule block, all sources.
_TRACE_CACHE_BYTES = 1 << 28
# Pair values retained per far block before scattering into the RHS rows.
_FAR_VALUE_BYTES = 1 << 27


def _validated_batch(handles):
    handles = tuple(handles)
    if not handles:
        raise ValueError('Batched lifted RHS requires at least one carrier handle')
    first = handles[0]
    if any(h.basis is not first.basis or h.exterior is not first.exterior or h.interior is not first.interior
           or h.integration != first.integration for h in handles[1:]):
        raise ValueError('Batched lifted RHS requires one shared basis, exterior, interior and integration')
    return handles


def _incident_traces(handles, points, faces):
    """Every carrier's exact traces [S,...] on shared nodes in one evaluation."""
    carriers = tuple(h.incident_carrier for h in handles)
    if any(type(carrier) is not IncidentCarrier for carrier in carriers):
        return tuple(torch.stack(parts) for parts in zip(*(incident_trace(h, points, faces) for h in handles)))
    electric, magnetic = IncidentCarrier.evaluate_many(carriers, points, handles[0].exterior)
    return _traces_from_fields(handles[0], electric, magnetic, faces)


def _difference_terms(delta, exterior, dk):
    """G_e, G_i-G_e and s with grad(G_i-G_e)=s*delta, as in difference_kernels."""
    radius = cartesian_distance(delta)
    phase = torch.exp(1j*exterior.k*radius)
    z = 1j*dk*radius
    increment = torch.expm1(z)
    green = phase/(4*math.pi*radius)
    scale = phase*(1j*exterior.k*radius*increment+_gradient_remainder(z, increment))/(4*math.pi*radius**3)
    return green, green*increment, scale


def _material_kernels(green, delta_green, exterior, interior):
    """(vector, scalar) kernels testing M (first row) and J (second row)."""
    de, dm, _ = medium_differences(exterior, interior)
    omega = exterior.omega
    magnetic = weighted_green_difference(green, delta_green, interior.epsilon, de,
                                         interior.mu, exterior.mu, dm, omega)
    electric = weighted_green_difference(green, delta_green, interior.mu, dm,
                                         interior.epsilon, exterior.epsilon, de, omega)
    return magnetic, electric


def _kernel_gradient_leading(exterior, dk):
    """Coefficient c of grad(G_i - G_e) = c (x - y)/R + O(R): c = -(k_i^2 - k_e^2)/(8 pi)."""
    return -dk*(2*exterior.k+dk)/(8*math.pi)


def _trace_expansion(handle, points, faces, sources):
    """Incident traces and their in-plane first derivatives at points [B,Q,3] on source planes.

    Derivatives use central differences of the analytic carrier along two
    unit tangents of each source plane, with a step of 1e-4 source diameters
    (truncation and roundoff both far below the quadrature tolerance).
    Returns (values, first, second, tangents): each of the first three is the
    (J, M, div J, div M) tuple; tangents is a pair of [B,3] unit vectors.
    """
    first = sources[:, 1]-sources[:, 0]
    normal = torch.linalg.cross(first, sources[:, 2]-sources[:, 0])
    normal = normal/torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
    t1 = first/torch.linalg.vector_norm(first, dim=-1, keepdim=True)
    t2 = torch.linalg.cross(normal, t1)
    step = 1e-4*torch.linalg.vector_norm(sources-sources.roll(1, 1), dim=-1).amax(-1)
    values = incident_trace(handle, points, faces)
    derivatives = []
    for tangent in (t1, t2):
        offset = (step[:, None]*tangent)[:, None]
        plus, minus = incident_trace(handle, points+offset, faces), incident_trace(handle, points-offset, faces)
        derivatives.append(tuple((a-b)/(2*step.to(a.dtype).reshape(-1, *([1]*(a.ndim-1)))) for a, b in zip(plus, minus)))
    return values, derivatives[0], derivatives[1], (t1, t2)


def _expanded(values, first, second, along1, along2, index):
    """Linear trace model value + first*along1 + second*along2 for component ``index``."""
    value = values[index]
    if value.ndim == along1.ndim+1:
        return value+first[index]*along1[..., None]+second[index]*along2[..., None]
    return value+first[index]*along1+second[index]*along2


def _paired_rhs(handle, rows, cols, category, obs, src, weight, projected=None):
    """Local [B,2,3] RHS rows from paired observation/source nodes [B,N,3].

    With ``projected`` (each observation node projected onto its source
    plane), the static part 1/(4 pi R) times the linear incident-trace model
    about that projection is removed; the remaining integrand vanishes like
    R at coincidence, and the caller adds the removed part in closed form.
    """
    basis, exterior, interior = handle.basis, handle.exterior, handle.interior
    _, free, slopes = local_geometry(basis)
    j, m, div_j, div_m = incident_trace(handle, src, cols)
    f = (slopes[rows, None, :, None]*(obs[:, :, None]-free[rows, None])).to(j.dtype)
    div_f = 2*slopes[rows]
    de, dm, dk = medium_differences(exterior, interior)
    g, _, dg, dgrad = difference_kernels(obs-src, exterior, interior, delta_k=dk)
    if projected is not None:
        radius = torch.linalg.vector_norm(obs-src, dim=-1)
        static = (1/(4*math.pi*radius)).to(j.dtype)
        # exp(ik R)/(4 pi R) - 1/(4 pi R), without cancellation.
        g = torch.expm1(1j*exterior.k*radius.to(j.dtype))*static
        values, first_derivative, second_derivative, (t1, t2) = _trace_expansion(handle, projected, cols, free[cols])
        offset = src-projected
        along1, along2 = (offset*t1[:, None]).sum(-1), (offset*t2[:, None]).sum(-1)
        traces = tuple(_expanded(values, first_derivative, second_derivative, along1, along2, i) for i in range(4))

    def electric(current, divergence, coefficient_i, delta_coefficient, dual_i, dual_e, delta_dual, at_projection=None):
        vector_kernel, scalar_kernel = weighted_green_difference(g, dg, coefficient_i, delta_coefficient,
            dual_i, dual_e, delta_dual, exterior.omega)
        weighted = vector_kernel*weight
        vector = torch.einsum('bqlc,bqc,bq->bl', f, current, weighted)
        scalar = (divergence*scalar_kernel*weight).sum(-1)
        if at_projection is not None:
            # delta_coefficient (current(y) - linear model about x_p)/(4 pi R), and likewise for divergences.
            inverse_difference = -delta_dual/(exterior.omega**2*dual_i*dual_e)
            current_p, divergence_p = at_projection
            vector = vector+delta_coefficient*torch.einsum('bqlc,bqc,bq->bl', f, current-current_p, static*weight)
            scalar = scalar+inverse_difference*((divergence-divergence_p)*static*weight).sum(-1)
        return vector-div_f*scalar[:, None]

    if category == 0:
        kj = torch.zeros_like(div_f, dtype=j.dtype)
        km = torch.zeros_like(kj)
    else:
        if projected is not None:
            # grad(G_i - G_e) = -(k_i^2 - k_e^2)(x - y)/(8 pi R) + O(R): remove that direction-
            # dependent leading term with the trace frozen at x_p; the caller adds it in closed form.
            leading = (_kernel_gradient_leading(exterior, dk)*(obs-src)/radius[..., None]).to(j.dtype)
            dgrad_j = torch.linalg.cross(dgrad, j)-torch.linalg.cross(leading, values[0])
            dgrad_m = torch.linalg.cross(dgrad, m)-torch.linalg.cross(leading, values[1])
        else:
            dgrad_j, dgrad_m = torch.linalg.cross(dgrad, j), torch.linalg.cross(dgrad, m)
        kj = torch.einsum('bqlc,bqc,bq->bl', f, dgrad_j, weight.to(j.dtype))
        km = torch.einsum('bqlc,bqc,bq->bl', f, dgrad_m, weight.to(j.dtype))
    m_projected = j_projected = None
    if projected is not None:
        j_projected, m_projected = (traces[0], traces[2]), (traces[1], traces[3])
    first = kj+1j*exterior.omega*electric(m, div_m, interior.epsilon, de, interior.mu, exterior.mu, dm, m_projected)
    second = km-1j*exterior.omega*electric(j, div_j, interior.mu, dm, interior.epsilon, exterior.epsilon, de, j_projected)
    return -torch.stack((first, second), dim=1)


def _subtracted_touching_rhs(handle, rows, cols, outer_order, inner_order, ratio, category):
    """Touching RHS rows: closed-form static source integrals with the trace frozen at the
    projected observer, plus a bounded product-rule remainder."""
    from .operator import TOUCHING_MAX_DEPTH, NEAR_FALLBACK_NODES, TOUCHING_REMAINDER_NODES
    basis, exterior, interior = handle.basis, handle.exterior, handle.interior
    _, free, slopes = local_geometry(basis)
    dtype = torch.complex128 if free.dtype == torch.float64 else torch.complex64
    de, dm, dk = medium_differences(exterior, interior)
    omega = exterior.omega
    value = torch.zeros((len(rows), 2, 3), device=free.device, dtype=dtype)

    def project(points, source):
        normal = torch.linalg.cross(source[:, 1]-source[:, 0], source[:, 2]-source[:, 0])
        normal = normal/torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
        return points-((points-source[:, None, 0])*normal[:, None]).sum(-1, keepdim=True)*normal[:, None]
    leaves, owner, depth = refine_toward_source_edges(free[rows], free[cols], ratio=ratio, max_level=TOUCHING_MAX_DEPTH)
    step = max(1, NEAR_FALLBACK_NODES//(64*outer_order**2))
    for start in range(0, len(leaves), step):
        leaf, pair = leaves[start:start+step], owner[start:start+step]
        obs, wo = triangle_rule(leaf, outer_order)
        q = obs.shape[1]
        r, c = rows[pair], cols[pair]
        i0, i1, _ = static_potential_integrals(obs.reshape(-1, 3), free[c, None].expand(-1, q, -1, -1).reshape(-1, 3, 3))
        i0, i1 = i0.reshape(len(leaf), q), i1.reshape(len(leaf), q, 3)
        weight = (wo/(4*math.pi)).to(dtype)
        values, first_derivative, second_derivative, (t1, t2) = _trace_expansion(handle, project(obs, free[c]), c, free[c])
        # int (linear model)/R dy = value I0 + d1 (I1.t1) + d2 (I1.t2).
        moment1, moment2 = (i1*t1[:, None]).sum(-1), (i1*t2[:, None]).sum(-1)
        j, m, div_j, div_m = (values[i]*(i0[..., None] if values[i].ndim == 3 else i0)
                              + _expanded((torch.zeros_like(values[i]),)*4, first_derivative, second_derivative,
                                          moment1, moment2, i) for i in range(4))
        f = (slopes[r, None, :, None]*(obs[:, :, None]-free[r, None])).to(dtype)
        div_f = 2*slopes[r]

        def electric(current, divergence, delta_coefficient, dual_i, dual_e, delta_dual):
            inverse_difference = -delta_dual/(omega**2*dual_i*dual_e)
            vector = delta_coefficient*torch.einsum('bqlc,bqc,bq->bl', f, current, weight)
            return vector-div_f*(inverse_difference*(divergence*weight).sum(-1))[:, None]
        first = 1j*omega*electric(m, div_m, de, interior.mu, exterior.mu, dm)
        second = -1j*omega*electric(j, div_j, dm, interior.epsilon, exterior.epsilon, de)
        if category != 0:
            # int c (x - y)/R dy = c ((x - x_p) I0 - I1), crossed with the trace at x_p.
            projected_obs = project(obs, free[c])
            moment = (_kernel_gradient_leading(exterior, dk)*((obs-projected_obs)*i0[..., None]-i1)).to(dtype)
            weight_k = (wo).to(dtype)
            first = first+torch.einsum('bqlc,bqc,bq->bl', f, torch.linalg.cross(moment, values[0]), weight_k)
            second = second+torch.einsum('bqlc,bqc,bq->bl', f, torch.linalg.cross(moment, values[1]), weight_k)
        value.index_add_(0, pair, -torch.stack((first, second), dim=1))
    # The remainder (cone at R = 0) keeps the joint touching rule.
    mesh = basis.mesh
    step = max(1, TOUCHING_REMAINDER_NODES//touching_remainder_nodes(category, inner_order))
    for first in range(0, len(rows), step):
        r, c = rows[first:first+step], cols[first:first+step]
        obs, src, weight = touching_remainder_rule(free[r], free[c], mesh.faces[r], mesh.faces[c], category, inner_order)
        value[first:first+step] += _paired_rhs(handle, r, c, category, obs, src, weight, projected=project(obs, free[c]))
    return value, {'leaves': len(leaves), 'depth': depth}


def _fan_near_rhs(handle, rows, cols, outer_order, inner_order, ratio, category=3):
    """Gap-aware near RHS: observer leaves toward source edges, distance-scaled source fans."""
    from .operator import NEAR_FALLBACK_MAX_DEPTH, NEAR_FALLBACK_NODES
    triangles = handle.basis.mesh.triangles
    leaves, owner, depth = refine_toward_source_edges(triangles[rows], triangles[cols], ratio=ratio,
                                                      max_level=NEAR_FALLBACK_MAX_DEPTH)
    dtype = torch.complex128 if triangles.dtype == torch.float64 else torch.complex64
    value = torch.zeros((len(rows), 2, 3), device=triangles.device, dtype=dtype)
    inner = 3*inner_order**2
    step = max(1, NEAR_FALLBACK_NODES//(outer_order**2*inner))
    for start in range(0, len(leaves), step):
        leaf, pair = leaves[start:start+step], owner[start:start+step]
        obs, wo = triangle_rule(leaf, outer_order)
        q = obs.shape[1]
        src, ws = distance_scaled_duffy_fan(obs.reshape(-1, 3), triangles[cols[pair], None].expand(-1, q, -1, -1).reshape(-1, 3, 3),
                                            inner_order, edge_grading=True, coplanar_tolerance=COPLANAR_TOLERANCE)
        paired_obs = obs[:, :, None].expand(-1, -1, inner, -1).flatten(1, 2)
        local = _paired_rhs(handle, rows[pair], cols[pair], category, paired_obs, src.reshape(len(leaf), q*inner, 3),
                            (wo[:, :, None]*ws.reshape(len(leaf), q, inner)).flatten(1, 2))
        value.index_add_(0, pair, local)
    return value, {'leaves': len(leaves), 'depth': depth}


def _fallback_near_rhs(handle, rows, cols, config, scale, *, category=3, rule=None, integrator=None, levels=None):
    """Gap-aware RHS pairs (near fallback or anisotropic touching); same levels as the operator."""
    from .operator import NEAR_FALLBACK_LEVELS, NEAR_FALLBACK_RULE
    rule = NEAR_FALLBACK_RULE if rule is None else rule
    integrator = _fan_near_rhs if integrator is None else integrator
    levels = NEAR_FALLBACK_LEVELS if levels is None else levels
    pending = torch.arange(len(rows), device=rows.device)
    value = None
    stats = {'rule': rule, 'pairs': len(rows), 'pair_evaluations': 0, 'maximum_level': 0,
             'maximum_depth': 0, 'maximum_leaves_per_call': 0, 'maximum_accepted_relative_difference': 0.,
             'levels': [list(level) for level in levels]}
    for level, (outer, inner, ratio) in enumerate(levels):
        updated, info = integrator(handle, rows[pending], cols[pending], outer, inner, ratio, category)
        stats['pair_evaluations'] += len(pending)
        stats['maximum_depth'] = max(stats['maximum_depth'], info['depth'])
        stats['maximum_leaves_per_call'] = max(stats['maximum_leaves_per_call'], info['leaves'])
        if value is None:
            value = updated
            continue
        norm = torch.linalg.vector_norm(updated*scale, dim=(-2, -1))
        relative = torch.linalg.vector_norm((updated-value[pending])*scale, dim=(-2, -1))/norm.clamp_min(torch.finfo(norm.dtype).tiny)
        value[pending] = updated
        accepted = relative <= config.near_tolerance
        if bool(accepted.any()):
            stats['maximum_accepted_relative_difference'] = max(stats['maximum_accepted_relative_difference'],
                                                                float(relative[accepted].max()))
            stats['maximum_level'] = level
        pending = pending[~accepted]
        if not len(pending):
            return value, stats
    raise RuntimeError(f'Incident-lifting quadrature failed tolerance {config.near_tolerance:g} in the {rule} rule; '
                       f'maximum relative level difference {float(relative.max()):.3g}.')


def _touching_values(handles, rows, cols, category, order):
    """Self/edge/vertex joint-Duffy RHS entries [S,B,2,3] for all carriers."""
    handle = handles[0]
    basis, exterior, interior = handle.basis, handle.exterior, handle.interior
    mesh = basis.mesh
    _, free, slopes = local_geometry(basis)
    triangles = mesh.triangles
    obs, src, weight = touching_pair_rule(triangles[rows], triangles[cols],
        mesh.faces[rows], mesh.faces[cols], category, order)
    j, m, div_j, div_m = _incident_traces(handles, src, cols)
    f = (slopes[rows, None, :, None]*(obs[:, :, None]-free[rows, None])).to(j.dtype)
    div_f = 2*slopes[rows]
    _, _, dk = medium_differences(exterior, interior)
    delta = obs-src
    green, delta_green, scale = _difference_terms(delta, exterior, dk)
    (vector_m, scalar_m), (vector_j, scalar_j) = _material_kernels(green, delta_green, exterior, interior)

    def electric(current, divergence, vector, scalar):
        return (torch.einsum('bqlc,sbqc,bq->sbl', f, current, vector*weight)
                - div_f*torch.einsum('sbq,bq->sb', divergence, scalar*weight)[..., None])

    if category == 0:
        kj = km = torch.zeros((len(handles), len(rows), 3), dtype=j.dtype, device=j.device)
    else:
        gradient = (delta.to(j.dtype)*(scale*weight)[..., None])[None]
        kj = torch.einsum('bqlc,sbqc->sbl', f, torch.linalg.cross(gradient.expand_as(j), j))
        km = torch.einsum('bqlc,sbqc->sbl', f, torch.linalg.cross(gradient.expand_as(m), m))
    first = kj+1j*exterior.omega*electric(m, div_m, vector_m, scalar_m)
    second = km-1j*exterior.omega*electric(j, div_j, vector_j, scalar_j)
    return -torch.stack((first, second), dim=2)


def _anisotropic_touching_values(handles, rows, cols, category, config, stats):
    """High-aspect touching RHS entries [S,B,2,3] by static subtraction, one carrier at a time."""
    from .operator import TOUCHING_ANISOTROPIC_RULE, TOUCHING_LEVELS, _merge_rule_stats
    scale = torch.tensor([abs(handles[0].exterior.eta), 1.], device=rows.device,
                         dtype=handles[0].basis.mesh.dtype)[None, :, None]
    parts = []
    for handle, record in zip(handles, stats):
        local, observed = _fallback_near_rhs(handle, rows, cols, config, scale, category=category,
            rule=TOUCHING_ANISOTROPIC_RULE, integrator=_subtracted_touching_rhs, levels=TOUCHING_LEVELS)
        touching = record.setdefault('touching_anisotropic', {'aspect_threshold': ANISOTROPIC_TOUCHING_ASPECT})
        _merge_rule_stats(touching, ('self', 'edge', 'vertex')[category], observed)
        parts.append(local)
    return torch.stack(parts)


def _touching_split(mesh, rows, cols):
    """Mask of touching pairs routed to the anisotropy-aware rule."""
    return anisotropic_touching_pairs(mesh.triangles[rows], mesh.triangles[cols])


def _product_values(handles, rows, cols, order, budget):
    """Disjoint product-rule RHS entries [S,P,2,3] for all carriers.

    Source traces are evaluated once per source face node rather than per
    pair node. Kernels are evaluated once per pair node and contracted with
    every carrier by batched matrix products over the source nodes.
    """
    handle = handles[0]
    basis, exterior, interior = handle.basis, handle.exterior, handle.interior
    mesh = basis.mesh
    _, free, slopes = local_geometry(basis)
    triangles = mesh.triangles
    count = len(handles)
    dtype = torch.complex128 if mesh.dtype == torch.float64 else torch.complex64
    result = torch.empty((count, len(rows), 2, 3), dtype=dtype, device=mesh.device)
    if not len(rows):
        return result
    _, _, dk = medium_differences(exterior, interior)
    nodes = order*order
    faces, inverse = torch.unique(cols, return_inverse=True)
    chunk = max(1, _TRACE_CACHE_BYTES//(count*nodes*8*torch.empty((), dtype=dtype).element_size()))
    if len(faces) <= chunk:
        groups = ((0, torch.arange(len(rows), device=rows.device)),)
    else:
        group = inverse//chunk
        ordered = torch.argsort(group, stable=True)
        sizes = torch.bincount(group, minlength=(len(faces)+chunk-1)//chunk).tolist()
        groups = tuple(zip(range(len(sizes)), ordered.split(sizes)))
    # Never below the configured near batch, whose workspace the original
    # per-source refinement already used at every order.
    pairs_per_batch = max(handle.integration.pair_batch, budget//(nodes*nodes))
    omega = exterior.omega
    for index, members in groups:
        start = index*chunk
        face_ids = faces[start:start+chunk]
        source_nodes, source_weights = triangle_rule(triangles[face_ids], order)
        j, m, div_j, div_m = _incident_traces(handles, source_nodes, face_ids)
        # Source traces as [F,(component,node),S] and divergences [F,node,2S].
        right_j = j.permute(1, 3, 2, 0).flatten(1, 2)
        right_m = m.permute(1, 3, 2, 0).flatten(1, 2)
        right_div = torch.cat((div_m, div_j)).permute(1, 2, 0)
        for batch in members.split(pairs_per_batch):
            r, local = rows[batch], inverse[batch]-start
            size = len(batch)
            obs, obs_weights = triangle_rule(triangles[r], order)
            delta = obs[:, :, None]-source_nodes[local][:, None]
            weight = obs_weights[:, :, None]*source_weights[local][:, None]
            green, delta_green, scale = _difference_terms(delta, exterior, dk)
            (vector_m, scalar_m), (vector_j, scalar_j) = _material_kernels(green, delta_green, exterior, interior)
            gradient = delta.to(dtype)*(scale*weight)[..., None]
            # Test functions first: f^T [vector_m, vector_j, grad_x, grad_y, grad_z]
            # over observation nodes, independent of the carrier count.
            kernels = torch.cat((vector_m*weight, vector_j*weight, *gradient.unbind(-1)), -1)
            f = slopes[r, None, :, None]*(obs[:, :, None]-free[r, None])
            tested = _real_complex_matmul(f.flatten(2).transpose(1, 2), kernels).reshape(size, 3, 3, 5, nodes)
            moments = tested[:, :, :, 2:]
            # f.(D x c) = sum eps_abc f_a D_b c_c with moments[l,a,b,node].
            curl = torch.stack((moments[:, :, 1, 2]-moments[:, :, 2, 1], moments[:, :, 2, 0]-moments[:, :, 0, 2],
                                moments[:, :, 0, 1]-moments[:, :, 1, 0]), 2).flatten(2)
            values_m = torch.cat((tested[:, :, :, 0].flatten(2), curl), 1)@right_m[local]
            values_j = torch.cat((tested[:, :, :, 1].flatten(2), curl), 1)@right_j[local]
            sums = torch.stack(((scalar_m*weight).sum(1), (scalar_j*weight).sum(1)), 1)@right_div[local]
            div_f = 2*slopes[r].T[None]
            first = values_j[:, 3:].transpose(0, 2)+1j*omega*(values_m[:, :3].transpose(0, 2)-div_f*sums[:, 0, None, :count].transpose(0, 2))
            second = values_m[:, 3:].transpose(0, 2)-1j*omega*(values_j[:, :3].transpose(0, 2)-div_f*sums[:, 1, None, count:].transpose(0, 2))
            # [S,3,B] -> [S,B,2,3]
            first, second = first.transpose(1, 2), second.transpose(1, 2)
            result[:, batch] = -torch.stack((first, second), dim=2)
    return result


def _near_values(handles, rows, cols, config, budget):
    """Refine near-disjoint entries per carrier and pair in balanced row units.

    Every carrier keeps the successive-order acceptance of its own entries;
    pairs pending for any carrier are refined together for all carriers, and
    accepted entries are frozen. All near pairs advance one order at a time.
    """
    count = len(handles)
    stats = [{'maximum_order': 0, 'pair_evaluations': 0, 'maximum_accepted_relative_difference': 0.}
             for _ in handles]
    order = config.singular_order
    value = _product_values(handles, rows, cols, order, budget)
    if not len(rows):
        return value, stats
    device = rows.device
    pending = torch.ones((count, len(rows)), dtype=torch.bool, device=device)
    evaluations = torch.full((count,), len(rows), dtype=torch.int64, device=device)
    maximum_error = value.real.new_zeros(count)
    final_order = [order]*count
    scale = value.real.new_tensor([abs(handles[0].exterior.eta), 1.])[None, None, :, None]
    while True:
        active = pending.any(0).nonzero().flatten()
        if not len(active):
            break
        order = min(order+2, config.near_max_order)
        updated = _product_values(handles, rows[active], cols[active], order, budget)
        previous = value[:, active]
        mask = pending[:, active]
        norm = torch.linalg.vector_norm(updated*scale, dim=(-2, -1))
        difference = torch.linalg.vector_norm((updated-previous)*scale, dim=(-2, -1))
        relative = difference/norm.clamp_min(torch.finfo(norm.dtype).tiny)
        value[:, active] = torch.where(mask[..., None, None], updated, previous)
        accepted = mask & (relative <= config.near_tolerance)
        evaluations += mask.sum(1)
        maximum_error = torch.maximum(maximum_error, torch.where(accepted, relative, 0).amax(1))
        for source, refined in enumerate(mask.any(1).tolist()):
            if refined:
                final_order[source] = order
        remaining = mask & ~accepted
        pending[:, active] = remaining
        if order == config.near_max_order and bool(remaining.any()):
            break
    for source, record in enumerate(stats):
        record.update(maximum_order=final_order[source], pair_evaluations=int(evaluations[source]),
                      maximum_accepted_relative_difference=float(maximum_error[source]))
    # Pairs still changing at near_max_order use the gap-aware fallback per carrier.
    from .operator import NEAR_FALLBACK_BATCH, _merge_rule_stats
    single = scale[0]
    for source, handle in enumerate(handles):
        unresolved = pending[source].nonzero().flatten()
        for ids in unresolved.split(NEAR_FALLBACK_BATCH) if len(unresolved) else ():
            value[source, ids], record = _fallback_near_rhs(handle, rows[ids], cols[ids], config, single)
            _merge_rule_stats(stats[source], 'fallback', record)
    return value, stats


def _merge_near(total, observed):
    total['maximum_order'] = max(total['maximum_order'], observed['maximum_order'])
    total['pair_evaluations'] += observed['pair_evaluations']
    total['maximum_accepted_relative_difference'] = max(total['maximum_accepted_relative_difference'],
                                                        observed['maximum_accepted_relative_difference'])


def build_lifted_rhs_batch(handles):
    """Integrate -(A_int-A_background,int)c_inc for several carriers at once.

    All handles share one basis, media and integration and differ only in
    their incident carriers. Classification, quadrature nodes and difference
    kernels are evaluated once; each carrier keeps its own near-quadrature
    acceptance. Returns stacked [S,2*edges] RHS rows and one record each.
    """
    handles = _validated_batch(handles)
    basis, config = handles[0].basis, handles[0].integration
    mesh = basis.mesh
    if mesh.device.type != 'cuda':
        raise ValueError('Incident-trace lifting requires CUDA.')
    started = time.perf_counter()
    nf, ne, count = len(mesh.faces), basis.n_edges, len(handles)
    classes = classify_all_pairs(mesh)
    dtype = torch.complex128 if mesh.dtype == torch.float64 else torch.complex64
    rhs = torch.zeros((count, 2, ne), dtype=dtype, device=mesh.device)
    stats = [{'maximum_order': 0, 'pair_evaluations': 0, 'maximum_accepted_relative_difference': 0.}
             for _ in handles]
    budget = _pair_budget(config)

    def scatter(rows, local):
        rhs.index_add_(2, basis.face_edges[rows].flatten(), local.permute(0, 2, 1, 3).flatten(2))

    counts = []
    for category in range(5):
        selected = torch.nonzero(classes == category).flatten()
        counts.append(len(selected))
        if category < 3:
            mask = _touching_split(mesh, selected//nf, selected % nf)
            for pairs in selected[mask].split(NEAR_BATCH) if bool(mask.any()) else ():
                rows, cols = pairs//nf, pairs % nf
                scatter(rows, _anisotropic_touching_values(handles, rows, cols, category, config, stats))
            for pairs in selected[~mask].split(_touching_batch(config, category, len(handles))):
                rows, cols = pairs//nf, pairs % nf
                scatter(rows, _touching_values(handles, rows, cols, category, config.singular_order))
        elif category == 3:
            rows, cols = selected//nf, selected % nf
            local, observed = _near_values(handles, rows, cols, config, budget)
            for total, record in zip(stats, observed):
                _merge_near(total, record)
            scatter(rows, local)
        else:
            block = max(1, _FAR_VALUE_BYTES//(count*6*rhs.element_size()))
            for pairs in selected.split(block):
                rows, cols = pairs//nf, pairs % nf
                scatter(rows, _product_values(handles, rows, cols, config.order, budget))
    torch.cuda.synchronize(mesh.device)
    seconds = time.perf_counter()-started
    records = [{'formulation_id': FORMULATION_ID, 'current_kind': CURRENT_KIND,
        'incident_carrier_identity': handle.incident_carrier.identity,
        'difference_kernel_version': KERNEL_DIFFERENCE_VERSION,
        'rhs_integration': vars(config), 'rhs_near_quadrature': record,
        'rhs_near_estimator': 'successive_order_difference_per_panel_pair',
        'rhs_near_norm': 'Euclidean norm of six local row entries after exterior-impedance balancing',
        'global_rhs_error_certified': False,
        'rhs_triangle_pair_counts': counts, 'rhs_seconds': seconds, 'rhs_source_batch': count,
        'rhs_touching_and_far_accuracy': 'independent_order_convergence_required'}
        for handle, record in zip(handles, stats)]
    return rhs.flatten(1), records


def build_lifted_rhs(handle):
    """Integrate -(A_int-A_background,int)c_inc without incident projection."""
    rhs, records = build_lifted_rhs_batch((handle,))
    return rhs[0], records[0]


def build_selected_lifted_rhs(handle, test_edges):
    """Integrate selected local RWG test rows without a global pair-class array.

    Source traces remain analytic on all source faces. Selection changes only
    the requested rows, preserving touching, near and far integration policies.
    """
    rhs, records = build_selected_lifted_rhs_batch((handle,), test_edges)
    return rhs[0], records[0]


def build_selected_lifted_rhs_batch(handles, test_edges):
    """Selected lifted RHS rows [S,2*selected] for several carriers of one body.

    Classification stays bounded by the configured pair batches. Classified
    pairs are queued per class and integrated in node-bounded batches shared
    by all carriers; each carrier keeps its own near acceptance and record.
    """
    from .nearfield import _selected_edges, _local_indices
    from .quadrature import _classify_pairs, _pair_geometry
    handles = _validated_batch(handles)
    handle = handles[0]
    basis, config = handle.basis, handle.integration
    if basis._mesh_version != (id(basis.mesh.vertices), basis.mesh.vertices._version,
                               id(basis.mesh.faces), basis.mesh.faces._version):
        raise ValueError('Selected lifted RHS basis geometry changed')
    started = time.perf_counter()
    edges, positions, faces = _selected_edges(basis, test_edges)
    count = len(handles)
    dtype = torch.complex128 if basis.mesh.dtype == torch.float64 else torch.complex64
    rhs = torch.zeros((count, 2, len(edges)), device=basis.mesh.device, dtype=dtype)
    nf = len(basis.mesh.faces)
    counts = [0]*5
    maximum_pairs = 0
    stats = [{'maximum_order': 0, 'pair_evaluations': 0, 'maximum_accepted_relative_difference': 0.}
             for _ in handles]
    budget = _pair_budget(config)
    far_block = max(1, _FAR_VALUE_BYTES//(count*6*rhs.element_size()))
    limits = [_touching_batch(config, category, count) for category in range(3)] + [far_block, far_block]
    queues = [[] for _ in range(5)]

    def integrate(category):
        if not queues[category]:
            return
        r = torch.cat([part[0] for part in queues[category]])
        c = torch.cat([part[1] for part in queues[category]])
        queues[category].clear()
        if category < 3:
            mask = _touching_split(basis.mesh, r, c)
            local = torch.empty((count, len(r), 2, 3), dtype=dtype, device=r.device)
            well, sliver = (~mask).nonzero().flatten(), mask.nonzero().flatten()
            if len(well):
                local[:, well] = torch.cat([_touching_values(handles, r[ids], c[ids], category, config.singular_order)
                                            for ids in well.split(limits[category])], 1)
            for ids in sliver.split(NEAR_BATCH) if len(sliver) else ():
                local[:, ids] = _anisotropic_touching_values(handles, r[ids], c[ids], category, config, stats)
        elif category == 3:
            local, observed = _near_values(handles, r, c, config, budget)
            for total, record in zip(stats, observed):
                _merge_near(total, record)
        else:
            local = _product_values(handles, r, c, config.order, budget)
        indices, keep = _local_indices(basis.face_edges[r], edges, positions)
        rhs.index_add_(2, indices[keep], local.permute(0, 2, 1, 3)[:, :, keep])

    queued = [0]*5
    batch = max(config.pair_batch, config.far_pair_batch)
    centres, radii = _pair_geometry(basis.mesh)
    categories = torch.arange(5, device=basis.mesh.device)
    # Copy counts once per lookahead window. Queue flushes keep their original
    # classification-batch boundaries and pair order, including near refinement.
    lookahead = 64
    for window in range(0, len(faces)*nf, lookahead*batch):
        chunks, sizes = [], []
        for start in range(window, min(window+lookahead*batch, len(faces)*nf), batch):
            ids = torch.arange(start, min(start+batch, len(faces)*nf), device=basis.mesh.device)
            rows, cols = faces[ids//nf], ids % nf
            classes = _classify_pairs(basis.mesh, rows, cols, centres, radii, 2.)
            maximum_pairs = max(maximum_pairs, len(ids))
            chunks.append((rows, cols, torch.argsort(classes, stable=True)))
            sizes.append((classes[:, None] == categories).sum(0))
        for (rows, cols, order), counts_in_chunk in zip(chunks, torch.stack(sizes).tolist()):
            for category, part in enumerate(order.split(counts_in_chunk)):
                size = counts_in_chunk[category]
                counts[category] += size
                if not size:
                    continue
                queues[category].append((rows[part], cols[part]))
                queued[category] += size
                if queued[category] >= limits[category]:
                    integrate(category)
                    queued[category] = 0
    for category in range(5):
        integrate(category)
    seconds = time.perf_counter()-started
    records = [{'version': 'selected_analytic_lifted_rhs_v1',
        'incident_carrier_identity': local.incident_carrier.identity,
        'difference_kernel_version': KERNEL_DIFFERENCE_VERSION,
        'rhs_integration': vars(config), 'selected_edges': len(edges),
        'test_faces': len(faces), 'source_faces': nf, 'triangle_pair_counts': counts,
        'maximum_classification_pairs': maximum_pairs,
        'rhs_near_quadrature': record, 'rhs_seconds': seconds, 'rhs_source_batch': count,
        'rhs_near_estimator': 'successive_order_difference_per_panel_pair',
        'rhs_near_norm': 'Euclidean norm of six local row entries after exterior-impedance balancing',
        'rhs_touching_and_far_accuracy': 'independent_order_convergence_required',
        'global_pair_classification_retained': False, 'global_rhs_error_certified': False}
        for local, record in zip(handles, stats)]
    return rhs.flatten(1), records
