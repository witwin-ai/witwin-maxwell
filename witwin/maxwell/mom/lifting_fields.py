"""Analytic carrier potentials and total currents for lifted scalar traces."""
from __future__ import annotations

import math
import torch

from .fields import _static_panel_gradient, _triangle_distance, _validate_arrays, _sum_observation_pairs
from .lifting import _gradient_remainder, difference_kernels, incident_trace, medium_differences
from .operator import local_geometry
from .quadrature import duffy_fan, triangle_rule


def evaluate_total_currents(handle, scattered_coefficients, points, face_indices):
    """Evaluate physical total J/M with explicit face ownership, including edges.

    These are pointwise physical currents, not coefficients of a total RWG
    approximation. The analytic incident contribution is not projected.
    """
    _validate_arrays(handle, scattered_coefficients, points)
    mesh = handle.basis.mesh
    if (not isinstance(face_indices, torch.Tensor) or face_indices.device != mesh.device
            or face_indices.dtype != torch.int64 or face_indices.shape != (len(points),)):
        raise ValueError('Total current face_indices must be an int64 vector on the mesh device.')
    if bool(((face_indices < 0) | (face_indices >= len(mesh.faces))).any()):
        raise ValueError('Total current face index is outside the surface.')
    tri = mesh.triangles[face_indices]
    a, ab, ac = tri[:, 0], tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
    delta = points-a
    d00, d01, d11 = (ab*ab).sum(-1), (ab*ac).sum(-1), (ac*ac).sum(-1)
    d20, d21 = (delta*ab).sum(-1), (delta*ac).sum(-1)
    denominator = d00*d11-d01*d01
    u, v = (d11*d20-d01*d21)/denominator, (d00*d21-d01*d20)/denominator
    tolerance = 128*torch.finfo(points.dtype).eps
    height = abs((delta*mesh.normals[face_indices]).sum(-1))
    scale = torch.maximum(d00, d11).sqrt()
    if bool(((u < -tolerance) | (v < -tolerance) | (u+v > 1+tolerance)
             | (height > tolerance*scale)).any()):
        raise ValueError('Total current points must lie on their explicitly declared faces.')
    edges, free, slope = local_geometry(handle.basis)
    displacement = (points[:, None]-free[face_indices]).to(scattered_coefficients.dtype)
    n = handle.basis.n_edges
    currents = []
    analytic = incident_trace(handle, points[:, None], face_indices)
    for offset, carrier in ((0, analytic[0]), (n, analytic[1])):
        local = scattered_coefficients[offset:offset+n][edges[face_indices]]*slope[face_indices]
        currents.append((local[..., None]*displacement).sum(1)+carrier[:, 0])
    return tuple(currents)


def _smooth_background_gradient(delta, k):
    """Remove the static gradient analytically before quadrature."""
    radius = torch.linalg.vector_norm(delta, dim=-1)
    z = 1j*k*radius
    remainder = _gradient_remainder(z)
    return delta*remainder[..., None]/(4*math.pi*radius[..., None]**3)


def carrier_difference_potentials(handle, queries, order, surface_faces):
    """Principal-value (Phi_int-Phi_ext)c_inc with static singularity extraction.

    Explicit one-sided normal jumps are applied separately. Tangential curl
    jumps cancel between media; normal divergence terms generally do not.
    """
    electric, magnetic = carrier_difference_batch((handle,), queries, order, surface_faces)
    return electric[0], magnetic[0]


def carrier_difference_batch(handles, queries, order, surface_faces):
    """carrier_difference_potentials for several carriers on one body: [S,P,3].

    Geometry, quadrature and difference kernels are shared by all carriers.
    """
    from .fields import QUERY_CHUNK, SAMPLE_CAP
    from .lifting import _incident_traces, _validated_batch
    handles = _validated_batch(handles)
    handle = handles[0]
    count = len(handles)
    mesh, exterior, interior = handle.basis.mesh, handle.exterior, handle.interior
    triangles = mesh.triangles
    de, dm, dk = medium_differences(exterior, interior)
    dtype = torch.complex128 if mesh.dtype == torch.float64 else torch.complex64
    if not len(queries):
        empty = torch.zeros((count, *queries.shape), device=queries.device, dtype=dtype)
        return empty, empty.clone()
    panel_radius = torch.linalg.vector_norm(triangles-triangles.mean(1)[:, None], dim=-1).amax(-1)
    # Paired carrier traces scale with the carrier count.
    samples = max(1, SAMPLE_CAP//max(1, count//4))
    output_e, output_h = [], []
    for start in range(0, len(queries), QUERY_CHUNK):
        points = queries[start:start+QUERY_CHUNK]
        faces = surface_faces[start:start+QUERY_CHUNK]
        distances = _triangle_distance(points, triangles)
        oi_all = torch.arange(len(points), device=mesh.device)[:, None].expand(-1, len(triangles)).flatten()
        ti_all = torch.arange(len(triangles), device=mesh.device)[None].expand(len(points), -1).flatten()
        near = (distances < 2*panel_radius).flatten()
        electric = torch.zeros((len(points), count*3), device=mesh.device, dtype=dtype)
        magnetic = torch.zeros_like(electric)
        for is_near in (False, True):
            selected = torch.nonzero(near == is_near).flatten()
            batch = max(1, samples//(3*order**2 if is_near else order**2))
            for indices in selected.split(batch):
                oi, ti = oi_all[indices], ti_all[indices]
                obs = points[oi]
                if is_near:
                    src, weights = duffy_fan(obs, triangles[ti], order)
                else:
                    src, weights = triangle_rule(triangles[ti], order)
                j, m, div_j, div_m = _incident_traces(handles, src, ti)
                g, grad, dg, dgrad = difference_kernels(obs[:, None]-src, exterior, interior, delta_k=dk)
                if is_near:
                    self_panel = (faces[oi] == ti) & (faces[oi] >= 0)
                    exact_static = _static_panel_gradient(obs, triangles[ti], principal_value=self_panel)
                    smooth = _smooth_background_gradient(obs[:, None]-src, exterior.k)
                    delta = obs[:, None]-src
                    radius = torch.linalg.vector_norm(delta, dim=-1)
                    static = -delta/(4*math.pi*radius[..., None]**3)
                    at_query = _incident_traces(handles, obs[:, None], ti)
                weighted = weights[..., None]

                def electric_part(current, divergence, coefficient_i, delta_coefficient,
                                  dual_i, dual_e, delta_dual, at_query_divergence):
                    weighted_g = delta_coefficient*g+coefficient_i*dg
                    vector = (weighted_g[..., None]*current*weighted).sum(-2)
                    a = 1/(exterior.omega**2*dual_i)
                    difference = -delta_dual/(exterior.omega**2*dual_i*dual_e)
                    if is_near:
                        value = at_query_divergence[..., 0]
                        gradient = difference*(exact_static*value[..., None]
                            + (static*(divergence-value[..., None])[..., None]*weighted).sum(-2))
                        gradient = gradient+((difference*smooth+a*dgrad)*divergence[..., None]*weighted).sum(-2)
                    else:
                        gradient = ((difference*grad+a*dgrad)*divergence[..., None]*weighted).sum(-2)
                    return vector+gradient

                ej = 1j*exterior.omega*electric_part(j, div_j, interior.mu, dm, interior.epsilon,
                    exterior.epsilon, de, at_query[2] if is_near else None)
                hm = 1j*exterior.omega*electric_part(m, div_m, interior.epsilon, de, interior.mu,
                    exterior.mu, dm, at_query[3] if is_near else None)
                curl_j = (torch.linalg.cross(dgrad.expand_as(j), j)*weighted).sum(-2)
                curl_m = (torch.linalg.cross(dgrad.expand_as(m), m)*weighted).sum(-2)
                # [S,B,3] -> [B,S*3] for the ordered observation reduction.
                electric = electric + _sum_observation_pairs((ej-curl_m).transpose(0, 1).flatten(1), oi, len(points))
                magnetic = magnetic + _sum_observation_pairs((curl_j+hm).transpose(0, 1).flatten(1), oi, len(points))
        output_e.append(electric.unflatten(1, (count, 3)))
        output_h.append(magnetic.unflatten(1, (count, 3)))
    return torch.cat(output_e).transpose(0, 1), torch.cat(output_h).transpose(0, 1)


def carrier_difference_jump(handle, queries, faces):
    """Interior-minus-exterior normal jump; tangential half jumps cancel."""
    _, _, div_j, div_m = incident_trace(handle, queries[:, None], faces)
    n = handle.basis.mesh.normals[faces].to(div_j.dtype)
    exterior, interior = handle.exterior, handle.interior
    de, dm, _ = medium_differences(exterior, interior)
    e = 1j*n*div_j[:, 0, None]*de/(exterior.omega*interior.epsilon*exterior.epsilon)
    h = 1j*n*div_m[:, 0, None]*dm/(exterior.omega*interior.mu*exterior.mu)
    return e, h


def decode_lifted(handle, scattered_coefficients, queries, region_id, field_kind,
                  trace_side=None, *, query_config=None):
    """Compose RWG and analytic carriers with explicit interior total semantics.

    Exterior queries use the unchanged linear equivalent-current potential.
    Interior adaptation estimates successive-order differences of the complete
    physical field, so cancellation between its components is included.
    """
    from .fields import decode, _query_context
    if region_id not in (1, 'interior') or isinstance(region_id, bool):
        if field_kind in ('total', 'incident'):
            _query_context(handle, scattered_coefficients, queries, region_id,
                           field_kind, trace_side, query_config)
            incident = handle.incident_carrier.evaluate(queries, handle.exterior)
            if field_kind == 'incident':
                return incident
            scattered = decode(handle, scattered_coefficients, queries, region_id, 'scattered',
                               trace_side, query_config=query_config)
            return tuple(a+b for a, b in zip(scattered, incident))
        return decode(handle, scattered_coefficients, queries, region_id, field_kind,
                      trace_side, query_config=query_config)
    e, h = _lifted_interior((handle,), scattered_coefficients[:, None], queries, region_id, field_kind,
                            trace_side, query_config)
    return e[0], h[0]


def _lifted_interior(handles, columns, queries, region_id, field_kind, trace_side, query_config):
    """Interior total fields [S,P,3] of carrier handles sharing one body.

    ``columns`` holds the local scattered coefficients [2*edges,S]. Each
    carrier is adapted and logged exactly as if it were queried alone.
    """
    from .fields import _potentials, _query_context, _record_query, _trace_jump, _validate_arrays, adapt_columns
    import time
    handle = handles[0]
    for index, local in enumerate(handles[1:], 1):
        _validate_arrays(local, columns[:, index], queries)
    _, config, faces, on_surface = _query_context(handle, columns[:, 0], queries, region_id,
                                                 field_kind, trace_side, query_config)
    start = time.perf_counter()
    medium = handle.interior
    order = max(handle.integration.order, handle.integration.singular_order)
    logs = [{'query_count': len(queries), 'surface_count': int(on_surface.sum()),
             'adaptive': config.adaptive, 'converged': False, 'checks': [],
             'relative_tolerance': config.relative_tolerance,
             'absolute_tolerance_V_per_m': config.absolute_tolerance,
             'norm': 'sqrt(sum(abs(E_total)^2)+abs(eta_int)^2*sum(abs(H_total)^2)) per query',
             'estimator': 'complete_lifted_field_successive_order_difference',
             'device': str(columns.device), 'peak_memory_scope': 'device_process_high_water_mark'}
            for _ in handles]

    def record():
        _record_query(handle, logs[0], start)
        for local, log in zip(handles[1:], logs[1:]):
            log.update(seconds=logs[0]['seconds'], peak_allocated_bytes=logs[0]['peak_allocated_bytes'])
            local.last_field_query_log = log
            if hasattr(local, 'metadata'):
                local.metadata['last_field_query'] = log

    if config.adaptive and order >= config.max_order:
        for log in logs:
            log['failure_reason'] = 'initial_order_leaves_no_refinement'
        record()
        raise ValueError('Field max_order must exceed the initial integration order for error estimation.')
    incident = [local.incident_carrier.evaluate(queries, local.exterior) for local in handles]
    incident_e = torch.stack([e for e, _ in incident], -1)
    incident_h = torch.stack([h for _, h in incident], -1)
    surface = bool(on_surface.any())
    if surface:
        jumps = []
        for index, local in enumerate(handles):
            raw_jump = _trace_jump(handle, columns[:, index], queries[on_surface], faces[on_surface], medium)
            carrier_jump = carrier_difference_jump(local, queries[on_surface], faces[on_surface])
            jumps.append([.5*(raw+carrier) for raw, carrier in zip(raw_jump, carrier_jump)])
        jump = [torch.stack([jump[index] for jump in jumps], -1) for index in range(2)]

    def evaluate(current_order):
        raw = _potentials(handle, columns, queries, medium, order=current_order,
                          singular_order=current_order, surface_faces=faces)
        carrier = carrier_difference_batch(handles, queries, current_order, faces)
        fields = [inc-value-difference.permute(1, 2, 0)
                  for inc, value, difference in zip((incident_e, incident_h), raw, carrier)]
        if surface:
            for index in range(2):
                fields[index] = fields[index].clone()
                fields[index][on_surface] = fields[index][on_surface]+jump[index]
        return tuple(fields)

    e, h, failed = adapt_columns(evaluate, order, config, medium.eta, logs, require_finite=False)
    record()
    if failed:
        raise RuntimeError('MoM lifted field quadrature did not converge within max_order; inspect last_field_query_log.')
    return e.permute(2, 0, 1), h.permute(2, 0, 1)


def decode_lifted_batch(handles, columns, queries, region_id, field_kind, trace_side=None, *, query_config=None):
    """decode_lifted for carrier handles sharing one body: fields [S,P,3].

    ``columns`` holds local scattered coefficients [2*edges,S]. Quadrature
    geometry and kernels are shared; every carrier keeps its own adaptive
    acceptance, result and query log.
    """
    from .fields import _evaluate_adaptive, _query_context, _trace_jump, _validate_arrays
    from .lifting import _validated_batch
    handles = _validated_batch(handles)
    if any(local.kernel is not handles[0].kernel for local in handles):
        raise ValueError('Batched field queries require one shared kernel')
    if columns.ndim != 2 or columns.shape[1] != len(handles):
        raise ValueError('Batched field queries require one coefficient column per carrier handle')
    if region_id in (1, 'interior') and not isinstance(region_id, bool):
        return _lifted_interior(handles, columns, queries, region_id, field_kind, trace_side, query_config)
    handle = handles[0]
    for index, local in enumerate(handles[1:], 1):
        _validate_arrays(local, columns[:, index], queries)
    _, config, faces, on_surface = _query_context(handle, columns[:, 0], queries, region_id,
                                                 field_kind, trace_side, query_config)
    if field_kind == 'incident':
        electric = torch.zeros((len(handles), *queries.shape), device=queries.device, dtype=columns.dtype)
        magnetic = torch.zeros_like(electric)
    else:
        e, h = _evaluate_adaptive(handle, columns, queries, handle.exterior, faces, config)
        for local, log in zip(handles, handle.last_field_query_logs):
            local.last_field_query_log = log
            if hasattr(local, 'metadata'):
                local.metadata['last_field_query'] = log
        if bool(on_surface.any()):
            e, h = e.clone(), h.clone()
            for index in range(len(handles)):
                je, jh = _trace_jump(handle, columns[:, index], queries[on_surface], faces[on_surface], handle.exterior)
                e[on_surface, :, index] = e[on_surface, :, index]+.5*je
                h[on_surface, :, index] = h[on_surface, :, index]+.5*jh
        electric, magnetic = e.permute(2, 0, 1), h.permute(2, 0, 1)
    if field_kind in ('total', 'incident'):
        incident = [local.incident_carrier.evaluate(queries, local.exterior) for local in handles]
        electric = electric+torch.stack([value for value, _ in incident])
        magnetic = magnetic+torch.stack([value for _, value in incident])
    return electric, magnetic


def system_local_coefficients(handle, coefficients, index):
    system = handle.system
    return torch.cat((coefficients[system.J_slices[index]], coefficients[system.M_slices[index]]))


def classify_system_queries(handle, coefficients, points):
    """Resolve global region and unique face-interior trace ownership on CUDA."""
    from .fields import classify_queries
    system = handle.system
    system.assert_current()
    if (not isinstance(coefficients, torch.Tensor) or coefficients.shape != (2*system.total_edges,)
            or coefficients.device != system.device):
        raise ValueError('MoM system coefficients must match the complete global body layout')
    regions = None
    owners = None
    distances = None
    for i, local in enumerate(handle.body_handles):
        _validate_arrays(local, system_local_coefficients(handle, coefficients, i), points)
        inside, distance, faces = classify_queries(local.basis.mesh, points, allow_surface=True)
        if regions is None:
            regions = torch.zeros(len(points), device=points.device, dtype=torch.int64)
            owners = torch.zeros_like(regions)
            distances = torch.full_like(distance, torch.inf)
        on = faces >= 0
        occupied = inside | on
        if bool((occupied & ((regions != 0) | (owners != 0))).any()):
            raise ValueError('MoM query has ambiguous body ownership')
        regions = torch.where(inside & ~on, i+1, regions)
        owners = torch.where(on, i+1, owners)
        distances = torch.minimum(distances, distance)
    return regions, owners, distances


def decode_system(handle, coefficients, points, region_id, field_kind, trace_side=None, *, query_config=None):
    """Sum exterior scattered potentials; decode each interior with its own trace."""
    from .fields import FieldQueryConfig
    region = handle.system.region(region_id)
    config = FieldQueryConfig() if query_config is None else query_config
    if not isinstance(config, FieldQueryConfig):
        raise TypeError('query_config must be FieldQueryConfig')
    if field_kind not in ('total', 'scattered', 'incident') or (region and field_kind != 'total'):
        raise ValueError("Interior queries require field_kind='total'; exterior supports total/scattered/incident")
    if trace_side not in (None, 'interior', 'exterior') or (trace_side is not None and (trace_side == 'interior') != bool(region)):
        raise ValueError('trace_side must match region_id')
    if region == 0 and field_kind != 'incident' and config.adaptive and len(handle.system.bodies) > 1:
        return _adaptive_system_exterior(handle, coefficients, points, field_kind, trace_side, config)
    regions, owners, _ = classify_system_queries(handle, coefficients, points)
    on = owners != 0
    if bool(on.any()) and trace_side is None:
        raise ValueError('MoM on-surface queries require explicit trace_side')
    if trace_side is not None and not bool(on.all()):
        raise ValueError('trace_side is only defined for on-surface queries')
    if bool(((~on & (regions != region)) | (on & (owners != region) & bool(region))).any()):
        raise ValueError('MoM query geometry does not match the requested region_id')
    logs = []
    if region:
        local = handle.body_handles[region-1]
        from .tensor_medium import TensorMedium
        if isinstance(local.interior, TensorMedium):
            from .anisotropic_fields import decode_tensor_lifted
            result = decode_tensor_lifted(local, system_local_coefficients(handle, coefficients, region-1),
                                         points, trace_side, query_config=config)
        else:
            result = decode_lifted(local, system_local_coefficients(handle, coefficients, region-1), points,
                                   1, 'total', trace_side, query_config=config)
        logs.append({'region_id': region, **getattr(local, 'last_field_query_log', {})})
    else:
        result = tuple(torch.zeros_like(points, dtype=coefficients.dtype) for _ in range(2))
        if field_kind != 'incident':
            for i, local in enumerate(handle.body_handles):
                local_x = system_local_coefficients(handle, coefficients, i)
                for own_trace in (False, True):
                    selected = (owners == i+1) if own_trace else (owners != i+1)
                    if not bool(selected.any()):
                        continue
                    contribution = decode_lifted(local, local_x, points[selected], 0, 'scattered',
                        'exterior' if own_trace else None, query_config=config)
                    # Index assignment preserves coefficient autograd through CopySlices.
                    for value, part in zip(result, contribution):
                        value[selected] = value[selected]+part
                    logs.append({'region_id': i+1, 'own_trace': own_trace,
                                 **getattr(local, 'last_field_query_log', {})})
        if field_kind in ('total', 'incident'):
            result = tuple(a+b for a, b in zip(result, handle.incident_carrier.evaluate(points, handle.exterior)))
    handle.last_field_query_log = {'region_id': region, 'query_count': len(points), 'body_queries': logs,
        'converged': all(log.get('converged', False) for log in logs),
        'estimator': 'per_body_quadrature_only', 'global_error_certified': False}
    return result


def decode_system_batch(handles, coefficients, points, region_id, field_kind, trace_side=None, *, query_config=None):
    """decode_system for several solved RHS of one prepared system: [S,P,3].

    Handles share the system, exterior and integration and differ in their
    carriers; ``coefficients`` holds one global scattered row per handle.
    Single-body scalar systems share query classification, quadrature and
    kernels across RHS; every RHS keeps its own adaptive acceptance and log.
    Other layouts are evaluated one RHS at a time.
    """
    from .fields import FieldQueryConfig
    from .tensor_medium import TensorMedium
    handles = tuple(handles)
    if not handles or len(coefficients) != len(handles):
        raise ValueError('Batched system queries require one coefficient row per handle')
    system = handles[0].system
    if any(h.system is not system or h.exterior is not handles[0].exterior
           or h.integration != handles[0].integration for h in handles[1:]):
        raise ValueError('Batched system queries require one shared system, exterior and integration')
    config = FieldQueryConfig() if query_config is None else query_config
    if len(system.bodies) != 1 or isinstance(system.bodies[0].interior, TensorMedium):
        fields = [decode_system(h, x, points, region_id, field_kind, trace_side, query_config=config)
                  for h, x in zip(handles, coefficients)]
        return tuple(torch.stack(parts) for parts in zip(*fields))
    region = system.region(region_id)
    if not isinstance(config, FieldQueryConfig):
        raise TypeError('query_config must be FieldQueryConfig')
    if field_kind not in ('total', 'scattered', 'incident') or (region and field_kind != 'total'):
        raise ValueError("Interior queries require field_kind='total'; exterior supports total/scattered/incident")
    if trace_side not in (None, 'interior', 'exterior') or (trace_side is not None and (trace_side == 'interior') != bool(region)):
        raise ValueError('trace_side must match region_id')
    if (not isinstance(coefficients, torch.Tensor) or coefficients.shape[1:] != (2*system.total_edges,)
            or coefficients.device != system.device):
        raise ValueError('MoM system coefficients must match the complete global body layout')
    # Geometry classification is shared; each row is validated when decoded.
    regions, owners, _ = classify_system_queries(handles[0], coefficients[0], points)
    on = owners != 0
    if bool(on.any()) and trace_side is None:
        raise ValueError('MoM on-surface queries require explicit trace_side')
    if trace_side is not None and not bool(on.all()):
        raise ValueError('trace_side is only defined for on-surface queries')
    if bool(((~on & (regions != region)) | (on & (owners != region) & bool(region))).any()):
        raise ValueError('MoM query geometry does not match the requested region_id')
    locals_ = tuple(h.body_handles[0] for h in handles)
    columns = torch.stack([system_local_coefficients(h, x, 0) for h, x in zip(handles, coefficients)], -1)
    logs = [[] for _ in handles]
    if region:
        result = _lifted_interior(locals_, columns, points, 1, 'total', trace_side, config)
        for log, local in zip(logs, locals_):
            log.append({'region_id': region, **getattr(local, 'last_field_query_log', {})})
    else:
        shape = (len(handles), *points.shape)
        result = (torch.zeros(shape, device=points.device, dtype=coefficients.dtype),
                  torch.zeros(shape, device=points.device, dtype=coefficients.dtype))
        if field_kind != 'incident':
            for own_trace in (False, True):
                selected = (owners == 1) if own_trace else (owners != 1)
                if not bool(selected.any()):
                    continue
                contribution = decode_lifted_batch(locals_, columns, points[selected], 0, 'scattered',
                    'exterior' if own_trace else None, query_config=config)
                for value, part in zip(result, contribution):
                    value[:, selected] = value[:, selected]+part
                for log, local in zip(logs, locals_):
                    log.append({'region_id': 1, 'own_trace': own_trace, **getattr(local, 'last_field_query_log', {})})
        if field_kind in ('total', 'incident'):
            incident = [h.incident_carrier.evaluate(points, h.exterior) for h in handles]
            result = tuple(value+torch.stack(parts) for value, parts in zip(result, zip(*incident)))
    for h, log in zip(handles, logs):
        h.last_field_query_log = {'region_id': region, 'query_count': len(points), 'body_queries': log,
            'converged': all(entry.get('converged', False) for entry in log),
            'estimator': 'per_body_quadrature_only', 'global_error_certified': False}
    return result


def query_fields_batch(datas, points, region_id, field_kind, trace_side=None, *, query_config=None):
    """Query E/H [S,P,3] of several physical results of one prepared system.

    Every SurfaceData is validated as in its own ``query_fields``; results
    sharing a non-periodic scalar system are decoded together.
    """
    from .runtime import SurfaceData
    datas = tuple(datas)
    if not datas or any(not isinstance(d, SurfaceData) for d in datas):
        raise TypeError('Batched field queries require SurfaceData results')
    first = datas[0]
    shared = (first.config.periodic is None and first.config.dyadic_integration is None
              and all(d.handle.system is first.handle.system and d.handle.exterior is first.handle.exterior
                      and d.config == first.config for d in datas))
    if not shared:
        fields = [d.query_fields(points, region_id, field_kind, trace_side, query_config=query_config) for d in datas]
        return tuple(torch.stack(parts) for parts in zip(*fields))
    for d in datas:
        d._validate_carrier_binding()
    return decode_system_batch(tuple(d.handle for d in datas), torch.stack([d.scattered_coefficients for d in datas]),
        points, region_id, field_kind, trace_side,
        query_config=first.config.query_config if query_config is None else query_config)


def far_field_system(handle, coefficients, directions):
    """Coherently sum complex amplitudes, retaining cross-body interference."""
    from .fields import far_field
    handle.system.assert_current()
    fields = [far_field(local, system_local_coefficients(handle, coefficients, i), directions)
              for i, local in enumerate(handle.body_handles)]
    return tuple(torch.stack(parts).sum(0) for parts in zip(*fields))


def _adaptive_system_exterior(handle, coefficients, points, field_kind, trace_side, config):
    """Estimate error after coherent summation, including inter-body cancellation."""
    import copy
    from dataclasses import replace
    order = max(handle.integration.order, handle.integration.singular_order)
    log = {'region_id': 0, 'query_count': len(points), 'checks': [], 'converged': False,
           'estimator': 'complete_multibody_field_successive_order_difference',
           'global_error_certified': False, 'relative_tolerance': config.relative_tolerance,
           'absolute_tolerance_V_per_m': config.absolute_tolerance}
    handle.last_field_query_log = log
    if order >= config.max_order:
        log['failure_reason'] = 'initial_order_leaves_no_refinement'
        raise ValueError('Field max_order must exceed the initial integration order for error estimation')

    def evaluate(order):
        local_handle = copy.copy(handle)
        local_handle.body_handles = tuple(copy.copy(h) for h in handle.body_handles)
        for h in local_handle.body_handles:
            h.integration = replace(h.integration, order=order, singular_order=order,
                                    near_max_order=max(h.integration.near_max_order, order+2))
        result = decode_system(local_handle, coefficients, points, 0, field_kind, trace_side,
                               query_config=replace(config, adaptive=False))
        log['last_body_queries'] = local_handle.last_field_query_log['body_queries']
        return result

    old = evaluate(order)
    if not len(points):
        log.update(converged=True, final_order=order, failure_reason=None)
        return old
    while order < config.max_order:
        order = min(2*order, config.max_order)
        result = evaluate(order)
        error = ((result[0]-old[0]).abs().square().sum(-1)
                 +abs(handle.exterior.eta)**2*(result[1]-old[1]).abs().square().sum(-1)).sqrt()
        norm = (result[0].abs().square().sum(-1)+abs(handle.exterior.eta)**2*result[1].abs().square().sum(-1)).sqrt()
        log['checks'].append({'order': order, 'max_absolute_error_V_per_m': float(error.max().detach()),
                             'max_relative_error': float((error/norm.clamp_min(torch.finfo(norm.dtype).tiny)).max().detach())})
        if bool((error <= config.absolute_tolerance+config.relative_tolerance*norm).all()):
            log.update(converged=True, final_order=order, failure_reason=None)
            return result
        old = result
    log.update(final_order=order, failure_reason='quadrature_order_budget_exhausted')
    raise RuntimeError('MoM complete multi-body field quadrature did not converge within max_order')
