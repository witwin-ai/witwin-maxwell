"""Off-surface tensor current potentials with weak static source derivatives."""
from __future__ import annotations

from types import SimpleNamespace
import torch
from ..constants import EPSILON_0, MU_0
from .dyadic_green import electrostatic_kernel
from .fields import _validate_arrays, classify_queries, _triangle_distance
from .operator import local_geometry
from .quadrature import triangle_rule, duffy_fan, triangle_outside_inscribed_disk


def tensor_current_potentials(basis, coefficients, queries, kernel, *, order=6,
                              pair_batch=16, query_batch=16, surface_faces=None):
    """Evaluate (E,H) radiated by complete closed-surface RWG J/M currents.

    No region sign, incident field, or trace jump is applied here. Surface
    points require explicit surface_faces and return a principal value. Integrating the static source derivative by parts
    replaces its Hessian with gradient times RWG divergence over the complete
    closed support. Surface-order convergence is a separate caller obligation.
    """
    sample = _rwg_current_sampler(basis, coefficients, queries, kernel)
    return _tensor_surface_potentials(basis, queries, kernel, sample, order=order,
        pair_batch=pair_batch, query_batch=query_batch, surface_faces=surface_faces)


def _rwg_current_sampler(basis, coefficients, queries, kernel):
    """Bind validated RWG coefficients for repeated panel sampling."""
    medium = kernel.medium
    medium.assert_current()
    _validate_arrays(SimpleNamespace(basis=basis), coefficients, queries)
    if (medium.epsilon_r.device != coefficients.device or medium.epsilon_r.dtype != coefficients.dtype):
        raise ValueError('Tensor medium and current coefficients must share CUDA device and precision')
    edges, free, slopes = local_geometry(basis)
    ne = basis.n_edges
    local_j = coefficients[:ne][edges]*slopes
    local_m = coefficients[ne:][edges]*slopes
    divj, divm = 2*local_j.sum(-1), 2*local_m.sum(-1)

    def sample(source, faces):
        displacement = (source[:, :, None]-free[faces, None]).to(coefficients.dtype)
        j = torch.einsum('pl,pqlc->pqc', local_j[faces], displacement)
        m = torch.einsum('pl,pqlc->pqc', local_m[faces], displacement)
        return j, m, divj[faces, None].expand(source.shape[:2]), divm[faces, None].expand(source.shape[:2])

    return sample


def tensor_current_potential_columns(basis, coefficients, queries, kernel, *,
                                     order=6, pair_batch=16, query_batch=16):
    """Off-surface [query, xyz, column] fields with shared Green evaluations.

    Columns are complete RWG [J,M] vectors. No incident sheet, region sign or
    trace jump is included. Field/output workspace scales with column count;
    geometry and spectral work do not repeat for each right-hand side.
    """
    if (not isinstance(coefficients,torch.Tensor) or coefficients.ndim!=2
            or coefficients.shape[0]!=2*basis.n_edges or coefficients.shape[1]<1
            or not bool(torch.isfinite(coefficients).all())):
        raise ValueError('Tensor field columns require finite [2*edges,columns] coefficients')
    _validate_arrays(SimpleNamespace(basis=basis),coefficients[:,0],queries)
    medium=kernel.medium;medium.assert_current()
    if medium.epsilon_r.device!=coefficients.device or medium.epsilon_r.dtype!=coefficients.dtype:
        raise ValueError('Tensor medium and current columns must share CUDA device and precision')
    edges,free,slopes=local_geometry(basis)
    n=basis.n_edges;columns=coefficients.shape[1]
    j=coefficients[:n][edges]*slopes[...,None]
    m=coefficients[n:][edges]*slopes[...,None]
    dj,dm=2*j.sum(1),2*m.sum(1)
    def sample(source,faces):
        delta=(source[:,:,None]-free[faces,None]).to(coefficients.dtype)
        return (torch.einsum('plr,pqlc->pqcr',j[faces],delta),
                torch.einsum('plr,pqlc->pqcr',m[faces],delta),
                dj[faces,None].expand(*source.shape[:2],columns),
                dm[faces,None].expand(*source.shape[:2],columns))
    e,h,receipt=_tensor_surface_potentials(basis,queries,kernel,sample,order=order,
        pair_batch=pair_batch,query_batch=query_batch,columns=columns)
    receipt.update(representation='tensor_radiated_current_columns',column_count=columns,
                   shared_column_kernel=True)
    return e,h,receipt



def tensor_selected_current_blocks(basis, queries, edge_indices, kernel, *, order=6,
                                   pair_batch=16, query_batch=16):
    """Radiated fields for selected J/M unit columns on complete RWG supports.

    Selected columns retain repetition and order. Only their supporting panels
    are integrated; no full edge-by-column identity matrix is allocated.
    Region signs and query admission belong to the interior decoder caller.
    """
    if (not isinstance(edge_indices,torch.Tensor) or edge_indices.device!=basis.mesh.device
            or edge_indices.dtype!=torch.int64 or edge_indices.ndim!=1
            or bool(((edge_indices<0)|(edge_indices>=basis.n_edges)).any())):
        raise ValueError('Selected tensor edges require matching CUDA int64 indices in range')
    selected,inverse=torch.unique(edge_indices,sorted=True,return_inverse=True)
    edges,free,slopes=local_geometry(basis)
    active=torch.isin(edges,selected).any(-1).nonzero().flatten()
    count=len(selected)
    def sample(source,faces):
        local=(edges[faces,:,None]==selected[None,None,:])*slopes[faces,:,None]
        delta=(source[:,:,None]-free[faces,None]).to(kernel.medium.epsilon_r.dtype)
        value=torch.einsum('plr,pqlc->pqcr',local.to(delta.dtype),delta)
        divergence=(2*local.sum(1)).to(delta.dtype)[:,None].expand(-1,source.shape[1],-1)
        zero=torch.zeros_like(value);dz=torch.zeros_like(divergence)
        return (torch.cat((value,zero),-1),torch.cat((zero,value),-1),
                torch.cat((divergence,dz),-1),torch.cat((dz,divergence),-1))
    if not count:
        # Apply the same geometry/material/query admission even for no columns.
        empty,_h,receipt=_tensor_surface_potentials(basis,queries,kernel,sample,order=order,
            pair_batch=pair_batch,query_batch=query_batch,columns=1,active_faces=active)
        return empty[:,:,:0],empty[:,:,:0].clone(),receipt
    e,h,receipt=_tensor_surface_potentials(basis,queries,kernel,sample,order=order,
        pair_batch=pair_batch,query_batch=query_batch,columns=2*count,active_faces=active)
    columns=torch.cat((inverse,inverse+count))
    receipt.update(representation='tensor_selected_rwg_columns',selected_edges=len(edge_indices),
                   unique_edges=count,support_panels=len(active),physical_qualification=False)
    return e[...,columns],h[...,columns],receipt

def tensor_incident_potentials(handle, queries, kernel, *, order=6, pair_batch=16, query_batch=16, surface_faces=None):
    """Radiate the exact analytic incident sheet in the chosen tensor medium.

    This is a homogeneous-medium representation contribution. In an interior
    total-field decoder both scattered-current and incident-sheet potentials
    acquire the interior minus sign. No incident RWG projection is performed.
    """
    from .lifting import incident_trace
    if kernel.medium.frequency != handle.exterior.frequency:
        raise ValueError('Incident sheet and tensor kernel frequencies must agree')
    def sample(source, faces):
        return incident_trace(handle, source, faces)
    e, h, receipt = _tensor_surface_potentials(handle.basis, queries, kernel, sample,
        order=order, pair_batch=pair_batch, query_batch=query_batch, surface_faces=surface_faces)
    receipt.update(representation='tensor_analytic_incident_sheet_potentials',
                   incident_carrier_identity=handle.incident_carrier.identity)
    return e, h, receipt


def tensor_lifted_potentials(handle, coefficients, queries, kernel, *, order=6,
                             pair_batch=16, query_batch=16, surface_faces=None):
    """Integrate the combined scattered and analytic incident sheet once.

    Linearity permits summing source currents and divergences before applying
    each regular/static kernel. The principal-value subtraction uses the same
    combined sheet. Region signs and physical trace jumps remain caller-owned.
    """
    from .lifting import incident_trace
    if kernel.medium.frequency != handle.exterior.frequency:
        raise ValueError('Incident sheet and tensor kernel frequencies must agree')
    scattered = _rwg_current_sampler(handle.basis, coefficients, queries, kernel)

    def sample(source, faces):
        return tuple(a+b for a, b in zip(scattered(source, faces),
                                        incident_trace(handle, source, faces)))

    e, h, receipt = _tensor_surface_potentials(handle.basis, queries, kernel, sample,
        order=order, pair_batch=pair_batch, query_batch=query_batch,
        surface_faces=surface_faces)
    receipt.update(representation='tensor_combined_lifted_sheet_potentials',
                   incident_carrier_identity=handle.incident_carrier.identity,
                   shared_sheet_kernel=True)
    return e, h, receipt


def _tensor_surface_potentials(basis, queries, kernel, sample, *, order, pair_batch, query_batch,
                               surface_faces=None, columns=None, active_faces=None):
    if columns is not None and (type(columns) is not int or columns<1 or surface_faces is not None):
        raise ValueError('Tensor field columns require positive column count and off-surface queries')
    medium = kernel.medium
    medium.assert_current()
    template = medium.epsilon_r
    if (basis.mesh.device != template.device or basis.mesh.dtype != template.real.dtype
            or not isinstance(queries, torch.Tensor) or queries.ndim != 2 or queries.shape[1] != 3
            or queries.device != template.device or queries.dtype != template.real.dtype
            or not bool(torch.isfinite(queries).all())):
        raise ValueError('Tensor field queries, mesh and medium must share CUDA device and precision')
    for name, value in (('order', order), ('pair_batch', pair_batch), ('query_batch', query_batch)):
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if surface_faces is None:
        classify_queries(basis.mesh, queries)
    else:
        if (not isinstance(surface_faces, torch.Tensor) or surface_faces.dtype != torch.int64
                or surface_faces.device != queries.device or surface_faces.shape != queries.shape[:1]):
            raise ValueError('Principal-value faces must be matching CUDA int64 query labels')
        _, _, actual = classify_queries(basis.mesh, queries, allow_surface=True)
        if not torch.equal(actual, surface_faces):
            raise ValueError('Principal-value faces must match the actual query geometry')
    face_ids=torch.arange(len(basis.mesh.faces),device=basis.mesh.device) if active_faces is None else active_faces
    if active_faces is not None:
        if (surface_faces is not None or columns is None or not isinstance(active_faces,torch.Tensor)
                or active_faces.dtype!=torch.int64 or active_faces.device!=basis.mesh.device
                or active_faces.ndim!=1 or bool(((active_faces<0)|(active_faces>=len(basis.mesh.faces))).any())
                or (len(active_faces)>1 and not bool((active_faces[1:]>active_faces[:-1]).all()))):
            raise ValueError('Selected tensor support panels require sorted unique off-surface column indices')
    triangles = basis.mesh.triangles[face_ids]
    panel_radius = torch.linalg.vector_norm(triangles-triangles.mean(1)[:, None], dim=-1).amax(-1)
    chunks, receipts = [], []
    for start in range(0, len(queries), query_batch):
        query = queries[start:start+query_batch]
        count, nf = len(query), len(triangles)
        observer = torch.arange(count, device=query.device).repeat_interleave(nf)
        panel = torch.arange(nf, device=query.device).repeat(count)
        near = (_triangle_distance(query, triangles) < 2*panel_radius).flatten()
        values = template.new_zeros((count, 6) if columns is None else (count,6,columns))
        for use_fan in (False, True):
            selected = torch.nonzero(near == use_fan).flatten()
            if not len(selected):
                continue
            for batch in selected.split(pair_batch):
                oi, ti = observer[batch], panel[batch]
                if use_fan:
                    source, weights = duffy_fan(query[oi], triangles[ti], order)
                else:
                    source, weights = triangle_rule(triangles[ti], order)
                delta = query[oi, None]-source
                regular, receipt = _regular_kernel(kernel, delta)
                j, m, divj, divm = sample(source, face_ids[ti])
                currents = torch.cat((j, m), -1 if columns is None else -2)
                field = ((regular@currents[..., None]).squeeze(-1) if columns is None
                         else regular@currents)
                own = torch.zeros(len(ti), dtype=torch.bool, device=ti.device) if surface_faces is None else surface_faces[start+oi] == ti
                if bool(own.any()):
                    j0, m0, dj0, dm0 = sample(query[oi, None], face_ids[ti])
                    constant_cross = torch.cat(((regular[..., :3, 3:]@m0[..., None]).squeeze(-1),
                                                (regular[..., 3:, :3]@j0[..., None]).squeeze(-1)), -1)
                    field = field-torch.where(own[:, None, None], constant_cross, 0.)
                    divj = divj-torch.where(own[:, None], dj0, 0.)
                    divm = divm-torch.where(own[:, None], dm0, 0.)
                static = []
                for divergence, kind, vacuum in ((divj, 'electric', EPSILON_0), (divm, 'magnetic', MU_0)):
                    _, gradient, _ = electrostatic_kernel(delta.reshape(-1, 3), medium, constitutive=kind)
                    factor=1j/(medium.omega*vacuum)*gradient.reshape(*weights.shape,3)
                    static.append(factor*divergence[...,None] if columns is None
                                  else factor[...,None]*divergence[...,None,:])
                field = field+torch.cat(static, -1 if columns is None else -2)
                integrated = (field*(weights[...,None] if columns is None else weights[...,None,None])).sum(1)
                if bool(own.any()):
                    constant, annulus_receipt = _odd_face_integral(query[oi[own]], triangles[ti[own]], kernel,
                        j0[own], m0[own], dj0[own], dm0[own], order)
                    integrated = integrated.index_add(0, torch.nonzero(own).flatten(), constant)
                    receipt = {**receipt, 'principal_value_constant_term': annulus_receipt}
                values = values.index_add(0, oi, integrated)
                receipts.append({'panel_pairs': len(batch), 'projected_fan': use_fan, 'kernel': receipt})
        chunks.append(values)
    result = torch.cat(chunks) if chunks else template.new_empty((0,6) if columns is None else (0,6,columns))
    return result[:, :3], result[:, 3:], {
        'representation': 'tensor_radiated_current_potentials', 'surface_order': order,
        'static_source_derivative_integrated_by_parts': True,
        'surface_error_certified': False, 'trace_jump_applied': False,
        'principal_value': surface_faces is not None,
        'physical_qualification': False, 'batches': receipts}


def _regular_kernel(kernel, displacement):
    values, receipt = kernel.evaluate_regular(displacement.reshape(-1, 3))
    medium = kernel.medium
    if (not isinstance(receipt, dict) or receipt.get('converged') is not True
            or receipt.get('static_diagonal_singularity_subtracted') is not True
            or receipt.get('static_diagonal_restored') is not False):
        raise ValueError('Tensor potentials require a converged regular-only dyadic kernel')
    if (values.shape != (displacement.numel()//3, 6, 6) or values.device != medium.epsilon_r.device
            or values.dtype != medium.epsilon_r.dtype or not bool(torch.isfinite(values).all())):
        raise ValueError('Regular kernel returned incompatible tensor field data')
    return values.reshape(*displacement.shape[:-1], 6, 6), receipt


def _odd_face_integral(points, triangles, kernel, j, m, divj, divm, order):
    """PV of constant-current odd kernels: the centered disk cancels exactly."""
    source, weights, radius = triangle_outside_inscribed_disk(points, triangles, order)
    delta = points[:, None]-source
    regular, receipt = _regular_kernel(kernel, delta)
    field = torch.cat(((regular[..., :3, 3:]@m[..., None]).squeeze(-1),
                       (regular[..., 3:, :3]@j[..., None]).squeeze(-1)), -1)
    static = []
    for divergence, kind, vacuum in ((divj, 'electric', EPSILON_0), (divm, 'magnetic', MU_0)):
        _, gradient, _ = electrostatic_kernel(delta.reshape(-1, 3), kernel.medium, constitutive=kind)
        static.append(1j/(kernel.medium.omega*vacuum)*gradient.reshape(*weights.shape, 3)*divergence[..., None])
    value = ((field+torch.cat(static, -1))*weights[..., None]).sum(1)
    return value, {'method': 'odd_kernel_centered_disk_cancellation_v1', 'kernel': receipt,
                   'disk_radius_min_m': float(radius.min()), 'order': order}


class TensorFieldConvergenceError(RuntimeError):
    """A tensor field query exhausted its surface quadrature budget."""
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__('Tensor field surface quadrature did not converge within max_order')


def evaluate_tensor_current_potentials(basis, coefficients, queries, kernel, *,
                                       query_config=None, initial_order=6,
                                       pair_batch=16, query_batch=16):
    """Refine off-surface panel quadrature with separate E and H checks.

    Magnetic absolute tolerance is the electric tolerance divided by vacuum
    impedance. Every field and query must pass. Differences of successive
    orders are estimators, not bounds; the inner spectral estimator is recorded
    separately and cannot be certified by surface refinement.
    """
    def evaluate(order):
        return tensor_current_potentials(basis, coefficients, queries, kernel, order=order,
                                         pair_batch=pair_batch, query_batch=query_batch)
    return _adaptive_tensor_fields(evaluate, queries, query_config, initial_order)


def _adaptive_tensor_fields(evaluator, queries, query_config, initial_order):
    import math
    from .fields import FieldQueryConfig
    config = FieldQueryConfig() if query_config is None else query_config
    if not isinstance(config, FieldQueryConfig):
        raise TypeError('query_config must be FieldQueryConfig')
    if type(initial_order) is not int or initial_order < 1:
        raise ValueError('initial_order must be a positive integer')
    if config.adaptive and initial_order >= config.max_order:
        raise ValueError('max_order must exceed initial_order for surface error estimation')
    eta0 = math.sqrt(MU_0/EPSILON_0)
    receipt = {'representation': 'tensor_radiated_current_potentials',
        'adaptive': config.adaptive, 'converged': False, 'surface_error_certified': False,
        'estimator': 'successive_surface_order_difference_per_field_and_query',
        'relative_tolerance': config.relative_tolerance,
        'absolute_tolerance_E_V_per_m': config.absolute_tolerance,
        'absolute_tolerance_H_A_per_m': config.absolute_tolerance/eta0,
        'physical_qualification': False, 'checks': [], 'evaluations': []}

    def evaluate(order):
        e, h, details = evaluator(order)
        receipt['evaluations'].append(details)
        receipt['final_order'] = order
        return e, h

    order = initial_order
    electric, magnetic = evaluate(order)
    if not config.adaptive or not len(queries):
        receipt.update(converged=not len(queries),
                       failure_reason=None if not len(queries) else 'fixed_order_error_unestimated')
        return electric, magnetic, receipt
    previous = (electric.detach(), magnetic.detach())
    while order < config.max_order:
        order = min(2*order, config.max_order)
        electric, magnetic = evaluate(order)
        check, accepted = {'order': order}, True
        for name, field, old, scale in zip(('E', 'H'), (electric, magnetic), previous, (1., eta0)):
            error = torch.linalg.vector_norm(field.detach()-old, dim=-1)
            norm = torch.linalg.vector_norm(field.detach(), dim=-1)
            threshold = config.absolute_tolerance/scale+config.relative_tolerance*norm
            passed = (torch.isfinite(error) & torch.isfinite(norm)
                      & torch.isfinite(threshold) & (error <= threshold))
            check[name] = {'max_absolute_difference': float(error.max()),
                'max_relative_difference': float((error/norm.clamp_min(torch.finfo(norm.dtype).tiny)).max()),
                'failed_query_count': int((~passed).sum())}
            accepted = accepted and bool(passed.all())
        receipt['checks'].append(check)
        if accepted:
            receipt.update(converged=True, failure_reason=None)
            return electric, magnetic, receipt
        previous = (electric.detach(), magnetic.detach())
    receipt['failure_reason'] = 'surface_quadrature_order_budget_exhausted'
    raise TensorFieldConvergenceError(receipt)


def tensor_sheet_jump(normals, electric_current, magnetic_current,
                      electric_divergence, magnetic_divergence, medium):
    """Return full positive-minus-negative field jumps of a homogeneous sheet.

    n x [H]=J, n x [E]=-M and n.[epsilon E]=div_s J/(i omega),
    n.[mu H]=div_s M/(i omega). The normal correction includes the tensor's
    tangential-normal coupling; multiplying a scalar jump by inverse tensors
    would violate the tangential current conditions. No principal value is
    evaluated by this routine.
    """
    medium.assert_current()
    tensor = medium.epsilon_r
    if (not isinstance(normals, torch.Tensor) or normals.ndim != 2 or normals.shape[1] != 3
            or normals.device != tensor.device or normals.dtype != tensor.real.dtype
            or not bool(torch.isfinite(normals).all())):
        raise ValueError('Sheet normals must be matching finite real CUDA [N,3] tensors')
    eps = torch.finfo(normals.dtype).eps
    if not bool(((torch.linalg.vector_norm(normals, dim=-1)-1).abs() <= 64*eps).all()):
        raise ValueError('Sheet normals must be unit vectors')
    normal = normals.to(tensor.dtype)
    for current in (electric_current, magnetic_current):
        if (not isinstance(current, torch.Tensor) or current.shape != normals.shape
                or current.device != tensor.device or current.dtype != tensor.dtype
                or not bool(torch.isfinite(current).all())):
            raise ValueError('Sheet currents must be matching finite complex CUDA [N,3] tensors')
        norm = torch.linalg.vector_norm(current, dim=-1)
        if not bool(((normal*current).sum(-1).abs() <= 128*eps*norm).all()):
            raise ValueError('Sheet currents must be tangential')
    for divergence in (electric_divergence, magnetic_divergence):
        if (not isinstance(divergence, torch.Tensor) or divergence.shape != normals.shape[:1]
                or divergence.device != tensor.device or divergence.dtype != tensor.dtype
                or not bool(torch.isfinite(divergence).all())):
            raise ValueError('Sheet divergences must be matching finite complex CUDA [N] tensors')
    tangents = (torch.linalg.cross(normal, magnetic_current), -torch.linalg.cross(normal, electric_current))
    output = []
    for tangent, divergence, constitutive in zip(tangents,
            (electric_divergence, magnetic_divergence), (medium.epsilon, medium.mu)):
        normal_constitutive = normal@constitutive
        denominator = (normal_constitutive*normal).sum(-1)
        if bool((denominator == 0).any()):
            raise ValueError('Characteristic sheet normal requires a separate tensor trace limit')
        charge = divergence/(1j*medium.omega)
        correction = (charge-(normal_constitutive*tangent).sum(-1))/denominator
        jump = tangent+normal*correction[:, None]
        if not bool(torch.isfinite(jump).all()):
            raise ValueError('Tensor sheet trace limit is not finite')
        output.append(jump)
    return tuple(output)


def tensor_rwg_trace_jump(basis, coefficients, queries, faces, medium):
    """Evaluate RWG sheet jumps at explicitly identified triangle-interior points."""
    _validate_arrays(SimpleNamespace(basis=basis), coefficients, queries)
    if (not isinstance(faces, torch.Tensor) or faces.dtype != torch.int64
            or faces.device != queries.device or faces.shape != queries.shape[:1]
            or bool(((faces < 0) | (faces >= len(basis.mesh.faces))).any())):
        raise ValueError('Trace faces must identify a valid triangle for every query')
    _, _, actual_faces = classify_queries(basis.mesh, queries, allow_surface=True)
    if not torch.equal(faces, actual_faces):
        raise ValueError('Trace queries must lie inside their declared surface triangle')
    edges, free, slopes = local_geometry(basis)
    local_j = coefficients[:basis.n_edges][edges[faces]]*slopes[faces]
    local_m = coefficients[basis.n_edges:][edges[faces]]*slopes[faces]
    displacement = (queries[:, None]-free[faces]).to(coefficients.dtype)
    j = torch.einsum('ql,qlc->qc', local_j, displacement)
    m = torch.einsum('ql,qlc->qc', local_m, displacement)
    return tensor_sheet_jump(basis.mesh.normals[faces], j, m,
                             2*local_j.sum(-1), 2*local_m.sum(-1), medium)


def decode_tensor_lifted(handle, coefficients, queries, trace_side=None, *, query_config=None):
    """Compose complete tensor interior fields in the analytic-lifted representation."""
    import copy
    from .tensor_medium import TensorMedium, scalar_medium_if_isotropic
    from .dyadic_green import DyadicGreenKernel
    from .fields import _query_context
    scalar = scalar_medium_if_isotropic(handle.interior)
    if scalar is not None and not isinstance(handle.exterior, TensorMedium):
        from .lifting_fields import decode_lifted
        local = copy.copy(handle)
        local.interior = scalar
        result = decode_lifted(local, coefficients, queries, 1, 'total', trace_side, query_config=query_config)
        handle.last_field_query_log = local.last_field_query_log
        return result
    _, config, faces, on_surface = _query_context(handle, coefficients, queries, 1, 'total', trace_side, query_config)
    surface_faces = faces if bool(on_surface.any()) else None
    if surface_faces is not None:
        from .lifting import incident_trace
        raw_jump = tensor_rwg_trace_jump(handle.basis, coefficients, queries[on_surface], faces[on_surface], handle.interior)
        j, m, divj, divm = incident_trace(handle, queries[on_surface, None], faces[on_surface])
        carrier_jump = tensor_sheet_jump(handle.basis.mesh.normals[faces[on_surface]], j[:, 0], m[:, 0],
                                         divj[:, 0], divm[:, 0], handle.interior)
    kernel = getattr(handle, 'tensor_kernel', None)
    if kernel is None:
        kernel = DyadicGreenKernel(handle.interior)
        handle.tensor_kernel = kernel
    if kernel.medium is not handle.interior:
        raise ValueError('Tensor decoder kernel must bind the exact interior sample')
    def evaluate(order):
        electric, magnetic, sheet_receipt = tensor_lifted_potentials(
            handle, coefficients, queries, kernel, order=order,
            pair_batch=handle.integration.pair_batch, surface_faces=surface_faces)
        fields = [-electric, -magnetic]
        if surface_faces is not None:
            for index in range(2):
                fields[index] = fields[index].clone()
                fields[index][on_surface] += .5*(raw_jump[index]+carrier_jump[index])
        return *fields, {'surface_order': order, 'combined_sheet': sheet_receipt}
    try:
        e, h, receipt = _adaptive_tensor_fields(evaluate, queries, config,
            max(handle.integration.order, handle.integration.singular_order))
    except TensorFieldConvergenceError as error:
        handle.last_field_query_log = error.receipt
        raise
    receipt.update(representation='tensor_complete_lifted_interior_field',
                   incident_carrier_identity=handle.incident_carrier.identity,
                   trace_side=trace_side, principal_value=surface_faces is not None)
    handle.last_field_query_log = receipt
    return e, h
