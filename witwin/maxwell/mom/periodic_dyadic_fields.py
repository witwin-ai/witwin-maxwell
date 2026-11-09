"""Physical RWG current convolution with nonprimary periodic tensor images."""
import torch
from .operator import local_geometry
from .quadrature import triangle_rule
from .periodic_operator import validate_periodic_cell
from .periodic_dyadic import REGULAR_DYADIC_KERNEL_TYPES


def regular_dyadic_surface_fields(system, coefficients, points, *, reference,
                                 order=4, face_batch=32, query_batch=8):
    """Return only nonprimary exterior E/H, retaining native current gradients.

    No wrapping is applied: primary exclusion refers to the actual source cell.
    Primary fields, incident carriers and interior recovery remain separate.
    Surface/image refinement is required before physical field qualification.
    """
    if not isinstance(reference, REGULAR_DYADIC_KERNEL_TYPES):
        raise TypeError('Tensor image fields require a supported regular dyadic kernel')
    for value in (order, face_batch, query_batch):
        if type(value) is not int or value < 1:
            raise ValueError('Surface order and batch sizes must be positive integers')
    validate_periodic_cell(system, reference.lattice)
    reference._validate()
    medium = reference.medium
    if any(body.interior.frequency != medium.frequency for body in system.bodies):
        raise ValueError('Tensor image fields and every body must share one frequency')
    if (not isinstance(coefficients, torch.Tensor) or coefficients.shape != (2*system.total_edges,)
            or coefficients.device != system.device or coefficients.dtype != medium.epsilon_r.dtype
            or not bool(torch.isfinite(coefficients).all())):
        raise ValueError('Currents must be finite matching complex CUDA all-J/all-M coefficients')
    if (not isinstance(points, torch.Tensor) or points.ndim != 2 or points.shape[1] != 3
            or points.device != system.device or points.dtype != system.dtype
            or not bool(torch.isfinite(points).all())):
        raise ValueError('Queries must be finite matching real CUDA [N,3] points')
    chunks, records = [], []
    for begin in range(0, len(points), query_batch):
        query = points[begin:begin+query_batch]
        fields = coefficients.new_zeros((len(query), 6))
        for index, body in enumerate(system.bodies):
            basis = body.basis
            edges, free, slopes = local_geometry(basis)
            j = coefficients[system.J_slices[index]][edges]*slopes
            m = coefficients[system.M_slices[index]][edges]*slopes
            for start in range(0, len(body.mesh.faces), face_batch):
                end = min(start+face_batch, len(body.mesh.faces))
                source, weights = triangle_rule(body.mesh.triangles[start:end], order)
                offsets = (source[:, :, None]-free[start:end, None]).to(coefficients.dtype)
                current = torch.cat((torch.einsum('fl,fqlc->fqc', j[start:end], offsets),
                                     torch.einsum('fl,fqlc->fqc', m[start:end], offsets)), dim=-1)
                difference = query[:, None, None]-source
                green, receipt = reference.evaluate(difference.reshape(-1, 3), exclude_primary=True)
                if receipt.get('primary_image_excluded') is not True:
                    raise ValueError('Tensor image fields cannot include the singular primary')
                green = green.reshape(len(query), end-start, source.shape[1], 6, 6)
                fields = fields+torch.einsum('bfqij,fqj,fq->bi', green, current, weights.to(green.dtype))
                records.append({'source_region': body.region_id, 'query_start': begin,
                                'face_start': start, 'images': receipt})
        chunks.append(fields)
    values = torch.cat(chunks) if chunks else coefficients.new_empty((0, 6))
    return (values[:, :3], values[:, 3:]), {
        'quantity': 'nonprimary_exterior_tensor_surface_fields',
        'primary_included': False, 'interiors_included': False, 'incident_included': False,
        'surface_order': order, 'image_evaluations': records,
        'surface_quadrature_converged': False, 'image_cutoff_converged': False,
        'complete_physical_fields': False, 'physical_qualification': False}


def exterior_dyadic_surface_fields(system, coefficients, points, *, reference,
                                   primary_kernel=None, order=6, face_batch=16, query_batch=8, trace_side=None):
    """Exterior scattered-current fields, including the free-space primary.

    Queries are wrapped into the source cell and multiplied by the Bloch phase.
    Interior queries reject. Face-interior queries require an explicit exterior
    trace and include the primary principal value plus the exterior half jump.
    """
    import math
    from .dyadic_green import DyadicGreenKernel
    from .anisotropic_fields import tensor_current_potentials, tensor_rwg_trace_jump
    from .fields import classify_queries
    if not isinstance(reference, REGULAR_DYADIC_KERNEL_TYPES):
        raise TypeError('Tensor exterior fields require a supported regular dyadic kernel')
    validate_periodic_cell(system, reference.lattice)
    reference._validate()
    if (not isinstance(points, torch.Tensor) or points.ndim != 2 or points.shape[1] != 3
            or points.device != system.device or points.dtype != system.dtype
            or not bool(torch.isfinite(points).all())):
        raise ValueError('Queries must be finite matching real CUDA [N,3] points')
    primary_kernel = DyadicGreenKernel(reference.medium) if primary_kernel is None else primary_kernel
    if primary_kernel.medium is not reference.medium:
        raise ValueError('Primary and periodic image fields must bind the exact same medium')
    shift = torch.floor(points@reference.lattice.reciprocal.T/(2*math.pi)+0.5)
    translation = shift@reference.lattice.vectors
    wrapped = points-translation
    if trace_side not in (None, 'exterior'):
        raise ValueError('Exterior tensor trace_side must be exterior or None')
    face_labels = []
    on_any_surface = torch.zeros(len(points), device=points.device, dtype=torch.bool)
    for body in system.bodies:
        inside, _, faces = classify_queries(body.mesh, wrapped, allow_surface=True)
        on_surface = faces >= 0
        if bool((inside & ~on_surface).any()):
            raise ValueError('Exterior tensor fields cannot recover body-interior queries')
        face_labels.append(faces)
        on_any_surface |= on_surface
    if trace_side is None and bool(on_any_surface.any()):
        raise ValueError('Surface tensor queries require explicit trace_side')
    if trace_side is not None and not bool(on_any_surface.all()):
        raise ValueError('Exterior trace_side requires every query to lie on a surface')
    (electric, magnetic), image_receipt = regular_dyadic_surface_fields(system, coefficients,
        wrapped, reference=reference, order=order, face_batch=face_batch, query_batch=query_batch)
    records = []
    for index, body in enumerate(system.bodies):
        local = torch.cat((coefficients[system.J_slices[index]], coefficients[system.M_slices[index]]))
        faces = face_labels[index]
        mask = faces >= 0
        surface_faces = faces if bool(mask.any()) else None
        e, h, receipt = tensor_current_potentials(body.basis, local, wrapped, primary_kernel,
            order=order, pair_batch=face_batch, query_batch=query_batch, surface_faces=surface_faces)
        if surface_faces is not None:
            je, jh = tensor_rwg_trace_jump(body.basis, local, wrapped[mask], faces[mask], reference.medium)
            ids = torch.nonzero(mask).flatten()
            e, h = e.index_add(0, ids, .5*je), h.index_add(0, ids, .5*jh)
        electric, magnetic = electric+e, magnetic+h
        records.append({'source_region': body.region_id, 'primary': receipt})
    phase = torch.exp(1j*(translation@reference.lattice.bloch))[:, None]
    return (electric*phase, magnetic*phase), {
        'quantity': 'periodic_exterior_scattered_current_fields', 'trace_side': trace_side,
        'primary_included': True, 'interiors_included': False, 'incident_included': False,
        'primary_evaluations': records, 'nonprimary': image_receipt,
        'surface_quadrature_converged': False, 'image_cutoff_converged': False,
        'physical_qualification': False}


def interior_dyadic_surface_fields(system, coefficients, points, *, carrier, reference,
                                   region_id, trace_side=None, config=None, query_config=None,
                                   interior_kernel=None):
    """Recover one body's total interior field in the analytic lifted basis.

    Only that body's scattered currents and analytic incident sheet enter its
    free-space interior representation. Periodicity enters through the solved
    global currents and the translated-cell Bloch phase, never interior images.
    """
    import math
    from types import SimpleNamespace
    from .operator import IntegrationConfig
    from .tensor_carrier import TensorIncidentCarrier
    from .tensor_medium import TensorMedium
    from .anisotropic_fields import decode_tensor_lifted
    if not isinstance(reference, REGULAR_DYADIC_KERNEL_TYPES) or not isinstance(carrier, TensorIncidentCarrier):
        raise TypeError('Tensor interior recovery requires a tensor carrier and regular dyadic kernel')
    validate_periodic_cell(system, reference.lattice)
    reference._validate()
    carrier.validate_medium(reference.medium)
    carrier.validate_lattice(reference.lattice)
    config = IntegrationConfig() if config is None else config
    if not isinstance(config, IntegrationConfig):
        raise TypeError('Tensor interior recovery requires IntegrationConfig')
    region = system.region(region_id)
    if region == 0:
        raise ValueError('Tensor interior recovery requires a body region')
    if (not isinstance(coefficients, torch.Tensor) or coefficients.shape != (2*system.total_edges,)
            or coefficients.device != system.device or coefficients.dtype != reference.medium.epsilon_r.dtype
            or not bool(torch.isfinite(coefficients).all())):
        raise ValueError('Currents must be finite matching complex CUDA all-J/all-M coefficients')
    if (not isinstance(points, torch.Tensor) or points.ndim != 2 or points.shape[1] != 3
            or points.device != system.device or points.dtype != system.dtype
            or not bool(torch.isfinite(points).all())):
        raise ValueError('Queries must be finite matching real CUDA [N,3] points')
    body = system.bodies[region-1]
    medium = body.interior
    if medium.frequency != reference.medium.frequency:
        raise ValueError('Interior and incident field frequencies must agree')
    if not isinstance(medium, TensorMedium):
        eye = torch.eye(3, device=coefficients.device, dtype=coefficients.dtype)
        medium = TensorMedium(medium.frequency, medium.epsilon_r*eye, medium.mu_r*eye)
    handle = SimpleNamespace(basis=body.basis, exterior=reference.medium, interior=medium,
                             incident_carrier=carrier, integration=config)
    if interior_kernel is not None:
        if interior_kernel.medium is not medium:
            raise ValueError('Interior field kernel must bind the exact body medium')
        handle.tensor_kernel = interior_kernel
    local = torch.cat((coefficients[system.J_slices[region-1]], coefficients[system.M_slices[region-1]]))
    translation = torch.floor(points@reference.lattice.reciprocal.T/(2*math.pi)+.5)@reference.lattice.vectors
    electric, magnetic = decode_tensor_lifted(handle, local, points-translation,
        trace_side, query_config=query_config)
    phase = torch.exp(1j*(translation@reference.lattice.bloch))[:, None]
    return (electric*phase, magnetic*phase), {
        'quantity': 'periodic_tensor_lifted_interior_total_fields', 'region_id': region,
        'interior_images_added': False, 'cross_body_interior_terms': False,
        'local_recovery': handle.last_field_query_log, 'physical_qualification': False}


def decode_periodic_dyadic(handle, coefficients, points, region_id, field_kind,
                           trace_side=None, *, query_config=None):
    """Recover complete fields with refinement of the requested E/H quantity."""
    import math
    from dataclasses import replace
    from .fields import FieldQueryConfig
    from .lifting_fields import classify_system_queries
    from .anisotropic_fields import _adaptive_tensor_fields, TensorFieldConvergenceError
    from .dyadic_green import DyadicConvergenceError
    from .periodic_dyadic import PeriodicDyadicConvergenceError
    handle.validate_tensor_state()
    config = handle._public_config.query_config if query_config is None else query_config
    if not isinstance(config, FieldQueryConfig):
        raise TypeError('Periodic dyadic queries require FieldQueryConfig')
    if (not isinstance(points, torch.Tensor) or points.ndim != 2 or points.shape[1] != 3
            or points.device != handle.system.device or points.dtype != handle.system.dtype
            or not bool(torch.isfinite(points).all())):
        raise ValueError('Periodic dyadic queries require finite matching real CUDA [N,3] points')
    region = handle.system.region(region_id)
    if field_kind not in ('total', 'scattered', 'incident') or (region and field_kind != 'total'):
        raise ValueError("Interior queries require field_kind='total'; exterior supports total/scattered/incident")
    if (trace_side not in (None, 'interior', 'exterior')
            or (trace_side is not None and (trace_side == 'interior') != bool(region))):
        raise ValueError('trace_side must match region_id')
    reference = handle.periodic_kernel
    translation = torch.floor(points@reference.lattice.reciprocal.T/(2*math.pi)+.5)@reference.lattice.vectors
    regions, owners, _ = classify_system_queries(handle, coefficients, points-translation)
    on = owners != 0
    if bool(on.any()) and trace_side is None:
        raise ValueError('MoM on-surface queries require explicit trace_side')
    if trace_side is not None and not bool(on.all()):
        raise ValueError('trace_side is only defined for on-surface queries')
    if bool(((~on & (regions != region)) | (on & (owners != region) & bool(region))).any()):
        raise ValueError('MoM query geometry does not match the requested region_id')
    handle.last_field_query_log = None
    if field_kind == 'incident':
        values = handle.incident_carrier.evaluate(points, handle.exterior)
        handle.last_field_query_log = {'representation': 'analytic_tensor_incident_modes',
            'converged': True, 'physical_qualification': False,
            'incident_carrier_identity': handle.incident_carrier.identity}
        handle.metadata['last_field_query'] = handle.last_field_query_log
        return values

    def evaluate(order):
        if region:
            integration = replace(handle.integration, order=order, singular_order=order,
                near_max_order=max(handle.integration.near_max_order, order+2))
            values, details = interior_dyadic_surface_fields(handle.system, coefficients, points,
                carrier=handle.incident_carrier, reference=reference, region_id=region,
                trace_side=trace_side, config=integration, query_config=replace(config, adaptive=False),
                interior_kernel=handle.body_handles[region-1].tensor_kernel)
        else:
            values, details = exterior_dyadic_surface_fields(handle.system, coefficients, points,
                reference=reference, primary_kernel=handle.primary_kernel, order=order,
                face_batch=handle.integration.pair_batch, trace_side=trace_side)
            if field_kind == 'total':
                incident = handle.incident_carrier.evaluate(points, handle.exterior)
                values = tuple(value+incoming for value, incoming in zip(values, incident))
        return *values, details

    try:
        electric, magnetic, receipt = _adaptive_tensor_fields(evaluate, points, config,
            max(handle.integration.order, handle.integration.singular_order))
    except TensorFieldConvergenceError as error:
        handle.last_field_query_log = error.receipt
        raise
    except (DyadicConvergenceError, PeriodicDyadicConvergenceError) as error:
        import copy
        handle.last_field_query_log = {
            'representation': 'periodic_dyadic_analytic_lifted_fields_v1',
            'region_id': region, 'field_kind': field_kind, 'trace_side': trace_side,
            'converged': False, 'physical_qualification': False,
            'failure_reason': 'component_quadrature_or_spectral_budget_failed',
            'component_failure': copy.deepcopy(error.receipt)}
        raise
    finally:
        if handle.last_field_query_log is not None:
            handle.metadata['last_field_query'] = handle.last_field_query_log
    receipt.update(representation='periodic_dyadic_analytic_lifted_fields_v1',
        region_id=region, field_kind=field_kind, trace_side=trace_side,
        incident_carrier_identity=handle.incident_carrier.identity,
        physical_qualification=False)
    handle.last_field_query_log = receipt
    handle.metadata['last_field_query'] = receipt
    return electric, magnetic
