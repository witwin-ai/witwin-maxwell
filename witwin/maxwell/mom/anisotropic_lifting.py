"""Analytic incident-carrier testing for homogeneous tensor interior regions.

The general spatial difference uses contrast-factored outgoing dynamic/static
spectra and independently refined transverse integration. Surface and physical
qualification remain separate. Exact scalar specialization retains its original
factored analytic kernel.
"""
from __future__ import annotations

import time
from types import SimpleNamespace
import torch
from ..constants import EPSILON_0, MU_0, ETA_0
from .dyadic_green import DyadicGreenKernel
from .dyadic_contrast import DyadicContrastKernel
from .tensor_medium import TensorMedium, scalar_medium_if_isotropic
from .lifting import incident_trace, build_lifted_rhs, FORMULATION_ID, CURRENT_KIND
from .operator import local_geometry
from .quadrature import classify_all_pairs, touching_pair_rule, triangle_rule


def _pair_rhs(handle, rows, cols, category, order, contrast_kernel):
    basis = handle.basis
    triangles = basis.mesh.triangles
    _, free, slopes = local_geometry(basis)
    if category < 3:
        obs, src, weight = touching_pair_rule(triangles[rows], triangles[cols],
            basis.mesh.faces[rows], basis.mesh.faces[cols], category, order)
    else:
        obs_nodes, obs_weights = triangle_rule(triangles[rows], order)
        src_nodes, src_weights = triangle_rule(triangles[cols], order)
        obs = obs_nodes[:, :, None].expand(-1, -1, src_nodes.shape[1], -1).flatten(1, 2)
        src = src_nodes[:, None].expand(-1, obs_nodes.shape[1], -1, -1).flatten(1, 2)
        weight = (obs_weights[:, :, None]*src_weights[:, None]).flatten(1, 2)
    displacement = (obs-src).reshape(-1, 3)
    value, receipt = contrast_kernel.evaluate_regular(displacement)
    if (not isinstance(receipt, dict) or receipt.get('converged') is not True
            or receipt.get('static_diagonal_singularity_subtracted') is not True
            or receipt.get('static_diagonal_restored') is not False):
        raise ValueError('Tensor carrier lifting requires a converged regular-only contrast kernel')
    if (value.shape != (len(displacement), 6, 6) or value.device != basis.mesh.device
            or value.dtype != handle.interior.epsilon_r.dtype or not bool(torch.isfinite(value).all())):
        raise ValueError('Tensor carrier contrast kernel returned incompatible data')
    records = [receipt]
    difference = value.reshape(*weight.shape, 6, 6)
    j, m, divj, divm = incident_trace(handle, src, cols)
    currents = torch.cat((j, m), -1)
    field = (difference@currents[..., None]).squeeze(-1)
    test = (slopes[rows, None, :, None]*(obs[:, :, None]-free[rows, None])).to(field.dtype)
    result = torch.stack((torch.einsum('bqli,bqi,bq->bl', test, field[..., 3:], weight.to(field.dtype)),
                         -torch.einsum('bqli,bqi,bq->bl', test, field[..., :3], weight.to(field.dtype))), 1)
    for row, kind, div, coefficient in ((0, 'magnetic', divm, -1j/(handle.exterior.omega*MU_0)),
                                         (1, 'electric', divj, 1j/(handle.exterior.omega*EPSILON_0))):
        potential, _, _ = contrast_kernel.evaluate_static(displacement, constitutive=kind)
        if (potential.shape != (len(displacement),) or potential.device != basis.mesh.device
                or potential.dtype != value.dtype or not bool(torch.isfinite(potential).all())):
            raise ValueError('Tensor carrier static contrast returned incompatible data')
        integral = (potential.reshape(weight.shape)*div*weight).sum(-1)
        result[:, row] += coefficient*2*slopes[rows]*integral[:, None]
    return -result, {'category': category, 'order': order, 'kernels': records,
                     'pair_count': len(rows), 'spatial_difference': contrast_kernel.version}


def build_tensor_lifted_rhs(handle, *, interior_kernel=None, contrast_kernel=None, test_edges=None):
    """Integrate -(A_int-A_ext)c_inc with analytic currents and divergences.

    No carrier RWG projection, boundary RHS halving, or total-current relabeling
    occurs. The general tensor candidate remains physically unqualified until
    its spatial contrast, singular integration, and reference gates pass.
    """
    started = time.perf_counter()
    basis, medium, config = handle.basis, handle.interior, handle.integration
    selection=None
    if test_edges is not None:
        from .nearfield import _selected_edges
        selection=_selected_edges(basis,test_edges)
    if not isinstance(medium, TensorMedium):
        raise TypeError('Tensor carrier lifting requires TensorMedium interior')
    medium.assert_current()
    if (medium.frequency != handle.exterior.frequency or medium.epsilon_r.device != basis.mesh.device
            or medium.epsilon_r.real.dtype != basis.mesh.dtype):
        raise ValueError('Tensor carrier medium, mesh and exterior frequency must agree')
    if medium.epsilon_r.requires_grad or medium.mu_r.requires_grad:
        raise NotImplementedError('Tensor carrier assembly currently requires fixed material samples')
    if interior_kernel is not None and interior_kernel.medium is not medium:
        raise ValueError('Tensor carrier kernel must bind the exact interior sample')
    tensor_exterior = isinstance(handle.exterior, TensorMedium)
    handle.incident_carrier.validate_medium(handle.exterior)
    if tensor_exterior and (handle.exterior.epsilon_r.requires_grad or handle.exterior.mu_r.requires_grad):
        raise NotImplementedError('Tensor carrier assembly currently requires fixed material samples')
    scalar = scalar_medium_if_isotropic(medium)
    if scalar is not None and not tensor_exterior:
        scalar_handle = SimpleNamespace(basis=basis, exterior=handle.exterior, interior=scalar,
            integration=config, incident_carrier=handle.incident_carrier)
        if selection is None:rhs, receipt = build_lifted_rhs(scalar_handle)
        else:
            from .lifting import build_selected_lifted_rhs
            rhs,receipt=build_selected_lifted_rhs(scalar_handle,test_edges)
        receipt.update(tensor_specialization='exact_isotropic_factored_difference', physical_qualification=False)
        return rhs, receipt
    if contrast_kernel is None:
        inside = DyadicGreenKernel(medium) if interior_kernel is None else interior_kernel
        if not isinstance(inside, DyadicGreenKernel):
            raise TypeError('An injected tensor oracle must provide an explicit paired contrast kernel')
        if not hasattr(handle, 'tensor_contrast_kernel'):
            eye = torch.eye(3, device=basis.mesh.device, dtype=medium.epsilon_r.dtype)
            exterior_sample = handle.exterior if tensor_exterior else TensorMedium(medium.frequency, handle.exterior.epsilon_r*eye, handle.exterior.mu_r*eye)
            budget = getattr(handle, 'tensor_rhs_cache_max_bytes', 0)
            handle.tensor_contrast_kernel = DyadicContrastKernel(exterior_sample, medium, inside.integration, budget)
        contrast_kernel = handle.tensor_contrast_kernel
    if contrast_kernel.interior is not medium:
        raise ValueError('Tensor carrier contrast kernel must bind the exact interior sample')
    contrast_kernel.exterior.assert_current()
    if tensor_exterior:
        if contrast_kernel.exterior is not handle.exterior:
            raise ValueError('Tensor carrier contrast must bind the exact exterior sample')
    else:
        eye = torch.eye(3, device=basis.mesh.device, dtype=medium.epsilon_r.dtype)
        if (contrast_kernel.exterior.frequency != handle.exterior.frequency
                or not torch.equal(contrast_kernel.exterior.epsilon_r, handle.exterior.epsilon_r*eye)
                or not torch.equal(contrast_kernel.exterior.mu_r, handle.exterior.mu_r*eye)):
            raise ValueError('Tensor carrier contrast exterior does not match its incident carrier')
    if (torch.equal(contrast_kernel.exterior.epsilon_r, medium.epsilon_r)
            and torch.equal(contrast_kernel.exterior.mu_r, medium.mu_r)):
        zero = medium.epsilon_r.new_zeros(2*(basis.n_edges if selection is None else len(test_edges)))
        if tensor_exterior:
            e, h = handle.incident_carrier.evaluate(basis.mesh.centroids[:1], handle.exterior)
            zero = zero+0*(e.sum()+h.sum())
        return zero, {
            'formulation_id': FORMULATION_ID, 'current_kind': CURRENT_KIND,
            'incident_carrier_identity': handle.incident_carrier.identity,
            'tensor_specialization': 'exact_zero_constitutive_contrast',
            'physical_qualification': False, 'rhs_integration': vars(config)}
    ne, nf = basis.n_edges, len(basis.mesh.faces)
    rhs = medium.epsilon_r.new_zeros((2,ne if selection is None else len(test_edges)))
    records, counts = [], [0]*5
    classification={'maximum_pairs':0}
    def batches():
        if selection is None:
            classes=classify_all_pairs(basis.mesh)
            classification['maximum_pairs']=len(classes)
            for category in range(5):
                pairs=(classes==category).nonzero().flatten()
                counts[category]+=len(pairs)
                batch=config.far_pair_batch if category==4 else config.pair_batch
                for ids in pairs.split(batch):
                    if len(ids):yield category,ids//nf,ids%nf
        else:
            from .quadrature import classify_pairs
            faces=selection[2];batch=max(config.pair_batch,config.far_pair_batch)
            for start in range(0,len(faces)*nf,batch):
                ids=torch.arange(start,min(start+batch,len(faces)*nf),device=basis.mesh.device)
                rows,cols=faces[ids//nf],ids%nf
                classes=classify_pairs(basis.mesh,rows,cols)
                classification['maximum_pairs']=max(classification['maximum_pairs'],len(ids))
                for category in range(5):
                    selected=(classes==category).nonzero().flatten()
                    counts[category]+=len(selected)
                    size=config.far_pair_batch if category==4 else config.pair_batch
                    for subset in selected.split(size):
                        if len(subset):yield category,rows[subset],cols[subset]
    scale = rhs.real.new_tensor([(ETA_0 if tensor_exterior else abs(handle.exterior.eta)), 1.])[None, :, None]
    for category,rows,cols in batches():
        order = config.order if category == 4 else config.singular_order
        local, receipt = _pair_rhs(handle, rows, cols, category, order, contrast_kernel)
        if category == 3:
            checks, accepted = [], False
            while order < config.near_max_order:
                order = min(2*order, config.near_max_order)
                updated, receipt = _pair_rhs(handle, rows, cols, category, order, contrast_kernel)
                norm = torch.linalg.vector_norm((updated*scale).flatten(1), dim=-1)
                error = torch.linalg.vector_norm(((updated-local)*scale).flatten(1), dim=-1)
                relative = error/norm.clamp_min(torch.finfo(norm.dtype).tiny)
                checks.append({'order': order, 'maximum_relative_difference': float(relative.max())})
                local = updated
                if bool((relative <= config.near_tolerance).all()):
                    accepted = True
                    break
            if not accepted:
                raise RuntimeError(f'Tensor carrier near-panel integration did not converge: {checks}')
            receipt['near_checks'] = checks
        records.append(receipt)
        if selection is None:
            indices=basis.face_edges[rows].flatten()
            for block in range(2):rhs[block].index_add_(0,indices,local[:,block].flatten())
        else:
            from .nearfield import _local_indices
            edges,positions,_=selection
            indices,keep=_local_indices(basis.face_edges[rows],edges,positions)
            for block in range(2):rhs[block].index_add_(0,indices[keep],local[:,block][keep])
    torch.cuda.synchronize(rhs.device)
    return rhs.flatten(), {'formulation_id': FORMULATION_ID, 'current_kind': CURRENT_KIND,
        'incident_carrier_identity': handle.incident_carrier.identity,
        'difference_kernel_version': 'tensor_factored_spatial_contrast_candidate_v2',
        'weak_contrast_accuracy_certified': False, 'global_rhs_error_certified': False,
        'physical_qualification': False, 'rhs_integration': vars(config),
        'rhs_triangle_pair_counts': counts, 'rhs_batch_receipts': records,
        'selected_edges':None if selection is None else len(test_edges),
        'maximum_classification_pairs':classification['maximum_pairs'],
        'global_pair_classification_retained':selection is None,
        'rhs_spectral_cache': contrast_kernel.mode_cache.stats() if isinstance(contrast_kernel, DyadicContrastKernel) else None,
        'rhs_spectral_cache_max_bytes': contrast_kernel.cache_max_bytes if isinstance(contrast_kernel, DyadicContrastKernel) else None,
        'rhs_seconds': time.perf_counter()-started}
