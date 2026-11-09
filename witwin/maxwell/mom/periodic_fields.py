"""Periodic exterior field corrections with free-space interior recovery."""
import copy
import math
from dataclasses import replace
import torch
from .kernels import FREE_SPACE
from .fields import FieldQueryConfig
from .operator import local_geometry
from .quadrature import triangle_rule
from .lifting_fields import decode_system,system_local_coefficients
from .periodic_operator import validate_periodic_cell,validate_periodic_sources
from .periodic import EwaldHelmholtzKernel


def regular_surface_fields(handle,coefficients,points,kernel,order):
    """Smooth primary-image subtraction integrated against physical RWG currents."""
    electric=torch.zeros_like(points,dtype=coefficients.dtype)
    magnetic=torch.zeros_like(electric)
    medium=handle.exterior
    for index,body in enumerate(handle.system.bodies):
        basis=body.basis
        local=system_local_coefficients(handle,coefficients,index)
        edges,free,slopes=local_geometry(basis)
        j,m=local[:basis.n_edges][edges]*slopes,local[basis.n_edges:][edges]*slopes
        div_j,div_m=2*j.sum(-1),2*m.sum(-1)
        for start in range(0,len(body.mesh.faces),handle.integration.pair_batch):
            stop=min(start+handle.integration.pair_batch,len(body.mesh.faces))
            source,weights=triangle_rule(body.mesh.triangles[start:stop],order)
            displacement=(source[:,:,None]-free[start:stop,None]).to(coefficients.dtype)
            current_j=torch.einsum('fl,fqlc->fqc',j[start:stop],displacement)
            current_m=torch.einsum('fl,fqlc->fqc',m[start:stop],displacement)
            for begin in range(0,len(points),8):
                end=min(begin+8,len(points))
                g,gradient=kernel.regular_part(points[begin:end,None,None]-source,medium.k)
                weighted=g*weights
                weighted_gradient=gradient*weights[...,None]
                sj=torch.einsum('bfq,fqc->bc',weighted,current_j)
                sm=torch.einsum('bfq,fqc->bc',weighted,current_m)
                gj=torch.einsum('bfqc,f->bc',weighted_gradient,div_j[start:stop])
                gm=torch.einsum('bfqc,f->bc',weighted_gradient,div_m[start:stop])
                cj=torch.linalg.cross(weighted_gradient,current_j[None]).sum((1,2))
                cm=torch.linalg.cross(weighted_gradient,current_m[None]).sum((1,2))
                electric[begin:end]=electric[begin:end]+1j*medium.omega*medium.mu*(sj+gj/medium.k**2)-cm
                magnetic[begin:end]=magnetic[begin:end]+cj+1j*medium.omega*medium.epsilon*(sm+gm/medium.k**2)
    return electric,magnetic


class PeriodicFieldConvergenceError(RuntimeError):
    def __init__(self,receipt):
        self.receipt=receipt
        super().__init__('Periodic field refinement failed: '+receipt['failure_reason'])


def decode_periodic(handle,coefficients,points,region_id,field_kind,trace_side=None,*,kernel,query_config=None,ewald_check=False):
    """Recover fields in any translated cell, retaining a single Bloch sector.

    Only exterior fields receive periodic image potentials. All primary Duffy
    and trace-jump operations use free-space local handles. Refinement compares
    complete physical E/H, so cancellation is included in the estimate. Optional
    Ewald checking independently doubles each kernel cutoff at the final surface
    order. It does not rerun primary free-space field integration.
    """
    if type(ewald_check) is not bool:
        raise TypeError('ewald_check must be a boolean')
    if not isinstance(kernel,EwaldHelmholtzKernel):
        raise TypeError('Periodic fields require EwaldHelmholtzKernel')
    validate_periodic_cell(handle.system,kernel.lattice)
    kernel._guard(complex(handle.exterior.k))
    validate_periodic_sources(handle.sources,kernel.lattice,handle.exterior)
    if any(local.kernel.receipt()!=FREE_SPACE.receipt() for local in handle.body_handles):
        raise ValueError('Periodic recovery requires free-space local body decoders')
    if (not isinstance(points,torch.Tensor) or points.ndim!=2 or points.shape[1]!=3
            or points.device!=handle.system.device or points.dtype!=handle.system.dtype
            or not bool(torch.isfinite(points).all())):
        raise ValueError('Periodic queries require finite matching CUDA [N,3] points')
    config=FieldQueryConfig() if query_config is None else query_config
    if not isinstance(config,FieldQueryConfig):
        raise TypeError('Periodic query configuration requires FieldQueryConfig')
    region=handle.system.region(region_id)
    shift=torch.floor(points@kernel.lattice.reciprocal.T/(2*math.pi)+0.5)
    translation=shift@kernel.lattice.vectors
    wrapped=points-translation
    phase=torch.exp(1j*(translation@kernel.lattice.bloch))[:,None]
    local=copy.copy(handle)
    local.body_handles=tuple(copy.copy(body) for body in handle.body_handles)
    fixed=replace(config,adaptive=False)
    def evaluate(order):
        integration=replace(handle.integration,order=order,singular_order=order,
            near_max_order=max(handle.integration.near_max_order,order+2))
        local.integration=integration
        for body in local.body_handles:
            body.integration=integration
        fields=decode_system(local,coefficients,wrapped,region,field_kind,trace_side,query_config=fixed)
        if region==0 and field_kind!='incident':
            correction=regular_surface_fields(local,coefficients,wrapped,kernel,order)
            fields=tuple(a+b for a,b in zip(fields,correction))
        return tuple(value*phase for value in fields)
    order=max(handle.integration.order,handle.integration.singular_order)
    receipt={'formulation':'periodic_exterior_free_interior_fields_v1','region_id':region,
        'query_count':len(points),'checks':[],'converged':False,'physical_qualification':False,
        'ewald_truncation_verified':False,'certified_error_bound':False,
        'kernel':kernel.receipt(),'failure_reason':None,'adaptive':config.adaptive}
    medium=handle.exterior if region==0 else handle.system.bodies[region-1].interior
    from .tensor_medium import TensorMedium, scalar_medium_if_isotropic
    if isinstance(medium,TensorMedium):
        scalar=scalar_medium_if_isotropic(medium)
        if scalar is not None:
            medium=scalar
    tensor_norm=isinstance(medium,TensorMedium)
    if tensor_norm:
        from ..constants import EPSILON_0,MU_0
        eta=math.sqrt(MU_0/EPSILON_0)
        receipt['field_error_metric']='separate_E_and_vacuum_impedance_H'
    else:
        eta=abs(medium.eta)
    def norm(fields):
        if tensor_norm:
            return torch.stack((fields[0].norm(dim=-1),eta*fields[1].norm(dim=-1)),dim=-1)
        return (fields[0].abs().square().sum(-1)+eta**2*fields[1].abs().square().sum(-1)).sqrt()
    def finish(fields,order):
        if ewald_check:
            checks=[]
            receipt['ewald_checks']=checks
            applicable=region==0 and field_kind!='incident' and len(points)>0
            receipt['ewald_check_applicable']=applicable
            if applicable:
                baseline=regular_surface_fields(local,coefficients,wrapped,kernel,order)
                for parameter in ('real_shells','reciprocal_shells','quadrature_order'):
                    refined=replace(kernel,**{parameter:2*getattr(kernel,parameter)})
                    correction=regular_surface_fields(local,coefficients,wrapped,refined,order)
                    delta=tuple((a-b)*phase for a,b in zip(correction,baseline))
                    difference=norm(delta)
                    candidate=tuple(a+b for a,b in zip(fields,delta))
                    # Require accuracy both relative to the complete field and
                    # the image correction. An incident carrier cannot mask
                    # inaccurate weak image fields; physical cancellation also
                    # tightens the complete-field criterion.
                    candidate_norm,correction_norm=norm(candidate),norm(correction)
                    scale=torch.minimum(candidate_norm,correction_norm)
                    threshold=config.absolute_tolerance+config.relative_tolerance*scale
                    passed=bool((torch.isfinite(difference)&torch.isfinite(candidate_norm)
                        &torch.isfinite(correction_norm)&torch.isfinite(threshold)
                        &(difference<=threshold)).all())
                    checks.append({'axis':parameter,'baseline':getattr(kernel,parameter),
                        'refined':getattr(refined,parameter),'passed':passed,
                        'absolute_difference':difference.detach().cpu().tolist(),
                        'threshold':threshold.detach().cpu().tolist()})
            receipt['ewald_truncation_verified']=all(check['passed'] for check in checks)
            if not receipt['ewald_truncation_verified']:
                receipt.update(converged=False,failure_reason='independent_ewald_estimate_failed')
                handle.last_periodic_field_log=receipt
                raise PeriodicFieldConvergenceError(receipt)
        handle.last_periodic_field_log=receipt
        return fields
    previous=evaluate(order)
    if not config.adaptive or len(points)==0 or field_kind=='incident':
        receipt.update(final_order=order,converged=len(points)==0 or field_kind=='incident',
            failure_reason=None if len(points)==0 or field_kind=='incident' else 'fixed_order_error_unestimated')
        return finish(previous,order)
    while order<config.max_order:
        order=min(2*order,config.max_order)
        current=evaluate(order)
        difference=norm(tuple(a-b for a,b in zip(current,previous)))
        reference=norm(current)
        threshold=config.absolute_tolerance+config.relative_tolerance*reference
        passed=bool((torch.isfinite(difference)&torch.isfinite(reference)
                     &torch.isfinite(threshold)&(difference<=threshold)).all())
        receipt['checks'].append({'order':order,'passed':passed,'absolute_difference':difference.detach().cpu().tolist(),
            'threshold':threshold.detach().cpu().tolist()})
        if passed:
            receipt.update(converged=True,final_order=order)
            return finish(current,order)
        previous=current
    receipt.update(failure_reason='surface_quadrature_budget_exhausted',final_order=order)
    handle.last_periodic_field_log=receipt
    raise PeriodicFieldConvergenceError(receipt)
