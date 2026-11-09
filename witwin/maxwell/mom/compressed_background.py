"""Fixed analytic-source background banks for compressed volume preparation."""
import math
import torch
from .implicit_surface_solve import FixedSurfaceInverse
from .volume_mesh import BoundVolumeRegion
from .runtime import DecodeHandle
from .lifting import build_selected_lifted_rhs
from .lifting_fields import decode_lifted
from .fields import FieldQueryConfig
from .volume_source_refinement import checked_dual_norms
from .tensor_medium import TensorMedium
from .dyadic_green import DyadicGreenKernel
from ..constants import ETA_0


class CompressedBackgroundBank:
    """Frozen declared-source responses with native real/complex scale gradients.

    Scales multiply each declared carrier, including its declared amplitude and
    phase. Geometry, materials and source shape are fixed. Empty source banks
    represent unpumped harmonics. These fields already include the analytic
    incident contribution; callers must add them only once.
    """
    def __init__(self,inverse,bindings,carriers,rhs,surface,electric,magnetic,receipt):
        self.inverse,self.bindings,self.carriers=inverse,tuple(bindings),tuple(carriers)
        self.rhs,self.surface,self.electric,self.magnetic=(v.detach().clone() for v in (rhs,surface,electric,magnetic))
        self.device,self.dtype=inverse.device,inverse.dtype
        self.receipt=receipt
        self._binding=self._identity()
        self._versions=tuple((id(t),t._version) for t in self.tensors())

    def _identity(self):
        return (id(self.inverse),tuple(map(id,self.bindings)),self.carriers,self.device,self.dtype)

    def tensors(self):return (self.rhs,self.surface,self.electric,self.magnetic)

    def assert_current(self):
        if self._identity()!=self._binding or tuple((id(t),t._version) for t in self.tensors())!=self._versions:
            raise ValueError('Background bank binding or arrays changed')
        self.inverse.assert_current()
        for binding in self.bindings:binding.assert_current()

    def combine(self,scales):
        self.assert_current()
        if (not isinstance(scales,torch.Tensor) or scales.ndim<1 or scales.shape[-1]!=len(self.carriers)
                or scales.device!=self.device or scales.dtype not in (self.dtype,self.electric.real.dtype)
                or not bool(torch.isfinite(scales).all())):
            raise ValueError('Background scales require finite matching CUDA real/complex source coefficients')
        values=scales.to(self.dtype)
        return {'background_surface':values@self.surface,
            'background_electric':torch.einsum('...s,sqc->...qc',values,self.electric),
            'background_magnetic':torch.einsum('...s,sqc->...qc',values,self.magnetic)}


@torch.no_grad()
def prepare_compressed_background(inverse,bindings,sources,*,query_config=FieldQueryConfig(),
        edge_batch=32,query_batch=16,relative_tolerance=1e-8,absolute_tolerance=1e-12,
        tensor_rhs_cache_max_bytes=0):
    """Solve declared analytic sources, check all original rows and interior fields.

    RHS quadrature follows the fixed surface integration configuration. Field
    quadrature is adaptive. This is not RHS quadrature/mesh convergence or
    physical qualification. Source-scale differentiation occurs in the returned
    bank without repeating preparation or retaining a solve tape.
    """
    if type(inverse) is not FixedSurfaceInverse:
        raise TypeError('Background preparation requires a guarded surface inverse')
    if (not isinstance(query_config,FieldQueryConfig) or not query_config.adaptive
            or any(type(v) is not int or v<1 for v in (edge_batch,query_batch))
            or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in (relative_tolerance,absolute_tolerance))
            or relative_tolerance+absolute_tolerance==0):
        raise ValueError('Invalid background integration/residual settings')
    if type(tensor_rhs_cache_max_bytes) is not int or tensor_rhs_cache_max_bytes<0:
        raise ValueError('Tensor RHS cache capacity must be a nonnegative integer')
    inverse.assert_current();provider=inverse.operator.provider
    system,exterior=provider.system,provider.exterior
    tensor_regions=[b.region_id for b in system.bodies if isinstance(b.interior,TensorMedium)]
    if tensor_rhs_cache_max_bytes and not tensor_regions:
        raise ValueError('Tensor RHS cache settings require tensor regions')
    kernels={}
    for region in tensor_regions:
        owned=getattr(provider,'interior_kernels',())
        k=owned[region-1] if len(owned)==len(system.bodies) else None
        medium=system.bodies[region-1].interior
        if (type(k) is not DyadicGreenKernel or k.medium is not medium
                or k.mode_cache.medium is not medium or k.mode_cache.max_bytes!=k.cache_max_bytes):
            raise ValueError('Tensor background requires the exact regional kernel and cache policy')
        kernels[region]=k
    bindings=tuple(bindings);sources=tuple(sources)
    if (not bindings or any(type(b) is not BoundVolumeRegion or b.surface_system is not system for b in bindings)
            or len({b.volume.region_id for b in bindings})!=len(bindings)):
        raise ValueError('Background requires unique exact owning-system bindings')
    for binding in bindings:binding.assert_current()
    count=sum(len(b.volume.cells) for b in bindings)
    rhs=torch.zeros(len(sources),inverse.shape[0],device=inverse.device,dtype=inverse.dtype)
    handles=[];carriers=[];rhs_logs=[]
    for index,source in enumerate(sources):
        local_handles=[]
        for body in system.bodies:
            handle=DecodeHandle(body.basis,exterior,body.interior,(source,),provider.config,{})
            if body.region_id in kernels:
                handle.tensor_kernel=kernels[body.region_id]
                handle.tensor_rhs_cache_max_bytes=tensor_rhs_cache_max_bytes
            for start in range(0,body.basis.n_edges,edge_batch):
                edges=torch.arange(start,min(start+edge_batch,body.basis.n_edges),device=inverse.device)
                if body.region_id in kernels:
                    from .anisotropic_lifting import build_tensor_lifted_rhs
                    value,log=build_tensor_lifted_rhs(handle,interior_kernel=handle.tensor_kernel,test_edges=edges)
                else:value,log=build_selected_lifted_rhs(handle,edges)
                ids=edges+body.edge_offset
                rhs[index,torch.cat((ids,ids+system.total_edges))]=value
                rhs_logs.append({'source_index':index,'region_id':body.region_id,**log})
            local_handles.append(handle)
        handles.append(local_handles);carriers.append(local_handles[0].incident_carrier)
    receipt={'version':'compressed_declared_source_background_v1','source_count':len(sources),
        'regions':[b.volume.region_id for b in bindings],'rhs':rhs_logs,'field_queries':[],
        'converged':False,'physical_qualification':False,'rhs_quadrature_error_estimated':False,
        'incident_background_included':True,'source_carrier_identities':[c.identity for c in carriers],
        'original_relative_dual_tolerance':relative_tolerance,'original_absolute_dual_tolerance':absolute_tolerance}
    if tensor_regions:
        receipt.update(tensor_regions=tensor_regions,
            tensor_rhs_cache_max_bytes_per_handle=tensor_rhs_cache_max_bytes,
            tensor_field_impedance_proxy='vacuum_eta',aggregate_cache_capacity_enforced=False)
    surface=inverse.apply(rhs) if sources else torch.zeros_like(rhs)
    if sources:
        residual=torch.zeros_like(rhs)
        for start in range(0,system.total_edges,edge_batch):
            edges=torch.arange(start,min(start+edge_batch,system.total_edges),device=inverse.device)
            ids=torch.cat((edges,edges+system.total_edges))
            residual[:,ids]=provider.sampled_action(edges,surface,source_batch=edge_batch)-rhs[:,ids]
        error=checked_dual_norms(residual,inverse.operator.metric)
        threshold=absolute_tolerance+relative_tolerance*checked_dual_norms(rhs,inverse.operator.metric)
        receipt['original_residuals']=error.cpu().tolist();receipt['original_thresholds']=threshold.cpu().tolist()
        if not bool(torch.isfinite(error).all() and torch.isfinite(threshold).all() and (error<=threshold).all()):
            from .volume_response_matrix import VolumeMatrixConvergenceError
            receipt['failure_reason']='background_original_equation_failed'
            raise VolumeMatrixConvergenceError(receipt)
    electric=[];magnetic=[]
    for index,local_handles in enumerate(handles):
        ee=[];hh=[]
        for binding in bindings:
            region=binding.volume.region_id;body=system.bodies[region-1]
            ids=torch.arange(body.edge_offset,body.edge_offset+body.basis.n_edges,device=inverse.device)
            coefficients=surface[index,torch.cat((ids,ids+system.total_edges))]
            points=binding.volume.vertices[binding.volume.cells].mean(1)
            for start in range(0,len(points),query_batch):
                handle=local_handles[region-1]
                if region in kernels:
                    from .anisotropic_fields import decode_tensor_lifted
                    e,h=decode_tensor_lifted(handle,coefficients,points[start:start+query_batch],query_config=query_config)
                else:e,h=decode_lifted(handle,coefficients,points[start:start+query_batch],1,'total',query_config=query_config)
                log=handle.last_field_query_log
                receipt['field_queries'].append({'source_index':index,'region_id':region,'query_start':start,**log})
                magnitude=torch.sqrt(e.abs().square().sum(-1)+(ETA_0 if region in kernels else abs(body.interior.eta))**2*h.abs().square().sum(-1))
                threshold=query_config.absolute_tolerance+query_config.relative_tolerance*magnitude
                if (not log.get('converged',False) or not bool(torch.isfinite(e).all() and torch.isfinite(h).all()
                        and torch.isfinite(magnitude).all() and torch.isfinite(threshold).all())
                        or any(not math.isfinite(value) for c in log['checks'] for value in
                            ([c['E']['max_absolute_difference'],c['H']['max_absolute_difference']]
                             if 'E' in c and 'H' in c else [c['max_absolute_error_V_per_m']]))):
                    from .volume_response_matrix import VolumeMatrixConvergenceError
                    receipt['failure_reason']='background_field_check_failed'
                    raise VolumeMatrixConvergenceError(receipt)
                ee.append(e);hh.append(h)
        electric.append(torch.cat(ee));magnetic.append(torch.cat(hh))
    e=torch.stack(electric) if sources else rhs.new_zeros(0,count,3)
    h=torch.stack(magnetic) if sources else rhs.new_zeros(0,count,3)
    inverse.assert_current()
    for binding in bindings:binding.assert_current()
    receipt.update(converged=True,failure_reason=None)
    return CompressedBackgroundBank(inverse,bindings,carriers,rhs,surface,e,h,receipt)
