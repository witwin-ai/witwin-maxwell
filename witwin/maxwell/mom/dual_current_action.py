"""Fixed electric/magnetic DC block actions without global dense matrices."""
import torch
from .static_current import FaceCurrentBalance
from .static_current_projector import StaticCurrentProjector
from .static_volume_curl import StaticVolumeCurlAction
from .fixed_tangent_inverse import _FixedTangentInverse


class _Product(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value,owner,mode):
        ctx.owner,ctx.mode=owner,mode
        return owner._product(value,mode)

    @staticmethod
    def backward(ctx,gradient):
        ctx.owner.assert_current()
        opposite={'apply':'adjoint','adjoint':'apply','source':'source_adjoint','source_adjoint':'source'}
        return _Product.apply(gradient,ctx.owner,opposite[ctx.mode]),None,None


class DualStaticCurrentAction:
    """A[J,M]=[J+sigma_e P C M, M-sigma_m P C J].

    P is block-diagonal conservative projection per body; C includes all global
    cell interactions. Conductivities are fixed positive homogeneous samples
    per body, frozen by value. Products expose vector derivatives only. The
    Euclidean adjoint reverses C/P/diagonal order and preserves both signs.

    Projectors and curls retain no global matrices, but direct curl work and
    projector admission remain expensive. Neither coupled nonsingularity nor
    physical/scaling qualification follows from constructing this action.
    """
    def __init__(self,volumes,sigma_e,sigma_m,*,projection_rule,column_batch=8,target_batch=32,source_batch=32):
        self.curl=StaticVolumeCurlAction(volumes,target_batch=target_batch,source_batch=source_batch)
        self.volumes=self.curl.volumes;self.counts=self.curl.counts
        if len(sigma_e)!=len(self.volumes) or len(sigma_m)!=len(self.volumes):
            raise ValueError('Dual DC action requires one conductivity per body and channel')
        self.blocks=tuple(FaceCurrentBalance(v) for v in self.volumes)
        electric=[];magnetic=[]
        for block,e,m in zip(self.blocks,sigma_e,sigma_m):
            for value,output in ((e,electric),(m,magnetic)):
                scalar=block._coefficient(value,'DC conductivity')
                if not bool((scalar>0).all()) or not bool((scalar==scalar[0]).all()):
                    raise ValueError('Dual DC action requires positive homogeneous conductivities')
                output.append(scalar.detach().repeat_interleave(3).clone())
        self.electric,self.magnetic=torch.cat(electric),torch.cat(magnetic)
        self.projectors=tuple(StaticCurrentProjector(b,projection_rule,column_batch=column_batch) for b in self.blocks)
        self.device,self.dtype=self.curl.device,self.curl.dtype
        self.shape=(2*self.curl.shape[0],)*2
        self._binding=self._state()
        self.receipt={'method':'cellwise_dual_current_action','global_coupled_matrix_stored':False,
            'global_projector_stored':False,'global_curl_stored':False,'coupled_nonsingularity_certified':False,
            'operator_derivatives':False,'physical_qualification':False,'fast_work_qualified':False}

    def _state(self):
        return (tuple(id(v) for v in self.volumes),self.counts,tuple(id(b) for b in self.blocks),
            tuple(id(p) for p in self.projectors),id(self.curl),id(self.electric),self.electric._version,
            id(self.magnetic),self.magnetic._version,self.shape,self.device,self.dtype)

    def assert_current(self):
        if self._binding!=self._state():raise ValueError('Dual current action binding changed')
        self.curl.assert_current()
        for block,projector in zip(self.blocks,self.projectors):
            if projector.block is not block:raise ValueError('Dual current projector block changed')
            projector.assert_current()

    def _project(self,value,adjoint):
        return torch.cat([(p.adjoint(v) if adjoint else p.apply(v)) for p,v in
            zip(self.projectors,value.split(tuple(3*n for n in self.counts),dim=-1))],dim=-1)

    def _product(self,value,mode):
        self.assert_current()
        if (not isinstance(value,torch.Tensor) or value.ndim<1 or value.shape[-1]!=self.shape[0]
                or value.dtype!=self.dtype or value.device!=self.device or not bool(torch.isfinite(value).all())):
            raise ValueError('Dual current action requires matching finite real packed channel coefficients')
        if value.numel()==0:return value.clone()
        j,m=value.chunk(2,dim=-1)
        if mode in ('source','source_adjoint'):
            result=torch.cat((self._project(j,mode=='source_adjoint'),self._project(m,mode=='source_adjoint')),dim=-1)
        elif mode=='adjoint':
            result=torch.cat((j-self.curl.adjoint(self._project(self.magnetic*m,True)),
                              m+self.curl.adjoint(self._project(self.electric*j,True))),dim=-1)
        else:
            result=torch.cat((j+self.electric*self._project(self.curl.apply(m),False),
                              m-self.magnetic*self._project(self.curl.apply(j),False)),dim=-1)
        if not bool(torch.isfinite(result).all()):raise ValueError('Nonfinite dual current action')
        self.assert_current()
        return result

    def apply(self,value):return _Product.apply(value,self,'apply')
    def adjoint(self,value):return _Product.apply(value,self,'adjoint')
    def source_rhs(self,value):return _Product.apply(value,self,'source')
    def source_rhs_adjoint(self,value):return _Product.apply(value,self,'source_adjoint')


class DualCurrentInverseConvergenceError(RuntimeError):
    def __init__(self,receipt):
        super().__init__('Dual current inverse failed its original RHS residual tolerance')
        self.receipt=receipt


class FixedDualCurrentInverse(_FixedTangentInverse):
    """Bounded current-vector GMRES; coupled nonsingularity remains separate."""
    def __init__(self,action,rule,*,refinement_steps=2):
        if type(action) is not DualStaticCurrentAction:raise TypeError('Dual inverse requires an exact native current action')
        super().__init__(action,rule,refinement_steps=refinement_steps,error_type=DualCurrentInverseConvergenceError,
            version='fixed_dual_current_inverse_v1',residual_metric='packed_current_Euclidean')


class DualCurrentAdmissionError(RuntimeError):
    def __init__(self,receipt):
        super().__init__('Dual current coupled inverse bound is inconclusive')
        self.receipt=receipt


class StreamedDualCurrentBalance:
    """Admitted fixed-conductivity source-to-field solve at volume cell centers.

    Admission checks every inverse column numerically; it does not certify
    inner projection error or physical accuracy. Source derivatives and optional
    live-conductivity derivatives are supported; geometry remains fixed.
    """
    def __init__(self,action,rule,*,column_batch=8):
        from .fixed_tangent_inverse import _streamed_inverse_bounds
        self.inverse=FixedDualCurrentInverse(action,rule)
        bound=_streamed_inverse_bounds(self.inverse,column_batch=column_batch)
        if not bound['admitted']:raise DualCurrentAdmissionError(bound)
        self.action=action
        self.bound=bound
        self._binding=(id(action),id(self.inverse))

    def assert_current(self):
        if self._binding!=(id(self.action),id(self.inverse)) or self.inverse.tangent is not self.action:
            raise ValueError('Dual current balance binding changed')
        self.inverse.assert_current()

    def _state(self):
        self.assert_current()
        return (self._binding,self.action._state())

    @property
    def volumes(self):return self.action.volumes

    @property
    def counts(self):return self.action.counts

    @property
    def blocks(self):return self.action.blocks

    @property
    def points(self):return self.action.curl.points

    @property
    def electric(self):return self.action.electric

    @property
    def magnetic(self):return self.action.magnetic

    def solve(self,electric_source,magnetic_source,*,conductivities=None):
        from .magnetostatic import cell_current_continuity
        self.assert_current()
        a=self.action
        shape=(sum(a.counts),3)
        for value in (electric_source,magnetic_source):
            if (not isinstance(value,torch.Tensor) or value.shape!=shape or value.device!=a.device
                    or value.dtype!=a.dtype or not bool(torch.isfinite(value).all())):
                raise ValueError('Dual DC sources require matching finite real CUDA [cells,3]')
        se,sm=electric_source.flatten(),magnetic_source.flatten()
        rhs=a.source_rhs(torch.cat((se,sm)))
        de,dm=a.electric,a.magnetic
        if conductivities is None:
            currents=self.inverse.apply(rhs)
        else:
            from .dual_current_material import DualMaterialInverse
            if not isinstance(conductivities,tuple) or len(conductivities)!=2:
                raise ValueError('Dual material solve requires electric/magnetic packed conductivities')
            de,dm=conductivities
            currents=DualMaterialInverse(self).apply(rhs,de,dm)
        j,m=currents.chunk(2)
        cm,cj=a.curl.apply(m),a.curl.apply(j)
        electric,magnetic=(j-se)/de,(m-sm)/dm
        potential_e,potential_m=[],[]
        sizes=tuple(3*n for n in a.counts)
        for block,p,drive_e,drive_m in zip(a.blocks,a.projectors,(se/de-cm).split(sizes),
                                 (sm/dm+cj).split(sizes)):
            for drive,target in ((drive_e,potential_e),(drive_m,potential_m)):
                phi=torch.cat((drive.new_zeros(1),p.recover(drive)))
                target.append(phi-(phi*block.nodal_volume).sum()/block.nodal_volume.sum())
        reconstructed_e=torch.cat([b.electric(p).flatten() for b,p in zip(a.blocks,potential_e)])-cm
        reconstructed_h=torch.cat([b.electric(p).flatten() for b,p in zip(a.blocks,potential_m)])+cj
        residual=torch.cat((j+de*a._project(cm,False),m-dm*a._project(cj,False)))-rhs
        # Preserve the dense formulation's unprojected physical-current scale.
        scales=torch.cat((j.abs()+(de*cm).abs()+rhs[:len(j)].abs(),
                          m.abs()+(dm*cj).abs()+rhs[len(j):].abs()))
        threshold=1024*torch.finfo(currents.dtype).eps*len(currents)*scales.norm()
        if (not all(bool(torch.isfinite(v).all()) for v in
                    (electric,magnetic,reconstructed_e,reconstructed_h,residual,threshold))
                or not bool(torch.isfinite(residual.norm())) or not bool(residual.norm()<=threshold)):
            raise RuntimeError('Dual DC original coupled current residual failed')
        continuity=[]
        for volume,jb,mb in zip(a.volumes,j.reshape(shape).split(a.counts),m.reshape(shape).split(a.counts)):
            continuity.append({key:float(cell_current_continuity(volume,v)['area_weighted_jump_rms_A_per_m2'].detach())
                               for key,v in (('electric',jb),('magnetic',mb))})
        self.assert_current()
        return {'electric':electric.reshape(shape),'magnetic':magnetic.reshape(shape),
            'electric_current':j.reshape(shape),'magnetic_current':m.reshape(shape),
            'electric_potentials':tuple(potential_e),'magnetic_potentials':tuple(potential_m),
            'receipt':{'formulation':'streamed_dual_conservative_cell_current_dc_v1',
                'coupled_current_residual':float(residual.norm().detach()),'threshold':float(threshold.detach()),
                'electric_reconstruction_error':float((electric-reconstructed_e).norm().detach()),
                'magnetic_reconstruction_error':float((magnetic-reconstructed_h).norm().detach()),
                'normal_current_jumps':continuity,'coupled_inverse_bound':self.bound,
                'current_solve':self.inverse.receipt['last_forward'],'global_coupled_matrix_stored':False,
                'operator_derivatives':conductivities is not None,'geometry_derivatives':False,
                'underlying_action_error_certified':False,
                'exterior_potential_coupling':False,'interface_trace_convergence':False,
                'physical_qualification':False,'fast_work_qualified':False}}
