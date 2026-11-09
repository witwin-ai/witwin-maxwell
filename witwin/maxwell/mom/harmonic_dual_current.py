"""Nonlinear time-averaged TPA feedback in conservative dual DC currents.

This closure consumes actual positive harmonic fields. Polarization has no DC
time derivative and does not enter the conductive current law. Public HB state,
decoder and snapshot integration remain separate obligations.
"""
import math
import torch
from .dual_static_current import DualStaticCurrentBalance
from .nonlinear import instantaneous_harmonic_sources


class _BoundConductivity(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value,owner):
        ctx.owner=owner
        return value.clone()

    @staticmethod
    def backward(ctx,gradient):
        ctx.owner.assert_current()
        return gradient,None


class DualCurrentConvergenceError(RuntimeError):
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__('Dual DC nonlinear closure failed: '+receipt['failure_reason'])


class DualHarmonicCurrentClosure:
    formulation = 'dual_conductive_nonlinear_cell_current_dc_v1'

    def __init__(self, volumes, sigma_e, sigma_m, *, relative_tolerance=1e-10,
                 absolute_tolerance=1e-12, max_iterations=30, max_backtracks=12,
                 max_matrix_bytes=512*1024**2, streamed_rule=None, streamed_column_batch=8):
        for value in (relative_tolerance,absolute_tolerance):
            if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
                raise ValueError('Dual DC tolerances must be finite and nonnegative')
        if relative_tolerance+absolute_tolerance == 0:
            raise ValueError('Dual DC tolerances cannot both be zero')
        for value in (max_iterations,max_backtracks):
            if type(value) is not int or value < 0:
                raise ValueError('Dual DC iteration budgets must be nonnegative integers')
        if type(streamed_column_batch) is not int or streamed_column_batch<=0:
            raise ValueError('Dual streamed column batch must be a positive integer')
        if streamed_rule is None and streamed_column_batch!=8:
            raise ValueError('Inactive dual streamed column batch must use its default')
        self.streamed_rule,self.streamed_column_batch=streamed_rule,streamed_column_batch
        self.relative_tolerance,self.absolute_tolerance = relative_tolerance,absolute_tolerance
        self.max_iterations,self.max_backtracks = max_iterations,max_backtracks
        if streamed_rule is None:
            self.current = DualStaticCurrentBalance(volumes,sigma_e,sigma_m,max_matrix_bytes=max_matrix_bytes)
            self.volumes,self.counts = self.current.volumes,self.current.counts
            size = len(self.current.electric)
            injection = torch.cat((self.current.projector,torch.zeros_like(self.current.projector)))
            currents = torch.linalg.solve(self.current.matrix,injection)
            eye = torch.eye(size,device=currents.device,dtype=currents.dtype)
            self.response = (currents[:size]-eye)/self.current.electric[:,None]
            if not bool(torch.isfinite(self.response).all()):
                raise ValueError('Dual nonlinear DC source response is nonfinite')
        else:
            from .dual_current_action import DualStaticCurrentAction,StreamedDualCurrentBalance
            from .dual_current_response import DualElectricResponse
            action=DualStaticCurrentAction(volumes,sigma_e,sigma_m,projection_rule=streamed_rule,
                                          column_batch=streamed_column_batch)
            self.current=StreamedDualCurrentBalance(action,streamed_rule,column_batch=streamed_column_batch)
            self.volumes,self.counts=action.volumes,action.counts
            self.response=DualElectricResponse(self.current)
            from .dual_current_material import DualMaterialInverse
            self.material_inverse=DualMaterialInverse(self.current)
            self.material_inputs=tuple(sigma_e)+tuple(sigma_m)
            self.material_coefficients=tuple(_BoundConductivity.apply(torch.cat([b._coefficient(v,'DC conductivity').repeat_interleave(3)
                for b,v in zip(action.blocks,values)]),self) for values in (sigma_e,sigma_m))
        self._versions = self._state()

    def _state(self):
        return (id(self.current),tuple(id(v) for v in self.volumes),self.counts,
                self.relative_tolerance,self.absolute_tolerance,self.max_iterations,self.max_backtracks,
                self.streamed_rule,self.streamed_column_batch,id(self.response),
                self.response._version if self.streamed_rule is None else
                (id(self.material_inverse),tuple((id(v),v._version) for v in self.material_coefficients),
                 tuple((id(v),v._version) if isinstance(v,torch.Tensor) else v for v in self.material_inputs)))

    def assert_current(self):
        if self._versions != self._state():
            raise RuntimeError('Dual nonlinear DC preparation was mutated')
        self.current.assert_current()
        if self.streamed_rule is not None:
            self.response.assert_current()
            self.material_inverse.assert_current()
            if self.material_inverse.balance is not self.current:
                raise RuntimeError('Dual nonlinear material preparation changed')

    def solve(self, ac_electric, tpa_scale):
        self.assert_current()
        sample = self.current.points if self.streamed_rule is None else self.current.action.curl.points
        complex_dtype = torch.complex128 if sample.dtype == torch.float64 else torch.complex64
        count = sum(self.counts)
        if (not isinstance(ac_electric,torch.Tensor) or ac_electric.device != sample.device
                or ac_electric.dtype != complex_dtype or ac_electric.ndim != 3
                or ac_electric.shape[1:] != (count,3) or not len(ac_electric)
                or not bool(torch.isfinite(ac_electric).all())):
            raise ValueError('Dual DC closure requires matching finite positive harmonics [H,cells,3]')
        if type(tpa_scale) in (float,int) and math.isfinite(tpa_scale):
            tpa_scale = sample.new_tensor(tpa_scale)
        if (not isinstance(tpa_scale,torch.Tensor) or tpa_scale.device != sample.device
                or tpa_scale.dtype != sample.dtype or tpa_scale.shape not in ((),(count,))
                or not bool(torch.isfinite(tpa_scale).all()) or bool((tpa_scale < 0).any())):
            raise ValueError('Dual DC TPA requires matching nonnegative scalar or per-cell coefficients')
        tpa = tpa_scale.expand(count)
        covariance = .5*torch.einsum('hti,htj->tij',ac_electric,ac_electric.conj()).real
        trace = covariance.diagonal(dim1=-2,dim2=-1).sum(-1)
        zero = torch.zeros_like(sample)
        third = instantaneous_harmonic_sources(torch.cat((zero[None].to(complex_dtype),ac_electric)),
                                               tpa_scale=1.).dissipative_current[0].real
        eye3 = torch.eye(3,device=sample.device,dtype=sample.dtype)
        eye = torch.eye(3*count,device=sample.device,dtype=sample.dtype) if self.streamed_rule is None else None
        respond = ((lambda value:self.response@value) if self.streamed_rule is None else
                   (lambda value:self.material_inverse.electric_response(value,*self.material_coefficients)))
        def law(vector):
            electric = vector.reshape(count,3)
            moment = ((electric.square().sum(-1)+trace)[:,None]*electric
                      +2*torch.einsum('tij,tj->ti',covariance,electric)+third)
            tangent = tpa[:,None,None]*((electric.square().sum(-1)+trace)[:,None,None]*eye3
                        +2*electric[:,:,None]*electric[:,None,:]+2*covariance)
            return (tpa[:,None]*moment).flatten(),tangent
        def equation(vector):
            source,_ = law(vector)
            return vector-respond(source)
        def derivative(vector):
            _,tangent = law(vector)
            if self.streamed_rule is None:
                return eye-self.response@torch.block_diag(*tangent.unbind(0))
            from .dual_current_response import DualNonlinearTangent
            return DualNonlinearTangent(self.response,tangent)
        norm = lambda value: float(value.detach().norm())
        vector = zero.flatten()
        initial_residual = equation(vector)
        threshold = self.absolute_tolerance+self.relative_tolerance*norm(initial_residual)
        receipt = {'formulation':self.formulation,'converged':False,'failure_reason':None,
                   'threshold_V_per_m_l2':threshold,'checks':[],'physical_qualification':False,
                   'gradient':'first_order_implicit_native_torch','nonlinear_law':'exact_mean_cubic_TPA'}
        if self.streamed_rule is not None:
            from dataclasses import asdict
            receipt.update(linear_solver='streamed_dual_checked_gmres',
                streamed_rule=asdict(self.streamed_rule),streamed_column_batch=self.streamed_column_batch,
                tangent_checks=[],global_response_stored=False,global_nonlinear_tangent_stored=False,
                conductivity_derivatives=True,fast_work_qualified=False)
        def fail(reason):
            receipt['failure_reason'] = reason
            raise DualCurrentConvergenceError(receipt)
        if not math.isfinite(threshold): fail('nonfinite_initial_residual')
        def inverse_at(vector,phase):
            matrix=derivative(vector)
            if self.streamed_rule is None:
                matrix=matrix.detach()
                if not bool(torch.isfinite(matrix).all()):fail('nonfinite_tangent')
                return lambda rhs:torch.linalg.solve(matrix,rhs)
            from .dual_current_response import FixedDualNonlinearInverse,DualNonlinearInverseError
            from .fixed_tangent_inverse import _streamed_inverse_bounds
            inverse=FixedDualNonlinearInverse(matrix,self.streamed_rule)
            bound=_streamed_inverse_bounds(inverse,column_batch=self.streamed_column_batch)
            check={'phase':phase,'bound':bound,'solve':None}
            receipt['tangent_checks'].append(check)
            if not bound['admitted']:fail('inconclusive_tangent_bound')
            def solve(rhs):
                try:result=inverse.apply(rhs)
                except DualNonlinearInverseError as error:
                    check['solve']=error.receipt
                    fail('tangent_inverse_residual_failed')
                check['solve']=inverse.receipt['last_forward']
                return result
            return solve
        for iteration in range(self.max_iterations+1):
            residual = initial_residual if iteration==0 else equation(vector)
            magnitude = norm(residual)
            receipt['checks'].append({'iteration':iteration,'residual_V_per_m_l2':magnitude})
            if not math.isfinite(magnitude): fail('nonfinite_residual')
            try:
                if magnitude <= threshold:
                    root = (vector-inverse_at(vector,'implicit')(residual.detach())).detach()
                    corrected = root-inverse_at(root,'implicit')(equation(root))
                    final = norm(equation(corrected))
                    if not math.isfinite(final) or final > threshold: fail('final_nonlinear_residual_failed')
                    source,_ = law(corrected)
                    result = self.current.solve(source.reshape(count,3),zero,
                        **({'conductivities':self.material_coefficients} if self.streamed_rule is not None else {}))
                    consistency = norm(result['electric'].flatten()-corrected)
                    if not math.isfinite(consistency) or consistency > threshold: fail('final_coupled_field_mismatch')
                    receipt.update(converged=True,iterations=iteration,final_residual_V_per_m_l2=final,
                                   coupled_field_error_V_per_m_l2=consistency)
                    result['nonlinear_receipt'] = receipt
                    result['tpa_current'] = source.reshape(count,3)
                    return result
                if iteration == self.max_iterations: fail('iteration_budget_exhausted')
                direction = inverse_at(vector,'newton')(-residual.detach())
            except torch.linalg.LinAlgError:
                fail('singular_tangent')
            for backtrack in range(self.max_backtracks+1):
                candidate = vector+2.**(-backtrack)*direction
                candidate_norm = norm(equation(candidate))
                if math.isfinite(candidate_norm) and candidate_norm <= (1-1e-4*2.**(-backtrack))*magnitude:
                    vector = candidate.detach()
                    break
            else:
                fail('backtracking_exhausted')

    def electric(self, ac_electric, tpa_scale, *, polarization=None):
        return self.solve(ac_electric,tpa_scale)['electric']
