"""Shared bounded GMRES and numerical all-column inverse-residual checks."""
import math
import torch


class _Inverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value,owner,adjoint):
        ctx.owner,ctx.adjoint=owner,adjoint
        return owner._solve(value,adjoint)

    @staticmethod
    def backward(ctx,gradient):
        ctx.owner.assert_current()
        return (ctx.owner.apply(gradient) if ctx.adjoint else ctx.owner.adjoint(gradient)),None,None


class _FixedTangentInverse:
    """Shared fixed-real action GMRES; native wrappers validate operator types."""
    def __init__(self,tangent,rule,*,refinement_steps=2,error_type,version,residual_metric):
        from .linalg import StoppingRule
        if (type(rule) is not StoppingRule or rule.method!='gmres' or rule.preconditioner!='none'
                or rule.recycle_size or rule.fallback!='none'):
            raise ValueError('Fixed tangent inverse requires unpreconditioned unrecycled GMRES without fallback')
        if type(refinement_steps) is not int or refinement_steps<0:
            raise ValueError('Refinement steps must be a nonnegative integer')
        tangent.assert_current()
        self.tangent,self.rule,self.refinement_steps=tangent,rule,refinement_steps
        self.shape,self.device,self.dtype=tangent.shape,tangent.device,tangent.dtype
        self.error_type=error_type
        self._binding=(id(tangent),rule,refinement_steps,self.shape,self.device,self.dtype,error_type)
        self.receipt={'version':version,'residual_metric':residual_metric,
            'operator_derivatives':False,'certified_nonsingular':False,'physical_qualification':False,
            'last_forward':None,'last_adjoint':None}

    def assert_current(self):
        if (id(self.tangent),self.rule,self.refinement_steps,self.shape,self.device,self.dtype,self.error_type)!=self._binding:
            raise ValueError('Fixed tangent inverse binding changed')
        self.tangent.assert_current()

    @torch.no_grad()
    def _solve(self,value,adjoint):
        from .linalg import _single_gmres
        self.assert_current()
        if (not isinstance(value,torch.Tensor) or value.ndim<1 or value.shape[-1]!=self.shape[0]
                or value.dtype!=self.dtype or value.device!=self.device or not bool(torch.isfinite(value).all())):
            raise ValueError('Fixed tangent inverse requires matching finite real RHS')
        action=self.tangent.adjoint if adjoint else self.tangent.apply
        results=[];logs=[]
        scale=value.new_ones(self.shape[0])
        for rhs in value.detach().reshape(-1,self.shape[0]):
            threshold=max(self.rule.atol,self.rule.rtol*float(torch.linalg.vector_norm(rhs)))
            result=torch.zeros_like(rhs);residual=rhs
            attempts=[]
            for step in range(self.refinement_steps+1):
                # Preserve relative accuracy for tiny physical sources without
                # relaxing the original RHS residual gate or Krylov policy.
                correction_scale=residual.abs().amax().clamp(max=1)
                correction_scale=torch.where(correction_scale>0,correction_scale,torch.ones_like(correction_scale))
                correction,log=_single_gmres(self.tangent,residual/correction_scale,
                    torch.zeros_like(rhs),self.rule,scale,scale,action=action)
                correction=correction*correction_scale
                log['correction_rhs_scale']=float(correction_scale)
                log['residual_units']='normalized_correction_rhs'
                attempts.append(log);result=result+correction
                residual=rhs-action(result)
                error=float(torch.linalg.vector_norm(residual))
                converged=math.isfinite(threshold) and math.isfinite(error) and bool(torch.isfinite(result).all()) and error<=threshold
                if converged:break
            logs.append({'converged':converged,'attempts':attempts,'absolute_residual':error,
                'threshold':threshold,'refinement_steps_used':step})
            results.append(result)
            if not converged:break
        log={'converged':all(item['converged'] for item in logs),'adjoint':adjoint,'rhs':logs,
            'iteration_budget_scope':'configured maximum per RHS per refinement solve'}
        self.receipt['last_adjoint' if adjoint else 'last_forward']=log
        self.assert_current()
        if not log['converged']:raise self.error_type(log)
        return torch.stack(results).reshape_as(value) if results else value.clone()

    def apply(self,value):return _Inverse.apply(value,self,False)
    def adjoint(self,value):return _Inverse.apply(value,self,True)


@torch.no_grad()
def _streamed_inverse_bounds(inverse,*,column_batch=8):
    """Accumulate complete J, B and I-JB norm bounds in bounded column batches.

    For R=I-JB with norm(R)<1, sigma_min(J)>=(1-norm(R))/norm(B).
    These numerical bounds do not certify underlying action errors or outward
    rounding. Partial solves cannot establish any global bound.
    """
    from dataclasses import asdict
    inverse.assert_current()
    tangent,rule=inverse.tangent,inverse.rule
    if type(column_batch) is not int or column_batch<1:
        raise ValueError('Tangent column batch must be a positive integer')
    size=tangent.shape[0]
    row_sums=[torch.zeros(size,device=tangent.device,dtype=torch.float64) for _ in range(3)]
    column_max=[0.,0.,0.]
    checked=0;failure=None
    for start in range(0,size,column_batch):
        count=min(column_batch,size-start)
        basis=torch.zeros(count,size,device=tangent.device,dtype=tangent.dtype)
        basis[torch.arange(count,device=tangent.device),torch.arange(start,start+count,device=tangent.device)]=1
        try:
            columns=inverse.apply(basis)
        except inverse.error_type as error:
            failure=error.receipt
            break
        residual=basis-tangent.apply(columns)
        direct=tangent.apply(basis)
        for index,values in enumerate((direct,columns,residual)):
            absolute=values.abs().to(torch.float64)
            row_sums[index]+=absolute.sum(0)
            column_max[index]=max(column_max[index],float(absolute.sum(1).max()))
        checked+=count
    tangent.assert_current()
    norms=[math.sqrt(float(rows.max()))*math.sqrt(cols) for rows,cols in zip(row_sums,column_max)]
    upper,inverse_upper,residual_upper=norms
    rounding=64*torch.finfo(tangent.dtype).eps*size*(1+upper*inverse_upper)
    residual_upper+=rounding
    complete=checked==size and failure is None
    finite=all(math.isfinite(v) for v in (*norms,rounding,residual_upper))
    lower=(max(0.,1-residual_upper)/inverse_upper
        if complete and finite and inverse_upper>0 else 0.)
    cutoff=64*torch.finfo(tangent.dtype).eps*size*upper
    admitted=complete and finite and math.isfinite(cutoff) and lower>cutoff
    return {'method':'complete_streamed_inverse_residual','columns_checked':checked,
        'column_batch':column_batch,'complete':complete,'tangent_rule':asdict(rule),
        'sigma_min_lower':lower,'sigma_max_upper':upper if complete else None,
        'inverse_norm_upper':inverse_upper if complete else None,
        'inverse_residual_norm_upper':residual_upper if complete else None,
        'numerical_singularity_cutoff':cutoff if complete else None,
        'admitted':admitted,'inconclusive':not admitted,'solve_failure':failure,
        'underlying_action_error_certified':False,'interval_certified':False,
        'physical_qualification':False,'fast_work_qualified':False}
