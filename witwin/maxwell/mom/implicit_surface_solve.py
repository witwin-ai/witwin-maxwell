"""Native fixed-operator inverse actions for coupled volume responses."""
import torch
from .operator_protocol import IterativeState
from .hmatrix import HierarchicalOperator
from .h2_operator import HierarchicalH2Operator
from .linalg import StoppingRule,solve_reference


class SurfaceInverseConvergenceError(RuntimeError):
    def __init__(self,receipt):
        super().__init__('Fixed surface inverse failed its true residual tolerance')
        self.receipt=receipt


class _SolveView:
    supports_direct=False
    def __init__(self,owner,adjoint):
        self.owner,self.is_adjoint=owner,adjoint
        self.shape,self.device,self.dtype=owner.shape,owner.device,owner.dtype
        self.iterative_state=IterativeState()
    def refresh(self):self.owner.assert_current()
    def read_access(self):return self.owner.operator.read_access()
    def apply(self,value):
        return self.owner.operator.adjoint(value) if self.is_adjoint else self.owner.operator.apply(value)
    def adjoint(self,value):
        return self.owner.operator.apply(value) if self.is_adjoint else self.owner.operator.adjoint(value)
    def riesz_inverse(self,value):return value
    def preconditioner_scales(self,kind):
        left,right,reused=self.owner.operator.preconditioner_scales(kind)
        return (right,left,reused) if self.is_adjoint else (left,right,reused)
    def persistence_receipt(self):return self.owner.receipt.copy()
    def operator_stats(self):return self.owner.operator.operator_stats()


class _ImplicitInverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value,owner,adjoint):
        ctx.owner,ctx.adjoint=owner,adjoint
        return owner._solve(value,adjoint)
    @staticmethod
    def backward(ctx,gradient):
        ctx.owner.assert_current()
        result=ctx.owner.apply(gradient) if ctx.adjoint else ctx.owner.adjoint(gradient)
        return result,None,None


class FixedSurfaceInverse:
    """Solve A or A^H with true Euclidean residuals and implicit RHS gradients.

    Operator geometry/material derivatives are excluded. The adjoint solves
    the fixed conjugate-transposed equation, not the finite Arnoldi program;
    forward/backward residuals satisfy separate tolerances. Physical H/H2
    handles also enforce their configured checked RWG dual residual tolerance.
    Solution and gradient
    errors additionally depend on conditioning.
    Backend compression and original-equation residual errors are additional.
    Bounded residual correction uses the same Krylov configuration per pass;
    correction RHS vectors are normalized separately so the inner absolute
    tolerance cannot suppress a correction required by the original dual gate.
    final acceptance always uses the original RHS threshold. No inverse matrix,
    iteration tape or persistent RHS is retained.
    """
    def __init__(self,operator,rule=StoppingRule(preconditioner='none'),*,refinement_steps=2):
        from .hierarchical_runtime import HierarchicalSystemOperator,HierarchicalH2SystemOperator
        if type(operator) not in (HierarchicalOperator,HierarchicalH2Operator,
                                 HierarchicalSystemOperator,HierarchicalH2SystemOperator):
            raise TypeError('Fixed inverse requires an immutable H or H2 surface operator')
        if (not isinstance(rule,StoppingRule) or rule.method not in ('gmres','block_gmres')
                or rule.fallback!='none' or rule.recycle_size
                or rule.preconditioner not in ('none','row_column_norm')):
            raise ValueError('Fixed inverse requires unrecycled Krylov with none or row-column scaling')
        if type(refinement_steps) is not int or refinement_steps<0:
            raise ValueError('Refinement steps must be a nonnegative integer')
        operator.refresh()
        self.refinement_steps=refinement_steps
        self.operator,self.rule=operator,rule
        self.shape,self.device,self.dtype=operator.shape,operator.device,operator.dtype
        if len(self.shape)!=2 or self.shape[0]!=self.shape[1] or self.device.type!='cuda':
            raise ValueError('Fixed inverse requires a square CUDA operator')
        self._binding=(id(operator),rule,self.shape,self.device,self.dtype,refinement_steps)
        self.receipt={'version':'fixed_surface_implicit_inverse_v1','residual_metric':'Euclidean',
            'operator_derivatives':False,'original_equation_verified':False,
            'physical_qualification':False,'refinement_steps':refinement_steps,'last_forward':None,'last_adjoint':None}

    def assert_current(self):
        if (id(self.operator),self.rule,self.shape,self.device,self.dtype,self.refinement_steps)!=self._binding:
            raise ValueError('Fixed inverse binding changed')
        self.operator.refresh()

    def _dual_gate(self,residual,value):
        config=getattr(self.operator,'_public_config',None)
        if config is None:return True,{}
        from .volume_source_refinement import checked_dual_norms
        policy=config.h2 if config.operator_backend=='h2' else config.hierarchical
        error=checked_dual_norms(residual,self.operator.metric)
        threshold=policy.solve_dual_tolerance*checked_dual_norms(value,self.operator.metric)
        passed=bool(torch.isfinite(error).all() and torch.isfinite(threshold).all() and (error<=threshold).all())
        return passed,{'dual_absolute_residuals':error.detach().cpu().tolist(),
            'dual_thresholds':threshold.detach().cpu().tolist(),'dual_converged':passed}

    @torch.no_grad()
    def _solve(self,value,adjoint):
        self.assert_current()
        view=_SolveView(self,adjoint)
        attempts=[]
        with view.read_access():
            result,log=solve_reference(view,value,self.rule)
            attempts.append(log)
            threshold=torch.maximum(torch.linalg.vector_norm(value,dim=-1)*self.rule.rtol,
                value.real.new_tensor(self.rule.atol))
            for step in range(self.refinement_steps+1):
                residual=value-view.apply(result)
                error=torch.linalg.vector_norm(residual,dim=-1)
                dual_passed,dual_log=self._dual_gate(residual,value)
                converged=dual_passed and bool(torch.isfinite(threshold).all() and torch.isfinite(result).all() and torch.isfinite(error).all()
                    and (error<=threshold).all())
                if converged or step==self.refinement_steps:break
                correction_scale=residual.abs().amax(dim=-1,keepdim=True)
                correction_scale=torch.where(correction_scale>0,correction_scale,torch.ones_like(correction_scale))
                correction,log=solve_reference(view,residual/correction_scale,self.rule)
                correction=correction*correction_scale
                log['correction_rhs_scale']=correction_scale.squeeze(-1).detach().cpu().tolist()
                log['residual_units']='normalized_correction_rhs'
                attempts.append(log)
                result=result+correction
        log={'converged':converged,'attempts':attempts,
            'refinement_steps_used':len(attempts)-1,
            'iteration_budget_scope':'configured maximum per RHS per refinement solve',
            'absolute_residuals':error.detach().cpu().tolist(),
            'thresholds':threshold.detach().cpu().tolist(),'adjoint':adjoint,**dual_log}
        self.receipt['last_adjoint' if adjoint else 'last_forward']=log
        if not converged:raise SurfaceInverseConvergenceError(log)
        return result

    def apply(self,value):
        if torch.is_grad_enabled() and isinstance(value,torch.Tensor) and value.requires_grad:
            return _ImplicitInverse.apply(value,self,False)
        return self._solve(value,False)

    def adjoint(self,value):
        if torch.is_grad_enabled() and isinstance(value,torch.Tensor) and value.requires_grad:
            return _ImplicitInverse.apply(value,self,True)
        return self._solve(value,True)
