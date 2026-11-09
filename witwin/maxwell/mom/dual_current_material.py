"""Material-aware implicit derivatives of admitted dual current inverses."""
import torch
from .dual_current_action import StreamedDualCurrentBalance


class _MaterialInverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx,rhs,electric,magnetic,owner,adjoint):
        owner._validate(rhs,electric,magnetic)
        result=owner.balance.inverse.adjoint(rhs) if adjoint else owner.balance.inverse.apply(rhs)
        ctx.owner,ctx.adjoint=owner,adjoint
        ctx.save_for_backward(result,electric,magnetic)
        return result

    @staticmethod
    def backward(ctx,gradient):
        owner=ctx.owner
        current,electric,magnetic=ctx.saved_tensors
        owner.assert_current()
        paired=_MaterialInverse.apply(gradient,electric,magnetic,owner,not ctx.adjoint)
        # For A^-T, exchange the primal and dual vectors in -lambda^T dA x.
        primal,dual=(paired,current) if ctx.adjoint else (current,paired)
        j,m=primal.chunk(2,dim=-1)
        lj,lm=dual.chunk(2,dim=-1)
        a=owner.balance.action
        de=-lj*a._project(a.curl.apply(m),False)
        dm=lm*a._project(a.curl.apply(j),False)
        if de.ndim>1:
            axes=tuple(range(de.ndim-1))
            de,dm=de.sum(axes),dm.sum(axes)
        owner.assert_current()
        return paired,de,dm,None,None


class DualMaterialInverse:
    """Differentiable A(sigma)^-1 at the admitted preparation's material values.

    Packed conductivities retain their graph; they must equal the fixed native
    action used for Krylov solves. Changed materials require new preparation.
    The implicit backward uses -lambda^T(dA)x, including transposed solves and
    arbitrary leading RHS batches. Geometry stays fixed. Iterative projection
    and solve errors remain numerical, not interval-certified material errors.
    """
    def __init__(self,balance):
        if type(balance) is not StreamedDualCurrentBalance:
            raise TypeError('Material inverse requires a native admitted dual balance')
        balance.assert_current()
        self.balance=balance
        self._binding=id(balance)

    def assert_current(self):
        if self._binding!=id(self.balance):raise ValueError('Dual material inverse binding changed')
        self.balance.assert_current()

    def _validate(self,rhs,electric,magnetic):
        self.assert_current()
        a=self.balance.action
        for value,reference in ((electric,a.electric),(magnetic,a.magnetic)):
            if (not isinstance(value,torch.Tensor) or value.shape!=reference.shape
                    or value.device!=reference.device or value.dtype!=reference.dtype
                    or not bool(torch.isfinite(value).all()) or not torch.equal(value.detach(),reference)):
                raise ValueError('Dual material values differ from admitted preparation')
        if (not isinstance(rhs,torch.Tensor) or rhs.ndim<1 or rhs.shape[-1]!=a.shape[0]
                or rhs.device!=a.device or rhs.dtype!=a.dtype or not bool(torch.isfinite(rhs).all())):
            raise ValueError('Dual material inverse requires matching finite packed current RHS')

    def apply(self,rhs,electric,magnetic):
        return _MaterialInverse.apply(rhs,electric,magnetic,self,False)

    def adjoint(self,rhs,electric,magnetic):
        return _MaterialInverse.apply(rhs,electric,magnetic,self,True)

    def electric_response(self,source,electric,magnetic):
        """Electric field response to electric impressed current and zero M source."""
        a=self.balance.action
        if (not isinstance(source,torch.Tensor) or source.ndim<1 or source.shape[-1]!=a.shape[0]//2
                or source.device!=a.device or source.dtype!=a.dtype or not bool(torch.isfinite(source).all())):
            raise ValueError('Dual material response requires matching finite electric sources')
        rhs=a.source_rhs(torch.cat((source,torch.zeros_like(source)),dim=-1))
        currents=self.apply(rhs,electric,magnetic)
        return (currents[...,:source.shape[-1]]-source)/electric
