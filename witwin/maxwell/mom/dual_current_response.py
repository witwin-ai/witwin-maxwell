"""Fixed dual conductive electric response and local nonlinear tangents."""
import torch
from .dual_current_action import StreamedDualCurrentBalance
from .fixed_tangent_inverse import _FixedTangentInverse


class _Product(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value,owner,adjoint):
        ctx.owner,ctx.adjoint=owner,adjoint
        return owner._product(value,adjoint)

    @staticmethod
    def backward(ctx,gradient):
        ctx.owner.assert_current()
        return _Product.apply(gradient,ctx.owner,not ctx.adjoint),None,None


def _validate(owner,value):
    owner.assert_current()
    if (not isinstance(value,torch.Tensor) or value.ndim<1 or value.shape[-1]!=owner.shape[0]
            or value.dtype!=owner.dtype or value.device!=owner.device or not bool(torch.isfinite(value).all())):
        raise ValueError('Dual response requires matching finite real vectors')


class DualElectricResponse:
    """R s = (first(A^-1 [P s,0])-s)/sigma_e, with fixed parameters."""
    def __init__(self,balance):
        if type(balance) is not StreamedDualCurrentBalance:
            raise TypeError('Dual response requires a native admitted current balance')
        balance.assert_current()
        self.balance=balance
        a=balance.action
        self.shape=(a.shape[0]//2,)*2
        self.device,self.dtype=a.device,a.dtype
        self._binding=(id(balance),self.shape,self.device,self.dtype)

    def assert_current(self):
        if self._binding!=(id(self.balance),self.shape,self.device,self.dtype):
            raise ValueError('Dual electric response binding changed')
        self.balance.assert_current()

    def _product(self,value,adjoint):
        _validate(self,value)
        a=self.balance.action
        if adjoint:
            weighted=value/a.electric
            solved=self.balance.inverse.adjoint(torch.cat((weighted,torch.zeros_like(weighted)),dim=-1))
            result=a.source_rhs_adjoint(solved)[...,:self.shape[0]]-weighted
        else:
            rhs=a.source_rhs(torch.cat((value,torch.zeros_like(value)),dim=-1))
            result=(self.balance.inverse.apply(rhs)[...,:self.shape[0]]-value)/a.electric
        if not bool(torch.isfinite(result).all()):raise ValueError('Nonfinite dual electric response')
        self.assert_current()
        return result

    def apply(self,value):return _Product.apply(value,self,False)
    def adjoint(self,value):return _Product.apply(value,self,True)


class DualNonlinearTangent:
    """Frozen J=I-R T, where T contains one real 3x3 constitutive block per cell."""
    def __init__(self,response,tangent):
        if type(response) is not DualElectricResponse:raise TypeError('Native dual response required')
        response.assert_current()
        self.response=response
        self.shape,self.device,self.dtype=response.shape,response.device,response.dtype
        if (not isinstance(tangent,torch.Tensor) or tangent.shape!=(self.shape[0]//3,3,3)
                or tangent.device!=self.device or tangent.dtype!=self.dtype or not bool(torch.isfinite(tangent).all())):
            raise ValueError('Dual nonlinear tangent requires matching finite cell blocks')
        self.tangent=tangent.detach().clone()
        self._binding=self._state()

    def _state(self):
        return (id(self.response),id(self.tangent),self.tangent._version,self.shape,self.device,self.dtype)

    def assert_current(self):
        if self._binding!=self._state():raise ValueError('Dual nonlinear tangent binding changed')
        self.response.assert_current()

    def _local(self,value,adjoint):
        block=self.tangent.transpose(-1,-2) if adjoint else self.tangent
        return torch.einsum('tij,...tj->...ti',block,value.reshape(*value.shape[:-1],-1,3)).flatten(-2)

    def _product(self,value,adjoint):
        _validate(self,value)
        if value.numel()==0:return value.clone()
        result=value-(self._local(self.response.adjoint(value),True) if adjoint else
                      self.response.apply(self._local(value,False)))
        if not bool(torch.isfinite(result).all()):raise ValueError('Nonfinite dual nonlinear tangent')
        self.assert_current()
        return result

    def apply(self,value):return _Product.apply(value,self,False)
    def adjoint(self,value):return _Product.apply(value,self,True)


class DualNonlinearInverseError(RuntimeError):
    def __init__(self,receipt):
        self.receipt=receipt
        super().__init__('Dual nonlinear tangent inverse failed its original RHS residual')


class FixedDualNonlinearInverse(_FixedTangentInverse):
    def __init__(self,tangent,rule,*,refinement_steps=2):
        if type(tangent) is not DualNonlinearTangent:raise TypeError('Native dual nonlinear tangent required')
        super().__init__(tangent,rule,refinement_steps=refinement_steps,error_type=DualNonlinearInverseError,
            version='fixed_dual_nonlinear_inverse_v1',residual_metric='electric_field_Euclidean')
