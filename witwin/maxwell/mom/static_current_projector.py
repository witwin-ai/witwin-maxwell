"""Cellwise conservative current projection without a global projector array."""
import torch
from .static_current import FaceCurrentBalance,FixedStaticCurrentInverse
from .fixed_tangent_inverse import _streamed_inverse_bounds


class _Product(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value,owner,mode):
        ctx.owner,ctx.mode=owner,mode
        return owner._product(value,mode)

    @staticmethod
    def backward(ctx,gradient):
        opposite={'apply':'adjoint','adjoint':'apply','recover':'recover_adjoint','recover_adjoint':'recover'}
        ctx.owner.assert_current()
        return _Product.apply(gradient,ctx.owner,opposite[ctx.mode]),None,None


class StaticCurrentProjector:
    """Fixed P=I-G (G^T W G)^-1 G^T W on one body's cell vectors.

    G maps pinned face-mean potentials to cell gradients and W is cell volume.
    The inverse is residual-checked GMRES with complete inverse-residual
    admission. P is W-orthogonal, generally not Euclidean symmetric; adjoint
    products explicitly transpose the weights. Recovery returns pinned reduced
    potentials, whose volume-mean gauge can be chosen separately.

    Storage is linear in mesh size and the Krylov restart/column batch. Complete
    admission still requires all inverse columns, so this is not a fast-work
    qualification. This primitive does not yet replace the dual DC curl,
    coupled solve, nonlinear response or public snapshot representation.
    """
    def __init__(self,block,rule,*,column_batch=8):
        if type(block) is not FaceCurrentBalance:
            raise TypeError('Current projector requires an exact native face-mean discretization')
        block._assert_current()
        sample=block.volume.vertices
        tangent=torch.eye(3,device=sample.device,dtype=sample.dtype).expand(len(block.volume.cells),3,3)
        self.inverse=FixedStaticCurrentInverse(block.tangent_action(tangent),rule)
        bound=_streamed_inverse_bounds(self.inverse,column_batch=column_batch)
        if not bound['admitted']:
            raise CurrentProjectionAdmissionError(bound)
        self.block=block
        self.shape=(3*len(block.volume.cells),)*2
        self.device,self.dtype=sample.device,sample.dtype
        self._binding=(id(block),id(self.inverse),self.shape,self.device,self.dtype)
        self.receipt={'method':'cellwise_weighted_current_projection','tangent_bound':bound,
            'global_projector_stored':False,'physical_qualification':False,'fast_work_qualified':False,
            'underlying_action_error_certified':False,'interval_certified':False}

    def assert_current(self):
        if self._binding!=(id(self.block),id(self.inverse),self.shape,self.device,self.dtype):
            raise ValueError('Current projector binding changed')
        self.block._assert_current();self.inverse.assert_current()

    def _gradient(self,value):
        potential=torch.cat((value.new_zeros(*value.shape[:-1],1),value),dim=-1)
        return torch.einsum('...ta,tai->...ti',potential[...,self.block.cell_dofs],self.block.gradients)

    def _weak(self,field,weighted):
        if weighted:field=field*self.block.volume.volumes[:,None]
        local=torch.einsum('tai,...ti->...ta',self.block.gradients,field)
        return field.new_zeros(*field.shape[:-2],self.block.n_dofs).index_add(
            -1,self.block.cell_dofs.flatten(),local.flatten(-2))[...,1:]

    def _product(self,value,mode):
        self.assert_current()
        size=self.block.n_dofs-1 if mode=='recover_adjoint' else self.shape[0]
        if (not isinstance(value,torch.Tensor) or value.ndim<1 or value.shape[-1]!=size
                or value.dtype!=self.dtype or value.device!=self.device or not bool(torch.isfinite(value).all())):
            raise ValueError('Current projector requires matching finite real coefficients')
        if mode=='recover_adjoint':
            result=(self._gradient(self.inverse.adjoint(value))*self.block.volume.volumes[:,None]).flatten(-2)
        else:
            field=value.reshape(*value.shape[:-1],len(self.block.volume.cells),3)
            if mode=='adjoint':
                correction=self._gradient(self.inverse.adjoint(self._weak(field,False)))*self.block.volume.volumes[:,None]
                result=value-correction.flatten(-2)
            else:
                potential=self.inverse.apply(self._weak(field,True))
                result=potential if mode=='recover' else value-self._gradient(potential).flatten(-2)
        if not bool(torch.isfinite(result).all()):raise ValueError('Nonfinite current projection')
        self.assert_current()
        return result

    def apply(self,value):return _Product.apply(value,self,'apply')
    def adjoint(self,value):return _Product.apply(value,self,'adjoint')
    def recover(self,value):return _Product.apply(value,self,'recover')
    def recover_adjoint(self,value):return _Product.apply(value,self,'recover_adjoint')


class CurrentProjectionAdmissionError(RuntimeError):
    def __init__(self,receipt):
        super().__init__('Current projection tangent bound is inconclusive')
        self.receipt=receipt
