"""Fixed native packed-real HB tangents without a global Jacobian array."""
import math
import torch
from .fixed_tangent_inverse import _FixedTangentInverse,_streamed_inverse_bounds
from .harmonic_balance import DenseHarmonicBalance
from .action_harmonic_balance import ActionHarmonicBalance


class _Product(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value,owner,adjoint):
        ctx.owner,ctx.adjoint=owner,adjoint
        return owner._product(value,adjoint)

    @staticmethod
    def backward(ctx,gradient):
        ctx.owner.assert_current()
        return (ctx.owner.apply(gradient) if ctx.adjoint else ctx.owner.adjoint(gradient)),None,None


class HarmonicJacobianAction:
    """Freeze d(pack(residual))/d(pack(E)) at one field and pump value.

    Real packing includes DC and both AC quadratures. Products use native
    reverse-over-reverse JVPs and VJPs, including implicit surface inverses.
    Product-vector derivatives are supported; linearization-point, material and
    geometry derivatives are outside this fixed-tangent contract. Conductive DC
    closures may still build their own dense local/global DC tangent; this class
    only removes storage of the complete HB Jacobian, not all dense solver work.
    """
    def __init__(self,model,electric,*,pump_scale=1.):
        if type(model) not in (DenseHarmonicBalance,ActionHarmonicBalance):
            raise TypeError('HB tangent requires an exact native harmonic model')
        model._assert_current()
        if (not isinstance(electric,torch.Tensor) or electric.shape!=model.background.shape
                or electric.dtype!=model.background.dtype or electric.device!=model.background.device
                or not bool(torch.isfinite(electric).all()) or bool(electric[0].imag.any())):
            raise ValueError('HB tangent requires matching finite fields with real DC')
        if type(pump_scale) not in (float,int) or not math.isfinite(pump_scale) or pump_scale<0:
            raise ValueError('HB tangent pump must be finite and nonnegative')
        self.model,self.pump_scale=model,float(pump_scale)
        self.vector=model._pack(electric).detach().clone()
        self.device,self.dtype=self.vector.device,self.vector.dtype
        self.shape=(len(self.vector),)*2
        self._binding=(id(model),id(self.vector),self.vector._version,self.pump_scale,self.shape,self.device,self.dtype)

    def assert_current(self):
        if self._binding!=(id(self.model),id(self.vector),self.vector._version,self.pump_scale,self.shape,self.device,self.dtype):
            raise ValueError('HB tangent linearization changed')
        self.model._assert_current()

    def _equation(self,vector):
        return self.model._pack(self.model.residual(self.model._unpack(vector),pump_scale=self.pump_scale))

    def _product(self,value,adjoint):
        self.assert_current()
        if (not isinstance(value,torch.Tensor) or value.ndim<1 or value.shape[-1]!=self.shape[0]
                or value.dtype!=self.dtype or value.device!=self.device or not bool(torch.isfinite(value).all())):
            raise ValueError('HB tangent vectors require matching finite packed-real coefficients')
        if value.numel()==0:return value.clone()
        products=[]
        with torch.enable_grad():
            for direction in value.detach().reshape(-1,self.shape[0]):
                point=self.vector.detach().requires_grad_(True)
                if adjoint:
                    result,=torch.autograd.grad(self._equation(point),point,grad_outputs=direction)
                else:
                    _,result=torch.autograd.functional.jvp(self._equation,point,direction,create_graph=False)
                if not bool(torch.isfinite(result).all()):raise ValueError('Nonfinite HB tangent action')
                products.append(result.detach())
        self.assert_current()
        return torch.stack(products).reshape_as(value)

    def apply(self,value):return _Product.apply(value,self,False)
    def adjoint(self,value):return _Product.apply(value,self,True)


class HarmonicTangentConvergenceError(RuntimeError):
    def __init__(self,receipt):
        super().__init__('Harmonic tangent solve failed its original RHS residual tolerance')
        self.receipt=receipt


class FixedHarmonicTangentInverse(_FixedTangentInverse):
    """Residual-checked GMRES and implicit RHS gradients on a fixed HB tangent.

    Nonsingularity is a separate admission obligation. No dense fallback,
    retained iteration tape or operator-parameter derivatives are provided.
    """
    def __init__(self,tangent,rule,*,refinement_steps=2):
        if type(tangent) is not HarmonicJacobianAction:
            raise TypeError('Harmonic inverse requires an exact native fixed tangent')
        super().__init__(tangent,rule,refinement_steps=refinement_steps,
            error_type=HarmonicTangentConvergenceError,
            version='fixed_harmonic_tangent_inverse_v1',residual_metric='packed_real_Euclidean')


@torch.no_grad()
def streamed_tangent_bounds(tangent,*,column_batch=8):
    """Complete numerical diagonal-dominance bounds without a global matrix.

    Every column is evaluated; this is not a projected or sampled certificate.
    For the numerically evaluated tangent, strict row/column dominance bounds
    inverse infinity/one norms, hence the smallest singular value. Lack of
    dominance is inconclusive, not proof of singularity. Underlying compressed
    action/inner-solve errors are not bounded by this algebraic check. Floating
    norm evaluations are not outward-rounded interval certificates. Storage
    is O(N*column_batch), but N tangent products are required: no fast-work or
    physical qualification is implied.
    """
    if type(tangent) is not HarmonicJacobianAction:
        raise TypeError('Tangent bounds require an exact native fixed tangent')
    if type(column_batch) is not int or column_batch<1:
        raise ValueError('Tangent column batch must be a positive integer')
    tangent.assert_current()
    size=tangent.shape[0]
    row_off=torch.zeros(size,device=tangent.device,dtype=torch.float64)
    column_off=torch.zeros_like(row_off);diagonal=torch.zeros_like(row_off)
    for start in range(0,size,column_batch):
        count=min(column_batch,size-start)
        indices=torch.arange(start,start+count,device=tangent.device)
        local=torch.arange(count,device=tangent.device)
        basis=torch.zeros(count,size,device=tangent.device,dtype=tangent.dtype)
        basis[local,indices]=1
        columns=tangent.apply(basis).abs().to(torch.float64)
        diagonal[indices]=columns[local,indices]
        columns[local,indices]=0
        row_off+=columns.sum(0)
        column_off[indices]=columns.sum(1)
    tangent.assert_current()
    row_max=float((row_off+diagonal).max());column_max=float((column_off+diagonal).max())
    upper=math.sqrt(row_max)*math.sqrt(column_max)
    rounding=64*torch.finfo(torch.float64).eps*size*upper
    row_margin=float((diagonal-row_off).min())-rounding
    column_margin=float((diagonal-column_off).min())-rounding
    lower=max(0.,row_margin/math.sqrt(size),column_margin/math.sqrt(size))
    if row_margin>0 and column_margin>0:
        lower=max(lower,math.sqrt(row_margin)*math.sqrt(column_margin))
    cutoff=64*torch.finfo(tangent.dtype).eps*size*upper
    finite=all(math.isfinite(v) for v in (upper,lower,cutoff,row_margin,column_margin))
    return {'method':'complete_streamed_diagonal_dominance','columns_checked':size,
        'column_batch':column_batch,'sigma_min_lower':lower,'sigma_max_upper':upper,
        'numerical_singularity_cutoff':cutoff,'row_margin':row_margin,'column_margin':column_margin,
        'admitted':finite and lower>cutoff,'inconclusive':not (finite and lower>cutoff),
        'underlying_action_error_certified':False,'interval_certified':False,'physical_qualification':False,
        'fast_work_qualified':False}


@torch.no_grad()
def streamed_inverse_tangent_bounds(tangent,rule,*,column_batch=8):
    """Numerical all-column inverse-residual bounds for nondominant tangents.

    Stream approximate inverse columns B and compute R = I - J B directly.
    If ||R||_2 < 1, sigma_min(J) >= (1 - ||R||_2) / ||B||_2.
    Norms use sqrt(||.||_1 ||.||_inf), accumulating only row sums and column
    maxima. No nonsingularity assumption about the iterative solver is needed:
    its returned columns are merely candidate columns, checked against J.

    This is a numerical sufficient check, not an interval certificate. Errors
    in underlying tangent actions are not certified. N iterative solves and N
    additional tangent columns are required, with O(N*(batch + restart)) work
    storage; neither fast work nor absence of dense DC work is implied. A failed
    iterative solve or norm gate is inconclusive, not proof of singularity.
    """
    return _streamed_inverse_bounds(FixedHarmonicTangentInverse(tangent,rule),column_batch=column_batch)
