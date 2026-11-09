"""Separable native CUDA translations for total-degree Cartesian moments."""
import math
import torch
from .fmm_expansion import multi_indices, _points


def shift_storage_bytes(order,node_count,real_bytes,storage):
    if type(node_count) is not int or node_count<1 or real_bytes not in (4,8):
        raise ValueError('Shift storage requires positive nodes and native real precision')
    count=len(multi_indices(order))
    if storage=='dense':return (node_count-1)*count*count*2*real_bytes
    if storage!='separable_v1':raise ValueError('Unsupported multipole shift storage')
    return (0 if node_count==1 else 6*count*(order+1)*(16+real_bytes))+(node_count-1)*3*real_bytes


class SeparableMultipoleShift:
    """Three axis translations, with bounded polynomial rows and no dense S.

    Layout is shared across all tree edges of a projection. Each edge retains
    only its real xyz displacement. Adjoint applies reversed transposed axes;
    Forward uses ordered gather/reduction and retains native current autograd.
    Native gather backward may accumulate repeated indices; no bitwise
    determinism of backward is claimed.
    """
    def __init__(self,order,*,device,dtype):
        indices=multi_indices(order)
        device=torch.device(device)
        if device.type!='cuda' or dtype not in (torch.float32,torch.float64):
            raise ValueError('Separable shifts require native real CUDA layout')
        self.order,self.count,self.device,self.real_dtype=order,len(indices),device,dtype
        self.dtype=torch.complex128 if dtype==torch.float64 else torch.complex64
        lookup={alpha:i for i,alpha in enumerate(indices)}
        layouts=[]
        for adjoint in (False,True):
            for axis in range(3):
                origins,factors,powers=[],[],[]
                for alpha in indices:
                    for k in range(order+1):
                        beta=list(alpha);beta[axis]=alpha[axis]+k if adjoint else k
                        valid=sum(beta)<=order if adjoint else k<=alpha[axis]
                        origins.append(lookup[tuple(beta)] if valid else 0)
                        factors.append(math.comb(beta[axis],alpha[axis]) if valid and adjoint
                                       else math.comb(alpha[axis],k) if valid else 0)
                        powers.append(k if valid and adjoint else alpha[axis]-k if valid else 0)
                shape=(self.count,order+1)
                layouts.append((torch.tensor(origins,device=device,dtype=torch.int64).reshape(shape),
                    torch.tensor(factors,device=device,dtype=dtype).reshape(shape),
                    torch.tensor(powers,device=device,dtype=torch.int64).reshape(shape)))
        self.layouts=tuple(layouts)
        self.device=self.layouts[0][0].device
        self._binding=self._state()
        self._versions=tuple((id(t),t._version) for t in self.tensors())

    def _state(self):
        return (self.order,self.count,self.device,self.real_dtype,self.dtype,
                tuple(tuple(id(t) for t in row) for row in self.layouts))

    def tensors(self):return tuple(t for row in self.layouts for t in row)

    def assert_current(self):
        if self._state()!=self._binding or tuple((id(t),t._version) for t in self.tensors())!=self._versions:
            raise ValueError('Separable multipole shift layout changed')

    def apply(self,displacement,value,*,adjoint=False,reject_underflow=False):
        self.assert_current();_points(displacement)
        if (displacement.shape!=(3,) or displacement.device!=self.device
                or displacement.dtype!=self.real_dtype or type(adjoint) is not bool or type(reject_underflow) is not bool):
            raise ValueError('Separable displacement must match its CUDA layout')
        if (not isinstance(value,torch.Tensor) or value.ndim<2 or value.shape[-2]!=self.count
                or value.device!=self.device or value.dtype!=self.dtype or not bool(torch.isfinite(value).all())):
            raise ValueError('Separable values require matching finite complex CUDA moment channels')
        for axis in ((2,1,0) if adjoint else (0,1,2)):
            origins,factors,powers=self.layouts[axis+3*adjoint]
            weights=factors*displacement[axis].pow(powers)
            if not bool(torch.isfinite(weights).all()):
                raise ValueError('Separable shift exceeds native precision')
            if reject_underflow:
                gathered=value[...,origins,:]
                product=gathered*weights[...,None]
                if bool(((gathered != 0) & (weights[...,None] != 0) & (product == 0)).any()):
                    raise ValueError('Separable nonzero current products underflow native precision')
                value=product.sum(-2)
                del gathered,product
            else:
                value=(value[...,origins,:]*weights[...,None]).sum(-2)
        if not bool(torch.isfinite(value).all()):raise ValueError('Separable translated moments are nonfinite')
        return value
