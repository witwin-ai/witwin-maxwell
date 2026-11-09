"""Independent polynomial binomial oracle for shared coefficient FFT translations."""
import json
import math
import pytest
import torch
from witwin.maxwell.mom.fmm_fft_coefficients import CartesianCoefficientFFTTranslation
from witwin.maxwell.mom.fmm_expansion import multi_indices, multipole_moments
from witwin.maxwell.mom.fmm_storage import m2l_storage_bytes
from witwin.maxwell.mom.fmm_capacity import storage_bytes

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def polynomial_tables(coefficients,order):
    indices=multi_indices(order);lookup={a:i for i,a in enumerate(multi_indices(2*order+1))}
    green=[];gradient=[]
    # Independent host integer oracle only; coefficient numerics remain CUDA.
    for beta in indices:
        grow=[];hrow=[]
        for alpha in indices:
            gamma=tuple(a+b for a,b in zip(alpha,beta))
            factor=(-1)**sum(alpha)*math.prod(math.comb(a+b,a) for a,b in zip(alpha,beta))
            grow.append(coefficients[lookup[gamma]]*factor)
            hrow.append(torch.stack([coefficients[lookup[tuple(g+(j==axis) for j,g in enumerate(gamma))]]
                *factor*(gamma[axis]+1) for axis in range(3)]))
        green.append(torch.stack(grow));gradient.append(torch.stack(hrow))
    return torch.cat((torch.stack(green)[None],torch.stack(gradient).movedim(-1,0)),dim=0)


@pytest.mark.parametrize('kind',('complex_polynomial','zero_constant','zero_polynomial'))
def test_arbitrary_polynomial_all_fields_adjoint_gradient_and_independent_ownership(kind,tmp_path):
    order=3;generator=torch.Generator(device='cuda').manual_seed(933)
    coefficients=torch.randn(len(multi_indices(2*order+1)),device='cuda',dtype=torch.complex128,generator=generator)
    if kind=='zero_constant':coefficients[0]=0
    if kind=='zero_polynomial':coefficients.zero_()
    center=torch.zeros(3,device='cuda',dtype=torch.float64);scale=center.new_tensor(.1)
    op=CartesianCoefficientFFTTranslation(coefficients,center,scale,order)
    tables=polynomial_tables(coefficients,order)
    assert storage_bytes(op.tensors())==m2l_storage_bytes(order,8,'fft_v1')
    assert all(t._base is None and t.untyped_storage().nbytes()==t.numel()*t.element_size() for t in op.tensors())
    for source,owned in ((coefficients,op.coefficients),(center,op.displacement),(scale,op.scale)):
        assert source.untyped_storage().data_ptr()!=owned.untyped_storage().data_ptr()
    points=torch.randn((9,3),device='cuda',dtype=torch.float64,generator=generator)*.01
    charges=torch.randn((3,7,9),device='cuda',dtype=op.dtype,generator=generator)
    value=multipole_moments(points,charges,order).transpose(0,1).detach().requires_grad_(True)
    actual=op.apply(value);expected=torch.einsum('fba,...a->...fb',tables,value)
    weights=.01**op.indices.sum(-1)
    error=float((((actual-expected)*weights).norm()/(expected*weights).norm().clamp_min(1e-300)).detach())
    dual=torch.randn(actual.shape,device='cuda',dtype=op.dtype,generator=generator)*weights
    adjoint=op.adjoint(dual);reference=torch.einsum('fba,...fb->...a',tables.conj(),dual)
    adjoint_error=float((((adjoint-reference)*weights).norm()/(reference*weights).norm().clamp_min(1e-300)).detach())
    gradient,=torch.autograd.grad(actual,value,dual)
    torch.testing.assert_close(gradient,adjoint,rtol=0,atol=0)
    lhs=(actual.conj()*dual).sum();rhs=(value.conj()*adjoint).sum()
    dot=float(((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs()).clamp_min(1e-300)).detach())
    record=dict(kind=kind,forward=error,adjoint=adjoint_error,dot=dot,receipt=op.receipt())
    (tmp_path/'polynomial.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
    assert error<1e-11 and adjoint_error<1e-11 and dot<1e-11
    coefficients.add_(1);center.add_(1);scale.add_(1);op.assert_current()
    op.scale.add_(1)
    with pytest.raises(ValueError,match='changed'):op.apply(value)


def test_coefficient_factory_rejects_gradients_invalid_normalization_and_native_precision(monkeypatch):
    import witwin.maxwell.mom.fmm_fft as shared
    order=3;center=torch.zeros(3,device='cuda',dtype=torch.float64);scale=center.new_tensor(.1)
    coefficients=torch.ones(len(multi_indices(2*order+1)),device='cuda',dtype=torch.complex128)
    def forbidden(*a,**k):raise AssertionError('Rejected polynomial reached FFT field construction')
    monkeypatch.setattr(CartesianCoefficientFFTTranslation,'_fourier_fields',forbidden)
    for c in (coefficients.cpu(),coefficients.cfloat(),coefficients.clone().requires_grad_(),coefficients*torch.nan,coefficients[:-1]):
        with pytest.raises(ValueError):CartesianCoefficientFFTTranslation(c,center,scale,order)
    for x in (scale*0,scale*torch.nan,scale.float(),scale.cpu(),scale.clone().requires_grad_()):
        with pytest.raises(ValueError):CartesianCoefficientFFTTranslation(coefficients,center,x,order)
    for kw in ({'retained_budget_bytes':1},{'tensor_workspace_budget_bytes':1}):
        with pytest.raises(MemoryError):CartesianCoefficientFFTTranslation(coefficients,center,scale,order,**kw)
    # The new arbitrary polynomial gate must not weaken the radial amplitude gate.
    def zero_coefficients(d,k,p):return torch.zeros(len(multi_indices(p)),device=d.device,dtype=torch.complex128)
    monkeypatch.setattr(shared,'helmholtz_taylor',zero_coefficients)
    with pytest.raises(ValueError,match='underflows'):shared.CartesianFFTTranslation(center+1,2.,order)


def test_generic_fft_requires_explicit_scale_even_with_nonzero_center(monkeypatch):
    center=torch.ones(3,device='cuda',dtype=torch.float64)
    coefficients=torch.ones(len(multi_indices(7)),device='cuda',dtype=torch.complex128)
    def forbidden(*a,**k):raise AssertionError('Generic FFT inferred a radial scale')
    monkeypatch.setattr(CartesianCoefficientFFTTranslation,'_prepare',forbidden)
    with pytest.raises(ValueError,match='explicit independent'):
        CartesianCoefficientFFTTranslation(coefficients,center,None,3)
