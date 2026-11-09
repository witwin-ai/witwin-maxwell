"""Actual Green/gradient and fixed-current FFT translation acceptance."""
import json
import pytest
import torch
from witwin.maxwell.mom.fmm_fft import CartesianFFTTranslation
from witwin.maxwell.mom.fmm_expansion import multipole_to_local,monomials,multipole_moments

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('order',(0,3,6,11,12))
@pytest.mark.parametrize('wave',(6.3,6.3+.4j))
def test_fft_translation_physical_moments_all_fields_dense_and_adjoint(order,wave,tmp_path):
    d=torch.tensor([1.,.2,-.1],device='cuda',dtype=torch.float64)
    op=CartesianFFTTranslation(d,wave,order,rhs_batch=2)
    green,gradient=multipole_to_local(d,wave,order)
    matrix=torch.cat((green[None],gradient),dim=0)
    generator=torch.Generator(device='cuda').manual_seed(712)
    points=torch.randn((9,3),device='cuda',dtype=torch.float64,generator=generator)
    points=points/points.norm(dim=-1,keepdim=True)*.05
    charges=torch.randn((2,3,9),device='cuda',dtype=torch.complex128,generator=generator)
    moments=multipole_moments(points,charges,order).detach().requires_grad_(True)
    expected=torch.einsum('fab,...b->...fa',matrix,moments)
    actual=op.apply(moments)
    degrees=op.indices.sum(-1);weight=.05**degrees
    relative=float(((actual-expected)*weight).norm()/(expected*weight).norm())
    # Dual data represents target polynomial evaluation, avoiding artificial
    # huge-SI-moment vectors that do not arise from bounded physical supports.
    dual=(torch.randn(actual.shape,device='cuda',dtype=actual.dtype,generator=generator)*weight).detach()
    reference_adjoint=torch.einsum('fab,...fa->...b',matrix.conj(),dual)
    adjoint=op.adjoint(dual)
    source_weight=.05**degrees
    adjoint_relative=float(((adjoint-reference_adjoint)*source_weight).norm()/(reference_adjoint*source_weight).norm())
    grad,=torch.autograd.grad(actual,moments,dual)
    derivative_relative=float(((grad-adjoint)*source_weight).norm()/(adjoint*source_weight).norm())
    lhs=(actual.conj()*dual).sum();rhs=(moments.conj()*adjoint).sum()
    dot=float((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs()).clamp_min(1e-300))
    record=dict(order=order,wave=[complex(wave).real,complex(wave).imag],forward_relative_error=relative,
        adjoint_relative_error=adjoint_relative,current_gradient_relative_error=derivative_relative,
        dot_identity_relative_error=dot,receipt=op.receipt(),physical_qualification=False)
    (tmp_path/'fft.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
    assert relative<2e-11 and adjoint_relative<2e-11
    assert dot<1e-11 and derivative_relative<1e-13
    assert op.apply(moments[:0]).shape==(0,3,4,op.count)
    assert op.adjoint(dual[:0]).shape==(0,3,op.count)


def test_fft_translation_rejection_and_fixed_state_before_coefficients(monkeypatch):
    import witwin.maxwell.mom.fmm_fft as module
    d=torch.tensor([1.,.2,-.1],device='cuda',dtype=torch.float64)
    native=module.helmholtz_taylor
    def forbidden(*args,**kwargs):raise AssertionError('Rejected FFT admission allocated coefficients')
    with monkeypatch.context() as patch:
        patch.setattr(module,'helmholtz_taylor',forbidden)
        for changed in ({'order':13},{'rhs_batch':False}):
            with pytest.raises(ValueError):CartesianFFTTranslation(d,6.3,**{'order':6,**changed})
        for changed in ({'retained_budget_bytes':1},{'tensor_workspace_budget_bytes':1}):
            with pytest.raises(MemoryError):CartesianFFTTranslation(d,6.3,6,**changed)
        for value in (d.float(),d.cpu(),d*torch.nan,d.clone().requires_grad_(),d*0):
            with pytest.raises(ValueError):CartesianFFTTranslation(value,6.3,6)
    op=CartesianFFTTranslation(d,6.3,6)
    assert op.retained_bytes==sum(t.numel()*t.element_size() for t in op.tensors())
    x=torch.zeros(op.count,device='cuda',dtype=op.dtype)
    for value in (x.float(),x.cpu(),x*torch.nan,x[:-1]):
        with pytest.raises(ValueError):op.apply(value)
    op.target_weights.add_(1)
    with pytest.raises(ValueError,match='changed'):op.apply(x)
    # Positive normalization/kernel amplitudes cannot silently become zeros.
    with pytest.raises(ValueError,match='underflows'):
        CartesianFFTTranslation(d,1e4j,6)


def test_fft_translation_saved_original_p12_geometry_full_kernel(tmp_path):
    import os,math,hashlib
    from pathlib import Path
    from witwin.maxwell.mom.fmm_error import multipole_truncation_bound
    required=('GSP_FFT_FMM_ADMISSION','GSP_FFT_FMM_ADMISSION_SHA256')
    if all(name not in os.environ for name in required):pytest.skip('Original growing-ka PASS artifact required')
    assert all(name in os.environ for name in required),'Incomplete original admission environment'
    path=Path(os.environ[required[0]]);digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest==os.environ[required[1]]
    admission=json.loads(path.read_text());assert admission['order_admitted'] and admission['edges']==3840
    assert admission['family']=='growing_ka' and admission['order_plan']['selected_order']==12
    witness=admission['support_pair_geometry_witnesses']['gradient_component_absolute'][-1]
    assert witness['order']==12 and witness['available']
    d=torch.tensor(witness['displacement'],device='cuda',dtype=torch.float64)
    a=d.new_tensor(witness['test_radius']);b=d.new_tensor(witness['source_radius']);wave=complex(*witness['wave_number'])
    op=CartesianFFTTranslation(d,wave,12,rhs_batch=4)
    generator=torch.Generator(device='cuda').manual_seed(1729)
    directions=torch.randn((64,2,3),generator=generator,device='cuda',dtype=torch.float64)
    directions/=directions.norm(dim=-1,keepdim=True)
    directions[:4]=d.new_tensor([[[1,0,0],[1,0,0]],[[1,0,0],[-1,0,0]],
        [[0,1,0],[0,0,1]],[[-1,0,0],[1,0,0]]])
    x,y=directions[:,0]*a,directions[:,1]*b
    separation=d+x-y;radius=separation.norm(dim=-1)
    exact=torch.exp(1j*wave*radius)/(4*math.pi*radius)
    exact_gradient=exact[:,None]*(1j*wave-1/radius)[:,None]*separation/radius[:,None]
    tx,sy=monomials(x,12).to(op.dtype),monomials(y,12).to(op.dtype)
    local=op.apply(sy);approximate=torch.einsum('np,nfp->nf',tx,local)
    error_g=float((approximate[:,0]-exact).abs().max())
    error_h=float((approximate[:,1:]-exact_gradient).abs().max())
    green,gradient=multipole_to_local(d,wave,12)
    dense_g=torch.einsum('ni,ij,nj->n',tx,green,sy)
    dense_h=torch.einsum('ni,aij,nj->na',tx,gradient,sy)
    native_g=float((approximate[:,0]-dense_g).abs().max())
    native_h=float((approximate[:,1:]-dense_h).abs().max())
    bound=multipole_truncation_bound(d,a,b,wave,12,kernel_bound_rule='direct_first_wave_min_v1')
    record=dict(admission_sha256=digest,sampled_green_error=error_g,sampled_gradient_error=error_h,
        fft_dense_green_absolute_difference=native_g,fft_dense_gradient_absolute_difference=native_h,
        bound=bound,scope='64 points on one saved original geometry; not full-system qualification',
        physical_qualification=False,complexity_qualification=False)
    (tmp_path/'witness.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
    assert error_g<=bound['green_absolute'] and error_h<=bound['gradient_component_absolute']
    assert native_g<1e-10 and native_h<1e-8
