"""Actual screened Ewald coefficients preserve near ownership and PMCHWT channels."""
import json
import weakref
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.periodic import LatticeSpec,EwaldHelmholtzKernel
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.fmm_ewald_translation import EwaldRootM2L
from witwin.maxwell.mom.fmm_ewald_fft import EwaldFFTTerminalM2L
from witwin.maxwell.mom.fmm_expansion import monomials,multipole_moments
from witwin.maxwell.mom.fmm_storage import m2l_storage_bytes
from witwin.maxwell.mom.fmm_capacity import storage_bytes
from witwin.maxwell.mom.kernels import FREE_SPACE

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def specification(dimension):
    vectors=torch.tensor([[1.,.2,.1],[.1,1.2,.3],[.2,.1,.9]],device='cuda',dtype=torch.float64)[:dimension]
    lattice=LatticeSpec(vectors,.17*vectors.sum(0))
    kernel=EwaldHelmholtzKernel(lattice,2.,real_shells=3,reciprocal_shells=2,quadrature_order=64,image_batch=13)
    return kernel,compile_medium(mw.Material(),9e7)


@pytest.mark.parametrize('dimension,near_radius',[(1,.02),(2,.02),(3,.02),(1,.2)])
def test_actual_ewald_fft_original_compact_and_full_regular_kernel_channels(dimension,near_radius,tmp_path,monkeypatch):
    import witwin.maxwell.mom.fmm_ewald_fft as module
    kernel,medium=specification(dimension);reference=EwaldRootM2L(kernel,medium,near_radius,near_radius,order=6,row_batch=7)
    original=module.EwaldRootM2L;temporaries=[]
    def observed(*a,**k):
        obj=original(*a,**k);temporaries.append(weakref.ref(obj));return obj
    monkeypatch.setattr(module,'EwaldRootM2L',observed)
    op=EwaldFFTTerminalM2L(kernel,medium,near_radius,near_radius,order=6,row_batch=7)
    assert len(temporaries)==1 and temporaries[0]() is None
    assert torch.equal(op.near_indices,reference.near_indices)
    assert torch.equal(op.translation.scalar.displacement,torch.zeros(3,device='cuda',dtype=torch.float64))
    assert len(op.near_indices)>1 if near_radius==.2 else len(op.near_indices)==1
    owned=storage_bytes(op.tensors())
    assert owned==op.planned_retained_bytes==m2l_storage_bytes(6,8,'fft_v1')+op.near_indices.numel()*8
    assert all(t._base is None and t.untyped_storage().nbytes()==t.numel()*t.element_size() for t in op.tensors())
    x=kernel.lattice.vectors.new_tensor([[.003,-.002,.004],[-.004,.002,-.001]])
    y=kernel.lattice.vectors.new_tensor([[-.002,.001,.003],[.001,-.003,-.002]])
    tx,sy=monomials(x,6).to(op.dtype),monomials(y,6).to(op.dtype)
    fields=op.translation.scalar.apply(sy)
    actual=tx@fields[:,0,:].mT
    actual_gradient=torch.einsum('ib,jab->ija',tx,fields[:,1:,:])
    delta=x[:,None]-y[None];expected,expected_gradient=kernel.regular_part(delta,medium.k)
    for index in op.near_indices[(op.near_indices!=0).any(-1)]:
        g,h=FREE_SPACE.evaluate(delta-kernel.lattice.translations(index),medium.k)
        expected-=kernel.lattice.phase(index)*g;expected_gradient-=kernel.lattice.phase(index)*h
    torch.testing.assert_close(actual,expected,rtol=3e-9,atol=3e-12)
    torch.testing.assert_close(actual_gradient,expected_gradient,rtol=3e-9,atol=3e-12)
    generator=torch.Generator(device='cuda').manual_seed(728+dimension)
    charges=torch.randn((2,5,2,4,len(y)),device='cuda',dtype=op.dtype,generator=generator)
    current=multipole_moments(y,charges,6).movedim(-1,-2).transpose(0,1).detach().requires_grad_(True)
    weights=(.01**op.translation.scalar.indices.sum(-1))[:,None]
    saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t,lambda t:t):applied=op.apply(current)
    assert not saved
    original_action=reference.apply(current)
    error=float((((applied-original_action)*weights).norm()/(original_action*weights).norm()).detach())
    dual=torch.randn(current.shape,device='cuda',dtype=op.dtype,generator=generator)*weights
    adjoint=op.adjoint(dual);original_adjoint=reference.adjoint(dual)
    adjoint_error=float((((adjoint-original_adjoint)*weights).norm()/(original_adjoint*weights).norm()).detach())
    gradient,=torch.autograd.grad(applied,current,dual)
    gradient_error=float((gradient-adjoint).norm()/adjoint.norm())
    lhs=(applied.conj()*dual).sum();rhs=(current.conj()*adjoint).sum()
    dot=float(((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs())).detach())
    record=dict(dimension=dimension,near_radius=near_radius,green_relative_error=float((actual-expected).norm()/expected.norm()),
        gradient_relative_error=float((actual_gradient-expected_gradient).norm()/expected_gradient.norm()),
        forward_weighted_error=error,adjoint_weighted_error=adjoint_error,current_gradient_error=gradient_error,
        dot_error=dot,owned_bytes=owned,receipt=op.receipt(),physical_qualification=False)
    (tmp_path/'ewald.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
    assert error<1e-11 and adjoint_error<1e-11 and gradient_error<1e-11 and dot<1e-11
    op.near_indices.add_(1)
    with pytest.raises(ValueError,match='changed'):op.apply(current)


def test_ewald_fft_capacity_and_workspace_before_expansions_and_kernel_bindings(monkeypatch):
    import witwin.maxwell.mom.fmm_ewald_translation as original
    kernel,medium=specification(1)
    def forbidden(*a,**k):raise AssertionError('Rejected Ewald FFT setup reached expensive coefficients')
    with monkeypatch.context() as patch:
        patch.setattr(original,'ewald_regular_taylor',forbidden)
        # One 1D near index needs8B; old compact would fit this declared budget,
        # but the exact additional FFT inventory must fail BEFORE coefficients.
        with pytest.raises(MemoryError,match='before expansions'):
            EwaldFFTTerminalM2L(kernel,medium,.02,.02,order=6,retained_budget_bytes=m2l_storage_bytes(6,8,'fft_v1')+7)
        with pytest.raises(MemoryError,match='before expansions'):
            EwaldFFTTerminalM2L(kernel,medium,.02,.02,tensor_workspace_budget_bytes=1)
        with pytest.raises(ValueError,match='every possible near image'):
            EwaldFFTTerminalM2L(kernel,medium,10.,10.)
    op=EwaldFFTTerminalM2L(kernel,medium,.02,.02)
    value=torch.zeros((2,op.count,4),device='cuda',dtype=op.dtype)
    with pytest.raises(ValueError,match='bound medium'):op.action(compile_medium(mw.Material(),9e7),value)
    # Deliberately bypass the frozen declaration to attack the runtime binding.
    object.__setattr__(kernel,'real_shells',kernel.real_shells+1)
    with pytest.raises(ValueError,match='changed'):op.apply(value)
