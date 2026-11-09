"""Strict proportional SPD spatial kernels against independent Maxwell routes."""
import math
import os
import time
import pytest
import torch
import witwin.maxwell.mom.dyadic_green as dg
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.matched_spd_green import proportional_spd_spatial_green

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def medium(dtype=torch.complex128):
    coefficient=torch.tensor([[2.,.25,.125],[.25,3.,.375],[.125,.375,4.]],device='cuda',dtype=dtype)
    return TensorMedium(3e7,coefficient,.5*coefficient)


def kernel(m):
    return dg.DyadicGreenKernel(m,dg.DyadicIntegrationConfig(spatial_rule='proportional_spd_closed_form_v1'))


@pytest.mark.parametrize('dtype',(torch.complex64,torch.complex128))
def test_full_regular_static_and_independent_fourier_identity(dtype):
    m=medium(dtype);selected=kernel(m)
    x=m.epsilon_r.real.new_tensor([[.12,.03,.07],[1.2e-5,3e-6,7e-6]])
    full,record=selected.evaluate(x);regular,other=selected.evaluate_regular(x)
    assert record['kernel_version']==other['kernel_version']=='analytic_proportional_spd_spatial_v1'
    assert record['quadrature']=='closed_form_no_spectral_quadrature'
    assert not record['physical_qualification'] and not record['constitutive_projection']
    assert selected.mode_cache.modal_matrices_built==0
    difference=full-regular
    tol=512*torch.finfo(x.dtype).eps
    for section,kind,scale in ((slice(0,3),'electric',dg.EPSILON_0),(slice(3,6),'magnetic',dg.MU_0)):
        expected=1j*dg.electrostatic_kernel(x,m,constitutive=kind)[2]/(m.omega*scale)
        torch.testing.assert_close(difference[:,section,section],expected,rtol=tol,atol=0)
    torch.testing.assert_close(difference[:,:3,3:],torch.zeros_like(difference[:,:3,3:]),rtol=0,atol=0)
    # The full 6x6 inverse Maxwell symbol supplies independent source signs,
    # electric/magnetic scaling and the Fourier change-of-variable Jacobian.
    # A symmetric square root differs from the production Cholesky frame.
    eigenvalues,eigenvectors=torch.linalg.eigh(m.epsilon_r.real)
    transform=(eigenvectors*eigenvalues.sqrt())@eigenvectors.T/eigenvalues.prod().sqrt()
    inverse=torch.linalg.inv(transform)
    dual=torch.block_diag(inverse,inverse).to(dtype)
    k0=m.omega*math.sqrt(dg.EPSILON_0*dg.MU_0)
    wavevectors=(k0*x.new_tensor([[.2,.1,.3],[4.,3.,2.]])).to(dtype)
    eye=torch.eye(3,device=x.device,dtype=dtype)
    scalar=TensorMedium(m.frequency,eye,.5*eye)
    actual=dg.current_green_spectrum(wavevectors,m)
    expected=torch.linalg.det(transform)*dual.T@dg.current_green_spectrum(wavevectors@transform.to(dtype),scalar)@dual
    torch.testing.assert_close(actual,expected,rtol=tol,atol=tol*float(expected.abs().amax()))


def test_exact_proportional_boundaries_queries_and_ownership(monkeypatch):
    m=medium();x=m.epsilon_r.real.new_tensor([[.12,.03,.07]])
    changed=m.mu_r.clone();changed[0,0]=torch.nextafter(changed[0,0].real,changed[0,0].real.new_tensor(float('inf')))
    for other in (TensorMedium(m.frequency,m.epsilon_r,changed),
            TensorMedium(m.frequency,m.epsilon_r*(1+.001j),m.mu_r),
            TensorMedium(m.frequency,m.epsilon_r.clone().requires_grad_(),m.mu_r),
            TensorMedium(m.frequency,-m.epsilon_r,-m.mu_r)):
        assert proportional_spd_spatial_green(x,other) is None
    fallback=[]
    def sentinel(*args,**kwargs):fallback.append(True);return 'original_path',{}
    monkeypatch.setattr(dg,'evaluate_spatial_green',sentinel)
    assert kernel(TensorMedium(m.frequency,m.epsilon_r,changed)).evaluate_regular(x)[0]=='original_path'
    assert fallback==[True]
    selected=kernel(m)
    for bad in (x.cpu(),x.float(),x[0],x[:,:2],x*float('nan')):
        with pytest.raises(ValueError,match='queries'):
            selected.evaluate(bad)
    with pytest.raises(ValueError,match='Coincident'):
        selected.evaluate(torch.zeros_like(x))
    m.epsilon_r.add_(.1)
    with pytest.raises(ValueError,match='changed'):
        selected.evaluate(x)


def test_proportional_spatial_native_query_gradient():
    m=medium();selected=kernel(m)
    x=m.epsilon_r.real.new_tensor([[.12,.03,.07]]).requires_grad_()
    assert torch.autograd.gradcheck(lambda value:selected.evaluate_regular(value)[0],(x,),
        eps=1e-6,atol=1e-5,rtol=1e-4,fast_mode=True)


def test_proportional_spatial_against_actual_weyl_quadrature(tmp_path):
    m=medium();x=m.epsilon_r.real.new_tensor([[.12,.03,.07],[.07,-.04,.11]])
    options=dg.DyadicIntegrationConfig(radial_map='proportional_spd_branch_v1',
        relative_tolerance=1e-5,max_radial_order=256,max_azimuthal_order=128)
    torch.cuda.synchronize();start=time.monotonic()
    actual,closed=kernel(m).evaluate_regular(x)
    torch.cuda.synchronize();closed_seconds=time.monotonic()-start
    start=time.monotonic()
    # Deliberately bypass kernel specialization: retain the actual Weyl route.
    expected,receipt=dg.evaluate_spatial_green(x,m,options,regular_part=True)
    torch.cuda.synchronize();weyl_seconds=time.monotonic()-start
    errors=[]
    for i in (0,3):
        for j in (0,3):
            left,right=actual[:,i:i+3,j:j+3],expected[:,i:i+3,j:j+3]
            error=(left-right).norm(dim=(-2,-1))/right.norm(dim=(-2,-1))
            errors.extend(error.tolist())
    torch.save(dict(closed=actual,weyl=expected,epsilon_r=m.epsilon_r,mu_r=m.mu_r,
        queries=x,errors=errors,closed_receipt=closed,weyl_receipt=receipt,
        closed_seconds=closed_seconds,weyl_seconds=weyl_seconds),tmp_path/'point-comparison.pt')
    assert all(math.isfinite(error) and error<.005 for error in errors)
    assert receipt['converged'] and receipt['kernel_version']=='dyadic_weyl_proportional_spd_branch_v1'
    print({'block_errors':errors,'closed_seconds':closed_seconds,'weyl_seconds':weyl_seconds,
        'concurrent_gpu_jobs':True,'complete_solve_speedup_claimed':False,'physical_qualification':False},flush=True)


def test_versioned_spatial_rule_preserves_default_route(monkeypatch):
    from witwin.maxwell.mom.runtime import _numerical_identity
    from witwin.maxwell.mom.tensor_harmonic_bundle import _config
    from witwin.maxwell.mom.linalg import StoppingRule
    import witwin.maxwell as mw
    default=dg.DyadicIntegrationConfig()
    assert 'spatial_rule' not in default.payload()
    selected=dg.DyadicIntegrationConfig(spatial_rule='proportional_spd_closed_form_v1')
    assert dg.DyadicIntegrationConfig(**selected.payload())==selected
    config=mw.MoMConfig(dyadic_integration=selected,stopping=StoppingRule(method='direct'))
    numerical=_numerical_identity(config)
    assert _numerical_identity(_config(numerical))==numerical
    m=medium();x=m.epsilon_r.real.new_tensor([[.12,.03,.07]])
    reached=[]
    def sentinel(*args,**kwargs):reached.append(True);return 'original_path',{}
    with monkeypatch.context() as patch:
        patch.setattr(dg,'evaluate_spatial_green',sentinel)
        assert dg.DyadicGreenKernel(m,default).evaluate_regular(x)[0]=='original_path'
    assert reached==[True]
    with pytest.raises(ValueError,match='spatial rule'):
        dg.DyadicIntegrationConfig(spatial_rule='unknown')


def test_saved_proportional_point_values_replay_exactly():
    import hashlib
    from pathlib import Path
    source=os.environ.get('GSP_PROPORTIONAL_POINT_COMPARISON')
    if not source:pytest.skip('Completed independent point artifact required')
    # Reuse completed independent Weyl data without repeating its integration.
    saved=torch.load(source,weights_only=True)
    m=medium()
    actual,receipt=kernel(m).evaluate_regular(saved['queries'].to(m.epsilon_r.device))
    torch.testing.assert_close(actual,saved['closed'].to(actual.device),rtol=0,atol=0)
    assert receipt==saved['closed_receipt']
    print(dict(exact_saved_point_replay=True,artifact=source,
        artifact_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest()),flush=True)


def test_versioned_spatial_rule_periodic_payload_identity_and_validation():
    import copy
    from dataclasses import replace
    from test_periodic_dyadic_persistence import problem
    from witwin.maxwell.mom.periodic_dyadic import QuasiperiodicDyadicRegularKernel
    original=problem(2)
    original=replace(original,integration=replace(original.integration,
        spatial_rule='proportional_spd_closed_form_v1'))
    payload=original.payload()
    restored=QuasiperiodicDyadicRegularKernel.from_payload(payload,medium=original.medium,lattice=original.lattice)
    assert restored.payload()==payload and restored.identity==original.identity
    invalid=copy.deepcopy(payload);invalid['integration']['spatial_rule']='unknown'
    with pytest.raises(ValueError,match='spatial_rule'):
        QuasiperiodicDyadicRegularKernel.from_payload(invalid,medium=original.medium,lattice=original.lattice)
