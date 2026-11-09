"""Versioned FFT system integration against original weak and storage oracles."""
import copy
import hashlib
import json
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.fmm_fft import CartesianFFTTranslation, FFTElectromagneticM2L
from witwin.maxwell.mom.fmm_storage import m2l_storage_bytes
from witwin.maxwell.mom.fmm_capacity import storage_bytes, multipole_action_tensors
from witwin.maxwell.mom.fmm_system import MultipoleSystemActions
from witwin.maxwell.mom.fmm_expansion import multipole_to_local, multipole_moments
from witwin.maxwell.mom.scalar_moment_action import scalar_moment_action
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.operator import IntegrationConfig, assemble_system

RULES=('bivariate_cauchy_v1','real_support_min_v1','positive_tail_min_v1','first_wave_min_v1',
       'direct_positive_tail_min_v1','direct_first_wave_min_v1')


def test_fft_config_canonical_policy_and_rejects_every_old_alias():
    for rule in RULES:
        for shift in ('dense','separable_v1'):
            cfg=mw.FMMConfig(table_storage='fft_v1',near_storage='full_matrix',kernel_bound_rule=rule,shift_storage=shift,
                max_order=12,kernel_error_budget=mw.KernelErrorBudget(1e-6,1e-5))
            payload=cfg.payload()
            assert payload['version']=='cartesian_fmm_fft_config_v1'
            assert mw.FMMConfig.restore(payload)==cfg
            for field in payload['parameters']:
                bad=copy.deepcopy(payload);del bad['parameters'][field]
                with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
            old=replace(cfg,table_storage='compact').payload()
            old['parameters']['table_storage']='fft_v1'
            with pytest.raises(ValueError):mw.FMMConfig.restore(old)
            bad=copy.deepcopy(payload);bad['parameters']['table_storage']='compact'
            with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
    for params in ({'order':13},{'max_order':13,'kernel_error_budget':mw.KernelErrorBudget(1e-6,1e-5)}):
        with pytest.raises(ValueError):mw.FMMConfig(table_storage='fft_v1',near_storage='full_matrix',**params)
    assert m2l_storage_bytes(12,8,'fft_v1')==77928
    assert m2l_storage_bytes(12,8,'compact')==66040


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_fft_factory_batches_independent_storage_and_em_chunks(monkeypatch,tmp_path):
    import witwin.maxwell.mom.fmm_fft as module
    medium=compile_medium(mw.Material(eps_r=2.,mu_r=1.3,sigma_e=.001,sigma_m=.002),3e8)
    d=torch.tensor([[1.+i*.01,.2,-.1] for i in range(65)],device='cuda',dtype=torch.float64)
    calls=[];native=module.helmholtz_taylor
    def observed(points,*args,**kwargs):
        calls.append(points.shape);return native(points,*args,**kwargs)
    monkeypatch.setattr(module,'helmholtz_taylor',observed)
    objects=FFTElectromagneticM2L.create_many(d,medium,6,retained_budget_bytes=65*m2l_storage_bytes(6,8,'fft_v1'))
    assert calls==[torch.Size([32,3]),torch.Size([32,3]),torch.Size([1,3])]
    assert storage_bytes(t for obj in objects for t in obj.tensors())==65*m2l_storage_bytes(6,8,'fft_v1')
    for channel in range(8):
        values=[obj.tensors()[channel] for obj in objects]
        assert len({t.untyped_storage().data_ptr() for t in values})==65
        assert all(t._base is None and t.untyped_storage().nbytes()==t.numel()*t.element_size() for t in values)
    op=objects[32]
    green,gradient=multipole_to_local(d[32],medium.k,6)
    gradient=gradient.movedim(0,-1).contiguous()
    generator=torch.Generator(device='cuda').manual_seed(947)
    points=torch.randn((7,3),device='cuda',dtype=torch.float64,generator=generator)
    points=points/points.norm(dim=-1,keepdim=True)*.02
    charges=torch.randn((2,9,2,4,7),device='cuda',dtype=op.dtype,generator=generator)
    value=multipole_moments(points,charges,6).movedim(-1,-2).transpose(0,1).detach().requires_grad_(True)
    assert not value.is_contiguous()
    chunks=[];chunk=op._action_chunk
    def observed_chunk(v,**kw):
        chunks.append((len(v),kw['adjoint']));return chunk(v,**kw)
    monkeypatch.setattr(op,'_action_chunk',observed_chunk)
    saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t,lambda t:t):actual=op.apply(value)
    assert not saved
    expected=scalar_moment_action(green,gradient,medium,value)
    weight=(.02**op.scalar.indices.sum(-1))[:,None]
    error=float((((actual-expected)*weight).norm()/(expected*weight).norm()).detach())
    dual=torch.randn(actual.shape,device='cuda',dtype=op.dtype,generator=generator)*weight
    derivative,=torch.autograd.grad(actual,value,dual)
    adjoint=op.adjoint(dual)
    expected_adjoint=scalar_moment_action(green,gradient,medium,dual,adjoint=True)
    adjoint_error=float((((adjoint-expected_adjoint)*weight).norm()/(expected_adjoint*weight).norm()).detach())
    assert error<2e-11 and adjoint_error<2e-11
    torch.testing.assert_close(derivative,adjoint,rtol=0,atol=0)
    assert chunks==[(4,False)]*4+[(2,False)]+[(4,True)]*4+[(2,True)]+[(4,True)]*4+[(2,True)]
    assert op.apply(value[:0]).shape==value[:0].shape
    for index in range(8):
        objects[index].tensors()[index].add_(1)
        with pytest.raises(ValueError,match='changed'):objects[index].assert_current()
    for obj in objects[8:]:obj.assert_current()
    record=dict(forward_relative_error=error,adjoint_relative_error=adjoint_error,coefficient_batches=[32,32,1],
        physical_rhs_batch=4,receipt=op.receipt(),qualification=False)
    (tmp_path/'factory.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_fft_preallocation_native_guards_and_all_periodic_entrypoints(monkeypatch):
    import witwin.maxwell.mom.fmm_fft as module
    from witwin.maxwell.mom.fmm_periodic import PeriodicImageM2L
    from witwin.maxwell.mom.fmm_periodic_regional import PeriodicMultipoleRegionalOperator
    from witwin.maxwell.mom.fmm_periodic_system import PeriodicMultipoleSystemActions
    from witwin.maxwell.mom.linalg import StoppingRule
    def forbidden(*a,**k):raise AssertionError('Rejected FFT construction reached coefficients')
    monkeypatch.setattr(module,'helmholtz_taylor',forbidden)
    d=torch.tensor([[1.,.2,-.1]]*65,device='cuda',dtype=torch.float64)
    for value,order,kw,exc in ((d,6,{'retained_budget_bytes':1},MemoryError),
        (d,6,{'tensor_workspace_budget_bytes':1},MemoryError),(d.float(),6,{},ValueError),(d,13,{},ValueError)):
        with pytest.raises(exc):CartesianFFTTranslation.create_many(value,2.,order,**kw)
    assert CartesianFFTTranslation.create_many(d[:0],2.,6)==()
    with pytest.raises(NotImplementedError,match='free space'):
        mw.MoMConfig(operator_backend='fmm',fmm=mw.FMMConfig(table_storage='fft_v1'),riesz_storage='local_rwg_v1',
            periodic=mw.MoMPeriodicConfig(((1.,0.,0.),)),stopping=StoppingRule(preconditioner='rwg_diagonal'))
    from witwin.maxwell.mom.fmm_periodic_order import plan_periodic_order, plan_periodic_parameters
    from witwin.maxwell.mom.fmm_periodic_capacity import plan_periodic_storage
    for func,kw in ((plan_periodic_order,dict(min_order=6,max_order=8,shells=2,exterior_budget=None,interior_budget=None)),
                    (plan_periodic_parameters,dict(min_shells=1,max_shells=2)),
                    (plan_periodic_storage,dict(order=6,shells=2,separation_ratio=.5,budget_bytes=1024))):
        with pytest.raises(NotImplementedError,match='Periodic FFT'):func(None,None,None,table_storage='fft_v1',**kw)
    for cls,args in ((PeriodicImageM2L,(None,None,0.,0.,2.)),
                     (PeriodicMultipoleRegionalOperator,(None,None,None,None)),
                     (PeriodicMultipoleSystemActions,(None,None))):
        with pytest.raises(NotImplementedError,match='Periodic FFT'):cls(*args,table_storage='fft_v1')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_fft_system_automatic_order_original_all_columns_capacity_and_fixed_gradient(tmp_path):
    exterior=compile_medium(mw.Material(),3e8)
    bodies=[(i,str(i),RWGBasis(sphere_mesh(radius=.01,center=(x,0.,0.),subdivisions=0,device='cuda')),
        compile_medium(mw.Material(eps_r=2.+i,mu_r=1.2+i*.1),3e8)) for i,x in enumerate((-.5,.5))]
    system=SurfaceSystem.create(bodies);integration=IntegrationConfig(order=3,singular_order=3)
    provider=ScalarSystemBlocks(system,exterior,integration)
    op=MultipoleSystemActions(provider,order=6,max_order=8,kernel_error_budget=mw.KernelErrorBudget(1e-6,1e-5),
        leaf_size=30,table_storage='fft_v1',shift_storage='separable_v1',kernel_bound_rule='direct_first_wave_min_v1')
    assert len(op.exterior.far)==2 and op.order_plan['attempts'][-1]['kernel_pass']
    actual_bytes=storage_bytes(multipole_action_tensors(op))
    assert actual_bytes==op.capacity_plan['planned_bytes']
    assert op.table_bytes==op.order_plan['table_bytes']==sum(c.receipt()['table_bytes'] for _,_,c in op.exterior.far)
    assert all(c.receipt()['table_storage']=='fft_v1' for _,_,c in op.exterior.far)
    assert op.exterior.receipt()['m2l_construction']=='fft_taylor_coefficients_v1'
    dense,_=assemble_system(system,exterior,integration)
    n=system.total_edges;actual=op.apply(torch.eye(2*n,device='cuda',dtype=op.dtype)).mT
    eta=abs(exterior.eta);left=torch.tensor([eta]*n+[1.]*n,device='cuda');right=torch.tensor([1.]*n+[eta]*n,device='cuda')
    error=float((left[:,None]*(actual-dense)*right).norm()/(left[:,None]*dense*right).norm())
    generator=torch.Generator(device='cuda').manual_seed(928)
    x=torch.randn((3,2*n),device='cuda',dtype=op.dtype,generator=generator,requires_grad=True)
    y=torch.randn(x.shape,device='cuda',dtype=op.dtype,generator=generator)
    applied=op.apply(x);adjoint=op.adjoint(y);expected=y@actual.conj()
    derivative,=torch.autograd.grad(applied,x,y)
    adjoint_error=float((adjoint-expected).norm()/expected.norm())
    gradient_error=float((derivative-adjoint).norm()/adjoint.norm())
    lhs=(applied.conj()*y).sum();rhs=(x.conj()*adjoint).sum()
    dot=float(((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs())).detach())
    record=dict(balanced_original_all_column_error=error,adjoint_error=adjoint_error,gradient_error=gradient_error,
        dot_error=dot,actual_owned_bytes=actual_bytes,table_bytes=op.table_bytes,order_plan=op.order_plan,
        capacity_plan=op.capacity_plan,physical_qualification=False,complexity_qualification=False)
    (tmp_path/'system.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
    assert error<1e-8 and adjoint_error<1e-11 and gradient_error<1e-11 and dot<1e-11
    op.exterior.far[0][2].table_storage='compact'
    with pytest.raises(ValueError,match='changed'):op.apply(x)


# Canonical byte hashes for the epoch with explicit storage policies and budgets.
STORAGE_HASHES={
('bivariate_cauchy_v1', 'dense'): 'a3daaa4ec4dc3e875f1b9a2839b9437586db0cb98c27b3e44800f4dd44d719be',
('bivariate_cauchy_v1', 'separable_v1'): '32d73c65914672f70e142e73763fb99d01e122abe1e83a60db3306583fe96ef1',
('real_support_min_v1', 'dense'): 'ac2cdc3f876450c00e92dc1db5e7f0fdddaa01883a44addccb3f1d6271db4bcb',
('real_support_min_v1', 'separable_v1'): 'c9652ff248d4aac323ab5a19e1a4d9c133c8a91ccee38d7a4874e83325323f59',
('positive_tail_min_v1', 'dense'): '9caac1ac146afb3c32ee032dab8b15013bc62d24953b199bfb899d0c5118dc15',
('positive_tail_min_v1', 'separable_v1'): 'c59d2ce921991d7ef82f5a3c7a14301e7a3580001e44c9f93123a009ad3b98d4',
('first_wave_min_v1', 'dense'): '4d7b1b17ded173ba1c8906a7f8d0b125a8b09b66ab88d6b35c185e09e7f133a4',
('first_wave_min_v1', 'separable_v1'): '53f05c0bd94c2e3bb2faead4fa2c21a6421c8b55e89c1105e63ffa7afeb68350',
('direct_positive_tail_min_v1', 'dense'): '8c3e8bb57885ec477b60395de5e47304f77356a6a30884499a826b08529ab578',
('direct_positive_tail_min_v1', 'separable_v1'): '0a7f63059fc35255ad5f935e2b8e6a9d8bef4177a9d485fd84022b060b7f0bd7',
('direct_first_wave_min_v1', 'dense'): 'a435bcba7b92f1f428298bd6b0434ebeae72f922c799ed3d163020522cfa1a2e',
('direct_first_wave_min_v1', 'separable_v1'): '5422976ab16055f6e1212353fa5a37759eb551a957c624b81e3bbd8d1d0cc9a9',
}

def test_fft_integration_explicit_storage_configuration_identities():
    for (rule,shift),expected in STORAGE_HASHES.items():
        cfg=mw.FMMConfig(table_storage='dense',near_storage='full_matrix',kernel_bound_rule=rule,shift_storage=shift)
        data=json.dumps(cfg.payload(),sort_keys=True,separators=(",",":")).encode()
        assert hashlib.sha256(data).hexdigest()==expected
        assert mw.FMMConfig.restore(cfg.payload())==cfg
