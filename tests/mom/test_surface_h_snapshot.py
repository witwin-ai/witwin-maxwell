"""Exact scalar surface H/H2 snapshots retain physical Krylov/metric state."""
import copy
import io
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis,SurfaceSystem,sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.hmatrix import HierarchicalOperator,HMatrixConfig
from witwin.maxwell.mom.h2_operator import HierarchicalH2Operator
from witwin.maxwell.mom.h2 import NestedH2Actions
from witwin.maxwell.mom.surface_h_snapshot import encode,restore
from witwin.maxwell.mom.implicit_surface_solve import FixedSurfaceInverse
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.volume_h_snapshot import _digest

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def provider(epsilon=3.,dtype=torch.float64):
    bodies=[]
    for i,x in enumerate((-.5,.5)):
        basis=RWGBasis(sphere_mesh(radius=.01,center=(x,0.,0.),subdivisions=0,device='cuda',dtype=dtype))
        bodies.append((i,str(i),basis,compile_medium(mw.Material(eps_r=2. if i==0 else epsilon),3e8)))
    return ScalarSystemBlocks(SurfaceSystem.create(bodies),compile_medium(mw.Material(),3e8),IntegrationConfig(order=3,singular_order=3))


def test_actual_surface_h_h2_restore_actions_metric_and_implicit_solve(monkeypatch,tmp_path):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    previous=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        source=HierarchicalOperator(provider(),HMatrixConfig(pivoting='adaptive'))
        assert source.metadata['far_blocks'] and source.metadata['exact_blocks']
        for original in (source,HierarchicalH2Operator(source)):
            payload=encode(original);torch.save(payload,tmp_path/(payload['kind']+'.pt'))
            stream=io.BytesIO();torch.save(payload,stream);stream.seek(0)
            payload=torch.load(stream,weights_only=True)
            fresh=provider()
            def forbidden(*args,**kwargs):raise AssertionError('Restore cannot sample, assemble, convert or solve')
            with monkeypatch.context() as patch:
                for cls in (HierarchicalOperator,HierarchicalH2Operator,NestedH2Actions):patch.setattr(cls,'__init__',forbidden)
                patch.setattr(ScalarSystemBlocks,'__call__',forbidden)
                for name in ('solve','svd','qr'):patch.setattr(torch.linalg,name,forbidden)
                restored=restore(payload,fresh)
            assert restored.provider.system is not original.provider.system
            assert encode(restored)['sha256']==payload['sha256']
            check_budget_and_metric_rejections(payload,fresh)
            x=torch.randn(2,original.shape[0],device='cuda',dtype=original.dtype,requires_grad=True)
            y=torch.randn_like(x)
            torch.testing.assert_close(restored.apply(x),original.apply(x),rtol=0,atol=0)
            torch.testing.assert_close(restored.adjoint(y),original.adjoint(y),rtol=0,atol=0)
            torch.testing.assert_close(restored.metric.apply(x),original.metric.apply(x),rtol=0,atol=0)
            for actual,expected in zip(restored.preconditioner_scales('row_column_norm')[:2],original.preconditioner_scales('row_column_norm')[:2]):
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            rule=StoppingRule(method='block_gmres',preconditioner='row_column_norm',rtol=1e-11,atol=1e-13,restart=120,stagnation_window=240,maxiter=240)
            actual=FixedSurfaceInverse(restored,rule).apply(x);expected=FixedSurfaceInverse(original,rule).apply(x)
            torch.testing.assert_close(actual,expected,rtol=2e-8,atol=1e-9)
            ga,=torch.autograd.grad((actual*y.conj()).real.sum(),x)
            gb,=torch.autograd.grad((expected*y.conj()).real.sum(),x)
            torch.testing.assert_close(ga,gb,rtol=2e-8,atol=1e-9)
            bad=copy.deepcopy(payload);bad['metric']['diagonal'][0]*=2
            with pytest.raises(ValueError,match='checksum'):restore(bad,fresh)
            bad=copy.deepcopy(payload);bad['metric']['records'][0][3][0,0,0]+=1
            bad['sha256']=_digest({k:v for k,v in bad.items() if k!='sha256'})
            with pytest.raises(ValueError,match='Gram'):restore(bad,fresh)
            with pytest.raises(ValueError,match='physical binding'):restore(payload,provider(4.))
            delayed=FixedSurfaceInverse(restored,rule).apply(x)
            with torch.no_grad():restored.metric.diagonal[0]*=2
            with pytest.raises(ValueError,match='changed'):torch.autograd.grad(delayed.real.sum(),x)
    finally:torch.use_deterministic_algorithms(previous,warn_only=warn)


def check_budget_and_metric_rejections(payload,fresh):
    def seal(value):
        value['sha256']=_digest({k:v for k,v in value.items() if k!='sha256'})
        return value
    bad=copy.deepcopy(payload);bad['metric']['diagonal'][0]*=2
    with pytest.raises(ValueError,match='diagonal'):restore(seal(bad),fresh)
    bad=copy.deepcopy(payload)
    key='source_configuration' if payload['kind']=='h2' else 'configuration'
    bad[key]['parameters']['retained_budget_bytes']=1
    with pytest.raises(MemoryError,match='budget'):restore(seal(bad),fresh)
    if payload['kind']=='h2':
        bad=copy.deepcopy(payload)
        bad['configuration']['parameters']['retained_budget_bytes']=1
        bad['data']['configuration']['retained_budget_bytes']=1
        with pytest.raises(MemoryError,match='budget'):restore(seal(bad),fresh)


def test_complex64_all_near_surface_snapshots_and_budget_rejections(monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    enabled=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        original=HierarchicalOperator(provider(dtype=torch.float32),HMatrixConfig(leaf_size=128))
        for op in (original,HierarchicalH2Operator(original)):
            payload=encode(op);fresh=provider(dtype=torch.float32)
            restored=restore(payload,fresh)
            assert encode(restored)['sha256']==payload['sha256']
            x=torch.randn(2,op.shape[0],device='cuda',dtype=torch.complex64,requires_grad=True)
            torch.testing.assert_close(restored.apply(x),op.apply(x),rtol=0,atol=0)
            torch.testing.assert_close(restored.adjoint(x),op.adjoint(x),rtol=0,atol=0)
            ga,=torch.autograd.grad(restored.apply(x).real.sum(),x)
            gb,=torch.autograd.grad(op.apply(x).real.sum(),x)
            torch.testing.assert_close(ga,gb,rtol=0,atol=0)
            check_budget_and_metric_rejections(payload,fresh)
            check_inverse_policy_roundtrip(FixedSurfaceInverse(op,StoppingRule(method='block_gmres',
                preconditioner='row_column_norm',rtol=3e-5,atol=2e-9),refinement_steps=3),fresh,monkeypatch)
    finally:torch.use_deterministic_algorithms(enabled,warn_only=warn)


def check_inverse_policy_roundtrip(inverse,fresh,monkeypatch):
    from witwin.maxwell.mom.surface_h_snapshot import encode_inverse,restore_inverse
    saved=encode_inverse(inverse)
    def forbidden(*args,**kwargs):raise AssertionError('Inverse restore must not solve')
    with monkeypatch.context() as patch:
        patch.setattr(FixedSurfaceInverse,'_solve',forbidden)
        loaded=restore_inverse(saved,fresh)
    assert loaded.rule==inverse.rule and loaded.refinement_steps==inverse.refinement_steps
    assert loaded.receipt==inverse.receipt
    assert encode_inverse(loaded)['sha256']==saved['sha256']
    bad=copy.deepcopy(saved);bad['stopping']['method']='direct'
    bad['sha256']=_digest({k:v for k,v in bad.items() if k!='sha256'})
    with pytest.raises(ValueError,match='Krylov'):restore_inverse(bad,fresh)
    return loaded
