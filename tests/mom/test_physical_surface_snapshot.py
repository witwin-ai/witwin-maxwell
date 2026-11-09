"""Physical surface snapshots preserve declared coherent carrier bindings."""
import copy
import io
import runpy
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.hierarchical_runtime import HierarchicalSystemOperator,HierarchicalH2SystemOperator
from witwin.maxwell.mom.hmatrix import HierarchicalOperator,HMatrixConfig
from witwin.maxwell.mom.h2_operator import HierarchicalH2Operator
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.surface_h_snapshot import encode,restore,encode_inverse,restore_inverse
from witwin.maxwell.mom.implicit_surface_solve import FixedSurfaceInverse
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.fields import FieldQueryConfig
from witwin.maxwell.mom.lifting_fields import decode_system
from witwin.maxwell.mom.volume_h_snapshot import _digest

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.fixture(autouse=True)
def deterministic_actions(monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    enabled=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:yield
    finally:torch.use_deterministic_algorithms(enabled,warn_only=warn)


@pytest.mark.parametrize('backend,kind',[('hmatrix',HierarchicalSystemOperator),('h2',HierarchicalH2SystemOperator)])
def test_physical_carrier_inverse_and_decoder_restore(backend,kind,monkeypatch):
    build=runpy.run_path('tests/mom/test_surface_volume.py')['problem']
    dense,_=build((2.,3.));fresh_dense,_=build((2.,3.))
    sources=(mw.PlaneWave(source_time=mw.CW(3e8,amplitude=1.2,phase=.2)),
        mw.PlaneWave(direction=(1,0,0),polarization=(0,1,0),source_time=mw.CW(3e8,amplitude=.8,phase=-.3)))
    config=mw.MoMConfig(operator_backend=backend,integration=dense.integration,
        hierarchical=HMatrixConfig(leaf_size=64),riesz_storage='local_rwg_v1',
        stopping=StoppingRule(method='block_gmres',preconditioner='row_column_norm'))
    original=kind(dense.system,dense.exterior,sources,config)
    original.build_rhs()
    payload=encode(original)
    buffer=io.BytesIO();torch.save(payload,buffer);buffer.seek(0);payload=torch.load(buffer,weights_only=True)
    fresh=ScalarSystemBlocks(fresh_dense.system,fresh_dense.exterior,fresh_dense.integration)
    def forbidden(*args,**kwargs):raise AssertionError('Physical restore cannot assemble, sample or solve')
    with monkeypatch.context() as patch:
        for cls in (HierarchicalOperator,HierarchicalH2Operator,HierarchicalSystemOperator,HierarchicalH2SystemOperator):patch.setattr(cls,'__init__',forbidden)
        patch.setattr(ScalarSystemBlocks,'__call__',forbidden)
        patch.setattr(FixedSurfaceInverse,'_solve',forbidden)
        restored=restore(payload,fresh,sources=sources)
    assert type(restored) is kind and restored.system is fresh.system
    assert restored.incident_carrier.identity==original.incident_carrier.identity
    assert encode(restored)['sha256']==payload['sha256']
    torch.testing.assert_close(restored.build_rhs(),original.build_rhs(),rtol=0,atol=0)
    rule=StoppingRule(method='block_gmres',preconditioner='row_column_norm',rtol=1e-11,atol=1e-13,stagnation_window=48)
    inverse=FixedSurfaceInverse(original,rule,refinement_steps=3)
    saved=encode_inverse(inverse)
    with monkeypatch.context() as patch:
        patch.setattr(FixedSurfaceInverse,'_solve',forbidden)
        loaded=restore_inverse(saved,fresh,sources=sources)
    assert loaded.refinement_steps==3 and loaded.rule==rule and encode_inverse(loaded)['sha256']==saved['sha256']
    x=torch.randn(original.shape[0],device='cuda',dtype=original.dtype,requires_grad=True)
    a=loaded.apply(x);b=inverse.apply(x)
    torch.testing.assert_close(a,b,rtol=2e-8,atol=1e-9)
    ga,=torch.autograd.grad(a.abs().square().sum(),x)
    gb,=torch.autograd.grad(b.abs().square().sum(),x)
    torch.testing.assert_close(ga,gb,rtol=3e-8,atol=1e-8)
    points=dense.system.bodies[0].mesh.vertices.new_tensor([[.8,.1,.2]])
    for a,b in zip(decode_system(restored,x,points,0,'total',query_config=FieldQueryConfig(adaptive=False)),
                   decode_system(original,x,points,0,'total',query_config=FieldQueryConfig(adaptive=False))):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    wrong=(mw.PlaneWave(source_time=mw.CW(3e8,amplitude=1.3,phase=.2)),sources[1])
    for changed in (wrong,sources[::-1],()):
        with pytest.raises(ValueError,match='carrier'):restore(payload,fresh,sources=changed)
    for name in ('rhs_identity','incident_carrier_identity','operator_identity'):
        previous=original.metadata[name]
        original.metadata[name]='0'*64
        with pytest.raises(ValueError,match='identity'):encode(original)
        original.metadata[name]=previous
        bad=copy.deepcopy(payload)
        bad['operator']['metadata'][name]='0'*64
        bad['operator']['sha256']=_digest({k:v for k,v in bad['operator'].items() if k!='sha256'})
        bad['sha256']=_digest({k:v for k,v in bad.items() if k!='sha256'})
        with pytest.raises(ValueError,match='identity|metadata'):restore(bad,fresh,sources=sources)
    old=original._public_config
    original._public_config=replace(config,stopping=replace(config.stopping,rtol=2e-7))
    with pytest.raises(ValueError,match='numerical binding'):encode(original)
    original._public_config=old
    delayed=loaded.apply(x)
    loaded.operator.body_handles[0].incident_carrier=None
    with pytest.raises(ValueError,match='changed'):torch.autograd.grad(delayed.real.sum(),x)
    bad=copy.deepcopy(payload);bad['operator']=copy.deepcopy(payload)
    bad['sha256']=_digest({k:v for k,v in bad.items() if k!='sha256'})
    with pytest.raises(ValueError,match='bare numerical'):restore(bad,fresh,sources=sources)


def test_empty_carrier_snapshot_rejects_pumped_sources():
    dense,_=runpy.run_path('tests/mom/test_surface_volume.py')['problem']((2.,3.))
    config=mw.MoMConfig(operator_backend='h2',integration=dense.integration,
        hierarchical=HMatrixConfig(leaf_size=64),riesz_storage='local_rwg_v1',
        stopping=StoppingRule(method='block_gmres',preconditioner='row_column_norm'))
    operator=HierarchicalH2SystemOperator(dense.system,dense.exterior,(),config)
    payload=encode(operator);restored=restore(payload,operator.provider)
    assert restored.incident_carrier.waves==() and encode(restored)['sha256']==payload['sha256']
    with pytest.raises(ValueError,match='carrier'):restore(payload,operator.provider,sources=(mw.PlaneWave(),))
