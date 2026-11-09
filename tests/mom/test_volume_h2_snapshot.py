"""Exact nested H2 snapshots retain arrays and never redo algebraic setup."""
import copy
import io
import runpy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.volume_h2_snapshot import encode,restore
from witwin.maxwell.mom.volume_h_snapshot import _digest
from witwin.maxwell.mom.volume_hmatrix import VolumeCellBlocks,VolumeHierarchicalAction,VolumeHConfig
from witwin.maxwell.mom.volume_h2 import VolumeNestedH2Action
from witwin.maxwell.mom.h2 import H2Config,NestedH2Actions
from witwin.maxwell.mom.h2_basis import NestedClusterBasis
from witwin.maxwell.mom.physics import compile_medium

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.fixture(autouse=True)
def deterministic_actions(monkeypatch):
    # Packed near index accumulation is nondeterministic in the default CUDA
    # path. Exact execution controls opt into deterministic native algorithms;
    # the storage hash remains exact independently of that execution policy.
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    enabled=torch.are_deterministic_algorithms_enabled()
    warn_only=torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:yield
    finally:torch.use_deterministic_algorithms(enabled,warn_only=warn_only)


@pytest.mark.parametrize('far',[False,True])
def test_actual_volume_h2_exact_independent_restore_without_reconversion(far,monkeypatch,tmp_path):
    make=runpy.run_path('tests/mom/test_volume_hmatrix.py')['elongated_volume'] if far else runpy.run_path('tests/mom/test_volume_green.py')['mesh']
    medium=compile_medium(mw.Material(eps_r=2.,mu_r=1.2),3e8)
    provider=VolumeCellBlocks(make(),medium,field='H',order=3)
    source=VolumeHierarchicalAction(provider,VolumeHConfig(leaf_size=6 if far else 64,max_rank=24,row_batch=12))
    action=VolumeNestedH2Action(source,H2Config())
    if far:assert action.nested.couplings and action.nested.near is not None
    payload=encode(action)
    torch.save(payload,tmp_path/'snapshot.pt')
    stream=io.BytesIO();torch.save(payload,stream);stream.seek(0)
    payload=torch.load(stream,weights_only=True)
    fresh=VolumeCellBlocks(make(),compile_medium(mw.Material(eps_r=2.,mu_r=1.2),3e8),field='H',order=3)
    assert fresh.mesh is not provider.mesh
    def forbidden(*args,**kwargs):raise AssertionError('Exact restore must not sample, decompose, convert or solve')
    with monkeypatch.context() as patch:
        for cls in (VolumeHierarchicalAction,VolumeNestedH2Action,NestedH2Actions,NestedClusterBasis):patch.setattr(cls,'__init__',forbidden)
        patch.setattr(VolumeCellBlocks,'__call__',forbidden)
        for name in ('solve','svd','qr'):patch.setattr(torch.linalg,name,forbidden)
        restored=restore(payload,fresh)
    x=torch.randn(2,action.shape[0],device='cuda',dtype=action.dtype,requires_grad=True)
    y=torch.randn_like(x)
    torch.testing.assert_close(restored.apply(x),action.apply(x),rtol=0,atol=0)
    torch.testing.assert_close(restored.adjoint(y),action.adjoint(y),rtol=0,atol=0)
    ga,=torch.autograd.grad((restored.apply(x)*y.conj()).real.sum(),x)
    gb,=torch.autograd.grad((action.apply(x)*y.conj()).real.sum(),x)
    torch.testing.assert_close(ga,gb,rtol=0,atol=0)
    assert encode(restored)['sha256']==payload['sha256']
    check_structural_rejections(payload,fresh)
    broken=copy.deepcopy(payload)
    array=broken['nested']['couplings'][0][2] if far else broken['nested']['near']['records'][0][4]
    before=array[0,0].clone();array[0,0]+=1
    assert not torch.equal(array[0,0],before)
    with pytest.raises(ValueError,match='checksum'):restore(broken,fresh)
    broken=copy.deepcopy(payload)
    broken['nested']['near']['records'].append(copy.deepcopy(broken['nested']['near']['records'][0]))
    broken['sha256']=_digest({k:v for k,v in broken.items() if k!='sha256'})
    with pytest.raises(ValueError):restore(broken,fresh)
    if far:
        basis=restored.nested.test_basis
        value=next(v for v in basis.transfers if v is not None and v.numel())
    else:value=restored.nested.near.groups[0][2]
    with torch.no_grad():value.add_(1)
    with pytest.raises(ValueError,match='changed'):restored.apply(x)


def check_structural_rejections(payload,provider):
    """Use saved actual factors to test schema guards without repeating setup."""
    def reseal(value):
        value['sha256']=_digest({k:v for k,v in value.items() if k!='sha256'})
        return value
    bad=copy.deepcopy(payload)
    bad['nested']['row_scale'][0]*=2
    with pytest.raises(ValueError,match='scaling'):restore(reseal(bad),provider)
    bad=copy.deepcopy(payload)
    ranks=list(bad['nested']['test_basis']['ranks']);ranks[-1]=bad['configuration']['max_rank']+1
    bad['nested']['test_basis']['ranks']=tuple(ranks)
    with pytest.raises(ValueError,match='rank'):restore(reseal(bad),provider)
    if payload['nested']['couplings']:
        bad=copy.deepcopy(payload)
        bad['nested']['couplings']=bad['nested']['couplings'][1:]
        with pytest.raises(ValueError,match='cover'):restore(reseal(bad),provider)
    bad=copy.deepcopy(payload);bad['configuration']['retained_budget_bytes']=1
    with pytest.raises(MemoryError,match='budget'):restore(reseal(bad),provider)
    wrong=VolumeCellBlocks(provider.mesh,provider.medium,field='E',order=provider.order)
    with pytest.raises(ValueError,match='physical binding'):restore(payload,wrong)
    changed=compile_medium(mw.Material(eps_r=3.,mu_r=1.2),provider.medium.frequency)
    wrong=VolumeCellBlocks(provider.mesh,changed,field=provider.field,order=provider.order)
    with pytest.raises(ValueError,match='physical binding'):restore(payload,wrong)
