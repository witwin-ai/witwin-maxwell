"""Shared E/H snapshot restores native coupling without assembly or solves."""
import copy
import io
import runpy
import pytest
import torch
from witwin.maxwell.mom.coupled_response_snapshot import encode,restore
from witwin.maxwell.mom.coupled_volume_action import CoupledVolumeAction
from witwin.maxwell.mom.surface_volume_hmatrix import SurfaceVolumeBlocks,SurfaceVolumeHierarchicalAction,SurfaceVolumeHConfig
from witwin.maxwell.mom.interior_volume_hmatrix import InteriorVolumeBlocks,InteriorVolumeHierarchicalAction,InteriorVolumeHConfig
from witwin.maxwell.mom.volume_hmatrix import VolumeCellBlocks,VolumeHierarchicalAction,VolumeHConfig
from witwin.maxwell.mom.volume_h2 import VolumeNestedH2Action
from witwin.maxwell.mom.implicit_surface_solve import FixedSurfaceInverse
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.volume_h_snapshot import _digest

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_shared_pair_native_actions_and_restore_guards(monkeypatch):
    setup=runpy.run_path('tests/mom/test_compressed_volume_preparation.py')['setup']
    _,bindings,inverse=setup();fresh,fresh_bindings,_=setup()
    sources=[SurfaceVolumeHierarchicalAction(SurfaceVolumeBlocks(b,order=3,inner_order=3),SurfaceVolumeHConfig(leaf_size=64)) for b in bindings]
    actions=[]
    for field in ('E','H'):
        decoders=[];direct=[]
        for b in bindings:
            decoders.append(InteriorVolumeHierarchicalAction(InteriorVolumeBlocks(b,field=field,order=3),InteriorVolumeHConfig(leaf_size=64)))
            d=VolumeHierarchicalAction(VolumeCellBlocks(b.volume,b.surface_system.bodies[b.volume.region_id-1].interior,field=field,order=3),VolumeHConfig(leaf_size=64))
            direct.append(VolumeNestedH2Action(d) if field=='H' else d)
        actions.append(CoupledVolumeAction(inverse,sources,decoders,direct))
    payload=encode(*actions);buffer=io.BytesIO();torch.save(payload,buffer);buffer.seek(0)
    payload=torch.load(buffer,weights_only=True)
    provider=ScalarSystemBlocks(fresh.system,fresh.exterior,fresh.integration)
    volumes=[b.volume for b in fresh_bindings]
    def forbidden(*args,**kwargs):raise AssertionError('Restore cannot sample or solve')
    with monkeypatch.context() as patch:
        for cls in (ScalarSystemBlocks,SurfaceVolumeBlocks,InteriorVolumeBlocks,VolumeCellBlocks):patch.setattr(cls,'__call__',forbidden)
        patch.setattr(FixedSurfaceInverse,'_solve',forbidden)
        restored=restore(payload,provider,volumes)
    assert restored[0].inverse is restored[1].inverse
    assert all(a is b for a,b in zip(restored[0].sources,restored[1].sources))
    assert encode(*restored)['sha256']==payload['sha256']
    x=torch.randn(2,6,device='cuda',dtype=inverse.dtype,requires_grad=True)
    for a,b in zip(actions,restored):
        torch.testing.assert_close(a.apply(x),b.apply(x),rtol=2e-8,atol=1e-10)
        torch.testing.assert_close(a.adjoint(x),b.adjoint(x),rtol=2e-8,atol=1e-10)
        ga,=torch.autograd.grad(a.apply(x).abs().square().sum(),x)
        gb,=torch.autograd.grad(b.apply(x).abs().square().sum(),x)
        torch.testing.assert_close(ga,gb,rtol=3e-8,atol=1e-10)
    with pytest.raises(ValueError,match='ordered volume'):restore(payload,provider,volumes[::-1])
    for change in ('field','receipt','order'):
        bad=copy.deepcopy(payload)
        if change=='field':bad['channels'][1]['field']='E'
        elif change=='receipt':bad['channels'][0]['receipt']['physical_qualification']=True
        else:bad['regions'][0]['inner_order']=4
        bad['sha256']=_digest({k:v for k,v in bad.items() if k!='sha256'})
        with pytest.raises(ValueError):restore(bad,provider,volumes)
    with torch.no_grad():restored[0].sources[0].blocks[0][2].add_(1)
    with pytest.raises(ValueError,match='changed'):restored[1].apply(x)
