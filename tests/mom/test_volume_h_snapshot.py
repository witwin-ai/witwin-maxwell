"""Stored H factors restore exactly without sampling, compression or solves."""
import copy
import io
import runpy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.volume_h_snapshot import encode,restore,_digest,_coverage
from witwin.maxwell.mom.volume_hmatrix import VolumeCellBlocks,VolumeHierarchicalAction,VolumeHConfig
from witwin.maxwell.mom.surface_volume_hmatrix import SurfaceVolumeBlocks,SurfaceVolumeHierarchicalAction,SurfaceVolumeHConfig
from witwin.maxwell.mom.interior_volume_hmatrix import InteriorVolumeBlocks,InteriorVolumeHierarchicalAction,InteriorVolumeHConfig
from witwin.maxwell.mom.physics import compile_medium

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def check(action,provider,monkeypatch):
    payload=encode(action)
    stream=io.BytesIO();torch.save(payload,stream);stream.seek(0)
    payload=torch.load(stream,weights_only=True)
    def forbidden(*args,**kwargs):raise AssertionError('Restore must not assemble, compress or solve')
    import witwin.maxwell.mom.hierarchical_blocks as blocks
    with monkeypatch.context() as patch:
        patch.setattr(type(provider),'__call__',forbidden)
        patch.setattr(blocks,'streamed_cross_approximation',forbidden)
        patch.setattr(torch.linalg,'solve',forbidden)
        restored=restore(payload,provider)
    x=torch.randn(2,action.shape[1],device='cuda',dtype=action.dtype,requires_grad=True)
    y=torch.randn(2,action.shape[0],device='cuda',dtype=action.dtype)
    torch.testing.assert_close(restored.apply(x),action.apply(x),rtol=0,atol=0)
    torch.testing.assert_close(restored.adjoint(y),action.adjoint(y),rtol=0,atol=0)
    ga,=torch.autograd.grad((restored.apply(x)*y.conj()).real.sum(),x)
    gb,=torch.autograd.grad((action.apply(x)*y.conj()).real.sum(),x)
    torch.testing.assert_close(ga,gb,rtol=0,atol=0)
    assert encode(restored)['sha256']==payload['sha256']
    broken=copy.deepcopy(payload)
    value=broken['blocks'][0]['factors'][0];before=value[0,0].clone();value[0,0]+=1
    assert not torch.equal(value[0,0],before)
    with pytest.raises(ValueError,match='checksum'):restore(broken,provider)
    broken=copy.deepcopy(payload);broken['blocks'].append(copy.deepcopy(broken['blocks'][0]))
    broken['sha256']=_digest({k:v for k,v in broken.items() if k!='sha256'})
    with pytest.raises(ValueError,match='cover'):restore(broken,provider)
    with torch.no_grad():restored.blocks[0][0][0]+=1
    with pytest.raises(ValueError,match='changed'):restored.apply(x)


def test_actual_rectangular_source_and_interior_snapshots(monkeypatch):
    _,bindings=runpy.run_path('tests/mom/test_surface_volume.py')['problem']((2.,3.))
    binding=bindings[1]
    source=SurfaceVolumeBlocks(binding,order=3,inner_order=3)
    field=InteriorVolumeBlocks(binding,field='H',order=3)
    for provider,cls,config in ((source,SurfaceVolumeHierarchicalAction,SurfaceVolumeHConfig),
                                (field,InteriorVolumeHierarchicalAction,InteriorVolumeHConfig)):
        action=cls(provider,config(leaf_size=64))
        check(action,provider,monkeypatch)
    wrong=InteriorVolumeBlocks(binding,field='E',order=3)
    with pytest.raises(ValueError,match='physical binding'):restore(encode(action),wrong)


def test_actual_volume_far_and_near_snapshot(monkeypatch):
    mesh=runpy.run_path('tests/mom/test_volume_hmatrix.py')['elongated_volume']()
    provider=VolumeCellBlocks(mesh,compile_medium(mw.Material(eps_r=2.,mu_r=1.2),3e8),field='H',order=3)
    action=VolumeHierarchicalAction(provider,VolumeHConfig(leaf_size=6,max_rank=24,row_batch=12))
    assert action.receipt['far_blocks'] and action.receipt['exact_blocks']
    check(action,provider,monkeypatch)


def test_coverage_sweep_detects_holes_and_overlaps():
    _coverage([(0,2,0,2),(0,1,2,3),(1,2,2,3)],2,3)
    for rectangles in ([(0,2,0,2)],[(0,2,0,3),(0,1,1,2)],[(0,1,0,3),(1,2,1,3)]):
        with pytest.raises(ValueError,match='cover'):_coverage(rectangles,2,3)


def test_independently_rebuilt_providers_and_explicit_corruption(monkeypatch):
    build=runpy.run_path('tests/mom/test_surface_volume.py')['problem']
    _,bindings=build((2.,3.));_,fresh_bindings=build((2.,3.))
    binding,fresh_binding=bindings[1],fresh_bindings[1]
    assert binding.volume is not fresh_binding.volume and binding.surface_system is not fresh_binding.surface_system
    factories=[(lambda b:SurfaceVolumeBlocks(b,order=3,inner_order=3),SurfaceVolumeHierarchicalAction,SurfaceVolumeHConfig),
        (lambda b:InteriorVolumeBlocks(b,field='E',order=3),InteriorVolumeHierarchicalAction,InteriorVolumeHConfig),
        (lambda b:VolumeCellBlocks(b.volume,b.surface_system.bodies[1].interior,order=3),VolumeHierarchicalAction,VolumeHConfig)]
    for factory,cls,config in factories:
        provider=factory(binding);action=cls(provider,config(leaf_size=64));fresh=factory(fresh_binding)
        assert fresh is not provider
        check(action,fresh,monkeypatch)
