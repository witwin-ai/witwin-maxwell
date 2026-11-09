"""Tensor coupled H/H2 factors restore on exact regional kernels and policies."""
import copy
import pytest
import torch
from test_tensor_compressed_background import inverse_for
from witwin.maxwell.mom.surface_volume_hmatrix import SurfaceVolumeBlocks,SurfaceVolumeHierarchicalAction
from witwin.maxwell.mom.interior_volume_hmatrix import InteriorVolumeBlocks,InteriorVolumeHierarchicalAction
from witwin.maxwell.mom.volume_hmatrix import VolumeCellBlocks,VolumeHierarchicalAction
from witwin.maxwell.mom.volume_h2 import VolumeNestedH2Action
from witwin.maxwell.mom.coupled_volume_action import CoupledVolumeAction
from witwin.maxwell.mom.implicit_surface_solve import FixedSurfaceInverse
from witwin.maxwell.mom.tensor_nearfield import GeneralizedSystemBlocks
from witwin.maxwell.mom import coupled_response_snapshot as codec
from witwin.maxwell.mom.volume_h_snapshot import _digest

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_tensor_coupled_policy_snapshot_without_sampling(monkeypatch,tmp_path):
    inverse,binding=inverse_for(True);k=inverse.operator.provider.interior_kernels[0]
    source=SurfaceVolumeHierarchicalAction(SurfaceVolumeBlocks(binding,order=2,inner_order=2,
        tensor_kernel=k,tensor_face_rule='projected_duffy',tensor_kernel_batch=8192))
    actions=[]
    for field in ('E','H'):
        f=InteriorVolumeHierarchicalAction(InteriorVolumeBlocks(binding,field=field,order=2,tensor_kernel=k))
        d=VolumeHierarchicalAction(VolumeCellBlocks(binding.volume,k.medium,field=field,order=2,
            tensor_kernel=k,radial_order=3,face_rule='projected_duffy',kernel_batch=8192))
        actions.append(CoupledVolumeAction(inverse,(source,),(f,),
            (VolumeNestedH2Action(d) if field=='H' else d,)))
    payload=codec.encode(*actions)
    assert payload['version']=='exact_tensor_coupled_eh_response_v2'
    path=tmp_path/'coupled.pt';torch.save(payload,path);payload=torch.load(path,weights_only=True)
    fresh,binding2=inverse_for(True);provider=fresh.operator.provider
    def forbidden(*args,**kwargs):raise AssertionError('Restoration must not sample, build factors or solve')
    with monkeypatch.context() as patch:
        for cls in (GeneralizedSystemBlocks,SurfaceVolumeBlocks,InteriorVolumeBlocks,VolumeCellBlocks):
            patch.setattr(cls,'__call__',forbidden)
        for cls in (SurfaceVolumeHierarchicalAction,InteriorVolumeHierarchicalAction,VolumeHierarchicalAction,VolumeNestedH2Action):
            patch.setattr(cls,'__init__',forbidden)
        patch.setattr(FixedSurfaceInverse,'_solve',forbidden)
        restored=codec.restore(payload,provider,(binding2.volume,))
    assert codec.encode(*restored)['sha256']==payload['sha256']
    assert restored[0].sources[0] is restored[1].sources[0]
    x=torch.tensor([1.+.2j,.3j,-.4],device='cuda',dtype=inverse.dtype)
    for a,b in zip(actions,restored):
        torch.testing.assert_close(a.apply(x),b.apply(x),rtol=2e-10,atol=1e-12)
        torch.testing.assert_close(a.adjoint(x),b.adjoint(x),rtol=2e-10,atol=1e-12)
        assert b.direct[0].provider.tensor_kernel is provider.interior_kernels[0]
        assert b.direct[0].provider.radial_order==3
    for kind in ('source','direct'):
        bad=copy.deepcopy(payload)
        policy=(bad['regions'][0]['tensor_policy'] if kind=='source' else
                bad['channels'][0]['regions'][0]['tensor_policy'])
        policy['kernel_batch']+=1
        bad['sha256']=_digest({k:v for k,v in bad.items() if k!='sha256'})
        with pytest.raises(ValueError,match='binding'):
            codec.restore(bad,provider,(binding2.volume,))
