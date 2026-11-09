"""A nonfirst material region keeps local cross rows and global routing distinct."""
import runpy
import pytest
import torch
from witwin.maxwell.mom.surface_volume_hmatrix import SurfaceVolumeBlocks,SurfaceVolumeHierarchicalAction,SurfaceVolumeHConfig
from witwin.maxwell.mom.volume_response_matrix import build_volume_source_matrix

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_nonfirst_heterogeneous_region_local_rows_match_global_slices():
    operator,bindings=runpy.run_path('tests/mom/test_surface_volume.py')['problem']((2.,3.))
    provider=SurfaceVolumeBlocks(bindings[1],order=2,inner_order=3)
    action=SurfaceVolumeHierarchicalAction(provider,SurfaceVolumeHConfig(leaf_size=64))
    full,_=build_volume_source_matrix(operator,[bindings[1]],order=2,inner_order=3)
    rows=torch.cat([torch.arange(part.start,part.stop,device='cuda') for part in
        (operator.system.J_slices[1],operator.system.M_slices[1])])
    assert not bool(full[operator.system.J_slices[0]].any())
    assert not bool(full[operator.system.M_slices[0]].any())
    current=torch.tensor([1+.3j,-.5j,.7],device='cuda',dtype=torch.complex128)
    torch.testing.assert_close(action.apply(current),full[rows]@current,rtol=1e-12,atol=1e-13)
    assert action.receipt['region_id']==2
    original_storage=operator.matrix.untyped_storage().data_ptr()
    assert all(t.untyped_storage().data_ptr()!=original_storage for t in provider.tensors())
