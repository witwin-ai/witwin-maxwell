"""Original-equation checks for actual compressed tensor response components."""
import pytest
import torch
from test_tensor_compressed_background import inverse_for
from witwin.maxwell.mom.surface_volume_hmatrix import SurfaceVolumeBlocks,SurfaceVolumeHierarchicalAction
from witwin.maxwell.mom.interior_volume_hmatrix import InteriorVolumeBlocks,InteriorVolumeHierarchicalAction
from witwin.maxwell.mom.volume_hmatrix import VolumeCellBlocks,VolumeHierarchicalAction
from witwin.maxwell.mom.coupled_volume_action import CoupledVolumeAction
from witwin.maxwell.mom.coupled_original_check import check_coupled_original_columns_many
from witwin.maxwell.mom.dyadic_green import DyadicGreenKernel

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_actual_tensor_coupled_original_rows_and_fields():
    inverse,binding=inverse_for(True)
    kernel=inverse.operator.provider.interior_kernels[0]
    source=SurfaceVolumeHierarchicalAction(SurfaceVolumeBlocks(binding,order=2,inner_order=2,
        tensor_kernel=kernel,tensor_kernel_batch=8192))
    actions=[]
    for field in ('E','H'):
        decode=InteriorVolumeHierarchicalAction(InteriorVolumeBlocks(binding,field=field,order=2,tensor_kernel=kernel))
        direct=VolumeHierarchicalAction(VolumeCellBlocks(binding.volume,kernel.medium,field=field,order=2,
            radial_order=2,tensor_kernel=kernel,kernel_batch=8192))
        actions.append(CoupledVolumeAction(inverse,(source,),(decode,),(direct,)))
    records=check_coupled_original_columns_many(actions,column_batch=3,edge_batch=3,query_batch=1)
    assert all(r['passed'] for r in records)
    assert all(not r['physical_qualification'] for r in records)
    # An equivalent but independently owned kernel must not silently replace
    # the exact surface provider's regional numerical policy.
    wrong=DyadicGreenKernel(kernel.medium,kernel.integration,kernel.cache_max_bytes)
    p=VolumeCellBlocks(binding.volume,kernel.medium,field='H',order=2,radial_order=2,
        tensor_kernel=wrong,kernel_batch=8192)
    other=VolumeHierarchicalAction(p)
    with pytest.raises(ValueError,match='exact surface regional kernel'):
        CoupledVolumeAction(inverse,(source,),actions[1].fields,(other,))


def test_tensor_refinement_rejects_changed_volume_policy():
    from witwin.maxwell.mom.coupled_field_refinement import _pair_settings
    from witwin.maxwell.mom.fields import FieldQueryConfig
    inverse,binding=inverse_for(True);kernel=inverse.operator.provider.interior_kernels[0]
    source=SurfaceVolumeHierarchicalAction(SurfaceVolumeBlocks(binding,order=2,inner_order=2,
        tensor_kernel=kernel,tensor_kernel_batch=8192))
    def action(order,radial,face):
        f=InteriorVolumeHierarchicalAction(InteriorVolumeBlocks(binding,field='E',order=order,tensor_kernel=kernel))
        d=VolumeHierarchicalAction(VolumeCellBlocks(binding.volume,kernel.medium,field='E',order=order,
            radial_order=radial,face_rule=face,tensor_kernel=kernel,kernel_batch=8192))
        return CoupledVolumeAction(inverse,(source,),(f,),(d,))
    coarse=action(2,2,'triangle')
    for radial,face,message in ((2,'triangle','radial orders'),(3,'projected_duffy','integration policy')):
        fine=action(3,radial,face)
        with pytest.raises(ValueError,match=message):
            _pair_settings(coarse,fine,FieldQueryConfig(),3)
