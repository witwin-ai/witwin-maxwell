"""Nonfirst scalar region and both CUDA precisions retain field ownership."""
import runpy
import pytest
import torch
from witwin.maxwell.mom.interior_volume_hmatrix import InteriorVolumeBlocks,InteriorVolumeHierarchicalAction,InteriorVolumeHConfig
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh,bind_volume_region
from witwin.maxwell.mom.mesh import SurfaceSystem,RWGBasis
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.media import Material
from witwin.maxwell.mom.runtime import DecodeHandle
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.fields import decode_interior_columns

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('field',['E','H'])
def test_second_region_material_and_precision(dtype,field):
    base=runpy.run_path('tests/mom/test_volume_green.py')['mesh']()
    volumes=[TetrahedralVolumeMesh(base.vertices.to(dtype)+i*.3,base.cells,region_id=i+1) for i in range(2)]
    system=SurfaceSystem.create([(i,str(i),RWGBasis(v.boundary),compile_medium(Material(eps_r=2.+i),3e8)) for i,v in enumerate(volumes)])
    blocks=InteriorVolumeBlocks(bind_volume_region(volumes[1],system),field=field,order=3)
    action=InteriorVolumeHierarchicalAction(blocks,InteriorVolumeHConfig(leaf_size=64))
    body=system.bodies[1]
    handle=DecodeHandle(body.basis,compile_medium(Material(),3e8),body.interior,(),IntegrationConfig(order=3,singular_order=3),{})
    eye=torch.eye(2*body.basis.n_edges,device='cuda',dtype=blocks.dtype)
    full=decode_interior_columns(handle,eye,blocks.points)[field=='H'].permute(1,0,2).reshape(action.shape)
    x=torch.randn(2,action.shape[1],device='cuda',dtype=blocks.dtype,requires_grad=True)
    y=torch.randn(2,action.shape[0],device='cuda',dtype=blocks.dtype)
    tolerance=3e-5 if dtype==torch.float32 else 2e-12
    actual=action.apply(x)
    torch.testing.assert_close(actual,x@full.mT,rtol=tolerance,atol=tolerance)
    torch.testing.assert_close(action.adjoint(y),y@full.conj(),rtol=tolerance,atol=tolerance)
    grad,=torch.autograd.grad((actual*y.conj()).real.sum(),x)
    torch.testing.assert_close(grad,action.adjoint(y),rtol=tolerance,atol=tolerance)
    assert action.receipt['region_id']==2
