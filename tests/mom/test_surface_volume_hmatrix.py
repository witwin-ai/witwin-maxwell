"""Rectangular weak cross actions use two support trees and one exact adjoint."""
import runpy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis,SurfaceSystem
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.volume_mesh import bind_volume_region
from witwin.maxwell.mom.surface_volume_hmatrix import SurfaceVolumeBlocks,SurfaceVolumeHConfig,SurfaceVolumeHierarchicalAction
from witwin.maxwell.mom.volume_response_matrix import build_volume_source_matrix
from witwin.maxwell.mom.fmm_capacity import storage_bytes

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def provider(long=False):
    if long:volume=runpy.run_path('tests/mom/test_volume_hmatrix.py')['elongated_volume']()
    else:volume=runpy.run_path('tests/mom/test_volume_green.py')['mesh']().refine(1)
    system=SurfaceSystem.create([(0,'body',RWGBasis(volume.boundary),compile_medium(mw.Material(eps_r=2.),3e8))])
    return SurfaceVolumeBlocks(bind_volume_region(volume,system),order=2,inner_order=3)


def check_action(blocks,action):
    assert action.shape==(2*blocks.row_size,3*blocks.column_size)
    assert action.shape[0]!=action.shape[1]
    assert action.receipt['retained_bytes']==storage_bytes(action.tensors())
    full,_=build_volume_source_matrix(blocks.layout,[blocks.binding],order=2,inner_order=3)
    full=full.reshape(action.shape[0],blocks.column_size,3).permute(0,2,1).reshape(action.shape)
    coverage=torch.zeros(action.shape,device='cuda',dtype=torch.int64)
    for r,c,_ in action.blocks:coverage[r[:,None],c[None,:]]+=1
    assert bool((coverage==1).all())
    torch.manual_seed(61)
    x=torch.randn(2,action.shape[1],device='cuda',dtype=blocks.dtype,requires_grad=True)
    y=torch.randn(2,action.shape[0],device='cuda',dtype=blocks.dtype)
    actual=action.apply(x);adjoint=action.adjoint(y)
    torch.testing.assert_close(actual,x@full.mT,rtol=2e-7,atol=1e-12)
    torch.testing.assert_close(adjoint,y@full.conj(),rtol=2e-7,atol=1e-12)
    torch.testing.assert_close((actual*y.conj()).sum(),(x*adjoint.conj()).sum(),rtol=1e-11,atol=1e-14)
    grad,=torch.autograd.grad((actual*y.conj()).real.sum(),x)
    torch.testing.assert_close(grad,adjoint,rtol=1e-12,atol=1e-14)


def test_rectangular_original_blocks_and_basis_guard():
    blocks=provider();action=SurfaceVolumeHierarchicalAction(blocks,SurfaceVolumeHConfig(leaf_size=64))
    check_action(blocks,action)
    with torch.no_grad():blocks.basis.plus_faces[0]=blocks.basis.minus_faces[0]
    with pytest.raises(ValueError,match='changed'):action.assert_current()


def test_rectangular_actual_far_compression_original_and_adjoint():
    blocks=provider(True)
    action=SurfaceVolumeHierarchicalAction(blocks,SurfaceVolumeHConfig(leaf_size=12,max_rank=24,row_batch=12,tolerance=1e-8))
    assert action.receipt['far_blocks']>0 and action.receipt['exact_blocks']>0
    check_action(blocks,action)
    assert not action.receipt['global_operator_error_verified']
