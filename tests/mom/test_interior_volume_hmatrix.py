"""Interior cross maps preserve original fields and complex adjoints."""
import runpy
import pytest
import torch
from witwin.maxwell.mom.interior_volume_hmatrix import InteriorVolumeBlocks,InteriorVolumeHConfig,InteriorVolumeHierarchicalAction
from witwin.maxwell.mom.fields import decode_interior_columns
from witwin.maxwell.mom.runtime import DecodeHandle
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.media import Material
from witwin.maxwell.mom.fmm_capacity import storage_bytes

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def provider(field,long=False):
    geometry=runpy.run_path('tests/mom/test_surface_volume_hmatrix.py')['provider'](long)
    return InteriorVolumeBlocks(geometry.binding,field=field,order=3)


def check(blocks,action):
    body=blocks.geometry.binding.surface_system.bodies[0]
    handle=DecodeHandle(body.basis,compile_medium(Material(),body.interior.frequency),body.interior,(),IntegrationConfig(order=3,singular_order=3),{})
    identity=torch.eye(2*body.basis.n_edges,device='cuda',dtype=blocks.dtype)
    original=decode_interior_columns(handle,identity,blocks.points)[blocks.field=='H']
    original=original.permute(1,0,2).reshape(action.shape)
    assert action.shape[0]!=action.shape[1]
    assert action.receipt['retained_bytes']==storage_bytes(action.tensors())
    coverage=torch.zeros(action.shape,device='cuda',dtype=torch.int64)
    for r,c,_ in action.blocks:coverage[r[:,None],c[None,:]]+=1
    assert bool((coverage==1).all())
    torch.manual_seed(67)
    x=torch.randn(2,action.shape[1],device='cuda',dtype=blocks.dtype,requires_grad=True)
    y=torch.randn(2,action.shape[0],device='cuda',dtype=blocks.dtype)
    result=action.apply(x);adjoint=action.adjoint(y)
    torch.testing.assert_close(result,x@original.mT,rtol=2e-6,atol=1e-10)
    torch.testing.assert_close(adjoint,y@original.conj(),rtol=2e-6,atol=1e-10)
    torch.testing.assert_close((result*y.conj()).sum(),(x*adjoint.conj()).sum(),rtol=2e-12,atol=1e-10)
    grad,=torch.autograd.grad((result*y.conj()).real.sum(),x)
    torch.testing.assert_close(grad,adjoint,rtol=2e-12,atol=1e-10)


@pytest.mark.parametrize('field',['E','H'])
def test_interior_cross_original_far_and_gradients(field):
    blocks=provider(field,True)
    action=InteriorVolumeHierarchicalAction(blocks,InteriorVolumeHConfig(leaf_size=12,max_rank=24,row_batch=12,tolerance=1e-8))
    assert action.receipt['far_blocks']>0 and action.receipt['exact_blocks']>0
    check(blocks,action)


def test_interior_cross_guards_and_selections():
    blocks=provider('E')
    rows=torch.tensor([2,0,2],device='cuda');cols=torch.tensor([1,0,1],device='cuda')
    full=blocks(torch.arange(blocks.row_size,device='cuda'),torch.arange(blocks.column_size,device='cuda'))
    selected=full[torch.cat([rows+k*blocks.row_size for k in range(3)])][:,torch.cat((cols,cols+blocks.column_size))]
    torch.testing.assert_close(blocks(rows,cols),selected)
    assert blocks(rows[:0],cols).shape==(0,6)
    assert blocks(rows,cols[:0]).shape==(9,0)
    for bad in (rows.cpu(),rows.float(),rows[:,None],rows-3):
        with pytest.raises(ValueError,match='indices'):blocks(bad,cols)
    with pytest.raises(MemoryError,match='budget'):
        InteriorVolumeHierarchicalAction(blocks,InteriorVolumeHConfig(retained_budget_bytes=1))
    action=InteriorVolumeHierarchicalAction(blocks,InteriorVolumeHConfig(leaf_size=64))
    check(blocks,action)
    with torch.no_grad():blocks.points[0,0]+=1e-5
    with pytest.raises(ValueError,match='changed'):action.assert_current()
