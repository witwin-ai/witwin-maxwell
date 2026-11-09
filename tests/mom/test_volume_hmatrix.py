"""Actual regional volume H blocks, original actions and retained-state guards."""
import runpy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.volume_hmatrix import VolumeCellBlocks,VolumeHConfig,VolumeHierarchicalAction
from witwin.maxwell.mom.vie_operator import cell_current_fields
from witwin.maxwell.mom.fmm_capacity import storage_bytes

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def elongated_volume():
    from itertools import permutations
    from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
    vertices=torch.cartesian_prod(torch.arange(17,device='cuda',dtype=torch.float64)*.01,
        torch.tensor([0.,.001],device='cuda',dtype=torch.float64),
        torch.tensor([0.,.001],device='cuda',dtype=torch.float64))
    cells=[]
    for x in range(16):
        for axes in permutations(range(3)):
            point=[x,0,0];ids=[4*x]
            for axis in axes:
                point[axis]+=1;ids.append(4*point[0]+2*point[1]+point[2])
            cells.append(ids)
    cells=torch.tensor(cells,device='cuda')
    tetrahedra=vertices[cells];edges=tetrahedra[:,1:]-tetrahedra[:,:1]
    cells=torch.where((torch.linalg.det(edges)<0)[:,None],cells[:,[0,2,1,3]],cells)
    return TetrahedralVolumeMesh(vertices,cells,region_id=1)


@pytest.mark.parametrize('field',['E','H'])
def test_volume_h_original_actions_adjoint_gradients_and_partition(field):
    mesh=elongated_volume()
    medium=compile_medium(mw.Material(eps_r=2.,mu_r=1.2),3e8)
    provider=VolumeCellBlocks(mesh,medium,field=field,order=3)
    action=VolumeHierarchicalAction(provider,VolumeHConfig(leaf_size=6,admissibility=1.,max_rank=24,tolerance=1e-8,row_batch=12))
    assert action.receipt['far_blocks']>0 and action.receipt['exact_blocks']>0
    assert action.receipt['retained_bytes']==storage_bytes(action.tensors())
    assert not action.receipt['physical_qualification']
    count=len(mesh.cells);ids=mesh.cells.new_tensor(range(count))
    cover=torch.zeros(3*count,3*count,device='cuda',dtype=torch.int64)
    for rows,cols,_ in action.blocks:cover[rows[:,None],cols[None,:]]+=1
    assert bool((cover==1).all())
    torch.manual_seed(34)
    x=torch.randn(2,3*count,device='cuda',dtype=torch.complex128,requires_grad=True)
    y=torch.randn_like(x)
    actual=action.apply(x)
    dense=provider(ids,ids)
    torch.testing.assert_close(actual,x@dense.mT,rtol=2e-7,atol=1e-9)
    source=x[0].reshape(3,count).mT
    fields=cell_current_fields(mesh,source,provider.points,medium,order=3)
    torch.testing.assert_close(actual[0],fields[0 if field=='E' else 1].mT.flatten(),rtol=2e-7,atol=1e-9)
    torch.testing.assert_close(action.adjoint(y),y@dense.conj(),rtol=2e-7,atol=1e-9)
    torch.testing.assert_close((y.conj()*actual).sum(),(action.adjoint(y).conj()*x).sum(),rtol=1e-11,atol=1e-10)
    gradient,=torch.autograd.grad((y.conj()*actual).real.sum(),x)
    torch.testing.assert_close(gradient,action.adjoint(y),rtol=1e-12,atol=1e-12)
    with torch.no_grad():action.blocks[0][0][0]+=1
    with pytest.raises(ValueError,match='changed'):action.apply(x)


def test_volume_h_budget_configuration_and_provider_mutation():
    mesh=runpy.run_path('tests/mom/test_volume_green.py')['mesh']()
    provider=VolumeCellBlocks(mesh,compile_medium(mw.Material(),3e8),order=2)
    with pytest.raises(MemoryError,match='budget'):
        VolumeHierarchicalAction(provider,VolumeHConfig(retained_budget_bytes=1))
    for kwargs in ({'leaf_size':True},{'tolerance':float('nan')},{'max_rank':0}):
        with pytest.raises(ValueError):VolumeHConfig(**kwargs)
    with torch.no_grad():provider.points[0,0]+=1
    with pytest.raises(ValueError,match='changed'):provider.assert_current()


@pytest.mark.parametrize('real_dtype',[torch.float32,torch.float64])
def test_forced_rank_rejection_descends_to_exact_leaves(monkeypatch,real_dtype):
    from witwin.maxwell.mom import hierarchical_blocks as module
    from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
    from witwin.maxwell.mom.lowrank import CompressionFailure
    original=runpy.run_path('tests/mom/test_volume_green.py')['mesh']().refine(1)
    mesh=TetrahedralVolumeMesh(original.vertices.to(real_dtype),original.cells,region_id=1)
    provider=VolumeCellBlocks(mesh,compile_medium(mw.Material(eps_r=2.),3e8),order=3)
    # Synthetic root rejection exercises recursive coverage only. The real
    # geometric far partition is checked separately on the elongated-cell control.
    def root_partition(tree,source=None,**kwargs):
        root=tree.starts.new_zeros(1)
        yield root,root,torch.ones(1,device=tree.device,dtype=torch.bool)
    def reject(*args,**kwargs):raise CompressionFailure('Forced rank exhaustion',{'forced':True})
    monkeypatch.setattr(module,'support_block_partition',root_partition)
    monkeypatch.setattr(module,'streamed_cross_approximation',reject)
    action=VolumeHierarchicalAction(provider,VolumeHConfig(leaf_size=1,max_rank=1))
    assert len(action.receipt['compression_rejections'])>1
    assert action.receipt['far_blocks']==0 and action.receipt['exact_blocks']==len(mesh.cells)**2
    ids=torch.arange(len(mesh.cells),device='cuda')
    dense=provider(ids,ids)
    x=torch.randn(3*len(mesh.cells),device='cuda',dtype=provider.dtype)
    tolerance=2e-6 if real_dtype==torch.float32 else 1e-12
    torch.testing.assert_close(action.apply(x),dense@x,rtol=tolerance,atol=tolerance)
    torch.testing.assert_close(action.adjoint(x),dense.mH@x,rtol=tolerance,atol=tolerance)
