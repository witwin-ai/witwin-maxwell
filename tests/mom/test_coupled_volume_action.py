"""Whole heterogeneous response matches original equations in reordered regions."""
import runpy
import pytest
import torch
from witwin.maxwell.mom.coupled_volume_action import CoupledVolumeAction
from witwin.maxwell.mom.implicit_surface_solve import FixedSurfaceInverse
from witwin.maxwell.mom.surface_volume_hmatrix import SurfaceVolumeBlocks,SurfaceVolumeHierarchicalAction,SurfaceVolumeHConfig
from witwin.maxwell.mom.interior_volume_hmatrix import InteriorVolumeBlocks,InteriorVolumeHierarchicalAction,InteriorVolumeHConfig
from witwin.maxwell.mom.volume_hmatrix import VolumeCellBlocks,VolumeHierarchicalAction,VolumeHConfig
from witwin.maxwell.mom.volume_h2 import VolumeNestedH2Action
from witwin.maxwell.mom.h2_operator import HierarchicalH2Operator
from witwin.maxwell.mom.hmatrix import HierarchicalOperator,HMatrixConfig
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.volume_mesh import bind_volume_region,TetrahedralVolumeMesh
from witwin.maxwell.mom.mesh import SurfaceSystem,RWGBasis
from witwin.maxwell.mom.operator import IntegrationConfig,assemble_system
from witwin.maxwell.mom.volume_response_matrix import build_volume_source_matrix
from witwin.maxwell.mom.vie_operator import cell_current_field_blocks
from witwin.maxwell.mom.fields import decode_interior_columns
from witwin.maxwell.mom.runtime import DecodeHandle
from witwin.maxwell.mom.linalg import StoppingRule

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def setup(field,nested):
    original,old=runpy.run_path('tests/mom/test_surface_volume.py')['problem']((2.,3.))
    volumes=[TetrahedralVolumeMesh(b.volume.vertices+i*2.7,b.volume.cells,region_id=i+1).refine(1) for i,b in enumerate(old)]
    system=SurfaceSystem.create([(i,str(i),RWGBasis(v.boundary),original.system.bodies[i].interior) for i,v in enumerate(volumes)])
    config=IntegrationConfig(order=3,singular_order=3)
    op=HierarchicalOperator(ScalarSystemBlocks(system,original.exterior,config),HMatrixConfig(leaf_size=24,tolerance=1e-9,pivoting='adaptive'))
    assert op.metadata['far_blocks']>0
    if nested:op=HierarchicalH2Operator(op)
    inverse=FixedSurfaceInverse(op,StoppingRule(method='block_gmres',preconditioner='row_column_norm',rtol=1e-11,atol=1e-13,maxiter=300,restart=100,stagnation_window=100))
    sources=[];fields=[];direct=[]
    # Reverse binding order so global body order cannot masquerade as cell order.
    bindings=[bind_volume_region(v,system) for v in reversed(volumes)]
    for binding in bindings:
        sources.append(SurfaceVolumeHierarchicalAction(SurfaceVolumeBlocks(binding,order=3,inner_order=3),SurfaceVolumeHConfig(leaf_size=128)))
        fields.append(InteriorVolumeHierarchicalAction(InteriorVolumeBlocks(binding,field=field,order=3),InteriorVolumeHConfig(leaf_size=128)))
        action=VolumeHierarchicalAction(VolumeCellBlocks(binding.volume,system.bodies[binding.volume.region_id-1].interior,field=field,order=3),VolumeHConfig(leaf_size=128))
        direct.append(VolumeNestedH2Action(action) if nested else action)
    return CoupledVolumeAction(inverse,sources,fields,direct),bindings,original.exterior,config


def original_matrix(action,bindings,exterior,config):
    system=action.inverse.operator.provider.system
    matrix,_=assemble_system(system,exterior,config)
    source,_=build_volume_source_matrix(action.sources[0].provider.layout,bindings,order=3,inner_order=3)
    currents=torch.linalg.solve(matrix,source)
    residual=matrix@currents-source
    assert float(residual.norm()/source.norm())<1e-11
    full=source.new_zeros(action.shape);offset=0
    for binding in bindings:
        index=binding.volume.region_id-1;body=system.bodies[index]
        points=binding.volume.vertices[binding.volume.cells].mean(1)
        local=torch.cat((currents[system.J_slices[index]],currents[system.M_slices[index]]))
        handle=DecodeHandle(body.basis,exterior,body.interior,(),config,{})
        fields=decode_interior_columns(handle,local,points)[action.field=='H']
        width=3*len(points)
        full[offset:offset+width]=fields.reshape(width,-1)
        own=cell_current_field_blocks(binding.volume,points,body.interior,order=3)[action.field=='H'].reshape(width,width)
        full[offset:offset+width,offset:offset+width]+=own
        offset+=width
    return full,matrix,source


@pytest.mark.parametrize('field,nested',[('E',False),('H',True)])
def test_complete_coupled_original_adjoint_and_gradients(field,nested):
    action,bindings,exterior,config=setup(field,nested)
    full,matrix,source=original_matrix(action,bindings,exterior,config)
    torch.manual_seed(79)
    x=torch.randn(2,action.shape[1],device='cuda',dtype=action.dtype,requires_grad=True)
    y=torch.randn_like(x)
    result=action.apply(x);adjoint=action.adjoint(y)
    torch.testing.assert_close(result,x@full.mT,rtol=2e-7,atol=1e-9)
    torch.testing.assert_close(adjoint,y@full.conj(),rtol=2e-7,atol=1e-9)
    torch.testing.assert_close((result*y.conj()).sum(),(x*adjoint.conj()).sum(),rtol=2e-7,atol=1e-9)
    grad,=torch.autograd.grad((result*y.conj()).real.sum(),x)
    torch.testing.assert_close(grad,y@full.conj(),rtol=2e-7,atol=1e-9)
    # Check the actual compressed-source/global solve against the original A/B.
    rhs=x.new_zeros(2,action.inverse.shape[0])
    for local,block,ids in zip(x.detach().split(action.sizes,dim=-1),action.sources,action.indices):
        rhs=rhs.index_add(-1,ids,block.apply(action._component_major(local)))
    surface=action.inverse.apply(rhs)
    original_rhs=x.detach()@source.mT
    assert float((surface@matrix.mT-original_rhs).norm()/original_rhs.norm())<2e-10
    assert action.receipt['regions']==[2,1]
    assert not action.receipt['physical_qualification']
    for args in ((action.sources*2,action.fields*2,action.direct*2),
                 (action.sources,action.fields[::-1],action.direct)):
        with pytest.raises(ValueError):CoupledVolumeAction(action.inverse,*args)
    subset=CoupledVolumeAction(action.inverse,action.sources[:1],action.fields[:1],action.direct[:1])
    width=action.sizes[0]
    torch.testing.assert_close(subset.apply(x.detach()[...,:width]),x.detach()[...,:width]@full[:width,:width].mT,rtol=2e-7,atol=1e-9)
    delayed=action.apply(x)
    # Saved index tensors also reject mutation during delayed native backward.
    with torch.no_grad():action.indices[0][0]+=1
    with pytest.raises(ValueError,match='changed'):action.apply(x)
    with pytest.raises(RuntimeError):torch.autograd.grad(delayed.real.sum(),x)
