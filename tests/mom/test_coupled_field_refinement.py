"""Streamed complete-column decisions match materialized response differences."""
import runpy
import pytest
import torch
from witwin.maxwell.mom.coupled_field_refinement import compare_coupled_field_orders
from witwin.maxwell.mom.coupled_volume_action import CoupledVolumeAction
from witwin.maxwell.mom.implicit_surface_solve import FixedSurfaceInverse
from witwin.maxwell.mom.surface_volume_hmatrix import SurfaceVolumeBlocks,SurfaceVolumeHierarchicalAction,SurfaceVolumeHConfig
from witwin.maxwell.mom.interior_volume_hmatrix import InteriorVolumeBlocks,InteriorVolumeHierarchicalAction,InteriorVolumeHConfig
from witwin.maxwell.mom.volume_hmatrix import VolumeCellBlocks,VolumeHierarchicalAction,VolumeHConfig
from witwin.maxwell.mom.hmatrix import HierarchicalOperator,HMatrixConfig
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.fields import FieldQueryConfig
from witwin.maxwell.mom.mesh import SurfaceSystem,RWGBasis
from witwin.maxwell.mom.volume_mesh import bind_volume_region

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def pair(field):
    op,bindings=runpy.run_path('tests/mom/test_surface_volume.py')['problem']((2.,3.))
    volumes=[bindings[0].volume.refine(1),bindings[1].volume]
    system=SurfaceSystem.create([(i,str(i),RWGBasis(v.boundary),op.system.bodies[i].interior) for i,v in enumerate(volumes)])
    bindings=[bind_volume_region(v,system) for v in reversed(volumes)]
    h=HierarchicalOperator(ScalarSystemBlocks(system,op.exterior,op.integration),HMatrixConfig(leaf_size=64))
    inverse=FixedSurfaceInverse(h,StoppingRule(method='block_gmres',preconditioner='row_column_norm',rtol=1e-11,atol=1e-13,stagnation_window=48))
    sources=[SurfaceVolumeHierarchicalAction(SurfaceVolumeBlocks(b,order=3,inner_order=3),SurfaceVolumeHConfig(leaf_size=64)) for b in bindings]
    def action(order):
        fields=[InteriorVolumeHierarchicalAction(InteriorVolumeBlocks(b,field=field,order=order),InteriorVolumeHConfig(leaf_size=64)) for b in bindings]
        direct=[VolumeHierarchicalAction(VolumeCellBlocks(b.volume,system.bodies[b.volume.region_id-1].interior,field=field,order=order),VolumeHConfig(leaf_size=64)) for b in bindings]
        return CoupledVolumeAction(inverse,sources,fields,direct)
    return action(3),action(6)


@pytest.mark.parametrize('field',['E','H'])
def test_streamed_all_columns_complete_field_and_shared_solves(field,monkeypatch):
    coarse,fine=pair(field);n=coarse.shape[0]
    eye=torch.eye(n,device='cuda',dtype=coarse.dtype)
    old=coarse.apply(eye).reshape(n,-1,3)
    new=fine.apply(eye).reshape(n,-1,3)
    error=(new-old).norm(dim=-1);magnitude=new.norm(dim=-1)
    config=FieldQueryConfig()
    absolute=torch.tensor([config.absolute_tolerance/(abs(s.provider.binding.surface_system.bodies[s.provider.binding.volume.region_id-1].interior.eta) if field=='H' else 1) for s,size in zip(coarse.sources,coarse.sizes) for _ in range(size//3)],device='cuda',dtype=old.real.dtype)
    threshold=absolute+config.relative_tolerance*magnitude
    calls=[];original=coarse.inverse.apply
    def counted(value):calls.append(value.shape);return original(value)
    monkeypatch.setattr(coarse.inverse,'apply',counted)
    receipt=compare_coupled_field_orders(coarse,fine,column_batch=4,config=config)
    assert len(calls)==7 and calls[-1][0]==3 and sum(shape[0] for shape in calls)==n
    assert receipt['passed']==bool((error<=threshold).all())
    assert sum(c['failed_column_query_pairs'] for c in receipt['checks'])==int((error>threshold).sum())
    assert max(c['maximum_difference'] for c in receipt['checks'])==pytest.approx(float(error.max()),rel=1e-8,abs=1e-12)
    assert not receipt['certified_error_bound']
    for kwargs in ({'column_batch':0},{'column_batch':True},{'config':FieldQueryConfig(adaptive=False)}, {'config':FieldQueryConfig(max_order=4)}):
        with pytest.raises(ValueError):compare_coupled_field_orders(coarse,fine,**kwargs)
    with pytest.raises(ValueError,match='increase'):compare_coupled_field_orders(coarse,coarse)
    subset=CoupledVolumeAction(fine.inverse,fine.sources[:1],fine.fields[:1],fine.direct[:1])
    with pytest.raises(ValueError,match='identical'):compare_coupled_field_orders(coarse,subset)
