"""Projection resource measurement at the original failed 3840-edge geometry."""
import json
import time
from pathlib import Path
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis,SurfaceSystem,sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.cluster import build_support_octree
from witwin.maxwell.mom.fmm_moments import NestedRWGMultipoles
from witwin.maxwell.mom.fmm_shift import shift_storage_bytes

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def measured(function):
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    baseline=torch.cuda.memory_allocated();start=time.monotonic()
    value=function();torch.cuda.synchronize()
    return value,dict(seconds=time.monotonic()-start,baseline_allocated_bytes=baseline,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        end_allocated_bytes=torch.cuda.memory_allocated())


def test_original_3840_edge_geometry_unreduced_order12_projection_resource(tmp_path):
    # Same geometry/frequency/leaf size as original fixed-ka scaling point.
    exterior=compile_medium(mw.Material(),3e8);interior=compile_medium(mw.Material(eps_r=2.),3e8)
    system=SurfaceSystem.create([(i,str(i),RWGBasis(sphere_mesh(radius=.01,center=(x,0.,0.),
        subdivisions=3,device='cuda',dtype=torch.float64)),interior) for i,x in enumerate((-.5,.5))])
    assert system.total_edges==3840
    provider=ScalarSystemBlocks(system,exterior,IntegrationConfig(order=4,singular_order=3))
    lower,upper=provider.support_boxes();tree=build_support_octree(lower,upper,leaf_size=30)
    order=12;count=(order+1)*(order+2)*(order+3)//6
    common=4*tree.node_count*8+8*tree.size+4*tree.size*count*16
    dense_bytes=common+shift_storage_bytes(order,tree.node_count,8,'dense')
    compact_bytes=common+shift_storage_bytes(order,tree.node_count,8,'separable_v1')
    print(dict(phase='projection_plan',edges=3840,order=12,nodes=tree.node_count,
        dense_projection_owned_bytes=dense_bytes,separable_projection_owned_bytes=compact_bytes),flush=True)
    projection,setup=measured(lambda:NestedRWGMultipoles(provider,tree,order=order,
        shift_storage='separable_v1',retained_budget_bytes=1024**3))
    assert projection.receipt()['owned_retained_bytes']==compact_bytes
    assert compact_bytes<dense_bytes and compact_bytes<=1024**3
    torch.manual_seed(1729)
    x=torch.randn((2,tree.size),device=tree.device,dtype=torch.complex128,requires_grad=True)
    values,upward=measured(lambda:projection.upward(x))
    coefficients=tuple(torch.randn_like(value) for value in values)
    adjoint,downward=measured(lambda:projection.downward(coefficients))
    loss=sum((coefficient.conj()*value).sum().real for coefficient,value in zip(coefficients,values))
    gradient,backward=measured(lambda:torch.autograd.grad(loss,x)[0])
    gradient_error=float((gradient-adjoint).norm()/adjoint.norm())
    record=dict(edges=3840,nodes=tree.node_count,order=12,frequency_Hz=3e8,
        dense_projection_owned_bytes=dense_bytes,separable_projection_owned_bytes=compact_bytes,
        setup=setup,upward=upward,downward=downward,backward=backward,
        relative_current_gradient_error=gradient_error,projection_retained_cap_bytes=1024**3,
        original_full_fmm_failure_closed=False,full_fmm_constructed=False,
        physical_qualification=False,complexity_qualification=False,full_peak_qualification=False,
        limitations=['projection scope only','single process on shared GPU','no full near/M2L/system/solve measurement',
            'original table/kernel-bound failures remain open','no physical/scaling qualification'])
    (tmp_path/'resource.json').write_text(json.dumps(record,indent=2)+'\n')
    print(record,flush=True)
    assert gradient_error<1e-11
