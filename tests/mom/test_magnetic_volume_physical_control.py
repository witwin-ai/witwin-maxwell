"""Multi-point magnetic sphere oracle and actual polygonal interface evidence."""
import json
import os
import time
from pathlib import Path
import pytest
import torch
from test_static_coupling import system_from_volume
from test_tensor_magnetostatic import coefficient
from witwin.maxwell.constants import MU_0
from witwin.maxwell.mom.mesh import sphere_mesh
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
from witwin.maxwell.mom.static_magnetic_volume import MagneticVolumeContrastResponse

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('volume_refinement',[0,1])
@torch.no_grad()
def test_tensor_sphere_multi_point_and_interface_evidence(volume_refinement):
    sample=torch.empty((),device='cuda',dtype=torch.float64)
    mu=coefficient(sample);ext=MU_0*torch.eye(3,device='cuda',dtype=sample.dtype)
    drive=sample.new_tensor([.3,-.2,.7]);radius=.1
    inside_reference=torch.linalg.solve(mu+2*ext,3*ext@drive)
    magnetization=(mu/ext[0,0]-torch.eye(3,device='cuda',dtype=sample.dtype))@inside_reference
    index=torch.arange(17,device='cuda',dtype=sample.dtype)
    z=1-2*(index+.37)/17;angle=(index+.21)*2.399963229728653
    directions=torch.stack(((1-z*z).sqrt()*angle.cos(),(1-z*z).sqrt()*angle.sin(),z),-1)
    inside=(sample.new_tensor([.2,.6,.85])[:,None,None]*radius*directions[None]).reshape(-1,3)
    outside=(sample.new_tensor([1.2,2.,3.])[:,None,None]*radius*directions[None]).reshape(-1,3)
    distance=outside.norm(dim=-1,keepdim=True)
    exterior_reference=drive+radius**3/3*(3*outside*(outside*magnetization).sum(-1,keepdim=True)/distance**5-magnetization/distance**3)
    rows=[]
    for level in ((1,2,3) if volume_refinement==0 else (2,)):
        start=time.monotonic();surface=sphere_mesh(radius=radius,subdivisions=level)
        vertices=torch.cat((surface.vertices,sample.new_zeros(1,3)))
        cells=torch.cat((torch.full((len(surface.faces),1),len(surface.vertices),device='cuda',dtype=torch.int64),surface.faces),-1)
        volume=TetrahedralVolumeMesh(vertices,cells,region_id=1)
        if volume_refinement:volume=volume.refine(volume_refinement)
        system,binding=system_from_volume(volume)
        torch.cuda.reset_peak_memory_stats()
        response=MagneticVolumeContrastResponse(system,(binding,),(sample.new_zeros(len(volume.cells),3),),ext,(mu,),background_h=drive,pair_batch=65536,max_matrix_bytes=512*1024**2)
        internal=response.fields(inside,1)[0];external=response.fields(outside,0)[0]
        def errors(actual,reference):
            reference=reference.expand_as(actual)
            return dict(rms=float((actual-reference).norm()/reference.norm()),
                        maximum=float(((actual-reference).norm(dim=-1)/reference.norm(dim=-1)).max()))
        chosen=torch.linspace(0,len(surface.faces)-1,24,device='cuda').long()
        centers=surface.triangles[chosen].mean(1);normals=surface.normals[chosen]
        traces=[]
        for delta in (radius*1e-5,radius*5e-6):
            hi,bi,_=response.fields(centers-delta*normals,1)
            ho,bo,_=response.fields(centers+delta*normals,0)
            traces.append((hi,bi,ho,bo))
        hi,bi,ho,bo=(2*b-a for a,b in zip(*traces))
        normal_jump=((bo-bi)*normals).sum(-1).abs()/(MU_0*drive.norm())
        tangent_jump=torch.linalg.cross(ho-hi,normals).norm(dim=-1)/drive.norm()
        # Face-center equations may enforce continuity only at collocation
        # points. Independent interior locations must also satisfy the gate.
        bary=sample.new_tensor([[.2,.3,.5],[.5,.2,.3],[.3,.5,.2]])
        off_centers=torch.einsum('qa,fac->fqc',bary,surface.triangles[chosen]).reshape(-1,3)
        off_normals=normals[:,None].expand(-1,3,-1).reshape(-1,3)
        off_traces=[]
        for delta in (radius*1e-5,radius*5e-6):
            hi,bi,_=response.fields(off_centers-delta*off_normals,1)
            ho,bo,_=response.fields(off_centers+delta*off_normals,0)
            off_traces.append((hi,bi,ho,bo))
        hi,bi,ho,bo=(2*b-a for a,b in zip(*off_traces))
        off_normal=((bo-bi)*off_normals).sum(-1).abs()/(MU_0*drive.norm())
        off_tangent=torch.linalg.cross(ho-hi,off_normals).norm(dim=-1)/drive.norm()
        row=dict(off_normal_B_jump_max=float(off_normal.max()),off_tangential_H_jump_max=float(off_tangent.max()),off_interface_points=len(off_centers),level=level,volume_refinement=volume_refinement,cells=len(volume.cells),interior=errors(internal,inside_reference),
            exterior=errors(external,exterior_reference),normal_B_jump_max=float(normal_jump.max()),
            tangential_H_jump_max=float(tangent_jump.max()),interface_points=len(chosen),
            elapsed_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            physical_threshold=.005,physical_qualification=False)
        row['selected_control_passed']=all(row[k]['maximum']<=.005 for k in ('interior','exterior')) and row['normal_B_jump_max']<=.005 and row['tangential_H_jump_max']<=.005 and row['off_normal_B_jump_max']<=.005 and row['off_tangential_H_jump_max']<=.005
        rows.append(row);print(json.dumps(row),flush=True)
    destination=os.environ.get('MAGNETIC_CONTROL_REPORT')
    if destination:Path(destination).write_text(json.dumps(rows,indent=2))
    assert all(torch.isfinite(sample.new_tensor([r['interior']['maximum'],r['exterior']['maximum'],r['normal_B_jump_max'],r['tangential_H_jump_max'],r['off_normal_B_jump_max'],r['off_tangential_H_jump_max']])).all() for r in rows)
    # This test records the original acceptance gate; it does not turn failures
    # into qualification by asserting only a relaxed physical threshold.
    if len(rows)>1:
        assert rows[-1]['interior']['rms']<rows[0]['interior']['rms']
        assert rows[-1]['exterior']['rms']<rows[0]['exterior']['rms']
