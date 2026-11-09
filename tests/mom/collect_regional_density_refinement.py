"""Fixed-polyhedron diagnostic: compare local and patch-continuous P1 density.

Run directly with GPU visibility selected externally. MAGNETIC_CONTROL_REPORT
sets the JSON destination. This is a collector, not physical acceptance: every
row retains the 0.005 combined gate and physical_qualification=False. The sphere
oracle is only an approximation control for this fixed polyhedron.
"""
import json
import os
import time
from pathlib import Path
import torch
from test_tensor_magnetostatic import coefficient
from witwin.maxwell.constants import MU_0
from witwin.maxwell.mom.mesh import sphere_mesh
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import SurfaceSystem, RWGBasis
from witwin.maxwell.mom.physics import compile_medium
from static_regional_density_diagnostic import RegionalDensityDiagnostic





@torch.no_grad()
def collect(*,refinements=(1,),spaces=('discontinuous','planar_patch'),degree=1):
    """Compare declared density refinements without changing the base surface."""
    if type(degree) is not int or degree not in (1,2):
        raise ValueError('Diagnostic density degree must be 1 or 2')
    if not refinements or any(type(level) is not int or level<0 for level in refinements):
        raise ValueError('Refinements must be a nonempty sequence of nonnegative integers')
    if not spaces or any(space not in ('discontinuous','planar_patch') for space in spaces):
        raise ValueError('Choose discontinuous or planar_patch density spaces')
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
    base=sphere_mesh(radius=radius,subdivisions=1)
    for refinement in refinements:
        level=1
        start=time.monotonic();surface=base.refine(refinement)
        torch.testing.assert_close(surface.areas.sum(),base.areas.sum(),rtol=1e-13,atol=0)
        parent=base.triangles.repeat_interleave(4**refinement,0)
        parent_normal=base.normals.repeat_interleave(4**refinement,0)
        assert float(((surface.triangles-parent[:,:1])*parent_normal[:,None]).sum(-1).abs().max())<1e-14
        def signed_volume(mesh):
            t=mesh.triangles
            return (t[:,0]*torch.linalg.cross(t[:,1],t[:,2])).sum()/6
        torch.testing.assert_close(signed_volume(surface),signed_volume(base),rtol=1e-13,atol=0)
        # Construct directly from the refined mesh, so parent IDs follow the
        # documented midpoint child ordering without a volume-boundary reorder.
        system=SurfaceSystem.create([(0,'dielectric',RWGBasis(surface),compile_medium(mw.Material(eps_r=2.),3e8))])
        torch.cuda.reset_peak_memory_stats()
        operator=RegionalDensityDiagnostic(system,ext,(mu,),pair_batch=8192,degree=degree)
        patch_ids=torch.arange(len(base.faces),device=surface.device).repeat_interleave(4**refinement)
        for continuity in spaces:
            native=operator.solve_uniform(drive,patch_ids=patch_ids if continuity=='planar_patch' else None)
            class Decode:
                def fields(self,points,region):
                    _,h=native.fields(points,region)
                    return h,h@(ext if region==0 else mu).T,{}
            response=Decode()
            internal=response.fields(inside,1)[0];external=response.fields(outside,0)[0]
            def errors(actual,reference):
                reference=reference.expand_as(actual)
                return dict(rms=float((actual-reference).norm()/reference.norm()),
                            maximum=float(((actual-reference).norm(dim=-1)/reference.norm(dim=-1)).max()))
            chosen=torch.linspace(0,len(base.faces)-1,24,device='cuda').long()
            centers=base.triangles[chosen].mean(1);normals=base.normals[chosen]
            traces=[]
            for delta in (radius*1e-5,radius*5e-6):
                hi,bi,_=response.fields(centers-delta*normals,1)
                ho,bo,_=response.fields(centers+delta*normals,0)
                traces.append((hi,bi,ho,bo))
            hi,bi,ho,bo=(2*b-a for a,b in zip(*traces))
            normal_jump=((bo-bi)*normals).sum(-1).abs()/(MU_0*drive.norm())
            tangent_jump=torch.linalg.cross(ho-hi,normals).norm(dim=-1)/drive.norm()
            off_bary=sample.new_tensor([[.2,.3,.5],[.5,.2,.3],[.3,.5,.2]])
            off_centers=torch.einsum('qa,fac->fqc',off_bary,base.triangles[chosen]).reshape(-1,3)
            off_normals=normals[:,None].expand(-1,3,-1).reshape(-1,3)
            off_traces=[]
            for delta in (radius*1e-5,radius*5e-6):
                hi,bi,_=response.fields(off_centers-delta*off_normals,1)
                ho,bo,_=response.fields(off_centers+delta*off_normals,0)
                off_traces.append((hi,bi,ho,bo))
            hi,bi,ho,bo=(2*b-a for a,b in zip(*off_traces))
            off_normal=((bo-bi)*off_normals).sum(-1).abs()/(MU_0*drive.norm())
            off_tangent=torch.linalg.cross(ho-hi,off_normals).norm(dim=-1)/drive.norm()
            nonedge_bary=sample.new_tensor([[.19,.33,.48],[.48,.19,.33],[.33,.48,.19]])
            nonedge_centers=torch.einsum('qa,fac->fqc',nonedge_bary,base.triangles[chosen]).reshape(-1,3)
            nonedge_traces=[]
            for delta in (radius*1e-5,radius*5e-6):
                hi,bi,_=response.fields(nonedge_centers-delta*off_normals,1)
                ho,bo,_=response.fields(nonedge_centers+delta*off_normals,0)
                nonedge_traces.append((hi,bi,ho,bo))
            hi,bi,ho,bo=(2*b-a for a,b in zip(*nonedge_traces))
            nonedge_normal=((bo-bi)*off_normals).sum(-1).abs()/(MU_0*drive.norm())
            nonedge_tangent=torch.linalg.cross(ho-hi,off_normals).norm(dim=-1)/drive.norm()
            row=dict(surface_refinement=refinement,density_degree=degree,fixed_geometry=True,continuity=continuity,density_dofs=native.density_dofs,local_residual=native.local_residual,additional_nonedge_normal_B_jump_max=float(nonedge_normal.max()),additional_nonedge_tangential_H_jump_max=float(nonedge_tangent.max()),off_normal_B_jump_max=float(off_normal.max()),off_tangential_H_jump_max=float(off_tangent.max()),off_interface_points=len(off_centers),reference='sphere_approximation_for_fixed_polyhedron',level=level,faces=len(surface.faces),interior=errors(internal,inside_reference),
                exterior=errors(external,exterior_reference),normal_B_jump_max=float(normal_jump.max()),
                tangential_H_jump_max=float(tangent_jump.max()),interface_points=len(chosen),
                elapsed_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                physical_threshold=.005,physical_qualification=False,relative_algebraic_residual=native.residual,quadrature=native.quadrature)
            row['selected_control_passed']=all(row[k]['maximum']<=.005 for k in ('interior','exterior')) and row['normal_B_jump_max']<=.005 and row['tangential_H_jump_max']<=.005 and row['off_normal_B_jump_max']<=.005 and row['off_tangential_H_jump_max']<=.005 and row['additional_nonedge_normal_B_jump_max']<=.005 and row['additional_nonedge_tangential_H_jump_max']<=.005
            rows.append(row);print(json.dumps(row),flush=True)
    destination=os.environ.get('MAGNETIC_CONTROL_REPORT')
    if destination:Path(destination).write_text(json.dumps(rows,indent=2))
    assert all(torch.isfinite(sample.new_tensor([r['interior']['maximum'],r['exterior']['maximum'],r['normal_B_jump_max'],r['tangential_H_jump_max'],r['off_normal_B_jump_max'],r['off_tangential_H_jump_max'],r['additional_nonedge_normal_B_jump_max'],r['additional_nonedge_tangential_H_jump_max']])).all() for r in rows)

    return rows


def test_fixed_geometry_density_evidence():
    collect()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refinements',nargs='+',type=int,default=[1],
                        help='Planar midpoint refinement levels of the same 80-face surface')
    parser.add_argument('--spaces',nargs='+',choices=('discontinuous','planar_patch'),
                        default=['discontinuous','planar_patch'])
    parser.add_argument('--degree',type=int,choices=(1,2),default=1)
    args=parser.parse_args()
    collect(refinements=tuple(args.refinements),spaces=tuple(args.spaces),degree=args.degree)
