"""Matched uniform/graded P1 density triangulations on fixed physical faces."""
import argparse,hashlib,json,os,time,importlib.util
from pathlib import Path
import torch
from witwin.maxwell.mom.mesh import SurfaceMesh
from witwin.maxwell.mom.static_tensor_green import StaticTensorGreen
from witwin.maxwell.mom.quadrature import triangle_rule
from static_regional_density_diagnostic import RegionalDensityDiagnostic
import collect_regional_density_refinement as collector

@torch.no_grad()
def collect(raw_state,output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    os.environ['MAGNETIC_CONTROL_REPORT']=str(root/'physical.json')
    source=Path(raw_state);raw=torch.load(source,map_location='cuda',weights_only=True)
    import witwin.maxwell as mw
    import maxwell_bench
    import static_regional_density_diagnostic as density_helper
    provider=Path(__file__).resolve().parents[2]
    assert Path(mw.__file__).resolve().is_relative_to(provider)
    for mod in (density_helper,collector):
        assert Path(mod.__file__).resolve()==provider/'tests/mom'/(mod.__name__+'.py')
    bench=Path(maxwell_bench.__file__).resolve().parents[1]
    inv={str(repo.name)+'/'+str(path.relative_to(repo)):hashlib.sha256(path.read_bytes()).hexdigest()
        for repo,folders in ((bench,('maxwell_bench','tests')),(provider,('witwin/maxwell','tests/mom')))
        for folder in folders for path in (repo/folder).rglob('*.py')}
    (root/'source_inventory.json').write_text(json.dumps(inv,sort_keys=True,indent=2))
    assert raw['source_inventory_sha256']==hashlib.sha256(json.dumps(inv,sort_keys=True).encode()).hexdigest()
    helper=Path(__file__).with_name('diagnose_mom_magnetic_density_bubbles.py')
    (root/'diagnostic_sources.json').write_text(json.dumps({str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),helper)},indent=2))
    spec=importlib.util.spec_from_file_location('bubble_diagnostic',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    op=object.__new__(RegionalDensityDiagnostic);op.mesh=SurfaceMesh(raw['vertices'],raw['faces'])
    op.exterior=StaticTensorGreen(raw['exterior']);op.interior=StaticTensorGreen(raw['interior']);op.field=raw['field']
    op.degree=1;op.nb=3;op.pair_batch=8192
    base=SurfaceMesh(raw['parent_vertices'],raw['parent_faces']);uniform=base.refine(2)
    source_owner=torch.arange(len(base.faces),device='cuda').repeat_interleave(16)
    parent=base.triangles[source_owner]
    edges=torch.stack((parent[:,1]-parent[:,0],parent[:,2]-parent[:,0]),-1)
    uv=torch.linalg.solve(edges.mT@edges,torch.einsum('pij,pqi->pjq',edges,uniform.triangles-parent[:,:1])).mT
    bary=torch.cat((1-uv.sum(-1,keepdim=True),uv),-1)
    dyadic=(bary*4).round()/4
    torch.testing.assert_close(bary,dyadic,rtol=0,atol=1e-13)
    assert (dyadic>=0).all() and torch.equal(dyadic.sum(-1),torch.ones_like(dyadic[:,:,0]))
    keys=torch.stack((source_owner[:,None].expand(-1,3).reshape(-1),uniform.faces.reshape(-1)),1)
    unique,inverse=torch.unique(keys,dim=0,return_inverse=True);node_ids=inverse.reshape(-1,3)
    nf=len(uniform.faces);nd=len(unique)
    points,w=triangle_rule(op.mesh.triangles,6);points=points.reshape(-1,3);nq=len(points)
    target_owner=torch.arange(len(op.mesh.faces),device='cuda').repeat_interleave(36)
    target_parent=raw['patch_ids'][target_owner];normals=op.mesh.normals[target_owner]
    weight=(w.reshape(-1)/w.sum()).sqrt();mu=op.exterior.coefficient.diagonal().mean()
    extent=(base.triangles.amax((0,1))-base.triangles.amin((0,1))).norm()
    rhs=torch.cat((torch.cat((points.new_zeros(3*nq),normals@((op.exterior.coefficient-op.interior.coefficient)@op.field)/mu))*weight.repeat(4),points.new_zeros(1)))
    original=collector.RegionalDensityDiagnostic
    def factory(system,exterior,interiors,**kwargs):
        assert torch.equal(system.bodies[0].mesh.vertices,op.mesh.vertices) and torch.equal(system.bodies[0].mesh.faces,op.mesh.faces)
        assert torch.equal(exterior,op.exterior.coefficient) and torch.equal(interiors[0],op.interior.coefficient)
        return op
    def solved(field,*,patch_ids=None):
        assert torch.equal(field,op.field) and torch.equal(patch_ids,raw['patch_ids'])
        return op
    op.solve_uniform=solved;results=[]
    try:
        collector.RegionalDensityDiagnostic=factory
        for power in (1,2):
            start=time.monotonic()
            mapped=dyadic.pow(power);mapped=mapped/mapped.sum(-1,keepdim=True)
            triangles=torch.einsum('pva,pac->pvc',mapped,parent)
            cross=torch.linalg.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
            assert ((cross*base.normals[source_owner]).sum(-1)>0).all()
            torch.testing.assert_close(cross.norm(dim=-1).reshape(-1,16).sum(-1)/2,base.areas,rtol=1e-13,atol=0)
            op.triangles=triangles;op.nf=nf
            evaluation=points.new_zeros(nq,nd);pv_counts=torch.zeros(nq,device='cuda',dtype=torch.int64)
            layers=[]
            for region,kernel in enumerate((op.exterior,op.interior)):
                value=points.new_zeros(nq,nd);grad=points.new_zeros(nq,nd,3)
                for begin in range(0,nq*nf,8192):
                    flat=torch.arange(begin,min(begin+8192,nq*nf),device='cuda');i=flat//nf;j=flat%nf
                    tri=triangles[j];e=torch.stack((tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),-1)
                    uv=torch.linalg.solve(e.mT@e,e.mT@(points[i]-tri[:,0])[:,:,None]).squeeze(-1)
                    local=torch.cat((1-uv.sum(-1,keepdim=True),uv),-1)
                    same=target_parent[i]==source_owner[j]
                    assert not (same&(local.amin(-1)>=-1e-10)&(local.abs().amin(-1)<1e-10)).any(), 'Training point on integration edge'
                    pv=same&(local.amin(-1)>0)
                    v,g=kernel.panel_linear_integrals(points[i],tri,principal_value=pv)
                    ii=i[:,None].expand(-1,3).reshape(-1);jj=node_ids[j].reshape(-1)
                    value.index_put_((ii,jj),v.reshape(-1),accumulate=True)
                    grad.index_put_((ii,jj),g.reshape(-1,3),accumulate=True)
                    if region==0:
                        evaluation.index_put_((ii,jj),torch.where(pv[:,None],local,0).reshape(-1),accumulate=True)
                        pv_counts.index_add_(0,i,pv.to(torch.int64))
                layers.append((value,grad))
            assert torch.equal(pv_counts,torch.ones_like(pv_counts))
            v=torch.cat((layers[0][0],-layers[1][0]),1)
            flux=torch.cat(((layers[0][1]@op.exterior.coefficient.T*normals[:,None]).sum(-1)-.5*evaluation,
                -(layers[1][1]@op.interior.coefficient.T*normals[:,None]).sum(-1)-.5*evaluation),1)
            g=torch.cat((layers[0][1],-layers[1][1]),1)
            tangent=g-(g*normals[:,None]).sum(-1,keepdim=True)*normals[:,None]
            gauge=(v*weight.square()[:,None]).sum(0)/extent
            matrix=torch.cat((torch.cat([tangent[:,:,axis] for axis in range(3)]+[flux/mu],0)*weight.repeat(4)[:,None],gauge[None]),0)
            del layers,g,tangent,v,flux,value,grad
            prepare_seconds=time.monotonic()-start
            solve_start=time.monotonic();density=torch.linalg.lstsq(matrix,rhs).solution
            op.external=density[:nd][node_ids].reshape(-1);op.internal=density[nd:][node_ids].reshape(-1)
            op.residual=float((matrix@density-rhs).norm()/rhs.norm());op.local_residual=None;op.density_dofs=nd
            solve_seconds=time.monotonic()-solve_start
            op.quadrature=[dict(formulation='physical_trace_least_squares',target_order=6,density_grading_power=power,certified=False)]
            training=module.physical_trace_residual_summary(matrix,rhs,density,weight,op.field,points,base.triangles[target_parent])
            torch.save(dict(kind='diagnostic_graded_density_not_owned_state',source_raw_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),triangles=triangles,node_ids=node_ids,matrix=matrix,rhs=rhs,density=density,points=points,power=power),root/('power'+str(power)+'.pt'))
            rows=collector.collect(refinements=(1,),spaces=('planar_patch',),degree=1)
            for row in rows:row.update(density_integration_panels=nf,density_grading_power=power,training_physical_residuals=training,preparation_seconds=prepare_seconds,solve_seconds=solve_seconds,timing_scope='collector_fields_only_preassembled_fit',local_residual_scope='unavailable_without_weak_blocks')
            results+=rows
            Path(os.environ['MAGNETIC_CONTROL_REPORT']).write_text(json.dumps(results,indent=2,allow_nan=False))
            print('GRADED_DENSITY_COMPLETE',power,training,flush=True)
            del matrix,density,evaluation
    finally:collector.RegionalDensityDiagnostic=original
    Path(os.environ['MAGNETIC_CONTROL_REPORT']).write_text(json.dumps(results,indent=2,allow_nan=False))
    assert all(not row['physical_qualification'] for row in results)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-state',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    collect(args.raw_state,args.output)
