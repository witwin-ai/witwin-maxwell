"""Frozen-source diagnostic: matched P2 versus P2 plus centroid hats."""
import argparse,hashlib,json,os,time
from pathlib import Path
import torch
from witwin.maxwell.mom.mesh import SurfaceMesh
from witwin.maxwell.mom.static_tensor_green import StaticTensorGreen
from witwin.maxwell.mom.quadrature import triangle_rule
import static_regional_density_diagnostic as density_diagnostic
from static_regional_density_diagnostic import RegionalDensityDiagnostic
import test_static_tensor_quadratic_weak as quadratic_helpers
from test_static_tensor_quadratic_weak import physical_basis
import collect_regional_density_refinement as collector

def edge_density_hat_panels(parent_triangles,width):
    """Return two P1 source panels for each original-face edge-midpoint hat.

    Shape is (face, edge, subpanel, vertex, xyz), edge order 01/12/20.
    Column zero of each subpanel is the hat's midpoint vertex. The other
    vertices have zero density. Width is a fraction of the opposite altitude;
    this changes integration support, not physical geometry. Hats on distinct
    original faces have independent coefficients. Exact artificial-edge PV
    queries still require the underlying kernel's strict admission.
    """
    if type(width) not in (int,float) or not 0<width<1:
        raise ValueError('Edge density support width must lie strictly between zero and one')
    if parent_triangles.ndim!=3 or parent_triangles.shape[1:]!=(3,3):
        raise ValueError('Edge density hats require physical triangle vertices')
    if not bool(torch.isfinite(parent_triangles).all()):
        raise ValueError('Edge density support vertices must be finite')
    if not bool((torch.linalg.cross(parent_triangles[:,1]-parent_triangles[:,0],parent_triangles[:,2]-parent_triangles[:,0]).norm(dim=-1)>0).all()):
        raise ValueError('Edge density support faces must be nondegenerate')
    pieces=[]
    for a,b,c in ((0,1,2),(1,2,0),(2,0,1)):
        mid=(parent_triangles[:,a]+parent_triangles[:,b])/2
        tip=(1-width)*mid+width*parent_triangles[:,c]
        pieces.append(torch.stack((torch.stack((mid,tip,parent_triangles[:,a]),1),torch.stack((mid,parent_triangles[:,b],tip),1)),1))
    return torch.stack(pieces,1)


@torch.no_grad()
def physical_trace_residual_summary(matrix,rhs,solution,weight,field,points,parent_triangles):
    """Report physical training errors without confusing them with LS norms.

    Rows are three Cartesian tangential gradients, normal flux divided by the
    exterior mean coefficient, and one gauge. Bins use minimum barycentric
    coordinate on ORIGINAL physical faces, not source integration subtriangles.
    This is a spatial diagnostic, not independent acceptance or a singularity
    certificate. Every training point appears in exactly one bin.
    """
    nq=len(points)
    if matrix.shape[0]!=4*nq+1 or weight.shape!=(nq,) or parent_triangles.shape!=(nq,3,3):
        raise ValueError('Physical trace diagnostic dimensions do not match')
    if not bool((weight>0).all()) or not bool(field.norm()>0):
        raise ValueError('Physical trace diagnostic requires positive weights and drive')
    residual=((matrix@solution-rhs)[:-1].reshape(4,nq)/weight).T/field.norm()
    if not bool(torch.isfinite(residual).all()):
        raise ValueError('Physical trace residual is nonfinite')
    edges=torch.stack((parent_triangles[:,1]-parent_triangles[:,0],parent_triangles[:,2]-parent_triangles[:,0]),-1)
    uv=torch.linalg.solve(edges.mT@edges,(edges.mT@(points-parent_triangles[:,0])[:,:,None])).squeeze(-1)
    bary=torch.cat((1-uv.sum(-1,keepdim=True),uv),-1)
    if not bool((bary>=-1e-10).all()):
        raise ValueError('Training points are outside their declared original faces')
    distance=bary.amin(-1)
    normal=residual[:,3].abs();tangent=residual[:,:3].norm(dim=1);combined=residual.norm(dim=1)
    def metrics(select):
        return dict(count=int(select.sum()),normal_max=float(normal[select].max()),
                    tangent_max=float(tangent[select].max()),combined_max=float(combined[select].max()))
    bins=[];lower=0.
    for upper in (.01,.025,.05,.1,float('inf')):
        select=(distance>=lower-1e-10 if lower==0 else distance>=lower)&(distance<upper)
        if bool(select.any()):bins.append(dict(min_bary_lower=lower,min_bary_upper=None if upper==float('inf') else upper,**metrics(select)))
        lower=upper
    assert sum(row['count'] for row in bins)==nq
    return dict(scope='training_PV_traces_not_independent_acceptance',
        **metrics(torch.ones(nq,device=points.device,dtype=torch.bool)),
        area_weighted_l4=float((weight.square()*combined.pow(4)).sum().pow(.25)),
        original_face_barycentric_bins=bins)


@torch.no_grad()
def collect(raw_state, output, *, order=4):
    if type(order) is not int or order not in (4,6):
        raise ValueError('This diagnostic supports checked target orders 4 and 6 only')
    root=Path(output)
    root.mkdir(parents=True,exist_ok=False)
    os.environ['MAGNETIC_CONTROL_REPORT']=str(root/'physical.json')
    source=Path(raw_state)
    raw=torch.load(source,map_location='cuda',weights_only=True)
    # Match the frozen experiment inventory before reusing its raw tensors.
    # This includes both provider/test sources and the imported Bench sources.
    import witwin.maxwell as mw
    import maxwell_bench
    provider=Path(__file__).resolve().parents[2]
    assert Path(mw.__file__).resolve().is_relative_to(provider)
    for module in (density_diagnostic,quadratic_helpers,collector):
        assert Path(module.__file__).resolve()==provider/'tests/mom'/(module.__name__+'.py'), 'Diagnostic helper import does not match hashed provider'
    bench=Path(maxwell_bench.__file__).resolve().parents[1]
    inventory={str(repo.name)+'/'+str(path.relative_to(repo)):hashlib.sha256(path.read_bytes()).hexdigest()
        for repo,folders in ((bench,('maxwell_bench','tests')),(provider,('witwin/maxwell','tests/mom')))
        for folder in folders for path in (repo/folder).rglob('*.py')}
    (root/'source_inventory.json').write_text(json.dumps(inventory,sort_keys=True,indent=2))
    assert raw['source_inventory_sha256']==hashlib.sha256(json.dumps(inventory,sort_keys=True).encode()).hexdigest()
    op=object.__new__(RegionalDensityDiagnostic)
    op.mesh=SurfaceMesh(raw['vertices'],raw['faces']);op.triangles=op.mesh.triangles
    op.exterior=StaticTensorGreen(raw['exterior']);op.interior=StaticTensorGreen(raw['interior'])
    op.degree=2;op.nb=6;op.nf=len(op.mesh.faces);op.nodes=raw['nodes'];op.pair_batch=8192
    op.field=raw['field'];nf=op.nf;n=6*nf
    start=time.monotonic()
    x,w=triangle_rule(op.triangles,order);points=x.reshape(-1,3);nq=len(points)
    owners=torch.arange(nf,device='cuda').repeat_interleave(order**2)
    normals=op.mesh.normals[owners]
    edges=torch.stack((op.triangles[:,1]-op.triangles[:,0],op.triangles[:,2]-op.triangles[:,0]),-1)
    uv=torch.linalg.solve(edges.mT@edges,torch.einsum('pij,pqi->pjq',edges,x-op.triangles[:,:1])).mT
    bary=torch.cat((1-uv.sum(-1,keepdim=True),uv),-1).reshape(-1,3)
    sorted_bary=bary.sort(-1).values
    assert (sorted_bary[:,1]-sorted_bary[:,0]).min()>1e-5
    sub_owner=(bary.argmin(-1)+1)%3
    centroid=op.triangles.mean(1)
    sub=torch.stack([torch.stack((centroid,op.triangles[:,a],op.triangles[:,b]),1) for a,b in ((0,1),(1,2),(2,0))],1).reshape(-1,3,3)
    keys=torch.stack((raw['patch_ids'][:,None].expand(-1,6).reshape(-1),op.nodes.reshape(-1)),1)
    unique,inverse=torch.unique(keys,dim=0,return_inverse=True)
    p=points.new_zeros(n,len(unique));p[torch.arange(n,device='cuda'),inverse]=1
    evaluation=points.new_zeros(nq,n).scatter_(1,6*owners[:,None]+torch.arange(6,device='cuda')[None],physical_basis(x,op.triangles).reshape(nq,6))@p
    bubble_eval=points.new_zeros(nq,nf).scatter_(1,owners[:,None],3*bary.amin(-1,keepdim=True))
    weight=(w.reshape(-1)/w.sum()).sqrt()
    extent=(op.triangles.amax((0,1))-op.triangles.amin((0,1))).norm()
    mu=op.exterior.coefficient.diagonal().mean()
    layers=[]
    for kernel in (op.exterior,op.interior):
        value=points.new_empty(nq,nf,6);grad=points.new_empty(nq,nf,6,3)
        for begin in range(0,nq*nf,8192):
            flat=torch.arange(begin,min(begin+8192,nq*nf),device='cuda');i=flat//nf;j=flat%nf
            value[i,j],grad[i,j]=kernel.panel_quadratic_integrals(points[i],op.triangles[j],principal_value=owners[i]==j)
        pv=value.reshape(nq,n)@p
        pg=torch.stack([grad.reshape(nq,n,3)[:,:,axis]@p for axis in range(3)],-1)
        del value,grad
        bv=points.new_zeros(nq,nf);bg=points.new_zeros(nq,nf,3)
        for begin in range(0,nq*len(sub),8192):
            flat=torch.arange(begin,min(begin+8192,nq*len(sub)),device='cuda');i=flat//len(sub);j=flat%len(sub)
            v,g=kernel.panel_linear_integrals(points[i],sub[j],principal_value=(owners[i]==j//3)&(sub_owner[i]==j%3))
            # Each batch can contain three contributions to the same parent.
            bv.index_put_((i,j//3),v[:,0],accumulate=True)
            bg.index_put_((i,j//3),g[:,0],accumulate=True)
        layers.append((torch.cat((pv,bv),1),torch.cat((pg,bg),1)))
    ev=torch.cat((evaluation,bubble_eval),1);nd=len(unique)+nf
    value=torch.cat((layers[0][0],-layers[1][0]),1)
    flux=torch.cat(((layers[0][1]@op.exterior.coefficient.T*normals[:,None]).sum(-1)-.5*ev,
                    -(layers[1][1]@op.interior.coefficient.T*normals[:,None]).sum(-1)-.5*ev),1)
    gradient=torch.cat((layers[0][1],-layers[1][1]),1)
    tangent=gradient-(gradient*normals[:,None]).sum(-1,keepdim=True)*normals[:,None]
    gauge=(value*weight.square()[:,None]).sum(0)/extent
    matrix=torch.cat((torch.cat([tangent[:,:,axis] for axis in range(3)]+[flux/mu],0)*weight.repeat(4)[:,None],gauge[None]),0)
    flux_rhs=normals@((op.exterior.coefficient-op.interior.coefficient)@op.field)
    rhs=torch.cat((torch.cat((flux_rhs.new_zeros(3*nq),flux_rhs/mu))*weight.repeat(4),flux_rhs.new_zeros(1)))
    torch.save(dict(kind='diagnostic_bubble_blocks_not_owned_state',source_raw_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),matrix=matrix,rhs=rhs,projection=p,points=points,subtriangles=sub),root/'blocks.pt')
    preparation_seconds=time.monotonic()-start
    del layers,gradient,tangent,value,flux
    print('MATCHED_BUBBLE_BLOCKS_SAVED',flush=True)
    base_fields=op.fields
    def fields(query,region):
        phi,h=base_fields(query,region)
        kernel=op.exterior if region==0 else op.interior
        density=op.bubbles[region].repeat_interleave(3)
        values=[];gradients=[]
        for begin in range(0,len(query)*len(sub),8192):
            flat=torch.arange(begin,min(begin+8192,len(query)*len(sub)),device='cuda');i=flat//len(sub);j=flat%len(sub)
            v,g=kernel.panel_linear_integrals(query[i],sub[j]);values.append(v[:,0]);gradients.append(g[:,0])
        return phi+torch.cat(values).reshape(len(query),-1)@density,h-torch.einsum('psc,s->pc',torch.cat(gradients).reshape(len(query),-1,3),density)
    op.fields=fields
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
        for enriched in (False,True):
            columns=torch.arange(2*nd,device='cuda')
            if not enriched:columns=columns[columns%nd<len(unique)]
            m=matrix[:,columns];solve_start=time.monotonic()
            solved_density=torch.linalg.lstsq(m,rhs).solution
            solution=points.new_zeros(2*nd);solution[columns]=solved_density
            density=solution.reshape(2,nd)
            op.external=p@density[0,:len(unique)];op.internal=p@density[1,:len(unique)]
            op.bubbles=density[:,len(unique):]
            op.residual=float((m@solved_density-rhs).norm()/rhs.norm())
            op.local_residual=None # No full weak bubble blocks were assembled.
            op.density_dofs=len(columns)//2
            op.quadrature=[dict(formulation='physical_trace_least_squares',target_order=order,centroid_hat=enriched,certified=False)]
            solve_seconds=time.monotonic()-solve_start
            parents=raw['parent_vertices'][raw['parent_faces']][raw['patch_ids'][owners]]
            training=physical_trace_residual_summary(matrix,rhs,solution,weight,op.field,points,parents)
            rows=collector.collect(refinements=(1,),spaces=('planar_patch',),degree=2)
            for row in rows:row.update(training_physical_residuals=training,centroid_hat=enriched,local_residual_scope='unavailable_without_weak_bubble_blocks',point_preparation_seconds=preparation_seconds,solve_seconds=solve_seconds,timing_scope='collector_fields_only_preassembled_fit')
            results+=rows
            torch.save(dict(density=solution,external=op.external,internal=op.internal,bubbles=op.bubbles),root/('enriched.pt' if enriched else 'baseline.pt'))
    finally:collector.RegionalDensityDiagnostic=original
    Path(os.environ['MAGNETIC_CONTROL_REPORT']).write_text(json.dumps(results,indent=2,allow_nan=False))
    assert all(not row['physical_qualification'] for row in results)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-state',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--order',type=int,choices=(4,6),default=4)
    args=parser.parse_args()
    collect(args.raw_state,args.output,order=args.order)
