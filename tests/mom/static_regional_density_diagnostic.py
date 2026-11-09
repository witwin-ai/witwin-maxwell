"""Test-only real-SPD regional density experiment; no production qualification.

One closed body, uniform background, separate exterior/interior single layers.
Raw rows divide each target P1/P2 weak equation by its face area. The conormal
rows additionally divide by the exterior coefficient/geometry scale. Touching
pairs use joint order 8; disjoint pairs use estimated adaptive quadrature.
Retained dense matrices, temporary allocations and autodiff are unqualified.
"""
import torch
from witwin.maxwell.mom.static_tensor_green import StaticTensorGreen

class RegionalDensityDiagnostic:
    def __init__(self,system,exterior,interiors,*,pair_batch=8192,degree=1):
        assert len(system.bodies)==1
        if type(degree) is not int or degree not in (1,2):
            raise ValueError('Diagnostic density degree must be 1 or 2')
        self.degree=degree;self.nb=3 if degree==1 else 6
        self.mesh=system.bodies[0].mesh;self.triangles=self.mesh.triangles
        self.exterior=StaticTensorGreen(exterior);self.interior=StaticTensorGreen(interiors[0])
        self.nodes=self.mesh.faces
        if degree==2:
            edges=self.mesh.faces[:,[[0,1],[1,2],[2,0]]].sort(-1).values
            _,inverse=torch.unique(edges.reshape(-1,2),dim=0,return_inverse=True)
            self.nodes=torch.cat((self.mesh.faces,inverse.reshape(-1,3)+len(self.mesh.vertices)),1)
        self.pair_batch=pair_batch;self.nf=len(self.mesh.faces)
        self.scale=exterior.diagonal().mean()/(self.triangles.amax((0,1))-self.triangles.amin((0,1))).norm()
        rows=torch.arange(self.nf,device=exterior.device).repeat_interleave(self.nf)
        cols=torch.arange(self.nf,device=exterior.device).repeat(self.nf)
        shared=(self.mesh.faces[rows,:,None]==self.mesh.faces[cols,None,:]).sum((1,2))
        self.quadrature=[]
        def assembly(kernel):
            value=self.triangles.new_empty(self.nf,self.nf,self.nb,self.nb)
            conormal=torch.empty_like(value)
            for count in (3,2,1,0):
                select=shared==count;i=rows[select];j=cols[select]
                if count:
                    touching=kernel.touching_linear_weak_layers if degree==1 else kernel.touching_quadratic_weak_layers
                    v,k=touching(self.triangles[i],self.triangles[j],
                        self.mesh.faces[i],self.mesh.faces[j],3-count,order=8)
                else:
                    disjoint=kernel.adaptive_disjoint_linear_weak_layers if degree==1 else kernel.adaptive_disjoint_quadratic_weak_layers
                    v,k,receipt=disjoint(
                        self.triangles[i],self.triangles[j],rtol=1e-6)
                    self.quadrature.append(dict(orders={int(order):int((receipt['orders']==order).sum()) for order in receipt['orders'].unique()},
                        maximum_change=float(receipt['relative_change'].max()),certified=False))
                value[i,j]=v/self.mesh.areas[i,None,None]
                conormal[i,j]=k/self.mesh.areas[i,None,None]
            return value.permute(0,2,1,3).reshape(self.nb*self.nf,self.nb*self.nf),conormal.permute(0,2,1,3).reshape(self.nb*self.nf,self.nb*self.nf)
        v0,k0=assembly(self.exterior);vi,ki=assembly(self.interior)
        mass=density_reference_mass(exterior,degree)
        self.mean_basis=exterior.new_tensor([1/3]*3 if degree==1 else [0,0,0,1/3,1/3,1/3])
        jump=torch.kron(torch.eye(self.nf,device=exterior.device,dtype=exterior.dtype),mass)
        self.matrix=torch.cat((torch.cat((v0,-vi),1),torch.cat(((k0-.5*jump)/self.scale,-(ki+.5*jump)/self.scale),1)))
        assert torch.isfinite(self.matrix).all()
    def solve_uniform(self,field,*,patch_ids=None):
        self.field=field
        contrast=self.exterior.coefficient-self.interior.coefficient
        rhs=torch.cat((field.new_zeros(self.nb*self.nf),((self.mesh.normals@(contrast@field))[:,None]*self.mean_basis).reshape(-1)/self.scale))
        if patch_ids is None:
            matrix=self.matrix;reduced_rhs=rhs;lift=None
        else:
            matrix,lift,weights=project_density_system(
                self.matrix,self.mesh.areas,self.nodes,patch_ids)
            reduced_rhs=lift.T@(weights*rhs)
        solution=torch.linalg.solve(matrix,reduced_rhs)
        self.residual=float((matrix@solution-reduced_rhs).norm()/reduced_rhs.norm().clamp_min(torch.finfo(rhs.dtype).tiny))
        lifted=solution if lift is None else lift@solution
        self.external=lifted[:self.nb*self.nf];self.internal=lifted[self.nb*self.nf:]
        self.local_residual=float((self.matrix@lifted-rhs).norm()/rhs.norm().clamp_min(torch.finfo(rhs.dtype).tiny))
        self.density_dofs=len(solution)//2
        print(dict(residual=self.residual,quadrature=self.quadrature),flush=True)
        return self
    def fields(self,points,region):
        assert region in (0,1)
        kernel=self.exterior if region==0 else self.interior
        density=self.external if region==0 else self.internal
        values=[];gradients=[];rows=len(points);cols=self.nf
        for start in range(0,rows*cols,self.pair_batch):
            flat=torch.arange(start,min(start+self.pair_batch,rows*cols),device=points.device)
            i=flat//cols;j=flat%cols
            layers=kernel.panel_linear_integrals if self.degree==1 else kernel.panel_quadratic_integrals
            value,gradient=layers(points[i],self.triangles[j])
            values.append(value);gradients.append(gradient)
        value=torch.cat(values).reshape(rows,self.nb*cols);gradient=torch.cat(gradients).reshape(rows,self.nb*cols,3)
        return -points@self.field+value@density,self.field-torch.einsum('pfc,f->pc',gradient,density)


def project_density_system(matrix, areas, faces, patch_ids):
    """Share P1/P2 nodes within explicitly declared patches, in both regions.

    Patch IDs are declarations, not inferred geometric regions. The caller
    owns patch membership and conformity; this helper does not certify planar
    geometry. Distinct patches retain distinct coefficients at shared vertices.
    Restore target areas BEFORE projection: P.T W M P and P.T W rhs.
    The matrix already includes its fixed conormal-row scaling.
    """
    nf=len(faces);nb=faces.shape[1] if faces.ndim==2 else 0
    if (nb not in (3,6) or patch_ids.shape!=(nf,) or areas.shape!=(nf,)
            or matrix.shape!=(2*nb*nf,2*nb*nf)):
        raise ValueError('Density projection dimensions do not match')
    if (faces.dtype!=torch.int64 or patch_ids.dtype!=torch.int64
            or any(t.device!=matrix.device for t in (faces,patch_ids,areas))
            or areas.dtype!=matrix.dtype):
        raise ValueError('Density projection requires matching geometry and int64 connectivity')
    if (nf==0 or bool((faces<0).any()) or bool((patch_ids<0).any())
            or not bool(torch.isfinite(areas).all()) or not bool((areas>0).all())):
        raise ValueError('Density projection requires nonnegative IDs and positive finite areas')
    keys=torch.stack((patch_ids[:,None].expand(-1,nb).reshape(-1),faces.reshape(-1)),1)
    unique,inverse=torch.unique(keys,dim=0,return_inverse=True)
    projection=matrix.new_zeros(nb*nf,len(unique))
    projection[torch.arange(nb*nf,device=matrix.device),inverse]=1
    lift=torch.block_diag(projection,projection)
    weights=areas.repeat_interleave(nb).repeat(2)
    return lift.T@(weights[:,None]*matrix)@lift,lift,weights


def density_reference_mass(sample,degree):
    """Exact integral N_i N_j divided by physical triangle area."""
    if degree==1:
        return (sample.new_ones(3,3)+torch.eye(3,device=sample.device,dtype=sample.dtype))/12
    if degree!=2:raise ValueError('Density reference mass requires degree 1 or 2')
    return sample.new_tensor([
        [6,-1,-1,0,-4,0],[-1,6,-1,0,0,-4],[-1,-1,6,-4,0,0],
        [0,0,-4,32,16,16],[-4,0,0,16,32,16],[0,-4,0,16,16,32]])/180
