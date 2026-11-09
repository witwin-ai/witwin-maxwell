"""Regional real-SPD particular fields of internal cell polarization.

For D=-A grad(phi)+P, div(D)=0 gives div(A grad(phi))=div(P).
Cellwise constant P contributes oriented face charge n.P, including internal
jumps. Analytic single layers retain the interior self field without a point
quadrature singularity or duplicate linear material contrast.
"""
import torch
from .static_tensor_green import StaticTensorGreen
from .volume_mesh import BoundVolumeRegion
from .fields import _pair_triangle_distance


class StaticTensorPolarization:
    """Owned particular potential/field for one exactly bound source region.

    This internal source is an induced polarization, not a public prescribed
    interior excitation. Coefficient units and real DC samples are explicit.
    Queries may be in any region; transmission corrections are separate.
    """
    def __init__(self,binding,coefficient,polarization,*,pair_batch=4096):
        if not isinstance(binding,BoundVolumeRegion):
            raise TypeError('Static polarization requires an exactly bound volume')
        binding.assert_current()
        if type(pair_batch) is not int or pair_batch<1:
            raise ValueError('Static polarization pair budget must be a positive integer')
        self.binding=binding;self.pair_batch=pair_batch
        self.kernel=StaticTensorGreen(coefficient)
        volume=binding.volume
        if coefficient.device!=volume.vertices.device or coefficient.dtype!=volume.vertices.dtype:
            raise ValueError('Static polarization coefficient must match volume geometry')
        self.kernel._validate(polarization,(len(volume.cells),3))
        self.polarization=polarization.clone()
        local=volume.cells.new_tensor([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
        self.triangles=volume.vertices[volume.cells[:,local]].reshape(-1,3,3)
        cross=torch.linalg.cross(self.triangles[:,1]-self.triangles[:,0],
                                 self.triangles[:,2]-self.triangles[:,0])
        self.normals=cross/cross.norm(dim=-1,keepdim=True)
        self.density=(self.normals*self.polarization.repeat_interleave(4,dim=0)).sum(-1)
        if not bool(torch.isfinite(self.density).all()):
            raise ValueError('Static polarization face density is not representable')
        # Connectivity matching only; all geometric and field arithmetic stays on CUDA.
        incidence={}
        local_faces=((1,2,3),(0,3,2),(0,1,3),(0,2,1))
        for cell_id,cell in enumerate(volume.cells.cpu().tolist()):
            for face in local_faces:
                incidence.setdefault(tuple(sorted(cell[i] for i in face)),[]).append(cell_id)
        boundary_to_volume=volume.boundary_vertex_ids.cpu().tolist()
        boundary_to_surface=binding.boundary_to_surface_vertices.cpu().tolist()
        surface_to_volume=dict(zip(boundary_to_surface,boundary_to_volume))
        body=binding.surface_system.bodies[volume.region_id-1]
        owners=[]
        for face in body.mesh.faces.cpu().tolist():
            entries=incidence[tuple(sorted(surface_to_volume[i] for i in face))]
            if len(entries)!=1:
                raise ValueError('Bound surface face must own exactly one source cell')
            owners.append(entries[0])
        self.boundary_cells=volume.cells.new_tensor(owners)
        internal=[]
        for cell_id,cell in enumerate(volume.cells.cpu().tolist()):
            for face_id,face in enumerate(local_faces):
                if len(incidence[tuple(sorted(cell[i] for i in face))])==2:
                    internal.append(4*cell_id+face_id)
        self.internal_faces=volume.cells.new_tensor(internal)
        self.receipt=dict(formulation='real_spd_cell_polarization_face_layers_v1',
            self_term='analytic_oriented_cell_face_layers',linear_contrast_added=False,
            physical_qualification=False,pair_batch=pair_batch)
        self._binding=self._state()

    def _state(self):
        return (id(self.binding),id(self.kernel),self.pair_batch,tuple((id(t),t._version)
                for t in (self.polarization,self.triangles,self.normals,self.density,self.boundary_cells,self.internal_faces)))

    def assert_current(self):
        self.binding.assert_current();self.kernel.assert_current()
        if self._state()!=self._binding:
            raise ValueError('Static polarization state changed')

    def fields(self,points,*,principal_value=False,interior_sources_only=False):
        """Return potential and E=-grad(phi), with explicit face-interior PV.

        interior_sources_only excludes the body's outer boundary layers, leaving
        the distributional volume charge -div(P) within the open body.
        Cell face edges/vertices reject even for cancelling densities. The PV
        field on a face is the arithmetic mean of the two one-sided fields;
        it is not a chosen material-side trace or an interface boundary solve.
        """
        self.assert_current();self.kernel._validate(points,(None,3))
        if type(principal_value) is not bool or type(interior_sources_only) is not bool:
            raise ValueError('Static polarization principal_value must be boolean')
        if len(points)==0:return points.sum(-1),points.clone()
        panels_owned=self.triangles[self.internal_faces] if interior_sources_only else self.triangles
        density=self.density[self.internal_faces] if interior_sources_only else self.density
        count=len(panels_owned)
        if count==0:return (points*0).sum(-1),points*0
        potential=points.new_zeros(len(points));gradient=torch.zeros_like(points)
        for start in range(0,len(points)*count,self.pair_batch):
            flat=torch.arange(start,min(start+self.pair_batch,len(points)*count),device=points.device)
            i=flat//count;j=flat%count
            observers=points[i];panels=panels_owned[j]
            mask=None
            if principal_value:
                lengths=torch.stack(((panels[:,1]-panels[:,0]).norm(dim=-1),
                    (panels[:,2]-panels[:,0]).norm(dim=-1),(panels[:,2]-panels[:,1]).norm(dim=-1)),dim=-1)
                tolerance=128*torch.finfo(points.dtype).eps*lengths.amax(-1)
                mask=_pair_triangle_distance(observers,panels)<=tolerance
            scalar,derivative=self.kernel.panel_integrals(observers,panels,principal_value=mask)
            potential=potential.index_add(0,i,scalar*density[j])
            gradient=gradient.index_add(0,i,derivative*density[j,None])
        if not bool(torch.isfinite(potential).all() & torch.isfinite(gradient).all()):
            raise ValueError('Static polarization field is not representable')
        return potential,-gradient

    def boundary_trace(self):
        """Particular potential, interior conormal and P.n at body face centers.

        The interior conormal is its PV value plus P.n/2. This jump follows
        from the outward face density and holds for arbitrary SPD A.
        """
        self.assert_current()
        body=self.binding.surface_system.bodies[self.binding.volume.region_id-1]
        normals=body.mesh.normals
        potential,mean_field=self.fields(body.mesh.triangles.mean(1),principal_value=True)
        normal_p=(self.polarization[self.boundary_cells]*normals).sum(-1)
        conormal=-(mean_field@self.kernel.coefficient.T*normals).sum(-1)+normal_p/2
        if not bool(torch.isfinite(conormal).all() & torch.isfinite(normal_p).all()):
            raise ValueError('Static polarization boundary trace is not representable')
        return potential,conormal,normal_p
