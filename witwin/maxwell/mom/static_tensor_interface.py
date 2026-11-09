"""Internal real-SPD regional static potential/conormal transmission system.

Face-center collocation uses P0 or continuous P1 potential and P0 conormal flux.
P1 uses an overdetermined unweighted least-squares system; its nonzero residual
is a discretization diagnostic, not a physical accuracy certificate.
Each region owns its Green kernel. This is a discretization candidate, not an
adaptive interface certificate or a public mixed harmonic-balance backend.
"""
import torch
from .static_tensor_green import StaticTensorGreen
from .fields import classify_queries
from .static_tensor_polarization import StaticTensorPolarization


class StaticTensorInterfaceOperator:
    """Continuity of potential and displacement, with normals out of bodies.

    Interior rows are (1/2+K_i)u-S_i q=0. Exterior rows are
    (1/2-K_0)u+S_0 q=phi_inc. K integrates the source conormal
    derivative of G; q is exterior conormal. Induced P adds P.n to the
    interior conormal and its internal distributional charge to the RHS.
    """
    def __init__(self,system,exterior,interiors,*,pair_batch=4096,max_matrix_bytes=256*1024**2,potential_basis="p0"):
        system.assert_current()
        if type(pair_batch) is not int or pair_batch<1 or type(max_matrix_bytes) is not int or max_matrix_bytes<1:
            raise ValueError('Static interface budgets must be positive integers')
        if len(interiors)!=len(system.bodies):
            raise ValueError('Static interfaces need one coefficient per body')
        if potential_basis not in ('p0','p1'):
            raise ValueError('Static potential basis must be p0 or p1')
        self.potential_basis=potential_basis
        self.system=system;self.pair_batch=pair_batch
        self.exterior=StaticTensorGreen(exterior)
        self.interiors=tuple(StaticTensorGreen(value) for value in interiors)
        if any(k.coefficient.device!=system.device or k.coefficient.dtype!=system.dtype
               for k in (self.exterior,*self.interiors)):
            raise ValueError('Static coefficients must match surface geometry device and precision')
        self.triangles=torch.cat([b.mesh.triangles for b in system.bodies])
        self.normals=torch.cat([b.mesh.normals for b in system.bodies])
        self.centers=self.triangles.mean(1)
        count=len(self.triangles)
        self.potential_counts=tuple(len(b.mesh.vertices) if potential_basis=='p1' else len(b.mesh.faces)
                                    for b in system.bodies)
        self.potential_count=sum(self.potential_counts)
        indices=[];offset=0
        for body,size in zip(system.bodies,self.potential_counts):
            indices.append(body.mesh.faces+offset if potential_basis=='p1' else
                           torch.arange(size,device=system.device)[:,None]+offset)
            offset+=size
        self.potential_indices=torch.cat(indices)
        required=2*count*(self.potential_count+count)*self.triangles.element_size()
        if required>max_matrix_bytes:
            raise ValueError('Static interface retained matrix exceeds its explicit budget')
        self.face_counts=tuple(len(b.mesh.faces) for b in system.bodies)
        extent=(self.triangles.amax((0,1))-self.triangles.amin((0,1))).norm()
        self.flux_scale=self.exterior.coefficient.diagonal().mean()/extent
        if not bool(torch.isfinite(self.flux_scale)) or not bool(self.flux_scale>0):
            raise ValueError('Static interface flux scaling is not representable')
        trace=self.triangles.new_zeros(count,self.potential_count)
        trace.scatter_add_(1,self.potential_indices,
                           torch.full_like(self.potential_indices,1/self.potential_indices.shape[1],dtype=system.dtype))
        single,double=self._potential_layers(self.exterior,self.centers,0,count,diagonal=True)
        self.matrix=self.triangles.new_zeros(2*count,self.potential_count+count)
        self.matrix[:count,:self.potential_count]=.5*trace-double
        self.matrix[:count,self.potential_count:]=self.flux_scale*single
        start=0
        for size,kernel in zip(self.face_counts,self.interiors):
            end=start+size
            single,double=self._potential_layers(kernel,self.centers[start:end],start,end,diagonal=True)
            self.matrix[count+start:count+end,:self.potential_count]=.5*trace[start:end]+double
            self.matrix[count+start:count+end,self.potential_count+start:self.potential_count+end]=-self.flux_scale*single
            start=end
        if not bool(torch.isfinite(self.matrix).all()):
            raise ValueError('Static interface matrix is not representable')
        self.metadata=dict(formulation=f'regional_static_potential_conormal_{potential_basis}_collocation_v1',
            physical_qualification=False,retained_matrix_bytes=required,pair_batch=pair_batch,
            surface_discretization_error_estimated=False,units='caller_supplied_constitutive_coefficient')
        if potential_basis=='p1':
            self.metadata.update(potential_basis='p1',conormal_basis='p0',
                solve_method='unweighted_face_center_least_squares')
        self._binding=self._state()

    def _state(self):
        return (id(self.system),self.pair_batch,self.face_counts,self.potential_basis,self.potential_counts,self.potential_count,tuple(map(id,(self.exterior,*self.interiors))),
            tuple((id(t),t._version) for t in (self.triangles,self.normals,self.centers,self.flux_scale,self.matrix,self.potential_indices)))

    def assert_current(self):
        self.system.assert_current()
        if self._state()!=self._binding:
            raise ValueError('Static interface operator changed')
        for kernel in (self.exterior,*self.interiors):kernel.assert_current()

    def _layers(self,kernel,points,triangles,normals,*,diagonal=False):
        rows,cols=len(points),len(triangles)
        if diagonal and (rows!=cols or not torch.equal(points,triangles.mean(1))):
            raise ValueError('Diagonal static layers require the declared face-center collocation')
        single=[];double=[]
        for start in range(0,rows*cols,self.pair_batch):
            flat=torch.arange(start,min(start+self.pair_batch,rows*cols),device=points.device)
            i=flat//cols;j=flat%cols
            if diagonal:
                # Form mathematical target centroids after removing the source
                # origin. Rounding a large absolute centroid first can move a
                # self query off its plane by more than local geometric ulps.
                origin=triangles[j,0]
                observers=(triangles[i]-origin[:,None]).mean(1)
                panels=triangles[j]-origin[:,None]
            else:
                observers,panels=points[i],triangles[j]
            value,gradient=kernel.panel_integrals(observers,panels,principal_value=(i==j) if diagonal else None)
            single.append(value)
            double.append(-(gradient@kernel.coefficient.T*normals[j]).sum(-1))
        if not single:return points.new_empty(rows,cols),points.new_empty(rows,cols)
        return torch.cat(single).reshape(rows,cols),torch.cat(double).reshape(rows,cols)

    def _potential_layers(self,kernel,points,start,end,*,diagonal=False):
        triangles=self.triangles[start:end];normals=self.normals[start:end]
        if self.potential_basis=='p0':
            single,double=self._layers(kernel,points,triangles,normals,diagonal=diagonal)
            return single,torch.nn.functional.pad(double,(start,self.potential_count-end))
        rows,cols=len(points),len(triangles)
        if diagonal and (rows!=cols or not torch.equal(points,triangles.mean(1))):
            raise ValueError('Diagonal static layers require the declared face-center collocation')
        single=[];moments=[]
        for first in range(0,rows*cols,self.pair_batch):
            flat=torch.arange(first,min(first+self.pair_batch,rows*cols),device=points.device)
            i=flat//cols;j=flat%cols
            if diagonal:
                origin=triangles[j,0]
                observers=(triangles[i]-origin[:,None]).mean(1)
                panels=triangles[j]-origin[:,None]
            else:observers,panels=points[i],triangles[j]
            pv=(i==j) if diagonal else None
            single.append(kernel.panel_integrals(observers,panels,principal_value=pv)[0])
            moments.append(kernel.panel_linear_double_layer(observers,panels,principal_value=pv))
        if not single:return points.new_empty(rows,cols),points.new_empty(rows,self.potential_count)
        weights=torch.cat(moments).reshape(rows,cols*3)
        indices=self.potential_indices[start:end].reshape(1,-1).expand(rows,-1)
        double=weights.new_zeros(rows,self.potential_count).scatter_add(1,indices,weights)
        return torch.cat(single).reshape(rows,cols),double

    def _source_rhs(self,field,offset,bindings,polarizations):
        """Build the same induced-source RHS for solves and trace recovery."""
        self.assert_current()
        self.exterior._validate(field,(3,))
        if type(offset) not in (int,float) or not torch.isfinite(field.new_tensor(offset)):
            raise ValueError('Static incident potential offset must be finite')
        incident=offset-self.centers@field
        bindings,polarizations=tuple(bindings),tuple(polarizations)
        if len(bindings)!=len(polarizations):
            raise ValueError('Each static induced volume requires a polarization array')
        sources=[None]*len(self.system.bodies)
        for binding,polarization in zip(bindings,polarizations):
            if getattr(binding,'surface_system',None) is not self.system:
                raise ValueError('Static induced volumes must bind this exact surface system')
            binding.assert_current()
            index=binding.volume.region_id-1
            if sources[index] is not None:
                raise ValueError('Duplicate static induced volume region')
            sources[index]=StaticTensorPolarization(binding,self.interiors[index].coefficient,
                                                    polarization,pair_batch=self.pair_batch)
        normal_p_values=[];interior_rhs=[]
        start=0
        for size,source in zip(self.face_counts,sources):
            end=start+size
            normal_p=incident.new_zeros(size);volume_potential=torch.zeros_like(normal_p)
            if source is not None:
                normal_p=(source.polarization[source.boundary_cells]*self.normals[start:end]).sum(-1)
                volume_potential=source.fields(self.centers[start:end],interior_sources_only=True)[0]
            normal_p_values.append(normal_p)
            count=len(incident)
            # Interior matrix stores -flux_scale*S_i. Volume charge excludes
            # boundary P.n, which enters through total interior q=q_exterior+P.n.
            interior_rhs.append(volume_potential
                -self.matrix[count+start:count+end,self.potential_count+start:self.potential_count+end]@(normal_p/self.flux_scale))
            start=end
        rhs=torch.cat((incident,torch.cat(interior_rhs)))
        if not bool(torch.isfinite(rhs).all()):
            raise ValueError("Static incident potential is not representable")
        return rhs,tuple(sources),torch.cat(normal_p_values)

    def solve_uniform(self,field,*,offset=0.,bindings=(),polarizations=()):
        """Solve for phi_inc=offset-field.x; no prescribed interior sources."""
        rhs,sources,normal_p=self._source_rhs(field,offset,bindings,polarizations)
        solution=(torch.linalg.solve(self.matrix,rhs) if self.potential_basis=='p0' else
                  torch.linalg.lstsq(self.matrix,rhs,driver='gels').solution)
        residual=self.matrix@solution-rhs
        if not bool(torch.isfinite(solution).all() & torch.isfinite(residual).all()):
            raise ValueError('Static interface solve is not finite')
        return StaticTensorInterfaceResponse(self,field,offset,solution,rhs,residual,
            sources,normal_p)


    def response_from_traces(self,field,potential,conormal,*,offset=0.,bindings=(),polarizations=()):
        """Recover an owned response from physical traces without solving.

        The source RHS and residual are recomputed for diagnostics. The supplied
        physical conormal is retained exactly, not rounded through flux scaling.
        """
        self.assert_current()
        self.exterior._validate(potential,(self.potential_count,))
        self.exterior._validate(conormal,(len(self.centers),))
        rhs,sources,normal_p=self._source_rhs(field,offset,bindings,polarizations)
        solution=torch.cat((potential,conormal/self.flux_scale))
        response=StaticTensorInterfaceResponse(self,field,offset,solution,rhs,
            self.matrix@solution-rhs,sources,normal_p)
        response.conormal=conormal.clone()
        response._binding=response._state()
        return response


class StaticTensorInterfaceResponse:
    """Owned uniform-drive solution with the operator's potential basis and P0 flux."""
    def __init__(self,operator,field,offset,solution,rhs,residual,sources,normal_p):
        self.operator=operator;self.field=field.clone();self.offset=offset
        self.sources=sources
        self.normal_p=normal_p.clone()
        count=operator.potential_count
        self.potential=solution[:count].clone()
        self.conormal=operator.flux_scale*solution[count:]
        if not bool(torch.isfinite(self.potential).all() & torch.isfinite(self.conormal).all()
                    & torch.isfinite(self.normal_p).all()):
            raise ValueError('Static physical potential/conormal is not representable')
        def stable_norm(value):
            scale=value.abs().max()
            return scale*(value/scale).norm() if bool(scale>0) else scale
        norm=stable_norm(rhs);error=stable_norm(residual)
        relative=error/norm if bool(norm>0) else error
        if not bool(torch.isfinite(norm) & torch.isfinite(error) & torch.isfinite(relative)):
            raise ValueError('Static residual diagnostics are not representable')
        self.receipt=dict(operator.metadata,algebraic_residual_norm=float(error.detach()),
            relative_algebraic_residual=float(relative.detach()),
            induced_regions=tuple(i+1 for i,source in enumerate(sources) if source is not None))
        self._binding=self._state()

    def _state(self):
        return (id(self.operator),self.offset,tuple(map(id,self.sources)),tuple((id(t),t._version) for t in
            (self.field,self.potential,self.conormal,self.normal_p)))

    def assert_current(self):
        self.operator.assert_current()
        if self._state()!=self._binding:raise ValueError('Static interface response changed')
        for source in self.sources:
            if source is not None:source.assert_current()

    def fields(self,points,region_id):
        """Recover potential and minus its gradient at classified off-face points."""
        self.assert_current();op=self.operator
        op.exterior._validate(points,(None,3))
        region=op.system.region(region_id)
        if len(points)==0:
            return points.sum(-1),points.clone()
        for body in op.system.bodies:
            inside,_=classify_queries(body.mesh,points)
            if bool((inside!=(body.region_id==region)).any()):
                raise ValueError('Static query does not lie in its declared region')
        with torch.enable_grad():
            observers=points if points.requires_grad else points.detach().requires_grad_()
            if region==0:
                single,double=op._potential_layers(op.exterior,observers,0,len(op.centers))
                value=self.offset-observers@self.field+double@self.potential-single@self.conormal
            else:
                start=sum(op.face_counts[:region-1]);end=start+op.face_counts[region-1]
                single,double=op._potential_layers(op.interiors[region-1],observers,start,end)
                value=single@(self.conormal[start:end]+self.normal_p[start:end])-double@self.potential
                source=self.sources[region-1]
                if source is not None:value=value+source.fields(observers,interior_sources_only=True)[0]
            gradient,=torch.autograd.grad(value.sum(),observers,create_graph=True)
        if not bool(torch.isfinite(value).all() & torch.isfinite(gradient).all()):
            raise ValueError("Static recovered potential/field is not representable")
        return value,-gradient
