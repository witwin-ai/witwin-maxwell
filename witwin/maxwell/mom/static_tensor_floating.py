"""Regional SPD dielectrics coupled to neutral floating conductive-limit bodies.

Conductor face potentials are induced current-closure traces plus an unknown
constant. This internal interface does not infer finite permittivity for a
conductive limit or expose prescribed interior sources through public Scene.
"""
import torch
from .static_tensor_green import StaticTensorGreen
from .static_tensor_interface import StaticTensorInterfaceOperator
from .fields import classify_queries
from .static_tensor_polarization import StaticTensorPolarization


class StaticTensorFloatingOperator:
    """P0 potential/conormal equations with one neutrality row per conductor."""
    _layers=StaticTensorInterfaceOperator._layers

    def __init__(self,system,exterior,interiors,*,pair_batch=4096,max_matrix_bytes=256*1024**2):
        system.assert_current()
        if type(pair_batch) is not int or pair_batch<1 or type(max_matrix_bytes) is not int or max_matrix_bytes<1:
            raise ValueError('Floating static budgets must be positive integers')
        interiors=tuple(interiors)
        if len(interiors)!=len(system.bodies) or not any(v is None for v in interiors):
            raise ValueError('Floating static requires one coefficient per body and a conductive limit marked None')
        self.system=system;self.pair_batch=pair_batch
        self.exterior=StaticTensorGreen(exterior)
        self.interiors=tuple(None if value is None else StaticTensorGreen(value) for value in interiors)
        for kernel in (self.exterior,*self.interiors):
            if kernel is not None and (kernel.coefficient.device!=system.device or kernel.coefficient.dtype!=system.dtype):
                raise ValueError('Floating static coefficients must match surface precision and device')
        self.current_regions=tuple(i+1 for i,k in enumerate(self.interiors) if k is None)
        self.face_counts=tuple(len(b.mesh.faces) for b in system.bodies)
        self.triangles=torch.cat([b.mesh.triangles for b in system.bodies])
        self.normals=torch.cat([b.mesh.normals for b in system.bodies])
        self.areas=torch.cat([b.mesh.areas for b in system.bodies])
        self.centers=self.triangles.mean(1)
        count=len(self.centers);size=2*count+len(self.current_regions)
        required=size*size*self.centers.element_size()
        if required>max_matrix_bytes:raise ValueError('Floating static retained matrix exceeds its budget')
        extent=(self.triangles.amax((0,1))-self.triangles.amin((0,1))).norm()
        self.flux_scale=self.exterior.coefficient.diagonal().mean()/extent
        if not bool(torch.isfinite(self.flux_scale)) or not bool(self.flux_scale>0):
            raise ValueError('Floating static flux scale is not representable')
        eye=torch.eye(count,device=system.device,dtype=system.dtype)
        single,double=self._layers(self.exterior,self.centers,self.triangles,self.normals,diagonal=True)
        self.matrix=self.centers.new_zeros(size,size)
        self.matrix[:count,:count]=.5*eye-double
        self.matrix[:count,count:2*count]=self.flux_scale*single
        start=0
        for region,(n,kernel) in enumerate(zip(self.face_counts,self.interiors),1):
            end=start+n
            if kernel is None:
                column=2*count+self.current_regions.index(region)
                self.matrix[count+start:count+end,start:end]=eye[start:end,start:end]
                self.matrix[count+start:count+end,column]=-1
                self.matrix[column,count+start:count+end]=self.areas[start:end]/self.areas[start:end].sum()
            else:
                single,double=self._layers(kernel,self.centers[start:end],self.triangles[start:end],self.normals[start:end],diagonal=True)
                self.matrix[count+start:count+end,start:end]=.5*eye[start:end,start:end]+double
                self.matrix[count+start:count+end,count+start:count+end]=-self.flux_scale*single
            start=end
        if not bool(torch.isfinite(self.matrix).all()):raise ValueError('Floating static matrix is not finite')
        self.metadata=dict(formulation='regional_tensor_floating_neutral_p0_v1',physical_qualification=False,
            retained_matrix_bytes=required,current_regions=self.current_regions,
            surface_discretization_error_estimated=False)
        self._binding=self._state()

    def _state(self):
        return (id(self.system),self.pair_batch,self.face_counts,self.current_regions,
            tuple(id(k) for k in (self.exterior,*self.interiors)),tuple((id(t),t._version) for t in
            (self.triangles,self.normals,self.areas,self.centers,self.flux_scale,self.matrix)))

    def assert_current(self):
        self.system.assert_current()
        if self._state()!=self._binding:raise ValueError('Floating static operator changed')
        for kernel in (self.exterior,*self.interiors):
            if kernel is not None:kernel.assert_current()

    def _source_rhs(self,field,boundary_potentials,bindings,polarizations):
        self.assert_current();self.exterior._validate(field,(3,))
        boundary={} if boundary_potentials is None else boundary_potentials
        if type(boundary) is not dict or any(type(r) is not int or r not in self.current_regions for r in boundary):
            raise ValueError('Induced potential traces must belong to conductive-limit regions')
        count=len(self.centers)
        rhs=torch.cat((-self.centers@field,field.new_zeros(count+len(self.current_regions))))
        for region,value in boundary.items():
            start=sum(self.face_counts[:region-1]);n=self.face_counts[region-1]
            self.exterior._validate(value,(n,));rhs[count+start:count+start+n]=value
        bindings,polarizations=tuple(bindings),tuple(polarizations)
        if len(bindings)!=len(polarizations):raise ValueError('Each dielectric volume requires its polarization')
        sources=[None]*len(self.interiors);normal_p=field.new_zeros(count)
        for binding,polarization in zip(bindings,polarizations):
            binding.assert_current();region=binding.volume.region_id
            if binding.surface_system is not self.system or region in self.current_regions or sources[region-1] is not None:
                raise ValueError('Polarization must uniquely bind this interface dielectric region')
            source=StaticTensorPolarization(binding,self.interiors[region-1].coefficient,polarization,pair_batch=self.pair_batch)
            sources[region-1]=source
            start=sum(self.face_counts[:region-1]);end=start+self.face_counts[region-1]
            jump=(source.polarization[source.boundary_cells]*self.normals[start:end]).sum(-1)
            normal_p[start:end]=jump
            rhs[count+start:count+end]=source.fields(self.centers[start:end],interior_sources_only=True)[0]-\
                self.matrix[count+start:count+end,count+start:count+end]@(jump/self.flux_scale)
        if not bool(torch.isfinite(rhs).all()):raise ValueError('Floating static RHS is not representable')
        return rhs,tuple(sources),normal_p

    def solve_uniform(self,field,*,boundary_potentials=None,bindings=(),polarizations=()):
        """Solve neutral floating offsets with optional induced-current face traces."""
        rhs,sources,normal_p=self._source_rhs(field,boundary_potentials,bindings,polarizations)
        solution=torch.linalg.solve(self.matrix,rhs)
        if not bool(torch.isfinite(solution).all()):raise ValueError('Floating static solve is not finite')
        return StaticTensorFloatingResponse(self,field,solution,rhs,tuple(sources),normal_p)

    def response_from_traces(self,field,potential,conormal,offsets,*,boundary_potentials=None,bindings=(),polarizations=()):
        """Own physical traces and their residual without repeating the solve."""
        self.assert_current()
        for value,shape in ((potential,(len(self.centers),)),(conormal,(len(self.centers),)),
                            (offsets,(len(self.current_regions),))):self.exterior._validate(value,shape)
        rhs,sources,normal_p=self._source_rhs(field,boundary_potentials,bindings,polarizations)
        solution=torch.cat((potential,conormal/self.flux_scale,offsets))
        if not bool(torch.isfinite(solution).all()):raise ValueError('Floating scaled traces are not representable')
        response=StaticTensorFloatingResponse(self,field,solution,rhs,sources,normal_p)
        response.conormal=conormal.clone();response._binding=response._state()
        return response

    def prepare_cell_response(self,bindings,*,max_response_bytes=256*1024**2):
        """Retain linear P/induced-trace maps; budget stored bytes, not workspace."""
        self.assert_current();bindings=tuple(bindings)
        if type(max_response_bytes) is not int or max_response_bytes<1:
            raise ValueError('Floating response budget must be positive')
        regions=tuple(b.volume.region_id for b in bindings)
        if len(set(regions))!=len(regions):raise ValueError('Floating response volumes must be unique')
        for b in bindings:
            b.assert_current()
            if b.surface_system is not self.system or b.volume.region_id in self.current_regions:
                raise ValueError('Floating response volumes must bind dielectric regions')
        counts=tuple(len(b.volume.cells) for b in bindings);cells=sum(counts);faces=len(self.centers)
        trace_counts=tuple(self.face_counts[r-1] for r in self.current_regions)
        columns=3*cells+sum(trace_counts);rows=3*cells+2*faces+len(self.current_regions)
        required=rows*columns*self.matrix.element_size()
        if required>max_response_bytes:raise ValueError('Floating retained response exceeds its budget')
        points=tuple(b.volume.vertices[b.volume.cells].mean(1) for b in bindings)
        def evaluate(vector):
            polarization=vector[:3*cells].reshape(cells,3).split(counts)
            boundary=dict(zip(self.current_regions,vector[3*cells:].split(trace_counts)))
            response=self.solve_uniform(self.matrix.new_zeros(3),boundary_potentials=boundary,
                                        bindings=bindings,polarizations=polarization)
            fields=torch.cat([response.fields(p,r)[1] for p,r in zip(points,regions)]).flatten() if bindings else vector.new_empty(0)
            return torch.cat((fields,response.potential,response.conormal,response.offsets))
        jacobian=torch.autograd.functional.jacobian(evaluate,self.matrix.new_zeros(columns),vectorize=False)
        if not bool(torch.isfinite(jacobian).all()):raise ValueError('Floating response is not finite')
        return dict(electric_response=jacobian[:3*cells],surface_response=jacobian[3*cells:3*cells+faces],
            conormal_response=jacobian[3*cells+faces:3*cells+2*faces],offset_response=jacobian[3*cells+2*faces:],
            receipt=dict(retained_response_bytes=required,physical_qualification=False,
                         material_geometry_gradients_retained=False,quadrature_convergence_certified=False))


class StaticTensorFloatingResponse:
    def __init__(self,operator,field,solution,rhs,sources,normal_p):
        self.operator=operator;self.field=field.clone();count=len(operator.centers)
        self.sources=sources;self.normal_p=normal_p.clone()
        self.potential=solution[:count].clone()
        self.conormal=(operator.flux_scale*solution[count:2*count]).clone()
        self.offsets=solution[2*count:].clone()
        if not bool(torch.isfinite(self.conormal).all()):
            raise ValueError('Floating physical conormal is not representable')
        residual=operator.matrix@solution-rhs
        def stable_norm(value):
            scale=value.abs().max()
            return scale*(value/scale).norm() if bool(scale>0) else scale
        norm,error=stable_norm(rhs),stable_norm(residual)
        relative=error/norm if bool(norm>0) else error
        if not bool(torch.isfinite(norm) & torch.isfinite(error) & torch.isfinite(relative)):
            raise ValueError('Floating residual diagnostics are not representable')
        self.receipt=dict(operator.metadata,algebraic_residual_norm=float(error.detach()),
                          relative_algebraic_residual=float(relative.detach()))
        self._binding=self._state()

    def _state(self):
        return (id(self.operator),tuple(map(id,self.sources)),tuple((id(t),t._version) for t in
            (self.field,self.potential,self.conormal,self.offsets,self.normal_p)))

    def assert_current(self):
        self.operator.assert_current()
        if self._state()!=self._binding:raise ValueError('Floating static response changed')
        for source in self.sources:
            if source is not None:source.assert_current()

    def fields(self,points,region_id):
        """Decode exterior or dielectric fields; conductor volumes need current closure."""
        self.assert_current();op=self.operator;region=op.system.region(region_id)
        op.exterior._validate(points,(None,3))
        if region in op.current_regions:
            raise NotImplementedError('Conductive interior fields require their actual volume-current closure')
        if len(points)==0:return points.sum(-1),points.clone()
        for body in op.system.bodies:
            inside,_=classify_queries(body.mesh,points)
            if bool((inside!=(body.region_id==region)).any()):raise ValueError('Floating query lies outside its declared region')
        with torch.enable_grad():
            observers=points if points.requires_grad else points.detach().requires_grad_()
            if region==0:
                single,double=op._layers(op.exterior,observers,op.triangles,op.normals)
                value=-observers@self.field+double@self.potential-single@self.conormal
            else:
                start=sum(op.face_counts[:region-1]);end=start+op.face_counts[region-1]
                single,double=op._layers(op.interiors[region-1],observers,op.triangles[start:end],op.normals[start:end])
                value=single@(self.conormal[start:end]+self.normal_p[start:end])-double@self.potential[start:end]
                if self.sources[region-1] is not None:
                    value=value+self.sources[region-1].fields(observers,interior_sources_only=True)[0]
            electric=-torch.autograd.grad(value.sum(),observers,create_graph=True)[0]
        if not bool(torch.isfinite(value).all() & torch.isfinite(electric).all()):
            raise ValueError('Floating recovered potential/field is not representable')
        return value,electric
