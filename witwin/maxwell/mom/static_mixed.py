"""Mixed dielectric displacement and floating-current DC interface equations."""
import torch
from .static_floating import FloatingPotentialOperator
from .static_coupling import StaticVolumeFieldResponse
from .quadrature import triangle_rule


class MixedStaticOperator:
    """Reusable interface matrix for dielectric and current-controlled bodies.

    None interior epsilon marks a current body. Its interior fields belong to
    the current closure; exterior/dielectric fields use the shared layer.
    """
    def __init__(self,laplace,epsilon_exterior,epsilon_interiors):
        if not isinstance(laplace,FloatingPotentialOperator):
            raise TypeError('Mixed DC requires a shared Laplace panel operator')
        laplace._assert_current()
        self.laplace=laplace
        interiors=tuple(epsilon_interiors)
        if len(interiors)!=len(laplace.body_faces):
            raise ValueError('Each mixed DC body requires a dielectric or current channel')
        self.current_regions=tuple(i+1 for i,value in enumerate(interiors) if value is None)
        count=len(laplace.triangles)
        samples=tuple(epsilon_exterior if value is None else value for value in interiors)
        template=StaticVolumeFieldResponse(laplace.system,epsilon_exterior,samples,[],[],
            laplace.triangles.new_zeros(count),testing_order=laplace.testing_order)
        self.exterior,self.interiors=template.exterior,template.interiors
        selected=torch.tensor([region-1 for region in self.current_regions],
                              device=laplace.triangles.device,dtype=torch.int64)
        self.indicator=laplace.indicator[:,selected]
        self.current_faces=self.indicator.sum(-1).bool()
        self.contrast=2*(template.face_epsilon-self.exterior)/(template.face_epsilon+self.exterior)
        normal_derivative=torch.einsum('qti,qi->qt',laplace.mean_gradient,laplace.normals)
        dielectric=torch.eye(count,device=laplace.triangles.device,dtype=laplace.triangles.dtype)+self.contrast[:,None]*normal_derivative
        surface=torch.where(self.current_faces[:,None],laplace.single_layer,dielectric)
        top=torch.cat((surface,-self.indicator),1)
        self.charge_test=laplace.charge_test[selected]
        self.lengths=laplace.lengths[selected]
        bottom=torch.cat((self.charge_test/self.lengths[:,None],
                          laplace.triangles.new_zeros(len(selected),len(selected))),1)
        self.matrix=torch.cat((top,bottom))
        self.lu,self.pivots=torch.linalg.lu_factor(self.matrix)
        self.nodes,weights=triangle_rule(laplace.triangles,laplace.testing_order)
        self.weights=weights/weights.sum(-1,keepdim=True)
        self._versions=self._state()

    def _state(self):
        return (id(self.laplace),self.current_regions)+tuple((id(t),t._version) for t in
            (self.exterior,*self.interiors,self.indicator,self.current_faces,self.contrast,
             self.charge_test,self.lengths,self.matrix,self.lu,self.pivots,self.nodes,self.weights))

    def assert_current(self):
        self.laplace._assert_current()
        if self._versions!=self._state():
            raise RuntimeError('Mixed DC operator was mutated')

    def solve(self,bindings,polarizations,boundary_potentials,*,net_charges=None):
        self.assert_current()
        return MixedStaticResponse(self,bindings,polarizations,boundary_potentials,net_charges=net_charges)


    def prepare_cell_response(self,bindings):
        """Prepare neutral-body maps from dielectric P and current face means.

        Rows follow the explicit dielectric binding/cell order. Input columns
        are dielectric polarization xyz followed by current-region face means.
        The Jacobian is exact for this fixed linear discretization; it is not
        a finite difference or a quadrature convergence certificate.
        """
        self.assert_current()
        bindings=tuple(bindings)
        for binding in bindings:
            binding.assert_current()
            if binding.surface_system is not self.laplace.system:
                raise ValueError('Mixed response volumes must bind the exact surface system')
        counts=tuple(len(binding.volume.cells) for binding in bindings)
        face_counts=tuple(self.laplace.body_faces[region-1] for region in self.current_regions)
        points=tuple(binding.volume.vertices[binding.volume.cells].mean(1) for binding in bindings)
        cell_columns=3*sum(counts)
        def evaluate(value):
            polarization=value[:cell_columns].reshape(-1,3).split(counts)
            boundary=value[cell_columns:].split(face_counts)
            response=self.solve(bindings,polarization,boundary)
            fields=[response.fields(point,binding.volume.region_id)[1].flatten()
                    for point,binding in zip(points,bindings)]
            electric=torch.cat(fields) if fields else value.new_empty(0)
            return torch.cat((electric,response.density,response.offsets))
        sample=self.laplace.triangles
        zero=sample.new_zeros(cell_columns+sum(face_counts))
        jacobian=torch.autograd.functional.jacobian(evaluate,zero,create_graph=True,vectorize=False)
        surface_count=len(sample)
        return {'electric_response':jacobian[:cell_columns],
                'surface_response':jacobian[cell_columns:cell_columns+surface_count],
                'offset_response':jacobian[cell_columns+surface_count:],
                'polarization_columns':cell_columns,'current_face_counts':face_counts,
                'source_regions':tuple(binding.volume.region_id for binding in bindings),
                'current_regions':self.current_regions,
                'receipt':{'preparation':'native_jacobian_of_fixed_mixed_dc_operator',
                           'neutral_current_bodies':True,'physical_qualification':False}}


class MixedStaticResponse:
    """Owned source response using an already factored mixed interface."""
    def __init__(self,operator,bindings,polarizations,boundary_potentials,*,net_charges=None):
        if not isinstance(operator,MixedStaticOperator):
            raise TypeError('Mixed DC response requires a prepared mixed operator')
        operator.assert_current()
        self.operator=operator
        laplace=operator.laplace
        self.current_regions=operator.current_regions
        bindings,polarizations=tuple(bindings),tuple(polarizations)
        if any(binding.volume.region_id in self.current_regions for binding in bindings):
            raise ValueError('Mixed DC polarization sources must belong to dielectric bodies')
        count=len(laplace.triangles)
        decoder=StaticVolumeFieldResponse(laplace.system,operator.exterior,operator.interiors,bindings,
            polarizations,laplace.triangles.new_zeros(count),testing_order=laplace.testing_order)
        boundary_values=tuple(boundary_potentials)
        if len(boundary_values)!=len(self.current_regions):
            raise ValueError('Each current body requires face potentials in current-region order')
        boundary=laplace.triangles.new_zeros(count)
        for region,value in zip(self.current_regions,boundary_values):
            laplace._real(value,(laplace.body_faces[region-1],),'Current boundary potential')
            boundary=boundary+laplace.indicator[:,region-1]*torch.cat([
                value if i==region-1 else value.new_zeros(size)
                for i,size in enumerate(laplace.body_faces)])
        nodes,weights=operator.nodes,operator.weights
        means=[]
        for points,normal,weight in zip(nodes,laplace.normals,weights):
            potential,electric,jump=decoder._particular(points,principal_value=True,
                                                       normals=normal.expand_as(points))
            means.append(torch.cat(((potential*weight).sum()[None],
                (electric*weight[:,None]).sum(0),(jump*weight).sum()[None])))
        means=torch.stack(means)
        exterior_normal=(means[:,1:4]*laplace.normals).sum(-1)+means[:,4]/2
        charges=nodes.new_zeros(len(self.current_regions)) if net_charges is None else laplace._real(
            net_charges,(len(self.current_regions),),'Current body net charge')
        self.rhs=torch.cat((torch.where(operator.current_faces,boundary-means[:,0],operator.contrast*exterior_normal),
                            charges/(operator.exterior*operator.lengths)))
        solved=torch.linalg.lu_solve(operator.lu,operator.pivots,self.rhs[:,None])[:,0]
        residual=operator.matrix@solved-self.rhs
        threshold=512*torch.finfo(nodes.dtype).eps*(operator.matrix.abs()@solved.abs()+self.rhs.abs())
        if not bool(torch.isfinite(solved).all()) or not bool((residual.abs()<=threshold).all()):
            raise RuntimeError('Mixed DC interface failed its componentwise residual')
        self.density,self.offsets=solved[:count],solved[count:]
        self.decoder=StaticVolumeFieldResponse(laplace.system,operator.exterior,operator.interiors,bindings,
            polarizations,self.density,testing_order=laplace.testing_order)
        self.receipt={'formulation':'mixed_dielectric_floating_current_dc_v1',
            'current_regions':self.current_regions,'offset_order':'current_region_order',
            'boundary_condition_testing':'constant_panel_face_mean','reused_factorization':True,
            'physical_qualification':False,'pointwise_interface_continuity_verified':False,
            'residual_max':float(residual.abs().max().detach())}
        self._versions=self._state()

    def _state(self):
        return (id(self.operator),id(self.decoder),self.current_regions)+tuple((id(t),t._version) for t in
            (self.rhs,self.density,self.offsets))

    def fields(self,points,region_id):
        self.operator.assert_current()
        if self._versions!=self._state():
            raise RuntimeError('Mixed DC interface state was mutated')
        if region_id in self.current_regions:
            raise ValueError('Current interior fields require their physical current closure')
        return self.decoder.fields(points,region_id)
