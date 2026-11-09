"""Exterior Laplace response for prescribed body potentials up to constants.

A single layer enforces the boundary potential in constant-panel weak tests.
One prescribed total charge per body fixes its floating potential offset.
The fictitious harmonic extension inside each body is used only to identify
its enclosed charge; physical interior fields belong to the volume DC block.
"""
import math
import torch
from .mesh import SurfaceSystem
from .quadrature import triangle_rule
from .fields import _static_panel_gradient,classify_queries
from .vie_operator import _static_panel_potential


class FloatingPotentialOperator:
    """Reusable dense CUDA exterior Dirichlet and per-body charge system."""
    def __init__(self,system,*,testing_order=4):
        if not isinstance(system,SurfaceSystem):
            raise TypeError('Floating potential requires a validated surface system')
        system.assert_current()
        if type(testing_order) is not int or testing_order < 2:
            raise ValueError('Floating potential testing order must be an integer >= 2')
        self.system,self.testing_order = system,testing_order
        self.triangles = torch.cat([body.mesh.triangles for body in system.bodies])
        self.normals = torch.cat([body.mesh.normals for body in system.bodies])
        self.areas = torch.cat([body.mesh.areas for body in system.bodies])
        self.body_faces = tuple(len(body.mesh.faces) for body in system.bodies)
        count = len(self.triangles)
        body_ids = torch.cat([torch.full((size,),index,device=self.triangles.device,dtype=torch.int64)
                              for index,size in enumerate(self.body_faces)])
        self.indicator = torch.nn.functional.one_hot(body_ids,len(system.bodies)).to(self.triangles.dtype)
        self.body_areas = self.indicator.T@self.areas
        self.charge_test = self.indicator.T*self.areas[None,:]
        self.lengths = self.body_areas.sqrt()
        nodes,weights = triangle_rule(self.triangles,testing_order)
        weights = weights/weights.sum(-1,keepdim=True)
        rows,gradient_rows = [],[]
        for face in range(count):
            points = nodes[face]
            observers = points[:,None].expand(-1,count,-1).reshape(-1,3)
            panels = self.triangles.repeat(len(points),1,1)
            normals = self.normals.repeat(len(points),1)
            self_mask = (torch.arange(count,device=points.device) == face).expand(len(points),-1).flatten()
            gradient = _static_panel_gradient(observers,panels,principal_value=self_mask)
            potential = _static_panel_potential(observers,panels,normals,gradient).reshape(len(points),count)
            rows.append(torch.einsum('qt,q->t',potential,weights[face]))
            gradient_rows.append(torch.einsum('qti,q->ti',gradient.reshape(len(points),count,3),weights[face]))
        self.single_layer = torch.stack(rows)
        self.mean_gradient = torch.stack(gradient_rows)
        top = torch.cat((self.single_layer,-self.indicator),1)
        bottom = torch.cat((self.charge_test/self.lengths[:,None],
                            self.triangles.new_zeros(len(system.bodies),len(system.bodies))),1)
        self.matrix = torch.cat((top,bottom))
        self.lu,self.pivots = torch.linalg.lu_factor(self.matrix)
        self._versions = self._state_versions()

    def _state_versions(self):
        return (id(self.system),self.testing_order,self.body_faces)+tuple((id(t),t._version) for t in (
            self.triangles,self.normals,self.areas,self.indicator,self.body_areas,self.charge_test,
            self.lengths,self.single_layer,self.mean_gradient,self.matrix,self.lu,self.pivots))

    def _assert_current(self):
        self.system.assert_current()
        if self._versions != self._state_versions():
            raise RuntimeError('Floating potential operator was mutated')

    def _real(self,value,shape,name):
        if (not isinstance(value,torch.Tensor) or value.device != self.triangles.device
                or value.dtype != self.triangles.dtype or value.shape != shape
                or not bool(torch.isfinite(value).all())):
            raise ValueError(name+' requires finite matching real CUDA values')
        return value

    def solve(self,boundary_potentials,*,epsilon_exterior,net_charges=None):
        """Solve exterior potential vanishing at infinity and floating offsets.

        Each boundary input is a continuous nodal potential in its body mesh
        vertex order, in volts. Net charges are coulombs (default individually
        neutral). The density has units V/m; it is an equivalent exterior layer,
        not the physical local free-charge density at a material interface.
        """
        self._assert_current()
        values = tuple(boundary_potentials)
        if len(values) != len(self.system.bodies):
            raise ValueError('Each floating body requires its boundary nodal potential')
        means = []
        for body,value in zip(self.system.bodies,values):
            self._real(value,(len(body.mesh.vertices),),'Boundary potential')
            means.append(value[body.mesh.faces].mean(-1))
        boundary = torch.cat(means)
        return self.solve_face_means(boundary,epsilon_exterior=epsilon_exterior,net_charges=net_charges)

    def solve_face_means(self,boundary,*,epsilon_exterior,net_charges=None):
        """Prescribe actual face-average potentials, including nonconforming DC traces."""
        self._assert_current()
        self._real(boundary,(len(self.triangles),),'Boundary face potential')
        body_count = len(self.system.bodies)
        if type(epsilon_exterior) in (int,float) and math.isfinite(epsilon_exterior):
            epsilon_exterior = boundary.new_tensor(epsilon_exterior)
        epsilon = self._real(epsilon_exterior,(),'Exterior permittivity')
        if not bool(epsilon > 0):
            raise ValueError('Exterior permittivity must be positive')
        charges = boundary.new_zeros(body_count) if net_charges is None else self._real(
            net_charges,(body_count,),'Body net charge')
        rhs = torch.cat((boundary,charges/(epsilon*self.lengths)))
        solved = torch.linalg.lu_solve(self.lu,self.pivots,rhs[:,None])[:,0]
        count = len(self.triangles)
        density,offsets = solved[:count],solved[count:]
        boundary_error = self.single_layer@density-self.indicator@offsets-boundary
        charge_error = epsilon*(self.charge_test@density)-charges
        roundoff = 512*torch.finfo(boundary.dtype).eps
        boundary_threshold = roundoff*(self.single_layer.abs()@density.abs()+self.indicator@offsets.abs()+boundary.abs())
        charge_threshold = roundoff*(epsilon*(self.charge_test@density.abs())+charges.abs())
        if (not bool(torch.isfinite(solved).all()) or not bool((boundary_error.abs() <= boundary_threshold).all())
                or not bool((charge_error.abs() <= charge_threshold).all())):
            raise RuntimeError('Floating potential solve failed boundary or per-body charge residual')
        receipt = {'formulation':'floating_exterior_laplace_constant_panel_v1',
            'testing_order':self.testing_order,'potential_at_infinity_V':0.,
            'surface_unknown':'equivalent_potential_single_layer_density_V_per_m',
            'boundary_residual_max_V':float(boundary_error.abs().max().detach()),
            'per_body_charge_residual_C':charge_error.detach().cpu().tolist(),
            'per_body_charge_threshold_C':charge_threshold.detach().cpu().tolist(),
            'charge_constraint':'specified_net_charge_per_body',
            'physical_interface_charge_recovered':False,'physical_interior_fields':False,
            'physical_qualification':False,'quadrature_error_estimated':False,
            'surface_discretization_error_estimated':False}
        return FloatingPotentialFields(self,density,offsets,epsilon,charges,receipt)


class FloatingPotentialFields:
    """Owned exterior layer state, preserving native input-potential gradients."""
    def __init__(self,operator,density,offsets,epsilon,charges,receipt):
        self.operator = operator
        self.density,self.offsets,self.epsilon,self.charges = (t.clone() for t in (density,offsets,epsilon,charges))
        self.receipt = receipt
        self._versions = self._state_versions()

    def _state_versions(self):
        return (id(self.operator),)+tuple((id(t),t._version) for t in
            (self.density,self.offsets,self.epsilon,self.charges))

    def _assert_current(self):
        self.operator._assert_current()
        if self._versions != self._state_versions():
            raise RuntimeError('Floating potential field state was mutated')

    def boundary_fields(self,*,interior_displacement=None):
        """One-sided exterior face means and optional physical charge accounting.

        interior_displacement supplies the actual interior face-mean D, including
        material polarization, in C/m^2. The layer density itself is not this
        physical free charge. Integrated bulk charge follows the interior flux;
        its spatial distribution requires the volume discretization separately.
        """
        self._assert_current()
        op = self.operator
        electric = -torch.einsum('qti,t->qi',op.mean_gradient,self.density)+op.normals*self.density[:,None]/2
        normal_flux = self.epsilon*(electric*op.normals).sum(-1)
        total = op.charge_test@normal_flux
        result = {'potential_V':op.single_layer@self.density,'exterior_electric_V_per_m':electric,
            'gauss_total_charge_C':total,'target_total_charge_C':self.charges.clone(),
            'gauss_charge_error_C':total-self.charges,'physical_qualification':False,
            'boundary_quadrature_converged':False}
        if interior_displacement is not None:
            op._real(interior_displacement,op.normals.shape,'Interior displacement')
            interior_flux = (interior_displacement*op.normals).sum(-1)
            surface_charge = normal_flux-interior_flux
            result.update(surface_free_charge_density_C_per_m2=surface_charge,
                surface_free_charge_C=op.charge_test@surface_charge,
                bulk_free_charge_C=op.charge_test@interior_flux)
        return result

    def fields(self,points):
        """Return potential/E only outside all bodies; reject surface traces."""
        self._assert_current()
        if not isinstance(points,torch.Tensor) or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('Exterior DC points require [Q,3] coordinates')
        self.operator._real(points,points.shape,'Exterior DC points')
        for body in self.operator.system.bodies:
            inside,_ = classify_queries(body.mesh,points)
            if bool(inside.any()):
                raise ValueError('Floating single layer decodes only exterior points')
        potential,electric = [],[]
        count = len(self.operator.triangles)
        for batch in points.split(32):
            observers = batch[:,None].expand(-1,count,-1).reshape(-1,3)
            panels = self.operator.triangles.repeat(len(batch),1,1)
            normals = self.operator.normals.repeat(len(batch),1)
            gradient = _static_panel_gradient(observers,panels)
            scalar = _static_panel_potential(observers,panels,normals,gradient).reshape(len(batch),count)
            potential.append(scalar@self.density)
            electric.append(-torch.einsum('qti,t->qi',gradient.reshape(len(batch),count,3),self.density))
        if not len(points):
            return points.new_empty(0),points.new_empty(0,3)
        return torch.cat(potential),torch.cat(electric)
