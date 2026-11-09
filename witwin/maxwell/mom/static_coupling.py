"""Dense DC dielectric interface response for induced volume polarization.

Piecewise-constant single-layer charge Galerkin testing is an internal candidate.
It is separate from finite-frequency PMCHWT and has no production qualification.
"""
import math
import torch
from .fields import _static_panel_gradient, _triangle_distance, classify_queries
from .vie_operator import cell_green_potentials, _static_panel_potential
from .volume_mesh import BoundVolumeRegion
from .quadrature import triangle_rule


class StaticVolumeFieldResponse:
    """Decode supplied DC potential-layer charge without assembling or solving.

    Sources and charge retain native gradients and own their tensor storage.
    Algebraic convergence is the caller's responsibility, not a query claim.
    """
    def __init__(self, system, epsilon_exterior, epsilon_interiors, bindings, polarizations,
                 charge, *, testing_order=4):
        self._initialize(system,epsilon_exterior,epsilon_interiors,bindings,polarizations,testing_order)
        mesh = system.bodies[0].mesh
        if (not isinstance(charge,torch.Tensor) or charge.device != mesh.device or charge.dtype != mesh.dtype
                or charge.shape != (len(self.triangles),) or not bool(torch.isfinite(charge).all())):
            raise ValueError('DC surface charge requires a finite matching real CUDA face vector')
        self.charge = charge.clone()
        self.receipt = {'formulation':'dc_volume_polarization_constant_panel_galerkin_v2',
            'testing_order':testing_order,'surface_unknown':'potential_single_layer_density_V_per_m',
            'recovered_charge_decoder':True,'physical_qualification':False,
            'testing_quadrature_error_estimated':False,'surface_discretization_error_estimated':False}
        self._versions = self._state_versions()

    def _initialize(self, system, epsilon_exterior, epsilon_interiors, bindings, polarizations, testing_order):
        system.assert_current()
        if type(testing_order) is not int or testing_order < 2:
            raise ValueError('DC surface testing requires integer order >= 2')
        self.testing_order = testing_order
        self.system = system
        mesh = system.bodies[0].mesh

        def sample(value):
            if isinstance(value, torch.Tensor):
                if value.device != mesh.device or value.dtype != mesh.dtype or value.ndim != 0:
                    raise ValueError('DC permittivities must be matching real CUDA scalars')
            elif type(value) in (int,float) and math.isfinite(value):
                value = mesh.vertices.new_tensor(value)
            else:
                raise ValueError('DC permittivities must be finite positive real samples')
            if not bool(torch.isfinite(value)) or not bool(value > 0):
                raise ValueError('DC permittivities must be finite positive real samples')
            return value.clone()

        self.exterior = sample(epsilon_exterior)
        self.interiors = tuple(sample(value) for value in epsilon_interiors)
        if len(self.interiors) != len(system.bodies):
            raise ValueError('Each body requires its own DC permittivity sample')
        self.bindings, source_values = tuple(bindings), tuple(polarizations)
        if len(self.bindings) != len(source_values):
            raise ValueError('Each induced volume requires a polarization array')
        used, owned = set(), []
        for binding, value in zip(self.bindings,source_values):
            if not isinstance(binding,BoundVolumeRegion) or binding.surface_system is not system:
                raise ValueError('DC volumes must bind the exact surface system')
            binding.assert_current()
            region = binding.volume.region_id
            if region in used:
                raise ValueError('Duplicate DC volume region')
            used.add(region)
            if (not isinstance(value,torch.Tensor) or value.device != mesh.device or value.dtype != mesh.dtype
                    or value.shape != (len(binding.volume.cells),3) or not bool(torch.isfinite(value).all())):
                raise ValueError('DC polarization requires finite matching real CUDA cell vectors')
            owned.append(value.clone())
        self.polarizations = tuple(owned)
        self.triangles = torch.cat([body.mesh.triangles for body in system.bodies])
        self.normals = torch.cat([body.mesh.normals for body in system.bodies])
        self.centers = self.triangles.mean(1)
        self.face_regions = torch.cat([torch.full((len(body.mesh.faces),),body.region_id,
            device=mesh.device,dtype=torch.int64) for body in system.bodies])
        self.face_epsilon = torch.cat([value.expand(len(body.mesh.faces))
                                     for body,value in zip(system.bodies,self.interiors)])

    def _state_versions(self):
        return (id(self.system),self.testing_order,tuple(id(binding) for binding in self.bindings))+tuple(
            (id(value),value._version) for value in (self.exterior,*self.interiors,*self.polarizations,
                self.triangles,self.normals,self.centers,self.face_regions,self.face_epsilon,self.charge))

    def _assert_current(self):
        self.system.assert_current()
        for binding in self.bindings:
            binding.assert_current()
        if self._state_versions() != self._versions:
            raise RuntimeError('DC interface response state was mutated')

    def _particular(self, points, *, principal_value=False, normals=None):
        potential, electric = points.new_zeros(len(points)),torch.zeros_like(points)
        jump = points.new_zeros(len(points))
        for binding,polarization in zip(self.bindings,self.polarizations):
            volume = binding.volume
            epsilon = self.interiors[volume.region_id-1]
            _,gradient,hessian = cell_green_potentials(volume,points,0.,order=2,principal_value=principal_value)
            potential = potential-torch.einsum('qti,ti->q',gradient.real,polarization)/epsilon
            electric = electric+torch.einsum('qtij,tj->qi',hessian.real,polarization)/epsilon
            if principal_value:
                tetrahedra = volume.vertices[volume.cells]
                face_ids = torch.tensor([[1,2,3],[0,3,2],[0,1,3],[0,2,1]],device=points.device)
                triangles = tetrahedra[:,face_ids].reshape(-1,3,3)
                tolerance = 128*torch.finfo(points.dtype).eps*torch.linalg.vector_norm(
                    volume.vertices.amax(0)-volume.vertices.amin(0))
                on = (_triangle_distance(points,triangles) <= tolerance).reshape(len(points),-1,4).any(-1)
                boundary_p = on.to(points.dtype)@polarization
                jump = jump+(boundary_p*normals).sum(-1)/epsilon
        return potential,electric,jump

    def fields(self, points, region_id):
        """Potential and E at off-interface points; geometry fixes region ownership."""
        self._assert_current()
        region = self.system.region(region_id)
        if (not isinstance(points,torch.Tensor) or points.device != self.centers.device
                or points.dtype != self.centers.dtype or points.ndim != 2 or points.shape[1] != 3
                or not bool(torch.isfinite(points).all())):
            raise ValueError('DC queries require matching finite real CUDA [Q,3] points')
        owners = torch.zeros(len(points),device=points.device,dtype=torch.int64)
        for body in self.system.bodies:
            inside,_ = classify_queries(body.mesh,points)
            owners[inside] = body.region_id
        if bool((owners != region).any()):
            raise ValueError('DC query geometry does not match requested region')
        potential,electric,_ = self._particular(points)
        count = len(self.triangles)
        for start in range(0,len(points),32):
            batch = points[start:start+32]
            observers = batch[:,None].expand(-1,count,-1).reshape(-1,3)
            panels = self.triangles.repeat(len(batch),1,1)
            normals = self.normals.repeat(len(batch),1)
            gradient = _static_panel_gradient(observers,panels)
            scalar = _static_panel_potential(observers,panels,normals,gradient).reshape(len(batch),count)
            potential[start:start+len(batch)] = potential[start:start+len(batch)]+scalar@self.charge
            electric[start:start+len(batch)] = electric[start:start+len(batch)]-torch.einsum(
                'qti,t->qi',gradient.reshape(len(batch),count,3),self.charge)
        return potential,electric


class StaticVolumeResponse(StaticVolumeFieldResponse):
    """Solve the global finite-dielectric DC interface and decode its fields."""
    def __init__(self, system, epsilon_exterior, epsilon_interiors, bindings, polarizations, *, testing_order=4):
        self._initialize(system,epsilon_exterior,epsilon_interiors,bindings,polarizations,testing_order)
        mesh = system.bodies[0].mesh
        count = len(self.triangles)
        nodes,weights = triangle_rule(self.triangles,testing_order)
        weights = weights/weights.sum(-1,keepdim=True)
        rows,source_means,jump_means = [],[],[]
        # Average exact source-panel gradients over each testing triangle.
        for face in range(count):
            points = nodes[face]
            normals = self.normals[face].expand_as(points)
            observers = points[:,None].expand(-1,count,-1).reshape(-1,3)
            panels = self.triangles.repeat(len(points),1,1)
            self_mask = (torch.arange(count,device=mesh.device) == face).expand(len(points),-1).flatten()
            gradient = _static_panel_gradient(observers,panels,principal_value=self_mask).reshape(len(points),count,3)
            rows.append(torch.einsum('qti,q->ti',gradient,weights[face]))
            _,particular,jump = self._particular(points,principal_value=True,normals=normals)
            source_means.append(torch.einsum('qi,q->i',particular,weights[face]))
            jump_means.append((jump*weights[face]).sum())
        self.mean_gradient = torch.stack(rows)
        self.normal_derivative = torch.einsum('qti,qi->qt',self.mean_gradient,self.normals)
        self.particular_mean = torch.stack(source_means)
        self.jump_mean = torch.stack(jump_means)
        exterior_particular = self.particular_mean+self.normals*self.jump_mean[:,None]/2
        contrast = 2*(self.face_epsilon-self.exterior)/(self.face_epsilon+self.exterior)
        matrix = torch.eye(count,device=mesh.device,dtype=mesh.dtype)+contrast[:,None]*self.normal_derivative
        rhs = contrast*(exterior_particular*self.normals).sum(-1)
        self.charge = torch.linalg.solve(matrix,rhs)
        residual = torch.linalg.vector_norm(matrix@self.charge-rhs)
        tolerance = 512*torch.finfo(mesh.dtype).eps*torch.linalg.vector_norm(rhs)
        if not bool(torch.isfinite(self.charge).all()) or not bool(residual <= tolerance):
            raise RuntimeError('DC interface solve failed its true algebraic residual check')
        self.receipt = {'formulation': 'dc_volume_polarization_constant_panel_galerkin_v2',
            'testing_order': testing_order, 'testing_quadrature_error_estimated': False,
            'surface_unknown': 'potential_single_layer_density_V_per_m',
            'absolute_residual': float(residual.detach()), 'residual_threshold': float(tolerance.detach()),
            'physical_qualification': False, 'surface_discretization_error_estimated': False,
            'linear_background': 'regional_particular_plus_dielectric_interface_only'}
        self._versions = self._state_versions()

    def _state_versions(self):
        return super()._state_versions()+tuple((id(value),value._version) for value in (
            self.normal_derivative,self.mean_gradient,self.particular_mean,self.jump_mean))

    def boundary_fields(self):
        """One-sided face-averaged fields using the assembled testing quadrature."""
        self._assert_current()
        mean = self.particular_mean-torch.einsum('qti,t->qi',self.mean_gradient,self.charge)
        half_jump = self.normals*((self.jump_mean+self.charge)/2)[:,None]
        return mean+half_jump,mean-half_jump
