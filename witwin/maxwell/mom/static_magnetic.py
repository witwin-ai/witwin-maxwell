"""Scalar-permeability DC interface response for conservative volume currents."""
import math
import torch
from .static_floating import FloatingPotentialOperator
from .fields import _static_panel_gradient,classify_queries
from .quadrature import triangle_rule
from .vie_operator import cell_green_potentials
from .magnetostatic import cell_current_continuity


class MagnetostaticInterfaceResponse:
    """Biot-Savart H plus a scalar-potential layer enforcing normal B continuity.

    The shared Laplace panel means are reused from the floating operator. This
    static magnetic solve is distinct from its electric Dirichlet/charge solve.
    No free sheet current is introduced: tangential H is continuous. The source
    current must independently satisfy distributional continuity.
    """
    def __init__(self,laplace,bindings,currents,mu_exterior,mu_interiors,*,background_h=None):
        self._initialize(laplace,bindings,currents,mu_exterior,mu_interiors,background_h=background_h)
        sample = laplace.triangles
        nodes,weights = triangle_rule(laplace.triangles,laplace.testing_order)
        weights = weights/weights.sum(-1,keepdim=True)
        field = self._source(nodes.reshape(-1,3),principal_value=True).reshape(*nodes.shape)
        self.source_mean = (field*weights[...,None]).sum(1)
        mu_face = torch.cat([mu.expand(size) for mu,size in zip(self.interiors,laplace.body_faces)])
        contrast = 2*(mu_face-self.exterior)/(mu_face+self.exterior)
        normal_derivative = torch.einsum('qti,qi->qt',laplace.mean_gradient,laplace.normals)
        matrix = torch.eye(len(mu_face),device=sample.device,dtype=sample.dtype)+contrast[:,None]*normal_derivative
        rhs = contrast*(self.source_mean*laplace.normals).sum(-1)
        self.density = torch.linalg.solve(matrix,rhs)
        residual = (matrix@self.density-rhs).norm()
        threshold = 512*torch.finfo(sample.dtype).eps*rhs.norm()
        if not bool(torch.isfinite(self.density).all()) or not bool(residual <= threshold):
            raise RuntimeError('Magnetostatic interface failed its true residual check')
        self.receipt = {'formulation':'scalar_mu_biot_savart_single_layer_v1','testing_order':laplace.testing_order,
            'surface_unknown':'magnetic_potential_layer_density_A_per_m',
            'boundary_condition_testing':'constant_panel_face_mean',
            'pointwise_interface_continuity_verified':False,'residual':float(residual.detach()),
            'threshold':float(threshold.detach()),'free_surface_current':False,'physical_qualification':False,
            'source_current_jump_rms_A_per_m2':[float(cell_current_continuity(b.volume,j)[
                'area_weighted_jump_rms_A_per_m2'].detach()) for b,j in zip(self.bindings,self.currents)],
            'quadrature_convergence_certified':False,'spatial_convergence_certified':False}
        self._versions = self._state()

    def _initialize(self,laplace,bindings,currents,mu_exterior,mu_interiors,*,background_h=None):
        if not isinstance(laplace,FloatingPotentialOperator):
            raise TypeError('Magnetic interface requires the shared Laplace panel operator')
        laplace._assert_current()
        self.laplace = laplace
        sample = laplace.triangles
        def scalar(value):
            if type(value) in (float,int) and math.isfinite(value):
                value = sample.new_tensor(value)
            laplace._real(value,(),'DC permeability')
            if not bool(value > 0):
                raise ValueError('DC permeability must be positive')
            return value.clone()
        self.exterior = scalar(mu_exterior)
        self.interiors = tuple(scalar(value) for value in mu_interiors)
        if len(self.interiors) != len(laplace.system.bodies):
            raise ValueError('Each magnetic body requires a DC permeability')
        self.bindings = tuple(bindings)
        values = tuple(currents)
        if len(values) != len(self.bindings):
            raise ValueError('Each magnetic source volume requires its total current')
        owned,regions = [],set()
        for binding,value in zip(self.bindings,values):
            binding.assert_current()
            if binding.surface_system is not laplace.system or binding.volume.region_id in regions:
                raise ValueError('Magnetic current volumes must bind unique regions of the exact system')
            regions.add(binding.volume.region_id)
            laplace._real(value,(len(binding.volume.cells),3),'Total DC current')
            owned.append(value.clone())
        self.currents = tuple(owned)
        value = sample.new_zeros(3) if background_h is None else background_h
        self.background = laplace._real(value,(3,),'Uniform magnetic background').clone()

    @classmethod
    def from_state(cls,laplace,bindings,currents,mu_exterior,mu_interiors,density,source_mean,receipt):
        """Bind owned recovered arrays without evaluating or solving the interface."""
        import copy
        result = cls.__new__(cls)
        result._initialize(laplace,bindings,currents,mu_exterior,mu_interiors)
        laplace._real(density,(len(laplace.triangles),),'Magnetic surface density')
        laplace._real(source_mean,laplace.normals.shape,'Magnetic source mean')
        result.density,result.source_mean = density.clone(),source_mean.clone()
        result.receipt = copy.deepcopy(receipt)
        result._versions = result._state()
        return result

    def _state(self):
        return (id(self.laplace),tuple(id(b) for b in self.bindings))+tuple((id(t),t._version) for t in
            (self.exterior,*self.interiors,*self.currents,self.background,self.source_mean,self.density))

    def _assert_current(self):
        self.laplace._assert_current()
        for binding in self.bindings:
            binding.assert_current()
        if self._versions != self._state():
            raise RuntimeError('Magnetostatic interface state was mutated')

    def _source(self,points,*,principal_value=False):
        magnetic = self.background.expand_as(points).clone()
        for binding,current in zip(self.bindings,self.currents):
            _,gradient,_ = cell_green_potentials(binding.volume,points,0.,order=2,principal_value=principal_value)
            magnetic = magnetic+torch.linalg.cross(gradient.real,current[None].expand_as(gradient.real)).sum(1)
        return magnetic

    def boundary_fields(self):
        """Return exterior/interior face-mean H; normal mu H is continuous."""
        self._assert_current()
        mean = self.source_mean-torch.einsum('qti,t->qi',self.laplace.mean_gradient,self.density)
        jump = self.laplace.normals*self.density[:,None]/2
        return mean+jump,mean-jump

    def fields(self,points,region_id):
        self._assert_current()
        op = self.laplace
        if not isinstance(points,torch.Tensor) or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('Magnetostatic queries require [Q,3] coordinates')
        op._real(points,points.shape,'Magnetostatic queries')
        region = op.system.region(region_id)
        owners = torch.zeros(len(points),device=points.device,dtype=torch.int64)
        for body in op.system.bodies:
            inside,_ = classify_queries(body.mesh,points)
            owners[inside] = body.region_id
        if bool((owners != region).any()):
            raise ValueError('Magnetostatic query geometry does not match its region')
        magnetic = self._source(points)
        count = len(op.triangles)
        correction = []
        for batch in points.split(32):
            gradient = _static_panel_gradient(batch[:,None].expand(-1,count,-1).reshape(-1,3),
                                              op.triangles.repeat(len(batch),1,1)).reshape(len(batch),count,3)
            correction.append(torch.einsum('qti,t->qi',gradient,self.density))
        if len(points):
            magnetic = magnetic-torch.cat(correction)
        mu = self.exterior if region == 0 else self.interiors[region-1]
        return magnetic,mu*magnetic
