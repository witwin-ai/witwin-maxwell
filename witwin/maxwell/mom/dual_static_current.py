"""Conservative coupled electric/magnetic DC currents at volume cell centers.

The vacuum Green representation gives E=-grad(phi_e)-C M and
H=-grad(phi_m)+C J, with C the global static Biot-Savart action. Face-mean
potentials enforce zero distributional divergence and boundary normal current
for each channel. Exterior potential offsets and interface traces are separate
obligations; this experimental dense closure is not a finite-body field decoder.
"""
import torch
from .static_current import FaceCurrentBalance
from .vie_operator import cell_green_potentials
from .magnetostatic import cell_current_continuity


class DualStaticCurrentBalance:
    def __init__(self, volumes, sigma_e, sigma_m, *, max_matrix_bytes=512*1024**2):
        self.volumes = tuple(volumes)
        if not self.volumes or len(self.volumes) != len(sigma_e) or len(self.volumes) != len(sigma_m):
            raise ValueError('Dual DC requires one electric/magnetic conductivity per volume')
        reference = self.volumes[0].vertices
        if len({v.region_id for v in self.volumes}) != len(self.volumes) or any(
                v.vertices.device != reference.device or v.vertices.dtype != reference.dtype for v in self.volumes):
            raise ValueError('Dual DC volumes require unique regions and matching CUDA precision')
        count = sum(len(v.cells) for v in self.volumes)
        if type(max_matrix_bytes) is not int or max_matrix_bytes <= 0 or 100*count*count*reference.element_size() > max_matrix_bytes:
            raise MemoryError('Dual DC dense preparation exceeds its declared matrix workspace estimate')
        self.blocks = tuple(FaceCurrentBalance(v) for v in self.volumes)
        self.counts = tuple(len(v.cells) for v in self.volumes)
        electric, magnetic, projectors, recovery = [], [], [], []
        for block, a, b in zip(self.blocks, sigma_e, sigma_m):
            for value, output in ((a,electric),(b,magnetic)):
                scalar = block._coefficient(value, 'DC conductivity')
                if not bool((scalar > 0).all()) or not bool((scalar == scalar[0]).all()):
                    raise ValueError('Dual DC requires positive homogeneous conductivity per body')
                output.append(scalar.repeat_interleave(3).clone())
            gradient = reference.new_zeros(len(block.volume.cells), 3, block.n_dofs)
            gradient.scatter_(2, block.cell_dofs[:,None,:].expand(-1,3,-1), block.gradients.transpose(1,2))
            gradient = gradient.reshape(-1,block.n_dofs)[:,1:]
            weighted = gradient.T*block.volume.volumes.repeat_interleave(3)[None]
            recover = torch.linalg.solve(weighted@gradient, weighted)
            recovery.append(recover)
            projectors.append(torch.eye(len(gradient),device=reference.device,dtype=reference.dtype)-gradient@recover)
        self.electric = torch.cat(electric); self.magnetic = torch.cat(magnetic)
        self.projector = torch.block_diag(*projectors)
        self.recovery = tuple(recovery)
        self.points = torch.cat([v.vertices[v.cells].mean(1) for v in self.volumes])
        columns = []
        for volume in self.volumes:
            _, gradient, _ = cell_green_potentials(volume,self.points,0.,order=2)
            x,y,z = gradient.real.unbind(-1); zero = torch.zeros_like(x)
            cross = torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(count,len(volume.cells),3,3)
            columns.append(cross.permute(0,2,1,3).reshape(3*count,-1))
        self.curl = torch.cat(columns,1)
        eye = torch.eye(3*count,device=reference.device,dtype=reference.dtype)
        projected_curl = self.projector@self.curl
        self.matrix = torch.cat((torch.cat((eye,self.electric[:,None]*projected_curl),1),
                                 torch.cat((-self.magnetic[:,None]*projected_curl,eye),1)),0)
        if not bool(torch.isfinite(self.matrix).all()):
            raise ValueError('Dual DC matrix is nonfinite')
        self._versions = self._state()

    def _state(self):
        return (self.counts,tuple(id(v) for v in self.volumes),tuple(id(b) for b in self.blocks),
                tuple((id(t),t._version) for t in (self.electric,self.magnetic,self.projector,self.points,self.curl,self.matrix,*self.recovery)))

    def assert_current(self):
        if self._versions != self._state():
            raise RuntimeError('Dual DC preparation was mutated')
        for block in self.blocks: block._assert_current()

    def solve(self, electric_source, magnetic_source):
        """Solve linear impressed currents; nonlinear constitutive closure is separate."""
        self.assert_current()
        shape = (sum(self.counts),3)
        for value in (electric_source,magnetic_source):
            if (not isinstance(value,torch.Tensor) or value.shape != shape or value.device != self.points.device
                    or value.dtype != self.points.dtype or not bool(torch.isfinite(value).all())):
                raise ValueError('Dual DC sources require matching finite real CUDA [cells,3]')
        rhs = torch.cat((self.projector@electric_source.flatten(),self.projector@magnetic_source.flatten()))
        currents = torch.linalg.solve(self.matrix,rhs)
        j,m = currents.chunk(2)
        if not bool(torch.isfinite(currents).all()):
            raise RuntimeError('Dual DC current solve is nonfinite')
        electric = (j-electric_source.flatten())/self.electric
        magnetic = (m-magnetic_source.flatten())/self.magnetic
        potential_e, potential_m = [], []
        for block, recovery, de, dm in zip(self.blocks,self.recovery,
                (electric_source.flatten()/self.electric-self.curl@m).split(tuple(3*n for n in self.counts)),
                (magnetic_source.flatten()/self.magnetic+self.curl@j).split(tuple(3*n for n in self.counts))):
            for drive, target in ((de,potential_e),(dm,potential_m)):
                phi = torch.cat((drive.new_zeros(1),recovery@drive))
                target.append(phi-(phi*block.nodal_volume).sum()/block.nodal_volume.sum())
        reconstructed_e = torch.cat([block.electric(phi).flatten() for block,phi in zip(self.blocks,potential_e)])-self.curl@m
        reconstructed_h = torch.cat([block.electric(phi).flatten() for block,phi in zip(self.blocks,potential_m)])+self.curl@j
        residual = self.matrix@currents-rhs
        scales = torch.cat((j.abs()+(self.electric*(self.curl@m)).abs()+rhs[:len(j)].abs(),
                            m.abs()+(self.magnetic*(self.curl@j)).abs()+rhs[len(j):].abs()))
        threshold = 1024*torch.finfo(currents.dtype).eps*len(currents)*scales.norm()
        if (not all(bool(torch.isfinite(value).all()) for value in
                    (electric,magnetic,reconstructed_e,reconstructed_h,residual,threshold))
                or not bool(torch.isfinite(residual.norm())) or not bool(residual.norm() <= threshold)):
            raise RuntimeError('Dual DC original coupled current residual failed')
        continuity = []
        for volume,jbody,mbody in zip(self.volumes,j.reshape(shape).split(self.counts),m.reshape(shape).split(self.counts)):
            continuity.append({key:float(cell_current_continuity(volume,value)['area_weighted_jump_rms_A_per_m2'].detach())
                               for key,value in (('electric',jbody),('magnetic',mbody))})
        return {'electric':electric.reshape(shape),'magnetic':magnetic.reshape(shape),
                'electric_current':j.reshape(shape),'magnetic_current':m.reshape(shape),
                'electric_potentials':tuple(potential_e),'magnetic_potentials':tuple(potential_m),
                'receipt':{'formulation':'dual_conservative_cell_current_dc_v1',
                    'coupled_current_residual':float(residual.norm().detach()),'threshold':float(threshold.detach()),
                    'electric_reconstruction_error':float((electric-reconstructed_e).norm().detach()),
                    'magnetic_reconstruction_error':float((magnetic-reconstructed_h).norm().detach()),
                    'normal_current_jumps':continuity,'exterior_potential_coupling':False,
                    'interface_trace_convergence':False,'physical_qualification':False}}


class DualStaticExterior:
    """Neutral floating potentials plus global electric/magnetic volume curls.

    Magnetic scalar potential is in amperes and electric scalar potential in
    volts. Both scalar Laplace solves have zero net body flux; their offsets
    remove body gauges. Face-mean matching is not pointwise trace certification.
    """
    def __init__(self, closure, result, bindings, laplace):
        from .harmonic_current import surface_face_potentials
        from ..constants import EPSILON_0, MU_0
        closure.assert_current()
        self.closure, self.bindings = closure, tuple(bindings)
        if len(self.bindings) != len(closure.volumes) or len(laplace.system.bodies) != len(self.bindings):
            raise ValueError('Dual exterior requires the complete owning surface system')
        for binding,volume in zip(self.bindings,closure.volumes):
            binding.assert_current()
            if binding.volume is not volume or binding.surface_system is not laplace.system:
                raise ValueError('Dual exterior must use its exact surface-volume bindings')
        if sorted(v.region_id for v in closure.volumes) != list(range(1,len(self.bindings)+1)):
            raise ValueError('Dual exterior volumes must follow the native body order')
        boundaries = []
        for key in ('electric_potentials','magnetic_potentials'):
            boundaries.append(torch.cat([surface_face_potentials(block,binding,phi)
                for block,binding,phi in sorted(zip(closure.blocks,self.bindings,result[key]),key=lambda item:item[1].volume.region_id)]))
        self.electric_layer = laplace.solve_face_means(boundaries[0],epsilon_exterior=EPSILON_0)
        self.magnetic_layer = laplace.solve_face_means(boundaries[1],epsilon_exterior=MU_0)
        self.j = result['electric_current'].clone(); self.m = result['magnetic_current'].clone()
        shape=(sum(closure.counts),3)
        for current in (self.j,self.m):
            if current.shape != shape or current.device != closure.points.device or current.dtype != closure.points.dtype:
                raise ValueError('Dual exterior currents must match the prepared volume order')
        self._versions = self._state()
        self.receipt = {'formulation':'dual_floating_vacuum_current_exterior_v1',
            'electric_potential_units':'V','magnetic_potential_units':'A',
            'electric_density_units':'V/m','magnetic_density_units':'A/m',
            'electric_boundary_residual_V':self.electric_layer.receipt['boundary_residual_max_V'],
            'magnetic_boundary_residual_A':self.magnetic_layer.receipt['boundary_residual_max_V'],
            'net_electric_charge_C':self.electric_layer.charges.detach().cpu().tolist(),
            'net_magnetic_flux_Wb':self.magnetic_layer.charges.detach().cpu().tolist(),
            'boundary_testing':'constant_panel_face_mean','pointwise_trace_convergence':False,
            'physical_qualification':False}

    def _state(self):
        return (id(self.closure),tuple(id(binding) for binding in self.bindings),
                id(self.electric_layer),id(self.magnetic_layer),
                tuple((id(t),t._version) for t in (self.j,self.m)))

    def fields(self,points):
        from .magnetostatic import cell_magnetostatic_current_fields
        if self._versions != self._state():
            raise RuntimeError('Dual exterior preparation or currents were mutated')
        self.closure.assert_current()
        for binding in self.bindings: binding.assert_current()
        phi_e,electric=self.electric_layer.fields(points)
        phi_m,magnetic=self.magnetic_layer.fields(points)
        for volume,j,m in zip(self.closure.volumes,self.j.split(self.closure.counts),self.m.split(self.closure.counts)):
            electric=electric-cell_magnetostatic_current_fields(volume,m,points)[0]
            magnetic=magnetic+cell_magnetostatic_current_fields(volume,j,points)[0]
        return {'electric_V_per_m':electric,'magnetic_A_per_m':magnetic,
                'electric_scalar_potential_V':phi_e,'magnetic_scalar_potential_A':phi_m}
