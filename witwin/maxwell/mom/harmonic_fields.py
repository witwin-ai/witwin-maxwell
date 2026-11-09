"""Internal regional queries for recovered dielectric harmonic states."""
import copy
from dataclasses import replace
import torch
from .harmonic_state import preparation_signature,HarmonicCoupledState
from .surface_volume import VolumeFieldResponse
from .static_coupling import StaticVolumeFieldResponse


class HarmonicFieldDecoder:
    """AC complete E/H and separately named DC potential/E queries.

    This decoder uses recovered AC coefficients and DC charge directly;
    queries do not rerun nonlinear or linear interface solves.
    """
    def __init__(self,prepared,state):
        if not isinstance(state,HarmonicCoupledState):
            raise TypeError('Harmonic fields require a recovered coupled state')
        state.assert_current()
        if state.preparation_identity != id(prepared):
            raise ValueError('Harmonic state belongs to a different preparation')
        if preparation_signature(prepared) != prepared['_signature']:
            raise RuntimeError('Harmonic field preparation was mutated')
        self.prepared,self.state = prepared,state
        self._state_versions = self._versions()
        self._ac,self._dc = {},None
        self._dc_magnetic = None
        self._dual = None

    def _versions(self):
        state = self.state
        return (state.receipt['pump_scale'],)+tuple((id(value),value._version) for value in (
            state.frequencies,state.cell_electric,state.polarization,state.dissipative_current,
            state.total_volume_current,*state.surface_coefficients,state.dc_surface_charge))

    def _assert_current(self):
        self.state.assert_current()
        if self._versions() != self._state_versions or preparation_signature(self.prepared) != self.prepared['_signature']:
            raise RuntimeError('Harmonic field state or preparation was mutated')
        for operator,bindings in zip(self.prepared['operators'],self.prepared['bindings']):
            operator.system.assert_current()
            for binding in bindings:
                binding.assert_current()

    def query_fields(self,points,region_id,*,harmonic,query_config=None):
        """Return complex E/H at one retained positive harmonic."""
        self._assert_current()
        if type(harmonic) is not int or not 1 <= harmonic < len(self.state.frequencies):
            raise ValueError('AC field queries require a retained positive harmonic; use query_dc for DC')
        if harmonic not in self._ac:
            original = self.prepared['operators'][harmonic-1]
            operator = copy.copy(original)
            pump = self.state.receipt['pump_scale']
            def scaled(carrier):
                return replace(carrier,waves=tuple(replace(wave,amplitude=tuple(pump*x for x in wave.amplitude))
                                                   for wave in carrier.waves))
            operator.incident_carrier = scaled(original.incident_carrier)
            operator.body_handles = tuple(copy.copy(handle) for handle in original.body_handles)
            for handle in operator.body_handles:
                handle.incident_carrier = scaled(handle.incident_carrier)
            bindings = self.prepared['bindings'][harmonic-1]
            counts = tuple(len(binding.volume.cells) for binding in bindings)
            self._ac[harmonic] = VolumeFieldResponse(operator,bindings,
                self.state.total_volume_current[harmonic].split(counts),self.state.surface_coefficients[harmonic-1])
        handle = self._ac[harmonic]
        result = handle.fields(points,region_id,query_config=query_config)
        self.last_query_receipt = {'harmonic':harmonic,'frequency_hz':float(self.state.frequencies[harmonic]),
            'fields':handle.last_field_receipt,'physical_qualification':False}
        return result

    def query_dc(self,points,region_id):
        """Return real electrostatic potential and E; no radiative DC H claim."""
        self._assert_current()
        if self.state.dc_tensor_response is not None:
            response=self.state.dc_tensor_response
            region=response.operator.system.region(region_id)
            if region in self.state.dc_current_regions:
                from .harmonic_current import face_potential_fields
                from .fields import classify_queries
                response.operator.exterior._validate(points,(None,3))
                inside,_=classify_queries(response.operator.system.bodies[region-1].mesh,points)
                if not bool(inside.all()):
                    raise ValueError('DC query geometry does not match requested region')
                index=self.state.dc_current_regions.index(region)
                closure=self.prepared['static']['tensor_current_closure']
                result=face_potential_fields(closure.current.blocks[index],
                    self.state.dc_face_potentials[index],self.state.dc_potential_offsets[index],points)
            else:
                result=response.fields(points,region)
            self.last_query_receipt={'harmonic':0,'frequency_hz':0.,'potential_units':'V',
                'electric_units':'V/m','static_response':response.receipt,'physical_qualification':False}
            return result
        if self.state.dc_dual_state is not None:
            fields=self.query_dc_dual(points,region_id)
            return fields['electric_scalar_potential_V'],fields['electric_V_per_m']
        if self.state.dc_potential_offsets is not None:
            return self._query_current_dc(points,region_id)
        if self._dc is None:
            bindings = self.prepared['bindings'][0]
            counts = tuple(len(binding.volume.cells) for binding in bindings)
            exterior,interiors = self.prepared['dc_permittivities']
            expected = self.prepared['static']['surface_response']@self.state.polarization[0].real.flatten()
            error = torch.linalg.vector_norm(expected-self.state.dc_surface_charge)
            scale = torch.linalg.vector_norm(self.state.dc_surface_charge)
            if not bool(error <= 512*torch.finfo(scale.dtype).eps*scale):
                raise ValueError('Recovered DC charge does not match the bound dielectric state')
            response = StaticVolumeFieldResponse(self.prepared['operators'][0].system,exterior,interiors,bindings,
                self.state.polarization[0].real.split(counts),self.state.dc_surface_charge,
                testing_order=self.prepared['dc_testing_order'])
            self._dc = response
        result = self._dc.fields(points,region_id)
        self.last_query_receipt = {'harmonic':0,'frequency_hz':0.,'potential_units':'V','electric_units':'V/m',
            'static_response':self._dc.receipt,'physical_qualification':False}
        return result

    def query_dc_magnetic(self,points,region_id):
        """Return recovered H0/B0 without repeating the magnetic interface solve."""
        self._assert_current()
        state,static = self.state,self.prepared['static']
        if state.dc_dual_state is not None:
            fields=self.query_dc_dual(points,region_id)
            if fields['magnetic_flux_density_T'] is None:
                raise NotImplementedError('Conductive magnetic DC has a flux pole; use query_dc_dual for H and pole residue')
            return fields['magnetic_A_per_m'],fields['magnetic_flux_density_T']
        if state.dc_magnetic_volume is not None:
            h,b,diagnostics=state.dc_magnetic_volume.fields(points,region_id)
            self.last_query_receipt=dict(harmonic=0,frequency_hz=0.,magnetic_units='A/m',flux_density_units='T',
                magnetic_response=state.dc_magnetic_volume.receipt,**diagnostics)
            return h,b
        if state.dc_magnetic_density is None:
            raise NotImplementedError('This harmonic state has no recovered DC magnetic response')
        if self._dc_magnetic is None:
            from .static_magnetic import MagnetostaticInterfaceResponse
            closure = static['current_closure']
            mu_e,mu_i = static['magnetic_permeabilities']
            self._dc_magnetic = MagnetostaticInterfaceResponse.from_state(static['floating_operator'],
                self.prepared['bindings'][0],state.total_volume_current[0].real.split(closure.counts),
                mu_e,mu_i,state.dc_magnetic_density,state.dc_magnetic_source_mean,state.receipt['dc_current']['magnetic'])
        result = self._dc_magnetic.fields(points,region_id)
        self.last_query_receipt = {'harmonic':0,'frequency_hz':0.,'magnetic_units':'A/m','flux_density_units':'T',
            'magnetic_response':self._dc_magnetic.receipt,'physical_qualification':False}
        return result

    def query_dc_dual(self,points,region_id):
        self._assert_current()
        if self.state.dc_dual_state is None:
            raise ValueError('Dual DC queries require a recovered dual conductive state')
        from .harmonic_dual_bridge import query_dual
        fields,self._dual=query_dual(self.prepared,self.state,points,region_id,self._dual)
        self.last_query_receipt={'harmonic':0,'dual_conductive':True,'physical_qualification':False}
        return fields

    def _query_current_dc(self,points,region_id):
        from .static_floating import FloatingPotentialFields
        from .fields import classify_queries
        static,state = self.prepared['static'],self.state
        operator = static['floating_operator']
        region = operator.system.region(region_id)
        if not isinstance(points,torch.Tensor) or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('DC queries require real CUDA [Q,3] coordinates')
        operator._real(points,points.shape,'DC query points')
        mixed = bool(state.dc_current_regions)
        closure = static['current_closure']
        if self._dc is None:
            if mixed:
                from .static_coupling import StaticVolumeFieldResponse
                epsilon,interiors = self.prepared['dc_permittivities']
                samples = tuple(epsilon if value is None else value for value in interiors)
                counts = tuple(len(b.volume.cells) for b in closure.dielectric_bindings)
                self._dc = StaticVolumeFieldResponse(operator.system,epsilon,samples,closure.dielectric_bindings,
                    state.polarization[0,closure.dielectric_indices].real.split(counts),state.dc_surface_charge,
                    testing_order=self.prepared['dc_testing_order'])
            else:
                epsilon = points.new_tensor(self.prepared['dc_permittivities'][0])
                self._dc = FloatingPotentialFields(operator,state.dc_surface_charge,state.dc_potential_offsets,
                    epsilon,points.new_zeros(len(operator.system.bodies)),copy.deepcopy(state.receipt['dc_current']['floating']))
        if mixed and region not in state.dc_current_regions:
            result = self._dc.fields(points,region)
        elif region == 0:
            result = self._dc.fields(points)
        else:
            inside,_ = classify_queries(operator.system.bodies[region-1].mesh,points)
            if not bool(inside.all()):
                raise ValueError('DC query geometry does not match requested region')
            if mixed:
                index = state.dc_current_regions.index(region)
                block = closure.current.blocks[index]
                offset = state.dc_potential_offsets[index]
            else:
                index = next(i for i,b in enumerate(self.prepared['bindings'][0]) if b.volume.region_id == region)
                block = closure.blocks[index]
                offset = state.dc_potential_offsets[region-1]
            from .harmonic_current import face_potential_fields
            result=face_potential_fields(block,state.dc_face_potentials[index],offset,points)
        self.last_query_receipt = {'harmonic':0,'frequency_hz':0.,'potential_units':'V','electric_units':'V/m',
            'static_response':copy.deepcopy(state.receipt['dc_current']),
            'dc_magnetic_fields_supported':state.dc_magnetic_density is not None or state.dc_magnetic_volume is not None,'physical_qualification':False}
        return result
