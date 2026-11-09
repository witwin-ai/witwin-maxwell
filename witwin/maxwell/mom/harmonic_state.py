"""Recovery of physical harmonic unknowns from the internal dielectric HB solve."""
from dataclasses import dataclass, field as dataclass_field, fields, is_dataclass
import copy
import torch
from .harmonic_balance import HarmonicBalanceSolution


def preparation_signature(prepared):
    """Bind numerical parameters, tensor versions and exact operator/volume ownership."""
    entries = []
    def visit(value,path):
        if isinstance(value,torch.Tensor):
            entries.append((path,id(value),value._version))
        elif value is None or type(value) in (bool,int,float,complex,str):
            entries.append((path,type(value).__name__,value))
        elif isinstance(value,dict):
            for key,item in value.items():
                if not (isinstance(key,str) and key.startswith('_')):
                    visit(item,path+(key,))
        elif isinstance(value,(tuple,list)):
            for index,item in enumerate(value):
                visit(item,path+(index,))
    def visit_physics(value,path):
        if is_dataclass(value) and not isinstance(value,type):
            entries.append((path,type(value)))
            for item in fields(value):
                visit_physics(getattr(value,item.name),path+(item.name,))
        elif isinstance(value,(tuple,list)):
            for index,item in enumerate(value):
                visit_physics(item,path+(index,))
        else:
            visit(value,path)
    visit(prepared,())
    static = prepared.get('static',{})
    if 'magnetic_volume' in static:
        for binding in static['magnetic_volume']['bindings']:
            binding.assert_current()
            entries.append(('magnetic_volume_binding',id(binding),id(binding.volume),id(binding.surface_system)))
    if 'tensor_current_closure' in static:
        closure=static['tensor_current_closure'];closure.assert_current()
        operator=static['tensor_floating_operator']
        if closure.operator is not operator:
            raise ValueError('Mixed tensor current closure and static operator differ')
        entries.extend((('tensor_current_closure',id(closure),closure._state()),
                        ('tensor_current_blocks',id(closure.current),closure.current._state()),
                        ('tensor_floating_operator',id(operator),operator._state())))
        entries.extend(('tensor_floating_kernel',id(k),k._state())
                       for k in (operator.exterior,*operator.interiors) if k is not None)
        entries.extend(('tensor_current_block',id(b),b._identity,b._face_versions) for b in closure.current.blocks)
    if 'tensor_operator' in static:
        operator=static['tensor_operator']
        operator.assert_current()
        entries.append(('tensor_static_operator',id(operator),operator._state()))
        entries.extend(('tensor_static_kernel',id(k),k._state())
                       for k in (operator.exterior,*operator.interiors))
    if 'current_closure' in static:
        closure,floating = static['current_closure'],static['floating_operator']
        entries.append(('current_closure',id(closure),closure._state()))
        from .harmonic_mixed import MixedHarmonicClosure
        if isinstance(closure,MixedHarmonicClosure):
            entries.append(('mixed_operator',id(closure.operator),closure.operator._state()))
            entries.append(('selected_current',id(closure.current),closure.current._state()))
            closure = closure.current
        from .harmonic_dual_current import DualHarmonicCurrentClosure
        if isinstance(closure,DualHarmonicCurrentClosure):
            entries.append(('dual_current',id(closure.current),closure.current._state()))
            closure=closure.current
        entries.extend(('current_block',id(block),id(block.volume),id(block.gradients),block.gradients._version,
                        id(block.nodal_volume),block.nodal_volume._version,id(block.faces),block.faces._version,
                        id(block.face_dofs),block.face_dofs._version) for block in closure.blocks)
        entries.append(('floating_operator',id(floating),floating._state_versions()))
    for index,operator in enumerate(prepared['operators']):
        if prepared.get('response_backend') == 'compressed':
            operator.refresh()
            entries.append(('operator',id(operator),id(operator.system)))
        else:
            entries.append(('operator',id(operator),id(operator.system),id(operator.matrix),operator.matrix._version))
        if hasattr(operator,'validate_tensor_state'):
            operator.validate_tensor_state()
            entries.append(('tensor_decoder',operator._tensor_state()))
        # Decoding and extension must retain the physics used to build the maps.
        for local,handle in enumerate((operator,*operator.body_handles)):
            path = ('operator_physics',index,local)
            entries.append((path,id(handle)))
            for name in ('exterior','integration','incident_carrier'):
                visit_physics(getattr(handle,name),path+(name,))
            entries.append((path+('kernel',),id(handle.kernel)))
            visit(handle.kernel.receipt(),path+('kernel_receipt',))
            if local:
                entries.append((path+('basis',),id(handle.basis)))
                visit_physics(handle.interior,path+('interior',))
    if prepared.get('response_backend') == 'compressed':
        for index,part in enumerate(prepared['harmonics']):
            for name in ('electric_action','magnetic_action','background_bank'):
                action=part[name]
                action.assert_current()
                entries.append(('compressed_action',index,name,id(action)))
    for group in prepared['bindings']:
        entries.extend(('binding',id(binding),id(binding.volume),id(binding.surface_system)) for binding in group)
    return tuple(entries)


@dataclass(frozen=True)
class HarmonicCoupledState:
    frequencies: torch.Tensor
    cell_electric: torch.Tensor
    polarization: torch.Tensor
    dissipative_current: torch.Tensor
    total_volume_current: torch.Tensor
    surface_coefficients: tuple
    dc_surface_charge: torch.Tensor
    receipt: dict
    preparation_identity: int
    dc_face_potentials: tuple = ()
    dc_potential_offsets: torch.Tensor | None = None
    dc_magnetic_density: torch.Tensor | None = None
    dc_magnetic_source_mean: torch.Tensor | None = None
    dc_current_regions: tuple = ()
    dc_dual_state: dict | None = None
    dc_dual_live: tuple | None = None
    induced_surface_coefficients: tuple = ()
    dc_dual_binding: dict | None = None
    dc_tensor_response: object | None = None
    dc_magnetic_volume: object | None = None
    _tensor_versions: tuple = dataclass_field(init=False,repr=False)

    def __post_init__(self):
        object.__setattr__(self,'_tensor_versions',self._versions())

    def _versions(self):
        extra = self.dc_face_potentials+(() if self.dc_potential_offsets is None else (self.dc_potential_offsets,))
        extra += tuple(value for value in (self.dc_magnetic_density,self.dc_magnetic_source_mean) if value is not None)
        def nested(value):
            if isinstance(value,torch.Tensor): return (id(value),value._version)
            if isinstance(value,dict): return tuple((k,nested(v)) for k,v in value.items())
            if isinstance(value,(tuple,list)): return tuple(nested(v) for v in value)
            return value
        live = None if self.dc_dual_live is None else (nested(self.dc_dual_live[0]),
            id(self.dc_dual_live[1]),self.dc_dual_live[1]._state())
        tensor_dc=None if self.dc_tensor_response is None else (id(self.dc_tensor_response),self.dc_tensor_response._state())
        magnetic=None if self.dc_magnetic_volume is None else (id(self.dc_magnetic_volume),self.dc_magnetic_volume._state())
        return (magnetic,tensor_dc,live,self.receipt['pump_scale'],self.preparation_identity,self.dc_current_regions,nested(self.dc_dual_state),nested(self.dc_dual_binding))+tuple((id(value),value._version) for value in (
            self.frequencies,self.cell_electric,self.polarization,self.dissipative_current,
            self.total_volume_current,*self.surface_coefficients,*self.induced_surface_coefficients,self.dc_surface_charge,*extra))

    def assert_current(self):
        if self.dc_tensor_response is not None:self.dc_tensor_response.assert_current()
        if self.dc_magnetic_volume is not None:self.dc_magnetic_volume.assert_current()
        if self._versions() != self._tensor_versions:
            raise RuntimeError('Recovered harmonic state was mutated')


def recover_harmonic_state(model,prepared,solution):
    """Recover lifted AC RWG coefficients and DC potential-layer density.

    DC potential density is not an RWG current. Positive-frequency volume
    current includes polarization current -i*n*omega*P exactly once. The
    incident background is scaled by the actual continuation pump value.
    """
    model._assert_current()
    from .harmonic_mixed import MixedHarmonicClosure
    mixed_dc = isinstance(model.dc_current_closure,MixedHarmonicClosure)
    from .static_tensor_floating import StaticTensorFloatingOperator
    tensor_current=mixed_dc and isinstance(model.dc_current_closure.operator,StaticTensorFloatingOperator)
    if prepared.get('_model_identity') != id(model) or preparation_signature(prepared) != prepared.get('_signature'):
        raise RuntimeError('HB preparation changed or belongs to another model')
    for operator,bindings in zip(prepared['operators'],prepared['bindings']):
        operator.system.assert_current()
        for binding in bindings:
            binding.assert_current()
    if not isinstance(solution,HarmonicBalanceSolution) or not solution.receipt.get('converged',False):
        raise ValueError('Recovery requires a converged harmonic-balance solution')
    field = solution.electric
    pump = solution.receipt['pump_scale']
    residual = model._pack(model.residual(field,pump_scale=pump))
    residual_norm = float(torch.linalg.vector_norm(residual.detach()))
    if not torch.isfinite(residual).all() or residual_norm > solution.receipt['threshold_V_per_m_l2']:
        raise ValueError('HB solution no longer satisfies the converged equations')
    sources = model.sources(field)
    retained = len(field)
    polarization = sources.polarization[:retained]
    dissipative = sources.dissipative_current[:retained]
    current = sources.polarization_current(model.frequency)[:retained]+dissipative
    current_dc = model.dc_current_closure is not None
    if not current_dc and bool((dissipative[0] != 0).any()):
        raise NotImplementedError('Nonzero mean dissipative current requires the DC current formulation')
    from .harmonic_dual_current import DualHarmonicCurrentClosure
    dual_dc=isinstance(model.dc_current_closure,DualHarmonicCurrentClosure)
    dual_state=dual_live=dual_binding=None
    tensor_response=None
    face_potentials,offsets,dc_receipt = (),None,None
    magnetic_density,magnetic_mean = None,None
    current_regions = ()
    if dual_dc:
        from .harmonic_dual_bridge import recover_dual,dual_ohmic_current
        closure=model.dc_current_closure
        dual_state,dc_charge,dual_live,dual_binding=recover_dual(closure,prepared,field,model.tpa_scale,
            electric_source=dissipative[0].real,threshold=solution.receipt['threshold_V_per_m_l2'])
        ohmic=dual_ohmic_current(closure,field[0].real)
        current=torch.cat(((current[0]+ohmic)[None],current[1:]))
        dc_receipt={'dual':(dual_state['state'] if dual_binding is not None else dual_state)['receipt'],
                    'physical_qualification':False}
    elif tensor_current:
        closure=model.dc_current_closure
        if prepared['static'].get('tensor_current_closure') is not closure:
            raise ValueError('Tensor current closure differs from the owning preparation')
        dc_states=closure.solve(field[1:],model.tpa_scale)
        face_potentials=tuple(s.potential.clone() for s in dc_states)
        current_regions=tuple(b.volume.region_id for b in closure.current_bindings)
        from .harmonic_current import surface_face_potentials
        boundary={b.volume.region_id:surface_face_potentials(block,b,potential)
            for block,b,potential in zip(closure.current.blocks,closure.current_bindings,face_potentials)}
        values=polarization[0,closure.dielectric_indices].real
        packed=torch.cat((values.flatten(),*(boundary[r] for r in closure.operator.current_regions)))
        raw_offsets=closure.maps['offset_response']@packed
        offsets=torch.stack([raw_offsets[closure.operator.current_regions.index(r)] for r in current_regions])
        tensor_response=closure.operator.response_from_traces(field.real.new_zeros(3),
            closure.maps['surface_response']@packed,closure.maps['conormal_response']@packed,raw_offsets,
            boundary_potentials=boundary,bindings=closure.dielectric_bindings,
            polarizations=values.split(tuple(len(b.volume.cells) for b in closure.dielectric_bindings)))
        ohmic=torch.zeros_like(field[0].real).index_copy(0,closure.current_indices,
            closure.current.ohmic_current(field[0,closure.current_indices].real))
        current=torch.cat(((current[0]+ohmic)[None],current[1:]))
        dc_charge=field.real.new_empty(0)
        dc_receipt=dict(current=[s.receipt for s in dc_states],floating=tensor_response.receipt,
            current_regions=current_regions,face_potential_order='selected_current_binding_order_then_unique_face_order',
            offset_order='selected_current_binding_order',potential_basis='crouzeix_raviart_face_mean',
            dc_total_current_includes_ohmic=True,dc_magnetic_fields_supported=False,physical_qualification=False)
    elif current_dc:
        closure = model.dc_current_closure
        if prepared['static'].get('current_closure') is not closure:
            raise ValueError('Recovered current closure must belong to the exact HB preparation')
        dc_states = closure.solve(field[1:],model.tpa_scale)
        face_potentials = tuple(state.potential.clone() for state in dc_states)
        from .harmonic_current import surface_face_potentials
        if mixed_dc:
            selected = closure.current
            current_regions = tuple(b.volume.region_id for b in closure.current_bindings)
            boundary = {b.volume.region_id:surface_face_potentials(block,b,potential)
                        for block,b,potential in zip(selected.blocks,closure.current_bindings,face_potentials)}
            packed = torch.cat((polarization[0,closure.dielectric_indices].real.flatten(),
                                *(boundary[r] for r in closure.operator.current_regions)))
            dc_charge = closure.maps['surface_response']@packed
            raw_offsets = closure.maps['offset_response']@packed
            offsets = torch.stack([raw_offsets[closure.operator.current_regions.index(r)] for r in current_regions])
            ohmic = torch.zeros_like(field[0].real).index_copy(0,closure.current_indices,
                selected.ohmic_current(field[0,closure.current_indices].real))
            floating_receipt = {'formulation':'mixed_dielectric_floating_current_dc_v1',
                                'charge_constraint':'separately_neutral_current_bodies','physical_qualification':False}
        else:
            selected = closure
            boundary = [None]*len(prepared['operators'][0].system.bodies)
            for block,binding,potential in zip(closure.blocks,prepared['bindings'][0],face_potentials):
                boundary[binding.volume.region_id-1] = surface_face_potentials(block,binding,potential)
            floating = prepared['static']['floating_operator'].solve_face_means(torch.cat(boundary),
                epsilon_exterior=prepared['dc_permittivities'][0])
            dc_charge,offsets = floating.density,floating.offsets.clone()
            floating_receipt = floating.receipt
            ohmic = closure.ohmic_current(field[0].real)
        current = torch.cat(((current[0]+ohmic)[None],current[1:]))
        from .static_magnetic import MagnetostaticInterfaceResponse
        mu_e,mu_i = prepared['static']['magnetic_permeabilities']
        magnetic = MagnetostaticInterfaceResponse(prepared['static']['floating_operator'],prepared['bindings'][0],
            current[0].real.split(closure.counts),mu_e,mu_i)
        magnetic_density,magnetic_mean = magnetic.density.clone(),magnetic.source_mean.clone()
        dc_receipt = {'current':[state.receipt for state in dc_states],'floating':floating_receipt,
                      'face_potential_order':'volume_binding_order_then_unique_face_order',
                      'potential_basis':'crouzeix_raviart_face_mean',
                      'offset_order':'selected_current_binding_order' if mixed_dc else 'surface_body_order',
                      'current_regions':current_regions,
                      'dc_total_current_includes_ohmic':True,'dc_magnetic_fields_supported':True,
                      'magnetic':magnetic.receipt}
        from .magnetostatic import cell_current_continuity
        diagnostic_current = current[0,closure.current_indices].real if mixed_dc else current[0].real
        diagnostics = [cell_current_continuity(block.volume,value) for block,value in
                       zip(selected.blocks,diagnostic_current.split(selected.counts))]
        dc_receipt['current_continuity'] = [
            {'normal_jump_rms_A_per_m2':float(item['area_weighted_jump_rms_A_per_m2'].detach()),
             'maximum_face_flux_defect_A':float(item['face_flux_defect_A'].abs().max().detach()),
             'distributional_continuity_certified':False} for item in diagnostics]
    elif 'tensor_operator' in prepared['static']:
        static=prepared['static'];operator=static['tensor_operator']
        bindings=prepared['bindings'][0]
        counts=tuple(len(b.volume.cells) for b in bindings)
        values=polarization[0].real
        tensor_response=operator.response_from_traces(values.new_zeros(3),
            static['potential_response']@values.flatten(),
            static['conormal_response']@values.flatten(),
            bindings=bindings,polarizations=values.split(counts))
        dc_charge=values.new_empty(0)
    else:
        dc_charge = prepared['static']['surface_response']@polarization[0].real.flatten()
    induced = tuple(part['electric_action'].surface_response(current[n+1].flatten())
                    if prepared.get('response_backend') == 'compressed' else part['surface_response']@current[n+1].flatten()
                    for n,part in enumerate(prepared['harmonics']))
    surface = tuple(pump*part['background_surface']+value
                    for part,value in zip(prepared['harmonics'],induced))
    receipt = {'formulation':'coupled_sie_vie_hb_dielectric_state_v1',
        'pump_scale':pump,'retained_harmonics':list(range(retained)),
        'ac_surface_representation':'analytic_incident_plus_rwg_scattered_v1',
        'ac_surface_layout':'all_body_J_scattered_then_all_body_M_scattered',
        'dc_surface_representation':'potential_single_layer_density_V_per_m',
        'volume_layout':'binding_cell_xyz','polarization_current_included_once':True,
        'verified_cell_residual_V_per_m_l2':residual_norm,
        'solver':copy.deepcopy(solution.receipt),'physical_qualification':False,
        'harmonic_truncation_converged':False}
    if tensor_response is not None:
        receipt.update(formulation='coupled_sie_vie_hb_tensor_spd_dc_state_v1',
            dc_surface_representation='regional_potential_exterior_conormal_v1',
            dc_static_response=copy.deepcopy(tensor_response.receipt))
    frequencies = torch.arange(retained,device=field.device,dtype=field.real.dtype)*model.frequency
    if current_dc:
        receipt.update(formulation='coupled_sie_vie_hb_current_face_state_v1',dc_current=dc_receipt,
            dc_surface_representation='equivalent_exterior_potential_single_layer_density_V_per_m')
    if dual_dc:
        receipt.update(formulation='coupled_sie_vie_hb_dual_current_state_v1',
            dc_surface_representation='dual_electric_magnetic_potential_layers_v1')
    if mixed_dc:
        receipt.update(formulation='coupled_sie_vie_hb_mixed_face_state_v1',
            dc_surface_representation='mixed_dielectric_current_potential_layer_density_V_per_m')
    if tensor_current:
        receipt.update(formulation='coupled_sie_vie_hb_tensor_current_dc_state_v1',
                       dc_surface_representation='regional_potential_exterior_conormal_v1')
    magnetic_volume=None
    if 'magnetic_volume' in prepared['static']:
        from .static_magnetic_volume import MagneticVolumeContrastResponse
        config=prepared['static']['magnetic_volume']
        selected=prepared['bindings'][0]
        parts=current[0].real.split(tuple(len(b.volume.cells) for b in selected))
        currents={b.volume.region_id:j for b,j in zip(selected,parts)}
        magnetic_volume=MagneticVolumeContrastResponse(prepared['operators'][0].system,config['bindings'],
            tuple(currents.get(b.volume.region_id,field.real.new_zeros(len(b.volume.cells),3)) for b in config['bindings']),
            config['exterior'],config['interiors'],pair_batch=config['pair_batch'],max_matrix_bytes=config['max_matrix_bytes'])
        receipt['dc_current'].update(dc_magnetic_fields_supported=True,magnetic=copy.deepcopy(magnetic_volume.receipt))
    return HarmonicCoupledState(frequencies,field.clone(),polarization.clone(),dissipative.clone(),
        current.clone(),tuple(value.clone() for value in surface),dc_charge.clone(),receipt,id(prepared),
        face_potentials,offsets,magnetic_density,magnetic_mean,current_regions,dual_state,dual_live,
        tuple(value.clone() for value in induced) if prepared.get('response_backend') == 'compressed' else (),
        dc_dual_binding=dual_binding,dc_tensor_response=tensor_response,dc_magnetic_volume=magnetic_volume)
