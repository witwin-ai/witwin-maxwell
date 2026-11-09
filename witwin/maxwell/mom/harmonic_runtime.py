"""Experimental harmonic preparation through the shared public simulation path."""
import copy
import torch
from dataclasses import replace,asdict
from ..media import Material
from .physics import validate_scene_structure
from .runtime import _compile_basis,_signature,_validate_surface_queries,SystemOperator
from .mesh import SurfaceSystem
from .harmonic_material import compile_harmonic_material,material_signature
from .harmonic_preparation import prepare_harmonic_balance
from .harmonic_state import recover_harmonic_state
from .harmonic_fields import HarmonicFieldDecoder


def _harmonic_operator(system,exterior,sources,config):
    if config.operator_backend=='dense':return SystemOperator(system,exterior,sources,config)
    from .hierarchical_runtime import HierarchicalSystemOperator,HierarchicalH2SystemOperator
    cls=HierarchicalH2SystemOperator if config.operator_backend=='h2' else HierarchicalSystemOperator
    return cls(system,exterior,sources,config)


def _response_settings(config):
    settings=config.harmonic_balance
    source=dict(max_order=settings.source_max_order,max_inner_order=settings.source_max_inner_order)
    if config.operator_backend=='dense':
        return dict(stopping=config.stopping,**source,tensor_face_rule=settings.tensor_face_rule,
                    tensor_kernel_batch=settings.tensor_kernel_batch)
    from .surface_volume_hmatrix import SurfaceVolumeHConfig
    from .interior_volume_hmatrix import InteriorVolumeHConfig
    from .volume_hmatrix import VolumeHConfig
    h=config.hierarchical
    shared={k:getattr(h,k) for k in ('leaf_size','max_rank','tolerance','admissibility','row_batch','retained_budget_bytes')}
    return {'compressed_settings':dict(inverse_rule=config.stopping,source_settings=source,
        tensor_face_rule=settings.tensor_face_rule,tensor_kernel_batch=settings.tensor_kernel_batch,
        source_config=SurfaceVolumeHConfig(**shared),field_config=InteriorVolumeHConfig(**shared),
        volume_config=VolumeHConfig(**shared),volume_h2_config=config.h2,
        original_relative_tolerance=(config.h2 if config.operator_backend=='h2' else h).original_sample_tolerance)}



def _validate_dense_response_policy(part,config,*,operator,bindings):
    """Bind source policy and tensor receipt coverage to restored regions."""
    from .tensor_medium import TensorMedium
    hb=config.harmonic_balance
    receipt=part.get('receipt',{})
    source=receipt.get('source_matrix',{})
    matrix=source.get('matrix',{})
    if (source.get('max_outer_order')!=hb.source_max_order
            or source.get('max_inner_order')!=hb.source_max_inner_order
            or source.get('tensor_face_rule','triangle')!=hb.tensor_face_rule
            or source.get('tensor_kernel_batch',128)!=hb.tensor_kernel_batch
            or matrix.get('tensor_kernel_batch',128)!=hb.tensor_kernel_batch):
        raise ValueError('Dense harmonic source quadrature policy mismatch')
    regions=[binding.volume.region_id for binding in bindings]
    tensor_regions={r for r in regions if isinstance(operator.system.bodies[r-1].interior,TensorMedium)}
    bodies=matrix.get('bodies',())
    if [body.get('region_id') for body in bodies]!=regions:
        raise ValueError('Dense harmonic source region coverage mismatch')
    records=[]
    for body in bodies:
        block=body.get('tensor_source_block')
        if body['region_id'] in tensor_regions:
            if (not isinstance(block,dict) or block.get('region_id')!=body['region_id']
                    or block.get('tensor_kernel_batch',128)!=hb.tensor_kernel_batch
                    or not block.get('tensor_volume_evaluations')):
                raise ValueError('Dense harmonic tensor source receipt coverage mismatch')
            records.extend(row['receipt'] for row in block['tensor_volume_evaluations'])
        elif block is not None:
            raise ValueError('Dense harmonic scalar region owns a tensor source receipt')
    groups={}
    for row in receipt.get('tensor_field_evaluations',()):
        groups.setdefault(row['order'],[]).append(row['region_id'])
        records.append(row['volume'])
    if ((tensor_regions and not groups) or any(set(group)!=tensor_regions or len(group)!=len(tensor_regions)
                                               for group in groups.values())):
        raise ValueError('Dense harmonic tensor field receipt coverage mismatch')
    if any(row.get('face_rule')!=hb.tensor_face_rule or row.get('kernel_batch')!=hb.tensor_kernel_batch
           for row in records):
        raise ValueError('Dense harmonic tensor evaluation policy mismatch')



def _validate_response_policy(part,config):
    """Bind public numerical policy to restored component configurations."""
    from .volume_h2 import VolumeNestedH2Action
    from .volume_hmatrix import VolumeHierarchicalAction
    options=_response_settings(config)['compressed_settings']
    _validate_tensor_response_policy(part,config)
    inverse=part['electric_action'].inverse
    if inverse.rule!=config.stopping or inverse.refinement_steps!=2:
        raise ValueError('Compressed harmonic inverse policy mismatch')
    for action in (part['electric_action'],part['magnetic_action']):
        if any(a.config!=options['source_config'] for a in action.sources):
            raise ValueError('Compressed harmonic source configuration mismatch')
        if any(a.config!=options['field_config'] for a in action.fields):
            raise ValueError('Compressed harmonic field configuration mismatch')
        for direct in action.direct:
            if config.operator_backend=='h2':
                valid=(type(direct) is VolumeNestedH2Action and direct.config==config.h2
                    and direct.receipt['source_h']['configuration']==asdict(options['volume_config']))
            else:valid=type(direct) is VolumeHierarchicalAction and direct.config==options['volume_config']
            if not valid:raise ValueError('Compressed harmonic volume configuration mismatch')
    logs=part['receipt']['sources']
    if len(logs)!=len(part['bindings']) or any(
            log.get('max_outer_order')!=config.harmonic_balance.source_max_order
            or log.get('max_inner_order')!=config.harmonic_balance.source_max_inner_order for log in logs):
        raise ValueError('Compressed harmonic source budget mismatch')

    relative=options['original_relative_tolerance'];query=config.query_config
    checks=part['receipt'].get('original_checks',())
    if len(checks)!=2 or any(log.get('relative_dual_tolerance')!=relative
            or log.get('absolute_dual_tolerance')!=1e-12
            or log.get('field_relative_tolerance')!=query.relative_tolerance
            or log.get('field_absolute_electric_tolerance')!=query.absolute_tolerance for log in checks):
        raise ValueError('Compressed harmonic original-check policy mismatch')
    bank=part['background_bank'].receipt
    if bank.get('original_relative_dual_tolerance')!=relative or bank.get('original_absolute_dual_tolerance')!=1e-12:
        raise ValueError('Compressed harmonic background residual policy mismatch')
    for log in bank.get('field_queries',()):
        if log.get('region_id') in bank.get('tensor_regions',()):continue
        if (log.get('relative_tolerance')!=query.relative_tolerance
                or log.get('absolute_tolerance_V_per_m')!=query.absolute_tolerance
                or log.get('final_order',0)>query.max_order):
            raise ValueError('Compressed harmonic background field policy mismatch')
    for step in part['receipt'].get('field_checks',()):
        if step['order']>query.max_order or any(log.get('relative_tolerance')!=query.relative_tolerance
                or log.get('absolute_electric_tolerance')!=query.absolute_tolerance for log in step['channels']):
            raise ValueError('Compressed harmonic field-check policy mismatch')


def _validate_tensor_background_policy(log,medium,exterior,query):
    from .tensor_medium import TensorMedium,scalar_medium_if_isotropic
    from ..constants import EPSILON_0,MU_0
    import math
    scalar=scalar_medium_if_isotropic(medium) is not None and not isinstance(exterior,TensorMedium)
    key='absolute_tolerance_V_per_m' if scalar else 'absolute_tolerance_E_V_per_m'
    schema=(log.get('estimator')=='complete_lifted_field_successive_order_difference' if scalar else
            log.get('representation')=='tensor_complete_lifted_interior_field')
    if (not schema or log.get('relative_tolerance')!=query.relative_tolerance
            or log.get(key)!=query.absolute_tolerance or log.get('final_order',0)>query.max_order):
        raise ValueError('Compressed harmonic tensor background field policy mismatch')
    # Match the actual tensor field integrator's impedance convention.
    if not scalar and log.get('absolute_tolerance_H_A_per_m')!=query.absolute_tolerance/math.sqrt(MU_0/EPSILON_0):
        raise ValueError('Compressed harmonic tensor magnetic field policy mismatch')


def _validate_tensor_response_policy(part,config):
    """Bind tensor volume and background policies to public configuration."""
    from .tensor_medium import TensorMedium
    inverse=part['electric_action'].inverse
    system=inverse.operator.provider.system
    all_regions=[b.region_id for b in system.bodies if isinstance(b.interior,TensorMedium)]
    selected=[b.volume.region_id for b in part['bindings'] if b.volume.region_id in all_regions]
    hb=config.harmonic_balance
    receipt=part['receipt'];bank=part['background_bank'].receipt
    if all_regions and (bank.get('tensor_regions')!=all_regions
            or bank.get('tensor_rhs_cache_max_bytes_per_handle')!=0):
        raise ValueError('Compressed harmonic tensor background policy mismatch')
    if not all_regions and ('tensor_regions' in bank or 'tensor_rhs_cache_max_bytes_per_handle' in bank):
        raise ValueError('Scalar harmonic background cannot own tensor policy')
    query=config.query_config
    for log in bank.get('field_queries',()):
        if log.get('region_id') not in all_regions:continue
        _validate_tensor_background_policy(log,system.bodies[log['region_id']-1].interior,
            inverse.operator.provider.exterior,query)
    expected=dict(face_rule=hb.tensor_face_rule,kernel_batch=hb.tensor_kernel_batch,
        radial_order_policy='follow_complete_field_order',regions=selected,rhs_cache_max_bytes_per_handle=0)
    if selected and receipt.get('tensor_volume_policy')!=expected:
        raise ValueError('Compressed harmonic tensor volume policy mismatch')
    if not selected and 'tensor_volume_policy' in receipt:
        raise ValueError('Scalar harmonic volumes cannot own a tensor volume policy')
    for action in (part['electric_action'],part['magnetic_action']):
        for source,direct,log in zip(action.sources,action.direct,receipt['sources']):
            if source.provider.binding.volume.region_id not in selected:continue
            policy=source.provider.tensor_policy()
            if (policy['face_rule']!=hb.tensor_face_rule or policy['kernel_batch']!=hb.tensor_kernel_batch
                    or log.get('tensor_source_policy')!=policy):
                raise ValueError('Compressed harmonic tensor source policy mismatch')
            provider=direct.provider
            if (provider.face_rule!=hb.tensor_face_rule or provider.kernel_batch!=hb.tensor_kernel_batch
                    or provider.radial_order!=provider.order):
                raise ValueError('Compressed harmonic tensor direct policy mismatch')


def validate_initial_harmonic_electric(value,config):
    """Own a numerical branch seed at the first pump value, without an AD graph."""
    if value is None:
        return None
    settings = config.harmonic_balance
    if settings is None:
        raise ValueError('Initial harmonic electric fields require harmonic-balance configuration')
    shape = (settings.harmonics+1,sum(len(v.cells) for v in settings.volumes),3)
    device = settings.volumes[0].vertices.device
    if (not isinstance(value,torch.Tensor) or value.device != device or value.dtype != config.dtype
            or value.shape != shape or not bool(torch.isfinite(value).all()) or bool(value[0].imag.any())):
        raise ValueError('Initial harmonic electric fields require matching finite CUDA [H+1,cell,3] with real DC')
    return value.detach().clone()


class HarmonicData:
    """Actual harmonic unknowns and regional queries for a coupled Result."""
    def __init__(self,prepared,state,config,scene_signature):
        self.state = state
        self.decoder = HarmonicFieldDecoder(prepared,state)
        self.config,self.scene_signature = config,copy.deepcopy(scene_signature)
        self.query_config = config.query_config

    def query_fields(self,points,region_id,field_kind,trace_side=None,*,harmonic,query_config=None):
        if trace_side is not None:
            raise NotImplementedError('Coupled harmonic interface traces are not yet available')
        if field_kind not in ('total','incident','scattered'):
            raise ValueError('field_kind must be total, incident or scattered')
        region = self.decoder.prepared['operators'][0].system.region(region_id)
        if region and field_kind != 'total':
            raise ValueError('Interior harmonic queries require total fields')
        total = self.decoder.query_fields(points,region,harmonic=harmonic,
            query_config=self.query_config if query_config is None else query_config)
        if field_kind == 'total':
            return total
        operator = self.decoder.prepared['operators'][harmonic-1]
        incident = tuple(self.state.receipt['pump_scale']*value
                         for value in operator.incident_carrier.evaluate(points,operator.exterior))
        return incident if field_kind == 'incident' else tuple(a-b for a,b in zip(total,incident))

    def query_dc_dual(self,points,region_id):
        return self.decoder.query_dc_dual(points,region_id)

    def far_field(self,directions,*,harmonic):
        """Outgoing amplitudes of one retained positive harmonic."""
        self.decoder._assert_current()
        if type(harmonic) is not int or not 1 <= harmonic < len(self.state.frequencies):
            raise ValueError('Far fields require a retained positive harmonic; DC is nonradiative')
        from .lifting_fields import far_field_system
        return far_field_system(self.decoder.prepared['operators'][harmonic-1],
                                self.state.surface_coefficients[harmonic-1],directions)

    def snapshot(self):
        if self.state.dc_tensor_response is not None:
            from .tensor_harmonic_runtime import snapshot
            return snapshot(self)
        from .harmonic_snapshot import snapshot
        return snapshot(self)

    def query_dc(self,points,region_id):
        return self.decoder.query_dc(points,region_id)

    def query_dc_magnetic(self,points,region_id):
        return self.decoder.query_dc_magnetic(points,region_id)


    def sample_point_monitors(self,monitors):
        """Batch coincident positions with separately requested DC E and H."""
        if not monitors:
            return {},{}
        from .lifting_fields import classify_system_queries
        operator = self.decoder.prepared['operators'][0]
        positions = tuple(dict.fromkeys(m.position for m in monitors))
        indices = {position:index for index,position in enumerate(positions)}
        points = torch.tensor(positions,device=operator.system.device,dtype=operator.system.dtype)
        regions,owners,distances = classify_system_queries(operator,self.state.surface_coefficients[0],points)
        if bool((owners != 0).any()):
            raise ValueError('Harmonic PointMonitor requires an off-interface position')
        count = len(self.state.frequencies)-1
        electric = self.state.cell_electric.new_empty(count,len(points),3)
        magnetic = torch.empty_like(electric)
        dc_electric = torch.zeros_like(points)
        dc_magnetic = torch.zeros_like(points)
        dc_potential = points.new_zeros(len(points))
        needs_dc = torch.zeros(len(points),device=points.device,dtype=torch.bool)
        needs_dc_h = torch.zeros_like(needs_dc)
        has_dc_h = self.state.dc_magnetic_density is not None or self.state.dc_dual_state is not None or self.state.dc_magnetic_volume is not None
        for monitor in monitors:
            if any(name.lower().startswith('e') for name in monitor.fields):
                needs_dc[indices[monitor.position]] = True
            if has_dc_h and any(name.lower().startswith('h') for name in monitor.fields):
                needs_dc_h[indices[monitor.position]] = True
        logs = {}
        for region in range(len(operator.system.bodies)+1):
            selected = regions == region
            if not bool(selected.any()):
                continue
            for harmonic in range(1,count+1):
                electric[harmonic-1,selected],magnetic[harmonic-1,selected] = self.query_fields(
                    points[selected],region,'total',harmonic=harmonic)
                logs[(region,harmonic)] = copy.deepcopy(self.decoder.last_query_receipt)
            dc_selected = selected & needs_dc
            if bool(dc_selected.any()):
                dc_potential[dc_selected],dc_electric[dc_selected] = self.query_dc(points[dc_selected],region)
                logs[(region,0)] = copy.deepcopy(self.decoder.last_query_receipt)
            magnetic_selected = selected & needs_dc_h
            if bool(magnetic_selected.any()):
                if self.state.dc_dual_state is not None:
                    dc_magnetic[magnetic_selected] = self.query_dc_dual(points[magnetic_selected],region)['magnetic_A_per_m']
                else:
                    dc_magnetic[magnetic_selected],_ = self.query_dc_magnetic(points[magnetic_selected],region)
                logs[(region,0,'H')] = copy.deepcopy(self.decoder.last_query_receipt)
        output = {}
        for monitor in monitors:
            index = indices[monitor.position]
            region = int(regions[index])
            values = dict(zip(('Ex','Ey','Ez','Hx','Hy','Hz'),
                              torch.cat((electric[:,index],magnetic[:,index]),dim=-1).unbind(-1)))
            components = {name.capitalize():values[name.capitalize()] for name in monitor.fields}
            dc_components = {name:dc_electric[index,'xyz'.index(name[1].lower())]
                             for name in components if name.startswith('E')}
            if has_dc_h:
                dc_components.update({name:dc_magnetic[index,'xyz'.index(name[1].lower())]
                                      for name in components if name.startswith('H')})
            output[monitor.name] = {'kind':'point','fields':tuple(components),'components':components,**components,
                'frequencies':tuple(float(f) for f in self.state.frequencies[1:]),
                'position':monitor.position,'region_id':region,'field_kind':'total',
                'coordinate_units':'m','phasor_convention':'exp(-i*n*omega*t); positive_peak',
                'sampling':'physical_position','distance_to_surface_m':float(distances[index]),
                'dc':{'frequency':0.,'components':dc_components,
                      'potential_V':dc_potential[index] if any(name.startswith('E') for name in components) else None,
                      'magnetic_available':has_dc_h},
                'physical_qualification':False}
        return output,{'unique_monitor_positions':len(points),'queries':logs}


class HarmonicCutoffError(RuntimeError):
    def __init__(self,receipt):
        super().__init__('Harmonic cutoff budget exhausted before successive cell estimates passed')
        self.receipt = receipt


class HarmonicRuntime:
    def __init__(self,simulation):
        self.simulation = simulation
        config = simulation.config
        self.config = config
        settings = config.harmonic_balance
        if settings.dc_magnetic_volumes is not None:
            raise NotImplementedError('Explicit DC magnetic volumes require the mixed tensor current runtime')
        if config.operator_backend=='dense' and (config.stopping.method != 'direct' or config.stopping.direct_storage != 'resident'):
            raise NotImplementedError('Harmonic response preparation currently requires resident direct linear solves')
        if simulation.has_trainable_parameters:
            raise NotImplementedError('Public harmonic material/geometry parameter compilation is not differentiable yet')
        structures = validate_scene_structure(simulation.scene,simulation.frequency)
        _validate_surface_queries(simulation.scene)
        self.materials = tuple(compile_harmonic_material(s.material,simulation.frequency,settings.harmonics)
                               for s in structures)
        self.exterior = compile_harmonic_material(Material(),simulation.frequency,settings.harmonics)
        # Resolve singular static channels before expensive AC matrix work.
        from ..constants import EPSILON_0,MU_0
        self.exterior.electric_dc.dielectric_sample(EPSILON_0)
        from .harmonic_dual_bridge import dual_materials
        dual_dc = dual_materials(self.materials)
        if settings.dc_tangent_rule is not None and (not any(
                m.tpa_scale!=0 or m.electric_dc.kind=='conductive_limit' for m in self.materials)):
            raise NotImplementedError('DC tangent policy requires electric current or mixed current closure')
        for material in self.materials:
            current_dc = material.tpa_scale != 0 or material.electric_dc.kind == 'conductive_limit'
            if current_dc:
                if material.electric_dc.conductivity == 0 and 0. in settings.pump_scales:
                    raise NotImplementedError('Pure TPA zero-pump continuation has an unresolved singular DC tangent')
                if material.electric_dc.kind not in ('finite_positive_dielectric','conductive_limit'):
                    raise NotImplementedError('Current DC requires a finite dielectric or conductive electric channel')
                if not dual_dc: material.magnetic_dc.dielectric_sample(MU_0)
            else:
                material.electric_dc.dielectric_sample(EPSILON_0)
        enabled = [(i,s) for i,s in enumerate(simulation.scene.structures) if s.enabled]
        bases = tuple(_compile_basis(simulation.scene,s.geometry,config) for _,s in enabled)
        linear_config = replace(config,harmonic_balance=None)
        operators = tuple(_harmonic_operator(SurfaceSystem.create([
            (index,s.name,basis,material.linear_samples[n])
            for (index,s),basis,material in zip(enabled,bases,self.materials)]),
            self.exterior.linear_samples[n],simulation.scene.sources if n == 0 else [],linear_config)
            for n in range(settings.harmonics))
        self.model,self.prepared = prepare_harmonic_balance(operators,settings.volumes,
            self.materials,self.exterior,query_config=config.query_config,dc_testing_order=settings.dc_testing_order,
            dc_policy=settings.dc_policy(),**_response_settings(config))
        self.signature = self._signature()

    def _signature(self):
        sim = self.simulation
        return (_signature(sim.scene,self.exterior.linear_samples[0],
                           tuple(m.linear_samples[0] for m in self.materials)),
                tuple(material_signature(s.material) for s in sim.scene.structures if s.enabled),sim.frequency)

    def _cutoff_materials(self,count):
        sim=self.simulation
        return tuple(compile_harmonic_material(s.material,sim.frequency,count)
                     for s in sim.scene.structures if s.enabled)

    def _prepare_cutoff_response(self,operators,materials,exterior,config,previous):
        settings=config.harmonic_balance
        return prepare_harmonic_balance(operators,settings.volumes,materials,exterior,
            query_config=config.query_config,dc_testing_order=settings.dc_testing_order,
            reuse_prepared=previous,dc_policy=settings.dc_policy(),**_response_settings(config))

    def _extend_cutoff(self,data,policy):
        from .harmonic_convergence import compare_harmonic_cells
        sim = self.simulation
        count = len(data.state.frequencies)-1
        receipt = {'initial_harmonics':count,'policy':asdict(policy),'checks':[],
            'cell_truncation_estimate_passed':False,'physical_qualification':False,
            'harmonic_truncation_converged':False,'failure_reason':None,
            'scope':'fixed_grid_cell_fields_with_independent_added_harmonic_budget'}
        consecutive = 0
        while count < policy.max_harmonics:
            next_count = count+1
            materials = self._cutoff_materials(next_count)
            exterior = compile_harmonic_material(Material(),sim.frequency,next_count)
            system = data.decoder.prepared['operators'][0].system
            actual_config = replace(self.config,harmonic_balance=replace(
                self.config.harmonic_balance,harmonics=next_count))
            operator = _harmonic_operator(SurfaceSystem.create([(body.structure_index,body.name,body.basis,
                material.linear_samples[-1]) for body,material in zip(system.bodies,materials)]),
                exterior.linear_samples[-1],[],replace(actual_config,harmonic_balance=None))
            model,prepared = self._prepare_cutoff_response((*data.decoder.prepared['operators'],operator),
                materials,exterior,actual_config,data.decoder.prepared)
            initial = torch.cat((data.state.cell_electric,data.state.cell_electric.new_zeros(
                (1,)+data.state.cell_electric.shape[1:])))
            solution = model.solve(initial,pump_scale=data.state.receipt['pump_scale'],
                                   **actual_config.harmonic_balance.solve_settings())
            solution.receipt.update(branch_id=actual_config.harmonic_balance.branch_id,
                branch_selection='warm_start_from_previous_harmonic_count_at_final_pump',initial_harmonic_count=count)
            state = recover_harmonic_state(model,prepared,solution)
            extended = HarmonicData(prepared,state,actual_config,self.signature)
            comparison = compare_harmonic_cells(data,extended,relative_tolerance=policy.relative_tolerance,
                absolute_tolerance=policy.absolute_tolerance,tail_absolute_tolerance=policy.tail_absolute_tolerance)
            consecutive = consecutive+1 if comparison['cell_truncation_estimate_passed'] else 0
            record = {key:value.detach().cpu().tolist() if isinstance(value,torch.Tensor) else value
                      for key,value in comparison.items()}
            record.update(successive_passes=consecutive,solver=copy.deepcopy(solution.receipt))
            receipt['checks'].append(record)
            data,count = extended,next_count
            if consecutive >= policy.successive_passes:
                receipt['cell_truncation_estimate_passed'] = True
                break
        receipt['final_harmonics'] = count
        if not receipt['cell_truncation_estimate_passed']:
            receipt['failure_reason'] = 'harmonic_budget_exhausted'
            if policy.failure == 'raise':
                raise HarmonicCutoffError(receipt)
        return data,receipt

    def run(self, *, initial_harmonic_electric=None):
        from ..result import Result
        sim = self.simulation
        validate_scene_structure(sim.scene,sim.frequency)
        if sim.config is not self.config or self._signature() != self.signature:
            raise ValueError('Scene or harmonic configuration changed after preparation')
        settings = self.config.harmonic_balance
        # Check geometry and response ownership before a possibly long solve.
        from .harmonic_state import preparation_signature
        if preparation_signature(self.prepared) != self.prepared['_signature']:
            raise RuntimeError('Harmonic preparation was mutated')
        for group in self.prepared['bindings']:
            for binding in group:
                binding.assert_current()
        initial = validate_initial_harmonic_electric(initial_harmonic_electric,self.config)
        solutions = self.model.solve_ramp(settings.pump_scales,branch_id=settings.branch_id,
                                         initial=initial,**settings.solve_settings())
        state = recover_harmonic_state(self.model,self.prepared,solutions[-1])
        data = HarmonicData(self.prepared,state,self.config,self.signature)
        cutoff = None
        if settings.cutoff is not None:
            data,cutoff = self._extend_cutoff(data,settings.cutoff)
            state = data.state
        monitors,monitor_logs = data.sample_point_monitors(sim.scene.monitors)
        return Result(method='mom',scene=sim.scene,frequencies=tuple(float(f) for f in state.frequencies),
            solver=self,harmonic_data=data,monitors=monitors,solver_stats={
                'monitor_queries':monitor_logs,**copy.deepcopy(state.receipt['solver']),
                'cutoff':cutoff,'continuation_harmonics':settings.harmonics,
                'initial_harmonic_electric':initial,
                'branch_initialization':'linear_background_at_first_pump' if initial is None else 'explicit_harmonic_cell_field',
                'continuation':[copy.deepcopy(solution.receipt) for solution in solutions]},
            metadata={**sim.metadata,**data.decoder.prepared['receipt'],'native_data':'coupled_sie_vie_hb',
                'fundamental_frequency_hz':sim.frequency,'solver_converged':True,
                'physical_qualification':False,'harmonic_truncation_converged':False,
                'cell_cutoff_estimate_passed':None if cutoff is None else cutoff['cell_truncation_estimate_passed']})
