"""Exact detached harmonic snapshots; restore never assembles or solves."""
import copy
import math
from dataclasses import asdict,replace
import torch
from .runtime import MoMConfig,SystemDecodeHandle,_compile_basis,_signature,_same_metadata
from .operator import IntegrationConfig
from .fields import FieldQueryConfig
from .linalg import StoppingRule
from .harmonic_config import HarmonicBalanceConfig,HarmonicCutoffConfig
from .harmonic_material import compile_harmonic_material,material_signature,cell_nonlinear_coefficients
from .harmonic_state import HarmonicCoupledState,preparation_signature
from .mesh import SurfaceSystem
from .volume_mesh import TetrahedralVolumeMesh,bind_volume_region
from .physics import validate_scene_structure
from ..media import Material

IDENTITY = {'schema_version':2,'payload_kind':'coupled_sie_vie_hb_dielectric',
    'phasor':'exp(-i*n*omega*t); positive_peak; real_DC','volume_basis':'constant_cell_xyz',
    'ac_surface':'analytic_incident_plus_rwg_scattered_v1',
    'dc_surface':'potential_single_layer_density_V_per_m','coordinate_units':'m'}
STATE_ARRAYS = ('frequencies','cell_electric','polarization','dissipative_current','total_volume_current',
                'surface_coefficients','dc_surface_charge')
CURRENT_IDENTITY = {**IDENTITY,'schema_version':5,'payload_kind':'coupled_sie_vie_hb_current',
    'dc_surface':'equivalent_exterior_potential_single_layer_density_V_per_m'}
CURRENT_ARRAYS = ('dc_face_potentials','dc_potential_offsets','dc_magnetic_density','dc_magnetic_source_mean')
MIXED_IDENTITY = {**IDENTITY,'schema_version':6,'payload_kind':'coupled_sie_vie_hb_mixed',
    'dc_surface':'mixed_dielectric_current_potential_layer_density_V_per_m'}
MIXED_ARRAYS = CURRENT_ARRAYS+('dc_current_regions',)

DUAL_IDENTITY={**IDENTITY,'schema_version':14,'payload_kind':'coupled_sie_vie_hb_dual_current',
               'dc_surface':'dual_electric_magnetic_potential_layers_v1'}


def tensors(value,device):
    if isinstance(value,torch.Tensor):
        return value.detach().to(device).clone()
    if isinstance(value,dict):
        return {k:tensors(v,device) for k,v in value.items()}
    if isinstance(value,tuple):
        return tuple(tensors(v,device) for v in value)
    if isinstance(value,list):
        return [tensors(v,device) for v in value]
    return copy.deepcopy(value)


def snapshot(data):
    data.decoder._assert_current()
    prepared,state,config = data.decoder.prepared,data.state,data.config
    if state.dc_tensor_response is not None:
        raise NotImplementedError('Tensor DC harmonic snapshots require the dedicated potential/conormal schema')
    mixed_dc = bool(state.dc_current_regions)
    dual_dc = state.dc_dual_state is not None
    current_dc = dual_dc or state.dc_potential_offsets is not None
    identity = DUAL_IDENTITY if dual_dc else MIXED_IDENTITY if mixed_dc else CURRENT_IDENTITY if current_dc else IDENTITY
    compressed = prepared.get('response_backend') == 'compressed'
    if compressed: identity = {**identity,'schema_version':15,'payload_kind':'compressed_'+identity['payload_kind']}
    saved_prepared = {name:prepared[name] for name in ('static','harmonics','dc_permittivities','dc_testing_order','receipt')}
    if dual_dc:
        from .harmonic_dual_bridge import encode_static
        saved_prepared['static']=encode_static(prepared['static'],prepared['bindings'][0])
    elif current_dc:
        if mixed_dc:
            from .harmonic_mixed_snapshot import encode
        else:
            from .harmonic_current_snapshot import encode
        saved_prepared['static'] = encode(prepared['static'])
    if compressed:
        from .compressed_harmonic_snapshot import encode as encode_compressed
        saved_prepared['harmonics']=tuple(encode_compressed(part,sources=op.sources)
            for part,op in zip(prepared['harmonics'],prepared['operators']))
    settings = config.harmonic_balance
    numerical = {name:getattr(config,name) for name in ('subdivisions','riesz_storage')}
    numerical.update(dtype=str(config.dtype),integration=asdict(config.integration),
        stopping=asdict(config.stopping),query_config=asdict(config.query_config))
    if compressed and config.operator_backend in ('hmatrix','h2'):
        from .runtime import _numerical_identity
        numerical=_numerical_identity(config)
    harmonic_config = settings.numerical_payload()
    return tensors({**identity,'numerical':numerical,
        'harmonic_config':harmonic_config,
        'volumes':[{'vertices':v.vertices,'cells':v.cells,'region_id':v.region_id} for v in settings.volumes],
        'scene_signature':data.scene_signature,
        'operators':[{**({} if compressed else {'matrix':op.matrix}),'metadata':op.metadata,'exterior':asdict(op.exterior),
            'interiors':[asdict(body.interior) for body in op.system.bodies],
            'meshes':[{'vertices':body.mesh.vertices,'faces':body.mesh.faces} for body in op.system.bodies],
            'carrier':op.incident_carrier.payload(),'carrier_identity':op.incident_carrier.identity}
            for op in prepared['operators']],
        'prepared':saved_prepared,
        'state':{**{name:getattr(state,name) for name in STATE_ARRAYS+(('dc_dual_state',)+(('dc_dual_binding',) if state.dc_dual_binding is not None else ()) if dual_dc else MIXED_ARRAYS if mixed_dc else CURRENT_ARRAYS if current_dc else ())},
                 'receipt':state.receipt,**({'induced_surface_coefficients':state.induced_surface_coefficients} if compressed else {})}},'cpu')


def restore(payload,scene):
    if isinstance(payload,dict) and payload.get('schema_version') in (21,22,23):
        from .tensor_harmonic_runtime import restore as restore_tensor
        return restore_tensor(payload,scene)
    compressed = isinstance(payload,dict) and payload.get('schema_version') == 15
    kind = payload.get('payload_kind','') if isinstance(payload,dict) else ''
    if compressed: kind = kind.removeprefix('compressed_')
    mixed_dc = kind == MIXED_IDENTITY['payload_kind']
    dual_dc = kind == DUAL_IDENTITY['payload_kind']
    current_dc = dual_dc or mixed_dc or (kind == CURRENT_IDENTITY['payload_kind'])
    identity = DUAL_IDENTITY if dual_dc else MIXED_IDENTITY if mixed_dc else CURRENT_IDENTITY if current_dc else IDENTITY
    if compressed: identity = {**identity,'schema_version':15,'payload_kind':'compressed_'+identity['payload_kind']}
    expected = {*IDENTITY,'numerical','harmonic_config','volumes','scene_signature','operators','prepared','state'}
    if not isinstance(payload,dict) or set(payload) != expected or not _same_metadata(
            {k:payload[k] for k in identity},identity):
        raise ValueError('Unsupported coupled harmonic snapshot identity or fields')
    if torch.device(scene.device).type != 'cuda':
        raise ValueError('Harmonic restore requires a CUDA Scene')
    numerical = payload['numerical']
    compressed_config=compressed and isinstance(numerical,dict) and numerical.get('operator_backend') in ('hmatrix','h2')
    if not compressed_config and (not isinstance(numerical,dict) or set(numerical) != {'subdivisions','riesz_storage','dtype','integration','stopping','query_config'}):
        raise ValueError('Invalid harmonic numerical configuration')
    dtype = {'torch.complex64':torch.complex64,'torch.complex128':torch.complex128}.get(numerical.get('dtype'))
    volumes = tuple(TetrahedralVolumeMesh(v['vertices'].to(scene.device),v['cells'].to(scene.device),
                    region_id=v['region_id']) for v in payload['volumes'])
    settings = dict(payload['harmonic_config'])
    if 'dc_tangent_rule' in settings:
        if not isinstance(settings['dc_tangent_rule'],dict):raise ValueError('DC tangent snapshot requires explicit policy fields')
        settings['dc_tangent_rule']=StoppingRule(**settings['dc_tangent_rule'])
    if 'tangent_rule' in settings:
        if not isinstance(settings['tangent_rule'],dict):
            raise ValueError('Harmonic tangent snapshot must contain explicit policy fields')
        settings['tangent_rule']=StoppingRule(**settings['tangent_rule'])
    if settings.get('cutoff') is not None:
        if not isinstance(settings['cutoff'],dict):
            raise ValueError('Harmonic cutoff snapshot must contain explicit configuration fields')
        settings['cutoff'] = HarmonicCutoffConfig(**settings['cutoff'])
    hb = HarmonicBalanceConfig(volumes=volumes,**settings)
    if hb.dc_tangent_rule is not None and not current_dc:
        raise ValueError('Stored DC tangent policy does not support this closure')
    if not _same_metadata(payload['harmonic_config'],hb.numerical_payload()):
        raise ValueError('Stored harmonic policy is not canonical')
    config = MoMConfig(subdivisions=numerical['subdivisions'],dtype=dtype,
        integration=IntegrationConfig(**numerical['integration']),stopping=StoppingRule(**numerical['stopping']),
        query_config=FieldQueryConfig(**numerical['query_config']),riesz_storage='compact_rwg_v1' if compressed_config else numerical['riesz_storage'],harmonic_balance=hb)
    if compressed_config:
        from .physical_surface_snapshot import _restore_config
        config=replace(_restore_config(numerical),harmonic_balance=hb)
    raw = payload['state']
    if not isinstance(raw,dict) or set(raw) != {*STATE_ARRAYS,*(('dc_dual_state',)+(('dc_dual_binding',) if hb.dc_tangent_rule is not None else ()) if dual_dc else MIXED_ARRAYS if mixed_dc else CURRENT_ARRAYS if current_dc else ()), 'receipt',*(('induced_surface_coefficients',) if compressed else ())}:
        raise ValueError('Invalid harmonic state fields')
    frequencies = raw['frequencies']
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    if (not isinstance(frequencies,torch.Tensor) or frequencies.shape != (hb.harmonics+1,)
            or frequencies.dtype != real_dtype or not bool(torch.isfinite(frequencies).all())
            or not bool(frequencies[1] > 0)):
        raise ValueError('Invalid harmonic frequency axis')
    frequency = float(frequencies[1])
    if not torch.equal(frequencies,torch.arange(hb.harmonics+1,device=frequencies.device,dtype=real_dtype)*frequency):
        raise ValueError('Harmonic frequency axis does not match integer harmonics')
    structures = validate_scene_structure(scene,frequency)
    from .runtime import _validate_surface_queries
    _validate_surface_queries(scene)
    materials = tuple(compile_harmonic_material(s.material,frequency,hb.harmonics) for s in structures)
    exterior = compile_harmonic_material(Material(),frequency,hb.harmonics)
    signature = (_signature(scene,exterior.linear_samples[0],tuple(m.linear_samples[0] for m in materials)),
                 tuple(material_signature(s.material) for s in scene.structures if s.enabled),frequency)
    if not _same_metadata(payload['scene_signature'],signature):
        raise ValueError('Harmonic snapshot does not match the supplied Scene')
    enabled = [(i,s) for i,s in enumerate(scene.structures) if s.enabled]
    bases = tuple(_compile_basis(scene,s.geometry,config) for _,s in enabled)
    if not isinstance(payload['prepared'],dict) or set(payload['prepared']) != {
            'static','harmonics','dc_permittivities','dc_testing_order','receipt'}:
        raise ValueError('Invalid stored harmonic preparation')
    if (not isinstance(payload['prepared']['harmonics'],(tuple,list))
            or len(payload['prepared']['harmonics'])!=hb.harmonics):
        raise ValueError('Invalid stored harmonic response count')
    records = payload['operators']
    if not isinstance(records,list) or len(records) != hb.harmonics:
        raise ValueError('Invalid harmonic operator count')
    operators,bindings,dynamic = [],[],[]
    from .carrier import IncidentCarrier
    for n,record in enumerate(records):
        expected_record={'metadata','exterior','interiors','meshes','carrier','carrier_identity'}
        if not compressed: expected_record.add('matrix')
        if not isinstance(record,dict) or set(record)!=expected_record:
            raise ValueError('Invalid harmonic operator record')
        system = SurfaceSystem.create([(i,s.name,basis,m.linear_samples[n])
                    for (i,s),basis,m in zip(enabled,bases,materials)])
        if (not _same_metadata(record['exterior'],asdict(exterior.linear_samples[n])) or
                not _same_metadata(record['interiors'],[asdict(b.interior) for b in system.bodies])):
            raise ValueError('Harmonic sampled material mismatch')
        if len(record['meshes']) != len(system.bodies):
            raise ValueError('Harmonic surface body count mismatch')
        for mesh,body in zip(record['meshes'],system.bodies):
            for key,reference in (('vertices',body.mesh.vertices),('faces',body.mesh.faces)):
                value = mesh[key]
                if not isinstance(value,torch.Tensor) or value.dtype != reference.dtype or not torch.equal(value.to(scene.device),reference):
                    raise ValueError('Harmonic surface mesh mismatch')
        sources = scene.sources if n == 0 else []
        if compressed:
            from .compressed_harmonic_snapshot import restore as restore_compressed
            from .nearfield import ScalarSystemBlocks
            part=restore_compressed(payload['prepared']['harmonics'][n],
                ScalarSystemBlocks(system,exterior.linear_samples[n],config.integration),volumes,sources=sources)
            operator=part['electric_action'].inverse.operator
            if compressed_config:
                from .harmonic_runtime import _validate_response_policy
                _validate_response_policy(part,config)
            if compressed_config and operator._public_config!=replace(config,harmonic_balance=None):
                raise ValueError('Compressed harmonic numerical configuration differs from outer policy')
            if not _same_metadata(record['metadata'],operator.metadata):
                raise ValueError('Compressed harmonic operator metadata mismatch')
            dynamic.append(part)
            group=part['bindings']
        else:
            matrix = record['matrix']
            size = 2*system.total_edges
            if (not isinstance(matrix,torch.Tensor) or matrix.dtype != dtype or matrix.shape != (size,size)
                    or not bool(torch.isfinite(matrix).all())):
                raise ValueError('Invalid stored harmonic matrix')
            sources = scene.sources if n == 0 else []
            operator = SystemDecodeHandle(system,exterior.linear_samples[n],sources,config,copy.deepcopy(record['metadata']))
            IncidentCarrier.restore(record['carrier'],expected_identity=record['carrier_identity'],sources=sources,exterior=operator.exterior)
            operator.matrix = matrix.to(scene.device).clone()
            operator.shape,operator.dtype,operator.device = operator.matrix.shape,dtype,operator.matrix.device
            group=tuple(bind_volume_region(volume,system) for volume in volumes)
        IncidentCarrier.restore(record['carrier'],expected_identity=record['carrier_identity'],sources=sources,exterior=operator.exterior)
        operators.append(operator)
        bindings.append(group)
    prepared = tensors({k:v for k,v in payload['prepared'].items() if not (compressed and k=='harmonics')},scene.device)
    if compressed: prepared.update(response_backend='compressed',harmonics=tuple(dynamic))
    prepared.update(operators=tuple(operators),bindings=tuple(bindings))
    from ..constants import EPSILON_0
    if current_dc and not mixed_dc and any((m.electric_dc.kind not in ('finite_positive_dielectric','conductive_limit')
                          or not (m.tpa_scale != 0 or m.electric_dc.kind == 'conductive_limit')) for m in materials):
        raise ValueError('Snapshot material does not define the current DC formulation')
    expected_dc = (exterior.electric_dc.dielectric_sample(EPSILON_0),
                   tuple(None if (m.tpa_scale != 0 or m.electric_dc.kind == 'conductive_limit') else
                         m.electric_dc.dielectric_sample(EPSILON_0) for m in materials) if mixed_dc else
                   None if current_dc else tuple(m.electric_dc.dielectric_sample(EPSILON_0) for m in materials))
    if (not _same_metadata(prepared['dc_permittivities'],expected_dc)
            or prepared['dc_testing_order'] != hb.dc_testing_order):
        raise ValueError('Harmonic DC material or quadrature mismatch')
    count = sum(len(v.cells) for v in volumes)
    for name in ('cell_electric','polarization','dissipative_current','total_volume_current'):
        value = raw[name]
        if (not isinstance(value,torch.Tensor) or value.dtype != dtype or value.shape != (hb.harmonics+1,count,3)
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Invalid harmonic cell state')
    if (len(raw['surface_coefficients']) != hb.harmonics or any(
            not isinstance(value,torch.Tensor) or value.shape != (op.shape[0],) or value.dtype != dtype
            or not bool(torch.isfinite(value).all()) for value,op in zip(raw['surface_coefficients'],operators))):
        raise ValueError('Invalid harmonic surface state')
    charge = raw['dc_surface_charge']
    if (not isinstance(charge,torch.Tensor) or charge.dtype != real_dtype or
            charge.shape != (sum(len(b.mesh.faces) for b in operators[0].system.bodies),) or not bool(torch.isfinite(charge).all())):
        raise ValueError('Invalid harmonic DC charge')
    receipt = raw['receipt']
    if (not isinstance(receipt,dict) or receipt.get('formulation') != (
            'coupled_sie_vie_hb_dual_current_state_v1' if dual_dc else
            'coupled_sie_vie_hb_mixed_face_state_v1' if mixed_dc else
            'coupled_sie_vie_hb_current_face_state_v1' if current_dc else 'coupled_sie_vie_hb_dielectric_state_v1')
            or receipt.get('physical_qualification') is not False or receipt.get('harmonic_truncation_converged') is not False
            or receipt.get('pump_scale') != hb.pump_scales[-1]
            or receipt.get('retained_harmonics') != list(range(hb.harmonics+1))
            or receipt.get('ac_surface_representation') != IDENTITY['ac_surface']
            or receipt.get('dc_surface_representation') != identity['dc_surface']
            or receipt.get('ac_surface_layout') != 'all_body_J_scattered_then_all_body_M_scattered'
            or receipt.get('volume_layout') != 'binding_cell_xyz'
            or receipt.get('polarization_current_included_once') is not True):
        raise ValueError('Invalid harmonic state representation or qualification receipt')
    solver = receipt.get('solver',{})
    threshold = solver.get('threshold_V_per_m_l2')
    if (solver.get('converged') is not True or solver.get('branch_id') != hb.branch_id or solver.get('method')!=hb.method
            or type(threshold) not in (float,int) or not math.isfinite(threshold) or threshold < 0
            or solver.get('physical_qualification') is not False
            or solver.get('harmonic_truncation_converged') is not False):
        raise ValueError('Invalid stored harmonic solver receipt')
    validate_solver_policy(solver,hb.numerical_payload(),harmonics=hb.harmonics,cell_count=count,real_dtype=real_dtype)
    if any(bool(raw[name][0].imag.any()) for name in ('cell_electric','polarization','dissipative_current','total_volume_current')):
        raise ValueError('DC harmonic unknowns must be real')
    def array(value,shape,expected_dtype):
        if (not isinstance(value,torch.Tensor) or value.shape != shape or value.dtype != expected_dtype
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Invalid harmonic response array')
    static = prepared['static']
    if dual_dc:
        from .harmonic_dual_bridge import restore_static
        restore_static(static,bindings[0],materials,hb.dc_testing_order)
        closure=static['current_closure']
        if closure.streamed_rule!=hb.dc_tangent_rule or closure.streamed_column_batch!=hb.dc_tangent_column_batch:
            raise ValueError('Stored dual DC policy differs from public configuration')
        if hb.dc_tangent_rule is not None:
            from .dual_current_receipt import validate_dual_receipts
            validate_dual_receipts(solver.get('dc_current_closure'),rule=hb.dc_tangent_rule,
                column_batch=hb.dc_tangent_column_batch,cell_count=count,dtype=real_dtype)
            recovery=raw['dc_dual_state'].get('state',{}).get('receipt',{}).get('hb_recovery',{})
            if (recovery.get('method')!='outer_source_linear_response'
                    or recovery.get('threshold_V_per_m_l2')!=threshold
                    or recovery.get('nonlinear_root_recomputed') is not False
                    or recovery.get('physical_qualification') is not False):
                raise ValueError('Stored dual recovery differs from owning HB acceptance')
    elif current_dc:
        if mixed_dc:
            from .harmonic_mixed_snapshot import restore as restore_current
        else:
            from .harmonic_current_snapshot import restore as restore_current
        restore_current(static,bindings[0],materials,hb.dc_testing_order)
        closure = static['current_closure']
        selected = closure.current if mixed_dc else closure
        from .harmonic_current import HarmonicCurrentClosure
        if (selected.relative_tolerance != HarmonicCurrentClosure.relative_tolerance_default
                or selected.absolute_tolerance != HarmonicCurrentClosure.absolute_tolerance_default
                or selected.tangent_rule!=hb.dc_tangent_rule
                or selected.tangent_column_batch!=hb.dc_tangent_column_batch):
            raise ValueError('Stored DC closure tolerance disagrees with the public preparation policy')
        if hb.dc_tangent_rule is not None:
            records=solver.get('dc_current_closure')
            if not isinstance(records,(tuple,list)) or len(records)!=len(selected.blocks):
                raise ValueError('Stored DC solver receipt count mismatch')
            for record,block in zip(records,selected.blocks):selected.validate_receipt(record,block)
        if mixed_dc and raw['dc_current_regions'] != tuple(b.volume.region_id for b in closure.current_bindings):
            raise ValueError('Invalid mixed current region order')
        if not isinstance(raw['dc_face_potentials'],(tuple,list)) or len(raw['dc_face_potentials']) != len(selected.blocks):
            raise ValueError('Invalid harmonic face potential count')
        for potential,block in zip(raw['dc_face_potentials'],selected.blocks):
            array(potential,(block.n_dofs,),real_dtype)
        array(raw['dc_potential_offsets'],(len(selected.blocks) if mixed_dc else len(materials),),real_dtype)
        array(raw['dc_magnetic_density'],charge.shape,real_dtype)
        array(raw['dc_magnetic_source_mean'],(len(charge),3),real_dtype)
    else:
        array(static['electric_response'],(3*count,3*count),real_dtype)
        array(static['surface_response'],(len(charge),3*count),real_dtype)
    if len(prepared['harmonics']) != hb.harmonics:
        raise ValueError('Invalid harmonic response count')
    if not compressed:
        from .harmonic_runtime import _validate_dense_response_policy
        for part,operator,group in zip(prepared['harmonics'],operators,bindings):
            _validate_dense_response_policy(part,config,operator=operator,bindings=group)
    expected_points = torch.cat([v.vertices[v.cells].mean(1) for v in volumes])
    expected_volumes = torch.cat([v.volumes for v in volumes])
    array(static['points'],(count,3),real_dtype)
    if not torch.equal(static['points'],expected_points):
        raise ValueError('Stored harmonic static points mismatch bound volume geometry')
    for part,operator in zip(prepared['harmonics'],operators):
        for name,expected in (('points',expected_points),('cell_volumes',expected_volumes)):
            array(part[name],expected.shape,real_dtype)
            if part[name].device != expected.device or not torch.equal(part[name],expected):
                raise ValueError('Stored harmonic '+name+' mismatch bound volume geometry')
        if not compressed:
            for name in ('electric_response','magnetic_response'):
                array(part[name],(3*count,3*count),dtype)
        for name in ('background_electric','background_magnetic'):
            array(part[name],(count,3),dtype)
        if not compressed: array(part['surface_response'],(operator.shape[0],3*count),dtype)
        array(part['background_surface'],(operator.shape[0],),dtype)
    from .harmonic_balance import DenseHarmonicBalance
    background = torch.stack([prepared['harmonics'][0]['background_electric'].new_zeros(count,3)]
        +[part['background_electric'] for part in prepared['harmonics']])
    packed = DenseHarmonicBalance._pack(hb.pump_scales[-1]*background)
    expected_threshold = hb.absolute_tolerance+hb.relative_tolerance*float(torch.linalg.vector_norm(packed))
    if threshold != expected_threshold:
        raise ValueError('Stored harmonic solver threshold disagrees with configured tolerances')
    prepared['_signature'] = preparation_signature(prepared)
    values = tensors(raw,scene.device)
    state = HarmonicCoupledState(**values,preparation_identity=id(prepared))
    # Recompute the local constitutive law and current convention without solving.
    from .nonlinear import instantaneous_harmonic_sources
    source = instantaneous_harmonic_sources(state.cell_electric,
        **cell_nonlinear_coefficients(materials,volumes,state.cell_electric))
    expected_current = source.polarization_current(frequency)[:hb.harmonics+1]+state.dissipative_current
    if current_dc:
        closure = static['current_closure']
        if dual_dc:
            ohmic=closure.current.electric.reshape(-1,3)*state.cell_electric[0].real
        elif mixed_dc:
            ohmic = torch.zeros_like(state.cell_electric[0].real).index_copy(0,closure.current_indices,
                closure.current.ohmic_current(state.cell_electric[0,closure.current_indices].real))
        else:
            ohmic = closure.ohmic_current(state.cell_electric[0].real)
        expected_current = torch.cat(((expected_current[0]+ohmic)[None],expected_current[1:]))
    for actual,reference in ((state.polarization,source.polarization[:hb.harmonics+1]),
            (state.dissipative_current,source.dissipative_current[:hb.harmonics+1]),
            (state.total_volume_current,expected_current)):
        if not torch.equal(actual,reference):
            raise ValueError('Stored harmonic sources violate the material/current convention')
    if current_dc and not dual_dc:
        selected = closure.current if mixed_dc else closure
        ac = state.cell_electric[1:]
        if mixed_dc:
            ac = ac[:,closure.current_indices]
        records = state.receipt.get('dc_current',{}).get('current',[])
        if len(records) != len(selected.blocks):
            raise ValueError('Invalid stored DC solver receipt count')
        for block,field,record in zip(selected.blocks,ac.split(selected.counts,dim=1),records):
            zero = torch.zeros_like(field[:1])
            third = instantaneous_harmonic_sources(torch.cat((zero,field)),
                tpa_scale=1.).dissipative_current[0].real
            current = materials[block.volume.region_id-1].tpa_scale*third
            expected = selected.absolute_tolerance+selected.relative_tolerance*float(
                torch.linalg.vector_norm(block.weak_current(current)[1:]))
            if record.get('threshold_A_l2') != expected:
                raise ValueError('Stored DC solver threshold disagrees with configured tolerances')
    pump = state.receipt['pump_scale']
    if dual_dc:
        from .harmonic_dual_bridge import validate_dual
        predicted=[validate_dual(prepared,state,threshold)]
    elif current_dc:
        if mixed_dc:
            from .harmonic_mixed_snapshot import validate_state
        else:
            from .harmonic_current_snapshot import validate_state
        predicted = [validate_state(static,state,bindings[0],threshold)]
    else:
        predicted = [static['electric_response']@state.polarization[0].real.flatten()]
    for n,part in enumerate(prepared['harmonics']):
        current = state.total_volume_current[n+1].flatten()
        if compressed:
            from .compressed_harmonic_snapshot import validate_surface_state
            if len(state.induced_surface_coefficients)!=hb.harmonics:
                raise ValueError('Invalid compressed harmonic induced surface count')
            induced=state.induced_surface_coefficients[n]
            response,_=validate_surface_state(part,current,induced)
        else:
            response=part['electric_response']@current
            induced=part['surface_response']@current
        predicted.append(pump*part['background_electric'].flatten()+response)
        expected = pump*part['background_surface']+induced
        tolerance = 512*torch.finfo(real_dtype).eps*torch.linalg.vector_norm(expected)
        if not bool(torch.linalg.vector_norm(state.surface_coefficients[n]-expected) <= tolerance):
            raise ValueError('Stored harmonic surface state violates its prepared response')
    if not current_dc:
        expected_charge = static['surface_response']@state.polarization[0].real.flatten()
        if not bool(torch.linalg.vector_norm(state.dc_surface_charge-expected_charge) <=
                    512*torch.finfo(real_dtype).eps*torch.linalg.vector_norm(expected_charge)):
            raise ValueError('Stored harmonic DC charge violates its prepared response')
    error = torch.linalg.vector_norm(torch.stack(predicted).reshape_as(state.cell_electric)-state.cell_electric)
    if not bool(error <= threshold):
        raise ValueError('Stored harmonic field violates its convergence threshold')
    from .harmonic_runtime import HarmonicData
    return HarmonicData(prepared,state,config,signature)


def validate_solver_policy(solver,settings,*,harmonics,cell_count,real_dtype):
    """Validate recorded solver policy without recomputing historical tangents."""
    if not isinstance(solver,dict) or solver.get('method')!=settings['method']:
        raise ValueError('Stored harmonic solver method mismatch')
    if settings['method']=='newton_krylov':
        expected_implicit={'dense':'dense_svd_checked','streamed_krylov':'streamed_dominance_checked_gmres',
            'streamed_inverse_krylov':'streamed_inverse_residual_checked_gmres'}[settings['implicit_solver']]
        if (solver.get('tangent_rule')!=settings['tangent_rule']
                or solver.get('newton_direction')!='matrix_free_gmres'
                or solver.get('implicit_root_solver')!=expected_implicit
                or solver.get('global_hb_jacobian_stored') is not (settings['implicit_solver']=='dense')
                or solver.get('fully_matrix_free') is not False):
            raise ValueError('Stored harmonic tangent solver policy mismatch')
        if settings['implicit_solver']!='dense':
            inverse_bound=settings['implicit_solver']=='streamed_inverse_krylov'
            expected_bound='complete_streamed_inverse_residual' if inverse_bound else 'complete_streamed_diagonal_dominance'
            checks=solver.get('implicit_tangent_checks')
            if not isinstance(checks,list) or len(checks)!=2 or any(
                    not isinstance(check,dict) or check.get('method')!=expected_bound
                    or check.get('admitted') is not True
                    or check.get('column_batch')!=settings['tangent_column_batch']
                    or check.get('columns_checked')!=(2*harmonics+1)*3*cell_count
                    or check.get('underlying_action_error_certified') is not False
                    or check.get('interval_certified') is not False
                    or check.get('physical_qualification') is not False
                    or check.get('fast_work_qualified') is not False for check in checks):
                raise ValueError('Stored streamed harmonic tangent checks mismatch')
            for check in checks:
                lower,upper,cutoff=(check.get(k) for k in ('sigma_min_lower','sigma_max_upper','numerical_singularity_cutoff'))
                if (any(type(v) not in (float,int) or not math.isfinite(v) for v in (lower,upper,cutoff))
                        or not 0<=cutoff<lower<=upper
                        or cutoff!=64*torch.finfo(real_dtype).eps*(2*harmonics+1)*3*cell_count*upper
                        or check.get('inconclusive') is not False
                        or not isinstance(check.get('solve'),dict) or check['solve'].get('converged') is not True):
                    raise ValueError('Invalid stored harmonic tangent numerical bound')
                if inverse_bound:
                    inverse_upper=check.get('inverse_norm_upper')
                    residual_upper=check.get('inverse_residual_norm_upper')
                    if (check.get('complete') is not True or check.get('solve_failure') is not None
                            or check.get('tangent_rule')!=settings['tangent_rule']
                            or any(type(v) not in (float,int) or not math.isfinite(v) for v in (inverse_upper,residual_upper))
                            or not inverse_upper>0 or not 0<=residual_upper<1
                            or lower!=(1-residual_upper)/inverse_upper):
                        raise ValueError('Invalid stored harmonic inverse residual bound')
