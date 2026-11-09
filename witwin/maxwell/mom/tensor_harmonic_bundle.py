"""Self-contained mixed harmonic preparation and state bundle.

The outer owner retains an independent digest. Restore rebuilds decode geometry
and exact retained maps, never surface/DC matrices or a solved harmonic state.
"""
import copy
import torch
from .volume_h_snapshot import _digest
from .harmonic_snapshot import tensors
from .tensor_harmonic_snapshot import (snapshot_mixed_tensor_harmonic_state,
    restore_mixed_tensor_harmonic_state)
from .static_tensor_snapshot import snapshot_static_tensor_preparation,restore_static_tensor_preparation
from .runtime import MoMConfig,SystemDecodeHandle,_numerical_identity
from .operator import IntegrationConfig
from .fields import FieldQueryConfig
from .linalg import StoppingRule
from .dyadic_green import DyadicIntegrationConfig
from .tensor_medium import sampled_medium_payload,restore_tensor_medium
from .physics import Medium,restore_scalar_medium
from .mesh import SurfaceMesh,RWGBasis,SurfaceSystem
from .volume_mesh import TetrahedralVolumeMesh,bind_volume_region
from .carrier import IncidentCarrier,CarrierWave
from .harmonic_state import preparation_signature
from .harmonic_balance import DenseHarmonicBalance

VERSION='dense-mixed-tensor-harmonic-bundle-v1'
CURRENT_VERSION='dense-mixed-tensor-current-harmonic-bundle-v1'
MAGNETIC_VERSION='dense-mixed-tensor-current-magnetic-harmonic-bundle-v1'
COMPRESSED_VERSION='compressed-mixed-tensor-harmonic-bundle-v1'
COMPRESSED_CURRENT_VERSION='compressed-mixed-tensor-current-harmonic-bundle-v1'
COMPRESSED_MAGNETIC_VERSION='compressed-mixed-tensor-current-magnetic-harmonic-bundle-v1'


def _sources(sources):
    """Retain original CW parameters rather than invert a sampled phasor."""
    return [dict(direction=tuple(s.direction),polarization=tuple(s.polarization),
        waveform=None if s.source_time is None else dict(frequency=s.source_time.frequency,
            amplitude=s.source_time.amplitude,phase=s.source_time.phase)) for s in sources]


def _restore_sources(records):
    from ..sources import PlaneWave,CW
    if type(records) is not list:raise ValueError('Invalid mixed compressed sources')
    result=[]
    for row in records:
        if type(row) is not dict or set(row)!={'direction','polarization','waveform'}:
            raise ValueError('Invalid mixed compressed source record')
        wave=CarrierWave(row['direction'],row['polarization'],(1.,0.))
        raw=row['waveform']
        if raw is not None and (type(raw) is not dict or set(raw)!={'frequency','amplitude','phase'}):
            raise ValueError('Invalid mixed compressed waveform')
        source=PlaneWave(direction=wave.direction,polarization=wave.polarization,
                         source_time=None if raw is None else CW(**raw))
        # CarrierWave validated these already normalized vectors. Repeating
        # normalization can change their last bits and the carrier identity.
        object.__setattr__(source,'direction',wave.direction)
        object.__setattr__(source,'polarization',wave.polarization)
        result.append(source)
    return tuple(result)


def snapshot_mixed_tensor_harmonic_bundle(state,prepared):
    """Store exact preparation and state with a separately retained owner hash."""
    closure=prepared.get('static',{}).get('tensor_current_closure')
    magnetic='magnetic_volume' in prepared.get('static',{})
    compressed=prepared.get('response_backend')=='compressed'
    version=((COMPRESSED_MAGNETIC_VERSION if magnetic else COMPRESSED_CURRENT_VERSION if closure is not None else COMPRESSED_VERSION)
             if compressed else MAGNETIC_VERSION if magnetic else CURRENT_VERSION if closure is not None else VERSION)
    state_payload,state_binding=snapshot_mixed_tensor_harmonic_state(state,prepared)
    records=[]
    for operator in prepared['operators']:
        config=operator._public_config
        if config.operator_backend not in (('hmatrix','h2') if compressed else ('dense',)) or config.periodic is not None:
            raise NotImplementedError('Mixed harmonic bundle requires matching free-space dense or H/H2 operators')
        state_record=dict(sources=_sources(operator.sources)) if compressed else dict(matrix=operator.matrix)
        records.append(dict(numerical=_numerical_identity(config),**state_record,
            metadata=operator.metadata,exterior=sampled_medium_payload(operator.exterior),
            bodies=[dict(structure_index=b.structure_index,name=b.name,vertices=b.mesh.vertices,
                         faces=b.mesh.faces,medium=sampled_medium_payload(b.interior)) for b in operator.system.bodies],
            carrier=operator.incident_carrier.payload(),carrier_identity=operator.incident_carrier.identity))
    key='tensor_floating_operator' if closure is not None else 'tensor_operator'
    static,static_binding=snapshot_static_tensor_preparation(prepared['static'][key])
    maps={k:v for k,v in prepared['static'].items() if k not in (key,'tensor_current_closure')}
    if magnetic:
        config=maps['magnetic_volume']
        maps['magnetic_volume']={k:v for k,v in config.items() if k!='bindings'}
        maps['magnetic_volume']['volumes']=[dict(region=b.volume.region_id,vertices=b.volume.vertices,cells=b.volume.cells) for b in config['bindings']]
    if closure is not None:
        from .harmonic_current_snapshot import encode_current_closure
        maps['current_closure']=encode_current_closure(closure.current)
    retained={key:prepared[key] for key in ('response_backend','harmonics','fundamental_frequency',
                                          'nonlinear_coefficients','receipt')}
    if compressed:
        from .compressed_harmonic_snapshot import encode
        retained['harmonics']=tuple(encode(part,sources=op.sources) for part,op in zip(prepared['harmonics'],prepared['operators']))
    payload=tensors(dict(version=version,operators=records,
        volumes=[dict(region=b.volume.region_id,vertices=b.volume.vertices,cells=b.volume.cells)
                 for b in prepared['bindings'][0]],
        static_operator=static,static_binding=static_binding,
        prepared=retained,
        static_maps=maps,
        state=state_payload,state_binding=state_binding),'cpu')
    return payload,dict(version=version,sha256=_digest(payload))


def _config(value):
    keys={'subdivisions','dtype','riesz_storage','integration','stopping','query_config'}
    if type(value) is dict and 'dyadic_integration' in value:keys|={'dyadic_integration','dyadic_cache_max_bytes'}
    if type(value) is not dict or set(value)!=keys:raise ValueError('Invalid mixed dense numerical policy')
    dtype={'torch.complex128':torch.complex128,'torch.complex64':torch.complex64}.get(value['dtype'])
    config=MoMConfig(subdivisions=value['subdivisions'],dtype=dtype,riesz_storage=value['riesz_storage'],
        integration=IntegrationConfig(**value['integration']),stopping=StoppingRule(**value['stopping']),
        query_config=FieldQueryConfig(**value['query_config']),
        dyadic_integration=DyadicIntegrationConfig(**value['dyadic_integration']) if 'dyadic_integration' in value else None,
        dyadic_cache_max_bytes=value.get('dyadic_cache_max_bytes',256*1024**2))
    if _numerical_identity(config)!=value:raise ValueError('Mixed numerical policy is not canonical')
    return config


def _medium(value,device):
    if type(value) is not dict:raise ValueError('Invalid mixed sampled medium')
    if value.get('medium_kind')=='homogeneous_passive_tensor':return restore_tensor_medium(value,device=device)
    return restore_scalar_medium(value)


def restore_mixed_tensor_harmonic_bundle(payload,*,expected_binding,device,max_retained_bytes=512*1024**2):
    """Restore without the original preparation; return model, preparation, state.

    The budget bounds stored tensor bytes, not transient geometry or validation
    workspace. Dense operators restore as decode-only handles; compressed
    operators rebind exact physical factors and original numerical sources.
    Neither path assembles an electromagnetic matrix or solves an equation.
    """
    token=payload.get('version') if type(payload) is dict else None
    compressed=token in (COMPRESSED_VERSION,COMPRESSED_CURRENT_VERSION,COMPRESSED_MAGNETIC_VERSION)
    magnetic=token in (MAGNETIC_VERSION,COMPRESSED_MAGNETIC_VERSION)
    current_kind=magnetic or token in (CURRENT_VERSION,COMPRESSED_CURRENT_VERSION)
    version=((COMPRESSED_MAGNETIC_VERSION if magnetic else COMPRESSED_CURRENT_VERSION if current_kind else COMPRESSED_VERSION)
             if compressed else MAGNETIC_VERSION if magnetic else CURRENT_VERSION if current_kind else VERSION)
    keys={'version','operators','volumes','static_operator','static_binding','prepared','static_maps','state','state_binding'}
    if (type(payload) is not dict or set(payload)!=keys or payload['version']!=version
            or type(expected_binding) is not dict or expected_binding!={'version':version,'sha256':_digest(payload)}):
        raise ValueError('Mixed harmonic bundle differs from its owning binding')
    device=torch.device(device)
    if device.type!='cuda':raise ValueError('Mixed harmonic bundle requires CUDA')
    if type(max_retained_bytes) is not int or max_retained_bytes<1:
        raise ValueError('Mixed bundle storage budget must be a positive integer')
    def tensor_bytes(value):
        if isinstance(value,torch.Tensor):
            if value.device.type!='cpu' or not bool(torch.isfinite(value).all()):
                raise ValueError('Mixed bundle requires finite host snapshot tensors')
            return value.numel()*value.element_size()
        if isinstance(value,dict):return sum(tensor_bytes(v) for v in value.values())
        if isinstance(value,(tuple,list)):return sum(tensor_bytes(v) for v in value)
        return 0
    if tensor_bytes(payload)>max_retained_bytes:raise ValueError('Mixed bundle exceeds retained storage budget')
    records=payload['operators']
    if type(records) is not list or not records:raise ValueError('Mixed bundle needs harmonic operators')
    operators=[];geometry=None
    for index,record in enumerate(records):
        if type(record) is not dict or set(record)!={'numerical','sources' if compressed else 'matrix','metadata','exterior','bodies','carrier','carrier_identity'}:
            raise ValueError('Invalid mixed harmonic operator record')
        if compressed:
            from .physical_surface_snapshot import _restore_config
            config=_restore_config(record['numerical'])
            if config.periodic is not None:raise ValueError('Mixed compressed bundle requires free space')
        else:config=_config(record['numerical'])
        exterior=_medium(record['exterior'],device)
        if not isinstance(exterior,Medium):raise ValueError('Mixed bundle requires scalar exterior')
        bodies=record['bodies']
        if type(bodies) is not list or not bodies:raise ValueError('Mixed bundle needs surface bodies')
        declaration=[]
        for body in bodies:
            if type(body) is not dict or set(body)!={'structure_index','name','vertices','faces','medium'}:
                raise ValueError('Invalid mixed body record')
            mesh=SurfaceMesh(body['vertices'].to(device),body['faces'].to(device))
            declaration.append((body['structure_index'],body['name'],RWGBasis(mesh),_medium(body['medium'],device)))
        system=SurfaceSystem.create(declaration)
        shape=(2*system.total_edges,)*2;matrix=None if compressed else record['matrix']
        if not compressed and (type(matrix) is not torch.Tensor or matrix.shape!=shape or matrix.dtype!=config.dtype):
            raise ValueError('Invalid stored mixed surface matrix')
        fingerprint=_digest([{k:v for k,v in b.items() if k!='medium'} for b in bodies])
        if geometry is None:geometry=fingerprint
        if fingerprint!=geometry or any(b.interior.frequency!=exterior.frequency for b in system.bodies):
            raise ValueError('Mixed harmonic body geometry or regional frequency differs')
        if system.dtype!=(torch.float64 if config.dtype==torch.complex128 else torch.float32):
            raise ValueError('Mixed surface geometry precision differs')
        carrier=record['carrier']
        if type(carrier) is not dict or set(carrier)!={'version','phasor_convention','frequency','epsilon_r','mu_r','k','eta','coordinate_units','electric_amplitude_units','waves'}:
            raise ValueError('Invalid mixed carrier record')
        restored=IncidentCarrier(**{k:carrier[k] for k in ('frequency','epsilon_r','mu_r','k','eta')},
                                 waves=tuple(CarrierWave(**w) for w in carrier['waves']))
        restored.validate_medium(exterior)
        if restored.payload()!=carrier or restored.identity!=record['carrier_identity']:
            raise ValueError('Mixed carrier identity differs')
        if compressed:
            from .hierarchical_runtime import _physical_provider
            from types import SimpleNamespace
            sources=_restore_sources(record['sources'])
            IncidentCarrier.restore(carrier,expected_identity=restored.identity,sources=sources,exterior=exterior)
            operators.append(SimpleNamespace(system=system,exterior=exterior,sources=sources,
                provider=_physical_provider(system,exterior,config)))
            continue
        operator=SystemDecodeHandle(system,exterior,(),config,copy.deepcopy(record['metadata']))
        operator.incident_carrier=restored
        for handle in operator.body_handles:handle.incident_carrier=restored
        operator.matrix=matrix.to(device).clone()
        operator.shape=shape;operator.dtype=config.dtype;operator.device=operator.matrix.device
        operators.append(operator)
    volumes=payload['volumes']
    if type(volumes) is not list or not volumes:raise ValueError('Mixed bundle needs volumes')
    for v in volumes:
        if type(v) is not dict or set(v)!={'region','vertices','cells'}:raise ValueError('Invalid mixed volume record')
    volumes=tuple(TetrahedralVolumeMesh(v['vertices'].to(device),v['cells'].to(device),region_id=v['region']) for v in volumes)
    if len({v.region_id for v in volumes})!=len(volumes):raise ValueError('Duplicate mixed volume region')
    raw=payload['prepared']
    if type(raw) is not dict or set(raw)!={'response_backend','harmonics','fundamental_frequency','nonlinear_coefficients','receipt'}:
        raise ValueError('Invalid mixed preparation maps')
    if raw['response_backend']!=('compressed' if compressed else 'dense') or len(raw['harmonics'])!=len(operators):
        raise ValueError('Mixed stored harmonic axis differs')
    if compressed:
        from .compressed_harmonic_snapshot import restore
        from .runtime import _same_metadata
        parts=tuple(restore(part,op.provider,volumes,sources=op.sources)
                    for part,op in zip(raw['harmonics'],operators))
        operators=[p['electric_action'].inverse.operator for p in parts]
        for op,record in zip(operators,records):
            if (not _same_metadata(_numerical_identity(op._public_config),record['numerical'])
                    or not _same_metadata(op.metadata,record['metadata'])
                    or op.incident_carrier.identity!=record['carrier_identity']):
                raise ValueError('Mixed compressed operator binding differs')
        bindings=tuple(p['bindings'] for p in parts)
        prepared=tensors({k:v for k,v in raw.items() if k!='harmonics'},device)
        prepared['harmonics']=parts
    else:
        bindings=tuple(tuple(bind_volume_region(v,op.system) for v in volumes) for op in operators)
        prepared=tensors(raw,device)
    static_operator=restore_static_tensor_preparation(payload['static_operator'],operators[0].system,
        expected_binding=payload['static_binding'],max_matrix_bytes=max_retained_bytes)
    prepared.update(operators=tuple(operators),bindings=bindings,static=tensors(payload['static_maps'],device))
    magnetic_config=prepared['static'].pop('magnetic_volume',None)
    if magnetic!=(magnetic_config is not None):raise ValueError('Mixed magnetic bundle inventory differs')
    closure=None
    if current_kind:
        closure=_restore_mixed_current(prepared['static'],static_operator,bindings[0])
        prepared['static'].update(tensor_current_closure=closure,tensor_floating_operator=static_operator)
    else:
        from .static_tensor_interface import StaticTensorInterfaceOperator
        if type(static_operator) is not StaticTensorInterfaceOperator:
            raise ValueError('SPD bundle requires a dielectric static preparation')
        prepared['static']['tensor_operator']=static_operator
    if magnetic:
        config=magnetic_config
        if type(config) is not dict or set(config)!={'volumes','exterior','interiors','pair_batch','max_matrix_bytes'}:
            raise ValueError('Invalid mixed magnetic preparation')
        rows=config.pop('volumes')
        if type(rows) is not list or len(rows)!=len(operators[0].system.bodies):
            raise ValueError('Mixed magnetic volume inventory differs')
        selected={v.region_id:v for v in volumes};magnetic_bindings=[]
        for region,row in enumerate(rows,1):
            if type(row) is not dict or set(row)!={'region','vertices','cells'} or row['region']!=region:
                raise ValueError('Mixed magnetic volumes require region order')
            if region in selected:
                volume=selected[region]
                if not torch.equal(row['vertices'],volume.vertices) or not torch.equal(row['cells'],volume.cells):
                    raise ValueError('Mixed magnetic current volume differs')
            else:volume=TetrahedralVolumeMesh(row['vertices'],row['cells'],region_id=region)
            magnetic_bindings.append(bind_volume_region(volume,operators[0].system))
        config['bindings']=tuple(magnetic_bindings)
        prepared['static']['magnetic_volume']=config
    frequency=prepared['fundamental_frequency'];parts=prepared['harmonics'];cells=sum(len(v.cells) for v in volumes)
    if (prepared['response_backend']!=('compressed' if compressed else 'dense') or len(parts)!=len(operators)
            or any(op.exterior.frequency!=(n+1)*frequency for n,op in enumerate(operators))):
        raise ValueError('Mixed stored harmonic axis differs')
    background=torch.stack([parts[0]['background_electric'].new_zeros(cells,3)]+[p['background_electric'] for p in parts])
    if compressed:
        from .action_harmonic_balance import ActionHarmonicBalance
        dc={} if closure is not None else dict(dc_polarization_response=prepared['static']['electric_response'].to(background.dtype),
            dc_current_response=background.new_zeros(3*cells,3*cells))
        model=ActionHarmonicBalance(background,[p['electric_action'] for p in parts],parts[0]['cell_volumes'],frequency,
            **dc,**prepared['nonlinear_coefficients'],dc_current_closure=closure)
    else:
        current=torch.stack([parts[0]['electric_response'].new_zeros(3*cells,3*cells)]+[p['electric_response'] for p in parts])
        polarization=torch.stack([torch.zeros_like(current[0]) if closure is not None else prepared['static']['electric_response'].to(current.dtype)]+
            [-1j*op.exterior.omega*p['electric_response'] for op,p in zip(operators,parts)])
        model=DenseHarmonicBalance(background,polarization,current,parts[0]['cell_volumes'],frequency,
                                  **prepared['nonlinear_coefficients'],dc_current_closure=closure)
    prepared['_model_identity']=id(model);prepared['_signature']=preparation_signature(prepared)
    # This also verifies all retained preparation arrays against the state owner.
    state=restore_mixed_tensor_harmonic_state(payload['state'],prepared,expected_binding=payload['state_binding'])
    return model,prepared,state


def _restore_mixed_current(static,operator,bindings):
    """Rebind current cells and exact retained maps without interface preparation."""
    from .static_tensor_floating import StaticTensorFloatingOperator
    from .harmonic_mixed import MixedHarmonicClosure
    from .harmonic_current_snapshot import restore_current_closure
    if type(operator) is not StaticTensorFloatingOperator or set(static)!={'maps','current_closure'}:
        raise ValueError('Current bundle requires floating preparation and current maps')
    closure=object.__new__(MixedHarmonicClosure)
    closure.formulation='mixed_tensor_dielectric_face_current_balance'
    closure.operator=operator;closure.bindings=tuple(bindings)
    regions=tuple(b.volume.region_id for b in bindings)
    if not set(operator.current_regions)<=set(regions):
        raise ValueError('Each floating conductor requires an owning current volume')
    closure.current_bindings=tuple(b for b in bindings if b.volume.region_id in operator.current_regions)
    closure.dielectric_bindings=tuple(b for b in bindings if b.volume.region_id not in operator.current_regions)
    closure.current=restore_current_closure(static.pop('current_closure'),closure.current_bindings)
    closure.volumes=tuple(b.volume for b in bindings)
    closure.counts=tuple(len(v.cells) for v in closure.volumes)
    sample=operator.centers;current_indices=[];dielectric_indices=[];offset=0
    for b,count in zip(bindings,closure.counts):
        indices=torch.arange(offset,offset+count,device=sample.device)
        (current_indices if b.volume.region_id in operator.current_regions else dielectric_indices).append(indices)
        offset+=count
    closure.current_indices=torch.cat(current_indices)
    closure.dielectric_indices=torch.cat(dielectric_indices) if dielectric_indices else torch.empty(0,device=sample.device,dtype=torch.int64)
    maps=static['maps']
    if type(maps) is not dict or set(maps)!={'electric_response','surface_response','conormal_response','offset_response','receipt'}:
        raise ValueError('Invalid stored floating response maps')
    cells=len(closure.dielectric_indices);faces=len(sample)
    columns=3*cells+sum(operator.face_counts[r-1] for r in operator.current_regions)
    for name,rows in (('electric_response',3*cells),('surface_response',faces),('conormal_response',faces),
                      ('offset_response',len(operator.current_regions))):
        value=maps[name]
        if (type(value) is not torch.Tensor or value.shape!=(rows,columns) or value.dtype!=sample.dtype
                or value.device!=sample.device or not bool(torch.isfinite(value).all())):
            raise ValueError('Invalid stored floating response map shape or precision')
    required=(3*cells+2*faces+len(operator.current_regions))*columns*sample.element_size()
    if maps['receipt']!=dict(retained_response_bytes=required,physical_qualification=False,
            material_geometry_gradients_retained=False,quadrature_convergence_certified=False):
        raise ValueError('Invalid stored floating response map receipt')
    closure.maps=maps;closure._versions=closure._state();closure.assert_current()
    return closure
