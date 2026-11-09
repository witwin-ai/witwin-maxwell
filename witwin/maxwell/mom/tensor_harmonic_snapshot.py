"""Exact mixed harmonic state bound to an independently owned preparation.

This component does not reconstruct the preparation or enable Result storage.
Restore never assembles, samples an AC kernel, or solves interface/HB equations.
"""
import copy
import math
import torch
from .volume_h_snapshot import _digest
from .harmonic_state import HarmonicCoupledState, preparation_signature
from .static_tensor_snapshot import (snapshot_static_tensor_response,
    restore_static_tensor_response, _preparation, _volume)
from .tensor_medium import sampled_medium_payload
from .runtime import _numerical_identity

VERSION='mixed-tensor-harmonic-owned-state-v1'
CURRENT_VERSION='mixed-tensor-current-harmonic-owned-state-v1'
MAGNETIC_VERSION='mixed-tensor-current-magnetic-harmonic-owned-state-v1'
COMPRESSED_VERSION='compressed-mixed-tensor-harmonic-owned-state-v1'
COMPRESSED_CURRENT_VERSION='compressed-mixed-tensor-current-harmonic-owned-state-v1'
COMPRESSED_MAGNETIC_VERSION='compressed-mixed-tensor-current-magnetic-harmonic-owned-state-v1'
CURRENT_ARRAYS=('dc_face_potentials','dc_potential_offsets','dc_current_regions')
ARRAYS=('frequencies','cell_electric','polarization','dissipative_current','total_volume_current',
        'surface_coefficients','dc_surface_charge')


def _version(prepared,current,magnetic):
    if prepared.get('response_backend')=='compressed':
        return COMPRESSED_MAGNETIC_VERSION if magnetic else COMPRESSED_CURRENT_VERSION if current else COMPRESSED_VERSION
    return MAGNETIC_VERSION if magnetic else CURRENT_VERSION if current else VERSION


def _prepared_digest(prepared):
    static=prepared.get('static',{})
    closure=static.get('tensor_current_closure')
    key='tensor_floating_operator' if closure is not None else 'tensor_operator'
    compressed=prepared.get('response_backend')=='compressed'
    if (prepared.get('response_backend') not in ('dense','compressed') or key not in static
            or preparation_signature(prepared)!=prepared.get('_signature')):
        raise ValueError('Mixed harmonic state requires unchanged tensor DC preparation')
    operators=[]
    for operator in prepared['operators']:
        if hasattr(operator,'refresh'):operator.refresh()
        else:
            operator.system.assert_current();operator.validate_tensor_state()
        if compressed:
            from .surface_h_snapshot import encode as encode_surface
            numerical_state=dict(compressed_operator=encode_surface(operator))
        else:numerical_state=dict(matrix=operator.matrix)
        operators.append(dict(**numerical_state,exterior=sampled_medium_payload(operator.exterior),
            numerical=_numerical_identity(operator._public_config),carrier=operator.incident_carrier.payload(),
            bodies=[dict(region=b.region_id,name=b.name,structure_index=b.structure_index,
                         vertices=b.mesh.vertices,faces=b.mesh.faces,medium=sampled_medium_payload(b.interior))
                    for b in operator.system.bodies]))
    harmonics=prepared['harmonics']
    if compressed:
        from .compressed_harmonic_snapshot import encode as encode_harmonic
        def identity(part,operator):
            payload=encode_harmonic(part,sources=operator.sources)
            # Krylov observations evolve during valid applications; they are
            # persisted in snapshots but are not part of preparation identity.
            payload.pop('sha256')
            actions=payload['actions'];actions.pop('sha256')
            inverse=actions['inverse'];inverse.pop('sha256');inverse.pop('diagnostics')
            return payload
        harmonics=tuple(identity(part,op) for part,op in zip(harmonics,prepared['operators']))
    return _digest(dict(operators=operators,bindings=[[_volume(b) for b in group] for group in prepared['bindings']],
        **static_preparation_payload(static),
        harmonics=harmonics,fundamental_frequency=prepared['fundamental_frequency'],
        nonlinear_coefficients=prepared['nonlinear_coefficients'],receipt=prepared['receipt']))


def static_preparation_payload(static):
    """Exact static identity shared by snapshot ownership and cutoff comparison."""
    closure=static.get('tensor_current_closure')
    key='tensor_floating_operator' if closure is not None else 'tensor_operator'
    maps={k:v for k,v in static.items() if k not in (key,'tensor_current_closure')}
    if 'magnetic_volume' in maps:
        config=maps['magnetic_volume']
        maps['magnetic_volume']={k:v for k,v in config.items() if k!='bindings'}
        maps['magnetic_volume']['volumes']=[_volume(b) for b in config['bindings']]
    if closure is not None:
        from .harmonic_current_snapshot import encode_current_closure
        maps['current_closure']=dict(current=encode_current_closure(closure.current),
            current_indices=closure.current_indices,dielectric_indices=closure.dielectric_indices,
            counts=closure.counts,maps=closure.maps,
            current_regions=tuple(b.volume.region_id for b in closure.current_bindings))
    return dict(static_operator=_preparation(static[key]),static_maps=maps)


def snapshot_mixed_tensor_harmonic_state(state,prepared):
    """Detach owned state; the outer owner must retain the returned binding."""
    if not isinstance(state,HarmonicCoupledState) or state.dc_tensor_response is None:
        raise TypeError('Mixed snapshot requires a tensor DC harmonic state')
    magnetic='magnetic_volume' in prepared.get('static',{})
    if magnetic!=(state.dc_magnetic_volume is not None):
        raise ValueError('Mixed magnetic state and preparation differ')
    current='tensor_current_closure' in prepared.get('static',{})
    if magnetic and not current:raise ValueError('Magnetic state requires current preparation')
    compressed=prepared.get('response_backend')=='compressed'
    version=_version(prepared,current,magnetic)
    if ((not current and (state.dc_face_potentials or state.dc_potential_offsets is not None or state.dc_current_regions))
            or state.dc_magnetic_density is not None or state.dc_magnetic_source_mean is not None
            or state.dc_dual_state is not None or state.dc_dual_live is not None
            or state.dc_dual_binding is not None or (state.induced_surface_coefficients and not compressed)):
        raise ValueError('Mixed SPD snapshot cannot discard additional current or compressed state')
    state.assert_current()
    if state.preparation_identity!=id(prepared):
        raise ValueError('Mixed harmonic state belongs to another preparation')
    digest=_prepared_digest(prepared)
    response,binding=snapshot_static_tensor_response(state.dc_tensor_response)
    from .harmonic_snapshot import tensors
    names=ARRAYS+(CURRENT_ARRAYS if current else ())+(('induced_surface_coefficients',) if compressed else ())
    payload=dict(version=version,preparation=digest,
        state=tensors({name:getattr(state,name) for name in names},'cpu'),
        receipt=copy.deepcopy(state.receipt),dc_response=response,dc_binding=binding)
    if magnetic:
        from .static_magnetic_snapshot import snapshot_magnetic_volume_response
        payload['magnetic_response'],payload['magnetic_binding']=snapshot_magnetic_volume_response(state.dc_magnetic_volume)
    owner=dict(version=version,sha256=_digest(payload))
    restore_mixed_tensor_harmonic_state(payload,prepared,expected_binding=owner)
    return payload,owner


def restore_mixed_tensor_harmonic_state(payload,prepared,*,expected_binding):
    """Restore exact arrays against independent owner hash and prepared physics."""
    closure=prepared.get('static',{}).get('tensor_current_closure')
    current_state=closure is not None
    magnetic='magnetic_volume' in prepared.get('static',{})
    if magnetic and not current_state:raise ValueError('Magnetic state requires current preparation')
    compressed=prepared.get('response_backend')=='compressed'
    version=_version(prepared,current_state,magnetic)
    keys={'version','preparation','state','receipt','dc_response','dc_binding'}
    if magnetic:keys.update(('magnetic_response','magnetic_binding'))
    if (type(payload) is not dict or set(payload)!=keys
            or payload['version']!=version or type(expected_binding) is not dict
            or expected_binding!={'version':version,'sha256':_digest(payload)}):
        raise ValueError('Mixed harmonic state differs from its owning binding')
    if payload['preparation']!=_prepared_digest(prepared):
        raise ValueError('Mixed harmonic state preparation differs')
    operators=prepared['operators'];count=len(operators)+1
    bindings=prepared['bindings'][0];counts=tuple(len(b.volume.cells) for b in bindings);cells=sum(counts)
    device=operators[0].device;dtype=operators[0].dtype;real_dtype=operators[0].system.dtype
    raw=payload['state']
    names=ARRAYS+(CURRENT_ARRAYS if current_state else ())+(('induced_surface_coefficients',) if compressed else ())
    if type(raw) is not dict or set(raw)!=set(names):raise ValueError('Invalid mixed harmonic state arrays')
    def array(value,shape,kind):
        if (type(value) is not torch.Tensor or value.device.type!='cpu' or value.dtype!=kind
                or tuple(value.shape)!=shape or not bool(torch.isfinite(value).all())):
            raise ValueError('Invalid exact mixed harmonic array')
        return value.to(device=device).clone()
    values={name:array(raw[name],(count,cells,3),dtype) for name in
            ('cell_electric','polarization','dissipative_current','total_volume_current')}
    values['frequencies']=array(raw['frequencies'],(count,),real_dtype)
    frequency=operators[0].exterior.frequency
    expected=torch.arange(count,device=device,dtype=real_dtype)*frequency
    if not torch.equal(values['frequencies'],expected):raise ValueError('Mixed harmonic frequency axis differs')
    if type(raw['surface_coefficients']) is not tuple or len(raw['surface_coefficients'])!=count-1:
        raise ValueError('Mixed harmonic surface inventory differs')
    values['surface_coefficients']=tuple(array(v,(op.shape[0],),dtype)
                                         for v,op in zip(raw['surface_coefficients'],operators))
    if compressed:
        induced=raw['induced_surface_coefficients']
        if type(induced) is not tuple or len(induced)!=count-1:
            raise ValueError('Mixed harmonic induced surface inventory differs')
        values['induced_surface_coefficients']=tuple(array(v,(op.shape[0],),dtype)
                                                   for v,op in zip(induced,operators))
    values['dc_surface_charge']=array(raw['dc_surface_charge'],(0,),real_dtype)
    for name in ('cell_electric','polarization','dissipative_current','total_volume_current'):
        if bool((values[name][0].imag!=0).any()):raise ValueError('Mixed harmonic DC arrays must be real')
    if not current_state and bool((values['dissipative_current']!=0).any()):
        raise ValueError('SPD mixed preparation cannot own a dissipative current closure')
    orders=torch.arange(count,device=device,dtype=real_dtype)
    current=(-2j*math.pi*frequency*orders[:,None,None])*values['polarization']+values['dissipative_current']
    if current_state:
        ohmic=torch.zeros_like(values['cell_electric'][0].real).index_copy(0,closure.current_indices,
            closure.current.ohmic_current(values['cell_electric'][0,closure.current_indices].real))
        current=torch.cat(((current[0]+ohmic)[None],current[1:]))
    if not torch.equal(current,values['total_volume_current']):
        raise ValueError('Mixed harmonic polarization current partition differs')
    receipt=payload['receipt']
    if (type(receipt) is not dict or receipt.get('formulation')!=('coupled_sie_vie_hb_tensor_current_dc_state_v1' if current_state else 'coupled_sie_vie_hb_tensor_spd_dc_state_v1')
            or receipt.get('dc_surface_representation')!='regional_potential_exterior_conormal_v1'
            or receipt.get('physical_qualification') is not False
            or receipt.get('harmonic_truncation_converged') is not False
            or type(receipt.get('pump_scale')) not in (int,float)
            or not math.isfinite(receipt['pump_scale']) or receipt['pump_scale']<0
            or receipt.get('retained_harmonics')!=list(range(count))):
        raise ValueError('Invalid mixed harmonic state receipt')
    source_bindings=closure.dielectric_bindings if current_state else bindings
    operator=prepared['static']['tensor_floating_operator' if current_state else 'tensor_operator']
    response=restore_static_tensor_response(payload['dc_response'],operator,source_bindings,
                                           expected_binding=payload['dc_binding'])
    if (not current_state and response.offset!=0) or bool((response.field!=0).any()):
        raise ValueError('Mixed harmonic DC response cannot add an external static drive')
    source_p=values['polarization'][0,closure.dielectric_indices].real if current_state else values['polarization'][0].real
    for binding,polarization in zip(source_bindings,source_p.split(tuple(len(b.volume.cells) for b in source_bindings))):
        source=response.sources[binding.volume.region_id-1]
        if source is None or not torch.equal(source.polarization,polarization):
            raise ValueError('Mixed harmonic DC polarization differs from the recovered source')
    from .nonlinear import instantaneous_harmonic_sources
    sources=instantaneous_harmonic_sources(values['cell_electric'],**prepared['nonlinear_coefficients'])
    if (not torch.equal(sources.polarization[:count],values['polarization'])
            or not torch.equal(sources.dissipative_current[:count],values['dissipative_current'])):
        raise ValueError('Mixed harmonic state differs from its nonlinear constitutive source law')
    if receipt.get('dc_static_response')!=response.receipt:
        raise ValueError('Mixed harmonic DC diagnostics differ from stored response')
    dc_field=None
    if current_state:
        dc_field=_restore_current(values,raw,receipt,response,closure,array,prepared['nonlinear_coefficients']['tpa_scale'],magnetic=magnetic)
    _validate_retained_response(values,receipt,prepared,dc_field=dc_field)
    if magnetic:
        from .static_magnetic_snapshot import restore_magnetic_volume_response
        config=prepared['static']['magnetic_volume']
        restored=restore_magnetic_volume_response(payload['magnetic_response'],operators[0].system,config['bindings'],
            expected_binding=payload['magnetic_binding'],max_matrix_bytes=config['max_matrix_bytes'])
        if (restored.pair_batch!=config['pair_batch'] or bool(restored.background.any())
                or any(not torch.equal(k.coefficient,v) for k,v in zip((restored.exterior,*restored.interiors),
                    (config['exterior'],*config['interiors'])))):
            raise ValueError('Mixed magnetic material or policy differs')
        by_region={b.volume.region_id:j for b,j in zip(bindings,values['total_volume_current'][0].real.split(counts))}
        for b,j in zip(restored.bindings,restored.currents):
            expected=by_region.get(b.volume.region_id,torch.zeros_like(j))
            if not torch.equal(j,expected):raise ValueError('Mixed magnetic current partition differs')
        if receipt['dc_current'].get('magnetic')!=restored.receipt:
            raise ValueError('Mixed magnetic diagnostics differ')
        values['dc_magnetic_volume']=restored
    return HarmonicCoupledState(**values,receipt=copy.deepcopy(receipt),preparation_identity=id(prepared),
                                dc_tensor_response=response)


def _restore_current(values,raw,receipt,response,closure,array,tpa_scale,*,magnetic=False):
    """Validate saved current equations and retained trace maps without solving."""
    regions=tuple(b.volume.region_id for b in closure.current_bindings)
    if type(raw['dc_current_regions']) is not tuple or raw['dc_current_regions']!=regions:
        raise ValueError('Mixed current region order differs')
    potentials=raw['dc_face_potentials']
    if type(potentials) is not tuple or len(potentials)!=len(regions):
        raise ValueError('Mixed current face-potential inventory differs')
    dtype=response.field.dtype
    potentials=tuple(array(p,(b.n_dofs,),dtype) for p,b in zip(potentials,closure.current.blocks))
    offsets=array(raw['dc_potential_offsets'],(len(regions),),dtype)
    info=receipt.get('dc_current')
    if (type(info) is not dict or info.get('current_regions')!=regions
            or info.get('dc_total_current_includes_ohmic') is not True
            or info.get('dc_magnetic_fields_supported') is not magnetic
            or info.get('physical_qualification') is not False
            or info.get('potential_basis')!='crouzeix_raviart_face_mean'
            or info.get('offset_order')!='selected_current_binding_order'
            or info.get('face_potential_order')!='selected_current_binding_order_then_unique_face_order'
            or info.get('floating')!=response.receipt
            or type(info.get('current')) is not list or len(info['current'])!=len(regions)):
        raise ValueError('Invalid mixed current receipt')
    threshold=receipt.get('solver',{}).get('threshold_V_per_m_l2')
    if type(threshold) not in (float,int) or not math.isfinite(threshold) or threshold<0:
        raise ValueError('Invalid mixed current electric threshold')
    currents=values['total_volume_current'][0,closure.current_indices].real.split(closure.current.counts)
    ac_fields=values['cell_electric'][1:,closure.current_indices].split(closure.current.counts,dim=1)
    if isinstance(tpa_scale,torch.Tensor) and tpa_scale.ndim:
        coefficients=tpa_scale[closure.current_indices].split(closure.current.counts)
    else:
        coefficients=(tpa_scale,)*len(regions)
    boundary={};electric=[]
    from .harmonic_current import surface_face_potentials
    for binding,block,potential,j,record,ac,tpa in zip(closure.current_bindings,closure.current.blocks,potentials,currents,info['current'],ac_fields,coefficients):
        closure.current.validate_receipt(record,block)
        gauge=(potential*block.nodal_volume).sum()
        bound=512*torch.finfo(dtype).eps*(potential.abs()*block.nodal_volume).sum()
        if not bool(gauge.abs()<=bound):raise ValueError('Mixed current potential violates volume gauge')
        from .nonlinear import instantaneous_harmonic_sources
        zero=ac.real.new_zeros(1,ac.shape[1],3).to(ac.dtype)
        third=instantaneous_harmonic_sources(torch.cat((zero,ac)),tpa_scale=1.).dissipative_current[0].real
        initial_current=block._coefficient(tpa,'TPA scale')[:,None]*third
        initial_norm=float(block.weak_current(initial_current)[1:].detach().norm())
        policy_threshold=closure.current.absolute_tolerance+closure.current.relative_tolerance*initial_norm
        if (record.get('converged') is not True or record.get('failure_reason') is not None
                or record.get('threshold_A_l2')!=policy_threshold):
            raise ValueError('Mixed current residual threshold differs from independent current policy')
        allowed=policy_threshold
        if (type(allowed) not in (float,int) or not math.isfinite(allowed) or allowed<0
                or not bool(block.weak_current(j).norm()<=allowed)):
            raise ValueError('Mixed current state violates current conservation')
        electric.append(block.electric(potential))
        boundary[binding.volume.region_id]=surface_face_potentials(block,binding,potential)
    if not bool((torch.cat(electric)-values['cell_electric'][0,closure.current_indices].real).norm()<=threshold):
        raise ValueError('Mixed current potential differs from cell electric state')
    packed=torch.cat((values['polarization'][0,closure.dielectric_indices].real.flatten(),
                      *(boundary[r] for r in closure.operator.current_regions)))
    expected_offsets=closure.maps['offset_response']@packed
    selected=torch.stack([expected_offsets[closure.operator.current_regions.index(r)] for r in regions])
    if (not torch.equal(response.potential,closure.maps['surface_response']@packed)
            or not torch.equal(response.conormal,closure.maps['conormal_response']@packed)
            or not torch.equal(response.offsets,expected_offsets) or not torch.equal(offsets,selected)):
        raise ValueError('Mixed current traces or offsets differ from retained response maps')
    dielectric=(closure.maps['electric_response']@packed).reshape(-1,3)
    if not bool((dielectric-values['cell_electric'][0,closure.dielectric_indices].real).norm()<=threshold):
        raise ValueError('Mixed dielectric DC field differs from retained response maps')
    values.update(dc_face_potentials=potentials,dc_potential_offsets=offsets,dc_current_regions=regions)
    return torch.zeros_like(values['cell_electric'][0].real).index_copy(0,closure.current_indices,
        torch.cat(electric)).index_copy(0,closure.dielectric_indices,dielectric)


def _validate_retained_response(values,receipt,prepared,*,dc_field=None):
    """Check retained field equations without assembling, sampling or solving."""
    solver=receipt.get('solver')
    threshold=solver.get('threshold_V_per_m_l2') if isinstance(solver,dict) else None
    if (not isinstance(solver,dict) or solver.get('converged') is not True
            or solver.get('failure_reason') is not None
            or solver.get('pump_scale')!=receipt['pump_scale']
            or solver.get('physical_qualification') is not False
            or solver.get('harmonic_truncation_converged') is not False
            or type(threshold) not in (int,float) or not math.isfinite(threshold) or threshold<0):
        raise ValueError('Invalid mixed harmonic solver convergence receipt')
    def norm(value):
        scale=value.abs().amax()
        result=scale if bool(scale==0) else scale*torch.linalg.vector_norm(value/scale)
        if not bool(torch.isfinite(result)):
            raise ValueError('Mixed harmonic response residual is not finite')
        return result
    if dc_field is None:
        dc_field=(prepared['static']['electric_response']@values['polarization'][0].real.flatten()).reshape(-1,3)
    predicted=[dc_field.to(values['cell_electric'].dtype)]
    pump=receipt['pump_scale']
    for n,part in enumerate(prepared['harmonics']):
        current=values['total_volume_current'][n+1].flatten()
        if prepared.get('response_backend')=='compressed':
            from .compressed_harmonic_snapshot import validate_surface_state
            induced=values['induced_surface_coefficients'][n]
            electric,_=validate_surface_state(part,current,induced)
        else:
            induced=part['surface_response']@current
            electric=part['electric_response']@current
        surface=pump*part['background_surface']+induced
        tolerance=512*torch.finfo(surface.real.dtype).eps*norm(surface)
        if not bool(norm(values['surface_coefficients'][n]-surface)<=tolerance):
            raise ValueError('Mixed harmonic surface state violates its prepared response')
        predicted.append(pump*part['background_electric']+electric.reshape(-1,3))
    if not bool(norm(torch.stack(predicted)-values['cell_electric'])<=threshold):
        raise ValueError('Mixed harmonic field violates its convergence threshold')
