"""Decode-only admitted dual current preparation with bounded stored arrays."""
import copy
from dataclasses import asdict
import torch
from .dual_current_action import DualStaticCurrentAction,FixedDualCurrentInverse,StreamedDualCurrentBalance
from .static_volume_curl import StaticVolumeCurlAction
from .static_projector_snapshot import snapshot_projector,restore_projector,_digest,_validate_bound
from .dual_static_snapshot import _tensor_identity
from .linalg import StoppingRule


def _identity(value):
    if isinstance(value,torch.Tensor):return _tensor_identity(value)
    if isinstance(value,dict):return {k:_identity(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_identity(v) for v in value]
    return value


def _binding(record):
    return {'schema_version':'streamed-dual-current-preparation-v1','sha256':_digest(_identity(record))}


def _curl_record(curl):
    curl.assert_current()
    return {'target_batch':curl.target_batch,'source_batch':curl.source_batch,
        'points':_tensor_identity(curl.points),'extents':tuple(_tensor_identity(g.extent) for g in curl.geometry),
        'receipt':copy.deepcopy(curl.receipt)}


def snapshot_dual_action(balance):
    if type(balance) is not StreamedDualCurrentBalance:raise TypeError('Native admitted dual balance required')
    balance.assert_current()
    a=balance.action
    record={'schema_version':'streamed-dual-current-preparation-v1',
        'projectors':tuple(snapshot_projector(p)[0] for p in a.projectors),
        'electric':a.electric.detach().cpu().clone(),'magnetic':a.magnetic.detach().cpu().clone(),
        'curl':_curl_record(a.curl),'action_receipt':copy.deepcopy(a.receipt),
        'rule':asdict(balance.inverse.rule),'refinement_steps':balance.inverse.refinement_steps,
        'bound':copy.deepcopy(balance.bound)}
    binding=_binding(record)
    restore_dual_action(record,a.blocks,expected_binding=binding)
    return record,binding


def restore_dual_action(record,blocks,*,expected_binding):
    """Rebind already validated face blocks; never integrate, solve or re-admit.

    The owning preparation supplies the expected identity. Only linear-sized
    material arrays and metadata persist. Reopening historical numerical bounds
    does not certify underlying action errors or physical accuracy.
    """
    keys={'schema_version','projectors','electric','magnetic','curl','action_receipt','rule','refinement_steps','bound'}
    if (not isinstance(record,dict) or set(record)!=keys or record['schema_version']!='streamed-dual-current-preparation-v1'
            or expected_binding!=_binding(record)):
        raise ValueError('Dual action snapshot differs from its preparation binding')
    blocks=tuple(blocks)
    if (not blocks or not isinstance(record['projectors'],tuple) or len(record['projectors'])!=len(blocks)
            or any(not isinstance(r,dict) for r in record['projectors'])):
        raise ValueError('Dual action snapshot body count differs')
    # Projector restoration also validates each exact native block and geometry.
    projectors=tuple(restore_projector(r,b,expected_binding={'schema_version':r.get('schema_version'),
        'sha256':_digest(r)}) for r,b in zip(record['projectors'],blocks))
    raw=record['curl']
    if not isinstance(raw,dict) or set(raw)!={'target_batch','source_batch','points','extents','receipt'}:
        raise ValueError('Invalid stored dual curl policy')
    curl=StaticVolumeCurlAction(tuple(b.volume for b in blocks),target_batch=raw['target_batch'],source_batch=raw['source_batch'])
    if raw!=_curl_record(curl):raise ValueError('Stored dual curl geometry or policy differs')
    action=object.__new__(DualStaticCurrentAction)
    action.curl,action.blocks,action.projectors=curl,blocks,projectors
    action.volumes,action.counts=curl.volumes,curl.counts
    action.device,action.dtype=curl.device,curl.dtype
    action.shape=(2*curl.shape[0],)*2
    for name in ('electric','magnetic'):
        value=record[name]
        if (not isinstance(value,torch.Tensor) or value.shape!=(curl.shape[0],) or value.dtype!=curl.dtype
                or not bool(torch.isfinite(value).all()) or not bool((value>0).all())):
            raise ValueError('Invalid stored dual conductivity')
        for body in value.split(tuple(3*n for n in action.counts)):
            if not bool((body==body[0]).all()):raise ValueError('Stored dual conductivity is not homogeneous')
        setattr(action,name,value.detach().to(curl.device).clone())
    expected_receipt={'method':'cellwise_dual_current_action','global_coupled_matrix_stored':False,
        'global_projector_stored':False,'global_curl_stored':False,'coupled_nonsingularity_certified':False,
        'operator_derivatives':False,'physical_qualification':False,'fast_work_qualified':False}
    if record['action_receipt']!=expected_receipt:raise ValueError('Invalid stored dual action qualification')
    action.receipt=copy.deepcopy(expected_receipt)
    action._binding=action._state()
    if not isinstance(record['rule'],dict):raise ValueError('Invalid stored dual inverse rule')
    try:rule=StoppingRule(**record['rule'])
    except (TypeError,ValueError) as error:raise ValueError('Invalid stored dual inverse rule') from error
    if asdict(rule)!=record['rule']:raise ValueError('Noncanonical stored dual inverse rule')
    inverse=FixedDualCurrentInverse(action,rule,refinement_steps=record['refinement_steps'])
    _validate_bound(record['bound'],rule,action.shape[0])
    cutoff=64*torch.finfo(action.dtype).eps*action.shape[0]*record['bound']['sigma_max_upper']
    if cutoff!=record['bound']['numerical_singularity_cutoff']:raise ValueError('Invalid stored dual inverse cutoff')
    balance=object.__new__(StreamedDualCurrentBalance)
    balance.action,balance.inverse,balance.bound=action,inverse,copy.deepcopy(record['bound'])
    balance._binding=(id(action),id(inverse))
    balance.assert_current()
    return balance


def _closure_binding(record):
    return {'schema_version':'streamed-dual-nonlinear-preparation-v1','sha256':_digest(_identity(record))}


def snapshot_dual_closure(closure):
    from .harmonic_dual_current import DualHarmonicCurrentClosure
    if type(closure) is not DualHarmonicCurrentClosure or closure.streamed_rule is None:
        raise TypeError('Native streamed dual nonlinear closure required')
    closure.assert_current()
    record={'schema_version':'streamed-dual-nonlinear-preparation-v1',
        'current':snapshot_dual_action(closure.current)[0],
        'settings':{k:getattr(closure,k) for k in ('relative_tolerance','absolute_tolerance','max_iterations','max_backtracks')},
        'rule':asdict(closure.streamed_rule),'column_batch':closure.streamed_column_batch}
    binding=_closure_binding(record)
    count=len(closure.volumes)
    restore_dual_closure(record,closure.current.action.blocks,sigma_e=closure.material_inputs[:count],
                        sigma_m=closure.material_inputs[count:],expected_binding=binding)
    return record,binding


def restore_dual_closure(record,blocks,*,sigma_e,sigma_m,expected_binding):
    """Restore native nonlinear preparation and bind caller-sampled materials.

    No nonlinear root is solved. Full state/exterior restoration and the public
    numerical policy must be handled by the owning harmonic snapshot.
    """
    import math
    from .harmonic_dual_current import DualHarmonicCurrentClosure,_BoundConductivity
    from .dual_current_response import DualElectricResponse
    from .dual_current_material import DualMaterialInverse
    keys={'schema_version','current','settings','rule','column_batch'}
    if (not isinstance(record,dict) or set(record)!=keys or record['schema_version']!='streamed-dual-nonlinear-preparation-v1'
            or expected_binding!=_closure_binding(record)):
        raise ValueError('Dual nonlinear snapshot differs from its preparation binding')
    settings=record['settings']
    if (not isinstance(settings,dict) or set(settings)!=
            {'relative_tolerance','absolute_tolerance','max_iterations','max_backtracks'}):
        raise ValueError('Invalid stored dual nonlinear settings')
    for key in ('relative_tolerance','absolute_tolerance'):
        if type(settings[key]) not in (int,float) or not math.isfinite(settings[key]) or settings[key]<0:
            raise ValueError('Invalid stored dual nonlinear tolerance')
    if settings['relative_tolerance']+settings['absolute_tolerance']==0:
        raise ValueError('Stored dual nonlinear tolerances cannot both be zero')
    for key in ('max_iterations','max_backtracks'):
        if type(settings[key]) is not int or settings[key]<0:raise ValueError('Invalid stored dual nonlinear budget')
    if type(record['column_batch']) is not int or record['column_batch']<1:
        raise ValueError('Invalid stored dual nonlinear column batch')
    blocks=tuple(blocks)
    if len(sigma_e)!=len(blocks) or len(sigma_m)!=len(blocks):raise ValueError('Dual material body count differs')
    current=restore_dual_action(record['current'],blocks,expected_binding=_binding(record['current']))
    rule=current.inverse.rule
    if (record['rule']!=asdict(rule) or current.bound['column_batch']!=record['column_batch']
            or current.inverse.refinement_steps!=2 or any(p.inverse.rule!=rule or p.inverse.refinement_steps!=2
                or p.receipt['tangent_bound']['column_batch']!=record['column_batch'] for p in current.action.projectors)):
        raise ValueError('Stored dual nonlinear and current policies differ')
    packed=tuple(torch.cat([b._coefficient(v,'DC conductivity').repeat_interleave(3)
                          for b,v in zip(blocks,values)]) for values in (sigma_e,sigma_m))
    if not all(torch.equal(v.detach(),reference) for v,reference in
               zip(packed,(current.action.electric,current.action.magnetic))):
        raise ValueError('Stored dual preparation differs from sampled materials')
    closure=object.__new__(DualHarmonicCurrentClosure)
    closure.current=current
    closure.volumes,closure.counts=current.action.volumes,current.action.counts
    for key,value in settings.items():setattr(closure,key,value)
    closure.streamed_rule,closure.streamed_column_batch=rule,record['column_batch']
    closure.response=DualElectricResponse(current)
    closure.material_inverse=DualMaterialInverse(current)
    closure.material_inputs=tuple(sigma_e)+tuple(sigma_m)
    closure.material_coefficients=tuple(_BoundConductivity.apply(value,closure) for value in packed)
    closure._versions=closure._state()
    closure.assert_current()
    return closure
