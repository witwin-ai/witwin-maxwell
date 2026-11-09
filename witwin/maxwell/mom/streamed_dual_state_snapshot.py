"""Bounded dual state replay using independently bound native contractions."""
import torch
from .dual_current_action import StreamedDualCurrentBalance
from .dual_static_snapshot import _state_payload,_restore_dual_state
from .dual_action_snapshot import _identity
from .static_projector_snapshot import _digest


def _binding(payload):
    return {'schema_version':'streamed-dual-state-v1','sha256':_digest(_identity(payload))}


@torch.no_grad()
def snapshot_streamed_dual_state(balance,result,exterior,electric_source,magnetic_source):
    """Evaluate original native contractions once, then seal their exact inputs.

    The returned binding belongs in an independent owning state identity.
    Reopening the payload must not trust a binding supplied inside that payload.
    """
    if type(balance) is not StreamedDualCurrentBalance:raise TypeError('Native streamed dual balance required')
    base=_state_payload(balance,result,exterior)
    # This is a linear source response, not another nonlinear root/admission.
    reference=balance.solve(electric_source,magnetic_source)
    witness={'electric_source':electric_source,'magnetic_source':magnetic_source,
        'curl_m':balance.action.curl.apply(result['magnetic_current'].flatten()).reshape_as(electric_source),
        'curl_j':balance.action.curl.apply(result['electric_current'].flatten()).reshape_as(electric_source),
        'electric_response':reference['electric'],'magnetic_response':reference['magnetic']}
    payload={'schema_version':'streamed-dual-state-v1','state':base,
             'witness':{k:v.detach().cpu().clone() for k,v in witness.items()}}
    binding=_binding(payload)
    restore_streamed_dual_state(payload,balance,exterior.bindings,exterior.electric_layer.operator,
                               expected_binding=binding)
    return payload,binding


def restore_streamed_dual_state(payload,balance,bindings,laplace,*,expected_binding):
    """Replay verified historical contractions; do not recompute kernel actions.

    This is exact bound replay, not an independent re-evaluation of C or R.
    The owning HB decoder must compare the stored response/source to its actual
    source and retain the original DC and cross-harmonic residual thresholds.
    """
    if type(balance) is not StreamedDualCurrentBalance:raise TypeError('Native streamed dual balance required')
    if (not isinstance(payload,dict) or set(payload)!={'schema_version','state','witness'}
            or payload['schema_version']!='streamed-dual-state-v1' or expected_binding!=_binding(payload)):
        raise ValueError('Streamed dual state differs from its independent owning identity')
    raw=payload['witness']
    keys={'electric_source','magnetic_source','curl_m','curl_j','electric_response','magnetic_response'}
    if not isinstance(raw,dict) or set(raw)!=keys:raise ValueError('Invalid streamed dual response witness')
    shape=(sum(balance.counts),3);sample=balance.points
    values={}
    for key,value in raw.items():
        if (not isinstance(value,torch.Tensor) or value.shape!=shape or value.dtype!=sample.dtype
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Invalid streamed dual witness shape, dtype or finiteness')
        values[key]=value.detach().to(sample.device).clone()
    result,exterior=_restore_dual_state(payload['state'],balance,bindings,laplace,
                                      curl_products=(values['curl_m'],values['curl_j']))
    for current,field,source,sigma in (
            (result['electric_current'],result['electric'],values['electric_source'],balance.electric),
            (result['magnetic_current'],result['magnetic'],values['magnetic_source'],balance.magnetic)):
        conductive=sigma.reshape(shape)*field
        scale=(current.abs()+conductive.abs()+source.abs()).max()
        threshold=2048*torch.finfo(sample.dtype).eps*sum(balance.counts)*scale
        if (not bool(torch.isfinite(threshold)) or not bool((current-conductive-source).abs().max()<=threshold)):
            raise ValueError('Streamed dual state violates its witnessed source constitutive equation')
    # Preserve the nonzero evaluated response for the owning HB residual check.
    result['response_witness']={k:values[k] for k in keys if not k.startswith('curl_')}
    result['replay_receipt']={'validation':'independently_bound_native_contraction_replay',
        'kernel_recomputed':False,'inverse_recomputed':False,'physical_qualification':False}
    return result,exterior


def validate_response_witness(result,electric_source,magnetic_source,actual_electric,threshold):
    """Keep the original DC field gate and return RSe for the total HB norm.

    The caller supplies the original owning HB threshold, never a replacement
    derived from local constitutive or flux tolerances.
    """
    import math
    if type(threshold) not in (int,float) or not math.isfinite(threshold) or threshold<0:
        raise ValueError('Invalid owning HB residual threshold')
    witness=result['response_witness']
    for key,source in (('electric_source',electric_source),('magnetic_source',magnetic_source)):
        value=witness[key]
        if (not isinstance(source,torch.Tensor) or source.shape!=value.shape or source.device!=value.device
                or source.dtype!=value.dtype or not torch.equal(source,value)):
            raise ValueError('Streamed dual response witness differs from owning HB source')
    expected=witness['electric_response'].flatten()
    field=result['electric'].flatten()
    if (not isinstance(actual_electric,torch.Tensor) or actual_electric.shape!=expected.shape
            or actual_electric.device!=expected.device or actual_electric.dtype!=expected.dtype
            or not bool(torch.isfinite(actual_electric).all())
            or not bool((actual_electric-expected).norm()<=threshold)
            or not bool((field-actual_electric).norm()<=threshold)):
        raise ValueError('Stored dual DC field violates the witnessed coupled equation')
    return expected
