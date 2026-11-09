"""Explicit fixed-discretization harmonic-tail estimates, not physical gates."""
import math
import torch
from .runtime import _same_metadata


def compare_harmonic_cells(previous,current,*,relative_tolerance=1e-4,
                           absolute_tolerance=1e-10,tail_absolute_tolerance=1e-8):
    """Compare retained peak fields and new harmonics in volume-weighted RMS V/m.

    A new harmonic must meet its own absolute field budget; the fundamental
    cannot hide a physically relevant weak generated line. This estimator
    cannot certify spatial resolution, observation fields or global power.
    """
    from .harmonic_runtime import HarmonicData
    if not isinstance(previous,HarmonicData) or not isinstance(current,HarmonicData):
        raise TypeError('Harmonic comparison requires native coupled data')
    for value in (relative_tolerance,absolute_tolerance,tail_absolute_tolerance):
        if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
            raise ValueError('Harmonic comparison tolerances must be finite nonnegative numbers')
    if relative_tolerance+absolute_tolerance == 0 or tail_absolute_tolerance == 0:
        raise ValueError('Harmonic comparison requires nonzero retained and tail budgets')
    previous.decoder._assert_current()
    current.decoder._assert_current()
    old,new = previous.state,current.state
    retained = len(old.frequencies)
    if (len(new.frequencies) <= retained or not torch.equal(old.frequencies,new.frequencies[:retained])
            or old.cell_electric.shape[1:] != new.cell_electric.shape[1:]
            or old.cell_electric.dtype != new.cell_electric.dtype or old.cell_electric.device != new.cell_electric.device):
        raise ValueError('Harmonic comparison requires a strict extension on the same field grid')
    branch = old.receipt['solver'].get('branch_id')
    if not isinstance(branch,str) or not branch.strip():
        raise ValueError('Harmonic comparison requires an explicitly identified branch')
    if (not _same_metadata(previous.scene_signature,current.scene_signature)
            or old.receipt['pump_scale'] != new.receipt['pump_scale']
            or old.receipt['solver'].get('branch_id') != new.receipt['solver'].get('branch_id')):
        raise ValueError('Harmonic comparison requires the same physical Scene, pump and branch')
    a,b = previous.decoder.prepared,current.decoder.prepared
    def equal(x,y):
        return x is y or (x.dtype == y.dtype and x.device == y.device and torch.equal(x,y))
    for old_group,new_group in zip(a['bindings'],b['bindings']):
        if len(old_group) != len(new_group) or any(
                left.volume.region_id != right.volume.region_id or
                not equal(left.volume.vertices,right.volume.vertices) or not equal(left.volume.cells,right.volume.cells)
                for left,right in zip(old_group,new_group)):
            raise ValueError('Harmonic comparison cannot change volume discretization')
    compressed=a.get('response_backend')=='compressed'
    if compressed!=(b.get('response_backend')=='compressed'):
        raise ValueError('Harmonic comparison cannot change response backend')
    _validate_dc_prefix(a,b)
    for old_op,new_op,left,right in zip(a['operators'],b['operators'],a['harmonics'],b['harmonics']):
        if compressed:
            if left is right and old_op is new_op:continue
            from .coupled_response_snapshot import encode as encode_actions
            from .background_bank_snapshot import encode as encode_background
            from .volume_h_snapshot import _digest
            def identity(part,operator):
                payload=encode_actions(part['electric_action'],part['magnetic_action'])
                # Runtime Krylov observations may evolve without changing maps.
                payload.pop('sha256');payload['inverse'].pop('sha256')
                payload['inverse'].pop('diagnostics')
                return _digest((payload,encode_background(part['background_bank'],sources=operator.sources)))
            same=identity(left,old_op)==identity(right,new_op)
            keys=('background_surface','background_electric','background_magnetic','cell_volumes','points')
        else:
            same=equal(old_op.matrix,new_op.matrix)
            keys=('electric_response','magnetic_response','surface_response','background_surface',
                'background_electric','background_magnetic','cell_volumes','points')
        if not same or any(not equal(left[key],right[key]) for key in keys):
            raise ValueError('Harmonic comparison cannot change existing frequency responses')
    volumes = b['harmonics'][0]['cell_volumes']
    rms = lambda field: ((field.abs().square().sum(-1)*volumes).sum(-1)/volumes.sum()).sqrt()
    difference = rms(new.cell_electric[:retained]-old.cell_electric)
    retained_rms = rms(new.cell_electric[:retained])
    threshold = absolute_tolerance+relative_tolerance*retained_rms
    tail = rms(new.cell_electric[retained:])
    if not bool(torch.isfinite(torch.cat((difference,retained_rms,threshold,tail))).all()):
        raise ValueError('Harmonic cutoff RMS observations and thresholds must be finite')
    retained_ok = bool((difference <= threshold).all())
    tail_ok = bool((tail <= tail_absolute_tolerance).all())
    return {'previous_max_harmonic':retained-1,'current_max_harmonic':len(new.frequencies)-1,
        'retained_field_rms_V_per_m':retained_rms,
        'retained_difference_rms_V_per_m':difference,'retained_threshold_rms_V_per_m':threshold,
        'added_harmonic_rms_V_per_m':tail,'tail_absolute_tolerance_V_per_m':tail_absolute_tolerance,
        'retained_fields_passed':retained_ok,'added_harmonics_passed':tail_ok,
        'cell_truncation_estimate_passed':retained_ok and tail_ok,
        'estimator':'same_response_volume_weighted_peak_field_RMS',
        'physical_qualification':False,'harmonic_truncation_converged':False,
        'spatial_convergence_verified':False,'global_power_balance_verified':False}


def _validate_dc_prefix(a,b):
    """Compare actual static response data, including separately restored dual DC."""
    def equal(x,y):
        return x is y or (x.dtype==y.dtype and x.device==y.device and torch.equal(x,y))
    tensor_a=any(k in a['static'] for k in ('tensor_operator','tensor_floating_operator'))
    tensor_b=any(k in b['static'] for k in ('tensor_operator','tensor_floating_operator'))
    if tensor_a!=tensor_b:raise ValueError('Harmonic comparison cannot change tensor DC formulation')
    if tensor_a:
        if a['static'] is not b['static']:
            from .tensor_harmonic_snapshot import static_preparation_payload
            from .volume_h_snapshot import _digest
            if _digest(static_preparation_payload(a['static']))!=_digest(static_preparation_payload(b['static'])):
                raise ValueError('Harmonic comparison cannot change the tensor DC response')
        return
    current_a,current_b = 'current_closure' in a['static'],'current_closure' in b['static']
    mixed_a,mixed_b = 'mixed_operator' in a['static'],'mixed_operator' in b['static']
    if current_a != current_b or mixed_a != mixed_b:
        raise ValueError('Harmonic comparison cannot change the DC formulation')
    from .harmonic_dual_current import DualHarmonicCurrentClosure
    dual_a=isinstance(a['static'].get('current_closure'),DualHarmonicCurrentClosure)
    dual_b=isinstance(b['static'].get('current_closure'),DualHarmonicCurrentClosure)
    if dual_a!=dual_b:raise ValueError('Harmonic comparison cannot change the dual DC formulation')
    if dual_a:
        if a['static'] is not b['static']:
            from .harmonic_dual_bridge import encode_static
            from .volume_h_snapshot import _digest
            if _digest(encode_static(a['static'],a['bindings'][0]))!=_digest(encode_static(b['static'],b['bindings'][0])):
                raise ValueError('Harmonic comparison cannot change the dual DC response')
        return
    if current_a:
        if a['static'] is not b['static']:
            if mixed_a:
                from .harmonic_mixed_snapshot import encode
            else:
                from .harmonic_current_snapshot import encode
            def same(left,right):
                if isinstance(left,torch.Tensor) or isinstance(right,torch.Tensor):
                    return isinstance(left,torch.Tensor) and isinstance(right,torch.Tensor) and equal(left,right)
                if isinstance(left,dict) and isinstance(right,dict):
                    return left.keys() == right.keys() and all(same(left[k],right[k]) for k in left)
                if isinstance(left,(tuple,list)) and isinstance(right,(tuple,list)):
                    return len(left) == len(right) and all(same(x,y) for x,y in zip(left,right))
                return _same_metadata(left,right)
            left,right = encode(a['static']),encode(b['static'])
            keys = ('current_closure','floating_operator','material_channels','magnetic_permeabilities','points')
            if mixed_a:
                keys += ('mixed_operator','mixed_closure')
            if not all(same(left[key],right[key]) for key in keys):
                raise ValueError('Harmonic comparison cannot change the DC current response')
    else:
        for key in ('electric_response','surface_response'):
            if not equal(a['static'][key],b['static'][key]):
                raise ValueError('Harmonic comparison cannot change the DC response')
