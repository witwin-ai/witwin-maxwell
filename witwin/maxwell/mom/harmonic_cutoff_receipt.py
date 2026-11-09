"""Validate adaptive harmonic decisions against policy and saved CUDA fields."""
import math
import torch


def validate_harmonic_cutoff(stats,expected,realized,field,weights,*,volumes=(),
                             dc_current_regions=(),dual_dc=False):
    """Check history, solver policy, thresholds and terminal field observations."""
    policy=expected['cutoff']; initial=expected['harmonics']; final=realized['harmonics']
    if not isinstance(stats,dict):
        raise ValueError('Adaptive cutoff statistics are missing')
    receipt=stats.get('cutoff')
    if (type(initial) is not int or initial<1 or type(final) is not int or not initial<=final<=policy['max_harmonics']
            or not isinstance(receipt,dict) or receipt.get('initial_harmonics')!=initial
            or receipt.get('final_harmonics')!=final or receipt.get('policy')!=policy
            or stats.get('continuation_harmonics')!=initial
            or receipt.get('physical_qualification') is not False
            or receipt.get('harmonic_truncation_converged') is not False):
        raise ValueError('Adaptive cutoff execution does not match the frozen policy')
    checks=receipt.get('checks'); consecutive=0
    if (field.ndim!=3 or field.shape[0]!=final+1 or field.shape[-1]!=3
            or weights.shape!=(field.shape[1],) or weights.device!=field.device
            or not bool(torch.isfinite(field).all())
            or not bool((torch.isfinite(weights)&(weights>0)).all()) or not weights.numel()):
        raise ValueError('Invalid adaptive cutoff saved cell fields or volumes')
    rms=((field.abs().square().sum(-1)*weights).sum(-1)/weights.sum()).sqrt()
    eps=torch.finfo(rms.dtype).eps
    def same_observation(observed,reference):
        value=torch.as_tensor(observed,device=rms.device,dtype=rms.dtype)
        return value.shape==reference.shape and bool((torch.isfinite(value)&torch.isfinite(reference)&
            ((value-reference).abs()<=256*eps*(value.abs()+reference.abs()))).all())
    if not isinstance(checks,list) or len(checks)!=final-initial:
        raise ValueError('Adaptive cutoff history is incomplete')
    dc_sizes=[]
    if 'dc_tangent_rule' in expected:
        from witwin.maxwell.mom.linalg import StoppingRule
        from witwin.maxwell.mom.harmonic_current import validate_current_receipt
        dc_rule=StoppingRule(**expected['dc_tangent_rule'])
        regions=dc_current_regions
        for volume in volumes:
            if regions and volume['region_id'] not in regions:continue
            cells=volume['cells'].to(field.device)
            local=cells.new_tensor([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
            dc_sizes.append(len(torch.unique(cells[:,local].reshape(-1,3).sort(-1).values,dim=0))-1)
    for old,check in enumerate(checks,initial):
        if (not isinstance(check,dict) or check.get('previous_max_harmonic')!=old
                or check.get('current_max_harmonic')!=old+1):
            raise ValueError('Adaptive cutoff history changes harmonic order')
        from witwin.maxwell.mom.harmonic_snapshot import validate_solver_policy
        try:
            validate_solver_policy(check.get('solver'),expected,harmonics=old+1,
                cell_count=field.shape[1],real_dtype=field.real.dtype)
            if dual_dc and 'dc_tangent_rule' in expected:
                from witwin.maxwell.mom.dual_current_receipt import validate_dual_receipts
                validate_dual_receipts(check['solver'].get('dc_current_closure'),rule=dc_rule,
                    column_batch=expected['dc_tangent_column_batch'],cell_count=field.shape[1],dtype=field.real.dtype)
            elif dc_sizes:
                records=check['solver'].get('dc_current_closure')
                if not isinstance(records,(tuple,list)) or len(records)!=len(dc_sizes):
                    raise ValueError('DC cutoff receipt count mismatch')
                for record,size in zip(records,dc_sizes):
                    validate_current_receipt(record,rule=dc_rule,column_batch=expected['dc_tangent_column_batch'],
                        size=size,dtype=field.real.dtype)
        except ValueError as error:
            raise ValueError('Adaptive cutoff solver policy differs from construction') from error
        def values(key,length):
            value=check.get(key)
            if (not isinstance(value,list) or len(value)!=length
                    or any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in value)):
                raise ValueError('Invalid adaptive cutoff field observations')
            return value
        difference=values('retained_difference_rms_V_per_m',old+1)
        threshold=values('retained_threshold_rms_V_per_m',old+1)
        tail=values('added_harmonic_rms_V_per_m',1)
        retained_rms=values('retained_field_rms_V_per_m',old+1)
        rebuilt=policy['absolute_tolerance']+policy['relative_tolerance']*torch.tensor(
            retained_rms,device=rms.device,dtype=rms.dtype)
        if not same_observation(threshold,rebuilt):
            raise ValueError('Adaptive cutoff retained threshold differs from frozen tolerances')
        if old+1==final and (not same_observation(retained_rms,rms[:final])
                            or not same_observation(tail,rms[final:])):
            raise ValueError('Adaptive cutoff final observations differ from the saved fields')
        retained=all(a<=b for a,b in zip(difference,threshold))
        added=all(x<=policy['tail_absolute_tolerance'] for x in tail)
        passed=retained and added
        consecutive=consecutive+1 if passed else 0
        if (check.get('retained_fields_passed') is not retained or check.get('added_harmonics_passed') is not added
                or check.get('cell_truncation_estimate_passed') is not passed
                or check.get('tail_absolute_tolerance_V_per_m')!=policy['tail_absolute_tolerance']
                or check.get('successive_passes')!=consecutive
                or check.get('solver',{}).get('converged') is not True
                or check.get('solver',{}).get('branch_id')!=expected['branch_id']
                or any(check.get(key) is not False for key in ('physical_qualification',
                    'harmonic_truncation_converged','spatial_convergence_verified','global_power_balance_verified'))):
            raise ValueError('Adaptive cutoff decision differs from recorded observations')
        if consecutive>=policy['successive_passes'] and old+1!=final:
            raise ValueError('Adaptive cutoff continued after its declared stopping criterion')
    passed=consecutive>=policy['successive_passes']
    if (receipt.get('cell_truncation_estimate_passed') is not passed
            or receipt.get('failure_reason')!=(None if passed else 'harmonic_budget_exhausted')
            or (not passed and (final!=policy['max_harmonics'] or policy['failure']!='record'))):
        raise ValueError('Adaptive cutoff terminal decision violates the frozen policy')

