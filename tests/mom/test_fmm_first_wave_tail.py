"""First-wave ideal bound: independent remainder oracle and policy controls."""
import copy
from decimal import Decimal, localcontext
import math
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.fmm_error import (first_wave_truncation_bounds_many,
    multipole_truncation_bounds_many, multipole_truncation_bound, _log_second_exponential_remainder,
    direct_positive_tail_truncation_bounds_many, direct_first_wave_truncation_bounds_many)

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
RULE='first_wave_min_v1'


def test_second_exponential_remainder_against_independent_decimal_series():
    values=[1e-300,1e-100,1e-12,1e-6,.099999999,.1,.100000001,.5,2.,20.,50.,51.,1000.]
    x=torch.tensor(values,device='cuda',dtype=torch.float64)
    actual=_log_second_exponential_remainder(x.log(),x)
    with localcontext() as context:
        context.prec=100
        for value,native in zip(values,actual.tolist()):
            z=Decimal.from_float(value)
            if value<=.5:
                # Independent positive recurrence, with explicit geometric
                # remainder once the term is below 1e-90 of the partial sum.
                term=z*z/2;partial=term;degree=2
                while True:
                    term*=z/Decimal(degree+1);degree+=1;partial+=term
                    if term<=partial*Decimal('1e-90'):
                        ratio=z/Decimal(degree+1)
                        upper=partial+term*ratio/(1-ratio)
                        break
                reference=float(upper.ln()-z)
            else:
                reference=float((1-(1+z)*(-z).exp()).ln())
            allowance=128*torch.finfo(torch.float64).eps*(1+abs(reference))
            assert native>=reference-allowance
            if value<=50.:
                assert abs(native-reference)<=allowance
            else:
                assert native==0.
    # Log representation retains the second remainder after x^2 underflow.
    log_x=x.new_tensor([-1000.])
    retained=_log_second_exponential_remainder(log_x,torch.exp(log_x))
    assert math.isfinite(float(retained))
    assert abs(float(retained)-(-2000.-math.log(2)))<1e-12


@pytest.mark.parametrize('rule', (RULE,'direct_positive_tail_min_v1','direct_first_wave_min_v1'))
def test_first_wave_available_minima_identity_and_extreme_scales(rule):
    first_function={RULE:first_wave_truncation_bounds_many,
        'direct_positive_tail_min_v1':direct_positive_tail_truncation_bounds_many,
        'direct_first_wave_min_v1':direct_first_wave_truncation_bounds_many}[rule]
    previous_rule={RULE:'positive_tail_min_v1','direct_positive_tail_min_v1':RULE,
                   'direct_first_wave_min_v1':'direct_positive_tail_min_v1'}[rule]
    bound_version={RULE:'cartesian_first_wave_min_bound_v1',
        'direct_positive_tail_min_v1':'cartesian_direct_positive_tail_min_bound_v1',
        'direct_first_wave_min_v1':'cartesian_direct_first_wave_min_bound_v1'}[rule]
    config_version={RULE:'cartesian_fmm_first_wave_config_v1',
        'direct_positive_tail_min_v1':'cartesian_fmm_direct_positive_tail_config_v1',
        'direct_first_wave_min_v1':'cartesian_fmm_direct_first_wave_config_v1'}[rule]
    d=torch.tensor([[1.,0,0]]*5,device='cuda',dtype=torch.float64)
    a=d.new_tensor([0.,0.,.3,.499999999999,.6]);b=d.new_tensor([0.,.2,.25,.499999999999,.6])
    improved=False
    for wave in (0.,1.,50.,-50.+2j,2000j,1e-200+1e200j,1e308+1e308j):
        first=first_function(d,a,b,wave,(0,6,12))
        previous=multipole_truncation_bounds_many(d,a,b,wave,(0,6,12),kernel_bound_rule=previous_rule)
        values=multipole_truncation_bounds_many(d,a,b,wave,(0,6,12),kernel_bound_rule=rule)
        assert bool(values['available'][0].all())
        assert bool((~values['available'][4]).all())
        for name in ('green_absolute','gradient_component_absolute'):
            expected=torch.minimum(torch.where(previous['available'],previous[name],torch.inf),
                torch.where(first['available'],first[name],torch.inf))
            expected=torch.where(previous['available']|first['available'],expected,0.)
            torch.testing.assert_close(values[name],expected,rtol=0,atol=0)
            nonzero=values['available'] & (a[:,None]+b[:,None]>0)
            assert bool((values[name][nonzero]>0).all())
            assert bool((values[name][0]==0).all())
            improved |= bool(((values[name]<previous[name]) & previous['available']).any())
        if wave==50.:
            receipt=multipole_truncation_bound(d[2],a[2],b[2],wave,6,kernel_bound_rule=rule)
            assert receipt['version']==bound_version
            assert receipt['green_absolute']==float(values['green_absolute'][2,1])
            assert receipt['physical_qualification'] is False
    assert improved
    r=d.new_tensor([.1])
    for scale in (1e-100,1.,1e100):
        first=first_function(d[:1]*scale,r*scale,r*scale,1e-200/scale,(6,))
        assert bool(first['available'].all())
        assert bool((first['green_absolute']>0).all())
        assert bool((first['gradient_component_absolute']>0).all())
    from witwin.maxwell.mom.linalg import StoppingRule
    from witwin.maxwell.mom.runtime import _numerical_identity
    options=dict(operator_backend='fmm',riesz_storage='local_rwg_v1',stopping=StoppingRule(rtol=1e-9, preconditioner='rwg_diagonal'))
    for shift in ('dense','separable_v1'):
        config=mw.FMMConfig(table_storage='dense',near_storage='full_matrix',shift_storage=shift,kernel_bound_rule=rule)
        payload=config.payload()
        assert payload['version']==config_version
        assert mw.FMMConfig.restore(payload)==config
        previous=mw.FMMConfig(table_storage='dense',near_storage='full_matrix',shift_storage=shift,kernel_bound_rule=previous_rule)
        assert _numerical_identity(mw.MoMConfig(fmm=config,**options))!=_numerical_identity(mw.MoMConfig(fmm=previous,**options))
        for field in payload['parameters']:
            bad=copy.deepcopy(payload);del bad['parameters'][field]
            with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
        for changed_rule in ('positive_tail_min_v1','real_support_min_v1','bivariate_cauchy_v1'):
            bad=copy.deepcopy(payload);bad['parameters']['kernel_bound_rule']=changed_rule
            with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
        with pytest.raises(NotImplementedError,match='free space'):
            mw.MoMConfig(fmm=config,periodic=mw.MoMPeriodicConfig(((1.,0,0),)),**options)
