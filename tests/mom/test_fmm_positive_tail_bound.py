"""Finite-wave positive-coefficient tails against actual Helmholtz M2L."""
import copy
import json
import math
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.fmm_error import (positive_tail_truncation_bounds_many,
    multipole_truncation_bounds_many, multipole_truncation_bound, first_wave_truncation_bounds_many, direct_positive_tail_truncation_bounds_many,
    direct_first_wave_truncation_bounds_many)
from witwin.maxwell.mom.fmm_expansion import multipole_to_local, monomials
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
RULE='positive_tail_min_v1'

@pytest.mark.parametrize('rule', (RULE, 'first_wave_min_v1', 'direct_positive_tail_min_v1', 'direct_first_wave_min_v1'))
@pytest.mark.parametrize('dtype', (torch.float64, torch.float32))
def test_positive_tail_actual_m2l_point_and_gradient_errors_with_full_kernel(tmp_path, dtype, rule):
    generator = torch.Generator(device='cuda').manual_seed(731)
    directions = torch.randn((44, 2, 3), generator=generator, device='cuda', dtype=dtype)
    directions /= directions.norm(dim=-1, keepdim=True)
    directions[:4] = directions.new_tensor([[[1,0,0],[1,0,0]],[[1,0,0],[-1,0,0]],
                                           [[0,1,0],[0,0,1]],[[-1,0,0],[1,0,0]]])
    samples = []
    cases = [(1., 0., .2, .12, (0,2,6)), (1., 0., .45, .4, (2,6)),
             (1., 1.+0j, .07, .13, (2,6)), (1., 10.+2j, .1, .1, (4,8)),
             (1., 25j, .03, .04, (4,8)), (1., -6.+.01j, .09, .08, (2,6)),
             (1e-4, 1e4+5e3j, .08, .1, (2,6)),
             (1e4, 1e-4+5e-5j, .08, .1, (2,6))]
    for distance, wave, a, b, orders in cases:
        d = directions.new_tensor([distance,0,0])
        x, y = directions[:, 0]*(a*distance), directions[:, 1]*(b*distance)
        separation = d+x-y
        radius = separation.norm(dim=-1)
        exact = torch.exp(1j*wave*radius)/(4*math.pi*radius)
        exact_gradient = exact[:, None]*(1j*wave-1/radius)[:, None]*separation/radius[:, None]
        bound_function = {RULE: positive_tail_truncation_bounds_many,
                          'first_wave_min_v1': first_wave_truncation_bounds_many,
                          'direct_positive_tail_min_v1': direct_positive_tail_truncation_bounds_many,
                          'direct_first_wave_min_v1': direct_first_wave_truncation_bounds_many}[rule]
        bounds = bound_function(d[None], d.new_tensor([a*distance]),
                                                    d.new_tensor([b*distance]), wave, orders)
        assert bool(bounds['available'].all())
        for column, order in enumerate(orders):
            if dtype == torch.float32 and distance == 1e-4 and order == 6:
                # The existing coefficient representation cannot retain these
                # SI high-degree powers in float32; bound availability alone
                # does not make the implemented polynomial representable.
                with pytest.raises(ValueError, match='native precision'):
                    multipole_to_local(d, wave, order)
                samples.append(dict(distance=distance, wave=[complex(wave).real,complex(wave).imag],
                    order=order, coefficient_representation_rejected=True))
                (tmp_path/'kernel-errors.json').write_text(json.dumps(samples, indent=2)+'\n')
                continue
            green, gradient = multipole_to_local(d, wave, order)
            tx, sy = monomials(x, order).to(green.dtype), monomials(y, order).to(green.dtype)
            approximate = torch.einsum('ni,ij,nj->n', tx, green, sy)
            approximate_gradient = torch.einsum('ni,aij,nj->na', tx, gradient, sy)
            error_g = (approximate-exact).abs().max()
            error_h = (approximate_gradient-exact_gradient).abs().max()
            # Explicit native arithmetic allowance; this is not a roundoff bound.
            absolute_terms_g = torch.einsum('ni,ij,nj->n', tx.abs(), green.abs(), sy.abs()).max()
            absolute_terms_h = torch.einsum('ni,aij,nj->na', tx.abs(), gradient.abs(), sy.abs()).max()
            eps = torch.finfo(dtype).eps
            allowance_g = 128*eps*(absolute_terms_g+exact.abs().max())
            allowance_h = 128*eps*(absolute_terms_h+exact_gradient.abs().max())
            bound_g, bound_h = bounds['green_absolute'][0,column], bounds['gradient_component_absolute'][0,column]
            samples.append(dict(distance=distance, wave=[complex(wave).real,complex(wave).imag], order=order,
                green_error=float(error_g), gradient_error=float(error_h), green_bound=float(bound_g),
                gradient_bound=float(bound_h), green_native_allowance=float(allowance_g),
                gradient_native_allowance=float(allowance_h)))
            (tmp_path/'kernel-errors.json').write_text(json.dumps(samples, indent=2)+'\n')
            assert error_g <= bound_g+allowance_g
            assert error_h <= bound_h+allowance_h
    assert len(samples) == 17



def test_positive_contours_availability_precision_and_policy_identity():
    d=torch.tensor([[1.,0,0]]*5,device='cuda',dtype=torch.float64)
    a=d.new_tensor([0.,0.,.3,.499999999999,.6]);b=d.new_tensor([0.,.2,.25,.499999999999,.6])
    for wave in (0.,50.,-50.+2j,2000j,1e-200+1e200j,1e308+1e308j):
        positive=positive_tail_truncation_bounds_many(d,a,b,wave,(0,6,12))
        previous=multipole_truncation_bounds_many(d,a,b,wave,(0,6,12),kernel_bound_rule='real_support_min_v1')
        values=multipole_truncation_bounds_many(d,a,b,wave,(0,6,12),kernel_bound_rule=RULE)
        assert bool(values['available'][0].all()) and bool((values['green_absolute'][0]==0).all())
        assert bool((~values['available'][4]).all())
        for name in ('green_absolute','gradient_component_absolute'):
            expected=torch.minimum(torch.where(previous['available'],previous[name],torch.inf),
                torch.where(positive['available'],positive[name],torch.inf))
            expected=torch.where(previous['available']|positive['available'],expected,0.)
            torch.testing.assert_close(values[name],expected,rtol=0,atol=0)
            nonzero=values['available'] & (a[:,None]+b[:,None]>0)
            assert bool((values[name][nonzero]>0).all())
        if wave==50.:
            assert bool(positive['available'][:4].all())
            scalar=multipole_truncation_bound(d[2],a[2],b[2],wave,6,kernel_bound_rule=RULE)
            assert scalar['version']=='cartesian_positive_tail_min_bound_v1'
            assert scalar['green_absolute']==float(values['green_absolute'][2,1])
    for bad in ((),(True,),(-1,),(1.5,),(0,)*17):
        with pytest.raises(ValueError):positive_tail_truncation_bounds_many(d,a,b,1.,bad)
    from witwin.maxwell.mom.linalg import StoppingRule
    from witwin.maxwell.mom.runtime import _numerical_identity
    options=dict(operator_backend='fmm',riesz_storage='local_rwg_v1',stopping=StoppingRule(rtol=1e-9, preconditioner='rwg_diagonal'))
    for shift in ('dense','separable_v1'):
        config=mw.FMMConfig(table_storage='dense',near_storage='full_matrix',shift_storage=shift,kernel_bound_rule=RULE)
        payload=config.payload()
        assert payload['version']=='cartesian_fmm_positive_tail_config_v1'
        assert mw.FMMConfig.restore(payload)==config
        previous=mw.FMMConfig(table_storage='dense',near_storage='full_matrix',shift_storage=shift,kernel_bound_rule='real_support_min_v1')
        assert _numerical_identity(mw.MoMConfig(fmm=config,**options))!=_numerical_identity(mw.MoMConfig(fmm=previous,**options))
        for field in payload['parameters']:
            bad=copy.deepcopy(payload);del bad['parameters'][field]
            with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
        bad=copy.deepcopy(payload);bad['parameters']['kernel_bound_rule']='real_support_min_v1'
        with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
        with pytest.raises(NotImplementedError,match='free space'):
            mw.MoMConfig(fmm=config,periodic=mw.MoMPeriodicConfig(((1.,0,0),)),**options)


def test_positive_tail_global_small_wave_and_extreme_scales():
    d=torch.tensor([[1.,0,0]],device='cuda',dtype=torch.float64)
    r=d.new_tensor([.1])
    for scale in (1e-100,1.,1e100):
        v=positive_tail_truncation_bounds_many(d*scale,r*scale,r*scale,1e-200/scale,(6,))
        assert bool(v['available'].all())
        assert bool((v['green_absolute']>0).all()) and bool((v['gradient_component_absolute']>0).all())
    # Cancellation-contour rejection is conservative even for huge native wave
    # constants; unsupported representations cannot become an available zero.
    for wave in (1e308+1e308j,1e-200+1e308j):
        v=positive_tail_truncation_bounds_many(d*1e100,r*1e100,r*1e100,wave,(6,))
        assert not bool(v['available'].all())
        assert bool((v['green_absolute']==0).all())
