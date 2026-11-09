"""Real-support Helmholtz majorant against actual Cartesian M2L polynomials."""
import copy
import json
import math
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.fmm_error import (
    multipole_truncation_bound, multipole_truncation_bounds_many,
    real_support_truncation_bounds_many,
)
from witwin.maxwell.mom.fmm_expansion import multipole_to_local, monomials
from witwin.maxwell.mom.fmm_compact import CompactElectromagneticM2L
from witwin.maxwell.mom.physics import compile_medium

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
RULE = 'real_support_min_v1'


def geometry(dtype=torch.float64):
    d = torch.tensor([[1., 0., 0.]]*5, device='cuda', dtype=dtype)
    rt = torch.tensor([0., .07, .3, .7, .6], device='cuda', dtype=dtype)
    rs = torch.tensor([0., .1, .25, .1, .6], device='cuda', dtype=dtype)
    return d, rt, rs


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_available_minimum_zero_limits_and_native_representation(dtype):
    d, rt, rs = geometry(dtype)
    orders = (0, 6, 2, 12, 6)
    for wave in (0., 6.+.01j, -4.+3j, 1e-200+1e200j, 1e308+1e308j, 2000j):
        real = real_support_truncation_bounds_many(d, rt, rs, wave, orders)
        old = multipole_truncation_bounds_many(d, rt, rs, wave, orders)
        both = multipole_truncation_bounds_many(d, rt, rs, wave, orders, kernel_bound_rule=RULE)
        assert bool(both['available'][0].all())
        assert bool((both['green_absolute'][0] == 0).all())
        assert bool((both['gradient_component_absolute'][0] == 0).all())
        for name in ('green_absolute', 'gradient_component_absolute'):
            expected = torch.minimum(torch.where(old['available'], old[name], torch.inf),
                                     torch.where(real['available'], real[name], torch.inf))
            expected = torch.where(old['available'] | real['available'], expected, 0.)
            torch.testing.assert_close(both[name], expected, rtol=0, atol=0)
            positive = both['available'] & (rt[:, None]+rs[:, None] > 0)
            assert bool((both[name][positive] > 0).all())
        if abs(complex(wave)) < 100:
            assert bool(real['available'][:4].all())
            assert bool((~old['available'][2:]).all())
            assert bool((~both['available'][4]).all())
            scalar = multipole_truncation_bound(d[2], rt[2], rs[2], wave, 6, kernel_bound_rule=RULE)
            assert scalar['version'] == 'cartesian_real_support_min_bound_v1'
            assert scalar['physical_qualification'] is False
            assert scalar['green_absolute'] == float(both['green_absolute'][2, 1])
    tiny_d = d[:1].double()*1e-100
    tiny_r = rt[1:2].double()*1e-100
    tiny = real_support_truncation_bounds_many(tiny_d, tiny_r, tiny_r, 1e-300, (6,))
    assert bool(tiny['available'].all()) and bool((tiny['green_absolute'] > 0).all())
    empty = real_support_truncation_bounds_many(d[:0], rt[:0], rs[:0], 1., (1, 3))
    assert empty['available'].shape == (0, 2)
    for bad in ((), (True,), (-1,), (1.5,), (0,)*17):
        with pytest.raises(ValueError):
            real_support_truncation_bounds_many(d, rt, rs, 1., bad)
    for args in ((d.cpu(), rt, rs, 1., (1,)), (d, -rt, rs, 1., (1,)),
                 (d.clone().requires_grad_(), rt, rs, 1., (1,)),
                 (d, rt, rs, -1j, (1,)), (d*0, rt, rs, 1., (1,))):
        with pytest.raises(ValueError):
            real_support_truncation_bounds_many(*args)
    with pytest.raises(ValueError):
        multipole_truncation_bounds_many(d, rt, rs, 1., (1,), kernel_bound_rule='unknown')


@pytest.mark.parametrize('dtype', (torch.float64, torch.float32))
def test_actual_m2l_point_and_gradient_errors_with_full_kernel(tmp_path, dtype):
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
        bounds = real_support_truncation_bounds_many(d[None], d.new_tensor([a*distance]),
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


def test_compact_actual_tables_and_single_side_laplace_tightness():
    dtype = torch.float64
    d = torch.tensor([.7,.3,-.1], device='cuda', dtype=dtype)
    wave = compile_medium(mw.Material(eps_r=2., mu_r=1.3, sigma_e=.001), 3e8)
    order = 6
    dense_g, dense_h = multipole_to_local(d, wave.k, order)
    compact = CompactElectromagneticM2L(d, wave, order, row_batch=9)
    cg, ch = zip(*(compact._tables(start, min(start+9, compact.count))
                  for start in range(0, compact.count, 9)))
    torch.testing.assert_close(torch.cat(cg), dense_g, rtol=2e-14, atol=1e-13)
    torch.testing.assert_close(torch.cat(ch).movedim(-1,0), dense_h, rtol=2e-14, atol=1e-13)
    # Collinear single-sided Laplace remainder attains both ideal tail bounds.
    d = d.new_tensor([1.,0,0]); x = d[None]*0; y = d[None]*.3
    for order in (0,2,6):
        g, h = multipole_to_local(d, 0., order)
        tx, sy = monomials(x,order).to(g.dtype), monomials(y,order).to(g.dtype)
        approximate = torch.einsum('ni,ij,nj->n',tx,g,sy)[0]
        gradient = torch.einsum('ni,aij,nj->na',tx,h,sy)[0,0]
        bounds = real_support_truncation_bounds_many(d[None],d.new_tensor([0.]),d.new_tensor([.3]),0.,(order,))
        torch.testing.assert_close((approximate-1/(4*math.pi*.7)).abs(),bounds['green_absolute'][0,0],rtol=2e-12,atol=1e-15)
        torch.testing.assert_close((gradient+1/(4*math.pi*.7**2)).abs(),bounds['gradient_component_absolute'][0,0],rtol=2e-12,atol=1e-15)


def test_policy_identity_exact_default_roundtrip_and_periodic_rejection():
    from witwin.maxwell.mom.runtime import _numerical_identity
    from witwin.maxwell.mom.linalg import StoppingRule
    options = dict(operator_backend='fmm', riesz_storage='local_rwg_v1',
                   stopping=StoppingRule(rtol=1e-9, preconditioner='rwg_diagonal'))
    for shifts in ('dense','separable_v1'):
        default = mw.FMMConfig(table_storage='dense',near_storage='full_matrix',shift_storage=shifts)
        assert 'kernel_bound_rule' not in default.payload()['parameters']
        assert mw.FMMConfig.restore(default.payload()) == default
        selected = mw.FMMConfig(table_storage='dense',near_storage='full_matrix',shift_storage=shifts,kernel_bound_rule=RULE)
        payload = selected.payload()
        assert payload['version'] == 'cartesian_fmm_real_support_config_v1'
        assert mw.FMMConfig.restore(payload) == selected
        assert selected.construction()['kernel_bound_rule'] == RULE
        assert _numerical_identity(mw.MoMConfig(fmm=default,**options)) != _numerical_identity(mw.MoMConfig(fmm=selected,**options))
        for field in payload['parameters']:
            bad=copy.deepcopy(payload); del bad['parameters'][field]
            with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
        for mutation in ('version','kernel_bound_rule'):
            bad=copy.deepcopy(payload)
            if mutation=='version':bad['version']='cartesian_fmm_separable_config_v1'
            else:bad['parameters'][mutation]='bivariate_cauchy_v1'
            with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
        with pytest.raises(NotImplementedError,match='free space'):
            mw.MoMConfig(fmm=selected,periodic=mw.MoMPeriodicConfig(((1.,0.,0.),)),**options)
    with pytest.raises(ValueError):mw.FMMConfig(kernel_bound_rule='unknown')
