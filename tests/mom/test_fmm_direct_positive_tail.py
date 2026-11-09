"""Direct positive tails against an independent total-degree coefficient sum."""
from decimal import Decimal, localcontext
import math
import pytest
import torch
from witwin.maxwell.mom.fmm_error import direct_positive_tail_truncation_bounds_many, direct_first_wave_truncation_bounds_many
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def coefficient_oracle(a,b,wave,order, *, first_wave_exact=False):
    # Independent host reference: total-degree majorant coefficients, then
    # binomial degree selection. It does not use the implemented rational tails.
    with localcontext() as context:
        context.prec=90
        a,b=Decimal.from_float(a),Decimal.from_float(b)
        k=Decimal.from_float(abs(complex(wave)));kappa=Decimal.from_float(complex(wave).imag)
        powers=[Decimal(1)]
        for j in range(1,142):powers.append(powers[-1]*k/Decimal(j))
        coefficients=[sum((Decimal(n-j+2)*powers[j]/2 for j in range(n+1)),Decimal(0)) for n in range(142)]
        if first_wave_exact:
            coefficients=[value-Decimal(n)/2-(k*Decimal(n-1)/2 if n>=2 else 0)
                          for n,value in enumerate(coefficients)]
        apowers=[Decimal(1)];bpowers=[Decimal(1)]
        for n in range(1,142):
            apowers.append(apowers[-1]*a);bpowers.append(bpowers[-1]*b)
        green=Decimal(0);gradient=Decimal(0)
        for n in range(order+1,141):
            missing=sum((Decimal(math.comb(n,j))*(apowers[n-j]*bpowers[j]+bpowers[n-j]*apowers[j])
                         for j in range(order+1,n+1)),Decimal(0))
            green+=coefficients[n]*missing
            gradient+=Decimal(n+1)*coefficients[n+1]*missing
        # Bound all remaining total degrees by a larger positive radial disk.
        t=Decimal('.6');r1=1/(1-t);r2=r1*r1;r3=r2*r1
        phi=(k*t).exp()*(r1+r2)/2
        derivative=(k*t).exp()*(k*r1/2+(k+1)*r2/2+r3)
        ratio=(a+b)/t
        green_upper=green+2*phi*ratio**141
        gradient_upper=gradient+2*derivative*ratio**141
        scale=(-kappa).exp()/Decimal.from_float(4*math.pi)
        return tuple(float(x*scale) for x in (green,green_upper,gradient,gradient_upper))


@pytest.mark.parametrize('rule', ('direct_positive_tail_min_v1','direct_first_wave_min_v1'))
def test_direct_tail_against_independent_binomial_coefficient_oracle(rule):
    bound_function=direct_positive_tail_truncation_bounds_many if rule=='direct_positive_tail_min_v1' else direct_first_wave_truncation_bounds_many
    d=torch.tensor([[1.,0,0]],device='cuda',dtype=torch.float64)
    for a,b in ((0.,.1),(.03,.07),(.08,.12)):
        for wave in (0.,1.,20.+2j,50.):
            for order in (0,4,12):
                values=bound_function(d,d.new_tensor([a]),d.new_tensor([b]),wave,(order,))
                assert bool(values['available'].all())
                g0,g1,h0,h1=coefficient_oracle(a,b,wave,order,first_wave_exact=rule=='direct_first_wave_min_v1')
                for name,lower,upper in [('green_absolute',g0,g1),('gradient_component_absolute',h0,h1)]:
                    native=float(values[name])
                    assert lower*(1-3e-12)<=native<=upper*(1+3e-12)


@pytest.mark.parametrize('rule', ('direct_positive_tail_min_v1','direct_first_wave_min_v1'))
def test_direct_tail_unavailable_workspace_and_exact_zero_support(rule):
    bound_function=direct_positive_tail_truncation_bounds_many if rule=='direct_positive_tail_min_v1' else direct_first_wave_truncation_bounds_many
    d=torch.tensor([[1.,0,0]]*3,device='cuda',dtype=torch.float64)
    a=d.new_tensor([0.,.1,.6]);b=d.new_tensor([0.,.1,.6])
    for wave in (0.,100.,1000.,1e308+1e308j):
        values=bound_function(d,a,b,wave,(12,64,65))
        assert bool(values['available'][0].all())
        assert bool((values['green_absolute'][0]==0).all())
        assert not bool(values['available'][2].any())
        assert not bool(values['available'][1,2])
        if wave==1000.:
            assert not bool(values['available'][1].any())
        rejected=~values['available']
        assert bool((values['green_absolute'][rejected]==0).all())
        assert bool((values['gradient_component_absolute'][rejected]==0).all())
