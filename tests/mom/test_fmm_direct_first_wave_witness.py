"""Saved original worst geometry: corrected direct tails before a large rerun."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.fmm_error import (direct_first_wave_truncation_bounds_many,
    multipole_truncation_bound, _log_positive_tail_difference, _rational_tail_log)
from witwin.maxwell.mom.fmm_expansion import multipole_to_local, monomials
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
RULE='direct_first_wave_min_v1'


def test_positive_difference_rejection_and_whole_geometric_tail():
    x=torch.tensor([0.,0.,0.,float('nan'),-1000.,2.],device='cuda',dtype=torch.float64)
    y=x.new_tensor([-1.,0.,1.,-1.,-1000.000000000001,-torch.inf])
    result,valid=_log_positive_tail_difference(x,y)
    assert valid.tolist()==[True,False,False,False,False,True]
    assert bool(torch.isposinf(result[~valid]).all())
    assert abs(float(result[0])-math.log(-math.expm1(-1.)))<1e-15
    assert float(result[-1])==2.
    log_l=x.new_tensor([math.log(.7)]);log_q=x.new_tensor([-torch.inf]);lr=x.new_tensor([0.])
    for m in (1,2,3):
        whole=_rational_tail_log(m,x.new_tensor([0.]),log_l,log_q,lr)
        torch.testing.assert_close(whole,x.new_tensor([[-m*math.log(.7)]]),rtol=0,atol=1e-15)
    d=x.new_tensor([[1.,0,0]]);r=x.new_tensor([.1])
    laplace=direct_first_wave_truncation_bounds_many(d,r,r,0.,(0,6,12))
    assert bool(laplace['available'].all())
    assert bool((laplace['green_absolute']>0).all())
    from witwin.maxwell.mom.fmm_error import real_support_truncation_bounds_many
    reference=real_support_truncation_bounds_many(d,r,r,0.,(0,6,12))
    for key in ('green_absolute','gradient_component_absolute'):
        torch.testing.assert_close(laplace[key],reference[key],rtol=3e-14,atol=0)
    config=mw.FMMConfig(table_storage='dense',near_storage='full_matrix',kernel_bound_rule=RULE)
    assert config.payload()['version']=='cartesian_fmm_direct_first_wave_config_v1'
    assert mw.FMMConfig.restore(config.payload())==config
    for rule in ('direct_positive_tail_min_v1','first_wave_min_v1','bivariate_cauchy_v1'):
        changed=copy.deepcopy(config.payload());changed['parameters']['kernel_bound_rule']=rule
        with pytest.raises(ValueError):mw.FMMConfig.restore(changed)
    from witwin.maxwell.mom.linalg import StoppingRule
    with pytest.raises(NotImplementedError,match='free space'):
        mw.MoMConfig(operator_backend='fmm',fmm=config,riesz_storage='local_rwg_v1',
            stopping=StoppingRule(preconditioner='rwg_diagonal'),periodic=mw.MoMPeriodicConfig(((1.,0,0),)))


def test_saved_original_worst_geometry_native_bound_and_actual_full_kernel(tmp_path):
    required=('GSP_DIRECT_FMM_WITNESS','GSP_DIRECT_FMM_WITNESS_SHA256')
    if all(name not in os.environ for name in required):
        pytest.skip('Requires completed original direct-tail failure with saved witnesses')
    assert all(name in os.environ for name in required), 'Incomplete witness environment'
    path=Path(os.environ[required[0]])
    digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==os.environ[required[1]]
    old=json.loads(path.read_text())
    assert old['family']=='growing_ka' and old['edges']==3840 and old['order_admitted'] is False
    assert old['order_plan']['kernel_budget']==dict(green_absolute=1e-6,gradient_component_absolute=1e-5)
    witness=old['support_pair_geometry_witnesses']['gradient_component_absolute'][-1]
    assert witness['order']==12 and witness['value']>1e-5
    d=torch.tensor(witness['displacement'],device='cuda',dtype=torch.float64)
    a,b=(d.new_tensor(witness[name]) for name in ('test_radius','source_radius'))
    wave=complex(*witness['wave_number']);order=12
    bound=multipole_truncation_bound(d,a,b,wave,order,kernel_bound_rule=RULE)
    generator=torch.Generator(device='cuda').manual_seed(1729)
    directions=torch.randn((64,2,3),generator=generator,device='cuda',dtype=torch.float64)
    directions/=directions.norm(dim=-1,keepdim=True)
    directions[:4]=d.new_tensor([[[1,0,0],[1,0,0]],[[1,0,0],[-1,0,0]],
        [[0,1,0],[0,0,1]],[[-1,0,0],[1,0,0]]])
    x,y=directions[:,0]*a,directions[:,1]*b
    separation=d+x-y;radius=separation.norm(dim=-1)
    exact=torch.exp(1j*wave*radius)/(4*math.pi*radius)
    exact_gradient=exact[:,None]*(1j*wave-1/radius)[:,None]*separation/radius[:,None]
    green,gradient=multipole_to_local(d,wave,order)
    tx,sy=monomials(x,order).to(green.dtype),monomials(y,order).to(green.dtype)
    approximate=torch.einsum('ni,ij,nj->n',tx,green,sy)
    approximate_gradient=torch.einsum('ni,aij,nj->na',tx,gradient,sy)
    error_g=float((approximate-exact).abs().max());error_h=float((approximate_gradient-exact_gradient).abs().max())
    terms_g=torch.einsum('ni,ij,nj->n',tx.abs(),green.abs(),sy.abs()).max()
    terms_h=torch.einsum('ni,aij,nj->na',tx.abs(),gradient.abs(),sy.abs()).max()
    allowance_g=float(128*torch.finfo(torch.float64).eps*(terms_g+exact.abs().max()))
    allowance_h=float(128*torch.finfo(torch.float64).eps*(terms_h+exact_gradient.abs().max()))
    record=dict(old_failure_sha256=digest,order=order,old_gradient_bound=witness['value'],bound=bound,
        sampled_green_error=error_g,sampled_gradient_error=error_h,green_native_allowance=allowance_g,
        gradient_native_allowance=allowance_h,scope='Saved single worst-support geometry only; no full-system admission',
        physical_qualification=False,full_original_admission=False)
    (tmp_path/'witness.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
    assert bound['green_absolute']<=1e-6 and bound['gradient_component_absolute']<=1e-5
    assert error_g<=bound['green_absolute']+allowance_g
    assert error_h<=bound['gradient_component_absolute']+allowance_h
