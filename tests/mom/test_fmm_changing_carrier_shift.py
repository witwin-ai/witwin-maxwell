"""Independent physical point phase changes, padded polynomials and current maps."""
import json
import math
import pytest
import torch
from witwin.maxwell.mom.fmm_directional_shift import DirectionalChangingCarrierShift
from witwin.maxwell.mom.fmm_expansion import multi_indices

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def _poly(points,order):
    powers = torch.tensor(multi_indices(order),device=points.device,dtype=torch.int64)
    return points[:,None].pow(powers).prod(-1).to(torch.complex128)


def _relative(a,b):
    return float(((a-b).norm()/b.norm()).detach())


def _args():
    d=torch.tensor([-.13,.07,.03],device='cuda',dtype=torch.float64)
    return dict(displacement=d,direction=d.new_tensor([1.,0.,0.]),
        target_direction=d.new_tensor([math.cos(.05),math.sin(.05),0.]),wave_number=80.+2j,
        source_order=12,target_order=3,phase_order=9,child_radius=d.new_tensor(.03))


@pytest.mark.parametrize('ps,pt,L,wave,angle,radius',[
    (0,0,0,0.,.5,.05), (3,3,0,5.+.7j,0.,.02),
    (6,3,3,5.+.7j,.02,.01), (12,3,9,80.+2j,.05,.03),
    (12,0,12,2000.+2j,.002,.05)])
def test_actual_padded_source_and_receiver_points_full_H_current_and_higher_derivatives(ps,pt,L,wave,angle,radius,tmp_path):
    args=_args();d=args['displacement']
    args.update(source_order=ps,target_order=pt,phase_order=L,wave_number=wave,
        target_direction=d.new_tensor([math.cos(angle),math.sin(angle),0.]),child_radius=d.new_tensor(radius))
    op=DirectionalChangingCarrierShift(**args)
    gen=torch.Generator(device='cuda').manual_seed(1216)
    points=torch.randn((11,3),device='cuda',dtype=torch.float64,generator=gen)
    points=points/points.norm(dim=-1).amax()*radius*.99
    assert bool((points.norm(dim=-1)<=op.child_radius).all())
    parent=points+d
    child_source=_poly(points,ps)*torch.exp(-1j*wave*(points@args['direction']))[:,None]
    parent_source=_poly(parent,pt)*torch.exp(-1j*wave*(parent@args['target_direction']))[:,None]
    current=torch.randn((3,5,4,len(points)),device='cuda',dtype=torch.complex128,generator=gen).transpose(0,1)
    moments=(current@child_source).movedim(-1,-2).detach().requires_grad_()
    expected=(current@parent_source).movedim(-1,-2)
    saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t,lambda t:t):
        actual=op.source(moments)
    assert not saved
    degrees=torch.tensor([sum(a) for a in multi_indices(pt)],device='cuda')
    balance=(args['child_radius']+d.norm()).pow(degrees)
    source_error=_relative(actual/balance[:,None],expected/balance[:,None])
    locals=torch.randn((3,5,op.target_count,4),device='cuda',dtype=torch.complex128,generator=gen).transpose(0,1)
    locals=(locals/balance[:,None]).detach().requires_grad_()
    child_receiver=_poly(points,ps)*torch.exp(1j*wave*(points@args['direction']))[:,None]
    parent_receiver=_poly(parent,pt)*torch.exp(1j*wave*(parent@args['target_direction']))[:,None]
    evaluated=torch.einsum('pa,...ac->...pc',child_receiver,op.receiver(locals))
    reference=torch.einsum('pa,...ac->...pc',parent_receiver,locals)
    receiver_error=_relative(evaluated,reference)
    # Direct scalar Taylor evaluation is independent of polynomial table assembly.
    z=-1j*wave*(points@(args['target_direction']-args['direction']))
    term=torch.ones_like(z);series=term.clone()
    for n in range(1,L+1):
        term=term*z/n;series=series+term
    observed_phase=float(((series-torch.exp(z)).abs()/torch.exp(z).abs()).max())
    ideal_bound=0. if op.exact_phase else math.exp(float(op.log_phase_bound))
    assert observed_phase <= ideal_bound*(1+128*torch.finfo(torch.float64).eps)+512*torch.finfo(torch.float64).eps
    checks={}
    for name,value in [('source',moments),('receiver',locals)]:
        fn=getattr(op,name);out=fn(value);dual=torch.randn_like(out)
        H=fn(dual,adjoint=True);gradient,=torch.autograd.grad(out,value,dual)
        count=value.shape[-2]
        eye=torch.eye(count,device='cuda',dtype=torch.complex128)[...,None].expand(-1,-1,4)
        matrix_rows=fn(eye)[...,0]
        explicitH=torch.einsum('ji,...ic->...jc',matrix_rows.conj(),dual)
        lhs,rhs=(out.conj()*dual).sum(),(value.conj()*H).sum()
        dot=float(((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs())).detach())
        # Fixed metadata custom backward must retain the linear higher derivative.
        gradient2,=torch.autograd.grad(fn(value).abs().square().sum(),value,create_graph=True)
        tangent=torch.randn_like(value)
        higher,=torch.autograd.grad(gradient2,value,tangent)
        higher_ref=2*fn(fn(tangent),adjoint=True)
        checks[name]=dict(adjoint=_relative(H,explicitH),gradient=_relative(gradient,H),dot=dot,
            higher_current_derivative=_relative(higher,higher_ref))
        assert all(v<1e-11 for v in checks[name].values())
    record=dict(source_error=source_error,receiver_error=receiver_error,observed_phase_error=observed_phase,
        ideal_phase_bound=ideal_bound,checks=checks,receipt=op.receipt(),qualification=False)
    (tmp_path/'transfer.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
    assert source_error<1e-11 and receiver_error<1e-11


def test_changing_carrier_exact_owned_preflight_admission_native_and_saved_backward_guards(monkeypatch,tmp_path):
    args=_args();op=DirectionalChangingCarrierShift(**args)
    expected=128+32*op.source_count*op.target_count
    tensors=op.tensors()
    assert len({t.untyped_storage().data_ptr() for t in tensors})==len(tensors)
    assert all(t.untyped_storage().nbytes()==t.numel()*t.element_size() for t in tensors)
    assert sum(t.untyped_storage().nbytes() for t in tensors)==op.retained_bytes==expected
    original_build=DirectionalChangingCarrierShift._build_tables
    def forbidden(*a,**kw):raise AssertionError('Rejected transfer reached table allocation')
    monkeypatch.setattr(DirectionalChangingCarrierShift,'_build_tables',forbidden)
    invalid=[({'source_order':11},ValueError),({'phase_relative_budget':1e-30},ValueError),
        ({'child_radius':args['child_radius']*1000},ValueError),({'target_order':True},ValueError),
        ({'source_order':13},ValueError),({'direction':args['direction']*2},ValueError),
        ({'wave_number':80.-1j},ValueError),({'child_radius':args['child_radius'].float()},ValueError),
        ({'retained_budget_bytes':expected-1},MemoryError),({'tensor_workspace_budget_bytes':1},MemoryError),
        ({'phase_relative_budget':True},ValueError),({'rhs_batch':False},ValueError),
        ({'displacement':args['direction']*1000,'wave_number':1j},ValueError),
        ({'wave_number':1e-320,'child_radius':args['child_radius']*1e-100},ValueError),
        # The total phase argument remains representable although one scaled
        # nonzero component underflows before term construction.
        ({'displacement':args['displacement'].new_zeros(3),
          'direction':args['direction'].new_tensor([0.,0.,1.]),
          'target_direction':args['direction'].new_tensor([1e-124,1e-100,1.]),
          'wave_number':1e-200,'child_radius':args['child_radius'].new_tensor(1e290),
          'source_order':1,'target_order':0,'phase_order':1},ValueError)]
    for change,exc in invalid:
        kw=dict(args);kw.update(change)
        with pytest.raises(exc):DirectionalChangingCarrierShift(**kw)
    monkeypatch.setattr(DirectionalChangingCarrierShift,'_build_tables',original_build)
    # Nonzero scalar arguments can have structurally nonzero high-degree terms
    # below native range. This must reject rather than silently drop terms.
    kw=dict(args);kw.update(wave_number=1e-100,target_order=0,phase_order=12)
    with pytest.raises(ValueError,match='coefficient terms'):DirectionalChangingCarrierShift(**kw)
    for name,count in [('source',op.source_count),('receiver',op.target_count)]:
        value=torch.randn((count,4),device='cuda',dtype=torch.complex128,requires_grad=True)
        output=getattr(op,name)(value)
        with pytest.raises(ValueError):getattr(op,name)(value,adjoint=1)
        for changed in [value*float('nan'),value.to(torch.complex64)]:
            with pytest.raises(ValueError):getattr(op,name)(changed)
        getattr(op,name+'_coefficients').add_(1)
        with pytest.raises(ValueError,match='changed'):torch.autograd.grad(output,value,torch.ones_like(output))
        op=DirectionalChangingCarrierShift(**args)
    args['displacement'].add_(1);args['direction'].add_(1);args['child_radius'].mul_(2)
    op.assert_current()
    for field,new in [('phase_relative_budget',1e-10),('source_order',11),('phase_order',8),('rhs_batch',3)]:
        original=getattr(op,field);setattr(op,field,new)
        with pytest.raises(ValueError,match='changed'):op.receipt()
        setattr(op,field,original);op.assert_current()
    op.child_radius.add_(1)
    with pytest.raises(ValueError,match='changed'):op.receipt()
    record=dict(exact_owned_bytes=expected,owned_tensor_count=len(tensors),all_unique_storages=True,
        before_table_preflight=True,ideal_phase_only=True,qualification=False)
    (tmp_path/'guards.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)


def test_changing_carrier_four_RHS_finite_checks_empty_batches_and_action_overflow(monkeypatch,tmp_path):
    op=DirectionalChangingCarrierShift(**_args());finite=torch.isfinite;observed=[]
    def bounded(value,*a,**kw):
        if isinstance(value,torch.Tensor) and value.is_complex() and value.ndim>=3:
            observed.append(value.shape[0]);assert value.shape[0]<=4
        return finite(value,*a,**kw)
    monkeypatch.setattr(torch,'isfinite',bounded)
    for name,input_count,output_count in [('source',op.source_count,op.target_count),
            ('receiver',op.target_count,op.source_count)]:
        for adjoint in (False,True):
            ni,no=(output_count,input_count) if adjoint else (input_count,output_count)
            value=torch.randn((3,5,ni,4),device='cuda',dtype=torch.complex128).transpose(0,1)
            assert getattr(op,name)(value,adjoint=adjoint).shape==(5,3,no,4)
            empty=value.new_empty((0,ni,4))
            assert getattr(op,name)(empty,adjoint=adjoint).shape==(0,no,4)
            with pytest.raises(ValueError):getattr(op,name)(value*float('inf'),adjoint=adjoint)
    assert observed and max(observed)==4
    # A finite input times representable, finite coefficients can still overflow.
    huge=torch.full((op.target_count,4),1e308,device='cuda',dtype=torch.complex128)
    with pytest.raises(ValueError,match='translated moments'):op.receiver(huge)
    record=dict(maximum_finite_check_RHS=max(observed),rhs_batch=4,qualification=False)
    (tmp_path/'bounded.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
