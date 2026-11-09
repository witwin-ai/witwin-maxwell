"""Exact radial derivative covariance with original native algebra references."""
import hashlib
import itertools
import json
import math

import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.fmm_storage import signed_axis_displacement_catalog
from witwin.maxwell.mom.fmm_compact import (CompactElectromagneticM2L,
    SignedAxisCompactElectromagneticM2L, signed_axis_compact_moment_actions)
from witwin.maxwell.mom.scalar_moment_action import scalar_moment_action
from witwin.maxwell.mom.fmm_capacity import storage_bytes

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def signed_axes():
    return tuple((p,s) for p in itertools.permutations(range(3))
                 for s in itertools.product((-1,1),repeat=3))


def medium():
    return compile_medium(mw.Material(eps_r=2.6+.017j,mu_r=1.41,sigma_m=.009),2.11e8)


def displacement():
    return torch.tensor([.4375,-.28125,.15625],device='cuda',dtype=torch.float64)


def test_new_two_chunk_catalog_retains_exact_input_bindings_and_one_ulp(tmp_path):
    r=displacement()
    variants=torch.stack([r[list(p)]*r.new_tensor(s) for p,s in signed_axes()])
    changed=variants[7].clone();changed[1]=torch.nextafter(changed[1],changed.new_tensor(math.inf))
    centers=torch.cat((torch.zeros_like(r)[None],variants,changed[None]))
    pairs=tuple((i%48+1,0) for i in range(1056))+((49,0),)
    owners,bindings=signed_axis_displacement_catalog(centers,pairs)
    assert len(owners)==2 and len(bindings)==len(pairs)==1057
    base=(.4375,-.28125,.15625)
    host=[(0.,0.,0.)]+[tuple(base[p[i]]*s[i] for i in range(3)) for p,s in signed_axes()]
    extra=list(host[8]);extra[1]=math.nextafter(extra[1],math.inf);host.append(tuple(extra))
    keys={}
    for index,(a,b) in enumerate(pairs):
        key=tuple(sorted(abs(x-y) for x,y in zip(host[a],host[b],strict=True)))
        keys.setdefault(key,index)
    aa,bb=torch.tensor(pairs,device='cuda',dtype=torch.int64).unbind(-1)
    oa,ob=torch.tensor([pairs[owners[owner]] for owner,_,_ in bindings],device='cuda',dtype=torch.int64).unbind(-1)
    axes=torch.tensor([p for _,p,_ in bindings],device='cuda',dtype=torch.int64)
    signs=torch.tensor([s for _,_,s in bindings],device='cuda',dtype=torch.float64)
    independent=(centers[oa]-centers[ob]).gather(1,axes)*signs
    torch.testing.assert_close(centers[aa]-centers[bb],independent,rtol=0,atol=0)
    assert len(keys)==len(owners) and owners==tuple(keys.values())
    assert all(owner==0 for owner,_,_ in bindings[:-1]) and bindings[-1][0]==1
    (tmp_path/'signed-axis-catalog.json').write_text(json.dumps(dict(
        completed=True,logical_input_pairs=1057,unique_physical_input_pairs=49,
        chunk_limit=1024,chunks=2,original_physical_displacement_owners=2,
        all_input_bindings_exact=True,all48_signed_permutations=True,
        one_ulp_distinct_owner=True,physical_partition_or_f30_not_replayed=True,
        production_qualification=False),indent=2)+'\n')


@pytest.mark.parametrize('order,row_batch',[(0,1),(4,7),(7,11)])
def test_new_all48_native_original_tables_actions_H_gradients_and_storage(order,row_batch,tmp_path):
    r=displacement();m=medium();transforms=signed_axes()
    actual=torch.stack([r[list(p)]*r.new_tensor(s) for p,s in transforms])
    owner=CompactElectromagneticM2L(r,m,order,row_batch=row_batch)
    views=tuple(SignedAxisCompactElectromagneticM2L(owner,p,s) for p,s in transforms)
    references=CompactElectromagneticM2L.create_many(actual,m,order,row_batch=row_batch)
    tables=tuple(op._tables(0,op.count) for op in references)
    artifact=tmp_path/'signed-axis-original-native-reference.pt'
    torch.save(dict(order=order,frequency_Hz=2.11e8,displacements=actual,
        original_independent_coefficients=torch.stack([op.coefficients for op in references]),
        green=torch.stack([t[0] for t in tables]),gradient=torch.stack([t[1] for t in tables]),
        original_reference_constructed_once=True),artifact)
    digest=hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert storage_bytes([t for op in (owner,*views) for t in op.tensors()])==owner.retained_bytes
    assert all(op.receipt()['additional_owned_tensor_bytes']==0 for op in views)
    generator=torch.Generator(device='cuda').manual_seed(19237+order)
    x=torch.randn((3,5,2,owner.count,4),device='cuda',dtype=torch.complex128,generator=generator).transpose(0,1).requires_grad_()
    y=torch.randn(x.shape,device='cuda',dtype=x.dtype,generator=generator)
    direction=torch.randn(x.shape,device='cuda',dtype=x.dtype,generator=generator)
    relative=lambda a,b:float(((a-b).norm()/b.norm()).detach())
    errors={k:[] for k in ('green','gradient','forward','original_H','native_H','current','higher')}
    expected_forward=[];expected_adjoint=[]
    for view,(green,gradient) in zip(views,tables,strict=True):
        vg,vd=view._tables(0,view.count)
        errors['green'].append(relative(vg,green));errors['gradient'].append(relative(vd,gradient))
        forward=scalar_moment_action(green,gradient,m,x.detach())
        # The reference Hermitian action uses its own original tables.
        adjoint=scalar_moment_action(green,gradient,m,y,adjoint=True)
        expected_forward.append(forward);expected_adjoint.append(adjoint)
        saved=[]
        with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t,lambda t:t):value=view.apply(x)
        assert saved==[]
        h=view.adjoint(y)
        errors['forward'].append(relative(value,forward));errors['original_H'].append(relative(h,adjoint))
        lhs=(value.conj()*y).sum();rhs=(x.conj()*h).sum()
        errors['native_H'].append(float(((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs())).detach()))
        derivative=torch.autograd.grad(value,x,y)[0]
        errors['current'].append(relative(derivative,adjoint))
        first=torch.autograd.grad(view.apply(x).abs().square().sum(),x,create_graph=True)[0]
        second=torch.autograd.grad(first,x,direction)[0]
        expected_second=2*scalar_moment_action(green,gradient,m,
            scalar_moment_action(green,gradient,m,direction),adjoint=True)
        errors['higher'].append(relative(second,expected_second))
    # All48 translations use two NEW bounded batches, without a new reference.
    batch_errors=[]
    for start in (0,32):
        ops=views[start:start+32];n=len(ops)
        bx=x.detach()[:3,0].unsqueeze(1).expand(-1,n,-1,-1,-1)
        by=y[:3,0].unsqueeze(1).expand_as(bx)
        with torch.no_grad():
            f=signed_axis_compact_moment_actions(ops,bx)
            h=signed_axis_compact_moment_actions(ops,by,adjoint=True)
        ef=torch.stack([v[:3,0] for v in expected_forward[start:start+n]],dim=1)
        eh=torch.stack([v[:3,0] for v in expected_adjoint[start:start+n]],dim=1)
        batch_errors.extend((relative(f,ef),relative(h,eh)))
    maximum={k:max(v) for k,v in errors.items()}
    maximum['batch']=max(batch_errors)
    (tmp_path/'signed-axis-native.json').write_text(json.dumps(dict(
        completed=True,order=order,row_batch=row_batch,transforms=48,
        original_reference_constructed_once=True,reference_artifact=str(artifact),
        reference_sha256=digest,errors=maximum,relative_gate=1e-11,
        owner_native_bytes=owner.retained_bytes,view_additional_owned_bytes=0,
        independent_original_owner_bytes=48*owner.retained_bytes,
        saved_forward_tensors=0,original_kernel_algebra_is_not_independent_EM=True,
        public_integration_or_f30_qualified=False,whole_peak_qualified=False,
        production_qualification=False),indent=2)+'\n')
    assert max(maximum.values())<1e-11
    assert hashlib.sha256(artifact.read_bytes()).hexdigest()==digest


@pytest.mark.parametrize('kind',['axes','signs','coefficient_alias','tensor_version','gradient_flag','backward_version'])
def test_new_signed_axis_owner_and_view_mutation_refused(kind):
    op=CompactElectromagneticM2L(displacement(),medium(),1,row_batch=2)
    view=SignedAxisCompactElectromagneticM2L(op,(2,0,1),(-1,1,-1))
    if kind=='backward_version':
        x=torch.ones((2,op.count,4),device='cuda',dtype=torch.complex128,requires_grad=True)
        value=view.apply(x);op.coefficients.add_(0)
        with pytest.raises(ValueError,match='changed'):
            torch.autograd.grad(value,x,torch.ones_like(value))
        return
    if kind=='axes':view.axis_permutation=(0,1,2)
    elif kind=='signs':view.axis_signs=(1,1,1)
    elif kind=='coefficient_alias':view.coefficients=op.coefficients.clone()
    elif kind=='tensor_version':op.coefficients.add_(0)
    else:op.coefficients.requires_grad_()
    with pytest.raises(ValueError,match='changed'):
        view.apply(torch.ones((2,op.count,4),device='cuda',dtype=torch.complex128))


def test_new_signed_axis_preallocation_budgets_and_invalid_bindings(monkeypatch):
    op=CompactElectromagneticM2L(displacement(),medium(),2,row_batch=3)
    for p,s in [((0,0,2),(1,1,1)),([0,1,2],(1,1,1)),((0,1,2),(True,1,1)),((0,1,2),(1,0,1))]:
        with pytest.raises(ValueError):SignedAxisCompactElectromagneticM2L(op,p,s)
    with pytest.raises(TypeError):SignedAxisCompactElectromagneticM2L(op.reversed_translation(),(0,1,2),(1,1,1))
    views=(SignedAxisCompactElectromagneticM2L(op,(2,1,0),(1,-1,1)),)
    value=torch.ones((1,1,2,op.count,4),device='cuda',dtype=torch.complex128)
    def forbidden(*args,**kwargs):raise AssertionError('Invalid budget reached temporary coefficient allocation')
    monkeypatch.setattr(SignedAxisCompactElectromagneticM2L,'_temporary_original_translation',forbidden)
    with monkeypatch.context() as finite_guard:
        finite_guard.setattr(torch,'isfinite',forbidden)
        with pytest.raises(MemoryError,match='before jets'):
            signed_axis_compact_moment_actions(views,value,workspace_budget_bytes=1)
        for budget in (True,0,128*1024**2+1):
            with pytest.raises(ValueError):signed_axis_compact_moment_actions(views,value,workspace_budget_bytes=budget)
        with pytest.raises(ValueError,match='fixed-current'):
            signed_axis_compact_moment_actions(views,value.requires_grad_())
    r=displacement();centers=torch.stack((r,r))
    with pytest.raises(ValueError,match='nonzero'):signed_axis_displacement_catalog(centers,[(0,1)])
    with pytest.raises(ValueError,match='physical'):signed_axis_displacement_catalog(centers,[(0,True)])
