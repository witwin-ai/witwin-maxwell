"""Original radial storage/composition controls with saved native references."""
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path

import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.fmm_capacity import storage_bytes
from witwin.maxwell.mom.scalar_moment_action import scalar_moment_action
from witwin.maxwell.mom.fmm_compact import (RadialSeriesElectromagneticM2L,
    radial_series_moment_actions,radial_series_batch_plan,_radial_series_temporary_translations)
from witwin.maxwell.mom.fmm_storage import radial_displacement_catalog,radial_series_m2l_owner_bytes,m2l_storage_bytes

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def _safe(value):
    if isinstance(value,float) and not math.isfinite(value):return dict(finite=False,native_value=repr(value))
    if isinstance(value,dict):return {str(k):_safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_safe(v) for v in value]
    return value


def write(path,value):
    path.write_text(json.dumps(_safe(value),indent=2,allow_nan=False)+'\n')


def medium():
    return compile_medium(mw.Material(eps_r=2.6+.017j,mu_r=1.41,sigma_m=.009),2.11e8)


def delta():
    return torch.tensor([.4375,-.28125,.15625],device='cuda',dtype=torch.float64)


def test_new_radius_catalog_equal_length_nonorbit_and_one_ulp(tmp_path):
    base=torch.tensor([[.1875,.25,0.],[.3125,0.,0.],[0.,-.3125,0.],[-.1875,0.,.25]],device='cuda',dtype=torch.float64)
    changed=base[1].clone();changed[0]=torch.nextafter(changed[0],changed.new_tensor(math.inf))
    vectors=torch.cat((base.repeat(264,1),changed[None]))
    owners,bindings=radial_displacement_catalog(vectors)
    radius=torch.linalg.vector_norm(vectors,dim=-1)
    assert owners==(0,1056) and len(bindings)==1057
    assert bindings[:-1]==(0,)*1056 and bindings[-1]==1
    assert not torch.equal(base[0].abs().sort().values,base[1].abs().sort().values)
    torch.testing.assert_close(radius,torch.stack([radius[owners[i]] for i in bindings]),rtol=0,atol=0)
    write(tmp_path/'radial-catalog.json',dict(completed=True,pairs=1057,native_radius_owners=2,
        equal_length_non_signed_axis_orbits=True,one_ulp_distinct=True,chunk_limit=1024,
        physical_vectors_retained=True,production_qualification=False))


@pytest.mark.parametrize('order,index,row_batch',[(0,0,1),(4,1,7),(7,2,11)])
def test_new_radial_composition_all48_saved_native_F_H_and_gradients(order,index,row_batch,tmp_path):
    location=os.environ.get('GSP_RADIAL_ORIGINAL_REFERENCE_'+str(order))
    if location is None:pytest.skip('Pinned closed original reference required; never regenerated')
    path=Path(location);assert path.is_file()
    # Declare the original physical vectors/material independently before load.
    r,m=delta(),medium()
    transforms=tuple((p,s) for p in itertools.permutations(range(3)) for s in itertools.product((-1,1),repeat=3))
    vectors=torch.stack([r[list(p)]*r.new_tensor(s) for p,s in transforms])
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest==os.environ['GSP_RADIAL_ORIGINAL_REFERENCE_'+str(order)+'_SHA256']
    saved=torch.load(path,map_location='cuda',weights_only=False)
    assert saved['order']==order and saved['frequency_Hz']==2.11e8
    torch.testing.assert_close(saved['displacements'],vectors,rtol=0,atol=0)
    views=RadialSeriesElectromagneticM2L.create_many(vectors,m,order,row_batch=row_batch)
    representatives,bindings=radial_displacement_catalog(vectors)
    owned=storage_bytes(t for op in views for t in op.tensors())
    assert owned==len(representatives)*radial_series_m2l_owner_bytes(order)+48*24
    assert len({id(op.owner) for op in views})==len(representatives)
    generator=torch.Generator(device='cuda').manual_seed(248091+order)
    x=torch.randn((3,48,2,views[0].count,4),device='cuda',dtype=torch.complex128,generator=generator)
    dual=torch.randn(x.shape,device='cuda',dtype=x.dtype,generator=generator)
    relative=lambda a,b:float(((a-b).norm()/b.norm()).detach())
    coefficients=[];forward=[];adjoint=[];expected_F=[];expected_H=[];plans=[]
    for start in (0,32):
        ops=views[start:start+32];n=len(ops)
        plan=radial_series_batch_plan(order,n,3,row_batch);plans.append(plan)
        # Inspect original full coefficients only inside an admitted chunk.
        for first in range(0,n,plan['admitted_pair_chunk']):
            native=_radial_series_temporary_translations(ops[first:first+plan['admitted_pair_chunk']])
            coefficients.extend(op.coefficients for op in native)
        forward.append(radial_series_moment_actions(ops,x[:,start:start+n]))
        adjoint.append(radial_series_moment_actions(ops,dual[:,start:start+n],adjoint=True))
        for i in range(start,start+n):
            expected_F.append(scalar_moment_action(saved['green'][i],saved['gradient'][i],m,x[:,i]))
            expected_H.append(scalar_moment_action(saved['green'][i],saved['gradient'][i],m,dual[:,i],adjoint=True))
    coefficients=torch.stack(coefficients);forward=torch.cat(forward,1);adjoint=torch.cat(adjoint,1)
    expected_F=torch.stack(expected_F,1);expected_H=torch.stack(expected_H,1)
    lhs,rhs=(forward.conj()*dual).sum(),(x.conj()*adjoint).sum()
    errors=dict(coefficients=relative(coefficients,saved['original_independent_coefficients']),
        F=relative(forward,expected_F),H=relative(adjoint,expected_H),
        native_H=float(((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs())).detach()))
    state=tmp_path/'radial-saved-reference-native-state.pt'
    torch.save(dict(coefficients=coefficients,x=x,dual=dual,forward=forward,adjoint=adjoint),state)
    raw=dict(completed=False,order=order,errors=errors,saved_original_reference_sha256=digest,
        original_reference_regenerated=False,actual_native_state=str(state),
        actual_native_state_sha256=hashlib.sha256(state.read_bytes()).hexdigest(),
        native_owned_bytes=owned,native_radius_owners=len(representatives),logical_translations=48,
        full_cartesian_coefficients_retained=False,batch_plans=plans,production_qualification=False)
    evidence=tmp_path/'radial-native.json';write(evidence,raw)
    gradient_errors=[];captured_count=0;states=[]
    for i in (0,17,47):
        z=x[:,i].detach().requires_grad_();h=dual[:,i];op=views[i];captured=[]
        with torch.autograd.graph.saved_tensors_hooks(lambda t:captured.append(t) or t,lambda t:t):value=op.apply(z)
        gradient=torch.autograd.grad(value,z,h)[0]
        first=torch.autograd.grad(op.apply(z).abs().square().sum(),z,create_graph=True)[0]
        second=torch.autograd.grad(first,z,h)[0]
        expected_second=2*scalar_moment_action(saved['green'][i],saved['gradient'][i],m,
            scalar_moment_action(saved['green'][i],saved['gradient'][i],m,h),adjoint=True)
        gradient_errors.extend((relative(value,expected_F[:,i]),relative(gradient,expected_H[:,i]),relative(second,expected_second)))
        captured_count+=len(captured)
        states.append(dict(forward=value.detach(),gradient=gradient.detach(),higher=second.detach()))
    gradients=tmp_path/'radial-gradient-native-state.pt';torch.save(states,gradients)
    raw.update(gradient_errors=gradient_errors,gradient_max=max(gradient_errors),saved_forward_tensors=captured_count,
        actual_gradient_state=str(gradients),actual_gradient_state_sha256=hashlib.sha256(gradients.read_bytes()).hexdigest(),
        errors_finite=all(math.isfinite(v) for v in (*errors.values(),*gradient_errors)),relative_gate=1e-11)
    write(evidence,raw)
    assert raw['errors_finite'] and max((*errors.values(),*gradient_errors))<1e-11 and captured_count==0
    raw['completed']=True;write(evidence,raw)


def test_new_p12_original_equal_radius_nonorbit_and_requested_B32_split(tmp_path):
    location=os.environ.get('GSP_RADIAL_ORIGINAL_EXPANSION')
    if location is None:pytest.skip('Pinned whole original expansion required')
    original=Path(location);assert hashlib.sha256(original.read_bytes()).hexdigest()==os.environ['GSP_RADIAL_ORIGINAL_EXPANSION_SHA256']
    spec=importlib.util.spec_from_file_location('_radial_whole_original_expansion',original)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    vectors=torch.tensor([[.1875,.25,0.],[.3125,0.,0.],[0.,-.3125,0.],[-.1875,0.,.25]],device='cuda',dtype=torch.float64)
    m=compile_medium(mw.Material(eps_r=3.2+.011j,mu_r=1.37,sigma_m=.008),197e6)
    green,gradient=module.multipole_to_local(vectors,m.k,12,table_budget_bytes=128*1024**2)
    coefficients=module.helmholtz_taylor(vectors,m.k,25)
    reference=tmp_path/'radial-new-p12-original-native-reference.pt'
    torch.save(dict(vectors=vectors,green=green,gradient=gradient,coefficients=coefficients,
        frequency_Hz=197e6,order=12,original_reference_constructed_once=True),reference)
    views=RadialSeriesElectromagneticM2L.create_many(vectors.repeat(8,1),m,12,row_batch=16)
    assert len({id(op.owner) for op in views})==1
    plan=radial_series_batch_plan(12,32,4,16)
    assert 1<=plan['admitted_pair_chunk']<32 and plan['chunks']>1
    generator=torch.Generator(device='cuda').manual_seed(643971)
    value=torch.randn((4,32,2,455,4),device='cuda',dtype=torch.complex128,generator=generator)
    dual=torch.randn(value.shape,device='cuda',dtype=value.dtype,generator=generator)
    result=radial_series_moment_actions(views,value)
    hermitian=radial_series_moment_actions(views,dual,adjoint=True)
    expected=torch.stack([scalar_moment_action(green[i%4],gradient[i%4].movedim(0,-1),m,value[:,i]) for i in range(32)],1)
    expected_H=torch.stack([scalar_moment_action(green[i%4],gradient[i%4].movedim(0,-1),m,dual[:,i],adjoint=True) for i in range(32)],1)
    relative=lambda a,b:float(((a-b).norm()/b.norm()).detach())
    errors=dict(F=relative(result,expected),H=relative(hermitian,expected_H))
    states=tmp_path/'radial-p12-actual-native-state.pt';torch.save(dict(value=value,dual=dual,F=result,H=hermitian),states)
    raw=dict(completed=False,order=12,requested_pairs=32,RHS=4,plan=plan,errors=errors,
        errors_finite=all(math.isfinite(v) for v in errors.values()),relative_gate=1e-11,
        original_new_reference=str(reference),original_new_reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        actual_native_state=str(states),actual_native_state_sha256=hashlib.sha256(states.read_bytes()).hexdigest(),
        one_original_radius_owner=True,native_owned_bytes=storage_bytes(t for op in views for t in op.tensors()),
        original_compact_unshared_bytes=32*m2l_storage_bytes(12,8,'compact'),
        public_f30_whole_resources_qualified=False,production_qualification=False)
    write(tmp_path/'radial-p12.json',raw)
    assert raw['errors_finite'] and max(errors.values())<1e-11
    raw['completed']=True;write(tmp_path/'radial-p12.json',raw)


@pytest.mark.parametrize('kind',('radial_version','radius_version','displacement_version','owner_identity',
    'view_alias','radial_gradient_flag','layout_version','view_budget','view_order','backward_version'))
def test_new_radial_owner_and_view_mutations_refused(kind):
    op=RadialSeriesElectromagneticM2L.create_many(delta()[None],medium(),1,row_batch=2)[0]
    if kind=='backward_version':
        value=torch.ones((2,4,4),device='cuda',dtype=torch.complex128,requires_grad=True)
        result=op.apply(value);op.owner.radial.add_(0)
        with pytest.raises(ValueError,match='changed'):torch.autograd.grad(result,value,torch.ones_like(result))
        return
    if kind=='radial_version':op.owner.radial.add_(0)
    elif kind=='radius_version':op.owner.radius.add_(0)
    elif kind=='displacement_version':op.displacement.add_(0)
    elif kind=='owner_identity':op.owner=RadialSeriesElectromagneticM2L.create_many(delta()[None],medium(),1,row_batch=2)[0].owner
    elif kind=='view_alias':op.displacement=op.displacement.clone()
    elif kind=='radial_gradient_flag':op.owner.radial.requires_grad_()
    elif kind=='layout_version':op.owner.binomial.add_(0)
    elif kind=='view_budget':op.workspace_budget_bytes=1
    elif kind=='view_order':op.order=2
    with pytest.raises(ValueError,match='changed'):op.apply(torch.ones((2,4,4),device='cuda',dtype=torch.complex128))


def test_new_radial_retained_and_workspace_refuse_before_coefficients(monkeypatch):
    import witwin.maxwell.mom.fmm_compact as compact
    r,m=delta(),medium();vectors=torch.stack((r,-r))
    required=radial_series_m2l_owner_bytes(3)+2*24
    def forbidden(*a,**kw):pytest.fail('Refused radial capacity reached coefficient composition')
    with monkeypatch.context() as patch:
        patch.setattr(compact._RadialSeriesM2LOwner,'__init__',forbidden)
        with pytest.raises(MemoryError,match='before original radial'):
            RadialSeriesElectromagneticM2L.create_many(vectors,m,3,retained_budget_bytes=required-1)
    views=RadialSeriesElectromagneticM2L.create_many(vectors,m,3)
    value=torch.ones((1,2,2,20,4),device='cuda',dtype=torch.complex128)
    monkeypatch.setattr(compact,'_radial_series_temporary_translations',forbidden)
    with pytest.raises(MemoryError,match='before original composition'):
        radial_series_moment_actions(views,value,workspace_budget_bytes=1)
    for budget in (True,0,128*1024**2+1):
        with pytest.raises(ValueError):radial_series_moment_actions(views,value,workspace_budget_bytes=budget)
    with pytest.raises(ValueError,match='fixed-current'):
        radial_series_moment_actions(views,value.requires_grad_())
    bad=delta().clone();bad[0]=torch.nextafter(bad[0],bad.new_tensor(math.inf))
    # Change the radius directly along a single axis; a vector-component ULP
    # need not change a rounded native norm and must not be asserted to do so.
    bad=torch.tensor([float(views[0].owner.radius)*2,0.,0.],device='cuda',dtype=torch.float64)
    with pytest.raises(ValueError,match='bit-identical'):RadialSeriesElectromagneticM2L(views[0].owner,bad)


def test_new_radial_standalone_large_RHS_finite_checks_remain_in_admitted_chunks(monkeypatch,tmp_path):
    op=RadialSeriesElectromagneticM2L.create_many(delta()[None],medium(),0,row_batch=1)[0]
    value=torch.ones((257,2,1,4),device='cuda',dtype=torch.complex128)
    original=torch.isfinite;observed=[]
    def bounded_finite(tensor):
        observed.append(tuple(tensor.shape))
        assert tensor.numel()<=4*2*op.count*4,'Unadmitted full-RHS finite-mask allocation'
        return original(tensor)
    monkeypatch.setattr(torch,'isfinite',bounded_finite)
    result=op.apply(value)
    assert result.shape==value.shape and observed
    state=tmp_path/'radial-large-rhs-state.pt';torch.save(dict(value=value,result=result),state)
    write(tmp_path/'radial-large-rhs.json',dict(completed=True,RHS=257,admitted_RHS_chunk=4,
        max_finite_mask_elements=max(math.prod(shape) for shape in observed),observed_finite_shapes=observed,
        native_state=str(state),native_state_sha256=hashlib.sha256(state.read_bytes()).hexdigest(),
        caller_and_final_output_excluded=True,production_qualification=False))


@pytest.mark.parametrize('kind',('zero','nonfinite','float32','gradient','invalid_order','invalid_rows','invalid_budget'))
def test_new_radial_factory_invalid_inputs_refused(kind):
    vectors=delta()[None];options={}
    if kind=='zero':vectors=torch.zeros_like(vectors)
    elif kind=='nonfinite':vectors=vectors.clone();vectors[0,1]=math.nan
    elif kind=='float32':vectors=vectors.float()
    elif kind=='gradient':vectors=vectors.requires_grad_()
    elif kind=='invalid_order':options['order']=13
    elif kind=='invalid_rows':options['row_batch']=17
    else:options['retained_budget_bytes']=True
    order=options.pop('order',2)
    with pytest.raises(ValueError):RadialSeriesElectromagneticM2L.create_many(vectors,medium(),order,**options)
