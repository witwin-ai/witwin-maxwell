"""Bounded native cache policy; saved original physics is never regenerated."""
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
from witwin.maxwell.mom.operator import IntegrationConfig, PreparedRWGGeometry
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.nearfield import scalar_regional_channels
from witwin.maxwell.mom.packed_near import (
    BoundedCachedReciprocalScalarNearChannels as Cached,
    bounded_original_near_cache_plan,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def _relative(a, b):
    return float(((a-b).norm()/b.norm()).detach())


def _layout(groups, edges):
    maps = [torch.cat((g,g+edges)) for g in groups]
    return [(a,b+len(groups),ra,cb+2*edges)
            for a,ra in enumerate(maps) for b,cb in enumerate(maps)]


def _embedded(op, value, adjoint=False):
    zero = torch.zeros_like(value)
    result = op.action(torch.cat((value,zero) if adjoint else (zero,value),dim=-1),
                       adjoint=adjoint)
    return result[...,op.test_size:] if adjoint else result[...,:op.test_size]


def _synthetic(cache=768, factory_failure=None, **options):
    groups = [torch.tensor(g,device='cuda') for g in ([4,1],[5,0],[3,2])]
    medium = compile_medium(mw.Material(eps_r=2.1,mu_r=1.31),149e6)
    physical = torch.ones(1,device='cuda',dtype=torch.float64)
    calls = []
    def factory(a,b,rows,cols):
        calls.append((a,b))
        r = rows[:2].to(torch.float64)+1
        c = (cols[:2]-12).to(torch.float64)+1
        d = (r[:,None]*c[None]+.2j*(r[:,None]+c[None])).to(torch.complex128)
        k = (.3*(r[:,None]+c[None])+.1j*r[:,None]*c[None]).to(torch.complex128)
        if factory_failure is not None:
            return factory_failure(d,k,len(calls))
        return d,k
    settings = dict(source_node_offset=3,medium=medium,integration=IntegrationConfig(),
        device='cuda',dtype=torch.complex128,physical_guard=lambda:None,
        physical_inputs=(physical,),action_batch=8,coefficient_workspace_budget_bytes=384,
        coefficient_cache_budget_bytes=cache)
    settings.update(options)
    op = Cached.from_layout(_layout(groups,6),12,12,factory,**settings)
    r = torch.arange(1,7,device='cuda',dtype=torch.float64)
    d = r[:,None]*r[None]+.2j*(r[:,None]+r[None])
    k = .3*(r[:,None]+r[None])+.1j*r[:,None]*r[None]
    a,b = 1j*medium.omega*medium.epsilon,-1j*medium.omega*medium.mu
    reference = torch.cat((torch.cat((k,a*d),dim=1),torch.cat((b*d,k),dim=1)),dim=0)
    return op,physical,calls,reference


def test_new_cached_original_columns_native_H_and_two_current_derivatives(monkeypatch,tmp_path):
    # Declare the identical physical problem before opening the saved reference.
    vertices = torch.tensor([[.003,.001,-.002],[.026,.002,.001],
                             [.005,.031,.002],[.004,.006,.033]],device='cuda',dtype=torch.float64)
    volume = TetrahedralVolumeMesh(vertices,torch.tensor([[0,1,2,3]],device='cuda'),region_id=17)
    basis = RWGBasis(volume.boundary)
    assert basis.n_edges == 6
    geometry = PreparedRWGGeometry(basis)
    medium = compile_medium(mw.Material(eps_r=2.35,mu_r=1.47,sigma_e=.006,sigma_m=.013),137e6)
    integration = IntegrationConfig(order=4,singular_order=3,pair_batch=16,far_pair_batch=32)
    groups = [torch.tensor(g,device='cuda') for g in ([5],[1,3],[4,0,2])]
    calls = []
    def factory(a,b,rows,cols):
        calls.append((a,b))
        d,k,_ = scalar_regional_channels(basis,basis,rows[:len(rows)//2],
            cols[:len(cols)//2]-12,medium,integration,geometry=(geometry,geometry))
        return d,k
    op = Cached.from_layout(_layout(groups,6),12,12,factory,source_node_offset=3,
        medium=medium,integration=integration,device='cuda',dtype=torch.complex128,
        physical_guard=lambda:geometry.assert_current(basis),
        physical_inputs=(*geometry.inputs(),*geometry.tensors()),action_batch=4,
        coefficient_workspace_budget_bytes=864,coefficient_cache_budget_bytes=800,
        retained_budget_bytes=1384)
    assert calls == [] and op.receipt()['cache_ready_blocks'] == 0
    location = os.environ.get('GSP_BOUNDED_CACHE_ORIGINAL_REFERENCE')
    if location is None:
        pytest.skip('Original SHA-bound weak reference must be supplied; never regenerated')
    path = Path(location)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == '265832b94589d656ad288151f4b8f7cb40b39c12bd2d13a08192c0c43dea7025'
    reference = torch.load(path,weights_only=False,map_location='cuda')['reference']
    assert reference.shape == (12,12) and reference.dtype == torch.complex128
    # Rebuilding the complete original weak matrix is forbidden in this epoch.
    import witwin.maxwell.mom.nearfield as nearfield
    monkeypatch.setattr(nearfield,'scalar_regional_block',lambda *a,**k:pytest.fail('Reference replay'))
    columns = _embedded(op,torch.eye(12,device='cuda',dtype=torch.complex128)).mT
    cold_calls = len(calls)
    generator = torch.Generator(device='cuda').manual_seed(273419)
    current = torch.randn((3,5,12),device='cuda',dtype=torch.complex128,
                          generator=generator).transpose(0,1).requires_grad_()
    dual = torch.randn(current.shape,device='cuda',dtype=torch.complex128,generator=generator)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t,lambda t:t):
        output = _embedded(op,current)
    native_H = _embedded(op,dual,True)
    grad, = torch.autograd.grad(output,current,dual)
    norm_output = _embedded(op,current)
    first, = torch.autograd.grad(norm_output.abs().square().sum(),current,create_graph=True)
    second, = torch.autograd.grad(first,current,dual)
    expected_second = 2*(dual@columns.mT)@columns.conj()
    lhs,rhs = (output.conj()*dual).sum(),(current.conj()*native_H).sum()
    errors = dict(columns=_relative(columns,reference),forward=_relative(output,current@reference.mT),
        original_H=_relative(native_H,dual@reference.conj()),
        native_H=float(((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs())).detach()),
        current_gradient=_relative(grad,native_H),higher_current_gradient=_relative(second,expected_second))
    gates = {k:(1e-8 if k in ('columns','forward','original_H') else 1e-11) for k in errors}
    # Preserve actual complete states before any acceptance assertion.
    torch.save(dict(columns=columns,current=current.detach(),dual=dual,output=output.detach(),
        native_H=native_H.detach(),gradient=grad.detach(),second=second.detach()),
        tmp_path/'cached-original-actual.pt')
    record = dict(completed=True,errors=errors,gates=gates,cold_factory_calls=cold_calls,
        warm_factory_calls=len(calls)-cold_calls,saved_forward_tensors=len(saved),
        original_reference_path=str(path),original_reference_sha256=digest,
        owned_bytes=sum(t.untyped_storage().nbytes() for t in op.tensors()),receipt=op.receipt(),
        no_complete_reference_replay=True,physical_qualification=False,production_qualification=False)
    (tmp_path/'cached-original.json').write_text(json.dumps(record,indent=2)+'\n')
    assert all(errors[k] < gates[k] for k in errors),errors
    assert cold_calls == 6 and len(calls) == cold_calls and saved == []
    assert record['owned_bytes'] == 1384
    assert record['receipt']['cache_ready_blocks'] == record['receipt']['cache_planned_blocks'] == 6


@pytest.mark.parametrize('budget,warm_calls,ready',((0,6,0),(127,6,0),(128,5,1),(767,1,5),(768,0,6)))
def test_new_cache_native_block_boundaries_and_diagonal_priority(budget,warm_calls,ready,tmp_path):
    op,_,calls,reference = _synthetic(budget)
    assert calls == []
    eye = torch.eye(12,device='cuda',dtype=torch.complex128)
    cold = _embedded(op,eye).mT
    cold_calls = len(calls)
    warm = _embedded(op,eye).mT
    receipt = op.receipt()
    if ready:
        assert op._cache_plan[0][1] == op._streamed._reverse_group_counts[0] == 3
    record = dict(completed=True,budget=budget,cold_factory_calls=cold_calls,
        warm_factory_calls=len(calls)-cold_calls,cold_error=_relative(cold,reference),
        warm_error=_relative(warm,reference),receipt=receipt,
        exact_native_copy=torch.equal(cold,warm),synthetic_scope_only=True,production_qualification=False)
    (tmp_path/'cache-boundaries.json').write_text(json.dumps(record,indent=2)+'\n')
    assert cold_calls == 6 and len(calls)-cold_calls == warm_calls
    assert receipt['cache_ready_blocks'] == ready
    assert receipt['retained_kernel_bytes'] == ready*128 <= budget
    assert receipt['retained_bytes'] == sum(t.untyped_storage().nbytes() for t in op.tensors())
    assert torch.equal(cold,warm) and record['cold_error'] < 1e-11


@pytest.mark.parametrize('warm', (False,True))
@pytest.mark.parametrize('mutation',('physical_version','physical_grad','original_factory',
    'streamed_factory','cache_version','cache_grad','cache_budget','map_version','ready','events'))
def test_new_cached_binding_changes_refuse_before_original_factory(warm,mutation):
    op,physical,calls,_ = _synthetic()
    value = torch.ones(24,device='cuda',dtype=torch.complex128)
    if warm:
        op.action(value)
    before = len(calls)
    if mutation == 'physical_version':physical.add_(.01)
    elif mutation == 'physical_grad':physical.requires_grad_()
    elif mutation == 'original_factory':op._original_factory = lambda *a:None
    elif mutation == 'streamed_factory':op._streamed._channel_factory = lambda *a:None
    elif mutation == 'cache_version':op._cache_arrays[0].fill_(0)
    elif mutation == 'cache_grad':op._cache_arrays[0].requires_grad_()
    elif mutation == 'cache_budget':op.coefficient_cache_budget_bytes += 1
    elif mutation == 'map_version':op._streamed.groups[0][0].add_(0)
    elif mutation == 'ready':op._ready = tuple(not x for x in op._ready)
    else:op._events = tuple(torch.cuda.Event() for _ in op._events)
    with pytest.raises(ValueError):op.action(value)
    assert len(calls) == before


@pytest.mark.parametrize('budget',(-1,True,1.5))
def test_new_invalid_cache_budgets_refuse_without_factory(budget):
    with pytest.raises(ValueError):_synthetic(budget)


def test_new_total_retained_budget_refuses_before_cache_allocation(monkeypatch):
    op,_,calls,_ = _synthetic()
    exact = op.receipt()['retained_bytes']
    assert calls == []
    original = torch.empty
    def reject_cache(shape,*args,**kwargs):
        if isinstance(shape,tuple) and len(shape) == 4 and shape[1] == 2:
            pytest.fail('Coefficient cache allocation before retained admission')
        return original(shape,*args,**kwargs)
    monkeypatch.setattr(torch,'empty',reject_cache)
    with pytest.raises(MemoryError,match='before coefficient cache allocation'):
        _synthetic(retained_budget_bytes=exact-1)


@pytest.mark.parametrize('failure',('nonfinite','gradient','oversized_backing'))
def test_new_invalid_original_fields_cannot_populate_cache(failure):
    def invalid(d,k,count):
        if failure == 'nonfinite':k[0,0] = complex(float('nan'),0)
        elif failure == 'gradient':d.requires_grad_()
        else:d = torch.ones((6,6),device='cuda',dtype=torch.complex128)[:2,:2]
        return d,k
    op,_,calls,_ = _synthetic(factory_failure=invalid)
    with pytest.raises(ValueError):op.action(torch.ones(24,device='cuda',dtype=torch.complex128))
    assert len(calls) == 1 and op.receipt()['cache_ready_blocks'] == 0
    assert all(event is None for event in op._events)


def test_new_partial_fill_failure_retries_on_another_stream_before_ready(tmp_path):
    def fail_once(d,k,count):
        if count == 2:
            raise RuntimeError('Injected original integration interruption')
        return d,k
    op,_,calls,reference = _synthetic(factory_failure=fail_once)
    eye = torch.eye(12,device='cuda',dtype=torch.complex128)
    first,second = torch.cuda.Stream(),torch.cuda.Stream()
    first.wait_stream(torch.cuda.current_stream())
    second.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(first):
        with pytest.raises(RuntimeError,match='Injected'):_embedded(op,eye)
    # The off-diagonal selected prefix has one copied block but is unreadable.
    assert op.receipt()['cache_ready_blocks'] == 0
    assert sum(event is not None for event in op._events) == 1
    with torch.cuda.stream(second):
        retry = _embedded(op,eye).mT
        retry_calls = len(calls)
        warm = _embedded(op,eye).mT
    second.synchronize()
    record = dict(completed=True,failed_factory_calls=2,retry_factory_calls=retry_calls-2,
        warm_factory_calls=len(calls)-retry_calls,error=_relative(retry,reference),
        exact_retry_warm=torch.equal(retry,warm),receipt=op.receipt(),production_qualification=False)
    (tmp_path/'partial-fill-retry.json').write_text(json.dumps(record,indent=2)+'\n')
    assert retry_calls == 8 and len(calls) == retry_calls
    assert op.receipt()['cache_ready_blocks'] == 6
    assert torch.equal(retry,warm) and record['error'] < 1e-11


def test_new_completed_cache_cross_stream_native_actions_without_host_barrier(tmp_path):
    op,_,calls,reference = _synthetic()
    eye = torch.eye(12,device='cuda',dtype=torch.complex128)
    first,second = torch.cuda.Stream(),torch.cuda.Stream()
    first.wait_stream(torch.cuda.current_stream())
    second.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(first):cold = _embedded(op,eye).mT
    cold_calls = len(calls)
    # No host synchronization or second.wait_stream(first) before cached reads.
    with torch.cuda.stream(second):warm = _embedded(op,eye).mT
    second.synchronize()
    record = dict(completed=True,cold_factory_calls=cold_calls,warm_factory_calls=len(calls)-cold_calls,
        error=_relative(warm,reference),exact_cold_warm=torch.equal(cold,warm),
        receipt=op.receipt(),production_qualification=False)
    (tmp_path/'completed-cross-stream.json').write_text(json.dumps(record,indent=2)+'\n')
    assert cold_calls == 6 and len(calls) == cold_calls
    assert torch.equal(cold,warm) and record['error'] < 1e-11


def test_new_host_plan_complete_native_prefixes_without_numerical_evaluation():
    shapes,reverse = ((4,1,1),(3,2,2)),(2,2)
    segments,total = bounded_original_near_cache_plan(shapes,reverse,224)
    assert segments == ((0,2,4,64),(1,2,3,128),(0,0,1,32)) and total == 224
    for bad_shapes,bad_reverse in (((),()),(((1,0,2),),(0,)),(((1,2,2),),(2,))):
        with pytest.raises(ValueError):bounded_original_near_cache_plan(bad_shapes,bad_reverse,224)
