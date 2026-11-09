"""Current bounded FAR resource intervals on both originally admitted large cases."""
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.fmm_system import MultipoleSystemActions
from witwin.maxwell.mom.fmm_capacity import multipole_action_tensors, storage_bytes
from test_scalar_dk_original_full_resource import _WHOLE_PEAK, capture_peak, measured

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('family,frequency,order,rule,owned',[
    ('fixed_ka',3e8,11,'positive_tail_min_v1',585948608),
    ('growing_ka',2.4e9,12,'direct_first_wave_min_v1',668235432)],ids=['fixed_ka','growing_ka'])
def test_current_bounded_compact_original_complete_resource_interval(family,frequency,order,rule,owned,tmp_path,monkeypatch):
    path=os.environ.get('GSP_BOUNDED_COMPACT_RESOURCE_INPUT')
    digest=os.environ.get('GSP_BOUNDED_COMPACT_RESOURCE_INPUT_SHA256')
    if path is None and digest is None:
        pytest.skip('Requires frozen original admission and resource provenance')
    assert path and digest and hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    inputs=json.loads(Path(path).read_text());selection=inputs['cases'][family]
    assert selection['frequency_Hz']==frequency and selection['selected_order']==order
    assert selection['kernel_bound_rule']==rule and selection['original_owned_bytes']==owned
    for field in ('admission','original_shared_resource'):
        source=Path(selection[field]['path'])
        assert hashlib.sha256(source.read_bytes()).hexdigest()==selection[field]['sha256']
    admission=json.loads(Path(selection['admission']['path']).read_text())
    final=admission['order_plan']['attempts'][-1]
    assert admission['order_admitted'] is True and admission['edges']==3840
    assert admission['family']==family and admission['frequency_Hz']==frequency
    assert admission['order_plan']['selected_order']==order and final['kernel_pass'] is True
    assert admission['order_plan']['kernel_bound_rule']==rule
    assert admission['order_plan']['kernel_budget']==dict(green_absolute=1e-6,gradient_component_absolute=1e-5)
    assert final['unavailable_bounds']==0 and final['green_absolute_max']<=1e-6
    assert final['gradient_component_absolute_max']<=1e-5
    original=json.loads(Path(selection['original_shared_resource']['path']).read_text())
    assert original['actual_owned_retained_bytes']==owned
    assert original['whole_setup_actions_backward_peak_measured'] is True
    assert original['adjoint_relative_error']<1e-11 and original['current_gradient_relative_error']<1e-11
    def forbidden(*args,**kwargs):
        raise AssertionError('Original order planning was replayed')
    monkeypatch.setattr('witwin.maxwell.mom.fmm_system.plan_multipole_order',forbidden)
    import witwin.maxwell.mom.fmm_compact as compact
    action=compact.compact_moment_actions
    observed=dict(calls=0,max_pairs=0,max_rhs=0,reverse_calls=0)
    def tracked(operators,value,**kwargs):
        assert len(operators)<=32 and len(value)<=4
        observed['calls']+=1
        observed['max_pairs']=max(observed['max_pairs'],len(operators))
        observed['max_rhs']=max(observed['max_rhs'],len(value))
        observed['reverse_calls']+=int(any(hasattr(o,'_reverse_owner') for o in operators))
        return action(operators,value,**kwargs)
    monkeypatch.setattr(compact,'compact_moment_actions',tracked)
    def build():
        exterior=compile_medium(mw.Material(),frequency)
        interior=compile_medium(mw.Material(eps_r=2.),frequency)
        system=SurfaceSystem.create([(i,str(i),RWGBasis(sphere_mesh(radius=.01,center=(x,0.,0.),
            subdivisions=3,device='cuda',dtype=torch.float64)),interior) for i,x in enumerate((-.5,.5))])
        assert system.total_edges==3840
        provider=ScalarSystemBlocks(system,exterior,IntegrationConfig(order=4,singular_order=3))
        return MultipoleSystemActions(provider,order=order,kernel_bound_rule=rule,
            kernel_error_budget=mw.KernelErrorBudget(1e-6,1e-5),leaf_size=30,separation_ratio=.25,
            table_storage='compact_shared_displacement_v1',shift_storage='separable_v1',
            near_storage='scalar_reciprocal_dk_v1',far_action_batch=32,m2l_row_batch=8,
            action_retained_budget_bytes=2*1024**3,table_budget_bytes=1024**3,
            near_budget_bytes=1024**3,projection_budget_bytes=1024**3)
    # Original two RHS/seed, geometry, quadrature, kernel and resource gates
    # stay fixed. The changed execution is measured from fresh setup onward.
    torch.empty(1,device='cuda');torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    _WHOLE_PEAK.clear()
    _WHOLE_PEAK.update(baseline_allocated_bytes=torch.cuda.memory_allocated(),
        baseline_reserved_bytes=torch.cuda.memory_reserved(),
        scope='process allocator before construction through final current/finite checks; all warmups, differentiable forward and reset gaps included; no solve or teardown')
    record=dict(version='bounded_compact_original_full_resource_v1',family=family,frequency_Hz=frequency,
        edges=3840,selected_order=order,far_action_batch=32,resource_input_sha256=digest,
        admission_sha256=selection['admission']['sha256'],original_shared_resource_sha256=selection['original_shared_resource']['sha256'],
        original_order_search_replayed=False,old_public_solve_or_archive_replayed=False,
        same_N_full_dense_bytes=943718400,phase='setup_start',physical_qualification=False,
        complexity_qualification=False,memory_crossover_qualified=False,whole_peak_qualified=False,
        production_qualification=False)
    def save():
        (tmp_path/'resource.json').write_text(json.dumps(record,indent=2)+'\n')
        print(dict(family=family,phase=record['phase']),flush=True)
    save();op,setup=measured(build)
    actual_owned=storage_bytes(multipole_action_tensors(op))
    record.update(phase='setup_done',setup=setup,capacity=op.capacity_plan,
        full_fmm_constructed=True,actual_owned_retained_bytes=actual_owned,
        logical_FAR_pairs=sum(len(r.far) for r in (op.exterior,*op.interiors)))
    save()
    assert actual_owned==owned==op.capacity_plan['planned_bytes']
    assert op.capacity_plan['categories']==original['capacity']['categories']
    assert record['logical_FAR_pairs']==final['far_pairs']==6738
    assert op.far_action_batch==32 and all(r.far_action_batch==32 for r in (op.exterior,*op.interiors))
    generator=torch.Generator(device='cuda').manual_seed(1729)
    x=torch.randn((2,op.shape[0]),generator=generator,device=op.device,dtype=op.dtype)
    y=torch.randn(x.shape,generator=generator,device=op.device,dtype=op.dtype)
    value,warmup_forward=measured(lambda:op.apply(x));del value
    value,warmup_adjoint=measured(lambda:op.adjoint(y));del value
    record.update(phase='warmups_done',warmup_forward=warmup_forward,warmup_adjoint=warmup_adjoint)
    save()
    applied,forward=measured(lambda:op.apply(x))
    adjoint,adjoint_timing=measured(lambda:op.adjoint(y))
    lhs=(applied.conj()*y).sum();rhs=(x.conj()*adjoint).sum()
    H_error=float((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs()).clamp_min(1e-300))
    record.update(phase='actions_done',forward=forward,adjoint=adjoint_timing,adjoint_relative_error=H_error)
    save()
    variable=x.detach().requires_grad_(True)
    output,differentiable_forward=measured(lambda:op.apply(variable))
    gradient,backward=measured(lambda:torch.autograd.grad(output,variable,y)[0])
    gradient_error=float((gradient-adjoint).norm()/adjoint.norm())
    finite=bool(torch.isfinite(applied).all() & torch.isfinite(adjoint).all() & torch.isfinite(output).all() & torch.isfinite(gradient).all())
    torch.cuda.synchronize();capture_peak()
    _WHOLE_PEAK.update(end_allocated_bytes=torch.cuda.memory_allocated(),end_reserved_bytes=torch.cuda.memory_reserved())
    interval=_WHOLE_PEAK.copy()
    record.update(phase='backward_done',differentiable_forward=differentiable_forward,backward=backward,
        current_gradient_relative_error=gradient_error,native_actions_finite=finite,
        native_actions_gate_passed=finite and H_error<1e-11 and gradient_error<1e-11,
        observed_compact_batches=observed.copy(),whole_setup_actions_backward_allocator=interval,
        whole_setup_actions_backward_peak_measured=True,
        measured_allocated_peak_below_same_N_dense=interval['peak_allocated_bytes']<943718400,
        measured_reserved_peak_below_same_N_dense=interval['peak_reserved_bytes']<943718400,
        limitations=['selected original large point; no all-size fit or crossover qualification',
                    'shared GPU timing is descriptive only','no solve, teardown, host peak, original matrix, fields or quadrature convergence'])
    save()
    assert finite and H_error<1e-11 and gradient_error<1e-11
    assert observed['max_pairs']==32 and observed['max_rhs']==2 and observed['reverse_calls']>0
    assert interval['peak_allocated_bytes']<943718400 and interval['peak_reserved_bytes']<943718400
