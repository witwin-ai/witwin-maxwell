"""Measure changed shared compact FAR tables and reciprocal near storage on the original admitted growing-ka problem."""
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
from witwin.maxwell.mom.fmm_capacity import multipole_action_tensors, storage_bytes, plan_multipole_storage
from test_scalar_dk_original_full_resource import _WHOLE_PEAK, capture_peak, measured

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_original_3840_growing_shared_setup_actions_and_current_backward(tmp_path, monkeypatch):
    required=('GSP_GROWING_FMM_ADMISSION','GSP_GROWING_FMM_ADMISSION_SHA256',
              'GSP_GROWING_FMM_RESOURCE','GSP_GROWING_FMM_RESOURCE_SHA256')
    if all(name not in os.environ for name in required):
        pytest.skip('Requires a completed original fixed-ka admission artifact')
    assert all(name in os.environ for name in required), 'Incomplete admission artifact environment'
    path=Path(os.environ['GSP_GROWING_FMM_ADMISSION'])
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest==os.environ['GSP_GROWING_FMM_ADMISSION_SHA256']
    admission=json.loads(path.read_text())
    assert admission['family']=='growing_ka' and admission['order_admitted'] is True
    assert admission['frequency_Hz']==2.4e9
    assert admission['capacity']['admitted'] is True and admission['edges']==3840
    plan=admission['order_plan'];order=plan['selected_order']
    assert plan['kernel_bound_rule']=='direct_first_wave_min_v1'
    assert plan['kernel_budget']==dict(green_absolute=1e-6,gradient_component_absolute=1e-5)
    assert admission['capacity']['budget_bytes']==2*1024**3
    assert plan['table_bytes']<=1024**3
    resource_path = Path(os.environ['GSP_GROWING_FMM_RESOURCE'])
    resource_digest = hashlib.sha256(resource_path.read_bytes()).hexdigest()
    assert resource_digest == os.environ['GSP_GROWING_FMM_RESOURCE_SHA256']
    original_resource = json.loads(resource_path.read_text())
    assert original_resource['phase'] == 'backward_done' and original_resource['full_fmm_constructed']
    assert original_resource['family']=='growing_ka' and original_resource['frequency_Hz']==2.4e9
    assert original_resource['selected_order']==order==12
    assert original_resource['capacity']['table_storage']=='compact'
    assert original_resource['actual_owned_retained_bytes']==1656499936
    assert original_resource['admission_sha256']==digest
    def forbidden(*args, **kwargs):
        raise AssertionError('Completed original order planning was repeated')
    monkeypatch.setattr('witwin.maxwell.mom.fmm_system.plan_multipole_order', forbidden)
    assert original_resource['adjoint_relative_error'] < 1e-11
    assert original_resource['current_gradient_relative_error'] < 1e-11
    def build():
        exterior=compile_medium(mw.Material(),2.4e9);interior=compile_medium(mw.Material(eps_r=2.),2.4e9)
        system=SurfaceSystem.create([(i,str(i),RWGBasis(sphere_mesh(radius=.01,center=(x,0,0),
            subdivisions=3,device='cuda',dtype=torch.float64)),interior) for i,x in enumerate((-.5,.5))])
        assert system.total_edges==3840
        provider=ScalarSystemBlocks(system,exterior,IntegrationConfig(order=4,singular_order=3))
        # Use the actual already-admitted selected order. Every constructed
        # coupling still independently enforces the same point-kernel budget.
        return MultipoleSystemActions(provider,order=order,kernel_bound_rule='direct_first_wave_min_v1',
            kernel_error_budget=mw.KernelErrorBudget(1e-6,1e-5),leaf_size=30,separation_ratio=.25,
            table_storage='compact_shared_displacement_v1',shift_storage='separable_v1',near_storage='scalar_reciprocal_dk_v1',action_retained_budget_bytes=2*1024**3,
            table_budget_bytes=1024**3,near_budget_bytes=1024**3,projection_budget_bytes=1024**3)
    torch.empty(1,device='cuda')
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    _WHOLE_PEAK.clear();_WHOLE_PEAK.update(baseline_allocated_bytes=torch.cuda.memory_allocated(), baseline_reserved_bytes=torch.cuda.memory_reserved(), scope='current process allocator from before construction through final current/finite checks, gaps and all warmups included')
    record=dict(version='original_growing_ka_shared_compact_full_resource_v1',table_storage='compact_shared_displacement_v1',near_storage='scalar_reciprocal_dk_v1',original_resource_sha256=resource_digest,phase='setup_start',family='growing_ka',edges=3840,frequency_Hz=2.4e9,
        selected_order=order,admission_sha256=digest,physical_qualification=False,
        complexity_qualification=False,whole_peak_qualified=False,full_fmm_constructed=False)
    def save():
        (tmp_path/'resource.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
    save();op,setup=measured(build)
    record.update(phase='setup_done',setup=setup,full_fmm_constructed=True,
        capacity=op.capacity_plan,actual_owned_retained_bytes=storage_bytes(multipole_action_tensors(op)),
        table_bytes=op.table_bytes,near_bytes=op.near_bytes,projection_bytes=op.projection_bytes)
    save()
    assert record['actual_owned_retained_bytes']==op.capacity_plan['planned_bytes']
    regions=(op.exterior,*op.interiors)
    trees=tuple(region.test.tree for region in regions)
    original_capacity=plan_multipole_storage(trees,order=order,separation_ratio=.25,budget_bytes=2*1024**3,
        table_storage='compact',shift_storage='separable_v1',near_storage='full_matrix')
    assert original_capacity['categories']==original_resource['capacity']['categories']
    unshared_reciprocal=plan_multipole_storage(trees,order=order,separation_ratio=.25,budget_bytes=2*1024**3,
        table_storage='compact',shift_storage='separable_v1',near_storage='scalar_reciprocal_dk_v1')
    near_savings=original_capacity['categories']['near']-op.near_bytes
    table_savings=original_capacity['categories']['m2l']-op.table_bytes
    assert near_savings>0 and table_savings>0
    assert original_resource['actual_owned_retained_bytes']-record['actual_owned_retained_bytes']==near_savings+table_savings
    near_receipts=[region.near.receipt() for region in regions]
    assert all(r['stored_blocks']<r['logical_blocks'] for r in near_receipts)
    record.update(original_full_matrix_compact_capacity=original_capacity,
        unshared_reciprocal_compact_capacity=unshared_reciprocal,near_receipts=near_receipts,
        only_near_and_FAR_table_storage_changed=True,original_order_planning_reused_not_repeated=True,
        actual_original_near_savings_bytes=near_savings,actual_original_FAR_table_savings_bytes=table_savings)
    for category,value in original_resource['capacity']['categories'].items():
        if category not in ('near','m2l'):
            assert op.capacity_plan['categories'][category]==value
        if category!='near':
            assert unshared_reciprocal['categories'][category]==value
    assert unshared_reciprocal['categories']['near']==op.near_bytes
    record.update(same_N_full_dense_bytes=943718400,
        actual_owned_below_same_N_dense=record['actual_owned_retained_bytes']<943718400,
        memory_crossover_qualified=False)
    assert original_capacity['categories']['m2l']==plan['table_bytes']
    assert record['table_bytes']<plan['table_bytes']
    logical=sum(len(r.far) for r in regions)
    owners=len({id(c._compact._reverse_owner if hasattr(c._compact,'_reverse_owner') else c._compact)
        for r in regions for _,_,c in r.far})
    assert owners<logical
    record.update(logical_FAR_pairs=logical,unique_FAR_coefficient_owners=owners)
    save()
    assert record['actual_owned_retained_bytes']<=2*1024**3
    assert sum(len(r.far) for r in (op.exterior,*op.interiors))==plan['attempts'][-1]['far_pairs']
    generator=torch.Generator(device='cuda').manual_seed(1729)
    x=torch.randn((2,op.shape[0]),generator=generator,device=op.device,dtype=op.dtype)
    y=torch.randn(x.shape,generator=generator,device=op.device,dtype=op.dtype)
    # One warmup; record subsequent actual forward, adjoint and first backward.
    warmup,warmup_forward=measured(lambda:op.apply(x));del warmup
    warmup,warmup_adjoint=measured(lambda:op.adjoint(y));del warmup
    record.update(warmup_forward=warmup_forward,warmup_adjoint=warmup_adjoint)
    applied,forward=measured(lambda:op.apply(x));adjoint,adjoint_timing=measured(lambda:op.adjoint(y))
    lhs=(applied.conj()*y).sum();rhs=(x.conj()*adjoint).sum()
    error=float((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs()).clamp_min(1e-300))
    record.update(phase='actions_done',forward=forward,adjoint=adjoint_timing,adjoint_relative_error=error)
    save();assert error<1e-11
    variable=x.detach().requires_grad_(True);output,differentiable_forward=measured(lambda:op.apply(variable))
    record.update(differentiable_forward=differentiable_forward)
    gradient,backward=measured(lambda:torch.autograd.grad(output,variable,y)[0])
    gradient_error=float((gradient-adjoint).norm()/adjoint.norm())
    record.update(phase='backward_done',backward=backward,current_gradient_relative_error=gradient_error,
        limitations=['single shared GPU process','growing-ka case only','no solve or original all-column error check',
            'absolute process peaks per measured phase, earlier tensors retained',
            'process allocator interval includes all warmups, differentiable forward and peak-reset gaps; no solve or teardown','no scaling/crossover/physical qualification'])
    save();assert gradient_error<1e-11
    assert bool(torch.isfinite(applied).all() & torch.isfinite(adjoint).all() & torch.isfinite(gradient).all())
    torch.cuda.synchronize();capture_peak()
    _WHOLE_PEAK.update(end_allocated_bytes=torch.cuda.memory_allocated(),end_reserved_bytes=torch.cuda.memory_reserved())
    record.update(whole_setup_actions_backward_allocator=_WHOLE_PEAK.copy(),whole_setup_actions_backward_peak_measured=True,
        measured_allocated_peak_below_same_N_dense=_WHOLE_PEAK['peak_allocated_bytes']<943718400,
        measured_reserved_peak_below_same_N_dense=_WHOLE_PEAK['peak_reserved_bytes']<943718400,
        historical_compact_per_phase_resources={k:original_resource[k] for k in ('setup','forward','adjoint','backward')},
        historical_compact_whole_interval_peak_measured=False,
        memory_crossover_qualified=False,whole_peak_qualified=False)
    save()
    print(dict(phase='resource_assertions_passed',full_fmm_constructed=True,physical_qualification=False),flush=True)
