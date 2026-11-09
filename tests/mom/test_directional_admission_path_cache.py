"""Reuse guarded automatic path products without changing map/action kernels."""
import json
import mpmath as mp
import pytest
import torch
from witwin.maxwell.mom.fmm_directional_moments import DirectionalRWGDAGProjection,directional_phase_path_workspace_bytes
from witwin.maxwell.mom.direct_capacity import storage_bytes
from test_directional_automatic_dag import _physical_plan,_select
from test_directional_dag_coupling import _mp_products

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.fixture(scope='module')
def owners():
    root=_physical_plan();plan=_select(root)
    opts=dict(quadrature_order=plan.quadrature_order,face_batch=plan.face_batch,rhs_batch=plan.rhs_batch,
        retained_budget_bytes=plan.retained_budget_bytes,tensor_workspace_budget_bytes=plan.tensor_workspace_budget_bytes,
        phase_relative_budget=plan.phase_relative_budget)
    args=(root.basis,root.tree,plan.carriers,plan.orders,plan.requests,plan.links,root.medium.k)
    bound=DirectionalRWGDAGProjection(*args,**opts,plan_guard=plan)
    uncached=DirectionalRWGDAGProjection(*args,**opts)
    return plan,bound,uncached


def test_borrowed_automatic_paths_match_uncached_actual_and_independent_all_ancestor_MP100(owners,tmp_path):
    plan,owner,uncached=owners
    assert owner.context_keys==uncached.context_keys==plan.context_keys
    assert owner.retained_bytes==uncached.retained_bytes==plan.planned_projection_retained_bytes
    assert storage_bytes(owner.tensors())==owner.retained_bytes
    original=uncached._ideal_phase_path_log_products(128*1024**2)
    borrowed=owner._ideal_phase_path_log_products(128*1024**2)
    assert borrowed is plan.path_products
    torch.testing.assert_close(borrowed,original,rtol=1e-11,atol=0)
    for _ in range(32):assert owner._ideal_phase_path_log_products(128*1024**2) is borrowed
    with pytest.raises(MemoryError):owner._ideal_phase_path_log_products(directional_phase_path_workspace_bytes(len(owner.node_counts))-1)
    errors=[]
    with mp.workdps(100):
        expected=_mp_products(owner)
        for i,value in enumerate(expected):
            delta=value-1;actual=mp.mpf(float(torch.expm1(borrowed[i])))
            if delta:errors.append(float(abs(actual/delta-1)))
            else:assert actual==0 and original[i]==0
    assert errors and max(errors)<1e-11
    (tmp_path/'paths.json').write_text(json.dumps(dict(all_ancestor_MP100_relative_errors=errors,mp_absolute_tolerance=0,
        borrowed_array_identity=True,repeated_requests=32,no_second_retained_array=True,
        retained_bytes=owner.retained_bytes,component_workspace=owner.planned_tensor_workspace_bytes,
        actual_support_and_keys_checked_before_maps=True,whole_resource_qualification=False,qualification=False),indent=2)+'\n')


def test_borrowed_path_modes_identity_and_saved_current_guard(owners,tmp_path):
    plan,owner,_=owners;trips=[]
    for name in ('mode','identity','links'):
        x=torch.ones((2,owner.tree.size),device='cuda',dtype=torch.complex128,requires_grad=True)
        value=owner.source_upward(x);loss=sum(t.abs().square().sum() for t in value if t is not None)
        if name=='mode':old=plan.path_products;old.requires_grad_(True)
        elif name=='identity':old=plan.path_products;plan.path_products=old.clone()
        else:old=plan.links;plan.links=old[::-1]
        try:
            with pytest.raises(ValueError):owner._ideal_phase_path_log_products(128*1024**2)
            with pytest.raises(ValueError):owner.receipt()
            with pytest.raises(ValueError):torch.autograd.grad(loss,x)
        finally:
            if name=='mode':old.requires_grad_(False)
            elif name=='identity':plan.path_products=old
            else:plan.links=old
        owner.assert_current();trips.append(name)
    x=torch.ones((2,owner.tree.size),device='cuda',dtype=torch.complex128,requires_grad=True)
    value=owner.receiver_upward(x);loss=sum(t.abs().square().sum() for t in value if t is not None)
    plan.path_products.add_(0)
    with pytest.raises(ValueError):torch.autograd.grad(loss,x)
    with pytest.raises(ValueError):owner._ideal_phase_path_log_products(128*1024**2)
    (tmp_path/'guards.json').write_text(json.dumps(dict(restored_saved_source_guards=trips,
        cached_path_version_saved_receiver_guard=True,qualification=False),indent=2)+'\n')


def test_guarded_empty_actual_magnetic_interior_path_array(tmp_path):
    from witwin.maxwell.mom.fmm_order import AutomaticDirectionalRegionPlan
    original=_physical_plan();body=original.basis.system.bodies[0]
    root=AutomaticDirectionalRegionPlan(body.basis,body.interior,geometry=original.basis.geometry[0],
        kernel_error_budget=original.kernel_error_budget,ideal_rwg_error_budget=original.ideal_rwg_error_budget,
        balancing_impedance=original.balancing_impedance,leaf_size=4,separation_ratio=.15,cone_subdivisions=(2**20,))
    plan=_select(root)
    assert not root.far_nodes and plan.context_keys==()
    owner=DirectionalRWGDAGProjection(root.basis,root.tree,plan.carriers,plan.orders,plan.requests,plan.links,root.medium.k,
        geometry=root.geometry,quadrature_order=plan.quadrature_order,face_batch=plan.face_batch,rhs_batch=plan.rhs_batch,
        retained_budget_bytes=plan.retained_budget_bytes,tensor_workspace_budget_bytes=plan.tensor_workspace_budget_bytes,
        phase_relative_budget=plan.phase_relative_budget,plan_guard=plan)
    paths=owner._ideal_phase_path_log_products(128*1024**2)
    assert paths is plan.path_products and paths.numel()==0 and paths.device.type=='cuda'
    assert storage_bytes(owner.tensors())==owner.retained_bytes==0
    assert owner.source_upward(torch.ones((2,root.tree.size),device='cuda',dtype=torch.complex128))==()
    with pytest.raises(MemoryError):owner._ideal_phase_path_log_products(directional_phase_path_workspace_bytes(0)-1)
    paths.requires_grad_(True)
    with pytest.raises(ValueError):owner._ideal_phase_path_log_products(128*1024**2)
    paths.requires_grad_(False);owner.assert_current()
    (tmp_path/'empty.json').write_text(json.dumps(dict(actual_magnetic_interior_all_near=True,
        guarded_empty_cuda_array_reused=True,zero_owned_bytes=True,workspace_preflight_preserved=True,
        no_positive_interior_FAR_qualification=True,qualification=False),indent=2)+'\n')
