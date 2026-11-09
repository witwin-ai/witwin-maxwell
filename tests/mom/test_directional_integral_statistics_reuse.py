"""Guarded automatic statistics reuse without repeated physical aggregation."""
import json
import mpmath as mp
import pytest
import torch
from witwin.maxwell.mom.rwg_interpolation_error import tree_integral_statistics
from witwin.maxwell.mom.fmm_directional_rwg import _sparse_directional_rwg_admission
from witwin.maxwell.mom.direct_capacity import storage_bytes
from test_directional_admission_path_cache import owners
from test_directional_sparse_coupling import _mp_integrals

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_actual_all_node_and_all_root_material_statistics_uncached_and_MP100(owners, monkeypatch, tmp_path):
    plan, owner, uncached = owners
    borrowed = tree_integral_statistics(owner)
    original = tree_integral_statistics(uncached)
    assert borrowed is plan._statistics
    torch.testing.assert_close(borrowed, original, rtol=0, atol=0)
    assert storage_bytes(owner.tensors()) == owner.retained_bytes == uncached.retained_bytes
    from witwin.maxwell.mom import rwg_interpolation_error as module
    def forbidden(*args, **kwargs):
        raise AssertionError('Borrowed statistics recomputed physical integrals or aggregation')
    with monkeypatch.context() as patch:
        patch.setattr(module, 'rwg_integral_logs', forbidden)
        patch.setattr(module, 'integral_statistics_from_logs', forbidden)
        for _ in range(64):
            assert tree_integral_statistics(owner) is borrowed
    errors = []
    with mp.workdps(100):
        edge_integrals = _mp_integrals(owner, 0)
        assert len(edge_integrals) == owner.tree.size
        starts, stops = owner.tree.starts.tolist(), owner.tree.stops.tolist()
        for index, (start, stop) in enumerate(zip(starts, stops, strict=True)):
            values = edge_integrals[start:stop]
            expected = (sum(u*u for u, q in values), sum(q*q for u, q in values), sum(u*q for u, q in values))
            for column, value in enumerate(expected):
                native = mp.exp(mp.mpf(float(borrowed[index, column])))
                errors.append(float(abs(native/value-1)))
        root = plan.region_plan
        index = {key: i for i, key in enumerate(owner.context_keys)}
        bounds = []
        for (a, b), family in zip(root.far_nodes, plan.family_ids, strict=True):
            ta, sb = index[a, family], index[b, family]
            options = (root.medium, root.balancing_impedance, 128*1024**2)
            _, _, actual = _sparse_directional_rwg_admission(owner, ta, owner, sb, *options, dag=True)
            _, _, expected = _sparse_directional_rwg_admission(uncached, ta, uncached, sb, *options, dag=True)
            assert actual['absolute_bound'] == pytest.approx(expected['absolute_bound'], rel=1e-11, abs=0)
            # Independent actual material weighting over every root rectangle.
            left, right = edge_integrals[starts[a]:stops[a]], edge_integrals[starts[b]:stops[b]]
            k, z = abs(mp.mpc(root.medium.k)), mp.mpf(root.balancing_impedance)
            e, m = abs(mp.mpc(1j*root.medium.omega*root.medium.epsilon)), abs(mp.mpc(-1j*root.medium.omega*root.medium.mu))
            dsum = sum((u*v+q*r/k**2)**2 for u, q in left for v, r in right)
            usum = sum((u*v)**2 for u, q in left for v, r in right)
            exact = mp.sqrt((z**4*e**2+m**2)*dsum*mp.mpf(actual['effective_green_absolute'])**2
                +6*z**2*usum*mp.mpf(actual['effective_gradient_component_absolute'])**2)
            errors.append(float(abs(mp.mpf(actual['absolute_bound'])/exact-1)))
            bounds.append(actual['absolute_bound'])
    assert errors and max(errors) < 1e-11
    (tmp_path/'statistics.json').write_text(json.dumps(dict(all_node_MP100_and_root_material_relative_errors=errors,
        maximum_relative=max(errors),mp_absolute_tolerance=0,physical_nodes=owner.tree.node_count,
        actual_root_rectangles=len(bounds),uncached_statistics_bit_equal=True,repeated_queries=64,
        no_repeated_edge_integrals_or_tree_aggregation=True,no_second_owner_statistics_array=True,
        planner_statistics_storage_bytes=borrowed.untyped_storage().nbytes(),
        retained_map_bytes=owner.retained_bytes,planner_owned_storage_remains_external_dependency=True,
        whole_resource_qualification=False,qualification=False),indent=2)+'\n')


def test_borrowed_statistics_identity_mode_and_saved_current_version_guards(owners, tmp_path):
    plan, owner, _ = owners
    trips = []
    for kind in ('identity', 'mode'):
        x = torch.ones((2, owner.tree.size), device='cuda', dtype=torch.complex128, requires_grad=True)
        values = owner.source_upward(x)
        loss = sum(v.abs().square().sum() for v in values if v is not None)
        old = plan._statistics
        if kind == 'identity': plan._statistics = old.clone()
        else: old.requires_grad_(True)
        try:
            with pytest.raises(ValueError): tree_integral_statistics(owner)
            with pytest.raises(ValueError): owner.receipt()
            with pytest.raises(ValueError): plan.receipt()
            with pytest.raises(ValueError): torch.autograd.grad(loss, x)
        finally:
            if kind == 'identity': plan._statistics = old
            else: old.requires_grad_(False)
        owner.assert_current(); trips.append(kind)
    x = torch.ones((2, owner.tree.size), device='cuda', dtype=torch.complex128, requires_grad=True)
    values = owner.receiver_upward(x)
    loss = sum(v.abs().square().sum() for v in values if v is not None)
    plan._statistics.add_(0)
    with pytest.raises(ValueError): torch.autograd.grad(loss, x)
    with pytest.raises(ValueError): tree_integral_statistics(owner)
    (tmp_path/'guards.json').write_text(json.dumps(dict(restored_saved_source_current_guards=trips,
        statistics_version_saved_receiver_current_guard=True,qualification=False),indent=2)+'\n')


def test_all_near_magnetic_interior_reuses_nonempty_physical_statistics(monkeypatch, tmp_path):
    from witwin.maxwell.mom.fmm_order import AutomaticDirectionalRegionPlan
    from witwin.maxwell.mom.fmm_directional_moments import DirectionalRWGDAGProjection
    from test_directional_automatic_dag import _physical_plan, _select
    original = _physical_plan()
    body = original.basis.system.bodies[0]
    root = AutomaticDirectionalRegionPlan(body.basis, body.interior, geometry=original.basis.geometry[0],
        kernel_error_budget=original.kernel_error_budget, ideal_rwg_error_budget=original.ideal_rwg_error_budget,
        balancing_impedance=original.balancing_impedance, leaf_size=4, separation_ratio=.15,
        cone_subdivisions=(2**20,))
    plan = _select(root)
    assert not root.far_nodes and plan.context_keys == ()
    options = dict(geometry=root.geometry, quadrature_order=plan.quadrature_order,
        face_batch=plan.face_batch, rhs_batch=plan.rhs_batch,
        retained_budget_bytes=plan.retained_budget_bytes,
        tensor_workspace_budget_bytes=plan.tensor_workspace_budget_bytes,
        phase_relative_budget=plan.phase_relative_budget)
    args = (root.basis, root.tree, plan.carriers, plan.orders, plan.requests, plan.links, root.medium.k)
    owner = DirectionalRWGDAGProjection(*args, **options, plan_guard=plan)
    uncached = DirectionalRWGDAGProjection(*args, **options)
    borrowed = tree_integral_statistics(owner)
    torch.testing.assert_close(borrowed, tree_integral_statistics(uncached), rtol=0, atol=0)
    assert borrowed is plan._statistics and borrowed.shape == (root.tree.node_count, 3)
    assert borrowed.numel() > 0 and borrowed.device.type == 'cuda'
    assert storage_bytes(owner.tensors()) == owner.retained_bytes == 0
    from witwin.maxwell.mom import rwg_interpolation_error as module
    def forbidden(*args, **kwargs):
        raise AssertionError('All-near statistics recomputed physical aggregation')
    with monkeypatch.context() as patch:
        patch.setattr(module, 'rwg_integral_logs', forbidden)
        patch.setattr(module, 'integral_statistics_from_logs', forbidden)
        assert tree_integral_statistics(owner) is borrowed
    borrowed.requires_grad_(True)
    with pytest.raises(ValueError): tree_integral_statistics(owner)
    borrowed.requires_grad_(False)
    owner.assert_current()
    (tmp_path/'all-near.json').write_text(json.dumps(dict(actual_magnetic_interior_all_near=True,
        physical_nodes=root.tree.node_count, nonempty_cuda_statistics_reused=True,
        uncached_statistics_bit_equal=True, zero_owned_map_bytes=True,
        planner_statistics_storage_bytes=borrowed.untyped_storage().nbytes(),
        no_positive_interior_FAR_qualification=True, whole_resource_qualification=False,
        qualification=False), indent=2)+'\n')
