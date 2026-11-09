"""Preregistered finer failed-physics successor: capacity before any expansion."""
import hashlib
import json
import math
import os
from pathlib import Path
import time
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import sphere_mesh, RWGBasis, SurfaceSystem
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.cluster import build_support_octree
from witwin.maxwell.mom.fmm_capacity import plan_multipole_storage
from witwin.maxwell.mom.fmm_order import plan_multipole_order, MultipoleOrderError

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_original_f30_dyadic_center_minimum_capacity_before_expansions(monkeypatch, tmp_path):
    required = ('GSP_RESONANT_F24_RECEIPT', 'GSP_RESONANT_F24_RECEIPT_SHA256')
    supplied = [bool(os.environ.get(name)) for name in required]
    if not any(supplied):
        pytest.skip('Historical failed resonant f24 receipt and SHA are required for this successor preflight')
    if not all(supplied):
        raise ValueError('Resonant successor preflight requires both receipt and SHA')
    from witwin.maxwell.mom.fmm_moments import NestedRWGMultipoles
    def forbidden(*args, **kwargs):
        raise AssertionError('capacity preflight reached numerical expansion or original integration')
    monkeypatch.setattr(NestedRWGMultipoles, '__init__', forbidden)
    monkeypatch.setattr(ScalarSystemBlocks, '__call__', forbidden)
    previous = Path(os.environ['GSP_RESONANT_F24_RECEIPT'])
    previous_sha = hashlib.sha256(previous.read_bytes()).hexdigest()
    assert previous_sha == os.environ['GSP_RESONANT_F24_RECEIPT_SHA256']
    old = json.loads(previous.read_text())
    assert old['level'] == 24 and old['status'] == 'measured'
    assert not old['sampled_targets_passed'] and not old['qualification']
    assert old['true_relative_residual'] <= 1e-10
    assert old['far_relative_error'] > .005 and max(old['field_errors'].values()) > .005
    frequency = 414635862.3319615
    record = dict(version='resonant_f30_dyadic_center_capacity_preflight_v1', phase='geometry_start',
        radius_m=.1, frequency_hz=frequency, mesh_frequency=30,
        epsilon_r=[12., .01], physical_relative_target=.005,
        true_original_vector_residual_target=1e-10, solver_absolute_tolerance=0.,
        original_f24_receipt=str(previous), original_f24_sha256=previous_sha,
        original_f24_failed=True, original_f24_not_replayed=True,
        old_f30_capacity_fail_preserved=True, geometry_successor_requires_fresh_kernel_admission=True,
        leaf_size=8, expansion_center_policy='dyadic_cell_v1', separation_ratio=.25, min_order=6, max_order=12,
        table_storage='compact_shared_displacement_v1', shift_storage='separable_v1',
        near_storage='scalar_reciprocal_dk_v1', kernel_bound_rule='positive_tail_min_v1',
        kernel_budget=dict(green_absolute=1e-6, gradient_component_absolute=1e-5),
        action_retained_budget_bytes=2*1024**3, table_budget_bytes=1024**3,
        near_budget_bytes=1024**3, projection_budget_bytes=1024**3,
        allocator_fraction=.75, dense_global_matrix_constructed=False,
        numerical_expansions_constructed=False, original_quadrature_evaluated=False,
        compressed_operator_constructed=False, solved=False,
        physical_qualification=False, complexity_qualification=False,
        peak_memory_qualified=False, production_qualification=False)
    def save():
        (tmp_path/'capacity.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(phase=record['phase'], reason=record.get('reason'),
                             elapsed_seconds=time.perf_counter()-started)), flush=True)
    torch.cuda.set_per_process_memory_fraction(.75)
    torch.empty(1, device='cuda'); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter(); save()
    basis = RWGBasis(sphere_mesh(radius=.1, frequency=30, device='cuda', dtype=torch.float64))
    assert basis.n_edges == 27000 and len(basis.mesh.faces) == 18000
    exterior = compile_medium(mw.Material(), frequency, background=True)
    interior = compile_medium(mw.Material(eps_r=12.,
        sigma_e=.01*2*math.pi*frequency*mw.constants.EPSILON_0), frequency)
    assert interior.epsilon_r.real == 12. and abs(interior.epsilon_r.imag-.01) < 1e-15
    provider = ScalarSystemBlocks(SurfaceSystem.create([(0, 'sphere', basis, interior)]), exterior,
        IntegrationConfig(order=4, singular_order=6, pair_batch=64, far_pair_batch=8192))
    lower, upper = provider.support_boxes()
    # Match the actual constructor's separate exterior and body trees even for
    # this one-body case; sharing one tree would understate retained ownership.
    trees = tuple(build_support_octree(lower, upper, leaf_size=8, expansion_center_policy='dyadic_cell_v1') for _ in range(2))
    record.update(phase='minimum_capacity_start', edges=27000, currents=54000,
        same_N_full_dense_bytes=54000**2*16, tree_receipts=[tree.receipt() for tree in trees],
        integration=dict(order=4, singular_order=6, pair_batch=64, far_pair_batch=8192,
                         near_tolerance=1e-5, near_max_order=16))
    save()
    common = dict(separation_ratio=.25, budget_bytes=2*1024**3,
        table_storage='compact_shared_displacement_v1', shift_storage='separable_v1',
        near_storage='scalar_reciprocal_dk_v1')
    capacity = plan_multipole_storage(trees, order=6, **common)
    def component_gates(plan):
        categories = plan['categories']
        projection = sum(categories[key] for key in ('centers_radii', 'leaf_maps', 'leaf_moments', 'shifts'))
        return dict(action=plan['admitted'], table=categories['m2l'] <= 1024**3,
                    near=categories['near'] <= 1024**3, projection=projection <= 1024**3,
                    planned_projection_bytes=projection)
    gates = component_gates(capacity)
    minimum_admitted = all(gates[key] for key in ('action', 'table', 'near', 'projection'))
    record.update(minimum_order_capacity=capacity, minimum_component_gates=gates,
                  phase='minimum_capacity_done')
    save()
    # Native geometric categories are independent of order; compact coefficient
    # and leaf-moment counts increase with order. A rejected minimum cannot be
    # repaired by an order search, so avoid that unnecessary numerical work.
    if not minimum_admitted:
        record.update(reason='minimum_order_retained_or_component_capacity_exceeded',
            kernel_order_search_performed=False, selected_order=None,
            configured_range_capacity_admitted=False,
            monotone_capacity_scope='same trees/near partition and compact storage, candidate orders6..12')
    else:
        record['kernel_order_search_performed'] = True
        save()
        try:
            plan = plan_multipole_order(((trees[0], exterior), (trees[1], interior)),
                mw.KernelErrorBudget(1e-6, 1e-5), min_order=6, max_order=12,
                separation_ratio=.25, table_budget_bytes=1024**3,
                table_storage='compact_shared_displacement_v1', kernel_bound_rule='positive_tail_min_v1')
        except MultipoleOrderError as error:
            record.update(reason='kernel_order_or_table_admission_failed', order_failure=error.receipt,
                          selected_order=None, configured_range_capacity_admitted=False)
        else:
            selected = plan['selected_order']
            actual = capacity if selected == 6 else plan_multipole_storage(trees, order=selected, **common)
            selected_gates = component_gates(actual)
            selected_admitted = all(selected_gates[key] for key in ('action', 'table', 'near', 'projection'))
            record.update(order_plan=plan, selected_order=selected, selected_capacity=actual,
                selected_component_gates=selected_gates, configured_range_capacity_admitted=selected_admitted,
                reason='selected_retained_capacity_admitted_only' if selected_admitted
                       else 'selected_order_retained_or_component_capacity_exceeded')
    provider.assert_current()
    for tree in trees:
        tree.assert_current()
    torch.cuda.synchronize()
    record.update(phase='preflight_done', elapsed_seconds=time.perf_counter()-started,
        preflight_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        preflight_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        preflight_peak_scope='geometry/provider/two physical trees/order-capacity preflight only; no expansion, action, solve, backward or physical fields')
    save()
    assert record['same_N_full_dense_bytes'] == 46656000000
    assert not record['numerical_expansions_constructed'] and not record['original_quadrature_evaluated']
    assert record['physical_relative_target'] == .005 and record['true_original_vector_residual_target'] == 1e-10
