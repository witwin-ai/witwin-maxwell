"""Complete retained dependency inventory against an independent object walk."""
import json
import pytest
import torch
from witwin.maxwell.mom.fmm_capacity import directional_retained_tensor_groups, directional_retained_inventory
from witwin.maxwell.mom.fmm_config import DirectionalFMMConfig
from witwin.maxwell.mom.fmm_system import DirectionalSystemActions
from test_directional_automatic_dag import _physical_plan

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def _object_storages(root):
    # Independent test oracle reads actual reachable state, not tensors() or
    # the production inventory. Only framework objects and containers recurse.
    pending, visited, result = [root], set(), {}
    while pending:
        value = pending.pop()
        if id(value) in visited: continue
        visited.add(id(value))
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            result[value.device, storage.data_ptr()] = storage.nbytes()
        elif isinstance(value, dict): pending.extend(value.values())
        elif isinstance(value, (tuple, list)): pending.extend(value)
        elif type(value).__module__.startswith('witwin.maxwell.mom.') and hasattr(value, '__dict__'):
            pending.extend(vars(value).values())
    return result


def _enumerated_storages(groups):
    result = {}
    for values in groups.values():
        for value in values:
            storage = value.untyped_storage()
            result[value.device, storage.data_ptr()] = storage.nbytes()
    return result


def test_public_factory_actual_all_region_inventory_and_borrowed_planner_guards(tmp_path, monkeypatch):
    provider = _physical_plan().basis
    actions = DirectionalSystemActions.from_automatic(provider,
        DirectionalFMMConfig(leaf_size=4, separation_ratio=.15, cone_subdivisions=(2**20,),
            max_contexts=4096, tensor_workspace_budget_bytes=512*1024**2))
    groups = directional_retained_tensor_groups(actions)
    expected = _object_storages(actions)
    actual = _enumerated_storages(groups)
    assert actual == expected
    record = directional_retained_inventory(actions)
    assert record['retained_native_bytes'] == sum(expected.values())
    assert record['unique_storages'] == len(expected)
    assert sum(c['attributed_storage_bytes'] for c in record['categories'].values()) == sum(expected.values())
    assert sum(c['standalone_storage_bytes'] for c in record['categories'].values()) > sum(expected.values())
    assert record['categories']['automatic_dag_plans']['attributed_storage_bytes'] > 0
    assert record['categories']['automatic_region_plans']['attributed_storage_bytes'] > 0
    regions = (actions.exterior, *actions.interiors)
    assert len(regions) == 4 and regions[0].test.plan_guard.selected_positive_links > 0
    assert all(not r.far.couplings for r in regions[1:])
    for region in regions:
        plan = region.test.plan_guard
        for value in (plan.carriers, plan.path_products, plan._statistics):
            storage = value.untyped_storage()
            assert actual[value.device, storage.data_ptr()] == storage.nbytes()
    def forbidden(*args, **kwargs):
        raise AssertionError('Retained inventory copied tensors or recomputed physics')
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, 'clone', forbidden)
        patch.setattr(torch.Tensor, 'cpu', forbidden)
        patch.setattr(type(provider), '__call__', forbidden)
        patch.setattr(type(provider), 'exterior_channels', forbidden)
        for _ in range(8):
            assert directional_retained_inventory(actions) == record
    old = regions[0].test.plan_guard._statistics
    regions[0].test.plan_guard._statistics = old.clone()
    try:
        with pytest.raises(ValueError): directional_retained_inventory(actions)
        with pytest.raises(ValueError): directional_retained_tensor_groups(actions)
    finally: regions[0].test.plan_guard._statistics = old
    actions.assert_current()
    record.update(independent_reachable_object_storage_equality=True, actual_bodies=3,
        exterior_positive_changing_links=regions[0].test.plan_guard.selected_positive_links,
        interior_actual_FAR=[len(r.far.couplings) for r in regions[1:]],
        no_tensor_copies_or_physics_recomputation=True, repeated_queries=8,
        stale_borrowed_statistics_rejected=True, inventory_only_no_actions_or_solve_replayed=True)
    (tmp_path/'inventory.json').write_text(json.dumps(record, indent=2)+'\n')


def test_invalid_directional_inventory_owner_rejected():
    with pytest.raises(TypeError): directional_retained_inventory(object())


def test_actual_nested_alternate_coupling_owner_and_shared_layout_included(tmp_path):
    import witwin.maxwell as mw
    from witwin.maxwell.mom.mesh import SurfaceMesh, RWGBasis, SurfaceSystem
    from witwin.maxwell.mom.physics import compile_medium
    from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
    from witwin.maxwell.mom.operator import IntegrationConfig
    from witwin.maxwell.mom.cluster import build_support_octree
    from witwin.maxwell.mom.interpolation_error import KernelErrorBudget
    from witwin.maxwell.mom.fmm_directional_moments import NestedDirectionalRWGProjection
    from witwin.maxwell.mom.fmm_directional_rwg import DirectionalFFTElectromagneticM2L, DirectionalNestedFarActions
    from witwin.maxwell.mom.fmm_regional import DirectionalRegionalOperator
    exterior = compile_medium(mw.Material(), 3e8)
    entries = []
    for i in range(2):
        vertices = torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
            device='cuda', dtype=torch.float64)*.015
        vertices[:, 0] += i*.8
        faces = torch.tensor([[0,2,1],[0,1,3],[0,3,2],[1,2,3]], device='cuda')
        basis = RWGBasis(SurfaceMesh(vertices, faces))
        entries.append((i, str(i), basis, compile_medium(mw.Material(eps_r=2.+i, mu_r=1.3+.1*i), 3e8)))
    provider = ScalarSystemBlocks(SurfaceSystem.create(entries), exterior,
        IntegrationConfig(order=3, singular_order=3, pair_batch=4, far_pair_batch=8))
    lower, upper = provider.support_boxes()
    tree = build_support_octree(lower, upper, leaf_size=6)
    minus = lower.new_tensor([-1.,0.,0.])
    options = dict(quadrature_order=12, face_batch=4)
    left = NestedDirectionalRWGProjection(provider, tree, minus, exterior.k, 8, **options)
    right = NestedDirectionalRWGProjection(provider, tree, -minus, exterior.k, 8, layout=left.layout, **options)
    a, b = left.children[0]
    eta = abs(exterior.eta)
    couplings = tuple(DirectionalFFTElectromagneticM2L.from_nested_nodes(p, t, p, s, exterior,
        lower.new_tensor(.01), kernel_error_budget=KernelErrorBudget(1e-6,1e-4),
        ideal_rwg_error_budget=1e-4, balancing_impedance=eta)
        for p, t, s in ((left,a,b),(right,b,a)))
    far = DirectionalNestedFarActions(left, left, exterior, couplings, ideal_rwg_error_budget=1e-4)
    regional = DirectionalRegionalOperator(far, ((a,a),(b,b)), integration=provider.config)
    interiors = []
    for body, geometry in zip(provider.system.bodies, provider.geometry, strict=True):
        start, stop = body.edge_offset, body.edge_offset+body.basis.n_edges
        local_tree = build_support_octree(lower[start:stop], upper[start:stop], leaf_size=6)
        local = NestedDirectionalRWGProjection(body.basis, local_tree, minus, body.interior.k, 8,
            geometry=geometry, **options)
        empty = DirectionalNestedFarActions(local, local, body.interior, (), ideal_rwg_error_budget=1e-4,
            balancing_impedance=eta)
        interiors.append(DirectionalRegionalOperator(empty, ((0,0),), integration=provider.config))
    actions = DirectionalSystemActions(provider, regional, interiors, ideal_rwg_error_budget=1e-4)
    groups = directional_retained_tensor_groups(actions)
    expected = _object_storages(actions)
    assert _enumerated_storages(groups) == expected
    record = directional_retained_inventory(actions)
    assert record['retained_native_bytes'] == sum(expected.values())
    assert right is not regional.test and right in far.targets and right.layout is left.layout
    assert groups['automatic_region_plans'] == groups['automatic_dag_plans'] == ()
    for tensor in right.tensors():
        storage = tensor.untyped_storage()
        assert expected[tensor.device, storage.data_ptr()] == storage.nbytes()
    record.update(independent_reachable_object_storage_equality=True,
        alternate_actual_coupling_owner_included=True, shared_layout_counted_once=True,
        prescribed_nested_control=True, inventory_only_no_original_matrix_or_solve=True)
    (tmp_path/'nested.json').write_text(json.dumps(record, indent=2)+'\n')


@pytest.mark.parametrize('guarded', (False, True))
def test_actual_DAG_independent_caller_carrier_view_storage_included(guarded, monkeypatch, tmp_path):
    from witwin.maxwell.mom.fmm_directional_moments import DirectionalRWGDAGProjection
    provider = _physical_plan().basis
    original = DirectionalRWGDAGProjection.__init__
    callers = []
    def separate_carriers(self, basis, tree, carriers, *args, **kwargs):
        # Exercise supported constructor inputs with equal numerical carriers
        # in independent oversized storage, including empty interior views.
        backing = torch.zeros((len(carriers)+2, 3), device=carriers.device, dtype=carriers.dtype)
        caller = backing[1:-1]
        caller.copy_(carriers)
        if not guarded: kwargs['plan_guard'] = None
        original(self, basis, tree, caller, *args, **kwargs)
        callers.append(caller)
    with monkeypatch.context() as patch:
        patch.setattr(DirectionalRWGDAGProjection, '__init__', separate_carriers)
        actions = DirectionalSystemActions.from_automatic(provider,
            DirectionalFMMConfig(leaf_size=4, separation_ratio=.15, cone_subdivisions=(2**20,),
                max_contexts=4096, tensor_workspace_budget_bytes=512*1024**2))
    groups = directional_retained_tensor_groups(actions)
    expected = _object_storages(actions)
    actual = _enumerated_storages(groups)
    assert actual == expected and len(callers) == 4
    record = directional_retained_inventory(actions)
    assert record['retained_native_bytes'] == sum(expected.values())
    assert record['categories']['projection_dependencies']['attributed_storage_bytes'] == sum(
        caller.untyped_storage().nbytes() for caller in callers)
    regions = (actions.exterior, *actions.interiors)
    for region, caller in zip(regions, callers, strict=True):
        assert region.test._caller_carriers is caller
        assert (region.test.plan_guard is not None) == guarded
        assert caller.untyped_storage().nbytes() > caller.numel()*caller.element_size()
        assert actual[caller.device, caller.untyped_storage().data_ptr()] == caller.untyped_storage().nbytes()
    record.update(independent_reachable_object_storage_equality=True, actual_bodies=3,
        guarded_DAG=guarded, independent_equal_caller_carriers_included=True,
        full_backing_storage_of_nonempty_and_empty_views_included=True,
        caller_backing_bytes=[x.untyped_storage().nbytes() for x in callers],
        inventory_only_no_actions_original_matrix_or_solve=True)
    (tmp_path/'caller.json').write_text(json.dumps(record, indent=2)+'\n')
