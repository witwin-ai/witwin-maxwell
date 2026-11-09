"""Sparse physical context forest versus independent direct carrier quadrature."""
import json
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh, SurfaceSystem
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import PreparedRWGGeometry, IntegrationConfig
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.cluster import build_support_octree
from witwin.maxwell.mom.fmm_directional_moments import SparseDirectionalRWGProjection
from witwin.maxwell.mom.fmm_expansion import multi_indices
from witwin.maxwell.mom.direct_capacity import storage_bytes
from test_fmm_directional_rwg import _independent_edge_moments

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def _fixture(kind='single'):
    def basis(c):
        return RWGBasis(sphere_mesh(radius=.04, center=c, subdivisions=0, device='cuda', dtype=torch.float64))
    if kind == 'single':
        b = basis((0., 0., 0.)); geometry = PreparedRWGGeometry(b)
        supports = torch.cat((geometry.triangles[b.plus_faces], geometry.triangles[b.minus_faces]), 1)
        lower, upper = supports.amin(1), supports.amax(1)
    else:
        medium = compile_medium(mw.Material(), 3e8)
        system = SurfaceSystem.create([(i, str(i), basis((x, 0., 0.)), medium) for i, x in enumerate((-.3, .3))])
        b = ScalarSystemBlocks(system, medium, IntegrationConfig(order=3, singular_order=3)); geometry = None
        lower, upper = b.support_boxes()
    tree = build_support_octree(lower, upper, leaf_size=len(lower) if kind == 'mixed_leaf' else 2)
    carriers = lower.new_tensor([[.7, -.2, .3], [-.2, .5, -.7]])
    carriers = carriers/carriers.norm(dim=-1, keepdim=True)
    if kind == 'mixed_leaf':
        requests = ((0, 0), (0, 1))
    else:
        nodes = [n for n in range(1, tree.node_count) if bool((tree.children[n] >= 0).any())
            and 4 <= int(tree.stops[n]-tree.starts[n]) <= 20]
        assert len(nodes) >= 2
        a = nodes[0]; child = int(tree.children[a][tree.children[a] >= 0][0])
        other = nodes[-1]
        requests = ((a, 0), (child, 0), (other, 1))
    return b, tree, carriers, geometry, requests


def _build(fixture, **kwargs):
    b, tree, carriers, geometry, requests = fixture
    options = dict(quadrature_order=12, geometry=geometry, face_batch=4)
    options.update(kwargs)
    return SparseDirectionalRWGProjection(b, tree, carriers, (3, 4), requests, 5.+.8j, **options)


def _flatten(values):return torch.cat([v.flatten(-2) for v in values], -1)


def _unflatten(values, owner):
    return tuple(v.reshape(*values.shape[:-1], count, 4) for v, count in zip(values.split([c*4 for c in owner.node_counts], -1), owner.node_counts))


def _relative(a, b):return float(((a-b).norm()/b.norm()).detach())


def _independent_maps(owner):
    width = sum(owner.node_counts)*4
    maps = [torch.zeros((width, owner.tree.size), device='cuda', dtype=torch.complex128) for _ in range(2)]
    at = 0
    for context, (node, carrier) in enumerate(owner.context_keys):
        ids = owner.tree.permutation[int(owner.tree.starts[node]):int(owner.tree.stops[node])]
        groups = owner.basis._body_selection(ids) if isinstance(owner.basis, ScalarSystemBlocks) else ((None, None),)
        for bi, selected in groups:
            if bi is None:basis, selected_ids, edges = owner.basis, ids, ids
            else:
                body = owner.basis.system.bodies[bi]; basis, selected_ids = body.basis, ids[selected]; edges = selected_ids-body.edge_offset
            for side, sign in enumerate((-1, 1)):
                fields = _independent_edge_moments(basis, edges, owner.centers[context], owner.carriers[carrier],
                    sign*owner.wave_number, owner.node_orders[context], 16)
                maps[side][at:at+owner.node_counts[context]*4, selected_ids] = fields.permute(1, 2, 0).flatten(0, 1)
        at += owner.node_counts[context]*4
    return maps


@pytest.mark.parametrize('kind', ('single', 'global', 'mixed_leaf'))
def test_sparse_requested_closure_independent_four_maps_all_h_current_higher(kind, tmp_path):
    fixture = _fixture(kind); owner = _build(fixture)
    source, receiver = _independent_maps(owner)
    if kind != 'mixed_leaf':
        assert len(owner.context_keys) < 2*owner.tree.node_count
        # Requested ancestor and descendant share exactly one physical context.
        keys = owner.context_keys
        assert len(keys) == len(set(keys))
        expected = set()
        for node, carrier in fixture[-1]:
            pending = [node]
            while pending:
                n = pending.pop(); expected.add((n, carrier))
                pending.extend(int(c) for c in owner.tree.children[n] if int(c) >= 0)
        assert set(keys) == expected
    else:
        assert owner.receipt()['integrated_leaf_body_groups'] == 4
    assert storage_bytes(owner.tensors()) == owner.retained_bytes
    scale = torch.cat([owner.radii[i].pow(torch.tensor([sum(a) for a in multi_indices(p)], device='cuda')).repeat_interleave(4)
        for i, p in enumerate(owner.node_orders)])
    generator = torch.Generator(device='cuda').manual_seed(92717)
    q = torch.randn((3, 5, owner.tree.size), device='cuda', dtype=torch.complex128, generator=generator).transpose(0, 1).requires_grad_()
    z = (torch.randn((3, 5, len(scale)), device='cuda', dtype=torch.complex128, generator=generator)/scale).transpose(0, 1).requires_grad_()
    references = dict(source_upward=q@source.mT, receiver_upward=q@receiver.conj().mT,
        source_downward=z@source.conj(), receiver_downward=z@receiver)
    opposite = dict(source_upward='source_downward', source_downward='source_upward',
        receiver_upward='receiver_downward', receiver_downward='receiver_upward')
    errors, current_errors, higher_errors = {}, {}, {}
    for name in references:
        upward = name.endswith('upward'); value = q if upward else z
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t, lambda t:t):
            raw = getattr(owner, name)(value if upward else _unflatten(value, owner))
        assert not saved
        result = _flatten(raw) if upward else raw
        errors[name] = _relative(result/scale, references[name]/scale) if upward else _relative(result, references[name])
        assert errors[name] < 1e-11
        dual = z if upward else q
        derivative, = torch.autograd.grad(result, value, dual, create_graph=True)
        backward = getattr(owner, opposite[name])(_unflatten(dual, owner) if upward else dual)
        backward = backward if upward else _flatten(backward)
        current_errors[name] = _relative(derivative, backward); assert current_errors[name] < 1e-11
        higher, = torch.autograd.grad(derivative, dual, value)
        higher_errors[name] = _relative(higher, result); assert higher_errors[name] < 1e-11
    # Missing cotangent slots reuse the shared protocol on a requested root only.
    request = owner.request_indices[0]
    selected = owner.source_upward(q)[request]
    derivative, = torch.autograd.grad(selected, q, torch.ones_like(selected))
    slots = [None]*len(owner.context_keys); slots[request] = torch.ones_like(selected)
    expected = owner._downward_dispatch(slots, 'source', q.shape[:-1], allow_missing=True)
    assert _relative(derivative, expected) < 1e-11
    record = dict(kind=kind, four_map_errors=errors, native_current=current_errors, native_higher=higher_errors,
        sparse_nodes=len(owner.context_keys), full_tree_per_carrier_nodes=2*owner.tree.node_count,
        receipt=owner.receipt(), owned_bytes=storage_bytes(owner.tensors()), full_wrapper_saved_tensors=0, qualification=False)
    (tmp_path/'maps.json').write_text(json.dumps(record, indent=2)+'\n'); print(record, flush=True)


def test_sparse_preflight_absent_support_zero_outputs_and_saved_backward_guards(monkeypatch, tmp_path):
    import witwin.maxwell.mom.fmm_directional_rwg as rwg
    import witwin.maxwell.mom.fmm_directional_moments as module
    fixture = _fixture(); owner = _build(fixture)
    def forbidden(*a, **kw):raise AssertionError('Rejected sparse capacity reached maps/transfers')
    with monkeypatch.context() as patch:
        patch.setattr(rwg, '_rwg_weighted_moments', forbidden)
        patch.setattr(module, 'DirectionalFixedCarrierShift', forbidden)
        for kw in ({'retained_budget_bytes':owner.retained_bytes-1},
                   {'tensor_workspace_budget_bytes':owner.planned_tensor_workspace_bytes-1}):
            with pytest.raises(MemoryError):_build(fixture, **kw)
    slots = tuple(torch.ones((count, 4), device='cuda', dtype=torch.complex128) for count in owner.node_counts)
    result = owner.source_downward(slots)
    active_edges = set()
    for node, _ in owner.context_keys:
        active_edges.update(owner.tree.permutation[int(owner.tree.starts[node]):int(owner.tree.stops[node])].tolist())
    absent = sorted(set(range(owner.tree.size))-active_edges)
    assert absent and bool((result[absent] == 0).all())
    empty = torch.empty((0, owner.tree.size), device='cuda', dtype=torch.complex128, requires_grad=True)
    output = owner.source_upward(empty); assert all(v.shape[0] == 0 for v in output)
    derivative, = torch.autograd.grad(sum(v.real.sum() for v in output), empty); assert derivative.shape == empty.shape
    tripwires = []
    for kind in ('carrier', 'layout', 'contexts', 'request', 'geometry', 'budget'):
        f = _fixture(); p = _build(f); current = torch.ones(p.tree.size, device='cuda', dtype=torch.complex128, requires_grad=True)
        saved = p.source_upward(current)[p.request_indices[0]]
        if kind == 'carrier':f[2].mul_(.99)
        elif kind == 'layout':next(iter(p.layouts.values())).order += 1
        elif kind == 'contexts':p.context_node_ids.add_(1)
        elif kind == 'request':p.request_indices = p.request_indices[::-1]
        elif kind == 'geometry':f[3].triangles.add_(.001)
        else:p.retained_budget_bytes += 1
        with pytest.raises(ValueError):p.receipt()
        with pytest.raises(ValueError):torch.autograd.grad(saved, current, torch.ones_like(saved))
        tripwires.append(kind)
    record = dict(omitted_physical_edge_outputs_exact_zero=len(absent), empty_current_backward=True,
        capacity_before_factories=True, saved_output_tripwires=tripwires, qualification=False)
    (tmp_path/'guards.json').write_text(json.dumps(record, indent=2)+'\n'); print(record, flush=True)


def test_same_order_shared_layout_and_strict_native_tree_support_configuration(monkeypatch, tmp_path):
    from dataclasses import replace
    import witwin.maxwell.mom.fmm_directional_moments as module
    import witwin.maxwell.mom.fmm_directional_rwg as rwg
    b, tree, carriers, geometry, requests = _fixture()
    owner = SparseDirectionalRWGProjection(b, tree, carriers, (3, 3), requests, 5.+.8j,
        quadrature_order=12, geometry=geometry, face_batch=4)
    assert len(owner.layouts) == 1
    layout = owner.layouts[3]
    assert all(shift.layout is layout and not shift.owns_layout for shift in owner.transfers.values())
    assert storage_bytes(owner.tensors()) == owner.retained_bytes
    def forbidden(*a, **kw):raise AssertionError('Invalid sparse configuration reached numerical factories')
    rejects = []
    with monkeypatch.context() as patch:
        patch.setattr(module, 'DirectionalFixedCarrierShift', forbidden)
        patch.setattr(rwg, '_rwg_weighted_moments', forbidden)
        for label, changed in (('upper_float32', replace(tree, upper=tree.upper.float())),
                               ('actual_support', replace(tree, lower=tree.lower+100, upper=tree.upper+100))):
            with pytest.raises(ValueError):
                SparseDirectionalRWGProjection(b, changed, carriers, (3, 3), requests, 5.+.8j,
                    quadrature_order=12, geometry=geometry)
            rejects.append(label)
        identical = carriers.clone(); identical[1] = identical[0]
        with pytest.raises(ValueError):
            SparseDirectionalRWGProjection(b, tree, identical, (3, 3), requests, 5.+.8j,
                quadrature_order=12, geometry=geometry)
        for invalid in (requests+requests[:1], ((tree.node_count, 0), (1, 1)), ((1, 0),)):
            with pytest.raises(ValueError):
                SparseDirectionalRWGProjection(b, tree, carriers, (3, 3), invalid, 5.+.8j,
                    quadrature_order=12, geometry=geometry)
    record = dict(shared_order_layout_count=1, transfer_layout_owned_once=True,
        exact_inventory=owner.retained_bytes, invalid_native_support_and_request_rejections=rejects,
        original5PASS_not_repeated=True, qualification=False)
    (tmp_path/'configuration.json').write_text(json.dumps(record, indent=2)+'\n'); print(record, flush=True)


def test_fixed_geometry_and_native_configuration_autograd_modes_guard_saved_output(monkeypatch, tmp_path):
    import witwin.maxwell.mom.fmm_directional_moments as module
    import witwin.maxwell.mom.fmm_directional_rwg as rwg
    def forbidden(*a, **kw):raise AssertionError('Trainable geometry reached sparse numerical factories')
    with monkeypatch.context() as patch:
        patch.setattr(module, 'DirectionalFixedCarrierShift', forbidden)
        patch.setattr(rwg, '_rwg_weighted_moments', forbidden)
        for kind in ('single', 'global'):
            f = _fixture(kind)
            geometry = f[3] if kind == 'single' else f[0].geometry[0]
            geometry.basis.mesh.vertices.requires_grad_(True)
            with pytest.raises(ValueError):_build(f)
    modes = []
    for kind in ('caller_carrier', 'owned_carrier', 'owned_map', 'tree_box', 'mesh', 'prepared'):
        f = _fixture(); owner = _build(f)
        q = torch.ones(owner.tree.size, device='cuda', dtype=torch.complex128, requires_grad=True)
        output = owner.source_upward(q)[owner.request_indices[0]]
        if kind == 'caller_carrier':f[2].requires_grad_(True)
        elif kind == 'owned_carrier':owner.carriers.requires_grad_(True)
        elif kind == 'owned_map':next(iter(owner.leaves.values()))[0][1].source_vector.requires_grad_(True)
        elif kind == 'tree_box':f[1].lower.requires_grad_(True)
        elif kind == 'mesh':f[0].mesh.vertices.requires_grad_(True)
        else:f[3].triangles.requires_grad_(True)
        with pytest.raises(ValueError):owner.receipt()
        with pytest.raises(ValueError):torch.autograd.grad(output, q, torch.ones_like(output))
        modes.append(kind)
    record = dict(initial_single_and_global_trainable_mesh_refused=True,
        saved_output_fixed_mode_tripwires=modes, native_numerical_checks_repeated=False, qualification=False)
    (tmp_path/'fixed_modes.json').write_text(json.dumps(record, indent=2)+'\n'); print(record, flush=True)


def test_sparse_context_protocol_shares_methods_and_rejects_full_tree_identity(tmp_path):
    from witwin.maxwell.mom.fmm_directional_moments import NestedDirectionalRWGProjection
    from witwin.maxwell.mom.fmm_directional_rwg import DirectionalFFTElectromagneticM2L
    from witwin.maxwell.mom.interpolation_error import KernelErrorBudget
    methods = ('_validate', '_finite', '_upward_dispatch', '_upward', '_downward_dispatch', '_downward',
        'source_upward', 'source_downward', 'receiver_upward', 'receiver_downward')
    for name in methods:
        assert getattr(SparseDirectionalRWGProjection, name) is getattr(NestedDirectionalRWGProjection, name)
    owner = _build(_fixture())
    assert not isinstance(owner, NestedDirectionalRWGProjection)
    medium = compile_medium(mw.Material(), 3e8)
    with pytest.raises(TypeError):
        DirectionalFFTElectromagneticM2L.from_nested_nodes(owner, 0, owner, 1, medium,
            owner.centers.new_tensor(.04), kernel_error_budget=KernelErrorBudget(1e-6, 1e-4),
            ideal_rwg_error_budget=1e-4, balancing_impedance=abs(medium.eta))
    record = dict(same_primary_native_method_objects=list(methods),
        sparse_context_indices_not_full_physical_tree_indices=True,
        existing_complete_tree_coupling_refused_before_support_or_coefficients=True,
        prior7PASS_not_repeated=True, qualification=False)
    (tmp_path/'protocol.json').write_text(json.dumps(record, indent=2)+'\n'); print(record, flush=True)
