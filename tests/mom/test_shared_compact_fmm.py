"""Exact displacement sharing preserves all logical Cartesian FMM couplings."""
import copy
import hashlib
import json
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.fmm_storage import shared_displacement_catalog, m2l_storage_bytes
from witwin.maxwell.mom.fmm_system import MultipoleSystemActions
from witwin.maxwell.mom.fmm_translation import ElectromagneticM2L
from witwin.maxwell.mom.fmm_compact import CompactElectromagneticM2L
from witwin.maxwell.mom.fmm_capacity import plan_multipole_storage, multipole_action_tensors, storage_bytes, MultipoleCapacityError
from witwin.maxwell.mom.fmm_order import plan_multipole_order
from witwin.maxwell.mom.rwg_interpolation_error import rwg_integral_logs, integral_statistics_from_logs
from witwin.maxwell.mom.fmm_operator import MultipoleOperator
from witwin.maxwell.mom.runtime import SurfaceData, SystemOperator
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.operator import IntegrationConfig
from test_fmm_scalar_near_channels import _provider

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
MODE = 'compact_shared_displacement_v1'


def _relative(actual, reference):
    return float(((actual-reference).norm()/reference.norm()).detach())


def test_shared_catalog_exact_native_bits_signed_zero_and_missing_reverse(tmp_path):
    centers = torch.tensor([[0., 0., 0.], [1., 0., -0.], [2., 0., 0.],
                            [0., 1., 0.], [1., 1., 0.]], device='cuda', dtype=torch.float64)
    perturbed = centers[1].clone()
    perturbed[0] = torch.nextafter(perturbed[0], perturbed.new_tensor(float('inf')))
    centers = torch.cat((centers, perturbed[None]))
    pairs = [(1, 0), (0, 1), (2, 1), (4, 3), (0, 5)]
    representatives, bindings = shared_displacement_catalog(centers, pairs)
    assert representatives == (0, 4)
    assert bindings == ((0, False), (0, True), (0, False), (0, False), (1, False))
    # Exact translation equality is enough: no physical reverse-pair closure
    # or approximate float binning is assumed by this table-only catalog.
    assert shared_displacement_catalog(centers, pairs[:1]) == ((0,), ((0, False),))
    for bad in ([(0, 0)], [(0, 6)], [(True, 1)]):
        with pytest.raises(ValueError): shared_displacement_catalog(centers, bad)
    with pytest.raises(ValueError): shared_displacement_catalog(centers.float(), pairs)
    with pytest.raises(ValueError): shared_displacement_catalog(centers.clone().requires_grad_(), pairs)
    (tmp_path/'catalog.json').write_text(json.dumps(dict(logical_pairs=5, unique_displacements=2,
        signed_zero_key_normalized=True, one_ULP_not_merged=True, different_nodes_same_delta_shared=True,
        incomplete_reverse_pairs_supported_without_partition_reduction=True,
        host_transfers_are_integer_bits_and_topology=True, production_qualification=False), indent=2)+'\n')


def test_shared_config_all_fields_old_versions_and_early_public_rejection(tmp_path):
    for near in ('full_matrix', 'scalar_dk_v1', 'scalar_reciprocal_dk_v1'):
        config = mw.FMMConfig(table_storage=MODE, near_storage=near, shift_storage='separable_v1')
        payload = json.loads(json.dumps(config.payload()))
        assert payload['version'] == 'cartesian_fmm_shared_compact_config_v1'
        assert mw.FMMConfig.restore(payload) == config
        for field in payload['parameters']:
            bad = copy.deepcopy(payload); del bad['parameters'][field]
            with pytest.raises(ValueError): mw.FMMConfig.restore(bad)
        for version in ('cartesian_fmm_reciprocal_dk_config_v1', 'cartesian_fmm_scalar_dk_config_v1',
                        'cartesian_fmm_config_v7', 'cartesian_fmm_positive_tail_config_v1'):
            bad = copy.deepcopy(payload); bad['version'] = version
            with pytest.raises(ValueError): mw.FMMConfig.restore(bad)
        bad = copy.deepcopy(payload); bad['parameters']['table_storage'] = 'compact'
        with pytest.raises(ValueError): mw.FMMConfig.restore(bad)
        options = dict(operator_backend='fmm', fmm=config, riesz_storage='local_rwg_v1', stopping=StoppingRule(preconditioner='rwg_diagonal'))
        with pytest.raises(NotImplementedError, match='free space'):
            mw.MoMConfig(periodic=mw.MoMPeriodicConfig(((1., 0., 0.),)), **options)
        with pytest.raises(ValueError, match='complex128'):
            mw.MoMConfig(dtype=torch.complex64, **options)
    assert mw.FMMConfig().payload()['version'] == 'cartesian_fmm_config_v7'
    assert mw.FMMConfig(near_storage='scalar_reciprocal_dk_v1').payload()['version'] == 'cartesian_fmm_reciprocal_dk_config_v1'
    (tmp_path/'config.json').write_text(json.dumps(dict(new_version=payload['version'],
        all_fields_required=True, old_versions_preserved=True, periodic_and_complex64_rejected=True,
        production_qualification=False), indent=2)+'\n')


@pytest.fixture(scope='module')
def complete():
    provider = _provider()
    op = MultipoleSystemActions(provider, order=8, quadrature_order=12, leaf_size=2,
        separation_ratio=.25, table_storage=MODE, shift_storage='separable_v1',
        near_storage='scalar_reciprocal_dk_v1', kernel_bound_rule='direct_first_wave_min_v1',
        kernel_error_budget=mw.KernelErrorBudget(1e-6, 1e-5), near_action_batch=2)
    # Independent original compact coefficient generation for every logical
    # FAR pair. The already integrated near objects and original projections
    # are shared; no near reintegration or new global physics oracle is implied.
    reference = copy.copy(op)
    ordered_regions = []
    for region in (op.exterior, *op.interiors):
        ordered = copy.copy(region)
        pairs = [(a, b) for a, b, _ in region.far]
        originals = ElectromagneticM2L.create_many(region.test, region.source, pairs, region.media[0],
            separation_ratio=.25, table_storage='compact', row_batch=region.m2l_row_batch,
            kernel_bound_rule='direct_first_wave_min_v1', error_budget=mw.KernelErrorBudget(1e-6, 1e-5))
        ordered.far = tuple((a, b, coupling) for (a, b), coupling in zip(pairs, originals, strict=True))
        ordered.table_storage = 'compact'
        ordered.table_bytes = len(pairs)*m2l_storage_bytes(8, 8, 'compact')
        ordered._binding = ordered._state()
        ordered_regions.append(ordered)
    reference.exterior, reference.interiors = ordered_regions[0], tuple(ordered_regions[1:])
    reference.table_bytes = sum(r.table_bytes for r in ordered_regions)
    trees = tuple(r.test.tree for r in ordered_regions)
    capacity = plan_multipole_storage(trees, order=8, separation_ratio=.25, budget_bytes=1024**3,
        table_storage='compact', shift_storage='separable_v1', near_storage='scalar_reciprocal_dk_v1')
    reference._capacity_text = json.dumps(capacity, sort_keys=True, allow_nan=False)
    reference._binding = reference._state()
    reference.assert_current()
    assert storage_bytes(multipole_action_tensors(reference)) == capacity['planned_bytes']
    return provider, op, reference


def test_shared_full_columns_H_current_higher_capacity_and_saved_owner(complete, monkeypatch, tmp_path):
    provider, op, reference = complete
    assert op.shape == (120, 120) and op.exterior.far
    assert op.table_bytes < reference.table_bytes
    assert op.capacity_plan['categories']['m2l'] == op.table_bytes
    assert storage_bytes(multipole_action_tensors(op)) == op.capacity_plan['planned_bytes']
    assert op.capacity_plan['planned_bytes']-reference.capacity_plan['planned_bytes'] == op.table_bytes-reference.table_bytes
    assert all(value == reference.capacity_plan['categories'][name]
        for name, value in op.capacity_plan['categories'].items() if name != 'm2l')
    assert [(a, b) for a, b, _ in op.exterior.far] == [(a, b) for a, b, _ in reference.exterior.far]
    eye = torch.eye(op.shape[1], device='cuda', dtype=op.dtype)
    actual, original = op.apply(eye).mT, reference.apply(eye).mT
    n = provider.system.total_edges; eta = abs(provider.exterior.eta)
    left = actual.real.new_tensor([eta]*n+[1.]*n); right = actual.real.new_tensor([1.]*n+[eta]*n)
    weak = _relative(left[:, None]*actual*right, left[:, None]*original*right)
    assert weak < 1e-11
    generator = torch.Generator(device='cuda').manual_seed(15485863)
    current = torch.randn((3, 5, op.shape[1]), device='cuda', dtype=op.dtype,
        generator=generator).transpose(0, 1).requires_grad_()
    dual = torch.randn(current.shape, device='cuda', dtype=op.dtype, generator=generator).requires_grad_()
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        forward = op.apply(current)
    assert not saved and not current.is_contiguous()
    H = op.adjoint(dual)
    gradient, = torch.autograd.grad(forward, current, dual, retain_graph=True)
    H_gradient, = torch.autograd.grad(H, dual, current, retain_graph=True)
    first, = torch.autograd.grad(forward.abs().square().sum(), current, create_graph=True)
    tangent = torch.randn_like(current)
    second, = torch.autograd.grad(first, current, tangent)
    lhs, rhs = (forward.conj()*dual).sum(), (current.conj()*H).sum()
    errors = dict(forward=_relative(forward, reference.apply(current)), H=_relative(H, reference.adjoint(dual)),
        current=_relative(gradient, H), adjoint_current=_relative(H_gradient, forward),
        higher=_relative(second, 2*op.adjoint(op.apply(tangent))),
        dot=float(((lhs-rhs).abs()/torch.maximum(lhs.abs(), rhs.abs())).detach()))
    assert max(errors.values()) < 1e-11
    def forbidden(*args, **kwargs): raise AssertionError('Shared capacity rejection reached moment/coefficients')
    with monkeypatch.context() as patch:
        patch.setattr('witwin.maxwell.mom.fmm_system.NestedRWGMultipoles', forbidden)
        for key, value, error in (('action_retained_budget_bytes', op.capacity_plan['planned_bytes']-1, MultipoleCapacityError),
                                  ('table_budget_bytes', op.table_bytes-1, MemoryError)):
            with pytest.raises(error):
                MultipoleSystemActions(provider, order=8, leaf_size=2, separation_ratio=.25,
                    table_storage=MODE, shift_storage='separable_v1', near_storage='scalar_reciprocal_dk_v1', **{key: value})
    borrowed = next(c._compact for _, _, c in op.exterior.far if hasattr(c._compact, '_reverse_owner'))
    output = op.apply(current); owner = borrowed._reverse_owner; original_batch = owner.row_batch
    owner.row_batch += 1
    for call in (op.receipt, lambda: op.adjoint(dual), lambda: torch.autograd.grad(output, current, dual)):
        with pytest.raises(ValueError, match='changed'): call()
    owner.row_batch = original_batch; op.assert_current()
    record = dict(physical_edges=n, logical_FAR_pairs=[len(r.far) for r in (op.exterior, *op.interiors)],
        original_independent_compact_FAR_with_shared_near=True, shared_near_not_reintegrated=True,
        all_columns_balanced_relative=weak, native_errors=errors, saved_internal_tensors=len(saved),
        shared_capacity=op.capacity_plan, ordered_capacity=reference.capacity_plan,
        exact_rejection_before_maps=True, saved_shared_owner_guard=True,
        original_global_RWG_or_physics_certified=False, whole_peak_or_crossover_qualified=False,
        production_qualification=False)
    (tmp_path/'system.json').write_text(json.dumps(record, indent=2)+'\n'); print(record, flush=True)


def test_shared_order_plan_preserves_all_bounds_plain_and_RWG(complete, tmp_path):
    provider, op, _ = complete
    regions = (op.exterior, *op.interiors)
    trees = tuple(r.test.tree for r in regions)
    logs = tuple(rwg_integral_logs(b.basis, g) for b, g in zip(provider.system.bodies, provider.geometry, strict=True))
    statistics = (integral_statistics_from_logs(trees[0], torch.cat(logs)),
        *tuple(integral_statistics_from_logs(t, v) for t, v in zip(trees[1:], logs, strict=True)))
    records = []
    for rwg in (False, True):
        options = dict(rwg_statistics=statistics, rwg_absolute_budget=1e-4,
                       balancing_impedance=abs(provider.exterior.eta)) if rwg else {}
        plans = [plan_multipole_order(tuple(zip(trees, (provider.exterior, *[b.interior for b in provider.system.bodies]), strict=True)),
            mw.KernelErrorBudget(1e-6, 1e-5), min_order=8, max_order=8, separation_ratio=.25,
            table_storage=mode, kernel_bound_rule='direct_first_wave_min_v1', **options)
            for mode in (MODE, 'compact')]
        assert plans[0]['table_bytes'] == op.table_bytes < plans[1]['table_bytes']
        a, b = (dict(p['attempts'][0]) for p in plans)
        a.pop('table_bytes'); b.pop('table_bytes'); assert a == b
        records.append(dict(RWG=rwg, shared=plans[0], ordered=plans[1]))
    (tmp_path/'order.json').write_text(json.dumps(dict(fixed_order_single_point_no_search_replay=True,
        all_ordered_kernel_bounds_and_RWG_accumulation_unchanged=True, plans=records,
        production_qualification=False), indent=2)+'\n')


def test_public_shared_compact_solve_exact_restore_and_policy_binding(tmp_path, monkeypatch):
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.01, position=(x, 0., 0.)),
        mw.Material(eps_r=2.+i, mu_r=1.2+.1*i)) for i, x in enumerate((-.2, .2))], sources=[mw.PlaneWave()])
    fmm = mw.FMMConfig(order=6, max_order=10, kernel_error_budget=mw.KernelErrorBudget(1e-6, 1e-5),
        kernel_bound_rule='direct_first_wave_min_v1', table_storage=MODE, shift_storage='separable_v1',
        near_storage='scalar_reciprocal_dk_v1', leaf_size=8, separation_ratio=.25,
        leaf_factor_kernel='primary', leaf_factor_max_edges=16)
    config = mw.MoMConfig(subdivisions=0, integration=IntegrationConfig(order=4, singular_order=3),
        operator_backend='fmm', fmm=fmm, riesz_storage='local_rwg_v1',
        stopping=StoppingRule(method='gmres', preconditioner='rwg_diagonal', fallback='none',
            rtol=1e-12, atol=0., maxiter=240, restart=120))
    result = mw.Simulation.mom(scene, frequency=9e7, config=config).run()
    stats = result.stats(); assert stats['converged'] and stats['error_budget']['checked']
    actions = result.surface_currents.handle.actions
    assert actions.exterior.far and actions.exterior.table_storage == MODE
    assert actions.capacity_plan['table_storage'] == MODE
    current = result.surface_currents.scattered_coefficients
    path = tmp_path/'shared-compact-result.pt'; result.save(path)
    def forbidden(*args, **kwargs): raise AssertionError('Shared compact restore constructed or solved an operator')
    with monkeypatch.context() as patch:
        for cls in (MultipoleOperator, MultipoleSystemActions, SystemOperator): patch.setattr(cls, '__init__', forbidden)
        patch.setattr(torch.linalg, 'solve', forbidden)
        restored = mw.Result.load(path, scene=scene)
        assert restored.surface_currents.config.fmm == fmm
        torch.testing.assert_close(restored.surface_currents.scattered_coefficients, current, rtol=0, atol=0)
        for field, value in (('near_storage', 'scalar_dk_v1'), ('table_storage', 'compact'),
                             ('kernel_bound_rule', 'bivariate_cauchy_v1'), ('shift_storage', 'dense')):
            bad = copy.deepcopy(result.surface_currents.snapshot()); bad['fmm_config']['parameters'][field] = value
            with pytest.raises(ValueError): SurfaceData.restore(bad, scene, 9e7)
    (tmp_path/'public.json').write_text(json.dumps(dict(artifact=str(path),
        artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), fmm_config=fmm.payload(), stats=stats,
        selected_order=actions.order_plan['selected_order'], capacity=actions.capacity_plan,
        exact_current_restore=True, original_residual_is_sample_not_global=True,
        whole_peak_or_crossover_qualified=False, production_qualification=False), indent=2, default=str)+'\n')


def test_actual_cross_body_representative_setup_uses_at_most32_displacements(complete, monkeypatch, tmp_path):
    provider, op, _ = complete
    projection = op.exterior.test
    tree = projection.tree
    starts, stops = tree.starts.cpu().tolist(), tree.stops.cpu().tolist()
    permutation = tree.permutation.cpu().tolist()
    leaves = (tree.children < 0).all(-1).nonzero().flatten().cpu().tolist()
    left = [n for n in leaves if all(e < 30 for e in permutation[starts[n]:stops[n]])]
    right = [n for n in leaves if all(e >= 30 for e in permutation[starts[n]:stops[n]])]
    pairs = [(a, b) for a in left for b in right]
    representatives, _ = shared_displacement_catalog(projection.centers, pairs)
    assert len(representatives) > 32
    batches = []
    original = CompactElectromagneticM2L.create_many
    def spy(cls, displacements, *args, **kwargs):
        batches.append(len(displacements))
        assert 0 < len(displacements) <= 32
        return original(displacements, *args, **kwargs)
    monkeypatch.setattr(CompactElectromagneticM2L, 'create_many', classmethod(spy))
    expected = len(representatives)*m2l_storage_bytes(8, 8, 'compact')
    with pytest.raises(MemoryError, match='before coefficients'):
        ElectromagneticM2L.create_many(projection, projection, pairs, provider.exterior,
            separation_ratio=.25, table_storage='compact_shared_displacement_v1', table_budget_bytes=expected-1)
    assert not batches
    couplings = ElectromagneticM2L.create_many(projection, projection, pairs, provider.exterior,
        separation_ratio=.25, table_storage='compact_shared_displacement_v1', table_budget_bytes=expected)
    assert len(couplings) == len(pairs) and sum(batches) == len(representatives)
    assert storage_bytes(t for coupling in couplings for t in coupling.tensors()) == expected
    record = dict(actual_cross_body_nodes=True, logical_pairs=len(pairs), representatives=len(representatives),
        actual_coefficient_generation_batches=batches, maximum_native_displacement_batch=32,
        one_byte_below_budget_rejected_before_coefficients=True, actual_unique_retained_bytes=expected,
        no_near_integration_or_complete_solve_repeated=True, production_qualification=False)
    (tmp_path/'batching.json').write_text(json.dumps(record, indent=2)+'\n'); print(record, flush=True)
