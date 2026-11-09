"""Actual original weak kernels, grouped D/K and full-system retained admission."""
from dataclasses import replace
import json
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig, assemble_system
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks, scalar_regional_channels
from witwin.maxwell.mom.packed_near import PackedScalarNearChannels
from witwin.maxwell.mom.fmm_system import MultipoleSystemActions
from witwin.maxwell.mom.fmm_capacity import storage_bytes, multipole_action_tensors, plan_multipole_storage, MultipoleCapacityError

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def _provider():
    exterior = compile_medium(mw.Material(), 3e8)
    bodies = [(i, str(i), RWGBasis(sphere_mesh(radius=.01, center=(x, 0., 0.),
        subdivisions=0, device='cuda', dtype=torch.float64)),
        compile_medium(mw.Material(eps_r=2.+i, mu_r=1.3+i*.1, sigma_e=.01, sigma_m=.02), 3e8))
        for i, x in enumerate((-.5, .5))]
    return ScalarSystemBlocks(SurfaceSystem.create(bodies), exterior, IntegrationConfig(order=3, singular_order=3))


def _relative(x, y):return float(((x-y).norm()/y.norm()).detach())


def _current_checks(op, dense, generator):
    width = dense.shape[1]
    value = torch.randn((3, 5, width), device='cuda', dtype=torch.complex128, generator=generator).transpose(0, 1).requires_grad_()
    dual = torch.randn((*value.shape[:-1], dense.shape[0]), device='cuda', dtype=torch.complex128, generator=generator)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t, lambda t:t):result = op.apply(value)
    assert not saved
    adjoint = op.adjoint(dual)
    grad, = torch.autograd.grad(result, value, dual)
    lhs, rhs = (result.conj()*dual).sum(), (value.conj()*adjoint).sum()
    errors = dict(forward=_relative(result, value@dense.mT), adjoint=_relative(adjoint, dual@dense.conj()),
                  gradient=_relative(grad, adjoint), dot=float(((lhs-rhs).abs()/torch.maximum(lhs.abs(), rhs.abs())).detach()))
    assert max(errors.values()) < 1e-11
    return errors


def test_packed_rectangular_original_channels_storage_preflight_and_mutation(monkeypatch, tmp_path):
    provider = _provider(); basis = provider.system.bodies[0].basis
    medium = provider.system.bodies[0].interior; config = provider.config
    # Distinct variable rectangular shapes and shared target nodes exercise additive CSR actions.
    r0, r1 = torch.tensor([7, 2], device='cuda'), torch.tensor([11], device='cuda')
    c0, c1 = torch.tensor([5, 1, 19], device='cuda'), torch.tensor([17, 0], device='cuda')
    nr, nc = basis.n_edges, basis.n_edges
    layout = [(a, b+2, torch.cat((r, r+nr)), torch.cat((c, c+nc))+2*nr)
              for a, b, r, c in [(0, 0, r0, c0), (0, 1, r0, c1), (1, 0, r1, c0)]]
    expected_bytes = 8*(2*2+1+3*len(layout))+sum(32*(len(r)//2)*(len(c)//2)+8*(len(r)+len(c)) for _, _, r, c in layout)
    calls = []
    def factory(a, b, rows, cols):
        calls.append((a, b))
        d, k, _ = scalar_regional_channels(basis, basis, rows[:len(rows)//2], cols[:len(cols)//2]-2*nr,
                                         medium, config, geometry=(provider.geometry[0], provider.geometry[0]))
        return d, k
    def forbidden(*args, **kwargs):raise AssertionError('Rejected retained budget reached original integration')
    with pytest.raises(MemoryError):PackedScalarNearChannels.from_layout(layout, 2*nr, 2*nc, forbidden,
        medium=medium, integration=config, device='cuda', dtype=torch.complex128, retained_budget_bytes=expected_bytes-1)
    op = PackedScalarNearChannels.from_layout(layout, 2*nr, 2*nc, factory, medium=medium, integration=config,
        device='cuda', dtype=torch.complex128, action_batch=1, retained_budget_bytes=expected_bytes)
    assert len(calls) == len(layout)
    assert op.receipt()['retained_bytes'] == expected_bytes == storage_bytes(op.tensors())
    assert all(t._base is None and t.untyped_storage().nbytes() == t.numel()*t.element_size() for t in op.tensors())
    assert len({t.untyped_storage().data_ptr() for t in op.tensors()}) == len(op.tensors())
    dense = torch.zeros((op.size, op.size), device='cuda', dtype=torch.complex128)
    for rows, cols, fields in op.groups:
        for rr, cc, (d, k) in zip(rows, cols, fields):
            full = torch.cat((torch.cat((k, op.electric*d), 1), torch.cat((op.magnetic*d, k), 1)), 0)
            dense[rr[:, None], cc] += full
    errors = _current_checks(op, dense, torch.Generator(device='cuda').manual_seed(2161))
    value = torch.ones(op.size, device='cuda', dtype=torch.complex128, requires_grad=True); output = op.apply(value)
    original = config.near_tolerance; object.__setattr__(config, 'near_tolerance', original*2)
    for action in (op.receipt, lambda:op.apply(value), lambda:torch.autograd.grad(output, value, torch.ones_like(output))):
        with pytest.raises(ValueError, match='changed'):action()
    object.__setattr__(config, 'near_tolerance', original); op.assert_current()
    op.groups[0][2].add_(0)
    with pytest.raises(ValueError, match='changed'):op.adjoint(value)
    record = dict(errors=errors, owned_bytes=expected_bytes, factory_calls=len(calls), full_matrix_in_factory=False, qualification=False)
    (tmp_path/'packed.json').write_text(json.dumps(record, indent=2)+'\n'); print(record, flush=True)


def test_global_exterior_channels_preserve_interleaved_routing_and_separate_source_provider(tmp_path):
    provider = _provider()
    source = ScalarSystemBlocks(provider.system, provider.exterior, provider.config)
    rows = torch.tensor([37, 2, 41, 7, 0], device='cuda'); cols = torch.tensor([5, 31, 49, 1, 22, 32, 17], device='cuda')
    d, k = provider.exterior_channels(rows, cols, source_provider=source)
    a, b = 1j*provider.exterior.omega*provider.exterior.epsilon, -1j*provider.exterior.omega*provider.exterior.mu
    full = torch.cat((torch.cat((k, a*d), 1), torch.cat((b*d, k), 1)), 0)
    original = provider.exterior_cross_block(source, rows, cols)
    error = _relative(full, original); assert error < 1e-11
    assert all(t._base is None and t.untyped_storage().nbytes() == t.numel()*16 for t in (d, k))
    record = dict(original_relative_error=error, row_order=rows.tolist(), col_order=cols.tolist(), qualification=False)
    (tmp_path/'routing.json').write_text(json.dumps(record, indent=2)+'\n'); print(record, flush=True)


@pytest.mark.parametrize('storage', ('dense', 'compact', 'fft_v1'))
def test_full_fmm_original_columns_current_and_exact_reduced_capacity(storage, monkeypatch, tmp_path):
    provider = _provider()
    # Original weak oracle is assembled before forbidding four-block near factories.
    original, _ = assemble_system(provider.system, provider.exterior, provider.config)
    import witwin.maxwell.mom.fmm_regional as module
    def forbidden(*args, **kwargs):raise AssertionError('D/K near construction used full four-block factory')
    monkeypatch.setattr(module, 'scalar_regional_block', forbidden)
    monkeypatch.setattr(provider, 'exterior_block', forbidden); monkeypatch.setattr(provider, 'exterior_cross_block', forbidden)
    op = MultipoleSystemActions(provider, order=6, leaf_size=30, table_storage=storage,
                                shift_storage='separable_v1', near_storage='scalar_dk_v1', near_action_batch=1)
    assert len(op.exterior.far) == 2 and all(not r.far for r in op.interiors)
    actual = op.apply(torch.eye(op.shape[1], device='cuda', dtype=torch.complex128)).mT
    n = provider.system.total_edges; eta = abs(provider.exterior.eta)
    left = actual.real.new_tensor([eta]*n+[1.]*n); right = actual.real.new_tensor([1.]*n+[eta]*n)
    weak = _relative(left[:, None]*actual*right, left[:, None]*original*right); assert weak < 1e-8
    errors = _current_checks(op, actual, torch.Generator(device='cuda').manual_seed(2261))
    regions = (op.exterior, *op.interiors); trees = tuple(r.test.tree for r in regions)
    old_plan = plan_multipole_storage(trees, order=6, separation_ratio=.5, budget_bytes=1024**3,
        table_storage=storage, shift_storage='separable_v1')
    count = storage_bytes(multipole_action_tensors(op)); assert count == op.capacity_plan['planned_bytes']
    savings = sum(fields.numel()*fields.element_size() for r in regions for _, _, fields in r.near.groups)
    assert old_plan['planned_bytes']-count == savings
    assert old_plan['categories']['near']-op.capacity_plan['categories']['near'] == savings
    assert sum(r.near.receipt()['retained_bytes'] for r in regions) == op.near_bytes
    assert all(r.near.receipt()['version'] == 'grouped_scalar_original_dk_csr_v1' for r in regions)
    import witwin.maxwell.mom.fmm_system as system_module
    with monkeypatch.context() as patch:
        patch.setattr(system_module, 'NestedRWGMultipoles', forbidden)
        with pytest.raises(MultipoleCapacityError):MultipoleSystemActions(provider, order=6, leaf_size=30,
            table_storage=storage, shift_storage='separable_v1', near_storage='scalar_dk_v1', action_retained_budget_bytes=count-1)
    record = dict(table_storage=storage, balanced_original_all_column=weak, current_errors=errors,
        actual_owned_bytes=count, original_full_matrix_plan_bytes=old_plan['planned_bytes'], near_kernel_savings_bytes=savings,
        receipt=op.receipt(), public_integration_qualified=False, memory_crossover_qualified=False, physical_qualification=False)
    (tmp_path/'system.json').write_text(json.dumps(record, indent=2)+'\n'); print(record, flush=True)


def test_single_medium_native_and_jm_layout_rejected_before_factory(monkeypatch):
    provider = _provider(); medium = provider.exterior
    rows = torch.tensor([1, 3], device='cuda'); cols = torch.tensor([5, 7], device='cuda')
    def forbidden(*args, **kwargs):raise AssertionError('Rejected packed map reached integration')
    for params in (dict(dtype=torch.complex64), dict(medium=(medium, medium)), dict(medium=replace(medium, k=0j))):
        kw = dict(medium=medium, integration=provider.config, device='cuda', dtype=torch.complex128); kw.update(params)
        with pytest.raises((TypeError, ValueError)):PackedScalarNearChannels.from_layout([(0, 1, rows, cols)], 4, 4, forbidden, **kw)
    bad_rows = torch.tensor([1, 2], device='cuda')
    with pytest.raises(ValueError, match='J then M'):PackedScalarNearChannels.from_layout([(0, 1, bad_rows, cols)], 4, 4, forbidden,
        medium=medium, integration=provider.config, device='cuda', dtype=torch.complex128)
    with pytest.raises(TypeError):PackedScalarNearChannels([], 8)
