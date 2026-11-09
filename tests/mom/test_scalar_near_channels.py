"""Original single-medium D/K, material factors, Hermitian actions and owned storage."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig, PreparedRWGGeometry
from witwin.maxwell.mom.nearfield import scalar_regional_block, scalar_regional_channels
from witwin.maxwell.mom.scalar_near_channels import ScalarNearChannels

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def _basis(center=(0., 0., 0.)):
    return RWGBasis(sphere_mesh(radius=.02, center=center, subdivisions=0, device='cuda', dtype=torch.float64))


def _legacy_module():
    names = ('GSP_ORIGINAL_NEARFIELD_SOURCE', 'GSP_ORIGINAL_NEARFIELD_SHA256')
    if all(n not in os.environ for n in names):pytest.skip('Original selected integration snapshot required')
    assert all(n in os.environ for n in names), 'Incomplete original integration pair'
    path = Path(os.environ[names[0]])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == os.environ[names[1]]
    spec = importlib.util.spec_from_file_location('witwin.maxwell.mom._original_nearfield_channels', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _relative(actual, expected):return float(((actual-expected).norm()/expected.norm()).detach())


@pytest.mark.parametrize('kind', ('self_vacuum', 'self_magnetic_lossy', 'cross_vacuum', 'cross_magnetic_lossy'))
def test_original_all_columns_and_complete_near_channel_adjoint_current_and_storage(kind, tmp_path):
    legacy = _legacy_module()
    test = _basis(); source = test if kind.startswith('self') else _basis((.3, .04, -.02))
    material = mw.Material(eps_r=2., mu_r=1.3, sigma_e=.01, sigma_m=.02) if 'lossy' in kind else mw.Material()
    medium = compile_medium(material, 3e8)
    config = IntegrationConfig(order=4, singular_order=3)
    # Full self columns include touching classes; rectangular cross retains order.
    rows = torch.arange(test.n_edges-1, -1, -1, device='cuda') if source is test else torch.tensor([17, 2, 11, 0, 29], device='cuda')
    cols = torch.roll(rows, 7) if source is test else torch.tensor([7, 1, 22, 0, 5, 29, 11], device='cuda')
    geometry = (PreparedRWGGeometry(test), PreparedRWGGeometry(source))
    op = ScalarNearChannels(test, source, rows, cols, medium, integration=config, geometry=geometry)
    reference, original_receipt = legacy.scalar_regional_block(test, source, rows, cols, (medium,), config, geometry=geometry)
    eye = torch.eye(op.shape[1], device='cuda', dtype=torch.complex128)
    actual = op.apply(eye).mT
    eta = abs(medium.eta); nr, nc = len(rows), len(cols)
    left = actual.real.new_tensor([eta]*nr+[1.]*nr); right = actual.real.new_tensor([1.]*nc+[eta]*nc)
    balanced = _relative(left[:, None]*actual*right, left[:, None]*reference*right)
    generator = torch.Generator(device='cuda').manual_seed(1351)
    currents = torch.randn((3, 5, op.shape[1]), device='cuda', dtype=torch.complex128, generator=generator).transpose(0, 1).requires_grad_()
    dual = torch.randn((*currents.shape[:-1], op.shape[0]), device='cuda', dtype=torch.complex128, generator=generator)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t:t):applied = op.apply(currents)
    assert not saved
    adjoint = op.adjoint(dual)
    forward = _relative(applied, currents@reference.mT)
    original_adjoint = _relative(adjoint, dual@reference.conj())
    dense_adjoint = _relative(adjoint, dual@actual.conj())
    derivative, = torch.autograd.grad(applied, currents, dual)
    gradient = _relative(derivative, adjoint)
    lhs, rhs = (applied.conj()*dual).sum(), (currents.conj()*adjoint).sum()
    dot = float(((lhs-rhs).abs()/torch.maximum(lhs.abs(), rhs.abs())).detach())
    assert all(t._base is None and t.untyped_storage().nbytes() == t.numel()*t.element_size() for t in op.tensors())
    assert len({t.untyped_storage().data_ptr() for t in op.tensors()}) == 4
    assert sum(t.untyped_storage().nbytes() for t in op.tensors()) == op.retained_bytes == 32*nr*nc+8*(nr+nc)
    record = dict(kind=kind, balanced_original_all_column=balanced, original_forward=forward,
        original_adjoint=original_adjoint, dense_adjoint=dense_adjoint, current_gradient=gradient, dot=dot,
        original_pair_counts=original_receipt['triangle_pair_counts'], channel_bytes=32*nr*nc,
        original_full_matrix_bytes=64*nr*nc, owned_retained_bytes=op.retained_bytes,
        row_order=rows.tolist(), col_order=cols.tolist(), receipt=op.receipt(), physical_qualification=False)
    (tmp_path/'near.json').write_text(json.dumps(record, indent=2)+'\n'); print(record, flush=True)
    assert balanced < 1e-8
    assert max(forward, original_adjoint, dense_adjoint, gradient, dot) < 1e-11


def test_default_four_block_helper_preserves_original_operations_and_selected_results(tmp_path):
    legacy = _legacy_module(); basis = _basis()
    rows = torch.tensor([7, 2, 11, 0], device='cuda'); cols = torch.tensor([5, 1, 19], device='cuda')
    media = tuple(compile_medium(mw.Material(eps_r=e, mu_r=1.3), 3e8) for e in (1.2, 2.))
    config = IntegrationConfig(order=4, singular_order=3); geometry = PreparedRWGGeometry(basis)
    actual, metadata = scalar_regional_block(basis, basis, rows, cols, media, config, geometry=(geometry, geometry))
    original, reference_meta = legacy.scalar_regional_block(basis, basis, rows, cols, media, config, geometry=(geometry, geometry))
    error = _relative(actual, original)
    assert metadata == reference_meta
    assert error < 1e-11
    record = dict(default_original_relative_error=error, metadata_exact=True,
        multi_medium_default_preserved=True, bit_equivalence_claimed=False, physical_qualification=False)
    (tmp_path/'default.json').write_text(json.dumps(record, indent=2)+'\n'); print(record, flush=True)


def test_near_channel_preflight_single_medium_native_owned_and_physical_mutation_guards(monkeypatch):
    import witwin.maxwell.mom.scalar_near_channels as module
    basis = _basis(); rows = torch.tensor([7, 2, 11], device='cuda'); cols = torch.tensor([5, 1], device='cuda')
    medium = compile_medium(mw.Material(eps_r=2., mu_r=1.3), 3e8)
    native = module.scalar_regional_channels
    def forbidden(*args, **kwargs):raise AssertionError('Rejected near representation reached integration')
    monkeypatch.setattr(module, 'scalar_regional_channels', forbidden)
    for changed, exception in [({'medium':(medium, medium)}, TypeError), ({'medium':replace(medium, k=0j)}, ValueError),
            ({'medium':replace(medium, k=1.-1j)}, ValueError), ({'rhs_batch':False}, ValueError),
            ({'retained_budget_bytes':32*len(rows)*len(cols)+8*(len(rows)+len(cols))-1}, MemoryError)]:
        kw = dict(test_basis=basis, source_basis=basis, test_edges=rows, source_edges=cols, medium=medium)
        kw.update(changed)
        with pytest.raises(exception):ScalarNearChannels(**kw)
    with pytest.raises(TypeError, match='exactly one'):scalar_regional_channels(basis, basis, rows, cols, (medium, medium))
    monkeypatch.setattr(module, 'scalar_regional_channels', native)
    op = ScalarNearChannels(basis, basis, rows, cols, medium)
    original_selection = op.test_edges.clone(); rows.add_(1); cols.add_(1)
    op.assert_current(); assert torch.equal(op.test_edges, original_selection)
    for fn, width in ((op.apply, op.shape[1]), (op.adjoint, op.shape[0])):
        with pytest.raises(ValueError):fn(torch.ones(width, device='cuda', dtype=torch.complex64))
        with pytest.raises(ValueError):fn(torch.ones(width, device='cuda', dtype=torch.complex128)*float('nan'))
    op.electric_kernel.add_(1)
    with pytest.raises(ValueError, match='changed'):op.apply(torch.ones(op.shape[1], device='cuda', dtype=torch.complex128))
    other = ScalarNearChannels(basis, basis, rows, cols, medium)
    basis.mesh.vertices.add_(0)
    with pytest.raises(ValueError, match='changed'):other.assert_current()
