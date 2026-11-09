"""Actual periodic FFT system: original weak blocks, ownership and exclusions."""
import json
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.periodic import LatticeSpec, EwaldHelmholtzKernel
from witwin.maxwell.mom.periodic_operator import assemble_periodic_system
from witwin.maxwell.mom.ewald_nearfield import EwaldSystemBlocks
from witwin.maxwell.mom.fmm_periodic_system import PeriodicMultipoleSystemActions
from witwin.maxwell.mom.fmm_periodic_capacity import periodic_action_tensors, storage_bytes
from witwin.maxwell.mom.fmm_ewald_fft import EwaldFFTTerminalM2L
from witwin.maxwell.mom.fmm_storage import m2l_storage_bytes

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def _provider(dimension=1, extra_near=False):
    exterior = compile_medium(mw.Material(), 9e7)
    centers = (-.2, .2) if extra_near else (0.,)
    system = SurfaceSystem.create([(i, str(i), RWGBasis(sphere_mesh(radius=.01,
        center=(x, 0., 0.), subdivisions=0, device='cuda')),
        compile_medium(mw.Material(eps_r=2.+i, mu_r=1.2+.1*i), 9e7)) for i, x in enumerate(centers)])
    vectors = torch.tensor([[1., .2, .1], [.1, 1.2, .3], [.2, .1, .9]][:dimension],
        device='cuda', dtype=torch.float64)
    lattice = LatticeSpec(vectors, .17*vectors.sum(0))
    kernel = EwaldHelmholtzKernel(lattice, 2., real_shells=2, reciprocal_shells=2,
        quadrature_order=32, image_batch=13)
    return EwaldSystemBlocks(system, exterior, IntegrationConfig(order=4, singular_order=3, pair_batch=64), kernel)


@pytest.mark.parametrize('dimension,extra_near,shift', [(1, False, 'dense'),
    (2, False, 'separable_v1'), (3, False, 'dense'), (1, True, 'separable_v1')],
    ids=['1d', '2d', '3d', '1d-extra-near'])
def test_ewald_fft_original_all_columns_capacity_adjoint_and_current_gradient(dimension, extra_near, shift, tmp_path):
    provider = _provider(dimension, extra_near)
    kernel = provider.kernel
    actions = PeriodicMultipoleSystemActions(provider, kernel.lattice, ewald_kernel=kernel,
        order=8 if extra_near else 6, leaf_size=30, separation_ratio=.25,
        table_storage='fft_ewald_v1', shift_storage=shift)
    remote = actions.exterior.remote
    assert isinstance(remote, EwaldFFTTerminalM2L)
    assert actions.capacity_plan['table_storage'] == 'fft_ewald_v1'
    assert actions.capacity_plan['root_kernel'] == 'ewald'
    actual_bytes = storage_bytes(periodic_action_tensors(actions))
    assert actual_bytes == actions.capacity_plan['planned_bytes']
    assert len(remote.near_indices) == len(actions.exterior.regions)
    assert int(((remote.near_indices == 0).all(-1)).sum()) == 1
    assert actions.exterior.phases.numel() == len(remote.near_indices)
    metadata_bytes = remote.near_indices.numel()*8 + actions.exterior.phases.numel()*16
    assert actions.capacity_plan['categories']['image_metadata'] == metadata_bytes
    fft_bytes = m2l_storage_bytes(actions.exterior.test.order, 8, 'fft_v1')
    far = [c for region in (*actions.exterior.regions, *actions.interiors) for _, _, c in region.far]
    assert all(c.receipt()['table_storage'] == 'fft_v1' for c in far)
    assert actions.table_bytes == fft_bytes + sum(c.receipt()['table_bytes'] for c in far)
    assert actions.capacity_plan['categories']['m2l'] == actions.table_bytes
    assert actions.receipt()['interiors_periodic'] is False
    if extra_near:
        assert actions.capacity_plan['translated_images'] > 0 and far
        assert len(remote.near_indices) > 1
    dense, _ = assemble_periodic_system(provider.system, provider.exterior, provider.config, kernel=kernel)
    n = provider.system.total_edges
    actual = actions.apply(torch.eye(2*n, device='cuda', dtype=torch.complex128)).mT
    eta = abs(provider.exterior.eta)
    left = torch.tensor([eta]*n+[1.]*n, device='cuda', dtype=torch.float64)
    right = torch.tensor([1.]*n+[eta]*n, device='cuda', dtype=torch.float64)
    error = float((left[:, None]*(actual-dense)*right).norm()/(left[:, None]*dense*right).norm())
    generator = torch.Generator(device='cuda').manual_seed(938+dimension)
    x = torch.randn((3, 2*n), device='cuda', dtype=actual.dtype, generator=generator, requires_grad=True)
    y = torch.randn(x.shape, device='cuda', dtype=actual.dtype, generator=generator)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        applied = actions.apply(x)
    assert not saved
    adjoint = actions.adjoint(y)
    expected = y@actual.conj()
    derivative, = torch.autograd.grad(applied, x, y)
    adjoint_error = float((adjoint-expected).norm()/expected.norm())
    gradient_error = float((derivative-adjoint).norm()/adjoint.norm())
    lhs, rhs = (applied.conj()*y).sum(), (x.conj()*adjoint).sum()
    dot = float(((lhs-rhs).abs()/torch.maximum(lhs.abs(), rhs.abs())).detach())
    record = dict(dimension=dimension, extra_near=extra_near, balanced_original_all_column_error=error,
        adjoint_error=adjoint_error, gradient_error=gradient_error, dot_error=dot,
        actual_owned_bytes=actual_bytes, table_bytes=actions.table_bytes, capacity_plan=actions.capacity_plan,
        near_indices=remote.near_indices.cpu().tolist(), free_fft_tables=len(far), interiors_periodic=False,
        original_reference_scope='configured_ewald_quadrature_and_cutoffs',
        physical_qualification=False, complexity_qualification=False, infinite_periodic_certified=False)
    (tmp_path/'system.json').write_text(json.dumps(record, indent=2)+'\n')
    print(record, flush=True)
    assert error < 1e-8 and adjoint_error < 1e-11 and gradient_error < 1e-11 and dot < 1e-11
    remote.near_indices.add_(1)
    with pytest.raises(ValueError, match='changed'):
        actions.apply(x)
