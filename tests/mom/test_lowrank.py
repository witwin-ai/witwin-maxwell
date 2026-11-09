"""Low-rank actions, independent residuals, adversarial pivots and rank failure."""
import pytest
import torch
from witwin.maxwell.mom.lowrank import streamed_cross_approximation, CompressionFailure

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('pivoting', ['complete', 'adaptive'])
def test_complete_pivot_handles_zero_initial_rows_and_balanced_adjoint_gradients(pivoting):
    g = torch.Generator(device='cuda').manual_seed(21)
    u = torch.randn((27, 3), generator=g, device='cuda', dtype=torch.complex128)
    v = torch.randn((3, 19), generator=g, device='cuda', dtype=torch.complex128)
    u[:4] = 0
    a = u@v
    rs = torch.linspace(.2, 2., 27, device='cuda', dtype=torch.float64)
    cs = torch.linspace(1., 3., 19, device='cuda', dtype=torch.float64)
    requests = []
    def rows(ids):
        requests.append(len(ids))
        return a[ids]
    block = streamed_cross_approximation(rows, lambda i: a[:, i], a.shape,
        device=a.device, max_rank=4, rtol=1e-12, row_batch=5, row_scale=rs, column_scale=cs, pivoting=pivoting)
    assert block.receipt['rank'] <= 4 and max(requests) <= 5
    if pivoting == 'complete':
        assert block.receipt['rank'] == 3
    independent = torch.linalg.vector_norm((a-block.left@block.right)*rs[:, None]*cs[None, :])
    assert float(independent) <= block.receipt['threshold']
    x = torch.randn((2, 19), generator=g, device='cuda', dtype=a.dtype, requires_grad=True)
    y = torch.randn((2, 27), generator=g, device='cuda', dtype=a.dtype)
    torch.testing.assert_close(block.apply(x), x@a.mT, rtol=2e-12, atol=1e-12)
    torch.testing.assert_close(block.adjoint(y), y@a.conj(), rtol=2e-12, atol=1e-12)
    defect = ((y.conj()*block.apply(x)).sum()-(block.adjoint(y).conj()*x).sum()).abs()
    assert float((defect/(torch.linalg.vector_norm(x)*torch.linalg.vector_norm(y))).detach()) < 1e-11
    loss = .5*(block.apply(x)-y).abs().square().sum()
    gradient, = torch.autograd.grad(loss, x)
    torch.testing.assert_close(gradient, block.adjoint(block.apply(x)-y), rtol=2e-12, atol=1e-12)
    block.left.add_(1)
    with pytest.raises(ValueError, match='changed'):
        block.apply(x)


@pytest.mark.parametrize('pivoting', ['complete', 'adaptive'])
def test_rank_exhaustion_reports_unmet_residual_and_zero_block_returns_rank_zero(pivoting):
    a = torch.eye(13, device='cuda', dtype=torch.complex128)
    with pytest.raises(CompressionFailure, match='exhausted rank 2') as caught:
        streamed_cross_approximation(lambda ids: a[ids], lambda i: a[:, i], a.shape,
            device=a.device, max_rank=2, rtol=1e-10, row_batch=3, pivoting=pivoting)
    receipt = caught.value.receipt
    assert not receipt['converged'] and receipt['balanced_residual_norm'] > receipt['threshold']
    a.zero_()
    block = streamed_cross_approximation(lambda ids: a[ids], lambda i: a[:, i], a.shape,
        device=a.device, rtol=0., atol=0., pivoting=pivoting)
    assert block.receipt['rank'] == 0
    assert torch.equal(block.apply(torch.ones(13, device=a.device, dtype=a.dtype)), a[0])


@pytest.mark.parametrize('pivoting', ['complete', 'adaptive'])
def test_actual_rwg_far_block_sampling_and_compression(pivoting):
    import witwin.maxwell as mw
    from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
    from witwin.maxwell.mom.physics import compile_medium
    from witwin.maxwell.mom.operator import IntegrationConfig
    from witwin.maxwell.mom.nearfield import ScalarSystemBlocks, ScalarEdgeBlockSampler
    a = RWGBasis(sphere_mesh(radius=.01, center=(-.5, 0., 0.), subdivisions=0, device='cuda'))
    b = RWGBasis(sphere_mesh(radius=.01, center=(.5, 0., 0.), subdivisions=0, device='cuda'))
    exterior = compile_medium(mw.Material(), 3e8)
    interior = compile_medium(mw.Material(eps_r=2.), 3e8)
    system = SurfaceSystem.create([(0, 'a', a, interior), (1, 'b', b, interior)])
    provider = ScalarSystemBlocks(system, exterior, IntegrationConfig(order=3, singular_order=3))
    rows = torch.arange(30, device='cuda')
    cols = torch.arange(30, 60, device='cuda')
    sampler = ScalarEdgeBlockSampler(provider, rows, cols)
    dense = provider(rows, cols)
    mixed = torch.tensor([31, 0, 30, 7, 1], device='cuda')
    torch.testing.assert_close(sampler.sample_rows(mixed), dense[mixed], rtol=2e-12, atol=1e-18)
    torch.testing.assert_close(sampler.sample_column(45), dense[:, 45], rtol=2e-12, atol=1e-18)
    rs = torch.cat((torch.full((30,), abs(exterior.eta), device='cuda', dtype=torch.float64),
                    torch.ones(30, device='cuda', dtype=torch.float64)))
    cs = rs.flip(0)
    block = streamed_cross_approximation(sampler.sample_rows, sampler.sample_column, sampler.shape,
        device=sampler.device, dtype=sampler.dtype, max_rank=32, row_batch=8,
        rtol=1e-8, row_scale=rs, column_scale=cs, pivoting=pivoting)
    error = torch.linalg.vector_norm((dense-block.left@block.right)*rs[:, None]*cs[None, :])
    reference = torch.linalg.vector_norm(dense*rs[:, None]*cs[None, :])
    assert float(error/reference) <= 1e-8
    assert block.receipt['rank'] < min(sampler.shape)
    assert block.receipt['factor_bytes'] < dense.numel()*dense.element_size()
    print({'pivoting': pivoting, 'receipt': block.receipt, 'relative_error': float(error/reference),
           'factor_bytes': block.receipt['factor_bytes'], 'dense_bytes': dense.numel()*dense.element_size()})


@pytest.mark.parametrize('pivoting', ['complete', 'adaptive'])
def test_nonbeneficial_exact_factorization_is_rejected(pivoting):
    a = torch.eye(4, device='cuda', dtype=torch.complex128)
    with pytest.raises(CompressionFailure, match='do not reduce storage') as caught:
        streamed_cross_approximation(lambda ids: a[ids], lambda i: a[:, i], a.shape,
            device=a.device, max_rank=4, rtol=1e-12, pivoting=pivoting)
    assert caught.value.receipt['converged']
    assert not caught.value.receipt['storage_reduced']


def test_adaptive_zero_update_cannot_hide_an_unsampled_component():
    a = torch.zeros((40, 40), device='cuda', dtype=torch.complex128)
    a[:10, :10] = 1.
    a[20:, 20:] = 7.j
    block = streamed_cross_approximation(lambda ids: a[ids], lambda i: a[:, i], a.shape,
        device=a.device, max_rank=4, rtol=1e-12, pivoting='adaptive', row_batch=4)
    assert block.receipt['full_residual_verified']
    assert block.receipt['residual_scans'] >= 2
    torch.testing.assert_close(block.left@block.right, a, rtol=1e-12, atol=1e-12)
