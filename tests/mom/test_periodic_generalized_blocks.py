"""Selected tensor Ewald blocks and streamed actions against dense assembly."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import SurfaceSystem
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.dyadic_green import DyadicGreenKernel
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.periodic import LatticeSpec, EwaldHelmholtzKernel
from witwin.maxwell.mom.periodic_operator import assemble_periodic_system
from witwin.maxwell.mom.ewald_nearfield import GeneralizedEwaldSystemBlocks
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from test_generalized_system_blocks import basis_at

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('dimension', (1, 2, 3))
@pytest.mark.parametrize('manufactured', (False, True))
def test_generalized_ewald_selected_action_and_gradient(dimension, manufactured, monkeypatch):
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    epsilon = torch.diag(eye.new_tensor([2., 2.4, 2.8]))+.1j*eye if manufactured else 2.*eye
    tensor = TensorMedium(3e8, epsilon, 1.3*eye)
    scalar = compile_medium(mw.Material(eps_r=3., mu_r=1.1), 3e8)
    exterior = compile_medium(mw.Material(sigma_e=.02), 3e8)
    system = SurfaceSystem.create([(0, 'tensor', basis_at(0.), tensor),
                                   (2, 'scalar', basis_at(.5), scalar)])
    sample = system.bodies[0].mesh.vertices
    lattice = LatticeSpec(2.*torch.eye(3, device='cuda', dtype=sample.dtype)[:dimension],
                          sample.new_tensor([.2, 0., 0.]))
    kernel = EwaldHelmholtzKernel(lattice, 2., real_shells=2, reciprocal_shells=3,
                                  quadrature_order=32)
    config = IntegrationConfig(order=2, singular_order=3, pair_batch=8)
    if manufactured:
        def regular(bound, points):
            value = torch.arange(36, device='cuda', dtype=sample.dtype).reshape(6, 6).to(eye.dtype)/1000
            value = (value+1j*value.mT)*bound.medium.epsilon_r.trace()
            return value.expand(len(points), 6, 6), dict(converged=True,
                static_diagonal_restored=False, static_diagonal_singularity_subtracted=True)
        monkeypatch.setattr(DyadicGreenKernel, 'evaluate_regular', regular)
    regional = {1: DyadicGreenKernel(tensor)}
    provider = GeneralizedEwaldSystemBlocks(system, exterior, config, kernel,
                                           interior_kernels=regional)
    assert not isinstance(provider, ScalarSystemBlocks)
    assert provider.interior_kernels[0] is regional[1]
    dense, _ = assemble_periodic_system(system, exterior, config, kernel=kernel,
                                        interior_kernels=regional)
    n = system.total_edges
    rows = torch.tensor([n-1, 0, n//2], device='cuda')
    cols = torch.arange(n-1, -1, -1, device='cuda')
    full_rows, full_cols = torch.cat((rows, rows+n)), torch.cat((cols, cols+n))
    block = provider(rows, cols)
    torch.testing.assert_close(block, dense[full_rows][:, full_cols], rtol=5e-10, atol=2e-13)
    generator = torch.Generator(device='cuda').manual_seed(241)
    current = torch.randn((2, 2*n), device='cuda', dtype=eye.dtype,
                           generator=generator, requires_grad=True)
    expected = current@dense[full_rows].T
    def forbidden(*args, **kwargs):
        raise AssertionError('Streamed smooth action must not materialize selected regular blocks')
    monkeypatch.setattr(provider, 'regular_block', forbidden)
    actual = provider.sampled_action(rows, current, source_batch=3)
    torch.testing.assert_close(actual, expected, rtol=5e-10, atol=2e-13)
    probe = torch.randn(actual.shape, device='cuda', dtype=eye.dtype, generator=generator)
    actual_gradient, = torch.autograd.grad(actual, current, probe)
    torch.testing.assert_close(actual_gradient, probe@dense[full_rows].conj(), rtol=5e-10, atol=2e-13)
    assert provider.metadata['physical_qualification'] is False
    with pytest.raises(ValueError, match='unique'):
        provider.sampled_action(rows[[0, 0]], current)
    provider.kernel = EwaldHelmholtzKernel(lattice, 2., real_shells=2,
                                          reciprocal_shells=3, quadrature_order=32)
    with pytest.raises(ValueError, match='binding changed'):
        provider.assert_current()
