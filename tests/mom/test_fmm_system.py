"""True multipole full-system composition with heterogeneous magnetic bodies."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig, assemble_system
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.fmm_system import MultipoleSystemActions

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('storage', ['dense', 'compact'])
def test_system_multipoles_match_original_and_preserve_fixed_current_derivative(monkeypatch, storage):
    exterior = compile_medium(mw.Material(), 3e8)
    bodies = [(i, str(i), RWGBasis(sphere_mesh(radius=.01, center=(x, 0., 0.),
        subdivisions=0, device='cuda')), compile_medium(mw.Material(eps_r=2.+i, mu_r=1.2+i*.1), 3e8))
        for i, x in enumerate((-.5, .5))]
    system = SurfaceSystem.create(bodies)
    config = IntegrationConfig(order=3, singular_order=3)
    provider = ScalarSystemBlocks(system, exterior, config)
    import witwin.maxwell.mom.interpolation_kernel as interpolation
    def forbidden(*args, **kwargs):
        raise AssertionError('Multipole construction must not use interpolation tables')
    monkeypatch.setattr(interpolation.InterpolationKernel, '__init__', forbidden)
    op = MultipoleSystemActions(provider, order=6, leaf_size=30, table_storage=storage)
    assert len(op.exterior.far) == 2 and len(op.interiors) == 2
    assert all(len(r.media) == 1 for r in (op.exterior, *op.interiors))
    for body, region in zip(system.bodies, op.interiors):
        assert region.media[0] is body.interior
    dense, _ = assemble_system(system, exterior, config)
    n = system.total_edges
    actual = op.apply(torch.eye(2*n, device='cuda', dtype=torch.complex128)).mT
    eta = abs(exterior.eta)
    left = torch.tensor([eta]*n+[1.]*n, device='cuda')
    right = torch.tensor([1.]*n+[eta]*n, device='cuda')
    error = torch.linalg.vector_norm(left[:, None]*(actual-dense)*right)/torch.linalg.vector_norm(left[:, None]*dense*right)
    print('global multipole weak error', float(error))
    assert float(error) < 1e-8
    x = torch.randn((2, 2*n), device='cuda', dtype=actual.dtype, requires_grad=True)
    y = torch.randn_like(x)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        applied = op.apply(x)
    assert not saved
    derivative, = torch.autograd.grad(applied, x, y)
    torch.testing.assert_close(derivative, op.adjoint(y), rtol=2e-11, atol=1e-14)
    torch.testing.assert_close(op.adjoint(y), y@actual.conj(), rtol=2e-11, atol=1e-14)
    receipt = op.receipt()
    assert receipt['table_bytes'] == op.exterior.table_bytes+sum(r.table_bytes for r in op.interiors)
    op.indices[0].add_(1)
    with pytest.raises(ValueError, match='changed'):
        op.apply(x)
