"""Electromagnetic multipole/local far blocks against original weak quadrature."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh
from witwin.maxwell.mom.cluster import build_support_octree
from witwin.maxwell.mom.fmm_moments import NestedRWGMultipoles
from witwin.maxwell.mom.fmm_translation import ElectromagneticM2L
from witwin.maxwell.mom.nearfield import scalar_regional_block
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.physics import compile_medium

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_magnetic_lossy_m2l_weak_block_adjoint_and_separation(monkeypatch):
    def projection(x):
        basis = RWGBasis(sphere_mesh(radius=.01, center=(x, 0., 0.), subdivisions=0, device='cuda'))
        triangles = basis.mesh.vertices[basis.mesh.faces]
        support = torch.cat((triangles[basis.plus_faces], triangles[basis.minus_faces]), dim=1)
        tree = build_support_octree(support.amin(1), support.amax(1), leaf_size=30)
        return NestedRWGMultipoles(basis, tree, order=6)
    test, source = projection(-.5), projection(.5)
    medium = compile_medium(mw.Material(eps_r=2., mu_r=1.3, sigma_e=.001, sigma_m=.002), 3e8)
    coupling = ElectromagneticM2L(test, source, 0, 0, medium)
    def action(value, adjoint=False):
        origin, target = (test, source) if adjoint else (source, test)
        n = origin.tree.size
        moments = origin.upward(torch.stack((value[..., :n], value[..., n:]), dim=-2))
        result = target.downward((coupling.action(moments[0], adjoint=adjoint),))
        return torch.cat((result[..., 0, :], result[..., 1, :]), dim=-1)
    n = test.tree.size
    eye = torch.eye(2*n, device='cuda', dtype=torch.complex128)
    actual = action(eye).mT
    reference, _ = scalar_regional_block(test.basis, source.basis, torch.arange(n, device='cuda'),
        torch.arange(n, device='cuda'), (medium,), IntegrationConfig(order=6, singular_order=3))
    eta = abs(medium.eta)
    left = torch.tensor([eta]*n+[1.]*n, device='cuda')
    right = torch.tensor([1.]*n+[eta]*n, device='cuda')
    error = torch.linalg.vector_norm(left[:, None]*(actual-reference)*right)/torch.linalg.vector_norm(left[:, None]*reference*right)
    print('M2L balanced weak error', float(error))
    assert float(error) < 1e-8
    x = torch.randn((2, 2*n), device='cuda', dtype=actual.dtype, requires_grad=True)
    probe = torch.randn_like(x)
    torch.testing.assert_close(action(probe, True), probe@actual.conj(), rtol=2e-11, atol=1e-14)
    derivative, = torch.autograd.grad(action(x), x, probe)
    torch.testing.assert_close(derivative, action(probe, True), rtol=2e-11, atol=1e-14)
    with pytest.raises(ValueError, match='separated complete-support'):
        ElectromagneticM2L(test, test, 0, 0, medium)
    from witwin.maxwell.mom.fmm_error import MultipoleTruncationError
    from witwin.maxwell.mom.interpolation_error import KernelErrorBudget
    import witwin.maxwell.mom.fmm_translation as translation
    def forbidden(*args, **kwargs):
        raise AssertionError('Failed truncation admission must precede coefficient tables')
    monkeypatch.setattr(translation, 'multipole_to_local', forbidden)
    with pytest.raises(MultipoleTruncationError):
        ElectromagneticM2L(test, source, 0, 0, medium, error_budget=KernelErrorBudget(1e-100, 1e-100))
    coupling.gradient.add_(1)
    with pytest.raises(ValueError, match='changed'):
        action(x)
