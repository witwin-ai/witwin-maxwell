"""Linear-storage metric against dense independent solves and gradients."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.riesz import LocalRWGRiesz

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def _system(subdivisions=0):
    medium = compile_medium(mw.Material(eps_r=2.), 3e8)
    a = RWGBasis(sphere_mesh(radius=.1, center=(-.3, 0., 0.), subdivisions=subdivisions, device='cuda'))
    b = RWGBasis(sphere_mesh(radius=.07, center=(.3, 0., 0.), subdivisions=subdivisions, device='cuda'))
    return SurfaceSystem.create([(0, 'a', a, medium), (1, 'b', b, medium)])


def test_multibody_local_metric_action_inverse_gradient_and_zero_rhs(monkeypatch):
    system = _system()
    gram = torch.block_diag(*(b.basis.mass_matrix() for b in system.bodies)).to(torch.complex128)
    dense = torch.block_diag(gram/377.**2, gram)
    def forbid(*args, **kwargs):
        raise AssertionError('Local Riesz must never allocate a global dense Gram')
    monkeypatch.setattr(RWGBasis, 'mass_matrix', forbid)
    metric = LocalRWGRiesz(system, 377.)
    g = torch.Generator(device='cuda').manual_seed(4)
    rhs = torch.randn((3, len(dense)), generator=g, device='cuda', dtype=torch.complex128)
    rhs[0] = 0
    rhs.requires_grad_()
    torch.testing.assert_close(metric.apply(rhs), rhs@dense.mT, rtol=2e-13, atol=1e-16)
    actual = metric.inverse(rhs)
    expected = torch.linalg.solve(dense, rhs.mT).mT
    torch.testing.assert_close(actual, expected, rtol=3e-10, atol=1e-6)
    relative = torch.linalg.vector_norm(actual@dense.mT-rhs, dim=-1)[1:]/torch.linalg.vector_norm(rhs, dim=-1)[1:]
    assert float(relative.detach().max()) <= 1.1e-12
    loss = .5*(rhs.conj()*actual).real.sum()
    gradient, = torch.autograd.grad(loss, rhs)
    torch.testing.assert_close(gradient, expected, rtol=3e-10, atol=1e-6)
    assert metric.last_solve['converged'] and metric.last_solve['actual_residual_checked']
    assert metric.receipt()['retained_bytes'] < dense.numel()*dense.element_size()
    metric.diagonal.add_(1)
    with pytest.raises(ValueError, match='changed'):
        metric.inverse(rhs)


def test_metric_storage_growth_and_inverse_failure():
    small, large = _system(0), _system(1)
    first, second = LocalRWGRiesz(small, 377.), LocalRWGRiesz(large, 377.)
    assert second.receipt()['retained_bytes'] == 4*first.receipt()['retained_bytes']
    failed = LocalRWGRiesz(small, 377., maxiter=1)
    rhs = torch.arange(failed.shape[0], device='cuda', dtype=torch.float64).to(torch.complex128)+1j
    with pytest.raises(RuntimeError, match='did not converge'):
        failed.inverse(rhs)
    assert not failed.last_solve['converged']
    small.bodies[0].basis.edge_lengths.add_(.01)
    with pytest.raises(ValueError, match='source basis changed'):
        first.apply(rhs)
