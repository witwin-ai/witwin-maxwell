"""Mixed scalar/tensor ownership and routing against complete weak assembly."""
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import SurfaceMesh, RWGBasis, SurfaceSystem
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.dyadic_green import DyadicGreenKernel
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.anisotropic_operator import assemble_generalized_system
from witwin.maxwell.mom.tensor_nearfield import GeneralizedSystemBlocks
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def basis_at(x):
    vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                            device='cuda', dtype=torch.float64)*.05
    vertices[:, 0] += x
    faces = torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], device='cuda')
    return RWGBasis(SurfaceMesh(vertices, faces))


@pytest.mark.parametrize('tensor_exterior', (False, True))
def test_generalized_selected_routing_and_sampled_action(tensor_exterior, monkeypatch):
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    tensor = TensorMedium(3e8, torch.diag(eye.new_tensor([2., 2.4, 2.8]))+.1j*eye,
                          eye.new_tensor([[1.3, .1, 0.], [.1, 1.6, 0.], [0., 0., 1.9]])+.1j*eye)
    scalar = compile_medium(mw.Material(eps_r=3., mu_r=1.5), 3e8)
    exterior = (TensorMedium(3e8, torch.diag(eye.new_tensor([1.2, 1.4, 1.6]))+.2j*eye,
                             (1.1+.1j)*eye) if tensor_exterior else compile_medium(mw.Material(), 3e8))
    system = SurfaceSystem.create([(0, 'a', basis_at(0.), tensor), (2, 'b', basis_at(.5), scalar)])
    config = IntegrationConfig(order=3, singular_order=3, pair_batch=2, far_pair_batch=3)
    # Explicit manufactured regular Green values isolate assembly ownership;
    # the anisotropic static terms still use the production weak formulation.
    def manufactured(kernel, points):
        value = torch.arange(36, device='cuda', dtype=torch.float64).reshape(6, 6).to(torch.complex128)/1000
        value = (value+1j*value.mT)*kernel.medium.epsilon_r.trace()
        return value.expand(len(points), 6, 6), dict(converged=True,
            static_diagonal_restored=False, static_diagonal_singularity_subtracted=True)
    monkeypatch.setattr(DyadicGreenKernel, 'evaluate_regular', manufactured)
    full, _ = assemble_generalized_system(system, exterior, config)
    provider = GeneralizedSystemBlocks(system, exterior, config)
    assert not isinstance(provider, ScalarSystemBlocks)
    rows, cols = torch.tensor([11, 0, 7, 4], device='cuda'), torch.tensor([1, 9, 6], device='cuda')
    n = system.total_edges
    rr, cc = torch.cat((rows, rows+n)), torch.cat((cols, cols+n))
    def forbidden(*args, **kwargs):
        raise AssertionError('Selected provider must not assemble a complete system')
    import witwin.maxwell.mom.anisotropic_operator as assembly
    monkeypatch.setattr(assembly, 'assemble_generalized_system', forbidden)
    monkeypatch.setattr(assembly, 'assemble_tensor_region_weak', forbidden)
    torch.testing.assert_close(provider(rows, cols), full[rr[:, None], cc], rtol=2e-12, atol=1e-13)
    assert provider.metadata['maximum_classification_pairs'] <= 3
    # Cross-body blocks contain only the common exterior.
    a, b = torch.tensor([0, 2], device='cuda'), torch.tensor([7, 10], device='cuda')
    torch.testing.assert_close(provider(a, b), provider.exterior_block(a, b), rtol=0, atol=0)
    current = torch.randn((2, 2*n), device='cuda', dtype=torch.complex128, requires_grad=True)
    actual = provider.sampled_action(rows, current, source_batch=3)
    torch.testing.assert_close(actual, current@full[rr].mT, rtol=2e-12, atol=1e-12)
    probe = torch.randn_like(actual)
    gradient, = torch.autograd.grad(actual, current, probe)
    torch.testing.assert_close(gradient, probe@full[rr].conj(), rtol=2e-12, atol=1e-12)
    assert len(provider.support_boxes()[0]) == n
    pointers = {t.untyped_storage().data_ptr() for t in provider.input_tensors()}
    assert tensor.epsilon_r.untyped_storage().data_ptr() in pointers
    saved = provider.interior_kernels
    provider.interior_kernels = (replace(saved[0]), saved[1])
    with pytest.raises(ValueError, match='binding changed'):
        provider(rows, cols)
    provider.interior_kernels = saved
    tensor.mu_r.add_(0)
    with pytest.raises(ValueError, match='medium changed'):
        provider(rows, cols)


def test_generalized_provider_rejects_wrong_kernel_ownership():
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    medium = TensorMedium(3e8, 2*eye, 1.3*eye)
    other = TensorMedium(3e8, 2*eye, 1.3*eye)
    exterior = compile_medium(mw.Material(), 3e8)
    system = SurfaceSystem.create([(0, 'a', basis_at(0.), medium)])
    with pytest.raises(ValueError, match='exact regional'):
        GeneralizedSystemBlocks(system, exterior, interior_kernels={1: DyadicGreenKernel(other)})
    with pytest.raises(ValueError, match='tensor body regions'):
        GeneralizedSystemBlocks(system, exterior, interior_kernels={True: DyadicGreenKernel(medium)})
    with pytest.raises(ValueError, match='Scalar regions'):
        GeneralizedSystemBlocks(system, exterior, exterior_kernel=DyadicGreenKernel(medium))


def test_mixed_tensor_provider_drives_actual_h_and_h2_far_actions(monkeypatch):
    from witwin.maxwell.mom.mesh import sphere_mesh
    from witwin.maxwell.mom.hmatrix import HierarchicalOperator, HMatrixConfig
    from witwin.maxwell.mom.h2_operator import HierarchicalH2Operator
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    medium = TensorMedium(3e8, torch.diag(eye.new_tensor([2., 2.4, 2.8]))+.1j*eye,
                          torch.diag(eye.new_tensor([1.3, 1.6, 1.9]))+.1j*eye)
    exterior = compile_medium(mw.Material(), 3e8)
    scalar = compile_medium(mw.Material(eps_r=3., mu_r=1.5), 3e8)
    a = RWGBasis(sphere_mesh(radius=.01, center=(-.5, 0., 0.), subdivisions=0, device='cuda'))
    b = RWGBasis(sphere_mesh(radius=.01, center=(.5, 0., 0.), subdivisions=0, device='cuda'))
    system = SurfaceSystem.create([(0, 'tensor', a, medium), (1, 'scalar', b, scalar)])
    config = IntegrationConfig(order=3, singular_order=3, pair_batch=1, far_pair_batch=3)
    def manufactured(kernel, points):
        value = torch.arange(36, device='cuda', dtype=torch.float64).reshape(6, 6).to(torch.complex128)/1000
        return (value+1j*value.mT).expand(len(points), 6, 6), dict(converged=True,
            static_diagonal_restored=False, static_diagonal_singularity_subtracted=True)
    monkeypatch.setattr(DyadicGreenKernel, 'evaluate_regular', manufactured)
    dense, _ = assemble_generalized_system(system, exterior, config)
    h = HierarchicalOperator(GeneralizedSystemBlocks(system, exterior, config),
                             HMatrixConfig(pivoting='adaptive'))
    print({'far_blocks': h.metadata['far_blocks'], 'compression_rejections': h.metadata['compression_rejections']})
    assert h.metadata['far_blocks'] > 0
    h2 = HierarchicalH2Operator(h)
    assert len(h2.actions.couplings) > 0
    x = torch.randn((2, 2*system.total_edges), device='cuda', dtype=torch.complex128, requires_grad=True)
    y = torch.randn_like(x)
    for op in (h, h2):
        # Check both output row families independently to retain physical units.
        actual, expected = op.apply(x), x@dense.mT
        for av, ev in zip(actual.chunk(2, dim=-1), expected.chunk(2, dim=-1)):
            assert float(((av-ev).norm()/ev.norm()).detach()) < 1e-8
        actual, expected = op.adjoint(y), y@dense.conj()
        for av, ev in zip(actual.chunk(2, dim=-1), expected.chunk(2, dim=-1)):
            assert float(((av-ev).norm()/ev.norm()).detach()) < 1e-8
        gradient, = torch.autograd.grad(op.apply(x), x, y)
        torch.testing.assert_close(gradient, op.adjoint(y), rtol=1e-11, atol=1e-12)
    medium.epsilon_r.add_(0)
    for op in (h, h2):
        with pytest.raises(ValueError, match='medium changed'):
            op.apply(x)
