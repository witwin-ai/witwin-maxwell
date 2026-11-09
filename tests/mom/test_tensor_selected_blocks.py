"""Selected tensor assembly ordering, bounded work and original dense agreement."""
from types import SimpleNamespace
import pytest
import torch
from witwin.maxwell.mom.mesh import SurfaceMesh, RWGBasis, sphere_mesh
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.anisotropic_operator import assemble_tensor_region_weak, tensor_regional_block

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def tetrahedron():
    vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                            device='cuda', dtype=torch.float64)*.05
    faces = torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], device='cuda')
    return RWGBasis(SurfaceMesh(vertices, faces))


class ManufacturedKernel:
    """Nonreciprocal regular values check assembly, not a physical Green function."""
    def __init__(self, medium):
        self.medium = medium
        self.mode_cache = SimpleNamespace(stats=lambda: {'manufactured': True})
        self.value = torch.arange(36, device='cuda', dtype=torch.float64).reshape(6, 6).to(torch.complex128)
        self.value = (self.value+1j*self.value.mT)/1000

    def evaluate_regular(self, points):
        return self.value.expand(len(points), 6, 6), dict(converged=True,
            static_diagonal_restored=False, static_diagonal_singularity_subtracted=True)


@pytest.mark.parametrize('cross', (False, True))
def test_selected_tensor_matches_dense_without_full_assembly(cross, monkeypatch):
    test = tetrahedron()
    source = (RWGBasis(sphere_mesh(radius=.02, center=(.5, 0., 0.), subdivisions=0, device='cuda'))
              if cross else test)
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    eps = torch.diag(torch.tensor([2., 2.3, 2.7], device='cuda', dtype=torch.complex128))+.1j*eye
    mu = torch.tensor([[1.4, .1, 0.], [.1, 1.7, .08], [0., .08, 1.2]], device='cuda', dtype=torch.complex128)+.05j*eye
    medium = TensorMedium(3e8, eps, mu)
    kernel = ManufacturedKernel(medium)
    config = IntegrationConfig(order=3, singular_order=3, pair_batch=2, far_pair_batch=3)
    dense, _ = assemble_tensor_region_weak(test, medium, config, kernel=kernel, source_basis=source)
    rows = torch.tensor([5, 0, 2], device='cuda')
    cols = torch.tensor([source.n_edges-1, 1], device='cuda')
    rr, cc = torch.cat((rows, rows+test.n_edges)), torch.cat((cols, cols+source.n_edges))
    def forbidden(*args, **kwargs):
        raise AssertionError('Selected tensor blocks must not assemble or classify every panel pair')
    import witwin.maxwell.mom.anisotropic_operator as module
    import witwin.maxwell.mom.quadrature as quadrature
    monkeypatch.setattr(module, 'assemble_tensor_region_weak', forbidden)
    monkeypatch.setattr(quadrature, 'classify_all_pairs', forbidden)
    actual, receipt = tensor_regional_block(test, source, rows, cols, medium, config, kernel=kernel)
    torch.testing.assert_close(actual, dense[rr[:, None], cc], rtol=2e-12, atol=1e-14)
    assert receipt['maximum_classification_pairs'] <= 3
    assert receipt['physical_qualification'] is False
    x = torch.randn((2, len(cc)), device='cuda', dtype=actual.dtype, requires_grad=True)
    y = torch.randn((2, len(rr)), device='cuda', dtype=actual.dtype)
    gradient, = torch.autograd.grad(x@actual.mT, x, y)
    torch.testing.assert_close(gradient, y@dense[rr[:, None], cc].conj(), rtol=2e-12, atol=1e-14)
    with pytest.raises(ValueError, match='unique'):
        tensor_regional_block(test, source, rows[:1].expand(2), cols, medium, config, kernel=kernel)
    test.mesh.vertices.add_(0)
    with pytest.raises(ValueError, match='changed'):
        tensor_regional_block(test, source, rows, cols, medium, config, kernel=kernel)


def test_selected_isotropic_tensor_preserves_scalar_reduction():
    basis = tetrahedron()
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    medium = TensorMedium(3e8, (2.+.1j)*eye, (1.3+.05j)*eye)
    config = IntegrationConfig(order=3, singular_order=4, pair_batch=2, far_pair_batch=3)
    full, _ = assemble_tensor_region_weak(basis, medium, config)
    rows, cols = torch.tensor([4, 0], device='cuda'), torch.tensor([5, 1, 3], device='cuda')
    block, receipt = tensor_regional_block(basis, basis, rows, cols, medium, config)
    rr, cc = torch.cat((rows, rows+basis.n_edges)), torch.cat((cols, cols+basis.n_edges))
    torch.testing.assert_close(block, full[rr[:, None], cc], rtol=2e-12, atol=1e-14)
    assert receipt['kernel_specialization'] == 'exact_isotropic_scalar_weak'


def test_selected_tensor_near_refinement_and_rejection():
    from dataclasses import replace
    test = tetrahedron()
    source = RWGBasis(SurfaceMesh(test.mesh.vertices+test.mesh.vertices.new_tensor([.06, 0., 0.]),
                                 test.mesh.faces.clone()))
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    medium = TensorMedium(3e8, torch.diag(eye.new_tensor([2., 2.3, 2.7]))+.1j*eye, (1.3+.1j)*eye)
    kernel = ManufacturedKernel(medium)
    rows, cols = torch.tensor([5, 0], device='cuda'), torch.tensor([4, 1], device='cuda')
    config = IntegrationConfig(order=4, singular_order=2, pair_batch=1, far_pair_batch=3,
                               near_tolerance=1e-5, near_max_order=32)
    block, receipt = tensor_regional_block(test, source, rows, cols, medium, config, kernel=kernel)
    assert receipt['triangle_pair_counts'][3] > 0
    assert receipt['maximum_panel_order'] > config.singular_order
    # One near pair per batch makes the adaptive stopping order identical;
    # larger batches can over-refine a pair until its batch partners converge.
    full, _ = assemble_tensor_region_weak(test, medium, config, kernel=kernel, source_basis=source)
    rr, cc = torch.cat((rows, rows+test.n_edges)), torch.cat((cols, cols+source.n_edges))
    torch.testing.assert_close(block, full[rr[:, None], cc], rtol=2e-12, atol=1e-14)
    # One refinement with a deliberately unresolved tolerance must reject.
    unresolved = replace(config, near_max_order=4, near_tolerance=1e-30)
    for action in (
        lambda: tensor_regional_block(test, source, rows, cols, medium, unresolved, kernel=kernel),
        lambda: assemble_tensor_region_weak(test, medium, unresolved, kernel=kernel, source_basis=source),
    ):
        with pytest.raises(RuntimeError, match='near-panel quadrature did not converge'):
            action()
