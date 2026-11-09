"""Direct kernel interpolation against independent RWG weak integration."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.interpolation import InterpolationBox
from witwin.maxwell.mom.rwg_moments import RWGInterpolationMoments
from witwin.maxwell.mom.interpolated_far import InterpolatedRegionalBlock
from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.nearfield import scalar_regional_block

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def real(value):
    return torch.tensor(value, device='cuda', dtype=torch.float64)


def test_cardinal_polynomials_nested_transfer_and_collapsed_axis():
    parent = InterpolationBox(real([2., -3., 4.]), real([4., 1., 4.]), 5)
    child = InterpolationBox(real([2.5, -2., 4.]), real([3.5, 0., 4.]), 5)
    points = child.lower+torch.rand((19, 3), device='cuda', dtype=torch.float64)*(child.upper-child.lower)
    polynomial = lambda p: p[..., 0]**3*p[..., 1]**2+2*p[..., 0]-p[..., 1]
    torch.testing.assert_close(parent.evaluate(parent.nodes), torch.eye(25, device='cuda', dtype=torch.float64), atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(parent.evaluate(points)@polynomial(parent.nodes), polynomial(points), atol=2e-12, rtol=2e-13)
    torch.testing.assert_close(child.evaluate(points)@parent.transfer_to(child), parent.evaluate(points), atol=2e-14, rtol=2e-13)
    parent.nodes.add_(1)
    with pytest.raises(ValueError, match='changed'):
        parent.evaluate(points)


def moments(basis, edges, order):
    v = basis.mesh.vertices
    return RWGInterpolationMoments(basis, edges, InterpolationBox(v.amin(0), v.amax(0), order), quadrature_order=4, face_batch=3)


def test_far_refinement_rectangular_adjoint_gradient_and_guards():
    a = RWGBasis(sphere_mesh(radius=.01, center=(-.5, 0., 0.), subdivisions=0, device='cuda'))
    b = RWGBasis(sphere_mesh(radius=.015, center=(.5, 0., 0.), subdivisions=0, device='cuda'))
    rows = torch.tensor([13, 0, 29], device='cuda')
    cols = torch.tensor([29, 12, 2, 15, 0], device='cuda')
    medium = compile_medium(mw.Material(eps_r=1.2, mu_r=1.1, sigma_e=.0005, sigma_m=40.), 3e8)
    reference, _ = scalar_regional_block(a, b, rows, cols, (medium,), IntegrationConfig(order=4, singular_order=4))
    errors = []
    for order in (2, 5, 6):
        test, source = moments(a, rows, order), moments(b, cols, order)
        block = InterpolatedRegionalBlock(test, source, medium, kernel_row_batch=7)
        matrix = block.apply(torch.eye(10, device='cuda', dtype=torch.complex128)).mT
        rs = real([abs(medium.eta)]*3+[1.]*3)
        cs = real([1.]*5+[abs(medium.eta)]*5)
        errors.append(float(torch.linalg.vector_norm(rs[:, None]*(matrix-reference)*cs)/torch.linalg.vector_norm(rs[:, None]*reference*cs)))
    print('interpolation balanced errors', errors)
    assert errors[2] < 1e-8 and errors[2] < errors[1] and errors[1] < errors[0]*1e-3
    torch.testing.assert_close(test.divergence.sum(-1), torch.zeros(3, device='cuda', dtype=torch.complex128), rtol=0, atol=2e-16)
    x = torch.randn((2, 10), device='cuda', dtype=torch.complex128, requires_grad=True)
    y = torch.randn((2, 6), device='cuda', dtype=torch.complex128)
    torch.testing.assert_close(block.adjoint(y), y@matrix.conj(), rtol=1e-11, atol=1e-20)
    loss = (block.apply(x).conj()*y).sum().real
    torch.testing.assert_close(torch.autograd.grad(loss, x)[0], block.adjoint(y), rtol=1e-11, atol=1e-20)
    with pytest.raises(MemoryError, match='before allocation'):
        InterpolatedRegionalBlock(test, source, medium, kernel_budget_bytes=1)
    with pytest.raises(ValueError, match='admissible'):
        InterpolatedRegionalBlock(test, test, medium)
    v = a.mesh.vertices
    with pytest.raises(ValueError, match='full support'):
        RWGInterpolationMoments(a, rows, InterpolationBox(v.amin(0)*.5, v.amax(0)*.5, 3))
    block.kernel.gradient.add_(1)
    with pytest.raises(ValueError, match='changed'):
        block.apply(x)


def test_moments_preserve_edge_order_and_integrated_vector():
    basis = RWGBasis(sphere_mesh(radius=.1, subdivisions=0, device='cuda'))
    edges = torch.tensor([29, 0, 13, 7], device='cuda')
    projected = moments(basis, edges, 4)
    reversed_moments = moments(basis, edges.flip(0), 4)
    torch.testing.assert_close(projected.vector, reversed_moments.vector.flip(0), rtol=0, atol=0)
    torch.testing.assert_close(projected.divergence, reversed_moments.divergence.flip(0), rtol=0, atol=0)
    geo = projected.geometry
    triangles = geo.triangles
    areas = .5*torch.linalg.vector_norm(torch.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0], dim=-1), dim=-1)
    local = geo.slopes[..., None]*areas[:, None, None]*(triangles.mean(1)[:, None]-triangles)
    integrated = torch.zeros((basis.n_edges, 3), device='cuda', dtype=torch.float64)
    integrated.index_add_(0, basis.face_edges.flatten(), local.reshape(-1, 3))
    torch.testing.assert_close(projected.vector.sum(1), integrated[edges].to(torch.complex128), rtol=2e-13, atol=1e-17)
    projected.edges[0] = 1
    with pytest.raises(ValueError, match='changed'):
        projected.receipt()


@pytest.mark.parametrize('changed', ['vertices', 'faces'])
def test_moments_reject_stale_basis_before_selection(changed, monkeypatch):
    basis = RWGBasis(sphere_mesh(radius=.1, subdivisions=0, device='cuda'))
    edges = torch.tensor([0], device='cuda')
    getattr(basis.mesh, changed).add_(0)
    import witwin.maxwell.mom.rwg_moments as module
    def forbid(*args):
        raise AssertionError('Stale basis must reject before using cached indices')
    monkeypatch.setattr(module, '_selected_edges', forbid)
    with pytest.raises(ValueError, match='Basis geometry changed'):
        moments(basis, edges, 3)
