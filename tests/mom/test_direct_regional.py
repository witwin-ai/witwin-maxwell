"""Complete regional partitions against original weak integration."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh
from witwin.maxwell.mom.cluster import build_support_octree
from witwin.maxwell.mom.nested_moments import NestedRWGMoments
from witwin.maxwell.mom.direct_regional import DirectRegionalOperator
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.nearfield import scalar_regional_block

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def projection(center, order=6):
    basis = RWGBasis(sphere_mesh(radius=.01, center=center, subdivisions=0, device='cuda'))
    triangles = basis.mesh.vertices[basis.mesh.faces]
    supports = torch.cat((triangles[basis.plus_faces], triangles[basis.minus_faces]), dim=1)
    tree = build_support_octree(supports.amin(1), supports.amax(1), leaf_size=8)
    return NestedRWGMoments(basis, tree, order=order, quadrature_order=6)


def compare(operator, reference, eta):
    nr, nc = operator.shape[0]//2, operator.shape[1]//2
    eye = torch.eye(2*nc, device='cuda', dtype=torch.complex128)
    actual = operator.apply(eye).mT
    rs = torch.tensor([eta]*nr+[1.]*nr, device='cuda', dtype=torch.float64)
    cs = torch.tensor([1.]*nc+[eta]*nc, device='cuda', dtype=torch.float64)
    error = float(torch.linalg.vector_norm(rs[:, None]*(actual-reference)*cs)/torch.linalg.vector_norm(rs[:, None]*reference*cs))
    print('regional balanced error', error, operator.receipt())
    assert error < 1e-8
    x = torch.randn((2, 2*nc), device='cuda', dtype=torch.complex128, requires_grad=True)
    y = torch.randn((2, 2*nr), device='cuda', dtype=torch.complex128)
    torch.testing.assert_close(operator.adjoint(y), y@actual.conj(), rtol=2e-11, atol=1e-16)
    loss = (operator.apply(x).conj()*y).sum().real
    torch.testing.assert_close(torch.autograd.grad(loss, x)[0], operator.adjoint(y), rtol=2e-11, atol=1e-16)


def test_cross_region_uses_nested_root_kernel_without_far_weak_blocks(monkeypatch):
    test, source = projection((-.5, 0., 0.)), projection((.5, 0., 0.))
    medium = compile_medium(mw.Material(eps_r=1.2, mu_r=1.1, sigma_e=.0005, sigma_m=40.), 3e8)
    config = IntegrationConfig(order=6, singular_order=6)
    edges = torch.arange(test.tree.size, device='cuda')
    reference, _ = scalar_regional_block(test.basis, source.basis, edges, edges, (medium,), config)
    import witwin.maxwell.mom.direct_regional as module
    def forbid(*args, **kwargs):
        raise AssertionError('Admissible far blocks must not integrate edge pairs')
    monkeypatch.setattr(module, 'scalar_regional_block', forbid)
    operator = DirectRegionalOperator(test, source, (medium,), integration=config)
    assert len(operator.far) == 1 and operator.near is None
    compare(operator, reference, abs(medium.eta))
    with pytest.raises(MemoryError, match='before allocation'):
        DirectRegionalOperator(test, source, (medium,), kernel_budget_bytes=1)


def test_self_region_exact_partition_sums_interior_and_exterior():
    test = projection((0., 0., 0.), order=3)
    media = tuple(compile_medium(mw.Material(eps_r=e, mu_r=m), 3e8) for e, m in ((1., 1.), (2., 1.3)))
    config = IntegrationConfig(order=2, singular_order=3)
    edges = torch.arange(test.tree.size, device='cuda')
    reference, _ = scalar_regional_block(test.basis, test.basis, edges, edges, media, config)
    operator = DirectRegionalOperator(test, test, media, integration=config, admissibility=.01)
    assert not operator.far and operator.near is not None
    compare(operator, reference, abs(media[0].eta))
    test.basis.mesh.vertices.add_(.01)
    with pytest.raises(ValueError, match='geometry changed'):
        operator.apply(torch.ones(60, device='cuda', dtype=torch.complex128))


def test_multibody_system_mixes_exact_self_and_direct_cross_actions():
    from witwin.maxwell.mom.mesh import SurfaceSystem
    from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
    from witwin.maxwell.mom.direct_system import DirectSystemActions
    from witwin.maxwell.mom.operator import assemble_system
    a = RWGBasis(sphere_mesh(radius=.01, center=(-.5, 0., 0.), subdivisions=0, device='cuda'))
    b = RWGBasis(sphere_mesh(radius=.015, center=(.5, 0., 0.), subdivisions=0, device='cuda'))
    exterior, ma, mb = tuple(compile_medium(mw.Material(eps_r=e, mu_r=m), 3e8) for e, m in ((1., 1.), (2., 1.3), (1.7, 1.1)))
    system = SurfaceSystem.create([(0, 'a', a, ma), (1, 'b', b, mb)])
    config = IntegrationConfig(order=3, singular_order=3)
    provider = ScalarSystemBlocks(system, exterior, config)
    operator = DirectSystemActions(provider, order=6, quadrature_order=6, leaf_size=8)
    assert len(operator.exterior.far) == 2
    assert all(p.geometry is g for p, g in zip(operator.projections, provider.geometry))
    reference, _ = assemble_system(system, exterior, config)
    compare(operator, reference, abs(exterior.eta))
    operator.indices[0].add_(1)
    with pytest.raises(ValueError, match='changed'):
        operator.apply(torch.ones(120, device='cuda', dtype=torch.complex128))


def test_rectangular_exact_embedding_has_distinct_source_and_target_widths():
    test = projection((-.5, 0., 0.), order=2)
    basis = RWGBasis(sphere_mesh(radius=.01, center=(.5, 0., 0.), subdivisions=1, device='cuda'))
    triangles = basis.mesh.vertices[basis.mesh.faces]
    supports = torch.cat((triangles[basis.plus_faces], triangles[basis.minus_faces]), dim=1)
    tree = build_support_octree(supports.amin(1), supports.amax(1), leaf_size=32)
    source = NestedRWGMoments(basis, tree, order=2, quadrature_order=2)
    medium = compile_medium(mw.Material(), 3e8)
    config = IntegrationConfig(order=2, singular_order=3)
    rows, cols = torch.arange(test.tree.size, device='cuda'), torch.arange(source.tree.size, device='cuda')
    reference, _ = scalar_regional_block(test.basis, source.basis, rows, cols, (medium,), config)
    operator = DirectRegionalOperator(test, source, (medium,), integration=config, admissibility=.0001)
    assert operator.shape == (60, 240) and operator.near is not None and not operator.far
    compare(operator, reference, abs(medium.eta))
