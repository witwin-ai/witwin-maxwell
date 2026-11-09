"""Complete multipole regional actions against original weak quadrature."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks, scalar_regional_block
from witwin.maxwell.mom.cluster import build_support_octree
from witwin.maxwell.mom.fmm_moments import NestedRWGMultipoles
from witwin.maxwell.mom.fmm_regional import MultipoleRegionalOperator

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def compare(op, reference, eta):
    actual = op.apply(torch.eye(op.shape[1], device='cuda', dtype=torch.complex128)).mT
    nr, nc = op.shape[0]//2, op.shape[1]//2
    left = torch.tensor([eta]*nr+[1.]*nr, device='cuda')
    right = torch.tensor([1.]*nc+[eta]*nc, device='cuda')
    error = torch.linalg.vector_norm(left[:, None]*(actual-reference)*right)/torch.linalg.vector_norm(left[:, None]*reference*right)
    print('regional FMM error', float(error), op.receipt())
    assert float(error) < 1e-8
    x = torch.randn((2, op.shape[1]), device='cuda', dtype=actual.dtype, requires_grad=True)
    y = torch.randn((2, op.shape[0]), device='cuda', dtype=actual.dtype)
    torch.testing.assert_close(op.adjoint(y), y@actual.conj(), rtol=2e-11, atol=1e-14)
    derivative, = torch.autograd.grad(op.apply(x), x, y)
    torch.testing.assert_close(derivative, op.adjoint(y), rtol=2e-11, atol=1e-14)


@pytest.mark.parametrize('table_storage', ('dense', 'compact'))
def test_global_exterior_combines_m2l_and_exact_self(monkeypatch, table_storage):
    exterior = compile_medium(mw.Material(), 3e8)
    interior = compile_medium(mw.Material(eps_r=2., mu_r=1.3), 3e8)
    system = SurfaceSystem.create([(i, str(i), RWGBasis(sphere_mesh(radius=.01,
        center=(x, 0., 0.), subdivisions=0, device='cuda')), interior) for i, x in enumerate((-.5, .5))])
    config = IntegrationConfig(order=3, singular_order=3)
    provider = ScalarSystemBlocks(system, exterior, config)
    lower, upper = provider.support_boxes()
    projection = NestedRWGMultipoles(provider, build_support_octree(lower, upper, leaf_size=30), order=6)
    requested = []
    original = provider.exterior_block
    def near_only(rows, cols):
        assert bool((rows < 30).all()) == bool((cols < 30).all())
        requested.append((len(rows), len(cols)))
        return original(rows, cols)
    monkeypatch.setattr(provider, 'exterior_block', near_only)
    op = MultipoleRegionalOperator(projection, projection, (exterior,), integration=config,
                                  table_storage=table_storage)
    assert len(op.far) == 2 and len(requested) == 2
    edges = torch.arange(system.total_edges, device='cuda')
    compare(op, original(edges, edges), abs(exterior.eta))
    # Shared geometry checks must not scale with the number of far couplings.
    checks = []
    check_projection = projection.assert_current
    def counted_check():
        checks.append(True)
        check_projection()
    monkeypatch.setattr(projection, 'assert_current', counted_check)
    value = torch.zeros(op.shape[1], device='cuda', dtype=torch.complex128)
    op.apply(value)
    assert len(checks) == 3  # Regional entry, upward and downward.
    coupling = op.far[0][2]
    moments = torch.zeros((2, projection.count, 4), device='cuda', dtype=op.dtype)
    checks.clear()
    coupling.action(moments)
    assert len(checks) == 1  # Standalone coupling still validates its space.
    # Replacements and in-place mutations remain rejected before composition.
    with monkeypatch.context() as patch:
        patch.setattr(projection, 'centers', projection.centers.clone())
        for action, arg in ((op.apply, value), (op.adjoint, value), (coupling.action, moments)):
            with pytest.raises(ValueError, match='changed'):
                action(arg)
    with monkeypatch.context() as patch:
        patch.setattr(coupling, 'test_node', coupling.test_node+1)
        with pytest.raises(ValueError, match='changed'):
            op.apply(value)
    table_owner = coupling if table_storage == 'dense' else coupling._compact
    # Select a retained tensor by identity without relying on compact names.
    table = table_owner.tensors()[0]
    table.add_(0)
    for action, arg in ((op.apply, value), (op.adjoint, value), (coupling.action, moments)):
        with pytest.raises(ValueError, match='changed'):
            action(arg)
    with pytest.raises(MemoryError, match='M2L table budget'):
        MultipoleRegionalOperator(projection, projection, (exterior,), integration=config, table_budget_bytes=1)
    op.near.groups[0][2].add_(1)
    with pytest.raises(ValueError, match='changed'):
        op.apply(torch.zeros(op.shape[1], device='cuda', dtype=torch.complex128))


def test_rectangular_near_embedding_with_different_rwg_counts():
    medium = compile_medium(mw.Material(eps_r=1.5, mu_r=1.2), 3e8)
    def projection(center, subdivision):
        basis = RWGBasis(sphere_mesh(radius=.01, center=center, subdivisions=subdivision, device='cuda'))
        triangles = basis.mesh.vertices[basis.mesh.faces]
        supports = torch.cat((triangles[basis.plus_faces], triangles[basis.minus_faces]), dim=1)
        return NestedRWGMultipoles(basis, build_support_octree(supports.amin(1), supports.amax(1), leaf_size=200), order=3)
    test, source = projection((0., 0., 0.), 0), projection((.025, 0., 0.), 1)
    config = IntegrationConfig(order=2, singular_order=3)
    op = MultipoleRegionalOperator(test, source, (medium,), integration=config)
    assert not op.far and op.shape == (60, 240)
    reference, _ = scalar_regional_block(test.basis, source.basis,
        torch.arange(30, device='cuda'), torch.arange(120, device='cuda'), (medium,), config)
    compare(op, reference, abs(medium.eta))
