"""Nested moments checked against independently integrated internal supports."""
import pytest
import torch
from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh
from witwin.maxwell.mom.cluster import build_support_octree
from witwin.maxwell.mom.nested_moments import NestedRWGMoments
from witwin.maxwell.mom.rwg_moments import RWGInterpolationMoments

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_nested_rwg_projection_matches_direct_nodes_and_adjoint():
    basis = RWGBasis(sphere_mesh(radius=.1, subdivisions=0, device='cuda'))
    triangles = basis.mesh.vertices[basis.mesh.faces]
    supports = torch.cat((triangles[basis.plus_faces], triangles[basis.minus_faces]), dim=1)
    tree = build_support_octree(supports.amin(1), supports.amax(1), leaf_size=4)
    nested = NestedRWGMoments(basis, tree, order=3, quadrature_order=8)
    assert int(tree.depth.max()) >= 2
    x = torch.randn((2, basis.n_edges), device='cuda', dtype=torch.complex128, requires_grad=True)
    projected = nested.upward(x)
    for node in range(tree.node_count):
        if not nested.children[node]:
            continue
        edges = tree.permutation[nested.starts[node]:nested.stops[node]]
        direct = RWGInterpolationMoments(basis, edges, nested.boxes[node], quadrature_order=8)
        expected = torch.cat((torch.einsum('...e,eac->...ac', x[..., edges], direct.vector),
                              (x[..., edges]@direct.divergence)[..., None]), dim=-1)
        torch.testing.assert_close(projected[node], expected, rtol=2e-11, atol=1e-15)
    y = tuple(torch.randn_like(v) for v in projected)
    loss = sum((a.conj()*b).sum().real for a, b in zip(projected, y))
    backward = nested.downward(y)
    torch.testing.assert_close(torch.autograd.grad(loss, x)[0], backward, rtol=2e-12, atol=1e-14)
    torch.testing.assert_close(loss, (x.conj()*backward).sum().real, rtol=2e-12, atol=1e-14)
    assert nested.receipt()['internal_rwg_matrices_retained'] is False
    next(iter(nested.transfers.values())).add_(1)
    with pytest.raises(ValueError, match='changed'):
        nested.upward(x)


def test_global_mixed_leaf_preserves_body_order_and_exterior_region():
    import witwin.maxwell as mw
    from witwin.maxwell.mom.mesh import SurfaceSystem
    from witwin.maxwell.mom.physics import compile_medium
    from witwin.maxwell.mom.nearfield import ScalarSystemBlocks, scalar_regional_block
    from witwin.maxwell.mom.operator import IntegrationConfig
    a = RWGBasis(sphere_mesh(radius=.01, center=(-.1, 0., 0.), subdivisions=0, device='cuda'))
    b = RWGBasis(sphere_mesh(radius=.01, center=(.1, 0., 0.), subdivisions=0, device='cuda'))
    exterior = compile_medium(mw.Material(), 3e8)
    interior = compile_medium(mw.Material(eps_r=2., mu_r=1.3), 3e8)
    provider = ScalarSystemBlocks(SurfaceSystem.create([(0, 'a', a, interior), (1, 'b', b, interior)]),
                                 exterior, IntegrationConfig(order=2, singular_order=3))
    tree = build_support_octree(*provider.support_boxes(), leaf_size=64)
    nested = NestedRWGMoments(provider, tree, order=3, quadrature_order=6)
    assert len(nested.leaves) == 1
    x = torch.randn(60, device='cuda', dtype=torch.complex128)
    actual = nested.upward(x)[0]
    expected = torch.zeros_like(actual)
    for body in provider.system.bodies:
        edges = torch.arange(30, device='cuda')
        local = RWGInterpolationMoments(body.basis, edges, nested.boxes[0], quadrature_order=6)
        value = x[body.edge_offset:body.edge_offset+30]
        expected += torch.cat((torch.einsum('e,eac->ac', value, local.vector), (value@local.divergence)[:, None]), dim=-1)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=1e-16)
    rows = torch.tensor([30, 0, 59, 29], device='cuda')
    cols = torch.tensor([59, 29, 30, 0], device='cuda')
    full, ext = provider(rows, cols), provider.exterior_block(rows, cols)
    interior_blocks = torch.zeros_like(full)
    for body in provider.system.bodies:
        ri = ((rows >= body.edge_offset) & (rows < body.edge_offset+30)).nonzero().flatten()
        ci = ((cols >= body.edge_offset) & (cols < body.edge_offset+30)).nonzero().flatten()
        block, _ = scalar_regional_block(body.basis, body.basis, rows[ri]-body.edge_offset,
            cols[ci]-body.edge_offset, (interior,), provider.config)
        rr, cc = torch.cat((ri, ri+4)), torch.cat((ci, ci+4))
        interior_blocks[rr[:, None], cc] = block
    torch.testing.assert_close(full, ext+interior_blocks, rtol=2e-12, atol=1e-16)
    provider.edge_stops.add_(1)
    with pytest.raises(ValueError, match='routing changed'):
        nested.upward(x)
