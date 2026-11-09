"""Nested physical RWG polynomial moments against independent node integration."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.cluster import build_support_octree
from witwin.maxwell.mom.fmm_moments import NestedRWGMultipoles, rwg_polynomial_moments
from witwin.maxwell.mom.fmm_expansion import multi_indices

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def provider():
    exterior = compile_medium(mw.Material(), 3e8)
    interior = compile_medium(mw.Material(eps_r=2.), 3e8)
    system = SurfaceSystem.create([(i, str(i), RWGBasis(sphere_mesh(radius=.1,
        center=(x, 0., 0.), subdivisions=1, device='cuda')), interior) for i, x in enumerate((-1.5, 1.5))])
    return ScalarSystemBlocks(system, exterior, IntegrationConfig(order=2, singular_order=3))


def test_global_nested_multipoles_match_every_node_and_adjoint(monkeypatch):
    blocks = provider()
    lower, upper = blocks.support_boxes()
    tree = build_support_octree(lower, upper, leaf_size=8)
    assert int(tree.depth.max()) >= 2
    nested = NestedRWGMultipoles(blocks, tree, order=4)
    currents = torch.randn((2, tree.size), device='cuda', dtype=torch.complex128, requires_grad=True)
    values = nested.upward(currents)
    degrees = torch.tensor([sum(alpha) for alpha in multi_indices(4)], device='cuda')
    maximum = 0.
    for node in range(tree.node_count):
        edges = tree.permutation[nested.starts[node]:nested.stops[node]]
        direct = torch.zeros_like(values[node])
        for bi, selected in blocks._body_selection(edges):
            body = blocks.system.bodies[bi]
            vector, divergence = rwg_polynomial_moments(body.basis, edges[selected]-body.edge_offset,
                nested.centers[node], 4, geometry=blocks.geometry[bi], quadrature_order=7)
            local = currents[..., edges[selected]]
            direct = direct+torch.cat((torch.einsum('...e,eac->...ac', local, vector),
                                      (local@divergence)[..., None]), dim=-1)
        scaling = nested.radii[node].pow(degrees)[..., None]
        error = torch.linalg.vector_norm((values[node]-direct)/scaling)/torch.linalg.vector_norm(direct/scaling)
        maximum = max(maximum, float(error.detach()))
    print('all-node scaled moment error', maximum, 'depth', int(tree.depth.max()))
    assert maximum < 1e-11
    coefficients = tuple(torch.randn_like(value) for value in values)
    pairing = sum((coefficient.conj()*value).sum().real for coefficient, value in zip(coefficients, values))
    derivative, = torch.autograd.grad(pairing, currents)
    torch.testing.assert_close(derivative, nested.downward(coefficients), rtol=2e-12, atol=1e-12)
    assert nested.receipt()['polynomial_exact_in_exact_arithmetic']
    assert nested.receipt()['owned_retained_bytes'] == sum(t.numel()*t.element_size() for t in nested.tensors())
    import witwin.maxwell.mom.fmm_moments as module
    def forbidden(*args, **kwargs):
        raise AssertionError('Rejected budget must precede moment integration and shifts')
    monkeypatch.setattr(module, 'rwg_polynomial_moments', forbidden)
    monkeypatch.setattr(module, 'shift_matrix', forbidden)
    with pytest.raises(MemoryError, match='before moments'):
        NestedRWGMultipoles(blocks, tree, order=4, retained_budget_bytes=1)
    from dataclasses import replace
    shifted = replace(tree, lower=tree.lower+10, upper=tree.upper+10)
    with pytest.raises(ValueError, match='complete RWG supports'):
        NestedRWGMultipoles(blocks, shifted, order=4)
    nested.centers.add_(1)
    with pytest.raises(ValueError, match='changed'):
        nested.downward(coefficients)


def test_mixed_global_leaf_and_single_body_moments_agree():
    blocks = provider()
    lower, upper = blocks.support_boxes()
    tree = build_support_octree(lower, upper, leaf_size=blocks.system.total_edges)
    nested = NestedRWGMultipoles(blocks, tree, order=3)
    assert len(nested.leaves) == 1
    edges, vector, divergence = nested.leaves[0]
    for bi, selected in blocks._body_selection(edges):
        body = blocks.system.bodies[bi]
        expected_v, expected_d = rwg_polynomial_moments(body.basis, edges[selected]-body.edge_offset,
            nested.centers[0], 3, geometry=blocks.geometry[bi], quadrature_order=6)
        torch.testing.assert_close(vector[selected], expected_v, rtol=1e-11, atol=1e-14)
        torch.testing.assert_close(divergence[selected], expected_d, rtol=1e-11, atol=1e-14)
    body = blocks.system.bodies[0]
    body_tree = build_support_octree(lower[:body.basis.n_edges], upper[:body.basis.n_edges], leaf_size=8)
    local = NestedRWGMultipoles(body.basis, body_tree, order=3, geometry=blocks.geometry[0])
    assert local.receipt()['integrated_leaves'] > 1
    low_order = NestedRWGMultipoles(body.basis, body_tree, order=3, quadrature_order=1, geometry=blocks.geometry[0])
    assert not low_order.receipt()['polynomial_exact_in_exact_arithmetic']
