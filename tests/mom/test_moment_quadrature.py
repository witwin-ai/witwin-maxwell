"""Polynomial degree admission for exactly integrated RWG interpolation moments."""
import pytest
import torch
from witwin.maxwell.mom.interpolation import InterpolationBox
from witwin.maxwell.mom.rwg_moments import RWGInterpolationMoments, required_moment_quadrature_order
from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh
from witwin.maxwell.mom.cluster import build_support_octree
from witwin.maxwell.mom.nested_moments import NestedRWGMoments

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_automatic_polynomial_degree_matches_overintegration_and_detects_low_order():
    basis = RWGBasis(sphere_mesh(radius=.1, subdivisions=0, device='cuda'))
    v = basis.mesh.vertices
    box = InterpolationBox(v.amin(0), v.amax(0), 6)
    edges = torch.tensor([29, 0, 7, 14], device='cuda')
    automatic = RWGInterpolationMoments(basis, edges, box)
    high = RWGInterpolationMoments(basis, edges, box, quadrature_order=12)
    low = RWGInterpolationMoments(basis, edges, box, quadrature_order=3)
    assert automatic.quadrature_order == 9
    assert automatic.receipt()['moment_quadrature']['polynomial_exact_in_exact_arithmetic']
    assert not low.receipt()['moment_quadrature']['polynomial_exact_in_exact_arithmetic']
    torch.testing.assert_close(automatic.vector, high.vector, rtol=2e-10, atol=2e-16)
    torch.testing.assert_close(automatic.divergence, high.divergence, rtol=2e-10, atol=2e-15)
    error = float(torch.linalg.vector_norm(low.vector-high.vector)/torch.linalg.vector_norm(high.vector))
    print('underintegrated moment relative error', error)
    assert error > 1e-4


def test_collapsed_axis_degree_and_nested_parent_integrals():
    point = torch.zeros(3, device='cuda', dtype=torch.float64)
    for upper, required in (([1., 1., 1.], 9), ([1., 1., 0.], 7), ([1., 0., 0.], 4), ([0., 0., 0.], 2)):
        box = InterpolationBox(point, torch.tensor(upper, device='cuda', dtype=torch.float64), 6)
        assert required_moment_quadrature_order(box) == required
    basis = RWGBasis(sphere_mesh(radius=.1, subdivisions=0, device='cuda'))
    triangles = basis.mesh.vertices[basis.mesh.faces]
    supports = torch.cat((triangles[basis.plus_faces], triangles[basis.minus_faces]), dim=1)
    tree = build_support_octree(supports.amin(1), supports.amax(1), leaf_size=8)
    nested = NestedRWGMoments(basis, tree, order=6)
    assert nested.moment_quadrature_receipt()['polynomial_exact_in_exact_arithmetic']
    value = torch.randn(30, device='cuda', dtype=torch.complex128)
    root = nested.upward(value)[0]
    direct = RWGInterpolationMoments(basis, torch.arange(30, device='cuda'), nested.boxes[0], quadrature_order=12)
    expected = torch.cat((torch.einsum('e,eac->ac', value, direct.vector), (value@direct.divergence)[:, None]), dim=-1)
    torch.testing.assert_close(root, expected, rtol=2e-10, atol=2e-14)


def test_global_system_default_records_exact_moment_degree():
    import witwin.maxwell as mw
    from witwin.maxwell.mom.mesh import SurfaceSystem
    from witwin.maxwell.mom.physics import compile_medium
    from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
    from witwin.maxwell.mom.direct_system import DirectSystemActions
    from witwin.maxwell.mom.operator import IntegrationConfig
    from witwin.maxwell.mom.rwg_interpolation_error import system_interpolation_bound
    basis = RWGBasis(sphere_mesh(radius=.01, subdivisions=0, device='cuda'))
    medium = compile_medium(mw.Material(eps_r=1.5), 3e8)
    provider = ScalarSystemBlocks(SurfaceSystem.create([(0, 'body', basis, medium)]), medium,
                                  IntegrationConfig(order=2, singular_order=3))
    operator = DirectSystemActions(provider, order=6, leaf_size=30)
    receipt = operator.receipt()['moment_quadrature']
    for record in (receipt['exterior'], *receipt['interiors']):
        assert record['orders'] == [9] and record['polynomial_exact_in_exact_arithmetic']
        assert not record['floating_point_certified']
    bound = system_interpolation_bound(operator)
    assert bound['absolute_bound'] == 0
    assert bound['exterior']['moment_quadrature']['test']['polynomial_exact_in_exact_arithmetic']
    assert not bound['implemented_operator_error_certified']
