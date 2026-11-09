"""Integral majorants and ideal weak-form interpolation error bounds."""
import pytest
import math
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh
from witwin.maxwell.mom.operator import PreparedRWGGeometry, IntegrationConfig
from witwin.maxwell.mom.quadrature import triangle_rule
from witwin.maxwell.mom.rwg_interpolation_error import rwg_integral_logs, regional_interpolation_bound
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.nearfield import scalar_regional_block
from witwin.maxwell.mom.cluster import build_support_octree
from witwin.maxwell.mom.nested_moments import NestedRWGMoments
from witwin.maxwell.mom.direct_regional import DirectRegionalOperator

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_rwg_absolute_integrals_majorize_independent_quadrature():
    basis = RWGBasis(sphere_mesh(radius=.1, subdivisions=0, device='cuda'))
    geo = PreparedRWGGeometry(basis)
    bounds = rwg_integral_logs(basis, geo).exp()
    points, weights = triangle_rule(geo.triangles, 12)
    f = geo.slopes[:, None, :, None]*(points[:, :, None]-geo.triangles[:, None])
    local = (torch.linalg.vector_norm(f, dim=-1)*weights[..., None]).sum(1)
    numerical = torch.zeros(basis.n_edges, device='cuda', dtype=torch.float64)
    numerical.index_add_(0, basis.face_edges.flatten(), local.flatten())
    assert bool((numerical <= bounds[:, 0]).all())
    torch.testing.assert_close(bounds[:, 1], 2*basis.edge_lengths, rtol=2e-14, atol=0)
    basis.mesh.vertices.add_(.01)
    with pytest.raises(ValueError, match='geometry changed'):
        rwg_integral_logs(basis, geo)


def test_regional_frobenius_bound_covers_far_weak_difference_and_media_overlap():
    spaces = []
    for center in ((-.5, 0., 0.), (.5, 0., 0.)):
        basis = RWGBasis(sphere_mesh(radius=.01, center=center, subdivisions=0, device='cuda'))
        triangles = basis.mesh.vertices[basis.mesh.faces]
        supports = torch.cat((triangles[basis.plus_faces], triangles[basis.minus_faces]), dim=1)
        tree = build_support_octree(supports.amin(1), supports.amax(1), leaf_size=8)
        spaces.append(NestedRWGMoments(basis, tree, order=3, quadrature_order=6))
    medium = compile_medium(mw.Material(eps_r=1.2, mu_r=1.3, sigma_e=.001), 3e8)
    operator = DirectRegionalOperator(*spaces, (medium,), integration=IntegrationConfig(order=6, singular_order=6))
    receipt = regional_interpolation_bound(operator, abs(medium.eta))
    rows = torch.arange(30, device='cuda')
    reference, _ = scalar_regional_block(spaces[0].basis, spaces[1].basis, rows, rows, (medium,), operator.integration)
    actual = operator.apply(torch.eye(60, device='cuda', dtype=torch.complex128)).mT
    eta = abs(medium.eta)
    rs = torch.tensor([eta]*30+[1.]*30, device='cuda', dtype=torch.float64)
    cs = torch.tensor([1.]*30+[eta]*30, device='cuda', dtype=torch.float64)
    error = float(torch.linalg.vector_norm(rs[:, None]*(actual-reference)*cs))
    print('weak error', error, 'absolute bound', receipt['absolute_bound'])
    assert error <= receipt['absolute_bound'] and error > 0
    assert not receipt['implemented_operator_error_certified']


def test_regional_overlap_uses_triangle_sum_and_system_combines_disjoint_interiors(monkeypatch):
    from types import SimpleNamespace
    import witwin.maxwell.mom.rwg_interpolation_error as module
    from witwin.maxwell.mom.interpolation_error import scalar_box_interpolation_bound
    basis = RWGBasis(sphere_mesh(radius=.01, subdivisions=0, device='cuda'))
    triangles = basis.mesh.vertices[basis.mesh.faces]
    supports = torch.cat((triangles[basis.plus_faces], triangles[basis.minus_faces]), dim=1)
    tree = build_support_octree(supports.amin(1), supports.amax(1), leaf_size=8)
    projection = NestedRWGMoments(basis, tree, order=3, quadrature_order=6)
    medium = compile_medium(mw.Material(), 3e8)
    lower = torch.tensor([0., 0., 0.], device='cuda', dtype=torch.float64)
    upper = lower+.01
    kernel_bound = scalar_box_interpolation_bound(lower, upper, lower+1, upper+1, medium.k, 3, 3)
    kernel = SimpleNamespace(medium=medium, truncation_bound=lambda: kernel_bound)
    # Abstract regional contributions on the same coefficient block must add,
    # rather than incorrectly taking a root sum of squares across media.
    operator = SimpleNamespace(test=projection, source=projection, far=((0, 0, kernel),), assert_current=projection.assert_current)
    single = regional_interpolation_bound(operator, abs(medium.eta))
    operator.far = operator.far*2
    doubled = regional_interpolation_bound(operator, abs(medium.eta))
    assert doubled['absolute_bound'] == pytest.approx(2*single['absolute_bound'], rel=1e-13)
    ext, one, two = object(), object(), object()
    log_values = {ext: 0., one: math.log(3.), two: math.log(4.)}
    def regional(region, eta):
        return {'log_absolute_bound': log_values[region]}
    monkeypatch.setattr(module, 'regional_interpolation_bound', regional)
    system = SimpleNamespace(exterior=ext, interiors=(one, two), device=basis.mesh.device,
                             provider=SimpleNamespace(exterior=medium), assert_current=lambda: None)
    result = module.system_interpolation_bound(system)
    assert result['absolute_bound'] == pytest.approx(6.)
