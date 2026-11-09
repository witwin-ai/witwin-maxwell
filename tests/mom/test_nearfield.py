"""Selected blocks against independently assembled dense weak matrices."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig, PreparedRWGGeometry, assemble_weak, assemble_cross
from witwin.maxwell.mom.nearfield import scalar_regional_block

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_selected_self_block_matches_all_five_dense_quadrature_classes(monkeypatch):
    basis = RWGBasis(sphere_mesh(radius=.1, subdivisions=1, device='cuda'))
    exterior = compile_medium(mw.Material(), 3e8)
    interior = compile_medium(mw.Material(eps_r=2., mu_r=1.3), 3e8)
    config = IntegrationConfig(order=2, singular_order=3, near_tolerance=1e-3,
                               pair_batch=32, far_pair_batch=128)
    dense, _ = assemble_weak(basis, exterior, interior, config)
    rows = torch.tensor([50, 0, 9, 51, 90, 1, 119], device='cuda')
    cols = torch.tensor([119, 50, 51, 0, 90, 9, 1], device='cuda')
    import witwin.maxwell.mom.operator as operator
    def forbid(*args, **kwargs):
        raise AssertionError('Selected blocks must not allocate global classifications')
    geometry = PreparedRWGGeometry(basis)
    monkeypatch.setattr(operator, 'classify_all_pairs', forbid)
    monkeypatch.setattr(operator, 'local_geometry', forbid)
    actual, receipt = scalar_regional_block(basis, basis, rows, cols, (exterior, interior), config, geometry=(geometry, geometry))
    ri, ci = torch.cat((rows, rows+basis.n_edges)), torch.cat((cols, cols+basis.n_edges))
    torch.testing.assert_close(actual, dense[ri[:, None], ci], rtol=2e-12, atol=1e-14)
    assert all(count > 0 for count in receipt['triangle_pair_counts'])
    assert receipt['maximum_classification_pairs'] <= 128
    assert receipt['matrix_bytes'] == actual.numel()*actual.element_size()
    with pytest.raises(ValueError, match='unique'):
        scalar_regional_block(basis, basis, rows[:1].expand(2), cols, (exterior,), config)


def test_selected_cross_block_keeps_only_exterior_and_preserves_rectangular_order():
    a = RWGBasis(sphere_mesh(radius=.1, subdivisions=0, device='cuda'))
    b = RWGBasis(sphere_mesh(radius=.07, center=(.4, 0., 0.), subdivisions=0, device='cuda'))
    exterior = compile_medium(mw.Material(eps_r=1.2, mu_r=1.1), 3e8)
    config = IntegrationConfig(order=3, singular_order=3)
    dense, _ = assemble_cross(a, b, exterior, config)
    rows = torch.tensor([13, 0, 29], device='cuda')
    cols = torch.tensor([29, 12, 2, 15, 0], device='cuda')
    actual, receipt = scalar_regional_block(a, b, rows, cols, (exterior,), config)
    ri, ci = torch.cat((rows, rows+a.n_edges)), torch.cat((cols, cols+b.n_edges))
    torch.testing.assert_close(actual, dense[ri[:, None], ci], rtol=2e-12, atol=1e-14)
    assert receipt['triangle_pair_counts'][:3] == [0, 0, 0]
    with pytest.raises(ValueError, match='only the common exterior'):
        scalar_regional_block(a, b, rows, cols, (exterior, exterior), config)


def test_global_provider_matches_mixed_body_selection_and_owns_no_matrix(monkeypatch):
    from witwin.maxwell.mom.mesh import SurfaceSystem
    from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
    from witwin.maxwell.mom.operator import assemble_system
    from witwin.maxwell.mom.cluster import build_support_octree
    a = RWGBasis(sphere_mesh(radius=.05, center=(-.2, 0., 0.), subdivisions=0, device='cuda'))
    b = RWGBasis(sphere_mesh(radius=.07, center=(.2, 0., 0.), subdivisions=0, device='cuda'))
    exterior, ma, mb = [compile_medium(mw.Material(eps_r=e), 3e8) for e in (1., 1.7, 2.3)]
    system = SurfaceSystem.create([(0, 'a', a, ma), (1, 'b', b, mb)])
    config = IntegrationConfig(order=2, singular_order=3)
    provider = ScalarSystemBlocks(system, exterior, config)
    rows = torch.tensor([31, 2, 0, 59], device='cuda')
    cols = torch.tensor([1, 58, 42, 4, 18], device='cuda')
    actual = provider(rows, cols)
    dense, _ = assemble_system(system, exterior, config)
    rr, cc = torch.cat((rows, rows+60)), torch.cat((cols, cols+60))
    torch.testing.assert_close(actual, dense[rr[:, None], cc], rtol=2e-12, atol=1e-14)
    import witwin.maxwell.mom.operator as operator
    def forbid(*args, **kwargs):
        raise AssertionError('Prepared block requests must reuse whole-basis geometry')
    monkeypatch.setattr(operator, 'local_geometry', forbid)
    torch.testing.assert_close(provider(rows, cols), actual, rtol=0, atol=0)
    assert not hasattr(provider, 'matrix')
    assert provider.metadata['maximum_block_bytes'] == actual.numel()*actual.element_size()
    tree = build_support_octree(*provider.support_boxes(), leaf_size=4)
    assert tree.size == system.total_edges
    original = provider.exterior
    provider.exterior = ma
    with pytest.raises(ValueError, match='binding changed'):
        provider(rows, cols)
    provider.exterior = original
    provider.geometry[0].slopes.add_(1)
    with pytest.raises(ValueError, match='Prepared RWG geometry changed'):
        provider(rows, cols)
    a.mesh.vertices.add_(.01)
    with pytest.raises(ValueError, match='geometry changed'):
        provider(rows, cols)
