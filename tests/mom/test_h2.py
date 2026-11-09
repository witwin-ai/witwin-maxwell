"""Nested coupling actions across multiple tree levels and actual RWG blocks."""
import pytest
import torch
from witwin.maxwell.mom.cluster import build_support_octree
from witwin.maxwell.mom.lowrank import LowRankBlock
from witwin.maxwell.mom.h2 import NestedH2Actions, H2Config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_multilevel_h2_couplings_forward_adjoint_gradient_and_ownership():
    g = torch.Generator(device='cuda').manual_seed(314)
    points = torch.rand((64, 3), device='cuda', dtype=torch.float64, generator=g)
    tree = build_support_octree(points-.001, points+.001, leaf_size=4)
    root_children = tree.children[0][tree.children[0] >= 0].tolist()
    a = next(node for node in root_children if bool((tree.children[node] >= 0).any()))
    b, c = [node for node in root_children if node != a][:2]
    a1 = int(tree.children[a][tree.children[a] >= 0][0])
    def indices(node):
        points = tree.permutation[tree.starts[node]:tree.stops[node]]
        return torch.cat((points, points+64))
    records = []
    dense = torch.zeros((128, 128), device='cuda', dtype=torch.complex128)
    for row, col in ((a, b), (b, a), (a1, c), (c, a1)):
        ri, ci = indices(row), indices(col)
        left = torch.randn((len(ri), 2), generator=g, device='cuda', dtype=dense.dtype)
        right = torch.randn((2, len(ci)), generator=g, device='cuda', dtype=dense.dtype)
        records.append((row, col, LowRankBlock(left, right, {})))
        dense[ri[:, None], ci] += left@right
    rs = torch.linspace(.2, 3., 128, device='cuda', dtype=torch.float64)
    cs = torch.linspace(4., .3, 128, device='cuda', dtype=torch.float64)
    h2 = NestedH2Actions(tree, records, row_scale=rs, column_scale=cs,
                         config=H2Config(max_rank=12, tolerance=1e-10))
    x = torch.randn((2, 3, 128), device='cuda', dtype=dense.dtype, requires_grad=True)
    y = torch.randn_like(x)
    actual = h2.apply(x)
    torch.testing.assert_close(actual, x@dense.mT, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(h2.adjoint(y), y@dense.conj(), rtol=1e-10, atol=1e-10)
    defect = ((y.conj()*actual).sum()-(h2.adjoint(y).conj()*x).sum()).abs()
    assert float((defect/(x.norm()*y.norm())).detach()) < 1e-11
    gradient, = torch.autograd.grad(actual.abs().square().sum()/2, x)
    torch.testing.assert_close(gradient, h2.adjoint(actual), rtol=1e-11, atol=1e-10)
    receipt = h2.receipt()
    assert not receipt['source_h_far_factors_retained']
    assert max(receipt['test_basis']['ranks']) <= 12
    assert h2.test_basis.ranks[a] and h2.test_basis.ranks[a1]
    # Construction owns its far representation and numerical scales.
    records[0][2].left.zero_()
    rs.mul_(2)
    torch.testing.assert_close(h2.apply(x), actual, rtol=0, atol=0)
    h2.couplings[0][2].add_(1)
    with pytest.raises(ValueError, match='changed'):
        h2.apply(x)


def test_actual_rwg_h_to_h2_conversion_keeps_exact_near_and_balanced_error():
    import witwin.maxwell as mw
    from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
    from witwin.maxwell.mom.physics import compile_medium
    from witwin.maxwell.mom.operator import IntegrationConfig, assemble_system
    from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
    from witwin.maxwell.mom.hmatrix import HierarchicalOperator, HMatrixConfig
    a = RWGBasis(sphere_mesh(radius=.01, center=(-.5, 0., 0.), subdivisions=0, device='cuda'))
    b = RWGBasis(sphere_mesh(radius=.01, center=(.5, 0., 0.), subdivisions=0, device='cuda'))
    exterior = compile_medium(mw.Material(), 3e8)
    interior = compile_medium(mw.Material(eps_r=2.), 3e8)
    system = SurfaceSystem.create([(0, 'a', a, interior), (1, 'b', b, interior)])
    integration = IntegrationConfig(order=3, singular_order=3)
    h = HierarchicalOperator(ScalarSystemBlocks(system, exterior, integration), HMatrixConfig(pivoting='adaptive'))
    h2 = NestedH2Actions.from_hmatrix(h)
    assert h2.near is h.packed_near
    identity = torch.eye(120, device='cuda', dtype=h.dtype)
    reconstructed = h2.apply(identity).mT
    reference_h = h.apply(identity).mT
    dense, _ = assemble_system(system, exterior, integration)
    rs, cs = h2.row_scale, h2.column_scale
    def norm(value):
        return torch.linalg.vector_norm(value*rs[:, None]*cs[None, :])
    conversion_error = float(norm(reconstructed-reference_h)/norm(reference_h))
    original_error = float(norm(reconstructed-dense)/norm(dense))
    assert conversion_error <= 1e-10
    assert original_error <= 1e-8
    for a, b, coupling in h2.couplings:
        assert coupling.shape == (h2.test_basis.ranks[a], h2.trial_basis.ranks[b])
    x = torch.randn((3, 120), device='cuda', dtype=h.dtype, requires_grad=True)
    y = torch.randn_like(x)
    torch.testing.assert_close(h2.adjoint(y), y@reconstructed.conj(), rtol=1e-11, atol=1e-12)
    gradient, = torch.autograd.grad(h2.apply(x).abs().square().sum()/2, x)
    torch.testing.assert_close(gradient, h2.adjoint(h2.apply(x)), rtol=1e-11, atol=1e-12)
    receipt = h2.receipt()
    print({'conversion_error': conversion_error, 'original_error': original_error,
           'h2': receipt, 'h_owned_retained_bytes': h.storage_breakdown()['owned_retained_bytes']})
    # Node maps are part of the H binding used during conversion.
    h.block_nodes = tuple(reversed(h.block_nodes))
    with pytest.raises(ValueError, match='changed'):
        NestedH2Actions.from_hmatrix(h)
