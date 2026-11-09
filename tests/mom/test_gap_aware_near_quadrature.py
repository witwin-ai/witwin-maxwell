"""Gap- and anisotropy-aware fallback for near-disjoint panel pairs."""
import math
from types import SimpleNamespace

import pytest
import torch

from benchmark import mom_near_quadrature_accuracy as accuracy
from witwin.maxwell.mom import operator as op
from witwin.maxwell.mom.quadrature import refine_toward_source_edges, source_boundary_distance, source_feature_distances


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
K = 2*math.pi
ROWS = lambda: torch.tensor([0], device='cuda')
COLS = lambda: torch.tensor([1], device='cuda')


def _legacy_adaptive(basis, rows, cols, medium, config):
    """The product-rule loop before the fallback existed, verbatim in arithmetic."""
    order = config.singular_order
    d, magnetic = op._pair_integrals(basis, rows, cols, medium, order, order, False, 3)
    pending = torch.arange(len(rows), device=rows.device)
    while len(pending):
        order = min(order+2, config.near_max_order)
        updated_d, updated_k = op._pair_integrals(basis, rows[pending], cols[pending], medium, order, order, False, 3)
        d_scale = torch.linalg.vector_norm(updated_d, dim=(-2, -1)).clamp_min(torch.finfo(basis.mesh.dtype).tiny)
        k_scale = torch.maximum(torch.linalg.vector_norm(updated_k, dim=(-2, -1)), d_scale*abs(medium.k))
        relative = torch.maximum(torch.linalg.vector_norm(updated_d-d[pending], dim=(-2, -1))/d_scale,
                                 torch.linalg.vector_norm(updated_k-magnetic[pending], dim=(-2, -1))/k_scale)
        d[pending], magnetic[pending] = updated_d, updated_k
        pending = pending[~(relative <= config.near_tolerance)]
        assert not (len(pending) and order == config.near_max_order)
    return d, magnetic


@pytest.mark.parametrize('name, arguments', [('parallel', (1.,)), ('parallel', (.3,)), ('perpendicular', (.1,)),
                                             ('coplanar_sliver', (1., 10))])
def test_pairs_accepted_by_the_product_rule_are_bit_identical(name, arguments):
    basis = accuracy.pair_basis(*getattr(accuracy, name)(*arguments))
    medium = SimpleNamespace(k=K)
    d, magnetic, stats = op._adaptive_near_integrals(basis, ROWS(), COLS(), medium, op.IntegrationConfig())
    legacy = _legacy_adaptive(basis, ROWS(), COLS(), medium, op.IntegrationConfig())
    assert 'fallback' not in stats
    assert torch.equal(d, legacy[0]) and torch.equal(magnetic, legacy[1])


def test_observer_refinement_tiles_and_stays_anisotropic_along_parallel_edges():
    observer, source = accuracy.coplanar_sliver(.1, 30)
    leaves, owner, depth = refine_toward_source_edges(observer[None], source[None], ratio=1., max_level=28)
    area = lambda t: torch.linalg.vector_norm(torch.linalg.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]), dim=-1).sum()/2
    assert torch.allclose(area(leaves), area(observer[None]), rtol=1e-13, atol=0)
    assert bool((owner == 0).all()) and depth > 3
    # Isotropic refinement of a 30:1 sliver needs about a thousand leaves.
    assert len(leaves) < 200
    vertex, edges = source_feature_distances(leaves, source[None].expand(len(leaves), -1, -1))
    lengths = torch.linalg.vector_norm(leaves.roll(-1, 1)-leaves.roll(1, 1), dim=-1).amax(-1)
    assert bool((vertex >= lengths*(1-1e-12)).all())


def test_source_boundary_distance_cases():
    source = accuracy._tensor([(0, 0, 0), (1, 0, 0), (0, 1, 0)])[None]
    above = accuracy._tensor([(.2, .2, .1), (.3, .2, .1), (.2, .3, .1)])[None]
    assert math.isclose(float(source_boundary_distance(above, source)), math.hypot(.1, .2), rel_tol=1e-12)
    straddling = accuracy._tensor([(-.1, .2, -.1), (.1, .2, -.1), (0, .2, .1)])[None]
    assert float(source_boundary_distance(straddling, source)) == 0


# Gap 1e-3 and the full aspect sweep live in benchmark/mom_near_quadrature_accuracy.py.
@pytest.mark.parametrize('name, gap, aspect', [('parallel', .05, 1), ('parallel', .01, 1), ('perpendicular', 1e-2, 1),
                                              ('coplanar_sliver', .1, 30), ('coplanar_sliver', .1, 100),
                                              ('hinged_sliver', .1, 30)])
def test_fallback_resolves_thin_gaps_and_slivers(name, gap, aspect):
    factory = getattr(accuracy, name)
    geometry = factory(gap) if aspect == 1 else factory(gap, aspect)
    row = accuracy.measure(name, gap, aspect, geometry, K)
    assert row['reference_self_difference'] < 1e-8
    assert row['production']['rule'] == 'fallback'
    assert row['production']['error'] < 1e-4


def test_fallback_still_fails_closed_when_levels_disagree():
    basis = accuracy.pair_basis(*accuracy.parallel(.01))
    with pytest.raises(RuntimeError, match='Panel quadrature failed .* observer_anisotropic_bisection'):
        op._adaptive_near_integrals(basis, ROWS(), COLS(), SimpleNamespace(k=K),
                                    op.IntegrationConfig(near_tolerance=1e-14))


def _disable_gap_aware_rules(monkeypatch):
    from witwin.maxwell.mom import lifting

    def forbidden(*args, **kwargs):
        raise AssertionError('gap-aware rule used on a well-shaped mesh')
    monkeypatch.setattr(op, '_fallback_near_integrals', forbidden)
    monkeypatch.setattr(op, '_anisotropic_touching_integrals', forbidden)
    monkeypatch.setattr(op, 'anisotropic_touching_pairs', lambda left, right: torch.zeros(len(left), dtype=torch.bool, device=left.device))
    monkeypatch.setattr(lifting, '_fallback_near_rhs', forbidden)
    monkeypatch.setattr(lifting, '_touching_split', lambda mesh, rows, cols: torch.zeros(len(rows), dtype=torch.bool, device=rows.device))


def test_dense_assembly_and_lifted_rhs_are_bit_identical_when_no_pair_needs_the_new_rules(monkeypatch):
    import witwin.maxwell as mw
    # The RHS scatter uses index_add_, whose CUDA atomics reorder sums between
    # runs; deterministic kernels make a bitwise comparison meaningful.
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        _compare_with_rules_disabled(monkeypatch, mw)
    finally:
        torch.use_deterministic_algorithms(deterministic)


def _compare_with_rules_disabled(monkeypatch, mw):
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.1), mw.Material(eps_r=4.))], sources=[mw.PlaneWave()])
    current = mw.Simulation.mom(scene, frequency=3e8, config=mw.MoMConfig(subdivisions=1)).prepare().solver
    estimate = current.handle.metadata['quadrature_error_estimate']
    assert 'fallback' not in estimate['near_disjoint'] and 'touching_anisotropic' not in estimate
    _disable_gap_aware_rules(monkeypatch)
    previous = mw.Simulation.mom(scene, frequency=3e8, config=mw.MoMConfig(subdivisions=1)).prepare().solver
    assert torch.equal(current.handle.matrix, previous.handle.matrix)
    assert torch.equal(current.rhs, previous.rhs)


def test_closed_form_static_potentials_match_numerical_fans():
    from witwin.maxwell.mom.quadrature import distance_scaled_duffy_fan, static_potential_integrals
    generator = torch.Generator(device='cuda').manual_seed(0)
    for aspect in (1, 30):
        width = .05/aspect
        triangle = torch.tensor([[(0, 0, 0), (.05, 0, 0), (0, width, 0)]], device='cuda', dtype=torch.float64)
        weights = torch.rand(64, 3, device='cuda', dtype=torch.float64, generator=generator)
        points = (weights/weights.sum(-1, keepdim=True))@triangle[0]
        cases = {'inside': points, 'above': points+torch.tensor([0, 0, 1e-2*width], device='cuda', dtype=torch.float64),
                 'outside': points+torch.tensor([0, -.4*width, 0], device='cuda', dtype=torch.float64)}
        for name, x in cases.items():
            sources = triangle.expand(len(x), -1, -1)
            i0, i1, gradient = static_potential_integrals(x, sources)
            nodes, w = distance_scaled_duffy_fan(x, sources, 40, edge_grading=True, coplanar_tolerance=1e-10)
            delta = x[:, None]-nodes
            radius = torch.linalg.vector_norm(delta, dim=-1)
            projected = x*torch.tensor([1., 1., 0.], device='cuda', dtype=torch.float64)
            assert torch.allclose(i0, (w/radius).sum(-1), rtol=1e-9, atol=0)
            assert torch.allclose(i1, (w[..., None]*(nodes-projected[:, None])/radius[..., None]).sum(1), rtol=1e-9, atol=1e-18)
            if name == 'above':
                numeric = (w[..., None]*(-delta/radius[..., None]**3)).sum(1)
                assert torch.allclose(gradient, numeric, rtol=1e-6, atol=0)


@pytest.mark.parametrize('pair', ['self', 'edge_long', 'edge_diagonal', 'vertex', 'edge_crease', 'edge_corner'])
def test_anisotropic_touching_rule_is_consistent_and_matches_duffy_on_well_shaped_pairs(pair):
    from benchmark import mom_touching_quadrature_accuracy as touching
    medium = SimpleNamespace(k=K*math.sqrt(6))
    r, c, category = touching.PAIRS[pair]
    rows, cols = torch.tensor([r], device='cuda'), torch.tensor([c], device='cuda')
    config = op.IntegrationConfig()
    well = touching.strip(1, True)
    d, magnetic, stats = op._anisotropic_touching_integrals(well, rows, cols, medium, config, category)
    duffy = op._pair_integrals(well, rows, cols, medium, 24, 24, True, category)
    if category == 0:
        duffy = (duffy[0], torch.zeros_like(duffy[1]))
    # An unrelated rule on a well-shaped pair: order-24 joint Duffy integration.
    assert float(op._near_relative_difference(d, magnetic, *duffy, medium, torch.float64)) < 1e-6
    sliver = touching.strip(30, True)
    d, magnetic, stats = op._anisotropic_touching_integrals(sliver, rows, cols, medium, config, category)
    deep = op._subtracted_touching_integrals(sliver, rows, cols, medium, 10, 12, 4., category=category, max_level=28)
    reference = (deep[0], torch.zeros_like(deep[1]) if category == 0 else deep[1])
    assert stats['rule'] == op.TOUCHING_ANISOTROPIC_RULE
    assert float(op._near_relative_difference(d, magnetic, *reference, medium, torch.float64)) < 1e-6


def test_touching_rule_selection_keeps_well_shaped_pairs_on_duffy():
    from witwin.maxwell.mom.quadrature import anisotropic_touching_pairs
    from witwin.maxwell.mom.primitive_mesh import box_arrays, rounded_box_arrays
    vertices, faces = box_arrays((.2, .2, .2), .05, device='cuda')
    triangles = vertices[faces]
    assert not bool(anisotropic_touching_pairs(triangles, triangles).any())
    vertices, faces = rounded_box_arrays((.2, .2, .2), .002, .05, device='cuda')
    triangles = vertices[faces]
    assert bool(anisotropic_touching_pairs(triangles, triangles).any())
