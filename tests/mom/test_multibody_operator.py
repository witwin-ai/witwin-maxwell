"""Block formulas and cross-body topology independent of local vertex IDs."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig, assemble_system, assemble_weak, classify_cross_pairs, _pair_integrals

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_cross_pairs_ignore_local_ids_and_allow_different_basis_sizes():
    a = RWGBasis(sphere_mesh(radius=.1, subdivisions=0, device='cuda'))
    b = RWGBasis(sphere_mesh(radius=.07, center=(.4, 0., 0.), subdivisions=1, device='cuda'))
    rows = torch.tensor([0, 1], device='cuda')
    cols = torch.tensor([0, 50], device='cuda')
    classes, distance = classify_cross_pairs(a, b, rows, cols)
    assert bool(((classes == 3) | (classes == 4)).all()) and bool((distance > 0).all())
    ext = compile_medium(mw.Material(), 3e8)
    d, k = _pair_integrals(a, rows, cols, ext, 4, 4, False, 4, source_basis=b)
    assert d.shape == k.shape == (2, 3, 3) and bool(torch.isfinite(d).all())
    assert torch.linalg.vector_norm(k[0]) > 0  # Equal local IDs must not zero a cross block.
    swapped_d, swapped_k = _pair_integrals(b, cols, rows, ext, 4, 4, False, 4, source_basis=a)
    torch.testing.assert_close(d, swapped_d.mT, rtol=1e-12, atol=1e-14)


def test_only_changed_interior_diagonal_changes_and_single_body_degenerates():
    a = RWGBasis(sphere_mesh(radius=.1, subdivisions=0, device='cuda'))
    b = RWGBasis(sphere_mesh(radius=.07, center=(.5, 0., 0.), subdivisions=0, device='cuda'))
    ext, m1, m2 = [compile_medium(mw.Material(eps_r=e), 3e8) for e in (1., 2., 3.)]
    config = IntegrationConfig(order=3, singular_order=4)
    original = SurfaceSystem.create([(0, 'a', a, m1), (1, 'b', b, m1)])
    changed = SurfaceSystem.create([(0, 'a', a, m1), (1, 'b', b, m2)])
    first, _ = assemble_system(original, ext, config)
    second, _ = assemble_system(changed, ext, config)
    ai = torch.cat((torch.arange(30, device='cuda'), torch.arange(60, 90, device='cuda')))
    bi = torch.cat((torch.arange(30, 60, device='cuda'), torch.arange(90, 120, device='cuda')))
    for left, right in ((ai, ai), (ai, bi), (bi, ai)):
        torch.testing.assert_close(first[left[:, None], right], second[left[:, None], right], rtol=2e-13, atol=1e-14)
    assert torch.linalg.vector_norm(first[bi[:, None], bi]-second[bi[:, None], bi]) > 1e-6
    single = SurfaceSystem.create([(0, 'a', a, m1)])
    combined, _ = assemble_system(single, ext, config)
    plain, _ = assemble_weak(a, ext, m1, config)
    torch.testing.assert_close(combined, plain, rtol=2e-13, atol=1e-14)


def test_permutation_covariance_of_matrix_lifted_rhs_and_solution():
    from witwin.maxwell.mom.runtime import SystemOperator, MoMConfig
    from witwin.maxwell.mom.linalg import solve_reference, StoppingRule
    a = RWGBasis(sphere_mesh(radius=.07, center=(-.2, 0., 0.), subdivisions=0, device='cuda'))
    b = RWGBasis(sphere_mesh(radius=.06, center=(.2, 0., 0.), subdivisions=0, device='cuda'))
    ext, ma, mb = [compile_medium(mw.Material(eps_r=e), 3e8) for e in (1., 1.01, 1.02)]
    config = MoMConfig(subdivisions=0, integration=IntegrationConfig(order=3, singular_order=4), riesz_storage='compact_rwg_v1')
    left = SystemOperator(SurfaceSystem.create([(0, 'a', a, ma), (1, 'b', b, mb)]), ext, [mw.PlaneWave()], config)
    right = SystemOperator(SurfaceSystem.create([(0, 'b', b, mb), (1, 'a', a, ma)]), ext, [mw.PlaneWave()], config)
    permutation = torch.cat([torch.arange(start, start+30, device='cuda') for start in (30, 0, 90, 60)])
    torch.testing.assert_close(right.matrix, left.matrix[permutation[:, None], permutation], rtol=1e-12, atol=1e-14)
    bl, br = left.build_rhs(), right.build_rhs()
    torch.testing.assert_close(br, bl[permutation], rtol=1e-12, atol=1e-14)
    xl, _ = solve_reference(left, bl, StoppingRule(method='direct'))
    xr, _ = solve_reference(right, br, StoppingRule(method='direct'))
    torch.testing.assert_close(xr, xl[permutation], rtol=1e-10, atol=1e-13)
