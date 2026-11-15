"""Streamed regular correction refinement agrees with dense correction controls."""
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceMesh, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.periodic import LatticeSpec, EwaldHelmholtzKernel
from witwin.maxwell.mom.periodic_operator import _assemble_regular
from witwin.maxwell.mom.ewald_nearfield import EwaldSystemBlocks
from witwin.maxwell.mom.ewald_refinement import measure_ewald_refinement

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def _tetrahedron(x):
    vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], device='cuda',
                            dtype=torch.float64)*.02
    vertices = vertices-vertices.mean(0)+vertices.new_tensor([x, 0., 0.])
    return RWGBasis(SurfaceMesh(vertices, torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], device='cuda')))


def problem(body='sphere'):
    exterior = compile_medium(mw.Material(), 9e7)
    interior = compile_medium(mw.Material(eps_r=2., mu_r=1.2), 9e7)
    basis = ((lambda x: RWGBasis(sphere_mesh(radius=.01, center=(x, 0., 0.), subdivisions=0, device='cuda')))
             if body == 'sphere' else _tetrahedron)
    system = SurfaceSystem.create([(i, str(i), basis(x), interior) for i, x in enumerate((-.05, .05))])
    lattice = LatticeSpec(system.bodies[0].mesh.vertices.new_tensor([[1., 0., 0.]]),
                         system.bodies[0].mesh.vertices.new_tensor([.17, 0., 0.]))
    kernel = EwaldHelmholtzKernel(lattice, .7, real_shells=1, reciprocal_shells=2, quadrature_order=16)
    config = IntegrationConfig(order=2, singular_order=3)
    provider = EwaldSystemBlocks(system, exterior, config, kernel)
    return provider


def test_streamed_ewald_refinement_and_regular_selected_blocks():
    provider = problem()
    system, exterior, config, kernel = provider.system, provider.exterior, provider.config, provider.kernel
    dense, _ = _assemble_regular(system, exterior, config, kernel)
    n = system.total_edges
    rows = torch.arange(n, device='cuda').flip(0)
    cols = torch.arange(0, n, 3, device='cuda')
    rr, cc = torch.cat((rows, rows+n)), torch.cat((cols, cols+n))
    torch.testing.assert_close(provider.regular_block(rows, cols), dense[rr[:, None], cc], rtol=3e-11, atol=1e-15)
    receipt = measure_ewald_refinement(provider, edge_batch=37, relative_tolerance=1e-7, absolute_tolerance=1e-12)
    assert not receipt['converged'] and not receipt['certified_error_bound']
    check = next(row for row in receipt['checks'] if row['axis'] == 'real_shells')
    assert check['initial'] == 1 and check['refined'] == 2 and not check['passed']
    refined, _ = _assemble_regular(system, exterior, config, replace(kernel, real_shells=2))
    left = dense.real.new_ones(2*n)
    right = left.clone()
    left[:n], right[n:] = abs(exterior.eta), abs(exterior.eta)
    error = (left[:, None]*(refined-dense)*right).norm().item()
    norm = (left[:, None]*refined*right).norm().item()
    assert check['metrics']['global']['absolute_difference'] == pytest.approx(error, rel=1e-10)
    assert check['metrics']['global']['reference_norm'] == pytest.approx(norm, rel=1e-10)
    assert check['metrics']['cross_body']['entries'] == 8*(n//2)**2
    print(receipt)


@pytest.mark.mom_milestone
def test_resolved_ewald_correction_refinement():
    _resolved_refinement('sphere', 37)


def test_resolved_ewald_correction_refinement_regression():
    # Regression twin: the same lattice, media, Ewald kernel and testing order, with two 6-edge
    # tetrahedra of 2 cm instead of two 30-edge icosahedral spheres of 1 cm radius (n = 12 instead
    # of 60). edge_batch=8 keeps a full and a partial row/column batch (60 = 37+23 in the original).
    # Every refinement axis is evaluated; same converged and cross-body assertions.
    _resolved_refinement('tetrahedron', 8)


def _resolved_refinement(body, edge_batch):
    original = problem(body)
    provider = EwaldSystemBlocks(original.system, original.exterior,
        replace(original.config, order=4),
        replace(original.kernel, eta=2., real_shells=4, quadrature_order=32))
    receipt = measure_ewald_refinement(provider, edge_batch=edge_batch)
    print(receipt)
    assert receipt['converged']
    assert all(check['metrics']['cross_body']['passed'] for check in receipt['checks'])
