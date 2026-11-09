"""Periodic Krylov acceptance checks the original periodic, not free-space, rows."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceMesh, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig, assemble_system
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks, scalar_regional_block
from witwin.maxwell.mom.periodic import LatticeSpec
from witwin.maxwell.mom.periodic_nearfield import PeriodicImageSystemBlocks
from witwin.maxwell.mom.fmm_periodic_operator import PeriodicMultipoleOperator
from witwin.maxwell.mom.linalg import StoppingRule

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize("table_storage", ["dense", "compact"])
def test_periodic_krylov_residual_matches_independent_image_matrix(table_storage):
    exterior = compile_medium(mw.Material(eps_r=1.5, sigma_e=.02), 3e8)
    interior = compile_medium(mw.Material(eps_r=2., mu_r=1.2), 3e8)
    basis = RWGBasis(sphere_mesh(radius=.04, subdivisions=0, device='cuda'))
    system = SurfaceSystem.create([(0, 'body', basis, interior)])
    config = IntegrationConfig(order=4, singular_order=3)
    lattice = LatticeSpec(torch.tensor([[1., 0., 0.]], device='cuda', dtype=torch.float64),
                          torch.tensor([.37, 0., 0.], device='cuda', dtype=torch.float64))
    provider = PeriodicImageSystemBlocks(system, exterior, config, lattice, shells=1)
    op = PeriodicMultipoleOperator(provider, order=6, leaf_size=30, separation_ratio=.1, table_storage=table_storage)
    primary, _ = assemble_system(system, exterior, config)
    dense = primary.clone()
    edges = torch.arange(basis.n_edges, device='cuda')
    for image in (-1, 1):
        index = torch.tensor([image], device='cuda')
        translated = RWGBasis(SurfaceMesh(basis.mesh.vertices+lattice.translations(index), basis.mesh.faces, validate=False))
        block, _ = scalar_regional_block(basis, translated, edges, edges, (exterior,), config)
        dense += lattice.phase(index)*block
    # Algebraic RHS with known currents; this is not a physical incidence claim.
    torch.manual_seed(2917)
    target = torch.randn((2, 2*basis.n_edges), device='cuda', dtype=dense.dtype)
    rhs = target@dense.mT
    torch.testing.assert_close(provider.sampled_action(edges, target, source_batch=17), rhs, rtol=3e-12, atol=1e-14)
    solved, log = op.solve(rhs, StoppingRule(method='block_gmres', preconditioner='rwg_diagonal',
        rtol=1e-12, atol=0., maxiter=120, restart=60), original_edges=edges, source_batch=32)
    assert log['converged'] and log['error_budget']['checked'], log
    true_residual = torch.linalg.vector_norm(solved@dense.mT-rhs)/torch.linalg.vector_norm(rhs)
    wrong_residual = torch.linalg.vector_norm(solved@primary.mT-rhs)/torch.linalg.vector_norm(rhs)
    print('periodic/free-space residuals', float(true_residual), float(wrong_residual))
    assert float(true_residual) < 1e-8 and float(wrong_residual) > 1e-8
    assert not op.metadata['infinite_periodic_certified']
    with pytest.raises(TypeError, match='original periodic'):
        PeriodicMultipoleOperator(ScalarSystemBlocks(system, exterior, config))
    _, failed = op.solve(rhs, StoppingRule(method='gmres', preconditioner='rwg_diagonal',
        rtol=1e-12, atol=0., maxiter=1, restart=1))
    assert not failed['converged'] and not failed['fallback_used']
    provider.shells = 2
    with pytest.raises(ValueError, match='changed'):
        op.apply(target)
