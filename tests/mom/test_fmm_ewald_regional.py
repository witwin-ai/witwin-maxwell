"""Shared Ewald regional hierarchy matches original selected weak blocks."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.periodic import LatticeSpec, EwaldHelmholtzKernel
from witwin.maxwell.mom.periodic_operator import assemble_periodic_system
from witwin.maxwell.mom.ewald_nearfield import EwaldSystemBlocks
from witwin.maxwell.mom.cluster import build_support_octree
from witwin.maxwell.mom.fmm_moments import NestedRWGMultipoles
from witwin.maxwell.mom.fmm_periodic_regional import PeriodicMultipoleRegionalOperator

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_ewald_selected_reference_and_shared_global_exterior():
    exterior = compile_medium(mw.Material(), 9e7)
    # Matched interiors let us subtract their independent free-space blocks
    # from the full dense reference before comparing the exterior action.
    interior = compile_medium(mw.Material(eps_r=2., mu_r=1.2), 9e7)
    system = SurfaceSystem.create([(i, str(i), RWGBasis(sphere_mesh(radius=.01,
        center=(x, 0., 0.), subdivisions=0, device='cuda')), interior) for i, x in enumerate((-.05, .05))])
    lattice = LatticeSpec(torch.tensor([[1., .2, .1]], device='cuda', dtype=torch.float64),
                          torch.tensor([.17, .034, .017], device='cuda', dtype=torch.float64))
    kernel = EwaldHelmholtzKernel(lattice, 2., real_shells=2, reciprocal_shells=2, quadrature_order=32, image_batch=13)
    config = IntegrationConfig(order=4, singular_order=3, pair_batch=64)
    provider = EwaldSystemBlocks(system, exterior, config, kernel)
    dense, _ = assemble_periodic_system(system, exterior, config, kernel=kernel)
    n = system.total_edges
    edges = torch.arange(n, device='cuda')
    selected = provider(edges.flip(0), edges[::3])
    rr, cc = torch.cat((edges.flip(0), edges.flip(0)+n)), torch.cat((edges[::3], edges[::3]+n))
    torch.testing.assert_close(selected, dense[rr[:, None], cc], rtol=3e-11, atol=1e-15)
    from witwin.maxwell.mom.nearfield import scalar_regional_block
    for body in system.bodies:
        e = torch.arange(body.basis.n_edges, device='cuda')
        block, _ = scalar_regional_block(body.basis, body.basis, e, e, (body.interior,), config)
        ids = torch.cat((e+body.edge_offset, e+body.edge_offset+n))
        dense[ids[:, None], ids] -= block
    lower, upper = provider.support_boxes()
    projection = NestedRWGMultipoles(provider, build_support_octree(lower, upper, leaf_size=30), order=8)
    op = PeriodicMultipoleRegionalOperator(projection, projection, exterior, lattice,
        integration=config, separation_ratio=.25, table_storage='compact', ewald_kernel=kernel)
    actual = op.apply(torch.eye(2*n, device='cuda', dtype=torch.complex128)).mT
    left = torch.ones(2*n, device='cuda', dtype=torch.float64)
    right = left.clone()
    left[:n], right[n:] = abs(exterior.eta), abs(exterior.eta)
    difference = left[:, None]*(actual-dense)*right
    balanced = left[:, None]*dense*right
    body_ids = torch.bucketize(edges, provider.edge_stops, right=True).repeat(2)
    cross = body_ids[:, None] != body_ids
    for mask in (torch.ones_like(cross), cross):
        relative = float(torch.linalg.vector_norm(difference[mask])/torch.linalg.vector_norm(balanced[mask]))
        print('ewald exterior relative', relative)
        assert relative < 1e-8
    x = torch.randn((2, 2*n), device='cuda', dtype=actual.dtype, requires_grad=True)
    y = torch.randn_like(x)
    derivative, = torch.autograd.grad(op.apply(x), x, y)
    torch.testing.assert_close(derivative, op.adjoint(y), rtol=3e-11, atol=1e-13)
    torch.testing.assert_close(op.adjoint(y), y@actual.conj(), rtol=3e-11, atol=1e-13)
    assert op._remote_active
    op.remote.near_indices.add_(1)
    with pytest.raises(ValueError, match='changed'):
        op.apply(x)


def test_complete_ewald_system_krylov_and_capacity_bind_original_reference():
    from witwin.maxwell.mom.fmm_periodic_operator import PeriodicMultipoleOperator
    from witwin.maxwell.mom.fmm_periodic_capacity import periodic_action_tensors, storage_bytes
    from witwin.maxwell.mom.linalg import StoppingRule
    from witwin.maxwell.mom.interpolation_error import KernelErrorBudget
    exterior = compile_medium(mw.Material(), 9e7)
    interior = compile_medium(mw.Material(eps_r=2., mu_r=1.2), 9e7)
    basis = RWGBasis(sphere_mesh(radius=.01, subdivisions=0, device='cuda'))
    system = SurfaceSystem.create([(0, 'body', basis, interior)])
    lattice = LatticeSpec(torch.tensor([[1., 0., 0.]], device='cuda', dtype=torch.float64),
                          torch.tensor([.17, 0., 0.], device='cuda', dtype=torch.float64))
    kernel = EwaldHelmholtzKernel(lattice, 2., real_shells=2, reciprocal_shells=2, quadrature_order=32)
    config = IntegrationConfig(order=4, singular_order=3)
    provider = EwaldSystemBlocks(system, exterior, config, kernel)
    op = PeriodicMultipoleOperator(provider, order=6, leaf_size=30, separation_ratio=.25, table_storage='compact')
    actual_bytes = storage_bytes(periodic_action_tensors(op.actions))
    assert actual_bytes == op.actions.capacity_plan['planned_bytes']
    assert op.actions.capacity_plan['root_kernel'] == 'ewald'
    assert op.provider is provider
    dense, _ = assemble_periodic_system(system, exterior, config, kernel=kernel)
    torch.manual_seed(795)
    target = torch.randn((2, 2*basis.n_edges), device='cuda', dtype=torch.complex128)
    rhs = target@dense.mT
    edges = torch.arange(basis.n_edges, device='cuda')
    solved, log = op.solve(rhs, StoppingRule(method='block_gmres', preconditioner='rwg_diagonal',
        rtol=1e-12, atol=0., maxiter=100, restart=60), original_edges=edges)
    residual = float(torch.linalg.vector_norm(solved@dense.mT-rhs)/torch.linalg.vector_norm(rhs))
    print('complete Ewald residual/retained', residual, actual_bytes)
    assert log['converged'] and log['error_budget']['checked'] and residual < 1e-8
    x = target.detach().requires_grad_()
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        output = op.apply(x)
    assert not saved
    gradient, = torch.autograd.grad(output, x, target)
    torch.testing.assert_close(gradient, op.adjoint(target), rtol=3e-11, atol=1e-13)
    assert op.metadata['original_residual_scope'] == 'configured_ewald_quadrature_and_cutoffs'
    with pytest.raises(NotImplementedError, match='complete Ewald error plan'):
        PeriodicMultipoleOperator(provider, order=6, max_order=8,
            exterior_kernel_error_budget=KernelErrorBudget(1e-6, 1e-5))
    kernel.lattice.bloch.add_(.01)
    with pytest.raises(RuntimeError, match='mutated'):
        op.apply(target)
