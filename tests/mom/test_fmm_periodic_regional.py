"""Periodic RWG actions match separately integrated translated physical images."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceMesh, sphere_mesh
from witwin.maxwell.mom.cluster import build_support_octree
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.nearfield import scalar_regional_block
from witwin.maxwell.mom.fmm_moments import NestedRWGMultipoles
from witwin.maxwell.mom.periodic import LatticeSpec
from witwin.maxwell.mom.fmm_periodic_regional import PeriodicMultipoleRegionalOperator

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def projection(center):
    basis = RWGBasis(sphere_mesh(radius=.01, center=(center, 0., 0.), subdivisions=0, device='cuda'))
    tri = basis.mesh.triangles
    support = torch.cat((tri[basis.plus_faces], tri[basis.minus_faces]), 1)
    return NestedRWGMultipoles(basis, build_support_octree(support.amin(1), support.amax(1), leaf_size=30), order=6)


@pytest.mark.parametrize('across_seam', [False, True])
def test_periodic_region_matches_image_weak_sum_and_complex_adjoint(across_seam):
    test = projection(-.48 if across_seam else 0.)
    source = projection(.48) if across_seam else test
    lattice = LatticeSpec(torch.tensor([[1., 0., 0.]], device='cuda', dtype=torch.float64),
                          torch.tensor([.37, 0., 0.], device='cuda', dtype=torch.float64))
    medium = compile_medium(mw.Material(eps_r=2., sigma_e=.02), 3e8)
    config = IntegrationConfig(order=4, singular_order=3)
    op = PeriodicMultipoleRegionalOperator(test, source, medium, lattice, shells=1, integration=config)
    assert len(op.regions) == 1
    assert int(op.remote.near_indices[0, 0]) == (-1 if across_seam else 0)
    edges = torch.arange(source.tree.size, device='cuda')
    reference = torch.zeros(op.shape, device='cuda', dtype=torch.complex128)
    for image in (-1, 0, 1):
        index = torch.tensor([image], device='cuda')
        basis = source.basis if image == 0 else RWGBasis(SurfaceMesh(
            source.basis.mesh.vertices+lattice.translations(index), source.basis.mesh.faces, validate=False))
        block, _ = scalar_regional_block(test.basis, basis, edges, edges, (medium,), config)
        reference += lattice.phase(index)*block
    actual = op.apply(torch.eye(op.shape[1], device='cuda', dtype=torch.complex128)).mT
    left = torch.ones(op.shape[0], device='cuda', dtype=torch.float64)
    right = left.clone()
    left[:len(edges)], right[len(edges):] = abs(medium.eta), abs(medium.eta)
    error = torch.linalg.vector_norm(left[:, None]*(actual-reference)*right)/torch.linalg.vector_norm(left[:, None]*reference*right)
    print('periodic region seam/error', across_seam, float(error), op.receipt())
    assert float(error) < 1e-8
    x = torch.randn((2, op.shape[1]), device='cuda', dtype=actual.dtype, requires_grad=True)
    y = torch.randn_like(x)
    torch.testing.assert_close(op.adjoint(y), y@actual.conj(), rtol=3e-11, atol=1e-14)
    derivative, = torch.autograd.grad(op.apply(x), x, y)
    torch.testing.assert_close(derivative, op.adjoint(y), rtol=3e-11, atol=1e-14)
    op.phases.add_(1)
    with pytest.raises(ValueError, match='changed'):
        op.apply(x)


@pytest.mark.parametrize('order,separation,expected_pass', [(6, .1, True), (14, .5, False)])
def test_global_periodic_exterior_uses_shared_tree_and_no_interior_images(order, separation, expected_pass):
    from witwin.maxwell.mom.mesh import SurfaceSystem
    from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
    medium = compile_medium(mw.Material(eps_r=1.5, sigma_e=.02), 3e8)
    bodies = [(i, str(i), projection(x).basis, compile_medium(mw.Material(eps_r=2.+3*i), 3e8))
              for i, x in enumerate((-.45, .45))]
    system = SurfaceSystem.create(bodies)
    config = IntegrationConfig(order=4, singular_order=3)
    provider = ScalarSystemBlocks(system, medium, config)
    lower, upper = provider.support_boxes()
    moments = NestedRWGMultipoles(provider, build_support_octree(lower, upper, leaf_size=30), order=order)
    lattice = LatticeSpec(torch.tensor([[1., 0., 0.]], device='cuda', dtype=torch.float64),
                          torch.tensor([.37, 0., 0.], device='cuda', dtype=torch.float64))
    op = PeriodicMultipoleRegionalOperator(moments, moments, medium, lattice, shells=2, integration=config, separation_ratio=separation)
    assert op.remote.far_count+len(op.regions) == 5
    assert all(region.test is moments for region in op.regions)
    assert all(isinstance(region.source.basis, ScalarSystemBlocks) for region in op.regions)
    n = system.total_edges
    reference = torch.zeros((2*n, 2*n), device='cuda', dtype=torch.complex128)
    for image in range(-2, 3):
        index = torch.tensor([image], device='cuda')
        for test_body in system.bodies:
            re = torch.arange(test_body.basis.n_edges, device='cuda')
            rr = torch.cat((re+test_body.edge_offset, re+test_body.edge_offset+n))
            for source_body in system.bodies:
                ce = torch.arange(source_body.basis.n_edges, device='cuda')
                cc = torch.cat((ce+source_body.edge_offset, ce+source_body.edge_offset+n))
                basis = source_body.basis if image == 0 else RWGBasis(SurfaceMesh(
                    source_body.mesh.vertices+lattice.translations(index), source_body.mesh.faces, validate=False))
                block, _ = scalar_regional_block(test_body.basis, basis, re, ce, (medium,), config)
                reference[rr[:, None], cc] += lattice.phase(index)*block
    actual = op.apply(torch.eye(2*n, device='cuda', dtype=reference.dtype)).mT
    scale = torch.ones(2*n, device='cuda', dtype=torch.float64)
    left, right = scale.clone(), scale.clone()
    left[:n], right[n:] = abs(medium.eta), abs(medium.eta)
    error = left[:, None]*(actual-reference)*right
    balanced = left[:, None]*reference*right
    body = torch.bucketize(torch.arange(n, device='cuda'), provider.edge_stops, right=True).repeat(2)
    cross = body[:, None] != body
    errors = []
    for mask in (torch.ones_like(cross), cross):
        relative = float(torch.linalg.vector_norm(error[mask])/torch.linalg.vector_norm(balanced[mask]))
        print('periodic global error', relative)
        errors.append(relative)
    assert (max(errors) < 1e-8) is expected_pass
    probe = torch.randn((2, 2*n), device='cuda', dtype=reference.dtype)
    torch.testing.assert_close(op.adjoint(probe), probe@actual.conj(), rtol=3e-11, atol=1e-14)


def test_hierarchy_budget_includes_near_image_truncation_before_tables(monkeypatch):
    from witwin.maxwell.mom.mesh import SurfaceSystem
    from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
    from witwin.maxwell.mom.fmm_periodic_error import periodic_hierarchy_error_bound, PeriodicMultipoleError
    from witwin.maxwell.mom.interpolation_error import KernelErrorBudget
    import witwin.maxwell.mom.fmm_periodic_regional as module
    medium = compile_medium(mw.Material(eps_r=1.5, sigma_e=.2), 3e8)
    bodies = [(i, str(i), projection(x).basis, compile_medium(mw.Material(eps_r=2.), 3e8))
              for i, x in enumerate((-.2, .2))]
    provider = ScalarSystemBlocks(SurfaceSystem.create(bodies), medium, IntegrationConfig(order=4, singular_order=3))
    lower, upper = provider.support_boxes()
    moments = NestedRWGMultipoles(provider, build_support_octree(lower, upper, leaf_size=30), order=2)
    lattice = LatticeSpec(torch.tensor([[1., 0., 0.]], device='cuda', dtype=torch.float64),
                          torch.tensor([.37, 0., 0.], device='cuda', dtype=torch.float64))
    bound = periodic_hierarchy_error_bound(moments.tree, moments.tree, lattice, medium.k,
        order=2, shells=3, separation_ratio=.1)
    print('complete hierarchy bound', bound)
    assert bound['hierarchical_far_blocks'] > 0
    assert bound['near_hierarchy_green_absolute'] > 0
    assert bound['near_hierarchy_gradient_component_absolute'] > 0
    remote = bound['remote']
    budget = KernelErrorBudget(green_absolute=(remote['green_absolute']+bound['green_absolute'])/2,
        gradient_component_absolute=(remote['gradient_component_absolute']+bound['gradient_component_absolute'])/2)
    original = module.PeriodicImageM2L
    def forbidden(*args, **kwargs):
        raise AssertionError('Complete hierarchy rejection must precede table allocation')
    monkeypatch.setattr(module, 'PeriodicImageM2L', forbidden)
    with pytest.raises(PeriodicMultipoleError) as failure:
        PeriodicMultipoleRegionalOperator(moments, moments, medium, lattice, integration=provider.config,
            shells=3, separation_ratio=.1, hierarchy_kernel_error_budget=budget)
    assert failure.value.receipt['hierarchical_far_blocks'] == bound['hierarchical_far_blocks']
    monkeypatch.setattr(module, 'PeriodicImageM2L', original)
    admitted = KernelErrorBudget(green_absolute=bound['green_absolute']*1.01,
        gradient_component_absolute=bound['gradient_component_absolute']*1.01)
    op = PeriodicMultipoleRegionalOperator(moments, moments, medium, lattice, integration=provider.config,
        shells=3, separation_ratio=.1, hierarchy_kernel_error_budget=admitted, table_storage='compact')
    # Independently evaluate the constructed polynomial kernel at interior
    # support-sphere offsets, for every planned near-image far block.
    from witwin.maxwell.mom.fmm_expansion import monomials
    from witwin.maxwell.mom.kernels import FREE_SPACE
    errors = torch.zeros(2, device='cuda', dtype=torch.float64)
    for region in op.regions:
        maximum = torch.zeros_like(errors)
        for a, b, coupling in region.far:
            x = moments.centers.new_tensor([[.2, -.3, .1], [-.1, .2, -.3]])*region.test.radii[a]
            y = moments.centers.new_tensor([[-.2, .1, .3], [.3, -.1, .2]])*region.source.radii[b]
            green, gradient = coupling._compact._tables(0, moments.count)
            tx, sy = monomials(x, 2).to(green.dtype), monomials(y, 2).to(green.dtype)
            actual = tx@green@sy.mT
            actual_gradient = torch.einsum('ib,bca,jc->ija', tx, gradient, sy)
            reference, reference_gradient = FREE_SPACE.evaluate(
                region.test.centers[a]-region.source.centers[b]+x[:, None]-y[None], medium.k)
            maximum = torch.maximum(maximum, torch.stack(((actual-reference).abs().max(),
                (actual_gradient-reference_gradient).abs().max())))
        errors += maximum
    assert float(errors[0]) <= bound['near_hierarchy_green_absolute']
    assert float(errors[1]) <= bound['near_hierarchy_gradient_component_absolute']
    receipt = op.receipt()
    assert receipt['hierarchy_kernel_admitted']
    assert len(op.regions) == bound['hierarchical_images']
    assert sum(len(r.far) for r in op.regions) == bound['hierarchical_far_blocks']
    receipt['hierarchy_kernel_bound']['green_absolute'] = 0
    assert op.receipt()['hierarchy_kernel_bound']['green_absolute'] > 0
    op.hierarchy_kernel_error_budget = budget
    with pytest.raises(ValueError, match='changed'):
        op.assert_current()
