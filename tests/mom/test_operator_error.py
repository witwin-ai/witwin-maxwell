"""Exhaustive streamed measurement agrees with a dense independent calculation."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig, assemble_system
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.fmm_system import MultipoleSystemActions
from witwin.maxwell.mom.operator_error import measure_original_operator_error

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('order,quadrature,expected_pass', [(0, 2, False), (6, 2, False), (6, 4, True)])
def test_streamed_error_covers_every_entry_and_exposes_cross_body_error(order, quadrature, expected_pass):
    medium = compile_medium(mw.Material(), 3e8)
    interior = compile_medium(mw.Material(eps_r=2.), 3e8)
    system = SurfaceSystem.create([(i, str(i), RWGBasis(sphere_mesh(radius=.01,
        center=(x, 0., 0.), subdivisions=0, device='cuda')), interior)
        for i, x in enumerate((-.5, .5))])
    config = IntegrationConfig(order=quadrature, singular_order=3)
    op = MultipoleSystemActions(ScalarSystemBlocks(system, medium, config), order=order, leaf_size=30)
    receipt = measure_original_operator_error(op, edge_batch=17)
    dense, _ = assemble_system(system, medium, config)
    n = system.total_edges
    actual = op.apply(torch.eye(2*n, device='cuda', dtype=op.dtype)).mT
    left = torch.ones(2*n, device='cuda', dtype=torch.float64)
    right = left.clone()
    left[:n], right[n:] = abs(medium.eta), abs(medium.eta)
    error = left[:, None]*(actual-dense)*right
    reference = left[:, None]*dense*right
    expected = torch.linalg.vector_norm(error)/torch.linalg.vector_norm(reference)
    assert receipt['metrics']['global']['entries'] == (2*n)**2
    assert receipt['metrics']['cross_body']['entries'] == (2*n)**2//2
    assert receipt['metrics']['global']['relative_frobenius'] == pytest.approx(float(expected), rel=1e-7)
    body_ids = torch.bucketize(torch.arange(n, device='cuda'), op.provider.edge_stops, right=True).repeat(2)
    cross = body_ids[:, None] != body_ids
    cross_expected = torch.linalg.vector_norm(error[cross])/torch.linalg.vector_norm(reference[cross])
    assert receipt['metrics']['cross_body']['relative_frobenius'] == pytest.approx(float(cross_expected), rel=1e-10)
    print('order', order, 'original quadrature', quadrature, receipt)
    assert receipt['metrics']['cross_body']['passed'] is expected_pass
    assert not receipt['physical_qualification']
    with pytest.raises(ValueError, match='edge_batch'):
        measure_original_operator_error(op, edge_batch=True)


@pytest.mark.parametrize('scale', [1e-200, 1e200])
def test_norm_accumulation_preserves_extreme_nonzero_errors(scale):
    from types import SimpleNamespace
    reference = torch.ones((4, 4), device='cuda', dtype=torch.complex128)*scale
    class Provider:
        system = SimpleNamespace(total_edges=2)
        exterior = SimpleNamespace(eta=1.)
        edge_stops = torch.tensor([1, 2], device='cuda')
        def __call__(self, rows, columns):
            rr, cc = torch.cat((rows, rows+2)), torch.cat((columns, columns+2))
            return reference[rr[:, None], cc]
    class Actions:
        provider = Provider()
        device, dtype = reference.device, reference.dtype
        def assert_current(self):
            pass
        def apply(self, probes):
            return probes@(2*reference).mT
    receipt = measure_original_operator_error(Actions(), edge_batch=1)
    for metric in receipt['metrics'].values():
        assert metric['relative_frobenius'] == pytest.approx(1.)
        assert metric['absolute_frobenius'] > 0
        assert not metric['passed']


@pytest.mark.parametrize('order,expected_pass', [(0, False), (6, True)])
def test_periodic_exhaustive_measurement_uses_matching_original_images(order, expected_pass):
    from witwin.maxwell.mom.mesh import SurfaceMesh
    from witwin.maxwell.mom.nearfield import scalar_regional_block
    from witwin.maxwell.mom.periodic import LatticeSpec
    from witwin.maxwell.mom.periodic_nearfield import PeriodicImageSystemBlocks
    from witwin.maxwell.mom.fmm_periodic_system import PeriodicMultipoleSystemActions
    exterior = compile_medium(mw.Material(eps_r=1.5, sigma_e=.02), 3e8)
    system = SurfaceSystem.create([(i, str(i), RWGBasis(sphere_mesh(radius=.01,
        center=(x, 0., 0.), subdivisions=0, device='cuda')),
        compile_medium(mw.Material(eps_r=2.+i, mu_r=1.2+i*.2), 3e8)) for i, x in enumerate((-.2, .2))])
    config = IntegrationConfig(order=4, singular_order=3)
    provider = ScalarSystemBlocks(system, exterior, config)
    lattice = LatticeSpec(torch.tensor([[1., 0., 0.]], device='cuda', dtype=torch.float64),
                          torch.tensor([.37, 0., 0.], device='cuda', dtype=torch.float64))
    op = PeriodicMultipoleSystemActions(provider, lattice, order=order, shells=1,
        leaf_size=30, separation_ratio=.1, table_storage='compact')
    assert isinstance(op.provider, PeriodicImageSystemBlocks)
    receipt = measure_original_operator_error(op, edge_batch=17)
    reference, _ = assemble_system(system, exterior, config)
    n = system.total_edges
    for image in (-1, 1):
        index = torch.tensor([image], device='cuda')
        for target in system.bodies:
            re = torch.arange(target.basis.n_edges, device='cuda')
            rr = torch.cat((re+target.edge_offset, re+target.edge_offset+n))
            for source in system.bodies:
                ce = torch.arange(source.basis.n_edges, device='cuda')
                cc = torch.cat((ce+source.edge_offset, ce+source.edge_offset+n))
                shifted = RWGBasis(SurfaceMesh(source.mesh.vertices+lattice.translations(index), source.mesh.faces, validate=False))
                block, _ = scalar_regional_block(target.basis, shifted, re, ce, (exterior,), config)
                reference[rr[:, None], cc] += lattice.phase(index)*block
    actual = op.apply(torch.eye(2*n, device='cuda', dtype=op.dtype)).mT
    left = torch.ones(2*n, device='cuda', dtype=torch.float64)
    right = left.clone()
    left[:n], right[n:] = abs(exterior.eta), abs(exterior.eta)
    difference = left[:, None]*(actual-reference)*right
    balanced = left[:, None]*reference*right
    body = torch.bucketize(torch.arange(n, device='cuda'), provider.edge_stops, right=True).repeat(2)
    for name, mask in [('global', torch.ones_like(body[:, None] != body)), ('cross_body', body[:, None] != body)]:
        metric = receipt['metrics'][name]
        relative = torch.linalg.vector_norm(difference[mask])/torch.linalg.vector_norm(balanced[mask])
        assert metric['entries'] == int(mask.sum())
        assert metric['relative_frobenius'] == pytest.approx(float(relative), rel=1e-7, abs=1e-13)
    print('periodic original exhaustive', order, receipt)
    assert receipt['metrics']['cross_body']['passed'] is expected_pass
    assert op.receipt()['original_reference_scope'] == 'configured_finite_periodic_image_cube'
    assert not op.receipt()['infinite_periodic_certified']
    with pytest.raises(ValueError, match='share lattice and image shells'):
        PeriodicMultipoleSystemActions(op.provider, lattice, shells=2)
    op.provider.shells = 2
    with pytest.raises(ValueError, match='changed'):
        measure_original_operator_error(op)
