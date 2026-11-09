"""Periodic system composition preserves one exterior and local magnetic interiors."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceMesh, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig, assemble_system
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks, scalar_regional_block
from witwin.maxwell.mom.periodic import LatticeSpec
from witwin.maxwell.mom.fmm_periodic_system import PeriodicMultipoleSystemActions

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize("table_storage", ["dense", "compact"])
def test_periodic_system_matches_full_image_reference_and_fixed_adjoint_backward(table_storage):
    exterior = compile_medium(mw.Material(eps_r=1.5, sigma_e=.02), 3e8)
    system = SurfaceSystem.create([(i, str(i), RWGBasis(sphere_mesh(radius=.01,
        center=(x, 0., 0.), subdivisions=0, device='cuda')),
        compile_medium(mw.Material(eps_r=2.+i, mu_r=1.2+i*.2), 3e8)) for i, x in enumerate((-.2, .2))])
    config = IntegrationConfig(order=4, singular_order=3)
    provider = ScalarSystemBlocks(system, exterior, config)
    lattice = LatticeSpec(torch.tensor([[1., 0., 0.]], device='cuda', dtype=torch.float64),
                          torch.tensor([.37, 0., 0.], device='cuda', dtype=torch.float64))
    op = PeriodicMultipoleSystemActions(provider, lattice, order=6, shells=1, leaf_size=30, separation_ratio=.1,
        table_storage=table_storage, m2l_row_batch=7)
    assert all(region.media == (body.interior,) for region, body in zip(op.interiors, system.bodies))
    assert op.exterior.test.basis is op.provider
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
                translated = RWGBasis(SurfaceMesh(source.mesh.vertices+lattice.translations(index), source.mesh.faces, validate=False))
                block, _ = scalar_regional_block(target.basis, translated, re, ce, (exterior,), config)
                reference[rr[:, None], cc] += lattice.phase(index)*block
    actual = op.apply(torch.eye(2*n, device='cuda', dtype=torch.complex128)).mT
    left = torch.ones(2*n, device='cuda', dtype=torch.float64)
    right = left.clone()
    left[:n], right[n:] = abs(exterior.eta), abs(exterior.eta)
    difference = left[:, None]*(actual-reference)*right
    balanced = left[:, None]*reference*right
    body_ids = torch.bucketize(torch.arange(n, device='cuda'), provider.edge_stops, right=True).repeat(2)
    cross = body_ids[:, None] != body_ids
    for mask in (torch.ones_like(cross), cross):
        relative = float(torch.linalg.vector_norm(difference[mask])/torch.linalg.vector_norm(balanced[mask]))
        print('periodic full system relative', relative)
        assert relative < 1e-8
    x = torch.randn((2, 2*n), device='cuda', dtype=actual.dtype, requires_grad=True)
    y = torch.randn_like(x)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        result = op.apply(x)
    assert not saved
    derivative, = torch.autograd.grad(result, x, y)
    torch.testing.assert_close(derivative, op.adjoint(y), rtol=3e-11, atol=1e-14)
    torch.testing.assert_close(op.adjoint(y), y@actual.conj(), rtol=3e-11, atol=1e-14)
    assert not op.receipt()['interiors_periodic']
    op.indices[0].add_(1)
    with pytest.raises(ValueError, match='changed'):
        op.apply(x)
