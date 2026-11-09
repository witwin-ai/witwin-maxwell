"""Batched near D/K assembly equals the per-block original integration."""
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom import fmm_regional
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceMesh, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks, scalar_block_channels, scalar_regional_channels
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.physics import compile_medium

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
FREQUENCY = 3e8
CONFIG = IntegrationConfig(order=4, singular_order=4)


def _relative(a, b):
    return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b))


def _system():
    bodies = []
    for index, centre in enumerate(((0., 0., 0.), (.75, 0., 0.))):
        mesh = sphere_mesh(radius=.3, frequency=5, center=centre, device='cuda')
        bodies.append((index, None, RWGBasis(SurfaceMesh(mesh.vertices, mesh.faces)), compile_medium(mw.Material(eps_r=4.), FREQUENCY)))
    return SurfaceSystem.create(bodies)


def _selections(count, size, total, generator):
    # Contiguous index windows mimic tree leaves and share supports with neighbours.
    starts = torch.randint(0, total-size, (count,), generator=generator)
    return torch.stack([torch.arange(s, s+size) for s in starts.tolist()]).cuda()


def test_exterior_channels_many_match_per_block_including_cross_bodies():
    system = _system()
    provider = ScalarSystemBlocks(system, compile_medium(mw.Material(eps_r=1.), FREQUENCY, background=True), CONFIG)
    generator = torch.Generator().manual_seed(4)
    rows = _selections(12, 20, system.total_edges, generator)
    cols = torch.cat((rows[:4], _selections(8, 20, system.total_edges, generator)))
    d, k = provider.exterior_channels_many(rows, cols)
    for i in range(len(rows)):
        reference_d, reference_k = provider.exterior_channels(rows[i], cols[i])
        assert _relative(d[i], reference_d) < 1e-14
        assert _relative(k[i], reference_k) < 1e-13 or float(torch.linalg.vector_norm(reference_k)) == 0


def test_interior_block_channels_match_per_block():
    system = _system()
    body = system.bodies[0]
    from witwin.maxwell.mom.operator import PreparedRWGGeometry
    geometry = PreparedRWGGeometry(body.basis)
    generator = torch.Generator().manual_seed(9)
    rows = _selections(10, 16, body.basis.n_edges, generator)
    cols = torch.cat((rows[:5], _selections(5, 16, body.basis.n_edges, generator)))
    d, k = scalar_block_channels(body.basis, body.basis, rows, cols, body.interior, CONFIG, geometry=(geometry, geometry))
    for i in range(len(rows)):
        reference_d, reference_k, _ = scalar_regional_channels(body.basis, body.basis, rows[i], cols[i], body.interior,
                                                               CONFIG, geometry=(geometry, geometry))
        assert _relative(d[i], reference_d) < 1e-14
        assert _relative(k[i], reference_k) < 1e-13


@pytest.mark.parametrize('near_storage', ['scalar_dk_v1', 'scalar_reciprocal_dk_v1'])
def test_public_fmm_near_storage_equals_per_block_assembly(monkeypatch, near_storage):
    mesh = sphere_mesh(radius=.5, frequency=3, device='cuda')
    scene = mw.Scene(structures=[mw.Structure(mw.Mesh(mesh.vertices, mesh.faces, recenter=False, fill_mode='solid'),
                                              mw.Material(eps_r=4.))], sources=[mw.PlaneWave()])
    config = mw.MoMConfig(integration=CONFIG, operator_backend='fmm', riesz_storage='local_rwg_v1',
                          fmm=mw.FMMConfig(order=4, leaf_size=16, separation_ratio=.25, near_storage=near_storage),
                          stopping=StoppingRule(method='gmres', preconditioner='rwg_diagonal', rtol=1e-6, atol=0.))
    batched = mw.Simulation.mom(scene, frequency=FREQUENCY, config=config).prepare().solver.handle
    monkeypatch.setattr(fmm_regional, 'BATCHED_NEAR_ASSEMBLY', False)
    reference = mw.Simulation.mom(scene, frequency=FREQUENCY, config=config).prepare().solver.handle
    for new, old in zip((batched.actions.exterior, *batched.actions.interiors),
                        (reference.actions.exterior, *reference.actions.interiors)):
        assert new.near.entries == old.near.entries
        for (_, _, fields), (_, _, expected) in zip(new.near.groups, old.near.groups):
            assert _relative(fields, expected) < 1e-14
    generator = torch.Generator(device='cuda').manual_seed(2)
    x = torch.randn(batched.shape[0], generator=generator, device='cuda', dtype=torch.complex128)
    assert _relative(batched.apply(x), reference.apply(x)) < 1e-13
