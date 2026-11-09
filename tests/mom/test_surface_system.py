"""Independent bodies, robust primitive gaps and global region ordering."""
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import SurfaceMesh, RWGBasis, SurfaceSystem, sphere_mesh, triangle_pair_distances
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.runtime import compile_surfaces, compile_surface

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def basis(center=(0., 0., 0.), radius=.1, subdivisions=0):
    return RWGBasis(sphere_mesh(radius=radius, center=center, subdivisions=subdivisions, device='cuda'))


def medium(eps=2.): return compile_medium(mw.Material(eps_r=eps), 3e8)


def test_separate_bodies_offsets_regions_and_mutation_guard():
    a, b = basis(), basis((.4, 0., 0.), .07)
    system = SurfaceSystem.create([(0, 'left', a, medium()), (2, 'right', b, medium(3.))])
    assert system.total_edges == 60 and system.total_faces == 40
    assert system.J_slices == (slice(0, 30), slice(30, 60))
    assert system.M_slices == (slice(60, 90), slice(90, 120))
    assert system.region('left') == 1 and system.region('right') == 2 and system.region('exterior') == 0
    with pytest.raises(ValueError, match='ambiguous'): system.region('interior')
    with pytest.raises(ValueError): system.region(True)
    assert system.pair_gaps[0][2] > .2
    torch.testing.assert_close(system.face_edges[20:], b.face_edges+30)
    b.mesh.vertices.add_(.01)
    with pytest.raises(ValueError, match='changed'): system.assert_current()
    with pytest.raises(ValueError, match='different geometry'): SurfaceSystem.create([(0, None, b, medium())])


@pytest.mark.parametrize('name', ['', 'exterior', 'interior'])
def test_reserved_or_empty_names_fail(name):
    with pytest.raises(ValueError, match='aliases'): SurfaceSystem.create([(0, name, basis(), medium())])


@pytest.mark.parametrize('kind', ['intersection', 'nested', 'identical', 'touching'])
def test_illegal_pair_geometries_fail(kind):
    a = basis()
    if kind == 'nested': b = basis(radius=.03)
    elif kind == 'identical': b = basis()
    elif kind == 'intersection': b = basis((.08, .01, 0.))
    else: b = basis((2*float(a.mesh.vertices[:, 0].max()), 0., 0.))
    with pytest.raises(ValueError, match='touch|intersect|Nested|overlapping'):
        SurfaceSystem.create([(0, 'a', a, medium()), (1, 'b', b, medium())])


def test_triangle_distance_crossing_coplanar_and_positive_gap():
    left = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]], device='cuda', dtype=torch.float64)
    crossing = torch.tensor([[[.2, .2, -1.], [.2, .2, 1.], [.8, .2, .2]]], device='cuda', dtype=torch.float64)
    assert triangle_pair_distances(left, crossing).item() == 0
    assert triangle_pair_distances(left, left+.1*torch.tensor([1., 1., 0.], device='cuda')).item() < 1e-14
    shifted = left+torch.tensor([0., 0., .3], device='cuda', dtype=torch.float64)
    torch.testing.assert_close(triangle_pair_distances(left, shifted), torch.tensor([.3], device='cuda', dtype=torch.float64))
    torch.testing.assert_close(triangle_pair_distances(shifted, left), triangle_pair_distances(left, shifted))


def test_public_scene_compilation_preserves_enabled_order_and_single_body():
    a, b = basis(), basis((.4, 0., 0.))
    structures = [mw.Structure(mw.Mesh(x.mesh.vertices, x.mesh.faces, recenter=False, fill_mode='solid', device='cuda'), mw.Material(eps_r=2.), name=name)
                  for x, name in ((a, 'a'), (b, 'b'))]
    scene = mw.Scene(structures=structures)
    config = mw.MoMConfig(subdivisions=0)
    system, exterior = compile_surfaces(scene, 3e8, config)
    assert [body.name for body in system.bodies] == ['a', 'b']
    single_scene = mw.Scene(structures=structures[:1])
    single, _ = compile_surfaces(single_scene, 3e8, config)
    original, _, _ = compile_surface(single_scene, 3e8, config)
    torch.testing.assert_close(single.bodies[0].basis.edge_lengths, original.edge_lengths, rtol=0, atol=0)
    assert single.region('interior') == 1
