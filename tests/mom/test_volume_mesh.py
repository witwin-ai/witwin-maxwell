"""Exact tetrahedral moments, orientation/ownership and native volume mass."""
import pytest
import torch
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
from witwin.maxwell.mom.volume_basis import CellVectorBasis

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def geometry():
    return (torch.tensor([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]], device='cuda', dtype=torch.float64),
            torch.tensor([[0,1,2,3]], device='cuda'))


def test_tetrahedral_exact_moments_and_native_mass_gradient():
    vertices, cells = geometry()
    mesh = TetrahedralVolumeMesh(vertices, cells, region_id=2)
    points, weights = mesh.quadrature(3)
    torch.testing.assert_close(weights.sum(), vertices.new_tensor(1/6), rtol=1e-13, atol=1e-15)
    torch.testing.assert_close((points*weights[..., None]).sum((0,1)), vertices.new_full((3,), 1/24), rtol=1e-13, atol=1e-15)
    torch.testing.assert_close((points.square()*weights[..., None]).sum((0,1)), vertices.new_full((3,), 1/60), rtol=1e-13, atol=1e-15)
    basis = CellVectorBasis(mesh)
    vector = torch.tensor([1+2j, 3-1j, -.5j], device='cuda', dtype=torch.complex128, requires_grad=True)
    value = basis.mass_apply(vector)
    torch.testing.assert_close(value, vector/6)
    gradient, = torch.autograd.grad(value.real.sum(), vector)
    torch.testing.assert_close(gradient, torch.full_like(vector, 1/6))
    assert basis.region_id == 2 and not mesh.report.surface_operator_bound
    vertices.add_(10)
    torch.testing.assert_close(mesh.vertices[0], vertices.new_zeros(3))
    mesh.cells[0,0] = 1
    with pytest.raises(ValueError, match='changed'): mesh.quadrature()


@pytest.mark.parametrize('kind', ['negative', 'duplicate', 'degenerate', 'region'])
def test_invalid_volume_mesh_is_rejected(kind):
    vertices, cells = geometry()
    if kind == 'negative': cells = cells[:, [0,2,1,3]]
    elif kind == 'duplicate': cells = cells.repeat(2,1)
    elif kind == 'degenerate': vertices[3] = 0
    with pytest.raises(ValueError):
        TetrahedralVolumeMesh(vertices, cells, region_id=0 if kind == 'region' else 1)


def test_conforming_subdivision_binds_only_to_its_actual_surface_body():
    import witwin.maxwell as mw
    from witwin.maxwell.mom.mesh import SurfaceMesh, RWGBasis, SurfaceSystem
    from witwin.maxwell.mom.physics import compile_medium
    from witwin.maxwell.mom.volume_mesh import bind_volume_region
    vertices, cells = geometry()
    original = TetrahedralVolumeMesh(vertices, cells, region_id=1)
    vertices = torch.cat((vertices, vertices.mean(0, keepdim=True)))
    faces = original.boundary.faces
    subdivided = torch.cat((torch.full((4,1), 4, device='cuda', dtype=torch.int64), faces), -1)
    volume = TetrahedralVolumeMesh(vertices, subdivided, region_id=1)
    torch.testing.assert_close(volume.volumes.sum(), original.volumes.sum())
    material = compile_medium(mw.Material(eps_r=2.), 3e8)
    surface = SurfaceMesh(original.boundary.vertices.clone(), original.boundary.faces.clone())
    system = SurfaceSystem.create([(0, 'body', RWGBasis(surface), material)])
    binding = bind_volume_region(volume, system)
    binding.assert_current()
    shifted = SurfaceMesh(surface.vertices+2., surface.faces.clone())
    other = SurfaceSystem.create([(0, 'body', RWGBasis(shifted), material)])
    with pytest.raises(ValueError, match='geometry'):
        bind_volume_region(volume, other)
    binding.boundary_to_surface_vertices.add_(1)
    with pytest.raises(ValueError, match='correspondence'):
        binding.assert_current()


@pytest.mark.parametrize('dtype',[torch.float64,torch.float32])
def test_red_refinement_conserves_parent_volume_and_matches_independent_surface(dtype):
    import witwin.maxwell as mw
    from witwin.maxwell.mom.mesh import RWGBasis,SurfaceSystem
    from witwin.maxwell.mom.physics import compile_medium
    from witwin.maxwell.mom.volume_mesh import bind_volume_region
    vertices = torch.tensor([[0.,0.,0.],[.07,0.,0.],[0.,.07,0.],[0.,0.,.07],[.01,.01,-.09]],device='cuda',dtype=dtype)
    original = TetrahedralVolumeMesh(vertices,torch.tensor([[0,1,2,3],[0,2,1,4]],device='cuda'),region_id=1)
    refined = original.refine(2)
    assert len(refined.cells) == 128 and refined.region_id == original.region_id
    assert refined.report.boundary_triangles == original.report.boundary_triangles*16
    tolerance = 64*torch.finfo(dtype).eps
    torch.testing.assert_close(refined.volumes.reshape(2,-1).sum(-1),original.volumes,rtol=tolerance,atol=0)
    assert bool((refined.volumes > 0).all())
    surface = original.boundary.refine(2)
    system = SurfaceSystem.create([(0,'bipyramid',RWGBasis(surface),compile_medium(mw.Material(eps_r=2.),3e8))])
    bind_volume_region(refined,system).assert_current()
    old_points,old_weights = original.quadrature(3)
    points,weights = refined.quadrature(3)
    for power in (1,2):
        torch.testing.assert_close((points.pow(power)*weights[...,None]).sum((0,1)),
            (old_points.pow(power)*old_weights[...,None]).sum((0,1)),rtol=tolerance,atol=1e-12)
    copied = original.refine(0)
    assert copied.vertices.data_ptr() != original.vertices.data_ptr()
    with pytest.raises(ValueError,match='nonnegative integer'):
        original.refine(True)
    original.vertices.add_(1)
    with pytest.raises(ValueError,match='changed'):
        original.refine()
