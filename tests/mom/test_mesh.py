"""GPU topology, current conservation, metric and projection acceptance."""

import math

import pytest
import torch

from witwin.maxwell.mom.mesh import (
    RWGBasis, SurfaceMesh, _has_intersections, deformed_sphere_mesh,
    ellipsoid_mesh, project_current, sphere_mesh,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required by MoM")


def test_closed_sphere_geometry_and_quadrature():
    mesh = sphere_mesh(radius=0.3, subdivisions=1)
    assert (len(mesh.vertices), len(mesh.faces), mesh.validation_report.edges) == (42, 80, 120)
    assert mesh.vertices.is_cuda
    assert torch.all((mesh.normals * mesh.centroids).sum(-1) > 0)
    for order in (1, 2, 4):
        points, weights = mesh.quadrature(order)
        torch.testing.assert_close(weights.sum(1), mesh.areas)
        assert points.is_cuda
    assert abs(mesh.areas.sum().item() / (4 * torch.pi * 0.3**2) - 1) < 0.08


def test_mesh_rejects_invalid_topology_orientation_and_degeneracy():
    mesh = sphere_mesh(subdivisions=0)
    with pytest.raises(ValueError, match="closed"):
        SurfaceMesh(mesh.vertices, mesh.faces[:-1])
    flipped = mesh.faces.clone()
    flipped[0] = flipped[0].flip(0)
    with pytest.raises(ValueError, match="orientation"):
        SurfaceMesh(mesh.vertices, flipped)
    with pytest.raises(ValueError, match="outward"):
        SurfaceMesh(mesh.vertices, mesh.faces.flip(1))
    degenerate = mesh.vertices.clone()
    degenerate[11] = degenerate[0]
    with pytest.raises(ValueError, match="degenerate"):
        SurfaceMesh(degenerate, mesh.faces)
    with pytest.raises(ValueError, match="connected"):
        SurfaceMesh(torch.cat((mesh.vertices, mesh.vertices + 4)), torch.cat((mesh.faces, mesh.faces + len(mesh.vertices))))


@pytest.mark.parametrize("other", [
    [[0.2, 0.2, -1.], [0.2, 0.2, 1.], [0.7, 0.2, 0.]],
    [[0.1, 0.1, 0.], [0.2, 0.1, 0.], [0.1, 0.2, 0.]],
    [[0.2, -0.2, 0.], [0.2, 1.2, 0.], [0.8, 0.2, 0.]],
])
def test_self_intersection_predicates(other):
    triangles = torch.tensor([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]], other], dtype=torch.float64, device="cuda")
    assert _has_intersections(triangles)


@pytest.mark.parametrize("other", [
    [[0., 0., 0.], [.3, .3, -1.], [.3, .3, 1.]],
    [[0., 0., 0.], [1., 0., 0.], [.3, .3, 0.]],
    [[0., 0., 0.], [1., 1., 0.], [-.5, .5, 0.]],
])
def test_shared_vertices_do_not_hide_improper_crossings(other):
    triangles = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], other],
                             dtype=torch.float64, device="cuda")
    assert _has_intersections(triangles)


def test_radial_harmonic_shared_endpoint_roundoff():
    # This legal smooth radial configuration formerly classified an exact
    # shared endpoint as t=3.00488e-14, slightly beyond the endpoint tolerance.
    harmonics = {(2, 0): 0.15082377802580596, (3, 2): -0.10666592493653297}
    bound = sum(abs(coefficient) * math.sqrt((2 * degree + 1) / (4 * math.pi))
                for (degree, _), coefficient in harmonics.items())
    mesh = deformed_sphere_mesh(radius=.25 / (1 + bound), subdivisions=2,
                                harmonics=harmonics)
    assert mesh.validation_report.self_intersections == 0
    assert torch.all((mesh.normals * mesh.centroids).sum(-1) > 0)


def test_rwg_charge_conservation_exact_metric_and_continuity():
    basis = RWGBasis(sphere_mesh(subdivisions=1))
    torch.testing.assert_close(basis.mesh.areas @ basis.divergence, torch.zeros(basis.n_edges, device="cuda", dtype=torch.float64), atol=1e-15, rtol=0)
    mass = basis.mass_matrix()
    torch.testing.assert_close(mass, mass.T)
    assert torch.linalg.eigvalsh(mass).min() > 0
    points, weights = basis.mesh.quadrature(4)
    coefficients = torch.randn(basis.n_edges, device="cuda", dtype=torch.complex128)
    current = basis.current(coefficients, points)
    integral = (current.abs().square().sum(-1) * weights).sum()
    torch.testing.assert_close(integral, torch.vdot(coefficients, mass.to(coefficients.dtype) @ coefficients).real)
    assert torch.max((current * basis.mesh.normals[:, None]).sum(-1).abs()) < 1e-14
    midpoint = basis.mesh.vertices[basis.edges].mean(1)
    tangent = (basis.mesh.vertices[basis.edges[:, 1]] - basis.mesh.vertices[basis.edges[:, 0]]) / basis.edge_lengths[:, None]
    plus = basis.edge_lengths[:, None] / (2 * basis.mesh.areas[basis.plus_faces, None]) * (midpoint - basis.mesh.vertices[basis.plus_vertices])
    minus = -basis.edge_lengths[:, None] / (2 * basis.mesh.areas[basis.minus_faces, None]) * (midpoint - basis.mesh.vertices[basis.minus_vertices])
    plus_flux = (plus * torch.linalg.cross(tangent, basis.mesh.normals[basis.plus_faces])).sum(-1)
    minus_flux = (minus * torch.linalg.cross(-tangent, basis.mesh.normals[basis.minus_faces])).sum(-1)
    torch.testing.assert_close(plus_flux, torch.ones_like(plus_flux))
    torch.testing.assert_close(minus_flux, -torch.ones_like(minus_flux))


def test_rwg_local_mass_blocks_reconstruct_dense_metric():
    basis = RWGBasis(sphere_mesh(subdivisions=1))
    blocks = basis.mass_blocks()
    assert blocks.shape == (len(basis.mesh.faces), 3, 3)
    assert blocks.dtype == basis.mesh.dtype
    assert blocks.device == basis.mesh.device
    torch.testing.assert_close(blocks, blocks.transpose(-1, -2), atol=0, rtol=0)

    indices = basis.face_edges[:, :, None] * basis.n_edges + basis.face_edges[:, None, :]
    reconstructed = torch.zeros(
        basis.n_edges**2, dtype=blocks.dtype, device=blocks.device
    )
    reconstructed.scatter_add_(0, indices.reshape(-1), blocks.reshape(-1))
    torch.testing.assert_close(
        reconstructed.reshape(basis.n_edges, basis.n_edges),
        basis.mass_matrix(),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_rwg_local_mass_apply_matches_dense_batched_and_autograd(dtype):
    basis = RWGBasis(sphere_mesh(subdivisions=0))
    coefficients = torch.randn(
        2, 3, basis.n_edges, dtype=dtype, device="cuda", requires_grad=True
    )
    actual = basis.mass_apply(coefficients)
    expected = torch.einsum(
        "ij,...j->...i", basis.mass_matrix().to(dtype), coefficients
    )
    torch.testing.assert_close(actual, expected, atol=2e-15, rtol=2e-15)

    probe = torch.randn_like(actual)
    actual_gradient, = torch.autograd.grad(
        (actual.conj() * probe).real.sum(), coefficients, retain_graph=True
    )
    expected_gradient, = torch.autograd.grad(
        (expected.conj() * probe).real.sum(), coefficients
    )
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-15, rtol=2e-15)


def test_rwg_local_mass_apply_rejects_invalid_coefficients():
    basis = RWGBasis(sphere_mesh(subdivisions=0))
    with pytest.raises(ValueError, match="count"):
        basis.mass_apply(torch.ones(basis.n_edges + 1, device="cuda"))
    with pytest.raises(ValueError, match="floating-point or complex"):
        basis.mass_apply(torch.ones(basis.n_edges, dtype=torch.int64, device="cuda"))


def test_vertex_and_face_permutations_transform_current_and_mass():
    mesh = sphere_mesh(subdivisions=0)
    basis = RWGBasis(mesh)
    permutation = torch.arange(len(mesh.vertices) - 1, -1, -1, device="cuda")
    reordered = SurfaceMesh(mesh.vertices[permutation], permutation[mesh.faces.flip(0)])
    other = RWGBasis(reordered)
    # Reverse numbering flips every canonical global edge direction.
    old_edge_lookup = {tuple(e): i for i, e in enumerate(basis.edges.cpu().tolist())}
    mapped = permutation[other.edges].sort(-1).values.cpu().tolist()
    indices = torch.tensor([old_edge_lookup[tuple(e)] for e in mapped], device="cuda")
    torch.testing.assert_close(other.mass_matrix(), basis.mass_matrix()[indices][:, indices])
    x = torch.randn(basis.n_edges, dtype=torch.float64, device="cuda")
    points, _ = mesh.quadrature(2)
    torch.testing.assert_close(other.current(-x[indices], points.flip(0)), basis.current(x, points).flip(0))


def test_projection_identity_batched_complex_and_autograd():
    basis = RWGBasis(sphere_mesh(subdivisions=0))
    coefficients = torch.randn(2, basis.n_edges, dtype=torch.complex128, device="cuda", requires_grad=True)
    projected, error = project_current(basis, basis, coefficients)
    torch.testing.assert_close(projected, coefficients, atol=1e-13, rtol=1e-13)
    assert error.max() < 1e-13
    projected.abs().square().sum().backward()
    torch.testing.assert_close(coefficients.grad, 2 * coefficients)


def test_ellipsoid_and_harmonic_geometry_scale():
    mesh = ellipsoid_mesh((1., 2., 3.), subdivisions=1)
    torch.testing.assert_close((mesh.vertices / torch.tensor((1., 2., 3.), device="cuda")).square().sum(-1), torch.ones(len(mesh.vertices), device="cuda", dtype=torch.float64))
    deformed = deformed_sphere_mesh(harmonics={(2, 0): 0.15, (3, -1): 0.04}, subdivisions=1)
    assert deformed.validation_report.closed
    small = RWGBasis(sphere_mesh(radius=1e-5, subdivisions=0))
    large = RWGBasis(sphere_mesh(radius=1., subdivisions=0))
    torch.testing.assert_close(small.mass_matrix(), large.mass_matrix() * 1e-10)


def test_sphere_three_level_area_convergence():
    errors = [abs(sphere_mesh(subdivisions=n).areas.sum().item() - 4 * torch.pi) for n in range(3)]
    assert errors[2] < errors[1] / 3 < errors[0] / 9


def test_projection_converges_on_independently_rotated_remesh():
    source = RWGBasis(sphere_mesh(subdivisions=2))
    def analytic(points):
        axis = torch.tensor([0., 0., 1.], dtype=points.dtype, device=points.device)
        return torch.linalg.cross(axis.expand_as(points), points)
    points, weights = source.mesh.quadrature(3)
    rhs_local = torch.einsum("tqkd,tqd,tq->tk", source.local_values(points), analytic(points), weights)
    rhs = torch.zeros(source.n_edges, dtype=torch.float64, device="cuda").scatter_add(0, source.face_edges.flatten(), rhs_local.flatten())
    coefficients = torch.linalg.solve(source.mass_matrix(), rhs)
    angle = 0.217
    rotation = torch.tensor([[math.cos(angle), -math.sin(angle), 0.], [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]], dtype=torch.float64, device="cuda")
    errors = []
    for level in range(3):
        mesh = sphere_mesh(subdivisions=level)
        target = RWGBasis(SurfaceMesh(mesh.vertices @ rotation.T, mesh.faces))
        projected, estimate = project_current(source, target, coefficients)
        query, w = target.mesh.quadrature(4)
        exact = analytic(query)
        exact = exact - (exact * target.mesh.normals[:, None]).sum(-1, keepdim=True) * target.mesh.normals[:, None]
        difference = target.current(projected, query) - exact
        errors.append(((difference.square().sum(-1) * w).sum() / (exact.square().sum(-1) * w).sum()).sqrt().item())
        assert torch.isfinite(estimate)
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 0.08


@pytest.mark.host_validation
def test_cpu_numerical_geometry_is_rejected():
    with pytest.raises(ValueError, match="CUDA"):
        sphere_mesh(device="cpu")


def test_uniform_planar_refinement_preserves_closed_tetrahedral_geometry():
    from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
    vertices = torch.tensor([[0.,0.,0.],[.07,0.,0.],[0.,.07,0.],[0.,0.,.07]],device='cuda',dtype=torch.float64)
    volume = TetrahedralVolumeMesh(vertices,torch.tensor([[0,1,2,3]],device='cuda'),region_id=1)
    original = volume.boundary
    refined = original.refine(2)
    assert len(refined.faces) == 64 and refined.validation_report.closed
    assert not _has_intersections(refined.triangles)
    torch.testing.assert_close(refined.areas.sum(),original.areas.sum(),rtol=1e-14,atol=0)
    def enclosed(mesh):
        triangles = mesh.triangles
        return (triangles[:,0]*torch.linalg.cross(triangles[:,1],triangles[:,2])).sum()/6
    torch.testing.assert_close(enclosed(refined),enclosed(original),rtol=1e-14,atol=0)
    copy = original.refine(0)
    assert copy.vertices.data_ptr() != original.vertices.data_ptr()
    with pytest.raises(ValueError,match='nonnegative integer'):
        original.refine(True)
