"""Edge classification, open RWG spaces and per-body formulation scope."""
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.topology import (BOUNDARY, INTERIOR, JUNCTION, BodyKind, body_kind, classify_surface,
                                         require_open_sheet, scene_formulation)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def grid_faces(nx, ny, hole=None):
    faces = []
    for i in range(nx):
        for j in range(ny):
            if hole is not None and (i, j) in hole:
                continue
            a = i*(ny+1)+j
            b, c, d = a+ny+1, a+ny+2, a+1
            faces += [(a, b, c), (a, c, d)]
    return torch.tensor(faces)


def compact(faces):
    used, inverse = torch.unique(faces, return_inverse=True)
    return inverse, len(used)


@pytest.mark.host_validation
def test_plate_edges_match_euler_counts():
    n = 3
    topology = classify_surface(grid_faces(n, n), (n+1)**2)
    assert topology.boundary_edges == 4*n
    assert topology.interior_edges == 3*n*n-2*n
    assert topology.junction_edges == 0 and topology.components == 1
    assert topology.orientable and topology.manifold_vertices and not topology.closed
    # Plus incidence traverses its sorted edge forwards.
    interior = topology.kind == INTERIOR
    faces, slots = topology.incidence_faces[interior], topology.incidence_slots[interior]
    assert bool((topology.face_edge_signs[faces[:, 0], slots[:, 0]] == 1).all())
    assert bool((topology.face_edge_signs[faces[:, 1], slots[:, 1]] == -1).all())
    assert bool((topology.incidence_faces[topology.kind == BOUNDARY][:, 1] == -1).all())


@pytest.mark.host_validation
def test_plate_with_hole_has_two_boundary_loops_of_edges():
    faces, count = compact(grid_faces(3, 3, hole={(1, 1)}))
    topology = require_open_sheet(classify_surface(faces, count))
    assert topology.boundary_edges == 12+4
    assert topology.interior_edges == 3*9-2*3-5  # removed diagonal plus four sides
    assert 16-len(topology.edges)+len(faces) == 0  # annulus Euler characteristic
    assert topology.components == 1 and topology.manifold_vertices


@pytest.mark.host_validation
def test_closed_tetrahedron_is_closed():
    topology = classify_surface(torch.tensor([(0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)]), 4)
    assert topology.closed and topology.interior_edges == 6 and topology.orientable


@pytest.mark.host_validation
def test_t_junction_is_classified_then_rejected():
    topology = classify_surface(torch.tensor([(0, 1, 2), (1, 0, 3), (0, 1, 4)]), 5)
    assert topology.junction_edges == 1
    assert int((topology.kind == JUNCTION).sum()) == 1 and int(topology.face_count.max()) == 3
    with pytest.raises(NotImplementedError, match='junction'):
        require_open_sheet(topology)


@pytest.mark.host_validation
def test_non_manifold_vertex_and_inconsistent_orientation_are_rejected():
    pinched = classify_surface(torch.tensor([(0, 1, 2), (0, 3, 4)]), 5)
    assert not pinched.manifold_vertices and pinched.components == 2
    with pytest.raises(ValueError, match='non-manifold'):
        require_open_sheet(pinched)
    flipped = classify_surface(torch.tensor([(0, 1, 2), (0, 1, 3)]), 4)
    assert not flipped.orientable
    with pytest.raises(ValueError, match='orientation'):
        require_open_sheet(flipped)


@pytest.mark.host_validation
@pytest.mark.parametrize('faces, count, match', [
    (torch.tensor([(0, 1, 1)]), 3, 'invalid or repeated'),
    (torch.tensor([(0, 1, 5)]), 3, 'invalid or repeated'),
    (torch.tensor([(0, 1, 2)]), 4, 'unused'),
])
def test_invalid_face_arrays_raise(faces, count, match):
    with pytest.raises(ValueError, match=match):
        classify_surface(faces, count)


@pytest.mark.host_validation
def test_body_kinds_and_scene_formulation():
    sheet = mw.Structure(mw.Box(size=(1, 1, 0)), mw.Material.pec())
    disk = mw.Structure(mw.Cylinder(radius=1, height=0), mw.Material.pec())
    dielectric = mw.Structure(mw.Sphere(radius=1), mw.Material(eps_r=2))
    assert body_kind(sheet) is BodyKind.PEC_OPEN and body_kind(disk) is BodyKind.PEC_OPEN
    assert body_kind(dielectric) is BodyKind.DIELECTRIC_CLOSED
    assert body_kind(mw.Structure(mw.Box(size=(1, 1, 1)), mw.Material.pec())) is BodyKind.PEC_CLOSED
    assert body_kind(mw.Structure(mw.Sphere(radius=1), mw.Material.pec())) is BodyKind.PEC_CLOSED
    assert scene_formulation([sheet, disk]) == 'efie_open_pec'
    assert scene_formulation([dielectric]) == 'pmchwt'
    with pytest.raises(NotImplementedError, match='multiregion'):
        scene_formulation([sheet, dielectric])
    with pytest.raises(NotImplementedError, match='combined-field'):
        scene_formulation([mw.Structure(mw.Sphere(radius=1), mw.Material.pec())])
    with pytest.raises(NotImplementedError, match='impedance sheets'):
        scene_formulation([mw.Structure(mw.Box(size=(1, 1, 0)), mw.Medium2D(sigma_s=1e-3))])
    with pytest.raises(ValueError, match='dielectric sheet'):
        body_kind(mw.Structure(mw.Box(size=(1, 1, 0)), mw.Material(eps_r=2)))
    with pytest.raises(ValueError, match='exactly one zero'):
        body_kind(mw.Structure(mw.Box(size=(1, 0, 0)), mw.Material.pec()))
    closed_shell = mw.Mesh(torch.tensor([(0., 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]),
                           torch.tensor([(0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)]),
                           fill_mode='surface', recenter=False)
    assert body_kind(mw.Structure(closed_shell, mw.Material.pec())) is BodyKind.PEC_CLOSED


@cuda
def test_open_basis_matches_closed_rwg_on_a_closed_surface():
    from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh
    from witwin.maxwell.mom.open_surface import OpenRWGBasis, OpenSurfaceMesh
    mesh = sphere_mesh(0.1, 1)
    closed, open_basis = RWGBasis(mesh), OpenRWGBasis(OpenSurfaceMesh(mesh.vertices, mesh.faces))
    for name in ('edges', 'face_edges', 'face_signs', 'plus_faces', 'minus_faces', 'plus_vertices',
                 'minus_vertices', 'edge_lengths'):
        assert torch.equal(getattr(closed, name), getattr(open_basis, name)), name
    assert torch.equal(closed.divergence, open_basis.divergence)
    torch.testing.assert_close(closed.mass_matrix(), open_basis.mass_matrix(), rtol=0, atol=0)


@cuda
def test_open_plate_basis_has_zero_boundary_flux_and_divergence_theorem():
    from witwin.maxwell.mom.open_surface import OpenRWGBasis, disk_mesh, rectangular_plate_mesh
    from witwin.maxwell.mom.operator import local_geometry
    for mesh in (rectangular_plate_mesh((0.3, 0.2), (5, 4)), disk_mesh(0.1, 4)):
        basis = OpenRWGBasis(mesh)
        assert basis.n_edges == mesh.topology.interior_edges
        boundary = mesh.topology.kind != INTERIOR
        slots = boundary[mesh.topology.face_edges]
        assert bool((basis.face_signs[slots] == 0).all()) and bool((basis.face_signs[~slots] != 0).all())
        # Normal flux of every basis function through every boundary edge vanishes.
        a, b = mesh.vertices[mesh.topology.edges[boundary, 0]], mesh.vertices[mesh.topology.edges[boundary, 1]]
        faces = mesh.topology.incidence_faces[boundary, 0]
        _, triangles, slopes = local_geometry(basis)
        values = slopes[faces, :, None]*((a+b)/2)[:, None]-slopes[faces, :, None]*triangles[faces]
        tangent = (b-a)/torch.linalg.vector_norm(b-a, dim=-1, keepdim=True)
        normal = torch.linalg.cross(tangent, mesh.normals[faces])
        flux = (values*normal[:, None]).sum(-1).abs().amax()
        assert float(flux) < 1e-12
        # Total charge of each RWG vanishes (divergence integrates to zero).
        total = (basis.divergence*mesh.areas[:, None]).sum(0)
        assert float(total.abs().max()) < 1e-12
        assert mesh.validation_report['boundary_edges'] == mesh.topology.boundary_edges


@cuda
def test_open_mesh_rejects_junctions_and_refines_conformingly():
    from witwin.maxwell.mom.open_surface import OpenSurfaceMesh, rectangular_plate_mesh
    vertices = torch.tensor([(0., 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)], device='cuda',
                            dtype=torch.float64)
    with pytest.raises(NotImplementedError, match='junction'):
        OpenSurfaceMesh(vertices, torch.tensor([(0, 1, 2), (1, 0, 3), (0, 1, 4)], device='cuda'))
    plate = rectangular_plate_mesh((1.0, 1.0), (2, 2))
    fine = plate.refine(2)
    assert len(fine.faces) == 16*len(plate.faces)
    assert fine.topology.boundary_edges == 4*plate.topology.boundary_edges
    torch.testing.assert_close(fine.areas.sum(), plate.areas.sum())


@pytest.mark.host_validation
def test_pmchwt_scope_rejects_sheet_bodies_by_formulation():
    from witwin.maxwell.mom.physics import validate_scene_structure
    scene = mw.Scene(structures=[mw.Structure(mw.Box(size=(1, 1, 0)), mw.Material.pec())],
                     sources=[mw.PlaneWave()], device='cpu')
    with pytest.raises(ValueError, match="'efie_open_pec' formulation"):
        validate_scene_structure(scene, 1e9)
    assert len(validate_scene_structure(scene, 1e9, formulation='efie_open_pec')) == 1
