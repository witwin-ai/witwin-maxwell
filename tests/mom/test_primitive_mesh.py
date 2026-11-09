"""Faceted primitive surface meshes: topology, exact geometry, grading and scope."""

import math
from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from witwin.core.geometry.base import _rotation_quaternion
from witwin.core.math import quat_to_rotation_matrix
from witwin.maxwell.mom import primitive_mesh as pm
from witwin.maxwell.constants import C_0
from witwin.maxwell.mom.mesh import _has_intersections, _topology
from witwin.maxwell.mom.runtime import SurfaceData


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required for SurfaceMesh validation')
CPU = dict(device='cpu', dtype=torch.float64)
GRADING = pm.EdgeGrading(min_size=.01, ratio=1.5)
EQUILATERAL = [(math.cos(a), math.sin(a)) for a in (math.pi/2, 7*math.pi/6, 11*math.pi/6)]


def surface_measures(vertices, faces):
    triangles = vertices[faces]
    cross = torch.linalg.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    shifted = triangles-vertices.mean(0)
    volume = float((shifted[:, 0]*torch.linalg.cross(shifted[:, 1], shifted[:, 2])).sum()/6)
    return float(torch.linalg.vector_norm(cross, dim=-1).sum()/2), volume


def assert_closed_surface(vertices, faces):
    """The same predicates as SurfaceMesh validation, evaluated on the host tensors."""
    _, incidence = _topology(faces, len(vertices))
    assert len(vertices)-len(incidence)+len(faces) == 2
    area, volume = surface_measures(vertices, faces)
    assert volume > 0 and area > 0
    # The intersection predicate uses a CUDA box tree; topology and measures stay host checks.
    if torch.cuda.is_available():
        assert not _has_intersections(vertices[faces].cuda())
    quality = pm.mesh_quality(vertices, faces)
    assert quality['min_angle_deg'] > 0
    return area, volume, quality


BUILDERS = {
    'box': lambda g: pm.box_arrays((1., .6, .4), .1, grading=g, **CPU),
    'cube': lambda g: pm.box_arrays((.5, .5, .5), .1, grading=g, **CPU),
    'slab': lambda g: pm.box_arrays((1., 1., .02), .1, grading=g, **CPU),
    'rounded_box': lambda g: pm.rounded_box_arrays((1., .6, .4), .05, .1, grading=g, **CPU),
    'rounded_sphere_limit': lambda g: pm.rounded_box_arrays((.2, .2, .2), .1, .1, grading=g, **CPU),
    'cylinder': lambda g: pm.cylinder_arrays(.3, 1., .1, grading=g, **CPU),
    'cone': lambda g: pm.cone_arrays(.3, 1., .1, grading=g, **CPU),
    'triangular_prism': lambda g: pm.prism_arrays(EQUILATERAL, .5, .1, grading=g, **CPU),
    'scalene_prism': lambda g: pm.prism_arrays([(0, 0), (1, 0), (.3, .8)], .5, .1, grading=g, **CPU),
    'hexagonal_prism': lambda g: pm.prism_arrays([(math.cos(j*math.pi/3), math.sin(j*math.pi/3)) for j in range(6)],
                                                 .5, .1, grading=g, **CPU),
}


@pytest.mark.parametrize('name', sorted(BUILDERS))
@pytest.mark.parametrize('graded', [False, True])
def test_builders_are_closed_oriented_manifold_and_embedded(name, graded):
    """Embedding (no self-intersection) is checked when CUDA is available."""
    vertices, faces = BUILDERS[name](GRADING if graded else None)
    assert vertices.dtype == torch.float64 and faces.dtype == torch.long
    assert_closed_surface(vertices, faces)


def test_graded_nodes_follow_the_linear_size_law():
    nodes = pm.graded_nodes(1., .1)
    assert len(nodes) == 11 and nodes[0] == 0 and nodes[-1] == 1
    nodes = pm.graded_nodes(1., .1, GRADING)
    steps = [b-a for a, b in zip(nodes, nodes[1:])]
    assert all(step > 0 for step in steps)
    assert all(abs(a-b) < 1e-15 for a, b in zip(steps, steps[::-1]))
    assert GRADING.min_size/GRADING.ratio < steps[0] <= GRADING.min_size*(1+1e-12)
    assert max(steps) <= .1*(1+1e-12)
    growth = [b/a for a, b in zip(steps, steps[1:len(steps)//2])]
    # Geometric growth, with one transition element joining the uniform size.
    assert sorted(growth)[-2] <= GRADING.ratio*(1+1e-9) and max(growth) <= 1.07*GRADING.ratio
    for x, step in zip(nodes, steps):
        local = pm.graded_size(min(x, 1-x-step), .1, GRADING)
        assert step <= 1.07*local
    one_sided = pm.graded_nodes(1., .1, GRADING, start=False, end=True)
    assert one_sided[1]-one_sided[0] > .09 and one_sided[-1]-one_sided[-2] <= GRADING.min_size*(1+1e-12)
    # A minimum size above the target length leaves uniform meshing unchanged.
    assert pm.graded_nodes(1., .1, pm.EdgeGrading(.2)) == pm.graded_nodes(1., .1)


def test_box_has_exact_face_planes_edges_and_measures():
    size = (1., .6, .4)
    half = torch.tensor(size, dtype=torch.float64)/2
    for grading in (None, GRADING):
        vertices, faces = pm.box_arrays(size, .1, grading=grading, **CPU)
        on_plane = (vertices.abs() == half)
        assert bool(on_plane.any(-1).all())
        triangle_planes = on_plane[faces].all(1)
        assert bool(triangle_planes.any(-1).all())
        signs = torch.sign(vertices[faces])
        plane_axis = triangle_planes.float().argmax(-1)
        sign = signs[torch.arange(len(faces)), 0, plane_axis]
        assert bool((signs[torch.arange(len(faces)), :, plane_axis] == sign[:, None]).all())
        triangles = vertices[faces]
        normal = torch.linalg.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        assert bool((normal[torch.arange(len(faces)), plane_axis]*sign > 0).all())
        # Mesh edges lying on box edges tile all twelve box edges exactly.
        segments = torch.cat((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])).sort(-1).values.unique(dim=0)
        both = on_plane[segments].all(1)
        sharp = both.sum(-1) >= 2
        lengths = torch.linalg.vector_norm(vertices[segments[:, 0]]-vertices[segments[:, 1]], dim=-1)
        assert math.isclose(float(lengths[sharp].sum()), 4*sum(size), rel_tol=1e-14)
        corners = (vertices.abs() == half).all(-1)
        assert int(corners.sum()) == 8
        area, volume = surface_measures(vertices, faces)
        assert math.isclose(area, 2*(1*.6+.6*.4+.4*1), rel_tol=1e-13)
        assert math.isclose(volume, .24, rel_tol=1e-13)


def test_uniform_cube_quality_and_mirror_symmetry():
    vertices, faces = pm.box_arrays((.4, .4, .4), .1, **CPU)
    quality = pm.mesh_quality(vertices, faces)
    assert math.isclose(quality['min_angle_deg'], 45, abs_tol=1e-9)
    assert math.isclose(quality['max_aspect_ratio'], math.sqrt(2)*(2+math.sqrt(2))/(2*math.sqrt(3)), rel_tol=1e-9)
    # Centre-pointing diagonals make the triangulation invariant under x -> -x.
    triangles = vertices[faces]
    key = lambda t: sorted(tuple(round(float(c), 12) for c in p) for p in t)
    original = sorted(key(t) for t in triangles)
    mirrored = sorted(key(t*torch.tensor([-1., 1., 1.], dtype=torch.float64)) for t in triangles)
    assert original == mirrored


@pytest.mark.parametrize('name', ['box', 'cylinder', 'cone', 'triangular_prism'])
def test_grading_refines_edges_with_bounded_anisotropy(name):
    uniform = pm.mesh_quality(*BUILDERS[name](None))
    graded = pm.mesh_quality(*BUILDERS[name](GRADING))
    assert graded['triangles'] > uniform['triangles']
    assert graded['min_edge_length'] <= GRADING.min_size*(1+1e-9) < uniform['min_edge_length']
    assert max(graded['max_edge_length'], uniform['max_edge_length']) <= 1.7*.1
    # Perpendicular grading makes edge-adjacent elements anisotropic by about h/h_min.
    assert graded['max_aspect_ratio'] < 1.5*.1/GRADING.min_size
    assert uniform['max_aspect_ratio'] < 3 and uniform['min_angle_deg'] > 14


def test_rotated_core_box_matches_signed_distance_and_rotation():
    rotation = (.3, -.7, 1.1)
    box = mw.Box(position=(.2, -.1, .05), size=(.4, .3, .2), rotation=rotation)
    config = pm.SurfaceMeshConfig(elements_per_wavelength=None, max_edge_length=.05)
    vertices, faces = pm.primitive_arrays(box, config, **CPU)
    local, local_faces = pm.box_arrays((.4, .3, .2), .05, **CPU)
    matrix = quat_to_rotation_matrix(_rotation_quaternion(box.rotation.double()))
    expected = local@matrix.T+box.position.double()
    # Core sizes are float32-authored; 1e-7 m covers their rounding.
    assert torch.allclose(vertices, expected, rtol=0, atol=1e-7)
    assert torch.equal(faces, local_faces)
    triangles = vertices[faces]
    for points in (vertices, triangles.mean(1)):
        distance = box.signed_distance(*points.T)
        assert float(distance.abs().max()) < 1e-6
    rotated, canonical = surface_measures(vertices, faces), surface_measures(local, local_faces)
    assert all(math.isclose(a, b, rel_tol=1e-6) for a, b in zip(rotated, canonical))
    edges = lambda v, f: torch.linalg.vector_norm(v[f]-v[f].roll(-1, 1), dim=-1).flatten().sort().values
    assert torch.allclose(edges(vertices, faces), edges(local, local_faces), rtol=1e-6)


@pytest.mark.parametrize('kind', ['cylinder', 'cone', 'prism'])
@pytest.mark.parametrize('axis', ['x', 'y', 'z'])
def test_revolved_and_extruded_core_primitives_follow_core_conventions(kind, axis):
    rotation = (0., .4, -.2)
    if kind == 'cylinder':
        geometry = mw.Cylinder(position=(.1, .2, -.3), radius=.15, height=.4, axis=axis, rotation=rotation)
    elif kind == 'cone':
        geometry = mw.Cone(position=(.1, .2, -.3), radius=.15, height=.4, axis=axis, rotation=rotation)
    else:
        geometry = mw.Prism(position=(.1, .2, -.3), radius=.15, height=.4, num_sides=3, axis=axis, rotation=rotation)
    config = pm.SurfaceMeshConfig(elements_per_wavelength=None, max_edge_length=.04)
    vertices, faces = pm.primitive_arrays(geometry, config, **CPU)
    assert_closed_surface(vertices, faces)
    # Every vertex lies on the exact primitive surface used by the volume solvers.
    assert float(geometry.signed_distance(*vertices.T).abs().max()) < 1e-6
    if kind == 'cone':
        apex = (vertices-geometry.position.double()).norm(dim=-1).argmin()
        assert float((vertices[apex]-geometry.position.double()).norm()) < 1e-7


@pytest.mark.parametrize('kind', ['cylinder', 'cone'])
def test_curved_bodies_converge_in_area_and_volume(kind):
    radius, height = 1., 1.5
    if kind == 'cylinder':
        exact = (2*math.pi*radius*height+2*math.pi*radius**2, math.pi*radius**2*height)
        build = pm.cylinder_arrays
    else:
        exact = (math.pi*radius*math.hypot(radius, height)+math.pi*radius**2, math.pi*radius**2*height/3)
        build = pm.cone_arrays
    errors = []
    for h in (.2, .1, .05):
        area, volume = surface_measures(*build(radius, height, h, **CPU))
        errors.append((abs(area-exact[0])/exact[0], abs(volume-exact[1])/exact[1]))
    for (a0, v0), (a1, v1) in zip(errors, errors[1:]):
        assert math.log2(a0/a1) > 1.7 and math.log2(v0/v1) > 1.7
    assert errors[-1][0] < 2e-3 and errors[-1][1] < 2e-3


def test_rounded_box_is_continuous_as_radius_vanishes():
    size, h = (1., .6, .4), .1
    sharp_vertices, sharp_faces = pm.box_arrays(size, h, **CPU)
    sharp = surface_measures(sharp_vertices, sharp_faces)
    zero = pm.rounded_box_arrays(size, 0., h, **CPU)
    assert torch.equal(zero[0], sharp_vertices) and torch.equal(zero[1], sharp_faces)
    half = torch.tensor(size, dtype=torch.float64)/2
    previous = None
    for radius in (.05, 1e-2, 1e-3, 1e-4):
        vertices, faces = pm.rounded_box_arrays(size, radius, h, **CPU)
        area, volume = surface_measures(vertices, faces)
        # Inscribed arcs: the faceted measures sit just below the smooth Steiner values.
        assert area <= pm.rounded_box_area(size, radius)+1e-14 <= area+4*radius
        assert volume <= pm.rounded_box_volume(size, radius)+1e-14
        assert abs(area-sharp[0]) <= 8*radius*sum(size) and abs(volume-sharp[1]) <= 4*radius*sum(size)
        outside = ((vertices.abs()-half).clamp_min(0)).norm(dim=-1)
        inside = (half-vertices.abs()).amin(-1).clamp_min(0)
        assert float((outside+inside).max()) <= (math.sqrt(3)-1)*radius*(1+1e-12)
        if previous is not None:
            assert abs(area-sharp[0]) < abs(previous-sharp[0])
        previous = area
    # Fine rounded meshes converge to the smooth surface measures.
    area, volume = surface_measures(*pm.rounded_box_arrays(size, .1, .02, **CPU))
    assert math.isclose(area, pm.rounded_box_area(size, .1), rel_tol=2e-3)
    assert math.isclose(volume, pm.rounded_box_volume(size, .1), rel_tol=2e-3)


def test_wavelength_sizing_uses_the_largest_medium_wavenumber():
    exterior, interior = SimpleNamespace(k=2*math.pi), SimpleNamespace(k=4*math.pi+1j)
    config = pm.SurfaceMeshConfig(elements_per_wavelength=10)
    assert math.isclose(config.edge_length((exterior, interior)), 2*math.pi/(abs(4*math.pi+1j)*10))
    assert pm.SurfaceMeshConfig(elements_per_wavelength=10, max_edge_length=.01).edge_length((exterior,)) == .01
    explicit = pm.SurfaceMeshConfig(elements_per_wavelength=None, max_edge_length=.02)
    assert explicit.edge_length() == .02
    with pytest.raises(ValueError, match='requires the body media'):
        config.edge_length()
    tensor = SimpleNamespace(frequency=3e8, epsilon_r=torch.diag(torch.tensor([1., 4., 9.], dtype=torch.complex128)),
                             mu_r=torch.eye(3, dtype=torch.complex128))
    free = 2*math.pi*3e8/C_0
    assert math.isclose(pm.medium_wavenumber(tensor), 3*free, rel_tol=1e-9)


def test_default_numerical_identity_is_unchanged():
    from witwin.maxwell.mom.runtime import _numerical_identity
    assert 'surface_mesh' not in _numerical_identity(mw.MoMConfig())
    config = mw.MoMConfig(surface_mesh=mw.SurfaceMeshConfig(edge_grading=mw.EdgeGrading(.01)))
    assert _numerical_identity(config)['surface_mesh']['edge_grading'] == {'min_size': .01, 'ratio': 1.5}
    from witwin.maxwell.mom.runtime import _restore_surface_mesh
    assert _restore_surface_mesh(config.surface_mesh.payload()) == config.surface_mesh


@pytest.mark.parametrize('arguments, match', [
    (dict(elements_per_wavelength=None), 'requires elements_per_wavelength'),
    (dict(elements_per_wavelength=0), 'positive finite'),
    (dict(elements_per_wavelength=True), 'positive finite'),
    (dict(max_edge_length=-1.), 'positive finite'),
    (dict(max_arc_angle=2.), 'pi/2'),
    (dict(min_edge_divisions=0), 'positive integer'),
    (dict(max_triangles=1.5), 'positive integer'),
])
def test_invalid_mesh_configuration_fails_closed(arguments, match):
    with pytest.raises(ValueError, match=match):
        pm.SurfaceMeshConfig(**arguments)


def test_invalid_geometry_and_grading_fail_closed():
    with pytest.raises(ValueError, match='exceed one'):
        pm.EdgeGrading(.01, ratio=1.)
    with pytest.raises(ValueError, match='positive'):
        pm.EdgeGrading(0.)
    with pytest.raises(TypeError):
        pm.SurfaceMeshConfig(edge_grading=(.01, 1.5))
    with pytest.raises(TypeError):
        mw.MoMConfig(surface_mesh={'elements_per_wavelength': 10})
    with pytest.raises(ValueError, match='positive and finite'):
        pm.box_arrays((1., 0., 1.), .1, **CPU)
    with pytest.raises(ValueError, match='positive and finite'):
        pm.cylinder_arrays(float('nan'), 1., .1, **CPU)
    with pytest.raises(ValueError, match=r'\[0, min\(size\)/2\]'):
        pm.rounded_box_arrays((1., 1., .2), .11, .1, **CPU)
    with pytest.raises(ValueError, match='strictly convex'):
        pm.prism_arrays([(0, 0), (1, 0), (.5, .1), (1, 1), (0, 1)], 1., .1, **CPU)
    with pytest.raises(ValueError, match='strictly convex'):
        pm.prism_arrays([(0, 0), (1, 0), (2, 0)], 1., .1, **CPU)
    with pytest.raises(ValueError, match='max_triangles'):
        pm.box_arrays((1., 1., 1.), .01, max_triangles=1000, **CPU)
    with pytest.raises(ValueError, match='not a faceted'):
        pm.primitive_arrays(mw.Sphere(radius=1.), pm.SurfaceMeshConfig(), **CPU)
    with pytest.raises(ValueError, match='slab thickness|Slab thickness'):
        pm.slab_mesh(1., 1., 2., edge_length=.1, device='cpu')


@pytest.mark.parametrize('geometry', [mw.Torus(major_radius=.2, minor_radius=.05), mw.Pyramid(base_size=.2, height=.2),
                                      mw.HollowBox(outer_size=(.2, .2, .2), inner_size=(.1, .1, .1))])
def test_unsupported_core_geometry_fails_closed(geometry):
    from witwin.maxwell.mom.runtime import _compile_basis
    scene = SimpleNamespace(device='cuda')
    with pytest.raises(ValueError, match=f'{type(geometry).__name__} is not supported'):
        _compile_basis(scene, geometry, mw.MoMConfig())


def test_builders_require_cuda_for_surface_mesh():
    with pytest.raises(ValueError, match='CUDA'):
        pm.box_mesh((1., 1., 1.), edge_length=.5, device='cpu')


@cuda
@pytest.mark.parametrize('builder', [
    lambda: pm.box_mesh((.2, .1, .05), edge_length=.02, grading=pm.EdgeGrading(.005)),
    lambda: pm.rounded_box_mesh((.2, .1, .05), .01, edge_length=.02),
    lambda: pm.slab_mesh(.2, .2, .005, edge_length=.02),
    lambda: pm.triangular_prism_mesh(.2, .1, edge_length=.02),
    lambda: pm.cylinder_mesh(.05, .2, edge_length=.02),
    lambda: pm.cone_mesh(.05, .2, edge_length=.02, center=(.1, 0., 0.)),
])
def test_cuda_builders_pass_surface_mesh_validation(builder):
    mesh = builder()
    assert mesh.device.type == 'cuda' and mesh.validation_report.self_intersections == 0
    mw.Mesh(mesh.vertices, mesh.faces, recenter=False, fill_mode='solid')


@cuda
def test_public_path_compiles_rotated_box_and_cylinder_bodies():
    from witwin.maxwell.mom.runtime import compile_surfaces
    scene = mw.Scene(structures=[
        mw.Structure(mw.Box(position=(-.15, 0., 0.), size=(.1, .08, .06), rotation=(.2, .3, .4)), mw.Material(eps_r=4.)),
        mw.Structure(mw.Cylinder(position=(.15, 0., 0.), radius=.04, height=.1, axis='x'), mw.Material(eps_r=2.))],
        sources=[mw.PlaneWave()])
    config = mw.MoMConfig(surface_mesh=mw.SurfaceMeshConfig(elements_per_wavelength=10))
    system, exterior = compile_surfaces(scene, 1e9, config)
    k = 2*math.pi*1e9/C_0*2
    for body in system.bodies:
        quality = pm.mesh_quality(body.mesh.vertices, body.mesh.faces)
        # Target h bounds axis-aligned edges; cell diagonals and ring chords reach ~1.7 h.
        assert quality['max_edge_length'] <= 1.7*2*math.pi/(k*10)


@cuda
def test_public_box_solve_round_trips_with_surface_mesh_identity():
    scene = mw.Scene(structures=[mw.Structure(mw.Box(size=(.06, .05, .04), rotation=(.1, .2, .3)),
                                              mw.Material(eps_r=3.))], sources=[mw.PlaneWave()])
    config = mw.MoMConfig(surface_mesh=mw.SurfaceMeshConfig(elements_per_wavelength=None, max_edge_length=.02))
    result = mw.Simulation.mom(scene, frequency=1e9, config=config).run()
    payload = result.surface_currents.snapshot()
    assert payload['surface_mesh'] == config.surface_mesh.payload()
    restored = SurfaceData.restore(payload, scene, 1e9)
    assert torch.equal(restored.scattered_coefficients, result.surface_currents.scattered_coefficients)
    payload.pop('surface_mesh')
    with pytest.raises(ValueError):
        SurfaceData.restore(payload, scene, 1e9)


@cuda
def test_point_dipole_clearance_applies_to_faceted_primitive_panels():
    from witwin.maxwell.mom.runtime import compile_surfaces
    config = mw.MoMConfig(surface_mesh=mw.SurfaceMeshConfig(elements_per_wavelength=None, max_edge_length=.02))
    box = mw.Box(size=(.06, .05, .04), rotation=(.1, .2, .3))

    def scene(position):
        return mw.Scene(structures=[mw.Structure(box, mw.Material(eps_r=3.))],
                        sources=[mw.PointDipole(position=position, polarization=(0., 0., 1.), profile='ideal')])
    # Panel radius is about 0.01 m here; the shared rule needs three panel radii.
    system, _ = compile_surfaces(scene((0., 0., .2)), 1e9, config)
    assert len(system.bodies) == 1
    with pytest.raises(ValueError, match='panel radii'):
        compile_surfaces(scene((0., 0., .04)), 1e9, config)
