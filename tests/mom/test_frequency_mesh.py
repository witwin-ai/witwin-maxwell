"""Conforming frequency geometry and preserved midpoint CUDA behavior."""

import ast
from collections import Counter
import hashlib
import inspect
import math
import sys
import types

import pytest
import torch

from witwin.maxwell.mom import mesh as native

from tests.support.pinned_baselines import baseline_tree


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for numerical geometry")
ORIGINAL_REVISION = "f112e8526511ce9d3cacf5df8d36bd8ba093dad6"
ORIGINAL_SHA256 = "7b8ce062c65aec18b13184b5fd98bf848a6209f25c6461a85f79f854656e0456"


def seed_faces():
    # Inspect only integer incidence; no production geometry is evaluated on host.
    tree = ast.parse(inspect.getsource(native._icosahedron))
    return next(ast.literal_eval(node.value) for node in ast.walk(tree)
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "faces" for t in node.targets))


@pytest.mark.parametrize("frequency", [1, 2, 3, 5, 9, 16])
def test_host_incidence_is_closed_oriented_and_welded(frequency):
    ids, weights, faces = native._frequency_topology(seed_faces(), frequency)
    assert len(ids) == 10 * frequency**2 + 2
    assert len(faces) == 20 * frequency**2
    edge_counts = Counter()
    orientation = Counter()
    for face in faces:
        assert len(set(face)) == 3
        for a, b in zip(face, (*face[1:], face[0])):
            edge = tuple(sorted((a, b)))
            edge_counts[edge] += 1
            orientation[edge] += 1 if a < b else -1
    assert len(edge_counts) == 30 * frequency**2
    assert set(edge_counts.values()) == {2}
    assert set(orientation.values()) == {0}
    assert len(ids) - len(edge_counts) + len(faces) == 2
    assert all(sum(row) == frequency and min(row) >= 0 for row in weights)


@pytest.mark.parametrize("constructor", [native.sphere_mesh, native.ellipsoid_mesh, native.deformed_sphere_mesh])
@pytest.mark.parametrize("arguments", [
    {"frequency": 0}, {"frequency": -1}, {"frequency": 1.5}, {"frequency": True},
    {"frequency": "3"}, {"subdivisions": 0, "frequency": 1},
    {"subdivisions": 1, "frequency": 2}, {"subdivisions": False},
    {"subdivisions": -1}, {"subdivisions": 1.5},
])
def test_invalid_resolution_rejected_before_cuda_allocation(constructor, arguments, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid resolution must not reach CUDA geometry")
    monkeypatch.setattr(native, "_cuda", forbidden)
    with pytest.raises(ValueError):
        constructor(**arguments)


@pytest.mark.parametrize("constructor", [native.sphere_mesh, native.ellipsoid_mesh, native.deformed_sphere_mesh])
def test_frequency_has_no_cpu_numerical_fallback(constructor):
    with pytest.raises(ValueError, match="CUDA"):
        constructor(frequency=3, device="cpu")


@cuda
@pytest.mark.parametrize("frequency", [1, 2, 3, 5])
def test_frequency_sphere_has_actual_valid_cuda_geometry(frequency):
    radius, center = .37, (.13, -.2, .09)
    mesh = native.sphere_mesh(radius, frequency=frequency, center=center)
    assert mesh.vertices.is_cuda and mesh.faces.is_cuda
    report = mesh.validation_report
    assert (report.vertices, report.triangles, report.edges) == (10*frequency**2+2, 20*frequency**2, 30*frequency**2)
    assert report.connected and report.closed and report.orientable and report.self_intersections == 0
    local = mesh.vertices - mesh.vertices.new_tensor(center)
    torch.testing.assert_close(local.norm(dim=-1), torch.full_like(local[:, 0], radius), rtol=2e-15, atol=2e-15)
    assert bool(((mesh.centroids-mesh.vertices.new_tensor(center))*mesh.normals).sum(-1).gt(0).all())
    assert bool(mesh.areas.gt(0).all())


@cuda
@pytest.mark.parametrize("frequency", [2, 3, 5])
def test_ellipsoid_and_harmonic_vertices_remain_on_continuous_locus(frequency):
    center = (.11, -.07, .03)
    axes = (.2, .31, .43)
    ellipsoid = native.ellipsoid_mesh(axes, frequency=frequency, center=center)
    local = ellipsoid.vertices - ellipsoid.vertices.new_tensor(center)
    torch.testing.assert_close((local/local.new_tensor(axes)).square().sum(-1), torch.ones_like(local[:, 0]), rtol=3e-15, atol=3e-15)
    radius, c20, c32 = .25, .15082377802580596, -.10666592493653297
    harmonic = native.deformed_sphere_mesh(radius, frequency=frequency, center=center, harmonics={(2, 0): c20, (3, 2): c32})
    local = harmonic.vertices-harmonic.vertices.new_tensor(center)
    radial = local.norm(dim=-1)
    x, y, z = (local/radial[:, None]).unbind(-1)
    # Independent closed polynomial values of the two frozen real modes.
    y20 = math.sqrt(5)/(4*math.sqrt(math.pi))*(3*z.square()-1)
    y32 = math.sqrt(105)/(4*math.sqrt(math.pi))*z*(x.square()-y.square())
    torch.testing.assert_close(radial, radius*(1+c20*y20+c32*y32), rtol=3e-15, atol=3e-15)
    assert ellipsoid.validation_report.self_intersections == harmonic.validation_report.self_intersections == 0


@cuda
@pytest.mark.parametrize("family", ["sphere", "ellipsoid", "harmonic"])
def test_frequency_translation_and_si_scale_preserve_topology(family):
    def build(scale, center):
        kwargs = {"frequency": 3, "center": center}
        if family == "sphere":
            return native.sphere_mesh(.25*scale, **kwargs)
        if family == "ellipsoid":
            return native.ellipsoid_mesh(tuple(scale*x for x in (.2, .25, .3)), **kwargs)
        return native.deformed_sphere_mesh(.25*scale, harmonics={(2, 0): .1, (3, 2): -.07}, **kwargs)
    unit = build(1., (0., 0., 0.))
    scale = 1e-6
    offset = unit.vertices.new_tensor((.2e-6, -.3e-6, .1e-6))
    moved = build(scale, offset)
    assert torch.equal(unit.faces, moved.faces)
    torch.testing.assert_close(moved.vertices, unit.vertices*scale+offset, rtol=3e-15, atol=1e-21)
    torch.testing.assert_close(moved.areas, unit.areas*scale**2, rtol=2e-14, atol=1e-28)


@cuda
@pytest.mark.parametrize("arguments", [
    {"center": (0.,)}, {"center": (0., float("nan"), 0.)},
    pytest.param({"radius": 0.}, marks=pytest.mark.host_validation),
    pytest.param({"radius": float("inf")}, marks=pytest.mark.host_validation),
    pytest.param({"dtype": torch.int64}, marks=pytest.mark.host_validation)])
def test_frequency_invalid_physical_inputs(arguments):
    with pytest.raises(ValueError):
        native.sphere_mesh(frequency=2, **arguments)


@cuda
@pytest.mark.parametrize("harmonics", [{(1, 2): .1}, {(True, 0): .1}, {(2, 0): float("nan")}, {(0, 0): -4.}])
def test_frequency_invalid_harmonics(harmonics):
    with pytest.raises(ValueError):
        native.deformed_sphere_mesh(frequency=2, harmonics=harmonics)


@pytest.fixture(scope="module")
def original_mesh_module(tmp_path_factory):
    tree = baseline_tree(ORIGINAL_REVISION, tmp_path_factory.mktemp("f112_baseline"))
    source = (tree / "witwin/maxwell/sie/mesh.py").read_bytes()
    assert hashlib.sha256(source).hexdigest() == ORIGINAL_SHA256
    name = "_original_f112_mesh_cuda_regression"
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, "<immutable-f112-mesh-source>", "exec"), module.__dict__)
    yield module
    del sys.modules[name]


@cuda
@pytest.mark.parametrize("level", [0, 1, 2])
def test_midpoint_sphere_bytes_equal_independent_f112_source(original_mesh_module, level):
    kwargs = {"radius": .37, "subdivisions": level, "center": (.13, -.2, .09)}
    expected = original_mesh_module.sphere_mesh(**kwargs)
    actual = native.sphere_mesh(**kwargs)
    assert torch.equal(actual.vertices, expected.vertices)
    assert torch.equal(actual.faces, expected.faces)


@cuda
def test_midpoint_defaults_and_other_families_equal_f112_bytes(original_mesh_module):
    for name, kwargs in [
        ("sphere_mesh", {}),
        ("ellipsoid_mesh", {"axes": (.2, .31, .43), "center": (.13, -.2, .09)}),
        ("deformed_sphere_mesh", {"radius": .25, "harmonics": {(2, 0): .15, (3, 2): -.1}, "center": (.13, -.2, .09)}),
    ]:
        actual, expected = getattr(native, name)(**kwargs), getattr(original_mesh_module, name)(**kwargs)
        assert torch.equal(actual.vertices, expected.vertices)
        assert torch.equal(actual.faces, expected.faces)


@cuda
def test_frequency_four_is_not_the_midpoint_level_two_sampling():
    frequency = native.sphere_mesh(frequency=4)
    midpoint = native.sphere_mesh(subdivisions=2)
    assert len(frequency.faces) == len(midpoint.faces)
    distances = torch.cdist(frequency.vertices, midpoint.vertices).min(-1).values
    assert float(distances.max()) > 1e-3


@cuda
def test_three_frequency_surface_area_errors_decrease():
    errors = [abs(float(native.sphere_mesh(frequency=n).areas.sum())-4*math.pi) for n in (2, 3, 5)]
    assert errors[2] < errors[1] < errors[0]
