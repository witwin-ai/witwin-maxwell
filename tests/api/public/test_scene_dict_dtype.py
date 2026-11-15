"""Scene dictionaries rebuild every geometry tensor bit for bit: dtype and values, through JSON text.

The corpus-wide round trip (gate scenes, datagen, benchmark antennas) is test_scene_dict_corpus.py.
"""

from __future__ import annotations

import inspect
import json

import torch
import witwin.maxwell as mw
from witwin.core import GeometryBase
from witwin.core.geometry.base import _rotation_quaternion

from benchmark.scenes.antenna.printed_dipole import printed_dipole_scene


FIELDS = ("structures", "thin_wires", "sources", "ports", "lumped_elements", "monitors",
          "grid", "boundary", "domain", "subpixel", "background")


def _f64(*values):
    return torch.tensor(values, dtype=torch.float64)


def _pairs(a, b, path, seen):
    """Yield (path, original, restored) for every tensor and geometry float reachable from a."""
    if isinstance(a, torch.Tensor):
        yield path, a, b
    elif isinstance(a, GeometryBase):
        assert type(a) is type(b), path
        for name in inspect.signature(type(a)).parameters:
            if name != "device" and hasattr(a, name):  # ComplexPolySlab keeps its loops as vertices
                left, right = getattr(a, name), getattr(b, name)
                if name == "rotation" and isinstance(left, torch.Tensor):
                    left, right = _rotation_quaternion(left), _rotation_quaternion(right)
                yield from _pairs(left, right, f"{path}.{name}", seen)
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b), path
        for index, (left, right) in enumerate(zip(a, b)):
            yield from _pairs(left, right, f"{path}[{index}]", seen)
    elif isinstance(a, dict):
        for key in a:
            yield from _pairs(a[key], b[key], f"{path}[{key!r}]", seen)
    elif type(a).__module__.startswith("witwin") and hasattr(a, "__dict__") and id(a) not in seen:
        seen.add(id(a))
        assert type(a) is type(b), path
        for key, item in vars(a).items():
            if not key.endswith("_cache"):
                yield from _pairs(item, getattr(b, key), f"{path}.{key}", seen)
    elif isinstance(a, float):
        assert a == b, path


def _tensor_pairs(scene, restored):
    seen = set()
    for field in FIELDS:
        yield from _pairs(getattr(scene, field), getattr(restored, field), field, seen)


def _assert_bit_exact(scene, restored):
    count = 0
    for path, expected, actual in _tensor_pairs(scene, restored):
        assert isinstance(actual, torch.Tensor), path
        assert actual.dtype == expected.dtype, f"{path}: {expected.dtype} -> {actual.dtype}"
        assert torch.equal(actual.cpu(), expected.detach().cpu()), path
        count += 1
    return count


def _round_trip(scene):
    text = json.dumps(scene.to_dict(), allow_nan=False)
    restored = mw.Scene.from_dict(json.loads(text))
    tensors = _assert_bit_exact(scene, restored)
    assert json.dumps(restored.to_dict()) == text
    arrays, again = {}, {}
    with_arrays = scene.to_dict(arrays=arrays)
    from_arrays = mw.Scene.from_dict(json.loads(json.dumps(with_arrays)), arrays=arrays)
    _assert_bit_exact(scene, from_arrays)
    assert from_arrays.to_dict(arrays=again) == with_arrays and again.keys() == arrays.keys()
    for key, tensor in arrays.items():
        assert again[key].dtype == tensor.dtype and torch.equal(again[key].cpu(), tensor.cpu()), key
    return restored, tensors


def _port_scene():
    surface = mw.Box(position=_f64(0., 0., -.0175), size=_f64(.03747405725, .0013879280462962963, 0.))
    aperture = mw.Box(position=_f64(0., 0., -.0175), size=_f64(1 / 300, 1 / 700, 0.))
    return mw.Scene(
        domain=mw.Domain(((-.03, .03),) * 3), grid=mw.GridSpec.uniform(.005), structures=[mw.Structure(mw.Box(position=_f64(0., 0., -.0007), size=_f64(.06245676208333333, .01, .001)),
                                 mw.Material(eps_r=2.), name="positive"),
                    mw.Structure(mw.Box(position=(0., 0., .0007), size=(.01, .01, .001)),
                                 mw.Material(eps_r=2.), name="negative")],
        ports=[mw.LumpedPort("feed", positive=(0., 0., -.015), negative=(0., 0., -.02),
                             voltage_path=mw.AxisPath("z"), current_surface=surface,
                             voltage_aperture=aperture)],
    )


def test_nested_port_geometry_keeps_dtype():
    scene = _port_scene()
    restored, tensors = _round_trip(scene)
    assert tensors > 10
    feed, = restored.ports
    assert feed.current_surface.size.dtype == feed.voltage_aperture.position.dtype == torch.float64
    dipole = printed_dipole_scene()
    terminal, = _round_trip(dipole)[0].ports
    for name in ("_declared_current_surface", "voltage_aperture"):
        expected, actual = getattr(dipole.ports[0], name), getattr(terminal, name)
        for field in ("position", "size"):
            assert getattr(actual, field).dtype == getattr(expected, field).dtype == torch.float64
            assert torch.equal(getattr(actual, field), getattr(expected, field))
    assert restored.structures[0].geometry.size.tolist()[0] == .06245676208333333
    doc = scene.to_dict()
    assert doc["ports"][0]["current_surface"]["size"] == {
        "type": "Tensor", "dtype": "float64", "shape": [3], "values": [.03747405725, .0013879280462962963, 0.]}
    assert doc["structures"][1]["geometry"]["size"]["dtype"] == "float32"


def test_mesh_dtypes_round_trip():
    vertices, faces = mw.Box(size=(1 / 3 * .01, .007, .009)).to_mesh()
    vertices64, faces32 = vertices.to(torch.float64), faces.to(torch.int32)
    big_vertices, big_faces = mw.Sphere(radius=.004).to_mesh(segments=32)
    meshes = [
        mw.Mesh(vertices64, faces32, recenter=False),
        mw.Mesh(vertices64, faces32, recenter=False, fill_mode="solid", topology_diagnostics=False),
        mw.Mesh(vertices64.numpy(), faces32.numpy(), position=_f64(.001, 0., 1 / 3 * .01)),
        mw.Mesh(big_vertices.to(torch.float64), big_faces.to(torch.int32), recenter=False,
                fill_mode="solid", topology_diagnostics=False),
    ]
    assert big_vertices.numel() > 4096
    scene = mw.Scene(domain=mw.Domain(((-.03, .03),) * 3), grid=mw.GridSpec.uniform(.005), structures=[mw.Structure(mesh, mw.Material(eps_r=2.)) for mesh in meshes])
    restored, _ = _round_trip(scene)
    dtypes = [(m.geometry.vertices.dtype, m.geometry.faces.dtype) for m in restored.structures]
    assert dtypes == [(torch.float64, torch.int32)] * 2 + [(torch.float32, torch.int32), (torch.float64, torch.int32)]
    assert [m.geometry.is_watertight for m in restored.structures] == [m.is_watertight for m in meshes]


def test_complex_polyslab_round_trips():
    outer = [(-.01, -.01), (.01, -.01), (.01, .01), (-.01, .01)]
    hole = [(-.003, -.003), (.003, -.003), (.003, .003), (-.003, .003)]
    slab = mw.geometry.ComplexPolySlab(loops=[outer, hole], bounds=(-.001, .001))
    for geometry in (slab, mw.geometry.Union(mw.Box(), slab)):
        scene = mw.Scene(domain=mw.Domain(((-.03, .03),) * 3), grid=mw.GridSpec.uniform(.005),
                         structures=[mw.Structure(geometry, mw.Material(eps_r=2.))])
        restored, tensors = _round_trip(scene)
        assert tensors > 3
        rebuilt = restored.structures[0].geometry
        rebuilt = rebuilt if isinstance(rebuilt, mw.geometry.ComplexPolySlab) else rebuilt.geometries[1]
        assert rebuilt.loop_sizes == (4, 4) and torch.equal(rebuilt.vertices, slab.vertices)
