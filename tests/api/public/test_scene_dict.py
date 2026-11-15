"""Acceptance contracts for ``witwin.scene/2`` scene dictionaries (CPU)."""

from dataclasses import fields
import json
import math

import numpy as np
import pytest
import torch
import witwin.maxwell as mw

from benchmark.core_path.fingerprint import canonical
from witwin.maxwell.compiler import compile_material_tensors
from witwin.maxwell.constants import C_0, EPSILON_0
from witwin.maxwell.io.scene_dict import SCHEMA
from witwin.maxwell.scene import prepare_scene


FREQUENCY = 1e9
WAVELENGTH = C_0 / FREQUENCY


def _monitors():
    return [
        mw.PointMonitor("point", (0., 0., .01)),
        mw.PlaneMonitor("plane", axis="x", position=.01, fields=("Ey", "Hz")),
        mw.FinitePlaneMonitor("finite", position=(0., 0., .01), size=(.02, .02, 0.)),
        mw.FluxMonitor("flux", axis="z", position=.015),
        mw.monitors.ClosedSurfaceMonitor.box("closed", size=(.04,) * 3),
        mw.monitors.AngleProjectionMonitor("angle", size=(.04,) * 3, theta=(.4,), phi=(.2,), radius=1.),
        mw.monitors.CartesianProjectionMonitor("cartesian", size=(.04,) * 3, x=(0.,), y=(0.,), distance=1.),
        mw.monitors.KSpaceProjectionMonitor("kspace", size=(.04,) * 3, ux=(0.,), uy=(0.,)),
        mw.monitors.PermittivityMonitor("eps", size=(.02,) * 3),
    ]


def _scene_a():
    lam = WAVELENGTH
    return mw.Scene(
        domain=mw.Domain(((-1.5 * lam, 1.5 * lam),) * 3),
        grid=mw.GridSpec.uniform(lam / 16), boundary=mw.BoundarySpec.pml(num_layers=8),
        structures=[
            mw.Structure(mw.Sphere(position=(.3 * lam, .1 * lam, .1 * lam), radius=.4 * lam), mw.Material(eps_r=4)),
            mw.Structure(mw.Box(position=(-.5 * lam, .5 * lam, .2 * lam),
                                size=(.44 * lam, .6 * lam, .36 * lam), rotation=(.21, -.32, .43)),
                         mw.Material(eps_r=6, sigma_e=.2 * 2 * math.pi * FREQUENCY * EPSILON_0)),
        ],
        sources=[mw.PointDipole(tuple(v * lam for v in (-.75, -.625, -.46875)),
                                polarization="Ez", profile="ideal",
                                source_time=mw.CW(FREQUENCY, amplitude=1.7, phase=.37))],
        monitors=[mw.monitors.PermittivityMonitor("eps", size=(2 * lam,) * 3)],
        subpixel=mw.SubpixelSpec(samples=4, averaging="polarized"),
    )


def _scene_b():
    nodes = np.array([-.03, -.025, -.02, -.015, -.01, -.005, 0., .005, .01, .015, .02, .026, .03])
    dielectric = mw.Material(eps_r=3.2, sigma_e=.01)
    foil = mw.media.CopperFoil(thickness=35e-6, frequency_range=(.8e9, 1.2e9), mode_count=4)
    polygon = [(-.01, -.01), (.01, -.01), (.01, .01), (-.01, .01)]
    hole = [(-.003, -.003), (.003, -.003), (.003, .003), (-.003, .003)]
    return mw.Scene(
        domain=mw.Domain(((-.03, .03),) * 3), grid=mw.GridSpec.custom(nodes, nodes, nodes),
        boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.),
        structures=[
            mw.Structure(mw.PolySlab(polygon, bounds=(-.01, -.005)), dielectric, name="board1"),
            mw.Structure(mw.Box(position=(0., 0., .0025), size=(.02, .02, .005)), dielectric, name="board2"),
            mw.Structure(mw.Box(position=(0., 0., -.005), size=(.01, .01, 0.)), foil, name="trace1"),
            mw.Structure(mw.PolySlab(polygon, bounds=(.005, .005)), foil, name="trace2"),
            mw.Structure(mw.geometry.PlanarPEC(polygon, holes=(hole,), coordinate=.015),
                         mw.Material.pec(), name="perforated"),
            mw.Structure(mw.geometry.SurfaceMesh([(-.01, -.01, .02), (.01, -.01, .02),
                                                  (.01, .01, .02), (-.01, .01, .02)],
                                                 [(0, 1, 2), (0, 2, 3)]), mw.Material.pec(), name="surface"),
        ],
        ports=[mw.LumpedPort("feed", positive=(0., 0., -.015), negative=(0., 0., -.02),
                             voltage_path=mw.AxisPath("z"),
                             current_surface=mw.Box(position=(0., 0., -.0175), size=(.01, .01, 0.)),
                             termination=mw.rf.SeriesRLC(r=50., l=1e-9, c=1e-12))],
        lumped_elements=[mw.rf.Resistor("resistor", (.01, 0., -.015), (.01, 0., -.02), 75.)],
        sources=[mw.PointDipole((-.015, 0., 0.), profile="ideal", source_time=mw.CW(FREQUENCY))],
        metadata={"case": "pcb", "nested": {"values": [1, True, None]}},
    )


def _scene_c():
    material = mw.Material(
        eps_r=2., drude_poles=(mw.DrudePole(plasma_frequency=.2e9, gamma=.1e9),),
        lorentz_poles=(mw.LorentzPole(delta_eps=.4, resonance_frequency=2e9, gamma=.1e9),),
    )
    return mw.Scene(
        domain=mw.Domain(((-.03, .03),) * 3), grid=mw.GridSpec.uniform(.005),
        boundary=mw.BoundarySpec.faces(x="bloch", y="bloch", z="pml", num_layers=2,
                                       strength=1., bloch_wavevector="auto"),
        structures=[mw.Structure(mw.Box(size=(.01, .01, .01)), material)],
        sources=[mw.PlaneWave(direction=(.3, .4, math.sqrt(.75)), polarization=(.8, -.6, 0.),
                              source_time=mw.CW(FREQUENCY, phase=.2),
                              injection=mw.TFSF.slab(axis="z", bounds=(-.02, .02)))],
        ports=[mw.LumpedPort("loaded", positive=(0., 0., .015), negative=(0., 0., .01),
                             voltage_path=mw.AxisPath("z"),
                             current_surface=mw.Box(position=(0., 0., .0125), size=(.01, .01, 0.)),
                             termination=mw.rf.ParallelRLC(r=100., l=2e-9, c=2e-12))],
    )


def _scene_d():
    return mw.Scene(
        domain=mw.Domain(((-.03, .03),) * 3), grid=mw.GridSpec.uniform(.005),
        boundary=mw.BoundarySpec.faces(x="pec", y="pmc", z_low="pec", z_high="pmc"),
        structures=[mw.Structure(mw.geometry.Difference(mw.Box(size=(.01,) * 3), mw.Sphere(radius=.003)),
                                 mw.Material(eps_r=2.5))],
        thin_wires=[mw.rf.ThinWire("dipole", [(-.015, 0., -.01), (-.015, 0., .01)],
                                   radius=.0001, conductor=mw.rf.WireConductor.pec(), snap="nearest")],
        sources=[mw.PointDipole((.015, 0., 0.), profile="gaussian", width=.005,
                                source_time=mw.GaussianPulse(FREQUENCY, fwidth=.2e9))],
        monitors=_monitors(),
    )


def _scene_e():
    lam = WAVELENGTH
    return mw.Scene(
        domain=mw.Domain(((-lam, lam),) * 3), grid=mw.GridSpec.uniform(lam / 16),
        boundary=mw.BoundarySpec.faces(
            default="pml", num_layers=4, z_high="mode_abc",
            mode_abc=mw.fdtd.ModeABC(order=2, frequency_range=(.8 * FREQUENCY, 1.2 * FREQUENCY)),
        ),
        structures=[mw.Structure(mw.Box(size=(.6 * lam, 1.2 * lam, 2 * lam)), mw.Material(eps_r=4))],
        sources=[mw.PointDipole((0., 0., -.5 * lam), polarization="Ey", profile="ideal",
                                source_time=mw.CW(FREQUENCY))],
    )


def _scene_f():
    """Every Scene collection witwin.scene/1 refused: regions, circuits, networks, symmetry, absorbers, ..."""
    design = mw.Box(position=(0., 0., 0.), size=(.02, .02, .01))
    circuit = mw.circuits.Circuit("load")
    node = circuit.node("n")
    circuit.add(mw.rf.Resistor("R", node, circuit.ground, 50.))
    circuit.bind_port("feed", positive=node, negative=circuit.ground)
    frequencies = torch.tensor([.5e9, 1e9, 2e9], dtype=torch.float64)
    network = mw.rf.NetworkData(frequencies=frequencies, s=torch.full((3, 1, 1), .1 + .05j, dtype=torch.complex128),
                                z0=50., port_names=("a",))
    return mw.Scene(
        domain=mw.Domain(((-.03, .03),) * 3), grid=mw.GridSpec.uniform(.005),
        boundary=mw.BoundarySpec.pml(num_layers=2),
        structures=[mw.Structure(mw.Box(position=(0., 0., -.012), size=(.01, .01, .002)),
                                 mw.Material(epsilon_tensor=mw.DiagonalTensor3(2., 3., 4.)), name="aniso")],
        material_regions=[mw.MaterialRegion("design", design, torch.rand((4, 4, 2), dtype=torch.float64),
                                            eps_bounds=(1., 4.))],
        ports=[mw.LumpedPort("feed", positive=(.015, 0., -.005), negative=(.015, 0., .005),
                             voltage_path=mw.AxisPath("z"),
                             current_surface=mw.Box(position=(.015, 0., 0.), size=(.005, .005, 0.))),
               mw.LumpedPort("net", positive=(-.015, 0., -.005), negative=(-.015, 0., .005),
                             voltage_path=mw.AxisPath("z"),
                             current_surface=mw.Box(position=(-.015, 0., 0.), size=(.005, .005, 0.)))],
        networks=[mw.rf.NetworkBlock("block", network=network, connections={1: "net"})],
        circuits=[circuit],
        internal_absorbers=[mw.fdtd.InternalAbsorber(position=(0., 0., .02), size=(math.inf, math.inf, 0.))],
        symmetry=("PEC", None, ("PMC", "high")),
        sources=[mw.PointDipole((0., .01, 0.), source_time=mw.CW(FREQUENCY))],
        metadata={"type": "user", "array": "not a reference", "values": (1, 2.5, math.inf)},
    )


def _json_round_trip(document, arrays=None):
    return mw.Scene.from_dict(json.loads(json.dumps(document, allow_nan=False)), arrays=arrays)


SCENES = pytest.mark.parametrize(
    "factory", [_scene_a, _scene_b, _scene_c, _scene_d, _scene_e, _scene_f],
    ids=["a-dielectrics", "b-pcb", "c-periodic", "d-wire", "e-mode-abc", "f-collections"])


@SCENES
def test_round_trip_is_exact(factory):
    scene = factory()
    original = scene.to_dict()
    assert original["schema"] == SCHEMA == "witwin.scene/2"
    assert original["package_version"] == mw.__version__
    assert {field.name for field in fields(mw.Scene)} <= original.keys()
    restored = _json_round_trip(original)
    assert restored.to_dict() == original
    assert canonical(dict(vars(restored))) == canonical(dict(vars(scene)))
    arrays = {}
    with_arrays = scene.to_dict(arrays=arrays)
    assert _json_round_trip(with_arrays, arrays).to_dict() == original


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="material compilation of these grids needs CUDA")
@SCENES
def test_round_trip_compiles_identically(factory):
    scene = factory()
    restored = _json_round_trip(scene.to_dict())
    actual = compile_material_tensors(prepare_scene(restored, device="cuda"), frequency=FREQUENCY)
    expected = compile_material_tensors(prepare_scene(scene, device="cuda"), frequency=FREQUENCY)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert torch.equal(actual_tensor, expected_tensor)
    assert torch.cuda.max_memory_allocated() < 8 * 1024**3


def test_document_is_strict_json_and_sort_proof():
    scene = _scene_f()
    document = scene.to_dict()
    text = json.dumps(document, allow_nan=False)
    assert "Infinity" not in text and "NaN" not in text
    assert document["internal_absorbers"][0]["size"][0] == {"type": "Float", "value": "inf"}
    # A store that sorts JSON object keys (the EMFC records) changes nothing: mappings are item lists.
    restored = mw.Scene.from_dict(json.loads(json.dumps(document, allow_nan=False, sort_keys=True)))
    assert canonical(dict(vars(restored))) == canonical(dict(vars(scene)))
    assert list(restored.metadata) == ["type", "array", "values"]
    assert restored.metadata["values"] == (1, 2.5, math.inf)

    def references(value):
        if isinstance(value, dict):
            own = [value["array"]] if isinstance(value.get("array"), str) else []
            return own + [name for item in value.values() for name in references(item)]
        if isinstance(value, list):
            return [name for item in value for name in references(item)]
        return []

    # The key "array" with a string value only ever names an array reference.
    assert references(document) == []
    arrays = {}
    assert sorted(references(scene.to_dict(arrays=arrays))) == sorted(arrays) == []


def test_scene_document_is_deterministic():
    scene = _scene_b()
    first, second = {}, {}
    assert json.dumps(scene.to_dict(arrays=first)) == json.dumps(scene.to_dict(arrays=second))
    assert first.keys() == second.keys()


_BUILD_IN_FRESH_PROCESS = """
import hashlib, json, sys
import torch
import witwin.maxwell as mw
assert mw.__file__.startswith(sys.argv[1]), mw.__file__
for _ in range(int(sys.argv[2])):  # unrelated history: advances every identity counter
    mw.Scene(structures=[mw.Structure(mw.Box(), mw.Material(eps_r=3.)), mw.Structure(mw.Sphere(), mw.Material())])
sys.path.insert(0, sys.argv[1])
from tests.api.public.test_scene_dict import _scene_b, _scene_f
torch.manual_seed(0)
arrays = {}
text = json.dumps([_scene_b().to_dict(arrays=arrays), _scene_f().to_dict()], allow_nan=False, sort_keys=True)
print(text)
print(json.dumps({key: hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest() for key, value in arrays.items()}))
"""


def test_documents_are_byte_identical_across_processes_and_build_histories():
    import os
    from pathlib import Path
    import subprocess
    import sys

    root = str(Path(__file__).resolve().parents[3])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([root, os.environ.get("PYTHONPATH", "")])}
    outputs = [subprocess.run([sys.executable, "-c", _BUILD_IN_FRESH_PROCESS, root, str(history)], env=env,
                              capture_output=True, text=True, check=True).stdout for history in (0, 7)]
    assert outputs[0] == outputs[1]
    document = json.loads(outputs[0].splitlines()[0])[0]
    assert [structure["structure_id"] for structure in document["structures"]] == list(range(6))
    assert {structure["material_id"] for structure in document["structures"]} == {None}


def test_identity_numbers_keep_equalities_and_inheritance():
    material = mw.Material(eps_r=2.)
    core_material = __import__("witwin.core", fromlist=["PhysicalMaterial"]).PhysicalMaterial(eps_r=3.)
    explicit = mw.Structure(mw.Box(), material, structure_id=900, surface_id=55)
    scene = mw.Scene(structures=[
        mw.Structure(mw.Sphere(), core_material), explicit,
        mw.Structure(mw.Box(position=(.1, 0., 0.)), material, surface_id=55, material_id=4242),
    ])
    document = scene.to_dict()
    records = document["structures"]
    assert [record["structure_id"] for record in records] == [0, 1, 2]
    assert records[1]["surface_id"] == records[2]["surface_id"] != records[0]["surface_id"]
    assert records[0]["material_id"] is None and records[1]["material_id"] is None
    assert document["materials"]["m0"]["material_id"] == 0 and records[2]["material_id"] == 1
    restored = _json_round_trip(document)
    first, second, third = restored.structures
    assert first.material_id == first.material.material_id
    assert second.material_id == second.material.material_id != third.material_id
    assert second.surface_id == third.surface_id
    assert restored.to_dict() == document


def test_large_arrays_are_named_by_document_path_with_dtype():
    coords = np.linspace(-1., 1., 4097)
    density = torch.rand((17, 17, 17), dtype=torch.float64)
    scene = mw.Scene(domain=mw.Domain(((-1., 1.),) * 3), grid=mw.GridSpec.custom(coords, [-1., 1.], [-1., 1.]),
                     material_regions=[mw.MaterialRegion("design", mw.Box(size=(.5, .5, .5)), density)])
    arrays = {}
    document = scene.to_dict(arrays=arrays)
    assert document["grid"]["x_coords"] == {"array": "grid/x_coords", "ndarray": True}
    assert document["material_regions"][0]["density"] == {"array": "material_regions/0/density"}
    assert arrays["material_regions/0/density"].dtype == torch.float64
    assert torch.equal(arrays["grid/x_coords"], torch.from_numpy(coords))
    restored = _json_round_trip(document, arrays)
    assert isinstance(restored.grid.x_coords, np.ndarray) and np.array_equal(restored.grid.x_coords, coords)
    assert torch.equal(restored.material_regions[0].density, density)
    inline = scene.to_dict()["grid"]["x_coords"]
    assert inline["type"] == "NDArray" and inline["dtype"] == "float64" and inline["shape"] == [4097]


def test_shared_materials_stay_shared():
    box, sphere = mw.Box(), mw.Sphere()
    shared = mw.Material(eps_r=3.)
    scene = mw.Scene(background=shared, structures=[mw.Structure(box, shared), mw.Structure(sphere, shared)])
    document = scene.to_dict()
    assert list(document["materials"]) == ["m0"]
    assert document["background"] == {"type": "MaterialRef", "key": "m0"}
    restored = mw.Scene.from_dict(document)
    assert restored.background is restored.structures[0].material is restored.structures[1].material


def test_trainable_tensors_come_back_as_fresh_leaves():
    """Q19: values plus the requires_grad flag; from_dict builds new leaves, optimizer state stays outside."""
    eps = torch.tensor(2.5, dtype=torch.float64, requires_grad=True)
    position = torch.tensor([0.0, 0.01, 0.0], requires_grad=True)
    density = torch.rand((4, 4, 4), dtype=torch.float64, requires_grad=True)
    scene = mw.Scene(
        domain=mw.Domain(((-.03, .03),) * 3), grid=mw.GridSpec.uniform(.005),
        structures=[mw.Structure(mw.Box(position=position * 1.0, size=(.01, .01, .01)), mw.Material(eps_r=eps))],
        material_regions=[mw.MaterialRegion("design", mw.Box(size=(.02, .02, .02)), density)],
    )
    arrays = {}
    document = scene.to_dict(arrays=arrays)
    record = document["materials"]["m0"]["eps_r"]
    assert record["requires_grad"] is True and record["dtype"] == "float64"
    assert document["structures"][0]["geometry"]["position"]["requires_grad"] is True
    assert "requires_grad" not in document["structures"][0]["geometry"]["size"]
    restored = _json_round_trip(document, arrays)
    for original, rebuilt in ((eps, restored.structures[0].material.eps_r),
                              (density, restored.material_regions[0].density),
                              (position, restored.structures[0].geometry.position)):
        assert rebuilt.is_leaf and rebuilt.requires_grad and rebuilt is not original
        assert rebuilt.dtype == original.dtype and torch.equal(rebuilt.detach(), original.detach())
    assert restored.to_dict() == document
    (restored.structures[0].material.eps_r * restored.material_regions[0].density.sum()).backward()
    assert restored.material_regions[0].density.grad is not None and eps.grad is None


def test_complex_and_non_finite_tensors_round_trip():
    values = torch.tensor([[1.0 + 2.0j, math.inf - 1.0j], [math.nan, -0.0]], dtype=torch.complex64)
    scene = mw.Scene(structures=[mw.Structure(mw.Box(), mw.Material(eps_r=2.))], metadata={"values": values})
    document = scene.to_dict()
    record = document["metadata"]["items"][0][1]
    assert record["dtype"] == "complex64" and record["shape"] == [2, 2]
    restored = _json_round_trip(document).metadata["values"]
    assert restored.dtype == torch.complex64
    assert torch.equal(torch.view_as_real(restored).isnan(), torch.view_as_real(values).isnan())
    assert torch.equal(torch.nan_to_num(torch.view_as_real(restored)), torch.nan_to_num(torch.view_as_real(values)))


class _PrivateBox(mw.Box):
    pass


@pytest.mark.parametrize("case", ["callable", "subclass", "object"])
def test_unserializable_objects_are_refused_with_their_path(case):
    if case == "callable":
        scene = mw.Scene(sources=[mw.PointDipole((0., 0., 0.), source_time=mw.sources.CustomSourceTime(
            lambda t: 0.0, characteristic_frequency=FREQUENCY))])
        match = r"sources/0/source_time/\w+: callable"
    elif case == "subclass":
        scene = mw.Scene(structures=[mw.Structure(mw.Box(), mw.Material()),
                                     mw.Structure(_PrivateBox(), mw.Material())])
        match = r"structures/1/geometry: .*_PrivateBox is not serializable in witwin.scene/2"
    else:
        scene = mw.Scene(metadata={"path": object()})
        match = r"metadata/0/value: builtins.object is not serializable"
    with pytest.raises(mw.SchemaError, match=match):
        scene.to_dict()


def test_reader_refusals():
    document = _scene_a().to_dict()
    with pytest.raises(mw.SchemaError, match=r"unsupported scene schema 'witwin.scene/3'"):
        mw.Scene.from_dict({**document, "schema": "witwin.scene/3"})
    # A /1 schema goes through the migrator, which refuses /2-only keys (test_scene_migrate.py covers /1).
    with pytest.raises(mw.SchemaError, match=r"unknown witwin.scene/1 document key"):
        mw.Scene.from_dict({**document, "schema": "witwin.scene/1"})
    with pytest.raises(mw.SchemaError, match=r"unknown scene document key"):
        mw.Scene.from_dict({**document, "extra": 1})
    with pytest.raises(mw.SchemaError, match=r"domain: unknown scene type 'Domain3'"):
        mw.Scene.from_dict({**document, "domain": {**document["domain"], "type": "Domain3"}})
    with pytest.raises(mw.SchemaError, match=r"domain: Domain has no serialized field\(s\) \['size'\]"):
        mw.Scene.from_dict({**document, "domain": {**document["domain"], "size": 1}})
    with pytest.raises(mw.SchemaError, match=r"array 'grid/x' is missing"):
        mw.Scene.from_dict({**document, "metadata": {"array": "grid/x"}})
    with pytest.raises(mw.SchemaError, match=r"unsupported units"):
        mw.Scene.from_dict({**document, "unit": {"length": "mm", "frequency": "Hz"}})
    with pytest.raises(TypeError):
        mw.Scene.from_dict(document, device="cpu")


def test_shared_trainable_tensor_reloads_as_one_shared_leaf():
    """User decision 2026-11-13: a tensor shared by identity is written once and reloads as one leaf."""
    eps = torch.tensor(2.5, requires_grad=True)
    position = torch.tensor([0.0, 0.01, 0.0], requires_grad=True)
    scene = mw.Scene(
        domain=mw.Domain(((-.03, .03),) * 3), grid=mw.GridSpec.uniform(.005),
        structures=[mw.Structure(mw.Box(position=position, size=(.01, .01, .01)), mw.Material(eps_r=eps)),
                    mw.Structure(mw.Sphere(position=position, radius=.004), mw.Material(eps_r=eps, sigma_e=.1))],
    )
    document = scene.to_dict()
    assert list(document["tensors"]) == ["t0", "t1"]
    assert document["tensors"]["t0"]["requires_grad"] is True
    assert document["structures"][0]["geometry"]["position"] == {"type": "TensorRef", "key": "t0"}
    assert document["materials"]["m1"]["eps_r"] == {"type": "TensorRef", "key": "t1"}
    restored = _json_round_trip(document)
    first, second = restored.structures
    assert first.geometry.position is second.geometry.position
    assert first.material.eps_r is second.material.eps_r
    leaves = {id(t): t for t in (first.geometry.position, second.geometry.position,
                                 first.material.eps_r, second.material.eps_r)}
    optimizer = torch.optim.SGD(list(leaves.values()), lr=0.1)
    assert len(optimizer.param_groups[0]["params"]) == 2 and all(t.is_leaf for t in leaves.values())
    (2.0 * first.material.eps_r + 3.0 * second.material.eps_r).backward()
    assert first.material.eps_r.grad.item() == 5.0  # both uses accumulate into the one parameter
    assert canonical(dict(vars(restored))) == canonical(dict(vars(scene)))
    assert restored.to_dict() == document


def test_many_same_type_monitors_are_written_as_columns():
    monitors = [mw.PointMonitor(f"probe_{index:04d}", (0.001 * index, 0.0, 0.5e-3), fields=("Ez",))
                for index in range(100)]
    scene = mw.Scene(domain=mw.Domain(((-.2, .2),) * 3), grid=mw.GridSpec.uniform(.01),
                     monitors=[mw.FluxMonitor("flux", axis="z", position=.01), *monitors,
                               mw.FluxMonitor("tail", axis="x", position=.01)])
    document = scene.to_dict()
    flux, block, tail = document["monitors"]
    assert flux["type"] == "FluxMonitor" and tail["name"] == "tail"
    assert block == {"type": "Records", "of": "PointMonitor", "count": 100, "columns": {
        "name": {"prefix": "probe_", "digits": 4, "start": 0},
        "position": {"values": [[0.001 * index, 0.0, 0.5e-3] for index in range(100)]},
        "fields": {"constant": ["Ez"]}}}
    restored = _json_round_trip(document)
    assert canonical(dict(vars(restored))) == canonical(dict(vars(scene)))
    assert restored.to_dict() == document
    arrays = {}
    packed = scene.to_dict(arrays=arrays)
    assert packed["monitors"][1]["columns"]["position"] == {"array": "monitors/1/columns/position"}
    assert arrays["monitors/1/columns/position"].dtype == torch.float64
    assert canonical(dict(vars(_json_round_trip(packed, arrays)))) == canonical(dict(vars(scene)))
    short = mw.Scene(monitors=monitors[:10]).to_dict()
    assert [record["type"] for record in short["monitors"]] == ["PointMonitor"] * 10
