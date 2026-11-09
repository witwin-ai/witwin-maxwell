"""Bitwise CUDA oracle for accelerated binary pole ownership and ADE drives."""
from contextlib import contextmanager
from dataclasses import fields, is_dataclass

import pytest
import torch
from torch.utils._pytree import tree_flatten

import witwin.maxwell as mw
from witwin.maxwell.compiler.accel.dispatch import setup_enabled
from witwin.maxwell.scene import prepare_scene


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Pole oracle requires CUDA.")

_KINDS = ("debye", "drude", "lorentz")
_SHAPES = ("sphere", "box_face", "box_minus_ulp", "box_plus_ulp",
           "rotated_box", "cylinder", "union")
_SPECS = (
    pytest.param(None, id="default"),
    pytest.param(mw.SubpixelSpec(samples=1, averaging="arithmetic"), id="arithmetic_1"),
    pytest.param(mw.SubpixelSpec(samples=4, averaging="arithmetic"), id="arithmetic_4"),
    pytest.param(mw.SubpixelSpec(samples=4, averaging="kottke_smooth"), id="kottke_smooth"),
    pytest.param(mw.SubpixelSpec(samples=3, averaging="polarized"), id="polarized_3"),
    pytest.param(mw.SubpixelSpec(samples=4, averaging="polarized_tensor"), id="polarized_tensor"),
)


@contextmanager
def _setup_scope(enabled):
    token = setup_enabled.set(enabled)
    try:
        yield
    finally:
        setup_enabled.reset(token)


def _assert_tensor_tree_equal(actual, expected, path="model"):
    actual_leaves, actual_spec = tree_flatten(actual)
    expected_leaves, expected_spec = tree_flatten(expected)
    assert actual_spec == expected_spec, path
    for index, (a, b) in enumerate(zip(actual_leaves, expected_leaves, strict=True)):
        leaf_path = f"{path}/{index}"
        assert type(a) is type(b), leaf_path
        if torch.is_tensor(b):
            assert (a.dtype, a.shape, a.device) == (b.dtype, b.shape, b.device), leaf_path
            assert torch.equal(a, b), leaf_path
        elif is_dataclass(b):
            # PyTree treats unregistered dataclasses as leaves; inspect their fields too.
            _assert_tensor_tree_equal(
                {f.name: getattr(a, f.name) for f in fields(a)},
                {f.name: getattr(b, f.name) for f in fields(b)}, leaf_path)
        else:
            assert a == b, leaf_path


def _geometry(shape, *, device="cuda"):
    if shape.startswith("box_"):
        # All six float32 faces coincide with node planes before the one-ulp shift.
        face = torch.tensor(.25, dtype=torch.float32, device=device)
        if shape != "box_face":
            direction = -float("inf") if shape == "box_minus_ulp" else float("inf")
            face = torch.nextafter(face, face.new_tensor(direction))
        return mw.Box(size=(2 * face).repeat(3), device=device)
    if shape == "sphere":
        return mw.Sphere(radius=.23, device=device)
    if shape == "rotated_box":
        return mw.Box(size=(.5, .375, .25), rotation=(.13, .27, .41), device=device)
    if shape == "cylinder":
        return mw.Cylinder(radius=.23, height=.5, axis="z", device=device)
    if shape == "union":
        return mw.Union(
            mw.Sphere(position=(-.125, 0., 0.), radius=.19, device=device),
            mw.Box(position=(.125, 0., 0.), size=(.375, .25, .375), device=device),
            device=device)
    raise ValueError(shape)


def _pole_material(kind="lorentz", magnetic=False, *, scale=1.):
    pole = {
        "debye": lambda: mw.DebyePole(3. * scale, 1e-10),
        "drude": lambda: mw.DrudePole(2e9 * scale, 1e8),
        "lorentz": lambda: mw.LorentzPole(3. * scale, 1.8e9, 1e8),
    }[kind]()
    key = ("mu_" if magnetic else "") + kind + "_poles"
    return mw.Material(eps_r=2., mu_r=1.5, **{key: (pole,)})


def _custom_material(strength, magnetic=False):
    key = ("mu_" if magnetic else "") + "lorentz_poles"
    return mw.Material(eps_r=2., mu_r=1.5,
                       **{key: (mw.CustomLorentzPole(strength, 1.8e9, 1e8),)})


def _scene(material, shape="sphere", *, dielectric=False, subpixel=None,
           boundary=None, device="cuda"):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5),) * 3),
        grid=mw.GridSpec.uniform(.0625),
        boundary=boundary or mw.BoundarySpec.pec(), device=device,
        **({} if subpixel is None else {"subpixel_samples": subpixel}),
    )
    if dielectric:
        scene.add_structure(mw.Structure(
            geometry=mw.Box(size=(2., 2., 2.), device=device), material=mw.Material(eps_r=3.)))
    scene.add_structure(mw.Structure(geometry=_geometry(shape, device=device), material=material))
    return scene


def _solver_tensors(solver):
    values = {f"eps_E{axis}": getattr(solver, f"eps_E{axis}") for axis in "xyz"}
    values.update({f"mu_H{axis}": getattr(solver, f"mu_H{axis}") for axis in "xyz"})
    for family in ("_dispersive_templates", "_magnetic_dispersive_templates"):
        values[family] = {
            component: {kind: [entry["drive"] for entry in templates[kind]] for kind in _KINDS}
            for component, templates in getattr(solver, family).items()
        }
    assert any(entries for family in ("_dispersive_templates", "_magnetic_dispersive_templates")
               for component in values[family].values() for entries in component.values())
    return values


def _check_pieces(prepared, enabled, accelerated, record_property):
    # Report membership is checked only when preparation exposes a pieces attribute.
    for owner in (prepared, prepared.solver):
        for name in ("accel_pieces", "_accel_pieces"):
            if hasattr(owner, name):
                pieces = getattr(owner, name)
                if pieces is not None:
                    handled = "poles" in pieces
                    record_property(f"{type(owner).__name__}.{name}.poles.{enabled}", handled)
                    assert handled == (enabled and accelerated)


def _assert_oracle(scene, record_property, *, accelerated, frequency=1.5e9, dt=3e-12):
    models, solver_values = [], []
    for enabled in (False, True):
        with _setup_scope(enabled):
            model = prepare_scene(scene).compile_materials()
        prepared = mw.Simulation.fdtd(
            scene, frequencies=(frequency,), accelerated_setup=enabled, fdtd_core_path=False,
            run_time=mw.TimeConfig(time_steps=2, dt=dt), cuda_graph=False,
            spectral_sampler=mw.SpectralSampler(normalize_source=False),
        ).prepare()
        _assert_tensor_tree_equal(prepared.solver._compiled_material_model, model, "solver/model")
        _check_pieces(prepared, enabled, accelerated, record_property)
        models.append(model)
        solver_values.append(_solver_tensors(prepared.solver))
    _assert_tensor_tree_equal(models[1], models[0])
    _assert_tensor_tree_equal(solver_values[1], solver_values[0], "solver")
    return models


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("magnetic", (False, True), ids=("electric", "magnetic"))
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dielectric", (False, True), ids=("vacuum", "dielectric"))
def test_binary_pole_geometry_oracle(kind, magnetic, shape, dielectric, record_property):
    scene = _scene(_pole_material(kind, magnetic), shape, dielectric=dielectric)
    models = _assert_oracle(scene, record_property, accelerated=True)
    key = ("mu_" if magnetic else "") + kind + "_poles"
    for model in models:
        for weight in model[key][0]["edge_weights"].values():
            assert bool(((weight == 0) | (weight == 1)).all())
            assert bool((weight == 0).any()) and bool((weight == 1).any())


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("magnetic", (False, True), ids=("electric", "magnetic"))
@pytest.mark.parametrize("dielectric", (False, True), ids=("vacuum", "dielectric"))
def test_later_object_priority_oracle(kind, magnetic, dielectric, record_property):
    scene = _scene(_pole_material(kind, magnetic), dielectric=dielectric)
    # Equal priority makes declaration order decide the overlapping constituent.
    scene.add_structure(mw.Structure(
        geometry=mw.Box(position=(.125, 0., 0.), size=(.375, .5, .5), device="cuda"),
        material=_pole_material(kind, magnetic, scale=.5)))
    scene.add_structure(mw.Structure(
        geometry=mw.Box(position=(0., .125, 0.), size=(.125, .375, .5), device="cuda"),
        material=mw.Material(eps_r=4., mu_r=2.)))
    models = _assert_oracle(scene, record_property, accelerated=True)
    key = ("mu_" if magnetic else "") + kind + "_poles"
    for model in models:
        first, later = model[key]
        for component, weight in first["edge_weights"].items():
            other = later["edge_weights"][component]
            assert not bool((weight.bool() & other.bool()).any())
            assert bool(weight.any()) and bool(other.any())


@pytest.mark.parametrize("magnetic", (False, True), ids=("electric", "magnetic"))
@pytest.mark.parametrize("dielectric", (False, True), ids=("vacuum", "dielectric"))
def test_uniform_custom_pole_oracle(magnetic, dielectric, record_property):
    strength = torch.full((3, 3, 3), 3., device="cuda")
    scene = _scene(_custom_material(strength, magnetic), "box_face", dielectric=dielectric)
    models = _assert_oracle(scene, record_property, accelerated=True)
    key = ("mu_" if magnetic else "") + "lorentz_poles"
    for model in models:
        for weight in model[key][0]["edge_weights"].values():
            assert bool(((weight == 0) | (weight == 1)).all())


@pytest.mark.parametrize("subpixel", _SPECS)
def test_subpixel_spec_oracle(subpixel, record_property):
    scene = _scene(_pole_material(), subpixel=subpixel)
    models = _assert_oracle(scene, record_property, accelerated=subpixel is None)
    if subpixel is not None:
        assert all("edge_weights" not in model["lorentz_poles"][0] for model in models)


def _excluded_scene(case, *, device="cuda"):
    material = _pole_material()
    if case in ("graded_custom", "trainable_uniform_custom"):
        strength = (torch.linspace(.5, 1., 27, device=device).reshape(3, 3, 3)
                    if case == "graded_custom" else
                    torch.full((3, 3, 3), 3., device=device, requires_grad=True))
        return _scene(_custom_material(strength), "box_face", device=device)
    if case == "modulated":
        material = mw.Material(eps_r=2., lorentz_poles=material.lorentz_poles,
                               modulation=mw.ModulationSpec(frequency=1e8, amplitude=.2))
    if case == "perturbation":
        material = mw.PerturbationMedium(
            material, perturbation=torch.full((3, 3, 3), .25, device=device), eps_sensitivity=.7)
        return _scene(material, "box_face", device=device)
    if case in ("periodic", "bloch"):
        boundary = (mw.BoundarySpec.periodic() if case == "periodic" else
                    mw.BoundarySpec.bloch((.7, -.3, .2)))
        scene = _scene(material, boundary=boundary, device=device)
        # Cross the domain face so periodic images participate in ownership.
        scene.structures = [mw.Structure(
            geometry=mw.Sphere(position=(.4375, 0., 0.), radius=.23, device=device),
            material=material)]
        return scene
    scene = _scene(material, device=device,
                   subpixel=mw.SubpixelSpec(samples=1, averaging="arithmetic") if case == "density" else None)
    if case == "density":
        scene.add_material_region(mw.MaterialRegion(
            name="design", geometry=mw.Box(position=(.125, 0., 0.), size=(.25, .5, .5), device=device),
            density=torch.full((3, 3, 3), .5, device=device, requires_grad=True), eps_bounds=(1., 3.)))
    if case == "tensor_background":
        scene.add_structure(mw.Structure(
            geometry=mw.Box(position=(.3125, 0., 0.), size=(.125, .25, .25), device=device),
            material=mw.Material(epsilon_tensor=mw.Tensor3x3(
                ((3., .2, 0.), (.2, 3., 0.), (0., 0., 3.))))))
    return scene


@pytest.mark.parametrize("case", (
    "graded_custom", "modulated", "perturbation", "density", "tensor_background",
    "trainable_uniform_custom", "periodic", "bloch",
))
def test_out_of_scope_oracle(case, record_property):
    _assert_oracle(_excluded_scene(case), record_property, accelerated=False)


def test_node_to_edge_pole_background_and_masks_oracle(record_property):
    scene = _excluded_scene("tensor_background")
    scene.subpixel = mw.SubpixelSpec(samples=1, averaging="kottke_edge")
    models = _assert_oracle(scene, record_property, accelerated=False)
    for model in models:
        assert model["edge_components"] is None
        for key in ("pole_edge_background", "pole_edge_masks"):
            assert set(model[key]) == {"Ex", "Ey", "Ez"}
            assert all(bool(value.any()) for value in model[key].values())
    for key in ("pole_edge_background", "pole_edge_masks"):
        _assert_tensor_tree_equal(models[1][key], models[0][key], key)


@pytest.mark.parametrize("kind", ("debye", "lorentz"))
@pytest.mark.parametrize("magnetic", (False, True), ids=("electric", "magnetic"))
@pytest.mark.parametrize("shape", ("sphere", "box"))
def test_micrometre_scale_oracle(kind, magnetic, shape, record_property):
    # Binary ownership must not depend on an absolute smoothing-width floor at µm scale.
    um = 1e-6
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.4 * um, 0.4 * um),) * 3), grid=mw.GridSpec.uniform(0.02 * um),
        boundary=mw.BoundarySpec.pec(), device="cuda",
        subpixel_samples=mw.SubpixelSpec(samples=4, averaging="kottke_edge"))
    geometry = (mw.Sphere(radius=0.3 * um, device="cuda") if shape == "sphere"
                else mw.Box(size=(0.37 * um, 0.29 * um, 0.43 * um), device="cuda"))
    scene.add_structure(mw.Structure(geometry=geometry, material=_pole_material(kind, magnetic)))
    models = _assert_oracle(scene, record_property, accelerated=True, frequency=1.5e14, dt=2e-17)
    key = ("mu_" if magnetic else "") + kind + "_poles"
    weights = models[1][key][0]["edge_weights"]
    assert weights and all(bool(((w == 0) | (w == 1)).all()) for w in weights.values())
