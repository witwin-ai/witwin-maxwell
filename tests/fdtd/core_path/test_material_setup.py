"""Exact scoped material setup and torch dispatch exclusions."""
from contextlib import contextmanager
import dataclasses
from unittest.mock import patch

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.compiler import materials
from witwin.maxwell.compiler.accel.dispatch import (
    UNCHANGED, setup_enabled, try_compile_material_model,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Setup requires CUDA.")


@contextmanager
def setup_scope(enabled):
    token = setup_enabled.set(enabled)
    try:
        yield
    finally:
        setup_enabled.reset(token)


def assert_model_equal(actual, expected, path="model"):
    if torch.is_tensor(expected):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, path
        assert actual.device == expected.device, path
        assert torch.equal(actual, expected), path
    elif dataclasses.is_dataclass(expected):
        assert type(actual) is type(expected), path
        for field in dataclasses.fields(expected):
            assert_model_equal(getattr(actual, field.name), getattr(expected, field.name),
                               f"{path}/{field.name}")
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            assert_model_equal(actual[key], expected[key], f"{path}/{key}")
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected), path
        for index, value in enumerate(expected):
            assert_model_equal(actual[index], value, f"{path}/{index}")
    else:
        assert actual == expected, path


def small_scene(geometry=None, material=None):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5),)*3), grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.pec(), device="cuda",
        subpixel_samples=mw.SubpixelSpec(samples=3, averaging="polarized"),
    )
    if geometry is not None:
        scene.add_structure(mw.Structure(geometry=geometry, material=material or mw.Material(eps_r=3.7)))
    return scene


@pytest.mark.parametrize("fdtd_core_path", [False, True])
def test_modulated_setup_keeps_torch(fdtd_core_path):
    scene = small_scene(mw.Sphere(radius=0.23), mw.Material(
        eps_r=3., modulation=mw.ModulationSpec(amplitude=0.1, frequency=1e8)))
    simulation = mw.Simulation.fdtd(scene, frequency=2e9, fdtd_core_path=fdtd_core_path)
    prepared = prepare_scene(scene)
    with setup_scope(simulation.config.fdtd_core_path):
        assert try_compile_material_model(prepared, 1., 1., prepared.subpixel, True) is UNCHANGED
        actual = materials.compile_material_model(prepared)
    with setup_scope(False):
        expected = materials.compile_material_model(prepared)
    assert_model_equal(actual, expected)


def test_trainable_geometry_keeps_torch_autograd():
    radius = torch.tensor(.23, device="cuda", requires_grad=True)
    geometry = mw.Union(mw.Sphere(radius=radius, device="cuda"),
                        mw.Box(position=(.15, 0, 0), size=.2, device="cuda"), device="cuda")
    prepared = prepare_scene(small_scene(geometry))
    with setup_scope(True):
        assert try_compile_material_model(prepared, 1., 1., prepared.subpixel, True) is UNCHANGED
        model = materials.compile_material_model(prepared, include_edges=False)
    gradient, = torch.autograd.grad(model["eps_r"].sum(), radius)
    assert torch.isfinite(gradient) and gradient.abs() > 0


def test_modulated_coefficients_keep_torch():
    from witwin.maxwell.compiler.accel import coefficients
    from witwin.maxwell.fdtd.runtime.materials import build_update_coefficients

    scene = small_scene(mw.Sphere(radius=.23), mw.Material(
        eps_r=3., modulation=mw.ModulationSpec(amplitude=.1, frequency=1e8)))
    solver = mw.Simulation.fdtd(scene, frequency=2e9).prepare().solver
    names = [f'c{family}{axis}_{kind}' for family in 'eh' for axis in 'xyz'
             for kind in ('decay', 'curl')]
    expected = {name: getattr(solver, name).clone() for name in names}
    uniformity = solver._coefficient_uniformity.copy()
    with setup_scope(True), patch.object(coefficients, 'build_extension') as extension:
        assert coefficients.try_build_update_coefficients(solver) is UNCHANGED
        build_update_coefficients(solver)
    extension.assert_not_called()
    assert_model_equal({name: getattr(solver, name) for name in names}, expected)
    assert_model_equal(solver._coefficient_uniformity, uniformity)


@pytest.mark.parametrize("kind", ["vacuum", "sphere_box"])
def test_small_setup_equal(kind):
    scene = small_scene()
    if kind == "sphere_box":
        scene.add_structure(mw.Structure(geometry=mw.Sphere(radius=.23), material=mw.Material(eps_r=3.7)))
        scene.add_structure(mw.Structure(
            geometry=mw.Box(position=(.1, 0, 0), size=(.3, .4, .2), rotation=(.1, .2, .3)),
            material=mw.Material(eps_r=2.1), priority=2))
    prepared = prepare_scene(scene)
    if kind == "sphere_box":
        from witwin.maxwell.compiler.accel.build import build_extension
        from witwin.maxwell.compiler.accel.geometry import descriptors

        # Off-lattice queries expose the box norm's reduction order.
        generator = torch.Generator(device="cuda").manual_seed(13)
        points = torch.rand((1003, 3), device="cuda", generator=generator) * 1.4 - .7
        box = prepared.structures[1]
        packed = descriptors([box], device="cuda", native_coverage=True)
        distance = build_extension().distance(packed, points)[0]
        expected_distance = box.geometry.signed_distance(*(points[:, i:i+1, None] for i in range(3))).flatten()
        assert torch.equal(distance, expected_distance)
    with setup_scope(False):
        expected = materials.compile_material_model(prepared)
    with setup_scope(True):
        actual = try_compile_material_model(prepared, 1., 1., prepared.subpixel, True)
        public = materials.compile_material_model(prepared)
    assert actual is not UNCHANGED
    assert_model_equal(actual, expected)
    assert_model_equal(public, expected)


@pytest.mark.parametrize("kind", ["lumped_port", "copper"])
def test_setup_only_features_prepare_with_accelerated_materials(kind):
    from benchmark.core_path.setup_gates import record_setup, scene_for, setup_simulation

    scene = scene_for(kind)
    simulation = setup_simulation(scene, accelerated=True)
    assert simulation.config.accelerated_setup and not simulation.config.fdtd_core_path
    with record_setup() as calls:
        prepared = simulation.prepare()
    assert any(calls) and calls.pieces.get("sample")
    assert not hasattr(prepared.solver, "_core_runtime")
    assert not setup_enabled.get()
    core_prepared = mw.Simulation.fdtd(
        scene, frequency=1.e9, accelerated_setup=True, fdtd_core_path=True).prepare()
    assert hasattr(core_prepared.solver, "_core_runtime")
