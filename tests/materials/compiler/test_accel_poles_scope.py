"""CPU-only admission checks for native binary pole edges."""
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.compiler import materials as m
from witwin.maxwell.compiler.accel.dispatch import UNCHANGED, setup_enabled, eligible
from witwin.maxwell.compiler.accel.poles import scope_reason, try_compile_binary_pole_edges


@pytest.fixture
def pole_case():
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.5, 0.5),) * 3),
                     grid=mw.GridSpec.uniform(.25), boundary=mw.BoundarySpec.pec(),
                     subpixel_samples=mw.SubpixelSpec(samples=4, averaging="kottke_edge"), device="cpu")
    scene.add_structure(mw.Structure(geometry=mw.Box(size=.5, device="cpu"),
        material=mw.Material(eps_r=2., debye_poles=(mw.DebyePole(1., 1e-10),))))
    scene = prepare_scene(scene)
    layout = m._build_dispersive_layout(scene)
    model = m._new_material_model(scene, layout, eps_fill=1., mu_fill=1.)
    model["edge_components"] = None
    kwargs = dict(eps_background=1., mu_background=1., region_densities=(),
                  geometry_periodic_shifts={id(s.geometry): ((0.,), (0.,), (0.,)) for s in scene.structures})
    return scene, model, layout, kwargs


@pytest.mark.parametrize("geometry", [
    lambda: mw.Box(size=.5, device="cpu"),
    lambda: mw.Box(size=.5, rotation=(.1, .2, .3), device="cpu"),
    lambda: mw.Sphere(radius=.25, device="cpu"),
    lambda: mw.Cylinder(radius=.2, height=.5, axis="y", device="cpu"),
])
def test_fixed_primitive_scope(pole_case, geometry):
    scene, model, layout, kwargs = pole_case
    scene.structures = [replace(scene.structures[0], geometry=geometry())]
    # Any CUDA use during the predicate raises, independent of earlier tests in the process.
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("scope predicate touched CUDA")):
        assert scope_reason(scene, model, layout, **kwargs) is None


@pytest.mark.parametrize("trainable,graded,expected", [
    (False, False, None), (True, False, "uniform_amplitude"), (False, True, "graded"),
])
def test_custom_pole_scope(pole_case, trainable, graded, expected):
    scene, model, _, kwargs = pole_case
    strength = torch.ones((2, 2, 2), requires_grad=trainable)
    if graded:
        strength[0, 0, 0] = torch.nextafter(strength[0, 0, 0], torch.tensor(float("inf")))
    material = mw.Material(eps_r=2., lorentz_poles=(
        mw.CustomLorentzPole(strength, resonance_frequency=1e9, gamma=1e8),))
    scene.structures = [replace(scene.structures[0], material=material)]
    layout = m._build_dispersive_layout(scene)
    assert scope_reason(scene, model, layout, **kwargs) == expected


@pytest.mark.parametrize("reason", [
    "averaging", "material_regions", "periodic", "periodic_shifts", "coordinates_or_dtype",
    "trainable_background", "tensor_background", "modulation", "perturbation",
    "trainable_geometry", "geometry", "geometry_dtype",
])
def test_excluded_scope_is_unchanged(pole_case, monkeypatch, reason):
    scene, model, layout, kwargs = pole_case
    structure = scene.structures[0]
    if reason == "averaging":
        scene.subpixel = mw.SubpixelSpec(samples=4, averaging="kottke_smooth")
    elif reason == "material_regions":
        kwargs["region_densities"] = (torch.zeros(1),)
    elif reason == "periodic":
        scene.boundary = mw.BoundarySpec.periodic()
    elif reason == "periodic_shifts":
        kwargs["geometry_periodic_shifts"] = {id(structure.geometry): None}
    elif reason == "coordinates_or_dtype":
        scene.coordinates = object()
    elif reason == "trainable_background":
        kwargs["eps_background"] = torch.tensor(1., requires_grad=True)
    elif reason == "tensor_background":
        model["eps_offdiag_components"]["xy"] = torch.ones(1)
    elif reason == "modulation":
        scene.structures = [replace(structure, material=mw.Material(
            modulation=mw.ModulationSpec(amplitude=.1, frequency=1e8)))]
    elif reason == "perturbation":
        scene.structures = [replace(structure, material=mw.PerturbationMedium(
            structure.material, perturbation=torch.zeros((2, 2, 2))))]
    elif reason == "trainable_geometry":
        structure.geometry.size.requires_grad_(True)
    elif reason == "geometry":
        scene.structures = [replace(structure, geometry=mw.Union(
            mw.Box(device="cpu"), mw.Sphere(device="cpu"), device="cpu"))]
    elif reason == "geometry_dtype":
        structure.geometry.size = structure.geometry.size.double()
    assert scope_reason(scene, model, layout, **kwargs) == reason
    # A CUDA-labelled scene with CPU tensors exercises rejection before a kernel
    # build or device allocation, independently of the machine's GPU inventory.
    monkeypatch.setattr(scene, "device", "cuda")
    token = setup_enabled.set(True)
    try:
        with (patch("witwin.maxwell.compiler.accel.poles.build_extension") as extension,
              patch("torch.cuda._lazy_init", side_effect=AssertionError("rejection touched CUDA"))):
            assert try_compile_binary_pole_edges(scene, model, layout, samples=(4, 4, 4), **kwargs) is UNCHANGED
        extension.assert_not_called()
    finally:
        setup_enabled.reset(token)
    assert "edge_weights" not in model["debye_poles"][0]
    assert "pole_edge_background" not in model


def test_cpu_and_disabled_leave_model_unchanged(pole_case, monkeypatch):
    scene, model, layout, kwargs = pole_case
    for device, enabled in (("cpu", True), ("cuda", False)):
        monkeypatch.setattr(scene, "device", device)
        token = setup_enabled.set(enabled)
        try:
            with (patch("witwin.maxwell.compiler.accel.poles.build_extension") as extension,
                  patch("torch.cuda._lazy_init", side_effect=AssertionError("refusal touched CUDA"))):
                assert try_compile_binary_pole_edges(scene, model, layout, samples=(4, 4, 4), **kwargs) is UNCHANGED
            extension.assert_not_called()
        finally:
            setup_enabled.reset(token)


def test_base_dispatch_accepts_dispersion_but_coefficients_keep_refusal(pole_case, monkeypatch):
    scene, _, _, kwargs = pole_case
    monkeypatch.setattr(scene, "device", "cuda")
    token = setup_enabled.set(True)
    try:
        assert eligible(scene, 1., 1., scene.subpixel)
        assert not eligible(scene, 1., 1., scene.subpixel, coefficients=True)
    finally:
        setup_enabled.reset(token)
