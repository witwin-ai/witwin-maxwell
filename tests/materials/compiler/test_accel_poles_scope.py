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
from witwin.maxwell.scene import _prepared_declarations


@pytest.fixture
def pole_case():
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.5, 0.5),) * 3),
                     grid=mw.GridSpec.uniform(.25), boundary=mw.BoundarySpec.pec(),
                     subpixel=mw.SubpixelSpec(samples=4, averaging="kottke_edge"),
                     structures=(
                         mw.Structure(geometry=mw.Box(size=.5, device="cpu"),
                             material=mw.Material(eps_r=2., debye_poles=(mw.DebyePole(1., 1e-10),))),
                     ))
    scene = prepare_scene(scene, device="cpu")
    layout = m._build_dispersive_layout(scene)
    model = m._new_material_model(scene, layout, eps_fill=1., mu_fill=1.)
    model["edge_components"] = None
    kwargs = dict(eps_background=1., mu_background=1., region_densities=(),
                  geometry_periodic_shifts={id(s.geometry): ((0.,), (0.,), (0.,)) for s in scene.scene.structures})
    return scene, model, layout, kwargs


@pytest.mark.parametrize("geometry", [
    lambda: mw.Box(size=.5, device="cpu"),
    lambda: mw.Box(size=.5, rotation=(.1, .2, .3), device="cpu"),
    lambda: mw.Sphere(radius=.25, device="cpu"),
    lambda: mw.Cylinder(radius=.2, height=.5, axis="y", device="cpu"),
])
def test_fixed_primitive_scope(pole_case, geometry):
    prepared, model, layout, kwargs = pole_case
    prepared.scene = _prepared_declarations(prepared.scene, structures=[replace(prepared.scene.structures[0], geometry=geometry())])
    # Any CUDA use during the predicate raises, independent of earlier tests in the process.
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("scope predicate touched CUDA")):
        assert scope_reason(prepared, model, layout, **kwargs) is None


@pytest.mark.parametrize("trainable,graded,expected", [
    (False, False, None), (True, False, "uniform_amplitude"), (False, True, "graded"),
])
def test_custom_pole_scope(pole_case, trainable, graded, expected):
    prepared, model, _, kwargs = pole_case
    strength = torch.ones((2, 2, 2), requires_grad=trainable)
    if graded:
        strength[0, 0, 0] = torch.nextafter(strength[0, 0, 0], torch.tensor(float("inf")))
    material = mw.Material(eps_r=2., lorentz_poles=(
        mw.media.CustomLorentzPole(strength, resonance_frequency=1e9, gamma=1e8),))
    prepared.scene = _prepared_declarations(prepared.scene, structures=[replace(prepared.scene.structures[0], material=material)])
    layout = m._build_dispersive_layout(prepared)
    assert scope_reason(prepared, model, layout, **kwargs) == expected


@pytest.mark.parametrize("reason", [
    "averaging", "material_regions", "periodic", "periodic_shifts", "coordinates_or_dtype",
    "trainable_background", "tensor_background", "modulation", "perturbation",
    "trainable_geometry", "geometry", "geometry_dtype",
])
def test_excluded_scope_is_unchanged(pole_case, monkeypatch, reason):
    prepared, model, layout, kwargs = pole_case
    structure = prepared.scene.structures[0]
    if reason == "averaging":
        prepared.scene = _prepared_declarations(prepared.scene, subpixel=mw.SubpixelSpec(samples=4, averaging="kottke_smooth"))
    elif reason == "material_regions":
        kwargs["region_densities"] = (torch.zeros(1),)
    elif reason == "periodic":
        prepared.scene = _prepared_declarations(prepared.scene, boundary=mw.BoundarySpec.periodic())
    elif reason == "periodic_shifts":
        kwargs["geometry_periodic_shifts"] = {id(structure.geometry): None}
    elif reason == "coordinates_or_dtype":
        prepared.scene = _prepared_declarations(prepared.scene, coordinates=object())
    elif reason == "trainable_background":
        kwargs["eps_background"] = torch.tensor(1., requires_grad=True)
    elif reason == "tensor_background":
        model["eps_offdiag_components"]["xy"] = torch.ones(1)
    elif reason == "modulation":
        prepared.scene = _prepared_declarations(prepared.scene, structures=[replace(structure, material=mw.Material(
            modulation=mw.media.ModulationSpec(amplitude=.1, frequency=1e8)))])
    elif reason == "perturbation":
        prepared.scene = _prepared_declarations(prepared.scene, structures=[replace(structure, material=mw.media.PerturbationMedium(
            structure.material, perturbation=torch.zeros((2, 2, 2))))])
    elif reason == "trainable_geometry":
        structure.geometry.size.requires_grad_(True)
    elif reason == "geometry":
        prepared.scene = _prepared_declarations(prepared.scene, structures=[replace(structure, geometry=mw.geometry.Union(
            mw.Box(device="cpu"), mw.Sphere(device="cpu"), device="cpu"))])
    elif reason == "geometry_dtype":
        structure.geometry.size = structure.geometry.size.double()
    assert scope_reason(prepared, model, layout, **kwargs) == reason
    # A CUDA-labelled prepared with CPU tensors exercises rejection before a kernel
    # build or device allocation, independently of the machine's GPU inventory.
    monkeypatch.setattr(prepared, "device", "cuda")
    token = setup_enabled.set(True)
    try:
        with (patch("witwin.maxwell.compiler.accel.poles.build_extension") as extension,
              patch("torch.cuda._lazy_init", side_effect=AssertionError("rejection touched CUDA"))):
            assert try_compile_binary_pole_edges(prepared, model, layout, samples=(4, 4, 4), **kwargs) is UNCHANGED
        extension.assert_not_called()
    finally:
        setup_enabled.reset(token)
    assert "edge_weights" not in model["debye_poles"][0]
    assert "pole_edge_background" not in model


def test_cpu_and_disabled_leave_model_unchanged(pole_case, monkeypatch):
    prepared, model, layout, kwargs = pole_case
    for device, enabled in (("cpu", True), ("cuda", False)):
        monkeypatch.setattr(prepared, "device", device)
        token = setup_enabled.set(enabled)
        try:
            with (patch("witwin.maxwell.compiler.accel.poles.build_extension") as extension,
                  patch("torch.cuda._lazy_init", side_effect=AssertionError("refusal touched CUDA"))):
                assert try_compile_binary_pole_edges(prepared, model, layout, samples=(4, 4, 4), **kwargs) is UNCHANGED
            extension.assert_not_called()
        finally:
            setup_enabled.reset(token)


def test_base_dispatch_accepts_dispersion_but_coefficients_keep_refusal(pole_case, monkeypatch):
    prepared, _, _, kwargs = pole_case
    monkeypatch.setattr(prepared, "device", "cuda")
    token = setup_enabled.set(True)
    try:
        assert eligible(prepared, 1., 1., prepared.scene.subpixel)
        assert not eligible(prepared, 1., 1., prepared.scene.subpixel, coefficients=True)
    finally:
        setup_enabled.reset(token)
