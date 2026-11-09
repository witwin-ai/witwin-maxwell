"""Boolean (CSG) geometry through the material compiler on CPU (no solver run)."""

from __future__ import annotations

import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene


def _scene(geometry, material, *, averaging="arithmetic", spacing=0.1):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))),
        grid=mw.GridSpec.uniform(spacing),
        device="cpu",
        # Polarized averaging needs more than one sub-sample per cell.
        subpixel_samples=mw.SubpixelSpec(averaging=averaging, samples=2 if averaging == "polarized" else 1),
    )
    scene.add_structure(mw.Structure(name="csg", geometry=geometry, material=material))
    return scene


def _components(geometry, material=None, **kwargs):
    material = mw.Material(eps_r=4.0) if material is None else material
    eps, _ = prepare_scene(_scene(geometry, material, **kwargs)).compile_material_components()
    return eps


@pytest.mark.parametrize("averaging", ["arithmetic", "polarized"])
def test_difference_of_boxes_compiles_exactly_like_hollow_box(averaging):
    rotation = (0.2, 0.0, 0.3)
    difference = mw.Difference(
        mw.Box(size=(1.2, 1.0, 0.8)), mw.Box(size=(0.8, 0.6, 0.4)), position=(0.1, 0.0, -0.05), rotation=rotation
    )
    hollow = mw.HollowBox(
        position=(0.1, 0.0, -0.05), outer_size=(1.2, 1.0, 0.8), inner_size=(0.8, 0.6, 0.4), rotation=rotation
    )
    eps_csg = _components(difference, averaging=averaging)
    eps_ref = _components(hollow, averaging=averaging)
    for axis in ("x", "y", "z"):
        torch.testing.assert_close(eps_csg[axis], eps_ref[axis], rtol=0.0, atol=1.0e-5)


def test_difference_material_fill_matches_analytic_volume():
    spacing = 0.05
    geometry = mw.Difference(mw.Box(size=(1.2, 1.2, 1.2)), mw.Sphere(radius=0.4))
    model = prepare_scene(_scene(geometry, mw.Material(eps_r=4.0), spacing=spacing)).compile_materials()
    eps = model["eps_r"]
    centre = tuple(index // 2 for index in eps.shape)
    assert float(eps[centre]) == pytest.approx(1.0, abs=1.0e-4)
    shell = (centre[0] + 10, centre[1], centre[2])  # x = 0.5, inside the box wall
    assert float(eps[shell]) == pytest.approx(4.0, abs=1.0e-4)
    fill = (eps - 1.0) / 3.0
    expected = 1.2**3 - 4.0 / 3.0 * math.pi * 0.4**3
    assert float(fill.sum()) * spacing**3 == pytest.approx(expected, rel=2.0e-2)


def test_compiled_permittivity_is_differentiable_in_child_parameters():
    radius = torch.tensor(0.4, requires_grad=True)
    shift = torch.zeros(3, requires_grad=True)
    geometry = mw.Difference(mw.Box(position=shift, size=(1.2, 1.2, 1.2)), mw.Sphere(radius=radius))
    scene = _scene(geometry, mw.Material(eps_r=4.0), averaging="polarized")
    prepared = prepare_scene(scene)
    # The prepared scene sees the nested parameters, so trainable compiles bypass the cache.
    assert prepared._has_trainable_geometry()
    eps, _ = prepared.compile_material_components()
    loss = sum(component.sum() for component in eps.values())
    loss.backward()
    assert radius.grad is not None and float(radius.grad) < 0.0
    assert shift.grad is not None and torch.isfinite(shift.grad).all()

    def fixed_loss(value):
        fixed = mw.Difference(mw.Box(size=(1.2, 1.2, 1.2)), mw.Sphere(radius=value))
        with torch.no_grad():
            components = _components(fixed, averaging="polarized")
        return float(sum(component.sum() for component in components.values()))

    step = 2.0e-3
    finite_difference = (fixed_loss(0.4 + step) - fixed_loss(0.4 - step)) / (2 * step)
    assert float(radius.grad) == pytest.approx(finite_difference, rel=5.0e-2)


def test_pec_difference_matches_hollow_box_occupancy():
    difference = mw.Difference(mw.Box(size=(1.2, 1.2, 1.2)), mw.Box(size=(0.8, 0.8, 0.8)))
    hollow = mw.HollowBox(outer_size=(1.2, 1.2, 1.2), inner_size=(0.8, 0.8, 0.8))
    csg = prepare_scene(_scene(difference, mw.Material.pec())).compile_materials()
    ref = prepare_scene(_scene(hollow, mw.Material.pec())).compile_materials()
    assert csg["pec_mode"] == ref["pec_mode"]
    torch.testing.assert_close(csg["pec_occupancy"], ref["pec_occupancy"], rtol=0.0, atol=1.0e-6)


def test_smooth_union_and_scene_placement():
    left = mw.Sphere(position=(-0.3, 0.0, 0.0), radius=0.3)
    right = mw.Sphere(position=(0.3, 0.0, 0.0), radius=0.3)
    sharp = _components(mw.Union(left, right))["x"]
    smooth = _components(mw.Union(left, right, smoothness=0.3))["x"]
    # Blending only adds material near the seam.
    assert bool(torch.all(smooth >= sharp - 1.0e-5))
    assert float((smooth - sharp).sum()) > 0.0

    scene = _scene(mw.Union(left, right), mw.Material(eps_r=4.0))
    placed = scene.structures[0].geometry
    tensors = {id(value) for value in vars(placed).values() if isinstance(value, torch.Tensor)}
    assert all(id(child.radius) in tensors for child in placed.geometries)


def test_periodic_scene_wraps_boolean_geometry_like_primitive():
    def compile_with(geometry):
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
            grid=mw.GridSpec.uniform(0.05),
            boundary=mw.BoundarySpec.periodic(),
            device="cpu",
            subpixel_samples=mw.SubpixelSpec(samples=1),
        )
        scene.add_structure(mw.Structure(geometry=geometry, material=mw.Material(eps_r=3.0)))
        eps, _ = prepare_scene(scene).compile_material_components()
        return eps

    # A rod crossing the periodic x faces with a spherical void.
    rod = mw.Box(position=(0.45, 0.0, 0.0), size=(0.4, 0.3, 0.3))
    void = mw.Sphere(position=(0.3, 0.0, 0.0), radius=0.08)
    eps_csg = compile_with(mw.Difference(rod, void))
    rod_only = compile_with(rod)
    # The periodic image of the rod reaches the low x face exactly as the primitive's does,
    # and the void only ever removes material.
    for axis in ("x", "y", "z"):
        torch.testing.assert_close(eps_csg[axis][0], rod_only[axis][0], rtol=0.0, atol=1.0e-5)
        assert bool(torch.all(eps_csg[axis] <= rod_only[axis] + 1.0e-5))
        assert float((rod_only[axis] - eps_csg[axis]).sum()) > 0.0
