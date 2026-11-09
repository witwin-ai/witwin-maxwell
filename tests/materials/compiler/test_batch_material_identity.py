"""Published core identity tokens must not invalidate declared design reuse."""
from dataclasses import replace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.materials import _material_definition, _same_material_definition


def test_declared_design_identity_changes_preserve_fixed_material_guards():
    first = mw.Structure(name="design", geometry=mw.Sphere(radius=.02, device="cpu"),
                         material=mw.Material(eps_r=2.))
    second = mw.Structure(name="design", geometry=mw.Sphere(radius=.03, device="cpu"),
                          material=mw.Material(eps_r=3.))
    assert first.structure_id != second.structure_id
    definition = _material_definition(first, structures={"design"}, snapshot=True)
    assert _same_material_definition(definition, _material_definition(second, structures={"design"}))
    assert not _same_material_definition(definition,
        _material_definition(replace(second, priority=2), structures={"design"}))
    assert not _same_material_definition(_material_definition(first), _material_definition(second))


@pytest.mark.parametrize("subpixel", [None, mw.SubpixelSpec(samples=1, averaging="arithmetic"),
    mw.SubpixelSpec(samples=4, averaging="arithmetic"), mw.SubpixelSpec(samples=4, averaging="kottke_smooth"),
    mw.SubpixelSpec(samples=3, averaging="polarized"), mw.SubpixelSpec(samples=4, averaging="polarized_tensor")])
@pytest.mark.parametrize("boundary", [mw.BoundarySpec.pml(num_layers=3), mw.BoundarySpec.periodic()])
def test_batch_member_materials_equal_single_scene_on_cpu(boundary, subpixel):
    from torch.utils._pytree import tree_flatten
    from witwin.maxwell.scene import prepare_scene

    simulations = []
    for index in range(3):
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.2, 0.2),) * 3),
            grid=mw.GridSpec.uniform(0.025), boundary=boundary, device="cpu",
            **({} if subpixel is None else {"subpixel_samples": subpixel}),
            structures=[mw.Structure(name="design", geometry=mw.Box(
                position=(0.05, 0., 0.), size=(0.07, 0.08, 0.09), device="cpu"),
                material=mw.Material(eps_r=2.0 + 0.25 * index))],
            sources=[mw.PointDipole(name="drive", position=(-0.05, 0., 0.),
                polarization="Ez", source_time=mw.CW(frequency=1e9, amplitude=1.0 + index))],
        )
        simulations.append(mw.Simulation.fdtd(scene, frequency=1e9))
    batch = mw.Simulation.batch(simulations, variation=mw.BatchVariation(sources=("drive",)),
        execution=mw.BatchExecution(devices=("cuda:0",), batch_size=3))
    for index, simulation in enumerate(simulations):
        member = prepare_scene(batch.scene(index).clone(device="cpu"))
        assert member.subpixel == (subpixel or mw.SubpixelSpec(samples=4, averaging="kottke_edge"))
        actual, actual_spec = tree_flatten(member.compile_materials())
        expected, expected_spec = tree_flatten(prepare_scene(simulation.scene).compile_materials())
        assert actual_spec == expected_spec
        for a, b in zip(actual, expected, strict=True):
            if torch.is_tensor(a):
                assert torch.equal(a, b)
