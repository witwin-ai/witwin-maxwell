"""CPU checks for the (not GPU-run) waveguide-bend topology-optimization example."""

import importlib.util
from pathlib import Path

import torch

import witwin.maxwell as mw

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "invdes" / "waveguide_bend_topology.py"


def test_waveguide_bend_example_builds_a_trainable_scene_on_cpu():
    spec = importlib.util.spec_from_file_location("waveguide_bend_topology", EXAMPLE)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)

    model = example.BendDesign(device="cpu", init=torch.rand(40, 40, 1))
    assert isinstance(model, mw.SceneModule)
    scene = model.to_scene()
    assert isinstance(scene, mw.Scene)
    (region,) = scene.material_regions
    assert region.density.shape == (40, 40, 1) and region.density.requires_grad
    assert region.filter_radius is None and region.projection_beta is None
    torch.testing.assert_close(region.density, region.density.transpose(0, 1))
    penalty = example.length_scale_penalty(model)
    (region.density.sum() + penalty).backward()
    assert model.design.params.grad.abs().sum() > 0
