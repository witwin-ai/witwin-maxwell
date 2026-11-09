"""CPU-authored geometry descriptors must follow CUDA query placement."""

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.accel.build import build_extension
from witwin.maxwell.compiler.accel.primitives import _CoverageGeometry, _DistanceGeometry
from witwin.maxwell.scene import prepare_scene


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Setup requires CUDA.")


def _scene():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5),) * 3),
        grid=mw.GridSpec.uniform(0.125), boundary=mw.BoundarySpec.pec(), device="cuda",
        subpixel_samples=mw.SubpixelSpec(samples=3, averaging="polarized"),
    )
    scene.add_structure(mw.Structure(
        geometry=mw.Sphere(position=(-0.125, 0., 0.), radius=0.23),
        material=mw.Material(eps_r=3.7, sigma_e=0.013),
    ))
    scene.add_structure(mw.Structure(
        geometry=mw.Box(position=(0.125, 0., 0.), size=(0.25, 0.5, 0.5)),
        material=mw.Material.pec(),
    ))
    return scene


def test_cpu_geometry_descriptor_follows_query_device():
    scene = _scene()
    prepared = prepare_scene(scene)
    for structure in scene.structures:
        assert structure.geometry.device.type == "cpu"
        for geometry in (_DistanceGeometry(structure.geometry),
                         _CoverageGeometry(structure.geometry, (0.125,) * 3)):
            assert not geometry.packed
            geometry.signed_distance(prepared.X, prepared.Y, prepared.Z)
            packed = geometry.packed[prepared.X.device]
            assert packed.device == prepared.X.device
            assert packed.is_cuda and packed.is_contiguous()
            geometry.signed_distance(prepared.X, prepared.Y, prepared.Z)
            if isinstance(geometry, _CoverageGeometry):
                geometry.to_mask(prepared.X, prepared.Y, prepared.Z)
            assert geometry.packed[prepared.X.device] is packed
        assert structure.geometry.device.type == "cpu"


def test_accelerated_pec_occupancy_repeats_with_cpu_geometry():
    scene = _scene()
    expected = mw.Simulation.fdtd(
        scene, frequency=1.e9, fdtd_core_path=False, accelerated_setup=False).prepare().solver
    expected = expected._compiled_material_model["pec_occupancy"].clone()
    assert bool((expected == 0.5).any())
    for _ in range(20):
        prepared = mw.Simulation.fdtd(
            scene, frequency=1.e9, accelerated_setup=True, fdtd_core_path=False).prepare()
        # Reuse small host allocations immediately after the asynchronous setup.
        scratch = [torch.empty(384, device="cpu", dtype=torch.float32) for _ in range(64)]
        for tensor in scratch:
            tensor.fill_(12345.)
        del tensor, scratch
        actual = prepared.solver._compiled_material_model["pec_occupancy"]
        assert torch.equal(actual, expected)
        del prepared


def test_distance_binding_rejects_cpu_descriptor():
    geometry = torch.zeros((1, 16, 24), dtype=torch.float32, device="cpu")
    points = torch.zeros((1, 3), dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="distance: geometry must be a contiguous CUDA tensor on the same device"):
        build_extension().distance(geometry, points)


def test_pole_binding_rejects_cpu_old_component():
    cuda = dict(device="cuda")
    f, d = dict(dtype=torch.float32, **cuda), dict(dtype=torch.float64, **cuda)
    axes = [torch.zeros(1, **d) for _ in range(3)]
    with pytest.raises(RuntimeError, match="poles: old must be a contiguous CUDA tensor on the same device"):
        build_extension().poles(
            torch.zeros((1, 16, 24), **f), torch.zeros((1, 16, 24), **d), torch.zeros((1, 6), **f),
            axes, axes, axes, [torch.zeros(1, **f) for _ in range(3)],
            torch.zeros(1, **f), torch.zeros(1, dtype=torch.bool, **cuda), torch.zeros(1, 1, **f),
            torch.zeros(1, dtype=torch.float32), torch.zeros(1, dtype=torch.int64, **cuda),
            [1] * 12, [1., 1., 1., 1.])
