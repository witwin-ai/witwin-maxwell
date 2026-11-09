"""Host admission/precision evidence and CUDA T0 gates for fp64 declarations."""
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import witwin.maxwell as mw
from benchmark.core_path.setup_gates import (
    FLOAT64_SCENES, F1_SCENES, MODES, scene_for, setup_rule2_rows, snapshot,
    write_setup_tensor_table,
)
from witwin.maxwell.compiler import materials
from witwin.maxwell.compiler.accel import conductors, dispatch, poles
from witwin.maxwell.compiler.accel.geometry import descriptors
from witwin.maxwell.scene import prepare_scene


@contextmanager
def enabled(value):
    token = dispatch.setup_enabled.set(value)
    try:
        yield
    finally:
        dispatch.setup_enabled.reset(token)


@pytest.mark.parametrize("name", FLOAT64_SCENES)
@pytest.mark.parametrize("samples,averaging", MODES)
def test_float64_setup_admission_host(name, samples, averaging):
    scene = scene_for(name, device="cpu").clone(
        subpixel_samples=mw.SubpixelSpec(samples=samples, averaging=averaging))
    scene = prepare_scene(scene)
    # Only the device label changes: this predicate must never allocate on CUDA.
    scene.device = "cuda"
    with enabled(True), patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA touched")):
        assert dispatch.eligible(scene, 1., 1., scene.subpixel)
    assert not dispatch._fp32_geometry(scene.structures[0].geometry)
    assert not conductors._qualified(scene.structures)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_lower_precision_geometry_still_refused(dtype):
    scene = prepare_scene(scene_for("float64_box", device="cpu"))
    scene.structures[0].geometry.size = scene.structures[0].geometry.size.to(dtype)
    scene.device = "cuda"
    with enabled(True), patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA touched")):
        assert not dispatch.eligible(scene, 1., 1., scene.subpixel)


@pytest.mark.parametrize("boolean", [False, True])
def test_float64_mesh_still_refused(boolean):
    vertices, faces = mw.Box(device="cpu").to_mesh()
    geometry = mw.Mesh(vertices.double(), faces, recenter=False)
    assert geometry.vertices.dtype == torch.float64
    if boolean:
        geometry = mw.Union(geometry, mw.Sphere(device="cpu"))
    scene = prepare_scene(scene_for("float64_box", device="cpu"))
    scene.structures = [replace(scene.structures[0], geometry=geometry)]
    scene.device = "cuda"
    with enabled(True), patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA touched")):
        assert not dispatch.eligible(scene, 1., 1., scene.subpixel)


@pytest.mark.parametrize("name", FLOAT64_SCENES[:5])
def test_float64_pole_ownership_still_refused(name):
    scene = prepare_scene(scene_for(name, device="cpu"))
    scene.subpixel = mw.SubpixelSpec(samples=4, averaging="kottke_edge")
    layout = materials._build_dispersive_layout(scene)
    model = materials._new_material_model(scene, layout, eps_fill=1., mu_fill=1.)
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA touched")):
        assert poles.scope_reason(scene, model, layout, eps_background=1., mu_background=1.,
            region_densities=(), geometry_periodic_shifts={
                id(s.geometry): ((0.,), (0.,), (0.,)) for s in scene.structures}) == "geometry_dtype"


@pytest.mark.parametrize("name", FLOAT64_SCENES)
def test_reference_sdf_uses_query_precision(name):
    geometry = scene_for(name, device="cpu").structures[0].geometry
    query = torch.rand((19, 1, 1), generator=torch.Generator().manual_seed(37)) - .5
    assert geometry.signed_distance(query, query / 2, -query).dtype == torch.float32
    assert geometry.signed_distance(query.double(), query.double() / 2, -query.double()).dtype == torch.float64


@pytest.mark.parametrize("kind", ["box", "cylinder"])
def test_descriptor_casts_before_halving(kind):
    # Near fp32 underflow, cast-then-divide differs from divide-then-cast.
    tiny = torch.tensor(1.49 * torch.finfo(torch.float32).smallest_normal * 2**-23,
                        dtype=torch.float64)
    geometry = (mw.Box(size=tiny.expand(3)) if kind == "box" else
                mw.Cylinder(height=tiny, radius=.2))
    row = descriptors([SimpleNamespace(geometry=geometry)], device="cpu", native_coverage=True)[0, 0]
    expected = tiny.float() / 2
    assert not torch.equal(expected, (tiny / 2).float())
    assert torch.equal(row[16:19] if kind == "box" else row[17],
                       expected.expand(3) if kind == "box" else expected)


@pytest.mark.parametrize("name", F1_SCENES)
def test_f1_gate_builds_original_cpu_sample(name):
    scene = prepare_scene(scene_for(name, device="cpu"))
    assert len(scene.monitors) == 16384
    scene.device = "cuda"
    with enabled(True), patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA touched")):
        assert dispatch.eligible(scene, 1., 1., scene.subpixel)


def test_setup_tensor_table_records_normalized_errors(tmp_path):
    expected = {"material": torch.tensor([1., 2.])}
    rows = setup_rule2_rows(expected, {"material": expected["material"] + 1e-6})
    output = tmp_path / "tensors.md"
    write_setup_tensor_table([dict(scene="example", samples=4, averaging="polarized", items=rows)], output)
    assert "T1" in output.read_text() and "Max normalized error" in output.read_text()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Setup requires CUDA.")
@pytest.mark.parametrize("name", FLOAT64_SCENES)
@pytest.mark.parametrize("samples,averaging", MODES)
@pytest.mark.parametrize("authored_device", ["cpu", "cuda"])
def test_cuda_float64_material_setup_t0(name, samples, averaging, authored_device, record_property):
    from tests.fdtd.core_path.test_material_setup import assert_model_equal
    from witwin.maxwell.compiler.accel.primitives import sampling_geometry
    from witwin.maxwell.compiler.subpixel import _subsample_fill_fraction

    scene = scene_for(name, device=authored_device).clone(
        device="cuda", subpixel_samples=mw.SubpixelSpec(samples=samples, averaging=averaging))
    prepared = prepare_scene(scene)
    geometry = prepared.structures[0].geometry
    queries = torch.meshgrid(*(getattr(prepared, a) for a in "xyz"), indexing="ij")
    spacing = (.125 / samples,) * 3
    with enabled(False):
        distance = geometry.signed_distance(*queries)
        reference_fill = _subsample_fill_fraction(geometry, *queries, spacing)
    with enabled(True):
        accelerated = sampling_geometry(geometry)
        actual_distance = accelerated.signed_distance(*queries)
        assert torch.equal(actual_distance < 0, distance < 0)
        assert torch.equal(actual_distance, distance)
        actual_fill = _subsample_fill_fraction(accelerated, *queries, spacing)
        for predicate in (lambda f: f == 0, lambda f: f == 1, lambda f: (f > 0) & (f < 1)):
            assert torch.equal(predicate(actual_fill), predicate(reference_fill))
        assert torch.equal(actual_fill, reference_fill)
    with enabled(False):
        expected = materials.compile_material_model(prepared)
    with enabled(True):
        actual = dispatch.try_compile_material_model(prepared, 1., 1., prepared.subpixel, True)
    assert actual is not dispatch.UNCHANGED
    rows = setup_rule2_rows(snapshot(expected), snapshot(actual))
    record_property("tensor_tiers", rows)
    assert_model_equal(actual, expected)
