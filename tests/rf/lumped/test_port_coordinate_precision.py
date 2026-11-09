"""Coordinate identity tests independent of field state precision."""

from pathlib import Path
import copy
import json

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.ports import _snap_index, compile_port_geometry
from witwin.maxwell.scene import prepare_scene


_SCENES = Path(__file__).resolve().parents[3] / "docs/assessments/thin-copper-2026-10-22/site-scenes"


@pytest.fixture
def site_scenes():
    if not _SCENES.exists():
        pytest.skip("repository evidence not in this checkout: docs/assessments/thin-copper-2026-10-22/site-scenes")


def _assert_same_geometry(actual, expected):
    assert actual.direction == expected.direction
    assert actual.axis == expected.axis
    assert actual.reference_plane == expected.reference_plane
    torch.testing.assert_close(actual.voltage_indices, expected.voltage_indices, rtol=0, atol=0)
    torch.testing.assert_close(actual.voltage_weights, expected.voltage_weights, rtol=0, atol=0)
    assert actual.current_components == expected.current_components
    for actual_indices, expected_indices in zip(actual.current_indices, expected.current_indices):
        torch.testing.assert_close(actual_indices, expected_indices, rtol=0, atol=0)
    for actual_weights, expected_weights in zip(actual.current_weights, expected.current_weights):
        torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)


def _site_port(data, *, dtype):
    axes = [np.asarray(data["mesh"]["axes_m"][axis], dtype=np.float64) for axis in "xyz"]
    positive, negative = np.asarray(data["ports"][0]["integration_line_m"], dtype=np.float64)
    normal_index = int(np.argmax(np.abs(positive - negative)))
    center = 0.5 * (positive + negative)
    size = np.zeros(3)
    bounds = {}
    for index, nodes in enumerate(axes):
        halves = 0.5 * (nodes[:-1] + nodes[1:])
        if index == normal_index:
            candidates = halves[(halves > min(positive[index], negative[index])) &
                                (halves < max(positive[index], negative[index]))]
            center[index] = candidates[np.argmin(np.abs(candidates - center[index]))]
        else:
            node_index = int(np.argmin(np.abs(nodes - center[index])))
            low, high = halves[max(0, node_index - 2)], halves[min(len(halves) - 1, node_index + 1)]
            center[index], size[index] = 0.5 * (low + high), high - low
            bounds[index] = (low, high)
    scene = mw.Scene(
        domain=mw.Domain(bounds=[(axis[0], axis[-1]) for axis in axes]),
        grid=mw.GridSpec.custom(*axes), boundary=mw.BoundarySpec.none(), device="cpu",
    )
    port = mw.LumpedPort(
        "feed", positive=positive, negative=negative, voltage_path=mw.AxisPath("xyz"[normal_index]),
        current_surface=mw.Box(position=torch.tensor(center, dtype=dtype), size=torch.tensor(size, dtype=dtype)),
        reference_plane=float(center[normal_index]),
    )
    return scene, port, bounds


@pytest.mark.parametrize("scene_name", ["pcb-flexible", "pcb-rigid", "metamaterial-solid", "metamaterial-hollow", "metamaterial-open-top"])
@pytest.mark.usefixtures("site_scenes")
def test_public_scene_float32_contours_have_the_same_identity_as_float64(scene_name):
    data = json.loads((_SCENES / f"metahz-scene-{scene_name}.json").read_text())
    scene, port32, bounds = _site_port(data, dtype=torch.float32)
    _, port64, _ = _site_port(data, dtype=torch.float64)
    expected = compile_port_geometry(scene, port64)
    _assert_same_geometry(compile_port_geometry(scene, port32), expected)
    if scene_name == "pcb-flexible":
        lower = float(port32.current_surface.position[2]) - 0.5 * float(port32.current_surface.size[2])
        assert bounds[2][0] == 25e-6
        assert abs(lower - bounds[2][0]) > 3.4e-11  # Original minimum-spacing tolerance failed.


def _synthetic_scene_port(*, axis="z", offset=0.0, dtype=torch.float64, reverse=False):
    nodes = np.array([-0.001, 0.0, 18e-6, 50e-6, 0.0005, 0.001]) + offset
    halves = 0.5 * (nodes[:-1] + nodes[1:])
    axis_index = "xyz".index(axis)
    negative, positive = np.full(3, nodes[2]), np.full(3, nodes[2])
    negative[axis_index], positive[axis_index] = nodes[1], nodes[4]
    center = np.full(3, 0.5 * (halves[1] + halves[4]))
    size = np.full(3, halves[4] - halves[1])
    center[axis_index], size[axis_index] = halves[2], 0.0
    if reverse:
        positive, negative = negative, positive
    scene = mw.Scene(
        domain=mw.Domain(bounds=((nodes[0], nodes[-1]),) * 3),
        grid=mw.GridSpec.custom(nodes, nodes, nodes), boundary=mw.BoundarySpec.none(), device="cpu",
    )
    port = mw.LumpedPort(
        "feed", positive=torch.tensor(positive, dtype=dtype), negative=torch.tensor(negative, dtype=dtype),
        voltage_path=mw.AxisPath(axis),
        current_surface=mw.Box(position=torch.tensor(center, dtype=dtype), size=torch.tensor(size, dtype=dtype)),
        reference_plane=float(halves[2]),
    )
    return scene, port


@pytest.mark.parametrize("axis", "xyz")
@pytest.mark.parametrize("offset", [0.0, -0.025, 0.025])
@pytest.mark.parametrize("reverse", [False, True])
def test_nonuniform_contours_preserve_polarity_under_translation_and_source_dtype(axis, offset, reverse):
    scene, port32 = _synthetic_scene_port(axis=axis, offset=offset, dtype=torch.float32, reverse=reverse)
    _, port64 = _synthetic_scene_port(axis=axis, offset=offset, reverse=reverse)
    actual = compile_port_geometry(scene, port32)
    _assert_same_geometry(actual, compile_port_geometry(scene, port64))
    assert actual.direction == (-1 if reverse else 1)
    assert actual.voltage_weights.dtype == torch.float64
    assert len(actual.voltage_indices) == 3


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_saved_reconstructed_and_copied_ports_compile_identically(tmp_path, dtype):
    scene, port = _synthetic_scene_port(dtype=dtype)
    expected = compile_port_geometry(scene, port)
    # There is no public Scene snapshot API; preserve the actual port object and
    # tensor geometry here, independently of Result's field-only snapshots.
    path = tmp_path / "port.pt"
    torch.save(port, path)
    restored = torch.load(path, weights_only=False)
    assert restored.current_surface.position.dtype == dtype
    assert restored.current_surface.size.dtype == dtype
    for candidate in (restored, copy.deepcopy(port), port):
        _assert_same_geometry(compile_port_geometry(prepare_scene(scene), candidate), expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numpy_terminal_coordinate_precision_is_retained(dtype):
    scene, template = _synthetic_scene_port()
    port = mw.LumpedPort(
        "numpy", positive=np.asarray(template.positive, dtype=dtype),
        negative=np.asarray(template.negative, dtype=dtype), voltage_path=template.voltage_path,
        current_surface=template.current_surface,
    )
    assert torch.equal(compile_port_geometry(scene, port).voltage_indices,
                       compile_port_geometry(scene, template).voltage_indices)


def test_double_precision_contours_retain_identity_at_large_global_offset():
    scene, port = _synthetic_scene_port(offset=1000.0)
    compiled = compile_port_geometry(scene, port)
    assert len(compiled.voltage_indices) == 3
    assert port.current_surface.position.dtype == torch.float64
    assert port.current_surface.size.dtype == torch.float64


def test_insufficient_precision_at_large_offset_is_rejected_instead_of_merging_layers():
    scene, port = _synthetic_scene_port(offset=1000.0, dtype=torch.float32)
    with pytest.raises(ValueError, match="feed.*ambiguous coordinate precision.*float64"):
        compile_port_geometry(scene, port)


@pytest.mark.parametrize("offset", [0.0, 1000.0])
def test_snap_uses_local_spacing_and_never_merges_18_micron_layers(offset):
    coords = np.array([0.0, 18e-6, 36e-6, 1.0]) + offset
    assert _snap_index(coords, coords[0], label="layer") == 0
    assert _snap_index(coords, coords[1], label="layer") == 1
    with pytest.raises(ValueError, match="off-grid"):
        _snap_index(coords, 0.5 * (coords[0] + coords[1]), label="layer")
    with pytest.raises(ValueError, match="off-grid"):
        _snap_index(coords, coords[1] + 1e-8, label="layer")


def test_float64_off_grid_input_is_not_treated_as_float32_roundoff():
    scene, port = _synthetic_scene_port()
    port.current_surface.position[0] += 1e-10
    with pytest.raises(ValueError, match="feed.*half-grid.*off-grid"):
        compile_port_geometry(scene, port)


def test_nonzero_surface_thickness_cannot_be_silently_flattened():
    scene, port = _synthetic_scene_port()
    port.current_surface.size[2] = 1e-14
    with pytest.raises(ValueError, match="planar"):
        compile_port_geometry(scene, port)


@pytest.mark.parametrize("grid_kind", ["uniform", "custom", "auto"])
@pytest.mark.parametrize("axis", "xyz")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_contours_use_actual_prepared_yee_coordinates_for_every_grid_kind(grid_kind, axis, dtype):
    nodes = np.array([0.0, 0.0001, 0.00025, 0.00045, 0.0007, 0.001])
    grid = {
        "uniform": mw.GridSpec.uniform(0.0001),
        "custom": mw.GridSpec.custom(nodes, nodes, nodes),
        "auto": mw.GridSpec.auto(wavelength=0.001, min_steps_per_wavelength=10),
    }[grid_kind]
    scene = prepare_scene(mw.Scene(
        domain=mw.Domain(bounds=((0.0, 0.001),) * 3), grid=grid,
        boundary=mw.BoundarySpec.none(), device="cpu",
    ))
    axis_index = "xyz".index(axis)
    negative, positive, center, size = [], [], [], []
    for index, coordinate_axis in enumerate("xyz"):
        coordinate_nodes = getattr(scene, f"{coordinate_axis}_nodes64")
        halves = getattr(scene, f"{coordinate_axis}_half64")
        negative.append(coordinate_nodes[1] if index == axis_index else coordinate_nodes[2])
        positive.append(coordinate_nodes[-2] if index == axis_index else coordinate_nodes[2])
        center.append(halves[1] if index == axis_index else 0.5 * (halves[0] + halves[-1]))
        size.append(0.0 if index == axis_index else halves[-1] - halves[0])
    port = mw.LumpedPort(
        "actual-grid", negative=torch.tensor(negative, dtype=dtype), positive=torch.tensor(positive, dtype=dtype),
        voltage_path=mw.AxisPath(axis),
        current_surface=mw.Box(position=torch.tensor(center, dtype=dtype), size=torch.tensor(size, dtype=dtype)),
    )
    compiled = compile_port_geometry(scene, port)
    assert compiled.direction == 1
    assert torch.sum(compiled.voltage_weights).item() == pytest.approx(positive[axis_index] - negative[axis_index])
    expected_normal_index = 1  # Chosen from actual halves, not the multigap midpoint.
    for indices in compiled.current_indices:
        assert torch.all(indices[:, axis_index] == expected_normal_index)


@pytest.mark.parametrize("offset", [0.0, 1000.0])
def test_scene_placement_and_mesh_transform_preserve_float64_geometry(offset):
    position = torch.tensor([offset + 25e-6, offset + 18e-6, offset + 35e-6], dtype=torch.float64)
    size = torch.tensor([18e-6, 35e-6, 0.0], dtype=torch.float64)
    geometry = mw.Box(position=position, size=size)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((offset - 0.001, offset + 0.001),) * 3),
        grid=mw.GridSpec.uniform(0.0001), boundary=mw.BoundarySpec.none(), device="cpu",
        structures=[mw.Structure(geometry=geometry, material=mw.Material.pec())],
    )
    placed = prepare_scene(scene).structures[0].geometry
    assert placed.position.dtype == torch.float64
    assert placed.size.dtype == torch.float64
    torch.testing.assert_close(placed.position, position, rtol=0, atol=0)
    torch.testing.assert_close(placed.size, size, rtol=0, atol=0)
    local_vertices = torch.tensor([[0.0, 0.0, 0.0], [18e-6, 35e-6, 0.0]], dtype=torch.float64)
    transformed = placed._transform_mesh_verts(local_vertices)
    assert transformed.dtype == torch.float64
    torch.testing.assert_close(transformed, local_vertices + position, rtol=0, atol=0)


@pytest.mark.parametrize("error", [1e-10, 9e-6])
def test_true_double_precision_voltage_endpoint_offset_is_rejected(error):
    scene, template = _synthetic_scene_port()
    positive = np.array(template.positive)
    positive[2] += error
    port = mw.LumpedPort(
        "bad-endpoint", positive=positive, negative=template.negative, voltage_path=template.voltage_path,
        current_surface=template.current_surface,
    )
    with pytest.raises(ValueError, match="bad-endpoint.*Yee z node.*off-grid"):
        compile_port_geometry(scene, port)


def test_terminal_contour_preserves_original_footprint_roundoff():
    structures = [mw.Structure(
        mw.Box(position=(0, 0, z), size=(.015, .015, .005)), mw.Material.pec(), name=name,
    ) for name, z in (("signal", .0125), ("ground", -.0125))]
    port = mw.TerminalPort("feed", positive_terminal=mw.TerminalRef("signal"),
        negative_terminal=mw.TerminalRef("ground"), integration_path=mw.AxisPath("z"), reference_plane=.0025)
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.02, .02),) * 3),
        grid=mw.GridSpec.uniform(.005), boundary=mw.BoundarySpec.none(),
        structures=structures, ports=[port], device="cpu")
    compiled = compile_port_geometry(scene, scene.ports[0])
    assert all(indices.numel() for indices in compiled.current_indices)
    for indices in compiled.current_indices:
        assert torch.all(indices[:, 2] == 4)


def test_uniform_grid_zero_cancellation_retains_strict_offgrid_rejection():
    from witwin.maxwell.compiler.ports import _snap_index
    # This construction matches the symmetric 0.205 m coax aperture grid.
    nodes = np.linspace(-.205,.205,165,dtype=np.float64)
    nodes[82] = 2.7755575615628914e-17
    assert _snap_index(nodes,0.,label='z node')==82
    with pytest.raises(ValueError,match='off-grid'):
        _snap_index(nodes,1e-12,label='z node')
