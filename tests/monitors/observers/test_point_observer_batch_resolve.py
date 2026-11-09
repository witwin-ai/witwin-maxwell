"""Exact host resolution, ordinary observer layouts and result ordering."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler import monitors as compiler
from witwin.maxwell.fdtd import observers
from witwin.maxwell.fdtd.coords import component_coords
from witwin.maxwell.scene import prepare_scene
from tests.monitors.observers import _point_observer_reference as reference


COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch, request):
    if request.node.get_closest_marker("cuda"):
        return
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CPU observer test touched CUDA"))


def grid_scene():
    # Binary fractions make the midpoint ties exact on both primal and dual axes.
    axes = [np.array(values) / 32 for values in (
        (-16, -12, -8, -2, 0, 2, 8, 16), (-16, -8, -4, 0, 2, 4, 12, 16),
        (-16, -12, -4, -2, 0, 4, 8, 16))]
    return mw.Scene(domain=mw.Domain(bounds=((- .5, .5),) * 3),
                    grid=mw.GridSpec.custom(*axes), boundary=mw.BoundarySpec.pec(), device="cpu")


def make_solver(scene):
    return mw.Simulation.fdtd(
        scene, frequencies=(.8e9, 1.e9),
        backend="cpu_reference" if torch.device(scene.device).type == "cpu" else "native", cuda_graph=False,
        full_field_dft=False, run_time=mw.TimeConfig(time_steps=8),
        spectral_sampler=mw.SpectralSampler(window="none", normalize_source=False),
    ).prepare().solver


def assert_exact(actual, expected):
    assert type(actual) is type(expected), (type(actual), type(expected))
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.device == expected.device and torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert list(actual) == list(expected)
        for key in expected:
            assert_exact(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_exact(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize("component", COMPONENTS)
def test_batch_resolution_matches_scalar_nonuniform(component):
    solver = SimpleNamespace(scene=prepare_scene(grid_scene()))
    coords = component_coords(solver.scene, component)
    positions = [tuple(axis[0] for axis in coords), tuple(axis[-1] for axis in coords),
                 (-1., -1., -1.), (1., 1., 1.)]
    # Exercise a tie on every axis independently and simultaneously.
    midpoint = tuple(float((axis[2] + axis[3]) / 2) for axis in coords)
    positions.append(midpoint)
    for dim in range(3):
        point = [float(axis[1]) for axis in coords]
        point[dim] = midpoint[dim]
        positions.append(tuple(point))
    positions += positions[:]
    expected = [observers.resolve_point_observer(solver, dict(component=component, position=p)) for p in positions]
    assert expected[4] == (2, 2, 2)
    # Force several chunks, independently of the real 32 MiB limit.
    with patch.object(observers, "component_coords", wraps=component_coords) as call:
        actual = observers._resolve_point_observers(solver, component, positions, max_bytes=256)
    assert call.call_count == 1
    assert_exact(actual, expected)


@pytest.mark.parametrize("position", [
    (np.float32(.13), np.float32(-.07), np.float32(.23)),
    (torch.tensor(.13), torch.tensor(-.07), torch.tensor(.23)),
])
def test_public_position_conversion_is_preserved(position):
    monitor = mw.PointMonitor("converted", position, fields=COMPONENTS)
    assert all(type(value) is float for value in monitor.position)
    assert monitor.position == tuple(float(value) for value in position)
    solver = SimpleNamespace(scene=prepare_scene(grid_scene()))
    for component in COMPONENTS:
        expected = observers.resolve_point_observer(solver, dict(component=component, position=monitor.position))
        assert observers._resolve_point_observers(solver, component, [monitor.position]) == [expected]


def mixed_scene():
    scene = grid_scene()
    scene.add_source(mw.PointDipole(name="source", position=(-.4, -.4, -.4),
        polarization=(.6, 0., .8), profile="ideal", source_time=mw.CW(frequency=1.e9)))
    sheet = mw.SurfaceMesh([[-.25, -.25, .25], [.25, -.25, .25],
                            [.25, .25, .25], [-.25, .25, .25]],
                           [(0, 1, 2), (0, 2, 3)], sheet_model="conformal_two_sided")
    scene.add_structure(mw.Structure(sheet, mw.Material.pec()))
    for i in range(24):
        if i == 7:
            scene.add_monitor(mw.PlaneMonitor("plane", axis="z", position=.013, fields=("Ex", "Hy"),
                                              frequencies=(1.1e9, .8e9)))
        scene.add_monitor(mw.PointMonitor(f"point_{i}", ((i % 6 - 3) / 32, .021, -.013), fields=COMPONENTS))
    scene.add_monitor(mw.DipoleEmissionMonitor("emission", "source", frequencies=(1.e9, 1.1e9)))
    scene.add_monitor(mw.FluxMonitor("flux", axis="z", position=.02, frequencies=(.8e9,)))
    scene.add_monitor(mw.PointMonitor("sheet_probe", (.0, .0, .0), fields=("Ez",)))
    scene.add_monitor(mw.PointMonitor("sheet_surface", (.0, .0, .25), fields=("Ex",)))
    scene.add_monitor(mw.FieldTimeMonitor("time", position=(0., 0., 0.), components=("Ex", "Ez")))
    return scene


def compiled_records(scene, compile_function):
    records = compile_function(scene)
    for record in records:
        if record.get("monitor_name") in ("point_2", "point_3"):
            # Distinct mutable inputs with equal values must produce immutable,
            # value-cached metadata; PointMonitor itself has no frequency option.
            record["monitor_frequencies"] = [1.1e9, .8e9]
    return records


def fill_groups(solver):
    for groups in (solver._point_observer_groups, solver._plane_observer_groups):
        for group in groups.values():
            for name in ("real", "imag", "aux_real", "aux_imag"):
                if name in group:
                    data = group[name]
                    data.copy_(torch.arange(data.numel(), dtype=data.dtype).reshape(data.shape) / 37 + .125)
    for index, entry in enumerate(solver._observer_spectral_entries):
        entry.update(sample_count=3 + index, window_normalization=2.5 + index,
                     source_dft_real=.7 + index, source_dft_imag=-.3)


@pytest.mark.parametrize("complex_fields,dtype", [(False, torch.float32), (False, torch.float64), (True, torch.float32)])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=[pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")])])
def test_original_prepare_layout_results_and_checkpoint(complex_fields, dtype, device):
    from witwin.maxwell.fdtd.observation_resume import capture_observations, preflight_observations

    scene = mixed_scene().clone(device=device)
    expected = make_solver(scene)
    actual = make_solver(scene)
    original_records = compiled_records(scene, reference.compile_fdtd_observers)
    new_records = compiled_records(scene, compiler.compile_fdtd_observers)
    assert_exact(new_records, original_records)
    for solver, records, prepare in ((expected, original_records, reference.prepare_observers),
                                     (actual, new_records, observers.prepare_observers)):
        solver.complex_fields_enabled = complex_fields
        solver._observer_dtype = dtype
        solver.observers = records
        prepare(solver, (.8e9, 1.e9), "none", 8)
        observers.prepare_time_observers(solver, 8)
    assert_exact(actual.observers, expected.observers)
    for name in ("_point_observer_groups", "_plane_observer_groups", "_observer_spectral_entries",
                 "_observer_point_groups_by_frequency", "_observer_plane_groups_by_frequency"):
        assert_exact(getattr(actual, name), getattr(expected, name))
    copies, updates = [], []
    preflight_observations(actual, capture_observations(expected), 0, copies, updates)
    assert copies
    # Mutable records and group lists remain private even when value caches hit.
    assert actual.observers[0] is not actual.observers[1]
    for record in actual.observers:
        assert type(record["frequencies"]) is tuple
        assert type(record["global_freq_indices"]) is tuple
        if "field_index" in record:
            assert type(record["field_index"]) is tuple
            assert all(type(index) is int for index in record["field_index"])
    for solver in (expected, actual):
        fill_groups(solver)
    assert_exact(observers.get_observer_results(actual), reference.get_observer_results(expected))
    # Source-normalization arithmetic also retains its original results and order.
    for solver in (expected, actual):
        solver._normalize_source = True
        solver._normalization_source_time = object()
    assert_exact(observers.get_observer_results(actual), reference.get_observer_results(expected))


def test_time_and_thin_sheet_consumers_keep_scalar_resolution():
    from witwin.maxwell.fdtd.thin_sheet import _warn_side_observers

    solver = make_solver(mixed_scene())
    scalar = observers.resolve_point_observer
    with patch.object(observers, "resolve_point_observer", wraps=scalar) as call:
        observers.prepare_time_observers(solver, 8)
    assert call.call_count == 2
    for component, meta in solver.time_observers[0]["component_meta"].items():
        assert meta["field_index"] == scalar(solver, dict(component=component, position=(0., 0., 0.)))
    index = scalar(solver, dict(component="Ez", position=(0., 0., 0.)))
    runtime = SimpleNamespace(complex=SimpleNamespace(e_split=np.array([True]),
        e_component=np.array([2]), e_index=np.array([index]), h_extra=np.array([], dtype=int)))
    with patch.object(observers, "resolve_point_observer", wraps=scalar) as call:
        with pytest.warns(RuntimeWarning, match="sheet_probe"):
            _warn_side_observers(solver, (runtime,))
    assert call.call_count == sum(1 for record in solver.observers if record["kind"] == "point") + 2


def test_component_work_and_spectral_entry_order():
    scene = mixed_scene()
    solver = make_solver(scene)
    solver.observers = compiled_records(scene, compiler.compile_fdtd_observers)
    solver.observers = [record for record in solver.observers if record["kind"] == "point"]
    with patch.object(observers, "component_coords", wraps=component_coords) as coords:
        with patch.object(observers, "build_observer_spectral_entry", wraps=observers.build_observer_spectral_entry) as entry:
            observers.prepare_observers(solver, (.8e9, 1.e9), "none", 8)
    assert coords.call_count == 6
    assert [call.args[1] for call in entry.call_args_list] == [.8e9, 1.e9, 1.1e9]
