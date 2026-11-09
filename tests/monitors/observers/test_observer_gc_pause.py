"""CPU-only GC state and exact observer layout/readout contracts."""

from contextlib import nullcontext
import gc

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell._gc import paused_cyclic_gc
from witwin.maxwell.compiler import monitors as compiler
from witwin.maxwell.fdtd import observers
from tests.monitors.observers.test_point_observer_batch_resolve import (
    COMPONENTS, assert_exact, fill_groups, grid_scene, make_solver,
)


@pytest.fixture(autouse=True)
def cpu_and_gc_state(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Observer GC tests must never initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", fail)
    enabled = gc.isenabled()
    try:
        yield
    finally:
        (gc.enable if enabled else gc.disable)()


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("raises", [False, True])
def test_pause_restores_entry_state_and_never_collects(monkeypatch, enabled, raises):
    def fail():
        pytest.fail("The pause must never force collection")

    monkeypatch.setattr(gc, "collect", fail)
    (gc.enable if enabled else gc.disable)()
    with pytest.raises(RuntimeError, match="inside") if raises else nullcontext():
        with paused_cyclic_gc():
            assert not gc.isenabled()
            if raises:
                raise RuntimeError("inside")
    assert gc.isenabled() is enabled


@pytest.mark.parametrize("enabled", [True, False])
def test_nested_pause_restores_outer_state(monkeypatch, enabled):
    monkeypatch.setattr(gc, "collect", lambda: pytest.fail("Unexpected collection"))
    (gc.enable if enabled else gc.disable)()
    with paused_cyclic_gc():
        assert not gc.isenabled()
        with paused_cyclic_gc():
            assert not gc.isenabled()
        assert not gc.isenabled()
        with pytest.raises(RuntimeError, match="nested"):
            with paused_cyclic_gc():
                assert not gc.isenabled()
                raise RuntimeError("nested")
        assert not gc.isenabled()
    assert gc.isenabled() is enabled


def point_scene():
    scene = grid_scene()
    for index in range(8):
        scene.add_monitor(mw.PointMonitor(
            f"point_{index}", ((index - 4) / 32, .021, -.013), fields=COMPONENTS,
        ))
    return scene


@pytest.mark.parametrize("stage", ["compile", "prepare", "readout"])
@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("raises", [False, True])
def test_bulk_helpers_pause_and_restore(monkeypatch, stage, enabled, raises):
    scene = point_scene()
    solver = make_solver(scene)
    observers.prepare_observers(solver, (.8e9, 1.e9), "none", 8)
    fill_groups(solver)
    if stage == "compile":
        module, name = compiler, "normalize_component"
        operation = lambda: compiler.compile_fdtd_observers(scene)
    elif stage == "prepare":
        module, name = observers, "_resolve_point_observers"
        operation = lambda: observers.prepare_observers(solver, (.8e9, 1.e9), "none", 8)
    else:
        module, name = observers, "combine_complex_spectral_components"
        operation = lambda: observers.get_observer_results(solver)
    original = getattr(module, name)
    seen = []

    def inner(*args, **kwargs):
        seen.append(gc.isenabled())
        if raises:
            raise RuntimeError("bulk helper")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, inner)
    (gc.enable if enabled else gc.disable)()
    with pytest.raises(RuntimeError, match="bulk helper") if raises else nullcontext():
        operation()
    assert seen and not any(seen)
    assert gc.isenabled() is enabled


def test_records_groups_and_results_equal_without_pause(monkeypatch):
    scene = point_scene()
    actual, expected = make_solver(scene), make_solver(scene)
    gc.enable()
    actual.observers = compiler.compile_fdtd_observers(scene)
    observers.prepare_observers(actual, (.8e9, 1.e9), "none", 8)
    fill_groups(actual)
    results = observers.get_observer_results(actual)
    assert gc.isenabled()
    with monkeypatch.context() as patch:
        patch.setattr(compiler, "paused_cyclic_gc", nullcontext)
        patch.setattr(observers, "paused_cyclic_gc", nullcontext)
        expected.observers = compiler.compile_fdtd_observers(scene)
        observers.prepare_observers(expected, (.8e9, 1.e9), "none", 8)
        fill_groups(expected)
        expected_results = observers.get_observer_results(expected)
    assert gc.isenabled()
    assert_exact(actual.observers, expected.observers)
    for name in ("_point_observer_groups", "_plane_observer_groups", "_observer_spectral_entries",
                 "_observer_point_groups_by_frequency", "_observer_plane_groups_by_frequency"):
        assert_exact(getattr(actual, name), getattr(expected, name))
    assert_exact(results, expected_results)
