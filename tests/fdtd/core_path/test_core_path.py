"""Core admission and fresh-interpreter bit identity."""
from pathlib import Path
import os
import re
import subprocess
import sys
from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch

import witwin.maxwell as mw
from benchmark.core_path.scenes import CORE_SCENES, DISPERSIVE_SCENES


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Core path requires CUDA.")


def _worker(kind, destination, arm):
    from benchmark.core_path.scenes import W1_SCENES, build_core_case
    from benchmark.core_path.gates import make_simulation
    from witwin.maxwell.fdtd.runtime.core_path import core_path_blockers, core_path_step_loop_blockers

    if kind == "closed_transverse":
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.1, 0.1),) * 3),
            grid=mw.GridSpec.uniform(.05), device="cuda",
            boundary=mw.BoundarySpec.faces(x="periodic", y="pec", z="pml", num_layers=2),
        )
        simulation = mw.Simulation.fdtd(scene, frequency=1e9, absorber="cpml",
                                        run_time=mw.TimeConfig(time_steps=1))
        label = "closed-transverse CPML (passivity guard)"
        solver = simulation.prepare().solver
        assert label in core_path_blockers(solver)
        assert not hasattr(solver, "_core_runtime")
        simulation.config.fdtd_core_path = True
        with pytest.raises(ValueError, match=re.escape(label)):
            simulation.prepare()
        return
    chunk = int(kind.removeprefix("chunk_")) if kind.startswith("chunk_") else 32
    failure = kind in ("capture_failure", "w2_capture_failure")
    host_observer = kind == "host_observer"
    case_name = ("w1_sheet_feed" if kind.startswith("w2_") else
                 "cpml_dense" if host_observer else "o2_capture_failure" if failure
                 else "o2_remainder" if kind.startswith("chunk_") else kind)
    case = build_core_case("cpml_dense" if kind == "existing" else case_name)
    if kind.startswith("w2_"):
        from dataclasses import replace
        case = replace(case, initial_fields=True)
    if host_observer:
        from dataclasses import replace
        case = replace(case, scene=case.scene.clone(monitors=[mw.PointMonitor(
            "point", position=(0.2, 0.1, 0.1), fields=("Ez",))]))
    simulation = make_simulation(case, arm=arm, steps=64, full_field_dft=failure or kind.startswith("chunk_"))
    prepared = simulation.prepare()
    solver = prepared.solver
    if kind == "w1_copper_patch":
        assert solver._copper_foils
    if kind in ("w1_thin_sheet_cavity", "w1_sheet_feed"):
        assert solver._thin_sheets
        assert any(sheet.feeds for sheet in solver._thin_sheets) is (kind == "w1_sheet_feed")
    if case.shape is not None:
        assert (solver.Nx, solver.Ny, solver.Nz) == case.shape
    if kind == "existing":
        assert not hasattr(solver, "_core_runtime")
    else:
        assert hasattr(solver, "_core_runtime") == (arm == "candidate")
        assert not core_path_blockers(solver)
        if arm == "candidate":
            with pytest.raises(ValueError, match="fdtd_core_path=True does not support resume / non-initializing prepare"):
                simulation._build_fdtd_solver(initialize=False)
    if case.initial_fields:
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            getattr(solver, name).fill_(2.0 ** -20)
    def fail_capture(_solver, step, count):
        step()
        raise RuntimeError("Forced capture failure with mutated nonzero fields and spectra.")
    if arm == "candidate":
        solver._core_runtime.chunk_size = chunk
    context = patch.object(solver._core_runtime, "capture_chunk", side_effect=fail_capture) if failure and arm == "candidate" else nullcontext()
    from witwin.maxwell.fdtd import observers as observer_module
    built_engine = []
    make_engine = observer_module.make_single_observer_engine

    def record_engine(_solver, **kwargs):
        run = make_engine(_solver, **kwargs)
        built_engine.append(_solver._single_observer_engine is not None)
        return run

    with context, patch.object(observer_module, "make_single_observer_engine", side_effect=record_engine):
        result = prepared.run()
    torch.cuda.synchronize()
    # The existing path keeps host-evaluated magnetic surface sources eager; the core arm captures whole steps.
    assert solver._cuda_graph_active is (arm == "candidate" or not solver._magnetic_source_terms)
    if arm == "candidate":
        step_loop = bool(core_path_step_loop_blockers(solver))
        expected_active = (not failure and not step_loop and not solver.time_observers_enabled
                           and (not solver.observers_enabled or all(built_engine)
                                or observer_module.uses_native_point_observers(solver)))
        assert solver._core_step_graph_active is expected_active
        if step_loop:
            assert solver._core_runtime is not None
            assert solver._core_step_graph_k is None
            assert solver._core_step_graph_chunks == []
        elif expected_active:
            assert solver._core_step_graph_k == chunk
    if destination:
        from benchmark.core_path.gates import tensors
        payload = tensors({name: getattr(solver, name) for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}, "fields")
        payload.update(tensors(result.fields, "phasors"))
        payload.update(tensors(result.monitors, "monitors"))
        if case.name in W1_SCENES + DISPERSIVE_SCENES:
            from benchmark.core_path.gates import physics_tensors
            payload.update(physics_tensors(solver, result))
        torch.save(payload, destination)


def _run_worker(kind, destination="", arm="baseline"):
    subprocess.run([sys.executable, str(Path(__file__).resolve()),
                    kind, str(destination), arm], check=True,
                   env=os.environ | {"PYTHONPATH": str(Path(__file__).resolve().parents[3])})


def test_closed_transverse_cpml_blocked():
    _run_worker("closed_transverse")


def test_explicit_false_uses_existing_path():
    _run_worker("existing")


@pytest.mark.parametrize("scene", ["cpml_dense", "cpml_compressed", "closed_surface_box", "absorber_pml",
                                   "absorber_graded", "closed_transverse_graded",
                                   *(name for name in DISPERSIVE_SCENES if name in CORE_SCENES)])
def test_core_fields_equal(scene, tmp_path):
    payloads = []
    for arm in ("baseline", "candidate"):
        destination = tmp_path / f"{arm}.pt"
        _run_worker(scene, destination, arm)
        payloads.append(torch.load(destination, weights_only=True))
    for name in payloads[0]:
        assert torch.equal(payloads[0][name], payloads[1][name]), name


@pytest.mark.parametrize("scene", ["w1_copper_patch", "w1_thin_sheet_cavity", "w1_sheet_feed"])
def test_w1_fields_equal_with_whole_step_graph(scene, tmp_path):
    payloads = []
    for arm in ("baseline", "candidate"):
        destination = tmp_path / f"{arm}.pt"
        _run_worker(scene, destination, arm)
        payloads.append(torch.load(destination, weights_only=True))
    assert payloads[0].keys() == payloads[1].keys()
    assert any(torch.count_nonzero(value) for key, value in payloads[0].items() if key.startswith("fields/"))
    for name in payloads[0]:
        assert torch.equal(payloads[0][name], payloads[1][name]), name


@pytest.mark.parametrize("kind", ["chunk_1", "chunk_16", "chunk_32", "capture_failure", "host_observer",
                                  "w2_nonzero", "w2_capture_failure"])
def test_chunk_and_capture_restore(kind, tmp_path):
    payloads = []
    for arm in ("baseline", "candidate"):
        destination = tmp_path / f"{arm}.pt"
        _run_worker(kind, destination, arm)
        payloads.append(torch.load(destination, weights_only=True))
    assert payloads[0].keys() == payloads[1].keys()
    for name in payloads[0]:
        assert torch.equal(payloads[0][name], payloads[1][name]), name


if __name__ == "__main__":
    _worker(*sys.argv[1:])
