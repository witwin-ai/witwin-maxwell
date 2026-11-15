"""Core admission and fresh-interpreter bit identity against the frozen existing path."""
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
from benchmark.core_path.scenes import CORE_SCENES, DISPERSIVE_SCENES, W7R5_SCENES
from tests.support import frozen_oracles


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Core path requires CUDA.")


# A failed whole-step capture raises after restoring the solver state (W7 row 16c);
# a fresh preparation then runs the scene from zero fields.
CAPTURE_FAILURE_KINDS = ("capture_failure", "w2_capture_failure")
FAILURE_REASON = "Forced capture failure with mutated nonzero fields and spectra."


def _worker(kind, destination):
    from benchmark.core_path.scenes import W1_SCENES, build_core_case
    from benchmark.core_path.gates import make_simulation
    from witwin.maxwell.fdtd.runtime.core_path import core_path_step_loop_blockers

    if kind == "closed_transverse":
        scene = mw.Scene(
                    domain=mw.Domain(bounds=((-0.1, 0.1),) * 3),
                    grid=mw.GridSpec.uniform(.05), boundary=mw.BoundarySpec.faces(x="periodic", y="pec", z="pml", num_layers=2),
                    sources=(
                mw.PointDipole(position=(0., 0., 0.), polarization="Ez",
                               source_time=mw.GaussianPulse(frequency=1e9, fwidth=.5e9)),
                ),
                    monitors=(mw.PointMonitor("point", position=(.05, 0., 0.), fields=("Ez",)),),
                )
        simulation = mw.Simulation.fdtd(scene, frequencies=1e9, config=mw.FDTDConfig(absorber="cpml",
                                        run_time=mw.TimeConfig(time_steps=64)))
        # The passivity guard samples at K-chunk boundaries, so explicit CPML here is admitted.
        prepared = simulation.prepare()
        solver = prepared.solver
        assert solver._core_runtime is not None and hasattr(solver, "_core_runtime")
        result = prepared.run()
        assert solver._core_step_graph_active
        assert result.metadata["passivity_guard"] == "active"
        assert solver._passivity_state["peak"] > 0  # sampled at steps 32 and 64
        return
    reference = kind.endswith("_reference")
    kind = kind.removesuffix("_reference")
    chunk = int(kind.removeprefix("chunk_")) if kind.startswith("chunk_") else 32
    failure = kind in CAPTURE_FAILURE_KINDS and not reference
    host_observer = kind == "host_observer"
    case_name = ("w1_sheet_feed" if kind.startswith("w2_") else
                 "cpml_dense" if host_observer else "o2_capture_failure" if kind == "capture_failure"
                 else "o2_remainder" if kind.startswith("chunk_") else kind)
    case = build_core_case(case_name)
    if kind.startswith("w2_"):
        from dataclasses import replace
        case = replace(case, initial_fields=True)
    if host_observer:
        from dataclasses import replace
        case = replace(case, scene=case.scene.updated(monitors=[mw.PointMonitor(
            "point", position=(0.2, 0.1, 0.1), fields=("Ez",))]))
    simulation = make_simulation(case, steps=64, full_field_dft=kind in CAPTURE_FAILURE_KINDS or kind.startswith("chunk_"))
    prepared = simulation.prepare()
    solver = prepared.solver
    if kind == "w1_copper_patch":
        assert solver._copper_foils
    if kind in ("w1_thin_sheet_cavity", "w1_sheet_feed"):
        assert solver._thin_sheets
        assert any(sheet.feeds for sheet in solver._thin_sheets) is (kind == "w1_sheet_feed")
    if case.shape is not None:
        assert (solver.Nx, solver.Ny, solver.Nz) == case.shape
    assert hasattr(solver, "_core_runtime")
    if case.initial_fields and not reference:
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            getattr(solver, name).fill_(2.0 ** -20)
    def fail_capture(_solver, step, count):
        step()
        raise RuntimeError(FAILURE_REASON)
    solver._core_runtime.chunk_size = chunk
    context = patch.object(solver._core_runtime, "capture_chunk", side_effect=fail_capture) if failure else nullcontext()
    from witwin.maxwell.fdtd import observers as observer_module
    built_engine = []
    make_engine = observer_module.make_single_observer_engine

    def record_engine(_solver, **kwargs):
        run = make_engine(_solver, **kwargs)
        built_engine.append(_solver._single_observer_engine is not None)
        return run

    if failure:
        before = {name: getattr(solver, name).clone() for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
        with context, pytest.raises(RuntimeError, match=f"^{re.escape(FAILURE_REASON)}$"):
            prepared.run()
        # The failed attempt restored the pre-run state (the worker's nonzero fill).
        assert any(torch.count_nonzero(value) for value in before.values())
        for name, value in before.items():
            assert torch.equal(getattr(solver, name), value), name
        assert not solver._core_step_graph_active
        prepared = simulation.prepare()
        solver = prepared.solver
        solver._core_runtime.chunk_size = chunk
        result = prepared.run()
    else:
        with patch.object(observer_module, "make_single_observer_engine", side_effect=record_engine):
            result = prepared.run()
    torch.cuda.synchronize()
    assert result.solver is solver and solver._core_runtime is not None
    if not failure:
        assert solver._cuda_graph_active
        step_loop = bool(core_path_step_loop_blockers(solver))
        expected_active = (not step_loop and not solver.time_observers_enabled
                           and (not solver.observers_enabled or all(built_engine)
                                or observer_module.uses_indexed_observers(solver)))
        assert solver._core_step_graph_active is expected_active
        if step_loop:
            assert solver._core_runtime is not None
            assert solver._core_step_graph_k is None
            assert solver._core_step_graph_chunks == []
        elif expected_active:
            assert solver._core_step_graph_k == chunk
    if destination:
        from benchmark.core_path.gates import api_dialect, tensors
        payload = tensors({name: getattr(solver, name) for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}, "fields")
        payload.update(tensors(api_dialect.fields(result), "phasors"))
        payload.update(tensors(api_dialect.monitors(result), "monitors"))
        if case.name in W1_SCENES + DISPERSIVE_SCENES:
            from benchmark.core_path.gates import physics_tensors
            payload.update(physics_tensors(solver, result))
        torch.save(payload, destination)


def _run_worker(kind, destination=""):
    subprocess.run([sys.executable, str(Path(__file__).resolve()),
                    kind, str(destination)], check=True,
                   env=os.environ | {"PYTHONPATH": str(Path(__file__).resolve().parents[3])})


def _payloads(kind, tmp_path):
    """The frozen existing-path payload and a fresh-interpreter default-path payload."""
    expected = frozen_oracles.load(f"tests/fdtd/core_path/test_core_path.py::{kind}")
    destination = tmp_path / "candidate.pt"
    _run_worker(kind, destination)
    return [expected, torch.load(destination, weights_only=True)]


def test_closed_transverse_cpml_admitted():
    _run_worker("closed_transverse")


@pytest.mark.parametrize("scene", ["cpml_dense", "cpml_compressed", "closed_surface_box", "absorber_pml",
                                   "absorber_graded", "closed_transverse_graded",
                                   # graphene_sheet (W7R5) is gated by G1 against the w7-row5 addendum.
                                   *(name for name in DISPERSIVE_SCENES
                                     if name in CORE_SCENES and name not in W7R5_SCENES)])
def test_core_fields_equal(scene, tmp_path):
    payloads = _payloads(scene, tmp_path)
    for name in payloads[0]:
        assert torch.equal(payloads[0][name], payloads[1][name]), name


@pytest.mark.parametrize("scene", ["w1_copper_patch", "w1_thin_sheet_cavity", "w1_sheet_feed"])
def test_w1_fields_equal_with_whole_step_graph(scene, tmp_path):
    payloads = _payloads(scene, tmp_path)
    assert payloads[0].keys() == payloads[1].keys()
    assert any(torch.count_nonzero(value) for key, value in payloads[0].items() if key.startswith("fields/"))
    for name in payloads[0]:
        assert torch.equal(payloads[0][name], payloads[1][name]), name


@pytest.mark.parametrize("kind", ["chunk_1", "chunk_16", "chunk_32", "host_observer", "w2_nonzero"])
def test_chunk_and_capture_restore(kind, tmp_path):
    payloads = _payloads(kind, tmp_path)
    assert payloads[0].keys() == payloads[1].keys()
    for name in payloads[0]:
        assert torch.equal(payloads[0][name], payloads[1][name]), name


@pytest.mark.parametrize("kind", CAPTURE_FAILURE_KINDS)
def test_capture_failure_raises_and_a_fresh_run_matches(kind, tmp_path):
    # Reference: the uninterrupted core run of the same scene from zero fields.
    _run_worker(kind + "_reference", tmp_path / "reference.pt")
    _run_worker(kind, tmp_path / "candidate.pt")
    expected, actual = (torch.load(tmp_path / name, weights_only=True) for name in ("reference.pt", "candidate.pt"))
    assert expected.keys() == actual.keys()
    assert any(torch.count_nonzero(value) for key, value in expected.items() if key.startswith("fields/"))
    for name in expected:
        assert torch.equal(expected[name], actual[name]), name


if __name__ == "__main__":
    _worker(*sys.argv[1:])
