"""Actual full-DFT counter rejection and disk-resume state contracts.

Controlled initial fields qualify discrete checkpoint behavior. These short
runs make no continuum propagation or attenuation claim.
"""
from collections.abc import Mapping
from enum import Enum

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd import observation_resume, resume
from witwin.maxwell.fdtd.checkpoint import capture_checkpoint_state, graph_state_tensors
from tests.fdtd.test_short_spectral_tail_execution import _scene as _short_copper_scene

FREQUENCIES = (0.8e9, 1.0e9, 1.2e9)
TOTAL_STEPS = 8


def _scene():
    # Reuse the qualified short-control physical scene without changing it.
    scene, _index = _short_copper_scene()
    return scene


def _execution(graph):
    simulation = mw.Simulation.fdtd(
        _scene().clone(device="cuda"),
        frequencies=FREQUENCIES,
        run_time=mw.TimeConfig(time_steps=TOTAL_STEPS),
        full_field_dft=True,
        cuda_graph=graph,
        spectral_sampler=mw.SpectralSampler(window="none", normalize_source=False),
    )
    execution = simulation.prepare()
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        getattr(execution.solver, name).zero_()
    execution.solver.Ey.fill_(1.0)
    execution.solver.Hx.fill_(0.25)
    assert execution.solver._copper_foils
    for entry in execution.solver._copper_foils:
        if entry["electric_name"] == "Ey":
            entry["right_e"].fill_(3.0)
    return execution


def _active_tables(solver, step):
    assert solver.dft_enabled and not solver.complex_fields_enabled
    assert len(solver._dft_entries) == len(FREQUENCIES)
    for name in ("_dft_weighted_cos_table", "_dft_weighted_sin_table"):
        table = getattr(solver, name)
        assert table.is_cuda and table.shape == (TOTAL_STEPS, len(FREQUENCIES))
    assert solver._dft_step.is_cuda and solver._dft_step.dtype == torch.int64
    assert solver._dft_step.shape == (1,) and int(solver._dft_step.item()) == step


def _bits(value):
    if isinstance(value, torch.Tensor):
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        return "tensor", str(value.dtype), tuple(value.shape), str(value.device), raw
    if isinstance(value, np.ndarray):
        return "numpy", str(value.dtype), value.shape, np.ascontiguousarray(value).tobytes()
    if isinstance(value, Enum):
        return _bits(value.value)
    if isinstance(value, Mapping):
        return tuple((key, _bits(item)) for key, item in sorted(value.items(), key=lambda pair: repr(pair[0])))
    if isinstance(value, (tuple, list)):
        return tuple(_bits(item) for item in value)
    if isinstance(value, float):
        return "float", value.hex()
    return value


def _snapshot(solver):
    physics = graph_state_tensors(solver)
    tables = {name: getattr(solver, name) for name in ("_dft_weighted_cos_table", "_dft_weighted_sin_table")}
    engine = getattr(solver, "_single_observer_engine", None)
    engine_state = None if engine is None else {"state": engine.state, "step": engine.step}
    return {
        "physical_tensor_pointers": {name: tensor.data_ptr() for name, tensor in physics.items()},
        "physical_live_tensors": _bits(physics),
        "physical_checkpoint_tensors": _bits(capture_checkpoint_state(solver, 0).tensors),
        "observations": _bits(observation_resume.capture_observations(solver)),
        "full_dft_host_entries": _bits(solver._dft_entries),
        "tables": _bits(tables),
        "table_pointers": {name: tensor.data_ptr() for name, tensor in tables.items()},
        "observer_engine": _bits(engine_state),
        "all_Cu_tensor_state_and_scratch": _bits([
            {key: value for key, value in entry.items() if isinstance(value, torch.Tensor)}
            for entry in solver._copper_foils
        ]),
        "complete_resume_auxiliary": _bits(
            resume.capture_resume_checkpoint(solver, step=0, total_steps=TOTAL_STEPS).auxiliary
        ),
    }


def _assert_final_parity(actual, expected):
    assert actual.stats()["steps_run"] == expected.stats()["steps_run"] == TOTAL_STEPS
    _active_tables(actual.solver, TOTAL_STEPS)
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        torch.testing.assert_close(getattr(actual.solver, name), getattr(expected.solver, name), rtol=0, atol=0)
    for frequency in FREQUENCIES:
        for component in ("Ex", "Ey", "Ez"):
            torch.testing.assert_close(
                actual.tensor(component, frequency=frequency), expected.tensor(component, frequency=frequency), rtol=0, atol=0
            )
    a_physics = capture_checkpoint_state(actual.solver, TOTAL_STEPS)
    b_physics = capture_checkpoint_state(expected.solver, TOTAL_STEPS)
    assert a_physics.schema == b_physics.schema and a_physics.tensors.keys() == b_physics.tensors.keys()
    for name in a_physics.tensors:
        torch.testing.assert_close(a_physics.tensors[name], b_physics.tensors[name], rtol=0, atol=0)
    for name in ("same_owned_Ey", "published_Ey"):
        for frequency in FREQUENCIES:
            a = actual.monitor(name, frequency=frequency)["components"]
            b = expected.monitor(name, frequency=frequency)["components"]
            for component in a:
                av = a[component]["data"] if isinstance(a[component], dict) else a[component]
                bv = b[component]["data"] if isinstance(b[component], dict) else b[component]
                torch.testing.assert_close(av, bv, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA execution required")
@pytest.mark.parametrize("graph", [False, True])
def test_actual_full_dft_counter_rejects_before_restore_and_disk_resume_matches_direct(graph, tmp_path, monkeypatch, record_property):
    direct = _execution(graph).run()
    _active_tables(direct.solver, TOTAL_STEPS)
    assert direct.solver._cuda_graph_active is graph
    assert direct.solver._tail_graph_active is graph
    assert direct.solver._observer_cuda_graph_active is graph
    rejected = 0
    for checkpoint_step in (1, 4):
        execution = _execution(graph)
        checkpoint = execution.run_until(checkpoint_step)
        _active_tables(execution.solver, checkpoint_step)
        assert checkpoint.step == checkpoint.physics.step == checkpoint_step
        assert bool((execution.solver.Ey != 0).any()) and bool((execution.solver.Hx != 0).any())
        assert any(bool((entry["state"] != 0).any()) for entry in execution.solver._copper_foils)
        copper_states = [value for name, value in checkpoint.physics.tensors.items()
                         if name.startswith("copper_foil_") and name.endswith("_state")]
        assert copper_states and any(bool((value != 0).any()) for value in copper_states)
        path = tmp_path / f"checkpoint-{checkpoint_step}.pt"
        checkpoint.save(path)
        del checkpoint, execution
        valid = mw.FDTDResumeCheckpoint.load(path, map_location="cpu")
        saved_counter = valid.auxiliary["observations"]["arrays"]["_dft_step"]
        assert saved_counter.device.type == "cpu" and int(saved_counter.item()) == checkpoint_step

        # Public zero-step execution prepares real observer buffers and both GPU
        # tables without advancing fields. The target is then deliberately
        # distinct from the saved physics, making an early restore detectable.
        target = _execution(graph)
        target.run_until(0)
        _active_tables(target.solver, 0)
        target.solver.Ey.fill_(-0.37)
        for entry in target.solver._copper_foils:
            entry["state"].fill_(0.17)
            entry["right_e"].fill_(-2.0)
        before = _snapshot(target.solver)
        for bad_value in (-1, 0, checkpoint_step + 1, TOTAL_STEPS + 1):
            bad = mw.FDTDResumeCheckpoint.load(path, map_location="cpu")
            counter = bad.auxiliary["observations"]["arrays"]["_dft_step"]
            assert counter.device.type == "cpu"
            counter.fill_(bad_value)
            with monkeypatch.context() as patch:
                def forbidden_physics_restore(*args, **kwargs):
                    pytest.fail("Physics restore was reached before invalid observation rejection.")
                patch.setattr(resume, "restore_checkpoint_state", forbidden_physics_restore)
                with pytest.raises(ValueError, match="DFT counter disagrees with the execution step"):
                    resume.restore_resume_checkpoint(target.solver, bad, total_steps=TOTAL_STEPS)
            assert _snapshot(target.solver) == before
            rejected += 1
        resumed = _execution(graph).run(resume_from=valid)
        _assert_final_parity(resumed, direct)
        assert resumed.solver._cuda_graph_active is graph
        assert resumed.solver._tail_graph_active is graph
        assert resumed.solver._observer_cuda_graph_active is graph
    record_property("actual_full_DFT_entries", len(FREQUENCIES))
    record_property("actual_weight_tables_active", True)
    record_property("actual_checkpoint_steps", "1,4")
    record_property("actual_checkpoint_nonzero_E_H_Cu_history", True)
    record_property("actual_CPU_counter_rejections_before_physics_restore", rejected)
    record_property("actual_target_all_tensor_bits_and_pointers_unchanged", True)
    record_property("actual_disk_resume_full_fields_and_full_DFT_bitwise", True)
    record_property("actual_graph_requested_and_asserted", graph)
    assert rejected == 8
