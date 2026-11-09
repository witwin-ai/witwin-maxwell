"""Approved termination gates on the production incident line, CPU diagnostics."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd import resume
from witwin.maxwell.fdtd.distributed import resume as distributed_resume
from tests.sources.tfsf.test_pulsed_incident_reference import _execution
from scripts.dev.tfsf_aux_line_termination import (
    CONFIGURATIONS, Configuration, appended_line, history_gate, lossless_reference,
    prepare_line, reflection, sampled_history, sampled_indices, slowest_pole,
)


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("config", CONFIGURATIONS, ids=lambda c: c.name)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_qualified_production_line_g1_g2_g3(config, dtype, record_property):
    old, old_faces = prepare_line(config, dtype)
    line, faces = prepare_line(config, dtype, production=True)
    expected = appended_line(old, 320, 2, 1e-8)
    assert len(line.electric) == len(expected.electric)
    assert (line.s_min, line.ds, line.dt, line.source_index, line.incident_reference) == (
        old.s_min, old.ds, old.dt, old.source_index, old.incident_reference)
    for kind in ("electric", "magnetic"):
        for suffix in ("_decay", "_curl"):
            actual = getattr(line, kind + suffix)
            assert torch.equal(actual, getattr(expected, kind + suffix))
            previous = getattr(old, kind + suffix)
            lossless = getattr(old, kind + "_decay") == 1
            assert torch.equal(actual[:len(previous)][lossless], previous[lossless])
    for key in ("electric_terms", "magnetic_terms"):
        for before, after in zip(old_faces[key], faces[key]):
            for name in ("sample_indices", "sample_positions"):
                if name in before:
                    assert torch.equal(before[name], after[name])
    edge = np.arcsin(line.wave_speed * line.dt / line.ds) / (np.pi * line.dt)
    frequencies = [2e9] if config.cw else np.linspace(1e9, min(3e9, .99 * edge), 17)
    ratios = [reflection(line, f) / reflection(old, f) for f in frequencies]
    assert max(ratios) <= 1
    record_property("G3_worst_ratio", max(ratios))
    if not config.cw:
        steps = line.record_steps
        assert steps == 65536
        reference = sampled_history(lossless_reference(line, steps), faces, steps)
        gate = history_gate(line, faces, reference, steps)
        before = history_gate(old, old_faces, reference, steps)
        record_property("candidate", str(gate))
        record_property("before", str(before))
        assert gate["full"] <= gate["bound"]
        assert gate["growth"] <= 4
        # The coarse surviving-mode line fails while the qualified line passes.
        assert before["full"] > before["bound"]


@pytest.mark.parametrize("h", [.025, .0125])
def test_production_extent_and_slowest_pole_padding_and_dtype_controls(h):
    records = []
    for half in (.15, .3):
        for layers in (4, 8, 12):
            pair = []
            for dtype in (torch.float32, torch.float64):
                line, _ = prepare_line(Configuration(h=h, half=half, layers=layers), dtype, production=True)
                pair.append(len(line.electric))
                if dtype == torch.float64:
                    records.append((len(line.electric), slowest_pole(line)))
            assert pair[0] == pair[1]
    assert len({r[0] for r in records}) == 1
    for _, pole in records[1:]:
        assert pole["frequency_Hz"] == pytest.approx(records[0][1]["frequency_Hz"], rel=1e-12)
        assert pole["per_period_modulus"] == pytest.approx(records[0][1]["per_period_modulus"], abs=2e-11)


def test_lossless_sampling_assert_rejects_loss_and_interpolation_neighbour():
    line, _ = prepare_line(Configuration(), production=True)
    # Direct H gather at the first lossy H node.
    with pytest.raises(AssertionError, match="overlaps auxiliary absorber"):
        line.assert_lossless_samples([dict(sample_indices=torch.tensor([line.sampled_nodes - 1]))], [])
    # The lower E neighbour is lossless; its upper interpolation neighbour is not.
    position = line.s_min + (line.sampled_nodes - .75) * line.ds
    with pytest.raises(AssertionError, match="overlaps auxiliary absorber"):
        line.assert_lossless_samples([], [dict(sample_positions=torch.tensor([position], dtype=line.dtype))])
    with pytest.raises(AssertionError, match="outside auxiliary line"):
        line.assert_lossless_samples([], [dict(sample_indices=torch.tensor([-1]))])


@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize("cw", [False, True])
def test_old_termination_checkpoint_refused_before_fingerprint_and_copy(distributed, cw, tmp_path):
    # Use an actual serialized checkpoint container and the real shared preflight.
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.5, .5),) * 3), grid=mw.GridSpec.uniform(.05),
                     boundary=mw.BoundarySpec.pml(num_layers=8), device="cpu")
    execution = mw.Simulation.fdtd(scene, frequency=1e9, backend="cpu_reference",
        run_time=mw.TimeConfig(time_steps=65536), full_field_dft=False).prepare()
    checkpoint = execution.run_until(2)
    local = _execution(waveform=mw.CW(1e9) if cw else None).solver
    # The actual affected CPU solver cannot execute native face kernels; its
    # distributed wrapper exercises the same shared preflight without a GPU.
    solver = (SimpleNamespace(shard_layouts=(), logical_scene=local.scene,
               shards=(SimpleNamespace(rank=0, solver=local),), _nccl=False, dt=local.dt,
               _time_plan=local._time_plan)
              if distributed else local)
    signature = distributed_resume._signature(solver) if distributed else resume._solver_signature(solver)
    assert signature.pop("tfsf_auxiliary_termination") == "appended_320_p2_r1e-8_v1"
    signature["configuration_fingerprint"] = "deliberately changed"
    checkpoint = replace(checkpoint, signature=signature, dt=local.dt)
    checkpoint.save(tmp_path / "old.pt")
    checkpoint = mw.FDTDResumeCheckpoint.load(tmp_path / "old.pt")
    before = local.Ex.clone()
    with pytest.raises(NotImplementedError) as error:
        resume.preflight_resume_checkpoint(solver, checkpoint, total_steps=65536)
    assert str(error.value) == "checkpoint predates the qualified TFSF auxiliary-line termination; re-run from the start"
    assert torch.equal(before, local.Ex)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_distant_reference_extension_is_invisible(dtype):
    line, faces = prepare_line(Configuration(), dtype)
    steps = line.record_steps
    reference = lossless_reference(line, steps)
    twice = lossless_reference(line, 2 * steps)
    np.testing.assert_array_equal(sampled_history(reference, faces, steps), sampled_history(twice, faces, steps))
