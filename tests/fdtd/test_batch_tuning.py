"""Measured policy selection and isolation of unpublished tuning runs."""
from dataclasses import replace

import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.execution.tuning import _select_candidate
from tests.fdtd.cuda.test_batch_runtime import simulation


def measurement(q, samples, memory):
    return dict(policy=dict(slots_per_device=q), samples_s=samples,
                median_s=sorted(samples)[1], peak_bytes={"cuda:0": memory})


def test_auto_slots_require_repeatable_gain_and_near_ties_prefer_less_memory():
    one = measurement(1, [1., 1.01, 1.02], 10)
    noisy = measurement(2, [.8, .9, 1.1], 20)
    assert _select_candidate([one, noisy], auto_slots=True) is one
    faster = measurement(2, [.8, .81, .82], 20)
    assert _select_candidate([one, faster], auto_slots=True) is faster
    near = measurement(1, [.98, .99, 1.], 30)
    assert _select_candidate([one, near], auto_slots=False) is one
    with pytest.raises(RuntimeError, match="single-slot baseline"):
        _select_candidate([faster], auto_slots=True)


def test_candidate_search_includes_balanced_cohorts_and_skips_idle_slots():
    from types import SimpleNamespace
    from witwin.maxwell.execution.tuning import _candidate_policies
    class Batch:
        execution = mw.BatchExecution(devices=("cuda:0", "cuda:1"), batch_size="auto",
            max_batch_size=10, slots_per_device="auto", tune=True, graph_steps=32)
        def __len__(self):
            return 10
        def plan(self):
            return SimpleNamespace(groups=[dict(variant_indices=list(range(10)))])
    policies = _candidate_policies(Batch())
    assert dict(batch_size=5, slots_per_device=1, graph_steps=32) in policies
    assert dict(batch_size=3, slots_per_device=2, graph_steps=32) in policies
    assert not any(p["batch_size"] >= 5 and p["slots_per_device"] == 2 for p in policies)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("graph", ["off", "required"])
def test_tuning_measures_full_runs_without_changing_public_results(graph):
    sims = [simulation(i, steps=19) for i in range(3)]
    policy = mw.BatchExecution(devices=("cuda:0",), batch_size="auto", max_batch_size=2,
        slots_per_device="auto", tune=True, graph=graph, graph_steps=8,
        memory_limit_bytes=128 * 1024**2)
    batch = mw.Simulation.batch(sims, variation=mw.BatchVariation(sources=("drive",)), execution=policy)
    state = torch.cuda.get_rng_state().clone()
    with batch.prepare() as prepared:
        report = prepared.report.to_dict()
        assert report["tuned"]
        assert len(report["candidate_measurements"]) == 4
        assert len(prepared.records) == 0
        assert batch.execution is policy
        torch.testing.assert_close(torch.cuda.get_rng_state(), state, rtol=0, atol=0)
        first, second = prepared.run(), prepared.run()
        assert prepared.report.tuned
        assert prepared.report.tuning["scope"] == "complete_forward_streamed_results"
        for m in report["candidate_measurements"]:
            assert len(m["samples_s"]) == 3 and min(m["samples_s"]) > 0
            assert max(m["peak_bytes"].values()) <= policy.memory_limit_bytes
    for i, sim in enumerate(sims):
        expected = sim.run().raw_monitor("probe")["components"]["Ez"]
        for result in (first, second):
            torch.testing.assert_close(result[i].raw_monitor("probe")["components"]["Ez"],
                                       expected, rtol=1e-4, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_tuned_training_preserves_inputs_and_backward_after_close():
    sim = simulation(steps=19)
    scales = torch.tensor([0., -1.], device="cuda:0", requires_grad=True)
    batch = mw.Simulation.batch(sim, parameters=mw.BatchParameters(source_scales={"drive": scales}),
        execution=mw.BatchExecution(devices=("cuda:0",), batch_size=2, tune=True,
            graph="off", memory_limit_bytes=128 * 1024**2))
    with batch.prepare() as prepared:
        assert scales.grad is None and scales._version == 0
        assert prepared.report.tuning["records_grad"]
        results = prepared.run()
    results.stack(lambda r: r.raw_monitor("probe")["components"]["Ez"].real.sum()).sum().backward()
    assert scales.grad is not None and torch.isfinite(scales.grad).all()
    assert scales.grad.abs().min() > 0


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices")
def test_tuning_selects_graph_chunks_on_two_real_devices():
    batch = mw.Simulation.batch([simulation(i, steps=19) for i in range(4)],
        variation=mw.BatchVariation(sources=("drive",)),
        execution=mw.BatchExecution(devices=("cuda:0", "cuda:1"), batch_size=1, tune=True,
            graph="required", graph_steps="auto", memory_limit_bytes=128 * 1024**2))
    with batch.prepare() as prepared:
        assert {m["policy"]["graph_steps"] for m in prepared.report.candidate_measurements} == {16, 32, 64}
        results = prepared.run()
        results.results()
        assert {r.device for r in results.records} == {"cuda:0", "cuda:1"}
        assert prepared.report.tuned


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices")
def test_tuning_restores_scene_module_rng_on_nonexecution_device():
    class Module(mw.SceneModule):
        def to_scene(self):
            scene = simulation(steps=8).scene.clone(device="cuda:1")
            scene.metadata["draw"] = float(torch.rand((), device="cuda:1"))
            return scene
    sim = mw.Simulation.fdtd(Module(), frequencies=(1e9,),
        run_time=mw.TimeConfig(time_steps=8, dt=2e-11),
        spectral_sampler=mw.SpectralSampler(window="none"), cuda_graph=False)
    batch = mw.Simulation.batch([sim], execution=mw.BatchExecution(
        devices=("cuda:0",), batch_size=1, graph="off", tune=True))
    state = torch.cuda.get_rng_state("cuda:1")
    with batch.prepare():
        tuned_state = torch.cuda.get_rng_state("cuda:1")
    torch.cuda.set_rng_state(state, "cuda:1")
    batch.execution = replace(batch.execution, tune=False)
    with batch.prepare():
        expected_state = torch.cuda.get_rng_state("cuda:1")
    torch.testing.assert_close(tuned_state, expected_state, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_tuning_rejects_capacity_and_propagates_physics_errors():
    sim = simulation(steps=8)
    policy = mw.BatchExecution(devices=("cuda:0",), batch_size=1, tune=True, graph="off", memory_limit_bytes=1024)
    batch = mw.Simulation.batch([sim], execution=policy)
    with pytest.raises(torch.cuda.OutOfMemoryError, match="No Batch tuning candidate"):
        batch.prepare()
    batch.execution = replace(policy, memory_limit_bytes=128 * 1024**2)
    batch.outputs = mw.BatchOutput(materials=True)
    with pytest.raises(NotImplementedError, match="material export"):
        batch.prepare()
