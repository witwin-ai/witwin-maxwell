"""Persistent parallel slots preserve ordering, budgets, streams and cancellation."""
import pytest
import torch
import witwin.maxwell as mw
from tests.fdtd.cuda.test_batch_runtime import simulation

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


@pytest.mark.parametrize("graph", ["off", "required"])
@pytest.mark.parametrize("order,scheduling", [("completion", "throughput"), ("input", "static")])
def test_two_slots_share_budget_and_preserve_results(graph, order, scheduling):
    sims = [simulation(i, steps=35) for i in range(7)]
    batch = mw.Simulation.batch(sims, variation=mw.BatchVariation(sources=("drive",)),
        execution=mw.BatchExecution(devices=("cuda:0",), slots_per_device=2, batch_size=2,
                                   memory_limit_bytes=128 * 1024**2, graph=graph, scheduling=scheduling),
        outputs=mw.BatchOutput(max_pending_results=1))
    with batch.prepare() as prepared:
        assert len(prepared._children) == 2
        with prepared.iter_results(order=order) as iterator:
            items = list(iterator)
        assert len(items) == 7
        assert all(item.result is not None for item in items), [item.failure for item in items]
        if order == "input":
            assert [item.index for item in items] == list(range(7))
        actual = {item.index: item.result for item in items}
        saved = actual[0].raw_monitor("probe")["components"]["Ez"].clone()
        repeat = prepared.run()
        repeat.results()
        memory = prepared.report.devices[0]["memory"]
        assert memory["peak_charged_bytes"] <= memory["limit_bytes"]
    for index, sim in enumerate(sims):
        expected = sim.run().raw_monitor("probe")["components"]["Ez"]
        torch.testing.assert_close(actual[index].raw_monitor("probe")["components"]["Ez"], expected,
                                   rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(actual[0].raw_monitor("probe")["components"]["Ez"], saved, rtol=0, atol=0)


def test_parallel_early_close_and_failure_records(monkeypatch):
    batch = mw.Simulation.batch([simulation(i, steps=16) for i in range(7)],
        variation=mw.BatchVariation(sources=("drive",)), outputs=mw.BatchOutput(max_pending_results=1),
        execution=mw.BatchExecution(devices=("cuda:0",), slots_per_device=2, batch_size=2))
    with batch.prepare() as prepared:
        iterator = prepared.iter_results()
        first = next(iterator)
        iterator.close()
        assert len(iterator.records) == 7
        assert first.result is not None
        def fail(*args):
            raise RuntimeError("Injected parallel slot failure")
        for child in prepared._children:
            monkeypatch.setattr(child, "_cohort_steps", fail)
        result = prepared.run()
        assert result.failed
        assert len(result.records) == 7


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices")
@pytest.mark.parametrize("destination", ["local", "cpu", "cuda:0"])
def test_two_devices_match_single_device_with_explicit_output_transfer(destination):
    sims = [simulation(i, steps=16) for i in range(5)]
    batch = mw.Simulation.batch(sims, variation=mw.BatchVariation(sources=("drive",)),
        execution=mw.BatchExecution(devices=("cuda:0", "cuda:1"), batch_size=1,
            scheduling="static", graph="required", memory_limit_bytes={"cuda:0": 64 * 1024**2, "cuda:1": 64 * 1024**2}),
        outputs=mw.BatchOutput(result_device=destination, max_pending_results=1))
    with batch.prepare() as prepared:
        result = prepared.run()
        assert not result.failed, "\n".join(str(r.failure) for r in result.records if r.failure)
        result.results()
        assert {record.device for record in result.records} == {"cuda:0", "cuda:1"}
        for entry in prepared.report.devices:
            assert entry["memory"]["peak_charged_bytes"] <= entry["memory"]["limit_bytes"]
    for index, sim in enumerate(sims):
        actual = result[index].raw_monitor("probe")["components"]["Ez"]
        if destination != "local":
            assert str(actual.device) == destination
        expected = sim.run().raw_monitor("probe")["components"]["Ez"]
        torch.testing.assert_close(actual.to("cuda:0"), expected, rtol=1e-4, atol=1e-6)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices")
def test_scene_device_clone_keeps_input_and_autograd_ownership():
    source = simulation().scene
    position = source.structures[0].geometry.position
    position.requires_grad_(True)
    moved = source.clone(device="cuda:1")
    target = moved.structures[0].geometry.position
    assert target.device == torch.device("cuda:1")
    assert position.device == torch.device("cuda:0")
    assert source.structures[0].geometry.position is position
    target.square().sum().backward()
    torch.testing.assert_close(position.grad, 2 * position)


def test_parallel_source_updates_from_caller_stream():
    scales = torch.tensor([1., 2., 3., 4.], device="cuda:0")
    batch = mw.Simulation.batch(simulation(steps=31),
        parameters=mw.BatchParameters(source_scales={"drive": scales}),
        execution=mw.BatchExecution(devices=("cuda:0",), slots_per_device=2, batch_size=1, graph="required"))
    with batch.prepare() as prepared:
        first = prepared.run().stack(lambda r: r.raw_monitor("probe")["components"]["Ez"])
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            scales.mul_(2)
            second = prepared.run().stack(lambda r: r.raw_monitor("probe")["components"]["Ez"])
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(second, 2 * first, rtol=1e-4, atol=1e-6)
