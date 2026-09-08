"""Full batch graph replay preserves state, tails, outputs and capture ownership."""
import threading
import pytest
import torch
import witwin.maxwell as mw
from tests.fdtd.cuda.test_batch_runtime import simulation
from witwin.maxwell.fdtd.cuda.runtime.graph import CudaGraphRunner, capture_participant

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


@pytest.mark.parametrize("fields", [(), ("Ez",)])
def test_required_graph_chunks_match_eager_and_reuse_monitor_captures(monkeypatch, fields):
    sims = [simulation(i, steps=31, frequencies=(.8e9, 1e9, 1.2e9)) for i in range(3)]
    options = dict(variation=mw.BatchVariation(sources=("drive",)), outputs=mw.BatchOutput(fields=fields))
    eager = mw.Simulation.batch(sims, execution=mw.BatchExecution(devices=("cuda:0",),
                                batch_size=3, graph="off"), **options).run()
    calls = []
    original = CudaGraphRunner.capture
    def capture(runner, function):
        calls.append(1)
        return original(runner, function)
    monkeypatch.setattr(CudaGraphRunner, "capture", capture)
    with mw.Simulation.batch(sims, execution=mw.BatchExecution(devices=("cuda:0",),
            batch_size=3, graph="required", graph_steps=7), **options).prepare() as prepared:
        first = prepared.run()
        first.results()
        assert prepared.report.execution_tier == "native_pipeline_graph"
        assert first[0].stats()["batch_graph_replays"] == 5
        captures = len(calls)
        assert captures == 2
        saved = first.stack(lambda r: r.raw_monitor("probe")["components"]["Ez"]).clone()
        second = prepared.run()
        second.results()
        assert len(calls) == captures * (2 if fields else 1)
    torch.testing.assert_close(first.stack(lambda r: r.raw_monitor("probe")["components"]["Ez"]), saved, rtol=0, atol=0)
    for i in range(3):
        torch.testing.assert_close(first[i].raw_monitor("probe")["components"]["Ez"],
                                   eager[i].raw_monitor("probe")["components"]["Ez"], rtol=0, atol=0)
        for name in fields:
            torch.testing.assert_close(first[i]._fields[name], eager[i]._fields[name], rtol=0, atol=0)
            torch.testing.assert_close(second[i]._fields[name], first[i]._fields[name], rtol=0, atol=0)


def test_concurrent_capture_participant_forces_auto_eager_and_required_failure():
    entered, release = threading.Event(), threading.Event()
    @capture_participant
    def busy():
        entered.set()
        assert release.wait(20)
    worker = threading.Thread(target=busy)
    worker.start()
    try:
        assert entered.wait(10)
        for policy in ("auto", "required"):
            batch = mw.Simulation.batch([simulation(steps=8)], execution=mw.BatchExecution(
                devices=("cuda:0",), batch_size=1, graph=policy))
            with batch.prepare() as prepared:
                results = prepared.run()
                if policy == "auto":
                    results.results()
                    assert prepared.report.execution_tier == "native_pipeline_eager"
                    assert "batch thread" in prepared.report.graph_fallback_reason
                else:
                    assert results.failed
    finally:
        release.set()
        worker.join(10)
    assert not worker.is_alive()
