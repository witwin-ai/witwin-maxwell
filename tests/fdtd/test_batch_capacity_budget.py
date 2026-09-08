"""Automatic capacity and hard backing-memory limits for public Batch execution."""
import pytest
import torch
import witwin.maxwell as mw
from tests.fdtd.cuda.test_batch_runtime import simulation

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


@pytest.mark.parametrize("graph", ["off", "required"])
def test_auto_capacity_uses_multiple_lanes_inside_hard_budget(graph):
    sims = [simulation(i, steps=31) for i in range(5)]
    limit = 128 * 1024**2
    batch = mw.Simulation.batch(sims, variation=mw.BatchVariation(sources=("drive",)),
        execution=mw.BatchExecution(devices=("cuda:0",), batch_size="auto",
                                   memory_limit_bytes=limit, graph=graph))
    with batch.prepare() as prepared:
        assert 1 < prepared.capacity <= 5
        first = prepared.run()
        first.results()
        saved = first.stack(lambda r: r.raw_monitor("probe")["components"]["Ez"]).clone()
        second = prepared.run()
        second.results()
        assert prepared._budget.stats()["peak_charged_bytes"] <= limit
        assert prepared.report.memory["accounting"] == "private_cuda_backing_segments"
    torch.testing.assert_close(first.stack(lambda r: r.raw_monitor("probe")["components"]["Ez"]), saved,
                               rtol=0, atol=0)
    for i, sim in enumerate(sims):
        torch.testing.assert_close(second[i].raw_monitor("probe")["components"]["Ez"],
            sim.run().raw_monitor("probe")["components"]["Ez"], rtol=1e-4, atol=1e-6)


def test_tiny_fixed_budget_fails_before_running():
    batch = mw.Simulation.batch([simulation()], execution=mw.BatchExecution(
        devices=("cuda:0",), batch_size=1, memory_limit_bytes=1024))
    with pytest.raises(torch.cuda.OutOfMemoryError):
        batch.prepare()


def test_auto_capacity_retries_without_duplicate_publication(monkeypatch):
    batch = mw.Simulation.batch([simulation(i, steps=8) for i in range(5)],
        variation=mw.BatchVariation(sources=("drive",)),
        execution=mw.BatchExecution(devices=("cuda:0",), batch_size="auto", graph="off"))
    with batch.prepare() as prepared:
        original = prepared._solve_cohort
        def constrained():
            if prepared.capacity > 2:
                raise torch.cuda.OutOfMemoryError("Injected competing allocation")
            return original()
        monkeypatch.setattr(prepared, "_solve_cohort", constrained)
        with prepared.iter_results() as iterator:
            items = list(iterator)
        assert [item.index for item in items] == list(range(5))
        assert all(item.result is not None for item in items)
        assert prepared.capacity <= 2
        assert prepared._capacity_retries


def test_budget_accounts_for_retained_full_field_results_and_graph_pool():
    batch = mw.Simulation.batch([simulation(i, steps=31, frequencies=(.9e9, 1e9, 1.1e9)) for i in range(3)],
        variation=mw.BatchVariation(sources=("drive",)), outputs=mw.BatchOutput(fields=("Ez",)),
        execution=mw.BatchExecution(devices=("cuda:0",), batch_size=3,
                                   memory_limit_bytes=128 * 1024**2, graph="required"))
    with batch.prepare() as prepared:
        first = prepared.run()
        first.results()
        saved = first[0]._fields["Ez"].clone()
        second = prepared.run()
        second.results()
        stats = prepared._budget.stats()
        assert stats["peak_charged_bytes"] <= stats["limit_bytes"]
        assert prepared.report.memory == stats
    torch.testing.assert_close(first[0]._fields["Ez"], saved, rtol=0, atol=0)
