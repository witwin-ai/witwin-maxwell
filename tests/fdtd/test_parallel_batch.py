"""Persistent parallel slots preserve ordering, budgets, streams and cancellation."""
import pytest
import torch
import witwin.maxwell as mw
from tests.fdtd.cuda.test_batch_runtime import simulation

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


@pytest.mark.parametrize("slots,devices", [(2, ("cuda:0",)), (1, ("cuda:0", "cuda:1"))])
def test_automatic_capacity_leaves_work_for_each_worker(slots, devices):
    if len(devices) > torch.cuda.device_count():
        pytest.skip("Requires two CUDA devices")
    sims = [simulation(i, steps=16) for i in range(6)]
    batch = mw.Simulation.batch(sims, variation=mw.BatchVariation(sources=("drive",)),
        execution=mw.BatchExecution(devices=devices, slots_per_device=slots, batch_size="auto",
            scheduling="static", graph="required", memory_limit_bytes=128 * 1024**2))
    with batch.prepare() as prepared:
        assert all(child.capacity <= 3 for child in prepared._children)
        result = prepared.run()
        result.results()
        assert all(child._models for child in prepared._children)
        assert all(child._last_graph_replays > 0 for child in prepared._children)
        assert {record.device for record in result.records} == set(devices)
    for i, sim in enumerate(sims):
        torch.testing.assert_close(result[i].raw_monitor("probe")["components"]["Ez"].to("cuda:0"),
            sim.run().raw_monitor("probe")["components"]["Ez"], rtol=1e-4, atol=1e-6)


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


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices")
def test_regional_envelopes_follow_the_worker_device_and_budget():
    sims = [simulation(steps=19) for _ in range(4)]
    for index, sim in enumerate(sims):
        sim.scene.structures = [mw.Structure(name="design",
            geometry=mw.Sphere(radius=.02, position=(.01*index, 0., 0.), device="cuda:0"),
            material=mw.Material(eps_r=2. + .1*index))]
    envelope = mw.Box(size=(.2, .2, .2), device="cuda:0")
    batch = mw.Simulation.batch(sims, variation=mw.BatchVariation(structures={"design": envelope}),
        execution=mw.BatchExecution(devices=("cuda:0", "cuda:1"), scheduling="static", batch_size=1,
            memory_limit_bytes=128*1024**2, reserve_bytes=0, graph="required"))
    with batch.prepare() as prepared:
        results = prepared.run()
        results.results()
        for child in prepared._children:
            cache = child._local_material_cache
            assert cache.structure_envelopes["design"].position.device == torch.device(child.device)
            assert cache.full_compilations == 1
        assert all(record["memory"]["peak_charged_bytes"] <= 128*1024**2
                   for record in prepared.report.devices)
    assert envelope.position.device == torch.device("cuda:0")
    for item, sim in zip(results, sims):
        torch.testing.assert_close(item.raw_monitor("probe")["components"]["Ez"].to("cuda:0"),
            sim.run().raw_monitor("probe")["components"]["Ez"], rtol=1e-4, atol=1e-6)


def test_static_waiter_rechecks_queue_after_last_task_is_taken(monkeypatch):
    from collections import deque
    from threading import Condition, Event
    from types import SimpleNamespace
    from witwin.maxwell.execution.batch_runtime import ParallelBatchResultIterator
    iterator = object.__new__(ParallelBatchResultIterator)
    iterator._condition = Condition()
    iterator._tasks = deque([(0, deque([0]))])
    iterator._assignment = 0
    iterator.prepared = SimpleNamespace(_cancel=Event(), _children=(None, None),
        batch=SimpleNamespace(execution=SimpleNamespace(scheduling="static")))
    def competing_worker_takes_last_task(timeout):
        iterator._tasks.popleft()
        iterator._assignment = 1
    monkeypatch.setattr(iterator._condition, "wait", competing_worker_takes_last_task)
    assert iterator._take_task(SimpleNamespace(capacity=1), worker=1) is None
    assert not iterator.prepared._cancel.is_set()


def test_final_publication_after_queue_timeout_is_delivered():
    from queue import Queue, Empty
    from threading import Condition, Lock, Semaphore
    from types import SimpleNamespace
    from witwin.maxwell.execution.batch_runtime import ParallelBatchResultIterator
    from witwin.maxwell.execution.records import BatchItem, ExecutionRecord
    iterator = object.__new__(ParallelBatchResultIterator)
    record = ExecutionRecord(index=0, device="cuda:0", status="completed")
    completed = BatchItem(index=0, variant_id="only", record=record, failure=None,
                          result=SimpleNamespace(_fields={}, _monitors={}))
    class Inputs(list):
        variant_ids = ("only",)
    class RacingQueue(Queue):
        def get(self, block=True, timeout=None):
            if iterator._running:
                self.put((completed, []))
                iterator._records[0] = record
                iterator._running = 0
                raise Empty
            return super().get(block, timeout)
    iterator._iteration_gate = Lock()
    iterator._condition = Condition()
    iterator._queue = RacingQueue()
    iterator._permits = Semaphore(0)
    iterator._futures = ()
    iterator._running = 1
    iterator._closed = False
    iterator._records = {}
    iterator._delivered = set()
    iterator.prepared = SimpleNamespace(batch=Inputs([None]), device="cuda:0")
    assert next(iterator) is completed


def test_batch_capture_allows_independent_consumer_thread_allocations(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from witwin.maxwell.fdtd.runtime.batch import NativeFieldSlot
    entered, release = Event(), Event()
    original = NativeFieldSlot.advance
    def pause_capture(self):
        if torch.cuda.is_current_stream_capturing() and not entered.is_set():
            entered.set()
            assert release.wait(10)
        return original(self)
    monkeypatch.setattr(NativeFieldSlot, "advance", pause_capture)
    batch = mw.Simulation.batch([simulation(steps=19)],
        execution=mw.BatchExecution(devices=("cuda:0",), batch_size=1, graph="required",
            memory_limit_bytes=128*1024**2, reserve_bytes=0))
    with batch.prepare() as prepared, ThreadPoolExecutor(max_workers=1) as executor:
        consumer_pool = torch.cuda.MemPool()
        future = executor.submit(prepared.run)
        try:
            assert entered.wait(10)
            # A fresh, independent allocator pool forces an allocation during
            # another thread's capture instead of relying on cached blocks.
            with torch.cuda.use_mem_pool(consumer_pool):
                value = torch.ones(1024**2, device="cuda:0")
                assert value.sum().item() == 1024**2
        finally:
            release.set()
        result = future.result(timeout=20)
        result.results()
        assert result[0].solver_stats["batch_execution_tier"] == "native_pipeline_graph"
