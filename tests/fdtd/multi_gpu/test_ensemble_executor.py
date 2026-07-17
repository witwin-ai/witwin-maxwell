"""CPU-constructible tests for the shared ensemble execution layer.

These exercise the generic scheduler (order preservation, deterministic device
leasing, failure isolation, fail-fast cancellation, capacity preflight, and the
timing hooks) without any CUDA dependency by driving plain Python task callables.
"""

from __future__ import annotations

import threading
import time

import pytest

from witwin.maxwell.execution import (
    DeviceCapability,
    DevicePool,
    DistributedFailure,
    ExecutionPlan,
    ExecutionTask,
    FailureKind,
    ResultSequence,
    execute_plan,
)


def _plan(tasks, *, placement="round_robin", max_concurrency=2, fail_fast=False):
    return ExecutionPlan(
        tasks=tuple(tasks),
        placement=placement,
        max_concurrency=max_concurrency,
        fail_fast=fail_fast,
    )


def _cpu_pool(count=2, **kwargs):
    return DevicePool(
        [f"cpu:{i}" for i in range(count)], require_cuda=False, **kwargs
    )


def test_result_sequence_preserves_submission_order():
    pool = _cpu_pool()

    def make(i):
        # Reverse the sleep so later tasks finish first if order were completion
        # based; a correct implementation still returns submission order.
        def _run(device):
            time.sleep(0.01 * (5 - i))
            return i * 100

        return ExecutionTask(index=i, run=_run, estimated_bytes=None, label=str(i))

    seq = execute_plan(_plan(make(i) for i in range(5)), pool)
    assert [seq[i] for i in range(5)] == [0, 100, 200, 300, 400]
    assert list(seq.entries) == [0, 100, 200, 300, 400]
    assert not seq.failed


def test_leasing_is_exclusive_and_uses_all_devices():
    pool = _cpu_pool(2)
    barrier = threading.Barrier(2, timeout=10)
    lock = threading.Lock()
    seen = []

    def make(i):
        def _run(device):
            with lock:
                seen.append(str(device))
            barrier.wait()  # force two tasks to overlap
            time.sleep(0.02)
            return i

        return ExecutionTask(index=i, run=_run, estimated_bytes=None, label=str(i))

    seq = execute_plan(_plan((make(i) for i in range(4)), max_concurrency=2), pool)
    assert [seq[i] for i in range(4)] == [0, 1, 2, 3]
    # Both devices were leased (concurrency forced two distinct leases at once).
    assert set(seen) == {"cpu:0", "cpu:1"}
    # No device was ever leased beyond its per-device capacity.
    for device in pool.devices:
        assert pool.peak_concurrency(device) <= pool.per_device_concurrency
    # Every completed task recorded exactly the device it was leased.
    for record in seq.records:
        assert record.device in pool.devices
        assert record.status == "completed"


def test_failure_isolation_records_failure_and_others_complete():
    pool = _cpu_pool(2)

    def make(i):
        def _run(device):
            if i == 2:
                raise RuntimeError("boom-2")
            return i * 10

        return ExecutionTask(index=i, run=_run, estimated_bytes=None, label=str(i))

    seq = execute_plan(_plan((make(i) for i in range(5)), fail_fast=False), pool)

    failure = seq.entries[2]
    assert isinstance(failure, DistributedFailure)
    assert failure.index == 2
    assert failure.kind is FailureKind.RUNTIME
    assert isinstance(failure.__cause__, RuntimeError)
    assert failure.device in pool.devices
    with pytest.raises(DistributedFailure):
        _ = seq[2]

    for i in (0, 1, 3, 4):
        assert seq[i] == i * 10
    assert seq.records[2].status == "failed"
    assert tuple(f.index for f in seq.failures) == (2,)
    # No cross-task bleed: the surviving records are unaffected.
    assert all(seq.records[i].status == "completed" for i in (0, 1, 3, 4))


def test_fail_fast_cancels_unstarted_tasks():
    pool = _cpu_pool(1)

    def make(i):
        def _run(device):
            if i == 0:
                raise ValueError("early")
            return i

        return ExecutionTask(index=i, run=_run, estimated_bytes=None, label=str(i))

    # Single worker => strict FIFO, so task 0 fails before 1..3 start.
    seq = execute_plan(
        _plan((make(i) for i in range(4)), max_concurrency=1, fail_fast=True), pool
    )
    assert seq.entries[0].kind is FailureKind.RUNTIME
    for i in (1, 2, 3):
        assert isinstance(seq.entries[i], DistributedFailure)
        assert seq.entries[i].kind is FailureKind.CANCELLED
        assert seq.records[i].status == "cancelled"


def test_capacity_preflight_rejects_oversized_task_before_running():
    caps = {"cpu:0": DeviceCapability("cpu:0", free_bytes=1_000, total_bytes=2_000)}
    pool = DevicePool(["cpu:0"], require_cuda=False, capabilities=caps)
    ran = []

    def _run(device):
        ran.append(device)
        return "ran"

    task = ExecutionTask(index=0, run=_run, estimated_bytes=5_000, label="big")
    seq = execute_plan(_plan((task,), placement="memory_aware", max_concurrency=1), pool)

    assert not ran  # never executed
    assert seq.entries[0].kind is FailureKind.CAPACITY
    assert seq.records[0].device is None
    assert seq.records[0].status == "failed"


def test_records_expose_timing_hooks():
    pool = _cpu_pool(1)

    def _run(device):
        time.sleep(0.01)
        return 1

    task = ExecutionTask(index=0, run=_run, estimated_bytes=None, label="x")
    seq = execute_plan(_plan((task,), max_concurrency=1), pool)
    record = seq.records[0]
    assert record.wall_time_s is not None and record.wall_time_s >= 0.0
    # CPU tasks have no CUDA event timing.
    assert record.device_time_s is None


def test_empty_plan_returns_empty_sequence():
    pool = _cpu_pool(1)
    seq = execute_plan(_plan(()), pool)
    assert isinstance(seq, ResultSequence)
    assert len(seq) == 0
    assert seq.entries == ()
