from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from .plan import ExecutionPlan, ExecutionTask
from .pool import DevicePool
from .records import (
    DistributedFailure,
    ExecutionRecord,
    FailureKind,
    ResultSequence,
)


class _Cancellation:
    """Cooperative fail-fast flag shared by the worker threads."""

    def __init__(self, enabled: bool):
        self._enabled = enabled
        self._event = threading.Event()

    @property
    def triggered(self) -> bool:
        return self._enabled and self._event.is_set()

    def trip(self) -> None:
        if self._enabled:
            self._event.set()


def _preflight_capacity(task: ExecutionTask, pool: DevicePool) -> DistributedFailure | None:
    """Reject a task before it runs when no device snapshot can hold it."""

    if task.estimated_bytes is None:
        return None
    frees = [
        pool.capability(device).free_bytes
        for device in pool.devices
        if pool.capability(device) is not None
        and pool.capability(device).free_bytes is not None
    ]
    if not frees:
        return None
    if task.estimated_bytes > max(frees):
        return DistributedFailure(
            index=task.index,
            kind=FailureKind.CAPACITY,
            device=None,
            message=(
                f"estimated {task.estimated_bytes} bytes exceed the largest leased "
                f"device free memory {max(frees)} bytes; run a smaller task or add a device."
            ),
        )
    return None


def _device_timer(device: torch.device):
    """Best-effort per-task device timing hook (never fails the task).

    Interpreting the recorded device time as a task-level speedup requires an
    exclusive-GPU window; this only records the events so the record is populated
    when it can be. Any CUDA hiccup degrades to ``None`` rather than raising.
    """

    if device.type != "cuda":
        return None
    try:
        with torch.cuda.device(device):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        return (start, end)
    except RuntimeError:
        return None


def _device_elapsed_s(device: torch.device, timer) -> float | None:
    if timer is None:
        return None
    start, end = timer
    try:
        with torch.cuda.device(device):
            end.record()
        torch.cuda.synchronize(device)
        return float(start.elapsed_time(end)) / 1.0e3
    except RuntimeError:
        return None


def _run_one(task: ExecutionTask, pool: DevicePool, cancellation: _Cancellation):
    if cancellation.triggered:
        failure = DistributedFailure(
            index=task.index,
            kind=FailureKind.CANCELLED,
            device=None,
            message="cancelled by fail_fast before the task started.",
        )
        return failure, ExecutionRecord(
            index=task.index,
            device=None,
            status="cancelled",
            estimated_bytes=task.estimated_bytes,
            failure=failure,
        )

    capacity_failure = _preflight_capacity(task, pool)
    if capacity_failure is not None:
        cancellation.trip()
        return capacity_failure, ExecutionRecord(
            index=task.index,
            device=None,
            status="failed",
            estimated_bytes=task.estimated_bytes,
            failure=capacity_failure,
        )

    lease = pool.lease(estimated_bytes=task.estimated_bytes)
    if cancellation.triggered:
        # fail_fast tripped while this task blocked in lease(); release the just
        # acquired slot and cancel before running instead of executing a doomed task.
        lease.release()
        failure = DistributedFailure(
            index=task.index,
            kind=FailureKind.CANCELLED,
            device=lease.device,
            message="cancelled by fail_fast after leasing a device but before running.",
        )
        return failure, ExecutionRecord(
            index=task.index,
            device=lease.device,
            status="cancelled",
            estimated_bytes=task.estimated_bytes,
            failure=failure,
        )
    device = lease.torch_device
    wall_start = time.perf_counter()
    timer = _device_timer(device)
    try:
        # A worker thread inherits the process-default CUDA device, not the leased
        # one. Bind it for the whole task so every stream, event, allocation and
        # graph capture the task opens belongs to the device it leased.
        if device.type == "cuda":
            with torch.cuda.device(device):
                result = task.run(device)
        else:
            result = task.run(device)
    except BaseException as exc:  # noqa: BLE001 - re-raised as a structured failure
        wall = time.perf_counter() - wall_start
        cancellation.trip()
        failure = DistributedFailure(
            index=task.index,
            kind=FailureKind.RUNTIME,
            device=lease.device,
            exception=exc,
        )
        record = ExecutionRecord(
            index=task.index,
            device=lease.device,
            status="failed",
            estimated_bytes=task.estimated_bytes,
            wall_time_s=wall,
            failure=failure,
        )
        return failure, record
    else:
        device_time = _device_elapsed_s(device, timer)
        wall = time.perf_counter() - wall_start
        record = ExecutionRecord(
            index=task.index,
            device=lease.device,
            status="completed",
            estimated_bytes=task.estimated_bytes,
            wall_time_s=wall,
            device_time_s=device_time,
        )
        return result, record
    finally:
        lease.release()


def execute_plan(plan: ExecutionPlan, pool: DevicePool) -> ResultSequence:
    """Run an ExecutionPlan over a DevicePool, preserving submission order.

    Independent tasks run concurrently up to ``plan.max_concurrency``; each
    leases exactly one device for its lifetime. A task failure is captured as a
    structured ``DistributedFailure`` at its slot; with ``fail_fast`` the plan
    stops scheduling further tasks and marks them cancelled -- both tasks that
    never reached their capacity preflight and tasks that had already blocked in
    ``pool.lease()`` and only then acquired a slot -- while a task already inside
    ``task.run`` is allowed to finish. Fail-fast is cooperative, so a task that
    entered ``task.run`` just before the trip still completes. Nothing is summed
    or reduced across tasks, so no coordinator global synchronization is required.
    """

    entries: list[object | None] = [None] * len(plan)
    records: list[ExecutionRecord | None] = [None] * len(plan)
    cancellation = _Cancellation(plan.fail_fast)

    if len(plan) == 0:
        return ResultSequence((), ())

    workers = min(plan.max_concurrency, len(plan))
    with ThreadPoolExecutor(max_workers=workers) as pool_executor:
        futures = {
            pool_executor.submit(_run_one, task, pool, cancellation): task.index
            for task in plan.tasks
        }
        for future in futures:
            index = futures[future]
            entry, record = future.result()
            entries[index] = entry
            records[index] = record

    return ResultSequence(entries, records)


__all__ = ["execute_plan"]
