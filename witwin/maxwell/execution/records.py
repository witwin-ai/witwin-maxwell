from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator


class FailureKind(str, Enum):
    """Deterministic taxonomy for a single ensemble task failure.

    - ``CAPACITY``: the per-task memory-estimation preflight predicted the task
      cannot fit on any leased device; the task was never executed.
    - ``RUNTIME``: the task callable raised while preparing or running.
    - ``CANCELLED``: ``fail_fast`` aborted the plan before this task began
      running -- either before its capacity preflight or right after it leased a
      device but before entering the task callable.
    """

    CAPACITY = "capacity"
    RUNTIME = "runtime"
    CANCELLED = "cancelled"

    def __str__(self) -> str:
        return self.value


class DistributedFailure(Exception):
    """Structured, order-preserving failure for one ensemble task.

    Retains the submission ``index``, the leased ``device`` (``None`` when the
    task never acquired one), the failure ``kind`` and the original exception so
    ``fail_fast=False`` can report every failure without swallowing the cause.
    """

    def __init__(
        self,
        *,
        index: int,
        kind: FailureKind,
        device: str | None = None,
        exception: BaseException | None = None,
        message: str | None = None,
    ):
        self.index = int(index)
        self.kind = FailureKind(kind)
        self.device = None if device is None else str(device)
        self.exception = exception
        detail = message
        if detail is None and exception is not None:
            detail = f"{type(exception).__name__}: {exception}"
        self.detail = detail or ""
        rendered = (
            f"ensemble task {self.index} failed on device {self.device!r} "
            f"({self.kind}): {self.detail}"
        )
        super().__init__(rendered)
        if exception is not None:
            self.__cause__ = exception


@dataclass(frozen=True)
class ExecutionRecord:
    """Per-task execution bookkeeping in submission order.

    ``wall_time_s`` / ``device_time_s`` are measurement hooks. Interpreting them
    as a task-level speedup requires an exclusive-GPU window and is intentionally
    left to the caller; the executor never asserts on them.
    """

    index: int
    device: str | None
    status: str  # "completed" | "failed" | "cancelled"
    estimated_bytes: int | None = None
    wall_time_s: float | None = None
    device_time_s: float | None = None
    failure: DistributedFailure | None = None

    @property
    def completed(self) -> bool:
        return self.status == "completed"


class ResultSequence:
    """Ordered container of ensemble task outcomes preserving submission order.

    ``entries`` returns ``Result | DistributedFailure`` per slot without raising;
    indexing and iteration both return the ``Result`` and raise the slot's
    ``DistributedFailure`` for a failed task, so a failure is never silently
    dropped and ``list(seq)`` agrees with ``[seq[i] for i in range(len(seq))]``.
    Use ``entries`` for the non-raising union view.
    """

    def __init__(self, entries, records):
        self._entries: tuple[Any, ...] = tuple(entries)
        self._records: tuple[ExecutionRecord, ...] = tuple(records)
        if len(self._entries) != len(self._records):
            raise ValueError("ResultSequence entries and records must be the same length.")

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int):
        entry = self._entries[index]
        if isinstance(entry, DistributedFailure):
            raise entry
        return entry

    def __iter__(self) -> Iterator[Any]:
        for index in range(len(self._entries)):
            yield self[index]

    @property
    def entries(self) -> tuple[Any, ...]:
        """Union view (``Result | DistributedFailure``) without raising."""

        return self._entries

    @property
    def records(self) -> tuple[ExecutionRecord, ...]:
        return self._records

    @property
    def failures(self) -> tuple[DistributedFailure, ...]:
        return tuple(entry for entry in self._entries if isinstance(entry, DistributedFailure))

    @property
    def failed(self) -> bool:
        return any(isinstance(entry, DistributedFailure) for entry in self._entries)

    def results(self) -> tuple[Any, ...]:
        """Return every Result in order, raising the first failure if any."""

        for entry in self._entries:
            if isinstance(entry, DistributedFailure):
                raise entry
        return self._entries


__all__ = [
    "DistributedFailure",
    "ExecutionRecord",
    "FailureKind",
    "ResultSequence",
]
