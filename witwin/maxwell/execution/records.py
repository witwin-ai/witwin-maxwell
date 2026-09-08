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


@dataclass(frozen=True)
class BatchItem:
    """One completed, failed, or cancelled variant, retaining its input identity."""

    index: int
    variant_id: str
    result: Any | None
    failure: DistributedFailure | None
    record: ExecutionRecord

    def __post_init__(self):
        if (self.result is None) == (self.failure is None):
            raise ValueError("Exactly one of result and failure must be present.")
        if self.index != self.record.index:
            raise ValueError("BatchItem index must match its execution record.")

    def raise_for_error(self):
        if self.failure is not None:
            raise self.failure


class BatchResult(ResultSequence):
    """Ordinary Results with stable variant IDs and explicit tensor stacking."""

    def __init__(self, entries, records, variant_ids):
        super().__init__(entries, records)
        self.variant_ids = tuple(variant_ids)
        if (len(self.variant_ids) != len(self) or len(set(self.variant_ids)) != len(self)
                or any(not isinstance(v, str) or not v for v in self.variant_ids)):
            raise ValueError("BatchResult requires one unique nonempty variant ID per entry.")

    def stack(self, selector, *, device=None):
        import torch
        values = [selector(result) for result in self]
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise TypeError("BatchResult.stack selector must return tensors.")
        if not values:
            raise ValueError("Cannot stack an empty BatchResult.")
        if any(value.shape != values[0].shape or value.dtype != values[0].dtype for value in values[1:]):
            raise ValueError("Selected batch tensors must have identical shapes and dtypes.")
        if device is None and any(v.device != values[0].device for v in values[1:]):
            raise ValueError("Mixed-device results require an explicit stack device.")
        return torch.stack(values if device is None else [v.to(device) for v in values])

    def save(self, directory):
        import json
        from pathlib import Path
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        if any(path.iterdir()):
            raise FileExistsError("BatchResult.save requires an empty output directory.")
        items = []
        for index, (entry, record) in enumerate(zip(self.entries, self.records)):
            item = {"index": index, "variant_id": self.variant_ids[index],
                    "device": record.device, "status": record.status,
                    "estimated_bytes": record.estimated_bytes, "wall_time_s": record.wall_time_s,
                    "device_time_s": record.device_time_s}
            if isinstance(entry, DistributedFailure):
                item["failure"] = {"kind": entry.kind.value, "detail": entry.detail}
            else:
                entry.save(path / f"result-{index}.pt")
            items.append(item)
        temporary = path / "manifest.json.tmp"
        temporary.write_text(json.dumps({"schema_version": 1, "items": items}, indent=2), encoding="utf-8")
        temporary.replace(path / "manifest.json")

    @classmethod
    def load(cls, directory, *, scenes, map_location="cpu"):
        import json
        from pathlib import Path
        from ..result import Result
        path = Path(directory)
        payload = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported batch result schema version.")
        items = payload["items"]
        if len(scenes) != len(items):
            raise ValueError("One Scene must be supplied for each saved batch variant.")
        entries, records, ids = [], [], []
        for index, item in enumerate(items):
            if item["index"] != index:
                raise ValueError("Batch result manifest indices must be contiguous and ordered.")
            failure = None
            if "failure" in item:
                failure = DistributedFailure(index=index, device=item["device"],
                    kind=item["failure"]["kind"], message=item["failure"]["detail"])
                entry = failure
            else:
                entry = Result.load(path / f"result-{index}.pt", scene=scenes[index], map_location=map_location)
            entries.append(entry)
            ids.append(item["variant_id"])
            records.append(ExecutionRecord(index=index, device=item["device"], status=item["status"],
                estimated_bytes=item["estimated_bytes"], wall_time_s=item["wall_time_s"],
                device_time_s=item["device_time_s"], failure=failure))
        return cls(entries, records, ids)


__all__ += ["BatchItem", "BatchResult"]
