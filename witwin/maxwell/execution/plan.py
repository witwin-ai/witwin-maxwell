from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class ExecutionTask:
    """One immutable unit of ensemble work in submission order.

    ``run`` executes the task on the leased device and returns its ``Result``.
    It must be self-contained: the shared execution layer knows nothing about
    electromagnetic semantics, only that a task runs on a device and returns a
    value or raises.
    """

    index: int
    run: Callable[[torch.device], object]
    estimated_bytes: int | None = None
    label: str | None = None


@dataclass(frozen=True)
class ExecutionPlan:
    """Immutable ordered task list plus placement and concurrency policy."""

    tasks: tuple[ExecutionTask, ...]
    placement: str
    max_concurrency: int
    fail_fast: bool

    def __post_init__(self) -> None:
        tasks = tuple(self.tasks)
        for expected, task in enumerate(tasks):
            if task.index != expected:
                raise ValueError(
                    "ExecutionPlan tasks must be contiguously indexed in submission order."
                )
        object.__setattr__(self, "tasks", tasks)
        if self.placement not in {"round_robin", "memory_aware"}:
            raise ValueError("placement must be 'round_robin' or 'memory_aware'.")
        if isinstance(self.max_concurrency, bool) or not isinstance(self.max_concurrency, int):
            raise TypeError("max_concurrency must be an integer.")
        if self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be > 0.")
        if not isinstance(self.fail_fast, bool):
            raise TypeError("fail_fast must be a bool.")

    def __len__(self) -> int:
        return len(self.tasks)


__all__ = ["ExecutionPlan", "ExecutionTask"]
