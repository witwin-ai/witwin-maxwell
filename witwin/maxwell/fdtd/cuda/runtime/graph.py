from __future__ import annotations

from collections.abc import Callable

import torch


class CudaGraphRunner:
    def __init__(self, *, enabled: bool = True, warmup_steps: int = 1):
        self.enabled = bool(enabled)
        self.warmup_steps = max(0, int(warmup_steps))
        self.graph: torch.cuda.CUDAGraph | None = None

    def capture(self, fn: Callable[[], None]) -> Callable[[], None]:
        if not self.enabled:
            return fn
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA Graph capture requires CUDA.")

        for _ in range(self.warmup_steps):
            fn()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize()
        self.graph = graph

        def replay() -> None:
            graph.replay()

        return replay
