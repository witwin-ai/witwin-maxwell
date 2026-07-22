from __future__ import annotations

import contextlib
import threading
import warnings
from collections.abc import Callable

import torch

_suspend_lock = threading.Lock()
_suspend_depth = 0


@contextlib.contextmanager
def suspend_capture():
    """Disable CUDA-graph capture process-wide for the duration of the block.

    Graph capture is a process-global operation, not a per-device one. Only one
    capture may be underway in a process at a time, and PyTorch captures in the
    default ``cudaStreamCaptureModeGlobal`` mode, which makes an ordinary
    synchronizing call (``Tensor.item()``, a device-to-host copy, an allocation)
    in *any* other thread fail with ``cudaErrorStreamCaptureUnsupported`` for as
    long as the capture is open. Executors that run several independent solves
    concurrently in worker threads must therefore step eagerly: one task's
    capture would otherwise abort an unrelated task on a different GPU.

    Reference counted, so nested executors compose.
    """

    global _suspend_depth
    with _suspend_lock:
        _suspend_depth += 1
    try:
        yield
    finally:
        with _suspend_lock:
            _suspend_depth -= 1


def capture_suspended() -> bool:
    """True while a concurrent executor owns the process (see suspend_capture)."""

    return _suspend_depth > 0


class CudaGraphRunner:
    """Capture a fixed-shape step closure into a CUDA graph and return a replay.

    ``device`` is mandatory and must be the device the captured work runs on.
    ``torch.cuda.graph`` opens its capture stream on whatever device happens to
    be current, and caches that stream on the class after the first capture, so
    a capture whose tensors live on a different device records an *empty* graph:
    the warmup and capture launches still execute eagerly on the tensors' own
    stream, and every subsequent ``replay()`` is a silent no-op that leaves the
    fields frozen at the warmup state. Binding the device here, and supplying a
    capture stream created on it, is what makes capture valid for devices other
    than the process default.

    Capture is additionally guarded against recording nothing: an empty graph is
    raised as a ``RuntimeError`` so the callers' existing ``except Exception``
    fallback degrades to eager stepping instead of replaying a no-op.
    """

    def __init__(self, *, device, enabled: bool = True, warmup_steps: int = 1):
        self.enabled = bool(enabled)
        self.warmup_steps = max(0, int(warmup_steps))
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(
                f"CUDA Graph capture requires a CUDA device, got {self.device}."
            )
        self.graph: torch.cuda.CUDAGraph | None = None

    def capture(self, fn: Callable[[], None]) -> Callable[[], None]:
        if not self.enabled:
            return fn
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA Graph capture requires CUDA.")
        if capture_suspended():
            # Raised rather than silently returning ``fn`` so the callers' existing
            # ``except Exception`` fallbacks take the eager path *and* leave their
            # ``*_graph_active`` flags cleared.
            raise RuntimeError(
                "CUDA Graph capture is suspended while a concurrent executor shares "
                "this process; stepping eagerly."
            )

        with torch.cuda.device(self.device):
            for _ in range(self.warmup_steps):
                fn()
            torch.cuda.synchronize(self.device)

            graph = torch.cuda.CUDAGraph()
            # Explicit per-capture stream on the target device: torch.cuda.graph
            # otherwise reuses a stream cached on its class from the first
            # capture in the process, which may belong to a different device.
            capture_stream = torch.cuda.Stream(device=self.device)
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                with torch.cuda.graph(graph, stream=capture_stream):
                    fn()
            torch.cuda.synchronize(self.device)

        empty = False
        for entry in recorded:
            if "CUDA Graph is empty" in str(entry.message):
                empty = True
                continue
            warnings.warn_explicit(
                entry.message, entry.category, entry.filename, entry.lineno
            )
        if empty:
            raise RuntimeError(
                f"CUDA Graph capture on {self.device} recorded no work; refusing a "
                "no-op replay and falling back to eager stepping."
            )

        self.graph = graph

        def replay() -> None:
            graph.replay()

        return replay
