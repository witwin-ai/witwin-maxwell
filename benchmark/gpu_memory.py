from __future__ import annotations

import threading

import torch


class GpuMemorySampler(threading.Thread):
    """Sample driver-level GPU usage and report the peak above baseline."""

    def __init__(self, interval_s: float = 0.02):
        super().__init__(daemon=True)
        self._interval_s = float(interval_s)
        self._stop_event = threading.Event()
        free, total = torch.cuda.mem_get_info()
        self._total_bytes = int(total)
        self._baseline_used = self._total_bytes - int(free)
        self._peak_used = self._baseline_used

    def run(self) -> None:
        while not self._stop_event.is_set():
            free, _ = torch.cuda.mem_get_info()
            self._peak_used = max(self._peak_used, self._total_bytes - int(free))
            self._stop_event.wait(self._interval_s)

    def stop(self) -> int:
        free, _ = torch.cuda.mem_get_info()
        self._peak_used = max(self._peak_used, self._total_bytes - int(free))
        self._stop_event.set()
        self.join()
        return max(self._peak_used - self._baseline_used, 0)


def release_gpu_caches() -> None:
    """Release Torch and CuPy cache blocks before an isolated benchmark run."""
    import cupy as cp

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


__all__ = ["GpuMemorySampler", "release_gpu_caches"]
