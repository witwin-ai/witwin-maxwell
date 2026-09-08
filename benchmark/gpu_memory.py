from __future__ import annotations

import os
import subprocess
import threading

import torch


class GpuMemorySampler(threading.Thread):
    """Sample driver-level GPU usage and report the peak above baseline."""

    def __init__(self, interval_s: float = 0.02, *, device=None):
        super().__init__(daemon=True)
        self._interval_s = float(interval_s)
        if self._interval_s <= 0:
            raise ValueError("interval_s must be positive.")
        self._device = torch.device(device or f"cuda:{torch.cuda.current_device()}")
        self._stop_event = threading.Event()
        free, total = torch.cuda.mem_get_info(self._device)
        self._total_bytes = int(total)
        self._baseline_used = self._total_bytes - int(free)
        self._peak_used = self._baseline_used

    def run(self) -> None:
        while not self._stop_event.is_set():
            free, _ = torch.cuda.mem_get_info(self._device)
            self._peak_used = max(self._peak_used, self._total_bytes - int(free))
            self._stop_event.wait(self._interval_s)

    def stop(self) -> int:
        free, _ = torch.cuda.mem_get_info(self._device)
        self._peak_used = max(self._peak_used, self._total_bytes - int(free))
        self._stop_event.set()
        self.join()
        return max(self._peak_used - self._baseline_used, 0)

    @property
    def peak_total_bytes(self) -> int:
        return self._peak_used

    @property
    def baseline_total_bytes(self) -> int:
        return self._baseline_used


class GpuComputeSampler(threading.Thread):
    """Observe competing compute processes without changing device state."""

    def __init__(self, devices, interval_s=.5):
        super().__init__(daemon=True)
        self.interval_s = float(interval_s)
        if self.interval_s <= 0:
            raise ValueError("interval_s must be positive.")
        self.uuids = {str(torch.cuda.get_device_properties(d).uuid).lower().removeprefix("gpu-") for d in devices}
        self.pid = os.getpid()
        self.samples = 0
        self.foreign = set()
        self.errors = []
        self._stop_event = threading.Event()

    def sample(self):
        try:
            output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits"], text=True, timeout=5)
            for line in output.splitlines():
                uuid, pid = [value.strip() for value in line.split(",")]
                if uuid.lower().removeprefix("gpu-") in self.uuids and int(pid) != self.pid:
                    self.foreign.add((uuid, int(pid)))
            self.samples += 1
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            self.errors.append(str(exc))

    def run(self):
        while not self._stop_event.is_set():
            self.sample()
            self._stop_event.wait(self.interval_s)

    def stop(self):
        self._stop_event.set()
        self.join()
        self.sample()
        return dict(interval_s=self.interval_s, samples=self.samples,
            foreign_compute_processes=sorted(self.foreign), errors=self.errors,
            exclusive_compute_observed=self.samples > 0 and not self.foreign and not self.errors,
            limitation="Periodic process sampling can miss activity between samples; display workloads are not excluded.")


def release_gpu_caches() -> None:
    """Release Torch and CuPy cache blocks before an isolated benchmark run."""
    import cupy as cp

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


__all__ = ["GpuMemorySampler", "GpuComputeSampler", "release_gpu_caches"]
