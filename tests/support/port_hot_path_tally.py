"""Shared profiler-tally helper for the RF port hot-path op-count gates.

Gate class: ``perf-opcount`` (see ``docs/reference/gate-classification.md`` §5).
This is a non-numerical performance gate: it counts host/device dispatch events
(kernel launches, allocations, device-to-device copies, host-side transfers,
scalar syncs, device memory), never wall-clock time.

Both op-count call sites profile the *same* per-step port apply+observe schedule;
this module is the single implementation of the profiled window and its event
classification so they cannot drift apart (CLAUDE.md forbids duplicate
implementations). Callers differ only in how they present the raw tally:

* ``tests/rf/performance/profile_port_hot_path.py`` normalizes to per-step counts
  for its reproducible JSON artifact;
* ``tests/rf/lumped/test_fdtd_port_end_to_end.py`` asserts per-window ceilings.

Both express their window through :func:`tally_hot_path_window`.
"""

from __future__ import annotations

from typing import Callable


# Canonical tally keys. ``device_mem`` is the summed self device-memory usage in
# bytes over the profiled window.
PORT_HOT_PATH_TALLY_KEYS = (
    "launches",
    "allocs",
    "memcpy_dtod",
    "memcpy_hostside",
    "scalar_sync",
    "device_mem",
)


def tally_hot_path_window(
    step: Callable[[], None],
    *,
    warmup_steps: int,
    profiled_steps: int,
) -> dict[str, int]:
    """Profile ``profiled_steps`` invocations of ``step`` and tally raw op counts.

    ``warmup_steps`` untimed invocations absorb lazy JIT/allocator setup before
    the profiled window opens, so the tally reflects only steady-state per-step
    dispatch. The return value is the raw window total (not per-step); callers
    normalize as needed.
    """

    import torch

    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")
    if profiled_steps < 1:
        raise ValueError("profiled_steps must be positive.")

    for _ in range(warmup_steps):
        step()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
        acc_events=True,
    ) as prof:
        for _ in range(profiled_steps):
            step()
    torch.cuda.synchronize()

    tally = {key: 0 for key in PORT_HOT_PATH_TALLY_KEYS}
    for event in prof.key_averages():
        key = event.key
        if "cudaLaunchKernel" in key:
            tally["launches"] += event.count
        if key in ("aten::empty", "aten::empty_strided", "aten::empty_like"):
            tally["allocs"] += event.count
        if "Memcpy DtoD" in key:
            tally["memcpy_dtod"] += event.count
        if "Memcpy HtoD" in key or "Memcpy DtoH" in key:
            tally["memcpy_hostside"] += event.count
        if key in ("aten::item", "aten::_local_scalar_dense"):
            tally["scalar_sync"] += event.count
        tally["device_mem"] += max(0, getattr(event, "self_device_memory_usage", 0))
    return tally
