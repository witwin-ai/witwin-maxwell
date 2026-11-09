"""CPU-only F1 observer GC timing and profiles, without time stepping.

Run: CUDA_VISIBLE_DEVICES="" python -m benchmark.core_path.f1b_observer_timing
Both arms use current code; only this script substitutes a no-op GC pause.
Forced collections happen outside measured intervals. Unrelated setup/stepping
and user callbacks are never enclosed in a library GC pause.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
import cProfile
import gc
import json
import pstats
import statistics
import time
from unittest.mock import patch

import torch

from benchmark.datagen.scenes import scene_builders
from tests.monitors.observers.test_point_observer_batch_resolve import make_solver, fill_groups
from witwin.maxwell.compiler import monitors
from witwin.maxwell.fdtd import observers
from witwin.maxwell.result import _normalize_monitor_data_mapping


def pause_mode(enabled):
    stack = ExitStack()
    if not enabled:
        for module in (monitors, observers):
            stack.enter_context(patch.object(module, "paused_cyclic_gc", nullcontext))
    return stack


def measure(operation):
    """Keep the return value alive until timing and GC callbacks have stopped."""
    gc.collect()
    gc.enable()
    counts = [0, 0, 0]
    gc_seconds = 0.0
    collection_start = 0.0

    def callback(phase, info):
        nonlocal gc_seconds, collection_start
        if phase == "start":
            collection_start = time.perf_counter()
        else:
            gc_seconds += time.perf_counter() - collection_start
            counts[info["generation"]] += 1

    gc.callbacks.append(callback)
    try:
        start = time.perf_counter()
        value = operation()
        elapsed = time.perf_counter() - start
    finally:
        gc.callbacks.remove(callback)
    del value
    return dict(wall_seconds=elapsed, gc_seconds=gc_seconds, collections=counts)


def summarize(samples):
    return dict(
        median_seconds=statistics.median(row["wall_seconds"] for row in samples),
        median_gc_seconds=statistics.median(row["gc_seconds"] for row in samples),
        median_collections=[statistics.median(row["collections"][gen] for row in samples) for gen in range(3)],
        samples=samples,
    )


def profile(operation, top):
    gc.collect()
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        value = operation()
    finally:
        profiler.disable()
    del value
    stats = pstats.Stats(profiler)
    rows = []
    for (filename, line, name), (primitive, calls, own, cumulative, _) in sorted(
        stats.stats.items(), key=lambda item: item[1][2], reverse=True,
    )[:top]:
        rows.append(dict(function=f"{filename}:{line}:{name}", calls=calls,
                         primitive_calls=primitive, tottime_seconds=own, cumulative_seconds=cumulative))
    return dict(total_seconds=stats.total_tt, top_tottime=rows)


def run(args):
    torch.set_num_threads(1)
    builder = next(builder for builder in scene_builders() if builder.name == args.scene)
    sample = builder.build(device="cpu")
    solver = make_solver(sample.scene)
    frequencies = tuple(sample.fdtd_kwargs["frequencies"])
    solver.observers = monitors.compile_fdtd_observers(sample.scene)
    prepare = lambda: observers.prepare_observers(solver, frequencies, "none", 8)
    prepare()
    fill_groups(solver)
    payload = observers.get_observer_results(solver)
    operations = {
        "compile_fdtd_observers": lambda: monitors.compile_fdtd_observers(sample.scene),
        "prepare_observers": prepare,
        "get_observer_results": lambda: observers.get_observer_results(solver),
        "result_monitor_normalization": lambda: _normalize_monitor_data_mapping(payload),
    }
    samples = {arm: {stage: [] for stage in operations} for arm in ("without_pause", "with_pause")}
    for stage, operation in operations.items():
        if stage == "get_observer_results":
            fill_groups(solver)
        # Warm each stage, then alternate arm order to reduce ordering bias.
        operation()
        for repeat in range(args.repeats):
            for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
                arm = "with_pause" if enabled else "without_pause"
                with pause_mode(enabled):
                    samples[arm][stage].append(measure(operation))
    rows = {arm: {stage: summarize(values) for stage, values in stages.items()}
            for arm, stages in samples.items()}
    profiles = {"prepare_observers": profile(prepare, args.profile_top)}
    fill_groups(solver)
    profiles["get_observer_results"] = profile(operations["get_observer_results"], args.profile_top)
    return dict(scene=args.scene, device="cpu", threads=1, repeats=args.repeats,
                monitors=len(sample.scene.monitors), observers=len(solver.observers),
                gc_thresholds=gc.get_threshold(), timings=rows, profiles_with_pause=profiles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--scene", default="f1_box_small")
    parser.add_argument("--profile-top", type=int, default=15)
    args = parser.parse_args()
    if args.repeats < 1 or args.profile_top < 1:
        parser.error("--repeats and --profile-top must be positive")

    def forbid_cuda(*args, **kwargs):
        raise AssertionError("CPU observer timing must never initialize CUDA")

    enabled = gc.isenabled()
    try:
        with patch.object(torch.cuda, "_lazy_init", forbid_cuda):
            report = run(args)
    finally:
        (gc.enable if enabled else gc.disable)()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
