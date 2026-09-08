"""Developer profiling of batch dispatch, field capture, and cohort compilation."""
from contextlib import ExitStack
import cProfile
import functools
import platform
import pstats
import statistics
import time
from unittest.mock import patch

import torch
import witwin.maxwell as mw
from benchmark.batch import build_simulation
from witwin.maxwell.execution.batch_runtime import PreparedBatch
from witwin.maxwell.execution.capacity import tensor_memory_manifest
from witwin.maxwell.fdtd.cuda.runtime.graph import CudaGraphRunner
from witwin.maxwell.fdtd.runtime.batch import NativeFieldSlot
from witwin.maxwell.fdtd.solver import FDTD
from witwin.maxwell.fdtd.observers import BatchObserverEngine
from witwin.maxwell.fdtd.excitation.temporal import BatchSourceEngine


def _environment(device):
    return {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "python": platform.python_version(), "gpu": torch.cuda.get_device_name(device)}


def audit_tuning(workload, *, devices, variants, capacity, warmups=2, repeats=5):
    """Record public tuning trials, independent hot repeats and spectrum parity."""
    from dataclasses import asdict
    from witwin.maxwell.execution.tuning import _consume
    if variants < 1 or capacity < 1 or warmups < 0 or repeats < 1:
        raise ValueError("Tuning audits require positive counts and nonnegative warmups.")
    sims = [build_simulation(workload, index=i, device=devices[0]) for i in range(variants)]
    batch = mw.Simulation.batch(sims, execution=mw.BatchExecution(
        devices=devices, batch_size="auto", max_batch_size=capacity,
        slots_per_device="auto", graph="required", tune=True))
    with torch.no_grad(), batch.prepare() as prepared:
        for _ in range(warmups):
            _consume(prepared)
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            _consume(prepared)
            samples.append(time.perf_counter() - started)
        report = prepared.report.to_dict()
        actual = prepared.run()
        actual.results()
    errors = []
    for i, sim in enumerate(sims):
        reference = sim.run().raw_monitor("probe")["components"]["Ez"]
        value = actual[i].raw_monitor("probe")["components"]["Ez"].to(reference.device)
        torch.testing.assert_close(value, reference, rtol=1e-4, atol=1e-6)
        errors.append(float((value - reference).abs().max()))
    return dict(schema_version=1, kind="batch_tuning_audit", workload=asdict(workload),
        variants=variants, devices=list(devices), environment=_environment(devices[0]),
        plan=report, holdout_samples_s=samples, holdout_median_s=statistics.median(samples),
        max_probe_absolute_error=max(errors),
        measured_candidate_threshold_passed=report["tuning"]["selected_relative_throughput"] >= .9,
        timing_scope="complete_forward_streamed_results_after_prepare",
        multi_gpu_scaling_acceptance="not_measured_requires_exclusive_device_comparison")


def audit_regional_materials(workload, *, device, warmups=2, repeats=5):
    """Interleave full/local compilation with numerical checks outside timing."""
    from dataclasses import replace
    from witwin.maxwell.scene import prepare_scene
    from witwin.maxwell.compiler.materials import LocalMaterialCache, compile_material_model
    if warmups < 0 or repeats < 1:
        raise ValueError("Regional audits require nonnegative warmups and positive repeats.")
    rows = []
    with torch.cuda.device(device), torch.no_grad():
        for mode in ("density", "structure"):
            scene = build_simulation(workload, device=device).scene
            scene = scene.clone(subpixel_samples=mw.SubpixelSpec(samples=2, averaging="polarized"))
            if mode == "density":
                scene.add_material_region(mw.MaterialRegion(name="rho",
                    geometry=mw.Box(size=(.03, .03, .03), device=device),
                    density=torch.full((4, 4, 4), .2, device=device), eps_bounds=(1., 4.)))
                cache = LocalMaterialCache(prepare_scene(scene), ("rho",))
                changed = scene.clone()
                changed.material_regions[0] = replace(changed.material_regions[0],
                    density=torch.full((4, 4, 4), .7, device=device))
            else:
                cache = LocalMaterialCache(prepare_scene(scene), structure_envelopes={
                    "design": mw.Box(position=(.08, 0., 0.), size=(.18, .18, .18), device=device)})
                changed = scene.clone()
                changed.structures[0] = mw.Structure(name="design",
                    geometry=mw.Box(position=(.09, 0., 0.), size=(.06, .06, .06), device=device),
                    material=mw.Material(eps_r=3.))
            prepared = prepare_scene(changed)
            times = {"full": [], "local": []}
            def compile(kind):
                return cache.compile(prepared) if kind == "local" else compile_material_model(
                    prepared, subpixel=prepared.subpixel)
            for round_index in range(warmups + repeats):
                for kind in (("full", "local") if round_index % 2 == 0 else ("local", "full")):
                    torch.cuda.synchronize(device)
                    began = time.perf_counter()
                    model = compile(kind)
                    torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - began
                    del model
                    if round_index >= warmups:
                        times[kind].append(elapsed)
            expected, actual = compile("full"), compile("local")
            errors = []
            def compare(a, b):
                if isinstance(b, torch.Tensor):
                    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
                    errors.append(float((a != b).any()) if b.dtype == torch.bool else float((a-b).abs().max()))
                elif isinstance(b, dict):
                    for key in b:
                        compare(a[key], b[key])
            compare(actual, expected)
            medians = {name: statistics.median(values) for name, values in times.items()}
            rows.append({"mode": mode, "wall_seconds": times, "median_seconds": medians,
                "subpixel_samples": list(prepared.subpixel.samples), "averaging": prepared.subpixel.averaging,
                "speedup": medians["full"]/medians["local"], "max_absolute_error": max(errors),
                "node_shape": [prepared.Nx, prepared.Ny, prepared.Nz],
                "patch_shape": [s.stop-s.start for s in cache.node_selection]})
    return {"kind": "batch_regional_material_audit", "environment": _environment(device),
            "workload": workload.name, "warmups": warmups, "repeats": repeats, "rows": rows,
            "scope": "Material compiler wall time; excludes solver construction, field stepping and initial bank creation."}


def _completed_run(prepared):
    result = prepared.run()
    result.results()
    torch.cuda.synchronize(prepared.device)


def _timed_run(prepared):
    started = time.perf_counter()
    _completed_run(prepared)
    return time.perf_counter() - started


def _annotated(function, label):
    @functools.wraps(function)
    def call(*args, **kwargs):
        with torch.profiler.record_function("audit." + label):
            return function(*args, **kwargs)
    return call


def _field_capture_ablation(prepared, repeats):
    def probe():
        return prepared.run().stack(lambda r: r.raw_monitor("probe")["components"]["Ez"])
    reference = probe()
    original = prepared._slot.advance
    with torch.cuda.device(prepared.device), torch.cuda.stream(prepared._stream):
        prepared._slot.reset()
        replay = CudaGraphRunner(device=prepared.device, warmup_steps=1).capture(original)
        prepared._slot.reset()
    controls, captured = [], []
    try:
        prepared._slot.advance = replay
        actual = probe()
        error = float((actual - reference).abs().max() / reference.abs().max().clamp_min(1e-30))
        if not torch.isfinite(actual).all() or error > 1e-4:
            raise AssertionError(f"Field capture changed the probe spectrum: {error}")
        for _ in range(repeats):
            for function, samples in ((original, controls), (replay, captured)):
                prepared._slot.advance = function
                samples.append(_timed_run(prepared))
    finally:
        prepared._slot.advance = original
    return {"control_s": controls, "field_graph_s": captured,
            "control_median_s": statistics.median(controls),
            "field_graph_median_s": statistics.median(captured),
            "relative_probe_error": error,
            "scope": "field block only; diagnostic capture in a quiescent process; not the public graph scheduler"}


def profile_batch(workload, *, device, capacity, warmups=2, repeats=5):
    """Profile a fully resident cohort, with separate uninstrumented latency."""
    if min(capacity, warmups, repeats) < 1:
        raise ValueError("Capacity, warmups, and repeats must be positive.")
    sims = [build_simulation(workload, index=i, device=device) for i in range(capacity)]
    batch = mw.Simulation.batch(sims, execution=mw.BatchExecution(
        devices=(device,), batch_size=capacity, graph="off"))
    started = time.perf_counter()
    with batch.prepare() as prepared:
        prepare_s = time.perf_counter() - started
        for _ in range(warmups):
            _completed_run(prepared)
        timings = [_timed_run(prepared) for _ in range(repeats)]
        profiler = cProfile.Profile()
        profiler.runcall(_completed_run, prepared)
        functions = [{"file": key[0].split("/witwin-maxwell/")[-1],
                      "line": key[1], "function": key[2], "calls": value[1],
                      "self_s": value[2], "cumulative_s": value[3]}
                     for key, value in pstats.Stats(profiler).stats.items()]
        manifest = tensor_memory_manifest({"slot": prepared._slot})
        wrappers = [(NativeFieldSlot, "advance", "fields"), (BatchSourceEngine, "advance", "source"),
                    (BatchObserverEngine, "advance", "observers"),
                    (FDTD, "_prepare_observers", "observer_prepare"),
                    (FDTD, "_synchronize_device", "synchronize"),
                    (PreparedBatch, "_export", "export"), (PreparedBatch, "_load", "load")]
        with ExitStack() as stack:
            for cls, name, label in wrappers:
                stack.enter_context(patch.object(cls, name, _annotated(getattr(cls, name), label)))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as trace:
                with torch.profiler.record_function("audit.run"):
                    _completed_run(prepared)
        keys = [{"name": e.key, "count": e.count, "cpu_total_us": e.cpu_time_total,
                 "self_cpu_us": e.self_cpu_time_total, "device_total_us": e.device_time_total,
                 "self_device_us": e.self_device_time_total} for e in trace.key_averages()]
        # CUDA user annotation ranges span idle gaps and are not device work.
        events = [e for e in trace.events() if e.device_type == torch.autograd.DeviceType.CUDA
                  and not e.name.startswith("audit.")]
        intervals = sorted((e.time_range.start, e.time_range.end) for e in events)
        merged, by_name = [], {}
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        for event in events:
            row = by_name.setdefault(event.name, {"name": event.name, "count": 0, "sum_us": 0.})
            row["count"] += 1
            row["sum_us"] += event.time_range.end - event.time_range.start
        output = {"schema_version": 1, "kind": "batch_dispatch_profile",
                  "environment": _environment(device), "device": device,
                  "workload": workload.name, "cells": workload.grid_cells,
                  "frequencies": len(workload.frequencies), "batch": capacity,
                  "steps": workload.time_steps, "warmups": warmups, "repeats": repeats,
                  "prepare_s": prepare_s, "warm_s": timings,
                  "warm_median_s": statistics.median(timings),
                  "cprofile": sorted(functions, key=lambda r: r["self_s"], reverse=True)[:35],
                  "profiler_scopes": [r for r in keys if r["name"].startswith("audit.")
                                      and r["cpu_total_us"] > 0],
                  "profiler_cpu_top": sorted(keys, key=lambda r: r["self_cpu_us"], reverse=True)[:25],
                  "gpu_events": sorted(by_name.values(), key=lambda r: r["sum_us"], reverse=True),
                  "gpu_busy_union_us": sum(end - start for start, end in merged),
                  "gpu_span_us": max(end for _, end in intervals) - intervals[0][0] if intervals else 0,
                  "storage_bytes": manifest["storage_bytes_by_device"],
                  "storage_manifest": manifest,
                  "field_shapes": {name: list(prepared._slot.buffers[name].shape)
                                   for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")},
                  "buffer_bytes": {name: value.numel() * value.element_size()
                                   for name, value in prepared._slot.buffers.items()},
                  "notes": ["Profiler adds overhead; use separate uninstrumented timings for latency.",
                            "GPU busy union is timeline activity, not SM occupancy or memory bandwidth."]}
        output["field_graph_ablation"] = _field_capture_ablation(prepared, repeats)
        return output


def audit_capacity(workload, *, device, variants, capacities, repeats=3):
    """Count real solver compilations when repeated runs rotate through cohorts."""
    if min(variants, repeats, *capacities) < 1:
        raise ValueError("Variants, capacities, and repeats must be positive.")
    sims = [build_simulation(workload, index=i, device=device) for i in range(variants)]
    rows = []
    original = mw.Simulation._build_fdtd_solver_for_scene
    counter = {"count": 0, "seconds": 0.}
    def measured(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            counter["count"] += 1
            counter["seconds"] += time.perf_counter() - started
    with patch.object(mw.Simulation, "_build_fdtd_solver_for_scene", measured):
        for capacity in capacities:
            batch = mw.Simulation.batch(sims, execution=mw.BatchExecution(
                devices=(device,), batch_size=capacity, graph="off"))
            with batch.prepare() as prepared:
                _completed_run(prepared)
                rounds = []
                for _ in range(repeats):
                    counter.update(count=0, seconds=0.)
                    elapsed = _timed_run(prepared)
                    rounds.append({"run_s": elapsed, "solver_builds": counter["count"],
                                   "solver_build_s": counter["seconds"]})
                rows.append({"capacity": capacity, "rounds": rounds,
                             "median_s": statistics.median(r["run_s"] for r in rounds)})
    return {"schema_version": 1, "kind": "batch_capacity_compilation_audit",
            "environment": _environment(device), "device": device,
            "workload": workload.name, "steps": workload.time_steps, "variants": variants,
            "rows": rows, "notes": ["Solver-build time includes material compilation and solver construction.",
                                      "One warmup per capacity; diagnostic sweep, not a throughput acceptance gate."]}
