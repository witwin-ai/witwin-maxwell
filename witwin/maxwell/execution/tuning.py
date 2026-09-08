"""Opt-in empirical policy selection using isolated, complete Batch runs."""
from __future__ import annotations

import copy
import contextlib
import gc
import statistics
import time
from collections.abc import Mapping
from dataclasses import replace
from itertools import chain

import torch

from .plan import BatchPlanReport


def isolate_scene_module(module, pools, memo):
    """Deep-copy trial state, charging CUDA tensors to their device accounts."""
    visited = set()
    def seed(value):
        if id(value) in memo or id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, torch.Tensor):
            pool = pools.get(str(value.device))
            with (torch.cuda.device(value.device) if value.is_cuda else contextlib.nullcontext()), (
                    pool.activate() if pool is not None else contextlib.nullcontext()):
                # Tensor deepcopy preserves storage aliases and strided views;
                # detaching also supports derived non-leaf module attributes.
                copied = copy.deepcopy(value.detach(), memo).requires_grad_(value.requires_grad)
                if isinstance(value, torch.nn.Parameter):
                    copied = torch.nn.Parameter(copied, requires_grad=value.requires_grad)
            memo[id(value)] = copied
        elif isinstance(value, Mapping):
            for child in value.values():
                seed(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                seed(child)
        elif (isinstance(value, torch.nn.Module) or type(value).__module__.startswith("witwin.")) and hasattr(value, "__dict__"):
            for child in vars(value).values():
                seed(child)
    seed(module)
    return copy.deepcopy(module, memo)


def _candidate_policies(batch):
    policy = batch.execution
    group_sizes = [len(group["variant_indices"]) for group in batch.plan().groups]
    maximum = min(max(group_sizes), policy.max_batch_size)
    capacities = {min(len(batch), policy.batch_size)} if policy.batch_size != "auto" else {maximum}
    if policy.batch_size == "auto":
        size = 1
        while size <= maximum:
            capacities.add(size)
            size *= 2
    slots = (1, 2) if policy.slots_per_device == "auto" else (policy.slots_per_device,)
    chunks = (16, 32, 64) if policy.graph_steps == "auto" and policy.graph != "off" else (policy.graph_steps,)
    candidates = []
    for q in slots:
        sizes = set(capacities)
        if policy.batch_size == "auto":
            workers = len(policy.devices) * q
            sizes.add(min(maximum, (max(group_sizes) + workers - 1) // workers))
        for b in sorted(sizes):
            # Extra slots cannot overlap work when there are no additional
            # independent cohorts. Explicit Q remains an explicit user policy.
            cohorts = sum((count + b - 1) // b for count in group_sizes)
            if policy.slots_per_device == "auto" and q == 2 and cohorts <= len(policy.devices):
                continue
            candidates.extend(dict(batch_size=b, slots_per_device=q, graph_steps=k) for k in chunks)
    return candidates


def _synchronize(devices):
    for device in devices:
        torch.cuda.synchronize(device)


def _consume(prepared):
    # Full requested physics, checkpoints, finalization and export participate.
    # Drop results as delivered: user-retained outputs are a separate workload.
    with prepared.iter_results() as iterator:
        for item in iterator:
            if item.failure is not None:
                raise item.failure.exception or item.failure
            del item
    _synchronize(prepared.batch.execution.devices)


def _peak_bytes(prepared):
    if hasattr(prepared, "_accounts"):
        return {device: account.stats()["peak_charged_bytes"]
                for device, account in prepared._accounts.items()}
    return {prepared.device: prepared._budget.stats()["peak_charged_bytes"]}


def _select_candidate(measurements, *, auto_slots):
    eligible = measurements
    if auto_slots:
        singles = [m for m in measurements if m["policy"]["slots_per_device"] == 1]
        if not singles:
            raise RuntimeError("Automatic slots require a successful single-slot baseline; retry tuning or set slots_per_device explicitly.")
        if singles:
            best_single = min(singles, key=lambda m: m["median_s"])
            # Require a repeatable >=5% gain, including the observed spread.
            eligible = [m for m in measurements if m["policy"]["slots_per_device"] == 1
                        or max(m["samples_s"]) <= .95 * min(best_single["samples_s"])]
    fastest = min(m["median_s"] for m in eligible)
    near = [m for m in eligible if m["median_s"] <= fastest * 1.05]
    return min(near, key=lambda m: (sum(m["peak_bytes"].values()),
                                  m["policy"]["slots_per_device"], m["median_s"]))


def prepare_tuned_batch(batch):
    """Measure finite B/Q/K candidates without publishing trial Results.

    Tuning measures hot, streamed forward throughput, including tape creation
    when gradients are enabled. It never calls backward or updates parameters.
    The final context is rebuilt so no trial state or captured graph leaks into
    the first public run. Explicit policies remain fixed during the search.
    """
    from .batch_runtime import PreparedBatch
    if not torch.cuda.is_available():
        raise RuntimeError("Batch throughput tuning requires an accessible CUDA device.")
    started = time.perf_counter()
    measurements, rejected = [], []
    devices = batch.execution.devices
    limits = batch.execution.memory_limit_bytes
    if limits is None:
        limits = {d: max(0, int(torch.cuda.mem_get_info(d)[0]) - batch.execution.reserve_bytes) for d in devices}
    rng_devices = {torch.device(d).index for d in devices}
    for model in batch.simulations:
        device = torch.device(model.scene.device)
        if device.type == "cuda":
            rng_devices.add(device.index if device.index is not None else torch.cuda.current_device())
        if model.scene_module is not None:
            rng_devices.update(value.device.index for value in chain(
                model.scene_module.parameters(), model.scene_module.buffers()) if value.is_cuda)
    # Each policy starts from the same random state, and unpublished trial
    # materializations must not update the caller's module buffers or counters.
    for candidate in _candidate_policies(batch):
        with torch.random.fork_rng(devices=sorted(rng_devices)):
            trial = copy.copy(batch)
            trial.execution = replace(batch.execution, **candidate, tune=False, memory_limit_bytes=limits)
            trial._isolate_scene_modules = True
            prepared = None
            try:
                began = time.perf_counter()
                prepared = PreparedBatch(trial)
                prepare_s = time.perf_counter() - began
                _consume(prepared)
                samples = []
                for _ in range(3):
                    began = time.perf_counter()
                    _consume(prepared)
                    samples.append(time.perf_counter() - began)
                measurements.append(dict(policy=candidate, prepare_s=prepare_s, samples_s=samples,
                    median_s=statistics.median(samples), peak_bytes=_peak_bytes(prepared)))
            except torch.cuda.OutOfMemoryError as exc:
                rejected.append(dict(policy=candidate, reason="capacity", message=str(exc)))
            finally:
                if prepared is not None:
                    prepared.close()
                prepared = None
                gc.collect()
                _synchronize(devices)
    if not measurements:
        raise torch.cuda.OutOfMemoryError("No Batch tuning candidate fit the device budgets; reduce outputs or max_batch_size.")
    selected = _select_candidate(measurements, auto_slots=batch.execution.slots_per_device == "auto")
    final = copy.copy(batch)
    final.execution = replace(batch.execution, **selected["policy"], tune=False, memory_limit_bytes=limits)
    final._auto_capacity = batch.execution.batch_size == "auto"
    # Competing allocations can change after measurement. Preserve automatic
    # bounded reduction without silently altering an explicitly requested B.
    rebuild_retries = []
    while True:
        try:
            prepared = PreparedBatch(final)
            break
        except torch.cuda.OutOfMemoryError:
            if not final._auto_capacity or final.execution.batch_size == 1:
                raise
            rebuild_retries.append(final.execution.batch_size)
            final.execution = replace(final.execution, batch_size=max(1, final.execution.batch_size // 2))
            gc.collect()
    prepared._tuning_report = dict(tuned=True, candidate_measurements=measurements,
        rejected_candidates=rejected, selected_policy=selected["policy"],
        tuning=dict(scope="complete_forward_streamed_results", warmups=1, repeats=3,
                    includes_backward=False, records_grad=torch.is_grad_enabled(),
                    elapsed_s=time.perf_counter() - started, rebuild_capacity_retries=rebuild_retries,
                    selected_relative_throughput=min(m["median_s"] for m in measurements) / selected["median_s"]))
    report = prepared.report.to_dict()
    report.update(prepared._tuning_report)
    prepared.report = BatchPlanReport.from_dict(report)
    return prepared
