"""Fixed workload inventory and live tensor audits for FDTD batch development."""
from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
import witwin.maxwell as mw
from witwin.maxwell.execution.capacity import tensor_memory_manifest
from benchmark.gpu_memory import GpuMemorySampler
from benchmark.paths import PLOTS_DIR


@dataclass(frozen=True)
class BatchWorkload:
    name: str
    grid_cells: int
    frequencies: tuple[float, ...]
    time_steps: int = 2000


BATCH_WORKLOADS = {
    "launch": BatchWorkload("launch", 16, (1e9,)),
    "bandwidth": BatchWorkload("bandwidth", 128, (1e9,)),
    "local_spectrum": BatchWorkload(
        "local_spectrum", 48, tuple(0.7e9 + i * 0.02e9 for i in range(31)),
    ),
}


def build_simulation(workload, *, index=0, identical=False, device="cuda:0"):
    """Build a fixed-grid problem with a bounded local material perturbation."""
    cell = 0.6 / workload.grid_cells
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.3, 0.3),) * 3),
        grid=mw.GridSpec.uniform(cell),
        boundary=mw.BoundarySpec.pml(num_layers=4), device=device,
    )
    scene.add_structure(mw.Structure(
        name="design",
        geometry=mw.Box(position=(0.08, 0.0, 0.0), size=(0.06, 0.06, 0.06), device=device),
        material=mw.Material(eps_r=2.0 if identical else 2.0 + 0.01 * index),
    ))
    scene.add_source(mw.PointDipole(
        name="drive", position=(-0.1, 0.0, 0.0), polarization="Ez", width=1.5 * cell,
        source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.5e9),
    ))
    scene.add_monitor(mw.PointMonitor(
        name="probe", position=(0.15, 0.0, 0.0), fields=("Ez",),
    ))
    return mw.Simulation.fdtd(
        scene, frequencies=workload.frequencies,
        run_time=mw.TimeConfig(time_steps=workload.time_steps, dt=cell / (2 * 3e8)),
        spectral_sampler=mw.SpectralSampler(window="none"),
        cuda_graph=False, full_field_dft=False,
    )


def audit(workload, *, device):
    """Measure prepare/run storage separately from driver and allocator peaks."""
    if not torch.cuda.is_available():
        raise RuntimeError("Batch GPU audits require an accessible CUDA driver and device.")
    with torch.cuda.device(device):
        torch.cuda.synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        sampler = GpuMemorySampler(device=device)
        sampler.start()
        started = time.perf_counter()
        try:
            simulation = build_simulation(workload, device=device)
            prepared = simulation.prepare()
            prepare_manifest = tensor_memory_manifest({"solver": prepared.solver})
            result = prepared.run()
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            completed_manifest = tensor_memory_manifest({"solver": prepared.solver, "result": result})
            allocated = torch.cuda.memory_allocated(device)
            peak_allocated = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
        finally:
            driver_peak_increment = sampler.stop()
    return {
        "schema_version": 1, "kind": "batch_single_run_audit",
        "workload": asdict(workload), "device": device,
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "python": platform.python_version(),
                        "gpu": torch.cuda.get_device_name(device)},
        "wall_time_s": elapsed, "timing_scope": "build_prepare_run_output_ready",
        "baseline_allocated_bytes": baseline,
        "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
        "driver_peak_increment_bytes": driver_peak_increment,
        "driver_sampling_interval_s": 0.02,
        "prepare_manifest": prepare_manifest, "completed_manifest": completed_manifest,
        "unattributed_live_bytes": allocated - completed_manifest["storage_bytes_by_device"].get(device, 0),
        "historical_memory_reproduction": "not_tested_business_geometry_unavailable",
        "concurrent_baselines": "pending",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--inventory", action="store_true", help="Print workloads without solver allocation")
    group.add_argument("--audit", choices=tuple(BATCH_WORKLOADS), help="Audit one independent GPU simulation")
    group.add_argument("--compare", choices=tuple(BATCH_WORKLOADS), help="Compare sequential/process/run_many baselines")
    group.add_argument("--profile", choices=tuple(BATCH_WORKLOADS), help="Profile native dispatch and diagnostic field capture")
    group.add_argument("--capacity-audit", choices=tuple(BATCH_WORKLOADS), help="Measure compilation while rotating cohorts")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--variants", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--capacities", type=int, nargs="+", default=[1, 2, 4, 10], help="Resident capacities for --capacity-audit")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--steps", type=int, help="Explicit step override; retained in the report")
    parser.add_argument("--identical", action="store_true", help="Use distinct copies of one physical design")
    parser.add_argument("--include-native", action="store_true", help="Also measure reusable native Batch with B=concurrency")
    parser.add_argument("--native-graph", choices=("off", "auto", "required"), default="off",
                        help="Graph policy for the native comparison tier")
    parser.add_argument("--update-report", action="store_true", help="Also update the baseline section of benchmark/RESULTS.md")
    args = parser.parse_args(argv)
    if args.update_report and not args.compare:
        parser.error("--update-report requires --compare.")
    if args.include_native and not args.compare:
        parser.error("--include-native requires --compare.")
    if args.native_graph != "off" and not (args.compare and args.include_native):
        parser.error("--native-graph requires --compare and --include-native.")
    if args.inventory:
        payload = {"schema_version": 1, "workloads": [asdict(w) for w in BATCH_WORKLOADS.values()],
                   "planned_variant_counts": [1, 10, 100], "deduplication": False,
                   "status": "baseline_instrumentation_only"}
    else:
        workload = BATCH_WORKLOADS[args.audit or args.compare or args.profile or args.capacity_audit]
        if args.steps is not None:
            if args.steps <= 0:
                parser.error("--steps must be positive.")
            workload = replace(workload, time_steps=args.steps)
        if args.audit:
            payload = audit(workload, device=args.device)
        elif args.profile:
            from benchmark.batch_profile import profile_batch
            payload = profile_batch(workload, device=args.device, capacity=args.concurrency,
                                    warmups=args.warmups, repeats=args.repeats)
        elif args.capacity_audit:
            from benchmark.batch_profile import audit_capacity
            payload = audit_capacity(workload, device=args.device, variants=args.variants,
                                     capacities=args.capacities, repeats=args.repeats)
        else:
            payload = benchmark_baselines(workload, device=args.device, variants=args.variants,
                concurrency=args.concurrency, identical=args.identical,
                warmups=args.warmups, repeats=args.repeats, include_native=args.include_native,
                native_graph=args.native_graph)
    encoded = json.dumps(payload, indent=2) + "\n"
    if args.audit or args.compare or args.profile or args.capacity_audit or args.output:
        suffix = "audit" if args.audit else f"n{args.variants}-{'identical' if args.identical else 'varied'}"
        if args.profile:
            suffix = f"b{args.concurrency}-profile"
        elif args.capacity_audit:
            suffix = f"n{args.variants}-capacity"
        name = args.audit or args.compare or args.profile or args.capacity_audit
        path = args.output or PLOTS_DIR / "batch" / f"{name}-{suffix}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding="utf-8")
        if args.update_report:
            from benchmark.report import record_batch_baseline
            record_batch_baseline(payload, path.resolve())
    print(encoded, end="")


def _probe_snapshot(result):
    value = result.raw_monitor("probe")["components"]["Ez"].detach().to("cpu")
    return {"shape": list(value.shape), "real": value.real.reshape(-1).tolist(),
            "imag": value.imag.reshape(-1).tolist() if value.is_complex() else [0.0] * value.numel()}


def _baseline_worker(connection, workload, device, mode, concurrency, native_graph="off"):
    """Persistent spawned worker; no CUDA context is inherited through fork."""
    prepared_batch = None
    prepared_key = None
    try:
        torch.cuda.set_device(device)
        connection.send({"kind": "ready"})
        while True:
            command = connection.recv()
            if command["kind"] == "close":
                return
            indices, identical = command["indices"], command["identical"]
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            if mode == "native_batch":
                key = (tuple(indices), identical)
                if key != prepared_key:
                    if prepared_batch is not None:
                        prepared_batch.close()
                    simulations = [build_simulation(workload, index=i, identical=identical, device=device) for i in indices]
                    prepared_batch = mw.Simulation.batch(simulations, execution=mw.BatchExecution(
                        devices=(device,), batch_size=min(concurrency, len(indices)),
                        max_batch_size=concurrency, graph=native_graph)).prepare()
                    prepared_key = key
                    del simulations
                with prepared_batch.iter_results() as iterator:
                    for item in iterator:
                        item.raise_for_error()
                        connection.send({"kind": "result", "index": indices[item.index],
                                         "probe": _probe_snapshot(item.result)})
                    del item
            elif mode == "run_many":
                simulations = [build_simulation(workload, index=i, identical=identical, device=device) for i in indices]
                sequence = mw.run_many(simulations, execution=mw.MultiGPUExecution.ensemble(
                    devices=(device,), max_concurrency=concurrency,
                    per_device_concurrency=concurrency,
                ))
                for index, result in zip(indices, sequence):
                    connection.send({"kind": "result", "index": index, "probe": _probe_snapshot(result)})
                del result, sequence, simulations
            else:
                for index in indices:
                    simulation = build_simulation(workload, index=index, identical=identical, device=device)
                    result = simulation.run()
                    connection.send({"kind": "result", "index": index, "probe": _probe_snapshot(result)})
                    del result, simulation
            torch.cuda.synchronize(device)
            connection.send({
                "kind": "done", "worker_wall_s": time.perf_counter() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "live_allocated_bytes": torch.cuda.memory_allocated(device),
            })
    except BaseException as exc:
        import traceback
        connection.send({"kind": "error", "error": f"{type(exc).__name__}: {exc}",
                         "traceback": traceback.format_exc()})
    finally:
        if prepared_batch is not None:
            prepared_batch.close()
        connection.close()


class BaselineSession:
    """One mode's workers, reused across warmups and timed repetitions."""

    def __init__(self, workload, *, device, mode, variants, concurrency, timeout_s=1800, native_graph="off"):
        import multiprocessing as mp
        from multiprocessing.connection import wait
        self.mode, self.variants = mode, variants
        self.device, self.timeout_s = device, timeout_s
        self.processes, self.connections = [], []
        self._cold_sampler = None
        self._startup_wall_s = None
        self._startup_started = time.perf_counter()
        self._cold_sampler = GpuMemorySampler(device=device)
        self._cold_sampler.start()
        context = mp.get_context("spawn")
        workers = min(variants, concurrency) if mode == "processes" else 1
        try:
            for _ in range(workers):
                parent, child = context.Pipe()
                process = context.Process(target=_baseline_worker,
                    args=(child, workload, device, mode, concurrency, native_graph))
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            remaining = set(self.connections)
            deadline = time.perf_counter() + timeout_s
            while remaining:
                ready = wait(remaining, timeout=max(0, deadline - time.perf_counter()))
                if not ready:
                    raise TimeoutError("Baseline workers did not initialize before the deadline.")
                for connection in ready:
                    message = connection.recv()
                    if message["kind"] != "ready":
                        raise RuntimeError(message)
                    remaining.remove(connection)
            self._startup_wall_s = time.perf_counter() - self._startup_started
        except BaseException:
            self.close()
            raise

    def run(self, *, identical):
        from multiprocessing.connection import wait
        sampler = self._cold_sampler
        cold = sampler is not None
        self._cold_sampler = None
        if sampler is None:
            sampler = GpuMemorySampler(device=self.device)
            sampler.start()
        started = self._startup_started if cold else time.perf_counter()
        outputs, records = {}, []
        first_result_s = None
        try:
            for rank, connection in enumerate(self.connections):
                connection.send({"kind": "run", "indices": list(range(rank, self.variants, len(self.connections))),
                                 "identical": identical})
            pending = set(self.connections)
            deadline = time.perf_counter() + self.timeout_s
            while pending:
                ready = wait(pending, timeout=max(0, deadline - time.perf_counter()))
                if not ready:
                    raise TimeoutError("Baseline round exceeded its deadline.")
                for connection in ready:
                    message = connection.recv()
                    if message["kind"] == "error":
                        raise RuntimeError(message["traceback"])
                    if message["kind"] == "done":
                        records.append(message)
                        pending.remove(connection)
                    elif message["kind"] == "result":
                        if message["index"] in outputs:
                            raise RuntimeError("A baseline worker published a duplicate result.")
                        outputs[message["index"]] = message["probe"]
                        if first_result_s is None:
                            first_result_s = time.perf_counter() - started
            elapsed = time.perf_counter() - started
        finally:
            driver_peak = sampler.stop()
        if set(outputs) != set(range(self.variants)):
            raise RuntimeError("Baseline round returned an incomplete result set.")
        return {
            "mode": self.mode, "cold": cold, "wall_time_s": elapsed,
            "first_result_s": first_result_s, "variants_per_second": self.variants / elapsed,
            "startup_wall_s": self._startup_wall_s if cold else 0.0,
            "driver_peak_increment_bytes": driver_peak,
            "driver_peak_total_bytes": sampler.peak_total_bytes,
            "driver_baseline_total_bytes": sampler.baseline_total_bytes,
            "workers": records, "probes": [outputs[i] for i in range(self.variants)],
        }

    def close(self):
        if self._cold_sampler is not None:
            self._cold_sampler.stop()
            self._cold_sampler = None
        for connection in self.connections:
            try:
                connection.send({"kind": "close"})
            except (OSError, EOFError, BrokenPipeError):
                pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for connection in self.connections:
            connection.close()
        self.connections.clear()
        self.processes.clear()


def compare_probes(reference, actual):
    """Check exported spectra against the independent sequential baseline."""
    import math
    if not reference or len(reference) != len(actual):
        raise ValueError("Reference and actual variant counts differ.")
    metrics = []
    for expected, observed in zip(reference, actual):
        if expected["shape"] != observed["shape"]:
            raise ValueError("Reference and actual probe shapes differ.")
        count = math.prod(expected["shape"])
        if count == 0 or any(len(p[part]) != count for p in (expected, observed) for part in ("real", "imag")):
            raise ValueError("Probe values must match their nonempty declared shape.")
        a = torch.complex(torch.tensor(expected["real"], dtype=torch.float64),
                          torch.tensor(expected["imag"], dtype=torch.float64))
        b = torch.complex(torch.tensor(observed["real"], dtype=torch.float64),
                          torch.tensor(observed["imag"], dtype=torch.float64))
        floor = 1e-20
        relative = float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(a).clamp_min(floor))
        maximum = float((a - b).abs().max() / a.abs().max().clamp_min(floor))
        if not torch.isfinite(a).all() or not torch.isfinite(b).all() or relative > 1e-4 or maximum > 1e-4:
            raise AssertionError(f"Baseline spectrum mismatch: rel_l2={relative}, relative_linf={maximum}.")
        metrics.append({"rel_l2": relative, "relative_linf": maximum})
    return metrics


def benchmark_baselines(workload, *, device, variants, concurrency, identical, warmups, repeats, include_native=False, native_graph="off"):
    import statistics
    if not torch.cuda.is_available():
        raise RuntimeError("Baseline comparisons require CUDA.")
    if min(variants, concurrency, repeats) < 1 or warmups < 0:
        raise ValueError("variants/concurrency/repeats must be positive and warmups nonnegative.")
    records = []
    sessions = {}
    reference = None
    modes = ("sequential", "processes", "run_many") + (("native_batch",) if include_native else ())
    try:
        # Separate mode sessions persist across interleaved rounds. Their idle
        # allocator reservations remain visible in driver-level measurements.
        for mode in modes:
            session = BaselineSession(workload, device=device, mode=mode,
                                      variants=variants, concurrency=concurrency, native_graph=native_graph)
            sessions[mode] = session
            cold = session.run(identical=identical)
            reference = cold["probes"] if reference is None else reference
            cold["correctness"] = compare_probes(reference, cold["probes"])
            records.append(cold)
        for iteration in range(warmups + repeats):
            order = modes if iteration % 2 == 0 else tuple(reversed(modes))
            for mode in order:
                record = sessions[mode].run(identical=identical)
                record["correctness"] = compare_probes(reference, record["probes"])
                record["warmup"] = iteration < warmups
                record["iteration"] = iteration
                records.append(record)
    finally:
        for session in sessions.values():
            session.close()
    isolated_memory = []
    # Peak memory comparisons get fresh, exclusive mode sessions. Keeping idle
    # sessions during interleaved timing is useful for fair warm latency, but
    # those sessions' contexts and caches must not inflate a mode's memory cost.
    for mode in modes:
        session = BaselineSession(workload, device=device, mode=mode,
                                  variants=variants, concurrency=concurrency, native_graph=native_graph)
        try:
            record = session.run(identical=identical)
            record["correctness"] = compare_probes(reference, record["probes"])
            isolated_memory.append(record)
        finally:
            session.close()
    summary = {}
    for mode in modes:
        values = [r["wall_time_s"] for r in records if r["mode"] == mode and not r["cold"] and not r["warmup"]]
        summary[mode] = {"median_wall_s": statistics.median(values), "min_wall_s": min(values),
                         "max_wall_s": max(values)}
    return {
        "schema_version": 1, "kind": "batch_independent_baselines", "workload": asdict(workload),
        "device": device, "variants": variants, "concurrency": concurrency,
        "include_native": include_native,
        "native_graph": native_graph,
        "identical": identical, "deduplication": False, "warmups": warmups, "repeats": repeats,
        "acceptance_protocol": warmups >= 2 and repeats >= 5,
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(device)},
        "output_contract": "probe complex spectra exported to CPU; no user-retained GPU results",
        "notes": ["All modes use eager stepping and identical physical inputs.",
                  "Cold timing includes worker startup, preparation, solve, export and readiness.",
                  "Independent warm timing reuses processes and extension caches, not PreparedSimulation or field state.",
                  "When included, native_batch warm timing also reuses PreparedBatch fields and the resident compiled cohort.",
                  "run_many naturally retains its complete ResultSequence until export.",
                  "Interleaved sessions keep idle CUDA contexts and allocator caches alive; total driver peaks include them.",
                  "isolated_memory uses one fresh mode session at a time, including startup and CUDA context overhead.",
                  "Per-worker allocator peaks are not summed to infer simultaneous device usage.",
                  "Device peaks are sampled at 20 ms and can miss shorter transients.",
                  "Business geometry and historical memory values have not been reproduced."],
        "summary": summary, "records": records, "isolated_memory": isolated_memory,
    }


if __name__ == "__main__":
    main()
