"""Interleave equal-physics spectral observer timings and verify every monitor tensor."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
import numpy as np
import torch
import witwin.maxwell as mw
from benchmark.scenes.antenna.printed_patch import printed_patch_scene
from witwin.maxwell.fdtd import observers
from scripts.dev.fdtd.pcb_acceptance import package_identity


@contextmanager
def observation_mode(grouped):
    factory = observers.make_single_observer_engine
    if not grouped:
        observers.make_single_observer_engine = lambda solver, enabled: factory(solver, enabled=False)
    try:
        yield
    finally:
        observers.make_single_observer_engine = factory


def tensor_digests(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            result.update(tensor_digests(child, prefix + "/" + str(key)))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            result.update(tensor_digests(child, prefix + "/" + str(index)))
    elif isinstance(value, (torch.Tensor, np.ndarray)):
        array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
        if array.dtype.hasobject:
            raise TypeError("Object monitor arrays cannot be compared numerically.")
        result[prefix] = {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}
    return result


def processes():
    proc = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"],
                          capture_output=True, text=True)
    return {"returncode": proc.returncode, "output": proc.stdout.strip(), "stderr": proc.stderr.strip()}


def run(grouped, duration):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    frequencies = tuple(float(f) for f in np.linspace(2.9e9, 3.5e9, 31))
    scene = printed_patch_scene(frequencies=frequencies, radiation_monitor=True)
    simulation = mw.Simulation.fdtd(scene, frequencies=frequencies,
        excitations=mw.PortExcitation("feed", source_time=mw.GaussianPulse(frequency=3.2e9, fwidth=2.2e9)),
        run_time=mw.TimeConfig(time_steps=1), spectral_sampler=mw.SpectralSampler(window="none"), full_field_dft=False)
    start = time.perf_counter()
    prepared = simulation.prepare()
    torch.cuda.synchronize()
    prepare_seconds = time.perf_counter() - start
    solver = prepared.solver
    steps = math.ceil(duration / solver.dt)
    simulation.config.run_time = mw.TimeConfig(time_steps=steps)
    before = processes()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with observation_mode(grouped):
        result = prepared.run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated()
    after = processes()
    port = result.port("feed")
    digests = tensor_digests({"monitors": result.monitors,
                             "voltage": port.voltage, "current": port.current,
                             "reflection": port.reflection_coefficient})
    return {"grouped": grouped, "physical_duration_s": steps * solver.dt, "requested_duration_s": duration,
            "steps": steps, "dt_s": solver.dt, "grid_shape": [solver.Nx, solver.Ny, solver.Nz],
            "prepare_seconds": prepare_seconds, "run_seconds": elapsed,
            "steps_per_second": steps / elapsed, "peak_allocated_bytes": peak,
            "observer_cuda_graph_active": result.solver_stats["observer_cuda_graph_active"],
            "digests": digests, "processes_before": before, "processes_after": after}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=16e-9)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 5 or args.duration <= 0:
        parser.error("Performance evidence needs at least five repeats and positive duration.")
    identity = package_identity(mw, "witwin-maxwell")
    report = {"schema": "pcb-observer-performance/1", "source": identity,
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "scope": "Preparation separately; run includes graph capture, marching, spectral export and synchronized GPU readiness; excludes angular NF2FF and file persistence.",
              "warmup": [], "samples": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for warmup, repeat_count in ((True, 1), (False, args.repeats)):
        for repeat in range(repeat_count):
            for grouped in ((False, True) if repeat % 2 == 0 else (True, False)):
                row = run(grouped, args.duration)
                report["warmup" if warmup else "samples"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({k: row[k] for k in ("grouped", "steps", "run_seconds", "peak_allocated_bytes")}), flush=True)
    samples = report["samples"]
    reference = samples[0]["digests"]
    report["all_outputs_bitwise_equal"] = all(row["digests"] == reference for row in samples + report["warmup"])
    timings = {str(mode): [row["run_seconds"] for row in samples if row["grouped"] == mode] for mode in (False, True)}
    report["timing_seconds"] = {mode: {"median": statistics.median(values), "min": min(values), "max": max(values)} for mode, values in timings.items()}
    report["median_speedup"] = statistics.median(timings["False"]) / statistics.median(timings["True"])
    report["source_unchanged"] = package_identity(mw, "witwin-maxwell")["source_sha256"] == identity["source_sha256"]
    report["status"] = "measured; process samples require exclusivity review before performance acceptance"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["all_outputs_bitwise_equal"] or not report["source_unchanged"]:
        raise SystemExit("Output or source identity changed during the timing experiment.")


if __name__ == "__main__":
    main()
