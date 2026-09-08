"""Fixed workload inventory and live tensor audits for FDTD batch development."""
from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.inventory:
        payload = {"schema_version": 1, "workloads": [asdict(w) for w in BATCH_WORKLOADS.values()],
                   "planned_variant_counts": [1, 10, 100], "deduplication": False,
                   "status": "baseline_instrumentation_only"}
    else:
        payload = audit(BATCH_WORKLOADS[args.audit], device=args.device)
    encoded = json.dumps(payload, indent=2) + "\n"
    if args.audit or args.output:
        path = args.output or PLOTS_DIR / "batch" / f"{args.audit}-audit.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
