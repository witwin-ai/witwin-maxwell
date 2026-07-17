"""Ensemble wall-clock speedup: N independent mid-size FDTD sims, serial vs 2 GPUs.

Builds ``--n`` independent declarative Scenes (one Simulation each) and runs them
through the public ensemble executor twice per repeat:

* serial   -- one device, ``max_concurrency=1`` (tasks run back-to-back);
* parallel -- all devices, ``max_concurrency=len(devices)`` (tasks run concurrently).

The reported figure is the serial/parallel total wall-clock ratio (median over
``--repeats`` with MAD). Interpreting it as a speedup requires an exclusive-GPU
window; this harness only measures it. Sims use eager mode (cuda_graph=False)
because CUDA graphs cannot be captured concurrently across ensemble threads. The executor's own per-task hooks
(``ExecutionRecord.wall_time_s`` / ``device_time_s``) are captured alongside.
For GPU-bound tasks on two GPUs the ratio is expected to approach ~2x.

Example::

    python tests/support/benchmark_ensemble_speedup.py \
        --n 8 --grid-cells 64 --steps 2000 --repeats 5 \
        --devices cuda:0 cuda:1 \
        --output docs/assessments/ensemble-speedup-2026-07-17.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path.cwd().resolve()))

import witwin.maxwell as mw  # noqa: E402


FREQUENCY_HZ = 1.0e9


def _mad(values: list[float]) -> float:
    median = statistics.median(values)
    return float(statistics.median(abs(value - median) for value in values))


def _build_simulations(*, n: int, grid_cells: int, steps: int) -> list[Any]:
    domain_size = 0.6
    cell = domain_size / grid_cells
    simulations: list[Any] = []
    for index in range(n):
        scene = mw.Scene(
            domain=mw.Domain(
                bounds=(
                    (-0.5 * domain_size, 0.5 * domain_size),
                    (-0.5 * domain_size, 0.5 * domain_size),
                    (-0.5 * domain_size, 0.5 * domain_size),
                )
            ),
            grid=mw.GridSpec.uniform(cell),
            boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
            device="cuda",
        )
        # A tiny per-index amplitude perturbation keeps the Scenes independent
        # objects with comparable cost, satisfying the ensemble's distinct-Scene
        # contract without changing the launch-count profile.
        scene.add_source(
            mw.PointDipole(
                position=(0.0, 0.0, 0.0),
                polarization="Ez",
                width=1.5 * cell,
                source_time=mw.GaussianPulse(
                    frequency=FREQUENCY_HZ,
                    fwidth=0.5 * FREQUENCY_HZ,
                    amplitude=1.0 + 1.0e-3 * index,
                ),
            )
        )
        scene.add_monitor(
            mw.PointMonitor(name="probe", position=(0.1, 0.0, 0.0), fields=("Ez",))
        )
        simulations.append(
            mw.Simulation.fdtd(
                scene,
                frequency=FREQUENCY_HZ,
                run_time=mw.TimeConfig(time_steps=steps),
                spectral_sampler=mw.SpectralSampler(window="none"),
                full_field_dft=False,
                cuda_graph=False,
            )
        )
    return simulations


def _time_run_many(
    simulations: list[Any], *, devices: tuple[str, ...], max_concurrency: int
) -> dict[str, Any]:
    execution = mw.MultiGPUExecution.ensemble(
        devices=list(devices), max_concurrency=max_concurrency
    )
    for device in devices:
        torch.cuda.synchronize(torch.device(device))
    start = time.perf_counter()
    sequence = mw.run_many(simulations, execution=execution)
    for device in devices:
        torch.cuda.synchronize(torch.device(device))
    wall_s = time.perf_counter() - start
    failures = [entry for entry in sequence.entries if isinstance(entry, mw.DistributedFailure)]
    if failures:
        raise RuntimeError(
            f"ensemble run reported {len(failures)} failure(s): "
            f"{[str(failure) for failure in failures[:3]]}"
        )
    records = sequence.records
    device_times = [record.device_time_s for record in records]
    task_wall_times = [record.wall_time_s for record in records]
    placements: dict[str, int] = {}
    for record in records:
        placements[str(record.device)] = placements.get(str(record.device), 0) + 1
    return {
        "wall_s": wall_s,
        "completed": sum(1 for record in records if record.completed),
        "device_task_counts": placements,
        "per_task_device_time_s": device_times,
        "per_task_wall_time_s": task_wall_times,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--grid-cells", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("The ensemble speedup benchmark requires CUDA.")
    if len(args.devices) < 2:
        raise ValueError("Provide at least two devices for a serial-vs-parallel ratio.")
    if args.repeats < 2 or args.n < 2:
        raise ValueError("repeats and n must both be >= 2.")

    devices = tuple(args.devices)
    serial_device = (devices[0],)

    simulations = _build_simulations(n=args.n, grid_cells=args.grid_cells, steps=args.steps)

    # Untimed warmup for both configurations so extension/allocator/graph builds
    # do not pollute the timed repeats.
    _time_run_many(simulations, devices=serial_device, max_concurrency=1)
    _time_run_many(simulations, devices=devices, max_concurrency=len(devices))

    serial_walls: list[float] = []
    parallel_walls: list[float] = []
    ratios: list[float] = []
    last_serial: dict[str, Any] = {}
    last_parallel: dict[str, Any] = {}
    for _ in range(args.repeats):
        serial = _time_run_many(simulations, devices=serial_device, max_concurrency=1)
        parallel = _time_run_many(
            simulations, devices=devices, max_concurrency=len(devices)
        )
        serial_walls.append(serial["wall_s"])
        parallel_walls.append(parallel["wall_s"])
        ratios.append(serial["wall_s"] / parallel["wall_s"])
        last_serial, last_parallel = serial, parallel

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    payload = {
        "schema_version": 1,
        "kind": "ensemble_speedup",
        "title": "Ensemble wall-clock speedup (serial vs multi-GPU) over N independent sims",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "simulations": args.n,
            "grid_cells_per_axis": args.grid_cells,
            "time_steps": args.steps,
            "repeats": args.repeats,
            "devices": list(devices),
            "serial_device": list(serial_device),
            "cuda_graph": False,
            "full_field_dft": False,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device_name": properties.name,
            "device_count": torch.cuda.device_count(),
            "compute_capability": f"{properties.major}.{properties.minor}",
        },
        "serial": {
            "wall_s_samples": serial_walls,
            "median_wall_s": float(statistics.median(serial_walls)),
            "mad_wall_s": _mad(serial_walls),
            "device_task_counts": last_serial["device_task_counts"],
        },
        "parallel": {
            "wall_s_samples": parallel_walls,
            "median_wall_s": float(statistics.median(parallel_walls)),
            "mad_wall_s": _mad(parallel_walls),
            "device_task_counts": last_parallel["device_task_counts"],
            "per_task_device_time_s": last_parallel["per_task_device_time_s"],
        },
        "speedup_ratio": {
            "samples": ratios,
            "median": float(statistics.median(ratios)),
            "mad": _mad(ratios),
        },
        "ideal_ratio": float(len(devices)),
    }
    encoded = json.dumps(payload, sort_keys=True, indent=2)
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "speedup_ratio_median": payload["speedup_ratio"]["median"],
                "speedup_ratio_mad": payload["speedup_ratio"]["mad"],
                "serial_median_wall_s": payload["serial"]["median_wall_s"],
                "parallel_median_wall_s": payload["parallel"]["median_wall_s"],
                "ideal_ratio": payload["ideal_ratio"],
            }
        )
    )


if __name__ == "__main__":
    main()
