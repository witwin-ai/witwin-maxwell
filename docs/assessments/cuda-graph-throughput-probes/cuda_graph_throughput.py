"""CUDA-graph vs eager FDTD stepping throughput on a plain vacuum-dipole scene.

Produces the tracked artifact
``docs/assessments/cuda-graph-throughput-2026-07-22.json``, which is the
reproducible basis for the release-notes claim about the ``cuda_graph=True``
public default.

Method
------
The public knob is ``Simulation.fdtd(..., cuda_graph=...)``; it is forwarded
verbatim to ``solver.solve(use_cuda_graph=...)``. Each sample therefore builds a
real ``Simulation`` with the public flag and reads the flag back off the
resolved config, so no private toggle is being measured.

Steady-state ms/step is obtained with a **two-point slope estimator**:

    ms_per_step = (T(2S) - T(S)) / S

Graph capture, allocator first-touch and every other one-shot cost enters both
``T(S)`` and ``T(2S)`` identically and cancels in the difference, so the reported
number is warmup-free by construction rather than by "long enough run" hand-waving.
The residual intercept ``T(S) - S * ms_per_step`` is also recorded, because for the
graph mode it *is* the capture cost and is worth seeing.

Discipline
----------
* every process pinned with ``numactl --cpunodebind=0 --membind=0``;
* both GPUs verified free of foreign compute apps before and after;
* SM/memory clocks recorded before and after;
* ``--rounds`` >= 5 paired ABBA rounds per grid;
* an **A/A calibration** (the same configuration measured against itself, once
  for graph and once for eager) run at every grid, whose CI95 width is the
  resolution floor: an A/B delta inside that floor is *not* a win;
* peak memory measured in a separate pass so allocator bookkeeping never sits
  inside a timed region.

Reproduce::

    export CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13
    export PATH="$CUDA_HOME/bin:$PATH"
    CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 \
        conda run -n maxwell --no-capture-output python \
        docs/assessments/cuda-graph-throughput-probes/cuda_graph_throughput.py \
        --rounds 7 --output docs/assessments/cuda-graph-throughput-2026-07-22.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "support"))

import witwin.maxwell as mw  # noqa: E402
from perf_variance_gate import (  # noqa: E402
    median_absolute_deviation,
    paired_ratios,
    student_t_95_one_sided,
)

FREQUENCY_HZ = 1.0e9
DOMAIN_SIZE = 0.6

# grid cells per axis -> base step count S (the run is S then 2S per sample)
GRID_STEPS: dict[int, int] = {48: 2500, 64: 2000, 96: 2000, 128: 1500, 160: 1000, 288: 400}


def _clocks() -> list[str]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,clocks.current.sm,clocks.max.sm,"
            "clocks.current.memory,persistence_mode,temperature.gpu,power.draw",
            "--format=csv,noheader",
        ],
        text=True,
    )
    return [line.strip() for line in out.strip().splitlines()]


def _foreign_apps() -> list[str]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        text=True,
    ).strip()
    mypid = str(os.getpid())
    rows = [row.strip() for row in out.splitlines() if row.strip()]
    return [row for row in rows if not row.startswith(mypid)]


def _scene(grid_cells: int) -> mw.Scene:
    """Plain vacuum FDTD scene: CPML box, one Gaussian point dipole, one probe.

    Deliberately feature-free -- no dispersion, no ports, no networks, no wires --
    so the measurement isolates the field-update stepping core that the CUDA graph
    captures.
    """

    cell = DOMAIN_SIZE / grid_cells
    half = 0.5 * DOMAIN_SIZE
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-half, half), (-half, half), (-half, half))),
        grid=mw.GridSpec.uniform(cell),
        boundary=mw.BoundarySpec.pml(num_layers=8, strength=1.0),
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=1.5 * cell,
            source_time=mw.GaussianPulse(
                frequency=FREQUENCY_HZ, fwidth=0.5 * FREQUENCY_HZ, amplitude=1.0
            ),
        )
    )
    scene.add_monitor(
        mw.PointMonitor(name="probe", position=(0.1, 0.0, 0.0), fields=("Ez",))
    )
    return scene


def _simulation(scene: mw.Scene, *, steps: int, cuda_graph: bool):
    return mw.Simulation.fdtd(
        scene,
        frequency=FREQUENCY_HZ,
        run_time=mw.TimeConfig(time_steps=steps),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
        cuda_graph=cuda_graph,
    )


def _timed_solve(scene: mw.Scene, *, steps: int, cuda_graph: bool) -> tuple[float, bool]:
    """One CUDA-event-timed ``solve`` through the public flag. Returns (ms, graph_active)."""

    simulation = _simulation(scene, steps=steps, cuda_graph=cuda_graph)
    flag = bool(simulation.config.cuda_graph)
    if flag is not cuda_graph:
        raise RuntimeError(f"public cuda_graph flag not honoured: {flag} != {cuda_graph}")
    solver = simulation.prepare().solver
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    solver.solve(
        time_steps=steps,
        dft_frequency=FREQUENCY_HZ,
        dft_window="none",
        full_field_dft=False,
        use_cuda_graph=flag,
    )
    end.record()
    end.synchronize()
    elapsed_ms = float(start.elapsed_time(end))
    active = bool(getattr(solver, "_cuda_graph_active", False))
    del solver, simulation
    return elapsed_ms, active


def _slope_sample(scene: mw.Scene, *, base_steps: int, cuda_graph: bool) -> dict[str, Any]:
    """Two-point slope: ms/step with every one-shot cost differenced out."""

    low_ms, active_low = _timed_solve(scene, steps=base_steps, cuda_graph=cuda_graph)
    high_ms, active_high = _timed_solve(scene, steps=2 * base_steps, cuda_graph=cuda_graph)
    if active_low != active_high:
        raise RuntimeError("graph activation differed between the two slope points")
    ms_per_step = (high_ms - low_ms) / base_steps
    return {
        "low_ms": low_ms,
        "high_ms": high_ms,
        "ms_per_step": ms_per_step,
        "intercept_ms": low_ms - base_steps * ms_per_step,
        "graph_active": active_low,
    }


def _one_sided_bounds(values: list[float]) -> tuple[float, float]:
    """(lower, upper) one-sided 95% bounds on the mean."""

    n = len(values)
    mean = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(n)
    t = student_t_95_one_sided(n - 1)
    return mean - t * standard_error, mean + t * standard_error


def _compare(
    scene: mw.Scene,
    *,
    base_steps: int,
    rounds: int,
    mode_a: bool,
    mode_b: bool,
    label: str,
) -> dict[str, Any]:
    """Paired ABBA comparison of two cuda_graph settings.

    ``mode_a == mode_b`` makes this an A/A calibration leg; the resulting CI is
    the resolution floor for this grid.
    """

    a_samples: list[dict[str, Any]] = []
    b_samples: list[dict[str, Any]] = []
    for index in range(rounds):
        forward = index % 2 == 0
        first, second = (mode_a, mode_b) if forward else (mode_b, mode_a)
        first_sample = _slope_sample(scene, base_steps=base_steps, cuda_graph=first)
        second_sample = _slope_sample(scene, base_steps=base_steps, cuda_graph=second)
        if forward:
            a_samples.append(first_sample)
            b_samples.append(second_sample)
        else:
            a_samples.append(second_sample)
            b_samples.append(first_sample)

    a_ms = [sample["ms_per_step"] for sample in a_samples]
    b_ms = [sample["ms_per_step"] for sample in b_samples]
    # ratio > 1 means B is slower than A.
    ratios = paired_ratios(a_ms, b_ms)
    lower, upper = _one_sided_bounds(ratios)
    return {
        "label": label,
        "mode_a_cuda_graph": mode_a,
        "mode_b_cuda_graph": mode_b,
        "rounds": rounds,
        "base_steps": base_steps,
        "a_ms_per_step": a_ms,
        "b_ms_per_step": b_ms,
        "a_raw": a_samples,
        "b_raw": b_samples,
        "a_median_ms_per_step": float(statistics.median(a_ms)),
        "b_median_ms_per_step": float(statistics.median(b_ms)),
        "a_mad_ms_per_step": float(median_absolute_deviation(a_ms)),
        "b_mad_ms_per_step": float(median_absolute_deviation(b_ms)),
        "a_median_intercept_ms": float(
            statistics.median([sample["intercept_ms"] for sample in a_samples])
        ),
        "b_median_intercept_ms": float(
            statistics.median([sample["intercept_ms"] for sample in b_samples])
        ),
        "paired_ratios_b_over_a": ratios,
        "ratio_mean": float(statistics.fmean(ratios)),
        "ratio_ci95_lower": float(lower),
        "ratio_ci95_upper": float(upper),
        "b_slower_than_a_pct": 100.0 * (float(statistics.fmean(ratios)) - 1.0),
        "b_slower_than_a_pct_ci95": [100.0 * (lower - 1.0), 100.0 * (upper - 1.0)],
        "abs_max_delta_pct": 100.0 * max(abs(ratio - 1.0) for ratio in ratios),
        "graph_active_a": bool(a_samples[0]["graph_active"]),
        "graph_active_b": bool(b_samples[0]["graph_active"]),
    }


def _peak_memory(scene: mw.Scene, *, steps: int, cuda_graph: bool) -> int:
    """Peak allocated bytes for one full solve, measured outside any timed region."""

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    simulation = _simulation(scene, steps=steps, cuda_graph=cuda_graph)
    solver = simulation.prepare().solver
    solver.solve(
        time_steps=steps,
        dft_frequency=FREQUENCY_HZ,
        dft_window="none",
        full_field_dft=False,
        use_cuda_graph=bool(simulation.config.cuda_graph),
    )
    torch.cuda.synchronize()
    peak = int(torch.cuda.max_memory_allocated())
    del solver, simulation
    gc.collect()
    torch.cuda.empty_cache()
    return peak


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--memory-steps", type=int, default=200)
    parser.add_argument(
        "--grids", type=int, nargs="*", default=sorted(GRID_STEPS), help="cells per axis"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    if args.rounds < 5:
        raise ValueError("the timing discipline requires at least 5 paired rounds.")

    clocks_before = _clocks()
    foreign_before = _foreign_apps()

    grids: list[dict[str, Any]] = []
    for grid in args.grids:
        base_steps = GRID_STEPS[grid]
        scene = _scene(grid)

        # Process/clock warmup: one full slope sample per mode, discarded.
        _slope_sample(scene, base_steps=base_steps, cuda_graph=True)
        _slope_sample(scene, base_steps=base_steps, cuda_graph=False)

        ab = _compare(
            scene,
            base_steps=base_steps,
            rounds=args.rounds,
            mode_a=True,
            mode_b=False,
            label="A/B graph(A) vs eager(B)",
        )
        aa_graph = _compare(
            scene,
            base_steps=base_steps,
            rounds=args.rounds,
            mode_a=True,
            mode_b=True,
            label="A/A graph vs graph",
        )
        aa_eager = _compare(
            scene,
            base_steps=base_steps,
            rounds=args.rounds,
            mode_a=False,
            mode_b=False,
            label="A/A eager vs eager",
        )

        if not ab["graph_active_a"]:
            raise RuntimeError(f"grid {grid}: cuda_graph=True did not activate a graph")
        if ab["graph_active_b"]:
            raise RuntimeError(f"grid {grid}: cuda_graph=False still activated a graph")

        graph_ms = ab["a_median_ms_per_step"]
        eager_ms = ab["b_median_ms_per_step"]
        # Throughput change of the graph default relative to eager:
        # ratios are eager/graph, so mean ratio - 1 is the graph throughput gain.
        gain_pct = ab["b_slower_than_a_pct"]
        gain_ci = ab["b_slower_than_a_pct_ci95"]
        floor_pct = max(aa_graph["abs_max_delta_pct"], aa_eager["abs_max_delta_pct"])
        floor_ci_pct = max(
            abs(aa_graph["b_slower_than_a_pct_ci95"][0]),
            abs(aa_graph["b_slower_than_a_pct_ci95"][1]),
            abs(aa_eager["b_slower_than_a_pct_ci95"][0]),
            abs(aa_eager["b_slower_than_a_pct_ci95"][1]),
        )
        resolvable = abs(gain_pct) > floor_ci_pct and (gain_ci[0] > 0.0 or gain_ci[1] < 0.0)

        peak_graph = _peak_memory(scene, steps=args.memory_steps, cuda_graph=True)
        peak_eager = _peak_memory(scene, steps=args.memory_steps, cuda_graph=False)

        grids.append(
            {
                "grid_cells_per_axis": grid,
                "total_cells": grid**3,
                "base_steps": base_steps,
                "ab": ab,
                "aa_graph": aa_graph,
                "aa_eager": aa_eager,
                "graph_median_ms_per_step": graph_ms,
                "eager_median_ms_per_step": eager_ms,
                "graph_throughput_gain_pct": gain_pct,
                "graph_throughput_gain_pct_ci95": gain_ci,
                "aa_resolution_floor_abs_max_pct": floor_pct,
                "aa_resolution_floor_ci95_pct": floor_ci_pct,
                "resolvable": bool(resolvable),
                "verdict": (
                    ("graph faster" if gain_pct > 0 else "graph slower")
                    if resolvable
                    else "no resolvable difference"
                ),
                "graph_capture_intercept_ms": ab["a_median_intercept_ms"],
                "eager_intercept_ms": ab["b_median_intercept_ms"],
                "peak_memory_bytes_graph": peak_graph,
                "peak_memory_bytes_eager": peak_eager,
                "peak_memory_steps": args.memory_steps,
                "peak_memory_graph_over_eager_pct": 100.0 * (peak_graph / peak_eager - 1.0),
                # Sampled right after this grid's timed legs, i.e. under load --
                # the before/after window samples are taken at process idle.
                "clocks_after_grid": _clocks(),
            }
        )
        del scene
        gc.collect()
        torch.cuda.empty_cache()

    clocks_after = _clocks()
    foreign_after = _foreign_apps()
    props = torch.cuda.get_device_properties(torch.cuda.current_device())

    payload = {
        "schema_version": 1,
        "kind": "cuda_graph_throughput",
        "title": "CUDA-graph vs eager FDTD stepping throughput (0.4.0 release-notes basis)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=str(ROOT)
        ).strip(),
        "git_dirty_paths": [
            line.strip()
            for line in subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, cwd=str(ROOT)
            ).splitlines()
            if line.strip()
        ],
        "claim_under_test": (
            "release-notes-0.4.0 section 3.2, inherited unmeasured from commit 9650439: "
            "'+29% throughput at 96^3, neutral at 288^3, +8-14% peak memory'"
        ),
        "method": (
            "Public knob Simulation.fdtd(cuda_graph=...) forwarded to solve(use_cuda_graph=...). "
            "Steady-state ms/step from a two-point slope (T(2S)-T(S))/S so graph capture and "
            "every other one-shot cost cancels. CUDA-event timing with explicit synchronize. "
            "N paired ABBA rounds per grid; A/A calibration legs (graph-vs-graph and "
            "eager-vs-eager) at every grid give the resolution floor. Peak memory measured in a "
            "separate untimed pass with reset_peak_memory_stats + gc + empty_cache."
        ),
        "scene": (
            "vacuum cube 0.6 m, uniform cell = 0.6/N, CPML 8 layers, one Ez Gaussian point "
            "dipole at 1 GHz, one point monitor; no dispersion/ports/networks/wires"
        ),
        "driver": "docs/assessments/cuda-graph-throughput-probes/cuda_graph_throughput.py",
        "command": (
            "CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 conda run -n maxwell "
            "--no-capture-output python "
            "docs/assessments/cuda-graph-throughput-probes/cuda_graph_throughput.py "
            f"--rounds {args.rounds} --output {args.output}"
        ),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "driver": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
            )
            .strip()
            .splitlines()[0],
            "device_name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "device_count": torch.cuda.device_count(),
            "numactl": "--cpunodebind=0 --membind=0",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "exclusive_window": {
            "foreign_compute_apps_before": foreign_before,
            "foreign_compute_apps_after": foreign_after,
            "clocks_before": clocks_before,
            "clocks_after": clocks_after,
            "verified": not foreign_before and not foreign_after,
        },
        "rounds": args.rounds,
        "grids": grids,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"\nwrote {args.output}\n")
    header = f"{'grid':>6} {'graph ms/step':>14} {'eager ms/step':>14} {'gain %':>9} {'CI95':>18} {'A/A floor %':>12} {'verdict':>26} {'mem %':>8}"
    print(header)
    for row in grids:
        ci = row["graph_throughput_gain_pct_ci95"]
        print(
            f"{row['grid_cells_per_axis']:>6} "
            f"{row['graph_median_ms_per_step']:>14.5f} "
            f"{row['eager_median_ms_per_step']:>14.5f} "
            f"{row['graph_throughput_gain_pct']:>9.2f} "
            f"[{ci[0]:>7.2f},{ci[1]:>7.2f}] "
            f"{row['aa_resolution_floor_ci95_pct']:>12.3f} "
            f"{row['verdict']:>26} "
            f"{row['peak_memory_graph_over_eager_pct']:>8.2f}"
        )


if __name__ == "__main__":
    main()
