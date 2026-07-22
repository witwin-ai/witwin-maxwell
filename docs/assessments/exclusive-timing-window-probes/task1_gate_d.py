"""Task 1 driver: plan-03 gate (d) re-measurement on current master.

Reuses the committed scene/sample builders in
tests/support/benchmark_network_embedding.py (the harness that produced the
tracked grid-sweep artifact) so the measurement methodology is identical; this
driver only adds N>=5 paired ABBA rounds, an A/A calibration leg, and the
per-step fixed-cost subtraction the gate statement asks for.

Not committed. Emits the machine-readable JSON artifact to --output.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "support"))

import benchmark_network_embedding as bne  # noqa: E402
from perf_variance_gate import (  # noqa: E402
    ci95_upper_bound,
    evaluate_regression_gate,
    median_absolute_deviation,
    paired_ratios,
)


# grid -> time_steps, matching the tracked grid-sweep artifact methodology
GRID_STEPS = {64: 2000, 96: 1500, 128: 1500, 176: 1500, 224: 1500}


def _clocks() -> list[str]:
    import subprocess

    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,clocks.current.sm,clocks.max.sm,"
            "clocks.current.memory,persistence_mode",
            "--format=csv,noheader",
        ],
        text=True,
    )
    return [line.strip() for line in out.strip().splitlines()]


def _foreign_apps() -> list[str]:
    import subprocess

    out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        text=True,
    ).strip()
    mypid = str(__import__("os").getpid())
    rows = [r.strip() for r in out.splitlines() if r.strip()]
    return [r for r in rows if not r.startswith(mypid)]


def _measure_connected_grid(grid: int, steps: int, rounds: int) -> dict:
    baseline_scene = bne._scene(dynamic=False, grid_cells=grid)
    connected_scene = bne._scene(dynamic=True, grid_cells=grid)
    warm = max(8, steps // 10)
    bne._sample(baseline_scene, steps=warm)
    bne._sample(connected_scene, steps=warm)

    base_samples: list[float] = []
    conn_samples: list[float] = []
    state_count = 0
    graph_state = (False, False, False)
    for r in range(rounds):
        if r % 2 == 0:
            first, second = "baseline", "connected"
        else:
            first, second = "connected", "baseline"
        for label in (first, second):
            if label == "baseline":
                ms, _, _, _ = bne._sample(baseline_scene, steps=steps)
                base_samples.append(ms)
            else:
                ms, sc, _, gs = bne._sample(connected_scene, steps=steps)
                conn_samples.append(ms)
                state_count = sc
                graph_state = gs
    if graph_state != (True, True, True):
        raise RuntimeError(
            f"grid {grid}: connected graph state {graph_state} != (True,True,True)"
        )

    base_med = float(statistics.median(base_samples))
    conn_med = float(statistics.median(conn_samples))
    per_step_ms = (conn_med - base_med) / steps
    overhead_pct = 100.0 * (conn_med / base_med - 1.0)
    # paired overhead ratio CI upper bound (round order paired)
    n = min(len(base_samples), len(conn_samples))
    ratios = paired_ratios(base_samples[:n], conn_samples[:n])
    gate = evaluate_regression_gate(ratios, target_ratio=1.10)
    return {
        "grid_cells_per_axis": grid,
        "time_steps": steps,
        "rounds": rounds,
        "baseline_samples_ms": base_samples,
        "connected_samples_ms": conn_samples,
        "baseline_median_ms": base_med,
        "connected_median_ms": conn_med,
        "baseline_mad_ms": float(median_absolute_deviation(base_samples)),
        "connected_mad_ms": float(median_absolute_deviation(conn_samples)),
        "connected_minus_baseline_ms_per_step": per_step_ms,
        "overhead_pct": overhead_pct,
        "overhead_ci95_upper_pct": gate.ci95_upper_regression_pct,
        "passed_10pct_gate_point": overhead_pct < 10.0,
        "passed_10pct_gate_ci95": bool(gate.ci95_upper_ratio < 1.10),
        "network_state_count": state_count,
        "expected_state_count": bne.PORT_COUNT * bne.MODEL_ORDER,
    }


def _aa_floor(grid: int, steps: int, rounds: int) -> dict:
    """A/A: measure the SAME baseline config twice per round -> resolution floor."""
    scene = bne._scene(dynamic=False, grid_cells=grid)
    warm = max(8, steps // 10)
    bne._sample(scene, steps=warm)
    a_samples: list[float] = []
    b_samples: list[float] = []
    for r in range(rounds):
        ms_a, _, _, _ = bne._sample(scene, steps=steps)
        ms_b, _, _, _ = bne._sample(scene, steps=steps)
        if r % 2 == 0:
            a_samples.append(ms_a)
            b_samples.append(ms_b)
        else:
            a_samples.append(ms_b)
            b_samples.append(ms_a)
    ratios = paired_ratios(a_samples, b_samples)
    upper = ci95_upper_bound(ratios)
    return {
        "grid_cells_per_axis": grid,
        "time_steps": steps,
        "rounds": rounds,
        "a_samples_ms": a_samples,
        "b_samples_ms": b_samples,
        "aa_ratios": ratios,
        "aa_mean_pct": 100.0 * (statistics.fmean(ratios) - 1.0),
        "aa_ci95_upper_pct": 100.0 * (upper - 1.0),
        "aa_abs_max_delta_pct": 100.0 * max(abs(x - 1.0) for x in ratios),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--aa-grid", type=int, default=224)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    clocks_before = _clocks()
    foreign_before = _foreign_apps()

    sweep = []
    for grid, steps in GRID_STEPS.items():
        sweep.append(_measure_connected_grid(grid, steps, args.rounds))

    aa = _aa_floor(args.aa_grid, GRID_STEPS[args.aa_grid], args.rounds)

    clocks_after = _clocks()
    foreign_after = _foreign_apps()

    # crossover: smallest grid whose point overhead < 10%
    crossover = None
    for row in sweep:
        if row["overhead_pct"] < 10.0:
            crossover = row["grid_cells_per_axis"]
            break
    fixed_costs = [row["connected_minus_baseline_ms_per_step"] for row in sweep]

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    payload = {
        "schema_version": 1,
        "kind": "network_embedding_gate_d_remeasure",
        "title": "plan 03 gate (d) re-measurement after round-E composite matvec coupling",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": __import__("subprocess")
        .check_output(["git", "rev-parse", "HEAD"], text=True)
        .strip(),
        "gate": "plan 03 Phase 4 gate (d): connected 8-port/order-32 step overhead < 10% at crossover grid",
        "method": (
            "Reuses tests/support/benchmark_network_embedding.py scene/sample builders "
            "(cuda_graph field+network+port_observer, CUDA-event timed). N>=5 paired "
            "ABBA rounds per grid; per-step fixed cost = (connected_med - baseline_med)/steps; "
            "overhead CI = one-sided 95% upper bound on paired connected/baseline ratio."
        ),
        "harness": "tests/support/benchmark_network_embedding.py",
        "driver": "scratch/task1_gate_d.py (not committed)",
        "command": (
            "numactl --cpunodebind=0 --membind=0 conda run -n maxwell python "
            "scratch/task1_gate_d.py --rounds 6 --aa-grid 224 --output "
            "docs/assessments/network-embedding-gate-d-remeasure-2026-07-20.json"
        ),
        "ports": bne.PORT_COUNT,
        "model_order": bne.MODEL_ORDER,
        "reference_before_composite": {
            "op_stream_artifact": "docs/assessments/e4-network-coupling-op-stream-2026-07-19.json",
            "launches_per_step_before": 78.0,
            "launches_per_step_after": 27.0,
            "prior_grid_sweep_artifact": "docs/assessments/network-embedding-phase-4-performance-grid-sweep.json",
            "prior_fixed_cost_ms_per_step_range": [0.193, 0.204],
            "prior_crossover_grid_approx": 224,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device_name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "numactl": "--cpunodebind=0 --membind=0",
            "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_governor": "performance",
        },
        "exclusive_window": {
            "foreign_compute_apps_before": foreign_before,
            "foreign_compute_apps_after": foreign_after,
            "clocks_before": clocks_before,
            "clocks_after": clocks_after,
            "verified": not foreign_before and not foreign_after,
        },
        "aa_calibration": aa,
        "sweep": sweep,
        "summary": {
            "new_fixed_cost_ms_per_step_median": float(statistics.median(fixed_costs)),
            "new_fixed_cost_ms_per_step_min": float(min(fixed_costs)),
            "new_fixed_cost_ms_per_step_max": float(max(fixed_costs)),
            "new_crossover_grid_point_overhead_lt_10pct": crossover,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "crossover": crossover,
        "fixed_cost_median_ms": payload["summary"]["new_fixed_cost_ms_per_step_median"],
        "aa_ci95_upper_pct": aa["aa_ci95_upper_pct"],
        "overhead_by_grid": {row["grid_cells_per_axis"]: round(row["overhead_pct"], 3) for row in sweep},
    }, indent=2))


if __name__ == "__main__":
    main()
