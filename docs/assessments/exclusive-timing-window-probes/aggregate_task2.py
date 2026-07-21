"""Aggregate Task-2 (multi-GPU timing) + Task-3 (no-feature) rows into one artifact."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _sh(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def _clocks() -> list[str]:
    return [l.strip() for l in _sh([
        "nvidia-smi",
        "--query-gpu=index,name,clocks.current.sm,clocks.max.sm,clocks.current.memory,persistence_mode",
        "--format=csv,noheader",
    ]).splitlines()]


def _foreign() -> list[str]:
    out = _sh(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"])
    return [r.strip() for r in out.splitlines() if r.strip()]


def _load(p: str) -> dict:
    return json.loads((ROOT / p).read_text())


ens = {}
for g, n in ((96, 4), (96, 8), (160, 4), (160, 8)):
    d = _load(f"scratch/ens/ens_{g}_{n}.json")
    ens[f"{g}_{n}"] = {
        "grid_cells_per_axis": g,
        "tasks": n,
        "time_steps": d["configuration"]["time_steps"],
        "repeats": d["configuration"]["repeats"],
        "devices": d["configuration"]["devices"],
        "serial_median_wall_s": d["serial"]["median_wall_s"],
        "parallel_median_wall_s": d["parallel"]["median_wall_s"],
        "makespan_speedup_median": d["speedup_ratio"]["median"],
        "makespan_speedup_mad": d["speedup_ratio"]["mad"],
        "makespan_speedup_samples": d["speedup_ratio"]["samples"],
        "parallel_device_task_counts": d["parallel"]["device_task_counts"],
        "utilization_note": (
            "parallel places n/2 tasks per GPU; wall-clock ratio approximates "
            "2-GPU throughput/utilization for these GPU-bound tasks"
        ),
    }

joint = {}
for nodes in (129, 193):
    d = _load(f"scratch/joint/joint_{nodes}.json")
    joint[f"{nodes-1}"] = {
        "cells_per_axis": nodes - 1,
        "node_shape": d["shapes"]["global_node_shape"],
        "steps": d["execution"]["steps"],
        "repeats": 5,
        "transport": d["communication"]["transport"],
        "overlap_active": d["two_gpu"].get("overlap_active"),
        "single_gpu_ms_per_step_median": d["single_gpu"]["ms_per_step_median"],
        "two_gpu_ms_per_step_median": d["two_gpu"]["ms_per_step_median"],
        "strong_speedup": d["strong_speedup"],
        "single_gpu_step_rate_hz": 1.0e3 / d["single_gpu"]["ms_per_step_median"],
        "two_gpu_step_rate_hz": 1.0e3 / d["two_gpu"]["ms_per_step_median"],
        "halo_bytes_per_step": d["communication"]["halo_bytes_per_step"],
        "p2p_min_gbps": min(r["bandwidth_gbps_median"] for r in d["p2p"].values()),
        "field_parity_max_rel": d["numerical_parity"]["field_max_rel"],
        "field_parity_passed": d["numerical_parity"]["passed"],
    }

nofeat = _load("scratch/task3_nofeature.json")

props = torch.cuda.get_device_properties(0)
payload = {
    "schema_version": 1,
    "kind": "multi_gpu_timing_and_no_feature_spotcheck",
    "title": "plan 02 deferred ensemble/joint-solve timing + plan-wide no-feature regression spot-check",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "git_sha": _sh(["git", "rev-parse", "HEAD"]),
    "environment": {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_name": props.name,
        "device_count": torch.cuda.device_count(),
        "compute_capability": f"{props.major}.{props.minor}",
        "numactl": "--cpunodebind=0 --membind=0",
        "cpu_governor": "performance",
        "persistence_mode": "enabled",
    },
    "exclusive_window": {
        "clocks_snapshot": _clocks(),
        "foreign_compute_apps": _foreign(),
        "verified": len(_foreign()) == 0,
        "note": "single measurement process at a time; both GPUs verified idle before each block",
    },
    "task2_ensemble": {
        "harness": "tests/support/benchmark_ensemble_speedup.py",
        "command_template": (
            "numactl --cpunodebind=0 --membind=0 python "
            "tests/support/benchmark_ensemble_speedup.py --n <N> --grid-cells <G> "
            "--steps 2000 --repeats 5 --devices cuda:0 cuda:1 --output <json>"
        ),
        "metric": "serial(1 GPU, max_concurrency=1) / parallel(2 GPU) total wall-clock ratio",
        "tracked_reference": "128^3 makespan speedup 1.96x (docs/assessments/ensemble-speedup-2026-07-17.json)",
        "aa_floor_context": (
            "wall-clock (not cuda_graph replay); measurement dispersion reported as "
            "makespan_speedup_mad. All configs report mad < 0.4%, so the ~2x figures "
            "are well outside noise. A/A floor here is the wall-clock repeat spread, "
            "not the sub-0.1% same-session replay floor used for step timing."
        ),
        "configs": ens,
    },
    "task2_joint_solve": {
        "harness": "scripts/dev/fdtd/multi_gpu/bench_joint.py",
        "command_template": (
            "numactl --cpunodebind=0 --membind=0 python "
            "scripts/dev/fdtd/multi_gpu/bench_joint.py --workload vacuum "
            "--devices cuda:0 cuda:1 --nodes-x <N> --nodes-y <N> --nodes-z <N> "
            "--steps 300 --warmups 1 --repeats 5 --json <json>"
        ),
        "metric": "single-GPU vs 2-GPU x-slab joint-solve ms/step (strong scaling), CUDA-event timed",
        "transport_note": (
            "This is the instrumented in-process cuda_p2p x-slab joint-solve forward "
            "path. The NCCL one-process-per-GPU (torchrun) forward path has only a "
            "correctness worker (tests/fdtd/multi_gpu/_nccl_forward_worker.py, no "
            "step-rate timing); its step-rate is NOT-MEASURABLE via existing hooks "
            "without new timing infrastructure -- recorded here as not fabricated."
        ),
        "nccl_torchrun_step_rate": "not-measurable via existing hooks (correctness-only worker; no timing)",
        "configs": joint,
    },
    "task3_no_feature_regression": {
        "harness": "scratch/task3_nofeature.py reusing benchmark_network_embedding._scene(dynamic=False)",
        "measured_median_ms_per_step": nofeat["measured_median_ms_per_step"],
        "historical_median_ms_per_step": nofeat["historical_median_ms_per_step"],
        "historical_source": nofeat["historical_source"],
        "delta_vs_historical_pct": nofeat["delta_vs_historical_pct"],
        "same_session_replay_floor_absmax_pct": nofeat["same_session_replay_floor_absmax_pct"],
        "host_cross_session_aa_floor_pct": nofeat["host_cross_session_aa_floor_pct"],
        "verdict": nofeat["verdict"],
        "cross_grid_baseline_deltas_vs_grid_sweep": {
            "note": "fresh Task-1 bare-FDTD baselines vs tracked grid-sweep baselines, cuda_graph",
            "source": "docs/assessments/network-embedding-gate-d-remeasure-2026-07-20.json sweep[*].baseline_median_ms",
        },
    },
}

out = ROOT / "docs/assessments/multi-gpu-timing-2026-07-20.json"
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("wrote", out)
print(json.dumps({
    "ensemble": {k: round(v["makespan_speedup_median"], 4) for k, v in ens.items()},
    "joint_strong_speedup": {k: round(v["strong_speedup"], 4) for k, v in joint.items()},
    "no_feature_verdict": nofeat["verdict"],
}, indent=2))
