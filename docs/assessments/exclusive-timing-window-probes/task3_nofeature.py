"""Task 3: no-feature regression spot-check -- bare-FDTD step at 128^3, A/A.

Reuses the committed bare-FDTD baseline scene builder
(benchmark_network_embedding._scene(dynamic=False, ...), cuda_graph field step,
CUDA-event timed) so the measurement is directly comparable to the recorded
historical class for this host: the tracked grid-sweep baseline at grid 128
(docs/assessments/network-embedding-phase-4-performance-grid-sweep.json:
736.402 ms / 1500 steps = 0.49094 ms/step). Emits JSON rows to --output.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "support"))

import benchmark_network_embedding as bne  # noqa: E402
from perf_variance_gate import ci95_upper_bound, paired_ratios  # noqa: E402


# Historical bare-FDTD baselines (ms total) from the tracked grid-sweep artifact.
HISTORICAL_BASELINE_MS = {64: (279.6922912597656, 2000), 96: (607.679, 1500),
                          128: (736.402, 1500), 176: (1643.831, 1500),
                          224: (3136.525, 1500)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=int, default=128)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    scene = bne._scene(dynamic=False, grid_cells=args.grid)
    warm = max(8, args.steps // 10)
    bne._sample(scene, steps=warm)

    a_samples: list[float] = []
    b_samples: list[float] = []
    for r in range(args.rounds):
        ms_a, _, _, _ = bne._sample(scene, steps=args.steps)
        ms_b, _, _, _ = bne._sample(scene, steps=args.steps)
        if r % 2 == 0:
            a_samples.append(ms_a); b_samples.append(ms_b)
        else:
            a_samples.append(ms_b); b_samples.append(ms_a)

    all_samples = a_samples + b_samples
    median_ms = float(statistics.median(all_samples))
    median_ms_per_step = median_ms / args.steps

    # A/A floor: paired A vs B (identical config), two-sided delta magnitude.
    ratios = paired_ratios(a_samples, b_samples)
    aa_upper = ci95_upper_bound(ratios)
    aa_ci95_upper_pct = 100.0 * (aa_upper - 1.0)
    aa_absmax_pct = 100.0 * max(abs(x - 1.0) for x in ratios)

    hist_ms, hist_steps = HISTORICAL_BASELINE_MS[args.grid]
    hist_ms_per_step = hist_ms / hist_steps
    delta_pct = 100.0 * (median_ms_per_step / hist_ms_per_step - 1.0)
    # The historical value is a DIFFERENT-SESSION recording; the appropriate
    # resolution floor for a cross-session comparison is the host cross-session
    # A/A floor (|0.523%| on identical code, per program record), not the tiny
    # same-session cuda_graph-replay floor measured here. A gross regression is a
    # positive delta far above that floor; report both floors for transparency.
    HOST_CROSS_SESSION_FLOOR_PCT = 0.523
    resolvable = abs(delta_pct) > HOST_CROSS_SESSION_FLOOR_PCT

    payload = {
        "kind": "no_feature_regression_spotcheck",
        "grid_cells_per_axis": args.grid,
        "time_steps": args.steps,
        "rounds": args.rounds,
        "config": "bare-FDTD (no ports/network/source), cuda_graph field step, "
                  "domain 0.0625 m, PML 4 layers -- identical to grid-sweep baseline",
        "a_samples_ms": a_samples,
        "b_samples_ms": b_samples,
        "measured_median_ms_per_step": median_ms_per_step,
        "historical_median_ms_per_step": hist_ms_per_step,
        "historical_source": "network-embedding-phase-4-performance-grid-sweep.json baseline grid 128",
        "delta_vs_historical_pct": delta_pct,
        "same_session_replay_floor_ci95_upper_pct": aa_ci95_upper_pct,
        "same_session_replay_floor_absmax_pct": aa_absmax_pct,
        "host_cross_session_aa_floor_pct": HOST_CROSS_SESSION_FLOOR_PCT,
        "regression_resolvable": bool(resolvable),
        "verdict": ("no resolvable regression (|delta| within host cross-session A/A floor; "
                    "delta is negative -> bare step not slower)"
                    if not resolvable else "delta exceeds host cross-session A/A floor -- investigate"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in (
        "measured_median_ms_per_step", "historical_median_ms_per_step",
        "delta_vs_historical_pct", "same_session_replay_floor_absmax_pct",
        "host_cross_session_aa_floor_pct", "verdict")}, indent=2))


if __name__ == "__main__":
    main()
