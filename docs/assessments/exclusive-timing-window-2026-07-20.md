# Exclusive timing window — plan 02/03 deferred gates (2026-07-20)

Main checkout, master `ead70c0`. Exclusive window: single measurement process at
a time, both GPUs (2× RTX A6000) verified idle before every block. Host state:
CPU governor `performance`, GPU persistence enabled, every timing process pinned
with `numactl --cpunodebind=0 --membind=0`. All numbers below are reproducible
from the committed harnesses via the commands recorded in each JSON artifact.

## Task 1 — plan 03 gate (d) re-measurement (after round-E composite matvec coupling)

Artifact: `docs/assessments/network-embedding-gate-d-remeasure-2026-07-20.json`
Harness: `tests/support/benchmark_network_embedding.py` (cuda_graph field+network+port_observer, CUDA-event timed), 6 paired ABBA rounds/grid.

Connected 8-port / order-32 feedback block vs bare FDTD:

| grid | steps | bare ms | connected ms | fixed cost ms/step | overhead % (CI95 up) | <10% gate |
|------|-------|---------|--------------|--------------------|----------------------|-----------|
| 64³  | 2000  | 279.41  | 637.86       | 0.1792             | 128.29 (128.49)      | FAIL |
| 96³  | 1500  | 606.26  | 880.80       | 0.1830             | 45.29 (45.42)        | FAIL |
| 128³ | 1500  | 733.88  | 1009.92      | 0.1840             | 37.61 (37.76)        | FAIL |
| 176³ | 1500  | 1639.49 | 1913.09      | 0.1824             | 16.69 (16.78)        | FAIL |
| 224³ | 1500  | 3148.32 | 3434.31      | 0.1907             | 9.08 (9.12)          | **PASS** |

A/A calibration (bare-vs-bare, grid 224): CI95-upper **0.019%**, abs-max delta
0.030% — cuda_graph replay is extremely stable, so every overhead figure is
well-resolved.

**Verdict.** Gate (d) holds only at **≥224³** (9.08%, CI95-upper 9.12% < 10%);
it FAILS at the representative/default grid (64³, 128%) and at all grids ≤176³.
New fixed per-step coupling cost **median 0.183 ms/step** (range 0.179–0.191),
a modest improvement over the pre-composite 0.193–0.204 ms/step. The **crossover
grid is unchanged (~224³)** and overhead at 224³ improved only 9.64% → 9.08%.

**Key finding.** The round-E composite matvec coupling cut kernel launches 78→27
(−65%, per `e4-network-coupling-op-stream-2026-07-19.json`) but the fixed
per-step cost dropped only ~9%. The connected-step cost is therefore
**compute-bound** (implicit solve + port observers), not launch-bound; reducing
launch count further will not move the crossover grid.

## Task 2 — plan 02 deferred multi-GPU timing

Artifact: `docs/assessments/multi-gpu-timing-2026-07-20.json`

### Ensemble makespan speedup (`tests/support/benchmark_ensemble_speedup.py`)
Serial (1 GPU) / parallel (2 GPU) total wall-clock ratio, 2000 steps, 5 repeats:

| grid | tasks | serial s | parallel s | speedup (mad) |
|------|-------|----------|------------|---------------|
| 96³  | 4     | 3.336    | 1.679      | **1.990** (0.0034) |
| 96³  | 8     | 6.688    | 3.371      | **1.982** (0.0040) |
| 160³ | 4     | 7.189    | 3.599      | **1.997** (0.0015) |
| 160³ | 8     | 14.374   | 7.195      | **1.998** (0.0008) |

Near-ideal 2× on both GPU-bound grids, MAD < 0.4% (well outside noise);
consistent with and above the tracked 128³ = 1.96×.

### Joint-solve forward strong scaling (`scripts/dev/fdtd/multi_gpu/bench_joint.py`)
Single-GPU vs 2-GPU x-slab, vacuum, 300 steps, 5 repeats, in-process **cuda_p2p**
transport (P2P 52.7 GB/s, field parity max_rel = 0):

| cells | single ms/step | 2-GPU ms/step | strong speedup |
|-------|----------------|---------------|----------------|
| 128³  | 0.452          | 0.832         | **0.544×** (slower) |
| 192³  | 1.474          | 0.854         | **1.726×** |

Joint solve is communication-bound at 128³ (2 GPUs slower than 1) and pays off at
192³ (1.73×) — the same amortization crossover as the network coupling.

**Transport caveat.** These are the instrumented in-process cuda_p2p joint-solve
numbers. The **NCCL one-process-per-GPU (torchrun) forward path** has only a
correctness worker (`tests/fdtd/multi_gpu/_nccl_forward_worker.py`) with no
step-rate timing; its step rate is **not-measurable via existing hooks** without
new infrastructure and is recorded as such (not fabricated).

## Task 3 — no-feature regression spot-check

Artifact rows in `multi-gpu-timing-2026-07-20.json` (`task3_no_feature_regression`).
Bare-FDTD 128³ (cuda_graph, identical to grid-sweep baseline), 8 A/A rounds:
measured **0.4905 ms/step** vs historical 0.4909 ms/step → **−0.098%**. Delta is
negative (not slower) and inside the host cross-session A/A floor (|0.523%|).
Cross-grid bare-baseline deltas vs the tracked grid-sweep are all within ±0.4%.

**Verdict: no resolvable regression** from the rounds-D/E machinery
(breakdown pre-scan, monitor passthrough, composite coupling) on the bare field step.

## Reproduction

Every command is stored under each artifact's `command`/`command_template`
field. Env: conda `maxwell`, `CUDA_HOME=.../nvidia/cu13`, prefix every process
with `numactl --cpunodebind=0 --membind=0`.
