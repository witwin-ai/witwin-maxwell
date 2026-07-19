# Port hot-path performance — audit step S2 progress (2026-07-18)

> Scope: audit `docs/assessments/next-functional-audit-2026-07-18.md` step S2,
> **excluding** S2b timed measurements (an exclusive-window run scores those).
> This round delivers the baseline artifact, the fresh-profiling conclusion, the
> diagnostics-off guard, and the variance-aware gate machinery. Op-count
> ceilings only — no wall-clock number is asserted here.

## S2 exit-gate mapping

Explicit mapping of each S2 sub-step to its evidence, landing commit, and status.
The scope call is: **S2.2's fix already landed pre-branch at `b70ee2a`** (the
SeriesRLC diagnostics-off fast path), and the audit's conditional weight-table
rewrite precondition is **false** (per-frequency accumulation does not dominate),
so no further code fix is owed for S2.2. **S2.3 timing measurement is deferred to
the exclusive-GPU window (S2b)**; this round lands only the deterministic
op-count evidence and the variance-gate machinery it will consume.

| S2 sub-step | Evidence | Commit | Status |
|---|---|---|---|
| S2.1 harness (reproducible baseline artifact) | `tests/rf/performance/profile_port_hot_path.py` + `docs/assessments/port-hot-path-op-inventory-2026-07-18.json`; op-count ceilings in `tests/rf/performance/test_port_hot_path_op_count.py` and `tests/rf/lumped/test_fdtd_port_end_to_end.py` (class `perf-opcount`) | this branch (`codex/port-perf-s2`) | done (op-count only) |
| S2.2 fix disposition (SeriesRLC fast path; weight-table precondition) | SeriesRLC schedule 62/12/16 → 25/0/3 per step; launches frequency-count independent ⇒ weight-table precondition false; diagnostics default-off pinned by `tests/rf/performance/test_port_energy_diagnostics_default_off.py` | fast path landed pre-branch at `b70ee2a` | landed pre-branch; no new fix owed |
| S2.3 measurement (timed `<5%` / `<2%` gate) | `tests/rf/performance/measure_port_overhead_s2b.py` + artifact `docs/assessments/port-perf-s2b-measurement-2026-07-18.json`; 22-round variance-aware gate on a representative 27 M-cell grid: single-port CI95-upper **1.57 % < 5 %**, per-extra-passive-port CI95-upper **1.53 % < 2 %**; A/A floor half-width 0.07 %; injected-overhead falsification detected (13.8 %). Machinery `tests/support/perf_variance_gate.py` + `tests/rf/performance/test_perf_variance_gate.py` (class `perf-statistical`) | `codex/port-perf-s2b` | **measured (PASS at representative grid; grid-dependent — see §S2.3 measured)** |

## S2.1 — reproducible baseline artifact

`tests/rf/performance/profile_port_hot_path.py` turns the audit's 14.4x port
diagnosis into a machine-readable, reproducible per-step op inventory for the
§9.4 scenarios. It profiles the checkout in place and reconstructs the
pre-optimization baseline (default `eb9258b`) with `git archive` in an isolated
subprocess (its own `PYTHONPATH` and CUDA build dir), then diffs the two.

Committed before/after artifact:
`docs/assessments/port-hot-path-op-inventory-2026-07-18.json`.

Per-step, per-port dispatch counts (A6000, `profile_memory` + acc events):

| scenario | eb9258b (before) | HEAD (after) |
|---|---|---|
| SeriesRLC terminated port | 62 launches / 12 allocs / 16 DtoD | 25 launches / 0 allocs / 3 DtoD |
| passive field-observation port | 30 launches / 5 allocs / 0 DtoD | 30 launches / 5 allocs / 0 DtoD |
| marginal per extra passive port | 30 launches / 5 allocs | 30 launches / 5 allocs |

The SeriesRLC reduction (launches 0.40x, allocations eliminated, DtoD 0.19x) is
the already-landed diagnostics-off fast path (`apply_lumped_runtime`); eb9258b
predates it and always ran the per-step energy/branch bookkeeping. The DtoD=16
before matches the audit's diagnosed figure.

## S2.2 — remaining fix, guided by fresh profiling

Fresh profiling settles the audit's *conditional* menu item ("the port DFT
weight-table path **if** the per-frequency accumulation still dominates"):

- **The per-frequency accumulation does not dominate.** Kernel launches are
  frequency-count independent: one DFT bin and 181 DFT bins launch the identical
  count (passive 30, SeriesRLC 25). The accumulation is a vectorized `[F]`
  update, so reusing `build_dft_step_tables` for the port DFT would not reduce
  the launch schedule — the condition that would justify it is false. This is
  recorded as evidence rather than pursued as a speculative rewrite, per the
  audit's evidence discipline.
- **Energy diagnostics are default-off on every port path.** `prepare_lumped_runtime`
  defaults `diagnostics=False`; no port-preparation call site forwards `True`
  (only the circuit MNA path opts in). The SeriesRLC `allocs == 0` measurement is
  the runtime proof that the allocation-free branch is active. Pinned by
  `tests/rf/performance/test_port_energy_diagnostics_default_off.py`.

Physics parity: no numerics were changed this round, so the bitwise fixed-scene
parity gates in the existing lumped/circuit suites remain the parity evidence;
the op-count tests assert schedule shape only.

## S2.3 — variance-aware performance gate machinery

`tests/support/perf_variance_gate.py` supplies the audit's criterion: a
regression is accepted only when the **one-sided 95% CI upper bound** of the mean
paired-round ratio clears the target (1.05 for the `<5%` single-port target,
1.02 for the `<2%` per-extra-passive-port target), with warmup dropping and MAD
reporting. It is pure arithmetic (Student-t table validated against SciPy to
1e-5) and unit tested with no GPU in
`tests/rf/performance/test_perf_variance_gate.py`, including the key case a
single-point median gate would wrongly pass (mean under target, wide spread →
CI upper bound over target → fail). The existing ABBA orchestrator
`tests/support/benchmark_rf_no_feature.py` now emits this verdict additively as
`variance_gate` without relaxing its existing single-point `passed`.

Op-count ceilings for the port hot path are gated by
`tests/rf/performance/test_port_hot_path_op_count.py` (F-independence, constant
per-additional-port marginal cost, SeriesRLC fast-path `allocs == 0`), with a
recorded falsification against the eb9258b schedule.

## S2.3 — measured (S2b exclusive window, 2026-07-19)

Run in an exclusive single-GPU window (RTX A6000, `cuda:0`, `numactl
--cpunodebind=0 --membind=0`, one measurement process, `nvidia-smi` compute-apps
verified empty before and after every measured session). Harness
`tests/rf/performance/measure_port_overhead_s2b.py`; artifact
`docs/assessments/port-perf-s2b-measurement-2026-07-18.json`. Per-step time is
isolated by two-point subtraction `(t_hi - t_lo)/(steps_hi - steps_lo)` of
CUDA-event-timed full `Simulation.run()` calls (cancels one-time prepare /
result-extraction cost), eager stepping (`cuda_graph=False`, the audit's
outside-the-graph regime), palindromic ABBA per-round ordering, paired per-round
ratios fed to `evaluate_regression_gate(target_ratio=1.05 / 1.02)`.

The comparison isolates the port: the base is a genuine bare-FDTD stepping
reference (source + Yee update + CPML + one 1-frequency point-monitor kernel; a
multi-frequency point monitor or the auto full-field DFT that a no-port
multi-frequency run enables would otherwise dominate the base and flatter the
ratio). Port configs carry the port and its 181-frequency observer, no monitor.

**Primary gate — representative 27,000,000-cell grid (300³ @ 5 mm, 22 rounds): PASS.**

| Gate | Target | Mean | CI95 upper | Verdict |
|---|---|---|---|---|
| single passive LumpedPort + 181-freq observer vs no-port | < 5 % | 1.54 % | **1.57 %** | PASS |
| each additional passive port | < 2 % | 1.46 % | **1.53 %** | PASS |

A/A calibration (base-vs-base, same 22 rounds): two-sided 95 % CI mean +0.126 %,
half-width **0.072 %** — the measurement floor, ~40× below the target margins and
below the previously cited ~0.5 % host floor. It *narrowly excludes zero* (a
+0.13 % residual ordering systematic between the two base blocks); recorded
honestly. It is smaller than the tightest gate margin (per-extra-port: 2.00 −
1.53 = 0.47 %), so it does not overturn the verdicts.

**Grid dependence (the honest headline).** The per-port cost is a roughly fixed
per-step launch cost, not a fraction of the field update: ~0.4 ms/port on small
grids, falling to ~0.07 ms/port once field-update kernels are long enough to hide
the port-observer launch latency (≳7 M cells). The base per-step cost scales with
cell count, so the overhead *ratio* is grid-dependent. Sweep (identical port
geometry, domain scaled at fixed 5 mm cell; 3–4 rounds each):

| cells | base ms/step | 1-port add | +1-port add | single CI95-up | extra CI95-up |
|---|---|---|---|---|---|
| 34,560 | 0.16 | 0.44 | 0.39 | 290 % ✗ | 72 % ✗ |
| 552,960 | 0.28 | 0.32 | 0.40 | 117 % ✗ | 73 % ✗ |
| 1,728,000 | 0.37 | 0.26 | 0.40 | 71 % ✗ | 66 % ✗ |
| 6,635,520 | 1.33 | 0.073 | 0.071 | 5.8 % ✗ | 5.2 % ✗ |
| 13,824,000 | 2.52 | 0.073 | 0.069 | 3.0 % ✓ | 2.9 % ✗ |
| 27,000,000 | 4.84 | 0.077 | 0.074 | 1.6 % ✓ | 1.7 % ✓ |
| 46,656,000 | 7.85 | 0.068 | 0.067 | 1.1 % ✓ | 1.1 % ✓ |

Crossover: single-port `< 5 %` above ~7 M cells; per-extra-port `< 2 %` (the
binding target) above ~27 M cells. **On small/medium grids (≤ ~2 M cells) the
passive-port overhead is large (single-port 70–290 %, per-extra 66–73 %)** —
because the eager 181-frequency port observer's fixed cost is comparable to an
entire bare Yee step there. This directly corroborates the audit's outside-the-
graph port-cost concern and bounds where the §9.4 targets hold.

**Falsification (harness wiring).** A synthetic port-only per-step cost injected
via monkeypatch of `apply_port_runtimes` (fires only when the scene has port
runtimes, so the base stays clean) added 0.669 ms/step at 27 M cells; the
single-port gate flipped PASS→FAIL (CI95 upper 13.79 % > 5 %), while the same
uninjected gate passes at 1.57 %. The gate detects a real regression.

**Honest notes.**
- Verdict is grid-dependent; PASS is asserted at a representative large RF
  full-wave grid (27 M cells). The `< 2 %` per-extra-port target is met only
  above ~27 M cells; smaller grids fail and are reported, not tuned away.
- `cuda_graph=False` throughout (apples-to-apples eager stepping; the passive
  port observer is not CUDA-graph-capturable, so this is also the production
  regime for a passive-port S-parameter run).
- A/A CI narrowly excludes zero (+0.13 %); the floor half-width (0.07 %) is still
  far below the gate margins.
