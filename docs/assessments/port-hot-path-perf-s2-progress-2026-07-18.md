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
| S2.3 measurement (timed `<5%` / `<2%` gate) | variance-aware gate machinery `tests/support/perf_variance_gate.py` + unit tests `tests/rf/performance/test_perf_variance_gate.py` (class `perf-statistical`); no wall-clock number asserted this round | this branch (machinery only) | **deferred to S2b exclusive-GPU window** |

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

## For the S2b exclusive window

Measure the paired baseline/candidate step time for (1 port + 181 freqs) and the
per-extra-passive-port increment over several ABBA rounds, feed the per-round
ratios to `evaluate_regression_gate(..., target_ratio=1.05 / 1.02)`, and assert
`RegressionGateResult.passed`. Do not assert a single-point median.
