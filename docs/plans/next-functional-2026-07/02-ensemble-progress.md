# Ensemble Multi-GPU Execution — Progress (Plan 02, Phases 0/1/2 ensemble track)

> Status: Phase 1 correctness legs delivered; Phase 2 RF-aware ensemble correctness delivered.
> Timing / speedup legs DEFERRED-pending-exclusive-window.
> Branch: `codex/ensemble-execution`. Plan of record: `02-multi-gpu-execution.md`.

## Scope delivered

This track implements the **ensemble half** of plan 02 (§4.1, §5.1, §5.2). The
joint-solve half (`FDTDParallelConfig`, `DistributedFDTD`, x-slab decomposition)
already existed on master and is untouched here. Ensemble execution and joint
solve are explicit, separate strategies that share only the device/record layer;
they never share physical state and are rejected from composing on one
`Simulation` (they compose later per plan §3).

## Execution package (`witwin/maxwell/execution/`)

- `MultiGPUExecution` (public) — structural-only ensemble descriptor
  (`.ensemble(devices=..., max_concurrency=..., placement=..., fail_fast=...)`),
  mirroring `FDTDParallelConfig`'s CPU-constructible validation contract.
- `run_many(simulations, *, execution)` (public) — ordered ensemble entry.
- `ResultSequence`, `ExecutionRecord`, `DistributedFailure`, `FailureKind`
  (public) — submission-ordered container, per-task record, structured failure.
- `DevicePool` / `DeviceLease` (internal) — deterministic first-free-in-order
  leasing, one solver per GPU by default, per-device peak-concurrency tracking.
- `ExecutionPlan` / `ExecutionTask` / `execute_plan` (internal) — immutable
  ordered task list and the thread-pool scheduler. The shared layer knows
  nothing about Yee fields, ports, or PDE stencils.
- `capacity.py` — per-task memory-estimation preflight that reuses
  `fdtd/distributed/capacity.py` + `output.py` estimators for a single-scene
  resident footprint (`_FIELD_STATE_MULTIPLIER` conservative heuristic + local
  DFT working set).

## Phase 1 (Ensemble MVP) — correctness legs: DONE

Exit-gate evidence (`tests/fdtd/multi_gpu/test_ensemble_executor.py`,
`test_ensemble_run_many.py`):

- N independent Simulations over 2 GPUs return a `ResultSequence` **identical**
  (order + values) to an isolated serial run on the same leased device
  (`torch.equal` on the returned fields) — proves order fidelity and no
  cross-task state bleed.
- Device lease respected: returned tensors live on the leased device; no device
  is leased beyond `per_device_concurrency`
  (`DevicePool.peak_concurrency`), and no task runs on a non-leased device.
- Failure matrix: a task raising inside its worker becomes an ordered
  `DistributedFailure` (kind `runtime`, original exception chained) while the
  other tasks complete; `fail_fast` marks unstarted tasks `cancelled`; the
  memory preflight rejects an oversized task (kind `capacity`) before it runs.
- Guards (ValueError, so **no capability-guard census change**): ensemble +
  trainable, ensemble + `FDTDParallelConfig`, `SceneModule` input, and two tasks
  sharing one `Scene` object are all rejected.

## Phase 2 (RF-aware ensemble) — correctness legs: DONE

Exit-gate evidence (`tests/fdtd/multi_gpu/test_ensemble_network_sweep.py`):

- A representative **4-port** `PortSweep` `NetworkRunManifest` expands into four
  independent single-active-port column tasks (plan 01's
  `build_network_column_run`) run through the same executor. Plan 01's
  `aggregate_network_columns` — **reused, not duplicated** — assembles the
  matrix; 02 never understands or generates an S-matrix.
- Single-device ensemble path reproduces the serial matrix **bit-for-bit**
  (`torch.equal` on `NetworkData.s`/`.z0`), isolating the executor path.
- Two-GPU path returns the same ordered matrix (tight `assert_close`) with the
  gathered ports/matrix on one `result_device` and **per-column provenance** in
  `Result.solver_stats["ensemble"]["column_devices"]`.
- Trigger: `Simulation.prepare().run(execution=...)` on a `PortSweep`
  Simulation; the coordinator pre-prepared scene is intentionally not reused so
  each column prepares on its own leased device.

## DEFERRED — timing / speedup (pending exclusive-GPU window)

Per the shared-machine constraint, **no** timing/benchmark number is measured or
asserted here. The measurement hooks are implemented — `ExecutionRecord`
carries `wall_time_s` and a best-effort per-device `device_time_s` (CUDA events,
exception-safe) — but the Phase 1 "任务确实并发" and Phase 2 "可解释的任务级
speedup" evidence, ensemble throughput/makespan/utilization, and the public
benchmark family remain DEFERRED-pending-exclusive-window. No number is faked.

## Falsification checks (scratch monkeypatch → red → restore)

- Order: `execute_plan` reversed the entry write → order test and RF exact-match
  test both red; restored.
- Leasing: `DevicePool._has_free_slot` forced `True` → over-subscription /
  device-coverage test red; restored.
- Capacity preflight: `_preflight_capacity` short-circuited to `None` → the
  oversized-task test ran the task and went red; restored.
- Failure isolation: worker `except` swallowed the exception as a success →
  failure-matrix and fail-fast tests red; restored.
- Device lease binding: `_run_simulation_on_device` ignored the leased device →
  the run_many lease-respected identity test red; restored.
