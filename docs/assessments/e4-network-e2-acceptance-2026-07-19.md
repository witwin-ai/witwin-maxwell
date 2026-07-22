# Track E4 (plan 03) acceptance — E4b: coupling fixed-cost reduction, delay checkpoint, WavePort disposition

Date: 2026-07-19
Worktree: `.worktrees/we4-network-gaps`, branch `fable/network-e2`
Env: conda `maxwell`, `CUDA_VISIBLE_DEVICES=1`, torch 2.13.0+cu130, CUDA 13.0, NVIDIA RTX A6000.
Builds on E4a (commit `26349db`, cascade/termination helpers + cross-check + passivity suite).

This doc covers E4b deliverables 4 (fixed-cost reduction + op-stream artifacts + no-regression),
5 (explicit-delay adjoint checkpoint support), and 6 (WavePort disposition), plus census /
FEATURE_LIST bookkeeping.

## Delivered items

### 4. Embedded-network coupling fixed-cost reduction (code now, timing later)

- **What changed.** The delay-free ordinary-Y same-step branch-current solve
  `M x = C@state + D@v` (M = I + D·diag(Z_f), constant across the whole run) was replaced by two
  precomputed composite matvecs `branch_current = (M^-1 C)@state + (M^-1 D)@v`. The composite
  operators `gain_state = M^-1 C` and `gain_voltage = M^-1 D` are formed once at prepare time by
  **LU-solving the constant C and D against the already-computed pivoted LU of M** (`torch.linalg.lu_solve`)
  — not by forming a naive matrix inverse. The per-step sequential triangular substitution
  (`_lu_solve_out`, O(port) tiny kernels) is removed from this path; `_lu_solve_out` / `_native_lu_solve_out`
  remain for the delayed reference-plane zero-subset solve.
- **Consequence.** `native_lu` no longer changes the delay-free result: the eager and CUDA-graph
  coupling paths are now bitwise-identical. The narrow full-step graph (which already used native LU)
  is byte-unchanged; the eager path and the network-only CUDA-graph capture (which used the Python-loop
  LU) both drop to the composite path.
- **(a) op-stream evidence (perf-opcount gate, NO wall-clock).** Connected 8-port/order-32 delay-free
  feedback block, eager, grid 48^3:
  - **launches/step: 78 -> 27 (65.4% reduction)**; DtoD/step: 4 -> 4 (unchanged, from the four matvec
    view writes); allocs/step: 0 -> 0.
  - **Pre-registered target: >= 30% launch reduction**, justified after profiling: the 56-launch solve
    is dominated by the sequential 8x8 triangular substitution (~51 launches), which collapses to two
    dense matvecs. Measured 65% clears the target with margin.
  - Artifact: `docs/assessments/e4-network-coupling-op-stream-2026-07-19.json` (both schedules profiled
    on the same prepared solver via `tests/support/network_coupling_op_stream.py`; the legacy
    sequential-LU schedule is re-driven from the still-present `_lu_solve_out`, so before/after is
    reproducible in one command).
- **(b) no-regression.** The composite and sequential-LU solves are mathematically identical; the
  op-stream artifact records `numerical_equivalence_residual_over_bound = 3.7e-4` (< 1), i.e. the two
  evaluation orders differ only within the ill-scaled matvec's floating-point roundoff bound
  (benchmark cond(M) = 1.0, ‖C‖ ~ 6e5). Load-bearing physics no-regression is carried by the existing
  gates that pass unchanged: `test_four_port_fdtd_currents_match_embedded_admittance` (D-dominated,
  exercises `gain_voltage`), the residue-heavy `test_network_multiport_fdtd` / graph-matches-eager
  gates (exercise `gain_state`), and `test_prepared_lu_matches_oracle_for_ill_conditioned_direct_loop`
  (rtol 1e-9 at loop_condition > 1e5).
- **(c) suites green.** Full `tests/rf/network/` + network adjoint remain green (see test inventory).
- **Deferred (out of scope this round, per brief): ms/step re-measurement in an exclusive GPU window.**
  No wall-clock number is quoted from this shared-GPU session.

### 5. Explicit-delay checkpoint/resume + adjoint disposition

- **Checkpoint schema + forward resume (landed).** The frozen FDTD checkpoint schema now captures the
  bidirectional-delay state per embedded network: `forward_ring`, `reverse_ring`, the four Thiran
  fractional-filter memory vectors (`{forward,reverse}_previous_{input,output}`), and the shared ring
  `cursor` (per-step scratch — sample/temporary buffers and read/write index vectors — is recomputed
  from the cursor and is deliberately excluded). Capture, restore-targets, and the schema `state_names`
  all include the new family. This fixes a latent lossy-resume bug: a resumed delayed network previously
  restarted its reference planes from zero.
- **Gate: checkpoint-replay bitwise reproduction with delay active.**
  `test_delayed_network_resume_matches_uninterrupted_run`: `run_until(24)` + `run(resume_from=...)`
  reproduces the uninterrupted delayed two-port fields (`torch.equal` on Ez) and embedded-network V/I
  bit-for-bit; the delay ring state is asserted present and nonzero in the checkpoint.
  `test_zeroing_captured_delay_state_breaks_resume` is a built-in drop-one falsification.
- **Differentiable adjoint through delay: intentionally fail-closed (no half-landing).** The discrete
  network adjoint reconstructs the forward from checkpoint-bounded state and reverses a **segment-local,
  self-contained same-step recurrence**. Explicit delay breaks that invariant two ways: (i) the
  bidirectional reference-plane ring couples a step to samples written up to `max_delay_steps` earlier,
  a coupling that can span checkpoint segments, so a step's read VJP must flow to a write in a possibly
  different segment; (ii) the Thiran fractional-delay filter is an IIR recurrence over consecutive steps.
  A correct delay adjoint therefore needs a delay-aware **reverse ring** whose cotangents are carried
  across segments (alongside the network-state cotangent) plus a reverse of the IIR filter — a
  substantial, high-risk reverse that is out of this stage's scope. Rather than ship an approximate
  gradient, both rejection sites (`fdtd/networks.py::replay_network_runtimes`,
  `fdtd/adjoint/bridge.py::_validate_supported_configuration`) now fail closed with that precise reason.
  The forward-resume schema work above is the enabling prerequisite a future agent needs.

### 6. WavePort embedding disposition (stays fail-closed)

- WavePort-as-network-terminal remains rejected. The rejection message (scene-level guard
  `scene.py`, and the compiler guard `compiler/networks.py`) was made accurate: an embedded state-space
  network couples through a scalar voltage/current terminal on a single lumped Yee edge (LumpedPort or
  resolved TerminalPort), whereas a WavePort is a modal port defined by a cross-sectional mode-overlap
  field pattern with no scalar time-domain terminal (V, I) contract — a missing design contract, not a
  bug. Guarded by `test_network_embedding_rejects_waveport_terminal_with_accurate_reason`. Not attempted
  this round, per brief.

## Files changed

- `witwin/maxwell/fdtd/networks.py` — composite `gain_state`/`gain_voltage` fields + prepare-time
  computation + rewritten delay-free solve; sharpened delay-replay rejection message.
- `witwin/maxwell/fdtd/checkpoint.py` — `network_delay_state_name`, `iter_network_delay_state_specs`,
  `network_delay_state_names` schema field wired into `state_names`, `checkpoint_schema`,
  `capture_checkpoint_state`, and `_checkpoint_tensor_targets`.
- `witwin/maxwell/fdtd/adjoint/bridge.py` — sharpened explicit-delay rejection message (still matches
  the existing `"explicit delay state"` gate).
- `witwin/maxwell/scene.py`, `witwin/maxwell/compiler/networks.py` — accurate WavePort rejection message.
- `tests/support/network_coupling_op_stream.py` (new) — reproducible before/after op-stream tally +
  numerical-equivalence measurement and JSON artifact generator.
- `tests/rf/network/test_network_coupling_op_stream.py` (new) — launch-reduction gate (>= 30%) +
  roundoff-bounded no-regression gate.
- `tests/rf/network/test_network_delay_checkpoint.py` (new) — delay checkpoint/resume bitwise round-trip
  + drop-one falsification.
- `tests/rf/network/test_network_block_contract.py` — WavePort disposition guard test.
- `docs/assessments/e4-network-coupling-op-stream-2026-07-19.json` (new, git add -f) — op-stream artifact.
- `FEATURE_LIST.md` — additive `e4b-network-coupling` subsection.

## Test inventory (env: conda `maxwell`, `CUDA_VISIBLE_DEVICES=1`)

- `tests/rf/network/test_network_coupling_op_stream.py` — 2 passed.
- `tests/rf/network/test_network_delay_checkpoint.py` — 2 passed.
- `tests/rf/network/test_network_block_contract.py` — 9 passed (incl. new WavePort guard).
- `tests/rf/network/test_network_runtime.py test_network_multiport_runtime.py test_network_observer_hot_path.py test_network_multiport_fdtd.py` — 24 passed.
- `tests/gradients/test_fdtd_network_adjoint.py tests/rf/network/test_network_delay*.py` — 27 passed.
- `tests/api/public/test_guard_census.py` — 3 passed (capability-guard census budget unchanged at 176).
- Broad adjacent battery `tests/rf/ tests/gradients/test_fdtd_network_adjoint.py test_fdtd_adjoint_bridge.py test_fdtd_rf_lumped_adjoint.py test_fdtd_circuit_adjoint.py` — **773 passed, 7 xfailed** (0 failed).
- `tests/api/public/test_public_api.py test_simulation_smoke.py tests/materials/surface_impedance/test_surface_impedance_resume.py tests/fdtd/multi_gpu/test_network_owner.py` — **38 passed, 7 skipped** (checkpoint-schema change + distributed composite solve both clean).

## Falsifications recorded (all red -> restored -> green, no residue)

- **F1 (gain_voltage composite):** scale `gain_voltage` by 1.05 at prepare -> the D-dominated physics
  gate `test_four_port_fdtd_currents_match_embedded_admittance` goes RED (AssertionError, relative_error
  > 0.02). Restored -> green.
- **F1b (gain_state composite):** scale `gain_state` by 1.05 -> the C-dominated op-stream no-regression
  gate `test_composite_solve_matches_legacy_lu_no_regression` goes RED (residual/bound ratio >> 1).
  Restored -> green.
- **F2 (delay checkpoint capture):** short-circuit `iter_network_delay_state_specs` to yield nothing ->
  `test_delayed_network_resume_matches_uninterrupted_run` goes RED (delay state absent from checkpoint).
  Restored -> green.
- **F3 (WavePort guard):** disable the scene-level `isinstance(port, WavePort)` guard ->
  `test_network_embedding_rejects_waveport_terminal_with_accurate_reason` goes RED (no matching raise).
  Restored -> green.
- The built-in `test_zeroing_captured_delay_state_breaks_resume` is a permanent drop-one falsification
  (zeroing the captured ring/filter state must diverge the resumed field).

## Pre-registered targets / tolerances

- Launch reduction target: >= 30% (measured 65.4%).
- No-regression: composite vs sequential-LU residual within the matvec floating-point roundoff bound
  (`residual/bound < 1`; measured 3.7e-4); ill-conditioned parity rtol 1e-9 preserved.
- Delay resume: exact (`torch.equal`) on fields and V/I.

## Known gaps / handoff

- Differentiable adjoint for explicit-delay embedded networks is fail-closed (see deliverable 5). The
  checkpoint schema now carries the delay ring/filter/cursor, which is the prerequisite; the remaining
  work is the segment-crossing reverse ring + IIR-filter reverse.
- ms/step re-measurement of the coupling fixed-cost reduction is deferred to a post-merge exclusive GPU
  window (timing out of scope this round). The op-stream launch-count evidence is the controlling
  perf artifact.
- Resume **disk** serialization (`resume.py::_SCHEMA_TUPLE_FIELDS`) already excludes network state
  (pre-existing); in-memory `run_until`/`run(resume_from=...)` (which uses the live schema) is what the
  delay-resume gate exercises and what fixes the lossy-resume bug. Extending disk serialization to
  network + delay state is a separate, untouched follow-up.
- Capability-guard census budget unchanged at 176: the message edits are to existing guards, and the
  composite change removes no `raise`.

## Exact commands

```bash
export CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=<worktree>
export CUDA_VISIBLE_DEVICES=1
# op-stream artifact
conda run -n maxwell --no-capture-output python -m tests.support.network_coupling_op_stream \
  --output docs/assessments/e4-network-coupling-op-stream-2026-07-19.json
# gates
conda run -n maxwell --no-capture-output python -m pytest \
  tests/rf/network/test_network_coupling_op_stream.py \
  tests/rf/network/test_network_delay_checkpoint.py \
  tests/rf/network/test_network_block_contract.py \
  tests/gradients/test_fdtd_network_adjoint.py \
  tests/api/public/test_guard_census.py -q
```
