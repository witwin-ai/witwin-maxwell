# Track F3 acceptance — stage F3b (ensemble 2-GPU array scene-gradient aggregation)

Anchor: worktree `.worktrees/wf3-array-vjp`, branch `fable/array-scene-vjp`, base master `15b38d3`.
Env: conda `maxwell`; `CUDA_HOME=.../nvidia/cu13`; `PYTHONPATH=<worktree>`. Hardware: 2x RTX A6000 (homogeneous, P2P). Correctness only, no timing claims.

Builds on F3a (single-device `ArrayBasisData.scene_gradient_vjp`, acceptance
`docs/assessments/f3-array-vjp-acceptance-2026-07-21.md`).

## Delivered

1. `witwin/maxwell/array_gradient.py` (new module):
   - `aggregate_scene_gradient_vjp(basis, *, columns, parameters, weights, objective|field_cotangents, reduction_order=None, reduction_device=None)` —
     reduces per-column scene-gradient VJPs whose columns may live on *different*
     devices. It forms the combined field and its cotangent on `reduction_device`
     from detached column values, seeds each column with `conj(w_n) * cot_E` on
     that column's own device, back-propagates the per-column VJP there, moves the
     gradient to `reduction_device`, and sums in a fixed public-port
     `reduction_order`. `parameters` is a per-column sequence of leaf(s) (the same
     object for every column in the single-device case; a per-device replica in the
     2-GPU case). Returns an `AggregatedSceneGradient` (gradient + provenance:
     reduction order, reduction device, per-column devices, port names).
   - `ensemble_scene_gradient_vjp(basis, *, column_forward, weights, execution, ...)` —
     distributes the per-column forwards over the ensemble `DevicePool`
     (`ExecutionPlan` + `execute_plan`, `fail_fast=True`) as independent tasks,
     synchronizes each column's device, then calls `aggregate_scene_gradient_vjp`.
     `column_forward(index, device)` runs column `index`'s forward on `device`
     under autograd and returns `(e_theta_n, e_phi_n, parameters_n)`. The per-column
     adjoint is intentionally *not* routed through `run_many` (that path refuses
     trainable simulations because it runs no backward); the pool only places and
     orders the forwards, and the seeded backward + deterministic reduction run on
     the caller thread.
   - `AggregatedSceneGradient` dataclass (gradient + provenance).
2. Public exports: `aggregate_scene_gradient_vjp`, `ensemble_scene_gradient_vjp`,
   `AggregatedSceneGradient` in `witwin/maxwell/__init__.py`.
3. Tests: `tests/rf/array/test_array_scene_gradient_ensemble.py` (new, 13 nodes).
4. `FEATURE_LIST.md` ensemble addendum (extends the `f3-array-scene-vjp` block).
5. Plan-06 revision note (`docs/plans/next-functional-2026-07/06-array-active-s-mimo.md`,
   §14 append-only).

No capability-guard census change: F3b adds/removes no `NotImplementedError`
capability guard (the `scene_gradient_vjp` guard was already removed in F3a,
budget `176 -> 175`). `CAPABILITY_GUARD_BUDGET` stays 175.

## Why the ensemble split is legitimate (derived)

The seed for column `n` is `conj(w_n) * cot_E`, where `cot_E = dL/dE` depends on the
combined field value `E = sum_n w_n e_n` only. Forming `E` needs the column
*values* (detached copies suffice); the combined-field cotangent taken from a
detached `E` leaf is bit-identical to the one the single-device path takes from
the live combine, because it is the same objective node evaluated at the same `E`.
The per-column VJP `VJP_n(seed_n)` is a self-contained backward through column
`n`'s own graph on its own device; cross-device it only needs the seed moved onto
that device and its gradient moved back to the reduction device. With a fixed
public-port reduction order the sum is associative-invariant, so the aggregated
gradient is invariant to which GPU each column ran on. Verified against the
single-device `scene_gradient_vjp` bit-for-bit
(`test_aggregate_matches_single_device_scene_gradient_vjp`, `torch.equal`).

## 1-vs-2-GPU parity: measured BITWISE

Reproduced by the two committed pytest nodes below (they assert the gates directly;
no auxiliary script is needed). Same objective, same fixed reduction order, only the
column-to-GPU placement differs:

- Synthetic analytic float64 map, 4 columns, 2 frequencies:
  `single_devs=(cuda:0)x4`, `dual_devs=(cuda:0,cuda:1,cuda:0,cuda:0)`,
  **maxabsdiff = 0.000e+00 (bitwise, `torch.equal` True)**.
- Real two-column FDTD array (float32 forward, float64 NF2FF transform, trainable
  `MaterialRegion` density), `dual_devs=(cuda:0,cuda:1)`:
  **maxabsdiff = 0.000e+00, rel = 0.000e+00 (bitwise)**, gradient max ≈ 4.0e4.

So the F3a-anticipated "assert_close at the `<1e-12` float64 floor or bitwise"
resolves to **bitwise** on the homogeneous A6000 pair. The synthetic parity gate
asserts `torch.equal`; the FDTD parity gate keeps a conservative `rel < 1e-4`
robustness band (documented measured value 0) because the FDTD float32 adjoint is
not contractually guaranteed bit-stable across arbitrary hosts.

## Test inventory (commands + counts)

All prefixed with:
`export CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13; export PATH="$CUDA_HOME/bin:$PATH"; export PYTHONPATH=<worktree>`

- CPU aggregation math + fail-closed (`CUDA_VISIBLE_DEVICES=""`):
  `conda run -n maxwell python -m pytest tests/rf/array/test_array_scene_gradient_ensemble.py -q`
  => **10 passed, 3 skipped** (the 3 skipped are the 2-GPU gates).
- Full file with both GPUs:
  `conda run -n maxwell python -m pytest tests/rf/array/test_array_scene_gradient_ensemble.py -q`
  => **13 passed**.
- Adjacent suites + guard census:
  `conda run -n maxwell python -m pytest tests/api/public/test_guard_census.py tests/api/public/test_public_api.py tests/api/public/test_simulation_smoke.py tests/rf/array/test_array_scene_gradient.py tests/rf/array/test_array_contracts.py tests/rf/array/test_array_mimo.py tests/rf/array/test_array_persistence.py tests/rf/array/test_array_codebook.py -q`
  => **106 passed**.
- Ensemble executor regression:
  `conda run -n maxwell python -m pytest tests/fdtd/multi_gpu/test_ensemble_run_many.py tests/fdtd/multi_gpu/test_ensemble_executor.py -q`
  => **22 passed**.

Key gate nodes (reproducible):
- `test_ensemble_one_vs_two_gpu_parity_synthetic` — 1-vs-2-GPU bitwise parity
  (`torch.equal`), asserts both GPUs used.
- `test_ensemble_two_gpu_matches_central_difference_synthetic` — 2-GPU aggregated
  gradient vs central difference, per-slot rel-err gate `< 1e-6` (float64).
- `test_ensemble_fdtd_array_one_vs_two_gpu_parity` — real two-column FDTD array,
  1-vs-2-GPU parity `rel < 1e-4` (measured 0).
- `test_aggregate_matches_single_device_scene_gradient_vjp` — aggregate equals the
  F3a single-device VJP bit-for-bit (`torch.equal`).

## Falsifications performed (recorded)

Method: temporary in-place edit to `witwin/maxwell/array_gradient.py` backed up to
`scratch/array_gradient.bak`, targeted pytest re-run observed RED, file restored
from backup, `grep -c FALSIFY == 0` after restore.

1. **Dropped weight conjugation.** `conjugate_weights = torch.conj(weight_matrix)`
   -> `= weight_matrix`. `test_aggregate_matches_single_device_scene_gradient_vjp`
   and `test_aggregate_per_column_replicas_equal_shared_leaf` went RED (greatest
   relative difference ~4.7 vs the `1e-12` band). Restored -> green.
2. **Skipped a column in the reduction loop.** Inserted `if _pos == 0: continue`.
   Both aggregation-equivalence gates went RED. Restored -> green.
3. **Removed the cross-device gradient move.** `totals[slot] + gradient.to(reduction)`
   -> `totals[slot] + gradient`. `test_ensemble_one_vs_two_gpu_parity_synthetic`
   went RED with `RuntimeError: Expected all tensors to be on the same device
   (cuda:0 and cuda:1)` — the per-device reduction move is load-bearing for the
   2-GPU aggregation. Restored -> green.

## Fail-closed / scope

- Detached columns (no autograd graph) -> `ValueError(... stores detached patterns
  ...)`; the retained basis stores columns detached by design.
- `parameters` count != N, inconsistent per-column parameter structure
  (leaf count/shape/dtype), both/neither `objective`/`field_cotangents`, wrong
  column count, shape/dtype mismatch -> `ValueError`/`TypeError` (usage errors, not
  capability guards; not census-counted).
- `ensemble_scene_gradient_vjp` with a non-`MultiGPUExecution` execution ->
  `TypeError`; a failed column forward -> `RuntimeError` chaining the underlying
  `DistributedFailure` exception.
- Out of scope (unchanged from F3a): the NCCL joint-solve adjoint (independent
  Simulations only); network / S-parameter scene gradients (this is a far-field
  VJP); `combine()` weight gradients (regression-gated in F3a).
- Zero-extra-FDTD-forward-steps stays a forward-combine contract only (plan 06
  Phase 1); gradients legitimately re-run the per-column forwards.

## Known gaps / next

- The driver distributes the per-column *forwards* over the pool and runs the
  seeded backward + reduction on the caller thread (correctness path). A fully
  pool-scheduled backward is possible but unnecessary for correctness and would
  add no parity guarantee beyond the deterministic reduction already proven.
- Timing / throughput scaling is explicitly not claimed (shared-GPU correctness
  window); the plan-06 §14 revision records this as deferred-pending-exclusive-window.
