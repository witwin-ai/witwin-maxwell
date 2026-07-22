# Track F3 acceptance — stage F3a (single-device array scene-gradient VJP)

Anchor: worktree `.worktrees/wf3-array-vjp`, branch `fable/array-scene-vjp`, base master `15b38d3`.
Env: conda `maxwell`; `CUDA_HOME=.../nvidia/cu13`; `PYTHONPATH=<worktree>`.

## Delivered

1. `ArrayBasisData.scene_gradient_vjp(...)` (`witwin/maxwell/array.py`) — replaces the
   fail-closed `NotImplementedError` with the aggregated per-column adjoint of the
   linear beam combine `E = sum_n w_n e_n`. Caller supplies live per-column
   embedded far-field columns `(e_theta_n, e_phi_n)` `[F, T, P]` from re-run
   forwards; the method forms the combined field, takes `cot_E = autograd.grad(L, E)`,
   seeds each column with `conj(w_n) * cot_E`, and sums the per-column VJPs onto the
   scene parameters in a caller-controllable deterministic `reduction_order`.
   Accepts an `objective` callable or pre-computed `field_cotangents`; single- or
   multi-tensor `parameters`; `[N]` or `[F, N]` weights.
2. Module helper `_single_beam_weights(...)` validating single-beam weights to `[F, N]`.
3. Guard-census reconciliation: capability guard removed, `CAPABILITY_GUARD_BUDGET`
   `176 -> 175` (`tests/api/public/test_guard_census.py`), census doc updated
   (`docs/reference/fdtd-capability-guard-census.md`).
4. Tests: `tests/rf/array/test_array_scene_gradient.py` (new); pinning test in
   `tests/rf/array/test_array_codebook.py` converted from fail-closed to a passing
   wired/fail-closed-on-detached gate.
5. `FEATURE_LIST.md` subsection `f3-array-scene-vjp`.

## Weight-conjugation convention (derived + verified, not assumed)

For real `L` of `E = sum_n w_n e_n` (`w_n` complex constants), the cotangent PyTorch
propagates to a leaf is `cot_E = autograd.grad(L, E)` (conjugate-Wirtinger). The
complex multiply `w_n * e_n` backward sends it to column `n` as `conj(w_n) * cot_E`,
so the per-column adjoint seed is `seed_n = conj(w_n) * cot_E = w_n^* . (dL/dE)^*`
and `dL/dtheta = sum_n VJP_n(seed_n)`. Verified structurally (rides PyTorch's own
convention for both the `L->E` and `E->params` steps) rather than by re-deriving
Wirtinger signs by hand: the seeded per-column sum is **bit-identical** to
end-to-end `autograd.grad(L, params)` for a fixed reduction order
(`test_scene_gradient_vjp_matches_end_to_end_autograd`, `torch.equal`).

Zero-extra-FDTD-forward-steps is a forward-combine contract only (plan 06 Phase 1);
gradients legitimately re-run the per-column forwards, cited in the docstring.

## Reduction-order determinism (measured)

A **fixed** order is bitwise deterministic (`torch.equal` passes). Two **different**
orders, and per-column accumulation vs autograd's internal reduction for a shared
parameter, agree only up to floating-point non-associativity: measured `< 1e-12`
rtol/atol on float64 (`test_scene_gradient_vjp_reduction_order_invariant`,
`test_scene_gradient_vjp_supports_multiple_parameters`). F3b's 1-vs-2-GPU parity
should therefore assert_close at that floor (or fix a single global order for
bitwise), not bitwise across differing gather orders.

## Test inventory (commands + counts)

All prefixed with:
`export CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13; export PATH="$CUDA_HOME/bin:$PATH"; export PYTHONPATH=<worktree>`

- CPU aggregation math + validation + census (`CUDA_VISIBLE_DEVICES=""`):
  `conda run -n maxwell python -m pytest tests/rf/array/test_array_scene_gradient.py tests/rf/array/test_array_codebook.py tests/api/public/test_guard_census.py -q`
  => **32 passed, 2 skipped** (the 2 skipped are the CUDA gates).
- CUDA end-to-end FDTD gates:
  `conda run -n maxwell python -m pytest tests/rf/array/test_array_scene_gradient.py::test_scene_gradient_vjp_matches_fd_on_fdtd_array tests/rf/array/test_array_scene_gradient.py::test_scene_gradient_vjp_second_column_shifts_gradient -q`
  => **2 passed**.
- Adjacent suites:
  `conda run -n maxwell python -m pytest tests/api/public/test_public_api.py tests/api/public/test_simulation_smoke.py tests/rf/array/test_array_contracts.py tests/rf/array/test_array_mimo.py tests/rf/array/test_array_persistence.py tests/api/public/test_guard_census.py -q`
  => **90 passed** (with codebook).

Key gate numbers (reproducible by the nodes above):
- CPU synthetic FD gate (`test_scene_gradient_vjp_matches_central_difference`):
  method-vs-central-difference relative error `~2.3e-9` (float64), gate `< 1e-6`.
- CPU autograd equivalence: `torch.equal` exact.
- CUDA FDTD FD gate (`test_scene_gradient_vjp_matches_fd_on_fdtd_array`): 2-column
  NF2FF array, trainable `MaterialRegion` density, `time_steps=192`, dominant-voxel
  scene-gradient relative error gated `< 3e-2` (FDTD float32 forward, NF2FF float64
  transform; matches the FDTD-adjoint tolerance floor of the sibling multisource
  adjoint gates).
- Weight-gradient regression (`test_combine_weight_gradient_unchanged_by_scene_gradient_addition`):
  `combine()` weight gradient vs central difference `< 1e-4`.

## Falsifications performed (recorded)

1. **Wrong weight conjugation.** Scratch-edit `witwin/maxwell/array.py`
   `conjugate_weights = torch.conj(weight_matrix)` -> `= weight_matrix`.
   `test_scene_gradient_vjp_matches_end_to_end_autograd` and
   `test_scene_gradient_vjp_matches_central_difference` both went RED (autograd
   `torch.equal` failed; FD gradient sign/magnitude wrong, e.g. `abs(gradient)`
   diverged). Restored -> both green. Also pinned as a permanent guard by
   `test_scene_gradient_vjp_weight_conjugation_is_load_bearing` (method vs a manual
   non-conjugated seed differ materially).
2. **Dropped per-column contribution.** Scratch-edit inserting `if index == 0:
   continue` into the per-column loop (skip column 0's VJP).
   `test_scene_gradient_vjp_matches_end_to_end_autograd` and
   `..._matches_central_difference` went RED. Restored -> green. Also guarded on the
   CUDA path by `test_scene_gradient_vjp_second_column_shifts_gradient` (zeroing one
   column's weight moves the aggregate gradient by `> 10%`).

Falsification method: temporary in-place edit backed up to scratch, targeted pytest
re-run observed red, file restored from backup, re-run observed green
(`grep -c FALSIFY` == 0 after restore).

## Fail-closed / scope

- Detached columns (the retained-basis default) -> `ValueError("... stores detached
  patterns ...")`. The basis still stores columns detached; scene gradients require
  re-run forwards, by design.
- No parameter receives a contribution -> `ValueError`.
- Both/neither `objective`/`field_cotangents`, wrong column count, batched
  `[B, F, N]` weights, non-scalar/complex objective, shape/dtype/device mismatch ->
  `ValueError`/`TypeError` (usage errors, not capability guards; not census-counted).
- Network / S-parameter scene gradients are out of scope for this far-field VJP; the
  method only receives far-field columns. Weight gradients through `combine()`
  (network + far field) are unchanged.

## Known gaps / next agent (F3b)

- Ensemble 2-GPU aggregation: distribute the per-column forwards over
  `witwin/maxwell/execution.py`, gather per-column VJPs in a fixed reduction order,
  gate 1-vs-2-GPU parity at the `< 1e-12` float64 floor measured here (or bitwise
  with a single global order). `scene_gradient_vjp` already accepts `reduction_order`
  and pre-computed `field_cotangents` to support a gather-then-aggregate split.
- The CUDA FD gate uses unit-drive NF2FF columns directly as live embedded columns
  (measured-`a_n` normalization is a per-column constant that does not affect the VJP
  aggregation being validated); a full PortSweep-driven differentiable extraction is
  not required for the aggregation contract.
- F3b owns `docs/assessments/f3-array-scene-vjp-acceptance-2026-07-21.md`,
  FEATURE_LIST ensemble addendum, and the plan-06 revision note.
