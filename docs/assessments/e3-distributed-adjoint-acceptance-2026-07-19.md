# Track E3 — Distributed adjoint acceptance (2026-07-19)

Worktree `.worktrees/we3-mgpu-adjoint`, branch `fable/distributed-adjoint`.
Invocation for every command below:

```bash
export CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=/home/xingyu/code/witwin/witwin-maxwell/.worktrees/we3-mgpu-adjoint
export CUDA_VISIBLE_DEVICES=0,1
conda run -n maxwell --no-capture-output python -m pytest <targets> -q
```

> Slice 2 (defense-in-depth trainable guard on `DistributedFDTD`, forward monitor
> gather, FEATURE_LIST) is E3b's deliverable and will be appended to this doc.

## Post-audit fix — Hy/Ey CPML psi axis-convention bug (2026-07-19)

An audit found that the S4 CPML-trainable adjoint shipped percent-level-wrong
gradients whenever the psi memory is numerically active (long run, objective
sensitivity flowing through a PML region). Two independent latent bugs on the
same `(pos, neg) = (z, x)` axis order of the Hy/Ey components — pre-existing at
parent `039cead`, made load-bearing by S4 — were fixed:

1. **Native reverse adjoint-psi carry** (`witwin/maxwell/fdtd/adjoint/native.py`,
   the operative bug): `_reverse_electric_cpml_ey` and `_reverse_magnetic_cpml_hy`
   read `AdjPsiPosPost`/`AdjPsiNegPost` from the swapped keys
   (`psi_ey_x`/`psi_hy_x` for the *pos* = z-family) while writing
   `AdjPsiPosPrev`/`AdjPsiNegPrev` to the canonical keys (`psi_ey_z`/`psi_hy_z`).
   Since `pre_step_adjoint` feeds straight back as the next `adjoint_state`
   (`bridge.py` `adjoint_state = dict(step_result.pre_step_adjoint)`), the psi
   cotangent recursion connected the wrong family step-to-step. Ex/Ez/Hx/Hz were
   already self-consistent; only Hy/Ey (5 call sites: the two split phase helpers
   + the conductive/nonlinear electric variants) were swapped. Fixed to match the
   canonical `_CPML_MEMORY_SPECS` convention.
2. **Forward replay psi storage** (`witwin/maxwell/fdtd/adjoint/core.py`, latent):
   `_step_state` and the `_forward_*_fields_cpml` mirrors unpacked
   `_update_magnetic_component`/`_update_electric_component` returns as
   `hy, psi_hy_x, psi_hy_z` / `ey, psi_ey_x, psi_ey_z`, storing the advanced
   z-family psi under the x key (the imag branch at ~line 3525 was already
   correct). Inert for a design region in the non-PML interior (psi ~ 0 there),
   but it corrupts the replayed psi ~4400x vs the native kernel and would corrupt
   the gradient for any design region overlapping a PML. Fixed to
   `hy, psi_hy_z, psi_hy_x` / `ey, psi_ey_z, psi_ey_x`.

**Why the S4 headline gates missed it:** the calibrated gate scene runs STEPS=50
with the probe in the interior, where psi maxima are ~1e-15..2.5e-6 vs |E|~1 —
psi-inert, so both swaps are undetectable (zeroing the replay psi leaves the
parity metric bit-identical). The 1-vs-2-GPU parity gates also compare
distributed against single-GPU, and both used the same buggy reverse, so they
agreed while both being wrong. New psi-active tests close the gap.

**Reproduction / evidence (single GPU, RTX A6000):** the audit's psi-active scene
(the CPML Box-density scene, STEPS=400, probe at x=0.48 inside the high x-PML):
before fix analytic grad `-298.72` vs central FD `-288.5`, rel `3.43e-2`
(h-converged over h=2e-2..5e-4); after fix analytic `-288.48` vs FD `-288.48`,
rel `9.1e-6`.

**New committed gates + falsifications:**

| Gate | Test | Result | Falsification |
|---|---|---|---|
| psi-active single-GPU CPML adjoint vs central FD (probe in PML) | `tests/gradients/test_fdtd_cpml_psi_active_adjoint.py::test_cpml_psi_active_adjoint_matches_central_fd` | passed (rel 9.1e-6, gate 2e-3) | reverting the native `AdjPsiPost` keys → rel 3.43e-2 (fail, 17x over gate) |
| distributed CPML replay reproduces native forward OWNED psi + fields | `tests/fdtd/multi_gpu/test_adjoint_replay.py::test_distributed_cpml_replay_reproduces_forward_owned_psi_and_fields` | passed (psi rel ~1e-7, gate rtol 1e-4/atol 1e-6) | reverting the `_step_state`/mirror unpack → psi moves ~O(scale) (fail) |

Non-vacuity in-code: both tests assert the psi memory is a significant fraction
of the field scale and that `psi_hy_z`/`psi_hy_x` are numerically distinct, so a
swap actually changes the compared numbers.

Regression check after the fix: `tests/fdtd/multi_gpu/test_adjoint_parity_cpml.py`
+ `test_adjoint_replay.py` (14+7 passed) and the single-GPU adjoint suites
`test_fdtd_adjoint_rigorous/bridge/materials/p6_acceptance.py` (106 passed) all
green; the fix is bit-identical in the psi-inert regime so the prior gates are
unaffected.

## Slice 1 — S4 distributed CPML-trainable bridge (E3a) — DELIVERED

### Delivered items

1. **CPML-aware forward replay half-steps** (`witwin/maxwell/fdtd/adjoint/core.py`):
   `_forward_magnetic_fields_cpml` / `_forward_electric_fields_cpml` mirror the real
   CPML magnetic/electric halves of `_step_state` exactly and additionally return
   the advanced psi_h / psi_e dicts. `replay_distributed_segment`
   (`fdtd/distributed/adjoint.py`) branches on `uses_cpml`: it threads the twelve
   psi fields through the two explicit halves (electric halo -> magnetic half ->
   magnetic halo -> electric half) and carries the full 18-field CPML state in the
   trajectory. The psi recurrence is a same-x-slice update, so no psi halo is
   copied.
2. **psi-carrying reverse loop** (`fdtd/distributed/adjoint.py`): `_reverse_one_step`
   dispatches to `_reverse_phases_cpml` when `uses_cpml`, running
   `reverse_cpml_phase_electric` -> `exchange_magnetic_adjoint` (Hy/Hz) ->
   `reverse_cpml_phase_magnetic` -> `exchange_electric_adjoint` (Ey/Ez), then the
   shared per-shard `_accumulate_source_term_gradients` tail. The adjoint state
   carries the 12 psi cotangents (`_CPML_STATE_NAMES`, seeded to zero; monitor seeds
   inject E/H only). No psi halo.
3. **Guard relaxation + PML-pinning assertion**:
   - `require_distributed_adjoint_support` accepts `active_absorber_type` in
     {`none`, `cpml`, `stablepml`}; graded-sigma `pml`/`absorber` and every
     dispersive/conductive/nonlinear/aniso/Bloch/TFSF/modulated/coupled channel
     stay rejected (ValueError).
   - `_assert_x_pml_pinned_to_outer_shards` (new, RuntimeError) asserts low x-PML is
     owned only by rank 0 and high x-PML only by the last rank; vacuous when there is
     no x-PML. Run inside `require_distributed_adjoint_support`.
   - `Simulation._validate_trainable_parallel_fdtd` relaxed: a PML boundary with
     absorber `cpml`/`stablepml` is accepted; graded-sigma `pml`/`absorber` raise.
4. **Census**: no NotImplementedError guard added or removed (the new pinning guard
   is a RuntimeError; the relaxed rejections are ValueError). `CAPABILITY_GUARD_BUDGET`
   stays 176 and `tests/api/public/test_guard_census.py` passes unchanged.

### Files changed / added

- `witwin/maxwell/fdtd/adjoint/core.py` (added two CPML half-step helpers)
- `witwin/maxwell/fdtd/distributed/adjoint.py` (CPML replay + reverse branch +
  guard relaxation + pinning assertion)
- `witwin/maxwell/simulation.py` (validator relaxation)
- `tests/fdtd/multi_gpu/test_adjoint_parity_cpml.py` (new — CPML gates)
- `tests/fdtd/multi_gpu/test_adjoint_parity.py` (guard test updated: graded-sigma only)
- `tests/fdtd/multi_gpu/test_adjoint_replay.py` (CPML now accepted; graded rejected)
- `tests/fdtd/multi_gpu/test_guard_regressions.py` (CPML accepted; graded rejected)

### Gates (all 2-GPU, executed on 2x A6000), calibration = measurement

Scene: x-CPML dielectric, 21 x-nodes + 4-cell CPML pad/face (Nx=29, cell_count=28),
Box density region x in [-0.3,0.3] straddling the x=0 split, PointDipole source /
PointMonitor objective, absorber="cpml", STEPS=50.

| Gate | Test | Measured | Gate |
|---|---|---|---|
| (a) objective parity 1-vs-2-GPU | `test_cpml_objective_and_gradient_parity_single_vs_two_gpu` | rel **0.0** (bit-identical) | rtol 5e-5/atol 5e-6 |
| (a) gradient parity 1-vs-2-GPU | same | rel **1.18e-7** | rtol 1e-4, atol 1e-6·max |
| (b) interface FD (on-split + interior) | `test_cpml_interface_finite_difference_split_and_interior` | on-split rel **7.4e-5**, interior **5.3e-6 / 3.1e-5** | 2e-3 |
| (c) no-psi-halo confirmation | `test_cpml_no_field_halo_falsification` | baseline **1.2e-7**; no-mag-halo **0.956**; no-elec-halo **0.016** | baseline<1e-4, falsified>1e-3 |
| (d) repeat determinism (gathered grad_eps) | `test_cpml_repeat_reverse_gradient_is_bitwise_deterministic` | `torch.equal` **bitwise** | bitwise |
| (e) guard regressions | `test_adjoint_parity.py::test_guard_graded_sigma_absorber_rejected_at_prepare`, `test_adjoint_replay.py::test_checkpoint_rejects_graded_sigma_absorber`, `test_guard_regressions.py::test_require_distributed_adjoint_support_rejects_graded_sigma_absorber` | reject `pml`/`absorber`, accept `cpml`/`stablepml` | — |
| positive prepare | `test_cpml_trainable_parallel_scene_prepares` | prepares, `active_absorber_type=="cpml"` | — |
| x-CPML pinning assertion | `test_adjoint_parity_cpml.py::test_x_pml_pinning_assertion_*` (CPU) | pinned passes; interior-low-PML and non-last-high-PML raise RuntimeError | — |

### Falsifications recorded (load-bearing)

1. **No-psi-halo confirmation (gate c, the mandated load-bearing falsification).**
   With both Yee field halos active the 2-GPU CPML gradient matches single-GPU to
   rel **1.18e-7** *without any psi halo*. Monkeypatching `exchange_magnetic_adjoint`
   to a no-op drives rel to **0.956**; monkeypatching `exchange_electric_adjoint`
   drives rel to **0.016** — both >=2 orders above the 1e-3 falsification threshold.
   Confirms the two field halos carry the entire cross-interface curl(H)/curl(E)
   coupling (including the psi-derived `adj_d` folds), so a separate psi halo would
   be redundant — the S4 audit's central prediction. Encoded as
   `test_cpml_no_field_halo_falsification`.
2. **Pinning assertion is non-vacuous.** `_assert_x_pml_pinned_to_outer_shards`
   accepts the real pinned 2-shard partition but raises RuntimeError on fabricated
   partitions where an interior shard owns a low x-PML cell, or rank 0 stretches
   into the high x-PML band. Encoded as the three `test_x_pml_pinning_assertion_*`
   CPU unit tests.

### Test commands + pass counts (executed)

```bash
# Slice-1 gates + guard suites + census (28.22s)
pytest tests/fdtd/multi_gpu/test_adjoint_parity_cpml.py \
       tests/fdtd/multi_gpu/test_adjoint_parity.py \
       tests/fdtd/multi_gpu/test_adjoint_replay.py \
       tests/fdtd/multi_gpu/test_guard_regressions.py \
       tests/api/public/test_guard_census.py
# -> 52 passed

# Single-GPU adjoint regression + public API smoke (88.98s)
pytest tests/gradients/test_fdtd_adjoint_bridge.py \
       tests/gradients/test_fdtd_adjoint_materials.py \
       tests/api/public/test_public_api.py \
       tests/api/public/test_simulation_smoke.py
# -> 109 passed

# Full multi-GPU suite (regression, 57.29s)
pytest tests/fdtd/multi_gpu/
# -> 248 passed
```

### Known gaps / deferred (E3a scope)

- Slice 2 (DistributedFDTD-layer trainable guard, forward monitor gather,
  FEATURE_LIST) is E3b's deliverable.
- NCCL (one-process-per-GPU) CPML adjoint and coupled-runtime joint-solve remain
  OUT of scope, fail-closed (#13/#18 tail).
- Timing/speedup is out of scope this round (deferred-pending-exclusive-window).
- S5 tiled seeds not attempted this slice.

## Slice 2 — defense-in-depth trainable guard + forward monitor gather (E3b) — DELIVERED

### Delivered items

1. **DistributedFDTD-layer trainable guard, defense in depth (closing the
   self-reported 00 §02 gap).** The pre-existing solver-layer guard
   (`_unsupported_distributed_trainable_tensors`, S3) already rejected trainable
   geometry / material-perturbation / circuit / RF-port channels at construction.
   The remaining hole after E3a: a trainable `Box` density resolving to a
   graded-sigma absorber constructed the `DistributedFDTD` coordinator without
   raising (only the reverse-time `require_distributed_adjoint_support` and the
   public `Simulation` validator caught it). E3b adds a construction-time check in
   `_validate_static_capabilities`: if the scene carries a trainable density and
   the resolved active absorber (`absorber_type` when `boundary.uses_kind("pml")`,
   else `"none"`) is graded-sigma (`"pml"`/`"absorber"`), it raises `ValueError`
   (substring `"graded-sigma"`) before any hardware allocation. CPML/stable-PML and
   open/PEC trainable-density scenes still construct. Both layers are tested: the
   public `Simulation.fdtd(..., parallel=...).prepare()` boundary and the direct
   `DistributedFDTD(...)` constructor.
2. **Forward distributed monitor gather — seam ownership documented + falsified.**
   The forward one-process monitor gather (`fdtd/distributed/monitor_merge.py`,
   `merge_sharded_monitor_payloads`) was delivered in an earlier forward slice and
   is exercised end-to-end by `test_monitor_numerical.py` (1-vs-2-GPU parity for a
   spanning y-normal plane, a spanning z-normal flux, an off-split x-normal plane,
   a split x-normal flux, and a mode monitor). E3b derives/documents the **seam
   ownership rule** and adds the mandated double-count falsification. Seam rule:
   for a y/z-normal plane tiled along the x split, each global x index is owned by
   exactly one shard (its `owned_global_slice`); every shard contributes only its
   owned strip (halo/ghost x cells cropped by `owned_local_slice`), so the seam
   sample between two shards is assembled exactly once and the flux quadrature —
   recomputed on the result device from the merged owned components with
   cell-center weights derived from the reassembled global coordinates, then
   cropped to the physical bounds — never double-counts a seam cell weight.

### Files changed / added

- `witwin/maxwell/fdtd/distributed/solver.py` (construction-time trainable-density
  + graded-sigma-absorber defense-in-depth guard in `_validate_static_capabilities`)
- `tests/fdtd/multi_gpu/test_guard_regressions.py` (new: both-layer graded-sigma
  rejection test + CPML positive-control construction test)
- `tests/fdtd/multi_gpu/test_monitor_merge_ownership.py` (pre-existing file;
  E3b added the `_widen_owned_x` helper and the seam double-count falsification
  `test_seam_double_count_is_rejected_by_the_owned_overlap_guard`, ~86 insertions,
  no deletions)
- `FEATURE_LIST.md` (additive Track E3 subsection: CPML-trainable adjoint,
  defense-in-depth guard, forward monitor gather + seam rule)
- `docs/assessments/e3-distributed-adjoint-acceptance-2026-07-19.md` (this section)

### Gates / tests (executed on 2x A6000)

| Gate | Test | Result |
|---|---|---|
| trainable density + graded-sigma rejected at BOTH layers | `test_guard_regressions.py::test_solver_trainable_density_with_graded_sigma_absorber_rejected_at_construction[pml,absorber]` | 2 passed (public prepare + direct construct both raise `graded-sigma`) |
| trainable density + CPML/stable-PML constructs (positive control) | `test_guard_regressions.py::test_solver_trainable_density_with_cpml_constructs[cpml,stablepml]` | 2 passed |
| unsupported-channel defense in depth (pre-existing, still green) | `test_guard_regressions.py::test_solver_trainable_guard_covers_circuit_parameter_channel` | passed |
| 1-vs-2-GPU monitor parity, spanning y-plane | `test_monitor_numerical.py::test_y_normal_single_component_plane_monitor_matches_one_gpu` | passed (rtol 5e-5 / atol 5e-6) |
| 1-vs-2-GPU flux parity, spanning z-flux | `test_monitor_numerical.py::test_z_normal_flux_monitor_matches_fields_and_integrated_flux` | passed (fields 5e-5; flux rel <=1e-3) |
| 1-vs-2-GPU parity, non-spanning off-split x-plane | `test_monitor_numerical.py::test_off_split_x_normal_plane_monitor_matches_one_gpu` | passed |
| 1-vs-2-GPU parity, split x-normal flux | `test_monitor_numerical.py::test_split_x_normal_flux_monitor_matches_fields_and_integrated_flux` | passed |
| seam double-count rejected (falsification test) | `test_monitor_merge_ownership.py::test_seam_double_count_is_rejected_by_the_owned_overlap_guard` | passed |
| non-gathered monitors stay fail-closed | `test_guard_regressions.py::test_material_monitors_are_explicitly_rejected_before_hardware_prepare` + solver static guards (closed-surface/diffraction/flux-time/non-point-time/breakdown) | passed |
| capability-guard census unchanged | `tests/api/public/test_guard_census.py` | passed (budget 176) |

### Falsifications recorded (load-bearing)

1. **DistributedFDTD graded-sigma trainable guard is load-bearing.** Disabled the
   new check (`if False and resolved_absorber in ("pml","absorber")`) in
   `_validate_static_capabilities`; reran
   `test_solver_trainable_density_with_graded_sigma_absorber_rejected_at_construction`.
   Layer 1 (public `Simulation`) still raised, but layer 2 (direct
   `DistributedFDTD(...)`) failed with `DID NOT RAISE ValueError` (2 failed).
   Restored the guard -> green. Confirms the constructor guard, not only the public
   validator, is what closes the direct-construction hole.
2. **Seam overlap guard is load-bearing.** Neutralized the overlap branch in
   `monitor_merge._stitch_owned_component` (kept only the `gap` case); reran
   `test_seam_double_count_is_rejected_by_the_owned_overlap_guard`. The test went
   red — the specific `"overlap"` diagnosis vanished and the double-counted strip
   was instead caught one layer later by the strictly-increasing-coordinate guard
   (`"owned global x coordinates are not strictly increasing"`), demonstrating both
   that the overlap guard is the primary seam-double-count defense and that a
   second line exists. Restored -> green.

### Test commands + pass counts (executed)

```bash
# New slice-2 tests + guard suites + census
pytest tests/fdtd/multi_gpu/test_monitor_merge_ownership.py \
       tests/fdtd/multi_gpu/test_guard_regressions.py \
       tests/api/public/test_guard_census.py
# -> 34 passed

# Full multi-GPU suite (regression, includes monitor parity + slice-1 CPML gates) + census
pytest tests/fdtd/multi_gpu/ tests/api/public/test_guard_census.py
# -> 256 passed

# Single-GPU adjoint/gradient regression + public API smoke
pytest tests/gradients/ tests/api/public/test_public_api.py \
       tests/api/public/test_simulation_smoke.py
# -> 229 passed, 3 failed. All 3 failures are in tests/gradients/test_fdfd_adjoint.py
#    (the user-deferred FDFD known-failing set, unrelated to this track); every
#    FDTD adjoint/gradient and public-API test passed.
```

### Known gaps / deferred (E3b scope)

- Forward monitor gather is single-process (`transport="cuda_p2p"`). NCCL
  (one-process-per-GPU) monitor gather, CPML adjoint, and coupled-runtime
  (circuit/network/wire) joint-solve stay OUT of scope and fail-closed — the
  remaining blueprint #13/#18 tail.
- Non-gathered monitor classes (closed-surface, diffraction, time-domain flux,
  non-point time, permittivity/medium material, breakdown/ESD observers) remain
  fail-closed; the capability-guard census budget is unchanged (176) — no guard
  was widened silently.
- Timing/speedup deferred-pending-exclusive-window.
- S5 tiled adjoint seeds not attempted (slices 1-2 delivered and audited first).
