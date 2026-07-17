# Joint-solve adjoint — Phase 7 slice S4 (CPML) progress + audit (2026-07-17)

Branch `codex/cpml-trainable-s4`, worktree `.worktrees/r4-cpml`. Follows S3
(`fdtd-joint-solve-adjoint-s3-progress-2026-07-17.md`). Blueprint:
`docs/plans/next-functional-2026-07/02-phase-7-8-blueprint-2026-07-16.md` (S4 + risk 1).

S4 target: CPML-trainable distributed joint-solve adjoint, then S5 tiled-monitor seeds.
This round delivers the mandated **step-1 audit** (written before any coding) and the
**step-2 phase-split refactor** (gated single-GPU bit-identical). Steps 3–5 (the
distributed CPML bridge + parity gates, and S5) are deferred with the wall/next-steps
recorded below — the audit removes two of the three S3-recorded blockers, so the
deferred work is smaller and better-specified than S3 left it.

## PART A — Step-1 audit: CPML reverse internal-face semantics (written before coding)

**Question (per mission step 1 + blueprint risk 1):** do the fused CPML reverse kernels
(`cpml_correction_active` / `resolve_electric_cell_status` at tensor x-faces) mishandle a
shard's *internal* x-face when launched per-shard, requiring bounded-x CPML reverse
CUDA variants and transposed psi halos — or are they interface-correct like the standard
kernels?

**Finding: the existing fused CPML reverse kernels ARE interface-correct per-shard. No
new CUDA kernels are required, and no transposed psi_e/psi_h halos are required.** The S4
distributed CPML reverse can reuse the S1/S3 transposed field transports
(`exchange_magnetic_adjoint` on Hy/Hz, `exchange_electric_adjoint` on Ey/Ez) unchanged.

### Evidence (file:line)

1. **Structural equivalence of the CPML reverse to the standard reverse.**
   `_reverse_step_cpml_native_core` (`fdtd/adjoint/native.py`) is two phases:
   - Phase A = 3 `_reverse_electric_cpml_e{x,y,z}` kernels + 6 `_accumulate_backward_diff_*`
     folds. The electric kernel *assigns* pre-step E (decay pullback), pre-step psi_e, eps
     gradient, and the curl(H)-derivative seeds `adj_d_*`; the backward-diff folds scatter
     those seeds into the mid-step H adjoint `ctx.magnetic_output_adjoint`.
   - Phase B = 3 `_reverse_magnetic_cpml_h{x,y,z}` kernels + 6 `_accumulate_forward_diff_*`
     folds. The magnetic kernel reads the mid-H adjoint, *assigns* pre-step H/psi_h, and
     emits the curl(E)-derivative seeds; the forward-diff folds scatter those into pre-step E.
   So `adj_h_mid = (adj_h_post seed, `reverse_common.py:94-98`) + Phase-A curl(H) folds`, and
   `pre["E"] = Phase-A decay + Phase-B curl(E) folds` — the *same* two quantities the standard
   core produces, just organized as (kernel + fold) instead of one fused kernel. The
   cross-interface coupling therefore rides the *same* two field planes: mid-H adjoint (Hy/Hz)
   and pre-step E adjoint (Ey/Ez). `adj_h_mid["Hx"]` receives only y/z folds (local in x), and
   Hx has no forward x-halo, consistent with S1/S3.

2. **The diff-accumulate folds write the interface planes.** `accumulate_diff_adjoint_kernel`
   (`cuda/kernels/adjoint.cu:1139-1192`) runs over every field index including the ghost cell
   (backward, H side) and ghost node (forward, E side): a right shard's first owned E node
   (interior, active) seeds `adj_d`, and `backward_diff_x` scatters it into the right shard's
   ghost H cell 0 — exactly the plane `exchange_magnetic_adjoint` ships left. Mirror for
   `forward_diff_x` into the left shard's ghost E node, shipped right by
   `exchange_electric_adjoint`. This is the identical mechanism S3 verified for standard.

3. **Internal x-face gating is correct.** The CPML magnetic reverse kernel
   (`adjoint.cu:1971-2025`) is *ungated* (computes for all cells); with pass-through
   x-coefficients (b=1, c=0, inv_kappa=1 outside PML) it reduces to the standard magnetic
   reverse. The CPML electric kernel (`adjoint.cu:1404-1466`) gates via
   `resolve_electric_cell_status` on the two transverse axes plus the x-axis mode; at an
   internal x-face the shard-local boundary mode is `NONE` (set in the forward shard-local
   compile, `solver.py:217-221`), so `status.inactive` → pure pass-through and `adj_d=0` at
   the *ghost* node. The *owner's* first interior node is active and supplies the cross term
   through its `adj_d` fold (point 2). Verified the forward mirror: `cpml_correction_active`
   (`electric.cu:1479-1486`) returns false at a `NONE` face, so the forward applies no psi
   correction there — the interface is treated as plain interior curl on both passes.

4. **The decisive invariant — no x-CPML straddles an internal interface.** The partition
   excludes PML from the split: `physical_cell_count = cell_count - low_pml_cells -
   high_pml_cells` and `_balanced_intervals(physical_cell_count, ...)` partition only physical
   cells (`fdtd_parallel.py:350-360`); `_build_shard_layout` pins all low x-PML to rank 0 and
   all high x-PML to the last rank, so interior shards own zero x-PML
   (`fdtd_parallel.py:614-666`). The distributed solver feeds the real x-CPML width in
   (`solver.py:553-554`, `pml_thickness_for_face("x", ...)`). Therefore x-direction psi
   (psi_e*_x / psi_h*_x, active only in x-PML) is identically inactive at every internal
   interface, and transverse psi (y/z direction) is local in x (its recursion reads only
   same-x differences). **The only cross-interface psi contribution would arise if an x-PML
   region were split across a shard boundary; the partition forbids that by construction.**

### Consequences for S4 (and correction to the S3-recorded blockers)

- **S3 blocker #1 (phase-split) — real, addressed this round** (PART B).
- **S3 blocker #2 (psi_e/psi_h transposed halos) — REFUTED by the audit.** No psi halos are
  needed; the psi cross-interface contribution flows through `adj_d` → the existing field
  halos, and x-psi is inactive at interfaces. This must still be *confirmed* by the S4
  distributed CPML parity/FD gate when the bridge lands (the audit predicts the existing
  Hy/Hz + Ey/Ez halos suffice; a passing interface-FD case on an x-CPML dielectric with a
  cross-shard objective is the empirical proof).
- **S3 blocker #3 (internal-face gating) — VERIFIED correct** (points 3–4).
- **Bounded-x CPML reverse CUDA variants (the one case the blueprint flagged as possibly
  needing CUDA work) are NOT required.** The audit demanded no CUDA changes.
- **Hard guard S4 must add (defensive, cheap):** reject any config where an x-CPML region
  would be owned by a non-outer shard, i.e. assert the partition's PML-pinning holds before
  trusting the per-shard CPML reverse. This is guaranteed by construction today but should be
  asserted so a future partition change cannot silently break the invariant.

## PART B — Step-2: phase-split of `_reverse_step_cpml_native_core` (LANDED, gated)

`fdtd/adjoint/native.py`: factored the CPML core into two pure helpers,
`reverse_cpml_phase_electric(...)` (3 electric kernels + 6 backward-diff folds) and
`reverse_cpml_phase_magnetic(...)` (3 magnetic kernels + 6 forward-diff folds), mirroring
the S1 standard-core split. `_reverse_step_cpml_native_core` now calls the two in sequence.
This is a line-for-line move of the original launch order/args, so the single-GPU result is
bit-for-bit identical, and it exposes the exact seam where the distributed reverse will
insert `exchange_magnetic_adjoint` (between A and B) and `exchange_electric_adjoint` (after
B). `_reverse_step_conductive_native` / `_reverse_step_kerr_native` /
`_reverse_cpml_magnetic_phase_native` were left untouched (their inline blocks differ per
medium; only the linear CPML core is on the S4 distributed path).

### Gate evidence (2× A6000, CUDA_VISIBLE_DEVICES=0)

- **Bit-identity fingerprint (`torch.equal`, all 24 reverse outputs).** A deterministic
  driver runs the CPML core with the suite's `_fake_cpml_reverse_solver` on CUDA, seeded
  random forward/adjoint states and an explicit mid-H (assign-semantics kernels + ordered
  `add_` folds, no grid_sample → bitwise reproducible). Pre- vs post-refactor: BIT-IDENTICAL
  across pre-step E/H, all 12 psi, adj_h_mid Hx/Hy/Hz, and grad_eps ex/ey/ez.
- **Falsification of the fingerprint gate.** Monkeypatching one Phase-B fold
  (`_accumulate_forward_diff_y`) to a no-op makes the fingerprint DIFFERENT (`pre::Ex`,
  `pre::Ez` diverge, max|diff|~0.3) — the gate is load-bearing, not a tautology. Restored.
- **End-to-end single-GPU adjoint suites green:**
  `tests/gradients/test_fdtd_adjoint_bridge.py` + `test_fdtd_adjoint_materials.py` = **80
  passed** (these drive the native CPML/conductive/kerr reverse cores end-to-end through the
  public gradient bridge and FD checks). Baseline pre-refactor was equally green.

## Deferred (steps 3–5) — clean deferral with recorded next-steps

The distributed CPML bridge + parity gates (steps 3–4) and S5 tiled-monitor seeds are NOT
landed this round. CPML-trainable+parallel therefore remains rejected at prepare
(`simulation.py:_validate_trainable_parallel_fdtd`, absorber=="cpml"), and the reference-doc
support matrix is unchanged (no new user-facing capability). Rationale: the bridge is a
genuine multi-day slice whose correctness must be proven on hardware; a forced
half-implementation without passing 2-GPU parity/FD/determinism gates is worse than a clean
deferral (mission guidance). The audit + phase-split are the sound, gated slices delivered.

Precise next-steps for the S4 bridge, now de-risked by the audit:
1. **Distributed CPML replay.** `replay_distributed_segment` (`fdtd/distributed/adjoint.py`)
   currently uses the standard half-step helpers (`_forward_magnetic_fields` +
   `_forward_electric_fields_standard`), which do not advance psi. Add CPML-aware half-steps
   that thread psi_e/psi_h (the generic single-GPU replay `core._step_state:3183` already does
   this via `_update_magnetic_component`; split it into a magnetic half and an electric half
   so the two Yee halos can be inserted between them). Checkpoints already capture psi
   (`checkpoint.capture_checkpoint_state`).
2. **psi-carrying reverse loop.** Generalize the bridge's `_STANDARD_STATE_NAMES` (6 fields)
   to include the 12 psi fields for the CPML branch; the adjoint_state and pre-step adjoint
   carry psi (seeds inject only E/H — psi adjoint starts at 0). Branch `_reverse_one_step` on
   backend: CPML runs `reverse_cpml_phase_electric` → `exchange_magnetic_adjoint` (Hy/Hz) →
   `reverse_cpml_phase_magnetic` → `exchange_electric_adjoint` (Ey/Ez), then per-shard
   `_accumulate_source_term_gradients`. No psi halo (audit).
3. **Guard relaxation + validator.** Relax the CPML rejection in
   `_validate_trainable_parallel_fdtd` and `require_distributed_adjoint_support` for the
   `cpml`/`stablepml` kind ONLY; keep graded-sigma `pml`/`absorber` and
   conductive/kerr/aniso/dispersive/modulated rejected. Add the defensive PML-pinning
   assertion (PART A consequences).
4. **Gates (2-GPU):** 1-vs-2-GPU objective/gradient parity on an x-CPML dielectric Box-density
   scene at the S3 calibrated gates; interface FD with a cross-shard objective (this is the
   empirical confirmation that no psi halo is needed — falsify by no-op'ing each field halo);
   repeat determinism on the gathered grad_eps; guard regressions.
5. **S5:** tiled plane/flux/mode seed scatter (unchanged from S3's deferral).

## Census / FEATURE_LIST / support matrix
- No guards added or changed → guard census budget (133) unchanged; no reconciliation.
- No user-facing capability changed (CPML-trainable-parallel still rejected) → FEATURE_LIST
  and the reference support matrix unchanged this round.

Invocation: conda env `maxwell`, `CUDA_HOME=.../nvidia/cu13`,
`PYTHONPATH`=this worktree, single-GPU gates `CUDA_VISIBLE_DEVICES=0`.
