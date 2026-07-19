# Track D13 — deterministic dynamic dielectric breakdown: D1 (runtime) acceptance

Date: 2026-07-19
Stage: D1 (runtime) of track `d13-breakdown` (plan 13 Phase 4, deterministic
field-duration/latching dynamic breakdown).
Worktree: `.worktrees/wd-d13-breakdown` (branch `fable/breakdown`).
Environment: conda env `maxwell`, `CUDA_VISIBLE_DEVICES=1`, `PYTHONPATH=<worktree>`.

## Delivered items

Public API and data types:
- `DielectricBreakdown(critical_field, post_breakdown_conductivity, minimum_duration=0.0, model="field_duration", state="latching", recovery=None, damage_parameters=None, ramp_time=None, default_ramp_steps=10)` dataclass in `media.py`, composed into `Material(breakdown=...)`. v1 accepts exactly `model="field_duration"` + `state="latching"`; other model/state and the reserved `recovery`/`damage_parameters` fail closed at construction.
- `BreakdownEvent` and `BreakdownResultData` typed containers in new `witwin/maxwell/breakdown.py`, exported from the package root; state codes (`intact=0`, `conducting=2`, reserved `recovering=1`/`failed=3`).

Compiler:
- `witwin/maxwell/compiler/breakdown.py`: `compile_breakdown_layout(scene)` stamps per-node breakdown parameters on the `(Nx,Ny,Nz)` material grid (staircase occupancy threshold `>=0.5`, last-writer-wins), and `scene_has_breakdown(scene)`.

Runtime:
- `witwin/maxwell/fdtd/runtime/breakdown.py`: compiled capable-node and capable-edge index sets; per-cell state machine (energy-consistent cell-center `|E|` colocation, contiguous-exceedance timer, latching trigger); linear conductivity ramp; in-place semi-implicit decay/curl scatter on capable edges reconstructed exactly from stored base coefficients (recovered external PML/PEC factor); dedicated breakdown-dissipation energy channel `∫ σ_breakdown·|E|² dV dt` accumulated on the conduction edges and scattered to nodes; bounded preallocated per-node event buffer (capacity = capable-cell count) with a defensive overflow hard-error; host transfer only at run end. All per-step work is GPU-resident with **no per-step host synchronization**.
- Step-loop integration in `fdtd/runtime/stepping.py`: `initialize_breakdown_runtime` after `build_update_coefficients`; `advance_breakdown_state(solver, n)` once per step after the full E update (sources + PEC/Mur/DFT tail); CUDA-graph capture forced off when breakdown is enabled.
- Result assembly in `simulation.py` (`_run_fdtd_from_solver`) via `finalize_breakdown_data`; `Result.breakdown` / `Result.breakdown_events` accessors in `result.py`.

Prepare-time guards:
- `Material` construction rejects breakdown on PEC / anisotropic-tensor / dispersive / instantaneous-nonlinear / time-modulated / 2D-sheet media.
- `initialize_breakdown_runtime` rejects complex (Bloch) fields and scenes mixing breakdown with dispersive / nonlinear / full-anisotropic / modulated / gyromagnetic media.
- `Simulation._validate_breakdown_support` rejects FDFD + breakdown and trainable + breakdown (non-differentiable hard switch; smooth surrogate deferred).
- `DistributedFDTD._validate_static_capabilities` rejects multi-GPU + breakdown.
- Prepare-time accuracy warning when `0.5·σ·dt/eps > 10` (PEC-like reflector).

## Files added / changed

Added: `witwin/maxwell/breakdown.py`, `witwin/maxwell/compiler/breakdown.py`, `witwin/maxwell/fdtd/runtime/breakdown.py`, this doc.
Changed: `witwin/maxwell/media.py`, `witwin/maxwell/__init__.py`, `witwin/maxwell/result.py`, `witwin/maxwell/simulation.py`, `witwin/maxwell/fdtd/runtime/stepping.py`, `witwin/maxwell/fdtd/distributed/solver.py`, `FEATURE_LIST.md`, `docs/reference/fdtd-capability-guard-census.md`, `tests/api/public/test_guard_census.py`.

## Verification (exact commands)

All commands prefixed with:
```
export CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=<worktree>
export CUDA_VISIBLE_DEVICES=1
```

Test suites (all passed):
- `pytest tests/api/public/test_public_api.py tests/api/public/test_simulation_smoke.py` → 27 passed
- `pytest tests/api/public/test_guard_census.py` → 3 passed (budget raised 144 → 159, reconciled)
- `pytest tests/materials/compiler/test_material_compiler.py tests/core/scene/test_scene.py` → 78 passed
- `pytest tests/materials/conductive/` → 9 passed (shared coefficient path intact)
- `pytest tests/boundaries/cpml/test_fdtd_cpml.py tests/monitors/observers/test_fdtd_observers.py` → 14 passed
- `pytest tests/validation/benchmark/test_media_validation_coverage.py tests/gradients/test_fdtd_adjoint_bridge.py` → 65 passed

Behavioral evidence (throwaway scripts under `scratch/`, not committed):
- Triggering plane-wave breakdown slab: 1212–1745 events, `total_dissipated_energy > 0`, final-state mask populated, `Result.breakdown_events` non-empty.
- No-breakdown-material scene: `Result.breakdown is None` (zero-cost path preserved).
- Below-threshold-with-descriptor parity: `torch.equal` on all three final E fields between with-descriptor (critical=1e12, never triggers) and without-descriptor runs → **bitwise identical** (max abs diff 0.0).
- Event-log determinism: two identical runs → identical `(step, cell_index, field_before)` logs, identical total dissipation and state masks, and the log is sorted by `(step, cell_index)`.
- Zero base conductivity (`sigma_e=0`, `conductive_enabled` global False) still triggers and dissipates.
- Guard rejects verified: unsupported `model`/`state`/`recovery`/`damage_parameters` (construction), FDFD+breakdown, anisotropic+breakdown (material), trainable+breakdown (`_validate_breakdown_support`).

## Falsifications recorded

1. **Trigger/threshold comparison is load-bearing.** In the 1743-event determinism scene, temporarily multiplied the trigger threshold by `1e9` (`exceeding = magnitude >= runtime["critical"] * 1e9`). Observed: events dropped 1743 → 0, final-state all intact, zero dissipation. Restored → machinery green again. Proves triggering is driven by the field/threshold comparison, not an unconditional path.
2. **Below-threshold parity gate probe.** Temporarily forced the coefficient scatter to overwrite unconditionally (`active = ones_like(...)` instead of `edge_extra > 0`). Below-threshold parity remained bitwise identical — the intact-edge decay/curl reconstruction from the stored base coefficients round-trips bit-exactly for these edges, so parity is robust by exact reconstruction rather than only by the write-gate. Restored to the gated form (kept as a guarantee and to skip work on inactive edges).

## Known gaps / deferred (for D2 and later phases)

- D1 does **not** author the formal `tests/breakdown/` suite; D2 owns the manufactured trigger/no-trigger golden tests, dt-convergence, energy-closure (PEC box), and the falsification-backed regression tests.
- `deposited_energy_at_trigger` is defined as the local stored electric field energy `0.5·eps·|E|²·cell_volume` at the trigger step (documented in `BreakdownEvent`); the cumulative breakdown dissipation at trigger is ~0 under latching.
- Per-cell `dissipated_energy` is edge-consistent (edge energy split equally among capable node endpoints so per-node sums equal the edge total); `total_dissipated_energy` is the authoritative closure scalar. Energy-balance tolerance to be quantified by D2's closed-box test.
- Ramp fraction uses `elapsed = (n − trigger_step)·dt` so the trigger step itself adds no conductivity (ramp starts the following step); documented.
- Event-buffer capacity equals the capable-cell count (exact latching upper bound), so overflow is structurally impossible; the overflow hard-error is retained defensively and D2 can shrink `runtime["capacity"]` to exercise it.
- Breakdown runs on the eager step path (no CUDA-graph capture); performance overhead of the machinery is not measured this round (single-GPU correctness only).
- Multi-GPU, trainable smooth surrogate, recovery/damage state machines, and coefficient-path compositions (dispersive/nonlinear/anisotropic/modulated/gyromagnetic) are fail-closed and deferred.

---

# Stage D2 (validation) acceptance

Date: 2026-07-19. Stage: D2 (validation) of track `d13-breakdown`.
Environment: conda env `maxwell`, `CUDA_VISIBLE_DEVICES=1`,
`PYTHONPATH=<worktree>`, `CUDA_HOME=.../nvidia/cu13`.

## Delivered items

New suite `tests/breakdown/` (20 tests, all passing). Two test styles:

* **Direct-drive** tests prepare a solver (`Simulation.fdtd(...).prepare().solver`,
  which compiles `solver.breakdown_runtime`), then prescribe the electric field
  by hand (`set_uniform_ez`, `Ex = Ey = 0` so the energy-consistent colocation
  gives node `|E| = Ez`) and call `advance_breakdown_state` step by step. Field
  evolution is exactly known, so trigger step, contiguous-reset behaviour, and
  dissipated energy are analytically predictable and independent of the Maxwell
  update.
* **Integration** tests run a real FDTD solve and inspect `Result.breakdown`.

Files added: `tests/breakdown/__init__.py`, `tests/breakdown/_common.py`,
`tests/breakdown/test_breakdown_state_machine.py`,
`tests/breakdown/test_breakdown_energy.py`,
`tests/breakdown/test_breakdown_parity.py`,
`tests/breakdown/test_breakdown_events.py`,
`tests/breakdown/test_breakdown_guards.py`.
Files changed: `FEATURE_LIST.md` (validation-coverage bullet in the existing
`d13-breakdown` block), this doc.

No production module was modified in D2; the runtime under test is exactly D1's
committed `breakdown.py` (verified `git diff --stat` clean after the falsification
sweep below).

## Test inventory (all pass; env as above)

`pytest tests/breakdown/ -q` -> **20 passed**. By file:

- `test_breakdown_state_machine.py` (5): golden trigger step
  `trigger_step = ceil(minimum_duration/dt) - 1` for K in {0,2,5} with
  `minimum_duration=(K+0.5)*dt` (rounding-robust); immediate trigger at
  `minimum_duration=0`; contiguous-timer reset delays a `[E0,E0,below,E0,E0,E0]`
  train to step 5 (vs 2 uninterrupted); no-trigger when held below threshold;
  trigger-time dt convergence.
- `test_breakdown_energy.py` (3): dissipation closure vs analytic
  `E0^2 * sigma_post * V_ez * sum_n frac(n) * dt` (`V_ez` recomputed from the
  layout node mask + scene dual/primal grid arrays, independent of the runtime
  accumulator), per-node partition == total, zero dissipation before trigger,
  monotonic non-decreasing cumulative dissipation after trigger.
- `test_breakdown_parity.py` (3): below-threshold descriptor six-field
  (Ex,Ey,Ez,Hx,Hy,Hz last-step) bitwise identity vs the descriptor-free scene;
  breakdown-free scene -> `Result.breakdown is None`; triggering control that
  DOES perturb the fields (guards against a vacuous parity gate).
- `test_breakdown_events.py` (3): triggering run populates a typed event log
  ordered by `(step, cell_index)` with a matching conducting final-state mask;
  two identical runs -> identical logs/masks/dissipation (determinism);
  high-threshold run -> present-but-empty breakdown result, all intact.
- `test_breakdown_guards.py` (6): unsupported `model` / `state` / reserved
  `recovery` / `damage_parameters` rejected at construction; trainable+breakdown
  rejected at `prepare()`; multi-GPU static-capability guard rejects breakdown
  (exercised directly on the scene -- only one physical GPU in this environment);
  event-buffer overflow is a hard `RuntimeError` when capacity is shrunk below the
  trigger count.

## Reproducible metrics

dt-convergence (`test_trigger_time_converges_with_dt`; regenerate with the
committed probe `python docs/assessments/d13-breakdown-probes/report_numbers.py`,
which reuses the `tests/breakdown` builders so its output is the numbers the tests
assert on), linear ramp `E(t)=5e12*t`, `critical=50`, `t* = 1.0000e-11 s`,
`minimum_duration=0`:

| h (m)  | dt (s)     | trigger_step | trigger_time (s) | error (s)  | band ceiling = dt (s) |
|--------|------------|--------------|------------------|------------|-----------------------|
| 0.0100 | 1.9258e-11 | 1            | 1.9258e-11       | 9.2583e-12 | 1.9258e-11            |
| 0.0050 | 9.6292e-12 | 2            | 1.9258e-11       | 9.2583e-12 | 9.6292e-12            |
| 0.0025 | 4.8146e-12 | 3            | 1.4444e-11       | 4.4437e-12 | 4.8146e-12            |

The discrete first-crossing error stays in `[0, dt)` (staircase band); the band
ceiling `dt` tightens 1.93e-11 -> 9.63e-12 -> 4.81e-12 as the grid refines.
Resolutions are chosen in the grid-CFL-limited regime (coarser grids clamp to the
source points-per-period ceiling `dt = 3.33e-11 s`, which would not vary with h).

Energy closure (`test_dissipated_energy_matches_analytic_closed_form`),
`post_sigma=4`, `default_ramp_steps=10`, `E0=100`, 30 steps:
runtime total `8.963733e-08 J` vs analytic `8.963733e-08 J`, relative error
`2.94e-08`; per-node channel sums to the same total. Test tolerance `rel_tol=2e-4`.

## Falsifications performed (scratch edits to D1's runtime; reverted)

Each broke `witwin/maxwell/fdtd/runtime/breakdown.py`, ran the relevant node,
observed red, restored from `scratch/breakdown_bak.py`, and reconfirmed
`git diff --stat` clean + `tests/breakdown/ -> 20 passed`.

1. **Golden trigger step.** Replaced `trigger = intact & exceeding & (new_timer
   >= minimum_duration)` with `trigger = intact & exceeding` (ignore duration).
   `test_trigger_at_golden_step...` and `test_contiguous_reset...` went red -- all
   cells triggered at step 0 (observed `trigger_step.max() == 0` vs expected 2/5).
2. **Below-threshold bitwise parity.** Perturbed the intact-edge decay
   reconstruction (`data["base_decay"] * 1.0000001`).
   `test_below_threshold_descriptor_is_bitwise_identical` went red (six-field
   `torch.equal` broke).
3. **Energy closure.** Scaled the dissipation integrand by `1.05`.
   `test_dissipated_energy_matches_analytic_closed_form` went red (runtime total
   diverged from the analytic ground truth beyond `rel_tol=2e-4`).

## Exact commands

```
export CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=<worktree>
export CUDA_VISIBLE_DEVICES=1
conda run -n maxwell --no-capture-output python -m pytest tests/breakdown/ -q
conda run -n maxwell --no-capture-output python -m pytest \
  tests/api/public/test_public_api.py tests/api/public/test_simulation_smoke.py \
  tests/api/public/test_guard_census.py -q
```

Adjacent suites (regression, this stage): `tests/breakdown/ -> 20 passed`;
`test_public_api.py + test_simulation_smoke.py + test_guard_census.py -> 30 passed`.

## Known gaps / deferred (D2)

- **Global boundary-flux energy balance — decision-6 gate, explicitly deferred
  (substitution, not silent).** The brief's decision-6 closed-PEC-box balance
  (`injected = stored-EM change + material dissipation + breakdown dissipation`,
  radiation zero) is **not** asserted on a live run, because the framework exposes
  no observer for two of its terms. Attempt made (2026-07-19,
  `scratch/energy_probe.py`, not committed): a tiny closed PEC-box scene
  (`BoundarySpec.pec()`, `0.24 m` box, `0.02 m` grid) with a CW point dipole and a
  breakdown dielectric was run for 60 steps; it triggers and dissipates
  (`Result.breakdown.total_dissipated_energy = 1.95e-2 J`). Inspecting `Result`
  shows the only energy/power accessor is `power_loss` — a **frequency-domain
  phasor** steady-state loss postprocessor (peak-phasor `sigma*|E|^2`), not a
  transient audit; there is **no injected-source-energy accessor and no running
  stored-EM-energy time series** (`has injected/source energy accessor: False`; a
  final field snapshot is available but not a time-integrated injected work or a
  per-step EM-energy audit). A clean global time-domain accounting therefore cannot
  be assembled from existing observers. **Substitution kept:** the breakdown
  dissipation channel `integral(sigma_breakdown*|E|^2 dV dt)` is validated instead
  against an analytic closed form on a prescribed-field cell
  (`test_dissipated_energy_matches_analytic_closed_form`, rel err `2.94e-08` at
  `rel_tol=2e-4`) plus per-node/total partition consistency — the load-bearing part
  of decision 6. The full closed-box global balance is deferred until an
  injected-source-energy or running stored-EM-energy monitor lands.
- **Multi-GPU guard** is exercised by invoking
  `DistributedFDTD._validate_static_capabilities` directly on the breakdown scene
  (only one physical GPU is visible under `CUDA_VISIBLE_DEVICES=1`); an end-to-end
  two-GPU rejection is deferred to an exclusive multi-GPU window.
- **Performance / zero-overhead timing** for the no-breakdown path is not measured
  (correctness-only shared-GPU window); deferred pending an exclusive window.

---

# Stage D3 (audit remediation) acceptance

Date: 2026-07-19. Stage: D3 (post-audit remediation) of track `d13-breakdown`.
Environment: conda env `maxwell`, `CUDA_VISIBLE_DEVICES=1`,
`PYTHONPATH=<worktree>`, `CUDA_HOME=.../nvidia/cu13`.

Supervisor-selected audit fixes applied to this worktree.

## Fix 1 — zero-impact-when-unused (merge-blocking priority)

`initialize_breakdown_runtime` (`fdtd/runtime/breakdown.py`) previously called
`compile_breakdown_layout` on **every** FDTD prepare, allocating seven full-grid
`(Nx,Ny,Nz)` tensors (~0.9 GB transient at 27M cells) even for scenes with no
breakdown material. Added a cheap structure pre-scan
(`scene_has_breakdown(solver.scene)`) that short-circuits **before** any
allocation/compilation when no material carries a breakdown descriptor.

Tests added (`tests/breakdown/test_breakdown_parity.py`):
- `test_plain_scene_never_compiles_breakdown_layout` — monkeypatches
  `compile_breakdown_layout` with a raising stub; a breakdown-free prepare+run must
  still succeed (proves the compiler is never reached).
- `test_plain_scene_run_is_bitwise_deterministic` — the headline field-level
  no-breakdown parity gate: two identical plain-scene runs reproduce all six Yee
  fields bit for bit with the breakdown module imported and its prepare-hook active.

Falsification (recorded): disabled the pre-scan short-circuit (unconditional
`compile_breakdown_layout`); `test_plain_scene_never_compiles_breakdown_layout`
went **red** (`AssertionError: compile_breakdown_layout must not run for a
breakdown-free scene`). Restored → green.

## Fix 2 — structure-overlap phantom capability

`compile_breakdown_layout` (`compiler/breakdown.py`) previously `continue`d past
non-breakdown structures, so a later (higher-priority) non-breakdown structure that
overwrote a breakdown region left stale breakdown capability behind. Now the loop
follows the material compiler's last-writer-wins: iterating the priority-sorted
bulk structures, a non-breakdown structure strips the breakdown mask and resets all
per-node parameters (`critical_field`, `minimum_duration`, `post_conductivity`,
`ramp_time_explicit`, `ramp_steps`, `material_id = -1`) on the cells it covers.

Tests added (`tests/breakdown/test_breakdown_compiler.py`, 3):
`test_later_nonbreakdown_structure_clears_overlapped_cells` (subset/strict-shrink
invariants + cleared params on removed cells),
`test_fully_overwritten_breakdown_box_never_triggers` (fully-covered box → empty
capable set, `solver.breakdown_runtime is None`),
`test_partial_overwrite_only_survivors_trigger` (drive supra-critical field; only
the surviving cells fire, count matches the shrunk mask).

Falsification (recorded): reverted the clearing branch to a bare `continue`; all 3
compiler tests went **red** (overwritten box kept full capability, `node_count`
343 == reference instead of the shrunk survivor count). Restored → 3 pass.

## Fix 3 — global energy balance

Attempted the true closed-PEC-box global balance; existing observers cannot supply
the injected-source-energy or running stored-EM-energy terms (see the decision-6
deferred-gate note in the D2 "Known gaps" section above — substitution stated
explicitly, not silent). Analytic dissipation closure retained as the load-bearing
check.

## Fix 4 — acceptance-doc reproduction pointer

Committed the dt-convergence / energy-closure reproduction probe at
`docs/assessments/d13-breakdown-probes/report_numbers.py` (`git add -f`; `docs/`
is gitignored wholesale). The D2 table pointer now references the committed path
instead of the uncommitted `scratch/report_numbers.py`. The probe reuses the
`tests/breakdown` builders so its printed numbers are exactly the asserted ones.

## Fix 5 — cosmetic

- Removed the always-true `if trigger is not None:` dead conditional in
  `advance_breakdown_state` (the mask is always a tensor); dedented the body.
- `test_below_threshold_descriptor_is_bitwise_identical` now asserts
  `total_dissipated_energy == 0.0` exactly (and the per-node channel sums to 0.0),
  replacing the `int(...) == 0` truncation that silently accepted any value in
  `[0, 1)`.

## Test inventory (D3, env as above)

- `pytest tests/breakdown -q` → **25 passed** (20 prior + 2 new parity + 3 new
  compiler).
- `pytest tests/api/public/test_guard_census.py` → passed.
- `pytest tests/api/public/test_simulation_smoke.py` → passed.

Falsifications this stage: Fix 1(a) zero-cost gate (red→restore→green); Fix 2
phantom-capability clearing (3 red→restore→3 green).
