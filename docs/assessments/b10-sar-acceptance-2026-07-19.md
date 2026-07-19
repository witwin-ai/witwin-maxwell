# Track B (Plan 10 SAR) — B1 acceptance (Phase 0+1)

Date: 2026-07-19
Worktree: `.worktrees/wd-b10-sar` (branch `fable/sar`)
Environment: conda env `maxwell`, `CUDA_VISIBLE_DEVICES=0`.

Reproduce commands (every command below assumes this prelude):

```bash
export CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=/home/xingyu/code/witwin/witwin-maxwell/.worktrees/wd-b10-sar
export CUDA_VISIBLE_DEVICES=0
conda run -n maxwell --no-capture-output python -m pytest <targets> -q
```

## Scope delivered (B1: Phase 0 + Phase 1)

- **Mass-density material channel.** `Material(mass_density=...)` (kg/m^3; positive
  scalar or 3D per-cell grid; `None` excludes from SAR). Flags `has_mass_density`,
  `is_electrically_lossy`. `witwin/maxwell/media.py`.
- **Mass compiler.** `Scene.compile_mass_density()` →
  `CompiledMassDensity(rho_cell, occupancy, tissue_id, cell_volume, tissue_names)` on
  the material node grid, built through the EM compiler's soft-occupancy path
  (`_geometry_occupancy`, priority order) so the SAR mass model and EM loss model
  share occupancy provenance. `rho_cell` is the occupancy-weighted **effective**
  density (true density = `rho_cell/occupancy`); `tissue_id` is stop-grad.
  `witwin/maxwell/compiler/mass_density.py`.
- **PowerLossData consumption + point SAR.** `Result.sar(monitor, averaging=,
  normalization=)` and the reducer `postprocess/sar.py::compute_sar`. Reuses the
  existing `PowerLossData` volumetric W/m^3 (no reimplemented loss extraction).
  Electric loss density on the staggered Yee edges (Ex/Ey/Ez) is colocated to
  material cell centers by a **power-conserving half-weight scatter** (the transpose
  of the compiler's node→edge midpoint average). Point SAR = `q_cell / rho_cell`
  (effective density → occupancy cancels, recovering true tissue SAR). Per-channel
  (`conduction`, `electric_dispersion`, `nonlinear`) and `total`, unit W/kg, NaN below
  the occupancy epsilon (`1e-6`), never zero-filled.
- **Region statistics.** Per `tissue_id`: total absorbed power [F], mean/max SAR [F],
  tissue mass, cell count.
- **Explicit-failure paths.** SAR request covering an electrically lossy material with
  no `mass_density` → `ValueError`; no volumetric electric-loss channel → `ValueError`;
  non-FDTD result → `NotImplementedError`; unknown monitor / missing conduction fields →
  raised by the reused `Result.power_loss` path; `accepted_power`/`input_power`
  normalization → `NotImplementedError` (resolved in B3).
- **Serialization.** `SARResult.save/load` writes a typed payload (`values`, units,
  `coordinates`, `frequencies`, `field_convention`, `normalization`, averaging
  profile, `grid_hash`, mass model, statistics) — typed data, not anonymous metadata.
- **Public API surface.** `SARAveraging`, `PowerNormalization`, `SARResult`,
  `AVERAGING_PROFILE` exported from `witwin.maxwell`.

## Loss-channel convention consumed (recorded per brief)

`PowerLossData` uses the **peak-phasor** convention with time-average electric
conduction density `q = 0.5 * sigma_e * |E|^2` (W/m^3) on the Yee electric edges
(`PowerLossData.normalization` / `.phasor_convention`). SAR formulas match this
exactly: point SAR = `q / rho_cell`, and the analytic gate uses
`SAR = 0.5*sigma*|E|^2 / rho` (equivalently `sigma|E|^2/(2 rho)`).

## Averaging-profile position (recorded per brief)

Only the mass-averaging **request** object ships in B1 (`SARAveraging`, profile
`"cubical-prefix-v1"`); the averaging computation is B2. The name is IEEE/IEC-inspired
but NOT certified. `connectivity` other than `"cube"` (tissue flood-fill) and
`boundary_policy` other than `"strict-interior"` raise `NotImplementedError`.

## Test inventory

The B1 files are `tests/sar/test_point_sar.py` (point-SAR reducer) and
`tests/sar/test_mass_density.py` (mass-density compiler). They have since been
extended by B2/B3 and the 2026-07-19 audit-fix pass, so their totals now exceed the
original B1 scope. Current collected counts are reproducible:

```
conda run -n maxwell ... python -m pytest \
  tests/sar/test_point_sar.py tests/sar/test_mass_density.py --collect-only -q | tail -1
# 18 tests collected  (test_point_sar.py: 14, test_mass_density.py: 4)
```

The CUDA-device test (`test_sar_stays_on_cuda_device`) runs and passes on device 0.

Headline gates:
- `test_point_sar_matches_analytic_conduction` — uniform lossy cube, constant Yee-edge
  fields ⇒ peak point SAR == `0.5*sigma*(|Ex|^2+|Ey|^2+|Ez|^2)/rho` to rtol 1e-5.
- `test_volume_integrated_power_closes_against_power_loss_total` — `sum(q*cell_volume)`
  over the region == electric channel power == `PowerLossData.total` (machine-tight).
- `test_cpu_float64_oracle_parity` — independent float64 numpy collocation reproduces
  the float32 reducer field (rtol 1e-4).
- `test_missing_mass_density_raises` — lossy material without density → `ValueError`.
- `test_per_channel_decomposition_sums_to_total`, `test_region_statistics_*`,
  `test_normalization_source_amplitude_scales_sar_by_square`,
  `test_point_sar_preserves_field_autograd`, `test_sar_result_serialization_roundtrip`.

## Falsifications performed (evidence discipline)

1. **Collocation / closure / analytic / oracle (load-bearing numerics).** Changed the
   scatter split weight in `postprocess/sar.py::_scatter_edge_power_to_nodes` from
   `0.5` to `0.7` (breaks power conservation). Observed: the analytic, closure, and
   float64-oracle tests all turned red (`3 failed`). Restored to `0.5` → green.
2. **Fail-closed missing-density guard.** Inserted an early `return` in
   `_reject_lossy_material_without_density`. Observed:
   `test_missing_mass_density_raises` turned red (`DID NOT RAISE ValueError`).
   Restored → green.

## Adjacent suites (regression)

```
python -m pytest tests/sar/ tests/rf/power_loss/ tests/api/public/test_public_api.py \
    tests/api/public/test_simulation_smoke.py -q        # 55 passed
python -m pytest tests/core/scene/test_scene.py \
    tests/materials/compiler/test_material_compiler.py -q # 78 passed
python -m pytest tests/materials/ -q                      # 434 passed, 1 skipped
```

## Known gaps / deferred (by design for B1)

- Mass averaging (1 g/10 g cubical-prefix-v1), peaks, provenance validity masks — **B2**.
- Power normalization `accepted_power`/`input_power` resolution, coherent/incoherent
  multi-source combination, `soft_peak`, finite-difference gradient gates — **B3**.
- `IncidentPowerDensityMonitor` — not implemented (absorbed-power SAR is the B1
  deliverable); revisit only if it falls out of existing Poynting/flux machinery.
- Multi-GPU SAR reduction and VOP (plan Phase 5) — OUT of scope, fail closed.
- Occupancy provenance uses the single-sample (`subpixel=(1,1,1)`) geometry occupancy;
  it matches the default EM occupancy but not a scene configured with subpixel
  averaging > 1. Fully-occupied interior cells are unaffected. Documented limitation.
- At an internal tissue/air interface a fraction of interface-edge power colocates to
  air-side cells and is excluded from per-tissue statistics; the region volume integral
  still closes exactly (recorded in `SARResult.provenance["boundary_note"]`).

## Files added / changed

Added: `witwin/maxwell/sar.py`, `witwin/maxwell/compiler/mass_density.py`,
`witwin/maxwell/postprocess/sar.py`, `tests/sar/test_point_sar.py`,
`tests/sar/test_mass_density.py`, this doc.
Changed: `witwin/maxwell/media.py` (Material.mass_density + flags),
`witwin/maxwell/compiler/materials.py` (none — helpers reused),
`witwin/maxwell/result.py` (`Result.sar`), `witwin/maxwell/scene.py`
(`Scene.compile_mass_density`), `witwin/maxwell/__init__.py` (exports),
`FEATURE_LIST.md`.

No existing fail-closed guard or test was removed or weakened; no FDTD capability-guard
census entry was added or removed.

---

# Track B (Plan 10 SAR) — B2 acceptance (Phase 2: mass averaging)

Date: 2026-07-19 (same worktree/branch/env prelude as B1 above).

## Scope delivered (B2: Phase 2 — cubical-prefix-v1)

- **Mass-averaging kernel.** `witwin/maxwell/postprocess/sar_averaging.py::compute_mass_averaged_sar`.
  For each target averaging mass and every candidate tissue center, the smallest
  symmetric axis-aligned cube (half-width `h` cells, edge `2h+1`) whose enclosed
  tissue mass reaches `m0` is selected; averaged SAR = `enclosed_power / enclosed_mass`
  with the ACTUAL enclosed mass recorded. Enclosed mass / power / tissue-volume /
  total-volume of any cube are read in O(1) from zero-padded 3D inclusive prefix sums
  (integral images) via 8-term inclusion-exclusion; the per-center search is one
  prefix lookup per half-width (O(N·H)) versus the O(N·k^3) brute force.
- **Validity rules (mask + NaN, never padded), all in provenance.**
  `boundary_policy="strict-interior"`: a center's max half-width is its Chebyshev
  distance to the region boundary; a cube that would clip is disallowed and an
  unreachable target within the interior cube marks the center invalid. Tissue fill
  fraction of the chosen cube must be ≥ `min_tissue_fraction` (default 0.1, the
  "no air-mass makeup" rule). Averaged SAR is reported only at tissue-bearing centers.
- **Peaks.** `SARResult.peak(mass)` → typed `SARPeak` (per-frequency peak value,
  center index + physical position, actual enclosed mass, cube half-width in cells,
  physical cube edge per axis). `argmax` center is stop-grad; the peak value keeps
  the field graph. `SARResult.averaged_sar(mass)` returns the `[F,nx,ny,nz]` field;
  `SARResult.averaging_masses` lists computed masses. Both accessors fail closed when
  no averaging was requested or an unrequested mass is asked for.
- **Differentiability.** Fixed-window averaged SAR stays in autograd: the search runs
  on detached prefixes, then enclosed power/mass are re-gathered WITH grad at the
  chosen half-width (grouped by unique `h`, selected via `torch.where`). Verified by
  a backward pass with nonzero, finite gradient (`test_averaged_sar_is_differentiable`).
- **Serialization.** `averaged` and `peaks` ride the existing `SARResult.save/load`
  typed payload (`SARPeak.payload/from_payload`). Roundtrip verified.
- **Public surface.** `SARPeak` exported from `witwin.maxwell`; `SARResult.peak/
  averaged_sar/averaging_masses` implemented (were `NotImplementedError` in B1).

## Difference from IEEE/IEC 62704-1 (documented, not certified)

Symmetric index-space cube — no cube-face expansion asymmetry; no tissue-connectivity
flood fill in v1 (`connectivity="cube"` only, others raise). On nonuniform grids the
cube is symmetric in index space, not a physical cube.

## Test inventory

Command (env prelude above):
`conda run -n maxwell --no-capture-output python -m pytest tests/sar/test_mass_averaging.py -q`
→ **15 passed** (incl. 1 CUDA device test, executed on `CUDA_VISIBLE_DEVICES=0`).

Adjacent regression:
`... python -m pytest tests/sar/ tests/rf/power_loss/ tests/api/public/test_public_api.py tests/api/public/test_simulation_smoke.py -q`
→ **70 passed**.

Key tests in `tests/sar/test_mass_averaging.py`:
- `test_uniform_one_gram_cube_exact_size_and_mass` — golden: dx=0.1, rho=1000 ⇒
  mass_cell=1 kg; m0=27 kg ⇒ exact 3×3×3 cube (h=1, size 0.3 m, mass 27.0), averaged
  SAR = point SAR; strict-interior invalidates the border (9³ of 11³ valid).
- `test_prefix_matches_bruteforce_random` — **load-bearing**: integral-image result
  equals an independent O(N·k^3) NumPy reference (avg field, enclosed mass, half-width,
  and validity mask) on random rho/occupancy/power over three (mass, min_fraction) cases.
- `test_peak_monotonic_in_mass` — 10 g peak ≤ 1 g peak (and half-width grows).
- `test_two_material_halfspace_average_is_mass_weighted` — straddling cube = Σ(qV)/Σ(ρV).
- `test_partial_occupancy_uses_effective_mass` — enclosed mass uses effective ρ·V.
- `test_min_tissue_fraction_rejects_air_makeup` — mostly-air cube invalid.
- `test_unreachable_mass_marks_invalid` / `test_strict_interior_boundary_invalidates_clipped_centers`.
- `test_grid_convergence_of_peak_averaged_sar` — Gaussian power on 3 grids (n=16/24/40),
  peak 1 g averaged SAR converges (successive differences shrink).
- `test_averaged_sar_is_differentiable`.
- Full-pipeline: `test_result_sar_peak_matches_point_analytic_for_uniform_field`,
  `test_peak_requires_averaging_request`, `test_peak_unknown_mass_fails_closed`,
  `test_averaged_and_peaks_serialization_roundtrip`, `test_averaging_stays_on_cuda_device`.

## Falsifications performed (recorded per brief)

1. **Box-sum inclusion-exclusion sign** — flipped `+ gather(ax, ay, bz)` to `-` in
   `sar_averaging.py::_box_sum`. `test_prefix_matches_bruteforce_random` went RED
   (`AssertionError`). Restored → green.
2. **Strict-interior clip guard** — replaced `within_interior = h <= max_halfwidth`
   with an all-True mask (clipped cubes allowed). `test_strict_interior_boundary_
   invalidates_clipped_centers` went RED. Restored → full file 15 passed.

## Known gaps / deferred (unchanged from B1 for B3)

- Power normalization `accepted_power`/`input_power`, coherent/incoherent multi-source
  combination, `soft_peak`, finite-difference gradient gates on σ/density — **B3**.
- Standard/independent-reference phantom cross-check (plan §9) not run (no
  redistributable phantom fixture in-repo yet); golden + brute-force parity + algorithmic
  grid convergence stand in for correctness. Full "completed" gating is the supervisor's.

## Files added / changed (B2)

Added: `witwin/maxwell/postprocess/sar_averaging.py`,
`tests/sar/test_mass_averaging.py`.
Changed: `witwin/maxwell/sar.py` (`SARPeak`, `SARResult.peak/averaged_sar/
averaging_masses`, averaged/peaks fields populated + serialization),
`witwin/maxwell/postprocess/sar.py` (wire averaging into `compute_sar`),
`witwin/maxwell/__init__.py` (`SARPeak` export), `FEATURE_LIST.md`.

No existing fail-closed guard or test was removed or weakened; no FDTD capability-guard
census entry was added or removed.

---

# B3 — Phase 3 (normalization + multi-source combination) + Phase 4 slice (soft_peak, gradient gates)

## Delivered items

- **Power normalization resolution** (`sar.py::PowerNormalization.resolve_scale`):
  - `source(amplitude)` → `amplitude**2` (unchanged, exact square law).
  - `accepted_power(port, watts)` → per-frequency scale `watts / measured` read from the
    named port's `accepted_power` at the SAR frequencies, keeping the port autograd graph.
    Fails closed on missing port (`KeyError`), frequency not in the port spectrum
    (`KeyError`), or non-positive measured accepted power (`ValueError`).
  - `input_power(watts)` fails closed (`NotImplementedError`): this build exposes no total
    injected source-power diagnostic (searched — none present). Sanctioned by supervisor
    decision #4 ("if not available fail closed").
  - Resolution is wired at `Result.sar` (which carries the ports); `compute_sar` accepts a
    resolved `power_scale` (scalar or `[F]`) and broadcasts it over `[F, nx, ny, nz]`.
- **Coherent combination** (`postprocess/sar_combine.py::combine_coherent_sar`): sums the
  complex electric spectra of same-frequency runs (optional complex weights) BEFORE the loss,
  then one SAR reduction — interference exact. Differentiable in per-run fields.
- **Incoherent combination** (`combine_incoherent_sar`): power-domain sum of absorbed-power
  densities, point SAR re-formed as `power/rho`, per-tissue statistics recomputed via the
  shared `compute_tissue_statistics` helper; `averaging` recomputes cubical-prefix-v1 peaks on
  the combined power using the dual cell sizes stored in provenance. Validates grid hash,
  tissue model, frequencies, field convention, channels, and normalization; fails closed on
  mismatch.
- **soft_peak** (`sar.py::SARResult.soft_peak`): temperature-weighted softmax over valid
  mass-averaged cube centers, per frequency; approaches the hard `peak(...)` as temperature
  → 0, stays in autograd. Explicitly non-regulatory. `mass` optional for a single averaging
  mass.
- Provenance additions: resolved `power_scale`, `cell_sizes_dual`, and a `combination`
  descriptor on combined results. `compute_tissue_statistics` factored out of `compute_sar`
  and reused by the incoherent combiner.
- Exports: `mw.combine_coherent_sar`, `mw.combine_incoherent_sar`.

## Test inventory (env `maxwell`, `CUDA_VISIBLE_DEVICES=0`)

Command:
```
export CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=<worktree>
conda run -n maxwell --no-capture-output python -m pytest tests/sar/ -q
```

- `tests/sar/` → **57 passed**.
- New files:
  - `tests/sar/test_normalization.py` (7): accepted-power watts/measured scaling, per-frequency
    scaling, missing-port fail-closed, frequency-mismatch fail-closed, non-positive-measured
    fail-closed, input_power fail-closed, source square-law exactness.
  - `tests/sar/test_sar_combine.py` (7): coherent in-phase 4×, coherent opposite-phase cancel,
    incoherent field sum, incoherent recomputed peak (2× single), normalization-mismatch
    fail-closed, two-operand requirement, coherent field autograd preserved.
  - `tests/sar/test_soft_peak.py` (6): below-hard/above-mean, approaches-hard-as-T-drops,
    single-mass default, differentiable, requires-averaging, rejects non-positive temperature.
  - `tests/sar/test_sar_gradients.py` (5): field-amplitude autograd vs central difference for
    point and fixed-cube averaged SAR; density-grid autograd vs FD; conductivity and density
    central-difference vs analytic.
- Updated `tests/sar/test_point_sar.py`: the B1/B2 placeholder
  `test_accepted_and_input_power_normalization_fail_closed` (which asserted the OLD
  "normalization stage" `NotImplementedError`) is replaced by
  `test_accepted_power_normalization_fails_closed_without_port` (KeyError) and
  `test_input_power_normalization_fails_closed` (NotImplementedError) to match the
  now-implemented capability. No guard was weakened — the fail-closed behavior is retained,
  only its trigger and message changed.

Adjacent suites (regression):
```
conda run -n maxwell ... python -m pytest tests/api/public/test_public_api.py \
  tests/api/public/test_simulation_smoke.py tests/rf/power_loss/ -q
```
→ **39 passed**.

## Numerical-precision note (gradient gates)

The point-SAR reducer runs in float32 (GPU-first design from B1). Autograd and central
differences therefore agree to float32 precision, not float64. The gate tolerances are
asserted in the tests themselves (reproducible via
`conda run -n maxwell ... python -m pytest tests/sar/test_sar_gradients.py -q`):
`rtol=2e-3` for autograd-vs-FD (field) and `rtol=1e-3/1e-4` for
central-difference-vs-analytic (density/conductivity), with wide FD steps to defeat float32
catastrophic cancellation. This is a documented limit of the differentiable path, not a bug in
the SAR math. `Material.sigma_e` as a plain scalar is floated at compile time (not an autograd
leaf in this build), so the conductivity gate is central-difference-vs-analytic rather than
autograd-vs-FD; the E-field amplitude and a 3D `mass_density` grid ARE autograd leaves and are
checked against FD directly.

## Falsifications performed (B3)

1. **Accepted-power scale inversion** — `watts / measured_at` → `watts * measured_at` in
   `sar.py::_resolve_accepted_power_scale`. `test_accepted_power_scales_by_watts_over_measured`
   and `test_accepted_power_is_per_frequency` went RED (ratio 0.75× wrong). Restored → green.
2. **Coherent field-sum sign** — `summed + weight*tensor` → `summed - weight*tensor` in
   `sar_combine.py`. `test_coherent_in_phase_doubles_field_and_quadruples_sar` went RED (result
   0 instead of 4×). Restored → green.
3. **soft_peak → softmin** — `flat / temp` → `-flat / temp` in `SARResult.soft_peak`.
   `test_soft_peak_below_hard_peak_and_above_mean` and
   `test_soft_peak_approaches_hard_peak_as_temperature_drops` went RED (surrogate collapsed to
   the min). Restored → green.
4. **SAR density law** — point-SAR total `total_q / safe_rho` → `total_q / safe_rho**2 * 1000`.
   `test_point_sar_density_central_difference_matches_analytic` went RED (FD derivative doubled,
   ∝ 1/rho² not 1/rho). Restored → `tests/sar/` 57 passed.

## Known gaps / deferred (B3)

- `input_power` normalization fails closed: no total injected source-power diagnostic exists in
  this build. Wiring one (from source monitors) is future work.
- `IncidentPowerDensityMonitor` not implemented (did not fall out cheaply from existing
  Poynting/flux machinery); absorbed-power SAR is the deliverable, per track brief.
- Gradient gates are float32-limited (reducer dtype); a float64 differentiable reducer path is
  not in scope for this slice.
- Multi-GPU SAR reduction and VOP (Phase 5) remain out of scope (fail closed).
- Standard/independent-reference phantom cross-check (plan §9) still not run (no redistributable
  phantom fixture in-repo).

## Files added / changed (B3)

Added: `witwin/maxwell/postprocess/sar_combine.py`, `tests/sar/test_normalization.py`,
`tests/sar/test_sar_combine.py`, `tests/sar/test_soft_peak.py`,
`tests/sar/test_sar_gradients.py`.
Changed: `witwin/maxwell/sar.py` (accepted-power resolution + frequency matcher + `soft_peak`),
`witwin/maxwell/postprocess/sar.py` (`power_scale` param + broadcast, `compute_tissue_statistics`
helper, cell-size/scale provenance), `witwin/maxwell/result.py` (resolve + pass `power_scale`),
`witwin/maxwell/__init__.py` (combiner exports), `tests/sar/test_point_sar.py` (placeholder
normalization tests updated for the now-implemented capability), `FEATURE_LIST.md`.

No FDTD capability-guard census entry was added or removed.

---

# Audit-fix pass (2026-07-19)

Same worktree/branch/env prelude as above. Supervisor-selected fixes from the B10 audit.

## Delivered fixes

1. **Occupancy-epsilon exclusion is now tested.** The `valid` mask in
   `postprocess/sar.py::compute_sar` is `(occupancy >= OCCUPANCY_EPSILON) & (rho_cell > 0)`,
   but no test isolated the occupancy clause (deleting it left the suite green). New test
   `tests/sar/test_point_sar.py::test_occupancy_below_epsilon_is_excluded_from_point_sar_and_statistics`:
   a fully-occupied uniform cube is reduced once for a baseline, then one interior region cell's
   occupancy is pushed to `0.1 * OCCUPANCY_EPSILON` (still `> 0`, `rho_cell` unchanged and positive)
   via `dataclasses.replace` on the compiled mass model. The cell must become NaN in point SAR,
   drop from `valid.sum()` by exactly 1, and drop from the tissue `cell_count` by exactly 1.
   Because only occupancy is starved, the `rho_cell > 0` clause alone would keep the cell valid,
   so the test pins the occupancy clause specifically.

2. **Incoherent error taxonomy unified.** `postprocess/sar_combine.py::_check_metadata_match` used
   `torch.testing.assert_close` for the `rho_cell` mismatch (raising `AssertionError`) while every
   sibling mismatch raised `ValueError`. Replaced with an explicit shape+`torch.equal` comparison
   raising `ValueError`. New test
   `tests/sar/test_sar_combine.py::test_incoherent_density_mismatch_raises_value_error` builds two
   otherwise-identical runs (same grid/tissue map/validity/frequencies/channels/normalization)
   that differ only in `rho_cell` (rho 1000 vs 1100) and asserts `ValueError`.

3. **`combine_coherent_sar` now validates the grid.** It previously checked monitor name,
   frequencies and field shape but not grid spacing (the incoherent path checks `grid_hash`). Added
   a node-coordinate grid hash (`_prepared_grid_hash`) comparison across runs. New test
   `tests/sar/test_sar_combine.py::test_coherent_rejects_same_shape_different_spacing` builds two
   runs with identical field shapes but different spacing (via the new `scale` parameter on
   `_uniform_cube_result`, which scales domain + spacing + geometry together so the node count is
   invariant) and asserts `ValueError`.

4. **Acceptance-doc corrections.** Fixed the internally-inconsistent B1 test-count breakdown
   (was `13+5=16`); the B1 files now collect 18 tests, stated with a reproducible `--collect-only`
   command. Removed the unreproducible "~2-4e-4 relative in measured runs" precision figure from the
   B3 gradient note and pointed instead at the gate tolerances asserted in
   `tests/sar/test_sar_gradients.py` (reproducible via a pytest command).

## Falsifications performed (recorded per brief)

1. **Occupancy-epsilon clause** — in `postprocess/sar.py` changed
   `valid = (occupancy >= OCCUPANCY_EPSILON) & (rho_cell > 0)` to `valid = (rho_cell > 0)`.
   `test_occupancy_below_epsilon_is_excluded_from_point_sar_and_statistics` went RED
   (`assert not bool(sar.valid[local])` → `assert not True`). Restored → green.
2. **Incoherent density-mismatch taxonomy** — reverted the new `ValueError` block to
   `torch.testing.assert_close`. `test_incoherent_density_mismatch_raises_value_error` went RED
   (raised `AssertionError`, not the expected `ValueError`). Restored → green.
3. **Coherent grid check** — short-circuited the new guard with `if False and ...`.
   `test_coherent_rejects_same_shape_different_spacing` went RED (no `ValueError` raised).
   Restored → green.

## Test inventory (this pass)

```
conda run -n maxwell ... python -m pytest tests/sar -q
# 60 passed  (was 57; +3 new: occupancy-epsilon, coherent grid-mismatch, incoherent density-mismatch taxonomy)
```

Adjacent regression:
```
conda run -n maxwell ... python -m pytest tests/api/public/test_public_api.py \
  tests/api/public/test_simulation_smoke.py tests/rf/power_loss/ -q
```

## Files changed (audit-fix pass)

Changed: `witwin/maxwell/postprocess/sar_combine.py` (ValueError for rho_cell mismatch;
`_prepared_grid_hash` + coherent grid check), `tests/sar/test_point_sar.py`
(`scale` parameter on `_uniform_cube_result`; occupancy-epsilon test),
`tests/sar/test_sar_combine.py` (coherent grid-mismatch + incoherent density-mismatch tests),
`docs/assessments/b10-sar-acceptance-2026-07-19.md` (this section + B1/B3 corrections).
`witwin/maxwell/postprocess/sar.py` unchanged (only used transiently for falsification 1).

No FDTD capability-guard census entry was added or removed; no existing guard or test was weakened.
