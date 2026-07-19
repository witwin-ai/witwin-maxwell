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

`tests/sar/test_point_sar.py` (13) + `tests/sar/test_mass_density.py` (5): 16 passed on
CPU; the CUDA-device test runs and passes on device 0.

```
conda run -n maxwell ... python -m pytest tests/sar/ -q
# 16 passed
```

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
