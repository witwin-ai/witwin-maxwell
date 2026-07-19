# Track C13 — ESD stress acceptance (C1: Plan 13 Phase 1)

Date: 2026-07-19
Worktree: `.worktrees/wd-c13-esd-stress` (branch `fable/esd-stress`)
Capability level delivered: **stress-only** (standard waveform reproduction + ideal terminal current injection + port V/I reporting). No failure prediction, no source-impedance network, no discharge-gun geometry, no arc/plasma model.

## Environment / reproduction

```bash
export CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=/home/xingyu/code/witwin/witwin-maxwell/.worktrees/wd-c13-esd-stress
export CUDA_VISIBLE_DEVICES=0
conda run -n maxwell --no-capture-output python -m pytest <targets> -q
```

## Delivered items

- `witwin/maxwell/esd.py` (new): `ESDWaveform` (IEC 61000-4-2 two-term Heidler sum, `n=1.8`, numeric per-term peak normalization, linear level-voltage scaling, revision-gated + provenance), `MeasuredWaveform`, `ESDDiagnostics`, `ESDResampledWaveform` (charge-conserving binned resampling + aliasing metric), `ESDCurrentSource` (terminal-port ideal current injection lowering to `UniformCurrentSource`), `ESDPortRecord`.
- `witwin/maxwell/scene.py`: `Scene.resolved_sources()` now expands any source exposing `resolve(scene)` (used by `ESDCurrentSource`); plain scenes are byte-for-byte unchanged (identity pass-through, verified by `test_scene_without_esd_is_unaffected`).
- `witwin/maxwell/result.py`: `Result.esd_waveform(name)` / `Result.esd_waveform_names()` typed plumbing (diagnostics + run-grid charge-conserving projection + provenance); reads run `dt`/`time_steps` from result metadata.
- `witwin/maxwell/__init__.py`: public exports `ESDWaveform`, `MeasuredWaveform`, `ESDCurrentSource`, `ESDDiagnostics`, `ESDResampledWaveform`, `ESDPortRecord`.
- `FEATURE_LIST.md`: additive `c13-esd-stress` subsection (delimited).
- Tests: `tests/esd/test_esd_waveform.py`, `tests/esd/test_esd_injection.py`.

## Test inventory (pass counts)

- `tests/esd/test_esd_waveform.py` — 15 passed (analytic diagnostics vs `scipy.integrate.quad`, dense-argmax peak, level scaling, IEC first-transient sanity band, rise-time reconstruction, charge conservation across 3 dt, action convergence, MeasuredWaveform).
- `tests/esd/test_esd_injection.py` — 10 passed (8 CPU construction/geometry/lowering + 2 CUDA end-to-end on GPU 0).
- Combined: `tests/esd/` — **25 passed in ~8 s**.
- Adjacent suites (regression): `tests/api/public/test_public_api.py`, `tests/api/public/test_simulation_smoke.py`, `tests/core/scene/test_scene.py`, `tests/rf/terminal/test_terminal_port_contract.py`, `tests/sources/definitions/test_custom_uniform_sources.py` — **90 passed**.

## Key measured numbers (all reproducible via the nodes above)

At `level_voltage=8000` V, `ed2-contact` (from `ESDWaveform.iec_61000_4_2(8000.0).diagnostics()`):
- peak current 34.11 A (model-inherent; ~14% above IEC nominal 30 A because the two Heidler terms superpose — documented, not a gate)
- current at 30 ns 16.65 A (IEC nominal 16 A), at 60 ns 8.36 A (IEC nominal 8 A)
- 10-90% rise time 0.809 ns (IEC band 0.7-1.0 ns)
- charge 1.223 uC, action integral 1.6246e-5 A^2 s
- charge/action vs `scipy.quad`: rel err 1.6e-7 / 3.5e-8
- resample charge_ratio = 1.000000 at dt in {2, 1, 0.5} ns; action aliasing_metric 1.79e-2 -> 7.00e-3 -> 1.56e-3 (monotone, ~2x per halving)

End-to-end FDTD (GPU 0, two PEC terminal boxes + `TerminalPort` + `ESDCurrentSource`, 1600 steps, dt=0.0385 ns):
- gap `Ez(0)=0` (causal); gap voltage ramps monotonically to a charge-accumulation plateau (`|Ez|` max at final step)
- `max|dV/dt|` at 1.62 ns vs analytic current peak 1.45 ns (< 3 ns tolerance)
- `|corr(dV/dt, target current)|` = 0.94 over the first 300 steps (documented tracking tolerance >= 0.90)

## Falsifications performed (red -> restore -> green)

1. **Analytic charge gate** (`test_charge_and_action_match_scipy_quadrature`): multiplied the diagnostics charge integral by 1.05 in `esd.py` -> FAILED (AssertionError at charge approx). Restored -> PASSED.
2. **Charge-conserving resampling gate** (`test_resampling_conserves_charge_across_three_dt`): replaced the per-bin mean integral with naive point sampling at the bin left edge (`bin_mean = self.current(edges[:-1])`) -> FAILED (charge_ratio != 1 within rel 1e-6). Restored -> PASSED.
3. **End-to-end injection tracking gate** (`test_esd_terminal_injection_drives_causal_transient_and_records_provenance`): replaced the shaped injection density with a constant (`density = [sign*3e5 ...]`) -> FAILED (slope-peak-time / correlation assertion). Restored -> PASSED.

## Known gaps / deferred (for C2/C3 and later phases)

- Injection uses the additive current-source path, NOT the existing Thevenin voltage-source `PortExcitation` runtime (which is voltage-source-only and explicitly rejects `CustomSourceTime`). Ideal current injection is naturally an additive current source; this is the fail-closed Phase-1 choice. Source-impedance networks / discharge-gun coupling are Phase 3 (out of scope here) and documented in the source provenance (`source_impedance: "none (Phase 3, out of scope)"`).
- The end-to-end "measured port current tracks target" is verified via the gap-voltage time derivative (`V=Q/C` for an ideal current source into the capacitive gap), which is a field-derived quantity proportional to the injected current. Absolute-amplitude calibration of a field-integrated H-contour port current is not asserted here (Phase-1 stress-only scope); the exact port current is reported through the charge-conserving injection record instead.
- The injection source-time table is a dense analytic table (point-sampled by the runtime scalar-injection path); the charge-conserving binned resampling is delivered and tested as the reporting/verification contract (`resample_to_grid`) and as the run-grid projection in `Result.esd_waveform`. Binding the injected table to the exact solver `dt` at prepare time (so the injected samples are literally the bin means) is a straightforward later refinement.
- Terminal footprint edges and `reference_plane` must land on the Yee half-grid (existing `TerminalPort` compile constraint), so ESD scenes must size PEC terminals accordingly.
- `BreakdownMonitor`, `ComponentRating`/`ComponentStressMonitor`, and typed stress results are Phase 2 (C2) — not in this change.

## Guard census

No fail-closed guard added, removed, or weakened; the FDTD capability-guard census budget is untouched. New fail-closed validations added inside `esd.py` (unsupported revision/discharge, non-terminal port, missing port, degenerate footprint) are local `ValueError`s, not tracked-census FDTD capability guards.

---

# C2 stage — Plan 13 Phase 2 (non-feedback stress/rating monitors, typed results)

Environment: conda env `maxwell`, `CUDA_VISIBLE_DEVICES=0`, `PYTHONPATH=<worktree>`, `CUDA_HOME` set to the env `nvidia/cu13` tree.

## Delivered items

- `witwin/maxwell/breakdown_stress.py` (new): core stress-only primitives.
  - `colocate_electric_magnitude(Ex, Ey, Ez)` — energy-consistent cell-center `|E|` (each Yee component averaged along its two node-staggered axes onto the common cell center `(x_half, y_half, z_half)`); exact on uniform fields.
  - `BreakdownStressAccumulator` — per-cell device-tensor running reduction with `allocate(...)` / `update(e_magnitude)` / `finalize(...)`. Update rule (pure device ops, no `.item()`/host sync): `max_field = max(max_field, |E|)` on occupied cells; `exceedance_time += dt·H(|E|−Ecrit)` with `H(0)=1` (`|E| >= Ecrit`); contiguous-run bookkeeping → `longest_exceedance`; optional `damage_integral += dt·(|E|/Ecrit)^k` accrued only while exceeding.
  - `BreakdownStressData` — typed result (peak field + index, exceedance / longest durations, qualifying-cell count, occupancy-weighted exceedance volume·time, damage volume, per-cell device maps, `locations()`, provenance with thresholds/colocation/occupancy-policy/model version/capability level).
  - `ComponentRating(voltage, current, energy, pulse_width, model)` and `ComponentStressData.from_time_series(t, V, I, rating)` → `P=V·I`, cumulative `∫P dt` (trapezoidal), peaks, coarse pulse width, per-channel exceedance summary vs the rating envelope.
- `witwin/maxwell/monitors.py`: `BreakdownMonitor(name, region|position/size, quantities, critical_field, minimum_duration, damage_exponent)` and `ComponentStressMonitor(name, port, rating, voltage_series, current_series)`, plus `_resolve_region_geometry`.
- Compiler: `compile_fdtd_breakdown_observers(scene)` (lightweight region/threshold records); breakdown/component monitors excluded from the spectral-observer path.
- FDTD runtime: `prepare_breakdown_observers` / `accumulate_breakdown_observers` / `get_breakdown_observer_results` in `fdtd/observers.py`; region cell-slice + fractional-overlap occupancy + control-volume resolution at prepare; region-sliced colocation each step; solver delegation methods; wired into `runtime/initialization.py` (compile), `runtime/stepping.py` (prepare/accumulate/finalize, resume guard), enabled only when a monitor is present (zero cost otherwise).
- `Result`: `breakdown(name)` / `breakdown_names()` (typed `BreakdownStressData`), `component_stress(name)` / `component_stress_names()` (resolves bound V/I series and reduces via `ComponentStressData`).
- Public exports (`BreakdownMonitor`, `ComponentStressMonitor`, `ComponentRating`, `BreakdownStressData`, `ComponentStressData`) in `witwin/maxwell/__init__.py`; `FEATURE_LIST.md` subsection appended.

## Test inventory (all under `tests/breakdown/`)

- `test_breakdown_accumulator.py` — 20 passed. Two-pulse exact exceedance + longest run; longest-run reset between pulses; exactly-at-threshold counts; just-below excluded; damage integral golden; damage disabled without exponent; minimum_duration qualifying mask/locations; zero-occupancy excluded from peak/exceedance; partial-occupancy volume·time weighting; colocation uniform-field magnitude; colocation-vs-energy-density consistency; shape/validation guards.
- `test_component_stress.py` — 11 passed (1 cuda-parity + 1 cpu-parity parametrization). Rating validation; power/energy golden; exceedance flags; disabled-channel; increasing-time guard; float32-vs-float64 parity (cpu + cuda); trapezoid-vs-rectangle falsification.
- `test_breakdown_monitor.py` — 8 passed. Construction/validation, region-with-bounds, region+box conflict, scene attach, ComponentStressMonitor binding/type guard.
- `test_breakdown_fdtd.py` — 3 passed (CUDA). End-to-end device stress maps + provenance; no-perturbation bitwise field parity with/without the monitor; component-stress reduction float64 parity on a real run.
- Total: **37 passed** (`tests/breakdown/`).

Adjacent suites rerun: `tests/api/public/test_public_api.py`, `tests/api/public/test_simulation_smoke.py`, `tests/core/scene/test_scene.py`, `tests/esd/` (C1), `tests/monitors/observers/test_fdtd_observers.py` — **96 passed**.

## Falsifications performed (red observed; source restored; green reconfirmed)

1. **Longest contiguous run** (`test_longest_run_resets_between_pulses`): removed the `torch.where(exceed, run+dt, 0)` reset so the run never resets → FAILED (longest = total exceedance). Restored → PASSED.
2. **Threshold convention** (`test_exactly_at_threshold_counts_as_exceedance`): changed `|E| >= Ecrit` to `>` → FAILED (exactly-at-threshold no longer counted). Restored → PASSED.
3. **Occupancy weighting** (`test_partial_occupancy_weights_region_volume_time`, `test_falsify_occupancy_ignored_would_change_weighted_metric`): dropped `occupancy` from the region weight (`weight = cell_volume`) → both FAILED. Restored → PASSED.
4. **Trapezoidal energy** (`test_falsify_rectangle_energy_would_break_trapezoid_parity`): replaced the trapezoid segment with a left-rectangle (`segment = power[:-1]*dt`) → FAILED. Restored → PASSED.
5. **Yee colocation** (`test_colocation_uniform_field_matches_analytic_magnitude`): dropped Ey/Ez from the magnitude (`sqrt(ex_c^2)`) → FAILED. Restored → PASSED.

(All five falsification edits were reversed and the full `tests/breakdown/` suite reconfirmed at 37 passed.)

## Known gaps / deferred

- `ComponentStressMonitor` consumes user-declared time-series monitors as V(t) and I(t) (e.g. a gap-voltage `FieldTimeMonitor` and a current proxy) and reduces them; it does not itself synthesize a calibrated H-contour port current (that stays with the RF terminal path / Phase 3). The reduction math (P, ∫P dt, exceedance) and its float64 parity are the tested contract. The measured pulse width is a coarse half-peak-of-power span, reported for information.
- Occupancy in the real run is computed as the fractional overlap of each cell control volume with the monitor box (partial voxels at the region boundary). Binding occupancy to a specific target *material* mask (rather than the monitor region) is a straightforward extension for material-scoped breakdown regions and is left for Phase 4 material breakdown descriptors.
- `BreakdownStressData` / `ComponentStressData` are not yet covered by `Result.save()` serialization (breakdown save/load is out of C2 scope; the payloads carry device tensors). If a stress run is persisted, exclude these monitors or extend the snapshot codec.
- Breakdown observers intentionally raise on FDTD resume/checkpoint (added to the existing resume guard), matching the field/time-observer policy.
- No dynamic feedback, conductivity switching, or event log — that is Phase 4 (other track), deliberately absent here (stress-only).

## Guard census

No fail-closed guard added, removed, or weakened; the FDTD capability-guard census budget is untouched. New validations (unsupported quantities, missing critical_field, damage-without-exponent, region/box conflict, rating type/positivity, increasing-time) are local `ValueError`/`TypeError`s, not tracked-census FDTD capability guards. The FDTD resume guard was extended to also reject `breakdown_observers` (same NotImplementedError policy as existing observer types), not weakened.
