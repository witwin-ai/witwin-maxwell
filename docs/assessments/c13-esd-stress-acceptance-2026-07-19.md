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
