# Track H3 (Plan 10 SAR) — H3a acceptance

Stage: **H3a** — IncidentPowerDensityMonitor with analytic gates + four canonical
phantom benchmark scenes with golden/analytic gates + falsifications.

Date: 2026-07-21
Worktree: `.worktrees/wh3-sar-phantom` (branch `fable/sar-phantom`).
Environment: conda env `maxwell`, `CUDA_VISIBLE_DEVICES=0`.

Reproduce prelude (every command below assumes it):

```bash
export CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=/home/xingyu/code/witwin/witwin-maxwell/.worktrees/wh3-sar-phantom
export CUDA_VISIBLE_DEVICES=0
conda run -n maxwell --no-capture-output python -m pytest <targets> -q
```

## Delivered

### IncidentPowerDensityMonitor (plan §3)

- `IncidentPowerDensityMonitor(name, axis=, position=, frequencies=, normal_direction=,
  spatial_average=)` — a `PlaneMonitor` subclass carrying the four tangential
  fields required to form the normal Poynting component; reuses the same flux
  machinery as `FluxMonitor` (`witwin/maxwell/monitors.py`).
- `plane_normal_poynting(result)` factored out of `_compute_plane_flux` in
  `witwin/maxwell/fdtd/observers.py` as the single source of truth for the per-cell
  time-averaged normal Poynting `S.n = 0.5*Re((E x conj(H)).n_hat)` and the cell-area
  weights. `_compute_plane_flux` now sums that helper, so the monitor's integrated
  flux is identically the `FluxMonitor` integral.
- `Result.incident_power_density(monitor, spatial_average=...)` (`witwin/maxwell/result.py`)
  → typed `IncidentPowerDensity` (`witwin/maxwell/postprocess/incident_power.py`):
  signed `normal_poynting`, `power_density = |S.n|` (W/m^2), plane-integrated `flux`,
  and (when requested) the versioned `spatial-average-v1` moving-window average of
  `|S.n|` — an axis-aligned `sqrt(area)`-side square per cell, O(N) via 2D inclusive
  prefix sums, edge-truncated, explicitly non-certified (provenance `certified: False`).
- Exports: `mw.IncidentPowerDensityMonitor`, `mw.IncidentPowerDensity`.
- Fails closed when the named monitor is not an `IncidentPowerDensityMonitor` (`KeyError`).

### Canonical phantom benchmark family (plan §9)

Under `benchmark/scenes/sar/` (redistributable canonical geometry only; tissue
dielectric/mass values are published-class 900 MHz numbers in `_tissue.py`,
Gabriel-1996 / IEC-IEEE-62704-class):

- `uniform_lossy_cube` — homogeneous lossy phantom under a normally incident plane wave.
- `layered_slab` — skin/fat/muscle three-layer slab (the convergence/conservation target for H3b).
- `one_gram_cube` — synthetic grid where a 3x3x3 window weighs EXACTLY 1 g (hand-computable golden).
- `antenna_near_phantom` — driven dipole near a tissue block (**recorded design blocker**, see below).

Each module exposes `build_scene(...)` and a `ScenarioDefinition`; a `SAR_SCENARIOS`
tuple is collected in `benchmark/scenes/sar/__init__.py`. Registration into the main
benchmark `SCENARIOS` registry and RESULTS rows are deferred to H3b.

## Test inventory

```
conda run -n maxwell ... python -m pytest tests/sar/ -q          # 109 passed
```

New files:
- `tests/sar/test_incident_power.py` (7 tests): plane-wave `|S| = |E|^2/(2*eta)`
  exact-class on synthetic payloads; `flux == _compute_plane_flux` and `== analytic
  |E|^2/(2*eta)*area`; sign flips with `normal_direction`; spatial average of a uniform
  field equals the constant; spatial average matches an independent brute-force reference
  on a non-uniform field; `plane_normal_poynting` sums to flux; and an end-to-end vacuum
  plane-wave FDTD run tying the monitor flux to a co-located `FluxMonitor` (CUDA).
- `tests/sar/test_phantom_benchmarks.py` (4 tests): `one_gram_cube` exact 1 g window
  (half-width 1, enclosed mass 1e-3, averaged == analytic point SAR); `uniform_lossy_cube`
  exact power-conservation closure + monotone averaging + golden anchors; `layered_slab`
  three-tissue peak-in-skin (fat lowest) + golden anchors; `antenna_near_phantom`
  fail-closed on conductive media (CUDA for the three run-based ones).

Adjacent regression (env prelude above):
```
python -m pytest tests/sar/ tests/api/public/test_guard_census.py \
  tests/api/public/test_public_api.py tests/api/public/test_simulation_smoke.py \
  tests/monitors/observers/test_fdtd_observers.py -q            # 109 passed
python -m pytest tests/postprocess/scattering/test_scattering_parameters.py \
  tests/monitors/ tests/sources/incident/test_soft_planewave_absolute_power.py -q
                                                                # 48 passed, 1 xpassed
```

The capability-guard census is unchanged at 175 (`tests/api/public/test_guard_census.py`
passes). The incident-power reducer lifts a NumPy payload to a CPU tensor rather than
raising, so no new capability guard was added.

## Golden values (deterministic, reproducible by rerunning the node)

Plane-wave scenes run at their default `dx`, `TimeConfig.auto(steady_cycles=6,
transient_cycles=15)`, `SpectralSampler(normalize_source=True)`, `full_field_dft=True`:

- `uniform_lossy_cube`: peak point SAR ~ 0.4386, peak 1 g ~ 0.3763, peak 10 g ~ 0.2841 W/kg;
  power-closure residual < 1e-5 (exact by the power-conserving colocation).
- `layered_slab`: peak 1 g ~ 0.5697 (in the skin layer, z in [0, 0.008]), peak 10 g ~ 0.2959 W/kg;
  fat (lowest sigma) has the lowest per-tissue max SAR; closure residual < 1e-5.
- `one_gram_cube`: enclosed mass exactly 1e-3 kg, cube half-width 1, averaged 1 g SAR
  = 0.5*sigma*3*e0^2/rho (analytic, machine-tight).

Golden peaks are asserted with `rel=3e-2` (CUDA reduction slack); the load-bearing
gates are the exact power closure, the exact 1 g window, and the analytic `|S|` relation.

## Falsifications performed (evidence discipline)

1. **Poynting factor (incident power analytic gate).** `plane_normal_poynting` torch branch
   `0.5 * torch.real(...)` -> `0.25 * ...`. `test_plane_wave_power_density_matches_analytic`
   and `test_flux_matches_plane_flux_helper_and_analytic` went RED (relative difference 0.5).
   Restored -> green.
2. **Spatial-average box-sum inclusion-exclusion.** `incident_power._box_sum`
   `+ gather(lo_u, lo_v)` -> `- gather(lo_u, lo_v)`.
   `test_spatial_average_matches_bruteforce_on_nonuniform_field` went RED (the uniform-field
   test cannot catch this because num/den scale together). Restored -> green.
3. **One-gram grid choice (scene golden).** `one_gram_cube.ONE_GRAM_DX` scaled by 1.05, so
   27 cells no longer weigh 1 g. `test_one_gram_cube_exact_hand_computed_average` went RED
   (enclosed mass / analytic assertion). Restored -> green.
4. **Power-conservation closure (scene load-bearing gate).** `postprocess/sar.py`
   colocation split `half = 0.5 * edge_full` -> `0.7 * edge_full`.
   `test_uniform_lossy_cube_power_closure_and_golden` went RED (closure residual 0.40).
   Restored -> green.

## Known gaps / deferred

- **Design blocker (recorded):** `antenna_near_phantom` cannot run the driven end-to-end
  SAR chain in this build. The FDTD port machinery fails closed on a conductive (lossy)
  background: the thin-wire runtime raises "Thin-wire FDTD does not yet support a conductive
  background electric update" and the lumped-port runtime raises "Lumped FDTD coupling in
  conductive media requires a conductance-aware port update coefficient"
  (`witwin/maxwell/fdtd/ports.py::_validate_supported_field_coupling`,
  `witwin/maxwell/fdtd/wire.py::_reject_unsupported_composition`). A tissue phantom is
  conductive by construction, so this is fundamental to the antenna+phantom combination.
  Closest fail-closed behaviour: the scene builds (declarative) and preparing it raises
  `NotImplementedError`, which is pinned as a gate. Unblocking needs a conductance-aware
  lumped-port update coefficient — out of this stage's scope; reported to the supervisor.
  The accepted-power normalization and save/load roundtrip are separately validated on
  synthetic ports (`tests/sar/test_normalization.py`, existing SAR serialization tests).
- **H3b (next stage in this track):** 3-grid convergence of the peak 1 g SAR on
  `layered_slab`, full power-conservation closure (incident-minus-scattered via flux
  monitors), benchmark RESULTS rows via the runner (analytic-reference class where no
  external run exists), optional single external reference run for `layered_slab` absorbed
  power density, FEATURE_LIST/census/acceptance updates, and the optional `input_power`
  normalization census-guard removal.

## Files added / changed (H3a)

Added: `witwin/maxwell/postprocess/incident_power.py`,
`benchmark/scenes/sar/{__init__,_tissue,uniform_lossy_cube,layered_slab,one_gram_cube,antenna_near_phantom}.py`,
`tests/sar/test_incident_power.py`, `tests/sar/test_phantom_benchmarks.py`, this doc.
Changed: `witwin/maxwell/monitors.py` (IncidentPowerDensityMonitor +
INCIDENT_SPATIAL_AVERAGE_VERSION), `witwin/maxwell/fdtd/observers.py`
(`plane_normal_poynting` factored out; `_compute_plane_flux` sums it),
`witwin/maxwell/result.py` (`Result.incident_power_density` + import),
`witwin/maxwell/__init__.py` (exports), `FEATURE_LIST.md`.

No existing fail-closed guard or test was removed or weakened; no FDTD capability-guard
census entry was added or removed (budget stays 175).
