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

---

# Stage H3b acceptance

Stage: **H3b** — layered-slab grid convergence + power-conservation closure +
RESULTS rows + docs/census/FEATURE_LIST (external run and `input_power` guard both
assessed and deferred with rationale).

Date: 2026-07-21. Same reproduce prelude as H3a (env `maxwell`, `CUDA_VISIBLE_DEVICES=0`).

## Delivered

### Power-conservation closure + grid convergence (plan §1)

- `benchmark/scenes/sar/layered_slab.py::build_conservation_scene(dx=, device=)` —
  a periodic-transverse variant of the layered slab. Under normal incidence on an
  infinite planar slab the physically correct transverse boundary is periodic (not
  PML), which makes the field transverse-uniform, so the closed-surface balance
  reduces to two z-planes: `P_absorbed = flux(z_in,+z) - flux(z_out,+z)`. The shipped
  `build_scene` (PML on all faces, golden-anchored in H3a) is unchanged.
- `tests/sar/test_phantom_convergence.py` (3 tests):
  - `test_layered_slab_power_conservation_via_flux` — the absorbed power measured as
    the volume conduction-loss integral (`sigma|E|^2`, the SAR basis) closes against
    the net surface Poynting balance (`flux_in - flux_out`, `E x H` on two planes) at
    dx=4 mm: residual 0.167 (< 0.20), magnitude ratio 0.833 in [0.75, 1.0].
    **wave-level** (surface `E x H` is independent of the volume loss).
  - `test_layered_slab_conservation_closure_converges` — 3 grids (5/4/3 mm): residual
    0.200 -> 0.167 -> 0.125, monotone, finest < 0.15 and coarse-minus-fine > 0.04.
  - `test_layered_slab_peak_1g_sar_three_grid_study` — records peak 1 g / 10 g SAR at
    3 grids (peak 1 g = 0.469 / 0.570 / 0.532 W/kg); gates the robust structure only
    (finite, positive, 10 g <= 1 g), and bounds the documented spread < 0.6. The peak
    is NOT gated to converge tightly: a source-normalized plane wave delivers a
    grid-dependent incident power density (peak `|S.n|` 9.7 -> 14.8 W/m^2 across the
    same 3 grids), and the peak is a pointwise max over a thin under-resolved 8 mm
    skin layer. The convergent, gate-bearing observable is the conservation closure.

### Benchmark RESULTS rows (plan §3)

- `benchmark/sar_validation.py` — a self-contained SAR phantom exposure validation
  harness (same pattern as `benchmark/rf_validation.py`): drives the phantom family
  through the public `Scene -> Simulation -> Result` path, writes a
  `## SAR exposure validation` section to `benchmark/RESULTS.md`, and a JSON artifact
  per scene under `docs/assessments/sar-phantom-validation/`. Wired as
  `python -m benchmark sar [scenes...]` (`benchmark/__main__.py`).
- Gate classes self-labelled with the verbatim `docs/reference/gate-classification.md`
  taxonomy: `sar/layered_slab` = **wave-level** (headline surface/volume closure
  16.7% at dx=4 mm, converging), `sar/one_gram_cube` = **analytic-identity** (rel 0.0),
  `sar/uniform_lossy_cube` = **analytic-identity** (volume/channel self-consistency
  0.0, supporting only), `sar/antenna_near_phantom` = **blocked**. Every row
  `external_reference: analytic-only`.
- `tests/sar/test_sar_validation.py` (5 tests): runner coverage; the analytic 1 g cube
  runner matches the hand-computed value (< 1e-4, GPU-free); the antenna runner reports
  `blocked` with the conductance-aware note and no headline metric; the layered-slab
  headline class is verbatim `wave-level`; the RESULTS section writer is idempotent
  (re-running replaces in place, preserves neighbouring sections).

## Test inventory (env prelude above)

```
python -m pytest tests/sar/test_phantom_convergence.py -q            # 3 passed (~25 s)
python -m pytest tests/sar/test_sar_validation.py -q                 # 5 passed (~3 s)
python -m pytest tests/sar/ tests/api/public/test_guard_census.py \
  tests/api/public/test_public_api.py tests/api/public/test_simulation_smoke.py -q
                                                                     # 109 passed
python -m benchmark sar                                              # 4 rows: 3 pass, 1 blocked
```

`tests/sar/` is now 79 tests (71 H3a + 8 H3b). The capability-guard census is unchanged
at **175** (`tests/api/public/test_guard_census.py` passes); H3b adds no
`NotImplementedError` guard and removes none.

## Falsifications performed (evidence discipline)

1. **Conservation closure (headline wave-level gate).** With the run held fixed
   (dx=4 mm), injecting a 30% error into the absorbed-power volume integral pushes
   the surface/volume residual to 0.359 (> 0.20 gate) and the magnitude ratio to
   0.641 (< 0.75 gate) — both RED. A 0.5 -> 0.25 Poynting-factor error (halving both
   surface fluxes) pushes the residual to 0.583 — RED. Restored -> green
   (`scratch/falsify.py`, not committed). Note: merely dropping the transmitted term
   `flux_out` does NOT falsify (transmittance is ~2%), which is itself the physical
   content — nearly all incident power is absorbed.
2. **RESULTS section writer idempotency.** `test_results_section_writer_is_idempotent`
   asserts a second `_replace_or_append_section` call keeps exactly one section header
   and preserves neighbouring `## ` sections; a duplicating writer would make the
   header count 2 (RED).

## Known gaps / deferred (with rationale)

- **External reference-solver run — assessed, deferred.** The plan authorizes at most
  one cloud run for `layered_slab`. Export feasibility is clean (the slab uses simple
  lossy `eps_r + sigma_e` media, already covered by the adapter's `sigma_e_drude_slab`
  / `debye_slab` media-family exports; `mass_density` is postprocess-only and stripped
  from the cache key). It is deferred because: (a) the binding evidence is already the
  independent, monotonically-converging conservation-law closure (wave-level); (b) the
  only scene whose *absorbed-power field* an external solver could naturally
  cross-check driven is `antenna_near_phantom`, which is blocked upstream; and (c) the
  single owner-authorized run is better reserved than spent duplicating a conservation
  check the closure already provides. Recorded for the supervisor to spend if desired.
- **`input_power` normalization census guard — assessed, not removed.** Removing the
  `PowerNormalization.input_power` fail-closed guard requires wiring a total injected
  source-power diagnostic from the conservation-suite machinery. That is a real
  capability addition (not a clean, incidental change), so per the "only if clean"
  instruction it is left fail-closed and the census stays 175. `accepted_power` and
  `source` normalization remain fully supported.
- **Peak 1 g SAR grid convergence.** Documented as grid-sensitive rather than tightly
  convergent (see the three-grid study above); this is an honest finding, not a gap in
  the gate — the conservation closure is the convergent wave-level observable.

## Files added / changed (H3b)

Added: `benchmark/sar_validation.py`, `tests/sar/test_phantom_convergence.py`,
`tests/sar/test_sar_validation.py`,
`docs/assessments/sar-phantom-validation/*.json` (4 artifacts).
Changed: `benchmark/scenes/sar/layered_slab.py` (`build_conservation_scene`),
`benchmark/__main__.py` (`sar` subcommand), `benchmark/RESULTS.md`
(`## SAR exposure validation` section), `FEATURE_LIST.md` (H3b subsection), this doc.

No existing fail-closed guard or test was removed or weakened.
