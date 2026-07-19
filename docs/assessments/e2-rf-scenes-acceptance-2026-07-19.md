# Track E2 (rf-scenes) acceptance — 2026-07-19

Worktree: `.worktrees/we2-rf-scenes` (branch `fable/rf-scenes-validation`).
Env: `maxwell`; `CUDA_VISIBLE_DEVICES=1`. Reproduce every run with the common-brief
CUDA_HOME / PYTHONPATH pattern (PYTHONPATH = this worktree).

This document is appended stage-by-stage. Stage E2b (antenna benches) and E2c (M3
cache generation + RESULTS.md rows + cloud task ids) extend it below.

---

## Stage E2a — PlaneMonitor plumbing + open/short/match rebuild + RLC rebuild

Commits:
- `49ba300` feat(rf): thread user monitors through WavePort/PortSweep Results
- `923c85a` feat(rf): rebuild open/short/match and RLC benches on the coax line

### 1. PlaneMonitor passthrough (deliverable 1)

Root cause: both WavePort Result assemblies (`witwin/maxwell/waveport_sweep.py`
`run_waveport_sweep` / `run_waveport_excitation`) built the final `Result` with
`monitors={}`, and `compact_array_column_result` retained only closed-surface
payloads — so user-declared monitors were computed in every column run and then
discarded.

Fix:
- `WAVEPORT_MONITOR_PREFIX` marks the internal per-port ModeMonitors; a
  `_user_monitor_payloads(result, scene)` helper keeps only user monitors.
- A direct `PortExcitation` (single drive column) carries its user monitors
  through unchanged (identical to a plain FDTD run of the injected mode).
- A `PortSweep` exposes the first drive channel at the flat top level (recorded in
  `Result` metadata `user_monitor_drive_channel` / `user_monitor_frequency`) and
  every drive column in `array_run_data.column_results` (`compact_array_column_result`
  widened to retain all non-internal monitor payloads).

Evidence (coax `guide_plane` PlaneMonitor, dx=0.01, 1 GHz):
- WavePort excitation: annulus mean |Ey| = 75.7, PEC inner mean = 0.0, outside
  shield mean = 0.0 (field is the injected TEM mode, exactly zero in the PEC).
- Magnitude-profile correlation WavePort vs an independent plain `ModeSource`
  launch over the annulus = 0.853.
- PortSweep: top-level monitor present (`drive = left::TEM0`); 2 columns, each with
  the monitor in `column_results`.

Tests: `tests/rf/wave_validation/test_planemonitor_waveport_passthrough.py` — 5
passed. Falsification (embedded): monkeypatch `_user_monitor_payloads -> {}`, then
`result.monitor("guide_plane")` raises `KeyError` (recorded green in-test).

### 2. lumped_open_short_match rebuild (deliverable 2)

Rebuilt as `benchmark/scenes/rf/lumped_open_short_match.py::coax_sol_scene`
(standard in {matched, short, open}) on the proven coax line: TEM WavePort feed →
de-embedded load plane. matched = reflectionless coax-through-PML (presents Z0);
short = PEC plug; open = truncated inner rod (shield continues; operated below the
outer-guide TM01 cutoff 0.717 GHz so the open does not leak). PML raised to 12
layers so the sub-GHz matched load reaches −20 dB.

Measured (dx=0.01, 0.45 & 0.50 GHz):
- matched |Γ| max = 0.073 (−23 dB) ≤ 0.1 gate;
- short |Γ| min ≈ 0.998, open |Γ| min ≈ 0.998 (≥ 0.944 = −0.5 dB);
- open/short phase separation ≈ 113–120° (> 90°); short-referenced open Re ≈
  0.40–0.50 (+1 class); short is the −1 reference (documented SOL convention);
- open-end fringe extension ≈ 5.5 cm (the measured, documented departure of the
  open/short separation from an ideal 180°).

Phase-reference convention: the short defines the −1 load-plane reference; the open
then lands in the +1 class. Documented in the scene docstring.

Tests: `tests/rf/wave_validation/test_open_short_match_wave_level.py` — 6 passed.
Falsification (recorded, break→red→restore): forcing `coax_sol_scene` to always
build the `matched` standard turned short/open into ~0.07 reflection and collapsed
the discrimination — `test_short_and_open_totally_reflect`,
`test_short_open_discriminate_by_phase`, `test_three_standards_are_mutually_distinct`
all went red; restored → green. Plus an in-test falsification: feeding identical
Γ (the decoupled-bench signature) into the discrimination metric fails the gate.

### 3. series_parallel_rlc rebuild (deliverable 3)

Rebuilt as `benchmark/scenes/rf/series_parallel_rlc.py::series_rlc_scene`
(`parallel=` flag): the RLC is an IN-LINE two-terminal `LumpedPort` element in the
coax inner conductor (2-cell axial gap, voltage path along the line, current
contour encircling the rod) ahead of a matched through-PML continuation, so it
carries the full axial line current. The single-spoke SHUNT termination was tried
first and rejected (it barely coupled to the symmetric coax mode — the notch was
stuck near the TE11 artifact and ignored C, the same failure mode as the retired
bench); the in-line element tracks C cleanly.

Measured (dx=0.01, L=25 nH; C = 3.2 / 4.0 / 4.8 pF = nominal ±20%):
- series |S11| notch (interp): 0.4787 / 0.4355 / 0.4000 GHz; `f_res*sqrt(C)`
  spread = 0.0097 (~1% → tracks 1/√(LC)); ±20% ratios [1.099, 1.000, 0.919] vs
  analytic [1.118, 1.000, 0.913] (within 2%); absolute notch = 0.865 × ideal f0
  (the consistent, documented ~13% parasitic downshift from the rod-gap fringe C);
- parallel |S11| peak (interp): 0.3616 / 0.3465 / 0.3324 GHz — monotone in C
  (correct direction); the slope is diluted because the fringe C adds directly to
  the parallel C (documented).

Tests: `tests/rf/wave_validation/test_rlc_resonance_wave_level.py` — 6 passed (the
retired strict-xfail open-gap test is replaced by real passing tracking gates; the
fast companion-impedance analytic-identity contract test is retained, non-gating).
Falsification (recorded, break→red→restore): forcing the scene to ignore the
requested C made `f_res` C-independent — `test_series_resonance_tracks_inverse_sqrt_c`
and `test_series_resonance_moves_correctly_under_20pct_c` went red (ratio 1.0 vs
1.118, monotone broken); restored → green. Plus an in-test falsification: a
constant `f_res` fails the `f_res*sqrt(C)=const` gate.

### Benchmark runners

`benchmark/rf_validation.py::run_lumped_open_short_match` and `run_series_rlc` are
rewritten to drive the rebuilt scenes and record the discrimination / tracking
metrics (status pass). `benchmark/scenes/rf/__init__.py` now exports
`coax_sol_scene` (was `lumped_one_port_scene`). RESULTS.md rows + external-reference
caches are E2c.

### Test commands and results (E2a)

```
python -m pytest tests/rf/wave_validation/                       # 30 passed, 6 xfailed (pre-existing te10 blocked-operator xfails)
python -m pytest tests/rf/wave_validation/test_planemonitor_waveport_passthrough.py   # 5 passed
python -m pytest tests/rf/wave_validation/test_open_short_match_wave_level.py          # 6 passed
python -m pytest tests/rf/wave_validation/test_rlc_resonance_wave_level.py             # 6 passed
python -m pytest tests/api/public/test_public_api.py tests/api/public/test_simulation_smoke.py tests/api/public/test_guard_census.py tests/monitors/observers/test_fdtd_observers.py  # 38 passed
python -m pytest tests/rf/waveport/test_waveport_sweep.py tests/rf/array/test_array_contracts.py tests/rf/antenna/test_result_antenna.py  # 39 passed
python -m pytest tests/rf/array/test_array_fullwave.py tests/api/public/test_guard_census.py  # 8 passed
```

Guard-census budget unchanged (176); no fail-closed guard added, removed, or
weakened. No FDFD tests touched.

### Known gaps / notes for E2b, E2c

- The matched SOL standard is realized by the reflectionless coax-through-PML
  termination (presents Z0), not a lumped resistor — documented; this reuses the
  proven coax_thru termination.
- The RLC bench operates at the dx=0.01 tier only (the current-loop half-extent is
  snapped to that grid via `_snapped_loop_half`; other dx are validated by the
  snap assertion but not swept). The parallel tracking slope is parasitic-diluted
  (direction-only gate) — an honest limitation, not hidden.
- E2c cloud caches: brief authorizes `rf/coax_thru`, the rebuilt
  `rf/lumped_open_short_match`, `antenna/half_wave_dipole`, `antenna/patch`.

---

## Stage E2b — FDTD antenna benchmark scenes + real `Result.antenna` gates

Env: conda `maxwell`, `CUDA_VISIBLE_DEVICES=1`, `PYTHONPATH=<worktree>`,
`CUDA_HOME=.../nvidia/cu13`. All numbers below are reproducible by the named
pytest node or scratch script (scratch scripts are throwaway, not committed).

### Delivered items

1. `benchmark/scenes/antenna/half_wave_dipole.py` — center-fed thin-wire
   half-wave dipole (node-bound `LumpedPort` gap feed) + `ClosedSurfaceMonitor`
   NF2FF box. Sized to `L = lambda/2` at `design_frequency`.
2. `benchmark/scenes/antenna/patch.py` — probe-fed rectangular microstrip patch
   on a FINITE grounded dielectric slab; exact-node integer-cell `GridSpec.custom`
   grid; NF2FF box in the homogeneous air exterior.
3. `benchmark/scenes/antenna/__init__.py` — exports both builders (honest status).
4. `tests/rf/antenna/test_antenna_benchmark_e2e.py` — real (non-monkeypatched)
   `Result.antenna` end-to-end gates for both scenes.
5. `FEATURE_LIST.md` additive subsection `e2b-rf-scenes`.

### Headline gate: half-wave dipole (real NF2FF path, no monkeypatch)

Config: `design_frequency=3.0e9`, `dx=1.25e-3`, domain ±0.05 m (grid 96^3),
8 PML, sweep `default_frequencies(3e9)` = {2.4,2.7,3.0,3.21,3.39,3.6} GHz,
Gaussian feed `fwidth=1.5e9`, 12 ns (6871 steps). Measured at the design
frequency (3.0 GHz):

| Gate | Target | Measured | Pass |
|---|---|---|---|
| E-plane `sin^2` correlation | >= 0.99 | 0.9957 | yes |
| Peak directivity | in [1.9, 2.4] dBi and within 0.3 of analytic 2.156 | 2.194 dBi | yes |
| Radiated-vs-accepted closure | < 0.08 | 0.0407 | yes |
| Radiation-resistance class | min(R) < 73 < max(R), a sample in [60,90] Ω | R sweep 19.6→88.0 Ω; 68.6 & 88.0 in band | yes |

Input reactance is a large positive delta-gap feed offset (X = +470→+148 Ω across
the band; electrical resonance X=0 sits above 3.6 GHz). Documented, not gated —
the radiation physics (pattern/directivity/power balance) is the binding
evidence. Node: `tests/rf/antenna/test_antenna_benchmark_e2e.py::test_half_wave_dipole_end_to_end_radiation_and_impedance` (PASS, ~76 s).

### Patch antenna: real pipeline PASS + documented physical GAP

Config: `patch_antenna_scene` (eps_r=2.2, h=3 mm, L=28 mm, W=34 mm, finite
ground/substrate +6 mm, probe inset −8 mm), `dx=1e-3`, grid 86×97×70, 8 PML,
freqs {4.4,4.8,5.2,5.6,6.0} GHz, 16 ns.

- PASS (pipeline): `Result.antenna(...)` runs end to end over the grounded
  dielectric slab and returns valid `AntennaData` — 6 air-exterior NF2FF faces per
  frequency, finite realized gain/directivity, `p_rad>0`, best-sample
  radiated-vs-accepted closure < 0.05 (measured min ≈ 0.005). Node:
  `...::test_patch_antenna_result_antenna_pipeline_is_valid` (PASS).
- GAP (physics, strict xfail): the probe on this thick finite-ground slab is
  reactance-dominated (`|Gamma| ≈ 1.0` across 3–6.4 GHz, R ≈ 0.4–3 Ω, X ≈
  +870→+366 Ω) and the pattern peaks off-broadside (θ ≈ 46–48°, broadside
  directivity NEGATIVE: −1.8 to −10.7 dBi). Cavity model predicts f_r ≈ 3.39 GHz;
  the structure never matches nor forms a broadside `TM010` lobe in band. Recorded
  as `...::test_patch_antenna_matched_broadside_gate` `xfail(strict=True)` (a
  future fix xpasses → strict-fails → forces closing the gap). Deferred to E2c
  (feed/ground redesign + external-reference cross-check).

### Falsifications recorded (dipole headline gates; `scratch/falsify_dipole.py`, one real run)

Each gate discriminates real physics from a regression (real → pass, perturbed → fail):

- `sin^2` correlation: REAL 0.9957 (pass); isotropic pattern 0.8142 (fail);
  `cos^2` (wrong-axis) pattern 0.3309 (fail).
- Directivity band: REAL 2.194 dBi (pass); isotropic 0.0 dBi (fail band).
- Power closure `< 0.08`: REAL 0.0407 (pass); 2× mis-scaled `p_rad` → 0.9186 (fail).
- Patch physical gate genuinely xfails on the real run (broadside never ≥ 5 dBi
  while matched < −10 dB), so the `strict=True` marker is exercised, not vacuous.

### Test inventory / commands

```
# CUDA, GPU 1, worktree PYTHONPATH, CUDA_HOME set
python -m pytest tests/rf/antenna/test_antenna_benchmark_e2e.py -q
#   -> dipole PASS (76 s); patch pipeline PASS + matched-broadside xfailed (56 s)
python -m pytest tests/api/public/test_public_api.py \
  tests/api/public/test_simulation_smoke.py tests/api/public/test_guard_census.py \
  tests/rf/antenna/test_result_antenna.py tests/rf/antenna/test_antenna_data.py \
  tests/rf/antenna/test_antenna_matching_block.py -q          # -> 47 passed
```

Guard-census budget unchanged at 176 (no witwin/maxwell source changed — only
benchmark scenes + tests added). No fail-closed guard added/removed/weakened. No
FDFD tests touched. The existing monkeypatched `test_result_antenna.py` is kept
(fast synthetic-surface kernel coverage); the new e2e file adds the real path.

### Known gaps / notes for E2c

- Patch matched-broadside `TM010` + `D >= 5 dBi` is the outstanding physics gate
  (feed reactance + small finite ground). E2c owns the redesign and the external
  cross-check for `antenna/patch` and `antenna/half_wave_dipole`.
- Scene builders return geometry/feed/NF2FF only; the benchmark `python -m
  benchmark` registration + cache/RESULTS wiring for the antenna family is the E2c
  M3 deliverable (not wired here).
- Dipole reactance offset is a thin-wire delta-gap feed characteristic; a de-embed
  or finer feed would recover X≈0 at resonance but was out of scope this stage.

---

## Stage E2c — M3 external-reference generation wiring + antenna RESULTS registration

Env: conda `maxwell`, `CUDA_VISIBLE_DEVICES=1`, `PYTHONPATH=<worktree>`,
`CUDA_HOME=.../nvidia/cu13`. Every number below is reproducible by the named
command or pytest node.

### Delivered items

1. `benchmark/rf_tidy3d_references.py` — rewritten from a pending-marker stub into
   a real adapter-driven generation harness: `attempt_reference(name)` builds the
   scene, exports it via `Scene.to_tidy3d`, gates on the export being physically
   runnable (>= 1 source), and — only if runnable — estimates the cloud cost,
   enforces the per-scene FlexCredit budget (`MAX_REFERENCE_COST = 2.0`), runs one
   cloud job, extracts monitors (`benchmark.runner._extract_tidy3d_monitors`),
   writes the `.h5` cache (`benchmark.cache.save_tidy3d_result`), and records the
   task id + cost. A non-runnable export or any cloud failure is recorded
   fail-closed as `reference: pending-generation` with the concrete reason. Writes
   per-scene JSON markers under `benchmark/cache/{rf,antenna}/` (regenerable;
   `.h5`/marker paths are gitignored) and a RESULTS.md section.
2. `benchmark/rf_validation.py` — registered the two FDTD antenna scenes as
   validation runners (`run_half_wave_dipole`, `run_patch`) driving the real
   `Result.antenna` path, and split the RESULTS writer by scene family so
   `antenna/*` owns an `## Antenna wave-level validation` section without
   overwriting `## RF wave-level validation`.
3. `benchmark/RESULTS.md` — new `## RF / antenna external reference generation`
   section (all four authorized scenes) and new `## Antenna wave-level validation`
   section (dipole + patch Maxwell metrics).
4. `tests/rf/wave_validation/test_rf_reference_generation.py` — M3 wiring gates
   (7 tests).
5. `FEATURE_LIST.md` additive `e2c-rf-scenes` subsection (+ corrected the stale
   "not yet wired" clause in the Tidy3D Benchmarking section).

### Blocker investigation (two candidate blockers; only one is real)

**Cloud credentials — NOT a blocker (verified logged in).** `~/.tidy3d/` does not
exist and no `TIDY3D_API_KEY` env var is set, but the account is authenticated by
another mechanism: a live cost estimate on a trivially-exportable vacuum + point
dipole scene returned **0.025 FlexCredits** (`web.Job(...).estimate_cost()`
succeeded). Cloud generation is therefore available; the brief's "logged in" claim
holds.

**Adapter capability — the real blocker (EXECUTED probe).** All four authorized
target scenes export through `Scene.to_tidy3d` with **`sources == 0`**:

| Scene | Excitation | Exported sources | Exported monitors |
|---|---|---:|---:|
| `rf/coax_thru` | `WavePort` TEM + `PortSweep` | 0 | 0 |
| `rf/lumped_open_short_match` | `WavePort` TEM + `PortExcitation` | 0 | 0 |
| `antenna/half_wave_dipole` | `LumpedPort` wire-gap feed | 0 | 6 |
| `antenna/patch` | `LumpedPort` probe feed | 0 | 6 |

The adapter's `_convert_source` (`witwin/maxwell/adapters/tidy3d.py:1098`) maps only
field sources (PointDipole, PlaneWave, GaussianBeam, ModeSource,
UniformCurrentSource, CustomField/CurrentSource) — there is no mapping for
port-driven (`WavePort`/`PortSweep`/`PortExcitation`) or lumped-feed
(`LumpedPort`) excitation, and `_convert_monitor` has no `ClosedSurfaceMonitor`
mapping. A source-less reference simulation cannot produce an S-matrix or a
radiation pattern (running it yields only numerical noise and wastes credits), so
the wiring refuses it at the runnable gate **before** any cloud cost.

Consequence: **0 / 4 references generated; 4 pending-generation; 0 FlexCredits
spent.** Per the brief ("if a cloud run fails for an adapter-capability reason,
record fail-closed honestly, pending-generation stays with the error") this is the
correct outcome. Mapping port/lumped excitation to the reference solver is a
separate adapter feature (deferred; overlaps the Track E1 mode-operator work).

### Cloud task ids / costs

No cloud jobs were submitted (all four refused at the runnable gate), so there are
no task ids and the cost is 0 FlexCredits. The cloud primitive itself was
live-verified once (the 0.025-credit estimate above) to prove the path is
functional, not the blocker.

### Antenna RESULTS metrics (real `Result.antenna`, GPU 1)

Command: `python -m benchmark rf antenna/half_wave_dipole antenna/patch`
(consistent with `tests/rf/antenna/test_antenna_benchmark_e2e.py`).

| Scene | Headline | Measured | Reference | Status |
|---|---|---:|---:|---|
| `antenna/half_wave_dipole` | broadside directivity | 2.194 dBi | 2.156 dBi (1.76%) | pass |
| `antenna/patch` | broadside directivity (in-band max) | −6.974 dBi | 5 dBi target | gap |

Dipole supporting: E-plane sin² correlation 0.9957, power closure 0.041, R sweep
through the 73 Ω class. Patch: pipeline valid (6 NF2FF faces/freq, p_rad>0,
closure_min < 0.05), off-broadside physics gap (matched-broadside TM010 remains a
strict xfail in the e2e test; cavity f_r ≈ 3.39 GHz sits below the 4.4–6.0 GHz run
band). Machine-readable artifacts:
`docs/assessments/rf-wave-validation-2026-07-18/antenna__{half_wave_dipole,patch}.json`.

### Tests + falsifications

```
python -m pytest tests/rf/wave_validation/test_rf_reference_generation.py     # 7 passed
python -m pytest tests/api/public/test_public_api.py \
  tests/api/public/test_simulation_smoke.py \
  tests/api/public/test_guard_census.py \
  tests/rf/wave_validation/test_rf_reference_generation.py                    # 37 passed
python -m benchmark.rf_tidy3d_references          # 0/4 generated, 4 pending (sources=0), 0 credits
python -m benchmark rf antenna/half_wave_dipole antenna/patch                 # dipole pass, patch gap
```

Falsifications (EXECUTED):
- **Runnable gate is load-bearing (break → red → restore).** Real gate:
  `attempt_reference('rf/coax_thru')` → `runnable=False, status=pending, sources=0`.
  Forcing `_runnable_reason` to always return `True` flips the same scene to
  `runnable=True` (it would then proceed to the cloud branch); restoring the gate
  returns `runnable=False`. Recorded live.
- **Wiring is not a vacuous always-pending stub.**
  `test_runnable_export_reaches_cloud_branch` forces the gate open with a stubbed
  cloud run and asserts `status=generated`, task id and cost recorded, cache
  written — proving the runnable → cloud → cache branch is reachable.
- **Cloud failure is fail-closed.** `test_cloud_failure_is_recorded_fail_closed`
  stubs the cloud run to raise and asserts `status=pending`, reason contains
  "cloud generation failed", and `save_tidy3d_result` is never called (no fabricated
  cache).
- **Source-less exports never fabricate a cache.**
  `test_target_scene_exports_source_less_and_fails_closed` (parametrized over all
  four targets) asserts no `.h5` is written and the cloud is never reached.

### Guard census

Budget unchanged at **176**: no `witwin/maxwell` source was modified (only
`benchmark/` and `tests/`). `tests/api/public/test_guard_census.py` green.

### Known gaps / deferred

- **Port/lumped excitation adapter export is the outstanding feature** blocking the
  four external-reference caches. A `WavePort` TEM port could in principle map to a
  `td.ModeSource` + `td.ModeMonitor` pair and a `LumpedPort` to the reference
  solver's RF lumped-port construct (which is now a licensed sub-package), plus a
  `ClosedSurfaceMonitor` → field-monitor-box mapping; all are non-trivial and were
  not attempted this stage. Until then the four references stay
  `pending-generation` with the reason recorded in `benchmark/RESULTS.md`.
- The patch matched-broadside TM010 physics gate (feed/ground redesign) remains
  open and is independent of the reference-generation blocker.
- RESULTS.md rows for `rf/coax_thru` / `rf/lumped_open_short_match` retain their
  E2a Maxwell metrics in the `## RF wave-level validation` section (not re-run this
  stage to conserve GPU time); their external-reference status is covered uniformly
  by the new reference-generation section.
