# Track F2 (RF trio) — F2c acceptance: adapter port/lumped mapping + four external caches

> Date: 2026-07-21. Track `f2-rf-trio`, worktree `.worktrees/wf2-rf-trio`, branch
> `fable/rf-trio`. GPU `CUDA_VISIBLE_DEVICES=1`. Conda env `maxwell`.
> Stage F2c. Prior stages: `docs/assessments/f2a-interior-pec-acceptance-2026-07-21.md`
> (interior-PEC masking + quasi-static line modes) and
> `docs/assessments/f2b-quasistatic-benches-acceptance-2026-07-21.md` (production
> microstrip/diff-pair benches + patch feed diagnosis). This doc covers F2c:
> adapter RF-port excitation mapping, the four owner-authorized external-reference
> cloud caches, the RESULTS rows, FEATURE_LIST, and the guard-census reconciliation.

## 0. Scope

Map the port excitations of the four reference scenes to external-reference-solver
constructs through the existing interoperability adapter, so the four previously
`sources=0` / `pending-generation` references become cloud-runnable; generate the four
caches (one authorized cloud job each, task ids + costs recorded); add the RESULTS rows;
keep the capability-guard census budget intact.

## 1. Delivered

1. **Adapter RF-port excitation mapping** (`witwin/maxwell/adapters/tidy3d.py`,
   `_convert_ports_for_reference` + helpers). Wired into `scene_to_tidy3d` so any
   port-driven `Scene` exports runnable:
   - **`WavePort` -> reference modal launch.** The first declared `WavePort` maps to a
     `ModeSource` of its fundamental (TEM/lowest) mode (`num_modes=1`, `mode_index=0`);
     every `WavePort` additionally maps to a receiving `ModeMonitor` at its aperture.
     S-parameters are read from the directional mode amplitudes.
   - **`LumpedPort` -> equivalent current injection.** A delta-gap feed (wire-bound via
     `scene.thin_wires` node lookup, or coordinate-bound) maps to a `UniformCurrentSource`
     electric-current filament spanning the feed gap along the negative->positive
     voltage-path axis (`current_amplitude_definition="total"`, unit reference current;
     zero transverse extent). Input impedance and pattern are amplitude-independent
     ratios, so the unit normalization cancels.
   - **NF2FF box** `ClosedSurfaceMonitor` already lowers to its six face field monitors
     through `Scene.resolved_monitors`, so the antenna near-field surface exports intact.
   - **Convention:** a single export drives port index 0; a full N-port scattering
     reference is one export per driven port. Ports carry no `source_time`; the adapter
     synthesizes a broadband Gaussian drive from the requested frequency band.
     `TerminalPort` is out of adapter-drive scope for now (a TerminalPort-only scene keeps
     the existing `sources=0` runnable-gate fail-close; not a new guard).
   - Every mapping decision is documented in the adapter module docstring block and pinned
     by `tests/api/adapters/tidy3d/test_port_source_mapping.py`.

2. **Four external-reference caches generated** (`benchmark/rf_tidy3d_references.py`).
   The four targets now export with `sources>=1` and were each cloud-generated with one
   authorized job (owner-authorized program spend; smallest honest exported grid; the
   reference solver auto-meshes). Task ids + costs in section 4. A new `--from-markers`
   CLI mode + `rebuild_from_markers()` / `load_marker()` rebuild the RESULTS aggregate
   table from on-disk markers with no cloud call (used to re-aggregate per-scene runs
   without re-spending; the `rf/rectangular_waveguide` row is preserved from its
   round-E marker rather than re-run).

3. **RESULTS rows** (`benchmark/RESULTS.md`, section
   `## RF / antenna external reference generation`): all five rows now `generated` with
   task ids + costs, regenerated via `--from-markers`.

4. **Docs**: this acceptance doc; `FEATURE_LIST.md` new subsection
   `f2c-adapter-port-mapping`; `docs/reference/rf-wave-validation-2026-07-18.md` sections
   1.3 / 4 / 5 updated (the four scenes are mapped and generated, superseding the prior
   "deferred adapter feature / sources=0" wording).

## 2. Test inventory (executed)

Environment: `conda run -n maxwell`, `CUDA_VISIBLE_DEVICES=1`, `CUDA_HOME` +
`PYTHONPATH=<worktree>` exported.

| Suite | Result |
|---|---|
| `tests/api/adapters/tidy3d/test_port_source_mapping.py` (new) | 6 passed |
| `tests/rf/wave_validation/test_rf_reference_generation.py` (updated + 1 new test) | 9 passed |
| `tests/api/adapters/tidy3d/` (full dir, regression) | included in 177-passed run |
| `tests/api/public/test_guard_census.py` | passed (budget 176, unchanged) |
| `tests/api/public/test_public_api.py` | passed |
| `tests/api/public/test_simulation_smoke.py` | passed |

Combined adapter-dir + public trio + census run: **177 passed**. Port-mapping +
reference-generation run: **15 passed** (later 9 in reference-generation after adding the
marker-rebuild test).

## 3. Falsifications (executed)

Recorded per the evidence-discipline requirement (temporarily break, observe red, restore,
observe green). Scratch edits under `scratch/` were reverted; not committed.

1. **Port mapping is load-bearing.** Short-circuiting `_convert_ports_for_reference` to
   `return [], []` (the pre-F2c `sources=0` state) turns 5 of the 6 port-mapping tests
   RED (the four export assertions + the frequencies-required gate, which is only reached
   through the port path). Restored -> 6 passed.
2. **Current-filament orientation follows the voltage-path axis.** Forcing the filament
   axis to a fixed `axis_idx = 0` (instead of `argmax|positive-negative|`) turns the
   dipole/patch/orientation tests RED — the dipole degenerates (zero gap along x trips the
   coincident-terminal guard) and the patch filament lands on the wrong component.
   Restored -> 6 passed.
3. **Runnable-gate discrimination (pre-existing, retained).** With the runnable gate not
   monkeypatched open, `run_cloud=False` records `pending-generation` with a "suppressed"
   reason and never reaches the cloud stub — proving the runnable branch is reached only by
   a genuine source (`test_target_scene_exports_runnable_with_a_source`,
   `test_runnable_export_reaches_cloud_branch`,
   `test_cloud_failure_is_recorded_fail_closed`).

## 4. External-reference cloud runs (owner-authorized, executed)

Command per scene: `python -m benchmark.rf_tidy3d_references <scene>`. Budget ceiling 2.0
FlexCredits/scene. Each is a single cloud job on the smallest honest exported grid.

| Scene | Task id | Cost (FlexCredits) | Exported sources / monitors |
|---|---|---:|---|
| rf/coax_thru | `fdve-80e800c3-76e0-42fd-ad74-31fd32cbc3fe` | 0.025 | 1 ModeSource / 2 ModeMonitor |
| rf/lumped_open_short_match | `fdve-7b95dafe-9149-44f1-9038-3c7bfc09df20` | 0.025 | 1 ModeSource / 1 ModeMonitor |
| antenna/half_wave_dipole | `fdve-3e798959-93ac-4db1-b30e-ced1106f0af5` | 0.025 | 1 UniformCurrentSource / 6 FieldMonitor |
| antenna/patch | `fdve-0f3e5d2c-2404-475a-8ab7-b24e858bc34e` | 0.025 | 1 UniformCurrentSource / 6 FieldMonitor |

New F2c cloud spend: **0.1 FlexCredits total** (4 x 0.025). The round-E
`rf/rectangular_waveguide` reference (task `fdve-3c2a2d95-4809-4dfb-98d6-1b6b5416c39a`,
0.025 FlexCredits) was NOT re-run; its RESULTS row is preserved via a reconstructed
on-disk marker carrying the committed task id + cost (the `.h5` cache was generated on
main and is gitignored). Caches (`benchmark/cache/**.h5`) and markers
(`benchmark/cache/**.json`) are gitignored per benchmark convention; `benchmark/RESULTS.md`
is the committed record.

Reproduce the RESULTS aggregate without cloud spend:
`python -m benchmark.rf_tidy3d_references --from-markers`.

## 5. Capability-guard census

Budget unchanged at **176**. The adapter mapping adds no `NotImplementedError`: a valid
`LumpedPort` always carries terminal coordinates by construction (wire-bound or
coordinate-bound), so `_resolve_lumped_terminals` needs no capability guard, and
`TerminalPort` is simply not collected as a drive (its scenes keep the existing runnable-gate
fail-close). `tests/api/public/test_guard_census.py` green (177-passed combined run).

## 6. Known gaps / handoff

- The caches are **generated cross-reference artifacts**; the RESULTS rows record the
  honest generation outcome (sources, monitors, task id, cost). The binding first-line gate
  remains the analytic transmission-line / waveguide / dipole reference. Wiring a
  per-scene numeric Maxwell-vs-reference comparison into each runner (as done for the
  waveguide in round E) is a separate follow-on; the caches now exist to support it.
- The **patch** reference was generated but the patch scene itself does not resonate at
  feasible resolution (F2b diagnosis; the matched-broadside TM010 gate stays a fail-closed
  strict xfail). The cache is an honest export of the current (non-resonant) design.
- `TerminalPort` adapter-drive mapping is out of current scope (documented; not a guard).
