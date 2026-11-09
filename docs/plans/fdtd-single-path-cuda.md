# FDTD single-path CUDA execution

## v0.6.0 default policy (2026-11-05)

Perf gate (a) retains both implementations and defaults `Simulation.fdtd` and `FDTDConfig`
to `fdtd_core_path="auto", accelerated_setup="auto"`. This supersedes the pre-release
opt-in policy below; removal of flags and reference runtime routes remains post-release work.

- `True` retains fail-closed core admission, including the existing error for
  `initialize=False`. `False` retains existing stepping and never warns.
- Auto uses the same `core_path_blockers` and configuration blockers as explicit admission,
  before and after initialization. Resume, non-initializing preparation and material-cache
  overrides select existing execution. Setup is independently dispatched per eligible piece;
  auto setup is disabled for resume, non-initializing preparation and material-cache overrides.
- Automatic admission/activation errors and whole-step capture errors (including dispersive
  errors and exceptions from `run_chunks`) rebuild a fresh existing-path solver. Preparation
  restores the incoming mode gauges and optimization timing before retrying. Fields, CPML/ADE
  histories, sources, observer accumulators and device clocks belong to the new preparation;
  no failed solver state is reused. Explicit-True capture behavior stays unchanged.
- Result metadata and `Result.stats()` report `execution_path` (`core` or `existing`) and
  `fallback_reason` (None, first blocker, or exception class/message). Public
  `FDTDCoreFallbackWarning` warns once per Simulation object, including across repeated
  prepare/run calls. Resume switches to existing execution before checkpoint preflight.
- `tests/fdtd/core_path/test_default_auto.py` covers CPU admission/validation, warning lifetime,
  setup controls, clean reconstruction and resume preflight with CUDA lazy initialization
  forbidden. Marked CUDA tests require exact field/monitor equality for PMC fallback, resume,
  ordinary/dispersive capture failures after mutation, eligible auto versus explicit True,
  and three repeat runs. These GPU checks remain pending; implementation was CPU-only with
  installed `witwin.core` 0.4.6. `runtime/stepping.py` and ports are unchanged.
- Core benchmark baselines now explicitly select both controls False. Setup-only comparisons
  explicitly disable the core so changing setup controls still isolates setup work.
- CPU receipt: core-path tests with `-k "not cuda"` pass (301 passed, 65 skipped,
  22 deselected; rc 0). Resume-identity and integration-batch checks pass (82 passed,
  three CUDA skips; rc 0); the integration TOML parses. The CPU public subset forbids
  CUDA lazy initialization and excludes the CUDA-only scene serialization module:
  129 passed, 23 skipped, four failed and three fixture errors (rc 1). Three stale
  fake-solver failures also reproduce with the original HEAD simulation implementation;
  the remaining failure/errors reference missing preserved 0.4.5 installation metadata
  after the shared environment upgrade. Provenance pins and the reviewed capability
  manifest are unchanged; the capability census passes in the final public run.

## User decision

2026-11-03 (relayed in full to the coordinator): the accelerated CUDA paths become **the only execution
path**. Opt-in flags (`fdtd_core_path`, `accelerated_setup`) are removed, and the slow torch/CPU
production routes they replace are deleted. New features are developed CUDA-kernel-first. Where a feature
is under development, its owner and the core-CUDA line design the kernel and its fusion together.

Scope and ordering (user answers, 2026-11-03):

- The CPU backend remains only as a test oracle. It is not a supported runtime path.
- Torch reference kernels leave the runtime and stay as frozen, test-only oracles: the thin-sheet torch kernel
  (`WITWIN_MAXWELL_THIN_SHEET_KERNEL=torch`), `copper_torch_kernel` and the per-component Yee path.
- Torch that defines autograd/adjoint stays until a CUDA path with qualified gradients exists.
- v0.6.0 is released first. W7 (flag removal and deletion of the replaced runtime routes) is the first
  post-release item. Until then W1-W6 and the S items land opt-in, and no runtime path is deleted.

Every switch is gated by `torch.equal` G1/G4 against the existing path, plus a default-path BITID A/B over
all `test_*bit_identity*.py` files.

## 1. Step architecture

Today the core path replaces only the Yee launch pair inside `update_magnetic_fields` /
`update_electric_fields` (`fdtd/runtime/stepping.py`). Everything else in `_field_update_block` runs around
those calls. O2a's whole-step K-chunk graph (`CorePathRuntime.run_chunks`) builds its own step closure,
which today contains only Yee, TFSF, sources, PEC, DFT and observers.

The single-path step is one closure, in **exactly** the order of `_field_update_block` and
`_port_step_block` (the order PCB confirmed):

1. `begin_copper_step`, `begin_thin_sheet_step`, the modulation clock.
2. Magnetic dispersive advance, then the gyromagnetic drive snapshot.
3. **Fused H**, plus the imaginary H for complex fields.
4. `correct_copper_magnetic`, then `advance_thin_sheet_magnetic`.
5. TFSF H correction and the auxiliary H line, then magnetic surface sources.
6. Magnetic dispersive corrections, the gyromagnetic coupled step, then wire sample/update.
7. Dispersive advance, the nonlinear coefficient update, then the aniso conduction capture.
8. **Fused E**, or the Bloch E variants.
9. Full-aniso corrections/conduction, `deposit_wire_current`, then `apply_surface_impedance`.
10. `advance_copper_electric`, `advance_thin_sheet_electric`, then `accumulate_copper_spectra`.
11. After the block, in solve_steps order (PCB review, 2026-11-03):
    - (a) TFSF E correction, then `advance_tfsf_auxiliary_electric`;
    - (b) `inject_electric_surface_source_terms` at `injection_time("equivalent_electric_current")`, i.e. n+½;
    - (c) `add_source` at n+½ (additive J).
12. Port phase:
    - `apply_port_runtimes` (drive table);
    - fed thin sheets' `sheet.electric` (only sheets with feeds; these skip `advance_thin_sheet_electric` in the block);
    - networks;
    - the **post-source tail**: `_apply_dispersive_corrections` → `enforce_pec_boundaries` (skipped with TFSF) →
      `apply_mur_boundaries` → `apply_cut_surface_constraints` → field DFT;
    - port V/I observers.
    PEC clamps, Mur, cut-surface constraints and the DFT therefore come **after** ports and networks.
13. `advance_breakdown_state`, spectral/field observers, time and breakdown observers, then the shutoff check.
    The shutoff `.item()` sits only at K-chunk boundaries, never inside a graph.

The network variants are **not** interchangeable, and the closure keeps the one each scene uses today:

| Scene class | Port phase today |
| --- | --- |
| No sheet feeds, eager ports | `apply_port_runtimes` → `run_network_updates()` → `post_source()` → `run_port_observers()` |
| Sheet feeds, or the captured port graph | `_port_step_block` → `apply_network_runtimes(native_lu=True)` |

Capture safety:

| Piece | Status |
| --- | --- |
| Copper fused kernels (P-9(c)), D3b sheet kernels | K-graph safe; the chord solve allocates in the graph pool |
| Ports | Graph-safe only under `_make_port_step_runner`'s conditions: no circuits, no embedded networks, GPU DFT. Circuits and networks run per step or eagerly |
| Wire sample/deposit, SIBC, cut-surface constraints | Must be checked for host syncs before W2 |
| Breakdown | Mutates coefficients (W3) |

G1 additions for W2: a cut-surface scene, and a scene with both a lumped port and a CopperFoil sheet feed,
so that the `_port_step_block` sheet-feed branch is covered.

Rules:

- **Admission is per piece, not per scene.** A piece either has a fused kernel or calls its existing
  function in the same position. A K-step graph is captured when every piece in the closure is capture-safe
  (no per-step host input). Otherwise the same closure is captured per step or runs eagerly. There is no
  separate legacy loop.
- **Coefficient-mutating features** (nonlinear, modulation, breakdown) invalidate the core coefficient table
  built at prepare. The fused kernel must read their live coefficients until they are ported (W3).
- **Today's interim guard:** if core eligibility is widened before W2, `run_chunks` must return `None`
  (per-step closure) for any scene whose pieces it does not contain. Otherwise it would silently skip
  copper, sheet or wire work.

## 2. Work items, by measured cost and owner

| Id | Item | Owner(s) | Gate |
| --- | --- | --- | --- |
| W1 | Widen fused-Yee eligibility to features that do not mutate coefficients and act outside the Yee pair: copper, thin sheets, wires, SIBC, ports, dispersive ADE state, full aniso; closed surfaces are done (O2d). Each feature is a G1 scene. | core + PCB (copper/sheets/ports/wires) | G1 K=1 per feature |
| W2 | The whole-step closure above, captured per K | core + PCB | G1 K=32/16/1 on copper_patch, copper_board_60k, the F3 printed dipole, coax + lumped port, and a wire scene |
| W3 | Nonlinear, modulation and breakdown: live coefficients in the fused kernel | core + gpu-memory-cuda-attribution (modulation) | G1 |
| W4 | F5 Bloch: device CW rows, complex grouped observers, complex fused E/H | core + maxwell-bench-dataset-reuse (semantics contract after r3) | G1 or user-approved tolerance |
| W5 | `pml` / graded `absorber` fused (refused today: ~9% G1 mismatch), plus the passivity-guard hook at the shutoff cadence | core + maxwell-bench-dataset-reuse | G1; must-trip/no-trip |
| W6 | Dispersive ADE on compact per-material edge lists (coverage's (a) layout) | core + boolean-geometry-stl-coverage | G1/G4 |
| W7 | Flag removal and deletion of replaced routes | core | full BITID A/B; after user answers |

Setup side:

| Id | Item | Owner(s) |
| --- | --- | --- |
| S0/S1 | `accelerated_setup`, wider primitives (pending merge) | core |
| S2 | Flags/uniformity reductions, observer resolution, O(T) host tables (after (A)) | core |
| S3 | Thin-sheet/copper/port/cut-surface setup (`pcb/setup-thin-sheet`, `pcb/setup-copper` in flight) | PCB |
| S6 | Mesh winding on GPU: bitwise winding sum order, or a user-approved tolerance | core |
| S7 | Route `Scene.compile_materials()` (FDFD, waveport sweep) through the accelerated compiler | core |

### W5 notes

2026-11-03: the first absorber mismatch comes from the final electric update's FMA
contraction order, exposed by nonunit decay, rather than omitted absorber state.
The old `witwin/maxwell/fdtd/cuda/core/yee.cu:146` expression compiled to
`fma(curl, factor, round(E * decay))`. The standard electric updates in
`witwin/maxwell/fdtd/cuda/kernels/electric.cu:329` and `:385` compile to
`fma(E, decay, round(factor * curl))`, including their uniform variants at
`:327` and `:383`. These expressions differ in float32. The historical G1
maximum relative errors alone do not distinguish missing physics from rounding
near small reference values; the old G4 comparison passed all 2088 setup items.

The existing absorber semantics are already represented by core coefficients:

- `witwin/maxwell/fdtd/boundary/runtime.py:164` selects `pml`/`absorber` with
  `uses_cpml=False`. `boundary/cpml.py:509` builds the quartic simple-PML sigma
  profile; `:532` builds the calibrated cubic graded profile. Both clear CPML
  auxiliary state; neither introduces split-field state or a per-step damping
  kernel.
- `witwin/maxwell/fdtd/runtime/materials.py:1770` averages sigma onto electric
  edges, and `:1785` starts the magnetic averages. The resulting decay multiplies
  both field decay and curl coefficients (`:1055`, `:1086`). Uniformity is stored
  after these folds (`:1835`). Core's `_compress_coefficients`
  (`runtime/core_path.py:103`) preserves the full padded coefficient bit patterns.
- `witwin/maxwell/fdtd/runtime/stepping.py:319` and `:837` dispatch these absorbers
  to standard H/E. PML face code 1 leaves outer tangential E inactive in both
  `cuda/kernels/electric.cu:50` and `cuda/core/yee.cu:49`; periodic code 2 uses
  wrapped differences in both. `prepare_step`/`run_chunks` need no additional
  per-step absorber work.

The fix at `witwin/maxwell/fdtd/cuda/core/yee.cu:146` explicitly rounds the curl
product with `__fmul_rn` before `__fmaf_rn(E, decay, ...)` for non-CPML E only.
Core build flags remain unchanged and the extension ABI is now `core_abi_v3`.
The admission blocker and its rejection tests are removed; equality tests now
include `absorber_pml`, `absorber_graded`, and `closed_transverse_graded`, with
shape checks accommodating the latter's extended absorbing grid.

CPU-only validation passed: Python compilation, an offline sm_86 extension
build, and PTX analysis against both existing standard E launch variants for
all three components and both uniformity modes (12 specializations). Only the
three non-CPML electric core specializations change; the other 15 are identical.
A 10000-sample CPU arithmetic replay distinguishes the two contractions in
3493 cases. CUDA was not initialized; GPU qualification remains pending.
The passivity-guard portion of W5 remains
separate: this checkout has no existing field passivity hook at the shutoff cadence.

W5 second cause (2026-11-03): six-face simple/graded absorbers dispatch through
`runtime/stepping.py:837` and `:520` to the standard interior E kernels;
`cuda/kernels/electric.cu:1912` selects the Ez variant because both transverse
face pairs are inactive. Its derivative expressions at `:379` and `:380` compile
to separate PTX multiplies, but sm_86 SASS contracts the positive product into
the subtraction: `curl = fma(delta_x, inv_dx, -round(delta_y * inv_dy))`.
Core previously rounded both products before subtracting. This explains an
interior discrepancy even with unit decay; the final-combine fix alone cannot
remove it. Periodic transverse faces select the boundary-aware kernel and
retain their existing arithmetic. Core now mirrors that per-component dispatch
condition and explicitly contracts the interior curl, retaining the rounded curl
product and fused decay combine. Source injection remains a separate launch
after E, with `fma(signal, patch, E)` on both nonperiodic paths. The ABI is
`core_abi_v4`. Offline sm_86 compilation and PTX/SASS checks pass: only three
non-CPML electric specializations change; all 15 others are identical.
CPU replay from captured GPU setup/source tensors reproduces exactly the four
reported step-2 Ez cells and their `1.7763568394002505e-15` maximum difference.
GPU G1 qualification at K=32/16/1 and full-list K=32 regression remain pending.

## 3. Interim status (2026-11-04)

- Merged: O1, O2a, O3a, O3b, the Kottke-width mirror, the audit, the pml/absorber refusal (046136b3),
  O2d (675b199a) and O3c + S0/S1 (615ddf41). Default-path bit identity of these merges:
  `docs/assessments/core-path-default-ab-2026-11.md`.
- Merged since: W1 (a14425c9), source/monitor admission (e2974a3f), W5 absorbers (3dc74d5c),
  W2 whole-step K graph (188c3750), accelerated pole setup (9999c3f4), Rule 1/Rule 2 gate modes
  (4b674804), W6 phase 1 dispersive ADE (ec91ef74) and W3 live coefficients (146b395e, core ABI v7).
- In progress: the v0.6.0 performance gate (roofline ratios, byte-reduction ledger, setup share,
  then the default flip for eligible scene classes); W4; W7 after the release.

### Source/monitor qualification notes

2026-11-03, code inspection and CPU-only contract checks; GPU G1/G4 qualification remains pending.
Admissions below remove only type refusals. Other scene blockers still apply.

| Type | Class | Existing execution and core disposition |
| --- | --- | --- |
| `ModeSource` | A | `excitation/injection.py:_prepare_mode_surface_source` solves the profile at setup and builds electric/magnetic surface terms. `core_tables.compile_source_tables` retains their patches, delays, CW quadratures and ordered launches. Admit. |
| `CustomFieldSource` | A | `_prepare_custom_field_source` builds equivalent J/M surface terms, including real/quadrature delayed channels. The generic source-table compiler already handles these terms. Admit; delayed custom waveforms retain the existing per-step fallback when `source_tables_supported` declines them. |
| `CustomCurrentSource` (electric) | A | `_prepare_custom_current_source` puts sampled J patches in `_source_terms`; ordinary core injection runs after E at n+1/2. Admit. |
| `UniformCurrentSource` | A | `_prepare_uniform_current_source` builds ordinary additive-J patches; the same core ordinary table route handles them. Admit. |
| Magnetic current | A | `magnetic_current_vacuum` and `pmc_cavity` use `CustomCurrentSource` with `CurrentDataset` Mz, not a separate source class. Preparation puts M patches in `_magnetic_source_terms`. Admit the source; PMC remains blocked independently. |
| `AstigmaticGaussianBeam` | A | `astigmatic_beam` uses this separate class, not `GaussianBeam`. `_prepare_power_normalized_surface_source` shares the electric/magnetic profile-term route with the circular beam; TFSF terms also use the existing generic compiler. Admit. |
| `ModeMonitor` | A | `compiler/monitors.py:_compile_plane_observers` emits ordinary plane groups with `mode_spec`; `observers.py:BatchObserverEngine` accumulates those groups and ordinary result/postprocess code performs modal projection. Admit. |
| `DiffractionMonitor` | A | The same compiler emits plane groups with `diffraction_spec`; grouped sampling and result-time order decomposition require no new step work. Admit. |
| `PermittivityMonitor` | A | Skipped by the running-observer compiler; material sampling occurs during Result export from compiled material tensors. No per-step work to omit. Admit. |
| `FieldTimeMonitor` | B | `compile_fdtd_time_observers` and `observers.py:accumulate_time_observers` maintain a host sample schedule and copy raw samples after the field/spectral tail. Admit only with `run_chunks` returning `None` whenever `time_observers_enabled`; retain the existing per-step loop and fused Yee pair. |
| `FluxTimeMonitor` | B | Also present in `time_monitor_vacuum`; shares the time-observer flag/schedule and computes instantaneous flux in that tail. Admit with the same guard. |
| PMC boundary (independent of source type) | C | Keep refused. `cuda/core/yee.cu:backward` returns zero on PMC end faces, whereas `cuda/kernels/electric.cu:boundary_diff_low/high` returns +2*H_low*inv_delta / -2*H_high*inv_delta. Implement these PMC ghost derivatives for every electric component/axis and CPML layout, then qualify the magnetic cavity with G1/G4 before admission. |

Magnetic injection is after the fused H update and before E, exactly as in
`runtime/stepping.py:_field_update_block`. The legacy field-block graph refuses magnetic terms because
their waveform scalar would freeze at capture time. The core closure instead reads a device-indexed
row from `compile_source_tables`: `injection_time("magnetic_current", n*dt, dt)` is n*dt;
ordinary J and equivalent electric surface terms use n*dt + 0.5*float(dt), preserving multiply-then-add
order. Scalar rows are cast to float32 just as the legacy native scalar argument is; delayed rows use
the existing float32 waveform expressions, and CW channels preserve the two-product expression.
The core patch kernel adds into the same target fields in term-list order. This is an arithmetic/order
inspection, not a claimed GPU bit-identity result. If source tables are unavailable, `run_chunks`
already returns `None`, preserving the host waveform evaluation around the fused Yee pair.

The existing `(observers_enabled and _single_observer_engine is None)` guard likewise preserves host
spectral observers. The explicit time-observer guard applies even when a grouped spectral engine also
exists, so a mixed spectral/time scene cannot silently lose raw samples.

`benchmark/core_path/scenes.py:SRCMON_SCENES` registers all 16 census scenes through the original
benchmark builders, including the independently blocked `pmc_cavity`. G1 capture accepts the expected
per-step execution for host/time observers. Run G1 at K=32/16/1 and G4 over the 15 admitted scenes;
check PMC as an expected refusal until its boundary work lands. Commands are recorded in
the source/monitor handoff's `GPU_TODO.md`. No GPU was used for this change.

### W1 notes

2026-11-03, W1 part 1: code and CPU-only checks; no CUDA initialization or GPU qualification.
Here (a) means the Yee decay/curl arrays and six boundary codes remain unchanged **after prepare**;
(b) means all feature step work stays outside `update_magnetic_fields` / `update_electric_fields`.
Both dispatchers replace only the Yee launch (`witwin/maxwell/fdtd/runtime/stepping.py:315` and `:823`).
Activation follows `init_field`, initial conditions and time-plan resolution
(`witwin/maxwell/simulation.py:1863`, `:1872`), so prepare-time coefficient changes are included.

All paths below are under `witwin/maxwell/`:

| Feature | (a) / (b) | Evidence and admission decision |
| --- | --- | --- |
| Lumped ports and elements, coaxial ports, TerminalPort | yes / yes | `fdtd/ports.py:686` restores declared PEC-gap decay/curl values **during preparation** and invalidates their uniformity entries at `:719`; `:817` prepares runtimes. `:2177` applies port/load corrections to fields and private branch/drive state; standalone elements and circuits follow at `:2226`. No step-time Yee coefficient or boundary-code writes. Remove ports and lumped-elements blockers. |
| WavePort and TerminalWavePort carriers | yes / yes | `waveport_sweep.py:606` and `:621`, and `terminal_waveport.py:355` lower columns to ModeSource/ModeMonitor and remove their original port declarations. `fdtd/excitation/injection.py:1346` prepares fixed mode patches; `:1408` appends ordinary electric/magnetic surface terms. Injection remains around the Yee pair (`fdtd/runtime/stepping.py:1908`, `:2544`); mode monitors compile to plane observers (`compiler/monitors.py:124`, `:153`). Admit ModeSource/ModeMonitor as well, and guard lowered scenes even when `scene.ports` is empty. |
| CopperFoil | yes / yes | `fdtd/copper_foil.py:165` reads prepared Yee coefficients and creates independent trace/response state; `:300`, `:322`, `:332` snapshot, correct H and advance E without changing Yee coefficients or boundary codes. Calls remain at `fdtd/runtime/stepping.py:1877`, `:1902`, `:1949`, `:1951`. Remove runtime and structure-level CopperFoil blockers. |
| Thin sheets, including conformal_two_sided | yes / yes | `fdtd/thin_sheet.py:355` tightens dt before material/boundary coefficient construction (`fdtd/runtime/stepping.py:1746`); `fdtd/thin_sheet.py:510` reads native coefficients into private matrices. `:835`, `:838`, `:848` update private patch state and scatter fields; `:993`–`:1007` place the calls around the pair. Remove runtime and structure-level conformal blockers. |
| CopperFoil sheet feeds | yes / yes | `fdtd/thin_sheet.py:1005` skips fed sheets in the field block. `fdtd/ports.py:2180` copies port drive into the feed; `fdtd/runtime/stepping.py:2142` applies ports then fed-sheet E, networks, post-source tail and V/I observers. No coefficient/code mutation. The guard retains this exact branch. |
| Thin wires | yes / yes | `fdtd/wire.py:818` prepares topology and dt-dependent private coefficients before base Yee coefficients are built (`fdtd/runtime/stepping.py:1746`). `fdtd/wire.py:1041` samples E and updates current/charge; `:1075` deposits current into E. The calls surround fused E at `fdtd/runtime/stepping.py:1917` and `:1946`. Remove wires blocker. |
| PEC cut surfaces | yes / yes | `fdtd/cut_surface.py:80` prepares a fixed weighted projection and only reads curl values (`:152`); `:180` gathers/projects/scatters E. Post-source application remains after ports/networks, before DFT (`fdtd/runtime/stepping.py:2063`, `:2569`). Remove cut-surfaces blocker. Existing overlap/composition refusals remain owned by that runtime. |
| SIBC, including rational surface state | yes / yes | `fdtd/runtime/materials.py:1249` and `:1467` mask metal interior coefficients during setup; `:1335` prepares face writes and private ADE state. Uniformity is stored after masking (`:1767`, `:1835`). `fdtd/runtime/stepping.py:1607` overwrites tangential E/advances private ADE after the Yee update (`:1948`), without touching coefficients/codes. Remove SIBC blocker; this is separate from volume-ADE admission. |
| Circuits | yes / yes | `fdtd/circuits.py:964` prepares field-port coupling using existing curl values. `:593` assembles the circuit matrix and updates circuit/field state; this matrix is not the Yee coefficient table. `:1187` executes the circuit phase from `fdtd/ports.py:2240`. Host schedule/factor-class work stays in the existing per-step loop. Under the task's explicit (a)/(b) exception, remove circuits blocker and guard chunks. |
| Embedded networks | yes / yes | `fdtd/networks.py:257` prepares the coupling; `:846` scatters branch currents into E, `:870` solves the private network system, and `:988` advances network state. `:1029`/`:1078` run after ports (`fdtd/runtime/stepping.py:2580`, or `:2146` for sheet feeds/captured port phase). No Yee coefficient/code writes. Remove embedded-networks blocker under the same exception and preserve each existing network variant. |

No requested feature fails (a)/(b). Independent refusals remain: volume dispersive/ADE, gyromagnetic,
full/off-diagonal anisotropy, nonlinear/modulated/breakdown media, magnetic conductivity/permeability,
coordinate charts, Bloch/complex fields, Mur/mode-ABC/PMC boundaries, unqualified simple PML/graded
absorbers, Batch/distributed execution, non-float32/non-CUDA execution, gradients/adjoints, unsupported
source/monitor types and non-native backends. In particular, nonlinear/modulation/breakdown need W3's
live coefficient work; removing surface/port blockers does not relax these other constraints or existing
feature-specific overlap and material restrictions.

`core_path_requires_step_loop` detects scene ports/elements/wires/circuits/networks, prepared copper,
thin sheets, cut surfaces and SIBC, plus lowered modal sources/monitors. Immediately after resetting
the whole-step graph flags, `run_chunks` returns `None` for these pieces. The ordinary ordered step
loop therefore still calls the active fused Yee pair. W2 owns extension of `prepare_step`; no PCB-owned
module or kernel was changed. This admission does not claim whole-step graph safety.

`benchmark/core_path/scenes.py` adds `W1_SCENES` to `CORE_SCENES`:
`w1_copper_patch`, `w1_thin_sheet_cavity`, `w1_lumped_port`, `w1_coax_port`, `w1_thin_wire`,
`w1_cut_surface`, `w1_sheet_feed`, `w1_sibc`, `f3_coarse_11f`, `f4_dipole_32`, `w1_circuit`,
`w1_network`, `w1_terminal_port`, `w1_wave_port`. The last case exercises the lowered modal carrier,
not the outer calibration sweep. Existing CPU scene/geometry helpers and datagen builders are reused;
the CUDA-only fixtures are adapted at CPU construction. Datagen frequencies and port drives are retained.
The largest prepared primal grid is 91 × 37 × 40 (134,680 nodes); the next is 49³. Their persistent
field/coefficient/CPML/11-frequency DFT storage is conservatively below 0.5 GiB per scene, leaving room
within the 2 GiB target for the small feature state and per-step graphs. GPU peak memory remains unmeasured.

The G1 harness now also snapshots surface/copper/thin-sheet/wire state and public ports, losses,
circuits and network outputs. G4 retains its full prepared coefficient/source/observer-table comparison.
W1 G1 capture requires an active core runtime and an inactive whole-step graph. Three fresh-process pytest
equality cases cover planar copper, the conformal cavity and the fed copper sheet, asserting the same guard.
Subprocess PYTHONPATH is explicitly the worktree in both pytest and the gate launcher.

CPU checks: all changed Python files compile; all fourteen scenes construct and materialize their grids
on CPU; admission reports only non-CUDA execution. Every separate guard condition was exercised with
source tables present, returning `None` and resetting graph flags. Modal source/monitor guards are covered
by the lowered scene. CUDA initialization was patched to raise, and `torch.cuda.is_initialized()` stayed
false. Artifacts and exact pending G1 K=32/K=1, G4 and pytest commands are under
the W1 handoff (`GPU_TODO.md`, `cpu_check.py`, `cpu-scenes.json`).

### W2 closure design (implemented, GPU qualification pending, 2026-11-03)

`CorePathRuntime.prepare_step` now selects the ordered separate-physics closure when prepared copper,
thin sheets, wires, ports, elements or networks are present. Scenes without these pieces retain their
existing source/TFSF closure. Both feed spectral observers and the core step commit. Small point spectra retain the native point kernel; other spectra retain the grouped engine.
Only the opt-in core path changes; the default path is untouched.

**Step order.**

1. `_field_update_block(solver, 0.0)` preserves copper/sheet begin, fused H, magnetic surface corrections,
   wire sampling, fused E, wire deposition, copper/sheet E and copper spectra in their existing positions.
2. `inject("electric")`, then `inject("ordinary")` read the existing device tables. Both electric source
   families retain their n+1/2 clock (`injection_time`); ordinary sources retain their immediate PEC clamp.
3. With port runtimes, `prepare_port_drive_tables(solver, time_steps)` runs before capture, and
   `solver._port_observer_step.fill_(runtimes[0].sample_index)` and
   `solver._port_observer_graph_active = True` match `_make_port_step_runner` exactly. Each step calls
   `_port_step_block(solver, counter, post_source=tail)`, then `counter.add_(1)`. That block applies ports,
   advances fed sheets (which skip E in the field block), runs native-LU network feedback, calls the tail,
   then samples port V/I. `solve_steps` still calls `complete_port_observer_graph` once after its loop,
   including when the K-graph supplies all steps or stops early.
4. Without port work, call `tail()` directly. Standalone passive elements without observed ports still
   call `_port_step_block(solver, None, post_source=tail)` so their loads cannot be omitted.
5. `tail` applies electric dispersive corrections, clamps PEC unless TFSF, applies Mur and cut-surface
   constraints, then launches the core field DFT. W6 adds the ADE correction to the original W2 tail.
   Explicit prepare-step assertions require no Mur/mode-ABC boundaries, TFSF, magnetic surface sources
   or modulation; no live step time is required by the field block or ADE helpers.
6. Core spectral observers run after the port phase, with native point accumulation after the grouped
   engine gather when present; the core commit advances the source/DFT
   counter once. Shutoff, progress and plotting remain at chunk boundaries.

**Admission.** `core_path_step_loop_blockers` replaces W1's blanket `core_path_requires_step_loop` guard.
It keeps the fused Yee pair inside the per-step loop for circuits, any circuit-bound port, SIBC,
cut-surface constraints, and separate physics combined with TFSF, magnetic surface sources or modulation.
The cut-surface call stays in the tail for future admission after capture review. Existing volume-material,
boundary, gradient, source-table and observer guards remain. Wire monitors have host phase/sample updates
in `accumulate_wire_monitors`; scenes with those monitors also retain the per-step loop.

Lossless and lossy wire advances are admitted: their current, charge, EMF and loss-ADE state are device
tensors, and `_update_wire_state_lossy`, `_segmented_sum` and `advance_current` have no host sync or
data-dependent branch. Embedded networks use `apply_network_runtimes(native_lu=True)` inside the existing
port block, including their delay state, rather than the eager network runner.

| Gate scenes (W1 plus W2) | Whole-step admission | Reason |
| --- | --- | --- |
| copper_patch, copper_board_60k, w1_copper_patch, w1_thin_sheet_cavity, w1_sheet_feed, f3_coarse_11f | yes | Copper and sheet work is in the field/port blocks |
| w1_lumped_port, w1_coax_port, w1_terminal_port, w1_network | yes | Device port rows, V/I observers and native-LU feedback |
| w1_thin_wire, f4_dipole_32 | yes | Device wire state, no host-driven wire monitors |
| w1_wave_port | yes | Lowered modal source/monitor uses the existing core source closure |
| w1_circuit | per-step | Circuits remain unqualified for whole-step capture |
| w1_cut_surface | per-step | Cut-surface capture safety remains unqualified |
| w1_sibc | per-step | SIBC capture safety remains unqualified |

**Capture restoration.** `graph_state_tensors` and its checkpoint enumerators already cover fields, CPML,
copper state/right-side E/loss/side spectra, sheet extra/chord/history state and feed outputs, wire
current/charge/EMF/loss-ADE state, and network state/carried voltage/delay histories. W2 additionally saves:

- Copper `old_e`, fused loss partials and counter: begin-step traces and reduction scratch.
- Sheet H/E gather/update buffers and fused RHS/history/midpoint/reduction/chord-solve scratch; feed
  `column_term`: persistent buffers written by the field and fed-sheet phases.
- Port counter, drive row/buffer, electric/magnetic clocks, V/I weight rows, current buffer, spectral sums
  and terms (including drive spectra): capture must not add samples or advance a drive clock.
- Lumped/coaxial branch state, all `last_*` diagnostics, edge/correction/fast-state buffers, wire-port
  coordinate/substep buffers: preserve loads and diagnostics on both success and failure.
- Network solve, voltage/current, terminal gather/scatter, delayed-wave scratch and all power/energy
  buffers: preserve feedback, delay cursor/history and exported energy totals.

The existing core/DFT/observer-engine snapshot remains. Duplicate tensor objects are saved once. Host port
accumulator sample counts are restored after warmup/each capture; capture failure also restores the prior
port graph flag before falling back. Physical tensor addresses stay unchanged.

**DFT identity by code inspection.** `compile_spectral_tables` passes the same weighted cosine/sine tables
and real/imaginary component accumulators as `accumulate_dft_gpu`. `cuda/core/step.cu:dft_kernel` reads row
`step[0] * frequencies + f`, avoiding only the intermediate row copy. It uses the same float32
`accum += field * weight` expression as every native batched DFT variant in `cuda/kernels/spectral.cu`,
with one writer per field/frequency entry and the same compiler contraction settings. Its single core
commit replaces `_dft_step.add_(1)`; after capture succeeds `_dft_step` aliases the core counter, advancing
exactly once per physical step. There is no new trigonometric evaluation, normalization or accumulation
order. G1 now snapshots raw DFT accumulators as well as final phasors and the completed counter. Actual
GPU bit identity still requires G1.

**Small-point capture follow-up.** The single-observer builder deliberately retains native point kernels
for one point group with at most two spectral entries. Both ordinary and separate-physics K-closures now
admit that path without selecting the batch engine. `observer_step_weights` is shared with eager
`accumulate_observers`: it computes the identical host phase recurrence, window, source spectrum and
normalization/sample bookkeeping on staged entry copies. `NativePointObserverTables` stages active flags
and the exact E/H-staggered scalar weights, converted once to the accumulator dtype just like the native
scalar launch. The indexed entry point specializes the existing point kernel, preserving its field gather,
`accum += value * weight` expressions, launch geometry and compilation settings. An inactive row returns
before reading fields or touching accumulators, preserving skipped samples and signed zeros. The device
core counter selects each row; point accumulation remains after field DFT and port V/I sampling, before
commit, in both closures. There is no host step index or time inside capture.

Point real/imaginary accumulators join the warmup/capture snapshot. Immutable weight rows and staged
host metadata are never mutated by capture; completed metadata is published once after replay using the
actual completed step count, including early shutoff. Capture failure therefore leaves eager entry state
and point accumulators unchanged. No `runtime/stepping.py` hunk is needed by this follow-up.
GPU bit identity remains pending coordinator qualification.

**Gates and CPU checks.** `W2_SCENES` adds copper_patch and copper_board_60k from the existing overhead
builder with D3b's 2.4 GHz/default time-step settings, plus f3_coarse_11f, w1_coax_port, w1_lumped_port and
w1_thin_wire. All are constructed on CPU, included in the default list without duplicates, and explicitly
require active whole-step capture in G1. W1's admitted fixtures now expect whole-step capture as well;
the three remaining fixture exclusions expect per-step execution. CPU controls cover all sixteen unique
scene/grid constructions, the admission table, exclusions, prepare assertions, field/source/port/feed/network/
tail ordering, and nonzero-state warmup/capture restoration on success and forced failure. G1 at K=32/16/1
on W1/W2, G1 K=32 on the full default list, G4 on W2 and GPU core-path pytest remain pending. No CUDA
initialization or GPU execution was performed for this implementation.

### W3: live electric coefficients

Core ABI v7 adds dense per-component E coefficient reads for instantaneous nonlinear,
modulated and deterministic breakdown media. The launch owns the exact decay/curl tensors
selected by stepping, including Kerr's dynamic curl and the general nonlinear dynamic pair;
no admitted mutator changes H coefficients. The nonlinear coefficient launches remain before E,
and breakdown remains after the complete field/source/PEC tail and before spectral observers.
Breakdown state and stress contiguous-run history now update in place. Capture restores those
histories, live coefficients and stress accumulators along with the existing clock/field state.

Modulation reuses `advanceModulationTime3D` at the beginning of each step. Starting from
`modulation_step = -1`, the device increments to `s`, then stores FP64 endpoints
`double(s - 1) * dt` and `double(s) * dt`; no accumulated or FP32 clock is substituted.
The fused modulated variant retains FP64 phase reduction, the standard modulated curl,
the flat dense/slab CPML curl and the original modulation-factor combine. W2's separate-physics
modulation exclusion and assertion remain; these scenes still use the ordered per-step closure.
With W6 phase 1 merged, electric ADE composes with these media in the existing order: ADE
advance, then the nonlinear coefficient update, then fused E; ADE correction stays in the tail
before PEC/Mur/cut-surface. Breakdown with dispersive poles stays refused by material validation.

The live E kernels pin their rounding with explicit `__fmaf_rn`/`__fmul_rn` intrinsics:
psi = fma(b, psi, c*d), the existing curl assembly, and E = fma(E, D, C*curl), the order nvcc
13.0.88 emits for the existing dynamic-coefficient sm_86 kernels. Those existing kernels rely on
default FMA contraction; when W7 removes them, the core kernel stays the sole reference.

`W3_SCENES` adds Kerr (two strengths), TPA, modulation (two phases), an inline breakdown slab
with a stress observer, six modulation/nonlinearity fixtures and three Debye compositions (Debye+Kerr,
Debye+modulation, Debye+modulation+TPA) to `CORE_SCENES`. G1 also records live coefficients, modulation clock,
nonlinear peak and all seven breakdown histories, and requires triggering plus dissipation in
the breakdown fixture. Host contracts (74 checks) and an offline sm_86 ABI-v7 build pass without
CUDA initialization. GPU G1 K=32/16/1, the full core gate set at K=32, core-path pytest and the
seven BITID files remain required. PCB review: the only `stepping.py` hunk narrows the existing
breakdown graph-disable guard to runs without a core runtime; field-update ordering is unchanged.

### W6 phase 1 — existing ADE on the core path (2026-11-03)

The blanket dispersive/ADE refusal and W2's no-dispersion assertion are removed. The ordinary K-step
closure now advances magnetic ADE before fused H, corrects H after magnetic sources, advances electric
ADE before fused E, and corrects E in the post-source tail before PEC/Mur/cut-surface/DFT. The separate
physics closure uses the unchanged `_field_update_block` and `_port_step_block`, with that same ADE tail
before port V/I sampling. There are **no `stepping.py` hunks**; PCB review concerns the tail in
`runtime/core_path.py`. Dispersive capture failure restores state and raises rather than returning to
per-step execution.

All four ADE helpers use in-place native CUDA launches with prepared tensors and constant recurrence
scalars; there is no step-time scalar, host read, synchronization or device allocation. Material setup
installs pole-owned eps_inf/mu_inf before `build_update_coefficients`, which records uniformity after
conductivity, absorber and PEC composition. Compression already consumes those final coefficients;
no ADE arithmetic or storage changes are needed. Magnetic ADE shares the same helpers and is admitted
subject to the independent nonvacuum-permeability and magnetic-conductivity blockers.

`DISPERSIVE_SCENES` adds 13 existing electric benchmark scenes, a dispersive lumped-port fixture and a
vacuum-mu_inf magnetic Lorentz slab. G1 pins their ADE state, performs a physics-checkpoint disk round trip
and compares configuration fingerprints. See the [W6 census and phase-2 proposal](../assessments/core-path-dispersive-expectations.md).
Whole-step G1 at K=32/16/1, default CORE_SCENES at K=32, GPU core-path tests and the seven BITID files remain
coordinator gates. Phase 2 must be reviewed before implementation; phase 1 does not fuse ADE into Yee.
