# Late-time growth of the default CPML in real-field periodic or closed-transverse cells (2026-11-01)

Status: **closure defect remains open**. User decision 2026-11-02 ("CPML 1+2")
adds deterministic automatic graded absorption and a fail-closed main-loop growth
guard while passive CPML closure research continues. `absorber="auto"` is now the
default: real-field closed-transverse scenes select graded sigma only with
**>= 40 explicitly configured PML layers**. Thin layers raise instead of changing
geometry. Complex fixed-k retains shifted CPML and gains the growth guard.
GPU qualification of this change remains pending. See
[the current boundary contract](../reference/fdtd-cpml.md).
**F5 production data still waits for the passive-closure repair.**

## What happens

Take a real-field cell whose transverse faces are periodic, or closed (PEC/PMC), and whose normal
faces use the explicit CPML (`absorber="cpml"`, alpha = 0, kappa = 1). The FDTD update has an unstable
grazing TM eigenmode. Any transverse perturbation seeds it, for example a partial-cell material
variation, a time modulation, or noise. It then grows exponentially. Short runs are unaffected;
long-horizon runs are not.

Measured in the benchmark `modulated_slab` cell: period 1.28 m, h = 0.025 m, 12 CPML layers,
dt = 16.67 ps.

- The native CUDA update matrix on the first transverse Floquet order (TM) has
  |λ| = 1.000135260 at 0.231794583 GHz. That is an amplitude growth of 8.115e6 s⁻¹, an e-fold of
  about 123 ns, with onset visible at about 2 µs in the benchmark scenes.
- A static slab with no modulation, seeded with 1e-5 noise, grows at 8.110e6 s⁻¹. Every modulation
  depth from 0.2 to 0.01 gives the same rate: the modulation only seeds the mode.
- The energy accounting places all of the gain in the CPML ψ pair. The magnetic ψ work supplies
  +7.089e-4 W per step and the electric ψ removes 4.384e-4 W, a net +2.705e-4 W per step, equal to
  |λ|² − 1. The balance closes to 3e-10.
- Closed boxes without PML conserve the discrete energy to FP32 round-off.
- Periodicity is not the trigger: a transverse PEC box also grows (9.5e6 s⁻¹), and so does a PMC
  box (1.0e7 s⁻¹).
- Halving dt does not cure it (8.09, 8.75 and 9.06e6 s⁻¹ at dt, dt/2 and dt/4).
- `absorber="stablepml"` grows faster (|λ| = 1.000170). A simple matched frequency shift
  (2πc/L, floor 1) still grows (|λ| = 1.000025).

Kernels: `fdtd/cuda/kernels/magnetic.cu::update_magnetic_cpml_kernel` and
`fdtd/cuda/kernels/electric.cu::update_electric_cpml_kernel`. Profile:
`fdtd/boundary/cpml.py::_cpml_profile_1d`.

Scope: **real-field periodic or closed-transverse cells only.** The complex fixed-k Bloch path had the
same mechanism with the legacy alpha = 0 (ρ − 1 up to 1.3e-6 on the open cell at 30°), but its
default is already CFS-CPML (master 337cee2f). That shift does not certify all guided
content: the eps4 fixed-k TM slab tail grows at +2.5e-6/step log energy and is
covered by the new guard.

## Interim default and explicit workaround (2026-11-02)

For long-horizon real-field closed-transverse runs, the default `absorber="auto"`
selects graded conductivity. **It does not apply to complex fixed-k (oblique Bloch) runs:**
there it gives r/t errors of 0.2–0.5 (TM εr 4 slab, k∥ 0.55 k0, 40 layers, 2026-11-02), so
`auto` keeps CPML for complex fields. Configure >= 40 layers in `BoundarySpec`; the same
absorber may be selected explicitly:

```python
sim = mw.Simulation.fdtd(scene, frequencies=..., absorber="absorber")
```

- In the native TM matrix it has only damped oscillatory modes. Both modulated benchmark scenes
  stay flat over 7.4 µs: fitted amplitude rate 313 s⁻¹ and 55 s⁻¹, i.e. bounded at the estimator's
  resolution.
- **Auto requires >= 40 layers.** Explicit thinner absorbers retain their former admission. The graded absorber needs more layers than the CPML for the same
  reflection. The CPML owner measured its real-field reflection with the study in
  `<evidence-root>/cfs-cpml-c2-2026-11-01/rcpml/`:
  - 40 layers: 4e-4 to 9e-4 below 1 GHz, 2e-5 at 3 GHz;
  - 24 layers: 1e-5 to 3e-3 across the band.
  Choose the layer count from the lowest frequency you need, and check the phase against the
  validation you rely on: its phase accuracy relative to CPML is not qualified for these workflows.
- **Limits that already apply to the graded absorbers:**
  - they are not accepted by the distributed adjoint;
  - an anisotropic structure overlapping them is rejected.
- **There is no runtime fallback.** Topology resolves `"auto"` before coefficient construction.
  Explicit `"cpml"` and `"stablepml"` use their existing profiles with a main-loop passivity
  guard. Short records may finish before its first 16-window fit; that is not a passivity certificate.
- The graded absorber is not a suitable complex fixed-k workaround: its r/t errors are 0.2–0.5.
  Those runs retain CPML and use CW per frequency or short pulsed records if the guard trips.

## Evidence

Archived at `<evidence-root>/instamax-line-modulated-growth-20261101/`, a sha256-verified
copy listed in its `SHA256SUMS.source`:
- `REPORT.md`: the full diagnosis. The native amplification matrices, edge work accounting,
  closed-box energy balance, the depth/dt/h and boundary discriminators, and the gates for a repair
  are in "Native amplification matrices", "Location and mechanism" and "Fix proposal and gates".
- `native_matrix.py` and `eigen.py`: the native spectral probe, which the option-B repair reuses as
  its gate.
- `growth.py`: the long-horizon growth check.
- The graded-absorber reflection study:
  `<evidence-root>/cfs-cpml-c2-2026-11-01/rcpml/`.

A separate defect found in the same diagnosis: the FP32 modulation clock drift (+62 rad at 0.35 GHz
over 7.4 µs). It is fixed on master (b156c42b, native ABI v7) and is unrelated to the CPML growth.

## Update 2026-11-04: modulated_slab no longer shows growth; the static CPML mode stands

- The CW-driven benchmark `modulated_slab` with explicit CPML shows no growth on the current base. Its
  guard windows stay at 6.927e-9 ±1% from 1.03 to 7.38 µs (443,951 steps), with no onset at 2.3 µs or
  later. Record: `<evidence-root>/cpml-guard-gates-20261104/cw_modulated_slab_passivity.json`.
- The growing grazing-TM mode is a property of the static z-CPML/grid operator, and is not superseded.
  - The eigenpair above was computed from the native update without modulation (|λ| = 1.000135260 at
    0.2318 GHz; the static and the forced-zero-modulation TM matrices are entrywise identical).
  - A seeded static run grows at the same rate.
  - On the current base the seeded, non-modulated εr3 cell still trips the passivity guard
    (`tests/boundaries/cpml/test_passivity_gpu.py::test_real_periodic_eps3_slab_tm_explicit_cpml_trips`).
- What changed is excitation: that scene no longer drives the mode measurably. The likely reason is
  that after the partial-cell modulation fix (f928126d) the cell stays exactly transversely uniform, so
  the transverse Floquet q=1 mode is never seeded. Only the clock-drift part of the 11-01 report (+62 rad
  over 7.4 µs) is historical, fixed by b156c42b.
- In the guard's GPU tests, the CW `modulated_slab` is now a no-trip control. The CW must-trip case is a
  CW-driven, non-modulated εr3 cell with an off-axis dipole, which seeds q=1.


## Merge completion audit (2026-11-04)

Rebased the 609accd4 branch onto master 2e8308d2. Preserve master's resume
schema **6** and its W3/dispersive fixtures. The six pinned closed-transverse
cases (`kerr_slab`, `kerr_slab_strong`, `tpa_slab`, `modulated_slab`,
`modulated_slab_phase`, `graphene_sheet`) retain their coefficient-storage
checks and explicit CPU guard-refusal coverage, and are excluded from the
admitted core GPU gate list. All other newer core gates remain. The removed
`planewave_periodic_vacuum` gate also keeps its actual benchmark builder and
CPML refusal, rather than falling through to a generic open-box fixture. The branch declaration is
before the first existing line-anchored merge-declaration header and explicitly
lists the branch test files and all 150 supplied mover nodes.

The 150 automatic-absorber refusals span 33 files: **138 CPML physics nodes**
retain their original grid/termination with explicit `absorber="cpml"`, and
**12 default-behavior nodes** retain `auto` with 40 layers. The two mixed files
are decided per test below. Counts are parameterized failing nodes in the
supplied broad-tier inventory, not the number of edits or the CPU pass count.
No physics assertion, tolerance, or acceptance budget is relaxed.

| File | Count | Choice | Reason |
| --- | ---: | --- | --- |
| `tests/materials/thin_sheet/test_thin_sheet_scattering.py` | 24 | CPML: 24 | Retain the calibrated complex scattering, reflection floor, and convergence budgets in the shared transmission harness. |
| `tests/sources/tfsf/test_complex_cw_amplitude_contract.py` | 16 | CPML: 16 | Match the periodic real-field control and complex-field CPML operator, including padded-seed phase invariance. |
| `tests/sources/incident/test_soft_sheet_field_sign.py` | 16 | CPML: 16 | Preserve the absolute analytic E/H amplitude and sign budget of the CPML launch experiment. |
| `tests/validation/physics/test_planar_pec_sheet.py` | 12 | CPML: 12 | Keep the terminated plane-reflection fixture and its exact aligned/conformal comparisons. |
| `tests/sources/definitions/test_custom_field_stencil.py` | 10 | CPML: 10 | Keep the original low-backwave termination for the absolute discrete-wave source oracle. |
| `tests/sources/point/test_periodic_point_dipole_wrap.py` | 7 | CPML: 2; auto/40: 5 | Pin the two dipole/current-sheet radiation comparisons; keep four unit-moment compilation cases and the shard-interface rebuild on the default with 40 layers. |
| `tests/materials/combinations/test_bloch_dispersive_forward.py` | 7 | CPML: 7 | The real quadratures must use the same CPML operator as their unit-phase Bloch reference. |
| `tests/materials/surface_impedance/test_copper_foil.py` | 6 | CPML: 6 | Retain the finite-layer scattering and small-absorption reference in the shared sheet measurement. |
| `tests/validation/physics/test_pec_curved_coax_tem.py` | 5 | CPML: 5 | Keep the terminated TEM propagation fixture used by the independent phase-velocity oracle. |
| `tests/validation/physics/test_lossy_metal_roughness_fdtd.py` | 4 | CPML: 4 | Keep the standing-wave reflection fixture used to quantify additional conductor absorption. |
| `tests/materials/thin_sheet/test_thin_sheet_faceted.py` | 4 | CPML: 4 | Retain the faceted-cylinder Bessel-series and refinement budgets in the shared harness. |
| `tests/materials/surface_impedance/test_copper_fused_kernel.py` | 4 | CPML: 4 | Already pinned at 609accd4; preserve the original recurrence and trajectory oracle, not a new absorber operator. |
| `tests/validation/physics/test_surface_impedance_broadband.py` | 3 | CPML: 3 | Retain the reflection, convergence, and graph-capture fixture qualified against the surface budget. |
| `tests/validation/physics/test_lossy_metal_sibc.py` | 3 | CPML: 3 | Keep the original normal-incidence Leontovich reflection experiment. |
| `tests/materials/thin_sheet/test_thin_sheet_cuda.py` | 3 | CPML: 3 | Use the same calibrated oblique-scattering harness as the CPU physics gates. |
| `tests/materials/surface_impedance/test_copper_foil_angles.py` | 3 | CPML: 3 | Keep matching real and Bloch terminations for the orientation/power oracle. |
| `tests/validation/physics/test_sibc_staircase.py` | 2 | CPML: 2 | Pin the voxelized flat-slab Leontovich reference; unrelated open-box cylinder tests retain auto. |
| `tests/materials/surface_impedance/test_copper_foil_edge.py` | 2 | CPML: 2 | Keep the terminated line geometry and independent attenuation reference. |
| `tests/materials/sheet/test_fdtd_graphene.py` | 2 | CPML: 2 | Keep both sheet-scattering experiments on the common Fresnel/Kubo reference termination. |
| `tests/materials/modulation/test_partial_cell_modulation.py` | 2 | CPML: 1; auto/40: 1 | Pin the weak-slab Born amplitude/phase experiment; the actual modulated-mass time-step admission test uses the default with 40 layers. |
| `tests/materials/combinations/test_rotated_birefringent_dispersive_slab.py` | 2 | CPML: 2 | Retain the crystal-index propagation oracle and original termination; kernel-only tests are unchanged. |
| `tests/fdtd/cuda/test_source_patch_determinism.py` | 2 | auto/40: 2 | Periodic seam and core-path repeatability validate default execution; increase only the seam fixture layers. |
| `tests/validation/physics/test_sibc_orientation.py` | 1 | CPML: 1 | Pin the double-sided passive-plate physics comparison; other orientation/open-box tests retain auto. |
| `tests/validation/physics/test_pec_conformal.py` | 1 | CPML: 1 | Pin the aligned-slab bitwise conformal/staircase comparison; closed-cavity tests retain auto. |
| `tests/validation/physics/test_dispersive_metasurface_bloch.py` | 1 | CPML: 1 | The periodic quadrature reference must match the patterned Bloch CPML operator. |
| `tests/materials/surface_impedance/test_surface_impedance_runtime.py` | 1 | auto/40: 1 | Exact-fill surface write coverage exercises default preparation and one update, independent of CPML reflection. |
| `tests/materials/surface_impedance/test_complex_cw_copper_normalization.py` | 1 | CPML: 1 | Keep identical real/complex film operators for absolute phasor and work comparisons. |
| `tests/materials/sheet/test_lossy_metal.py` | 1 | auto/40: 1 | Public default preparation must reach surface-impedance configuration; compile-only scenes keep their grids. |
| `tests/materials/sheet/test_fdtd_medium2d.py` | 1 | CPML: 1 | Retain the analytic Fresnel sheet experiment through the shared measurement helper. |
| `tests/materials/pmc/test_pmc_medium.py` | 1 | CPML: 1 | Keep matching terminated PEC/PMC dual plane-wave experiments. |
| `tests/materials/combinations/test_lossy_anisotropic_slab.py` | 1 | CPML: 1 | Preserve the analytic conductive-crystal attenuation experiment; uniform/kernel cases retain auto. |
| `tests/fdtd/test_cuda_graph_capture_coverage.py` | 1 | auto/40: 1 | Graph coverage tests default absorber admission, so enlarge the closed-transverse surface fixture. |
| `tests/fdtd/test_cpu_reference_backend.py` | 1 | auto/40: 1 | Keep auto and 40 layers; an open z face resolves CPML so the fixture reaches the intended non-periodic refusal, preserving its assertion. |

The refusal now identifies effective closed/periodic axes (after symmetry),
every PML face below 40 layers and its count, and both remedies. For example:

```text
absorber='auto' selects graded sigma for real-field closed-transverse PML topology: closed/periodic transverse axes: x(low=periodic, high=periodic), y(low=pec, high=pec); PML faces with fewer than 40 layers: z-low=39, z-high=39. Requires >= 40 layers on every PML face. Either pass absorber="cpml" (CPML under the fail-closed passivity guard) or use >= 40 PML layers for the graded absorber; geometry is never changed.
```

### Signature singles

| Signature / nodes | Root cause and disposition |
| --- | --- |
| `FakeFDTD._compiled_sources` (two public API cases and smoke) | The production time planner reads compiled sources directly. The old test stand-ins omitted the field. Add `_compiled_sources = ()` to each stand-in; no library stub accommodation. This contract also exists on master, so it is not introduced by the absorber guard. |
| `source_reference_factor` (normalization stand-in) | The observer-origin/source-reference change added spectral entry metadata. Master 2e8308d2 already constructs it through `source_reference_metadata` in this test. Keep that fix; CPU passes. |
| `kerr_slab_strong` catalog identity | Branch-induced: central `dataclasses.replace` CPML pinning cloned canonical scenario objects. Pin all 13 scenarios at their definitions and pass the absorber through `_make`; retain the exact identity assertion and add a complete pin-set check. |
| Two parallel-public-API call-list assertions | Branch-induced: the default forwarded configuration is now `AUTO`, resolved in the runtime, rather than `CPML`. Assert `AUTO`. These mocked routing tests now construct a CPU scene, making their unchanged routing assertions CPU-runnable. |
| Surface field-time sample-list assertion | The observer-origin change advances physical labels from `n*dt` to `(n+1)*dt`; unrelated to this branch. Master already corrects this exact expectation. Keep it unchanged. |
| Batched multi-frequency DFT scalar assertion | The same observer-origin change labels Ex samples at `dt` and `2*dt`, not `0` and `dt`. Correct the analytic two-sample complex sum without changing tolerances. Run the same test on the explicit CPU reference backend as well as native CUDA. |
| Two-photon transmission allclose | The flux-time helper asserted old `n*dt` labels. Master already expects `(n+1)*dt`; the energy/absorption physics assertions are unchanged. CUDA physics is not rerun in this CPU-only task. |
| Metal half-space allclose | An unrelated master fix extends the half-space through transverse PML; otherwise a vacuum shell contributes spurious edge power. Keep master's geometry and physics assertions. The failing GPU solve cannot be reproduced under CPU-only scope. |
| Fixed-k must-trip guard single | Already addressed by d2bf7ea9 before this task: the point-dipole fixture's measured slope 2.09e-7/step is below the documented 4e-7 detection resolution. Retain its pinned sub-resolution band and no-trip check; no new physics change. CUDA requalification remains pending. |
| Guard manifest mismatch | Exact raising-site identities must be reconciled after source freeze; see census review below. |
| Three `SAFETY_GUARDS` import errors | External `/tmp/pcb-consumer` sys.path contamination imported another tree's collector. No workaround or library change. Clean-process local census gates exercise this worktree's collector. |

The first full CPU mover run exposed one additional admission-order issue in
`test_cpu_reference_rejects_unqualified_profiles`: raising the closed-transverse
fixture to 40 layers resolves graded sigma, which the CPU profile refuses before
periodicity. Keep auto and 40 layers, but open its z face so automatic CPML is
admitted and the unchanged non-periodic-boundary assertion isolates its intended
capability. The existing closed-transverse graded-profile refusal remains covered
by `test_cpu_reference_auto_resolves_before_capability_check`.

### Census review

Register all five branch safety refusals: distributed automatic graded selection,
thin automatic graded layers, sustained CPML energy growth, resume policy/history
mismatch, and the core-path CPML blocker. Retain the modified CPU-reference
absorber-admission review, with automatic resolution before profile validation.
The final inventory has **558 active reviewed sites**, including these six
branch new/changed sites, and **48 reviewed retirements**. Historical branch
message reviews and master's unrelated entries are preserved. The newly retired
generic auto-refusal identity is replaced by the axis/face diagnostic; the
40-layer predicate and both remedies remain enforced. Negative references point
to matching assertions, including the effective-symmetry-face diagnostic test.
No historical baseline or contract exclusion was reclassified.

### CPU validation

Validation commands use the maxwell environment with `CUDA_VISIBLE_DEVICES=""`.
The final receipts cover **803 passed and 263 skipped unique nodes**. CUDA
execution was disabled throughout; skipped native/GPU physics remains unqualified
by this task. No `.cg-*` inputs or temporary runner files are staged.

| Scope | Final passed | Skipped | Return-code evidence |
| --- | ---: | ---: | --- |
| `tests/api/public/test_guard_census.py`, `tests/validation/benchmark/test_integration_batch.py`, `tests/boundaries/cpml/` | 174 | 31 | Combined command rc=0; census rechecked after its final reference update: 7 passed, rc=0. |
| Existing branch boundary/core coverage, including W2/W3 admission and live coefficients | 115 | 40 | Final six-file command rc=0. |
| All 33 mover files, public API/smoke, normalization, DFT, parallel routing, catalog, nonlinear/scattering/monitor CPU parts, thin-sheet admission, observer-origin coverage | 514 | 192 | The first 44-file command reported 513 passed, 192 skipped, one admission-fixture failure (rc=1). After the isolated fixture correction, the same selection with `--lf` passed its remaining failure (1 passed, rc=0), and the complete affected CPU-reference file passed all 28 tests (rc=0). Unchanged passing cases retain their full-run receipt. |

The real-periodic seam fixtures use 40 layers; the unaffected complex Bloch seam
keeps its original three layers. Its final CPU collection/recheck returned rc=0
with seven CUDA skips. Rechecks are not counted twice in the unique-node totals.
Physics tolerances and acceptance budgets remain unchanged. The separately
identified analytic DFT expectation follows master's physical time labels;
admission, diagnostic and routing assertions retain the contracts described above.

Final structural checks also confirm master 2e8308d2 is an ancestor, resume
schema 6 is retained, all 48 changed branch test files and all 150 original mover
nodes appear in the first merge declaration, every other integration-manifest
setting/declaration is identical to master, master's guard reviews are preserved,
`AGENTS.md` and `CLAUDE.md` match, and `git diff --check` passes.

## D2: complex fixed-k warns once (user decision 2026-11-06)

The main-loop guard remains fail-closed for real-field closed-transverse cells:
`FDTDPassivityError` and every real-field must-trip gate are preserved. For
complex nonzero fixed-k Bloch cells, the same growth criterion now emits one
`RuntimeWarning` per run and execution continues. The warning reports measured
log-energy slope, stderr and the fitted start/end steps. Thresholds, fit windows,
sample interval, source gating and the energy functional are unchanged.
Nonfinite history follows the same topology split, with infinite measurements.

Measured fixed-k evidence supplied with D2:

| Cell | Log-energy growth per step |
| --- | ---: |
| Point-dipole eps4 slab | 2.1e-7 |
| eps4 TM slab CW, k_parallel 0.25 k0, 0.9 f0, resolution 160 | 8.2e-7 |
| eps4 TM CW parity at 1.1 f0, continuity at 20 degrees | 3.5–5.0e-7 |

These measurements explain warn-only policy; they do not establish passivity.
The dipole fixture stays a below-resolution control. A new CUDA-marked eps4 TM
CW gate uses the measured 0.25 k0 / 0.9 f0 / resolution-160 scene and requires
completion with one warning and matching solver/Result metadata. GPU execution
is reserved for the reviewer; all commands for this change disable CUDA.

Existing metadata keys are retained. `passivity_guard` becomes
`warned: complex fixed-k growth` after the first warning. The adjacent
`passivity_guard_warning` field is `None` or a record with `slope`, `stderr`,
`window` (the first and last fitted window-end steps) and `step` (the first
warning's completed step). Both appear in Result metadata and solver statistics;
solver attributes are `passivity_guard_status` and `passivity_guard_warning`.

The passivity checkpoint dictionary adds `warning=None` or that record.
`capture_passivity` copies host data as well as tensors; preflight restores
state and solver metadata without re-warning. Schema **6 is retained**. The
existing exact-key check requires the new field, so older guarded payloads
without it are rejected; no compatibility path is introduced.

The requested full test grep was performed. Four existing synthetic nodes move
their complex-field expectation from raise to warning plus measurements:
`test_exponential_growth_trips` (complex coverage now in the new warn-once
test), `test_resume_preserves_partial_window_and_trip_step`,
`test_nonfinite_energy_fails_closed`, and
`test_upper_three_stderr_and_rising_endpoint_required`. Their detailed old/new
expectations are listed in the guard merge declaration as
**T2 movers under user decision D2 (complex fixed-k warns)**. No existing GPU
raise assertion was a complex fixed-k mover. Added CPU coverage checks one
warning, measurement contents, warned-state disk round-trip, resumed suppression,
and full main-loop Result/statistics metadata across disk resume. The real-field
growing-history and main-loop/disk-resume raise tests remain active.

The guard census was regenerated to `/tmp/d2-guard-proposal.json`. Only current
line numbers and the two passivity review descriptions were reconciled into the
manifest; all top-level keys and historical entries are retained. The inventory
has **559 active reviewed sites** and **48 reviewed retirements**.

D2 CPU gate receipts use `python`,
`CUDA_VISIBLE_DEVICES=""`, `-v` and `--junitxml=/tmp/d2-<gate>.xml`:

| Scope | Passed | Skipped | rc | JUnit |
| --- | ---: | ---: | ---: | --- |
| `tests/boundaries/cpml` | 88 | 32 | 0 | `/tmp/d2-cpml.xml` |
| `tests/api/public/test_guard_census.py` | 7 | 0 | 0 | `/tmp/d2-census.xml` |
| `tests/validation/benchmark/test_integration_batch.py` | 82 | 0 | 0 | `/tmp/d2-integration.xml` |
| `tests/fdtd/core_path/test_w3_live_coefficients.py` | 19 | 4 | 0 | `/tmp/d2-w3.xml` |
| Total unique nodes | **196** | **36** | **0** | |

The targeted detector file also passed all 63 nodes; those are included in the
CPML total, not counted twice. CUDA physics, including the new scene, remains
pending reviewer execution. The two instruction files still match, and
`git diff --check` passes. Briefs and `.cg-*`/`.cp-*` inputs are excluded.
