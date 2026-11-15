# Exact capability-guard review ledger

The user-approved observer origin option A adds
`fdtd/resume.py::_preflight_observer_origin`, shared by single-domain and
 distributed preflight before any state restoration. Format 6 requires
`observer_spectral_origin="physical_yee_v1"`,
`source_reference_label="provider_reference_v1"`, and the selected compiled
provider/waveform/beta and incident descriptor. Missing or mismatched metadata
is refused; format 5 is rejected on load. Reference stamps are derived metadata
and are excluded from the unchanged physical configuration fingerprint.
Negative coverage:
`tests/fdtd/test_observer_origin_reference.py::test_old_observer_checkpoint_is_refused_before_restoring`
and `test_format_five_is_refused_and_new_spectral_checkpoint_round_trips`.

The approved pulsed TFSF reference fix adds the shared
`fdtd/resume.py::_preflight_pulsed_tfsf_reference` boundary. Auxiliary-line pulses
must record `pulsed_tfsf_reference="physical_entry_plane"`; absent or different
metadata means the old padding-dependent reference and is refused before both
single-domain and distributed fingerprint comparisons. The configuration
fingerprint algorithm is unchanged. CW and analytic providers carry no new key.
The CPU negative assertion is
`tests/sources/tfsf/test_pulsed_incident_reference.py::test_old_pulsed_tfsf_checkpoint_is_refused_before_fingerprint_comparison`
(including distributed metadata). The native round-trip gate is
`tests/sources/tfsf/test_pulsed_incident_reference.py::test_native_pulsed_reference_pml_causality_and_resume`.
Restart an old affected run rather than mixing incident conventions.

CPML 1+2 (2026-11-02; resolution wording updated 2026-11-03) adds five inventoried safety refusals: distributed
automatic graded absorption, automatic graded PML faces thinner than 40 layers,
`FDTDPassivityError` on sustained energy-envelope growth, missing or changed
guard history at resume, and the core-path closed-transverse CPML blocker.
The automatic refusal records effective transverse axes and every deficient PML
face count, including symmetry overrides, while retaining both remedies.
Their matching CPU negative assertions are in
`tests/boundaries/cpml/test_auto_passivity.py`. The core blocker also has a GPU
`test_blockers[closed_transverse]` case. The existing CPU-reference absorber
boundary is reviewed with automatic resolution before admission; its prior
predicate is retired with positive default-resolution and resume evidence.

The collector now inventories the explicitly scoped ValueError and
FDTDPassivityError sites in `scripts/dev/fdtd/guard_census.py::SAFETY_GUARDS`,
as well as every NotImplementedError. Ordinary input-validation ValueErrors
remain outside this inventory. The immutable historical baseline and contract
exclusions are unchanged. After rebasing onto master 2e8308d2 and reviewing the detailed automatic-refusal
diagnostic there are 558 active reviewed entries and 48 reviewed retirements; this count does not assert physical qualification. The passivity error message now states that growth slower than 4e-7/step
(about 1.5x energy over 1e6 steps) is not detected; its prior message
identity is retired without removing or changing the guard. The CW-driven
`modulated_slab` is a no-trip control; the off-axis CW-driven static slab is the
must-trip case with onset and trip diagnostics. CUDA
growth, passive-control, graph and disk-resume gates remain pending.


The approved additive-J resume rule adds one shared boundary in
`fdtd/resume.py::_preflight_additive_j_timing`, called before both single-domain
and distributed signature comparisons. A checkpoint with additive electric
currents must record `additive_j_injection="n+1/2"`; absent metadata means exactly
the old `"n"` convention. Old checkpoints raise:

> checkpoint predates the additive-J n+1/2 source timing; re-run from the start

The timing label enters fingerprints only for PointDipole, UniformCurrentSource,
and CustomCurrentSource with electric components. Magnetic-only custom currents
and other non-J configurations retain their fingerprints. CPU prepared-scene
single-domain and distributed metadata signatures match `f52aa33f` bitwise.
The matching negative assertions are
`tests/fdtd/test_resume_additive_j_timing.py::test_old_additive_j_checkpoint_is_refused_before_fingerprint_comparison`
(CPU, missing/old metadata, single-domain/distributed preflight),
`tests/fdtd/test_resume_additive_j_timing.py::test_native_resume_refuses_old_additive_j_checkpoint`
(CUDA, all three additive electric-current families), and
`tests/fdtd/multi_gpu/test_radiation_observation.py::test_distributed_resume_refuses_old_additive_j_timing`
(two-GPU P2P, missing/old metadata). This boundary must remain; old additive-J
runs need to restart.

P-9(c) performance scope: planar real-field CopperFoil entries without material-trace
ADE select the fused FP64 CUDA electric step plus begin/magnetic/spectral
companion phases. State/gain arithmetic remains FP64; final H additions retain
field dtype and the reference indexed-add semantics. E1 uses its original
elementwise state/trace gate; E1b uses normwise relative <= `1e-13` per spectral
tensor. Paired/phase and material-trace ADE
entries retain torch; CPU and adjoint/replay retain torch. This adds no capability
raise or retirement. `tests/materials/surface_impedance/test_copper_fused_kernel.py`
owns the tolerance and fused graph/eager gates; historical pins retain torch.

The authoritative inventory is [fdtd-capability-guards.json](fdtd-capability-guards.json), checked by `tests/api/public/test_guard_census.py`. It records unsupported capabilities; an audit pass does not mean those capabilities are implemented or physically qualified.

`compiler/materials.py::_PCBConductivityReplay.__init__` rejects `kottke_smooth`,
whose fractional local-moment edge quadrature is outside scalar conductivity replay.
The matching negative assertion is
`tests/materials/compiler/test_kottke_smooth.py::test_trainable_geometry_and_material_are_accepted_and_replay_is_rejected`.
Admission requires bitwise replay parity with complete compilation for this mode.

The existing `kottke_edge` mode review also records its geometry `ValueError`
(outside the `NotImplementedError` identity census). The default
`Scene()` / `SubpixelSpec()` now selects four samples per axis and this mode.
A trainable structure or material-region geometry tensor, including one produced
by a `SceneModule` geometry parameter, fails during material preparation with:

> kottke_edge (the default subpixel averaging) is not differentiable in geometry; set SubpixelSpec(averaging='kottke_smooth') or 'polarized' for shape gradients.

A trainable tensor in any `MaterialRegion` field (including design density and
nested tensor values) also fails during material preparation with:

> kottke_edge (the default subpixel averaging) does not support gradients through MaterialRegion design densities; set SubpixelSpec(samples=1, averaging='arithmetic') for density/topology optimization.

`Structure` material-value gradients remain allowed. `kottke_smooth` is not
qualified for density gradients. The mode never changes automatically.
`test_trainable_density_requires_explicit_averaging` in the module below checks
the default rejection and explicit polarized admission.
The default/explicit triggers and explicit differentiable alternatives are covered
by `tests/materials/compiler/test_subpixel_default_guard.py`.

The immutable starting source is commit `10d351310b36e9681867347e772a21977e234bdf`: 329 `NotImplementedError` sites, including 300 capability sites and the unchanged 29 contract exclusions. Its committed scalar ceiling of 206 was already stale by 94 capability sites before this development. The original snapshot and a predicate/scope snapshot reconstructed from that commit are retained under `docs/assessments/pcb-development-2026-10-22/guard-census/`; their hashes are pinned in the audit tool. The [historical census](../archive/fdtd-capability-guard-census-through-2026-09-11.md) remains available and is not a current ceiling.

## Reviewed sites, not a scalar allowance

Each active entry stores its path, qualified scope, exception expression, enclosing branches/loop capability sets, exact literal message and stable identity. Line-number-only changes do not change identity. A changed message, scope, exception or captured predicate requires review. Arbitrary data-flow equivalence is not proven by an AST inventory; ordinary code review still applies.

Every new or modified boundary records the unsupported physical/runtime operation, rationale, responsible phase, negative-test reference and exit acceptance. `negative_control_status` distinguishes a matching negative assertion from `coverage_pending`, where only a module reference exists and individual trigger coverage is not claimed. A referenced test module is never treated as evidence that every guarded branch ran. Missing branch-level coverage remains visible audit debt. No capability is reclassified as a permanent contract to improve the count.

A retired entry retains the old identity, explicit implementation/narrowing reason, positive-test evidence and replacement boundaries. A narrower supported subset does not clear the entire formerly rejected capability. Unrelated additions and removals cannot cancel each other because the test compares exact identities, not just totals. Every original baseline identity must remain active or have a retirement record. Unreachable/private experimental guards are still included by the AST census; recording them does not enable their public feature.

The `CONTRACT_GUARDS` exclusions are fixed by an exact tuple hash; the only change since the 29-entry baseline is the authorized 29→28 reclassification recorded below. The tuple hash and the immutable baseline hashes are checked. The existing contract-accuracy and public-forward wording tests remain active.

## Review workflow

After relevant numerical source freezes, generate a separate proposal:

```bash
python scripts/dev/fdtd/guard_census.py --proposal /tmp/guard-review.json
```

The command never approves or overwrites the reviewed manifest. It preserves unchanged reviews, marks new/changed boundaries pending, and retains retirement candidates. Review each difference, document the actual unsupported boundary and evidence without upgrading its qualification, and apply the reviewed JSON as a deliberate code-review change. Set the snapshot status to `reviewed` only after the final source comparison. Then run:

```bash
python -m pytest tests/api/public/test_guard_census.py
```

The gate rejects pending snapshots, unknown/changed/missing identities, unreviewed entries, falsified inherited identity, missing rationale/exit/test references and incomplete retirement evidence. The collector is independently cross-checked against the original complete AST exception inventory, with controls for line movement, changed predicates/scopes/messages and dynamic loop capability sets.

The current reconciliation and per-boundary review are recorded in [the assessment](../assessments/fdtd-capability-guard-reconciliation-2026-10-22.md). It remains separate from physical solver qualification and performance acceptance.


## Fixed-chart electrical material review, 2026-10-28

The reviewed main snapshot contains **424** capability identities: **397** entries
are unchanged, the blanket mapped-adjoint rejection is retained as a reviewed
retirement, and **27** narrower admission boundaries replace it. The immutable
300-capability baseline and 29 contract exclusions have not changed. The three
core namespace guards present in private source snapshots are separate core
contracts and are not imported into this Maxwell inventory.

The new entries are reviewed boundaries, not 27 implemented features. Passive
exception tracing of 363 CPU contracts observed the exact raising site for 25;
two defensive ancestry/scalar-input invariants remain `coverage_pending` because
normal scalar constructors did not provide genuine triggers. References to
surrounding test modules do not close those two gaps. A site witness also does
not prove every alternative of a compound predicate: the manifest retains such
limits for parallel/initial-condition, UV, constitutive and other branches.

The retired blanket guard records the actual public 24-case GPU qualification on
frozen `e160f271` / core `f5c7bdae`, without a private admission bypass. The main
source additionally includes the independently validated Result alias fix;
main CPU integration has its own source identity. Neither result is relabelled
as a final integrated GPU/full-suite run. See the [admission evidence](../assessments/fdtd-mapped-material-admission-2026-10-28/README.md)
for the retained initial 3-pass/21-fail lifecycle result, exact-value dependency
census, original finite-difference gates and remaining scope limitations.

## Native spectral/time products, 2026-10-28

The current reviewed inventory has **438** active capability identities. All
424 previous identities and their review metadata remain unchanged; nine
spectral and five time-flux boundaries describe the new bounded sided-Cu
reduction. The existing retirement and all 29 contract exclusions remain.

Passive tracing of 25 CPU controls observes 13 of the 14 new exact raising
sites, with opcode and test-node evidence. The time sampler's exact-native-E
check remains `coverage_pending`: ordinary inputs snap to a single electric
plane, and no genuine ordinary-input trigger was available. Related negative
tests are not presented as a witness for that branch. Compound-predicate gaps
remain explicit in each entry, including permeability representations,
magnetic poles, missing-trace variants and endpoint directions.

The numerical candidate separately passes 39 actual GPU controls on the same
frozen source, genuine core 0.4.1 and the exact ABI4 library. They cover the
supported Cartesian scalar-unit-permeability, actual Cu-side and normalization
contracts; they do not implement the rejected magnetic/mapped/ambiguous-side
combinations or complete physical PCB convergence. The frozen
[joint evidence](../assessments/printed-lines-2026-10-28/native-product-joint/README.md)
retains original failed prototypes and the corrected selector run.


## Selected native queries: explicit bounded extension

The selected-component/frequency API retains the default all-six single-frequency boundary under its reviewed `components=None` predicate. The historical identifier is archived only to preserve the baseline identity chain; its replacement remains active debt. Seven newly explicit unsupported selected-query operations raise at their actual admission boundaries: coordinate/shape derivatives, analytic time-bank clocks, mapped coordinates, periodic/Bloch seams, undoing source normalization, weighted windows and matched-reference incident/scattered fields. This brings the reviewed Maxwell inventory to445 capability sites with29 unchanged contract exclusions; no scalar target or exception-class reclassification was used.

All seven new sites and the preserved default are observed at exact `RAISE_VARARGS` instructions in11CPU admission cases. [Preparation/ROOT review evidence](../assessments/printed-lines-2026-10-28/public-selected-query-v3/preparation/README.md) preserves pending proposals, actual events and manual receipts. The separate [actual11GPU qualification](../assessments/printed-lines-2026-10-28/public-selected-query-v3/actual/README.md) covers the supported Cartesian/fixed-position/raw/none/real-bank subset, not the rejected operations or full physical PCB convergence. All437 other prior guard entries retain their existing review metadata and original coverage debt.

## PMC, custom Sellmeier, pole perturbation and surface waveform tables, 2026-10-29

The reviewed Maxwell inventory now has **453** active capability identities with the
29 unchanged contract exclusions. Of the twelve new identities, eight are new
boundaries and four replace inherited identities whose predicate or message changed
(three `modified_boundary`, plus the widened breakdown message as a
`feature_boundary`). The four inherited identities are retired as
`boundary_replaced` and admit no formerly rejected case.

- PMC medium: electrostatic, zero-thickness sheet, mapped-adjoint and mode-plane
  rejections; `is_pmc` added to the local-cache and batch-geometry-adjoint
  rejections; breakdown rejection widened from PEC to PEC/PMC.
- Custom Sellmeier: trainable coefficient grids and too many distinct resonance
  levels.
- PerturbationMedium poles: gain Lorentz poles and trainable perturbations with
  nonzero pole sensitivities.
- Surface waveform tables: the callable `CustomSourceTime` adjoint rejection now
  tests `table is None`, an equivalent predicate.

Four Sellmeier/perturbation sites have matching negative assertions. The eight
PMC/waveform sites remain `coverage_pending`: their test modules are referenced,
but no individual trigger is claimed. Open-surface EFIE and MoM topology/runtime
identities are not part of this review.

## In-simulation monitors, 2026-10-29

Four in-simulation monitor boundaries are reviewed, bringing the inventory to
**457** active capability identities with the 29 unchanged contract exclusions:
FDTD-only recording of projection, Gaussian-overlap, surface field-time and
auxiliary field-time monitors, plus three reference-adapter export rejections
(an unmatched k-space hemisphere, ADE auxiliary state with no reference
equivalent, and a -normal Gaussian-overlap beam). Each cites a matching negative
assertion. No identity is retired.

The -normal Gaussian-overlap message was reworded before review: the reference
monitor records both `+` and `-` overlap amplitudes, so the limit is the
adapter's missing backward-amplitude mapping, not the reference monitor. Detached time-trace gradient and multi-GPU time-monitor
rejections raise `RuntimeError`/`ValueError` and are outside this inventory.

## Coaxial lumped ports and tabulated port drives, 2026-10-29

Four new boundaries and one replaced inherited identity bring the reviewed inventory
to **460** active capability identities with the 29 unchanged contract exclusions:

- annular coaxial branch on dispersive (ADE) row support (`fdtd/coaxial.py`);
- `CoaxialLumpedPort` bound to a circuit or embedded network (`fdtd/ports.py`);
- differentiable runs with `CoaxialLumpedPort` / `CoaxialLumpedResistor`
  (`simulation.py`, forward-only rejection);
- callable `CustomSourceTime(fn)` port drives (`fdtd/ports.py`). This replaces the
  inherited `device-native CW, GaussianPulse, or RickerWavelet` rejection, retired as
  `boundary_replaced`: sampled tables, `BroadbandPulse` and the tabulated baseband
  pulses are now admitted, with the matched-coax drive-equivalence test as positive
  evidence. Callables remain rejected.

Each new entry cites a matching negative assertion. The ten open-surface EFIE / MoM
topology identities are not part of this review.

## Multi-conductor terminal wave port, 2026-10-29

Five new boundaries bring the reviewed inventory to **470** active capability
identities after merging master's 465 (which include the boundaries-branch review:
six new entries and one retirement, 25 retired in total); no existing identity changed (the terminal mode solve enters through an
early dispatch, so the enclosing branches of the TEM / vector mode guards are intact)
and the 29 contract exclusions are unchanged:

- lossy (complex) and magnetic `TerminalWavePort` cross-sections
  (`fdtd/excitation/modes/__init__.py`, two entries);
- direct `PortExcitation` of a `TerminalWavePort` (`simulation.py`);
- differentiable runs containing a `TerminalWavePort` (`simulation.py`, forward-only);
- one sweep mixing `TerminalWavePort` and single-mode `WavePort` apertures
  (`terminal_waveport.py`).

Each entry cites a matching negative assertion in
`tests/rf/terminal_wave/test_terminal_wave_guards.py`. The ten open-surface EFIE / MoM
topology identities remain outside this review.

## Conformal thin-sheet runtime review, 2026-10-31

This scoped review adds **42** current fail-closed boundaries: 13 in the
thin-sheet compiler, 18 in the runtime, one checkpoint boundary, eight explicit
CPU-profile admission boundaries and two CPU-reference kernel boundaries. The
CPU reference backend is a sanctioned, explicit opt-in parity mirror for small
validation problems. It is never selected implicitly for a CPU scene.

The native batch runtime tuple now includes `_thin_sheets`; its modified identity
retains the existing source, wire, surface and absorber exclusions. The previous
batch identity is retired as `boundary_replaced`, with no newly admitted case.
The blanket `SurfaceMesh` finite-copper rejection in `compile_copper_faces` is
retired because the conformal runtime now carries independent side fields and
passive copper diffusion state. Positive tests run a folded finite-copper mesh
and compare a planar mesh with the existing foil path. All narrower topology,
medium, composition, gradient and lifecycle boundaries remain active.

The pending thin-sheet proposal was checked against current source rather than
copied wholesale: the dense trace-operator budget guard has been removed by the
chord implementation, runtime predicates changed, and several old test references
did not exercise the cited raises. The manifest contains only current identities.

The [site evidence](../assessments/fdtd-thin-sheet-guard-census-2026-10-31.json)
records **59 passing targeted cases**, including both positive copper controls.
Passive exception tracing observed all **43** reviewed raising sites (42 new and
one modified); none lacks a negative test. Low-level geometry/profile controls
exercise sites hidden behind earlier public validation, and material/coefficient
controls isolate the individual update assumptions. These witnesses establish
admission coverage, without extending physical or native CUDA qualification.

This review added one implemented retirement and one replaced batch identity to the
prior 27 retirements. It was merged together with the separate wire, antenna and contract
review below (master `c6c40996`). The combined manifest has **524 active** and **32 retired**
identities, and the full census passes (**7 passed**).

## Wire and antenna identity mappings, one contract reclassification, 2026-10-31

Two reviewed identities changed without a capability change, and one contract
exclusion left the census:

- `fdtd/wire.py::replay_wire_state` (inherited debt): a6e2a778 reworded the message.
  The lossy-wire predicate, scope and exception class are unchanged. The old identity
  is retired as `identity_mapping_only`; the replacement is a reviewed
  `modified_boundary` with the matching negative assertion
  `tests/fdtd/thin_wire/test_thin_wire_lossy_forward.py::test_lossy_reverse_replay_fails_closed`.
- `postprocess/antenna.py::_far_fields_from_result` (inherited debt): 88ccca7a changed
  only the loop variables. It is mapped the same way, with the new matching negative
  assertion `tests/rf/antenna/test_result_antenna.py::test_result_antenna_rejects_a_lossy_exterior`.
- `fdtd/runtime/timeplan.py` "normalize_source requires a single incident source
  spectrum": 740c6fdd changed this invalid-input check to `ValueError`.
  Authorization: "the coordinator, 2026-10-31: 740c6fdd changed an invalid-input
  check to ValueError; contract list 29→28". The contract tuple hash in
  `scripts/dev/fdtd/guard_census.py` is re-pinned to the 28 remaining exclusions; the
  hash lock itself is unchanged. Without the exclusion, the baseline guard counts as
  inherited debt, so it is retired with positive evidence that amplitude-only scaling
  is admitted and every other waveform difference raises `ValueError`.

The 42 thin-sheet and CPU-reference identities, the thin-sheet `batch.py` predicate
change and the `copper_foil.py` retirement from the 3D thin-sheet merge are reviewed
separately by that line.

## FDFD scene-monitor refusal, 2026-10-31

`simulation.py::Simulation._validate_method_scene_features` adds one new
`feature_boundary`. FDFD Results populate only `PermittivityMonitor`/`MediumMonitor`
material payloads and `FDFDConfig.observations`, so any other scene monitor is refused
before execution. It no longer runs and returns without a payload. The matching negative
assertion is `tests/fdfd/test_fdfd_monitor_refusal.py::test_fdfd_refuses_scene_monitor`,
parametrized over every other public monitor type. The manifest has **525 active** and
**32 retired** identities, and the full census passes (**7 passed**).

## Antenna lossless-exterior identity mapping, 2026-11-04

The scale sweep replaced the antenna guard's absolute imaginary-part tolerance
with a tolerance relative to each real part. The message and unsupported
lossless-exterior boundary remain unchanged. The existing review, negative test,
rationale and exit acceptance carry from `ba40c60c…:0` to `825129b7…:0`; only the
active identity, captured conditions and source line change. The old identity is
retired as `identity_mapping_only`, and the earlier `011bb701…:0` mapping now
points to the current identity while retaining the intermediate history.

On this branch alone the inventory had **546 active** and **38 retired** identities, with **28**
current contract exclusions; with the mode-source replay registration alone it had
**548 active**, **38 retired** and **28** contract exclusions. This reconciliation carries existing review evidence and
claims no capability implementation or new qualification.
## Mode-source replay eigen-adjoint debt, 2026-11-04

The frozen source proposal adds two raising-site identities, bringing the reviewed
inventory from **546 to 548 active** capability sites. The **37 retired** identities
and **28 contract** exclusions are unchanged: **576** total exception sites. All
546 existing active reviews are preserved without re-review or identity mapping.
This registration changes no runtime behavior or public feature.

Both new sites in `fdtd/adjoint/bridge.py` retain the negative boundary:
**mode-shape eigen-adjoint for vector-forward / scalar-replay on trainable
ModeSource/ModePort launch planes**. Other mismatched forward/replay families are
also refused when their launch-plane material depends on trainable inputs.
Follow-up route (i) is `fix/mode-source-replay-vector-grad`; registration records
capability debt and does not qualify that missing derivative.

- `2892c16a2566ef908b0c18a3ef9fa530038609c96dd8dc9f6899b61b51c6e456:0`
  is the standalone `_validate_mode_source_replay` raise (line 413).
- `ca5c27d10e9a5108cfa9c17fd0cf988f123fe41dd7d9fbd484c2c81d08499aea:0`
  is `_FDTDGradientBridge._validate_mode_source_replay` (line 477), which raises
  the cached verdict before rebuilding sources or reverse stepping.

`_mode_source_replay_error` builds the shared reason but does not raise. Both
sites use `NotImplementedError(error)`, so their census literal messages are
empty. The proposal labels them `modified_boundary` because an empty literal
message matches unrelated baseline guards in the same file. Source review
identifies two new feature-boundary sites, not replacements; their reviewed
origin is therefore `feature_boundary` with an explicit unsupported boundary.

The reported `347...316746` / `347...5af239` pair is not an identity change in
this checkout. The full identities ending in those suffixes are separate guards:

- `e07546afab0ab1d98abe64b4b90596770b2470b14e3234f9342c6ed29d316746:0`
  is `waveport_sweep.py::resolve_waveport_run_manifest`, rejecting unterminated
  non-wave ports; its inherited review already matches the source.
- `6c09f7085c3964ad66486133ccec6ce350b3dc96a9fc48526a011df82c5af239:0`
  is `fdtd/resume.py::_preflight_observer_origin`, rejecting old or mismatched
  observer-origin/provider metadata. Commit `f36cdaf1` already registered it;
  its review also matches the source. Neither guard moved because of the replay
  edits, and the proposal removes no active identity.

Matching negative assertions in
`tests/gradients/test_mode_source_replay_guard.py` cover spatial ModeSource and
ModePort refusal, dense/sparse vector launches, complex-forward/real-replay
mismatch, multiple-source naming, exact epsilon/mu sampling-edge dependencies,
and cached refusal with material-graph invalidation. Fixed launch planes and
matching scalar families retain their admission. The manifest also pins
`test_rectangular_guide_replay_against_richardson_bound[source]`: its native
backward refusal and ordinary-forward finite differences require CUDA and are
not executed by this CPU-only review. These references claim matching negative
assertions, not passive raising-site tracing or GPU qualification.

CPU gates in the `maxwell` environment with `CUDA_VISIBLE_DEVICES=` all returned
rc 0: `tests/api/public/test_guard_census.py` (**7 passed**),
`tests/validation/benchmark/test_integration_batch.py` (**82 passed**), and
`tests/gradients/test_mode_source_replay_guard.py` (**40 passed, 4 skipped**).
The four skipped cases require CUDA, including both FD-pinned guide cases.

## Dense vector replay reconciliation, 2026-11-04

Route (i) narrows the preceding review: lossless conductor-free trainable
ModeSource/ModePort planes now use matching `vector_dense_torch` forward/replay
profiles for actual operator row counts `n <= 4096` with an isolated selected
eigenpair. Before the scale-guard merge, the inventory contained **552 active**, **43 retired** and
**28 contract** identities: **580** current exception sites. Of the previous
548 active entries, **540 reviews are unchanged**, **two are narrowed**, and
**six are carried to changed predicate identities**; **four new boundaries**
are registered. No shared bridge raising site is removed.

The standalone `2892c16a…:0` and cached `ca5c27d1…:0` identities persist exactly.
Their `unsupported_boundary` now covers remaining family mismatches and names
the dedicated cap, isolation and loss limits. Both retain `positive_tests` and
a `narrowing_review` using the retirement evidence fields (`retirement_review`,
`retirement_kind`, `retired_boundary`, `retirement_reason`, `positive_tests`,
`replacement_boundaries`). This retires only the supported subset of the old
refusal, without creating a fictitious removed identity or retiring PEC/TEM or
complex replay debt. Positive acceptance references are:

- `test_mode_source_replay_guard.py::test_rectangular_guide_replay_against_richardson_bound[source]`;
- `test_mode_source_vector_replay.py::test_metre_micrometre_gradient_scaling`;
- `test_mode_source_vector_replay.py::test_bend_material_region_crosses_launch_plane`.

These tests live under `tests/gradients/` and require CUDA. They are retained
acceptance gates, not claims of GPU execution in this CPU-only reconciliation.
The source FD case now requires an admitted, FD-correct gradient. Existing
WavePort admission pins are unchanged. The sampling-edge and cached-refusal
fixtures now use a synthetic `vector_pec_dense_torch` family, preserving tests
of actual remaining family mismatch rather than refusing the admitted subset.

Four new exact raising sites are reviewed with matching negative assertions in
`tests/gradients/test_mode_source_vector_replay.py`:

| Identity prefix | Unsupported boundary | Negative assertions |
| --- | --- | --- |
| `9dc2f013…:0` | Actual dense operator rows `n > 4096` | `test_above_cap_refuses`, `test_above_cap_source_refuses_before_stepping` |
| `bc5f466d…:0` | Relative selected eigen-gap at or below `sqrt(eps64) ||A||_infinity / abs(lambda)` | `test_square_guide_refuses_degenerate_polarizations` |
| `907a99f0…:0` | Lossy/conductive relative epsilon or mu above the router's `1e-7` imaginary threshold | `test_lossy_trainable_vector_planes_refuse_before_real_projection`, `test_yee_operator_refuses_lossy_plane`, `test_lossless_threshold_uses_router_precision`, `test_conductive_trainable_launch_plane_refuses` |
| `8b3671b5…:0` | Selected eigenvalue imaginary part above `sqrt(eps64) abs(lambda)` | `test_selected_eigenpair_checks_imaginary_part[1e-06-True]` |

The remaining shared bridge negatives are the complex-forward/real-replay
control, exact epsilon/mu sampling-edge dependencies and cached graph
invalidation in `test_mode_source_replay_guard.py`. Cap/gap/loss controls pin
their dedicated earlier raises; they are not raising-site witnesses for the
bridge. All new controls are recorded as matching assertions, without a claim
of passive raising-site tracing.

Six old conductor/TEM identities are retired as `boundary_replaced`, with
replacement identities carrying their original reviews and explicit coverage
debt. The enclosing dispatch now admits torch vector replay; these conductor
capabilities remain unsupported:

| Previous identity prefix | Current identity prefix | Preserved boundary |
| --- | --- | --- |
| `f822b1e4` | `f667ef58` | Lossy TEM locally uniform axial spacing |
| `db458bb9` | `fa8fdab2` | Spatially varying lossy quasi-TEM potential |
| `6ca19a7b` | `6c5ae0c3` | Lossy TEM material-profile adjoint |
| `97de73c9` | `fc7f1af7` | Lossy PEC-aware vector solve |
| `9870e7af` | `69cc9701` | Lossy TEM pole projection |
| `4b8eabbf` | `9a8653c0` | Internal-PEC material eigen-adjoint |

All six retain `coverage_pending`; module references are not upgraded to
individual trigger evidence. Positive profile-equality and fixed-PEC admission
controls explain the dispatch change, without claiming conductor adjoints.

## Scale-guard merge into dense vector replay, 2026-11-04

The current reviewed inventory has **552 active**, **44 retired** and **28
contract** identities: **580 current exception sites**. Reconciliation began
with the union of both parents and a fresh proposal from the merged source.
Seven obsolete active identities already had reviewed retirement records: six
conductor/TEM predicate replacements from route (i), and the antenna identity
carry from the scale-guard merge. Those records are retained once each; no
pending review or duplicate retirement remains. Active source lines are refreshed.

The two shared replay guards keep route (i)'s narrowed boundaries, positive
acceptance references and narrowing reviews. Its four added cap/gap/loss/complex
eigenpair boundaries and six reviewed conductor/TEM replacements remain intact.
Master's antenna carry `ba40c60c…:0 -> 825129b7…:0` and the earlier
`011bb701…:0` mapping to the current identity are preserved, including the
intermediate history. Compared with route (i)'s pre-merge ledger, this replaces
one active antenna identity and adds one retirement; no capability is added or
retired by this identity reconciliation.

Torch dense vector launches and scipy launches use the same unit-power
normalizer: `abs(power) <= 64 * eps(dtype) * reference`, where the reference
integrates absolute transverse field products with the identical Yee quadrature.
The obsolete torch-only aperture rescaling is removed. Vector finalization uses
no absolute propagation-constant or imaginary-profile floor. Modal surface
injection uses the same `64 * eps(float64)` pruning threshold relative to each
E/H profile family on both paths. Forward/replay profile and patch equality
remain exact acceptance requirements at metre and micrometre scales.

The audit also removed dimensional floors in the shared numpy candidate
selectors reached by the torch solve: reciprocal-power overlaps handle exactly
zero norms explicitly, residuals divide by every nonzero reference, and
propagation-constant grouping, duplicate detection and effective-index
diagnostics use relative scales. Nodal and Yee candidates are checked at length
scales `1e-6` and `1e6`. These helpers remain shared with the scipy path; no new
solver family or capability raising site is introduced.

Final CPU gates in the `maxwell` environment with `CUDA_VISIBLE_DEVICES=` each
returned pytest rc 0:

| Gate | Passed | Skipped |
| --- | ---: | ---: |
| `tests/api/public/test_guard_census.py` | 7 | 0 |
| `tests/validation/benchmark/test_integration_batch.py` | 82 | 0 |
| `tests/gradients/test_mode_source_replay_guard.py` | 40 | 4 |
| `tests/gradients/test_mode_source_vector_replay.py` | 81 | 9 |
| `tests/sources/mode/` | 148 | 42 |
| `tests/sources/test_source_term_count_roundoff.py` | 0 | 18 |
| Total | 358 | 73 |

The exact forward/replay profile, preparation-patch and fp64 material-replay
checks also passed in three consecutive CPU runs (each **9 passed, 4 skipped**,
rc 0). CUDA-only cases remain skipped; these CPU results make no claim of native
FDTD gradient or GPU term-count qualification.


## Fixed-k broadband slab injection, 2026-11-01

The complex fixed-k incident line makes broadband TFSF-slab plane waves exact
under fixed k. The directional-source guard
`af76d41cf4d34b6e812b08aaa77d0cdd2a521e08f0802c4cc7fd294974507130:0` is retired
and replaced by `84b8fe15…`, which keeps beams, mode and custom-field sources
and soft plane waves CW-only. The GPU physical gates are
`scripts/dev/fdtd/bloch_floquet_acceptance/oblique_slab_airy.py` and
`broadband_checks.py`; their results are in the batch-4 merge note.
Automatic wavevector resolution still requires CW.

## Fixed-k merge reconciliation, 2026-11-04

Merging master `2e8308d2` preserves its reviewed boundaries and carries the
fixed-k slab admission above. The merged ledger contains **553 active**, **45
retired** and **28 contract-excluded** identities: **581 current exception
sites**. The one active replacement is `af76d41c...:0 -> 84b8fe15...:0`; its
retirement and positive slab controls survive alongside all master reviews.
Sixteen source-line anchors are refreshed without changing their identities,
predicates or review evidence. No additional raising site is introduced by the
provider-reference registration or exported incident-spectrum time conversion.
The timing scope and deferred raw-exact work are recorded in
`docs/memory/bloch-fixed-k-master-reconciliation-2026-11-04.md`.

## 0.7.0 construction refusals (p2e), 2026-11-08

Two reviewed feature boundaries are added; no identity is changed or retired,
so the ledger holds **586 active** and **50 retired** identities (after the p2d merge).

- `simulation.py::_reject_optical_structure_assignments` refuses
  `Structure.phase_screen` and `uv`/`face_uv` for every Maxwell solver (audit
  G16/N7). It runs first in `_validate_method_scene_features` (construction,
  `prepare()`, `run()`) and in `Simulation.electrostatic/capacitance`. The
  deeper FP64-forward (`ValueError`) and mapped-admission refusals stay in place;
  their tests now drive them directly.
- `scene.py::MaterialRegion.__post_init__` refuses any `symmetry` other than
  `None` (audit G17); the field was never consumed.

The trainable-coefficient refusals (G10-G12) reuse the existing
`_fixed_inputs.py::require_fixed_input` raise site and add no identity. Line
anchors of unchanged identities are not refreshed in this change.

## FDFD construction-time refusals, 2026-11-08 (0.7.0 phase 2a)

`simulation.py::_reject_unsupported_fdfd_scene` adds one reviewed `feature_boundary`
(`661ebd0f…:0`). At `Simulation.fdfd(...)` construction (and again in `prepare()`/`run()`)
it raises one `NotImplementedError` listing every FDFD-unsupported scene feature. New
items, all previously accepted and dropped or approximated: `LossyMetalMedium`,
`SurfaceImpedanceMedium` and `CopperFoil` conductors (solved as vacuum), `ModulationSpec`
(static eps), lumped elements, terminated, wire-bound and modal ports, `InternalAbsorber`,
`SurfaceMesh` structures, `SubpixelSpec(pec="conformal")` (staircased) and CW sources whose
frequency differs from a solve frequency. Four reviewed identities move into the same
guard without a capability change and are retired with their tests as positive evidence
that the refusal still fires: the three `fdfd/solver.py::_validate_supported_fdfd_materials`
inherited-debt guards (gyromagnetic, nonlinear, full tensor) and the FDFD scene-monitor
refusal `374da1d4…:0`. The face-kind and symmetry `ValueError`s leave `FDFD.__init__` for
the same aggregate. Negative assertions are in
`tests/fdfd/test_fdfd_silent_ignore_refusals.py`. On top of the p2d and p2e merges the
manifest has **583 active** and **54 retired** identities.

## 0.7.0 core-path flag removal (W7 row 3), 2026-11-09

Removing the public `fdtd_core_path` option deletes `core_path.py::validate_core_path_request`,
so its safety-refusal identity `7ded1a98…:0` ("fdtd_core_path=True is blocked by: ") is retired
and its `SAFETY_GUARDS` entry is removed from `scripts/dev/fdtd/guard_census.py`. Nothing is
implemented: closed-transverse CPML stays a core-path blocker, and automatic admission keeps it on
the guarded existing path with `FDTDCoreFallbackWarning`; the retirement's positive tests assert
that fallback. The two `configuration_guards` records for the explicit-True refusals
(blocked scene; `initialize=False`) are removed because neither raise exists any more. The
manifest holds **586 active** and **55 retired** identities.

## W7 row 4 core admissions, 2026-11-10

The core path admits PMC faces (including PMC symmetry planes), permeability, magnetic conductivity,
Mur and mode-ABC faces, `MediumMonitor` and the projection and Gaussian-overlap monitors. These were
`core_path_blockers` labels, not raises, so no census identity is added, changed or retired: the
proposal from `scripts/dev/fdtd/guard_census.py` is identical on the branch and on its base. The
separate-physics core graph's Mur/mode-ABC `assert` is removed; assertions are not tracked. The
remaining blocker labels are asserted by `tests/fdtd/core_path` (closed-transverse CPML,
configuration blockers) and listed in `docs/plans/fdtd-single-path-cuda.md`.

## W7 row 5 closed-transverse CPML on the core path, 2026-11-11

`core_path_blockers` drops `closed-transverse CPML (passivity guard)`: the core path samples the guard at
its K-step chunk boundaries. The label was not a raise, so no identity is added or retired; the guard's
own `FDTDPassivityError` identities are unchanged. Only the retired `7ded1a98…:0` entry is updated: its
positive tests now assert core admission and the core-chunk trip and verdict gates
(`tests/boundaries/cpml/test_passivity_core_chunks.py`), and its retirement reason states that the
capability is implemented. Counts are unchanged.

## W7 row 16c failure policy, 2026-11-13

The last `core_path_blockers` labels ("non-CUDA execution", "unsupported source type") and the fallback machinery
(`_FDTDCoreFallback`, the existing-path rebuilds, `FDTDCoreFallbackWarning`) are removed; a failed core load,
activation or capture raises its own error and a failed accelerated setup retries with the torch setup. None of these
were census identities, so no identity is added, changed or retired: the proposal from
`scripts/dev/fdtd/guard_census.py` is identical to its base except line numbers. The retired `7ded1a98…:0` entry's
positive test `test_default_auto.py::test_closed_transverse_absorbers_are_not_request_blockers` is replaced by
`test_default_auto.py::test_every_scene_is_admitted`. Counts are unchanged.

## W7 row 7 resume on the core path, 2026-11-11

`core_path_decision` drops its `resume_from` and `initialize=False` refusals. They were decision
labels, not raises, so no identity is added, changed or retired. The resume refusals that are raises
(detached forward, distributed transport, breakdown observers, finite-conductor wires) are unchanged.

## 0.7.0 capability registry (phase 3c), 2026-11-09

The construction-time scene refusals move into `witwin/maxwell/capability/`: a key extractor
(`keys.py`), one table per method (`tables/`) and a default-deny check (`check.py`) that raises one
`UnsupportedFeatureError(items=...)`. Twelve reviewed identities are retired as moved, not
implemented: `simulation.py::_reject_unsupported_fdfd_scene`, `_reject_optical_structure_assignments`,
the four `_reject_folded_domain_monitors` refusals, the `PlanarPEC`, `LayerStack`, homogeneous-background
(inherited debt) and in-simulation-monitor refusals of `Simulation._validate_method_scene_features`,
`compiler/electrostatic.py::require_supported_electrostatic_scene` and
`compiler/mom_scene.py::require_supported_mom_scene` (module deleted). Each retired entry carries
`registry_rows` (`<method>:<key>`) naming the table rows that now hold the refusal; its positive tests
are `tests/api/public/test_capability_registry.py::test_refused_rows_are_reported`, which observes every
refusal row, plus the original negative tests. The single `UnsupportedFeatureError` raise site in
`CapabilityReport.raise_if_unsupported` joins the census when p3b tracks the type. The mapped
material-cache refusal keeps its identity. The manifest holds **561 active** and **84 retired**
identities.

Retired identities whose refusal moved into the capability registry, with the rows that hold it
(checked against the tables by `tests/api/public/test_capability_docs.py`):

<!-- capability-table: registry-rows -->
<!-- Generated by scripts/dev/capability_tables.py; do not edit. -->

| Identity prefix | Former scope | Registry rows (`method:key`) |
| --- | --- | --- |
| `457c10d9…` | `Simulation._validate_method_scene_features` | `fdtd:background.homogeneous`, `fdfd:background.homogeneous`, `fdtd:source.MaxwellPlaneModes`, `fdfd:source.MaxwellPlaneModes` |
| `05f6abb1…` | `Simulation._validate_method_scene_features` | `fdfd:monitor.AngleProjectionMonitor`, `fdfd:monitor.CartesianProjectionMonitor`, `fdfd:monitor.KSpaceProjectionMonitor`, `fdfd:monitor.GaussianOverlapMonitor`, `fdfd:monitor.AstigmaticGaussianOverlapMonitor`, `fdfd:monitor.SurfaceFieldTimeMonitor`, `fdfd:monitor.AuxFieldTimeMonitor`, `mom:monitor.AngleProjectionMonitor`, `mom:monitor.CartesianProjectionMonitor`, `mom:monitor.KSpaceProjectionMonitor`, `mom:monitor.GaussianOverlapMonitor`, `mom:monitor.AstigmaticGaussianOverlapMonitor`, `mom:monitor.SurfaceFieldTimeMonitor`, `mom:monitor.AuxFieldTimeMonitor`, `mom.layered:monitor` |
| `f6c2f43a…` | `Simulation._validate_method_scene_features` | `fdfd:geometry.planar_pec`, `mom:geometry.planar_pec` |
| `8ae42b59…` | `Simulation._validate_method_scene_features` | `fdtd:background.layer_stack`, `fdfd:background.layer_stack`, `mom:background.layer_stack` |
| `024542fa…` | `_reject_optical_structure_assignments` | `fdtd:structure.optical_assignment`, `fdfd:structure.optical_assignment`, `mom:structure.optical_assignment`, `mom.layered:structure.optical_assignment`, `electrostatic:structure.optical_assignment`, `capacitance:structure.optical_assignment` |
| `92ca3cdf…` | `_reject_folded_domain_monitors` | `fdtd:symmetry.closed_surface_auto_grid` |
| `aa185432…` | `_reject_folded_domain_monitors` | `fdtd:symmetry.closed_surface_image_plane` |
| `a35d724b…` | `require_supported_electrostatic_scene` | `electrostatic:coordinates`, `electrostatic:symmetry`, `electrostatic:structure.optical_assignment`, `electrostatic:material.lossy_metal`, `electrostatic:material.surface_impedance`, `electrostatic:material.copper_foil`, `electrostatic:material.medium2d`, `electrostatic:material.sigma_e`, `electrostatic:material.core_surface_fields`, `electrostatic:source`, `electrostatic:monitor`, `electrostatic:port`, `electrostatic:lumped_element`, `electrostatic:internal_absorber`, `electrostatic:network`, `electrostatic:thin_wire`, `electrostatic:circuit`, `electrostatic:electrostatic_terminal.trainable_geometry`, `capacitance:coordinates`, `capacitance:symmetry`, `capacitance:structure.optical_assignment`, `capacitance:material.lossy_metal`, `capacitance:material.surface_impedance`, `capacitance:material.copper_foil`, `capacitance:material.medium2d`, `capacitance:material.sigma_e`, `capacitance:material.core_surface_fields`, `capacitance:source`, `capacitance:monitor`, `capacitance:port`, `capacitance:lumped_element`, `capacitance:internal_absorber`, `capacitance:network`, `capacitance:thin_wire`, `capacitance:circuit`, `capacitance:charge_density`, `capacitance:electrostatic_terminal.floating`, `capacitance:electrostatic_terminal.trainable_geometry` |
| `8f8ee9d8…` | `require_supported_mom_scene` | `mom:material.pmc`, `mom:material.core_surface_fields`, `mom:material.trainable_mu_r`, `mom:internal_absorber`, `mom:source.injection_tfsf`, `mom:source.injection_axis`, `mom:source.MaxwellPlaneModes.trainable` |
| `661ebd0f…` | `_reject_unsupported_fdfd_scene` | `fdfd:background.homogeneous`, `fdfd:background.layer_stack`, `fdfd:coordinates`, `fdfd:structure.optical_assignment`, `fdfd:symmetry`, `fdfd:subpixel.pec_conformal`, `fdfd:boundary.periodic`, `fdfd:boundary.bloch`, `fdfd:boundary.pmc`, `fdfd:boundary.mur`, `fdfd:boundary.mode_abc`, `fdfd:geometry.surface_mesh`, `fdfd:geometry.planar_pec`, `fdfd:material.lossy_metal`, `fdfd:material.surface_impedance`, `fdfd:material.copper_foil`, `fdfd:material.gyromagnetic`, `fdfd:material.nonlinear`, `fdfd:material.full_tensor`, `fdfd:material.modulation`, `fdfd:lumped_element`, `fdfd:network`, `fdfd:thin_wire`, `fdfd:internal_absorber`, `fdfd:source.MaxwellPlaneModes`, `fdfd:source.cw_frequency_mismatch`, `fdfd:monitor.PointMonitor`, `fdfd:monitor.WireMonitor`, `fdfd:monitor.PlaneMonitor`, `fdfd:monitor.FluxMonitor`, `fdfd:monitor.IncidentPowerDensityMonitor`, `fdfd:monitor.FinitePlaneMonitor`, `fdfd:monitor.ClosedSurfaceMonitor`, `fdfd:monitor.PowerLossMonitor`, `fdfd:monitor.BreakdownMonitor`, `fdfd:monitor.ComponentStressMonitor`, `fdfd:monitor.DipoleEmissionMonitor`, `fdfd:monitor.FieldTimeMonitor`, `fdfd:monitor.FluxTimeMonitor`, `fdfd:monitor.ModeMonitor`, `fdfd:monitor.DiffractionMonitor`, `fdfd:monitor.AngleProjectionMonitor`, `fdfd:monitor.CartesianProjectionMonitor`, `fdfd:monitor.KSpaceProjectionMonitor`, `fdfd:monitor.GaussianOverlapMonitor`, `fdfd:monitor.AstigmaticGaussianOverlapMonitor`, `fdfd:monitor.SurfaceFieldTimeMonitor`, `fdfd:monitor.AuxFieldTimeMonitor`, `fdfd:port.ModePort`, `fdfd:port.WavePort`, `fdfd:port.TerminalWavePort`, `fdfd:port.termination`, `fdfd:port.wire_binding` |
| `a0a889f6…` | `_reject_folded_domain_monitors` | `fdtd:symmetry.projection_monitor` |
| `af173e45…` | `_reject_folded_domain_monitors` | `fdtd:symmetry.diffraction_folded_axis` |
| `3f13fab4…` | `Simulation._validate_layered_mom_gradient_fields` | `fdtd:structure.trainable_conductor_loss`, `fdfd:structure.trainable_conductor_loss`, `mom:structure.trainable_conductor_loss`, `electrostatic:structure.trainable_conductor_loss`, `capacitance:structure.trainable_conductor_loss` |
| `7348947e…` | `_require_electrostatic_boundary` | `electrostatic:boundary.pml`, `electrostatic:boundary.periodic`, `electrostatic:boundary.bloch`, `electrostatic:boundary.pec`, `electrostatic:boundary.pmc`, `electrostatic:boundary.mur`, `electrostatic:boundary.mode_abc`, `capacitance:boundary.pml`, `capacitance:boundary.periodic`, `capacitance:boundary.bloch`, `capacitance:boundary.pec`, `capacitance:boundary.pmc`, `capacitance:boundary.mur`, `capacitance:boundary.mode_abc` |
<!-- /capability-table: registry-rows -->

## Layered MoM conductor-loss gradient admission moves into the registry, 2026-11-09

The RFIC layered MoM round 1 (GR1) refusal `simulation.py::Simulation._validate_layered_mom_gradient_fields`
(a `requires_grad` `LossyMetalMedium.conductivity`, `SurfaceRoughness` field or `PlanarPEC.physical_thickness`
outside layered MoM) becomes the registry key `structure.trainable_conductor_loss`: refused by the fdtd, fdfd,
mom, electrostatic and capacitance tables, supported by mom.layered. Its identity `3f13fab4...:0` is retired as
moved with `registry_rows`; the electrostatic and capacitance runners, which never reached the old
`Simulation.__init__` check, now refuse a trainable sheet thickness instead of reading it as a fixed value. The
manifest holds **563 active** and **89 retired** identities.

## 0.7.0 raise migration (phase 3b), 2026-11-09

The census now tracks `UnsupportedFeatureError` as well as `NotImplementedError`
(`guard_census.py::TRACKED_EXCEPTIONS`). The codemod
`scripts/dev/codemods/070_p3b_raise_migration.py` rewrote 461 capability refusals outside
`witwin/maxwell/mom/` from `NotImplementedError(message)` to
`UnsupportedFeatureError(message, feature=..., method=...)`; the per-site classification is
`scripts/dev/codemods/070_p3b_raise_sites.csv` (461 `capability`, 1 `abstract`:
`SceneModule.to_scene`, and 4 `pinned` refusals inside `media.py::Material.evaluate_static` and
`Material.evaluate_at_frequency`, scopes the consumer-metadata bridge pins byte for byte; they keep
`NotImplementedError` and their identities until that bridge is re-baselined). The rewrite changes each raise expression and therefore each identity,
but no message, scope or predicate. `scripts/dev/fdtd/raise_migration_receipt.json` maps every old
identity to its new identity and expression; `guard_census.py` carries reviews (and the inherited
baseline, after the MoM rename receipt) through it, so no migrated entry is reopened or retired.
A `guard:<8 hex>` feature key is the identity prefix of the raise before migration; it stays fixed
when the message is later reworded, and grepping this manifest for it finds the entry.

Four `UnsupportedFeatureError` raises that phases 2f and 3c added outside the census enter it as
reviewed `feature_boundary` entries with observed raising sites:
`adjoint_inputs.py::require_material_gradient_channels`, `require_claimed_trainable_inputs`,
`simulation.py::_reject_uncensused_trainable_tensors` and
`capability/model.py::CapabilityReport.raise_if_unsupported`. The manifest holds **567 active**
and **89 retired** identities; the 28 contract exclusions are unchanged. MoM raise sites keep their
identities until the RFIC line runs the codemod with `--scope mom`.

## 0.7.0 frozen solver configs (api070/p4a), 2026-11-10

`material_cache` moves from `Simulation.__init__` into `FDTDConfig.material_cache`. Two reviewed
cache refusals keep their message, scope, exception and refused set but now read the cache from the
config, so their identities change; the old identities are retired as `boundary_replaced` with the
new identities as `replacement_ids`: the mapped-coordinate cache refusal `347d6659…:0` →
`18a2cf8d…:0`, and the distributed/gradient cache refusal `b5e95554…:0` → `e31c0b4a…:0` (inherited
debt, now a `modified_boundary` with the negative test
`tests/fdtd/core_path/test_default_auto.py::test_material_cache_refuses_gradient_execution`). No
capability is implemented. The removed `Simulation(reference_frequency=..., material_cache=...)`
argument checks were plain `ValueError`/`TypeError` validation outside the census. The manifest
holds **567 active** and **91 retired** identities.

## 0.7.0 typed CPML config (api070/p4b), 2026-11-10

The wave-port, terminal wave-port and modal-port sweep runners (and the direct wave-port excitation,
which shares the wave-port column runner) ran every column with `material_cache=None`, so a configured
`FDTDConfig.material_cache` was silently ignored. One new `feature_boundary` refusal,
`material_cache.py::refuse_column_material_cache` (`22efb9ba…:0`, feature key `guard:96a41650`), is called
first by `waveport_sweep._execute_columns`, `terminal_waveport.run_terminal_waveport_sweep` and
`modal_sweep.run_modal_port_sweep`, with four negative tests. A new raise has no pre-migration identity,
so its `guard:<8 hex>` key is the identity prefix of the raise written with an empty key
(`feature="guard:"`). The `FDTDConfig.cpml` checks are `SetupError` input validation outside the census.
The manifest holds **568 active** and **91 retired** identities.

## 0.7.0 static solver configs (api070/p4d), 2026-11-10

`Simulation.electrostatic/capacitance` return a `Simulation` (PLAN section 6 Q6), so construction,
`check()`, `prepare()` and `run()` admit the scene through the electrostatic/capacitance method tables.
Those tables admitted every `Scene.boundary` face kind while `compile_electrostatics` refused any kind
but `none`; the compile refusal `compiler/electrostatic.py::_require_electrostatic_boundary`
(`7348947e…:0`, feature key `guard:8ff680a9`, inherited debt) is retired as moved with `registry_rows`:
`boundary.<kind>` for every kind but `none` is an `INHERENT` row of both tables, observed by
`tests/electrostatic/test_silent_ignore_refusals.py::test_full_wave_boundary_is_refused`. The predicate is
unchanged (some face kind is not `none`), and `compile_electrostatics(scene)` keeps the refusal through its
`check_scene` call. The new `ElectrostaticConfig`/`CapacitanceConfig` checks are `TypeError`/`ValueError`
input validation outside the census. The manifest holds **567 active** and **92 retired** identities.

## 0.7.0 namespaces: transistor placeholders removed (api070/p5b), 2026-11-10

Plan Q10 (user decision 2026-11-08) removes the reserved `BJT` and `MOSFET` classes from
`circuit_devices.py`: they only raised, and a public symbol that always raises is not an API. Their two
raises were contract exclusions, so the `CONTRACT_GUARDS` list goes from 28 to 26 and the contract tuple
hash in `scripts/dev/fdtd/guard_census.py` is re-pinned (the hash lock itself is unchanged). Without the
exclusion both baseline raises count as inherited debt; they are retired with the removal as positive
evidence (`tests/rf/nonlinear/test_nonlinear_device_contract.py::test_transistor_placeholders_are_removed`,
`tests/api/public/test_public_api.py::test_removed_public_names_are_absent`). No active identity changes.
The manifest holds **567 active** and **94 retired** identities with **26 contract** exclusions.
