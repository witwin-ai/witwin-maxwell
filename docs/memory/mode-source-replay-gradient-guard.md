# Mode-source replay gradient guard

Trainable, lossless, conductor-free vector launch planes use matching
`vector_dense_torch` forward and replay profiles when the actual operator has
`n <= 4096` rows and the selected eigenpair is isolated. Their source-profile,
propagation-constant and injection-coefficient derivatives are admitted. Above
the cap, near-degenerate selected eigenpairs, lossy/conductive trainable planes
and remaining mismatched mode families still refuse. The dense vector replay
sections below describe the implementation and numerical limits.

The initial guard history below records why scalar replay was insufficient;
its blanket trainable-vector refusal has been narrowed by this extension.

The cached validator admits `vector_dense_torch` only with the torch-vector
launch flag, preserving the forward/replay family match. Other mismatches still
return a refusal reason so rejected verdicts remain cached. The accepted verdict
also owns the vector material graph and restores it after backward cleanup for
repeated source-profile pullbacks; new forwards and input versions invalidate it.
The vector eigensolve retains its operator-size and selected-mode isolation
refusals.

Validation after resolving the rebase onto `a7f80efa`, with the maxwell
interpreter, `CUDA_VISIBLE_DEVICES=`, `OMP_NUM_THREADS=1`, and
`OPENBLAS_NUM_THREADS=1`:

- Gradients, mode sources, and integration-batch selection: 199 passed,
  382 skipped, 20 warnings, rc 0 (371.37 s).
- Guard suite, including sampling edges and verdict invalidation: 35 passed,
  3 skipped, rc 0.
- Combined guard/vector suite, including repeated cached vector pullbacks after
  solver cleanup: 52 passed, 10 skipped, 18 warnings, rc 0.

The native FDTD adjoint launches a forward mode under `no_grad`, then rebuilds
source terms against differentiable component-permittivity leaves. Before the
dense vector extension, a lossless forward launch used `vector_dense` or
`vector_sparse`, whereas
`fdtd/excitation/modes/common.py:596` disables the vector solver when
`_mode_source_rebuild_from_fields` is set. Differentiating that scalar profile
did not differentiate the launched vector profile. Trainable supported planes
now retain their node-material graph and use the same torch vector family on
both passes; fixed planes retain the original dispatch.

The scalar magnetic-vector construction at
`witwin/maxwell/fdtd/excitation/modes/__init__.py:985` uses
`-cross(direction, polarization)`. For a transverse unit electric polarization,
`E cross H` therefore gives signed axial power `-direction_sign`. The vector
launch uses its eigenvector's electric and magnetic components instead. This
change leaves the fixed-plane/scalar forward solve, normalization and injection
untouched. Supported trainable planes differentiate the vector H construction
and its power normalization, without changing the scalar H convention.

`fdtd/adjoint/bridge.py:330` compares the recorded forward `mode_solver_kind`
with the scalar family selected by replay's component-permittivity plane. It
runs on the first backward for each forward material graph, at
`fdtd/adjoint/bridge.py:1187`, before source rebuild or reverse stepping. Only
mismatched families with a nonzero material-input
VJP on the aperture raise `NotImplementedError`. The message names the source
(including a generated port source name), declares its plane material trainable,
and identifies both solver families. The probe samples the forward node-grid
epsilon/mu planes and the replay Yee-component epsilon plane; a grid-wide
`requires_grad` flag alone would incorrectly reject downstream design regions.
The weighted plane probes retain the temporary material graph across aperture
tensors; it is released after the verdict.
Scenes that keep forward source terms rather than rebuilding them are excluded
using the same rebuild-kind admission as `core._resolved_source_term_lists`.

Forward solver metadata was already recorded at
`fdtd/excitation/injection.py:1352`. `fdtd/adjoint/core.py:490` now copies compiled
source dictionaries for replay so initialization cannot overwrite that record.

## Solver-family admission

- `scalar_dense` / `scalar_dense_torch` and `scalar_sparse` /
  `scalar_sparse_implicit` implement the same real scalar profile family and
  retain their existing admission. The host scalar-forward/profile comparison
  and dense/sparse derivative checks in
  `tests/sources/mode/test_mode_solver_gradients.py:136` remain unchanged.
- `scalar_complex_dense` / `scalar_complex_sparse` remain admitted by this guard
  when replay also sees a complex scalar plane. The existing non-Hermitian
  eigen-adjoint refusal at `modes/__init__.py:878` still applies if that complex
  plane itself requires gradients. This guard does not add complex adjoints.
- A lossy complex scalar forward with a real component-permittivity replay is
  a mismatch, and now raises if the plane depends on trainable inputs. Fixed
  lossy planes with design inputs elsewhere are unchanged. Thus a scalar prefix
  alone is insufficient to establish agreement.
- Lossless conductor-free vector planes within the dense cap and selected-mode
  isolation bound use matching torch forward/replay profiles and are admitted.
  PEC-aware, TEM or other profile kinds still differing from scalar replay
  raise when the launch plane is trainable. Existing internal-PEC and
  good-conductor eigen-adjoint refusals remain in their primary mode paths.

## Exposure census

All paths below refer to `witwin/maxwell/` unless stated otherwise.

| Consumer | Path and replay exposure | Coverage or reason for exclusion |
| --- | --- | --- |
| ModeSource, including surface TFSF injection | `fdtd/excitation/injection.py` solves the launch; `fdtd/adjoint/core.py` rebuilds its source terms. | Spatial tests admit supported trainable torch-vector planes and fixed/downstream controls. Remaining family mismatches refuse. The CUDA rectangular-guide cases require FD-correct source and downstream gradients. |
| ModePort source | `scene.py` calls `ports.py`, generating `<port name>::source`. It compiles as an ordinary ModeSource and reaches the same replay. | Host tests admit supported trainable/fixed launch planes and downstream controls for generated port sources. CUDA profile-equality and source-gradient tests exercise the same torch-vector route. |
| ModePort direct PortExcitation | `simulation.py:1040` rejects ModePort because it is not an RF port. | New host exception-contract test proves it cannot enter this replay via direct excitation. Passive ModePorts produce monitors only. |
| WavePort sweep and direct excitation columns | `waveport_sweep.py:565` builds source columns; `:717` sets `_fixed_waveport_mode_sources`; `simulation.py:1889` rewrites their kind to `fixed_waveport_mode_source`. The direct excitation uses the same manifest/columns (`simulation.py:1366`). | `core.py:594` retains forward terms for kinds outside the rebuild set. New host test proves fixed-column exclusion. The existing cross-section guard at `waveport_sweep.py:372` rejects trainable launch planes before column execution; `tests/gradients/test_fdtd_waveport_adjoint.py:85` covers it. |
| TerminalWavePort mode columns | `terminal_waveport.py:356` builds columns and `:443` sets the same fixed-source flag. | The same kind rewrite and exclusion prevent source replay rebuilding. This exclusion does not qualify terminal-port material gradients or add a terminal eigen-adjoint. |
| Mode monitors and overlaps | `compiler/monitors.py:155` stores mode specifications; `fdtd/observers.py:1435` returns them. `postprocess/modal.py:77` obtains the result context and `:171` solves a local reference profile for overlaps. | Postprocessing does not append the local reference source to `_compiled_sources`, set the rebuild flag, or call the source-replay builder. Monitor field cotangents may enter backward, but the reference profile does not enter source replay. |
| ModeABC and internal mode absorbers | `fdtd/boundary/abc.py:79` creates a local compiled source, solves it at `:117`, and returns beta at `:133`. Initialization is at `:291` and `:355`. | These local sources never enter `_compiled_sources`; beta builds one-way boundary coefficients rather than injection terms. The existing coefficient-gradient refusal is at `:119`. No ABC eigen-adjoint is added here. |
| Ordinary TFSF | `fdtd/excitation/tfsf_state.py:817` admits plane waves and Gaussian beams. | It does not solve a mode profile. ModeSource's surface TFSF patches are covered by the first row. |
| Other rebuild-flag readers | `modes/common.py:545` selects component permittivity; `modes/__init__.py:710` selects rebuild/PEC handling and `:852` excludes the forward scalar mu path; `modes/conductors.py:42` refuses good-conductor replay. | These are helpers of the ModeSource solve above, not additional public replay entrypoints. |

## Validation and merge impact

`tests/gradients/test_mode_source_replay_guard.py` supplies host-constructible
guards and two CUDA-skipped reproducer cases. Each CUDA case executes one
gradient attempt and four no-grad forwards, with h = 0.05 and 0.025. The control
uses exactly `abs(D(h)-D(h/2))/3` plus
`2 * eps_float32 * sqrt(2000) * max(abs(J)) / (h/2)` as its error bound. The source
and control gradients must lie within that bound. The source case uses
`vector_dense_torch`; the fixed-plane control retains `vector_dense`.

No existing test is intended to start raising. A host material-graph census of
`_DensityModeSourceScene`, `_TwoPortModeScene` and `_TwoPortBarrierScene` at their
original grid spacings admits all launch planes; their design regions lie away
from the planes. Full CUDA execution remains reviewer validation. The merge
declaration lists `tests/gradients/test_mode_source_replay_guard.py`, which also
holds the source-metadata and resume-fingerprint preservation pin.

CPU validation with `CUDA_VISIBLE_DEVICES=` and the `maxwell` environment:

- `python -m pytest tests/gradients/test_mode_source_replay_guard.py -q --tb=short`:
  18 passed, 2 skipped, rc 0.
- `python -m pytest tests/gradients/ tests/sources/mode/ tests/validation/benchmark/test_integration_batch.py -q --tb=short`:
  165 passed, 374 skipped, 2 warnings, rc 0.

The two new rectangular-guide cases need CUDA. The skipped native adjoint and
source-physics cases in the broader selection also remain GPU validation work.

## Sampling-edge coverage and cached decisions

The sampling-edge and cache fixtures now pin the synthetic
`vector_pec_dense_torch` / scalar mismatch, which remains refused. They do not
claim refusal of the admitted conductor-free torch vector family.

The coordinator checks use a CPU scene with 0.1 m cells and a source at x=0.
Each node-grid epsilon and mu plane entry, and each replay Ey-plane entry, is
differentiated separately with autograd. This ground truth does not use the
guard's weighted reduction. Separate epsilon-only and mu-only designs exercise
both constitutive dependencies, including zero derivatives through a constant
epsilon bound on a mu-only region.

| Case | Physical support / sampling footprint | Node eps derivative | Node mu derivative | Replay Ey eps derivative | Guard |
| --- | --- | --- | --- | --- | --- |
| One-cell overlap, epsilon design | Box x=[-0.05,0.05], one node layer across x | nonzero | zero | nonzero | raises |
| Neighboring density layer, epsilon design | Box x=[-0.2,0.2]; only the texture layer centered at x=+0.1 is trainable, one cell away; x=0 trilinear sampling reaches it | nonzero | zero | nonzero | raises |
| Padded box, epsilon design | Box x=[0.025,0.125] misses x=0; smoothed occupancy and border-padded texture sampling reach the plane | nonzero | zero | nonzero | raises |
| Distant box, epsilon design | Box x=[0.25,0.35] | zero | zero | zero | admits |
| One-cell overlap, mu design | Same one-cell box, fixed eps=2 | zero | nonzero | zero | raises |
| Neighboring density layer, mu design | Same neighboring trainable texture layer, fixed eps=2 | zero | nonzero | zero | raises |
| Padded box, mu design | Same missed bounding box, fixed eps=2 | zero | nonzero | zero | raises |
| Distant box, mu design | Same distant box, fixed eps=2 | zero | zero | zero | admits |

No false negatives were found in these cases; the spatial probe is unchanged.
The one-cell-away density case refers to the trainable texture layer, not its
enclosing Box: its dependence reaches the plane only through trilinear sampling.

`_mode_source_replay_error` factors the complete CPU-testable decision from the
exception and native backward wiring. The bridge caches both admitted and
rejected verdicts, keyed by the accepted solver and scene identity plus the input
tensor identities and `_version` values. `accept_forward` resets the verdict for
every new forward, including a SceneModule that changes its construction without
changing parameter values. It does not cache across optimization forwards.
Repeated backward on the same forward graph does not rebuild the material scene,
compile materials, or run plane VJPs again. Parameter edits and replacement scene
graphs invalidate the key. As with the forward fields themselves, structural
SceneModule edits require a new forward to define the corresponding design graph.

The CPU cache test counts exactly one probe and one material compilation across
two unchanged calls; a parameter-version change increases both counts to two,
and a replacement graph making the plane trainable increases both to three and
raises. Repeating the rejected call leaves the counts at three. A CUDA wiring
test performs two actual backwards and counts only the probe's compilations,
separately from the required final material pullback compilation.

`scripts/dev/time_mode_source_replay_probe.py` times the full uncached guard
(scene materialization, preparation, material compilation and VJPs), its first
cached call, and cache hits on the unmodified bend example with the launch plane
outside the design. Solver initialization, the forward mode solve and warmup are
outside the measurements. Times use synchronized wall-clock intervals so host
work and device synchronizations are included. Run on an assigned GPU with:

```bash
python -m scripts.dev.time_mode_source_replay_probe --repeats 5 --backward
```

The optional full backward measurement uses the same example and device. No GPU
timings were collected in this CPU-only worktree.

## Replay source mutation audit and reader census

Audit scope: `initialize_source_terms` and its eight admitted source kinds in
`core._SOURCE_REPLAY_REBUILD_KINDS`, including mode solvers, surface/beam profiles,
custom dataset interpolation, term builders, and gather preparation. Searches
covered subscript assignments, `.update`, `.setdefault`, list mutation and
in-place tensor methods, followed by reads of the reachable constructors.

The only writes into a compiled source are seven top-level assignments in
`fdtd/excitation/injection.py:1348` and `:1369`: `effective_index`, `beta`,
`effective_index_complex`, `beta_complex`, `mode_solver_kind`,
`normal_stagger_power_factor`, and `prepared_mode_power`. No nested source-time,
profile, dataset, tensor or list is written by replay construction or
`initialize_source_terms`. Profile dictionaries and tensors produced by the mode
solve are fresh results. Term lists and gather dictionaries belong to the replay
solver. `temporal.py:69` checks a sampled waveform table without modifying it;
table device-cache population in time evaluation is outside initialization.
The forward already populates the same device cache before backward.
Consequently the shallow source-dictionary copy is sufficient; no deeper copy
or extra GPU transfer was added.

CPU tests construct a replay solver and execute its real initialization for all
eight rebuild kinds. Snapshots recursively compare every nested value, including
custom field/current dataset arrays, sampled waveform arrays and device caches;
tensor equality uses `torch.equal`. WaveformTable snapshots explicitly traverse
its slots because its own `deepcopy` intentionally shares the object. CUDA tests
also compare all compiled-source contents after a complete backward (or the
guarded refusal).

The reader census below is relative to master `d3b11a6b`, which let replay
overwrite the forward dictionary. Paths are under `witwin/maxwell/` unless
otherwise stated. Readers of independent `mode_data` dictionaries are included
to distinguish their outputs from the compiled-source metadata.

| Reader | Fields read and output difference versus master |
| --- | --- |
| `fdtd/adjoint/bridge.py:365`, `:403` | Direct production reads of compiled-source `mode_solver_kind`: family comparison and refusal message. These are new guard behavior; they retain the forward kind across repeated backwards. |
| `fdtd/adjoint/core.py:493`, `:529` | Copies the complete source records into the temporary solver. Replay's own seven metadata values can change, while the forward records remain intact. |
| `fdtd/resume.py:353`, recursive hash at `:278`, signature at `:248` | Reads every compiled-source field. A fingerprint after backward previously incorporated scalar replay metadata; now it remains the forward fingerprint. The CPU regression pins the entire fingerprint before/after replay initialization and proves the replay kind and beta actually differ. |
| `fdtd/distributed/resume.py:45` | Shard signatures delegate to the same single-device fingerprint; logical-source hashing at `:36` uses public sources, not these compiled metadata keys. Logical output is unchanged; any shard exposed to such a rebuild inherits preservation of its forward identity. |
| `result.py:1473` | Keeps the live solver accessible. Inspecting `result.solver._compiled_sources` after backward now shows forward `vector_dense`/`vector_sparse`, effective index, beta and power factor, instead of scalar replay values. This is an observable correction. |
| `result.py:1484`, `simulation.py:2024`, `fdtd/adjoint/bridge.py:1778` | Result metadata is copied from simulation metadata, with separate port/material diagnostics. None reads the seven compiled-source keys, so default metadata has no numerical/output change. |
| `simulation.py:2050`, `result.py:2132`, `result.py:2756` | Solver statistics and `stats()` do not read these launch keys. No statistics or default log-output change. Repository searches found no production logging reader of the seven compiled-source keys. |
| `result.py:2770`, `:2804` | Result export serializes fields, monitors, ports, metadata and statistics, without serializing `_compiled_sources`. No default Result snapshot/export change. Resume export is separately affected as above. |
| `postprocess/modal.py:171`, `:319` | Computes an independent reference mode and exports that solve's effective index/beta. It does not read the overwritten compiled-source entries; modal monitor results are unchanged. |
| `modal_sweep.py:517`, `waveport_sweep.py:151`, `:635`, `terminal_waveport.py:229`, `:400`, `fdtd/boundary/abc.py:124` | Read beta/complex-beta from independent local mode results, not the forward compiled-source metadata. Their outputs do not change from dictionary isolation. |
| `benchmark/runner.py:1989`, `:1992`, `:2020`, `:2059` | Benchmark effective-index metrics consume monitor outputs from the independent postprocess solve. No metric change from the dictionary copy. |
| `benchmark/core_path/gates.py:84`, recursive collector at `:58`, setup capture at `:148` | Traverses `_compiled_sources`, including scalar numbers converted to tensors. Normal benchmark capture happens after prepare and before any backward, so its output is unchanged. Calling this collector after backward now captures the forward effective index/beta/power factor rather than scalar replay values. The CPU isolation test pins this numerical inventory before/after replay. Solver-kind strings and None are omitted. |
| `tests/sources/mode/test_fdtd_mode_source.py:38`, `:39`; `test_fdtd_broadband_mode_source.py:127`, `:128`; `test_fdtd_complex_lossy_slab_mode.py:136`, `:137`; `test_fdtd_anisotropic_aperture.py:203`, `:204` | Existing direct compiled launch-value readers inspect forward initialization, before any backward; their expectations remain unchanged. The new guard tests additionally pin their persistence after replay/backward. |

For the real guide used by the CPU isolation test, forward/replay values differ:
effective index 3.1524356/3.1698855, beta 66.0701646/66.4358877, and normal stagger
factor 0.2468161/0.2397214. Those forward values now survive replay. The two
complex-valued entries are None in that real case; in a complex forward/real
replay they preserve the complex forward values rather than being overwritten
with None. `prepared_mode_power` remains 1.0 in both paths and has no value
change. No existing test is expected to start raising from this follow-up.

Coordinator validation, with `CUDA_VISIBLE_DEVICES=` and the maxwell interpreter:

- Targeted guard selection: 35 passed, 3 skipped, rc 0.
- Earlier CPU selection (`tests/gradients/ tests/sources/mode/ tests/validation/benchmark/test_integration_batch.py`): 182 passed, 375 skipped, 2 warnings, rc 0.
- Timing script `--help`: rc 0. Actual timing and the three native backward tests require reviewer GPU execution.

## Differentiable dense vector replay

`fix/mode-source-replay-vector-grad` replaces the scalar replay for trainable,
lossless vector launch planes without an interior conductor. Forward and replay
both use `vector_dense_torch`. Spatial dependence is probed against trainable
scene material tensors, including MaterialRegion density, rather than inferred
from a whole-grid `requires_grad`. Native forward retains the module scene's
material ancestry; only its material compilation enables gradients during the
native no-grad forward.
The torch operator is assembled in FP64 on the scene device; the single
`torch.linalg.eig` and the selected-pair dense transpose solve stay there.
Only detached eigenpairs/material diagnostics reach the existing numpy selectors.
Their filters and candidate choice are not differentiated.

Torch is the autograd definition here, which the CUDA-first rule accepts.
The nodal operator retains all four (Hu, Hv, Eu, Ev) blocks exactly, including
diagonal epsilon and mu. The uniform nonmagnetic path retains the forward Yee
operator, Gauss/curl H reconstruction and node interpolation. Preferred-component
phase fixing at the sticky reference index described below is differentiable,
followed by the existing unit-power
normalization. Scalar forward and its `-cross(direction, polarization)` H sign
are unchanged: scalar E cross H has signed power `-direction_sign`, whereas
vector H comes from its own eigenvector/curl reconstruction. This change does
not use that scalar H construction to differentiate a vector launch.

For selected eigenvalue lambda, let g = min(j != selected) |lambda-lambda_j|
and S = ||A||_infinity. A backward-stable FP64 eigenproblem has an absolute
roundoff scale O(eps64 S). Eigenvector differentiation amplifies this as 1/g.
Requiring that scale estimate below sqrt(eps64) gives
`delta = sqrt(eps64) S / |lambda|` for the relative gap `g / |lambda|`.
The refusal prints the source name, measured gap and delta. This is a
scale-aware numerical isolation criterion, not a general nonnormal condition
number guarantee. Both S and lambda scale identically when physical lengths
change. Square-guide Ey/Ez degeneracy refuses before backward.

The stock full-spectrum eig backward produces NaNs when *unselected*
eigenvalues coincide, even if the selected mode is isolated. The selected real
column therefore uses the equivalent dense bordered derivative:

```
[A - lambda I, -v] [dv      ] = [-dA v]
[v.T,           0] [dlambda ]   [0    ]
```

`v.T v = 1`; the transpose system applies its VJP once. It has no divisions
between unused eigenvalues and does not perturb or split any eigenvalues.
The explicit phase/power operations remain ordinary torch autograd. This dense
selected-column implementation is required for rectangular uniform guides with
unused degenerate modes. The post-release replacement, route (ii), is an
analytic **sparse** vector eigenpair VJP, avoiding dense assembly/factorization.

The cap is on actual operator rows, `n <= 4096`, before assembly. The brief's
`n = 2 N_interior` does not describe the existing first-order nodal operator:
it is `4 N_interior`; Yee uses `(Nu+1) Nv + Nu (Nv+1)` staggered electric
unknowns, where Nu/Nv count interior nodes. One maximum-size FP64 matrix is
128 MiB; complex128 eigenvectors are 256 MiB. Assembly/autograd intermediates,
the bordered solve and eig workspace require additional multiples of that
storage (order GiB, backend dependent). There is no above-cap scalar substitute.

The propagation constant also remains a tensor through axial half-cell power
correction and time delay. CW phase is already folded into differentiable
cosine/sine patches. Pulse replay accumulates the delay cotangent using the
waveform's torch forward-mode derivative at each injection time, then applies
the delay graph together with the profile graph in the single source pullback.
There is no per-step reverse autograd call.

Replay source cotangents pull the node-plane material graph directly to the
custom-Function material-input frontier. Component-epsilon leaves separately
retain injection/update-coefficient derivatives, so no node-to-edge inverse or
CPU material gradient is introduced. Fixed launch planes keep their existing
scipy forward and scalar replay path, including the downstream control.

CPU tests in `test_mode_source_vector_replay.py` cover both operator entries,
selection and profile agreement, phase invariance, selected-pair finite
differences, unselected degeneracy, square-guide refusal, cap refusal,
forward/replay `torch.equal` and the actual source-profile pullback.
CUDA-only coverage additionally checks the rectangular source/control FD bound,
control gradient bit identity against the base dispatch, metre/micrometre
scaling, a coarsened example bend with its MaterialRegion over the source plane,
CUDA forward/replay equality for ModeSource/ModePort and the cap. These CUDA
cases are authored but not run here; this worktree has no GPU allocation.

Before the existing power normalization, the torch family scales its arbitrary
peak-normalized eigenvector amplitude by inverse square-root aperture area.
This cancels analytically in unit-power normalization, while preventing the
absolute zero-power check from confusing a micrometre aperture with zero power.
The existing scipy/scalar normalization receives no change. Host scale coverage
compares both normalized fields and material gradients. Scale comparisons use
two-sample arithmetic coverage at both lengths: single-sample occupancy in the
shared geometry dependency floors its smoothing length at FP32 epsilon in
metres, so scaling a single-sample scene does not preserve its discrete material
plane. The original metre FD/control reproducer retains samples=1.

Validation for this branch used `python`
with `CUDA_VISIBLE_DEVICES=`, `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`:

- `-m pytest tests/gradients/ tests/sources/mode/ tests/validation/benchmark/test_integration_batch.py -q --tb=short`:
  180 passed, 381 skipped, 20 warnings, pytest rc 0 (325.85 s).
- Final focused run, including the additional FP64 case added after broad
  collection: `-m pytest tests/gradients/test_mode_source_vector_replay.py tests/gradients/test_mode_source_replay_guard.py tests/validation/benchmark/test_integration_batch.py -q --tb=short`:
  60 passed, 9 skipped, 18 warnings, pytest rc 0 (12.35 s).
- Standalone integration-manifest checks: `-m pytest tests/validation/benchmark/test_integration_batch.py -q --tb=short`:
  26 passed, pytest rc 0.

The branch-keyed merge declaration lists 11 node-level trainable-plane movers,
including synthetic guard-label fixtures. The original downstream design
scenes remain fixed-plane controls. The nine task-specific CUDA nodes (source
and control FD, two launch-profile equality cases, direct and scene-level cap
refusal, control bit identity, length scaling and the bend) remain unrun.

## Coverage review: lossless checks, material reuse and persistent gauge

The trainable vector dispatcher and both torch operator builders check relative
epsilon/mu before taking their real parts. An imaginary magnitude above `1e-7`
raises `NotImplementedError` with the source name and constitutive reason,
matching the forward router's dimensionless loss test. Frequency evaluation
already includes electric conductivity as `-i sigma_e / (omega eps0)`, so the
same check refuses conductive trainable apertures. The selected-column
eigenpair operation also checks `|Im lambda| > sqrt(eps64) |lambda|` before
projecting its eigenvalue and conditioned eigenvector to real values. Its
global-vector phase rotation is internal conditioning only;
`_gauge_vector_profiles` is the sole launch gauge contract.

For scenes with material inputs and mode sources, native forward material
initialization enables gradients around its own single material compilation.
The spatial trainability probe reuses that compiled graph, including its
frequency evaluation and source-plane slicing. It no longer compiles the scene
again. No design-graph cache or cache key was introduced: every new forward
compiles its own current scene, so parameter identities/versions, geometry,
region bounds, beta/subpixel settings, source-plane and grid edits receive the
forward's current materials. Native no-grad launch slices remain detached;
retaining the compiler graph does not change fixed-aperture PEC admission.
The first-backward family validator retains its per-forward verdict cache.
Matching torch-vector launches reuse the retained forward slices; only
mismatched solver families build a separate graph for the refusal probe.

The preferred-component gauge index is sticky. It remains the phase reference
while its magnitude is at least half the current peak; below that threshold
the new argmax becomes the reference. Amplitude normalization still uses the
current peak magnitude. Each solver records an immutable
`(preferred component, profile shape, index)` tuple in per-source runtime state.
The SceneModule (or the declarative Scene) retains these records by source name across solver
and Simulation rebuilds, including new scenes materialized by a SceneModule.
A changed preferred component or profile shape resets the index. Replay copies
the forward runtime record separately from the source dictionary,
so later solver rebuilds cannot alter a recorded forward gauge. Creating a new
SceneModule or declarative Scene starts new gauge state. The threshold crossing
can still change phase; near-equal peaks exchanging order no longer do.

Gauge-on-resume behavior follows that ownership: checkpoints do not serialize
or restore the sticky index. Preparation with the same in-memory Scene or
SceneModule uses its retained gauge; a fresh owner selects a new reference from
the current profile. Routing/gauge state is excluded from the configuration
fingerprint. Cross-process gauge continuity is therefore not a checkpoint
contract; the fixed-source fingerprint regression pins identity preservation.

CPU coverage checks all six lossy epsilon/mu components before projection,
the Yee builder, the router's tolerance, the selected eigenvalue criterion,
conductive apertures with trainable epsilon or sigma, opposite-sign competing
peaks and the half-peak boundary, and gauge retention across solver and
Simulation rebuilds for both scene forms. The production material-initialization
count test performs two unchanged no-grad forwards and observes exactly two
material compilations:
one per forward and zero additional compilations for the launch-plane probe.

Coverage-review validation used `python`
with `CUDA_VISIBLE_DEVICES=`, `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`:

- `-m pytest tests/gradients/ tests/sources/mode/ tests/validation/benchmark/test_integration_batch.py -q --tb=short`:
  221 passed, 382 skipped, 20 warnings, pytest rc 0 (379.91 s).
- Final focused validation, including SceneModule ownership, fixed-PEC
  admission and FP32/FP64 threshold edges:
  `-m pytest tests/gradients/test_mode_source_vector_replay.py tests/gradients/test_mode_source_replay_guard.py tests/validation/benchmark/test_integration_batch.py -q --tb=short`:
  104 passed, 10 skipped, 18 warnings, pytest rc 0 (13.77 s).

The final focused selection includes four regression cases added after the
broad selection was collected. CUDA-only cases remain unrun. There are no
additional existing-test movers beyond the branch's recorded declaration.


### Preparation replay and resume identity repair

The GPU preparation failures at `99ae5776` came from initialization order, not
from missing trainability detection. Native-adjoint preparation calls
`_resolved_source_term_lists` and `initialize_source_terms` before the backward
family validator. Only that later validator populated `_source_vector_materials`,
so preparation replay indexed an empty map for the trainable launch.

`solve_mode_source_profile` now takes graph-bearing plane slices from the
forward's compiled materials, probes those same slices, and retains them on the
solver when the launch is trainable, before detaching the native forward launch.
`_build_source_replay_solver` carries those tensor references and copies the
per-source routing/gauge record. Replay reads the retained slices directly;
there is no material recompilation or missing-entry recomputation. The backward
verdict cache retains the forward slice map and restores it after backward
cleanup. Each rebuilt forward owns a new map and its own material graph.

The discrete resume-fingerprint mover was the `_torch_vector_plane=False` key
written into every fixed mode source. Routing flags and sticky gauges now live
in `solver._mode_source_vector_state`, outside the hashed compiled-source
records. The fingerprint algorithm is unchanged. A CPU regression compares a
fixed ModeSource scene's fingerprint against its records with both introduced
runtime keys removed, matching the `d3b11a6b` source-record behavior, and confirms
that editing solver runtime state leaves the fingerprint unchanged.

CPU regressions call the production replay builder and source initializer before
any backward guard for both ModeSource and ModePort, at metre and micrometre
scales, across two scene rebuilds. The metre cases also include an independent
fixed source. They require identity of retained material tensors, bitwise equal
trainable electric launch patches, copied gauge records, and finite nonzero
material gradients. Only CPU admission is mocked for these initialization tests;
no differentiable time stepping runs on CPU.

Repair validation used `python` with
`CUDA_VISIBLE_DEVICES=`, `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`:

- `-m pytest tests/gradients/ tests/sources/mode/ tests/validation/benchmark/test_integration_batch.py -q --tb=short`:
  230 passed, 382 skipped, 20 warnings, pytest rc 0 (330.11 s).
- `-m pytest tests/gradients/test_mode_source_vector_replay.py tests/gradients/test_mode_source_replay_guard.py tests/validation/benchmark/test_integration_batch.py -q --tb=short`:
  109 passed, 10 skipped, 18 warnings, pytest rc 0 (18.94 s).

CUDA execution and the GPU baseline fingerprint suite remain for the reviewer.
No fingerprint rebaseline or additional mover declaration is required: fixed
launches restore their original compiled-source records.

### No-grad replay coefficient graph repair

The GPU failures at `28b77abb` affect the metre source case as well as the
length-scaled test. CPU isolation does not implicate the vector eigen-adjoint:
weighted E/H profile and beta VJPs on the actual guide plane agree with central
finite differences at both lengths. The nodal regression also checks an
arbitrary spatial profile cotangent at metre and micrometre lengths against
finite differences of the torch dense solve. Forward and replay both select
`vector_dense_torch`; the CUDA length test now labels each scale and checks the
solver kind on finite-difference forwards too.

The lost derivative is the electric injection coefficient. Custom
`Function.backward` and checkpoint reconstruction enter with gradients disabled.
`_resolved_source_term_lists` built its temporary solver in that context, so
`_dynamic_electric_curl(base_curl, base_eps, eps)` evaluated
`base_curl * base_eps / eps` without recording the edge-epsilon graph. Enabling
gradients only around `initialize_source_terms` retained the mode-shape graph
but could not recover this earlier `1/eps` dependence. The once-per-pass source
pullback therefore returned `None` for all three edge-epsilon leaves. Existing
CPU source-graph tests built the temporary solver with gradient recording
already enabled and missed the production context. The downstream control has
no trainable injection-plane coefficient, consistent with its passing GPU gate.

The fix also enables gradient recording around temporary-solver construction
when source patches require their material graph. It reuses that same predicate
for profile initialization. Forward values, eigenpair mathematics, gauge,
normalization, leaf identities, CUDA execution and fixed-source dispatch remain
unchanged. No material recompile, alternative eigensolve or CPU runtime is added.

The new CPU regression initializes the real guide through the existing reference
backend, without time stepping, then resolves and pulls back source patches
under `no_grad`. It checks that the edge coefficient gradient is present,
combines it with the mode-profile gradient through the production material
pullback, and compares their sum with central differences of independently
initialized forward patches. Coverage crosses both lengths, ModeSource/ModePort,
and one-/two-sample arithmetic material compilation. The one-sample micrometre
scene deliberately retains the shared geometry smoothing floor; its own FD and
adjoint still agree. The FD tolerance is Richardson truncation plus FP32
contraction roundoff, and forward/replay patch values use the setup-tensor
`64 eps32` bound (the existing dynamic coefficient expression can round by one
ulp). Profile equality retains its existing exact checks.

For the two-sample `.04 * scale` CPU guide and the electric-patch objective
`scale**2 * sum(patch**2)`, the detached-coefficient pullback gives 587.7374 at
metre scale and 587.7397 at micrometre scale. The restored coefficient terms
are -189.4152 and -189.4154, giving 398.3222 and 398.3243; central FD with
`h=.025` is 398.3447 at both lengths. These are initialization-only source-patch
measurements, not the full FDTD GPU objective. The original guide-plane probe
at `.02 * scale` also checks the weighted E/H profiles and beta independently.

Pre-fix regression check (temporarily restoring the old production scope):
`-m pytest tests/gradients/test_mode_source_vector_replay.py -k no_grad_replay_retains -q --tb=short`:
8 failed, 56 deselected, pytest rc 1; every failure is the absent edge-coefficient
gradient. The fixed focused selection
`-m pytest tests/gradients/test_mode_source_vector_replay.py tests/gradients/test_mode_source_replay_guard.py tests/validation/benchmark/test_integration_batch.py -q --tb=short`
passes: 118 passed, 10 skipped, 18 warnings, pytest rc 0 (36.53 s).

The required full CPU selection
`-m pytest tests/gradients/ tests/sources/mode/ tests/validation/benchmark/test_integration_batch.py -q --tb=short`
passes: 239 passed, 382 skipped, 20 warnings, pytest rc 0 (397.73 s).
Both selections use `python`,
`CUDA_VISIBLE_DEVICES=`, `OMP_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1`.
CUDA end-to-end source/control FD bounds, length scaling, control bit identity,
bend gradients, CUDA launch-profile equality and cap refusal remain for reviewer
GPU execution. No tolerance in those physics gates was changed.

## Public fixed-term admission pins

The fixed-term path cannot admit a trainable launch plane through a public
WavePort run: `waveport_sweep.py:460-465` rejects independent ModeSources and
excited ModePorts before building columns, and `:372-405` rejects trainable
WavePort cross-sections, including adjacent launch planes. The public-path
tests `test_waveport_sweep_rejects_independent_mode_launches` and
`test_waveport_sweep_rejects_trainable_cross_section_through_simulation` pin
these refusals. `test_fixed_waveport_column_records_keep_forward_terms` builds
real fixed-column records with a downstream design on CPU;
`test_public_fixed_waveport_column_admits_backward` pins native backward
admission on CUDA. No additional fixed-term refusal is needed, and these
admission pins add no runtime or feature-inventory change.

Final CPU gates (`CUDA_VISIBLE_DEVICES=`, maxwell environment): manifest parsing
through `scripts.dev.integration_batch.load_manifest` passed; the integration
batch, mode-source replay guard, and wave-port adjoint test files together
reported 123 passed, 6 skipped, rc 0. The native column-backward regression is
CUDA-marked and was not executed in this CPU-only validation.
