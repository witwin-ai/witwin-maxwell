# FDTD single-path CUDA execution

## The existing runner is deleted (W7 row 16e, 2026-11-13)

Every FDTD solver runs the fused core Yee update; the per-component runner is gone.
- Package: `stepping.py` keeps `update_magnetic_fields(solver, *, imag=False)` and `update_electric_fields(solver)`
  as the core launches (an explicit error when a solver has no core runtime) and drops the standard/CPML/compressed/
  modulated per-component dispatch; `fdtd/solver.py` drops the `_update_*_fields[_cpml]` wrappers and the runtime
  package their re-exports. `DistributedFDTD` loses `core_path` and the bounded per-component launchers (the core
  launch has no interior/halo split, so overlap stays inactive as before). The gradient bridge's per-component
  forward loop is deleted (the forward is the core solve, row 14). `Simulation._initialize_fdtd_solver_for_scene`
  takes `batch_lane` instead of `core_enabled`; `solver._fdtd_execution_path` is removed.
- CUDA: 37 operators (`update_magnetic_h*_{standard,standard_bounded,cpml,cpml_compressed}`,
  `update_electric_e*_{standard,standard_bounded,modulated,cpml_modulated,cpml,cpml_compressed,
  cpml_modulated_compressed}`, the four batched lane updates), their kernels and launchers, the 33 backend wrappers,
  `kernels/magnetic.cu` (whole file) and about two thirds of `kernels/electric.cu`. Kept: the modulation clock, the
  Bloch and Bloch-standard-z kernels and the six CPML correction kernels. `stable_abi_v16 -> v17`.
- Kept by decision 3 and because the core per-step loop uses it: `_field_update_block`, the field-block graph,
  the tail/port/embedded runners.
- Tests: the A/B references that ran the existing runner now load addendum `w7-row16e` (19 payloads captured on
  master a462f31ba through the row-16d driver); the per-component parity tests run the core launches against the
  Torch Yee oracle (`tests/oracles/yee_cpu_reference.py`, new step driver); tests that only exercised the
  per-component kernels or the existing runner's own graphs are deleted.

## Test-side existing-runner driver (W7 row 16d, 2026-11-13)

`core_path_decision` is deleted: `Simulation._build_fdtd_solver_for_scene` prepares every solver with
`core_enabled=True` (batch lanes, as before, bind the core launches themselves). The library has no path
switch left; `_initialize_fdtd_solver_for_scene(core_enabled=False)` is reached only by the test oracle
driver `tests/support/existing_runner.py::with_existing_runner(simulation)`, which replaces that one
simulation's solver builder, so `run`, `prepare`, `run_until`, resume, port sweeps and the gradient bridge
all prepare the existing runner. Its users: the existing runner's own graph tests (`test_cuda_graph.py`,
`test_cuda_graph_capture_coverage.py`, `test_thin_sheet_feeds.py`, `test_port_step_graph.py`, the resumed
embedded-network test in `test_default_auto.py`) and the core == existing A/B tests (`test_bloch_admission.py`
incl. its refusal matrix on both runners, `test_monitor_admission.py`, `tests/esd/test_esd_core_path.py`).
Row 16e deletes the runner, the driver and the `core_enabled` parameter together.

## Failure policy: no fallback route (W7 row 16c, 2026-11-13)

User decisions 2026-11-12 (coordinator session): (1) a setup failure on the core path retries on core
with torch setup; a capture failure or a core-load failure raises; (2) `FDTDCoreFallbackWarning` and the
`execution_path`/`fallback_reason` stats/metadata keys are deleted in 0.7.0 without a shim; (3) the
field-block graph stays. Every route of the row-16 audit (E1-E6) is closed:

| Route | Closed by |
|---|---|
| E1 Bloch + separate physics | core admission (row 16b) |
| E2 unadmitted monitor types | core admission (row 16a) |
| E3 `"non-CUDA execution"` | blocker deleted; a CPU scene raises `ValueError` from the CUDA requirement before any path is built |
| E4 `"unsupported source type"` | blocker deleted. Not dead: `ESDCurrentSource` (in `scene.sources`, lowered by `resolved_sources()` to a `UniformCurrentSource`) kept the existing path and now runs on core; unknown objects are refused by `compile_fdtd_sources`/`compile_fdtd_observers` on every path |
| E5 decision raise | `core_path_decision` returns `("core", None)` and cannot raise |
| E5 activation / core load | raises (`activate_core_path`, `CorePathRuntime`, shard `_activate_core_path`) |
| E5 accelerated setup | `_AcceleratedSetupFailure` -> one retry on core with torch setup and a `MaxwellWarning`; the retry's failure raises |
| E5 solve / capture | `run_chunks` restores the state and re-raises; `_solve_fdtd_with_fallback` is deleted (`_execute_fdtd_solve`) |
| E5 gradient forward | the bridge's existing-path rebuild is deleted; the error raises |
| E5 distributed shards | the `core_path=False` rebuild is deleted; the activation error raises |
| E6 test forcing | `tests/support/existing_runner.py` replaces `core_path_decision` for marked scenes (row 16d replaces it by a tests-side driver and deletes the decision); `torch_setup_oracle.py` no longer catches a fallback |

A failed preparation restores the public preparation outputs (mode gauges, optimization dt) before
raising. `hatch_build.py` refuses a wheel with only some of the three prebuilt libraries. The existing
runner code stays in place (row 16e deletes it); no public route reaches it.

## Complex fields beside separate physics on the core path (W7 row 16b, 2026-11-13)

Inventory E1 (row-16 audit): `core_path_blockers` refused `"Bloch/complex fields with separate physics"`. No kernel
was missing: the separate-physics closure already calls `_field_update_block`, whose complex branch launches the
core imaginary H and the Bloch E kernels (row 13), and `_port_step_block` applies each lumped branch to both field
banks. The blocker is deleted. The whole-step graph saves and restores, around its warm-up step and captures, the
imaginary lumped bank (`LumpedRuntime.imaginary`, `*_imag` state; its scratch is shared with the real bank) and the
imaginary port DFT accumulator (`accumulator_imag`, also in the restored sample counts).

Admitted (every separate-physics piece whose runtime accepts complex fields): passive R/L/C lumped elements, undriven
`LumpedPort`s with R/L/C, `SeriesRLC` or `ParallelRLC` terminations or the reference load (the imaginary
dispersive-history bank when the voltage path crosses a dispersive medium), and `CopperFoil` (real and imaginary foil
entries), in the whole-step graph; beside fixed-k TFSF in the per-step core loop (step-loop blocker "TFSF with
separate physics", as for real fields). Still refused, on every path, by the pieces themselves (unchanged; row 16c
turns no admission refusal into a new error here): driven lumped ports (`ports.py`, "deferred to v2"), coaxial and
terminal ports and coaxial elements (passive LTI R/L/C and LumpedPort only), lumped terminals on copper foil, circuits
and embedded networks, conformal thin sheets (periodic/Bloch boundaries), thin wires (Bloch phase topology), PEC cut
surfaces (real isotropic linear fields only) and SIBC (real surface update). `test_bloch_admission.py` pins this
matrix on both paths.

Gate scenes `BLOCH_SEPARATE_SCENES`: R, L and C elements in the Bloch x/y + z PML dipole cell; three undriven ports
(series RLC, parallel RLC, reference load, one in a Debye block); a Bloch-periodic copper foil on a conductive Debye
substrate under a CW drive with a PML x axis and a power-loss volume; an all-Bloch foil under a pulse beside a
resistor; a fixed-k TFSF slab with a resistor. `gates.py` also pins the complete physical state
(`graph_state_tensors`: both lumped banks, copper foil state) for these scenes. Baselines are the pinned addendum
`addenda/w7-row16b` (base dbac8adec, existing path forced, torch setup). Evidence: `<evidence-root>/w7-20261108/row16b/`.

## Every monitor type on the core path (W7 row 16a, 2026-11-12)

Inventory E2 (row-16 audit): `core_path_blockers` refused six public monitor types with `"unsupported monitor
type"`. None of them needs a step-time piece the core path lacks, so the row is qualification only (no CUDA, no
ABI, no stepping/ports change); the blocker now lists every public monitor type (`capability.keys.MONITOR_TYPES`).

- `PowerLossMonitor`: reduced at export (`Result.monitor`) from the full-field DFT, the static conductivity and,
  beside copper foil, the foil side spectra and bulk traces the core run already accumulates; the chart variant
  reads the chart-scaled tables. The power-loss monitor no longer keeps chart scenes on the existing path.
- `ComponentStressMonitor`: reduced at export from the field/flux time series it names (those keep the per-step
  observer tail).
- `DipoleEmissionMonitor`: compiles to point observers on the dipole cell (`monitor_type="dipole_emission"`),
  which join the grouped core point observers; the emitted power is formed at finalization from the source
  spectrum.
- `SurfaceFieldTimeMonitor`, `AuxFieldTimeMonitor`: host time observers like `FieldTimeMonitor`; a scene with one
  runs the per-step core loop (`time_observers_enabled`), and the auxiliary traces read the same ADE state tensors.
- `WireMonitor`: host-driven wire monitors are the existing step-loop blocker; the per-step core loop calls
  `accumulate_wire_monitors` after the step as before.

Gate scenes `MONITOR_SCENES` (benchmark/core_path/scenes.py): a conductive Debye slab with two power-loss volumes
(native cells, both automatic channels; physical intersection, one frequency), a copper foil on a one-sided
conductive Debye substrate fed by a port (split bulk loss, full and one-frequency subsets), the concave chart board
with a conduction-loss volume, a named oblique dipole in a dielectric sphere (two emission monitors), a driven port
gap with two field time series bound to a component-stress monitor, surface (box and start/stop patch) and
auxiliary-state (patch and point, electric and absent magnetic state) time traces over a Debye slab, and a thin
wire with a wire monitor. `gates.py` pins the export-time reductions (`export_tensors`: power-loss data, component
stress, typed dipole emission) besides the stored payloads, and treats the new scenes as physics scenes. Baselines
are the pinned addendum `addenda/w7-row16a` (base dbac8adec, existing path forced, torch setup). Evidence:
`<evidence-root>/w7-20261108/row16a/`.

## Multi-GPU shards on the core path (W7 row 15, distributed half, 2026-11-12)

Inventory R12/B15: every `DistributedFDTD` shard runs the fused core Yee update. `core_path_request_blockers`
no longer emits `"multi-GPU/distributed execution"` (a parallel run has no single-device admission
blockers: `DistributedFDTD` fences every feature it does not support), and `core_path_blockers` no longer
checks for shards. `DistributedFDTD(core_path=True)` binds one `CorePathRuntime` to each shard solver's local
storage at the end of `init_field`, after the owner-resident circuit/network/wire/copper runtimes, so the
compressed coefficient tables see the final shard coefficients. The coordinator's serialized step launches
`runtime.magnetic()` and `runtime.electric()` in place of the six bounded per-component kernels (standard
real fields) and the per-component CPML/modulated kernels; everything else in the step (pre-magnetic ADE and
modulation clock, halo exchange, magnetic sources and corrections, wire sample/deposit, copper, electric
sources, circuits/networks, PEC/Mur, observation halos, accumulation) is unchanged and in the same order.
The in-process CUDA P2P transport, the one-process-per-GPU NCCL transport, the distributed adjoint forward
(checkpoints are captured from the same step) and distributed resume all run through this step.

- Ghost planes: each launch covers the shard's whole local storage. The seam faces of a shard solver are
  `none`, so ghost E node planes stay inactive exactly as in the bounded launches; the launch additionally
  updates the rank>0 ghost `Ex` cell plane and the non-last ghost `Hx` node plane from the received halos and
  recomputes the rank>0 ghost `Hy/Hz` cell plane before the magnetic exchange overwrites it. No owned cell
  reads these planes, so every owned field value is unchanged. One output reads a ghost plane: a point
  monitor inside rank 0's last owned cell interpolates `Hx` to the high ghost node, which the bounded
  launches never updated (the probe read 0 there on non-CPML runs); it now holds the updated value and the
  probe equals the single-device run (behavior change, see the row 15b evidence). The seam-cell `Ey/Ez`
  interpolation still reads the one-step-old received ghost node on both paths (pre-existing, recorded).
- DFT: each shard's full-field DFT gathers its weight row from the precomputed device table
  (`build_dft_step_tables` at output preparation, `accumulate_dft_gpu` per step, the device step counter
  restored by resume); the rows are bit-identical to the former per-step host weights and no host DFT
  arithmetic or host-to-device copy remains in the step. A field shutoff restores the sample counts and the
  source DFT of the steps actually run (`restore_dft_partial_sums`), as the per-step host path reported.
- Overlap: the fused launch has no interior/halo x split, so `overlap=True` runs the serialized schedule and
  reports `parallel_stats["overlap_active"] is False`. The byte model favors the fused launch by far more
  than the hidden halo copy (one tangential plane pair per half step): fused about 76 B per cell and step
  against about 144 B for the per-component kernels.
- Stays as is: the distributed adjoint reverse (native reverse kernels, the torch segment replay that defines
  the transposed halos, C4) and the `_DistributedFDTDMaterialGradientFunction` binding (C2); spectral and time
  observers keep the shard's eager accumulation; owner runtimes keep their eager steps.
- Fallback: a failed shard core activation raises `_FDTDCoreFallback`; `Simulation` rebuilds the existing
  distributed path (`core_path=False`, bounded kernels) once with `FDTDCoreFallbackWarning`. Solve-time errors
  of a distributed run are not retried. Row 16 deletes the bounded kernels with the other per-component kernels.

Follow-ups (not in this row):

- Row 15c (performance only, numerics unchanged, needs a core ABI bump): (1) an x-range in `CoreHalf`
  (`x_begin`, `x_end`) to restore the interior/halo overlap with two fused launches per half step, after which
  `overlap=True` reports `overlap_active` again; (2) per-shard device tables for the spectral point/plane
  observers; (3) whole-step CUDA graphs across the shards (per-shard half-step graphs around the exchanges, or
  one multi-device graph per K steps with peer memcpy nodes and cross-device event edges), once every piece of
  the eager step (owner circuit/network/wire runtimes, copper and observation halos) is capture-safe; (4) the
  NCCL interior/halo overlap on work handles.
- Seam-cell `Ey/Ez` observation defect (pre-existing on both paths, recorded by row 15b): a point monitor
  inside rank 0's last owned cell interpolates `Ey/Ez` to the high ghost node, which holds the halo received
  before the magnetic update (one step old) while the owned nodes are already advanced. On the row-15b seam
  scene the probe's `Ez` differs from the single-device run by about 10 % (full horizon) and 32 % (half
  horizon). Fix candidate: refresh the electric ghost plane (a second electric halo, or an observation halo)
  before accumulation. Owner to be assigned; not fixed in row 15b.

Gates: `benchmark/core_path/distributed_gate.py` (`DIST_SCENES`: open, PEC dielectric on a graded grid,
dense CPML, slab CPML with a conductive Lorentz medium, Debye/Drude/magnetic-Lorentz media, mixed
PEC/PMC/Mur faces, uniform current source, lumped circuit, embedded network, thin wire, copper foil with a
sampled closed surface, field shutoff with source normalization, density gradients on open and CPML
boundaries, and copper-foil resume) captures the two-GPU run at the full and half horizon and the same scene
on one device. Baselines are the pinned frozen-store addendum `addenda/w7-row15b`, captured on the base
commit (bounded/per-component shard kernels, host DFT). Evidence: `<evidence-root>/w7-20261108/row15b/`.

## Differentiable forward on the core path (W7 row 14, 2026-11-11)

`core_path_blockers` no longer refuses gradient-requiring scene tensors and `core_path_request_blockers`
no longer refuses trainable or `adjoint_checkpoint_stride` runs (inventory R15/R16, B13). The gradient
bridge builds its solver through the ordinary preparation, so a differentiable run of an admitted scene
gets the core runtime; material setup is unchanged (trainable scene tensors keep the PyTorch setup, which
defines the material pullback, C1).

- Forward: `_FDTDGradientBridge._run_core_forward` runs the ordinary solve lifecycle (`solve_steps`) with
  `checkpoints`/`checkpoint_stride`. The adjoint checkpoint is captured at the start step and after every
  completed multiple of the stride below the end: `run_chunks` ends its K-step chunks on the stride
  multiples (like the passivity and shutoff boundaries) and captures between chunks; the per-step loop
  captures before the step. The run output keeps the bridge layout (time monitors apart from the
  differentiable observers; embedded networks kept for rebinding), so `accept_forward` and the reverse
  sweep are unchanged.
- Fallback: a failed core forward (activation, capture or replay) rebuilds a fresh existing-path solver
  and reruns the existing gradient forward, as in the ordinary solve. Blocked scenes keep the existing
  gradient forward loop (`_run_forward_with_checkpoints`), which row 16 deletes with the existing path.
- Reverse: unchanged. Segment replay (`_replay_segment_states`, C4) still re-simulates from the
  checkpoints with explicit state tensors and stays the autograd definition; it never dispatches to the
  core launches.
- Differentiable circuits keep the eager generic step (`EMCircuitRuntime.adjoint_replay`, set by the
  bridge, now survives `prepare_time_series` instead of a post-hoc `graph_eligible` override).
- Behavior change: gradient runs of closed-transverse CPML scenes now sample the passivity guard, as
  every ordinary run does.

Gates: `benchmark/core_path/grad_gates.py` (`GRADIENT_SCENES`: scalar/module/density/TFSF/Lorentz-pole/
conductive/Kerr/full-aniso/lumped-port/circuit/time-monitor/Mur/odd-stride/eager/thin-wire/guarded-CPML
and Bloch scenes) captures the forward tape (all checkpoints, all bridge
output tensors), the loss and every input gradient. Baselines are the pinned frozen-store addendum
`addenda/w7-row14`, captured on the base commit (where every gradient run took the existing forward).
Evidence: `<evidence-root>/w7-20261108/row14/`.
## Native batch lanes on the core path (W7 row 15, batch half, 2026-11-11)

Inventory R11/B14: native batch lanes run the fused core Yee update. `core_path_blockers` and
`core_path_request_blockers` no longer emit `"Batch lanes"` (the 4-D field check and the
`_batch_source_scales` marker are gone). `PreparedBatch` builds lane solvers with
`_build_fdtd_solver_for_scene(..., batch_lane=True)`, which skips the solo admission decision, its fallback
warning and `activate_core_path`: the batch field slot owns the lane's execution, sources, observers and
adjoint checkpoints. `NativeFieldSlot._bind_operations` binds one `CorePathRuntime` per lane to the lane's slot
storage (fields, psi, coefficients) and records `execution_path="core"` on the lane solver;
`NativeFieldSlot.advance(lanes)` launches the fused H and E kernels of the running lanes in place of the six
lane-strided per-component launches. Retired lanes get no launch, so their fields stay frozen (the activity
mask still gates sources, ADE, foils and observers). Lanes whose coefficient storage is shared (equal
coefficients, for example source-scale templates) share one set of compressed coefficient tables
(`CorePathRuntime(coefficient_tables=...)` memoizes by storage).

Design choice (lane loop rather than a lane-strided core kernel): the fused kernel moves the same bytes per
cell for one lane either way, because lanes share no field data; a lane-strided kernel would only fold the
2·B fused launches per step into 2. A batch step already issues per-lane work (port/tail block, PEC clamp,
DFT accumulation), so the step stays O(B) launches either way, and all of it is replayed from CUDA graphs.
A lane-strided kernel would also need one coefficient mode for all lanes (uniform and indexed lanes mix in
one cohort), padded per-lane tables and a new core ABI. It pays off only for lanes too small to fill the GPU
in one launch; it can be added later with the same per-cell arithmetic. The six batched per-component
kernels (`update_{magnetic,electric}[_cpml]_batch`) have no caller now; they leave the main extension with
the other per-component kernels in row 16 (one ABI bump there).

Gates: every lane is bitwise equal to the same variant run alone on the core path and to its frozen
existing-path batch payload (`benchmark/core_path/batch_gate.py`, 12 `BATCH_SCENES`, frozen-store addendum
`addenda/w7-row15a`), including gradients through batch lanes. Evidence: `<evidence-root>/w7-20261108/row15a/`.

## Complex (Bloch split) fields on the core path (W7 row 13, 2026-11-11)

`core_path_blockers` no longer refuses Bloch/complex fields (inventory R06, B12). Only a complex run
beside separate physics (ports, lumped/coaxial elements, copper, thin sheets, wires, circuits,
networks, SIBC, cut surfaces) keeps the existing path, with the reason
`Bloch/complex fields with separate physics`: those pieces' complex branches have not qualified
inside the separate-physics closure. The pieces, in `_field_update_block` order:

| Piece | Core path | Bytes / launches |
| --- | --- | --- |
| H | The fused core H launch twice: real fields + real psi, then imaginary fields + imaginary psi (`CorePathRuntime.launches["h"]`, `["h_imag"]`, sharing the compressed coefficient table). H has no Bloch wrap (forward differences stay inside the grid), so this is the existing H update | 2 launches |
| E | Closure piece: the existing Bloch split-field kernels (`update_electric_fields_bloch` or `_bloch_cpml`: wrap phase on the Bloch axes, CPML stretch folded in per PML axis, real and imaginary PEC/PML face clamps) in the fused E position. No real fused E launch is built for complex fields | 3 + 2 per PML slab region and part |
| Sources | `CoreStepLaunch` op 7 (`core_abi_v15`): the `add_bloch_source_patch_kernel` scatter (interior and phase-rotated wrap images, prepared gather) reading a per-step row. Scalar terms stage `(signal phase_r + quadrature phase_i, signal phase_i - quadrature phase_r)`, CW terms the `(cos, sin)` quadratures and apply the backend's two torch-ordered passes, time-shifted terms evaluate `evaluate_time_shifted_waveform_torch`'s float32 operation order per cell (one rounding per torch op) and its causal gate. Rows are the eager host doubles cast to float32 once per run | 1 launch per term |
| Fixed-k TFSF | Unchanged disjoint complex patch additions (already capture safe) | 3 launches per term |
| Fixed-k line | Its torch curl/CPML update, now device driven: a CUDA step clock and per-run drive (`complex64`) and phase (`complex128`) rows staged by `prepare_tables` at solve entry from the eager formulas; the per-step `.item()` and host waveform/phase evaluation are gone (both paths) | about 60 small torch launches per step |
| DFT | `build_dft_step_tables` accepts complex fields: the same weight rows drive a second launch on the imaginary field into the auxiliary accumulators (core op 2 twice, `accumulate_dft_gpu` on the existing path) | 2 launches |
| Spectral monitors | `IndexedObserverTables` (generalizing the native point tables): exact eager host weights for every point and plane group, plane cells enumerated as indexed points in accumulator order, real then imaginary field (`uses_indexed_observers`) | 1 launch per group, frequency and part |
| Passivity guard | Complex fixed-k cells sample at the K-chunk ends (row 5 hook) | outside the graph |

Everything is bitwise identical to the base existing path (G1 below), so the frozen store is the oracle.
Follow-ups (performance only, recorded, not in this row): (1) a fused fixed-k line kernel (H and E advance,
drive, ends and spectrum: about 60 launches to 3) plus one fused fixed-k TFSF launch per family; the torch line
also defines the CPU/float64 unit tests, which move to a tests-side oracle with it. (2) A complex fused core E
(Bloch wrap mode in `yee.cu` `backward`, CPML stretch inline): saves the dense `cex_*` coefficient streams and
the CPML correction passes (about 20 % of the complex E bytes and 3 + 2n launches to 1); it changes the
rounding order of the Bloch+CPML update, so it needs a Rule 2 (T1) declaration.
Gate scenes: `BLOCH_SCENES` (frozen addendum `addenda/w7-row13`, captured on master 12f63f934 with the existing
path forced and torch setup).

## Float64 fields on the core path (W7 row 12, 2026-11-11)

`core_path_blockers` and `core_path_request_blockers` no longer refuse `field_dtype=float64` ("non-float32
fields"), and `validate_fp64_forward` no longer requires `cuda_graph=False`. The float64 admission is otherwise
unchanged (fixed grid, direct `LumpedPort` Norton drives, Box scalar media and two-sided `CopperFoil`, all-face
dense CPML, one frequency, float64 sparse spectra, detached inputs); widening it (sources, dispersive media,
periodic faces, TFSF) is a separate capability question, because the existing path has no float64 reference
for those pieces either.

- Kernel: `core/yee.cu` is templated on the field precision. `CoreHalf<T>`, `CoreAxis<T>`, `CorePsi<T>` carry
  `T` fields, inverse spacings, CPML profiles, psi and coefficient tables; the pinned float32 intrinsics go
  through `fma_rn`/`mul_rn`/`sub_rn` overloads (`__fmaf_rn`... for float, `__fma_rn`... for double), so the
  float32 instantiations are unchanged source. Float64 instantiates the uniform and indexed coefficient modes
  for every CPML layout; live and modulated coefficients stay float32-only (nonlinear, modulation and breakdown
  are outside the float64 admission) and `CoreLaunch` refuses them for double fields. `_compress_coefficients`
  deduplicates float64 coefficient tuples by their int64 bit patterns (the accelerated hash table is float32-only).
- Observers: the core fused gather (`core/observers.cu`) and the main grouped and indexed point gathers accept
  float64 fields with float64 weights, so `make_single_observer_engine` and `uses_native_point_observers` no
  longer decline float64 runs; float64 point/plane/closed-surface spectra run inside the whole-step graph.
- Setup: material setup stays on PyTorch for float64. `compiler/accel` (`eligible`, `compress`,
  `try_build_update_coefficients`) and its kernels are float32-only, so the core path's setup request falls
  through to the PyTorch compile. Dense CPML is still forced for float64 (`_initialize_fdtd_solver_for_scene`);
  the core kernel would take slab psi too, but no float64 slab reference exists. `absorber="auto"` resolves to
  CPML for the all-PML admission, as fixed in 0.6.0.
- ABI: `core_abi_v13` -> `core_abi_v14`, `stable_abi_v15` -> `stable_abi_v16`.
- Still on the existing path in float64: `PowerLossMonitor` (an unadmitted core monitor type), now also in
  graphs (existing field/port/observer graphs). Coordinate charts moved to the core path in W7 row 10.

Gates: `FP64_SCENES` (`fp64_vacuum_port`: uniform coefficients, native point observers; `fp64_slab_port`:
graded x grid, lossy slab, indexed coefficients, finite-plane and closed-surface spectra through the grouped
gather; `fp64_copper_port`: two-sided copper over a lossy board, closed-surface spectra; `fp64_two_ports`:
driven and undriven ports, no monitors) join `GATE_SCENES`, with port outputs and copper losses pinned as
physics tensors. Their existing-path baselines (existing path forced, PyTorch setup, float64 eager, the only
float64 execution the base admitted) are the pinned frozen-store addendum `addenda/w7-row12`. New path vs
reference is Rule 2 with float64 bounds; evidence: `<evidence-root>/fdtd-coord-20261111/w7-row12/`.

## Coordinate charts on the core path (W7 row 10, 2026-11-11)

`core_path_blockers` no longer refuses `scene.coordinates` ("coordinate charts", inventory R05/B10). A
`BendCoordinates` chart has no runtime piece: `compiler/coordinates.attach_coordinate_metric` folds the
metric (`1/h` for the s-component, `h` for u and z, `h = 1 - kappa u` at each component's own u sample)
into the cell and edge eps/mu/sigma_e/sigma_m tables once at compile time, `_mapped_dual_face_weights`
weights the E_z dual-face material quadrature, `compiler/ports` integrates the physical aperture metric,
and `fdtd/copper_foil.initialize_copper_foils` scales the split-foil half-cell capacity/conductance by the
integrated half-cell metric and the trace response (B, D) by `alpha`. The existing runner then steps the
lowered tables with its ordinary kernels, and so does the core path: the metric-varying u profile only
makes the coefficients non-uniform, so `_compress_coefficients` takes the indexed table (one entry per
distinct coefficient tuple, about Ny per material). Copper runs through `_field_update_block` in the
separate-physics closure and the observers through the ordinary core gathers; the chart branches there
are all setup-time (`material_metric_factors`, `response_alpha`, observer stencil validation), so nothing
needed porting into the core runtime. Accelerated setup still refuses charts (`compiler/accel/dispatch.eligible`),
so chart material setup stays on PyTorch on both paths (inventory B28, not in this row).

Gates: `CHART_SCENES` (`chart_board_port`: concave chart around a lossy board, slab CPML, point and
closed-surface spectra; `chart_copper_port`: the public split-copper fixture, dense CPML;
`chart_convex_two_ports`: convex chart on a graded u grid with PEC u faces (explicit CPML under the passivity guard), a driven and a loaded port and
a finite chart-face plane; `chart_fp64_copper`: the float64 public fixture) join `GATE_SCENES` with ports
and copper losses pinned as physics tensors. Their existing-path baselines (existing path forced, PyTorch
setup, graphs, as the main store) were captured at master 12f63f934 into the pinned frozen-store addendum
`addenda/w7-row10`. Chart scenes with `PowerLossMonitor` keep the existing path through the monitor
blocker (R18). Evidence: `<evidence-root>/w7-20261108/row10/`.

## Gyromagnetic ferrites on the core path (W7 row 9, 2026-11-11)

`core_path_blockers` no longer refuses `gyromagnetic_enabled`. The Torch drive snapshot and coupled
implicit-midpoint step (about 20 Torch launches per step) are replaced by two CUDA kernels in the main
extension (`kernels/gyromagnetic.cu`, ops `gyromagnetic_snapshot` and `gyromagnetic_step`,
`stable_abi_v14` -> `stable_abi_v15`), which both paths call through `solver._snapshot_gyromagnetic_drive` /
`solver._step_gyromagnetic_coupled`. `CorePathRuntime.prepare_step`'s ordinary closure calls them in
`_field_update_block`'s position: the snapshot after the magnetic dispersive advance and before the fused H
launch, the coupled step after the magnetic sources and the magnetic dispersive corrections, ahead of the
electric half. The separate-physics closure already calls `_field_update_block`. The magnetization ADE is
local, so ferrites need no PEC-clamp or source-table change; `m_u`, `m_v`, `dm_u` and `dm_v` were already in
`graph_state_tensors` (checkpoint schema), and the drive snapshot is rewritten every step before it is read.

Kernel design: one thread per active ferrite cell of the shared (Nx-1, Ny-1, Nz-1) overlap
(`state["cells"]`, int64 overlap offsets), so cells outside the ferrite cost nothing (the Torch hooks ran
about 25 elementwise launches over dense whole-grid overlap fields per step). Each overlap cell owns one entry of each H
component (identity collocation), so the H read-modify-write needs no atomics and runs are deterministic.
The coupled B/C blocks are packed into one `(8, overlap)` tensor and the general-bias frame into one
`(6, overlap)` tensor; the named state entries (`b00`, `ux`, ...) are views. The arithmetic keeps the Torch
order (fused multiply-adds where Torch used `addcmul`): the kernels are bitwise equal to the Torch oracle on
every gate scene in float32 and float64, and the general path still reduces to the fast path bitwise for an
axis-aligned bias. The per-step Torch scratch buffers (`hu`, `hv`, `new_u`, `new_v`,
`corr_*`, `active_f`) are gone from the runtime state; the Torch hooks live on as test oracles in
`tests/oracles/gyromagnetic_recurrence.py` (with the pure Cayley recurrence). Follow-up (not in this row):
the dense per-cell coefficient fields (8 coupled + 8 Cayley + inverse permeability + frame) could be
compacted to the active cells.

Gates: `FERRITE_SCENES` (`ferrite_axis_dipole`: -y bias fast path, vector dipole inside the ferrite, slab
CPML; `ferrite_oblique_tfsf`: oblique conductive lossy ferrite under a TFSF plane wave, general path, dense
CPML; `ferrite_mixed_bias`: +z/-z latching regions, one lossless, mixed-bias general path, open boundary;
`ferrite_lumped_port`: a ferrite beside a lumped port, separate-physics closure) join `GATE_SCENES`, with the
magnetization state pinned as physics tensors. Their existing-path baselines (existing path forced, torch
setup, G1/G1-rule2/G4) were captured at master 5d1d244de into the pinned frozen-store addendum
`addenda/w7-row9` (32 payloads). New path vs reference is Rule 2 (step bound per kernel step against the
Torch oracle, run bound at the half and full horizons against the frozen store); evidence:
`<evidence-root>/fdtd-coord-20261111/w7-row9/`.

## Cheap admissions (W7 row 4, 2026-11-10)

Inventory items B1-B4 leave `core_path_blockers`: PMC faces and PMC symmetry planes (R09), permeability
and magnetic conductivity (R10, R04), Mur and mode-ABC faces (R07, R08), and `MediumMonitor` plus the
projection and Gaussian-overlap monitors (part of R18). The monitor check reads `scene.resolved_monitors()`,
so a monitor that lowers to finite planes is admitted by what it records.

- PMC: core `yee.cu` `backward` returns the odd-image derivative of `electric.cu`
  `boundary_diff_low/high` (`2 H_0 / d` low, `-2 H_{N-1} / d` high) for every E component, axis and CPML
  layout; H needs no change. The interior fast path already excludes PMC axes, as the standard launch does.
  `core_abi_v12` -> `core_abi_v13`.
- Permeability and sigma_m need no kernel change: both enter only the H decay/curl tables that the core H
  kernel reads; magnetic ADE uses the shared ordered helpers.
- Mur/mode ABC: the core tail already called `apply_mur_boundaries` (native first-order planes plus the
  Higdon torch planes, captured inside the whole-step graph as on the existing tail graph); `_mur_state`
  and `_abc_higdon_state` are part of `graph_state_tensors`. The separate-physics graph's Mur assertion is
  removed: its tail applies Mur at the existing port block's position. Internal absorbers were already
  admitted without a gate scene; `internal_absorber_guide` now gates them.
- Gate scenes: `W7R4_SCENES` in `benchmark/core_path/scenes.py` (15, including `pmc_cavity`, which leaves
  `SRCMON_REFUSED_SCENES`). Their frozen existing-path payloads were captured on the base commit (master
  fe59b3cd2, existing per-component path and torch setup) into the pinned store addendum
  `addenda/w7-row4` (`tests/support/frozen_oracles.py` loads pinned addenda).
- `tests/support/existing_runner.py` marks its scenes with a test-side request blocker instead of the
  now-admitted `MediumMonitor`.
- Still blocked: gyromagnetic media (row 9), full anisotropy (row 6), coordinate charts, Bloch/complex fields,
  closed-transverse CPML, batch lanes, distributed, fp64, gradients, resume/non-initializing preparation,
  material-cache overrides, and the `PowerLossMonitor`, `SurfaceFieldTimeMonitor`, `AuxFieldTimeMonitor`,
  `DipoleEmissionMonitor`, `ComponentStressMonitor` and `WireMonitor` types.

## 0.7.0 flag removal (W7 row 3, 2026-11-09)

`fdtd_core_path` and `accelerated_setup` are removed from `Simulation.fdtd(...)` and
`FDTDConfig` (user decision 2026-11-08: breaking change, no shims, no private test switch).
`validate_core_path_request`, the explicit-True `initialize=False` refusal and the explicit-True
per-step continuation after a failed whole-step capture are deleted. The automatic blocker
decision (`core_path_decision`), the fresh existing-path rebuild and `FDTDCoreFallbackWarning`
stay while scenes remain blocked. Tests of the existing runner's own CUDA graphs drive it with a
naturally blocked scene (`tests/support/existing_runner.py` adds a `MediumMonitor`); when a later
row admits that monitor, those tests fail on their path assertion and need another blocker. The
`True`/`False` bullets of the v0.6.0 policy below are superseded.

## Material cache on the core path (W7 row 11, 2026-11-12)

`core_path_decision` no longer refuses a configured `FDTDMaterialCache` (inventory R22, B8). The
cache retains the compiled material model (cell and Yee-component eps/mu/sigma channels, PEC and
copper masks, dispersive pole banks), which is the input of the coefficient builder, not a core
table. A bound cache still disables accelerated setup (the snapshot is the PyTorch compilation,
and `compiler.accel.dispatch.eligible` refuses a material override), so its coefficients come from
PyTorch setup; the core path then builds its compressed coefficient, source and observer tables
from the solver tensors as for any PyTorch-setup scene (charts, float64, dispersive coefficients).
No runtime code changes beyond dropping the blocker. Gate scenes `CACHE_SCENES` (miss, hit and
local Box recompile on a Mesh dielectric, slab, Debye slab, copper foil, copper terminals with a
port and a fixed-k Bloch slab) bind a cache and compare against the frozen payloads of their base
gate scene (G1 and G4), so a cached core run equals a fresh core run and the existing path. Cache
misses still compile with PyTorch; letting a miss use accelerated setup is a separate,
performance-only follow-up. Evidence: `<evidence-root>/w7-20261108/row11/`.

## Resume on the core path (W7 row 7, 2026-11-11)

`core_path_decision` no longer refuses `resume_from` or `initialize=False` (inventory R20/R21, B7).
A resumed run keeps its prepared solver: `PreparedSimulation.run(resume_from=...)` preflights and
restores the checkpoint into the core solver's own tensors (`restore_resume_checkpoint` copies in
place, so the captured `CoreLaunch` pointers stay valid), and `CorePathRuntime.run_chunks` enters at
`start_step`. Every core table is indexed by the absolute device step (source, auxiliary TFSF, DFT
weight and port drive tables are compiled for the whole run at activation or from the restored port
clocks), so the only changes are: `run_chunks` sets `runtime.step` to `start_step` before warmup and
capture (the restored `_dft_step`, observer-engine step and port counter already equal it), and
`NativePointObserverTables` starts its host weight recurrence from the restored observer entries at
`start_step`. Chunk boundaries (guard, shutoff, progress, plot) are absolute multiples, as in the
per-step loop. The resume rebuild (`_fdtd_solver_for_resume`, torch setup) is deleted, so a resumed
eligible scene uses accelerated setup, and a blocked scene resumes on its prepared existing-path
solver. A capture failure during a resumed run still rebuilds a fresh existing-path solver and
restores the checkpoint into it. `initialize=False` records the decision without activating; its
initializer owns activation (the tests-side torch-setup oracle activates the core path after its
own `init_field`). Gates: split runs (`run_until(17)` and `run_until(steps // 2)` + disk checkpoint +
resume) equal the uninterrupted core run bitwise on all GATE_SCENES that admit checkpoints, and the
base existing-path resume; the resume/checkpoint tests give identical outcomes on base and branch.
Evidence: `<evidence-root>/fdtd-coord-20261111/w7-row7/`. The resume bullets of the v0.6.0 policy
below are superseded.

## Passivity guard in core chunks (W7 row 5, 2026-11-11)

`core_path_blockers` no longer refuses real-field closed-transverse CPML/stable-PML
(`closed-transverse CPML (passivity guard)`, inventory R01/B5). `CorePathRuntime.run_chunks`
ends a chunk at every multiple of `passivity.SAMPLE_INTERVAL` (32) while
`solver._passivity_state` exists, and after the replay calls
`passivity.sample_completed_step`, the same hook the per-step loop calls. Trip rule: the
guard samples after exactly the completed steps the per-step loop samples (every multiple
of 32), with the same `total_energy` reduction (torch fixed-order device reductions, no
atomics), so the sampled history, the check windows and the trip step are identical at
K = 32, 16 and 1; the shutoff, progress and plot boundaries only add chunk ends. Chunk
granularity therefore never delays a trip. `_solve_fdtd_with_fallback` re-raises
`FDTDPassivityError` instead of rebuilding the existing path: a trip is the guard's
verdict on the run. Skipped guards (uncounted storage) keep their warning and status and
do not split chunks. G1 pins the guard state (`passivity/*`) for the 14 `W7R5_SCENES`
(frozen addendum `addenda/w7-row5`). Complex fixed-k cells keep the Bloch blocker. Each
sample is about 40 small torch launches every 32 steps outside the graph.

## Full anisotropy on the core path (W7 row 6, 2026-11-10)

`core_path_blockers` no longer refuses `full_aniso_enabled`. `CorePathRuntime.prepare_step`'s
ordinary closure runs the existing correction kernels in `_field_update_block`'s position:
`capture_aniso_conduction_currents` (sigma . E^n, conductive media only) before the fused E launch,
then `apply_full_aniso_corrections` (plain or CPML-stretched off-diagonal curl(H)) and
`apply_full_aniso_conduction` after it, ahead of TFSF E and the sources. Off-diagonal ADE
subtraction already lives in `_apply_dispersive_corrections`. `pec_clamp_plan` keeps both
historical clamp sites for full anisotropy, because the corrections write E after the fused
stores that P1/P2 rely on. Full anisotropy refuses copper, sheets, wires, lumped elements and
cut surfaces, so the separate-physics closure (which already calls `_field_update_block`) does not
see it in practice. Gradient runs stay on the existing forward (R15/R16), and the C4 replay is unchanged.

Closure piece rather than fusion into core E, by fp32 streaming bytes per cell per step:

| Piece | Whole grid | Inside the tensor support |
| --- | ---: | ---: |
| Fused core E (O1 model) | about 36 + 12 coefficients | same |
| Off-diagonal curl correction, 3 kernels | 96 (two 4-weight tables per edge, read before the zero test) | + about 60 (E read/write, H stencil) |
| CPML variant | 96 | + about 60 + 72 (three psi read/write per edge) |
| Conductive capture + conduction | 36 + 96 | + about 36 |

Fusing into the core E kernel would save at most the E read/write and the H re-reads inside the
support (about 60 B/cell, zero outside). The full-grid coefficient scan (96 B/cell) remains, because
the tables are dense. It would also need six more coefficient streams, nine psi streams and periodic
flags in `yee.cu`, new CPML-layout specializations and an ABI bump, with every non-aniso
specialization re-qualified. The closure piece changes no CUDA code, adds 3 launches per step
(7 when conductive) inside the K-step graph and is bitwise by construction. The larger saving is
compacting the weight tables to the tensor support (W6-style edge lists); that is left as a
follow-up. The per-component kernels stay until row 16.

Gates: `ANISO_SCENES` (`full_tensor_slab`; `aniso_conductive_dipole`: all off-diagonals, sigma_e,
dipole inside the crystal, CPML without overlap; `aniso_debye_periodic`: Debye pole, periodic x/y,
40-layer graded absorber) join `GATE_SCENES`. Their existing-path baselines (existing path forced, torch setup,
G1/G1-rule2/G4) were captured at fe59b3cd2 into the pinned frozen-store addendum `addenda/w7-row6`
(24 payloads). Evidence: `<evidence-root>/w7-20261108/row6/`.

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

### W7 row 2: frozen existing-path oracles (2026-11-08)

User decision 2026-11-08: tests and gates compare against **frozen payloads** captured once from the
existing path, not against a private test-only path switch.

- Store: `$WITWIN_EVIDENCE_ROOT/w7-20261108/frozen-oracles/` (outside git; captured at master
  0888c6cf on GPU 1). Its `manifest.json` records commit, environment, GPU and a sha256 per payload;
  `tests/support/frozen_oracles_manifest.json` pins the manifest digest. `tests/support/frozen_oracles.py`
  loads and verifies payloads and skips with a clear reason when the store is absent.
- Gates: `benchmark/core_path/gates.py` captures only the candidate (the default execution). Its
  baseline is `<store>/gates/{g1,g1-rule2,g4,g2}`; `--baseline` selects another directory.
  `setup_gates.py` takes its torch arm from `tests/support/torch_setup_oracle.py` and its G1 run
  baseline from `<store>/setup-gates/g1`.
- Tests: the A/B nodes load the existing arm from the store and keep their criteria (`torch.equal`
  or the declared Rule 2 bound). Accelerated-vs-torch setup nodes call the torch-setup oracle, which
  runs the solver's `init_field` outside the accelerated-setup scope; it never changes how
  `Simulation.prepare` executes.
- A changed payload is a re-capture: a new store id and pin, with the mover tier declared per
  `numerical_equivalence_standard.md`.
- Port BITID scope (user decision 2026-11-09): fixtures whose default execution steps the core
  whole-step graph are compared by end state only; their per-step V/I/drive histories are skipped because the step-loop
  history hook cannot observe the core whole-step graph.
- Backlog (W7): core-side per-step history capture for the port BITID fixtures, then restore their
  per-step history comparison in `test_port_step_graph_bit_identity.py` and
  `test_raw_phase_bit_identity.py[ports]`.

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
