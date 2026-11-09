# FDTD core-path deep CUDA optimization (setup + stepping)

2026-11-03 release follow-up: source-patch atomics are replaced by prepared ordered
gathers, including core periodic injection. See
[the source accumulation record](../memory/fdtd-source-patch-determinism.md) for
ordering, rounding, potential expectation movement, and outstanding GPU gates.
The existing oblique capture/core admission exclusions remain until separately qualified.

## Rule-2 comparison and Rule-1 determinism (2026-11-04)

Bitwise comparison remains the default until a branch explicitly declares `rule2`.
For that declaration, capture G1 with both horizons, then compare on the CPU:

```bash
python -m benchmark.core_path.gates capture --gate g1 --half-horizon --scenes planewave_vacuum --output /path/to/capture
python -m benchmark.core_path.gates compare --gate g1 --equivalence rule2 --scenes planewave_vacuum --output /path/to/capture
python -m benchmark.core_path.gates compare --gate g4 --equivalence rule2 --scenes planewave_vacuum --output /path/to/setup-capture
python -m benchmark.core_path.setup_gates g4 --equivalence rule2 --output /path/to/setup-receipts
python -m benchmark.core_path.gates determinism --dirs /path/to/repeat1 /path/to/repeat2 /path/to/repeat3 --arm candidate --output /path/to/repeat-receipt
```

G1 saves `tensors.pt` and `tensors_half.pt` in each arm/scene directory, with
separate capture metadata. Each horizon uses a fresh worker; the half worker keeps
the full run's configuration and tables and stops at `steps // 2` through the
existing solve stop boundary. Rule 2 uses the shared comparator's two-horizon run
bounds and growth gate. E and H components, volume DFT field families and each
monitor's component set receive independent family scales; monitor coordinates
and frequency metadata cannot inflate a component scale. Physics records retain
their own per-record bounds. G4 uses setup bounds without families. Discrete
names and integer/bool tensors always remain exact. Rule-2 receipts include
per-tensor T0/T1/T2 metrics and tier counts; any T2 fails.

For Rule 1, first run the same capture command into at least three distinct output
directories, each with fresh workers. The CPU `determinism` command requires
matching snapshot manifests and tensor keys, shapes and dtypes, and applies
`torch.equal` to every tensor at every saved horizon. The candidate arm is its
default. Missing snapshots, too few directories or any mismatch fail. Capture and
accelerated setup execution require CUDA; comparison and determinism load tensors
onto the CPU. This harness update was validated with synthetic CPU tensors only.

Rebase onto `04ae392e` (old -> new): `ff1e4118` -> `a15ed17c`, `648cf7e9` -> `d692c1ab`, `f607c545` -> `96a796cf`, `dddf847e` -> `31dda471`, `e577cc4c` -> `99843739`, `f63949d7` -> `dc4b4d0c`, `3a138943` -> `c8daca2f`.

Branch `perf/core-cuda`, worktree `.worktrees/core-cuda`, base master `65de004b`.
User request 2026-11-01: deep CUDA optimization of the existing basic features, setup and
iteration both, with correctness guaranteed. It runs in parallel and must not touch the
features currently under development.

## 1. Starting point (evidence, not estimates)

- The per-cell Yee update is native CUDA but naive. Each field component has its own kernel
  (3 for H, 3 for E per step). There is no shared-memory tiling, no register or temporal
  blocking, and no E/H component fusion. The block is `dim3(128,2,1)`, the code is FP32,
  and it is built with `-O3` only. See `fdtd/runtime/stepping.py:120-311` and
  `fdtd/cuda/kernels/{electric,magnetic}.cu`.
- These kernels already sit at the bandwidth roofline: 4.8 Gcells/s, about 690 GB/s, roughly
  90% of A6000 peak at 300³ PEC, and 4.47 Gcells/s at 200³ with CPML. The traffic is about
  144 B/cell/step (`docs/assessments/perf-architecture-audit-2026-07-21.md`). **Every gain
  must come from moving fewer bytes or from removing launch and host work. Tuning the
  existing kernels gains nothing.**
- The step is driven by a Python loop (`stepping.py:2504`) with graph capture per block.
  The waveform is evaluated on the host every step (`excitation/temporal.py:460-530`),
  and source-patch launches stay outside the graphs. The full-field DFT does fp32
  read-modify-write on every step, with no decimation (`cuda/kernels/spectral.cu:9`).
- Setup is PyTorch eager plus CPU code:
  - Kottke/subpixel averaging loops over components × subsamples × structures, each pass
    over the full grid (`compiler/materials.py:538-630, 1300-1420`).
  - The mesh SDF is brute-force points × triangles in torch, with a CPU BVH
    (witwin-core `mesh_sdf.py:234-267`). Cold compile measured 39 s at 5.4M cells
    (`docs/assessments/datagen-p0-profile-2026-10-30.md:41`).

## 2. Scope: the "core path"

A scene is **core-eligible** when it uses only:

- isotropic or diagonal, non-dispersive ε and σ_e (μ = 1), with compiled subpixel
  coefficients allowed;
- PEC, CPML/PML and periodic boundaries (real fields);
- PointDipole, PlaneWave (soft or TFSF) and GaussianBeam, with standard waveforms;
- PointMonitor, PlaneMonitor, FluxMonitor and full-field DFT;
- single GPU, forward only.

Every other configuration keeps the existing path, **bit-for-bit unchanged**. This
includes dispersive/ADE, full anisotropy, nonlinear and modulated media, copper/thin
sheet/thin wire, lumped and wave ports, circuits, Bloch/complex fields, mode sources and
monitors, multi-GPU, adjoint, Batch lanes and resident updates. Phase O4 may extend
eligibility (ADE, adjoint), each step under the same gates.

## 3. Non-interference rules

- **New code goes in new files:**
  - `fdtd/cuda/core/` for kernels, built as a **separate extension** with its own ABI
    marker `core_abi_v1`. This avoids colliding with stable_abi v6 (PCB) and v7
    (reserved for F5 Bloch). `extension.cpp` and the existing `.cu` files are not edited.
  - `fdtd/runtime/core_path.py` for eligibility, dispatch and the graph loop.
  - `compiler/accel/` for setup kernels.
- **Shared hot files get one additive hook each, no edits elsewhere:**
  - `fdtd/runtime/stepping.py`: a single dispatch branch.
  - `compiler/materials.py`: a single accelerator call site.
  - `simulation.py`: one option.
  - PCB must ack the stepping.py hunk before merge (rule from the resident-update line).
- Do not touch:
  - `compiler/{copper_foil,surface_impedance,planar_pec,thin_wire,ports}.py`,
    `fdtd/ports.py`, thin-sheet/cut-surface modules, `fdtd/distributed/`, `fdtd/adjoint/`,
    `mom/`;
  - witwin-core (the mesh SDF accelerator is registered from the maxwell side through
    the existing hook);
  - benchmark caches and references.
- **Opt-in until accepted:** `Simulation(..., fdtd_core_path=True)`. The default flips only
  after all gates pass AND the user approves. Then the old per-component kernels serve
  only the configurations the core path does not cover. No permanent duplicate path
  remains for core-eligible scenes.
- **Deliberate exception to "no parallel implementations":** while the line is opt-in, the
  core kernels coexist with the per-component kernels so that features under development
  are not disturbed. The exception ends at the default flip. From then on, core-eligible
  scenes have exactly one path, and the replaced dispatch branches and unused kernel
  instantiations are deleted in that same merge. This needs the user's explicit OK.
- Rebase on master at least daily. Keep each phase a separate, mergeable commit series.

## 4. Correctness gates (every phase)

| Gate | What | Pass criterion |
|---|---|---|
| G1 bit identity | Core path vs existing path on the core scene set: fields after N steps, monitor phasors, flux. Fresh interpreter for each arm. | `torch.equal`. Any non-identical item, such as a reassociated sum or DFT decimation, must be named in the phase spec in advance and pass G1′ instead. |
| G1′ bounded | For the items pre-named under G1 | max relative difference ≤ 1e-6 on fields and ≤ 1e-5 on phasors, with the source of the difference written down |
| G2 external reference | Basic benchmark rows: planewave_vacuum, pml_only, planewave_periodic_vacuum, dipole_vacuum, dipole_ey, dipole_offcenter, dielectric_slab, dielectric_sphere, pec_sphere, sigma_e slab, anisotropic_uniform_grid | Every scored metric within 1e-4 absolute of the pre-change row. Pass/fail status unchanged. |
| G3 convergence | ε_r = 4 sphere ladder at ppw 16/24/32 against the converged external reference (`benchmark/label_qualification/external_ref_lq1b.py`) | Reproduces 11.1 / 4.9 / 2.9 % (shell) to ±0.05 percentage points, with order ≈ 2 |
| G4 setup | Compiled coefficient tensors (ε/σ/curl/decay per component), source and monitor tables: accelerated vs current compile on the scene set, including mesh and boolean geometry | `torch.equal`. Mesh SDF sign and distance: identical sign, and distance ≤ 1 ULP·scale where a reduction order differs, pre-named. |
| G5 performance | Exclusive-GPU timing (A6000, sm_86) on the size ladder 64³/128³/256³/384³, vacuum + CPML and dielectric + CPML, with and without a plane DFT. Nsight Compute for DRAM bytes/cell/step. | Each phase states its target before the code is written. Results are recorded in an assessment doc with raw JSON. |

The core scene set (O0 defines it in `benchmark/core_path/scenes.py`) is the G2 rows plus
mesh, boolean and multi-structure dielectric scenes, and the CPML-dense/compressed and
odd-size grids.

## 5. Phases

| Phase | Content | Expected gain (to be confirmed by O0) |
|---|---|---|
| **O0** baseline + harness | The core scene set; the G1–G5 harness `benchmark/core_path/gates.py`; a per-stage timing breakdown of setup and stepping; Nsight bytes/cell per kernel; a byte-floor model per configuration. Assessment `docs/assessments/core-path-baseline-<date>.md`. | — |
| **O1** fused Yee update | One H kernel and one E kernel updating all three components, with CPML ψ fused and PEC/periodic handled in-kernel. Neighbour reads are shared through registers/smem tiles. Coefficient compression: uniform → scalar; lossless → curl only; bulk cells → material-index + table, with interface cells kept as per-cell values. FP32 arithmetic expression order is kept identical so that G1 holds. | Vacuum 144 → ~72–96 B/cell/step (1.5–2×); dielectric about 1.4× |
| **O2** whole-step graph | Device-side waveform tables (precomputed per step, same float expressions). Source injection fused into the E-kernel epilogue. Point/plane DFT fused into the update epilogue. The K-step loop is captured as one graph, with no Python per step. Optional DFT time decimation (opt-in, G1′). | Small grids 1.4–2.4 → about 3–4 Gcell·step/s; DFT overhead −50% or more |
| **O3** setup on GPU | A single-pass CUDA voxelization + Kottke averaging over all structures, using per-cell structure candidate lists from a coarse bin. Mesh SDF/inside test through a GPU BVH, registered via the core accelerator hook. Fused coefficient assembly. | Mesh compile 39 s → a few seconds (to be measured) |
| **O4** extensions (each needs separate approval) | Temporal/wavefront blocking of the interior (2.5D); ADE fused into the E update; adjoint for the core path; eligibility widened to diagonal anisotropy and Batch lanes | Temporal blocking can reach 2× beyond O1, but carries high complexity |

Order: O0 → O1 → O2, with O3 running in parallel with O1/O2 because its files do not
overlap. Each phase merges separately after review, following the merge checklist.

## 5a. Coordinator conditions (the coordinator, 2026-11-01)

- **O3 depends on batch 4.** The O3 call site in `compiler/materials.py` lands only after
  coverage/batch4-kottke-default merges, rebased onto it. G4 must pass `torch.equal` for
  `kottke_edge`, `kottke_smooth` and `polarized`, not just the default.
- **Batch 4 changes the default subpixel spec** (coverage/batch4-kottke-default, from
  boolean-geometry-stl-coverage) to `SubpixelSpec(samples=4, averaging="kottke_edge")`,
  and adds a fail-closed error when kottke_edge meets trainable geometry. Consequences:
  - Take the G2/G3/G4 golden data from the master the branch is currently rebased on, and
    recapture it after rebasing onto the batch-4 merge.
  - G1 is unaffected, because both arms run in the same tree.
  - Scenes that set an explicit subpixel mode keep `torch.equal` goldens across the merge.
  - The coverage line sends its materials.py hunks before its merge note.
  - Hunks received (3434e87b, not yet merged):
    - `_resolve_subpixel` (~L336): `subpixel=None` now resolves through `SubpixelSpec()`.
    - `_compile_kottke_edge_component` (~L1280): returns early when the interface band is
      empty.
    - `compile_material_model` (~L2826): falls back to `scene.subpixel` when no subpixel
      spec is given.
    - O3's GPU Kottke path must reproduce the empty-band early return and the resolved
      spec. Its call site stays clear of these three spots and of the trainable-geometry
      guard.
    - 6e90fe55: `compile_material_model` now starts with the coordinate-chart
      `validate_bounds` check. Scenes with a coordinate chart (BendCoordinates) are not
      core-eligible. O3's call site goes after this check.
- **excitation/ is shared with gpu-memory-cuda-attribution,** which is fixing
  `injection.py` (custom field source) and the modulation coefficient in
  `fdtd/runtime/materials.py`. Notify that line before touching anything in `excitation/`.
- **The O2 DFT epilogue must preserve the pulse-DFT semantics:** the per-bin dt·f scale,
  the averaged phasor for all-ongoing sources, and the fail-closed rule for mixed sources.
  The core scene set must therefore contain a pulsed case and an all-CW case.
- **stepping.py and PCB (coax-port-bitwise-baseline, answered 2026-11-01):** D3b does not
  edit stepping.py. The hook goes at the `update_magnetic_fields`/`update_electric_fields`
  entry (~314–336) plus the graph-loop entry. Stay away from the PCB code at
  ~1740–1769 (copper/sheet init), `_field_update_block` (~1859–1939) and `_port_step_block`
  (~2123).
  - The core-eligible predicate excludes ports, CopperFoil, thin sheets, cut surfaces,
    lumped elements, wires and circuits.
  - The non-eligible path stays byte-identical.
  - O1 adds a test asserting that a copper scene and a thin-sheet scene take the existing
    path.
  - Send the stepping.py hunk to PCB for ack before merge.
- **GPU 1 standing slot:**
  - ≤ 4 GiB per process, one process at a time.
  - While batch 3 runs, require ≥ 26 GiB free before each run and never take the cuda1b
    lock.
  - G5 exclusive windows are booked by the coordinator and serialized with PCB D3b E5.
  - No cloud plan for now.
- **GPU 1 exclusive (user, 2026-11-01; allocation agreed with the coordinator):** from the end of
  batch 3 (ETA ~00:30 PST, the coordinator announces it), GPU 1 belongs to this line alone. Vacate
  it fully for exactly two booked exceptions:
  - the PCB D3b E5 final timing, one ~20–30 min window with ≥ 15 min notice;
  - integration-batch milestones such as batch 4, for their whole duration (several hours),
    with advance notice.
  Before that, the ≤ 4 GiB no-lock slot applies.

- **O2 file map (gpu-memory-cuda-attribution, 2026-11-01):**
  - `temporal.py` and `sources.py` are clear.
  - Avoid the injection.py custom-field helpers: `_prepare_custom_field_source`,
    `_axis_index_window`, `_trilinear_sample` and `_axis_interp_weights`.
  - fix/modulation-clock (native ABI v7, after batch 3) adds a one-line clock counter in
    `_field_update_block`. Any O2 hunk near it rebases onto that merge first.
  - The next native ABI bump is v8; it does not affect `core_abi_v1`.

### O1 implementation notes

- The opt-in `fdtd_core_path` option adds no solver state on default runs. Resume
  fingerprints hash explicit physical inputs and coefficient tensors, not this
  execution option; `fdtd/resume.py` is unchanged.
- The separate `core_abi_v1` extension builds under the configured build root's
  `core/` directory. Launch objects retain tensors and prepare pointer arguments
  once, before graph capture. Each half-step makes one extension call and one
  kernel launch; the two stepping entry hooks are the only stepping changes.
- Threads cover the union of the three staggered component shapes. Cross-field
  center loads are shared in registers; each directional neighbor is read once.
  No shared-memory tile is introduced without measured evidence of a benefit.
- A nonuniform half uses one padded per-cell index into six-float tuples
  `(x_decay, x_curl, y_decay, y_curl, z_decay, z_curl)`. Padding covers only absent
  staggered faces. GPU `torch.unique(..., dim=0, return_inverse=True)` runs on
  int32 views of the stacked float32 tuples to preserve every coefficient bit,
  including signed zeros; indices use uint16 up to 65535 tuples,
  uint32 above that. Uniform halves keep six scalars and no per-cell index.
- Dense CPML uses grouped corrected derivatives. Slab E preserves the existing
  flat expression; slab H preserves the grouped form. All coefficient products,
  including decay factors equal to one, remain in the kernel expression.
- Admission reads initialized physics flags and declared scene objects. Early
  admission also rejects unsupported execution before allocating a distributed
  solver or entering a port-specific preparation path. Batch is detected from
  its existing simulation marker before it expands field storage into lanes.
- Targeted coverage lives in `tests/fdtd/core_path/test_core_path.py`; all GPU
  workers run serially in fresh interpreters. The existing small copper fixture
  lives under `tests/materials/surface_impedance/` in this revision, rather than
  the task's older `tests/materials/copper_foil/` location.
- Verification (2026-11-01): the separate extension builds cleanly with `-O3`,
  `sm_86` SASS and `compute_86` PTX under
  `<evidence-root>/core-cuda-gates-o1/build/core`, with marker `core_abi_v1`. The existing
  extension and all its kernel sources remain unchanged.
- G1: all 22 `CORE_SCENES`, 256 steps, fresh interpreters for each arm;
  4661/4661 captured items satisfy `torch.equal`, with zero absolute and relative
  differences. Final candidates were recaptured after the review edits against
  the unchanged baseline path. Receipts:
  `<evidence-root>/core-cuda-gates-o1/g1/comparison.json`, `summary.json`, and
  `final-*.compare.log` under the same artifact root.
- Targeted tests: 5 passed in 49.13 seconds on the final source, including both
  64-cubed, 64-step equality cases and active graph capture. Log:
  `<evidence-root>/core-cuda-gates-o1/tests-final.log`. Source/JSON/TOML checks and exact
  verification of the two additive stepping hooks also pass; both materials
  modules and `fdtd/resume.py` remain byte-identical to the parent revision.
- GPU 1 only, one owned GPU worker at a time, with a free-memory check before
  every worker. Final G1 workers peak at a sampled 478 MiB; targeted test workers
  peak at 420 MiB. All memory checks pass; per-PID receipts are the matching
  `*.log.memory.json` files. No GPU lock files or cloud resources are used.
- SASS inspection was unavailable: the installed toolkit has no `cuobjdump` or
  `nvdisasm`. Arithmetic expression trees, operand order and coefficient products
  are retained; exact G1 is the numerical evidence. Performance is not measured
  in this task. The stepping hunk still requires PCB acknowledgement before a
  coordinator merge.

## 6. Resources

- **GPU.** Correctness gates fit in ≤ 4 GiB per process on the shared GPU, under a lock and
  with the coordinator's OK. G5 timing needs **exclusive** windows; contended numbers are
  invalid (see the 2026-10-31 copper 3.3 ms correction). These come from coordinator-booked
  local windows, or from a cloud A6000 through `wcloud plan`. Every cloud plan needs the
  user's approval.
- **Build.** Use a private `WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR` and `CUDA_HOME=<sp>/nvidia/cu13`.
  Never set the PREBUILT flag.
- **Implementation.** Codex in this worktree (or sub-worktrees per phase). Claude writes the
  specs, reviews and runs the gates.

### O2 design

Design-only audit of `e577cc4c` (2026-11-01); no GPU work. Package paths start at
`witwin/maxwell/`; repeated basenames abbreviate prior citations. `E` marks estimates.
O2 adds a whole-step capability check within O1 admission; unsupported execution keeps O1.

**Present sequence and launch accounting.** Counts assume the O1 harness configuration:
core enabled, graphs successfully captured, one frequency, full-field Ex/Ey/Ez DFT,
fp32 observers, no shutoff/plotting (`benchmark/core_path/gates.py:44-51`). Graph replay
reduces host submissions, not kernel count. Counts exclude setup, capture and export.

| Case | Ordered work each step | GPU work / host submissions |
|---|---|---|
| `cpml_compressed` | Host `n`, `n*dt`; field graph: fused H, fused E; eager Ez patch; tail graph: two DFT row gathers, two copies, batched DFT, counter increment; observer graph: weights, 13 gathers, counter increment | 22 kernels + 2 device copies; 3 graph replays + 1 eager patch submission |
| `planewave_vacuum` | Host `n`, `n*dt`; field graph: fused H, batched TFSF H correction, auxiliary H, fused E; eager batched TFSF E correction, auxiliary E with host waveform; same tail; observer graph: weights, 24 gathers, counter increment | 36 kernels + 2 device copies; 3 graph replays + 2 eager submissions |

- Evidence for ordering: `fdtd/runtime/stepping.py:1881-1947,2510-2574`; O1 dispatch
  `:314-317,822-825` and `fdtd/runtime/core_path.py:124-162`. CPML psi is already in
  the two fused kernels. Material/port/network calls are no-ops here; port graphs
  require runtimes (`stepping.py:2164-2177`). No PEC clamps launch (`:1537-1539`).
- One Ez dipole patch (`benchmark/core_path/scenes.py:49-66,122-131`;
  `fdtd/excitation/injection.py:459-507`) evaluates the pulse on the host, multiplies
  by `phase_real`, and submits one patch (`fdtd/excitation/temporal.py:501-532`).
- The plane wave is +z/Ex with a pulse (`benchmark/scenes/planewave/planewave_vacuum.py:12-27`).
  Axis-aligned state packs one correction batch per half (`fdtd/excitation/tfsf_state.py:141-148,392`);
  each batch launches once (`fdtd/excitation/tfsf_apply.py:45-67,136-154`). Only the
  auxiliary E advance evaluates a host waveform and increments a Python clock
  (`fdtd/excitation/spatial.py:436-458`); the H correction/advance is graphable already
  (`stepping.py:1971-1988,2020-2056`). The E correction precedes auxiliary E advancement.
- DFT tail is **four kernels plus two copies**, not six native spectral launches:
  `fdtd/runtime/spectral.py:612-637` has two `index_select`, two contiguous `copy_`,
  one native accumulation and `add_(1)`. Native single-frequency dispatch is one
  kernel (`fdtd/cuda/kernels/spectral.cu:208-223`). Copies are logical operations;
  lowering is runtime-dependent. Tail capture requires GPU DFT (`stepping.py:2073-2126`).
- Observer counts are resolved groups, not monitor count: point keys are component,
  plane keys `(component, axis, plane_index)` (`fdtd/observers.py:488-535,583-641`).
  The 64-node dipole grid has two point groups, five y=0 E plane groups (2/1/2),
  and six flux groups (two E planes + four H planes): G=13. The benchmark has 53
  physical nodes + 16 PML nodes (`scene.py:1613-1615,1649-1660`): three plot planes
  give 12 groups (4/axis), the main y=0 plane shares those, and two flux planes give
  12 more: G=24 (`benchmark/scenes/_common.py:6-8,58-69`). Flux snaps to primal E
  nodes; H brackets them (`fdtd/observers.py:307-413`; `fdtd/coords.py:15-26`).
  Engine advance is 1 weight + G gather + 1 counter kernels, separately captured
  (`fdtd/observers.py:898-958`; `fdtd/cuda/kernels/observers.cu:9-54`); raw Yee
  phasors are interpolated and flux reduced only on export.
- Without graphs, observer window/phase work is on the host with G kernels
  (`observers.py:690-785`); field/source/tail ordering persists.
  Thus these rows have 20/34 kernels + two copies eagerly. Capture failure can instead
  leave the grouped engine eager (G+2 kernels), so report active graph flags in receipts.

**Device waveform and scalar contract.** Prepare from compiled patches after dt/horizon
resolution (`simulation.py:1860-1867`). A persistent `int64[1]` addresses tables;
retain term order, offsets, clipping/wrap and H/E timing.

- Uniform dipoles: tabulate exactly `evaluate_source_time(term_time, n*dt)` followed
  by the existing double `phase_real` multiplication and final float32 conversion.
  CW phased patches tabulate the separate NumPy cosine/sine expressions in
  `temporal.py:79-82,142-153`; auxiliary hard seeds use their own `aux.source_time`,
  phase and `aux.time_step*aux.dt` (`spatial.py:361-362,436-445`; `tfsf_state.py:58-80`).
- Exactness qualification: these scalar paths use Python/libm or NumPy double
  expressions (`sources.py:788-823`; `waveforms.py:41-67,311-326`), unlike delayed
  CUDA pulses. GPU `sin/exp` with the same algebra does **not** guarantee the same bits.
  G1 requires staging exact scalar results once at prepare and uploading GPU-resident
  tables, removing per-step host evaluation. Strict GPU-only generation is a candidate
  only after every rounded row passes equality; it cannot silently relax G1.
- Delayed soft PlaneWave/GaussianBeam terms: precompute amplitude tables **on GPU**,
  per term and unique delay, using the existing fp32 constants/conversion order,
  `time-delay_patch`, causal comparison, and device waveform expression
  (`fdtd/cuda/kernels/sources.cu:154-190,315-359,1170-1197`). Upload exact
  `float32(n*dt)` and `float32(n*dt+0.5*dt)` time rows; do not form them by fp32
  recurrence. Preserve patch multiplication/contraction. Storage E: 4*S*D bytes
  (S steps, D unique delay samples); cap allocation, using device-indexed time rows
  and the identical inline evaluator when a full amplitude table is too large.
- Soft waves/beams are **two-sided currents**, not just `_source_terms`:
  `injection.py:1138-1186,1424-1445` fills `_magnetic_source_terms` before E and
  `_electric_source_terms` after E; H samples n*dt, E samples n*dt+dt/2
  (`stepping.py:1900-1906,2524-2534`). Delays/beam phase stay in compiled patches.
  TFSF consumes live auxiliary arrays or tabulated analytic signals
  (`tfsf_apply.py:136-173`); preserve oblique overlap exclusion (`:176-191`).
- Host per-step inventory today: n/steps_run/time_value; half-step time; source
  cache keys and evaluated signal/phase-scaled signal or cos/sin pair; delayed `time`
  (and custom-table `timeOffset`, outside O2); auxiliary `time_step`, electric_time
  and sourceValue. Repeated offsets, source parameters, sourceIndex, sample indices,
  origin/ds and scales are static launch arguments, not evolving state. Observer
  fallback also computes active/start/end tests, window, double phase recurrence,
  H rotation, counts/weight sums and optional source spectrum (`observers.py:690-785`).
  GPU DFT/grouped graphs remove host spectral scalars. Python selects blocks;
  shutoff computes energy/peak, interval/floor tests and a synchronous bool; progress/plotting use n and
  sample counters (`stepping.py:2510-2597`). No port scalar work belongs to O2.

**Fusion and numerical gates.** Tables/index/graphs require G1. Preserve store/reload rounding.

- Fuse additive E patches after the completed O1 E value, then perform the same final
  PEC clamp before any DFT. Retain term order for overlaps and periodic copies;
  keep the rounded E result and the original multiply/add contraction (unwrapped
  patches can contract, periodic deltas are separately rounded; `sources.cu:248-310,446-499`).
  Keep sources separate from curl arithmetic. Target G1; pre-name
  **G1′-source-contraction** only for a selected rounding change passing §4 bounds.
  Process sources/sampling even on cells skipped by O1's early returns (`yee.cu:90-97`).
- Soft currents need H injection **before** E. Retain TFSF H correction -> auxiliary H -> E
  and E correction -> auxiliary E dependencies. Initially capture both corrections
  unchanged. Folding TFSF atomic/warp face sums into a cell owner changes their
  reduction tree (`sources.cu:805-925`): pre-name **G1′-TFSF-face-reduction**, separate
  from ordinary patches; defer unless its measured launch saving warrants it.
- DFT epilogues can give each field cell ownership of all its point/plane accumulator
  destinations (including duplicates), in the existing dtype, with one update per
  sample per step and chronological order. Use the rounded final field and original
  multiply-add form (`spectral.cu:9-175`; `observers.cu:41-54`); no temporal reduction,
  spatial averaging/interpolation. E DFT follows sources/clamps; H follows corrections with its retard
  (`fdtd/runtime/spectral.py:228-251`). Cross-thread neighbor reads of E being
  updated are forbidden; use prepared routes from field cells to raw Yee samples.
  Correct placement/rounding targets G1; pre-name **G1′-DFT-contraction** for a
  deliberately changed contraction. Preserve fp64 or retain gathers. TFSF destinations
  retain post-correction sampling kernels until their correction epilogues qualify.
- Pulse semantics are mandatory, not G1′ candidates: exp(+iwt) sums, start=0/window=none
  for transients (`spectral.py:266-286`), output scale dt*f **per bin**, or explicit
  reference frequency; all-ongoing sources use factor/weight_sum (real factor=2),
  not dt*f (`fdtd/runtime/representation.py:32-46`; `timeplan.py:81-86`). Preserve
  source normalization's unit envelope, physical amplitude/phase, and separate
  E/H weights. Keep mixed ongoing/transient absolute spectra fail-closed while
  allowing existing ratios (`representation.py:10-18`; `result.py:1512-1515`).
  Early stop retains the planned CW denominator, never rescales a pulse
  (`stepping.py:1844-1862`). DFT decimation is deferred; no skipped samples in O2.

**K-step graph.** Specialize a prepared whole-step closure once: H -> H corrections/
auxiliary H -> E -> E corrections/auxiliary E -> additive sources -> clamps -> DFT/
observers -> one device step commit. Capture K repetitions on the host **once**, then
one replay per K steps. A separate commit kernel prevents a counter race with E blocks
still reading it. Boundaries provide global ordering. Use direct table[counter] reads. Keep observer
weight/state kernel initially; later prepare identical E/H weight rows with that same
double recurrence and finalization state, avoiding a new trigonometric recurrence.
One canonical counter replaces independent DFT/observer clocks; export restores their
clocks/actual observer counts (`spectral.py:520-608`; `observers.py:906-914`).

Chunk only up to the next shutoff check, final/stop step, progress or plot boundary;
use a K=1/remainder graph when necessary. Evaluate the existing energy/peak expression
and bool at the **same** steps, outside capture (`stepping.py:2576-2586`), so no extra
steps execute after a stop. Reject cooperative/batch/adjoint and unknown callable
waveforms for whole-step capture; keep their current execution. Resume/stop must use
absolute step rows/restored auxiliary clocks, or retain O1 until qualified.
Snapshot/restore fields, psi, auxiliary E/H, observer state/weights/accumulators,
DFT buffers and every counter on warmup/capture success **and** failure. Tables are
immutable. Restore after each warmup; capture with zero automatic warmups so short
horizons stay in bounds (`fdtd/cuda/runtime/graph.py:125-161`). The driven zero-field
state is not a fixed point. Preallocate/build; honor capture suspension and fallback.

**Exact proposed edit surface / ownership.** Existing files: `fdtd/runtime/core_path.py`
(`CorePathRuntime.__init__`, `activate_core_path`; add `prepare_step`, `capture_chunk`,
`run_chunks`); `fdtd/cuda/core/yee.cuh` (`CoreHalf` epilogue descriptors), `yee.cu`
(`component`, `yee`, `launch_core`), `binding.cpp` (`CoreLaunch`, new `CoreStepLaunch`
launch bindings), `build.py` (new sources/private ABI version); `fdtd/runtime/stepping.py`
(one loop dispatch near :2506, returning steps_run and skipping the old iterator;
reuse common preparation/finalization). Charge unused old graph setup initially. New:
`fdtd/runtime/core_tables.py` (`compile_source_tables`, `compile_spectral_tables`,
`compile_monitor_routes`), `fdtd/cuda/core/step.cuh` and `step.cu` (new CUDA functions
`compile_delayed_amplitudes`, `source_patch_indexed`, `auxiliary_electric_indexed`,
`dft_indexed`, `observer_gather_indexed`, `commit_step`). Tests extend
`tests/fdtd/core_path/test_core_path.py`, `benchmark/core_path/scenes.py`, `gates.py`;
update `FEATURE_LIST.md` and this plan for implemented behavior.
No `excitation/` edits: consume compiled records/auxiliary arrays. Shared ownership:
`temporal.py`, `spatial.py`, `tfsf_apply.py` are reads only;
any later edit requires notifying gpu-memory-cuda-attribution first. **injection.py
and the custom-field path are off limits.** No edits to stable CUDA sources/extension,
`observers.py`, `runtime/spectral.py`, `timeplan.py`, `representation.py`, or result
normalization are needed. Do not touch PCB `stepping.py:1740-1769`,
`_field_update_block:1865-1947`, `_port_step_block:2129-2147`; send the single loop hook
for PCB acknowledgement before merge. Keep non-core execution byte-identical.

**E gains and implementation order (three tasks).** Let N be cells, P injected field
samples, Q gathered Yee samples and F bins. O1 costs B1*N bytes/step (use measured B1,
not the old 144 B floor). Graphs change no field traffic. Ordinary patch fusion saves
about 8*P bytes and its launch; sparse dipoles are launch savings, not a bulk-byte win.
DFT fusion saves 4*Q bytes, retaining 16*Q*F accumulator bytes (fp32): one-bin saving
is at most 20% of DFT traffic. Full Ex/Ey/Ez DFT costs 60*N bytes at F=1; 12*N is removable.
For B1=72-96, E bandwidth-only speedup is (B1+60)/(B1+48)=1.08-1.10; plane-only
savings scale as N^(2/3), not N. Routing/table reads and occupancy can reduce these.

1. **O2a exact tables + graph:** implement device-indexed sources, direct DFT row reads,
   single counter and K=16/32 chunk replay, keeping corrections and observer gathers.
   E: 4/5 submissions per step -> 1/K; remove two DFT gathers, two copies and one
   counter: 19/33 kernels, plus no host waveform/dictionary work. Target G5 >=1.2x
   at 64³, <=3% regression at 256³/384³; launch counts alone are not timing evidence.
   Gate: table-row G4 equality; G1 fields/phasors/flux at K=1/16/32, short/remainder/
   zero runs, nonzero initial fields, pulse+all-CW, multi-bin and overlapping sources,
   soft/TFSF/beam, early shutoff and capture failure; untouched PCB paths checked.
2. **O2b patch epilogues:** fuse ordered ordinary E/H patches, preserve clamps and
   periodic writes; retain TFSF correction kernels. E: saves 1 launch/step in the
   dipole row and one per soft-current term, plus 8*P bytes; likely small after O2a.
   Gate: G1/G4, overlapping/boundary-source equality, G2/G3, G5 no regression >3%.
3. **O2c spectral epilogues:** fuse eligible point/plane/full-field accumulation and
   prepare exact observer weight rows; retain separate gathers where ownership or
   precision is unqualified. E: removes up to 13/24 gathers plus full-field DFT/weights;
   full fusion gives dipole H+E+commit (3 kernels).
   TFSF retains 7 physics/counter kernels plus post-correction sampling. Target incremental
   overhead reduction >=50% on small plane-monitor cases; bandwidth ceiling above
   forbids promising 50% for large full-field DFT. Gate: G1 unless a named G1′ variant
   is explicitly selected, then §4 bounds; multi-bin pulse dt*f/CW/mixed-source and
   source-normalization tests, G2/G3/G4 and exclusive G5 byte/launch receipts. All
   performance targets are E, to be accepted only with booked exclusive-GPU evidence.

### O2a implementation notes

- Whole-step replay is internal to `fdtd_core_path=True` and graph-enabled forward
  runs. `CorePathRuntime.chunk_size` defaults to 32; the gate harness selects 1, 16
  or 32 without adding a public simulation flag. Zero-step execution captures no graph.
- The zero-step runtime gate uses a one-step plan with `stop_step=0`: the existing
  checkpoint boundary skips export when configured monitors have no samples.
- Scalar dipole, CW pair and auxiliary hard-seed rows are evaluated once on the host
  at preparation with the existing double expressions. Both exact fp32 time rows are
  uploaded independently. Delayed analytic amplitudes are evaluated on CUDA per unique
  delay bit pattern with the existing fp32 expression order. The aggregate amplitude
  allocation budget is 64 MiB; larger terms read the exact time row and evaluate inline.
- Source-table sample instants now use the shared `injection_time` helper: ordinary
  electric currents use `electric_current`, magnetic terms use `magnetic_current`,
  and electric surface/TFSF terms use `equivalent_electric_current`, matching the
  main loop. Auxiliary hard-seed timing remains `(aux.time_step+n)*aux.dt`.
  CPU-only checks verify bitwise fp32 time rows and host sample instants for dipole,
  soft plane-wave and auxiliary/analytic TFSF scenes over 257 steps and four dt values;
  `py_compile` passes. GPU requalification remains with the coordinator.
- Observer weights and gathers and reference auxiliary TFSF corrections retain their
  existing kernels. The direct DFT kernel reads immutable rows through the same device
  counter used by sources and observer weights; a separate kernel commits each step.
- Observer configurations that use the existing host recurrence retain O1. Whole-step
  replay reuses an already prepared grouped observer engine without changing routes.
- Capture uses one explicit warmup, restoring after it and after each capture, with
  zero automatic warmups. Capture failure restores physical fields, CPML, auxiliary
  fields, spectral buffers, observer state/weights and counters before retaining O1.
  The restore snapshot is cleared in `finally` before chunk replay or failure return.
- Chunk scheduling stops at shutoff checks, progress updates, plot samples and the
  requested end step. Cooperative execution, resume, callable waveforms, sampled
  delayed custom waveforms and unqualified oblique auxiliary reductions retain O1.
- Extending a prepared horizon beyond its immutable source rows retains O1. The G2
  harness rebuilds after the benchmark's one-step dt probe resolves the final horizon,
  so validation runs use tables uploaded once per preparation at their final horizon.
- Current scene inspection gives 31 kernels per `planewave_vacuum` step: two Yee,
  two soft-current patches, one DFT, one observer-weight kernel, 24 gathers and one
  commit. This scene uses soft injection in this tree; the older design accounting
  above described four TFSF correction/auxiliary launches instead of two soft patches.
- `cpml_compressed` has 19 kernels per step: two Yee, one dipole patch, one DFT,
  one observer-weight kernel, 13 gathers and one commit. Each complete chunk uses
  one graph submission; these counts are code/graph inspection, not timing results.
- Qualification receipts are under `<evidence-root>/core-cuda-gates-o2a/`: G1 passes all 36
  scenes at each K=1/16/32 (7,458 exact leaves per K); G4 passes 36 scenes and
  35,941 exact setup/table leaves; G2 preserves all 11 scenes' outputs, numerical
  scores and statuses exactly (6,940 leaves, including existing exclusions).
- Whole-step capture is active for 34/36 G1 scenes at each selected K; zero-step
  and forced-failure cases report inactive/K=None. All 22 candidate G2 physical
  runs report active/K=32. Ten targeted tests pass, including the copper/sheet
  blockers, default admission, K chunks, capture restore and host-observer admission.
- Per-process GPU sampling peaks at 674 MiB, with no memory or overlap violations.
  No G5 or standalone timing run was performed; no performance claim is made.

### O3 design

Read-only audit of `c8daca2f` on batch-4 master `04ae392e` (2026-11-02).
Paths below are relative to `witwin/maxwell/`, except `core:` denotes installed
`witwin/core/geometry/`. `E` means an estimate, not a measured speedup.
Receipts/scripts: `<evidence-root>/core-cuda-o3-design/` (copied by the
coordinator session from the task's /tmp working directory).
No library code changes or numerical qualification are part of this task.

**Measurements and limits.** Fresh GPU-1 workers, private build root, fp32,
7.25 GiB allocator cap reserving context space beneath the 8 GiB process limit.
Profiler prepare seconds: vacuum256 0.831; sphere256 polarized3 8.247;
sphere+box256 edge4 12.286 / smooth4 11.342. Peaks: 4.89 / 5.58 / 5.58 / 5.58 GiB
allocated; highest reservation 6.52 GiB. Both unmodified 384³ prepares hit the cap
at 7.47 GiB process use inside node compilation; two live node+inverse sets alone
need 8.86 GiB. Their full reruns cannot satisfy the cap. Supplied exclusive G5
baselines remain 384³ vacuum material 1.971 s and sphere polarized3 23.4 s.
Vacuum256 material construction is 0.667 s: zero SDF work, but 27 sample models,
six component loops and gradient reconstruction; exclusive device costs include
0.179 s node accumulation, 0.111 s model allocation and 0.188 s edge loops.
This repeated dense traffic explains the 2 s vacuum cost; coefficient formulas
cost only 2.94 ms at 256³. The first 64³ probe includes a 58 s cold extension build.
These are profiling receipts, **not G5 qualification**: another worker appeared
during smooth-profile CPU result processing. Further GPU work is held pending
serialization; polarized4, mesh128/256, boolean and core-table profiles remain
unmeasured. `summary.json` records these gaps rather than extrapolating measurements.

Top ten measured **self device** regions, maximum per function across these cases;
S=sphere256 polarized3, K=pair256 edge4. Counts are profiler-linked kernel/copy
events (not Python calls); nested inclusive timings are separate in JSON. E targets
below share fused work and must not be added. Mesh/boolean/table rankings are open.

| Region (file:line) | Case | ms / events | Replacement; local E ms |
|---|---|---:|---|
| `core:primitives.py:128` Sphere SDF | S | 3103.8 / 13233 | Per-query fused SDF; 150-500 |
| `compiler/materials.py:326` blend | K | 2630.5 / 9226 | Register material/channel blends; 80-250 |
| `core:primitives.py:93` Box SDF | K | 2434.9 / 6154 | Fused local-frame distance; 100-300 |
| `compiler/subpixel.py:237` coverage | S | 2316.8 / 10217 | Fused seven-query ramp/curvature; 120-400 |
| `core:base.py:164` rotate | K | 2187.7 / 22691 | Descriptor matrix, identical scalar products; 80-250 |
| `compiler/materials.py:731` structure apply | K | 569.4 / 2310 | One ordered cell owner/channel pass; 30-100 |
| `compiler/materials.py:2852` node orchestration | K | 401.0 / 1252 | Register chronological sample sums; 40-120 |
| `compiler/subpixel.py:199` fill dispatch | S | 290.3 / 1392 | Reuse rounded queries/bin candidates; 20-70 |
| `compiler/materials.py:240` model allocation | K | 252.2 / 1952 | Allocate final fields once, fused fill; 30-100 |
| `compiler/materials.py:1303` edge orchestration | S | 191.3 / 1671 | Native component loops in one kernel; 20-70 |

**Data and exact arithmetic contract.** Compile structure descriptors once in
`_sorted_structures` order (priority, then insertion order), retaining the original
boolean expression tree, transforms and periodic-image order. Use fp32 structure
SoA plus int32 coarse-bin CSR `(offsets, structure_ids)`; bins cover the full rounded
subsample/ramp/stencil reach, including periodic images. Preserve sorted candidate
order and later-structure priority; unknown/unbounded geometry stays in a global
candidate list. A geometric AABB alone cannot prune a nonzero smooth occupancy.
Skip a candidate only after proving the current fill rounds to exactly zero for
the whole window. For smooth booleans expand bounds by their blending reach.

Flatten each native node/Ex/Ey/Ez/Hx/Hy/Hz grid with z contiguous; keep component
shapes and output tensors unchanged. Upload/reuse the current 1D rounded bases,
offset arrays, primal/dual widths and scalar-spacing decisions. Never reconstruct
query coordinates by a new `origin + i*dx` formula; custom/graded axes are first
class. Fused per-cell loops process all candidate structures for each subsample,
then accumulate samples in `product(x,y,z)` order with z fastest. Retain seven
SDF evaluations for curvature coverage, rather than replacing the current finite
difference normals by analytic normals. Share repeated query work, not algebra.

- Node `_compile_material_model:2917-2985`: start each accumulator at zero,
  round every sample addition, multiply once by the same fp32-converted Python
  `1/len(sample_offsets)`. Edge `_compile_edge_component:1500-1525`: initialize
  sums from the first sample, then add chronologically, then scale. Reciprocal
  material values use the current `1/(value+1e-12)` conversion order. A uniform
  fast path must reproduce these sums (including reciprocal noise), not simply
  substitute the declared material. Preserve canonical node summaries/aliases,
  zero channels, flags and metadata, even if stepping consumes only edge fields.
- `subpixel.py:199-257`: preserve +/-delta coordinate rounding, gx/gy/gz,
  norm, projected width/max, Python-sum Laplacian order, curvature clamp and
  strict snap comparisons (`fill>1-1e-3`, `fill<1e-3`). Preserve `torch.gradient`
  nonuniform coefficients and boundary stencils in a separate neighbor pass.
  Node polarized reconstruction sums squared gradients by component then axis;
  edge reconstruction uses `(gx²+gy²)+gz²+1e-24`, then the existing inverse blend
  with the `<1e-5` contrast threshold (`subpixel.py:366-399`).
- `_compile_kottke_edge_component:1270-1300`: keep bulk ordered `where(sdf<0)`
  or smooth sequential blends; band is `.5*vector_norm(widths)+1e-6`, with strict
  `abs(sdf)<band`. Emit ascending near indices and the same 4096-cell chunking.
  Empty near indices return the bulk material **before** quadrature/moment work;
  all-background entries retain the earlier shortcut at :1207. H bulk shortcuts
  must also preserve which work is skipped. Smooth chunk spacing uses the exact
  current selected-width minimum; changing chunks can change graded-grid results.
- `subpixel.py:402-412`: retain separate arithmetic mean, reciprocal mean,
  `(sample-real_mean)*local_offset` moment, norm, conditional squared denominator,
  squared-axis/norm weight, and `weight*harmonic+(1-weight)*arithmetic` order.
  Do not replace this by the polarized inverse blend. Match the installed ATen
  reduction tree for 27/64 samples, including strided moment reductions; initially
  keep these reductions in torch on packed interface tiles while fusing SDF/fill.
  Node `kottke_edge`/`kottke_smooth` use the polarized inverse reconstruction
  (:3001), not window moments; `A+(H-A)*n²`/offdiagonal projection is the separate
  `polarized_tensor` branch, which remains outside initial O3 eligibility.
- CUDA uses round-to-nearest fp32 operations with contraction disabled where eager
  torch stored intermediates (`--fmad=false`, no fast math; explicit rounded
  operations if needed). Preserve PyTorch reciprocal/sqrt/divide implementations,
  thresholds, NaN behavior and signed zeros. `torch.compile` fusion alone is no
  bit-identity guarantee. Qualify intermediate fill/means/normals as well as final
  G4 tensors; retain the torch primitive/reduction whenever exact emulation fails.

**Mesh acceleration and its limits.** Register a lazy maxwell-owned object through
`core:mesh_sdf.py:26 _register_mesh_sdf_accelerator`; implement `supports`,
`is_available`, `unsigned_distance`, `winding_angle`, `smooth_signed_distance`,
`unsigned_distance_static_bvh`, `signed_distance_static_bvh`. Require CUDA/fp32,
fixed geometry and non-grad points/triangles; otherwise the existing torch path
stays active. Registration is once, but `supports` checks the scoped O3 opt-in.
Do not install/uninstall a process-global hook around each prepare (thread race).

The hook consumes existing CPU-authored BVH buffers when provided; CUDA-authored
meshes need a device LBVH builder (Morton keys + stable triangle-ID ties), cached
by device, dtype, storage/version and transform state, without device-to-host
geometry copies. Store node AABBs in fp32 SoA, children/ranges/triangle IDs int32,
and original-order triangles `[T,3,3]`; preserve core topology-staleness checks.
Distance traversal uses conservative outward-rounded AABB bounds; visit equality
ties, retain original triangle IDs, and reproduce all point-triangle predicates,
`finfo.eps` denominator clamps and `sqrt(max(d²,0)+1e-12)-1e-6`. A mathematically
improved distance would change today's result and is outside O3. Small meshes
(the measured pyramid has six triangles) use a fused direct triangle evaluator;
their timing cannot establish a 5k-triangle BVH speedup.

Sign is **smooth winding**, `-tanh((abs(angle)-2*pi)/.5)`, multiplied by unsigned
distance (`core:mesh_sdf.py:180-190,270-317`). Ray parity, nearest-normal sign,
hierarchical approximate winding and dropping far triangles are not equivalent.
Initial exact path retains torch winding with original triangle order, 256-triangle
chunks, same sum tree and chronological chunk accumulation; later fuse it only
after exact qualification. BVH prunes distance, not winding. Thus exact signing
still has O(points*triangles) work; no unconditional "39 s -> a few seconds" claim.

Tie suite includes face/edge/vertex queries (vertex denominator substituted by one),
shared/coplanar faces, opposite/duplicate or reversed triangles, zero-area/sliver
triangles, medial equal minima, `abs(angle)==2*pi`, +/-zero, open/nonmanifold or
uncertified meshes and AABB boundaries. Import the two `a302ac9` strict-xfail
surface cases as **current-output** goldens; do not fix their known distance error
in O3. Preserve strict `sdf<0` for hard windows versus smooth occupancy semantics,
surface-band behavior, and cached coverage's bit-pattern coordinate keys/budget.

Pre-name **G1′-O3-winding-tree** only as a deferred coordinator decision: changing
winding reduction/transcendentals can change the sign near threshold, so generic
G1′ field bounds and a raw-distance 1-ULP allowance do not justify it. A candidate
must additionally retain every occupancy/sign and pass exact G4 coefficients;
otherwise it needs an explicit new setup-gate exception, not silent admission.
No bounded item is selected by this design. Analytic fusion and all three named
subpixel modes remain under exact G4. No core edits are required by the hook;
GPU-BVH creation/cache belongs to maxwell, not `Mesh._sdf_query_data`.

**Integration and files.** Add one dispatch inside `compile_material_model`'s
existing `_mesh_fill_cache` scope, immediately before `_compile_material_model`
at `compiler/materials.py:2849`. It follows chart `validate_bounds` (:2832), resolved
scene subpixel fallback and **both** trainable guards (:2836-2847). The helper
returns a complete model or a sentinel for the unchanged path. Do not edit
`_resolve_subpixel`, batch-4 empty-band logic or guards. Reject coordinate charts,
partial selections/material-cache overrides and unqualified geometry/materials.
Recursive tensor inspection covers nested boolean children, transforms, mesh
vertices, material parameters and region density: trainable smooth/polarized
scenes keep native torch/autograd; trainable hard Kottke keeps its existing error.

Scope opt-in in `Simulation._build_fdtd_solver_for_scene` around `solver.init_field`
(:1860), using a ContextVar reset in `finally`; derive it from existing
`fdtd_core_path`, never a permanent Scene mutation. Apply eligibility again inside
the compiler. Add files `compiler/accel/{__init__,dispatch,geometry,subpixel,mesh,
coefficients,tables,build}.py` and `compiler/accel/cuda/{binding.cpp,setup.cu,mesh.cu,
bvh.cu,coefficients.cu,tables.cu}`; separate extension marker `setup_abi_v1`, build
under configured root `/setup`, independent of stable ABI and `core_abi_v1`.
Retain PyTorch tensors/current stream/device guards; no CPU voxelization fallback.

Coefficient fusion needs **one additional shared-file hook** beyond §3:
`fdtd/runtime/materials.py:1725 build_update_coefficients`, after physics,
dt and boundary profiles exist. Coordinator/owner agreement is required before
implementation (the modulation line owns nearby logic). Compose native staggered
output SoA in one kernel, retaining rounded `eps_r*eps0` / `mu_r*mu0` stores,
`half=((.5*sigma)*dt)/eps`, denom, decay, `(dt/eps)/denom`, then CPML multiplication
and PEC rules; retain `_store_coefficient_uniformity`/its aminmax tree initially,
including signed-zero scalar selection. Do not fuse across dt selection or checks.

Replace compression within `fdtd/runtime/core_path.py:117-136`: hash **all six
int32 coefficient bit patterns**, full-key equality for collisions, then compact
and lexicographically sort the K distinct signed-int32 rows, remapping inverse
indices. Preserve padding zeros, sorted row order, uint16/uint32 threshold and
uniform scalar bypass exactly; write no N-by-6 padded stack/sort workspace.
High-cardinality/overflow cases retain existing `torch.unique` until qualified.

**Implementation split and gains (E, to be measured under G5).**

1. **O3a orchestration/bulk:** scoped dispatch, bin lists, exact uniform/node loops,
   packed sphere/box/boolean coverage. Supplied 384³ vacuum material stage
   1.97 s -> E 0.15-0.40 s; sphere polarized3 23.4 s -> E 2-5 s after all O3b work,
   not a promise for this first task. Gate G4 on canonical node/edge metadata,
   custom/graded/periodic/odd grids, overlap priority, empty scenes; G1/G2/G3.
2. **O3b subpixel/coefficients/tables:** retain torch window reductions first, then
   emulate proven trees; fuse coefficient elementwise work and distinct-row table
   build. Edge4/smooth4 window spans 0.256/0.730 s -> E 0.08-0.20/0.15-0.35 s;
   their node sample loops cost 11.39/10.03 s, so O3a is the priority. Coefficient
   formulas 2.94 ms -> E 1-2 ms; table baseline/target awaits profiling, not guessed.
   Gate exact G4 for `kottke_edge4`, `kottke_smooth4`, `polarized3/4`, empty bands,
   trainable errors/gradients and invalid-chart errors; G1/G2/G3 before merge.
3. **O3c mesh:** implement hook, small direct evaluator, GPU LBVH and conservative
   distance pruning with unchanged winding. Prior 39 s/5.4M-cell mesh compile is
   a different workload, not a measured pyramid baseline. E: 2-5x on unsigned
   distance after BVH/fusion; signed-total gain depends on unchanged winding share.
   Gate raw SDF sign/distance (pre-named §4 allowance only), current-output tie
   goldens, boolean/mesh G4 coefficients exactly, G1/G2/G3. Benchmark a separate
   5k-triangle case before claiming BVH gains. All tasks recapture batch-4 master
   goldens after rebases; update `FEATURE_LIST.md` when behavior is implemented.

### O3a implementation notes

Implementation work on `937746bf`, rebased on `f928126d`; coordinator approval
`the coordinator` (2026-11-02) covers compiler hook (1) only. The only shared compiler
change is dispatch inside `_mesh_fill_cache`, after chart validation, resolved
subpixel defaults and both existing trainable guards. The single-device FDTD
builder sets a ContextVar from `fdtd_core_path`, excluding material-cache
overrides, around `solver.init_field` and resets it in `finally`. No Scene
attributes, modulation quadrature, guards, subpixel resolver or stepping CUDA
sources are changed.

The separate `compiler/accel` extension uses `setup_abi_v1`, the configured build
root's `setup/` directory, current CUDA streams/device guards, `--fmad=false`
and no fast math. Eligibility is checked again by the compiler. It excludes
coordinate charts, cropped selections, cache overrides, conductivity capture,
any modulated material (including disabled structures), recursively discovered
gradient-requiring tensors anywhere in the scene, region densities, meshes,
non-fp32 geometry/material sampling, and unqualified materials/subpixel modes.
The initial material envelope is fixed scalar, nondispersive, linear bulk media;
the qualified modes are edge4, smooth4, polarized3/4 and arithmetic1.

Descriptors retain `_bulk_structures`/`_sorted_structures` order and the complete
primitive/boolean tree, including local frames and smooth composition order.
The packed tree limit is sixteen nodes; larger trees retain torch. Sphere SDF,
box SDF/rotation, sharp and smooth union/intersection/difference, seven-query
coverage, ordered material blends and chronological node/ordinary-edge sample
loops run in CUDA. Rounded 1D queries and existing primal/dual sample spacings
come from torch helpers. Conservative 8-cell coarse-bin CSR lists retain ordered
candidates. Only native sphere coverage uses rounded-bin exclusion, with a
four-projected-width reach plus an outward rounding margin; other primitives
remain global candidates. Uniform/empty scenes use the same native sum loop,
including reciprocal sample accumulation noise.

Torch retains single-sample tanh coverage, periodic image/seam coverage,
graded-grid gradient reconstruction, canonical summary arithmetic, and
Kottke interface-window sampling/mean/moment reductions. Scalar lossless Kottke
bulk stamping and distances run in CUDA; the all-background shortcut, strict
empty-band return, ascending indices and 4096-cell window chunks retain the
existing arithmetic and ownership. Conductive Kottke edges retain the complete
torch edge primitive. These retained pieces are deliberate O3a boundaries;
coefficient fusion, table compression and meshes remain O3b/O3c work.

Reproduce the setup gates with `python -m benchmark.core_path.setup_gates g4`,
`python -m benchmark.core_path.setup_gates g1`, and
`python -m benchmark.core_path.setup_gates timing`. Receipts default to
`<evidence-root>/core-cuda-gates/setup/`; use `--output DIR` to
select another directory. Each configuration records actual setup dispatch,
and each command reports accelerated-versus-torch counts. The original
qualification artifacts remain under `<evidence-root>/core-cuda-gates-o3a/`. The two initial
small-scene test failures were `model/surface_impedance`: accelerated setup
returned `None` instead of `CompiledSurfaceImpedanceLayout(metals=(), faces=(),
total_area=0.0)`. No shared tensor inequality explained those failures. The
earlier twenty-case probe iterated only candidate tensor keys, omitting the
baseline-only `model/surface_impedance/total_area` key and object metadata. Setup
now calls the original `compile_surface_impedance_layout(scene)` helper. Tests
recursively compare dataclass types/fields, dictionary keys, tensor dtype,
device, shape and exact values, and exercise both the direct dispatch helper
and public compiler hook. The small cases use the same uniform 0.1 grid over
[-0.5, 0.5]^3, polarized3, fp32/CUDA and `include_edges=True` as the original
vacuum probe; node summaries, zero channels, flags and all six edge components
match. Expanded G4 snapshots preserve the entire model hierarchy and check
original tensor metadata before host snapshots; the old probe also checks
complete models and key sets now. G1 additionally compares compiled models and
runtime coefficient-uniformity summaries before comparing evolved outputs.

Additional off-lattice primitive qualification exposed a box SDF reduction-order
error that the initial full-model grids did not expose: 3,731 of 100,003 random
queries differed, with maximum absolute difference 5.960464477539063e-8. The
CUDA norm used `(x*x + y*y) + z*z`; the installed ATen contiguous three-element
norm uses `(x*x + z*z) + y*y`. Matching that exact tree fixes the primitive;
there is no tolerance exception. Eight SDF checks (sphere, rotated box, sharp
union/intersection/difference, and smooth transformed booleans) and eight
seven-query coverage checks then pass with zero mismatches. The existing
sphere-plus-box test also covers 1,003 deterministic off-lattice queries to
retain this regression check.

Final GPU-1 qualification after the norm correction (no tolerance allowances):

- G4: 160/160 complete-model comparisons, 32 scenes across all five requested
  modes, comprising 5,765 tensor comparisons plus metadata/flags. This includes
  all 22 `CORE_SCENES`, empty vacuum at 64^3 and 256^3, odd, graded/custom,
  periodic, overlapping-priority, rotated-box, boolean and sphere-plus-box
  scenes. Setup accelerated 150 comparisons; mesh and PEC scenes retained torch
  for the other ten. Receipt: `g4.json`.
- G1: 22/22 `CORE_SCENES`, 64 steps at K=32 with `fdtd_core_path=True`, 4,661
  field/phasor/monitor tensor comparisons, and exact compiled-model plus runtime
  coefficient-uniformity comparisons. Dispatch instrumentation confirms twenty
  accelerated material setups and two torch setups (mesh and PEC). Receipt:
  `g1.json`. The preceding G4/G1 receipts are retained separately with the
  `-before-box-norm-fix.json` suffix.
- Targeted test file: 5/5 passed in 10.26 s, including both modulation opt-in
  settings retaining torch, nested trainable geometry retaining autograd, and
  both small-scene complete-model comparisons through direct/public dispatch.
  The sphere-plus-box case also checks off-lattice native SDF equality. Receipt:
  `tests.xml` and `run.log`. Python syntax, integration-manifest parsing and
  whitespace checks also pass.

Informational synchronized prepare timings, one run per arm, setup extension
loaded before timing, 256^3 computational grids, polarized3, eight CPU threads.
The default FDTD arm uses torch setup; the candidate uses `fdtd_core_path=True`.
Material timing wraps the actual function imported by `scene.py` and records
exactly one call per prepare. These are single-run observations, not G5 claims.
Receipt: `timings.json`.

| Scene | Torch prepare (s) | Accelerated prepare (s) | Torch materials (s) | Accelerated materials (s) |
|---|---:|---:|---:|---:|
| Vacuum 256^3 | 0.783796 | 0.206135 | 0.617064 | 0.169068 |
| Sphere 256^3 polarized3 | 7.273676 | 0.370360 | 7.249760 | 0.210582 |

Own peak allocator memory across the gates and timings was 5,989,509,120 bytes
allocated (5.578 GiB), 6,041,894,912 bytes reserved (5.627 GiB). Sampled process
memory peaked at 6,068 MiB (5.926 GiB), excluding other sessions' processes;
receipt: `process-memory.json`. Only one own GPU worker ran at a time, using the
required wrapper and private build directory. No cloud resources or stable
stepping CUDA sources were used or changed.

### O3b implementation notes

**Draft; GPU qualification blocked (2026-11-02).** Work starts at `a1e9827a`
on `perf/core-cuda-o3b`. The user authorized coefficient hook (2), with owner
acknowledgments for coefficient composition and the independent boundary work.
There are no changes to modulation, nonlinear-channel construction, boundary
profiles, compiler/materials.py, the compiler guards or the batch-4 empty-band
implementation. No exactness or performance qualification is claimed below.

The setup extension advances to `setup_abi_v2`, still under the configured
`setup/` build directory with `--fmad=false` and no fast math. The draft adds:

- Native window arithmetic/reciprocal means, strided first moments, norm,
  conditional denominator, axis weight and blend. The installed
  `ATen/native/cuda/Reduce.cuh` supplies the contiguous x-reduction configuration
  and four-accumulator strided sum tree for 27/64 samples; the three-element norm
  uses `(x*x+z*z)+y*y`. Chunk sizes stay 4096, with ascending near indices and
  the empty-band early return before window work.
- Polarized inverse reconstruction fuses squares, chronological node component
  accumulation, norm, contrast threshold and diagonal blend. The isotropic
  eligibility envelope permits sharing one arithmetic field's gradients across
  the three equal node components. `torch.gradient` remains responsible for
  nonuniform coefficients and endpoint stencils; no new stencil is substituted.
- One early opt-in call in `build_update_coefficients`, reusing scoped compiler
  eligibility and additionally retaining PMC boundary cases on torch. It reads
  the already rounded absolute `eps_*`/`mu_*` tensors from material binding,
  composes all six staggered decay/curl pairs in one launch, and follows current
  semi-implicit conductivity and absorber-profile arithmetic. Reverse scalar
  division retains torch's reciprocal-then-multiply order. Uniformity and its
  aminmax reductions remain torch. PEC/PMC materials, tensor media, dispersion,
  nonlinear/modulated media, charts, caches and trainable tensors retain torch
  through the existing eligibility envelope.
- The core table builder hashes all six padded signed-int32 bit patterns,
  resolves collisions with full-key equality against immutable input rows,
  compacts occupied slots, performs six stable signed-int32 lexicographic sort
  passes over distinct rows, and remaps directly to uint16/uint32. It allocates
  no N-by-6 padded workspace. The scalar-uniform bypass and >65535 threshold
  stay unchanged. The 2^20-slot table permits 128 probes; probe overflow returns
  the designed sentinel and retains the existing unique implementation.

Periodic/tanh fills, smooth interface-window fill sampling and conductive
Kottke edge primitives remain torch. Their existing periodic image/seam logic,
transcendentals and conductive sampling have not been demonstrated bitwise
identical in this environment. They must not be admitted on an approximation.
Native window reductions and reconstruction are likewise **unqualified drafts**
until the GPU gates pass; installed-header inspection and compilation do not
establish device equality.

`benchmark.core_path.setup_gates` now configures 180 G4 comparisons (the original
160 plus four scenes times five modes): curved+rotated-box interfaces,
conductive Kottke interfaces, a periodic smooth boolean and a full-interior
empty-band case. Receipts include actual extension operation counts. Additional
commands are `windows` (22 configurations / 66 axis comparisons for 27/64
samples), `coefficients` (22 core scenes plus eight new Kottke configurations),
`tables` (both families for 22 core scenes plus a 256^3 edge4 sphere), and
`o3b-timing` (256^3/384^3 sphere+box, edge4/smooth4/polarized3, both arms).
Table comparisons check index/table dtype, shape, torch.equal and float-table
int32 bits, including signed zeros. Coefficient comparisons include all stored
component coefficients, CPML tensor attributes and coefficient uniformity.
The existing test file adds only modulated-coefficient exclusion; its existing
trainable-geometry/autograd exclusion is retained. No new test file or runtime
raise is introduced.

Execution receipts and the required GPU-1 wrapper are under
`<evidence-root>/core-cuda-gates-o3b-work/`. The wrapper sets the requested device, CUDA toolkit,
private build directory, thread counts and worktree import path before torch;
it caps allocator use at 14.5 GiB to leave process/context headroom below 16 GiB.
`nvidia-smi` sees both devices, but this managed filesystem sandbox exposes no
`/dev/nvidia*` nodes. The required wrapper fails at CUDA initialization with
`RuntimeError: No CUDA GPUs are available`. The approval policy is `never`, so
there is no permitted elevated execution route in this session. This is an
execution-environment blocker, not a booked-window pause.

Offline architecture-8.6 compilation and loading of the v2 extension succeeds;
receipt: `static-build.log`. `tests/fdtd/core_path/` collects 16 tests and skips
all 16 because CUDA is unavailable (`pytest.log`). GPU gate passes: **zero**;
G4, coefficient, table, window, G1 K=32, G2 and timing qualification all remain
pending. Prepare seconds, table-build seconds and peak GPU memory are
**unmeasured**. Do not use this draft as evidence that O3b is complete.

### O3c implementation notes

The maxwell-owned `compiler/accel/mesh.py` registers once through the installed
core `_register_mesh_sdf_accelerator` hook at package import. Its seven methods
are `supports`, `is_available`, `unsigned_distance`, `winding_angle`,
`smooth_signed_distance`, `unsigned_distance_static_bvh` and
`signed_distance_static_bvh`. Selection requires scoped `setup_enabled`, matching
CUDA devices, fp32 and fixed nonempty query/geometry tensors. Other inputs retain
core's torch selection. Availability performs no device probe or build; extension
loading happens in the operation so build/execution failures propagate rather
than being swallowed by core's optional-loader exception handler. No core files
are changed and no runtime capability raises are added.

Distance uses a fused evaluator below 32 triangles. Larger device meshes use
30-bit Morton codes with original triangle IDs in the low 32 bits, a stable
device sort, radix LBVH topology and bottom-up device refitting. Supplied core
static trees retain their topology/ranges/original IDs; bounds are refit to
enclose actual rounded vertex/edge projections. Bounds are fp32 SoA; topology
and triangle IDs are int32. Original triangles remain in authored order. The
bounded weak-reference cache keys device, dtype, storage pointer, offset, shape,
strides and tensor version. Core's transformed-triangle cache invalidates the
tensor for transform/vertex/topology revisions; maxwell does not copy device
geometry to the host or bypass topology-staleness checks. CPU-authored static
trees can still be supplied by core's existing cache. Cached hierarchies carry
a readiness event and record consuming streams for safe reuse and eviction.

Clamped face projections are not assumed to lie in the geometric triangle box.
Each node additionally encloses authored vertices and rounded edge vectors.
Outward directed arithmetic must prove a vertex/edge predicate for every
triangle in a node before traversal can prune it. Otherwise it is visited,
including degenerate/ill-conditioned or nonfinite uncertainty. The box encloses
all rounded clamped-edge projections; its distance lower bound follows the
three-term reduction order with downward rounding. Pruning uses strict
`lower > best`, so equality ties are visited. Minima retain original triangle
IDs, chronological predicate overrides, fp32 epsilon clamps and current
square-root smoothing. This certificate can reduce pruning; performance must
be measured rather than inferred from tree complexity.

Winding calls the installed torch implementation unchanged: original triangle
order, default 256-triangle chunks, installed sum tree and chronological chunk
accumulation. Sign composition calls core's existing torch helper. Mesh coverage,
mesh-containing boolean transforms/composition, periodic handling and smooth
occupancy stay torch, with the hook accelerating their unsigned distance.
Solid-mesh coverage reuses the existing packed/cached helper, including its
bit-pattern coordinate chart and budget. Mesh and mesh-containing booleans are
admitted by compiler eligibility; mirrored integer topology attributes on boolean
nodes do not disqualify otherwise fp32 geometry. Native scalar accumulation,
reconstruction and coefficient/table paths keep the existing O3 implementation.

The extension marker advances to `setup_abi_v3`, with `mesh.cu` and `bvh.cu`
included and `--fmad=false` retained. Mesh dot products follow the installed
contiguous three-element reduction: x/z before y, with positive-zero initial
accumulators. Explicit rounded additions preserve stored torch intermediates.

`setup_gates` adds `mesh-ties`, three G4 scenes (six-triangle pyramid, exactly
5000-triangle closed mesh and nested mesh/box boolean), `cpu-dry-run` and
`o3c-timing`. Tie controls cover vertices/edges/faces, AABB boundaries, coplanar
shared faces, duplicates/reversals, degenerate/sliver triangles, equal minima,
signed zero, exact 2*pi winding, open/nonmanifold input and tree/chunk thresholds.
The two a302ac9 strict-xfail fixtures are embedded losslessly, retaining original
float64 authored vertices and fp32 query/world materialization. Their current
torch distance/winding/signed outputs are the goldens, including the known
distance error; there is no corrected near-zero assertion. The gate compares
`torch.equal` and int32 bits, device/supplied trees, public hook entrypoints,
custom winding beta, surface/solid/auto behavior, transforms, storage versions
and strided inputs. Timings use 128^3/256^3 meshes with 5000 triangles and one
arithmetic sample, recording total prepare, prepare distance/winding calls,
cold/warm raw distance and exact winding for both arms.

This task runs code and CPU-only checks: py_compile, offline sm_86 extension
build/load with the specified toolkit, and construction of all 23 dry-run
fixtures. CUDA initialization is asserted false. Receipts and coordinator
commands live under `<evidence-root>/core-cuda-gates-o3c-work/`. GPU equality, G4 admission and
timings remain **unqualified and unmeasured** until the coordinator runs the
commands after the shared GPU is released. These notes do not claim completed
O3 numerical/performance acceptance.

### S0/S1 implementation notes

Code-only work on `perf/core-cuda-s01`, stacked on O3c `6d33b174`. No GPU
execution, cloud work, push or merge. The new public fixed control is
`Simulation.fdtd(..., accelerated_setup=False)`, stored/coerced in FDTDConfig.
The existing finally-reset ContextVar around single-device `solver.init_field`
now uses `(accelerated_setup or fdtd_core_path) and material_cache is None`.
The core stepping admission and activation branches are unchanged. Setup-only
preparation never invokes that admission; unsupported pieces return their
sentinels independently. Distributed preparation is unchanged and does not opt
its shard initialization into this single-device scope.

The complete `eligible()` condition audit follows. Coefficient eligibility uses
the same common guards but consumes bound constitutive tensors rather than
sampling geometry. No source, port, monitor or wire type admission is added.

| Condition | S0/S1 decision and arithmetic reason |
| --- | --- |
| ContextVar disabled | Keep: default path must remain torch. |
| Non-CUDA scene | Keep: extension operates on CUDA storage only. |
| Coordinate chart present | Keep: mapped metrics are outside these Cartesian formulas. |
| Material dtype other than fp32 | Keep: kernels operate on fp32. |
| Axes not 1D or node query grid not 3D | Add sentinel: no Batch lane dimension is consumed. |
| Material-model override | Keep: cache-owned constitutive inputs must not be recomputed. |
| Conductivity capture active | Keep: replay requires per-owner occupancies, absent from fused sampling. |
| Cropped component cardinalities | Keep: kernels assume full three-dimensional Yee extents. |
| Any recursively trainable scene/background/subpixel tensor | Keep: kernels do not implement the authored autograd path; includes trainable source/port controls. |
| MaterialRegion density present | Keep for material sampling: density/filter/polarized arithmetic differs. Remove for coefficients: already bound tensors carry the result. |
| Policy outside edge4/smooth4/polarized3/polarized4/arithmetic1 | Keep for material sampling. Remove for coefficients: sampling policy cannot change the bound-tensor formula. |
| Background not a Python int/float | Keep: scalar-background packing is the qualified material contract. Coefficients pass scalar unit placeholders. |
| Modulation on any declared structure, including disabled | Keep: quadrature/forward-AD replay is separate. |
| Exact Material type | Restrict to enabled bulk structures: custom constitutive sampling remains torch. Remove for separately compiled PEC/PMC, copper and 2D surfaces. |
| Unsupported primitive/boolean geometry | Keep for bulk sampling; add Cylinder, Pyramid and PolySlab. Nonbulk geometries use their original helpers. Remove for coefficients: geometry is no longer an input. |
| Geometry tree larger than 16 nodes | Keep for material sampling; native descriptors have fixed capacity. Remove for coefficients. |
| PEC/PMC material | Remove global rejection: excluded from bulk blends; compile masks/planar terms separately, then retain original coefficient suppression. Unqualified conductor geometry returns its piece sentinel. |
| Dispersive bulk material | Keep: pole weights are out of scope. Coefficients also reject declared dispersive materials; no ADE constitutive extension is inferred. |
| Epsilon/mu/conductivity tensor descriptor | Keep for bulk media: scalar isotropic packing does not implement tensor constitutive sampling. |
| Nonlinearity, Kerr or breakdown | Keep for bulk media: field-dependent coefficients differ. |
| Tensor-valued bulk material attribute | Keep: fixed scalar Python values are the qualified packing envelope, even without gradients. |
| Non-fp32 bulk geometry tensor | Keep for material sampling; original geometry arithmetic is preserved. Remove for coefficients. |
| Ports, monitors, sources, wires | No type blocker before or after: these declarations do not enter bulk arithmetic. Nonbulk surface type rejection was the copper blocker; it is removed. Recursive trainable exclusion still applies. |

Coefficient composition additionally checks real contiguous fp32 3D tensors on
one device, with no gradients; other layouts return UNCHANGED. Runtime full
anisotropy, dispersion, nonlinearity, modulation, breakdown and gyromagnetic
flags also retain the coefficient sentinel, including full tensors synthesized
by an otherwise unsupported subpixel policy. The old blanket
PMC-boundary exclusion is removed. Both `_apply_pec_edge_suppression` and
`_apply_pmc_face_suppression` run in their original position before surface
configuration/uniformity. Table compression now returns its sentinel for
unsupported dtype/device/stride/rank/gradient inputs; the hash-overflow sentinel
remains. Mesh support explicitly excludes lane-shaped queries/geometry. Cache
overrides disable the setup scope, including the independent mesh hook.

Newly admitted arithmetic (all device qualification remains pending):

| Piece | Implementation and retained arithmetic |
| --- | --- |
| Cylinder SDF/coverage | Native local-frame, radial/cap distance and seven-query coverage; all three axes and transformed boolean composition. Operation order follows installed core, with FMA disabled. |
| Pyramid SDF/coverage | The existing scoped triangle-distance hook accelerates the six original local triangles. Original torch local/canonical coordinates, half-extent sign predicates and coverage stay unchanged. No world-mesh substitution. |
| PolySlab SDF/coverage | New native 2D closed-edge minimum and even-odd crossing count, retaining core projection/denominator/closest-point order. Torch retains transforms, loop splitting, tan(sidewall_angle), taper, axial composition and coverage. Concavity and taper have fixtures. Ear clipping is untouched. PolySlab-containing boolean trees retain their original torch polygon sampling; their blend sums may still accelerate. |
| PEC/PMC occupancy | Native qualified primitive distance (or scoped triangle distance) with original torch tanh, half-boundary convention, ordered union and PMC detach. Unknown/double conductor geometry returns UNCHANGED for that mask. Planar terms keep their original helpers. |
| Conformal PEC edges | Original periodic union distance and boundary snapping followed by one native three-component endpoint-fill kernel. Planar-support maximum remains torch. |
| Lossy Kottke edges | Native chronological material/sigma/inverse six-channel sample sums, including first-sample assignment and subsequent ordered additions. The isotropic final material is still the existing Kottke window result; its unused reconstructed sample mean is not evaluated. Conductivity is no longer forced through the whole torch sample loop. |
| Periodic images | Native seven-query coverage for Sphere/Box/Cylinder and native boolean trees per image. Original image subtraction/order, maximum and seam/midpoint wrapping stay torch, including reduced-axis spacing minima. Other geometries keep torch coverage with their independently accelerated distance where supported. |
| Single-sample tanh | Distance and ordered material blends can accelerate. Tanh itself stays the installed torch operation: no unverified libdevice transcendental substitution. |
| Sheets/copper/wires/ports/closed surfaces | Material compilation uses the unchanged sheet/planar/surface layout helpers and `_edge_native_eligible`; sheets and surface-impedance media retain their original node-to-edge choice. Copper, ports, wires and closed-surface observers retain their own setup runtimes. |

No dispersive pole-weight implementation is added. Unsupported bulk materials,
charts, trainable inputs and caches keep their material sentinel; coefficient
and mesh selection remain independent. The extension advances to setup_abi_v4
without changing the stable stepping extension or `--fmad=false`.

Prepared coordinator gates:

- G4 now uses public `accelerated_setup=True` with core stepping off, comparing
  complete compiled model, all stored update/CPML coefficient tensors, uniformity
  and tensor metadata against normal prepare. Its 310 configurations include
  23 S0/S1 scenes times all five policies (115 new comparisons), covering the
  new primitives, conductor masks/conformal edges, periodic/lossy combinations,
  CopperFoil, lumped port, thin wire, ClosedSurfaceMonitor, dispersive bulk and
  static 2D sheet. Receipts record actual native operation counts and per-piece
  accelerated/torch paths; tables are marked not requested in setup-only G4.
- `primitives` compares 10,006 off-lattice/boundary queries for ten primitive
  fixtures: raw SDF, coverage, tanh occupancy and int32 bits. It also checks the
  new periodic coverage entrypoint against the original coverage helper.
- Coefficient gates include all 115 new policy/scene configurations; existing
  windows, tables and mesh-ties regressions remain ready. Two parameterized
  tests require material acceleration and normal preparation for port/copper,
  then assert the unchanged core stepping rejection.
- `s01-timing` prepares every new fixture at exactly 256^3 computational nodes,
  all five policies, both torch and accelerated_setup arms (230 prepares),
  recording total prepare time, piece counts and allocator peaks.

CPU checks passed: syntax compilation of all 14 changed/imported Python source
files plus the coordinator runner; offline architecture-8.6 build/load with the
specified environment/toolkit; 333 CPU dry-run fixtures, including every G4
scene/policy and old mesh ties. Additional CPU-only port geometry and copper
support compilation at 19^3 and 256^3 caught off-grid fixture coordinates, which
are now derived from each grid. CUDA initialization was guarded during build
and asserted false after build/dry construction. Whitespace checks pass.
Receipts/build products and ready-to-run commands are under
`<evidence-root>/core-cuda-gates-s01-work/`, especially `GPU_TODO.md`. GPU equality passes:
**zero**; no S1 device piece is declared physically/numerically qualified and no
prepare speedup is claimed until the coordinator runs these gates.
