# FDTD resident design updates

Status: archived 2026-10-31.

- R1 and R3 are merged into master: merge commits `80298120` (R1) and `a1516ff7` (R3).
- R2 was implemented and is bit-identical, but it measured no gain, so it was closed
  without merging (user decision, 2026-10-30). Branch `resident/update-api` is kept for
  reference.
- R4 is approved at low priority and is tracked in its own plan.

Measured outcomes are in the status blocks below and in "Outcome".

## Goal

Make repeated FDTD evaluations of geometry/material variants on a fixed grid cheap:
keep one prepared solver resident on the GPU, update only the coefficients a design
change touches, reset dynamic state, and run again. Every resident result must be
bit-identical to a fresh `Simulation.prepare().run()` of the same scene.

An external resident explicit-Yee solver reported 0.217 s per changed design on a
60 GHz patch-antenna sequence where this package took 0.343 s. That solver uses the
same explicit algorithm; its advantage is purely that state stays resident.

## Measured baseline (fe25f531, one RTX A6000, idle)

Case: finite-ground patch, `PlanarPEC` ground and patch, 0.2 mm eps_r 2.56 substrate
`Box`, 50 ohm `LumpedPort`, CPML(6), custom grid 0.1/0.4 mm, 144,900 cells, 1335
steps, dt 1.733e-13 s. The patch top edge moves through y = 0.75, 0.375, 0, -0.375 mm.
Scripts and raw JSON were produced in a throwaway probe worktree
(`.worktrees/resident-probe`, branch `probe/resident-update`, no source changes).

| Stage, ms | Cold first design | Changed design (median of 9) |
|---|---:|---:|
| Scene + Simulation construction | 6 | 2 |
| `prepare()` | 501 | 44 |
| of which `compile_materials` | 378 | 37 (planar PEC ownership 26, edge materials 8, raster 2.5) |
| of which port runtimes / CPML / coefficients / dt | 28 / 14 / 8 / 9 | 2.9 / 1.7 / 1.0 / 0.7 |
| `run()` | 591 | 552 |
| of which step loop | 565 | 546 (0.41 ms/step) |
| **Wall per design** | **1099** | **599** |

Findings that shape the plan:

1. **`prepare()` is only 7 % of a changed design.** The step loop is 91 %, and it is
   host-launch bound: the GPU is busy 169 ms of 545 ms. The six Yee/CPML kernels take
   about 52 us per step.
2. **Driven lumped ports run eagerly every step.** `_evaluate_drive`
   (`fdtd/ports.py:1248`) costs 154 us/step, `apply_lumped_runtime` 149 us/step, and
   the port V/I DFT 93 us/step. `make_port_observer_runner` (`fdtd/ports.py:2162`)
   captures only undriven ports.
3. **The batch runtime already solves this.** `execution/batch_runtime.py:418-490`
   precomputes a drive table and captures port apply, tail and port DFT in 32-step
   graphs. `Simulation.batch` with batch_size 1 runs the same design in 257 ms with
   bit-identical S11.
4. **A design change touches very few tensors.** Moving the patch sheet changed only
   `cex_decay`, `cex_curl`, `cey_decay` and `cey_curl`. eps, mu, sigma and the port
   stamps stayed equal.
5. **In-place updates are graph-safe, with three caveats.** Native kernels read data
   pointers, so `copy_` into existing storage is visible to captured graphs. However:
   - `build_update_coefficients` and the PEC suppression reassign attributes, which
     changes the pointers;
   - uniform coefficients are baked in as scalar kernel arguments
     (`fdtd/runtime/stepping.py:134-403`);
   - port runtimes hold direct tensor references.
6. **`PreparedSimulation` is single-use because nothing resets state.** A second run
   continues the port clocks from t_end and fails the excitation-spectrum check.
7. **Bit-identical resident updates already work in a prototype.** A scratch test,
   using internals only, did two things:
   - snapshotting all 174 solver tensors after prepare (41 MiB) and restoring them
     with `copy_` takes 1.15 ms, and the re-run is bit-identical;
   - restoring that snapshot and copying design B's four coefficient tensors into
     design A's solver reproduces a fresh prepare of B exactly (V/I and S11).
8. **The inverse-design loop re-prepares every iteration.**
   `_FDTDGradientBridge.forward` (`fdtd/adjoint/bridge.py:887`) rebuilds the whole
   solver on each iteration.

## Plan

### R1. Graph-captured driven-port step path (default path, biggest win)

Move the batch runtime's driven-port machinery into the single-run stepping path:

- precomputed per-run drive table, replacing per-step `_evaluate_drive`;
- lumped apply and port V/I DFT captured together with the field block, in fixed-size
  step chunks plus an exact tail;
- the batch runtime and the single-run path share one implementation, not two copies.

Expected: changed design 599 ms to about 250 ms (the batch_size-1 measurement). This
speeds up every lumped/terminal-port run, not only design loops.

Acceptance:
- In a fresh interpreter, `torch.equal` against the pre-change tree for fields, port
  V/I histories, S/Z and monitors.
- Fixtures: LumpedPort, TerminalPort, CoaxialLumpedPort, wave-port, embedded circuits,
  multi-port sweeps, and resume across a chunk boundary.
- The drive table must use the same fp64 expression order as `_evaluate_drive`.
- Any fixture that is not bit-identical keeps the eager path for that port kind (fail
  closed). It must not be quietly accepted with a tolerance.

**R1 status (merged, `80298120`; developed on base `ddca021c`):**

- `fdtd/ports.py::prepare_port_drive_tables` tabulates every applied port drive. The
  sample times replay the float64 `t += dt` recurrence of the port clock from its
  current (possibly restored) value; the old batch table used `t0 + n * dt`, which
  differs in the last bit for most rows and changed about 7 % of float32 CW drive
  rows. The waveform is the unchanged `_drive_value`.
- `fdtd/runtime/stepping.py::_port_step_block` (ports, networks, tail, port DFT) is
  the one sequence used by both the single-run port graph and the batch step chunks.
  `_make_port_step_runner` runs it eagerly on the first step, captures it on the
  second and replays one graph per step after that.
- Deviation from the plan: the port phase is captured per step, not in 32-step
  chunks with the field block. The per-step E/H update sequence and the step loop
  stay unchanged so the thin-sheet step hooks of the PCB line merge cleanly. The
  patch case is now close to GPU-bound (step loop 160 ms against about 169 ms of
  GPU work measured on the old path), so chunking would add little.
- Graph-captured: driven and terminated `LumpedPort`, `TerminalPort`,
  `CoaxialLumpedPort` and wire-gap/wire-node ports, passive V/I observation ports,
  lumped elements, Hann/none windows, full-field GPU DFT, shutoff and resume. Kept
  eager (explicit guard): circuit-bound ports or any circuit, embedded networks, a
  host-side running DFT (complex Bloch fields), `cuda_graph=False`. WavePort and
  TerminalWavePort runs use mode sources, not port runtimes, and are unaffected.
- Evidence: `tests/rf/lumped/test_port_step_graph_bit_identity.py` runs 28 fixtures
  in fresh interpreters on `ddca021c` and on the branch and compares every field,
  per-step port V/I/drive history, port phasor, S matrix, monitor payload and lumped
  state with `torch.equal`; all equal. `tests/rf/lumped/test_port_step_graph.py`
  pins drive-table rows to the eager drive for five waveform kinds, graph to eager
  and resume at steps 1, 2, 64 and 101.
- Measured (patch case, RTX A6000 shared with other jobs, medians of 11 warm runs,
  interleaved processes): wall 588 ms (idle, before) / 627 ms (contended, before)
  to 250 ms (contended, after); step loop 534 to 161 ms. S11 identical.

### R2. Resident preparation API (opt-in)

Public shape, inside `Scene -> Simulation -> Result`:

```python
prepared = mw.Simulation.fdtd(scene_a, ...).prepare(resident=True)
result_a = prepared.run()
prepared.update(scene_b)          # returns an update report; raises if not admitted
result_b = prepared.run()
```

`PreparedSimulation` stays single-use unless `resident=True`.

**Admission** (fail closed; the error names the first violated condition):

- identical grid node arrays, boundary spec, sources, ports (ids and geometry),
  monitors, frequencies and step count;
- dt unchanged: either `TimeConfig.dt` is fixed, or the recomputed material CFL dt is
  bitwise equal;
- same material families and state shapes: dispersive, anisotropic, nonlinear,
  surface-impedance and copper banks. Copper/SIBC banks are refused in R2;
- no trainable parameters (see R4);
- CPML memory layout pinned to the first prepare's decision, so that
  `memory_mode="auto"` does not re-decide based on free memory.

**Update procedure:**

1. Compile the new scene's materials and build eps/mu/sigma, decay/curl coefficients,
   PEC suppression and port PEC-gap reopening into temporaries, using the existing
   builders unchanged. This makes the result equal to a fresh prepare by construction.
2. `copy_` the temporaries into the resident tensors and rebuild lumped stamps in
   place. Diff to skip unchanged tensors.
3. If the coefficient-uniformity map changed, re-capture the affected graphs (about
   2.3 ms). Otherwise reuse them.
4. Restore all dynamic state from the post-prepare snapshot:
   - fields, psi, lumped buffers and port clocks;
   - dispersive, gyro, wire, circuit and network state;
   - DFT, observer and port accumulators.

   The state list comes from the existing checkpoint enumeration
   (`fdtd/checkpoint.py:287`, `:564`), not a hand-written list.
5. Recompute the resume configuration fingerprint.

**Ownership:** the resident solver owns private storage. It never writes into tensors
aliased from `FDTDMaterialCache` snapshots or batch-shared banks; a cache hit is
cloned once at resident prepare.

Expected: `prepare()` 44 ms becomes update about 40 ms (still a full material compile)
plus 1 ms reset.

Acceptance:
- For every fixture and edit, resident `run()` is `torch.equal` to a fresh prepare in
  fields, V/I, S/Z and monitors.
- Edits covered: PlanarPEC move, dielectric Box resize, conductivity change,
  same-family dispersive value change, and PEC toggle.
- Sources covered: lumped-port, point-dipole and TFSF scenes.
- Refusal tests: grid change, dt change, material-family change, port move, trainable
  scene and copper bank.

**R2 status (closed without merge, branch `resident/update-api` at `507c13e5`):**

- Implemented as specified. All 30 fixtures (five edits × three sources, cached and
  uncached) are `torch.equal` to a fresh prepare, and the refusal tests pass.
- Measured on the patch case with the material cache on both paths (idle A6000, 12
  changed designs):

  | Path | Prepare or update | Run | Total |
  |---|---:|---:|---:|
  | Fresh `prepare()+run()` | 20.0 ms | 141.8 ms | 161.4 ms |
  | Resident `update()+run()` | 21.9 ms | 141.5 ms | 163.4 ms |

  The resident numbers are measured without the resume fingerprint recomputation.
  With it, `update()` takes 134 ms.
- Why there is no gain: after R1 and R3, a cached `prepare()` costs only 20 ms, and
  the resident update must rebuild the same material-dependent state that `prepare()`
  builds. The step loop is GPU-bound.
- The branch adds 662 lines, including hunks in `fdtd/ports.py` and `stepping.py`, for
  no measured benefit, so it was closed without merging.

### R3. Local material recompile

Recompile only the bounding region of the changed structures. This means extending
`LocalMaterialCache` (`compiler/materials.py:2768`, currently Box/Sphere and non-PEC)
to `PlanarPEC` sheets and PEC edge ownership (`compiler/planar_pec.py:77`, 26 ms of
the 37 ms). It must be bit-identical to the full compile, with the full compile kept
as the fallback.

Expected: update falls to a few ms.

**R3 status (merged, `a1516ff7`):**

- `compile_planar_pec` now samples each sheet separately (`_compile_planar_sheet`) and
  then composes the owners. All guards stay in the composer loop. An optional `sheets=`
  map lets a caller reuse sheets it has proven unchanged.
- `LocalMaterialCache(scene, base=model).recompile(scene, unchanged)` handles two cases:
  - a changed PEC PlanarPEC sheet is resampled on its own;
  - a changed static Box/Sphere volume is resampled inside the union of its old and new
    circumspheres plus the occupancy and Yee halos.

  Anything else returns `None`, and the caller falls back to a full compile.
- `FDTDMaterialCache` recompiles a cache miss locally when the grid, boundary,
  frequency and subpixel frame match the last full compile (outcome `"local"`).
- `BatchVariation` admits PEC PlanarPEC sheets.
- Evidence:
  - 24 edit fixtures are `torch.equal` to a full compile;
  - end to end on the patch sequence (114 solver tensors, fields, V/I, S11): equal;
  - default-path models of 44 scenes are equal to the base tree.
- On a contended GPU, a sheet edit compiles in 41 ms locally against 198 ms in full.

### R4. Follow-ups (separate plans when started)

- Allow `PlanarPEC` in `BatchVariation`, which currently refuses: "Local structure
  support is proven for Box and Sphere geometry". Same-grid design sweeps could then
  run as batch lanes, measured at 0.236 s/design with 5 lanes.
- Reuse the resident path in `_FDTDGradientBridge.forward`, so inverse-design
  iterations stop re-preparing. dt is already frozen there by `_fdtd_optimization_dt`.

## Outcome (measured, patch case, idle RTX A6000)

| Per changed design, this case | Estimate | Measured |
|---|---:|---:|
| Before (fe25f531) | | 599 ms |
| R1 + R3 merged, default path | about 250 ms | 242 ms |
| R1 + R3 merged, `Simulation(material_cache=FDTDMaterialCache())` | about 175 ms | 161–186 ms |
| R2 resident update on top of that | about 210 ms | 163 ms (no gain, closed) |

The external resident solver measured 217 ms per design on this sequence.

## Execution and coordination

- **Branches.** Each phase gets its own worktree branch: `resident/port-graph`,
  `resident/update-api`, `resident/local-compile`. R2 depends on R1. R3 is independent
  of R1.
- **Shared-file rules.** R1 changes the default stepping path in `fdtd/ports.py` and
  `fdtd/runtime/stepping.py`, which are shared with the PCB thin-copper line:
  - port numerics stay bit-identical by default;
  - the stepping and simulation changes are additive;
  - the PCB line reviews the `fdtd/ports.py` diff before merge.
- **Resources.** GPU 1 only. `nice -n 10` and 8 threads per test process. Private
  native build directory.
- **Feature list.** `FEATURE_LIST.md` is updated with R1 and R2 in the same change.
