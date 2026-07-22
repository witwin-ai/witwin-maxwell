# Track G4 (SIBC orientation generalization) acceptance -- stage G4a (2026-07-21)

Worktree `.worktrees/wg4-sibc`, branch `fable/sibc-oblique`, GPU `CUDA_VISIBLE_DEVICES=0`,
conda env `maxwell`. Baseline master `406fced` (clean).

## Scope delivered (G4a)

All-orientation staircased exposed-face surface-impedance boundary, plus the
orientation-equivalence and mixed-orientation stability gates with recorded
falsifications.

1. **Orientation-equivalence + stability gates on the existing axis-aligned layout**
   (commit `62eb215`). The multi-orientation Box layout already enumerates all six
   exposed faces; this stage locks in its correctness with the headline gates.
2. **Staircase (voxelized) generalization for curved conductors** (commit `2836f6f`).
   A non-`Box` good conductor is staircased from its node occupancy; every axis-aligned
   voxel face becomes a masked Leontovich surface-impedance write. Covers cylinder/sphere,
   all six orientations, and mixed orientations. The narrowband good conductor (order-0
   resistance) is wired for the staircase; the per-edge rational ADE and rotated Box stay
   fail-closed.

## Files added / changed

- `witwin/maxwell/compiler/materials.py` -- `CompiledSurfaceMetal.interior_node_mask`,
  `CompiledSurfaceFace.transverse_mask`; `_compile_voxel_surface_metal`,
  `_make_voxel_face`, `_is_axis_aligned_box`, `_faces_transverse_overlap`; non-`Box`
  routing in `compile_surface_impedance_layout` (curved -> staircase, rotated Box ->
  fail closed).
- `witwin/maxwell/fdtd/runtime/materials.py` -- `_surface_occupancy_interior_mask`
  (node->edge occupancy interior termination), `_voxel_surface_writes` (per-component
  masked Leontovich writes), voxel routing in `_configure_surface_impedance`.
- `witwin/maxwell/fdtd/runtime/stepping.py` -- masked `torch.where` write and
  `_surface_plane_index` in `apply_surface_impedance`.
- `witwin/maxwell/fdtd/resume.py` -- per-face mask fingerprint.
- `tests/validation/physics/test_sibc_orientation.py` (new).
- `tests/validation/physics/test_sibc_staircase.py` (new).
- `docs/reference/fdtd-capability-guard-census.md` -- G4a reconciliation note (budget
  unchanged at 176; funnel narrowed to conformal/oblique + Bloch + rational-on-curved).
- `FEATURE_LIST.md` -- additive staircased-SIBC bullet.

## Commits

- `62eb215` test(sibc): orientation-equivalence and mixed-orientation stability gates
- `2836f6f` feat(sibc): staircase voxelized curved conductors into exposed-face surfaces
- (docs commit for this file / census / FEATURE_LIST -- see `git log`)

## Test inventory (pass counts)

Environment for every run:
```bash
export CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=/home/xingyu/code/witwin/witwin-maxwell/.worktrees/wg4-sibc
export CUDA_VISIBLE_DEVICES=0
conda run -n maxwell --no-capture-output python -m pytest <targets> -q
```

- `tests/validation/physics/test_sibc_orientation.py` -- **6 passed**
  (`test_sibc_orientation_equivalence_is_near_bitwise[perm0/perm1]`,
  `test_double_sided_plate_exercises_both_signs_and_stays_passive`,
  `test_mixed_orientation_finite_block_is_stable[50.0/5.0/1.0]`). Cyclic-permutation
  equivalence residual ~1.7e-7 (float32 round-off).
- `tests/validation/physics/test_sibc_staircase.py` -- **6 passed**
  (`test_staircased_slab_matches_analytic_leontovich_at_three_frequencies` rel err
  0.006 / 0.000 / 0.000 at 1/2/3 GHz; `test_staircased_slab_absorbs_more_than_pec`
  0.936 vs 0.999; `test_voxel_cylinder_enumerates_all_orientations_with_masks` 30 faces
  on axes {0,1,2}; `test_generic_rational_surface_on_curved_geometry_fails_closed`;
  `test_staircased_cylinder_is_stable_over_a_long_run`;
  `test_staircased_orientation_equivalence_z_to_x` residual 3.2e-4).
- Regression (no changes): **24 passed** for
  `test_sibc_staircase.py test_surface_impedance_funnel.py test_lossy_metal_sibc.py
  test_guard_census.py` (census budget 176 intact).
- Adjacent SIBC + public smoke: **71 passed** for
  `tests/materials/surface_impedance tests/materials/sheet/test_lossy_metal.py
  tests/fdtd/surface_impedance tests/validation/physics/test_surface_impedance_broadband.py
  tests/api/public/test_public_api.py tests/api/public/test_simulation_smoke.py`.

## Falsifications recorded

All performed by a temporary scratch edit to
`witwin/maxwell/fdtd/runtime/materials.py`, restored immediately after (grep confirms
`FALSIFY` marker count returns to 0).

1. **Orientation equivalence, face-normal sign (Box path).** Flipped the face-normal
   sign for `axis == 2` only (`sn *= -1` when `axis == 2`). Run:
   `test_sibc_orientation_equivalence_is_near_bitwise`. Result: `perm0` (plate normal
   maps onto z) residual jumped 1.7e-7 -> **5.5e-2 (red)**; `perm1` (plate normal on the
   unbroken y axis) stayed green -- the falsification is orientation-precise.
2. **Stability, active branch (Box path).** Flipped the passive Leontovich sign to the
   active branch (`sn = -1.0 if high else 1.0`). Run:
   `test_mixed_orientation_finite_block_is_stable`. Result: **all three conductivities
   diverged to non-finite (red)** -- the resistive update's passivity depends on the sign.
3. **Staircase, active branch (voxel path).** Flipped the voxel write sign to the active
   branch (`((b,c,-sn),(c,b,+sn))`). Runs:
   `test_staircased_slab_matches_analytic_leontovich_at_three_frequencies` and
   `test_staircased_cylinder_is_stable_over_a_long_run`. Result: **both diverged to
   non-finite (red)** -- the staircase write is the energy-absorbing branch.

## Known gaps / deferred

- **True oblique/conformal (non-staircase) SIBC** remains fail-closed. A rotated `Box`
  (grid-unaligned face normal) is not staircased and fails closed with a phase; a curved
  conductor is handled by staircasing, which is the delivered scope.
- **Generic rational (`SurfaceImpedanceMedium`) on a curved conductor** fails closed: the
  per-edge Z-form ADE is wired only for axis-aligned Box faces. Only the narrowband good
  conductor is staircased.
- **Bloch + SIBC**, **adjoint / distributed SIBC**, **trainable SIBC**,
  **adapter export of a generic surface-impedance medium** -- unchanged, still fail-closed.
- **Voxel-voxel conflicting-owner detection** uses boolean mask intersection; a mixed
  Box/voxel pair uses the mask restricted to the Box slice window (conservative, never
  under-rejects). Not exercised by the gates (single-metal curved scenes).
- **Physics convergence gate (staircased cylinder vs resolved reference) and the
  wave-level attenuation benchmark + RESULTS row** are stage **G4b**, not delivered here.
  The staircase physics is anchored in G4a by the flat-plate-from-voxel-faces analytic
  match (<1%).

## Census

`docs/reference/fdtd-capability-guard-census.md` budget unchanged at **176**;
`tests/api/public/test_guard_census.py` passes. The single `_reject_surface_impedance`
funnel narrowed (curved conductors now compile) without adding or removing a guard.

---

# Stage G4b -- staircased-cylinder physics gate + wave-level skin-effect attenuation bench

Same worktree / branch / env as G4a. Baseline: G4a tip `bf6321f`.

## Scope delivered (G4b)

1. **Staircased-cylinder physics convergence gate** (`tests/validation/physics/
   test_sibc_cylinder_convergence.py`). The physics gate that the staircase
   generalization actually models conductor loss (not a PEC termination): a
   good-conductor cylinder is illuminated by a point dipole and the power ABSORBED
   by the cylinder is measured as the net inward Poynting flux through a closed
   6-face box enclosing it (source outside the box). The staircased SIBC (coarse,
   skin depth unmeshed) is compared against the identical cylinder as a resolved
   volumetric conductor `Material(sigma_e=sigma)` on a fine grid that meshes the
   skin depth.
2. **Wave-level skin-effect attenuation benchmark** (`benchmark/scenes/rf/
   lossy_waveguide_attenuation.py` + `run_lossy_waveguide_attenuation` in
   `benchmark/rf_validation.py`, RESULTS row `rf/lossy_waveguide_attenuation`). A
   lossy-wall TE10 rectangular waveguide (four PEC walls replaced by ONE shared
   good-conductor surface-impedance boundary); the conductor attenuation `alpha` is
   extracted from the two-line `|S21|` ratio of a short and a long guide and
   compared against the analytic TE10 `alpha_c` (Pozar 3.96).
3. **External-reference cache** (the one authorized track cloud run): a TE10
   `ModeSource`-driven lossy guide with two forward `ModeMonitor` planes
   (`lossy_waveguide_reference_scene`), registered in
   `benchmark/rf_tidy3d_references.py`. Generated; the cross-check is recorded
   honestly (a documented external-adapter gap, below).

## Files added / changed

- `tests/validation/physics/test_sibc_cylinder_convergence.py` (new) -- absorbed-power
  physics gate.
- `benchmark/scenes/rf/lossy_waveguide_attenuation.py` (new, `git add -f`) -- lossy
  waveguide two-port bench scene + analytic `alpha_c` + ModeSource reference scene.
- `benchmark/rf_validation.py` -- `run_lossy_waveguide_attenuation`,
  `_lossy_waveguide_reference_alpha`, `_lossy_wg_reference_status`, `SCENE_RUNNERS`
  entry, RF intro sentence.
- `benchmark/rf_tidy3d_references.py` -- `_lossy_waveguide_attenuation` reference
  target + section intro update.
- `benchmark/RESULTS.md` -- additive rows in the RF wave-level and RF/antenna
  external-reference tables (kept additive; the harness regenerates the full section
  from all scenes).
- `FEATURE_LIST.md` -- additive staircased-SIBC validation + adapter-export bullets.
- `docs/assessments/rf-wave-validation-2026-07-18/rf__lossy_waveguide_attenuation.json`
  (artifact, `git add -f`).

## Commits

- `3848588` test(sibc): staircased-cylinder absorbed-power physics gate vs resolved conductor
- (benchmark + docs commits -- see `git log`)

## Test inventory (pass counts)

Environment for every run (as G4a):
```bash
export CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=/home/xingyu/code/witwin/witwin-maxwell/.worktrees/wg4-sibc
export CUDA_VISIBLE_DEVICES=0
conda run -n maxwell --no-capture-output python -m pytest <targets> -q
```

- `tests/validation/physics/test_sibc_cylinder_convergence.py` -- **5 passed** (57 s).
  Measured (delta = 1.19 mm at 6 GHz, sigma = 30, R/delta = 6.7): resolved absorbed
  power 6.72e6 (dx 0.6 mm) -> 6.89e6 (dx 0.5 mm), tier change 2.6% (<6% gate,
  reference grid-converged); SIBC absorbed power 5.6e6, grid-independent across
  dx 1.5/1.2/1.0/0.8 mm (change <3%); SIBC vs resolved-fine ~0.18 (<0.25 documented
  gate); PEC absorbed 1.0e5 = 1.8% of SIBC (< 5% gate; loss falsification). The ~0.18
  SIBC-vs-resolved gap is grid-independent on BOTH sides and R/delta-independent
  (measured 0.172 at R/delta=6.7 and 0.194 at R/delta=10.1), i.e. the intrinsic
  first-order-Leontovich-on-a-staircased-curve systematic -- on a FLAT surface the same
  boundary matches the analytic Leontovich value to <1% (`test_sibc_staircase`). The
  R/delta-independence and 4-tier SIBC grid-independence numbers are reproduced by the
  committed probes `docs/assessments/g4-sibc-oblique-probes/probe_cyl_conv.py`
  (R/delta=6.7 -> 0.172), `probe_cyl_rd10.py` (R/delta=10.1 -> 0.194), and
  `probe_cyl_sibc_tiers.py` (dx 1.5/1.2/1.0/0.8 mm SIBC absorbed-power tiers); the
  committed convergence gate itself is `test_sibc_cylinder_convergence.py`.
- Adjacent (unchanged solver): **67 passed** for `tests/api/public/test_guard_census.py
  tests/api/public/test_public_api.py tests/api/public/test_simulation_smoke.py
  tests/validation/physics/test_sibc_staircase.py tests/validation/physics/
  test_sibc_orientation.py tests/validation/physics/test_lossy_metal_sibc.py
  tests/materials/surface_impedance` (census budget 176 intact).

## Wave-level bench measurement (reproducible)

`run_lossy_waveguide_attenuation()` (or `python -m benchmark.rf_validation
rf/lossy_waveguide_attenuation`), guide a=0.10 m, b=0.05 m, sigma=25 S/m walls,
dx=2.5 mm, lengths 0.60 / 0.90 m, design frequencies 2.0/2.4/2.8 GHz:

- alpha (Np/m) measured / analytic: 2.217/2.226, 1.839/1.840, 1.701/1.700.
- median rel error **0.049%**, max **0.37%** -- inside the pre-registered 5% gate.
- cond(A) 1.015, max singular value 0.363 (passive), |S11| max 0.017 (matched).
- PEC-wall falsification: extracted alpha collapses to 0.00017 Np/m (vs 1.839).
- Status **pass**.

## External-reference cloud run (one authorized track run)

`python -m benchmark.rf_tidy3d_references rf/lossy_waveguide_attenuation` (or
`attempt_reference(..., run_cloud=True)`): **generated**, task
`fdve-6dfcf3ff-cc08-4171-b90f-2666841fa75c`, cost **0.0336 FlexCredits**, exported
1 source + 2 monitors, runnable. HONEST cross-check outcome: the reference forward-mode
attenuation `ln(|amp_in|/|amp_out|)/L` = 0.20/0.13/0.09 Np/m vs analytic 2.23/1.84/1.70
(|amp_in|/|amp_out| ~ 1.08, i.e. ~8% decay over 0.6 m vs the analytic ~3x). The external
RF lossy-metal surface-impedance export under-applies the wall loss at the coarse
export grid; the PHASE-based `rf/rectangular_waveguide` beta cross-check on the SAME
adapter path agrees to ~1%, so the gap is specific to the lossy-metal surface-impedance
export fidelity -- a documented external-adapter limitation, NOT the FDTD bench (which
matches the analytic alpha_c to 0.05%) nor the binding analytic reference. Recorded, not
hidden; the cache (h5) is a local gitignored artifact as for every reference.

## Falsifications recorded

1. **Physics gate, surface resistance -> 0 (PEC-like).** Scratch edit to
   `witwin/maxwell/fdtd/runtime/materials.py` setting both `surface_r = 0.0`
   (`# FALSIFY`), restored via `git checkout`. Run the two headline nodes: SIBC
   absorbed power collapses ~200x (5.6e6 -> **2.85e4**, comparable to the PEC 6.6e4
   floor); `test_staircased_sibc_reproduces_the_resolved_conductor_absorption` and
   `test_pec_cylinder_absorbs_negligibly_falsification` both go **red**. Restored:
   `FALSIFY` count 0, both green.
2. **Wave bench, PEC walls (in-run companion).** A PEC-wall guide of identical
   geometry is lossless; its two-line `|S21|` ratio ~1 and the extracted alpha
   collapses to **0.00017 Np/m** (vs 1.839 for the good conductor). Runs every time
   the bench runs (recorded in the artifact `falsification` / conservation
   `pec_alpha_np_per_m`).

## Known gaps / deferred

- The staircased first-order-Leontovich SIBC under-predicts a curved conductor's
  absorbed power by ~18% at sigma=30 (grid- and R/delta-independent). This is the
  documented first-order-boundary-on-a-staircased-curve systematic (the flat-surface
  value is <1%); a second-order / curvature-corrected surface impedance is the future
  improvement. The gate is set at 25% so it fails closed on any regression toward PEC.
- **Half-cell surface-node placement asymmetry (convention, not a bug).** In
  `_compile_voxel_surface_metal` (`witwin/maxwell/compiler/materials.py`) both face
  orientations write the tangential-E surface node at node index `p` across the
  metal/vacuum interface between nodes `p-1` and `p`, but that index sits on opposite
  sides of the physical boundary depending on the outward normal: a `-axis`-normal
  (low-side) face writes E at the first metal node while a `+axis`-normal (high-side)
  face writes E at the first vacuum node, so the effective surface of a `+axis`-facing
  step lands a half cell farther into the vacuum than a `-axis`-facing step at the same
  interface. On a flat axis-aligned plate this is exact (a Yee symmetry, validated to
  near-bitwise by `test_staircased_orientation_equivalence_z_to_x`); on a staircased
  curved conductor the low-side and high-side steps are therefore offset by up to one
  cell relative to the true surface, which is one contributor to the grid-independent
  ~18% absorbed-power under-prediction above. Consequence: this is a half-cell
  discretization convention consistent with the Yee staggering, not an implementation
  error, and it is removed by the same curvature-corrected surface impedance that would
  close the ~18% systematic. A source comment at the placement site records the same.
- External lossy-metal surface-impedance export under-applies the wall loss (above):
  an adapter-fidelity gap, not fixed here.
- All G4a deferrals stand: true oblique/conformal (non-staircase) SIBC, rotated `Box`,
  rational-on-curved, Bloch + SIBC, adjoint/distributed/trainable SIBC, generic
  surface-impedance adapter export -- all still fail-closed.

## Census

No capability guard added or removed in G4b (only a benchmark scene, a benchmark run
function, an external-reference target, and a physics test). Budget unchanged at
**176**; `tests/api/public/test_guard_census.py` passes.

---

# Round-G audit-minor addendum (2026-07-21)

Supervisor-selected round-G cleanup applied on master (post-merge). Items G4a-e:

- **(a)** Corrected the `test_sibc_staircase.py` pass count in the G4a inventory from
  `7 passed` to **6 passed** (the file has six test functions; the count was a
  miscount).
- **(b)** The R/delta-independence (`0.172` at R/delta=6.7, `0.194` at R/delta=10.1)
  and the 4-tier SIBC grid-independence (`dx 1.5/1.2/1.0/0.8 mm`) numbers, previously
  from untracked scratch, are now reproduced by committed probes under
  `docs/assessments/g4-sibc-oblique-probes/` (`probe_cyl_conv.py`, `probe_cyl_rd10.py`,
  `probe_cyl_sibc_tiers.py`, `git add -f`). The committed convergence gate is
  `tests/validation/physics/test_sibc_cylinder_convergence.py`.
- **(c) Explicit zero-impact gate (new committed test).**
  `tests/validation/physics/test_sibc_zero_impact.py` asserts a SIBC-free scene's six
  raw last-step Yee fields are **bitwise identical** with the SIBC machinery present
  vs. a monkeypatched-out `compile_surface_impedance_layout` (patched to the empty
  layout it returns for a metal-free scene). It mirrors the breakdown track's
  `test_breakdown_parity.py` zero-impact pattern and uses a fixed step count + full
  PML box + `window="none"` so the single-GPU forward is run-to-run deterministic.
  A committed control (`test_removing_compile_hook_changes_a_real_sibc_scene`) proves
  the monkeypatch is load-bearing: with a real `LossyMetalMedium` box, removing the
  hook changes the fields.
  - **Falsification (recorded).** Forcing the SIBC compile path on for the free-scene
    gate (`present = _run(_scene(metal=True))`) made the `present` run carry the metal
    surface writes while the hook-removed `removed` run dropped them, so the six fields
    diverged and the bitwise assertion fired **RED** (`test ... FAILED, 1 failed`).
    Restored to `metal=False` -> **2 passed**. This proves the gate has teeth (it
    catches a SIBC effect leaking into a supposedly SIBC-free run).
- **(d) Documented half-cell surface-node placement asymmetry.** A source comment at
  the placement site (`_compile_voxel_surface_metal` in
  `witwin/maxwell/compiler/materials.py`) and the Known-gaps note above record that the
  low-side (`-axis`-normal) and high-side (`+axis`-normal) staircase faces write the
  tangential-E surface node at the same node index `p` but on opposite sides of the
  physical metal/vacuum boundary, a half-cell convention (exact on a flat plate,
  up-to-one-cell offset on a curved conductor) that contributes to the grid-independent
  ~18% absorbed-power under-prediction. Convention, not a bug.
- **(e)** The FEATURE_LIST lossy-metal export line was reworded so it no longer reads
  as a new adapter capability (the `LossyMetalMedium` export predates G4; no adapter
  code changed) -- it is now a validation note pointing at the pre-existing export.
