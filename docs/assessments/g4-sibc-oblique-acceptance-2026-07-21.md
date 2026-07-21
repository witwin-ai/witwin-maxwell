# Track G4 (SIBC orientation generalization) acceptance -- stage G4a (2026-07-21)

Worktree `.worktrees/wg4-sibc`, branch `fable/sibc-oblique`, GPU `CUDA_VISIBLE_DEVICES=0`,
conda env `maxwell`. Baseline master `b89a75c` (clean).

## Scope delivered (G4a)

All-orientation staircased exposed-face surface-impedance boundary, plus the
orientation-equivalence and mixed-orientation stability gates with recorded
falsifications.

1. **Orientation-equivalence + stability gates on the existing axis-aligned layout**
   (commit `8b3ac21`). The multi-orientation Box layout already enumerates all six
   exposed faces; this stage locks in its correctness with the headline gates.
2. **Staircase (voxelized) generalization for curved conductors** (commit `8e3665a`).
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

- `8b3ac21` test(sibc): orientation-equivalence and mixed-orientation stability gates
- `8e3665a` feat(sibc): staircase voxelized curved conductors into exposed-face surfaces
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
- `tests/validation/physics/test_sibc_staircase.py` -- **7 passed**
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
