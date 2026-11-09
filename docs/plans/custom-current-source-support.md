# Custom current source: physical volume support

Status: coordinator-approved fix (2026-11-02). Branch `fix/custom-current-support`, worktree
`<workspace>/wt/maxwell-ccur-fix`, base master f928126d.
Diagnosis: `<evidence-root>/instamax-line-classc-triage-20261101/custom-current/REPORT.md`
(original in `<scratch>/witwin-classc-current/`). Read §1, §2, §6 and §7 first.

## Defect

`fdtd/excitation/injection.py::_prepare_custom_current_source` → `_append_interpolated_current` samples the
whole covering window chosen by `_region_index_range` / `_axis_index_window`. It uses clamped
`_axis_interp_weights` and deposits the full density in every window cell. That includes cells whose control
volume has zero overlap with the dataset volume, and it overweights the partial cells.

At ×1 the compiled current moment is 2.3596× the declared integral for the benchmark's custom Jx box and
2.1751× for the Mz box. The enlarged support also moves the radiation zeros. `_prepare_uniform_current_source`
is already correct: it uses `_yee_box_overlap` fractional weights.

## Required behaviour

- **Support.** For every J (E positions) and M (H positions) component, deposit the dataset only over the
  intersection of each Yee control cell with the endpoint-defined dataset volume: [first sample, last sample]
  per axis, the same convention as the reference export. Multiply by covered volume / full control volume.
  - Cells with zero overlap get no current.
  - Partial cells sample at an in-span point (as the merged custom-field fix does: `_sheet_axis_quadrature`
    and its in-span queries), or integrate the trilinear interpolant over the covered subvolume.
  - Never continue the endpoint density into uncovered volume.
  - Do NOT rescale the old patch by a scalar.
- **No duplicated helper.** Extend the primary coordinate/overlap helpers (`_yee_component_coords`,
  `_yee_box_overlap`, or their shared core) to H positions as well, instead of writing a second
  implementation. Keep the existing callers' results bitwise.
- **Unchanged paths.**
  - `_prepare_uniform_current_source` and its products: coefficients, runtime fields and flux are bitwise
    unchanged.
  - The existing singleton-axis (zero-span) current behaviour is unchanged. A zero-span axis keeps its
    present nearest-plane rule; do not multiply by a zero span.
  - Timing: J and M injection time levels stay exactly as they are. The additive-J half-step correction is a
    separate change owned by another line. Do not edit `runtime/stepping.py`.
- **Public docs.** Clarify in the CustomCurrentSource/CurrentDataset docstrings that values are volume
  densities in SI units (J: A/m², M: V/m²) over the endpoint-defined volume. Update FEATURE_LIST's
  custom-current entry.

## Gates (fast tier; pass/fail is the pytest/script process exit code; run on branch AND base f928126d)

1. **Integral and extent.**
   - For every J and M component, and for centered and translated boxes, constants of several signs and
     amplitudes, both meshes (h = 1.28/52 and 1.28/103), a nonuniform grid, physical-domain clipping and
     zero-valued components: Σ density·control_volume equals the declared integral. Tolerances: rtol 1e-6
     for fp32, 1e-12 for fp64.
   - Recover density from the PREPARED patch (`-patch·eps/dt`, `-patch·mu/dt`), not from helper weights.
   - Equal geometry: custom and uniform sources give equal moments and profiles.
   - Put this in a new regression test file; tests/sources already has an owner.
2. **Continuum convergence.** Rerun the diagnosis's public CPU procedure (scripts in the diagnosis directory)
   at ×1 and ×2 on branch and base, with the same masks and normalization.
   - Report CPU vs the finite-box continuum radiator (L2/shape/Linf) and six-face power for all three
     scenarios.
   - The custom rows must move toward the arithmetic counterfactual (§7 table); uniform must be identical.
3. **Uniform invariant.** `uniform_current` preparation coefficients, fields and flux are torch.equal to base
   on both grids.
4. **Adjoints and readers**, each file in its own process, on branch AND base:
   - `tests/gradients/test_fdtd_custom_source_adjoint.py`, `test_fdtd_tensor_current_adjoint.py`,
     `test_fdtd_adjoint_bridge.py`, and the full-tensor current contraction tests;
   - `tests/api/adapters/tidy3d/`;
   - `tests/sources/`;
   - every file in the diagnosis's `preparation_readers.txt` inventory that has tests. Re-inventory after the
     patch.
   - `tests/api/public/test_guard_census.py`.
   - Do not count a skipped CUDA test as a pass.
5. **GPU.** The Codex sandbox may have no GPU. Run every GPU-requiring file through
   `env CUDA_VISIBLE_DEVICES=1 python <scratch> --log <scratch>/witwin-ccur-fix/<name>.log --pytest <file>`
   (the runner moves the run to GPU 0 and enforces the shared rule). If that fails for lack of a device, list
   every pending GPU command in the report under "GPU runs for the reviewer" and stop. The reviewer runs them.

## Rules

- witwin-maxwell CLAUDE.md applies: English comments, no shims, no commercial product names.
- No defensive programming. Small commits. One command per shell call, no `&&`.
- No cloud. Never modify caches. Outputs go to `<scratch>/witwin-ccur-fix/`.
- Commit as the repository owner. No AI attribution. No
  push or merge.
- Report:
  - files changed;
  - design;
  - gate tables;
  - suite exit codes for branch and base;
  - the reader inventory;
  - guard census;
  - pending GPU runs.
