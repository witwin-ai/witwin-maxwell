# Custom field source: physical aperture and consistent E/H staggering

Status: coordinator-accepted fix (class-(c) triage, 2026-11-01). Branch `fix/custom-field-source-stencil`,
worktree `<workspace>/wt/maxwell-cfs-fix`, base master d7b91f1f.
Diagnosis: `<scratch>` (copy;
originals and scripts in `<scratch>/witwin-classc-customfield/`). Read it first.

## Defects (in `witwin/maxwell/fdtd/excitation/injection.py::_prepare_custom_field_source`)

1. **Aperture over-coverage.**
   - `_axis_index_window` selects a covering node window, and `_trilinear_sample` /
     `_axis_interp_weights` clamp positions outside the dataset to the edge value.
   - A declared 0.8×0.8 m aperture therefore becomes a full-weight 34×35 patch of 0.721 m²
     (+12.7%) at h = 0.0246 m.
2. **Inconsistent normal staggering.**
   - The plane snaps to a primal node and reuses the discrete TFSF low face. The electric sheet sits
     at z_E and the magnetic sheet at z_H = z_E − h/2.
   - Both receive the SAME supplied tangential profile, with no continuation over the half-cell
     separation.
   - A unit plane wave consequently launches forward gain cos(k_Y h/4)/cos(k_Y h/2) = 1.1245 and a
     backward wave sin(k_Y h/4)/cos(k_Y h/2) = 0.312. Both errors are first order in h.

## Required behaviour

- **Physical aperture.**
  - Each tangential component's current support respects the declared dataset region.
  - Use fractional overlap weights at that component's Yee positions: the cell or dual-cell length
    fraction inside [lo, hi] along each transverse axis, applied to its sheet current.
  - Positions outside the region get zero. No edge-value extrapolation.
  - A dataset interpolates only inside its sample span. Outside the span and inside the region,
    the nearest-edge rule is allowed only within half a sample spacing; document it.
  - Do not shrink or taper the declared aperture as calibration.
- **One declared plane, consistent E/H pair.**
  - The supplied (E_t, H_t) are the tangential fields ON the declared plane, at the source time
    phasor convention.
  - Inject an equivalent-current pair that radiates exactly those fields into the declared half-space
    and nothing into the other half-space, in the discrete sense.
  - The diagnosis's closed form shows what this needs. The value applied at each staggered sheet
    must be the supplied field continued to that sheet's actual normal position, using the
    discrete (Yee) normal propagation for each transverse spatial-frequency component: an angular
    spectrum on the plane's tangential grid, with the discrete dispersion k_z,Y(k_t, ω̃) at the run's
    h and dt.
  - The supplied E/H pair fixes the direction. Do not add a direction argument.
  - If an alternative common-plane equivalent-current discretization is cleaner, it must meet the
    same gates. State why in the report.
- **Runtime constraints.**
  - Keep the runtime GPU-first and PyTorch-native. The preparation may use torch FFTs on the
    device.
  - Keep the existing gradient path differentiable. `tests/gradients/test_fdtd_custom_source_adjoint.py`
    must pass unchanged.
  - No fitted scalars. No new public API, unless a choice would change the public contract; in
    that case stop and report.


## Contract clarification (reviewer, 2026-11-01): option A

The public contract stays as it is. The supplied tangential components define independent
equivalent surface currents on the declared plane: J = n × H from the H components and M = −n × E
from the E components. Partial datasets (for example Ex alone) remain valid.

The target of the fix is the CONTINUUM radiation of exactly those currents over the declared
aperture, discretized consistently. Remove the artificial discretization artifacts:
- aperture over-coverage;
- the half-cell E/H sheet inconsistency that adds O(h) gain and O(h) spurious backward radiation.

Do NOT force one-sided radiation where the supplied currents physically radiate both ways.

- **Consistent stagger.** The J and M sheets sit at their Yee positions half a cell apart. Each
  sheet's current must be the supplied current continued to that sheet's actual normal position.
  Continue it as the field radiated by the declared-plane currents, using the discrete two-way Yee
  propagator per transverse spatial frequency. Equivalently, use a common-plane current
  discretization whose discrete radiation equals the continuum radiation of the declared-plane
  currents to second order in h.
  - For a complete Maxwell-compatible plane-wave spectrum (η H_t matching the discrete modal
    admittance κ_z/K per k_t), the result is one-sided.
  - For incompatible or partial inputs, the physical two-sided equivalent-current field remains.
- **Gates, replacing gates 1–3:**
  - **1'.** Full-span compatible unit plane wave: forward gain within 1e-3 of 1, backward amplitude
    below 1e-3, both against exact Yee propagation at the same h/dt, at ×1/×2/×4.
  - **2'.** Oblique compatible plane wave (supply the matching η H / E = κ_z/K for that k_t): the
    same checks.
  - **3'.** Finite rectangular benchmark aperture (an incompatible pair): forward AND backward
    fields converge under ×2 to the independent continuum equivalent-current radiator
    (`<scratch>/witwin-classc-customfield/arbiter.py`; its physical backward lobe is ~0.0073 of forward
    on axis). Report L2/shape/Linf forward and backward at ×1/×2; both must improve over base.
  - **3''.** Ex-only (M-only) partial input: matches the continuum M-sheet radiator, two-sided and
    symmetric, and converges under ×2.
  - The adjoint tests stay unchanged and must pass.
- **FEATURE_LIST.md:** update the custom-source entries (around lines 465 and 924–925) to describe the
  corrected discretization. The input semantics are unchanged.

## Gates (fast tier)

1. **Full-span unit plane wave** (PEC/PMC transverse walls, as in the diagnosis's `cpu_plane.py`). At
   the benchmark h/dt and at 2× and 4× grids:
   - forward gain within 1e-3 of 1 against EXACT Yee propagation at the same h/dt;
   - backward amplitude below 1e-3;
   - continuum error decreasing with refinement.
   - Add a normal half-cell translation control: source-strength artifacts no longer O(h).
2. **Oblique plane wave:** a full-span Bloch case, or a large aperture, with the same unit-gain and
   no-backward checks against discrete propagation. This proves the per-k_t continuation, not only
   normal incidence.
3. **Finite aperture:** the benchmark's rectangular sheet converges under grid ×2 to the
   independent dyadic-Green radiator (`<scratch>/witwin-classc-customfield/arbiter.py`), with the
   declared aperture unchanged. Report L2 / shape / Linf at 1× and 2×. The fixed coarse result
   should beat today's 0.851 L2 against the radiator.
4. **Benchmark scenario custom_field_source,** against the read-only cache
   `benchmark/cache/custom_field_source.h5`. Report field
   L2 / shape / Linf / corr / flux before and after. This is not a pass/fail gate, because the
   coarse reference has its own discretization error.
5. **Suites,** each file in its own GPU process with the process exit code recorded, on branch AND
   base master:
   - `tests/sources/definitions/test_custom_uniform_sources.py`
   - `tests/gradients/test_fdtd_custom_source_adjoint.py`
   - `tests/core/grid/test_nonuniform_uniform_equivalence.py`
   - `tests/api/adapters/tidy3d/test_tidy3d_adapter.py`
   - every other test file that builds a custom field source, or reads TFSF face specs or
     `_prepare_custom_field_source` products. Do a reader sweep over the whole `tests/` tree with
     grep and list what you found.
   - Also `tests/api/public/test_guard_census.py` (CPU). If a guarded raise changes, reconcile it in
     its own commit.
   - Plus one new regression test file holding gates 1–2 at small size (CPU or GPU, ≤2 min).
     It needs an owner entry in `scripts/dev/integration_suites.toml` unless its directory already
     has an owner.

## Rules

- witwin-maxwell `CLAUDE.md`/`AGENTS.md` apply.
  - English comments.
  - No compatibility shims.
  - No commercial product names in code, comments, commits or branch.
  - Update `FEATURE_LIST.md` only if user-visible behaviour is described there; the injected field
    is user-visible, so check its custom-field-source entry.
- No defensive programming. Small commits. One command per shell call, no `&&`.
- **GPU:** GPU 1 only, through exactly
  `env CUDA_VISIBLE_DEVICES=1 python <scratch> --log <scratch>/witwin-cfs-fix/<name>.log (--pytest <args> | --script <path> <args> | --module benchmark <args>)`
  - The runner serializes on one lane lock, waits for ≥26 GiB free, and kills a process above
    2.6 GiB. The batch keeps priority.
  - No other `VAR=`. Run `nvidia-smi` first.
  - If the STOP file `<scratch>` exists, start no GPU
    run. Continue on CPU (`backend="cpu_reference"`) where possible.
- Never use GPU 0. No cloud. Never modify benchmark caches.
- Write outputs under `<scratch>/witwin-cfs-fix/` (the sandbox can write only the worktree and /tmp).
- Commit as the repository owner. No AI
  attribution. No push or merge.
- **Report:**
  - files changed;
  - the design chosen and why;
  - the gate tables;
  - the benchmark before/after;
  - suite exit codes on branch and master;
  - the reader sweep;
  - guard census;
  - GPU peaks.

## External reference semantics (checked 2026-11-01, coordinator request)

The installed external reference library (2.11.2) documents its custom field source as follows.
- It uses the equivalence principle to define the injected currents from the tangential components
  only. Normal components have no effect. At least one tangential component is required.
- It applies no Maxwell-compatibility projection. Its own notes say that data not produced by its
  own solver will likely not be perfectly directional, even when both E and H are supplied.
- E-only or H-only input injects equal power in both directions.

These are the same input semantics as option A, so the benchmark row is a like-for-like
comparison. The remaining differences are each code's own surface-current discretization and the
coarse reference grid. The adapter export (`adapters/tidy3d.py::_custom_source_dataset`, and the
CustomFieldSource branch) passes the supplied Ex/Hy values and signs unchanged, per the diagnosis.
Coverage confirmation (adapter owner, master c7fb3797, `_custom_source_dataset` L1236–1266 and the
CustomFieldSource branch L1432–1446): **no hidden transform.** No H normalization against E, no
dropped or derived components, no apodization or taper, and no interpolation in the adapter. The only
reshaping:
1. The component map is the identity over the supplied components; nothing is synthesized.
2. Values are cast to complex128 unchanged, including phase, with a singleton frequency axis.
3. **Aperture extent:** center = the midpoint of each sample-coordinate span; source size = the span
   from the first to the last sample. The reference's source box therefore ends exactly at the outer
   sample points. The library interpolates onto its own grid inside that box. After this fix,
   witwin's declared region (`_dataset_region_bounds`) and its fractional-overlap support must
   equal that same span, not samples treated as cell centres or extended by half a cell. Add
   this to the report as a checked statement.
4. A single frequency (source_time.frequency).
5. Units: one amplitude scale on the source time, applied to both E and H, so the E/H ratio is
   preserved.
