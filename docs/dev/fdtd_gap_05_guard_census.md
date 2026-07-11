# FDTD Gap P5 — Guard Census and Contract-Guard Exclusion List

Date: 2026-07-10
Status: committed baseline for the P5 guard-convergence metric (see `fdtd_gap_05_functional_completeness.md` §2.1, §5.1)

## Census (measured on master @ b04d385)

- `raise NotImplementedError` statements in `witwin/`: **107**
  (the historical "111" grep count included 2 `except NotImplementedError` handlers in
  `fdtd/meshing/autogrid.py` and 2 incidental mentions; neither is a guard)
- **Contract guards (excluded from the capability metric): 20**
- **Capability guards (the P5 target): 87**

The machine-checked version of this census lives in
`tests/api/public/test_guard_census.py`. That test walks every `raise NotImplementedError`
in `witwin/` via AST, subtracts the exclusion list below, and fails if the capability count
exceeds the committed budget. Each P5 phase lowers the budget constant in the same commit
that removes guards, so the metric cannot drift or be gamed.

## Contract-guard exclusion list

A guard is a *contract* when closing it would be wrong: it is an abstract-method contract,
input validation, or a case that is physically/mathematically undefined. Keyed by
(file, distinctive message substring) so line numbers may drift.

| File | Message key | Reason |
|---|---|---|
| `witwin/maxwell/media.py` | `Nonlinear Material frequency evaluation is not defined without a field amplitude` | No single-frequency linear sample exists without a field amplitude (plan §2.1). |
| `witwin/maxwell/media.py` | `relative_permittivity() currently supports isotropic Material only` | A scalar cannot represent a tensor permittivity; returning one would be wrong. |
| `witwin/maxwell/media.py` | `relative_permittivity() is not defined for nonlinear Material` | Undefined physics without a field amplitude (plan §2.1 verbatim example). |
| `witwin/maxwell/media.py` | `relative_permittivity() is not defined for spatially-varying custom dispersive poles` | Homogeneous scalar undefined for per-cell poles; redirect to compiled model (plan §2.1). |
| `witwin/maxwell/media.py` | `relative_permeability() currently supports isotropic Material only` | Same scalar-vs-tensor argument on the magnetic side. |
| `witwin/maxwell/media.py` | `relative_permeability() is not defined for nonlinear Material` | Same field-amplitude argument on the magnetic side. |
| `witwin/maxwell/media.py` | `relative_permeability() is not defined for spatially-varying custom dispersive poles` | Same spatially-varying redirect on the magnetic side. |
| `witwin/maxwell/media.py` | `PerturbationMedium frequency evaluation is spatially varying` | eps(x) = eps_base + s·p(x) has no homogeneous frequency sample. |
| `witwin/maxwell/media.py` | `relative_permittivity() is not defined for PerturbationMedium` | Same spatially-varying redirect. |
| `witwin/maxwell/scene.py` | `SceneModule subclasses must implement to_scene()` | Abstract-method contract (plan §2.1). |
| `witwin/maxwell/geometry/polyslab.py` | `ComplexPolySlab does not support mesh export` | Documented low-value v1 limitation, excluded by plan §2.1. |
| `witwin/maxwell/fdtd/adjoint/bridge.py` | `FDTD backward currently supports trainable scene inputs that contribute` | Fires only when no trainable input feeds the material tensors; removing it yields a silent no-op backward. |
| `witwin/maxwell/fdtd/adjoint/bridge.py` | `FDTD adjoint requires complex field state for Bloch faces` | Internal invariant enforced by the forward path; a Bloch face physically requires complex fields. |
| `witwin/maxwell/fdfd/adjoint/bridge.py` | `FDFD backward currently supports trainable scene inputs that contribute` | Same empty-trainable-input precondition as the FDTD bridge. |
| `witwin/maxwell/adapters/tidy3d.py` | `Tidy3D export for magnetic dispersive Material` | Tidy3D has no magnetic-material model; genuinely no equivalent construct. |
| `witwin/maxwell/adapters/tidy3d.py` | `Tidy3D export currently assumes mu_r = 1` | Tidy3D fixes permeability at 1; genuinely no equivalent construct. |
| `witwin/maxwell/fdtd/excitation/tfsf_state.py` | `TFSF slab mode is required for Bloch-boundary TFSF injection` | Under Bloch periodicity the transverse faces coincide with the periodic boundary; a 6-face total/scattered split is ill-posed. |
| `witwin/maxwell/postprocess/stratton_chu.py` | `requires at least two exterior material samples` | Input validation for the exterior-homogeneity check. |
| `witwin/maxwell/postprocess/stratton_chu.py` | `must have at least one material cell outside the surface` | Input validation (plan §2.1). |
| `witwin/maxwell/postprocess/stratton_chu.py` | `must expose tangential_bounds metadata` | Input validation of required face metadata (plan §2.1). |

The `tidy3d.py` magnetic pair was reworded in P5.6 to state the physical reason
directly ("Tidy3D has no magnetic-material model": the Tidy3D FDTD solver fixes
mu_r = 1 and exposes no permeability-pole construct), replacing the former
"not implemented yet" wording. They remain contract guards regardless of wording,
still keyed by the substrings `Tidy3D export for magnetic dispersive Material` and
`Tidy3D export currently assumes mu_r = 1`, so `CONTRACT_GUARDS` is unchanged.

## Capability-guard budget schedule

| Milestone | Budget | Notes |
|---|---|---|
| P5.0 baseline (this commit) | 87 | measured |
| after P5.1 (adjoint) | **85 (measured)** | multi-source + `normalize_source` single-source raises deleted (bridge+core 13 → 11). Most P5.1 capability (σ_e, χ²/TPA, full-aniso ε, Bloch+dispersive, custom/uniform sources) was lifted by narrowing the message-branch conditions inside the single `_unsupported_adjoint_medium` raise and generalizing `_validate_supported_configuration`, so the AST raise-count dropped only 87 → 85 even though the differentiable forward surface grew substantially. The projected `≤ 76` assumed guard *deletion*; the realized route was branch-condition lifting. |
| after P5.2 (combinations) | **74 (measured)** | Projected `≤ 62`; realized `74`. As with P5.1, most P5.2 edges (nonlinear+dispersive, aniso+dispersive, aniso+σ_e, aniso+PML-overlap under CPML, modulated+dispersive/nonlinear, multi-frequency modulation, modulated-slab-CPML) were enabled by *narrowing branch conditions and composing coefficient paths*, not by deleting `raise` statements, so the AST count fell only 85 → 74. `runtime/materials.py` went 10 → 6 (not ≤ 3): the 6 that remain guard genuinely-unsupported combinations — nonlinear / full-aniso / modulated media under complex Bloch fields (need complex-field kernels), modulated + full-aniso (per-step 3×3 re-inversion), full-aniso + nonlinear cross-material defense, and full-aniso overlapping a split-field (non-CPML) PML. `media.py` retains 3 material-combination construction guards (nonlinear+aniso, modulated+aniso, modulated+σ_e) plus the dispersive-full-tensor and PerturbationMedium-full-tensor frequency-evaluation deferrals; all are physics-worded. Every remaining `media.py` / `runtime/materials.py` combination guard was reworded off "not implemented yet" / "in v1" in this phase. |
| after P5.3 (grid) | **74 (measured)** | Projected `≤ 58`; realized `74` (unchanged). As in P5.1/P5.2, P5.3 lifted the grid × feature coherence cases by generalizing *ValueError-gated / approximation* paths, not by deleting `NotImplementedError` raises, so the AST capability count did not move. Subpixel averaging and conformal PEC now scale their per-sub-sample offsets by the local Yee dual-cell width instead of the scalar `Scene.dx` (that `Scene.dx` nonuniform guard is a `ValueError`, never counted); the TFSF / mode-plane "locally uniform" hard rejects became bounded region-uniform `ValueError` contracts; and the soft-`PlaneWave` numerical-dispersion correction switched from global-minimum to launch-local spacing. None of these are `NotImplementedError` guards. The grid-related `NotImplementedError` raises that remain are out of P5.3 scope: `tidy3d.py` / `simulation.py` nonuniform-grid rejects belong to P5.6 (cross-solver parity) and the TFSF-slab `axis='z'` / slab-runtime rejects to P5.4/P5.5. Acceptance proof: `tests/validation/physics/test_autogrid_subpixel_thesis.py` shows `GridSpec.auto` + `SubpixelSpec(samples=(2,2,2), averaging="polarized")` reaches lower field error at fewer cells than uniform + subpixel on a high-contrast (`eps_r=12`) sphere. |
| after P5.5 (stubs) | **71 (measured)** | Projected `≤ 45`; realized `71` (74 → 71). The four P5.5 stubs each landed a runtime path rather than a large guard sweep: `sigma_m` folded into the H update with no pre-existing raise to delete; the `Graphene` interband Lorentz sheet-pole fit deleted the `media.py` interband raise (media 8 → 7); the non-periodic TFSF slab forward runtime deleted one `stepping.py` raise (6 → 5) and one `tfsf_state.py` raise (3 → 2); and the `LossyMetalMedium` SIBC runtime replaced its descriptor-only raise with a single physics-worded SIBC-configuration guard (compiler unchanged at 6). Net −3. Every remaining forward-path guard message now states a physical or mathematical reason: the compiler Tensor3x3-`mu_r`/`sigma_e`/`sigma_m` guard and the SIBC guard were reworded off "not implemented yet" / "v1" in this phase, and the phrase gate (below) is now live. |
| after P5.4 (Bloch broadband) | **66 (measured)** | Projected `≤ 50`; realized `66` (71 → 66). Pulsed (broadband) Bloch source injection deleted both `excitation/temporal.py` CW-only guards (2 → 0); the general grating TFSF slab runtime deleted one `stepping.py` raise (5 → 4) and one `tfsf_state.py` raise (2 → 1); the mixed Bloch/CPML single-PML-axis generalization deleted one `adjoint/core.py` raise (6 → 5). Net −5. The `excitation/` phrase-gate allowlist entry was removed: every remaining `excitation/` guard already states a physical reason (`currently supports only none, pml, pec, or pmc boundaries`, real-valued-eps mode caps, the Bloch 6-face split ill-posedness), none uses a deferral phrase. |
| after P5.6 (parity) | **63 (measured)** | Projected `≤ 33`; realized `63` (66 → 63). The Tidy3D export commits (PEC→PECMedium, `sigma_e`+dispersive→PoleResidue, Kerr/TPA→NonlinearSpec, diagonal/full anisotropy, `Medium2D`/`Graphene`/`LossyMetalMedium`, modulation, uniform custom poles + `PerturbationMedium`, `Ellipsoid`/`PolySlab` geometry + source/monitor kinds, nonuniform grids) lifted `adapters/tidy3d.py` capability guards from 13 → 10. The plan's `tidy3d 13 → ≤ 4` (≤ 6 in the plan body) target was **not** reached: the 10 that remain are the 3 unknown-kind geometry/source/monitor fallbacks (`851`/`1121`/`1309`), 4 genuine "Tidy3D has no such construct" guards (full-tensor+dispersive, anisotropic-magnetic, `chi2`/SHG, modulated+dispersive/nonlinear), and 3 spatially-varying architectural limits (per-cell modulation / custom pole / `PerturbationMedium` grids are defined relative to the owning `Structure` `Box` and are not resolvable into Tidy3D absolute coordinates at material-conversion time). Every one states a physical/architectural reason and none uses a deferral phrase, so `adapters/tidy3d.py` was **retired from the phrase-gate allowlist** this phase. **FDFD static parity (Kerr / `Tensor3x3` / magnetic / in-domain PEC) and FDFD nonuniform grids are deferred by user decision (2026-07-11)**: no `fdfd/` guard was removed; the four `fdfd/solver.py` guards were reworded to honest user-deferral statements ("… is deferred (user decision 2026-07-11): … Use FDTD, which …") off the former "in v1" / "not implemented yet" wording, and `fdfd/` **stays** in the phrase-gate allowlist with a comment recording the deferral. Coverage instead of static parity: every public medium capability now has a verified validation path enforced by `tests/validation/benchmark/test_media_validation_coverage.py` and recorded in `benchmark/RESULTS.md` (§ Validation coverage). |
| after P5.7–P5.9 | ≤ 25 | plan §5 global target |

Every remaining guard's message must state a physical or mathematical reason,
never "not implemented yet" / "not supported yet" / "in v1". As of P5.5 this is
**enforced** by `test_no_deferral_phrase_in_public_forward_path` in
`tests/api/public/test_guard_census.py`, which fails on any of those deferral
phrases in a public forward-path module (`media.py`, `compiler/`,
`fdtd/runtime/`, `fdtd/boundary/`, `scene.py`, `simulation.py`). Modules whose
wording a later phase owns carry an explicit allowlist entry naming that phase
(`fdfd/` → P5.6 user-deferred, `postprocess/` → P5.9, `fdtd/adjoint/` → P5.7+).
The `fdtd/excitation/` allowlist entry was retired in P5.4, and the
`adapters/tidy3d.py` entry was retired in P5.6: every excitation and Tidy3D
export guard now states a physical or architectural reason directly. `fdfd/`
remains allowlisted because FDFD static parity is deferred by user decision
(2026-07-11), not because its guards use a deferral phrase — they were reworded
to explicit user-deferral statements.

## Per-cluster capability-guard counts (baseline)

```
14  witwin/maxwell/media.py                    6  witwin/maxwell/fdtd/runtime/stepping.py
13  witwin/maxwell/adapters/tidy3d.py          4  witwin/maxwell/fdfd/solver.py
10  witwin/maxwell/fdtd/runtime/materials.py   3  witwin/maxwell/fdtd/excitation/tfsf_state.py
 7  witwin/maxwell/compiler/materials.py       2  witwin/maxwell/fdtd/excitation/temporal.py
 7  witwin/maxwell/fdtd/adjoint/core.py        2  witwin/maxwell/fdtd/excitation/injection.py
 7  witwin/maxwell/fdtd/excitation/modes.py    2  witwin/maxwell/postprocess/stratton_chu.py
 6  witwin/maxwell/fdtd/adjoint/bridge.py      1  simulation.py / fdfd/adjoint/bridge.py /
                                                  boundary/cpml.py / scattering_parameters.py
```

## Per-cluster capability-guard counts (after P5.2 — measured, total 74)

```
13  witwin/maxwell/adapters/tidy3d.py          6  witwin/maxwell/fdtd/runtime/materials.py
 8  witwin/maxwell/media.py                    6  witwin/maxwell/fdtd/runtime/stepping.py
 7  witwin/maxwell/fdtd/excitation/modes.py    4  witwin/maxwell/fdfd/solver.py
 6  witwin/maxwell/compiler/materials.py       3  witwin/maxwell/fdtd/excitation/tfsf_state.py
 6  witwin/maxwell/fdtd/adjoint/bridge.py      2  witwin/maxwell/fdtd/excitation/injection.py
 6  witwin/maxwell/fdtd/adjoint/core.py        2  witwin/maxwell/fdtd/excitation/temporal.py
                                               2  witwin/maxwell/postprocess/stratton_chu.py
                                               1  simulation.py / fdfd/adjoint/bridge.py /
                                                  scattering_parameters.py
```

P5.2 deltas from the baseline: `media.py` 14 → 8, `runtime/materials.py` 10 → 6,
`compiler/materials.py` 7 → 6, `tidy3d.py` 13 → 13 (untouched; P5.6),
`adjoint/*` 13 → 12. The material combination matrix that documents the true
post-P5.2 composability is `tests/materials/combinations/test_combination_matrix.py`.

## Per-cluster capability-guard counts (after P5.3 — measured, total 74)

Unchanged from the P5.2 block above (every file keeps the same count): P5.3 removed
no `NotImplementedError` raise, so the distribution is identical. The relevant P5.3
files carry only `ValueError`/approximation paths, not capability guards:
`compiler/materials.py` (subpixel + conformal PEC per-node offsets), `scene.py`
(the `Scene.dx` scalar-spacing `ValueError`, a contract not a capability guard),
`fdtd/excitation/tfsf_common.py` and `modes.py` (region-uniform bounds), and
`fdtd/excitation/spatial.py` / `injection.py` (launch-local soft-`PlaneWave` spacing).
The three `fdtd/excitation/tfsf_state.py` capability guards and the two
`temporal.py` Bloch guards are P5.4/P5.5 items, not grid coherence. The acceptance
proof that `GridSpec.auto` + subpixel now compose to lower field error at fewer
cells is `tests/validation/physics/test_autogrid_subpixel_thesis.py`.

## Per-cluster capability-guard counts (after P5.5 — measured, total 71)

```
13  witwin/maxwell/adapters/tidy3d.py          5  witwin/maxwell/fdtd/runtime/stepping.py
 7  witwin/maxwell/fdtd/excitation/modes.py    4  witwin/maxwell/fdfd/solver.py
 7  witwin/maxwell/media.py                    2  witwin/maxwell/fdtd/excitation/injection.py
 6  witwin/maxwell/compiler/materials.py       2  witwin/maxwell/fdtd/excitation/temporal.py
 6  witwin/maxwell/fdtd/adjoint/bridge.py      2  witwin/maxwell/fdtd/excitation/tfsf_state.py
 6  witwin/maxwell/fdtd/adjoint/core.py        2  witwin/maxwell/postprocess/stratton_chu.py
 6  witwin/maxwell/fdtd/runtime/materials.py   1  simulation.py / fdfd/adjoint/bridge.py /
                                                  scattering_parameters.py
```

P5.5 deltas from the after-P5.2/P5.3 block: `media.py` 8 → 7 (Graphene interband
raise deleted by the Lorentz sheet-pole fit), `fdtd/runtime/stepping.py` 6 → 5 and
`fdtd/excitation/tfsf_state.py` 3 → 2 (both from the non-periodic TFSF slab forward
runtime). `sigma_m` added no guard (it mirrors the P1 `sigma_e` Ca/Cb fold, which
never had a raise). The `LossyMetalMedium` SIBC runtime replaced its descriptor-only
raise with one contextual SIBC-configuration guard, so `compiler/materials.py` stays
at 6. Acceptance proofs: `tests/materials/conductive/test_fdtd_magnetic_conductive.py`
(sigma_m slab absorption `rel_err < 0.02`), `tests/materials/sheet/test_fdtd_graphene.py`
(interband Kubo fit `< 0.03`), `tests/validation/physics/test_lossy_metal_sibc.py`
(SIBC reflection `rel_err < 0.05` at `>= 10x` fewer cells), and
`tests/sources/tfsf/test_fdtd_grating_tfsf.py` (non-periodic slab forward confines the
total field). The phrase gate `test_no_deferral_phrase_in_public_forward_path` is now
live in the same census test file.

## Per-cluster capability-guard counts (after P5.4 — measured, total 66)

```
13  witwin/maxwell/adapters/tidy3d.py          4  witwin/maxwell/fdfd/solver.py
 7  witwin/maxwell/fdtd/excitation/modes.py    4  witwin/maxwell/fdtd/runtime/stepping.py
 7  witwin/maxwell/media.py                    2  witwin/maxwell/fdtd/excitation/injection.py
 6  witwin/maxwell/compiler/materials.py       2  witwin/maxwell/postprocess/stratton_chu.py
 6  witwin/maxwell/fdtd/adjoint/bridge.py      1  witwin/maxwell/fdtd/excitation/tfsf_state.py
 6  witwin/maxwell/fdtd/runtime/materials.py   1  simulation.py / fdfd/adjoint/bridge.py /
 5  witwin/maxwell/fdtd/adjoint/core.py           scattering_parameters.py
```

P5.4 deltas from the after-P5.5 block: `fdtd/excitation/temporal.py` 2 → 0 (both
CW-only Bloch-source guards deleted by the pulsed/broadband Bloch injection),
`fdtd/runtime/stepping.py` 5 → 4 and `fdtd/excitation/tfsf_state.py` 2 → 1 (the
grating TFSF slab runtime generalized to any normal axis), and
`fdtd/adjoint/core.py` 6 → 5 (the mixed Bloch/CPML single-PML-axis generalization
lifted the x/y-Bloch+z-PML-only adjoint guard). The two `tfsf_state.py` Bloch
contract guards became one; the surviving `TFSF slab mode is required for
Bloch-boundary TFSF injection` guard stays a contract (a 6-face total/scattered
split is ill-posed under transverse periodicity). Acceptance proofs:
`tests/validation/physics/test_bloch_broadband_vs_cw.py` (a single broadband
GaussianPulse run reproduces the per-frequency CW transmission of a normal-incidence
periodic metasurface within 2%, plus oblique complex-field Bloch injection confines
the total field per frequency), `tests/validation/physics/test_dispersive_metasurface_bloch.py`
(a patterned Lorentz metasurface runs forward under oblique Bloch and matches its
periodic-equivalent unit-phase reference to float32 round-off; FDFD is unavailable
as a cross-check because it rejects periodic/Bloch faces), and the existing
`tests/sources/tfsf/test_fdtd_grating_pulsed_bloch.py` and
`tests/monitors/diffraction/test_diffraction_monitor.py` (order decomposition
conserves plane flux within 5% and agrees with the independent Yee flux integrator
within 8%; absolute per-order `T + R = 1` is a documented normalization limit, kept
as a strict-`False` xfail tripwire).

## Per-cluster capability-guard counts (after P5.6 — measured, total 63)

```
10  witwin/maxwell/adapters/tidy3d.py          4  witwin/maxwell/fdfd/solver.py
 7  witwin/maxwell/fdtd/excitation/modes.py    4  witwin/maxwell/fdtd/runtime/stepping.py
 7  witwin/maxwell/media.py                    2  witwin/maxwell/fdtd/excitation/injection.py
 6  witwin/maxwell/compiler/materials.py       2  witwin/maxwell/postprocess/stratton_chu.py
 6  witwin/maxwell/fdtd/adjoint/bridge.py      1  witwin/maxwell/fdtd/excitation/tfsf_state.py
 6  witwin/maxwell/fdtd/runtime/materials.py   1  simulation.py / fdfd/adjoint/bridge.py /
 5  witwin/maxwell/fdtd/adjoint/core.py           postprocess/scattering_parameters.py
```

P5.6 deltas from the after-P5.4 block: `adapters/tidy3d.py` 13 → 10 (the export
commits mapped PEC, `sigma_e`+dispersive, Kerr/TPA, anisotropy, sheet/2D metals,
modulation, uniform custom poles / `PerturbationMedium`, extra geometry/source/
monitor kinds, and nonuniform grids to real Tidy3D constructs). Every other
cluster is unchanged: **FDFD static parity (`fdfd/solver.py` = 4) and FDFD
nonuniform grids are deferred by user decision (2026-07-11)**, so the four
`fdfd/solver.py` guards (Kerr, full `Tensor3x3` ε, magnetic media/dispersion,
in-domain PEC) were reworded to honest user-deferral statements but kept, and
`simulation.py`'s FDFD-nonuniform reject stays. The remaining 10
`adapters/tidy3d.py` guards split into 3 unknown-kind geometry/source/monitor
fallbacks, 4 "Tidy3D has no such construct" guards (full-tensor+dispersive,
anisotropic-magnetic, `chi2`/SHG, modulated+dispersive/nonlinear), and 3
spatially-varying architectural limits (per-cell modulation / custom pole /
`PerturbationMedium` grids defined relative to the owning `Structure` `Box`);
all state a physical/architectural reason, so `adapters/tidy3d.py` left the
phrase-gate allowlist. Acceptance proof that every public medium has a validation
path: `tests/validation/benchmark/test_media_validation_coverage.py` (discovers
each `is_*`/`has_*` capability flag on `media.py`, fails on any with no declared
path, and verifies each declared Tidy3D export / analytic-reference claim), with
the per-medium path table recorded in `benchmark/RESULTS.md` (§ Validation
coverage). New P3-media Tidy3D benchmark scenarios registered in `SCENARIOS`:
`debye_slab`, `sigma_e_drude_slab`, `anisotropic_slab`, `kerr_slab`,
`modulated_slab`, `graphene_sheet` (under `benchmark/scenes/media/`).
