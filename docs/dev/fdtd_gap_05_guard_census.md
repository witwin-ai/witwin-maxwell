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

Two contract guards currently say "not implemented yet" in their message
(`tidy3d.py` magnetic pair); their wording must change to state the physical reason
("Tidy3D has no magnetic-material model") as part of P5.6, but they are contract
guards regardless of wording.

## Capability-guard budget schedule

| Milestone | Budget | Notes |
|---|---|---|
| P5.0 baseline (this commit) | 87 | measured |
| after P5.1 (adjoint) | **85 (measured)** | multi-source + `normalize_source` single-source raises deleted (bridge+core 13 → 11). Most P5.1 capability (σ_e, χ²/TPA, full-aniso ε, Bloch+dispersive, custom/uniform sources) was lifted by narrowing the message-branch conditions inside the single `_unsupported_adjoint_medium` raise and generalizing `_validate_supported_configuration`, so the AST raise-count dropped only 87 → 85 even though the differentiable forward surface grew substantially. The projected `≤ 76` assumed guard *deletion*; the realized route was branch-condition lifting. |
| after P5.2 (combinations) | **74 (measured)** | Projected `≤ 62`; realized `74`. As with P5.1, most P5.2 edges (nonlinear+dispersive, aniso+dispersive, aniso+σ_e, aniso+PML-overlap under CPML, modulated+dispersive/nonlinear, multi-frequency modulation, modulated-slab-CPML) were enabled by *narrowing branch conditions and composing coefficient paths*, not by deleting `raise` statements, so the AST count fell only 85 → 74. `runtime/materials.py` went 10 → 6 (not ≤ 3): the 6 that remain guard genuinely-unsupported combinations — nonlinear / full-aniso / modulated media under complex Bloch fields (need complex-field kernels), modulated + full-aniso (per-step 3×3 re-inversion), full-aniso + nonlinear cross-material defense, and full-aniso overlapping a split-field (non-CPML) PML. `media.py` retains 3 material-combination construction guards (nonlinear+aniso, modulated+aniso, modulated+σ_e) plus the dispersive-full-tensor and PerturbationMedium-full-tensor frequency-evaluation deferrals; all are physics-worded. Every remaining `media.py` / `runtime/materials.py` combination guard was reworded off "not implemented yet" / "in v1" in this phase. |
| after P5.3 (grid) | ≤ 58 | subpixel/autogrid coherence |
| after P5.4 (Bloch broadband) | ≤ 50 | temporal/stepping/tfsf Bloch guards |
| after P5.5 (stubs) | ≤ 45 | SIBC/graphene/sigma_m/TFSF-slab |
| after P5.6 (parity) | ≤ 33 | tidy3d 13 → ≤ 4 capability; fdfd static parity |
| after P5.7–P5.9 | ≤ 25 | plan §5 global target |

Every remaining guard's message must state a physical or mathematical reason,
never "not implemented yet" — enforced by the phrase gate in the same test file
once P5.5 lands.

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
