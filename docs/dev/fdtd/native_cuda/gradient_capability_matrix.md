# FDTD Adjoint — Native Reverse Gradient Capability Matrix

This is the acceptance map for the P6 effort that moved the FDTD adjoint per-step
reverse **math** off Torch and onto fused native CUDA kernels. Python still
orchestrates the backward sweep (segment replay, kernel launches, gradient
accumulation) and a small, calibrated set of once-per-step Torch helpers stays
Torch (the "residual" column). Everything in the per-cell reverse recurrence —
electric→mid-H adjoint, mid-H→pre-step-E adjoint plus the eps/chi3 gradient, the
CPML psi pullback, the ADE-current VJP, the collocation transpose, and the TFSF
auxiliary reverse — runs inside the compiled kernels.

The backend is selected per step by `WITWIN_MAXWELL_FDTD_ADJOINT_BACKEND`
(`auto` / `native` / `torch_reference` / `torch_vjp`). In `auto` mode a qualifying
CUDA scene prefers the native runner; otherwise it falls to the analytic Torch
reference, otherwise the torch-autograd VJP.

## Native reverse classes

Each of these differentiable scene classes has a fused native CUDA reverse runner
(`witwin/maxwell/fdtd/adjoint/native.py`) whose per-cell math is bit-for-bit
equivalent to the analytic Torch reference it mirrors.

| Scene class | `auto` backend label | Reverse-math kernels (native) |
| --- | --- | --- |
| Standard (open / PEC, non-CPML) | `native_standard` | `reverseElectricAdjointToH*Standard3D`, `reverseMagneticAdjointToE*Standard3D`, `reverseMagneticAdjointDecay` |
| CPML (absorbing) | `native_cpml` | `reverseElectricComponent*Cpml3D`, `reverseMagneticComponent*Cpml3D`, `accumulate{Backward,Forward}DiffAdjoint` |
| Static conductive (`sigma_e`) + CPML | `native_conductive` | `reverseElectricComponent*CpmlConductive3D` + linear CPML magnetic reverse |
| Instantaneous Kerr (`chi3`) + CPML | `native_kerr` | `reverseElectricComponent*CpmlKerr3D`, `collocateFieldSquare`, `collocationTranspose` |
| Full (off-diagonal) anisotropic + CPML | `native_full_aniso` | `fullAnisoCurlAdjoint` + backward-diff fold + linear CPML core |
| Bloch (complex split-field) | `native_bloch` | `reverseElectricAdjointToH*Bloch3D`, `reverseMagneticAdjointToE*Bloch3D` |
| Bloch + electric-dispersive | `native_bloch_dispersive` | complex Bloch core + `reverse{Debye,Drude,Lorentz}Current`, `reverseDispersiveCorrection` (real+imag) |
| Electric-dispersive (ADE, standard or CPML base) | `native_dispersive` | standard/CPML core + `reverse{Debye,Drude,Lorentz}Current`, `reverseDispersiveCorrection` |
| TFSF (total-field/scattered-field) | `native_tfsf` | standard/CPML core + `reverseTfsfAuxiliary*1D`, `accumulateTfsf*SampleAdjoint3D` |

Additive **magnetic conductivity** (`sigma_m`) carries no separate ADE state — it
folds into the semi-implicit magnetic decay/curl coefficient — so a `sigma_m`
medium routes through `native_standard` / `native_cpml` unchanged and is covered
under the standard/CPML rows.

### Residual Torch on the native hot path (the calibrated whitelist)

The per-step reverse region legitimately dispatches a small set of Torch `aten.*`
ops for orchestration and replay; the reverse **math** is never one of them. These
are the "same bar" remainders every landed P6 item reports:

- **mid-H magnetic replay** (`_forward_magnetic_fields` /
  `_forward_magnetic_fields_complex`) — reconstructs the post-magnetic H the
  forward electric update consumed.
- **electric-dispersive ADE replay** (`_advance_dispersive_state`) — reconstructs
  the post-update polarization currents (`exp`/`pow`/`clamp`/`sum` on the ADE
  coefficients).
- **`dynamic_electric_curls`** — casts the base curl coefficient by the eps leaf.
- **reverse-context allocation** (`allocate_cpml_reverse_context`) and the
  **source-term VJP** (`_accumulate_source_term_gradients`, per-step
  source-region scatter/`1/eps`).

The exact whitelist (overload-packet names) is pinned in
`tests/gradients/test_fdtd_adjoint_p6_acceptance.py::_REVERSE_REGION_ATEN_WHITELIST`.
It deliberately excludes the allocation/masking ops (`aten.zeros`,
`aten.new_zeros`, `aten.bitwise_*`, `aten.where`, `aten.slice_backward`, ...) that
only the Torch analytic reverse math emits, so a native runner silently regressing
to Torch trips the `__torch_dispatch__` gate.

## Honestly-reported remainder (not native)

These differentiable classes have **no** fused native reverse runner yet; the
backend column is the truthful `auto` selection.

| Scene class | `auto` backend | Why it is not native |
| --- | --- | --- |
| General nonlinear (`chi2` / two-photon absorption) | `torch_vjp` | the general nonlinear forward rewrites both the electric decay **and** curl from the pre-update fields; that fused dynamic-coefficient VJP is not a native kernel. |
| Magnetic-dispersive (`mu`-pole ADE) | `python_reference_*` (analytic) / `torch_vjp` under Bloch | the magnetic ADE correction/state VJP is not nativized; the electric-dispersive runner qualifier rejects `magnetic_dispersive_enabled`. |
| Grating-slab TFSF (Bloch + CPML complex) | `python_reference_grating_tfsf` | analytic Torch reference only; no native grating-TFSF reverse kernel yet. |
| Mixed Bloch + CPML (complex, general) | `torch_vjp` | no fused analytic complex-CPML reverse. |

These are pinned so the native/remainder boundary stays explicit:
`test_general_nonlinear_reverse_remains_torch_vjp` (this module),
`test_chi2_reverse_routes_through_torch_vjp` and
`test_scene_with_magnetic_dispersive_medium_gradient_matches_fd`
(`test_fdtd_adjoint_materials.py`), and
`test_general_mixed_bloch_cpml_remains_torch_vjp_remainder`
(`test_fdtd_adjoint_b_complex_classes.py`).

## Acceptance coverage

| Guarantee | Test |
| --- | --- |
| Hot-path gate: native backend + `witwin_maxwell_fdtd_cuda.*` reverse kernels present + only whitelisted `aten.*` ops in the reverse region, per class | `test_fdtd_adjoint_p6_acceptance.py::test_hotpath_reverse_region_is_native` |
| Whitelist is tight (Torch reference trips it) | `...::test_torch_reference_reverse_region_leaves_the_whitelist` |
| Step-level native == analytic reference parity matrix | `...::test_native_reverse_matches_reference` |
| Additive `sigma_m` design gradient == finite difference | `...::test_additive_sigma_m_gradient_matches_fd` |
| Multi-frequency + dispersive design gradient == finite difference | `...::test_multi_frequency_dispersive_gradient_matches_fd` |
| Per-kernel native == Torch-reference parity | `tests/fdtd/cuda/test_cuda_adjoint_parity.py` |
| End-to-end design-gradient == FD per class | `tests/gradients/test_fdtd_adjoint_rigorous.py`, `test_fdtd_adjoint_materials.py` |

## Performance

`scripts/perf_native_adjoint_reverse.py` times one reverse step under the native
vs Torch-reference backend per class (CUDA events, quiet GPU, RTX 5080, 300 iters):

| class | native ms/step | torch ms/step | speedup |
| --- | ---: | ---: | ---: |
| standard | 2.07 | 11.78 | 5.7x |
| cpml | 6.10 | 21.98 | 3.6x |
| conductive | 7.96 | 20.36 | 2.6x |
| kerr | 4.96 | 23.84 | 4.8x |
| full_aniso | 3.82 | 21.84 | 5.7x |
| bloch | 3.87 | 30.00 | 7.8x |
| dispersive_standard | 2.39 | 10.07 | 4.2x |
| dispersive_cpml | 3.18 | 16.30 | 5.1x |
| tfsf | 4.19 | 13.53 | 3.2x |
| bloch_dispersive | 7.93 | 34.48 | 4.4x |

The native reverse math replaces the dominant Torch reverse cost with fused
kernels; the residual Torch replay/orchestration is shared by both backends, so
the ratio isolates the reverse-math speedup (2.6x–7.8x here). Absolute numbers are
GPU- and contention-dependent; run the harness on a quiet GPU to reproduce.
