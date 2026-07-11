# FDTD Gap P5 — Implementation Status (Final)

Date: 2026-07-11
Status: **executed** — all nine sub-phases landed on master (P5.1–P5.9), with two explicitly
deferred scopes and every deviation from the plan's acceptance numbers recorded below.
Plan: `fdtd_gap_05_functional_completeness.md`. Census details: `fdtd_gap_05_guard_census.md`.

## Execution summary

| Phase | Outcome | Key acceptance evidence |
|---|---|---|
| P5.0 census | done | 107 raises = 20 contract + 87 capability; AST ratchet gate committed (`test_guard_census.py`) |
| P5.1 adjoint | done | multi-source, σ_e, χ²/TPA, Tensor3x3 ε, Bloch+dispersive, custom/uniform sources differentiable; FD checks 1.7e-5…4.2e-4; 2-port S-param optimization −34% strictly monotonic over 24 iters |
| P5.2 combinations | done | 9 edges landed; combination-matrix harness (68 tests: 19 supported / 8 physics-worded deferrals / 1 ill-posed); birefringent dispersive slab within 1.7% (bar 2%) |
| P5.3 grid | done | subpixel + conformal PEC on nonuniform grids (uniform bit-exact); AutoGrid thesis proven: auto+subpixel 175k cells → err 0.022 vs uniform 284k cells → 0.117 vs 3M-cell reference |
| P5.4 Bloch broadband | done (5/6) | derived plane-wave normalization (empirical 0.958/1.65e-10 constants removed); pulsed Bloch gratings; broadband-vs-CW within 0.9/1.1/0.05%; mixed Bloch+CPML any single-PML axis |
| P5.5 stubs | done | σ_m 0.64% (bar 2%); Graphene interband <3%; SIBC 2.3–3.8% (bar 5%) at >10× fewer cells; TFSF slab forward runtime; forward-path deferral-phrase gate |
| P5.6 parity | done | per-medium validation coverage gate (introspects capability flags); 6 P3-media benchmark scenarios; PEC-exports-as-vacuum bug fixed; conductivity unit bug (1e6×) fixed |
| P5.7 modal | done | complex/lossy n_eff vs analytic 4e-5 (bar 1e-3); ring S21 vs CMT ~1e-7 (bar 3%); modes.py guards 7 → 3 (plan target met) |
| P5.8 perf | done | CUDA-graph capture for dispersive/Kerr/complex-Bloch/reference-TFSF, bit-exact per class; measured 8.25×/step on launch-bound dispersive AutoGrid (bar ≥2×); coverage-matrix test |
| P5.9 postprocess | done | layered-exterior RCS 4.03% (bar 5%); curved closed surfaces (spherical Huygens <1e-3); broadband incident_power="auto" |

Full suite on final master: **1107 passed, 65 skipped, 2 xfailed** (baseline before P5: 741 passed —
net +366 tests). Working tree clean at `5e9b6b6`.

## Plan criteria not met, honestly

1. **Guard budget 87 → 58, not ≤25.** Most capability was delivered by narrowing branch
   conditions inside shared raise statements rather than deleting raises, so the AST raise-count
   under-reports the lifted surface. Remaining 58: 10 tidy3d (genuine no-equivalents + honest
   narrows), 6 FDFD (deferred, below), and 42 physics-worded or honestly-deferred cases. The
   forward-path phrase gate enforces wording; the ratchet keeps the count monotone.
2. **Absolute diffraction efficiency (P5.4) reverted.** The two-plane TFSF slab injector forms a
   Fabry–Pérot cavity that biases absolute T by 4–36% (proven over 19 diagnostics); relative
   per-order reflection matches TMM at 0.6–3%. Prerequisite (single-plane injector or per-order
   forward/backward decomposition) is filed as a follow-up task.
3. **χ² SHG absolute efficiency (P5.2) asserted as ratio physics, not 5% absolute.** FDTD
   numerical dispersion shifts Maker fringes in a small test scene; the committed test pins
   phase-matched linear growth (1.888 vs 2.0), a 23× coherence-length null, and the sinc ratio
   (0.591 vs 0.637). Documented in the test.
4. **Full-aniso adjoint e2e FD at 3e-2, not 1e-3.** Float-level torch-VJP vs native-kernel
   divergence (~7e-8/step) compounds through the reverse rollout; the off-diagonal coupling
   transpose is separately pinned at 1e-5 by a replica-vs-kernel test.
5. **tidy3d guards 13 → 10, not ≤6.** Remaining are genuine no-equivalents (magnetic pair, χ²,
   spatial custom poles at conversion time) plus honest narrows; each message states why.

## Explicit deferrals

- **FDFD static parity + FDFD nonuniform grids** — deferred by user decision (2026-07-11).
  The 6 fdfd guards are worded as honest deferrals; the coverage gate counts FDFD validation
  only where FDFD already supports the medium.
- **RF wave ports / lumped elements, EME/heat/charge solvers, multi-GPU, float64** — plan §7
  non-goals, untouched.

## Pre-existing bugs found and filed during execution

- Source-eps pullback collocation error (~15%) when a source overlaps the trainable design
  region (affects pre-P5 PointDipole path too) — task chip filed.
- Tidy3D single-pole Lorentz export uses 2× linewidth; Debye τ off by 2π (Drude/Sellmeier
  correct) — task chip filed by the P5.6 acceptance agent.
- Fixed during execution: full-aniso checkpointed forward dropped the off-diagonal correction
  (~4.7%); anisotropic media diverged inside CPML; PEC exported to Tidy3D as vacuum;
  non-dispersive conductivity export off by 1e6; `CustomSourceTime` adjoint precision floor
  documented (~few×1e-3, pre-existing).

## Sub-phase commit map

P5.1 `c2472bb..7b6ede6` + `3089cfc` · P5.2 `7a6b633..6ec1193` · P5.3 `2bd1c62..08ee4ef` ·
P5.5 `1a2b281..dfd4278` · P5.4 `02e25d3..6d9cca2` · P5.6 `21d0dce..944835f` ·
P5.9 merge `da668a1` · P5.8 merge `bc0c29b` · P5.7 merge `c8867be` · reconciliation `5e9b6b6`.
