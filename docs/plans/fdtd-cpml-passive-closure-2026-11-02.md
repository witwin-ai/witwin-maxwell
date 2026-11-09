# Passive CPML closure (option B)

Status: approved by the coordinator on 2026-11-02, with additions. The user approved option B the same
day (relayed by the coordinator). B0 is in progress.

## 1. Problem and decision

The CPML auxiliary-memory closure is not passive for content below the vacuum light line at
the CPML face. Below-light-line content is a slab-guided or waveguide mode that is evanescent
along the PML normal. Measured with the native CUDA update on harmonic basis states, so the
result does not depend on any CPU reimplementation:

- Real-periodic slab cell (1.28 m period, eps_r 3 slab, 12-layer z-CPML, alpha = 0, kappa = 1):
  - The TM first Floquet order has |lambda| = 1.000135 at 0.2318 GHz, a growth rate of
    8.1e6 s^-1.
  - Energy accounting puts all the gain in the z-CPML psi pair: magnetic psi supplies
    +7.09e-4 W per step and electric psi removes 4.38e-4.
  - The growth persists at dt/2 and dt/4 at the same rate per unit time, so the instability is
    semi-discrete in space, not a time-stepping artefact.
- PEC- and PMC-closed transverse cells grow too. The half-wave mode at 0.117 GHz has
  rho - 1 = 5.6e-5 per step.
- Complex fixed-k Bloch runs with the legacy profile (alpha_max = 0) have one-step rho - 1 up to
  1.3e-6 on the open cell and 7.4e-5 with lumped loads. The current default (CFS, alpha =
  2*pi*eps0*f_min) is passive but costs about 0.4 point of 35-degree TE transmission. Half of
  that shift fails the 120k-step LC energy gate at 60 degrees.

Option A, a profile-only fix, was rejected. Only CFS profiles with the conductivity raised about
17x are passive on the real-periodic block. Their normal-incidence reflection is 0.6-20% in
amplitude, against about 2e-5 for today's real-field default. The required strength also does
not follow from the geometry, because PEC/PMC cells need more. Evidence:
`<evidence-root>/cfs-cpml-c2-2026-11-01/rcpml/` and
`<evidence-root>/instamax-line-modulated-growth-20261101/`.

**Decision:** replace the psi recurrence with a closure that has a discrete energy (storage plus
dissipation) certificate. One closure law covers real-periodic, PEC/PMC-closed, open and complex
fixed-k CPML. Reflection stays CPML-class: no worse than the current real-field CPML at
0.3-3 GHz.

**Interim (no default change):** long-horizon closed-transverse workflows use the opt-in graded
`absorber` at 24-40 layers, per `docs/memory/fdtd-real-periodic-cpml-growth-2026-11-01.md`.
F5 production data waits for this work.

## 2. Phases

### B0: CPU semi-discrete model and closure selection (Codex, no GPU)

1. Build a numpy model of the reduced TM/TE block:
   - one x cell, periodic or PEC/PMC y walls, z-CPML;
   - harmonic basis in y, the same Yee staggering, and coefficient vectors read from
     `boundary/cpml.py`;
   - reproduce the native matrix of the evidence directory (`native_matrix.py`) to within 1e-8
     in |lambda| for the default real profile. That reproduction is the validity gate for the
     model; Claude supplies the native numbers.
2. Derive the discrete energy identity of the current closure in this model, and show where the
   positive psi work arises. Report it as a short derivation in
   `docs/memory/fdtd-cpml-passive-closure-derivation-2026-11-02.md`.
3. Evaluate candidate closures in the model. For each, give the discrete storage functional and
   prove or numerically certify that its increment is non-positive for source-free updates:
   - (c1) time-centered (trapezoidal) psi ADE with E/H memory matched in time level;
   - (c2) an energy-stable ADE stretched-coordinate PML of the Appelö–Hagstrom / Duru–Kreiss
     family, with whatever extra auxiliary field it needs;
   - (c3) any variant that keeps the existing one-memory-per-stretched-derivative layout and
     reaches passivity through a provably passive coefficient or ordering change. This is
     preferred if it exists, because it keeps the memory layout and the core-path mirror small.
4. Select by these criteria, in priority order:
   - (i) max oscillatory |lambda| <= 1 + 1e-12 in float64 over the B0 sweep (section 3, G1);
   - (ii) the 1D discrete normal and oblique reflection, computed in the model, at most the
     current CPML at 0.3-3 GHz for 12 layers;
   - (iii) the smallest state and kernel change.
5. B0 output: the chosen closure with exact update equations and float operation order, the
   coefficient formulas, the storage functional, and the model scripts under
   `scripts/dev/fdtd/cpml_passive/`. **Coordinator and Claude review B0 before B1 starts.**

### B1: CUDA implementation (Codex; Claude runs every GPU gate)

- Kernels:
  - electric.cu: the standard, slab/compact, modulated and region paths;
  - magnetic.cu: the standard and compact paths;
  - aniso.cu: full-anisotropic CPML;
  - adjoint.cu and `fdtd/adjoint/core.py`: reverse psi, exact discrete adjoint of the new
    closure;
  - the complex-field (real/imag bank) variants.
- The profile builder `_cpml_profile_1d` and `cpml_line_profiles` (the fixed-k incident line,
  which must use the same closure in its 1D form) get the new coefficient vectors.
- Remove the old recurrence; no parallel implementation stays in the main path. The legacy
  coefficient presets (`alpha_max = 0` and so on) remain as profile choices under the new
  closure.
- Checkpoint and resume, CUDA-graph capture and the slab memory layouts carry any new state.
- **ABI:** bump the main-extension marker to the next free number at merge time, per the
  coordinator's rule. The core extension (core_abi_*) is untouched.
- No hunks in `fdtd/ports.py` or `fdtd/runtime/stepping.py` beyond what the closure strictly
  needs. Any such hunk goes to PCB (coax-port-bitwise-baseline) for acknowledgement before
  merge.

### B2: qualification, movers and merge (Claude)

Section 3 gates, section 4 movers plan, then the downstream notice.

## 3. Gates

All GPU gates run on GPU 0 under the shared rule: at most 4 GiB per process, a free-memory
check first, and at most 2 concurrent processes.

- **G1, spectral.** The native-matrix gate (evidence `native_matrix.py`, generalized as
  `scripts/dev/fdtd/cpml_passive/native_block.py`) must show max oscillatory |lambda| <= 1 + 2
  float32 ULP over:
  - periodic transverse orders 1-3 and PEC/PMC orders 1-3;
  - TE and TM;
  - dt, dt/2 and dt/4;
  - transverse and normal h refinement;
  - 6, 12 and 24 layers;
  - the profile presets (default real, default complex fixed-k CFS, explicit alpha_max = 0);
  - complex fixed-k at 0/10/30/45/60/75 degrees.
  - Also a larger one-step cell, or a long-run spectral estimate (power iteration over 10^5
    native steps), so that bound modes of large cells cannot be missed. The small one-step cell
    missed the 60-degree LC case.
- **G2, energy.** The 120k-step `bloch_lumped_acceptance/energy_balance.py` at 10/30/45/60/75
  degrees, all loads: LC one-sided slope bound 4e-7, lossy decay 1e-6, carrier-CW power
  balance.
- **G2b, slab-guided TM mode under complex fixed-k (added 2026-11-02).** Use the Bloch Airy scene
  (`bloch_floquet_acceptance`, TM εr 4 slab, k∥ 0.55 k0) driven by the cutoff pulse (0.825 f0,
  fwidth 0.18 f0).
  - The 3D field energy must be non-growing over 320k steps after source-off. Today it grows at
    +2.5e-6 per step.
  - The 0.6 and 0.7 f0 r/t must then meet the cutoff gate's absolute bound (0.005). Today the error
    is about 0.013.
  - B0b-1 (M-PML) must pass this case.
- **G3, real-field growth.** Run `growth.py` (evidence directory) on `modulated_slab` and
  `modulated_slab_phase` over 7.4 µs:
  - depth 0.2/0.1/0.05/0.01, ordinary zero, forced zero, and the seeded static and zero paths;
  - pass when the late RMS variation is <= 2% and |gamma| <= 2e3 s^-1;
  - include the PEC and PMC variants.
- **G4, closed-box energy including psi storage.**
  - Closed PEC box with CPML on one face: run source-free from random fields for 10^5 steps.
  - The discrete total storage (fields plus the closure's storage functional) must be
    non-increasing step by step, to float32 round-off: each increment <= 1e-6 W0.
  - The dissipation must equal the storage decrease to 1e-5 W0, summed over the run.
- **G5, reflection.** Normal and oblique (35 and 60 degree) reflection at 0.3, 0.6, 1, 2 and
  3 GHz for 12 layers must be <= the current real-field CPML (2e-5 normal at 20 cells per
  wavelength; harness `rcpml/reflection_real.py`, generalized). The existing native CPML
  reflection gate must also pass.
  - **Evanescent and near-cutoff absorption.** This is where alpha-type fixes trade accuracy.
  - It must be no worse than the current CPML for:
    - a PEC waveguide terminated in CPML, driven below its cutoff (evanescent mode at the face);
    - a grazing 80-degree case.
- **G6, accuracy.**
  - Copper-foil oblique: TE and TM <= 3%, which removes the strict TE xfail.
  - The εr4 sphere ladder and the basic external-reference rows unchanged within the existing
    tolerances.
  - The oblique Bloch Airy gates (CW and pulsed) pass as merged.
- **G7, adjoint.** The FDTD adjoint finite-difference checks with CPML (gradients suite, Bloch
  and mixed-boundary cases) pass with unchanged tolerances.
- **G8, regression.**
  - The whole `tests/` sweep;
  - BITID baselines per section 4;
  - test_guard_census 7/7;
  - graph/eager and disk-resume equality for the new state.

## 4. Movers, BITID and downstream

- **Every CPML run moves.** Real fields, complex fields, with or without lumped elements. This
  is intended: the old closure is non-passive.
- **BITID baselines** with CPML: `test_boundary_bit_identity`, `test_bit_identity_baseline`,
  RF/port bit-identity and step-graph identity, and existing-monitor identity.
  - Classify each config as CPML or no CPML. No-CPML configs (PEC, periodic-only, the graded
    absorber, Mur, simple PML) must stay bitwise identical.
  - Re-baseline the CPML configs once, with a movers declaration: the config list, the max
    field delta and the reflection or energy change.
  - Probes are pinned only where a no-CPML path exists. There is no legacy-closure switch:
    the repository forbids parallel legacy paths, and the old closure is the defect.
  - The CPML re-baseline is **one dedicated commit**, carrying the declaration table and
    merged in the same batch as B1.
- **Benchmarks and tests:**
  - list every test that asserts a CPML-dependent value tighter than its physics tolerance, and
    update it with a reason;
  - estimate the movers count before merge (grep of CPML scenes; last count about 60-80 test
    files on periodic or PEC-closed plus PML, with most open-PML tests moving within
    tolerance).
- **Downstream notice** (coordinator sends):
  - instamax (FDTD oracle);
  - maxwell-bench: provider-epoch movers, re-frozen per the epoch ritual;
  - the core-CUDA line: section 5;
  - PCB:
    - CPML configs in the pcb-44, copper and thin-sheet suites move, and so do PCB's
      frozen-tree comparisons;
    - D3b's fused-vs-torch correctness compares on one tree and is unaffected, but is told;
  - the integration runner: the movers manifest for the B batch.
- **Docs:** `docs/reference/fdtd-cpml.md` (the closure and the storage functional), the
  FEATURE_LIST CPML entry, the behaviour-change line, and retirement of the alpha_max = 0
  hazard section once G1 passes for that preset.

## 5. Core-path mirror contract (for the core-CUDA owner)

The core path (`fdtd/cuda/core/yee.cu` `psi_update<Axis,Cpml>` and the `d*kappa + psi`
combination in `component()`; `fdtd/runtime/core_path.py` CorePathRuntime; the O2a capture
snapshot) mirrors B in the same batch after B merges. B0 must therefore produce, and B1 must
keep fixed:

1. the per-cell state: the count per stretched derivative, the dtype, and the dense and slab
   layouts (keep the existing slab layout machinery if at all possible);
2. the exact update equations, with their float operation order and FMA grouping, for the psi
   step and for the corrected-derivative combination;
3. any new coefficient vectors, with their shape and axis indexing;
4. whether the E/H update form inside PML cells changes (for example a time-centered closure
   that mixes field time levels). Interior cells must not change.

Until B merges, the core path keeps today's recurrence and its G1 (bitwise against the main
path) stays valid.

## 6. Ownership and order

- Spec, B0/B1 review, all GPU gates, movers declaration and merge note: Claude (Bloch/CPML
  line).
- B0 and B1 implementation: Codex, CPU-only in B0. B1 is compiled and run on GPU by Claude.
- Coordinator review at three points: this spec, after B0 (the closure choice), and before
  merge.
- Order:
  - the r2 broadband merge first;
  - B0 (CPU) runs in parallel with the reduced F1 Phase 1 (GPU 0). F1 scenes are open or
    all-PML single scatterers; any closed-transverse F1 scene uses the interim absorber;
  - the B0 review;
  - B1.
- F5 production data waits for B.
