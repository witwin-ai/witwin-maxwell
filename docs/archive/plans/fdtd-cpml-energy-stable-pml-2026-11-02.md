# Energy-stable PML for complex fixed-k runs (CPML-B research track)

**Outcome (2026-11-03): ENDED, NO-GO.** R0/R1 rejected the AHK transverse-memory
discretization (docs/memory/fdtd-cpml-energy-stable-pml-r0-2026-11-02.md,
docs/assessments/fdtd-cpml-energy-stable-pml-r1-2026-11-02.md). The one extra timeboxed round
rejected the SBP/SAT interface couplings at R0 with an exact counter-certificate
(docs/memory/fdtd-cpml-energy-stable-pml-sbp-r0-2026-11-03.md). Per the user's hard-stop
decision, items (1) graded-absorber auto default and (2) fail-closed passivity guard are
permanent, and oblique F5 data is CW-only or uses the guarded CPML.

Original status: spec for coordinator review (2026-11-02). The user decided "CPML 1+2+3". This is item 3: a
research track and the prerequisite for oblique F5 data. It is **not** a v0.6.0 release gate,
because the fail-closed passivity guard (item 2) counts as the release fix.

## 1. Goal

A PML closure for the Yee scheme that, for complex fixed-k (oblique Bloch) runs with dielectric
layers, is

- **passive**, with a discrete storage functional whose source-free increment is non-positive and
  certified on the B0 model, and
- **CPML-accurate**: reflection no worse than the current CPML at normal and oblique incidence, and
  strictly better on evanescent harmonics.

If it succeeds, it also replaces the graded-absorber default for real-field closed-transverse runs
(item 1), so one absorber covers every topology.

## 2. What is already known (do not repeat)

Evidence is in `docs/memory/fdtd-cpml-passive-closure-derivation-2026-11-02.md` (master 3ae0afce),
with per-case data in `<evidence-root>/cpml-b0-results-2026-11-02/`.

- **The B0 model is the qualified harness.** It is a float64 reduced Yee block (periodic, PEC or PMC
  transverse; TE/TM; slab) that reproduces the native CUDA spectrum to about 1e-9, and it carries
  an exact discrete work identity.
- **The current closure fails** 927 of 1458 Cartesian and 98 of 288 fixed-k cases. It returns about
  0.5 of an evanescent harmonic.
- **Rejected candidates:**
  - time-centered ψ (c1);
  - a constitutive stretched-coordinate ADE (c2). It has a local obstruction: Re[s ε_n(s)] < 0;
  - 81 memory orderings (c3);
  - multi-axial PML at every p. It is not passive in the large-cell check, and its reflection is
    up to 2e4 worse;
  - strong CFS profiles: passive only at 0.6–20% reflection;
  - the graded absorber for fixed k: r/t errors of 0.2–0.5.
- **Only certified-passive scheme:** matched scalar conductivity with global-midpoint stepping. It
  changes the interior integrator, so it is not admissible. It still serves as the passivity
  control.

## 3. Candidate families (R0 selects at most two)

1. **SBP/SAT-coupled PML.** Treat the interior/PML interface as a summation-by-parts boundary with a
   simultaneous-approximation-term coupling chosen to make the discrete interface energy
   dissipative. The PML equations follow the Duru–Kreiss formulation, with energy estimates for
   piecewise-constant layers. The Yee staggered difference is an SBP operator pair, so an interface
   SAT is plausible without touching interior cells.
2. **Appelö–Hagstrom–Kreiss "stable PML for hyperbolic systems"**, the general first-order
   formulation with auxiliary variables. Its stability condition (on α relative to σ and the
   tangential wavenumbers) is to be mapped onto our fixed-k transverse symbol. Below-light-line
   modes are the case it targets.
3. **A modal or "complex-shift with corrected time-stepping" PML** in the Bécache–Joly line of
   stability analyses. Use it only if R0 finds a formulation whose discrete energy is provable on a
   staggered grid.

R0 must show, on paper, a discrete storage identity for the reduced block (piecewise-constant
layers first) for each candidate it carries forward. Candidates that rely on a continuous estimate
without a discrete one are dropped. This is the lesson from c2.

## 4. Stages and go/no-go

All CPU stages are Codex in the B0 model (no GPU). Claude reviews each stage and runs every GPU
step.

| Stage | Content | Exit criterion | Timebox |
|---|---|---|---|
| R0 | Literature mapping and discrete derivation for at most 2 candidates, on paper plus symbolic checks | A discrete storage identity whose increment is ≤ 0 in the reduced block, at least for piecewise-constant σ | 3 days |
| R1 | Implement the R0 candidates in the B0 model (`scripts/dev/fdtd/cpml_passive/`), with graded profiles | The B0 sweep (1458 Cartesian + 288 fixed-k) with 0 cases above 1+1e-12; the 4×-period large-cell check flat; the G2b slab-guided TM case non-growing | 4 days |
| **Go/no-go** | Coordinator and user | **GO** if R1 passes **and** reflection (B0 `reflection.py`; normal, 60°, 80°, evanescent; 12 layers) is ≤ current at normal/60°, ≤ 2× current at 80°, and strictly lower on evanescent. **NO-GO** otherwise. In that case, stop and keep items (1)+(2) permanently; the oblique F5 scope becomes CW-only or uses the guarded CPML. | — |
| R2 | A 3D CPU prototype on one PML axis (a small numpy/torch reference), validated against the B0 block and a 3D reflection measurement | Agrees with the B0 model; 3D reflection within criteria | 3 days |
| R3 | B1-style kernel spec: electric, magnetic, aniso and adjoint kernels, complex banks, checkpoint/graph state, ABI bump at merge, core-path mirror contract for the core-CUDA owner, movers and BITID plan | Coordinator approval | 1 day |
| R4 | Implementation (Codex) plus GPU qualification (Claude), with the gates of `fdtd-cpml-passive-closure-2026-11-02.md` §3 (G1–G8, including G2b) | All gates pass | 1–2 weeks |

Total to the go/no-go point is about 1.5 weeks of CPU work. The research is uncertain, and the
go/no-go point is where it stops if no candidate is passive and accurate.

## 5. Acceptance at merge (R4)

- **Passivity:** G1 (spectral, including the larger-cell check), G2 (120k LC energy at
  10/30/45/60/75°), G2b (slab-guided TM non-growing over 320k steps, and its 0.6/0.7 f0 r/t within
  0.005 absolute), G3 (real-field growth scenes flat), and G4 (closed-box storage non-increasing).
- **Accuracy:** G5 reflection as above, plus G6 (copper-foil TE/TM ≤ 3%, removing the strict xfail;
  the Airy gates as merged).
- **Adjoint:** G7 finite-difference checks.
- **Regression:** G8, the whole tests/ sweep, with no-CPML BITID configs bitwise and the CPML
  re-baseline in one dedicated commit.
- **Retirements:** the fail-closed guard (item 2) and the graded-absorber auto default (item 1) are
  retired or relaxed only after R4 qualifies, as a separate, reviewed change.

## 6. Interfaces and coordination

- **Core path (the core-CUDA owner):** mirror contract at R3 (state, op order, coefficients, E/H form in PML
  cells). The core path keeps refusing guarded topologies until R4 lands.
- **ABI:** the next free number at R4 merge time.
- **Data plan:** oblique F5 labels wait for R4. Real-field F5 can proceed under item (1).
