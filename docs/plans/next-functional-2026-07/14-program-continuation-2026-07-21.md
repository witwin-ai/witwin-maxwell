# Program continuation plan — rounds F/G/H (post round-E tiers 1–4)

> Status: active
> Date: 2026-07-21
> Anchor: master `164c263` (round E merged; final battery 2836 passed / 16 expected-FDFD / 3 xfailed)
> Governing status doc: `00-status-and-gaps-2026-07-19.md` (round-E revision)
> Execution model: parallel single-writer worktrees, Workflow orchestration, per-track
> dev stages → two adversarial audits (regression + claim lens, evidence reproduced
> independently, load-bearing tests falsified) → fix rounds → supervisor merge gate →
> integration phase → full battery. Same governance as the Wave-D / round-E deliveries.

## Tier structure (owner-ordered 2026-07-21)

The four tiers recorded here are executed as three rounds. Wave-C unfreeze (S0.2)
requires S3 to pass, and S3's only remaining member is plan 04 — hence 04 leads.

### Tier 1 → Round F track F1 — plan 04 E2 evidence (closes S3)

- Multi-scenario conservation/energy-residual gates for coupled EM+circuit runs
  (≥3 scenarios: resistive, resonant RLC, controlled-source): source input =
  EM stored + boundary outflow + material dissipation + circuit dissipation +
  circuit stored, from V/I records and field diagnostics. Honest gate classes per
  the E4 lesson (a term algebraically forced by the coupling is consistency-class,
  not conservation-class — annotate).
- Independent circuit cross-check breaking the self-reference: the coupled
  FDTD+MNA port V(t)/I(t) versus an independent transient integration (separately
  derived state equations, scipy-class integrator, no shared code with the MNA
  runtime) of the equivalent circuit driven by the EM structure's measured
  port characterization. Pre-registered tolerance.
- Stretch (gated on F2's adapter mapping): external-reference cross-check of one
  lumped-load scene. Exit: 04's E2-blocking gaps in `00-status-and-gaps` closed;
  S3 recorded as passed → S6 (Wave C) unfreeze condition met.

### Tier 2 → Round F track F2 — plan 01 trio

1. Interior-PEC masking on the Yee-staggered transverse operator (signal strip /
   ground inside the aperture): symmetric elimination of PEC-occupied Eu/Ev/node
   unknowns. Gates: microstrip quasi-TEM eps_eff vs Hammerstad–Jensen closed form
   (few-%, pre-registered), differential-pair even/odd splitting with orthogonal
   symmetric/antisymmetric profiles; then production `rf/microstrip_two_port` and
   `rf/differential_pair` wave-level benches (B=S·A pipeline) + external reference
   caches (spend owner-authorized).
2. Adapter port/lumped source mapping so `rf/coax_thru`, `rf/lumped_open_short_match`,
   `antenna/half_wave_dipole`, `antenna/patch` export runnable; generate the four
   pending caches (one run each, task ids + cost recorded); RESULTS rows.
3. Patch antenna feed redesign (current strict-xfail: off-broadside, reactance-
   dominated). Gates: cavity-model resonance, broadside pattern, D ≥ 5 dBi;
   external cross-check; flip the xfail.

### Tier 3 → Round F track F3 — plan 06 scene-gradient VJP (+ 02 P7 aggregation)

- Implement the per-column adjoint result-aggregation contract (02 Phase-7 item
  that blocked 06 Phase 4) on top of the landed S4 adjoint; implement
  `ArrayBasisData.scene_gradient_vjp(...)` single-device first, ensemble 2-GPU
  aggregation where the contract holds. Gates: central-difference gradient checks
  on a small array scene (S3-calibrated tolerances), falsified; multi-GPU
  aggregation parity if delivered; census reconciliation for every guard removed.

### Tier 4 → Round F track F4 (first slice) + rounds G/H

- **F4 (this round): S5 geometry/subpixel lever** — per-Yee-component staggered
  material sampling (drop the node→edge arithmetic smear), conformal-PEC default
  in the benchmark harness (the 2026-07-17 validation audit's biggest systemic
  lever, ~20 scenes at 0.10–0.63). Gates: geometry-cluster benchmark metrics
  improve against existing caches (no new cloud runs needed), no regression in
  analytic/flux/port suites (physically-justified tolerance shifts documented
  scene-by-scene), 3-grid convergence retained. This intentionally changes
  numerics — the audit pair must check every tolerance edit is justified, not
  laundered.
- **Round G (starts when F merges; S3 must be green for the Wave-C tracks):**
  02 tail — NCCL one-process-per-GPU adjoint, coupled-runtime joint solve,
  S5 tiled monitor seeds, NCCL timing hooks (the "not-measurable" finding);
  Wave C solver consumption — 07 lossy-wire recurrence (B2→B3), 08 ferrite
  forward consumption widening, 09 SIBC oblique/curved. Four tracks.
- **Round H:** Wave D deepening — 12 tensor-eps/open-boundary, 10 IEEE/IEC
  phantom benchmark + incident power density, 13 circuit-ESD co-simulation
  (consumes F1) + smooth breakdown surrogate; leftovers from G.

## Standing rules for all rounds

- Evidence discipline unchanged: no number without a committed reproduction
  command; every load-bearing new test falsified (red→restore→green) and the
  falsification recorded; census budget reconciled per the census doc (176 at
  anchor); FEATURE_LIST additive per track; commercial names only in adapter
  paths; docs `git add -f`.
- Timing measurements ONLY in exclusive windows (A/A calibration, numactl,
  variance-aware CI); correctness tracks never quote wall-clock numbers.
- `completed` phase marks still require wave-level gate + independent reference +
  non-author review (audit §4); rounds record delivery + evidence grade only.
- Owner decision recorded 2026-07-21: gate (d) small-grid fixed cost — the
  compute-bound ruling stands; further pursuit (implicit-solve/observer fusion)
  deferred unless small-grid embedding becomes a product need.
