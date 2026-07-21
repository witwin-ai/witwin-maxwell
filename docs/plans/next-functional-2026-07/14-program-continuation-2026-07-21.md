# Program continuation plan — rounds F/G/H (post round-E tiers 1–4)

> Status: active — **Rounds F & G DELIVERED & merged (2026-07-21); Round H launching**
> Date: 2026-07-21
> Anchor: master `164c263` (round E merged; final battery 2836 passed / 16 expected-FDFD / 3 xfailed)
> Round-F merge anchor: master `b89a75c` (F1 `07e8e99`, F2 `0546b0a`, F3 `7ec99c7`,
> F4 `431bd7f`, hygiene `b89a75c`; full battery **2911 passed / 16 expected-FDFD / 3 xfailed**)
> Round-G merge anchor: master `18bc42a` (G1 `42ac3f1`, G2 `3884bb7`, G3 `5dd100e`,
> G4 `c9b4bfc`, audit-minor cleanup `18bc42a`; full battery **2982 passed / 16
> expected-FDFD / 3 xfailed / 1 xpassed**)
> Governing status doc: `00-status-and-gaps-2026-07-19.md` (round-E + round-F + round-G revisions)

## Delivery status (2026-07-21)

**Round F — DELIVERED & merged to master `b89a75c`** (all four tracks adversarially
audited; load-bearing tests falsified; per-track acceptance docs under `docs/assessments/`):

- **F1 — plan 04 E2 evidence (closes S3), merge `07e8e99`.** Multi-scenario coupled
  EM+circuit conservation suite (3 two-way-coupled scenarios, honest gate classes) +
  independent offline `scipy` circuit cross-check (port-voltage rel err 1.16e-5, no shared
  runtime code). Acceptance `f1-cosim-e2-acceptance-2026-07-21.md`.
- **F2 — plan 01 trio, merge `0546b0a`.** Interior-PEC masking on the Yee-staggered
  operator + quasi-static line-mode engine (F2a); production quasi-TEM microstrip/diff-pair
  benches un-BLOCKED (F2b, recorded `gap` — resolution-limited, honest); adapter port/lumped
  source mapping + four external caches generated (F2c); patch feed diagnosis (xfail kept
  fail-closed, not flipped). Acceptances `f2a-…`, `f2b-…`, `f2-rf-trio-acceptance-2026-07-21.md`.
- **F3 — plan 06 scene-gradient VJP + 02 P7 aggregation, merge `7ec99c7`.** Single-device
  `ArrayBasisData.scene_gradient_vjp` (census 176→175) + 2-GPU ensemble aggregation;
  1-vs-2-GPU parity measured **bitwise**. Acceptances `f3-array-vjp-…`, `f3-array-scene-vjp-…`.
- **F4 — S5 geometry/subpixel lever, merge `431bd7f`.** Edge-native per-Yee-component
  material sampling + conformal-PEC benchmark default; geometry-cluster median field_l2
  0.2072→0.0836 (−59.6%), 11 improved / 0 regressed. Acceptance
  `f4-subpixel-lever-acceptance-2026-07-21.md` + `f4-geometry-cluster-{before,after,delta}.json`.

**Consequence:** **route step S3 PASSED** (03 round-E cross-check + gate-(d) ruling, 04 F1,
06 F3) → the **Wave-C (S6) unfreeze condition is met**. No Round-F phase is `completed`
(audit §4 non-author-review + external-reference bar unmet); rounds record delivery +
evidence grade only.

**Round G — DELIVERED & merged to master `18bc42a`** (all four tracks adversarially
audited; load-bearing tests falsified; per-track acceptance docs under `docs/assessments/`;
census reconciled to **175**, round-G net ±2):

1. **02 — NCCL reverse-halo adjoint transport primitives, merge `42ac3f1`.**
   `prepare_adjoint_staging` / `exchange_magnetic_adjoint` / `exchange_electric_adjoint`
   gated by a **bitwise** 2-process discrete-transpose identity + an opt-in per-rank
   step-rate instrument (zero-cost-off asserted, resolves the round-E "not-measurable"
   finding). The **end-to-end per-rank reverse driver is honestly deferred, fail-closed**,
   with a written **7-step plan** (the in-process bridge is structurally single-process).
   Acceptance `g1-nccl-adjoint-acceptance-2026-07-21.md`.
2. **07 — lossy-wire B2 recurrence + B3 conductivity adjoint, merge `3884bb7`.** Passive
   lossy-current ADE companion consumed by the runtime + real `ohmic_loss`
   (`0.5·Re(Z')·L·|I|²`), analytic-AC (<8%, fit-limited) / DC-exact / bitwise-PEC-parity /
   spectral-stability gates; closed-form `dZ'/dσ` adjoint vs central difference <1e-6.
   Field-coupled `dI/dσ` + distributed lossy stay fail-closed; closed-box field-energy
   closure not performed (companion-level only). Acceptance
   `g2-lossy-wire-acceptance-2026-07-21.md`.
3. **08 — ferrite arbitrary-bias forward + mixed-bias support, merge `5dd100e`.** Per-cell
   coordinate-rotation general path (rotation-equivalence bit-for-bit, oblique-vs-Polder-oracle
   1.197e-13, handedness, per-cell independence, zero-impact, CUDA passivity); contract-doc
   §7 item 6 superseded; census `175 → 173`. Acceptance `g3-ferrite-bias-acceptance-2026-07-21.md`.
4. **09 — all-orientation staircased SIBC, merge `c9b4bfc`.** Curved conductors staircased
   into masked Leontovich exposed-face writes + staircased-cylinder absorbed-power convergence
   gate (documented ~18% first-order-on-curve systematic, grid/R/δ-independent) + wave-level
   skin-effect attenuation bench (alpha median rel err 0.049%) + committed zero-impact gate.
   True oblique/conformal remains fail-closed; external cross-check is a documented adapter gap.
   Acceptance `g4-sibc-oblique-acceptance-2026-07-21.md`.

**Consequence:** the three Wave-C solver-consumption tracks (07/08/09) landed post-unfreeze
(07 lossy E0→E1–E2; 08 E0→E1; 09 E0→E1) and the 02 NCCL adjoint transport foundation is in
place. No Round-G phase is `completed` (audit §4 non-author-review + external-reference bar
unmet; 09's external cross-check is a documented adapter-fidelity gap, not a pass).

**Round H — LAUNCHING (2026-07-21), tracks:**

1. **02 — NCCL end-to-end reverse DRIVER** per the written 7-step plan in
   `g1-nccl-adjoint-acceptance-2026-07-21.md` (guard-relax → per-rank checkpoint capture →
   NCCL forward-replay dict halos → local separable seed → per-rank reverse loop → grad_eps
   gather + rank-0 pullback → parity/determinism/falsification gates), on top of the landed
   reverse-halo transport primitives.
2. **12 — tensor-eps + open boundary** (electrostatics Phase 4: nonuniform grid,
   tensor/anisotropic eps, open-boundary truncation — the P4 items rejected at prepare).
3. **10 — IEEE/IEC phantom benchmark + incident power density** (SAR: standard-phantom
   cross-check + the `IncidentPowerDensity` monitor / `input_power` normalization currently
   fail-closed).
4. **13 — circuit-ESD co-simulation + smooth breakdown surrogate** (Phase 3 source-impedance/
   discharge-network coupling consuming F1's co-sim path + the Phase-5 trainable smooth
   breakdown surrogate).

> Execution model unchanged (parallel single-writer worktrees, two adversarial audits,
> supervisor merge gate, integration battery — see below).
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
