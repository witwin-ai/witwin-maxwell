# next-functional-2026-07 — Status and Gaps (plans 01–13)

> Date: 2026-07-19
> Commit anchor: `a96338a` (master; the `integration` worktree was fast-forwarded
> to this commit so it contains the S2b measurement it cites — see Caveats C-1).
> **Round-E update: 2026-07-21, master `6b523b8`.** Rounds E delivered and merged
> for plans 01/02/03 (merges `a963512`/`aa866ba`/`ead70c0`, `2364533`/`f7e8e9a`,
> `6c621a2`; exclusive-window timing `6b523b8`). This update revises the 01/02/03
> rows, per-plan sections, route steps S1/S4, and owner decision points 1/2; the
> 04–13 rows are unchanged from the 2026-07-19 snapshot.
> **Round-F update: 2026-07-21, master `b89a75c`.** Round F delivered and merged:
> F1 plan-04 coupled-EM+circuit conservation suite + independent offline circuit
> cross-check (`07e8e99`); F2 plan-01 trio — interior-PEC staggered modes, production
> quasi-TEM microstrip/diff-pair benches, adapter port/lumped source mapping with four
> external caches, patch feed diagnosis (`0546b0a`); F3 plan-06 array scene-gradient
> VJP single-device + 2-GPU ensemble (`7ec99c7`); F4 S5 geometry subpixel lever —
> edge-native per-Yee-component material sampling + conformal-PEC benchmark default
> (`431bd7f`); hygiene (`b89a75c`). Full battery **2911 passed / 16 expected-FDFD /
> 3 xfailed**. This update revises the 01/04/06 rows and sections, S5, marks **route
> step S3 = DONE** (all three members landed), records the Wave-C (S6) unfreeze
> condition met + Round G launched, and resolves owner decision points 1/3. All
> Round-F deliveries were adversarially audited; no phase is `completed` (no
> non-author review — the audit §4 bar is unmet).
> **Round-G update: 2026-07-21, master `18bc42a`.** Round G delivered and merged
> (all four tracks adversarially audited; audit-minor cleanup `18bc42a`; full
> battery **2982 passed / 16 expected-FDFD / 3 xfailed / 1 xpassed**): 02 NCCL
> reverse-halo adjoint **transport primitives** + bitwise 2-process transpose
> identity + opt-in step-rate instrumentation (`42ac3f1`), the end-to-end per-rank
> reverse **driver** honestly deferred with a written 7-step plan; 07 lossy-wire
> B2 passive current recurrence + real `ohmic_loss` + B3 conductivity adjoint
> (`3884bb7`); 08 ferrite arbitrary-bias forward + mixed-bias support (`5dd100e`);
> 09 all-orientation staircased SIBC + cylinder physics gate + skin-effect
> attenuation benchmark (`c9b4bfc`). This update revises the 02/07/08/09 rows and
> sections, the census history (→175, round-G ±2), route steps S4/S6, and marks the
> three Wave-C consumption tracks (07/08/09) landed post-unfreeze. Sources:
> `docs/assessments/g1-nccl-adjoint-acceptance-2026-07-21.md`,
> `g2-lossy-wire-acceptance-2026-07-21.md`, `g3-ferrite-bias-acceptance-2026-07-21.md`,
> `g4-sibc-oblique-acceptance-2026-07-21.md`. All Round-G deliveries were
> adversarially audited; no phase is `completed` (audit §4 non-author-review +
> external-reference bar unmet).
> **Round-H update: 2026-07-21, master `6f3b0c8`.** Round H delivered and merged
> (all four tracks adversarially audited; audit-minor cleanup `6f3b0c8`; final
> battery **3076 passed / 16 expected-FDFD / 3 xfailed / 1 xpassed**; census
> reconciled to **176**): 02 per-rank collective NCCL end-to-end reverse **driver**
> + S5 separable tiled-plane monitor seeds, with psi-active and stressed parity
> gates, and a cross-stream caching-allocator race found and fixed (merge `acea86e`);
> 12 SPD tensor-permittivity FVM operator + open-boundary fail-close +
> domain-extension `truncation_estimate` (merge `4a0555d`); 10 incident power density
> monitor + canonical phantom benchmark family + SAR RESULTS rows (merge `8ebaec0`);
> 13 circuit-driven ESD through the standard source network + differentiable
> `SmoothBreakdownRisk` surrogate (merge `df8ef96`). This update revises the
> 02/10/12/13 rows and sections, the census history (→176), route steps S4/S5/S7, and
> records the cross-stream allocator-race root-cause + the in-process `cuda_p2p` load
> hazard as a new open item. Sources:
> `docs/assessments/h1-nccl-driver-acceptance-2026-07-21.md` (incl. the cross-stream
> allocator-race root-cause section), `h2-es-tensor-acceptance-2026-07-21.md`,
> `h3-sar-phantom-acceptance-2026-07-21.md`, `h4-esd-circuit-acceptance-2026-07-21.md`.
> All Round-H deliveries were adversarially audited; no phase is `completed`
> (audit §4 non-author-review + external-reference bar unmet).
> **I1 update: 2026-07-21, master `625baca`.** The in-process
> `transport="cuda_p2p"` distributed-adjoint cross-stream load hazard that Round-H
> recorded as a "new open item" is **CLOSED** (fix `1a579b3`, merged at `625baca`).
> It was a **distinct** hazard from the NCCL allocator-reuse race — a
> checkpoint-capture happens-before race in `capture_distributed_checkpoint`
> (default-stream clone unordered w.r.t. the next compute-stream forward update),
> discriminated by `CUDA_LAUNCH_BLOCKING=1` collapsing the drift to the ~2e-7 floor
> while `PYTORCH_NO_CUDA_MEMORY_CACHING=1` left it untouched. Fix: clone the
> checkpoint on the shard's `compute_stream`. A committed stressed parity gate +
> falsification pin it (standard/x-CPML 1-vs-2-GPU parity ~2e-7 under a saturating
> co-tenant burner; reverting to the default-stream clone reddens to ~8.09e-2 while
> the forward loss stays bitwise clean); census unchanged at 176. This update
> flips the 02 row, the 02 section gaps + evidence grade, and route step S4 from
> "open pre-existing item" to CLOSED. Source:
> `docs/assessments/i1-p2p-race-acceptance-2026-07-21.md`.
> Governing audit: `docs/assessments/next-functional-audit-2026-07-18.md`.
> Binding vocabularies: E0–E3 evidence grades (audit §0/§4), the five-class gate
> taxonomy + `perf` labels (`docs/reference/gate-classification.md`), the S0–S7
> convergence route (audit §3).

## Purpose

Single objective status-and-gaps reference for the next-functional-2026-07
program (plans 01–13; plan 05 = nonlinear devices; plan 11 bioheat DROPPED by
owner 2026-07-19, plan file deleted in commit `4c521ab`, so thirteen active
plans remain).
It records, per plan, what is DELIVERED with evidence pointers, what is NOT
delivered with each gap's blocker or route dependency, and the conservative
evidence grade. It uses the audit's vocabulary verbatim and never restates a
"done" without a tracked artifact. Drifting per-scene numbers live in the
machine-readable artifacts; this document quotes only headline metrics.

**Maintenance rule.** Update this file at every merge that changes a plan's
delivered-phase count, evidence grade, freeze state, or a route step (S0–S7).
Prefer pointers over copied numbers so the artifacts stay the single source of
truth.

## Summary table

Evidence grade = the audit's **measured** grade, updated only where S0/S1/S2
landed new tracked evidence. A claim without a tracked artifact keeps its audit
grade.

| # | Title | Phases delivered / total | Evidence grade (measured) | Headline gaps | Freeze state |
|---|---|---|---|---|---|
| 01 | RF engineering workflow | 0–5 declared; per-gate re-classified; S1 keystone (round-E) + F2 trio | **E2 for the validated scene set** (`coax_thru` + `rectangular_waveguide` wave-level, external-reference cross-check); microstrip/diff-pair now production quasi-TEM benches (measured `gap`, resolution-limited); E0–E1 elsewhere; §9.4 perf gates PASS (`perf`) via S2 | microstrip/diff-pair `eps_eff` resolution-limited (~24% low at dx=5 mm; quasi-static engine itself converges to H–J); patch broadside honest strict-xfail (not flipped); guided interior-PEC path not yet production-wired | reopened; S1 + F2 trio landed (microstrip/diff-pair un-BLOCKED); four external caches generated; S2/S3 done |
| 02 | Multi-GPU execution | ensemble + joint-solve forward + S4 distributed CPML-trainable adjoint (psi-active) + Round-H per-rank collective NCCL end-to-end reverse **driver** + S5 separable tiled-plane monitor seeds | forward **E2**; CPML-trainable distributed adjoint **E2**; NCCL reverse transport **E2**; **NCCL end-to-end reverse driver E2** (per-rank point/plane parity ~2e-7 incl. psi-active + stressed; cross-stream allocator race fixed); coupled joint-solve E0 | flux/mode/x-normal NCCL objectives + monitor gather beyond forward; coupled-runtime joint solve; NCCL driver timing pending exclusive window; static-capability exclusions remain (in-process `cuda_p2p` load hazard **CLOSED** in I1, `625baca`) | S4 landed; Round-H NCCL driver + S5 tiled seeds landed (driver E2); timing pending exclusive window |
| 03 | Touchstone network embedding | 0–4 + round-E E2 evidence (gate (d) grid-conditional) | **E2** (embedded path: independent raw-sample S-cascade cross-check <1e-5 + multi-scenario passivity/conservation) | gate (d) still grid-conditional (PASS ≥224³, compute-bound ruling); delay adjoint fail-closed; WavePort embedding = missing design contract; inherits 01 port-power (now partly wave-validated) | E2 evidence landed; gate (d) grid-conditional; delay-adjoint/WavePort open |
| 04 | SPICE/MNA co-simulation | 0–4 + F1 conservation suite + independent offline cross-check | **E2** (multi-scenario conservation + code-path-independent circuit cross-check landed) | strong-coupling end-to-end ref still `future-xfdtd`; reactive `dU_circuit` channel stays consistency-class (not lifted by F1b); external lumped-load cross-check pending | E2 evidence landed (F1); S3 closed |
| 05 | Nonlinear circuit devices | Phase 0 + N1 standalone transient | E0 | FDTD coupling / transient companion / adjoint / benchmark not built; BJT/MOSFET fail-closed | Wave C — S3 passed 2026-07-21, unfrozen; not scheduled in Round G |
| 06 | Array / Active-S / MIMO | 0–3 + Phase-4 weight gradients + F3 scene-gradient VJP (single-device + 2-GPU ensemble) | **E2** (VJP bit-for-bit vs single-device path; 1-vs-2-GPU bitwise parity; central-difference gated) | throughput/scaling deferred-pending-exclusive-window; network/S-param scene gradients + NCCL joint-solve adjoint out of scope; inherits 01 port-power risk | reopened; Phase-4 exit gate closed (F3); S3 closed |
| 07 | Thin-wire model | 0–3 + Phase-4 + Round-G B2 lossy recurrence + B3 conductivity adjoint | PEC **E2** / lossy **E1–E2** | field-coupled `dI/dσ` reverse fail-closed (nondeterministic shared fit); B4 distributed lossy reverse fail-closed; distributed lossy forward now fail-closed (was silent-PEC); analytic-AC gate 8% (fit-limited, not 2%); closed-box field-energy closure not performed (companion-level only) | Wave C unfrozen (S3 passed); Round G B2+B3 landed |
| 08 | Gyromagnetic ferrite | Phase 0 + arbitrary-bias forward + mixed-bias support (Round G) | **E1** | FDFD, multi-GPU, adjoint, Bloch, `PerturbationMedium`-over-ferrite fail-closed; identity collocation (not 4-point Yee gather) | Wave C — unfrozen (S3 passed); Round G general+mixed-bias landed |
| 09 | Surface impedance / roughness | Phase 0 + Phase-1 + Round-G all-orientation staircased SIBC + cylinder physics gate + skin-effect bench | **E1** | true oblique/conformal (non-staircase) fail-closed; rotated Box / rational-on-curved / Bloch fail-closed; adapter export fail-closed (external ref under-applies wall loss); ~18% staircase-curve absorbed-power systematic | Wave C — unfrozen (S3 passed); Round G orientation generalization landed |
| 10 | SAR | 0–3 + P4 slice + Round-H incident power density + canonical phantom family + SAR RESULTS rows / 0–5 | **E1–E2** (analytic/golden/brute-force-parity/grid-convergence + wave-level conservation closure; no external ref) | IEEE/IEC *certified* phantom profiles + external ref still deferred; `antenna_near_phantom` blocked (conductive-media port); `input_power` normalization + VOP (P5) + multi-GPU scale-out (P4) fail-closed/deferred | Wave D — selective start 2026-07-19; Round-H IPD + phantom benches landed; S7 partial |
| 11 | Bioheat | — (dropped by owner) | — | plan dropped by owner 2026-07-19; plan file deleted (commit `4c521ab`) | Wave D — DROPPED (not implemented) |
| 12 | Electrostatic / capacitance | 0–3 + P5 diff slice + Round-H P4 SPD tensor-eps + open-boundary truncation_estimate / 0–6 | **E2-class for the delivered envelope** (SPD-symmetry/rotated-MMS-convergence/reciprocity/energy-identity/domain-extension; no external ref) | exact open boundary (BEM) + trainable tensor-eps backward (P5) fail-closed; 1st-order wall cross-flux; multi-GPU (P5), touchscreen workflow (P6) deferred | Wave D — selective start 2026-07-19; Round-H P4 tensor-eps + open-boundary landed; S7 partial |
| 13 | ESD / dielectric breakdown | 0–2 + P3 pre-bias + Round-H P3 circuit-ESD + P4 + P5 smooth-risk surrogate / 0–7 | **E1–E2** (analytic waveform/golden state-machine/energy-closure/dt-convergence + independent offline circuit cross-check + EM-load-bearing gate; no external/calibrated ref) | P3 substantially delivered (prebias + circuit ESD); conductive-media breakdown-feedback port fail-closed; P5 multi-GPU, P6–7 calibration/standard workflow deferred | Wave D — selective start 2026-07-19; Round-H circuit-ESD + smooth-risk surrogate landed; S7 partial |

Route status (S0–S7) and owner decision points are in the final two sections.

---

## Per-plan status

### 01 — RF engineering workflow

Plan: `docs/plans/next-functional-2026-07/01-rf-engineering-workflow.md`
(status `reopened`, 2026-07-18). Audit §1.1 is the governing per-gate review.

**Delivered (with evidence):**
- Public API surface: LumpedPort/TerminalPort/WavePort, RLC terminations,
  `PortData`/`NetworkData`/`AntennaData`, `PortSweep`, S/Z/Y + renormalization +
  mixed-mode + Touchstone export. Phase acceptance docs
  `docs/assessments/rf-workflow-phase-0-acceptance.md` … `-phase-5-acceptance.md`
  (single-CUDA-device scope).
- Adjoint mechanism (Phase 5): analytic VJP vs CUDA autograd to ~2e-13 and
  bitwise checkpoint replay — the `wave-level` mechanism evidence
  (`gate-classification.md` §2.5).
- **S1 (audit step, this route):** `benchmark/scenes/rf/coax_thru` is the first
  genuine **wave-level PASS** — terminated FDTD two-port run through the padded
  computational PML, S from `B=S·A`, gated on extraction conditioning (cond(A)
  ~1.2) + post-solve passivity (max singular value ~1.0), with `beta` from
  `arg(S21)/L` within 0.83% of `k0`. Reference:
  `docs/reference/rf-wave-validation-2026-07-18.md` §1.1, artifacts under
  `docs/assessments/rf-wave-validation-2026-07-18/`. New wave-level replacement
  gates: `tests/rf/wave_validation/test_matched_s11_wave_level.py`,
  `test_asymmetric_reciprocity_power_balance.py`.
- **S2 (audit step):** both plan-01 §9.4 timing gates PASS at a 27M-cell grid
  under a variance-aware one-sided 95%-CI-upper-bound criterion — single
  LumpedPort + 181 freq observer CI95-upper **1.57% < 5%**, per-extra-passive-port
  **1.53% < 2%**, A/A measurement floor 0.072%, with executed injection
  falsification. Artifact `docs/assessments/port-perf-s2b-measurement-2026-07-18.json`;
  progress `docs/assessments/port-hot-path-perf-s2-progress-2026-07-18.md`. The
  port hot-path fix itself landed earlier (SeriesRLC schedule 62→25 launches /
  16→3 DtoD / 0 allocs, `perf-opcount`).
- **Round-E S1 keystone (merges `a963512`/`aa866ba`/`ead70c0`):**
  - **Yee-staggered transverse full-vector mode operator** wired into the mode
    selector: the six former strict-xfail TE10 eigenvector pins now pass as real
    asserts (`sin`-correlation `1.0000` across five dx tiers + high-frequency;
    `beta`-vs-analytic committed at `0.127% / 0.020% / 0.008%` for dx =
    `0.05 / 0.02 / 0.0125`); uniform-dielectric-fill routing carries real eps
    (selector-path pin), uniform-magnetic routes to the legacy operator, half-filled
    LSE hybrid physics validated to `~1e-14` vs an independent 1D Sturm-Liouville
    reference. Acceptance `docs/assessments/e1-rf-modes-acceptance-2026-07-19.md`,
    `e1-rf-mode-operator-acceptance-2026-07-19.md` (`test_te10_mode_selection.py`,
    `test_transverse_operator.py`).
  - **`rf/rectangular_waveguide` is now a committed wave-level PASS** (second
    validated scene): terminated TE10 two-port through the padded PML, S from
    `B=S·A`, gated on `cond(A) ≤ 10` + max singular value ≤ 1.05 + `beta` median rel
    err ≤ 1%; measured across three tiers `sin`-corr `1.0000`, `cond(A)`
    `1.09–1.27`, beta rel err `0.05–0.41%`. **External-reference cross-check**
    executed (one authorized cloud run, task id
    `fdve-3c2a2d95-4809-4dfb-98d6-1b6b5416c39a`, 0.025 FlexCredits): reference
    `beta_ref` vs analytic TE10 dispersion **median 1.21%, max 2.74%**. Acceptance
    `docs/assessments/round-e-integration-2026-07-20.md`
    (`tests/rf/wave_validation/test_waveguide_wave_level.py`).
  - **`rf/lumped_open_short_match` rebuilt to wave-level** on the coax line (SOL:
    matched `|Γ|max = 0.073` ≤ 0.1, short/open `|Γ| ≈ 0.998`, open/short phase
    separation 113–120°); **`rf/series_parallel_rlc` rebuilt** as an in-line
    two-terminal LumpedPort (series notch tracks `1/√(LC)`, `f_res·√C` spread ~1%;
    ±20% C ratios within 2% of analytic). Acceptance
    `docs/assessments/e2-rf-scenes-acceptance-2026-07-19.md`
    (`test_open_short_match_wave_level.py`, `test_rlc_resonance_wave_level.py`).
  - **PlaneMonitor passthrough** through WavePort/PortSweep Results delivered
    (`test_planemonitor_waveport_passthrough.py`).
  - **FDTD-driven antenna benchmarks** landed: `antenna/half_wave_dipole` real
    NF2FF `Result.antenna` PASS (E-plane `sin²` corr 0.9957, directivity 2.194 dBi
    vs analytic 2.156, power closure 0.041); `antenna/patch` pipeline valid but the
    matched-broadside `TM010` gate is an honest `strict=True` xfail (off-broadside,
    reactance-dominated). Acceptance `e2-rf-scenes-acceptance-2026-07-19.md`
    (`tests/rf/antenna/test_antenna_benchmark_e2e.py`).

**Round-F F2 trio (merge `0546b0a`):**
- **Interior-PEC masking on the Yee-staggered transverse operator + quasi-static
  electrostatic line-mode engine** (`witwin/maxwell/fdtd/excitation/modes.py`):
  symmetric Dirichlet elimination of PEC-occupied `Eu`/`Ev`/`Ew` unknowns, per-conductor
  connectivity check, and a variable-coefficient Laplace `div(eps grad phi)=0`
  capacitance-ratio (`eps_eff = C/C0`) engine for TEM/quasi-TEM lines. Operator-level
  gates (`tests/rf/wave_validation/test_interior_pec_operator.py`, 13 passed): coax
  `eps_eff = 2.2500` exact + `beta` to `rel ≤ 1e-6` of analytic and legacy;
  microstrip `eps_eff = 3.0920` vs Hammerstad–Jensen `3.0701` (`0.71%`, gate `≤ 3%`);
  diff-pair even/odd distinct (`3.2040` / `2.9048`, parity-classified); masked septum
  half-guide `beta = 2.4788` vs analytic `2.4760` (`0.11%`). Acceptance
  `docs/assessments/f2a-interior-pec-acceptance-2026-07-21.md`.
  - **Corrected TEM claim (audit remediation):** an earlier draft asserted the staggered
    curl-curl operator "structurally excludes TEM" — that is **false and retracted**. The
    operator carries the gradient TEM branch as the exact eigenvalue `eps·k0²`; it is the
    shipped occupancy rasterization (threshold 0.5, which eliminates the conductor-surface
    straddling normal-`E` samples) that drops TEM from the *masked* reduced operator, a
    masking choice, not a structural property. TEM/quasi-TEM lines therefore route
    fail-closed to the quasi-static engine (a `ValueError`, no census change). Evidenced by
    `test_masked_operator_tem_branch_is_a_masking_artifact` (keep-straddle threshold 0.75
    recovers `beta² = 56.25 = eps·k0²` exactly). The coax cross-implementation gate on the
    masked operator is deferred to the supervisor (switching the elimination criterion
    trades off thin-sheet masking).
- **`rf/microstrip_two_port` and `rf/differential_pair` un-BLOCKED to production quasi-TEM
  wave-level benches** (F2b, `benchmark/scenes/rf/*`, `benchmark/rf_validation.py`): the
  inhomogeneous interior-PEC quasi-TEM mode is wired into the WavePort path
  (`quasistatic_line_torch`), the terminated FDTD sweeps run, `B=S·A` extraction is
  well-conditioned (microstrip cond(A) 1.23, diff-pair 1.31), and mixed-mode conversion is
  correct (`|Sdd21|≈0.88 ≠ |Scc21|≈0.68`, `|Sdc21|≈0`). Status recorded **`gap` (measured,
  not forced)**, not BLOCKED. `tests/rf/wave_validation/test_microstrip_diffpair_wave_level.py`
  (6 passed). Acceptance `docs/assessments/f2b-quasistatic-benches-acceptance-2026-07-21.md`.
- **Adapter port/lumped source mapping + four external caches** (F2c,
  `witwin/maxwell/adapters/tidy3d.py`): `WavePort`→reference `ModeSource`+`ModeMonitor`,
  `LumpedPort`→`UniformCurrentSource` delta-gap filament; `TerminalPort` out of scope. The
  four previously `sources=0` scenes (`coax_thru`, `lumped_open_short_match`,
  `half_wave_dipole`, `patch`) now export runnable and were each cloud-generated (one
  authorized job, 0.025 FlexCredits each, task ids recorded; `benchmark/RESULTS.md` rows
  `generated`). `tests/api/adapters/tidy3d/test_port_source_mapping.py` (6 passed).
  Acceptance `docs/assessments/f2-rf-trio-acceptance-2026-07-21.md`.
- **Patch antenna feed diagnosis (fail-closed, xfail NOT flipped):** a galvanic PEC probe
  via cut the feed reactance ~5× but the patch still does not resonate (off-resonance
  historical band + lumped gap capacitively shorted by adjacent PEC + finite ground too
  small); the matched-broadside `TM010` strict xfail stays the fail-closed guard, reason
  string updated to the F2b diagnosis. Redesign deferred (multi-run antenna co-design).

**Not delivered / open gaps:**
- `rf/microstrip_two_port` and `rf/differential_pair` — **SUPERSEDED by Round-F F2
  above**: no longer BLOCKED; the interior-PEC masking + quasi-static line-mode engine
  landed (F2a) and the production quasi-TEM benches are wired and run (F2b). The
  *residual* gap is now **absolute `eps_eff` accuracy**: the shielded bench reads
  `eps_eff ≈ 1.86` vs H–J `3.27` (`~24%` low at dx = 5 mm), attributed to
  aperture-shielding + substrate resolution — **not** an extraction defect (the
  quasi-static engine itself converges monotonically to H–J, `<1%` at 16 substrate
  cells, `test_microstrip_eps_eff_converges_toward_hammerstad_with_resolution`).
  Recorded `gap` (measured), not BLOCKED. The **guided (non-TEM) interior-PEC** branch
  of `_assemble_vector_mode_data` is still on the legacy operator — the masked guided
  path is exercised only by the F2a operator tests, not production (no scene needs it;
  wiring also requires resolving the surface-sample rasterization tradeoff).
- Patch matched-broadside `TM010` + `D ≥ 5 dBi` remains an open physics gap (feed
  reactance + small finite ground), recorded as a strict xfail.
- `rf/series_parallel_rlc` parallel tracking slope is parasitic-diluted
  (direction-only gate) and validated at the dx=0.01 tier only — honest limitation.
- Retired Phase-1/2 headline gates re-classified and stripped of exit-gate status:
  coax/microstrip reciprocity = `symmetric`; power imbalance = `tautology`
  (`rf-wave-validation-2026-07-18.md` §2). (The matched/open/short and RLC
  `analytic-identity` retirements are superseded by the round-E wave-level rebuilds
  above.)
- External reference caches for the four port/lumped scenes — **SUPERSEDED by
  Round-F F2c above**: the adapter port/lumped source-mapping feature landed (owner
  decision point 1 funded), so all four scenes now export `sources ≥ 1` and were
  cloud-generated (0.025 FlexCredits each; task ids in
  `f2-rf-trio-acceptance-2026-07-21.md` §4; `benchmark/RESULTS.md` rows `generated`).
  These are generated cross-reference artifacts; wiring a per-scene numeric
  Maxwell-vs-reference comparison into each runner (as done for the waveguide in
  round E) remains a separate follow-on.

**Evidence grade:** **E2 for the validated scene set** — `coax_thru` (S1) and
`rectangular_waveguide` (round-E, with an external-reference cross-check) are
tracked `wave-level` PASS; the open/short/match and RLC coax rebuilds and the
dipole antenna benchmark are wave-level/real-NF2FF PASS. Round-F un-BLOCKED the
microstrip/diff-pair benches (now production quasi-TEM, recorded `gap` — the
extraction/mode physics validate but absolute `eps_eff` is resolution-limited); the
xfail patch broadside stays E0–E1 (honest strict xfail, not flipped). S2's §9.4
gates are `perf` (never a physical-usability grade).

### 02 — Multi-GPU execution

Plan of record `02-multi-gpu-execution.md` / blueprint `02-phase-7-8-blueprint-2026-07-16.md`;
progress `docs/plans/next-functional-2026-07/02-ensemble-progress.md`,
`docs/memory/fdtd-joint-solve-adjoint-s1-s2/s3/s4-progress-2026-07-17.md`,
`docs/memory/fdtd-multi-gpu-nccl-progress-2026-07-17.md`. Audit §1.2.

**Delivered (with evidence):**
- **Ensemble** (Phase 1/2 correctness): N independent Simulations over 2 GPUs
  reproduce serial output bit-for-bit (`torch.equal`); 4-port `PortSweep` expands
  into per-column tasks and reproduces the serial S-matrix bitwise (single-device)
  and to tight `assert_close` (2-GPU). `tests/fdtd/multi_gpu/test_ensemble_executor.py`,
  `test_ensemble_run_many.py`, `test_ensemble_network_sweep.py`, with recorded
  scratch-monkeypatch falsifications.
- **Joint-solve NCCL forward** end-to-end (one-process-per-GPU): seam-carries-signal
  precondition (160 steps, ~2.6× margin) + no-op-halo falsifications;
  `tests/fdtd/multi_gpu/_nccl_forward_worker.py`, `test_nccl_transport.py`.
  Fail-closed fences (monitors, coupled circuit/network/wire/port, trainable
  density, field shutoff) pinned and falsified.
- **Joint-solve adjoint internal infrastructure**: transposed halo transports
  (discrete-transpose pairing identity), phase-split standard reverse core and
  CPML reverse core both **bit-identical** pre/post refactor with load-bearing
  falsifications. S4 audit found the fused CPML reverse kernels are interface-correct
  per-shard (no new CUDA kernels, no psi halos needed).
- **Ensemble speedup** (one tracked figure): median **1.96×** on 2 GPUs at 128³,
  GPU-bound regime; 64³ is GIL/dispatch-bound (0.93×). `docs/assessments/ensemble-speedup-2026-07-17.json`.
- **Round-E S4 distributed CPML-trainable adjoint (merges `2364533`/`f7e8e9a`):**
  - **Distributed adjoint now delivered for the CPML-trainable envelope.** CPML-aware
    forward replay half-steps + a psi-carrying reverse loop thread the 12 psi
    cotangents per shard **with no psi halo** (the two Yee field halos carry the
    entire cross-interface coupling — falsified: no-op magnetic halo → rel 0.956,
    electric halo → 0.016 vs a 1.18e-7 baseline). The public
    `_validate_trainable_parallel_fdtd` is relaxed to accept `cpml`/`stablepml`
    (graded-sigma `pml`/`absorber` still reject) with an x-PML-pinned-to-outer-shards
    assertion. Acceptance `docs/assessments/e3-distributed-adjoint-acceptance-2026-07-19.md`.
  - **Pre-existing single-GPU CPML psi axis cross-wiring found and fixed** (commit
    `a2d2cb7`): the Hy/Ey `(pos,neg)=(z,x)` adjoint-psi carry and forward-replay psi
    storage read/wrote swapped keys — percent-level-wrong gradients whenever psi is
    numerically active. After fix, psi-active single-GPU adjoint vs central FD rel
    **9.1e-6** (was 3.43e-2); falsification reverting the keys reproduces the failure.
  - **psi-active distributed gradient parity** (the residual coverage gap): objective
    driven deep into the high x-PML band (STEPS=360, psi/E-H ratio 1.82, genuinely
    active), 1-vs-2-GPU gathered-gradient parity rel **5.94e-7** with a
    **~1.1e5× falsification** (zeroing the distributed psi carry → rel 6.55e-2).
    `tests/fdtd/multi_gpu/test_adjoint_parity_cpml.py -k psi_active`.
  - **DistributedFDTD-layer trainable guard** (defense-in-depth, closes the
    self-reported gap) + **forward monitor gather with a documented seam-ownership
    rule** and a double-count falsification (`test_monitor_merge_ownership.py`).
- **Round-E exclusive-window timing** (`docs/assessments/multi-gpu-timing-2026-07-20.json`,
  `exclusive-timing-window-2026-07-20.md`): ensemble makespan speedup **1.98–2.00×**
  at 96³/160³ (4 and 8 tasks, MAD < 0.4%); joint-solve forward strong scaling
  **0.544× at 128³** (communication-bound, 2 GPUs slower) → **1.726× at 192³**
  (in-process cuda_p2p); a no-feature bare-step regression spot-check shows −0.098%
  (no resolvable regression).
- **Round-G NCCL reverse-halo adjoint transport primitives (merge `42ac3f1`,
  acceptance `g1-nccl-adjoint-acceptance-2026-07-21.md`):**
  - **Transport-level reverse halos delivered** (`fdtd/distributed/nccl_transport.py`:
    `prepare_adjoint_staging` / `exchange_magnetic_adjoint` / `exchange_electric_adjoint`) —
    engine-based coordinator primitives that are exact transposes of the forward Yee
    halos, same `(engines, adjoint_states)` signature as the in-process
    `CudaP2PHaloTransport` so the reverse loop stays transport-agnostic. Gated by a
    **2-process `torchrun` discrete-transpose identity** `<A x, y> == <x, A^T y>`
    asserted **bitwise** (pure-copy halos, atol == 0) plus ghost-adjoint-zero and
    repeat-determinism (`tests/fdtd/multi_gpu/test_nccl_transport.py`,
    `_nccl_transport_adjoint_worker.py`). Falsification:
    `NCCL_TRANSPOSE_FALSIFY=magnetic|electric` corrupts only the local accumulation →
    identity fires, exit 1.
  - **Opt-in per-rank step-rate instrument** (`fdtd/distributed/instrumentation.py`
    `StepRateInstrument`, env-gated `WITWIN_FDTD_STEP_TIMING`): writes per-rank
    `step_timing_rank{r}.json`, **zero-cost-off asserted** (0 device synchronizations
    + no artifact when disabled; exactly `2+2N` when enabled), wired into the NCCL
    forward worker so the exclusive window can flip it on later. **No wall-clock number
    asserted** (shared-GPU). Addresses the round-E "not-measurable-by-hooks" finding.
  - **The end-to-end per-rank reverse DRIVER is honestly deferred, fail-closed**, with
    a written **7-step implementation plan** in the G1 acceptance doc (G1b §2): the
    validated in-process adjoint bridge is structurally single-process (global output +
    single autograd graph on rank 0), so a torchrun reverse needs a new
    distributed-collective driver (guard-relax, per-rank checkpoint capture, NCCL
    forward-replay dict halos, local separable seed, per-rank reverse loop, grad_eps
    gather + rank-0 pullback, gates). `_validate_nccl_capabilities` still raises for
    trainable density; census unchanged at **175**.
- **Round-H per-rank collective NCCL end-to-end reverse DRIVER + S5 tiled-plane seeds
  (merge `acea86e`, acceptance `h1-nccl-driver-acceptance-2026-07-21.md`):**
  - **The end-to-end per-rank reverse driver is DELIVERED** (the G1 7-step plan executed):
    a trainable Box-`MaterialRegion` density scene now backpropagates over a
    one-process-per-GPU NCCL launch and matches the single-process single-GPU adjoint.
    New forward-replay dict halos (`NcclHaloTransport.forward_electric_halo` /
    `forward_magnetic_halo`, the transpose siblings of the G1 reverse halos) route the
    replay's per-rank state-dict Yee x-planes collectively; a narrow `allow_adjoint` entry
    fence admits the trainable-density + point-monitor scene on the NCCL path (every other
    fence stays; census unchanged); `_NcclDistributedFDTDGradientBridge` reuses the
    in-process reverse math verbatim with the three single-process assumptions replaced
    (per-rank LOCAL output pack, separable local objective seeded per rank, slab-wise
    `grad_eps` gather + rank-0-only pullback); off-owner ranks still drive the full
    collective reverse so no deadlock, and the rank-0 gather-capacity preflight is
    **collective-safe** (all ranks `allreduce` the verdict and raise together). Gates
    (2 procs × 1 GPU, honest 1e-4-class tolerances): objective+grad parity for standard
    open-boundary cross-seam / x-CPML interior / x-CPML **psi-active** (~2e-7); reverse
    magnetic/electric halo + psi-carry falsifications red (rel 9.31e-1 / 1.64e-2 /
    6.88e-2); repeat-determinism < 1e-9; symmetric guard/capacity reject with no hang.
  - **S5 separable tiled-plane monitor adjoint seed delivered:** a y/z-normal `PlaneMonitor`
    objective is now seedable on the NCCL driver — each rank sums its owned plane strip
    (`plane_monitor_l2_objective`, owned-`owned_local_slice[0]` crop) so the world sum
    reproduces the single-process full-plane objective with every seam cell counted once and
    no cross-rank cotangent scatter. Parity vs an independent single-GPU full-plane adjoint
    at the point-gate tolerances; **seam-ownership falsification** (sum the FULL local strip
    → double-count the live seam cell) reddens parity (grad rel 4.005e-1). Flux/mode/finite/
    x-normal-plane objectives stay fail-closed (they need seam-crossing tangential-field
    assembly, not an owned-cell separable sum), rejected by
    `require_distributed_adjoint_objective_support`.
  - **Cross-stream caching-allocator race found and fixed** (commit `c233d8b`): before the
    fix the driver's gathered gradient was **load-dependent** — bitwise-correct on exclusive
    GPUs (~1.5e-7) but deterministically wrong at the partition seam under concurrent GPU
    load (standard rel ~1.2e-4, CPML ~3.1e-4, seam-spanning plane ~2.7e-2, occasional
    blow-ups), with the forward loss staying bitwise identical (isolating the defect to the
    distributed reverse). Root cause: the four reverse/replay NCCL halos ran their in-place
    accumulate + posted the collective inside `with torch.cuda.stream(engine.compute_stream)`,
    but the per-step adjoint-state planes they touch are allocated on the **default** stream
    by the reverse kernels; the CUDA caching allocator tags a freed block only with its
    allocation stream, so under load a default-stream reuse could overwrite a block the
    compute-stream halo was still writing (the live-forward halos are immune — they slice
    persistent field storage — which is why the forward loss stays clean). Fix: run those
    four halos on the current (default) stream (the reverse driver already host-synchronizes
    between phases, so `compute_stream` bought no overlap) — matching allocation-stream to
    use-stream closes the window at zero cost. **Honest tolerances restored** (the prior
    round's rtol-1e-3/atol-1e-2 grad-gate widening that laundered the drift is reverted) and
    a **committed stressed-mode parity gate**
    (`test_two_rank_nccl_adjoint_parity_under_stress[standard|cpml|plane]`) runs the three
    objectives at the honest 1e-4 gate under an in-worker saturating co-tenant burner (max
    ~3e-7). Load-bearing falsification: reverting the four halos collectively onto
    `compute_stream` under the burner reddens standard/CPML deterministically (~1.223e-4 /
    ~3.120e-4) and the plane gate intermittently (worst-case run 2.684e-2).
  - **Round-H recorded a "new open item" here (in-process `cuda_p2p` load hazard); it is
    now CLOSED (I1, `1a579b3`, merged `625baca`).** Reproducing it first showed it was a
    *distinct* hazard from this NCCL allocator-reuse race — a checkpoint-capture
    happens-before race in `capture_distributed_checkpoint` (default-stream clone unordered
    w.r.t. the next compute-stream forward update), not the caching-allocator reuse class
    (discriminated by `CUDA_LAUNCH_BLOCKING=1` collapsing the drift to the ~2e-7 floor while
    `PYTORCH_NO_CUDA_MEMORY_CACHING=1` left it untouched). Fix: clone on the shard's
    `compute_stream`; a committed stressed parity gate + falsification pin standard/x-CPML
    1-vs-2-GPU parity ~2e-7 under a co-tenant burner (reverting reddens to ~8.09e-2), census
    unchanged at 176. Source: `docs/assessments/i1-p2p-race-acceptance-2026-07-21.md`.

**Not delivered / open gaps:**
- **NCCL end-to-end reverse driver + S5 y/z-plane seeds DELIVERED (Round H, merge
  `acea86e`)** — see the Round-H block above. **Still OUT of scope, fail-closed:** flux /
  mode / finite-plane / x-normal-plane NCCL adjoint objectives (they need seam-crossing
  tangential-field interpolation, not an owned-cell separable sum; rejected by
  `require_distributed_adjoint_objective_support`), CPML **psi-carrying** NCCL reverse beyond
  the delivered psi-active point/plane parity, monitor gather **beyond the forward path** (the
  driver reads each rank's shard-local monitor output directly, not a collective per-monitor
  gather), and coupled-runtime (circuit/network/wire) joint solve — the blueprint #13/#18 tail.
- **In-process `cuda_p2p` bridge cross-stream load hazard — CLOSED (I1, `1a579b3`, merged
  `625baca`).** Once reproduced it proved a *distinct* hazard from the NCCL allocator-reuse
  race (`c233d8b`): a checkpoint-capture happens-before race in
  `capture_distributed_checkpoint` — the mid-forward clone read persistent field storage on
  the device default stream while forward updates run on each shard's `compute_stream`, so
  under load the next update tore the snapshot and the replayed seam gradient drifted
  (~8.09e-2) while the forward loss stayed bitwise clean. Fix: clone on the shard's
  `compute_stream` (serializes previous-update → clone → next-update on one stream, zero
  added host-sync). Committed stressed parity gate
  (`tests/fdtd/multi_gpu/test_adjoint_parity_stress.py`) + falsification hold standard/x-CPML
  1-vs-2-GPU parity ~2e-7 under a co-tenant burner; census unchanged at 176. Source:
  `docs/assessments/i1-p2p-race-acceptance-2026-07-21.md`.
- **NCCL driver timing pending an exclusive window:** the opt-in per-rank step-rate instrument
  (Round G) is delivered and unit-tested, but the driver produces wall-clock numbers only
  under an exclusive window (none asserted here; correctness-only shared GPUs).
- Joint-solve forward strong scaling is grid-conditional (payoff only ≥192³; 128³ is
  communication-bound). Ensemble+joint-solve composition still rejected.
- `_validate_static_capabilities` still rejects Bloch/x-periodic/x-symmetry,
  MaterialRegion density (except CPML/stable-PML trainable), ports,
  closed-surface/diffraction/flux-time/non-point time monitors, SIBC, material
  monitors, split-face Ex, non-point/non-uniform sources — the distributed path is
  mutually exclusive with most RF/research features. Long-run float32 halo drift
  known.

**Evidence grade:** forward **E2**; **CPML-trainable distributed adjoint E2**
(psi-active 1-vs-2-GPU parity 5.94e-7, load-bearing falsified); **NCCL reverse-halo
adjoint transport E2** (Round G, bitwise 2-process transpose identity); **NCCL
end-to-end reverse DRIVER E2** (Round H — per-rank point/plane objective+grad parity
~2e-7 including psi-active, honest 1e-4-class gates green on exclusive GPUs AND under a
saturating co-tenant burner after the cross-stream allocator-race fix, load-bearing
falsifications recorded). Timing measured on the exclusive window (ensemble 2×,
joint-solve grid-conditional); NCCL driver timing pending an exclusive window.
Coupled-runtime joint solve **E0**; the in-process `cuda_p2p` load hazard is **CLOSED**
(I1, `625baca`, distinct checkpoint-capture race fixed + stressed gate — see gaps).

### 03 — Touchstone network embedding

Plan: `docs/plans/next-functional-2026-07/03-touchstone-network-embedding.md`
(status `reopened-for-evidence`). Audit §1.3. Reference
`docs/reference/network-embedding.md`, memory
`docs/memory/network-embedding-progress-2026-07-16.md`, `network-embedding-implementation.md`.

**Delivered (with evidence):** Phases 0–3 (rational-fit → state-space → synchronous
per-step coupling, pivoted-LU, CUDA-graph replay, single-device adjoint) landed and
in `FEATURE_LIST`. Phase 4 gate (a) residue/direct-term gradients <2%
(`tests/gradients/test_fdtd_network_adjoint.py`); gate (b) single-vs-2-GPU port V/I
bitwise parity (`tests/fdtd/multi_gpu/test_network_owner.py` `split_net`) — see
Caveats C-2; gate (c) no-feature regression <1% via op-stream equivalence
(`docs/assessments/network-embedding-phase-4-no-feature-op-stream.json` — per-step
kernel identity, one-time prepare delta of 4 aten calls so the artifact's top-level
flag is `equivalent:false`). Trapezoidal port-interface unification pinned
(`tests/rf/circuits/test_fdtd_circuit_coupling.py` etc., `rtol≈2e-6`).

**Round-E E2 evidence (merges `6c621a2` = `3e7b0c4` + `c328fe7`):**
- **Public `NetworkData.cascade(...)` / `terminate(...)` helpers** — first-principles
  N-port star connection + single-port closure, batched over frequency,
  differentiable, no third-party dependency, fail-closed on complex/mismatched
  reference impedance etc. (`witwin/maxwell/network.py`).
- **Independent raw-sample S-cascade cross-check** breaking the fit-model-class
  circularity: raw Touchstone samples (read via `from_touchstone`, NOT the rational
  fit) connected by the first-principles algebra vs the same network embedded in the
  time domain (rational fit + state-space stepping) — **no shared code path**.
  Observed residuals `~4.6e-9` (lossy) / `~5.7e-8` (reactive), gate `<1e-5`, with the
  connection changing S11 by `~2e-4` (≫ tol). `tests/rf/network/test_network_cascade_crosscheck.py`,
  acceptance `docs/assessments/e4a-network-cascade-acceptance-2026-07-19.md`.
- **Multi-scenario passivity/conservation suite** (3 embedded FDTD scenarios): terminal
  power-balance (`|P1−P2|/|P1| < 1e-3`, honestly annotated *consistency*-class for the
  two memoryless scenarios and *genuine two-sided* for the reactive one), passivity
  (`generated ≤ 1e-6·absorbed`, cumulative net ≥ 0), stability (T→2T convergence).
  `test_network_conservation.py`.
- **Composite same-step coupling** cut kernel launches **78→27 (−65.4%)** by replacing
  the sequential triangular LU substitution with two precomputed composite matvecs
  (`gain_state = M⁻¹C`, `gain_voltage = M⁻¹D`), bitwise-identical eager vs CUDA-graph;
  `docs/assessments/e4-network-coupling-op-stream-2026-07-19.json` (`test_network_coupling_op_stream.py`).
- **Explicit-delay checkpoint/resume** landed (bidirectional ring + Thiran filter
  memory in the frozen schema; fixes a latent lossy-resume-from-zero bug; bitwise
  `run_until`+`resume` reproduction) — **but the differentiable adjoint through
  explicit delay stays fail-closed** (segment-crossing reverse ring + IIR reverse
  out of scope; both rejection sites fail closed with the precise reason).
  `test_network_delay_checkpoint.py`.
- **WavePort embedding disposition**: stays fail-closed with an accurate reason (a
  modal WavePort has no scalar (V,I) terminal contract — a **missing design contract**,
  not a bug). `test_network_block_contract.py`.

**Not delivered / open gaps:**
- **Gate (d) still grid-conditional** (re-measured round-E after the composite matvec
  coupling): 8-port/order-32 step overhead <10% holds only at **≥224³**
  (`224³`=9.08%/CI95-up 9.12% PASS; `64³`=128% FAIL). The composite matvec cut
  launches 65% but the fixed per-step cost dropped only ~9% (median **0.183 ms/step**,
  was 0.193–0.204) — the **key finding is the connected step is compute-bound**
  (implicit solve + port observers), not launch-bound, so further launch reduction
  will not move the ~224³ crossover. Artifact
  `docs/assessments/network-embedding-gate-d-remeasure-2026-07-20.json`,
  `exclusive-timing-window-2026-07-20.md`.
- Multiport S-parameter <0.02 gate **not independently backed** (reference inside the
  fit's own model class; margin float64-round-trip dominated) — separate from the new
  raw-sample cascade cross-check, which validates the connection algebra + embedding.
- Explicit-delay differentiable adjoint fail-closed (checkpoint schema is the
  prerequisite, now landed; the reverse ring + IIR reverse remain). `WavePort`
  embedding = missing design contract. Distributed trainable + spatial multi-GPU
  composition fail-closed (needs 02 Phase 7 result-aggregation). Inherits 01
  port-power-convention risk — now **partly** wave-validated upstream (`coax_thru` +
  `rectangular_waveguide`), not fully.

**Evidence grade:** **E2** (embedded path) — the round-E independent raw-sample
S-cascade cross-check (`<1e-5`, code-path-independent) + multi-scenario
passivity/conservation suite supply the E2 evidence the audit required. Gate (d)
stays grid-conditional (compute-bound ruling); the E3 upgrade needs the combination
matrix + named-hardware performance envelope + public benchmark residence.

### 04 — SPICE/MNA co-simulation

Plan: `docs/plans/next-functional-2026-07/04-spice-mna-cosimulation.md`
(status `reopened-for-evidence`). Audit §1.3. Acceptance
`docs/assessments/spice-mna-phase-4-acceptance.md`, memory
`docs/memory/04-spice-mna-cosimulation-progress-2026-07-16.md`.

**Delivered (with evidence):** Phases 0–3 (rational/companion coupling,
pivoted-LU, CUDA-graph replay, single-device adjoint) landed and in `FEATURE_LIST`.
All four Phase-4 gates evidenced:
(a) parameter gradients vs FD <1% (`tests/gradients/test_fdtd_circuit_adjoint.py`, 25/25);
(b) single-vs-2-GPU parity bitwise (0.000e+00 on six fields + port V/I + circuit
node/branch, `communication_bytes_per_step=8`; `tests/fdtd/multi_gpu/test_circuit_owner.py`, 7/7);
(c) 32-unknown step overhead <10% (matched baseline −64.5%, single factorization,
cost flat 8/32/128 unknowns; `docs/assessments/spice-mna-phase-4-circuit-performance.json`);
(d) no-circuit single-GPU regression <1% via **op-stream equivalence**
(`-no-feature-op-stream.json`, identical 512-step op table, `equivalent:true`) with an
**A/A calibration pattern** as the controlling row (Phase-3-vs-itself −0.523%,
`-no-feature-aa-calibration.json`; the clock-floored +1.227% B-run lies inside the
calibrated no-difference band).

**Round-F F1 evidence (merge `07e8e99`, closes the E2-blocking gap):**
- **Coupled EM+circuit conservation suite** (`tests/rf/circuits/test_circuit_conservation.py`,
  6 passed): three genuinely two-way-coupled scenarios (resistive load / series RLC /
  VCVS network), each driven from an in-circuit source in a closed PEC vacuum box, close
  the global balance `S_source = dU_field + dU_circuit + D_circuit` to a bounded
  half-step-stagger residual (**absolute constant in step count**, ~1.6–3.1e-14 J;
  relative 1.0e-4–6.7e-3 of throughput at 6000 steps, gate 1.5e-2). Honest gate classes
  are annotated: the field-coupling term rides the load-bearing **two-sided field-link
  gate** `dU_field == -W_port` (raw Yee E/H energy vs the MNA port V/I record, no shared
  code path; ~2.9e-3 of peak field, gate 2e-2), while `S_source`/`D_circuit`/`dU_circuit`
  are Tellegen/companion **consistency-class** within the coupled run. The port is bound
  *behind* a series impedance so the field genuinely back-reacts (anti-degeneracy guard
  `test_scenarios_are_distinct_em_coupling_cases`). Falsifications: in-suite +3% channel
  imbalance rejected; field-link injection over-scatter drives the residual 1.5%→44%.
- **Independent offline circuit cross-check** (`tests/rf/circuits/test_circuit_independent_crosscheck.py`,
  4 passed) — the E2-blocking item: a hand-derived equivalent-circuit ODE integrated by
  `scipy.integrate.solve_ivp` **sharing no runtime code** with the MNA solver predicts the
  coupled port voltage of a one-port EM structure characterized from a **different** drive
  waveform and series resistance. Headline gate: port-voltage rel err **1.16e-5** (tol
  5e-4, ~43× headroom); port-current corroboration (cancellation-limited) 7.02e-3.
  Falsification: perturbing the MNA field-port companion conductance by 1.05 drives the
  coupled voltage off the independent prediction (1.16e-5→4.10e-3, ~350× separation). This
  lifts `S_source`/`D_circuit` and the resistive port coupling off consistency-class.
  Acceptance `docs/assessments/f1-cosim-e2-acceptance-2026-07-21.md`.

**Not delivered / open gaps:** the end-to-end EM+circuit strong-coupling gate still
has no external reference (tagged `reference: future-xfdtd`); F1b's cross-check circuit
is source + series **R** only, so the reactive companion storage `dU_circuit` (C/L)
**stays consistency-class**, not lifted by F1b; the external-reference lumped-load
cross-check is still pending. gate (c) −64.5% is a matched baseline (native SeriesRLC
termination is itself slow), not "co-simulation is nearly free" (+407% vs bare FDTD).
Distributed adjoint rejected before allocation (forward-only multi-GPU contract).
Inherits 01 port-power risk.

**Evidence grade:** **E2** (re-annotated from E1–E2) — the F1 multi-scenario
conservation suite + code-path-independent offline circuit cross-check supply the E2
evidence the audit required for S3; the E3 upgrade still needs the combination
matrix + named-hardware envelope + public-benchmark residence (+ an external
strong-coupling reference the reference solver does not cover).

### 05 — Nonlinear circuit devices

Plan: `docs/plans/next-functional-2026-07/05-nonlinear-circuit-devices.md`.
Audit §1.6 / §3 row 05. Census
`docs/reference/fdtd-capability-guard-census.md` (nonlinear reconciliations).

**Delivered (with evidence):** Phase 0 Device + Newton contract
(`witwin/maxwell/circuit_devices.py`, `compiler/nonlinear_devices.py`) and the N1
standalone charge-aware transient solve (`NonlinearMNASystem` folds the
trapezoidal/backward-Euler stored-charge companion; `run_nonlinear_transient` with
gmin/source DC continuation). Fail-closed hardening so a diode cannot run silently
absent (three capability guards on the linear/coupled/FDTD paths; budget 140→139
after N1). `q(v)`/`C(v)` exercised through `CompiledNonlinearDevice.charge`.

**Not delivered / open gaps:** FDTD coupling, transient companion into the Yee
update, adjoint, and benchmarks are later slices — **solver does not consume** the
nonlinear device in the field path. Diode `series_resistance` internal-node ohmic
branch still fails closed. BJT/MOSFET reserved and fail-closed (contract guards;
independent Phase-5 go/no-go).

**Evidence grade:** **E0** (contract + standalone-transient layer only).
**Freeze:** Wave C — S3 passed 2026-07-21 (unfrozen); plan 05 is not among the Round G
tracks (02/07/08/09), so its solver consumption is eligible but not yet scheduled.

### 06 — Array / Active-S / MIMO

Plan: `docs/plans/next-functional-2026-07/06-array-active-s-mimo.md`
(status `reopened-for-evidence`). Audit §1.4. Acceptance
`array-active-s-mimo-phase-3/-phase-4-acceptance.md`.

**Delivered (with evidence):** Phases 0–3 + Phase-4 **weight gradients** —
codebook/scan/max-hold, field- and S-parameter-based MIMO metrics (ECC vs Clarke
closed form), all in the autograd graph, gradcheck-passing
(`tests/rf/array/test_array_contracts.py`, `test_array_codebook.py`,
`test_array_mimo.py`; full suite 65 passed). The frozen **96³/4096-step
qualification PASS** landed post-audit on the exclusive window: power closure
`|P_accepted−P_rad|/P_incident = 0.0697%` (<1% gate), basis+16-combine = 20.55% of
16 direct solves (≤40% gate), one combine = 0.0343% of a direct solve (<10% gate),
zero extra FDTD steps. `docs/assessments/array-active-s-mimo-phase-1-qualification.json`
(`"verdict":"PASS"`, `"qualifying":true`).

**Round-F F3 (merge `7ec99c7`) — the Phase-4 exit-gate item closes:**
- **F3a single-device `ArrayBasisData.scene_gradient_vjp(...)` delivered**, replacing the
  `NotImplementedError` (capability-guard census `176 → 175`, reconciled). Far-field VJP
  propagates through the embedded-pattern basis columns to scene/material/geometry
  parameters via an aggregated per-column adjoint; the weight-conjugation seed
  `seed_n = conj(w_n)·cot_E` is **bit-for-bit** identical to end-to-end `autograd.grad`,
  and a real two-column FDTD array (trainable `MaterialRegion` density, NF2FF column) passes
  central-difference gates within the FDTD-adjoint band. Acceptance
  `docs/assessments/f3-array-vjp-acceptance-2026-07-21.md`.
- **F3b 2-GPU ensemble aggregation** (`witwin/maxwell/array_gradient.py`:
  `aggregate_scene_gradient_vjp` / `ensemble_scene_gradient_vjp`): per-column forwards
  distribute over the ensemble `DevicePool` as independent Simulations (the NCCL joint-solve
  adjoint stays out of scope); the seeded backward + a fixed public-port-order reduction run
  on the caller thread. **1-vs-2-GPU aggregated-gradient parity is measured BITWISE**
  (`maxabsdiff = 0`, `torch.equal`) on the homogeneous A6000 pair — both the synthetic
  float64 map and a real two-column FDTD array — so the F3a-anticipated "`<1e-12` floor or
  bitwise" resolves to bitwise. Falsifications: dropped weight conjugation, skipped column,
  and removed cross-device gradient move each go RED. Acceptance
  `docs/assessments/f3-array-scene-vjp-acceptance-2026-07-21.md`.

**Not delivered / open gaps:** throughput / speedup is **not** claimed this round
(shared-GPU correctness window) — recorded deferred-pending-exclusive-window. Network /
S-parameter scene gradients, `combine()` weight gradients (regression-gated unchanged),
and the NCCL joint-solve adjoint remain out of scope. Task-level multi-GPU was **removed
from scope** by user decision earlier. All EIRP/realized-gain still descend from 01's
port-power chain (not wave-level validated).

**Evidence grade:** **E2** (re-annotated from E1–E2) — the F3 scene-gradient VJP
(bit-for-bit vs the single-device path, 1-vs-2-GPU bitwise parity, central-difference
gated, load-bearing falsified) closes the Phase-4 exit-gate item that capped this plan.

### 07 — Thin-wire model

Plan: `docs/plans/next-functional-2026-07/07-thin-wire-model.md`
(status `reopened-for-evidence`). Audit §1.5. Acceptance
`thin-wire-phase-2/-phase-3-acceptance.md`, memory
`thin-wire-finite-conductivity-model-2026-07-17.md`.

**Delivered (with evidence):** PEC paths (straight/bent/branched/closed-loop) —
forward + single-device adjoint (radius rel err 1.8e-5, host-material 2.5e-4) +
**multi-GPU forward with bitwise parity** and energy-consistent recurrence
(`tests/fdtd/multi_gpu/test_wire_owner.py`: cross-split bitwise = single GPU,
closed-loop energy parity, deterministic reduction order, Mur+wire fail-closed).
Phase-3 arbitrary-direction conservative coupling, nonuniform/real-periodic grid,
fixed-stencil geometry adjoint, single-device port binding
(`thin-wire-phase-3-acceptance.md`). Phase-4 **finite-conductor series-impedance
layer** (analytic scaled-Bessel internal impedance, exact DC, passive rational ADE
fit reusing the shared network rational stack, <2% analytic gate;
`witwin/maxwell/compiler/wire_impedance.py`, `tests/fdtd/thin_wire/test_wire_finite_conductivity.py`).

**Round-G B2 lossy recurrence + real `ohmic_loss` + B3 conductivity adjoint (merge
`3884bb7`, acceptance `g2-lossy-wire-acceptance-2026-07-21.md`):**
- **B2 passive lossy-current ADE companion consumed by the runtime**
  (`fdtd/wire_lossy.py`, `fdtd/wire.py`): the per-unit-length internal series impedance
  folds into the current update as an energy-consistent trapezoidal (Tustin) companion
  with the loss terms on the integer-step trapezoidal current; the finite-conductor
  compile deferral is removed. Gates (`tests/fdtd/thin_wire/test_wire_lossy_recurrence.py`
  9 passed, `test_thin_wire_lossy_forward.py` 7 passed): **analytic-AC** realized
  internal resistance vs scaled-Bessel pre-registered **< 8%** (fit-limited, NOT the 2%
  compile-layer gate — the shared complex vector fit recovers `Re(Z')` only to ~1–6%
  nondeterministically, blocker B1); **DC** `R0 == R_dc·length` to rel **1e-12** (exact);
  **PEC-parity** the PEC (`lossy_model is None`) path is **bitwise** identical run-to-run
  (and SHA256-identical vs base, auditor-confirmed); **stability** an adaptive spectral
  certificate fits the highest order in `[6,13]` with combined `[I;x]` spectral radius
  `< 1` (else fails closed). Falsifications committed
  (`docs/assessments/g2-lossy-wire-probes/falsify.py`): zeroing the ADE feedback blows the
  analytic-AC error 0.0225→1205; +1% `R0` breaks DC exactness; order-16 without the
  certificate gives spectral radius 1.000075.
- **Real `ohmic_loss` monitor**: emits `0.5·Re(Z')·length·|I(f)|²` per segment (was zeros);
  PEC segments report exactly zero; PEC ohmic-only monitor returns a zeros channel (fixed
  a blocking PEC regression that raised at finalize). The reported dissipation matches the
  companion formula to `rtol 1e-4` on the run's own fit.
  - **Energy-closure honest substitution history:** the shipped
    `test_energy_closure_single_tone` is a **companion-level** closure only (isolated
    per-segment input power == companion dissipation + reported-formula identity, load-bearing
    assertion falsified by a 1.2× `ac_resistance` scaling). The brief's **closed-box 3D
    field-energy closure** (source input == field-energy delta + radiated/absorbed + wire
    ohmic on a real FDTD run) was **NOT performed** — scope reduction, supervisor sign-off;
    physical risk assessed low, but the wire↔field coupled-passivity guarantee is an open item
    (the companion is not positive-real by construction; it is active out of band, stability is
    an isolated per-segment spectral-radius certificate).
- **B3 conductivity adjoint** (deterministic dissipation channel):
  closed-form `dZ'/dσ` of the analytic scaled-Bessel impedance
  (`compiler/wire_impedance.py::internal_impedance_conductivity_gradient`) + a PyTorch-native
  autograd path (`analytic_ac_resistance`). Gated vs float64 central difference **< 1e-6**
  across a frequency sweep and DC limit `dR_dc/dσ = −1/(π a²σ²)` exact
  (`tests/gradients/test_fdtd_thin_wire_conductivity_adjoint.py` 7 passed;
  `verify_grad.py` committed). Falsifications: `(1−R²)→(1+R²)` and a 2× backward scaling
  each go RED.

**Not delivered / open gaps:** the **field-coupled current sensitivity `dI/dσ`** through
the multi-step recurrence **stays fail closed** — the ADE coefficients come from the
nondeterministic shared rational fit (B1), not a differentiable/reproducible map of σ, so
an exact reverse replay cannot be certified (`fdtd/wire.py::replay_wire_state` sharpened,
not lifted). The ADE loss-state checkpoint/resume fails closed (B3+ schema). **B4
distributed lossy reverse** fails closed; the **distributed lossy forward** now fails
closed too (`fdtd/distributed/solver.py::_validate_distributed_wire_support`) — closing a
verified silent-PEC hole where the distributed owner built only the lossless update. No
scene-level trainable-conductivity leaf yet (`WireConductor.finite(conductivity=)` takes a
Python float; the shipped adjoint is the model-layer differentiable readout). No wall-clock
timing (shared GPU); the lossy update is a torch path gated to lossy wires (PEC CUDA kernel
untouched, brief-sanctioned). Some Phase-3 impedance/far-field numbers flagged UNEVIDENCED
(regenerate before citing).

**Evidence grade:** **PEC E2 / lossy E1–E2** — B2 recurrence (analytic-AC/DC/PEC-parity/
stability gated + falsified) and B3 conductivity adjoint (closed-form vs central-difference
< 1e-6, autograd == FD, falsified) landed; the lossy grade is capped below full E2 by the
fit-limited 8% analytic-AC gate, the companion-level (not closed-box) energy substitution,
and the fail-closed field-coupled `dI/dσ`.
**Freeze:** lossy-wire solver consumption is Wave C step S6, now **unfrozen (S1–S3 all
passed 2026-07-21)**; Round G B2+B3 landed (merge `3884bb7`).

### 08 — Gyromagnetic ferrite

Plan: `docs/plans/next-functional-2026-07/08-ferrite-materials.md`. Audit §1.6 /
§3 row 08. Reference `docs/reference/ferrite-physics-contract.md`. Census
`fdtd-capability-guard-census.md` (gyromagnetic reconciliations).

**Delivered (with evidence):** Phase 0 physics contract + torch reference oracle +
`GyromagneticFerrite` material type + Polder tensor + energy-passivity proof; an
axis-aligned forward slice (`compile_gyromagnetic_layout`,
`fdtd/runtime/gyromagnetic.py`) so an axis-aligned ferrite compiles as its diagonal
background with the gyrotropy produced by the magnetization-ADE hooks. A
**fail-closed export guard** now rejects gyromagnetic media in the adapter — a
`GyromagneticFerrite` previously exported silently as a plain isotropic medium,
dropping the Polder tensor (`adapters/tidy3d.py`; guard census 143→144).

**Round-G arbitrary-bias forward + mixed-bias support (merge `5dd100e`, acceptance
`g3-ferrite-bias-acceptance-2026-07-21.md`):**
- **General (non-axis-aligned) bias forward** (`fdtd/runtime/gyromagnetic.py`): the
  axis-aligned fail-closed guard is lifted. The general path is a pure per-cell
  coordinate rotation of the SAME axis-aligned contracted implicit-midpoint (Cayley)
  magnetization ADE — RF drive projected onto the per-cell right-handed local frame
  `[u|v|w]` (`w = b̂`) and back-reaction scattered onto all three lab components; the
  2×2 solve is proven frame-independent (module docstring), so **no new integrator or
  coefficients**. The axis-aligned fast path is retained as an exact optimization.
- **Mixed-bias support** (disposition: SUPPORT, not reject): the "single uniform bias
  direction" guard is removed; different bias axes / opposed signs (`+z`/`−z` latching
  circulator) / differing materials route through the per-cell general path. The
  magnetization ADE carries no spatial coupling, so a mixed-bias scene is the exact
  direct sum of independent per-cell passive blocks.
- **Gates** (`tests/materials/ferrite/test_gyromagnetic_general_bias.py`, mock float64 +
  CUDA; suite 107 passed): **rotation-equivalence** general reduces to fast **bit-for-bit**
  (max|diff| = 0.0, b∈{z,x,y}, incl. the production CUDA solver over 200 steps/6 fields);
  **oblique-vs-oracle** b=(1,1,1)/√3 vs a discrete Polder oracle `chi_uu` rel **1.197e-13**
  ≤ 1e-5; **handedness** oblique bias reversal flips lab gyrotropy, co-pol unchanged;
  **Polder spot-check** compiled tensor == frozen Polder, antisymmetric part flips under
  reversal (rtol 1e-10); **mixed-bias per-cell independence** combined == direct sum
  bit-for-bit; **zero-impact** ferrite-free and PEC-only scenes leave hooks bitwise no-op;
  **CUDA oblique driven-cavity passivity** energy-envelope non-growth over 12k steps.
  Falsifications recorded (dropped `uz` projection breaks the oblique oracle + b=y reduction;
  global-frame substitution breaks mixed-bias independence; sign-stripped frame breaks
  handedness).
- **Contract-doc supersession**: `docs/reference/ferrite-physics-contract.md` §7 item 6 is
  marked **superseded** (mixed / per-material bias now ships via the per-cell general path),
  historical text retained.

**Not delivered / open gaps:** Bloch-periodic ferrite, FDFD ingest, multi-GPU (ShardEngine),
adjoint (no reverse core for the non-reciprocal correction), and `PerturbationMedium` over a
ferrite all **fail closed**. Scalar frequency-evaluation is a permanent contract. **Identity
collocation** (`C = I`, each lab `H` truncated to the shared cell overlap) is reused from the
axis-aligned slice — accurate for smooth/uniform fields and passive in a closed cavity, but
it is NOT the 4-point-collocated Yee gather; higher-order collocation is a later refinement.

**Evidence grade:** **E1** (arbitrary/mixed-bias forward: rotation-equivalence/oblique-oracle/
handedness/passivity/zero-impact gated + falsified; no external reference, no adjoint).
**Freeze:** Wave C — **unfrozen (S3 passed 2026-07-21)**; Round G general+mixed-bias landed
(merge `5dd100e`; census `175 → 173`, both bias guards removed).

### 09 — Surface impedance / metal roughness

Plan: `docs/plans/next-functional-2026-07/09-surface-impedance-metal-roughness.md`.
Audit §1.6 / §3 row 09. References
`docs/reference/rf-port-conventions.md` (SIBC runtime), memory
`sibc-resistive-stability-fix-2026-07-17.md`, `surface-impedance-phase0-p09-fixer-2026-07-17.md`.
Census `fdtd-capability-guard-census.md` (surface-impedance reconciliations).

**Delivered (with evidence):** Phase 0 contract + rational model + shared fitter +
fail-closed funnel + adapter export fail-closed. Phase-1 runtime generalization
(axis-aligned exposed-face layout, mid-domain double-sided plates, multiple
metals/orientations, generic per-edge Z-form ADE forward with the narrowband good
conductor as the order-0 case). The **SIBC resistive-Leontovich stability fix**
landed (dropped the non-passive inductive term; stateless surface update) with the
3D-empirical stability domain characterized across the good-conductor regime
(`tests/validation/physics/test_lossy_metal_sibc.py`, 3/3 pass; falsification by
re-adding the inductive term reproduces divergence).

**Round-G all-orientation staircased SIBC + cylinder physics gate + skin-effect bench
(merge `c9b4bfc`, acceptance `g4-sibc-oblique-acceptance-2026-07-21.md`):**
- **All-orientation staircased exposed-face SIBC** (`compiler/materials.py`,
  `fdtd/runtime/materials.py`, `fdtd/runtime/stepping.py`): a non-`Box` good conductor is
  staircased from its node occupancy — every axis-aligned voxel face becomes a masked
  Leontovich surface-impedance write, covering cylinder/sphere, all six orientations, and
  mixed orientations. Orientation-equivalence (cyclic-permutation residual ~1.7e-7, float32
  round-off) + mixed-orientation stability gates
  (`tests/validation/physics/test_sibc_orientation.py` 6 passed,
  `test_sibc_staircase.py` 6 passed). Falsifications: face-normal sign flip breaks
  orientation-precise equivalence; active-branch sign flip diverges (Box + voxel paths).
- **Staircased-cylinder physics convergence gate**
  (`test_sibc_cylinder_convergence.py`, 5 passed): absorbed power (net inward Poynting flux)
  of a staircased-SIBC cylinder vs the same cylinder as a resolved volumetric conductor on a
  skin-depth-meshed grid. **Documented ~18% offset**: SIBC/resolved ratio ≈ **0.18**, and it
  is **grid-independent on both sides AND R/δ-independent** (0.172 at R/δ=6.7, 0.194 at
  R/δ=10.1) — i.e. the intrinsic **first-order-Leontovich-on-a-staircased-curve systematic**
  (on a FLAT surface the same boundary matches analytic Leontovich to <1%); a contributing
  half-cell low/high surface-node placement asymmetry is documented (convention, not a bug).
  Gate set at 25% so it fails closed on any regression toward PEC (PEC absorbs 1.8% of SIBC).
  Committed probes `docs/assessments/g4-sibc-oblique-probes/` reproduce the R/δ and 4-tier
  grid-independence numbers.
- **Wave-level skin-effect attenuation benchmark**
  (`benchmark/scenes/rf/lossy_waveguide_attenuation.py`, RESULTS row): a lossy-wall TE10
  guide's conductor attenuation `alpha` from a two-line `|S21|` ratio vs analytic
  `alpha_c` (Pozar 3.96) — median rel err **0.049%**, max **0.37%** (< 5% gate), passive
  (max singular value 0.363), matched (|S11| max 0.017); PEC-wall falsification collapses
  alpha to 0.00017 Np/m. The one authorized external-reference cloud run is recorded
  **honestly**: the external lossy-metal surface-impedance export **under-applies the wall
  loss** at the coarse export grid (forward-mode ~0.1–0.2 Np/m vs analytic ~1.7–2.2) — a
  documented **adapter-fidelity gap**, NOT the FDTD bench (matches analytic to 0.05%); the
  same adapter's phase-based beta cross-check agrees to ~1%.
- **Explicit committed zero-impact gate** (`test_sibc_zero_impact.py`): a SIBC-free scene's
  six raw last-step Yee fields are **bitwise identical** with the compile hook present vs
  monkeypatched out, with a load-bearing control (a real `LossyMetalMedium` box changes the
  fields when the hook is removed). Falsification recorded.

**Not delivered / open gaps:** **true oblique/conformal (non-staircase) SIBC** remains the
recorded gap, fail closed — a rotated `Box` (grid-unaligned face normal) is not staircased
and fails closed; the generic rational (`SurfaceImpedanceMedium`) ADE on a curved conductor
fails closed (only the narrowband good conductor is staircased). Bloch + SIBC,
adjoint/distributed/trainable SIBC, and generic surface-impedance adapter export remain
fail closed (external export under-applies wall loss, above). The ~18% staircase-curve
absorbed-power systematic awaits a second-order / curvature-corrected surface impedance.

**Evidence grade:** **E1** (all-orientation staircased SIBC: orientation-equivalence /
staircased-cylinder absorbed-power convergence / wave-level analytic-alpha bench / zero-impact
gated + falsified; the external cross-check is a documented adapter gap, not a passing
reference, and the ~18% curved-conductor systematic caps the grade below E2).
**Freeze:** Wave C — **unfrozen (S3 passed 2026-07-21)**; Round G orientation generalization
landed (merge `c9b4bfc`; census unchanged at the SIBC funnel — no guard added/removed).

### Wave D (10 / 11 / 12 / 13) — owner-authorized selective start (2026-07-19)

On 2026-07-19 the owner lifted the S0.2/S7 freeze **for plans 10, 12, and 13
only** (an explicit selective start, not a general Wave D unfreeze) and dropped
plan 11 (bioheat); its plan file was deleted in commit `4c521ab`. Plans 10/12/13
were implemented in parallel worktrees and merged to master (`git log`:
`d0ca5e5` electrostatics, `315d4ee` ESD stress, `de053e6` breakdown, `549c6a0`
SAR, plus follow-up fixes `f05c4c2`/`6056c60`/`e819777`/`cfb5a32`/`841aecb`/
`b237946`/`1b89b4f`). Each delivery is E1–E2 evidence (analytic / golden /
convergence / falsified gates) with **no external reference-solver cross-check**,
so no phase is `completed` (the audit §4 `completed` bar — wave-level headline
gate + independent reference + non-author review — is unmet). **Round H
(2026-07-21)** deepened all three — 12 Phase-4 SPD tensor-eps + open-boundary
`truncation_estimate` (`4a0555d`), 10 incident power density + canonical phantom
benches + SAR RESULTS rows (`8ebaec0`), 13 circuit-driven ESD + `SmoothBreakdownRisk`
surrogate (`df8ef96`), audit-minor cleanup `6f3b0c8`; still E1–E2 / E2-class with no
external reference, so none is `completed`. Bioheat is not covered below (dropped).

#### 10 — SAR

Plan: `10-sar.md` (status `in-progress`, 2026-07-19). Acceptance
`docs/assessments/b10-sar-acceptance-2026-07-19.md`; tests `tests/sar/`
(60 passed per the audit-fix pass in that doc).

**Delivered (with evidence):** Phase 0 (mass-density `Material` channel +
`Scene.compile_mass_density`), Phase 1 (point SAR reusing 01's `PowerLossData`
W/m³ with a power-conserving Yee→cell colocation; analytic conduction gate,
volume-integral closure vs `PowerLossData.total`, float64 oracle parity), Phase 2
(1 g/10 g cubical-prefix-v1 mass averaging; integral-image vs brute-force parity,
golden 3×3×3 cube, strict-interior/min-fraction validity, grid convergence),
Phase 3 (`accepted_power`/`source` normalization, coherent/incoherent
multi-source combination), and a Phase-4 slice (`soft_peak` surrogate +
finite-difference gradient gates, float32-limited per the acceptance doc's
precision note). Falsifications recorded per phase (see acceptance doc).

**Round-H incident power density + canonical phantom family + RESULTS rows (merge
`8ebaec0`, acceptance `h3-sar-phantom-acceptance-2026-07-21.md`):**
- **`IncidentPowerDensityMonitor` + `Result.incident_power_density`** (the previously
  fail-closed incident power density): a `PlaneMonitor` subclass carrying the four tangential
  fields for the normal Poynting component, reusing the `FluxMonitor` machinery
  (`plane_normal_poynting` factored as the single source of truth so the monitor's integrated
  flux is identically the `FluxMonitor` integral); typed `IncidentPowerDensity` output with
  signed `normal_poynting`, `|S.n|`, plane-integrated flux, and an explicitly **non-certified**
  (`certified: False`) versioned `spatial-average-v1` moving-window average. Analytic gates:
  plane-wave `|S| = |E|²/(2·eta)` exact-class, flux == `_compute_plane_flux` == analytic, sign
  flips with `normal_direction`, spatial-average vs independent brute-force reference, and an
  end-to-end vacuum plane-wave FDTD tie to a co-located `FluxMonitor`.
- **Canonical phantom benchmark family** (`benchmark/scenes/sar/`, redistributable canonical
  geometry; published-class 900 MHz tissue values): `uniform_lossy_cube`, `layered_slab`
  (skin/fat/muscle), `one_gram_cube` (hand-computable exact 1 g window), `antenna_near_phantom`.
  **SAR RESULTS rows** via a self-contained `benchmark/sar_validation.py` harness
  (`python -m benchmark sar`), gate classes self-labelled from `gate-classification.md`:
  `layered_slab` **wave-level** (surface/volume power-conservation closure 16.7% at dx=4 mm,
  monotonically converging 0.200→0.167→0.125 over 5/4/3 mm), `one_gram_cube`
  **analytic-identity**, `uniform_lossy_cube` **analytic-identity** (self-consistency,
  supporting only), `antenna_near_phantom` **blocked**; every row
  `external_reference: analytic-only`. Falsifications: Poynting-factor, box-sum
  inclusion-exclusion, one-gram grid, power-conservation colocation split each redden the
  headline gate.
- **Honest reclassifications (cleanup `6f3b0c8`):** the `uniform_lossy_cube` volume/channel
  closure is reclassified from analytic-identity to a **tautology** (self-comparison of the
  same field), and the blocked `antenna_near_phantom` row records its wave-level target as a
  **target only, never an achieved gate class** (blocked-with-target-class convention).
- **`antenna_near_phantom` blocked (recorded design blocker):** the driven end-to-end SAR
  chain fails closed on a conductive (tissue) background — the thin-wire and lumped-port
  runtimes require a conductance-aware port update coefficient (`fdtd/ports.py`,
  `fdtd/wire.py`); a tissue phantom is conductive by construction, so this is fundamental to
  the antenna+phantom combination. The scene builds and preparing it raises
  `NotImplementedError`, pinned as a gate.

**Not delivered / open gaps:** the **IEEE/IEC-*certified* standard-phantom profiles +
external reference-solver cross-check** are still NOT run — the canonical phantom family
(Round H) supplies redistributable geometry + wave-level conservation closure, but the single
owner-authorized external run is assessed and **deferred** (the only naturally driven
cross-check target, `antenna_near_phantom`, is blocked upstream; the conservation closure
already provides the binding evidence). `input_power` normalization stays **fail-closed**
(removing the guard needs a total-injected-source-power diagnostic — a real capability, left
out per the "only if clean" rule; census stays). `IncidentPowerDensity` is now delivered (was
fail-closed). Peak 1 g SAR is documented **grid-sensitive** (not tightly convergent — a
source-normalized plane wave delivers grid-dependent incident power density; the conservation
closure is the convergent wave-level observable). VOP and multi-GPU SAR reduction (Phase 5 /
Phase-4 scale-out) OUT of scope, fail closed. Gradient path is float32-limited.

**Evidence grade:** **E1–E2** (analytic/golden/brute-force-parity/grid-convergence + the
Round-H wave-level surface/volume power-conservation closure on `layered_slab`; no external
reference — the certified-phantom + external cross-check caveat holds).

#### 12 — Electrostatic / capacitance

Plan: `12-electrostatics-capacitance.md` (status `in-progress`, 2026-07-19).
Acceptance `docs/assessments/a12-electrostatics-acceptance-2026-07-19.md`
(A1/A2/A3 stages + audit-fix pass); tests `tests/electrostatic/` (46 passed).
Reproduction script `docs/assessments/a12_electrostatics_metrics.py`.

**Delivered (with evidence):** Phase 0+1 (electrostatic API objects,
matrix-free FVM `-div(eps grad phi)` operator, float64 Jacobi-PCG, Dirichlet/
Neumann boundaries, `ElectrostaticResultData`; parallel-plate/concentric-sphere/
coax analytic gates, monotone grid convergence, Gauss/energy-identity
conservation), Phase 2 (floating conductors by exact linear superposition, pure-
Neumann gauge handling), Phase 3 (N-terminal Maxwell `CapacitanceData` matrix;
reciprocity/sign/row-sum/energy gates), and a Phase-5 differentiability slice
(implicit-function-theorem backward on the reduced SPD solve; six central-
difference gradient gates rel err `< 1e-4`). Falsifications recorded per stage.

**Round-H Phase-4 SPD tensor-eps + open-boundary truncation_estimate (merge `4a0555d`,
acceptance `h2-es-tensor-acceptance-2026-07-21.md`):**
- **SPD full 3×3 tensor permittivity in the FVM electrostatic operator** (H2a): a `Structure`
  material with `DiagonalTensor3(...)` / `Tensor3x3(...)` now compiles and solves through
  `Simulation.electrostatic(...)` / `.capacitance(...)`. The operator is `A = A_diag + A_cross`,
  where `A_cross` (the off-diagonals) is defined as the gradient of a discrete quadratic energy
  `W_cross(phi)`, so it is **symmetric by construction**; the compiler validates the
  relative-permittivity matrix as real/symmetric/positive-definite (asymmetric/indefinite →
  `ValueError`). Gates: dense/random operator **symmetry** (asymmetry < 1e-9, min symmetric
  eigenvalue > 0), **rotated-frame MMS 2nd-order convergence** (orders > 1.9), Gauss closure,
  **anisotropic capacitance reciprocity** (< 1e-6) and energy consistency, diagonal-reduction
  parity, isotropic byte-identical scalar path. Falsifications: detaching the cross-term
  symmetrization (asymmetry 8.4e-2, reciprocity 1.92e-5) and dropping the cross term (MMS order
  collapse to ~0.03) each redden the discriminating gate. **Wall-tangential MMS added in
  cleanup** (`6f3b0c8`): the sin² MMS is flat at the walls, so a `phi = prod sin(t+0.5)`
  manufactured solution with nonzero wall gradient now exercises the wall cross-flux — it shows
  the documented **1st-order** wall boundary layer (interior stays 2nd-order); dropping the
  cross term makes it diverge. The per-PCG-iteration autograd `_apply_cross` was replaced with a
  precomputed-stencil direct application (bit-equal, equality-gated).
- **`open` electrostatic boundary fail-close + `truncation_estimate` domain-extension API**
  (H2b): an `open` (infinite-domain) boundary kind now raises `NotImplementedError` on any face
  (no exact radiation condition on the scalar potential at a finite Cartesian face; a BEM open
  boundary is a later phase). The opt-in `TruncationEstimate(padding_cells=N)` /
  `TruncationReport` runs ONE additional enlarged grounded-box capacitance solve (interior cells
  fixed to ULP, ~1.1e-16 drift), isolating the pure boundary-truncation effect; reports
  base/enlarged/delta/`max_relative_delta` + a 1/L Richardson extrapolation to the
  infinite-domain limit. Two-axis domain-extension convergence study committed (C(L) monotone
  toward a stable Richardson `C_inf`; grid-Cauchy at fixed L). **Boundary-touching-structure
  confound fails closed** (cleanup `6f3b0c8`): if a dielectric structure reaches the base domain
  wall the enlarged solve would replace it with background there, confounding truncation with a
  medium change — `_structure_reaches_boundary` now raises `ValueError`. Falsifications:
  removing the `open` special-case (falls through to a `ValueError`), neutralizing the
  enlargement (delta ~0, Richardson nan), forcing the confound check false (confounded delta
  reported) each redden.
- **Differentiability disposition — fail-closed (decided):** Phase 4 owns the forward SPD
  tensor-eps operator only (gradients are Phase 5); a trainable input under a tensor dielectric
  fails closed on both the electrostatic and capacitance public paths rather than detaching an
  unverified gradient.

**Not delivered / open gaps:** Phase-4 **tensor eps + open-boundary `truncation_estimate`
DELIVERED (Round H)** — see above. Still open: an **exact open boundary** (boundary-element /
infinite-domain) is deferred (the domain-extension `truncation_estimate` is the
controlled-truncation substitute); the **trainable tensor-eps backward** (off-diagonal
cross-flux VJP, Phase 5) fails closed; the **wall cross-flux is 1st-order** for a field with a
strong tangential wall gradient (interior is 2nd-order); the Richardson extrapolation assumes a
1/L leading term (estimate, not a certified bound). Phase-5 multi-GPU — no electrostatic
distributed entrypoint (single `scene.device`), noted out-of-scope rather than a dead guard.
Phase 6 (touchscreen/packaging workflow) not started. Floating-conductor gradients and
trainable-terminal-voltage / `Material`-eps public differentiation fail closed (only
`ChargeDensity` is a public differentiable leaf). No external reference-solver cross-check.

**Evidence grade:** **E2-class for the delivered envelope** (Round H: SPD operator symmetry /
rotated-MMS 2nd-order convergence / anisotropic-capacitance reciprocity / energy-identity /
two-axis domain-extension convergence + `truncation_estimate`, all falsified; earlier
Phases 0–3 + diff slice analytic/convergence/conservation/energy/gradient) — capped below full
E2 only by the absence of an external reference-solver cross-check and the 1st-order wall
cross-flux.

#### 13 — ESD / dielectric breakdown

Plan: `13-esd-dielectric-breakdown.md` (status `in-progress`, 2026-07-19).
Acceptance `docs/assessments/c13-esd-stress-acceptance-2026-07-19.md` (Phases
1–2, stages C1–C3) and `docs/assessments/d13-breakdown-acceptance-2026-07-19.md`
(Phase 4, stages D1–D3). Tests: `tests/esd/` (27 passed, Phase-1 waveform +
injection) and `tests/breakdown/`, which after merge holds BOTH the Phase-2
non-feedback stress/rating suite (42 passed per the C2/C3 doc) and the Phase-4
dynamic-breakdown suite (25 passed per the D2/D3 doc). Reproduction probe
`docs/assessments/d13-breakdown-probes/report_numbers.py`.

**Delivered (with evidence):** Phase 0+1 (IEC 61000-4-2 two-term Heidler
`ESDWaveform`, charge-conserving resampling, ideal terminal-port current
injection; analytic charge/action vs `scipy.quad`, IEC rise-time/current-band
sanity, end-to-end causal-transient tracking), Phase 2 (non-feedback
`BreakdownMonitor` + `ComponentStressMonitor` stress/rating reduction;
exceedance/longest-run/occupancy/trapezoid-energy/Yee-colocation golden gates,
bitwise no-perturbation parity), and Phase 4 (deterministic field-duration/
latching dynamic dielectric breakdown: per-node state machine, dynamic-
conductivity ramp, typed event log, dedicated breakdown-dissipation channel;
golden trigger-step, dt-convergence, analytic energy-closure rel err `2.94e-08`,
below-threshold six-field bitwise parity, event-log determinism, structure-
overlap last-writer-wins, zero-cost-when-unused). Falsifications recorded per
stage.

**Round-H circuit-driven ESD + SmoothBreakdownRisk surrogate (merge `df8ef96`,
acceptance `h4-esd-circuit-acceptance-2026-07-21.md`):**
- **Circuit-driven ESD through the standard source-impedance network** (H4a, Phase 3): a new
  `ESDVoltageSource` assembles the standard 330 Ω / 150 pF generator `Circuit` (voltage source
  driven by `R_d·i_esd(t)` → discharge resistor → storage capacitor to ground, bound to the
  named scene port) with generator provenance stamped and surfaced via
  `Result.esd_generator(...)`; `_WaveformBase.to_circuit_waveform(...)` resamples any
  ESD/measured waveform onto a `PiecewiseLinearWaveform` MNA source table (charge/impulse
  converges to analytic). Framed as a circuit **approximation of the standard network**, NOT
  discharge-gun geometry or a certification target. Gates: (a) RC-load analytic cross-check via
  an **independent scipy `solve_ivp` integration** sharing no runtime code (port-voltage rel
  7.8e-4, gate 1e-2; fit rel 1.9e-5) with a storage-cap-perturbation falsification (rel 4.5e-2);
  (b) closed-box coupled global energy conservation (residual 1.34e-4, gate 1.5e-2) with a
  channel-imbalance falsification; (c) provenance ride-through; (d) circuit-driven end-to-end
  (electrostatic pre-bias + circuit-driven ESD through the `TerminalPort` strong MNA coupling +
  non-feedback stress, one FDTD run). **EM-load-bearing companion gate added in cleanup**
  (`6f3b0c8`): the standard 150 pF network so heavily shunts the ~0.13 pF field one-port that
  zeroing the EM one-port in gate (a)'s prediction changes nothing (0.89×), so gate (a) alone is
  EM-insensitive — a variant high-impedance network (1 pF / 2 kΩ) where the field one-port
  materially shifts the port voltage was added, with the true prediction beating the zeroed-EM
  prediction **~12.7×** (a fresh runtime falsification).
- **`SmoothBreakdownRisk` differentiable surrogate** (H4b, Phase 5;
  `witwin/maxwell/breakdown_risk.py`, typed SEPARATELY, **non-physical / non-regulatory**):
  `risk = reduce_cells(occupancy · sum_t sigmoid((|E|−Ecrit)/w) dt)` with sum/mean/softmax
  reductions and an optional non-invasive `soft_damage` map. Gates (CPU float64): analytic
  backward vs central difference < 1e-4 with opposite-sign source/material sensitivities and a
  `torch.nn.Parameter` reach; strict monotonicity in source amplitude; vanishing far below
  threshold; colocation reuse; typing / non-physical-tag provenance; fail-closed input
  validation. Falsifications: detaching the sigmoid margin (grad tests error) and flipping the
  margin sign (monotonicity/threshold tests) redden. The `simulation.py` hard-breakdown
  trainable-rejection guard is KEPT (only its hint updated to point at the surrogate).
- **Conductive-media breakdown-feedback port fail-closed (recorded design blocker):** the strong
  FDTD+MNA port coupling does not support conductive media (a `DielectricBreakdown` introduces
  post-breakdown conductivity), so the **dynamic conductive breakdown feedback cannot ride the
  circuit-driven port path** in the current runtime; gate (d) pairs the circuit-driven port with
  a lossless dielectric + non-feedback stress monitor, and the dynamic feedback stays on the
  ideal-current-injection path (unchanged companion test). A conductance-aware port update
  coefficient is the required future work; the fail-closed guard is pinned.

**Not delivered / open gaps:** **Phase 3 is now substantially delivered** — the electrostatic
pre-bias slice (commit `d180125`, `ElectrostaticInitialCondition.from_result(...)`) PLUS the
Round-H circuit-driven ESD through the standard 330 Ω / 150 pF source-impedance network (merge
`df8ef96`, above), so excitation is no longer additive-current-only and the `TerminalPort` MNA
coupling is now tested (was recorded untested). The **Phase-5 trainable smooth surrogate is
delivered** (`SmoothBreakdownRisk`, typed non-regulatory); Phase-5 **multi-GPU / scale-out**
remains fail-closed. The **dynamic conductive breakdown feedback through the circuit-driven
port** stays fail-closed (conductive-media port coefficient, above). **Phases 6–7**
(surface/random/thermal feedback, gun/system calibrated-standard workflow) remain **excluded /
not started**. Decision-6 closed-box global energy balance for the ideal path stays deferred
(analytic dissipation closure substituted, stated not silent — the Round-H gate (b) does close a
coupled global balance on the circuit-driven path). `ESDPortRecord.measured` is `None` for the
ideal-injection path. No external / calibrated failure-prediction cross-check.

**Evidence grade:** **E1–E2** (Round H adds the independent offline circuit cross-check
[port-voltage rel 7.8e-4, no shared runtime code], the EM-load-bearing companion gate [~12.7×],
and the coupled global energy conservation closure on the circuit-driven path, on top of the
earlier analytic waveform / golden state-machine / energy-closure / dt-convergence /
bitwise-parity gates; the `SmoothBreakdownRisk` surrogate is gradient/monotonicity gated) —
capped below full E2 by the absence of an external or calibrated failure-prediction reference.

#### 11 — Bioheat (DROPPED)

Dropped by owner 2026-07-19; the plan file `11-bioheat.md` was deleted in commit
`4c521ab`. No implementation, no evidence, no residual scope.

---

## Route status (S0–S7)

The convergence route (audit §3) is a **correctness route, not a new-feature route**.
Each step requires machine-readable artifact + independent reference + CI-tiered
regression before it is checked off.

- **S0 — Freeze and stop-the-bleed: DONE.** Taxonomy `docs/reference/gate-classification.md`
  (5 classes + `perf` labels, S0.3); overstated plans re-annotated to
  `reopened`/`reopened-for-evidence` (S0.1, 01/03/04/06/07); Wave C/D new-physics
  freeze in force (S0.2).
- **S1 — RF port wave-level validation: DONE; keystone landed round-E.**
  `coax_thru` was the first wave-level PASS; the round-E Yee-staggered transverse
  operator (`aa866ba`) un-BLOCKED **`rectangular_waveguide`** (now a committed
  wave-level PASS with an authorized external-reference cross-check, `ead70c0`), and
  the coax-line rebuilds lifted **`lumped_open_short_match`** (SOL discrimination)
  and **`series_parallel_rlc`** (in-line RLC tracking) to wave-level PASS. The FDTD
  **`antenna/half_wave_dipole`** benchmark passes the real NF2FF path. Still BLOCKED:
  microstrip/diff-pair (interior-PEC masking of the staggered operator, not TEM this
  time) and the patch broadside `TM010` (honest strict xfail). Reference
  `docs/reference/rf-wave-validation-2026-07-18.md`,
  `docs/assessments/round-e-integration-2026-07-20.md`,
  `e1-rf-mode-operator-acceptance-2026-07-19.md`, `e2-rf-scenes-acceptance-2026-07-19.md`.
- **S2 — Port hot-path performance: DONE.** Both §9.4 timing gates PASS at 27M
  cells under the variance-aware 95%-CI criterion (grid-conditional — see below).
  Artifact `docs/assessments/port-perf-s2b-measurement-2026-07-18.json`.
- **S3 — Network/co-sim/array E2 evidence: DONE (2026-07-21).** All three members
  landed with tracked artifacts:
  - **03** — round-E independent raw-sample S-cascade cross-check (`<1e-5`,
    code-path-independent) + multi-scenario passivity/conservation suite supply the E2
    evidence; gate (d) received the **compute-bound ruling** (grid-conditional PASS at
    ≥224³, further launch reduction will not move the crossover;
    `network-embedding-gate-d-remeasure-2026-07-20.json`,
    `e4a-network-cascade-acceptance-2026-07-19.md`).
  - **04** — Round-F F1 multi-scenario conservation suite + independent offline circuit
    cross-check (port-voltage rel err 1.16e-5), merge `07e8e99`,
    `f1-cosim-e2-acceptance-2026-07-21.md`.
  - **06** — Round-F F3 `scene_gradient_vjp` single-device + 2-GPU ensemble (bit-for-bit /
    bitwise parity), merge `7ec99c7`, `f3-array-vjp-acceptance-2026-07-21.md` +
    `f3-array-scene-vjp-acceptance-2026-07-21.md`.

  **Consequence:** the S0.2 rule "no Wave-C new-physics solver consumption starts until
  S3 passes" is now satisfied — the **Wave-C (S6) unfreeze condition is met**. **Round G
  launched 2026-07-21** (four tracks: 02 NCCL one-process-per-GPU adjoint; 07 lossy-wire
  recurrence; 08 ferrite general/non-axis-aligned bias; 09 SIBC oblique/curved orientation
  generalization). No plan phase is thereby `completed` (each Round-F/S3 member still lacks
  the audit §4 non-author-review + external-reference bar).
- **S4 — Multi-GPU convergence: DISTRIBUTED CPML ADJOINT LANDED (round-E).** The
  distributed CPML-trainable adjoint bridge shipped with psi-active 1-vs-2-GPU
  gradient parity (rel 5.94e-7, ~1.1e5× falsification) after fixing a pre-existing
  single-GPU CPML psi axis cross-wiring (`a2d2cb7`); forward monitor gather with a
  seam-ownership rule + defense-in-depth trainable guard landed
  (`2364533`/`f7e8e9a`, `docs/assessments/e3-distributed-adjoint-acceptance-2026-07-19.md`).
  Timing measured on the exclusive window (`multi-gpu-timing-2026-07-20.json`):
  ensemble 1.98–2.00×, joint-solve 128³ 0.544× / 192³ 1.726×. **Round-G NCCL
  reverse-halo adjoint TRANSPORT PRIMITIVES landed** (merge `42ac3f1`,
  `g1-nccl-adjoint-acceptance-2026-07-21.md`): the transposed reverse halos are gated by a
  bitwise 2-process discrete-transpose identity, and an opt-in per-rank step-rate
  instrument (zero-cost-off asserted) resolves the round-E "not-measurable-by-hooks"
  finding (numbers pending an exclusive window). **Round-H NCCL end-to-end reverse DRIVER
  + S5 tiled-plane seeds LANDED** (merge `acea86e`,
  `h1-nccl-driver-acceptance-2026-07-21.md`): the G1 7-step plan is executed — a
  trainable-density scene backpropagates over a one-process-per-GPU NCCL launch with per-rank
  point/plane objective+grad parity ~2e-7 (incl. psi-active) at honest 1e-4-class tolerances,
  green both on exclusive GPUs AND under a saturating co-tenant burner after a **cross-stream
  caching-allocator race** was found and fixed (commit `c233d8b`; honest tolerances restored,
  stressed-mode gate committed). **S4 is substantially closed** (driver landed). **Still
  open:** flux/mode/x-normal NCCL adjoint objectives + monitor gather beyond forward, CPML
  psi-carrying NCCL reverse beyond the delivered psi-active parity, coupled-runtime
  (circuit/network/wire) joint solve, and the NCCL driver timing (pending an exclusive
  window). The in-process `cuda_p2p` load hazard is **CLOSED** (I1, `1a579b3`/`625baca`; a
  *distinct* checkpoint-capture happens-before race, fixed with a committed stressed parity
  gate; `i1-p2p-race-acceptance-2026-07-21.md`).
- **S5 — Full benchmark convergence: GEOMETRY LEVER LANDED (Round-F F4); rest QUEUED.**
  The §1.7 geometry/subpixel systemic lever shipped: **edge-native per-Yee-component
  material sampling** (drops the node→edge arithmetic smear; the Kottke/arithmetic subpixel
  blend is now evaluated at each Yee edge/face directly) + the **conformal-PEC benchmark
  default** (`SubpixelSpec(pec="conformal")` in `benchmark/scenes/_common.py`), merge
  `431bd7f`. Pre-registered geometry-cluster gate PASSES against the **identical existing
  caches** (no new cloud runs): **median field_l2 0.2072 → 0.0836 (−59.6%), 11 scenes
  improved / 0 regressed / 5 flat**, worst per-scene Δ = +0.0000, every scene's field
  correlation improves or holds (lowest after 0.911). The 5 flat scenes are the
  axis-and-node-aligned grid/slab controls (edge-native = node-smear there, bit-identical).
  Committed artifacts `docs/assessments/f4-geometry-cluster-{before,after,delta}.json`;
  acceptance `docs/assessments/f4-subpixel-lever-acceptance-2026-07-21.md`. Two
  `tests/validation` gates were re-anchored (autogrid ratio 0.6→0.8 and Rayleigh
  `sigma_max_ratio` band) as **intended, quantified** edge-native consequences (both
  absolute errors drop; box-independence re-verified), and one dispersive tolerance was
  restructured into a tight deterministic bound (stdev 0 over 12 runs). **Caveat:** the
  before/after was scored under `WITWIN_BENCHMARK_TRUST_CACHE=1` — a new off-by-default,
  loudly-warning diagnostic hook in `benchmark/runner.py` — because the 2026-07-14 caches'
  stored keys no longer byte-match after the physics-neutral null-material-field strip
  (`ee3d631`) + export-contract-version stamps; the default `python -m benchmark`
  staleness guard is untouched, and the improvement is a delta against the *same* fixed
  reference for both material paths (`field_corr` is the reference-validity sentinel).
  **Still QUEUED:** §1.7 remaining over-target scenes, TFSF/diffraction R/T/A normalization,
  S1 RF scenes resident in RESULTS.
- **S6 — Wave C solver consumption: UNFROZEN (2026-07-21); Round G DELIVERED & merged.**
  S1/S2/S3 all pass (see S3 above), so the S0.2 freeze on Wave-C new-physics solver
  consumption is lifted. **Round G delivered the three Wave-C consumption tracks**
  (master `18bc42a`): **07** lossy-wire B2 recurrence + real `ohmic_loss` + B3 conductivity
  adjoint (`3884bb7`, lossy E0→**E1–E2**); **08** ferrite arbitrary-bias forward + mixed-bias
  support (`5dd100e`, E0→**E1**); **09** all-orientation staircased SIBC + cylinder physics
  gate + skin-effect bench (`c9b4bfc`, E0→**E1**) — plus the 02 NCCL reverse-halo adjoint
  transport primitives (`42ac3f1`, driver deferred). Each still carries fail-closed items
  (07 field-coupled `dI/dσ` + distributed lossy; 08 Bloch/multi-GPU/adjoint; 09
  oblique/conformal/rational-on-curved) and none is `completed` (audit §4 non-author-review +
  external-reference bar unmet — 09's external cross-check is a documented adapter gap, not a
  pass). **Round H** carries the follow-ons (see `14-program-continuation-2026-07-21.md`).
- **S7 — Wave D (10–13): OWNER-AUTHORIZED SELECTIVE START (2026-07-19).** The
  owner lifted the S0.2/S7 freeze for plans **10, 12, 13 only** and dropped plan
  11 (bioheat, plan file deleted in commit `4c521ab`). Delivered against master:
  plan 12 electrostatics Phases 0–3 + diff slice (`d0ca5e5`), plan 13 ESD Phases
  0–2 (`315d4ee`) + breakdown Phase 4 (`de053e6`), plan 10 SAR Phases 0–3 + P4
  slice (`549c6a0`), with follow-up census/audit fixes (`f05c4c2` etc.). SAR
  reuses 01's `PowerLossData` contract (no duplicate data model), per the S7
  precondition. **Round-H Wave-D deepening delivered & merged (2026-07-21):** 12
  Phase-4 SPD tensor-eps + open-boundary `truncation_estimate` (`4a0555d`); 10 incident
  power density monitor + canonical phantom family + SAR RESULTS rows (`8ebaec0`); 13
  circuit-driven ESD through the standard network + `SmoothBreakdownRisk` surrogate
  (`df8ef96`); audit-minor cleanup `6f3b0c8`. **Still open per plan** — 12: exact open
  boundary (BEM) + trainable tensor-eps backward (P5), multi-GPU (P5), touchscreen
  workflow (P6); 10: IEEE/IEC *certified* phantom profiles + external reference
  cross-check, VOP + multi-GPU (P5), `antenna_near_phantom` conductive-media port blocker;
  13: P3 substantially delivered (prebias + circuit ESD), conductive-media
  breakdown-feedback port, P5 multi-GPU, P6–7 calibration/standard workflow. All three
  remain E1–E2 / E2-class (no external reference), so none is `completed`.
  **Capability-guard census budget history** (`docs/reference/fdtd-capability-guard-census.md`,
  `CAPABILITY_GUARD_BUDGET` in `tests/api/public/test_guard_census.py`, now **176**):
  `144 → 153` (plan 12 electrostatics merge) `→ 168` (plan 13 Phase-4 breakdown
  merge, +15 guards) `→ 172` (plan 10 SAR merge reconciliation, +4 guards)
  `→ 176` (plan 13 Phase-3 pre-bias slice, commit `d180125`, +4 guards) `→ 175`
  (Round-F F3a `scene_gradient_vjp` lands, −1) `→ 175` (Round G, **net ±2**: 07
  lossy-wire **+2** — −1 finite-conductor compile deferral removed, +2 lossy
  reverse/checkpoint guards, +1 distributed lossy-forward reject; 08 ferrite **−2** —
  general-bias and mixed-bias-direction guards removed; 02 NCCL and 09 SIBC net 0,
  no guard added/removed) `→ 176` (Round H, **net +1**: 12 electrostatics H2b
  open-boundary reject **+1** [H2a net 0 — anisotropic-tensor reject removed −1,
  `_reject_trainable_tensor` added +1]; 02 NCCL driver, 10 SAR, 13 ESD net 0 — the
  `allow_adjoint` relaxations only re-condition existing `ValueError` fences, and the
  new features use `ValueError`/`TypeError` validation the census does not count). ESD
  Phases 1–2 added no capability guards (local `ValueError`/`TypeError` only).

**Freeze rules restated:** the S0.2 rule that no Wave C/D new-physics
implementation starts until S3 passes was first **selectively overridden by the owner
on 2026-07-19 for plans 10/12/13 only** (see S7 above), and is now **generally lifted for
Wave C (07/08/09) as of 2026-07-21** because **S3 has passed** (see route step S3) —
Round G carries those tracks. No plan phase may be marked `completed` without a `wave-level`
headline gate + independent reference + convergence report + RESULTS presence +
non-author review (audit §4). External-solver cross-references follow the
reference-solver policy: use the covered external reference solver where it covers
the capability (via the existing Tidy3D benchmark adapter under
`witwin/maxwell/adapters/` and `benchmark/cache/`), otherwise mark
`reference: future-xfdtd` and hold the line with analytic/independent references —
never downgrade a gate to self-certification for lack of an external run.

## Owner decision points

1. **External reference-solver generation — RESOLVED (Round-F F2c).** The adapter
   port/lumped source-mapping feature was funded and landed (`WavePort`→`ModeSource`+
   `ModeMonitor`, `LumpedPort`→`UniformCurrentSource` delta-gap filament; `TerminalPort`
   out of scope). All four previously-blocked scenes now export `sources ≥ 1` and were
   cloud-generated (`coax_thru`, `lumped_open_short_match`, `half_wave_dipole`, `patch`;
   0.025 FlexCredits each, task ids in `f2-rf-trio-acceptance-2026-07-21.md` §4;
   `benchmark/RESULTS.md` rows `generated`), alongside the round-E
   `rf/rectangular_waveguide`. Remaining follow-on (not an owner decision): wiring a
   per-scene numeric Maxwell-vs-reference comparison into each runner.
2. **§9.4 / gate (d) grid-conditional pass — now has compute-bound evidence
   (round-E).** The exclusive-window remeasure
   (`network-embedding-gate-d-remeasure-2026-07-20.json`) shows the composite matvec
   coupling cut launches 65% yet fixed per-step cost dropped only ~9% (median
   0.183 ms/step), so the connected step is **compute-bound** (implicit solve + port
   observers), not launch-bound — the ~224³ crossover is unchanged and further launch
   reduction will not move it. Owner must rule whether "PASS at representative
   production scale (≥224³)" satisfies §9.4 / gate (d), or whether a compute-bound
   small-grid fixed-cost reduction is required.
3. **S3 go/no-go — RESOLVED: S3 PASSED (2026-07-21).** S0/S1/S2 complete; S3 proceeded
   on the validated scene set and all three members landed with artifacts (03 round-E
   cross-check + gate-(d) compute-bound ruling, 04 Round-F F1 evidence, 06 Round-F F3 VJP —
   see route step S3). The microstrip/diff-pair production wiring also landed (F2, now
   `gap`-classified, no longer BLOCKED). **Consequence:** the Wave-C (S6) unfreeze condition
   is met and **Round G launched 2026-07-21** (02 NCCL adjoint, 07 lossy wire, 08 ferrite
   general bias, 09 SIBC orientation generalization). No S3 member is `completed` (audit §4
   non-author-review + external-reference bar still unmet).

## Caveats and flagged discrepancies

- **C-1 (commit anchor).** The task stated the `integration` branch was "currently
  at master `a96338a`". As found, `integration` HEAD was `8e58ec6`, **two commits
  behind** master — it lacked the S2b commits (`a5659b9`,
  `a96338a`) that carry the §9.4 timing-gate PASS evidence and
  `docs/assessments/port-perf-s2b-measurement-2026-07-18.json`. The branch was
  fast-forwarded to `a96338a` (clean fast-forward; `8e58ec6` is a direct ancestor)
  so the tree matches the stated anchor and contains the cited artifact.
- **C-2 (plan 03 gate (b) source conflict).** The plan revision record (2026-07-18)
  claims gate (b) single-vs-2-GPU port V/I **bitwise PASS**
  (`tests/fdtd/multi_gpu/test_network_owner.py` `split_net`), consistent with the
  program-supervisor anchor. The earlier memory snapshots
  (`network-embedding-progress-2026-07-16.md`, `network-embedding-implementation.md`,
  both dated 2026-07-16) record the same gate as **not passed / blocked** on plan
  01/02 distributed contracts. The later plan record supersedes the memory (the
  network-owner multi-GPU slice landed after the memory snapshot); recorded here per
  the objectivity rule.
- **C-3 (S1 outcome vs anchor).** The anchor described `rectangular_waveguide`
  staggered-branch `beta` as "~7.6% low"; the binding source
  (`rf-wave-validation-2026-07-18.md` §5) states "~10% low". Source followed.
- **C-4 (guard census total).** The census document's area table sums to 140 while
  its running prose tracks 144 after the gyromagnetic export guard (143→144). The
  144 measured total is the current budget; the 140 table is stale relative to the
  later reconciliations in the same document.
- **C-5 (plan corpus force-added to tracking, 2026-07-19).** `docs/` is gitignored,
  so the full plan corpus existed only as untracked files in the main checkout, not
  in this worktree — the first draft of this document therefore wrongly stated "No
  plan file under `next-functional-2026-07/`" for plans 05/08/09/10–13. On this date
  the previously-untracked plan files were copied into the worktree at the same path
  and `git add -f`'d so the whole corpus is tracked and the worktree trap is closed:
  `02-multi-gpu-execution.md`, `02-phase-7-8-blueprint-2026-07-16.md`,
  `03-distributed-networks-spec-2026-07-16.md`, `05-nonlinear-circuit-devices.md`,
  `08-ferrite-materials.md`, `09-surface-impedance-metal-roughness.md`, `10-sar.md`,
  `11-bioheat.md`, `12-electrostatics-capacitance.md`,
  `13-esd-dielectric-breakdown.md`, and `README.md`. The already-tracked
  01/02-ensemble/03-touchstone/04/06/07 files (which carry the S0.1 evidence-grade
  re-annotations) were left untouched. Plans 10–13 remain **not started (frozen)**:
  their newly-tracked files are `proposed` design documents, not progress. The
  S0.2/S3/S7 gating identifiers come from the audit, not from the 10–13 plan files
  themselves (which express an internal deferral gate).

---

## Capability boundary (stable-release reference, 2026-07-21)

> Added 2026-07-21 at master `625baca` (the cuda_p2p checkpoint-capture race fix is
> merged; that stable-release blocker is **CLOSED** — see the I1 update at the top of
> this document and `docs/assessments/i1-p2p-race-acceptance-2026-07-21.md`). This is
> the single authoritative statement of what this codebase supports today, at what
> evidence grade, and where every boundary lies. It is a **reference digest of the
> per-plan sections above**: those sections and their acceptance docs remain the source
> of truth, this section adds no new claim, and every row cites an artifact.
>
> **The machine-checked boundary.** The **176** fail-closed capability guards
> inventoried in `docs/reference/fdtd-capability-guard-census.md` and enforced by
> `tests/api/public/test_guard_census.py` (`CAPABILITY_GUARD_BUDGET = 176`) **ARE** the
> authoritative, machine-verified statement of where support ends: every
> `NotImplementedError` capability path is counted, and any new unlisted guard fails the
> census test. The prose tables below are a human-readable index over that guard set
> plus the graded evidence; where a "fail-closed remainder" is named, a census guard
> backs it. Last recorded full battery: **3076 passed / 16 expected-FDFD (user-deferred)
> / 3 xfailed / 1 xpassed** at `6f3b0c8` (round-H) — **superseded as a release figure,
> see correction 4 in the 2026-07-22 update below**; the I1 fix (`625baca`) added a
> stressed parity gate + falsification and left the census at 176
> (`i1-p2p-race-acceptance-2026-07-21.md`).
>
> **Vocabularies.** E0–E3 evidence grades (audit §0/§4); the five-class gate taxonomy
> (`analytic-identity` / `tautology` / `symmetric` / `postprocess-only` / `wave-level`)
> + the `perf` label family (`docs/reference/gate-classification.md`). "E2-class" here =
> a wave-level or convergence/conservation headline gate is present but there is **no
> external reference-solver cross-check**, so the audit §4 `completed` bar is unmet; **no
> plan phase is `completed`** anywhere in this program.
>
> ---
>
> **UPDATED 2026-07-22 at master `19b30a1`** (this section was written at `f9c6bef`,
> anchored on `625baca`). Audited against every commit landed since. Four corrections;
> the physics/capability tables in §a–§e are unchanged by the perf/cleanup line.
>
> 1. **I1 (cuda_p2p) — no change needed, re-confirmed.** The section was authored
>    *after* the I1 merge, so it already records the in-process `transport="cuda_p2p"`
>    distributed adjoint as **load-race-free** (§c, §d) and the hazard as CLOSED (§g
>    tail). Re-verified against `docs/assessments/i1-p2p-race-acceptance-2026-07-21.md`
>    (fix `1a579b3`, merged `625baca`; stressed standard/x-CPML 1-vs-2-GPU grad parity
>    2.175e-7 / 1.522e-7..2.283e-7 over six rounds under a saturating co-tenant, gate
>    1e-4; falsification returns 8.090e-2). No superseded statement to retract.
> 2. **CUDA-graph stepping became the public default** (`9650439`, phase A) and was
>    then bound and guarded by J1 (`494a836`, `55c90e4`, `19b30a1`). This is a
>    user-visible execution-model change with a device precondition and a concurrency
>    carve-out; recorded as new §f bullets. Source:
>    `docs/assessments/j1-perf-regression-fixes-2026-07-22.md` regressions 2 and 3.
> 3. **`ETA_0` was briefly derived, then restored** to the CODATA 2018 literal
>    (`9650439` → `365ac45`). Absolute power / impedance / far-field normalizations
>    moved by ~3e-12 relative for the commits in between; the shipped value is the
>    pre-phase-A one. Recorded as a new §f bullet. Source: same J1 doc, regression 1.
> 4. **Battery figure is stale.** The `3076 / 16 / 3 / 1` count above was measured at
>    `6f3b0c8` (round-H) and predates the whole perf/cleanup line
>    (`9650439`, `8a1e5bd`, `64df75f`, `2bd33bb`, `bf9c3aa`) plus the J1 fixes. It is
>    **not** the release number. The only post-J1 measurements on record are slices in
>    the J1 doc (`tests/fdtd/multi_gpu` 285 passed; a 1157-item sources/multi-GPU/
>    gradients/materials/monitors/constants/census slice at 3 failed / 1153 passed /
>    1 skipped / 2 xfailed / 1 xpassed, the three failures all FDFD-`nvmath`;
>    `tests/rf` 778 passed / 1 xfailed). A full battery must be re-run before release;
>    do not quote `3076` as the release count.
>
> **Unchanged and re-verified at `19b30a1`:** `CAPABILITY_GUARD_BUDGET = 176`
> (`tests/api/public/test_guard_census.py:303`) — neither the perf/cleanup line nor
> J1 added or removed a `NotImplementedError` capability guard, so the machine-checked
> boundary in §a–§d still holds exactly as written. The FDFD disposition (16 expected
> failures, `nvmath` absent, user-deferred, do NOT triage) is unchanged and was
> re-confirmed by the J1 verification run.

### a. Solver runtimes

| Runtime | Release status | Supported envelope | Evidence grade | Boundary / citation |
|---|---|---|---|---|
| **FDTD** | **Production** | Yee-grid time-domain, native CUDA kernels, CPML/stable-PML + Mur + Bloch/periodic, differentiable single-GPU adjoint, ports / monitors / sources | **E2-class core** (wave-level RF scene set; adjoint mechanism analytic-VJP vs autograd ~2e-13) | §01/§02 above; `docs/reference/rf-wave-validation-2026-07-18.md`; the guard census is the fail-closed inventory |
| **Electrostatics / capacitance** | **Delivered envelope (experimental)** | matrix-free FVM `-div(eps grad phi)`, float64 Jacobi-PCG, Dirichlet/Neumann, floating conductors, N-terminal Maxwell `CapacitanceData`, **SPD full 3×3 tensor-eps**, implicit-diff backward, `truncation_estimate` domain extension | **E2-class for the delivered envelope** (SPD symmetry / rotated-MMS 2nd-order / reciprocity / energy-identity / two-axis domain-extension; no external ref) | §12; `a12-electrostatics-acceptance-2026-07-19.md`, `h2-es-tensor-acceptance-2026-07-21.md` |
| **FDFD** | **Unsupported this release** | frequency-domain sparse solver present but **user-deferred** | — (not graded; do not triage) | 16 known FDFD test failures are **expected** (env-dependent; needs `nvmath` — do NOT install). MEMORY `maxwell-fdfd-deferred`; brief baseline |

### b. Physics / material families

Per family: supported envelope / evidence grade / fail-closed remainder (each remainder
census-backed). All grades are the measured per-plan grades above.

| Family | Supported envelope | Grade | Fail-closed remainder | Citation |
|---|---|---|---|---|
| Standard media | isotropic `eps`/`mu`/`sigma`, full FDTD forward + adjoint | E2-class | — | §01 |
| Dispersive | pole-based ADE forward | E1–E2 | composition with breakdown / ferrite / SIBC fails closed | `media.py`; breakdown census |
| Anisotropic | diagonal + **full-tensor SPD** (electrostatic operator) | E1–E2 | tensor-eps trainable backward; Bloch+ferrite; per-cell tensor in FDTD | §12; `h2-es-tensor-acceptance-2026-07-21.md` |
| Nonlinear (circuit devices) | Phase-0 Device+Newton contract + **N1 standalone transient only** | **E0** | FDTD field-path coupling, transient companion into Yee, adjoint, BJT/MOSFET all fail closed | §05; census "nonlinear-device" reconciliations |
| Modulated (time-modulation) | forward | E1 | breakdown/ferrite/Bloch composition fail closed | census breakdown/ferrite |
| Gyromagnetic ferrite | **arbitrary-bias + mixed-bias forward** (per-cell rotation of Cayley ADE) | **E1** | no adjoint / FDFD / multi-GPU / Bloch / `PerturbationMedium`-over-ferrite; identity (not 4-point Yee) collocation | §08; `g3-ferrite-bias-acceptance-2026-07-21.md` |
| SIBC (surface impedance / roughness) | **all-orientation staircased** good-conductor (cylinder/sphere, six + mixed orientations) | **E1** | **true oblique/conformal (non-staircase)**, rational-on-curved, rotated Box, Bloch, adjoint/distributed, adapter export; ~18% curved-conductor absorbed-power systematic | §09; `g4-sibc-oblique-acceptance-2026-07-21.md` |
| Lossy wire | PEC paths + **B2 lossy-current ADE recurrence** + real `ohmic_loss` + **B3 conductivity adjoint** | **PEC E2 / lossy E1–E2** | field-coupled `dI/dσ`, distributed lossy forward+reverse, closed-box field-energy closure; analytic-AC gate is 8% (fit-limited, not 2%) | §07; `g2-lossy-wire-acceptance-2026-07-21.md` |
| Breakdown (dielectric) | **deterministic field-duration / latching** dynamic conductivity + typed event log | **E1–E2, uncalibrated** | conductive-media feedback port; recovery/damage models; FDFD; hard-trainable; distributed | §13; `d13-breakdown-acceptance-2026-07-19.md`; census "dielectric breakdown" |
| SAR | point SAR + **1 g/10 g cubical-prefix-v1** mass averaging + **incident power density** + canonical phantom family | **E1–E2, non-certified** | **IEEE/IEC certified** phantom profiles + external ref; `input_power` normalization; VOP (P5); multi-GPU; `antenna_near_phantom` (conductive-media port) | §10; `h3-sar-phantom-acceptance-2026-07-21.md`; census SAR |
| ESD | **stress-only** waveform+injection + **circuit-driven** (standard 330 Ω / 150 pF network) + electrostatic pre-bias | **E1–E2** | conductive-media breakdown-feedback port; **phases 6–7** (surface/random/thermal, calibrated gun/system) excluded; multi-GPU | §13; `c13-esd-stress-acceptance-2026-07-19.md`, `h4-esd-circuit-acceptance-2026-07-21.md` |

### c. Differentiability matrix

| Backpropagates (supported) | Evidence / citation |
|---|---|
| Single-GPU FDTD adjoint — material density, fixed-stencil geometry, **CPML psi-active** | §01/§02; `e3-distributed-adjoint-acceptance-2026-07-19.md` (single-GPU psi fix `a2d2cb7`) |
| **Distributed CPML-trainable adjoint** (`transport="cuda_p2p"`) — psi-active 1-vs-2-GPU parity 5.94e-7, **load-race-free** (stressed gate) | §02; `e3-…`, `i1-p2p-race-acceptance-2026-07-21.md` |
| **NCCL per-rank end-to-end reverse driver** — point + separable **y/z-plane** objectives incl. psi-active, ~2e-7, **load-race-free** (stressed co-tenant gate) | §02; `h1-nccl-driver-acceptance-2026-07-21.md` |
| **Array scene-gradient VJP** — single-device bit-for-bit + 2-GPU ensemble **bitwise** | §06; `f3-array-vjp-…`, `f3-array-scene-vjp-acceptance-2026-07-21.md` |
| **Electrostatic implicit diff** — `d(energy)/d(eps)`, `dC_ij/d(eps)`, `d(energy)/d(free_charge)` for **scalar/diagonal** eps (`ChargeDensity` is the public differentiable leaf) | §12; `a12-…`, differentiability-slice census |
| Network / circuit **coefficient gradients** — residues + direct terms | §03/§04; `test_fdtd_network_adjoint.py`, `test_fdtd_circuit_adjoint.py` |
| **Wire conductivity adjoint** — closed-form `dRe(Z')/dσ` (deterministic dissipation channel) < 1e-6 vs FD | §07; `g2-lossy-wire-acceptance-2026-07-21.md` (B3) |
| `SmoothBreakdownRisk` differentiable **surrogate** (typed non-physical / non-regulatory) | §13; `h4-esd-circuit-acceptance-2026-07-21.md` (H4b) |

| Rejects — fail-closed (census-backed) | Citation |
|---|---|
| **Hard** dielectric breakdown trainable (non-differentiable trigger) | §13; census "dielectric breakdown" (`simulation.py`) |
| Explicit-delay network adjoint (segment-crossing reverse ring + IIR reverse) | §03; census network-embedding delay guards |
| Floating-conductor superposition gradients; **trainable tensor-eps backward** | §12; census electrostatics (`_reject_trainable_tensor`) |
| WavePort network embedding (no scalar `(V,I)` terminal contract — missing design contract) | §03; `test_network_block_contract.py` |
| Field-coupled wire `dI/dσ`; distributed lossy reverse | §07; census finite-conductor wire (`replay_wire_state`) |
| Ferrite adjoint; SIBC adjoint/trainable | §08/§09; census ferrite/SIBC |
| Trainable RationalModel `poles`/`proportional`; state-space `A/B/C/D` | §03; census network-embedding capability guards |
| **Flux / mode / x-normal / finite-plane** NCCL adjoint objectives (need seam-crossing tangential assembly) | §02; `require_distributed_adjoint_objective_support`, `h1-…` |
| Distributed density beyond CPML/stable-PML trainable; ports; non-Box density | §02; `_validate_static_capabilities` |

### d. Multi-GPU matrix

| Mode | Support | Measured (exclusive window) | Fail-closed / boundary | Citation |
|---|---|---|---|---|
| Ensemble (N independent Simulations) | **supported, bitwise vs serial** | **1.98–2.00×** at 96³/160³ (MAD < 0.4%) | — | `ensemble-speedup-2026-07-17.json`, `multi-gpu-timing-2026-07-20.json` |
| Joint-solve forward (one-process-per-GPU) | **supported** | **grid-conditional**: 0.544× @128³ (comm-bound) → 1.726× @192³ | payoff only ≥192³; ensemble+joint composition rejected | `multi-gpu-timing-2026-07-20.json` |
| Distributed adjoint | **cuda_p2p + NCCL driver**, incl. **S5 plane seeds** | point/plane objective+grad parity **~2e-7** (incl. psi-active), load-race-free | flux/mode/x-normal/finite-plane objectives fail closed | §02; `e3-…`, `h1-nccl-driver-acceptance-2026-07-21.md`, `i1-…` |
| Monitor gather | **forward path only** (seam-ownership rule) | double-count falsification recorded | collective per-monitor gather beyond forward not built | §02; `test_monitor_merge_ownership.py` |
| Coupled-runtime joint solve (circuit/network/wire) | **NOT supported** | — | fails closed at prepare (blueprint #13/#18 tail) | §02 gaps |

### e. Validation tiers

Which capabilities carry which grade of reference (gate-classification vocabulary):

| Tier | Capabilities (examples) | Citation |
|---|---|---|
| **External-reference cross-check** | `rf/rectangular_waveguide` (`beta` vs one authorized cloud run, median 1.21% / max 2.74% vs analytic TE10); F4 geometry cluster scored vs the identical existing caches (median field_l2 −59.6%) | `round-e-integration-2026-07-20.md`; `f4-subpixel-lever-acceptance-2026-07-21.md` |
| **Analytic / golden-only** (`wave-level` head, no external ref) | `coax_thru`; dipole real NF2FF; SAR `layered_slab` conservation closure; electrostatic MMS/analytic; ESD/breakdown energy closure; SIBC `alpha_c` (Pozar 3.96); network raw-sample S-cascade cross-check | per-plan §01/§03/§09/§10/§12/§13 |
| **Consistency-class** (annotated, not lifted) | circuit reactive `dU_circuit` (C/L companion storage); memoryless network power-balance | §04 (`f1-cosim-e2-…`), §03 (`test_network_conservation.py`) |

**Wave-level validated RF scene set:** `coax_thru`, `rectangular_waveguide`,
`lumped_open_short_match`, `series_parallel_rlc`, `half_wave_dipole`
(`docs/reference/rf-wave-validation-2026-07-18.md`,
`e2-rf-scenes-acceptance-2026-07-19.md`, `round-e-integration-2026-07-20.md`).

### f. Known numerical conventions a release consumer must know

- **Edge-native per-Yee-component material sampling (F4, merge `431bd7f`).** The subpixel
  (Kottke/arithmetic) blend is now evaluated at each Yee edge/face directly, dropping the
  old node→edge arithmetic smear. **Geometry-dependent field values MOVED** (median
  `field_l2` 0.2072 → 0.0836, −59.6% vs the *same* caches, 11 improved / 0 regressed / 5
  flat) — improved vs references, but any pre-F4 per-scene numeric baseline must be
  regenerated. `f4-subpixel-lever-acceptance-2026-07-21.md`,
  `f4-geometry-cluster-{before,after,delta}.json`.
  - *Scope clarification added 2026-07-22 (no behavior change; the shipped scope was
    under-described here).* Only the **FDTD update coefficients** switch to the edge
    fields — the diagonal background `eps`/`mu` and the static `sigma_e`/`sigma_m`. The
    node-centered model is still produced as the canonical representation for summaries,
    monitors, the mode solver, and the SAR / mass models, so a consumer reading
    `Result.material(...)` or a SAR mass model reads the node grid, not the edge grid.
    The edge-native path covers isotropic, axis-aligned diagonal-anisotropic and
    `PerturbationMedium` families (with conductivity / dispersion / nonlinearity /
    modulation layered on top); **full off-diagonal anisotropy, 2D sheets, and
    surface-impedance metals stay on the node→edge path** (unchanged capability scope,
    no census guard added or removed). The material VJP follows the forward, so
    geometry / region-density / diagonal-anisotropy gradients stay consistent by
    construction. Benchmark harness default is now `pec="conformal"` (dielectric scenes
    unaffected). Source: FEATURE_LIST `f4-subpixel-lever` block, merge `431bd7f`.
- **CUDA-graph stepping is the public default (added 2026-07-22).** `9650439` flipped
  `FDTDConfig.cuda_graph` / `Simulation.fdtd(..., cuda_graph=...)` to `True` (measured in
  that commit: +29% throughput at 96³, neutral at 288³, +8–14% peak memory), keeping the
  pre-existing graceful eager fallback; distributed (`parallel=...`) configs still force
  it off (`simulation.py` `FDTDConfig.__post_init__`). Two J1 corrections define the
  supported envelope:
  - **Device precondition.** `CudaGraphRunner` now takes a **mandatory** `device`, runs
    warmup/capture/synchronize inside `torch.cuda.device(device)`, and uses an explicit
    per-device capture stream. Before `494a836` a solver whose tensors lived on `cuda:1`
    while the calling thread had `cuda:0` current recorded an **empty** graph and the run
    silently stopped integrating after warmup (40-step vacuum dipole, peak |Ez|
    2.126512e+04 eager vs 7.333388e+04 graphed on `cuda:1`). A capture that records no
    work is now a hard `RuntimeError`, so the caller's fallback degrades to eager
    stepping instead of installing a no-op replay.
  - **Concurrency carve-out.** Graph capture is process-global and PyTorch captures in
    `cudaStreamCaptureModeGlobal`, so an open capture aborts any synchronizing call in
    any other thread. `execute_plan` therefore wraps any plan with `workers > 1` in a
    reference-counted `suspend_capture()` (`55c90e4`): **concurrent ensemble tasks
    (`run_many`, the ensemble network sweep, the array-gradient plan) step eagerly** and
    give up the small-grid graph speedup. Serial plans (`max_concurrency == 1`) keep the
    graph default. `docs/assessments/j1-perf-regression-fixes-2026-07-22.md` §Regression
    2/3 and §For the supervisor item 1.
- **`ETA_0` provenance (added 2026-07-22).** `9650439` centralized the vacuum constants
  into `witwin/maxwell/constants.py` and redefined `ETA_0 = MU_0 * C_0`; `365ac45`
  restored the CODATA 2018 recommended literal `376.730313668`. `MU_0` and `EPSILON_0`
  are themselves twelve-significant-digit literals, so both "internally consistent"
  derivations land at `376.7303136668…` — a **−3.043e-12** (resp. −3.000e-12) relative
  offset that propagates into every absolute normalization consuming `ETA_0`: soft
  `PlaneWave`/`GaussianBeam` unit-power scales, TEM and vector-mode impedances,
  far-field constants, and `array.py`'s default wave impedance. The repo convention is
  **CODATA recommended literals per constant, not derivation**, now pinned by
  `tests/core/constants/test_physical_constants.py`. Only the intermediate commits
  `9650439`…`bf9c3aa` carry the derived value; the shipped value equals the pre-phase-A
  one, so absolute results are continuous across the release boundary. Same J1 doc,
  §Regression 1.
- **Spectral DFT conventions.** Half-step (`dt/2`) H-field colocation in the running DFT
  (the H-DFT colocation fix); consumers reading complex field spectra must honour the
  Yee time stagger. MEMORY `maxwell-validation-audit-2026-07-17`.
- **Trapezoidal port interface.** Embedded network / circuit port coupling is unified on a
  trapezoidal `(V,I)` interface (pinned `rtol ≈ 2e-6`). §03; `test_fdtd_circuit_coupling.py`.
- **`TimeConfig.auto` + periodic nondeterminism caveat.** An auto time-config combined with
  periodic boundaries has a recorded nondeterminism trap — pin an explicit `TimeConfig` for
  reproducible periodic runs. MEMORY `maxwell-rounds-f-g-2026-07-21`.

### g. Residual open items (stable-release backlog)

Folded from the round-H seed list, **minus the now-closed cuda_p2p item**:

- **NCCL driver timing** — pending an exclusive measurement window (correctness-only on
  shared GPUs; opt-in per-rank step-rate instrument is delivered/unit-tested). §02; `g1-…`.
- **Coupled-runtime joint solve** (circuit/network/wire multi-GPU) — fail-closed. §02.
- **Microstrip / diff-pair absolute `eps_eff`** — resolution gap (~24% low at dx = 5 mm;
  the quasi-static engine itself converges to Hammerstad–Jensen). §01; `f2b-…`.
- **Patch antenna broadside `TM010` + `D ≥ 5 dBi`** — honest strict xfail (feed reactance +
  small finite ground). §01; `test_antenna_benchmark_e2e.py`.
- **True-oblique / conformal SIBC** + the ~18% curved-conductor absorbed-power systematic —
  fail-closed / awaiting a curvature-corrected surface impedance. §09; `g4-…`.
- **Port / lumped external per-scene caches** — the four scenes export and were cloud-generated,
  but a per-scene numeric Maxwell-vs-reference comparison is not yet wired into each runner.
  §01 owner decision point 1; `f2-rf-trio-acceptance-2026-07-21.md`.
- **Certified SAR** — IEEE/IEC certified phantom profiles + external reference cross-check
  deferred (`antenna_near_phantom` blocked upstream by the conductive-media port). §10; `h3-…`.
- **ESD phases 6–7** (surface/random/thermal feedback; calibrated gun/system standard workflow)
  + conductive-media breakdown-feedback port — excluded / fail-closed. §13; `h4-…`.
- **FDFD** — user-deferred; 16 expected test failures, env-dependent (needs `nvmath`, do NOT
  install). MEMORY `maxwell-fdfd-deferred`.
- **NCCL flux/mode/x-normal adjoint objectives; monitor gather beyond forward** — fail-closed.
  §02; `h1-…`.
- **Concurrent-ensemble graph throughput trade (added 2026-07-22)** — a plan running more
  than one task at a time steps eagerly, forfeiting the +29%-at-96³ graph gain measured in
  `9650439`. Recovering it needs concurrent capture from several threads, which PyTorch
  explicitly does not support; a capture mutex plus `capture_error_mode="thread_local"`
  was considered and rejected (that mode scopes only the error check, and the allocator
  interaction across concurrent captures is not a supported configuration).
  `j1-perf-regression-fixes-2026-07-22.md` §For the supervisor item 1.
- **Release battery not yet re-measured (added 2026-07-22)** — see correction 4 in the
  header note above: the `3076` figure predates the perf/cleanup line and J1. A full
  `python -m pytest tests` run and a `python -m benchmark` regeneration are the two
  outstanding release-evidence items.

**Recently CLOSED (was a stable-release blocker):** in-process `cuda_p2p` distributed-adjoint
cross-stream load hazard — a *distinct* checkpoint-capture happens-before race (not the NCCL
allocator-reuse class), fixed by cloning the checkpoint on the shard `compute_stream`, with a
committed stressed parity gate + falsification. Fix `1a579b3`, merged `625baca`, census
unchanged at 176. `docs/assessments/i1-p2p-race-acceptance-2026-07-21.md`.
