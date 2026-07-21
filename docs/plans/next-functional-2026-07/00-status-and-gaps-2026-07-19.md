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
| 02 | Multi-GPU execution | ensemble + joint-solve forward + S4 distributed CPML-trainable adjoint (psi-active) | forward **E2**; CPML-trainable distributed adjoint **E2** (psi-active parity 5.94e-7); NCCL-adjoint/coupled E0 | NCCL one-process-per-GPU adjoint + monitor gather; coupled-runtime joint solve; joint-solve strong scaling grid-conditional (128³ 0.544×, 192³ 1.726×); static-capability exclusions remain | S4 CPML-adjoint landed; NCCL-adjoint/coupled-runtime open |
| 03 | Touchstone network embedding | 0–4 + round-E E2 evidence (gate (d) grid-conditional) | **E2** (embedded path: independent raw-sample S-cascade cross-check <1e-5 + multi-scenario passivity/conservation) | gate (d) still grid-conditional (PASS ≥224³, compute-bound ruling); delay adjoint fail-closed; WavePort embedding = missing design contract; inherits 01 port-power (now partly wave-validated) | E2 evidence landed; gate (d) grid-conditional; delay-adjoint/WavePort open |
| 04 | SPICE/MNA co-simulation | 0–4 + F1 conservation suite + independent offline cross-check | **E2** (multi-scenario conservation + code-path-independent circuit cross-check landed) | strong-coupling end-to-end ref still `future-xfdtd`; reactive `dU_circuit` channel stays consistency-class (not lifted by F1b); external lumped-load cross-check pending | E2 evidence landed (F1); S3 closed |
| 05 | Nonlinear circuit devices | Phase 0 + N1 standalone transient | E0 | FDTD coupling / transient companion / adjoint / benchmark not built; BJT/MOSFET fail-closed | Wave C — S3 passed 2026-07-21, unfrozen; not scheduled in Round G |
| 06 | Array / Active-S / MIMO | 0–3 + Phase-4 weight gradients + F3 scene-gradient VJP (single-device + 2-GPU ensemble) | **E2** (VJP bit-for-bit vs single-device path; 1-vs-2-GPU bitwise parity; central-difference gated) | throughput/scaling deferred-pending-exclusive-window; network/S-param scene gradients + NCCL joint-solve adjoint out of scope; inherits 01 port-power risk | reopened; Phase-4 exit gate closed (F3); S3 closed |
| 07 | Thin-wire model | 0–3 + Phase-4 partial | PEC E2 / lossy E0 | lossy current recurrence, `ohmic_loss`, conductivity adjoint, distributed wire reverse all fail-closed | Wave C unfrozen (S3 passed); Round G lossy-wire track; series-Z layer landed |
| 08 | Gyromagnetic ferrite | Phase 0 + axis-aligned forward slice | E0 | FDFD, multi-GPU, adjoint, Bloch, arbitrary-bias all fail-closed | Wave C — unfrozen (S3 passed); Round G general-bias track |
| 09 | Surface impedance / roughness | Phase 0 + Phase-1 runtime generalization | E0 | oblique/curved/Bloch fail-closed; adapter export fail-closed; no wave-level SIBC benchmark | Wave C — unfrozen (S3 passed); Round G orientation-generalization track |
| 10 | SAR | 0–3 + P4 slice / 0–5 | E1 (analytic/golden/brute-force-parity/grid-convergence; no external ref) | IEEE/IEC phantom benchmark, incident power density, VOP (P5), multi-GPU scale-out (P4) all deferred/fail-closed | Wave D — owner-authorized selective start 2026-07-19; S7 partial |
| 11 | Bioheat | — (dropped by owner) | — | plan dropped by owner 2026-07-19; plan file deleted (commit `4c521ab`) | Wave D — DROPPED (not implemented) |
| 12 | Electrostatic / capacitance | 0–3 + P5 diff slice / 0–6 | E1–E2 (analytic/convergence/conservation/energy-identity/gradient; no external ref) | tensor eps + open boundary (P4), multi-GPU (P5), touchscreen/packaging workflow (P6) fail-closed/deferred | Wave D — owner-authorized selective start 2026-07-19; S7 partial |
| 13 | ESD / dielectric breakdown | 0–2 + P3 pre-bias slice + P4 / 0–7 | E1 (analytic waveform/golden state-machine/energy-closure/dt-convergence; no external/calibrated ref) | P3 circuit-ESD co-sim, P5 multi-GPU/smooth surrogate, P6–7 calibration/standard workflow deferred | Wave D — owner-authorized selective start 2026-07-19; S7 partial |

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

**Not delivered / open gaps:**
- **NCCL one-process-per-GPU (torchrun) adjoint, monitor gather, and coupled-runtime
  (circuit/network/wire) joint solve remain OUT of scope, fail-closed** — the
  blueprint #13/#18 tail. The NCCL forward path has only a correctness worker
  (`_nccl_forward_worker.py`); its step rate is **not-measurable via existing hooks**
  and is recorded as such (not fabricated).
- Joint-solve forward strong scaling is grid-conditional (payoff only ≥192³; 128³ is
  communication-bound). Ensemble+joint-solve composition still rejected.
- `_validate_static_capabilities` still rejects Bloch/x-periodic/x-symmetry,
  MaterialRegion density (except CPML/stable-PML trainable), ports,
  closed-surface/diffraction/flux-time/non-point time monitors, SIBC, material
  monitors, split-face Ex, non-point/non-uniform sources — the distributed path is
  mutually exclusive with most RF/research features. Long-run float32 halo drift
  known.

**Evidence grade:** forward **E2**; **CPML-trainable distributed adjoint E2**
(psi-active 1-vs-2-GPU parity 5.94e-7, load-bearing falsified); timing measured on
the exclusive window (ensemble 2×, joint-solve grid-conditional). NCCL adjoint /
coupled-runtime joint solve **E0** (deferred).

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

**Not delivered / open gaps:** finite-conductor pieces **all fail closed** — the
lossy current recurrence, the `ohmic_loss` monitor (currently emits zeros),
conductivity adjoint, and distributed wire **reverse** reject with explicit errors;
trainable wire under multi-GPU, distributed CPML/Mur, and wire-with-circuit/network
mixing also fail closed (a finite conductor compiled into an FDTD run raises
`NotImplementedError`, `compiler/thin_wire.py`). The series-impedance layer is
**compile-time only — the solver does not yet consume lossy current physics**.
Blockers: energy-consistent lossy recurrence + CUDA `update_wire_state` extension
(B2), conductivity adjoint (B3, depends B2), distributed wire reverse (B4, depends
02 Phase 7 wire-state channel; also needs a second local GPU). Some Phase-3
impedance/far-field numbers flagged UNEVIDENCED (regenerate before citing).

**Evidence grade:** **PEC E2 / lossy E0**.
**Freeze:** lossy-wire solver consumption is Wave C step S6, now **unfrozen (S1–S3 all
passed 2026-07-21)**; it is a Round G track (B2 lossy recurrence → B3 conductivity adjoint).

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

**Not delivered / open gaps:** general/non-axis-aligned bias, mixed-bias scenes,
Bloch-periodic ferrite, FDFD ingest, multi-GPU (ShardEngine), adjoint (no reverse
core for the non-reciprocal correction) all **fail closed**. `PerturbationMedium`
over a ferrite rejected. Scalar frequency-evaluation is a permanent contract.

**Evidence grade:** **E0** (contract + narrow axis-aligned forward slice).
**Freeze:** Wave C — **unfrozen (S3 passed 2026-07-21)**; general/non-axis-aligned bias
is a Round G track.

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

**Not delivered / open gaps:** oblique/rotated/curved and Bloch cases still funnel
through the single `_reject_surface_impedance` `NotImplementedError`; the generic
`SurfaceImpedanceMedium` adapter export mapping fails closed (no external construct).
No wave-level SIBC/roughness benchmark in RESULTS. The documented 1D stability
narrative is sign-mirrored/setup-dependent — treat the 3D empirical characterization
as ground truth.

**Evidence grade:** **E0** (contract + runtime generalization; stability fix is a
correctness/robustness landing, not a wave-level validation grade).
**Freeze:** Wave C — **unfrozen (S3 passed 2026-07-21)**; oblique/curved orientation
generalization is a Round G track.

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
gate + independent reference + non-author review — is unmet). Bioheat is not
covered below (dropped).

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

**Not delivered / open gaps:** IEEE/IEC 62704-1 standard-phantom benchmark and
independent-reference cross-check NOT run (no redistributable phantom fixture;
golden + brute-force parity + grid convergence stand in). `IncidentPowerDensity`
monitor and `input_power` normalization fail closed (no injected-source-power
diagnostic in this build). VOP and multi-GPU SAR reduction (Phase 5 / Phase-4
scale-out) OUT of scope, fail closed. Gradient path is float32-limited.

**Evidence grade:** **E1** (analytic/golden/parity/convergence; no external ref).

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

**Not delivered / open gaps:** Phase 4 (nonuniform grid, tensor/anisotropic eps,
open boundary) rejected at prepare / out of scope. Phase-5 multi-GPU — no
electrostatic distributed entrypoint (single `scene.device`), noted out-of-scope
rather than a dead guard. Phase 6 (touchscreen/packaging workflow) not started.
Floating-conductor gradients and trainable-terminal-voltage / `Material`-eps
public differentiation fail closed / require caller-supplied compiled tensors
(only `ChargeDensity` is a public differentiable leaf). No external
reference-solver cross-check.

**Evidence grade:** **E1–E2** (analytic/convergence/conservation/energy/gradient;
no external ref).

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

**Not delivered / open gaps:** Phase 3 is PARTIALLY delivered post-snapshot:
the electrostatic pre-bias slice landed in commit `d180125`
(`ElectrostaticInitialCondition.from_result(...)` node-interpolated exact
discrete-gradient seed — Yee curl ~1e-15, PEC-plate interior drift 0.0 over 300
no-source steps — plus Gauss-residual gate and a prebias+waveform+stress+
breakdown cross-feature e2e; `docs/assessments/wave-d-integration-acceptance-2026-07-19.md`).
System circuit-ESD co-simulation remains NOT delivered — injection uses the
additive current-source path, not a source-impedance/discharge-gun network;
MNA/SPICE coupling through `TerminalPort` is recorded untested. Phase 5
(multi-GPU, scale-out, trainable smooth surrogate) fail-closed. Phases 6–7
(surface/random/thermal feedback, gun/system calibrated-standard workflow) not
started. Decision-6 closed-box global energy balance explicitly deferred
(framework exposes no injected-source-energy / running-stored-EM-energy monitor;
analytic dissipation closure substituted, stated not silent). `ESDPortRecord.
measured` is `None` for the ideal-injection path. No external / calibrated
failure-prediction cross-check.

**Evidence grade:** **E1** (analytic waveform/golden state-machine/energy-closure/
dt-convergence/bitwise parity; no external or calibrated reference).

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
  ensemble 1.98–2.00×, joint-solve 128³ 0.544× / 192³ 1.726×, NCCL step-rate
  not-measurable-by-hooks. **Still open:** NCCL one-process-per-GPU adjoint + monitor
  gather, coupled-runtime (circuit/network/wire) joint solve, guard-list disposition.
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
- **S6 — Wave C solver consumption: UNFROZEN (2026-07-21); Round G in flight.**
  S1/S2/S3 all pass (see S3 above), so the S0.2 freeze on Wave-C new-physics solver
  consumption is lifted. **Round G** carries the Wave-C tracks — 07 lossy-wire recurrence
  (B2→B3), 08 ferrite general/non-axis-aligned bias, 09 SIBC oblique/curved orientation
  generalization — plus the 02 NCCL one-process-per-GPU adjoint tail (see
  `14-program-continuation-2026-07-21.md`). Each still targets E0→E2 with an independent
  reference before any `completed` mark.
- **S7 — Wave D (10–13): OWNER-AUTHORIZED SELECTIVE START (2026-07-19).** The
  owner lifted the S0.2/S7 freeze for plans **10, 12, 13 only** and dropped plan
  11 (bioheat, plan file deleted in commit `4c521ab`). Delivered against master:
  plan 12 electrostatics Phases 0–3 + diff slice (`d0ca5e5`), plan 13 ESD Phases
  0–2 (`315d4ee`) + breakdown Phase 4 (`de053e6`), plan 10 SAR Phases 0–3 + P4
  slice (`549c6a0`), with follow-up census/audit fixes (`f05c4c2` etc.). SAR
  reuses 01's `PowerLossData` contract (no duplicate data model), per the S7
  precondition. **Still open per plan** — 12: tensor eps / open boundary (P4),
  multi-GPU (P5), touchscreen workflow (P6); 10: IEEE/IEC phantom benchmark,
  incident power density, VOP + multi-GPU (P5); 13: P3 circuit-ESD co-simulation (pre-bias slice delivered, `d180125`),
  P5 multi-GPU/smooth surrogate, P6–7 calibration/standard workflow. All three
  are E1–E2 (no external reference), so none is `completed`.
  **Capability-guard census budget history** (`docs/reference/fdtd-capability-guard-census.md`,
  `CAPABILITY_GUARD_BUDGET` in `tests/api/public/test_guard_census.py`, now 176):
  `144 → 153` (plan 12 electrostatics merge) `→ 168` (plan 13 Phase-4 breakdown
  merge, +15 guards) `→ 172` (plan 10 SAR merge reconciliation, +4 guards)
  `→ 176` (plan 13 Phase-3 pre-bias slice, commit `d180125`, +4 guards); ESD
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
