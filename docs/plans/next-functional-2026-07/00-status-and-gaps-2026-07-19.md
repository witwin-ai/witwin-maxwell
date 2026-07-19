# next-functional-2026-07 — Status and Gaps (plans 01–13)

> Date: 2026-07-19
> Commit anchor: `a96338a` (master; the `integration` worktree was fast-forwarded
> to this commit so it contains the S2b measurement it cites — see Caveats C-1).
> Governing audit: `docs/assessments/next-functional-audit-2026-07-18.md`.
> Binding vocabularies: E0–E3 evidence grades (audit §0/§4), the five-class gate
> taxonomy + `perf` labels (`docs/reference/gate-classification.md`), the S0–S7
> convergence route (audit §3).

## Purpose

Single objective status-and-gaps reference for the fourteen-plan
next-functional-2026-07 program (plans 01–13; plan 05 = nonlinear devices).
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
| 01 | RF engineering workflow | 0–5 declared; per-gate re-classified | E0–E1 overall; one scene (`coax_thru`) at wave-level PASS via S1; §9.4 perf gates PASS (`perf`) via S2 | Only `coax_thru` reaches wave-level; waveguide BLOCKED, microstrip/diff-pair BLOCKED, RLC gap, matched/open/short FAIL | reopened; S1/S2 landed (partial), S3 pending |
| 02 | Multi-GPU execution | ensemble correctness + joint-solve NCCL forward | forward E2 | timing/speedup mostly deferred; distributed adjoint (CPML bridge, monitor gather, coupled runtimes) open | forward landed; adjoint/timing = S4 (queued) |
| 03 | Touchstone network embedding | 0–4 (gate (d) grid-conditional) | E1 | gate (d) PASS only ≥~224³; no independent multi-scenario passivity/conservation; inherits 01 port-power risk | E2 upgrade gated on S1-pass + S3 |
| 04 | SPICE/MNA co-simulation | 0–4 (4/4 gates evidenced) | E1–E2 | no conservation/energy-residual + independent circuit-solver cross-check; strong-coupling ref = `future-xfdtd`; inherits 01 risk | E2 upgrade gated on S1-pass + S3 |
| 05 | Nonlinear circuit devices | Phase 0 + N1 standalone transient | E0 | FDTD coupling / transient companion / adjoint / benchmark not built; BJT/MOSFET fail-closed | Wave C — FROZEN until S3 (S0.2) |
| 06 | Array / Active-S / MIMO | 0–3 + Phase-4 weight gradients | E1–E2 | scene-gradient VJP fail-closed (needs 02 Phase 7 aggregation); inherits 01 port-power risk | reopened; scene-grad gated on 02 P7 + S3 |
| 07 | Thin-wire model | 0–3 + Phase-4 partial | PEC E2 / lossy E0 | lossy current recurrence, `ohmic_loss`, conductivity adjoint, distributed wire reverse all fail-closed | Wave C consumption = S6; series-Z layer landed |
| 08 | Gyromagnetic ferrite | Phase 0 + axis-aligned forward slice | E0 | FDFD, multi-GPU, adjoint, Bloch, arbitrary-bias all fail-closed | Wave C — FROZEN until S3 (S0.2) |
| 09 | Surface impedance / roughness | Phase 0 + Phase-1 runtime generalization | E0 | oblique/curved/Bloch fail-closed; adapter export fail-closed; no wave-level SIBC benchmark | Wave C — FROZEN until S3 (S0.2) |
| 10 | SAR | 0 / — | — (design only) | no implementation | Wave D — not started (frozen), S7 |
| 11 | Bioheat | 0 / — | — (design only) | no implementation | Wave D — not started (frozen), S7 |
| 12 | Electrostatic / capacitance | 0 / — | — (design only) | no implementation | Wave D — not started (frozen), S7 |
| 13 | ESD / dielectric breakdown | 0 / — | — (design only) | no implementation | Wave D — not started (frozen), S7 |

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

**Not delivered / open gaps:**
- **No RF scene other than `coax_thru` reaches wave-level pass.**
  `rf/rectangular_waveguide` = **BLOCKED** on the transverse mode-operator defect
  (centered branch decouples odd/even sublattices — TE10 `sin`-correlation caps
  at 0.51–0.59; staggered branch has asymmetric wall BCs, `beta` ~10% low). Pinned
  by 6 strict-xfail eigenvector tests (`tests/rf/wave_validation/test_te10_mode_selection.py`,
  5 dx-parametrized + 1 high-frequency, `strict=True`). Blocker: transverse
  mode-operator redesign.
- `rf/microstrip_two_port` and `rf/differential_pair` = **BLOCKED** — TEM
  categorically inapplicable to inhomogeneous (substrate+air) cross-sections; a
  contour-snap `ValueError` fires first, `WaveModeSpec('tem')` raises underneath.
  Blocker: hybrid full-vector mode solve.
- `rf/series_parallel_rlc` = open **gap** (parasitic-dominated; load-port peak
  does not track C); strict-xfail `tests/rf/wave_validation/test_rlc_resonance_wave_level.py`.
- `rf/lumped_open_short_match` = **FAIL** — feed decoupled from load (identical
  Γ for matched/open/short). Blocker: needs a propagating feed line terminated by
  the load.
- Antenna Phase 4 is `postprocess-only` (textbook patterns fed in; `Result.antenna`
  integration monkeypatches the far-field). No FDTD-driven antenna benchmark
  (`antenna/half_wave_dipole`, `antenna/patch` from §10 do not exist).
- Retired Phase-1/2 headline gates re-classified and stripped of exit-gate status:
  matched/open/short = `analytic-identity`; RLC resonance = `analytic-identity`;
  coax/microstrip reciprocity = `symmetric`; power imbalance = `tautology`
  (`rf-wave-validation-2026-07-18.md` §2).
- External reference-solver caches: `pending-generation`, deferred pending owner
  cost authorization (`rf-wave-validation-2026-07-18.md` §4; adapter-driven M3 not
  wired). `PlaneMonitors` silently dropped from WavePort/PortSweep Results (open
  item, blocks field-level falsification).

**Evidence grade:** audit-measured **E0–E1** overall. S1 lifts the single
`coax_thru` scene to a tracked `wave-level` PASS (E2 for that scene only); S2's
gates are `perf` (never a physical-usability grade). The remaining §10 scene
matrix stays E0–E1.

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

**Not delivered / open gaps:**
- Timing/speedup for most configs = **DEFERRED-pending-exclusive-window** (ensemble
  throughput/makespan/utilization; measurement hooks exist, no number faked).
- **Distributed adjoint** not delivered: public trainable+parallel bridge still
  rejected at prepare (`_validate_trainable_parallel_fdtd`); the S4 distributed
  CPML bridge + 2-GPU parity/FD gates are deferred (clean deferral with recorded
  next-steps). NCCL adjoint, monitor gather, and coupled-runtime joint solve are
  open (blueprint items #13/#18 frozen-adjacent).
- `_validate_static_capabilities` rejects Bloch/x-periodic/x-symmetry,
  MaterialRegion density, ports, closed-surface/diffraction/flux-time/non-point
  time monitors, SIBC, material monitors, split-face Ex, non-point/non-uniform
  sources — the distributed path is mutually exclusive with most RF/research
  features. `DistributedFDTD` lacks a trainable guard (defense-in-depth gap,
  self-reported); long-run float32 halo drift known; ensemble+joint-solve
  composition still rejected.

**Evidence grade:** forward **E2**; adjoint/timing E0–E1 (deferred).

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

**Not delivered / open gaps:**
- **Gate (d) grid-conditional**: 8-port/order-32 step overhead <10% holds only at
  ≥~224³ (`224³`=9.64% PASS; `64³`=138.10% FAIL). Blocker: near-grid-independent
  fixed ~0.20 ms/step port/network coupling cost (0.193–0.204 ms/step across the
  sweep). Artifacts `network-embedding-phase-4-performance.json`,
  `-performance-grid-sweep.json`.
- Multiport S-parameter <0.02 gate **not independently backed** (reference inside
  the fit's own model class; margin float64-round-trip dominated).
- No multi-scenario passivity/conservation + independent S-cascade cross-check.
  `WavePort` embedding unsupported; explicit-delay differentiable runs rejected
  (delay state not in adjoint checkpoint schema); distributed trainable + spatial
  multi-GPU composition fail-closed (needs 02 Phase 7 result-aggregation). Inherits
  01 port-power-convention risk (not wave-level validated). Entry gate for the E2
  upgrade is blocked on S1 passing, then S3.

**Evidence grade:** **E1** (re-annotated from claimed E3).

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

**Not delivered / open gaps:** cap below E3 = no multi-scenario
conservation/energy-residual checks and **no independent circuit-solver (offline)
cross-validation**; the end-to-end EM+circuit strong-coupling gate has no external
reference (tagged `reference: future-xfdtd`). gate (c) −64.5% is a matched baseline
(native SeriesRLC termination is itself slow), not "co-simulation is nearly free"
(+407% vs bare FDTD). Distributed adjoint rejected before allocation (forward-only
multi-GPU contract). Inherits 01 port-power risk. E2 upgrade gated on S1-pass + S3.

**Evidence grade:** **E1–E2** (re-annotated from claimed E3).

### 05 — Nonlinear circuit devices

No plan file under `next-functional-2026-07/` (design/§8 referenced in audit).
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
**Freeze:** Wave C — solver consumption FROZEN until S3 passes (audit S0.2).

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

**Not delivered / open gaps:** `ArrayBasisData.scene_gradient_vjp(...)` **fail-closed
(NotImplementedError)** — scene/material/geometry gradients do not propagate through
the basis (retained columns store detached embedded-pattern tensors); blocked on the
aggregated per-column adjoint, which depends on **02 Phase 7 distributed
result-aggregation** (Phase-4 exit gate). Task-level multi-GPU was **removed from
scope** by user decision (not delivered, not evidence). All EIRP/realized-gain
descend from 01's port-power chain (not wave-level validated).

**Evidence grade:** **E1–E2** (re-annotated from claimed E3).

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
**Freeze:** lossy-wire solver consumption is Wave C step S6 (unfrozen after S1–S3).

### 08 — Gyromagnetic ferrite

No plan file under `next-functional-2026-07/`. Audit §1.6 / §3 row 08. Reference
`docs/reference/ferrite-physics-contract.md`. Census
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
**Freeze:** Wave C — solver consumption FROZEN until S3 passes (audit S0.2).

### 09 — Surface impedance / metal roughness

No plan file under `next-functional-2026-07/`. Audit §1.6 / §3 row 09. References
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
**Freeze:** Wave C — solver consumption FROZEN until S3 passes (audit S0.2).

### 10 — SAR  •  11 — Bioheat  •  12 — Electrostatic / capacitance  •  13 — ESD / dielectric breakdown

No plan files under `next-functional-2026-07/`; audit §1 row "10–13 `proposed`"
(design only) and §3 route step S7. **Not started (frozen).** Zero implementation
exists; there is no delivered phase, artifact, or test to point to. Per audit S0.2
they may not begin until S3 passes, and per S7 only after an explicit product goal,
reusing 01's `PowerLossData` contract (no duplicate data model). Do not read the
audit's reference-mapping table (§3: 10/13 `future-xfdtd`, 11/12 🟡) as progress —
it is forward planning, not delivered capability.

**Evidence grade:** — (no measured grade; nothing to grade).

---

## Route status (S0–S7)

The convergence route (audit §3) is a **correctness route, not a new-feature route**.
Each step requires machine-readable artifact + independent reference + CI-tiered
regression before it is checked off.

- **S0 — Freeze and stop-the-bleed: DONE.** Taxonomy `docs/reference/gate-classification.md`
  (5 classes + `perf` labels, S0.3); overstated plans re-annotated to
  `reopened`/`reopened-for-evidence` (S0.1, 01/03/04/06/07); Wave C/D new-physics
  freeze in force (S0.2).
- **S1 — RF port wave-level validation: DONE (with honest partial outcome).**
  `coax_thru` is the first genuine wave-level PASS; waveguide BLOCKED (transverse
  operator), microstrip/diff-pair BLOCKED (TEM inapplicable), RLC gap, open/short/
  matched FAIL. Reference `docs/reference/rf-wave-validation-2026-07-18.md` +
  `docs/assessments/rf-wave-validation-2026-07-18/`.
- **S2 — Port hot-path performance: DONE.** Both §9.4 timing gates PASS at 27M
  cells under the variance-aware 95%-CI criterion (grid-conditional — see below).
  Artifact `docs/assessments/port-perf-s2b-measurement-2026-07-18.json`.
- **S3 — Network/co-sim/array E2 evidence: QUEUED.** Not started; **user
  authorization pending**. Enters only after S1 passes (03/04/06 inherit 01's
  port-power convention). Includes 03 gate (d) generalization, 04 conservation +
  independent circuit-solver cross-check, 06 `scene_gradient_vjp` (depends 02 P7).
- **S4 — Multi-GPU convergence: QUEUED.** Timing/speedup artifacts, distributed
  adjoint (CPML bridge de-risked by the S4 audit), guard-list disposition.
- **S5 — Full benchmark convergence: QUEUED.** Diagnose §1.7 over-target scenes,
  TFSF/diffraction R/T/A normalization, S1 RF scenes resident in RESULTS.
- **S6 — Wave C solver consumption: FROZEN until S1–S3 pass** (07 lossy wire, 08
  ferrite kernel, 09 SIBC runtime; each E0→E2 with independent reference).
- **S7 — Wave D (10–13): FROZEN.** Not started; selective start only after a product
  goal, reusing `PowerLossData`.

**Freeze rules restated:** no Wave C/D new-physics implementation may start until
S3 passes (S0.2). No plan phase may be marked `completed` without a `wave-level`
headline gate + independent reference + convergence report + RESULTS presence +
non-author review (audit §4). External-solver cross-references follow the
reference-solver policy: use the covered external reference solver where it covers
the capability (via the existing Tidy3D benchmark adapter under
`witwin/maxwell/adapters/` and `benchmark/cache/`), otherwise mark
`reference: future-xfdtd` and hold the line with analytic/independent references —
never downgrade a gate to self-证 for lack of an external run.

## Owner decision points

1. **External reference-solver generation cost authorization.** The RF S1 caches
   are deliberately left as `pending-generation` markers under `benchmark/cache/rf/`;
   the adapter-driven generation (M3) is not wired and no cloud run was executed —
   a cost decision recorded as an S1.1 re-scope question. Blocks the external
   cross-check line for 01/03/06/09 and RESULTS residence of the covered families.
2. **§9.4 grid-conditional pass interpretation.** The S2 timing gates PASS at 27M
   cells but the per-extra-port <2% target holds only at ≥~27M cells; small/medium
   grids show 66–290% per-step overhead (fixed per-step launch cost dominates a bare
   Yee step). Mirrors 03 gate (d) (fixed ~0.20 ms/step, PASS only ≥~224³). Owner
   must rule whether "PASS at representative production scale" satisfies §9.4 or
   whether a small-grid fixed-cost reduction is required.
3. **S3 go/no-go.** S0/S1/S2 are complete; S3 is queued pending explicit user
   authorization. Because 03/04/06 inherit 01's port-power convention and only
   `coax_thru` is wave-level, the owner should decide whether S3 proceeds now, or
   waits on additional S1 wave-level scenes (waveguide/microstrip) that are
   currently BLOCKED on the transverse mode-operator redesign.

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
