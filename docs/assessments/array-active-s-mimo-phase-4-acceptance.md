# Array workflow Phase 4 acceptance (gradients)

Status: reopened-for-evidence (2026-07-18 audit; see "Measured evidence grade"
section at end).

Original status (archived): weight-gradient gates accepted; scene-gradient scope
sliced and fails closed (OPEN) on branch `codex/array-phases-2-4`.
Date: 2026-07-17
Scope: one device.

## Delivered contract

- Complex incident power-wave weight gradients through `ArrayBasisData.combine`
  are fully supported with zero solver reruns, including through the batched
  codebook path, `BeamData.max_hold` (subgradient value; non-differentiable
  winning-beam index), and the MIMO metric kernels (torch-native, no NumPy
  detach).
- Scene / material / geometry gradients through the retained-column basis fail
  closed via `ArrayBasisData.scene_gradient_vjp(...)`, which raises
  `NotImplementedError` naming the missing aggregated per-column adjoint envelope.

## Weight-gradient gate

The existing high-precision gate is retained and unchanged:
`tests/rf/array/test_array_contracts.py::test_complex_weight_gradient_matches_high_precision_gradcheck`
asserts `torch.autograd.gradcheck` on `combine(weights).realized_gain.square().mean()`
(complex128, eps 1e-6, atol 1e-5, rtol 0.02) — i.e. autograd agrees with a
finite-difference reference on an independent numerical path.

Phase 2/3 extend the differentiable surface with their own in-suite gates:
- `test_array_codebook.py::test_codebook_weight_gradient_backprops_through_max_hold`:
  rank-3 `[B, F, N]` codebook weights with `requires_grad` backpropagate through
  the batched `combine` and `max_hold("realized_gain").envelope.sum()`, asserting
  finite and nonzero real/imag gradients. Falsification recorded: detaching the
  normalized weights inside `combine` severs the graph so `backward` raises,
  reddening the test.
- `test_array_codebook.py::test_codebook_weight_gradient_matches_high_precision_gradcheck`:
  a `torch.autograd.gradcheck` leg on the same codebook -> `max_hold` envelope
  path (complex128, eps 1e-6, atol 1e-5, rtol 0.02), so the batched codebook
  gradient is gradcheck-gated, not only the rank-1 combine path.
- `test_array_mimo.py::test_mimo_metrics_stay_in_the_torch_autograd_graph`: a
  trainable embedded pattern backpropagates through
  `ecc[0,0,1] + mean_effective_gain.sum()` with finite real/imag grads, proving
  the MIMO metrics carry no NumPy detach.
- `test_array_mimo.py::test_polarization_correlation_cross_term_matches_brute_force_integral`:
  the `rho != 0` dual-polarized cross term in `mimo()` is validated against an
  independent brute-force trapezoidal integral (previously exercised only at the
  default `rho = 0`).

## Scene-gradient slice (OPEN, fails closed)

Per the plan-of-record amendment and the mission's explicit allowance, the
scene-parameter adjoint through the N-column basis is sliced out of this landing.
The retained-column basis stores detached embedded-pattern tensors, so a
scene-parameter VJP cannot be produced from it. The public method
`ArrayBasisData.scene_gradient_vjp(...)` fails closed rather than returning silent
`None` gradients, and is reconciled as one new capability guard:

- census budget 132 -> 133 in `tests/api/public/test_guard_census.py`;
- disposition in `docs/reference/fdtd-capability-guard-census.md`
  (Array scene-gradient reconciliation, 2026-07-17), counted under "Public
  simulation, result, and network workflows" (23 -> 24).

This becomes a public promise only after the aggregated per-column adjoint lands
(plan 06 Phase 4 exit gate, gated on the plan 02 Phase 7 distributed
result-aggregation contract). The single-GPU material/geometry basis-adjoint and
the frozen 96^3 / 4096-step qualification run remain deferred to an exclusive
window.

## Commands

```bash
conda run -n maxwell --no-capture-output python -m pytest \
  tests/rf/array/test_array_contracts.py::test_complex_weight_gradient_matches_high_precision_gradcheck \
  tests/rf/array/test_array_codebook.py::test_codebook_weight_gradient_backprops_through_max_hold \
  tests/rf/array/test_array_codebook.py::test_codebook_weight_gradient_matches_high_precision_gradcheck \
  tests/rf/array/test_array_codebook.py::test_scene_gradient_through_basis_fails_closed \
  tests/rf/array/test_array_mimo.py::test_mimo_metrics_stay_in_the_torch_autograd_graph \
  tests/rf/array/test_array_mimo.py::test_polarization_correlation_cross_term_matches_brute_force_integral -q
# 6 passed
conda run -n maxwell --no-capture-output python -m pytest tests/rf/array -q
# 65 passed
conda run -n maxwell --no-capture-output python -m pytest tests/api/public/test_guard_census.py -q
# 3 passed (budget 133)
```

## Measured evidence grade (2026-07-18 audit rollback)

Appended per `docs/assessments/next-functional-audit-2026-07-18.md` §1.4 and §4
(no-inflation rule). Everything above is retained verbatim; this section records
the **measured grade, outstanding debt, and one post-audit positive update**.
Where it conflicts with the "accepted" claim, this section's grade governs.

- **Measured grade: E1–E2** (not the claimed E3). Phase 0-3 plus the Phase 4
  weight gradients hold: codebook / max-hold / MIMO / ECC live inside autograd
  with passing gradcheck; MIMO vs independent Clarke closed form and a positive
  definite PSD `Q_rad` (4-layer PML) are credible E2-grade evidence.
- **Scene / material / geometry gradient fails closed.**
  `ArrayBasisData.scene_gradient_vjp(...)` raises `NotImplementedError` (census
  132 -> 133, no silent `None`); it becomes a public promise only after the
  aggregated per-column adjoint lands, gated on plan 02 Phase 7 distributed
  result-aggregation.
- **Inherits the 01 port-power chain risk.** All EIRP / realized-gain metrics
  descend from `01`'s accepted/available-power convention, which is not yet
  wave-level validated (audit §1.1).
- **Post-audit positive update (2026-07-18).** The frozen `96^3 / 4096-step`
  qualification, previously deferred-pending-exclusive-window, was executed on
  this host (2x RTX A6000, exclusive window) and **PASSED**. Physical power
  closure `|P_accepted - P_rad| / P_incident = 6.971e-4 (0.0697%)`, well inside
  the 1% frozen gate; the timing contract passed (basis+16-combine is 20.55% of
  16 direct solves, gate <= 40%; one combine is 0.0343% of one direct solve,
  gate < 10%, zero extra FDTD steps); `local_hardware` re-anchored to this host.
  **Artifact: `docs/assessments/array-active-s-mimo-phase-1-qualification.json`**
  (`verdict = PASS`, corrected flux/observer convention at commit `1cc4a71`).
  The audit's data predates this run, so it is not reflected in the audit body.
- **Evidence required to reach E2/E3 (convergence route, audit S3.3):**
  1. complete `scene_gradient_vjp` aggregated per-column adjoint, close the
     fail-closed guard (gated on plan 02 Phase 7);
  2. analytic array-factor + external-reference-backend (🟡, needs a capability
     landing confirmation, else tag `future-xfdtd`) multi-scenario comparison
     entering RESULTS;
  3. inherit the port-power chain only after 01 completes S1 wave-level
     validation (lifts to E3 combination matrix / public benchmark, README §7).
- Entry gate: this plan's S3.3 convergence work is blocked on S1 (01 port
  wave-level validation) passing first.
