# Array workflow Phase 4 acceptance (gradients)

Status: weight-gradient gates accepted; scene-gradient scope sliced and fails
closed (OPEN) on branch `codex/array-phases-2-4`.
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

Phase 2/3 extend the differentiable surface:
- `test_array_codebook.py`: a rank-3 codebook weight tensor backpropagates
  through `max_hold("realized_gain").envelope.sum()` with finite real/imag grads.
- `test_array_mimo.py::test_mimo_metrics_stay_in_the_torch_autograd_graph`: a
  trainable embedded pattern backpropagates through
  `ecc[0,0,1] + mean_effective_gain.sum()` with finite real/imag grads, proving
  the MIMO metrics carry no NumPy detach.

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
  tests/rf/array/test_array_codebook.py::test_scene_gradient_through_basis_fails_closed \
  tests/rf/array/test_array_mimo.py::test_mimo_metrics_stay_in_the_torch_autograd_graph -q
# 3 passed
conda run -n maxwell --no-capture-output python -m pytest tests/api/public/test_guard_census.py -q
# 3 passed (budget 133)
```
