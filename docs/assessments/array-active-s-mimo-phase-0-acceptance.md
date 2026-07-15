# Array workflow Phase 0 acceptance

Status: accepted for the analytical contract

Evidence level: E0 experimental

## Delivered

- Public torch-native `BeamWeights`, `EmbeddedElementPatternData`, `ArrayBasisData`,
  and beam result contracts.
- Exact `b = S @ a`, embedded-pattern combination, explicit active-reflection mask,
  complex-reference active impedance, absolute-power antenna metrics, and weight
  autograd without NumPy detachment.
- A shared `AntennaData` torch metric kernel used by both the existing antenna path and
  array combination; no second gain/EIRP normalization is maintained.
- `Scene.compile_array_monitors(...)` and internal
  `compile_array_basis_request(...)`, including deterministic port order, exact
  frequency/angular grids, phase-center provenance, and linear time-invariant guard.

## Exit-gate evidence

| Requirement | Evidence |
| --- | --- |
| Two-port analytical network | Independent direct tensor formulas match `a`, `b`, active reflection, and complex-reference active impedance. |
| Two embedded-element patterns | Independent direct sum matches complex `E_theta/E_phi` pointwise. |
| Port/frequency/angular broadcasting | `[N]`, exact `[F,N]`, and `[B,F,N]` covered; ambiguous or interpolated shapes rejected. |
| Error paths | Real weights, invalid frequency/angular shape, zero incident beam, missing/non-RF port or monitor, independent source, nonlinear/time-varying scene, convention/order/frequency mismatch, and invalid metadata are rejected. |
| Autograd/device/dtype | Complex weight gradcheck uses the frozen 2% tolerance; CUDA complex64 is compared with an independent CPU result at `rtol=2e-5`, `atol=1e-6`. |
| Dark/full-reflection behavior | Active network, far field, realized gain, and EIRP remain available; undefined directivity/gain quantities return explicit false masks and NaNs. |

## Commands

All Python commands use the `witwin2` environment and worktree-local CUDA/build caches.

```text
python -m pytest -q --basetemp .cache/pytest/phase0 tests/rf/array/test_array_contracts.py
python -m pytest -q --basetemp .cache/pytest/phase0-regression tests/rf/contracts/test_network_data_contract.py tests/rf/network/test_network_sweep.py tests/rf/antenna/test_antenna_data.py
```

The initial regression run reported two setup errors because pytest attempted to use an
inaccessible shared temporary directory; reruns use the worktree-local `--basetemp` and
do not alter the numerical result.

## Measured result

- Phase 0 targeted plus RF conventions, network sweep, antenna, gradient, public API,
  guard-census, and Scene regressions: `148 passed`.
- Ruff and working/cached diff checks: passed.
- Independent array/RF contract review: passed after all findings were closed.
- Independent Phase 0 exit-gate review: passed with no remaining blockers.

## Deliberate boundary

Phase 0 contains analytical combination only. Full FDTD basis extraction, persistence,
direct multi-source comparisons, codebook scheduling, MIMO metrics, and scene-adjoint
aggregation belong to later phases and are not claimed here.
