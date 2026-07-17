# Thin-Wire Phase 0 Acceptance

Date: 2026-07-15
Status: accepted
Scope: numerical derivation and independent torch reference

## Delivered

- A pre-registered `AcceptanceBudget` matching the roadmap gates.
- The oriented incidence, reciprocal `G/G^T`, current/charge recurrence, exact
  staggered energy, and strict stability bound.
- Kernel-matched per-unit coefficients and explicit segment/dual-node length
  assembly without a second exterior-energy proxy.
- A recorded comparison of the historical edge-radius, contour-path, and
  kernel-matched auxiliary-network candidates.
- A small torch-native axis-aligned network reference with continuity and energy
  diagnostics. It is not a production solver or a CPU fallback.

## Exit-Gate Evidence

- Fixed physical radius across three transverse grids spanning `10x`: the
  physical radius remains explicit and distinct from the kernel distance.
- BS1 x BS1 square-grid coupling distance matches the independent analytic
  geometric-mean oracle within the registered `rtol <= 1e-5`.
- Radius sweep over one decade: every physical radius produces a distinct
  impedance and the propagation speed remains `1 / sqrt(mu epsilon)`.
- The actual recurrence's fundamental phase velocity converges over 16, 64, and
  256 segments (with their corresponding time steps): relative errors are
  `4.831e-3`, `3.012e-4`, and `1.882e-5`.
- Radius autograd derivative: matches the analytic derivative.
- Sampling/deposition power identity: exact transpose test passes.
- An axis-aligned open segment exchanges energy with a sampled Yee electric
  degree without growth while satisfying endpoint continuity.
- Closed lossless ring: continuity and total charge hold every step; staggered
  energy relative span is below `1e-11` over 2,000 steps. Continuity is checked by
  reconstructing `B I` from the segment tail/head node lists, independently of the
  incidence matrix and the matmul that `step` itself uses. `continuity_residual`
  is an exact algebraic identity of the discretization and is recorded as such,
  not used as a falsifiable gate.
- The derived CFL boundary separates bounded and unstable recurrences.
- The same reference recurrence remains device-resident on the available CUDA
  device.

Command:

```powershell
$env:TORCH_EXTENSIONS_DIR='<worktree>/.cache/torch_extensions'
$env:CUDA_CACHE_PATH='<worktree>/.cache/cuda'
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest -q -p no:cacheprovider tests\fdtd\thin_wire\test_thin_wire_reference.py
```

Measured result at acceptance: 18 tests passed; lint passed.

The reference is consumed by the native implementation's acceptance gates, not
only by its own unit tests: see
`tests/fdtd/thin_wire/test_thin_wire_forward.py::test_native_wire_step_matches_the_torch_reference_oracle`
together with the CFL bound pair
`::test_native_wire_cfl_bound_is_tight_on_uniform_topology_and_never_exceeds_exact_limit`
and `::test_native_wire_cfl_bound_has_strict_gershgorin_slack_on_nonuniform_topology`,
which bind `AxisAlignedWireReference` and the frozen `AcceptanceBudget` to the
compiled CUDA path. The bound is proven tight (an equality with the exact
`2/omega_max` leapfrog limit) on a uniform wire and strictly conservative only
where non-uniform coefficients give Gershgorin real slack, rather than merely
"conservative against the reference spectrum".

Independent module and phase reviews found no remaining must-fix items after the
kernel-distance oracle, coefficient assembly, convergence, positivity, CFL scope,
and input-validation corrections.

## Boundary Of This Acceptance

Phase 0 freezes the numerical contract only. It does not claim a public wire
scene object, native GPU stepping, wire monitors, network adjoint, arbitrary
orientation, RF port binding, finite conductivity, or multi-device execution.
Those capabilities remain gated by Phases 1 through 4.
