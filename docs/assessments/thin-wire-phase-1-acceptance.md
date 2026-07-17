# Thin-Wire Phase 1 Acceptance

Date: 2026-07-15
Status: accepted
Maturity: E1 experimental
Scope: axis-aligned single-device PEC forward

## Delivered

- Public `ThinWire`, `WireConductor`, `WireEnd`, `WireMonitor`, and `WireData`
  contracts in the existing `Scene -> Simulation -> Result` workflow.
- A deterministic compiler for straight and bent axis-aligned paths, open and
  named-PEC-grounded endpoints, local isotropic host material, compressed graph
  state, sparse Yee sampling, and its sorted exact-transpose deposition.
- Three native CUDA operators for EMF sampling, the energy-paired current/charge
  recurrence, and current deposition. They use the current stream, make no hot-
  path allocation or synchronization, and are included in stable ABI v3.
- Stagger-aware frequency-domain wire monitoring, typed result persistence, a
  conservative joint Maxwell/wire CFL adjustment, and solver statistics for
  state size and stability bounds.

The implementation is a subgrid auxiliary network. It does not voxelize a thin
cylinder and does not introduce a CPU FDTD fallback.

## Exit-Gate Evidence

- Straight and L-shaped wires propagate finite current through native CUDA.
- A center-driven half-wave dipole was run at 8, 10, and 12 segments with fixed
  physical radius and equal physical observation time. Every normalized current
  profile is within 15% RMS of the sinusoidal thin-dipole oracle, mirror symmetry
  is within 15%, and the 10-to-12 segment profile change is smaller than the
  8-to-10 change.
- Changing physical radius below one cell changes the forward current, while
  `L'` and `C'` retain the frozen logarithmic-radius law.
- Each CUDA operator matches an inline torch reimplementation of its own formula
  for sampling, recurrence, charge, and deposition
  (`tests/fdtd/cuda/test_cuda_wire_parity.py`); CUDA Graph replay is bit exact.
  That suite is opt-in behind `WITWIN_RUN_CUDA_EXTENSION_BUILD=1` (it skips by
  default, not because of missing hardware); re-verified on-host with the flag
  set: 9 passed.
- The assembled native recurrence is compared against the Phase 0 reference
  oracle itself, not only against per-kernel reimplementations. In float64,
  `test_native_wire_step_matches_the_torch_reference_oracle` steps the compiled
  network and `AxisAlignedWireReference` from the same state and holds current,
  charge, and the sampled electric degrees within
  `ACCEPTANCE_BUDGET.reference_rtol` (`1e-5`) over 64 steps.
- `G`/`G^T` adjointness is verified end-to-end through the production CUDA
  kernels by `test_wire_sample_and_deposit_kernels_are_energy_adjoint_on_shared_edges`:
  it runs the real `sampleWireEmf3D` and `depositWireCurrent3D` kernels on random
  field/current tensors and checks the energy-adjoint identity
  `<sample(E), I> == sum_e mass_e * E_e * (-dE_e) / dt` to float64 machine
  precision, on an oblique fixture where several fragments deposit onto a shared
  Yee edge (asserted, so the per-edge segmented reduction is exercised). The
  transpose check inside `_dense_wire_operators` is only a self-consistency check
  of the compiler's own bookkeeping: the deposition list is emitted as a
  permutation of the sampling weights, so `deposition == sampling.mT` holds for
  any weights and does not, on its own, establish reciprocity.
- The production Gershgorin wire CFL bound is checked against the oracle's exact
  eigenvalue limit `2 / omega_max` from identical float64 coefficients. On a
  uniform-coefficient axis-aligned wire the bound is TIGHT (the measured ratio is
  `1.0`, an equality, not a conservative margin): Gershgorin's disc theorem still
  lower-bounds `dt`, but the constant-sign coupling makes the maximal row sum
  coincide with the top eigenvalue
  (`test_native_wire_cfl_bound_is_tight_on_uniform_topology_and_never_exceeds_exact_limit`).
  Strongly non-uniform per-segment radius and length recover strict slack
  (measured ratio ~`0.936`,
  `test_native_wire_cfl_bound_has_strict_gershgorin_slack_on_nonuniform_topology`).
  The prepare-time bound is now accumulated in float64 so the reported limit does
  not depend on float32 rounding.
- The real stability gate is the dt-straddle instability test, verified for the
  native path (not only the reference): the same closed loop stepped through the
  production sample/update/deposit kernels stays bounded at `0.99x` the wire CFL
  limit (peak ~`5.9`) and diverges to a non-finite peak at `1.01x`
  (`test_native_wire_recurrence_diverges_when_dt_straddles_the_cfl_limit`).
- A 512-step lossless recurrence satisfies continuity and keeps staggered energy
  drift below the registered 1% budget.
- The low-frequency Courant-limit reproduction that previously grew from
  `3.2e-3` to `1.06e29` is bounded after combining the Maxwell and wire spectral
  upper bounds. The measured adjusted step was `6.0819e-11 s`, below the joint
  limit `6.1433e-11 s`.
- Static compiler checks cover path splitting, incidence, transpose ownership,
  physical self/inter-wire collision, narrow PEC intersections, unbound port
  overlap, PML/domain bounds, PEC boundary-face contact, local isotropy, cache
  hits/invalidation, finite logarithmic self terms, and device-resident radius
  autograd. Runtime composition guards also reject periodic/Bloch fields and
  overlapping surface-impedance conductor ownership until their owning phases.
- The consolidated Phase 1 target matrix passed. The default run skips the
  opt-in per-kernel CUDA parity suite (behind `WITWIN_RUN_CUDA_EXTENSION_BUILD=1`);
  those skips are opt-in, not hardware-dependent, and the suite passes when the
  flag is set. `git diff --check` passed; `ruff` is not installed in this env.

## Performance And Memory Evidence

- No-wire 16^3 interleaved benchmark, 9 samples of 8,000 field steps: exact
  branch-free baseline median `0.598395 s`; current median `0.588348 s`; measured
  regression `-1.679%`, satisfying the registered `<1%` regression budget.
- Native wire sequence at 100 segments: `61.814 ms` per 100,000 segment-steps;
  at 1,000 segments: `9.059 ms` per 100,000 segment-steps. The launch-amortized
  path used sorted reduction and no atomics.
- Runtime state was exactly `1,204 bytes` for 100 segments and `12,004 bytes` for
  1,000 segments, matching `(3S + 1) * sizeof(float)` for current, EMF, and node
  charge. Peak incremental allocation during warmed stepping was zero bytes.

## Independent Review

Independent public API, compiler, native CUDA, and runtime reviews all reached
GO after their findings were corrected. Corrections included FDTD-only and
trainable guards, arbitrary-Mapping persistence, quantity-selective monitoring,
host-material coefficients, PEC/port/proximity validation, static caching,
native topology validation, source-segment-aware self-collision, local-isotropy
enforcement, PEC-face/SIBC ownership checks, robust finite self terms, periodic
composition guards, and the joint stability bound.

## Boundary Of This Acceptance

- Phase 1 supports single-device forward only. Any trainable scene containing a
  wire is rejected before adjoint dispatch until Phase 2 adds wire state
  checkpoint/replay and the exact reverse recurrence.
- Junctions, branches, and closed loops remain Phase 2.
- Arbitrary direction, periodic paths, and RF port binding remain Phase 3.
- Finite conductivity, broadband skin effect, ohmic loss production, and
  distributed wire ownership/reverse communication remain Phase 4. Multi-GPU
  wire scenes are rejected before hardware initialization in this phase.
