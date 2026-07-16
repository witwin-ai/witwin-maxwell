# Network embedding implementation record

Status: active
Plan: `next-functional-2026-07/03-touchstone-network-embedding.md`
Branch: `codex/network-embedding`

## Dependency audit

- The RF `NetworkData` schema and power-wave convention required from plan 01 are present at the implementation baseline.
- `LumpedPort` and `TerminalPort` lower to the same compiled sparse Yee geometry and are available for single-device FDTD coupling.
- `WavePort` does not yet expose a time-domain terminal injection operator, so embedded-network connections must reject it until that separate contract exists.
- Spatial FDTD currently rejects RF ports. The required public port reference-point ownership field and a hot-path scalar reduce/scatter transport are not present. Phase 4 spatial multi-device ownership remains blocked on those plan 01/02 contracts; the implementation must not guess ownership from sparse edge order.

## Phase 0 evidence

Implemented:

- Strict Touchstone 1.x/2.0 parsing with line-numbered errors.
- RI, MA, and DB pairs; Hz, kHz, MHz, and GHz frequencies.
- S, Z, and Y input with the version 1 normalization rules.
- Full, Lower, and Upper matrices, both explicit 2-port orders, and per-port version 2 reference impedances.
- Preserved source format metadata, comments, parser warnings, port order, and reference impedances.
- N-port S/Z/Y writing and `NetworkData.from_touchstone(...)` interoperability.

Acceptance evidence:

- Static one-, two-, and four-port corpus fixtures.
- Float64 RI/MA/DB import/export error below `1e-10`.
- Independent 2-port ordering assertions.
- Bad-token, incomplete-data, NaN, frequency-order, impedance, and suffix-mismatch diagnostics assert source line numbers.
- Targeted parser/writer and `NetworkData` contract suite: 63 passed before the final name-boundary regression.
- Full RF network plus `NetworkData` contract regression: 102 passed after all review fixes.
- Static analysis of all changed Python modules and tests: passed.

Independent review findings resolved before acceptance:

- Named port comments now survive export/import.
- Touchstone 1.x rows with more than four pairs wrap conformingly.
- DC samples are accepted while negative frequencies remain rejected.
- Version-specific data-line layout, strict ASCII, and canonical keyword syntax are validated.
- Exact zero DB output uses a finite floor instead of a non-standard infinity token.
- Duplicate resolved port names report the second offending comment line.
- Public network port names reject leading/trailing whitespace so named-port text round-trips are lossless.

## Phase 1 evidence

Implemented:

- Direct generalized-Kurokawa S/Y conversion for complex per-frequency, per-port reference impedances without explicit matrix inversion.
- Shared-pole real rational fitting for scalar and multiport S/Y/Z responses under the repository `exp(-i*omega*t)` convention.
- Stable conjugate pole/residue validation, auditable fit reports, and versioned rational-model persistence.
- Sampled raw-data S passivity plus pole-aware rational S/Y/Z in-band interval verification, and explicit direct-term enforcement with a reported relative change and configurable acceptance threshold.
- Real continuous state-space realization with `Nports * order` states and bilinear/trapezoidal discretization through `torch.linalg.solve`.
- `NetworkData.validate_physicality(...)` and `NetworkData.fit_rational(...)` public interoperability.

Acceptance evidence:

- Analytic RC, RLC, and matched delayed-transmission responses meet the `< 1e-3` in-band max complex-error gate.
- Real state-space and rational frequency responses agree to float64 numerical tolerance; bilinear matrices agree with SciPy's independent oracle.
- Every accepted discrete model is gated at `max |z| < 1-1e-7`; a driven passive corpus obeys the discrete supply-energy inequality while an active-sign control violates it.
- Sparse-sample regression contains an arbitrarily narrow active resonance between the caller's samples; pole locations plus rational interval bounds detect and certify the violation.
- Uniform-DC sweeps distinguish delayed from advanced responses through a negative-time-energy diagnostic; nonuniform/non-DC sweeps report causality as indeterminate.
- Pre-fitted residue/direct gradients propagate through evaluation, while automatic fitting and enforcement reject trainable inputs without silent detach.
- A real CUDA run keeps pre-fitted evaluation gradients and continuous/discrete state-space tensors device-resident; rational-model save/load round-trips coefficients and response.

Known limits recorded rather than hidden:

- Raw `NetworkData` passivity remains sampled. Rational models use a finite-band pole-aware interval certificate, not an all-frequency Hamiltonian certificate. Enforcement is an explicit isotropic direct-term shift, not a residue-minimal projection.
- Raw-data causality is a finite-band heuristic and is only evaluated for a uniform sweep starting at DC. A proper stable fitted realization is structurally causal.
- Pure broadband delay extraction and bounded delay buffers belong to Phase 3; Phase 1 only validates a controlled narrow-band delayed transmission fit.

## Phase 2 evidence

Implemented:

- `NetworkBlock` and `TouchstoneNetwork` declarations with complete one-based/name mapping to unique Scene ports, explicit automatic-fit versus pre-fitted contracts, and no silent detach.
- Scene/compiler lowering to a stable, interval-certified passive admittance state-space descriptor connected to the existing sparse `LumpedPort` / resolved `TerminalPort` Yee geometry; unsupported `WavePort`, non-FDTD solvers, and out-of-fit-band result requests or excitation spectra fail before stepping. Phase 2 acceptance scoped this path to one port.
- Fixed-shape, device-resident same-step feedback solving `(I + D_d Z_f) i = C_d x + D_d u`, followed by midpoint voltage correction and `x_next = A_d x + B_d v` without a one-step delay.
- A post-source network update block with reusable hot-path buffers and independent CUDA Graph capture, inserted before dispersion, PEC/Mur, DFT, and port observation.
- Tensor-native `EmbeddedNetworkData` diagnostics exposed through `Result.embedded_network(...)`, including network-oriented V/I, absorbed/generated frequency power, state norm, typed fit report, model identity, warnings, and runtime provenance.
- Explicit Phase 4 guards for trainable embedded-network coefficients, FDTD adjoint replay, and embedded-network Result persistence.

Acceptance evidence:

- The algebraic one-port direct-loop test matches an independent same-step formula and detects a one-step-delay implementation.
- An analytic passive series RLC is written to `.s1p`, read through `TouchstoneNetwork`, automatically fitted to a second-order rational model, and run against native `SeriesRLC` on the same CUDA Yee scene with identical source/grid; V, I, and derived reflection coefficient agree well inside the `< 1%` magnitude / `< 2 deg` phase gate.
- The same physical run verifies signed absorbed power `P_abs >= -1e-5 P_inc`, network current orientation, model identity, and Result statistics.
- CUDA profiler coverage spans the real network update plus port observer and asserts no per-step `aten::item`, `_local_scalar_dense`, allocation, or host/device memcpy; pointer-reuse coverage verifies fixed hot-path state, phase tables, and scratch storage.
- Eager and independently captured network updates agree on field and state after repeated steps.
- A CUDA regression protects the bilinear transition matrix from the eigensolver's Schur workspace by computing stability poles from a clone.
- A stable narrow-resonance active model that looks passive at its input samples is rejected by the state-space resolvent interval certificate; a zero-state pure direct conductance compiles and discretizes without dummy poles.

Known limits recorded rather than hidden:

- Embedded-network coefficients are fixed for forward execution. Their analytic reverse and persistence schema belong to Phase 4.
- Spatial multi-device ownership remains blocked on the missing public port reference-point ownership and hot-path scalar transport contracts recorded in the dependency audit.

## Phase 3 evidence

Implemented:

- Ordered 2/4/N-port compilation and runtime mapping, with all matrices, diagnostics, and Scene connections normalized to `NetworkData.port_names`.
- Coupled same-step direct feedthrough using a prepare-time pivoted-LU factorization of `I + D_d diag(Z_f)` and fixed solve scratch; no explicit inverse, hot-path allocation, or one-step feedback delay is introduced.
- Optional per-port `delay_seconds=None | "auto" | tuple[...]` plus a strict `max_delay_steps` storage bound. Automatic extraction solves weighted path-delay equations deterministically in CPU float64 and reports rank, equation count, residual, warnings, and the resolved one-way delays.
- Delay de-embedding/re-embedding follows the repository `exp(-i*omega*t)` convention, so a causal path contributes `exp(+i omega (tau_i + tau_j))`. The de-embedded passive S core is rationally fitted/certified, while the time-domain reference planes use fixed bidirectional integer rings plus first-order Thiran fractional delay.
- Mixed zero-delay and delayed ports share one power-wave runtime: the zero-delay subset receives its own prepare-time same-step direct-loop solve, and delayed waves are read before and written after the current terminal solution. All buffers, indices, cursor state, and fractional-filter state are CUDA Graph captured without step-local allocation or host synchronization.
- Real, positive, frequency-independent reference impedances are required for the time-domain delay adapter; shorter-than-one-step explicit delays and over-budget buffers fail at compile time instead of being silently approximated.

Acceptance evidence:

- Independent NumPy matrix oracles cover non-diagonal direct terms for 2- and 4-port networks; a four-port permutation produces identical physical fields, currents, and state after repeated updates.
- Static 2/4-port network-file fixtures compile through automatic rational fitting and reproduce their independent scattering references with maximum complex error below `0.02`.
- A real four-port CUDA FDTD run reproduces its fitted admittance current within `2%`, preserves signed total power, and activates the network CUDA Graph. Per-port signed powers are summed before classifying network-total absorbed/generated power, so a passive transmitting network does not mislabel delivered output power as generation.
- Explicit and automatic delay compilation re-embed a known two-port network below `0.02`; automatic extraction recovers `(0.2 ns, 0.3 ns)` and persists the typed delay report.
- A measured 40.5-step fractional delay has unit magnitude to `1e-3` and phase error below `3 deg`; complete reflection paths are gated on their two-way phase, not only one-way port phase. A compiled delayed `NetworkBlock` driven through the prepared FDTD terminal runtime also matches its full frequency-domain response within `2%` and `3 deg` after steady-state DFT.
- Real CUDA profiler coverage finds no hot-path allocation, scalar synchronization, or host/device copy in either the coupled N-port update or bidirectional delay. Eager/Graph state and field updates agree.
- Phase 3 targeted plus related RF network/port regression: 149 passed. Independent review findings on explicit inversion, per-port power clamping, and missing integrated delay evidence were fixed with LU parity coverage for an ill-conditioned direct loop, network-total passivity accounting, and a delayed-runtime frequency-response gate.

Known limits recorded rather than hidden:

- Explicit time-domain delay currently uses a real, constant power-wave reference impedance. Complex or frequency-dependent references remain available for frequency-domain `NetworkData` algebra but are rejected by this real-valued FDTD adapter.
- Embedded-network coefficients remain fixed during forward stepping; gradient replay and persisted embedded results belong to Phase 4.
- `WavePort` embedding and spatial multi-device port ownership remain blocked on the missing contracts recorded in the dependency audit.
