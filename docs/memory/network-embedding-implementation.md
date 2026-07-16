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
- Scene/compiler lowering to a stable, interval-certified passive admittance state-space descriptor connected to the existing sparse `LumpedPort` / resolved `TerminalPort` Yee geometry; unsupported `WavePort`, non-FDTD solvers, out-of-fit-band result requests or excitation spectra, and multiport cases fail before stepping.
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

- Phase 2 executes one network port. Multiport implicit coupling and bounded delay realization belong to Phase 3.
- Embedded-network coefficients are fixed for forward execution. Their analytic reverse and persistence schema belong to Phase 4.
- Spatial multi-device ownership remains blocked on the missing public port reference-point ownership and hot-path scalar transport contracts recorded in the dependency audit.
