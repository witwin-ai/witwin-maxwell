# Network embedding progress snapshot

Snapshot date: 2026-07-16

Branch: `codex/network-embedding`

Implementation HEAD before this snapshot: `2ff000359a69617d633d5ea7e87f9304f75fd227`

Overall status: Phase 0 through Phase 3 are functionally complete, with one recorded Phase 3 gap: the multiport `< 0.02` S-parameter gate is not yet backed by a reference *outside* the fit's model class (see "Incomplete, blocked, or intentionally unsupported scope" below and the "Independence caveat" in network-embedding-implementation.md). The independently implementable single-device portion of Phase 4 is complete, but Phase 4 as a whole is blocked on missing spatial multi-device RF contracts.

## Completed phases and commits

- Phase 0, external network-file I/O and `NetworkData` interoperability: `53ae82c` (`Implement network file interoperability`).
- Phase 1, physicality diagnostics, passive rational models, and state-space realization: `25639ac` (`Implement passive rational network models`).
- Phase 2, same-step single-port FDTD network embedding: `33c1ca5` (`Implement single-port network embedding`).
- Phase 3, ordered multiport embedding and bounded explicit/automatic delay: `f76a27a` (`Implement multiport network delay embedding`).
- Phase 4 single-device work, including discrete network adjoints, bounded checkpoint/replay, persisted embedded results, batched terminal updates, and CUDA Graph performance work: `2ff0003` (`Implement differentiable network embedding`).

The implementation follows the existing FDTD-only, GPU-first, PyTorch-native `Scene -> Simulation -> Result` architecture. No alternate public solver architecture or CPU fallback was added.

## Current work

There is no uncommitted implementation or checkpoint/WIP at the time of this snapshot. Development is paused at the spatial multi-device dependency boundary. The branch contains the complete local commit chain listed above and was clean before this documentation-only snapshot was created.

## Incomplete, blocked, or intentionally unsupported scope

- The combined Phase 4 single-/multi-GPU port voltage/current parity gate (`rtol <= 2e-5`) is **not passed**. Single-device numerical coverage passed, but the multi-GPU comparison cannot be executed: spatial `DistributedFDTD` currently rejects or removes RF ports and does not expose shard-owned port fragments, a unique reference-point owner, or per-step scalar reduce-to-owner/scatter-from-owner transport. These are plan 01/02 public contracts and were not privately invented in task 03.
- Spatial multi-GPU network placement, communication-byte measurement, and distributed adjoint execution remain blocked by the same contracts. The single-device benchmark correctly reports zero spatial communication and makes no distributed-performance claim.
- The Phase 3 multiport `< 0.02` S-parameter gate is **not yet independently backed**. `test_touchstone_multiport_fit_matches_independent_network_reference` currently uses references inside the order-1 fit's own model class (a single-pole `StateSpaceNetwork` fitted at `order=1`, and a frequency-flat 4-port conductance representable by the direct term `D` alone), so the measured margin is `~1e-16` and the gate is dominated by the float64 Touchstone round-trip. The test still guards round-trip fidelity, port ordering, and connection mapping. Backing plan section 7 Phase 3's independent-reference intent needs a reference outside the fit's model class (higher-order or non-rational, frequency-dependent for the 4-port); an initial out-of-class RLC-ladder attempt tripped the fit's own `relative_tolerance` acceptance gate, so this is a calibration task, not a one-line swap.
- Explicit delay remains a supported forward feature, but delay ring/filter/cursor state is not part of the differentiable checkpoint schema. Differentiable runs containing explicit delay fail explicitly.
- Trainable poles, proportional terms, automatic fitting or passivity enforcement, and directly trainable `StateSpaceNetwork` matrices are intentionally rejected rather than silently detached. The implemented coefficient-gradient contract covers pre-fitted ordinary-Y `RationalModel.residues` and `direct` tensors.
- `WavePort` network embedding remains unsupported because it has no compatible time-domain terminal injection contract. Embedded networks currently connect `LumpedPort` or resolved `TerminalPort` terminals.
- The optimized full-step CUDA Graph is a narrow source-free, non-dispersive, non-nonlinear, non-modulated, non-Mur, non-SIBC path. Other supported scenes retain the ordinary field/network graph paths instead of using the specialized full-step capture.
- The antenna matching-block corpus item (`tests/rf/antenna/test_antenna_matching_block.py`) validates only the realized-gain **bookkeeping** that `Result.antenna` performs on a real FDTD-radiated far field: a driven z-directed lumped port radiates inside a PML box, a `ClosedSurfaceMonitor` supplies the genuine near field, and Stratton-Chu transforms it. It asserts intrinsic gain (referenced to accepted power) is invariant under an external lossless match and realized gain (referenced to incident power) scales by the network-predicted mismatch ratio. It does **not** validate an input-impedance-driven match against the radiator's own port impedance: an electrically small lumped-port radiator does not yield a physical raw input reflection (the test asserts `|Gamma| > 1` to record this), so the load presented to the matching network is a modelled representative reactive load, not a value read back from the port. A full radiating-structure input-impedance-and-match validation remains out of scope. Separately, the public network surface exposes `NetworkData.from_y` and `NetworkData.s` but no two-port/one-port cascade or termination helper, so the matched-reflection prediction uses a single load-cascade expression on top of the public S-matrix; a public cascade helper is a possible future addition.

## Recent verification evidence

- Related RF network and FDTD network-gradient regression after terminal batching and native-LU graph isolation. Re-measured 2026-07-16 on 1x RTX A6000 (torch 2.13.0+cu130, CUDA 13.0) with
  `pytest tests/rf/network tests/rational tests/gradients/test_fdtd_network_adjoint.py`: **183 passed, 0 failed** of 183 collected.
  The same selection with CUDA hidden is **158 passed, 25 skipped**. Read the pass count with that gate in mind: 25 of 183 tests
  are CUDA-gated, and they are the ones that carry every end-to-end numerical claim (FDTD embedding parity, multiport S-parameters,
  delay integration, finite-difference gradients). On a CUDA-less runner this selection is green while validating no FDTD physics at all.
  A previously recorded "219 passed" for this evidence line cited neither a command nor a skip count and did not reproduce here; it is withdrawn.
- Resolved (owner ratification, 2026-07-16):
  `tests/rf/network/test_network_delay.py::test_cuda_delay_hot_path_has_no_allocation_or_host_transfer` previously asserted no
  profiler key matched `"memcpy"`, but on torch 2.13/CUDA 13.0 the delay read emits `Memcpy DtoD (Device -> Device)` /
  `cudaMemcpyAsync` from its on-device `copy_` calls (the ring's read-before / write-after split that lets the implicit
  zero-delay solve land between read and write). Device-to-device copies are not host transfers, and measured device-memory
  allocation in that window is 0, so the hot path is clean. The assertion has been aligned with the contract the test name
  states ("host transfer") and with the sibling hot-path tests (`test_network_observer_hot_path.py:130`,
  `test_network_runtime.py:173`): it now bans `aten::item`, `aten::_local_scalar_dense`, and `Memcpy HtoD`/`Memcpy DtoH` and
  keeps the zero device-allocation guard. This narrows the transfer ban (DtoD ring copies are now permitted, with the recorded
  reason above) while strengthening the host-sync ban from the old broad `"item"` substring to explicit
  `aten::item`/`aten::_local_scalar_dense` keys. Verified green on 1x A6000; a fail-check injecting a `.cpu()` DtoH transfer into
  the profiled loop trips the assertion.
- Targeted static analysis with Ruff on every changed Python implementation and test file: passed.
- `git diff --check`: passed.
- Independent persistence review: passed, including ordinary and sharded result round trips, typed fit reports, CPU-detached tensor payloads, malformed-schema rejection, and unsafe-metadata rejection.
- Independent Phase 4 review found no remaining single-device must-fix items.
- Residue, direct-conductance, and material-near-terminal gradients passed three-step central finite differences with relative error below the plan's `2%` gate.
- Checkpoint/replay stores bounded state rather than every time step and reproduces the embedded-network terminal state.
- The Phase 4 no-feature regression gate (`< 1%`) is **unmet, pending re-measurement**. The ABBA comparison against Phase 3
  reported `-6.98%` on a change that adds no feature to the measured configuration, so the true effect is ~0% by construction and
  `-6.98%` is a measurement of the harness noise floor, not of a regression. A noise floor near 7% cannot resolve a 1% gate in
  either direction. This reading was previously recorded as "satisfying the `<1%` gate"; that was wrong, and it is inconsistent with
  the same document rejecting a `+7.26%` probe *as noise* at the same magnitude. Accepting one sign and rejecting the other from the
  same noise distribution is not a pass. Re-measure with a harness whose dispersion is reported and small relative to 1% before
  claiming this gate.
- The true-baseline 8-port/order-32 benchmark recorded a 272-cells-per-axis grid, 200 steps, three ABBA/BAAB rounds, 256 network
  states, an 8-by-8 implicit solve, median baseline/connected times of 708.98 ms and 751.10 ms, and `5.9403%` overhead against the
  `<10%` single-device gate, with runtime assertions that the field, network, and port-observer CUDA Graphs were active.
  **Unevidenced at this HEAD**: the cited artifact `.cache/phase4-network-performance-true-grid272.json` is not present in the
  repository or on the current host, so the number cannot be re-verified here. It was recorded on different hardware (see below).
  Treat it as a claim awaiting a reproducible artifact, not as a satisfied gate.
- A nonzero-field and nonzero-network-state regression verifies capture-state restoration and eight-step eager/full-graph parity. A separate regression verifies fallback when nested SIBC state is present.

## Known issues and risks

- The branch does not satisfy the plan's full completion definition until the spatial multi-GPU parity gate can be executed and passed.
- Two Phase 4 performance gates (`< 1%` no-feature regression, `< 10%` connected 8-port/order-32 overhead) are not currently
  backed by a reproducible artifact at this HEAD. Neither should be reported as passed until re-measured with recorded dispersion.
- The performance result is hardware- and configuration-specific. It was measured on an NVIDIA GeForce RTX 5080 with PyTorch 2.10.0 and CUDA 12.8; future runtime or kernel changes must rerun the benchmark rather than reuse the recorded percentage.
- Raw sampled-data causality and passivity diagnostics remain finite-band diagnostics. Rational-model acceptance uses the documented pole-aware finite-band certificate rather than an all-frequency Hamiltonian certificate.
- Automatic delay extraction and the first-order fractional-delay realization have bounded, documented accuracy; complex or frequency-dependent time-domain reference impedances are rejected.
- Embedded `state_norm` is a detached diagnostic. Differentiable voltage, current, and power outputs remain the supported optimization observables.

## Resume steps

1. Integrate or rebase onto the plan 01/02 public distributed-port contracts once they provide shard terminal fragments, a unique owner, and GPU-resident scalar reduce/scatter transport.
2. Add owner-side network-state placement and local-connection collective bypass without changing the public `Scene -> Simulation -> Result` contract.
3. Add real split-port single-/multi-GPU voltage/current parity tests and communication-byte accounting, then run the `rtol <= 2e-5` gate.
4. Add distributed checkpoint/adjoint coverage only after the shared distributed adjoint contract exists.
5. Rerun the complete RF network, gradient, persistence, no-feature regression, and 8-port/order-32 performance suites before declaring Phase 4 complete.
