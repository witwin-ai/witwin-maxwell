# Network embedding progress snapshot

Snapshot date: 2026-07-16

Branch: `codex/network-embedding`

Implementation HEAD before this snapshot: `2ff000359a69617d633d5ea7e87f9304f75fd227`

Overall status: Phase 0 through Phase 3 are complete. The independently implementable single-device portion of Phase 4 is complete, but Phase 4 as a whole is blocked on missing spatial multi-device RF contracts.

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
- Explicit delay remains a supported forward feature, but delay ring/filter/cursor state is not part of the differentiable checkpoint schema. Differentiable runs containing explicit delay fail explicitly.
- Trainable poles, proportional terms, automatic fitting or passivity enforcement, and directly trainable `StateSpaceNetwork` matrices are intentionally rejected rather than silently detached. The implemented coefficient-gradient contract covers pre-fitted ordinary-Y `RationalModel.residues` and `direct` tensors.
- `WavePort` network embedding remains unsupported because it has no compatible time-domain terminal injection contract. Embedded networks currently connect `LumpedPort` or resolved `TerminalPort` terminals.
- The optimized full-step CUDA Graph is a narrow source-free, non-dispersive, non-nonlinear, non-modulated, non-Mur, non-SIBC path. Other supported scenes retain the ordinary field/network graph paths instead of using the specialized full-step capture.

## Recent verification evidence

- Related RF network and FDTD network-gradient regression after terminal batching and native-LU graph isolation: **219 passed**, with one PyTorch profiler warning about event-cycle clearing.
- Targeted static analysis with Ruff on every changed Python implementation and test file: passed.
- `git diff --check`: passed.
- Independent persistence review: passed, including ordinary and sharded result round trips, typed fit reports, CPU-detached tensor payloads, malformed-schema rejection, and unsafe-metadata rejection.
- Independent Phase 4 review found no remaining single-device must-fix items.
- Residue, direct-conductance, and material-near-terminal gradients passed three-step central finite differences with relative error below the plan's `2%` gate.
- Checkpoint/replay stores bounded state rather than every time step and reproduces the embedded-network terminal state.
- The no-feature ABBA comparison against the Phase 3 implementation reported `-6.98%` regression on the accepted stable run, satisfying the `<1%` regression gate. An earlier noisy 24-cell probe was rejected as evidence and is not treated as a pass.
- The true-baseline 8-port/order-32 benchmark used a 272-cells-per-axis grid, 200 steps, three ABBA/BAAB rounds, 256 network states, and an 8-by-8 implicit solve. Median baseline and connected times were 708.98 ms and 751.10 ms, respectively, for `5.9403%` overhead, satisfying the `<10%` single-device gate. Runtime assertions confirmed that the field, network, and port-observer CUDA Graphs were active.
- A nonzero-field and nonzero-network-state regression verifies capture-state restoration and eight-step eager/full-graph parity. A separate regression verifies fallback when nested SIBC state is present.

## Known issues and risks

- The branch does not satisfy the plan's full completion definition until the spatial multi-GPU parity gate can be executed and passed.
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
