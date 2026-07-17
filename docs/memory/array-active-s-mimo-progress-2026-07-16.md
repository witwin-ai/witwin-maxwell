# Array and MIMO implementation progress snapshot

Date: 2026-07-16

Branch: `codex/array-mimo`

Status: Phase 0 committed; Phase 1 checkpoint/WIP; later phases not started

## Completed work

### Phase 0: analytical contract

Phase 0 is complete and committed as `67064e5` (`Freeze array basis analytical
contracts`). It freezes the array power-wave, embedded-element-pattern, phase-center,
weight-shape, active-network, dtype/device, and linear-scene capability contracts. The
public torch-native analytical combination path and its API/gradient/error-path tests
were accepted at the experimental evidence level.

### Phase 1 implementation checkpoint

The working tree contains a substantial, uncommitted Phase 1 implementation. This is a
checkpoint/WIP, not a completed phase commit. Implemented work includes:

- compact per-column `PortSweep` execution data with measured incident waves;
- `Result.array_basis(...)` extraction without an FDTD rerun;
- measured-wave-normalized embedded element patterns and stable content fingerprints;
- Hermitian closed-surface `Q_rad` and absolute `real(a^H Q_rad a)` radiated power;
- broadside/endfire full-wave basis-versus-direct comparisons for two- and four-element
  arrays, including complex far fields and port powers;
- simultaneous independent lumped-port excitations for linear scenes;
- exact Yee E/H observer time staggering, exact adjoint transpose schedules, and exact
  nonuniform primal-cell surface quadrature;
- strict tensor dtype/device behavior through the array, antenna, and network paths;
- safe `ArrayBasisData` persistence using the shared network schema and
  `weights_only=True` loading;
- a frozen single-device Phase 1 benchmark scene and qualification driver;
- user-facing feature inventory and active implementation/acceptance records.

The benchmark now synchronizes each direct beam before its temporary result can leave
scope. This closes the lifecycle issue found after an asynchronous device-memory error,
but the frozen performance qualification has not subsequently completed.

## Current work and unmet gates

Phase 1 remains in progress because its frozen performance gate has no successful
qualifying artifact. The required protocol is three warmups, five samples, and four
alternating-order rounds on the exact four-element, `96 x 96 x 96`, 4,096-step,
`181 x 361` angular-grid workload.

Existing exploratory outputs are explicitly non-qualifying. A two-sample,
two-alternating-round exploratory run measured:

- basis plus 16 combinations median: `78.4747 s`;
- 16 direct solves median: `391.2515 s`;
- 16 combinations median: `0.003182 s`;
- basis/direct ratio: `0.2006` against the `<= 0.40` threshold;
- combination/one-solve ratio: `0.000130` against the `< 0.10` threshold.

These results support the expected performance direction but do not satisfy the frozen
sampling protocol. Phase 1 must not be described as fully accepted until that protocol
produces a successful qualifying record and the benchmark summary is updated.

## Validation evidence

The most recent completed evidence includes:

- the broad Phase 1 RF/FDTD regression: `257 passed in 46.23 s`;
- focused independent Phase 1 acceptance: `33 passed`;
- exact forward/adjoint observer staggering tests: `5 passed`;
- real CUDA full-wave tests covering two- and four-element broadside/endfire basis
  comparisons, simultaneous direct sources, complex field phase, port powers, and a
  two-frequency modal-port basis;
- Hermitian `Q_rad` error of zero at the tested precision, minimum eigenvalue
  `2.47e-4`, and accepted/radiated residual `0.1766%` in independent review;
- canonical broadside/endfire weighted complex-field errors of `1.766e-6` and
  `2.219e-6`, with phase RMS errors below `1.52e-4 deg`;
- canonical physical power residual `0.0700%`, below the frozen 1% numerical gate;
- Ruff and `git diff --check` passing, aside from Windows line-ending notices;
- independent API/RF and Phase 1 reviews with all implementation findings closed.

## Known issues and risks

- The frozen Phase 1 performance qualification is blocked by the current local
  device/driver execution state. Multiple isolated retries did not complete the first
  canonical `_build_basis` within 30 minutes. The process accumulated only seconds of
  CPU time while the GPU remained mostly at 0%/P8 with a retained CUDA context.
- A separate synchronized CUDA tensor probe completed normally, and no competing
  Python process, compiler process, or worktree-local build lock was present. The
  problem is therefore specific to the large canonical solver workflow on the current
  execution state; it is not recorded as a passed or failed numerical gate.
- One earlier frozen-protocol attempt ran for approximately 99 minutes before an
  asynchronous illegal-memory-access report during repeated direct solves. The
  direct-result synchronization fix was added afterward, but the current execution
  state has prevented a complete post-fix qualification.
- `benchmark/RESULTS.md` intentionally has no Phase 1 acceptance row yet because no
  qualifying record exists.
- Phase 1 changes are checkpoint/WIP and are being committed together only because the
  user requested a durable progress checkpoint and remote backup. They are not a
  phase-completion commit.

## Incomplete and skipped scope

- Phase 1 frozen performance qualification and final phase-completion commit are
  incomplete.
- Phase 2 codebook, scan-grid, max-hold, metadata, and basis-cache work has not started.
- Phase 3 global dual-polarization correlation, ECC, diversity gain, mean effective
  gain, and environment spectra have not started.
- Phase 4 scene-gradient aggregation, constrained weight tools, and
  domain-decomposition aggregation integration have not started.
- Task-level multi-device scheduling, 1/2/4-device scaling, and multi-device
  value/gradient parity were explicitly removed from scope by the user. This is a scope
  reduction, not passing evidence. Single-device work and the domain-decomposition
  aggregation contract remain in scope.

## Resume steps

1. Start from a clean device/driver execution window, preferably after a host or driver
   restart, with no competing GPU workload.
2. Re-run the exact Phase 1 benchmark with the `witwin2` environment and worktree-local
   `TORCH_EXTENSIONS_DIR` and `CUDA_CACHE_PATH`.
3. Require a successful `qualifying: true` 3/5/4 artifact; retain all raw samples,
   hardware metadata, numerical comparisons, and failure logs.
4. Update the Phase 1 acceptance record and `benchmark/RESULTS.md`, then repeat the
   independent Phase 1 review.
5. If every Phase 1 gate passes, create a separate neutral Phase 1 completion commit.
6. Continue with the remaining single-device Phase 2 deliverables; do not reintroduce
   the user-skipped task-level multi-device work.
