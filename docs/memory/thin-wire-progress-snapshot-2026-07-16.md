# Thin-Wire Progress Snapshot

Date: 2026-07-16

Branch: `codex/thin-wire`

Implementation HEAD before this snapshot: `799471f`

Plan: `docs/plans/07-thin-wire-model.md`

## Current Status

Phases 0 through 3 are implemented, independently reviewed, accepted, and
preserved as separate commits. There is no uncommitted thin-wire implementation
checkpoint at the time of this snapshot. Phase 4 has not started: its complete
exit gate is blocked by missing distributed-adjoint infrastructure and by the
absence of a second local GPU required for real 1/2-GPU parity testing.

This snapshot records the dependency boundary; it does not claim Phase 4 or any
of its gates as complete.

## Completed Phases And Commits

| Phase | Commit | Status | Delivered scope |
| --- | --- | --- | --- |
| 0 | `d65f9b7` | Accepted | Frozen numerical contract, dense PyTorch reference, registered acceptance budgets, and reference tests. |
| 1 | `a68410c` | Accepted | Axis-aligned single-device PEC forward model, native CUDA sampling/update/deposition, public API, monitoring, persistence, stability guard, and performance evidence. |
| 2 | `eef3881` | Accepted | Polyline/junction/branch/loop topology, charge continuity, checkpoint/replay, native adjoint, and radius/host-material gradients. |
| 3 | `799471f` | Accepted | Arbitrary-direction conservative fragment coupling, custom/automatic grids, legal interior real-periodic paths, node/gap port binding, RF observation, and port adjoint. |

The implementation is an energy-paired subgrid current/charge network. It does
not represent a thin wire by voxelizing a small PEC cylinder.

## Work In Progress

No feature implementation is currently in progress and the worktree was clean
at `799471f` before this documentation-only snapshot. Phase 4 remains pending,
not partially accepted.

## Incomplete Or Explicitly Unsupported Scope

The following Phase 4 deliverables are not implemented or accepted:

- finite-conductivity wire laws and analytic AC-resistance validation;
- a shared passive rational skin-effect model and wire ADE state;
- nonzero ohmic-loss production, loss-output adjoint seeds, and conductor-law
  parameter gradients;
- fragment/state ownership for wires crossing device partitions;
- forward sparse reduction/scatter and the exact reverse communication path;
- single/multi-GPU value, gradient, and energy parity;
- distributed wire-port and closed-surface power/loss assembly;
- Phase 4 state-memory scaling and multi-GPU performance optimization evidence.

The current public conductor law remains PEC, and PEC wire ohmic loss is zero.
The distributed solver deliberately fails closed for thin wires, ports, and
closed-surface monitors. Trainable parallel FDTD is also rejected before
execution. These guards must not be interpreted as Phase 4 support.

Within Phase 3, geometry gradients are defined only while the compiled
fragment-to-grid stencil remains fixed. A segment crossing a cell boundary is a
discrete recompilation event. Real-periodic interior paths are supported, while
face-touching wraps and Bloch wire paths continue to fail closed.

## Most Recent Verification Evidence

The latest focused Phase 3 acceptance matrix passed 88 tests covering compiler,
CUDA forward execution, core adjoint, port runtime and adjoint, rotation, and
public API behavior. Independent module and phase reviewers reached GO after
their findings were addressed.

The registered Phase 3 numerical evidence includes:

- finest-grid axis-aligned versus oblique dipole input-impedance difference:
  `6.56e-4` relative;
- finest-grid normalized far-field pattern difference: `1.42e-3`;
- accepted port power: `6.6332e-5 W`;
- direct closed-surface Poynting flux: `6.6377e-5 W`;
- relative power-closure residual: `6.8e-4`, below the registered 1% budget;
- finite-difference checks for source amplitude, radius, and continuous oblique
  gap coordinates within the registered 2% gradient budget;
- no `aten::item`, scalar extraction, or host/device copy detected in the
  profiled wire-port stepping and observer path.

Earlier accepted evidence is retained in:

- `docs/assessments/thin-wire-phase-0-acceptance.md`;
- `docs/assessments/thin-wire-phase-1-acceptance.md`;
- `docs/assessments/thin-wire-phase-2-acceptance.md`;
- `docs/assessments/thin-wire-phase-3-acceptance.md`.

Phase 2's broader 210-test matrix contained two no-wire material-effect
precondition failures. The same values reproduced at the Phase 1 commit, so
they were documented as baseline failures rather than treated as Phase 2
regressions. No Phase 4 numerical or performance gate has been run or passed.

## Known Risks And Blockers

1. Plan 07 explicitly depends on Plan 02 Phase 7 for wire reverse
   communication. That prerequisite is still proposed and the current tree has
   no distributed checkpoint/replay, reverse halo accumulation, or global
   trainable-parameter gradient reduction for this feature.
2. Distributed thin-wire fragment/state ownership and sparse communication do
   not yet exist. Distributed RF port and closed-surface monitor ownership are
   also unavailable, preventing the required energy-parity measurement.
3. The available validation host reports one CUDA device, an NVIDIA GeForce RTX
   5080. The mandatory real 1/2-GPU value/gradient/energy parity gate therefore
   cannot be executed locally.
4. The existing surface-impedance runtime is a narrowband series R-L path, not
   the shared passive broadband rational-model infrastructure required by the
   Phase 4 contract.
5. Adding loss changes forward state, checkpoint schemas, RF monitoring, and
   reverse seeds together. Implementing only a single-device forward loss term
   would leave Phase 4 internally incomplete and must not be recorded as an
   accepted phase.

## Resume Steps

1. Complete and integrate Plan 02 Phase 7: sharded checkpoint/replay, reverse
   halo accumulation, and global parameter-gradient reduction.
2. Freeze wire-specific partition ownership: fragments by Yee-component owner,
   physical segment/node/ADE state by one stable owner, forward EMF reduction
   and current scatter, and the exact reversed communication sequence.
3. Add distributed ownership for wire-bound ports and power/loss monitors.
4. Provide at least two compatible GPUs and first validate the baseline Plan 02
   1/2-GPU objective/gradient parity contract.
5. Implement the shared passive rational descriptor and finite-conductor wire
   law, including ADE checkpoint/replay, ohmic-loss monitoring, loss adjoint
   seeds, and conductor-parameter VJPs.
6. Run targeted analytic skin-effect, energy, finite-difference, checkpoint,
   profiler, memory-scaling, and real 1/2-GPU parity tests. Only then update the
   plan and feature list and create the Phase 4 acceptance commit.
