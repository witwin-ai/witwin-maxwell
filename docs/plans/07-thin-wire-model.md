# Thin-Wire Subgrid Model Implementation Tracker

Status: active
Source plan date: 2026-07-14
Target evidence: E3 production

This tracked implementation follows the full roadmap source
`docs/plans/next-functional-2026-07/07-thin-wire-model.md` from the originating
workspace. The source was read in full before implementation; this worktree was
created from a revision that did not contain the roadmap directory.

The implementation is an energy-paired auxiliary current/charge network. It is
not a thin-cylinder voxelization and it does not add a CPU solver path. Public
capabilities remain within `Scene -> Simulation -> Result`, with device-resident
PyTorch coefficients. Native FDTD stepping is a Phase 1 deliverable.

## Frozen Acceptance Budget

- Torch/reference parity: `rtol <= 1e-5`.
- Analytic core impedance or propagation quantities: relative error `<= 2%`.
- Energy and charge residuals: relative error `<= 1%`.
- Convergence evidence: at least three grid/time-step levels.
- Supported parameter gradients: relative error `< 2%`.
- Multi-device field/monitor/power parity: inherit the spatial shard contract.
- No-wire runtime regression: `< 1%`.

## Phase Progress

| Phase | Scope | Status | Evidence |
| --- | --- | --- | --- |
| 0 | Discrete energy derivation, effective radius, torch reference | accepted | `docs/assessments/thin-wire-phase-0-acceptance.md` |
| 1 | Axis-aligned PEC single-wire GPU forward | accepted | `docs/assessments/thin-wire-phase-1-acceptance.md` |
| 2 | Network topology, checkpoint, and adjoint | pending | - |
| 3 | Arbitrary direction, nonuniform grid, and port binding | pending | - |
| 4 | Finite conductivity, broadband loss, and multi-device parity | pending | - |

Every phase is committed independently after targeted tests and independent
review. Phase 3 requires the standard RF port contract. Phase 4 uses the existing
spatial shard ownership/halo contract and must reject trainable distributed
execution if reverse sparse communication is unavailable.
