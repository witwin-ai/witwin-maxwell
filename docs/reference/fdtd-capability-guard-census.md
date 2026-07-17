# FDTD Capability-Guard Census

This document is the maintained baseline for the AST guard gate in
`tests/api/public/test_guard_census.py`. The gate distinguishes capability gaps
from deliberate public-contract boundaries so new `NotImplementedError` paths
cannot enter unnoticed.

## Reconciled baseline

The 2026-07-16 integrated repository state (circuit co-simulation and Touchstone
network embedding merged) contains 140 guards:

- 119 capability guards tracked by the non-increasing test budget;
- 21 contract guards excluded by exact file and message substring.

The single-GPU circuit adjoint now provides the circuit replay and transpose
linear-solve VJP. Its remaining explicit limits are the omitted t=0
operating-point/initial-state pullback (therefore trainable DC source values and
circuit initial conditions), tensor seeds on the initial `CircuitData` sample,
and trainable port reference impedance. Multi-circuit execution, distributed
adjoints, and full-field/observer forward-resume accumulation remain separate
capability gaps rather than reclassified permanent contracts.

The capability baseline is distributed as follows:

| Area | Capability guards |
| --- | ---: |
| External interoperability adapter | 18 |
| Material compilers | 9 |
| Frequency-domain runtime | 5 |
| Time-domain adjoint | 18 |
| Time-domain excitation | 12 |
| Time-domain ports and lumped elements | 12 |
| Time-domain runtime | 11 |
| Public simulation, result, and network workflows | 24 |
| Material models | 7 |
| Postprocessing | 3 |
| **Total** | **119** |

This integrated baseline is the 2026-07-15 circuit co-simulation state (113
capability guards) plus the six 2026-07-16 network-embedding capability guards
(Time-domain adjoint +3, Public simulation/result/network workflows +3). The
network-embedding reconciliation is detailed in the
[Network-embedding reconciliation](#network-embedding-reconciliation-2026-07-16)
section below.

## Contract exclusions

`CONTRACT_GUARDS` in the test is the canonical exact-match inventory. Its 21
entries cover these stable contract families:

| Contract family | Count |
| --- | ---: |
| Material frequency-evaluation domains | 9 |
| Module-style scene implementation requirement | 1 |
| Non-meshable complex polygon geometry | 1 |
| Time-domain adjoint input and complex-state requirements | 2 |
| Frequency-domain adjoint input requirement | 1 |
| External adapter medium limitations | 2 |
| Bloch-boundary total-field/scattered-field slab requirement | 1 |
| Closed-surface exterior-sampling requirements | 3 |
| Time-domain-only embedded network feedback | 1 |
| **Total** | **21** |

When an implementation removes a capability guard, lower
`CAPABILITY_GUARD_BUDGET` in the same commit. Reclassifying a guard as a public
contract requires adding an exact entry to `CONTRACT_GUARDS` and updating the
two tables above. Raising the budget requires an explicit capability review and
an updated baseline here.

## Network-embedding reconciliation (2026-07-16)

The Touchstone network-embedding series (`33c1ca5`, `f76a27a`, `2ff0003`) added
exactly seven `NotImplementedError` guards and removed none (verified by
`git log -S "raise NotImplementedError" master..HEAD -- witwin/`). Six are
capability gaps and raise `CAPABILITY_GUARD_BUDGET` by six; combined with the
five circuit co-simulation guards on the 2026-07-15 baseline of 108 this yields
the integrated budget of 119 (108 + 5 + 6). The seventh network-embedding guard
is a permanent architectural boundary reclassified into `CONTRACT_GUARDS`. Each
guard was reviewed individually below. All seven are correct fail-closed
rejections and are retained; this reconciliation records the baseline instead of
deleting them.

### Capability guards (+6: "Time-domain adjoint" +3, "Public simulation, result, and network workflows" +3)

| Guard | Message | Capability review |
| --- | --- | --- |
| `fdtd/adjoint/bridge.py` `_validate_supported_configuration` (delay branch) | "Differentiable embedded-network FDTD does not support explicit delay state." | Deferred capability. The forward explicit fractional-delay line is a fixed-length buffer recurrence; its reverse-mode adjoint is not yet derived, so a differentiable run with a delay network is rejected rather than returning a wrong gradient. Implementable later (delay-line adjoint), so it counts against the budget. |
| `fdtd/adjoint/bridge.py` `_validate_supported_configuration` (RationalModel poles/proportional) | "Differentiable embedded RationalModel supports residues and direct terms only; trainable {…} are not supported." | Deferred capability. Plan 03 §5.3 defers gradients through pole relocation / passivity enforcement to a later version; trainable `poles`/`proportional` are rejected instead of silently detached. Implementable later, counts against the budget. |
| `fdtd/adjoint/bridge.py` `_validate_supported_configuration` (state-space A/B/C/D) | "Differentiable embedded networks accept trainable residues/direct on a pre-fitted RationalModel; direct trainable state-space matrices are not supported." | Deferred capability. Direct differentiation of raw discrete `A/B/C/D` is not implemented; users differentiate a pre-fitted `RationalModel`'s residues/direct terms instead. Implementable later, counts against the budget. |
| `fdtd/networks.py` `finalize_embedded_networks` (replay delay branch) | "Differentiable embedded-network replay does not support explicit delay state." | Deferred capability. Mirrors the bridge delay guard on the replay path so the checkpoint/replay adjoint cannot silently run without delay-state gradients. Implementable later, counts against the budget. |
| `simulation.py` `_validate_trainable_rf_support` (RationalModel poles/proportional) | "Differentiable embedded RationalModel supports residues and direct terms only; trainable {…} are not supported." | Deferred capability. Public-entry mirror of the bridge poles/proportional guard so the rejection surfaces at `Simulation` construction, not deep in the adjoint. Implementable later, counts against the budget. |
| `simulation.py` `_validate_trainable_rf_support` (state-space A/B/C/D) | "Differentiable embedded networks accept trainable residues/direct on a pre-fitted RationalModel; direct trainable state-space matrices are not supported." | Deferred capability. Public-entry mirror of the bridge state-space guard. Implementable later, counts against the budget. |

### Contract guard (+1)

| Guard | Message | Contract review |
| --- | --- | --- |
| `simulation.py` `_validate_network_solver` | "Embedded network feedback is defined only for the time-domain FDTD update; frequency-domain solvers cannot ignore Scene.networks." | Permanent architectural boundary, not a capability gap. Per plan 03 (2026-07-14 decision, §2.1 goal 3 and §5.2) network embedding is unified as a **time-domain** passive state-space recurrence `x[n+1]=Ad x[n]+Bd v[n]` fed back into each FDTD step; there is no per-step update in a frequency-domain solve for a network to couple into, and the plan explicitly rejects a post-simulation S-matrix cascade. This is definitionally the same family as the material "…is defined only for / is not defined for…" contract guards, so it is excluded by exact `(file, substring)` match rather than counted against the capability budget. |
