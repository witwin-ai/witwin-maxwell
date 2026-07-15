# FDTD Capability-Guard Census

This document is the maintained baseline for the AST guard gate in
`tests/api/public/test_guard_census.py`. The gate distinguishes capability gaps
from deliberate public-contract boundaries so new `NotImplementedError` paths
cannot enter unnoticed.

## Reconciled baseline

The Phase 2 circuit-coupling state on 2026-07-15 contains 133 guards:

- 113 capability guards tracked by the test budget;
- 20 contract guards excluded by exact file and message substring.

The five additions are explicit phase-owned circuit guards: live `CircuitData`
snapshot persistence and multi-port execution are removed in Phase 3, while
transpose-solve differentiation and distributed execution are removed in
Phase 4. They were reviewed as capability guards rather than reclassified as
permanent contracts; their owning phases must lower this baseline as they land.

The capability baseline is distributed as follows:

| Area | Capability guards |
| --- | ---: |
| External interoperability adapter | 18 |
| Material compilers | 9 |
| Frequency-domain runtime | 5 |
| Time-domain adjoint | 14 |
| Time-domain excitation | 12 |
| Time-domain ports and lumped elements | 12 |
| Time-domain runtime | 10 |
| Public simulation, result, and network workflows | 23 |
| Material models | 7 |
| Postprocessing | 3 |
| **Total** | **113** |

## Contract exclusions

`CONTRACT_GUARDS` in the test is the canonical exact-match inventory. Its 20
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
| **Total** | **20** |

When an implementation removes a capability guard, lower
`CAPABILITY_GUARD_BUDGET` in the same commit. Reclassifying a guard as a public
contract requires adding an exact entry to `CONTRACT_GUARDS` and updating the
two tables above. Raising the budget requires an explicit capability review and
an updated baseline here.
