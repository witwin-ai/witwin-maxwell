# FDTD Capability-Guard Census

This document is the maintained baseline for the AST guard gate in
`tests/api/public/test_guard_census.py`. The gate distinguishes capability gaps
from deliberate public-contract boundaries so new `NotImplementedError` paths
cannot enter unnoticed.

## Reconciled baseline

The 2026-07-15 repository state contains 137 guards:

- 117 capability guards tracked by the non-increasing test budget;
- 20 contract guards excluded by exact file and message substring.

The capability baseline is distributed as follows:

| Area | Capability guards |
| --- | ---: |
| External interoperability adapter | 18 |
| Material compilers | 10 |
| Frequency-domain runtime | 5 |
| Time-domain adjoint | 14 |
| Time-domain excitation | 12 |
| Time-domain ports and lumped elements | 12 |
| Time-domain runtime | 17 |
| Public simulation, result, and network workflows | 19 |
| Material models | 7 |
| Postprocessing | 3 |
| **Total** | **117** |

Thin-wire Phase 1 added nine reviewed capability guards rather than silently
running unsupported compositions: one locally anisotropic self-term guard in
the compiler; five Bloch, off-diagonal, conductive, dispersive, and nonlinear/
modulated host guards plus one surface-impedance conductor-ownership guard in
the single-device runtime; one distributed ownership
guard; and one adjoint checkpoint/reverse guard. They remain capability debt,
not contract exclusions. Phase 2 owns the adjoint guard, Phase 3 owns the Bloch
path guard. Phase 4 owns distributed execution and SIBC/finite-loss ownership;
the other host-composition guards remain tracked capability debt for a later
compatibility plan rather than expanding the Phase 4 exit gate. Each
implementation phase must lower this budget as it removes its guards.

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
