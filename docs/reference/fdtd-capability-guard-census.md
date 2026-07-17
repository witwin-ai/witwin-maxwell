# FDTD Capability-Guard Census

This document is the maintained baseline for the AST guard gate in
`tests/api/public/test_guard_census.py`. The gate distinguishes capability gaps
from deliberate public-contract boundaries so new `NotImplementedError` paths
cannot enter unnoticed.

## Reconciled baseline

The 2026-07-16 repository state contains 129 guards:

- 109 capability guards tracked by the non-increasing test budget;
- 20 contract guards excluded by exact file and message substring.

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
| Public simulation, result, and network workflows | 18 |
| Material models | 7 |
| Postprocessing | 4 |
| **Total** | **109** |

Capability review (2026-07-16): the budget was raised from 108 to 109 for the
array-basis feature. `witwin/maxwell/postprocess/array.py` raises
`NotImplementedError("Array basis extraction is FDTD-only.")` when a non-FDTD
`Result` is passed to `array_basis()`. It is a genuine capability gap rather
than a public contract: `array_basis()` consumes the retained in-memory
full-wave PortSweep field columns, which only the FDTD backend currently
produces. It stays counted in the budget (Postprocessing 3 -> 4) so extending
basis extraction to another backend later lowers the count again.

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
