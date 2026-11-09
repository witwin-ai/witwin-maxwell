# Prescribed complete directional regional and scalar actions

`DirectionalRegionalOperator` in `fmm_regional.py` takes an admitted
`DirectionalNestedFarActions`, explicit original-near leaf pairs and the actual
integration configuration. The FAR rectangles and near rectangles must share
physical tree permutations, be disjoint and have integer area exactly Nt*Ns.
The half-open rectangle sweep plus bounded domains proves complete physical
edge-pair coverage without allocating an N² mask. Missing or overlapping pairs
fail before original integration or packed near allocation. Global near blocks
bind the actual exterior provider medium/configuration identities; local near
blocks use the original regional scalar D/K integrator. An empty near partition
allocates no packed arrays.

Packed near retained bytes are planned exactly before integration:
8*(2*unique_test_leaves+1+3*near_pairs) plus the sum of
32*nr*nc+16*(nr+nc) per pair, or zero with no pairs. Forward/Hermitian combine
bounded original near and FAR physical RHS actions using fixed current custom
backward. Partition, configuration, budgets, far dependencies and near numeric
versions are rechecked at forward, receipt and saved backward. Geometry/material
parameter differentiation is not supported by this fixed operator path.

`DirectionalSystemActions` in `fmm_system.py` requires the complete actual global
exterior provider plus exactly one complete original-basis/interior-medium
regional operator per disjoint body. All regions use one positive balancing
impedance eta, defaulting to abs(exterior.eta). The numerical helper
`directional_system_ideal_bound` combines already admitted ideal receipts using
B_ext+sqrt(sum(B_int²)). Exterior and interiors overlap, so RSS over all media
would be invalid. Physical body provenance is checked by the system constructor;
the standalone numerical helper is not a provenance admission API.

The system preflights embedding16N bytes, conservative action workspace and the
global ideal absolute budget before allocating embedding indices. Forward adds
interior contributions at actual J/M body indices; full Hermitian and current
backward use the same complete composition. New owned indices, region identity,
body offsets, common eta, budgets and cached receipts are guarded against
mutation. Current actions retain no generated graph; higher current derivatives
use the guarded linear operator.

Ideal exact-integration truncation receipts do not bound actual finite near or
projection quadrature or certify native roundoff. Caller FAR and regional
ownership remains excluded from component inventories. Action plans exclude
near integration setup, caller input/output, allocator cache and backend FFT
workspace. No public directional configuration, automatic cones, full resources,
crossover, scaling or production qualification is claimed. See the matching
assessment for the frozen selected evidence and unclosed capabilities.
