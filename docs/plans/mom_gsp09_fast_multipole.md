# GSP-09 free-space and periodic fast multipole implementation

Status: OPEN. Worktree <scratch>/witwin-maxwell-gsp09, branch gsp09-fmm, based on
GSP-08 commit 459d57e. GSP-08 resource/complexity/physical gates and GSP-01
physical failures stay open; starting this stage does not waive those gates.

## Required scope

- Genuine low/mid-frequency Helmholtz M2M, M2L and L2L expansions.
- Tolerance- and electrical-size-dependent expansion order.
- Directional/high-frequency branch for growing electrical sizes.
- Exact near correction with complete, nonoverlapping pair coverage.
- Actual periodic M2L/remote-image local expansions. PME and H2/Ewald do not
  satisfy periodic FMM acceptance.
- Shared operator, adjoint/Riesz/Krylov and Scene/Simulation/Result contracts.
- H2 and dense numerical controls, full physical thresholds, pre-registered N
  sequences for fixed and growing electrical size, repeated timings, all-point
  slopes, order/rank growth, crossover and retained/setup/peak resources.

No linear-complexity or production FMM claim is permitted before those gates.

## Cartesian expansion foundation

fmm_expansion.py now provides actual polynomial multipole moments and local
coefficients, exact M2M/L2L polynomial shifts, and free-space Helmholtz Green and
gradient M2L tables. No H2 interpolation basis or converted H factors are used.
The dimensionless radial Helmholtz ODE generates multivariate Taylor
coefficients; normalization and equations are maintained in
../reference/mom_cartesian_helmholtz_expansions.md.

/tmp/gsp09-expansion-v1.log: 5 passed in 5.14s. Two successive M2M/L2L shifts
agree with direct polynomial evaluation. Separated source/target point clouds
compare both Green and gradient matrices with direct free-space evaluation at
orders 2, 4 and 6 for k=0, 6.3 and 6.3+0.4i. Maximum relative matrix errors
at order6 are 3.10e-12, 2.23e-11 and 2.40e-11 respectively, below the 1e-8
control. Order4 finite-frequency errors exceed 1e-8 and are retained as
convergence evidence, not marked qualified. Exact charge adjoints, the local
Helmholtz differential identity and early table-budget rejection also pass.

Next: support/separation-aware expansion spaces and nested upward/downward
passes, then RWG moments, original exact near composition and physical operator
integration. Fixed order and point-cloud controls are not full FMM qualification.

Fixed-input contract follow-up: /tmp/gsp09-expansion-contract-v1.log,
2 passed, 4 deselected in 2.90s. Trainable wave-number tensors, trainable
geometry, active-sign loss and coincident centers reject; table-budget
rejection remains before coefficient generation. No broad regression repeated.

## Nested RWG multipole moments

NestedRWGMultipoles supports one RWG basis or a global ScalarSystemBlocks
provider. Only leaf RWGs are integrated against Cartesian monomials centered
at their node. Internal nodes use the actual binomial M2M shift; downward
applies its Hermitian transpose and projects leaf local coefficients onto
RWGs. No expanded internal RWG moment matrices are retained. Mixed-body
leaves route each edge to its original body geometry while preserving order.

Each node is checked against the complete physical RWG support bounds.
Centers and enclosing-sphere radii are owned and guarded along with leaf
moments and transfers. Geometry/provider/tree bindings are checked on action.
A retained budget counts centers/radii, leaf edge maps and complex moments,
and all shift matrices before moment/transfer allocations. Shared geometry,
trees and workspace are excluded; no peak qualification is claimed.

For total polynomial degree p, linear RWGs plus the Duffy Jacobian require
q=floor((p+4)/2) Gauss nodes per coordinate as a sufficient exact-arithmetic
rule. Explicit smaller q remains experimental and receipts mark it nonexact.
Floating-point integration and transfer errors remain uncertified.

Initial controls /tmp/gsp09-nested-moments-v1.log: 2 passed in 16.12s.
A depth-6 global two-body tree matches independent q7 integration at every
node (maximum scale-normalized relative error 4.826450584447342e-15).
All-node upward/downward pairing, native current gradient, mixed-body leaves,
body-local trees, moment storage equality, early capacity failure and stale
center rejection pass. Follow-up adds rejection of a shifted tree that does
not enclose the supplied RWG supports.

Next remains separated multipole/local regional actions, original exact near
composition and the global-exterior/body-interior physical operator. This
increment does not close a complete FMM backend or any production gate.

Support-bound follow-up: /tmp/gsp09-nested-moments-v2.log, 2 passed in
17.30s; maximum all-node normalized difference 5.81081810976396e-15.
No full regression or independent audit repeated for this primitive increment.

## Electromagnetic separated M2L

ElectromagneticM2L maps two current families and four vector/divergence
channels from source multipoles to target local coefficients. It reuses the
scalar PMCHWT channel algebra extracted from InterpolationKernel, including
the conjugated electric/magnetic cross coefficients and negative conjugate
transpose of the curl-gradient kernel in the adjoint. Projection and medium
bindings and both owned coefficient tables are guarded.

Admission requires equal order/precision/device and a finite positive center
distance with summed complete-support sphere radii at most a declared ratio
strictly below one (default 0.5). This is a geometric separation condition,
not a truncation or native arithmetic error certificate. The table budget
excludes construction transients and shared projection storage. Full near/far
pair coverage, regional/system composition and adaptive order remain next.

Evidence: /tmp/gsp09-m2l-v1.log recorded 1 failed, 4 passed in 89.35s;
the new fixture passed an unsupported Material keyword sigma. Existing
direct regional/system controls all passed after extracting the shared algebra.
The fixture now uses public sigma_e and sigma_m. Targeted rerun
/tmp/gsp09-m2l-v2.log: 1 passed in 4.51s, impedance-balanced weak matrix
relative error 3.8086516986391496e-11 against original order6 quadrature
(unchanged 1e-8 gate). Native current gradients, exact finite-map adjoint,
coincident-support rejection and stale M2L table rejection pass. No broad
regression or independent audit repeated for this primitive increment.

## Complete-support sphere partition

sphere_block_partition reuses the existing bounded depth-first CUDA pair
traversal. Its far predicate matches ElectromagneticM2L: positive center
distance and sum of full-support radii <= separation_ratio*distance, with
0 < separation_ratio < 1. Otherwise it splits the larger nonleaf sphere, or
emits an exact pair when both nodes are leaves. Nonfinite sphere arithmetic
rejects. No global edge-pair list is constructed by the traversal.

The existing strong-box predicate remains unchanged for H/H2. Terminal blocks
form a disjoint covering of the ordered coefficient-pair space, including
rectangular trees and coincident supports. This does not prove an error bound
or linear complexity; regional M2L/near composition remains the next step.

/tmp/gsp09-partition-v1.log: 12 passed in 108.19s. Square (73x73) and
rectangular (73x41) coverage counts are exactly one for every ordered pair;
pair batches 1, 7 and 64 yield identical terminal partitions, with both
near and far blocks. Far sphere predicates and near leaf status are checked.
Coincident-support, invalid-ratio, stale-tree and existing H/H2 tree controls
also pass. No full regression or independent audit repeated for this increment.

## Regional M2L and exact near composition

MultipoleRegionalOperator now consumes the complete-support sphere partition.
Far records are actual ElectromagneticM2L maps; leaf near records are original
weak quadrature, streamed into final packed storage. One upward pass supplies
all far interactions and one downward pass projects accumulated local
coefficients. Forward, adjoint, far accumulation and rectangular exact-near
embedding share the already-tested regional action code; only expansion-sized
local coefficient allocation differs from the interpolation representation.

Global spaces require the same ScalarSystemBlocks provider, its common
exterior object and its integration configuration. Body-local spaces support
regional medium sums; cross-body spaces allow one exterior medium. M2L table
bytes are cumulatively limited before each table, while the packed-near gate
rejects before original weak integration. Those are separate retained scopes,
not total or setup peak-memory admission. All region/projection/medium/near/far
bindings remain guarded.

The current stage still requires global exterior plus body-interior system
assembly, Krylov/Riesz/public integration, adaptive order/error control,
periodic and high-frequency branches, measured complexity and full physical
qualification. A regional primitive passing algebraic controls does not close
those requirements.

/tmp/gsp09-regional-v1.log: 3 passed in 9.64s. Two-body global exterior
has two M2L cross blocks and two exact self blocks; sentinels prohibit
far weak-block requests during construction. Full 120x120 weak-matrix
relative error is 2.024217684052538e-15 (dominated by exact self terms;
the isolated far accuracy is separately covered by the M2L tests). A
60x240 exact rectangular action matches its quadrature oracle with zero
observed difference. Both check adjoints and native current gradients.
Cumulative table rejection and near-storage mutation rejection pass. The
affected direct-interpolation far control also passes, error 4.77e-11.
No unrelated full regression or independent audit repeated for this increment.

## Global system and shared Krylov adapter

MultipoleSystemActions builds one global exterior multipole tree and one
interior tree per body. Each region contains precisely its own medium; global
J/M index maps scatter interior contributions into the shared coefficient
layout. The existing direct-system forward/adjoint orchestration and native
fixed-operator backward are reused. Interpolation tables and reference H
construction are not part of this path. Three separate retained budgets
accumulate projection moments/transfers, M2L tables and packed near blocks
across the regions; trees, body maps, provider/input data and all setup/solve/
backward/allocator/host workspace are explicitly excluded.

MultipoleOperator uses the shared fixed-kernel Riesz/Krylov adapter with a
multipole action factory. It preserves RWG diagonal preconditioning, inner
iteration failures, no dense fallback, dual plus original-row sampled residual
admission, stale-state checks and diagnostic-only persistence receipts.
Metadata explicitly states internal FMM, no interpolation construction and
no physical/complexity/truncation qualification. Public Scene FMM configuration
and exact Result schema integration remain next; this is not a public backend.

/tmp/gsp09-system-v1.log: 1 passed in 7.46s. Two heterogeneous magnetic
bodies produce full weak relative error 1.2580204217799189e-15 versus original
assembly. Exact self blocks dominate this metric; isolated far accuracy stays
covered separately. Regional medium identities, global placement, adjoints,
native current derivatives with no saved forward tensors, retained-table sum
and stale index rejection pass. Construction forbids interpolation tables.

/tmp/gsp09-krylov-v1.log: 2 passed in 57.74s, covering direct H2 and FMM
through the shared physical two-body Ex/Ey GMRES/block-GMRES controls.
FMM original full dual residuals 5.667922231070019e-13 and
9.902067807833857e-12; balanced current errors 2.902183543119382e-11
and 2.5963410040964504e-10, below unchanged 1e-8/1e-7 gates.
Riesz current gradient, iteration-failure preservation, direct/fallback
rejection and stale metric checks pass. Independent read-only system/solver
integration audit found no P1/P2; it does not certify truncation, peak memory,
complexity or production scope. Full phase/cross-repository tests remain pending.

## Public free-space FMM and exact decode persistence

MoMConfig(operator_backend="fmm", fmm=FMMConfig(...)) now uses the physical
multipole runtime in Scene/Simulation/Result. Local RWG Riesz storage and
iterative stopping without dense fallback are required; preconditioning is
none or rwg_diagonal. Reference-H/H2/direct-H2 configurations cannot be
supplied for this backend. Periodic and HB combinations reject pending their
dedicated implementation; the scalar scene compiler still rejects tensors.

FMMConfig version 1 includes order, moment quadrature, leaf size, sphere
separation, cumulative table/near/projection budgets and both solve residual
tolerances. Every field is validated and enters numerical, sampled operator
and RHS identities. Schema 11 restores exact scattered coefficients, incident
carrier and field decode geometry without constructing any FMM/H/interpolation
operator. It does not persist executable multipole tables. Outer Result
metadata is checked against the inner physical payload as for other backends.

The fmm name denotes actual multipole/local algebra, not a complexity claim.
This is fixed-order experimental free-space scalar capability: tolerance/ka
order selection, directional/high-frequency and periodic remote-image
expansions, full resource/physical/complexity acceptance and cross-repository
requirements are still open. Previous notes saying public integration was
pending are superseded by this section, without closing those other gates.

/tmp/gsp09-public-v1.log: 20 passed in 50.80s, including strict complete
configuration roundtrips/invalid inputs and H/H2/direct-H2/FMM public solve
and restore controls. FMM solve prohibits dense assembly, reference H and
interpolation construction; restore prohibits all hierarchical runtime
constructors. Restored currents and queried fields match exactly, and
backend/order/qualification identity corruption rejects. Independent read-only
public integration audit found no P1/P2; its feature-inventory update request
is addressed. Full phase/cross-repository/physical acceptance remains pending.

## Cauchy kernel truncation admission foundation

fmm_error.py now evaluates analytic absolute Green and gradient-component
bounds for the actual double-total-degree M2L truncation. A bivariate complex
scaling of source and target offsets admits uniform coefficient bounds on
complete spheres. The maintained reference gives the radial analyticity,
passive exponential envelope, gradient formula and exact rectangular tail.
Independent mathematical review found no P1/P2, including negative real wave
numbers with passive loss and one/both zero-radius degeneracies.

ElectromagneticM2L(error_budget=KernelErrorBudget(...)) checks this envelope
before coefficient-table generation. The existing fixed-order path remains
experimental. The sufficient analytic contour condition is stricter than
the geometric partition condition; missing contours explicitly raise
KernelBoundUnavailable. Native logs avoid premature exponential underflow;
positive underflow is raised to float64's smallest normal value. No native
floating-point enclosure, RWG/near quadrature bound or physical certificate
is implied. System-wide automatic order selection remains next.

/tmp/gsp09-cauchy-v1.log:5 passed,5.81s. For static, real and passive
negative-real Helmholtz waves, orders2/4/6 actual Green and gradient errors
lie below the analytical bounds, which decrease with order in these controls.
For k=6.3,p6, actual Green error 2.5716e-11 versus bound 1.7139e-5;
gradient error 1.3708e-10 versus bound 3.1973e-4. The conservative bounds
are not replaced with these much smaller sampled errors. Degenerate supports,
missing contours, small-scale decay cancellation and positive-underflow
handling pass, alongside the actual magnetic/lossy M2L weak control.

/tmp/gsp09-cauchy-admission-v1.log:1 passed,4.64s; an added sentinel
confirms failed truncation admission precedes M2L coefficient generation.
No full regression repeated. Mathematical audit is scoped to the derivation,
not a completed production/complexity audit.

## System-wide automatic Cartesian order

FMMConfig version 2 adds max_order and optional KernelErrorBudget. The
system builds its support trees once, then plans all common-exterior and
body-interior regions before numerical moment/shift/M2L construction. Every
candidate order streams the sphere partitions and checks every far kernel
bound using that region's wave number. Both absolute kernel components and
cumulative M2L table bytes must pass. The count is 4*binom(p+3,3)^2 complex
entries per far regional interaction. Actual construction verifies the total.

Unavailable analytic contours or unrepresentable bounds never pass. Later
orders may recover a finite bound; order exhaustion preserves per-candidate
failures. Since table count grows monotonically for fixed trees/partitions,
exceeding the table budget terminates higher candidates. This does not assume
monotone numerical error bounds. Plan receipts are immutable serialized
state; access returns a copy. Fixed-order optional budgets also propagate
into each regional M2L constructor.

This gate addresses ideal kernel truncation only. Explicit low moment
quadrature is still marked nonexact; no full RWG error qualification follows
from a kernel pass. Near/projection capacity is independently enforced, and
complete retained/peak/setup/solve/backward admission remains open.

/tmp/gsp09-order-v1.log:16 passed,6 deselected,18.76s. The two-body physical
setup selects p6 from p2..p10 for Green 1e-6 and gradient-component 1e-5
budgets: maxima 5.032365580169729e-7 and 9.498146497826256e-6, table bytes
903168 matching construction. All lower orders fail. Sentinels prove order
exhaustion and insufficient table bytes reject before moments. Nested budget
roundtrip and affected public FMM solve/exact restore also pass.

/tmp/gsp09-order-frequency-v1.log:1 passed,6.58s. Fixed geometry at
300MHz and 900MHz selects p7 and p10 respectively. A sphere ratio .45
passes geometric .5 but lacks a .4-or-smaller analytic contour; every
candidate records two unavailable bounds and rejects. No full regression
or independent audit repeated for this incremental planner integration.

## Ideal RWG multipole absolute admission

The kernel-to-RWG integral bound is shared with the previously derived
interpolation analysis: triangle second moments bound integral |f|; exact
divergence integrals and node sums of u^2, q^2, u*q bound balanced PMCHWT
Frobenius error without forming far edge matrices. Only the source kernel
envelope changes to the Cartesian Cauchy tail. fmm_rwg_error.py provides
regional/system diagnostics with explicit ideal multipole scope.

FMMConfig v3 adds optional rwg_absolute_budget, requiring automatic order
selection. The system computes geometry-only integral statistics before
moment construction. Planner region zero is the common exterior; remaining
regions must occupy disjoint body-interior coefficient spaces. Disjoint far
blocks combine by squared bounds; overlapping media combine by triangle
inequality in diagnostics. The system combines exterior plus the root-sum-
square of interiors. Both kernel and RWG gates must pass, as must table
capacity. Missing kernel bounds suppress the global RWG receipt and fail
admission, rather than reporting a partial sum as a complete bound.

Explicit moment quadrature below floor((p+4)/2) rejects before construction
when RWG admission is enabled. Default automatic quadrature satisfies that
exact-arithmetic polynomial condition. Native recurrence/integration/shift
rounding, original near quadrature, relative operator and solver/physical
certification are not included. These original requirements remain open.

/tmp/gsp09-rwg-admission-v1.log:16 passed,15.83s. Loose kernel budgets
(1,10) with RWG budget1e-6 select p6; all lower orders pass the kernel gate
but fail RWG. Planned and constructed diagnostic bound agree at
8.004234734708086e-7. Early low-quadrature/order failure sentinels, nested
configuration roundtrip and previous kernel-order/invalid-config controls
pass. No full regression or independent audit repeated for this increment.

/tmp/gsp09-rwg-receipt-v1.log:3 passed,6 deselected,14.94s. Missing
analytic contours fail RWG admission with no partial global bound; affected
public FMM configuration/solve/exact restore controls pass after v3 update.

## Complete action retained storage admission

FMMConfig v4 adds action_retained_budget_bytes. After selecting the expansion
order and before constructing numerical moments, a structural traversal counts
all action-owned retained tensors: exterior/interior trees, centers/radii,
leaf maps and moments, M2M/L2L shifts, M2L tables, packed exact near values
and indices/CSR, and body coefficient maps. Construction verifies the planned
total against distinct actual storage, counting aliases once. A failed budget
returns a capacity receipt before moment construction. Receipt access returns
a copy, and serialized capacity state participates in mutation guards.

This budget excludes shared provider/input geometry, Riesz and solver state,
setup/action/backward temporaries, allocator overhead and host metadata. It
is not peak-memory admission or production qualification.

/tmp/gsp09-capacity-v1.log: 15 passed, 6 deselected, 17.62s. The mixed-body
fixture uses exactly 455744 bytes, matching every category and existing
projection/table/near counters. A budget one byte below the plan rejects
before numerical moments. Public FMM solve/restore and configuration checks
pass with the v4 payload. No full regression or duplicate archive was run.

## Exhaustive implemented operator measurement

operator_error.measure_original_operator_error streams coefficient probes and
original weak tiles to measure every entry of the impedance-balanced operator
difference. Working arrays scale with coefficient count times edge_batch plus
reference tiles; no complete dense matrix is retained. This is an explicitly
quadratic validation procedure, not a fast construction or solve path.
Global and cross-body Frobenius ratios are reported separately so dominant
self interactions cannot hide inaccurate cross-body coupling. Empty cross-body
spaces do not pass a cross-body gate. Zero reference norm with nonzero error
returns no finite ratio and fails. Nonfinite measurements reject.

The reference is the implemented original quadrature at the configured order.
All coefficient coverage does not certify continuum physics, directed rounding,
spectral relative error, or peak memory. No production metadata is promoted.

/tmp/gsp09-original-error-v2.log: 1 passed, 12.93s. All 14400 coefficients
and 7200 cross-body coefficients agree with independent dense norm calculations
using uneven edge batches of 17. The p0 negative control has global relative
error 2.2867139621530937e-6 and cross-body error .5088467843421596;
both fail the unchanged 1e-8 threshold. The initial v1 test also passed; v2
retains double-precision impedance weights and adds the independent cross-body
norm comparison. No full regression repeated.

## Six-order coupling check and original quadrature sensitivity

/tmp/gsp09-original-order6-v1.log: 2 passed,1 deselected,22.66s. The
same two-body p6 operator compared with original quadrature order2 has global
relative error9.184328263201338e-11 (passes1e-8), but cross-body relative
error2.0437256169374217e-5 (FAIL). This failure is preserved as an explicit
negative control; global self-block dominance must not qualify coupling.
With original quadrature order4, global error2.0892847987320023e-16
and cross-body error2.198205817707684e-12 both pass1e-8. All coefficients
and the separate cross-body norm agree with independent dense calculations.

The p6 moments use automatic polynomial-exact quadrature order5 in both
cases. Thus increasing the reference quadrature resolves this fixture's
observed discrepancy without changing the multipole expansion. This is
evidence of reference-quadrature sensitivity, not a proof that quadrature
order4 suffices for other geometries or electrical sizes. Subsequent scaling
acceptance must retain reference-quadrature convergence evidence and separate
coupling metrics. No physical qualification follows from either algebraic
comparison. The existing p0 negative control was deselected, not rerun.

## Preregistered resource pilot

scripts/dev/measure_mom_fmm_scaling.py fixes two sphere families with global
edge counts60,240,960,3840 (subdivisions0..3), three independent-process
repetitions per point. Radius.01 and centers+/- .5 stay fixed; fixed-ka uses
300MHz, growing-ka doubles frequency at each refinement to preserve nominal
points per wavelength. Automatic p6..12 uses kernel budgets1e-6/1e-5,
separation.25, leaf30, complete retained budget2GiB and individual1GiB budgets.
Original quadrature4 and singular order3 remain explicit and unqualified.

The manifest records the plan, revision and Python source hash before cases.
Each terminal case records success or failure and is resumed without rerun;
changed plans/source identities reject reuse. Setup includes geometry/provider
construction. CUDA-synchronized wall time and allocator peak allocated/reserved
bytes are measured for setup, warmed forward/adjoint and cold first backward.
Fixed-current gradient and complex adjoint checks accompany each case. Small
first-repetition cases also measure every original operator coefficient.
Large cases explicitly lack complete reference accuracy checks.

This is a resource pilot, not full acceptance: host accounting, solve resources,
all-size reference convergence, H2/dense crossover and physical qualification
remain open. Full slopes must use every preregistered point and repetition,
including admission failures; no selected-tail regression may qualify it.

Initial driver smoke /tmp/gsp09-scaling-v1/case-00.json (before source-hash
hardening) measured60edges,p6,retained2012224bytes,setup peak68886528bytes,
setup3.54s,forward.0293s,adjoint.0493s,cold backward.313s. Cross-body
error2.1982e-12 and adjoint1.81e-16; fixed-current gradient agrees. This
smoke is retained as pilot-only evidence, not a full-series result.

## Independent audit corrections

Read-only increment audit found two P2 issues: direct squared Frobenius
accumulation could underflow, and a worker source change could be published
without an immediate identity check. In the separate gsp09-audit worktree,
scaled norms plus hypot accumulation now preserve extreme nonzero errors;
nonfinite ratios fail instead of passing. Workers recheck source identity at
completion, and publication compares identity/revision/source_hash/case.
The original gsp09-fmm worktree remains frozen while its pilot runs.

/tmp/gsp09-norm-extreme-v1.log:2passed,3deselected,3.82s, nonzero relative
error1 at scales1e-200 and1e200 fails the gate.
/tmp/gsp09-runner-identity-v1.log:5passed,1.48s, each mismatched identity
field rejects; an in-worker source change discards the measurement.
/tmp/gsp09-norm-ordinary-v1.log:1passed,4deselected,12.58s; ordinary
p6/order4 full and cross-body metrics still agree with independent dense norms.
Independent re-review closed both P2 findings without rerunning GPU checks.

Frozen pilot /tmp/gsp09-scaling-4c04b80 is on a shared GPU. Audit-fix
GPU checks may overlap cases03..05; timings are exploratory only, and must
not be used as controlled final performance evidence. No cases are discarded.
As of this checkpoint cases00..04 completed (60edges three repeats;240edges
first two repeats). Remaining cases continue in session26564.

## Complete-series resource summarization

scripts/dev/summarize_mom_fmm_scaling.py checks every declared case and
its source identity, coordinates and finite positive resource measurements.
Each family reports all observed points, repetitions, selected orders and
min/median/max resources. Log-log OLS uses every individual repetition at
every declared size only when the complete family is measured. Missing or
failed cases suppress fits; failures remain explicit records. No largest-
three-size or passing-subset fit is substituted. Available original-operator
accuracy receipts remain separate from resource fits. Qualification remains
false even when a full empirical slope is available.

/tmp/gsp09-scaling-summary-v1.log:4passed,1.35s. A synthetic N^2 sequence
recovers exponent2 across4sizes/12samples; missing and failed smallest cases
both suppress tail fits, and mixed source hashes reject. Live pilot summary
/tmp/gsp09-scaling-4c04b80-summary.json was generated without GPU work.
At generation6/24cases were measured, so neither family had a fitted slope.
Case06 (960edges) was confirmed live in the frozen worker; no restart.

## Batched native M2L construction

Helmholtz Taylor recurrence and polynomial convolution now accept arbitrary
leading displacement dimensions on CUDA with one shared wave number. M2L
gather/factor tables are formed once per batch. The table gate counts every
displacement before coefficient allocation. Scalar entry behavior is retained.
Regional construction groups admitted pairs within each bounded traversal
batch by medium, validates each pair's support and optional kernel budget,
then builds same-medium tables together. Pair/media action ordering remains
unchanged. Couplings retain views into shared batch storage; distinct-storage
capacity accounting counts the batch once and mutation guards remain active.
No unbounded all-interaction batch or persistent indexing cache is added.

Validation on GPU1 while the frozen pilot stays on GPU0:
/tmp/gsp09-batch-expansion-v1.log:8passed,3.54s; batch versus individual
static/passive tables, analytic Green/gradient centers, whole-batch early
capacity rejection and existing scalar convergence/adjoint/invalid controls.
/tmp/gsp09-batch-regional-v1.log:4passed,5.58s; regional rectangular/
global actions, full heterogeneous system/current adjoint and retained bytes.
/tmp/gsp09-batch-storage-v1.log:1passed,3.47s; shared storage identity
and invalidation controls plus existing capacity checks.
/tmp/gsp09-batch-timing-v1.log:16pairs,p6,three repeats: individual
.3875/.3904/.3899s versus batched.02422/.02472/.02431s, approximately
15.8–16.0x for table construction, each table numerically compared. This
local measurement does not qualify complete setup/solve complexity.

Frozen pre-batch resource pilot continues unchanged at4c04b80. Case06
(960edges) completed setup in340.33s; its results describe the old implementation
and must not be relabeled as evidence for batched construction.

## Periodic remote-image M2L primitive

fmm_periodic.PeriodicImageM2L forms actual Cartesian multipole/local
coefficients for a finite absorbing 1D/2D/3D lattice image cube. Each image
uses displacement R-A*n and weight exp(+i beta.A*n), matching LatticeSpec.
Complete-support sphere separation partitions images into remote M2L tables
and an explicit exact-near index list. The caller owns those near evaluations
once; the remote sum never includes them. Same-medium image tables are
batched and reduced into one retained Green/gradient M2L pair. Retained plus
returned batch-table capacity and near-index storage have separate gates.
Recurrence/reduction workspace and allocator overhead remain outside the
table gate; there is no complete peak-memory claim. State/lattice mutations
reject. Public periodic FMM remains unavailable.

This is the remote-image expansion foundation, not an infinite-periodic
solution: positive absorption is required, outer image truncation is not
certified, and local expansion error/physical qualification remain false.
Infinite-image tails, lossless/Wood handling, exact periodic near integration,
regional/system/runtime composition and original physical gates remain open.

/tmp/gsp09-periodic-m2l-v1.log:4passed,3.78s on GPU1. Nonzero Bloch
phase and skew1D/2D/3D lattices match independent finite-image Green and
gradient sums. Remote-only comparisons prevent dominant exact-near terms
from hiding errors. Cubes3/9/27images split into2/8/26remote plus one
exact-near image with no duplicate ownership. Mutation, lossless rejection
and early insufficient-table-budget controls pass. No full regression run.

## Absorbing infinite-image tail admission

fmm_periodic_error.py bounds all omitted integer cube shells uniformly over
complete target/source support spheres. Smallest lattice singular value sigma
and extentB=|R|+a+b give first omitted distance r0=sigma*(S+1)-B. The
1D/2D/3D shell counts2,8m,24m^2+2 yield closed geometric-series bounds
for Green; multiply by(|k|+1/r0) for each gradient component. Positive
absorption and r0>0 are required. Log-domain CUDA float64 evaluation and a
positive underflow floor avoid false zero tails. Proof is maintained in
docs/reference/mom_absorbing_lattice_image_tail.md. Independent mathematical
review found no P1/P2, without rerunning GPU checks.

PeriodicImageM2L(image_tail_budget=KernelErrorBudget(...)) rejects insufficient
shells before coefficient construction and binds copied tail receipts into
operator state. The budget controls only ideal omitted-image error. Retained
M2L truncation, floating-point, near quadrature, lossless and full physical
qualification remain open. No public periodic backend is enabled.

/tmp/gsp09-periodic-tail-v1.log:4passed,3.31s onGPU1. Skew1D/2D/3D
omitted finite increments lie below the analytic infinite bound; bounds
decrease forS1/2/4, unavailable support separation rejects, strong decay
stays positive, tight budget rejects before M2L. The finite increment
comparison supports implementation, not proof of the infinite sum.
/tmp/gsp09-periodic-tail-admission-v1.log:1passed,4deselected,3.03s;
successful admission, receipt copy isolation and tail-state mutation guard.

## Combined periodic ideal kernel admission

PeriodicImageM2L now accepts kernel_error_budget in addition to the separate
optional tail budget. A shared streamed image partition is used by error
planning and table construction. Each retained far image contributes its
Cartesian Cauchy truncation bound; real Bloch phase has modulus one, so
triangle addition combines these with the complete absorbing outer-image
tail. Missing analytic contours or nonfinite totals reject, never report a
partial sum. The combined gate runs before any M2L coefficient table.
Construction verifies planned near/far image counts; serialized receipts and
budgets participate in mutation guards and return copies. Exact-near
quadrature, floating-point, RWG/system/physical qualification remain open.

/tmp/gsp09-periodic-combined-v1.log:4passed,6deselected,3.92s onGPU1.
An order2 case has an admitted outer tail but fails the combined kernel
budget before table construction. Order6,shells5 passes1e-6: total Green
1.9758364163383674e-9, gradient2.3856045536750104e-8, including
outer-tail gradient1.708897255673313e-9. Ten remote and one exact-near
image match the plan. Receipt copy isolation and shared-partition1D/2D/3D
finite-image/remote-only comparisons pass. No full regression repeated.

## Absorbing periodic body-pair RWG action

fmm_periodic_regional.PeriodicMultipoleRegionalOperator composes root-level
remote-image M2L with existing hierarchical free-space regions for each
remaining image. It accepts body-local nested RWG spaces and one exterior
medium. Bodies must lie strictly inside the centered periodic cell. The root
remote sum never overlaps near-image regions. Nonzero near images translate
the source geometry and tree while preserving canonical RWG edge order, then
reuse nested moments, sphere traversal, M2L and original exact leaf quadrature.
No all-edge dense matrix is constructed by this path.

Forward/adjoint/current-gradient actions include the conjugated Bloch phase
on adjoint near-image contributions. Medium, projections, phase tensors,
image expansion and regional state remain guarded. Table/near retention and
additional translated projections accumulate budgets. Input projections,
translated geometry/tree storage and transient costs remain excluded; full
action/peak resource admission is not claimed. The optional remote kernel
budget covers the root remote expansion and outer tail only, excluding
near-image hierarchical truncation. This is not a periodic system or public
solve adapter; interiors must remain free-space and are not included here.

/tmp/gsp09-periodic-regional-v1.log:2passed,4.15s onGPU1. Same-body
primary near image and a cross-seam nonzero near image both match independently
integrated translated weak blocks. Balanced full errors are respectively
1.143212722948022e-16 and1.1075486512561765e-14. Complex adjoints,
current gradients and phase mutation rejection pass. Each fixture has two
root remote images and one hierarchical near image; nonzero translation
retains161552projection bytes. No full regression repeated.

## Shared global periodic exterior

PeriodicMultipoleRegionalOperator now accepts one global ScalarSystemBlocks
projection, so every body shares one original tree and one root remote-image
expansion. Each nonzero near image translates the full source system, preserving
Scene body/edge order and original interior metadata without adding interior
terms. Global exterior block routing now supports distinct test/source providers
with identical exterior medium, integration, device and precision. Regional
FMM reuses that routing for exact selected near blocks across translated
spaces. No per-body-pair operator construction is introduced; touched bodies
are routed only when integrating selected blocks. Interior system composition
and complete body-count/setup complexity remain open.

The wide two-body root exposes fixed-order truncation failures, all preserved:
/tmp/gsp09-periodic-global-v1.log:1failed,4passed,5.76s; p6,separation.5
global relative error1.8456576873174714e-6 fails1e-8.
/tmp/gsp09-periodic-global-order10-v1.log:1failed,2deselected,6.38s;
p10 global5.619719494659071e-8 still fails.
/tmp/gsp09-periodic-global-order14-v1.log:1failed,2deselected,16.81s;
p14 global4.313007112965728e-10 passes, cross-body1.0070219543177844e-6
FAILS. This configuration remains an explicit negative control.
/tmp/gsp09-periodic-global-separated-v1.log:1failed,3deselected,4.81s;
p6,separation.2 global3.009758066394537e-8/cross6.148677407923422e-5
FAIL. Some retained near-image internal-node far blocks still truncate poorly.
/tmp/gsp09-periodic-global-separated-v2.log:1passed,3deselected,5.05s;
p6,separation.1 on the SAME geometry gives global4.844494513527342e-17
and cross8.670266592035957e-14, both below unchanged1e-8; adjoint matches.
All tests run onGPU1; frozen resource pilot remains untouched onGPU0.

Stricter geometric separation descends these large support clusters into
smaller interactions. This is configuration-specific evidence, not automatic
error qualification: the near-image hierarchy still requires a complete
combined error budget before production admission. Public periodic backend,
lossless/physical/system/resource gates remain open.

## Periodic full-system composition

PeriodicMultipoleSystemActions builds one shared global periodic exterior and
one disjoint free-space regional action per body interior. All currents retain
global all-J/all-M ordering. The existing guarded fixed-operator action/backward
protocol is reused, avoiding saved forward action intermediates. Projection,
M2L and exact-near retained budgets accumulate over the exterior and interiors.
These budgets exclude trees/maps, shared and translated provider geometry,
Riesz/solver state and transient/allocator/host costs; complete capacity remains
open. No public periodic solve adapter or global error certificate is implied.

/tmp/gsp09-periodic-system-v1.log:1passed,4.81s onGPU1. A heterogeneous
magnetic two-body system matches independent original free-space system
assembly plus only translated exterior weak contributions. Global balanced
error4.93191698369108e-16, cross-body5.500744342496366e-11, both below
unchanged1e-8. Interior medium identity, global coefficient routing, complex
adjoint, fixed-current gradient, zero saved forward tensors and stale-index
rejection pass. Independent read-only review of system/periodic regions/global
selected blocks found noP1/P2; no GPU checks repeated by the reviewer.

Frozen resource pilot update: cases09..11 (fixed-ka,3840edges,all3repeats)
FAIL before construction: order6 plans6738far pairs and3042772992table
bytes versus the preregistered1073741824byte budget. Its kernel maxima
13.812701969109515 and14033.392795244636 also fail. No budget or
threshold was increased to make these points pass. Therefore fixed-ka full
complexity qualification fails; no subset fit substitutes for the failed size.
Cases12..18 progressed through growing-ka960edges first repetition; original
process remains live. These results concern frozen pre-batch4c04b80 only.

## Original periodic residuals and internal Krylov adapter

PeriodicImageSystemBlocks extends the original selected-block provider with
a finite absorbing image cube. The primary block contains original exterior
plus free-space interiors; nonzero images contribute exterior-only translated
blocks. Source geometry streams one image at a time. Inherited exterior-only
methods remain free-space primitives for hierarchical image construction, so
the periodic action does not accidentally periodize an already periodic block.
Translation of full systems is shared with periodic regional construction.

PeriodicMultipoleOperator implements the existing direct Riesz/Krylov protocol
and requires this original periodic provider. Sampled residual admission thus
uses configured periodic images, never the free-space operator. It preserves
inner failure, no dense fallback, RWG diagonal scaling, state guards and
explicit finite-image scope. Omitted infinite images, physical incident lifting,
public runtime/storage and complete error/peak-resource gates remain open.

/tmp/gsp09-periodic-krylov-v1.log:1passed,5.68s onGPU1. An algebraic
two-RHS block-GMRES solve agrees with independent original primary assembly
plus translated exterior blocks. Every original row is checked in the fixture.
True periodic residual1.7235862914340423e-14 passes1e-8; using free-space
instead would give3.319448987042939e-5 and fail. Uneven source-batch
sampling matches the independent matrix. A free-space-only provider rejects,
maxiter1 remains a failed solve without fallback, and shell mutation rejects.
This known-current RHS is not a physical periodic incidence test.

## Compact coefficient M2L storage foundation

CompactElectromagneticM2L retains Taylor coefficients through2p+1, the
Cartesian multi-index list and a small binomial table. A closed total-degree
index formula gathers each target-row batch's Green/gradient tables on CUDA.
Its electromagnetic algebra is identical to dense M2L; forward concatenates
target chunks and adjoint accumulates their source-space contributions.
Retained storage is O(p^3), while arithmetic remains O(p^6); generated
row-table storage is bounded by row_batch. Index/weight/reduction workspace
and allocator overhead remain outside that table-size statement.

Fixed-operator adjoint backward regenerates tables without saving them from
forward. Construction admits exact retained bytes before Taylor generation;
coefficient and generated-table nonfinite values reject. Geometry/material
gradients, truncation certification and full peak resource qualification remain
open. This is a primitive, not yet the default regional/system representation.
The frozen pilot's3.04GB table-budget failure remains unchanged.

/tmp/gsp09-compact-m2l-v1.log:2passed,1failed,3.73s. Numerical dense
forward/adjoint/current-gradient controls passed forp0/3/6; thep3failure was
an incorrect test assumption that every order saves at least10x. Actual
p3 storage2656vs25600bytes saves9.64x; p6 storage11760vs451584 saves
38.4x. p0 has104vs64bytes, so small-order overhead is explicitly retained.
/tmp/gsp09-compact-m2l-v2.log:1passed,2deselected,3.42s; corrected
the test to require a storage saving, then verified p3 early capacity and
mutation guards. No original production tolerance was changed.
/tmp/gsp09-compact-overflow-v1.log:1failed because an extreme-distance
fixture already rejected during Taylor construction; it never reached the
intended generated-table guard. v2 uses explicitly injected finite coefficients
to isolate weight overflow:1passed,3deselected,2.69s. All checks onGPU1.

## Compact storage public integration

FMMConfig version 5 explicitly selects dense or compact M2L and binds the
row batch into numerical identity and exact persistence. Regional/system
construction, order planning and retained capacity share one byte formula,
including coefficient, index and binomial arrays. Dense remains the default.
Representation changes do not weaken kernel or RWG error admission.

GPU1 targeted evidence: /tmp/gsp09-compact-backend-v1.log:4passed,
13deselected,10.47s covers compact capacity, heterogeneous system actions,
adjoint/current gradients, automatic order and public solve/exact restoration.
The order fixture still chooses p6, with identical kernel bounds and two far
tables using 23520 rather than 903168 bytes. Planned and actual full-action
storage match (p3 fixture409856bytes). Config validation and strict restoration:
/tmp/gsp09-compact-config-v1.log:14passed,2.85s. Earlier dense/config control:
/tmp/gsp09-compact-integration-v1.log:13passed,3.56s. Independent integration
audit found no P1/P2 within this scope; it did not run additional GPU tests.

The frozen 4c04b80 pilot is terminal:24cases,18measured,6failed,0missing.
Both 3840-edge families fail all three repetitions; compact integration does
not relabel those results or repair the separate kernel-bound failures. No
large rerun, full regression or duplicate archival was performed here. Periodic
compact storage, full peak resources and physical/complexity gates remain open.

## Compact periodic M2L integration

Remote images accumulate exp(+i beta.A*n)-weighted Taylor coefficients
through degree2p+1 before target/source binomial expansion. The shared compact
action kernel generates bounded target-row tables; conjugate-transpose action
therefore preserves complex Bloch weights. Periodic regional/system selection
propagates to hierarchical near images and nonperiodic body interiors.
The remote setup gate includes retained coefficients/indices/binomials plus
the returned coefficient batch. Recurrence, reduction, generated action tables,
other workspace and allocator overhead are explicitly outside this gate.
Cumulative regional/system retained-table accounting uses actual arrays.

GPU1 /tmp/gsp09-compact-periodic-v1.log:13passed,7.84s. This covers
original compact primitive checks, existing finite-image controls, new skew
1D/2D/3D nonzero-Bloch dense/compact forward and adjoint comparisons, budget
rejection and coefficient mutation, and full periodic system dense/compact
comparison with independently assembled image blocks and fixed current
backward. p6 remote storage11760bytes versus dense451584bytes. Public
periodic integration, complete infinite-image/near-hierarchy error admission,
lossless support, physical and peak-resource qualification remain open.

Compact periodic Krylov integration: /tmp/gsp09-compact-periodic-krylov-v1.log
1passed,1deselected,5.52s. Algebraic known-current solve retains original
periodic residual1.7235862914340423e-14; the deliberately wrong free-space
reference gives3.319448987042939e-5. The test also preserves early solver
failure without fallback and provider mutation rejection. No physical
incidence or public periodic qualification is inferred from this fixture.

Independent read-only phase audit found no P1/P2 in this integration. It
checked Bloch linearity/adjoint conjugation, root/near/interior configuration
propagation, retained accounting and state guards, and read both test logs
without rerunning GPU checks. Peak memory and complete periodic error/
physical qualification remain outside the proved scope.

## Periodic exterior hierarchy kernel admission

periodic_hierarchy_error_bound follows the same root image classification and
translated sphere-tree partition as construction. Within a near image, each
edge-pair belongs to one disjoint block, so the largest far-block kernel bound
covers all pairs (exact leaves have zero ideal truncation error). The triangle
sum across images uses real-Bloch phase modulus1, then adds the root remote
expansion and infinite absorbing tail bound. Unavailable bounds reject; no
partial sum admits. Regional hierarchy_kernel_error_budget is checked before
M2L allocation; system exterior_kernel_error_budget explicitly forwards this
exterior-only contract. Immutable serialized receipts and budgets are guarded.

GPU1 /tmp/gsp09-periodic-hierarchy-budget-v1.log:1passed,4deselected,
3.76s. The fixture has7nearimages,22farblocks,2exactleaves. Root-only bounds
1.420458384785565e-24/3.143403988903174e-23 miss hierarchy terms
0.004677995311189832/0.18741214772936787. A budget between these
values rejects before remote table construction. Explicit looser test-only
budgets check admission, exact planned/constructed far counts, receipt copies
and mutation guards; no production threshold is altered. v2 adds independent
pointwise Green/gradient comparisons for every constructed near-image far
block:1passed,4deselected,3.93s. No broad regression was repeated.

Independent read-only audit found no P1/P2 in partition/sum reasoning,
construction arithmetic, admission order and state protection; it reviewed
v1 evidence without GPU execution. This is ideal exterior kernel truncation
only. Original near quadrature, RWG projection, native floating point, body
interior errors and implemented relative error/physical qualification remain
open. Root radius addition uses individually converted scalars, matching
construction even for float32 inputs.

## Exhaustive periodic implemented-operator error

PeriodicMultipoleSystemActions previously retained a supplied free-space
provider even though its action included periodic images. Generic original
operator diagnostics could therefore compare against the wrong reference.
The constructor now binds PeriodicImageSystemBlocks with the same geometry,
material, integration, lattice and shells. An already periodic provider must
match lattice identity and shell count. Inherited exterior-only methods stay
free-space, preventing double periodization during hierarchy construction;
the provider's full selected blocks include the declared image cube.

GPU1 /tmp/gsp09-periodic-exhaustive-v1.log:2passed,5deselected,11.94s.
Uneven edge batches measure every14400global/7200cross-body coefficients
and agree with independently assembled primary and translated exterior blocks.
p0 remains measuredFAIL:global4.96218046236076e-6,
cross0.5903827707806611. p6 passes the unchanged1e-8 threshold:
global5.230731069808823e-16,cross5.5007435411017295e-11.
Shell mismatch and post-construction mutation reject. These are compact
representation, positive-absorption, two-body scalar magnetic fixtures.

The full measurement remains quadratic diagnostic work with bounded probes,
not a scalable error certificate or complexity result. Reference quadrature,
finite image cube and native floating point remain explicit limits. These
results do not close infinite-periodic physical accuracy, near-quadrature
convergence, full resources, or any preserved production failures.

Affected dense/compact system and Krylov controls:
/tmp/gsp09-periodic-reference-binding-v1.log:4passed,12.21s. Independent
read-only audit found no P1/P2 in reference identity, free-space exterior
primitives, mutation guards or declared storage scope; read exhaustive logs
without GPU execution. The replacement provider owns geometry caches, still
explicitly excluded from the existing retained-action budget claims.

## Periodic retained capacity including translated geometry

Periodic system construction now builds only support trees before planning
all retained action arrays. The plan counts original provider caches (even
when shared), global/body trees, translated lower/upper boxes with topology
aliases counted once, polynomial moments/shifts, root and regional M2L,
packed near matrices/routing, body maps, image phases/indices/displacement,
and every retained translated vertex/basis/system/prepared-geometry array.
Translated mesh faces alias input faces and are excluded with input meshes.
The translated geometry formula uses actual vertex/edge/face/body counts;
no translated meshes or numerical expansions are created to obtain it.

An explicit action_retained_budget_bytes gate rejects before moments,
translated geometry, near blocks and M2L allocation. Support trees and
original provider caches already exist at that point. After construction,
actual distinct CUDA storage must exactly equal the structural plan.
Serialized plan metadata participates in state guards and returned copies
cannot alter it. Input meshes/system maps, shared lattice, Riesz/solver
state, transient setup/action/backward, allocator and host metadata remain
outside this retained scope; full peak qualification remains open.

GPU1 /tmp/gsp09-periodic-capacity-v1.log:2passed7.07s affected full
system dense/compact checks. New focused cases:
/tmp/gsp09-periodic-capacity-v2.log4passed5.38s. p3 two-body1D:
dense895920bytes,compact643536bytes;2Dcompact1321128bytes including
8 translated geometries. Root-remote1D fixture529552bytes has no translated
geometry. All exact planned/actual/category sums agree; budgets one byte
below actual reject before numerical moments, receipts return copies, and
plan mutation rejects. These are retained-storage controls, not scalability
or physical qualification; no large scaling rerun or full regression.

3D follow-up /tmp/gsp09-periodic-capacity-3d-v1.log:1passed,
4deselected,5.10s:27nearimages/26translated geometries, planned and
actual3348736bytes, including310752bytes translated geometry. Independent
read-only audit found no P1/P2 in retained scope, alias counting, geometry
formula or admission placement; reviewed initial4cases without GPU reruns.

## Fixed-cube periodic automatic order admission

max_order enables an explicit uniform order search starting at order. Both
exterior_kernel_error_budget and interior_kernel_error_budget are required.
Each candidate checks the full exterior hierarchy plus infinite absorbing
tail, each free-space interior kernel, optional separate root-remote budget,
and action/table/near/projection retained budgets. Root setup must additionally
fit its retained representation plus at least one returned image table or
coefficient batch. Fixed partitions make storage nondecreasing with order,
so storage failures stop immediately; error failures preserve their receipts
and continue through the configured range. No budget or shell count changes.

The selected order and capacity plan are bound to system state and actual
construction rechecks error gates and distinct-storage equality. Returned
plans are copies. Shell-count adaptation and public periodic configuration
remain separate work; this primitive keeps original reference blocks bound
to the exact declared image cube. Kernel admission does not imply RWG,
quadrature, floating-point, implemented relative operator or peak guarantees.

GPU1 /tmp/gsp09-periodic-order-v1.log:1failed4.17s: the initial test
incorrectly expected success within orders2..10. The solver correctly
rejected; no numerical expansion occurred. Diagnostic
/tmp/gsp09-periodic-order-diagnostic-v1.log evaluated through20 without
constructing numerical actions and found first admitted order11. At order10,
Green1.811081358417491e-7 passes1e-6 but gradient1.1109470994466347e-5
fails1e-5. Order11 gives4.432705889180483e-8/2.7238196808994383e-6
and planned retained42391408bytes. The revised test permits up to12 while
keeping exactly the same budgets, geometry, material, separation and shells.
/tmp/gsp09-periodic-order-v2.log:1passed9.24s, constructs order11 and
verifies order10 remains rejected before moments, resource failure stops at
the first candidate, reference shells stay3, and plan copy/mutation guards.
The coarse body-interior trees have no far blocks in this fixture; this is
not independent physical validation of interior approximation.

Independent read-only audit found no P1/P2 in candidate gates, root setup
minimum, resource monotonicity, reference binding or plan guards. It ran no
GPU checks. Original failed resource/physical sequences remain unchanged.

Root setup boundary follow-up:
/tmp/gsp09-periodic-order-root-budget-v1.log:1passed,1deselected,
3.15s. A single-body compact candidate fits all retained M2L and global
capacity, and satisfies kernel budgets, but cannot hold the additional
returned image coefficients; planning rejects before moments as required.

## Joint image-shell and order selection with matching residual reference

max_shells explicitly enables a bounded ascending search from shells; it
requires max_order and the existing exterior/interior budgets. Each shell
candidate runs the complete fixed-cube planner. The first admitted shell
cube is selected, then its lowest admitted order. Failed cubes and all their
order attempts remain in the plan. A higher-order storage failure at one
shell does not terminate the shell search: a larger cube can reduce tail
error enough to make a lower order viable. No error/resource budget changes.

If selection changes shells, system construction creates matching original
PeriodicImageSystemBlocks before numerical moments. The shared direct
Krylov adapter now binds actions.provider, enforcing identity of input system,
exterior medium and integration config. Thus action and original residual
use the same selected cube while metric/geometry stay consistent. The input
provider's original shell count is not mutated. Existing refresh guards bind
the selected reference and reject subsequent changes.

GPU1 /tmp/gsp09-periodic-shells-v1.log:1passed6.40s. Positive-absorption
single magnetic body, nonzero Bloch: shell search1..12 and order3..6
select shells4/order6 under unchanged1e-6/1e-5 kernel budgets. Smaller
cubes remain rejected. Algebraic known-current block-GMRES gives selected
original-cube residual2.350879827599225e-15; old and selected original
matrices differ. A range ending at shells3 rejects before numerical moments,
and selected-provider mutation rejects. This is a finite-cube algebraic
fixture, not physical incidence or infinite/lossless-periodic validation.

Affected shared direct/fixed-periodic solve checks:
/tmp/gsp09-selected-reference-regression-v1.log4passed18.20s.
Independent read-only audit found no P1/P2 in shell search ordering,
continuation after per-shell resource failures, selected-reference binding
or guards; reviewed new shell-selection evidence without GPU execution.
No unrelated full regression or failed large resource run was repeated.

## Lossless Ewald reciprocal Taylor foundation

Public MoM validates a lossless exterior; its periodic RHS/field/persistence
contract uses EwaldHelmholtzKernel. The absorbing image FMM cannot be
connected to that public contract directly. Work therefore proceeds through
lossless Ewald translation primitives instead of relaxing background rules
or labeling finite absorbing images as the public periodic solver.

The new ewald_spectral_taylor computes one center's total-degree Cartesian
coefficients for the configured reciprocal contribution. In3D each Fourier
mode contributes exp(iq.R) amplitude times (iq)^alpha/alpha!. In1D/2D the
existing outgoing radial/arc heat contour has dependence exp(L.h+h.Q.h).
Differentiation gives alpha_i c_alpha = L_i c_(alpha-e_i) +
2 sum_j Q_ij c_(alpha-e_i-e_j). Native batched CUDA recurrence uses the
actual open-space projection matrix, including mixed Cartesian terms for
skew lattices. Existing spectral pole search and radiation branch choices
are preserved; no finite differences, dense operator sampling or CPU
numerical fallback is used.

Coefficient storage admission bounds output plus one returned batched
coefficient array; open modes include all contour quadrature nodes in this
minimum. Polynomial recurrence, power/gather, contour and reduction workspace
are excluded, so this is not a complete peak budget. Nonfinite polynomial
coefficients, sums and factorial weights reject.

GPU1 /tmp/gsp09-ewald-spectral-taylor-v1.log6passed3.83s:skew
1D/2D/3D, lossless k=+2/-2, total-degree8 values and gradients at small
offsets agree with independent configured Ewald reciprocal evaluation
(rtol2e-10,atol2e-12). Insufficient coefficient budget rejects. These tests
cover the reciprocal component, including its native contour quadrature;
they do not qualify complete Ewald convergence, real-space translation,
full periodic FMM, physical fields or complexity. Public runtime unchanged.

Independent read-only audit found one P2: the3D branch retained the previous
returned coefficient batch while forming the next, contradicting the
output-plus-one-batch scope. Fixed by releasing each batch immediately after
aggregation. Re-review closed P2 and found no further P1/P2; mathematical
recurrence/projector/Fourier/outgoing-branch review was clean. Weight tables
are now built once outside the3D loop with finite guards. No GPU audit rerun.
/tmp/gsp09-ewald-spectral-budget-v1.log1passed,6deselected,3.87s:
new single-mode-budget coefficients match normal batching. This is numerical
batch consistency, not a complete allocator peak measurement.

## Screened real-space and complete regular-origin Taylor coefficients

For a nonzero center R, screened real-space coefficients use the radial
inhomogeneous Helmholtz equation in s=|R+h|^2/|R|^2. The Gaussian source
is positive S=eta^3 exp(k^2/(4eta^2))/pi^(3/2); seeds are the configured
real-kernel value and half grad(G).R. Native recurrence generates the
s-1 coefficients, then the same Cartesian radial composition as free-space
Helmholtz. The existing free-space composition was extracted without changing
its polynomial arithmetic. Returned coefficients have explicit storage
admission before radial seeds; workspaces/recurrence/allocator are excluded.

At the primary origin, the smooth G_real-G_free remainder uses the existing
entire r^2 series directly: even Cartesian indices receive its multinomial
coefficients; odd indices vanish. This avoids singular subtraction. The
original Ewald regular evaluator and new Taylor path share the extracted
radial coefficient generator. The complete ewald_regular_taylor adds this
primary remainder, all configured nonzero real-image coefficients about
-A*n weighted by exp(+i beta.A*n), and the reciprocal coefficients. Each
returned image batch is freed after aggregation. No free singularity enters
the returned origin coefficients.

GPU1 /tmp/gsp09-ewald-real-taylor-v1.log3passed3.83s: k=+2/-2 and
2+.7j, batched nonzero centers, degree8 values/gradients versus original
screened real kernel and primary regular evaluator. Budget and zero-radius
rejections checked. /tmp/gsp09-ewald-regular-taylor-v1.log3passed,
3deselected,3.91s: complete lossless1D/2D/3D skew-lattice nonzero-Bloch
regular-origin values/gradients match original regular_part at local offsets.
/tmp/gsp09-ewald-radial-shared-v1.log11passed,26deselected,4.87s:
affected original free-space expansions and periodic regular-origin controls.

Independent read-only mathematical/code audit found no P1/P2 in source sign,
radial recurrence, primary multinomials, Bloch phase, shared extraction or
returned-batch accounting. It did not run GPU. These are configured-kernel
local expansion controls, not high-order stability or support-radius error
certificates. Nonzero image singularities still limit the regular-origin
series; complete periodic M2L must subtract/admit near images appropriately.
Public periodic FMM and all original production gates remain open.

## Ewald root M2L and explicit near-image ownership

EwaldRootM2L uses coincident global root centers and the configured Ewald
regular-origin coefficients through2p+1. The primary free-space singularity
is already absent. All nonzero near images subtract their free-space Taylor
coefficients with positive Bloch phase; near_indices includes primary and
nonzero near images for the caller to evaluate each exactly once. Compact
electromagnetic table generation and fixed-operator adjoint backward reuse
the existing common algebra.

Near classification is radius_sum > separation_ratio*|A*n|, with primary
always near. sigma_min(A)*(real_shells+1) > radius_sum/separation_ratio
conservatively ensures no possibly near image lies outside the real cube;
otherwise construction rejects. This removes proximity singularities from
the represented correction, but does not certify Taylor truncation or native
Ewald cutoff convergence. Owned coefficients, indices, binomials and near
indices are charged and guarded; kernel/medium parameters are bound.

GPU1 /tmp/gsp09-ewald-root-m2l-v1.log4passed4.98s:lossless skew
1D/2D/3D, plus a1Dcase with nonzero near images. Small-offset Green/
gradient controls match original Ewald regular_part minus near free images.
Electromagnetic adjoint/current gradients pass with no generated forward
tables saved; insufficient retained budget, insufficient real cutoff and
near-index mutation reject.

Wider-support diagnostic /tmp/gsp09-ewald-root-support-v1.log keeps
test/source radii0.2m and points reaching0.18m, with3nearimages. The
unchanged1e-8 relative gate gives measuredFAIL atp6:
Green3.484947318375914e-8,gradient2.563950241763138e-7. Atp10:
Green1.1395457566713632e-12,gradient1.980774517509812e-11 pass.
No geometry/tolerance change or removal of the lower-order failure. This
point fixture is not whole-support, weak-operator or physical qualification.

Independent read-only audit found no P1/P2 in image accounting, cutoff
coverage, compact action, budgets or guards. It ran no GPU checks. Full
periodic hierarchy integration, automatic implemented error admission,
cutoff/roundoff and public runtime remain open; original Goal unchanged.

## Full Ewald hierarchy, original selected blocks and internal Krylov

EwaldSystemBlocks adds smooth original Ewald exterior blocks to primary
free-space PMCHWT, streaming selected support-face pairs under pair_batch.
Arbitrary edge order and global J/M routing are preserved. Inherited
exterior_block/exterior_cross_block remain free-space primitives, so near
images are neither double-periodized nor assigned interior contributions.
The reference uses the same configured Ewald regular quadrature as dense
assembly, with no global matrix retained by the provider.

Periodic regional composition accepts an explicit Ewald kernel with identical
root projections and lattice. Ewald root correction is always applied, even
when no separated free-space images would be counted as far. Near-image
regions reuse the shared translated hierarchy. Complete systems bind an
EwaldSystemBlocks reference, add each body interior once, and reuse the
existing fixed-adjoint/current-backward and Krylov/Riesz protocols. Retained
capacity accounts for compact Ewald root arrays and near indices without
charging a nonexistent root displacement. Ewald metadata identifies the
actual kernel and configured cutoff reference.

Automatic order/shell selection and absorbing-image error budgets cannot
certify this Ewald path; explicit requests reject until its own complete
error plan exists. Fixed-order internal actions are experimental and public
periodic Scene/Result integration remains disabled.

GPU1 /tmp/gsp09-ewald-regional-v1.log1passed23.59s:two magnetic
bodies, lossless nonzero-Bloch1D Ewald, selected reordered/strided reference
blocks match independent dense assembly. Shared p8 exterior gives global
1.849334736521575e-15/cross5.293397379862363e-12 at1e-8.
Adjoints, current gradients and near-index mutation checks pass. The strided
selection triggers an informational native bucketize contiguity warning.
/tmp/gsp09-ewald-system-krylov-v1.log1passed11.41s:complete p6
single-body system gives original independent dense residual
7.645182170238818e-13, exact planned/actual retained455904bytes;
no forward tables saved for current backward, unsupported error admission
and lattice mutation reject. Algebraic RHS only, not physical incidence.

Independent read-only audit found no P1/P2 in selected indexing, single
exterior/interior accounting, reference identity, root activation, capacity
or guards; reviewed full system logs without GPU execution. Ewald cutoff/
roundoff/implemented global error certification and physical gates remain
open; existing failed tests/pilots remain unchanged.

Affected finite-image dense/compact controls:
/tmp/gsp09-ewald-integration-finite-control-v1.log2passed7.13s.
No full regression or previously failed large resource sequence rerun.

### Public Ewald FMM integration

The public scalar periodic configuration can now select FMM. Construction uses
EwaldSystemBlocks and the internal shared periodic exterior/free-space interior
actions. Local analytic incident carriers remain free-space; periodic field
queries and observables retain the declared Ewald kernel. Sampled operator and
RHS identities include the actual backend. Schema 12 stores both periodic and
FMM configuration and restores a decode-only handle without matrix/FMM assembly.
Automatic Ewald order and kernel/RWG error-budget admission are explicitly rejected.

Targeted evidence: `/tmp/gsp09-public-ewald-fmm-v4.log`, one CUDA test passed in
10.11 s. The lossless 1D radius-0.01 m scalar magnetic sphere at 90 MHz, order 6,
compact tables and separation 0.25 matches original dense configured Ewald
currents with relative difference 4.201934334412128e-12. Exterior fields, cell
translation, exact persisted currents/fields, schema/backend/configuration
corruption and live kernel replacement checks passed. This is a small algebraic
and persistence control, not cutoff or physical qualification.

Preserved development failures: v1/v2 used nonexistent test-side stopping-rule
exports; v3 reached the solve but its default stopping settings did not satisfy
FMM convergence admission. v4 uses the established strict FMM test settings
(rtol 1e-12, atol 0, maxiter 240, restart 120), retaining the original 1e-8
FMM residual admission and all comparison thresholds. No production capability
is inferred from this test. Independent read-only integration audit found no
P1/P2; complete Ewald truncation, physical qualification and final regression
remain open.

Affected-contract controls: `/tmp/gsp09-public-ewald-controls-v1.log`, 21 passed
in 8.83 s, covering strict FMM configuration/error-admission rejection, unsupported
hierarchical combinations, existing free-space compact FMM roundtrip and existing
periodic dense roundtrip. No full regression was repeated for this increment.

### Public dimensional and multibody Ewald controls

Added a public skew-2D control with two radius-0.01 m bodies centered at
x = +/-0.05 m, distinct (epsilon, mu) = (2, 1.2), (3, 1.1), and oblique
(.6, 0, .8) incidence at 90 MHz. The Bloch vector is the incident wavevector
projected into the periodic plane. Fixed order 8, compact tables, separation
0.25, quadrature 4/singular 3, Ewald eta 2, real/reciprocal shells 2 and contour
order 32 are used without changing any acceptance threshold.

`/tmp/gsp09-public-ewald-dimensional-v2.log`: 1 passed in 25.89 s. Full original
dense configured-Ewald residual is 6.199959174992559e-13; current relative
difference is 7.525783116927738e-12 (both below 1e-8). Exterior fields agree with
the independent dense solve; translations along both skew lattice vectors
carry the required nonzero Bloch phase. Exact current restoration preserves
both bodies; an incompatible source sector rejects. No cutoff or physical
qualification follows from agreement at fixed Ewald parameters.

The first run `/tmp/gsp09-public-ewald-dimensional-v1.log` preserves a test
construction failure: its Bloch vector incorrectly included the open-axis
component. It also passed the new 3D pole rejection control (3.13 s combined).
A real plane-wave wavevector compatible with all three lattice directions is
one of the fully periodic background spectral modes, so its lossless resolvent
is at a true pole. The current public path correctly rejects it. GSP-07C's
spectral-point eigen/residue/projected quantities and finite unambiguous limiting
paths remain distinct open requirements; a rejection is not a successful 3D
forced scattering solve. No previous checks or full regression were repeated.

### Streamed Ewald regular-correction refinement

`EwaldSystemBlocks.regular_block` exposes the smooth exterior correction without
subtracting the primary/interior operator numerically. Full selected blocks still
include that correction exactly once. `measure_ewald_refinement` traverses all
correction coefficients in bounded edge batches and independently doubles regular
panel order, real shells, reciprocal shells and contour quadrature. It measures
impedance-balanced coefficient Frobenius norms globally and across bodies,
preventing large self/interior terms from masking a periodic correction error.
Empty cross-body sets are reported as unmeasured rather than successful; single
body global convergence does not require nonexistent cross-body coefficients.

This is a quadratic validation diagnostic. Its metric is explicitly different
from the dense mass-normalized L2 proxy, and its differences are refinement
estimates rather than rigorous infinite-tail or implemented-operator error bounds.
Primary singular quadrature, FMM truncation, floating-point enclosure and physical
qualification remain separate obligations. It does not enable automatic Ewald
order/error-budget admission. Independent read-only audit found no P1/P2; the
CUDA test also covers uneven edge batches rather than a single global block.

Evidence: `/tmp/gsp09-ewald-refinement-v2.log`, 1 passed in 25.95 s, two
30-edge bodies, 37-edge batches, 14,400 global and 7,200 cross-body coefficients.
The deliberately shallow eta=0.7, real-shell=1, panel-order=2 fixture fails
both panel and real-shell refinement at rtol=1e-7/atol=1e-12. Real-shell global
difference 3.0660420174761763e-6 exceeds threshold 2.849445767531039e-12;
independent dense correction norms agree. v1 preserves an invalid zero-shell
test configuration rejected by the existing positive-shell kernel contract.

`/tmp/gsp09-ewald-refinement-resolved-v1.log`: 2 passed in 333.05 s. The same
geometry with eta=2, panel order 4, real shells 4, reciprocal shells 2 and
contour order 32 passes all four independent refinements globally and across
bodies. Global differences: panel 4->8 = 7.371192619297495e-17; real 4->8 = 0
at native precision; reciprocal 2->4 = 1.180531527243957e-14; contour 32->64 =
4.5453749650096585e-14, against thresholds about 2.85e-12. Native zero is not
an exact mathematical tail statement. The affected original full-block and
shared exterior test also passes (global 1.849334736521575e-15, cross-body
5.293397379862363e-12); its existing strided-selection bucketize warning remains.
The expensive panel refinement timing is recorded, not a performance claim.
No full suite or already-covered public roundtrips were repeated.

### Independent extinction-work diagnostic for periodic physical closure

Added `incident_surface_work`: direct bounded surface integration of
Re(conj(E_inc).J_s + conj(H_inc).M_s)/2 using the analytic carrier, independently
of modal outgoing amplitudes and flux normalization. It reports per-body work
and preserves native coefficient gradients. This is extinction work, not an
independent material absorption integral, and does not confer physical status.

`/tmp/gsp09-incident-work-v1.log`: 1 passed in 4.03 s. Two bodies, oblique
nonzero Bloch incidence, two coherent oppositely directed normal components,
random complex J/M currents. Surface work 0.0018000174433222664 W agrees with
modal extinction 0.0018000174433222644 W; current gradients agree at rtol 1e-11.

`/tmp/gsp09-saved-periodic-work-v1.log` reuses the stored coefficients in
`/tmp/gsp07-power-refinement-v2.pt` and reconstructs its documented 64-face
geometry. Current modal scattering/extinction/defect first reproduce the saved
ledger at rtol 1e-12. No new periodic matrix assembly or solve is performed.
Surface extinction is 1.800834829714547e-10 W; order 6->12 gives
1.8008348297145554e-10 W; modal extinction is 1.8008348297148676e-10 W.
Scattering remains 9.175952437278452e-11 W and lossless relative defect remains
0.4904611857861883 (FAIL). Thus an extinction sign/normalization mismatch does
not explain this fixture. Discrete-current/boundary energy consistency and
independent radiated-work/material-work diagnostics remain the next physical
investigation. No acceptance thresholds changed, no full regression repeated.

### Original exterior weak-work check of the preserved power failure

`exterior_surface_work` evaluates all original exterior weak blocks, adding the
Ewald regular correction exactly once and excluding interior material terms.
For PMCHWT rows (H,-E), emitted source power is
Re(J* dot row_E - M* dot row_H)/2; the real jump work cancels. The diagnostic
requires a lossless exterior and reports configured-quadrature work, not a
physical certificate. It is quadratic work with bounded forward block storage;
current autograd may retain blocks for backward and is not peak-memory qualified.

`/tmp/gsp09-exterior-work-test-v1.log`: arbitrary complex electric/magnetic
currents on the four-face body at integration order 3 fail the unchanged 1e-5
power tolerance (weak 0.00837435380087781 W, modal 0.00784490316734589 W,
relative difference 0.06748976019688024). This failure remains evidence.
Increasing regular/singular integration to 6 with identical currents, geometry,
kernel and thresholds gives relative difference 3.624357739001782e-8; power and
native current-gradient checks pass in 6.81 s
(`/tmp/gsp09-exterior-work-test-v2.log`). No universal order-6 claim is implied.

`/tmp/gsp09-exterior-work-probe-v1.log` reuses the historical 64-face currents,
without a new solve. Original exterior weak power is 9.175952437283818e-11 W;
Rayleigh scattering is 9.175952437278452e-11 W (relative difference
5.848258543861196e-13). Previously measured surface and modal extinction also
agree. Thus both modal power normalization explanations are excluded for this
fixture: the lossless deficit 0.4904611857861883 remains a discrete-current/field
energy-consistency failure, not a qualified physical result. Boundary/lifted-RHS
consistency and mesh approximation still require investigation. No full regression
or larger mesh was launched.

Independent read-only audit found no P1/P2 in signs, exterior composition or
current derivatives. Its storage-scope observation is reflected above and in the
docstring: differentiable diagnostics can retain quadratic block storage.

### Lifted equation energy decomposition and integration sensitivity

For the saved 64-face current, define W(q,z)=Re(J* z_E - M* z_H)/2.
The equation (A_ext+A_int)q=b implies W(q,A_ext q)=W(q,b)-W(q,A_int q),
regardless of whether the current is physically accurate. This separates an
algebraic energy identity from the physical equality P_extinction=P_scattered.

`/tmp/gsp09-lifted-energy-probe-v1.log` at original regular/singular orders4/6:
interior equivalent work4.955472304938197e-10 W, RHS work5.873067548667094e-10 W,
inferred exterior work9.175952437288972e-11 W, modal scattered9.175952437278452e-11 W.
Equation power residual1.0520703415908821e-22 W. Incident extinction remains
1.8008348297145554e-10 W, with physical deficit8.832395859867102e-11 W.
Thus the linear solve's algebraic energy identity closes while physical
lossless power does not. Interior equivalent work is not material absorption.

`/tmp/gsp09-lifted-energy-refined-v1.log` holds this same current fixed and
increases regular/singular orders to6/8: interior work4.955472304970038e-10 W,
RHS work5.873060750822066e-10 W, inferred exterior9.175884458520276e-11 W.
The change is about6.80e-16 W, only7.7e-6 of the physical deficit. Tested
quadrature sensitivity therefore does not explain the failure. This does not
prove a unique root cause; discrete approximation and boundary-trace consistency
remain to be assessed. No periodic reassembly/solve or larger mesh was used.

Reproduction is maintained in `scripts/dev/diagnose_mom_periodic_lifted_energy.py`.
It reconstructs the fixed historical fixture directly, checks recomputed modal
outputs against the trusted saved artifact at rtol1e-12 before interpretation,
and accepts independent integration orders. The ported runner passes in
`/tmp/gsp09-lifted-energy-runner-v1.log` (equation residual1.0386286566368954e-22 W).
Its small RHS accumulation difference from the earlier fixture-based probe does
not change any conclusion. The original coefficients and failure record are
unchanged; this is diagnostic evidence, not solver or production qualification.

### Nested RWG transfer and enriched-test-space residual

Added `RWGRefinement` for one planar midpoint refinement. Fine coefficients
are determined from coarse normal flux on each fine edge; each row uses only
three parent-face RWGs. The transfer preserves the original piecewise-linear
current and divergence without fitting or moving the surface. Sparse adjoint,
fixed-geometry rules and mesh/transfer mutation guards are explicit. GPU field,
divergence, mass-metric and current-gradient tests pass in3.71s
(`/tmp/gsp09-rwg-refinement-v1.log`).

`build_selected_lifted_rhs` integrates only requested test-edge supports against
all analytic source faces, preserving all five pair integration policies and
arbitrary unique row order. Pair classification is bounded by the configured
batch rather than allocating a global face-pair array. Original full RHS is
unchanged. Selected/full comparison and bounded-workspace checks pass in4.24s
(`/tmp/gsp09-selected-lifted-rhs-v1.log`). Independent read-only audit found
noP1/P2 in refinement orientation, sparse adjoint, source completeness or guards.

For historical64-face currents, one planar refinement gives256 faces/384 edges
without changing the represented current. Preselected fine edges[0,128,256]
produce six J/M test rows. At original integration4/6, their impedance-balanced
relative residual is0.24447109486817128 (absolute1.3059304412903926e-6,
reference5.34186032092916e-6). All256 source faces are integrated, using6 test
faces and class counts[6,18,48,352,1112]. Evidence:
`/tmp/gsp09-enriched-residual-v1.log`. No fine solve or global fine matrix is
formed. This exposes residual outside the coarse Galerkin test space despite
its tiny algebraic residual; it is a sampled indicator, not a global error bound.
The maintained energy runner now offers `--enriched-residual` for reproduction.

Refining the same three probes to integration6/8 gives relative residual
0.24447184732862984, absolute1.3059348100153052e-6 and reference5.341861749258229e-6
(`/tmp/gsp09-enriched-residual-refined-v1.log`, maintained runner). This stability
rules out tested integration sensitivity as the source of the large indicator.
The evidence now supports targeted discrete-space improvement rather than more
modal normalization or coarse-solve tolerance adjustments. Physical power defect
and all previous failed gates remain open; no full regression was repeated.

### Targeted 256-face periodic successor protocol

The stable enriched-test-space residual (~0.24447) motivates one actual planar
refinement solve, implemented in `scripts/dev/solve_mom_periodic_refined_control.py`.
The fixed tetrahedron, epsilon=2, vacuum exterior, 300 MHz normal incidence,
0.7 x 0.8 m cell, Ewald eta=3.5, real/reciprocal shells3/8, contour64, and
regular/singular integration4/6 remain the historical control's settings.
Only the discrete surface is refined64->256 faces. The public solver uses
fixed-order8 compact Ewald FMM, separation0.25, local RWG metric, GMRES with
rwg_diagonal scaling and rtol1e-12/atol0 (500 iterations, restart120). Original
sample/dual admission thresholds remain1e-8. Sampled original residuals cannot
certify all rows; no production or complexity claim follows from this run.

A fresh output directory contains strict-schema executable Result currents,
solver/construction/power diagnostics and a terminal receipt, including measured
failures. The power ledger is compared to direct incident surface work; physical
qualification is always false. Timing/memory cover the single process and are
not repeated scaling evidence. Existing coarse failure artifacts are untouched.

Outcome v1 (`/tmp/gsp09-periodic-refined-control-v1/`, matching `.log`): setup
43.75 s; solver stops at iteration64 for stagnation, relative residual0.0501220.
Original admission was not reached. Its power defect1.00639 is retained but
cannot evaluate mesh improvement because the numerical solve failed. Total
measurement51.52s, peak allocated380281856/reserved614465536 bytes.

Only the Krylov search budget changes for v2: maxiter/restart768, stagnation
window769, identical residual and physical criteria. This allows a complete
768-dimensional search without the premature window stop. The physical/numerical
operator configuration is otherwise unchanged. `/tmp/gsp09-periodic-refined-control-v2/`
and matching `.log` retain strict Result, diagnostics and terminal receipt.
The solve converges at610 iterations/1221 matvecs: true residual7.989658896108002e-13,
compressed dual4.19094668833752e-12, original sampled3.990440209059466e-12
(edges[0,55,109,164,219,274,328,383]); both1e-8 admission gates pass. Original
unobserved rows are not globally certified.

Scattering9.174848314735634e-11 W, extinction9.92800859710112e-11 W,
inferred absorption7.531602823654865e-12 W, lossless defect0.07586217064572262.
Direct incident work9.928008597106781e-11 W agrees. The physical defect improves
from0.490461 to0.075862 under this targeted refinement but remains FAIL; no
threshold was relaxed. Elapsed measurement254.41s, setup44.10s, peak allocated/
reserved380281856/614465536 bytes. These are single-run measurements, not resource
or complexity qualification. The610-iteration burden remains a solver-efficiency
obligation, not an acceptable production scaling demonstration.

Independent runner audit found one failure-archiving P2: diagnostics were written
after an independent power-consistency assertion. Moved diagnostic/energy storage
before that assertion, retaining failure exit behavior; re-review closed P2.
The solve-stage label now avoids implying original admission was necessarily
executed. All prior failure artifacts remain intact; no full regression repeated.

### Original leaf-block preconditioner foundation

`LeafBlockInverse` partitions the original provider's coefficient space by tree
leaves, explicitly splitting any oversized leaf by `max_block_edges`. Every
principal block includes all J/M channels and is integrated using the original
provider, including Ewald corrections when periodic. Native CUDA LU factors,
pivots and coefficient indices have a retained-storage admission before any
original block integration and an actual-byte check afterward. Singular or
nonfinite factors fail closed. Provider/tree/factor mutations are guarded.

`RightPreconditionedOperator` exposes A P and P^H A^H for the common Krylov
protocol without redefining the physical operator. Its current coordinates must
be recovered with P before physical residual admission; recycle state belongs
to the view. This remains internal, not a new public config option. Tests:
`/tmp/gsp09-leaf-inverse-v1.log`,1passed4.05s (original blocks, adjoint/current
gradient, one-byte-short capacity rejection before integration, mutations);
`/tmp/gsp09-leaf-inverse-krylov-v1.log`,1passed4.30s (Krylov coordinate recovery,
original action residual, view adjoint). Independent read-only audit noP1/P2.

`measure_mom_leaf_preconditioner.py` restores the same qualified-numerically,
failed-physically256-face periodic current snapshot, reconstructs its identical
physical FMM, and evaluates leaf32 right preconditioning using the earlier
500-iteration/restart120/window20 GMRES budget with unchanged1e-12 stopping and
1e-8 original/dual gates. The output separates coordinate solve from recovered
physical admission, keeps failures and reports the unchanged physical power
criterion. LU/provider setup cost is part of the measured experiment; factor
storage is not a complete peak-memory guarantee.

Initial256-face leaf32 experiment (`/tmp/gsp09-leaf-preconditioner-control-v1/`
plus `.log`) fails maxiter500: compressed dual0.0028395185217398606 and original
sample0.001323307907380474. Its power defect1.57244 is not a physical comparison
because numerical admission fails.20LU blocks retain511104bytes. Original FMM
setup43.34s, leaf factor construction128.77s, Krylov58.92s, total361.11s including
original residual checking. This is a failed efficiency experiment, not a speedup.

The initial view used only left RWG scaling after applying the block inverse.
The final view applies similarity scaling L(AP)L^-1 in Krylov coordinates,
preserving the local identity block rather than introducing its row-unit spread.
Recovered currents still satisfy q=P y and all residual checks use A q-b. The
updated recovery/adjoint test passes4.29s
(`/tmp/gsp09-leaf-inverse-krylov-v2.log`); v1 numerical failure remains recorded.

Similarity-scaled v2 (`/tmp/gsp09-leaf-preconditioner-control-v2/` and `.log`)
passes numerical admission in284iterations with the500/restart120/window20
budget. Compressed dual1.3955760523673525e-12, original sample2.5230980092687176e-12,
unchanged1e-8 gates. Recovered current differs from the previous admitted610-step
solution by1.531656919787456e-10; lossless defect0.07586217282853039 remainsFAIL.
The20 factors retain511104bytes. Timings: original setup43.22s, factor build128.63s,
Krylov33.43s, total334.87s including original residual checking. Previous
unpreconditioned wide-Krylov solve alone took73.97s, total254.41s. Thus iteration
count/time improve, but this single-RHS end-to-end run is slower due to factor
setup; no overall speedup, amortization or production claim is made. Original
Ewald factor construction and expensive original reference checking remain
optimization obligations. Public preconditioner configuration remains open.
Independent review confirmed the similarity-coordinate correction and current
recovery, with no newP1/P2. No full regression was repeated.

### Primary-image factor option

`LeafBlockInverse(..., factor_kernel='primary')` explicitly uses the shared
scalar regional integration primitive for free-space exterior plus interior
principal blocks. It reuses the original provider's geometry caches without
creating a second numerical operator identity. The original provider/tree
binding remains mandatory for the A P view and all physical residual checks.
`factor_kernel='original'` retains full configured periodic blocks. Receipt
`scalar_leaf_block_inverse_v2` records both the selected mode and its factor
scope; changing the mode after construction rejects. Only the preconditioner
is approximate in primary mode; no Ewald solve or acceptance kernel is changed.

`/tmp/gsp09-primary-leaf-inverse-v1.log`:1passed4.01s. The test prevents Ewald
regular-block evaluation during factor construction, checks each inverse against
independent free-space-plus-interior blocks, verifies the full Ewald provider
binding and tests mode mutation rejection. The same256-face control is measured
with unchanged solve budgets, error gates and original residual sampling; the
factor mode is the sole numerical construction change from the admitted original
leaf-block experiment. No full regression is repeated.

Primary-mode control `/tmp/gsp09-primary-leaf-control-v1/` passes numerical
admission in284iterations. Compressed dual1.3809881045339913e-12 and original
sample2.5163722386466272e-12 retain the same1e-8 gates and eight original rows.
Current difference from the admitted610-step baseline is1.5321035582264543e-10;
lossless power defect0.0758621728825573 remainsFAIL. The20blocks retain511104bytes.
Measured setup43.282s, factors9.293s, Krylov33.393s, total216.070s including
original residual checking. This single run reduces factor construction and
elapsed time versus both previous controls, without establishing repeatability,
scaling, global residual certification or production qualification.

### Public leaf-factor configuration and admission

FMMConfig v6 adds `leaf_factor_kernel` (none/original/primary),
`leaf_factor_max_edges` and `leaf_factor_budget_bytes`. Default remains none.
Factors are prepared once with the original provider/tree and guarded against
replacement. They have a separate retained-factor budget, not an aggregate
peak-memory guarantee. Construction kwargs exclude these solve-only settings;
strict numerical/snapshot identity includes them. Old v5 payloads are rejected,
not silently upgraded; historical artifacts remain usable at their recorded HEAD.
Public free-space and periodic runs recover physical scattered currents before
shared original residual admission, including nonconverged returns. Solver stats
record factor scope/storage and distinguish Krylov and returned coordinates.
Exact Result restore remains decode-only; LU factors are not serialized.

Targeted configuration run:16passed in the v1 test batch. Public comparison v1
also has2FAIL: two of60 small current components exceed the extra elementwise
absolute1e-12 allowance (max5.63e-12 free,1.57e-11 periodic), despite passing
original admission. v2 tighter inner rtol1e-14 has2FAIL from numerical breakdown
at60iterations; neither failure is erased. Final v3 uses the original1e-12 inner
rtol and explicitly checks J and M norm-relative errors separately at1e-8, rather
than asserting resolved digits of individual near-zero components. This does
not establish the failed elementwise criterion. `/tmp/gsp09-public-leaf-v3.log`:
2passed11.35s including complete dense original action, strict Result roundtrip,
configuration tamper rejection and prepared-factor replacement rejection.
Free:54iterations, original sample2.6631642320264136e-12, current relative
8.504919408634534e-11, J1.74735591275427e-9, M8.504916158305044e-11.
Periodic:54iterations, original sample1.3993973467530941e-12, current relative
3.896421915874948e-11, J1.5603896653555496e-9, M3.89641624829181e-11.
Logs v1/v2/v3 are preserved. Independent code audit reports no P1/P2. No physical,
complexity or production qualification follows; final thresholds remain unchanged.

Affected shared admission/default direct-operator controls:7passed15.28s
(`/tmp/gsp09-leaf-shared-admission-v1.log`). Audit follow-up confirms that the
v3 family-norm evidence does not close v1 elementwise or v2 tighter-convergence
failures. No full regression repeated for this increment.

### Original Ewald sampled-action source-face contraction

Original residual evaluation previously constructed full selected edge blocks
for each source-edge batch, repeating smooth face-pair integration when a face's
three edges belonged to different batches. EwaldSystemBlocks now keeps primary
singular/regional blocks on the unchanged bounded edge path, but integrates each
selected-test-support/source-face pair once for the smooth correction. It
contracts all local J/M coefficients and scatters selected original test rows.
Kernel, quadrature, cutoffs, source faces and original residual samples remain
unchanged. Forward pair workspace is bounded by integration pair_batch; native
autograd can retain all pair blocks and is not a bounded-backward-memory claim.

`/tmp/gsp09-ewald-face-action-v1.log`:1passed4.37s. Reversed/mixed-body selected
rows, multiple leading RHS batches and native current gradients match independent
selected full Ewald blocks. The test forbids smooth edge-block assembly in the
optimized action and retains invalid-batch/duplicate-selection rejection.
The fixed256-face public runner now accepts an explicit leaf-factor mode and
records hashes of the changed original-reference sources. The public primary
control retains the earlier500/restart120/window20 budget and all physical and
numerical thresholds; performance and qualification are recorded separately.

Public256 control `/tmp/gsp09-public-primary-face-control-v1/` passes unchanged
numerical admission:284iterations, compressed dual1.3814708608587944e-12,
original sample2.5183992699475088e-12 on the unchanged eight rows. Physical
lossless defect0.07586217288114645 remainsFAIL.20factors retain511104bytes.
Public geometry/FMM/factors/RHS preparation51.972s, solve including original
admission102.641s, inner Krylov32.519s; receipt elapsed154.645s and final power
stage155.149s. Peak allocated389635584bytes, reserved616562688bytes; this is
single-run instrumented storage, not a qualified all-size resource envelope.
The earlier edge-block-reference primary run took216.070s; both run artifacts
remain, but different public/internal runner overheads and lack of repetitions
preclude a controlled speedup or complexity claim. Independent source-face
contraction audit reports noP1/P2. No full regression repeated.

### Registered 1024-face physical successor

The next control uses two planar midpoint refinements of the same64-face
boundary:1024faces/1536edges, fixed geometry, eps_r2,300MHz, original Ewald
cell/cutoffs and quadrature4/6 unchanged. FMM remains p8 compact, separation.25,
primary leaf factors(max32edges,256MiB retained budget). Existing FMM retained
budgets remain1GiB action,512MiB tables,1GiB near and projections. No larger
capacity is granted to force a pass. Krylov is preregistered max1000/restart240/
stagnation_window1001, inner rtol1e-12/atol0; original sample and dual1e-8 gates
are unchanged. Increased Krylov budget is explicit and does not change physical
or original-error thresholds. The same modal power and independent incident-work
check remain, with exact public Result and diagnostics retained on terminal runs.
Single-size evidence is neither complexity qualification nor global certification.
Output is `/tmp/gsp09-public-primary-1024-control-v1/`, with a fresh directory
and source/runner hashes. No repeat of the already-covered256-face solve.

Registration checkpoint: the above control reached fmm_and_rhs_ready at
336.6208722786978s and remained live at32minutes (session32849, PID954143).
This commit freezes the runner/registration only; numerical/physical outcomes
remain pending. Runtime source_head remains db9a7aa plus the recorded runner
hash, matching the launched code. Later compact finite-check candidate edits
were not loaded by this running process and cannot explain its timing.

Terminal1024 evidence (`/tmp/gsp09-public-primary-1024-control-v1/` and `.log`):
normal exit,448iterations/897matvecs, dual1.5166115174215502e-12 and original
sample2.0773994804693794e-10 pass unchanged1e-8 gates. Eight sampled edges are
[0,219,439,658,877,1096,1316,1535]; no global residual certificate follows.
Scattering9.176931476356881e-11W, extinction9.15931839750759e-11W, inferred
absorption-1.7613078849290515e-13W; lossless relative defect0.0019192775814735267
(0.192%) versus256's7.59%. Independent incident work9.159318397505758e-11W
matches the modal extinction check. This is improved mesh evidence, not closure
of independent physical reference, global error, cutoff or production gates.
Preparation336.621s, inner solve2162.594s, solve+admission2449.781s,
receipt elapsed2786.481s, final power stage2787.028s. Peak allocated804945920bytes,
reserved1019215872bytes.107leaf factors retain1676160bytes. The substantial
per-iteration cost increase from256faces requires performance investigation;
no complexity/speedup qualification. This run used the pre-aggregation compact
code atdb9a7aa, not the later uncommitted finite-check candidate. No full regression.

### Rejected compact finite-check aggregation experiment

A candidate accumulated all generated Green/gradient finite flags on-device and
checked once per action. Direct table checks remained immediate; independent
review found noP1/P2. Eight compact/current-gradient/overflow/Ewald-root controls
passed5.33s (`/tmp/gsp09-compact-finite-reduction-v1.log`). A p8 isolated M2L
comparison is bitwise equal for forward/adjoint. Five alternating20-action timing
batches yield median20.163529ms baseline versus20.583045ms aggregated, so this
experiment establishes no speed benefit. The candidate was reverted, not shipped.
Patch `/tmp/gsp09-compact-finite-reduction-v1.patch` preserves all three changed
files; `/tmp/gsp09-compact-finite-timing-v1.{py,json,log}` preserves script,
source hashes and measurements. This is not a full-operator performance claim.

The1024 construction receipt instead identifies84free-space regional far tables
(42exterior/42interior) plus one compact Ewald root. The fixed p8 dense regional
alternative would use84*4*165^2*16=146361600bytes of tables versus1973664compact
regional bytes. With the unchanged other categories, estimated action retention
is602962400bytes including the compact root, below the existing1GiB action cap.
This is an arithmetic estimate from the saved receipt, not runtime admission.
Next evaluate the already-supported dense regional table choice under the same
registered problem and budgets before adding a new caching representation.

### Registered dense regional-table comparison

The1024-face successor now exposes the existing FMM table_storage option in the
runner; default remainscompact. The next run choosesdense regional tables, with
the periodic Ewald root retaining its declared compact representation. All other
problem, quadrature, expansion, leaf factor, solver(max1000/restart240/window1001)
and admission settings match the completed1024compact control. No capacity is
increased. The estimated602962400bytes action retention must still pass actual
construction admission; the estimate does not override a rejection. Output:
`/tmp/gsp09-public-primary-1024-dense-v1/`, fresh path, actual configuration and
runner hash recorded. This is a representation/performance control, not a full
regression or a new physical/complexity qualification.

Dense comparison terminal evidence (`/tmp/gsp09-public-primary-1024-dense-v1/`
and `.log`):448iterations/897matvecs, dual1.5170873897873578e-12, original
sample2.0773813515543772e-10 pass unchanged1e-8 gates on the same eight rows.
Lossless defect0.0019192775858750217, scatter9.176931476356904e-11W and
extinction9.159318397467221e-11W retain the compact physical result. Independent
incident work agrees. Actual admitted action retention602962400bytes matches
preflight estimate, with146385096table bytes including the compact Ewald root.
Preparation343.292s, inner635.327s, solve+admission930.177s, receipt elapsed
1273.552s (final power stage1274.097s). Peak allocated958051840bytes and reserved
1174405120bytes include excluded workspaces/allocator; the1GiB cap applies to
owned action retention, not reserved peak. Relative to the single compact run,
inner time drops from2162.594s and total from~2787s to~1274s. This observed tradeoff
is not repeated/all-size performance, complexity or production qualification.

`/tmp/gsp09-compare-1024-table-currents-v1.{py,json,log}` checks identical scene,
body geometry/order/materials, incident carrier, integration, stopping, periodic
configuration and all other FMM settings before native-CUDA comparison. Jrelative
1.7312939125486613e-12 and Mrelative3.679571737114483e-13 pass1e-8 separately.
Both source solves must be numerically admitted. No new solve or full regression
is used for this comparison. Original physical/global/cutoff/high-frequency/
resource/complexity obligations remain open. The saved near blocks retain
310632128bytes; their storage and execution remain important performance targets.

### Configurable near-action batching

The saved1024 layout has11407near blocks/81shape groups per regional operator.
`/tmp/gsp09-near-batch-control-v1.{py,json,log}` reproduces this exact geometry,
tree, partition and coefficient layout with seeded random fixed blocks. It is a
structural timing proxy, not a physical oracle. Both packed objects retain
155316064bytes and contain identical blocks. Forward, adjoint and current-gradient
comparisons pass1e-11. Five alternating10-action single-RHS batches give median
31.918489ms for action_batch32 and8.476449ms for256; measured extra allocated
memory176128/535552bytes. No complete-solver speedup or peak qualification follows.

FMMConfig v7 exposes `near_action_batch`, default32, through all free-space and
periodic external/internal regions. It uses the existing PackedNearBlocks
parameter, bounding both block entries and RHS vectors per chunk. Increasing it
can therefore substantially grow batched workspaces; retained storage/planning is
unchanged and does not certify those peaks. Positive-integer validation occurs
before regional/system construction, and packed-state guards reject mutation.
Numerical/snapshot identity includes the setting; v6 payloads require their
recorded source revision and are not silently upgraded.

`/tmp/gsp09-public-near-batch-v1.log`:21passed11.34s, including configuration
strictness and public free-space/Ewald propagation to every regional near object,
original numerical admission, separate J/M dense-current controls, Result exact
restore and identity guards. Both public controls take54iterations; sampled
original residuals2.6631642320264136e-12/2.2258588161710935e-12 pass unchanged gates.
Independent source review reports noP1/P2. Defaults, mathematical blocks, physical
thresholds and qualification status are unchanged. No full regression repeated.

### Shared projection validation in regional actions

Regional FMM validation now checks each distinct test/source projection at entry
and validates each coupling's identity, node, medium and retained coefficients
without recursively walking those shared projections. The private composed
action repeats its own coupling checks and preserves finite-input validation.
Standalone M2L calls still validate complete projection state; upward/downward
checks and guarded system backward remain. No persistent validation cache or
mathematical/block/storage change is introduced.

`/tmp/gsp09-shared-space-checks-v1.log`:6passed32.53s, one existing noncontiguous
searchsorted warning. Controls cover dense/compact original weak blocks, adjoints,
current gradients, periodic Ewald composition, projection replacement, node
changes and in-place coefficient mutation. Independent source review: noP1/P2.

Timing uses the real1024-face projection and42dense regional far tables, with
seeded random near blocks and near_action_batch256. This is a structural proxy,
not a physical solve. Frozen original methods are retained in
`/tmp/gsp09-before-shared-{translation,regional}.py`. The first comparison
`/tmp/gsp09-shared-space-timing-v1.{py,log}` FAILED exact bitwise equality before
timing: maximum absolute difference8.644125458514253e-14. It is preserved.
The successor `/tmp/gsp09-shared-space-timing-v2.{py,json,log}` additionally
measures original-vs-original repeats: forward1.0173423748423965e-13 and
adjoint9.947598300641403e-14 maximum absolute differences; candidate-vs-original
forward7.888304520226981e-14 and adjoint9.532931148469529e-14. Forward, adjoint
and gradient comparisons pass the existing1e-11 numerical tolerance. No bitwise
equality claim or physical-gate change follows.

Five alternating rounds of five single-RHS actions produce original times
[.1756336473,.1773046372,.1757757772,.1762985883,.1757370235]seconds and candidate
[.1692411589,.1690497607,.1688387608,.1683152111,.1687630504]. Medians175.775777ms
and168.838761ms show a modest~3.95% regional proxy reduction. Complete solve,
all-size scaling, physical accuracy and peak-memory qualifications remain open.
No full regression or additional1024physical solve was run for this change.


## Separable shift integration phase — 2026-10-02

Explicit separable M2M/L2L is implemented through public free-space/Ewald and finite absorbing-image construction with canonical numerical identity and complete projection storage accounting. See [phase assessment](../assessments/sie_fmm_separable_shifts.md) and its hashed JSON evidence. Selected algebra/public replay and eight-module merge regression pass; original periodic fixture failures and order6 Ewald field failure are preserved. Original 3840-edge full FMM table/kernel-bound failures remain open; projection-only order12 storage is not full FMM resource or complexity qualification. Default dense identity and all final gates remain unchanged.


## Batched native order planning implementation — 2026-10-02

Native support-pair batches replace per-pair Cauchy evaluation while retaining original mathematical bounds and strict admission. Five new scalar-reference controls and 12 affected existing regressions pass; independent phase review finds no P1/P2. See [assessment](../assessments/sie_fmm_batched_order.md) and hashed JSON evidence. Two original 3840-edge collectors remain live in separate frozen providers, so their full admission/qualification remains pending. No FMM production, full resource, scaling or measured planning speedup claim.


## Shared candidate-order planning and saved large-case failure — 2026-10-02

At most16orders share native contour work and one geometric traversal per region;32support-pair batches preserve bounded temporary space and original table/first-kernel-pass semantics. Six new controls and18affectedexisting regressions pass; see [assessment](../assessments/sie_fmm_multiorder_planning.md) and hashed JSON evidence. Earlier original3840edge fixed-ka preallocation now saves actual maximum_order_or_unavailable_bound rejection: allp6..12 kernelsFAIL with0unavailable, p12G.827852/gradient1804.279 vs1e-6/1e-5. Compacttables and structuralownedretention fit their caps, but failedorder/fullresource/physics/scaling remain unqualified. Growing-ka/globalproducer terminal remains pending. Advance valid sharperbounds/fastrepresentation withoutthreshold relaxation; no measuredplanning speedup claim.


## Explicit real-support kernel admission phase — 2026-10-02

The actual original compact/separable3840-edge old-bound collector is terminal FAIL for both fixed/growing ka, preserving kernel1e-6/1e-5, table1GiB/action2GiB. A new independently reviewed real-support Helmholtz tail bound is implemented as explicit `real_support_min_v1`, with unchanged default identities and free-space-only admission. Actual dense/compact polynomial/kernel controls, far-action and native current gradients and public exact state-policy persistence cover the new path. See [assessment](../assessments/sie_fmm_real_support_bound.md) and hashed manifest for concentrated affected regression and new original-size preallocation status. Full FMM action/solve/peak, physical accuracy, complexity, periodic bounds and production qualification remain open.


## Positive-coefficient one-sided tail phase — 2026-10-02

The actual real-support3840-edge collector passes fixed-ka planning atp11 but fails growing-ka atp12 under original budgets. New explicit `positive_tail_min_v1` uses a reviewed finite-wave positive generating majorant and enlarges only the omitted-degree side. Actual M2L/fullkernel and far-action/gradient controls cover the new path; concentrated regression/public-state/originalsize results are recorded in the [assessment](../assessments/sie_fmm_positive_coefficient_tail.md) and hashed manifest. No directionalFMM, fullaction/solve/peak or physical/complexity qualification follows from any preallocation result.


## Compact coefficient batch construction — 2026-10-02

Actual compact coefficients are generated in32center calls and cloned into independent exact-size per-coupling allocations. Default identities/capacity/formulas and strict geometry/kernel admission remain unchanged. Three algebra/ownership controls, a specific previous-batch lifetime fix/control and20affected integration regressions cover the implementation; see [assessment](../assessments/sie_fmm_compact_coefficient_batch.md). Whole FMM setup/action/solve/peak and speed/scaling qualification are still required.


## Exact first-wave positive tail phase — 2026-10-02

Explicit first_wave_min_v1 / cartesian_fmm_first_wave_config_v1 retains the
exact first-wave Legendre coefficient bound, analytic small-x positive-series
upper tail, stable logarithmic attenuation/growth and old available minima.
Free-space only; periodic/Ewald reject. The default identities/gates are
unchanged. GPU1 new controls6PASS20.014s, public actualsolve/replay1PASS40.613s,
affected six existing modules41PASS118.883s, all terminalrc0 and fullsource
before/after/merge match. One earlier runner SyntaxError had no testsrun and
is preserved; no passed controls/public query/solve were repeated.

Original growing-ka3840edge/p6..12 unchanged strict readmission is terminal
FAIL1 428.129s rc1 session62194; old positive-tail gradientFAIL1.2735512327556758e-5>1e-5 stays
preserved. Separate baselinef392fec fixed-ka full-resource collector alsoLIVE;
neither is presently full-resource/physical/scaling production qualification.
See reference proof and assessment JSON for exact source/terminal hashes.
GSP-01OPEN; original global error/resources/scaling/finalGoal gates unchanged.

The first-wave rule leaves the worst p12 G1.4552670511006985e-8/H1.2735512327556758e-5 unchanged;0unavailable. Planned tables444977520B/action1656499936B within caps do not admit the failed order. Full FMM construction/scaling/physics remain unqualified. No fixed-case/repeated archive or numerical re-execution was performed for this terminal update.


## Direct positive missing-degree tail phase — 2026-10-02

Explicit direct_positive_tail_min_v1 / cartesian_fmm_direct_positive_tail_config_v1
implements reviewed direct T1/T2/T3 positive rational-exponential tails, analytic
positive exp-series upper remainder and bounded native p<=64/x<=32 workspace.
Unavailable representations reject; zero support exact0/K0 l0 retained. All
older available bounds/default identities/gates remain; periodic/Ewald reject.
controls-v1 actual6PASS2FAIL20.368s rc1 (test callable mistake),
ONLYaffected2PASS6.047s rc0 after test-onlyfix; no production
change or sixPASS repeat. Publicfreshsolve/replay1PASS50.821s, affected
sevenexistingmodules40PASS102.726s, rc0/source exact except
explained test-onlysource difference. No repeated field query.

Originalgrowing3840/p6..12 unchanged strictreadmission session86520 is terminalFAIL1 456.024s rc1;
G/H worstsupport geometry observed separately during original pass, with real
previous failures bound. No full FMM/resource/solve/scaling/physics qualification.
Independent final phase review found no remaining P1/P2, finalGoal audits not counted.

The direct majorant also leaves the original p12 worst G/H unchanged (H1.2735512327556758e-5>1e-5),0unavailable. G/H witness geometry is saved and points to the same interior support pair. Planned capacity fits but the failed order is not admitted; no full FMM/physical/resource qualification. Pure terminal update, no rerun or duplicate archive.


## Direct tails with exact first-wave correction — 2026-10-02

Explicit direct_first_wave_min_v1 / cartesian_fmm_direct_first_wave_config_v1
subtracts exact positive rational Delta/Delta' corrections from reviewed direct
positive upper tails. Stable native log differences reject ambiguous signs or
unresolved cancellation; K0 direct Laplace and zero-support cases are explicit.
Bounded p<=64/x<=32/97-term workspace, old available minima/default identities
and original gates remain. Free-space only; periodic/Ewald reject.
GPU1 saved actual failure-geometry witness2PASS5.200s, general controls8PASS25.073s,
public fresh solve/exact state replay1PASS43.852s, affected eight modules43PASS94.122s,
all rc0/source before/after/merge exact. One new Bench archive node reuses this
completed producer1PASS3.249s, exactJ/M/canonicalidentity/falsequal with independently
declared Scene/config and assembly/solve forbidden. New controls/field queries,
older producers and fixed-ka original planning were not repeated.

Original3840edge growing-ka strict readmission is tracked in
[phase assessment](../assessments/sie_fmm_direct_first_wave.md); original budgets,
p6..12 and geometry unchanged. Earlier actual failures remain preserved.
Availability-aware per-order G/H witnesses and original admission receipt are
saved before observer/strict admission assertions. Separate frozenf392fec
fixed-ka complete-resource collector cannot qualify this new corrected policy.
Full FMM resources/global accuracy/physical/scaling and production qualification
remain OPEN. GSP-01OPEN; final Goal scope and acceptance thresholds unchanged.

Original3840edge growing-ka readmission terminal1PASS506.025s, rc0, session52407, all389source before/after/merge exact. Selectedp12, G8.108469868062111e-09/H7.0744402135223725e-06 under original1e-6/1e-5,0unavailable,6738far pairs. Tables444977520B, planned retained action1656499936B within original1GiB/2GiB caps. Planning322.3716488694772s/absolute CUDA planning peak1496576B includes witness observation; excludes later capacity walk/full setup/actions/solve. This admits only ideal kernel truncation and planned retained storage: NO fullFMM constructed or global operator/wholepeak/scaling/physics qualification. Actual old failures remain SHA-bound and preserved. Pure terminal documentation update; no repeated numerical run or duplicate archive.


## Batched electromagnetic M2L support admission — 2026-10-02

Shared projection validation occurs once per table batch; complete-support
separation and fixed kernel budgets evaluate in native CUDA groups of at most32
pairs. All admission precedes coefficients. Earliest rejected pair uses scalar
failure details; ambiguous scalar/batch rounding boundaries conservatively reject.
The scalar constructor, original gates, canonical identities and compact/dense
retained ownership are preserved. See phase assessment for actual terminal
controls/regressions and retained test failures. This change does not establish
all-input bit-exact admission, large-case speedup, whole FMM resource/scaling or
physical production qualification. Existing LIVE providers remain frozen.


## Compact M2L layout construction — 2026-10-02

One fixed layout is generated/checked per nonempty batch, with independent
exact-size index/binomial copies retained by every coupling. Original polynomial
formulas, coefficient batches32, capacities and identities remain. GPU1 controls
3PASS6.312s and five affected modules10PASS29.889s,
rc0/fullsourceexact. One temporary layout template is additional construction
workspace; no unchanged-peak or large-speedup claim. See phase assessment;
original full-resource jobs remain separately frozen. No production qualification.


## Native scalar Cartesian FFT translation — 2026-10-02

Actual multipole/local Green/gradient coefficients become circular convolution
through factorial/length scaling and negative source-index embedding; cube>2p
is alias-free on retained target totaldegree. Explicit FFT Hermitian adjoint
and fixed-current backward are implemented. Nativep0..12/complex128, no silent
underflow/fallback, tensor-budget guards and chunked RHS. GPU1 controls12PASS
8.045s and actual33RHS p12 resource/lifetime1PASS3.602s,
rc0/fullsourceexact. Existing numerical/default/public paths unchanged and not
retested. Per-coupling O(p3logp) convolution is not system scaling/highfrequency
directional qualification. See proof/assessment for exact originalpoint and
visible workspace versus actual peak limitations. Next EM/public composition
must retain original gates; all production/fullGoal qualification remains OPEN.


## Electromagnetic FFT channel composition — 2026-10-02

InternalFFTElectromagneticM2L composes actual scalar M2L convolution with
existingJ/M,vector/divergence/material/curl formula and explicit Hermitian
adjoint. OriginalmagneticlossyRWG weakallcolumn1PASS within1e-8 balancedgate;
two initialwronggradient-layout oracleFAIL retained, ONLYtestlinefix and
ONLYtwofailed materialnodes2PASS5.377s rc0.
Scalar primitive source prefix byte-exact unchanged; completed controls/default
public/Bench checks not repeated. See proof/assessment for exact errors/hashes
and wrapperworkspace exclusions. Publictable/config/system/periodic integration
and fullphysics/resources/scaling/highfrequency qualification remain OPEN.


## Public free-space FFT integration — 2026-10-02

Explicit fft_v1 storage/config version, planner/actual owned inventory, native batch32 independent coefficients and EM/scalar RHS chunks4 now compose with public FMM. All old configuration hashes unchanged; all periodic/native unsupported entries fail before construction. Controls4PASS1FAIL preserved, only failed gate moved and1PASS; fresh public/native guards2PASS40.644s, affected57PASS148.608s, exact independently bound Bench archive1PASS3.550s. See mom_fmm_public_fft assessment/reference for hashes, descriptive-only receipt reconciliation and unchanged resource/physics/scaling/high-frequency/periodic/final Goal gates. No production qualification.


## Internal Ewald coefficient FFT — 2026-10-02

Owned arbitraryTaylor FFT uses explicit positive normalization and physical center; zero constant/wholepoly accepted, originalfree radial gate retained. Actual guardedEwaldroot coefficients preserve singleprimary/near exclusion, completebindings andpreexpansioncapacity; temporarycompactsetup disclosed. Native controls8PASS1FAIL16.119s, test-only1Dindexbyte correction then1FAIL3.924s fromfrozenmutation test; testattackfix+None-scalecontractguard ONLYfailed/new2PASS3.475s. Affected20PASS35.810s; no previous8PASS/public/Bench/planner repeats. See proof/assessment for exact source reconciliation and qualification limits. PublicperiodicFFT/fullscope gates remainOPEN.


## Public configured Ewald FFT integration — 2026-10-02

Explicit fft_ewald_v1 selects actual screened root FFT, declared near-image free
FFT hierarchies and free-space body interiors, with strict full config version,
exact metadata/owned inventory and native/binding/unsupported-admission gates.
Initial controls5PASS1FAIL208.281s; only regex failed, sole rerun1PASS3.912s.
Fresh public1FAIL4.253s from incident/Bloch mismatch retained; valid incident
second run1FAIL208.814s after converged/residual-accepted solve and exact replay,
from expected exception type. Independent no-solve recovery1PASS4.872s and
Bench exact archive1PASS4.261s reuse that artifact; none relabels the producerFAIL.
Affected28PASS389.140s; previous successful controls/planning/queries/archives
not repeated. See assessment/reference for exact hashes, inverse source edits,
remaining full-scope qualifications and phase independent audit.


## Internal centered directional coefficient FFT — 2026-10-02

Analytic radial carrier removal plus separable residual-direction coefficients
feeds actual scalar FFT with full physical-gradient carrier terms and Hermitian
current backward. Fixed CUDAfloat64/complex128, p0..12, native batches32 and
independent nine-tensor ownership; extreme physical scaling explicitly rejects.
GPU1 original controls9PASS15.223s, new guards1PASS1FAIL5.507s preserved, only
failed same-polynomial reference normalization corrected and node1PASS3.882s.
Affected21PASS1optional paired-artifactSKIP35.019s is reported literally; old
pairedPASS referenced without repetition. No threshold/physical-case relaxation.
See assessment/reference and exact inverse source reconciliation. Next directional
RWG/projection/cone/tree transitions, actual truncation/native admission, public/
periodic identity and whole resources/Nscaling/physics remain required. Existing
seven LIVE providers remain frozen; consistent tensor AC/DC job is terminalFAIL
from dyadic quadrature budget. No production or final Goal qualification.


## Independent directional RWG electromagnetic projections — 2026-10-02

Internal two-sign source/receiver RWG maps compose the directional scalar with
originalPMCHWTchannels. Physical R A S and fullHermitian S^H A^H R^H retain
complexpassive carriers, full supports/selectedorder and physical divergence.
GPU1 controls4PASS9.302s; affected polynomial/nestedmoment/EM8PASS18.571s.
AuditP2 default-createdgeometry ownedinventory corrected; targeted actualowned/
shared/preflight/byteequal/mutation1PASS3.852s. Exact inverse accounting-only
source reconciliation; originalnumerics not repeated. See assessment/reference.
Explicitcarrierquadrature is not polynomialexact; highka integration/cone/tree/
truncation/nativeadmission/publicperiodic/fullresources/scaling/physics remainOPEN.


## Fixed-carrier directional source/receiver shifts — 2026-10-02

Distinct exp(-ik*u.d)S and exp(+ik*u.d)S^T use unchanged separable layouts,
completecomplexHmaps and fixedcurrent custombackward/chunks4. Bit-exact matching
unitdirections required; changingcarriers require additionalphaseadmission.
GPU1actualpoint/receiver/fullmatrixadjoint/current/storage/native controls5PASS
8.421s; addedfinitephase actionoverflow/underflow/savedbinding1PASS3.134s.
One descriptive testfield corrected with exactinverse link; original records
unchanged. Failedrange-v2preparation is NOT_RUN, notnumericalresult. No existing
production changes or coveredregression repeats. See assessment/reference for
phaseaudit, budgets and remainingcone/tree/error/publicperiodic/fullscope gates.


## Original fixed-ka full setup/action resource result — 2026-10-02

Frozenf392fec original3840/p11/300MHz compact+separable collector terminal
1PASS15654.141s/389stable; actualowned1520814112B/table354688320B/near984969448B.
Measuredsetuppeak1679498752B/backward1706372608B, dot5.527e-15/gradient6.980e-17
pass1e-11. Source/result/exclusion scopes in phase assessment. No currentHEAD,
wholepeak, originalallcolumn/solve/physics/Nscaling/speed qualification; original
runner lacks runtimeidentityexecutionreceipt and no retrospective one is invented.
Owned and near alone exceed sameN dense943718400B: memorycrossover notpassed.
Independentlyintegrated scalarD/K nearcache proposal must preserve original
quadrature/material/H/native budgets; no prospective memorycredit or rerun yet.

## Internal original single-medium D/K near channels — 2026-10-02

Original quadrature now exposes independent D/K without inverse material factors;
the default four-block expressions/scatter order remain unchanged. Owned selected
channels preserve rectangular order, passive magnetic media and native Hermitian
current backward with exact retained budgets. GPU1 controls6PASS6.177s and
affected quadrature/tensor reduction4PASS8.649s complete. Audit configuration
binding P2 fixed; only new six-field mutation/backward test1PASS3.560s ran.
Exact inverse guard source proof avoids repeated covered numerics. Initial SHA
preparation failure remains NOT_RUN. See phase assessment/reference for original
source bindings, numeric thresholds and exclusions. Packed/global FMM integration,
capacity/public identities and actual whole-resource/crossover remain required.

## Explicit original D/K packed free-space FMM integration — 2026-10-02

Shared grouped layout streams two original scalar kernels, with complete J/M
indices, material Hermitian factors, native current backward and global exterior
routing. Exact total capacity changes numeric channels only. Opt-in public
near_storage scalar_dk_v1 has independent canonical identity; default legacy
payload/construction bytes remain unchanged. GPU1 controls6PASS26.196s, affected
numeric14PASS36.000s, config/public3PASS47.342s, independent Bench1PASS3.320s.
Identity regression22PASS1FAIL2.464s preserved; assertion-only legacy field
inventory correction first had1collectionERROR0.126s, preserved; only failed
node final1PASS2.353s. No thresholds/physicalcases altered and no covered solve/
numeric/archive repetition. Full selected actualcapacity and original weak1e-8/
H/current1e-11 pass; public86iter original sampledresidual1.7411e-12 passes1e-8.
See assessment/reference for exact receipts, source reconciliation and phaseaudit.
Original3840 changed-storage resources must next measure actual retained and
whole-lifetime/phase scope before any crossover claim. Existing original workers
remain frozen; directional cone/tree, physical populations and full Goal remainOPEN.


## Fixed-carrier nested directional RWG trees — 2026-10-02

Leaf-only source/receiver maps compose fullsupport trees with separate all-node
physical/Hermitian traversals and native higher-current derivatives. GPU1 installed
Core controls6PASS224.233s/398stable, zero errors/skips/failures. Multilevel depth2/5
and global mixed-leaf controls pass1e-11; selected nonrootka82.8 original weak/H
errors<=2.654725384242167e-11 pass1e-8. Exact owned/shared storage and four-RHS
finite guards pass after a pre-run audit correction. Existing395 production files
unchanged; covered primitive/public/Bench regressions not repeated. Assessment
and reference record all evidence and limitations. Independent phase audit precedes
commit. Next: admitted direction-changing carrier transitions and cone/tree
composition; full quadrature/admission/physical/resources/scaling/population/final
Goal requirements and GSP-01 OPEN remain unchanged.


## Original growing-ka compact/FFT resource evidence — 2026-10-02

Original3840-edge2.4GHz/order12 compact baseline2d3bbb8 andFFT baseline00c5363
finish1PASS each15986.550/11947.451s, rc0/389and390stable sources. Actualowned
1656499936/1736601280B equals expectedcapacity; H/current gates1e-11 pass.
GPU1/installedCore/native execution identity recorded. Earlier actualorder/kernel
admission directly reused, no planning/resource reruns or duplicate archives.
Both owned andnear984969448B exceed dense943718400B; crossover NOTPASSED.
Phasepeaks omitwarmups/differentiableforward, sharedtimings do notprove speedup
orcomplexity. Assessment/reference andindependentphaseaudit bind historicalscope;
currentmain/newDKresource, fullsolve/physics/cones/Nscaling/finalGoal remainOPEN.


## Padded direction-changing carrier transfer — 2026-10-02

Explicit p_source>=p_target+L within0..12 plus ideal phase-only logbound supports
independently positive/negative source/receiver tables and complete native H/
current/higher derivatives. GPU1 installedCore final7PASS6.038s/399stable, zero
errors/skips/failures; balanced point/native gates1e-11 unchanged. First7FAIL3.453s
table finiteguard from nativecomplex0**0 preserved; onlyfailed7nodesrerun after
algebraiczero-power fix. Pre-run componentunderflow P2 closed with negativeguard.
Exact291328B p12→3/tenownedstorages/RHS4 bounds tested. Original9269-byte prefix
and395otherproduction unchanged, coveredoldregressions not repeated. Assessment/
reference andindependentphaseaudit bind limitedscope. Next directionchanging
supporttree/cones mustverify radiusenclosure and independentactualkernel/RWG
errors; phase bound alone notFP/derivative/current/kernel/globalqualification.
Fullresources/Nscaling/crossover/physics/populations/finalGoal/GSP-01 remainOPEN.


## Explicit variable-carrier/order support trees — 2026-10-02

Primary nestedRWG supports explicit complete pernodecarriers/orders/phaseorders,
actualfullsupport radii, peredgepadding/idealphasebudget and distinctallnode4maps/H/
current/highercurrent withvariablecounts. GPU1 installedCore12PASS240.933s/400stable,
zero errors/skips/failures:6new plus6necessaryfixedtree affectedregressions ONCE
aftercommonconstructor/countloopschange. No unrelatedprimitive/public/Bench repeats.
Depth2/5/mixedleaf independentq16 vsprodq12 fourmapmax2.436768923064328e-15<1e-11;
nonrootp8/rootp6/descendantp10 highka82.8 originalweak/H<=2.882651605402313e-11<1e-8.
Exactowned868912(single)/12845024(global)/77456(mixed) andsavedmutation/RHS4 guards
pass. Selectedhighorder eachtree49960112B densephasetables: NOcrossover/resources
advantagequalified. All395otherproduction unchanged; fixedconfig/receipt/math/
inventory preserved. Assessment/reference/independentphaseaudit bind scope.
Nextfactorizephasegathers+sharedseparablecentertransfers; thenactualcones/global
kernel/phasequadrature/nativeadmission/public/periodic/resources/scaling/physics/
populations/finalfourrepoGoal; GSP-01OPEN/alloriginalfailures/thresholds unchanged.

## 2026-10-02 — Factorized changing-carrier transfer primitive

Primary internal compact_phase_v1 uses signed phase coefficient vectors and
shared padded-index/separable-center layouts; default dense/fixed algebra remains
unchanged. GPU1 stage16PASS/1FAIL31.148s/401stable, then only failed storage
control1PASS8.702s/400stable; original zero-tolerance comparator failure is
preserved, corrected after independent review with exact metadata and existing
1e-11 action gates. Production unchanged across runs, sixteen passes not repeated.
Axis-lifetime pre-review P2 closed and all nine affected original separable
controls pass. p12→3/L9 compact60928B vsdense291328B; shared edge7168B, not universal
storage savings or bit-exact actions. Assessment/reference
mom_fmm_compact_changing_carrier.* bind execution and inverse source evidence.
Independent final PHASE report /tmp/gsp-compact-changing-carrier-independent-phase-audit-v1.json
has no P1/P2 and is not final Goal audit. Next actual tree cache/preflight/layout
sharing, then cones/global admission/public/periodic/resources/crossover/Nscaling/
physics/populations/final four-repo checks and >=2FINALaudits. Original GSP-01
failures, all scope/thresholds and false production qualification unchanged.

## 2026-10-02 — Compact changing-carrier tree sharing

Primary nested RWG tree compact_phase_v1 now owns/cache-shares phase layouts per
order triple and separable centers per parent order, each counted once. Full
retained/workspace preflight precedes geometry/layout/quad/transfers; actual
full-support radii and padded phase policies preserved. GPU1 stage19PASS410.813s/
rc0/405stable:7new and12affectedoriginal variable/fixed controls once.395otherprod
and all imported original tests byteequal3a303bf, four map loops unchanged.
Independent q16/current/H/higher max5.0584164037800926e-15<1e-11; actualnonroot
ka82.8 weak/originalH<=2.882656274965703e-11<1e-8. Uniqueowned167632single/
1065488global/77456mixed; eachhighordertree1440784B vs samehistorical49960112B.
Old receipts/case SHA reused without reruns; this is treeinventory notcrossover.
Cache sharing/uniqueinventory/preflight-before-factories/mutation/savedbackward/RHS4
controls pass. See assessment/reference mom_fmm_compact_changing_carrier_tree.*.
Next continuous directional Green and physical-gradient ideal bounds, weighted/
phasepath/global/quad/native error allocation and cones; then public/periodic/
resources/crossover/Nscaling/physics/populations/finalfourrepo/≥2FINALaudits.
No production qualification; original failures/scope/thresholds unchanged.

## 2026-10-02 — Continuous ideal directional kernel admission

Primary fmm_error continuous bivariate Cauchy Green and physical-gradient PERCOMPONENT
bounds support optional directional FFT scalar/all-row batch admission before
coefficients, exact16B independent radii and semantic/savedbackward guards. Bound
workspace includes17validfixed/adaptivecontours; availablepositive tails neverzero.
GPU1 installedCore stage21PASS/1FAIL26.904s/405stable preserved: axialp6 ideal
gradbound1.0066621761278174e-8>unchanged1e-8; noactualerror-exceeds-budgetproof.
Onlyfailedp6 rejection+new samecasep7 ran:2PASS24.978s/399stable; p7gradbound
8.388851575035643e-10, actualfourfieldrelative<=1.7707086541095904e-11.
21priorPASS notrerun; productionunchanged. Loworder/EMcomputedbound*1.01 are
NOTfixed1e-8EM/globalcontrols. Defaultcoefficient/FFT ASTs/errorprefix394otherprod
andoriginaloracle sources unchanged. Assessment/reference mom_directional_kernel_budget.*
bind47SHA and independentPHASEaudit, notFINALGoal. Supportsassumed, boundsideal
only: next weightedRWG/phasepath/global/native/quadrature budget beforecones;
public/periodic/resources/crossover/Nscaling/physics/populations/final4repo/≥2FINAL
remainrequired. GSP-01OPEN/allfailures/scope/thresholds unchanged; no productionqualification.

## 2026-10-02 — Historical original scalar D/K full resources

Frozen1c403d2/installedCore/GPU1 actual3840 fixedka scalar_dk_v1 collector
/tmp/gsp-scalar-dk-original-full-resource-v1 closed1PASS7621.401s/rc0/400stable,
zeroerrors/skips/failures. Original geometry/p11/policy/gates unchanged; original
admission/resourceSHA reused, no planning/defaultproducer repeat. Exactowned
1051291168/table354688320/near515446504/proj180911744B; actualnear+total reduction
469522944B andallothercategoriesunchanged. Owned>dense943718400:crossoverFAILED.
WholePROCESS setup/actions/backward peak allocated1224886272/reserved1317011456B
includeswarmups/difffwd/resetgaps; excludesdevice/solve/teardown andwholeSystemqual.
Adjoint1.6072017103952612e-15/current7.921092909612202e-17<1e-11; sharedGPUtiming
not speedup/Nscaling. Transferredtestedfixturebyteexact/no numericrepeat. Assessment
reports exactnewermainproductiondifferences: historicalreceipt, notcurrentqual.
Seeassessment/reference mom_fmm_scalar_dk_original_resources.* andindependent
PHASEaudit, notFINALGoal. Growing/FFT/directional/currentresources/crossover/
Nscaling/physics/populations/final4repo/≥2FINAL remainOPEN. GSP01OPEN/allfailures/
scope/thresholds unchanged; GoalACTIVE/no productionqualification.

## 2026-10-02 — Direct prescribed RWG weighted directional admission

Primary from_projections derives complete selectedplus/minus support radii and
reusescontinuouskernel/existingmaterial-balancedFrobenius formula beforecoefficients.
Explicitkernel1e-6/1e-4+idealRWGabsolute1e-4 controls are NOTfinalrelativegatechanges.
InstalledCore/GPU1 6PASS9.750s/rc0/398stable/zeroerrors/skips/failures:4new plusonly2
necessaryoriginaldefaultEMweak/H/current controls, no unrelatedrepeats.395otherprod/
originalprojectionAST/weakoraclebytesunchanged. Independentq16L2/peredgeaggregate
1e-11, complete5×3unsorted/shifted actualsupports; weak/H<=1.0232836701534997e-10
<1e-8, nativeH/current/dot/higher<=1.934975691901519e-15<1e-11. Idealweightedbounds
1.2556673124068132e-7/2.8336715573845234e-8; callerprojections excluded, exact
scalar13736B/16Bradii, fullbasisadmissionworkspace219648B. Preallocation/policy/
externalmap/geometry/budget/impedance/receipt/savedbackwardguards pass. Assessment/
reference mom_directional_rwg_budget.* bindexecution+source+diagnosticreceipts;
independentfinalPHASEauditnotFINALGoal. Momentoperator idealcompositiononly with
exactprescribedmaps, noFP/quadrature/phasepath/global/physical/productioncert. Next
phasepathaccumulation/native+quad/global allocation/cones/public/periodic; full
resources/crossover/Nscaling/physics/populations/final4repo/≥2FINAL remainOPEN.
GSP01OPEN/allfailures/scope/thresholds unchanged; GoalACTIVE.

## 2026-10-02 — Prescribed node ideal phase-path plus weighted admission

Native maxdescendant ∏(1+δedge)-1 distinguishesexactsentinels/positivefloors;
from_nested_nodes combinesphysicalG/eachgradient E+Δ(M+E), actualfullcorner radii
never shrink, materialbalancedselectednodestatistics anddoublebudgetsbeforecoefficients.
GPU1installedCore first4PASS124.234s/397stable/rc0 preservephysicalweak/H/current/
higher/guardcoverage. LateP2newimplicitabsolutepytestmathcomparison keptfrozenLIVE;
only2newstrictMP100 root/nodebranch/combinedformula checks2PASS8.777s/397stable.
Explicitabs0/rel1e-11, maxcombinedrel8.138041079178311e-16, pathmaxrel0; production
unchanged/all4coveredPASSnotrerun. Completecompactnonrootp8 300MHz6×6/q16vsq12
weak/H<=8.033058151751436e-14<1e-8; nativeH/current/higher/dot<=1.082470882250305e-15
<1e-11. Kernel1e-6/1e-4 andabsolute1e-4 arecontrolbudgets, NOTfinalgatechanges.
Balancedidealbounds7.068760208257873e-8/1.6377664972307507e-8; ownedscalar27552B/
16Bradii, no persistentpatharrays, externaltreesexcluded. Combined-not-purebudget
negative/preallocation/owner/transfer/savedbackwardcontrols pass.394otherprod/old
allfunctionASTs/wholenestedbyteprefix unchanged; no oldtree/FFT/resource/public
checksrepeat. Assessment/reference mom_directional_phase_path_budget.* bindinverse
strictcorrection/runtime/oracles/source; independentPHASEauditnotFINALGoal.
Idealexactprescribednodecomposition only; no finitequad/FP/global/physical/
resources/scaling/productioncert. Next globalfarpartition/cancellation-safe
allocation and leafquad/nativebudgets beforeactualcones/public/periodic/full
resources/crossover/Nscaling/physics/populations/final4repo/≥2FINAL. GSP01OPEN/
allfailures/scope/thresholds unchanged; GoalACTIVE.


## 2026-10-02 — Explicit disjoint directional FAR actions completed

The internal DirectionalNestedFarActions composes actual admitted complete-node
blocks across distinct carrier owners on shared basis/tree identities. Integer
half-open sweeps reject overlapping/duplicate physical rectangles without an N²
mask; CUDA log-RSS enforces a common-impedance ideal regional absolute budget.
Source-upward/node-M2L/receiver-downward and complete Hermitian/current/higher
maps cache each active owner once per physical RHS chunk and release generated
products. No newly owned retained CUDA arrays; caller dependencies are excluded.
Near coefficient slots remain zero: this is a FAR component, not a full system.

GPU1/installedCore evidence retains initial3PASS2FAIL93.85s (p6 kernel admission),
then only2FAIL12.66s (p8 default128MiB action workspace), then only2PASS652.33s
(p8 explicit512MiB workspace). Sessions68457/28854/16348 terminal and closed,
397 stable sources each, production unchanged across corrections; all original
failures/tests/log/XML/returncodes preserved and3coveredPASS controls not rerun.
Explicit kernel1e-6/1e-4 and ideal node/regional absolute1e-4 budgets, q12 maps/
q16 original references and final weak/H1e-8/nativecurrent1e-11 gates unchanged.
Two30-edge disjoint spheres/83nodes/3blocks/2opposite carrier owners at300MHz,
lossless+lossymagnetic: weak<=4.644375218209839e-14, originalH<=6.568807929616342e-14,
nativeforward/H/current/adjointcurrent/higher/dot<=1.3144393615093706e-15.
MP100 RSS comparisons useabs0/rel1e-11; ideal bounds4.776982970786096e-7 and
1.1416855496644979e-7. Independent766integercell cases, metadata/budget/caller
identity/externaltransfer savedbackward,15noncontiguousRHS/empty/lifetime and
injectednonfiniteoutput controls pass. Plannedcomponentworkspace147240864B,
owned0B; no wholeprocesspeak/crossoverqualification. Entireoldprimarybyteprefix/
23oldfunctionASTs/395otherproductionfiles unchanged; no already-covered tree/
phase/kernel/FFT/public/resource/Bench repeat. Evidence summary andfinalsource
scope /tmp/gsp-directional-far-actions-evidence-v1; assessment/reference and
FEATURE inventory accompany this stage. IndependentfinalPHASEaudit is not a
FINALGoal audit. GSP01OPEN/allfailures/scope/finalgates unchanged; GoalACTIVE.

Next coherent phase adds original packed near correction with exact integer
coverage and complete scalar exterior/body-local interiors using common-eta
B_ext+RSS(interior) rather than RSS across overlapping media. Reviewed design
/tmp/gsp-directional-complete-regional-proposal-v1.md. Leaf/native error budgets,
automatic cones/public/periodic/whole resources/crossover/Nscaling/physics/
populations/finalfourrepo/at least2FINALaudits remain required.


## 2026-10-02 — Complete prescribed directional near/FAR scalar systems

Original packed D/K near plus explicit disjoint directional FAR now proves full
coefficient pair coverage by half-open integer disjointness and total area before
integration. Actual scalar bodies bind complete exterior/interior regions and one
common eta; native ideal admission uses B_ext+RSS(disjoint interiors), not RSS of
overlapping media. Full H/current/higher actions, bounded caches, exact owned near
and embedding inventories and semantic/numeric saved-backward guards implemented.
GPU1 installedCore/nativeCUDA targeted4PASS18.469s/397stable/rc0/session90946 closed;
no covered FAR/tree/phase/kernel/FFT/public/resource/Bench repeat. Full24x24 lossy
magnetic two-body weak2.3857070725312304e-16/originalH2.9327541750632917e-16,
native/current maximum2.7836295418106014e-16, original1e-8/1e-11 gates unchanged.
All physical interiors are original near; actual positive interior FAR physics is
not qualified. Strict independent MP100abs0 overlap-norm control and all-FAR empty,
coverage/capacity/identity/mutation guards pass. Original near reference shares q4
quadrature and proves composition only. Owned near2776+1392+1392B/system192B;
planned system workspace91,119,680B excludes setup/callerI/O/cache/FFT, no total
resource/processpeak/crossover claim. Entire3primarybyteprefixes/23oldAST/393other
production files unchanged. Evidence and independent PHASE review paths:
/tmp/gsp-directional-complete-regional-evidence-v1/summary.json and
/tmp/gsp-directional-complete-regional-independent-phase-audit-v1.json.
Review is not FINAL Goal acceptance. GSP01OPEN/priorfailures/scope/gates unchanged.
Leaf/native certification, positive interior FAR controls, automatic cones/public/
periodic whole resources/crossover/Nscaling/physics/populations/finalfourrepo/atleast
2FINAL independent audits remain required; no production qualification granted.


## 2026-10-02 — Prescribed directional shared Riesz/Krylov integration

Actual complete DirectionalSystemActions/provider bind before native LocalRWGRiesz;
shared Krylov/current/residual/preconditioning/persistence reused, no rebuilding
or dense/H conversion. GPU1 installedCore/nativeCUDA targeted3PASS
20.27s/397stable/rc0/session38258 CLOSED, old primary whole prefix/2oldASTs/
395otherproduction unchanged; no covered physical/tree/kernel/public/resources
checks repeated. Two6-edge lossy magnetic bodies at400MHz, actualEx/Ey2RHS
GMRES/blockGMRES; full original dual<=1.4249006418883665e-12, dense current
relative<=1.5594300533200172e-11, native Riesz residual-current gradient
0.0 with unchanged1e-8/1e-7/1e-11 selected gates.
Both residual admissions pass, nofallback; one-step nonconvergence/directfallback
refusal/zero-current wrongresidual/preRiesz identity/savedactualaction+metric
guards pass. Raw allocator-since-external-reset fields are not wholeprocesspeak
proof, timings notscaling. Physicalinteriors allNEAR; originalq3near oracle shares
integration and qualifies composition only, not positive interiorFAR ornear physics.
Idealcommoneta bound notRiesz/residualcert; runtimeoriginalsample remainsnotglobal.
Evidence /tmp/gsp-directional-krylov-evidence-v1/summary.json; independent PHASE
review /tmp/gsp-directional-krylov-independent-phase-audit-v1.json, NOTFINALGoal.
GSP01OPEN/priorfailures/scope/gates unchanged. Leaf/nativebudgets, positive interior
FAR, automaticcones/public/periodic/wholeresources/crossover/Nscaling/physics/
populations/finalfourrepo/atleast2FINAL audits remainrequired. No productionqual.


## 2026-10-02 — Actual connected-body selected internal directional FAR controls

No production edits:396productionfiles byteequal. Actual connected480-RWG closed
sphere300MHzeps2mu1.3 lossless/lossy(.01sigmae/.02sigmam), commonactualexterior eta,
full960physicaldomain retained in prescribedcomplete6node supporttree. Twoordered
single-RWG internalFAR blocks, twooppositep12/q16 carriers and originalq20selected
4x4 reference, zeroomittednear/FARslots; notcompleteinteriorregion/systemphysics.
Initial2FAIL1PASS9.170s/397stable/rc1/session76645 CLOSED: actualweak
.8723484927515467/3.008941180965765 with length.01 remainunqualified/preserved.
Onlyfailed2PASS30.380s/397stable/rc0/session45280 CLOSED aftersingleexplicitlength
changes toactualD/(2p+2); allproduction/support/material/order/quadrature/budgets/
original1e-8weakH/1e-11nativegates unchanged. Kernel100/1e5 andideal.1 arenew
conservativegeometrycontrolbudgets, not finalgates. Inverse restoresoriginaltest
byteexact; coarsePASSfunctionASTunchanged/notrerun. Currentweak<=2.7822870141741786e-12,
originalH<=2.79032687459648e-12, nativecurrent/higher/H/dot<=3.632134563489498e-14.
15noncontigphysicalRHS/zerosavedinternals/exactzeroomittedoutputs/exactcaller
projectionbytes28,811,776+27,960,016 pass. NewFARowned0B/componentworkspace210,184,272B
excludescallerI/O/cache/FFTlibraries, no wholepeak/crossover/scaling claim.
Coarseactual120-RWG support ratio.633 unavailablep8/10/12 rejectszeroplaceholders;
480selectedratio.3047209342521525. Idealbound.05079/.04920 unchangedacrosslength,
notnativeerrorcert. Evidence /tmp/gsp-directional-interior-far-evidence-v1/summary.json;
independentPHASEaudit /tmp/gsp-directional-interior-far-independent-phase-audit-v1.json
isnotFINALGoal. No oldcoveredweak/tree/phase/kernel/FFT/public/resource/Bench
repeat. GSP01OPEN/alloldfailures/finalscope/gates unchanged. Actualcompleteinterior
near+nonzeroFARphysics/finiteleafnativebudgets/automaticnormalization/cones/order/
public/periodic/wholeresources/crossover/Nscaling/physics/populations/finalfourrepo/
atleast2FINAL audits remainrequired; no productionqualification.

### Bounded automatic directional cone/order planning completed

Primary fmm_order.py append only/old prefix and3 original ASTs unchanged/395other
production files identical. At most32 fixed CUDA float64 declared support pairs;
canonical dominant-axis cube keys, minimum passing p0..12 then coarsest grid;
available required before budget, all rows pass before selected primitive factory.
Automatic min(D/(2p+2), max positive support radius), zero-support distance fallback.
All flags remain false for physical/native/production/global qualification.
Original v1/session51749 CLOSED4FAIL2.92s/397stable (defaultfloat32sign). One dtype
correction v2/session99688 CLOSED2PASS2FAIL4.72s/397stable: independentMP100
selectionmax1.197930643570544e-13<1e-11/owned576B/strictcapacity+mutation PASS;
distance-only nativephysicalmax730073.4164/208.7741 FAILpreserved. Supportcap
v3/session53907 CLOSED onlyfailed2+newnormalization1 PASS3.89s/397stable, unchanged
physicaltestAST/kernel1e-8/relativephysical1e-8/native1e-11 gates. Actual p6/p5
grid32 fourfieldsmax8.330138812244727e-10, nativecurrent/higher0/dot4.00655e-16.
No priorPASS rerun. Plan144B/pair/primitive13736or9032B; componentworkspace excludes
caller/library/allocator, not wholepeak/crossover/scaling. Evidence
/tmp/gsp-directional-cone-order-evidence-v1/summary.json; independent PHASE audit
/tmp/gsp-directional-cone-order-independent-phase-audit-v1.json, notFINALGoal.
GSP01OPEN/alloriginalfailures/finalscope+gates unchanged. Full sparse tree contexts,
actualsupport/leafnative+quadrature/phase/global admission, complete interiorFAR
systemphysics/public/periodic/wholeresources/crossover/Nscaling/populations/current
fourrepo FINALacceptance and atleast2FINALaudits remainrequired; no qualification.

### Automatic actual-support directional RWG blocks completed

Primary fmm_directional_rwg/fft only;394 other production files unchanged. Two
original constructors share extracted exact structural capacity formulas with
new actual-support automatic direct block, inverse old module prefixes byteexact.
Complete selected plus/minus vertices and native U/Q material weights derive
conservative half-squared ideal budget/channel before automatic cone/order and
owned direct-map/PMCHWT factories. All ideal/capacity/quadrature checks before
integration/coefficients, actual support/geometry/medium/configuration guards.
v1/session25560 CLOSED7PASS10.21s/399stable (3new cases+4affected original only);
v2/session94699 CLOSED onlyNEW MP100 budget1PASS3.69s/397stable; all396prod and
originaltestprefixunchanged, old7PASS notrerun. Lossless/lossy automaticp6/p5->
tighteridealp10/p9, unsorted5x3 actualRWGs/common exterior eta376.7303136668617.
Originalweakmax4.032152582963799e-9/fullH5.640352022062615e-9<1e-8, nativecurrent/
higher0/dot2.4344881870058463e-15<1e-11/15noncontigRHS/no savedM2L internals.
IndependentMP100 effectivebudgetsmaxrel4.489116308638273e-15/combined2.35306590941802e-15
<1e-11; unsplit sqrt2counterexample verified. Owned100096/66720B includes184plan+
generatedinputs, geometryexcluded; workspace119389776/85897296B componentonly,
notwholepeak/crossover/scaling. Evidence /tmp/gsp-directional-actual-support-evidence-v1/summary.json;
independentPHASEaudit /tmp/gsp-directional-actual-support-independent-phase-audit-v1.json, notFINALGoal.
GSP01/GSP09OPEN/GoalACTIVE/alloldfailures/finalscope+gates unchanged. Actualcomplete
interiornear+FAR/sparsecone-treecontexts/finitequadrature+native/phase-global/public/
periodic/wholeresources/crossover/Nscaling/physics/populations/finalfourrepo/atleast2
FINALaudits remainrequired; no productionqualification.

Initial PHASEaudit2P2 preserved: unbounded wrapper intermediates and outer
saved-backward guard bypass. Unified correction changes only new class3action
methods: reuse4RHS chunks +shared fixed linear protocol. v3/session4370 CLOSED
onlyNEW changedpath3PASS8.88s/397stable; old7PASS+MP1PASS notrerun/originalweakmatrix
SHA reusedinplace/no integrationrepeat. Hmax4.965299572218501e-9/native1.8707764225587183e-15,
alloriginalgatesunchanged; wholewrappersaved0/maxchunk4/lifetime+empty+14saved
backwardtripwires PASS. Final consolidated evidence
/tmp/gsp-directional-actual-support-evidence-v2/summary.json; correctedPHASEaudit
/tmp/gsp-directional-actual-support-independent-phase-audit-v2.json; notFINALGoal.

Finaltestportabilitycorrection removes unconditional historical/tmp dependency:
optional explicitSHA-bound reuse, defaultfreshsameq16originalreference/no skips.
OnlyNEW lifecycle v4/session56849 CLOSED1PASS2.42s/397stable/all396prod unchanged;
no covered numericalrepeat. Final evidence
/tmp/gsp-directional-actual-support-evidence-v3/summary.json andPHASEaudit
/tmp/gsp-directional-actual-support-independent-phase-audit-v3.json.

### Prescribed sparse directional RWG context forests completed

Primary fmm_directional_moments only/395 other prod unchanged. Reverse3context
cardinality expressions restoresentireoldprefixbyteexact; old fullowners counts
equalphysicalnodecount. New selected(node,carrier/order) descendantclosures only,
shared samefamily ancestor/descendantcontexts, uniqueorderlayouts, actualsupport
andglobal mixedleafroutes; no fulltreepercone. Allfourmap/current/higher methods
are shared primary methodobjects, no duplicate numeric kernels. Sparseindices
explicitlynotoldfulltreetype; oldEMconstructorrejectsearlyuntilcorrectsparsebinding.
v1/session97374 CLOSED5PASS42.61s/398stable (4newcases+1necessaryaffectedoldsingle);
v2/session22256 onlyNEWconfig1PASS3.45s/397stable, oldprefix inverse/testprefixexact;
v3/session52254 onlyNEWfixedmodes1PASS6.53s/397stable;v4/session55099 onlyNEWprotocol
1PASS3.45s/397stable; no earliercoveredPASS repeats. Independentq16 sparsefourmap
max1.907498768330214e-15<1e-11/current+higher0/no savedwrapper/15noncontigRHS/missing
cotangents/empty/20omittedphysicaledgesexactzero/capacity+savedmutationmodeguards
PASS. Active12/11vsfull-treepercarrier78/102slots; sameorderlayoutsharedonce.
Owned72224/84216/423696B/componentworkspace4031376/4151376/4534752B exclcaller/library/
allocator/notwholepeak/crossover/scaling. Mathematical samefamilyphase truncation0
notnativecertification. Evidence /tmp/gsp-directional-sparse-contexts-evidence-v1/summary.json;
independentPHASEaudit /tmp/gsp-directional-sparse-contexts-independent-phase-audit-v1.json,
notFINALGoal. GSP01/GSP09OPEN/GoalACTIVE/alloldfailures/finalscope+gates unchanged.
Next actualsparsecoupling/kernel+weighted+phase-global/automaticpartition-order
contexts/completeinteriornear+FAR/nativequad/publicperiodic/wholeresources/crossover/
Nscaling/physics/populations/finalfourrepo/atleast2FINALaudits allremainrequired.
No production/physics/complexity qualification.

### Sparse actual-medium directional M2L and FAR completed

Primary fmm_directional_rwg only/395 other production files unchanged; shared
support-radius/complete FAR initialization inverse recovers old module byte equal.
Sparse material-bound factory uses explicit active context IDs and separately
records physical nodes; actual complete supports, common exterior eta, continuous
G/per-gradient and weighted U/Q ideal admission before coefficients. Fixed family
ideal phase truncation0 excludes native phase/shift/quadrature. Shared old FAR
numerical/current/adjoint/higher protocol, physical disjoint rectangle sweep and
RSS aggregation; no duplicate PMCHWT kernels or full-tree-per-cone maps.
v1/session72117 CLOSED6PASS129.832s/401stable/rc0:4new+2necessary affected extraction
checks. v2/session85665 onlyNEW unequal-order/native-mode/saved-adjoint1PASS12.693s/
398stable/rc0; sparse-only guard inverse/test prefix exact, original6PASS not rerun.
Actual30+30RWG spheres300MHz eps2/mu1.3 lossless/lossy sigma_e.01/sigma_m.02 p8/q12
vs originalq16: weak<=3.63859e-14/H<=5.52113e-14<unchanged1e-8, native/current/higher/
dot<=1.11253e-15<1e-11/no savedwrapper/15noncontig/omitted edges exact0. MP100
independent triangle/material weighted relative<=2.72581e-15<1e-11 and RSS abs0.
Physicaltree83/active24 vs hypothetical166; owned map640640B/couplings82656B/new FAR0B,
component mapworkspace19985168B/FAR103484288B excludes caller/library/allocator,
not wholepeak/crossover/scaling. All preflight/saved configuration/mode guards,
physical4RHS/cache lifetimes/empty backward PASS. Three original GPU jobs still LIVE.
Evidence /tmp/gsp-directional-sparse-coupling-evidence-v1/summary.json; independent
PHASE audit required before commit, not FINAL Goal audit. GSP01/GSP09 OPEN/Goal
ACTIVE/all original failures/final scope+gates unchanged. Next automatic sparse
partition/cone/order and full regional/system integration; complete interior near+
FAR/nativequad/global/publicperiodic/whole resources/crossover/Nscaling/physics/
populations/final fourrepo/at least2 FINAL independent audits remain required.
No production/physics/complexity qualification.

### Complete sparse directional near/FAR system and Riesz/Krylov completed

Primary fmm_regional only/395 other production unchanged/whole module inverse
exact/five original action/state/bound ASTs unchanged. Accept admitted sparse FAR;
near IDs always complete physical tree, including nodes outside active context ID
range. Shared packed original D/K/full coverage/actual body+medium/common eta/
regional+system/native guarded Riesz/Krylov; no duplicate physics implementation.
Same closed physical slender boxes: coarse18/body near_tol1e-5/max16 fails.0193,
v1/session30936 CLOSED old affected1PASS+4ERROR7.30s/399stable, oldPASS not rerun.
Refine same geometry longitudinally to198/body, max32/tolunchanged, all396prod and
four original gate ASTs unchanged. v2/session10796 CLOSED3PASS1FAIL541.59s/398stable:
whole originalq6/singular3 weak2.77639e-16/six independentq16 FAR≤1.94774e-13,
originalH6.90310e-16/native-current-higher≤1.70104e-15/no savedwrapper/15noncontig,
MP100 ideal triangle+RSS abs0, physical coverage/capacity/saved-system+Riesz PASS.
Both actual eps2/mu1.3 and eps3/mu1.5 lossy interiors have2 positive FAR+7near;
exterior2crossbody FAR+34near, complete396 edges/792 physical currents preserved.
Diagonal-only GMRES original dual.275741/current.956607 FAIL retained. Existing
bounded original32-edge inverse: v3/session52653 CLOSED NEW reference lifecycle
1PASS+solve1FAIL442.429s/400stable, full original2.33421e-10/current2.10933e-9 but
192 steps do not meet inner1e-12. v4/session27812 onlyFAILED solve capacity768/384,
all original tolerances/material/geometries/q/p/preconditioner/reference unchanged.
No previousPASS repeat/oracle reintegration: optional identity+SHA-bound original
matrix1940060e97fb5df1341c3d87b51f5577e0ee670883b839d15cfe258020f272d5 reusedinplace,
portable fresh ordinary default/strictwrong-SHA/identity/missing/partial lifecycle.
Owned near5094936/1273632/1273632B/maps84944each/couplings55104each/indices6336B,
not wholepeak/crossover/scaling. All original failures retained/unqualified.
Evidence /tmp/gsp-directional-sparse-system-evidence-v1/summary.json; independent
PHASE audit /tmp/gsp-directional-sparse-system-independent-phase-audit-v1.json,
not FINAL Goal. GSP01/GSP09 OPEN/Goal ACTIVE/original final scope+gates unchanged.
Automatic sparse partition/cone/order/changing-carrier DAG/global certification,
complete nativequad/physical/publicperiodic/whole resources/crossover/Nscaling/
populations/final fourrepo/at least2 FINAL audits remain required. No qualification.

v4 CLOSED1PASS/511.75s pytest/512.9551367191598s runner/400stable/rc0/no failures/errors/skips. All phase sessions drained and actual processes absent; original three GPU jobs remain LIVE.

Configured solve results: [{"method": "gmres", "original_full_dual": 7.778820926838563e-13, "current": 3.2365996804172426e-12, "iterations": 367}, {"method": "block_gmres", "original_full_dual": 1.4353432449029596e-12, "current": 5.86589436237549e-12, "iterations": 127}]

### Automatic complete sparse directional regional planning completed

Primary fmm_order appended auto actual support tree/complete shared sphere partition,
material/RWG weighted pair B/sqrt(F), half-square G/grad budget, shared minimum-order/
coarsest-cube selection/canonical carrier+order/active requests/native final RSS.
Build uses shared sparse maps/actual medium FFT M2L/original D/K near; no duplicate
physics kernels. Four primary sparse owner empty-forest constructor expressions;
old module inverse byte equal/old order prefix byte equal/394 other prod unchanged.
All-near owners allocate0B/no maps/layouts/shifts; complete physical domain retained.
v1/session27340 CLOSED5PASS8.91s/401stable/rc0:4new+1necessary affected nonempty guard.
Two actual lossy magnetic6-edge tetra bodies300MHz, complete12edges/exterior2FAR+
2near/two all-near interiors, automatic p7/two families. Originalq4/singular3 whole
weak3.07380e-16/two independentq16 FAR<=7.63734e-13<1e-8/H2.87939e-16/native current+
higher0/dot2.93129e-16<1e-11/no savedwrapper/15noncontig. MP100 independent actual
material limits/exhaustive cube/order/RSS max7.10462e-16<1e-11/abs0. Complete physical
coverage/allnear empty/no factory capacities/order/quadrature/family/guards PASS.
Initial independentP2 localmedium physical value binding closed by state-only
frequency/eps/mu/k/eta +treeversion/leaf/morton. v2/session77918 onlyNEW guard1PASS
3.37s/400stable/rc0; reversefix recovers entiretestedv1/order, oldtestprefix exact,
old5PASS not rerun. Both sessions drained/actual processes absent; original3LIVE.
Exact plan inventories1016/216/216B include owned tree/catalog; separate component
workspace excludes caller geometry/Python metadata/library/allocator/solver/oracle,
notwholepeak/crossover/scaling. Evidence /tmp/gsp-directional-automatic-region-evidence-v1/summary.json.
IndependentPHASEaudit required beforecommit/notFINALGoal. GoalACTIVE/GSP01/GSP09OPEN;
alloldfailures/finalscope+gates unchanged. Automatic positiveinteriorFAR/global
native+quadrature/changingcarrierDAG/publicperiodic/wholeresources/crossover/Nscaling/
physics/populations/finalfourrepo/atleast2FINALaudits required. No qualification.

A final NEW planner-only control stresses actual material tightening rather than explicit kernel clipping. Starting from loose G/grad budgets1/1 and ideal regional B=1e-10, both ordered FAR pairs receive independent100-digit derived limits G=1.63813114568e-10 and grad=1.01635123603e-6. Exhaustive cube/order selection still gives p7; material-weighted pair limits and final RSS relative errors are at most4.26696396932e-15 with absolute tolerance0. The map factory is forbidden throughout. v3/session78499 CLOSED1PASS3.69s/400stable/rc0, all396 production files byte equal to tested v2, entire v2 test prefix SHA exact. No existing6PASS, maps, near physics, original matrix or FAR oracle is repeated. The independent final phase audit includes this supplemental evidence.

### Prescribed shared changing-carrier RWG DAG completed

Primary fmm_directional_moments only/395 other prod unchanged. Sharedinitializer
DAG=False exactly original sparseconstructor AST; original two actionASTs exact
when _transfer lookup restored;23otheroldmethodAST exact. All8native/currentaction
methodobjects shared. Explicit canonicalreachabletopological physical contexts/
complete direct child coverage/sourceorder>=parent+L/phase<=12; multi-parent shared
child integratedonce/sourceupwardonce/downward sumsallparent locals beforedescend.
Existing compactchanging transfers keyed(parent,child), uniquephase+centerlayouts;
actualsupport+phase+capacity beforemaps. DistinctDagtype oldphase0M2L refuses.
v1/session39713 CLOSED5PASS1FAIL178.832s/401stable/rc1:3newactual mapcontrols+2necessary
oldprotocolPASS; negativeM2L omittedkeywordsFAIL preserved. v2/session23034 onlyFAILED
1PASS4.959s/399stable/rc0, all396prod unchanged/oldpassinggate+fixtureAST exact; old5PASS
notrerun. Actual30edge sphere/global60/mixedfullleaf underdeclaredk5+.8j maps only:
allfourindependentq16 maps+H max3.37472e-15/nativecurrent5.23087e-17/higher5.52108e-17
<1e-11/15noncontig/no savedwrapper/missingcotangents/sharedlocals sum. MP100 allcontext
maxphaseproducts exact0orpositive rel<=4.68365e-15<1e-11/abs0/notindegree multiplying.
Owned1244568/2059560/1723672B/context40vs78,52vs102,3vs4; componentworkspaces29715584/
32858544/26215152B excludescaller tree/geometry/metadata/wholeI/O/allocator/library/
solver/oracle, notwholepeak/crossover/Nscaling. Additionalimportedoldhelper absent
frominitialfrozeninventory honestlyseparateimmutablebaselineblobproof/notrewritten.
Bothcollectorsdrained/actualprocessesabsent. IndependentPHASEaudit beforecommit/
notFINALGoal. GoalACTIVE/GSP01/GSP09OPEN/alloldfailures/finalscope+gates unchanged.
Next phase-inclusive actualmaterial DAGM2L/fullnear+FAR/automaticchildconesorders/
fullnativequad/publicperiodic/wholeresources/crossover/Nscaling/physics/populations/
finalfourrepo/atleast2FINALaudits. No production/materialweak/complexity qualification.

### Material-bound changing-carrier DAG FAR coupling completed

Primary fmm_directional_rwg shares the existing phase majorant and sparse factory;
395 other production modules byte equal and37 original method ASTs exact. Distinct
DAG material admission includes both maximum path products before effective
kernel/RWG material budgets; physical support/context IDs separate, phase workspace
before statistics and effective budget failure before coefficient allocation.
Disjoint DAG FAR owner shares original physical/full-H/current/higher actions.
Actual300MHz eps2/mu1.3 lossless/lossy .01/.02 sigma, two30-edge spheres, common
exterior eta. Two selected6x4/4x6 physical rectangles in120 current coordinates;
not complete near/FAR or global operator certification. p8->12/L4 with signed
neighboring carriers/8requests/28contexts/20sharedchildren/40transfers/tree83.
Originalq16 weak<=3.70824e-14/fullH<=5.15265e-14<1e-8, native<=6.01163e-16<1e-11,
15noncontiguous RHS/no saved intermediates. MP100 phase/material/RSS<=5.16627e-15
relative1e-11/abs0; phase defects6.1e-26/6.8e-26, low-order measurable positive
phase-only budget rejection before coefficients, capacity/protocol/saved-F/H guards.
v1/session62970 CLOSED2PASS3FAIL21.696s/402stable/rc1 retained. Only failed fixture
signed carrier routing and invalid-ID call corrected; all396prod/physical gates
unchanged. v2/session76332 only3FAIL CLOSED3PASS163.631s/399stable/rc0; old2PASS
not repeated. Both sessions drained/processes absent/all transitive imports frozen.
Component map1478488B/couplings55104B/action0B; map/action workspace59163192/
155892360B excludes caller/allocator/library/solver/oracle, notwholepeak/crossover/
Nscaling. Independent PHASE audit required beforecommit/not FINAL Goal audit.
Evidence /tmp/gsp-directional-dag-coupling-evidence-v1/summary.json. GoalACTIVE,
GSP01/GSP09OPEN; all original failures/final scope/gates unchanged. Next complete
DAG near/FAR/automatic child cones/orders/nativequad/publicperiodic/whole resources/
crossover/Nscaling/physics/populations/finalfourrepo/atleast2FINALaudits required.
No production qualification. Earlier automatic magnetic actual collector remains
frozen in its original provider; its terminal acceptance is still pending.

### Automatic complete magnetic sparse interior controls completed

Tests/evidence only; no production change in this stage. Genuine numerical baseline
21e46ff retained in immutable collectors; after terminal source reconciles to
auditedf747466 by394byteequal modules and exact legacy constructor/lookup plus
inlined DAG=False sparse factory/admission/workspace/sparse=True FAR initializer
AST proofs;37 original class methods exact. No long run repeated/no current full
regression or production qualification. Original reference identity/SHA-bound
matrix1940060e97fb5df1341c3d87b51f5577e0ee670883b839d15cfe258020f272d5 reused
inplace/no reintegration/duplicate archive; all transitive helpers frozen.
Two198-edge closed magnetic boxes300MHz, eps2mu1.3sigma.01/.02 and eps3mu1.5sigma
.015/.005; originalgeometry/q6singular3nearmax32/commonexterior eta/gates unchanged.
Automatic leaf8/ratio.1/cones(2):exterior550FAR5124near/26families p8..12, both
interiors62FAR1469near/8families p7..10; every physical pair covered exactly once.
Plan/session21331 CLOSED1PASS32.283s/399stable/rc0. Actions/session95502 onlyNEW
CLOSED1PASS5029.069s/401stable/rc0; oldplanPASS not repeated/both handles drained/
actualPIDs absent.15noncontig complete-domain original forward1.29362e-15/H5.18661e-16
<1e-8/native current7.58317e-17/higher8.40392e-17/dot1.99502e-16<1e-11/saved0.
Sampled actions not a full-matrix norm certificate. Six first/last selectedq16
FAR probes max6.78076e-12<1e-8; remaining FAR quadrature uncertified.
Plan owned131032/26288/26288B/near4911752/1428176/1428176B/regional component
workspace1323360784/245560688/250833520B/explicit2GiB; excludes caller/allocator/
library/solver/oracle, notwholepeak/crossover/Nscaling or qualified performance.
Evidence /tmp/gsp-directional-automatic-interior-evidence-v1/summary.json; PHASE
audit beforecommit/not FINAL Goal. GoalACTIVE/GSP01/GSP09OPEN/alloldfailures/final
scope+gates unchanged. Automatic shared changing child cones/orders/fullnativequad/
publicperiodic/wholeresources/crossover/Nscaling/physics/populations/finalfourrepo/
atleast2FINALaudits required. No production qualification.

### Complete prescribed material DAG system completed

Only originalregional import/typegate admits distinctDirectionalDAGFarActions;
entireprimaryinverseexact/12oldAST/395otherprodunchanged. Sharednear/system/native/
Riesz/Krylov kernels untouched. FourNEW controls CLOSED4PASS/rc0/XML1383.867s/
runner1385.1786307068542s/403stable inclfivehelpers+explicitrefmatrixmanifest;
session30076drained/2203044+2203200actualpsabsent. Genuine numericalbaselinef747466
retained; terminal-onlyrebase db60224e addsoldautomatictest/docs only/all396prod
interveningexact/no numericreplay/no currentfullregression. Originalmatrix1940060e
97fb5df1341c3d87b51f5577e0ee670883b839d15cfe258020f272d5 reusedinplace/no duplicate.
Two198-edge closed magneticboxes300MHz/originalq6sing3nearmax32/commonexterioreta.
Prescribedcomplete-support childtrees; eachregion12ctx/4shared/8transfers/4maps,
rootorders8/child12/L4positivechangingcarrierpaths; sixactualFAR/near34+7+7/full
physicalpaircoverageonce. Whole792-column originalweak2.77639e-16<1e-8/sixq16FAR
max1.95395e-13<1e-8; originalnearquadratureshared/compositionnotphysicalconvergence.
15noncontig fullcurrents originalH6.92767e-16/nativecurrent5.04269e-18/higher
1.22340e-16/dot2.05765e-15<1e-11/saved0. StrictMP100phase max6.33948e-15rel1e-11
abs0/systemexterior+RSS9.28022e-14 idealonly/nativequad/globalnotcertified.
GMRES+blockGMRES actualEx/Ey/fulloriginaldual+current gatesunchanged;
rtol1e-12/atol0/max768/restart384/boundedleaf32originalfactors/DAGactions no fallback.
Oneiterfailure/directfallbackrejection/physicalcoverage+capacities+actualmedium/
commoneta/saved8fullH+DAGphasemode+Rieszmutation guards PASS. Near5094936/1273632/
1273632B/maps505048each/couplings55104each/embeddings6336B/systemcomponentworkspace
152545224B excludingnearsetup/caller/allocator/FFTlibrary/notwholepeak/saving/
crossover/Nscaling. Evidence /tmp/gsp-directional-dag-system-evidence-v1/summary.json;
independentPHASE audit beforecommit/notFINALGoal. GoalACTIVE/GSP01/GSP09OPEN/allold
failures/scope/gates unchanged. Automaticsharedchangingchildcones/orders/global
nativequad/publicperiodic/wholeresources/crossover/Nscaling/physics/populations/
finalfourrepo/atleast2independentFINALaudits remainrequired. No production qualification.

Actual solver results: [{"method": "gmres", "original_dual_relative": 7.783996510071385e-13, "dense_current_relative": 3.237148175109244e-12, "iterations": 367, "fallback_used": false, "original_global_residual_certified": false}, {"method": "block_gmres", "original_dual_relative": 1.4356491055488424e-12, "dense_current_relative": 5.8676952349760104e-12, "iterations": 127, "fallback_used": false, "original_global_residual_certified": false}]

### Automatic shared directional DAG selected FAR completed

Primaryorder appendplanner; primarymoments shares exactoriginalcapacityformula,
SparseinitializerinverseASTexact/oldordermoduleprefixbyteequal/394otherprodexact.
Finitecandidate/contextscratchbounded; fixedphysicalnear/FAR+rootcones/orders;
allancestorpathproducts+allrootE+delta(M+E)/kernel/materialperpair/regionalRSS
recomputed beforemaps/coefficients; completeclosureleaf/bodymaps/uniquephasecenter
layouts/transfers/workspace scored; deterministic no rejectedFAR->near fallback.
Planningv1 CLOSED4PASS/399stable/2NEW+2necessaryoldcapacitycontrols; importedlegacy
oraclehelpernotinoriginal399inventory/honestbaseline+terminalexactnotretrofrozen.
PreliminaryP2plannerbindingpreserved/fixedonlyafterterminal; optionalboundguard
identity/config/current chain no actionkernelcopies. Actualv2 CLOSED3PASS/403stable
2NEW+1necessaryolddefaultguard. Boundaryv3 CLOSED2PASS/400stable/onlyNEW/all396prod
unchanged/actualtestexactprefix. 9distinctPASS/coveredPASS not repeated/no fullreg.
Actual3closed30-edge spheres300MHz/middley1e-6/vacuumEXTERIOR; distinctmagneticbodies
do not establishpositiveinteriorDAG. Auto65node/10FAR507near/6rootfamiliesp7+p5/
4positivechanginglinks/4sharedchildren/178ctx/166transfers/104leafcontexts;
allphysicalpairsonce/strictMP100allancestors+allrootphasekernelmaterialRSSrel1e-11
abs0max1.91695e-15. Originalq16all10selectedFAR complete180coordFAR-onlyweak4.40775e-11
<1e-8/H6.11756e-11<1e-8/nativecurrent0/higher4.65036e-16/dot5.02826e-15<1e-11/saved0.
Nearmatrixzero/NOTcompletefullMaxwellsystem/no duplicateweakarchive. Boundplanner
links/phasebudget/rootpolicy/removedguard rejectsreceipt/fwd/H/savedH. Exactmixed
p7+p5case4shared/zeroactualphase; paddedp12infeasiblepositivecasecorrectexactbaseline/
no savingclaim; finite1candidatepreservesphysicalpartition/reportslimit.
Wholeclosurecomponentcost39663504->35030768B/map4794192->3918960B/workspace34869312
->31111808B/candidatescratch1155584B/exclusionsrecorded/notwholepeak/perf/crossover/
Nscaling. Prototypeorder12/catalog32 are notGoalnarrowing. Evidence
/tmp/gsp-directional-automatic-dag-evidence-v1/summary.json/PHASEauditbeforecommit
notFINALGoal. Independentcovariancev4 CLOSED1FAIL72208.719s/715stable/handle98167
drained/actual846722+846726absent/DyadicConvergenceErrorbeforecovariancegate/no
restart/identitynotcollectednotinvented. Originaltensorcell+publictensorACLIVE
unchanged. GoalACTIVE/GSP01/GSP09OPEN/alloldfailures/scope/gatespreserved. Complete
automagneticnear+FAR/solves/nativequadglobal/publicperiodic/tensornonlinear/
wholeresources/crossover/Nscaling/physics/populations/currentfourrepo/atleast2FINAL
independentaudits remainrequired. No production qualification.

Terminal collectors: [{"path": "/tmp/gsp-directional-automatic-dag-planning-v1", "tests": 4, "XML_seconds": 78.128, "runner_seconds": 79.61735257599503, "source_count": 399, "status": "CLOSED", "returncode": 0}, {"path": "/tmp/gsp-directional-automatic-dag-actual-v2", "tests": 3, "XML_seconds": 408.961, "runner_seconds": 410.199377710931, "source_count": 403, "status": "CLOSED", "returncode": 0}, {"path": "/tmp/gsp-directional-automatic-dag-boundaries-v3", "tests": 2, "XML_seconds": 8.151, "runner_seconds": 9.263014268130064, "source_count": 400, "status": "CLOSED", "returncode": 0}]

### Complete automatic directional DAG discrete system completed

Test/evidenceonly/all396prodexact76927fc/401frozen completehelperclosurestable.
Session75825 CLOSED1NEWPASS/XML585.093s/runner586.4474143227562s/rc0/drained/
2294416+2294639actualpsabsent; coveredFAR/PASS/fullreg not repeated.
Three30-edge radius.02 closed spheres300MHz/x0,1,2middley1e-6/eps2,3,4mu1.3,1.4,
1.5sig_e.01sig_m.02/vacuum. Originalnearq6sing3pair4far16max32/tolerance/commoneta
unchanged; auto leaf4ratio.15cube2^20/exterior10FAR507near/4positivechanginglinks/
4sharedcontexts; eachactualmaginterior169nearZERO FAR/notpositiveinteriorevidence.
Full180-column originalweak assembledonce16edgetiles/no existingfulloracle/no
matrixarchive; weak2.24047e-15/H2.51689e-15<1e-8/nativeH4.98176e-16/current7.41389e-17/
higher1.59685e-16/Riesz9.38182e-17<1e-11/15noncontig/saved0. Wholeoriginalnearquad
shared/discretecompositionnotconvergedphysics. Planlinks savedcompleteH+receipt
guards PASS; solveacceptanceNOTRUN. Idealexterior+RSS6.31969e-6 excludesfinitequad/
native/globalcert;embedding1440B/systemcomponentworkspace85952048B/exclusions
recorded/notwholepeak/perf/crossover/Nscaling. Evidence
/tmp/gsp-directional-automatic-dag-system-evidence-v1/summary.json/independentPHASE
beforecommit/notFINALGoal. GoalACTIVE/GSP01/GSP09OPEN/allfailures/scope/gates
unchanged. PositiveautoMagInteriorDAG/solves/publicperiodic/tensornonlinear/
nativequadglobal/physics/wholeresources/crossover/Nscaling/populations/current
fourrepo/atleast2FINALindependentaudits remainrequired. No production qualification.

### Guarded automatic DAG path admission reuse completed

Primarymoments2newbranches: exactnumericalkeys/supportbeforemaps plus borrowed
immutableplannerCUDAallancestorarray afteroriginalworkspace/currentguard. Default
prescribed traversal/maps/transfers/FFT/actionkernels unchanged; inversewholeprimary
byteexact76927fc/395otherprodexact. Bothcollectors genuinelytested76927fc; intervening
533f5928tests/docs only/all396prodbyteexact; originalsourcehashesretained.
v1CLOSED2NEWPASS/XML38.473s/runner39.709867808036506s/401stable; emptyv2CLOSED1NEWPASS/
XML3.566s/401stable/onlyNEWtestappend/exactv1testSHAsnapshot; handles77795+69516drained/
actualPIDsabsent/installedCore/GPU1. No oldPASS/q16/fullregreplay.
Actual3sphere90-edge exterior borrowedvsuncachedrel1e-11abs0/MP100allancestorsmax
6.64201e-16/exactzero0/32requestsidenticalarray/no2ndcache. Pathmode/identity/links
receipt/query/savedsourcegradient guards andpathversion savedreceiver guard PASS.
Actual30-edge magneticinterior ZERO FAR/emptyCUDAidentity/0ownedmapbytes/workspace
andmodeguards PASS/notpositiveinteriorFAR. torch.equal exactnumericalnotbyteidentity.
Map3918960B/componentworkspace31111808B unchanged; plannerownedstorageexternal
dependencycountsaggregate/notwholepeak/perf/crossover/Nscaling qualification.
Evidence /tmp/gsp-directional-admission-path-cache-evidence-v1/summary.json;
independentPHASE auditbeforecommit/notFINALGoal. GoalACTIVE/GSP01/GSP09OPEN/allold
failures/scope/gatesunchanged. PositiveautoMagInteriorDAG/solves/publicperiodic/
tensornonlinear/nativequadglobal/physics/wholeresources/crossover/Nscaling/
populations/currentfourrepo/atleast2FINALindependentaudits remainrequired.
No production qualification.

### Public automatic scalar directional DAG integration completed

DirectionalFMMConfig strictnewversion/allfieldsbind; existingCartesianpayloads
unchanged. Primaryautomaticglobalfactory G/2 exterior+G/(2sqrt(bodycount)) eachactual
interior, exterior+RSS checkedbeforemaps/near and embeddingcapacitybeforeplanning;
primarynative map/FFT/actions/Riesz/Krylov reused. PublicScene->Simulation->Result
free-space scalarcomplex128/iterative nofallback only; periodic/tensor/HB remain
requiredOPEN. Auto/prescribedmetadata typedowners safe.390otherprodbyteexact5efbc2d/
inverseexistingprimarypaths and ASTproof/threecollectors396prodidentical.
FourcollectorsCLOSEDrc0/stable398+401+397+473/10distinctPASS=7NEW+3necessaryaffectedold;
coveredq16/native/fullregnotreplayed. QuickpytestPIDsnotobserved/notinvented;
originalhandlesdrained/knownrunnerabsent/childwaitedrc0. Publicactual2355117+2355251
absent/36704drained. Public3sphere90edge/180currents300MHz/eps2,3,4mu1.3,1.4,1.5/
sig_e.01sig_m.02/q6sing3pair4far16nearmax32tol1e-5; exterior10FAR/4positivechanging
links; threeallnear magneticinteriors ZERO FAR/notpositiveinteriorqualification.
Originalbounded16leaf GMRES115iterations/rtol1e-12atol0max240restart120/nofallback;
compresseddual1.11552e-12/original8edgesample1.57532e-12<1e-8/notglobalresidual/current
error/physicscertificate. ExactResultcurrents/configrestore withconstructors/solve
forbidden; policytamperrejected by numerical/operator/RHSidentity/schema11.
Bench69produnchanged; independentphysical/configdeclaration + nativeidentity,
exactJ/Mbytesarchive/reusedproducerSHAee789a6f2b369fc1fb108dc0ef28859436871a8e5c4ec72de8b670e5bd9c1c62/
noadditionalassemblyor solve/unqualifiedcandidateonly. Regionalcomponentcapacities/
plannerexternalstorage/notwholepeak/perf/crossover/Nscaling. Evidence
/tmp/gsp-directional-public-evidence-v1/summary.json/independentPHASEbeforecommit/
notFINALGoal. GoalACTIVE/GSP01/GSP09OPEN/alloldfailures/scope/gates preserved.
PositiveautoChangingMagInteriorDAG/publicperiodic/tensornonlinear/nativequadglobal/
physics/wholeresources/crossover/Nscaling/populations/currentfourrepo/atleast2FINAL
independentaudits remainrequired. No production qualification.

### Guarded automatic DAG integral statistics reuse completed

Two primary changes bind planner statistics identity/version/mode and borrow its
existing CUDA physical-tree integral statistics after owner validation. No second
owner statistics cache. Unguarded traversal and integral/map/transfer/M2L/FFT/near/
complete action/public solve kernels unchanged. Inverse entire two primary modules
byteexact23457af1; other394 production modules byteexact. No covered PASS replay.
Collector /tmp/gsp-directional-integral-statistics-reuse-v1 CLOSED3NEW PASS;
XML30.128s/runner31.344613726250827s/rc0/402stable fullhelperclosure;
original87673drained/observed2400148+2400310absent/installedCore/GPU1.
Actual3sphere90edge exterior all65physicalnodes vsuncached rtol0/atol0 numerical
exact; MP100 allnodes+all10actualrootmaterialbounds maxrelative3.799433498426853e-15
<1e-11/abs0. 64queries identicalplannerarray/aggregationforbidden/map3918960B
unchanged/no2ndcache; externalplanner1560B countsaggregate/notwholepeakclaim.
Statistics identity/mode savedsource/receipts/query guards and version savedreceiver
backward guard PASS. Actual30edge magneticallnear15nodes/emptyDAGcontexts/nonempty
CUDAstats/uncachedexact/0ownedmapbytes/external360B/modeguard PASS; no positive
magneticinteriorFAR qualification. Exactcomparison notgeneraldeterminism orbyteproof.
Evidence /tmp/gsp-directional-integral-statistics-reuse-evidence-v1/summary.json;
independent PHASE audit beforecommit/notFINAL. No measured speed/wholeresource/
crossover/Nscaling qualification. GoalACTIVE/GSP01/GSP09OPEN/alloldfailures/scope/gates
unchanged. PositiveautochangingmagneticDAG/globaloriginalresidual/periodic/tensor/
nonlinear/nativequadphysics/whole resources/crossover/Nscaling/populations/current
fourrepo/atleast2 independentFINALaudits remainrequired. No production qualification.

### Complete directional retained dependency inventory completed

Primaryfmm_capacity appends two nonnumerical inventory functions; originalmodule
prefixbyteexact42896b3/395otherprodexact/componentreceipts/numericalkernels unchanged.
Completeactualphysicalinputs/preparedgeometry/alltrees/rootplans/boundDAGcarriers/
paths/stats/allmapowners inclfar.targets+origins/caller carriers/sharedlayouts/M2L/
originalnear/embeddings included. Actualdevice+backingpointerdedup/fullviewstorage;
disjointattributedcategories/overlappingstandalone. No cachedcopy orphysicsrecompute.
Workspaces/callercurrents/Rieszsolver/allocatorlibrary/host excluded/notpeakclaim.
v1CLOSED2NEWPASS1FAIL57.303s/401stable/96777drained/2416566+2416773absent;
leaf-onlyNestedlayoutNone fixedafterterminal; onlyfailedv2CLOSED1PASS7.278s/401stable/
7681drained/runner2449741absent/pytestPIDhonestlynullquickcompletion/childwaited.
IndependentpreliminaryP2 DAGcaller_carriers_missing preservedandfixedafterterminal;
only2NEW guarded+unguarded independentoversizedcarrierstorage controls v3CLOSED2PASS/
108.685s/401stable/66590drained/2763199+2764071absent/installedCore/GPU1.
Independentactualreachableobjectwalk equality automatic3sphere90edge4regions
4416024B/2893storageidentities; Nested2body12edgesalternatecouplingowner+sharedlayout
1039904B/186; unguardedcaller4406648B/2742/guarded4416360B/2897 withfull192+48+48+48B
backingstorage inclnonempty+emptyviews. Dense518400B/9216B respectively: no crossover.
Exactv1+v2primary andoriginaltest snapshots retained; inverse2fixes wholeprevious
modules exact/original3testprefixexact/no oldcollectorhash relabelling/no PASS replay.
Evidence /tmp/gsp-directional-retained-inventory-evidence-v1/summary.json;
finalindependentPHASE audit beforecommit/notFINALGoal. GoalACTIVE/GSP01/GSP09OPEN;
oldfailures/finalscope/gatesunchanged. Fullnativequadphysics/globaloriginalresidual/
positiveautochangingmagneticDAG/periodictensornonlinear/whole resources/crossover/
Nscaling/populations/currentfourrepo/atleast2FINALindependentaudits required.
No production qualification.

### Shared directional action validation traversal completed

PrimaryFFT extracts unchangednumericalbody behindoriginalstandaloneentryguard;
primaryDirectionalRWG uniquecompleteprototype/targets/origins ownerboundary checks
plusalltranslationtable/material/context/modeguards, then internaldirectnumeric
translationdispatch. FAR/systementry+exit/native savedcurrent guards retained.
No validationcache/newtensor/formula/workspace change. Inverseentire2primarymodules
byteexactd95e793/394otherprodexact. Actual10FAR/1owner boundary24→1 visits only;
no timing/scalingqualification. PublicstandaloneM2L fullcontextguards unchanged.
v1CLOSED5PASS1FAIL/XML471.841s/runner473.39361087139696s/404stable/54134drained/
2862959+2865011absent/installedCore/GPU1. Uncontrolled exact180cols FAIL106entries/
32400/maxabs6.308948006325748e-29/maxrel6.573500093894881e-05 preserved/notPASS.
Onlyfailednumerical+NEWportableoraclelifecycle v2CLOSED2PASS/XML705.607s/runner
707.3861014246941s/402stable/89329drained/3129032+3131424absent. Deterministiccolumns
exactunchangedrtol0/atol0 then previousenabled/warnflagsrestored; defaultsubsequent
15noncontigH/current/higher/Riesz/dot max9.485997359820047e-16<1e-11/saved0/
retainedinventorysame. Defaultelementwise determinism unqualified. No originalweak
matrixreintegrated/archive/no coveredPASSreplay. Fixture3bodies90edges/allmagintsnear.
Savedwhole+standaloneM2L statistics/path/table/context identity/mode guards and
statisticsversionafterexterior caughtbefore systemreturn PASS. Necessaryaffectedold
2CartesianFFTmaterial+1Nestedreverseowner failure/configcontrols PASS.
PreliminaryP2 tmporacledependency preserved; exactbaselineAST557fd963…3233c now
bundledrelative tests/mom/fixtures/directional_action_validation_legacy_far.py;
NEWavailability/SHA/names/compile/lifecycle PASS. Inversetestchanges restoresgenuine
v1snapshot/all396productionidenticalbothcollectors.4NEW+3necessaryold=7distinctPASS.
Evidence /tmp/gsp-directional-action-validation-evidence-v1/summary.json;
independentfinalPHASE auditbeforecommit/notFINALGoal. GoalACTIVE/GSP01/GSP09OPEN/
oldfailures/scope/gatesunchanged. PositiveautochangingmagneticDAG/globaloriginal
residual/nativequadphysics/periodictensornonlinear/whole resources/repeatedtimings/
crossover/Nscaling/populations/currentfourrepo/atleast2FINALaudits required.
No production qualification.

### Configured native directional carrier catalog completed

Primarypublicconfig/region remove hard32 bound while retaining typed configured capacity/default32;
DAG reachable catalog obeys root configured max_families, including deliberate lowerthan32 closure constraint.
Shared Sparse/DAG projection nativeCUDA exactbit/order unique replaces pairwise duplicate loop;
signedzero/order distinct; successfulbits/degrees/keys released before mapcapacity;
4096+128C tensorcatalog allowance admittedbeforeunique, maxwithcontext allowance, library/allocator excluded/notmeasuredpeak.
Inverse3entireprimarymodules byteexactb6b949f/393otherprodexact; kernels/material/current/nearformulasunchanged.
Actual9closedtetra54edges72rootFAR/72root+DAGfamilies configured512;
all72rootMP100 max5.0230605398192025e-15<1e-11/abs0;
family71originalq16weak7.489663898904781e-13/H8.886222954509061e-13<unchanged1e-8;
15noncontig nativeforward/H/current/adjointcurrent/higher/dot max9.23843472041669e-15<1e-11/saved0.
Mapowned9570240B componentonly/notcompleteinventory/wholepeak/crossover.
v1CLOSED3PASS2FAIL/XML55.135s/runner56.680527531541884s/400stable/session53046drained/
3871275+3871452absent; originalmissingstopping+nonexistentmapinterception testfailurespreserved.
Onlytwofailedtestbodies fixed afterterminal; v2CLOSED2PASS/XML6.570s/runner8.169815125875175s/
399stable/session31640drained/runner3953504absent/quickpytestPIDnotobservedhonestNULL/childwaited.
396productionidenticalbothcollectors/passedtest+helper+fixtureASTunchanged;
failedcapacityfixtureonlyphysicalroot omitscoveredDAGmaps/no coveredphysicalPASS replay.
4NEW+1necessarychangedold=5distinctPASS/installedCore/GPU1both;
actualdefault32capacityreject/noFARrelabel/duplicatekeys/preuniqueworkspacerejectbeforemaps;
nativeunique signedzero/order+temporaryrelease lifecycle PASS.
Original3prelaunchP2 preservedinprelimv1/fixedbeforelaunch/clearprelimv2.
Evidence /tmp/gsp-directional-configured-catalog-evidence-v1/summary.json;
independentfinalPHASEauditbeforecommit/notFINALGoal. GoalACTIVE/GSP01/GSP09OPEN/
alloldfailures/finalscope/gatesunchanged. Completeglobaloriginalresidual/current/nativequadphysics,
positiveautochangingmagneticDAG/periodictensornonlinear/wholeresources/repeatedtimings/crossover/
allpointNscaling/populations/currentfourrepo/atleast2FINALaudits remainrequired. No production qualification.

### Original reciprocal scalar near packing completed

Primarypackednear new explicit reciprocal alignedsquare/free-space fixedscalar primitive;
allreversepair/JM/normalizedphysicalnode maps validated BEFOREdiscardinganyrecords;
canonicaloffdiagfirst/fullindependentdiagonals, reverse D/K byT notH;
sharedoriginalmaterial/adjoint/current kernel, actionbatchboundedreverseindexscratch,
no extra retainednumeric/cache/globalmatrix. Policy/count/id/version/mode/savedcurrent guards.
Inverseentireprimary byteexact56a937a/395otherproductionexact.
GPU1/installedCore originalsession66280 drainedrc0/235674+235878absent;
v1CLOSED5PASS/XML10.438s/runner11.845809794962406s/398stable/4NEW+1necessaryaffectedold.
Actual2closedtetra12edges externalinterleaved5/4/3rectanglemaps and6edge lossymagint2/2/2
samegroupdiagonal/reverse:9logical/6stored/6integrationcalls each;
all9ordered originalindependentD/K reference maxbalancedweak2.3571155598893304e-16/
originalH2.7138230498049454e-16<unchanged1e-8;15noncontignativeforward/H/current/
adjointcurrent/higher/dot max6.814520471709526e-16<1e-11/saved0.
Primitiveexterior6032→4072B/int2000→1352B exactcompactdisjointstorage/notwholeinventory/crossover.
Onebytebelowpreallocation/missingreverse/duplicate/JMmap/strictnodeoffsetrejectbeforefactory;
no-diagonal4reverse-consistentpairs inconsistentnormalizednode1mapsrejectprecanonical.
InitialstaticP2 namespace-mapbindingpreserved/fixedbeforelaunch/prelimv2clear.
No coveredPASSreplay/no oldfullmatrixreintegrated/archive.
Evidence /tmp/gsp-reciprocal-scalar-near-evidence-v1/summary.json;
independentfinalPHASEauditbeforecommit/notFINALGoal. Internalpackingonly; publicCartesian/
directionalconfiguration/capacity/factory/periodicexclusion/exactstate next. Original3840resources
failurespreserved; wholepeaks/repeatedtimings/crossover/allpointNscaling/globaloriginalcurrent/
residual/nativequadphysics/periodictensornonlinear/populations/currentfourrepo/atleast2FINALaudits
remainrequired. GoalACTIVE/GSP01/GSP09OPEN/alloldfailures/finalscope/gatesunchanged.
No production qualification.

### Public reciprocal scalar Cartesian FMM integration completed

`Scene -> Simulation.mom -> Result` accepts `FMMConfig(near_storage='scalar_reciprocal_dk_v1')`. Strict payload `cartesian_fmm_reciprocal_dk_config_v1` binds all explicit parameters. Existing default full-matrix and ordered scalar-D/K identities remain unchanged. The new policy is free-space scalar float64 geometry/complex128 currents only; tensor and periodic extensions remain required separately. Dense/compact/FFT tables and dense/separable shifts retain their existing semantics. Public Ewald/periodic and complex64 cases reject before setup.

The existing regional factory requires identical test/source physical basis and tree for reciprocal near storage, uses the original independent D/K integration callback and delegates to the primary PackedReciprocalScalarNearChannels. One off-diagonal direction is stored; reverse kernels use ordinary transpose, while diagonals stay full and Hermitian conjugation remains separate in the common adjoint. Whole action retained admission counts canonical near pairs, maps and CSR before allocating moment maps. Every non-near category is unchanged. The exact retained receipt excludes input geometry, Riesz/solver state and setup/action/backward workspace; it is not a whole peak certificate.

Inverse proofs recover all five primary modules byteexact to 6cf4638;391 other production modules, including the numerical packed primitive, remain byteequal. All396 production hashes are stable across both Maxwell collectors. Bench69 production modules are unchanged. New archive coverage uses existing native numerical/operator/RHS bindings, with no duplicate numerical implementation.

Original collector v1 CLOSED4PASS1FAIL (XML55.325s/runner56.895664491690695s). Its original 12-edge two-tetra integration fixture has only diagonal near pairs, so the compression assertion failed before numerical comparison. This coverage failure and the original tested test bytes are preserved. After the original handle drained and actual PIDs were absent, only the failed test changed and one new 60-edge two-sphere fixture was added; all passed test/fixture ASTs and production bytes are unchanged. Failed-only v2 CLOSED1PASS (XML181.116s/runner182.60391335748136s). No old PASS was replayed. Both399 frozen inputs remained stable. Their observed runner/pytest PIDs are absent and original handles drained.

The new full operator fixture uses two closed radius.01 spheres at x=+/- .5,300MHz, eps2/3, mu1.3/1.4, electric conductivity.01 and magnetic conductivity.02. Original near q3/singular3, order8/quadrature12, leaf2/separation.25, FFT/separable and existing budgets remain explicit. All120 columns of the original independently assembled full system pass balanced relative1.3300657565261127e-15 and original H1.4490868705064444e-15 against1e-8 gates. Native15-RHS forward/H/current/adjoint-current/higher/dot maximum6.08908883020928e-16 passes1e-11; saved action intermediates are zero. Exterior1458 logical near blocks become756 stored; interiors900 become465 each. Exact owned retained2,156,512B equals admission, versus ordered-D/K2,302,528B; only near304,680B becomes158,664B. The actual compressed case rejects capacity minus one before maps. Shared original near quadrature is not a convergence oracle; no physics or crossover qualification follows.

The original v1 public solve already passed and is reused once: two spheres60edges at +/- .2,90MHz, eps2/3 and mu1.2/1.3, q4/singular3, automatic order6..10/G1e-6/H1e-5/direct-first-wave, leaf8/separation.25/FFT/separable, original bounded16-edge primary factors. GMRES rtol1e-12/atol0/max240/restart120 converges in93 iterations. Compressed dual2.1070537409796903e-13 and original eight-edge sample3.020825082849042e-13 pass1e-8. The sample is not a global original residual certificate. Exterior242 becomes132 stored (46,920B), each interior64 becomes36 (21,640B). Exact currents/configuration restore with constructors and solve forbidden, and changed near/table/bound/shift policies reject.

Completed producer SHA256 0f7e0fa8e7a3f1079ca9eb7404dc6f266c3f4ce71d204f06221a6310179b3949 is reused by a new Bench control without another solve or producer save. Independent Scene/full numerical declarations precede loading. J/M archive bytes are exact; four changed policies reject in both native restore and Bench manifest validation. Native schema11 results remain unqualified candidates. Bench collector CLOSED1PASS/XML3.857s/runner5.340552922338247s with473 stable source/producer inputs. Its quick pytest PID was not observed live and is not invented; recorded runner absent, original handle drained, child waited rc0.

Six distinct PASS consist of four NEW Maxwell controls, one necessary affected old public scalar-scope control and one NEW Bench archive control. This is selected phase acceptance, not current full regression or FINAL Goal audit. Independent PHASE audit precedes commit. Shared GPU timings are not a performance benchmark. Original3840-edge crossover/peak failures remain preserved; whole setup/warmups/differentiable forward/backward/solver peaks, all-point scaling, directional reciprocal integration, periodic/tensor/nonlinear paths, populations, current four-repository final acceptance and at least two independent FINAL audits remain required. Goal ACTIVE; GSP01/GSP09 OPEN; original scope/failures/gates unchanged. No production qualification.

### Public reciprocal directional near storage integration completed

`DirectionalFMMConfig(near_storage='scalar_reciprocal_dk_v1')` selects the primary reciprocal original D/K near factory through `Scene -> Simulation.mom -> Result`. The strict full-field payload is `automatic_directional_reciprocal_dag_fmm_config_v1`. Default ordered-D/K payloads omit the new default field and retain `automatic_directional_dag_fmm_config_v1` byte identities. The public policy remains free-space scalar float64 geometry/complex128 currents with iterative no-fallback solves; tensor, periodic and nonlinear directional paths remain open requirements.

The primary structural helper validates native float64 physical terminal nodes, unique pairs and complete reverse closure, then counts canonical a<=b kernels, full diagonals, maps and CSR. Both automatic Sparse and DAG builders check reciprocal capacity before map or original integration factories. The complete regional factory separately checks exact full near/FAR coverage and identical physical basis/tree, then delegates to PackedReciprocalScalarNearChannels without changing FAR maps, tables, plans or global ideal bounds. Reverse D/K use ordinary transpose; Hermitian material/current composition remains independent. Added bounded reverse-index workspace is explicit and still excludes setup, allocator and backend library peaks. The near policy participates in saved-current regional guards and appears only for the new mode in receipts.

Five primary whole-module inverse proofs recover baseline8ecd696;391 other production modules, including all numerical FAR and packed D/K kernels, remain byteequal. Three Maxwell collectors retain the identical396 production hashes;69 Bench production modules are unchanged. Existing default configuration identity is the one necessary affected old control. No previously covered FAR/native-quadrature/full-original assembly or old solve is replayed.

The original collector CLOSED4PASS1FAIL/XML1817.850s/runner1819.5387367866933s/401 stable inputs. Its configuration test mistakenly used StoppingRule's row_norm default, rejected by the existing public FMM contract. The original tested source and failure remain preserved. Only the failed function changed after the original handle drained and actual708736/708896 were absent; all passed functions and the physical fixture retain exact ASTs and all production remains unchanged. Failed-only v2 CLOSED1PASS/XML3.268s/runner4.508753350004554s/406 stable inputs. The additional changed Sparse-builder negative control CLOSED1PASS/XML3.969s/runner5.425949591211975s/400 stable inputs, without constructing maps, original near or a positive Sparse FAR action. These two quick pytest PIDs were not observed live and are not invented; recorded runners absent, original handles drained and children waited rc0.

The actual three-sphere fixture has90 physical edges/180 currents, radius.02 at (i,1e-6*(i%2),0),300MHz, eps2/3/4, mu1.3/1.4/1.5, electric conductivity.01 and magnetic conductivity.02. Original q6/singular3/pair4/far16/nearmax32, leaf4/separation.15/cone2^20/quad12 and all existing budgets remain unchanged. Exterior has four positive changing-carrier links; all three magnetic interiors have zero FAR. Exterior507 logical near blocks become273 stored (74,864B); each interior169 becomes91 (24,960B). Exact retained complete native dependency inventory4,416,024B becomes4,292,472B, saving123,552B. This inventory includes shared inputs/plans and excludes workspace, solver/Riesz and allocator; it is not a whole peak or crossover certificate.

All180 complete columns compare the new reciprocal action with independently integrated original ordered D/K near blocks and the identical shared FAR owners. Balanced relative1.5446254372149055e-16 and independently integrated ordered-near H5.258060311068895e-16 pass unchanged1e-11 comparison gates. Native15 noncontiguous RHS forward/H/current/adjoint-current/higher/dot maximum7.656287059748157e-16 passes1e-11; no internal tensors are saved by current backward. Changed near policy rejects receipt and saved backward. This controlled composition comparison does not repeat or certify original full-matrix FAR quadrature or physical convergence. Its explicit shared-FAR limit is retained in test and evidence metadata.

The public GMRES solve uses unchanged original bounded16-edge factors, rtol1e-12/atol0/max240/restart120 and converges in115 iterations without fallback. Compressed dual1.115405123955778e-12 and original eight-edge sample1.5741683304210806e-12 pass1e-8. The sample is not global original residual/current error/physics certification. Exact currents and configuration restore with constructors and solve forbidden; near/candidate/phase/cone policy changes reject. Native physical schema11 stores the physical result, not an executable compressed operator.

Producer SHA256 aa0937fa56d6dd7291ce872dfefe65f87d65e15f256297e7a4c9abb45301ff24 is reused once by the independently declared Bench Scene/full configuration. J/M bytes are exact; the four policies reject both native restore and the independently bound Bench manifest. No producer save, assembly or solve is repeated. Bench CLOSED1PASS/XML4.078s/runner5.825879503972828s/473 stable inputs. Its quick pytest PID is honestly unobserved; runner770814 absent, original handle51596 drained, child waited rc0. The archive remains an unqualified candidate.

Seven distinct PASS are five NEW Maxwell controls, one necessary affected old configuration control and one NEW Bench archive control. Independent PHASE audit precedes commits, without numerical reruns. This is not current full regression or FINAL Goal audit. Original3840-edge reciprocal resource collector is separately LIVE on its frozen8ecd696 provider; no source, order, budget or failure is changed here. Its resources and all previous failures remain separate. Positive automatic changing-carrier magnetic FAR, general tensor/periodic/nonlinear paths, converged quadrature/native/global physics, whole setup/actions/backward/solver resources, all-point scaling/crossover, populations, current four-repository final acceptance and at least two independent FINAL audits remain required. Goal ACTIVE;GSP01/GSP09 OPEN; original scope/failures/gates unchanged. No production qualification.

### Native reciprocal compact translation primitive completed

The primary `CompactElectromagneticM2L.reversed_translation()` returns the opposite free-space displacement action while borrowing the original coefficients, indices and binomial weights. Radial Taylor coefficients change by degree parity under r -> -r; generated Green tables apply that parity and gradient tables apply one further minus. Complex wave number and electromagnetic material factors are never conjugated by displacement reversal. The existing Hermitian action independently conjugates the full material/kernel algebra. No reverse coefficient vector, dense retained table or numerical layout is allocated by construction. The reverse receipt distinguishes shared dependency bytes from zero additional owned tensor bytes.

Owner identity, fixed material/configuration/tensor identity and versions, plus borrowed requires_grad modes participate in reverse guards. Saved-current backward re-enters the guarded original and reverse actions. Reverse-of-reverse construction and differentiable coefficient borrowing fail closed. Existing default state, receipt and numerical branches remain unchanged; one complete primary-module inverse proof recovers870d0eed, and395 other production modules remain byteequal. Public FMM configuration, topology, regional assembly and capacity planners have not been changed by this internal primitive stage.

The original GPU1/installed-Core collector CLOSED6PASS/no skips/XML5.003s/runner6.511025075800717s with399 stable sources. Five NEW controls comprise orders0/3/6 against the independent original dense factory at the opposite displacement, MP100 analytic radial derivative samples, and saved owner/mode/version guards. One necessary affected original compact order0 control passes. The short child completed before live PID observation; its pytest PID is honestly NULL. Execution runner1048008 is captured, original handle3042 drained rc0, outside ps confirms runner absent and the runner waited its child rc0. No previous complete FMM, solve or resource control is rerun.

Lossy magnetic scalar medium uses eps1.5/mu1.3, electric conductivity.02/magnetic.03 at300MHz and displacement(1,.3,-.2). All generated table entries agree with opposite-center original dense tables: max Green relative0, gradient7.128781524525757e-17. Fifteen noncontiguous native RHS per order pass forward/H/current/adjoint-current/higher/dot at1e-11, maximum2.5930417383723824e-15, with no saved internal tensors. MP100 independently differentiates exp(ikr)/(4*pi*r) at both centers for nine degree0-3 samples, max relative1.0179553828707576e-15<1e-11. Unique two-direction retained dependencies are104/2656/11760 bytes, exactly one original coefficient/layout inventory rather than208/5312/23520 independently owned bytes. Constructor forbids Taylor/layout generation and changes no live CUDA allocation; this is not a full setup/action peak bound.

Independent read-only PHASE audit precedes commit. This primitive does not yet enter public FMM construction/capacity, and it does not certify full assembled RWG errors, native quadrature, physical convergence, high-order/full-system resource peaks, solver resources, scaling or crossover. The separately completed original3840-edge reciprocal-near resource result remains bound to its frozen8ecd696 provider; its full allocator peak failure is not closed by this primitive. All original failures, scope and gates remain unchanged. General tensor/periodic/nonlinear paths, positive automatic magnetic changing-carrier FAR, all-point resource/complexity acceptance, Bench populations, current four-repository acceptance and at least two independent FINAL Goal audits remain required. Goal ACTIVE;GSP01/GSP09 OPEN; no production qualification.

### Original reciprocal-near full GPU resource measurement completed

One NEW original fixed-ka resource control completed on the frozen8ecd696 provider. The two spheres retain radius.01 at x=-.5/+.5, subdivisions3/3840 physical edges, vacuum exterior/eps2 mu1 interior,300MHz, admitted p11, positive_tail_min_v1 G1e-6/H1e-5, leaf30/separation.25, original q4/singular3, compact tables/separable shifts and all original1GiB/2GiB budgets. The completed original order plan and ordered-DK resource are hash-pinned; automatic replanning is forbidden. Only near storage selects scalar_reciprocal_dk_v1. All396 production modules equal8ecd696; the new test borrows only the existing original whole-interval measurement helpers, whose complete source is frozen. No old test, original all-column reference, solve or producer save is replayed.

Original handle77232 drained rc0; actual observed runner665864/pytest666067 are absent outside the sandbox. Collector CLOSED1PASS/no skips/XML4137.406s/runner4138.9490486979485s/412 stable frozen sources. Installed Core, Torch2.13.0+cu130, mpmath1.3.0, RTX A6000/GPU1 execution identities are captured. The test PASS covers faithful measurement, structural equality, native finite actions and fixed-current derivative gates; it does not mean resource crossover PASS.

Owned retained tensors equal the complete structural plan795,719,008B versus original ordered-DK1,051,291,168B and same-N full complex128 dense943,718,400B. All non-near categories remain exact: table354,688,320B, leaf moments178,913,280B, shifts1,908,384B, with the same tree/center/map/body categories. Near259,874,344B replaces515,446,504B, saving255,572,160B exactly. Exterior22,050 logical blocks become11,130 stored/126,173,720B; each interior58,136 becomes29,192/66,850,312B. The retained-only comparison falls below the dense storage number at this one point, not an all-point production crossover.

Two native RHS pass full forward/H dot9.583499407993958e-16 and actual current-gradient6.380181322921783e-17 at unchanged1e-11. The allocator interval begins before construction and includes setup, all warmups, actual forward/H, differentiable forward, saved-current backward, peak-reset gaps and final finite checks. Whole peak allocated965,273,088B and reserved1,048,576,000B both exceed the unchanged same-N dense storage943,718,400B: the resource peak gate remains FAIL. Prior ordered-DK peaks1,224,886,272B/1,317,011,456B are preserved. The setup allocated934,558,208B alone would miss the later actual backward peak and cannot be used as a full certificate. No allocator reset, omission of warmup/differentiable forward, loosened gate or retained-only substitution is made.

Setup1695.65195s; warm forward406.51509s/H411.42072s; measured forward400.54230s/H402.51946s; differentiable forward387.51176s/backward396.79543s. These shared-GPU timings are observations, not isolated performance or N-scaling certification. Absolute current-process allocator peaks include provider/caller allocations live in the fixture; retained action capacity excludes provider geometry, solver/Riesz and allocator workspace. The interval has no solver or teardown and is not an all-device/external-library resource certificate.

Later incoming main8acdafe source applicability is recorded separately, never relabelled as the tested source:390 modules are byteequal; five directional-only changes and one compact opposite-displacement extension have complete whole-module inverse proofs. This original Cartesian resource path never calls reversed_translation; its original compact coefficient/layout/material/action and default state/receipt branches remain unchanged. Independent PHASE audit checks that chain and the planned append-only merge. No production code changes in this resource evidence stage. The isolated8ecd696 test/docs commit is integrated onto8acdafe by preserving both earlier phase entries and appending this resource entry once; all396 incoming main production hashes remain unchanged.

The new measurement reduces but does not close resource peak/crossover failure. Original GSP01/other physical failures remain unchanged. Whole solver resources, all-point fixed/growing/high-frequency scaling and crossover, tensor/periodic/nonlinear paths, positive automatic magnetic changing-carrier FAR, population acceptance, current four-repository full acceptance and at least two independent FINAL Goal audits remain required. Goal ACTIVE;GSP01/GSP09 OPEN; no physical, complexity, resource or production qualification.

### Public exact-displacement shared compact FMM integration completed

`FMMConfig(table_storage="compact_shared_displacement_v1")` selects exact same/opposite native displacement sharing within each physical region, medium and uniform expansion order through Scene/Simulation/Result. Strict full-field payload `cartesian_fmm_shared_compact_config_v1` binds the new table representation and all near policies. Existing default/compact/FFT/near-reciprocal payload identities remain unchanged. Public configuration and internal periodic system reject this new free-space mode for periodic kernels; native float64 geometry/complex128 currents are required. No tensor or nonlinear FMM qualification is added.

The primary catalog processes at most32pairs per geometry/key tensor. CUDA float64 center subtraction and exact IEEE integer keys choose one representative for equal or opposite displacement; signed-zero coordinates normalize for key identity only, while a1ULP displacement difference never merges. Only integer bits, orientation and topology transfer to host control. No approximate bins or additional numerical retained cache exist. Logical FAR pairs, order, physical projection/context and traversal order remain intact, including partitions without physical reverse closure. All original ordered separation/kernel-error/RWG bounds remain independently checked. Representatives generate original compact coefficients in batches at most32; reverse uses the already verified guarded degree-parity primitive. Each original/borrowed owner preserves material, identity, version, mode and saved-current guards.

Capacity and both plain/RWG order planners count actual distinct coefficient/layout storage before numerical setup, rather than logical-pair bytes. All ordered kernel bounds and RWG accumulation remain unchanged. Region checks cumulative table capacity before coefficients; system checks whole owned capacity and new table budget before moments. Native retained storage equals exact structural planning, with shared tensor pointers counted once. Near original integrals, maps, diagonal values and all existing electromagnetic/current kernels are unchanged. Nine complete primary-module inverse proofs recover638a2a89;387 other production modules, including compact parity/math kernels, remain byteequal.69 Bench production modules are unchanged.

Independent prelaunch review preserved and fixed two P2 findings before any numerical launch: shared admission initially kept all delta views and stacked all representatives; it now retains no such views and rebuilds<=32representative displacements per coefficient batch. A batching test imported a module fixture from a different test module; it was moved into the original module with all prior function/fixture ASTs unchanged, ensuring one complete near/reference construction. Prepared v1/v2 collectors were never executed. Original review snapshots and SHA proofs remain available; v3 prelaunch audit has no P1/P2.

Original v3 collector CLOSED6PASS1FAIL/XML335.830s/runner337.7495450126007s/411 stable frozen sources. The configuration negative omitted the established local_rwg_v1 Riesz declaration and failed before the intended periodic rejection. Original tested source, traceback and six successful controls remain unchanged. Only the failed function added that required setting after original handle8009 drained rc1 and observed1563920/1565056 exited. Failed-only v4 CLOSED1PASS/XML3.269s/runner4.513600739650428s/404 stable inputs. Its quick pytest PID was unobserved and remains NULL; execution runner1872351 absent, original handle69794 drained rc0 and child waited rc0. No successful physical/native/planner/public control was rerun.

The actual controlled system has two radius.01 spheres at x=-.5/+.5,300MHz,60 physical edges/120 currents, eps2/3, mu1.3/1.4, electric conductivity.01 and magnetic.02, original q3/singular3, p8/q12, leaf2/separation.25 and unchanged kernel budgets1e-6/1e-5. Exterior has two logical FAR pairs and one shared owner; both magnetic interiors have zero FAR. All120 complete columns compare independently generated original compact FAR tables plus identical near/projection objects: balanced relative0. Native15 noncontiguous forward/H/current/adjoint-current/higher/dot maximum1.5498535897101087e-16<1e-11, with zero saved internal tensors. Owned2,124,984B replaces2,148,480B; only M2L changes46,992B->23,496B. Capacity-minus-one/action and table budgets reject before moments; saved original-owner mutation rejects receipt/H/backward. This controlled sharing comparison reuses near integration and is not a new full original RWG or physics certificate.

The same physical leaf nodes supply729 actual cross-body logical couplings and462 distinct displacements. Every representative generation call is<=32 (fourteen32-pair batches and one14-pair batch); actual unique retained10,855,152B equals exact planned storage and budget-minus-one rejects before coefficients. No second near integration or complete solve is needed for this control. Both order planning branches use a single fixed p8 diagnostic, preserving every ordered contour/RWG result while changing only table bytes; this does not repeat the original global order search or certify implemented floating-point truncation.

The public two-body radius.01 x=-.2/+.2 eps2/3 mu1.2/1.3 scene at90MHz uses unchanged p6-10 admission, original q4/singular3, leaf8/separation.25, primary bounded16-edge factors, GMRES rtol1e-12/atol0/max240/restart120/no fallback. Selectedp6 converges in93 iterations. Compressed dual2.1372545739545198e-13 and original eight-edge sample2.88799511817202e-13 pass unchanged1e-8. The sample is not a global original residual/current-error or physical certificate. Public retained1,013,696B includes11,760B M2L. Physical Result schema11 saves once and restores exact currents/configuration with constructor/solve tripwires and near/table/kernel/shift tamper rejection. Its allocator stats were not reset over the whole test interval and are not resource evidence.

The completed producer SHA7231028437f24cb7a3e2b7e9cd41c8467f5be472eeba6cf1734e5f168250bfaa is reused once by Bench under independently declared physical Scene/full numerical identity. Exact J/M bytes, native schema11, unqualified-candidate role, constructor/solve refusal and the four native/manifest tamper rejections pass. Bench CLOSED1PASS/XML4.574s/runner6.0688778180629015s/473 stable inputs; actual observed1872942/1873810 absent and original handle95412 drained rc0. No producer solve/save is repeated. The original producer aggregate FAIL is not relabelled PASS; its successful public subtest/artifact is explicitly selected.

Eight distinct PASS comprise six NEW Maxwell controls, one necessary affected old configuration and one NEW Bench archive. Independent final PHASE audit precedes separate commits. Original3840 reciprocal-near allocator peak965,273,088B/reserved1,048,576,000B still exceeds same-N dense943,718,400B; this small shared-table stage does not close that failure. A changed original-resource measurement is separately prepared but has not launched or passed here. Native quadrature/global physical convergence, positive automatic magnetic changing-carrier FAR, tensor/periodic/nonlinear capabilities, full solver/setup/action/backward peaks, all-point scaling/crossover, populations, current four-repository full acceptance and at least two independent FINAL Goal audits remain required. Goal ACTIVE;GSP01/GSP09 OPEN; original scope/failures/gates unchanged; no production qualification.

### Complete scaling resource interval workflow completed

The primary measurement workflow now records a continuous current-process CUDA allocator interval from before construction through all warmups, measured forward/H, differentiable forward, first current backward, reset gaps and final finite checks. Each reset first captures any preceding transient peak. Setup includes original geometry/provider construction; current differentiation reuses the measured H instead of computing a second H action. Every native action must be finite and meet the original1e-11 adjoint/current-gradient gates. An explicit complete-interval v2 plan chooses shared compact displacement tables, separable shifts, reciprocal scalar near and positive-tail admission. All27 pre-existing plan values except version remain AST-identical: two size families, four subdivisions, three repetitions, physical geometry/materials/frequencies, kernel/reference thresholds and original budgets are unchanged. The subsequent small original-reference control remains separately scoped outside this interval; no whole solver or teardown resources are claimed.

The primary summary includes differentiable-forward and continuous allocated/reserved peaks in every complete-family fit. It rejects missing warmups, raw native-action errors at or above1e-11, understated interval peaks, mismatched closed failure/source identities and incompatible resumed worker records. It continues to refuse fits for any incomplete or failed family; no tail selection or asymptotic/crossover qualification is introduced. Process-failed records now preserve their source hash.

Independent v1 static review found one P2: an affected existing negative still expected the previous source-hash error text although stricter early identity validation rejects first. The entire original test snapshot and v1 report/collector inputs are preserved. Only that negative regex changes to identity, with an exact whole-file inverse proof. Prepared v1 was never launched; v2 static review closes the finding before numerical execution. All396 production modules remain byteequal to mergedf5a80a7.

The one-shot v2 collector CLOSED23PASS/XML5.980s/runner7.558053915388882s/412 stable inputs:14NEW controls and9 necessary affected old workflow checks. Only one new actual GPU smoke runs fixed_ka/subdivision0/repetition1,60edges/120currents/300MHz/selectedp6 with original2RHS seed1729. It performs no original matrix assembly, solver or archive, and does not launch the24-case series. Actual owned864,544B equals planning, M2L11,760B; adjoint2.267006868810647e-16/current-gradient0 pass1e-11. Whole allocated92,140,544B and reserved138,412,032B exceed same-N dense230,400B: those point resource comparisons are FAIL, faithfully retained rather than converted from the23 test PASS. Backward increases reserved peak above setup104,857,600B, directly demonstrating why setup alone is insufficient.

Original session77366 drainedrc0; actual observed runner2560660 and pytest2562071 exited outside the sandbox. Installed Core, nativeCUDA float64/complex128 and GPU1 are recorded. No full regression, successful producer or covered physics control was replayed. The independent original3840 shared-table resource collector remains LIVE on its frozenf5a80a7 provider and is neither changed nor relabelled by this development-script stage. Its result is pending; previous3840 reciprocal-near peak FAIL and all original physical failures remain preserved.

This is a workflow/PHASE checkpoint, not an all-point resource series or FINAL Goal audit. Full solver resources, native quadrature/global physical acceptance, positive automatic magnetic changing-carrier FAR, tensor/periodic/nonlinear combinations, preregistered all-point scaling/crossover, populations, current four-repository acceptance and at least two independent FINAL audits remain required. Goal ACTIVE;GSP01/GSP09 OPEN; original scope/gates unchanged; no production qualification.

### Original shared compact fixed-ka full GPU resource measurement completed

The original two-sphere3840-edge fixed-ka experiment now measures exact native displacement sharing together with the already integrated reciprocal original scalar D/K near storage. Radius0.01m/centers±0.5m/subdivision3/eps2-vacuum/300MHz/p11/leaf30/separation0.25/q4-singular3/positive_tail_min_v1, G1e-6/H1e-5, all original retained/table/near/projection budgets and native two-RHS seed1729/1e-11 gates remain unchanged. Completed original p11 order admission and prior reciprocal-near resource receipts are SHA-bound and reused, never re-searched or replayed. Only FAR table ownership changes; logical physical interactions and original near receipts remain exact.

Actual owned retained storage585,948,608B equals the primary structural plan, down from795,719,008B by exactly209,770,400B of table savings. M2L144,917,920B, near259,874,344B and projection180,911,744B are separately recorded. All6738 logical FAR couplings remain;2753 unique coefficient owners share their exact same/opposite displacement storage. All other capacity categories and reciprocal-near receipts are unchanged. No geometry/coefficient approximation, threshold relaxation or interaction removal is used.

The complete current-process allocator interval includes original construction, every forward/H warmup, measured forward/H, differentiable forward, first current backward, peak-reset gaps and final current/finite checks. Peak allocated751,039,488B and reserved803,209,216B are BOTH below unchanged same-N dense943,718,400B. This selected point comparison is PASS. Previous reciprocal-only peaks965,273,088B/1,048,576,000B and ordered-D/K peaks1,224,886,272B/1,317,011,456B remain preserved FAIL evidence, not overwritten. Native adjoint3.8408795747260305e-15 and current-gradient7.12969985810129e-17 pass1e-11.

The original collector CLOSED1NEWPASS/XML4297.158s/runner4299.230414410122s/417 stable inputs. Original session66501 drainedrc0; observed runner2055404/pytest2055588 are absent outside the sandbox. Installed Core/nativeCUDA/GPU1 are recorded. Timing is from a shared busy GPU and is not a performance claim. No solver, original all-column reference, repeated producer archive, full regression or all-point scaling series runs in this measurement.

Frozen tested baselinef5a80a7 remains explicit. Current92806ac source applicability is separate:395 primary modules byteexact, and removing only the appended unused complete-original-residual diagnostic restores the entire old operator_error module. The intervening89621e7 development-workflow phase changes no production module. Never relabel this collector as numerically tested at the newer revision.

This is one original fixed-ka setup/actions/current-backward resource point, not a whole solver resource certificate, scaling/crossover series or FINAL Goal audit. Growing-ka3840 and other original collectors remain LIVE on frozen providers. Full solver/workspace/host resources, every preregistered size/repetition, general/tensor/periodic/nonlinear native quadrature/physics, production populations, current four-repository acceptance and at least two independent FINAL audits remain required. Goal ACTIVE;GSP01/GSP09 OPEN;original scope/gates unchanged;production qualification false.

### Original resonant f30 compressed capacity preflight completed

The failed-physics successor preserves the original resonant sphere radius0.1m/frequency414635862.3319615Hz/epsilon12+0.01j, frequency-mesh30 with27000edges/54000currents, q4/singular6/pair64/far8192/near1e-5/max16, physical0.005 and complete original-vector residual1e-10/atol0. Historical f24 physical failures remain SHA-bound and are not replayed. No predicted finer-mesh accuracy is reported as measured.

Native GPU1 geometry/provider/two independent physical support trees feed the existing primary capacity planner. Same trees/partition, leaf30/separation0.25, reciprocal scalar D/K near, exact displacement-shared compact tables and separable shifts are retained. Minimum p6 requires14,829,648,416B against unchanged2GiB action budget: M2L11,473,832,160B and near3,062,449,136B each exceed unchanged1GiB component budgets; projection291,371,328B passes. For this exact partition/storage, increasing p6..12 cannot repair the minimum capacity rejection. No kernel order search, expansion, original quadrature, compressed action, solve, backward, dense global matrix or field acceptance is launched. This is a configured candidate capacity rejection, not a claim that every compressed backend is impossible.

The preflight itself records peak allocated70,482,944B/reserved92,274,688B, limited to geometry/provider/tree/capacity planning. These peaks are not setup/actions/solver or whole-resource qualification. Same-N dense matrix alone would require46,656,000,000B; it is not constructed. One NEW preflight instrumentation control PASS/XML2689.929s/runner2691.5254482720047s with411 unchanged frozen inputs records the rejection; its pytest PASS grants no physical, resource or production qualification. Original session37714 drainedrc0; observed runner3362671/pytest3363646 are absent outside the sandbox. Installed Core/native CUDA/GPU1 remain explicit.

Frozen tested baseline92806ac remains explicit. Current d06eed1 applicability is separate:394 other primary modules exact; two subsequent runtime/lifting RHS changes are unused by this geometry/capacity-only test and documented by the preserved continuation proof. No old PASS/full regression/archive or numerical rerun. GSP01/GSP09 OPEN;physical successor solve awaits a newly preregistered admissible compression strategy under the same original gates. Goal ACTIVE;full scope, current four-repository final acceptance and at least two independent FINAL audits remain required;production qualification false.

### Original shared compact growing-ka full GPU resource measurement completed

The original two-sphere3840-edge growing-ka resource experiment preserves subdivision3/radius0.01m/centers±0.5m/eps2-vacuum/2.4GHz/p12/leaf30/separation0.25/q4-singular3/direct_first_wave_min_v1, G1e-6/H1e-5, all original2GiB action/1GiB table-near-projection budgets and native two-RHS seed1729/1e-11 gates. Closed original p12 admission and original full-matrix compact resource receipts are SHA-bound and reused, with no repeated order search or historical producer archive. Reciprocal original scalar D/K near and exact native displacement-shared compact tables preserve the same physical logical interactions and all other retained categories.

Actual owned668,235,432B equals the primary structural plan, compared with original full-matrix compact1,656,499,936B. Near259,874,344B saves725,095,104B; table181,808,120B saves263,169,400B. Unshared reciprocal compact intermediate931,404,832B isolates the two storage savings. Projection226,308,368B and every other category remain exact. All6738 logical FAR couplings remain with2753 unique native coefficient owners; no interaction removal, geometry/coefficient approximation or gate relaxation.

Complete process allocator peaks853,474,304B allocated/868,220,928B reserved include construction, every forward/H warmup, measured forward/H, differentiable forward, first current backward, peak-reset gaps and final current/finite checks. Both are below unchanged same-N dense943,718,400B: this selected growing-ka point comparison PASS. Native H2.030842010454122e-15 and current-gradient7.113627934723499e-17 pass original1e-11. Original full-matrix compact retained/allocated/reserved point failures remain preserved; that historical collector omitted warmups/differentiable forward from its per-phase measurements and is not relabelled as complete-interval evidence.

Original collector CLOSED1NEWPASS/XML5977.152s/runner5978.764860432595s/417 stable inputs. Original session22853 drainedrc0; observed runner3119305/pytest3120332 are absent outside the sandbox. Installed Core/native CUDA/GPU1 are recorded. Shared busy GPU timing is not a performance claim. No original all-column reference, solve, producer archive, full regression or full24-case series is run.

Frozen tested baseline89621e7 remains explicit. Current main source applicability is separately documented by continuation proofs:393 other primary modules exact; later operator_error/runtime/lifting changes are unused here and original module inverses are preserved. Never relabel this numerical collector as tested at current HEAD. This selected-point full resource PASS does not qualify all-size scaling/crossover, whole solver or physics. Goal ACTIVE;GSP01/GSP09 OPEN;full scope/gates unchanged;current four-repository final acceptance and at least two independent FINAL audits remain required;production qualification false.

### SHA-bound scaling order admission reuse workflow completed

The current development measurement workflow binds fixed-ka to positive_tail_min_v1 and growing-ka to direct_first_wave_min_v1, the respective closed original3840-edge admissible configurations p11 and p12. Existing failed growing positive-tail admission is preserved. Original two-sphere geometry/materials/q4-singular3/leaf30/separation0.25/G1e-6/H1e-5, all resource/native gates and the complete setup/all-warmups/forward/H/differentiable-forward/cold-current-backward/reset-gap/final-finite interval remain unchanged. No production numerical module changes.

A full series requires a SHA-bound closed-order manifest matching the current complete plan,396-module production source identity and each physical case. Curated original receipt hashes and original producer inputs/tests bind the largest configured order selections. The completed fixed60-edge minimum-p6 native workflow receipt can supply that minimum configuration without re-searching any earlier order. This reuses admissible configured orders, not current minimum-order proof or historical measurements as current repetitions. Every current constructor still checks its native coupling, retained and component budgets. Manual configured constructors report their actual null order_plan; cached selections carry separate provenance and no artificial constructor plan is injected.

Each same-size subsequent repetition consumes the completed current first-repetition selection, source SHA and original matching coordinates/current source/plan/order/native gates. A failed first repetition produces explicit not_launched failures for identical later attempts, without replaying failed setup/admission. Foreign failed first records are rejected. Receipt tampering, changed physical parameters/rules, invalid integer order, changed production/plan identity, unavailable or over-budget kernel bounds, failed/nonfinite/negative native errors and source changes are rejected. Summary validation binds physical parameters and first-repetition provenance; fits use every individual repetition at every declared size and never incomplete tails.

Targeted collector CLOSED39PASS:25NEW order-reuse metadata controls plus14 necessary affected runner/summary controls, XML17.792s/403 stable frozen inputs; no covered native reference/solve/resource PASS is replayed. Original session78252 drainedrc0; runner PID from execution receipt is absent outside the sandbox and short-lived pytest PID was not observed. The initial launcher exited at its missing CUDA_VISIBLE_DEVICES guard before execution/pytest and then received the declared full environment. Actual targeted collector uses installed Core/native CUDA/GPU1. These metadata controls do not establish numerical/physical/scaling qualification.

The24-case full series remains to run at a frozen merged revision:two families×four sizes×three process repetitions. Closed p11/p12 and fixed minimum-p6 selections prevent old planner replay; remaining newly required size/family order searches run once each and current repetitions reuse them. Actual resource repetitions remain separate current measurements. Shared-GPU timing is descriptive only. Original source/gates/failures remain; GSP01/GSP09 OPEN;Goal ACTIVE;four-repository final acceptance and at least two independent FINAL audits remain required;production qualification false.

### Bounded native compact FAR execution and exact archive stage completed

Explicit FMMConfig(far_action_batch=2..32) batches existing original compact electromagnetic translations on CUDA. Each regional action bounds the entire upward/FAR/downward/near sequence to four RHS and each translation to at most sixteen target rows. Order0..12, fixed native complex128, compact or exact-displacement shared compact tables and free-space scalar media are required. Existing default batch1 paths and fourteen archived full payload configurations remain exact; DirectionalFMMConfig is unchanged. New strict cartesian_fmm_batched_compact_config_v1 binds every configuration field. Periodic, complex64, unsupported order/storage and invalid structural batches reject before finite scans/stacks.

Original scalar Green/curl/charge formulas use batched matmul without new kernels, coefficient polynomials or near quadrature. Native reversal parity and every logical FAR interaction remain; per-pair accumulation keeps original order. All stacks/tables are transient; retained ownership/capacity is unchanged. Existing fixed-operator autograd forwards save no generated tensor and route first/higher current derivatives through the original native Hermitian action. A conservative visible-tensor preflight at maximum order12/pairs32/RHS4/rows16 is121,388,800B below128MiB; this is not a measured allocator/library or whole-system resource certificate.

Fourteen distinct Maxwell checks PASS: eleven NEW and three necessary affected old scalar/FFT controls. Original v2 remains10PASS/3fixtureERROR, source418 stable. The blocked positive controls continued alone in v3, obtaining3PASS with complete120-column original weak matrix error3.9795055549581184e-16, native Hermitian dot2.225837265142875e-16,15 noncontiguous RHS, first/higher current gradients, zero saved forward tensors, exact1,013,696B retained capacity, native reversal and mutation refusal. A single full original weak matrix supplies all new physical controls and the public all120-row current residual1.3686043304280155e-13 at the unchanged1e-10/atol0 gate. Public native solve/strict exact restore PASS; one result is saved once. This dense consistency/residual evidence does not certify continuum physics or quadrature accuracy.

New p12/32pair/4RHS boundary initially failed before coefficients: v3 complex-eps public material compilation TypeError, then v4 incorrect sigma keyword TypeError. Both original failures and whole test sources remain preserved. Only the failed new fixture changed to declared sigma_e conductivity with real eps2/mu1.3; all396 primary sources and all successful controls remain exact. The failed-only v5 now1PASS, original forward1.5139264402408674e-17/H2.1592377384656235e-17 and native dot4.37790243810346e-16 below unchanged1e-11. Two original coefficient owners and their reversals exercise all32pairs without duplicate polynomial construction. The complex-eps compiler limitation remains unresolved and no complex-eps qualification is inferred.

One NEW Bench check PASS with69 production modules unchanged. Its independent Scene/full config is declared before loading terminal v3 successful public artifact b24923a12fb3ee40b7628f1b9cba120eece225746f7d9580c84ac7ee19f1b4ea. Constructor/solve tripwires, exact schema11 J/M bytes, strict version restoration and batch/table/near/kernel/shift manifest/native tamper rejection pass. Unqualified-candidate role and false qualification remain. Original v3 aggregate3PASS1FAIL is preserved; its successful public subtest is selected explicitly. No repeated Maxwell solve, original matrix, order search or producer archive.

All native checks use GPU1 outside the sandbox with installed Core. Original sessions68023/v2,49336/v3,86096/v4,77182/v5 and17428/Bench are drained; observed v3 runner589002/pytest589162 are absent, other short-lived pytest PIDs were not observed and remain null. Complete receipts/XML/hash closures and source inverses accompany the assessment. No per-edit full regression or covered PASS replay. This GSP09/GSP10/GSP11 implementation stage leaves resource/complexity/physics and all final gates open. The original24-case all-point series remains LIVE and untouched at frozen1204329; changed scalar numerical source in this stage prevents relabeling that earlier-source series as current optimized evidence. Shared-GPU timing is descriptive. Original resonant f30 retained-capacity rejection, all GSP01 failures and all other open capabilities remain. Goal ACTIVE; complete scope/thresholds and current four-repository final acceptance with at least two independent FINAL audits remain required; production qualification false.
