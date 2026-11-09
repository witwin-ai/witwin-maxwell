# Surface integral backend delivery

Requirements: workspace `witwin_maxwell_development_plan.md` (2026-09-11),
with physical signs from `revised_research_plan.md`, sections 3–7.

Current measured outcomes and remaining work are consolidated in
[the acceptance checkpoint](../assessments/sie_local_acceptance_closure.md).
The [current completion and handoff ledger](mom_handoff_status.md) records
checked local responsibilities, every pending research gate and the recommended
next project. Detailed stage measurements below retain their historical scope;
the final acceptance sections and current ledger supersede earlier progress
statements without rewriting their original observations.

Each stage is committed separately. Implementation and acceptance are distinct:
unmeasured physical accuracy, missing external handoffs, or skipped GPU tests
must never be reported as accepted.

| Stage | Implementation | Acceptance evidence |
|---|---|---|
| MW-01 | SI passive scalar materials, loss once, outgoing branch, shared Scene scope, complex plane waves | 15 CUDA physical-contract tests; H0 cross-project review pending |
| MW-02 | Closed surface validation, sphere/ellipsoid/harmonic meshes, RWG, L2 Gram, projection | 12 CUDA tests; three-level rotated remesh convergence; cross-project H1 review pending |
| MW-03 | Double-RWG weak PMCHWT, joint touching transforms, adaptive near panels | CUDA quadrature tests and four-level independent sphere E/H/far/cross-section convergence pass for the documented samples |
| MW-04 | Batched A/A*, SPD metric inverse, native complex residual gradients, condition diagnostics | Actual PMCHWT inner products and gradients pass on CUDA; metric is an explicitly named L2 proxy |
| MW-05 | Cached LU, arbitrary-initial-guess restarted scalar/block GMRES, reusable scalar initial augmentation | CUDA actual lossy PMCHWT true residual, reuse and failure tests pass; 14 new block tests plus independent rank/weak-source/cost probes pass |
| MW-06 | Joint E/H, analytic singular-gradient subtraction, one-sided traces, adaptive queries, far field/RCS/flux | CUDA decoder and trace tests pass; finest independent sphere volume error is 0.236539%, with final radial refinement change 0.000706591% |
| MW-07 | Unified Simulation/Result candidate, exact native persistence, shared physical queries; independent sphere, cross-backend and resource drivers | Four-level sphere samples meet 0.5%; three-level FDTD and independent ellipsoid controls pass their documented budgets; 3877-node regression coverage closes with no unresolved failures; three-level 32x8 resources and actual streamed forward/adjoint feasibility are measured; external H2/G0 remain pending |
| MW-08 | Regular TE/TM modes, weak modal RHS, full response reuse, control-energy normalization and cutoff calibration | Cutoff 9 contains 198 columns; finest far-response operator error is 0.2154%, added-band ratio 0.03441% meets the 0.05% budget; per-column limitations and external H5 remain explicit |
| MW-09 | Optional, gated after H2 | Stable mixed spaces required; double RWG is not a substitute |
| MW-10 | Exact source/environment/evidence archive implemented; physical extension gated after G2 | Archive integrity and tamper rejection checked; choose extension after external G2 |

## Environment

Use `python`. CUDA is hidden inside the
restricted execution sandbox; approved execution outside it sees two RTX A6000
devices with driver 595.71.05. PyTorch is 2.13.0+cu130. Other processes occupy
both GPUs, so shared-device timing is not an exclusive throughput measurement.

## MW-01 contract

Internal APIs are `compile_medium`, `validate_scene`, `source_phasor`, and
`incident_fields` in `witwin.maxwell.mom.physics`. The public architecture
remains Scene / Simulation / Result; no parallel public scene type is introduced.
Current Scene background is vacuum. Conductivity is added exactly once as
`+i sigma/(omega epsilon_0)`. The branch is fixed before computing impedance.
Shared CW `A*cos(omega*t+phase)` maps to `A*exp(-i phase)`.

Validation command: `python -m pytest tests/mom/test_physics.py -q` on CUDA:
15 passed. This validates material/source building blocks, not PMCHWT or H0
agreement with other projects.

## MW-02 discretization

`SurfaceMesh` requires a connected closed outward manifold without degenerate
or intersecting triangles. Numeric geometry, quadrature, Gram matrices and
projection execute on CUDA; connectivity preprocessing is host work. RWG
metadata records orientation, J/M coefficient units and fixture state.
The implemented metric and nearest-point tangential projection error are
explicit **surface L2 proxies**, not fractional trace-space norm certification.

`python -m pytest tests/mom/test_mesh.py -q`: 12 passed on CUDA. Coverage includes
charge conservation, SPD mass, vertex/face reorder covariance, complex batch
projection gradients, independent rotated remesh convergence and SI scaling.

## MW-03 weak operator and integration

Rows test `(H,-E)` against RWG, equivalent to testing the `(J,M)` trace against
rotated RWG. The right hand side is the integrated `(-H_inc,E_inc)` trace.
Unknowns retain J in A/m and M in V/m. Self, edge and vertex triangle pairs use
joint Duffy transforms with six, six and two sectors respectively. Disjoint
near panels refine regular product quadrature until an order-difference tolerance
is met, or raise a numerical failure. Far panels use regular triangle rules.
Order differences are numerical estimates, not certified error bounds.

CUDA validation includes exact area, first/cross moments, an independently
integrated static self-panel Green function, complex kernel convergence,
reciprocity, all five actual RWG block categories and assembly under arbitrary
edge sign changes (`A'=SAS`). `test_quadrature.py` plus
`test_singular_convergence.py`: 14 passed. Independent sphere solution accuracy,
not these unit tests, decides whether this can supply reference labels.

Mathematical convention references: [boundary weak forms](https://bempp.com/handbook/api/boundary_operators.html)
and [Maxwell trace discretization](https://arxiv.org/abs/1703.10900), read alongside
the workspace research plan's explicit trace definition.

## MW-04 adjoints and residual metric

Native complex matrix operations implement the Hermitian adjoint. The residual
loss is one half of `real(r* R^-1 r)`; its PyTorch current-vector gradient is
`A* R^-1 r`. GPU Cholesky applies the chosen SPD metric, including batched
vectors. Mutable operator/metric replacement is revalidated before numerical
work; cache invalidation cannot trigger an implicit CPU factorization.
Small SVD diagnostics are explicitly noncertifying. Actual no-contrast
PMCHWT inner-product/gradient tests pass; lossy real/imaginary finite differences
are also part of the combined algebra/solve regression.

## MW-05 reference solution and correction

`solve_reference` and `correct` accept the same dense handle and stopping rule.
LU factors and row-norm preconditioning are reused only while the operator is
unchanged. GMRES uses the unpreconditioned residual, a fixed absolute floor,
explicit stagnation/budget failures and logs every residual check and matvec.
Multiple RHS share setup. Optional bounded recycling minimizes the initial
residual over retained corrections before ordinary GMRES; it is **initial
augmentation**, not a full block-GMRES implementation. Cache projection,
updates, extra matvecs, storage and time are counted separately.

Independent review fixed validation of mutated matrices/metrics and unnecessary
preconditioner setup. Regression covers arbitrary guesses, complex multiple
RHS, zero/tiny RHS, near-singular failure, cache mutation, scale invariance,
the small-residual/large-error counterexample and actual lossy PMCHWT correction.
The combined algebra and public subset reported 20 CUDA passes after recycling
was added. Numerical residual success is not physical reference certification.

## MW-06 fields and traces

The decoder evaluates both E and H from the same physical RWG J/M coefficients.
Exterior total field adds incident fields; interior field negates the potential.
Region checks use the actual oriented polyhedron. Face-interior traces use an
analytic principal-value static gradient and explicit electric/magnetic jumps,
including the normal divergence terms. Edges/vertices reject pointwise traces.
Near-field gradient integration subtracts the exact edge-log/solid-angle static
part, leaving a smooth frequency-dependent remainder. Optional adaptive queries
record order differences and fail explicitly when the budget is insufficient.

CUDA tests cover three-mesh Huygens cancellation/recovery, approach to both
traces, real/imaginary coefficient derivatives, near-panel convergence, exact
static-gradient comparison, radiation-zone field limits, RCS and Poynting flux.
The decoder is differentiable in current coefficients; this is not a geometry
derivative implementation. Scalar query state reads and host metadata are
separate from CUDA kernel evaluation.

## Cross-backend source correction

The MoM/FDTD empty-background comparison exposed mismatched damping in the
existing one-dimensional incident line: equal absolute electric/magnetic
conductivities imply unequal damping rates. The auxiliary absorber now uses
equal rates and scales its quadratic profile with physical thickness. Two
CUDA traveling-wave decomposition regressions bound reflected amplitude by
0.005. Existing source/public tests reported 31 passes and one pre-existing
expected failure. At FDTD spacings 0.06/0.05/0.04 m, empty-scene E/H errors
now decrease to approximately 0.303%/0.306%, using one common phasor gain and
the existing H half-step correction exactly once. This fixes calibration;
the dielectric cross-backend acceptance is evaluated separately.

## Independent audit corrections

Two independent reviews are maintained in `docs/assessments/`. Numerical review
found and fixed nonincident geometric contacts missed by combinatorial topology
and an implicit CUDA-device quadrature-cache key that could reuse GPU 0 arrays
on GPU 1. Integration review found prepared-Scene mutation and incomplete FDTD
reference/provenance/TFSF guards; these received direct regression probes.

The correction pilot exposed a real row-only preconditioning failure. Optional
two-sided row/column balancing now updates physical coefficients through the
right scaling and still checks the original residual. A measured 120-edge
lossy case that exceeded the row-only budget converged in 216 iterations with
balancing. Optional direct fallback preserves both attempted solver logs and
all work; it is never substituted without its configured policy.

## MW-07 public integration candidate

`Simulation.mom` is an actual Simulation backend and returns the existing Result.
`surface_currents` retains native RWG J/M and mesh semantics; `query_fields`
decodes physical fields and `far_field` returns complex radiation amplitudes.
Sphere, Ellipsoid and explicit solid Mesh compile from the shared Scene.
Supported PointMonitors are grouped by region and duplicate query positions
are evaluated once. Persisted currents/meshes roundtrip exactly; loading validates
the supplied Scene and restores CUDA queries without storing a dense operator.
FDTD queries use actual Yee-coordinate interpolation and require matching
sampling/normalization provenance when subtracting an empty background.

Public/cross-query audit closure: 15 tests pass on the final grouped-monitor
version; the combined numerical/algebra/public audit subset reported 46 passes.
Independent volume dissipation and far-field power observations are implemented
separately. Their current test is a coarse sign/scale check, not a 0.5% power
certificate. See `docs/reference/mom_protocol.md` for exact supported contracts.

## MW-08 finite regular source space

Regular spherical waves use an explicit vector-spherical-harmonic angular
spectrum. The TE field is `j_l(kr) X_lm`; the TM convention is `curl(TE)/(i k)`.
Ordering is increasing l, m from -l to l, then TE/TM, with orthonormal
Condon–Shortley harmonics. Positive E/H energy on an enclosing control sphere
defines normalization; the net flux of a regular wave is not used as a norm.
Singular sources inside that sphere and mismatched backgrounds reject.

`build_modal_rhs`, `modal_response` and `calibrate_modal_response` keep all
columns on CUDA and reuse the same factorization. Calibration increases cutoff
by two and angular order by four, checks all added response columns and records
the weighted new-band operator norm. A real dielectric test with cutoffs 1/3/5
(6/30/70 columns) has final new-band ratio 0.00308838 under a specified 0.005
threshold. Angular response differences are below 1e-12 in that test.
Independent spherical-Bessel tests verify radial shape and complex phase.
Independent modal scattering improves from 58.3% to 20.7% on two coarse meshes;
this confirms convergence/signs but is explicitly not spatial H5 acceptance.

## MW-07 reproducible acceptance drivers

`python -m benchmark mom` dispatches `convergence`, `fields`, `pilot`, `cross`,
`algebra` and `archive`. The independent sphere oracle evaluates complex
Cartesian total/scattered/internal E/H, far fields and cross sections with
scientific special functions. It satisfies no-contrast, Maxwell curls,
tangential and normal interface conditions, volume absorption and far-phase
checks. This host oracle does not enter the CUDA solver or its gradient path.
The field driver archives physical shell samples and 288-point solid-angle
weighted far-field samples, along with query and solve logs.

The 32-by-eight pilot passes at 120 and 480 edges, measuring cached multi-RHS
algebra/resources rather than reference accuracy. Current medium-size algebra
acceptance covers 480/1920 edges and complex64/128. At 480 edges the zero,
random and perturbed guesses converge in 264/518/442 iterations. At 1920 all
three reach the 1000-iteration budget and the explicitly configured cached LU
fallback succeeds; those rows are not GMRES convergence evidence.

The precision comparison at 480 edges reports forward/adjoint differences
about 1.12e-6/1.11e-6, scaled-current difference 3.04e-5, near E/H differences
6.32e-6/1.15e-4, far-field difference 5.04e-6 and absorption difference 2.02e-6.
A representative profile contains 2674 CUDA events and 52 scalar device-to-host
state transfers, without a host numerical solve. Own allocator peaks are about
181 MB/1.53 GB for the two sizes; shared-device timings are not isolated speed
comparisons. The full setup was not profiled event by event.

The cross driver uses shared physical queries for no contrast, lossless sphere,
lossy sphere and ellipsoid, with three resolutions and separate PML/time-step/
window sweeps. Its closed-surface workflow compares complex far fields, RCS,
fluxes and far-integrated power, while `--volume` independently integrates
conductivity-weighted Yee fields and the actual MoM polyhedron's tetrahedral
volume. Coarse results do not establish the final physical budgets.

## Reproduction and baseline regression work

See `docs/reference/mom_reproduction.md` for toolkit, sparse-direct dependency,
commands and the archive format. Archive checks reject changed evidence bytes
and visibly retain stale/missing report source provenance. Early performance
reports captured earlier implementations and remain preliminary.

Full regression exposed baseline issues reproduced at `2b99536`: a stale mesh
bound assertion, an understated capability census, obsolete sparse-library
tolerance calls, TFQMR overflow, and ESD circuit failures. Geometry/census
corrections, explicit sparse tolerances and stable TFQMR rotations are committed.
The FDFD focused groups pass 18 direct/cache/preconditioner/adjoint tests and
16 Krylov tests. ESD failures came from treating PWL interpolation spacing as
a physical frequency: sampling now uses the amplitude-normalized slope, and
all 44 ESD tests plus four independent circuit ODE checks pass. Native CUDA
extension and layout-adapter prerequisites were enabled; their combined group
passes 210 tests. Complete regression continues with per-test records.

The final cross-backend drivers hold weak integration at order 4 in regular
and touching rules. An independent 1920-edge same-point sphere comparison
against order 6 changes E/H by at most 2.48e-6 relative to the analytic fields.
That calibration is explicitly a sphere/material sample, not certification of
all future shapes. Fine queries use bounded CUDA far-panel batching with
unchanged field/gradient rules. New reports freeze source snapshots, reject
incompatible resumed integration settings, and distinguish analytically zero
scattering and physical angular norms from historical sample metrics.

## Final physical acceptance evidence (2026-09-12)

The completed independent sphere series uses 120/480/1920/7680 RWG edges
with common physical probes and fixed order-4 integration. At 7680 edges,
lossless and lossy sampled E/H errors are about 0.22%, weighted complex far
errors about 0.216%, and scattering cross-section errors about 0.43%. The
1920-edge solutions do not meet the same 0.5% physical target. See
[final sphere and resource evidence](../assessments/sie_final_physical_evidence.md).

The shared-scene FDTD comparison includes three grid resolutions and separate
PML, time-step, time-window, angular and control-radius checks. Fine samples
meet the budgets frozen before refinement; those budgets include each
backend's discretization error and are not all equal to 0.5%. Independent
volume dissipation is compared separately. Ellipsoid MoM self-convergence
also includes three meshes, with fixed physical probes and preserved source
snapshots. See [cross-backend acceptance](../assessments/sie_cross_fine_acceptance.md).

The independent modal study completes all 198 response columns at cutoff 9
and checks increasing cutoffs, angular quadrature and spatial resolution.
Its 0.2154% weighted far-response operator error and 0.03441% added-band
ratio satisfy the documented aggregate budgets. Only 25 of 198 individual
column relative errors meet 0.5%; very small reference columns require their
reported absolute errors and amplitudes. This is not an all-column relative
certificate. See [modal acceptance](../assessments/sie_modal_response_acceptance.md).

The independent auxiliary electric-dipole reference uses CUDA strong boundary
collocation and a separate field kernel, with analytic sphere checks and
held-out tangential and normal transmission conditions. Immutable ellipsoid
MoM reports are compared only after full physical metadata or a reviewed,
SHA-bound source sidecar establishes their input contract. Reference field
and power convergence have separate 0.05% budgets. The final source-matched
independent ellipsoid run and fine sphere volume check both pass and are archived;
the 3877-node regression inventory is covered, and block-Krylov/streamed-action work is implemented with independent audits. Cross-project H0/H1/H2/H5 handoffs, the neural
error-budget ratio and G2 authorization are not inferred from these measurements.

The subsequent full-plan work adds actual shared block Arnoldi through
`StoppingRule(method="block_gmres")`, with rank deflation, independent RHS
locking and unscaled residual certification. Its final 480-edge coefficient
qualification is reported separately; a residual pass alone is insufficient.
The streamed experiment reuses the same scalar GMRES algorithm and performs
actual matrix-free solves, matched forward/adjoint and near-field/loss checks.
It recomputes quadratic panel interactions and is not a production accelerated
backend. See [the independent algorithm audit](../assessments/sie_block_streamed_independent_audit.md)
and [the three-level resource report](../assessments/sie_resource_scaling_acceptance.md).
