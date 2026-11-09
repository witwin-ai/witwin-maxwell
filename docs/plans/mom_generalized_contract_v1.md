# Generalized integral-equation contract v1

Status: frozen design, not an implementation or qualification receipt.
Parent: [execution plan](mom_generalized_production_development_plan.md).
No new public configuration is enabled by this document.

The later [lifting representation decision](mom_lifting_representation_contract_v2.md)
amends this design's migration executability, inner-schema switch timing and
GSP-01 prerequisite for the isolated numerical repair candidate. Its specific
clauses govern that candidate; main runtime v1 remains unchanged until the
separately qualified enabling commit. The other scope and acceptance budgets
below remain in force.

## Identity and conventions

Use `exp(-i omega t)`, outgoing `exp(+i k r)`, outward normals from
each body to the common exterior, electric J in A/m and magnetic M in V/m.
Unknowns are all bodies' J followed by all bodies' M, in enabled Scene
structure order; rows are all H tests followed by all E tests. The PMCHWT
block and RHS signs in GSP-03 are normative. Absorption is positive inward
time-average Poynting flux; outward scattered power is positive. Extinction
equals scattering plus absorption for an isolated passive linear system.
Nonlinear power sums include all retained harmonics and dissipative terms;
DC contributes static fields, not propagating radiated power.

For passive tensors the Hermitian matrices `(epsilon-epsilon.H)/(2i)` and
`(mu-mu.H)/(2i)` are positive semidefinite at positive frequency; reciprocity
requires transpose symmetry, not Hermitian symmetry. Scalar outgoing branch
selection requires Im(k)>=0 and outward positive power, including double
negative media. Undetermined or singular branches fail before assembly.

Cache, result, snapshot and evidence identity includes formulation, kernel,
backend, algorithm version, tolerance, dtype, frequency/harmonics, sampled
materials, Scene order, geometry and source/Bloch identity. Identical physical
scenes with different numerical backends remain in one Bench split.

## Capability matrix

Every row starts unqualified at the current source epoch. Historical single
body receipts retain their original scope. No row grants another row.

| Formulation | Kernels to qualify | Backends to qualify | Required combinations |
| --- | --- | --- | --- |
| pmchwt_scalar_sie | free_space_helmholtz, quasiperiodic_ewald_1d/2d/3d | dense, hmatrix, h2, fmm; pme separately | multi-body + magnetic + dense/h2/fmm; periodic + multi-body + magnetic |
| anisotropic_dgf_sie | free-space dyadic and quasiperiodic dyadic versions | dense, hmatrix, h2 | multi-body + reciprocal tensors; periodic + reciprocal tensors; nonreciprocal guards |
| coupled_sie_vie_linear | explicitly registered surface/volume kernels | dense then fast surface/volume actions | zero volume source scalar reduction; no duplicated linear contrast |
| coupled_sie_vie_hb | explicitly registered per-harmonic kernels plus DC | dense then fast surface/volume actions | multi-body + nonlinear; chi2/chi3/Kerr/TPA; no reciprocity assumptions |

The dyadic and DC kernels require separate versioned identifiers before their
implementation epochs; a scalar kernel identifier cannot label a dyadic kernel.
`auto` is a capacity policy selecting an independently qualified backend,
not a numerical backend. PME cannot qualify FMM. Undeclared combinations
remain rejected. Open, touching, intersecting or nested bodies, internal
sources and geometry gradients remain outside this goal.

## Snapshot v2 and migration

This version refers only to the inner `SurfaceData` payload (currently v1).
The outer `Result` snapshot already has schema v2 and is not renumbered here.
The future inner v2 schema requires: `schema_version=2`, basis and units, explicit
global coefficient order, coefficients and dtype, a body table containing
region ID, Scene structure index, name, vertices/faces, edge/face offsets and
J/M slices, sampled exterior and per-body materials, full Scene signature,
formulation/kernel/backend versions, solver/integration/query settings,
lattice/Bloch identity where relevant, harmonic/volume payload where relevant,
and deterministic tree/compression construction and error/performance receipts.
Validate lengths, finite values, contiguous disjoint slices, region uniqueness,
geometry/material identity and order before allocating production operators.

Keep precisely one explicit offline `v1 -> v2` migration: require the supplied
matching single-body Scene and frequency, validate v1 with its frozen contract,
construct body region 1 with original structure index and sampled media, record
the source payload hash and migration version. Never invent missing evidence.
The v2 runtime accepts v2 only; migration does not run silently on load.
Other versions, reordered bodies or changed materials fail closed. Existing
runtime v1 behavior remains unchanged until GSP-03/GSP-10 implement this design.

## Preregistered acceptance budgets

Use the stricter historical threshold wherever it exists. Relative errors use
the reference norm with a declared physical absolute floor; inputs, normalization
and floors must be in the frozen case manifest before numerical execution.

| Quantity | Maximum error |
| --- | --- |
| complex128 dense equivalence and free-space kernel | 1e-12 relative, 1e-14 normalized absolute floor |
| complex128 forward/adjoint fast operator in impedance-balanced Riesz norm | 1e-8 relative, 1e-12 normalized absolute floor |
| normalized adjoint dot defect | 1e-11 |
| key field/cross-section physical error | 0.005 relative; reference floor 1e-8 times incident normalization |
| fast contribution to physical error | 0.0005, and <= one tenth of the case's stricter budget |
| normalized lossless power imbalance | 0.005 relative to positive incident reference power, absolute floor 1e-8 |
| normalized lossy power closure | 0.005 with same floor; absorption >= -1e-8 |
| periodic diffraction/flux power closure | 0.005 with unit-cell incident-power normalization and floor 1e-8 |
| nonlinear total-power/Manley–Rowe closure | 0.005 with pump-power normalization and floor 1e-8 |
| external reference | <= neural target / 4, desired / 10 |

For an isolated object, the positive reference power is `I_inc * A_ref`,
where `I_inc = Re(E_inc cross H_inc*) dot propagation_direction / 2` and
`A_ref = pi * a_bound**2`; `a_bound` is the radius of the declared enclosing
sphere of the full body system. A cell uses the positive incoming propagating
modal power through its incident port face. Harmonic balance uses positive
injected pump power. Freeze these values per case. Never normalize by net
incident flux through a closed surface, which vanishes for a plane wave.

Zero incident power uses absolute error and cannot qualify relative scattering
accuracy. Complex64 is measured separately with per-case error budgets frozen
before execution; it cannot inherit complex128 algebra qualification.
Fast and iterative errors share the physical budget; both approximate-operator
residual and original dense spot residual must pass their preregistered gates.

Complexity manifests must freeze at least five N values, both fixed electrical
size refinement and increasing electrical size at fixed points/wavelength,
three repeats after warm-up, the memory/time ceiling, all-rung slope estimator,
rank/order, crossover and cumulative retained/setup/peak device bytes.
These manifests must be audited before any timed qualification run. Absent a
manifest the result is exploratory only. Geometry gap, passivity tolerance,
anomaly distance and spectral-pole thresholds also require case manifests;
there is no implicit support claim for an unregistered domain.

## Negative configuration and collection gates

GSP-00 uses the existing validation tests without changing runtime contracts.
Later enabling commits must reject unknown identifiers, invalid tolerances,
unqualified combinations, incompatible Bloch RHS, CPU production operands,
ambiguous regions, invalid snapshots and insufficient rank/capacity before
assembly. The full parent plan's per-stage tests and independent references
remain mandatory; this contract does not replace them.

GSP-01 is a prerequisite to changing operator mathematics. Freeze a new
source inventory and collection, leave exact-e95 receipts and allowlists
immutable, then execute both CUDA devices and independent source audit.
Missing GPUs are a pending acceptance gate, never a passing skip.
