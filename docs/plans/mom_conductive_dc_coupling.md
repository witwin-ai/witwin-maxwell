# Coupled conductive DC path

Status: homogeneous point and cell source primitives implemented; boundary and
HB coupling OPEN. This does not replace the finite-permeability magnetostatic
closure or remove its admission guard.

## Trigger and equations

Bench candidate0399ea0 preserves sampled magnetic loss as sigma_m. Its nonlinear
H=2 attempt /tmp/gsp11-nonlinear-candidate-control-v1 fails at the magnetic
conductive_limit guard. Both static electric and magnetic conduction require

    curl E = -sigma_m H - K
    curl H =  sigma_e E + J

with real DC fields and source current densities. Eliminating H gives
(curl curl + kappa²)E=-sigma_m J-curl K, kappa²=sigma_e sigma_m.
This is not a finite-mu magnetostatic scalar-potential problem.

For strictly positive homogeneous scalar conductivities, set

    g = exp(-kappa*r)/(4*pi*r)
    Dg = I*g - Hess(g)/kappa²
    E = -sigma_m Dg*J - curl(g*K)
    H =  curl(g*J) - sigma_e Dg*K

because (-Delta+kappa²)g=delta and
(curl curl+kappa²)(I-Hess/kappa²)g=I*delta. No conductivity epsilon or
frequency regularization is introduced. Zero channels require their separate
static formulations. The distributional Hessian is essential for interior cells.

## Implementation and evidence

`conductive_static.conductive_dc_green` returns the four source-separated CUDA
blocks. It retains coordinate/material autograd and rejects coincidence, nonfinite
inputs, nonpositive coefficients and unresolved screening arithmetic.
`conductive_dc_cell_fields` integrates real constant tetrahedral electric/magnetic
currents using existing `cell_green_potentials(k=1j*kappa)`, including its boundary
formula for the distributional Hessian. It preserves source-current gradients.
Conductivities there must be fixed Python scalars because native volume quadrature
freezes k; it does not claim material gradients for that integrated operation.
Fixed quadrature order is not an adaptive error certificate.

/tmp/gsp11-conductive-dc-kernel-v1.log8passed3.60s: automatic coordinate derivatives
check both Maxwell curl equations and divergence away from sources, full positive
frequency Maxwell Green approaches the DC kernel at decreasing omega, reciprocity
and material finite-difference gradcheck, singular/unresolved channels rejected.
/tmp/gsp11-conductive-dc-volume-v1.log1passed3.55s: volume Hessian trace equals
kappa²*integral(g)-cell_indicator, finite self fields/source gradients and shrinking
volume approaches the integrated point-current limit. No full regression repeated.

## Remaining coupling obligations

- Couple screened conducting interiors to the vacuum exterior, where sigma_e and
  sigma_m vanish and the above inverse cannot be evaluated by substitution.
- Define shared interface traces and constraints preserving both tangential fields
  and distributional current continuity, including every disjoint body.
- Include nonlinear DC electric current from the current HB iterate, and the
  magnetic conductive current self-consistently; retain charge/flux constraints.
- Add preparation/state/decoder/snapshot identities, native original-equation
  residuals and exact restoration before admitting the branch in HarmonicRuntime.
- Validate the zero-magnetic-conduction limit against existing current/magnetostatic
  closure, positive-frequency limits, physical independent references, nonlinear
  branch and harmonic truncation convergence, and power balances.

The original Bench nonlinear failure remains open. These kernels alone cannot
establish a coupled finite-body solution, production qualification, or close any
original physical threshold.

Independent source review of the kernel/cell primitive found no P1/P2: signs,
distributional contact handling and declared gradient boundaries match the code.
No GPU tests were repeated by the reviewer. This review excludes unimplemented
interfaces/HB coupling and is not independent finite-body physical acceptance.

## Conservative dual currents and vacuum exterior construction

The finite-body implementation uses an equivalent whole-space vacuum-current
representation to avoid inserting zero exterior conductivities into the screened
kernel. Let C map all body currents to their global static Biot-Savart field.
For positive bodywise-constant conductivities,

    E = -grad(phi_e) - C M; H = -grad(phi_m) + C J
    J = sigma_e E + source_J; M = sigma_m H + source_M.

`DualStaticCurrentBalance` uses the existing Crouzeix-Raviart face-mean gradient
and volume weights. Removing one potential gauge per body gives the weighted
projection P onto discrete currents with zero normal face jumps, including the
boundary's zero normal current. The dense coupled equations are

    J + sigma_e P C M = P source_J
    M - sigma_m P C J = P source_M.

The implementation checks the original coupled residual, reports field
reconstruction and individual channel normal-current jumps, and keeps source
current autograd. A declared dense preparation estimate rejects oversized cases;
it is not a measured global peak-memory certificate. Internal impressed currents
are a response-building interface, not a public user/internal-source capability.
The caller must supply validated disjoint volume regions; surface coupling binds
them to the complete owning native surface system.

`DualStaticExterior` maps both potential families to their exact surface face
means and uses the existing floating Laplace operator with separate zero net
charge/flux constraints. E adds -C M, H adds +C J. Electric scalar potential and
density use V,V/m; magnetic equivalents use A,A/m. The wrapper exposes those
units explicitly instead of relabeling the electric helper's receipts. Constant
body gauges cancel through floating offsets. This remains face-mean matching,
not certified pointwise tangential-trace or spatial convergence.

/tmp/gsp11-dual-current-v1.log3passed3.71s: nonzero two-body electric/magnetic
currents, conservative weak flux, reconstruction against native cell Biot-Savart,
source gradients, magnetic-conduction-to-zero limit against existing electric
FaceCurrentBalance, memory estimate refusal and geometry mutation rejection.
/tmp/gsp11-dual-exterior-v1.log1passed3.77s: complete surface-volume binding, neutral
flux in both channels, gauge-invariant external fields, gradients and owned-current
mutation rejection. No nonlinear/HB execution, physical accuracy or production
qualification follows from these tests. The original Bench failure remains open.

Independent review confirmed signs, projected current equations, potential
recovery and CR flux interpretation. It found a P2: exterior field layers could
be replaced by another valid object without invalidating the enclosing state.
The exterior guard now binds the closure, all volume bindings, both field-layer
identities and current tensor versions before evaluation. Native component guards
still run afterward. /tmp/gsp11-dual-current-exterior-v2.log4passed3.91s includes
the valid-but-wrong layer replacement control and finite residual/reconstruction
checks. Independent follow-up closes P2 with no new findings. No full regression
or physical convergence suite is claimed at this primitive integration stage.

## Nonlinear current feedback

`DualHarmonicCurrentClosure` prepares the linear impressed-electric-current to DC
E map R from the conservative dual-current system. With positive harmonic field
a(t), its covariance Q=<a a^T> and cubic moment b=<|a|²a>, the exact TPA source is

    T(E0) = tpa * [(|E0|²+tr Q)E0 + 2 Q E0 + b].

The closure solves E0-R T(E0)=0 by damped Newton, using the exact local cubic
tangent. Its convergence check applies to that original nonlinear equation and
to the final dual-current field reconstruction. There is no source polarization
DC derivative, no TPA double counting and no conductivity regularization. Scalar
or per-cell TPA coefficients are explicit. Finite input/tangent/residual checks,
iteration/backtrack budgets and typed errors preserve failure receipts.

At the converged root a native correction with a detached root Jacobian provides
first-order implicit derivatives to AC harmonics and TPA. This does not promise
second-order derivatives, branch stability or global root uniqueness. The dense
response is version-guarded together with its exact dual-current preparation.

/tmp/gsp11-dual-nonlinear-v1.log4passed5.10s: two-body nonzero DC response checked
against independent257-point real-time averaging of the actual AC+DC waveform;
original total electric current equals sigma_e E0 + mean TPA; implicit amplitude
and TPA derivatives match resolved central differences; TPA=0 yields exact zero
DC; small sigma_m agrees with the established electric-only nonlinear closure;
zero Newton budget retains an explicit failure and response mutation rejects.

Public HB wiring is deliberately still pending: current harmonic state/schema5
assumes a single electric face-potential family and a finite-permeability magnetic
interface response. The dual branch must preserve magnetic volume current,
magnetic face potentials and both floating offsets, together with native recovery,
decoder, preparation signatures and decode-only snapshot support. It cannot reuse
schema5 while discarding those unknowns. The original Bench conductive_limit
failure and all physical/truncation/power gates remain OPEN.

## Exact dual state component

`dual_static_snapshot.snapshot_dual_state` records every E/H/J/M array, both
face-potential families and neutral exterior density/offset pairs as owned CPU
bytes. Versioned preparation binding covers actual volume geometry/connectivity,
surface mapping, conductivities, discrete projector/curl/current matrices,
potential recovery maps and Laplace preparation arrays. Decode requires the exact
already-restored closure/bindings/Laplace preparation; it does not assemble one.

`restore_dual_state` rejects missing/extra state fields, changed preparation,
precision/shape/nonfinite arrays and qualification promotion. It rechecks both
field reconstructions, weak per-body current continuity, surface potential means
and neutral flux before creating owned field layers. It runs no linear/nonlinear
solve and preserves the exact stored state and outside queries. This state
component does not independently validate its original nonlinear source: the
outer HB payload must bind AC fields/TPA coefficients and check that current law.
It also does not yet provide complete preparation serialization or full Result
round trips. The public magnetic conductive_limit admission guard stays intact.

/tmp/gsp11-dual-state-v1.log1passed3.90s verifies exact two-body state and exterior
fields while monkeypatching linear solves and constructors to fail. Changed
magnetic current, E, layer density/offset and preparation identity reject.
/tmp/gsp11-dual-state-io-v2.log1passed3.87s adds an actual torch.save/weights_only
load round trip before that no-solve restoration. Independent combined review of
nonlinear closure and state component found no P1/P2; it confirms the outer
nonlinear constitutive and full HB/Result acceptance remain required.

The finite-permeability query path must not silently interpret magnetic conductive
DC data as B=mu_static H: the sampled constitutive law has a zero-frequency pole.
Dual state preserves H and magnetic current explicitly; finite-flux or pole-residue
semantics need explicit treatment when extending the public DC magnetic query.

## Rebinding complete dual numerical preparation

`dual_preparation_snapshot` saves the nonlinear response/settings, linear dual
current matrices, CR face gradients/topology/weights/recovery maps and complete
Laplace arrays. `restore_dual_preparation` binds these arrays to freshly restored
native surface/volume geometry and explicit caller-sampled sigma_e/sigma_m.
The independent expected binding covers the full linear preparation plus nonlinear
settings and response bytes; it must not be trusted merely because it is included
inside an untrusted preparation payload.

Restore validates geometry-affine gradient identities, exact face topology,
volume weights, potential recovery equations, projectors, the coupled matrix,
nonlinear source-response equations, Laplace geometry/matrix/LU consistency and
conductivities. It does not run kernel quadrature, inverses, factorization or
linear/nonlinear solves. Kernel arrays are bound by the independently retained
hashes rather than recomputed. The existing current-HB Laplace restoration was
extracted into one shared helper, preserving its validations.

/tmp/gsp11-dual-preparation-v1.log1failed4.04s: the full fresh-object restore and
re-execution succeeded, but an adversarial test used flatten()[0] on a noncontiguous
matrix and modified a temporary copy, so its expected rejection did not occur.
The test now changes [0,0] directly. Original evidence is preserved.
/tmp/gsp11-dual-preparation-v2.log2passed13.74s: actual weights-only disk reopening,
new geometry/binding objects, preparation/state restore under forbidden kernel
quadrature/constructors/solve/inverse/factorization, exact original exterior fields,
exact repeated nonlinear solve, operator/response/setting/material tampering
rejection; existing electric-current HB preparation/snapshot control also passes.
Independent read-only review found no new P1/P2 and confirmed shared Laplace
validation preservation. This is not full outer HB/Result acceptance: outer AC/DC
constitutive identity and dual state integration remain required.

## Public dual-conductive HB integration

All-body positive electric/magnetic conductivity now selects the coupled dual DC
closure during public HB preparation. Recovery retains the native computation
graph separately from owned persistence arrays. Schema 14 saves complete dual
preparation and state, with outer AC/DC constitutive and residual validation.
`Result.query_dc_dual` returns E/H, scalar potential parts, finite exterior B
and interior lim(omega B)=i sigma_m H in T/s. Interior finite B is refused.
Point monitors include DC H both inside and outside. Volume binding order may
differ from native region order; boundary potential arrays use native order.

The original Bench nonlinear draw, without dropping either conductivity or TPA,
passed public solve and exact Result persistence: /tmp/gsp11-nonlinear-candidate-control-v2.log,
1 passed in 65.61s. The original v1 conductive-limit failure is retained.
/tmp/gsp11-dual-public-v1.log failed in 2.84s because the new test fixture had
negative tetrahedral orientation; only fixture orientation was corrected.
/tmp/gsp11-dual-public-v2.log: 1 passed in 16.94s, covering public live nonzero
finite gradients, internal/external DC H monitors, interior flux-pole semantics,
exact disk restoration with solves forbidden and mutation rejection.
Independent review identified and then closed two P2 findings: detached live
queries and omitted DC H monitors. No further P1/P2 was found in that review.
These are integration controls, not physical or production qualification.
/tmp/gsp11-dual-integration-controls-v1.log: 4 passed in 14.87s, including\nreversed multi-body preparation/state rebinding, existing public electric-current\nHB monitoring/persistence and mixed DC snapshot restoration. No full regression\nwas repeated for this implementation increment.

## Integration checkpoint

At Maxwell 16bdc65 with Bench 46a982f, the complete related HB/DC test group
(test_harmonic*, test_dual*, test_conductive_static, test_static* and
 test_magnetostatic) passed: /tmp/gsp11-maxwell-hb-integration-v1.log and XML,
64 passed in 170.97s, session 43324 terminal exit 0. This is implementation
integration coverage, not complete Maxwell or physical acceptance. Independent
composition audit found no new P1/P2 in public solve/plan/archive integration;
dual nonzero DC current power is still explicitly refused by harmonic_observables
and must be implemented for complete energy diagnostics and qualification.

## Dual DC power diagnostics

The public harmonic power ledger now admits dual conductive states. DC surface
flux is the full real E cross H without a peak-phasor half factor. Interior work
is sigma_e |E|^2 + sigma_m |H|^2 + E dot J_nonlinear, with conductivities mapped by
region and nonlinear cell currents sliced in binding order. Interior H comes
from the dual query; no finite interior B is assumed. Existing independent
surface/volume refinement reuses these terms. All physical/global/quadrature
qualification remains false in the raw ledger.

Independent source review found no P1/P2 in signs, units or binding order. The
single-body term integration control is not heterogeneous multi-body, interface,
quadrature-convergence or physical energy acceptance.
/tmp/gsp11-dual-power-v1.log: 1 failed, 2 passed in 30.50s. Public dual query and
existing nonlinear power controls passed; actual AC interior field queries in
the new full-power test exhausted order 48 at the original tolerance. This
failure is retained. The follow-up increases only the field order budget to 192,
keeping original relative/absolute tolerances unchanged.

/tmp/gsp11-dual-power-v2.log: 1 passed in 19.96s with max_order=192 and
unchanged field tolerances. Public DC flux, both positive ohmic contributions
and nonlinear work agree with direct queried-field integrations. Actual relative
balance residuals for DC/fundamental/second harmonic are respectively
1.9068600048052518e-08, 0.17554545811664668, 0.12841356051965153. These AC defects
remain physical failures, not qualified energy closure. v1 passed controls were
not repeated. Sessions 30827 and 52967 are terminal.

Adaptive comparison receipts now include retained_field_rms_V_per_m. This enables
archive consumers to reconstruct each threshold from its frozen policy and bind
final observations to the exact saved state. It is a recorded local estimator;
earlier step differences remain execution observations unless intermediate
fields are separately retained. No physical truncation qualification is added.

## Dual harmonic power refinement checkpoint (2026-09-22)

Added optional AC query batching and detached CPU progress callbacks to public
harmonic power diagnostics. Default behavior remains unbatched. Each completed
integration step can be preserved before a later query fails. Adaptive batch
stopping orders may differ; no bitwise equivalence is claimed. The diagnostic
runner saves owned snapshots, exact source inventory, partial reports and a
nonfinite-safe terminal receipt. Snapshot reuse reads one verified byte buffer
and checks numerical/HB settings, exact volume bindings and the complete float64
300 MHz harmonic frequency axis before native restoration.

Actual control v1 exhausted interior E query order 192 after 31.23s, preserving
result.pt and the failure receipt but no intermediate power steps. V2 reused
that result without solving, with query max_order=384 and batch_size=32 at the
same rtol=1e-5/atol=1e-10. It terminated exit 1 after 63.10s: interior H relative
successive-order difference 8.394789909834375e-5 exceeded 1e-5. Process peak CUDA
allocation was 29075393024 bytes. Both failures remain preserved under
/tmp/gsp11-dual-power-refinement-v1 and /tmp/gsp11-dual-power-refinement-v2.
Four complete V2 observations are retained: surface/volume 2/2, 4/2, 8/2, 8/3.
The last relative DC/H1/H2 power residuals were 1.146770184785266e-13,
0.019271331398019172 and 0.05037587348302499. These are partial observations,
not converged power integrals or production qualification; volume order 4 failed.

Targeted tests: /tmp/gsp11-power-progress-tests-v1.log, 3 passed in 20.01s
(callback isolation and partial-failure evidence, nonfinite JSON/frequency guards,
and existing actual public dual-power term accounting). Independent audit closed
nonfinite receipt, single-buffer restoration and omitted-frequency P2 findings;
no further P1/P2 found. Sessions 73265, 37660 and 69080 are terminal. No broad
regression repeated. Next numerical work is interior field quadrature convergence
and memory, followed by mesh/volume convergence and independent physical checks.
GSP-01 remains OPEN; original Goal scope and all acceptance thresholds remain.

## High-order query workspace checkpoint (2026-09-22)

Near RWG and analytic-carrier quadrature now limits query-panel batches by
262144/(3*order^2) fan samples. Volume Green integration flattens query/cell/face
pairs into the same bounded expansion, then restores the original four-face
cell sum. Broadcast-compatible pair distances avoid a Cartesian expansion.
Exact static/contact terms and native derivatives remain; PV edge rejection
checks complete cell faces before pair partitioning. A single very high-order
pair, outputs and retained autograd graphs are not bounded by this work budget.
This is not a total peak-memory guarantee or bitwise-equivalence claim.

Targeted existing volume/surface-volume/lifted/near-field regressions:
/tmp/gsp11-query-memory-tests-v1.log, 27 passed in 64.78s.
New partition invariance test checks multiple queries/cells, potential/gradient/
Hessian values, query derivatives, PV edge rejection and empty PV queries:
/tmp/gsp11-query-pairs-tests-v2.log, 1 passed in 4.30s. The v1 fixture failure
(disconnected cells in one body) is retained and corrected to a center-fan mesh.
Independent audit found no P1/P2; its suggested cross-cell/PV checks are covered.

Actual same-snapshot public power control v3 completed in 192.443s, exit 0,
under /tmp/gsp11-dual-power-refinement-v3 (session 47483 terminal). It retained
all five steps with unchanged field rtol=1e-5/atol=1e-10, integration rtol=1e-4/
atol=1e-20, max_query_order=768. Previously failing volume order 4 now completes.
Final CUDA process allocation high-water mark: 1218397184 bytes, versus
29075393024 bytes in the preserved v2 order-384 failure; different terminal
budgets/workloads, not a same-workload speed/scaling certificate. Common completed
steps agree to roundoff. Input snapshot SHA256 remains
b6a700fc9dfbac81efc83d6273aa6ed8f7c095a0d288dbd75af1491ff76ba6e0.

Final surface/volume orders 8/4, relative DC/H1/H2 power residuals:
1.146058347302466e-13, 0.01927546654141604, 0.025402260599599668.
Both integration-refinement axes remain FAILED. No physical qualification:
next work must refine independent surface/volume integration and then the
underlying spatial solution, retaining all prior physical failures. GSP-01 OPEN,
full Goal ACTIVE, scope and acceptance thresholds unchanged. No full regression
repeated. Sessions 14837, 18283, 82199 and 47483 are terminal; no live runs.

## Analytic affine Coulomb extraction checkpoint (2026-09-22)

Power control v4 reused the original solved snapshot, surface orders 8/16/32,
volume orders 4/6/8, unchanged max query order 768 and all tolerances. Surface
8->16 passed, so surface 32 was correctly skipped. Volume 6 failed after
426.841s with H maximum relative successive difference 8.143041862089068e-5
above 1e-5. Evidence: /tmp/gsp11-dual-power-refinement-v4; session 32007 terminal.

Small same-snapshot diagnostic /tmp/gsp11-query-components.py (receipt and log
under /tmp/gsp11-query-components-v1) separated H2 surface/volume fields at the
first 32 volume-order-6 points. Its combined order differences reproduce the
v4 failing receipt: at 384 surface H difference 1.4025017036914734e-11 versus
volume 6.925686428298577e-18. Worst-point term diagnostic
/tmp/gsp11-query-surface-terms.py and /tmp/gsp11-query-surface-terms-v1.json/log
found vector potential 1.4024822544622034e-11, divergence 1.8439681909311905e-17,
and curl 7.0680520652622055e-16. These are internal diagnostic observations,
not independent physical references. Sessions 2663/77670 terminal.

Near RWG vector potentials now extract the complete affine Coulomb integral:
J(query)*integral(G0) + slope_sum*integral((source-query)*G0). The vector moment
uses the planar divergence theorem and exact edge integrals of radial distance;
the remaining numerical Green term is expm1(i*k*r)/(4*pi*r). The existing VIE
static scalar helper moved to fields for shared use without formula changes.
Independent audit found no P1/P2 in normal/conormal signs, RWG reconstruction,
gradients or migration. Extreme edge/vertex cancellation remains unresolved;
no claim of resolving that boundary or certifying physical error is made.

/tmp/gsp11-static-moment-tests-v1.log: 31 passed in 7.29s, including independent
surface cubature, query derivatives and existing field/VIE/lifted controls.
/tmp/gsp11-static-moment-orientation-v1.log: 6 passed in 3.65s, explicitly adding
negative normals and vertex reversal. The latter light check ran concurrently
with v5; v5 timing must not be treated as an isolated performance benchmark.
Sessions 47401/16265 terminal. No full regression repeated.

Power v5 final: /tmp/gsp11-dual-power-refinement-v5, session 47385 terminal,
exit 0, 1059.7646s. All four planned steps completed; surface 8->16 passed,
volume 4->6->8 still failed the polarization-work refinement criterion. At 8,
linear absorption and nonlinear dissipation terms passed their refinement tests.
DC/H1/H2 relative power residuals: 6.520431342417711e-14,
0.01928733324554301, 0.006754097519314863. Actual max field order 768,
peak allocated bytes 1247335936. This is completed_unqualified, not physical
power closure. Runtime overlapped light tests and later a spatial control and
must not be used as an isolated performance benchmark. Original failures and
all thresholds remain; Goal ACTIVE and GSP-01 OPEN.

## Public source quadrature budgets and spatial control (2026-09-22)

HarmonicBalanceConfig now exposes source_max_order=32 and
source_max_inner_order=64, strict integers >=3. Initial preparation and added
cutoff harmonics pass both limits to independent source-matrix quadrature.
Its receipt records initial/max outer/inner orders. Error tolerances remain
rtol=1e-4 and atol=1e-12 (split between axes), unchanged. Native snapshots and
Bench bindings preserve explicit budgets; absent fields in pre-budget native
snapshots mean the original fixed limits 32/64. The power-control runner only
fills these two historical defaults before strict configuration comparison.
Bench new frozen variants require all settings explicitly; old plans are not
rewritten to inherit new declarations silently.

New public spatial diagnostic refines the same flat tetrahedral physical body,
with unchanged materials, sources, frequency, pump sequence and solver tolerances.
Level 1 has 32 volume cells / 16 surface faces. Actual v1 source-max32 failed in
51.2197s: outer order32 maximum threshold ratio1.3777791313212508, inner8 passed.
Evidence: /tmp/gsp11-dual-spatial-level1-v1, session99525 terminal. This original
failure is retained. Actual v2 changes only the outer budget to64 and completes
in409.5519s, original nonlinear residual1.8726382648472964e-16, threshold
4.539515811517305e-9. /tmp/gsp11-dual-spatial-level1-v2/result.pt SHA256
8981c8d98abb73dea3d83eaf08d0a3074ba92c44ac32b3b15db2d76fda8df30d.
Session45628 terminal; timing overlaps the targeted native test and is not an
isolated performance benchmark. Solved_unqualified; no spatial certification.

Targeted controls: spatial-setup v1 test failed on comparing auto material IDs
from two separately constructed scenes; corrected to compare refinement of the
same scene, preserving the strict identity assertion (v2:1pass3.79s).
/tmp/gsp11-source-budget-tests-v1.log:3pass3.49s, spatial invariants/strict budget
validation/diagnostic evidence guards. /tmp/gsp11-source-budget-native-v1.log:
1pass27.93s, actual64/96 public solve/gradients/exact snapshot/no-solve restore.
/tmp/gsp11-source-budget-archive-v1.log:2pass4.59s, Bench explicit setting and
exact archive/tamper controls reusing the existing actual nonlinear candidate
snapshot, without solving again. Independent audit found no P1/P2 in geometry,
configuration propagation, defaults, storage or qualification boundaries.

No-solve coarse/fine comparison /tmp/gsp11-dual-spatial-compare.py completed in
1.913s; full values, hashes, query receipts and failures are retained under
/tmp/gsp11-dual-spatial-compare-v1. Coarse cell fields are prolonged to the eight
parent-major child cells and compared in fine-volume-weighted RMS. Relative
DC/H1/H2 electric-field differences:0.1507252311/0.1216496671/0.2928099349;
absolute RMS differences:6.9528002e-7/0.0954618754/3.9750198e-5 V/m.
Two shared interior probes have H1 E differences3.56-5.04%, H2 E6.29-23.34%,
H2 H37.71-78.82%. Large DC H relative differences correspond to tiny fields
(max absolute interior difference6.74e-13 A/m), so relative values alone are
not physical thresholds. Two exterior probes: H1 E/H relative differences
below4.64e-5; H2 E/H about9.8-12.9%. Neither the few exterior probes nor the
small algebraic residual proves spatial/global physical accuracy.

All sessions14851/76304/23659/89110/40722/99525/45628/21099 are terminal; no live
runs. Next work: reduce repeated per-column response preparation before deeper
spatial/volume refinement, and continue the missing tensor/periodic/fast-volume
combinations. GSP-01 OPEN, full Goal ACTIVE, original acceptance unchanged.
No new full regression; final cross-repository tests and independent audits due.

## Shared field-column quadrature checkpoint (2026-09-22)

The primary field potential routine now integrates the three geometric RWG
panel basis functions once per query/panel pair, then contracts electric and
magnetic coefficients for every response column. Scalar inputs still return
[query,3]; private fixed-order interior columns return [query,3,column]. The
private entry validates all column values and shares dtype/device/region/trace
checks with scalar decode. No quadrature-by-column tensor is created, but local
coefficients, panel/column contractions, dense outputs and autograd storage
still grow with column count. This is not a complete workspace guarantee.

prepare_cell_field_response uses shared field columns in each refinement step.
Original complete E/H checks still inspect every query and source column;
background carrier evaluation and direct volume response terms are unchanged.
No tolerances or physical qualification flags changed. Independent source audit
found no P1/P2 in contractions, shape/order, interior sign, guards or gradients.

/tmp/gsp11-field-columns-tests-v1.log:38 passed in67.09s, including float32/64
column/individual value and gradient comparisons, empty/invalid-region/trace/
nonfinite guards, analytic field, lifted, surface-volume and periodic controls.
The column tests share current geometry kernels, so a retained pre-change
source reference was also evaluated on the actual saved level1 response.
Reference source: /tmp/gsp11-fields-before-columns.py (hash in receipt).
/tmp/gsp11-column-control.py reads the reference and native snapshot once into
verified bytes. v1 failed before comparison because the loaded old module has
its own FieldQueryConfig type; v2 uses that original type. Failure retained.
/tmp/gsp11-column-control-v2/receipt.json and matching log: both actual harmonics,
96 source columns and32 cell queries. H1/order24 old5.01370s vs shared0.052854s;
H2/order48 old8.16901s vs shared0.084240s. Process allocation peaks respectively
old91708928/shared90273280 and old202032128/shared197401600 bytes. Single timing
observations, no repeat/crossover/all-size/full-solver performance certificate.
Max absolute E/H differences: H1 8.882e-16/3.036e-18; H2 4.441e-16/7.999e-19.
Five-column three-query coefficient-gradient differences:2.274e-13/1.172e-13.
Same-discretization implementation equivalence is not independent physics.

Sessions72023/45145/85536 terminal, no live runs. No new full regression or
repeat of the expensive nonlinear solve. Existing coarse/fine spatial failures
and power refinement failures remain. Next: source-matrix assembly cost and
fast volume/surface actions, deeper spatial integration/mesh convergence and
remaining capability combinations. Full Goal ACTIVE; GSP-01 OPEN, all original
scope, thresholds and final cross-repository acceptance requirements unchanged.

### 2026-09-22 — streamed induced-volume source matrix preparation

The weak source matrix now integrates one boundary face and a bounded query
batch at a time. Query/cell work is capped at262144 elements (at least one
query); caller query_batch_size can lower this cap. Exact original triangle
nodes, RWG signs, global J/M row layout and binding/cell/xyz column layout are
preserved. cell_green_potentials optionally skips its unused dynamic Hessian
and returns None in that slot; complete-Hessian callers remain unchanged.
Static panel potential/gradient evaluation and PV/edge/vertex guards remain.
The cap is not a byte or total peak bound: dense outputs, full face nodes,
constant geometry and retained autograd graphs are outside it. Floating sum
association may differ. Existing adaptive source tolerances are unchanged.

Targeted /tmp/gsp11-source-streaming-tests-v1.log:18 passed in64.26s covering
Green values/query derivatives, skipped Hessian work, empty/invalid requests,
heterogeneous multi-body reversed binding columns, chunking and existing
surface-volume controls. Independent source audit found no P1/P2 issues.

/tmp/gsp11-source-stream-control-v1/receipt.json: actual saved level1 native
snapshot SHA8981c8d98abb73dea3d83eaf08d0a3074ba92c44ac32b3b15db2d76fda8df30d,
32 cells,16 boundary faces,48x96 source matrix, H1 outer32/inner8. Public
restore only, no nonlinear re-solve. A diagnostic layout view supplies the
same owning system/device/dtype/shape to retained old and current assembly.
Retained source /tmp/gsp11-source-matrix-before-streaming.py uses the current
unchanged default complete Green kernel: implementation comparison, not an
independent physical reference. Both source hashes and exact matrices saved.
Maximum absolute matrix difference4.33893e-19, relative Frobenius1.22594e-16.
Old9.99367s/867597824 peak allocated bytes; streamed9.01901s/79780864 bytes.
Additional peak above each baseline857610240/69791232 bytes respectively.
These are single observations, not repeat/crossover/full-solver complexity
qualification. Control script /tmp/gsp11-source-stream-control.py, session10318
terminal exit0; test session56892 terminal. No live runs or full-suite rerun.

Existing coarse/fine spatial and power failures remain; no new capability has
production qualification. GSP-01 OPEN, complete Goal ACTIVE, original scope
and acceptance thresholds unchanged. Next: exploit reduced preparation cost
for deeper spatial convergence, then fast volume/surface actions and remaining
capability combinations. Full regression/cross-repository acceptance remains
required at major integration and final acceptance points.

### 2026-09-22 — selected volume Green blocks for fast-action preparation

Development is isolated in <scratch>/witwin-maxwell-volume, branch gsp11-fast-volume,
based on ab5fa99. Primary <scratch>/witwin-maxwell-gsp09 remains unchanged while its
level2 spatial control runs in session14033. No primary source hashes changed.

cell_green_potentials accepts optional CUDA int64 cell_indices. Selection is
applied before tetrahedron/face construction and quadrature, preserving order
and repetitions; invalid device/dtype/rank/range rejects. Empty selections and
queries return explicit zero-size outputs. The original complete owning mesh,
static self contact, selected-cell face PV and edge/vertex guards remain.
cell_current_field_blocks exposes [query,xyz,cell,xyz] E/H blocks using these
potentials. Existing full dense response assembly calls this shared helper,
retaining original complete per-column E/H refinement checks and tolerances.
No disconnected partial mesh is constructed, and no unselected cell quadrature
is performed. Full geometry validation is still required; outputs and autograd
storage remain dense in requested query/cell count. This is a selected fixed-
order provider, not yet an H/H2/FMM volume operator or total memory guarantee.

Independent read-only source audit found no new P1/P2 in selection ownership,
self/trace behavior, E/H sign/axis layout, empties or gradient flow. Tests compare
selected and full Green blocks, repeated-column scatter actions, query/current
gradients, adjoint identity, and selected-only quadrature instrumentation.
These equivalence controls are not independent physical qualification; existing
volume Green tests provide the separate direct cubature/self-contact controls.
No spatial, energy or original physical failure is closed by this change.

/tmp/gsp11-selected-volume-tests-v1.log:22 passed in87.02s; session31268 terminal
exit0. No full-suite rerun. The tests overlapped the ongoing spatial solve, so
neither wall time is an isolated performance measure. Goal ACTIVE; GSP-01 OPEN.
Next: tree/basis projection and selected near/far volume actions, followed by
original-source accuracy/adjoint/storage controls and coupled HB integration.

### 2026-09-22 — regional volume hierarchical action development

VolumeCellBlocks selects original cell-centroid scalar E/H integration in
component-major Cartesian ordering. VolumeHierarchicalAction reuses the CUDA
support octree, bounded pair traversal and verified streamed adaptive cross
approximation. E and H are separate 3N-by-3N regional actions; no surface
coupling or nonlinear solve is introduced. Far blocks use sqrt(Vrow)/sqrt(Vcol)
scaling for cell-volume L2 coordinates; physical factors are unscaled on return.
Every far entry is scanned for the block Frobenius residual check. This does
not certify quadrature, global operator or physical discretization error.
Failed rank verification subdivides the block; terminal exact near blocks stay
within the configured leaf size. One retained object supports batched forward,
complex adjoint and current-vector autograd. Mesh, representative points, tree,
indices and factors have binding/version guards.

Initial tests v1 (2 failures,1 pass) had no far blocks at leaf8; the required
far-coverage assertion exposed this and was kept. v2 (2 failures,1 pass) with
leaf4 rejected oversized leaves: several tetrahedra share box centers. The
shared octree now accepts explicit fixed interior partition_points solely for
Morton keys. Volume trees use actual cell centroids; node bounds still enclose
full tetrahedra and admissibility still uses those boxes. Quantization-induced
coincident keys still reject oversized leaves. No guard was removed.

Independent source audit found no P1/P2 in component ordering, samplers,
volume scaling, partition subdivision, adjoint or mutation guards. Incremental
partition-point audit also found no P1/P2. Synthetic forced root compression
rejection separately tests recursive exact-leaf coverage in both precisions;
it is not the physical far-interaction test. The initial64-cell E/H control produced geometric far blocks but no retained
far factors: all verified factors failed the storage-reduction requirement and
correctly fell back to exact blocks (v3:2fail3pass,178.03s). The far assertion
was retained. The replacement96-cell conforming elongated body has16 cubes,
each split into six positively oriented tetrahedra, with leaf6/admissibility1.
It requires actual near blocks and storage-reducing far factors, and compares
original fixed-order fields, complete partition coverage, dense actions,
adjoints and current gradients at unchanged compression tolerance1e-8.

Retained accounting includes referenced mesh geometry, tree, coefficient index
maps, exact blocks and low-rank factors. Near/factor ceilings are checked before
block construction and final retained bytes measured. Geometry tree creation,
quadrature/cross workspaces, host receipts, allocator overhead and backward
graphs are outside the budget; this is not an OOM or setup-peak guarantee.
Compression setup scans all original far entries; no fast setup, linear storage
slope, crossover, coupled HB qualification or physical acceptance is asserted.

Validation: /tmp/gsp11-volume-hmatrix-tests-v4.log:2passed3deselected in205.62s
for actual E/H far controls. Remaining3 controls passed in v3: retained-budget/
mutation/configuration plus synthetic forced recursive rejection in complex64
and complex128. They were not repeated. /tmp/gsp11-volume-cluster-tests-v1.log:
6passed10.03s, including default-tree regressions and same-support/different-
representative splitting, full-box near classification and invalid points.
Sessions63390/11453 terminal exit0. All failure logs retained. No full-suite
rerun. Tests overlapped primary spatial session14033; timings are not isolated
performance observations. Primary ab5fa99 worktree remains unchanged and live.
Full Goal ACTIVE; GSP-01 OPEN. Next: nested volume H2 conversion and original
source error/storage controls, then complete surface/volume HB integration.

### 2026-09-22 — nested regional volume H2 conversion

VolumeNestedH2Action converts the verified regional H object with three
component-major cell axes and sqrt(Vrow)/sqrt(Vcol) scaling. Packed near storage
is copied, then referenced by the nested action; source H exact storage is not
shared. Far blocks become nested test/trial bases, transfers and couplings.
The existing H2 converter checks each reconstructed coupled far block against
the source H factors. H compression error and fixed-order quadrature error
remain additional; no global operator, physical or coupled HB claim follows.
The wrapper retains provider geometry and the tree, but not the source H object
or far factors. Its source receipt is a copied diagnostic only.

Retained budgeting subtracts unique referenced geometry/tree storage and
reserves the structural H2 owned upper bound before allocating packed near.
Nested construction independently checks its owned allocation bound. Final
wrapper accounting deduplicates actual CUDA storage across provider, tree,
near blocks, both bases, scales and couplings. This budget excludes a source H
object still held by the caller, conversion/seed/SVD workspaces, allocator
state, host diagnostics and backward graphs; it is not a conversion peak bound.
Independent read-only source audit found no P1/P2 in indices, scaling, ownership,
capacity accounting or mutation guards. New targeted tests cover actual E/H
far conversion against H and original fields, adjoints/current gradients,
source-object weakref release, all-near action, capacity and mutation rejection.

/tmp/gsp11-volume-h2-tests-v1.log:3passed204.86s, session58492 terminal exit0.
Actual96-cell E/H actions contain retained far couplings and verify H/original
field agreement, batched complex adjoints, dot identities and current gradients.
The source H object is garbage-collected before original-field comparisons;
mutation of a nested coupling and copied packed near each rejects. The all-near
case also passes, and a one-byte retained budget rejects before nested allocation.
No repeated full regression. Source audit found no P1/P2; its minor copied-near
wording clarification is included. Tests overlap primary spatial session14033,
so wall time is not isolated performance. No complete fast HB/global operator/
physical or scaling qualification. Goal ACTIVE; GSP-01 OPEN.

### 2026-09-22 — selected weak surface-volume cross blocks

build_volume_source_block binds one exact owning surface/volume region and
selects local J edges followed by local M edges, with cell/xyz columns. Row and
cell order/repetitions are preserved; empty and invalid selections are handled
explicitly. Complete RWG support faces are derived from face_edges, the same
incidence used by weak testing. Only selected-support faces and selected cells
are integrated, with the original [+<f,Hv>,-<f,Ev>] formula and transferred
surface divergence term. Original nodes, query/cell batching, PV treatment and
unused-Hessian omission remain. Full global source assembly reuses the helper;
J/M routing and binding/cell/xyz ordering and adaptive tolerances are unchanged.
This is a fixed-order cross-block provider, not a compressed cross action.

Independent initial audit found P2: using unguarded plus/minus face caches could
silently drop a support face after cache mutation. Selection now derives from
face_edges instead, eliminating the new cache dependency. A regression changes
only plus_faces and verifies unchanged output and exactly two integrated faces.
Independent re-review closed P2. This does not certify all RWG internal-array
mutation protection. The old implementation was retained for a separate control.

/tmp/gsp11-selected-cross-tests-v1.log:13passed83.90s (selected blocks, streaming
and existing surface-volume controls), session86093 terminal. After the audit
fix, /tmp/gsp11-selected-cross-tests-v2.log:3passed4.33s, session7112 terminal.
Covered complete supports, no unselected-cell quadrature, repeated/reordered
rows/cells, owner/empty/invalid guards and native current/adjoint gradient.
No repeated full regression or nonlinear solve.

/tmp/gsp11-selected-cross-control-v1/receipt.json: same saved actual32-cell
level1 snapshot SHA8981c8d98abb73dea3d83eaf08d0a3074ba92c44ac32b3b15db2d76fda8df30d,
48x96 matrix at outer8/inner6. Full selected-helper assembly is bitwise identical
to retained pre-change /tmp/gsp11-source-before-selected.py, with source hashes
and matrices stored. Session2845 terminal exit0, elapsed2.96955s. Single timing
observations overlapped the primary spatial control and support no performance
claim. This is implementation equivalence, not independent physical accuracy.
Main spatial session14033 remains live in unchanged ab5fa99 primary worktree.
Goal ACTIVE; GSP-01 OPEN; original physical failures and thresholds unchanged.
Next: rectangular hierarchical surface-volume cross actions and complete
surface/volume response integration, followed by public HB and exact persistence.

### 2026-09-22 — rectangular surface-volume hierarchical development

The volume H block engine is extracted into hierarchical_blocks.py to share
partition, compression, capacity accounting, mutation guards and actions.
It accepts independent test/source trees, component counts and positive scales;
rectangular forward and adjoint validate different input/output dimensions.
VolumeHierarchicalAction remains a regional wrapper with its original volume
L2 scaling and .tree access for nested H2 conversion. Persistent scale vectors
are now explicitly counted and guarded. No duplicate compression algorithm.

SurfaceVolumeBlocks owns a BoundVolumeRegion and a lightweight immutable
system/device/dtype/shape layout, not a dense surface operator. Complete RWG
support boxes are accumulated through face_edges; source boxes cover full
tetrahedra. All referenced system/basis/mesh tensors, support boxes and scales
are version guarded. Original selected weak blocks become J/M component-major
rows and component-major cell columns. SurfaceVolumeHierarchicalAction uses
2 row components,3 column components and separate trees. Cross scaling uses
impedance times inverse square-root RWG Gram diagonal for J testing, inverse
square-root diagonal for M testing, and inverse square-root cell volume for
current coefficients. This is an explicit diagonal dual proxy, not full Riesz
norm certification. Fixed-order block compression/error limitations remain.

Independent source audit found no P1/P2 in dimensional mapping, supports,
ownership, extraction equivalence, memory accounting or guards. Audit requested
nonfirst-region coverage: heterogeneous two-body region2 local rows now match
its global J/M slices; other body source rows remain zero, and provider retains
no original dense matrix storage. /tmp/gsp11-rectangular-region-tests-v1.log:
1passed4.34s, session46325 terminal. Main cross/volume/H2 regression /tmp/gsp11-rectangular-h-tests-v1.log:
8passed2deselected604.16s, session56670 terminal exit0. Actual rectangular
far compression, original matrix actions, distinct input/output dimensions,
complete pair coverage, complex adjoints/current gradients, volume rejection
controls and E/H nested H2 regression all passed. The two prior standalone
volume original-action cases were not repeated; H2 cases check original fields.
Retained budgets exclude tree/quad/compression workspaces, host and backward
state; no setup peak, global Riesz error, coupled solve or physical certificate.

No full-suite rerun. Independent source audit found no P1/P2 and its nonfirst-
region followup is covered. Timings overlap primary spatial session14033 and
are not isolated performance qualification. Goal ACTIVE; GSP-01 OPEN. Next:
selected surface-to-volume field blocks and full coupled compressed response,
with original equation/error checks before public nonlinear integration.

### 2026-09-22 — selected surface-to-interior field blocks

Internal selected_interior_field_blocks computes only the requested complete
RWG supports, with ordered/repeated J/M columns, scalar interior ownership and
full-geometry query/trace validation. Near-panel affine static potential and
static-gradient extraction remain unchanged. Selected integration maps local
faces back to global face ids for normals and trace checks. Empty selections
preserve shapes. Native query and current gradients remain connected. Selected
identity storage is quadratic in unique selected edges; full-geometry query
classification remains. This is fixed-order integration, not a global memory,
quadrature-error or physical certificate.

/tmp/gsp11-selected-field-tests-v1.log:25passed6.02s, session66811 terminalexit0.
Includes selected/full fields, near integration, complete supports, repeated
indices, empty/invalid inputs and query gradients. Explicit current-gradient
followup /tmp/gsp11-selected-field-gradient-tests-v1.log:2passed1deselected3.87s,
session58130 terminalexit0. Independent source audit found no P1/P2.

Retained pre-change /tmp/gsp11-fields-before-selected.py separately evaluated
saved actual32-cell state SHA8981c8d98abb73dea3d83eaf08d0a3074ba92c44ac32b3b15db2d76fda8df30d.
/tmp/gsp11-selected-interior-control-v1/receipt.json and fields.pt bind source
and snapshot hashes. Session27479 terminalexit0, completed_unqualified.
H1/order24 E/H maximum differences4.440892098500626e-16/8.673617379884035e-19;
H2/order48 differences3.66205343881779e-15/8.673617379884035e-19.
No nonlinear re-solve or repeated full suite; this is implementation equivalence,
not an independent physical reference. Next: rectangular surface-to-volume
hierarchical field maps and complete compressed response integration.
Primary spatial session14033 remains live at unchanged ab5fa99; do not restart
or merge package/runner changes before its source-bound terminal check.
Goal ACTIVE; GSP-01 OPEN; all prior physical failures and thresholds preserved.

### 2026-09-22 — rectangular surface-to-volume interior H maps

Previous selected interior field stage committed7dcb83c. New
InteriorVolumeBlocks maps local J/M RWG currents to cell-centroid E or H,
using complete cell boxes and complete RWG two-face support boxes. It reuses
SurfaceVolumeBlocks geometry/ownership and the shared rectangular H engine,
retaining no dense surface operator or decoder handle. Row scaling is sqrt(V);
column scaling is [Gdiag^-1/2,abs(eta)*Gdiag^-1/2], an explicitly labeled
impedance-balanced diagonal RWG proxy. Field integration is fixed-order with
original negative interior sign and analytic near-panel terms. Components are
cell-component-major rows and J-then-M columns. Geometry/scale/action guards
and retained-storage accounting include shared referenced tensors.

/tmp/gsp11-interior-h-tests-v1.log:3passed139.52s, session55758 terminalexit0.
Actual elongated96-cell E/H far compression and exact leaves are present;
every matrix coefficient is covered once. Original full field action, batched
adjoint/dot/current gradient, repeated selection, empty/invalid inputs,
capacity and mutation controls pass. Independent source audit found no P1/P2;
its nonfirst-region/precision gaps are covered separately in
/tmp/gsp11-interior-h-regions-v1.log:4passed4.46s, session55186 terminalexit0.
Those four tests use a heterogeneous two-body region2, E/H, float32/float64,
original full-column decoding and native current gradients. No full suite or
nonlinear solve rerun. Timings overlap primary spatial solve and confer no
performance qualification. Fixed-order algebraic agreement does not establish
quadrature convergence, full Riesz error or physical production qualification.

Next integration: weak volume-to-surface source map, common surface solve,
interior surface-to-volume field map and direct volume Green actions. Preserve
regional/global J/M routing and convert component-major internal layouts to
binding/cell/xyz response layout explicitly. Validate full forward/adjoint,
original surface residuals and complete surface+volume field errors before
public nonlinear integration and persistence. Cross H2 conversion remains open.
Primary spatial session14033 is still live atab5fa99; no restart or source merge
until its terminal/source check. Goal ACTIVE; GSP-01 OPEN; physical failures and
original scope/thresholds unchanged.

### 2026-09-22 — fixed surface inverse with implicit RHS differentiation

Prior interior H stage committed9a88e1f. Complete fast response composition
requires differentiable matrix-free surface inversion: existing correct/GMRES
is deliberately no_grad. FixedSurfaceInverse now supplies A^-1 and A^-H via
the existing CUDA GMRES/block-GMRES and native implicit autograd, recursively
supporting higher RHS derivatives. No inverse matrix, iteration tape or RHS
is retained. Operator/material derivatives are explicitly excluded. Each
orientation uses Euclidean true residuals, with swapped real left/right scales
for the conjugate-transposed equation. Original operator/compression error and
condition-dependent solution/gradient error remain separate obligations.

Independent audit found P2 in initially accepting arbitrary surface protocols:
DenseOperator.refresh accepts matrix mutation, so delayed backward could use a
new matrix. Admission now uses exact H/H2 classes with rejecting fixed-action
guards, and rejects mutable dense in-place/replacement controls. Audit closed
that P2. Other backend admission, including FMM, remains open for this inverse.

/tmp/gsp11-implicit-solve-tests-v1.log:1failed2passed6.15s, session60323 terminal.
GMRES adjoint hit stagnation_window20 before completing its24-dimensional
cycle. Diagnostic /tmp/gsp11-implicit-adjoint-failure-v1.json, session76502
terminal, preserves the failure. Test window increased to48 with unchanged
rtol1e-11/atol1e-13; no residual threshold relaxation.
v2:2failed4passed9.85s, session86232 terminal; large backward RHS still failed
ordinary GMRES. Added configurable bounded residual correction (default2),
using the same Krylov configuration per pass. Final acceptance independently
checks the complete original-RHS residual against max(rtol*norm(rhs),atol).
All attempts and final residuals/thresholds are recorded; exhausted failure
raises with receipt. Iteration budget explicitly applies per RHS per pass.
/tmp/gsp11-implicit-solve-tests-v3.log:6passed10.69s, session15012 terminalexit0.
Actual heterogeneous two-body H and H2, ordinary/block GMRES, dense forward and
adjoint references, native second RHS derivatives, delayed geometry mutation,
dense admission rejection and exhausted-budget failure are covered.

No full regression or new nonlinear solve. Complete response is still pending:
assemble regional source and field actions around this inverse plus direct
volume blocks, preserve binding/cell/xyz layouts and global J/M maps, and check
original equation residuals and complete field errors before public HB wiring.
Primary session14033 remains live on frozenab5fa99; no restart or source merge.
Goal ACTIVE; GSP-01 OPEN; original physical failures and thresholds unchanged.

Residual-refinement independent re-review found no P1/P2. Its finite-threshold
suggestion is implemented: an infinite/NaN acceptance threshold cannot pass.
/tmp/gsp11-implicit-solve-threshold-v1.log:1passed6deselected3.54s, session12316
terminalexit0; existing six successful cases were not repeated. Returned Krylov
failure logs are retained in attempts; a direct backend exception propagates
and may precede publication of the current receipt. This limitation is not a
claim that every exceptional backend failure has a completed solver receipt.

### 2026-09-22 — complete compressed induced-volume action composition

Previous fixed inverse stage committed859f0e9. CoupledVolumeAction now composes
D + F A^-1 B and D^H + B^H A^-H F^H. B uses regional selected weak-source H,
A the guarded global surface H/H2 inverse, F regional interior field H and
D direct volume H/H2. It supports arbitrary caller binding order and subsets;
interbody effects propagate through the full surface system. Public-facing
response vectors retain binding/cell/xyz order, with internal component-major
conversion and global J/M edge-offset scatter/gather. No dense response matrix,
incident background or duplicate linear contrast is added.

Constructor enforces common system/exact binding/mesh/medium, unique regions,
field type, field quadrature order/centroid equality, dimensions and precision.
All component guards and index-map identities/versions are checked on use.
Receipt reports unique referenced component/index storage only; it explicitly
excludes surface inverse/solver workspace/backward and does not claim aggregate
capacity. Fixed-order error, original-equation error and physical qualification
flags remain false; narrow controls do not qualify arbitrary configurations.

Independent source audit found no P1/P2. Initial all-near heterogeneous two-body
control /tmp/gsp11-coupled-action-tests-v1.log:2passed31.27s, session78470 terminal.
Audit called out lack of jointly exercised far blocks and untested delayed-map
comment. Added an actual surface far assertion and delayed native backward
mutation test, plus a subset-region response comparison. v2 close geometry
produced no retained far factors:2failed11.28s, session42362 terminal; failures
at the far-count assertion are preserved, with no relaxed error gate.
Used more separated physical geometry for actual far-factor admission while
keeping H compression1e-9 and original residual/action thresholds unchanged.
/tmp/gsp11-coupled-action-tests-v3.log:2passed50.96s, session90868 terminalexit0.
Both bodies use refined8-cell tetrahedra, heterogeneous eps2/3 and reversed
binding order [2,1]. E uses H surface/direct actions; H uses H2 surface/direct
actions. Original independently assembled surface A and weak B, complete
original decoded plus direct E/H, batched forward/adjoint/dot/current gradients,
original A/B residual<2e-10, partial-region projection and delayed map mutation
all pass. Surface far factors are required; regional cross/direct components
in this combined fixture remain all-near. Their separate actual far controls
are existing evidence, not a claim of jointly validated all-component far error.

No repeated full suite, nonlinear re-solve or physical promotion. Next: complete
response error/refinement controls, public action-backed HB preparation and
exact persistence; aggregate capacity and cross H2 remain open. Primary spatial
session14033 remains live on frozenab5fa99, no restart or merge pending terminal.
Goal ACTIVE; GSP-01 OPEN; original scope, physical failures and gates unchanged.

### 2026-09-22 — streamed complete coupled-field order comparison

Previous complete-action stage committedd8101dc. compare_coupled_field_orders
compares every unit-current source column at every bound cell centroid, using
bounded column batches instead of complete response matrices. Exact shared
source-action objects and inverse ensure fixed source/surface response; field
order strictly increases within the original FieldQueryConfig budget. One
surface solve per batch is reused across the two field orders. Complete
surface plus direct volume fields are summed before computing xyz norms.
Regional H absolute tolerances divide the electric absolute tolerance by each
region's abs(eta); relative and absolute defaults remain unchanged. Nonfinite
errors/magnitudes/thresholds reject. Per-batch receipts count every failed
column/query pair and retain maximum difference/threshold ratio. No early exit
omits unchecked columns. Arbitrary cancelling combinations, compression/source
quadrature/mesh errors and physical qualification are explicitly not certified.

CoupledVolumeAction helper extraction (_split/_surface/_decode) shares the
same forward/adjoint implementation with the checker. Independent source audit
found no P1/P2. /tmp/gsp11-coupled-refinement-tests-v1.log:4passed53.34s,
session32958 terminalexit0: streamed E/H decisions versus materialized complete
responses plus original full coupled H/H2 forward/adjoint regression.
Audit suggested multi-cell/unequal-region/tail coverage. Followup uses reversed
heterogeneous regions with counts[1,8],27 source columns and batches4 ending3;
checks exactly7 surface solves, all27 columns and region-dependent tolerances.
/tmp/gsp11-coupled-refinement-tests-v2.log:2passed14.87s, session6210 terminalexit0.
No unchanged full coupled rerun after the fixture-only extension. These tests
verify estimator decisions against explicit response differences; their passing
is not a claim that the underlying low-order physical responses are converged.

Next: error-controlled compressed preparation (including source/original
operator checks), action-backed HB model/preparation and exact persistence.
Primary spatial session14033 still live at frozenab5fa99; no restart or merge.
Goal ACTIVE; GSP-01 OPEN; original physical failures, scope and gates unchanged.

### 2026-09-22 — compressed weak-source original columns and separate quadrature axes

Previous complete-field checker stage committed66138ab. New
check_source_columns streams all unit cell/Cartesian columns without a global
source matrix; component-major batches include partial final cell groups.
Original compression compares H actions to original selected weak integration
at identical orders. Quadrature comparisons require identical region bindings
and exactly one increasing axis. Rows are embedded into owning global J/M
positions and measured with the exact local RWG L2 dual proxy. Inner and outer
axes retain half of the existing1e-4/1e-12 tolerances, with finite threshold
checks. Compression uses its separately supplied tolerance and is explicitly
not an additive total-error certificate or trace-space norm certificate.

prepare_source_action constructs original-column-admitted H actions and refines
inner then outer orders within the existing32/64 default limits. Outer tests
hold inner order fixed, rebuilding the previous outer action at the accepted
inner order if it changed. This avoids interpreting simultaneous inner/outer
changes as an isolated outer estimate. Compression failure or order exhaustion
raises with a receipt; successful receipt remains physically unqualified.

/tmp/gsp11-source-refinement-tests-v1.log:2passed17.53s, session57172 terminal.
Independent audit found P2: permissive LocalRWGRiesz inverse settings could
return zero and suppress both measured error and reference norms. _norms now
independently checks the true Gram equation residual at relative1e-12 for
complex128 /1e-5 for complex64, without the caller's absolute floor. Nonfinite
residuals/norms reject. The metric residual gate is recorded separately from
unchanged source tolerances. Loose rtol1 and atol1e30 metrics combined with1%
source corruption are negative controls. /tmp/gsp11-source-refinement-tests-v2.log:
4passed18.29s, session5002 terminalexit0. Also covers actual region2 refined8-cell
partial batches3, independent dense Gram norm decisions, original-column
corruption rejection, actual default-tolerance adaptive preparation and explicit
budget exhaustion. The core norm changed, justifying this targeted rerun;
no full suite, previous unrelated controls or nonlinear solves were repeated.

Next: assemble admitted source and complete-field refinement into compressed
response preparation, original surface-equation checks, action-backed public HB
and exact persistence. Aggregate capacity and total-error bounds remain open.
Primary spatial session14033 remains live on frozenab5fa99; no restart or merge.
Goal ACTIVE; GSP-01 OPEN; original scope, physical failures and gates unchanged.

Independent re-review closed the loose-metric P2. Exact Gram denotes the full
assembled metric, not an exact-arithmetic inverse or condition-free certificate.

### 2026-09-22 — admitted compressed induced E/H preparation

Previous source-admission stage committeddcd8b6f. New
check_coupled_original_columns streams every unit-current column and every
original global J/M row via the existing selected original action. RHS is
recomputed from original weak source blocks, not compressed B. Complete field
reconstruction decodes actual surface-current columns and adds original direct
selected-cell fields in bounded query batches. Shared checked_dual_norms keeps
the independent true Gram residual gate; original surface tolerance uses
atol+rtol*norm, separately from inner Krylov's max(atol,rtol*norm). Original field
agreement is evaluated at the computed currents, not a new independent solve;
conditioning amplification and physical/quad error remain outside this check.

prepare_compressed_cell_response now connects all regional source admission,
bounded complete E/H field refinement and all original-column checks. It returns
induced electric/magnetic actions only after all gates pass. E/H multi-channel
helpers share exactly the same sources/inverse and perform one surface solve
per column batch across both channels/coarse-fine orders; original equation
checks are also shared. Existing single-channel functions delegate to the same
implementation. Exhausted field/source budgets and failed original checks raise
without returning converged preparation. Background/DC/HB wiring, persistence,
aggregate capacity and total-error certification remain separate open work.
Backend memory/solve exceptions still propagate; a calling evidence runner must
retain their terminal context, since not every exception has a full aggregate
preparation receipt.

Independent original-check audit found no P1/P2. First
/tmp/gsp11-original-coupled-tests-v1.log:6passed44.12s, session41904 terminalexit0,
covering full original E/H equations/fields,1% source/direct corruption and shared
norm regressions. Then the multi-channel/orchestrator integration changed the
helpers: /tmp/gsp11-compressed-prepare-tests-v1.log:6passed88.93s, session73272
terminalexit0. Covers actual default preparation versus dense preparation,
shared-solve counts, initial-budget rejection and changed E/H single-channel
refinement/original-check regressions. No unrelated full-suite rerun.

Actual small reversed heterogeneous two-body preparation uses H2 surface and
H2 direct volume actions with all-near components. Source orders are outer32,
inner16/8 in caller order; complete fields pass at order24 after3 refinements.
Exactly12 surface solves serve both channels (6 columns,batch2,3 field stages
plus original check), instead of duplicating E/H solves. Original maximum dual
residual3.6132316694404935e-15; same-current original E/H differences
2.035087206416736e-15/1.116199047548973e-17. Receipt retained at
/tmp/gsp11-compressed-prepare-tests-v1/test_actual_default_preparatio0/receipt.json.
This is fixed-discretization control, not complete far-pipeline or physical
qualification. Independent orchestrator/multi-channel audit found no P1/P2.
Its requested failure-path followup:
/tmp/gsp11-compressed-rejection-tests-v1.log:1passed30.71s, session23158 terminal.
A source is changed1% after actual source admission; field refinement still
passes, but original-equation admission rejects the whole preparation.

Next: incident-background and action-backed HB integration, exact persistence,
then stage integration regression and broader original-scope physical/scaling
acceptance. Primary spatial session14033 remains live at frozenab5fa99; no
restart or source merge until terminal. Goal ACTIVE; GSP-01 OPEN; all prior
physical failures, original scope and acceptance thresholds remain unchanged.

### 2026-09-22 — analytic background banks and native source scales

Previous full induced-response preparation committed41a091a. New
prepare_compressed_background builds each declared external plane-wave carrier
with the existing source contract, integrates selected local lifted RHS rows,
scatters global J/M positions, solves via the guarded H/H2 inverse and checks
all original surface rows in the independently checked RWG dual proxy. RHS
quadrature uses the fixed surface integration configuration and is not yet
independently refined here. Region-ordered total interior fields use existing
adaptive lifted decoding, including the analytic incident field exactly once.
Finite outputs, combined field magnitude/threshold and accepted finite estimator
records are required. Frozen bank arrays/carrier snapshots and inverse/binding
versions are guarded. Geometry/material/source-shape gradients are excluded.

CompressedBackgroundBank.combine accepts native real or complex scale tensors
with arbitrary leading batch dimensions. Scales multiply each declared source's
full amplitude/phase; they do not replace a zero declared amplitude with a unit
wave. Source coherence and scale gradients are retained using fixed field/current
banks, without new solves. Empty source banks produce correctly shaped zero
backgrounds without invoking the inverse. prepare_compressed_cell_response now
optionally accepts incident_sources and returns a background_bank; aggregate
converged=True is assigned only after background preparation succeeds. The
induced-only default remains available. Public action-backed HB wiring and exact
persistence remain unfinished; total/physical errors are not certified.

/tmp/gsp11-background-tests-v1.log:2passed6.12s, session91360 terminalexit0:
H2 two-source coherent background matches independently prepared dense lifted
surface/E/H responses, complex batched source gradients, mutation/invalid guards
and empty-harmonic no-solve behavior. Independent source audit found no P1/P2.
Finite-field gate and optional whole-preparation branch then validated in
/tmp/gsp11-background-preparation-tests-v1.log:4passed52.08s, session27354 terminal.
Whole source/induced/background results match dense preparation at default
tolerances; shared E/H preparation uses12 solves plus1 declared-source solve.
Actual bank original residual1.3225496071971934e-16 <=1.6008801011574479e-10,
background fields reach order24 in both regions. Receipt:
/tmp/gsp11-background-preparation-tests-v1/test_actual_default_preparatio0/receipt.json.
Audit requested real-scale and leading-batch empty coverage: separate
/tmp/gsp11-background-scales-tests-v1.log:1passed4.93s, session52806 terminal.
Real gradient matches centered finite differences; empty bank with scales[2,3,0]
returns zero E/H[2,3,2,3] and surface[2,3,N]. Scale combination is instrumented
to reject any attempt at another surface solve. No full-suite or nonlinear rerun.

Next: action-backed harmonic response/model/preparation, public HB dispatch and
exact persistence; dense nonlinear Newton Jacobians and DC responses remain
separate fast-backend work. Primary spatial session14033 is still live on frozen
ab5fa99; no restart or merge pending terminal/source validation. Goal ACTIVE;
GSP-01 OPEN; original physical failures, scope and acceptance thresholds unchanged.

### Action-backed harmonic balance residual and nonlinear roots

ActionHarmonicBalance now applies one complete electric current action to
J_n-i*n*omega*P_n per positive harmonic, sharing the existing native constitutive,
DC closure, nonlinear iteration and implicit root machinery. It requires exact
ordered volume identities, actual cell weights, matching harmonic frequencies
and guarded response actions. Dielectric DC maps remain explicit; conductive DC
closures allocate no dummy dense DC response arrays. Raw supplied actions must
be separately admitted by compressed preparation. Dense nonlinear Jacobians and
DC operators remain; public fast HB dispatch and exact persistence are unfinished.
No physical, scaling or production qualification is claimed.

Targeted /tmp/gsp11-action-hb-tests-v1.log:7passed64.23s, session43145 terminal:
actual two-harmonic H/H2 response residuals, state gradients, dielectric nonlinear
roots and implicit source-scale gradients against original dense maps, plus
shared dense HB/current/mixed regressions. Independent audit found no P1/P2 and
requested action-specific conductive DC coverage. Initial added test failed at
AC residual max7.717360751162501e-10 with the fixture inner absolute solve floor
1e-13 (/tmp/gsp11-action-hb-current-tests-v1.log, session38593 terminal).
Tightening only the fixture inner solve atol to1e-20, retaining every comparison
threshold, gives1passed7.82s (/tmp/gsp11-action-hb-current-tests-v2.log,
session40054 terminal). This checks conductive DC rectification, residual/state
gradients and exact ordered closure binding. The initial failure is retained;
small nonlinear RHS require appropriately scaled inner solves.

Primary spatial session14033 remains live at frozen ab5fa99; no restart or merge.
GSP-01 OPEN; existing physical failures and final scope/thresholds unchanged.
Next: compressed harmonic preparation, public dispatch and exact persistence,
then matrix-free nonlinear Jacobians. Full regression remains at integration.

### Physical H/H2 fixed inverse integration

The fixed inverse explicitly admits the existing HierarchicalSystemOperator and
HierarchicalH2SystemOperator exact classes. Their refresh paths retain the base
numerical epoch guards and additionally guard physical provider, integration,
carrier and body decoder bindings. Arbitrary subclasses and mutable dense
operators remain rejected. Public HB dispatch is still closed until preparation,
recovery and exact persistence are integrated.

/tmp/gsp11-implicit-physical-tests-v1.log:2failed7passed12.05s, session40469
terminal. Both new fixtures initially omitted the required iterative MoMConfig
stopping rule and were rejected before solving; all seven existing fixed inverse
controls passed. Corrected fixtures only were rerun:
/tmp/gsp11-implicit-physical-tests-v2.log:2passed6.04s, session38662 terminal.
Actual heterogeneous H/H2 forward solutions and RHS gradients match original
dense controls; delayed decoder integration/carrier changes, source tuple
replacement and unknown subclasses are rejected. This does not separately test
every possible source object's internal mutation. Independent source audit found
no P1/P2. No new physics or public fast-HB qualification is claimed.

Action HB implementation is committed as cb61035. Goal ACTIVE; GSP-01 OPEN;
primary spatial session14033 remains live on frozen ab5fa99, with no restart or
merge. Next: compressed harmonic preparation/recovery and exact persistence.

### Compressed harmonic preparation and physical state recovery

prepare_harmonic_balance now has an explicit compressed_settings branch using
admitted induced E/H actions and per-declared-source background banks. It shares
all existing material sampling, geometry/frequency validation and dielectric,
conductive, mixed and dual DC compilation. The configurable inverse_rule defaults
to block GMRES with row/column scaling, rtol1e-11, atol1e-20 and stagnation48;
this numerical choice is not a conditioning or global-error certificate.
No positive-frequency dense response arrays are created in this branch.

Preparation signatures guard compressed actions/banks and physical operators;
reuse requires the exact existing operator/volume prefix and the same response
backend. Recovery applies the induced surface response to J-i*n*omega*P and adds
the pump-scaled incident background once. VolumeFieldResponse no longer requires
a matrix for decode-only views; dense matrix identity/version checks remain and
both paths guard global/local decoder identities, carriers and integration.
Public Scene/Simulation dispatch, exact action snapshots and matrix-free Newton
Jacobians remain unfinished. DC remains dense. Material-driven compressed
conductive/mixed/dual recovery requires separate validation; shared compilation
and earlier residual controls do not establish those complete paths.

Evidence (GPU1 only, overlapping the existing spatial run; no timing claims):
- /tmp/gsp11-compressed-hb-preparation-v1.log:1failed1passed49.66s,
  session6439 terminal. Actual H2 dielectric nonlinear root/state controls passed,
  but field query exposed a remaining unconditional operator.matrix access.
  Existing dense harmonic preparation and extension passed.
- /tmp/gsp11-compressed-hb-preparation-v2.log:3failed7passed107.98s,
  session44425 terminal. After removing that access, actual compressed root,
  recovered surface/DC charge and exterior E/H match dense controls; query does
  not call the interface inverse. Three multi-body dense tests failed because
  the new guard read the global operator's single-body basis property.
- /tmp/gsp11-compressed-hb-guards-v1.log:1failed1passed97.62s,
  session30703 terminal. Actual two-body compressed H1->H2 prefix extension
  passed: only the new frequency prepared, old DC/H1 objects retained, H2
  background zero, expanded nonlinear solve/recovery succeeds, backend mixing
  rejects. The cached decoder negative control used the pre-fix loaded module
  and failed on the same global basis access. The source fix was applied while
  that already-imported test process completed; no restarted process is claimed.
- Guard fix reads basis/interior only on local handles. Final targeted four-test
  recheck /tmp/gsp11-compressed-hb-decode-fix-v1.log, session64212 terminal:
  1failed3passed74.31s. Dense native gradients, direct response and cached
  carrier/integration/dense-matrix mutation guards pass. Remaining high-order
  oracle mutated a copied cached response, correctly rejected by the new guard.
  It now constructs a fresh VolumeFieldResponse from identical currents and
  recovered coefficients, preserving order64 and comparison thresholds, without
  another surface solve. Single affected node rerun session34066 terminal:
  /tmp/gsp11-compressed-hb-decode-fix-v2.log:1passed25.11s.
Independent source review found no remaining P1/P2 including prefix reuse and
decode mutation coverage. Final targeted GPU failures are closed; the fresh
order64 oracle adaptation was independently reviewed without weakened gates.

GSP-01 OPEN, original physical failures and final thresholds unchanged. Primary
spatial session14033 remains live at frozen ab5fa99; no restart or merge. Goal
ACTIVE. Next: exact compressed representation persistence and public HB dispatch,
plus the unvalidated material-driven DC paths and fast nonlinear Jacobians.

### Exact regional H block snapshot foundation

volume_h_snapshot stores actual volume, weak-source and interior H block arrays,
node layouts, configuration and receipts using primitive CPU tensors/metadata.
Restore accepts an independently rebuilt native provider, checks physical input,
material/order, deterministic geometry trees and scaling digests, and directly
loads stored exact blocks or low-rank factors without kernel sampling,
compression or solves. Index/component layouts and dimensions are checked;
a range-add min/max sweep proves coverage exactly once without an N-by-M mask.
Native forward/adjoint/gradient and version guards remain. SHA256 detects
accidental corruption, not authenticity or renewed compression certification.
Retained checks precede factor loading, but geometry/tree/scale construction,
index temporaries and allocator peak memory are not preflight bounded here.

Evidence on GPU1, overlapping the existing spatial process (no timing claims):
/tmp/gsp11-volume-h-snapshot-v1.log:3passed106.32s, session80775 terminal.
Actual rectangular source/interior all-near maps and a96-cell volume H map with
both far and exact blocks survive weights_only serialization and restore;
forward, adjoint, current gradients and re-encoded SHA are exact. Restore is
instrumented to forbid provider sampling, compression and torch.linalg.solve.
Checksum, physical field mismatch, overlapping/missing coverage and post-restore
index mutation checks are covered. Independent audit found no P1/P2, requested
explicit stored-entry corruption and separately rebuilt providers. Direct[0,0]
mutation with a value-change assertion replaces flatten mutation to avoid a
noncontiguous-copy ambiguity. /tmp/gsp11-volume-h-snapshot-rebind-v1.log:
1passed4.45s, session72433 terminal; fresh providers on existing bindings.
Strengthened to separately constructed volume meshes, surface systems and all
three provider kinds: /tmp/gsp11-volume-h-snapshot-rebind-v2.log:1passed4.53s,
session48640 terminal. Independent review confirms both requested gaps closed.
The expensive far construction was not repeated for those binding controls.

This is a component persistence foundation, not a complete HB snapshot: nested
volume H2, global surface H/H2 inverse state, coupled response reconstruction,
public snapshot schema/Bench validation and public HB dispatch remain open.
No new physical or production qualification. Previous preparation/recovery
integration is cf88232. Goal ACTIVE; GSP-01 OPEN; original failures, full scope
and final acceptance thresholds unchanged. Primary spatial session14033 remains
live on frozen ab5fa99; no restart or merge.

### Exact regional volume H2 snapshot foundation

nested_h2_snapshot stores actual leaf bases/indices, child transfers, couplings,
scales and packed near records in original group order. volume_h2_snapshot binds
this state to independently reconstructed volume geometry, material, field/order
and deterministic tree fingerprints. Restore validates ranks, node/component
maps, combined near/far coverage, configuration and receipts, then reinstates
native epoch guards without reconstructing source H, kernel sampling, QR/SVD,
conversion or solves. Fixed geometry plus nested owned and packed-near retained
bytes are counted after tree reconstruction and before factor loading; transient
uploads/repacking and allocator peaks remain excluded. Checksums establish
integrity, not authenticity or renewed physical/compression certification.

GPU1 evidence, overlapping spatial session14033 (no timing/scaling claims):
/tmp/gsp11-volume-h2-snapshot-v1.log:1failed1passed105.31s, session57971 terminal.
All-near independent restore passed; the actual96-cell far/near case failed its
zero-tolerance forward check at2.7939250479261703e-20 (relative3.6149e-16).
The original failure is retained. Both constructed payloads are saved under
/tmp/gsp11-volume-h2-snapshot-v1/test_actual_volume_h2_exact_in{0,1}/snapshot.pt.
No expensive far construction was repeated for diagnosis.

Saved-factor replay43150 terminal, /tmp/gsp11-h2-snapshot-replay-v1.json:
the SAME restored instance repeated outputs differ by3.030437198788175e-20;
two independently restored objects re-encode to the exact original SHA.
This establishes default CUDA accumulation is not bitwise reproducible even
without a storage roundtrip. Exact-storage claims are separate from execution.
The test fixture opts into native deterministic algorithms and a configured
BLAS workspace, restores prior settings afterwards and retains ZERO comparison
tolerance. Replay46570 terminal, /tmp/gsp11-h2-snapshot-replay-v2.json/log:
eight same-instance repeats and cross-restore forward/adjoint/gradient differences
are all zero. This claim is limited to the tested device/software environment,
not cross-device or library-version reproducibility. Rehashed invalid scaling,
rank, missing far coverage and capacity records reject; E/H provider mismatch
rejects. Separate changed eps_r material binding also rejects in terminal47298,
/tmp/gsp11-h2-snapshot-material-guard.log. Replayed from saved actual arrays only.

Updated deterministic all-near pytest including structural negatives:
/tmp/gsp11-volume-h2-snapshot-v2.log:1passed4.35s, session36288 terminal.
Independent audit found no P1/P2; reviewed exact storage versus deterministic
execution distinction and retained original failed gate. The full far pytest was
not rerun; its saved actual factors were validated by the deterministic replay.
Global surface H/H2/inverse state, coupled reconstruction, public HB schemas and
Bench persistence integration remain open. No new production/physical status.

Previous exact H component commit fdbbcfd; Goal ACTIVE; GSP-01 OPEN. Full scope,
thresholds and retained physical failures unchanged. Primary spatial14033 remains
live at frozen ab5fa99; no restart or merge. Next: global surface H/H2 snapshot
and inverse reconstruction, then complete action-backed HB/public persistence.

### Exact scalar surface H/H2 and fixed inverse state

surface_h_snapshot preserves actual global scalar H near/far blocks or H2
nested actions, local RWG Gram records/diagonal and metric tolerances. Restore
checks independent surface geometry, material, integration, tree/component
layout, exact coverage and metadata. Local geometric Gram construction validates
saved records and diagonal exactly; electromagnetic kernel assembly, H setup,
H2 conversion and solves are not used. New objects restore original native
identity/version guards and fresh iteration caches. encode_inverse/restore_inverse
also preserve the configured StoppingRule, residual refinement count and saved
forward/adjoint diagnostics without executing a solve. Arbitrary/mutable backend
types and non-Krylov policies remain rejected.

H owned-retained checks include prepared geometry, tree, metric, near/far arrays
and index maps. H2 nested-owned admission is separate from an added source-H
budget check for shared geometry/metric/tree/near state. These are not aggregate
peak allocations or fast setup/physical qualification claims. The near snapshot
codec is shared with regional H2; existing ordering and coverage are unchanged.
Physical runtime subclasses/carriers, tensor/periodic providers, complete coupled
HB snapshots and public dispatch still require integration.

GPU1 evidence (overlapping spatial14033, no performance claims):
/tmp/gsp11-surface-h-snapshot-v1.log:1passed89.89s, session3497 terminal. Actual
heterogeneous two-body120-edge surface H/H2 with far+near blocks restores on a
new system with exact re-encoded SHA. Under native deterministic CUDA execution,
forward/adjoint, metric actions and preconditioner scales match exactly; fixed
inverse solutions and native implicit gradients match original operators at
unchanged2e-8/1e-9 gates. Restore forbids kernel sampling, operator construction,
QR/SVD and solve. Corruption, changed material/Gram and delayed metric mutation
reject. Saved H/H2 payloads:
/tmp/gsp11-surface-h-snapshot-v1/test_actual_surface_h_h2_resto0/{hmatrix,h2}.pt.

Final stricter diagonal/shared-near budget gates were checked from those saved
actual factors, without repeating physical setup:
/tmp/gsp11-surface-h-replay-v1.log/json, session93774 terminal. Both native inverse
forward and adjoint converge; rehashed metric-diagonal and H/H2 owned/shared
budget modifications reject. /tmp/gsp11-surface-h-snapshot-f32-v1.log:
1passed5.03s, session83889 terminal, complex64 all-near H/H2 exact action/gradient
roundtrip plus budget/diagonal negatives. This is not complex64 far qualification.
/tmp/gsp11-surface-h-inverse-replay-v1.log/json, session8797 terminal: stored
Krylov configuration and actual forward/adjoint diagnostics roundtrip exactly;
restore is instrumented against inverse solves, and rehashed direct policy
rejects. Saved-factor replay uses the existing120-edge snapshots only.
Shared near extraction verified on saved actual96-cell volume H2 factors:
/tmp/gsp11-h2-snapshot-shared-near-v1.log, session29516 terminal; exact deterministic
forward/adjoint/gradient and structural negatives pass. No far reconstruction.

Previous component commit49fe0ac. Goal ACTIVE; GSP-01 OPEN; original physical
failures, final scope and thresholds unchanged. Primary spatial14033 still live
at frozen ab5fa99; no restart or merge. Next: physical source/carrier bindings and
complete coupled/background/HB reconstruction, public schema and Bench alignment.

Final incremental independent review found no P1/P2 in strict metric-diagonal,
separate H2 shared-near capacity or inverse codecs. Saved historical diagnostics
are observations, not newly recomputed convergence evidence. Exact action and
strict regenerated-diagonal checks are validated in the deterministic test
environment, not claimed across devices or default nondeterministic execution.


### Physical H/H2 snapshot carrier bindings and level-2 spatial observations

Physical scalar H/H2 envelopes preserve exact numerical factors, complete public
linear configuration and coherent carrier payloads. Restore rebinds the native
physical decoder and fixed inverse to an independently constructed system and
supplied sources without electromagnetic assembly, compression or solves.
Encode/restore independently recompute operator identity/payload and validate
present RHS/carrier identities. Independent audit P2 (copied RHS metadata was
not independently validated) is fixed; live metadata and nested+outer rehashed
metadata corruption negatives cover both backends. Final independent source
review reports no P1/P2.

/tmp/gsp11-physical-surface-snapshot-v1.log: 1 failed, 2 passed, 16.51 s.
H2 rebuilt RHS differed by 1.183764530776631e-22 under default CUDA execution;
zero-tolerance failure retained. Deterministic-only H2 rerun v2: 1 passed,
10.65 s, unchanged zero tolerances. Final audited identity fix:
/tmp/gsp11-physical-surface-snapshot-v3.log, session56684 terminal exit0,
3 passed, 8.52 s. Covers weights-only roundtrip, exact deterministic RHS and
field decoding, configured inverse/gradient comparisons, empty sources,
source amplitude/order mismatch, mutation guards and no-solve restoration.
Physical wrapper fixtures are all-near; preceding bare far-factor validation
is separate evidence, not combined far-wrapper qualification. Default CUDA,
cross-device or cross-library bitwise execution is not claimed.

Primary frozen ab5fa99 spatial session14033 is now terminal exit0:
/tmp/gsp11-dual-spatial-level2-v1/terminal.json, solved_unqualified,
256 cells / 64 surface faces, source outer budget64, 10754.65807610564 s.
Result SHA256 a3b4d3f278e6c65ba27ef3a5c1d0667b6af8921c4639455d882b9a38b73c58bf.
Audited saved-state comparison /tmp/gsp11-dual-spatial-compare-level2.py,
session80117 terminal exit0, 2.8506133146584034 s; receipt at
/tmp/gsp11-dual-spatial-compare-level2-v1/receipt.json. No repeated solve.
32-to-256-cell volume-weighted electric RMS relative differences DC/H1/H2:
0.18040227062621506 / 0.08279489781631942 / 0.21541550308770022.
Interior H2 E probe differences 4.05-10.45%, H differences 20.21-26.71%;
exterior H2 E/H differences approximately 0.83-1.14%. Spatial convergence
remains UNVERIFIED; completed_unqualified is not production qualification.
Both saved-result hashes and full numerical/geometry/source bindings checked.

Previous component commit ff3d336. Goal ACTIVE, GSP-01 OPEN; all original
physical failures, final scope and thresholds unchanged. No full regression
repeated. Next: shared coupled E/H action and background-bank snapshots,
complete HB reconstruction, public schema/Bench alignment, integration checks.


### Shared coupled E/H and declared-source background snapshots

Built on physical snapshot commit86ad8b2. Coupled response payload stores one
inverse and one ordered set of source actions, plus E/H interior and regional
H/H2 direct factors. Restore binds independent supplied volumes/provider and
reinstates shared identities, region/cell ordering, native guards and default
unqualified receipts; no EM sampling, compression or solve. Background codec
stores coherent carrier identities and exact RHS/surface/E/H arrays, bound to
supplied sources, ordered meshes and fixed inverse factors/policy. Mutable
solver observations are excluded from fixed inverse identity. Stored historical
background convergence records are not re-evaluated or promoted to qualification.

/tmp/gsp11-coupled-snapshot-v1.log, session38068 terminal exit0: 1 passed,
6.63 s. Actual two-body reversed regions, bare surface H2 inverse, mixed H/H2
direct representations, independently rebuilt geometry, weights-only SHA
roundtrip, shared object identities, original/restored forward/adjoint and
native gradients. Rehashed order/receipt/channel changes, reordered volumes,
and delayed shared source mutation reject. Restoration forbids provider sampling
and inverse solve. Scope: equal-size all-near integration; earlier far and
physical-carrier component tests are not combined far/physical integration.
Independent source audit reports no P1/P2 for coupled codec.

/tmp/gsp11-background-snapshot-v1.log, session64207 terminal exit0: 2 passed,
5.16 s. Pumped and empty source banks, independent geometry/inverse restore,
exact weights-only roundtrip, exact array combination and source-scale gradients,
no solve during restoration, changed carrier/order/qualification rejection,
mutable historical diagnostics excluded and delayed array mutation guards.
No full regression repeated. No new physical or aggregate capacity qualification.
Goal ACTIVE; GSP-01 OPEN; retained physical failures and final thresholds unchanged.
Next: combined prepared-response/HB state reconstruction and public schema/Bench
integration, including broader integration validation at the stage merge point.

Final independent background source audit reports no P1/P2. New background
integration covers empty/single sources and a bare inverse; multi-source ordered
physical-inverse integration is still open. Fixed inverse identity currently
serializes actual factors to CPU; storage work/peak cost is not benchmarked or
claimed aggregate-bounded. This remains a persistence operation, not a solver
CPU fallback.


### Admitted compressed harmonic preparation and saved-current verification

Built on76bc6f0. compressed_harmonic_snapshot integrates shared E/H factors and
background-bank codecs into the exact preparation dictionary consumed by HB.
Shared ordered binding/inverse identity, derived points/weights/backgrounds and
historical admission receipt are checked. Restore performs geometry/layout and
factor reconstruction, not EM sampling, quadrature admission or inverse solves.
Stored convergence remains historical; qualification flags stay false.

validate_surface_state accepts induced current coefficients separately from
incident background. It forms B*J, checks A*x-B*J against the unchanged configured
max(rtol*norm(B*J),atol) Euclidean inverse tolerance, then decodes both D*J+F*x
without any inverse. This detached validation is not an implicit derivative,
original-kernel residual, physical certificate or complete HB-state restore.
Future full-state format must preserve induced coefficients so background
subtraction cancellation does not silently weaken this gate.

/tmp/gsp11-compressed-harmonic-snapshot-v1.log, session95088 terminal exit0:
1 passed,30.35 s. Actual admitted heterogeneous two-body physical H2 preparation,
reversed regions, two coherent sources, independently rebuilt geometry; stored
weights-only preparation SHA and bindings match. No-sampling/no-solve restored
current residual validation and E/H decoding agree with original inverse actions;
corrupted surface currents, source ordering, qualification flags and derived
background changes reject. Deterministic CUDA enabled and restored in fixture.
All-near fixture: no far-factor/full-HB/complexity qualification claim.
Saved payload under /tmp/gsp11-compressed-harmonic-snapshot-v1/test*/preparation.pt.

Independent source audit no P1/P2. Nonblocking generator consumption suggestion
fixed by owning a source tuple at restore entry. Final fix checked by saved-payload
replay (no repeated admission): /tmp/gsp11-compressed-harmonic-replay-v1.log,
session23141 terminal exit0; generator sources, exact SHA, forbidden solve/sampling.
Full initial fixture was not rerun after this iterator-only change.

Goal ACTIVE; GSP-01 OPEN. Original physical failures, final scope and thresholds
unchanged. Next integration: complete static+dynamic HB preparation/state schema,
constitutive/DC/AC residual and saved-current validation, public dispatch and Bench
schema binding. Dense nonlinear Jacobian remains a separate unfinished fast path.


### In-progress full compressed HB state integration (uncommitted)

Current base f1f1512. WIP harmonic_state.py retains separately induced surface
coefficients under mutation guards. WIP harmonic_snapshot.py schema15 uses
compressed preparation codecs and existing material/DC/constitutive checks,
validates saved induced A*x=B*J without inverse solves, then total surface and
cell convergence. Public compressed dispatch and Bench schema remain unfinished.
Independent audit found raw harmonic count could be silently truncated; moved
prepared schema/list/count checks before operator loop and added exact operator
record fields. Audit P2 closed at source level; count negatives are added.

/tmp/gsp11-compressed-hb-state-v1.log, session60602 terminal exit1:
1failed1passed41.06s. Existing dense public roundtrip passed; new test fixture
incorrectly passed branch_id to solve, corrected to actual solve_ramp.
/tmp/gsp11-compressed-hb-state-v2.log, session71926 terminal exit1:
1failed45.18s. Actual two-harmonic compressed nonlinear solve/recovery/save
completed; restore rejected H source receipt retained byte mismatch. Saved
full state /tmp/gsp11-compressed-hb-state-v2/test*/hb.pt must be reused; do not
repeat nonlinear preparation/solve to diagnose restoration.
/tmp/gsp11-compressed-hb-replay.py and -replay-v1.log, session21284 terminal exit1:
saved source retained_bytes2800, restored measured2896. Values/digests match;
original scene surface and volume geometry share vertices, reconstructed
volume has independent storage (96bytes =4*3*float64). Existing exact receipt
measurement wrongly assumes identical physical-geometry storage aliasing.
Required next fix: distinguish historical geometry/storage layout from actual
restored retained capacity, keep strict actual budget enforcement and factor
integrity. Do not relax capacity or numerical thresholds; audit requested.
Full compressed current/mixed/dual roundtrip evidence also remains open.
Goal ACTIVE; no external blocker; no stage commit yet for this WIP.


### Full compressed dielectric HB state restoration and actual storage admission

Schema15 compressed payloads reuse existing material, DC, constitutive, source
convention, solver-threshold and full-cell residual validation. Independently
saved induced AC coefficients avoid subtraction of pump background, retain
mutation guards, and validate compressed A*x=B*J without inverse solves. Total
surface must equal background+induced. Existing dense formats remain unchanged.
Raw prepared harmonic count and operator records are checked before reconstructing
parts; independent audit P2 (silent extra harmonic truncation) is closed.

The actual HB snapshot exposed a legitimate geometry alias difference: original
surface/volume vertex storage shared96bytes, restored independent volumes do not.
H codec now preserves historical retained_bytes as a bounded integer observation,
while actual restored provider/tree/scales, factor preflight and final actual
storage remain subject to the same retained budget. restored_retained_bytes and
coupled restored_component_referenced_bytes report current layout separately.
Coupled receipt permits only the historical capacity field to differ; all other
fields remain strict. No factor/numerical/qualification gates were weakened.
Original storage alias manifests were not saved, so historical uniqueness cannot
be reconstructed from geometry values. Volume H2 owns only its volume geometry;
this surface/volume alias fix does not alter its nested budget allocation.

Original v1 fixture-interface failure and v2 storage-layout failure remain in
previous entries. No nonlinear preparation/solve repeated after v2 save.
/tmp/gsp11-compressed-hb-replay-v2.log, session75717 terminal exit0: full restore
succeeds; original/restored source2800/2896bytes, interior2920/3016bytes.
/tmp/gsp11-compressed-hb-acceptance.py and -acceptance-v1.log,
session3567 terminal exit0: saved actual dielectric two-harmonic state restores,
re-encodes exact SHA, and a second roundtrip yields bitwise matching interior E/H
queries for both harmonics. Inverse solve, torch.linalg.solve and component EM
sampling are forbidden. Missing/extra harmonic, induced/total surface corruption,
induced mutation and invalid historical byte observations reject. Configuring a
2800byte H budget rejects actual2896byte restored layout despite historical fit.
Saved state: /tmp/gsp11-compressed-hb-state-v2/test_complete_compressed_hb_st0/hb.pt.
Equivalent actual-budget negative added to the main integration test for future
runs; final validation uses saved-state replay, not another nonlinear solve.
Existing dense public snapshot test already passed in v1; no full suite repeated.

Final independent source audit: no P1/P2, actual preflight/final capacity gates
retained. All-near dielectric H2 integrated evidence only; full compressed
current/mixed/dual roundtrip, public configuration/runtime dispatch, Bench schema,
far/unequal-region integration and final broad regression still open. No physical
or fast nonlinear qualification (dense nonlinear Jacobians remain). Goal ACTIVE,
GSP-01 OPEN; original failures and final thresholds unchanged. Base f1f1512.


### Public scalar H/H2 HB dispatch and schema15 Bench integration

Base da3a6ba. Scene/Simulation/Result now dispatches scalar free-space hmatrix/h2
HB through admitted compressed preparation. Initial and cutoff paths share
operator/settings helpers. Shared H block settings govern regional source,
interior and direct factors; pivoting is surface-only. H2 additionally controls
regional nested conversion. Source budgets and selected-backend original-check
tolerance bind public preparation. Physical inverses enforce user Euclidean
stopping plus backend checked RWG dual tolerance, including backward and stored
surface validation. Dense DC and dense nonlinear Jacobians remain.

Public schema15 retains outer numerical backend/H/H2 settings; restore verifies
physical operator, inverse rule/refinement count, regional configs, source budgets,
and historical original/background/field admission tolerances against that policy.
Bench freezes the same numerical identity, inventories schema15 arrays, and derives
cutoff weights from ordered stored volume geometry. Exact hashes/ownership gates
remain. Cutoff comparisons handle real compressed factors and exclude mutable
Krylov diagnostics. Independent dual static comparison uses native dual encoding.
Audit findings on policy binding and dual comparison fixed; final audit no P1/P2.

Evidence (GPU1): public-v1 log /tmp/gsp11-compressed-hb-public-v1.log,
session20906 terminal:3failed4.24s, legacy direct-only runtime guard corrected.
public-v2 /tmp/gsp11-compressed-hb-public-v2.log, session19172 terminal:
3failed216.19s. Actual current/H, mixed/H2 and dual/H2 each prepared, solved,
saved and restored; fixture erroneously compared configs containing independently
rebuilt volume object identities. Replaced with numerical comparison. All v2
result.pt files retained under /tmp/gsp11-compressed-hb-public-v2/test*_r{0,1,2}/.
No three-case nonlinear rerun: /tmp/gsp11-public-hb-replay-v1.log, session55105
terminal exit0, all three exact SHA, new policy/dual gates and queries without solve.
/tmp/gsp11-physical-inverse-dual-v1.log, session27871 terminal exit0:
2passed5.38s actual H/H2 dual-gate rejection, solve and gradient controls.

/tmp/gsp11-compressed-bench-replay-v1.log, session61136 terminal exit0:
three saved-result archives at /tmp/gsp11-compressed-bench-v1/{current,mixed,dual},
exact restored fields, full inventory/native checks, four policy-change negatives
per case, H2 original-tolerance selection. No solve. Manifest SHA:
current 9bd64de762f2ba628d4249a169b18c7efc4075b7753c48106f8ae990169daaf6
mixed 983d2d2d3f416d7957626b9d7d02615cfef41739e5cf07985052b86319926eb1
dual 92ac604ed27af366397da2ce8be3b28122b467174dbdece747cee37da7fa7e45

Cutoff v1 /tmp/gsp11-compressed-cutoff-v1.log, session44067 terminal exit1:
new H3 solve completed but old comparator required .matrix. No extension snapshot
survived. Corrected comparator and saved-before-comparison runner:
/tmp/gsp11-compressed-cutoff-v2.log, session40573 terminal exit0, existing saved
current H backend HARMONIC2→3 extension; original static/AC prefix reused. Bench
cutoff checks pass; harmonic_budget_exhausted retained, no cutoff qualification.
Snapshot /tmp/gsp11-compressed-cutoff-completed-v2.pt and receipt in
/tmp/gsp11-compressed-cutoff-v2/. Independent saved current-prefix comparison
/tmp/gsp11-compressed-cutoff-replay-v1.log, session74299 terminal exit0; evolving
solver observations excluded, no solves. Dual-only independent static helper
/tmp/gsp11-dual-prefix-replay-v1.log, session94401 terminal exit0; actual saved
public dual data, type-switch negative. This is not a full dual H2→H3 cutoff run.

No full regression repeated. Public fixtures remain small/all-near; combined
far/unequal-region, full dual cutoff, stage-merge broad tests and final independent
physical/complexity qualification remain open. Goal ACTIVE; GSP-01 OPEN; all prior
physical FAILs and final scope/thresholds unchanged. Next: integration merge/broader
regression and remaining fast nonlinear Jacobian/resource/physics work.

## Cutoff runtime consistency and integration checkpoint (2026-09-22)

Integration branch fast-forwarded to 2453d50; Bench remains 22aee90. Sequential
GPU 1 stage regression is live in session 98447, with source hashes and terminal
receipts under /tmp/gsp11-merge-regression-v1. Both source trees are frozen.
This is MoM plus full Bench coverage, not final four-repository acceptance.

Cutoff extension retains the runtime's initial model/preparation and obtains
final metadata from the extended result decoder. Public PreparedSimulation is
single-use, including failed runs; the audit's alleged public repeat-run P2 was
withdrawn. This change only preserves internal model/config consistency.
Targeted v1 failed because the test violated public single-use; v2 had one pass
(raise policy/internal retry) and one test error (nonexistent public metadata
property). Corrected record-policy node passed in 26.40 s, session 23717,
/tmp/gsp11-cutoff-repeat-v3.log. Both policies now have passing evidence without
repeating the passed raise case. Independent read-only audit found no confirmed
P1/P2. No new physical qualification; GSP-01 remains OPEN.

## Fixed packed-real harmonic tangent (2026-09-22)

HarmonicJacobianAction owns a detached linearization point and applies native
JVP/VJP products without constructing the complete HB Jacobian. Product-vector
first/second gradients use the opposite fixed action. Model, closure and action
mutation guards remain active. No linearization-point/material/geometry gradients
are claimed. Dense DC closure tangents and the existing Newton/root dense
Jacobian/SVD remain; this is not yet a full fast nonlinear solver.

Dense algebra oracle: /tmp/gsp11-hb-tangent-v1.log, session 9880, 1 passed in
3.76 s, including batched transpose, first/second vector derivatives, empty
batch and delayed mutation. Actual saved schema-15 current H, mixed H2 and
dual H2 states reused without nonlinear root solves: session 31944 exit 0,
/tmp/gsp11-hb-tangent-replay-v1.log. Centered difference relative errors were
4.8618e-12, 1.4139e-12 and 6.0037e-12; transpose defects 8.8818e-16, 0, 0.
The durable shared test helper also passed on these three saved states,
session 64504 exit 0, /tmp/gsp11-hb-tangent-replay-v2.log. Fresh public prepare
fixtures are committed for subsequent integration coverage, not rerun here.
These are small all-near correctness controls with TPA and conductive closures,
not far-field scaling, isolated timing or production qualification. Independent
read-only audit found no confirmed P1/P2. GSP-01 and all physical FAILs stay open.

## Fixed tangent inverse component (2026-09-22)

FixedHarmonicTangentInverse reuses the native single-RHS Arnoldi action hook in
packed-real space, without changing the public complex surface protocol.
Restart/iteration limits remain configurable. Each bounded correction pass
uses the same GMRES policy; acceptance uses the original RHS threshold.
There is no preconditioner, recycling, dense fallback or retained iteration
tape. Implicit RHS gradients use independent transpose solves and assume a
fixed nonsingular tangent; a residual receipt does not certify nonsingularity,
conditioning, root stability, full fast nonlinear behavior or physics.

/tmp/gsp11-tangent-inverse-v1.log, session 20365 terminal exit 0: one dense
oracle test passed in 10.92 s. Covers forward/transpose batched solutions,
first/second RHS derivatives, no full Jacobian/direct solve/SVD, zero/empty RHS,
failed iteration budget and delayed mutation. Independent audit found an
overflowed tolerance threshold acceptance path; outer acceptance now requires
a finite threshold. /tmp/gsp11-tangent-inverse-overflow-v1.log, session 4044
terminal exit 0: rejection test passed in 3.14 s. Audit rereview found no
remaining confirmed P1/P2.

Actual saved schema-15 compressed current/mixed/dual states are reused by
/tmp/gsp11-tangent-inverse-replay.py; no nonlinear roots or operator assembly
are repeated. The durable public preparation test shares tangent and inverse
checks to avoid a second preparation. Those fresh preparation nodes remain
deferred to integration coverage; saved-state checks use nonlinear root fields.

Saved-state replay session 2887 terminal exit 0, all three passed. Original
RHS forward/transpose residuals and thresholds from the native receipts:
current: [('last_forward', 2.1323152083914747e-11, 2.881577593086782e-09), ('last_adjoint', 4.8301566921197983e-11, 2.881577593086782e-09)]
mixed: [('last_forward', 5.5737551159131023e-11, 4.314348611079179e-09), ('last_adjoint', 5.719806690174487e-11, 4.314348611079179e-09)]
dual: [('last_forward', 2.071485495360258e-10, 6.9180425509185855e-09), ('last_adjoint', 1.9391977851901083e-10, 6.9180425509185855e-09)]
The test also verifies transpose RHS gradients against an independent adjoint
solve. Evidence is numerical correctness on small/all-near fixed tangents, not
isolated timing or scaling. Stage regression 98447 is still live on frozen
integration 2453d50 and Bench 22aee90; no duplicate run started.

## Internal Newton-Krylov directions (2026-09-22)

The native HB solve accepts internal method newton_krylov with an explicit
unpreconditioned, unrecycled, no-fallback GMRES tangent rule. Newton directions
use fixed matrix-free tangents and original-RHS residual gates; failed tangent
solves propagate an HB convergence failure with the inner receipt. Existing
line search and both final dense implicit Jacobians/SVD remain unchanged.
Receipts explicitly declare fully_matrix_free=False. Public configuration and
full fast nonlinear qualification are not widened by this stage.

/tmp/gsp11-newton-krylov-v1.log, session86853 terminalexit0: 2passed4.27s.
Covers nonlinear root and material-gradient agreement with dense Newton, only
two global Jacobian calls (final implicit checks), rejected tangent budget,
invalid policies and unchanged singular final root rejection. An independent
read-only audit found no new P1/P2 but identified inherited outer threshold
overflow in all old/new methods. A finite outer threshold is now required
before root acceptance; /tmp/gsp11-newton-krylov-overflow-v1.log, session77615
terminalexit0: all three methods rejected overflow, 3passed3.08s.

Actual saved schema-15 current H/mixed H2/dual H2 operators and nonlinear roots
are reused by /tmp/gsp11-newton-krylov-replay.py. The new algorithm starts at
95% of the saved field and must converge back to that reference under unchanged
configured residual tolerances. This intentionally tests a new iteration path;
no operator preparation or original dense reference solve is repeated. Future
public compressed regression shares its existing preparation and reference
root through check_compressed_newton. Those combined pytest nodes have not
been rerun during the frozen integration regression.

Replay86726 terminalexit0, all three physical preparations passed:
current: iterations=2, residual=2.5835718886183235e-17, threshold=8.643251648302818e-10
mixed: iterations=2, residual=4.5876776950565023e-17, threshold=1.0935436106180252e-09
dual: iterations=2, residual=7.379879087085806e-17, threshold=1.6278896261413723e-09
Solutions and full receipts are saved as /tmp/gsp11-newton-krylov-{kind}-v1.pt.
Independent final read-only audit found no remaining confirmed P1/P2. The
shared helper explicitly uses the configured final pump scale. These small
all-near controls are not physical convergence or complexity qualification.
Integration98447 remains live on frozen2453d50/Bench22aee90; new commits stay
on development pending the merge point. GSP-01 OPEN, Goal ACTIVE.

## Streamed implicit root checks (2026-09-22)

Optional internal implicit_solver=streamed_krylov evaluates all packed-real
tangent columns in bounded batches. It accumulates row/column off-diagonal
sums and diagonal magnitudes, obtains single-sided margin/sqrt(N) or combined
sqrt(row_margin*column_margin) numerical lower bounds and sqrt(norm1*norminf)
upper bounds, and compares against the existing 64*eps*N singularity scale.
No positive dominance margin means inconclusive, not singular. The dense
implicit option remains available; this sufficient gate does not replace
general nonsingularity assessment for all allowed materials/branches.

Both final implicit corrections use their own frozen tangent and residual-
checked GMRES, preserving first-order implicit parameter gradients. This avoids
a complete global HB Jacobian but still requires N tangent products per check,
and conductive DC can still use dense Jacobians. fully_matrix_free remains
false and fast_work_qualified false. Float norm bounds are not outward-rounded
interval certification; compressed action/inner-solve error bounds remain open.

64128 terminalexit0:2passed5.22s,/tmp/gsp11-streamed-root-v1.log, root/material
gradient versus dense oracle with global Jacobian/direct solve/SVD forbidden,
complete-column bound comparison and singular-root rejection. 83055 terminal
exit0:1passed3.22s,/tmp/gsp11-streamed-inconclusive-v1.log, actual invertible
but nondominant tangent correctly returns inconclusive. Independent read-only
math/AD audit found no confirmed P1/P2 and explicitly retained the above
numerical rather than strict interval scope.

Existing public compressed preparation/reference tests now choose the streamed
implicit option for their additional Newton-Krylov path, sharing the preparation
and saved reference rather than adding another root check. Fresh combined
pytest nodes are deferred to integration. Actual saved current H/mixed H2/dual
H2 path validation uses /tmp/gsp11-streamed-root-replay.py.

Regression tool handle98447 disappeared, but unsandboxed process inspection
confirmed original runnerPID3236505 and pytestPID3236512 live, with progress
past76%. No restart; continue monitoring these original processes and
/tmp/gsp11-merge-regression-v1/receipt.json. Integration2453d50 and Bench22aee90
remain frozen. A running receipt has no source_after; absence is not a source
mutation failure.

Streamed physical replay14651 terminalexit0, all three cases passed:
current: iterations=2, residual=2.0632512802762968e-14, threshold=8.643251648302818e-10, bounds=[(0.9552168776703849, 1.0449770893395414), (0.9552168776703849, 1.0449770893395414)]
mixed: iterations=2, residual=2.0276950915983625e-14, threshold=1.0935436106180252e-09, bounds=[(0.9549180024848838, 1.045277405716303), (0.9549180024848838, 1.045277405716303)]
dual: iterations=2, residual=4.978478580974911e-14, threshold=1.6278896261413723e-09, bounds=[(0.9340896910245464, 1.0659882549799762), (0.9340896910245464, 1.0659882549799762)]
Full native receipts and solved fields are retained at
/tmp/gsp11-streamed-root-{current,mixed,dual}-v1.pt. No physical or fast-work
qualification is inferred. Integration regression original processes remain
live; no repeated run or source-tree merge.

## Public Newton-Krylov configuration and exact policy binding (2026-09-22)

Public HarmonicBalanceConfig accepts newton_krylov with materialized native
StoppingRule, implicit_solver=dense or streamed_krylov and tangent_column_batch.
Active policies pass through solve_settings for initial continuation and cutoff.
Canonical numerical_payload omits inactive tangent fields, preserving existing
method payloads. Exact restore requires canonical settings and checks solver
method, tangent policy, implicit path and streamed dimensions/admission/finite
numerical bound consistency. Historical bounds are not recomputed or proven
again during decode-only restoration. No physical or fast-work qualification.

Bench worktree <scratch>/witwin-bench-hb-krylov, branch gsp11-hb-krylov, isolates
changes from frozen regression Bench22aee90. Archive and generated-volume
bindings use the same native canonical policy. Candidate construction requires
all active settings; inactive options remain absent for old methods.

Public tests51694 terminalexit0:2passed254.29s,
/tmp/gsp11-public-krylov-v1.log; actual dense/dense and current H2/streamed
Scene->Simulation->Result paths, exact decode-only restore/fields/SHA and
method/tangent/canonical-policy negatives. Saved originals:
/tmp/gsp11-public-krylov-v1/test_public_krylov_policy_rest{0,1}/result.pt.

Bench88958 terminalexit1:3passed1failed150.26s,
/tmp/gsp11-bench-krylov-policy-v1.log. Existing geometry/configuration/actual
nonlinear solve+restore tests passed. New test incorrectly compared JSON-list
pump scales with native tuple; corrected using the existing portable encoder.
Only failed new node rerun47413 terminalexit0:1passed4.23s,
/tmp/gsp11-bench-krylov-policy-v2.log. Preserve original failure evidence.

Saved new public results archived once, without surface/nonlinear/direct solves:
50661 terminalexit0,/tmp/gsp11-krylov-bench-replay-v1.log, root
/tmp/gsp11-krylov-bench-v1. Dense and H2 each passed exact state and three
changed-policy rejections. Manifest SHA256:
dense a3f96836b1cd33048d9b2ba1940e27662e9adc0b878a39a4839a78ee904872cb
h2 89bd695aaf07ce4e347ff6f7e5582e9505ceca5c34301484d4cb3fc07d033a72
Old saved current/mixed/dual payloads retain exact SHA under canonical policy
encoding:71692 terminalexit0,/tmp/gsp11-krylov-old-policy-replay-v1.log.
No repeated roots or archives for these old cases.

Independent source audit found no confirmed P1/P2 in Maxwell or Bench changes.
Public backend/closure/cutoff cross-product, fast complexity, inner error bounds
and production physics remain open. Original sequential regression runner
3236505/pytest3236512 remains live, logpast96%; no restart or source-tree merge.
GSP-01 OPEN; Goal ACTIVE; full scope and thresholds unchanged.

## Krylov cutoff history and tiny-RHS correction (2026-09-22)

Native solver-policy validation is shared by exact restore and each Bench
cutoff-history step, using that step's harmonic count and actual cell count.
Historical tangent bounds remain observations, not recomputed certificates.
Cutoff comparison rejects nonfinite RMS/threshold observations before setting
pass flags. Saved actual old H2->H3 current fields tested finite-input overflow
(1.79e308 relative/absolute tolerances):20290 terminalexit0,
/tmp/gsp11-cutoff-finite-replay-v1.log, no solves. Durable tests capture the
already-computed comparison pair; no additional preparation is required.

New H2-backend streamed Krylov HARMONIC2->3 extension initially failed:
66423 terminalexit1,/tmp/gsp11-krylov-cutoff-v1.log. Failure was at the 300MHz
fundamental surface inverse inside the final implicit correction JVP, not
a newly identified H3 Green-kernel error. Diagnostic12908 terminalexit1
retained original failure and cached H3 preparation in
/tmp/gsp11-krylov-cutoff-diagnostics-v1/{failure.json,failed-rhs.pt,added-harmonic.pt}.
RHS norm5.0464e-18: first Euclidean residual5.7374e-21 met atol1e-20, but
dual2.7967e-17 exceeded9.5881e-25. Both correction solves stopped at zero
iterations because their RHS was below atol. A normalized probe satisfied
unchanged Euclidean6.0534e-33 and dual5.1704e-31 gates.

Fix: normalize each correction RHS by its own max absolute component, solve
with the same policy, then rescale the correction. Initial solve, original
Euclidean/checked-dual gates and per-pass iteration/refinement budgets stay
unchanged. Attempt logs identify normalized units/scales; fixed snapshot
receipt identity and policy keys are unchanged. Captured failed-RHS replay
88405 passed (/tmp/gsp11-small-rhs-replay-v1.log); its displayed last_forward
receipt is from the normalized reference probe, while the original-RHS dual
acceptance is separately asserted. Physical H/H2 two-case tests with heterogeneous
1e-18/1e-28 RHS and gradients:5144 terminal2passed7.06s,
/tmp/gsp11-normalized-physical-inverse-v1.log. Independent source audit found
no confirmedP1/P2 in correction math or cutoff policy hardening.

Cached H3 preparation reused, no re-admission or initial root repeat:
26045 terminalexit0,/tmp/gsp11-krylov-cutoff-v2.log. H3 algebraic residual
2.6710e-15, prefix reused, Bench policy passed and three mutated historical
policies rejected. harmonic_budget_exhausted remains: no truncation qualification.
Outputs /tmp/gsp11-krylov-cutoff-completed-v2.pt and
/tmp/gsp11-krylov-cutoff-v2/{harmonic.pt,receipt.json}. Decode-only replay51282
terminalexit0,/tmp/gsp11-krylov-cutoff-restore-v1.log: exact SHA, independent
restored H2 prefix and exterior H3 fields; no solves. Fresh combined cutoff
pytest nodes remain deferred to the integration point.

Frozen integration2453d50 MoM regression completed:1037passed3skipped1warning,
3058.44s; three skips require two visible GPUs. JUnit1040tests,0errors,0failures
at /tmp/gsp11-merge-regression-v1/maxwell-mom.xml. This does not cover later
development commits or certify physics. Original runner3236505 now executes
frozen Bench22aee90 full suite; failures/errors are visible but terminal
diagnostics and final sourcehash check are pending. Do not restart or merge
into frozen trees. GoalACTIVE;GSP-01OPEN; all physicalFAILs and final gates retained.

### Complete streamed inverse-residual tangent bounds — 2026-09-22

An internal alternative to diagonal dominance streams every approximate inverse
column B and evaluates R = I - J B. Complete one/infinity norm accumulation gives
upper bounds for J, B and R. When the numerical residual bound is below one,
sigma_min(J) is bounded below by (1 - norm(R)) / norm(B), with an explicit
floating-point margin and the unchanged 64 eps N sigma_max singularity cutoff.
This argument does not assume that the GMRES inverse is nonsingular; its
returned columns are candidate vectors, and every residual column is evaluated.
Partial or failed solves are inconclusive, with no partial upper-bound claim.

No complete tangent/inverse matrix is retained. Storage includes bounded column
batches, three row accumulators and Krylov restart vectors. N iterative solves
plus N direct tangent columns are required; no fast-work claim is made. The
underlying action error and interval certification flags remain false, and
dense DC work may remain. Public configuration and exact snapshot policies are
unchanged; selecting this gate in public implicit roots is subsequent work.

Targeted GPU1 tests: session16695 terminal exit0, 2passed in5.30s, command
`python -m pytest -q tests/mom/test_harmonic_inverse_bounds.py` in development.
An actual native nondominant HB tangent rejected by the prior dominance check
passes the new bound, bounded against an independent dense SVD oracle for two
batch sizes. Global Jacobian/direct solve/SVD calls are forbidden during checks.
Singular tangents, exhausted iteration budgets, and deliberately loose solves
returning zero candidate columns cannot be admitted.

Integration trees remain frozen at Maxwell ad34e45 / Bench325dfb4 for live
session71174, /tmp/gsp11-merge-regression-v2. Prior Bench environment recheck
completed193passed63.71s with unchanged source hashes before that merge.
This new internal primitive is not included in that frozen regression.
Independent read-only audit found no confirmed P1/P2 in the norm derivation,
batch orientation, incomplete/loose solve rejection or mutation guard. Dedicated
nonfinite norm-accumulation/rounding overflow tests have not been run; source
inspection only confirms the finite gate for those cases.

### Public streamed inverse-residual policy — 2026-09-22

HB configuration and native solve explicitly accept streamed_inverse_krylov.
Both final implicit corrections create their own fixed tangent and require a
complete inverse-residual bound before their residual correction solve. There
is no automatic fallback or tolerance change. Snapshots validate the selected
method, inner rule, exact column count/batch, complete/no-failure status, finite
inverse/residual norms, residual bound below one, and the identical lower-bound
formula in addition to the existing numerical singularity cutoff. Historical
checks remain consistency-checked records, not recomputed conditioning proofs.
Shared native validation also applies to Bench cutoff history policies.

Targeted GPU1 outcomes:
- 56501 terminal exit0:2passed11.70s, /tmp/gsp11-inverse-root-v1.{log,xml};
  native root/material gradient versus dense with global Jacobian/direct solve/
  SVD forbidden on the new path, and singular root rejection.
- 47884 terminal exit0:1passed24.28s, /tmp/gsp11-inverse-public-v1.{log,xml};
  actual public dense dielectric solve, exact snapshot/field restore with
  solves forbidden, and policy/bound tamper rejection. Saved result at
  /tmp/gsp11-inverse-public-v1/test_public_krylov_policy_rest0/result.pt.
- Bench41504 terminal exit0:1passed3.88s, inverse-bench-policy-v1.{log,xml}
  under /tmp; generated candidates bind the new option and all active settings.
- 83465 terminal exit0, /tmp/gsp11-inverse-bench-replay-v1.log: reused saved
  public result, exact Bench archive plus three changed-policy rejections,
  all solves forbidden. /tmp/gsp11-inverse-bench-v1/dense manifest SHA256
  634962f7785717b872e16e49a9f67626aaabda884934353e64bb161f715782ae.

Independent read-only source audit found no confirmed P1/P2 in public selection,
final correction integration or policy/bound restoration. Primitive and new
public tests do not grant physical, material/branch or fast-work qualification.
No additional full regression was launched: frozen integration ad34e45 and
Bench325dfb4 remain in live71174 stage regression.

Additional non-dominant converged-root control:91477 terminal exit0,
1passed5.61s,/tmp/gsp11-nondominant-root-v1.{log,xml}. The actual converged
native HB root fails the previous dominance gate; the new inverse-residual
path recovers the dense root and material gradient with global Jacobian/direct
solve/SVD forbidden. This verifies a newly admitted algebraic case, not physics.

Compressed saved-root follow-up36751 remains live at this checkpoint:
/tmp/gsp11-inverse-root-replay.py, /tmp/gsp11-inverse-root-replay-v1.log.
It restores the previous current-H, mixed-H2 and dual-H2 preparations and starts
at the saved roots, checking only the new final implicit path (zero outer
Newton iterations expected). No new preparation or archive. This coverage is
pending, not part of the passed public dense fixture evidence above. Do not
restart the live job; use its session handle or inspect authoritative processes.

### Cellwise DC current tangent primitive — 2026-09-22

NodalCurrentBalance and FaceCurrentBalance now expose tangent_action(T), a
frozen reduced G^T W T G action. Vertex/face degree zero is pinned exactly as
in the current dense Newton solve. Products contract cell gradients and the
local 3x3 derivative, scatter weak contributions and remove the pinned row.
Transpose uses T^T and the diagonal uses the same local contraction, without
forming a global matrix. Storage/work per vector scale with cells and degrees
of freedom; retained T is a detached clone, not a material-gradient path.

The custom vector backward applies the opposite action, allowing higher vector
derivatives while checking delayed operator/geometry changes. T need not be
symmetric or positive definite. This is not nonsingularity admission: zero T
correctly acts as zero. The original dense DC nonlinear solve is unchanged;
integrating bounded iterative inverses and their conditioning gates remains
necessary before claiming removal of dense DC work.

GPU1 targeted session63158 terminal exit0:5passed4.12s,
/tmp/gsp11-dc-tangent-v1.{log,xml}. Tests cover nodal/face bases in float32/64,
heterogeneous nonsymmetric T versus original assembled matrix, diagonals,
batched products, first/second vector gradients, empty/zero batches, external
T isolation, delayed mutation and geometry rejection. Global matrix assembly
and Jacobian calls are forbidden after constructing the comparison oracle.
An independent weak-current identity checks gauge/sign for the face basis.
Independent read-only audit found no confirmed P1/P2 in contractions, transpose,
batch layout, vector autodiff or guards. No new full suite was launched.

Compressed inverse-root follow-up36751: current-H case completed with zero
outer Newton iterations, final residual2.5835717058159237e-17; both complete
inverse checks lower0.9571673377615676/residual-upper7.585685464976064e-11.
Mixed/dual H2 cases remain pending; the session is still confirmed live.
Integration session71174 remains live on frozen ad34e45/325dfb4.

### Native DC streamed inverse solves — 2026-09-22

Native nodal/face solve accepts an explicit tangent_rule and column batch.
Each Newton direction and both final implicit corrections use a frozen cellwise
current derivative, complete inverse-residual admission, then residual-checked
GMRES. Global DC tangent assembly/eigendecomposition/direct solves are absent
on this path. The internal Ohmic/TPA law has PSD local derivatives for accepted
nonnegative coefficients and AC covariance; numerical nonsingularity admission
can therefore replace its prior positive-definiteness check. This does not
assert positivity for arbitrary nonsymmetric tangent primitives.

HB and DC share the extracted fixed_tangent_inverse implementation. Existing
HB receipt versions/fields, per-RHS refinement budgets, original-RHS thresholds
and implicit transpose backward are retained. DC reports reduced-current
Euclidean residuals. Default dense DC retains its matrix checks and equations;
when iteration budget is already exhausted it no longer computes an unused
matrix, which can change priority between two simultaneous failure reasons.
Outer DC threshold overflow rejects before any successful root claim.

Public HB closure/config selection and persistence for the DC option remain
unimplemented. N complete inverse-column solves per check can be expensive;
no fast-work, interval, underlying-action-error or physical qualification.

Targeted GPU1 terminal evidence:
- 41207:9passed36.62s,/tmp/gsp11-dc-krylov-v1.{log,xml}. New nodal/face
  nonlinear roots and conductivity/AC parameter gradients versus dense with
  assembly/direct solve/eigendecomposition/Jacobian forbidden; fixed DC inverse
  transpose and first/second RHS derivatives; budget rejection and overflow;
  extracted HB inverse/bounds existing controls, including nondominant roots.
- 19250:8passed6.20s,/tmp/gsp11-dc-default-v1.{log,xml}, original default
  nodal/face roots, physical identities and gradient controls.
- 92862 originally FAILED, /tmp/gsp11-dc-singular-v1: zero DC law with
  nonzero AC bypasses the unexcited shortcut. CUDA least-squares returned
  inf/NaN for a singular Arnoldi problem, and the native operator raised an
  unstructured input error before convergence evidence could be recorded.
  Shared GMRES now checks candidate finiteness before replacing x or applying
  the operator. It retains the last finite x/residual with nonfinite_iterate.
  Corrected-node19700:1passed3.42s,/tmp/gsp11-dc-singular-v2.{log,xml}.
- 15889:1passed3.15s,/tmp/gsp11-dc-nonfinite-v1.{log,xml}; deterministic
  NaN least-squares injection produces three bounded failed correction
  receipts, unchanged original-RHS residual and no invalid operator call.
- 76101 common-GMRES controls:3passed1failed3.90s, preserved in
  /tmp/gsp11-gmres-finite-v1.{log,xml}. Existing inconsistent-system test
  expected older failure reason vocabulary. Updated to include the explicit
  nonfinite_iterate reason and added finite-returned-solution assertion.
  Failed-node-only95127:1passed2.97s,/tmp/gsp11-gmres-finite-v2.{log,xml}.

Independent read-only audits found no confirmed P1/P2 in extraction, PSD-law
argument, implicit derivatives, failure receipts or the nonfinite candidate
fix. Frozen integration71174 and compressed-root follow-up36751 remain live;
no new full regression, electromagnetic preparation or archive was started.

### HB current closure and native policy codec — 2026-09-22

HarmonicCurrentClosure explicitly carries the native DC tangent rule and column
batch, guards them against mutation and passes them through every per-body
current solve. Active policy serialization is canonical; inactive dense defaults
emit no new fields. Native exact preparation restoration validates the paired
active fields, complete StoppingRule representation, supported GMRES controls
and positive integer batch. It initializes dense defaults on old records.

Public HB configuration does not yet select this DC policy. Accordingly public
restore explicitly rejects a nonempty inner policy instead of silently accepting
an injected one. This is an intermediate dependency: public selection, full DC
solver-receipt policy binding and Bench contract propagation remain required.
Mixed closures inherit the inner current codec, but dual-conductive closure
uses a different formulation and is not covered by this new selection.

47557 new closure tests initially1failed1passed4.27s: the algebraic fixture
used unit cell weights rather than the real closure tetrahedron volume. Fixed
the fixture to volume.volumes; failed-node-only33631 terminal1passed10.09s,
/tmp/gsp11-dc-closure-v2.{log,xml}; originalv1 retained. The test solves a
complete algebraic HB root with the iterative DC closure, forbids global DC
matrix assembly, and compares electric state plus conductivity/TPA gradients
against the original dense closure. Outer HB still uses dense final checks.

28974 saved current-H preparation replay terminal exit0:
/tmp/gsp11-dc-policy-codec-replay.py and /tmp/gsp11-dc-policy-codec-v1.log.
Native active policy encode/restore/encode digest is exact with solve and closure
construction forbidden. Four malformed active policies reject; injection into
the saved public default also rejects. Source result reused from prior current-H
public validation; no EM preparation or root solve. The reusable helper now
runs within the existing public-current fixture at a future merge regression;
that full fixture was not rerun in this stage.

Long-running36751 has now passed current-H and mixed-H2 final inverse checks
at saved roots with0outeriterations, residuals2.5835717058159237e-17 and
4.5876777540110835e-17. Dual-H2 still pending. Integration71174 stillrunning
on frozenad34e45/325dfb4; no new full regression started.
Independent read-only audit found no confirmed P1/P2 in policy propagation,
canonical fields, default/mixed restoration or public injection rejection.

### Public DC tangent policy and exact contract — 2026-09-22

HB configuration now accepts explicit dc_tangent_rule/dc_tangent_column_batch.
Inactive fields are omitted from canonical payloads; active pairs are explicit.
Electric-current and mixed current/dielectric preparation propagates the policy,
including cutoff extension and exact-prefix policy guards. Pure dielectric and
separate dual-conductive formulations reject this unsupported choice before
basis/operator assembly; their remaining implementation obligations are open.

Restore binds public settings to native closure settings. Both final state DC
records and the HB solver's DC records validate method, rule, column batch,
actual reduced face dimension/precision, Newton iteration count plus two final
implicit phases, complete inverse-residual checks and numerical bound formula.
Historical records are checked for consistency without rerunning their solves.
Default representations remain unchanged. This removes the earlier temporary
blanket public rejection of a nondefault native DC policy.

Bench candidate settings explicitly bind active DC fields. Cutoff history uses
shared validate_current_receipt on each extension's DC records, with reduced
face counts derived from actual volume-cell topology and mixed current-region
ordering. Numerical history acceptance is not physical/spatial/fast-work proof.

Targeted evidence (GPU1):
- 41096:1failed2passed4.80s,/tmp/gsp11-dc-public-v1.{log,xml}. Unsupported
  pure dielectric/dual pre-assembly rejection passed. New current fixture
  mistakenly asked its compressed setup helper for a dense backend with local
  Riesz/hierarchical settings; failure occurred during config construction.
  Corrected fixture constructs H config then explicitly chooses dense settings.
- Failed-node-only70930:1passed28.12s,/tmp/gsp11-dc-public-v2.{log,xml};
  actual public current root with DC global matrix assembly forbidden, exact
  DC fields and encode/decode digest with solves forbidden, five tamper cases
  spanning public/prepared/state/solver/bound records rejected. Saved result:
  /tmp/gsp11-dc-public-v2/test_public_current_dc_policy_0/result.pt.
- Bench49859:4passed5.54s,/tmp/gsp11-dc-bench-binding-v1.{log,xml}; new
  DC candidate fields and old inactive/Newton-Krylov candidate controls.
- 66264 terminal exit0,/tmp/gsp11-dc-public-bench-v1.log: reused the new
  public saved result, exact Bench roundtrip and three changed-policy rejects,
  all solves forbidden. /tmp/gsp11-dc-public-bench-v1 manifest SHA256
  1d1d43c3db592fd4f4547c0b2329214976f9fba17120ad02afd1f8e9d81f8c82.
- 27389:1passed60.10s,/tmp/gsp11-dc-cutoff-v1.{log,xml}; actual public
  current+TPA H1->H2 extension with active DC policy, exact Bench restore,
  unchanged recorded harmonic_budget_exhausted, and three mutated historical
  DC policies/bounds rejected. Saved state under the single test directory
  in /tmp/gsp11-dc-cutoff-v1. Existing cutoff nodes were not repeated.

Independent source audit found no confirmed P1/P2 in public policy propagation,
exact binding, reduced-dimension/phase validation or Bench history ordering.
No new full suite was launched; original frozen ad34e45/325dfb4 regression71174
remains running. Mixed public test66683 remains live at this checkpoint,
/tmp/gsp11-dc-mixed-public-v1.{log,xml}, no pass yet claimed.

Mixed follow-up66683 now TERMINAL exit0:1passed84.46s,
/tmp/gsp11-dc-mixed-public-v1.{log,xml}. Actual two-body mixed public solve
uses reversed volume ordering and the active current policy; exact DC query,
snapshot digest and public/prepared/state/solver/bound mutations all checked.
This closes this stage's mixed public fixture, not mixed cutoff/physics/scaling.

### Cellwise conservative current projector — 2026-09-22

StaticCurrentProjector implements P=I-G K^-1 G^T W and reduced potential
recovery R=K^-1 G^T W, where K=G^T W G is the pinned face-mean stiffness.
The fixed cellwise DC inverse performs complete numerical inverse-residual
admission at construction and original-RHS checks for each subsequent solve.
No global projector, recovery or gradient matrix is built or stored.

P is W-orthogonal, not generally Euclidean symmetric: adjoint products use
I-W G K^-T G^T and recovery adjoints use W G K^-T. Custom vector backward
switches the paired actions and retains delayed mutation guards. Geometry and
operator derivatives remain outside this fixed-action contract. Construction
budget failure is inconclusive, not proof of singularity. Storage per vector
is linear, but complete admission still needs all inverse columns. Numerical
projection idempotence/conservation are tolerance-dependent, not interval proofs.

GPU1 targeted77359 terminal exit0:2passed7.17s,
/tmp/gsp11-dc-projector-v1.{log,xml}. A four-tetrahedron unequal-volume mesh
has demonstrably nonsymmetric P, so transpose placement of W is tested rather
than hidden by equal weights. Products/recovery/diagonal-free construction are
compared with an independently assembled dense oracle; direct global solves
and block assembly are forbidden during construction and use. Controls cover
weak-current conservation, weighted self-adjointness, idempotence, forward/
transpose recovery, batched first/second vector gradients, zero-length batch,
insufficient budgets, delayed tangent mutation and geometry changes.
One warning from converting a grad-bearing test diagnostic to float was
preserved in v1; the diagnostic now detaches explicitly, with no solver change.
No redundant rerun was made for that warning-only test edit.

Independent read-only audit found no confirmed P1/P2 in weight placement,
transpose/autodiff, admission or guards. This is an internal dependency for
removing dense dual DC projectors; the dual curl, coupled solve, nonlinear
response and public snapshot integration are still unchanged and open.

### Bounded-block static volume curl — 2026-09-22

StaticVolumeCurlAction evaluates the existing static volume Green gradient at
all owning cell centers, selecting both target and source batches before native
integration and skipping unused Hessians. Cross products retain all self and
cross-body interactions in supplied volume order. The Euclidean transpose uses
the actual transposed blocks, not a reciprocity/equal-volume assumption. Fixed
vector backward calls that transpose; geometry/config changes reject, including
delayed backward. Only center points and mesh references persist, with no global
curl matrix or retained kernel blocks. Kernels are recomputed on each action:
work remains quadratic, and no fast-work/physical qualification is implied.

55978 terminal exit0:2passed4.09s,/tmp/gsp11-dc-curl-v1.{log,xml}. Two bodies
with unequal cell counts and reversed volume ordering compare against original
full native integration using an independent cross-product basis construction.
The test asserts nonzero cross-body blocks, checks selected source/target limits
and Hessian omission at every integration call, validates forward/transpose and
first/second vector derivatives, plus empty batches, invalid policy and delayed
mutation/geometry guards. This proves operator equivalence to the existing
native discretization, not an independent physical reference.

Earlier36751 has now TERMINATED exit0: current-H, mixed-H2 and dual-H2 saved
roots all pass the new HB inverse-residual final checks with0outeriterations;
final residuals2.5835717058159237e-17,4.5876777540110835e-17 and
7.36322297746377e-17. /tmp/gsp11-inverse-root-replay-v1.log and
/tmp/gsp11-inverse-root-{current,mixed,dual}-v1.pt. No reassembly or rerooting
was required. This does not validate new DC projection/curl integration, which
is still pending. Frozen stage regression71174 remains live; do not restart.

Independent audit found a complexity P2 in the first implementation: each
selected native integration recomputed the full mesh vertex extent. Fixed
batching would therefore add a cubic scanning term despite the quadratic pair
kernel work. Fixed with PreparedVolumeGreenGeometry, a complete-mesh extent
cache guarded by mesh identity, geometry versions and extent identity/version.
cell_green_potentials accepts this optional native geometry context, preserving
the original complete-mesh trace tolerance; selected-cell extents are never used.
Each curl volume now builds one context and reuses it across all blocks.

66544 terminal exit0:3passed4.21s,/tmp/gsp11-dc-curl-v2.{log,xml}. Updated
product/gradient tests intercept complete-mesh vertex amax/amin and reject any
such repeated scan after preparation. Cached/uncached selected-column results
match exactly, face-query rejection remains unchanged, and mutated cached
extent rejects. The earlier v1 result remains numerical-equivalence evidence,
not proof that its original implementation met the stated work bound.
Independent read-only follow-up confirmed the repeated-scan P2 is closed;
no remaining confirmed P1/P2 was found in cache binding, trace semantics,
block products, transposes or vector autodiff.

### Dual conductive DC block action — 2026-09-22

DualStaticCurrentAction combines admitted per-body conservative projectors with
the global blocked static curl. Its packed channels are A[J,M]=[J+De P C M,
M-Dm P C J], and the Euclidean transpose explicitly reverses C/P/diagonal order:
[J-C^T P^T Dm M, M+C^T P^T De J]. Impressed source projection and its transpose
are also available. Conductivities must be positive homogeneous samples per
body and are frozen by value; operator/material/geometry derivatives are not
part of this fixed-vector action contract. Mutation guards include nested
projectors/curl, conductivity tensors, body ordering and numerical bindings.

FixedDualCurrentInverse reuses bounded fixed-action GMRES and original-RHS
checks with implicit transpose vector backward. Coupled nonsingularity is not
certified merely by constructing either the action or inverse. No global
coupled, projector or curl matrix is stored; direct blocked curl and complete
projector admission still prevent a fast-work claim. Full dual source-to-field
recovery, nonlinear closure, public policy and exact snapshot integration remain
open; the original public dual closure has not been replaced in this stage.

GPU1 session47523 terminal exit0:2passed17.85s,
/tmp/gsp11-dual-action-v1.{log,xml}. A two-body, reversed-order, unequal-volume
fixture compares with the original dense dual matrix and projection. Global
direct solves, block-diagonal assembly and DC stiffness assembly are forbidden
after reference construction. Checks cover A/A^T, projected source/transpose,
batched first/second action-vector gradients, fixed inverse versus dense solve,
original dense residual and inverse RHS gradient, empty batches, failed Krylov
budget, delayed conductivity mutation and invalid/nonhomogeneous samples.
These are discretization/algebra controls, not independent physical references.
Independent read-only audit found no confirmed P1/P2 in signs, coefficient/
projection/curl ordering, fixed-vector gradients or nested guards. Inner
projection residuals are not an aggregate coupled-operator error certificate.

### Streamed dual current field recovery

StreamedDualCurrentBalance requires complete numerical coupled inverse-residual
admission before solving impressed electric/magnetic currents. It recovers both
cell-center fields and face potentials with each body's volume-mean gauge,
retains the original dense formulation's unprojected physical-current scale and
1024*eps*dimension residual threshold, and reports reconstruction differences
and normal-current jumps. Source derivatives use the fixed inverse transpose
and native projection/recovery/curl transposes. Global matrices are not stored;
complete admission still requires every inverse column. Inner projection errors
are not an aggregate action certificate. Material/geometry derivatives,
nonlinear/public policy and snapshot integration remain separate obligations.

GPU1 session30826 terminal exit0:1passed54.67s,
/tmp/gsp11-dual-fields-v1.{log,xml}. Reversed two-body unequal-volume dense
reference covers E/H, both currents, potentials, volume-mean gauge, source
loss gradients and the original dense matrix residual. Dense solve/block-diagonal
and stiffness assembly are forbidden after reference preparation. Zero sources,
invalid shapes, conductivity mutation and a deliberately incorrect inverse are
checked; the latter cannot bypass the original physical-current gate.
Independent read-only audit found no confirmed P1/P2. Reconstruction errors
remain reported diagnostics, not independent field-convergence thresholds.
Frozen integration regression71174 has completed Maxwell MoM1055passed3skipped;
Bench remains live, so whole-run source hashes and final status are pending.

### Frozen integration regression v2 terminal evidence

Runner71174 is terminal exit0; /tmp/gsp11-merge-regression-v2/receipt.json has
status passed and sources_unchanged true. Frozen Maxwell ad34e45 tests/mom:
1055passed3skipped3930.82s. Frozen Bench325dfb4 tests:1650passed36skipped526.96s.
Independent read-only audit matched both XMLs, logs, return codes, exact HEADs,
runner hash and all694 package/test Python hashes before, after and currently.
The three Maxwell skips require two visible GPUs. Bench skips comprise six
coordinated geometry CUDA tasks,17 explicit calibration executions,one explicit
order smoke and12 retained local replay-evidence checks. Skips are not passes.
External EMFC/evidence/historical-provider paths were configured explicitly but
not included in the694-file source hash set. These results do not cover later
development commits, all Maxwell tests, four-repository final acceptance or
physical qualification. The original v1 failures and successful environment
recheck remain retained; no archived evidence or physics threshold is replaced.

### Streamed dual nonlinear closure integration

DualHarmonicCurrentClosure has an explicit internal streamed_rule and column
batch. That rule controls conservative projection, coupled currents and local
nonlinear tangent inverses. DualElectricResponse applies the admitted coupled
inverse to projected electric sources; its transpose reverses projection,
inverse and conductivity scaling. DualNonlinearTangent applies I-R*T and its
transpose with only cell-local3x3 constitutive blocks. Newton directions and
both final implicit corrections require complete numerical inverse-residual
admission and original-RHS GMRES checks. Original nonlinear residual, line
search and final coupled-field consistency thresholds remain unchanged.

The dense default and existing exact snapshot format remain in use publicly.
The new internal mode explicitly rejects trainable conductivities and snapshot
requests until material derivatives and exact persistence are implemented.
AC-field and TPA gradients remain native implicit gradients; no global response
or nonlinear tangent array is stored on the streamed path. Nested iterative
cost, underlying action errors, public policy, exterior decoding and physical/
fast-work qualification remain open; fixed-conductivity scope is not final scope.

GPU1 session79562 terminal exit0:8passed637.63s,
/tmp/gsp11-dual-nonlinear-v1.{log,xml}. New unequal-cell-volume one-body fixture
compares the nonlinear root, E/H/currents/TPA source and AC-amplitude/TPA
implicit gradients with the original dense closure. It also compares R/R^T,
non-symmetric local-tangent forward/adjoint and first/second vector gradients,
forbids dense global solves/block assembly, checks Newton plus two implicit
admissions, exhausted nonlinear budget, mutation, trainable-conductivity and
snapshot rejection. Six affected original dense closure/codec cases also pass.
Independent read-only audit found no confirmed P1/P2. Runtime demonstrates
nested admission cost; it is not a fast-work qualification. This control does
not establish multiple-body nonlinear material gradients or public integration.

### Dual current material inverse derivatives

DualMaterialInverse wraps an admitted StreamedDualCurrentBalance. Packed live
electric/magnetic conductivity tensors must exactly match the fixed action's
values; changing them requires new preparation. Normal and transposed inverse
backwards use paired transposed/normal solves and -lambda^T(dA)x. The backward
exchanges primal/dual roles for A^-T, keeps both conductivity placement/signs,
and sums leading RHS batch dimensions. Saving the forward output and recursively
using the material-aware paired inverse preserves mixed second derivatives.
Electric-source field response also retains explicit division by conductivity.
Original source/action guards and saved material version checks remain active.
Geometry derivatives are not introduced. This independent kernel does not yet
remove the nonlinear closure's trainable-conductivity rejection or add public
configuration/persistence support; those integrations remain required.

GPU1 session52697 terminal exit0:1passed78.33s,
/tmp/gsp11-dual-material-v1.{log,xml}. Reversed two bodies with unequal volumes
and distinct electric/magnetic conductivities compare normal inverse,
transposed inverse, electric response, batched RHS/material gradients and mixed
second derivatives with a dense differentiable reference. Dense solve/block
and stiffness assembly are forbidden after reference construction. Empty batch,
material mismatch and delayed mutation rejection are checked. Independent
read-only audit found no confirmed P1/P2. Numerical material derivative agreement
is not an independent physical or scaling qualification.


### Material derivatives in the streamed dual nonlinear closure

The residual now uses DualMaterialInverse.electric_response with live packed
conductivities; frozen DualElectricResponse remains the same-value operator for
Newton and implicit tangent solves. Final current recovery uses the live
conductivities in the material inverse, field divisions and potential drives.
The coupled physical-current residual keeps the same threshold. Original
material tensors and packed values are version-bound; a native coefficient
backward guard checks the owning preparation on delayed material backpropagation.
The existing first-order implicit nonlinear-gradient contract is unchanged;
the independent inverse kernel's second-derivative result does not qualify
second-order derivatives of the complete nonlinear root.

GPU1 session23022 runs the updated nonlinear root fixture plus affected dense
snapshot controls, /tmp/gsp11-dual-material-closure-v1.{log,xml}; no result yet.
The new fixture adds both conductivities and potential losses to the dense
reference gradient comparison and checks solve/delayed-backward material
mutation. Previous fixed-conductivity root evidence79562 remains retained.
Independent source audit found no confirmed P1/P2; numerical verification is
pending. Public selection and exact streamed persistence remain unsupported.

Next integration must address all dense-only assumptions together: prepared
state identity currently calls current._state; DualStaticExterior and query
recovery access volumes/counts/blocks/points; snapshot validators use retained
projector/curl/response matrices. Public policy cannot simply enable the new
closure before its exact decoder and current/field validation are ready.
A decode-only restoration must not rerun admission or silently reintroduce
global dense matrices. Existing physical and complexity gates stay unchanged.


Material integration GPU1 session23022 now TERMINAL exit0:4passed675.15s,
/tmp/gsp11-dual-material-closure-v1.{log,xml}. Actual nonlinear root fixture
compares AC-amplitude,TPA and both conductivity gradients including recovered
potential losses against the dense reference. Original material mutation is
rejected on reuse and delayed backward. Inactive policy and two existing dense
preparation/state decode-only cases also pass. Independent read-only audit found
no confirmed P1/P2. This closes internal first-order material integration;
public selection, streamed exact state persistence and physics remain open.

### Decode-only conservative projector component

static_projector_snapshot encodes the native projector rule, refinement budget,
receipt and identity of the independently validated face block, including cell
volumes and local topology/gradient/weight arrays. The owning preparation passes
an expected identity to restore. Restoration checks the exact schema/digest,
native geometry binding, supported canonical GMRES policy, complete numerical
admission flags, finite inverse-bound arithmetic and precision-dependent cutoff.
It rebuilds the local identity tangent and fixed inverse metadata without any
inverse solve, factorization or repeated all-column admission. The block itself
must already have been restored and validated; this component does not replace
full dual state/current/exterior consistency checks. Historical bounds are not
newly certified merely by reopening a record.

GPU1 session64835 terminal exit0:1passed4.01s,
/tmp/gsp11-projector-codec-v1.{log,xml}. Explicit deterministic follow-up87129
terminal exit0:1passed4.57s, /tmp/gsp11-projector-codec-v2.{log,xml}; fixture uses
CUBLAS_WORKSPACE_CONFIG=:4096:8 and restores prior Torch deterministic settings.
Saved weights-only record rebinds a fresh equivalent native block while solve,
inverse, singular-value computation, projector construction, block construction
and admission are forbidden. Projection/recovery match bitwise after reopening.
Changed policy/geometry/bound records are rejected; an inconsistent lower bound
is rejected even with a recomputed metadata identity. Nested tangent mutation
is also rejected. This is a component replay, not full public persistence.

Independent read-only projector-codec audit found no confirmed P1/P2. Restored
adjoint/recovery-adjoint and delayed backward are not separately exercised by
this codec fixture; native action/guard coverage and source inspection support
those paths only indirectly. Rebound illegal rule/column/cutoff/qualification
fields have source validation, but the fixture directly exercises only one
rebound inconsistent-bound negative. No broader coverage is claimed.


The restored action-gradient coverage gap above is now closed by GPU1
session10305 terminal exit0:1passed5.46s, /tmp/gsp11-projector-codec-v3.{log,xml}.
Under explicit deterministic settings, restored adjoint, recovery-adjoint and
combined projection/recovery vector gradient also match bitwise. A delayed
backward after nested tangent mutation is rejected. Production codec did not
change for this test extension. Rebound invalid-metadata cases beyond the
inconsistent-lower-bound example remain source-reviewed, not independently
executed per field. No complete dual/public codec or physical claim is added.

### Streamed dual current and nonlinear preparation codecs

The dual_action_snapshot component binds electric/magnetic material arrays,
body-ordered projector records, blocked curl centers/extents/configuration,
current inverse policy and complete historical inverse bounds. Restore accepts
independently validated native face blocks and an owning expected identity;
geometry-only center/extent construction replaces no kernel integration and
reopens no solve/admission. Only linear-sized material arrays and metadata are
stored, with no coupled/projector/curl matrices. Historical bound consistency
and cutoff checks do not re-certify numerical conditioning or action error.

Nonlinear preparation restoration additionally checks all nonlinear settings,
identical projector/current/nonlinear rules and column batches, and native
refinement budgets. Caller-sampled conductivities must exactly match the saved
fixed action. Their native graph is retained through the material inverse and
_BoundConductivity guard. No nonlinear root is recomputed. These codecs remain
components: the owning archive must restore/validate face arrays, physical
state, exterior layers and full HB consistency before public selection can be
enabled. Existing public streamed snapshot rejection remains in place.

GPU1 session96586 terminal exit0:1passed84.98s,
/tmp/gsp11-dual-action-codec-v1.{log,xml}. Reversed two-body unequal-volume and
distinct-conductivity preparation survives weights-only round trip with a fresh
native block set while constructors, integrations, solves/inverses and singular
value work are forbidden. Replayed currents, E/H and gauge-fixed potentials
match bitwise under explicit deterministic CUDA settings. Material/policy/curl/
bound identity changes, rebound inconsistent lower bound, reversed blocks and
nested material mutation are rejected. This fixture does not directly test
restored transposed current inverse or every rebound illegal metadata field.

GPU1 session27741 terminal exit0:1passed20.80s,
/tmp/gsp11-dual-closure-codec-v1.{log,xml}. Only the new nonlinear-preparation
node ran; the prior two-body case and long nonlinear roots were not repeated.
Reopened response and source/material first derivatives match bitwise, wrong
caller material samples and rebound mismatched column batch reject, and delayed
original material mutation is guarded. Restored nonlinear root/state/exterior
has not yet been exercised by this component fixture. Independent read-only
audits found no confirmed P1/P2 in either codec. No physical or fast-work
qualification is inferred from either preparation replay.

### Streamed dual exterior and bound state-contraction replay

StreamedDualCurrentBalance now exposes guarded geometry/material properties and
_state for existing exterior and preparation ownership checks, without matrix
aliases. The original DualStaticExterior handles neutral layers and native
volume-current fields on this preparation. Shared dense snapshot construction
and restore checks were factored without changing their formulas or thresholds.

A new streamed_dual_state_snapshot envelope computes Cj/Cm and the linear source
response during encoding and binds them with the full physical state, sources,
preparation and receipts. Restore requires an independent owning identity and
replays these historical native calculations without integration or inverse
solve. It retains the original field reconstruction, weak-current, layer and
flux checks, and verifies source constitutive consistency. This is a replay
trust boundary, not independent re-evaluation of C/R or new physical evidence.

Independent design review identified a P2 risk in replacing the old E-RSe gate
by local constitutive/flux equations: finite residuals can be inverse-amplified.
That substitution was not implemented. The actual electric response is retained,
and validate_response_witness keeps both original DC norms and returns RSe,
not actual E, for the future owning HB cross-harmonic norm. Source values must
match exactly and the caller must supply the original finite threshold. Full
HB identity/threshold/total-norm integration remains required before public use.

GPU1 replay88498 terminal exit1, /tmp/gsp11-streamed-exterior-replay-v1.log:
reversed volume order was incorrectly passed to a test helper that enumerates
native surface IDs. This fixture-only binding mismatch was corrected by building
the surface system in region order and restoring supplied volume order. Failure
is retained. GPU1 replay32494 terminal exit0,
/tmp/gsp11-streamed-exterior-replay-v2.log: exterior fields and source gradients
match the dense formulation; bound state replay matches decoded exterior fields
bitwise with kernel/inverse/exterior-constructor work forbidden. State/current,
source, curl, response and receipt changes reject against the independent
identity. The original DC norm remains nonzero for an admitted perturbed field;
larger defects and mismatched owning sources reject. Reused admitted current
preparation /tmp/gsp11-dual-action-preparation-v1.pt; no admission repeated.
One native state artifact is retained at /tmp/gsp11-streamed-dual-state-v1.pt.
Helpers now accept optional output_path and do not overwrite that evidence.

GPU1 session65313 terminal exit0:3passed5.08s,
/tmp/gsp11-dual-state-default-v1.{log,xml}, covering affected original dense
state/preparation snapshots. The replay helpers are also connected to the
collected test_dual_preparation_decode_without_kernels_or_admission fixture for
future stage regression; its original85s preparation was not repeated now.
Independent source audit found no confirmed new P1/P2. The response-gate design
issue is closed only at helper level, not at the still-pending public HB bridge.
No new physical, complexity or production qualification follows from replay.

### Complete native streamed dual preparation dispatch

snapshot_dual_preparation/restore_dual_preparation now dispatch the explicit
`dual-streamed-preparation-v1` schema. The owning identity binds actual face
arrays, floating-potential arrays, nonlinear preparation and the linear
surface/volume binding. Face topology, affine gradients and volume weights are
restored through the same extracted helper and original tolerance used by the
dense path. Its projector/recovery and coupled-matrix checks remain unchanged.
The streamed branch restores the closure component against those validated
blocks and caller material samples, then the native floating operator, and
compares the complete surface/volume preparation identity. No constructor
assembly, integration, solve or inverse admission occurs on reopen.

The existing dense floating Laplace matrices and factors remain stored; only
current/projector/curl global arrays are removed on the streamed branch. Do not
claim linear storage for the complete preparation. Public selection and owning
HB state identity/response-total-norm integration remain pending, even though
the native preparation entry no longer rejects streamed closures.

GPU1 session91821 terminal exit0:3passed18.52s,
/tmp/gsp11-complete-dual-preparation-v1.{log,xml}; exclusive fixture directory
/tmp/gsp11-complete-dual-preparation-v1. A new one-body preparation/state record
rebinds fresh face and floating operators with constructors, kernel integration,
solves/inverses and SVD forbidden; restored exterior fields and state match
bitwise. Face/Laplace/policy changes reject against owning identity, and rebound
invalid gradients reject against native affine geometry. The two existing dense
preparation/state cases also pass. The old long nonlinear test only drops its
now-obsolete unsupported-snapshot assertion; its root was not recomputed.
Independent source audit found no confirmed P1/P2. Reversed multi-body and
material-gradient component evidence remains separate from this single-body
complete chain; no complete public HB or physical qualification is inferred.

### Internal HB streamed state decoder bridge

The HB state now has a separate optional `dc_dual_binding`, included in the
nested mutation guard and appended after existing positional fields. The dual
bridge dispatches streamed decode through that owning identity. Validation
matches the actual HB electric source exactly and requires zero magnetic source,
retains both native field residual gates, shared constitutive and surface-charge
checks, and returns the witnessed response for the outer total residual. External
queries can decode without a live dual object or inverse/kernel recomputation.
Dense states reject a streamed binding; their saved representation is unchanged.

GPU1 session96744 terminal exit0, /tmp/gsp11-streamed-hb-bridge-v1.log,
reused the admitted complete preparation from stage5fc22eb. The helper checks
bitwise response/exterior fields, wrong or missing binding, source alteration,
excess field residual, nonzero accepted residual and binding mutation. No
nonlinear root or admission was rerun. The helper is also attached to the
existing collected complete-preparation test for future regression. This turn
ran it independently against the retained artifact, not a second preparation.

GPU1 session8136 terminal exit0,
/tmp/gsp11-streamed-hb-bridge-dense-v1.log, reopens the previously saved public
compressed dual result, checks interior/exterior finite fields, exact complete
payload digest and rejection of a streamed binding on the dense closure, with
surface inverses and direct solves forbidden. No root or preparation rerun.
Independent read-only audit found no confirmed P1/P2. The streamed fixture is a
constructed DC state, not a complete nonlinear HB root; streamed interior query,
public recovery/save/load and policy/cutoff/Bench wiring remain pending.

Recovery integration must take care with inner DC versus outer HB sources:
existing recovery solves the inner nonlinear root again, so its source need not
be bitwise identical to the outer HB source. Bind the actual owning HB source
and retain the original field/constitutive tolerances; do not weaken exact source
identity or replace the total residual by local flux checks. Nonlinear acceptance
metadata must be sealed before hashing, with independently owned state identity.

### Outer-source streamed HB recovery

The streamed recovery branch evaluates the admitted linear current response to
the actual outer HB dissipative source. It checks its electric field against the
outer DC field using the original HB threshold, retains live material inputs,
and constructs the exterior. Its recovery receipt is sealed before the state
identity is computed. It does not recompute an inner nonlinear root or assert
new nonlinear convergence. Recovery returns the independent binding to the HB
state; conditional snapshot fields retain it for the forthcoming public policy
integration. The public streamed dual policy still rejects until restore and
policy/history/cutoff validation are fully connected.

Independent audit found an internal P2: total volume current used the frozen
admitted conductivity, losing the direct E*d(sigma) material gradient. The
shared dual_ohmic_current helper now selects live material_coefficients for
streamed closures; the dense behavior is unchanged. The recovery test compares
both electric and magnetic material gradients of ohmic+source against the
native recovered electric_current and checks a finite nonzero source gradient.
The collected complete-preparation fixture now restores trainable conductivities
so this material branch runs during future regression.

GPU1 session23505 terminal exit0, /tmp/gsp11-streamed-hb-recovery-v1.log,
reused the saved complete preparation for outer-source recovery, forbidden inner
nonlinear solve, original-threshold rejection and sealed receipt tampering.
Session27172 terminal exit1, /tmp/gsp11-streamed-hb-recovery-v2.log: new material
test incorrectly demanded bitwise equality of sigma*((J-S)/sigma)+S and J;
maximum difference5.551115123125783e-17. Failure preserved. Value comparison now
uses8eps*max|S|; material gradient tolerance remains rtol1e-12/atol1e-14.
Session48944 terminal exit0, /tmp/gsp11-streamed-hb-recovery-v3.log: complete
helper including both material gradients passed. No root or admission rerun.
Session62427 terminal exit0, /tmp/gsp11-streamed-hb-recovery-dense-v1.log:
saved public dense-DC/compressed-AC result reopens and re-encodes with exact
payload digest, internal/external queries and solves forbidden.

These checks exercise internal recovery on a constructed DC state, not a new
public nonlinear HB root or complete streamed public archive. Geometry, physical
and fast-work qualification remain unchanged. Next connect public preparation,
canonical policy validation, outer snapshot restore and cutoff/Bench ownership;
then perform a genuine public root/save/load integration test once at the merge
point. Do not repeat saved preparations or historical full regressions first.

### Public streamed dual integration under validation

The pending stage connects dc_tangent_rule and dc_tangent_column_batch to dual
preparation and prefix reuse. Restore checks the selected native policy, an
independently owned state binding, the sealed owning recovery threshold and the
historical nonlinear/current receipt pair. The validator checks complete 3N and
6N inverse bounds, canonical policies, phases, finite original residuals and the
native nonlinear threshold derived from the ordered initial residual history.
The producer reuses that initial residual for threshold and first history entry.
Bench cutoff history dispatches dual records to the same validator rather than
interpreting them as per-body electric face solves.

First actual public integration, GPU1 session2073, terminated exit1 after
116.49s: /tmp/gsp11-streamed-dual-public-v1.{log,xml}. At the initial outer
residual, an approximately2e-22 DC source let absolute GMRES tolerance suppress
the current response, and the unchanged physical coupled-current gate rejected.
No solved root/archive was produced. The shared fixed tangent inverse now
normalizes each small correction RHS by min(maxabs,1), with zero mapped to1,
and rescales its solution. Original RHS thresholds, all physical gates and
iteration/refinement budgets remain unchanged. Attempt logs record normalized
units and scales. Independent audit confirmed no new P1/P2 in this change.

GPU1 session80748 terminated exit0, /tmp/gsp11-tiny-dual-v1.log. Reused saved
complete preparation: native electric/magnetic fields and currents retain
homogeneity at source scales1e-22 and1e-30 and pass the original coupled-current
gate. Equivalent coverage is attached to the complete-preparation bridge helper.
The earlier audit finding that historical nonlinear threshold was merely
self-reported is fixed by exact reconstruction from native tolerances and initial
residual; malformed history rejects. No physical threshold was relaxed.

Second public integration session11475 remains live as of this entry:
/tmp/gsp11-streamed-dual-public-v2.{log,xml}. It includes the real nonlinear
public root/archive path and existing shared HB inverse gradient/failure tests.
Do not restart a live or temporarily unobservable session. On completion, use
its saved result for /tmp/gsp_streamed_dual_bench.py (prepared but not yet run).
That archive check is distinct from a real H2-to-H3 adaptive dual extension;
the extension can reuse this saved H2 state and its preparation. Public success,
stage commits, merge regression and qualification remain unproven pending
actual evidence. The next checkpoint must replace this live-status note with
the observed terminal result, without deleting either failed run's evidence.

### Public H=2 integration and cross-repository archive evidence

Session11475 is TERMINAL exit0:3passed3882.30s,
/tmp/gsp11-streamed-dual-public-v2.{log,xml}. The actual public nonlinear dual
H=2 test took3874.551s; it uses nonzero chi2/chi3/TPA and both original pump
scales. Exact restore forbids fixed inverse solves, dual closure construction
and direct solves, compares inside/outside field tensors bitwise, preserves the
complete payload digest, and rejects independent binding, policy, historical
bound/threshold and source mutations. Two existing fixed HB inverse gradient/
failure tests pass in the same run. This is one real nonlinear public state,
not a physical reference, multi-body qualification or scaling demonstration.
All-column nested solve time remains a material limitation.

Retained Result:
/tmp/gsp11-streamed-dual-public-v2/test_public_streamed_dual_root0/result.pt
SHA256 c137410f6976ad490a0babd67bcf432e168add2794e333223a1eba210919315c.

GPU1 session32515 TERMINAL exit0, /tmp/gsp11-streamed-dual-bench-v1.log,
reused that result for Bench construction-bound exact write/read, policy-change
rejection and actual historical threshold/history/column tampering, with inverse
and direct solves forbidden. Archive /tmp/gsp11-streamed-dual-bench-v1;
manifest SHA256 d073e5e39f10563249097aa8bc124099fc50cb14898a4cdd3b86793c95444e5d.
No repeated nonlinear root, admission or preparation. This plain archive check
does not cover a genuine adaptive dual cutoff history.

The next native H=2-to-H=3 extension is now running in GPU1 session92860,
/tmp/gsp11-streamed-dual-cutoff-v1.log, using the saved H=2 root and exact prefix.
Its harness saves added H=3 preparation before the nonlinear solve, then records
native comparison outcomes and checks Bench historical dual policy mutations.
Do not restart it on missing log output or observation timeout. Stage commits
remain pending that acceptance and independent audit; no physical/truncation or
fast-work qualification follows from the H=2 software integration success.

Independent evidence audit matched XML/log and inspected the saved payload.
The actual AC backend is dense: H=2 denotes harmonic count, not H2 compression.
Both prior P2 fixes confirmed; no new confirmed P1/P2. No frozen pre-run source
manifest was captured, so this is targeted stage evidence rather than a fixed
HEAD full regression. Public material gradients, compressed-AC composition,
actual adaptive extension and final physical/scaling acceptance remain separate.

The follow-up decode-only adaptive archive harness is prepared at
/tmp/gsp_streamed_dual_cutoff_archive.py (syntax checked, not executed).
After session92860 succeeds, reuse its harmonic.pt and receipt.json to run this
harness on GPU1 with the same Maxwell/tests/Bench import paths. It constructs a
Result from the actual extended native state and recorded solver history, then
checks complete Bench manifest write/read, exact payload identity, and interior/
exterior DC fields with inverse/direct solves and dual closure construction
forbidden. It writes /tmp/gsp11-streamed-dual-cutoff-archive-v1 once. Do not run
it before the extension's successful terminal exit, or rerun the H=2 root.
Independent review found no new confirmed P1/P2 in the adaptive history dispatch
or live extension harness; the original runner's native history check alone
does not cover a full adaptive Result/Bench archive roundtrip.

### Streamed dual adaptive integration terminal acceptance

Session92860 is TERMINAL exit0, /tmp/gsp11-streamed-dual-cutoff-v1.log.
The actual H=2-to-H=3 root converged, reusing the identical static and first two
harmonic preparations. Native/Bench history validation and threshold, batch and
bound mutations passed. Original cutoff remains FAILED: harmonic_budget_exhausted,
added harmonic RMS4.8931783246590025e-5 >1e-8 V/m. Retained DC/H2 differences
6.682061060701131e-10/1.7134685162270747e-8 exceed their original thresholds
5.86389391676398e-10/1.388098344501812e-8 V/m; H1 comparison passes. No truncation
or physical qualification follows from the engineering acceptance.

Native harmonic.pt SHA256:
c6c9f51bb745e261b1ea0f8f6bfe74b9958a7082d7c53e38681517b598a90084.
Native receipt.json SHA256:
2cd770882aadf06016b0fafdc718b7fcdc7c39615f5090351cbd1590bbc3d8fb.
Both are under /tmp/gsp11-streamed-dual-cutoff-v1.

Session43403 is TERMINAL exit0,
/tmp/gsp11-streamed-dual-cutoff-archive-v1.log. The previously reviewed harness
restored the actual saved H=3 native state, constructed its Result using original
continuation plus actual extension receipts, then wrote/read the complete Bench
archive with inverse/direct solves and dual closure construction forbidden.
Full native payload digest, cutoff records and interior/exterior DC tensors
match exactly. Archive /tmp/gsp11-streamed-dual-cutoff-archive-v1 has manifest
SHA2562e097a8b49ef722b6d6cd1b472027eb718cfd5495bd22cf770e25716b3a4dc30.
This is solved-state archive replay, not a second runtime.run/monitor test or an
independent physical reference. Both original failed cutoff flags are retained.
No live jobs remain from this stage; no duplicate preparation or roots were run.

Independent read-only terminal audit verified the native and archive hashes,
history/final-solver equality, original cutoff arithmetic and unqualified flags.
No confirmed P1/P2 remains for this engineering stage. The audit explicitly
preserves failed cutoff, dense-AC scope, incomplete adaptive runtime.run/monitor
coverage and the requirement for later fixed-HEAD merge/final regression.
