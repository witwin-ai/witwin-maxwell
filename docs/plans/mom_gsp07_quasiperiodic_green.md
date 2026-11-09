# GSP-07 quasiperiodic Green implementation

Status: OPEN. Based on GSP-06 experimental checkpoint 8ae92c9 in the independent
gsp07-periodic worktree. No periodic public solve or production qualification.
GSP-01 physical failures and all final acceptance thresholds remain unchanged.

## Frozen conventions and implemented foundation

`LatticeSpec` stores owned real CUDA row vectors A[D,3] in meters, D=1,2,3,
and a real Bloch vector beta in the periodic span, in rad/m. Native QR produces
reciprocal rows B with A B^T = 2 pi I without squaring the condition number.
Nonfinite, dependent, numerically unresolved, mismatched or trainable geometry
rejects. Tensor identity/version guards protect derived state. Input mutations
do not modify owned state; owned-state mutations fail closed.

For exp(-i omega t), outgoing free-space g=exp(+ik|r|)/(4 pi |r|), use
sum_R exp(+i beta.R) g(r-R). Thus the infinite sum satisfies
G(r+R)=exp(+i beta.R)G(r). Incoming wavevectors must match beta modulo a
reciprocal lattice vector on the periodic axes; open-axis components do not
enter this condition. The lattice receipt records vectors, beta and convention.

`LossyImageReference` is an internal independent reference, not the final Ewald
implementation. It requires Im(k)>0 and fixed integer image shells. Integer
images are generated in bounded batches rather than allocating a global 3D
image grid. G and gradient retain native displacement derivatives. Excluding
the zero image exposes the regular local remainder; coincidences with retained
images reject. Finite shell receipts explicitly deny truncation certification
and physical qualification. Shell convergence must be checked by callers.

Targeted CUDA evidence: `/tmp/gsp07-periodic-foundation-v2.log`, 7 passed in
3.82 seconds. Includes skew 1D/2D/3D reciprocal/Bloch tests, positive-phase shift
convergence for G and gradient, analytic/native-gradient equality, exact zero-
image separation, invalid geometry, singular guards and changed-batch agreement.
These checks do not constitute Ewald, diffraction or periodic scattering tests.

## Required remaining implementation and acceptance

- Native CUDA real/reciprocal Ewald sums for 1D/2D/3D, stable complex special
  functions where needed, G/gradient and singular separation, independent
  real/spectral truncation evidence against the lossy direct-image reference.
- Exterior-only periodic assembly and near-field decoding; homogeneous body
  interiors stay free-space. Multi-body ownership and non-seam geometry guards.
- Floquet/Rayleigh outputs, propagating and evanescent branches, grazing guards,
  outgoing open-axis sign and unit-cell power. Reject isolated far-field/RCS.
- Versioned signatures and one Bloch sector per operator/multiple RHS;
  Scene/Simulation/Result configuration and exact persistence.
- Actual quasiperiodic dyadic kernel, adjoint and decoder; integrate unmerged
  GSP-05 tensor implementation rather than disguising scalar tests as dyadic.
- Dimension/kernel-specific anomaly support, limiting absorption convergence,
  genuine pole/BIC rejection; 3D resolvent support and finite-limit acceptance.
- Independent periodic references, reciprocity where applicable, power and
  truncation acceptance; Bench configuration/DR/exact storage and phase audit.

Do not rerun completed GSP-06 regression merely for these isolated additions.
Use targeted checks during implementation and consolidated regression/audit at
meaningful merge points. Commit the coherent major stage independently.


## Scalar Ewald implementation checkpoint

`EwaldHelmholtzKernel` now implements D=1,2,3 using the same lattice and phase
contract. No public operator admission yet. It accepts passive scalar k and
rejects unresolved spectral poles/grazing thresholds. The reciprocal cutoff
must cover every potentially resonant integer mode according to the smallest
reciprocal singular value; a too-small cutoff rejects before evaluation.

The real-space split uses the native heat representation
(2*pi**1.5)^-1 integral_eta^infinity exp(-r^2*t^2+k^2/(4*t^2)) dt.
The Coulomb tail and first three k-dependent heat moments are analytic; the
remaining entire correction uses GPU Gauss quadrature with a stable exponential
remainder series. This avoids complex-special-function CPU fallbacks. For D=3,
the reciprocal term is exp(i*q.r)*exp(-(q^2-k^2)/(4*eta^2))/(V*(q^2-k^2)).
For D<3, the low-heat spectral integral has t^-D weight, normalized by
1/(2*pi**((3-D)/2)*cell_measure). Each mode follows a decaying radial complex
leg from zero and an arc back to eta. The sheet follows the sign of Re(k),
with the passive outgoing limit. Numerical points are wrapped to the periodic
cell and multiplied by the prescribed Bloch translation phase.

Derivation references (conventions cross-checked; implementation is native):
- Campione and Capolino, 2012, scalar/3D Ewald representation and exp(-i omega t):
  https://doi.org/10.1029/2012RS005031
- Moroz, 2006, partially periodic representations and Ewald contour deformation:
  https://arxiv.org/abs/math-ph/0602021

`estimate()` deepens real shells, reciprocal shells and heat quadrature
independently, checking both G and gradient. It raises with the actual receipt
on failure. These are successive numerical estimates, not certified bounds.
Fixed `evaluate()` alone makes no convergence claim. Large-open-distance contour
conditioning and a stable zero-image regular-part API still need work before
quadrature integration; ordinary coincident nodes currently reject.

Evidence:
- 3D lossy direct-image reference, two eta values, native gradient, Bloch shift,
  poles and negative cutoff-budget case: 2 passed, `/tmp/gsp07-ewald3d-v2.log`.
- A near-origin probe exposed the original raw heat quadrature defect:
  G differences 2.79e-6/1.02e-5 and gradient differences 8.81e-4/0.134 between
  orders 64 and 256. After analytic heat moments, the same probe reports zero
  differences at the computed precision. Before/after logs are
  `/tmp/gsp07-near-ewald-before.log` and `/tmp/gsp07-near-ewald-after.log`.
  A separate near-origin comparison against 12-shell lossy images also passes.
- Initial finite-absorption test had a final error 0.0026918 at loss=0.01,
  exceeding the fixed diagnostic 0.002 threshold. The sequence was extended to
  loss=0.003; the threshold was unchanged. This is a single off-pole reference,
  not a general limiting-absorption qualification.
- D=1/2 contours agree with 60-shell lossy direct images for k=2+3i,
  k=2+0.5i and k=-2+0.5i, including gradients.
- Independent lossless Rayleigh (D=2) and Bessel/Hankel (D=1) spectral series
  agree on both sides of the open axis. Reciprocity and the differential
  Helmholtz equation pass away from singularities; exact poles still reject.
- Current complete periodic-kernel test block: 19 passed in 10.23 seconds,
  `/tmp/gsp07-ewald-foundation-v3.log`. No unrelated whole-suite rerun.

Next implement stable local regular-part evaluation and integrate the periodic
exterior with free-space body interiors. Diffraction normalization, actual
periodic scattering/power, tensor dyadics, anomalous finite limits, Bench and
public persistence remain required. This checkpoint does not close GSP-07.

## Primary-image regular part

`regular_part()` now evaluates G_periodic-G_free without wrapping the removed
image to a nearest neighbor. Nonzero real images and the reciprocal sum are
combined with the smooth zero-image heat remainder F. Near zero, F=sum c_m r^(2m)
uses (Delta+k^2)F=S exp(-eta^2 r^2), S=eta^3 exp(k^2/(4eta^2))/pi^1.5,
and c_(m+1)=(S*(-eta^2)^m/m! - k^2*c_m)/(2*(m+1)*(2*m+3)).
The finite c_0 is obtained from the nonsingular heat integral; 32 terms are used
only where both eta*r and |k|*r are at most one. Outside this disk the primary
real contribution and free-space value are well scaled for direct subtraction.
Other retained lattice-image coincidences reject, as they remain singular.

`estimate(..., regular=True)` independently checks all three refinement axes
for this quantity and identifies it in its receipt. CUDA evidence:
`/tmp/gsp07-regular-part-v1.log`: 3 passed in 5.18 seconds, covering all dimensions,
origin/1e-10-scale/general query points, 12-shell independent nonzero-image sums,
native value-to-gradient derivatives and non-primary coincidence rejection.
No public periodic assembly or scattering acceptance is inferred from this.

## Exterior-only periodic assembly

`periodic_operator.assemble_periodic_system()` now assembles the free-space
system once, then adds the regular periodic exterior contribution to every
self/cross-body block. No interior material is used to construct this correction.
Primary singular terms retain the existing joint Duffy rules; the regular term
uses product triangle quadrature, including matching primary quadrature nodes.
The centered-cell guard checks all body vertices in fractional (-1/2,1/2)
coordinates along periodic axes before free-space matrix allocation/assembly.

The free-space magnetic self-panel zero identity does not hold for the periodic
regular kernel. `_pair_integrals` now has an explicit `zero_planar_self` flag,
defaulting to existing free-space behavior; periodic regular assembly disables
it. A real-kernel test demonstrates a nonzero same-panel magnetic contribution
that the old assumption would suppress. This is not a mock kernel test.

Evidence:
- `/tmp/gsp07-periodic-assembly-v1.log`: 5 passed in 8.02 s. Real two-body
  corrections unchanged when all interior epsilon/mu samples change; all
  self/cross electric blocks have nonzero corrections. Seam rejection occurs
  before assembly. Includes the three regular-part dimension controls.
- `/tmp/gsp07-periodic-self-k-v1.log`: 4 passed in 5.17 s. Actual periodic
  magnetic self-panel control plus existing free-space kernel injection tests.

Fixed product quadrature and Ewald cutoffs are explicitly unqualified in the
assembly receipt. Public runtime still rejects periodic kernels. Next attach
proper cutoff/quadrature budgets, periodic incident/RHS consistency and field
recovery before public admission. Far-field/RCS must remain rejected in favor
of validated diffraction/mode outputs. No stage completion or physical pass is
inferred from the assembly ownership tests.

## Assembly refinement budgets and incident sector

`PeriodicAssemblyCheck` optionally checks four independent refinements of the
regular exterior matrix: surface product order, real-image shells, reciprocal
shells and heat quadrature order. Each doubles alone. A free-space primary
matrix is assembled once, then reused. The metric is the Frobenius norm after
impedance balancing and left/right inverse RWG L2 Cholesky transformations;
it is a documented proxy, not a certified trace-operator bound. Thresholds
apply to the regular correction itself, so a large primary matrix cannot hide
its error. Failure raises `PeriodicAssemblyConvergenceError` with all checks.
The primary singular quadrature still has its own existing obligations.

Actual single-body failure evidence was retained:
- order2/reciprocal4: surface difference3.65537e-8, reciprocal difference2.05989e-9,
  each above approximately1.595e-12; real-image and heat checks passed.
- order3/reciprocal4: surface difference5.00068e-11 and reciprocal2.07735e-9,
  still failing unchanged thresholds. `/tmp/gsp07-assembly-check-v2.log`.
- order4/reciprocal8: surface difference3.20498e-14, real-image0, reciprocal0,
  heat2.46966e-17; all below1.59521e-12. Relative tolerance1e-7 and absolute1e-12
  were not relaxed. `/tmp/gsp07-assembly-check-v3.log`, 2 passed in54.61s,
  including the new incident-sector test. The low-order case remains a
  negative test; the refined case is separate to avoid repeated failed probes.

`validate_periodic_sources` accepts plane waves only, checks every source
against the same real Bloch vector modulo reciprocal translations, and rejects
nonzero attenuation along periodic axes. Open-axis attenuation is allowed.
These guards are not yet wired into a public periodic runtime. Public source,
field, diffraction and persistence integration remains OPEN.

## Periodic field and incident lifting integration

`periodic_fields.decode_periodic` now composes the free-space primary decoder
with explicit smooth exterior RWG image potentials. Local body decoders must
remain free-space; interior fields never call the periodic correction. Queries
are wrapped to the primary cell for geometric ownership and multiplied by the
Bloch translation phase. Primary one-sided jumps remain in the existing decoder.
The function validates kernel spectral support even for interior queries.

Adaptive surface integration compares complete E/H with physical impedance
weighting per point. It records actual differences and thresholds, and raises
`PeriodicFieldConvergenceError` on budget exhaustion. This receipt explicitly
leaves field Ewald truncation unverified; a surface-order pass is not a kernel
cutoff or physical certificate. This path is not yet public runtime admission.

`build_periodic_rhs` validates the same real Bloch sector, then assembles the
existing exact analytic-carrier local material-difference RHS in global
all-J/all-M order. The regular periodic image contribution of each exact closed
incident Cauchy trace vanishes; it must not be re-added as artificial forcing.
A direct carrier-integral test supports this identity on the actual two-body
fixture: weighted E/H residual falls from1.38150e-6 to1.65847e-18 V/m as order
increases from2 to6, at both exterior and interior primary-cell queries.

Targeted evidence (not production reference qualification):
- `/tmp/gsp07-fields-v2.log`:2 passed in5.69s. Nonzero Bloch translation phase,
  actual current finite-perturbation/native-gradient agreement, bitwise free
  interior recovery, independent12-shell lossy full-image surface potentials,
  complete-field refinement and actual insufficient-order rejection.
- `/tmp/gsp07-periodic-carrier-v1.log`:1 passed in4.08s, analytic-carrier regular
  image cancellation described above.
- `/tmp/gsp07-periodic-rhs-v1.log`:1 passed in4.55s, exact local lifting agreement
  and wrong-sector rejection before integration.

Next: solve a coherent periodic matrix/RHS and query its fields; close field
cutoff and boundary/reciprocity controls, then diffraction/unit-cell power and
public configuration/signature/persistence. Existing scalar implementation
checks do not replace tensor-periodic kernels or anomaly qualification.

## Actual periodic solve and Rayleigh output checkpoint

`tests/mom/test_periodic_solve.py` now integrates the actual periodic matrix,
analytic lifted RHS, existing CUDA dense solver, regional field recovery and
translated-cell queries. True relative residuals are2.68880e-16 (1D) and
4.76216e-16 (2D) on the single-tetrahedron scalar fixture. Matching interior
material gives exactly zero scattered forcing and exact incident exterior
fields. `/tmp/gsp07-actual-periodic-solve-v1.log`:2 passed in7.07s. This is
coarse algebraic integration, not physical scattering convergence.

`periodic_diffraction.rayleigh_orders` integrates scattered RWG currents against
exp(-i p.r_source), where p=q+side*gamma*n and gamma takes the passive outgoing
root. The source phase includes propagation to the requested exterior plane
before summation, avoiding separately growing/decaying evanescent exponentials.
It returns E/H modal amplitudes at that plane, wavevectors, integer orders,
outward power per primitive cell and explicit unqualified flags. Scattered
amplitudes do not include incident waves and are not transmission efficiency.
The plane must be strictly outside all bodies. Near-grazing spectral gaps reject
with a precision floor; they are not silently regularized.

Evidence:
- Initial order3 Rayleigh/near-field comparison failed on both sides, maximum
  errors1.18e-8/2.59e-8 versus original1e-7 relative/1e-9 absolute criteria.
  `/tmp/gsp07-rayleigh-v1.log` retains this failure. Both integrals were raised
  to order6; no comparison threshold was relaxed.
- `/tmp/gsp07-rayleigh-v2.log`:2 passed in4.92s, positive and negative open sides,
  actual periodic-near-field reconstruction and transverse Maxwell amplitudes.
- `/tmp/gsp07-rayleigh-power-v1.log`:2 passed in4.84s, additionally verifies that
  the modal power sum matches the primitive-cell plane Poynting integral and
  rejects an exact grazing order despite floating-point reciprocal arithmetic.

This does not yet prove incident/reflected/transmitted/global power balance,
independent physical reference agreement, order truncation convergence, tensor
periodic support, anomaly limits, or public Result integration. Those remain
required before GSP-07 completion.

### Coherent total-plane Rayleigh flux

Added internal `rayleigh_total`, preserving the outgoing scattered record while
combining incident E/H by tangential order at the requested plane. Opposing
normal directions can share one order; total fields therefore carry no single
outgoing wavevector. Missing incident orders reject. Interference uses direct
cross terms instead of subtracting incident-dominated powers. Targeted CUDA
validation `/tmp/gsp07-rayleigh-total-v1.log`: 2 passed in 4.95 s, covering both
planes, phased co- and counter-propagating waves, complete-field reconstruction,
cell-plane flux integration, and omitted-order rejection. No global energy
balance, absorbed-power or production qualification is claimed.

### Actual solved-current power diagnostic

`rayleigh_power_balance` combines both exterior planes and reports scattering,
extinction from direct interference, flux-inferred absorption and net incident/
total flux. Flux-inferred absorption is not an independent material-work check.
The lossless defect is normalized by max(abs(scattering), abs(extinction)), not
incident power; apply it as a zero-absorption criterion only to lossless bodies.
The four-face tetrahedron at 300 MHz gives scattering 8.747287396741658e-11 W,
extinction -4.973190353344912e-10 W, inferred absorption
-5.847919093019078e-10 W and relative defect 1.175888851526834: physical FAIL.
Its algebraic residual is 4.762161177758644e-16. Orders [-3,3]^2 to [-6,6]^2 and
field integration 6 to 12 agree at rtol 1e-8, atol 1e-15 W. Transparent controls
remain exactly zero. `/tmp/gsp07-rayleigh-balance-v1.log`: one integration test
passed in 5.84 s; this verifies diagnostics, not the failed physical criterion.

The first mesh diagnostic completed at 16 faces: scattering
9.235530715629833e-11 W, extinction 9.200270779072355e-10 W and lossless defect
0.8996167511000037 (FAIL); residual 5.406149309162947e-16. Its record is retained
in `/tmp/gsp07-power-refinement-v1.pt`. The incomplete 64-face run was explicitly
interrupted after identifying fixed 16-query kernel batching overhead. Ewald
query batches are now configurable (default 256); three dimension-wise value,
analytic-gradient and native-autograd tests pass in
`/tmp/gsp07-ewald-query-batch-v1.log`. Only the unfinished 64-face level resumes
with unchanged quadrature/cutoffs, under `/tmp/gsp07-power-refinement-v2.log`.

### Scalar implementation checkpoint integration and audit

The resumed 64-face diagnostic completed with algebraic residual
1.696250680273516e-15, scattering 9.175952437278452e-11 W, extinction
1.8008348297148676e-10 W, inferred absorption 8.832395859870225e-11 W and lossless
relative defect 0.4904611857861883 (FAIL). Both mesh levels remain in
`/tmp/gsp07-power-refinement-v2.pt`; terminal log is the matching `.log`.
No acceptance threshold was relaxed and no larger mesh was blindly launched.

Stage integration across all five periodic test modules:
`/tmp/gsp07-scalar-checkpoint-v1.log`, 41 passed in 33.04 s. Independent read-only
scalar checkpoint audit found a real defect in distant `regular_part` queries:
finite origin-centered image truncation could omit the nearest translated
singularity and its neighborhood. Fixed using wrapped periodic evaluation minus
the original free-space primary outside the primary fractional neighborhood;
near-origin stable series is preserved. Targeted follow-up
`/tmp/gsp07-regular-audit-fix-v1.log`: 9 passed in 6.10 s, including skew lattice,
nonzero Bloch, distant-image singularities, neighborhood values and gradients
in all three dimensions. Independent re-audit confirmed the finding closed.
The audit also identified missing skew/nonzero-incident-order total-flux coverage;
this remains a validation obligation, not an established implementation defect.

This checkpoint is an experimental scalar implementation commit, not closure of
GSP-07. Dyadic kernels, supported anomaly limits, complete 1D/3D observables,
public runtime/persistence, independent physical references and passing power
balance remain required under the unchanged full plan.

### Modal estimates and oblique-cell validation

Added `estimate_rayleigh_total`: independently doubles surface quadrature and
square diffraction-order range, zero-padding omitted modes. Scattered E/eta-H
and scattered/interference power use separate absolute tolerances and a shared
relative tolerance. Incident-dominated total norms cannot conceal weak-scattering
error. The initial square must cover all potentially propagating orders by a
reciprocal singular-value bound. Failed checks raise with both quantitative
receipts; successful estimates do not certify a tail bound or physical accuracy.

`/tmp/gsp07-rayleigh-estimates-v1.log` retains one passing estimator test and two
then-failing skew controls. The estimator's positive far-plane control passes;
its near-plane negative control has order-tail field difference .08330631037
versus threshold 1.160896662e-8, while power difference 4.094361022e-23 W passes.
Thus a power-only cutoff check would miss a large evanescent field tail.

Skew-cell controls use co- and counter-propagating oblique sources, nonzero Bloch
vector, and incident order (1,0), on both exterior planes. Initial heat order64
failed field reconstruction even after deepening reciprocal/modal ranges.
Independent scalar Rayleigh probe `/tmp/gsp07-skew-kernel-probe-v1.log` resolves
the source: heat64/128/256 gradient differences are 7.062466544e-6,
2.295834095e-10, 2.232466372e-14. Heat128 with reciprocal/modal shells12 passes
the unchanged rtol1e-7/atol1e-9 field thresholds. Both plane flux integration and
missing-incident-order rejection also pass: `/tmp/gsp07-skew-rayleigh-v2.log`,
2 passed in 4.64 s. Original failures are retained, not reclassified as passes.
This demonstrates why field Ewald error estimation remains a required next step.

### Independent Ewald estimates in field recovery

`decode_periodic(..., ewald_check=True)` now doubles real images, reciprocal
modes and heat quadrature independently at the final surface integration order.
It reuses the free-space primary field and recomputes only regular image
potentials. Each query uses an impedance-balanced E/H difference; the tolerance
scale is the smaller of complete-field and regular-correction norms so neither
incident dominance nor physical cancellation conceals error. Failures retain
all three checks through `PeriodicFieldConvergenceError`. Surface and Ewald
convergence remain distinct, and fixed surface quadrature remains unestimated
even if Ewald checks pass. Interior/incident/empty queries record nonapplicability.

Known skew case: heat64→128 fails with 2.615970329e-7 versus threshold
9.072693625e-9; heat128→256 passes with 8.503891780e-12. Real/reciprocal checks
pass independently. Native coefficient gradients are retained. Combined adaptive
surface6→12 plus Ewald checks pass without relaxing the original tolerances.
`/tmp/gsp07-field-ewald-v1.log`: new targeted control passed (4.50 s).
`/tmp/gsp07-field-ewald-v2.log`: affected field module, 5 passed (11.87 s),
including existing translated fields, direct-image reference, interior ownership,
carrier cancellation and RHS checks. These successive estimates do not close
physical power/reference qualification or certify global truncation bounds.

### Three-dimensional exterior-current Bloch spectra

Added `bloch_spectral_fields` for off-spectrum 3D kernels. Actual RWG electric
and magnetic currents are integrated against exp(-i q.r), normalized by cell
volume, then mapped by the transverse Maxwell resolvent with denominator
q.q-k^2. The zero mode is retained. Global kernel pole scanning is performed
before returning any requested subset; omitting a resonant mode does not admit
an undefined forced resolvent. No incident source is included in this diagnostic.
The record explicitly labels homogeneous-exterior equivalent-current potentials,
which are not the piecewise physical fields inside material bodies. No isolated
far field or diffraction-plane power is attached to these 3D modes.

A bounded face/mode Fourier integrator is shared with 2D Rayleigh output. Its
custom phase preserves stable combined evanescent propagation. Validation
`/tmp/gsp07-bloch-spectra-v1.log`: 9 passed in 6.89 s, covering source Maxwell
identities (i q×E-i omega mu H=-M and i q×H+i omega epsilon E=J), independent
zero-mode centroid moments, native current gradients versus differences, and
unrequested-pole rejection, plus the affected 2D modal suite. These tests do not
establish physical periodic scattering qualification or interior reconstruction.
Complete 3D physical fields/flux, supported spectral-limit quantities, public
runtime and persistence remain required.

### One-dimensional cylindrical outgoing observables

Added `cylindrical_orders`, using the shared RWG Fourier integrator at wavevector
q_axial+gamma*n_radial. Amplitudes multiply exp(i q_axial.r+i gamma*rho)/sqrt(rho).
For lossless exterior media, propagating orders report power per radian per
primitive axial period; evanescent coefficients retain their decay and have zero
power at infinity. Directions must be unit vectors perpendicular to the axis.
Grazing modes reject, and angular/order/surface convergence is not implied.
The outgoing prefactor follows i H0^(1)/(4 L) and its large-argument form from
[NIST DLMF 10.2.5](https://dlmf.nist.gov/10.2.E5); the negative-index propagating
branch uses the conjugate outgoing scalar prefactor. Lossy exterior power is
not silently interpreted as radiation at infinity.

Independent Hankel G/gradient surface-potential integration at radii200/400
approaches the amplitude with relative errors .0027971123/.0013985599 for
positive index (`/tmp/gsp07-cylindrical-v1.log`,1passed4.13s). Double-negative
Drude electric/magnetic sampling gives .0027966231/.0013983085
(`/tmp/gsp07-cylindrical-negative-v2.log`,1passed4.02s). Tests also check
transversality, Maxwell E/H relation, propagating power normalization, evanescent
zero radiation, current gradients and grazing/direction guards. The first
negative-index fixture used forbidden negative constant Material parameters and
failed constructor validation; retained in `...negative-v1.log`, replaced using
the existing public lossless Drude contract rather than changing material rules.

### Observable/convergence checkpoint integration

`/tmp/gsp07-observables-checkpoint-v1.log`: all seven periodic test modules,
52 passed in 23.84 s. Independent read-only audit reviewed 1D outgoing branches,
3D distributional spectral semantics, Rayleigh tail/power estimates and field
Ewald checks, with no new confirmed correctness defect. It retained explicit
coverage gaps: independent nonzero axial propagating-order/Bloch cylindrical
references and convergence of reconstructed truncated 3D spectral fields.
This is an experimental implementation checkpoint, not completion of GSP-07 or
physical qualification. Existing lossless power failure and all dyadic, anomaly,
public runtime/storage and final cross-repository acceptance obligations remain.

### Explicit 3D spectral-point quantities

Added `periodic_limits.projected_bloch_green` for the scalar 3D Bloch Laplacian.
The caller selects an eigenvalue by reciprocal order; the entire numerically
resolved degenerate eigenspace is collected with a recorded 128-epsilon relative
resolution. Near but unresolved distinct eigenvalues and incomplete reciprocal
searches reject. The support domain is explicit in the returned record.
No ordinary forced-solution admission or automatic regularization is introduced.

For spectral parameter z=k² and R(z)=(A-z)^-1, the residue is -P and the reduced
kernel on the complement obeys (Delta+z)G=-delta+P. The sign convention agrees
with the resolvent/projector definition discussed in
[Faris, Methods of Applied Mathematics](https://math.arizona.edu/~faris/methlec.pdf).
The Ewald correction is derived directly from its Fourier split: the real part
contains (1-exp(-D/(4 eta²)))/D, whose resonant limit is 1/(4 eta²). Thus removing
only reciprocal singular modes is insufficient; P/(4 eta²) and its gradient
must also be subtracted. The implementation does both, retains Bloch wrapping
and native point gradients, and exposes the projector and residue separately.

`/tmp/gsp07-projected-resolvent-v1.log`:3 passed7.02s. Covers nonzero-Bloch single
mode and cubic six-fold degeneracy, ordinary pole rejection, PDE, native gradient,
Bloch shift, eta3→2 invariance and two independent approaches z=lambda+i delta
and z=lambda+(1+i)delta. After explicit pole subtraction the final combined G/
gradient errors at delta=.003 are 7.7801e-5/1.1003e-4 (single mode) and
1.0777e-4/1.5243e-4 (six-fold), decreasing linearly from delta=.03/.01.
`/tmp/gsp07-projected-ambiguity-v1.log` adds near-degenerate rejection. These are
explicit projected quantities, not proof of a finite unprojected limiting-
absorption forced solution, Maxwell dyadic poles, or physical body scattering.
1D/2D anomaly treatment and general dyadic periodic support remain required.

### Actual reciprocal-tensor periodic spectral foundation

Ported the existing GSP05 owned `TensorMedium` contract and Fourier primitives
(`cross_matrix`, `maxwell_symbol`, `maxwell_resolvent`, `current_green_spectrum`)
into the corresponding modules on this branch. The GSP05 worktree is unchanged;
its full spatial dyadic implementation/public material compiler is not silently
merged. This establishes the shared physical symbol needed for periodic work.

`QuasiperiodicDyadicGreenKernel` now computes actual 6×6 tensor Maxwell mode
responses for 3D Bravais lattices, normalized by cell volume. It supports explicit
forward/complex-adjoint maps and bounded finite heat-filtered Fourier sums.
Requested singular or unresolved symbols reject without artificial damping.
The heat filter is explicit, includes contact distributions, and is not called
the unfiltered spatial kernel or its primary-image regular part. Global
resolvent verification is explicitly false: finite mode sampling does not prove
that an indefinite tensor has no omitted real-wavevector poles. Ordinary periodic
assembly still requires its existing admitted scalar kernel type.

`/tmp/gsp07-dyadic-spectrum-v1.log`:3passed3.99s, with noncommuting non-diagonal
reciprocal electric/magnetic tensors, Maxwell source residuals, native material
finite-difference gradient, heat-sum Bloch shift/reciprocity/point gradient,
requested-pole rejection and owned-material mutation guard.
`/tmp/gsp07-dyadic-adjoint-v1.log`:forward/adjoint inner-product and native source
VJP agreement. These are genuine tensor responses, not scalar-kernel combination
tests. They do not close spatial singular separation, truncation convergence,
1D/2D tensor radiation, periodic PMCHWT/field coupling, anomaly support, public
runtime/storage or physical-reference qualification.

### Two-dimensional tensor outgoing spatial sums

Ported GSP05's checked plane-mode/residue compilation and native CUDA 4x4 spectral
solver, preserving the existing tensor Maxwell symbol. The new
`QuasiperiodicDyadicGreenKernel.evaluate_open` rotates constitutive tensors into
an oriented frame of a general 2D Bravais lattice, uses outgoing q-normal
residues and reciprocal-cell weights1/area, then transforms the physical6x6
kernel back. Negative open-axis queries use a proper reversed frame, not an
incorrect reflection of axial magnetic vectors. No heat filter or artificial
loss is introduced on this path.

The current support explicitly excludes the source plane, including nonsingular
lateral source-plane separations, because singular/contact separation and its
convergent completion are not yet implemented. Finite-sum output records do not
claim global pole certification, cutoff convergence or physical qualification.
The original scalar periodic assembly admission is unchanged.

`/tmp/gsp07-dyadic-open-v1.log`:2passed4.45s. An independent scalar Ewald G/gradient
Hessian constructs all electric/magnetic dyadic blocks for upper/lower queries;
separate noncommuting anisotropic tensors on a tilted plane obey bulk Maxwell
curl identities and Bloch translation. `/tmp/gsp07-dyadic-open-gradient-v1.log`:
1passed4.27s for material finite-difference/native gradient and opposite-side
reciprocity. These are spatial tensor controls, not evidence that general
periodic tensor surface assembly, singular-plane queries or physical scattering
have been completed.

### Lossy tensor direct-image reference and primary exclusion

Ported the remaining existing GSP05 free-space dyadic integration/cache and
contrast dependency into this branch without modifying the GSP05 worktree.
`LossyDyadicImageReference` independently integrates retained translated dyadic
kernels and optionally excludes the primary image. It requires strictly positive
electric and magnetic constitutive loss forms; no lossless continuation is
implicit. All lattice coincidences are checked independently of image cutoff,
while the excluded primary origin is allowed. Spatial quadrature exceptions
retain the contributing image indices and primary-exclusion context.

`/tmp/gsp07-dyadic-image-reference-v1.log` initially had1passed/1failed. Scalar
specialization at origin and a lateral source-plane point agreed with independent
Ewald regular G/Hessian. The true anisotropic case exhausted radial64: its
balanced radial relative difference1.9282975e-4 failed, while angular32 passed
(3.00237e-6). Full receipt remains in
`/tmp/gsp07-dyadic-image-receipt-v1.log`. Only the unresolved radial budget was
deepened:64→128, angular16→32 retained, unchanged rtol1e-4/atol1e-9.
`/tmp/gsp07-dyadic-image-reference-v2.log`:1passed39.82s, with finite anisotropic
regular origin max|G|=.02667163119 and opposite-Bloch reciprocity.
`/tmp/gsp07-dyadic-image-guards-v1.log`:1passed2.94s, rejecting missing strict
loss in either constitutive tensor and retained primary singularity.

This is a finite direct-image reference, not the completed fast periodic dyadic
regular kernel. Image-cutoff convergence, general lossless primary singular
separation, source-plane spectral completion and PMCHWT/field coupling remain
required. No physical qualification is inferred from these local controls.

### Tensor/projection implementation checkpoint and independent audit

Scoped periodic stage integration `/tmp/gsp07-tensor-checkpoint-v1.log`:
66passed68.24s. Existing GSP05 tests for the ported Fourier/spatial core, checked
CUDA eigensolver, cache and adaptation were also ported. Their first run
`/tmp/gsp07-tensor-port-checkpoint-v1.log` had25passed/1failed45.42s: fixed
radial96/angular16 did not meet the unchanged closed-form threshold2e-8.
Radial192/angular16 left the error unchanged (`...port-refinement-v1.log`);
radial96/angular32 passed (`...port-refinement-v2.log`,1passed4.16s). The shared
coordinate-axis integral retains transverse oscillations, so angular16 was
insufficient. Failed records are retained; no accuracy threshold was weakened.

Independent read-only audit found that 2D evaluate_open ignored the configurable
pole tolerance. Fixed by checking the minimum opposite-radiation-sheet root gap,
normalized by max(k0,abs(qz_i),abs(qz_j)); same-sheet polarization degeneracy is
allowed. The optional guard is passed by the periodic path and recorded in its
receipt; inherited free-space callers retain their existing default behavior.
The audit's near-grazing isotropic fixture now accepts default1e-10 and rejects
0.01. `/tmp/gsp07-tensor-audit-fix-v1.log`:4passed5.24s. Independent read-only
re-audit confirmed that the configuration finding is closed.

This is a tensor/projected-quantity implementation checkpoint only. It does not
admit general periodic tensor surface assembly, source-plane spatial completion,
1D tensor radiation, unprojected finite solutions at anomalous poles, public
runtime/storage or physical qualification. Prior scalar power failures remain
open and the original plan's full scope/thresholds remain unchanged.

### Tensor reciprocal/image cutoff estimates

Added `QuasiperiodicDyadicGreenKernel.estimate_open` and
`LossyDyadicImageReference.estimate_regular`. They compare every query's full
6x6 block in the impedance-balanced norm diag(I,eta0 I) G diag(I,eta0 I), with
explicit relative/absolute tolerances (absolute units ohm/m²). Failure raises
`PeriodicDyadicConvergenceError` with the measured differences and thresholds.
These are successive-cutoff estimates, not global pole or total-error bounds.

Image estimates reuse the entire inner sum and integrate only the added outer
shells. The difference is measured directly from the outer contribution rather
than subtracting two large sums. Per-image spatial-integration receipts remain
attached separately; no lossless limit is inferred.

`/tmp/gsp07-dyadic-cutoff-v1.log`:2passed4.27s. True anisotropic far-plane
reciprocal check passes; near-plane shells1→2 fails with balanced difference
9121.3566 versus threshold .1274831. Strong-loss image check passes and verifies
exactly four nonprimary images evaluated for a1D shells1→2 estimate, with no
inner reintegration. Weak-loss control fails with differences175.8099/178.2588
versus thresholds .0656993/.0680476. Stable direct-tail follow-up
`/tmp/gsp07-dyadic-tail-stable-v1.log` passes the affected image control.
Source-plane completion, spatial primary singular separation, general tensor
surface/field coupling and physical qualification remain open.

### Regular tensor exterior matrix contribution

Added `assemble_periodic_dyadic_correction` in periodic_dyadic_operator.py. It
uses actual6x6 nonprimary image fields with rows(H,-E), columns(J,M), and global
all-body-J/all-body-M storage. Smooth product quadrature includes self panels;
no free-space static term or interior term is introduced. Only the explicit
strict-loss image reference is admitted. Complete primary/interior assembly,
surface/image convergence and a solved physical system remain separate tasks.

`/tmp/gsp07-dyadic-correction-v1.log`:1passed5.29s. For two real tetrahedral
bodies, direct dyadic weak testing agrees with independent scalar Ewald weak
regular assembly at balanced relative difference2.283751296e-7. Changing both
interior electric/magnetic materials leaves this exterior matrix unchanged;
all self/cross couplings are nonzero. `/tmp/gsp07-anisotropic-correction-v1.log`:
1passed21.45s for an actual noncommuting/non-diagonal epsilon/mu sample, with
all per-image spatial quadrature receipts converged and nonzero four current/
field blocks (matrix norm5.397105267e-7). This latter coarse surface-order1
control verifies integration plumbing only; no surface convergence or physical
qualification is inferred. Existing physical failures remain open.

### Regional and periodic matrix composition

Ported GSP05 weak dyadic integration and its static-divergence restoration,
with a shared scalar regional assembler and exact isotropic specialization.
Extended regional tensor assembly to separate test/source bases, including
strict separation, near-pair refinement and rectangular all-J/all-M scatter.
The common exterior may now be tensor-valued; each interior contributes only
to its own diagonal block. Periodic lossy reference composition adds the
nonprimary image correction exactly once. `complete_operator=True` records
contribution ownership only: fixed surface orders and finite image cutoffs
remain unqualified, and no tensor public solve/RHS/field admission is implied.

Ported regional controls: `/tmp/gsp07-regional-port-v1.log`, 5 passed in 6.92s.
Composed two-body mixed tensor/scalar interior and tensor-exterior isotropic
limit versus independent scalar Ewald: balanced relative matrix difference
1.671600260022981e-13. With affected regional regression controls,
`/tmp/gsp07-composed-dyadic-v1.log`: 3 passed in 7.82s. These are algebraic
composition checks, not closure of earlier physical acceptance failures.
Actual noncommuting epsilon/mu exterior cross-body matrices satisfy Lorentz
reciprocity under electric/magnetic current exchange: relative difference
1.0749687593463922e-14, `/tmp/gsp07-tensor-cross-v1.log`: 1 passed in139.27s.
Both directions used converged free-space spectral integration. Surface order1
is intentionally a coarse assembly control, not a surface-convergence campaign.

### Tensor periodic current-to-field composition

`regular_dyadic_surface_fields` integrates the physical 6x6 nonprimary Green
map against RWG J/M in bounded query/face batches, preserving current gradients.
It applies no primary, interior or incident term and no implicit query wrapping.
`exterior_dyadic_surface_fields` wraps queries to the centered cell, adds each
body's free-space primary tensor potential and applies the Bloch phase. It
rejects interior and surface queries instead of applying an exterior potential
as body-local recovery. Fixed surface/image cutoffs remain explicitly unqualified.

The free-space tensor potential/static-gradient, surface-estimator and sheet-jump
helpers are reused from GSP05, together with disk-complement quadrature. GSP05
source files are unchanged. The lifted tensor decoder has not been ported or
claimed available; full tensor source/public runtime integration remains open.

- `/tmp/gsp07-tensor-fields-v1.log`: 1 passed4.38s. Independent scalar Ewald
  divergence-form image fields vs dyadic image fields: relative8.299899577e-13;
  native complex current gradients also agree.
- `/tmp/gsp07-primary-fields-port-v1.log`: 5 passed4.32s. Primary scalar reduction,
  gradients, adaptive surface success/failure, Maxwell sheet jumps and analytic
  incident-sheet representation controls.
- `/tmp/gsp07-complete-tensor-fields-v1.log`: 1 passed4.25s. Complete exterior
  primary-plus-image scalar comparison, translated Bloch phase and rejection of
  translated interior and surface queries.

These are implementation controls. Previously failed power/physical gates and
remaining lossless/anomaly/source/public/Bench requirements are unchanged.

### Assembly/field checkpoint integration and independent audit

Scoped phase regression `/tmp/gsp07-assembly-fields-checkpoint-v1.log`:
84 passed,1 deselected93.01s. The already-passed139.27s anisotropic cross-current
reciprocity control was deliberately not repeated; no all-repository or physical
qualification claim follows from this scoped integration.
Independent read-only audit `gsp07_tensor_assembly_fields_audit` found a P2:
ties-to-even round() gave inconsistent half-cell representatives under lattice
translation. Replaced canonical cell wrapping with floor(fractional+0.5) in
tensor/scalar periodic fields, scalar Ewald evaluation and projected resolvents.
Nearest-lattice singularity detection and integer mode matching are unchanged.
`/tmp/gsp07-halfcell-audit-fix-v1.log`:1passed4.32s, short lossy image shell at
-.5,.5,1.5 cells. Independent re-audit confirmed the original finding closed.
Affected scalar modules: `/tmp/gsp07-halfcell-scalar-regression-v1.log`,37passed
1failed16.35s. The new test initially used prohibited real_shells=0; changed only
the fixture to the valid minimum1, preserving its comparison tolerances.
`/tmp/gsp07-halfcell-scalar-fixture-v2.log`:1passed3.53s. Initial failure retained;
no duplicate rerun of the37 passing controls. All task test sessions terminal.

Checkpoint remains experimental. Tensor incident/source and local interior/
surface periodic recovery, lossless dyadic Ewald/source-plane completion,
anomaly paths, physical reference/power qualification, public/Bench integration
and all remaining GSP stages remain required.

### Tensor incident carrier and lifted RHS work in progress

`TensorIncidentCarrier` owns complex q/E/H arrays and a real phase origin. It
checks both source-free Maxwell curl equations against the exact bound tensor
sample, rejects mutated/invalid modes and overflowing evaluations, and validates
tangential real-Bloch phases without attenuation along periodic lattice vectors.
Metadata identities include exact electric/magnetic tensors, modes and origin;
metadata host transfer is separate from native CUDA field/gradient evaluation.
This internal mode carrier is not yet public Scene source compilation.

`incident_trace` now contracts tensor displacement/induction with the normal
for exact surface divergences. Reused GSP05 analytic difference RHS supports
an exact tensor exterior and a factored paired contrast kernel; no incident RWG
projection is introduced. Exact zero constitutive contrast returns zero before
quadrature. `build_periodic_dyadic_rhs` validates the carrier/medium/Bloch sector
and assembles only each body's local free-space contrast contribution, in global
all-J/all-M order. Periodic image forcing and cross-body interior terms are not
added. General nonzero full-surface convergence/physical gates remain open.

`/tmp/gsp07-tensor-carrier-v1.log`:2passed3.78s (actual noncommuting tensor mode,
analytic sheet divergences, sector/state rejection, scalar field limit and native
complex amplitude derivative). `/tmp/gsp07-tensor-carrier-rhs-v1.log`:4passed4.18s
(three scalar contrast limits and exact anisotropic background zero contrast).
`/tmp/gsp07-periodic-tensor-rhs-v1.log`:1passed3.65s (multi-body composition and
wrong-sector rejection). These are scoped controls, not physical qualification.
Zero contrast retains the carrier amplitude autograd connection with an exact
zero derivative: `/tmp/gsp07-tensor-zero-rhs-gradient-v1.log`,1passed3.92s.
Actual nonzero tensor-background pair scaling control DID NOT PASS:
`/tmp/gsp07-tensor-rhs-nonzero-v1.log`,1failed105.35s. The first contrast step
1e-5 exhausted spatial spectral orders (radial max256, azimuthal max64,
rtol1e-4,atol1e-12) at displacement[-.0951,.1114,0]. No tolerance was relaxed.
The test now prints the quantitative exception receipt on subsequent runs;
next investigation must distinguish radial/azimuthal deficits before choosing
refinement. This is an open numerical failure, not evidence of nonzero RHS
qualification. New carrier/RHS code remains WIP, not a completed stage.

### RHS spectral diagnosis and body-local interior recovery

The retained nonzero tensor RHS failure is now quantitatively diagnosed in
`/tmp/gsp07-tensor-rhs-axis-receipt-v1.log`: at radial256/azimuth64, radial relative
error2.9894210876e-4 failed while angular8.4548609351e-8 passed. The initial
128/32 corner failed both axes. A targeted continuation raises only the radial
ceiling to512, starting256/32, with angular ceiling64 and rtol1e-4/atol1e-12
unchanged. Periodic RHS now accepts an explicit DyadicIntegrationConfig so its
spectral budget is configurable independently of surface quadrature.

Reused `decode_tensor_lifted` with tensor-exterior-aware dispatch. New
`interior_dyadic_surface_fields` validates the incident sector, selects only the
requested region's J/M, wraps queries with the half-open cell convention, and
applies the common Bloch phase to body-local interior recovery. It uses no
interior image or other body's current contribution. Interior surface queries
require an explicit interior trace and retain principal-value/jump receipts.
`/tmp/gsp07-tensor-interior-recovery-v1.log`:1passed4.57s. The isotropic tensor
background with zero contrast recovers the analytic incident field through
adaptive surface integration; changing every other body's currents leaves the
result bitwise identical. This is a representation/ownership check, not a
physical qualification for general anisotropic bodies.
`/tmp/gsp07-tensor-interior-trace-v1.log`:1passed9.41s. The same isotropic tensor
background recovers the analytic translated interior surface trace with explicit
principal-value integration and sheet jumps; no shifted surface sample is used.

The diagnosed nonzero RHS pair now passes without tolerance changes:
`/tmp/gsp07-tensor-rhs-radial512-v1.log`,1passed329.56s, small-contrast scaling
relative difference1.0528938920646614e-6 (<1e-3). Spatial radial ceiling512,
azimuth64, rtol1e-4/atol1e-12; previous256 failure and axis receipt retained.
This is a nonzero pair control, not full tensor RHS surface convergence.
`/tmp/gsp07-periodic-rhs-spectral-config-v1.log`:1passed5.30s, including explicit
spectral config type rejection and multi-body zero-source amplitude derivative.

Exterior tensor current fields now accept explicit trace_side='exterior',
classify unique face-interior ownership, integrate primary principal values and
add the positive exterior half jump. Image terms remain smooth and jump-free;
interior points, edge/vertex traces, missing side or off-surface trace requests
reject. `/tmp/gsp07-tensor-exterior-trace-v1.log`:1passed4.42s, compared against
independent scalar primary/static-gradient, analytic jump and Ewald image fields.

Independent read-only audit `gsp07_tensor_assembly_fields_audit` reviewed this
carrier/RHS/recovery checkpoint and found no blocking issue. Scope covered owned
carrier state and medium identity, Maxwell/Bloch checks, constitutive divergence
contractions, lifted RHS signs/zero gradients, regional field ownership, half
jumps and cross-body surface labels. No tests were duplicated by the auditor;
this source audit does not close source compilation or physical qualification.
Scoped carrier/recovery stage integration completed:
`/tmp/gsp07-carrier-recovery-checkpoint-v1.log`:115passed,2deselected122.78s.
It covers periodic modules, tensor region/field/lifting/carrier controls and the
shared scalar analytic lifting/field paths. Previously verified anisotropic
cross-region reciprocity139.27s and nonzero RHS pair329.56s were not repeated.
All task processes are terminal. This checkpoint is implementation evidence,
not closure of physical, complexity, public runtime or cross-repository gates.

### Scalar periodic public runtime checkpoint

`PeriodicSystemOperator` now implements the shared dense CUDA/Riesz protocol
while binding a separate Ewald kernel and keeping every local analytic carrier/
interior handle free-space. Its RHS metadata records the actual periodic kernel
identity after the common local difference assembly. Explicit query dispatch
uses the periodic decoder; isolated-scatterer far-field requests reject.
Prepared source, region, carrier and kernel changes reject reuse of the matrix.

`periodic_operator_identity` hashes sampled geometry/media, kernel lattice/Bloch,
all Ewald settings and numerical configuration, separately from incident
amplitude. It is a sampled-operator identity, not a replacement for complete
public Scene declaration identity. Public configuration, Result dispatch and periodic snapshot restoration now
use this identity together with the existing exact Scene declaration signature. Harmonic balance and tensor
media reject in this scalar adapter.

`/tmp/gsp07-periodic-runtime-v1.log`:2passed5.26s, direct algebraic residual
3.3638047223008963e-16, periodic field translation, free-space local ownership,
replacement-kernel rejection and identity separation for Bloch/lattice/Ewald/
quadrature changes. `/tmp/gsp07-periodic-runtime-multirhs-v1.log`:1failed3.48s
because the test used a nonexistent PlaneWave amplitude keyword. Corrected to
the actual public PlaneWave(source_time=CW(...,amplitude=...)) declaration;
`/tmp/gsp07-periodic-runtime-multirhs-v2.log`:1passed5.15s. Distinct amplitudes
share the operator identity, have different RHS identities and preserve linear
multi-RHS solutions. Initial fixture failure retained. No physical qualification.
`/tmp/gsp07-periodic-runtime-bindings-v1.log`:1passed4.69s, including prepared
source replacement rejection. A default-query-config integration check then
found the periodic decoder used a separate log attribute instead of the shared
runtime field log (`...query-config-v1.log`:1failed6.61s). The adapter now forwards
its prepared query configuration and publishes the periodic receipt through the
shared last_field_query_log/metadata contract on success and failure, clearing
stale receipts before queries. `/tmp/gsp07-periodic-runtime-query-config-v2.log`:
1passed7.50s. This integration fix does not change field convergence thresholds.


`mw.MoMPeriodicConfig(vectors_m=..., bloch_rad_per_m=..., ...)` now enters
`mw.MoMConfig(periodic=...)` through the ordinary Scene/Simulation/Result path.
The lattice declares MoM repetition independently of the volumetric grid;
Scene boundary declarations are not inferred as lattice vectors. Existing
unsupported Scene boundary combinations still reject. Tensor and harmonic-
balance combinations reject in this scalar public adapter.

Periodic Results expose cylindrical orders (1D), Rayleigh orders (2D), and
homogeneous-exterior equivalent-current Bloch spectra (3D), not an isolated
far field. Three-dimensional forced plane-wave sectors generally encounter
an exterior spectral pole and reject; the spectrum API is not a claim of
ordinary forced solutions at those poles or physical body-interior spectra.
Point monitors classify a canonical-cell point while evaluating the original
translated point with its Bloch phase.

Periodic native solution schema 7 preserves the exact scattered coefficients,
carrier, sampled media/geometry, versioned periodic declaration, numerical
configuration and separate operator/RHS identities. Restore constructs only
decode handles and verifies identities without matrix assembly. Free-space
schema 3 remains its own representation. Physical qualification must remain
false in both native and outer Result metadata. Runtime Result guards reject
replacement numerical configurations, systems, sources, local media/bases and
periodic kernels for both newly solved and restored results.

Validation: `/tmp/gsp07-periodic-public-v1.log` retained one fixture failure
(the test used a nonexistent Result.metadata attribute) and one pass. Using
the actual native snapshot contract, `...public-v2.log` passed in 7.09s;
`...public-observables-v1.log` passed in 6.71s. The scoped public/runtime/field
and shared scalar integration run `/tmp/gsp07-public-runtime-checkpoint-v1.log`
passed 39 tests in 94.40s. Independent source audit identified mutable Result
configuration and local decode-handle binding bypasses; both now fail closed.
`/tmp/gsp07-periodic-public-binding-audit-v1.log` passed in 8.03s, covering
configuration and local-medium replacement for new and restored Results.
Existing physical power-balance failures, unsupported anomaly limits, public
tensor/HB integration and cross-repository acceptance remain open.

Independent read-only re-audit confirmed both Result binding findings closed
and found no remaining blocker in this public runtime checkpoint. It reviewed
configuration, local-region ownership, periodic queries and schema-7 restore
without repeating tests. This does not close physical qualification.

The shared Result binding change also passed its affected free-space public,
multi-body public and lifted public tests: `/tmp/gsp07-shared-result-binding-v1.log`,
28 passed in 58.03s. No full regression or previously covered physical probe
was repeated for this change.
