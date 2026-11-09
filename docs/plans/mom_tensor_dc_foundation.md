# Tensor DC foundation for mixed-body harmonic balance

The mixed scene target includes a passive linear tensor body and a separate
scalar nonlinear body. The existing public Material rejection of tensor and
nonlinear constitutive laws on the same object is retained. This increment is
an internal kernel foundation, not public mixed-HB admission or qualification.
Complex finite-frequency samples cannot silently become real DC coefficients.
Conductive and dispersive zero-frequency limits remain explicit future inputs.

## Constant real symmetric positive-definite coefficient

For an explicitly supplied coefficient A, define L by A = L L^T, y = L^-1 r,
and d = det(L). Then

    G_A(r) = 1 / (4 pi d |y|)
    grad_r G_A = L^-T grad_y G_I / d
    div_r(A grad_r G_A) = -delta(r).

The last equality follows by transforming the divergence to the isotropic
Laplacian and using delta(L y) = delta(y)/d. Units are the caller's explicit
coefficient units; this module neither inserts epsilon0 nor guesses whether A
is a relative permittivity, absolute permittivity, permeability or conductivity.

For a constant-density physical triangular single layer, transform every vertex
and the observer by L^-1. Multiply the existing analytic Coulomb panel integral
by (physical panel area)/(transformed panel area)/d. Multiply its gradient by
L^-T as well. The surface area Jacobian is essential: the point kernel's
1/d factor alone is insufficient for integration over a physical panel.
The same orientation is preserved because det(L)>0. With a physical outward
normal n, the difference of the positive-side and negative-side conormal
n dot A grad integrals is -1 for unit density. An explicitly declared
face-interior principal value is their limiting average; edge queries reject.

Related primary literature on linking anisotropic and isotropic conduction
Green functions by coordinate transformations:
[On linking n-dimensional anisotropic and isotropic Green's functions for
infinite space, half-space, bimaterial, and multilayer for conduction](https://www.sciencedirect.com/science/article/pii/S0020768305000107).
The implementation equations above are derived here for homogeneous infinite
space; no half-space, interface or multilayer result is imported from that work.

## Validation and remaining dependencies

Controls use the PDE contraction tr(A Hess G)=0 off source, physical unit-sphere
conormal flux, rotation covariance, scalar degeneration and direct quadrature
in the original physical coordinates. The panel quadrature already supplies
physical weights. A mistaken extra area factor and weight shape in the first
reference test were corrected; the failed run remains recorded. Native autograd
checks cover coefficient and observation coordinates, not a public geometry
optimization contract. Large-coordinate norms use scaling; unrepresentable
transformed panels reject rather than silently emitting zero fields.

The required next components are regional potential/flux interface equations,
source-volume DC coupling and self terms, real DC material sampling with causal
limit admission, exact state ownership/storage, mixed AC response construction,
HB integration and independent physical/convergence comparisons. The existing
scalar potential-density representation must not be reused for arbitrary
anisotropy merely by replacing a scalar epsilon with a matrix. General
nonreciprocal, singular conductive/static and complex AC-to-DC limits remain
open. No finite-frequency volume contrast is added by this static primitive.

## Regional interface foundation

`StaticTensorInterfaceOperator` now assembles real-SPD regional P0 potential
and body-outward conormal unknowns. Interior rows are `(1/2+K_i)u-S_i q=0`;
the shared exterior rows are `(1/2-K_0)u+S_0 q=phi_inc`. The exterior couples
all bodies; each interior owns only its body's panels. Analytic transformed
panel integrals supply S and source-conormal K. A native dense solve uses a
constitutive/geometric flux scale, bounded pair batches and a retained-matrix
budget (not a peak-memory guarantee). Uniform incident potentials and off-face
regional potential/field recovery retain PyTorch coefficient/drive gradients.

The response rejects nonfinite incident data, scaled solves, rescaled physical
flux, stable residual diagnostics and recovered fields. Owned state versions
reject mutations. Empty region queries preserve shapes and query gradients.
The independent audit found and closed missing physical-flux/diagnostic
finite checks; no remaining confirmed P1/P2 in the revised implementation.

GPU 1 targeted run `/tmp/gsp-static-tensor-interface-v2` passed four controls
in 9.8963168608 seconds, with 775 source hashes unchanged. Runner SHA256:
`74b5990b5c80d6f74657493b4104c5ea94581f743a46d99a04358170a4054802`.
Receipt SHA256:
`e286df8e409415cccf28bfcafb7400fd82a4d3ec76ced04046ad6269f09f06f4`.
The previous three-test v1 run remains retained. Controls cover two-body
constant potential and regional block ownership, independent anisotropic
sphere convergence, same-medium/gradient behavior and extreme representability.
Sphere levels 0/1/2 have relative internal-field errors
0.11449263899195401 / 0.0321018270413047 / 0.008309218513978072.
The last error still fails the original 0.005 physical threshold; the unit
control asserts refinement, not physical qualification. General volume-source
coupling, real material DC sampling, mixed AC/DC state and public HB integration
remain open. No production qualification is granted by these controls.

A separate selected level-3 diagnostic (1280 faces) reduced the same reference
point's relative error to 0.0020976763774735036 < original 0.005. Algebraic
relative residual was 1.3213809570648172e-15, retained matrix 52428800 bytes.
This is one selected physical reference pass, not general physical qualification;
level-2 failure remains. `/tmp/gsp-static-tensor-sphere-level3` completed in
18.5368320411 seconds with source hashes unchanged. Runner SHA256:
`d3be86d93c38fe62e1e0ca7db1f5ece68f6b6aff7a25ffdf766369b31abf30cb`.
Receipt SHA256: `0df97c67d39368891e174f167d88afe2151bc3934ad00c52c3578636c3263dd1`.
Result SHA256: `53588bd448580be05bf2aad71bad1e862c558391a573e159d6937e5a30e04752`.
The runner completion status records successful execution; the result's separate
selected_reference_pass field reports comparison against the original threshold.

## Induced cell-polarization particular fields

`StaticTensorPolarization` owns a polarization array on an exactly bound volume
and an explicit real-SPD coefficient. For `D=-A grad(phi)+P`, the field equation
is `div(A grad(phi))=div(P)`. Each constant tetrahedral P contributes outward
face density `n.P`, so its particular potential is the sum of analytic single
layers. The field is the negative sum of their observer gradients. Opposite
internal face orientations cancel for equal P and retain the correct net face
charge for discontinuous P. This includes the interior cell self field; adding
another point-contact correction would double count it. No linear contrast is
added, and this is not a public prescribed interior excitation.

Off-face queries and explicit face-interior PV are supported. PV fields are
arithmetic means of the two limits, not one-sided regional traces. Cell edges
and vertices reject even when some densities cancel. Pair batching bounds the
kernel calls; exact volume/surface binding, cloned source arrays, version checks
and finite-output checks protect source ownership. Material, polarization and
observation derivatives stay in native PyTorch. Geometry optimization remains
outside the contract.

GPU1 `/tmp/gsp-static-tensor-polarization-v1`: 3 pass, 6.8589775087 seconds,
777 source hashes unchanged. Runner:
`342d6f0518f86c7a5841a126af5cffb51f62ab755106c60c0649c138d85af0d8`;
receipt `1db129b09053365c1dfd577d2571d3987fc60cd09a696160dc442f6a876feb96`.
The independent audit found no confirmed P1/P2. A test-only extension then
checked nonuniform cell P on a refined tetrahedron. Only its two updated
parameter nodes ran in v2: 2 pass, 8.0538374251 seconds, 777 unchanged hashes.
Runner `e4caf9f2a3a48e3927bbdc9126d2cb33aa50f2f176b143e8a98cc9b82930593b`;
receipt `92b1975c7195add456a59260cf87775f7b0284c1bf1a2ba337fc3f77927770ca`.
Current evidence comprises the two unchanged v1 controls plus the two v2
parameter controls. Independent audit verified this source/evidence mapping.

The separate reference evaluates the volume dipole gradient with physical
12th-order Duffy cubature, not this implementation's surface integrals. It
checks anisotropic off-volume potential/field for uniform and nonuniform cell P.
Scalar self-field comparison, uniform-P refinement cancellation, anisotropic
external face jump/PV, gradients, empty queries and mutation refusal are separate
controls. Nonuniform-P off-face agreement is not a direct internal-face trace
jump test. General source-to-interface closure, real causal material DC samples,
exact mixed state, HB integration and physical qualification remain open.

## Induced polarization transmission

The exterior unknown q remains `n.A0.grad(phi0)`. With no free interface charge,
`D=-A grad(phi)+P` gives the interior conormal `qi=q+P.n`. Topological face
incidence now partitions each source's analytic layers into outer boundary
faces and internal cell faces. The internal faces alone represent `-div(P)`
within the open body; call their potential `v_i`. Each body therefore uses
`(1/2+K_i)u-S_i q = v_i + S_i(P.n)`. The exterior rows are unchanged.
Interior recovery is `S_i(q+P.n)-D_i u+v_i`; exterior recovery remains the shared
all-body representation. This avoids double counting the boundary polarization
and avoids interpolating a particular field's boundary traces a second time.
An earlier equivalent trace-subtraction discretization and its results remain
in the first run's hashes; it is not the final implementation.

Exactly bound volume topology supplies each surface face's owning source cell.
Duplicate regions and foreign surface bindings reject; source ordering is
canonical by region. Responses own the polarization sources and boundary P.n,
check their versions, and retain source/material/observation gradients. Uniform
incident drive and induced sources share the existing matrix and solve. There
is still no public prescribed interior source or public mixed HB admission.
The particular-field helper also provides explicit interior conormal traces:
`n.A.grad(psi)_inside = n.A.grad(psi)_PV + P.n/2`.

Targeted GPU1 evidence is retained under `/tmp/gsp-static-tensor-volume-interface-vN`:
- v1: 5 pass / 1 fail, 19.1318059405 seconds. A finite inward offset differed
  from the limiting trace beyond the original control tolerance. The corrected
  control uses two inward offsets and first-order Richardson extrapolation,
  preserving that tolerance. Sphere level1/2 errors were .0736663/.0414933;
  both fail the physical .005 threshold and are retained.
- v2: 11 pass, 20.4060725234 seconds, 778 source hashes unchanged. The direct
  internal-charge formulation gives polarized anisotropic sphere errors
  .012229033578/.002938491107. Only the selected finer reference comparison
  passes .005; the refinement unit control is explicitly separately labeled .05.
  Controls include the affected source/interface suites and nonzero two-body
  source superposition, canonical source ordering and receiving-body response.
- v3: only the new nonuniform-P matched-medium refinement control, 1 pass,
  6.3364447625 seconds. A fixed piecewise P distribution is subdivided without
  changing its physical source; an independent existing scalar analytic cell
  field supplies the exact same-medium reference. Errors .022869130501 and
  .010081603417 decrease but BOTH FAIL .005. This is refinement evidence, not
  physical qualification. It exercises nonzero internal-charge interface forcing.

Independent audit verified equations, topology/region ownership and reference
scope, then found a zero-internal-charge numerical edge: summing very large
finite query coordinates before multiplying by zero could produce NaN. The
implementation now multiplies coordinates by zero before summation, preserving
a finite zero and its gradient. Targeted closure is recorded below. General
nonuniform-P accuracy, material DC sampling, exact mixed-state persistence and
HB integration remain open, together with all original acceptance obligations.

Audit closure: v4 passed three affected controls in 5.6449853014 seconds,
778 hashes unchanged; independent review closed P2 with no remaining confirmed
P1/P2. Final coverage is the unchanged v2 controls, the new v3 control and
the three v4 affected/added controls, not a claim of another complete rerun.

v1 runner SHA256 `3315eb2b7f9a5bce9dbbc33d6a53f77e92a57c45418edf7bebfd5956b90d8e87`;
receipt SHA256 `a01a53c63829633437d2e88dd47594b708f52b51d220d2c022801cbad71d3da1`.
v2 runner SHA256 `c18f27ca95840225f6b996b2e95986ebde909108cc5f9f3096155a4f18f2aa94`;
receipt SHA256 `1561aa590c42aa7625645106650ad1aa643b8b8d1a22543a086da38e7aa50e9b`.
v3 runner SHA256 `ee94196d875baca127d979b6a6bd12fd35711b05feb8169334cd351945675f4c`;
receipt SHA256 `29c9716780a7b2f896352ab687ff1c74ddd1b2a37a993f56ba2261fd5a488e29`.
v4 runner SHA256 `d9964e8ed505f0b0a6b06abea0aa1ba97b9024961c9012d4e7b2c34daf6c811e`;
receipt SHA256 `e4826cc0119453017fc9781a689b80ebaf7641ee77053be86324957c15d15ab9`.

## Analytic public-material DC samples

`compile_static_tensor_material` now consumes supported public Material,
PhysicalMaterial and GyromagneticFerrite declarations directly. Tensor epsilon,
mu and electric conductivity replace their corresponding scalar bases.
Homogeneous Debye/Lorentz increments and damped/undamped Drude Laurent terms use
the existing scalar expansion isotropically, preserving conductivity and plasma
coefficients separately from the finite relative tensor. Instantaneous nonlinear
coefficients do not add linear contrast to the background. Spatially custom,
modulated, breakdown, PEC and gain-enabled pole models reject in this path.

A channel returns an SI potential-equation coefficient only if its analytic
limit is finite, real, symmetric and positive definite. Conductive, lossless
plasma, unresolved complex and nonpositive/degenerate channels remain explicitly
classified; there is no small-positive-frequency substitution, imaginary-part
truncation, or scalar projection. A constant complex gyrotropic AC declaration
is not automatically a real static constitutive law. The causal ferrite model
instead has the exact real limit `mu_inf*I + Ms/|H0|*(I-b*b.T)`; this compiler
uses that expression directly. Selected decreasing-frequency comparisons verify
its approach to the existing AC model, not a general DC solver qualification.

`compile_static_tensor_interface` selects the electric or magnetic channel and
compiles all public-material samples before assembling the internal regional
operator. It reuses the existing SI-coefficient operator and does not expand the
public Scene/HB admission rules. Both channels' actual matrix and solve agree
bitwise with explicit SI coefficients in the selected control. Singular-channel
classification does not implement the corresponding conductive/plasma solver.

GPU1 evidence: `/tmp/gsp-static-tensor-material-v1` passed three controls in
4.9652385246 seconds, 780 source hashes unchanged. Tests cover tensor-base/pole
replacement, scalar nonlinear-background parity, singular and unresolved-complex
admission, mutation refusal and the causal ferrite DC limit. The internal
interface factory was then added; only its new control ran in v2, one pass in
5.6796080265 seconds, 780 hashes unchanged. Current coverage consists of the three
unchanged v1 tests and the new v2 test. No full regression was repeated.

v1 runner SHA256 `f895bc7d5b075ca27e1e974febb9de1c709a291bd4822f216b3945a4eb4a0a4b`;
receipt SHA256 `e484a305d5af92a78e3c8ae813f61f83125dac6ed71783f2067320893b2011b3`.
v2 runner SHA256 `9c91d8403cdb777949f95b2898ee32996e0c150dc8b764209532fa3469842be1`;
receipt SHA256 `e96236d4a362bc978adaa8697d2cd2b00cfcae436267de8a7e8878f7800fef88`.
Independent audit verified both stages and the actual evidence mapping, with
no confirmed P1/P2. Exact mixed DC state persistence, mixed AC/DC construction,
public HB integration and outstanding physical accuracy/acceptance remain open.

## Exact prepared-response state component

`static_tensor_snapshot` now snapshots a native regional static response against
its existing operator and exact bound source volumes. The preparation identity
covers body geometry, regional coefficients, pair policy, matrix, flux scale and
metadata. The state contains exact uniform drive/offset, potential, exterior
conormal, boundary P.n, source polarization and historical diagnostics. Source
records bind exact volume coordinates/connectivity and surface correspondence.

Restore requires the owning state's independently fixed expected binding and
already validated operator/volume preparation. It checks canonical source region
order, finite array shape/dtype, metadata scope and P.n recomputed from source
cells. It clones arrays and reconstructs the response without interface assembly
or a numerical solve. Kernel constitutive validation and topology bookkeeping
remain allowed. The component does not re-prove the solved equations from
historical residual metadata; replacing both payload and its supposedly external
binding is outside that integrity boundary. The outer mixed-state owner must
store and verify the expected binding independently.

GPU1 `/tmp/gsp-static-tensor-snapshot-v1`: two controls passed in
6.4224152723 seconds, 782 source hashes unchanged. Save/load uses the native
safe tensor payload, and restore is tested with operator assembly, layer assembly,
uniform solve and torch.linalg.solve forbidden. Exact state and selected
interior/exterior fields match bitwise after restore. Controls also cover cloned
storage, mutated decoded sources, changed state against the owner hash, internally
inconsistent boundary P.n, mutated preparation and missing volume inventory.
Independent audit found no confirmed P1/P2 and verified the evidence mapping.
A separate new control checks a valid different operator, equivalent physical
geometry with different volume vertex numbering and a rehashed false production
qualification flag; only that new node is run in v2.

This is an internal prepared-response state component, not standalone Result
persistence. The owning preparation must still be restored independently.
Restored material/source-gradient qualification and full mixed AC/DC/HB state
integration remain open. Physical and final acceptance obligations are unchanged.

v2 passed its new control in 4.7526718210 seconds, 782 source hashes unchanged.
Current evidence is the unchanged two v1 controls plus the new v2 control.
v1 runner SHA256 `08861ef129251cd062a6e9b8335c10a3b446148192891167e141055b0d83ca7c`;
receipt SHA256 `656a8e66afaad1a6e95d76a58a113ae45b018e84a0af1fce5fed27ce04412ce6`.
v2 runner SHA256 `9ba7819b5a2879f620571241397f414ee70c1a8bcd4b22f5443a7ef97e82cfe7`;
receipt SHA256 `ad822f42df19461d97ae8b333f329b44df7041216bcc6bdebcb70a9def2f204a`.

## Exact static numerical preparation

The preparation snapshot now stores the actual matrix, flux scale, regional
coefficient/factor/root determinant and numerical metadata. Restore binds an
independently validated surface system and owner digest, checks the retained
matrix budget before device matrix allocation, and decodes the operator without
panel assembly or solve. Constitutive Cholesky admission validates each stored
coefficient independently; the stored factors themselves are preserved exactly
and checked for triangular structure, positive diagonal, finite reconstruction
and determinant consistency. The matrix's authenticity depends on the outer
owner's independently fixed digest, not a hidden matrix reassembly.

Response snapshots are now version `regional-spd-static-response-v2`: their
preparation identity includes actual kernel factors and root determinants as
well as coefficients. The previous v1 response identity is not accepted as v2.
Snapshot also requires each source kernel to equal its owning regional kernel;
restore copies saved regional factors into independently owned source kernels,
rather than retaining a fresh decomposition's rounding. This preserves actual
query numerics while leaving restored-gradient qualification open.

Independent audit identified two P2 admission gaps in the initial addition:
approximate factor reconstruction did not itself prove coefficient SPD, and
response identities omitted the factors used by field queries. Both were fixed
with explicit controls, including a slightly indefinite coefficient that would
otherwise fall within the reconstruction tolerance and a different legal
approximate factor that must not share the old response binding.

GPU1 targeted evidence, all 782 source hashes unchanged within each run:
- `/tmp/gsp-static-tensor-preparation-v1`: 4 pass, 8.5372887133 seconds, before
  the audit fixes; retained as initial coverage, not final admission evidence.
- v2: 5 pass, 8.6264213575 seconds, after SPD/response-factor identity fixes.
- v3: only two affected exact-replay controls, 2 pass, 6.4148191819 seconds,
  after source-factor replay. A deliberately changed fresh factor during decode
  verifies that stored regional factors, and the exact resulting fields, prevail.
Final coverage is the unaffected v2 controls plus the two v3 replay controls.
Native save/load, independent new operator identity, bitwise state and interior/
exterior fields, cloning, budget checks, altered factor refusal and no assembly/
solve are covered. Full mixed Result wiring and physical acceptance remain open.

v1 runner SHA256 `64a71f8e486dfef0419a3206ffa0cacaff4d698047fbb597a982837a5f4ece78`;
receipt SHA256 `a19f363bfc3a83526001190e1d16c14c63b6f1944041c579e5463a8e221645ca`.
v2 runner SHA256 `489ebb3fc73919eecb0376a7d8c86746f8ed72d3db07862b16c5052d03f6de64`;
receipt SHA256 `1d75866e0ce40318af4f485f844023290ec1aca2ec634605835468f872385790`.
v3 runner SHA256 `ada455e106f136065310dba4f28aad617882e33f057c47b204d4532338b0a242`;
receipt SHA256 `b2ef794337fa9f1b8a9412d380ffb84aa0a89ef07db8e7964caac9759995ff04`.
Independent final stage audit verified source-factor cloning and the actual
v2/v3 evidence mapping, closed the identified P2 gaps and found no remaining
confirmed P1/P2 within this internal preparation/state component.

## Optional material-scaled dyadic radial quadrature

The reciprocal and nonreciprocal original candidate index0 diagnostics showed
radial refinement, rather than angular resolution, dominating the selected error.
An explicit `DyadicIntegrationConfig.radial_map='material_split_v1'` now allocates
one Gauss rule to `[0,R]` and one to `[R,infinity)`, where
`R=2*sqrt(norm2(epsilon_r))*sqrt(norm2(mu_r))`. This scale is a node-allocation
choice, not a pole bound. Jacobians retain the physical rho measure. The infinite
tail, static subtraction/restoration and original directional tolerances remain.
Each configured radial order N means 2N total radial nodes in this option.
Default infinite-rational mapping and existing scalar contrast branch handling
remain unchanged. This option explicitly rejects contrast-kernel evaluation;
its current implementation covers single regional kernels, including the actual
interior queries that failed in the original batch.

The radial map is explicit in numerical identity, cache keys and versioned
kernel receipts. Spectral receipts report two intervals and two nodes per unit
radial order. No gain, artificial loss, finite transverse cutoff or implicit
retry policy is introduced. Selected same-solver high-order comparisons are not
independent physical references or proof of a complete candidate solve.

Diagnostic `/tmp/gsp-tensor-split-radial-study-v1` completed 225.7651545228s,
614 source hashes unchanged. It used only index0 from each frozen original query,
not all1536 displacements. Relative differences against original1024/128 values:
reciprocal per-interval128:8.5479841876e-9,256:5.3284694562e-13;
gyrotropic per-interval128:5.3542151510e-9,256:1.6880471930e-9.
Both selected radial64->128 and128->256 changes fit the original half-budget.
Total radial node counts are256/512 respectively, not128/256.

Production targeted GPU1 `/tmp/gsp-dyadic-material-split-v1`:14pass,
5.5780574502s. Includes complete-domain exponential moments, a separate closed
scalar dyadic Green formula and affected existing adaptation/cache controls.
The added v2 cache isolation/adaptive receipt/public numerical-identity control
alone passed in4.9274898823s. No production Python changed between these runs;
current coverage is14v1 plus1newv2. Broader actual tensor query checks and full
candidate/physical qualification remain required, with original failures retained.

gsp-tensor-split-radial-study-v1: runner `da9a39e3dfb2df914e2ae42a38bd076830b93cbee42eabed66f76c3d0869737f`, receipt `0bec756ef857ecf7326b3cffb2e42a0d5425dfa479980aa69dc35b5306495580`.
gsp-dyadic-material-split-v1: runner `f70980bd429dba778717369da9fd99bb5ad0fa9576e773fa76c2851579b4f74c`, receipt `e623901e3d1797ccf100f86e6789192d6e73efb25632eacf10c97cc63cc8fc8c`.
gsp-dyadic-material-split-v2: runner `fcc0fc20b45eb7189e493a0b55555fb0f3817ba2ca2023daf0c315034b43fd38`, receipt `9d69c9b84a18b20cd336e127b0f052378f33c94e60a1eccdf4afa4a15da009cd`.

Canonical default serialization omits `radial_map`; nondefault serialization
includes it. This preserves default numerical/operator/RHS and periodic kernel
identities. Independent audit identified the original default identity issue.
Actual prechange provider 69e53f7 generated schema20/nested13 H-matrix result and
1D/2D/3D periodic payload fixtures, without stripping fields from new snapshots.
The v3 reader restored the old result with bitwise coefficients/fields while
assembly/operator construction/solve were prohibited; its 15 targeted tests passed
in 6.5572983492s. The v4 old periodic reader failed before pytest because the
strict decoder still required the new field; its missing-XML exception left a
stale running receipt. Terminal session88611 exit1 and a separate
terminal-failure.json preserve that failure. The decoder now validates canonical
default keys and the optional nondefault string strictly. v5 old periodic preflight
preserves payloads and identities in all three dimensions; 7 routing plus 3
nondefault roundtrip/invalid-value controls passed in 4.0456193974s, with 783
source hashes unchanged. This is persistence/configuration coverage, not periodic
spectral or physical qualification. A subsequent unused-import removal is inert.

/tmp/gsp-radial-prechange-fixture-v1.py: `14c0c9063d56665d108d0ef503356a7f6b3c735716153d9954f9b842fb91f398`.

/tmp/gsp-radial-prechange-fixture-v1/receipt.json: `b7bd508ec640e8b4be5d8fe4fc1cd1769de36b50708a0aba1eb3ef7724f13dbb`.

/tmp/gsp-dyadic-material-split-v3.py: `54a2a246b6d781e7bea101d3446a120603ef1e212644b0c4d1f7035b3287ba93`.

/tmp/gsp-dyadic-material-split-v3/receipt.json: `edaac625593fe306ccdd3d8b7912cb68f793999773f730c03bbbcfc130754939`.

/tmp/gsp-radial-old-periodic-fixture-v1.py: `c61cb1e84f90a5f79c0650066209ddbab89cccc358e4470e146dfa9d2a0bb32a`.

/tmp/gsp-radial-old-periodic-fixture-v1/receipt.json: `686baae6e3b102f5451ff842b2f6362247978f43b248c3861c085124a2a4db6b`.

/tmp/gsp-dyadic-material-split-v5.py: `984cadea2f9a78131045e05a27b09dfb2e19bd3f4e8c4ccf632fb41304ec1c26`.

/tmp/gsp-dyadic-material-split-v5/receipt.json: `df09d7aad263c9090ddd7fce7bf73f7e6d2c7bf76cd0dfd4cd6806882caea3b7`.

Independent audit closed the canonical identity P2 after inspecting v5 terminal
artifacts and prior exact-state restoration. No remaining confirmed P1/P2 in
this optional mapping stage. Final physical and full-candidate qualification
remain open; selected numerical and persistence checks do not promote capability.


## Fixed tensor DC cell response preparation

`prepare_static_tensor_cell_response` prepares the existing real-SPD regional
interface's induced polarization map at exactly bound volume cell centers.
Native CUDA autograd differentiates the fixed linear transmission solve, including
internal polarization jumps, exterior cross-body coupling and all surface
potential/exterior-conormal/normal-polarization rows. Input/output cell axes follow
binding order; surface axes follow operator body/face order. Linear background
contrast is already in the regional operator and is not added to induced P.

An explicit retained-response byte budget rejects before differentiation. It is
not a peak workspace bound. Returned matrices are fixed-discretization values;
material/geometry derivatives through preparation are not retained. No zero-source
quadrature certification is claimed. This is an internal prerequisite, not a
new public mixed tensor/scalar nonlinear HB entry point. Public material sampling,
AC response integration, mixed DC constitutive admission, harmonic state identity,
exact result storage and public wiring remain required before that path exists.

Targeted GPU1 v1: 2pass in7.3313772213s. Nonzero anisotropic one-cell direct solve,
polarization tangent, reversed multibody binding order and nonzero cross coupling
agree with response columns; budget/duplicate/foreign-binding rejection checked.
v2: added nonuniform eight-cell polarization control alone passed in6.5980464444s,
including internal-face jumps and direct derivative comparison. Production code
unchanged between v1/v2; final parameterization also rescales the old one-cell
fixture but that rescaled parameter was not rerun. These establish algebraic
preparation consistency, not independent physical correctness of the underlying
collocation discretization. Existing nonuniform DC physical failure remains open.

Runner naming initially collided with old preparation snapshot v1 and mkdir
rejected before GPU work. The overwritten old runner was restored byte-for-byte
from v2 with its version substitution; its SHA matches the original receipt
64a71f8e486dfef0419a3206ffa0cacaff4d698047fbb597a982837a5f4ece78.
Distinct cell-response artifact names now preserve both sets of evidence.

/tmp/gsp-static-tensor-cell-response-v1.py: `2be4fc150cef8efab2f1ab9f8c881bece43666423343b84a6defff4d90338eb3`.

/tmp/gsp-static-tensor-cell-response-v1/receipt.json: `6ba16e9437b6bfc9058d999bba769a24206f0af1c721da3fa641349808753d6c`.

/tmp/gsp-static-tensor-cell-response-v2.py: `a6c0dd370575beebac12d38699f32b946df6cad3d8bb3074204384fc3aa2a6cf`.

/tmp/gsp-static-tensor-cell-response-v2/receipt.json: `030f330d1786f49f48c53bc2f5dbe951b5d9e3200dfa5f67830d3b8c56bf90d5`.

Independent audit found no confirmed P1/P2 in this preparation layer and verified
v1/v2 terminal evidence, unchanged production code and restoration of the old
runner hash. Full public mixed HB and final physical qualification remain open.


## Tensor harmonic material sampling

`compile_tensor_harmonic_material` retains full public reciprocal/nonreciprocal
AC constitutive tensors at every positive retained harmonic and compiles electric
and magnetic DC Laurent channels analytically. Shared linear-background extraction
preserves scalar instantaneous chi2/chi3/TPA coefficients separately. Ferrite
keeps its original constitutive model, including gyrotropic frequency dependence;
only exact ordinary Material instances are cloned to remove instantaneous sources.
The existing scalar compiler keeps its original admission and numerical behavior.

This sampler returns the existing HarmonicMaterial container with explicit tensor
static channels; existing scalar runtime consumers are not automatically tensor
compatible. Public mixed HB still requires AC response/preparation identities,
DC closure admission, exact state storage and runtime wiring. Singular/complex
DC remains classified and rejected by SPD admission, never projected or replaced
by finite-frequency samples. No tensor nonlinear susceptibility law is invented.

GPU1 v1 session81468 TERMINAL1:3pass2fail in5.8276344510s. One failure was incorrect
iteration over HarmonicSources in the test; the other exposed Ferrite inheritance
being cloned to ordinary Material, losing its magnetic model. Both fixed; original
failure evidence retained. v2 session90132 TERMINAL0:5pass in5.8139879322s, including
existing scalar regressions and reciprocal/Ferrite harmonic/DC sampling controls.
v3 session70499 TERMINAL0:one added testpass4.7332489984s verifies explicit complex
nonreciprocal DC rejection and nonzero TPA/Kerr/chi source identity. Production
unchanged between v2/v3; 786 source hashes unchanged in each run. These are sampling
and source-law consistency checks, not a coupled solve or physical qualification.
Independent audit found no remaining confirmed P1/P2 in the production change.

/tmp/gsp-tensor-harmonic-material-v1.py: `c147320d185f5ba1989b2a6338203019850684db8fd337663c9ecc2869f861b0`.

/tmp/gsp-tensor-harmonic-material-v1/receipt.json: `10b9d6855842fd3017449d22b6a67ae6c7d877a08409c589f2b29cdca9d734fb`.

/tmp/gsp-tensor-harmonic-material-v2.py: `d29a2a0fec28a4409f14164abacf1fa7c453aa0c1f51bd09b35d5481f5d8b6c1`.

/tmp/gsp-tensor-harmonic-material-v2/receipt.json: `997b08ebf0cb88e6cf7989864293fa85da30c2c35a1e5bbec5d94a1233431160`.

/tmp/gsp-tensor-harmonic-material-v3.py: `3bd21a37aa2dde800e6f463ed4e112aa34c261516b6eaf2464287b7c297e33f8`.

/tmp/gsp-tensor-harmonic-material-v3/receipt.json: `99a5ca172b17cba3b9e77cbce27ebdec31120ac151be1e8ead70e3d00f87efc5`.


## Actual mixed AC response integration control

The existing `prepare_cell_field_response` supports scalar source/observation
volumes while a separate tensor body participates in the generalized surface
system. This avoids introducing a tensor volume-current kernel where public
nonlinearity exists only in a separate scalar body. General tensor volume sources
and conductive/non-SPD mixed DC remain separate open requirements.

GPU1 /tmp/gsp-mixed-tensor-volume-response-v1 session80062 TERMINAL0:
1pass25.9501051176s, 787 source hashes unchanged. The actual two-body assembly uses
one scalar-limit TensorMedium and one scalar volume, no manufactured kernels.
Default adaptive source and field budgets are retained. Source RHS on the tensor
body is zero while its induced surface response is nonzero. Nonzero complex
current applied to the prepared columns matches direct solve and complete
surface-plus-volume cell fields. This proves ownership/routing consistency,
not general anisotropic convergence, physical accuracy or nonlinear HB support.
Independent audit found no confirmed P1/P2. Two unused test-only statements were
removed after the run; production code did not change.

Next mixed HB integration must preserve scalar source-volume admission, compare
full sampled media without scalar tuple equality, use the new SPD tensor DC
response, and bind tensor kernel/material versions in preparation identity.
DC potential/conormal state must receive its own exact recovery/storage semantics;
it must not be mislabeled as the existing scalar single-layer charge state.

/tmp/gsp-mixed-tensor-volume-response-v1.py: `295809100cb4d33f91250c3610e4619596ae7af134ae9c5e41fe04fa7f3f9d02`.

/tmp/gsp-mixed-tensor-volume-response-v1/receipt.json: `3c7a9577818ead7669c0371f5c6a21eaaaf75e854b6a9694a2659abeb24085eb`.

## Internal mixed tensor/scalar harmonic preparation

`prepare_mixed_tensor_harmonic_balance` combines actual scalar-source AC response
maps through mixed surface systems with the analytic real-SPD regional DC cell
response. Scalar nonlinear volumes retain chi2/chi3 source laws; linear tensor
bodies carry no extra volume contrast. Positive-frequency P/J maps obey
J=-i*n*omega*P. The model solves retained DenseHarmonicBalance equations, not a
public Scene entry. Only dense scalar exterior/source-volume AC maps and finite
SPD electric/magnetic DC channels are admitted at this stage. TPA mean current,
conductive/plasma/complex DC and tensor volume sources remain explicit gaps.

Material samples, geometry, precision and ownership are checked before map
preparation. Preparation identity now binds static potential/conormal operator
and kernel states and tensor decoder kernel policies. The old scalar charge
recovery explicitly rejects this preparation; dedicated recovered potential and
conormal fields, exact harmonic state storage and public runtime wiring remain
required. No field or exact-storage support is implied by the model alone.

GPU1 v1 /tmp/gsp-tensor-harmonic-preparation-v1: actual two-body/two-harmonic model
passed in31.8467138419s, 789 source hashes unchanged. Scalar-limit TensorMedium
body plus nonlinear scalar body generated DC and second harmonic; direct DC
transmission matched prepared potential/conormal/normalP maps and retained DC E.
Mutation and unsupported source-region controls passed. This is retained algebra
and scalar-limit routing, not general anisotropic physics or truncation closure.

Independent audit found a P2 missing actual integer frequency-axis admission.
The new preflight requires finite positive f0 and each exterior/body sample at
n*f0 before static/AC preparation or operator refresh; exact sample/operator
payload matching then transfers the check to the actual operators. Added swapped
samples and coherently wrong declared fundamental negative controls. Audit
confirmed the code fix; final v2 evidence recorded below when terminal.


v2 session47930 TERMINAL0:4pass78.6199121494s, 789 source hashes unchanged.
Covers the mixed retained solve, two invalid actual frequency-axis controls, and
existing actual scalar HB generation/recovery/extension regression affected by
preparation identity changes. No broader regression was rerun for this change.

/tmp/gsp-tensor-harmonic-preparation-v1.py: `c57840901d47ee4753b427004bb57be018d801ace7e7252ae624d8daead2aebd`.

/tmp/gsp-tensor-harmonic-preparation-v1/receipt.json: `bffb3f643ffa529147f46d24f9285cda235053bf453b8849477123789b03f02f`.

/tmp/gsp-tensor-harmonic-preparation-v2.py: `d63486521381d6d88c3c72b71a1f48f753b2c3e4b1375ed0b7ccf7ae3491f7ce`.

/tmp/gsp-tensor-harmonic-preparation-v2/receipt.json: `65f71e21682819ad63859f172009794da35faf3f11503f826bb205f95c8957ad`.

Independent audit verified v2 and closed the actual-frequency-axis P2; no
remaining confirmed P1/P2 in this internal preparation stage. Public mixed HB,
dedicated potential/conormal harmonic state and final physics remain open.

## Mixed tensor DC recovery and regional queries

The interface source RHS is shared by direct solves and `response_from_traces`.
The latter computes source ownership and a diagnostic residual from already
prepared potential/conormal coefficients without a new linear solve. It retains
the supplied physical conormal exactly, avoiding a scaling roundtrip's rounding.

Harmonic recovery now owns this dedicated DC response alongside AC currents and
surface coefficients. The scalar single-layer charge array is empty and the
receipt explicitly identifies regional potential/exterior-conormal data. DC
queries dispatch to the restored trace representation; sources and tensors retain
mutation checks. Pump scaling remains attached to AC background plus recovered
induced current. The old scalar harmonic snapshot explicitly rejects this state
until a dedicated exact tensor DC harmonic schema is implemented. Public runtime,
standalone exact storage and general physical qualification remain open.


GPU1 /tmp/gsp-tensor-harmonic-recovery-v1 session58286 TERMINAL0:14pass,
84.388s, 789 source hashes unchanged. Includes actual mixed retained solution
and half-pump recovery with torch.linalg.solve forbidden during recovery and
queries, DC exterior/both interiors, AC scalar source-region field consistency,
mutation rejection, existing static polarization/interface/snapshot controls and
scalar HB recovery/extension regression. It does not test AC tensor-region queries.
Original nonuniform DC refinement physical error remains1.008%>0.5%; no physical
qualification follows from passing the algebraic recovery controls. Independent
audit found no confirmed P1/P2 in this stage. Dedicated tensor harmonic snapshot
and public entry remain open.

/tmp/gsp-tensor-harmonic-recovery-v1.py: `764f7ba4aec93ead3a07bd713bf42fba3ad71ec31516efd7eab091a01a07f2c5`.

/tmp/gsp-tensor-harmonic-recovery-v1/receipt.json: `f23f7180902fde58e89571d7a4362b87600067f004eb835d79a8f0c535f82c20`.

## Owner-bound mixed harmonic state snapshot component

`tensor_harmonic_snapshot` stores exact detached harmonic arrays and the existing
versioned regional DC response, bound to an independently retained owner digest.
Preparation identity covers AC matrices, full sampled media, body geometry,
incident carriers, numerical policies, ordered volume bindings, DC preparation
and response maps, dynamic maps, fundamental frequency and exact cell nonlinear
coefficients. Restore requires that matching preparation already exists. This is
not standalone Result persistence or reconstruction of the preparation.

Typed finite array inventories, real DC, harmonic axes, polarization-current
partition, nonlinear P(E)/dissipative source laws, DC source polarization and
source-free static drive are checked without solving. Extra current/dual or
compressed state is rejected rather than silently omitted. Restored arrays are
owned clones. The original scalar charge snapshot still rejects mixed tensor DC.

v1 GPU1 session96625 TERMINAL0:3pass34.5984192947s,790 source hashes unchanged.
Actual .5-pump mixed state restored under forbidden surface/static constructors,
linear solve and HB solve, with exact arrays and bitwise DC three-region plus
AC scalar-source field queries. Independent audit identified a P2: the initial
prepared digest omitted nonlinear coefficients. The fix owns exact model cell
coefficients and fundamental frequency in preparation/signature/digest and
checks the constitutive source law on restore. Added shared-linear-map but
2*chi2 model/preparation rejection, resealed inconsistent E rejection, and
extra-state omission rejection. v1 is prior-fix evidence only.


v2 session64186 TERMINAL0:3pass34.0663959691s,790 source hashes unchanged.
Independent audit verified the terminal artifact chain and closed nonlinear
constitutive identity P2; no remaining confirmed P1/P2. Covers exact state and
DC/AC scalar queries plus corrected constitutive and extra-state admission.
It still requires the matching existing preparation, not a standalone Result.

/tmp/gsp-tensor-harmonic-state-snapshot-v1.py: `d44cbab792d20b573401c12fc6dcf00bef6b74357f9f82b54d6fcc370f70cfa4`.

/tmp/gsp-tensor-harmonic-state-snapshot-v1/receipt.json: `a1783139cf54647dedbc64eda44b243ba16c720de21a2e60206611d66c78113a`.

/tmp/gsp-tensor-harmonic-state-snapshot-v2.py: `06cac14ff44fc2dd119b01c6419f38062c7115831617b2735f63a6eb35976c67`.

/tmp/gsp-tensor-harmonic-state-snapshot-v2/receipt.json: `0351b9f68a5157e88f6edd09097ce7b3acc87d33efe50a6a2a977c576d7fddb1`.


## Self-contained dense mixed harmonic bundle

`tensor_harmonic_bundle` stores actual dense AC matrices, sampled regional media,
body geometry, exact incident carriers, numerical settings, volume geometry,
static preparation/factors, all retained response maps and nonlinear coefficients,
plus the owner-bound harmonic state component. Restore needs the external owner
binding and CUDA device, not the original model/preparation. New decode-only
handles retain stored matrices; the DenseHB model is reconstructed from stored
maps without solving. Carrier records are restored exactly, without reconstructing
approximate source phases. Stored tensor bytes have an explicit admission budget;
this is not a peak-memory bound. Public Scene validation/Result dispatch is still
required, and restored decode handles do not promise operator reassembly APIs.

v1 /tmp/gsp-tensor-harmonic-bundle-v1 session73516 TERMINAL1:2pass1fail
35.8191063544s. It incorrectly wrapped a complex scalar sample in public Material,
whose nonlinearity property requires real base declarations. Original failure
retained; downstream restore assertions had not executed. The fix adds native
restore_scalar_medium and shares the original passive outgoing branch expressions
with compile_medium. It validates finite passive nonzero samples and exact k/eta
without resampling a public declaration; original metadata is retained.

v2 session91673 TERMINAL0:5pass37.0866898168s. Actual mixed .5-pump state and maps
roundtrip through a disk file into newly allocated geometry/operators/model with
surface/static assembly constructors and linear/HB solves forbidden. Harmonic
arrays, DC exterior/both interiors, AC scalar source-region fields and HB residual
match bitwise. Storage budget and owner tamper reject; complex lossy and double-
negative sampled branch controls retain exact data and reject a changed k.
Separate affected scalar test_physics regression session19378 TERMINAL0:14pass
4.6596984649s. Both final runs retain791 unchanged source hashes. No fullsuite
repeat. These are exact persistence checks, not general anisotropic or physical
qualification; restored material/geometry gradients are not qualified.

/tmp/gsp-tensor-harmonic-bundle-v1.py: `c68c3850ddf1ca19f8c5093a70cb5ba36942252b60e048bbeb7558d3cc05a7d6`.

/tmp/gsp-tensor-harmonic-bundle-v1/receipt.json: `82b17193144c28745f221bedcf31e41ec6a824024c0a98bdd3ac798b802729cc`.

/tmp/gsp-tensor-harmonic-bundle-v2.py: `c07124bd4dbdfea14f24375c9a26ffc031964eeeb273003b304bbda276b32335`.

/tmp/gsp-tensor-harmonic-bundle-v2/receipt.json: `a819db328f23d943d78e3e173f6f0089bd8df5f177b967abb4d3b47535ee2777`.

/tmp/gsp-tensor-harmonic-bundle-scalar-regression-v1.py: `78b8d06140801a7f53203691d79ba67c717865f6c79819894e7b663be31e8b46`.

/tmp/gsp-tensor-harmonic-bundle-scalar-regression-v1/receipt.json: `0e9ccac075a202d765237516d63cf54e36d55d5ac84559f689446f7b0f7dad12`.

Independent audit verified both final runs, closed scalar restoration P2 and
found no remaining confirmed P1/P2 in this bundle stage. Original v1 failure,
internal-only scope and stored-bytes (not peak-memory) budget remain explicit.

## Public mixed tensor harmonic bridge

The public dyadic+dense+free-space HB configuration now selects a mixed runtime
while scalar configurations retain their existing runtime. It preflights real
SPD electric/magnetic DC limits and scalar source-volume regions, then combines
actual retained AC operators and regional DC maps. H/H2/periodic mixed HB, TPA
mean-current closure, tensor volume source kernels, cutoff extension and public
parameter gradients still reject. No scope or acceptance threshold is relaxed.

Schema21 wraps the self-contained bundle with public configuration and Scene
identity. Restore resamples declared material harmonics/DC limits, matches body
geometry, sources and nonlinear coefficients, and checks the retained numerical
policy and final pump/branch without reassembly or solving. Current positive
control is a scalar-limit tensor body coupled to a scalar nonlinear tetrahedron;
it is not general anisotropic physical qualification.

Public v1 session58982 TERMINAL1 failed after successful solve and field queries:
the new test referenced nonexistent Result.metadata. The assertion now uses
harmonic_state.receipt. Public v2 session29075 TERMINAL0:1pass37.7434016662s,
793 unchanged source hashes. Bitwise restored state, DC/AC source-region fields
and monitors pass with surface/static constructors and linear/HB solves forbidden;
altered Scene nonlinearity, altered outer HB policy and cutoff admission reject.
Independent audit found an outer Result metadata binding P2. The fix checks the
complete native preparation receipt plus public representation, frequency and
qualification/convergence claims when loading schema21. Existing Result frequency
axis validation already binds native state; the audit withdrew its initial claim
that this axis was unchecked. New controls alter only outer metadata or frequency.

Affected scalar public regression v1 session79064 TERMINAL1:4pass3fail61.8118s,
unchanged hashes. All scalar single-body controls passed. The three multibody
controls did not execute because their relative runpy fixture path was resolved
from the Bench runner cwd. The follow-up v3 uses provider cwd and reruns only the
previously unexecuted multibody cases and the modified public mixed test.

/tmp/gsp-tensor-harmonic-public-v1.py: `4cbe961fac6696e94c8535ade7331251989b1afd3e44575c9e111bb62015a19e`.

/tmp/gsp-tensor-harmonic-public-v1/receipt.json: `7ff1e40a5bfce118538ac64621a471d09ff47f9da013c894f56c188fca4d26de`.

/tmp/gsp-tensor-harmonic-public-v2.py: `b843087a192e0654acb98d346580a5cfb6d65e45f0c5f4a6c63d6ad194a28367`.

/tmp/gsp-tensor-harmonic-public-v2/receipt.json: `2266abd65ee389251ea5a3ee61665338a2dcc91a458eb9061e228484fba469e1`.

/tmp/gsp-tensor-harmonic-public-scalar-regression-v1.py: `05012ebf1eb25d3871f3688a8ce242569cde40d44e5b0a7c0a109507a9b8131a`.

/tmp/gsp-tensor-harmonic-public-scalar-regression-v1/receipt.json: `447057f1658e35e745faa49bf063c2f4ed651aa0e44578c0a8d417e5b6de0f75`.

Public v3 session41475 TERMINAL0:4pass, no skips, 218.56977382395416s;793 source hashes unchanged and independently verified. New public schema21 exact restore and metadata/frequency negatives plus all three scalar multi-body modes passed. Independent review closed the metadata P2 with no remaining confirmed P1/P2 in this stage. Prior failures remain above; no full regression or physical qualification is inferred.

/tmp/gsp-tensor-harmonic-public-v3.py: `8a8bb01f31a0818a4e3ef26a5c0399c2ebfecaf119ed159790fea44d66dfce76`.

/tmp/gsp-tensor-harmonic-public-v3/receipt.json: `b48352ea4b625db7f83a58b05dcc0590c0619946b57414f2cf779d61b0860028`.

## Neutral floating conductive-limit surfaces with tensor dielectrics

New internal StaticTensorFloatingOperator couples the actual SPD exterior and
finite-SPD dielectric kernels to conductive-limit bodies marked None. It does
not invent a finite conductor permittivity. Unknowns are surface potential,
scaled exterior conormal and one floating constant per conductor. Exterior
Calderon rows couple all bodies; dielectric rows retain their regional kernels;
conductors enforce u=induced_current_trace+constant and one area-weighted zero
conormal row. Exterior and dielectric fields decode natively; conductive volume
fields explicitly require their actual current closure. Public mixed HB, induced
dielectric polarization maps, storage and full current closure remain open.

Tests cover tensor-dielectric/conductor coupling, neutrality, trace gauge,
nonconstant trace retention, native trace-to-field gradients, region ownership,
mutation, storage budget and independent conductor-sphere exterior refinement.
Selected external-point relative errors at levels0/1/2 are .0230570098592,
.00758160971358,.00205515017460. The finest selected point is below the unchanged
.005 physical threshold; the test asserts monotonic refinement only, not broad
physical qualification. All stage receipts remain physically unqualified.

v1 session44328 TERMINAL0:2pass8.2941638930s. Independent audit found missing
finite physical-response admission. Fixed finite RHS before solve, physical
conormal scaling, stable residual norms/relative diagnostics and decoded fields.
v2 session53545 TERMINAL1:2pass1fail9.0547315115s. Extreme-drive finite diagnostics,
conormal overflow and field overflow controls executed; a final RHS overflow
fixture (.1-radius surface centered100) failed earlier in existing Green panel
principal-value matching. That translated-panel numerical robustness failure is
OPEN, unchanged and not hidden by broadening the PV tolerance. v3 uses radius100
centered0 solely to reach the intended RHS-overflow admission. v3 session60841
TERMINAL0:3pass8.8417024594s,795 unchanged source hashes. v2->v3 changes only that
fixture; production finite checks are unchanged. Independent review verified
hashes/runner/results and closed the finite-response P2; no other confirmed P1/P2
in this floating-interface stage. This does not close the translated-panel issue
or the general mixed DC/TPA, physical, population or final regression gates.

/tmp/gsp-static-tensor-floating-v1.py: `2e7446596611069142f183fd1f64a6b1da7b1f09b483cb8c800ad4e2d1b0efe3`.

/tmp/gsp-static-tensor-floating-v1/receipt.json: `fd58b5f906bd9da5ab1f18f94de9a50da483bb656696de8a2b91e653a01a7698`.

/tmp/gsp-static-tensor-floating-v2.py: `1c9bd784ee96903968aaa41b08d92dda82fae135c3a5303ed4c864f2276565e7`.

/tmp/gsp-static-tensor-floating-v2/receipt.json: `93429046aa0f6e4516f1ff712b30ba5e6b9fc1eec7731a947b3f79d4f69fd8ee`.

/tmp/gsp-static-tensor-floating-v3.py: `10f37c44add8d2dcab752bcad2be162f4dc1be7376ab525209422ad150b203bc`.

/tmp/gsp-static-tensor-floating-v3/receipt.json: `e0fd7a271cf3fbb0c0b4deb79d97b7e6b9a4d396b9f7e02416f5fc818557aefe`.

## Relative-coordinate static collocation

The translated .1-radius sphere centered100 failure is reproduced by the old
face-distance calculation: maximum global distance1.4298647848e-14, four rejected
faces against local tolerance about2.9884353814e-15. Subtracting the origin only
after forming the rounded global center still leaves4.4258694627e-15 and four
rejects. Computing the centroid of source-relative target vertices first yields
1.6365348542e-17 and zero rejects at the SAME tolerance. The shared static layer
builder now requires exact declared face-center collocation for diagonal mode
and constructs that mathematical centroid locally. Arbitrary field observers
are never projected or reinterpreted. StaticTensorGreen also centers supplied
panel/observer coordinates before geometric predicates and tensor transforms.
PV/edge thresholds are unchanged; representably off-plane queries and edges
still reject improper principal values. Translation cannot recover precision
already lost when geometry vertices were represented; tests compare the same
rounded translated vertices with their local copies.

v1 session83973 TERMINAL0:14pass8.6151118521s no skips,796 unchanged source hashes.
Both actual dielectric and floating interfaces now assemble at center100 and
match local operator matrices, conormal, gauge-shifted potential and external/
dielectric fields. Off-plane/edge controls and affected old static Green,
interface, sensitivity and floating/overflow controls passed. The prior v2
failure remains in its receipt; this regression closes that specific translated
collocation failure, not general geometry or physical qualification. Existing
anisotropic sphere coarse/fine physical failure .0083092185>.005 is unchanged.

## Captured full-query radial split terminal evidence

Session28834 is TERMINAL1, not live. /tmp/gsp-tensor-full-query-split-v1 completed
in4447.3750719754s with626 unchanged source hashes on tensor-dc d4cb915. Reciprocal
1536-query case failed original radial half-budget:6.81471254985e-6; angular
2.96542869996e-10 passed. Gyrotropic1536-query case converged in2693.9771380834s:
radial1.24368204535e-7, angular1.20862653177e-9. Overall failed_query_estimators
is retained. Only radial_map changed; max256 means256 nodes PER INTERVAL,512 total.
No finite tail cutoff, loss addition or tolerance relaxation. This is still the
first captured query of each original candidate, not a complete solve or an
independent physical reference. Independent audit verified input/runner/value
hashes and source inventories, including finite complex128[1536,6,6] gyro values.
The tensor-dc worktree no longer has a live-job pin; keep its evidence baseline.

/tmp/gsp-static-tensor-translation-v1.py: `4509621e183804fe12581aace2c9d22d5c62d4f010de013a24e47150c983dc15`.

/tmp/gsp-static-tensor-translation-v1/receipt.json: `c48204a56e9db82456f0e3eb36c79fa4f174f54854b9ce8bfa919a69dcfed3cf`.

/tmp/gsp-tensor-full-query-split-v1.py: `2646f4d36b373a2503528ca029d9a0e149b6b1613291a118cbfabf99d4c58c24`.

/tmp/gsp-tensor-full-query-split-v1/receipt.json: `1e03620b2281b660c3eeea37daf965c3599961a8888442d07c2e51f899d2d722`.

Independent translation-stage review verified14/14 terminal controls,796 current
and before/after hashes and runner SHA; no confirmed P1/P2 remains. It closes the
selected translated collocation failure without claiming arbitrary extreme
translation certification or mixed DC/HB physical qualification.

## Native face-current closure coupled to tensor static surfaces

StaticTensorFloatingOperator now includes dielectric induced cell polarization
through internal distributional charge and the P.n displacement jump, using the
same regional Green convention as the finite-SPD interface. It prepares retained
linear maps from ordered dielectric cell P and conductor face traces to cell E,
surface potential/conormal and floating offsets. Budget checks precede native
Jacobian construction and bound retained map bytes, not peak workspace. Material/
geometry preparation gradients and quadrature convergence are not certified.

Existing MixedHarmonicClosure accepts this operator and retains its native
FaceCurrentBalance conductivity/TPA solves. Actual cell order and current-body
ownership are unchanged. Nonlinear iterations use the prepared static maps;
current-only volume sets may coexist with unmeshed linear tensor dielectric
surfaces. Nonzero TPA retains the existing explicit per-cell coefficient
contract; scalar zero is allowed. Public runtime admission is unchanged and
recovered tensor-current state rejects explicitly until its potential/conormal
current schema is implemented; scalar charge recovery must not absorb this state.

The new three-tetrahedron control has a linear tensor surface body1, scalar
conducting nonlinear volume2 and scalar dielectric nonlinear volume3, supplied
in order[3,2]. Actual face-current solves and nonzero TPA couple through the
retained tensor interface. HB uses zero positive-frequency response matrices
for this DC algebra/gradient control, NOT real AC scattering acceptance. DC
cell fields match direct induced-P/current-trace interface fields, conductor
neutrality is retained and linear tensor-body fields are nonzero. Implicit chi2
gradients match centered differences while static-interface solves are forbidden
inside HB iterations. Map mutation, P ownership, noncurrent TPA and storage-budget
negatives reject. The current-only volume control retains every linear surface.

v1 session46909 TERMINAL1:5pass1fail11.3943745894s,797 hashes unchanged. Existing
scalar mixed and floating regressions passed along with the new reversed-order
control. Current-only test incorrectly supplied scalar nonzero TPA and hit the
existing explicit per-cell guard. v2 changes only that test to one-cell [.8];
production policy is unchanged. Session91192 TERMINAL0:2pass8.2741454020s,797
unchanged hashes. Only new tests rerun; old unaffected passing evidence retained.
The v1 failure remains intact. No full regression, physical or public mixed-HB
qualification is inferred.

/tmp/gsp-tensor-harmonic-current-closure-v1.py: `4dd6d38d260ad800ad12c59dc95014d6d48380ef0c4ff09a806bc6687a0ed34a`.

/tmp/gsp-tensor-harmonic-current-closure-v1/receipt.json: `0c173d2b48264f8f184adc62a3676696ff52a3af4ced45301f67e834fedcf1e9`.

/tmp/gsp-tensor-harmonic-current-closure-v2.py: `d65c65e5fe6014e4d6d33da4113602d66e084ece7a9dd2425edd475f837d0ec4`.

/tmp/gsp-tensor-harmonic-current-closure-v2/receipt.json: `0ec8764ebb3b97e15b97ffaa38a891853d3ffb40a8b359748eb9c4189bba2fdc`.

Independent review verified v2 runner/hash inventories and terminal controls,
confirmed only the test changed since v1, and closed the per-cell TPA interface
scope. No remaining confirmed P1/P2 in this DC closure stage. Actual AC maps,
public recovered-state integration, storage and full physical qualification
remain explicit dependencies.

## Actual AC maps with scalar current closure and tensor surfaces

Internal prepare_mixed_tensor_harmonic_balance now selects actual scalar
conductive/TPA DC channels, requires their volume ownership, and combines the
floating-interface/current closure with actual regional AC response matrices.
Every other electric and all magnetic DC channels still require their analytic
finite-SPD admission; tensor conductivity and tensor AC volume-source regions
remain unsupported explicitly. DC polarization/current matrices are zero in the
current branch so DenseHarmonicBalance delegates that channel once to the closure.
Positive-frequency current remains -i*n*omega*P + Jdiss once. The finite-SPD branch
retains its original behavior. Current policy and static retained-response budget
are forwarded, and preparation identity now binds floating kernels, conductivity,
current discretization, closure ownership and all response tensors. No public
runtime admission, magnetic DC recovery or current-state snapshot is enabled.

The real two-tetrahedron/two-harmonic control uses a scalar-limit TensorMedium
linear body and a scalar epsilon2, sigma_e=.001, chi2=.002, chi3=.001, TPA beta5
body. It assembles nonzero AC response matrices, solves the actual HB equations,
checks generated DC and H2, nonzero Jdiss, AC response equations and equality to
the actual DC current closure. Surface construction/static solves are forbidden
during the retained-map HB solve. Budget and mutated conductivity reject, and
state recovery still explicitly refuses its missing potential/conormal current
schema. This is not a general anisotropic physical reference or a public result.

v1 session87660 TERMINAL1:5pass1fail24.1818652302s. New test incorrectly attempted
to patch nonexistent SystemOperator.solve after successful preparation, so its
solve assertions did not run. Existing SPD preparation/material/storage controls
all passed. v2 corrects only the test guard to SystemOperator.__init__; production
is unchanged. Session45943 TERMINAL0:1pass16.7101093894s,798 source hashes unchanged.
Only the affected new node reran; old five passing controls and initial failure
remain. Independent review verified hashes and runner, no confirmed P1/P2 remains
in this internal preparation stage. Full current recovery/storage/public routing,
magnetic DC closure and physical/population/final acceptance are still open.

/tmp/gsp-tensor-harmonic-current-preparation-v1.py: `bd378fb8113bab2500d83ff529d076221a18c2dc2ea3923797e13f24f9f6edd1`.

/tmp/gsp-tensor-harmonic-current-preparation-v1/receipt.json: `2b785c28ff22fa9cefdb5b6c6711941208fc2dc9ae525b34ca77066a82d91501`.

/tmp/gsp-tensor-harmonic-current-preparation-v2.py: `17fc4f2ceedcede3120b11e29171087f3f186a09298a7f7e46f1eb841a3c161a`.

/tmp/gsp-tensor-harmonic-current-preparation-v2/receipt.json: `5419482949780a304496e1c47ec2371da3b98fa585875c97dee6fa5a55662e63`.


### 2026-09-28 mixed tensor/current recovered state and field decoding

The internal current preparation now recovers regional potential and exterior
conormal traces, separately neutral conductor offsets, selected-current-order
face potentials and total DC current with physical ohmic sigma E added once only
on current cells. StaticTensorFloatingOperator can own retained physical traces
and compute their residual without repeating its interface solve. Recovery still
evaluates the current closure and verifies the HB equations; it is not an entirely
solve-free operation. Field queries reuse the state: exterior/dielectric static
layer evaluation, conductive-volume CR face means and retained AC coefficients.
The shared CR interpolation also serves existing scalar states. DC magnetic
queries and the SPD-only snapshot reject this new current state; no public
admission or new storage schema is enabled.

The actual two-body/two-harmonic control now checks recovery, direct interface
trace agreement, ohmic partition, all-region DC/AC queries with current/interface
solves and surface construction forbidden, unsupported magnetic/storage paths,
wrong-region rejection and mutation ownership. A second four-body control has
linear tensor region1, different scalar current regions2/3 and dielectric nonlinear
region4, with selected volumes (4,3,2) and current binding order (3,2). It checks
nontrivial offset reordering against the independently solved interface and
dielectric polarization sources. AC tensor samples are scalar-limit controls,
not general anisotropic physical references.

v1 found an existing coupled AC query defect: VolumeFieldResponse tried to read
TensorMedium.eta. The fix uses the established anisotropic field metric, separate
E and vacuum-impedance-normalized H; scalar regions retain regional impedance.
Receipts identify the metric and absolute H threshold, without asserting an
error bound or physical qualification. Original failed evidence is retained.

v1 session25735 TERMINAL1: 2pass/1fail,57.1220002035s; both existing scalar
current/mixed preparation regressions passed. v2 session30631 TERMINAL0:
2pass,45.9320187597s; only the affected new parameterized controls reran. Both
runs retained 798 identical before/after Python source hashes.

/tmp/gsp-tensor-current-recovery-v1.py: `9c6193e85335173b6b5698dd355b01e4deabf609b95802f957a0ffc6eb143938`.

/tmp/gsp-tensor-current-recovery-v1/receipt.json: `0d4bad4b747515b98a7640f65a3808b5d73716f9ce081d5450904edf75da1abb`.

/tmp/gsp-tensor-current-recovery-v2.py: `aca1f7b56242533cdd57245297d556a297a894c199442fc17247b5bb6cb77523`.

/tmp/gsp-tensor-current-recovery-v2/receipt.json: `ff04145062cf2c151066fdbfaf1052462ae1427d66e29134aee472022c2ad13a`.

Independent read-only audit verified v2 terminal status, runner SHA and all 798
source hashes against current files. The AC tensor query P2 is closed; no
confirmed P1/P2 remains within this internal recovery scope. Magnetic current
DC, exact storage/public admission and general anisotropic physics remain open.


### 2026-09-28 floating tensor static storage foundation

The existing static preparation/response codec now handles native floating
operators under independent regional-floating-static-preparation-v1 and
regional-floating-static-response-v1 versions. It stores actual matrices and
SPD factors, uses None exclusively for conductive interior kernels, reconstructs
conductive-region ordering and geometry-derived areas, and stores exact physical
potential/conormal traces and per-conductor offsets. Ownership binds complete
geometry, preparation and source-volume inventory. Conductor source bindings,
malformed shapes, unsupported versions, missing exterior kernel, invalid factors,
changed owner hashes and insufficient stored-matrix budgets reject. Restore
replays exact factors and validates structural/SPD admission without interface
assembly or solve; it does not re-certify stored algebraic residuals or physics.

New tests use a genuine off-diagonal SPD tensor dielectric coupled to one or
two conductive-limit bodies, optional dielectric polarization and nonconstant
conductor face traces. Save/load/restore preserves all arrays and decoded
exterior/dielectric fields exactly, with interface construction/layers/solve
forbidden during restoration. Current interior decoding remains delegated to
the current-volume closure. This is the static dependency for full mixed-current
HB persistence, not a completed current HB bundle or public Result schema.

GPU1 /tmp/gsp-floating-tensor-storage-v1.py session36810 TERMINAL0:
7pass (two floating + five existing SPD),5.8537905402s; Python source inventories
unchanged before/after. No full regression repeated.

/tmp/gsp-floating-tensor-storage-v1.py: `403396504db0d643a7edea62be01ee278a147f2e5e55946c54f2e13f3c11bca9`.

/tmp/gsp-floating-tensor-storage-v1/receipt.json: `6a8d1064afc3c39033908539650fcccbec3c81b21057746c418b5c09d9b24e27`.

Independent read-only audit confirmed all 799 Python source hashes, runner SHA
and terminal 7pass/no skips. No confirmed P1/P2 in the static codec scope;
full current HB state/face-block/bundle/public persistence remain open.


### 2026-09-28 owned mixed tensor/current harmonic state

The common mixed harmonic state codec now has a separate current-state version
with face potentials, selected-current-region order and floating offsets in
addition to retained AC/DC fields and currents. Its preparation digest binds
actual AC matrices, material/carrier/numerical identities, floating preparation,
ordered volumes, conductivity, current discretization, inner numerical policy,
indices and all retained maps. A shared encode_current_closure preserves the
existing scalar numerical record without another implementation.

Restore requires the independent owner and exact preparation. It validates
nonlinear polarization/dissipative sources, total current including sigma E
once on current cells, dielectric source ownership, potential gauge, native
weak current residual and historical inner policy, cell electric consistency,
exact mapped surface potential/conormal and both operator-order and selected-
current-order offsets. It never solves the current, interface or HB equations.
The existing SPD path remains a separate version. Magnetic/compressed/dual
state is not silently discarded. The self-contained current preparation bundle
still explicitly rejects; this component alone does not enable public storage.

Actual two- and four-body HB controls save/load the state and forbid current
and interface solves, linear solve and surface construction during restoration.
All regional DC/AC replay queries must be bitwise equal to the original state.
Changed offsets/current partition reject with the original owner, and rehashed
invalid offsets/current/gauge reject by structural/numerical validation. These
are scalar-limit AC tensor controls, not general anisotropic physical evidence.

The independent audit identified a P2 in the first implementation: the stored
current residual threshold was not re-derived from the bound inner policy. The
fix reconstructs the zero-DC initial current from each saved AC block and TPA
coefficient, then derives absolute+relative*initial_weak_residual exactly as the
original solve. Stored threshold, converged flag and failure reason must agree;
rehashed changes to each are rejected. This numerical check performs no solve.
The outer HB electric threshold remains a historical solver setting protected
by the independent owner; restore does not independently re-certify the original
HB nonlinear solve or physical accuracy.

v1 session57717 TERMINAL0:4pass66.9222558755s, including old SPD owned/bundle
and scalar current storage controls. Its passing tests did not cover the
audit-found threshold issue. v2 session74555 TERMINAL0:2pass46.1900343848s,
rerunning only the affected two current nodes with the added negative cases.
Both runs retained 799 identical before/after source hashes; no full rerun.

/tmp/gsp-tensor-current-owned-state-v1.py: `6189b024cbe7dab61072bdae0854e713801554d1b6fa2795d12bee7a9be647b0`.

/tmp/gsp-tensor-current-owned-state-v1/receipt.json: `f5b5f7df07d8f58f876d6cf2eeb4b6c3aa9d4bb4f18be4a7375b52e3a745b83d`.

/tmp/gsp-tensor-current-owned-state-v2.py: `3b1b8d0b6c0073ccf789de33d643d54b1b56e1d7c41a7a7cafef69d5b8132d91`.

/tmp/gsp-tensor-current-owned-state-v2/receipt.json: `1b900f8cba7ea2a9e5f534e7b6365842124e2afcf70712b801e9eab01b7e5408`.

Independent audit verified terminal v2, runner SHA and all 799 current hashes.
The current-policy P2 is closed, with no remaining confirmed P1/P2 within owned
state scope. Self-contained bundle, public persistence and physics stay open.


### 2026-09-28 self-contained mixed tensor/current harmonic bundle

A separate dense-mixed-tensor-current-harmonic-bundle-v1 now stores the complete
internal preparation and owned state. The existing scalar face-current decoder
was extracted into restore_current_closure and shared; it validates stored face
topology, affine gradients, volume weights, nonnegative conductivity and active
numerical policy, cloning numerical arrays. The scalar wrapper retains material
and magnetic channel checks. The internal current bundle binds conductivity via
its independent owner and nested preparation digest, without attempting to infer
analytic DC conductivity from a finite-frequency AC sample. Public material
resampling remains a later public runtime/storage requirement.

Restoration rebuilds validated geometry and decode handles, replays exact AC and
floating static matrices/factors, restores current blocks without constructors,
and derives current/dielectric cell indices from ordered volume ownership. It
checks map dimensions and unqualified receipts, restores zero direct DC maps
and the mixed current closure, then invokes the owned-state conservation/source/
trace/offset checks. The original prepared object is unnecessary. The byte budget
bounds stored tensors, not peak restore workspace. Restoration does not recertify
HB or physical accuracy, and magnetic DC/public admission remain unsupported.

The actual two- and four-body controls forbid surface/static/current-block/mixed-
closure constructors, interface layers, current/interface solves and linear solve
during restore. Rebuilt surface systems are independent objects. Restored model
residuals match exactly when evaluated separately afterward (that residual
evaluation does invoke the current closure). All regional DC/AC field replay is
bitwise equal. Retained budget and changed conductivity/topology reject; changed
conductivity with a rehashed bundle still fails the nested prepared-state binding.
Saved-array mutation does not change the reconstructed preparation. These remain
scalar-limit AC tensor controls, not a general anisotropic physical reference.

GPU1 /tmp/gsp-tensor-current-bundle-v1.py session45475 TERMINAL0:
4pass68.3531056046s, no skips, 799 unchanged before/after Python source hashes.
Includes the two current bundle cases, old SPD owned/bundle and scalar current
storage regressions. Independent audit verified current hashes and runner SHA;
no confirmed P1/P2 within internal self-contained bundle scope. No full rerun.

/tmp/gsp-tensor-current-bundle-v1.py: `3dfe3784fae85fe2ed6b97ee2d0598688efb0c4ca54926e0909d41f55fac20e5`.

/tmp/gsp-tensor-current-bundle-v1/receipt.json: `942a6df2ba6bc5580d69ed9383dd29f7511829a4a6af658bcc16534f65f075a6`.


### 2026-09-28 public mixed tensor/current harmonic runtime and schema22

The public dense free-space mixed runtime now admits scalar conductive/TPA
volumes coupled to linear finite-SPD tensor bodies, forwarding an optional DC
current tangent policy. Current bodies require explicit scalar-kernel volume
ownership; unresolved tensor conductive/complex electric limits and non-SPD
magnetic channels still reject. This does not implement magnetic DC recovery,
cutoff extension, compressed/periodic HB or public material gradients.

Native public storage uses schema22 / coupled_sie_vie_hb_mixed_tensor_current;
the SPD path retains schema21. Result restore routes and outer metadata checks
cover both. Load reconstructs the exact internal bundle and independently samples
Scene AC/DC material channels, source carriers and nonlinear coefficients, then
checks body/volume identity, current-region inventory, scalar DC conductivity,
noncurrent static kernels and the public current tangent/default stopping policy.
Schema, Scene current inventory and actual state must agree. A changed analytic
DC conductivity is rejected even when finite-frequency AC samples and the Scene
signature remain the same. DC magnetic monitor components are unavailable, not
zero-valued predictions; explicit magnetic queries reject.

Public controls solve an actual two-tetrahedron H2 problem with a scalar-limit
TensorMedium linear body and scalar chi2/chi3 material, with or without sigma_e
and TPA. They include physical-position point monitors, Result save/load with
constructors and solves forbidden, exact fields/state/monitor replay, changed
Scene/policy/metadata/frequency rejection and independent DC material sampling
checks. A separate current control enables DC GMRES tangent policy with an
explicit column batch to test forwarding and exact restoration. These selected
controls do not establish general anisotropic physical or production accuracy.

GPU1 v1 session74957 TERMINAL0:2pass29.330935617s (SPD/current public).
v2 session88258 TERMINAL0:1pass20.4557127682s (only new current_krylov node);
production unchanged between runs. Both have 799 unchanged before/after source
hashes. Independent audit verified terminal results, hashes and runner identities;
no confirmed P1/P2 in this public routing/storage scope. No full rerun.

/tmp/gsp-tensor-current-public-v1.py: `da5e0fdb8dbe8fef8f7936b1d2802b5c6c62915b17931e405ce0d3ab2aaffffc`.

/tmp/gsp-tensor-current-public-v1/receipt.json: `fd02d86217942a881ed8e7b54c6039b7eb02b23010a9db10e42212c82063198b`.

/tmp/gsp-tensor-current-public-v2.py: `eb850e0f8bc0fcfd647c9e7b70752c7352f080f5dca9d151200b5759e1aea20a`.

/tmp/gsp-tensor-current-public-v2/receipt.json: `611cd20a7293db1955e32e6c0595c8d3ebc150cf8cf11b2936a597cecf063bed`.


### 2026-09-28 original mixed population DC admission census

The unchanged seed81/count64/max_bodies3 design was frozen with SHA256
fe343a8c485eefc1fd57650fd73ad265ec9ea5aedba407b68362a0b3d108945c. It contains
16 scenes mixing scalar nonlinear and tensor bodies (45 bodies total). Native
analytic material compilation identifies 23 tensor bodies with unresolved
complex electric and magnetic DC channels, plus 22 scalar bodies with electric
and magnetic conductive limits. The supported scalar electric current/TPA path
does not resolve the 46 tensor complex or 22 scalar magnetic channels. No mixed
scene passes DC admission; none was solved or physically qualified. Volumes
cover every actual scalar current body, not just explicitly nonlinear bodies.

The native public/internal preparations now share admit_mixed_dc_channels,
retaining original coefficients and rejecting the same unsupported formulations.
MixedHarmonicDCAdmissionError (a NotImplementedError) records region, channel,
kind and original reason for every unsupported channel, rather than hiding later
failures behind the first one. Invalid declarations/volume ownership still raise
validation errors. No imaginary part is discarded, no causal DC model is inferred
from fixed-frequency complex data, and no candidate or threshold is changed.

v1 census session6672 TERMINAL0,5.1204868117s recorded the former first-error
behavior. v2 session88278 TERMINAL0,5.3075845642s reused that same frozen plan
file and verified all 68 native missing-channel records without AC assembly.
The census is an admission diagnostic, not a solver or population acceptance.
Native targeted run session70599 TERMINAL0:4pass46.1139791282s (complete-channel
error + SPD/current/current-Krylov public regressions).

gsp-original-cohort-dc-admission-v1: 801 source hashes unchanged.

/tmp/gsp-original-cohort-dc-admission-v1.py: `87243891d06486e367d96cfd309b2453544d3a677c2309eeb130af6e4f9a67f0`.

/tmp/gsp-original-cohort-dc-admission-v1/receipt.json: `13ab161318676e7e155c00b6b554f482a5c2cf9e3b3c14622fc85ea45f6d7d86`.

gsp-original-cohort-dc-admission-v2: 802 source hashes unchanged.

/tmp/gsp-original-cohort-dc-admission-v2.py: `e7fc33665e269ce5e5ff2fdad84058d94f66a3b20b0fd61451ca829f91a74767`.

/tmp/gsp-original-cohort-dc-admission-v2/receipt.json: `b50ac24759c049e81eb17c7756869427d3d27b82fc59f3d70254c67c42b97074`.

gsp-mixed-dc-admission-v1: 800 source hashes unchanged.

/tmp/gsp-mixed-dc-admission-v1.py: `edb5cbafaa502d6f2418b4035edac721bf9d9fe16069a04ab0ad972838b2ff6a`.

/tmp/gsp-mixed-dc-admission-v1/receipt.json: `52f4369afc9f2283aefa785a68e4f0721b8ed2067ec7b91ef5a38ced442ddd6f`.

Independent audit found no confirmed P1/P2, verified original-plan SHA, all68
channel records and terminal native4pass/Bench1pass. Native800 and Bench802
source hashes match their own frozen runs. Census v1/v2 had801/802 hashes;
the later Bench test change is covered by the separate replay run. These
results are admission diagnostics, not candidate solves or 32x8 acceptance.


### 2026-09-28 homogeneous SPD magnetic current kernel

The internal cell_tensor_magnetostatic_current_fields evaluates induced P0
currents in one homogeneous real-SPD permeability. For div(mu grad G)=-delta,
H=integral (mu grad G) cross J and B=mu H. Applying the divergence theorem to
an oriented tetrahedron gives a face density -(mu n) cross J against G.
Differentiated analytic panel integrals retain the interior current contact;
curl H equals that contact plus mu Hess(G)*J. The latter is an explicit Ampere
defect and vanishes for distributionally divergence-free compact currents.
The implementation returns the complete H derivative and div B as diagnostics.
No imaginary coefficient is discarded and no interface correction is implied.

The new controls compare an off-diagonal SPD kernel to independent order12
volume integration (rtol1e-9), check scalar Biot-Savart reduction, uniform mu
scaling, observer derivatives and native mu derivatives against finite differences.
A native nullspace current with zero normal jumps on every internal and external
face checks interior contact and curl H (rtol1e-10,atol1e-12), Ampere defect
below1e-12 and div B below1e-17. Arbitrary nonconservative currents retain their
nonzero defect. Face observations, non-SPD coefficients and invalid pair budgets
reject. Existing scalar current diagnostics remain unchanged and unqualified.

GPU1 targeted run /tmp/gsp-tensor-magnetostatic-v1.py, session4321 TERMINAL0:
5 passed, no failures/errors/skips, 5.08859973680228s; all801 source hashes stable.
The receipt head dc4fa77 is the generic runner's Bench periodic-integration head;
import assertions and provider hashes bind Maxwell <scratch>/witwin-maxwell-tensor-hb
on parent64e9d19. No Bench code changed.
Runner SHA256: 46c5e9d9c1ded18cf6001f9bcbcbe07e0a8200616709f85b61488112c3873888.
Receipt /tmp/gsp-tensor-magnetostatic-v1/receipt.json SHA256:
6e351c3a6bca1e62ccab93aa24b3e48ee992bc25b67045a5ede4e01e19327669.
Independent read-only audit found no confirmed P1/P2, checked signs/contact/
defect/divergence, terminal counts and all801 hashes. pair_batch limits individual
panel-pair batches only: all faces and the autograd graph remain allocated.

This is a homogeneous particular-field foundation, not regional interface/HB
recovery or production qualification. Weak face-current balance does not certify
distributional continuity. Complex tensor DC and magnetic conductive-limit
channels from the unchanged original mixed cohort remain unsupported. GSP01
stays open and all original final acceptance thresholds remain unchanged.


### 2026-09-28 regional magnetic volume contrast bridge

MagneticVolumeContrastResponse now solves an internal dense P0 magnetic VIE:
H=H_J(mu_ext)+H_background+K_ext P, P=(mu_region-mu_ext)H, where K_ext is
minus the gradient of analytic single face layers for -div(P). All bodies
require bound tetrahedral volumes in region order. The cell-center matrix is
I-K_ext*(mu_region-mu_ext). The global current particular field preserves its
Ampere defect; adding scalar polarization gradients does not remove that defect.
The formulation does not use potential jumps or cut-surface periods and does not
assume simply connected bodies. Topology-specific controls are still outstanding.

This volume formulation includes the anisotropic volume correction missing from
a direct reuse of scalar-mu Biot-Savart interface layers. It is a bridge for
regional coupling, not a replacement for the final surface-only linear path.
P0 magnetization and constitutive H are matched at centroids only. Queries return
H and physical constitutive B=mu_region H; pointwise div B and exact interface
traces are not claimed. Cell refinement, power/physical controls, conservative
current discretization, public HB recovery and exact persistence remain required.
No scalar magnetic conductive or complex tensor DC admission was relaxed.

The retained dense matrix has an explicit byte budget; assembly temporaries,
autograd history and total peak memory are not bounded by that retained budget.
Owned currents, coefficients, matrix, collocation fields and induced sources
reject in-place mutation. Native permeability derivatives remain connected.

Targeted GPU1 runs, all with803 stable provider/test source hashes and no skips:
- v1 session69193 TERMINAL0:2 passed,7.303301330655813s; matched-medium actual
  current, independent native-mu finite difference, and tensor sphere refinement.
- v2 session38325 TERMINAL0:3 passed,8.116277249529958s; adds actual two-body
  sources, all-region superposition, cross-body action against independently
  applied StaticTensorPolarization, reversed ownership and finite1e308 background
  rejection. Also covers residual-diagnostic overflow fixes from independent audit.
- v3 session80921 TERMINAL0:1 passed,5.2581278290599585s; only the multi-body/
  extreme-value test after the final summed H/Ampere-defect finite guard. No
  numerical assembly or solve equations changed after v2.

Selected anisotropic sphere internal-point errors for levels0/1/2 were
0.04285361233,0.00736449044,0.00057241200. The last is below0.005 only for
this single selected control; full physics and interface qualification remain open.

The audit identified P2 residual norms/threshold overflow and unchecked multi-body
source diagnostic sums. Stable norm evaluation and explicit nonfinite rejection
now cover RHS, matrix, solution, residual, threshold, jump diagnostics and summed
H/Ampere defects. A finite extreme drive control rejects unrepresentable diagnostics.

Runner /tmp/gsp-magnetic-volume-v1.py SHA256: c5b5de76ee12c8d7d337b47ee0342afb80af5d79eeb63d6489e0e5d3c2274876.
Receipt SHA256: f10c58b0b68f95a9e222219a3293f366c1decb1449435fdc017d36a56383b45d.

Runner /tmp/gsp-magnetic-volume-v2.py SHA256: e45086a68699984ed9a9323f91cf2e08210080aa3d825ddc18f49bef55b946be.
Receipt SHA256: 37a58b22e79065b1475453f0965f186fefef1660dcfc98d9c3a26044e53e6e9b.

Runner /tmp/gsp-magnetic-volume-v3.py SHA256: 106bc7952c080147843aa00296d756008928643fcd4d5fa1ae13d105afad4522.
Receipt SHA256: 25fb6a2ec89ab3781e12c756478c1a1215e604be98db8afcbdf60f86f5595407.

Receipt head dc4fa77 belongs to the generic runner Bench periodic-integration;
provider import assertions and hashes bind Maxwell tensor-hb parent218e011.
Bench production sources unchanged. GSP01 remains OPEN, Goal ACTIVE.

Independent final review closed P2 and found no new confirmed P1/P2. It verified
v2/v3 terminal runner/XML/log evidence, each803 before/after hashes and exact v3
current-source match. Scope remains internal P0 collocation, unqualified.


### 2026-09-28 mixed HB magnetic current recovery

The internal mixed preparation accepts optional dc_magnetic_volumes, an ordered
volume for every surface body. This is separate from scalar AC nonlinear/current
source volumes: linear tensor bodies need a DC magnetic volume but do not become
AC volume sources. Any region shared with the selected HB volumes must use the
same volume object, so no implicit grid transfer or cell reorder occurs. Analytic
magnetic coefficients are sampled from the same body/exterior material DC
channels already checked by mixed admission; no imaginary part is discarded.
The existing matrix byte and pair budgets govern magnetic recovery too.

Magnetic preparation parameters and bindings participate in the preparation
signature. After ordinary HB recovery adds the ohmic current exactly once, its
final DC current is split in selected binding order and mapped by region into
the all-body magnetic volumes. Non-source linear bodies get zero induced current.
The recovered state owns MagneticVolumeContrastResponse, including its solved
magnetization and mutation checks. HarmonicFieldDecoder.query_dc_magnetic returns
regional H/B and explicit source Ampere-defect diagnostics without another HB or
magnetic solve. Recovery itself still assembles and solves the magnetic system.

This is an optional internal path only. Public Scene configuration, separate
magnetic volume persistence and schema support remain unfinished. The current
tensor state/bundle codecs explicitly reject this additional state instead of
silently dropping it. Existing nonmagnetic mixed current snapshots are unchanged.
General conductive magnetic DC, complex tensor DC, distributionally conservative
current discretization and pointwise magnetic interface qualification remain open.

The targeted new control uses actual dense two-harmonic AC operators, a linear
scalar-limit tensor body and a refined scalar conductive/TPA body. It verifies
nonzero final induced current, exact region/current mapping, all-region solve-free
magnetic queries, missing/wrong volume rejection, snapshot rejection and mutation
checks. It does not certify general anisotropic scattering or a compressed H2
backend; two harmonics refers only to the frequency truncation.

v1 session2877 TERMINAL1:1 failed,47.7334690094s. Recovery reached regional
queries; independently repeated CUDA accumulation differed by3.1554436209e-30
(maxrelative4.74767909e-15), violating an inappropriate bitwise query assertion.
Only that query assertion changed to rtol1e-12/atol1e-28; exact current equality,
HB solve tolerances and all physical acceptance thresholds are unchanged.
The independent audit verified v1-to-v2 source inventory changed only this test.

v2 session53252 TERMINAL0:2 passed,0 skips,57.31752638053149s,804 stable
source hashes. Includes new magnetic recovery and the old current snapshot/bundle
round trip. Independent final audit found no confirmed P1/P2, verified all804
current hashes, terminal runner/XML evidence and retained v1 failure.

Runner /tmp/gsp-mixed-magnetic-recovery-v1.py SHA256: 96c92891053c82f39e368d3f1ded2fb369442267ead060968b28003f374ce994.
Receipt SHA256: 27ce83cf25b922558d24096341ca5317371dbf9ec8d00e0eede5655718a55589.

Runner /tmp/gsp-mixed-magnetic-recovery-v2.py SHA256: d47d89d5c83fd01c2fa627ea091cf3e5474e096be9115387505541df11f15a43.
Receipt SHA256: 3a2125a759f174d12da11f512dde7ee6e56d7238337ff07dec9b049dce756feb.

Runner receipt head is unchanged Bench periodic-integration dc4fa77; import
assertions and provider hashes bind Maxwell tensor-hb parentb7c57cf. Bench has
no changes. GSP01 and all original final physical gates remain OPEN.


### 2026-09-28 exact magnetic contrast response component

The separately versioned regional-magnetic-volume-contrast-response-v1 codec
owns the actual matrix, source RHS, background, cell fields, contrast, currents,
polarizations and SPD coefficient/factor/determinant data. MagneticVolumeContrastResponse
now retains source_field and includes it in mutation tracking. Restoration binds
an independently supplied exact surface system and ordered volume geometries,
clones arrays, restores exact factors and checks the constitutive partition and
stored-system residual without panel integration, source evaluation or solving.
The external owner SHA binds matrix/RHS provenance: restoration does not recompute
physical source provenance or certify discretization accuracy. Saved receipt
qualification remains false. Bound geometry must be reconstructed separately.

Tests use independent surface/volume objects, actual torch.save/load with weights
only, and guards on response construction, source evaluation, panel integrals and
linear solve. Arrays and factors match exactly. Regional field queries compare
at numerical tolerance, not bitwise arithmetic replay. Corrupted owner bytes,
contrast, solved fields, matrix, kernel factors, inadequate restore budget and
in-place mutation reject. Payload mutation does not mutate restored state.

GPU1 retained evidence, each806 frozen source hashes stable:
- v1 session64508 TERMINAL0:4 passed,8.297572849318385s: one-body snapshot and
  existing three magnetic contrast controls.
- v2 session35727 TERMINAL0:1 passed,5.2016324726864696s: newly added two-body
  independent restore, exact regional currents/polarizations and all-region fields.
- v3 session59799 TERMINAL0:1 passed,5.293833333067596s: same two-body node after
  audit budget fix, additionally checks the actual budget passed by save self-check.

Independent audit identified P2: save self-check implicitly used the default
256 MiB restore budget, rejecting larger matrices admitted by explicit original
budgets. Self-check now uses the actual retained matrix byte count. External
restore continues to enforce its caller budget. This is not a total peak-memory
bound, nor evidence of large-matrix resource qualification.

Runner /tmp/gsp-magnetic-snapshot-v1.py SHA256: 95b68a29c3a8ef1e50d3d82d733d99bdac2e909612150e5afc6ae58e1e2ba65a.
Receipt SHA256: 5149df89d2e346272816f606eb967bcd74253d7df13f1c461dec78320b269366.

Runner /tmp/gsp-magnetic-snapshot-v2.py SHA256: 176e585f50e729dec4c835f459e98dc950e8d981b817505dcb4bc051ab3f031a.
Receipt SHA256: 0f6ff556374059b999233d9b4bdf716c0c9b9552cad2b996ffe428b6fe361736.

Runner /tmp/gsp-magnetic-snapshot-v3.py SHA256: 03274e0d26bf5f4cd1faa5265e22d326aac65805152c7f1c8bcf220ca0f4768f.
Receipt SHA256: 1c451326c966d15afcb2b4907fd9e1dab915e06b65b0fd93832d95b72ec9e631.

Receipt head dc4fa77 belongs to unchanged Bench periodic-integration; import
assertions and provider hashes bind Maxwell tensor-hb parent8dd2c1e. Full mixed
HB state/bundle integration and public Scene/Result/Bench magnetic persistence
are still pending; current mixed snapshot rejection remains explicit. All
original material, physical, resource and population acceptance gates remain.

Final independent review closed P2 with no new confirmed P1/P2, verifying v3
terminal evidence and all806 current hashes. v2 alone did not cover the later
budget repair; v3 explicitly covers that repair.


### 2026-09-28 mixed magnetic HB exact state and bundle

New versions mixed-tensor-current-magnetic-harmonic-owned-state-v1 and
dense-mixed-tensor-current-magnetic-harmonic-bundle-v1 integrate the magnetic
response codec. Existing SPD and current-without-magnetic versions are unchanged.
Prepared identity binds all magnetic volume geometry, analytic coefficients and
budgets in addition to original AC/DC maps. The nested magnetic response is bound
by its own digest and the containing harmonic state digest.

State restore checks that magnetic coefficients and pair policy match preparation,
that no external DC magnetic background was added, and that each region's current
is exactly the final recovered volume current (zero for bodies without AC source
volumes). The magnetic receipt must match the harmonic receipt. This retains the
existing source/ohmic partition checks and adds no extra ohmic contribution.

Bundles store independent magnetic volume records for every body, in region order.
Any region also present in selected HB volumes must have identical geometry and
reuses the very same restored volume object. Remaining tensor-body DC volumes are
constructed separately. Exact magnetic matrix, RHS and field arrays are restored
without evaluating the particular source, panel kernels or magnetic linear solve.
Decode geometry and native small SPD factor validation remain allowed operations.
The stored system is independently owned; payload mutation does not change it.

The actual two-harmonic dense control now saves/loads a self-contained bundle,
restores with source evaluation, panel integration, magnetic construction, AC
operator construction, HB current solving and torch.linalg.solve forbidden, then
checks electric and magnetic regional queries, exact magnetic arrays, shared
volume identity, changed magnetic coefficient and restore budget rejection.
Existing nonmagnetic mixed current snapshot/bundle regression is included.
Public Scene/Result and Bench magnetic persistence remain separate work. Neither
this codec nor the underlying P0 volume bridge grants physical qualification.


GPU1 v1 /tmp/gsp-mixed-magnetic-bundle-v1.py session45516 TERMINAL0:
2 passed, no skips,57.18565704859793s,806 unchanged source hashes. Independent
initial audit found no confirmed P1/P2 and requested direct new-branch negatives.
v2 session71629 TERMINAL0:1 passed,6.552301097661257s,806 unchanged source hashes.
Only the tracked test helper changed after v1; production sources did not change.
The helper modifies nested magnetic current, adjusts its current-jump diagnostic,
reseals nested response/state/outer bindings and is rejected by the cross-check
against actual total_volume_current. A current-only version downgrade rejects
because magnetic bundle inventory remains present.

v2 reuses the actual v1 saved bundle instead of repeating AC/HB preparation:
/tmp/pytest-of-xingyu/pytest-2893/test_mixed_hb_magnetic_volume_0/magnetic-hb.pt,
SHA25614c933c55a581910b50278a3ac2543873efc0575f2a254b9b7163d75d51b704a.
Temporary pytest wrapper /tmp/test_magnetic_bundle_negative_replay.py SHA256
ce118f08fbf58bc95e54c527a65f95dcb01fa72c0e68eb9ed018980e9d1b084a invokes the
tracked helper and verifies the bundle SHA before loading. The same helper is
part of the full repository test for future runs. These consistency negatives
do not claim protection against replacement of every independent trusted owner.

Runner /tmp/gsp-mixed-magnetic-bundle-v1.py SHA256: 0fef811258eca337694f31de6d91688d37eee0f4c7d735895718b127e6f48371.
Receipt SHA256: 2de9e515bb4c3be69903282befc52401582f77102504c3d852b14519bfd76c08.

Runner /tmp/gsp-mixed-magnetic-bundle-v2.py SHA256: 9129e4b4dff603708fcc1fd9da2cb5127e4f1e1e692b374cd1da4f50aae5c3ec.
Receipt SHA256: 4a41357e0ebcbe6813064ac407db1332145e26bc0cc01fc03f4893a47d11b8e6.

Receipt head dc4fa77 remains the unchanged Bench periodic-integration import;
provider imports/hashes bind Maxwell tensor-hb parentbc72e73. No Bench code or
public persistence schema changed. GSP01 and original qualification gates remain.

Final independent audit verified v2 terminal evidence, all806 current hashes,
actual artifact/wrapper/runner SHA and test-only changes since v1. Both new
negative branches are covered; no confirmed P1/P2 remains.


### 2026-09-28 public mixed DC magnetic Result path

HarmonicBalanceConfig now accepts optional dc_magnetic_volumes, a nonempty tuple
covering every surface body in increasing region order. Regions shared with
volumes must reuse the exact same volume objects. For example, with a linear
first body and a current-carrying second body, volumes=(current_volume,) and
dc_magnetic_volumes=(linear_volume,current_volume) keep DC magnetic geometry
separate from the scalar AC source-volume contract. The option is supported by
the dense free-space mixed runtime selected through dyadic_integration. The
ordinary scalar runtime rejects it explicitly; no silent option discard occurs.

The new native schema23 kind coupled_sie_vie_hb_mixed_tensor_current_magnetic
routes through the independent magnetic bundle. Restored public configuration
reuses the stored magnetic volumes; scalar numerical policy payloads exclude
geometry objects. Schema/state consistency is checked. Result.load resamples the
Scene's actual analytic DC magnetic coefficients and compares every body plus
exterior, in addition to existing AC/material/source/geometry/current checks.
An isolated changed magnetic DC sample with unchanged AC and Scene signature
rejects. Point monitors mark magnetic_available=True and retain actual DC H
components for magnetic states; schema22 remains unavailable as before.

The selected public control uses the same two-body scalar-limit tensor/scalar
conductive/TPA Scene. It covers run, DC H query, monitor components, native
save/load and exact owned arrays. Load runs under guards forbidding panel/source
evaluation and linear/HB/magnetic construction or solves. Repeated field values
use numerical tolerance; stored monitor/state values compare exactly. This is
public integration and persistence evidence, not general magnetic physics.

GPU1 v1 session99299 TERMINAL0:2 passed,29.94428558088839s,806 stable hashes:
new public magnetic and original current schema22 controls. A supplemental
admission test verifies incomplete volumes, reused-object mismatch, wrong order,
empty tuple and unsupported scalar runtime rejection before AC assembly.
v2 session63087 TERMINAL1:1 failed,3.8245036294683814s because the test tried
assigning to frozen Structure.material. v3 session52709 TERMINAL1:1 failed,
4.184964926913381s because the corrected Scene still retained dyadic_integration
and therefore selected the mixed runtime, contrary to the test's scalar-route
expectation. v4 sets dyadic_integration=None for that rejection control:
session80347 TERMINAL0:1 passed,4.100615171715617s,806 stable hashes. Production
code did not change after v1; only the new admission test changed. All failed
receipts are retained, no physical threshold or material admission was relaxed.

Runner /tmp/gsp-magnetic-public-v1.py SHA256: 248e84311826fee26525c659959b196d668ac081842d237c8cea19b22e4fca21.
Receipt SHA256: f03f0d27726b724da39bc58713470ed36535e81e15ef4f89dde7761c5d5316b1.

Runner /tmp/gsp-magnetic-public-v2.py SHA256: 737759fe3d27ac877e22e865a80799cd444f8f90791e07d8ad59c7bceeaa2fdb.
Receipt SHA256: 81ab613916e327211060d49e54d0ad52720a0c61bbf87da7b7934ce8a9aabd8d.

Runner /tmp/gsp-magnetic-public-v3.py SHA256: e6c2c91195136d401893617b1d19f6db3fbc4acc6af4030a15110231864357b7.
Receipt SHA256: 7b162437e766fcb3171f5a325bc49f83b0a7598c4022942e4783b8aac96cab56.

Runner /tmp/gsp-magnetic-public-v4.py SHA256: 28e2ad98aaae8861c94fad2975cbacaed0383e59d27d707d5178be3d090e2650.
Receipt SHA256: e1c8017679a54b444f54eb11657fb49a133f6183830dff999b3ef2ce10729c25.

Provider imports/hashes bind Maxwell tensor-hb parent941e43b. Receipt head
dc4fa77 is unchanged Bench periodic-integration. Bench schema23 archive support
is next; full populations and original physical/resource gates remain open.

Independent engineering audit found no confirmed P1/P2. Final supplement
verified v4 terminal evidence and all806 current hashes, test-only changes
since v1, early admission failures before assembly, and preserved v2/v3 failures.


### 2026-09-28 magnetic interface failure localization

Correction to earlier checkpoint language: FaceCurrentBalance uses CR face-mean
basis functions. For P0 currents, its weak_current vector equals the oriented
individual face-flux sums, including exterior zero-flux conditions. Existing
 tests/mom/test_face_current.py independently checks this identity, a nonlinear
TPA time-domain current oracle and Ampere consistency. The older nodal weak
balance counterexample must not be attributed to CR. Finite solver tolerances
and the final HB recovered current still need quantitative checks; no blanket
continuity certification is inferred. No already covered CR tests were rerun.

The expanded sphere control uses homogeneous off-diagonal real-SPD mu, uniform
background H,51 interior and51 exterior points across17 fixed directions and
three radii each. The independent sphere oracle uses
H_in=(mu+2mu0 I)^(-1)(3mu0 H_bg) and exterior dipole magnetization
M=(mu/mu0-I)H_in with moment(4pi a^3/3)M. The reference geometry is the smooth
sphere; surface approximation error remains in field comparisons.

Twenty-four selected actual polygonal faces are checked at two one-sided offsets,
with linear Richardson extrapolation. Normal B jump is normalized by mu0|H_bg|;
tangential H jump by |H_bg|. These are sampled estimates, not a certified trace
error bound. The original threshold0.005 applies to both maximum H errors and
interface jumps. Every combined physical control FAILS:

| Surface level | Volume refinement | Cells | Interior H max | Exterior H max | Normal B jump max |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | 80 | .05695705 | .04383290 | .14088381 |
| 2 | 0 | 320 | .00550141 | .01092975 | .08933641 |
| 3 | 0 | 1280 | .00081887 | .00281949 | .03612442 |
| 2 | 1 | 2560 | .00468839 | .01106457 | .08200782 |

Tangential H jumps are below3.1e-14 in these samples. Level3 field values pass
only the selected H-point checks; its3.61% normal B jump fails the0.5% gate.
Uniformly refining the level2 volume keeps the same polyhedral boundary geometry
and gives only a modest normal-B improvement. Thus simple volume refinement does
not establish the interface behavior needed for qualification. Both tensor
magnetic current source and interface discretization must remain distinguished.

The collector test asserts finite observations and field refinement trends,
while explicitly storing selected_control_passed=False and
physical_qualification=False. A pytest pass denotes successful evidence collection,
not physical acceptance. No numerical production code changed in this stage.
v1 session53774 TERMINAL0:collector1pass6.381662270054221s; v2 session40383
TERMINAL0:additionalvolumecontrol1pass10.366350307129323s. Each807 source hashes
is unchanged within its own run. v2 changes only the test to select the additional
uniform-volume refinement, leaving v1 evidence intact. The512MiB retained-matrix
budget is an explicit probe configuration, not a change to runtime defaults or
resource acceptance: v2 observed peak allocated memory2398363648 bytes. Peaks
include CUDA allocations retained during each probe, not independently isolated
solver workspace bounds.

Next formulation work should use regional SPD potential/conormal transmission
with known particular-current correction. For a reference current field H_J
satisfying div(mu_ext H_J)=0, the correction potential obeys
 div(mu grad(phi))=div((mu-mu_ext)H_J), with H=H_J-grad(phi).
Uniform background can be carried by the existing regional uniform-drive solve.
A sampled induced polarization (mu-mu_ext)H_J can provide a volume source for
that interface system; its approximation and boundary trace consistency must be
verified, not assumed. This is a proposed next implementation, not a completed
capability or qualification claim.

Runner /tmp/gsp-magnetic-physical-v1.py SHA256: 46121ff381766483e444968ce920d5465b736dbbca0fd51f76162d2e5bebc89d.
/tmp/gsp-magnetic-physical-v1/receipt.json SHA256: c4a2e8514c35c4993d8a07562bc6b7d831ede3849d394ad0ca7d27fa3046f583.
/tmp/gsp-magnetic-physical-v1/physical.json SHA256: 2bbb9b9a70f6c066c7dcf177b9617e6e2fd2963f4d4e4b36b242891a9970f047.

Runner /tmp/gsp-magnetic-physical-v2.py SHA256: 80ad5276511f15f7a521ece745e7a144985e3cb1d33ba651334d0b5e66480c36.
/tmp/gsp-magnetic-physical-v2/receipt.json SHA256: 484a7a9a34da4719fb5459d016a06deb2e845b124e95f2bac78eb1773685cc0c.
/tmp/gsp-magnetic-physical-v2/physical.json SHA256: c355a70a94e54403a67af3bd7446709b892d60d535046c847e139607712723fd.

Independent audit found no confirmed P1/P2 in the oracle/evidence process,
verified both807-hash runs and explicitly concluded physical control failure.
It confirmed CR correction and limited the Richardson interpretation to sampled
estimates. Original GSP01/population/other physics failures remain open.


### 2026-09-28 surface comparison and analytic linear double layers

A same-point comparison used the existing regional P0 potential/conormal surface
operator for the uniform magnetic sphere, before attempting to replace the
public volume path. It did NOT improve interface continuity. Surface levels1/2/3
normal B jump maxima were .20175187/.13729094/.10583116; tangential H maxima
were .03909881/.02081371/.00940610. At level3 interior/exterior H maximum errors
were .00473706/.00311686. All combined0.005 physical gates fail. Thus the proposed
regional P0 correction is not yet an acceptable replacement. Public routing was
not changed. The comparison script uses exactly the prior51+51 field locations
and24 polygonal face observations with two-sided Richardson estimates.

The new StaticTensorGreen.panel_linear_double_layer integrates physical P1
barycentric basis functions against the source conormal derivative. Let
r=x-y, d=(x-y0).n, lambda_a(y)=lambda_a(x_parallel)-grad(lambda_a).r.
For real-SPD mu, mu grad_x G=-c r/(r^T mu^-1 r)^(3/2), and d is constant
on a planar source triangle. Therefore
K[lambda_a]=lambda_a(x_parallel)K[1]+d grad(lambda_a).mu grad_x S[1].
This uses the existing analytic constant-panel potential gradient without extra
quadrature. Columns follow input triangle vertex order. Reversal changes the
conormal sign and permutes columns; their sum is the original constant double
layer. No half-jump is added. Existing exact PV mask and edge rejection apply.

The new tests compare to order16 independent physical point-kernel cubature,
including observer derivatives and material variations within the symmetric SPD
contract. They also cover constant-basis partition, orientation, coefficient
scaling, translation and principal-value admission. This is the integral basis
needed for continuous P1 surface potential; it is not a completed interface
solver, public API change or physical qualification.

GPU1 surface comparison session36300 TERMINAL0:collector1pass6.4394844193s,
807 frozen source hashes unchanged. Its physical results are failures as above.
/tmp/test_magnetic_surface_comparison.py SHA256:
e2474b2b3febfa438694f5be384c3a113b40a6867bb230bce41fdccb146cdf2b.
The helper lives outside the provider hash inventory; this explicit SHA records
its bytes separately from the runner and provider hashes.

Linear-layer v1 session18066 TERMINAL1:4 old Green controls passed,2 new controls
failed,5.6445194539s,808 stable hashes. The material-gradient test differentiated
nine unconstrained matrix entries, including directions outside symmetry, while
the layer identity is for SPD tensors; the PV test supplied a boolean instead of
the existing CUDA per-pair mask. Only tests changed: symmetric parameterization
A=(parameter+parameter.T)/2 and an explicit CUDA boolean mask. No production
formula or old Green checks were changed. v2 session75630 TERMINAL0:2 new tests
passed,4.6974749062s,808 stable hashes; old controls were not repeated.

Runner /tmp/gsp-magnetic-surface-comparison-v1.py SHA256: 86b990d087b87367c36d1c3b98bfee2428e27f16fca4ab6fc3958ac217f1671c.
Receipt SHA256: 3c3fffca2cf86b8c3691326afcc40eee60c939b6824f5a171231108179d8bfeb.

Runner /tmp/gsp-static-linear-layers-v1.py SHA256: dfc2f2ae634e702839da6ab8eaa578fd50c6be52404dc4785fd96ab558ea5338.
Receipt SHA256: 100a4353367e6657b72e6083f707f8dee5be2712caa1ad47f1d7ee4229f415b7.

Runner /tmp/gsp-static-linear-layers-v2.py SHA256: 1d330b0cd1f8ba819ecbed5885a7c4030da31ace92e6040ee4d24f3225254e63.
Receipt SHA256: 3783ab01f196d1efc1e15c89e47ee231c1b03f238b7f992ab604ee7b0bb77ea6.

Next work is continuous-potential interface discretization using these exact
linear moments, with the same independent physical gates. Existing P0 failures,
CR correction, unchanged cohort and full final acceptance obligations remain.

Independent final audit found no confirmed P1/P2, checked the analytic moment
sign/basis ordering and no-jump convention, verified v2 terminal/current808
hashes and test-only changes, and confirmed the P0 comparison remains failed.


### 2026-09-28 continuous P1 potential interface candidate

StaticTensorInterfaceOperator accepts potential_basis='p1': each body's global
vertices own continuous affine potential coefficients; faces retain P0 conormal
flux. Exact analytic linear double-layer moments scatter to body-owned nodes.
Exterior and interior equations retain their existing signs and source terms,
with face-center averaging for the half-jump. The 2F by (V+F) system uses native
CUDA unweighted least squares; a nonzero equation residual is expected and is
reported without claiming discretization accuracy. Retained-matrix budgets do
not bound peak workspace. P0 remains the default and its canonical metadata is
unchanged. Surface field decoding differentiates the same regional potentials.

Controls include two-body constant and homogeneous affine solutions, material
directional finite differences within SPD, drive homogeneity, observer gradients,
nonuniform induced volume polarization with internal charge and boundary flux,
exact trace recovery, mutation guards and explicit unsupported-route rejection.

Independent audit identified two P2 composition issues before commit: old P0
codec needed reconstructed derived potential indexing, and the HB cell-response
preparation assumed face-count potential coefficients. Both are fixed. P0
restoration reconstructs derived indexing without assembly or solve; P1 snapshots
and HB cell preparation explicitly reject unsupported potential basis. No public
route or persisted schema is silently changed. Existing targeted snapshot and
cell-preparation controls pass.

The same original 51 interior + 51 exterior magnetic sphere points and 24 actual
polygon faces give the following relative maxima for levels1/2/3:
interior H .02038488/.00565990/.00137550; exterior H
.04557922/.01183203/.00304736; normal B jump
.08143048/.04532566/.02210876; tangential H jump
.05746269/.02842119/.01424259. Every combined .005 gate FAILS.
Level3 normal continuity improves compared with P0 surface .10583116, but
tangential continuity worsens compared with P0 surface .00940610. This candidate
is not an acceptable public replacement. Two-height Richardson estimates are
sampled evidence, not certified trace bounds; collector pass is not physics pass.

GPU1 evidence (each 809 frozen provider/source hashes unchanged):
- v1: 6 passes in 6.705197904s, new P1 controls plus affected P0 interface controls.
- physical collector v1: 1 pass in 7.144721468s; all physical rows remain failed.
- v2: 8 passes in 5.874618803s, P1 controls plus five existing exact snapshot tests.
- v3: 4 passes in 6.139521913s, changed nonuniform-source/admission control plus
  three existing cell-preparation tests. Prior unaffected checks were not repeated.
Only metadata/codec admission and cell-preparation rejection changed after the
physical collector; the tested P1 numerical solve and field recovery are unchanged.

/tmp/gsp-static-p1-interface-v1.py: SHA256 c9bbf1550ba1780cfc3aac8df80cb61e428027a7728078abafb9d733713aa33e; receipt SHA256 a66fe287eac3e5f9d47478ff534b0414d7f876c3298719736ff0902e3c5ed673; 809 stable hashes.

/tmp/gsp-magnetic-p1-surface-comparison-v1.py: SHA256 73829fc2709efb7068ab54f26a98490bbbbcace2f482b67ef550f7b6732dfaed; receipt SHA256 35c36373f3e15d6945d08fecfd65910e7e7510ccc6c1e0962d9ca3dc0d3bb82a; 809 stable hashes.

/tmp/gsp-static-p1-interface-v2.py: SHA256 f83b715638d1dc14afe8ef2beadbe33b9710e6427d54809091ce8b57da0958ea; receipt SHA256 80580526bc7173cc61d1f6cb87a7d3e6f0ee4c0796c20506d261cddc897e1594; 809 stable hashes.

/tmp/gsp-static-p1-interface-v3.py: SHA256 3dbd0251a7cab80eb8307bbebad1e0906e97d7fe0ec203822c3ec6d8c4c5fc74; receipt SHA256 da2800a035e01b11c5b57895321061e160b901ae5a3224a5ddb0af55dd19c2ab; 809 stable hashes.

Physical probe outside provider inventory: /tmp/test_magnetic_p1_surface_comparison.py, SHA256 48030f445c1104456e33c29526c65d59479e6eb37a6494087df88c47716f54cd.

Next work must resolve the residual interface inconsistency, not infer production qualification from improved fields or least-squares residuals. Original GSP01 failures, complex/magnetic-conductive DC cohort admission, full population and final acceptance obligations remain open.

Independent final audit closed both identified P2 issues with no remaining confirmed P1/P2. It verified v2/v3 receipts, XML, runner hashes and the current v3 inventory. The P1 candidate remains physically unqualified and explicitly unavailable to HB preparation or storage.


### 2026-09-28 higher-order static layer moments and unresolved transmission

Added P1 barycentric single-layer and P2 Lagrange conormal double-layer moments
to StaticTensorGreen. P2 ordering is vertices0/1/2, then edge midpoints01/12/20.
Near-panel P1 single layers use the transformed-plane identity
 integral r_parallel/R = -sum_edges outward_m integral_edge R.
P2 conormal moments use
 K[lambda_a lambda_b]=lambda_b(x_parallel)K[lambda_a]
                     +d grad(lambda_b).A grad S[lambda_a].
The P1 gradient is differentiated natively; higher material/observer derivatives
are retained when enabled. Shared exact face/PV admission and edge rejection
remain. No trace jump is silently added.

An independent far-distance probe exposed severe cancellation in the initial
analytic representation: at distances1/10/100/1000, P1 single relative errors
were 1.01e-13/3.97e-12/5.96e-10/1.17e-6 and P2 double errors were
1.86e-12/3.35e-9/1.94e-5/.2815285. The collector passed only finite checks;
these errors are retained as failed numerical evidence, not qualification.

The fix uses positive order16 CUDA Gauss-Duffy quadrature when the maximum
transformed source radius / target-center distance is <=1/32. For
|s|<=65/1024, the inverse-radius binomial series through k14 multiplied by
P1/P2 is polynomial degree<=30, which the product rule integrates exactly.
For exponent -3/2 the remaining coefficient series is bounded by
t^15*(16-15*t)/(1-t)^2=1.8777518456e-17; the integral and positive-quadrature
remainders each contribute this scale. This is an ABSOLUTE truncation bound
scaled by kernel/basis magnitudes. It excludes floating-point error and does
not certify derivatives, near-zero relative error, or switch-boundary smoothness.
The far double-layer density uses constant physical panel height directly,
without differentiating cancellation-prone moment differences. Pair chunks of
256 bound immediate quadrature arrays, not total retained autograd memory.

Final five targeted tests cover independent point cubature and SPD material/
observer derivatives, orientation/partition/scaling, translation, face symmetry,
general anisotropic single-layer conormal jump and P2 potential jump, explicit
PV and edge admission, empty/invalid queries, no_grad behavior, and distances
1/10/100/1000 with independently normalized per-pair gradient comparisons.
All five pass; old unchanged P0/P1-double tests were not rerun. There is no
claim of complete extreme-scale or float32 accuracy qualification.

Three one-body uniform-drive diagnostic discretizations were compared at the
UNCHANGED 51 inside + 51 outside sphere field points and24 polygon faces.
They are external probes, not integrated production operators or public HB.
Potential coefficients are continuous nodal P1/P2; flux coefficients are
discontinuous face-local P1. Three or seven interior collocation points per
face form an unweighted overdetermined least-squares system. Final level3:

| potential / face points | inside H max | outside H max | normal B jump | tangent H jump |
|---|---|---|---|---|
| P1 / 3 | .00098801 | .00291699 | .02191360 | .00333716 |
| P2 / 3 | .00181069 | .00358497 | .25855304 | .00126574 |
| P2 / 7, stabilized kernel | .00097055 | .00281007 | .01765467 | .00404262 |

ALL combined .005 physical gates FAIL. P2/3 residual2.79865e-5 is smaller
than P2/7 residual3.09096e-4 yet has much worse normal continuity. Small
algebraic residuals and selected field passes cannot substitute for interface
physics. P2/7 was rerun only because the underlying numerical kernel changed;
normal/tangential results were unchanged to rounding. No successful production
replacement has been demonstrated. Two-height Richardson sampling is not a
certified trace bound. Final P2/7 observed peak3410300928bytes is diagnostic,
not resource qualification or a default budget change.

GPU1 evidence: initial single-layer2pass5.993180638s; initial P2 double1pass
5.947137573s; anisotropic/empty plus far collector2pass5.025535732s; final
stabilized kernels5pass6.990887905s. Physical collectors: P1/3 1pass8.834305932s;
P2/3 1pass15.415797872s; P2/7 v1 1pass26.372456596s and v2 1pass28.963001815s.
Every receipt has809 stable source hashes. Collector pass NEVER means physics pass.
Independent audit found no confirmed P1/P2, verified signs/basis, bounded far
branch, final five-node evidence and current source inventory.

/tmp/gsp-static-linear-single-v1.py: SHA256 73376fc2effce416996184f666620f4a21db5d98630bb53e6449461a623cd167; receipt SHA256 e87538645d3d62785b95e0a7b2e581cf162e886b2fd91df1c41ad8be0f8b1a83.

/tmp/gsp-static-quadratic-double-v1.py: SHA256 9090f31dd55033cb076223fa5b74893070ad884f7321979757d6c7cc185cbba3; receipt SHA256 aa696c3894dbc278c13ea747ae61e4549f1b42b1376b41a5475eb1e1e5cc8692.

/tmp/gsp-static-higher-layer-boundaries-v1.py: SHA256 6a1f42e980cd615e5b5fb43dd451bb9119bcd04808fe12d319b61cac2ed25c46; receipt SHA256 e60b836dab7ed489c73093f12e2f358bcb73663d8090a768c2c106822a2589e5.

/tmp/gsp-static-higher-layers-v2.py: SHA256 e940b01943c94fe57eedd356a44bd4738a24fa05ea5963e6303ebb4f13f5b65c; receipt SHA256 c56c484f5403ad49187fc9a981b4e7ad274effce0aea66b97edb1a6ba5afa20b.

/tmp/gsp-magnetic-linear-flux-comparison-v1.py: SHA256 091b57931ce79bf885cf3c089dbb3ba7f7594e243ef48e100c1d41062b261b6d; receipt SHA256 cf802a87af2797c6ac7d74dd75cf21896b59c6e870fd310acd71a0a65273c046.

/tmp/gsp-magnetic-quadratic-potential-comparison-v1.py: SHA256 a346698d42aa5a3a15a34846c9c6ee0ab351ca1dcc5d02fc97ada56cb8083764; receipt SHA256 70d400e17813c5977bc88815cd317a08f3bce54d80fd77dc710a1580032c077c.

/tmp/gsp-magnetic-quadratic-seven-comparison-v1.py: SHA256 a6d7e61b8a672d5b6540e9bde203d4870c8cbf96aa252a9dd01e66db38c849f8; receipt SHA256 1b85535bce9c309c66007cb6a453292d3da29418047e303e37bfa493f58130d3.

/tmp/gsp-magnetic-quadratic-seven-comparison-v2.py: SHA256 4e3b6205d52e5380f4178b215c79f6baf607487ce84b57bf012311baa63de65e; receipt SHA256 b023c82eed0d727a1faabd4f06f4f26e2c75670d161327499d608bf4e456fa8d.

External probe/helper files are outside provider source inventory and bound separately:

/tmp/p1_flux_probe.py: SHA256 fd212d45610be7b09579d8df0b27606c6a1a7bea754036a55aedffed4e39f713.

/tmp/p2_flux_probe.py: SHA256 aaf0b067fc8fa3641b26b255e53a0f20dae2e7193db90debd9c985278bf45c8d.

/tmp/p2_seven_flux_probe.py: SHA256 11e02d6571e33bc3aed28bce97b46dda7594df88d08b639fe12020c866dcc8c3.

/tmp/test_magnetic_linear_flux_comparison.py: SHA256 3085db07b473cf1181ca8a58666c98a66c938ebbb4603bc7524b2ca71e684f18.

/tmp/test_magnetic_quadratic_potential_comparison.py: SHA256 e065dc2fea09aeb701f6c37f432007fc661d26fb7bab21ab28192e99a2da24a1.

/tmp/test_magnetic_quadratic_seven_comparison.py: SHA256 daa33b5cfe48765f83ac011f47fdc95db66523bc24476c400b7a3519aca3141a.

/tmp/test_static_linear_far_probe.py: SHA256 cd7f3a12ddd69b836d0797f1029d44d99af16893604cf20ab0438b5576520d19.

Next investigate a boundary formulation that directly balances normal flux,
rather than treating higher potential order or increased potential-equation
collocation alone as the missing remedy. A possible diagnostic is separate
regional single-layer densities with potential and conormal transmission
equations, including the exact conormal half-jumps. This is not implemented or
validated yet. Any collocation-based improvement also needs off-collocation
face samples to avoid mistaking collocation identities for a physical certificate.
Public routing, GSP01 OPEN, original cohort admission failures, populations and
final four-repo full GPU/physical/complexity acceptance obligations remain.


### 2026-09-28 independent interface sampling and regional density diagnostics

The maintained magnetic volume physical collector adds three barycentric
locations (.2,.3,.5),(.5,.2,.3),(.3,.5,.2) on each of the same24 selected
polygon faces. The original 51 interior/51 exterior fields and24 centers remain.
Both new72-point normal B and tangential H maxima enter the SAME .005 combined
gate. It is not enough to satisfy flux equations at their own collocation points.
Two-sided two-height Richardson estimates remain sample estimates, not certified
trace bounds. Tests collect evidence; their pytest pass does not mean physics pass.

New maintained P1 single-layer regression checks explicit PV gradients against
two-sided gradient means at three general anisotropic face locations. Richardson
uses heights1e-6 and5e-7; self conormal n.A.grad S[lambda] is independently zero
because every physical displacement lies in the plane. This fills the gap left
by jump DIFFERENCE checks, which alone cannot rule out a common gradient offset.
The new control passes; no production kernel change was indicated.

External diagnostic probes tested regional single-layer densities with
 phi_r=-H_background.x+S_r sigma_r,
 V_0 sigma_0 - V_i sigma_i=0,
 (K'_0-1/2) sigma_0 -(K'_i+1/2) sigma_i=n.(A_0-A_i)H_background.
Signs and background RHS were independently audited. P0 densities use centers;
P1 densities use either three (.6,.2,.2) permutations or the degree2 exact
triangle-rule nodes (2/3,1/6,1/6). The latter still uses collocation, NOT Galerkin
integration. Further P1 weak probes integrate target P1 test functions using
order4/order8 Gauss-Duffy rules, analytic source layers, face-area normalization
and exact mass half-jumps (ones+I)/12. Constant RHS has each test integral1/3.
These probes are one-body, uniform-drive diagnostics, not production APIs: no
owned-state, source-current, resource-budget or differentiability qualification.
No failed candidate was inserted into public HB or storage.

Selected results (relative maxima; all combined .005 gates FAIL):

| method | level | inside H | outside H | center normal B | off-center normal B | center tangent H | off-center tangent H |
|---|---|---|---|---|---|---|---|
| P0 densities, centers | 3 | 0.01113699 | 0.00772476062 | 1.15585622e-09 | 0.0185214833 | 0.00316980181 | 0.00436703413 |
| P1 densities, three points | 3 | 0.00642038257 | 0.00561643452 | 0.00880745406 | 0.00563374776 | 0.000246021865 | 0.00165430167 |
| P1 densities, Gauss nodes | 3 | 0.00381481377 | 0.00443830914 | 0.0142926377 | 0.0111075323 | 0.000351036142 | 0.00172298774 |
| P1 weak target order4 | 2 | 0.0102066653 | 0.0118563988 | 0.038124895 | 0.0331481429 | 0.0021616508 | 0.00352297903 |
| P1 weak target order8 | 2 | 0.00967942619 | 0.0113522004 | 0.040904211 | 0.0354258339 | 0.00218530061 | 0.00351385676 |
| Current magnetic volume | 3 | 0.000818874875 | 0.00281949103 | 0.0361244241 | 0.0476579851 | 2.86872256e-14 | 4.61472366e-13 |

P0 flux continuity at the actual collocation centers is ~1e-9, but off-center
normal error is1.852%. Thus a center-only gate would give misleading evidence
for that constraint. Current volume level3 off-center normal error4.766% is
worse than its already failed center3.612%. No prior failures are overwritten.

The weak order4-to-order8 comparison at level2 changes center normal B from
.0381248950 to .0409042110 and off-center B from .0331481429 to .0354258339.
This single order doubling is not a convergence certificate. There is no
evidence of a sign/PV bug, but target quadrature and spatial discretization
errors are not yet separately controlled. Next work must establish accurate
near/touching-panel weak integrals and their error, before expanding to larger
meshes or treating the remaining error as solely a density-space issue.

Only the new maintained PV control and the changed unrefined magnetic collector
were run on GPU1. The old refined-volume case/full suites were not repeated.
External diagnostics add formulation evidence, not broader regression coverage.
Every receipt below records809 stable source hashes. Earlier receipts precede
the two test additions; the final off-center receipt matches current inventory.

static-single-pv-mean-v1: terminal 0, {'tests': 1, 'failures': 0, 'errors': 0, 'skipped': 0}, 4.723632775s. Runner SHA256 19533486646f8500c98c63b4c40f04017773cbbf5ce21b8c4913370452f279e2; receipt SHA256 acfbe8489e706456a8522445081b631e33b5be5d3dffc33da8ed1445a6579809.

magnetic-regional-single-comparison-v1: terminal 0, {'tests': 1, 'failures': 0, 'errors': 0, 'skipped': 0}, 6.262334664s. Runner SHA256 4a31f0c7baa53124d637a8af02cfc06008cbb943b731778222ab8eeb4f60bef6; receipt SHA256 e07fab99415b69273973688d9d1a573b6b591690bd91828a3197d1aac9b61f47.

magnetic-regional-linear-single-comparison-v1: terminal 0, {'tests': 1, 'failures': 0, 'errors': 0, 'skipped': 0}, 39.857307629s. Runner SHA256 ada2e1cf0fc3b820a5221f6f0d63df0ce0a3eb34db94b072c31585696eb1cdf6; receipt SHA256 ec2ee5f0fdc19c21c2895c5e40fb622b0807be3ddead073c5c81118381029406.

magnetic-regional-gauss-single-comparison-v1: terminal 0, {'tests': 1, 'failures': 0, 'errors': 0, 'skipped': 0}, 38.913246853s. Runner SHA256 0fb06bd11e047db8ebf31e288f67ba21478c9d08f94d76a2adab99cb16dc43eb; receipt SHA256 8cd8ee6f7db94a5d97a46bed0e5f8be499afca7e3a6e726f804980d39bc2a25a.

magnetic-regional-weak-single-comparison-v1: terminal 0, {'tests': 1, 'failures': 0, 'errors': 0, 'skipped': 0}, 15.897162074s. Runner SHA256 0bc86843cc6e4c00f72f0f868fa13fe99fbb8b4cf502854cc09f61aafb791c6b; receipt SHA256 a08f2ef46b7bcdf6243dc7641924d266c3d9a5edaa5d82881b7498ff92ed2635.

magnetic-regional-weak8-single-comparison-v1: terminal 0, {'tests': 1, 'failures': 0, 'errors': 0, 'skipped': 0}, 45.932391080s. Runner SHA256 09a7d7ba19dfa70d3fecee5ec8302bf9d9cfdf14fd9a9b4e35ef72ee8ca7e118; receipt SHA256 6c43af005bbaafe97a639d3ab7bf09eb662af928bbb130a0f8ca5ab497f143a1.

magnetic-off-center-control-v1: terminal 0, {'tests': 1, 'failures': 0, 'errors': 0, 'skipped': 0}, 6.520176752s. Runner SHA256 a0e7b92f1d341c57afd84ce9e5d3e926856fd88783b628836461c2bf0e29ebbb; receipt SHA256 3d85cb6f45651c5e053ec822d409cdf02e08ef049d5dc21630fdc30f15868d85.

External helpers/probes are outside provider inventory; bind their exact bytes separately:

/tmp/regional_single_probe.py: SHA256 72d2477b9ef562e27c5b5a63822c0919a4456d9917cd0c5c1383e6bf8445ae84.

/tmp/regional_linear_single_probe.py: SHA256 a3f10e6394f9beabdf83ad4acc432d54c32d338a6433e91df4a6fdea56c6db83.

/tmp/regional_gauss_single_probe.py: SHA256 2f78d8397dde57ba350b1472012ceb616597fe945788164c46d0cc68b057f0a8.

/tmp/regional_weak_single_probe.py: SHA256 ca874799a9c174bf20ab6f08570d7096fde566ce7ecbbdf5a1e72a46907f0f71.

/tmp/regional_weak8_single_probe.py: SHA256 c656ca805d9040defe284fac166c72489eeba7937f0b50e0e58ed0666f0fff16.

/tmp/test_magnetic_regional_single_comparison.py: SHA256 4b93f314435a6547c76f0e78c6d13a71dac1e2417ca6db15e813e937e364a0e1.

/tmp/test_magnetic_regional_linear_single_comparison.py: SHA256 71c5aba4ae88ed5950ea97734b78b8bd534fdd3d9e2aa36ec8e3b4610a37a1d2.

/tmp/test_magnetic_regional_gauss_single_comparison.py: SHA256 ac86ff9d77e88df0672841c5a47bb1c05c9e3aac13672d21282b7f251fb68085.

/tmp/test_magnetic_regional_weak_single_comparison.py: SHA256 e6b22493bf25b5d6a42aaff7098e444e2b9763557870d53ed7cc014d16a0e80b.

/tmp/test_magnetic_regional_weak8_single_comparison.py: SHA256 889067c3775e83bf4d601a9b123256dacb8e9ecaf494e64e66a5ddf566069df3.

GSP01 stays OPEN, all original failures and cohort/population obligations remain. Goal ACTIVE; final four-repo acceptance and final audits are not complete.

Final independent audit found no confirmed P1/P2, checked the stronger gate and unchanged old metrics, and verified the final terminal receipt/current809 hashes. The refined-volume parameter has not been rerun for its newly added observations; it is not claimed covered.


### 2026-09-28 joint touching weak layers and stable edge logarithms

StaticTensorGreen.touching_linear_weak_layers returns paired physical P1 test/
source matrices V and target-normal conormal K' for self/edge/vertex topology.
It uses existing joint Duffy rules, physical barycentrics and native coefficient
gradients. Self K' is exactly zero on a flat panel. No half jump is included;
that belongs to the weak trace mass matrix. Integer ids, shared coordinates,
resolved geometry, category and node budgets are checked before integration.
Input faces must come from a conforming nonintersecting mesh; local id checks
do not certify absence of additional intersections. The node cap bounds one
generated batch, not peak memory or all retained autograd graphs.

Controls include the independent closed-form scalar right-triangle self sum,
self symmetry/positive energy, reordered source nodes, SPD directional material
derivatives, scaling, reciprocity, chunking and invalid admission. Joint orders
4/8/12 compared with16 give edge relative errors1.06694e-4/1.38421e-8/1.07958e-11
and vertex6.78789e-4/3.83114e-7/3.84694e-10 on the selected test geometry.
These are selected convergence controls, not universal integration bounds.

Independent analytic-source P1 integration was also compared with the joint
point-kernel method. Vertex target orders32/64 gave4.38346e-6/2.82537e-7, passing
the2e-5 check. Edge orders32/64 gave.00135313/.000343896 and FAILED that unchanged
check. Increasing the independent reference to128/512 exposed a genuine failure
in existing static panel primitives: r0+r1-edge_length rounded to zero near an
edge, causing nonrepresentable gradients at admitted nonsingular coordinates.
Both failed receipts remain intact.

The edge log is now shared by static scalar gradient/potential/vector moment
and the tensor P1 single-layer edge integral. With source-axis coordinates
t0,t1=t0+L and perpendicular squared distance b^2, the denominator is evaluated
as 2 max(t0,-t1,0)+b^2/(r0+|t0|)+b^2/(r1+|t1|), mathematically equal to
r0+r1-L. No epsilon, omitted region or singularity cutoff is introduced.
Maximum tie splitting at projected endpoints cancels the absolute-value kink.
Independent asinh values and native derivatives cover very near-segment queries,
projected endpoints and nonsingular collinear extensions. True edge admission
remains unchanged. With this fix, independent edge orders128/512 give
8.66666e-5/5.44886e-6, passing the ORIGINAL2e-5 threshold.

GPU1 weak v1:4pass4.920689870s. Independent-source v2:1pass1fail4.058718720s.
v3 refined edge:1fail10.599661998s, nonrepresentable kernel. v4 after fix:
20pass28.600071132s, including the failed edge control and affected old scalar
moment/Green/P1/P2 controls. All810 source hashes stable within each run.
Independent audit found no confirmed P1/P2, checked the gap identity and endpoint
derivatives, final receipt/current hashes, and qualification limits.

A diagnostic weak regional-density assembly replaced only touching blocks with
order8 joint moments, divided by target face area to match the existing normalized
weak equations. Nontouching blocks initially retained target order4. At level2
inside H max .0095131568/outside .0111586297, center normal B .0419657240,
off-center normal B .0369408380; all combined .005 physical gates FAIL. The
collector terminal1pass23.000366821s does not indicate physical qualification.
An additional target-order8 diagnostic is recorded below after completion.
No failed diagnostic is integrated into public HB/storage; public routing is
unchanged. Old GSP01, magnetic physics, original cohort and population failures
remain open; final full four-repo acceptance and independent audits are pending.

/tmp/gsp-static-weak-touching-v1.py: SHA256 d7c4acb6c7c1064a16ed4d1500cb9cb8354a3d6cd0c381ef07f0b930c00be314; receipt SHA256 24e1dc41f2c24b66f266af34bda987e5d2eda059f5e913e1e0832d9e6f213b32.

/tmp/gsp-static-weak-touching-v2.py: SHA256 94abd3cc2b223880afc9e48a44744ab12f7824b88258d35fcd6a2fe700b1dc47; receipt SHA256 96e9aba5d96bb2663917926472e7c3a9498d3f4da739e20ca6aef6f40c262a5f.

/tmp/gsp-static-weak-touching-v3.py: SHA256 5dc0d7c6755ae5ffcdf403b7f681d02de4792af486fb60b54bb6951fb385d1af; receipt SHA256 36c9097a9fb4df9844862d976054ef4d6875d1087b2403ca03c03196915d873d.

/tmp/gsp-static-weak-touching-v4.py: SHA256 34e3e80f5bc47db98b3ed745e6adcf69a97739f09b1d5321a8bb316aff8ff9d1; receipt SHA256 b73c05a3d74a6abae18b1c5dfbbbe0de66359a5ad11933df01d473dc6bbaf401.

/tmp/gsp-magnetic-regional-joint-weak-single-comparison-v1.py: SHA256 9df12087bf552da44d6411b6d6fa5302c5d85a2121d099214cc2f52f75d56a2d; receipt SHA256 7a3cb9c78aa14df9def0e9a4057789280875c83e0918abb54f7f81157664ab22.

External helper outside provider inventory: /tmp/regional_joint_weak_single_probe.py, SHA256 44f48bcddf993b89b35ad421b62f8041159a4014d9eab26c44c2d62363b309b3.

External helper outside provider inventory: /tmp/test_magnetic_regional_joint_weak_single_comparison.py, SHA256 9c4bcfb9f0865e68d1420f253a97bf7a6cab04a8b9cf039aa8a0fcdacf0433d8.

Final target-order8 diagnostic: terminal collector1pass71.063036176s,810 stable
current hashes. Level2 center normalB .04196595797586475/off normalB
.036941983657071735, insideHmax .009513096109991457/outside .011158618251253702.
Combined .005 physical gate remains FAIL. Compared with joint-touch8/target4,
center/off normal changes are2.3394e-7/1.1456e-6, much smaller than the previous
untreated-touching order4-to-order8 changes. This selected sensitivity control
is not a universal quadrature certificate. The target4 run preceded the
algebraically equivalent edge-log fix; no full error-bound claim is inferred.
Peak2614333952bytes and wall time are diagnostic, not resource qualification.
Next separate spatial-density/refinement error using these controlled touching
blocks and further independent near-disjoint error checks; avoid interpreting
small equation residuals as qualification. No jobs remain live.

/tmp/gsp-magnetic-regional-joint-weak8-single-comparison-v1.py: SHA256 b47e01ee0b054eb7247b1694d5cf69979fdf24dc2d386382cb142d99be81ca44; receipt SHA256 1efc2592be6266db1cae371fbac9f88c6ec328e3697238ed702b2cd120d9cb9e.
External helper outside inventory: /tmp/regional_joint_weak8_single_probe.py, SHA256 bf737f81a9114ef092d587d2df04bb3b06d251dedd9ebd8f87188734d62967de.
External helper outside inventory: /tmp/test_magnetic_regional_joint_weak8_single_comparison.py, SHA256 7cbff0359058b048ce29b1b9d1424a70fdc83650c61cd0b423294cea359a6ec4.

Final independent audit confirmed weak block area normalization and current terminal810-hash evidence with no confirmed P1/P2. Because the two target-order runs also straddle the edge-log fix, their small difference is not a strict single-variable convergence proof. All physical and final acceptance obligations remain open.


### 2026-09-28 analytic P1 gradients and adaptive disjoint weak layers

StaticTensorGreen.panel_linear_integrals now returns physical P1 single-layer
values and observer gradients. Near-panel gradients differentiate the existing
analytic edge moments explicitly: for edge coordinates t0,t1, perpendicular
vector p from observer to edge and edge tangent e,
 grad_x integral_edge R = -p integral_edge(1/R) -e (r1-r0).
The radial difference is rationalized. Barycentric and constant-single-layer
terms are included and transformed back with L^-T. The separated branch
integrates point Green gradients directly with its existing positive order16
rule. panel_linear_single_layer delegates to the same implementation; P2
double layers use the analytic gradients, eliminating three internal backward
calls. Native material/observer graphs remain available. Inference-mode queries
are tested on a kernel constructed outside inference mode; this is not a claim
about constructing all owned objects inside inference mode.

New disjoint_linear_weak_layers validates resolved nondegenerate paired faces
and a positive physical gap using the existing full triangle-distance primitive.
It combines source layer values/gradients with target Gauss-Duffy integration
and returns physical P1 V/K' blocks without half jumps. max_points bounds target
evaluation pairs per batch, not all source quadrature or retained autograd
workspace. Fixed order alone does not certify accuracy.

adaptive_disjoint_linear_weak_layers compares successive orders per pair; only
unresolved pairs advance. V and K' are combined after scaling K' by L/mean(A).
The returned per-pair order and relative-change estimator are explicit and
certified_error_bound/physical_qualification remain False. It raises when the
order cap is exhausted, rather than accepting budget exhaustion. Independent
auditing identified one P2 admission defect: overflowed flux scale could erase
K' from the comparison. Fixed before commit by validating positive finite
length/flux scale BEFORE initial integration. A finite SPD diag(1e308,1,1)
negative control demonstrates rejection; normal adaptive cases still pass.

GPU1 evidence:
- Analytic gradients v1:10pass8.109076834s,811 stable hashes, new direct/autograd
  and higher-derivative/PV/no-backward inference controls plus affected oldP1/P2.
- Disjoint weak v1:2pass6.369055392s,812 stable hashes; independent double point
  quadrature, material derivatives, reciprocity, mixed separations and failures.
  Gaps .2/.015 select orders8/32 with changes9.79287e-8/1.16171e-8 and agree
  with independent order48 target integration. Exhausted orders reject.
- After audit scale fix, v2:2pass4.632660789s,812 stable current hashes; extreme
  flux-scale rejection and ordinary adaptive controls. Other covered controls
  were not repeated. The estimator is not a universal integration bound.

External one-body regional-density diagnostic now assembles touching blocks
with native order8 joint rules and disjoint blocks with native adaptive orders
(4,8,16,32), rtol1e-6. Fields use analytic P1 gradients. No public HB/storage
route or owned-state qualification is added. The collector terminal1pass
60.098117149s with812 stable source hashes predates only the flux-scale guard
fix; no later numerical formula changes. Its physical gates remain FAIL.

At level2 each regional kernel has98300 disjoint pairs. Exterior selects93940
at8 and4360 at16, maxchange9.3549695e-7. Interior selects93716 at8 and4584 at16,
maxchange9.9978067e-7. Original51+51 field points,24 center traces and72 off-center
traces give insideHmax .00951309610996/outside .01115861825126, center normalB
.04196595797569/off normalB .03694198365743, center tangentH .00218876971937/
off tangentH .00351442091822. Same .005 combined gate FAILS. Values agree to
rounding with the preceding joint-touch8/fixed-target8 diagnostic. Selected
integration estimates are no longer the demonstrated source of that discrepancy;
this does not prove full quadrature convergence or close density-space error.

Observed level2 peak194110976bytes belongs only to this diagnostic. Avoiding
full target-query gradient tensors and internal backward calls reduces the
observed footprint, but this is not a production resource/scaling qualification.
Public magnetic volume routing and every prior failure remain unchanged.
Final independent audit closed the P2 with no remaining confirmed P1/P2 and
verified current v2 hashes; pre-fix physical and post-fix admission evidence
are kept separate. Next assess density-space refinement with controlled
integration and advance an owned native regional assembly only when validated.
GSP01 OPEN; Goal ACTIVE; original cohort/population/finalfour-repo obligations
and final independent audits remain pending. No jobs remain live.

/tmp/gsp-static-linear-gradients-v1.py: SHA256 8fbd02ce43c12601db58d1fc71f03350cca95d55233ec9190dce81f7e908b20b; receipt SHA256 89db33587f3ccf9699b1ac78eaf8dbca1631fdd4b6b457310a4b15118401eb87.

/tmp/gsp-static-disjoint-weak-v1.py: SHA256 609b9156664c8a28baadeb76eaeabd3ff38448413176b382295d8ed1785cebb3; receipt SHA256 bb2629c4321a079a0344b8e4f1677feee14032d29092472ca0fbd465d50628ec.

/tmp/gsp-static-disjoint-weak-v2.py: SHA256 79212206165373162224327371f54104841f0bf68dd8fe74f2490077e69d38f7; receipt SHA256 cb701cd0caa01b9440b9c855213b2e99139d2abfc4b357a073925db32b73aad3.

/tmp/gsp-magnetic-regional-adaptive-weak-single-comparison-v1.py: SHA256 f6b1311a6f988a8f79a26e4190abd5cc4c5d7638266f4034065d8680d351edcc; receipt SHA256 54b8a435ce749666f437cb5c296dc129eb18730c60dec426ca7fe39b699eda12.

External helper outside provider inventory: /tmp/regional_adaptive_weak_single_probe.py, SHA256 30c5919b5062e5258fd4e553d164ae49858bf3eee37a1e786425079568e8e1ff.
External helper outside provider inventory: /tmp/test_magnetic_regional_adaptive_weak_single_comparison.py, SHA256 970a8b3f6bb840f59d6fee9bb67e9baa8035fe23c3ebda46ae3ed5ae824fb550.

### 2026-09-28 Fixed geometry and planar-patch density continuity

Following 4d9ecb9, keep the 80-face sphere approximation fixed and subdivide
its planar faces once (320 child faces). Preserve total area, signed volume,
and parent planes. Keep the original 51 interior and 51 exterior field points,
24 parent centers, and 72 parent off-center points. Add 72 nonedge locations
without replacing any original point: some original barycentric coordinates
with component 0.5 lie on artificial child edges. The exact sphere solution
is only an approximation control for this fixed polyhedron, not its exact
field solution. The .005 threshold and combined gate remain unchanged.

The test-only `static_regional_density_diagnostic.py` owns regional P1
single-layer assembly with joint touching order 8 and adaptive disjoint
quadrature. Exterior and interior densities are independent. Weak rows are
normalized by target area; the conormal rows additionally share a fixed
coefficient/length scaling. `project_density_system` restores target areas
before applying P.T W M P and P.T W rhs. A vertex shared by child triangles
has one coefficient inside a declared patch, but a distinct coefficient in
each neighboring original face. No continuity is forced across real dihedrals.
The helper accepts explicit patch membership; it does not certify geometry.

`collect_regional_density_refinement.py` constructs the surface system directly
from the refined mesh so midpoint child order supplies explicit parent IDs.
It assembles once and solves both spaces, with 960 versus 480 density DOFs per
region. The earlier external probe inferred parents from the unique supporting
plane for this particular convex mesh; that inference is not a general mesh
mapping algorithm and is not used by the committed collector. Dense resource
bounds, gradients, multibody/current-source integration, owned snapshots and
public routing are outside this diagnostic. Timing and peak memory are
cumulative within this selected comparison, not independent scaling evidence.

Targeted controls use unequal coplanar face areas and independent analytical
P1 mass integrals, preserve a linear field, demonstrate that omitting area
restoration changes the result, check equivalent row normalization, and keep
distinct-patch vertex coefficients separate. A small closed-body assembly
checks one-patch-per-face equivalence to the discontinuous solve; raw exterior
blocks are reused for an exact zero-contrast algebra/field control (not a
second independently assembled zero-contrast test).

Reproduction: select GPU 1 externally and use the maxwell interpreter with the
provider checkout and tests/mom on PYTHONPATH. Run the collector explicitly;
MAGNETIC_CONTROL_REPORT selects the JSON destination. Collector success means
finite evidence was recorded, not that its `selected_control_passed` gate passed.

GPU 1 final checkpoint `/tmp/gsp-regional-density-diagnostic-v1` completed with
3 collector/algebra tests passing in 54.800438984 s and 815 unchanged provider/
Bench Python hashes. The two physical rows exactly reproduce the preceding
external diagnostics. Both combined physical gates FAIL:

| Metric (relative maximum) | Discontinuous P1 | Planar-patch continuous P1 |
| --- | ---: | ---: |
| Interior H versus sphere approximation | .0385753995842 | .0372605745074 |
| Exterior H versus sphere approximation | .0451091560974 | .0451062589487 |
| Parent-center normal B jump | .0166787131646 | .0176005147213 |
| Original off-center normal B jump | .0126019138715 | .0193641858123 |
| Parent-center tangential H jump | .00142290930418 | .00168141568427 |
| Original off-center tangential H jump | .0272290854855 | .00378848766733 |
| Additional nonedge normal B jump | .00900813098566 | .0187447809444 |
| Additional nonedge tangential H jump | .00321902448864 | .00336128545323 |

Continuity reduces the artificial-edge tangential error but worsens normal
flux error. Reduced residual 5.2042e-16 and raw local residual .00476124 are
algebra diagnostics only. The discontinuous residual is 8.7213e-16. Neither
small residual nor a selected passing tangential metric closes qualification.
Observed peak192174592bytes and elapsed49.13805s for the combined diagnostic
are not independently measured per-space resource or complexity acceptance.

External predecessor evidence remains immutable: fixed-geometry discontinuous
collector1pass55.158681836s, then patch-continuous collector1pass54.904101951s,
each with812 stable hashes. These external scripts and the saved raw diagnostic
matrix are provenance artifacts, not owned production state or a storage schema.
The committed collector was run once after migration to verify explicit parent
mapping and reusable assembly; no full regression was repeated.

Next investigate the remaining flux/density approximation error with controlled
geometry and integration, including refinement within continuous planar patches.
Do not mistake fixed-polyhedron sphere bias for discretization convergence.
GSP01 remains OPEN; Goal ACTIVE. All earlier failures, public cohort gaps,
actual populations, full four-repository/complexity acceptance, and final
independent audits remain open.

/tmp/gsp-magnetic-fixed-geometry-density-v1.py: SHA256 164c1cfa6bbe3a26c49f363f54c443356955ce794eed550dc0235c175dc0cdcc; receipt SHA256 c0e3317478f11430dc6132879d1c6fa3dc5a1c4563364fbd0c9c34c58a2aedec; source count 812.

/tmp/gsp-magnetic-patch-continuous-density-v1.py: SHA256 a63db1a2732116953e29fd78832cac0fb6a6885750cce7ea2bd446cc5a921227; receipt SHA256 c5585f751439e0869242511c6d2750c2e397168e4cfb78d88ec9df3257d6fb53; source count 812.

/tmp/gsp-regional-density-diagnostic-v1.py: SHA256 d6207c616eb00c049461b81cceb9b3412eff1628b0cf538869bf6aa28a5d3fa1; receipt SHA256 670af5ba05d4155c571247a669f2a47f3d90119d97162da2de3accf4fd1dc6e4; source count 815.

External artifact /tmp/test_magnetic_fixed_geometry_density_refinement.py: SHA256 9127407235885b82e808d851b53c2510c9a48b9765a5ed21825fda37e0264828.

External artifact /tmp/regional_patch_continuous_probe.py: SHA256 185ac7d6cbb22e0dff375263ad46de2da952f41ec970e58a46d7c4c9caf46293.

External artifact /tmp/test_magnetic_patch_continuous_density.py: SHA256 b2701f89a0326e9fdcbffce6a5fca96616ea3ba71b30a1b8ca56a39cb6fba128.

External artifact /tmp/gsp-magnetic-patch-continuous-density-v1/raw_state.pt: SHA256 49669f2f075941267364e4ceece2b3cd6ee069002ba58367cc1c4b13ac66f6b2.

Independent checkpoint audit found no confirmed P1/P2: verified regional
single-layer signs and mass jumps, target-area restoration, explicit parent
mapping, all observation/gate preservation, and 815 current matching hashes.
It confirms the diagnostic-only scope and both failed physical gates. This
checkpoint audit does not substitute for the required final independent audits.

### 2026-09-28 Further fixed-geometry refinement and analytic P2 single layers

On the same 80-face polyhedron, a second planar midpoint refinement produces
1280 faces and 1200 continuous-within-parent P1 density DOFs per region. The
external collector differs from a3b19d6 only in refinement1->2 and selecting
planar_patch alone. All previous observations and .005 gates remain intact.
GPU1 `/tmp/gsp-regional-density-refinement2-v1` terminal collector1pass in
687.050550242s with815 unchanged hashes. The combined physical gate FAILS.

| Relative maximum | Patch P1, refinement1 (reused) | Patch P1, refinement2 |
| --- | ---: | ---: |
| Parent-center normal B jump | .0176005147213 | .0172610249915 |
| Original off-center normal B jump | .0193641858123 | .00845241632076 |
| Additional nonedge normal B jump | .0187447809444 | .00740582338921 |
| Parent-center tangential H jump | .00168141568427 | .000378536276255 |
| Original off-center tangential H jump | .00378848766733 | .00387229678114 |
| Additional nonedge tangential H jump | .00336128545323 | .00405195955220 |
| Interior H versus sphere approximation | .0372605745074 | .0377090924337 |
| Exterior H versus sphere approximation | .0451062589487 | .0450685039486 |

Refinement helps several observations but leaves a center normal-flux error
above threshold. The sphere approximation is not the exact fixed-polyhedron
solution; field error cannot be attributed entirely to geometry while the
interface error also remains unresolved. Algebra residual8.4398e-16 and raw
local residual .00734235 do not close a physical gate; raw local residuals
on different meshes are not an accuracy/convergence certificate. Disjoint
quadrature estimates remain below1e-6 (exterior1611260 pairs order8/10560 order16;
interior1608621 order8/13199 order16), not a certified global integration bound.
Peak2643554304bytes and native682.01875s are selected diagnostic measurements,
not resource/complexity qualification. The previous refinement was not rerun.

To investigate higher density order without another uniform fourfold panel
increase, `StaticTensorGreen.panel_quadratic_integrals` now returns six P2
single-layer values and analytic observer gradients. `panel_quadratic_single_layer`
delegates values. The node order is vertices0/1/2 followed by edges01/12/20.
In the transformed source plane, y is source minus projected observer, h its
signed height, R=sqrt(y.y+h*h), and T is the tangent projector. Integration by
parts yields Q_ij=boundary(y_i R m_j)-T_ij*integral(R), where
integral(R)=(boundary(y.m R)+h*h*integral(1/R))/3. Expand barycentric quadratic
moments using Q, the existing first moments, and the constant single layer.
The tangential gradient integrates the source polynomial gradient minus its
edge integral; the existing P2 source-conormal supplies the normal component.
Apply the physical-area/determinant weight and L^-T observer transformation.
There are no internal backward calls. These are primitives only: no P2 weak
regional assembly, solve, source/HB integration, or production qualification
has been claimed. Quadrature/density/geometry errors remain to be separated.

The separated branch retains the radius/distance<=1/32 admission and order16
positive Duffy quadrature, with an explicit conormal flag preserving the old
P2 double-layer path. Independent audit found and corrected a P2 proof issue:
the new observer-gradient numerator adds one polynomial degree. The old P1
gradient/P2 source-conormal proof still integrates binomial terms through k14;
the new P2 observer gradient guarantees exact terms through k13 (degree<=29,
plus one radial Duffy Jacobian degree). For t=65/1024 its coefficient tail is
bounded by t^14*(15-14t)/(1-t)^2=2.77407494209757e-16<3.2e-16. Positive quadrature
adds the integral and quadrature remainder contributions. This remains an
absolute, magnitude-scaled bound before floating error, not a relative bound
near zero or a global derivative/solver certificate. No numerical algorithm
or acceptance threshold was changed to fix the proof.

Targeted evidence:
- `/tmp/gsp-density-refinement-cli-and-p2-prototype-v1`: 3pass17.572701683s,
  815 stable hashes. New explicit CLI zero-refinement patch result agrees
  with previous unrefined DG interface values within1e-11; P2 external near
  prototype passes independent order32 cubature, observer/material derivatives,
  gradient-of-value and anisotropic jump/PV checks. This is pre-core evidence.
- `/tmp/gsp-quadratic-single-integrals-v1`: 5pass6.853798722s,816 stable hashes.
  Committed P2 near/far values and gradients, native higher derivatives,
  permutation/scaling/translation, empty/admission and inference queries of an
  existing kernel; affected old higher-layer far control. Far distances1/10/
  100/1000 use independent order24 point cubature. No full suite was repeated.
- After this run only the far-bound docstring changed. The saved pre-proof-fix
  source has an identical executable AST after stripping docstrings. Evidence
  binds the pre-proof-fix hash; no repeat numerical run was needed for prose.

The collector now accepts `--refinements 2 --spaces planar_patch` (or other
explicit nonnegative levels), returns rows for targeted reuse, and labels the
sphere approximation explicitly. Its default original comparison remains.
GSP01 OPEN, Goal ACTIVE; all original cohort, physical, resource, population,
four-repository regression and final independent-audit obligations remain open.

/tmp/gsp-regional-density-refinement2-v1.py: SHA256 3ab431875600d9872421f24330d0c60d9a8b118501d4a7d488cd58072478c385; receipt SHA256 88d6a57c771062e814acca2158b0a2c77a75daff1fb57b9897672e1ed435f00b; source count 815.

/tmp/gsp-density-refinement-cli-and-p2-prototype-v1.py: SHA256 79c4cd7a8cffb0044b542abe4e69768b922d774856c10d1485d89e3dce6bf0bd; receipt SHA256 fa9e7c74f8ea4f5339b31820d3b706fac1670e750825c652fa7386461b627d85; source count 815.

/tmp/gsp-quadratic-single-integrals-v1.py: SHA256 856275214972ea8b268a65cc8ddf0abd84f05b68c53d6382f53ba6716c0afeee; receipt SHA256 45ef987c2ea7580bc04b072abb0ab5ece3e2dc48cf15648fb86724ed10c25d9e; source count 816.

External artifact /tmp/test_regional_density_refinement2.py: SHA256 19b35e426a811e78348fe27f6aba0f892567fe182b4d8d6aeb34711b5d5b4698.

External artifact /tmp/quadratic_single_layer_probe.py: SHA256 7e9632727c99a55ee26a03013bfbb3110586342c0e1a139a55f4b409116f3364.

External artifact /tmp/test_quadratic_single_layer_probe.py: SHA256 9d62bd406c1098a27867bacdfd8354c26ce19794bc4d053e7cbe1c1a0ca1f925.

External artifact /tmp/test_density_cli_controls.py: SHA256 cc126e79195b4277e0b7fa4d429e27c8dacf90fa4e8a406b67ca17520f1b0f81.

External artifact /tmp/static_tensor_green_before_p2_far_proof_fix.py: SHA256 870da2972a47d07950ab72ad73fcd87a1de2203daa132fe2b30bdf27949072f6.

Final independent re-audit closed the P2 proof issue with no new confirmed
P1/P2. It verified the saved pre-fix source SHA against the 5-pass receipt and
the executable-AST identity, so a proof-only correction did not trigger another
numerical run. The audit also preserves the failed physical gate and does not
replace either required final independent audit. No jobs remain live.

### 2026-09-28 Quadratic weak layers and regional density comparison

Following c563d03, native real-SPD static weak blocks support P2 target/source
Lagrange bases. New touching/disjoint/adaptive_disjoint_quadratic_weak_layers
use the same geometry checks, joint Duffy touching rule, physical gap admission,
node/point budgets and scaled successive-order estimator as the P1 entrypoints.
Shared private implementations keep the existing P1 algorithm. P2 columns are
vertices0/1/2 and edge midpoints01/12/20. Each conormal block excludes the
mass-matrix half jump; self K' remains exactly zero on a planar face.

The regional diagnostic accepts degree2, assembles six local DOFs per face,
and uses the exact normalized P2 mass M/180:
[6,-1,-1,0,-4,0; -1,6,-1,0,0,-4; -1,-1,6,-4,0,0;
 0,0,-4,32,16,16; -4,0,0,16,32,16; 0,-4,0,16,16,32].
The uniform-background test moments are [0,0,0,1/3,1/3,1/3], not the P1 means.
Global edge IDs provide consistent midpoint nodes; (patch ID,node ID) keys
share coefficients only within the same declared parent plane. Restore target
areas before reducing the weak equations. The existing P1 route remains the
default. `collect_regional_density_refinement.py --degree 2 --refinements 0 1
--spaces planar_patch` selects the higher-order experiment explicitly.

GPU1 `/tmp/gsp-quadratic-weak-layers-v1` terminal14pass12.935406123s,817 stable
source hashes. Coverage includes independent double point quadrature and native
material gradients for disjoint P2 blocks, reciprocity, exact P1 embedding of
all three touching categories, order8/order16 convergence, positive self energy
and zero self K'. Independent analytic-source edge integration at target orders
128/512 has relative errors9.08833944e-5/5.71374848e-6 against the joint order16
reference (unchanged2e-5 gate). Mixed disjoint gaps.2/.015 select orders8/32,
with successive changes8.61328719e-7/2.51976687e-8 and an independent target48
comparison. Exhausted orders, contacts and insufficient budgets reject.

Diagnostic controls integrate the P2 mass and background moments independently,
reproduce a quadratic polynomial on unequal-area triangles, check weighted
energy/projection/solve and midpoint sharing, and compare independent-per-face
patch nodes to the DG solve. Both degrees retain the zero-contrast field control
using reused exterior blocks. Affected P1 disjoint/overflow and closed-form self
controls pass. These selected controls do not certify global quadrature error,
peak memory, material-gradient physical accuracy, or the regional solve.

Independent engineering audit found no confirmed P1/P2 and verified all817
current hashes and runner/XML/log consistency. Physical comparison follows
separately; a collector pass cannot substitute for its .005 combined gate.

GPU1 `/tmp/gsp-quadratic-regional-density-v1` terminal collector1pass in
266.041702766s,817 unchanged hashes. P2 refinement0/refinement1 use80/320 faces
and480/1200 DOFs per region on the same fixed polyhedron. Reuse prior P1 results
at the same DOF counts; no P1 physical case was rerun. All original51+51 field
points,24 centers,72 off-center and72 additional nonedge observations remain.
The .005 combined gate FAILS in both P2 cases:

| Density space / per-region DOFs | Center normal B | Original off normal B | Interior H vs sphere | Exterior H vs sphere |
| --- | ---: | ---: | ---: | ---: |
| P1 patch /480 (reused) | .0176005147213 | .0193641858123 | .0372605745074 | .0451062589487 |
| P2 patch /480 | .00930000029076 | .0222268226799 | .0361711909256 | .0451684484802 |
| P1 patch /1200 (reused) | .0172610249915 | .00845241632076 | .0377090924337 | .0450685039486 |
| P2 patch /1200 | .0364662031997 | .0117125909204 | .0376911119024 | .0450586920341 |

P2/1200 additionally has nonedge normalB .00710086585413, center tangentialH
.00111394806093, off tangentialH .00216111949579 and nonedge tangentialH
.00346463720137. The reduced residual8.8249e-16 and raw local residual
.00415037905744 are distinct algebra diagnostics, neither a physical gate.
P2/480 nonedgeB .0198476764948, center tangentH .00313927396596, off tangentH
.00351013917793 and nonedge tangentH .00364141254746; reduced residual6.3265e-16.

P2/1200 disjoint orders: exterior89440 pairs order8/8860 order16, maximum
change9.98621677e-7; interior88233 order8/10067 order16, max9.99469864e-7.
The touching rule remains joint order8. These estimators/selected convergence
checks do not rule out every integration contribution. No monotonic maximum
interface-error improvement is established by increasing degree/refinement;
the finer P2 center error is worse than both its coarse result and the
same-DOF P1 result. The cause is unresolved: do not claim a demonstrated edge
singularity mechanism or attribute all field error to sphere geometry bias.

Observed P2/1200 elapsed234.959408651s and peak580155904bytes are selected
measurements within this two-case process, not an isolated resource/scaling
qualification or a certified performance advantage over previous P1 runs.
No public solver, owned snapshot, multibody/current source or HB route changed.

Final independent evidence audit verified runner/XML/results and817 current
hashes, found no confirmed P1/P2 or implementation defect explaining the
non-improvement, and confirmed both failed physical gates. The external test
wrapper is separately hashed below. No full suite or repeated P1 physical
acceptance was run. Next diagnose the spatial flux residual and density
behavior under controlled integration before selecting further density-space
changes; blind degree/refinement increases are not established remedies.
GSP01 OPEN, Goal ACTIVE; original cohort/DC-channel gaps, all preserved failures,
actual populations, final four-repository GPU/physical/complexity acceptance and
required final independent audits remain open. No jobs remain live.

/tmp/gsp-quadratic-weak-layers-v1.py: SHA256 411db351cb8aea3f00afe54b2048812fe40a4cd4b7eaa69e5d058d9e7e9e1107; receipt SHA256 2bafd82cf154928f97e25a03781fb4ee6130947fb6db2697ea12e4233d683205; source count 817.

/tmp/gsp-quadratic-regional-density-v1.py: SHA256 81af52cb476e9b7dc248ef0409fe077b311c27d52a3c06e9c3c5da6708e6b3af; receipt SHA256 46dc71fcb21665d81422df0f8b378736b557b18921c43b59fdaa0e913595377e; source count 817.

External wrapper /tmp/test_quadratic_regional_density_comparison.py: SHA256 5d0105d5af31f7896d1b3477845be94280a0c943347c660dd41678ac9de7812c.

### 2026-09-28 P2 gradient reuse and spatial residual diagnosis

The P2 single-layer gradient now reuses its already computed P1 gradient rather
than re-evaluating layers inside the P2 double-layer routine. With L the real-SPD
Cholesky factor, use K1=-n_physical.A.grad(S1), transformed affine barycentrics b
and their gradients g, and physical height h:
C_ab=K1_a*b_b+h*(grad(S1)_a@L).g_b. Any affine extension agreeing on the source
plane obeys the identity; b and g must use the same extension. Symmetrize C,
convert to the six P2 bases, and recover the transformed normal gradient as
before. The separated path is unchanged. This removes repeated evaluations;
no isolated speedup or resource qualification is claimed.

GPU1 `/tmp/gsp-quadratic-gradient-reuse-v1` terminal6pass6.621961148s with817
unchanged source hashes. Includes existing P2 independent near/far cubature,
native observer/material derivatives, jump/PV, orientation/scaling/translation,
empty/admission/inference controls, affected independent P2 disjoint weak
material-gradient control, and a new per-column conormal comparison against
the old P2 double-layer computation at positive/negative heights, near traces
and PV for a general non-diagonal SPD coefficient. Independent audit confirms
the affine-extension identity and implementation; no confirmed P1/P2.

The next diagnostic keeps the 320-face,1200-DOF-per-region P2 state fixed and
saves raw matrix/geometry/material/density tensors before field analysis. The
artifact is explicitly `diagnostic_raw_tensors_not_owned_state`, binds the
runner's source inventory SHA, and is not a public restore format or owned
snapshot. GPU1 `/tmp/gsp-quadratic-density-spatial-v1`: collector1pass129.131627181s,
817 stable hashes. New evidence at the original24 centers:
- Direct PV plus the explicit half-density jump: maximum normalB .0364662037252.
- Original two-offset Richardson result: .0364662031997 (agrees with prior
  7ed7460 physical result to rounding); difference from PV5.25466e-10.
- Tenfold smaller offsets: .0364662037199, difference from PV5.25200e-12.
These selected center discrepancies are not caused by unresolved offset limits.
This does not certify every near-interface query or quadrature pair.

384 additional interior face samples produce maximum normalB .0848684408771
and RMS .0234997794526. Worst selected parent-center face is61. Potential
mismatch at centers, normalized by radius*|Hbg|, is .000163635191803. The maximum
normalized sum of regional nodal densities is4.55531 on original-patch boundary
nodes and3.77699 on interior nodes. Those values alone do not establish a
singularity mechanism or a density-space error bound.

For parent61, independently integrate PV flux jumps over its four children
without changing the solved matrix. Recover raw local weak flux moments as
-scale*(M*density-rhs)_flux and retain target areas when aggregating shared
patch nodes. Target orders8/24 give maximum local moment differences
.0007087777445/.0000845419618284; patch moment residuals
.000177194436125/.0000211354904572, normalized by original area and MU_0*|Hbg|.
This selected quadrature converges toward the weak equations while pointwise
flux jumps remain large. It does not prove a global integration certificate;
small weak moments cannot substitute for the original pointwise physical gates.
Raw-state and spatial tensor files are separately hashed below for reuse.

A subsequent external diagnostic reuses that frozen geometry and raw weak state
(with matching source inventory), assembles analytic P2 values/gradients at
nine interior Gauss points per child, and compares area-weighted point least
squares. It saves projected point blocks once; no weak-matrix reassembly occurs.
The potential equations are divided either by global extent or local maximum
face edge, and flux equations by mean exterior permeability, with sqrt(area
quadrature weight) on all rows. These are two explicit diagnostic scalings,
not certified norms. A factory only supplies the already solved response to
the unchanged committed field collector, with geometry/material/drive/patch
identity assertions; it does not rerun the weak solve or replace any gate.

GPU1 `/tmp/gsp-quadratic-density-point-fit-v1`: collector1pass13.634553079s,
817 stable hashes. BOTH original .005 combined gates FAIL:

| Scaling | Center B | Off B | Center tangent H | Off tangent H | Interior H maximum | Exterior H maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Global extent | .0129707599414 | .00932573787846 | .127110583611 | .505663869677 | .0882485686205 | .0485587560138 |
| Local face edge | .0215790997371 | .0111772869275 | .0076919904732 | .0358596976662 | .0392922069418 | .0461554282742 |

Global/local nonedge normalB .00706108936877/.00885849403855 and nonedge tangentH
.498430906668/.0420662624408 also remain recorded. Least-squares residuals
.00425675207219/.00853260123211 use the respective weighted point systems;
`local_residual` .00789937873408/.00566341674807 remains the original raw weak
system residual. They are different norms, not interchangeable accuracy claims.
Collector timing here measures fields with preassembled point-fit data; point
preparation and solve timing are separate. None is a resource/scaling certificate.

This experiment lowers some normal errors by sacrificing tangential continuity.
It is not an acceptable replacement route. The cause of the broader weak
approximation error is not fully established. Next investigate simultaneous
physical normal/tangential residual control using the saved data, rather than
blind degree increases or accepting only a favorable scalar metric. No public
solver or owned-state route changed. GSP01 OPEN, Goal ACTIVE; all preserved
failures, original cohort/DC gaps, actual populations, final four-repository
GPU/physical/complexity acceptance and required final independent audits remain
open. No full regression or unrelated covered check was repeated.

/tmp/gsp-quadratic-gradient-reuse-v1.py: SHA256 2e08a95698e8a0849ed84075035dcbffd3bbe35f008ab46a5aacab429f57f4f0; receipt SHA256 5523c1255d9896ebbcc69c107372b0fcd5903078d93ca75cc32652858440df93; source count 817.

/tmp/gsp-quadratic-density-spatial-v1.py: SHA256 d8ea2ddc2f2f5142f0bb495dfcd86fd74fa0b50e867aea618726b94069251da4; receipt SHA256 a481fad40cb95868442d405a84dab6e4f9f5ba7492fab60794ed1c93e30445d8; source count 817.

/tmp/gsp-quadratic-density-point-fit-v1.py: SHA256 0c05bc38f595844478f96eec91ace152ee0b693cf48e1d3464fae4ffef05a59d; receipt SHA256 457cbe143781a868e7ed00859c37ec16d34e34f14d77b5fbd1fcf145244b3b65; source count 817.

External diagnostic artifact /tmp/test_quadratic_density_spatial_probe.py: SHA256 4623e9ab06d760c09b58c6c1448cb480ce9257f2844976a6e222f5f094a82ef5.

External diagnostic artifact /tmp/test_quadratic_density_point_fit_probe.py: SHA256 3aaa28238bfe874079a3decb083ac5831fb0000e309b4a72e1b3819ad34e197e.

External diagnostic artifact /tmp/gsp-quadratic-density-spatial-v1/raw_state.pt: SHA256 09241d2d026e5e93e76bd5b238a747109a1e800e18f711a9c91cd3e88e8bb188.

External diagnostic artifact /tmp/gsp-quadratic-density-spatial-v1/spatial_tensors.pt: SHA256 bbcad08ece541d4df1ae40ff6a9e1bc3aaf46687e7b334d047858ac1ebddb026.

External diagnostic artifact /tmp/gsp-quadratic-density-point-fit-v1/point_fit_blocks.pt: SHA256 5f9ddf014f96c039947dd9eb04caea8fb086356cba869b0c0aab9faa6d3830d2.

External diagnostic artifact /tmp/gsp-quadratic-density-point-fit-v1/density_global_extent.pt: SHA256 2463c850ec17e98cfc5abdce799926c095aacb724e500b83bd458a8855042ef0.

External diagnostic artifact /tmp/gsp-quadratic-density-point-fit-v1/density_local_face_edge.pt: SHA256 b84d175545ab05b12ae3e5d383bf62793aa675c64c8d85ec1a5c43c530cc603c.

Final independent audit verified the spatial and point-fit signs, explicit half
jumps, area weights, unchanged collector gates and the factory's already-solved
response. It checked raw inventory/raw-state SHA bindings, both terminal receipts
and817 current hashes. No confirmed P1/P2 remains; both point-fit physical gates
remain failed. This checkpoint audit does not replace final independent audits.
No jobs remain live.


## Physical trace and centroid-hat diagnostic checkpoint

The order3 simultaneous tangential-H/normal-B fit retains the mean potential
difference gauge. It fails the unchanged combined .005 gate: center normalB
.0223801029159, off .0135605991311, nonedge .0112650484977; tangentH
.00107606477111/.00412062305304/.00548257371090. Inside/outside H maxima
.0395564012953/.0458640197810. Weighted LS residual .00959640204721 differs
from the original raw weak residual .00563593115609. This reuses the saved
order3 point blocks; no public route changed.

The new developer diagnostic adds one centroid hat per child triangle, with
boundary value zero and value 3 min(barycentric coordinates) inside. Three
coplanar source subtriangles integrate each hat through the existing analytic
P1 kernel. They do not alter the original physical geometry. Exact internal
subtriangle-edge PV queries remain unsupported by this decomposition; training
points explicitly require unique barycentric minima and select exactly one PV
subtriangle. Original collector offset traces remain unchanged.

Both P2 and enriched solves use identical physical rows, area weights, gauge,
and target order within each comparison. DOFs per region are 1200/1520.

| Target order / space | center normalB | off normalB | nonedge normalB | max inside H | max outside H |
|---|---:|---:|---:|---:|---:|
| 4 / P2 | .0284388821945 | .00942533823917 | .00761818081024 | .0386507019029 | .0453429832237 |
| 4 / P2 + hat | .00255978326313 | .00558809235169 | .00741684806965 | .0387322223478 | .0454211247245 |
| 6 / P2 | .0324134241528 | .0103881276271 | .00646240780540 | .0381937987267 | .0448391668798 |
| 6 / P2 + hat | .000522710435091 | .00610206636261 | .00779046543329 | .0382535224094 | .0450072176906 |

Every combined gate FAILS. Order6 enriched center/off/nonedge tangentH are
.00186271980833/.00456036411116/.00358434500639. Its LS residual .0152191504124
is not a pointwise accuracy certificate. Full weak bubble blocks were not
assembled; local_residual is null, not a P2-only residual mislabeled as full.
The improvement at centers survives the sampling change, but the remaining
errors and increasing LS residual do not establish convergence or identify the
complete error mechanism. Sphere field references remain approximations for
the fixed polyhedron. Collector elapsed time covers fields only; saved dense
blocks, peaks up to 3279474688 bytes and solves confer no resource qualification.

The reproducible developer entry is
`scripts/dev/diagnose_mom_magnetic_density_bubbles.py`. Set PYTHONPATH to this
provider root, its tests/mom directory and the frozen Bench provider, then use
`--raw-state /tmp/gsp-quadratic-density-spatial-v1/raw_state.pt --output NEW_DIR
--order 4` (or 6). Use GPU1 and the maxwell environment. It rejects an existing
output directory and source inventory mismatch. Its computation/collector AST
was checked identical to the completed external order4 experiment, except
parameterizing the order. Source inventory verification now hashes the actual
imported providers. This is a frozen single-body real-SPD uniform-background
diagnostic, not public owned state. No full regression was repeated.

Next investigate the unresolved off-center boundary residual on the same
geometry using these saved matrices/densities, before selecting another basis
or production formulation. GSP01 OPEN; Goal ACTIVE; original scope, thresholds,
failures, original-cohort DC gaps, actual populations, final four-repository
GPU/physical/complexity acceptance and final independent audits remain open.

/tmp/gsp-quadratic-density-physical-trace-v1: {'tests': 1, 'failures': 0, 'errors': 0, 'skipped': 0}; elapsed 12.745326284319162s; stable sources 817; sources_unchanged=True.

/tmp/gsp-quadratic-density-physical-trace-v1.py SHA256 7f159622f7daf135926de69b06e658ccdf06ebf1a24912cc78230161569c57f7.

/tmp/gsp-quadratic-density-physical-trace-v1/receipt.json SHA256 cb9d739758d1a0db32e54469f2ffb45dd802232399ccd7c7f7576a6689a83682.

/tmp/gsp-quadratic-density-physical-trace-v1/physical.json SHA256 d1565a233cf5d70541c0e4ad2b531d377555d09e96b9d5ff996a062c1f6f0b84.

/tmp/gsp-quadratic-density-physical-trace-v1/density_tangential_H_normal_B_with_mean_potential_gauge.pt SHA256 643e739a15387a98e32387ea6d03e70687692a045ade56962194e406b2c7939c.

/tmp/gsp-quadratic-density-physical-trace-v1/physical_trace_blocks.pt SHA256 3af79dff581c96452e98e11f59bb34dcb77bcb879a129357dee6bb7bd83f03b7.

/tmp/test_quadratic_density_physical_trace_probe.py SHA256 cb26f904a452cf8632d76e446f22cb563e2c7fd7e3392f2bd879884653835710.

/tmp/gsp-quadratic-density-bubble-v1: {'tests': 1, 'failures': 0, 'errors': 0, 'skipped': 0}; elapsed 32.77347035612911s; stable sources 817; sources_unchanged=True.

/tmp/gsp-quadratic-density-bubble-v1.py SHA256 872f431a251e43d0f6d675bcbed6334feab4e04ba7100dfdf5b50ac89d3a1143.

/tmp/gsp-quadratic-density-bubble-v1/receipt.json SHA256 bf24bde1783074cd20727fc25d09fc45af208d87c50eb75727b6fbf9a70cbe60.

/tmp/gsp-quadratic-density-bubble-v1/physical.json SHA256 7094d0f6b3716445133cea3a198aeb789abf1fbd06bae6f570906a0e9abddd6e.

/tmp/gsp-quadratic-density-bubble-v1/baseline.pt SHA256 06cf6e921bacac0da09b0950213449ad04995729c39e22d76b10f278572446d6.

/tmp/gsp-quadratic-density-bubble-v1/blocks.pt SHA256 147021724377ee819547eef793685a53d4d46e6f0cd982ab0714f314902de054.

/tmp/gsp-quadratic-density-bubble-v1/enriched.pt SHA256 a515b7737be93391ea0dd76b7cfe507dbd19c2cf8997660369708f6b029c20ce.

/tmp/test_quadratic_density_bubble_probe.py SHA256 be9c2d134bc3ff0a6969b486797d6dd2a1546a46d8bae2f5db3b9821ce11f5a4.

/tmp/gsp-quadratic-density-bubble-order6-v1: {'tests': 1, 'failures': 0, 'errors': 0, 'skipped': 0}; elapsed 60.70429107546806s; stable sources 817; sources_unchanged=True.

/tmp/gsp-quadratic-density-bubble-order6-v1.py SHA256 65c4fb29f837182b5e496df5edec1dbe642b0665818957e2bd9d588b14e76585.

/tmp/gsp-quadratic-density-bubble-order6-v1/receipt.json SHA256 a6644680413d7cd8d264d8eb36fea60b1111c4e673eb93f75bfed5e47a3df962.

/tmp/gsp-quadratic-density-bubble-order6-v1/physical.json SHA256 89960c88bbc88a641f84367dac4d91196a7f06a2773757458a1d8f52364560d0.

/tmp/gsp-quadratic-density-bubble-order6-v1/baseline.pt SHA256 17813e214408bdcccbad456904bbfd34a95e6e70fc783d8f20914d4055a12231.

/tmp/gsp-quadratic-density-bubble-order6-v1/blocks.pt SHA256 78866cc4608d3fedc2d1300a7d1704b9a371aabcc381d6fe69cad78f01ac2a49.

/tmp/gsp-quadratic-density-bubble-order6-v1/enriched.pt SHA256 56227a3bb63e8d31cefb63db983f63a5e66c44948f6fa34804f6dda334781f14.

/tmp/test_quadratic_density_bubble_order6_probe.py SHA256 f847b7d458f4f09184f4e99f2a36d58e9ee55d3dea38d2e779d513a62b7181a0.

Independent checkpoint audit found no confirmed numerical P1/P2. It verified
PV ownership, half jumps, matched columns/weights and unchanged collector gates.
A CLI helper-import provenance P2 was fixed by checking all three actual module
paths against provider/tests/mom. The reviewer confirmed closure. CLI --help and
final numerical AST identity pass; these are entry/source checks, not a new CLI
numerical acceptance run. Numerical evidence remains the external runs above.
This checkpoint audit does not replace required final independent audits. No
jobs remain live.


## Physical training residual localization and rejected reweighting

Reused order6 matrices and the enriched density from the previous checkpoint;
no Green-layer reassembly or full regression. Four damped IRLS steps use all
11520 training points, retaining the original gauge. Each multiplier targets
sqrt(rho^2 + 1e-6), normalized in the existing area weights, and is averaged
with the previous multiplier. This is a finite smoothed-L4 experiment, not a
converged minimax solve. Training area-weighted L4 decreases .0382729756784 to
.0292155090596, but every original combined .005 acceptance gate still FAILS
and original observations worsen: normalB center .0125781732923, off
.0150429969650, nonedge .0155555704519; tangentH center .0144797350069, off
.0182670913666, nonedge .0180365572207. Inside/outside H maxima
.0415683091044/.0427368075846. Reweighting is not adopted as a solver.

The committed diagnostic now reports training physical residuals explicitly.
It undoes each quadrature square-root weight, divides by drive norm, and keeps
normal flux (scaled by exterior mean coefficient) distinct from tangential H.
The last gauge row is not a field observation. Barycentric bins refer to the
original physical faces, not child/source integration edges. All11520 points
are included exactly once; minimum barycentric coordinate is not Euclidean
distance and this correlation does not prove a singularity mechanism.

| Original-face min barycentric | points | original normal max | reweighted normal max |
|---|---:|---:|---:|
| [0,.01) | 1600 | .332715089663 | .211991624915 |
| [.01,.025) | 3040 | .0956026744450 | .0595334727447 |
| [.025,.05) | 800 | .0344887974877 | .0689408422120 |
| [.05,.1) | 3040 | .0394528228402 | .0494348722351 |
| [.1,infinity) | 3040 | .0258850468995 | .0359142940456 |

Training tangent maxima are .0708297351163/.0614194556316. These are training
PV values, not independent physical acceptance. To check their physical
interpretation, each density's eight largest combined-residual points were
queried through actual regional fields at +/-1e-7 and +/-5e-8m, using the
original P2 and hat fields, followed by Richardson extrapolation. Component
trace differences against the negative algebraic physical residual are
4.36103001544e-7/4.04691000849e-7 (normalized), under the diagnostic2e-6
comparison tolerance. This confirms large actual trace discrepancies at those
points; it is not a new .005 physical qualification or an independent kernel.

External reweight collector1pass21.412862999s; saved-residual field check
1pass10.211117256s. Both817source inventories remained unchanged. Reweight
collector timing is fields-only; its inherited solve_seconds measures density
installation after precomputed IRLS, not IRLS runtime. No speed/resource claim.
The helper script is outside the817inventory and separately SHA-bound by the
field-check output. Raw blocks/densities remain diagnostic tensors, not owned
state. GSP01 OPEN, Goal ACTIVE, all original acceptance obligations retained.
Next investigate density resolution near original physical face edges while
retaining all observations; repeated global weight tuning is not supported by
these results. The large residual is not solely a center-hat artifact.

/tmp/gsp-quadratic-density-reweighted-v1.py SHA256 4db92137de49a33090adb319f541c7ab2cb23f501e7ae2ed402d487355cef79e.

/tmp/gsp-quadratic-density-reweighted-v1/receipt.json SHA256 3128dffe8267c7475029593c86b67acaf0f6750f7f40cce1862c99e5c9f63328.

/tmp/gsp-quadratic-density-reweighted-v1/physical.json SHA256 f7ce52965b6dd2307fadb1230bfb343aeaa9e440240a78b102f6e1e95508e822.

/tmp/test_quadratic_density_reweighted_probe.py SHA256 5a0f1b9a3a7c8b094d8993279f40ae1551f8dafcb68582e6b20b5ff1823539ac.

/tmp/gsp-quadratic-density-reweighted-v1/enriched.pt SHA256 799d5392a010c28af4158a94c10a3661a8a723fdb228dec245af18b4a6ef36ca.

/tmp/gsp-quadratic-density-reweighted-v1/training_trace.json SHA256 eef133640ffa681139f34c061a65a6ca413ce812292e756c8378e2b623992eee.

/tmp/gsp-magnetic-training-residual-v1.py SHA256 96ea998049740ac3c2bfd4a2547102af2a1cbf26e8d3a87df4590590e579e6bb.

/tmp/gsp-magnetic-training-residual-v1/receipt.json SHA256 8dcbede9b8a9c192d534c7aa42c0df0ec758156dad5b8308d0a43e98134c458e.

/tmp/gsp-magnetic-training-residual-v1/physical.json SHA256 f0ed4beff32de445888270d4921536b73fc4ce7aa7e105aaa77c556fa4e15f58.

/tmp/test_magnetic_training_residual_probe.py SHA256 6358a8e7a2f11aeda68d9b550dadb33b2280b9ddd01d678f15ecf656c1948c2b.

Independent checkpoint audit found no confirmed P1/P2 in the summary helper.
It verified physical units, complete bin coverage, unchanged solve/acceptance,
817 stable hashes, runner/XML and saved block/density/helper provenance. Its
IRLS solve_seconds limitation is explicitly recorded above. This audit is not
a final goal audit. No jobs remain live.


## Edge support and density-node distribution experiments

The next experiment reuses the saved order6 P2+centroid physical matrix and
adds three edge-midpoint P1 tents per original physical face. For original edge
(a,b), midpoint m, opposite vertex c and eta=.05, the tip is
(1-eta)m+eta*c. Subpanels (m,tip,a) and (m,b,tip) have source basis column0;
other vertices have zero density. Each support has eta times the original face
area. Supports alter density representation only. Different original faces
retain independent coefficients at their shared physical edges. Training points
on artificial support edges would be rejected, not removed or perturbed.

DOFs increase1520→1760 per region with unchanged training/collector points and
.005 gates. Weighted LS residual falls .0152191504124→.0116970568414, but
combined physical acceptance FAILS: center normalB .00716886207337, off
.0117996581097, nonedge .00853234268759; corresponding tangentH
.00120389193736/.00362870518565/.00238075468172. Inside/outside H maxima
.0381038772378/.0452217344857. This does not support a production replacement.

Saved-block residual analysis includes every11520 training point. Normal max
.315127504539 remains large in original-face minbary[0,.01), tangent max
.0702109662645, area-weighted L4 .0336837143031. The other four minbary bins
[.01,.025)/[.025,.05)/[.05,.1)/[.1,infinity) have normal maxima
.0872566319967/.0328668716891/.0383302234278/.0145486221509. Each bin uses the
same points as the previous checkpoint. Actual offset-field Richardson traces
at eight largest combined-residual points differ by at most4.39831016774e-7
from negative algebraic physical residuals, under the same2e-6 diagnostic
comparison tolerance; this does not establish physical qualification.

The reusable developer helper `edge_density_hat_panels` returns shape
(face,edge,subpanel,vertex,xyz), with edge order01/12/20. Targeted checks verify
its geometry bitwise equals the saved experiment and each support area equals
eta times parent area. Its width must be strictly between0and1; finite,
nondegenerate triangle admission is explicit. Exact artificial-edge PV remains
subject to the existing kernel guard. No public API or solve route changes.

Edge collector1pass32.317869603s, residual/helper check1pass18.572845855s, both
817 source inventories unchanged. Collector timing covers fields only and peak
4026652160bytes is not a resource certificate. External edge probe's unexecuted
`not enriched` branch retains an obsolete column selector; only its explicit
`enriched=True` path was exercised, and that branch is not promoted into a
reusable solver. Independent audit confirmed the exercised signs, layouts,
geometry, PV and unchanged gates without numerical P1/P2.


The matched node-distribution experiment uses the same80 original physical
faces,1280 source integration panels and1200 P1 density DOFs per region for
both powers1and2. The macroface patch connectivity is unchanged. Uniform
quarter-grid barycentric coordinates are first verified against their exact
dyadic values, then mapped by lambda_i^p / sum_j(lambda_j^p). Both choices
retain positive panel orientation and each original face area. The common
order6 training targets and all original collector observations remain fixed.
Every training point must belong to exactly one source panel in its declared
physical face; artificial-edge hits fail rather than being discarded. Same
DOF count does not mean identical approximation space.

| Density grading power | center B | off B | nonedge B | inside H max | outside H max |
|---|---:|---:|---:|---:|---:|
| 1 | .0166902487249 | .00897826267034 | .00709665347994 | .0384677731722 | .0447586063744 |
| 2 | .00681641658611 | .0104510759671 | .0111778048121 | .0378814557247 | .0450406724807 |

Both unchanged .005 combined gates FAIL. Power1/2 center tangentH
.00385099141741/.00208716997598, off .00462366234730/.00219933768644,
nonedge .00488856794596/.00220895577681. LS residuals
.0197187527591/.0135347708519; training normal maxima
.323562914466/.283895811684, tangent maxima .0711354240759/.0589903328609 and
L4 .0421361539431/.0306611414791. Finer normal/tangential resolution cannot be
inferred from the center metric alone. The comparison provides no physical
qualification or convergence proof. The sphere is still an approximation
reference for the unchanged polyhedron.

Collector1pass133.301682433s,817 stable hashes. Output `faces=320` describes the
collector-bound geometry; `density_integration_panels=1280` describes the source
density triangulation. The test-only response retains the former for exact
factory binding and explicitly uses the latter for P1 fields. Local weak
residual remains null because no corresponding weak blocks were assembled.
Peak1179294208bytes and fields-only collector timing are not resource
qualification. No covered full regression was repeated.

The reusable entry `scripts/dev/diagnose_mom_magnetic_density_grading.py` takes
`--raw-state /tmp/gsp-quadratic-density-spatial-v1/raw_state.pt --output NEW_DIR`.
Use maxwell Python, GPU1, and PYTHONPATH containing this provider, its tests/mom
and the frozen Bench provider. It rejects existing output, verifies imported
provider/helpers and raw source inventory, separately records both diagnostic
script hashes, and writes cumulative physical results after each comparison.
Its mathematical/collector AST matches the completed external probe; only
input/provenance handling and incremental JSON differ. --help verifies the CLI
entry, not a new CLI numerical acceptance run.

The next investigation should distinguish unresolved boundary-layer shape
from insufficient along-edge resolution using these saved matrices and
densities, rather than accepting favorable center points or repeating global
weight tuning. GSP01 OPEN; Goal ACTIVE. All original scope, thresholds,
failures, frozen cohort/DC gaps, actual populations, final four-repository
GPU/physical/complexity checks and final independent audits remain open.

/tmp/gsp-magnetic-edge-hat-v1.py SHA256 fc4287abb20715ad7f515a0addac418ca65ac7f22e68998a1bb9cc03e5b466de.

/tmp/gsp-magnetic-edge-hat-v1/receipt.json SHA256 217a860aafb9c7b4af7c1cdadee044a808f2f6ed58c3d2a447a9a2eb08105a2c.

/tmp/gsp-magnetic-edge-hat-v1/physical.json SHA256 8deb7a543063c7da923f20f0a0f71db5c9bc4ebc36580341d7edea98da836fa1.

/tmp/test_magnetic_edge_hat_probe.py SHA256 a84f92d9aeffa7a56de6c2faaad267486a453893e9599f1ff98932e5e7d9f6d0.

/tmp/gsp-magnetic-edge-hat-v1/edge_blocks.pt SHA256 863826d82d6274e4dfc61a48c196706d90ec18676062dd46604689f8e25a51d5.

/tmp/gsp-magnetic-edge-hat-v1/enriched.pt SHA256 a3813a5c191dcfbbaf8d285b83b67d15278570e0928a28dab396fc257adf0706.

/tmp/gsp-magnetic-edge-residual-v1.py SHA256 fa850b7dbfb76c2dda99f72692ffaefae771be0b24f2e9980fb5cb951b26d416.

/tmp/gsp-magnetic-edge-residual-v1/receipt.json SHA256 350610521f85e4864801959c8b6e96430c14beca1e18bf81645e2779b8c6234b.

/tmp/gsp-magnetic-edge-residual-v1/physical.json SHA256 02ebbe8e7e01da935bea59a187e0da2b761b3ee6fe6f51deae2f191dc8600faf.

/tmp/test_magnetic_edge_residual_probe.py SHA256 996c960737b1116f5850c2ed1c6e5036d47132f523725dc80d1854f50c9f9589.

/tmp/gsp-magnetic-graded-density-v1.py SHA256 5daa0a0a912db3434f99a7968e17c59e5fe2cac2ad3f03c4cea7f68e2fb4971e.

/tmp/gsp-magnetic-graded-density-v1/receipt.json SHA256 08d7832e5aed08a6af0b51c1bffde6767a2ed67d95de7f7eec5741d1b96aefa4.

/tmp/gsp-magnetic-graded-density-v1/physical.json SHA256 0983f7fe9221722e1c2bebe7762e2aa64027bde6e6d25e6b5191d34610a97b99.

/tmp/test_magnetic_graded_density_probe.py SHA256 9f55dbbb213d2ecd3a89a18e62ff22456dc5b779dac1ca0e36d05563e8e07165.

/tmp/gsp-magnetic-graded-density-v1/power1.pt SHA256 099946a2358134c7c25be9d9f44b7b575f219a612e689d1bd4aeb2f85d0a3662.

/tmp/gsp-magnetic-graded-density-v1/power2.pt SHA256 2e2188977ddfdd4187c7a13063e75b962ce69e317b5d0eed99d3a1cb45f39886.

Shared diagnostic helper SHA during both residual and graded runs: efd2a03cb8573076fd360bc0398087c1456042cf4ebd97b0b70afee7b5071861.

Final checkpoint audit reports no confirmed P1/P2 in the exercised numerical
paths or CLI integration. It verified geometry/DOF comparisons, PV and field
layouts, stable hashes and input/partial-output provenance, retaining all
physical failures. No final-goal audit or production qualification is implied.
No live jobs remain.


## Mixed tensor harmonic cutoff integration

After preserving failed magnetic density experiments, implementation proceeds
on the separate GSP06 cutoff dependency. Mixed tensor/scalar preparation now
accepts reuse_prepared only for the exact dense operator and volume prefix.
It revalidates static electric samples, current regions and conductivities,
current tangent policy, optional magnetic volume identities/coefficients,
nonlinear coefficients, fundamental frequency and static preparation policies
and budgets. Static maps and old positive-frequency responses/bindings remain
the same objects. Only new frequencies receive fresh AC maps. Requested AC
integration settings apply only to appended frequencies.

The shared cutoff loop now calls material/preparation hooks so the mixed
runtime compiles actual tensor samples and uses its own DC preparation. It
retains the existing warm start, added-harmonic absolute budget, successive
pass requirement and strict-versus-record failure policy. Tensor DC comparison
shares the snapshot's complete static payload, including independently restored
DC operators/maps. The terminal cutoff record is attached to a copy of native
preparation, preserving the original prepared simulation and earlier Results.
Public schema21/22/23 config decoding restores the cutoff dataclass and checks
metadata/statistics against the native bound record. Existing schema layouts
remain unchanged; no new representation is claimed.

Initial targeted v1 retained two real failures: the comparison helper still
looked for scalar DC keys in mixed tensor layouts. The old scalar cutoff node
passed. Independent audit additionally found ignored static reuse policy and
budgets; exact policy binding now rejects changes before appended preparation.
An incorrect test-only Result.metadata access was corrected to the existing
internal metadata record. v2's two mixed public controls pass, including actual
linear SPD cell-estimate success and nonlinear current+magnetic budget failure,
nonzero generated harmonic, exact prefix reuse and no-solve public restoration.
No independent tensor or nonlinear physical qualification follows.

Bench removes its old blanket mixed-cutoff exclusion and validates nested
state/map/volume records through the existing cutoff history checks. Frozen
initial count/policy, retained/tail observations, terminal decisions and native
receipt binding all remain enforced. Two Bench archive controls reuse the
already solved v2 Results, reconstruct their input bindings independently, and
verify exact restore and changed-history/field rejection without new solves.
The runner did not originally hash these input files; their separate post-run
SHA inventory is explicitly labeled and must not be described as an original
runner binding.

These are implementation and targeted integration checks, not full regression,
spatial/harmonic physical convergence, general anisotropic nonlinear-volume
support, or production qualification. GSP01 and all retained failures remain
OPEN; original final scope/thresholds and four-repository/final audit obligations
remain unchanged.


### Mixed harmonic cutoff stage evidence

Public controlled cases use a scalar-limit tensor linear body and one scalar
source body. The linear SPD case extends H1→H2→H3 with zero generated fields
and two successive cell-estimate passes. The nonlinear current+magnetic case
retains actual H2 RMS .000137846691322 V/m against1e-8 V/m and DC retained
difference4.89558529240e-6 V/m against5.89558529240e-10 V/m; it correctly records
harmonic_budget_exhausted and fails both retained/tail tests. Its failed
physical/truncation status is not relabeled by an engineering test pass.

Maxwell initial v1:2failed/1passed45.693436415s,818stable hashes. Fixed v2:
2passed36.190958596s,818stable. Stage merge:10passed146.876334004s,818stable,
covering both new cutoff cases, independent restored-DC comparison, existing
fixed-count SPD/current/current-Krylov/magnetic public Results, admission, scalar
cutoff and repeated/internal cutoff behavior. Bench:2passed5.740283466s,821stable,
using v2 exact states without repeating solves. No full regression was run.

Bench archives independently recreate the frozen initial-H1 binding before
reading the reused state, preserve schema21/23 and realized harmonic count,
reject altered histories/thresholds/tails/decisions and changed final fields,
and restore without matrix assembly or linear/nonlinear solves. Schema22 uses
the same mixed record decoding; the explicit new current case includes its
magnetic schema23 superset. Existing native schema22 fixed-count restoration
is covered at the Maxwell merge point.

/tmp/gsp-mixed-tensor-cutoff-v1.py SHA256 612c55ba9476721bd7da3d4fa95851488dbc659ed5c97c0641ec32b464dce79b.

/tmp/gsp-mixed-tensor-cutoff-v1/receipt.json SHA256 df0fd92ff77d13a1e5982267a1772eeb56c90a7c81ce8d3047553ac9ecaa0f04.

/tmp/gsp-mixed-tensor-cutoff-v1/tests.xml SHA256 9baa7b736dbb407dc6150e9a5ddabac05ebca6b5ef517e060261e6b65a0b89ec.

/tmp/gsp-mixed-tensor-cutoff-v2.py SHA256 c6f2d569178112f22068e8d4041a95153b590ad9308d6d1ed40de12c8f8f876c.

/tmp/gsp-mixed-tensor-cutoff-v2/receipt.json SHA256 c4e8738e70aec5d58e653fe45f93222f8073baaed49478b73d089471998f742c.

/tmp/gsp-mixed-tensor-cutoff-v2/tests.xml SHA256 dbd0a57a6b2bde739555ecfea0f95f2d4e12f88e4ebfcd28dbefe246e46384b5.

/tmp/gsp-mixed-tensor-cutoff-merge-v1.py SHA256 4167b06e61615499526d387d675bec4ae73db5661e70e35400d996f95c341fbb.

/tmp/gsp-mixed-tensor-cutoff-merge-v1/receipt.json SHA256 4c42ddee46ce41d5e6be336a50cd4cb4628c45741e1bddae406e58435a804a87.

/tmp/gsp-mixed-tensor-cutoff-merge-v1/tests.xml SHA256 048fc682719455a2a0b1e831194acb485b8f27f639089aaf57160b28a402af85.

/tmp/gsp-mixed-tensor-cutoff-bench-v1.py SHA256 45fb3c6eabdd52db709064a6c393cba9ca3c9b29e33eb741b4bfe781fe459f6e.

/tmp/gsp-mixed-tensor-cutoff-bench-v1/receipt.json SHA256 cb25cd9d0a3c759f1180ed80eb3f6c83b0411626ded063a7de178c042c70bbc2.

/tmp/gsp-mixed-tensor-cutoff-bench-v1/tests.xml SHA256 af391c06a44be316baa22c7c05bac6290d15f334366c0b9eaffe9b54c0ddc82a.

/tmp/gsp-mixed-tensor-cutoff-bench-v1/reused_inputs_sha256.json SHA256 319936fc60d40182edcd6923db79ad47907eb89ac9488e819984473d1aafeb89.

Reused v2 input files (post-run inventory):
{
  "recording_scope": "post_run_input_inventory_not_in_original_runner_receipt",
  "files": {
    "/tmp/gsp-mixed-tensor-cutoff-v2/states/spd.pt": "147c233691105b90d1296fc3cb95f570d8af3b8f33935f78e27da6f3b595d242",
    "/tmp/gsp-mixed-tensor-cutoff-v2/states/magnetic.pt": "4c66da1cd7108e6895b77bb34a58289e6b77bf18e2d64348a84ad2d2e42882c9"
  }
}

Final independent engineering re-audit found no remaining P1/P2. It verified
the10/2 passing stage receipts, all818/821 current source hashes, independent
restored static comparison, exact archive checks and separately labeled input
SHA inventory. This is a stage audit, not final qualification. No jobs remain
live. Goal ACTIVE; original scope, thresholds and all physics/population/final
regression and final-audit obligations remain unchanged.
