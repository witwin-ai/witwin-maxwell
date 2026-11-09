# GSP-06 surface-volume and harmonic-balance implementation

Status: in progress, unqualified, no stage commit. This worktree is
<scratch>/witwin-maxwell-gsp06 on gsp06-surface-volume, based on committed07601da.
It develops scalar nonlinear prerequisites independently of GSP-05 tensor
physics; the public contract still rejects tensor+nonlinear combinations.
No change to the full GSP objective or acceptance thresholds is implied.

## Implemented constitutive primitive

mom/nonlinear.py implements instantaneous_harmonic_sources on native CUDA
complex64/complex128 arrays [H+1,...,3]. DC is real; positive entries are peak
phasors with exp(-i*n*omega*t). Internal positive/negative Fourier coefficients
are E_n/2 and conjugates, with full DC. A6H+1 temporal grid exactly resolves the
complete cubic product bandwidth [-3H,3H] without aliasing. Outputs retain all
positive harmonics through3H rather than folding or silently truncating them.

The only sources are eps0*chi2*E_i^2, eps0*chi3*sum(E_j^2)*E_i and
J_TPA=tpa_scale*sum(E_j^2)*E_i for the reconstructed real field. No linear
constitutive contrast is added. Coefficients may be matching real CUDA tensors
broadcast over spatial points; native gradients are retained. Negative TPA is
rejected. Average dissipative power is computed from the time samples. The
polarization current is -i*n*omega*P_n; its zero DC current does not remove DC
polarization, which must drive a separate electrostatic volume block.

/tmp/gsp06-harmonic-sources-v1.log:3passed,4.72s. Independent scalar trig
identities check chi2 DC/2omega and chi3 fundamental/3omega amplitude and phase;
vector/multifrequency TPA checks Parseval power and coefficient gradients;
a third-harmonic input generates the ninth harmonic without aliasing.

## Required next work

- Native volume mesh/basis with body ownership and validated surface coupling.
- Linear coupled SIE-VIE operator, zero-volume-source exact PMCHWT reduction,
  and explicit double-counting falsification.
- Public Material sampling of nonlinear coefficients with existing FDTD units.
- DC electrostatic block, conjugate real-field constraints and harmonic sets.
- Newton/Picard, continuation, branch/multistability identities, power/residual
  gates, harmonic truncation and volume convergence.
- Scene/Simulation/Result composition, multi-body nonlinear gate, exact storage,
  FDTD independent reference and final physical qualification.

No solver, nonlinear production capability, full regression or audit is claimed
by these constitutive checks. GSP-05 work/evidence stays in its separate tree.

## Volume geometry and exact surface ownership

New volume_mesh.py validates owned positive CUDA tetrahedra, host face
incidence/orientation/connectivity, and the existing fully validated closed
SurfaceMesh boundary. Geometry gradients are rejected. Interior Duffy cubature
keeps SI m^3 weights on CUDA. Volume/boundary tensor versions and region identity
are guarded. Surface association is never inferred solely from a region number.
bind_volume_region requires a bijective boundary vertex match (at the existing
mesh roundoff tolerance), exact matching triangle sets, and the actual named
body in a validated SurfaceSystem. Its bound record guards both systems and
correspondence mutation. This is geometry binding, not a coupled VIE operator.
CellVectorBasis provides three physical discontinuous Cartesian components per
cell and a native complex mass application with gradients. No linear contrast
or volume Green self term is introduced by this primitive.

/tmp/gsp06-volume-geometry-v1.log:5passed,4.11s: exact tetrahedral volume/first/
second moments, complex mass derivative, ownership-copy/stale mutation, and
negative/degenerate/duplicate/invalid-region rejection.
/tmp/gsp06-volume-surface-binding-v1.log:1passed,4.60s: conforming four-cell
subdivision preserves volume and binds its actual surface; a displaced body
and modified correspondence are rejected. Region-ID guards were tightened
after this run; broader stage checks remain pending.

Next implement volume-current Green integration and its singular/self terms,
then the coupled operator with zero-source reduction and double-counting
falsification. Existing mass and geometry tests do not establish physical VIE
accuracy or public harmonic-balance capability. No phase commit or full suite.

## Scalar volume Green potentials and distributional self fields

New vie_operator.py computes tetrahedral integral(g), integral(grad g) and
integral(distributional Hess g). The latter two use the divergence theorem on
oriented cell faces. The scalar volume potential uses signed query-centered
radial fans and an analytic integral of t*(exp(z*t)-1); all Coulomb face and
volume contributions are restored analytically. Existing exact static panel
gradients are reused. This includes the interior contact term rather than
sampling a singular point Hessian or omitting the delta distribution.
cell_current_fields applies E=i*omega*mu*Vg*J+i/(omega*epsilon)*VHess*J and
H=Vgrad cross J, preserving native current and query gradients. Inputs are
constant electric volume currents in A/m^2, not prescribed new public sources.
Queries on cell faces explicitly fail pending a trace formulation. These are
fixed-order primitives; adaptive convergence, source-face traces, stable far
attenuated evaluation and coupled operators remain to implement/qualify.

Evidence retained:
- /tmp/gsp06-volume-green-v1.log: collection error from the initial wrong import
  location of _triangle_distance; corrected to the existing fields helper.
- /tmp/gsp06-volume-green-v2.log:2passed,4.69s, static interior Hessian trace=-1,
  symmetry, face-query rejection and off-cell direct volume cubature agreement.
- /tmp/gsp06-volume-maxwell-v1.log:1failed,4.63s. Interior Maxwell source identity
  exposed about0.001266 relative error because scalar face potentials were still
  numerically integrated while their static gradients were analytic.
- /tmp/gsp06-volume-green-static-extraction-v1.log:3passed,5.14s after analytic
  static scalar face/volume extraction. All prior numerical thresholds are
  unchanged; curl H+i*omega*epsilon E=J is checked through native derivatives
  inside the source tetrahedron, not just in a source-free observation region.

No full regression, stage commit, coupled-solver claim or qualification.

## Adaptive complete volume-current fields

evaluate_cell_current_fields now uses the shared FieldQueryConfig to refine
complete E/H separately at every query. Absolute tolerances use V/m for E and
V/m divided by |eta| for H. Accepted fields retain native gradients; previous
refinement graphs are detached. Nonfinite values and exhausted budgets fail
explicitly with VolumeFieldConvergenceError receipts. Fixed-order opt-out
reports converged=None and cannot imply qualification. No linear contrast is
added; receipts explicitly retain false physical qualification.
/tmp/gsp06-volume-adaptation-v1.log:1passed,5.05s. Actual interior/exterior
queries converge, agree with a higher-order field evaluation, preserve current
gradients and reject an actual insufficient order budget. This remains an
empirical refinement estimate, not an error bound or coupled-solver gate.

Next source-face principal values/traces are needed for MoM boundary testing.
Use exact static panel PV and avoid coincident smooth quadrature nodes with
query-centered triangle fans. Tangential boundary testing is continuous across
the volume-source normal field jump; a full normal trace requires explicit
ownership/side semantics. Do not silently offset queries or drop self terms.

## Cell-face principal values and near-face quadrature

Volume Green/current/adaptive queries now accept explicit principal_value=True.
Static panel gradients remove the normal half jump only on identified cell-face
interiors. Query-centered Duffy fans replace near-panel quadrature, avoiding
coincident points without offsets or cutoffs; distant panels keep the original
rule. Multiple incident faces within one cell identify edges/vertices and fail
explicitly. Default face queries still reject missing trace semantics. Receipts
identify principal_value_per_cell rather than asserting an interior/exterior
normal trace. This is sufficient for tangential MoM testing; full normal traces
still require explicit owning-cell/side semantics.

/tmp/gsp06-volume-face-pv-v1.log:1failed,5.07s. The first implementation fanned
only exact-face queries; its off-face comparison retained underresolved regular
near-panel quadrature (about7.52e-5 relative discrepancy). Near panels now use
the same query-centered fan treatment.
/tmp/gsp06-volume-near-fan-v1.log:5passed,7.73s, all numerical thresholds
unchanged. Coverage includes PV vs two-sided extrapolation, the analytic normal
E jump i*n*(n.J)/(omega*epsilon), edge rejection, interior Maxwell source
identity, static contact term, independent direct cubature and adaptive budgets.
No coupled SIE-VIE or nonlinear production qualification follows yet.

## Owning-region volume RHS in the lifted surface equations

New surface_volume.build_volume_source_rhs tests adaptive principal-value
volume fields as [+<f,Hv>,-<f,Ev>] on the exact owning MoM body. This is opposite
the incident-wave RHS sign, following exterior+interior surface continuity.
Each volume must bind the same SurfaceSystem object and have a unique region;
its contribution occupies only that body's global J/M rows. The existing
exterior surface operator transports inter-body interaction. No extra linear
constitutive contrast is introduced. Metadata marks coupled_sie_vie_linear
response groundwork, induced-current role and false qualification; outer RWG
testing is currently fixed-order and explicitly lacks an error estimate.
This private primitive does not expose new prescribed interior sources.

/tmp/gsp06-surface-volume-rhs-v1.log:1failed,17.27s. Zero-source RHS and exact
original PMCHWT solve equality passed before the nonzero source exhausted the
chosen max_order24 field budget. /tmp/gsp06-boundary-query-budget-v1.log shows
actual refinement2/4/8/16/32 converging at32 with maximum E difference9.68384e-6
relative and H6.80662e-7, under unchanged rtol1e-5/atol1e-10.
The test now uses the existing default max_order64 instead of24.
/tmp/gsp06-surface-volume-rhs-v2.log:1passed,17.78s. Actual two-body scalar
surface matrix, exact zero-volume-source solve reduction, nonzero region2-only
RHS, native current gradients and duplicate-volume ownership rejection pass.
The original insufficient-budget failure remains; no tolerance was relaxed.

Next complete volume-field response through the existing surface solve,
linear-background double-counting falsification, outer surface testing
convergence, DC polarization block and nonlinear harmonic iteration.

## Differentiable dense surface-volume response

DenseVolumeResponse snapshots the existing SystemOperator matrix under its
read lease and uses native torch.linalg.solve, preserving RHS/current gradients.
The general reference solver intentionally runs under no_grad, so calling it
here would silently break the nonlinear chain. This private candidate neither
reassembles PMCHWT nor exposes prescribed internal sources. A true unweighted
surface residual is checked at 512 machine eps times RHS norm. This algebraic
check does not confer physical qualification. The response owns cloned source
arrays, guards coefficient/current/matrix mutations and validates volume bindings.
It reuses decode_system for the lifted surface fields, adding direct regional
volume fields only inside the owning body. Exterior and other-body interaction
passes through the existing coupled surface matrix. Fields currently retain
separate component estimators: complete summed-field error is not certified.

/tmp/gsp06-volume-response-v1.log:1passed,14.06s. Actual two-body matrix:
native source-amplitude derivative agrees with the affine solution difference;
current-to-E gradients are finite/nonzero in exterior and both bodies; direct
volume field inclusion is checked by region; owned-current mutation rejects.
No physical accuracy threshold is claimed by this composition check.

New adaptive_volume_source_rhs refines the outer RWG testing order separately
from inner adaptive volume field queries. Successive-order error uses the
existing impedance-balanced L2 dual proxy; absolute tolerance is explicitly
in that norm, not V/m. Accepted RHS keeps its native graph; old estimates are
detached. Exhaustion raises VolumeTestingConvergenceError with actual checks.
This is a numerical estimator, not a certified global RHS bound.

/tmp/gsp06-outer-volume-testing-v1.log:1passed,41.87s. Actual two-body
operator: zero-current outer refinement2->4 remains exactly zero and retains
autograd; nonzero current under deliberately insufficient max_order4 and
rtol1e-14/atol1e-20 raises the explicit outer-budget exception with measured
difference exceeding its threshold. Nonzero accepted outer convergence is
not yet demonstrated. DenseVolumeResponse currently uses fixed outer testing;
wire the adaptive outer estimator and complete summed-field convergence before
promoting this private candidate. No full regression was repeated this turn.

## Required outer adaptation and complete summed-field refinement

DenseVolumeResponse now requires adaptive_volume_source_rhs before solving.
Its outer max order/dual-proxy tolerances are explicit constructor parameters;
the accepted source receipt remains separate from the algebraic residual.
fields() resolves names through SurfaceSystem.region, refines all body surface
quadratures and the owning volume Green at the same order, sums the complete
fields, then requires separate E/H successive-order differences to meet the
requested tolerances at every query. H absolute tolerance is E tolerance divided
by the querying region's |eta|. Previous graphs are detached; accepted graphs
remain native. A copied decode handle prevents changing the original operator's
integration settings. Explicit fixed-order diagnostic queries remain available,
with no convergence claim. Nonfinite fields or exhausted budget raise
CoupledFieldConvergenceError carrying the actual receipt. These checks estimate
quadrature convergence, not discretization error or certified physical accuracy.

Volume Green query integration now uses batches of16 queries instead of a
Python loop per query, preserving per-cell distributional contacts, face PV,
near-panel fans and native gradients. This bounds the query multiplier, not
overall mesh workspace. /tmp/gsp06-volume-query-batching-v1.log:5passed7.46s,
existing direct-cubature, Maxwell-source, staticcontact, adaptivebudget and
face-PV/jump checks, unchanged tolerances.

Pending targeted runs: /tmp/gsp06-response-outer-adaptive-v1.log (started before
query batching), and /tmp/gsp06-complete-field-adaptation-v1.log (batched kernel,
query-batch ordering plus nonzero required outer convergence and complete-field
refinement/reference/budget/gradient checks). Their outcomes must be read before
claiming these newly integrated paths passed. No full regression/audit repeated.

/tmp/gsp06-complete-field-adaptation-v1.log:1failed,1passed,81.07s;
session23398 terminal1. The batch-boundary/near-far ordering check passed.
Nonzero outer RWG testing exhausted max_order32 at rtol1e-4/atol1e-12 before
complete-field checks could run. This failure is retained and is not a field
accuracy pass. The earlier unbatched session42857 was still running at last
poll; do not restart or infer a terminal status from its empty log.

The outer electric testing now integrates by parts over complete RWG support:
<Ev,f> = i*omega*mu*<Vg J,f> - i/(omega*epsilon)*<J.Vgradg,div f>.
Shared-edge contributions cancel exactly in the continuous RWG space. This
avoids directly testing the boundary Hessian singularity while preserving
regional material response exactly once. Magnetic testing remains <Vgradg x
J,f>. The existing adaptive volume field evaluation supplies the accepted
source quadrature order; vector/scalar potentials are evaluated at that order.
This extra potential evaluation is currently redundant work, to consolidate
after correctness. Weak source-potential convergence itself is not yet
independently certified by that inner E/H criterion. Versioned receipt uses
electric_testing=full_rwg_support_divergence_volume_potential_v1.
Outer budget exceptions now include their measured check list in the message
so a failed test preserves actionable convergence data.
New targeted run /tmp/gsp06-complete-field-weak-testing-v1.log, session24142,
uses the same max_order32, rtol1e-4/atol1e-12 outer thresholds and complete-field
thresholds as the failed test; outcome pending.

## Weak-form outer acceptance and refined field configuration

/tmp/gsp06-complete-field-weak-testing-v1.log:1failed121.19s. The nonzero
outer RHS passed its unchanged rtol1e-4/atol1e-12/max_order32 requirement; the
test then failed while creating a refined IntegrationConfig at order24 with
near_max_order16. This is configuration validation, not physical acceptance.
Both the new coupled fields() and the existing adaptive multibody exterior
decoder now set the copied near budget to at least refined singular_order+2.
The original integration config remains unchanged. The high-order test oracle
likewise constructs a valid order64 config.
/tmp/gsp06-multibody-field-refinement-config-v1.log:1passed7.71s, real two-body
operator, initialorder12 refined>=24, true decoder path and unchanged original
near budgets. /tmp/gsp06-complete-field-weak-testing-v2.log session88639 pending
at last poll; numerical thresholds unchanged and actual checks printed.

The superseded pre-batch/direct-Hessian session42857 was positively identified
as PID2127094, exact argv checked and SIGINT sent to avoid duplicate work.
It is terminal exit2; its log retains KeyboardInterrupt after463.27s. This is
an intentional supersession, not a numerical convergence failure. Details in
/tmp/gsp06-response-outer-adaptive-v1-supersession.json. The direct-Hessian
outer failure remains independently recorded by terminal session23398.

## DC polarization volume response

cell_static_polarization_fields implements phi=-integral(grad(g0).P)/epsilon
and E=integral(distributional Hess(g0).P)/epsilon using the same validated
tetrahedral geometry. It requires real finite DC polarization and positive
finite absolute regional permittivity, preserves native P/epsilon/query
gradients, and includes the static contact without dividing by frequency.
This is a regional volume primitive; no dielectric interface conditions or
zero-frequency material sampling are claimed. Conducting/degenerate DC
limits and complete coupled HB DC response remain pending.
/tmp/gsp06-dc-polarization-v1.log:1passed6.65s. Interior E=-grad(phi), native
permittivity derivative, independent far dipole limit and zero-epsilon rejection.
The DC primitive carries interface_conditions_solved=False and qualification=False.

/tmp/gsp06-complete-field-weak-testing-v2.log:1passed116.50s; session88639
terminal0. Actual outer order2/4/8/16/32: final dual difference1.14142605e-5
versus threshold6.92258160e-5. Complete E/H refinement3/6/12/24 accepted24:
E relative difference1.16894e-9, H5.06798e-8, both under rtol1e-5 plus stated
absolute tolerances. Independent-order64 evaluation agrees at the test's
rtol2e-5/atol1e-10; native current gradient and deliberately exhausted order6
complete-field failure receipt pass. These establish implementation-level
quadrature behavior, not physical discretization/production qualification.

## DC dielectric interface candidate and retained physical-reference failure

New static_coupling.StaticVolumeResponse takes explicit positive real absolute
DC permittivities, validated SurfaceSystem and exactly bound volume polarization.
It constructs a global single-layer potential with constant face charges and
face-centroid collocation. Each region's particular potential uses its own
permittivity; interface correction couples all body boundaries. Native Torch
solve, source/epsilon gradients, geometry-region query validation, owned tensors
and mutation guards are retained. Zero-frequency material sampling, conductor
limits and public API integration remain separate unfinished requirements.

With K'=n.grad(S), the normalized system is
[I+2(eps_i-eps_e)/(eps_i+eps_e) K'] sigma
=2(eps_i-eps_e)/(eps_i+eps_e) n.E_particular_ext.
The exterior particular trace adds the owning volume's n.P/(2 eps_i) normal
half jump to its PV. Full normal E jump adds sigma, while potential is
continuous. All direct polarization potentials are evaluated globally; this
DC particular-plus-interface formulation is distinct from regional finite-
frequency PMCHWT and introduces no second linear volume contrast.

/tmp/gsp06-dc-interface-v1.log:1passed,1failed12.71s, session21249terminal1.
Actual tetrahedron test passes eps_e Eext_n-eps_i Eint_n=P_n, tangential E
continuity, potential differentiation, source gradients and exact matched-
medium sigma=0 reduction. Polarized sphere levels1/2 (80/320 panels with
conforming center-fan tetrahedra) produce relative interior-field errors
0.05602743458 and0.03081161243 against E=-P/(eps_i+2eps_e). Refinement reduced
error but failed the test's factor2 reduction requirement. This diagnostic
rate was not a production tolerance, and it remains failed rather than being
loosened. No physical qualification follows. Improve surface discretization
and validate actual accuracy before combining this DC candidate into a
production HB path. Explicit public material sampling is still absent.

## DC face-averaged Galerkin improvement

StaticVolumeResponse now integrates the exact constant source-panel gradient
over each testing face using configurable testing_order(default4), normalized
by face area. The particular source and its normal jump are averaged with the
same quadrature. boundary_fields now explicitly returns one-sided face means;
the algebraic transmission equation is tested in that space. The charge basis
remains constant per face. Numerical identity changes to
dc_volume_polarization_constant_panel_galerkin_v2. Fixed-order testing is not
yet an adaptive integration bound and surface discretization error remains
uncertified. Query fields retain exact static source-panel integrals.

The k=0 volume Green path now directly uses the analytic panel potential and
gradient; it skips identically zero dynamic remainder quadrature. Existing
complex return dtype, self contacts and query/source gradients are preserved.
/tmp/gsp06-dc-galerkin-v2.log:4passed29.90s, session52368terminal0. Original
static self/DC gradient/jump/matched-medium and factor2 sphere refinement
requirements are unchanged. Sphere level1/2 errors are0.01670785328 and
0.005560560831. The original collocation 5.60%->3.08% rate failure remains
in its log. New fine error0.556% still does not close final0.5% accuracy gates.

## Public harmonic material sampling

compile_harmonic_material constructs the scalar linear background via public
Material constructor parameters while extracting only public nonlinear_chi2,
nonlinear_chi3 and tpa_sigma_scale for the nonlinear source law. Each positive
harmonic reuses compile_medium, so dispersion and conductivity occur once.
StaticChannel retains the exact small-frequency finite relative part, DC
conductivity and lossless-plasma weight; damped Drude contributes sigma=
absolute_scale*omega_p^2/gamma and finite part -omega_p^2/gamma^2. Finite
Debye/Lorentz DC susceptibilities are their delta values. Conductive, lossless
plasma, unresolved complex and nonpositive/degenerate DC cases cannot be
requested as finite positive dielectric samples. This classifies missing
limit formulations without claiming they are solved. General DC dissipative
current/magnetostatic coupling remains unfinished, as does nonlinear HB.

/tmp/gsp06-harmonic-material-v1.log:2failed3.21s because Material inherits
material_id, which dataclasses.replace wrongly passes to its public constructor.
The adapter now explicitly forwards all existing linear constructor fields.
/tmp/gsp06-harmonic-material-v2.log:2passed3.93s, session46789terminal0.
Checks public combined chi3/chi2/TPA, positive-frequency linear samples, damped
Drude DC conductivity/finite part and undamped plasma classification. Native
CUDA source check verifies chi=0 gives exactly zero nonlinear source despite
eps_r=3 (a deliberately added linear contrast would be nonzero), and public
chi2 optical-rectification amplitude. This is source-law falsification, not
a completed coupled-solution double-counting physical oracle.

## Dense retained-harmonic nonlinear algebra

New harmonic_balance.DenseHarmonicBalance accepts explicit cellwise linear
background E, P->E and J->E maps for DC and every retained positive harmonic,
cell volumes and the existing native nonlinear law. Positive-frequency maps
must obey G_P=-i*n*omega*G_J. DC maps must be real. This class does not
assemble physical response maps or infer a zero DC-current response from a
material; physical preparation and unresolved DC conduction remain pending.
The real nonlinear unknown vector stores DC.real and separate AC real/imag
blocks, enforcing real DC without redundant imaginary unknowns. Sources retain
the complete generated cubic bandwidth; final receipts report omitted P/J
norms with distinct physical units, TPA cell-volume dissipation, and explicitly
false harmonic_truncation_converged and physical_qualification.

Newton/Picard use actual residual checks and damped Armijo backtracking.
Named continuation paths warm-start the explicitly selected initial state,
retain completed-step receipts on failure and never restart a different branch
silently. Branch stability is explicitly unassessed. Native dense Jacobians
use supported rowwise autograd on CUDA; implicit first derivatives use the
converged equation and a detached corrected-root Jacobian. No higher-order
derivative or fast-backend claim is made. Singular-value checks reject
numerically singular Newton/implicit Jacobians; this is not a stability proof.

Targeted tests are deliberately algebraic fixtures, not EM production oracles.
/tmp/gsp06-harmonic-balance-algebra-v1.log:3failed4.54s, odd DC-prefix storage
offset was incompatible with view_as_complex. v2:3failed6.15s, vectorized
complex Jacobian backward encountered unsupported strides. Packing now uses
separate real/imag blocks and torch.complex; Jacobian evaluation is native
rowwise CUDA autograd instead of experimental vmap.
v3:2passed1failed12.90s. Linear reduction, cubic equation/implicit derivative,
Picard agreement, DC/second-harmonic retention, omitted-tail receipts, P/J
contract and actual iteration-budget failure pass. Zero-pump nonzero cubic
root has phase/orientation degeneracy, so its Jacobian is singular.
The continuation fixture now checks two distinct roots along nonzero pumps
(.2,.5,1.) and separately requires the zero-pump degeneracy to fail explicitly.
/tmp/gsp06-harmonic-branch-v4.log:1failed12.58s; v5:1failed17.28s. Both
nonzero branch paths succeeded, but zero-pump degeneracy was detected only at
the next ramp step. The approximate root's Jacobian can be invertible while
the corrected root is singular. The solver now recomputes/checks the Jacobian
at the corrected root before constructing the final implicit-gradient graph.
/tmp/gsp06-harmonic-balance-algebra-v6.log session19189 is the current scoped
validation of that corrected-root guard plus the previously affected gradient
path; inspect its outcome before claiming the final implementation passed.

/tmp/gsp06-harmonic-balance-algebra-v6.log:3passed26.69s, session19189
terminal0. Corrected-root singularity now fails at the actual degenerate
zero-pump step; nonzero-pump distinct branches succeed. Cubic implicit gradient,
linear reduction, Picard agreement, DC retention and budget/tail contracts pass.
These are algebraic references only. Next prepare response matrices from the
actual coupled surface-volume discretization; avoid repeated nonlinear solves
for basis columns by assembling weak volume-to-surface maps and solving all
RHS with the existing surface matrix. Each map column's quadrature and complete
field error must be checked, not inferred from a zero-source probe.

## Direct weak volume-source matrices and cell response preparation

New volume_response_matrix.py assembles all cell-xyz source columns using
H testing coefficient f cross Vgrad(g), and E coefficient
i*omega*mu*Vg*f - i/(omega*epsilon)*div(f)*Vgrad(g). Global J/M rows belong
only to each source region; column order follows bindings, cells, xyz. Native
per-column current action avoids rerunning complete source solves per basis.
Independent inner and outer order refinements each receive half the requested
existing dual-proxy tolerance and check every column. This is not a rigorous
bound for arbitrary cancelling source combinations.
/tmp/gsp06-volume-source-matrix-v1.log:2passed49.56s, session68187terminal0.
Actual two-body matrix action matches direct adaptive current evaluation,
column ownership and native source gradients pass. Adaptive source matrix
accepts outer32/inner8; maximum outer threshold ratio0.382959, inner0.038561.
An intentionally insufficient budget raises VolumeMatrixConvergenceError.

prepare_cell_field_response now solves incident plus all source RHS together
using the existing actual surface matrix and checks each true residual. It
decodes regional RWG response directly (no subtraction of two incident fields),
adds the own-region direct volume Green columns, and refines complete E/H
for every unit-current column and cell-centroid query. The original incident
background uses the existing lifted decoder. Source map, complete-field and
background receipts remain distinct. Cell-centroid collocation is explicit;
volume discretization and physical errors are not certified.
/tmp/gsp06-prepared-cell-response-v1.log session12770 remains running at last
poll; compares the prepared map with a direct coupled induced-current solve
and source gradients. Do not infer success before collecting the result.

## Resume: physical cell maps and first actual dielectric HB integration

The paused session12770 handle is now absent; authoritative completed log
/tmp/gsp06-prepared-cell-response-v1.log reports1passed134.41s. Its complete
E/H columns accepted order24. Prepared maps agree with the existing direct
coupled current solve at the unchanged3e-4 comparison tolerance; current
gradients pass. Do not rerun this covered check.

New harmonic_preparation.py prepares fixed-order DC E/charge response columns
by native differentiation of the linear StaticVolumeResponse. Zero input
does not select an adaptive order; testing_order is explicit and no accuracy
certificate follows. /tmp/gsp06-prepared-dc-response-v1.log:1passed4.99s,
session56582terminal0; nonzero actual DC E and surface-charge action agree.

prepare_dielectric_harmonic_balance validates the common body geometry/table,
every positive-frequency background sample, material/frequency axes and
nonlinear volume coverage before preparing maps. It binds real DC P maps and
actual AC current maps, derives AC P maps by -i*omega, and supplies public
per-cell chi2/chi3 to DenseHarmonicBalance. DC current response is zero only
for this explicitly finite-dielectric polarization candidate. Nonzero TPA
rejects because its mean-current DC interface formulation is unfinished;
singular/conductive material DC samples also fail explicitly. These remain
requirements of the full Goal, not excluded final scope.

/tmp/gsp06-actual-dielectric-hb-v1.log:1passed17.36s, session23767terminal0.
Actual tetrahedral dielectric, fundamental300MHz and second harmonic600MHz
operators, incoming wave only at the fundamental, public chi2=.002/chi3=.001.
The coupled solution has residual4.079619912384109e-18, DC field norm
0.0001221664050812603V/m, second-harmonic norm0.00013757099528571233V/m.
This is the first actual SIE-VIE-to-HB integration check, not an independent
physical oracle or a harmonic/volume discretization certificate. Final
truncation, power, FDTD/reference, precision/device and production gates remain
open. Next recover the actual harmonic surface/volume state and public Result
semantics, then extend unresolved DC current/limits and acceptance paths.

## Harmonic physical-state recovery

New harmonic_state.py recovers each AC scattered RWG vector as
y_h = pump_scale*y_background,h + response_h*J_h, with
J_h=-i*n*omega*P_h+J_dissipative,h exactly once. DC potential-layer density
is recovered separately from P0; it is never mislabeled as an RWG current.
HarmonicCoupledState stores the actual CUDA frequency axis, cloned cell E/P,
dissipative and total volume currents, all AC surface vectors and DC charge.
Recovery recomputes sources from the converged field and verifies the actual
retained-harmonic residual. Preparation signatures bind tensors, exact operator
matrices and volume ownership; another model or changed preparation rejects.
This is an internal data object, not yet a public Result/query/snapshot API.

/tmp/gsp06-harmonic-state-v1.log:1passed18.03s, session64838terminal0. Actual
two-frequency dielectric HB from the existing fixture, half-pump solve, exact
frequency axis, nonzero DC charge, zero DC polarization current and each
harmonic's reconstructed surface equation pass. Returned field storage does
not alias the solved field; preparation map mutation rejects recovery. The
test subsequently uses a saved tensor clone instead of repeating the same
half-pump solve for its no-alias assertion, avoiding redundant work. Existing
physical/truncation qualification remains false.
Next public field decoding must scale the immutable analytic incident carrier
consistently with pump_scale, retain the complete summed-field estimator and
keep DC potential/electric semantics distinct from positive-frequency E/H.

## Recovered harmonic field decoding

harmonic_fields.py now decodes retained AC E/H through VolumeFieldResponse,
reusing the complete summed-field estimator and recovered surface/current
arrays. Immutable global and body incident carriers are scaled by the actual
continuation pump. DC potential/E is queried separately; no DC radiative H is
invented. HarmonicCoupledState guards owned tensor versions; preparation
signatures now also bind primitive material/order parameters.

/tmp/gsp06-harmonic-fields-v1.log: 1 passed in 18.08s, terminal 0. AC fields at
both retained frequencies agree with the HB cell solution within 3e-4; real
DC E agrees within rtol 1e-8/atol 1e-10. DC through the AC API and mutated
states reject. This first decoder rebuilt the DC interface once lazily.

The current decoder removes that redundant solve: StaticVolumeFieldResponse
shares source validation and field evaluation with StaticVolumeResponse, but
accepts recovered charge without assembly. Charge consistency is checked
against the prepared linear map. /tmp/gsp06-harmonic-fields-reuse-v2.log:
2 passed in 18.09s, session 48055 terminal 0. The actual two-harmonic query
fixture explicitly forbids torch.linalg.solve during the first DC query;
scalar DC permittivity/order mutations reject. The existing dielectric jump,
matched-medium, potential-gradient and source-gradient checks also pass after
the shared decoder refactor. No full suite or repeated physical sweep ran.

Public Result/persistence integration remains next. Conductive/mean-current
DC, TPA physical coupling, volume/harmonic convergence, power, independent
reference and production gates remain open. All recorded probe jobs are
terminal; no stage completion or phase commit is claimed here.

## Public finite-dielectric harmonic entry

HarmonicBalanceConfig and TetrahedralVolumeMesh are now public exports.
MoMConfig.harmonic_balance explicitly selects retained DC..H, volume meshes,
branch_id, Newton/Picard tolerances/damping and a named pump continuation.
Simulation.prepare dispatches to HarmonicRuntime using shared scene scope and
source checks. Per-harmonic linear samples compile the actual MoM operators;
nonlinear material channels feed only the induced-volume constitutive law.

The runtime returns the shared Result with a separate HarmonicData payload.
Result.harmonic_state exposes all recovered unknowns; query_fields requires
an explicit positive harmonic, and query_dc returns real potential/E. The
Result frequency axis is the actual retained (0, f, ..., H*f), with fundamental
frequency stored explicitly in metadata. The constructor binds this axis to
the harmonic payload. Continuation receipts retain each step and branch name;
algebraic convergence does not certify truncation, stability or production.

/tmp/gsp06-public-hb-v1.log: 2 passed in 18.11s, session45208terminal0.
A public Mesh Scene with chi2=.002/chi3=.001, 300MHz fundamental, H=2 and
pump path (.5,1.) solves via Simulation.run; Result has (0,300MHz,600MHz),
AC query/cell agreement <3e-4, real DC agreement rtol1e-8/atol1e-10, actual
surface/volume unknowns and named branch receipts. Invalid config and the
unqualified multi-body nonlinear combination reject. Exact snapshot support
is next; save currently raises before filesystem writes rather than silently
omitting harmonic data. This remains an implementation gap, not final scope.

/tmp/gsp06-public-shared-scope-v1.log: 2 passed in 2.41s,
session45391terminal0. Existing linear public scope and shared scene boundary/
internal-source guards pass after extracting common structural validation.
No full suite or independent audit was run at this intermediate point.

Public TPA/mean-current DC, singular DC samples, harmonic monitors, material/
geometry compilation gradients and multi-body nonlinear combinations reject
explicitly. The full Goal still requires these applicable physics/combination
capabilities; fixed-geometry native internal source gradients are not a public
material-gradient claim. Next implement exact harmonic snapshot/restore without
re-solving, then proceed with the remaining physics and acceptance work.

## Exact harmonic snapshots and solve-free restore

harmonic_snapshot.py defines a strict version-1 coupled dielectric payload,
separate from the scalar lifted SurfaceData schema. It stores actual DC/AC
state arrays, native volume topology, frequency samples, incident carriers,
configuration, source/material/geometry scene signature, prepared dense
responses and matrices, plus retained branch/solver receipts. CPU snapshot
copies are detached and own their storage. Result.save/load dispatches to the
native harmonic payload and rejects missing or mixed native representations.

Restore recompiles geometry and public material samples to check the supplied
Scene, binds saved volume meshes to each actual harmonic surface system and
creates SystemDecodeHandle objects with the saved matrices. It never calls
assemble_system, the HB solve or a linear interface solve. Stored constitutive
sources and -i*n*omega current accounting, real DC, frequency axis, array
shapes/dtypes/finiteness, DC material/order, representation identities and
unqualified/converged receipts are checked. Surface and DC charge are compared
with prepared response actions; retained cell E must meet the recorded solver
threshold. This validates stored consistency, not independent physical truth.

/tmp/gsp06-harmonic-snapshot-v1.log: 2 passed in 19.07s,
session65249terminal0. Exact public H=2 snapshot/load arrays and AC/DC query
roundtrip pass while matrix assembly and torch.linalg.solve are forbidden.
Corrupted volume current rejects. After adding response/receipt checks,
/tmp/gsp06-harmonic-snapshot-v2.log: 2 passed in 19.14s,
session45882terminal0. Also verifies restored-data re-save and rejection of
false physical qualification and inconsistent AC surface coefficients.
No full regression or independent audit at this intermediate point.

These snapshots retain dense preparation arrays; no compression/scaling claim
is made. Cross-device/precision acceptance and broader malformed-payload tests
remain for the merge-point audit. Harmonic monitors, DC current/TPA and singular
limits, multi-body nonlinear acceptance, harmonic/volume refinement, power and
independent references remain open. Next progress the missing physics/product
capabilities rather than repeat these completed snapshot checks.

## Harmonic PointMonitor spectra

The shared PointMonitor path now records positive-frequency E/H arrays with
an explicit (f,...,H*f) axis. A separate dc record carries frequency=0, real
requested E components and electrostatic potential; H-only monitors do not
invent a DC magnetic result. Result itself retains the full DC..H frequency
axis. Result.monitor frequency selection uses the monitor's own positive
axis, and query_dc remains the explicit DC query API.

HarmonicData.sample_point_monitors deduplicates physical positions, batches
queries by owning region and harmonic, and records complete-field query
receipts. DC evaluation is performed only where at least one electric
component was requested. On-interface positions reject because the coupled
trace formulation is still unfinished. Existing shared PointMonitor name,
component and CUDA checks apply; non-point monitors remain unsupported.

/tmp/gsp06-harmonic-monitors-v1.log: 2 passed in 19.93s,
session35112terminal0. The existing public H=2 fixture now requests all E/H
components plus a coincident H-only monitor. Each frequency-selected component
matches direct queries exactly, DC E matches query_dc, only one unique position
is sampled, and saved/restored monitor arrays are exact. The solve-free native
state snapshot checks run in that same fixture; no separate repeated solve or
full suite was launched. Physical qualification and harmonic truncation gates
remain open. Next advance power/truncation and unresolved DC-current physics.

## Local nonlinear harmonic power ledger

nonlinear.harmonic_power_ledger integrates cell work by retained harmonic:
DC uses Re(E0*J0), positive peak phasors use 0.5*Re(conj(E_n)*J_n).
Polarization and dissipative currents have separate W arrays; positive work
means field-to-matter transfer. Conservative polarization may exchange power
between harmonics but its total cycle work must vanish. Dissipative work is
checked against the separately time-collocated constitutive dissipation using
a roundoff-scale threshold. Returned tensor powers retain native gradients.
Converged DenseHarmonicBalance receipts contain a detached numerical ledger,
which follows existing Result continuation/state/snapshot receipt storage.

/tmp/gsp06-harmonic-power-v1.log: 1 passed in 3.39s, session91698terminal0.
A two-cell vector field with DC, complex fundamental and second harmonic,
chi2=.4/chi3=.7 and TPA=.6 is checked against 4096 direct trigonometric time
samples and the analytic time derivative of the nonlinear polarization.
Every harmonic's work agrees at rtol1e-12; exchange is nonzero between positive
harmonics but sums conservatively. TPA dissipation and coefficient gradient
agree; omission of the peak-phasor half factor fails the nonzero-DC fixture.
This reference does not replay the source FFT implementation.
/tmp/gsp06-power-receipt-v1.log: 1 passed in 3.39s, session9479terminal0.
Existing DC/second-harmonic algebra fixture verifies the converged ledger,
zero dissipative work and explicit false global-power/truncation flags.

All jobs terminal. These are constitutive/algebraic gates, not the remaining
full-field power or Manley–Rowe acceptance. Next combine independently
integrated exterior Poynting flux, regional linear absorption and nonlinear
work, then quantify discretization/harmonic convergence. Physical TPA still
requires the unfinished mean-current DC interface; source-power validation
does not enable that public capability. No full regression/audit/phase commit.

## Complete-field power diagnostic

New harmonic_observables.py backs Result.harmonic_power_balance. An owned,
validated convex control SurfaceMesh must strictly enclose every body; triangle
quadrature integrates scattered plus incident-interference Poynting flux.
The source-free incident closed-surface flux is analytically zero. Body volume
rules come from the actual bound tetrahedra; independently queried E/H gives
linear absorption, while queried E contracts with cell-constant nonlinear
polarization/dissipative currents. This uses the actual spatial field rather
than substituting HB cell-center work. Per-region/per-harmonic W and complete
query receipts are retained. Real dielectric DC with no current has zero
power; nonzero DC current explicitly requires the missing formulation.

/tmp/gsp06-full-field-power-v1.log: 1 passed in 17.79s,
session54445terminal0. Existing public H=2 tetrahedral nonlinear candidate,
four-face enclosing tetrahedron at 5x scale, surface/volume orders2 and fixed
field quadrature. Decomposed exterior flux agrees with independently formed
total E cross conj(H) surface integration (rtol1e-10,atol1e-18). Lossless linear
absorption and absent TPA are exactly zero. Non-enclosing/coincident control
geometry rejects. This is an implementation diagnostic, not a power gate.
Observed outward flux W [0,-4.332314229675202e-8,6.916260387744376e-16],
volume polarization work W [0,2.8439387535099396e-13,-1.1191407775676597e-15],
balance residual W [0,-4.3322857902876666e-8,-4.2751473879322214e-16].
No quadrature convergence or global closure claimed. Next separate scattering
and interference scales to normalize residuals and refine surface/volume
integration independently before interpreting the physical discrepancy.
No full regression/audit/phase commit; all jobs terminal.

## Independent power quadrature refinement and failed diagnostic

Power output now separates scattered and incident-interference W. The physical
residual scale is |scattering|+|interference|+|linear absorption|+|polarization
work|+|nonlinear dissipation|, per harmonic, with zero channels handled without
NaNs. Result.refine_harmonic_power separately refines each surface contribution
and each volume work contribution. Per-term successive-order differences and
tolerances are retained; cancellation cannot make the quadrature test pass.
An invocation-local cache reuses covered surface/volume terms. Refinement
success is empirical and never sets global_power_balance_verified.

/tmp/gsp06-power-refinement-v1.log/session84618 terminal1: external script
imported main checkout before reaching the calculation; no numerical result.
Corrected PYTHONPATH explicitly selects <scratch>/witwin-maxwell-gsp06.
/tmp/gsp06-power-refinement-v2.log and .pt/session76559 terminal0: reused
/tmp/pytest-of-xingyu/pytest-2672/test_public_harmonic_result_fr0/hb.pt via
solve-free restore. Same public nonlinear tetrahedron and 5x control boundary;
fixed field integration, surface orders(2,4,8), volume orders(2,3,4), relative
integration budget1e-4/absolute1e-22W. Both surface and volume refinement FAIL.
Every (harmonic,region,node-count) query occurs once; assembly and solve are
explicitly forbidden. No repeated nonlinear solve or previously covered term.
Final scattered W [0,2.0047870811756632e-10,1.0895717184472367e-15];
interference W [0,1.7948847544218217e-8,0];
polarization work W [0,3.7685797341640363e-13,-1.1175557556334533e-15];
residual W [0,1.8149703110309202e-8,-2.798403718621655e-17];
normalized residual [0,1.0,0.012678940167636862]. These are failed coarse
integration/physical diagnostics, not qualification. Next continue quadrature
and field-evaluation accuracy separation before assigning the discrepancy to
the electromagnetic discretization. Full physical power gate remains open.
All jobs terminal; no full suite/audit/phase commit.

## Power discrepancy isolated from nonlinear source integration

Result.far_field(directions,harmonic=n) now dispatches recovered harmonic
surface coefficients to the existing coherent global outgoing-amplitude
implementation. Positive retained harmonic is mandatory; DC rejects. No
volume source is added again outside its body, because the solved surface
unknowns already transmit that response.

All following probes reuse the exact pytest2672 public snapshot and forbid
assembly/solve. /tmp/gsp06-power-far-diagnostic-v3.log/.pt,
session61326terminal0: only new surface16/volume8 integrals, plus independent
far-field angular orders8/16. Surface-interference flux changes strongly from
order8, showing remaining control-surface integration error. At16/8, near
residuals [0,4.18746047864543e-10,-1.2618632653523421e-17]W, normalized
[0,1,0.005677145369347267]. Far angular8/16 give identical scattering
[2.0646213020615104e-10,1.1097151201128455e-15]W and fundamental extinction
-3.790386563261294e-10W. This negative extinction contradicts passive physical
qualification for the current coarse candidate.

/tmp/gsp06-far-source-order-v1.log/.pt, session22495terminal0: independent
source-panel quadrature6/12 at angular16, unchanged solved coefficients.
Scattering stabilizes at [2.06462238425056e-10,1.10972435136131e-15]W;
fundamental extinction remains -3.79048524639418e-10W. The negative value is
not resolved by far angular or source integration refinement.

/tmp/gsp06-linear-background-power-v1.log/.pt, session95975terminal0:
use the already stored fundamental background_surface column, scaled by the
same pump, with no nonlinear contribution and no new solve. Source12/angular16
scattering2.0632240708416823e-10W, extinction-3.793640178817054e-10W.
Thus the failed coarse passive-power diagnostic is already present in the
linear background discretization, not uniquely introduced by nonlinear source
coupling. Background assembly/discretization refinement is required before
this fixture can support physical qualification. Do not repeat these covered
angular/source-order checks or interpret the algebra fixture as a production
reference. All jobs terminal; no fullsuite/audit/phasecommit.

## Background assembly versus spatial resolution; planar predicate fix

/tmp/gsp06-background-assembly-order-v1.log/.pt/session50142terminal0:
same original tetrahedral linear eps2 background, new assembly6/9, fixed
far source12/angular16. Scattering W2.0608946557e-10/2.0609307288e-10;
extinction remains negative -3.8548991262e-10/-3.8555041031e-10. Higher assembly
quadrature does not resolve the coarse-space passive-power failure.

SurfaceMesh.refine(levels) now uniformly inserts shared edge midpoints on CUDA,
returns owned validated geometry and preserves the original piecewise-planar
shape. /tmp/gsp06-background-space-refinement-v1.log/session94685terminal1:
level1 (16 faces) completed at assembly9, scattering2.1783903186224714e-10W,
extinction1.0478697913212429e-9W, relative lossless imbalance0.65578333994.
Its exact Result is saved; level2 initially rejected as self-intersection.

/tmp/gsp06-refinement-intersection-debug.log/session70733terminal0 isolated
legal coplanar children6/7: orientation signs -0.0361,1.0631e-19 and
0.0361,-2.0357e-19 caused product<-tol² despite a shared endpoint. The crossing
predicate now requires each opposing orientation to exceed the same existing
tol independently. It does not bypass topology or loosen true-crossing checks.
/tmp/gsp06-planar-refinement-predicates-v1.log/session8763terminal0:
8passed3.93s (13deselected), valid two-level tetrahedral refinement/area/volume,
existing actual crossings/overlaps/shared-vertex counterexamples and radial
endpoint regression. No full suite at this intermediate fix.

Only the previously uncompleted level2 was resumed:
/tmp/gsp06-background-space-refinement-v2.log/.pt/session19276terminal0,
64faces, assembly9, elapsed28.6814s. Scattering2.1640476439414193e-10W,
extinction3.0657760654548763e-10W, normalized imbalance0.172420424018.
Spatial refinement changes the negative coarse extinction to positive and
reduces imbalance from65.6% to17.2%, but this remains a failed power gate.
All snapshots/logs retained; no repeat of level1, angular/source-order checks,
or nonlinear solve. Next conforming volume refinement is needed to carry
spatial convergence into the actual coupled HB system. No qualification,
fullsuite, independent audit or phase completion/commit claimed.

## Conforming tetrahedral spatial refinement

TetrahedralVolumeMesh.refine(levels) now returns owned, validated CUDA geometry
with eight children per parent. Global shared edge midpoints prevent hanging
nodes; four corner cells plus a central octahedron split along its shortest
opposite-midpoint diagonal fill each original tetrahedron. Signed orientation
is resolved natively before the normal positive-volume/topology/boundary checks.
Children remain in parent-major order. The public region ID is preserved;
invalid levels and mutated source meshes reject. Geometry gradients stay outside
scope. The boundary's four-child planar triangulation matches SurfaceMesh.refine.

/tmp/gsp06-volume-refinement-v1.log/session12049terminal0:
2passed4.11s,6deselected. A two-cell bipyramid is refined twice to128cells in
both float64 and float32. Parent volume sums, positive orientation, total first/
second moments, boundary face counts, independent refined-surface binding,
owned zero-level copy and mutation rejection pass. This exercises shared-face
conformity and both precisions, not just an isolated tetrahedron.

The new public HB spatial probe completed:
/tmp/gsp06-refined-hb-probe.py -> /tmp/gsp06-refined-hb-v1.log,
session75186terminal0. Same chi2/chi3 public material and H=2/pump(.5,1.) as
the algebra fixture, now8tetrahedra/16boundaryfaces and assembly6, unchanged
public field query tolerance. Preparation62.3988s, total64.8188s; residual
8.106238047192845e-17, frequencies(0,300MHz,600MHz), DC cell norm
0.00038928077468603677 and second-harmonic norm0.0003835101518277206.
Exact result /tmp/gsp06-refined-hb-v1.pt is saved for subsequent queries;
do not repeat the solve. This verifies actual refined coupled preparation/run,
not physical spatial convergence (cell-vector norms across different mesh
sizes are not a valid accuracy comparison). All jobs terminal. No full
regression/audit or phase commit; physical power and harmonic/volume acceptance
remain open. Next compare shared physical observations using saved states.

## Common physical observations reject spatial convergence

/tmp/gsp06-common-observation-probe.py reuses the original one-cell public
snapshot (pytest2672/hb.pt) and /tmp/gsp06-refined-hb-v1.pt (8cells); matrix
assembly and linear solves are explicitly forbidden. Both use the same
physical tetrahedral body/material/source and H=2, but the original fixture
used assembly3 while the refined one uses6; this is an overall numerical
refinement comparison, not an isolated volume-error estimate.

/tmp/gsp06-common-observations-v1.log/.pt, session56640terminal0:
same two interior and two exterior positions; AC complete-field query tolerance
rtol1e-5/atol1e-10/max48, DC separate, and the same six Cartesian far directions.
Relative changes normalized by the refined observation:
- Interior DC potential36.9626%, E15.5261%; H1 E12.3905%, H0.269191%;
  H2 E103.428%, H22.0723%.
- Exterior DC potential15.4218%, E12.3079%; H1 E0.0645366%, H0.00540525%;
  H2 E5.92140%, H12.1885%.
- Complex outgoing far amplitudes H1 E/H2.84367%; H2 E/H9.33329%.
Raw points/directions and both complex fields are saved in .pt. Observations
clearly fail spatial numerical convergence; exterior total fundamental field
alone is insufficient because the incident carrier dominates it. The current
8cell solve is implementation/algebra evidence, not an accepted physical
reference. No new solve or repeat of earlier power integrations occurred.
All jobs terminal. Remaining work includes controlled spatial/harmonic
refinement and unresolved material/DC formulations. No production qualification,
fullsuite/audit or phase completion/commit is claimed.

## Exact-prefix harmonic extension

prepare_dielectric_harmonic_balance now accepts reuse_prepared for a strictly
extended harmonic axis. It verifies the saved tensor/operator signature, exact
operator prefix and volume object identities, body geometry/material samples,
and unchanged DC permittivities/testing order. Existing bindings/static/dynamic
response objects are reused, not reconstructed. Only appended frequencies run
prepare_cell_field_response. New query/matrix settings apply only to those
frequencies; existing empirical accuracy receipts are not upgraded implicitly.
The new model owns its stacked field/source maps and retains reuse provenance.

/tmp/gsp06-harmonic-extension-v1.log/session24607terminal0:
1passed24.11s. Existing real one-cell H2 fixture extended to H3. Test explicitly
forbids DC re-preparation and requires dynamic preparation to receive only the
new 900MHz operator; prior response identity is preserved. Padded H2 warm start
converges (residual5.51085623244e-17), H3cellnorm2.2744965872472404e-5;
changed DC testing order rejects reuse. No physical/truncation claim.

/tmp/gsp06-refined-harmonic-extension-probe.py uses saved8cellH2 result,
computes only the new900MHz response, then warm-starts the same pump1 branch.
/tmp/gsp06-refined-h3-v1.log/session45573terminal0: prepare33.4605s,
total35.0417s,residual7.018241392787176e-17. Exact public Result snapshot
/tmp/gsp06-refined-h3-v1.pt retained for later observations. On the identical
cell grid, retained DC/H1/H2 relative changes are
[1.676666420127502e-8,2.8683536495863657e-9,5.166956149361857e-5];
new H3cellnorm6.322304527541093e-5. This controlled harmonic extension produces
much smaller retained-field changes than the prior spatial comparison, but
H3 itself is nonzero and no higher-cutoff/truncation certificate follows.

All jobs terminal. Next reuse the extended state for higher harmonic checks
and public truncation strategy/branch integration, while retaining unresolved
spatial and physical power failures. No repeated DC/old-frequency preparation,
fullsuite/audit, phase commit or production qualification.

## Fixed-response cutoff estimator and H4/H5 evidence

New harmonic_convergence.compare_harmonic_cells backs public
Result.compare_harmonic_cutoff(previous). Both states must use the same typed
Scene signature, pump, explicit branch, volume vertices/cells and exact old
DC/AC response matrices, including backgrounds/volume measures/query points.
The frequency axis must strictly extend. Native volume-weighted RMS peak-field
changes are checked per retained harmonic against atol+rtol*current amplitude.
Every added harmonic must independently meet an absolute V/m tail budget;
normalization by the strong fundamental cannot hide a weak generated line.
All physical/spatial/global-power flags remain false. The helper provides a
local empirical estimate, not a bound on all uncomputed higher harmonics.

All probes below reuse the saved 8cell prefix and prepare only the new frequency.
/tmp/gsp06-refined-h4-v1.log/session31255terminal0: new1200MHz,39.5634s,
residual1.0144626284459894e-16. Exact Result /tmp/gsp06-refined-h4-v1.pt.
H3→H4 retained fields pass rtol1e-4/atol1e-10V/m, but H4 RMS1.8181e-8V/m
fails the predeclared tail1e-8V/m. Comparison saved as
/tmp/gsp06-h3-h4-cell-comparison-v1.pt. Threshold was not relaxed.

/tmp/gsp06-refined-h5-v1.log/session33334terminal0: new1500MHz,40.6615s,
residual7.524307698306106e-17. Exact Result /tmp/gsp06-refined-h5-v1.pt.
H4→H5 retained-field and added-tail local tests both pass the same budgets;
H5 RMS1.9017e-9V/m. Public comparison rejects a strict1e-30 tail budget and a
reversed cutoff axis. Comparison /tmp/gsp06-h4-h5-cell-comparison-v1.pt.
No previous DC or frequency response was rebuilt. The local passing estimate
does not overwrite harmonic_truncation_converged or physical qualification;
independent observation, spatial and power checks remain open, as does the
public automatic cutoff strategy/branch workflow. All jobs terminal.
No fullsuite, independent audit, phase completion or phase commit claimed.

## Public budgeted automatic cutoff workflow

New public HarmonicCutoffConfig specifies maximum harmonic, retained/tail
budgets, required successive local passes (default2), and failure='raise' or
'record'. HarmonicBalanceConfig.cutoff enables this path. After the initial
named pump ramp, Simulation appends one frequency at a time, reuses exact
prior responses and warm-starts the same final pump/branch. Every failed
comparison resets the successive-pass count. HarmonicCutoffError exposes the
budget-exhaustion receipt by default; record mode returns explicit local failure.

Final Result frequencies/configuration reflect actual retained harmonics.
Solver statistics distinguish the initial pump-path harmonic count from later
cutoff checks, with per-step solver/branch receipts and all policy thresholds.
Nested cutoff configuration and diagnostics persist through exact snapshot
restore. Local acceptance stays separate from physical qualification and the
all-observations/global harmonic-convergence flag, which remain false.

/tmp/gsp06-public-cutoff-v1.log/session81042terminal0:
1passed23.68s. Actual public zero-nonlinearity tetrahedral candidate startsH1,
appendsH2/H3 and passes two successive local estimates. Dynamic preparation
is called only at600/900MHz; repeated static preparation is forbidden. The
fundamental field stays exactly equal to the initial background and all added
fields are exactly zero. Final axes, successive history and snapshot policy/
statistics roundtrip pass. At the current cutoff, budget exhaustion raises or
records failure without another solve; malformed policy values reject. This
is workflow/zero-nonlinearity evidence, not a new physical power certificate.
All jobs terminal. No fullsuite/audit/phasecommit. Remaining public branch
initialization/stability, DC current/TPA/singular limits and physical acceptance
remain open; GSP-01 failures are unaffected.

## Public explicit harmonic branch seed

Simulation.run and PreparedSimulation.run now accept keyword
initial_harmonic_electric for the configured initial retained count and volume
cell layout. It is a peak-phasor electric field at the first pump value, with
real DC, matching finite CUDA shape/dtype/device. Other solver configurations
reject the option. Prepared input validation precedes the single-use claim,
so an invalid seed does not consume a valid preparation. HarmonicRuntime
passes an owned detached numerical guess to solve_ramp; root differentiation
remains with respect to physical model parameters, not the initialization.
The original seed and explicit/default initialization choice are preserved in
Result statistics and snapshots, separate from later warm-start cutoff steps.

/tmp/gsp06-public-branch-initial-v1.log/session19333terminal0:
1passed28.42s. Public cutoff integration fixture uses an explicit0.3*background
seed, checks invalid complex DC rejection before preparation consumption,
recovers the exact zero-nonlinearity field and adds only new harmonics. Mutation
of the caller's seed does not affect the recorded initial field, and snapshot
restore preserves it exactly. First continuation receipt says explicit_state;
all previous cutoff/budget branches in this fixture still pass. This is seed/
workflow evidence, not independent physical multistability or stability proof.
No fullsuite/audit/phasecommit. All jobs terminal. Next unresolved conductive/
mean-current DC and TPA formulations require actual physics work; numerical
spatial/power failures and GSP-01 remain open.

### Preparation physics ownership guard

Response signatures now bind global/local carriers, sampled media, integration, kernel identity/receipt and body handle ownership. The actual CUDA preparation/recovery/query/extension regression rejects changed physics after decoder caching while preserving exact old-response reuse. `/tmp/gsp06-preparation-physics-guard-v2.log`: 1 passed in 24.63 s. Initial v1 failed because broad dataclass traversal included mutable diagnostic state; traversal is now limited to explicit physical records. No physical gate changed, full regression or phase commit performed.

### Isolated DC current weak block

`static_current.NodalCurrentBalance` implements a continuous linear potential
on the existing tetrahedral volume mesh. Cell E0 = -grad(phi); each nodal test
requires integral grad(Na).Jmean = 0, including natural zero normal total-current
boundary data. This is native CUDA volume testing for the coupled formulation;
no CPU numerical backend is introduced. One pinned node removes the constant
nullspace during solution; the returned potential has exact volume mean zero.
This gauge is not a physical electrode or a neutral-body charge constraint.
The pure-Neumann constant nullspace is documented in the
[singular Poisson reference](https://jsdokken.com/dolfinx-tutorial/chapter2/singular_poisson.html).

For positive peak AC phasors, C = 0.5 Re sum En En*, and q = <|a|^2 a>
comes from the alias-free constitutive transform. The exact mean current is
sigma E0 + tpa[(|E0|^2 + tr C)E0 + 2 C E0 + q] + supplied mean source.
The local differential conductivity is sigma I + tpa[(|E0|^2 + tr C)I +
2 E0 E0^T + 2 C]. Polarization contributes no mean time derivative.
A native dense reduced nodal Newton solve uses Armijo steps, explicit budgets,
positive resolved tangent checks and first-order implicit differentiation at
the corrected root. Final residual includes the eliminated nodal equation.
Unexcited dielectric/cubic singular tangents fail, including nonzero initial
guesses that otherwise could hide the singular limiting root.

`/tmp/gsp06-static-current-v1.log`: five CUDA tests passed in 5.60 s:
float32/64 skew/refined tetrahedral affine Ohmic solutions and gauge invariance;
analytic conductivity derivatives; independent scalar two-frequency cubic
roots and harmonic-amplitude derivatives with zero/positive Ohmic conduction;
independent 4096/512-point direct temporal means; element-face flux testing,
weak electric-current work identity and singular/budget/mutation failures.
The current residual is a weak discretization condition, not pointwise interface
current continuity or a spatial error certificate.

Still required: exterior electrostatic Dirichlet coupling, neutral-body potential
offsets/free-charge recovery, magnetic DC fields, mixed dielectric/conductive
DC blocks, public HB coupling and independent physical/spatial acceptance.
Existing dielectric-only DC maps and public TPA rejection remain in place until
these actual coupling requirements are handled. No production gate changed.

### Floating exterior DC potential and charge coupling

`static_floating.FloatingPotentialOperator` assembles one global CUDA constant-panel
single-layer Dirichlet operator for all bodies, with exact source-triangle Coulomb
potentials and outer face testing. The representation follows the standard
[Laplace single-layer operator](https://bempp.com/handbook/api/boundary_operators.html).
The block system is S lambda - B c = phi_boundary_mean and
A lambda = Q/epsilon_ext, where A integrates the layer separately on each body.
The lower rows are scaled by each body's square-root area to match voltage units.
The layer's harmonic continuation inside each body has zero net flux; its normal
jump therefore identifies enclosed exterior charge as epsilon_ext A lambda.
Offsets c fix the physical potential relative to zero at infinity. They are not
the numerical mean-zero gauge used by the interior current solve.

Native LU reuse accepts each body's actual nodal boundary potential and net charge
(neutral by default). Boundary and charge residuals are independently checked
against scaled roundoff budgets. Equivalent layer density is NOT local physical
free charge. Exterior queries reject interior/surface points; the physical interior
potential/E remains the responsibility of the DC volume solution. Owned layer,
offsets and material/charge tensors retain native gradients and mutation guards.

`boundary_fields(interior_displacement=...)` uses the exterior normal half jump
and stored principal-value gradient means. Supplied physical interior D includes
polarization. Surface free charge is n.(epsilon_ext E_ext-D_int); integrated bulk
free charge follows the interior D boundary flux. The exterior Gauss total is
reported separately from the enforced layer-integral net charge, exposing outer
quadrature error instead of treating the charge constraint as a field certificate.
The spatial bulk-charge distribution is not yet reconstructed.

`/tmp/gsp06-static-floating-v1.log`: 3 passed in 5.13 s. Actual TPA current solve
feeds exterior neutral fields; arbitrary interior gauge shifts cancel in offsets;
conductivity-to-exterior-field derivatives match centered differences. Separate
charged/neutral bodies show induced floating potential/density and exact separate
net-charge constraints. Sphere levels 1/2, order 6 give relative errors:
- Floating voltage: 4.493250378% -> 1.140306878% (FAIL final 0.5% threshold).
- Exterior potential: 0.005612497% -> 0.001128790%.
- Exterior E: 0.088777553% -> 0.019642123%.
`/tmp/gsp06-static-floating-charge-v1.log`: 1 passed in 4.67 s verifies one-sided
constant-potential means and zero-interior-D surface/Gauss accounting; coarse
Gauss relative error checked below 1% only, NOT the final 0.5% qualification gate.
No full regression, independent audit or phase commit performed.

Still required: prepared/current HB elimination or joint nodal equations, owned
coupled interior/exterior state and query/snapshot integration, spatial free-charge
recovery, magnetic DC fields and mixed dielectric/conductive DC coupling. The
public TPA path remains closed until these physical obligations are implemented.

### DC current closure inside harmonic balance

`harmonic_current.HarmonicCurrentClosure` owns one validated nodal current block
and physical DC conductivity per volume. At each actual HB AC iterate it solves
the isolated-body mean-current equation using the HB model's TPA coefficients.
`DenseHarmonicBalance(dc_current_closure=...)` explicitly replaces its DC equation
with E0 - E_current(E1,...,EH) = 0; positive-frequency P/J response equations remain
unchanged. DC background and P/J maps must be exactly empty to prevent double
counting, and actual volume weights/precision/device must match. Closure and
coefficient mutation guards remain active. Native nested implicit derivatives
contribute the eliminated current equation to the outer Newton Jacobian; the
converged solver receipt stores each DC current residual receipt. This is a
first-order differentiation path, not a higher-order derivative claim.

`/tmp/gsp06-harmonic-current-v1.log`: actual public-prepared dielectric AC maps
(with their unchanged linear material) feed an internal zero-Ohmic chi2/chi3/TPA
model. TPA is added once through nonlinear current; DC maps are replaced with
the new current closure. One actual solve passed in 18.29 s, with a diagnostic
print warning subsequently removed. No public TPA material admission is implied.
`/tmp/gsp06-harmonic-current-gradient-v1.log`: 1 passed in 18.66 s, including
centered differences from two converged TPA perturbations on the same AC matrices:
- Final HB residual: 5.426621239971941e-17 V/m (unweighted packed l2).
- DC electric norm: 6.879563702563006e-5 V/m.
- Second-harmonic electric norm: 1.3760721020141905e-4 V/m.
- Fundamental squared-field derivative with respect to TPA: -0.017131704141939985.
The original constitutive DC current satisfies nodal conservation; the recovered
current potential feeds a neutral exterior floating solve. An attempted simultaneous
dielectric DC map is rejected. Existing dielectric linear reduction/implicit
cubic derivative remains covered: `/tmp/gsp06-harmonic-current-dielectric-regression-v1.log`,
1 passed in 4.60 s. No full regression, phase commit or independent audit ran.

Remaining integration: validated physical current-material preparation (including
linear conductivity in AC exactly once), combined owned AC/nodal/floating state,
interior/exterior field queries and exact snapshots, magnetic DC fields, mixed
current/dielectric static bodies and singular excitation limits. Public conductive/
TPA admission remains closed. Existing spatial/power/floating-voltage failures
are retained, and physical/production qualification remains false.

### Actual current-material preparation

The shared internal entry is now `prepare_harmonic_balance`; runtime and tests
use this name directly, without a legacy alias. Each positive-frequency operator
must still match the actual harmonic material samples and common body geometry.
The same physical electric DC sample selects the closure: all finite dielectric
bodies use the existing displacement block; all conductive/TPA bodies use nodal
current closure and the global floating operator. Mixed dielectric/current bodies
and unresolved complex/plasma static channels reject. Each current body requires
its exact volume binding. Its conductivity comes from the sampled DC channel,
including resolved damped-pole conduction, rather than a regularization value.
The public TPA coefficient feeds the nonlinear law once; the linear AC material
already includes linear conductivity once.

Preparation owns closure, floating operator, DC channel identities and common
cell ordering. Current DC P/J maps are zero because their equation is replaced,
not because mean dissipative current is ignored. Extension checks DC channel
identity and reuses the old static object; operator/preparation signatures track
closure coefficients/gradients and floating matrices. Full current extension
combination testing remains pending. Existing dielectric extension behavior was
rechecked after sharing this entry point.

`/tmp/gsp06-current-material-preparation-v1.log`: 2 passed in 38.75 s. Actual
public Material(eps_r=2, sigma_e=0.001, chi2=0.002, chi3=0.001, TPA beta=5)
samples produce TPA scale 9.394585044692992e-5. Each AC epsilon agrees with
2 + i sigma/(n omega epsilon0); dielectric DC assembly is forbidden by the test.
The actual internal H2 solve converges to residual 5.726593310528998e-18 and
DC E norm 4.871488327609039e-6. Original constitutive total mean current
sigma E0 + J_TPA satisfies the nodal equation. Existing dielectric preparation,
mutation rejection, query recovery and H2-to-H3 map reuse also pass.

Public HarmonicRuntime still rejects current materials before preparation while
current-formulation state recovery, query/monitor/snapshot support and DC magnetic
fields are incomplete. No physical qualification, full regression, phase commit
or independent audit is claimed by this preparation change.

### Recovered current-formulation state and fields

`HarmonicCoupledState` now carries optional owned `dc_nodal_potentials` in volume
binding order and `dc_potential_offsets` in surface body order. Current recovery
verifies the actual HB residual and exact closure ownership, recovers the nodal
current solutions, maps boundary nodes through the validated volume-to-surface
correspondence, and solves the neutral global floating exterior. Its surface
representation is explicitly equivalent exterior potential-layer density (V/m),
not local physical interface free charge. Total DC current includes sigma E0;
positive-frequency currents keep only induced polarization/nonlinear current
because linear conductivity already belongs to their Green/MoM background.

`HarmonicFieldDecoder` uses the existing recovered AC coefficients and supplied
current state. Interior DC potential uses nodal barycentric interpolation plus
the body floating offset; E uses its constant cell gradient. Queries require an
unambiguous cell interior, rejecting unresolved internal cell-interface traces.
Exterior DC uses the saved layer and offsets directly. No current/interface solve
is performed during either query. All new state tensors participate in mutation
guards. DC magnetic fields remain absent; they are not inferred to be zero.

`/tmp/gsp06-current-state-fields-v1.log`: 2 passed in 38.32 s. The real conductive/
TPA material case recovers its state and queries interior/exterior DC and interior
AC while closure.solve, floating.solve and torch.linalg.solve are forbidden.
DC E matches the solved HB cell field; potential includes the physical offset;
AC matches the prepared complete field within the existing query tolerance.
Total DC current is checked against sigma E0 + J_TPA, and mutation of a recovered
nodal potential rejects. Existing dielectric recovery/query/extension checks also
pass with the extended state type. No full regression or phase commit ran.

Current snapshots explicitly reject rather than omit nodal/floating state from
the existing dielectric payload. Exact current serialization/restoration and
public dispatch are the next integration obligation; magnetic DC, mixed static
coupling, spatial charge/trace recovery and all open physical gates remain.

### Exact current-state snapshots

The current payload now has schema_version=2 and payload_kind=
`coupled_sie_vie_hb_current`; existing dielectric schema 1 remains a distinct
formulation. Both use the common snapshot/restore entry. The current payload
adds nodal potentials and body offsets, plus explicit tensor/primitive records
for current closure conductivity/tolerances/gradients/weights and the complete
floating operator, including layer/gradient matrices and LU factors/pivots.
No live Python operator object is stored in the payload.

`harmonic_current_snapshot.py` restores decode handles via saved arrays, without
calling DC constructors, factorization or solves. Validation checks exact sampled
conductivity and DC channels, affine nodal-gradient identities, nodal volume
weights, surface geometry/body integration maps, block matrix structure and LU
reconstruction. Shared restore checks actual Scene/material/carrier/AC data.
Current state additionally checks real DC, the constitutive total current with
Ohmic DC only, nodal gauge/current residuals, E=-grad(phi), floating boundary
potential and per-body neutrality. The payload's own current thresholds and all
qualification flags remain explicit; restoration is not physical qualification.

`/tmp/gsp06-current-snapshot-v1.log`: 1 passed in 24.26 s. Real conductive/TPA
material state is saved/loaded with nodal/floating constructors and solve,
lu_factor, lu_solve forbidden. Restored interior and exterior DC potential/E
match exactly; selected new state arrays survive re-save exactly, and corrupted
floating offsets reject. The same case retains AC query/current-state checks.
`/tmp/gsp06-current-snapshot-dielectric-v1.log`: existing public dielectric
Result frequency/monitor/query/save/load/continuation test passed in 19.49 s.
No full regression, phase commit or independent audit ran.

This removes the temporary current-snapshot rejection. Remaining public admission
work includes DC magnetic fields, complete monitor/power behavior and current
extension combinations; mixed dielectric/current static bodies and spatial/
physical gates remain open. Detached exact restoration does not restore an AD
history. Broader adversarial/cross-device snapshot audits remain at merge/final
acceptance rather than being claimed from these targeted checks.

### Magnetostatic source and a verified current-continuity failure

`magnetostatic.cell_magnetostatic_current_fields` computes H = integral grad(g)
cross J with g=1/(4 pi r), using the existing exact static tetrahedral kernel.
It returns actual curl/divergence diagnostics, retaining the Hessian volume
contact. The continuous-current representation and the steady-current constraint
are described in the [magnetostatic volume-current reference](https://farside.ph.utexas.edu/teaching/355/Surveyhtml/node93.html)
and [current conservation discussion](https://farside.ph.utexas.edu/teaching/355/Surveyhtml/node94.html).
For a current with distributional divergence, curl(H)-J equals the Hessian
contraction grad div integral(g J); this is explicitly exposed, not set to zero.
No scalar permeability interface conditions have been applied by this primitive.

`cell_current_continuity` computes each unique face's outward-current sum, with
zero exterior current on the body boundary. Cell-constant current has zero
classical cell-interior divergence, but nonzero face jumps are distributional
sources. Recovered current-state receipts now record face-flux maxima and
area-weighted normal-jump RMS, separately from nodal weak residuals.

`/tmp/gsp06-magnetostatic-source-v1.log`: 3 passed in 5.03 s:
- Direct independent order-12 volume cubature agrees with Biot-Savart H; source
  amplitude gradients agree analytically.
- A nonzero closed current is constructed in the nullspace of the actual face
  flux constraints on eight tetrahedra. Curl(H)=J inside, curl(H)=0 outside and
  div(H)=0 pass; the contact term is essential for the interior check.
- A nonuniform nodal TPA solution exposes the unresolved source defect:
  nodal weak residual 2.3688279439212935e-20 A (l2), normal-jump RMS
  0.025139638139978968 A/m^2, exterior Ampere defect 1.6380111517821353e-6 A/m^2.

The last item is retained FAILURE evidence for physical DC current/magnetic
qualification, not an accepted magnetic solution. A scalar magnetic interface
correction is a gradient and cannot cancel this curl defect. The next substantive
step is a conservative current discretization (or consistent reconstruction fed
back into the coupled equations); silently projecting only the reported magnetic
source would change the constitutive problem and is not sufficient. Mixed
current/potential closure and spatial convergence must be verified. All existing
public admission and physical-gate restrictions remain in place.

### Conservative face-potential DC current solve

`FaceCurrentBalance` shares the native constitutive Newton/implicit-gradient
implementation with the nodal block, but its DOFs are unique tetrahedral faces.
On each tetrahedron, the basis opposite vertex a is psi_a = 1 - 3 lambda_a;
its mean is one on face a and zero on the other faces. Therefore, for the actual
constant cell current J, integral grad(psi_a).J equals the outward flux through
face a exactly. Assembling the weak equation imposes zero summed normal current
on every shared face and zero normal current on the exterior boundary. The
constitutive J = sigma E0 + <J_TPA> is unchanged and never projected afterward.
Face potentials use a volume-mean-zero gauge and native first-order derivatives.

The relation between nonconforming face-potential and conservative mixed flux
spaces is established in the [original mixed-method reference](https://epubs.siam.org/doi/10.1137/0722029).
Here conservation for the nonlinear cellwise law follows directly from the
face-mean test identity above; it is independently checked by oriented flux
integration and the original time-domain TPA law. Face-average potential
continuity does NOT imply pointwise trace continuity, so potential/spatial
convergence remains a separate obligation.

`/tmp/gsp06-conservative-face-current-v1.log`: 7 passed in 5.96 s. New checks:
- Skew refined tetrahedral affine Ohmic solution, exact face-average potential
  and native conductivity derivative.
- Same nonuniform eight-cell TPA data as the retained nodal failure: direct
  1024-point temporal constitutive current agrees; every face flux balances;
  nonzero current circulation and inside/outside Ampere identities hold.
- Native TPA derivative agrees with two perturbed nonlinear solves.
Existing nodal float32/64 and nonlinear/current failure tests remain covered
following the shared DOF-indexing refactor.

On the repeated counterexample, face-jump RMS is 8.817891590289845e-18 A/m^2,
Ampere defect norm is 1.5600968066180264e-17 A/m^2, and the current norm is
0.004595368471970713 A/m^2. This is evidence that the new local discretization
resolves the prior continuity defect; it does not qualify the older nodal HB
candidate or establish a full physical/spatial gate. Old failure evidence is
retained. Next integrate face potentials into the actual HB closure, floating
Dirichlet face means, coupled query state and explicit snapshot representation.
No full regression, independent audit or phase commit ran.

### Conservative face-potential HB/state/snapshot integration

The actual `HarmonicCurrentClosure` now owns `FaceCurrentBalance` blocks.
`FloatingPotentialOperator.solve_face_means` accepts the physical boundary face
means directly; the existing nodal-input method delegates after averaging.
`surface_face_potentials` maps unique volume faces to the exact surface ordering
through the validated vertex binding, without assuming matching face/vertex order.
HB current closure and recovered state therefore use the conservative basis.

The coupled state explicitly stores `dc_face_potentials`, with representation
`coupled_sie_vie_hb_current_face_state_v1`. Interior potential is evaluated with
1-3*lambda face basis functions; E uses their gradients. Floating data receives
exact boundary face means. New topology arrays participate in mutation guards.
Current snapshots advance to version 3 and store face DOF maps and face lists,
validate their geometry, CR affine-gradient identities, face weights and actual
current/floating equations without assembly. Earlier experimental current
version 2 is explicitly rejected rather than interpreted as face data; its old
nodal failure evidence remains in the execution records. Dielectric schema 1
is a separate unchanged formulation.

`/tmp/gsp06-face-hb-state-snapshot-v1.log`: 2 passed in 32.29 s, covering actual
conductive/TPA material preparation, recovered AC/DC fields with solves forbidden,
version-3 exact DC snapshot round trip and corrupt-offset rejection; actual AC
matrix HB and end-to-end TPA gradient finite differences also pass. These are
one-cell coupled cases, so they establish wiring, not a multi-cell spatial gate.
`/tmp/gsp06-face-boundary-order-v1.log`: 1 passed in 3.71 s verifies exact face
mapping under independently reversed surface vertices and triangles, and equality
of direct face-mean and affine nodal boundary solves.

Next use the existing saved eight-cell AC response evidence to validate actual
nonuniform coupled HB current conservation without rebuilding covered AC matrices.
Then implement scalar permeability interfaces for DC magnetic fields. Potential
trace/spatial, power and all remaining physical gates are still open. No full
regression, independent audit or phase commit was performed for this integration.

### Actual eight-cell HB conservation and magnetic permeability interface

`/tmp/gsp06-eight-cell-face-hb-probe.py` loads the existing eight-cell H2 dielectric
snapshot and reuses both exact AC responses. Public material gains TPA beta=5
with unchanged chi2/chi3 and zero linear conductivity; every sampled AC medium
is checked against its saved operator. Only the new current/floating DC problem
is prepared. AC assembly is explicitly forbidden. The new Scene signature names
the actual TPA material, and the resulting current state is saved as schema 3.

`/tmp/gsp06-eight-cell-face-hb-v1.log` completed in 5.25917 s:
- HB residual 9.561361339097799e-17 V/m (packed l2).
- DC E norm 1.9065039794284344e-4 V/m; total DC J norm 5.760026205320079e-12 A/m^2.
- Normal-jump RMS 1.2325582860080634e-21 A/m^2.
- Inside/outside Ampere error 2.221912456456446e-21 A/m^2.
- H norm at the recorded points 2.3055353371236092e-14 A/m.
Native state: `/tmp/gsp06-eight-cell-face-hb-v1.pt`.
Raw points/current-continuity/magnetic evidence:
`/tmp/gsp06-eight-cell-face-hb-diagnostics-v1.pt`.
These are retained-harmonic algebra/current-conservation checks, not spatial,
power or full magnetic qualification. AC matrices were not rebuilt.

New `static_magnetic.MagnetostaticInterfaceResponse` reuses the floating
operator's Laplace principal-value gradient means. Its source is the global
Biot-Savart H from actual total DC currents. With c=2(mu_i-mu_e)/(mu_i+mu_e),
the layer equation is (I+c K')q=c n.H_source. Total H=H_source-grad(Sq);
one-sided means differ by n q, enforcing continuous tangential H and normal
mu H without adding a free sheet current. Permeabilities are positive finite
absolute samples. The solved layer has units A/m; query returns H and B=mu H.
Source-current continuity remains separately reported and cannot be repaired by
this curl-free scalar correction. Mutation guards cover geometry/current/material.

`/tmp/gsp06-static-magnetic-interface-v1.log`: 2 passed in 5.02 s. A conservative
nonuniform TPA volume current passes interface jump checks; matched permeability
reduces exactly to the direct source field. Uniform applied-H sphere refinement
(mu_i/mu_e=3, levels 1/2, fixed order 6) gives interior-H relative errors
0.015113313838851117 -> 0.004476402926613215 at the recorded interior point.
The finer value is below 0.5% at that point only. The permeability derivative
also agrees with the analytic sphere response within the test's approximation
budget. No broad production qualification follows from this local oracle.

Next integrate solved magnetic density/source mean/permeability into HB state,
no-solve H0/B0 queries and exact snapshots. Magnetic conductive/plasma DC limits,
mixed DC bodies, trace/spatial/harmonic/power and final independent references
remain open. No full regression, audit or phase commit ran this turn.

### Recovered DC magnetic state, query and exact snapshot

Current preparation now samples positive finite absolute DC permeabilities from
the actual public harmonic materials. Conductive/plasma/complex magnetic DC
limits reject through the static sample contract. Extension must preserve these
samples. During state recovery, the magnetic interface solves once using total
DC current (Ohmic plus nonlinear) and shared Laplace geometry. Owned magnetic
surface density and source face means join the coupled state's version guards.

`MagnetostaticInterfaceResponse.from_state` binds supplied arrays without
constructor assembly or solve. `HarmonicFieldDecoder.query_dc_magnetic` and the
public Result forwarding method return H0 and B0 from that recovered state.
The electric DC query continues to return potential/E separately. Existing
current states now report magnetic-field availability, with physical qualification
still false. No magnetic-conductivity limit is approximated by a finite mu.

Current snapshots advance to version 4, adding magnetic density/source means;
prepared records include exact sampled DC permeabilities. Restore validates those
samples, recomputes the source face field from stored volume current, and checks
the magnetic layer equation directly. It does not assemble/factor/solve an
interface matrix. Earlier experimental current schemas reject explicitly; the
separate dielectric schema remains unchanged.

`/tmp/gsp06-hb-magnetic-state-snapshot-v1.log`: 3 passed in 19.90 s, covering
actual current material/HB recovery, query with magnetic constructor/solve
forbidden, exact restored H0/B0 and magnetic arrays, corrupt magnetic density
rejection, plus magnetic-sphere/interface regressions. Source-field validation
at restore is explicit numerical checking, not a hidden repeated interface solve.
Result/HarmonicData forwarding methods were then added without changing numerics.
No full regression, phase commit or independent audit ran.

Remaining public work: current-material Simulation admission, DC H monitors and
complete DC work/power accounting; broader multi-cell magnetic-material coupled
oracles, magnetic DC singular limits and all outstanding physical gates remain.


## DC public integration and cutoff evidence

Public experimental single-body conductive/TPA Simulation admission, DC H point
monitors and complete current-state DC power accounting are implemented. Mixed
static formulations and unresolved magnetic DC limits still reject explicitly.
No physical qualification flags were relaxed.

- `/tmp/gsp06-dc-magnetic-monitors-v1.log`: 2 passed, 33.02 s.
- `/tmp/gsp06-current-public-v1.log`: 1 passed, 18.71 s, public Simulation,
  H/B monitors and exact Result restore with interface solve forbidden.
- `/tmp/gsp06-current-dc-power-v1.log` and `.pt`: saved-state evaluation with
  solves forbidden and a second cached pass forbidding all field queries.
  DC flux 1.1792644853114312e-34 W; linear work 1.3566449490714448e-18 W;
  nonlinear work -1.3566449490716878e-18 W; relative DC residual
  8.950748910149038e-14. AC normalized residuals 0.36699671077491386 and
  0.14314325977054546 FAIL. This does not close global power acceptance.
- `/tmp/gsp06-current-cutoff-reuse-v3.log`: completed H=2 to H=3 with only
  900 MHz assembled and identical static operator reuse. Added harmonic RMS
  4.837575782349051e-5 V/m exceeds 1e-8 V/m; retained-field check also FAILS.
  Recorded `harmonic_budget_exhausted`; all qualification flags remain false.
  `/tmp/gsp06-current-cutoff-reuse-v1.pt` contains the resulting schema-4
  historical candidate and receipt. Earlier v1/v2 logs failed before assembly
  because material allocator IDs differed, not because response reuse failed.

The material-ID failure motivated a shared material-field signature excluding
allocator identity and revision counters. Dielectric/current snapshot schemas
are now 2/5. Previous schemas are retained as historical evidence, not silently
converted or reinterpreted. Relevant public round-trip tests also reconstruct an
identical material with a different ID and reject changed nonlinear physics.
No full regression, phase commit or independent audit has run at this point.

Signature validation: `/tmp/gsp06-material-signature-v1.log` has dielectric
public round-trip PASS and a current test setup failure (attempted assignment to
frozen Structure). The test now reconstructs Scene using its public constructor.
Only that failed test reran: `/tmp/gsp06-material-signature-v2.log`, 1 passed in
18.88 s. Different material IDs restore exactly with solves forbidden; changed
chi2 rejects. `git diff --check` passes.


## Magnetic spatial qualification: independent exterior and point traces

Expanded the existing magnetic-sphere analytic test from a single internal
point to three internal and three external observations, measuring the external
scattered field (not total field dominated by the background) and its native
permeability derivative. `/tmp/gsp06-magnetic-sphere-spatial-v2.log`:
1 trend test passed in 4.54 s; this is NOT physical qualification.
Internal relative RMS errors: [0.014785067801788747,0.004430098533882857].
External scattered H errors: [0.14735977484672155,0.040860189055073284].
External scattered H gradient errors: [0.16257437474640146,0.046878671664995326].
Final field threshold remains 0.005 and the field gate FAILS. The v1 diagnostic
stopped at a newly introduced 3% coarse-gradient assertion; v2 records the full
refinement trend without treating that diagnostic assertion as an acceptance
threshold. The original final physical threshold was not changed.

`/tmp/gsp06-magnetic-point-trace-v1.log/.pt` evaluates actual fields on both
sides of the same eight-cell conservative source, mu ratio 3, at original face
centers plus/minus offsets 1e-4,1e-5,1e-6 m. Normal B defect uses
norm(Bout_normal-Bin_normal)/(norm(Bout_normal)+norm(Bin_normal)). At 1e-6 m
it is 0.19824555089663556 for integration order4 and 0.1996763224332886 for
order8; increasing quadrature does not fix this pointwise defect. Tangential H
defect decreases linearly with the observation offset. Exact face-mean algebra
therefore cannot be used as evidence for physical pointwise continuity.

`/tmp/gsp06-magnetic-trace-refine-v1.log/.pt`: one controlled refinement to
64 cells/64 boundary faces, keeping the SAME piecewise constant current through
parent-major repeat_interleave(8), testing order8 and original observation
locations. Analytic source H before/after refinement differs by only
3.338956796856728e-20 A/m; fine normal-current jump RMS6.7770731316048174e-18.
At offsets 1e-5/1e-6 m, normal B defects are .06894580196677909 and
.06811873666856375. Spatial refinement reduces the defect but still FAILS the
0.5% physical criterion. This isolates surface discretization as an unresolved
contribution without changing the constitutive current source or rebuilding AC.

Magnetic receipts now explicitly label constant_panel_face_mean testing and
pointwise_interface_continuity_verified=False. Further spatial convergence or
higher-order surface representation is required. No full regression, independent
audit, phase commit, or production qualification occurred this increment.


## Disjoint heterogeneous-body public harmonic candidate

Removed the blanket one-body restriction from HarmonicRuntime and harmonic
snapshot restore. Existing SurfaceSystem geometry validation continues to reject
touching/intersecting/nested bodies. Experimental implementation supports all-body
dielectric DC or all-body current DC; mixed formulations remain rejected. This
admission is not independent physical combination qualification.

Found and corrected a real single-body assumption in snapshot validation:
`materials[0].sources(all_cells)` incorrectly applied the first body's nonlinear
law everywhere. Preparation and restoration now share
`cell_nonlinear_coefficients`, expanding chi2/chi3/TPA by explicit volume region
ID and then cell order. No new schema is needed: existing schema2/5 already owns
the complete body/volume/material tables.

Added public two-body tests with heterogeneous eps/mu/chi2, reverse volume order
[2,1], actual nonzero second-harmonic generation, both directions of cross-body
response, regional centroid E/DC queries and exact AC/DC/magnetic restoration
with solve/factorization/assembly forbidden. These tests verify implementation
and ownership consistency; they do not constitute external physical references.
`/tmp/gsp06-multibody-public-v1.log`: two candidates solved, then both tests
stopped at a test-only invalid `HarmonicData.prepared` attribute; corrected to
`HarmonicData.decoder.prepared`. The overlapping-body rejection passed and was
not rerun. V1: 2 failed,1 passed in58.36s, handle93171 terminal1.

`/tmp/gsp06-multibody-public-v2.log`: 2 passed in62.28s, handle61194 terminal0.
Second-harmonic cell norms: dielectric .00019811772645856134 V/m,
current .00019799952030186596 V/m. Both preserve volume order[2,1] and all
qualification flags false. No full regression, phase commit or independent
audit ran. Feature inventory and main plan stage status updated; diffcheck clean.


## Mixed dielectric/current static interface foundation

Added `static_mixed.MixedStaticResponse`, sharing FloatingPotentialOperator panel
means and StaticVolumeFieldResponse particular/decoded fields. None interior
epsilon identifies a current-controlled body; finite positive epsilon identifies
a dielectric body. One layer density spans all surfaces, with dielectric rows
(I+c Kprime)rho=c n.Eparticular_out and current rows
S rho - body_offset = prescribed_face_potential - particular_potential.
Only current bodies add independent net-charge equations. Particular volume
polarization is accepted only from dielectric regions. Current interior queries
reject because their physical potentials belong to the current closure.

The solve uses native Torch CUDA linear algebra and componentwise true residual
checking; state/geometry mutation guards and differentiation are retained.
It is an internal foundation, not yet a public mixed HB implementation. The
public rejection remains until full volume mapping, residual, state and snapshot
integration is complete. Physical pointwise/spatial qualification remains false.

`/tmp/gsp06-mixed-static-v1.log`: all-dielectric reduction agrees with existing
StaticVolumeResponse, and all-current reduction with independent nonzero charges
agrees with FloatingPotentialOperator. One limit test passed. The second test
stopped at an arbitrary diagnostic voltage-change floor 1e-5 V: the actual
coupling change was 5.6025e-6 V. Replaced that nonphysical amplitude floor with
512 epsilon times voltage norm to test resolution above numerical roundoff.
This is not a change to any physical acceptance threshold.
`/tmp/gsp06-mixed-static-v2.log`: only the interrupted mixed test reran and passed
in4.11s, including neutral current-body charge, nonzero dielectric/current
coupling, dielectric-field native gradient versus perturbed solves, and explicit
current-interior rejection. V1 1failed1passed4.12s, handle16512 terminal1;
v2 handle25799 terminal0. No full regression, phase commit or independent audit.


## Reusable mixed DC operator and prepared response maps

Refactored the preceding internal foundation into MixedStaticOperator plus
MixedStaticResponse. The operator owns the fixed mixed matrix, LU/pivots,
material samples, panel quadrature and current-region mapping. Subsequent
responses reuse LU and check componentwise residuals. Operator tensor/metadata
mutation invalidates existing decoded responses. The prior temporary response
constructor signature was replaced rather than preserved as a compatibility path.

MixedStaticOperator.prepare_cell_response(bindings) prepares fixed linear maps
using a native Torch Jacobian with create_graph=True. Columns are dielectric
polarization xyz in explicit binding/cell order followed by face potentials in
current-region order. Rows return dielectric cell E, global layer density and
current-body offsets. These maps assume separately neutral current bodies;
nonzero prescribed charges remain supported by direct operator.solve. No
finite-difference maps, DC regularization, or hidden numerical CPU fallback.

`/tmp/gsp06-mixed-static-reuse-v1.log`: 2 passed4.50s,51611terminal0.
Existing dielectric/all-current limits still agree. Mixed source response,
neutrality and source gradient checks run with torch.linalg.solve/lu_factor
forbidden after preparation; mutated cached LU rejects later field queries.
`/tmp/gsp06-mixed-static-maps-v1.log`: 1 passed4.09s,74145terminal0.
Only changed mixed test reran: prepared E/density/offset maps agree with direct
response at rtol1e-12/atol1e-14, including explicit region order. The unchanged
limit tests were not repeated. No full regression/audit/phase commit ran.

This completes fixed mixed-interface preparation, not full mixed HB support.
Public mixed DC cases continue rejecting pending actual residual, recovered
state/field routing and exact persistence integration. All physical gates and
prior failed spatial/power/harmonic checks remain unchanged.


## Mixed DC closure in retained harmonic residuals

Added harmonic_mixed.MixedHarmonicClosure. It owns all binding/cell order,
selects current volumes with physical conductivities keyed by region ID, and
prepares dielectric response maps once. Its current solve uses only the selected
AC cells and their TPA coefficients. Dielectric cells reject a nonzero TPA law;
no fake conductivity or current regularization is used. Actual current face
potentials are reordered by current-region ID for the mixed interface. The
current nonlinear P0 of dielectric cells supplies the other input columns.
Both outputs scatter back to original user volume/cell order. Geometry, closure,
index and numerical map mutation guards are enforced.

DenseHarmonicBalance now accepts current-only or mixed closures, validates full
volume weights, passes the actual nonlinear P0 without detaching it, and records
the explicit closure formulation. Current-only closure gained a common volumes
property and accepts the unused polarization keyword; its physics is unchanged.

`/tmp/gsp06-mixed-hb-residual-v1.log`, 2 passed22.69s,16569terminal0:
- Mixed two-body reversed cell order[2,1], internal DC algebra with zero AC maps:
  final residual0, dielectric chi2 implicit derivative2.431289016934719e-6,
  agreeing with perturbed nonlinear solves. Prepared maps are enforced by
  forbidding MixedStaticOperator.solve during HB iterations. An independent
  direct mixed-interface query agrees at the recovered dielectric cell; current
  cell E agrees with the current closure. Illegal dielectric TPA and mutated
  response maps reject. This is NOT an actual SIE/VIE AC integration oracle.
- Existing actual AC/current test retains residual5.426621239971941e-17,
  DC field norm6.879563702563005e-5 and TPA gradient-.017131704141939985.

Public prepare_harmonic_balance still rejects mixed materials deliberately:
actual mixed preparation, recovery/field routing and explicit snapshot layout
must be integrated before admission. No full regression, major-stage commit,
independent audit, or physical qualification occurred. Existing failed gates
and final thresholds remain unchanged.


## Actual mixed-material SIE/VIE harmonic preparation

prepare_harmonic_balance now selects the mixed interface and closure from actual
sampled body materials. Mixed dc_permittivities retain None only for current
regions, positive samples for dielectric regions. Physical conductivity and TPA
coefficients retain volume-region mapping; existing AC compilation includes
linear conductivity once. Existing all-dielectric/all-current branches remain.
Prepared identity includes the mixed operator/factor arrays and the selected
current closure. Reuse retains material/static validation.

Public HarmonicRuntime now rejects mixed execution explicitly before constructing
AC operators because state/snapshot integration remains unfinished. Recovery
also rejects mixed closure explicitly instead of accidentally using all-current
layout. Internal preparation receipts mark mixed magnetic state/recovery support
false. These guards will be removed only as their actual paths are integrated.

`/tmp/gsp06-mixed-actual-preparation-v1.log`,1passed31.67s,6185terminal0:
Two separated tetrahedral bodies, body1 eps2/sigma.001/chi2.002/chi3.001/TPA5,
body2 eps3/mu2/chi2.004/chi3.001, reversed volumes[2,1], actual 300/600 MHz
SIE/VIE response maps and pump continuation(.5,1). Final residual
4.521034796141838e-17, DC cell-field norm.00011801604210641542 V/m,
second-harmonic norm.00019806618698525895 V/m. Both directions of AC interbody
response are nonzero. Mixed-interface solve forbidden during HB iterations;
dielectric DC state agrees with a separate direct mixed-interface field query,
current body satisfies its actual Ohmic+TPA weak current residual. Public mixed
rejection precedes AC construction; unsupported recovery rejects explicitly;
mutated mixed LU changes preparation identity and invalidates model residual.
The test stores only detached candidate fields/receipt as mixed-algebra.pt in its
pytest tmpdir, NOT a public exact Result snapshot. Physical qualification remains
false. No full regression, independent audit or phase commit ran.


## Mixed recovered fields and explicit current-region layout

HarmonicCoupledState now has dc_current_regions (empty for existing dielectric
and all-current layouts). For mixed states it records selected current binding
order; face potentials and offsets use that same order. Density/offset recovery
applies the prepared mixed maps to actual dielectric P0 and current face means.
Ohmic DC current is added only at selected current cells; dielectric cells carry
no invented conduction. AC source conventions are unchanged. The full scalar-mu
magnetic interface covers all bodies using the actual total cell currents.

Mixed DC queries use physical CR face potentials inside current bodies and the
polarization particular field plus recovered shared layer in dielectric/exterior
regions. Magnetic queries use recovered interface density/source means via the
existing no-solve decoder. Region layout participates in state version guards.
All-current recovery/offset order remains unchanged. Mixed snapshot serialization
explicitly rejects before attempting the all-current encoder; current schema5
and dielectric schema2 remain unchanged. Public mixed admission remains blocked
until exact persistence and remaining observable paths are integrated.

`/tmp/gsp06-mixed-state-fields-v1.log`: 2 passed47.77s,99979terminal0.
Actual mixed 300/600 MHz candidate retains residual4.521034796141838e-17.
Recovered mixed density agrees with direct static solve; DC current is zero in
dielectric cells and includes Ohmic work only in current cells. Both interior
regions reproduce DC and H2 cell E. Exterior DC potential/E agree with a separate
direct response; H0/B0 queries are finite in both interiors and outside. These
queries execute with current/mixed/interface solves forbidden. Unsupported mixed
snapshot rejects explicitly. Shared all-current preparation/state/query/exact
snapshot regression also passed (residual5.726593310528998e-18).
After these checks, only the public rejection message and matching test regex
were updated to name the remaining snapshot/observable integration, since state
recovery is now implemented. No numerical changes followed verification.
No full regression, phase commit or independent audit; diffcheck clean.


## Mixed DC regional power accounting

harmonic_power_balance no longer zips all bound volumes against an all-current
conductivity tuple. It builds the conductivity table from actual selected
current blocks keyed by region ID, then visits all bound volumes in their
original order. Dielectric regions have zero Ohmic coefficient, while their
fields and nonlinear work entries remain present. This fixes both wrong region
assignment and accidentally omitted dielectric entries for mixed/reversed grids.

`/tmp/gsp06-mixed-power-v1.log`: 1passed33.27s,52375terminal0.
Actual mixed two-body H2 candidate with volumes[2,1]. Checks both DC region
entries, exact zero dielectric linear/nonlinear dissipative work, current-body
Ohmic volume work against sigma*|E0|^2*V, and identical repeated power residuals
with all field query methods forbidden during the cached repeat. Existing mixed
state/field and early public/snapshot guards are exercised by this same fixture.

Coarse surface/volume order2 power residuals [DC,H1,H2] are
[-6.8139401432420404e-31,4.836876162624289e-7,-1.540290525259412e-15] W.
These are diagnostic raw residuals; no quadrature convergence or global power
qualification is established. The solver residual remains4.521034796141838e-17.
No extra integration refinement or full regression was repeated. Physical flags
remain false; exact mixed snapshots and independent-static cutoff comparisons
are still outstanding before public admission.


## Explicit mixed numerical records and decode-only restore

Added harmonic_mixed_snapshot.encode/restore for internal prepared static data.
The encoder reuses existing current/floating primitive records for selected
current volumes, then explicitly stores mixed operator samples, masks, quadrature,
matrices/factors, full region/count/index mapping and response maps. No live
operator or closure object enters the record. Restore uses __new__ and shared
current/floating validation; no matrix assembly, constructor, factorization or
linear solve is invoked.

Checks include expected current regions from actual material channels, exact
selected/full volume order, geometric panel data, scalar samples, quadrature,
mixed row structure, LU reconstruction/pivots, response dimensions/order,
separate neutral-charge columns and boundary-potential response equations.
This is not yet a complete mixed Result schema: integration into harmonic_snapshot,
recovered-state checks, DC source/field validation and full payload round-trip
remain required. In particular, dimensional/neutrality/boundary checks alone do
not establish the physical correctness of every dielectric-polarization response
column. Do not claim those missing checks are covered by this record test.

`/tmp/gsp06-mixed-record-restore-v1.log`:1passed4.44s,41651terminal0.
`/tmp/gsp06-mixed-record-restore-v2.log`:1passed4.41s,68591terminal0 after adding
neutral and boundary response checks. Constructors and torch solve/lu_factor/
lu_solve are forbidden throughout restore. Restored arrays match exactly and a
subsequent allowed source solve reproduces density/offsets bitwise. Corrupt
current cell indices, LU and a nonneutral surface-response column reject.
No actual AC regression was repeated for this isolated record implementation.
No full regression, phase commit, independent audit or production qualification.


## Mixed harmonic payload schema 6 and physical state consistency

harmonic_snapshot now dispatches mixed payload kind coupled_sie_vie_hb_mixed,
schema6, with explicit dc_current_regions. Existing dielectric2/current5 stay
unchanged. Current face/offset shapes follow selected current bindings. Restored
Ohmic current is scattered only to those cells. Current metadata and magnetic
validation are shared through extracted validate_magnetic; its source means are
checked by source-field evaluation, not a new interface solve.

The mixed validator checks gauges and weak current conservation, actual P0 and
face-mean response inputs, layer/offset maps, independently decoded dielectric E,
current CR E, mixed dielectric/Dirichlet equations, separate neutral charges and
magnetic source/interface equations. This validates the stored state's physical
consistency with its fixed discretization; it does not establish spatial or
reference accuracy or independently certify all response columns for every input.
The latter remains part of broader adversarial/physical acceptance.

`/tmp/gsp06-mixed-payload-v1.log`:2passed48.24s,85631terminal0. Actual heterogeneous
mixed H2 candidate restores exterior DC potential/E, H0/B0 and H2 E/H bitwise;
both body interior DC queries are bitwise identical. Matrix solve/factorization
and MixedStaticOperator construction are forbidden during restore/query.
Corrupted current-region ordering and floating offset reject. Existing schema5
actual conductive/TPA state/query/snapshot test also passed after the shared
magnetic validator extraction. Mixed algebra residual4.521034796141838e-17;
physical flags false and coarse power residuals unchanged.

The one reusable schema6 candidate was moved (not duplicated) from pytest-2688
temporary output to `/tmp/gsp06-mixed-schema6-v1.pt` to preserve it for cutoff
extension without reassembling covered AC responses. Public gate wording now
names remaining cutoff comparison integration; its condition is unchanged.
No full regression, phase commit or independent audit. Diffcheck clean.


## Mixed cutoff comparison and public admission

compare_harmonic_cells now distinguishes dielectric/current/mixed formulations
and uses mixed numerical records when static preparations are independent.
Current-only comparison remains unchanged. The identity fast path still reuses
the exact static object for a normal harmonic extension.

`/tmp/gsp06-mixed-cutoff-v1.log/.pt`,14811terminal0: restored the existing schema6
H2 candidate and an independent copy, then extended only 900 MHz. Constructors
for both floating and mixed static operators were forbidden; retained AC and
static responses reused. Both same-static and independent-static comparisons
work. Retained RMS changes [4.5891e-10,2.0770e-8,1.2259e-8] V/m pass local
thresholds, but added H3 RMS3.510767367033171e-5 V/m FAILS1e-8 V/m. Receipt records
harmonic_budget_exhausted; all global/harmonic/physical qualification flags false.
The script is /tmp/gsp06-mixed-cutoff-probe.py; no covered300/600 MHz response
was reassembled.

Removed the remaining blanket public mixed guard. Public scalar finite dielectric,
conductive and TPA bodies may now combine experimentally; existing singular DC,
trainable-compile and geometry guards remain. Retired the internal test's old
public-rejection assertion and added a mixed case to the public multibody test.
`/tmp/gsp06-mixed-public-v1.log`:1passed34.83s,13608terminal0. Actual public
Simulation.run with reversed volumes[2,1], heterogeneous materials and DC E/H
monitors in each body; exact regional AC/DC/H0/B0 Result.save/load with solves
and assembly forbidden, including restored monitor values. Nonlinear H2 norm
.00019806618698525895 V/m. No production qualification is inferred.

FEATURE_LIST consolidates the implemented mixed workflow and removes superseded
pending/reject claims; detailed progression/failures remain in this plan. No
full regression, phase commit or independent audit in this increment. Final
physical thresholds and all existing failed gates remain unchanged.


## Candidate stage merge readiness review started

The implemented scalar dielectric/current/mixed public workflows now form a
candidate integration checkpoint, not a completed GSP-06 production phase.
Started one consolidated regression:
`python -m pytest -q tests/mom tests/api/public/test_public_api.py tests/api/public/test_simulation_smoke.py tests/api/public/test_dependency_contract.py --junitxml=/tmp/gsp06-stage-integration-v1.xml`
Log `/tmp/gsp06-stage-integration-v1.log`, process handle85055. This covers MoM
and directly affected public API integration, not the final whole-repository or
cross-repository acceptance. Keep numerical source unchanged until this baseline
finishes; then fix findings together and rerun affected checks only.

Independent read-only audits started for snapshot/recovery integrity and for
material/HB/public combination contracts. A preliminary snapshot blocker is
unchecked saved per-frequency cell_volumes/points, which feed cutoff RMS despite
having no validation against the restored tetrahedral mesh. This is not repaired
yet. Final audit reports and terminal regression result are pending.

Remaining requirement inventory (no qualification inferred from implementation):
- Surface/volume ownership and zero-source reduction have targeted tests;
  deliberate linear-contrast double-count falsification still needs explicit
  evidence, beyond a false linear_contrast_added receipt.
- Scalar source convolution, DC/current/mixed algebra, native internal derivatives,
  public fields/monitors/persistence and local harmonic extension are implemented.
- General public trainable material compilation and singular DC channels remain
  unsupported. Tensor+nonlinearity remains outside the accepted public contract.
- Actual spatial/power/cutoff failures remain as recorded; physical chi scaling,
  Kerr phase/TPA attenuation, Manley–Rowe, independent FDTD amplitude/phase/spectrum
  and branch stability acceptance are incomplete.
- Bench development worktree search has no harmonic_balance/harmonic_state/
  coupled_sie_vie integration references; dedicated nonlinear exact storage/DR
  and cross-repository acceptance remain work, not validated coverage.
- GSP05 remains a separate unmerged implementation branch; GSP07 and later stages
  must still be developed. The overall goal is unchanged and incomplete.


Independent snapshot audit returned three findings (static code analysis; focused
corruption reproductions/fixes still pending after baseline):
1. P1: saved per-frequency cell_volumes/points are unchecked against bound mesh;
   restored cutoff RMS can therefore use modified weights.
2. P1: stored HB convergence threshold is only checked finite/nonnegative, not
   recomputed from configuration and actual background. The current-closure
   threshold has a similar trust issue. Inflating it can mask corrupted responses.
3. P2: mixed electric response maps are validated only along the stored state's
   packed input; orthogonal perturbations can change later reuse while preserving
   current-state checks. Full-column no-solve identities are required rather than
   treating a matching single state as complete map validation.
Audit found no clear region-reordering bug or hidden solve in current recovery.
These findings block candidate merge. No fixes applied during the running
baseline. Regression log already contains failures; terminal details pending.

### Candidate stage regression and audit fixes, 2026-09-21

The frozen candidate integration run completed: 533 passed, 22 failed in 672.83 seconds; `/tmp/gsp06-stage-integration-v1.log` and `.xml` retain the single baseline. This covers MoM and selected public API tests, not final whole-repository/cross-repository acceptance. Most failures were stale pre-multibody test fixtures (single compact Gram versus block metric, schema 2 versus 3, absent kernel, old allocation injection target, string versus integer monitor region). One exact detached field comparison differed by 2.96e-23; field pair accumulation used unordered CUDA atomics. The implementation now reduces bounded observation pairs without atomic scatter addition; exact comparison remains unchanged.

Independent contract audit also found ignored linear solver configuration and lost continuation context on inner DC errors. Resident direct preparation now rejects unsupported strategies before assembly and checks configured dual residual tolerances. Pure TPA zero-pump continuation remains an unresolved singular-DC limit and rejects early. DC errors retain their original receipt plus branch/failed-step/completed-step context. Structural multi-body coupling tests are not independent physical qualification; physical combination, double-count falsification, power/stability/gradient and cross-backend gates remain open.

The three snapshot integrity defects were reproduced using the existing schema-6 artifact: altered volumes, points, and background+inflated threshold were all accepted before repair (`/tmp/gsp06-snapshot-integrity-before.log`). Restore now validates geometry, recomputes HB and DC current thresholds, and checks every mixed-map column through no-solve operator/decoder identities. Four targeted tests passed in 49.52 seconds (`/tmp/gsp06-audit-fixes-v1.log`), including actual mixed and current snapshots, unused-direction corruption, and continuation/configuration guards. A separate affected-test run is pending in `/tmp/gsp06-stage-regression-fixes-v1.log` (session 97788). No physical failure was relabeled as passed and no phase commit has yet been made.

The affected regression group completed with 24 passes and one further stale monitor-log key failure; that key was corrected to region 0 and passed in the public follow-up. `/tmp/gsp06-audit-public-v1.log` records 5 passes in 112.15 seconds (monitor semantics, all three multi-body DC variants, single-body public snapshot). `/tmp/gsp06-audit-policy-falsification-v1.log` records 2 passes in 43.66 seconds: actual heterogeneous-background double-count falsification and the extended mixed corruption test. The 22 failures from the frozen baseline have now been covered by passing targeted follow-ups; this is not represented as a fresh all-green whole-suite run.

Independent read-only re-audit confirmed the original three snapshot findings are closed and found no new concrete field-reduction gradient/determinism blocker. Its remaining DC-policy finding was then fixed: public snapshot restoration requires the actual frozen current-closure default tolerances, rejecting simultaneous changes to closure tolerances and stored threshold. The corruption test exercises this case. Cross-device bitwise equality is not promised by the observation reduction.

The heterogeneous two-body falsification deliberately reintroduces the already represented linear contrast as a volume current. It changes both the original PMCHWT residual and exterior field beyond roundoff while the corrupted algebraic solve still converges. Together with zero-source reduction, ownership and shared-exterior coupling checks this records a structural combination gate, not independent physical combination qualification. No nonlinear multi-body production qualification is granted.

This checkpoint is an independent experimental implementation commit. GSP-06 remains OPEN: public trainable material compilation, singular DC limits, stable/multistable branch qualification, physical scaling/phase/attenuation/global power and spatial convergence, independent FDTD comparison, nonlinear Bench storage/DR and final cross-repository acceptance are incomplete. GSP-01 remains OPEN with its original failures. GSP-05 integration and GSP-07 onward remain necessary; final scope and thresholds are unchanged.


## Historical compressed mixed magnetic recovery — 2026-10-02

Original pending8334ca5 provider magneticcontrol terminal1PASS33996.628s/rc0/
388stable/zero errors/skips/failures. ActualH2leaf64/scalarσchi2chi3TPA8cellsource
andisotropictensor-linear region, H0/H1/H2; savedHBresidual6.701439893862893e-17
passes originalgates. Exactfourmagneticstatearrays/currentpartition/tamper/version/
storage guards. OriginalartifactSHAe7dbd94e02e8110136afac3cf0a86ea0954501261cd482ec76958943e9fca593
directlyreferenced, no copy/requery/resolve. Historicalrunner lacksactualCore/
Torch/deviceexecution receipt, notinvented. Oldsession59607 noXML remainsUNKNOWN.
Restoredqueries outsideforbiddencontext andmagquery1e-12/1e-28 notbitexact;
inverse/providerguards limited. Exacttestedcase transferredcurrentregression,
19historical/currentprod differences recorded; no currentfull acceptanceclaim.
Assessment/reference/independentphaseaudit recordpartial scope. GeneraltensorAC/
fullrestoredguards/physical/P0magaccuracy/tail/stability/globalpower/far/resources/
scaling/public/periodic/populations/finalfourrepo/GSP-01OPEN remain unchanged.
