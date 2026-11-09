# Automatic actual-support directional RWG block

`AutomaticDirectionalRWGBlock` is an internal selected direct PMCHWT block. It
consumes actual test/source bases, unique ordered edge subsets, fixed centers,
caller-owned prepared geometries and a finite-frequency scalar medium. It does
not construct full-tree projection owners per cone or implement sparse global
cone contexts. Public Scene/Simulation/Result integration remains separate.

Before integrating source/receiver maps or allocating scalar coefficients, it
computes enclosing radii from every selected plus/minus triangle vertex about
actual input centers, inflated by512 float64 eps. Independently guarded caller
geometry must match each basis. Native existing U/Q integral majorants reduce
to three selected log statistics. The original material-balanced block bound
satisfies B_ideal² = A² G² + B² H², where G bounds ideal Green absolute error and
H bounds each physical Cartesian gradient component. A² comprises the material
D channel coefficient times sum(Ut Us + Qt Qs/|k|²)²; B² =6 eta² sum(Ut Us)².
Eta is the supplied common balancing impedance, including common exterior
impedance for magnetic interior blocks.

Each effective kernel budget is min(original budget, ideal budget/(sqrt(2)*
channel weight)). The split is conservative, not joint optimal; it may reject
an order range for which a different joint allocation might pass. Native
unrepresentable effective budgets reject; positive bounds are never rounded
into false exact zero. Reuse the bounded native canonical cube-face/order
planner with these budgets. Check the combined original ideal RWG bound again
before factories. Quadrature must resolve the selected polynomial factor;
this does not establish carrier quadrature convergence or native error.

The selected source/receiver maps use the actual chosen order, carrier and
medium. Existing admitted directional electromagnetic M2L derives supports
again and binds both projections, material factors and Hermitian/current
channels. Automatic distance/order normalization is capped by maximum actual
support radius. The original numerical formulas are preserved; projection and
scalar-FFT structural capacity estimators are shared by old constructors and
new combined preflight, avoiding duplicate numerical factory implementations.

Exact ownership comprises144 planner bytes +40 generated planner-input bytes,
both independently owned direct projection maps, and the scalar kernel.
Caller bases/centers/edge inputs/prepared geometries remain guarded but excluded
from ownership. Sum conservative stage workspace allowances before integration
or coefficients, including planner/admission/projection/scalar/EM components.
Caller arrays, library/allocator/cache and whole-process peak are excluded;
this is not aggregate tree/system or crossover/scaling qualification. Exact
storage is checked at construction. Actions and receipts guard borrowed support,
owned numerical state, scalar material and semantic budgets. Physical action chunks at most four RHS through all map/coupling stages and
releases each intermediate before the next chunk. The shared fixed linear
autograd protocol saves no wrapper tensors and checks the outer support/budget
owner on backward, including derivatives of the Hermitian map. Caller output
size is excluded from workspace; chunk intermediates are included.

Evidence: `/tmp/gsp-directional-actual-support-evidence-v3/summary.json`.
Selected unsorted5x3 RWG blocks on actual closed30-edge spheres cover magnetic
lossless/lossy media, common exterior impedance and15 noncontiguous RHS. Actual
all-column original weak/H controls retain1e-8 gates and native current/higher/
dot checks1e-11. A100-digit independent positive triangle integral formula and
explicit material-channel sum verifies both effective budgets and combined
ideal bound with1e-11 relative/zero absolute checks; omitted sqrt(2) would exceed
the declared budget. Only the new independent node ran after the original phase
checks; previously covered physical controls were not repeated.

GSP01 and GSP09 remain OPEN. Actual complete interior near/FAR system physics,
sparse cone/tree/global/phase admission, finite quadrature/native errors,
public/periodic integration, whole resources/crossover/N scaling/populations,
current four-repository FINAL acceptance and at least two FINAL audits remain
required. This phase grants no physics, complexity or production qualification.

The second independent phase review closed both production P2 and found a test
portability issue: the changed-path matrix reuse required historical /tmp files.
The final test-only interface accepts optional GSP_DIRECTIONAL_RWGMATRIX_MANIFEST.
Explicit reuse strictly requires matching artifact/diagnostic SHA, never skips;
unconfigured ordinary regression obtains the same actual fixture/q16 original
reference freshly. Production code and numerical checks/gates are unchanged.
Only NEW lifecycle control ran in v4/session56849 CLOSED1PASS2.42s/397stable,
no prior numerical PASS or original integration repeated. Missing/partial/tampered
explicit evidence rejects; actual accepted matrices resolve in place unchanged.
Final PHASE review: /tmp/gsp-directional-actual-support-independent-phase-audit-v3.json.
This remains unrelated to FINAL Goal audits and grants no production qualification.
