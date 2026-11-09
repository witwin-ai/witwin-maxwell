# Bounded native directional cone and order planning

`DirectionalConeOrderPlan` in primary `mom/fmm_order.py` admits at most32 fixed
CUDA float64 declared center/support pairs. It is an internal primitive, not
public Scene configuration or a full-tree cone/global system planner.

For each configured strictly increasing cube-face subdivision m, select the
largest absolute center coordinate (first Cartesian axis on ties). Its signed
face coordinate is ±1. Other coordinates divide by the dominant absolute value
and map to indices floor((ratio+1)*m/2+0.5), clamped to0..m. Normalize the resulting
cube point once. Keys are (m, signed face, u index, v index); face is2*axis plus
one for negative direction. Other axes retain increasing Cartesian order. No
catalog of all possible carriers or duplicated full-tree projection owners is
allocated. This bounded planner does not construct sparse node/cone contexts.

Evaluate all configured grids over candidate orders0..12 through the existing
continuous directional complex-ball Cauchy bound. Availability is mandatory:
unavailable zero placeholders never satisfy budgets. Select the smallest
passing order, breaking ties by coarsest configured grid. Every row passes
before publishing a plan or entering a selected-row coefficient factory.
Kernel budgets bound ideal Green absolute error and each Cartesian physical
gradient component on declared support balls; no directed-rounding certificate
or verified actual RWG support is inferred.

Automatic normalization uses min(D/(2p+2), max(rt,rs)) when at least one radius
is positive, otherwise D/(2p+2). Both distance and support cases are verified in
a separate changed-path control. The distance-only initial high-electrical-size
physical failures are preserved: successful ideal admission did not ensure
native accuracy. The support cap recovers only the selected physical cases;
this rule is not a general roundoff or native implementation bound. Extreme
nonrepresentable selected scales reject. It is not a replacement for finite
kernel/phase/quadrature/global error admission.

Persistent inventory is144 bytes/pair: center24, radii16, carrier24, cone key32,
order8, scale8, two absolute and two log bounds32. Caller inputs are excluded
from ownership but guarded. Before contour allocation, preflight this inventory
and a conservative component workspace: existing contour allowance plus4096
bytes and1024*pair_count*order_count. Caller arrays, allocator/cache/library
state, returned translation inventories and whole-process peaks are excluded.
Selected-row translation uses the unchanged primitive with its own additional
retained/workspace preflight; no aggregate translation/tree/system capacity
claim is made. Borrowed inputs, owned tensor identity/version and semantic
configuration/budget values are guarded before receipts or factories. Receipts
return independent list/dictionary containers.

Acceptance evidence: `/tmp/gsp-directional-cone-order-evidence-v1/summary.json`.
Strict100-digit independent cube/positive contour selection agrees within
1.197930643570544e-13 relative (1e-11 gate, zero absolute tolerance for the
corresponding native tensor checks). Selected lossless/lossy actual four-field
controls pass1e-8; complete native current/higher/dot controls pass1e-11 with no
saved primitive tensors. Initial scalar-type and distance-only physical failures
remain archived. Only failed/new/changed-path checks were repeated. No prior
covered kernel/tree/RWG/public/resource/Bench test was rerun.

GSP01 stays OPEN. Complete actual interior near/FAR, physical support/leaf
quadrature/native/phase-path/global budgets, automatic sparse tree integration,
public/periodic integration, whole resources/crossover/N scaling/populations,
current four-repository full final acceptance and at least two FINAL independent
audits remain required. All production/physics/resource qualification flags are
false. This phase audit is separate from FINAL Goal acceptance.
