# Periodic hierarchical tensor composition

This independent tree starts at Maxwell 06dd5fd. The original tree at
<scratch>/witwin-maxwell-volume remains frozen while genuine tensor sheet equivalence
session 8826 runs on GPU 1. Do not restart or overlap that numerical control.

The first implementation extracts common Ewald selected-block correction and
streamed source-face action into a mixin. The existing scalar Ewald provider
retains its ScalarSystemBlocks ancestry for FMM; the new
GeneralizedEwaldSystemBlocks uses tensor/scalar primary interiors without being
accepted by scalar FMM admission. Six CUDA tests cover 1D/2D/3D scalar-limit and
manufactured tensor blocks, original sampled actions, current gradients,
permuted selections, no regular-block materialization in the streamed action,
and kernel mutation rejection. Syntax/diff checks passed; CUDA execution pending.
Also validate existing Ewald sampled actions and affected FMM paths after this
refactor. Manufactured kernel routing is not tensor spectral/physical accuracy.

Continue through public H/H2 configuration, periodic operator identity, decoder
binding, original residual admission and exact native/Bench persistence. Reuse
native hierarchical construction/conversion and original periodic field decoding;
do not synthesize dense all-column operators or call them fast. Preserve local
interior ownership, spectral cache budgets and current gradients. Final runtime
validation must include mixed bodies, nonzero Bloch scalar limits, tensor failure
propagation and exact restore without assembly. Perform independent stage audit
and commit the completed integration separately. FMM, nonlinear, 3D physical
qualification and all prior failed gates remain obligations of the full goal.

Runtime progress: new PeriodicHierarchicalSystemOperator and H2 counterpart use
native H construction/H2 conversion, original Ewald residual sampling, shared
regional tensor kernels and existing periodic decoder bindings. MoMConfig admits
periodic H/H2 without enabling tensor FMM or periodic harmonic balance. Native
coefficient/decode snapshots retain schema 16 for tensor periodic combinations
and introduce schema 17 for scalar periodic H/H2; this is not yet exact compressed
factor replay. Eight new public tests compare dense controls and nonzero Bloch
fields, require nonempty far blocks/couplings, and forbid assembly/solves during
Result.load. Fourteen new test nodes collected in 5.39s without GPU exposure;
CUDA execution remains pending. At that checkpoint exact compressed factor codecs and Bench schema 17 handling
were pending; their current implementation status is recorded below. Mixed-body
public controls and full targeted CUDA acceptance still need completion.

Original genuine fusion control PID 4181007/session 8826 is confirmed live. Its
numerical tree remains unchanged at <scratch>/witwin-maxwell-volume, commit 06dd5fd.


The native bare factor codec now binds exact scalar/tensor free-space and Ewald
provider kinds, sampled materials, regional spectral policies and periodic
kernel receipts. Regional v2 physical wrappers restore H/H2 factors, source
carrier and runtime bindings without rebuilding numerical factors; original
scalar v1 behavior remains. Public tests additionally forbid H/H2 constructors,
block sampling and direct solves during factor restoration and compare forward
and adjoint actions exactly. Runtime and codec imports passed on host; CUDA
validation remains pending.

Bench work is isolated at <scratch>/witwin-bench-periodic-hierarchy, branch
gsp-periodic-tensor-hierarchy at 4690caa. Schema 17 scalar periodic H/H2 and schema
16 tensor periodic H/H2 archive contracts bind compression and lattice settings.
/tmp/gsp-periodic-hierarchy-bench-host-v1 records 13 pass/8 fail: all eight failures
were the scalar host fixture lacking sampled material fields, before archive
validation. The fixture was completed; only the eight affected nodes were rerun
in /tmp/gsp-periodic-hierarchy-bench-host-fix-v1, all pass (5.30s). The prior 13
passes remain applicable. These are host storage contracts, not CUDA or physics.
Independent read-only pre-audit of runtime/provider/codecs/Bench is in progress.


Independent pre-audit identified two concrete issues, both corrected before CUDA:
(1) an existing test still rejected the newly admitted periodic tensor H/H2
configurations; it now explicitly admits iterative/local-Riesz H/H2 while
retaining tensor FMM/direct-H2 rejection. (2) schema 17 lacked a backend guard
and could accept a relabeled scalar dense schema 7 payload; restoration now
requires hmatrix/h2 and a test reuses the dense reference payload to reject this
relabeling. Public coverage adds mixed scalar/tensor bodies and actual low-budget
dyadic failure propagation for periodic H/H2. Existing free tensor H/H2 roots
are reused to cover newly admitted generalized factor codec restoration with
construction, block sampling and solves forbidden. No new free-space root was
added solely for this codec coverage. These CUDA checks are still pending.

Independent pre-audit is terminal: confirmed schema 17 and outdated admission
fixture defects are corrected; the added free tensor factor replay was inspected.
No unresolved confirmed P1/P2 remains in static review. This is not CUDA evidence.
The numerical/test source is now frozen for queued session 33880, runner
/tmp/gsp-periodic-hierarchy-targeted-v1.py (log same prefix, receipt directory
/tmp/gsp-periodic-hierarchy-targeted-v1). It observes PID 4181007's start identity,
waits for a terminal genuine-fusion receipt, verifies all queued source hashes,
then runs new provider/public tests and affected tensor public, Ewald sampled
residual, periodic FMM, scalar/physical factor codec and periodic tensor paths
on GPU 1. Nonempty XML with zero failures/errors/skips and unchanged sources is
required for a pass. Do not change numerical/test sources or duplicate this run.
Bench actual archive replay should reuse the saved public roots in its pytest
basetemp after terminal validation; do not launch additional root solves for I/O.

The actual Bench replay harness is prepared, not executed:
/tmp/gsp_periodic_hierarchy_bench_replay_v1.py. After the targeted suite passes,
run on GPU 1 with PYTHONPATH containing this Maxwell tree, its tests/mom directory
and <scratch>/witwin-bench-periodic-hierarchy. It requires the unchanged successful
suite receipt and exactly twelve distinct saved public combinations (1D/2D,
H/H2, scalar/tensor/mixed), reconstructs the matching test scenes and forbids
operator construction, assembly and direct solves. It writes fresh schema 16/17
archives to /tmp/gsp-periodic-hierarchy-bench-replay-v1, validates native metadata
and byte-exact J/M, and binds source/input/manifest digests. No new roots are
needed. Syntax passed; execution and evidence audit remain pending.


Targeted v1 is terminal failed: 28 passed, 12 failed, 214.28 seconds;
source hashes unchanged. All failures are the new public cases' requirement
for nonempty compressed far blocks/couplings, after solver and dense coefficient
checks passed. Preserve the v1 receipt/XML. Diagnose compression admission before
re-running only affected public cases; unchanged passing modules need no repeat.


The v1 failure was an invalid test assumption, not failed compression admission:
two 12-by-12 cross-body blocks require rank 12 at tolerance 1e-10, so their
factors cost more than dense storage. Public tiny cases now verify the recorded
storage rejection and continue all original solve/field/replay checks. Four
additional mixed-material cases (1D/2D times H/H2) use two levels of conforming
tetrahedron refinement, unchanged geometry and precision thresholds, and must
produce real far factors/couplings. Independent static audit found no confirmed
blocking defect or weakened admission. Session 43105 runs only this public
module (16 nodes), with /tmp/gsp-periodic-hierarchy-targeted-v2.py and immutable
receipt/XML under the corresponding v2 directory. Source/tests are frozen while
it runs. The 28 unchanged v1 passing nodes remain evidence, without rerunning.


V2 is terminal: 12 tiny public cases passed, four refined cases failed inner
GMRES stagnation after 42/44 iterations; 433.53 seconds, unchanged source hashes.
No refined compressed-path acceptance is claimed. V3 runs only the 1D mixed H
refined case, increasing maxiter/restart/stagnation-window budgets while retaining
rtol=1e-12, compression tolerance=1e-10 and all field/original residual thresholds.
Per-case stats/operator diagnostics now persist before the convergence assertion.
Runner /tmp/gsp-periodic-hierarchy-targeted-v3.py, session 6834; frozen while live.
Retain all v1/v2 failures. Do not repeat passing tiny cases only for archiving.
The prepared Bench harness now covers 16 saved roots, but needs a combined
explicit node-level admission across the preserved tiny and corrected refined
runs before execution; its current whole-v2 gate correctly rejects v2 failures.


Refined diagnostic progression (all prior failures retained): v3 increased
iteration budgets but failed at 480 steps (relative residual 1.2094e-5). V4
allowed a full 384-dimensional first Krylov cycle and rank 96: solver passed at
334 iterations, relative residual 7.9674e-13, original/dual gates passed and dense
coefficients agreed, but no far factors passed compression. Single-block GPU
SVD diagnostics under /tmp/gsp-periodic-hierarchy-rank-diagnostic-v1 prove that
this 192-by-192 block needs rank 114 at the unchanged 1e-10 tolerance; UV storage
only decreases below rank 96. Complete and adaptive cross approximation both
correctly refused storage qualification. This is not a compression defect.

A distinct compression control scales each tetrahedron by 0.5 about its first
vertex, then refines twice; the lattice, materials and all precision thresholds
remain the same. This preserves the anchor, not the centroid. Diagnostic v2
finds optimal rank 75, complete-cross rank 89 and adaptive/recompressed rank 80,
all reducing storage at the declared tolerance. Public controls allow rank 192
for H construction and rank 128 for H2 conversion. Their interior query point
scales consistently. These are selected software controls, not general physical
or complexity qualification and do not supersede original-geometry failures.

V5 is terminal passed: the new 1D mixed H public control passes all solve,
compression, field, exact-factor and Result restore checks, 141.55 seconds,
unchanged sources. V6 (session 24389) runs only the remaining 2D H and 1D/2D H2
controls; source/tests remain frozen until terminal. The Bench harness will
explicitly combine the twelve passed v2 tiny nodes with passed v5/v6 refined
nodes, requiring unchanged production sources and permitting only the recorded
public test fixture source delta. It binds XML, producer receipts, node names
and each result at replay entry; producer-side artifact SHA was not recorded,
so no earlier artifact hash-chain claim is made. No duplicate tiny roots.


Stage targeted validation is terminal: v6 passed all three remaining nodes in
519.31 seconds, with unchanged sources. Combined current coverage is 44 distinct
Maxwell nodes (28 unchanged v1 passes, 12 v2 tiny passes, 1 v5 and 3 v6 compressed
controls); v1/v2/v3/v4 failed records remain unchanged. Bench host coverage is
21 distinct nodes across the initial 13 passes and eight corrected fixture
passes. This is combined evidence, not a claim that every historical run passed.

The first actual Bench replay is terminal passed: all 16 saved results preserve
J/M bytes and native schema/configuration, without new assembly or root solves,
in 3.952383 seconds. Source, XML and receipt bindings are unchanged.
Receipt: /tmp/gsp-periodic-hierarchy-bench-replay-v1/receipt.json; SHA256 6a16b0aa7e752efa6574bae617b15665a04babd6288a15969ed2162905824d9a.
The harness records each node's original producer (including failed v2 overall),
source differences restricted to the public fixture, result SHA from replay
entry, and each exact manifest digest. Refined cases use genuine compressed
H/H2 factors, but tensor materials are isotropic tensor representations; these
are software/algebra controls, not genuine anisotropic physical qualification.
Independent terminal stage audit and commit follow; full MoM/Bench regression
runs once at the combined periodic-composition integration point.


Independent terminal audit completed with no confirmed P1/P2. It verified the
44-node combined XML accounting, unchanged production sources, 21 distinct
Bench host passes, all 16 result/manifest digests and byte-exact J/M arrays,
96 saved array digests, three producer XML/receipt bindings and 653 unchanged
replay sources. The audit explicitly retains scalar-limit public tensor scope,
manufactured general tensor block coverage, real low-budget rejection coverage,
all prior failures, and the replay-entry artifact hash boundary. This is a
stage audit, not the final physical/performance/four-repository audit.
