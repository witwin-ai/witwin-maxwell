# GSP-08: GPU hierarchical operator development

Status: experimental scalar H/H2 reference and public backends implemented;
complete production qualification remains open. Based on GSP-07 public checkpoint
f0c1f6a, in <scratch>/witwin-maxwell-gsp08 (branch gsp08-hierarchical).

The initial geometry layer uses CUDA Morton sorting with full support-box
bounds. The octree owns its tensors and rejects subsequent mutation. Equal
quantized centers remain an explicitly reported oversized leaf; they never
force an invalid geometric split. All numerical geometry operations are native
Torch CUDA. Construction keeps O(N) temporary arrays per bounded Morton level,
with no point-pair list or dense electromagnetic matrix.

Block traversal performs strong support-box admissibility and descends touching
boxes to exact leaves. It streams bounded batches with depth-first scheduling;
a consumer must not concatenate all point pairs. A support-box admissibility
flag only permits a candidate approximation and does not certify low rank or
kernel error. Complete RWG support means both incident triangles, not only an
edge midpoint or triangle centroid. Source/test trees can be distinct.

Next: bounded-rank far-block construction, exact near-block storage and the
common forward/adjoint/Riesz protocol. H-matrix precedes nested H2. Tensor and SIE-VIE compositions, periodic
traits, frequency identities, resource preflight, error budgets and persistence
remain required. Dense probes and small algebraic examples cannot substitute
for the final impedance-balanced operator tolerance, physical-error budgets,
large-scale storage slopes and independent audit in the root plan.

Initial CUDA checks: `/tmp/gsp08-cluster-v1.log`, 4 passed, 1 failed in
35.14s. The failing coarse single-sphere fixture expected a far block where
strict full-support admissibility produced none. The replacement fixture uses
two genuinely separated RWG spheres, preserving the admissibility threshold:
`/tmp/gsp08-cluster-rwg-v2.log`, 1 passed in 4.26s. Tree ownership/enclosure,
exact rectangular and square pair coverage, duplicate centers and actual RWG
support separation are covered. This is not an operator or complexity gate.


## Exact selected scalar weak blocks

`nearfield.scalar_regional_block` now computes only requested RWG edge rows
and columns. It enumerates their support face products in bounded batches,
classifies all five pair types and reuses the reference joint Duffy/adaptive
near/regular far rules. Source and test selections retain arbitrary order.
Adaptive near estimates still use the full local 3x3 face integrals so edge
selection does not silently alter the dense quadrature convergence decision.
Regional contributions are explicit: self blocks include exterior/interior
once, cross-body blocks permit only the common exterior.

`ScalarSystemBlocks` routes global edge selections to body pairs and returns
all selected J followed by all selected M. It provides complete two-triangle
RWG support boxes for the geometry tree, validates bound geometry/media/config,
and owns no global matrix. Receipts report cumulative face classes, maximum
classification batch and requested matrix bytes. These counters are not yet
complete allocator/workspace accounting. The initial helper path recomputed whole-basis local geometry; the prepared
geometry integration below removes that repeated work. Full setup complexity
acceptance remains open. Tensor, periodic and volume blocks remain explicit
future compositions; this provider admits only sampled free-space scalar media.

CUDA dense comparisons: `/tmp/gsp08-nearfield-v1.log`, 2 passed in 9.63s;
selected self blocks cover all five quadrature classes and arbitrary ordering,
and rectangular cross blocks match the exterior-only dense oracle. Global
mixed-body requests, support-tree construction and binding/geometry invalidation
passed `/tmp/gsp08-system-blocks-v1.log`, 1 passed in 6.33s. Tolerance remained
2e-12 relative / 1e-14 absolute for these algebraic block comparisons. Existing
cluster checks were not repeated. No compressed solve or production claim.


## Streamed reference far-block compression

`lowrank.streamed_cross_approximation` performs complete-pivot cross
approximation with bounded row samples and selected columns, without a full
block allocation. It scans every balanced residual entry after each rank
update. This reference setup costs quadratic work per rank; a storage saving
is not a claim of fast or asymptotically qualified setup. Explicit positive
row/column weights define a block Frobenius check, not the final global
impedance-balanced Riesz operator criterion. The returned physical-unit factors
share forward/adjoint actions and native current-vector gradients. Kernel and
material gradients through the rank selection are rejected. Rank exhaustion,
nonfinite values and factors without a storage saving fail closed with receipts.

`ScalarEdgeBlockSampler` maps arbitrary J/M coefficient-row samples onto unique
edge requests to ScalarSystemBlocks and exposes selected columns without a
global matrix. The sampler binds its provider and owns its edge selections.

Validation: `/tmp/gsp08-lowrank-v1.log`, 2 passed in 4.87s (a test-only scalar
conversion warning was subsequently removed). Controls include zero initial
rows, complete balanced residual comparison, complex dot identity <1e-11,
current-vector gradients, factor mutation, rank-limit failure and rank-zero
blocks. `/tmp/gsp08-lowrank-rwg-v1.log` failed storage reduction in 14.41s:
the actual 20x20 RWG block reached rtol1e-8 with rank16, but its factors took
10240 bytes versus 6400 bytes dense. This failure is retained; verified factors
now explicitly reject when storage does not decrease.

The same geometry's complete 60x60 cross-body block, at unchanged rtol1e-8,
passed `/tmp/gsp08-lowrank-rwg-v2.log`: 2 tests in 21.57s (including explicit
nonbeneficial-factor rejection). Rank18, independently computed balanced
relative error9.335378831445693e-9, factor34560 bytes versus dense57600 bytes.
This is one block-size crossover control, not a multi-scale storage slope,
global operator certificate, near-field physical budget or H/H2 qualification.
Next: hierarchical assembly with bounded exact leaves and rank-failure
subdivision, a shared action/adjoint object, sparse exact RWG metric actions,
solver coupling and complete storage/workspace accounting.


## Linear-storage metric and shared hierarchical actions

`LocalRWGRiesz` retains exact per-face 3x3 Gram blocks and incidence, with
Jacobi-PCG inverse actions in the all-J/all-M layout. Every iteration checks
the actual residual. Failed inverses raise; native current-vector backward
uses the same fixed inverse adjoint. Tensor/layout/configuration and geometry
mutations reject. This is the existing impedance-balanced L2 proxy, not a
certified trace-space norm. No global Gram/factor is formed.
`/tmp/gsp08-local-riesz-v1.log`: 2 passed in 5.65s, comparing dense action,
inverse and gradient; zero RHS and forced inverse failure covered. Fourfold
edge refinement produces exactly fourfold retained metric bytes.

`HierarchicalOperator` now owns one tuple of exact leaves and verified low-rank
far factors for forward, adjoint and the matrix-free protocol. Rejected far
compression descends existing nodes until a bounded exact leaf is available;
there is no global dense retry. Coincident Morton supports exceeding the exact
leaf limit currently reject explicitly. The Morton tree now uses a common
physical coordinate scale (cubic cells), avoiding artificial per-axis stretching
of slender/disconnected geometries. Native current-vector derivatives share
the same block action. Content receipts hash actual factors, indices, metric
and tree tensors; executable Result persistence is not implemented yet.

The current budget covers retained tensor data, not full setup/allocator peak.
Preparation still uses complete residual scans; whole-basis geometry is now
prepared once per provider.
The initial action checkpoint admitted only preconditioner='none'; the solver
integration below adds row/column equilibration. Public Scene backend routing, full
resource preflight, directional/tensor/volume/periodic traits and H2 remain open.
`/tmp/gsp08-hmatrix-actions-v1.log`: 1 passed in 39.08s. Actual two-body
operator has two exact self blocks and two compressed cross blocks. Full
independent dense reconstruction has weighted block Frobenius relative error
4.1952377627662785e-14 (self terms dominate this global norm); exact leaf equality,
complex adjoint defect <1e-11, metric-residual gradient and mutation rejection
pass. Total retained tree/metric/block/index tensors:196728 bytes; dense matrix
alone:230400 bytes. This small control is not a crossover curve, a global Riesz
operator error certificate or a physical/complexity qualification.


## Krylov integration and original-operator residuals

Hierarchical row norms and column-then-row equilibration now come from the
same exact/low-rank blocks. Far norms use small factor Gram products, never a
reconstructed far matrix. The common GMRES and block-GMRES protocols operate
on two independent physical incident polarizations with native lifted RHSs.

`/tmp/gsp08-hmatrix-solve-v1.log` failed in 45.35s: inner Euclidean rtol1e-10
reported convergence, but one RHS had compressed Riesz relative residual
1.0573332824968536e-8, above the unchanged1e-8 acceptance line. This failure is
retained. Tightening inner rtol to1e-12, without relaxing the dual residual or
current thresholds, passed `/tmp/gsp08-hmatrix-solve-v2.log`:2tests52.13s.
GMRES compressed/original dual residuals6.901832246492794e-13 /
8.342264418553024e-13, relative balanced current difference7.260363863530704e-12.
Block-GMRES values2.6074689010906786e-11 /2.6076182561894053e-11,
current difference2.3467389215977025e-11. This does not automatically couple
arbitrary outer physical budgets to Krylov stopping; that integration remains
required before public backend admission.

ScalarSystemBlocks.sampled_action now streams bounded source-edge blocks for
selected original test rows. HierarchicalOperator.residual_report reports the
full compressed L2-dual residual separately from original sampled-row
impedance-balanced Euclidean residuals, including exact tested edge IDs.
Default samples are deterministic evenly spaced edge IDs. It explicitly does
not certify global original residuals or unobserved rows. The report test
compares actual original quadrature to independently assembled dense rows,
including a deliberately perturbed RHS and nontrivial row ordering.

Original-row report float64 oracle check at rtol2e-12:
`/tmp/gsp08-original-residual-precision-v1.log`,1passed7.35s. Only this new
report control was rerun; the expensive hierarchical solve was not repeated.


## Prepared geometry and baseline audit

PreparedRWGGeometry caches triangles, signed RWG slopes, centers and support
radii once per body. Selected integration, adaptive orders and same/cross-body
pair classification share these caches; the uncached dense reference remains
available. Basis identity, edge/topology/support tensors and cache versions are
checked before reuse. `/tmp/gsp08-prepared-geometry-v1.log`:3passed12.15s,
including all five dense integration classes, cross-body ordering, explicit
failure if whole-basis local_geometry is called after preparation, and cache
mutation rejection. Cached arrays are included in the H retained byte budget.

storage_breakdown reports exact blocks, far factors, block indices, tree,
metric and prepared geometry separately, then deduplicates shared input
system/basis/mesh CUDA storages. The existing retained budget covers owned
retained tensors only. Shared inputs, host control objects, solver vectors and
allocator workspace are explicitly distinguished; full allocation preflight is
still open. The checkpoint test now samples the PyTorch CUDA setup peak before
the dense oracle exists, in a shared pytest process with existing caches excluded.

Independent read-only audit gsp08_hmatrix_audit found one P2: standalone
LocalRWGRiesz omitted source basis version binding (the enclosing H provider
already rejected those mutations). Changing edge lengths could reuse an old
Gram; the auditor's small CUDA reproduction had relative action error0.75.
LocalRWGRiesz now binds each basis and source geometry/index/sign/length tensors
for apply/inverse/backward. `/tmp/gsp08-riesz-source-binding-v1.log`:1passed3.84s;
subsequent static re-audit confirmed the P2 closed and no remaining P1/P2 in
this baseline scope. No full regression was duplicated by the auditor.


## Internal reference checkpoint integration

`/tmp/gsp08-hmatrix-checkpoint-v1.log`:33passed,1failed98.59s. The failure was
ordinary GMRES with row-only scaling, raw rtol1e-12: one RHS reported stagnation
and did not claim convergence. The earlier isolated run passed this setup;
its convergence stability therefore remains unqualified. The failure is kept.
No physical/current/dual threshold was relaxed.

`/tmp/gsp08-hmatrix-equilibration-v1.log`:1passed52.58s. The diagnostic adds
ordinary GMRES with column-then-row equilibration alongside block GMRES, at the
same raw1e-12, dual1e-8 and balanced-current1e-7 acceptance thresholds. Measured
condition numbers7456.3518(row-only) and3567.4397(two-sided). Original dual
residuals: ordinary two-sided1.7673153247053639e-12;
block two-sided2.6122869737233663e-11. Current differences4.147872395588249e-11
and2.351858527362951e-11. Row-only also converged in this diagnostic, which does
not erase its earlier stagnation. Its test branch only permits the explicitly
observed finite, above-threshold stagnation receipt; other failure reasons fail.
The independent auditor reviewed this distinction, and recommended the strict
stagnation-only guard that was adopted. Public error-budget admission remains open.

With prepared geometry, owned CUDA tensors total201848 bytes:
exact115200, far factors69120, block indices3840, tree888, metric7680,
prepared geometry5120. Shared system/basis/mesh storage7776 bytes makes total
referenced tensor storage209624 bytes. The dense main matrix alone is230400
bytes. Measured PyTorch CUDA setup peak increments57851392 bytes in the stage
run and66797056 bytes in the diagnostic. These shared-process measurements
exclude already-resident caches and do not include every driver/allocator or
host allocation; the retained budget is not a total-memory preflight guarantee.

This commit is an experimental internal H reference checkpoint, not GSP-08
closure. Public dispatch and exact Result restore, full resource preflight,
general error-budget coupling, packed near storage, accelerated setup, true
H2, tensor/volume/periodic traits, large-scale slopes/crossover and physical
budgets remain required. All earlier physical failures remain unchanged.


## Public-admission prerequisite: explicit residual budgets

HMatrixConfig now includes solve_dual_tolerance and original_sample_tolerance
(default1e-8 each), plus strict versioned payload/restore. HierarchicalOperator.solve
uses the shared Krylov solver (default two-sided GMRES, inner rtol1e-12), preserves
inner failures, and only admits an inner-converged result when its complete
compressed L2-dual residual and declared original-row sample both pass. The
log separates inner_converged, checked budgets and final converged. Original
row checks still explicitly cannot certify the full original operator; physical
qualification remains false. Public Scene routing and Result schema integration
are not yet enabled by this internal helper.
`/tmp/gsp08-hmatrix-admission-v1.log`:5passed9.90s. Actual solve/config roundtrip,
false inner success, original-action fault injection and inner stagnation are
covered. No previous expensive hierarchical or full regression was repeated.


## Public scalar H runtime and exact physical Result restore

MoMConfig now explicitly selects operator_backend='hmatrix', versioned
HMatrixConfig, local_rwg_v1 Riesz storage and iterative stopping without a
direct fallback. HierarchicalSystemOperator shares prepared scalar weak blocks,
local free-space decoders and the lifted per-body contrast RHS. Runtime dispatch
uses the residual-budget admission helper. Unsupported periodic/HB combinations,
dense-only stopping/storage and H2 reject before assembly; tensor materials
remain rejected by the scalar compiler.

Sampled operator identity binds geometry, sampled media, kernel and all numerical
settings including compression and residual budgets. RHS identity also binds the
incident carrier. Schema 8 persists exact scattered currents and reconstructs
only the decoder, validating both identities and the false physical-qualification
flag. This is executable physical Result decoding, not persisted compressed
factors or resumed Krylov state. Original sampled rows remain only samples.

`/tmp/gsp08-public-hmatrix-v1.log`:2passed7.25s. Public solve and checked residual
admission, bitwise restored currents/fields with assembly forbidden, configuration
tampering, physical-qualification tampering and unsupported combinations pass.
Independent read-only audit found one P2: root/local decoder integration settings
could be replaced without invalidating operator/RHS identity. Root/local integration
now binds to prepared provider/configuration for RHS, field queries and snapshots;
negative controls cover both new and restored results. Follow-up evidence is
recorded below. Earlier row-only stagnation and all physical failures are retained.

`/tmp/gsp08-public-binding-v2.log`:2passed7.54s, including the audit regression.
Independent static re-audit confirmed the P2 closed with no remaining P1/P2 in
this public checkpoint scope; no duplicate GPU run. Shared public integration:
`/tmp/gsp08-public-integration-v1.log`:59passed57.33s across lifted, multibody,
direct restore and periodic public paths. That run started before the binding
fix; the v2 targeted run is the evidence for the new guard. These checks do not
repeat or replace the preserved internal baseline 33pass/1stagnation failure.

Next required work: full capacity/workspace preflight, packed exact near blocks,
accelerated setup and true nested H2, tensor/volume/periodic compositions,
large-scale complexity/crossover and physical acceptance. GSP-08 remains open.


## Packed exact near blocks

PackedNearBlocks stores variable-size exact leaves in shape groups without
padding. CSR row pointers, geometric node IDs and group/entry locators identify
the block rows; coefficient maps retain arbitrary all-J/all-M global ordering.
Forward and conjugate-adjoint share these fixed owned values and use bounded
block and RHS chunks. No global matrix or global pair table is introduced.
Native current gradients use the same grouped CUDA actions.

The H constructor replaces individual exact storage with views into these
packed groups. Norm calculations, selected-block checks and content receipts
continue using the same views; CSR metadata is included in retained bytes and
hashes. Duplicate node pairs, inconsistent node maps, mutable/nonfinite values
and insufficient retained budgets reject. Input assembly arrays are not aliased.
The packer checks its final retained requirement before stacking buffers.
Repacking still temporarily holds assembly inputs and packed tensors together;
this does not implement full setup/OOM preflight. Autograd can retain per-chunk
intermediates, so chunk bounds alone do not bound full backward memory.

`/tmp/gsp08-packed-near-v1.log`:3passed4.08s. Uneven rectangular block actions,
multi-RHS adjoint/current gradients, CSR locators, exact retained-byte counting,
input ownership, mutation rejection and pre-allocation budget failure covered.
Independent read-only audit found no P1/P2 in this increment and confirmed the
memory-boundary claims; no duplicate GPU run. Actual RWG integration results
are recorded below when available. No complexity or physical qualification.

`/tmp/gsp08-packed-integration-v1.log`:12passed65.57s, covering actual two-body
RWG action/adjoint/current gradient, exact leaves, equilibration and ordinary/
block Krylov, original-row reports, public restore and residual admission.
Balanced reconstructed operator error4.1952377627662785e-14. Ordinary two-sided
GMRES original dual residual1.7758260626038373e-12/current difference
4.1419957946829624e-11; block two-sided2.58809493491404e-11 /
2.342556493636277e-11. Row-only passed this run; its previous stagnation remains
an explicit unqualified control. No acceptance threshold changed.

CSR bookkeeping adds88bytes in this example: owned201936/shared7776/total209712
bytes versus dense main matrix230400. Shared-process PyTorch setup peak
increment66797056bytes; this excludes prior caches and is not a complete peak
capacity guarantee. This increment is an experimental packed-near checkpoint;
GSP-08 remains open for full resource admission, faster setup, genuine H2 and
all remaining composition/physical/complexity requirements.


## Verified adaptive cross construction

The optional adaptive pivot path samples one row and column per rank update.
Update size only schedules a complete residual scan; a zero/exhausted row,
candidate convergence or rank ceiling triggers that scan. Failed verification
restarts from the largest observed residual entry. Every returned factorization
passes the same complete balanced Frobenius gate and storage-saving requirement.
QR and a small-core SVD remove redundant pivots in balanced coordinates. The
triangle error budget only proposes truncation; the new factors are verified
again against every original entry before admission. These are native CUDA
operations on fixed sampled kernels; current-vector gradients remain supported.

HMatrixConfig(pivoting='adaptive') selects this path; complete remains default.
The configuration payload version is scalar_hmatrix_config_v2, strictly binding
pivot selection into public numerical/operator and snapshot identities. Historic
v1 configuration payloads are not silently interpreted as the new declaration.
Full verification remains quadratic work. This reduces repeated integration,
not the outstanding asymptotic setup requirement or global Riesz error gate.

`/tmp/gsp08-adaptive-cross-v1.log`:8passed29.77s for complete/adaptive controls
before recompression. Actual 60x60 RWG block: complete rank18/scans19/rowcalls152,
adaptive rank28/scans1/rowcalls36. The latter saved less storage; this tradeoff
motivated explicit recompression rather than relaxing error or storage gates.
`/tmp/gsp08-adaptive-cross-v2.log`:5passed,4deselected12.24s for affected adaptive
controls including a hidden disconnected component that a zero row update must
not miss. Actual block after recompression: rank16/scans2/rowcalls44/columncalls28,
relative balanced error4.558044418088344e-9 at unchanged rtol1e-8;
factors30720bytes versus dense57600bytes. Complete-pivot reference was not
repeated in this follow-up. Independent read-only audit found no P1/P2, verified
complex recompression and mandatory final scans, and did not repeat GPU tests.

`/tmp/gsp08-adaptive-integration-v1.log`:9passed47.30s. Actual two-body H
forward/adjoint/current gradient, preconditioner norms, original residuals,
public v2 snapshot identities and admission controls pass. Balanced global
block Frobenius error2.057569938904046e-14. Ordinary two-sided original dual
residual1.7700708249897534e-12/current difference4.1470023211448994e-11;
block two-sided1.8208357483805265e-10 /3.7801331878710354e-10, both below
unchanged1e-8 dual/1e-7 current gates. Row-only again stagnated: first RHS raw
relative residual1.6924897758453562e-12 above1e-12, and remained nonconverged.
The explicit pre-existing stagnation branch records this unqualified case;
9 passing tests must not be read as row-only convergence qualification.

Owned tensors194256bytes (far factors61440), shared7776,total202032 versus
dense main230400. Shared-process setup peak increment66788864bytes is only
an observational receipt, not peak/OOM admission. This checkpoint retains
complete pivoting as default and does not certify asymptotically fast setup.
Full resource preflight, genuine H2/FMM, compositions and all original physical,
complexity and cross-repository acceptance requirements remain open.


## Genuine nested H2 basis and actions

NestedClusterBasis retains orthonormal leaf bases and child-parent transfer
matrices, without expanded internal-node bases. Construction restricts ancestor
seed spaces into descendants, adds each node's own seeds, then builds parents
in the direct sum of their children's spaces. A rank ceiling fails explicitly.
Every supplied seed is projected and independently reconstructed through the
nested representation at the declared tolerance before acceptance. Fixed CUDA
SVD constructs the bases; current-vector upward/downward passes are native Torch.

NestedH2Actions converts balanced H far factors into distinct shared test and
trial spaces, and small test-rank by trial-rank coupling matrices. Each actual
coupling is checked against the input balanced factor product by a complete
bounded-row scan, independently of the individual seed checks. Forward uses
trial upward aggregation, coupling products and test downward evaluation;
adjoint reverses the same operations and conjugates the same coupling tensors.
Physical impedance scaling is applied on the appropriate input/output sides.
Exact packed near blocks and the tree are explicitly shared and guarded.
The source H object and its per-block far factors are not retained.
H block node IDs now follow the packed/exact/far ordering and are mutation-bound.

This is an internal algebraic action layer, not yet the shared solver protocol
or public H2 runtime. It inherits the reference H setup and temporarily retains
seed/ancestor spaces during conversion; no fast setup or full peak-memory
preflight is claimed. Per-seed and H-to-H2 block tolerances do not replace the
original quadrature error, global Riesz or physical-error acceptance budgets.
Directional high-frequency and tensor/volume/periodic traits remain required.

`/tmp/gsp08-h2-basis-v1.log`:2passed5.19s. A multilevel octree with ancestor and
local seed spaces verifies exact nested transfer identities, orthogonality,
upward/downward dot identity <1e-11, current gradients, ownership/mutation,
rank-limit failure and empty spaces. Only leaf bases and transfers are retained.
`/tmp/gsp08-h2-actions-v1.log`:2passed22.21s. Independent multilevel synthetic
couplings cover internal and descendant interactions, unequal positive scales,
batched forward/adjoint/gradients and independent storage ownership. Actual
two-body RWG H conversion preserves shared exact near storage and passes dense
comparison: H-to-H2 balanced global Frobenius error1.371707600499067e-20,
original dense error2.0575699399219372e-14. Individual converted far-block
errors1.878265972611242e-15 and3.802475760948123e-15 are below the declared1e-10.
Both test/trial leaf ranks are16. Independent read-only audit found no P1/P2 in
this increment, reviewed both logs and did not repeat GPU tests.

Action-owned tensors73472bytes, shared tree888 and near117208,total191568.
This action-only total excludes the source H metric, prepared quadrature geometry,
physical input meshes and solver state; it must not be compared as though it
were a complete solver footprint. Leaf bases/couplings can cost more than H
factors on small trees. Storage slopes/crossover and full resource admission
remain unqualified. No stage closure or production claim follows from these
small controls. Next: shared Riesz/Krylov protocol with nested norm calculation,
public physical runtime/persistence, resource preflight and scale acceptance.


## Shared H2 solver protocol and nested equilibration

HierarchicalH2Operator converts a verified H partition into the nested action
representation, sharing only the provider, local Riesz metric, tree and near
storage. The source H object and far factors are released. Arbitrary overlapping
far blocks cannot enter this solver adapter: squared block norms are additive
because its input is the verified disjoint H partition.

Weighted node Grams Q.H diag(w) Q are computed by leaf evaluation and upward
E.H Gchild E transfers. Far couplings form node quadratic matrices, which travel
downward as E Wparent E.H and sum with local forms. Leaf diagonals yield row or
column squared norms, with the appropriate physical impedance and requested
column scaling. Exact near norms are added in bounded batches. No internal
basis expansion or far-block reconstruction occurs in this equilibration path.

The adapter reuses the shared H vector/read-lease/Riesz/residual-report/Krylov
admission algorithms. H2Config now has strict nested_h2_config_v1 payload/restore
and explicit compressed-dual/original-sample budgets (default1e-8 each). Original
samples remain separate from a global original Riesz certificate. Content
receipts hash nested tensors and mark executable_snapshot=False; public physical
snapshot/runtime integration is not provided by this internal adapter.

Storage receipts now add the shared metric, prepared quadrature geometry and
physical input storage to the H2 action footprint. Solver/autograd workspace,
host objects and allocator caches remain outside these retained counts; full
capacity preflight is still open. Independent read-only static audit found no
P1/P2 in this increment and did not repeat GPU tests. Numerical evidence follows.

`/tmp/gsp08-h2-krylov-v1.log`:3passed38.30s. Actual two-body RWG operator
passes protocol detection, nested row/column norms against an independently
reconstructed oracle at rtol2e-12, current residual-loss gradient, two physical
incident polarizations and both ordinary/block GMRES. Internal-basis expand
methods are forbidden during equilibration and solves. Weak-reference collection
confirms the original H object is released. Metric mutation rejects reuse.
A separate multilevel test compares weighted Grams and overlapping node quadratic
form diagonals against explicit expanded test oracles; config budgets roundtrip.

Ordinary two-sided GMRES original full dual residual1.7710305101509657e-12,
balanced current difference4.1322771108500876e-11. Block two-sided values
1.8171022325400224e-10 /3.82649941703808e-10. Original row-sample maximum
relative residuals1.8164738293829025e-12 and2.456677199927622e-10. The original
1e-8 dual/sample and1e-7 current thresholds are unchanged. Previous row-only
stagnations remain recorded; this test does not requalify that branch.

H2 complete referenced tensor bytes212144 = action191568 +metric7680
+prepared geometry5120 +shared inputs7776. This is larger than the comparable
adaptive H referenced footprint202032 for this small example; no H2 storage
crossover or complexity claim is made. Complete setup/solver/backward peak
accounting and capacity preflight remain open. Next public runtime and exact
physical snapshot integration must preserve both H construction and H2
conversion/solve identities and explicitly reject unsupported compositions.


## Public H2 physical runtime and exact decoder restore

MoMConfig now selects operator_backend='h2', with separate hierarchical
(HMatrixConfig construction) and h2 (H2Config conversion/solve) declarations.
Both versioned declarations bind numerical/operator/RHS identity. Local Riesz
storage and iterative stopping without direct fallback are required. Periodic,
harmonic-balance and scalar-compiler tensor exclusions remain explicit.

HierarchicalH2SystemOperator constructs the temporary H reference, converts it
to nested H2 and releases the H object. Physical decoder/carrier/local integral
bindings and lifted contrast RHS logic are shared with the public H runtime.
The H2 base constructor performs its own completed-state check before derived
physical handles are initialized. Runtime dispatch uses H2 residual-budget
admission. Source H exact/far counts and compression rejection receipts are
preserved in metadata, including cases that contain only exact near leaves.

Schema 9 retains exact scattered currents, source carriers and both configuration
identities. Restore compiles only the physical decoder, never an H/H2 operator;
outer Result metadata must match the native backend/operator/RHS identity.
Changing construction tolerance, H2 budgets, source/region/integration bindings
or the false physical-qualification flag rejects. This is physical field
restoration, not persistence of compressed factors or Krylov continuation.

Initial `/tmp/gsp08-h2-public-v1.log`:1failed,3passed26.95s. The two-body
order2/singular3 example solved, but the test requiring two actual far couplings
observed zero. `/tmp/gsp08-h2-public-compression-diagnostic-v1.log` explains why:
adaptive factors of rank32/31 passed residual tolerance but cost61440/59520bytes,
exceeding dense57600bytes. Compression correctly rejected and retained four
exact leaves. This failure is preserved. The follow-up public H2 fixture uses
order3/singular3, matching the earlier actual H2 reference controls; numerical
and storage acceptance thresholds are unchanged. This does not qualify order2
compression or general adaptive rank robustness. Follow-up evidence is below.

Independent read-only static audit found no P1/P2 in the public integration
scope; no duplicate GPU run. Final resource/physical/complexity qualification
and all outstanding directional/tensor/volume/periodic/FMM requirements remain.

`/tmp/gsp08-h2-public-integration-v2.log`:63passed81.87s. Public H and actual
two-body H2 paths (requiring two far couplings), lifted/multibody/direct-restore
and periodic public integration pass. Exact restored currents and exterior fields
are bitwise equal with both H/H2 constructors forbidden; native and outer backend
identity tampering reject. Construction/H2 budget changes and root/local integral
changes reject for new and restored results. Public solve uses checked H2 residual
budgets and preserves physical_qualification=False. This stage check does not
repeat or replace full cross-repository, physical or complexity acceptance.

Final independent re-audit confirmed the additional source diagnostics are deep
copies, do not retain the source H object or alter numerical identity, and add
no P1/P2. The auditor checked the63pass log without repeating GPU tests.


## H2 structural retained-storage admission

H2Config v2 adds retained_budget_bytes (default1GiB), independently of the source
H construction budget. plan_h2_storage consumes tree topology and input factor
shapes before balancing or nested conversion. For each test/trial family it
propagates ancestor seed column counts, then bounds accepted leaf ranks by
min(max_rank,seed_columns,node_width) and accepted parent ranks additionally by
the sum of child rank bounds. The plan counts both families' leaf values and
indices, transfer matrices, every small coupling and the two real scaling vectors.
Zero-rank spaces still charge their index maps and scales.

The bound may exceed the ultimately needed numerical rank/storage; conservative
rejection is explicit. Exceeding the owned retained budget raises H2CapacityError
with the full structural receipt. The generic constructor checks before its own
arrays are allocated; the H conversion entry also checks before input scaling
arrays. After accepted construction, actual owned retained bytes must not exceed
the structural bound. No dense matrix or global pair list is introduced.

The budget does not cover shared tree/near/metric/geometry/input storage, source
H factors held during conversion, ancestor seeds, SVD/allocator workspace,
solver/backward state or host control objects. It is not a complete OOM preflight
or a qualification of peak GPU capacity. Versioned H2 configuration and public
schema9 numerical identities include the new declaration; older config payloads
are not silently reinterpreted.

`/tmp/gsp08-h2-capacity-v1.log`:1failed,3passed5.13s. The synthetic root-factor
oracle omitted the Morton permutation and compared different coefficient orders;
its reference mapping was corrected without changing tolerances or the operator.
`/tmp/gsp08-h2-capacity-integration-v2.log`:4passed26.19s. Structural bounds cover
actual nested ranks and storage, exact-budget admission and one-byte-below-bound
rejection work, zero spaces retain map/scale charges, and allocation sentinels
confirm rejection precedes owned scales/seeds/SVD. The actual public two-body
H2 solve and schema9 exact restore also pass with the version2 budget identity.
Independent read-only audit found no P1/P2 and confirmed the bound/counting logic.
It noted that the H factory still allocated input scales before its constructor;
the additional factory gate addresses this boundary. Its focused test
`/tmp/gsp08-h2-capacity-entry-v3.log`:1passed3.29s, without repeated full solves.

Full conversion/setup/solve/backward memory admission remains required, as do
fast kernel-based construction, directional/tensor/volume/periodic composition,
scale/crossover, physical and final cross-repository acceptance. This is one
resource-control increment, not GSP-08 or Goal completion.

## Direct kernel interpolation foundation

New internal InterpolationBox owns CUDA Chebyshev-Lobatto cardinal nodes,
including collapsed axes and parent-to-child polynomial transfers.
RWGInterpolationMoments integrates vector and divergence moments over complete
selected edge supports, preserving arbitrary edge order and bounded face batches.
It rejects a stale basis before reading cached indices or creating geometry.
InterpolatedRegionalBlock samples only free-space Green and gradient node tables
and evaluates the scalar regional D/K weak form, including magnetic/lossy media,
rectangular forward/adjoint and native current gradients. Strong full-support
box admissibility protects this far-only primitive; exact near rules are unchanged.
The explicit preallocation budget covers retained kernel tables only.

Targeted CUDA evidence:
- `/tmp/gsp08-interpolated-far-v1.log`:1 failed,1 passed; the fixture used complex
  static public Material parameters, which its existing constructor rejects.
  Replaced with supported electric/magnetic conductivity for sampled complex media.
- `/tmp/gsp08-interpolated-far-v2.log`:1 failed,1 passed; five nodes per axis gave
  balanced relative error 1.2987038311134344e-8, failing the unchanged 1e-8 gate.
- `/tmp/gsp08-interpolated-far-v3.log`:2 passed,5.03s; two/five/six node errors
  9.34583884726723e-3 / 1.2987038311134344e-8 / 1.08551627106382e-10.
  Cardinal polynomial/transfer, collapsed axis, rectangular adjoint/current
  gradient, complete-support, table-budget and mutation controls pass.
- `/tmp/gsp08-interpolated-moments-v1.log`:1 passed,4.22s; independent integrated
  RWG vector identity, divergence cancellation, edge permutation and mutation.

Independent read-only audit found a P2 stale-basis construction gap; the explicit
pre-selection geometry-version check fixes it. Focused stale-vertices/faces
sentinels verify rejection before any old indices are read.
This is a direct-kernel building block, not a whole-system nested construction.
It supplies neither automatic interpolation error certification nor a storage
crossover, setup-complexity or physical qualification. Global nested moments,
error-controlled tree couplings, high-frequency directional/tensor/volume/periodic
composition and all original final qualification requirements remain open.
Focused audit fix evidence: `/tmp/gsp08-interpolated-stale-v1.log`:2 passed,
3.04s, with vertices/faces mutation and pre-selection sentinels. No covered full
solve or regression was repeated for this isolated constructor guard.
Independent read-only re-audit closed the stale-basis P2 and found no remaining
P1/P2 in this increment; the documentation scope and retained failure were checked.

## Nested direct RWG projection

NestedRWGMoments connects complete-support octrees to direct interpolation.
It integrates only leaf RWG supports, shares one prepared geometry, retains
leaf vector/divergence moments and child-parent cardinal transfers, and aggregates
four-channel moments upward. Downward evaluates the Hermitian adjoint of the
combined all-node projection, including contributions injected at internal nodes.
Both paths preserve leading batch dimensions and native current gradients.
No expanded internal RWG matrices are retained. This entry currently covers one
whole single-body basis; global regional/multibody coupling remains to implement.

`/tmp/gsp08-nested-moments-v1.log`:1 passed,4.80s. A tree of depth at least two
compares every internal projection to independent direct support integration;
combined all-node adjoint and autograd agree, and transfer mutation rejects.
The single-body/tree identity, full permutation, geometry and owned tensor
versions are guarded. Retained receipt includes owned interpolation boxes,
transfers and leaf moments, excluding shared geometry/tree. It is accounting,
not peak-memory admission. No new physical or complexity qualification follows.
Independent read-only audit found no P1/P2 in transfer directions, all-node
adjoint, edge ordering, complete support or mutation guards; it verified the
existing targeted log without repeating GPU work.

## Direct regional and multi-body actions

InterpolationKernel now owns the shared Green/gradient node coupling and four
xyz/divergence channel algebra. InterpolatedRegionalBlock and the new tree path
reuse it. DirectRegionalOperator consumes the disjoint complete-support block
partition, couples admissible nodes directly, and calls original selected weak
quadrature only for terminal exact blocks. Rectangular near matrices reuse the
packed action primitive through disjoint coefficient/node namespace embedding.
The adjoint reverses both embedding and tree propagation.

DirectSystemActions composes body-tree regional maps in global all-J/all-M
ordering. Each self pair sums exterior and its own interior; each cross pair
contains exterior once. Projection geometry is shared with ScalarSystemBlocks.
No reference H construction/conversion or far RWG pair integration is involved.
The independent cumulative budget counts retained Green/gradient tables; it does
not count moments, transfers, exact blocks, packing copies, allocator/solver or
backward workspaces. The multi-body implementation still enumerates body pairs
and repeats projection work per regional action. Therefore neither body-count
scaling nor a storage/performance crossover is qualified. Public H2 remains on
the existing reference-H conversion until error/resource admission is completed.

Evidence:
- `/tmp/gsp08-direct-regional-v1.log`:7 passed,9.09s. Shared-kernel refactor retains
  cardinal/moment/gradient controls. Actual far tree root coupling forbids all
  selected weak block calls, relative balanced error 4.770851176392813e-11.
  Exact self partition sums both media with error 2.215033609460333e-16.
  Batched adjoints/gradients, geometry guards and kernel budget rejection pass.
- `/tmp/gsp08-direct-system-v1.log`:1 passed,12.93s. Two bodies have two exact self
  maps plus two directly sampled cross maps. Full 120x120 original weak matrix
  error 7.646345715648447e-15; adjoint/gradient and global index mutation pass.
  Total far kernel tables occupy 5,971,968 bytes, exceeding the small dense
  reference's storage: no compression/crossover claim is made from this fixture.

Independent read-only audit found no P1/P2 in D/K coefficient/adjoint algebra,
rectangular embedding, regional counting, cumulative kernel budgets or bindings.
Fixed-order interpolation remains explicitly uncertified. Automatic error
control, complete retained/peak admission, Riesz/Krylov/public runtime integration,
scaling, directional/tensor/volume/periodic and final physical qualification remain.
Additional distinct-width validation `/tmp/gsp08-direct-rectangular-v1.log`:
1 passed,5.79s, 60x240 exact near map balanced error 3.35659329677387e-14,
with native adjoint and current-gradient checks. Covered square/system tests
were not repeated for this added mapping case.

## Shared direct-system tree passes

DirectSystemActions now gathers each body's coefficients once, lazily computes
one upward map for each body participating as a far source, accumulates all
regional kernel contributions in per-target node arrays, and evaluates one
downward map per active target. Adjoint reverses region/source/target while
preserving the same shared schedule. Exact near contributions keep the packed
regional action and original quadrature. Regional standalone actions share the
same coupling and exact-embedding helpers, avoiding a second numerical formula.

This supersedes the repeated-per-region projection limitation above. The body
pair list and regional kernel tables remain quadratic in body count in the worst
case; there is no global body tree or complexity qualification from this change.
All-node projection arrays for every active body coexist during an action, so
reduced arithmetic does not imply a demonstrated peak-memory reduction. Full
resource admission and fixed-order interpolation error control remain required.

Independent read-only audit found no P1/P2 in source/target reversal, shared
four-channel accumulation, leading batches, autograd or entrypoint guards.
`/tmp/gsp08-direct-shared-passes-v1.log`:5 passed,32.19s. Three bodies and two
leading batch dimensions compare shared forward/adjoint to standalone regional
sums, enforce exactly one upward/downward call per body/action, and verify native
current gradients. The affected regional far, exact self, complete two-body and
60x240 rectangular tests also pass. No unrelated/full regression was rerun.

## Global exterior tree and body-local interiors

DirectSystemActions now constructs one complete-support tree over all global
RWG edges for the common exterior, plus one tree per body for its interior only.
The old ordered body-pair operator list is removed. Global exterior partitioning
can couple clusters spanning multiple bodies, and the exterior action performs
one shared upward/downward traversal. Internal media cannot couple distinct bodies.

NestedRWGMoments accepts either a single-body basis or ScalarSystemBlocks. Global
leaves integrate each touched body's local supports and scatter back into the
original global leaf order; all internal transfer/action algorithms are shared.
ScalarSystemBlocks.exterior_block reuses selected weak quadrature with exactly
one exterior contribution, including self pairs. Its general full-system path
still includes each same-body interior once. A guarded CUDA boundary index routes
leaf and near selections only to touched bodies and is included in provider
geometry storage accounting.

`/tmp/gsp08-global-exterior-v1.log`:9 passed,47.73s. Three-body all-node projection
count is exactly one exterior upward/downward per action; actions/adjoints match
original full weak assembly and native current gradients agree. Affected
single-region, rectangular, two-body, nested-moment and selected-block checks pass.
The subsequent routing change is checked separately below; this first log is not
claimed as validation of code changed after test startup.

This replaces the body-pair enumeration limitation in previous checkpoints.
No empirical complexity qualification follows: rank/order growth, retained kernel
size, all-state version checks, near density and peak workspaces still need controlled
scaling/resource evidence. Interpolation remains fixed-order and uncertified, and
the experimental public H2 solver still uses its existing reference-H conversion.
`/tmp/gsp08-global-routing-v1.log`:5 passed,7.46s. One global leaf explicitly
mixes both bodies and matches independently integrated body moments in global
order. Interleaved selections straddling exact body-edge boundaries satisfy
full = exterior + same-body interiors; routing mutation rejects. Affected
single-body nesting and selected weak integration checks remain passing.
Independent read-only audit found no P1/P2 in regional counting, mixed leaves,
complete supports, edge routing or shared provider storage/guards.
`/tmp/gsp08-global-grouped-far-v1.log`:1 passed,33.09s. Four bodies form two
separated groups; two far node pairs each contain multiple bodies on both sides.
The complete inter-group block matches independently selected original weak
integration with balanced relative error 1.1801821213581889e-11 (<1e-8).
This explicitly verifies aggregation across material-body boundaries, not the
accuracy of every within-group interaction or a whole-family error certificate.

## Analytic scalar kernel truncation admission

interpolation_error.py derives a uniform exact-arithmetic interpolation bound for
G and each component of grad G from one-dimensional Bernstein ellipses, a bound
on the outgoing complex radial square root, and six-axis tensor telescoping.
The proof and primary approximation-theory references are maintained in
`docs/reference/mom_scalar_interpolation_bounds.md`. Geometry/bound arithmetic
runs on CUDA float64; no edge-pair or quadrature sample scan is needed.

KernelErrorBudget declares separate positive absolute G/gradient thresholds.
InterpolationKernel optionally checks these before allocating its tables;
DirectRegionalOperator and DirectSystemActions propagate the declaration.
The regional receipt records both budget and exact kernel-truncation scope.
This does not change existing relative operator/physical tolerances, does not
include interval-certified floating-point arithmetic, and does not certify
RWG quadrature/projection/transfers or a complete system operator. Default
fixed-order behavior remains experimental when no budget is supplied.

`/tmp/gsp08-kernel-bound-v1.log`:8 passed,4.18s, including independent kernel
samples, order refinement, degenerate axes, preallocation rejection and affected
single-block actions. Independent audit then found a P2 beyond these samples:
an exponential could underflow before multiplication by a large inverse distance
or wave number, incorrectly reducing a representable bound to float64 tiny.
All exponential/distance/wave/interpolation/Lebesgue factors now combine in log
space; exponentiation/flooring occurs only after complete bound accumulation.

`/tmp/gsp08-kernel-bound-v2.log`:6 passed,3.46s. The degree-zero extreme-scale
counterexample has actual errors approximately 2.92e-249 (G), 2.34e-146 (gradient),
while the corrected bounds are 7.768e-240 and 6.546e-137. Other checks cover
normal sampled errors/refinement, accepted and rejected budgets, configuration
mutation, collapsed axes and global-system budget propagation. Audit re-review
closed the P2 without repeating GPU work.

For one ordinary six-node example the G bound is 1.213e-6 and gradient bound
2.299e-5, much larger than sampled errors around 1e-12. Conservative rejection
is expected; measured sample errors cannot replace the analytic gate. Automatic
order/resource selection and translation to rigorous RWG/system error budgets
remain to implement before claiming error-certified direct construction.

## Preallocation automatic uniform order

scalar_box_interpolation_bound evaluates prospective node counts directly from
four box bounds, without tensor grids. The existing InterpolationBox entry uses
the same calculation. plan_uniform_kernel_order streams each global-exterior
and body-interior partition for every candidate in the declared inclusive order
range. Every far block must pass both absolute kernel bounds; no monotonic-error
assumption or sample test replaces this check. Tables count four complex entries
per pair of interpolation nodes, respecting collapsed axes. A cumulative table
budget failure can terminate the search because node counts cannot decrease.

DirectSystemActions(order=min_order, max_order=..., kernel_error_budget=...)
builds its support trees once, plans before any interpolation nodes/moments or
regional operator, then constructs the selected uniform order. Actual retained
kernel bytes must exactly match the plan. The immutable, guarded JSON plan is
returned by copy in diagnostics. Fixed-order construction remains available when
max_order is omitted. Quadrature order remains an independent caller declaration:
a kernel interpolation bound does not certify moment integration or transfers.

`/tmp/gsp08-order-plan-v1.log`:9 passed,4.85s: prospective/current bound controls,
first passing order, every earlier failed order, full table counts, no tensor-grid
allocation during planning, failed-plan sentinel before moments, actual table
count equality, degenerate axes and original kernel-bound controls.
Independent audit identified a P2: a low-order bound exceeding native finite
range could prematurely terminate instead of allowing later order decay.
KernelBoundUnavailable now marks only that candidate failed; other domain errors
remain fatal. Each subsequent candidate still checks every far block.
`/tmp/gsp08-order-plan-v2.log`:4 passed,9.04s. A real high-wave-number 1D-box case
starts with overflowing bounds and selects order186 within its declared range
and table budget. This uses a deliberately huge absolute budget to exercise
finite-range recovery, not to relax any operator or physical acceptance gate.
Independent read-only re-audit closed the P2 and verified plan-copy guards.

The preflight covers only retained kernel tables and analytic exact-arithmetic
kernel truncation. It excludes trees, projection/transfer/near storage, workspaces,
quadrature, floating-point certification, full operator-relative error and all
physical/complexity qualification. Those and final scope remain required.

## Kernel-to-RWG absolute interpolation error

rwg_interpolation_error.py derives per-edge u>=integral|f| using exact triangle
second moments and Cauchy-Schwarz, plus q=integral|div f|. Scaled geometry and
log-space face/edge/tree reductions avoid recoverable underflow and unnecessary
expanded matrices. Each cluster stores only log sums of u^2, q^2 and u*q.
Uniform kernel G/gradient-component bounds then majorize D and K weak errors,
including complex wave number/material factors and the existing impedance
balance. The derivation is in docs/reference/mom_rwg_interpolation_bounds.md.

Media contributing on one coefficient block overlap, so their error bounds add.
Disjoint partition blocks combine by root sum of squares. Body-interior blocks
also combine by root sum of squares, then their joint bound adds to the global
exterior bound. These rules avoid incorrectly assuming orthogonality across
regional contributions occupying the same entries.

`/tmp/gsp08-rwg-bound-v1.log`:2 passed,4.32s. Independent high-order triangle
quadrature stays below vector integral bounds, divergence integrals equal twice
the edge length, stale basis rejection works, and an actual lossy/magnetic far
weak-matrix difference 7.005563098889053e-9 is below its absolute balanced bound
0.006016835573004948. This large margin is conservative, not an accuracy claim.
`/tmp/gsp08-rwg-bound-composition-v1.log`:2 passed,3.75s, including the far control
and explicit overlapping-media/disjoint-interior combination checks. Independent
read-only mathematical audit found no P1/P2 in triangle geometry, sqrt(3) gradient
factor, complex |k|, impedance weights or overlap rules.

The result is an absolute error component for an ideal exactly integrated
interpolant, numerically evaluated in float64. It excludes actual moment
quadrature, nested-transfer rounding, floating-point kernel evaluation and solve
error. It does not yet provide a reference-norm lower bound or an implemented
operator relative-error certificate. Those, resource admission, public direct
solves and all final qualification remain required.

## Degree-controlled moment integration

The default direct moment quadrature is now selected from the actual cardinal
axis degrees, including collapsed axes, the linear RWG factor and Duffy Jacobian:
q=(sum(axis_node_count-1)+4)//2. This propagates from local/mixed-body leaves to
nested and global-system construction. User-specified smaller quadrature orders
remain available for controlled experiments but are explicitly marked insufficient
for polynomial exactness. Actual/required orders and exact-arithmetic scope are
included in moment/system and ideal RWG error diagnostics; metadata is guarded.

`/tmp/gsp08-moment-degree-v1.log`:6 passed,9.99s. For six nodes per axis, automatic
q9 agrees with independent q12 moments; explicit q3 differs by relative
0.053210614700684604 and is marked nonexact. Collapsed-axis degree counts and
nested root transfer agree with overintegrated direct parent moments. The affected
automatic interpolation-order planning checks also pass. Independent read-only
mathematical audit found no P1/P2 in degree, default propagation or scope claims.
`/tmp/gsp08-moment-degree-system-v1.log`:1 passed,3.54s verifies the actual global
system default and moment-condition propagation into ideal error receipts.

This establishes a sufficient exact-arithmetic polynomial quadrature condition.
It does not certify native Gauss/barycentric/accumulation/transfer rounding,
nonpolynomial exact-near quadrature, a relative operator error or physical accuracy.
Full resource/relative-error/public-direct-solve and final qualification remain.

## Preallocation RWG absolute admission

DirectSystemActions(..., max_order=..., rwg_absolute_budget=...) now computes
geometry-only integral logs and tree statistics before interpolation construction.
The planner uses each region's own medium, sums squared bounds over its disjoint
partition blocks, and adds the global exterior bound to the root-sum-square body
interior bound. Both absolute kernel budgets and the declared balanced RWG budget
must pass, together with retained table capacity. Statistics are version-guarded;
failed candidates do not bypass any bound. Failure receipts preserve the requested
RWG budget and balancing impedance as well as per-order diagnostics.

The direct-system path requires automatic order planning for this admission and
rejects an explicit quadrature order below the selected degree before moments
are allocated. Default automatic polynomial integration satisfies the condition.
The bare planner's RWG-input contract explicitly requires the first region to be
the common exterior and the rest to represent disjoint interior coefficient spaces;
the system wrapper supplies precisely that validated decomposition.

`/tmp/gsp08-rwg-order-admission-v1.log`:8 passed,11.10s. A two-body setup with loose
kernel budgets selects order6 due to the RWG budget, planned bound
5.973196309066437e-7 <= 1e-6, matching the post-construction ideal system bound.
Every lower candidate fails the RWG condition. Sentinels confirm budget failure
and underintegration rejection precede moment construction. Existing absolute
bound/composition, kernel-order, degenerate-axis and overflow-recovery controls
remain passing. Independent read-only audit found no P1/P2.

This closes ideal absolute interpolation admission only. No reference-norm lower
bound, native floating-point error enclosure, near-quadrature/solve certificate
or physical qualification is claimed. The original relative/physical thresholds,
full resource work, public direct solve and all final scope remain required.

## Direct-kernel iterative adapter

DirectHierarchicalOperator now wraps DirectSystemActions with the shared surface
operator protocol, LocalRWGRiesz and hierarchical residual admission. It never
constructs the reference H representation. Left geometry scaling is
diag(eta/sqrt(Gii), 1/sqrt(Gii)); right scaling swaps the J/M halves. This is
explicitly named rwg_diagonal and does not claim operator row/column equilibration.
Forward, adjoint and native current gradients use the direct nested actions.

The two-body Ex/Ey control uses original weak quadrature and a dense oracle only
in the test. GMRES and block GMRES pass the unchanged full original dual residual
1e-8 and balanced current error 1e-7 gates. Construction is instrumented to reject
any reference H initialization. Iterative failures remain failures without
post-solve admission or fallback; explicit direct/LU requests reject. Diagnostic
persistence receipts do not claim an executable compressed snapshot.

The public Scene runtime still uses the existing H-to-H2 conversion. Direct public
configuration and Result identity integration, complete resource admission,
relative implemented-operator error, complexity and physical qualification remain
required. Independent audit and full regression are deferred to the stage merge
and final acceptance, as requested.

Evidence: /tmp/gsp08-direct-krylov-v2.log, 1 passed in 14.35s. GMRES full
original dual residual 5.664485803187231e-13, current error
2.9004076785595226e-11; block GMRES residual 1.10452388863486e-12,
current error 2.1188747559284423e-11. This includes failure/no-fallback
checks and the Riesz current-gradient control. No full regression was repeated.

## Public direct H2 construction and decode persistence

The experimental direct_h2 backend now routes Scene/Simulation through
DirectH2SystemOperator, reusing the common physical decoder and lifted RHS.
DirectH2Config version 1 preserves order range, moment quadrature, leaf size,
admissibility, table bytes, optional KernelErrorBudget and ideal RWG absolute
budget, plus both solve residual tolerances. Invalid configuration rejects before
operator allocation. Reference H configuration is disallowed on this path.

Schema 10 stores those inputs with exact scattered currents, carrier and body
geometry/material data. Numerical, sampled operator and RHS identities include
the direct configuration. Restore reconstructs only the field decoder and checks
the identities; it does not claim to persist executable compressed tensors.
Mutation of backend, interpolation order or physical-qualification metadata rejects.
The constructor calls its own structural guard before physical decoder binding
to avoid premature dispatch into the physical subclass refresh method.

Public direct smoke/restore: /tmp/gsp08-direct-public-v1.log, 2 passed,
4 deselected, 7.49s. Sentinels prohibit dense system assembly and reference H
construction during solve, and all hierarchical runtime constructors during
restore; restored coefficients and queried fields agree exactly. Independent
read-only integration audit found no P1/P2 and requested the feature inventory
update, which is included here.

This supersedes earlier notes that direct public integration was pending. It
does not close complete resource accounting, relative implemented-error bounds,
FP/near-quadrature certification, high-frequency/directional/periodic/tensor
compositions, physical or measured complexity qualification.

Affected interface integration evidence: /tmp/gsp08-direct-public-merge-v1.log,
20 passed in 15.03s, covering strict configuration roundtrips/invalid inputs
and all H/H2/direct-H2 public solve/restore controls. No unrelated full suite
was repeated. Full phase/cross-repository acceptance remains pending.

## Direct action retained capacity

DirectSystemActions now plans every action-owned retained tensor using existing
support trees and the selected interpolation order, before creating moments or
assembling weak near blocks. Collapsed axes use one interpolation node. Boxes
include bounds/center/width, cardinal coordinates/weights and physical nodes;
leaf moments include owned edge maps and four complex channels. Every transfer
and far kernel table is counted. Exact near packing counts matrix entries,
per-block row/column maps and CSR locators; global body maps and all trees are
included once. Distinct native storages are checked against the plan after
construction, avoiding duplicate counts for aliases/views.

The immutable serialized capacity receipt reports categories and exclusions.
DirectH2Config version 2 adds action_retained_budget_bytes (default 1 GiB),
which participates in public numerical/operator/RHS identities. The kernel-table
limit remains an independently enforced subset. Capacity failure includes a
receipt and occurs before interpolation or original weak-block construction.

This scope excludes provider geometry/input tensors, LocalRWGRiesz, solver
and backward state, quadrature/repacking/action workspace, allocator cache and
host metadata. Trees and provider geometry already exist at planning time.
It is not a complete resource or peak-memory gate. That original requirement
remains open together with complexity and physical qualification.

The first test invocation named a nonexistent test_direct_system.py; no tests
ran (/tmp/gsp08-direct-capacity-v1.log). Corrected existing regional/system
controls: /tmp/gsp08-direct-capacity-v2.log, 4 passed in 26.27s, including
actual post-construction storage equality in the multibody path.

Capacity/config/public direct controls: /tmp/gsp08-direct-capacity-v3.log,
18 passed, 4 deselected in 8.23s. Includes one-byte-under-budget rejection
with construction sentinels, independent actual storage enumeration, immutable
receipt copies, alias storage counting and public exact restore. No full suite
or independent audit repeated for this increment; phase/final checks remain.

## Streaming exact near construction

PackedNearBlocks.from_layout accepts fixed block maps and a value factory.
The shared initializer validates the layout and retained budget before invoking
the factory, allocates each final shape group, copies one validated fixed block
and deletes that temporary before the next factory call. The existing record
constructor reuses this same path. DirectRegionalOperator now gathers only
near layout metadata during partition traversal; original weak quadrature is
called from the factory, preserving rectangular embedding and the global
exterior versus body-interior decomposition. No list of all weak value matrices
is retained alongside the packed result.

/tmp/gsp08-stream-near-v1.log: 10 passed in 27.76s, covering packed forward,
adjoint and current gradients, actual regional/system weak-matrix comparisons,
retained storage equality and early rejection. Weak references prove each
factory-produced matrix dies before the next request; failed retained admission
never calls the factory. Existing retained-action capacity counts are unchanged.

This removes one concrete setup overlap. Layout maps, one temporary block,
quadrature and native allocator workspace still contribute to peak usage.
A complete setup/solver/backward/host capacity gate and final complexity/physical
qualification remain required.

Native allocated-peak control: /tmp/gsp08-stream-near-peak-v1.log, 1 passed
in 5.37s. Sixteen 256x256 complex128 blocks: buffered construction peak
34,408,448 bytes; streamed construction peak 18,679,808 bytes. Both measure
process allocated bytes above the same input-layout baseline. This small
synthetic storage control does not certify full physical setup peak or scaling.

## Fixed-current adjoint backward

DirectSystemActions now wraps differentiable current actions in a native
PyTorch linear autograd primitive. Forward runs the existing guarded action
without recording its internal projection graph; backward invokes the opposite
operator action. It retains a reference to the fixed operator but saves no
forward current or action intermediates. Re-entering the primitive under
create_graph preserves higher current derivatives. The original numerical
action is unchanged, and stale operator storage is rejected before backward.

/tmp/gsp08-direct-backward-v1.log: 2 passed in 14.79s. Forward/adjoint controls
use noncontiguous multi-axis RHS batches, compare gradients with the original
recorded action graph, verify quadratic Hessian actions and reject a kernel
mutation between forward and backward. Saved-tensor hooks observe 74 entries
for each raw action graph and zero for each wrapped action. These counts are
not a measured byte saving: some raw entries alias fixed operator arrays.
The actual Krylov/Riesz integration also passes; full original dual residuals
are 5.664891162014745e-13 (GMRES) and 7.874663420731501e-13 (block GMRES),
with unchanged residual/current-error thresholds.

This removes saved forward action graphs, not backward execution workspace or
the retained operator. Complete setup/action/solver/backward/host resource
admission, material/geometry derivatives and final physical/complexity
qualification remain unclosed. No full regression or independent audit is
repeated for this increment; both remain scheduled at phase/final acceptance.
