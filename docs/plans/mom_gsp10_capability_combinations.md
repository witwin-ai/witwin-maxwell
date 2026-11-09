# GSP-10 capability combinations and public integration

Status: OPEN. Single-path implementation or algebraic agreement does not qualify
an untested material/kernel/backend combination. Original physical, resource,
complexity and final cross-repository requirements remain unchanged.

## Free-space heterogeneous magnetic multibody gate

`tests/mom/test_magnetic_backend_combinations.py` constructs one public Scene
with two disjoint radius0.01m coarse spheres centered at x=-0.5m/+0.5m. Materials
have eps_r2/3, mu_r1.3/1.7, sigma_e0.001/0.002 and sigma_m0.002/0.003, at300MHz.
One original dense solve is shared by the H, nested H2, direct H2 and compact
Cartesian FMM variants. Integration is order4/singular4. Iterative variants use
rtol1e-12, max240/restart120 and their supported balancing/preconditioning.
FMM uses order6. Every public accelerated run forbids dense system assembly.
H must retain compressed far blocks; H2 and direct/FMM must retain actual far
couplings. Near-only degeneration cannot count as this combination gate.

Each variant checks numerical admission, separately compares J and M on each
body with relative tolerance1e-8, and checks the complete original dense action
residual after impedance balancing at1e-8. Total E/H are compared in exterior
and both interiors. Exterior scattered E/H have separate1e-8 comparisons to
avoid masking by incident fields. Result save/load must preserve currents
exactly and retain physical_qualification=False.

Evidence history:
- `/tmp/gsp10-magnetic-combinations-v1.log`:4failed20.22s at field queries because
  the test used exterior region0 for a point inside the first body. Earlier
  current/residual gates passed; the wrong fixture is not a solver qualification.
- `/tmp/gsp10-magnetic-combinations-v2.log`:4passed20.17s after selecting named
  body regions. Review identified missing explicit H far-block and exterior
  scattered-field assertions; v3 adds these checks without changing tolerances.

This is a single-frequency, single-source, coarse-mesh numerical consistency
fixture against the shared original dense implementation. It is not an
independent magnetic physical reference, exhaustive operator error bound,
mesh-convergence, material-gradient or production qualification.

## Remaining combination gates

- Multi-body reciprocal anisotropic dense/H2, with independent tensor references.
- Periodic magnetic multibody: integrate existing periodic controls into the
  final combination gate without rerunning unchanged evidence unnecessarily.
- Periodic reciprocal anisotropy and valid nonreciprocal guards.
- Coupled nonlinear SIE-VIE with fast surface/volume actions and multiple bodies.
- Strict snapshots across formulations, kernels, lattice, object/material tables
  and construction identities; Bench exact-state consumption and physical split
  identity separation from numerical backend choices.
- Major-stage full regression, cross-repository acceptance and independent audit.

GSP-01 physical failures and GSP-09 high-frequency/global-error/resource/scaling
obligations remain OPEN; this stage does not replace their requirements.

Final targeted evidence `/tmp/gsp10-magnetic-combinations-v3.log`:
4passed20.22s. H/H2 each require84iterations; direct-H2/FMM each90. Maximum
per-body current relative difference is5.453645284013943e-9 (H2, body0 M).
Full original balanced residuals are1.4242917048630625e-10 (H),
1.4242969073231368e-10 (H2),1.958611201744211e-13 (direct-H2), and
1.948032285746456e-13 (FMM). Exterior scattered E/H maximum relative difference
is2.6544924353625055e-10. All gates remain1e-8. Independent review found noP1/P2
and identified the two coverage gaps now exercised by v3. No full regression
was repeated; remaining GSP-10 combinations and all production gates stay open.

## Selected tensor blocks for hierarchical integration

`anisotropic_operator.tensor_regional_block` supplies a missing prerequisite for
tensor H/H2: selected complete RWG supports in input order, rectangular J/M
blocks, bounded pair classification and quadrature batches. It does not assemble
a full regional matrix or allocate the global panel-pair classification. Fixed
material and current geometry checks precede integration. Isotropic tensor
samples use the original selected scalar path exactly. General tensors retain
regular dyadic testing plus a single weak static-divergence restoration.
`_integrated_dyadic_pairs` shares the original adaptive near algorithm between
dense and selected assembly; limits and nonconvergence rejection are unchanged.
Output-block storage, complete geometry and spectral-kernel caches are separate
costs from the bounded pair workspaces. No public solver integration or peak
memory certificate is implied.

`/tmp/gsp10-tensor-selected-v1.log`:3passed4.10s. Controls cover unsorted selections,
rectangular cross-body and self blocks, fixed noncommuting constitutive tensors,
coefficient gradients, changed geometry/duplicate indices, forbidden full dense
assembly/classification, and exact isotropic scalar reduction. The general tensor
regular kernel is explicitly manufactured; this is assembly algebra evidence,
not an independent physical tensor Green reference. Independent source review
found noP1/P2 and requested direct adaptive-near coverage.

Near evidence: `/tmp/gsp10-tensor-selected-near-v1.log` FAILED a2e-12 relative
selected/dense comparison with pair_batch2 and near_tolerance1e-5. Maximum
relative difference1.8612733971403448e-7 (absolute8.553624596838679e-10) arises
because the original tensor near algorithm advances every pair in a batch until
all converge. Selecting fewer supports changes batch partners and can change
final orders. No exact equality across different groupings is claimed; this
failure remains recorded. The test also originally proposed an invalid zero
refinement allowance, but execution did not reach that branch.
`/tmp/gsp10-tensor-selected-near-v2.log`:1passed5.47s with pair_batch1, giving
identical adaptive groups for the strict assembly comparison. It confirms
category3 is actually visited and refined, and that both selected and dense
paths reject a valid deliberately unresolved near_max_order4/tolerance1e-30
configuration. No production threshold or integration algorithm was relaxed.
General near accuracy, actual tensor Green references, public H/H2 composition,
full operator error and all original physical/resource gates remain open.

## Generalized selected-system provider

`tensor_nearfield.GeneralizedSystemBlocks` composes scalar and tensor selected
regional blocks. Global unsorted selections retain all-J/all-M order; same-body
blocks include their own interior once, cross-body blocks contain only the common
exterior. Original sampled actions stream every source block and support native
current gradients. Geometry routing reuses the scalar provider through a regional
block hook; this is not a ScalarSystemBlocks subtype and cannot accidentally
enter scalar FMM type guards. Material samples, regional kernel/config/cache
identities, geometry and routing arrays remain guarded. Tensor material arrays
are included in the provider input-storage inventory. Spectral cache storage is
explicitly excluded from H/H2 owned action reports and remains under separate
kernel budgets; complete peak/resource qualification is still open.

`/tmp/gsp10-generalized-blocks-v1.log`:6passed8.07s, comprising three new mixed
provider/ownership checks and three affected scalar controls. Scalar and tensor
exterior cases both compare selected blocks and complete-source sampled actions
with original dense assembly, including gradients and interleaved body indices.
Kernel ownership, bool region aliases and material/kernel mutation reject.
General tensor regular kernels are manufactured for this assembly test; they
are not physical Green references. Independent review found noP1/P2.

First H/H2 integration attempt `/tmp/gsp10-generalized-h2-v1.log` FAILED because
the coarse tensor-tetrahedron/scalar-sphere fixture retained zero compressed far
blocks. It cannot count as a compressed combination pass. A successor uses two
30-edge spheres and one near panel per batch to preserve identical selected/dense
adaptive grouping. Thresholds and the requirement for actual far compression are
unchanged. These internal actions still do not constitute public tensor runtime,
incident-carrier, field-decoder or exact-storage integration.

`/tmp/gsp10-generalized-h2-v2.log`:1passed81.39s, two H far blocks and zero
compression rejections; nested H2 also contains actual couplings. Both actions
and adjoints match complete dense assembly at1e-8 for each output family
separately; native current gradients match adjoints at1e-11. Constitutive
mutation rejects both H and H2 actions. One test-only scalar-conversion warning
was removed by detaching the measured norm before conversion, without changing
numerical assertions. No public tensor solve or real anisotropic regular Green
qualification is claimed. Together with the six provider/scalar controls this
completes the current internal provider increment; no full regression rerun.

## Native material sampling and exact sampled identity

The active branch now includes `compile_tensor_medium`, `sampled_medium_payload`,
`restore_tensor_medium` and `sampled_tensor_identity` from the inspected GSP-05
work, adapted to declarations actually present on this branch. The source WIP
worktree is untouched. Existing public diagonal/full reciprocal electric tensor,
scalar and ferrite declarations compile on CUDA. Tensor declarations replace
corresponding scalar bases; scalar dispersion increments add isotropically;
electric conductivity tensors replace scalar conductivity. Full ferrite samples
retain off-diagonal gyrotropy and explicit nonreciprocity. Complex arbitrary
public tensor declarations and public runtime dispatch are not added by this
increment and remain part of the larger integration work.

The public Polder function previously built CPU constants and lacked leading
frequency dimensions. It now chooses the input accelerator, rejects conflicting
accelerators, creates constants there, and broadcasts scalar responses into
[...,3,3] tensors. Existing CPU-only calls continue their original behavior;
CUDA sampling introduces no numerical CPU fallback. This retains native frequency
and parameter autograd, without claiming MoM material derivatives.

Sample payloads bind schema, convention, frequency, precision, reciprocity and
exact real/imaginary tensor values. Restoration rejects malformed values, silent
precision rounding, gain and false reciprocal declarations and revalidates the
sample on CUDA. Serialization alone transfers metadata to the host. Scalar
sample records retain their existing shape.

`/tmp/gsp10-tensor-material-v1.log`:38passed3.40s, including affected public
ferrite material tests. New controls verify tensor/scalar override ownership,
electric/magnetic poles, conductivity, ferrite bias reversal, batched CUDA
frequency gradients, discarded scalar branches, CPU/active-conductivity rejection
and exact complex64/complex128 sample round trips with corruption rejection.
This establishes material compilation and serialization, not tensor scattering,
field accuracy, public Scene/Simulation/Result integration or production status.
Independent source review reports noP1/P2 for this material increment; the
38-test result is separate executed evidence, not inferred from the review.

## Public tensor-interior dense/H/H2 integration

The explicit configuration
`MoMConfig(dyadic_integration=DyadicIntegrationConfig(...), operator_backend=...)`
enables the generalized interior path for dense/H/H2. Existing public reciprocal
tensor and ferrite declarations compile as TensorMedium; scalar bodies can share
the same system. The exterior remains the current scalar vacuum and incident
carriers retain analytic PlaneWave semantics. Periodic, direct-H2, FMM and HB
combinations reject this configuration before construction. They remain required
future integrations rather than silently reusing scalar kernels.

Public preparation connects generalized dense assembly or selected H/H2 providers,
tensor contrast-factored RHS and complete tensor interior decoding. Exterior
fields retain the shared scalar operator. Explicit volume absorption uses the
anti-Hermitian constitutive quadratic form, so lossless gyrotropy is not counted
as absorption. This does not enable MoM material/geometry derivatives.

`dyadic_cache_max_bytes` is a retained cache allowance **per tensor body**:
half for the regional kernel shared by construction and decoding, the remainder
for the paired contrast RHS kernel. It is not an all-body/allocator/solver/peak
budget. Both kernels are precreated with bound media, integration, cache identities
and material versions. The scalar legacy path retains its existing numerical
identity when this optional configuration is absent.

Schema13 records typed sampled tensors, dyadic settings/cache budgets, backend,
analytic carrier, physical scattered coefficients, body routing and the complete
operator/RHS identities. Decode-only restoration rechecks Scene material/geometry
and configuration, requires physical_qualification=False and protects both native
and outer Result metadata. It does not reconstruct H/H2 or assemble a dense
matrix. Previous schemas keep their existing explicit meanings; they are not
silently relabeled as tensor solutions.

`/tmp/gsp10-tensor-public-v1.log`:5passed13.28s, but independent review then found
two P2 gaps not exercised by that first suite: lazy contrast-kernel replacement
could evade numerical binding, and dense tensor restore lacked operator/RHS and
qualification checks plus the outer metadata guard. Both were fixed before
commit: precreate/bind the contrast kernel; apply the complete sampled identity
and qualification checks to dense tensor snapshots and outer Result metadata.
The regional construction/decoder kernel is now shared. Follow-up source review
closed both P2s and found no new P1/P2.

`/tmp/gsp10-tensor-public-v2.log`:22passed55.33s, including new tensor controls
and affected existing scalar multibody, H/H2/direct-H2/FMM and periodic-FMM public
paths. New controls cover two-body tensor scalar-limit current/field comparisons,
actual H/H2 far blocks, exact save/load without assembly, budget/material/operator/
RHS/qualification tampering, changed contrast/decoder kernels, unavailable
combination rejection and a genuine anisotropic Green convergence failure that
must propagate instead of falling back to scalar physics.
`/tmp/gsp10-tensor-loss-v1.log`:1passed3.33s, independently compares tensor volume
loss and native field gradients to a factored positive loss quadratic form,
including lossless gyrotropic permeability.

No general anisotropic scattering accuracy, high-frequency, tensor-exterior H/H2,
complete resource or production qualification follows from these controls. The
existing general tensor physical/RHS failures remain open. No full regression
was repeated for this integration increment; final cross-repository gates remain.

## Registered real anisotropic public control

`scripts/dev/solve_mom_tensor_public_control.py`
registers a four-face/six-edge tetrahedron at300MHz with reciprocal electric
tensor [[2,.1,0],[.1,2.4,0],[0,0,3]], isotropic imaginary increment .8 supplied
through conductivity, scalar permeability and a plane wave. Dense public prepare/
run uses surface/singular order1, pair_batch4, spectral initial8/8 and maxima512/64,
rtol1e-4/atol1e-9. The per-body retained spectral allowance is1GiB; this is not a
peak-memory cap. Admission requires solver convergence, the complete original
impedance-balanced algebraic residual <1e-8, cache-budget compliance and bitwise
current/exterior-scattered-field restoration. Raw matrix/RHS and source hashes
are retained, including failure evidence. No interior-field/physical-reference/
mesh-convergence or production claim is permitted.

Historical actual-tensor runtime success is not reused as current-head evidence.
Independent runner review found inherited cache-get double wrapping (P2); before
registration it was corrected by wrapping only the base class once, labeling by
the actual instance type and restoring the original base method. CLI import was
checked with PYTHONPATH=.; no full regression was repeated. The actual measurement
is pending at registration.

### Actual anisotropic control result at7781144

/tmp/gsp10-public-anisotropic-control-v1/receipt.json is terminal measured:
352.894028s total, complete original balanced residual2.842608074169925e-15,
solver converged, exact current/exterior-scattered-field restoration. Spectral
cache retention699376704bytes fits1GiB; allocated peak823247360bytes and reserved
peak1000341504bytes are observations, not capacity qualification. Source hashes
matched. result.pt and diagnostics.pt preserve currents, original matrix/RHS and
regional/contrast receipts. Contrast210misses/196evictions/989184modal matrices;
regional60misses. Exec10170/PID1406451 is terminal; do not restart or poll it.
Short material CUDA tests ran concurrently near the end; this elapsed time must
not support a speedup or isolated resource comparison against historical runs.
The successful public path now includes a genuinely anisotropic electric tensor,
but remains one coarse algebra/storage control. Independent physical references,
interior decoding accuracy, mesh/spectral/global convergence and production gates
remain open; all historical failures are retained.

## Full complex constitutive declarations

`ComplexTensor3x3(rows, reciprocal=True)` stores finite complex3x3 constants with
explicit transpose-reciprocity semantics. Public Material accepts epsilon/mu;
real reciprocal full Tensor3x3 mu is also accepted (symmetric positive definite).
Static and scalar-frequency evaluators explicitly require a tensor-aware spectral
compiler for these declarations, so the existing Yee compiler cannot silently
drop imaginary or off-diagonal components. Conductivity keeps its real existing
contract; complex conductivity descriptors are rejected. Constant complex tensors
are frequency-domain declarations, not a causal time-domain dispersion model.

CUDA sampling separately enforces each declared reciprocal tensor, even if the
other tensor is nonreciprocal; the complete sampled response retains the existing
passive anti-Hermitian check. Exact sample payload and public schema13 are reused.
No MoM material-gradient capability is introduced by these frozen declarations.

/tmp/gsp10-complex-declaration-cpu-v1.log preserved a collection-only failure
from an incorrect test import. Corrected v2:8passed2.38s, finite/shape/boolean
checks and actual CPU scene-compiler rejection; affected static mu guards now
reject at sampling rather than construction. CUDA material suite:
/tmp/gsp10-complex-declaration-cuda-v1.log9passed3.89s, including new full complex
epsilon/gyrotropic mu preservation, full real mu, exact storage, independently
asserted reciprocity and indefinite-loss rejection. Public control:
/tmp/gsp10-complex-public-v1.log1passed4.65s, lossy complex scalar-limit current
comparison against conductivity declarations and exact Result restoration.
Independent read-only review found no P1/P2. No full regression repeated.
General anisotropic/nonreciprocal physical references and all final gates remain
open; the earlier7781144 anisotropic run remains bound to its original source.
