# Native GPU FDFD local-response development plan

Status: implementation and declared-scope qualification completed on 2026-09-16.
See [acceptance evidence](../../assessments/fdfd-local-response-gpu.md) for results and limits.

Requested on 2026-09-16. Scope: implement precomputed local dielectric responses,
small observation projections, low-rank candidate evaluation/acceptance, native
PyTorch material gradients, and the FDFD capabilities needed to cover the published
PNGF implementation, including nonuniform Yee grids. Each stage is committed
separately; final qualification is independent of implementation unit checks.

## References and implementation policy

- Paper: https://arxiv.org/html/2501.18054v4
- Published paper: https://doi.org/10.1038/s41467-026-69477-y
- Reference implementation: https://github.com/ACME-Lab-Stanford/PNGF
- Reviewed reference revision: a3864e2841919cad9fcea1dc5478779111ff9de1
- Starting Maxwell revision: 8c0a8a3

Implement the published mathematics independently. Do not copy AGPL implementation
code into this dual-licensed package. Core assembly, projection, factorization,
updates and gradients must execute on CUDA; no CPU numerical fallback. Preserve
Scene -> Simulation -> Result and the node-to-Yee material differentiation graph.

## Capability and acceptance matrix

A functional superset means covering the reference's published executable workflow,
not merely exposing a Green matrix. Qualification must cover:

| Capability | Required behavior | Evidence |
| --- | --- | --- |
| Cartesian Yee grids | Uniform, unequal constant axis steps, custom/automatic nonuniform axes | Uniform regression and independent nonuniform curl oracle |
| Background | Dielectric/lossy materials, fixed PEC and absorbing boundaries | Direct physical residual and PEC constraints |
| Local design | Finite complex dielectric contrast and binary PEC constraints | Random designs versus full reassembly |
| Precomputation | GPU background factor/solve reuse, bounded RHS batches | Full residual, storage and device checks |
| Observations | Small linear electric/magnetic field projections, port voltage/current and far field | Direct full-field projection comparison |
| Updates | Trial without mutation, accepted low-rank update, stale-trial rejection, stable rebuild | Long randomized accept/reject sequence |
| Gradients | PyTorch real/complex material parameters through scene compilation | Independent full-system autograd and finite differences |
| Optimization | Multi-frequency local binary search and objective aggregation | Reproducible end-to-end design run |
| Output | Standard Result, optional full field recovery, metadata | Public API tests |
| Caching | Immutable operator context; invalidation/rejection on changed dependencies | Mutation and incompatible-scene tests |
| Numerical safety | Residual checks, finite values, singular-update failures | Failure-injection tests |
| Performance | Cold and warm timings, accepted/rejected update cost, peak GPU memory | Synchronized CUDA benchmark; no extrapolated speedup claim |

APF is an optional precomputation algorithm, not a prerequisite for equivalent
responses: GPU factorization with blocked selected inverse actions is the initial
implementation. Report this algorithm difference explicitly. Do not label an
unverified capability a superset. Broader nonlinear/transient/SIE physics is not
implied by the linear FDFD reference workflow.

## Stages and commit boundaries

0. Record scope, acceptance matrix, environmental limitations and implementation sequence.
1. Extend the primary FDFD operator to metric-aware nonuniform Yee discretization,
   preserve reciprocal similarity scaling, add exact fixed PEC constraints and
   reusable physical-space multi-RHS solves. Validate against independent operators.
2. Add public local-response configuration, GPU blocked precomputation, observation
   projections and Result output, with strict support and cache validation.
3. Add sparse low-rank trial/accept updates, finite-contrast and PEC designs,
   candidate lifecycle, residual certification and recovery.
4. Integrate native PyTorch material gradients with scene compilation and add
   multi-frequency/optimization workflows and reference observation/excitation parity.
5. Independently audit, compare with full reassembly and reference mathematics,
   run regression and GPU performance qualification, repair findings, and record
   a scored capability matrix with explicit pass/fail/blocked statuses.

Update FEATURE_LIST.md with each user-facing stage. Commit only task changes;
use the configured human git identity. Never claim skipped GPU tests as passing.

## Numerical design

For A = A0 + B Delta B^T, precompute G = B^T A0^-1 B and
H = L A0^-1 B. Solve (I + G Delta) u = B^T A0^-1 b and
observe y = L A0^-1 b - H Delta u. Delta is assembled from actual
compiled Yee edge coefficients, including averaging support, rather than a
geometric voxel count. Account explicitly for S = D A D^-1:
G = B^T D^-1 S^-1 D B. PEC constraints use exact elimination/current
constraints rather than large artificial permittivities.

Keep only G, H and background responses; do not retain full-domain A0^-1 B.
Full-field recovery requires an additional background solve when requested.
Small low-rank systems use solves and checked factorization. Cache accepted state
separately from trials; retain enough state to rebuild after accumulated error.
Autograd differentiates the reduced solve and the material compiler, not a history
of in-place Woodbury updates. Reject unsupported trainable background/source/grid
parameters rather than silently detaching them.

## Execution record

- Stage 0: recorded plan. Working tree was clean.
- Local GPU preflight: nvidia-smi cannot communicate with the NVIDIA driver;
  previous torch CUDA availability check was false. GPU execution/benchmark
  qualification is blocked until a working CUDA environment is supplied.
- User asked for GPU environment details asynchronously; implementation continues.

- GPU preflight resolved: CUDA works outside the sandbox; two RTX A6000 cards. Tests use physical GPU 1 via CUDA_VISIBLE_DEVICES=1.
- Stage 1: nonuniform manufactured curl-curl, reciprocal scaling, exact PEC and owned multi-RHS storage checks passed (3 GPU tests). Uniform regressions in progress.
- Stage 1 regression: 10 existing GPU cache/direct/PML tests passed. Grid/material
  sweep: 54 passed; 5 unrelated FDTD tests failed because nvcc/CUDA_HOME was not
  configured, not because CUDA devices were inaccessible. Those failures remain
  tracked for final environment setup and regression.
- Stage 2: CUDA selected inverse actions, small electric projections, material
  contrast from Scene, optional full-field recovery and context guards implemented.
  Two independent GPU comparisons passed: fresh full Scene reassembly and direct
  dense solve of a perturbed full operator. Fixed the cuDSS RHS descriptor issue
  with padded/split blocks so remainder blocks and source solves reuse factors.
- Stage 3: unified finite-contrast/PEC current equations and sparse row updates
  implemented independently. Fifteen randomized candidate updates with accepted
  and rejected steps matched direct reduced solves; periodic rebuild and stale
  trial checks passed. A mixed PEC/dielectric candidate matched a separately
  eliminated full GPU operator. Four local-response GPU tests passed.
- Stage 4: native complex material autograd, implicit full-field background adjoint,
  multifrequency transactions, gap voltage/current, magnetic/far-field projections
  and binary tile search implemented. A three-frequency GPU 1 antenna smoke run
  (112 local unknowns, 9 trials, 4 accepted) reduced the objective from 285.59 to
  164.13 in one pass; this is not a converged or published-design reproduction.
  Uniform source cutoff rounding was diagnosed against the pre-change solver;
  regression validation and independent qualification continue.
- Stage 4 qualification: 28 targeted GPU tests passed, including existing direct,
  symmetric-PML and material-adjoint regressions. Broader grid/material/public
  checks passed 137 cases; a new singular-candidate boundary check exposed a
  one-ulp threshold issue, fixed with a roundoff margin and all 8 local-response
  tests rerun successfully. Three additional native GPU mode tests passed against
  discrete rectangular-waveguide eigenvalues and graded dielectric convergence.
- Stage 5 independent reference: 18 randomized accepted/rejected rank 1/4/12
  updates matched freshly modified and refactorized full sparse systems. The
  compact planar case measured 4.01 ms median trial versus 348 ms full refactor,
  with 1.64 s separate preparation. Largest field/projection relative errors were
  below 6.2e-12; the independent full-solve gradient difference was 8.3e-12.
  These are case-specific GPU measurements, not timings of the upstream CPU code.
- Broad GPU regression: 184 tests passed. Independent review additionally fixed
  projection frequency slicing/persistence, sweep Scene metadata, delayed-adjoint
  factor release, full-solve vacuum-surface guards, and double-output downcasts.
  Targeted checks cover those corrections before final measurement and report.
- Source-level coverage audit found public diagonal permeability arrays in the
  reference backend, beyond its nonmagnetic executable default. Added a native
  CUDA factorized curl operator for fixed diagonal magnetic backgrounds and H
  projections, with checked fixed-permeability gradient/update scope. Independent
  constant-mu, variable-mu manufactured, reciprocal/PML and full-local comparisons
  pass. During magnetic-loss qualification, repaired FDFD-only conductivity/phase
  convention handling without reversing the shared pole susceptibilities.
- Final extended regression: 197 GPU tests passed in 120.98 s, including all FDFD
  tests, FDFD gradients, custom/automatic grids, material compiler and public API
  suites, plus the nonreciprocal magnetic rejection check. Tests used the maxwell
  conda environment and physical GPU 1. The final measured source revision is
  committed before benchmark reruns so raw reports can identify it exactly.
- Added native persisted response caches to cover separate precomputation and
  optimizer processes. Operator/source/projection/material fingerprints and
  checksums prevent stale or corrupt cache reuse; restored projected workflows
  skip factorization and retain native material gradients. Single and multifrequency
  persistence tests and a separate-process antenna replay qualify this addition.
- Cache/material integration: 12 local-response tests passed, including exact
  restored projections/gradients with sparse factorization forbidden, corruption
  and incompatible-operator rejection, lazy full-field factorization, frequency
  sweep persistence, and a trainable SceneModule as the reference. Preparation
  freezes its reference operator under no_grad; run recompiles the material graph.
- Final source regression including native cache persistence and trainable reference
  preparation: 200 GPU tests passed in 128.38 s. Independent measurements and
  separate-process replay are rerun against the committed cache-capable revision.
- Final dispatch review: physical-space multi-RHS solves now honor every configured
  Krylov engine. Nine graded/RHS tests passed in 12.34 s, including six additional
  engine cases against an independent dense CUDA solve. This correction does not
  change the direct-solve path used by the recorded performance measurements.
- Completed qualification: the scoped capability matrix passes 13/13 gates.
  Fixed-revision independent planar/volume runs use 23,898 full unknowns and
  112/532 local unknowns. Warm trial medians are 4.091/4.103 ms, versus
  346.899/355.781 ms full refactorization; cold preparation is 1.650/5.069 s.
  Including preparation and accepted updates over 18 trials yields timed-scope
  ratios of 3.42x/1.22x, rather than the 84.79x/86.70x warm-trial ratios.
  The three-frequency antenna converged to a one-tile local optimum after
  18 trials; a separate cache-loading process reproduced its full objective
  history and final pattern exactly. Original CPU/APF and published-device scale
  reproduction remain explicitly unqualified. Full evidence is in
  docs/assessments/fdfd-local-response-gpu.md.
