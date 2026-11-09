# MoM large-scale scattering with learned local operators (L line, L-1 design)

Status: ACTIVE design, branch `sie/large-scale-l1` (from master `c90c8346`).
Scope: multi-body and sharp-edged scalar PMCHWT scenes from 4λ to 64λ with
sparse field queries; the classical L-1 baseline; and the solver-side hooks a
learned preconditioner (L-2), local response or warm start (L-3) and sparse
query accelerator (L-4) need. Single dielectric sphere scaling against Mie
(L-0) is measured in a separate session and is not repeated here.

All file references are relative to `witwin/maxwell/mom/` unless noted.
Capacity numbers below are order-of-magnitude estimates from the code's own
capacity models and from retained measurements in
`docs/plans/mom_gsp09_fast_multipole.md`; they are not measured qualification.

## 0. Summary

- Every iterative backend (`hmatrix`, `h2`, `direct_h2`, Cartesian `fmm`,
  directional `fmm`) accepts several disjoint scalar dielectric bodies,
  explicit solid `Mesh` bodies and, once the polyhedral branch lands, faceted
  `Box/Cylinder/Cone/Prism` bodies, because geometry enters only through
  `SurfaceSystem` and `ScalarSystemBlocks`.
- None of them is shown, or predicted by its capacity model, to fit a 16λ
  dielectric scene on one 48 GB GPU. Dense CUDA is the only backend that runs
  reliably; it is memory-feasible up to about 4–5λ (N ≈ 5·10^4), but its
  O(N²) assembly already takes minutes at N ≈ 4·10³ and hours at N ≈ 3·10⁴
  on a shared GPU. 32λ and 64λ need a
  high-frequency fast operator that does not exist in a qualified form, plus
  O(N log N) mesh/separation validation and an accelerated field query.
- The Krylov path is restarted GMRES with a fixed diagonal left/right
  scaling, block GMRES, recycling and (FMM only) a fixed LU leaf right inverse.
  There was no flexible (variable) right preconditioner. Stage 2 adds one as an
  additive `MoMConfig.right_preconditioner` hook with flexible right GMRES;
  the default path is unchanged.
- The L-1 baseline therefore has two tiers: a dense tier (2–5λ, exact operator,
  LU reference available) that measures preconditioners and multi-body
  conditioning now, and a fast-operator tier (8–64λ) whose commands are
  prepared but which is gated on backend capacity (Section 5).

## 1. Fast backends for multi-body scalar PMCHWT (question a)

### 1.1 What is admitted today

Dispatch is in `runtime.py` `MoMRuntime.__init__`; admission is in
`MoMConfig.__post_init__`. Bodies are compiled one per enabled `Structure` by
`compile_surfaces` → `_compile_basis` (Sphere, Ellipsoid, solid `Mesh`; the
polyhedral branch extends this function with `FACETED_PRIMITIVES` and a
`SurfaceMeshConfig`). `SurfaceSystem.create` (`mesh.py`) concatenates bodies in
Scene order, J block then M block, and `ScalarSystemBlocks` (`nearfield.py`)
evaluates any selected original block across bodies. No backend inspects the
primitive type, so a faceted or explicit mesh body is supported wherever a
sphere is, subject to the same quadrature (sharp-edge accuracy is the S-1b/S-1c
concern, not a backend one).

| Backend (`operator_backend`) | Runtime class (`hierarchical_runtime.py`) | Multi-body scalar | Accuracy / order controls | Setup | Retained memory model |
| --- | --- | --- | --- | --- | --- |
| `dense` | `SystemOperator` (`runtime.py`) | yes | `IntegrationConfig` only (exact Galerkin matrix) | full assembly O(N²) quadrature, `assemble_system` (`operator.py`) | 16 N² B matrix (+ compact Riesz); `_preflight_dense_capacity` rejects before assembly |
| `hmatrix` | `HierarchicalSystemOperator` → `HierarchicalOperator` (`hmatrix.py`) | yes, one global support octree | `HMatrixConfig`: `leaf_size`, `max_rank` (32), ACA `tolerance`, `admissibility`; rejected far blocks are split | Python traversal of every block, streamed ACA sampling of original rows/columns (`lowrank.py`) | exact leaves (packed CSR) + rank-r factors; `retained_budget_bytes` guards owned bytes only |
| `h2` | `HierarchicalH2SystemOperator` (`h2_operator.py`, `h2.py`) | yes | `H2Config.max_rank` (64), `tolerance`; inherits the H setup | H setup plus algebraic H→H2 conversion | `h2_capacity.plan_h2_storage` structural upper bound |
| `direct_h2` | `DirectH2SystemOperator` (`direct_operator.py`, `direct_system.py`) | yes, exterior tree + per-body interior trees | `DirectH2Config`: interpolation `order` or automatic `max_order` with `KernelErrorBudget`, `rwg_absolute_budget` | direct nested interpolation, no ACA | `direct_capacity.plan_direct_storage`: order³ nodes per box, 4 kernels per far pair |
| `fmm` + `FMMConfig` | `MultipoleSystemOperator` (`fmm_operator.py`, `fmm_system.py`) | yes, global exterior tree + interior trees | Cartesian Taylor `order` (0–12 for compact/FFT tables) or automatic `max_order` + `kernel_error_budget` + `rwg_absolute_budget` (`fmm_order.py`); `kernel_bound_rule`; `separation_ratio`; `leaf_size` | order admission (`fmm_order.plan_multipole_order`), RWG moments, M2L tables, original near blocks | `fmm_capacity.plan_multipole_storage`: leaf moments 4·N·C(p+3,3)·16 B, M2L per pair or shared catalogue, near 32–64 B per near edge pair |
| `fmm` + `DirectionalFMMConfig` | `DirectionalSystemOperator` (`fmm_system.DirectionalSystemActions`) | yes (exterior plus per-body interiors) | `kernel_error_budget`, `global_ideal_rwg_error_budget`, orders 0–12 per cone, `cone_subdivisions`, `separation_ratio` | automatic cone/order planning with hard caps `max_candidates=1e4`, `max_contexts=1e5`, `max_pairs=1e5` | regional budgets (`regional_*_budget_bytes`); reciprocal/streamed/cached near (`fmm_capacity.plan_directional_reciprocal_near_storage`) |

All hierarchical backends require `riesz_storage='local_rwg_v1'`, iterative
stopping and no direct fallback. Their `solve` (`HierarchicalOperator.solve`,
shared by `DirectHierarchicalOperator.solve`) admits a solution only when both
the compressed dual residual (full vector, compressed operator, `LocalRWGRiesz`
metric) and a sampled original-row residual (default 8 rows,
`residual_report`) meet `solve_dual_tolerance` / `original_sample_tolerance`.
That sampled check is not a global original-operator certificate
(`original_global_residual_certified: False`).

Exterior excitation: a plane wave (any backend) or a `PointDipole`, which is
admitted only on the dense free-space linear path (`_point_dipole_scope`).

### 1.2 Evidence and capacity at large electrical size

Unknown counts. For a sphere of diameter Kλ with mean edge h, faces
F ≈ πK²λ²/(0.433 h²), edges E ≈ 1.5F, unknowns N = 2E. With exterior ppw 10
(h = λ/10): N ≈ 2.2·10³ K². A dielectric with εr = 4 resolved at ppw 10 of the
interior wavelength needs h = λ/20 and four times that.

| Size | N, h = λ/10 | N, h = λ/20 (εr = 4) | Dense matrix 16 N² |
| --- | --- | --- | --- |
| 4λ | 3.5·10⁴ | 1.4·10⁵ | 20 GB / 0.3 TB |
| 8λ | 1.4·10⁵ | 5.6·10⁵ | 0.3 TB |
| 16λ | 5.6·10⁵ | 2.2·10⁶ | 5 TB |
| 32λ | 2.2·10⁶ | 8.9·10⁶ | – |
| 64λ | 8.9·10⁶ | 3.6·10⁷ | – |

A multi-body scene of total extent Kλ has fewer unknowns than the
circumscribed sphere, roughly in proportion to its surface area.

Per-edge retained cost of the existing fast backends, from measured records in
`mom_gsp09_fast_multipole.md`:

- Cartesian FMM, two small spheres, 3840 edges, p11/p12 (sections "Original
  shared compact fixed-ka/growing-ka full GPU resource measurement"): owned
  0.59–0.67 GB, i.e. 150–175 kB per edge, of which near ≈ 68 kB/edge; whole
  allocator peak 0.75–0.85 GB, just below the same-N dense matrix (0.94 GB).
  Setup 1.7·10³ s and ~400 s per forward action on a shared GPU in that
  measurement workflow.
- Cartesian FMM, resonant sphere f30, 27 000 edges ("Original resonant f30
  compressed capacity preflight"): minimum p6 requires 14.8 GB (M2L 11.5 GB,
  near 3.1 GB), i.e. ≈ 550 kB per edge, for a body that is electrically small
  outside (ka ≈ 0.9) and moderate inside (εr = 12).
- Directional FMM: public evidence only at 90 physical edges
  ("Public automatic scalar directional DAG integration completed").
- H/H2: no retained measurement above a few thousand edges; ranks are capped
  at 32 (H) / 64 (H2), and rejected far blocks are split toward exact leaves.

Order-of-magnitude feasibility on one 48 GB GPU (owned storage only; solver
vectors, Riesz metric, setup workspace and allocator slack are extra):

| Backend | 4λ | 8λ | 16λ | 32λ | 64λ | Limiting mechanism |
| --- | --- | --- | --- | --- | --- | --- |
| dense | yes (≤ ~5λ at h = λ/10) | no | no | no | no | 16 N² |
| Cartesian FMM | marginal | no | no | no | no | near ≥ 70 kB/edge alone gives 19 GB at 16λ, 78 GB at 32λ, 0.3 TB at 64λ (h = λ/10); fixed Taylor order ≤ 12 cannot make wavelength-sized boxes admissible, so far work collapses toward fine levels (high-frequency breakdown) |
| H / H2 | marginal | unlikely | no | no | no | ACA rank of well-separated blocks grows with electrical size; caps 32/64 force splitting; Python block traversal setup |
| direct H2 | marginal | unlikely | no | no | no | order³ interpolation nodes per box; same high-frequency rank growth |
| directional FMM | unknown | unknown | unknown | no without new caps | no without new caps | the only high-frequency design; planner caps (1e5 contexts/pairs) are hit first at leaf_size 8; near storage still ≥ tens of kB/edge unless streamed (`scalar_reciprocal_streamed_dk_v1`, which recomputes near blocks every action) |

Independent of the operator, three preparation steps are quadratic and become
blockers before any operator exists at 16λ+ multi-body:

1. `SurfaceSystem._validate_separation` (`mesh.py`) tests all face pairs of
   every body pair in Python batches of 4096 pairs: ~10¹¹ pairs for two
   10⁶-face bodies.
2. `validate_mesh` coincident-vertex (128-row batches over V²) and
   `_has_intersections` (128-row bounding-box batches over F²) are GPU O(F²).
3. `RWGBasis.__init__` builds edges with host Python loops (linear but slow at
   10⁶–10⁷ edges).

These need octree/sort-based replacements (owner: MoM runtime, separate work
package, outside the polyhedral branch's geometry dispatch).

Conclusion for (a): the L-1 baseline can run multi-body and sharp scenes now
only on the dense tier (≤ 4–5λ). The fast tier needs (i) the L-0 data to place
the measured crossover, (ii) a capacity-qualified high-frequency operator
(directional FMM is the candidate), (iii) streamed or compressed near storage
below ~10 kB/edge, and (iv) O(N log N) validation. The commands in Section 5
are written so they run unchanged once a backend admits the size; until then
they fail closed at the backend's own capacity preflight.

## 2. Krylov path (question b)

| Piece | Location | Notes |
| --- | --- | --- |
| `StoppingRule` | `linalg.py` | `rtol`, `atol`, `maxiter`, `restart`, stagnation window; `method` in direct/gmres/block_gmres; diagonal `preconditioner` in none/row_norm/row_column_norm/rwg_diagonal; `recycle_size`; `fallback='direct'` (dense only) |
| scalar GMRES | `linalg._single_gmres` | restarted, twice-modified Gram–Schmidt, CUDA `lstsq`; Arnoldi on `L A R`; recomputes the unscaled true residual `b - A x` after every step (two operator actions per iteration); only the true residual admits convergence |
| block GMRES | `linalg._block_gmres` | rank-deflated variable-block Arnoldi; per-column locking and true residuals; `docs/reference/mom_block_gmres.md` |
| recycling | `_recycle_initial_correction`, `_recycle_update` | initial true-residual minimization over stored A U, then ordinary GMRES |
| batched RHS | `_correct` | leading batch dims share setup; `gmres` loops per RHS, `block_gmres` shares one block Krylov space; operators accept `[..., N]` inputs |
| fixed right inverse | `block_preconditioner.LeafBlockInverse` + `RightPreconditionedOperator`, used by `HierarchicalOperator.solve(right_inverse=...)` | FMM only via `FMMConfig.leaf_factor_kernel` (`original`/`primary`); LU of original leaf principal blocks; GMRES runs on `A P` in similarity-scaled coordinates and recovers `x = P y`. Linear and fixed by construction |
| hierarchical admission | `HierarchicalOperator.solve`, `residual_report` | compressed dual + sampled original rows after the inner solve |
| private action hook | `_single_gmres(..., action=...)` | benchmark prototypes only |

Variable right preconditioner: not possible before this work. `_single_gmres`
forms `x = x0 + R V y`, which is valid only for a fixed linear `R`;
`RightPreconditionedOperator` likewise assumes `P` is linear. A learned action
evaluated in reduced precision, with data-dependent normalisation or with an
inexact inner solve is not exactly linear, so flexible GMRES is required
(revised_research_plan §17).

Exposure of residuals and near blocks. The residual history (absolute and
relative per iteration), matvec counts, setup/solve seconds and allocator peak
are in the solve log (`Result.stats()`). Per-leaf near-field blocks are
available from `ScalarSystemBlocks.__call__(edges, edges)` (any backend that
owns a provider), from the dense matrix by slicing, and leaf partitions from
`SupportOctree` (`cluster.py`; `handle.tree` on hierarchical backends). None of
this was reachable through the public path; the Stage 2 context object
exposes it.

## 3. Hook APIs for learned components (question c)

Principles: additive fields that default to `None`; defaults run the
unchanged code; the operator identity, RHS identity, snapshot schema and
admission rules do not change; the hook receives CUDA tensors and returns CUDA
tensors; no network code in witwin-maxwell.

### 3.1 L-2: right preconditioner (implemented in Stage 2)

```python
config = mw.MoMConfig(..., stopping=StoppingRule(method='gmres', ...),
                      right_preconditioner=factory)
```

- `factory.bind(context) -> bound`, called once in `MoMRuntime.__init__`
  (`preconditioning.bind_right_preconditioner`); `bound.apply(residual) ->
  correction` with shape `[2n]`, the residual in physical RWG test
  coefficients (all J rows, then all M rows), the correction in trial
  coefficients; optional `bound.receipt() -> dict`.
- `PreconditionerContext` (`preconditioning.py`): `shape/device/dtype`,
  `J_slices/M_slices`, `media()` (exterior and per-body εr, μr, k, η),
  `support_tree()`, `leaf_partition(max_block_edges)`,
  `coefficient_indices(edges)`, `principal_block(edges)` (original block
  `A[I, I]`: dense slice or provider integration), `edge_support_boxes()`,
  `edge_features()` (midpoint, tangent, length, both face centroids, normals
  and areas, body id) and `apply_operator(x)`.
- Solver: `linalg._flexible_gmres`, selected only when a preconditioner is
  given. Arnoldi on `L A P(L⁻¹ ·)` (left diagonal scaling unchanged, right
  diagonal scaling replaced by `P`), all preconditioned directions stored
  (`x = x0 + Z y`), true residual after every step, same stagnation/breakdown
  semantics and log keys plus `krylov_method`, `preconditioner_applications`,
  `preconditioner_seconds`. Hierarchical backends keep their compressed and
  sampled-original admission unchanged.
- Built-ins: `IdentityRightPreconditioner` (reproduces unpreconditioned
  GMRES), `LeafBlockRightPreconditioner(max_block_edges, budget_bytes)`
  (block Jacobi on support-tree leaves, batched LU per block-size class;
  dense handles build their tree with `leaf_size`, hierarchical handles reuse
  the operator tree, so blocks never exceed its leaves),
  `FunctionRightPreconditioner(factory, name)` (plain callable).
- Admission: `method='gmres'`, no recycling, no fallback, free-space scalar
  linear runtime (no periodic/harmonic/linear-coupled/tensor), not combined
  with `FMMConfig.leaf_factor_kernel`.
- Identity decision: the preconditioner only changes the iteration path, so
  it is recorded in `Result.stats()['right_preconditioner']` (receipt plus
  setup seconds) and not in `_numerical_identity`. Snapshots restore
  decode-only exactly as before and never need the preconditioner.
- Training contract (§17): a learned `P` must be trained on arbitrary
  algebraic residuals of the target operator, e.g. Krylov vectors `v_j` and
  random/structured residuals, with loss `‖A P(r) − r‖ / ‖r‖`, never only on
  physical RHS `b`. `context.apply_operator` and `principal_block` provide the
  required data without dense matrices. Inputs to the network are the residual
  restricted to a leaf or patch, its principal block (or its LU/inverse
  action), `edge_features` and `media()`; the network is exactly complex-linear
  in the residual if it is a learned linear map or a residual-linear
  architecture with geometry-only nonlinear conditioning. FGMRES tolerates a
  nonlinear residual path, but the linear one is the default design target.

Where it lives: hook, context and classical preconditioners in witwin-maxwell
(`mom/preconditioning.py`, `mom/linalg.py`). The networks, datasets and
training scripts live in instamax (`instamax/precond/`, sharing the Phase 2
methodology) or maxwell-baselines; they implement `bind(context)` and depend on
witwin-maxwell, never the reverse.

### 3.2 L-3: per-body local response and warm start (proposed)

Minimal additive API, not yet implemented:

```python
config = mw.MoMConfig(..., initial_guess=provider)
# provider.bind(context, rhs) -> x0 [2n], CUDA, finite
```

- Runtime: `MoMRuntime.run` passes `x0` to `correct(handle, rhs, x0, rule)`
  (dense) or to a new `x0` argument of `HierarchicalOperator.solve`, which
  today always calls `solve_reference` with zeros. The log records the initial
  true residual (already the first `residual_history` entry), so §17's
  residual-check mode is free: accept if the initial residual passes, else
  correct with GMRES/FGMRES.
- Per-body local response (a network that predicts body i's isolated
  response `x_i = N_i(b_i)`) is expressible through both hooks: as an
  `initial_guess` (one-shot, then corrected) and as a right preconditioner
  `P(r) = ⊕_i N_i(r_i)` on `context.J_slices/M_slices`, which is the
  block-Jacobi-by-body preconditioner with learned blocks; inter-body coupling
  stays in the exact operator/FMM.
- Also needed: `PreconditionerContext.body_rhs(i)` (slice of the lifted RHS)
  and per-body `principal_block` over a whole body for small bodies.
- Repository: hook in witwin-maxwell; local response networks in instamax or
  maxwell-baselines.

### 3.3 L-4: sparse field queries (proposed)

Current cost model (`SurfaceData.query_fields` → `lifting_fields.decode_system`
→ `fields._potentials`, `_evaluate_adaptive`, `_adaptive_system_exterior`):

- Direct, every query against every panel of every body, in batches of 16
  queries; near panels (distance < 2 panel radii) use Duffy fan quadrature with
  analytic static parts, far panels a tensor rule of `order`.
- Adaptive by default (`FieldQueryConfig.adaptive=True`): order doubles from
  `max(order, singular_order)` until successive results agree within
  `relative_tolerance=1e-5`; the multi-body exterior repeats whole-system
  evaluation per order. Cost ≈ Σ_passes N_q · F · q(order) kernel
  evaluations, q ≈ order² samples, plus O(N_q F) ownership classification.
- Example: 64λ (F ≈ 3·10⁶ at h = λ/10), N_q = 10⁴, two passes (orders 8, 16):
  ~10¹³ kernel evaluations, i.e. hours on one GPU. 1e5 points: ten times more.
- There is no FMM-accelerated query path. The FMM trees, moments and local
  expansions are built for the Galerkin test space only.

Measured (L-0, shared GPU): ≈ 10.5 s fixed plus 2.8 ms per triangle per
10⁴ points, i.e. ≈ 0.3 μs per point–triangle pair, 16 points per chunk.
Cost model of a batched/accelerated evaluator: with P points and F faces,
direct cost ≈ c_d·P·F·q (q quadrature samples); a pFFT/FMM evaluator costs
c_g·G log G (grid or tree) + c_n·P·n_near + c_p·P·s, where n_near ≈ 50–100
near faces per point and s the interpolation stencil. For F = 3·10⁶ and
P = 10⁴ the direct path is ≈ 2.5 h at the L-0 rate; an accelerated path
reuses the solve's grid/tree (G ≈ 10⁸) and costs seconds, so the field query
should share the solve's accelerator.

Proposal (witwin-maxwell): `FieldQueryConfig(accelerator='fmm')` that reuses
the prepared exterior tree and multipole moments, builds a target octree over
query points, evaluates far contributions by M2L/L2P (or directional
translation) and near panels with the existing direct quadrature, with an
explicit per-point error budget tied to the backend's kernel error budget. The
direct path stays the reference and default. A learned sparse-query decoder is
not proposed; the Green representation is exact given the currents.

## 4. L-1 baseline protocol (question d)

### 4.1 Scene families

| Family | Bodies | Geometry source | Reference |
| --- | --- | --- | --- |
| `spheres` | 2–N dielectric spheres, εr ∈ {2.5, 4}, diameter 1–4λ, gaps ≥ 0.5λ, linear and planar clusters | `Sphere` | dense LU (same mesh, ≤ 5λ); Mie for each isolated sphere; multi-sphere T-matrix reference supplied by maxwell-baselines (external adapter, not yet available); mesh-refinement self-convergence |
| `boxes` | 2–N boxes/prisms, sharp edges | explicit `Mesh` (structured box triangulation) now, `Box`/`Prism` after the polyhedral branch lands | dense LU; self-convergence under refinement; optical theorem |
| `cylinders` | finite cylinders | explicit `Mesh` now, `Cylinder` later | as boxes |
| `mixed` | spheres + boxes + cylinders | as above | as above |

Every case records the total extent, per-body unknowns, minimum gap
(`SurfaceSystem.pair_gaps`) and mesh edge length in exterior and interior
wavelengths.

### 4.2 Sizes and backends

- Dense tier: 2, 3, 4λ (5λ if memory allows), 2–4 bodies, all families.
- Fast tier: 8, 16, 32, 64λ total extent, run only on a backend whose
  preflight admits the size; first candidate is whichever backend L-0 shows to
  scale furthest.

### 4.3 Preconditioners

`none` (identity, FGMRES), `none-long` (no preconditioner with restart
300–500 and the stagnation window set to the restart length; plain GMRES),
`diag` (existing `rwg_diagonal`/`row_norm`/`row_column_norm` scaling, plain
GMRES — measured to hurt: it minimises a scaled residual and can leave O(1)
current errors, and `row_norm` diverged in L-0), `leaf_block` (`LeafBlockRightPreconditioner`, block sizes 16/32/64),
`body_block` (exact LU per body, dense tier only; an upper bound on what a
learned per-body block can achieve). Calderón preconditioning is not
implemented in this repository; it is a separate backend branch whose extra
operator and Gram costs must be counted (revised_research_plan §18.2).

### 4.4 Stopping, error and timing

- Stop on the true unpreconditioned residual `‖b − A x‖ ≤ rtol ‖b‖`
  (rtol 1e-6 for timing, 1e-8 for reference agreement), maxiter and restart
  stated per run; hierarchical runs additionally report the compressed dual and
  sampled original residuals.
- Errors: relative current error against dense LU (same mesh), far-field
  amplitude error on a fixed direction set, optical theorem defect
  (σ_ext from forward amplitude vs σ_sca from far-field integration, lossless),
  Mie error per isolated sphere where applicable, near-field point error at
  fixed d/h.
- Timing (§20): T_mesh (compile), T_operator setup, T_precond setup,
  T_solve, T_query (direct path; 10³ and 10⁴ points), each CUDA-synchronised;
  iterations, operator actions and preconditioner applications; peak
  allocated/reserved bytes per phase; device, dtype, git revision and other
  GPU tenants. For learned preconditioners, Q_break uses the classical
  preconditioner setup as T_off,base.

## 5. Implementation status, validation and prepared runs

### 5.1 Stage 2 deliverables

- `mom/preconditioning.py` (context, built-ins, binding), `_flexible_gmres`
  in `linalg.py`, `right_preconditioner` threading in `hmatrix.py`,
  `direct_operator.py` and `runtime.py` (`MoMConfig.right_preconditioner`),
  public exports, `FEATURE_LIST.md`.
- `tests/mom/test_right_preconditioner_hook.py`: default path bit-identical to
  the unchanged `solve_reference`; identity FGMRES reproduces unpreconditioned
  GMRES (iterations within one, currents within 1e-8); a user-written
  block-Jacobi callable and the built-in leaf preconditioner reduce iterations
  with true-residual admission; a deliberately non-stationary nonlinear action
  still converges to the LU solution; the FMM backend keeps compressed and
  sampled-original admission; configuration admission rejects direct, block,
  recycled, fallback and FMM-leaf-factor combinations and a wrong-shape action.
- `benchmark/mom_large_scale_l1.py`: families `spheres|boxes|cylinders|mixed`
  (boxes and cylinders as explicit structured solid `Mesh` until the faceted
  primitives land), all backends, preconditioners `diag` (backend diagonal
  scaling, plain GMRES), `none`, `leaf:<edges>`, `body` (exact per-body LU,
  dense only), references `lu|tight|none`, `--dry-run` host-only estimates,
  JSON output with per-phase CUDA peaks.
- Changes to `mom/*.py` require a new Bench provider epoch before Bench
  consumes them.

Practical setup observations on the shared cuda:1 (other tenants at ~99 %
utilisation, so absolute times are pessimistic): dense assembly 3.2 s at
N = 480 and 34 s at N = 1920 (≈ N²); H-matrix setup did not finish within
5 minutes at N = 120 (streamed ACA row sampling in `lowrank.py`), and within
12 minutes at N = 1920; Cartesian FMM at fixed p6 prepared N = 120 in 22 s.
Every iterative step costs two operator actions (Arnoldi plus true residual).

### 5.2 Small validation (dense tier, cuda:1 lane C, 2026-10-28)

`python -m benchmark.mom_large_scale_l1 --backend dense --reference lu
--preconditioners diag none leaf:16 leaf:32 leaf:64 leaf:128 body --rtol 1e-6
--maxiter 1000 --restart 200 --queries 1000`, εr = 4 / 2.5 alternating,
plane wave along +z. Iterations to the true relative residual 1e-6 (✗ = not
reached within 1000; value in brackets is the final residual). Errors are
far-field relative errors against same-mesh dense LU. Archived JSON:
`docs/assessments/sie-large-scale-l1-small-2026-10-28/` (recorded as
`source_dirty` against `1a6a27f6`: they ran on the uncommitted Stage 2 tree,
whose code was committed unchanged afterwards). The GPU was shared with
other tenants, so seconds are descriptive only.

| Case | N | diag (row/col GMRES) | none (FGMRES) | leaf:16 | leaf:32 | leaf:64 | leaf:128 | body LU | far err (best leaf) | optical theorem |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 spheres, 1λ, extent 3λ, ppw 6 | 1 920 | ✗ [8e-2] | ✗ [1.4e-3] | 592 | 559 | 559 | 576 | 6 | 7.9e-7 | 1.1e-3 |
| 4 spheres, 0.6λ, extent 4λ, ppw 8 | 3 840 | ✗ [4e-2] | ✗ [2.2e-3] | 652 | 741 | 850 | 771 | 9 | 8.9e-7 | 5.2e-5 |
| 2 cubes, 0.8λ, extent 3λ, ppw 8 | 3 528 | ✗ [0.42] | ✗ [0.29] | ✗ [2.9e-6] | ✗ [5.0e-6] | ✗ [8.7e-5] | ✗ [1.7e-4] | 6 | 1.6e-6 | 1.8e-4 |
| sphere + cube + cylinder, 0.8λ, extent 4λ, ppw 7 | 3 228 | ✗ [0.47] | ✗ [0.34] | ✗ [9.2e-6] | ✗ [6.8e-5] | ✗ [3.3e-5] | 974 | 13 | 8.2e-7 | 1.1e-5 |
| 1 sphere, 1λ, ppw 8 (Mie check) | 3 840 | ✗ [6e-3] | 925 | – | 367 | – | – | 1 | 2.1e-6 | 3.1e-5 |

Findings:

1. The hook works end to end on dense multi-body and sharp scenes; every
   converged FGMRES solution agrees with LU to the residual level (current
   errors 2–5e-6, far-field ≤ 2e-6), and the optical theorem holds to
   1e-5–1e-3 (the 1e-3 case is the coarse ppw 6 mesh).
2. The existing diagonal-scaled plain GMRES does not reach 1e-6 on any case;
   its scaled-space minimisation even leaves current errors of O(1). Identity
   FGMRES (true-residual Euclidean minimisation) is better but also fails on
   all multi-body cases. This is a weak classical baseline, not a solved one.
3. Leaf block Jacobi helps (≈ 2× on spheres) but is insensitive to block size
   from 16 to 128 edges and fails on sharp-edged bodies within 1000
   iterations: the ill-conditioning is not local to support-tree leaves.
4. Exact per-body LU converges in 6–13 iterations in every case: for gaps of
   0.5–2.8λ the inter-body coupling is benign and intra-body (resonant,
   sharp-edge) conditioning dominates. Per-body LU costs O(N_body³) and is an
   upper bound only. This makes a learned per-body (L-3) or per-patch (L-2)
   inverse, applied through the right-preconditioner hook, the most promising
   target; Calderón-type preconditioning is the classical competitor to add.
5. Single-sphere far field is within 1.2 % of Mie at ppw 13 exterior /
   8.4 interior, consistent with the S-0 mesh-convergence record.
6. Direct field queries cost 6–12 s per 1000 exterior points at ~10³ faces
   (adaptive order doubling), i.e. ≈ 10 μs per point-face pair.

### 5.3 Prepared large runs (not executed; scheduled by the coordinator)

Each command runs as one process under the scheduled lane, for example
`CUDA_VISIBLE_DEVICES=1 flock <workspace>/wt/cuda1.lock $L1 ...`
after checking that the expected peak plus 4 GB is free; `--memory-fraction`
caps the process. Common prefix (worktree root):

```bash
cd .worktrees/sie-large-scale
export PYTHONPATH=$PWD PY=~/miniconda3/envs/maxwell/bin/python
L1="$PY -m benchmark.mom_large_scale_l1 --family spheres --bodies 2 --eps 2.5 --ppw 10"
```

Scene: two εr = 2.5 spheres of diameter K/3 at total extent K (gap K/3),
exterior ppw 10 (interior ppw 6.3). Sphere meshes are icosahedral, so
`subdivisions` and N jump by four per doubling of K. Always run the dry run
first (host only, no CUDA):

```bash
for K in 8 16 32 64; do $L1 --extent $K --body-size $(python3 -c "print($K/3)") --dry-run \
  --restart 60 --output runs/l1/estimate_${K}l.json; done
```

| Run | Command (append to `$L1`) | N | Memory estimate | Time estimate | Lane |
| --- | --- | --- | --- | --- | --- |
| R8-dense | `--extent 8 --body-size 2.6667 --backend dense --reference tight --preconditioners diag none leaf:32 leaf:128 --restart 100 --maxiter 2000 --queries 10000 --memory-fraction 0.42 --output runs/l1/spheres2_8l_dense.json` | 30 720 | matrix 14.1 GiB + assembly/solver workspace ≈ 17–19 GiB | assembly ≈ N² scaling of the measured rate: 1–2.5 h on a shared GPU; solves minutes; 10⁴ direct queries ≈ 20 min | A (≤ 20 GB) |
| R16-fmm | `--extent 16 --body-size 5.3333 --backend fmm --fmm-order 8 --budget-gb 36 --reference none --preconditioners diag none leaf:32 --restart 60 --maxiter 2000 --queries 1000 --memory-fraction 0.85 --output runs/l1/spheres2_16l_fmm.json` | 122 880 | Cartesian owned 8.6–31.5 GiB (+ FGMRES basis 0.2 GiB) | unknown; expect fail-closed capacity or kernel-order admission within the first hour; time-box 4 h | dedicated cuda:1 |
| R16-dir | as R16-fmm with `--backend directional --output runs/l1/spheres2_16l_directional.json` | 122 880 | unmeasured; planner caps (1e5 contexts/pairs) are the first expected rejection | time-box 4 h | dedicated cuda:1 |
| R32-fmm | `--extent 32 --body-size 10.6667 --backend fmm --fmm-order 8 --budget-gb 40 --reference none --preconditioners none leaf:32 --restart 40 --queries 1000 --memory-fraction 0.9 --output runs/l1/spheres2_32l_fmm.json` | 491 520 | Cartesian owned 34–126 GiB: exceeds 48 GB at the upper end; body-pair separation check 1.6·10⁶ Python batches (≈ 0.5–1.5 h before any operator) | do not launch before R16 passes | dedicated cuda:1 |
| R64-fmm | as R32 with `--extent 64 --body-size 21.3333 --output runs/l1/spheres2_64l_fmm.json` | 1 966 080 | Cartesian owned 137–504 GiB: cannot fit; separation check 2.6·10⁷ batches (≈ 7–22 h) | blocked by preparation and capacity | none until fixed |

Box family equivalents (`--family boxes`) have 1.7× the unknowns of the
sphere family at the same K (8λ: 52 488, dense 41 GiB, so no dense run on a
shared lane). The 64λ goal requires, in order: an O(N log N) body
separation/intersection check, a high-frequency operator with near storage
below ~10 kB/edge, and an accelerated field query (Section 3.3).

## 6. Risks

1. Memory at 32–64λ: no backend's capacity model fits; near storage alone
   exceeds 48 GB at 32λ for the measured Cartesian FMM layout.
2. FMM accuracy controls: Cartesian Taylor orders are capped at 12 and
   admission is per configured kernel bound; electrically large boxes fail
   admission rather than degrade gracefully. Directional FMM accuracy is
   budgeted (`global_ideal_rwg_error_budget`) but unmeasured beyond small
   scenes. The hierarchical admission checks only sampled original rows.
3. Each GMRES step costs two operator actions (Arnoldi + true residual); this
   is part of the residual-certified design and applies equally to all
   preconditioners, but doubles absolute solve time at large N. Checking the
   true residual every k steps would be a separate, explicitly logged option.
4. FGMRES stores both V and Z: 2·restart·N·16 B (restart 100, N = 9·10⁶:
   29 GB). Large runs must use short restarts.
5. Quadratic preparation (separation, intersection, vertex checks) blocks
   multi-body runs above ~10⁵ faces per body pair before any operator exists.
6. No multi-sphere T-matrix reference exists in the four repositories yet.

## 7. L-1b: preparation profile, quadratic preparation and accelerator choice

### 7.1 FMM preparation: where the time and memory go

Tool: `python -m benchmark.mom_fmm_prepare_profile` wraps every preparation
stage of the public `Simulation.prepare()` path with CUDA-synchronised timers
and allocator snapshots (retained bytes, transient peak) and attaches a host
profile. Configuration of the L-0 repro: εr = 4 sphere, D = 1λ,
`sphere_mesh` frequency n (20n² faces), `IntegrationConfig(order=4,
singular_order=4)`, `FMMConfig(order=6, leaf_size=30, separation_ratio=0.25,
compact_shared_displacement_v1, separable_v1, scalar_reciprocal_dk_v1)`.
All timings are on a shared, saturated cuda:1 and are descriptive.

Time, 320 faces (480 RWG, 960 unknowns):

| Tree | Prepare | Near-field assembly | Other |
| --- | --- | --- | --- |
| base `a4eb33c2` | 353.6 s | 326 s (3192 leaf-pair blocks, one Python call chain each) | moments 3.2 s, capacity planner 2.4 s, RHS ≈ 21 s |
| base + S-perf (`75772d09`, `32cc9217`, scratch merge) | 337.6 s, then failed in `build_rhs` | 314 s | see defect (f) |
| L-1b (this branch) | 13.8 s | 8.9 s | moments 0.5 s, planner 0.2 s |
| L-1b, 720 faces | 132.7 s | ≈ 55 s | RHS lifting 75.6 s (`lifting.build_selected_lifted_rhs`) |

Root cause. With leaf_size 30 and separation_ratio 0.25 a 1λ body has no
admissible far pair at all (m2l = 0 bytes, 56 leaves): the "FMM" is a block
dense operator, and the packed near store assembled each of the
56² ≈ 3·10³ leaf-pair blocks separately through
`exterior_channels → _regional_selection → scalar_regional_channels →
_scalar_regional_accumulate`, plus per-block layout validation
(`torch.equal`, `unique`, `nonzero`, range checks). Each call forces a host
synchronisation; on a busy GPU each sync costs milliseconds, so 3·10³ blocks ×
≈ 30 syncs dominated (host profile: 80 s in `torch.equal`, 38 s in
`nonzero`, 36 s in `unique`, 53 s in the per-block accumulate loop). The
quadrature itself was a small part, which is why S-perf's faster kernels
change the total by only 5 %.

Fix (local, in `nearfield.py`, `packed_near.py`, `fmm_regional.py`):

- `nearfield.scalar_edge_pair_channels` integrates the original D/K values
  for many paired RWG edges at once: all triangle pairs of a request are
  deduplicated, classified once and integrated with the unchanged
  `_pair_integrals` / `_adaptive_near_integrals` calls and batch bounds; each
  entry sums its four support contributions. `scalar_block_channels` and
  `ScalarSystemBlocks.exterior_channels_many` (body-pair routing) build
  equal-shape blocks [B, r, c] from it.
- `PackedNearBlocks._initialize` accepts an optional batched factory
  (`factory.many`), storing a shape group 256 blocks at a time with one
  finiteness check; `fmm_regional` attaches it for the retained D/K storages
  (`scalar_dk_v1`, `scalar_reciprocal_dk_v1`). Streamed/cached reciprocal
  storages still integrate per block at every action (next candidate).
- Layout validation is vectorised per index length (`_check_node_maps`), with
  the same accept/reject conditions; the reciprocal reverse-map equalities are
  implied by one map per physical node and are no longer re-checked pairwise.
- `tests/mom/test_batched_near_assembly.py`: batched blocks match the
  per-block path to ≤ 1e-14 relative (exterior, cross-body and interior), and
  the public FMM near storage and operator action match the per-block
  reference for both D/K storages (`fmm_regional.BATCHED_NEAR_ASSEMBLY`
  toggles the reference path).

After the fix, preparation at 720 faces is dominated by RHS lifting
(`lifting.py`, S-perf's scope) and by genuine near quadrature on a block-dense
operator.

Measured time at 1620 faces (1λ, 2430 RWG, L-1b, shared GPU): prepare 724 s,
of which RHS lifting 447 s (`lifting.py`), regional near/far construction
244 s (near quadrature ≈ 200 s, M2L tables 8.7 s), capacity planning 26 s
(host loops in `cluster._block_partition`: 64 s cumulative), moments 6 s.
Planning alone takes 49 s at 1λ and 180 s at 2λ (L-0), and did not finish
within 400 s at 2λ here: the structural planner's per-pair host loops are the
next preparation wall after RHS lifting.

Retained bytes per RWG edge by component (same configuration; owned tensors
equal the structural plan exactly — 317,892,704 B at 1620 faces both measured
and planned):

| Case | Faces / edges | Near D/K | M2L tables | Leaf moments + shifts | Trees/maps | Total | Far pairs per region (unique owners) | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1λ | 320 / 480 | 17.6 kB | 0 | 11.1 kB | 0.1 kB | 28.8 kB | 0 | measured |
| 1λ | 720 / 1080 | 37.2 kB | 0 | 10.9 kB | 0.1 kB | 48.1 kB | 0 | measured |
| 1λ | 1620 / 2430 | 84.5 kB | 35.4 kB | 10.8 kB | 0.1 kB | 130.8 kB | 11,096 (11,096) | measured = plan |
| 2λ | 6480 / 9720 | 127.2 kB | 269.8 kB | 10.8 kB | 0.1 kB | 408 kB | – | plan (L-0) |
| resonant sphere f30, εr=12 | – / 27,000 | 113 kB | 425 kB | 10.8 kB | – | ≈ 550 kB | – | plan (GSP-09 record) |

Not in the table, per edge: Riesz metric 128 B, RHS 32 B, GMRES basis
(restart+1)·32 B (16 kB at restart 500; FGMRES twice), leaf LU factors
≈ 2 kB at 32-edge blocks when enabled; prepared geometry and quadrature
caches are a fixed ≈ 9 MB at these sizes; setup transients peaked at
0.5 GB at 1620 faces.

Scaling and dominant component:

- Leaf moments are constant at C(p+3,3)·4·16 B per edge per region
  (10.75 kB at p=6, two regions).
- Near D/K grows while leaves fill and then saturates near 110–130 kB/edge:
  with separation_ratio 0.25 a pair of leaves is admissible only at four
  times their combined radius, so each edge keeps ≈ 2–4·10³ near partners
  (32 B of D/K per stored pair per region; reciprocal storage keeps one
  direction).
- M2L tables dominate from 2λ on and grow superlinearly: total M2L rose
  30× for 4× edges between 1λ and 2λ (≈ N^2.5). Every logical far pair kept
  its own coefficient owner (11,096 unique of 11,096): adaptive support-box
  centres make every displacement unique, so the exact displacement sharing
  saves nothing on curved surfaces, and the strict admissibility plus the
  order cap push interactions down to many small boxes. Two regions (exterior
  and interior trees) double it.
- Why the resonant sphere needed ≈ 550 kB/edge: 425 kB of unshared M2L
  tables (≈ 4–11 kB per far pair at p6 times several hundred far pairs per
  leaf, times two regions with εr = 12 inside) plus 113 kB of saturated near
  field.
- Extrapolation (edges ∝ D² at the L-0 density: 6.2·10⁵ / 2.5·10⁶ / 10⁷ at
  16/32/64λ): near alone 75 GB / 300 GB / 1.2 TB at 120 kB/edge, M2L at least
  170 GB / 680 GB / 2.7 TB at the 2λ rate and growing. The Cartesian FMM
  cannot be made to fit by storage policy; this feeds Section 7.3.

Defects found in L-0 and here (owner, size):

- (a) Automatic Cartesian order (`max_order=12`, `KernelErrorBudget(1e-6,
  1e-4)`) raises `MultipoleOrderError` at 1λ. Not a bug: Taylor order ≤ 12
  cannot meet the bound for wavelength-sized admissible boxes. Owner: FMM
  backend; resolution is architectural (Section 7.3), not a local fix.
- (b) Default `FMMConfig(table_storage='dense')` plans 24.5 GB of M2L tables
  at 1λ and is rejected by capacity admission. Defaults are unusable beyond
  ≈ 0.5λ; switching the default to `compact_shared_displacement_v1` +
  `separable_v1` + reciprocal D/K is a one-line config change but changes
  default payload identities (Bench epoch). Owner: MoM runtime, small.
- (c) `compressed_dual_residual_above_budget` after inner GMRES reached
  9.85e-5: the admission is correct but measures a different norm. GMRES
  stops on the Euclidean relative residual; admission checks the
  impedance-balanced Riesz dual norm (`LocalRWGRiesz`), which weights J and M
  rows differently, so a 1e-4 Euclidean residual can exceed a 1e-4 dual
  budget. Fix: set the inner `rtol` about 10× below `solve_dual_tolerance`
  (benchmark/config guidance), or stop GMRES on the dual norm. Owner: MoM
  Krylov, small; no admission change recommended.
- (d) `StoppingRule(stagnation_window=20, stagnation_rtol=1e-4)` stops
  long-restart GMRES on PMCHWT plateaus (seen in L-0 and L-1). Fix: default
  the window to the restart length; changes `asdict(StoppingRule)` in
  numerical identities (Bench epoch). Owner: MoM Krylov, small. The L-1
  benchmark already sets the window explicitly.
- (e) Field queries are direct sums in chunks of 16 points (Section 3.3);
  owner L-4 (accelerated evaluator) with S-perf's batching as the interim.
- (f) The S-perf branch (`75772d09`, `32cc9217`) merged onto `a4eb33c2` fails
  in every hierarchical/FMM `build_rhs`: `SystemOperator.build_rhs` now calls
  `self._record_rhs`, which `HierarchicalSystemOperator.build_rhs` (calling
  `SystemOperator.build_rhs(self)` unbound) does not provide.
  Owner: S-perf, one-line fix before merge.
Default changes deferred to the Bench v2 epoch (not applied; each alters
default numerical/operator identities):

- (b) `witwin/maxwell/mom/fmm_config.py:11-13`, `FMMConfig` defaults:
  `table_storage: str = 'dense'` → `'compact_shared_displacement_v1'`,
  `shift_storage: str = 'dense'` → `'separable_v1'`,
  `near_storage: str = 'full_matrix'` → `'scalar_reciprocal_dk_v1'`.
  Effect: default retained memory at 1λ drops from a rejected 24.5 GB plan to
  ≈ 0.32 GB (Section 7.1 table); `FMMConfig().payload()['version']` changes
  from `cartesian_fmm_config_v7` to `cartesian_fmm_shared_compact_config_v1`,
  so every default-FMM operator identity and snapshot schema payload changes.
  The shared and D/K storages require free space and complex128, so periodic
  FMM users and complex64 configurations must then set `table_storage` and
  `near_storage` explicitly (their `MoMConfig` admission would otherwise
  reject the new defaults).
- (c) No admission change. The exact change is one guard in
  `witwin/maxwell/mom/runtime.py`, `MoMConfig.__post_init__`, in the
  hierarchical branch:
  `if self.stopping.rtol > 0.1*solver_config.solve_dual_tolerance: raise ValueError(...)`
  (with `solver_config` the selected `hierarchical`/`h2`/`direct_h2`/`fmm`
  config). Effect: configurations whose inner Euclidean GMRES tolerance is
  looser than one tenth of the Riesz dual admission budget are rejected up
  front instead of returning `compressed_dual_residual_above_budget` after a
  full solve. No identity change, but existing configurations with
  `rtol == solve_dual_tolerance` (such as the L-0 1e-4/1e-4 setting) start
  failing at construction. The alternative (stopping GMRES on the dual norm)
  is a larger Krylov change.
- (d) `witwin/maxwell/mom/linalg.py:817`, `StoppingRule`:
  `stagnation_window: int = 20` → `stagnation_window: int | None = None`, with
  `if self.stagnation_window is None: object.__setattr__(self, 'stagnation_window', self.restart)`
  at the top of `__post_init__`. Effect: the default window equals the
  restart length (40 by default), so long-restart PMCHWT runs are no longer
  stopped on transient plateaus; `asdict(StoppingRule())` changes
  (`stagnation_window` 20 → 40), which changes `_numerical_identity` of every
  default configuration and every stored `stopping` payload.

- (g) After rebasing onto S-1c, its anisotropic-bisection fallback for near
  pairs unresolved at `near_max_order` must also be reached from
  `scalar_edge_pair_channels` (the batched path calls
  `_adaptive_near_integrals` directly, like the per-block path did before
  their change). Owner: L-1b at rebase.


### 7.2 Quadratic preparation removed (commit c9877af4)

`spatial.BoxTree` sorts boxes along a Morton curve and groups them eight at a
time; `overlapping_pairs` and `minimum_item_distance` traverse node pairs
level by level and prune only when node boxes prove that no item pair can
pass. The original exact predicates then decide:

- `mesh._has_coincident_vertices`: tree candidates at 2× tolerance, then the
  original L∞ test (replaces the V² scan in `validate_mesh`).
- `mesh._has_intersections`: tree candidates at 2× tolerance, then the original
  box test and the original edge/face/coplanar predicates (moved unchanged into
  `_pair_intersections`).
- `SurfaceSystem._validate_separation` → `mesh._minimum_triangle_distance`:
  exact branch and bound; representative vertices give an upper bound and a
  pair is pruned only when its box gap exceeds it by a 1024·eps margin, so the
  minimising pair is always evaluated with `triangle_pair_distances`. The
  recorded gap is bitwise equal to the all-pairs minimum.
- `_topology`: vertex stars built in one pass instead of one face scan per
  vertex (host, linear).

`tests/mom/test_spatial_validation.py` compares against verbatim copies of the
original scans: coincident vertices planted at 0–1000× tolerance, 14
self-intersection soups (overlapping, touching and 1e-15/1e-13 gaps, coplanar
face contact, partial overlap, random soups, a graded mesh with one huge
triangle), and 12 separation cases (gaps 0–3, facing aligned cubes with tied
minima, rotated cubes, an edge 1e-9 from a face, float32); all decisions and
every gap match exactly.

`python -m benchmark.mom_validation_scaling` (two icosahedral spheres of F
faces each, gap 0.1 radius; all-pairs reference loaded from git c90c8346):

| Faces per body | Pruned total (s) | of which topology / vertices / intersections / separation | All-pairs total (s) | All-pairs separation (s) |
| --- | --- | --- | --- | --- |
| 980 | 0.19 | 0.006 / 0.032 / 0.079 / 0.054 | 3.8 | 3.5 |
| 3 920 | 0.42 | 0.024 / 0.042 / 0.075 / 0.084 | 58.9 | 57.5 |
| 10 580 | 0.53 | 0.069 / 0.049 / 0.059 / 0.101 | 375.6 | 368.8 |
| 20 480 | 0.71 | 0.48 / 0.009 / 0.025 / 0.042 | (≈ 1.4·10³ extrapolated F²) | |
| 81 920 | 3.2 | 1.64 / 0.015 / 0.079 / 0.031 | (≈ 2.2·10⁴) | |
| 200 000 | 6.9 | 4.2 / 0.011 / 0.13 / 0.025 | (≈ 1.3·10⁵) | |
| 499 280 | 15.8 | 11.0 / 0.015 / 0.31 / 0.028 | (≈ 8·10⁵, about 9 days) | |

The GPU checks are now negligible; the remaining linear cost is host
combinatorics (`_topology` and the `RWGBasis` edge loop, ≈ 17 s at 5·10⁵
faces), which is acceptable but would be the next item to move to CUDA.


### 7.3 Accelerator comparison for 16–64λ full-wave scenes

Only full-wave methods are compared; asymptotic approximations are out of
scope. Every candidate must report (i) the true residual of the original
discrete operator (sampled original rows plus a full residual with the
accelerated operator at a tighter accuracy setting) and (ii) a refinement
convergence check (mesh h → h/√2 and accelerator accuracy setting one step
tighter). Candidates are ranked first on whether their accuracy is
controllable and verifiable that way.

Target scenes and unknown counts (h = λ/10 outside, λ_in/10 inside;
346 RWG per λ² of surface at λ/10, four times that at λ/20):

| Scene | Surface | RWG edges | Unknowns |
| --- | --- | --- | --- |
| S1 16λ PEC cube (EFIE/CFIE) | 1536 λ² | 5.3·10⁵ | 5.3·10⁵ |
| S2 16λ εr=4 cube (PMCHWT, λ_in/10 mesh) | 1536 λ² | 2.1·10⁶ | 4.3·10⁶ |
| S3 two 10λ εr=4 cubes in a 32λ box | 1200 λ² | 1.7·10⁶ | 3.3·10⁶ |
| S3-PEC same, PEC | 1200 λ² | 4.2·10⁵ | 4.2·10⁵ |
| S4 8×8 array of 2λ εr=4 cubes, 64λ × 64λ × 2λ slab | 1536 λ² | 2.1·10⁶ | 4.3·10⁶ |
| S5 open PEC sheet (later; needs open-surface RWG) | – | – | – |

A dielectric body needs two kernels: the exterior k₀ over the whole scene and
the interior k₁ = 2k₀ over each body's own bounding box.

#### C1. FFT-based accelerators (AIM / precorrected FFT / IE-FFT)

Mechanism: project each RWG basis (and its divergence) onto a uniform
stencil of (M+1)³ grid points by moment matching, convolve with the exact
Helmholtz Green's function on the grid (zero-padded 2× per axis, cuFFT through
`torch.fft`), interpolate back, and replace the grid's near interactions by the
original near-field quadrature (precorrection = exact block − grid block).

- Accuracy controls: grid spacing h_g, stencil order M, precorrection radius.
  Typical targets: 1e-3 with h_g ≈ λ/10 (λ_in/10 inside) and M = 2–3
  (27–64 points); 1e-4 with h_g ≈ λ/15 and M = 3–4 (64–125 points). Errors are
  smooth and uniform in the far zone and are measured directly by comparing
  the accelerated action with the exact original action on sampled rows (the
  existing `residual_report` pattern) — the most verifiable of the candidates.
- Memory: padded grid points P = 8·(L/h_g)³ for a box of side L. One complex64
  array: 32λ box at λ/10 → 2.6·10⁸ points, 2.1 GB (c128 4.2 GB); 64λ box at
  λ/10 → 2.1·10⁹ points, 16.8 GB (c128 33.6 GB). A matvec needs the Green's
  transform plus two working arrays when the eight potential components
  (A, φ, F, ψ for PMCHWT; four for EFIE) are streamed one at a time:
  32λ ≈ 6 GB c64 / 13 GB c128; 64λ cube ≈ 50 GB c64 — infeasible on 48 GB.
  At λ/15 (1e-4) the 32λ box needs ≈ 21 GB c64.
- Cost grows with the empty volume of the bounding box, not the surface: grid
  points ∝ (L/h_g)³ while unknowns ∝ (L/h)². A slab (S4: 64×64×2λ) needs only
  ≈ 6.6·10⁷ padded points (0.5 GB c64); a dense 64λ cube needs 2.1·10⁹.
  Sparse 3D scenes therefore need per-cluster grids plus an outer coupling
  layer (C4).
- Precorrection storage: near pairs within ≈ 3h_g are scale invariant at
  ≈ 100 neighbours per RWG; two channels (D, K) per medium, reciprocal half:
  ≈ 1.6 kB/edge per medium in complex64 (3.2 kB complex128). Stencils add
  (M+1)³ × 4 components × 8 B ≈ 1–2 kB/edge per medium. S2 total ≈ 9–15 GB.
- Matvec time: 16 three-dimensional FFTs of 2.6·10⁸ points (memory bound,
  ≈ 20–40 ms each on this GPU) plus sparse projection/interpolation and the
  precorrection SpMV: ≈ 0.5–1 s at 32λ in complex64.
- GPU/PyTorch fit: excellent. FFTs, sparse projection and the precorrection
  are standard tensor operations; autograd through `torch.fft` and sparse
  matmuls gives current-vector gradients for free (geometry gradients through
  projection weights are possible later).
- Implementation in `mom/`: ≈ 1.5–2.5 kLOC (grid/projection, Green's
  transform cache, potentials and their derivatives on the grid, precorrection
  built with the batched near-field path of 7.1, protocol adapter).
- Reuses: the near-field quadrature and batched D/K assembly, `BoxTree` for
  neighbour lists, `HierarchicalOperator.solve` admission, the Krylov path and
  the right-preconditioner hook.
- Risks: box-volume scaling (64λ cubes), high-contrast interiors (k₁ grid
  inside each body), stencil conditioning on graded meshes near edges, and
  near-singular precorrection on thin features.

#### C2. High-frequency MLFMA

Mechanism: octree with leaf boxes ≈ 0.25λ (of the faster medium), radiation
patterns sampled on (L+1)×2(L+1) spherical grids with L ≈ kd + 1.8 d₀^{2/3}(kd)^{1/3},
diagonal plane-wave translations, local interpolation/anterpolation between
levels, exact near field below the leaf.

- Accuracy controls: digits d₀ (L), interpolation order, leaf size; standard
  error estimates; verifiable as in C1 by sampled original rows.
- Memory: near field ≈ 300 neighbour RWG per edge at 0.25λ leaves: ≈ 5 kB/edge
  per medium (complex64, reciprocal) or 19 kB/edge (complex128, both
  directions). Leaf patterns ≈ 150 directions × 2 components: 2.4 kB/edge per
  medium in c64 (or recomputed on the fly). Upper levels ≈ 0.2 kB/edge per
  level. Totals 6–25 kB/edge: S3 (1.7·10⁶ edges) 10–42 GB; volume independent.
- Complexity O(N log N); matvec ≈ 0.2–1 s at 10⁶ unknowns on one GPU.
- Low-frequency breakdown: plane-wave translations are unstable for boxes
  below ≈ 0.2λ, so mesh grading at sharp edges or thin features inflates the
  near field instead of adding levels; a broadband version needs a
  low-frequency expansion at fine levels (the existing Cartesian FMM can play
  that role in principle).
- GPU fit: good but irregular (per-level gathers, spherical interpolation);
  autograd through translations is feasible but adds memory.
- Implementation: ≈ 4–6 kLOC (spherical sampling/interpolation, translation
  tables, aggregation/disaggregation, two media, protocol adapter).
- Reuses: tree code, near field, admission, Krylov, hook.
- Risks: implementation size, interpolation error control near the top level,
  low-frequency breakdown at fine features.

#### C3. Current Cartesian / directional FMM made scalable

- The Cartesian Taylor FMM (orders ≤ 12) cannot admit boxes larger than about
  a wavelength, so above ~4–8λ the far work falls to fine levels and grows
  quadratically regardless of storage; measured retained cost is
  150–550 kB/edge (Section 1.2, 7.1).
- The directional DAG FMM is the in-repository high-frequency design, but its
  planner caps (10⁵ contexts/pairs), Python-level per-pair setup and lack of
  any measurement beyond ~100 edges make it the largest-risk path.
- Strength: the most rigorous accuracy admission in the repository (explicit
  kernel and RWG error budgets, original sampled residual), so it remains the
  verification reference for small and mid-size cases and a candidate
  low-frequency fine-level component for C2.
- Recommendation: do not scale; keep for ≤ 2–4λ references.

#### C4. Domain decomposition / multiple scattering as an outer layer

These do not replace C1–C3; they structure the solve on top of one of them.

- Per-body T-matrix (multiple scattering): T of each body from 2L(L+2)
  vector-spherical RHS (block GMRES exists); L ≈ kR + 7. For S4 (2λ cubes,
  kR ≈ 11): ≈ 720 harmonics, 8 MB per T-matrix, one T-matrix for 64 identical
  bodies. Inter-body coupling by translation (dense 64×64 blocks or plane-wave
  FMM). Accuracy control: L truncation and the body solves' tolerance; the
  final answer is verified by recovering surface currents and evaluating the
  true residual with C1/C2.
- CBFM / macro basis: per-block macro basis from plane-wave excitations,
  SVD-truncated; reduced coupling matrix via C1/C2. Accuracy: SVD threshold;
  verified by the full true residual.
- IE-DDM: per-body subdomain solves with transmission conditions as a
  preconditioner of the global FGMRES on C1/C2.
- Learned components: the L-1 dense validation showed that exact per-body LU
  reduces iterations from ≥ 1000 to 6–13. L-2 (learned right preconditioner)
  sits exactly at the per-body/per-block inverse inside the FGMRES on C1/C2;
  L-3 (learned local response) sits at the body T-matrix or the macro basis
  generator. Both remain inside a loop that checks the true residual of the
  original operator.

#### Ranking and recommendation

| Candidate | Accuracy control and verification | 16λ (S1/S2) | 32λ (S3) | 64λ (S4 slab / 3D cube) | GPU/autograd | Size | Risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C1 pFFT/AIM | grid h_g, stencil M; uniform, directly measurable | yes / yes (≈ 10–15 GB c64) | yes (≈ 6 GB grid + 6–10 GB near, c64) | slab yes; dense cube no (≥ 50 GB) | excellent | 1.5–2.5 kLOC | box volume |
| C2 MLFMA | d₀, interpolation; standard estimates | yes | yes (10–40 GB) | yes (volume independent) | good | 4–6 kLOC | size, LF breakdown |
| C3 current FMM | rigorous admission, but fails to admit at HF | no | no | no | fair | exists | quadratic HF work |
| C4 DDM/T-matrix/CBFM | outer; verified by C1/C2 residual | – | helps S3 | needed for sparse 3D | good | 0.5–1.5 kLOC each | depends on inner |

Primary path (approved): C1 (precorrected FFT/AIM) for the single-body and
filled-box cases (16λ cubes, slab/board scenes), with per-medium grids
(exterior grid on the scene box, interior grids on each body box), complex64
grids, complex128 precorrection and true residual, and the batched near-field
assembly of 7.1 for precorrection. The backend applies the single-layer L
block on its own (open-PEC EFIE) as well as the PMCHWT 2×2 blocks. Fallback:
C2 (MLFMA) for volume-filling 3D 64λ scenes. C4 is the outer layer.

#### Sparse multi-body scenes as the first large-scale target

MoM cost follows scatterer surface, not domain size: several 1–3λ bodies
spread over a 32–64λ domain are the realistic first large benchmark, and a
whole-box pFFT is wasteful there. The operator is therefore split:

- Self/near part, per body or per cluster of nearby bodies: dense (≤ ~5·10⁴
  unknowns), or a per-cluster pFFT grid, or H-matrix where it qualifies; the
  existing provider supplies exact near blocks for all of them.
- Inter-body coupling, a separate operator between well-separated clusters.
  First choice: a single-level plane-wave (diagonal) translation between
  cluster bounding spheres — each cluster's currents are aggregated into a
  far-field pattern on an (L+1)×2(L+1) sphere grid with L ≈ kR + 1.8 d₀^{2/3}(kR)^{1/3},
  translated with the exact diagonal translation operator for the centre
  vector, and disaggregated onto the receiving cluster's test functions.
  Pairs whose bounding spheres violate the separation criterion fall back to
  exact cross blocks (batched near path). For 64 bodies of 2λ (kR ≈ 11,
  L ≈ 20, ≈ 900 directions) the translations are 4096 diagonal products of 900
  samples; aggregation matrices are shared between identically shaped bodies
  up to a phase. Accuracy is controlled by d₀ and the separation ratio, and is
  measured against the exact cross blocks on sampled rows.
- Why this first: it is the cheapest coupling that is exact up to a
  controllable truncation, it needs no VSWF translation algebra, it reuses
  the tree/near code, and it is the top level of C2, so it is not throw-away.
  The VSWF T-matrix outer layer (per-body T from 2L(L+2) block-GMRES solves)
  is built second, on the same aggregation patterns, for repeated identical
  bodies.
- Preconditioning: the L-1 dense result (per-body LU, 6–13 iterations) maps
  directly onto this split; the right-preconditioner hook applies the
  per-body inverse (exact for small bodies, learned L-2/L-3 later).

#### Independent reference: multi-sphere T-matrix

Add a host-side (NumPy/SciPy) multi-sphere T-matrix (generalized multiparticle
Mie) reference to `benchmark/`: Mie coefficients per sphere from
`benchmark/sie_reference.py`, vector-spherical-wave translation coefficients
by the standard recursions, a dense interaction solve of size
Σ 2L_i(L_i+2), and far-field/cross-section outputs. It is validated against
single-sphere Mie, symmetry and reciprocity, the optical theorem, and
convergence in L. It is exact for sphere arrays, cheap for tens of spheres
of a few λ, and independent of every MoM component.

#### L-1c staging (revised)

1. L-1c.0: pFFT grid projection/interpolation and Green's transform for one
   medium; apply the single-layer L and the PMCHWT blocks; operator-action
   error against dense on 1–4λ spheres and cubes at h_g ∈ {λ/8, λ/10, λ/15},
   M ∈ {2, 3, 4}. New files `mom/pfft_*.py`.
2. L-1c.1: precorrection from the batched near path; exterior plus per-body
   interior grids; public backend with the unchanged admission and hook; true
   residual and refinement checks at 4–8λ. In parallel: the host-side
   multi-sphere T-matrix reference.
3. L-1c.2: split operator for sparse scenes — per-body/per-cluster self
   operator (dense or per-cluster pFFT) plus the single-level plane-wave
   coupling; 4–64 spheres of 1–3λ in 32–64λ domains checked against the
   T-matrix reference; per-body block preconditioner through the hook.
4. L-1c.3: 16λ PEC and dielectric cubes on a single pFFT grid; slab/board
   scenes; T-matrix outer layer for repeated bodies; decide on C2 from the
   measured box-volume limit. The 20 GB figure at 32λ is reported as a data
   point; the hard limit is the 48 GB card.

## 8. L-1c.0: pFFT projection and single-medium Green's transform

Branch `sie/pfft-l1c` (worktree `.worktrees/sie-pfft`), from `63e5c695`.

### 8.1 Implementation

- `mom/pfft_grid.py`: `PFFTConfig(spacing, order M, precision,
  precorrection_radius)` and `PFFTProjection`. Each RWG basis gets an
  (M+1)³ stencil centred at its edge midpoint; weights for the x, y, z current
  components and the surface divergence match every tensor-product moment of
  order ≤ M per axis about the stencil centre. Moments use the existing
  triangle rule at polynomial-exact order; the stencil matrix is one fixed
  Kronecker inverse of a 1D Vandermonde matrix shared by all bases.
- `mom/pfft_operator.py`: `PFFTRegion(basis, medium, config)` for one body in
  one scalar medium. `apply_single_layer` is the original D channel
  (`f_i·f_j − div f_i div f_j / k²` against G): it is the standalone
  single-layer L block for EFIE (Z = iωμ D) and the D part of PMCHWT.
  `apply_double_layer` is the K channel `f_i·(∇G × f_j)`. Both scatter the
  projected sources, convolve with G (or the three components of ∇G) sampled at
  grid offsets through zero-padded FFTs (5-smooth sizes ≥ 2n−1, G(0) := 0),
  and gather with the same weights (Galerkin, so the grid D is symmetric).
  `grid_entries(rows, cols)` returns the grid operator's own entries from a
  cached integer-offset table of G and ∇G; L-1c.1 replaces these by the
  original near entries (precorrection).
- `tests/mom/test_pfft_operator.py`: zeroth moments equal the RWG integrals
  (total charge zero), `grid_entries` equal grid action columns, grid D is
  symmetric, precorrected actions meet 3e-2 (M=2, λ/10) and 3e-3 (M=3,
  λ/15) against original rows, configuration validation.
- `benchmark/mom_pfft_accuracy.py`: samples R test rows, integrates them
  exactly with the batched near path, and reports the precorrected and raw
  action errors, a far-entry error sample, grid/kernel/projection memory and
  the median D and K matvec times; `--fft-sizes` times padded convolutions.

### 8.2 Accuracy against original rows (exterior medium, complex128)

`python -m benchmark.mom_pfft_accuracy`: 512 random test rows integrated
exactly; two random complex vectors; error = max ‖(A x)_rows − (A_pFFT x)_rows‖ /
‖(A x)_rows‖ after precorrection inside (M+2)·h_g. Mesh edge ≈ λ/15 (spheres)
and λ/10 (cube). Grid ppw is per wavelength of the medium.

| Grid ppw | M | D (=L), sphere 1λ | K, sphere 1λ | D, cube 1λ | K, cube 1λ | D, sphere 2λ | K, sphere 2λ | near pairs/row (sphere) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 2 | 1.1e-3 | 1.6e-2 | 3.5e-3 | 7.4e-3 | 1.8e-3 | 2.0e-2 | 610 |
| 8 | 3 | 2.0e-4 | 5.8e-3 | 7.6e-4 | 4.8e-3 | 3.7e-4 | 6.9e-3 | 950 |
| 8 | 4 | 5.2e-5 | 1.1e-3 | 1.9e-4 | 4.7e-4 | 1.2e-4 | 1.6e-3 | 1370 |
| 10 | 2 | 1.0e-3 | 9.8e-3 | 2.0e-3 | 4.3e-3 | 1.4e-3 | 1.0e-2 | 390 |
| 10 | 3 | 1.4e-4 | 2.8e-3 | 3.7e-4 | 1.8e-3 | 2.2e-4 | 3.1e-3 | 610 |
| 10 | 4 | 3.6e-5 | 4.5e-4 | 8.2e-5 | 1.9e-4 | 5.7e-5 | 5.6e-4 | 880 |
| 15 | 2 | 7.4e-4 | 5.5e-3 | 1.6e-3 | 1.1e-3 | 8.7e-4 | 5.0e-3 | 175 |
| 15 | 3 | 1.0e-4 | 1.1e-3 | 2.4e-4 | 5.9e-4 | 1.2e-4 | 1.1e-3 | 270 |
| 15 | 4 | 2.1e-5 | 1.3e-4 | 3.7e-5 | 2.9e-5 | 2.0e-5 | 1.2e-4 | 390 |

- The error does not grow from 1λ to 2λ: it is set by h_g and M, as expected
  for a translation-invariant grid convolution.
- K (∇G) is about one order less accurate than D at the same setting; K sets
  the configuration for PMCHWT, D alone for EFIE.
- Without precorrection both channels are wrong at O(1) (grid near
  interactions, G(0) = 0), so precorrection is mandatory; individual far
  entries are 5–30× less accurate than the action (far D entries 5e-2 at
  M=2), which is why the action error, not entry error, is the admission
  quantity.
- Settings for the targets: 1e-3 on both channels needs M=3 at λ/15 or M=4 at
  λ/8–λ/10; 1e-4 needs M=4 at λ/15 (K 1.2–1.3e-4, D ≈ 2e-5). EFIE alone
  reaches 1e-3 with M=2 at λ/10 and 1e-4 with M=3–4 at λ/10.

Further cases (D / K precorrected errors):

| Grid ppw | M | sphere 4λ (128 rows, 38,880 RWG) | sphere 2λ, εr = 4 medium (mesh λ_in/7.5) | sphere 2λ, complex64 grids |
| --- | --- | --- | --- | --- |
| 10 | 3 | 3.0e-4 / 3.1e-3 | 8.6e-4 / 2.5e-3 | 2.4e-4 / 3.1e-3 |
| 10 | 4 | 8.7e-5 / 6.1e-4 | 2.4e-4 / 5.2e-4 | 5.7e-5 / 5.7e-4 |
| 15 | 3 | 1.4e-4 / 1.0e-3 | 3.0e-4 / 6.1e-4 | 1.3e-4 / 1.1e-3 |
| 15 | 4 | 2.6e-5 / 1.2e-4 | 4.7e-5 / 1.0e-4 | 2.2e-5 / 1.2e-4 |

- 4λ confirms size independence.
- The interior medium (k = 2k₀, grid per interior wavelength) reaches the
  same K accuracy; D is 2–4× worse because this mesh resolves the interior
  wavelength only at 7.5 points (the original near integrals and the moments
  see a coarser mesh relative to the grid).
- complex64 grids and kernels match complex128 to the reported digits down to
  2e-5, so complex64 grids with complex128 precorrection and residuals are
  the default for L-1c.1.


### 8.3 Memory and matvec time

Measured per edge: projection (indices + four float64 weights)
1.08 / 2.56 / 5.00 kB at M = 2 / 3 / 4 (halved with int32 indices and
float32 weights). Small-grid matvecs take 1–9 ms and are overhead-bound.
Padded convolution (one forward FFT, one product, one inverse FFT), cubic
grids on the shared cuda:1:

| Side | Points | complex64 | complex128 |
| --- | --- | --- | --- |
| 256 | 1.7·10⁷ | 11 ms (0.13 GB/array) | 43 ms (0.27 GB) |
| 384 | 5.7·10⁷ | 42 ms (0.45 GB) | 159 ms (0.91 GB) |
| 512 | 1.3·10⁸ | 92 ms (1.07 GB) | 370 ms (2.15 GB) |
| 640 | 2.6·10⁸ | 189 ms (2.10 GB) | over the 19 GB process cap with 3 work arrays |

Action cost: D (EFIE) = 4 forward + 4 inverse transforms ≈ 4 convolutions;
K = 3 + 3 ≈ 3; one PMCHWT region sharing the forward transforms of J and M ≈
11. Retained kernel spectra: G plus three ∇G components (4 arrays).

Projected for whole-box grids (complex64, M = 3; λ/10 grid ⇒ 1e-3 on D,
λ/15 needed for 1e-3 on K):

| Scene box | Grid / padded side | Kernel + 2 work arrays | EFIE matvec | PMCHWT region matvec |
| --- | --- | --- | --- | --- |
| 16λ cube, λ/10 | 163 / 360 | 1.7 GB | ≈ 0.1 s | ≈ 0.3 s |
| 16λ cube, λ/15 | 243 / 486 | 5.5 GB | ≈ 0.35 s | ≈ 0.9 s |
| 32λ box, λ/10 | 323 / 648 | 13 GB | ≈ 0.8 s | ≈ 2.1 s |
| 32λ box, λ/15 | 483 / 972 | 44 GB (c64; full card) | ≈ 2.7 s | ≈ 7 s |
| 64λ × 64λ × 4λ slab, λ/10 | 643×643×43 / 1296²×90 | 7.3 GB | ≈ 0.45 s | ≈ 1.2 s |
| 64λ cube, λ/10 | 643 / 1296 | ≈ 105 GB per array | – | – |

Precorrection storage (L-1c.1) per edge per medium with reciprocal (half)
storage: 16 B per near pair for D and K in complex128 (PMCHWT), 8 B for D only
(EFIE); complex64 halves both. With the measured near pairs per row (mesh
λ/15): PMCHWT ≈ 4.3 kB (λ/15, M=3), 9.8 kB (λ/10, M=3), 14 kB (λ/10, M=4) per
edge per medium. A 16λ PEC cube (5.3·10⁵ RWG, EFIE) at λ/10, M=3 therefore
needs ≈ 2.6 GB of precorrection, 1.4 GB of projection and 1.7 GB of grids.

Consequences for L-1c.1–.3: the ∇G spectra should be replaced by one G
spectrum with spectral derivatives only if that keeps the measured K error
(to be tested, it would cut kernel memory by 4×); dense 32λ boxes at λ/15
need the full card, which favours the sparse multi-body split of §7.3
(per-body grids plus plane-wave coupling) over whole-box grids.


## 9. L-1c.1: public precorrected-FFT PMCHWT backend

### 9.1 Design

- `MoMConfig(operator_backend='pfft', riesz_storage='local_rwg_v1',
  pfft=PFFTConfig(...))`. `PFFTConfig` (`mom/pfft_config.py`, payload
  `pfft_precorrected_config_v1`) holds points per wavelength (per region
  medium), stencil order, grid precision (complex64 default), precorrection
  factor (default order+2 grid spacings), the exact-pair batch, the support-tree
  leaf size used by leaf preconditioners, and the shared admission tolerances.
  Its payload enters `_numerical_identity`, so operator identities are versioned.
- `mom/pfft_system.py`: `PFFTSystemActions` = one exterior `PFFTRegion` over
  all bodies on a common grid (k₀) plus one interior region per body on its
  own grid (k_b), each with a `PFFTPrecorrection`. Precorrection pairs are all
  bases closer than the radius (found with `spatial.BoxTree`); their exact
  D/K entries come from `nearfield.scalar_edge_pair_channels` (the L-1b
  batched near path, equal to the per-block integration to 1e-14), the grid
  entries from `PFFTRegion.grid_entries`; one direction per pair is stored in
  complex128 because D and K are symmetric. Each region applies
  `[K, iωε D; −iωμ D, K]` with forward transforms shared between D and K;
  the adjoint is `conj(Aᵀ conj y)` with the material factors swapped.
- `PFFTSystemOperator` implements the shared surface protocol and reuses the
  hierarchical admission (`HierarchicalOperator.solve`/`residual_report`: the
  compressed dual residual of the pFFT operator and the sampled original rows
  of `ScalarSystemBlocks`), `LocalRWGRiesz`, `rwg_diagonal` scaling, the
  right-preconditioner hook (`PreconditionerContext` gets the support tree and
  exact principal blocks), lifted RHS and field decode of the other fast
  backends.
- `runtime.py` changes are dispatch only: the backend name, the PFFTConfig
  field and default, its identity line, one handle constructor branch, the two
  solve-dispatch tuples, and a fail-closed `snapshot()` (pFFT results are not
  persisted yet). `simulation.py` is untouched.
- Fails closed (tested): direct/fallback stopping, scalings other than
  none/rwg_diagonal, complex64 currents, periodic, tensor/dyadic,
  harmonic-balance and linear-coupled configurations, foreign backend configs,
  point dipoles (dense-only lifting), and persistence.

### 9.2 EFIE interface with S-1 (`sie/open-pec-efie`)

S-1's dense single layer (`single_layer.assemble_single_layer`) keeps
`vector[m,n] = ∫∫ f_m·f_n G` and `scalar_p0[T,S] = ∫∫ G` separate, with
`<f_m, L f_n> = iωμ (vector − divᵀ scalar_p0 div / k²)`. `PFFTRegion` exposes
the same split: `apply_vector_potential(x)` = vector·x,
`apply_scalar_potential(x)` = divᵀ scalar_p0 div·x (the projection carries the
piecewise-constant divergence as its fourth channel), and
`apply_single_layer(x)` = their combination, so `iωμ·apply_single_layer` is a
drop-in for the dense L action; zero-slope boundary slots of open RWG spaces
project to zero weights. Precorrection for EFIE needs only the D channel
(8 B per stored pair). The EFIE wiring waits until both branches are on master.

### 9.3 Sparse multi-body route (L-1c.2/3 sketch)

Per-cluster regions instead of one exterior grid: clusters of nearby bodies
(single-linkage on bounding-sphere gaps below a few wavelengths) each get an
exterior grid on their own box, which already removes the empty volume;
inter-cluster coupling is a single-level plane-wave operator between cluster
bounding spheres (aggregate each cluster's currents into radiation patterns on
an (L+1)×2(L+1) sphere grid, diagonal translation per cluster pair,
disaggregate onto receiving test functions), with exact fall-back cross blocks
for pairs that violate the separation criterion. The VSWF T-matrix outer layer
follows on the same patterns for repeated bodies. Interior grids are unchanged.

### 9.4 Acceptance (cuda:1 lane C, shared GPU; `benchmark/mom_pfft_acceptance.py`)

Setting for all cases: εr = 4 (multi-body: 4 and 2.5 alternating), grid
15 points per wavelength of each region's medium, M = 3, complex64 grids,
precorrection radius 5 h_g + 2 × largest RWG support radius, GMRES rtol 1e-5
on `rwg_diagonal`-scaled coordinates (complex64 grids floor the achievable
residual near 1e-6), compressed-dual budget 1e-4, sampled original-row
budget 3e-2. Reported solves use the exact per-body LU right preconditioner
unless stated. "Original rows" is the worst relative residual of the
converged pFFT solution on the 8 sampled rows of the original operator — the
true residual, which sits at the accelerator error.

| Case | Unknowns | pFFT vs dense LU: far field / currents | Original rows | Optical theorem | Mie far-field error (pFFT / dense) | Prepare / matvec | Retained (pFFT) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sphere 1λ, n=9 (mesh λ/15) | 4,860 | 3.2e-4 / 1.5e-3 | 1.5e-3 | 3.9e-5 | 2.11 % / 2.10 % | 119 s / 8 ms | 0.08 GB |
| Cube 1λ, 10 divisions (λ/10) | 3,600 | 1.2e-3 / 1.6e-3 | 1.2e-3 | 1.7e-3 | – | 90 s / 8 ms | 0.05 GB |
| Two spheres 1λ, gap 1λ | 9,720 | 2.2e-4 / 1.5e-3 | 1.1e-3 | 2.7e-5 | – | 629 s / 23 ms | 0.15 GB |
| Sphere 2λ, n=14 (λ/10.5) | 11,760 | 6.5e-4 / 3.9e-3 | 1.5e-3 | 4.9e-4 | 2.92 % / 2.92 % | 480 s / 15 ms | 0.21 GB |
| Cube 2λ, 14 divisions (λ/7) | 7,056 | 5.2e-3 / 1.1e-2 | 6.1e-3 | 5.9e-3 | – | 628 s / 29 ms | 0.14 GB |

The 2λ sphere and cube rows ran before the support-aware radius (commit
`c792c43e`); with it the 1λ cube improved from 3.9e-3 to 1.2e-3 and the 1λ
sphere from 6.2e-4 to 3.2e-4 in far field.

Refinement and accelerator checks:

- 1λ sphere, mesh h → h/√2 (n 9 → 13, 10,140 unknowns): Mie error 2.11 % →
  1.04 % (second order); far-field change between levels 1.1e-2, i.e. the
  discretisation error dominates the accelerator error by 30×.
- 1λ sphere, M 3 → 4 on the same mesh: far-field change 2.9e-4 (accelerator
  convergence). 1λ cube: M 3 → 4 changes 1.3e-3, h → h/√2 changes 2.0e-2.
- pFFT and dense LU have the same Mie error to three digits at 1λ and 2λ.

Preconditioning (pFFT operator, four εr = 4/2.5 spheres of 1λ, 19,440
unknowns, rtol 1e-5):

| Scene | No right preconditioner (rwg_diagonal) | Leaf blocks (32 edges) | Exact per-body LU |
| --- | --- | --- | --- |
| centres (±2, ±2, 0)λ, extent 5λ | ✗ 1000 its, residual 0.28 | 821 its, 60 s (+ 90 s setup) | 6 its, 13 s (+ 258 s setup) |
| centres (±3.5, ±3.5, 0)λ, extent 8λ | ✗ 1000 its, residual 0.23 | 586 its, 77 s (+ 118 s setup) | 5 its, 19 s (+ 352 s setup) |

The 8λ scene's exterior grid is 250×250×40 padded (80 MB of kernel spectra)
for 19,440 unknowns, with a 60 ms matvec: most of it is empty volume, the case
the sparse split of §9.3 removes. Per-body LU setup is the exact integration
of each body's principal block (the dense cost of one body) plus its LU.

4λ sphere (n = 36, 77,760 unknowns, leaf-64 preconditioner, Mie check): not
completed. The single lane-C hold (29 min on a shared GPU) ended inside
preparation — lifted RHS, 1.1·10⁷ exact precorrection pairs and 608 exact
64-edge principal blocks for the preconditioner — before the first solve.
Dense LU at this size needs the full card in any case. It is rescheduled
for L-1c.2, after the batched RHS lifting (S-perf) lands, as a full-card run.

Findings:

1. The public pFFT backend reproduces dense LU to 2e-4–1.2e-3 in the far
   field on 1–2λ spheres, cubes and a two-body scene at the setting that the
   L-1c.0 action errors predicted, and its Mie error equals the dense one —
   the discretisation, not the accelerator, sets the physical error.
2. The true (original-row) residual of a converged pFFT solution is 1e-3, the
   accelerator level; the admission must therefore set
   `original_sample_tolerance` above it (3e-2 here). Tightening it requires a
   tighter accelerator (M = 4 or a finer grid) — the check is meaningful and
   catches a too-loose accelerator.
3. Accuracy degrades when the grid is much finer than the mesh: the 2λ cube
   with a λ/7 exterior and λ_in/3.5 interior mesh but a λ_in/15 grid gives
   5e-3. The grid spacing should be tied to the mesh (stencil span M·h_g not
   much smaller than the RWG support); L-1c.2 item.
4. Preparation is dominated by the exact precorrection pairs and by the
   lifted RHS / per-body block integration on the shared GPU; the pFFT
   operator itself is 0.05–0.3 GB and its matvec 8–30 ms at these sizes.
5. Iterations: as in the dense L-1 study, without a preconditioner the
   multi-body problem does not converge in 1000 iterations, leaf blocks need
   ≈ 800, and exact per-body inverses need 6 — the L-2/L-3 target is
   unchanged and the right-preconditioner hook works on the fast backend.


## 10. L-1c.2: mesh-tied grid, sparse multi-body route and EFIE on pFFT

### 10.1 Grid spacing tied to the mesh (`PFFTConfig` v2)

Action error against original rows (`benchmark/mom_pfft_accuracy.py
--support-aware`, the public precorrection radius (M+2) h_g + 2 × largest
support radius; 1λ sphere, vacuum, complex128 grids, 384 sampled rows) for two
meshes, mean edge h = λ/15 (n = 9) and λ/30 (n = 18). D and K are the single-
and double-layer channels; "near/row" is the number of precorrected pairs per
row.

| h_g / h | mesh λ/15: ppw, K error M=3 / M=4 | mesh λ/30: ppw, K error M=3 / M=4 | near/row (M=3, λ/15 / λ/30) |
| --- | --- | --- | --- |
| 4 | – | 7.5: 6.4e-3 / 1.2e-3 | – / 5,187 |
| 2 | 7.5: 5.7e-3 / 8.8e-4 | 15: 8.0e-4 / 8.6e-5 | 1,535 / 1,528 |
| 1.5 | 10: 2.2e-3 / 3.5e-4 | 20: 4.4e-4 / 4.1e-5 | 958 / 956 |
| 1 | 15: 5.3e-4 / 6.6e-5 | 30: 2.4e-4 / 2.5e-5 | 516 / 517 |
| 0.75 | 20: 2.0e-4 / 2.2e-5 | 40: 1.7e-4 / 1.6e-5 | 345 / 349 |
| 0.5 | 30: 5.2e-5 / 6.2e-6 | 60: 6.4e-5 / 7.6e-6 | 208 / 212 |
| 0.375 | 40: 2.0e-5 / 2.7e-6 | 80: 2.8e-5 / 3.6e-6 | 156 / 158 |
| 0.25 | 60: 4.2e-5 / 1.6e-6 | – | 108 / – |
| 0.19 | 80: 1.1e-4 / 6.3e-6 | – | 89 / – |

D errors are 10–30× below K at every point. Reading:

1. At fixed h_g / h the error barely depends on the mesh density: it is set
   by how the stencil (span M h_g) resolves the RWG support, not by λ/h_g
   alone. Wavelength resolution still matters when the grid is coarse
   (ratio 2: λ/7.5 gives 5.7e-3, λ/15 gives 8.0e-4).
2. Below h_g ≈ h/3 the error rises again (M = 3: 2.0e-5 → 4.2e-5 → 1.1e-4
   at ratios 0.375 → 0.25 → 0.19) while the grid volume grows as
   (h/h_g)³ — the "grid much finer than the mesh" failure of §9.4 item 3.
3. The precorrection count per row depends only on h_g / h (and M): about
   500 at ratio 1, growing as ratio² above it.

Rule (`SPACING_RULE_VERSION = pfft_mesh_tied_spacing_v1`, `ACCURACY_TABLE`
in `pfft_config.py`): for an operator-action target ε,

| ε | stencil order M | grid | 
| --- | --- | --- |
| 1e-2 | 3 | h_g = min(λ/10, 1.5 h) |
| 1e-3 (default) | 3 | h_g = min(λ/15, h) |
| 1e-4 | 4 | h_g = min(λ/15, h) |

with h the mean RWG edge of the region's bases and λ the region's medium
wavelength, and never below h/4 (`quarter_edge_floor`). Tighter targets are
rejected as not characterised. An explicit `points_per_wavelength` keeps
λ/ppw but fails closed when it is finer than h/4. `PFFTConfig` gains
`target_accuracy`, `exterior_grids`, `cluster_separation`, `coupling_digits`
(payload version `pfft_precorrected_config_v2` with the spacing-rule
version); every region receipt records spacing, mean edge, ratio and the
rule that chose it.

### 10.2 Open-sheet PEC EFIE on pFFT

`operator_backend='pfft'` is admitted for `efie_open_pec` (dispatch only in
`efie_open.py`; logic in `pfft_efie.py`). The handle applies
iωμ (grid D + precorrection) on one exterior grid over all sheets, built
without ∇G spectra (one kernel FFT instead of four). Exact precorrection
entries come from the S-1 face-pair rules (`single_layer_pair_integrals`,
`adaptive_near_single_layer`, the same classification and orders as the dense
assembly), so they equal the dense matrix to 1e-12 on single- and two-sheet
scenes (test). The residual metric is the S-1 RWG Gram applied from local
face blocks with a residual-checked PCG inverse (`SheetRWGRiesz`); RHS and
decode are the dense handle's. Direct LU is rejected; GMRES is required.

1λ square plate, mesh side/cells (`benchmark/mom_pfft_efie.py`, default
`PFFTConfig`: target 1e-3, complex64 grid). RCS at normal incidence, x
polarization, pFFT GMRES rtol 1e-8 against dense LU:

| Cells (unknowns) | Monostatic RCS dense / pFFT [m²] | Δ monostatic / bistatic 30° / σ_s | Δ currents | Grid (rule, h_g/h) | Retained |
| --- | --- | --- | --- | --- | --- |
| 8 (176) | 10.5166 / 10.5187 | 2.0e-4 / 1.6e-4 / 1.6e-4 | 1.7e-4 | 36×36×9 (λ/15, 0.46) | 0.95 MB |
| 16 (736) | 10.9775 / 10.9751 | 2.2e-4 / 5.8e-5 / 1.6e-5 | 6.3e-4 | 40×40×9 (λ/15, 0.93) | 7.0 MB |
| 32 (3008) | – | 1.9e-4 / 1.3e-4 / 1.7e-4 | 5.4e-4 | (mesh rule, h_g = h) | – |

Dense values reproduce the S-1 table (10.517 / 10.977 m²). Iterations,
oblique incidence (0, 0.6, −0.8), GMRES rtol 1e-6, no restart, no diagonal
scaling (the S-1 iteration study); solution error against dense LU:

| Cells (unknowns) | Dense: none / Calderón | pFFT: none / Calderón | pFFT solution error | Calderón setup (dense dual) |
| --- | --- | --- | --- | --- |
| 8 (176) | 56 / 16 | 64 / 17 | 1.3e-4 | 6.7 s |
| 16 (736) | 103 / 18 | 118 / 20 | 6.0e-4 | 36 s |
| 32 (3008) | 188 / 20 (S-1) | 216 / 23 | 5.0e-4 | 319 s |

Findings: RCS agrees with dense S-1 to 2e-4 and currents to 6e-4 (the
accelerator level of the target). With Calderón the counts match dense within
1–3 iterations and stay flat under refinement. Without a preconditioner pFFT
needs about 15 % more iterations: the accelerator perturbs an operator whose
condition grows as 1/h. Known limit: the Calderón setup (dense dual EFIE on the
barycentric refinement plus a dense LU) is 319 s at 3008 unknowns against
0.4 s for its 23-iteration solve, so it does not scale with the pFFT operator.
A pFFT dual operator on the barycentric mesh is the follow-up. The dense
multi-sheet assembly defect found here (empty cross-pair chunks) was fixed on
master (7d54c1d3); the multi-sheet pFFT test now compares against that handle.

### 10.3 Sparse multi-body route: per-cluster grids plus plane-wave coupling

`PFFTConfig(exterior_grids='clusters')` (`pfft_coupling.py`):

- Bodies are merged into clusters until every pair of cluster bounding
  spheres has |X_ab| ≥ `cluster_separation` (default 2) × (R_a + R_b).
- Each cluster gets its own exterior `PFFTRegion` and precorrection, so the
  empty volume between clusters costs no grid.
- Clusters are coupled by the single-level plane-wave (diagonal translation)
  expansion of G. Truncation follows L = kd + 1.8 d0^{2/3} (kd)^{1/3}, with
  d the largest radius sum and d0 = 1 − log10(target).
- The direction rule is (L+1) Gauss × (2L+2) azimuths. It is symmetric under
  s → −s, so the coupled D and K stay symmetric and the adjoint is unchanged.
- Each RWG function stores transmit patterns ∫ f e^{−ik s·(r−c)} and the
  matching divergence pattern (N × Q × 4, grid precision). The receive
  patterns are their conjugates; a lossless exterior is required.
- Interior grids stay per body. A tested translation reproduces G to 1e-5 for
  |d|/|X| ≤ 0.43.

The new acceptance driver is `benchmark/mom_pfft_sparse.py`. Setup:

- Bodies are 1λ spheres with εr 4 / 2.5 alternating.
- Grids are complex64 with target 1e-3 (h_g = min(λ/15, h)).
- 256 random rows of the original operator are integrated exactly (J and M
  rows), and A x is compared on them for a random x.
- Solves use the exact per-body LU right preconditioner with GMRES rtol 1e-5.
- The far field is compared with the independent multi-sphere T-matrix
  reference `benchmark/mom_multisphere_reference.py`. That reference uses
  Mie T-matrices, with outgoing-to-regular translations obtained by
  projecting VSWFs on each sphere. It reproduces the single-sphere Mie
  series to 7e-15, its curl identity holds to 1e-9, and its two-sphere far
  field changes by 1e-13 from order 17 to 21.

| Scene (unknowns) | Exterior: single grid | Exterior: clusters | Action error vs original rows (single / clusters) | Clusters vs single | Its (per-body LU) | Far field vs T-matrix |
| --- | --- | --- | --- | --- | --- | --- |
| 4 spheres, (±2, ±2, 0)λ, n=9 (19,440) | 160×160×40, 0.20 GB, 20 ms | 4×40³ + L=15, 0.33 GB, 42 ms | 4.0e-5 / 3.8e-5 | 1.3e-5 | 6 | 1.51 % |
| 4 spheres, (±3.5, ±3.5, 0)λ, n=9 (19,440) | 250×250×40, 0.25 GB, 29 ms | same as above, 0.33 GB, 39 ms | 3.9e-5 / 3.8e-5 | 7.8e-6 | 5 | 1.57 % |
| 8 spheres at the corners of a 16λ box, n=7 (23,520) | 500³, 4.15 GB, 2.5 s; prepare 1021 s, peak 8.6 GB | 8×40³ + L=15, 0.36 GB, 93 ms; prepare 512 s (with LU), peak 4.2 GB | 4.6e-5 / 4.6e-5 | 6.0e-6 | 4 | 2.29 % |
| 8 spheres at the corners of a 32λ box, n=7 (23,520) | 1000³ estimated: 32 GB of kernel spectra + 8 GB per work grid (does not fit) | 0.36 GB, 80 ms | – / 4.6e-5 | – | 4 | 2.33 % |

Exterior bytes are kernel spectra, projection and precorrection; matvec
times are for the full PMCHWT action including the interiors. Total retained
operator memory (exterior plus the four or eight interior grids) is
0.33 / 0.37 / 4.32 GB for the single grid against 0.46 / 0.46 / 0.53 / 0.53 GB
for clusters.

Findings:

1. Clusters reproduce the single-grid action to 6e-6–1.3e-5, far below the
   1e-3 target. Both sit at 4e-5 against exact original rows, so the coupling
   adds no visible error at the default digits.
2. Clustered memory and matvec time do not depend on the box size: 16λ and
   32λ cost the same.
   - The single 16λ grid costs 8× the memory and 27× the matvec time.
   - The single 32λ grid does not fit the card.
   - The single 16λ grid first ran out of memory at the 18 GB lane cap while
     sampling G in float64. Kernel samples are now evaluated by x-slabs
     (bitwise-identical spectra), and its peak is 8.6 GB.
3. For flat, moderately spaced scenes (the 4-sphere 5λ/8λ cases, z extent
   1λ) the single grid is already thin. There, the stored N×Q×4 patterns
   (0.16 GB at Q = 512) make clusters 1.4× larger and 2× slower per matvec.
   This is the single-level memory limit; the multilevel version will
   aggregate patterns from children instead of storing them per RWG
   function.
4. With exact per-body LU, iterations are 4–6 on every scene. The
   L-2/L-3 target is unchanged.
5. The far field differs from the T-matrix reference by 1.5 % (n=9, h≈λ/15)
   and 2.3 % (n=7). This is the mesh discretisation error of the icosphere
   (the single-sphere Mie error is 2.1 % at n=9 in §9.4). The pFFT-vs-dense
   check on the same mesh is in §10.4.
6. The optical-theorem defect printed by the driver is not meaningful for
   these scenes: its 24 × 48 angular rule does not resolve the far-field
   interference of 8–32λ arrays.

### 10.4 Dense LU reference on the sparse route

The four spheres at (±2, ±2, 0)λ with n = 7 (11,760 unknowns) were solved
three ways: dense LU, single-grid pFFT and clustered pFFT (per-body LU,
6 iterations each). Each variant ran as its own resumable chunk
(`--stage`), each lane-C hold staying under 14 min.

| Far field | vs dense LU | vs T-matrix |
| --- | --- | --- |
| Dense LU | – | 2.46 % |
| Single-grid pFFT | 1.5e-4 | 2.47 % |
| Clustered pFFT | 1.5e-4 (6.9e-5 vs single grid) | 2.47 % |

The accelerator error (1.5e-4) is 160× below the mesh error, and all three
share the same T-matrix error. The independent reference therefore checks
the PMCHWT discretisation, and dense LU checks the accelerator. Dense
preparation took 731 s against 247 s for the single grid and 310 s for
clusters (with per-body LU setup).

### 10.5 Status and limits

- Done:
  - Mesh-tied grid with fail-closed explicit grids.
  - Clustered exterior with plane-wave coupling.
  - EFIE on pFFT (single and multi sheet).
  - Multi-sphere T-matrix reference.
  - Records: `docs/assessments/sie-pfft-l1c2-2026-10-29/`.
- Limits:
  - Patterns are stored per RWG function (single level).
  - Clusters need a lossless exterior and pairwise separation of at least
    1.5 (default 2) times the radius sum. Closer bodies are merged into one
    grid rather than coupled.
  - The Calderón setup stays dense.
  - Persistence fails closed as before.
- Precorrection routing on sliver and thin-slab meshes:
  - The selected-block near path (`nearfield._scalar_regional_accumulate`,
    `scalar_edge_pair_channels`) now sends high-aspect touching pairs to
    `_anisotropic_touching_integrals`, as the dense assembly does after
    S-1c/S-1d. The pFFT precorrection and the hierarchical near blocks
    therefore reproduce dense entries to 1e-12.
  - Checked by a test on a rounded cube with 25:1 arc strips and on a 1 cm
    slab.
- The 4λ sphere (77,760 unknowns) needs one long full-card hold. Its
  preparation cannot be chunked because pFFT operators have no snapshot.
  - A 14-minute full-card probe (all three cuda:1 locks, 43.5 GB cap,
    S-perf merged) ended inside `prepare()`.
  - Measured on lane C, precorrection alone takes about 14 min of exact
    integration (10.2 M exterior pairs at 26 k pairs/s, 4.2 M interior
    pairs at 13 k pairs/s), plus about 2 min of grid entries.
  - On top of that come the lifted RHS, the leaf-64 setup and the solve.
    The estimated total is 35–45 min.
  - Command: `python -m benchmark.mom_pfft_acceptance --shape sphere --size 4
    --frequency-mesh 36 --mie --solve-preconditioner leaf:64
    --memory-fraction 0.88`.
