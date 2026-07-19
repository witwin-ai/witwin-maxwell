# Track A12 — Electrostatics Phase 0+1 acceptance (2026-07-19)

Stage A1 of `a12-electrostatics`: API objects, scene collection, compiler,
matrix-free FVM operator, PCG solver, Dirichlet/Neumann boundaries,
`ElectrostaticResultData`, analytic + convergence + conservation tests.

Worktree: `.worktrees/wd-a12-electrostatics` (branch `fable/electrostatics`).
Env: conda `maxwell`, `CUDA_VISIBLE_DEVICES=1`, 1x RTX A6000.

All commands assume:

```bash
export CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=$PWD   # worktree root
export CUDA_VISIBLE_DEVICES=1
conda run -n maxwell --no-capture-output python -m pytest <targets> -q
```

## Delivered items

- Public API (`witwin/maxwell/electrostatic/api.py`): `ElectrostaticTerminal`,
  `ChargeDensity`, `ElectrostaticBoundarySpec`, `ElectrostaticSolverConfig`;
  re-exported from `witwin/maxwell/__init__.py` and listed in `__all__`.
- Scene collection: `Scene.electrostatic_terminals`, `Scene.charge_densities`,
  `Scene.add_electrostatic_terminal(...)`, `Scene.add_charge_density(...)`,
  `Scene.compile_electrostatics(...)`; carried through `clone()` and
  `PreparedScene`. Terminals are kept out of `Scene.ports`.
- Compiler (`witwin/maxwell/compiler/electrostatic.py`): cell-centred geometry
  (centres, widths, centre-distances, cell volume), harmonic-mean-ready
  `epsilon_r`, free-charge integration to Coulombs, terminal cell masks, and the
  `CompiledElectrostatics` block.
- Runtime (`witwin/maxwell/electrostatic/runtime.py`): matrix-free FVM Laplacian
  `-div(eps grad phi)` (harmonic-mean face permittivity, half-cell Dirichlet
  ghost faces, symmetric projection for interior conductors), float64 Jacobi-PCG,
  `ElectrostaticResultData`, and the `ElectrostaticSimulation` runner.
- `Simulation.electrostatic(...)` classmethod and `Result.electrostatic`
  accessor (`Result(method="electrostatic")`).
- Guard census reconciled `144 -> 152` (+8 electrostatic capability guards),
  documented in `tests/api/public/test_guard_census.py` and
  `docs/reference/fdtd-capability-guard-census.md`.
- `FEATURE_LIST.md`: new "Electrostatics (experimental)" section.

## Test inventory

New suite `tests/electrostatic/` (26 tests, all pass):

```
conda run -n maxwell --no-capture-output python -m pytest tests/electrostatic -q
# 26 passed
```

- `test_api.py` (18): exports, terminal/boundary/solver validation, scene
  collection separation from ports, clone, result accessor, and fail-closed
  guards (PML boundary, floating charge, pure-Neumann, thin conductor swallowed,
  overlapping terminals, dispersive material).
- `test_analytic.py` (5): parallel-plate (linear potential to `< 1e-9`, C to
  `< 1e-6`), concentric spheres (C to `< 6%`, radial `1/r` to `< 0.05`),
  three-level sphere convergence (monotone), coaxial cylinder (C to `< 6%`,
  monotone convergence), dielectric fill scales C by eps_r to `< 1e-3`.
- `test_conservation.py` (6): free-cell Gauss residual at solver floor,
  terminal charge conservation to `< 1e-9`, Poisson volume-charge Gauss closure
  to `< 1e-9`, energy identity `0.5 int(E.D) = 0.5 sum(VQ)` to `< 1e-9`, and
  operator-energy vs cell-integral agreement to `< 5%`.

### Recorded metrics (reproducible from the tests / scratch scripts)

- Parallel plate (Dirichlet z, Neumann sides, 40x40x20): max potential error
  `2.4e-14`; C from energy matches `eps0 A / d` to machine precision.
- Concentric spheres a=0.2, b=0.8 (C_analytic = 2.9671e-11 F): relative C error
  9.1% (n=40) -> 6.9% (n=60) -> 4.6% (n=80); Q_inner + Q_outer ~ 1e-22;
  energy identity to `1e-16`.
- Coaxial a=0.2, b=0.8 (C'_analytic = 4.0130e-11 F/m): 5.1% (n=48) -> 4.4%
  (n=72) -> 3.8% (n=96).
- Poisson uniform charge in grounded box: Q_free + Q_boundary = 0 to `7e-14`
  relative.
- Dielectric fill eps_r=4: C ratio 4.00000.

Adjacent suites (no regressions):

```
conda run -n maxwell --no-capture-output python -m pytest \
  tests/api/public/test_public_api.py tests/api/public/test_simulation_smoke.py \
  tests/api/public/test_guard_census.py tests/api/public/test_dependency_contract.py \
  tests/core/scene/test_scene.py tests/materials/compiler/test_material_compiler.py -q
# all passed (guard census green at budget 152)
```

## Falsifications performed

Each edit was applied to a scratch copy, the target test was shown red, then the
code was restored and re-run green.

1. Energy identity gate — `field_energy` first term factor `0.5 -> 0.6`
   (`electrostatic/runtime.py`). Result: `test_energy_identity_field_vs_terminal_work`
   FAILED (AssertionError). Restored -> green.
2. Dirichlet boundary / parallel-plate gate — Dirichlet ghost distance
   `half_cell * 0.5 -> half_cell * 1.0`. Result:
   `test_parallel_plate_linear_potential_and_capacitance` FAILED. Restored -> green.
3. FVM operator / sphere analytic gate — interior x-face conductance scaled `x2`.
   Result: `test_concentric_spheres_potential_and_capacitance` FAILED.
   Restored -> `tests/electrostatic/test_analytic.py` 5 passed.

## Known gaps / deferred (fail closed, for later stages)

- Floating conductor with prescribed charge (linear-superposition solve) — A2.
- N-terminal capacitance matrix / `CapacitanceData` — A2.
- Pure-Neumann gauge fixing + charge compatibility — A2.
- Implicit-diff backward (`torch.autograd.Function`) and FD gradient gates — A3.
- Tensor/anisotropic permittivity, open boundary, multi-GPU — out of program scope
  (rejected at prepare).
- Dispersive materials require an explicit static permittivity API — not yet
  exposed (currently rejected).

## Design notes for the next agent (A2/A3)

- Terminal charge is defined once as `reaction = apply_full(phi) - rhs_full()`
  summed over the terminal cell mask; this single definition backs Gauss closure,
  energy, and (later) the C-matrix. Reuse it.
- The reduced free-cell system `A_free(x) = free * apply_full(free * x)` is SPD;
  the same operator/preconditioner serves the A3 adjoint solve `A^T lambda = ...`.
- `ElectrostaticOperator.apply_full` is out-of-place (cat-based divergence) so it
  is autograd-safe for A3 implicit differentiation; the forward PCG runs under
  `no_grad`.
- Floating-charge superposition (A2) should reuse `ElectrostaticOperator` and
  solve the V=1/all-off and V=0/sources-on systems, then pick alpha to match the
  prescribed induced charge (charges are linear in the floating potential).

---

# Track A12 — Electrostatics Phase 2+3 acceptance (2026-07-19, stage A2)

Stage A2 of `a12-electrostatics`: floating-conductor linear superposition,
pure-Neumann gauge handling, N-terminal Maxwell capacitance matrix
(`Simulation.capacitance` + `CapacitanceData`), and matrix-property tests. Same
worktree/env/command preamble as A1 above.

## Delivered items

- Floating conductors (`ElectrostaticTerminal(..., charge=)`) resolved by exact
  linear superposition in `electrostatic/runtime.py`: one base solve (fixed
  electrodes at their potentials, floating conductors grounded, sources/boundary
  on) plus one unit solve per floating conductor (conductor at 1 V, everything
  else homogeneous), then a tiny dense `k x k` charge-constraint solve
  (`M alpha = q_target - q_base`, `M_ij` = induced charge on floating `i` from a
  unit potential on floating `j`). Reusable helpers extracted:
  `_reduced_solve` (Dirichlet-reduced SPD solve), `_pinned_value`,
  `_terminal_charges`, `_solve_floating_superposition`.
- Gauge handling: an insulated all-floating problem is charge-compatibility
  checked (prescribed charges + free charge must sum to zero, else `ValueError`)
  and gauge-fixed by `mean(phi)=0`; the unreachable duplicate pure-Neumann guard
  from the A1 draft was removed (see census reconciliation).
- Capacitance extraction (`electrostatic/capacitance.py`): `CapacitanceData`
  (`matrix`, `terminal_order`, `reference`, `charges`, `energy`,
  `reciprocity_error`, `row_sum_error`, plus `capacitance`, `mutual_capacitance`,
  `capacitance_to_reference`, `two_terminal_capacitance` accessors) and the
  `CapacitanceSimulation` runner. Raw matrix; NO silent symmetrization.
- `Simulation.capacitance(scene, *, terminals=, reference=, boundary=, solver=)`
  classmethod and `Result.capacitance` accessor (`Result(method="capacitance")`).
- Exports: `CapacitanceData` added to `witwin/maxwell/electrostatic/__init__.py`,
  `witwin/maxwell/__init__.py`, and `mw.__all__`.

## Test inventory

New suite total `tests/electrostatic/` = 39 tests, all pass:

```
conda run -n maxwell --no-capture-output python -m pytest tests/electrostatic -q
# 39 passed
```

- `test_floating.py` (5): floating sphere holds its prescribed charge (rel
  `< 1e-8`), floating conductor is exactly equipotential (spread/level `< 1e-9`)
  and floats to an intermediate level, charge-neutral midplane slab floats to
  0.5 V between 1 V/0 V plates (`< 5e-3`), two floating conductors each keep their
  own charge (rel `< 1e-7`), and a charged isolated conductor in a Neumann box is
  rejected as incompatible (`ValueError`).
- `test_capacitance.py` (9): sphere-in-grounded-shell analytic C (rel `< 6%`) with
  monotone grid convergence (n=40/60/80), 3-terminal matrix symmetry
  (`reciprocity_error < 1e-6`) + positive diagonal + non-positive off-diagonal,
  `0.5 V^T C V` vs field energy (rel `< 1e-6`), terminal-reordering invariance
  (`< 1e-9`), insulating-boundary row-sum conservation (`row_sum_error < 1e-6`),
  two-terminal/`capacitance_to_reference` consistency, no-return-path rejection,
  and the `result.capacitance` accessor guard.
- `test_api.py` (17): A1 guards retained; the A1 "floating rejected in this stage"
  test was removed (floating is now implemented) and `CapacitanceData` added to
  the export check.

Adjacent suites (no regressions):

```
conda run -n maxwell --no-capture-output python -m pytest \
  tests/electrostatic tests/api/public/test_public_api.py \
  tests/api/public/test_simulation_smoke.py tests/api/public/test_guard_census.py \
  tests/api/public/test_dependency_contract.py tests/core/scene/test_scene.py -q
# 106 passed (guard census green, budget ceiling 152, measured 151)
```

### Recorded metrics

- Sphere-in-shell a=0.2, b=0.8 (C_analytic = 2.9671e-11 F) via
  `Simulation.capacitance`: C(inner,inner) rel error `< 6%` at n=80, monotone
  decreasing across n=40/60/80.
- 3-terminal grounded-box matrix (three spheres): `reciprocity_error < 1e-6`,
  diagonal `> 0`, off-diagonal `<= 1e-18`; `0.5 V^T C V` matches field energy to
  rel `< 1e-6` at V=(1,2,-1).
- Row-sum conservation (Neumann box, explicit reference): `row_sum_error < 1e-6`.
- Floating midplane slab floats to 0.5 V to `< 5e-3` (grid-symmetric mask);
  floating sphere charge reproduced to rel `< 1e-8`.

## Falsifications performed

Each break was applied in place, the target test shown red, then reverted and
re-run green. (F1 used a scratch break restored via git; F2/F3 used in-place edit
then edit-restore — never `git checkout` on uncommitted work.)

1. Floating superposition reconstruction — `phi = phi + alpha[j]*phi_units[j]`
   -> `phi - alpha[j]*phi_units[j]` (`electrostatic/runtime.py`). Result:
   `test_floating_conductor_prescribed_charge_conservation` FAILED. Restored.
2. Capacitance charge sign — `reaction = apply_full(phi) - b_full` ->
   `b_full - apply_full(phi)` (`electrostatic/capacitance.py`). Result:
   `test_three_terminal_matrix_symmetry_and_signs` FAILED (diagonal went
   negative). Restored.
3. Row-sum reference inclusion — uniform excitation `{name: 1.0 for name in
   terminals}` -> `... for name in active` (reference dropped). Result:
   `test_row_sum_charge_conservation_under_insulating_boundary` FAILED. Restored.

Final state after restores: `tests/electrostatic` 39 passed.

## Guard census reconciliation (Phase 2+3)

Floating superposition + gauge handling implement the previously deferred
floating-conductor `NotImplementedError`, which is removed. Net measured
capability guards `152 -> 151`; `CAPABILITY_GUARD_BUDGET` is a ceiling and stays
`152` (`151 <= 152`). New incompatibility/return-path failures use `ValueError`
(not capability guards). Documented in `test_guard_census.py` comment and
`docs/reference/fdtd-capability-guard-census.md`.

## Known gaps / deferred (for A3)

- Implicit-diff backward (`torch.autograd.Function`) + FD gradient gates on
  `d(energy)/d(eps_region)` and `dC_ij/d(eps)` — A3.
- Prepare-time rejections for dispersive-without-static-value / tensor-eps /
  multi-GPU hardening pass — A3 (mostly already fail closed from A1).
- Tensor/anisotropic eps, open boundary, multi-GPU — out of program scope.

## Design notes for A3

- The capacitance/floating solves all route through `_reduced_solve`, whose
  operator `A_free(x) = free * apply_full(free * x)` is SPD — reuse it verbatim
  for the adjoint `A^T lambda = grad_phi` (A is symmetric).
- The `k x k` floating solve runs on CPU float64 (`torch.linalg.lstsq`, gelsd,
  `rcond=1e-10`) as a tiny control-plane reduction; it is not a field-solver CPU
  fallback. For A3 differentiability, the alpha reconstruction is linear in the
  unit solves, so gradients flow through the superposition weights.
- Capacitance ignores volumetric free charge by design (uses
  `operator.rhs_boundary()`), matching the Maxwell-matrix definition (pure
  conductor response).
