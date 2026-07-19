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
