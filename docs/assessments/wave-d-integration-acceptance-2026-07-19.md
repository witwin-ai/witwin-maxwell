# Wave-D Integration Acceptance (2026-07-19)

Track: plan-13 Phase 3 electrostatic pre-bias slice + cross-feature end-to-end.
Environment: conda env `maxwell`, single GPU `CUDA_VISIBLE_DEVICES=0` (2x RTX A6000
host). All commands below prepend:

```bash
export CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
```

## Delivered

1. `ElectrostaticInitialCondition` (`witwin/maxwell/electrostatic/initial_condition.py`,
   re-exported from `witwin.maxwell`). `from_result(dc_result, tolerance=1e-3)` wraps a
   `Result(method="electrostatic")`; `Simulation.fdtd(scene, ..., initial_condition=...)`
   consumes it and seeds the staggered E buffers before step 0 (H and CPML memory left
   zero).
2. Yee mapping (verified against the repo's actual layout — see "Grid finding"):
   interpolate the cell-centred DC potential onto the primary nodes, then take Yee edge
   differences `E = -grad(phi)`. Produces exactly the native FDTD component shapes and a
   field whose discrete Yee curl is exactly zero, so a lossless interior cell begins in a
   discrete FDTD steady state.
3. Discrete-Gauss consistency gate: `gauss_residual` (Yee divergence of `eps_fdtd*E_init`
   vs the electrostatic free-charge density at shared interior nodes, conductor-adjacent
   nodes excluded, normalised by the peak displacement-flux scale). Injection fails closed
   (`ValueError`) when it exceeds `tolerance`.
4. Grid-identity guard: the electrostatic cell-centre grid must match the FDTD scene nodes;
   mismatch (including PML grid extension) fails closed.
5. Runtime injection at FDTD prepare (`Simulation._apply_initial_condition`) plus fail-closed
   capability guards (`Simulation._validate_initial_condition_support`): non-FDTD method,
   distributed/multi-GPU, trainable/adjoint, and Bloch (complex-field) runs are rejected.
6. `Result.electrostatic_prebias` provenance accessor; the pre-bias provenance (DC
   diagnostics, terminal charges, mapped `gauss_residual`, tolerance) is merged into the run
   metadata.
7. `ElectrostaticResultData.conductor_mask` added (union of pinned terminal cells) so the
   Gauss check can exclude induced-surface-charge nodes.
8. Tests, `FEATURE_LIST.md` subsection, capability-guard census reconciliation (172 -> 176).

## Grid finding (verified, not assumed)

The repo's Yee layout (`fdtd/coords.py`, `fdtd/runtime/stepping.py init_field`) places
`Ex` at `(x_half, y_node, z_node)` with shape `(Nx-1, Ny, Nz)` (and cyclically for
Ey/Ez). The electrostatic solver's potential is cell-centred at `(x_half, y_half, z_half)`,
shape `(Nx-1, Ny-1, Nz-1)`. The two grids are offset by half a cell in the transverse
axes, so the supervisor's "difference adjacent cells -> exactly the Yee edge" premise does
**not** hold as-is: differencing cell-centred phi along x lands on the `Hx` grid, not `Ex`.

The correct construction that is exact by a discrete identity: interpolate phi onto the
primary **nodes** (`x_nodes64`, ...), then take Yee edge differences. The result sits
exactly on the Yee E locations, and the discrete curl of a discrete gradient is identically
zero (`d.d = 0`), so H stays zero and the interior E is frozen. Boundary nodes use linear
**extrapolation** (not clamping) so a linear potential recovers the electrode value exactly
and a uniform field maps with zero Gauss residual. Verified numerically: discrete curl of
the mapped field is ~1e-15; a uniform 2 V/m field is recovered exactly; a PEC-plate pre-bias
holds bit-exactly (interior drift 0.0) over 300 no-source steps.

Compatible boundaries are the grid-non-extending ones: electrostatic `BoundarySpec.none()`
pairs with FDTD PEC/PMC/periodic. A full PEC box clips a uniform field's tangential
component at the side walls, so the validated exact-steady-state geometry uses PEC-material
plate structures with periodic side boundaries (documented in the test). PML extends the
grid and is rejected by the grid-identity guard.

## Test inventory

`tests/electrostatic/test_initial_condition.py` — 15 tests (12 CPU, 3 CUDA):
- construction/provenance (3), mapping shape/curl-free/uniform recovery (3), grid-mismatch +
  PML rejection (2), fail-closed guards non-IC-type/Bloch/distributed/trainable (4),
  CUDA: steady-state hold, checkerboard-corruption falsification, Gauss-gate fail-closed (3).

`tests/esd/test_prebias.py` — 2 CUDA tests: cross-feature end-to-end (pre-bias + IEC
61000-4-2 ESD terminal injection + `DielectricBreakdown` + `BreakdownMonitor`) with
provenance/stress/event-log/ESD-record assertions; pre-bias-shifts-initial-transient sanity.

Pass counts (this GPU):

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n maxwell --no-capture-output \
  python -m pytest tests/electrostatic/test_initial_condition.py tests/esd/test_prebias.py -q
# 17 passed

CUDA_VISIBLE_DEVICES=0 conda run -n maxwell --no-capture-output \
  python -m pytest tests/electrostatic tests/esd tests/breakdown \
    tests/api/public/test_public_api.py tests/api/public/test_simulation_smoke.py -q
# 187 passed

CUDA_VISIBLE_DEVICES=0 conda run -n maxwell --no-capture-output \
  python -m pytest tests/api/public/test_guard_census.py -q
# 3 passed  (CAPABILITY_GUARD_BUDGET 172 -> 176)
```

## Falsifications recorded

1. **Curl-free / steady-state gate.** Injected a transverse x-ramp on `Ez` in
   `_node_edge_fields` (breaks the discrete-gradient property).
   `test_mapped_field_has_zero_discrete_yee_curl` went red (`curl_y max = 0.80` vs tol
   `4.4e-08`) and `test_prebias_charged_plates_are_an_fdtd_steady_state` went red. Reverted;
   both green.
2. **Discrete-Gauss gate.** Made `_gauss_residual` return `0.0` unconditionally.
   `test_gauss_residual_gate_fails_closed` went red ("DID NOT RAISE ValueError"). Reverted;
   green.
3. **Steady-state discrimination (permanent test).** `test_non_curlfree_seed_is_not_a_steady_state`
   injects a checkerboard-corrupted seed (max non-gradient content) and asserts interior
   drift `> 1e-2 * peak` (observed ~0.40), while the true seed holds at drift 0.0 — proving
   the steady-state assertion is load-bearing, not vacuously true.

## Known gaps / deferred (honest scope)

- **Circuit / MNA-SPICE ESD co-simulation through the TerminalPort is UNTESTED here.** The
  e2e run uses ideal (prescribed) ESD current injection (the Phase-1 stress-only path). The
  plan-04 circuit machinery may already couple through a TerminalPort, but this slice does
  not demonstrate source-impedance / gun-network co-simulation; recorded as untested, not
  delivered.
- **Subcycled electrostatic/transient coupling** is out of scope (plan §4.4 "first version
  does not subcycle"); a later phase must derive its own energy contract.
- **Non-uniform / dielectric-interface pre-bias** carries a genuinely larger `gauss_residual`
  (the electrostatic harmonic-face vs FDTD edge-averaged permittivity differ, plus fringing).
  The e2e run uses `tolerance=1.0`; such fields are still valid curl-free steady states of the
  source-free system but are not tightly Gauss-consistent on the half-cell-offset grids. This
  is reported, never silently accepted.
- **H pre-bias is always zero** (electrostatics carries no magnetic pre-bias).
- Distributed/trainable/Bloch pre-bias fail closed (census-tracked capability gaps).

## Files added / changed

- add `witwin/maxwell/electrostatic/initial_condition.py`
- edit `witwin/maxwell/electrostatic/__init__.py`, `witwin/maxwell/__init__.py` (exports)
- edit `witwin/maxwell/electrostatic/runtime.py` (`conductor_mask` on `ElectrostaticResultData`)
- edit `witwin/maxwell/simulation.py` (`FDTDConfig.initial_condition`, `Simulation.fdtd(...)`
  param, `_validate_initial_condition_support`, `_apply_initial_condition`, provenance merge)
- edit `witwin/maxwell/result.py` (`Result.electrostatic_prebias`)
- add `tests/electrostatic/test_initial_condition.py`, `tests/esd/test_prebias.py`
- edit `FEATURE_LIST.md`, `docs/reference/fdtd-capability-guard-census.md`,
  `tests/api/public/test_guard_census.py` (budget 172 -> 176)
