# Track D13 — deterministic dynamic dielectric breakdown: D1 (runtime) acceptance

Date: 2026-07-19
Stage: D1 (runtime) of track `d13-breakdown` (plan 13 Phase 4, deterministic
field-duration/latching dynamic breakdown).
Worktree: `.worktrees/wd-d13-breakdown` (branch `fable/breakdown`).
Environment: conda env `maxwell`, `CUDA_VISIBLE_DEVICES=1`, `PYTHONPATH=<worktree>`.

## Delivered items

Public API and data types:
- `DielectricBreakdown(critical_field, post_breakdown_conductivity, minimum_duration=0.0, model="field_duration", state="latching", recovery=None, damage_parameters=None, ramp_time=None, default_ramp_steps=10)` dataclass in `media.py`, composed into `Material(breakdown=...)`. v1 accepts exactly `model="field_duration"` + `state="latching"`; other model/state and the reserved `recovery`/`damage_parameters` fail closed at construction.
- `BreakdownEvent` and `BreakdownResultData` typed containers in new `witwin/maxwell/breakdown.py`, exported from the package root; state codes (`intact=0`, `conducting=2`, reserved `recovering=1`/`failed=3`).

Compiler:
- `witwin/maxwell/compiler/breakdown.py`: `compile_breakdown_layout(scene)` stamps per-node breakdown parameters on the `(Nx,Ny,Nz)` material grid (staircase occupancy threshold `>=0.5`, last-writer-wins), and `scene_has_breakdown(scene)`.

Runtime:
- `witwin/maxwell/fdtd/runtime/breakdown.py`: compiled capable-node and capable-edge index sets; per-cell state machine (energy-consistent cell-center `|E|` colocation, contiguous-exceedance timer, latching trigger); linear conductivity ramp; in-place semi-implicit decay/curl scatter on capable edges reconstructed exactly from stored base coefficients (recovered external PML/PEC factor); dedicated breakdown-dissipation energy channel `∫ σ_breakdown·|E|² dV dt` accumulated on the conduction edges and scattered to nodes; bounded preallocated per-node event buffer (capacity = capable-cell count) with a defensive overflow hard-error; host transfer only at run end. All per-step work is GPU-resident with **no per-step host synchronization**.
- Step-loop integration in `fdtd/runtime/stepping.py`: `initialize_breakdown_runtime` after `build_update_coefficients`; `advance_breakdown_state(solver, n)` once per step after the full E update (sources + PEC/Mur/DFT tail); CUDA-graph capture forced off when breakdown is enabled.
- Result assembly in `simulation.py` (`_run_fdtd_from_solver`) via `finalize_breakdown_data`; `Result.breakdown` / `Result.breakdown_events` accessors in `result.py`.

Prepare-time guards:
- `Material` construction rejects breakdown on PEC / anisotropic-tensor / dispersive / instantaneous-nonlinear / time-modulated / 2D-sheet media.
- `initialize_breakdown_runtime` rejects complex (Bloch) fields and scenes mixing breakdown with dispersive / nonlinear / full-anisotropic / modulated / gyromagnetic media.
- `Simulation._validate_breakdown_support` rejects FDFD + breakdown and trainable + breakdown (non-differentiable hard switch; smooth surrogate deferred).
- `DistributedFDTD._validate_static_capabilities` rejects multi-GPU + breakdown.
- Prepare-time accuracy warning when `0.5·σ·dt/eps > 10` (PEC-like reflector).

## Files added / changed

Added: `witwin/maxwell/breakdown.py`, `witwin/maxwell/compiler/breakdown.py`, `witwin/maxwell/fdtd/runtime/breakdown.py`, this doc.
Changed: `witwin/maxwell/media.py`, `witwin/maxwell/__init__.py`, `witwin/maxwell/result.py`, `witwin/maxwell/simulation.py`, `witwin/maxwell/fdtd/runtime/stepping.py`, `witwin/maxwell/fdtd/distributed/solver.py`, `FEATURE_LIST.md`, `docs/reference/fdtd-capability-guard-census.md`, `tests/api/public/test_guard_census.py`.

## Verification (exact commands)

All commands prefixed with:
```
export CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=<worktree>
export CUDA_VISIBLE_DEVICES=1
```

Test suites (all passed):
- `pytest tests/api/public/test_public_api.py tests/api/public/test_simulation_smoke.py` → 27 passed
- `pytest tests/api/public/test_guard_census.py` → 3 passed (budget raised 144 → 159, reconciled)
- `pytest tests/materials/compiler/test_material_compiler.py tests/core/scene/test_scene.py` → 78 passed
- `pytest tests/materials/conductive/` → 9 passed (shared coefficient path intact)
- `pytest tests/boundaries/cpml/test_fdtd_cpml.py tests/monitors/observers/test_fdtd_observers.py` → 14 passed
- `pytest tests/validation/benchmark/test_media_validation_coverage.py tests/gradients/test_fdtd_adjoint_bridge.py` → 65 passed

Behavioral evidence (throwaway scripts under `scratch/`, not committed):
- Triggering plane-wave breakdown slab: 1212–1745 events, `total_dissipated_energy > 0`, final-state mask populated, `Result.breakdown_events` non-empty.
- No-breakdown-material scene: `Result.breakdown is None` (zero-cost path preserved).
- Below-threshold-with-descriptor parity: `torch.equal` on all three final E fields between with-descriptor (critical=1e12, never triggers) and without-descriptor runs → **bitwise identical** (max abs diff 0.0).
- Event-log determinism: two identical runs → identical `(step, cell_index, field_before)` logs, identical total dissipation and state masks, and the log is sorted by `(step, cell_index)`.
- Zero base conductivity (`sigma_e=0`, `conductive_enabled` global False) still triggers and dissipates.
- Guard rejects verified: unsupported `model`/`state`/`recovery`/`damage_parameters` (construction), FDFD+breakdown, anisotropic+breakdown (material), trainable+breakdown (`_validate_breakdown_support`).

## Falsifications recorded

1. **Trigger/threshold comparison is load-bearing.** In the 1743-event determinism scene, temporarily multiplied the trigger threshold by `1e9` (`exceeding = magnitude >= runtime["critical"] * 1e9`). Observed: events dropped 1743 → 0, final-state all intact, zero dissipation. Restored → machinery green again. Proves triggering is driven by the field/threshold comparison, not an unconditional path.
2. **Below-threshold parity gate probe.** Temporarily forced the coefficient scatter to overwrite unconditionally (`active = ones_like(...)` instead of `edge_extra > 0`). Below-threshold parity remained bitwise identical — the intact-edge decay/curl reconstruction from the stored base coefficients round-trips bit-exactly for these edges, so parity is robust by exact reconstruction rather than only by the write-gate. Restored to the gated form (kept as a guarantee and to skip work on inactive edges).

## Known gaps / deferred (for D2 and later phases)

- D1 does **not** author the formal `tests/breakdown/` suite; D2 owns the manufactured trigger/no-trigger golden tests, dt-convergence, energy-closure (PEC box), and the falsification-backed regression tests.
- `deposited_energy_at_trigger` is defined as the local stored electric field energy `0.5·eps·|E|²·cell_volume` at the trigger step (documented in `BreakdownEvent`); the cumulative breakdown dissipation at trigger is ~0 under latching.
- Per-cell `dissipated_energy` is edge-consistent (edge energy split equally among capable node endpoints so per-node sums equal the edge total); `total_dissipated_energy` is the authoritative closure scalar. Energy-balance tolerance to be quantified by D2's closed-box test.
- Ramp fraction uses `elapsed = (n − trigger_step)·dt` so the trigger step itself adds no conductivity (ramp starts the following step); documented.
- Event-buffer capacity equals the capable-cell count (exact latching upper bound), so overflow is structurally impossible; the overflow hard-error is retained defensively and D2 can shrink `runtime["capacity"]` to exercise it.
- Breakdown runs on the eager step path (no CUDA-graph capture); performance overhead of the machinery is not measured this round (single-GPU correctness only).
- Multi-GPU, trainable smooth surrogate, recovery/damage state machines, and coefficient-path compositions (dispersive/nonlinear/anisotropic/modulated/gyromagnetic) are fail-closed and deferred.
