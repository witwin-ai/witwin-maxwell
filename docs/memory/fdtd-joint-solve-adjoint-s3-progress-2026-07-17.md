# Joint-solve adjoint — Phase 7 slice S3 progress (2026-07-17)

Branch `codex/joint-solve-adjoint`. Follows S1+S2
(`fdtd-joint-solve-adjoint-s1-s2-progress-2026-07-17.md`). S3 completes and wires the
public distributed joint-solve adjoint for a trainable Box `MaterialRegion` density on
the pure real standard path. S4 (CPML-trainable) and S5 (tiled-monitor seed scatter)
remain follow-ups.

Authoritative spec: `docs/plans/next-functional-2026-07/02-phase-7-8-blueprint-2026-07-16.md`
Part B. Deviations recorded below.

## What landed

### Distributed gradient bridge (`fdtd/distributed/adjoint.py`)
- `_DistributedFDTDGradientBridge` + `run_distributed_fdtd_with_gradient_bridge` and a
  `_DistributedFDTDMaterialGradientFunction` (torch.autograd.Function).
- Forward: builds `DistributedFDTD`, `require_distributed_adjoint_support` +
  `require_distributed_adjoint_objective_support`, preflights the global grad_eps gather
  (`require_gather_capacity`), then runs `_advance_one_step` per step capturing
  per-shard checkpoints at `_checkpoint_stride` boundaries, and merges monitors via
  `_collect_output` into a global `_prepare_forward_pack`.
- Reverse loop per step (blueprint order): per-shard seed inject → Phase 1 all shards →
  `exchange_magnetic_adjoint` → Phase 2 all shards → `exchange_electric_adjoint` →
  Phase 3 → per-shard `_accumulate_source_term_gradients`. Reuses the S1 phase-split
  helpers (`reverse_phase1_electric_to_h/2/3`) and the S1 transposed transports
  unchanged. Devices synchronized between phases/halos (correctness over overlap; the
  reverse pass has no timing gate).
- Seed routing: per-shard `_build_output_seeds` on the shard's own solver + local pack,
  with global cotangents routed by ownership — point monitors 1:1 to the single owner,
  full-field DFT via `_scatter_field_grad_to_shard` (transpose of `_gather_component`).
  Non-owner shards get empty seeds and receive adjoint only through the transposed
  halos.
- Global parameter reduction (blueprint key simplification): gather per-shard grad_eps
  OWNED slices via `_gather_component` into global tensors, then run the EXISTING
  single-GPU `pullback_material_input_gradients` once on the logical scene. No per-shard
  density windowing.

### Public wiring + capability-scoped guards (`simulation.py`)
- Replaced the blanket `_reject_trainable_parallel_fdtd` with
  `_validate_trainable_parallel_fdtd`: a prepare-time, scene/config-static validator
  that runs BEFORE the distributed solver is built (before any shard allocation). It
  rejects the exact blueprint list — trainable geometry/perturbation/circuit/RF-port,
  RF excitation/port sweep, CPML absorber, dispersive/conductive/nonlinear/anisotropic/
  modulated media (`_nonstandard_medium_reason`), shutoff>0, multi-source normalization,
  and tiled plane/flux/mode/closed/diffraction/non-point-time objectives
  (`_tiled_monitor_objective_reason`) — and allows a trainable Box density, routing it
  to `_run_distributed_fdtd_with_gradient_bridge`.

### Guard relaxations (`fdtd/distributed/solver.py`)
- `_validate_static_capabilities`: MaterialRegion guard relaxed to accept Box geometry
  (non-Box still rejected); the trainable guard is now capability-scoped
  (`_unsupported_distributed_trainable_tensors` rejects geometry/perturbation/circuit/rf
  but allows density). Box densities rasterize per shard by physical position
  (`_sample_material_region_density` grid_sample vs global center/size), so no
  distributed density slicing is needed for the forward.

### Census
- The two bridge guards (no-trainable-input, shutoff) are ValueError, matching the
  distributed guard family; the census (NotImplementedError only) stays 119, no
  reconciliation required. `tests/api/public/test_guard_census.py` green.

## Exit-gate evidence (2× A6000, CUDA_VISIBLE_DEVICES=0,1)
- `tests/fdtd/multi_gpu/test_adjoint_parity.py` (11): 1-vs-2-GPU objective parity
  (bit-identical forward, loss diff 0.0) and gradient parity (measured rel drift ~1e-7;
  gate rtol=1e-4 + atol=1e-6·max|grad|); central finite differences on density texels on
  the x-split and interior to each shard (`(1,2,2)/(2,2,2)/(3,2,2)`, source x=-0.3,
  monitor x=0.1) and with source+monitor on the interface node (min-rel-error over 3 h,
  measured ~1e-5, gate 2e-3); bitwise-reproducible gathered grad_eps (`torch.equal`);
  and seven prepare-time guard regressions (raise before allocation).
- `tests/fdtd/multi_gpu/test_material_region_forward.py` (1): distributed Box density
  forward — gathered per-shard compiled eps equals single-GPU exactly (atol 0), plus
  field/monitor parity within the plan gates.
- `tests/fdtd/multi_gpu` 147 passed; `tests/api/public` 31 passed;
  `tests/fdtd/multi_gpu/{test_parallel_public_api,test_adjoint_replay,test_guard_regressions}`
  updated for the new capability boundary.

## Falsification checks (scratch monkeypatch → red → restore)
- Transposed adjoint halos are load-bearing: no-op'ing
  `CudaP2PHaloTransport.exchange_electric_adjoint`/`exchange_magnetic_adjoint` makes the
  left-shard-interior texel gradient collapse to 0.0 (FD min-rel-error 1.0, gate 2e-3
  FAIL) while the objective is on the right shard; restoring returns to rel ~2.7e-4
  (PASS). This is why the propagating-scene FD case uses a left-shard texel with a
  right-shard monitor — the interface-node case alone does not exercise the cross-shard
  transport (objective and texel share one owner; no-op'ing the halos barely moves it).
- Per-shard density rasterization is load-bearing: stripping `material_regions` from the
  local scene in `_build_local_scene` makes the gathered compiled eps differ from
  single-GPU by 2.5e-11 (the full eps0-scaled region contrast, eps0≈8.85e-12) vs the
  test's exact-0 gate → the forward parity test fails.

## Deviations from blueprint (recorded)
- Determinism is asserted bitwise on the DISTRIBUTED REVERSE PRODUCT (gathered grad_eps),
  not the final density gradient. The density VJP passes through torch's `grid_sample`
  backward, which uses atomicAdd and is not bitwise reproducible single- or multi-GPU
  alike (observed ~6.6e-24 run-to-run). The blueprint frames determinism as `torch.equal`
  on parameter gradients; the S3 reverse (assign-semantics fused kernels, ordered `add_`,
  deterministic gather) IS bitwise, and the shared grid_sample step is gated with a tight
  tolerance (rtol 1e-5) with the technical reason recorded — matching the blueprint's
  "if a component breaks it, record the reason and a quantified tolerance" clause.
- Reverse loop synchronizes devices between phases/halos rather than overlapping streams.
  Correctness-first; Phase 7 has no reverse-pass timing gate, so overlap is a later
  optimization.

## Deferred (S4/S5)
- S4: CPML face-semantics audit + phase-split of `_reverse_step_cpml_native_core` +
  CPML-trainable distributed reverse/parity/FD. Guarded (CPML absorber rejected at
  prepare for trainable+parallel) until audited.
- S5: tiled plane/flux/mode monitor adjoint seed scatter. Guarded
  (`require_distributed_adjoint_objective_support` + prepare-time
  `_tiled_monitor_objective_reason`).
- No reverse-pass timing/speedup numbers were measured (shared-GPU window; evidence
  deferred).

Invocation: conda env `maxwell`, `CUDA_HOME`=.../nvidia/cu13, multi-GPU tests
`CUDA_VISIBLE_DEVICES=0,1`, `PYTHONPATH` set to this worktree.
