# Joint-solve adjoint — Phase 7 slices S1+S2 progress (2026-07-17)

Branch `codex/joint-solve-adjoint`. Internal infrastructure only; the public
trainable+parallel bridge is intentionally still rejected (that is S3). No
user-visible feature yet, so `FEATURE_LIST.md` is unchanged.

Authoritative spec: `docs/plans/next-functional-2026-07/02-phase-7-8-blueprint-2026-07-16.md`
(Part B). Reality deviations from the blueprint are recorded below.

## S1 — transposed transport + phase-split reverse core

### Transposed halo transport (`fdtd/distributed/transport.py`)
- Added `exchange_magnetic_adjoint` / `exchange_electric_adjoint` to the
  `HaloTransport` ABC and `CudaP2PHaloTransport`, plus `prepare_adjoint_staging`.
  Each is the discrete transpose of the matching forward halo: copy the
  neighbour's ghost adjoint plane to a preallocated staging plane on the owner
  device (event-ordered on the comm stream), `add_` into the owner plane on the
  owner compute stream, then zero the ghost on the source compute stream. Staging
  planes are allocated once (keyed by kind+destination rank) and reused — never
  per step. Deterministic ascending-rank order; existing shard events reused, no
  new event allocation.
- Tests `tests/fdtd/multi_gpu/test_transport_adjoint.py` (4): discrete-transpose
  pairing identity `<forward(x), y> == <x, adjoint(y)>` for both halos, ghost
  zeroing, bitwise determinism across repeats (`torch.equal`), and no
  allocator growth across 32 reverse steps after warmup.
- Falsification: mis-wiring the adjoint owner index (accumulate into `stop-2`
  instead of `stop-1`) makes the pairing identity diverge (rhs 0.0 vs lhs 6.79),
  so the identity test is non-vacuous.

### Phase-split of the standard native reverse core (`fdtd/adjoint/native.py`)
- Factored `_reverse_step_standard_native_core` into pure helpers
  `reverse_phase1_electric_to_h`, `reverse_phase2_magnetic_to_e`,
  `reverse_phase3_decay`. The core now calls the three in sequence; same kernels,
  same launch order.
- Bit-identity evidence: a controlled fingerprint harness (open-boundary
  trainable-density scene, two seeds; routes through `native_standard`) produced
  identical loss AND parameter-gradient tensors pre- and post-refactor
  (`torch.equal`, max_abs_diff 0.0 on `loss_0/grad_0/loss_1/grad_1`).
- Falsification: perturbing `reverse_phase2_magnetic_to_e`'s `grad_eps_ex` output
  by 1.0001x changes the gradient fingerprint, so the bit-identity check is
  non-vacuous.
- Single-GPU adjoint suites after the refactor: `test_fdtd_adjoint_bridge.py`
  54 passed; `test_fdtd_adjoint_rigorous.py` + `test_fdtd_adjoint_materials.py`
  44 passed, **2 pre-existing failures unrelated to this change** (`conductive` /
  `bloch_dispersive` FD tests fail on their "medium inactive" forward sanity
  assertion in 3.01s on the untouched base c840f3b too — they do not route
  through the standard core); `p6_acceptance` + `custom_source` + `mode_source`
  13 passed.

### Deviation from blueprint (recorded): CPML core left unfactored
- The blueprint says to also factor `_reverse_step_cpml_native_core` "once its
  face semantics are verified" (blueprint risk 1: CPML reverse kernels may gate
  x-faces differently than the standard kernels at internal faces). That audit is
  scheduled for S4, and S1/S2 target only the pure real standard path, so the
  CPML core is deliberately left unfactored to avoid unverified/dead code.

## S2 — distributed checkpoint capture + replay + guards

### `fdtd/distributed/adjoint.py` (new, internal)
- `capture_distributed_checkpoint(distributed, step)`: per-shard
  `capture_checkpoint_state` inside `torch.cuda.device(shard.device)`, returned as
  a partition-signature-keyed `DistributedCheckpoint`. The capture clones only the
  persistent padded field storage; the magnetic receive halos are views into that
  same storage and are refreshed deterministically on replay, so no transient
  buffer is stored.
- `replay_distributed_segment(...)`: lockstep per-shard forward replay advancing
  each shard as two explicit halves with the forward Yee halos between them
  (electric halo -> magnetic half via `_forward_magnetic_fields` -> magnetic halo
  -> electric half via new `_forward_electric_fields_standard`), mirroring the
  distributed serialized forward schedule. Mid-step H captured per shard **after**
  the magnetic halo so interface forward H is valid for later Phase-2 grad_eps
  reads.
- New shared helper `_forward_electric_fields_standard` added to
  `fdtd/adjoint/core.py`, symmetric to the existing `_forward_magnetic_fields`
  extraction; composes the same `_update_electric_component` /
  `_apply_source_term_list` primitives `_step_state` uses for the standard path.

### Deviation from blueprint (recorded): replay parity is tolerance-gated, not bitwise
- The blueprint frames replay parity as `torch.equal`. In reality the distributed
  forward runs fused CUDA bounded kernels while the replay runs the torch
  reference update; these differ at floating-point reduction order, exactly as the
  single-GPU bridge already tolerance-gates native-forward vs torch-replay
  (`test_fdtd_gradient_bridge_checkpoint_replay_matches_forward_state`,
  rtol=1e-4/atol=5e-5). Trusting the code over the blueprint, the parity test
  (`tests/fdtd/multi_gpu/test_adjoint_replay.py`) gates owned states at
  rtol=1e-5/atol=1e-7, calibrated from measurement: on the 40-step standard scene
  the true owned drift is ~5e-10 absolute, disabling the electric halo yields
  ~8e-5 and the magnetic halo ~1e-2 of interface-localized error. The gate sits
  ~200x above the true drift and ~500x below the smallest halo bug, so it passes
  the correct decomposition and fails a broken halo (falsification-checked by
  no-op'ing each halo). The uniform (non-interface-localized) drift proves the x
  decomposition and both forward halos introduce no algorithmic error.

### Guards (all ValueError, consistent with the DistributedFDTD guard family)
- `require_distributed_adjoint_support`: fail-closed rejection of CPML,
  dispersive/magnetic-dispersive, nonlinear, conductive, full-aniso, complex/
  Bloch, TFSF, modulated, and coupled circuit/port/network configurations at the
  checkpoint/replay entry points.
- `DistributedFDTD._validate_static_capabilities` now runs a defense-in-depth
  trainable rejection (`_scene_trainable_tensors`) so the solver fails closed on a
  trainable scene even if constructed directly, independent of the public
  `Simulation` trainable guard.
- Census: these are ValueError guards (the census gate counts only
  `NotImplementedError`), matching the existing distributed guard convention, so
  the capability-guard budget stays 119 and no census reconciliation is required.

## Exit-gate evidence
- Transport adjoint unit tests: `tests/fdtd/multi_gpu/test_transport_adjoint.py`
  4 passed (pairing identity + determinism + no-allocation).
- Single-GPU adjoint bit-identity after phase split: fingerprint `torch.equal`,
  max_abs_diff 0.0; suites green (see S1).
- Replay parity (2 shards, both GPUs): `test_adjoint_replay.py` passing with the
  calibrated gate + falsification.
- `tests/fdtd/multi_gpu` 133 passed; `tests/api` 137 passed, 42 skipped (census
  budget unchanged).

Invocation: conda env `maxwell`, `CUDA_HOME` = .../nvidia/cu13, multi-GPU tests
`CUDA_VISIBLE_DEVICES=0,1`, `PYTHONPATH` set to this worktree so the editable
`witwin.maxwell` from the primary checkout does not shadow it.
