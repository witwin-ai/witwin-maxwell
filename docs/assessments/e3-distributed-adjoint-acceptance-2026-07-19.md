# Track E3 — Distributed adjoint acceptance (2026-07-19)

Worktree `.worktrees/we3-mgpu-adjoint`, branch `fable/distributed-adjoint`.
Invocation for every command below:

```bash
export CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=/home/xingyu/code/witwin/witwin-maxwell/.worktrees/we3-mgpu-adjoint
export CUDA_VISIBLE_DEVICES=0,1
conda run -n maxwell --no-capture-output python -m pytest <targets> -q
```

> Slice 2 (defense-in-depth trainable guard on `DistributedFDTD`, forward monitor
> gather, FEATURE_LIST) is E3b's deliverable and will be appended to this doc.

## Slice 1 — S4 distributed CPML-trainable bridge (E3a) — DELIVERED

### Delivered items

1. **CPML-aware forward replay half-steps** (`witwin/maxwell/fdtd/adjoint/core.py`):
   `_forward_magnetic_fields_cpml` / `_forward_electric_fields_cpml` mirror the real
   CPML magnetic/electric halves of `_step_state` exactly and additionally return
   the advanced psi_h / psi_e dicts. `replay_distributed_segment`
   (`fdtd/distributed/adjoint.py`) branches on `uses_cpml`: it threads the twelve
   psi fields through the two explicit halves (electric halo -> magnetic half ->
   magnetic halo -> electric half) and carries the full 18-field CPML state in the
   trajectory. The psi recurrence is a same-x-slice update, so no psi halo is
   copied.
2. **psi-carrying reverse loop** (`fdtd/distributed/adjoint.py`): `_reverse_one_step`
   dispatches to `_reverse_phases_cpml` when `uses_cpml`, running
   `reverse_cpml_phase_electric` -> `exchange_magnetic_adjoint` (Hy/Hz) ->
   `reverse_cpml_phase_magnetic` -> `exchange_electric_adjoint` (Ey/Ez), then the
   shared per-shard `_accumulate_source_term_gradients` tail. The adjoint state
   carries the 12 psi cotangents (`_CPML_STATE_NAMES`, seeded to zero; monitor seeds
   inject E/H only). No psi halo.
3. **Guard relaxation + PML-pinning assertion**:
   - `require_distributed_adjoint_support` accepts `active_absorber_type` in
     {`none`, `cpml`, `stablepml`}; graded-sigma `pml`/`absorber` and every
     dispersive/conductive/nonlinear/aniso/Bloch/TFSF/modulated/coupled channel
     stay rejected (ValueError).
   - `_assert_x_pml_pinned_to_outer_shards` (new, RuntimeError) asserts low x-PML is
     owned only by rank 0 and high x-PML only by the last rank; vacuous when there is
     no x-PML. Run inside `require_distributed_adjoint_support`.
   - `Simulation._validate_trainable_parallel_fdtd` relaxed: a PML boundary with
     absorber `cpml`/`stablepml` is accepted; graded-sigma `pml`/`absorber` raise.
4. **Census**: no NotImplementedError guard added or removed (the new pinning guard
   is a RuntimeError; the relaxed rejections are ValueError). `CAPABILITY_GUARD_BUDGET`
   stays 176 and `tests/api/public/test_guard_census.py` passes unchanged.

### Files changed / added

- `witwin/maxwell/fdtd/adjoint/core.py` (added two CPML half-step helpers)
- `witwin/maxwell/fdtd/distributed/adjoint.py` (CPML replay + reverse branch +
  guard relaxation + pinning assertion)
- `witwin/maxwell/simulation.py` (validator relaxation)
- `tests/fdtd/multi_gpu/test_adjoint_parity_cpml.py` (new — CPML gates)
- `tests/fdtd/multi_gpu/test_adjoint_parity.py` (guard test updated: graded-sigma only)
- `tests/fdtd/multi_gpu/test_adjoint_replay.py` (CPML now accepted; graded rejected)
- `tests/fdtd/multi_gpu/test_guard_regressions.py` (CPML accepted; graded rejected)

### Gates (all 2-GPU, executed on 2x A6000), calibration = measurement

Scene: x-CPML dielectric, 21 x-nodes + 4-cell CPML pad/face (Nx=29, cell_count=28),
Box density region x in [-0.3,0.3] straddling the x=0 split, PointDipole source /
PointMonitor objective, absorber="cpml", STEPS=50.

| Gate | Test | Measured | Gate |
|---|---|---|---|
| (a) objective parity 1-vs-2-GPU | `test_cpml_objective_and_gradient_parity_single_vs_two_gpu` | rel **0.0** (bit-identical) | rtol 5e-5/atol 5e-6 |
| (a) gradient parity 1-vs-2-GPU | same | rel **1.18e-7** | rtol 1e-4, atol 1e-6·max |
| (b) interface FD (on-split + interior) | `test_cpml_interface_finite_difference_split_and_interior` | on-split rel **7.4e-5**, interior **5.3e-6 / 3.1e-5** | 2e-3 |
| (c) no-psi-halo confirmation | `test_cpml_no_field_halo_falsification` | baseline **1.2e-7**; no-mag-halo **0.956**; no-elec-halo **0.016** | baseline<1e-4, falsified>1e-3 |
| (d) repeat determinism (gathered grad_eps) | `test_cpml_repeat_reverse_gradient_is_bitwise_deterministic` | `torch.equal` **bitwise** | bitwise |
| (e) guard regressions | `test_adjoint_parity.py::test_guard_graded_sigma_absorber_rejected_at_prepare`, `test_adjoint_replay.py::test_checkpoint_rejects_graded_sigma_absorber`, `test_guard_regressions.py::test_require_distributed_adjoint_support_rejects_graded_sigma_absorber` | reject `pml`/`absorber`, accept `cpml`/`stablepml` | — |
| positive prepare | `test_cpml_trainable_parallel_scene_prepares` | prepares, `active_absorber_type=="cpml"` | — |
| x-CPML pinning assertion | `test_adjoint_parity_cpml.py::test_x_pml_pinning_assertion_*` (CPU) | pinned passes; interior-low-PML and non-last-high-PML raise RuntimeError | — |

### Falsifications recorded (load-bearing)

1. **No-psi-halo confirmation (gate c, the mandated load-bearing falsification).**
   With both Yee field halos active the 2-GPU CPML gradient matches single-GPU to
   rel **1.18e-7** *without any psi halo*. Monkeypatching `exchange_magnetic_adjoint`
   to a no-op drives rel to **0.956**; monkeypatching `exchange_electric_adjoint`
   drives rel to **0.016** — both >=2 orders above the 1e-3 falsification threshold.
   Confirms the two field halos carry the entire cross-interface curl(H)/curl(E)
   coupling (including the psi-derived `adj_d` folds), so a separate psi halo would
   be redundant — the S4 audit's central prediction. Encoded as
   `test_cpml_no_field_halo_falsification`.
2. **Pinning assertion is non-vacuous.** `_assert_x_pml_pinned_to_outer_shards`
   accepts the real pinned 2-shard partition but raises RuntimeError on fabricated
   partitions where an interior shard owns a low x-PML cell, or rank 0 stretches
   into the high x-PML band. Encoded as the three `test_x_pml_pinning_assertion_*`
   CPU unit tests.

### Test commands + pass counts (executed)

```bash
# Slice-1 gates + guard suites + census (28.22s)
pytest tests/fdtd/multi_gpu/test_adjoint_parity_cpml.py \
       tests/fdtd/multi_gpu/test_adjoint_parity.py \
       tests/fdtd/multi_gpu/test_adjoint_replay.py \
       tests/fdtd/multi_gpu/test_guard_regressions.py \
       tests/api/public/test_guard_census.py
# -> 52 passed

# Single-GPU adjoint regression + public API smoke (88.98s)
pytest tests/gradients/test_fdtd_adjoint_bridge.py \
       tests/gradients/test_fdtd_adjoint_materials.py \
       tests/api/public/test_public_api.py \
       tests/api/public/test_simulation_smoke.py
# -> 109 passed

# Full multi-GPU suite (regression, 57.29s)
pytest tests/fdtd/multi_gpu/
# -> 248 passed
```

### Known gaps / deferred (E3a scope)

- Slice 2 (DistributedFDTD-layer trainable guard, forward monitor gather,
  FEATURE_LIST) is E3b's deliverable.
- NCCL (one-process-per-GPU) CPML adjoint and coupled-runtime joint-solve remain
  OUT of scope, fail-closed (#13/#18 tail).
- Timing/speedup is out of scope this round (deferred-pending-exclusive-window).
- S5 tiled seeds not attempted this slice.
