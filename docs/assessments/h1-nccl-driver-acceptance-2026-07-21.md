# H1 NCCL driver — acceptance (stages H1a / H1b)

Track: `h1-nccl-driver`. Worktree: `.worktrees/wh1-nccl-driver`, branch
`fable/nccl-driver`. Base: master `18bc42a` (clean). Hardware: 2x RTX A6000.
Env: `maxwell`. All commands below export:

```bash
export CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=/home/xingyu/code/witwin/witwin-maxwell/.worktrees/wh1-nccl-driver
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING=false
```

Correctness only (shared GPUs); no wall-clock/timing number is produced or
asserted anywhere in this track.

---

## Stage H1a — per-rank collective NCCL reverse driver (DELIVERED, all gates green)

Implements the G1 7-step plan end to end: a trainable Box-`MaterialRegion`
density scene now backpropagates over a one-process-per-GPU NCCL launch, matching
the single-process single-GPU adjoint.

### Delivered

1. **Forward-replay dict halos (the one new transport primitive).**
   `NcclHaloTransport.forward_electric_halo` / `forward_magnetic_halo`
   (`fdtd/distributed/nccl_transport.py`) move the replay's per-rank state-dict
   Yee x-planes collectively (transpose siblings of the landed G1a adjoint halos,
   mirroring the live-field `exchange_electric`/`exchange_magnetic`).
   `CudaP2PHaloTransport` gains the same-named methods
   (`fdtd/distributed/transport.py`) reproducing the previous module-level
   `_forward_electric_halo`/`_forward_magnetic_halo` byte-for-byte, and
   `replay_distributed_segment` now routes both forward halos through
   `distributed.transport` so the replay is transport-agnostic. (The dead
   module-level halo functions were removed.)
2. **Narrow entry-fence relaxation.** `DistributedFDTD.__init__` gains an internal
   `allow_adjoint` flag (default `False`). When set by the driver it admits a
   trainable-density scene and a point-region monitor on the NCCL path; every other
   fence stays, and the forward-only NCCL path (`allow_adjoint=False`) rejects a
   trainable/monitor scene exactly as before. No `NotImplementedError` guard was
   added or removed, so the capability-guard census budget is unchanged.
3. **Per-rank collective driver.**
   `_NcclDistributedFDTDGradientBridge` + `run_nccl_distributed_reverse`
   (`fdtd/distributed/adjoint.py`) subclass the in-process bridge and reuse its
   reverse math verbatim (`_reverse_phases_standard` / `_reverse_phases_cpml`, the
   NCCL adjoint halos, `_accumulate_source_term_gradients`, checkpoint/replay). The
   three single-process assumptions are replaced: (i) the forward output is a
   per-rank LOCAL pack (`_run_forward_capture_local`, no global `_collect_output`);
   (ii) the objective is a separable local functional
   (`point_monitor_l2_objective`) differentiated per rank into local seed
   cotangents — a point monitor is owned by exactly one rank, so off-owner ranks
   seed zero and receive adjoint only through the halos, and the field-grad scatter
   is bypassed (`_scatter_field_grad` override); (iii) `grad_eps` is gathered
   slab-wise to rank 0 (already-NCCL-capable `_gather_component`) and the material
   pullback runs on rank 0 only (`_material_pullback` override). Off-owner ranks
   still drive the full collective reverse (`_backward_is_noop` override → always
   `False`) so the ranks never deadlock.

### Gates (2 processes × 1 GPU) — all green

Worker: `tests/fdtd/multi_gpu/_nccl_adjoint_worker.py` (mode via
`WITWIN_NCCL_ADJ_MODE`). Pytest launchers in `tests/fdtd/multi_gpu/test_nccl_transport.py`.

| Gate | Mode | Result |
|------|------|--------|
| 2a objective+grad parity, standard open-boundary cross-seam | `standard` | `NCCL_ADJOINT_WORKER_OK[standard]` |
| 2a parity, x-CPML interior probe | `cpml` | `NCCL_ADJOINT_WORKER_OK[cpml]` |
| 2a parity, x-CPML psi-active (probe deep in high x-PML) | `cpml_psi` | world-max psi cotangent significant (>0.1·E/H), parity holds → OK |
| 2b no-op magnetic adjoint halo → parity red | `falsify_mag_halo` | diverges rel=9.31e-1 → OK |
| 2b no-op electric adjoint halo → parity red | `falsify_elec_halo` | diverges rel=1.64e-2 → OK |
| 2c zero psi cotangent carry → psi-active parity red | `falsify_psi` | diverges rel=6.88e-2 → OK |
| 2d bitwise repeat-determinism of gathered grad_eps | `determinism` | `torch.equal` on Ex/Ey/Ez → OK |
| 2e unsupported-adjoint scene rejects on all ranks, no hang | `guard_deadlock` | clean symmetric reject under 200 s timeout → OK |

Parity tolerances mirror `tests/fdtd/multi_gpu/test_adjoint_parity_cpml.py`: loss
rtol 5e-5 / atol 5e-6; grad rtol 1e-4 with an atol floor 1e-6·max|grad|.
Falsification separation threshold 1e-3 (baseline drift ~1e-7).

### Falsifications recorded

- **Reverse halos are load-bearing (2b).** No-op'ing either NCCL reverse field
  halo (`exchange_magnetic_adjoint` → rel 9.31e-1; `exchange_electric_adjoint` →
  rel 1.64e-2) drives the 2-GPU gradient far off the single-GPU reference; both
  >> the 1e-3 gate. Permanent launcher
  `test_two_rank_nccl_adjoint_falsification[falsify_mag_halo|falsify_elec_halo]`.
- **psi cotangent carry is load-bearing (2c).** Zeroing the accumulated psi
  cotangents before every distributed CPML reverse step (single-GPU reference
  untouched) moves the psi-active gradient rel 6.88e-2 off parity. Permanent
  launcher `...[falsify_psi]`.
- **Forward-replay NCCL halo is load-bearing (ad-hoc, recorded).** Temporarily
  early-`return`ing `NcclHaloTransport.forward_electric_halo` (so the replay
  interface mid-step is stale) reran mode `standard`: parity went red at the
  on-split density texel column (index (2,2,·)) — greatest relative difference
  1.51e-3 vs the 1e-4 grad gate, exit 1. File restored (`git checkout`); verified
  clean afterwards. This proves the new forward-replay primitive is doing real
  cross-seam work in the replay (not the empty-loop no-op the single-local-shard
  in-process halo would degrade to).

### Test inventory (H1a)

New/edited:
- `tests/fdtd/multi_gpu/_nccl_adjoint_worker.py` (new worker; 8 modes).
- `tests/fdtd/multi_gpu/test_nccl_transport.py` (+8 launcher nodes:
  `test_two_rank_nccl_adjoint_parity[standard|cpml|cpml_psi|determinism]`,
  `test_two_rank_nccl_adjoint_falsification[falsify_mag_halo|falsify_elec_halo|falsify_psi]`,
  `test_two_rank_nccl_adjoint_unsupported_rejects_without_deadlock`).

Direct torchrun runs (all exit 0 with the OK token; falsify modes exit 0 because
the worker asserts the *divergence*):

```bash
for mode in standard cpml cpml_psi determinism falsify_mag_halo falsify_elec_halo falsify_psi guard_deadlock; do
  WITWIN_NCCL_ADJ_MODE=$mode python -m torch.distributed.run --standalone --nnodes=1 \
    --nproc-per-node=2 tests/fdtd/multi_gpu/_nccl_adjoint_worker.py
done
```

Pytest (fast subset confirmed under pytest wiring; long psi modes validated by
direct torchrun above):

```bash
python -m pytest \
  "tests/fdtd/multi_gpu/test_nccl_transport.py::test_two_rank_nccl_adjoint_parity[standard-400]" \
  "tests/fdtd/multi_gpu/test_nccl_transport.py::test_two_rank_nccl_adjoint_parity[cpml-400]" \
  "tests/fdtd/multi_gpu/test_nccl_transport.py::test_two_rank_nccl_adjoint_falsification[falsify_mag_halo-400]" \
  tests/fdtd/multi_gpu/test_nccl_transport.py::test_two_rank_nccl_adjoint_unsupported_rejects_without_deadlock \
  -q
#   -> 4 passed
```

No-regression (the risky replay-routing refactor) on the in-process cuda_p2p
adjoint suites:

```bash
python -m pytest tests/fdtd/multi_gpu/test_adjoint_replay.py \
  tests/fdtd/multi_gpu/test_adjoint_parity.py -q          # -> 23 passed
python -m pytest tests/fdtd/multi_gpu/test_adjoint_parity_cpml.py -q   # -> 10 passed
```

Adjacent suites (census + public/smoke + edited-module suites):

```bash
python -m pytest tests/api/public/test_guard_census.py tests/api/public/test_public_api.py \
  tests/api/public/test_simulation_smoke.py tests/fdtd/multi_gpu/test_transport_adjoint.py \
  tests/fdtd/multi_gpu/test_guard_regressions.py -q       # -> 57 passed
```

Commit: `2e99e6c feat(fdtd-distributed): per-rank collective NCCL reverse driver`.

---

## Stage H1b — S5 tiled monitor seeds (DEFERRED, precise notes) + docs/census/FEATURE_LIST

### Census reconciliation

Capability-guard census budget verified against THIS base: `CAPABILITY_GUARD_BUDGET
= 175` (`tests/api/public/test_guard_census.py`), and `test_guard_census.py`
passes unchanged. H1a added no `raise NotImplementedError` and removed none; the
`allow_adjoint` relaxation only re-conditions existing `ValueError` fences (not
counted by the census). Budget stays **175**.

### FEATURE_LIST

Additive subsection `h1-nccl-driver` appended (multi-GPU NCCL trainable-density
adjoint).

### S5 tiled monitor seeds — DEFERRED (never half-land)

Supervisor decision 3 makes S5 a stretch conditional on H1a landing green. H1a is
green, but S5 is a genuinely separate subsystem, not an increment on the H1a
driver, and is deferred with the executable plan below rather than risk a partial
landing.

**Why it is not a small add-on:**
- Tiled (plane/flux/mode) monitors are not wired through the NCCL *forward* at all:
  `DistributedFDTD._validate_nccl_capabilities` admits only point-region monitors
  even under `allow_adjoint` ("per-monitor payload gather across ranks is not wired
  yet"). A separable per-strip objective can avoid the forward gather (each rank
  sums its owned strip; the world sum reproduces the global objective), but that
  still requires relaxing this forward fence for the separable-tiled case.
- `require_distributed_adjoint_objective_support` fail-closed-rejects every tiled
  monitor cotangent today; it must be relaxed for the separable case.
- The owner-strip decomposition must mirror the forward monitor seam rule exactly.
  `fdtd/distributed/monitor_merge.py` (l.381+) records that point monitors and
  x-normal planes have exactly one owner, but **y/z-normal planes are stitched
  from per-shard strips**; a reverse seed that double-counts a seam-boundary strip
  cell would corrupt both the objective and the gradient. This seam-consistent
  owner-strip rule is the load-bearing subtlety.

**Executable plan for the next agent:**
1. Relax `_validate_nccl_capabilities` (under `allow_adjoint`) and
   `require_distributed_adjoint_objective_support` to admit a *separable* tiled
   objective (`sum|E_strip|^2` per owned strip), keeping non-separable tiled
   objectives rejected.
2. Confirm the shard-local forward pack exposes each rank's owned plane strip with
   the seam-boundary cell owned by exactly one shard (reuse the `monitor_merge`
   owner rule; assert single-owner per strip cell, mirroring l.412's "more than one
   shard owner" guard on the reverse side).
3. Reuse the H1a local-seed path unchanged: the per-strip cotangent is already the
   shard-local owned-strip grad, so `_build_output_seeds` consumes it directly
   (no scatter), exactly as the point-monitor path does.
4. Gate: seeded-objective + gathered-grad_eps parity vs single-GPU for a y/z-normal
   plane objective spanning the x seam (loss rtol 5e-5/atol 5e-6, grad rtol
   1e-4/atol-floor 1e-6·max|grad|), plus a **seam-ownership falsification** (assign
   a seam-boundary strip cell to the wrong shard → parity red).

No code change and no census impact for S5.

### Known gaps / deferred (H1b)

- S5 tiled (plane/flux/mode) monitor adjoint seed scatter over NCCL — deferred as
  above, fail-closed retained.
- No wall-clock/timing numbers produced (correctness-only shared GPUs).
- The census budget is unchanged at 175 (verified against this base; no guard
  added or relaxed).
