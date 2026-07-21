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
   `False`) so the ranks never deadlock. The rank-0-only gather-capacity preflight
   in `forward()` is **collective-safe**: rank 0 evaluates capacity, then all ranks
   `allreduce_scalar` the verdict and raise together, so a capacity failure can
   never leave peers blocked in the forward halo collectives (fixes the
   rank-asymmetric-reject deadlock window).

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
| 2d reduction-order repeat-determinism of gathered grad_eps | `determinism` | rel < 1e-9 on Ex/Ey/Ez → OK |
| 2e unsupported-adjoint scene rejects on all ranks, no hang | `guard_deadlock` | clean symmetric reject under 200 s timeout → OK |
| 2e' rank-0-only gather-capacity failure rejects on all ranks, no hang | `capacity_deadlock` | collective verdict → both ranks raise under 200 s timeout → OK |
| headline parity under a saturating co-tenant burner, honest 1e-4 gate | `standard`/`cpml`/`plane` + `WITWIN_NCCL_ADJ_STRESS=1` | ~2e-7 → OK (3 passed) |

> Gate results above hold at the **honest** tolerances below, both on exclusive
> GPUs and under a saturating co-tenant burner on both boards — see the stressed
> gate and "Load-dependence episode and fix" below.

Parity tolerances (honest, 1e-4-class, mirroring the in-process
`test_adjoint_parity_cpml`): loss rtol 5e-5 / atol 5e-6 (the forward is
load-insensitive — bitwise-identical loss across GPU cohabitation); grad rtol 1e-4
with an atol floor 1e-6·max|grad|. Determinism gate: reduction-order equality,
relative < 1e-9. Falsification separation threshold 1e-3 (healthy parity, exclusive
OR stressed, stays ~2e-7; deliberate breaks are >=1.64e-2). A prior round had
widened the grad gate to rtol 1e-3 / atol 1e-2·scale to absorb the drift below;
that laundering is reverted — the fix removes the drift instead.

### Load-dependence episode and fix (root cause + stressed-gate evidence)

**Symptom.** Before the fix the driver's *gathered gradient* was load-dependent:
bitwise-correct on exclusive GPUs (~1.5e-7 relative, 6/6) but deterministically
wrong at the partition seam under concurrent GPU activity — standard rel ~1.2e-4,
CPML ~3.1e-4, seam-spanning plane ~2.7e-2 (`grad_scale` 5.3e-2, a non-degenerate
gradient), with occasional catastrophic blow-ups (rel ~9.7e+2 on standard). In the
same process the forward loss stayed **bitwise identical** and the single-GPU
reference moved only ~7.6e-8, isolating the defect to the distributed **reverse**.
The prior round mischaracterised the worst case as "~5e-3 of scale"; the measured
worst case is ~2.7e-2 on the plane objective (that understatement is removed).

**Race class (confirmed).** `CUDA_LAUNCH_BLOCKING=1` collapses the drift to the
reference floor (standard 2.2e-7, CPML 7.6e-8) — an async happens-before violation,
not intra-kernel reduction order (the `grad_eps` kernels write with `=`, and
`TORCH_NCCL_BLOCKING_WAIT=1` does **not** help). The error is spatially localized to
the two density texels straddling the seam (x=2: 1.2e-4, x=3: 3.7e-5; off-seam
cells at the ~1e-7 floor), i.e. an interface hazard, not spread noise. The same
drift reproduces on the in-process `transport="cuda_p2p"` bridge, placing the class
in the shared reverse machinery.

**Root cause (single line).** `PYTORCH_NO_CUDA_MEMORY_CACHING=1` also eliminates the
drift → a **caching-allocator cross-stream memory-reuse hazard**. The four reverse
and forward-replay NCCL halo methods (`exchange_magnetic_adjoint`,
`exchange_electric_adjoint`, `forward_electric_halo`, `forward_magnetic_halo` in
`fdtd/distributed/nccl_transport.py`) ran their in-place accumulate/zero — and
posted the collective — inside `with torch.cuda.stream(engine.compute_stream)`. But
the per-step adjoint-state planes they touch are allocated on the **default** stream
by the reverse kernels. The CUDA caching allocator tags a freed block with only its
*allocation* stream, so a block freed on the default stream could be handed to the
next default-stream allocation while the compute-stream halo was still writing it;
under concurrent load the compute-stream work lags and the block is overwritten
mid-`add_`. The **live-forward** halos are immune because they slice the solver's
*persistent* (never-freed) field storage — which is exactly why the forward loss
stays bitwise-clean while only the reverse (which churns per-step allocations)
corrupts.

**Fix.** Run those four halos on the current (**default**) stream (keep the device
guard, drop the `compute_stream` context; record the vestigial forward-overlap
events on the current stream). The reverse driver already fully host-synchronizes
between phases (`_synchronize(devices)`), so `compute_stream` bought no overlap
there; matching allocation-stream to use-stream closes the window at zero cost.
Commit `c233d8b`.

**Stressed-gate evidence (honest tolerances, both GPUs saturated by a committed
co-tenant burner).** Distribution of grad relative error vs the single-GPU
reference, 6 runs each (gate 1e-4):

| Mode | runs (grad rel) | max | gate |
|------|-----------------|-----|------|
| standard | 2.18e-7 ×3, 1.09e-7 ×3 | 2.18e-7 | 1e-4 |
| cpml | 2.28e-7, 1.52e-7, 3.04e-7, 1.52e-7, 2.28e-7, 1.52e-7 | 3.04e-7 | 1e-4 |
| plane (`grad_scale` 5.3e-2) | 1.40e-7 ×4, 1.36e-10, 1.40e-7 | 1.40e-7 | 1e-4 |

The committed `test_two_rank_nccl_adjoint_parity_under_stress[standard|cpml|plane]`
runs these three at the honest gate with the burner load spawned in-worker
(`WITWIN_NCCL_ADJ_STRESS=1`): **3 passed**.

**Load-bearing falsification (the fix is the load-bearing sync).** Re-wrapping the
four halos back onto `engine.compute_stream` (the pre-fix behaviour) under the same
saturating burner reddens every headline gate at the honest 1e-4 tolerance:
standard rel 1.223e-4, CPML 3.120e-4, plane 2.684e-2 — each `> 1e-4`, RED. Restoring
the default-stream fix returns them to ~2e-7 (green). Exact site: the
`with torch.cuda.stream(engine.compute_stream)` wrapper on the four reverse/replay
halos; observed drift as tabulated.

### Audit minors addressed

- **Collective-safe gather-capacity reject + deadlock-freedom test.** The eps-shaped
  grad gather lands only on rank 0, so only rank 0 can size it; the reject is made
  collective by having rank 0 evaluate capacity, all ranks `allreduce_scalar` the
  verdict, and all raise together (already in `_NcclDistributedFDTDGradientBridge.forward`).
  Now covered by a committed deadlock-freedom test: worker mode `capacity_deadlock`
  forces the rank-0 capacity preflight to raise (as an over-large grid would) and
  asserts every rank raises within the launcher timeout, so a capacity failure can
  never leave a peer blocked in the forward-halo collectives. Launcher
  `test_two_rank_nccl_adjoint_capacity_reject_without_deadlock` (2 passed with
  `guard_deadlock`).
- **Seam-liveness numbers repointed at the committed gate.** No uncommitted per-strip
  sums are cited; the seam-liveness conclusion rests entirely on the committed
  `plane_seam` falsification launcher (double-counting the live seam cell reddens
  parity, grad rel 4.005e-1), as already recorded in the H1b section below.

### Falsifications recorded

- **Stream-discipline fix is load-bearing (load).** Re-wrapping the four
  reverse/replay halos onto `engine.compute_stream` under a saturating co-tenant
  burner reddens every headline gate at the honest 1e-4 tolerance (standard 1.223e-4,
  CPML 3.120e-4, plane 2.684e-2). Restoring the default-stream fix returns them to
  ~2e-7. See "Load-dependence episode and fix" above for the exact site and mechanism.
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

## Stage H1b — S5 separable tiled-plane monitor adjoint seed (DELIVERED, all gates green) + docs/census/FEATURE_LIST

### Delivered

A y/z-normal `PlaneMonitor` objective is now seedable on the per-rank NCCL adjoint
driver. The plane is tiled across the x seam; the objective sums each rank's owned
plane strip (`plane_monitor_l2_objective`), so the world sum reproduces the
single-process full-plane objective with every seam cell counted on exactly one
rank and each rank seeding only its owned strip. No cross-rank cotangent scatter is
introduced — the seam ownership is carried entirely by the separable objective.

Commit `b402074 feat(fdtd-distributed): separable tiled-plane monitor adjoint seed
over NCCL`.

**Code (`witwin/maxwell/fdtd/distributed/`):**
1. `adjoint.is_separable_plane_monitor(monitor)` — the exact-class `PlaneMonitor`
   on a y/z normal without flux. Flux/mode/finite/x-plane are excluded (they need
   seam-crossing tangential-field assembly, not owned-strip separable).
2. `adjoint.require_distributed_adjoint_objective_support` — the tiled rejection is
   now gated on `distributed._allow_adjoint`: a separable y/z plane is admitted only
   on the NCCL adjoint driver; the in-process bridge (`allow_adjoint=False`) still
   rejects every tiled monitor.
3. `solver._validate_nccl_capabilities` — the forward NCCL fence admits a separable
   y/z plane under `allow_adjoint` (the adjoint driver reads each rank's
   shard-local monitor output directly via `_shard_local_raw_output`, never the
   collective per-monitor gather the forward output path lacks, so the "gather not
   wired" concern does not apply).
4. `adjoint._index_global_grads` / `_build_shard_seed_runtimes` route plane-template
   cotangents: one output tensor per plane component (its `["data"]` strip), indexed
   by `(monitor, component)` exactly like the point path; `_build_output_seeds`
   already builds plane seed batches from the shard solver's
   `_plane_observer_groups`, so the shard-local owned-strip cotangent is consumed
   directly (no scatter).
5. `adjoint.plane_monitor_l2_objective` — owned-strip separable `sum|spectrum|^2`
   (plane x axis at dim −2, restricted to `layout.component(name).owned_local_slice[0]`,
   the forward monitor-merge seam rule); point components contribute their full
   spectrum.

**Load-bearing seam fact:** the shard-local plane observer records the DFT only on
cells the shard drives, and the ghost/halo column carries real field when the
forward halo keeps it live. With the source placed just left of the seam
(`source_x=-0.05`), both owned strips carry comparable field (genuinely dual-sided)
AND the discarded ghost column is non-trivial, so the owned-`owned_local_slice[0]`
crop is load-bearing on a seam-spanning plane, not a no-op. This is proven
directly by the committed `plane_seam` falsification gate below: summing the FULL
local strip (halo included) double-counts the live seam cell and drives parity red
(grad rel 4.005e-1) — a dead ghost column would instead leave parity intact and
fail that launcher's `> 1e-3` assert. (Earlier drafts of this doc quoted specific
per-strip sums that no committed worker mode prints; those unreproducible numbers
are removed per the evidence-discipline rule — the seam-liveness conclusion rests
entirely on the `plane_seam` launcher.)

### Gates (2 processes × 1 GPU) — all green

Worker mode via `WITWIN_NCCL_ADJ_MODE`; pytest launchers in
`tests/fdtd/multi_gpu/test_nccl_transport.py`.

| Gate | Mode | Result |
|------|------|--------|
| S5 objective+grad parity, y-normal plane spanning the x seam | `plane` | `NCCL_ADJOINT_WORKER_OK[plane]` |
| S5 seam-ownership falsification: sum full local strip (double-count live seam cell) → parity red | `plane_seam` | grad rel 4.005e-1, loss rel 2.599e-1 (>> 1e-3 gate) → OK |

Parity tolerances mirror the point gates: loss rtol 5e-5 / atol 5e-6; grad rtol
1e-4 with an atol floor 1e-6·max|grad|. The plane scene is `_plane_scene` (open
boundary, y-normal `PlaneMonitor("plane", axis="y", position=0.0, fields=("Ez",))`,
source at x=−0.05 just left of the seam so both owned strips and the seam ghost are
live), 60 steps.

Reference oracle: an independent single-GPU adjoint on the same scene summing
`|Ez|²` over the FULL global plane (`result.monitors["plane"]["Ez"]`,
`loss.backward()`). The distributed owned strips tile that plane with no overlap, so
the world sum of the separable per-strip objective — and its gathered `grad_eps` —
must reproduce it. Single-GPU plane-monitor adjoint is itself covered by
`tests/gradients/test_fdtd_adjoint_rigorous.py::test_plane_flux_gradient_matches_fd`.

### Falsifications recorded (H1b)

- **Owned-strip seam rule is load-bearing (S5 seam-ownership).** Replacing the
  owned-strip sum with the FULL local-strip sum (`_plane_full_leaf_objective`,
  worker mode `plane_seam`) double-counts the live seam cell on both shards; the
  single-GPU full-plane reference then rejects the world sum and its gradient —
  grad rel 4.005e-1, loss rel 2.599e-1, both >> the 1e-3 gate. Permanent launcher
  `test_two_rank_nccl_adjoint_falsification[plane_seam-400]`.

### Fail-closed regression checks (verified this stage)

- In-process bridge (`allow_adjoint=False`): `require_distributed_adjoint_objective_support`
  still **rejects** a tiled plane scene (ValueError). Now covered by the committed
  `tests/fdtd/multi_gpu/test_guard_regressions.py::test_objective_guard_rejects_separable_plane_on_in_process_bridge`
  (which also positive-controls the `allow_adjoint=True` admit).
- Flux plane (`FluxMonitor`) is **rejected even under `allow_adjoint`** (not
  separable — `is_separable_plane_monitor` admits only the exact `PlaneMonitor`
  class). Now covered by the committed
  `tests/fdtd/multi_gpu/test_guard_regressions.py::test_objective_guard_rejects_flux_monitor_even_under_allow_adjoint`.
- H1a point-path gates unchanged: `standard` parity OK, `falsify_mag_halo` rel
  9.311e-1 OK, `guard_deadlock` OK (re-run this stage after the shared-plumbing
  edits to `_index_global_grads` / `_build_shard_seed_runtimes` / objective guard).

### Census reconciliation

Capability-guard census budget verified against THIS base (`18bc42a` lineage):
`CAPABILITY_GUARD_BUDGET = 175` (`tests/api/public/test_guard_census.py` passes
unchanged). S5 added no `raise NotImplementedError` and removed none; the
`allow_adjoint`-gated relaxations only re-condition existing `ValueError` fences,
and the added plane branches are ordinary routing (not census-tracked). Budget
stays **175**.

### FEATURE_LIST

The additive `h1-nccl-driver` subsection's final bullet is updated: the separable
y/z `PlaneMonitor` objective is now supported on the NCCL adjoint driver with the
seam-ownership guarantee; flux/mode/finite/x-plane and the in-process bridge stay
fail-closed.

### Test inventory (H1b)

- `tests/fdtd/multi_gpu/_nccl_adjoint_worker.py`: `_plane_scene`, modes `plane` /
  `plane_seam`, `plane_monitor_l2_objective` selection, plane reference, and the
  `_plane_full_leaf_objective` falsification.
- `tests/fdtd/multi_gpu/test_nccl_transport.py`: `plane` added to the parity
  parametrize, `plane_seam` to the falsification parametrize.

Commands (env exports as in the H1a header):

```bash
# S5 pytest launchers
python -m pytest \
  "tests/fdtd/multi_gpu/test_nccl_transport.py::test_two_rank_nccl_adjoint_parity[plane-400]" \
  "tests/fdtd/multi_gpu/test_nccl_transport.py::test_two_rank_nccl_adjoint_falsification[plane_seam-400]" \
  -q                                                    # -> 2 passed
# in-process no-regression (shared adjoint.py plumbing)
python -m pytest tests/fdtd/multi_gpu/test_adjoint_replay.py \
  tests/fdtd/multi_gpu/test_adjoint_parity.py \
  tests/fdtd/multi_gpu/test_adjoint_parity_cpml.py -q   # -> 33 passed
# adjacent (census + public + smoke + transport_adjoint + guard_regressions)
python -m pytest tests/api/public/test_guard_census.py tests/api/public/test_public_api.py \
  tests/api/public/test_simulation_smoke.py tests/fdtd/multi_gpu/test_transport_adjoint.py \
  tests/fdtd/multi_gpu/test_guard_regressions.py -q     # -> 57 passed
```

### Known gaps / deferred (H1b)

- Flux / mode / finite-plane / x-normal-plane adjoint objectives over NCCL remain
  fail-closed: they need seam-crossing tangential-field interpolation (the Poynting
  cross product / mode overlap is not a per-owned-cell separable sum), so the
  owned-strip trick does not apply. Rejected precisely by
  `require_distributed_adjoint_objective_support`.
- The in-process `transport="cuda_p2p"` bridge continues to reject every tiled
  monitor (its `_index_global_grads` path is point/full-field only; the plane path
  is exercised solely under `allow_adjoint` on the NCCL driver).
- No wall-clock/timing numbers produced (correctness-only shared GPUs).
- Census budget unchanged at 175.
