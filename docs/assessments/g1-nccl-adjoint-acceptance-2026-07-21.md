# G1 NCCL adjoint — acceptance (stage G1a)

Track: `g1-nccl-adjoint`. Worktree: `.worktrees/wg1-nccl-adjoint`, branch `fable/nccl-adjoint`.
Hardware: 2x RTX A6000 (NV4). Env: `maxwell`. All commands below set
`CUDA_HOME=.../nvidia/cu13`, `PATH=$CUDA_HOME/bin:$PATH`,
`PYTHONPATH=<this worktree>`.

## Scope delivered (G1a)

Stage G1a = **transport-level transpose identity + NCCL reverse worker (standard
scene) + parity/determinism/falsification gates**. This stage delivers the
mandated-first item in full (the discrete-transpose test written first at the
transport level, per supervisor decision 1) and, after establishing the
architecture, records the end-to-end trainable-density NCCL reverse as a scoped,
fail-closed deferral (design-blocker clause) with a concrete implementation plan.

### Delivered

1. **NCCL reverse-halo adjoint transport** (`witwin/maxwell/fdtd/distributed/nccl_transport.py`).
   Engine-based coordinator primitives mirroring the in-process reference transport:
   - `prepare_adjoint_staging(engines, adjoint_states)` — preallocate the rank-local
     per-kind staging planes (magnetic receives the right neighbour's low ghost;
     electric receives the left neighbour's high ghost node); endpoints allocate
     nothing on the missing side.
   - `exchange_magnetic_adjoint(engines, adjoint_states)` — transpose of the forward
     magnetic halo: ship this rank's low ghost adjoint Hy/Hz plane left and zero it;
     receive the right neighbour's ghost into staging and `add_` into the last owned
     cell.
   - `exchange_electric_adjoint(engines, adjoint_states)` — transpose of the forward
     electric halo: ship the high ghost adjoint Ey/Ez node plane right and zero it;
     receive the left neighbour's ghost into staging and `add_` into the first owned
     node.
   Reuses the existing transposed plane primitives (`_magnetic_adjoint_planes` /
   `_electric_adjoint_planes`); preallocated staging keyed by halo kind ⇒ no per-step
   allocation. Same signature contract (`engines`, rank-keyed `adjoint_states`) the
   in-process `CudaP2PHaloTransport` exposes, so the reverse loop never branches on
   transport kind.

2. **Transport-level discrete-transpose identity gate** (2-rank `torchrun` worker
   `tests/fdtd/multi_gpu/_nccl_transport_adjoint_worker.py`; launchers in
   `tests/fdtd/multi_gpu/test_nccl_transport.py`).
   For both the magnetic and electric Yee x-halos, on two real processes:
   - inner-product identity `<A x, y> == <x, A^T y>` (each side formed on one rank of
     the pair, combined by `all_reduce`; pure-copy halos ⇒ asserted **bitwise**,
     atol == 0), driven through the new engine-based exchanges over
     `SimpleNamespace` engines on a real `FDTDPartitionPlan` layout;
   - ghost-adjoint-zero invariant after the shipped-away ghost is sent;
   - bitwise determinism of the accumulated owner across two repeats.

### Fail-closed deferral (design blocker, documented per common brief)

**End-to-end trainable-density NCCL reverse + objective/gathered-gradient parity vs
single-GPU (gate 2a/2b/2c) is NOT delivered this stage.** It is retained
fail-closed: `_validate_nccl_capabilities` still raises
`"Multi-GPU NCCL adjoint (trainable density) is not wired yet; run the trainable
scene with transport='cuda_p2p'."` (unchanged; covered by
`test_nccl_forward_rejects_trainable_density`). No capability guard was relaxed, so
the census budget (176) is unchanged.

Reason (architectural, not a stall): the validated in-process adjoint bridge
(`fdtd/distributed/adjoint.py`, `_DistributedFDTDGradientBridge`) is structurally
single-process — it holds every shard in one process, obtains the global monitor/
field output from `DistributedFDTD._collect_output()` (which returns `None` on
non-root NCCL ranks), and drives a single-process `torch.autograd.Function`. A
torchrun one-process-per-GPU reverse therefore needs a **new distributed-collective
reverse driver**, not a per-rank call of the existing bridge. Concretely it requires:
(i) a **distributed forward replay**: the in-process `replay_distributed_segment`
copies interface halos with in-process tensor-dict operations
(`_forward_electric_halo` / `_forward_magnetic_halo`) that are no-ops with a single
local shard, so the interface mid-step H would be wrong — replay must instead drive
dict-based **forward** NCCL halos; (ii) a **per-rank reverse loop** using the
now-landed adjoint halos (this stage's deliverable) around the reused per-engine
reverse cores (`reverse_phase1/2/3_*`, `_accumulate_source_term_gradients`); (iii)
**local separable seeds** (a full-field-DFT objective `sum|E_owned|^2` is separable,
so each rank seeds its own owned DFT output — no cotangent scatter needed for the
standard gate); (iv) **grad_eps gather to rank 0** (transpose of
`gather_component_slabs`, already NCCL-capable in the forward direction) and the
existing single-GPU material pullback on rank 0 only. This is a multi-file subsystem;
this stage lands and independently verifies its transport foundation (the transposed
halos are provably exact transposes), which is the load-bearing correctness
prerequisite for that driver.

## Test inventory (executed on 2x A6000)

New/edited:
- `tests/fdtd/multi_gpu/_nccl_transport_adjoint_worker.py` (new worker).
- `tests/fdtd/multi_gpu/test_nccl_transport.py` (+3 launcher nodes:
  `test_two_rank_nccl_reverse_halo_transpose_identity`,
  `test_two_rank_nccl_reverse_transpose_identity_falsification[magnetic|electric]`).

Direct worker runs (via `python -m torch.distributed.run --standalone --nnodes=1
--nproc-per-node=2`):
- clean: prints `NCCL_TRANSPOSE_ADJOINT_WORKER_OK`, exit 0.
- `NCCL_TRANSPOSE_FALSIFY=magnetic`: exit 1 (identity assertion fires).
- `NCCL_TRANSPOSE_FALSIFY=electric`: exit 1 (identity assertion fires).

Pytest (pass counts):
- `tests/fdtd/multi_gpu/test_nccl_transport.py -k "transpose or roundtrip"` +
  `tests/fdtd/multi_gpu/test_transport_adjoint.py` +
  `tests/api/public/test_guard_census.py` → **6 passed, 30 deselected** (guard census
  deselected by the `-k` filter; run separately below).
- `tests/api/public/test_guard_census.py tests/api/public/test_public_api.py
  tests/api/public/test_simulation_smoke.py` → **30 passed** (census budget 176
  intact).

Exact commands:
```
export CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=/home/xingyu/code/witwin/witwin-maxwell/.worktrees/wg1-nccl-adjoint
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# transport transpose-identity worker (clean + falsification)
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  tests/fdtd/multi_gpu/_nccl_transport_adjoint_worker.py
NCCL_TRANSPOSE_FALSIFY=magnetic python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  tests/fdtd/multi_gpu/_nccl_transport_adjoint_worker.py   # exit 1
NCCL_TRANSPOSE_FALSIFY=electric python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  tests/fdtd/multi_gpu/_nccl_transport_adjoint_worker.py   # exit 1
# pytest launchers + adjacent suites
python -m pytest tests/fdtd/multi_gpu/test_nccl_transport.py -k "transpose or roundtrip" \
  tests/fdtd/multi_gpu/test_transport_adjoint.py -q
python -m pytest tests/api/public/test_guard_census.py tests/api/public/test_public_api.py \
  tests/api/public/test_simulation_smoke.py -q
```

## Falsifications recorded

- **Transpose identity has teeth**: `NCCL_TRANSPOSE_FALSIFY=magnetic|electric` drops
  the accumulated owner plane on the receiving rank after the (still-symmetric)
  collective, so `<x, A^T y>` no longer equals `<A x, y>`; the worker's equality
  assertion fires and the process exits nonzero. Both modes verified exit 1;
  permanent launcher nodes assert the nonzero exit. (First iteration exposed and
  fixed a real test bug — computing `<A x, y>` from the ghost *after* the adjoint had
  correctly zeroed it — and a deadlock from skipping the collective on one rank; the
  final falsification keeps the collective symmetric and corrupts only the local
  accumulation.)

## Known gaps / deferred

- End-to-end NCCL trainable-density reverse + parity/determinism/falsification on the
  standard dielectric Box scene (gate 2a/2b/2c) — deferred as above, fail-closed
  retained. This is the primary remaining G1a item; plan recorded above.
- CPML psi-carrying NCCL reverse + psi-active parity, timing hooks, S5 tiled seeds —
  stage G1b (unchanged scope).
- No wall-clock/timing numbers produced (correctness-only shared GPUs, per brief).

## Notes for the next agent

- The engine-based adjoint exchanges are drop-in for a distributed reverse driver:
  same `(engines, adjoint_states)` signature as `CudaP2PHaloTransport`, so the reverse
  loop body can stay transport-agnostic exactly like the forward loop.
- Build the end-to-end driver as a **collective** reverse (each rank runs its own
  process), NOT by calling `_DistributedFDTDGradientBridge` per rank — the bridge is
  single-process (global output + single autograd graph on rank 0). Reuse the
  per-engine cores and this stage's adjoint halos; add a distributed replay with
  dict-based forward NCCL halos and a `scatter`/local-seed path (a separable
  full-field-DFT objective avoids a cotangent scatter for the standard gate).
- Census budget unchanged at 176; no guard added or relaxed.

---

# G1 NCCL adjoint — acceptance (stage G1b)

Stage G1b scope: **CPML psi-carrying NCCL reverse + psi-active parity + opt-in
timing hooks (zero-cost-off asserted) + S5 tiled-seed stretch or deferral +
docs/census/FEATURE_LIST**. Same hardware/env/PYTHONPATH preamble as G1a.

## Delivered (G1b)

### 1. Opt-in per-rank step-rate timing instrument (supervisor decision 3) — DONE

`witwin/maxwell/fdtd/distributed/instrumentation.py` :: `StepRateInstrument`.
Env-gated (`WITWIN_FDTD_STEP_TIMING`, off by default; `WITWIN_FDTD_STEP_TIMING_DIR`
selects the output directory; `WITWIN_FDTD_STEP_TIMING_STEPS` sizes the worker's
timing pass). Brackets a rank's time loop (`loop_begin` / `step_begin` /
`step_end` / `loop_end`) and `finalize()` writes one machine-readable
`step_timing_rank{r}.json` per rank (schema `witwin.fdtd.step_timing/1`: per-step
wall mean/median/min/max/p95 in ms, aggregate steps-per-second, plus
rank/world-size/device/step-count).

* **Zero-cost-off (asserted):** with the env var unset the bracket calls return
  immediately and never synchronize the device — a unit test injects a counting
  `synchronize` stand-in and asserts **0** synchronizations across a full disabled
  loop and that **no** artifact is written; the enabled path synchronizes exactly
  `2 + 2·steps` times (bracketing each step so the recorded interval reflects
  completed GPU work, not launch latency).
* **NCCL worker wiring:** `_nccl_forward_worker.py` runs the instrument as an
  opt-in collective pass after its parity solve; disabled (default, and always in
  CI) it drives no collective and is byte-identical to the un-instrumented run, so
  the exclusive-GPU window can flip the env var and collect per-rank JSON later. No
  wall-clock number is asserted anywhere.

Files:
- `witwin/maxwell/fdtd/distributed/instrumentation.py` (new).
- `tests/fdtd/multi_gpu/test_step_instrument.py` (new host unit tests).
- `tests/fdtd/multi_gpu/_nccl_forward_worker.py` (`_run_timing_pass`, opt-in).
- `tests/fdtd/multi_gpu/test_nccl_transport.py`
  (`test_two_rank_nccl_step_timing_emits_per_rank_json` launcher).
- `FEATURE_LIST.md` (additive G1b subsection).

Commit: `feat(fdtd-distributed): opt-in per-rank step-rate timing instrument (zero-cost-off)`.

### 2. CPML psi-carrying NCCL reverse + psi-active parity (decisions 1–2) — FAIL-CLOSED DEFERRAL

Retained fail-closed (unchanged guard: `_validate_nccl_capabilities` still raises
`"Multi-GPU NCCL adjoint (trainable density) is not wired yet; ..."`; census
budget 176 unchanged). Reason (design blocker, per common-brief clause): the CPML
psi-carrying NCCL reverse layers directly on top of the **end-to-end
trainable-density NCCL reverse driver** that G1a deferred as a multi-file
subsystem. That base driver does not exist yet, so the CPML variant has no host to
extend. Verifying it to the mandated gates (objective + gathered-gradient parity
vs the single-GPU reference on the standard dielectric Box AND the psi-active CPML
scene, repeat-determinism, and the no-op-halo / zero-psi-carry falsifications —
mirroring `tests/fdtd/multi_gpu/test_adjoint_parity_cpml.py` over NCCL) requires a
working, green driver; shipping an unverified per-rank gradient path would violate
the fail-closed-over-unverified-gradient rule and evidence discipline. It is
therefore recorded as a documented deferral rather than a partial/flaky landing.

**What is already proven (the load-bearing prerequisites):**
- Transport-level discrete-transpose identity of both NCCL reverse halos across 2
  real processes (G1a, bitwise) — the reverse loop's cross-interface transport is
  exact.
- The in-process CPML psi-carrying reverse itself is validated end-to-end
  (`_DistributedFDTDGradientBridge._reverse_phases_cpml`, no psi halo, S4 audit +
  `test_adjoint_parity_cpml.py`): the per-engine reverse cores and the "field
  halos carry all cross-interface coupling, psi stays rank-local" result the NCCL
  driver will reuse verbatim are already gated.

**Concrete remaining subsystem for the next agent (executable order):**
1. **Relax the entry guard** narrowly: add an internal `allow_adjoint` path so an
   adjoint driver may construct the per-rank `DistributedFDTD` with a point-monitor
   (or full-field-DFT) trainable scene without tripping the forward-only
   monitor/trainable fences — keep every other fence and add a census-tracked
   guard if the relaxation widens the envelope (reconcile the 176 budget then).
2. **Per-rank checkpoint capture** (`capture_checkpoint_state(shard.solver, step)`
   for the single local shard) inside the real NCCL forward loop
   (`_advance_one_step`, collective).
3. **NCCL forward-replay dict halos** — the one genuinely new transport primitive:
   dict-based forward electric/magnetic x-plane send/recv (transpose siblings of
   the G1a adjoint halos) so `replay_distributed_segment`'s in-process tensor-dict
   halos (no-ops with one local shard) become collective. Gate it standalone: a
   2-process worker replaying a checkpoint segment must reproduce a second
   independent NCCL `solve()`'s owned+interface state to reduction-order drift
   (rtol 1e-5 / atol 1e-7, matching the in-process replay-parity gate).
4. **Local separable seed:** for the point-monitor objective (as in
   `test_adjoint_parity_cpml.py`) only the owning rank has a nonzero seed — read
   its shard-local monitor output and build the seed with `_build_output_seeds`;
   every other rank seeds zero and receives adjoint solely through the halos.
   (Avoids the global monitor gather NCCL forward rejects.)
5. **Per-rank reverse loop:** reuse the per-engine reverse cores + this track's
   NCCL adjoint halos exactly as `_reverse_phases_standard` / `_reverse_phases_cpml`
   already do — those methods iterate `distributed.shards` (which is `(local,)` per
   NCCL rank) and call `distributed.transport.exchange_*_adjoint`, so they are
   already transport-agnostic; factor them to module-level free functions (re-run
   the in-process CPML parity suite to prove no regression) OR reimplement the
   ~40-line orchestration in the driver.
6. **grad_eps gather + rank-0 pullback:** `distributed._gather_component` (NCCL:
   global on rank 0, `None` elsewhere) then `pullback_material_input_gradients`
   on rank 0; compare loss + density grad against a single-GPU trainable reference
   built in the same worker on rank 0 (standard scene at the plan monitor gate;
   CPML psi-active scene at rtol 1e-4 / atol-floor 1e-6·max|grad|, `_PSI_STEPS`/
   `_PSI_MONITOR_X` from the in-process test).
7. **Gates:** parity (2a), no-op-one-adjoint-halo falsification (2b),
   repeat-determinism of gathered grad_eps (2c), and the psi-active variant + the
   zero-psi-carry falsification.

### 3. S5 tiled monitor seeds (decision 4, stretch) — DEFERRED

Deferred, as the decision permits ("stretch only if 1–3 are green and
audited-quality"). Item 2 (the CPML psi-carrying reverse) is not green, so S5 is
not entered. The plane/flux/mode cotangent scatter across shards remains rejected
fail-closed by `require_distributed_adjoint_objective_support` (unchanged); the
scatter design is the S3 deferral (`fdtd-joint-solve-adjoint-s4-progress`
step 5 / `_scatter_field_grad_to_shard`'s per-owned-interval transpose applied to
tiled monitor cotangents). No code change; no census impact.

## Test inventory (G1b, executed on 2x A6000)

Exact commands (preamble: `CUDA_HOME=.../nvidia/cu13`, `PATH=$CUDA_HOME/bin:$PATH`,
`PYTHONPATH=<this worktree>`, `CUDA_VISIBLE_DEVICES=0,1`,
`TORCH_NCCL_ASYNC_ERROR_HANDLING=1`):

```
# instrument host unit tests + full NCCL transport suite + census + public/smoke
python -m pytest tests/fdtd/multi_gpu/test_step_instrument.py \
  tests/fdtd/multi_gpu/test_nccl_transport.py \
  tests/api/public/test_guard_census.py tests/api/public/test_public_api.py \
  tests/api/public/test_simulation_smoke.py -q
#   -> 65 passed

# opt-in timing pass emits per-rank JSON in the real NCCL worker (schema only)
WITWIN_FDTD_STEP_TIMING=1 WITWIN_FDTD_STEP_TIMING_DIR=<dir> WITWIN_FDTD_STEP_TIMING_STEPS=64 \
  python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  tests/fdtd/multi_gpu/_nccl_forward_worker.py
#   -> NCCL_FORWARD_WORKER_OK; step_timing_rank0.json + step_timing_rank1.json written
```

- `test_step_instrument.py`: **5 passed** (disabled-by-default; zero-cost-off with
  0 synchronizations + no artifact; enabled schema + `2+2N` sync count; per-rank
  distinct files; `step_end`-without-`step_begin` raises).
- `test_nccl_transport.py`: full suite green including
  `test_two_rank_nccl_step_timing_emits_per_rank_json` (new), the G1a forward /
  transpose-identity / falsification launchers, and the host guard matrix.
- `test_guard_census.py`: budget **176**, unchanged (no guard added/relaxed).

## Falsifications recorded (G1b)

- **Zero-cost-off has teeth:** editing `StepRateInstrument.step_begin` to
  synchronize *before* the `enabled` guard (i.e. even when disabled) makes
  `test_disabled_never_synchronizes_and_writes_nothing` fail with
  `assert 25 == 0` (25 = one synchronize per disabled step across the 25-step
  loop). Restored → green. This proves the zero-cost-off assertion catches an
  unconditional per-step synchronize regression.

## Known gaps / deferred (G1b)

- CPML psi-carrying NCCL reverse + psi-active parity: fail-closed deferral with the
  executable subsystem plan above (blocked on the G1a-deferred base driver).
- S5 tiled monitor seeds: deferred (gated on item 2).
- No wall-clock/timing numbers produced (correctness-only shared GPUs); the timing
  instrument is delivered and unit-tested but the exclusive window produces the
  numbers.
- Census budget unchanged at 176; no guard added or relaxed.
