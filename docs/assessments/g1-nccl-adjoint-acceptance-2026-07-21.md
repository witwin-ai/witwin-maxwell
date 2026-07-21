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
