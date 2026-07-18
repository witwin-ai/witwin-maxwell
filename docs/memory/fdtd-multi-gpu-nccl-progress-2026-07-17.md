# FDTD Multi-GPU NCCL Forward — ShardEngine Progress and Audit Closure

- Date: 2026-07-17
- Branch: `codex/shard-engine`
- Scope: one-process-per-GPU (`torchrun`) NCCL forward field solve, built on the
  `ShardEngine` + coordinator split.

This note records the audit-closure work for the two ACCEPT_WITH_FIXES reviews of
the NCCL forward slice, plus two commit-message corrections the auditors found.

## Transport branch scope (truthful statement)

The runtime split expresses every cross-rank operation through transport
primitives, so the coordinator's **per-step time loop is branch-free over the
transport primitives** (`exchange_electric`/`exchange_magnetic`,
`reduce_owned_energy`, `gather_component_slabs`, `gather_monitor_payloads`,
`gather_stats`). The earlier "no per-transport behavior branch" wording was
overstated: `DistributedFDTD` does branch on `self._nccl` outside the per-step
loop, in construction/validation/gather scope. The branch sites in
`witwin/maxwell/fdtd/distributed/solver.py` are approximately:

- `:227` transport selection (`self._nccl = parallel.transport == "nccl"`);
- `:272` transport construction (NCCL `from_env` vs. in-process P2P);
- `:575` NCCL-specific hardware preflight (skips the peer-access matrix);
- `:657` rank-local single-engine build vs. all-shard build + layout binding;
- `:697` owner-resident circuit/network/wire runtime prepare (P2P only);
- `:762` overlap gate (NCCL forces the serialized schedule);
- `:1024` field-shutoff gate (NCCL fails closed on `shutoff > 0`).

`_validate_nccl_capabilities` (`:479`) is the narrower NCCL capability envelope.
FEATURE_LIST.md and `docs/reference/fdtd-multi-gpu-joint-solve.md` were reworded
to state the per-step-loop scope explicitly rather than a whole-class claim.

## Audit findings closed

1. **[MAJOR] Vacuous conformance gate at 48 steps.** The forward worker
   (`tests/fdtd/multi_gpu/_nccl_forward_worker.py`) ran only 48 steps, so the
   Gaussian pulse had not crossed the x=0 seam and a no-op'd electric halo still
   passed within atol 5e-6. Fix: `_STEPS` raised to 160, and an in-worker
   precondition (`_assert_seam_carries_signal`) now asserts, on the independent
   single-GPU reference DFT at the drive frequency, that the seam-plane amplitude
   of at least one gathered E component reaches `_SEAM_AMPLITUDE_FLOOR = 1e-2`
   relative to that component's domain max. Measured single-GPU seam/domain-max
   ratios (largest component): 48 steps 6.1e-3 (below floor), 120 steps 1.3e-2,
   160 steps 2.65e-2 (~2.6x margin). The floor is justified: pre-crossing configs
   sit at 6.1e-3 and fail the precondition, so the gate can never silently regress
   to a pre-seam-crossing step count.

   Falsification (2xA6000, `torchrun --nproc-per-node=2`):
   - no-op `exchange_electric` (skip plane exchange, still record the event) →
     worker exits nonzero; `assert_close` fails at 25.7% mismatched elements,
     greatest abs diff 3.50 at index (0,5,4,2).
   - no-op `exchange_magnetic` likewise → nonzero exit; 30.0% mismatch, greatest
     abs diff 3.50 at index (0,6,4,2). Both decisive.
   - regress `_STEPS` to 48 → the precondition fires first with ratios
     `{Ex: 6.1e-3, Ey: 6.9e-5, Ez: 4.2e-4}` below the 1e-2 floor (nonzero exit),
     confirming the regression guard.
   - clean 160-step run prints `NCCL_FORWARD_WORKER_OK` (exit 0).

2. **[MAJOR] NCCL fail-closed fences had no pinning tests.** Added host-level
   tests in `tests/fdtd/multi_gpu/test_nccl_transport.py` that set
   RANK/WORLD_SIZE/LOCAL_RANK so `from_env` binds without a launcher, then assert
   each fence's specific message: monitors
   (`test_nccl_forward_rejects_monitors`), coupled circuit/network/wire/port
   (`test_nccl_forward_rejects_coupled_runtimes[wire|port]`), trainable density
   (`test_nccl_forward_rejects_trainable_density`), and field shutoff
   (`test_nccl_forward_rejects_field_shutoff`, driven through `solve(shutoff>0)`
   which raises before `init_field`). Falsified by neutralizing each fence's
   condition (`and False`) → all red (monitors/density/wire: no raise;
   port: raises the unrelated "unbound ports" message; shutoff: reaches CUDA
   preflight) → restored → green.

3. **[MINOR] Overstated transport-branch claim** — reworded (see above).

4. **[MINOR] `gather_stats` reported misleading numbers on NCCL.** A rank-local
   NCCL transport cannot see both sides of any halo, yet `gather_stats` returned
   `halo_bytes_per_step: 0` and rank-local partitions/peak-memory as if global.
   Now returns `halo_bytes_per_step: None` and a `rank_local: True` marker;
   `_build_parallel_stats` propagates `stats_rank_local` and emits
   `halo_bytes_per_step`/`halo_bytes_total` as `None` on the rank-local path
   instead of coercing to 0.

5. **[MINOR] `gather_component_slabs` could `dist.recv` on a non-bound device.**
   When `result_device` differed from rank 0's NCCL-bound device, rank 0 would
   post `dist.recv` into a buffer on an unbound device. Added a fail-fast
   `ValueError` naming the constraint, pinned by
   `test_gather_component_slabs_rejects_non_bound_result_device`. Falsified by
   neutralizing the check → the test hits a downstream `AttributeError` instead →
   restored → green.

## Commit-message corrections (auditor-found)

- **809347a** ("end-to-end one-process-per-GPU NCCL forward solve") states
  "236 passed" for the multi-GPU suite; the actual count is **233 passed**. The
  commit message is immutable history; the correct count is 233 (collected and
  re-verified during this audit closure).
- **002c749** ("split distributed solver into ShardEngine + coordinator") claimed
  a bitwise baseline but its baseline artifacts were **not retained**. The
  auditors independently re-derived bit-identity via a `git archive` checkout of
  the pre-refactor tree and confirmed the split is bit-identical; the artifacts
  themselves are gone, so the claim rests on that independent re-derivation rather
  than on retained files.
