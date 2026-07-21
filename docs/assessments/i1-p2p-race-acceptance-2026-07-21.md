# I1 in-process cuda_p2p adjoint race — acceptance

Track: `i1-p2p-race`. Worktree: `.worktrees/wi1-p2p-race`, branch
`fable/p2p-race`. Base: master `16985a1` (clean). Hardware: 2x RTX A6000. Env:
`maxwell`. All commands export:

```bash
export CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=/home/xingyu/code/witwin/witwin-maxwell/.worktrees/wi1-p2p-race
export CUDA_VISIBLE_DEVICES=0,1
```

Correctness only (shared GPUs); no wall-clock/timing number is produced or
asserted.

---

## Summary

The in-process `transport="cuda_p2p"` distributed adjoint (the round-E
CPML-trainable flagship path) drifted its gathered gradient at the partition seam
under concurrent GPU load — deterministic rel ~8.09e-2 on a seam-straddling
density objective — while the forward output stayed **bitwise identical** and
exclusive/serialized execution was stable. The defect is a **checkpoint-capture
cross-stream happens-before race**, a *distinct* hazard from the NCCL path's
caching-allocator cross-stream reuse (the supervisor's persistent-staging /
`record_stream` directives targeted the reverse peer-copy halos, which were
already correct here). Reproducing first (directive 3) established the real site.

**Race class confirmation (directive 3).** On the reproduction scene under load:

| Condition | grad rel vs single-GPU | interpretation |
|-----------|------------------------|----------------|
| exclusive / `CUDA_LAUNCH_BLOCKING=1` | 2.175e-7 (floor), all rounds | async happens-before race |
| `PYTORCH_NO_CUDA_MEMORY_CACHING=1` under load | 8.090e-2 (unchanged) | **not** the allocator-reuse class |
| default, under co-tenant burner | 8.090e-2, deterministic | the race |

`CUDA_LAUNCH_BLOCKING=1` collapsing the drift to the floor while
`PYTORCH_NO_CUDA_MEMORY_CACHING=1` leaves it untouched is the discriminator: this
is an async ordering race, **not** the caching-allocator reuse class that the
NCCL fix addressed. The drift also scaled with checkpoint frequency — worst at
`adjoint_checkpoint_stride=1` (rel 1.4e-1..1.6e-1, a capture racing every
following step), absent at `stride == time_steps` (rel 2.175e-7, where the only
capture is the all-zero step-0 state) — which localizes it to checkpoint capture,
not the replay or the reverse halos.

## Root cause (single site)

`capture_distributed_checkpoint` (`witwin/maxwell/fdtd/distributed/adjoint.py`)
cloned each shard's persistent padded field storage on the device **default
stream** (`with torch.cuda.device(shard.device):`). The forward field updates
(`_advance_one_step` → `ShardEngine` phases) run on each shard's
**`compute_stream`**. The forward loop is:

```
if capture: _synchronize_all(); capture_distributed_checkpoint(distributed, n)
_advance_one_step(n)                       # writes field storage on compute_stream
```

The pre-capture `_synchronize_all()` orders the clone after the **previous**
step, but nothing orders the **next** `_advance_one_step` (compute_stream) after
the default-stream clone. The two streams are unordered, so under concurrent GPU
load the compute-stream update overwrote the field storage mid-clone; the replay
then reconstructed a torn trajectory and the gathered gradient drifted at the
seam. The forward loss stayed bitwise clean because it is collected separately
(`_collect_output`), not from the checkpoints.

## Per-site fix table (allocation/ordering site → mechanism)

| Site | Hazard | Mechanism chosen |
|------|--------|------------------|
| `capture_distributed_checkpoint` clone stream | default-stream clone unordered w.r.t. the next compute-stream forward update | **Clone on the shard's `compute_stream`** — serializes previous-update → clone → next-update on one stream (FIFO). Zero added host-sync; the existing pre-capture `_synchronize_all()` still guarantees the clone reads settled cross-stream/halo data. |
| `exchange_magnetic_adjoint` / `exchange_electric_adjoint` (reverse peer-copy halos) | (audited per directive) | **No change needed.** These already use persistent staging (`prepare_adjoint_staging`) and cross-device events, and the reverse driver fully host-synchronizes (`_synchronize(devices)`) before and after each halo. A scratch rewrite moving them to the default stream was **inert** (drift unchanged at 8.09e-2), confirming they are not the site; reverted. |
| forward-replay halos (`forward_electric_halo` / `forward_magnetic_halo`) | (audited) | **No change needed.** They run on the default stream inside `_synchronize`-bracketed replay steps. |

The fix is directive-family #2 (stream ordering to defeat the race by liveness),
realized as a same-stream serialization rather than an explicit event because the
producer and consumer are the same device's `compute_stream`.

## Fix

`witwin/maxwell/fdtd/distributed/adjoint.py`, `capture_distributed_checkpoint`:
clone each shard's state inside
`with torch.cuda.device(shard.device), torch.cuda.stream(shard.compute_stream):`.
Commit `1a579b3`.

## Stressed distribution (before / after)

Reproducible from the committed stressed nodes with `-s`
(`tests/fdtd/multi_gpu/test_adjoint_parity_stress.py`), co-tenant burner on both
boards, single-GPU adjoint as the reference, honest tolerances unchanged (loss
rtol 5e-5 / atol 5e-6; grad rtol 1e-4 with a 1e-6·max|grad| atol floor):

| Mode | grad rel vs single-GPU (6 rounds) | gate |
|------|-----------------------------------|------|
| standard, before fix (reverted, under load) | 8.090e-2 (deterministic) | 1e-4 → RED |
| standard, after fix (under load) | 2.175e-7 ×6 | 1e-4 → GREEN |
| x-CPML, after fix (under load) | 1.522e-7 .. 2.283e-7 ×6 | 1e-4 → GREEN |

## Falsification (directive 4c)

`test_p2p_capture_default_stream_falsification_under_stress`: under the same
co-tenant burner, monkeypatch `capture_distributed_checkpoint` back to a
default-stream clone (`_capture_on_default_stream`, byte-for-byte the pre-fix
capture). Recorded:

```
[falsification] reverted(default-stream) grad_rel=8.090e-2  restored(compute_stream) grad_rel=2.175e-7
```

The reverted clone drifts 8.090e-2 at the seam (>> the 1e-3 separation) while the
forward loss stays bitwise identical (asserted); restoring the compute_stream
clone returns the gradient to the 2.175e-7 floor. Exact site: the checkpoint
clone stream in `capture_distributed_checkpoint`.

## Zero-impact (directive 4d)

- Forward-only and single-GPU paths are untouched: the change is confined to
  `capture_distributed_checkpoint`, only reached by the distributed adjoint
  (checkpoint/replay) path. The unstressed in-process parity/replay/determinism
  suites (which pin bitwise `grad_eps` reproducibility) are unchanged and green.
- No tolerance changed anywhere. The stressed gate uses the same constants as the
  unstressed parity gates.

## Gates run (all green unless noted)

```bash
# fix no-regression (unstressed in-process adjoint parity + replay + CPML)
python -m pytest tests/fdtd/multi_gpu/test_adjoint_parity.py \
  tests/fdtd/multi_gpu/test_adjoint_replay.py \
  tests/fdtd/multi_gpu/test_adjoint_parity_cpml.py -q          # 33 passed

# committed stressed gate + falsification
python -m pytest tests/fdtd/multi_gpu/test_adjoint_parity_stress.py -q   # 3 passed (~81 s)

# census + public smoke
python -m pytest tests/api/public/test_guard_census.py \
  tests/api/public/test_public_api.py \
  tests/api/public/test_simulation_smoke.py -q                 # 30 passed

# gradients
python -m pytest tests/gradients/ -q      # 210 passed, 3 failed (FDFD, user-deferred)

# full multi_gpu directory
python -m pytest tests/fdtd/multi_gpu/ -q                      # see below
```

The 3 `tests/gradients/test_fdfd_adjoint.py` failures are the known
user-deferred FDFD failures (no nvmath), not touched by this change.

## Census reconciliation

`CAPABILITY_GUARD_BUDGET = 176` against this base (`16985a1`);
`tests/api/public/test_guard_census.py` passes unchanged. The fix adds/removes no
`NotImplementedError` guard (a stream-context change and a test), so the budget
stays **176**.

## FEATURE_LIST

The `<!-- END h1-nccl-driver -->` bullet that described the cuda_p2p hazard as
"the same class … tracked separately" is replaced with the fixed statement: the
cuda_p2p bridge is now load-safe, the drift was a distinct checkpoint-capture
happens-before race (not the allocator-reuse class), the fix clones on the shard
compute stream, and a committed stressed gate + falsification pins it.

## Known gaps / deferred

- No wall-clock/timing numbers (correctness-only shared GPUs).
- The stressed nodes spawn a co-tenant burner; on an already-saturated shared box
  they add load but the gate tolerances are unchanged and the fixed path is
  deterministic at the ~2e-7 floor, so they are not flaky.
