# FDTD Batch implementation evidence

Date: 2026-09-08. The complete Batch implementation is **not finished**.
The active plan is [the batch implementation plan](../plans/fdtd-batch-simulation-api-and-gpu-scheduling.md).

## Environment

- Python commands run in the `maxwell` conda environment, with PyTorch `2.13.0+cu130`.
- Sandbox GPU access fails; approved execution outside the sandbox sees two CUDA GPUs.
- GPU 1 was used for these tests. GPU 0 carried another user's existing compute workload and was not modified.
- Toolkit discovery requires `CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13`.
- No exclusive-machine or four-GPU performance acceptance is claimed.

## Completed subphase P0.1: baseline instrumentation

Commit: `25c133f`.

The storage manifest counts aliased tensor storage once, records owners/strides,
and explicitly excludes allocator reservations, opaque native allocations and
saved autograd tensors that cannot be enumerated through ordinary attributes.
The GPU sampler binds the requested device in its background thread.

The launch audit completed on GPU 1: live tensor storage at completion was
4,809,124 bytes, peak allocator allocation 4,933,632 bytes, peak reservation
6,291,456 bytes, and sampled driver increment 52,428,800 bytes. Unattributed
live allocator bytes were 50,780. Driver context/library allocations are not
mistaken for tensor storage. This is one instrumentation check, not a speed gate.

Raw audit: `benchmark/plots/batch/launch-audit.json`.

## Completed subphase P1.1: declaration and planning API

Commit: `1caa879`.

The public Batch descriptors, list/template entrypoints, named target checks,
output conflicts, density tensor views, and provisional declaration grouping
are implemented. Reports always remain unresolved until compiled material,
layout, capacity and runtime proofs are implemented. `prepare/run` reject;
there is no hidden independent-solver execution inside the Batch API.

Validation: 43 new public Batch tests passed. The public API, simulation smoke,
Batch and ensemble guard suite produced **79 passed, 3 skipped** on GPU 1.
The skipped legs required multiple visible devices in that invocation.

## Subphase P0.2: concurrent benchmark tooling and first measurement

Commit: `82d6467`. Ten benchmark instrumentation/report tests pass. Ordinary
benchmark report regeneration preserves the tagged Batch baseline sections.

Three modes use identical per-variant scenes and exported complex probe spectra:
independent sequential runs, persistent spawned worker processes, and existing
`run_many`. Cold timing includes worker startup and final export readiness;
warm timing reuses workers and extension caches but prepares fresh solvers.
No deduplication is performed. The CLI accepts arbitrary positive N and provides
identical-copy controls, workload selection and explicit step overrides.

Measured launch workload: N=10, 2,000 time steps, two warmup rounds, five timed
rounds, one local material change per variant, eager stepping for all modes.

| Mode | Warm median (s) | Minimum–maximum (s) |
| --- | ---: | ---: |
| sequential | 4.881813 | 3.818385–4.943592 |
| processes | 0.611841 | 0.534666–0.845338 |
| run_many | 11.372335 | 9.990380–16.252934 |

All exported probe comparisons were exactly equal to the sequential reference.
These measurements show that the independent-process baseline is already strong
on this launch-heavy workload. They establish no claim about the new Batch
runtime, which is still unavailable.

Raw repeated-run data: `benchmark/plots/batch/launch-n10-varied.json`.
This file predates the added isolated-memory leg; its driver totals include idle
sessions and cannot be used as isolated per-mode memory comparisons.

The later isolated-memory protocol uses one fresh mode session at a time.
Its three-variant, 32-step smoke check completed with matching spectra; see
`benchmark/plots/batch/launch-n3-smoke.json`. These short smoke measurements are
not substituted for the 2,000-step performance results above.

Remaining P0 acceptance: N=1/100 and identical-copy runs, the bandwidth and
31-frequency workloads, isolated N=10 memory measurements, environment/power
provenance for performance signoff, and the user's original business geometry.
The historical 1.11/11.07 GiB figures have not been reproduced.

Reproduction commands:

```bash
conda activate maxwell
export CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13
python -m benchmark batch --inventory
python -m benchmark batch --audit launch --device cuda:1
python -m benchmark batch --compare launch --device cuda:1 --variants 10 --update-report
```

## Remaining execution work

P3.1 (`6150023`) adds private native batch field-update operators using the existing E/H
and electric interior kernels, with 64-bit lane offsets, shared or per-lane
coefficients and an int32 active mask. The private operators reject invalid
Yee shapes, overlapping writable/read buffers, unsupported boundary modes and
spatial extents below two. They use the caller's current CUDA stream.

Validation: **94 batch GPU tests passed**, including B=1/2/3/4/8/10,
all six components, real periodic/PEC/PMC/none boundaries, inactive poisoned
lanes and CUDA Graph replay after mask updates. All per-component comparisons
were bitwise equal to independent native calls. **86 existing regressions
passed** with native-extension gates enabled, covering field/CPML kernels,
Graph replay, CPML solves, the existing adjoint bridge and modulated slabs.
The earlier 15 opt-in skips were subsequently enabled and exercised in this
regression run. This is not complete-batch throughput, memory-sanitizer,
cross-architecture, CPML-batch or native-batch-adjoint acceptance.

The P2.1 (`115f522`) material-sampling foundation extends the existing edge compiler with
contiguous component selections and a radius-one differentiation halo. It uses
the same numerical formulas as full-component compilation. Six Yee components,
uniform and graded grids, arithmetic/polarized subpixel sampling, filtered and
projected density gradients, corner crops and invalid selections were verified:
43 new GPU checks plus 42 existing compiler checks, **85 passed**. Partial
periodic-axis selection fails explicitly pending opposite-face routing. This
does not establish compact influence support or a shared-background cache.

The native CPML/cooperative-timestep subphase adds dense and slab batch kernels
by extending the original kernels with lane addressing, including the dense
electric interior path. Keeping the original dense arithmetic matters: replacing
it with the algebraically equivalent flat slab expression changed a weak CPML
component's relative norm by approximately 1.9e-4 after 80 steps. The final
implementation preserves each storage mode's original expressions.

`NativeFieldSlot` retains fixed-capacity field, CPML and coefficient buffers;
compiled inverse spacing and CPML profiles share storage only after value equality
checks. `solve_steps(cooperative=True)` suspends before each complete field block,
while source injection, observation and normalization reuse ordinary `solve`.
Tail lanes are inactive and independently indexed. Reloading a compatible cohort
preserves buffer pointers, and repeated runs reset field and CPML state.

Validation: **167 GPU tests passed** across native field tests, complete batch
cohorts, CPML regressions and the existing adjoint bridge. Cohort checks cover
B=1/3/4, incomplete tails, none/periodic/CPML boundaries, material and source
changes, repeated resets, slab storage and 31-frequency observation. This does
not establish throughput, allocation-budget or batch-gradient acceptance.

Compiled sharing proofs, regional coefficient updates, native batched sources
and observers, coordinated graph execution, byte-level admission, multi-GPU
scheduling, public result/tape lifetimes and batch adjoints are not accepted yet.
Advanced physics and the response-bank research gate remain open.
The active plan must not be archived or marked complete based on this evidence.
