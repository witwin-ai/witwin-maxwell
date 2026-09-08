# FDTD Batch implementation evidence

Date: 2026-09-08. The complete Batch implementation is **not finished**.
The active plan is [the batch implementation plan](../plans/fdtd-batch-simulation-api-and-gpu-scheduling.md).

## Optimization follow-up: full native pipeline and coordinated graphs

The historical eager limitations below are superseded for the supported single
GPU forward subset. Native source injection and observer accumulation batch lanes
and frequencies. Persistent buffers support complete timestep capture, exact tail
chunks, independently stopping lanes and monitor-only capture reuse. Full-field
DFT buffers currently require recapture on repeated runs. Registered Batch work
and existing capture suspension cooperate under a capture lock; arbitrary external
GPU threads still require a quiescent capture window. Reports record actual eager,
graph or mixed execution and any automatic fallback reason.

Graph/public regression coverage: **54 passed**; final focused rerun after report
changes: **51 passed**. Benchmark CLI/report coverage: **11 passed**. Source tests
include bitwise periodic corner and overlapping injection parity; explicit native
floating-point contraction preserves ordinary solver rounding. Equal compiled
coefficients share storage and promote to per-lane storage when needed. Inactive
material channels and empty dispersion templates no longer allocate full volumes.

Fresh comparison: N=B=10, 2000 steps, two warmups, five formal repeats, physical
GPU 1. Command: `python -m benchmark batch --compare launch --variants 10
--concurrency 10 --include-native --native-graph required --warmups 2 --repeats 5
--output benchmark/plots/batch/optimized-launch-n10-graph.json --update-report`.

| Mode | Warm median (s) | Range (s) | Isolated driver increment (MiB) |
| --- | ---: | ---: | ---: |
| sequential | 3.615720 | 3.607710–3.646495 | 315.5625 |
| processes | 0.556994 | 0.507809–0.609569 | 3157.6250 |
| run_many | 10.742862 | 9.617411–13.272421 | 351.5625 |
| native_batch, required graph | 0.126706 | 0.125081–0.129508 | 401.5625 |

All probe comparisons have zero error. Native graph is **4.40 times faster than
ten processes**, with **87.3% less sampled driver increment** in this workload.
The combined throughput/memory gate passes for this measured small-grid workload.
It does not certify the large-grid, original business geometry or complete N/B
matrix. Driver memory uses 20 ms sampling. This fresh comparison supersedes the
earlier small-grid gate failure without treating measurements from different
sessions as a controlled before/after timing comparison.

Strict byte admission, throughput tuning, multi-GPU/two-slot scheduling, regional
background compilation and Batch adjoints remain separate unaccepted capabilities.

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
layout, capacity and runtime proofs are implemented. At this historical commit,
`prepare/run` rejected; the executable forward subphase below supersedes that limitation.

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
runtime, which had not been connected when those measurements were collected.

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

## Public forward execution and result lifetime subphase

`BatchSimulation.prepare/run` now execute supported real static-material FDTD
through `PreparedBatch`, one GPU, one persistent field slot and native resident
lanes. Source and monitor loops remain per lane; the time-marching field block is
one native update per component across B lanes. The implementation never launches
B independent `Simulation.run()` calls to represent a native field batch.

The public tier provides repeated runs, density/source-scale template updates,
tail cohorts, compatible-group input/completion order, cancellation, failure
records, independent Result storage, explicit stacking and directory persistence.
Top-level iterators own their prepared contexts and must be closed on early exit.
The demand-driven iterator has no background completed-result queue. Result's
plotting facade is now created on demand to avoid a Result/facade reference cycle;
GPU outputs release without waiting for cyclic collection.

Validation: **89 public Batch, ordinary public API/smoke and graph tests passed**.
The execution tests cover B=1/2/3/4, N=1/2/3/5/7, density updates, source A/B/A,
in-place parameter version changes, repeated buffer addresses, old results after
close, incomplete tails, 31-frequency monitor-only execution without full-field
DFT, CPU export, persistence, early/unstarted iterator close, cancellation,
injected runtime failure and post-construction trainable Scene detection. With
cyclic GC disabled, released Results disappear immediately and 25 repeated runs
remain within a 1 MiB live-allocation growth bound. This small check does not
replace the N=1000, full-output or business-memory acceptance matrix.

The report truthfully retains unresolved full capacity/compiler proofs and names
the available tier `native_field_cohort_eager`. Automatic B selects one lane;
automatic graph policy selects eager. Strict byte limits, required graph capture,
explicit graph chunks, tuning, multiple GPUs/slots, RF ports, unsupported physics,
magnetic full-field/material export and trainable batch execution reject rather
than silently falling back. Complete material compilation still occurs when a
cohort is changed. Regional/background caching, vectorized source/observer work,
precise output allocations, full GPU scheduling and the batch adjoint remain open.

## Native eager execution comparison

Runtime baseline: `c8ccb48`. Command: `python -m benchmark batch --compare launch
--variants 10 --concurrency 10 --include-native --warmups 2 --repeats 5`, on physical
GPU 1 (mapped to `cuda:0` in this invocation), using the same toolkit environment
as above. Every variant ran 2000 timesteps. The benchmark coordinator retains
separate sessions during interleaved timing, then measures memory with fresh
isolated sessions. Native warm rounds reuse a PreparedBatch; independent modes
reuse their processes and extension caches. CPU spectra are ready before timing
ends, and no user retains GPU results.

| Mode | Warm median (s) | Range (s) | Isolated driver increment (MiB) |
| --- | ---: | ---: | ---: |
| sequential | 5.216941 | 4.735446–5.954700 | 317.5625 |
| processes | 0.902762 | 0.794191–0.957207 | 3175.6250 |
| run_many | 19.657694 | 17.604438–22.929485 | 359.5625 |
| native_batch | 2.342167 | 1.980125–2.541879 | 411.5625 |

Every cold, warmup, formal and isolated-memory probe comparison has zero error.
Cold makespans were 11.369580, 8.120045, 28.798055 and 8.050688 seconds respectively.
Native eager is about 2.23 times faster than sequential execution on this workload,
but about 2.59 times slower than ten independent processes. Its sampled driver
increment is about 87% lower than ten processes. This demonstrates context-memory
reduction, not acceptance of shared-background tensor savings: the native worker's
allocator peak is 48.069824 MiB, versus 46.468262 MiB for run_many, while each
independent process peaks at 4.705078 MiB (per-process peaks must not be treated as
a measurement of simultaneous summed usage).

The combined memory/throughput gate is **not passed**: native warm makespan exceeds
1.05 times the ten-process baseline. Source/observer dispatch and coordinated GPU
scheduling still require optimization. These are small-grid, single-frequency
measurements, not the unavailable original business geometry or the full baseline
matrix. Driver peaks use 20 ms sampling and may miss short transients.

Raw evidence: `benchmark/plots/batch/launch-n10-native-eager.json`.
`benchmark/RESULTS.md` retains both the historical independent baseline and this
new comparison. The CLI/report workflow has **11 passing tests**.

## Correctness audit follow-up

An adversarial audit reproduced stale source parameters after Python object-ID
reuse, premature slot-lock release when another thread closed an active iterator,
and iterator creation racing with prepared-context closure. These are corrected
with input identity retention, iterator-operation ownership, and a closed-state
check under the prepared lock. Inference tensors now bypass compilation reuse
because they have no version counter. Structure variation envelopes fail before
execution until compiled influence validation exists; full-scene compilation
without envelope declarations remains supported.

The five new cache/concurrency/envelope regressions and existing public Batch
suites pass together: **65 passed**. Additional numerical audit coverage checks
graded grids, PEC/PMC/CPML boundaries, normalized windowed multifrequency fields,
plane/flux spectra, retained full-field outputs, and independently terminating
lanes. Ordinary-solver parity is a compatibility check, not independent physical
validation or batch-adjoint acceptance.

The detailed [performance and correctness audit](fdtd-batch-performance-and-correctness-audit.md)
records both exploratory and CLI profiling runs. It separates host dispatch from
device work, measures field-only capture without claiming a public graph scheduler,
and confirms that N=10 with B=1/2/4 rebuilds ten solvers on every repeated run.
The unified benchmark entrypoint now supports `--profile` and `--capacity-audit`;
the raw evidence is retained under `benchmark/plots/batch/audit-*`.
The final combined audit/native/CPML/ordinary-adjoint/benchmark regression run
completed with **247 passed**, no failures or skips, and one existing sparse
tensor invariant warning. This extends compatibility evidence only; pending
batch-gradient and business-performance gates remain open.
