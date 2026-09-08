# WiTwin Maxwell 0.5.0

Previous release: 0.4.0. Release date: 2026-09-08.

## Batch simulation

`Simulation.batch(...)` adds an experimental execution tier for variants of a common FDTD scene. The public architecture remains `Scene -> Simulation -> Result`. A prepared batch reuses a bounded number of resident field slots; variants retain independent dynamic fields even when fixed geometry, material coefficients, or source declarations can share storage.

- Native kernels batch field updates, CPML histories, source injection, and spectral observers across resident lanes. Persistent electric spectral buffers and complete timestep CUDA Graphs are reusable between compatible runs.
- Proven local density/structure changes reuse fixed material compilation outside the declared influence region.
- Automatic capacity, private CUDA allocation budgets, and one or two slots per device bound resident execution state. Multi-GPU scheduling supports ordered or completion-order result consumption.
- Opt-in throughput tuning measures complete streamed forward runs. Candidate module state and RNG are isolated, and reports expose timing samples, allocation peaks, rejected capacities, and tuning cost.
- Results own their scene/output snapshots and remain valid after slot reuse or close. Supported density, dielectric geometry, and named source-scale inputs use PyTorch-native checkpoint adjoints, including delayed backward.

For API details and phase acceptance criteria, see the [Batch development plan](../plans/fdtd-batch-simulation-api-and-gpu-scheduling.md). The [feature inventory](../../FEATURE_LIST.md) is the authoritative capability list.

## FDTD correctness repairs

This release also includes material-aware stable time-step selection, unified source-time planning, corrected complex CPML/Mur histories, complete capture/resume state, and validation of material bounds and resume configuration. These repairs can change results for previously incorrect combinations; recompute affected cached simulations.

## Validation evidence

The recent audit recorded the following targeted test groups on the release source lineage; groups overlap and their counts must not be summed as unique tests:

- Persistent spectra, nonuniform numerics, native spectral kernels, and ordinary adjoints: 67 passed.
- Tuning isolation, result snapshots, and Batch adjoints on two GPUs: 35 passed.
- Parallel scheduling, strict capacity budgets, and spectral reuse: 24 passed.
- Benchmark command, process sampling, report preservation, and benchmark system: 119 passed, 3 skipped.

Release preparation additionally passed 89 public API, simulation smoke, Batch API, and prebuilt ABI checks with CUDA enabled. Local source and wheel builds passed, with Batch modules and native kernel sources verified in the archives. Release platform binaries are built and load-tested by the publishing workflow before upload. These checks are targeted regression evidence, not a new claim that the entire historical test suite passes. Existing adjoint stream-synchronization warnings remain visible.

A two-device bandwidth measurement produced zero maximum absolute probe error against independent simulations. Another compute process was observed during timing, so its 0.942x measured ratio does not establish exclusive-device scaling. The 1.7x performance acceptance criterion remains open.

See the [recent Batch audit](../assessments/fdtd-batch-recent-changes-audit.md) and [FDTD repair validation](../assessments/fdtd-repair-validation-2026-09-08.md) for detailed evidence.

## Current limits

Batch supports a subset of ordinary FDTD: real static-material scenes, supported point sources and point/plane/flux observations, selected electric full-field spectra, and supported boundary/material adjoint combinations. RF ports, advanced material dynamics, magnetic full-field spectra, and material export are not implemented in Batch and reject explicitly.

Reverse execution still uses native per-lane adjoints; fused batch reverse kernels are pending. A device allocation budget accounts for private CUDA backing segments, including retained results/tapes, but excludes driver/library allocations and producer devices outside the execution policy. Keeping all N results can exhaust a budget during execution; exact all-output preplanning is pending.

Tuning measures streamed forward throughput and tape construction, not backward or retention of all results. Its cost can exceed savings for short workloads. Module copying cannot undo arbitrary external side effects; comparable tuning requires stable scene declarations.
