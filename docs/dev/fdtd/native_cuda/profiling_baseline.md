# Native CUDA FDTD Profiling Baseline

Baseline established 2026-06-12 on NVIDIA GeForce RTX 5080 (16 GB, driver 596.49),
torch 2.10.0+cu128, CUDA toolkit 12.9, Windows 11. All timings use the native CUDA
backend with the compiled extension:

```
WITWIN_MAXWELL_FDTD_BACKEND=cuda
WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION=1
```

## Reproducing

- Forward + adjoint timing: `python docs/dev/fdtd/native_cuda/bench_baseline.py`
  (one warm-up solve per scenario in-process, then best of 2 timed solves with
  `torch.cuda.synchronize()` around the run).
- Nsight Systems capture: build the extension once, then set
  `WITWIN_MAXWELL_FDTD_CUDA_PREBUILT=1` and run
  `nsys profile --trace=cuda --sample=none --cuda-event-trace=false python
  docs/dev/fdtd/native_cuda/profile_scenario.py <scenario>`.
  The prebuilt flag is required: Nsight's process injection breaks torch's MSVC
  detection in child processes, which rewrites `build.ninja` and triggers a rebuild
  that fails under the profiler (and leaves the build cache dirty).
  `build.py` loads the existing `.pyd` directly when the flag is set.

## Forward solve (benchmark scenarios, 128x128x128 grid, 6652 steps, CPML slab mode)

| Backend | Scenario | Best solve | ms/step | steps/s |
| --- | --- | ---: | ---: | ---: |
| native CUDA ext | `dipole_vacuum` | 4.640 s | 0.697 | 1434 |
| native CUDA ext | `planewave_vacuum` | 4.449 s | 0.669 | 1495 |
| Slang reference | `dipole_vacuum` | 4.894 s | 0.736 | 1359 |
| Slang reference | `planewave_vacuum` | 4.588 s | 0.690 | 1450 |

## Adjoint (8x8x8 design region in 40x40x40 grid, PML 8 layers, 300 steps)

| Configuration | Forward | Backward |
| --- | ---: | ---: |
| CUDA backend, default adjoint (`auto` -> python reference) | 0.198 ms/step | 8.256 ms/step |
| CUDA backend, native reverse kernels (`WITWIN_MAXWELL_FDTD_ADJOINT_BACKEND=slang`) | 0.177 ms/step | 4.725 ms/step |
| Slang backend, Slang reverse kernels | 0.217 ms/step | 5.278 ms/step |

With the CUDA forward backend, the adjoint backend resolver defaults to the pure-torch
"python reference" reverse path even though native CUDA reverse kernels exist and pass
the parity suite. Routing the reverse step through the native kernel surface
(`WITWIN_MAXWELL_FDTD_ADJOINT_BACKEND=slang`, which resolves to the NativeFDTDModule
when the main backend is `cuda`) is 1.75x faster end to end. The resolver default
lives in `fdtd/adjoint/dispatch.py` (outside `cuda/**`).

## Nsight Systems kernel summary (forward, native CUDA extension)

`planewave_vacuum` (6652 steps, wall 4.89 s under profiler, GPU kernel total ~3.1 s):

| Time % | Total | Instances | Avg | Kernel |
| ---: | ---: | ---: | ---: | --- |
| 17.0 | 531 ms | 6652 | 79.9 us | `update_electric_ez_cpml_compressed_kernel` |
| 16.2 | 505 ms | 6652 | 75.9 us | `update_magnetic_hy_cpml_compressed_kernel` |
| 15.4 | 479 ms | 6652 | 72.0 us | `update_electric_ey_cpml_compressed_kernel` |
| 13.4 | 418 ms | 6652 | 62.8 us | `update_magnetic_hx_cpml_compressed_kernel` |
| 12.8 | 399 ms | 6652 | 59.9 us | `update_electric_ex_cpml_compressed_kernel` |
| 11.4 | 354 ms | 6652 | 53.3 us | `update_magnetic_hz_cpml_compressed_kernel` |
| 8.3 | 259 ms | 133040 | 1.9 us | `accumulate_plane_observer_kernel` (20/step) |
| 2.7 | 85 ms | 13304 | 6.4 us | `add_time_shifted_source_patch_kernel` (TFSF, 2/step) |
| 2.8 | 86 ms | 53216 | 1.6 us | `accumulate_plane_observer_kernel` (other axes, 8/step) |

`dipole_vacuum` is the same picture with a point source patch (6652 launches, 7 ms)
instead of TFSF, and 166300 + 53216 plane-observer launches (33/step) at 415 ms.

## Bottleneck analysis

- The six CPML update kernels are ~86% of GPU time. Minimum traffic per kernel on a
  128^3 grid is ~50 MB (field + decay + curl + two source fields read, field written),
  giving ~620 GB/s for the 80 us kernels and ~940 GB/s for the 53 us kernels against
  ~960 GB/s peak. The fastest kernels are already bandwidth-saturated; the slowest
  (Ez/Ey/Hy, whose stencils difference along x/y rather than z) lose ~35% to
  uncoalesced neighbour access, so the practical headroom on the update kernels is
  roughly 1.2-1.5x from access-pattern and block-shape tuning, not algorithmic.
- ~1.4 s/run of wall time is not GPU work: ~226k kernel launches per run
  (40 launches/step: 6 updates + 28-33 observer accumulations + sources + DFT) each
  pay Python dispatch (`NativeFDTDModule.__getattr__` -> closure -> `_Launch`
  dataclass -> wrapper -> ~20 TORCH_CHECKs) plus cudaLaunchKernel overhead.
- The plane-observer accumulation runs one launch per (component, plane, frequency)
  pair per step; each kernel moves only ~128 KB and runs 1.6-2.0 us. The launch loop
  lives in `fdtd/observers.py` (outside `cuda/**`); within `cuda/**` the per-call
  Python overhead is the addressable part.
- Adjoint backward is 24-42x forward per step. Top costs: pure-torch reverse path by
  default (see above), per-reverse-step allocation of ~10 `zeros_like` full-field
  buffers, and segment replay of the forward update per checkpoint stride.

## Optimization backlog (prioritized)

1. Python dispatch overhead in `cuda/backend.py` (in scope, ~226k calls/run):
   cache bound kernels instead of rebuilding closure + `_Launch` per call; move
   the per-call env-var check (`_use_compiled_field_kernels`) out of the hot path.
2. Update-kernel access-pattern tuning in `electric.cu`/`magnetic.cu` (in scope):
   block-shape experiments (wider x-extent for coalescing), `__ldg`/read-only loads
   for neighbour fields, and removing redundant boundary-branch work in the interior.
   Expected 1.2-1.5x on 86% of GPU time.
3. Adjoint reverse kernels (in scope for kernels; resolver default out of scope):
   the native reverse CPML kernels under `adjoint.cu` are 1.75x faster than the
   default python-reference path; recommend flipping the `auto` resolution for the
   CUDA backend in `fdtd/adjoint/dispatch.py` (owner decision, outside `cuda/**`).
   In-scope: reduce reverse-kernel count/allocations behind the same module surface.
4. CUDA Graphs for the steady-state step (in scope via `cuda/runtime/graph.py`):
   would eliminate most of the 40 launches/step overhead; requires capture-safe
   handling of per-step scalar arguments (time value, DFT phase) as device tensors,
   and integration is currently deferred because the solve loop lives outside
   `cuda/**`.
5. Plane-observer batching across frequencies/planes (kernel side in scope,
   launch-loop side in `fdtd/observers.py`): 28-33 launches/step -> a handful.

## History

- 2026-06-12: baseline above; earlier smoke-run table (5.0/4.6 s mixed-overhead
  timings vs Slang) superseded by the warm-run protocol in `bench_baseline.py`.
