# Native CUDA FDTD Profiling Baseline

Representative benchmark timing and Nsight Systems CUDA kernel summaries have been collected after native forward and adjoint kernel parity reached the current migration target.

Known build baseline:

- Windows developer build requires a Visual Studio CUDA environment such as `vcvars64.bat`.
- `WITWIN_RUN_CUDA_EXTENSION_BUILD=1` enables the explicit extension build tests.
- Wheel build was verified with `python -m build --wheel --no-isolation`; the resulting wheel contained 14 native CUDA source/header artifacts and 13 Slang source artifacts before the local `dist/` output was removed.
- The compiled extension currently covers a no-op launch, row-major indexing helpers, standard/Bloch E and standard H update kernels, dense/compressed CPML kernels, boundary/projection kernels, source/TFSF kernels, DFT kernels, observer kernels, dispersive/Kerr kernels, and standard/CPML/Bloch/TFSF/dispersive adjoint reverse kernels.
- Recent native CUDA verification used forced-backend full `tests`, solver regression, gradient regression, `tests/fdtd/cuda` extension parity, gated Slang-vs-CUDA full-solve parity, wheel build validation, two benchmark scenarios, and Nsight Systems captures for the representative benchmark scenes.
- `CudaGraphRunner` can capture/replay static in-place CUDA work. Main solver-loop graph integration remains deferred because in-place timestep capture must avoid changing the counted simulation step semantics.

Representative benchmark smoke runs with `WITWIN_MAXWELL_FDTD_BACKEND=cuda` and `WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION=1`:

| Scenario | Field L2 | Field Corr | Flux err | Maxwell time | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `dipole_vacuum` | 5.4598e-02 | 1.0000 | 4.4856e-01 | 28.40 s | Tidy3D cache hit; command included first-process extension/build overhead |
| `planewave_vacuum` | 1.2307e+00 | 0.9931 | 7.5295e-01 | 5.17 s | Tidy3D cache hit; extension already built in process environment |

Warm-run timing used `_run_maxwell(...)` twice in the same process after preloading the native extension for CUDA. The Slang run used the same scenes and runner with `WITWIN_MAXWELL_FDTD_BACKEND=slang`.

| Backend | Scenario | First run | Warm run |
| --- | --- | ---: | ---: |
| native CUDA extension | `dipole_vacuum` | 5.035 s | 4.822 s |
| Slang | `dipole_vacuum` | 5.460 s | 4.870 s |
| native CUDA extension | `planewave_vacuum` | 4.650 s | 4.045 s |
| Slang | `planewave_vacuum` | 4.370 s | 4.409 s |

In these smoke timings, native CUDA warm-run time is no worse than Slang for both representative scenarios.

Nsight Systems captures were generated with `--trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-event-trace=false` using `docs/dev/fdtd/native_cuda/profile_scenario.py`, which runs the Maxwell side of a benchmark scenario without updating `benchmark/RESULTS.md`.

| Backend | Scenario | Report | Dominant kernel families |
| --- | --- | --- | --- |
| native CUDA extension | `planewave_vacuum` | `profiles/nsys_cuda_planewave_vacuum.nsys-rep` | compressed CPML E/H kernels, plane observer, TFSF source patch |
| native CUDA extension | `dipole_vacuum` | `profiles/nsys_cuda_dipole_vacuum.nsys-rep` | compressed CPML E/H kernels, plane observer, point source patch |
| Slang reference | `planewave_vacuum` | `profiles/nsys_slang_planewave_vacuum.nsys-rep` | compressed CPML E/H kernels, plane observer, TFSF source patch |
| Slang reference | `dipole_vacuum` | `profiles/nsys_slang_dipole_vacuum.nsys-rep` | compressed CPML E/H kernels, plane observer, point source patch |

Top-kernel summary from `nsys stats --report cuda_gpu_kern_sum`:

| Backend | Scenario | Top update kernel total | Instances per update kernel | Observer/source notes |
| --- | --- | ---: | ---: | --- |
| native CUDA extension | `planewave_vacuum` | `updateElectricFieldEzCpmlCompressed3D`, 446.2 ms | 6652 | `accumulatePlaneObserver3D`: 186256 launches, 358.5 ms; TFSF time-shifted source patches: 13304 launches, 35.4 ms |
| Slang reference | `planewave_vacuum` | `updateElectricFieldEyCpmlCompressed3D`, 450.3 ms | 6652 | `accumulatePlaneObserver3D`: 186256 launches, 357.3 ms; TFSF time-shifted source patches: 13304 launches, 35.9 ms |
| native CUDA extension | `dipole_vacuum` | `updateElectricFieldEzCpmlCompressed3D`, 445.4 ms | 6652 | `accumulatePlaneObserver3D`: 219516 launches, 420.4 ms; point source patch: 6652 launches, 7.5 ms |
| Slang reference | `dipole_vacuum` | `updateElectricFieldEyCpmlCompressed3D`, 449.6 ms | 6652 | `accumulatePlaneObserver3D`: 219516 launches, 420.3 ms; point source patch: 6652 launches, 7.4 ms |
