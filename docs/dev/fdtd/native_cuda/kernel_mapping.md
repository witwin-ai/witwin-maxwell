# Native CUDA FDTD Kernel Mapping

The current migration stage routes `WITWIN_MAXWELL_FDTD_BACKEND=cuda` to an internal native-CUDA module surface without calling `slangtorch.loadModule()`. The C++/CUDA source tree exists under `witwin/maxwell/fdtd/cuda/`; covered kernels build and run when `WITWIN_RUN_CUDA_EXTENSION_BUILD=1` and `WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION=1` are set. Slang remains the default hidden reference backend until packaging/default-switch risk is resolved.

| Slang kernel | CUDA target function | Python caller | Parity test | Status |
| --- | --- | --- | --- | --- |
| common linear/unflatten helpers | `kernels/common.cu` | `cuda/backend.py` | `tests/fdtd/cuda/test_cuda_common_parity.py` | native `.cu` helper covered |
| `updateMagneticFieldHxStandard3D` | `kernels/magnetic.cu` | `runtime/stepping.py` | `tests/fdtd/cuda/test_cuda_magnetic_parity.py` | native `.cu` covered |
| `updateMagneticFieldHyStandard3D` | `kernels/magnetic.cu` | `runtime/stepping.py` | `tests/fdtd/cuda/test_cuda_magnetic_parity.py` | native `.cu` covered |
| `updateMagneticFieldHzStandard3D` | `kernels/magnetic.cu` | `runtime/stepping.py` | `tests/fdtd/cuda/test_cuda_magnetic_parity.py` | native `.cu` covered |
| `updateMagneticFieldHx3D` / `Hy3D` / `Hz3D` | `kernels/magnetic.cu` | `runtime/stepping.py` | `tests/fdtd/cuda/test_cuda_cpml_extension_parity.py`, `tests/boundaries/cpml/test_fdtd_cpml.py` | native `.cu` covered |
| `updateMagneticField*CpmlCompressed3D` | `kernels/magnetic.cu` | `runtime/stepping.py` | `tests/fdtd/cuda/test_cuda_cpml_extension_parity.py`, `tests/boundaries/cpml/test_fdtd_cpml.py` | native `.cu` covered |
| `updateElectricField*Standard3D` | `kernels/electric.cu` | `runtime/stepping.py` | `tests/fdtd/cuda/test_cuda_magnetic_parity.py`, `tests/fdtd/cuda/test_cuda_electric_boundary_parity.py` | native `.cu` covered |
| `updateElectricField*Cpml3D` | `kernels/electric.cu` | `runtime/stepping.py` | `tests/fdtd/cuda/test_cuda_cpml_extension_parity.py`, `tests/boundaries/cpml/test_fdtd_cpml.py` | native `.cu` covered |
| `updateElectricField*CpmlCompressed3D` | `kernels/electric.cu` | `runtime/stepping.py` | `tests/fdtd/cuda/test_cuda_cpml_extension_parity.py`, `tests/boundaries/cpml/test_fdtd_cpml.py` | native `.cu` covered |
| `updateElectricField*Bloch3D` | `kernels/electric.cu` | `runtime/stepping.py` | `tests/fdtd/cuda/test_cuda_electric_boundary_parity.py` | native `.cu` covered |
| `clampPecBoundary3D` | `kernels/boundary.cu` | boundary projection tests | `tests/fdtd/cuda/test_cuda_electric_boundary_parity.py` | native `.cu` covered |
| `projectPeriodicBoundary3D` | `kernels/projection.cu` | boundary projection tests | `tests/fdtd/cuda/test_cuda_electric_boundary_parity.py` | native `.cu` covered |
| `projectBlochBoundary3D` | `kernels/projection.cu` | boundary projection tests | `tests/fdtd/cuda/test_cuda_electric_boundary_parity.py` | native `.cu` covered |
| `addSourcePatch*` | `kernels/sources.cu` | `excitation/temporal.py` | `tests/fdtd/cuda/test_cuda_sources_parity.py`, `tests/sources/point/test_fdtd_sources.py` | native `.cu` covered |
| `addBatchedReferenceSourcePatches3D` | `kernels/sources.cu` | `excitation/tfsf_apply.py` | `tests/fdtd/cuda/test_cuda_sources_parity.py`, `tests/sources/tfsf/test_fdtd_tfsf.py` | native `.cu` covered |
| `addBatchedInterpolatedSourcePatches3D` | `kernels/sources.cu` | `excitation/tfsf_apply.py` | `tests/fdtd/cuda/test_cuda_sources_parity.py`, `tests/sources/tfsf/test_fdtd_tfsf.py` | native `.cu` covered |
| `accumulateRunningDftYee3DBatched` | `kernels/spectral.cu` | `runtime/spectral.py` | `tests/fdtd/cuda/test_cuda_spectral_observer_parity.py`, `tests/sources/point/test_fdtd_sources.py` | native `.cu` covered |
| `accumulatePointObservers3D` | `kernels/observers.cu` | `observers.py` | `tests/fdtd/cuda/test_cuda_spectral_observer_parity.py`, `tests/monitors/observers/test_fdtd_observers.py` | native `.cu` covered |
| `accumulatePlaneObserver3D` | `kernels/observers.cu` | `observers.py` | `tests/fdtd/cuda/test_cuda_spectral_observer_parity.py`, `tests/monitors/observers/test_fdtd_observers.py` | native `.cu` covered |
| `updateDebyeCurrent3D` / `updateDrudeCurrent3D` / `updateLorentzCurrent3D` | `kernels/dispersive.cu` | `runtime/materials.py` | `tests/fdtd/cuda/test_cuda_dispersive_parity.py`, `tests/materials/dispersive/test_fdtd_dispersive.py` | native `.cu` covered |
| `applyPolarizationCurrent3D` | `kernels/dispersive.cu` | `runtime/materials.py` | `tests/fdtd/cuda/test_cuda_dispersive_parity.py`, `tests/materials/dispersive/test_fdtd_dispersive.py` | native `.cu` covered |
| `updateKerrElectricField*Curl3D` | `kernels/dispersive.cu` | `runtime/materials.py` | `tests/fdtd/cuda/test_cuda_dispersive_parity.py` | native `.cu` covered |
| `updateAuxiliaryMagnetic1D` / `updateAuxiliaryElectric1D` | `kernels/sources.cu` | `excitation/spatial.py` | `tests/fdtd/cuda/test_cuda_sources_parity.py`, `tests/sources/incident/test_fdtd_incident.py` | native `.cu` covered |
| `reverseElectricAdjointToH*Standard3D` | `kernels/adjoint.cu` | `adjoint/core.py` native module surface | `tests/fdtd/cuda/test_cuda_adjoint_parity.py` | native `.cu` covered |
| `reverseMagneticAdjointToE*Standard3D` | `kernels/adjoint.cu` | `adjoint/core.py` native module surface | `tests/fdtd/cuda/test_cuda_adjoint_parity.py` | native `.cu` covered |
| `reverseMagneticAdjointToH*Standard3D` | `kernels/adjoint.cu` | `adjoint/core.py` native module surface | `tests/fdtd/cuda/test_cuda_adjoint_parity.py` | native `.cu` pointwise decay covered |
| `reverseElectricComponent*Cpml3D` | `kernels/adjoint.cu` | `adjoint/core.py` native module surface | `tests/fdtd/cuda/test_cuda_adjoint_parity.py` | native `.cu` covered |
| `reverseMagneticComponent*Cpml3D` | `kernels/adjoint.cu` | `adjoint/core.py` native module surface | `tests/fdtd/cuda/test_cuda_adjoint_parity.py` | native `.cu` covered |
| `reverseElectricAdjointToH*Bloch3D` | `kernels/adjoint.cu` | `adjoint/core.py` native module surface | `tests/fdtd/cuda/test_cuda_adjoint_parity.py` | native `.cu` covered; no-Slang-JIT reverse-step covered |
| `reverseMagneticAdjointToE*Bloch3D` | `kernels/adjoint.cu` | `adjoint/core.py` native module surface | `tests/fdtd/cuda/test_cuda_adjoint_parity.py` | native `.cu` covered; no-Slang-JIT reverse-step covered |
| `reverseDebyeCurrent3D` / `reverseDrudeCurrent3D` / `reverseLorentzCurrent3D` | `kernels/adjoint.cu` | `adjoint/reverse_common.py` native module surface | `tests/fdtd/cuda/test_cuda_adjoint_parity.py` | native `.cu` covered |
| `accumulateTfsf*SampleAdjoint3D` | `kernels/adjoint.cu` | `adjoint/core.py` TFSF reverse helpers | `tests/fdtd/cuda/test_cuda_adjoint_parity.py` | native `.cu` covered |
| `reverseTfsfAuxiliary*1D` | `kernels/adjoint.cu` | `adjoint/core.py` TFSF reverse helpers | `tests/fdtd/cuda/test_cuda_adjoint_parity.py` | native `.cu` covered |
| Full adjoint workflow dispatch | native module plus Slang-free Python reference | `adjoint/dispatch.py` | `tests/fdtd/cuda/test_cuda_adjoint_parity.py`, `tests/gradients/test_fdtd_adjoint_bridge.py`, `tests/gradients/test_fdtd_adjoint_rigorous.py`, `tests/gradients/test_fdtd_mode_source_adjoint.py` | forced CUDA backend gradient matrix passing |
| Full public solve parity | native module vs Slang reference | `Simulation.fdtd(...).run()` | `tests/fdtd/cuda/test_cuda_solver_parity.py` | gated Slang oracle passing for standard and CPML scenes |
| CUDA Graph capture helper | `cuda/runtime/graph.py` | opt-in runtime utility | `tests/fdtd/cuda/test_cuda_graph.py` | infrastructure covered; solver-loop integration deferred |
