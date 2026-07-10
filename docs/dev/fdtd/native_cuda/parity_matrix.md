# Native CUDA FDTD Parity Matrix

Native CUDA is the only FDTD backend. Set `WITWIN_RUN_CUDA_EXTENSION_BUILD=1` from a Visual Studio CUDA build environment to compile and run the explicit native C++/CUDA extension tests. Slang comparisons below are retained as historical migration evidence and are not runnable backend options.

| Area | Coverage | Status |
| --- | --- | --- |
| Backend selector | `tests/fdtd/cuda/test_cuda_extension_load.py` | passing |
| No Slang JIT for CUDA backend | `tests/fdtd/cuda/test_cuda_no_slang_jit.py` | passing |
| Common indexing helpers | `tests/fdtd/cuda/test_cuda_common_parity.py` | passing; compiled `.cu` helper gated by `WITWIN_RUN_CUDA_EXTENSION_BUILD=1` |
| Standard E/H step | native `.cu` extension vs Python reference, optional Slang oracle | passing; Slang oracle gated |
| CPML dense | native `.cu` extension one-step parity plus existing CPML tests with CUDA backend forced | passing |
| CPML compressed/slab | native `.cu` extension slab-vs-dense one-step parity | passing |
| Standard electric boundary modes | none, PEC, PMC, periodic extension parity | passing |
| Bloch electric update and boundary projection | Bloch forward E update, periodic/Bloch projection, PEC clamp tests | passing; native `.cu` Bloch E/projection/clamp kernels covered |
| Point sources | existing point source tests with CUDA backend forced plus native `.cu` source patch parity | passing |
| Full-field DFT | native `.cu` extension DFT parity plus multi-frequency DFT tests with CUDA backend forced | passing |
| Observers | native `.cu` extension point/plane observer parity plus observer/full-field DFT tests with CUDA backend forced | passing |
| Dispersive media | existing dispersive tests with CUDA backend forced plus native `.cu` Debye/Drude/Lorentz/Kerr parity | passing |
| TFSF auxiliary/source patches | native `.cu` source, TFSF, and auxiliary forward parity plus incident/TFSF solver tests | passing for covered source/TFSF cases |
| Mode source | existing mode source tests with CUDA backend forced | passing through grouped source and gradient runs |
| Full solve parity | historical Slang-vs-CUDA public solve parity for standard and CPML point-dipole scenes | passed before removal of the reference backend |
| Adjoint | CUDA backend avoids Slang reverse dispatch by default; explicit native-module reverse path covers standard, CPML, Bloch, TFSF auxiliary/source-adjoint, and dispersive reverse kernels; existing gradient tests pass with CUDA backend forced | passing for covered native kernels and full Python-reference workflow |
| Compiled `.cu` extension | C++/CUDA extension builds and runs no-op/index helpers, standard E/H kernels, dense/compressed CPML, boundary/projection, source/TFSF, DFT, observer, dispersive/Kerr, and adjoint reverse kernels | passing for covered kernels |
| CUDA Graphs | `CudaGraphRunner` captures/replays static in-place CUDA work | infrastructure passing; solver-loop graph integration remains opt-in design work |
| Performance smoke | `dipole_vacuum` and `planewave_vacuum` warm-run timing against Slang | CUDA warm-run no slower in measured scenes |

Known validation limitations:

- The former Slang oracle passed the public FDTD baseline before native migration; its runtime and JIT files have since been removed.
- Full gradient workflows under `WITWIN_MAXWELL_FDTD_BACKEND=cuda` still default to the Slang-free Python reference reverse path for conservative correctness. Explicit `WITWIN_MAXWELL_FDTD_ADJOINT_BACKEND=slang` with CUDA backend now resolves the same Slang-style kernel names to the native module and is covered for Bloch no-Slang-JIT reverse-step parity.
- Nsight Systems captures now exist for native CUDA and Slang on `dipole_vacuum` and `planewave_vacuum`. Native CUDA is the default runtime; Slang profiles are retained only for explicit comparison runs.
