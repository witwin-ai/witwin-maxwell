# FDTD Native CUDA Gradient Capability Matrix

The compiled CUDA reverse operators are the only production FDTD adjoint
implementation. Python owns checkpointing, replay, launch orchestration, and
PyTorch autograd integration; it does not provide an alternative reverse
numerical implementation.

Differentiable FDTD requires a CUDA scene and the packaged native extension.
Preparation fails before forward stepping if the device, extension, or scene
capability is unsupported. There is no backend setting and no automatic
fallback.

## Native variants

| Differentiable scene class | Native label | Principal reverse kernels |
| --- | --- | --- |
| Standard real fields | `native_standard` | standard Yee electric/magnetic transpose kernels |
| Real CPML | `native_cpml` | CPML electric/magnetic and derivative-fold kernels |
| Electric conductivity + CPML | `native_conductive` | semi-implicit conductive coefficient pullback + CPML |
| General instantaneous nonlinearity (standard or CPML) | `native_general_nonlinear` | dynamic decay/curl pullback for `chi2`, `chi3`, and TPA + collocation transpose |
| Full electric anisotropy + CPML | `native_full_aniso` | off-diagonal curl pullback + CPML core |
| Bloch complex fields | `native_bloch` | split real/imag Bloch phase transpose kernels |
| Electric ADE dispersion | `native_dispersive` | Debye/Drude/Lorentz recurrence and dispersive correction kernels |
| Bloch + ADE dispersion | `native_bloch_dispersive` | complex Bloch core + real/imag electric and magnetic ADE pullbacks |
| One PML axis + two Bloch axes | `native_mixed_bloch_cpml` | complex Bloch phase transpose + native CPML correction pullback |
| TFSF | `native_tfsf` | native base reverse + auxiliary-line and sample-gather pullbacks |
| Grating-slab TFSF | `native_grating_tfsf` | mixed Bloch/CPML reverse + any-axis grating TFSF auxiliary pullback |

Electric and magnetic Debye, Drude, and Lorentz poles may be composed where the
forward solver supports them. Magnetic ADE includes real and imaginary state for
Bloch scenes. The mixed complex-CPML specialization supports each permutation of
one paired-PML axis and two paired-Bloch axes. Additive magnetic conductivity is
part of the standard/CPML magnetic coefficient policy and needs no separate
stateful variant.

The machine-readable inventory is
`witwin/maxwell/fdtd/adjoint/capabilities.py::NATIVE_ADJOINT_CAPABILITIES`.

## Single-device RF composition

The RF adjoint does not introduce a second Maxwell reverse implementation. It
composes the native standard/CPML reverse with an analytic local VJP for the
implicit lumped circuit update and replays the circuit state from the same
checkpoints as the Yee fields.

| RF workflow | Differentiable inputs and outputs | Explicit boundary |
| --- | --- | --- |
| Active `LumpedPort` / `TerminalPort` | Source amplitude, material/design tensors, port V/I and accepted/incident/reflected/available power | Real, nondispersive, nonconductive standard/CPML field coupling |
| Finite positive `SeriesRLC` or standalone R/C/L | R/L/C values, material/design tensors, and the same port objectives | No `ParallelRLC`, open internal resistance, observer-only contour, or trainable source/reference impedance |
| Fixed single-mode `WavePort` direct run or `PortSweep` | Eligible material/design tensors through `PortData` or `NetworkData` | Fixed amplitude and modal aperture/launch plane; no multimode or mixed lumped workflow |
| Bound `Circuit` graph (linear MNA co-simulation) | R/L/C and independent-source waveform tensors (direct, restricted-SPICE-expression, or `SceneModule`-derived), bound-port material/geometry inputs, port outputs, and every live `CircuitData` node-voltage, branch-current, device-power, energy-balance, port-power, field-energy, and tensor-diagnostic output | Requires an exactly zero coupled DC solution and companion history; a tensor seed at `CircuitData` sample 0, trainable DC source values, trainable circuit initial conditions, trainable port reference impedance, and distributed execution all fail explicitly. Differentiable circuit runs bypass the RC and CUDA-Graph forward fast paths |
| `NetworkData` and `AntennaData` postprocessing | Torch-native S/Z/Y, renormalization, reference-plane, gain, realized-gain, and efficiency objectives | Detached persistence/export is not differentiable |

The circuit adjoint follows the same composition rule: it adds no second Maxwell
reverse implementation. Checkpoint replay rebuilds each strongly coupled MNA step
and PyTorch's linear-solve VJP supplies the transpose solve, seeded against the
field adjoint at the bound ports.

Differentiable lumped `PortSweep`, conductive/dispersive/nonlinear/modulated or
full-anisotropy circuit coupling, and Bloch/complex-field circuit coupling fail
at preparation with a capability-specific error. No finite-difference or CPU
fallback is selected.

## Correctness and acceptance

The production tree contains no analytic Torch reverse or autograd-VJP fallback.
Correctness is checked independently through:

- end-to-end central finite differences in `tests/gradients` for material,
  geometry, boundary, dispersive, nonlinear, and source gradients;
- local discrete transpose tests for the six mixed CPML axis/tangent
  permutations;
- central-difference checks of the native `chi2`, `chi3`, and TPA coefficient
  pullbacks;
- existing Yee, CPML, ADE, Bloch, source, checkpoint-replay, dtype, device, and
  contiguity tests;
- preparation-time failure tests when a native runner is unavailable.
- three-step central finite differences for lumped R/L/C, source amplitude,
  material-region density, fixed single-mode WavePort material response, and RF
  postprocessing objectives;
- central finite differences for bound-`Circuit` R/L/C and source-amplitude
  parameters (RC cutoff, RLC near resonance, two-port insertion loss, bound-port
  material), plus explicit rejection tests for each unsupported circuit-adjoint
  configuration, in `tests/gradients/test_fdtd_circuit_adjoint.py`.

## Profiling record (2026-07-12)

Validation host: NVIDIA GeForce RTX 5080 (16 GiB, compute capability 12.0),
driver 596.49, PyTorch 2.10.0 with CUDA 12.8, and NVCC 12.9.41. Fields and
reverse state used FP32. Timings synchronize immediately before and after the
complete checkpointed backward. Each representative scene used 24 time steps;
peak allocation was reset after forward preparation.

| Scene | Native runner | Backward ms | ms/step | Peak backward MiB |
| --- | --- | ---: | ---: | ---: |
| CPML | `native_cpml` | 580.262 | 24.178 | 0.254 |
| Bloch | `native_bloch` | 188.197 | 7.842 | 0.699 |
| Electric ADE | `native_dispersive` | 155.089 | 6.462 | 0.310 |
| General nonlinear | `native_general_nonlinear` | 525.064 | 21.878 | 17.664 |
| Mixed Bloch + CPML | `native_mixed_bloch_cpml` | 286.098 | 11.921 | 1.802 |
| Grating TFSF | `native_grating_tfsf` | 301.663 | 12.569 | 1.069 |

These are full-backward results rather than isolated-kernel timings. Every run
recorded exactly 24 invocations of the named native runner. PyTorch profiler
inspection of an isolated mixed reverse step found no CUDA memcpy event; the new
per-cell reverse operators accept device tensors and launch compiled kernels
without a host round trip.

Nsight Compute 2025.2 connected to the focused nonlinear reverse workload, but
the driver denied hardware counters with `ERR_NVGPUCTRPERM`; occupancy,
registers/thread, spills, cache, throughput, and branch-efficiency counters need
administrator-enabled counter access. Nsight Systems is not installed on this
host (`nsys` is absent). These environment limits are recorded explicitly rather
than treating an unavailable profiler as a successful measurement.

Acceptance results on this GPU: 9 native local tests, 30 material/capability
tests, 54 bridge tests, and 19 rigorous-gradient tests passed. The combined
`tests/fdtd tests/gradients` run completed with 170 passed and 44 optional forward
tests skipped; the CUDA adjoint acceptance cases executed.
