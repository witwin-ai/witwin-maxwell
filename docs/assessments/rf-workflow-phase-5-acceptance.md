# RF Workflow Phase 5 Acceptance

Status: accepted

Date: 2026-07-15

Scope: one CUDA device. Cross-device execution, sharding, and reduction are
outside this acceptance.

## Delivered contract

- Lumped-port, terminal-port, and standalone R/C/L state is checkpointed and
  replayed with the FDTD fields.
- Finite positive series RLC uses an analytic local VJP composed with the native
  CUDA Maxwell reverse. There is no finite-difference or CPU fallback.
- Live `PortData` voltage, current, and available-power tensors participate in
  the custom autograd output contract.
- Source amplitude, series R/L/C values, material tensors, and supported smooth
  `MaterialRegion` parameters receive semantic gradients.
- Fixed single-mode `WavePort` direct and sweep results preserve the graph for
  eligible material/design inputs.
- `NetworkData` algebra and `AntennaData` metrics remain torch-native.

## Numerical acceptance

All end-to-end gradient gates use three central-difference step sizes and accept
the best stable relative error below 2%.

| Objective/input | Best relative error | Gate |
| --- | ---: | ---: |
| Lumped-port series resistance | 0.0767% | < 2% |
| Lumped-port series inductance | 0.1621% | < 2% |
| Lumped-port series capacitance | 0.6263% | < 2% |
| Material-region density with active port | 0.0148% | < 2% |
| Source amplitude including available power | 0.0000679% | < 2% |
| Standalone resistor, inductor, and capacitor | each < 2% | < 2% |
| Fixed single-mode WavePort material response | < 2% | < 2% |
| RF network and antenna postprocessing objectives | < 2% | < 2% |

The local float64 circuit VJP agrees with CUDA autograd within `2e-13`; a
three-step checkpoint replay is bitwise identical.

## Regression and performance evidence

- `tests/gradients/test_fdtd_rf_lumped_adjoint.py`: 17 passed.
- Native bridge plus lumped, terminal, and WavePort cross-regression: 145
  passed.
- RF postprocessing gradients: 6 passed.
- Fixed single-mode WavePort adjoint: 3 passed.
- No-feature fast-path structure check: 1 passed.
- Ruff and `git diff --check`: passed.

The no-feature performance comparison uses a 24-cubed CPML scene, 2,000 time
steps, CUDA Events, three warmups and seven timed runs per block, and symmetric
ABBA/BAAB ordering. Baseline `88dafa0` and clean candidate `d00830e` produced a
paired-round regression of 1.6305%, satisfying the `< 2%` gate. The complete
machine-readable record is `rf-workflow-phase-5-performance.json`.

## Explicit unsupported combinations

- `ParallelRLC` in a differentiable circuit run;
- observer-only lumped contours and open internal resistance;
- trainable source impedance or port reference impedance;
- differentiable lumped `PortSweep`;
- conductive, dispersive, nonlinear, modulated, full-anisotropy, Bloch, or
  other complex-field circuit coupling;
- trainable WavePort amplitude, multiple differentiable modes, a design that
  changes the aperture or adjacent launch plane, or mixed WavePort/lumped
  differentiation.

These combinations fail with a capability-specific error before an unsupported
gradient can be returned.
