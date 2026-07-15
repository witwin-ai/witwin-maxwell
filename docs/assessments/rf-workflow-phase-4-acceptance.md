# RF Engineering Workflow Phase 4 Acceptance

Date: 2026-07-15

## Scope

Phase 4 adds the single-device antenna engineering result path and the shared
power-loss contract. The public workflow remains `Scene -> Simulation -> Result`:

- `Result.antenna(...)` converts a declared closed Huygens surface into typed
  `AntennaData`.
- `AntennaData` keeps an explicit frequency axis and provides radiation
  intensity, radiated/accepted/incident power, directivity, gain, realized gain,
  radiation/total/mismatch efficiency, Ludwig-3 co/cross polarization, axial
  ratio, coordinate-frame and phase-center metadata, normalization provenance,
  and equivalent surface currents.
- `PowerLossMonitor` and `PowerLossData` provide one typed loss contract for
  sparse volume, surface, line, and integrated channels. They preserve units,
  geometric measures, sparse Yee/global IDs, normalization, source-result, and
  autograd provenance.
- Static bulk electric conduction is computed automatically from peak-phasor
  Yee electric fields with the `0.5 * sigma * |E|^2` convention. Unsupported
  loss physics is never represented by fabricated zero channels; it must be
  supplied explicitly or fails with a precise error.

All tensors remain on one selected device. This phase does not introduce or
claim multi-device execution.

## Independent Exit-Gate Results

| Gate | Required | Result |
| --- | ---: | ---: |
| Half-wave dipole peak directivity | `< 0.25 dB` from analytic reference | `1.33e-11 dB` |
| Radiated plus loss power balance | `< 3%` relative error | Passed |
| Realized-gain identity | `< 1e-5` absolute error | `8.88e-16` |
| Circular-polarization boresight axial ratio | `< 0.5 dB` | `0 dB` |

The half-wave gate evaluates the standard thin, sinusoidally excited half-wave
dipole radiation integral against its analytic peak-directivity reference. It
validates the angular integration and directivity normalization contract; it is
not presented as a full feed-and-conductor FDTD benchmark.

The power-balance gate combines a 0.7 W isotropic radiated pattern with an
explicit 0.3 W circuit-loss channel and a 1.0 W accepted port power. This
validates that `AntennaData` and `PowerLossData` share the same peak-phasor power
normalization. It does not infer loss channels that the solver did not produce.

## Verification

Targeted Phase 4 suite:

```text
python -m pytest tests/rf/antenna tests/rf/power_loss
28 passed
```

The coverage includes CPU contract tests, CUDA device-preservation tests when
CUDA is available, live autograd preservation, closed-surface Result
integration, actual frame and phase-center transformations, equivalent Huygens
surface currents, sparse nonuniform-grid loss integration, explicit
surface/line measures, channel rejection, and the four numerical exit gates.

Ruff passes for all Phase 4 implementation and test files. The public API and
monitor/postprocessing regression suites also pass as recorded in the final
phase validation run.

## Supported Boundary

- Antenna conversion currently requires a homogeneous, isotropic, lossless
  exterior and a closed six-face frequency-domain surface monitor.
- `surface_currents` are Huygens equivalent currents on the monitor surface,
  not subcell conductor-current reconstruction.
- Antenna normalization currently represents one excitation column per result.
- Automatic loss evaluation currently covers static bulk electric conduction.
  Electric/magnetic dispersion, nonlinear/circuit, surface, and wire loss
  channels require explicit physically computed inputs.
- Magnetic conductivity and zero-thickness sheet compilation are rejected
  until their staggered-grid contracts are implemented.
- `PowerLossData` always retains its explicit frequency axis; frequency
  selection is performed explicitly on its typed tensors.
