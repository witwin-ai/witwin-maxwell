# RF Workflow Phase 1 Acceptance

Date: 2026-07-15

Scope: single-device FDTD only

Maturity: E1 experimental

## Delivered contracts

- `LumpedPort` supports an optional `SeriesRLC` or `ParallelRLC` termination.
- `Resistor`, `Capacitor`, and `Inductor` can be attached to a `Scene` through
  `add_lumped_element(...)` and compiled with `compile_lumped_elements(...)`.
- `Simulation.fdtd(..., excitations=PortExcitation(...))` supports one active
  lumped port in a run.
- FDTD applies an energy-consistent implicit-midpoint field/circuit correction.
  R, L, C, series RLC, and parallel RLC auxiliary state remains on the target
  device throughout stepping.
- Port voltage and current are accumulated at their physical Yee sample times:
  E at the integer step and the implicit branch/H current at the half step.
  Loaded ports use the exact midpoint branch current entering the field network;
  unloaded observers retain the compiled right-hand H contour.
- Explicit voltage paths that bridge two distinct PEC terminal surfaces open
  only their declared feed-gap Yee edges. Paths embedded in or shorting the same
  PEC conductor remain rejected during preparation.
- `Result.port(name)` returns live `PortData`. Standalone `PortData` and
  `NetworkData` persistence is schema-versioned; result snapshots embed detached
  CPU port payloads.

## Independent acceptance evidence

| Gate | Evidence | Result |
| --- | --- | --- |
| 50 ohm matched load | Kurokawa decomposition of the discrete midpoint V/I state | return loss greater than 30 dB |
| Open and short | Discrete field/circuit limiting cases | reflection coefficients +1 and -1 within `1e-12` |
| Series and parallel RLC resonance | Bilinear-transform sweep against analytic resonance | relative frequency error below 2% |
| Passivity and energy | Per-step field, source, loss, and stored-energy identity | equality at float64 tolerance |
| Direction convention | Reversed sparse voltage/current orientation | identical field update and energy |
| Device parity | CPU/CUDA reference runtime comparison | float64 parity tests pass |
| End-to-end FDTD | CUDA `Scene -> Simulation -> Result.port` run | finite nonzero device-resident V/I and available power |
| PEC terminal gap | Distinct-conductor and embedded-edge CUDA preparation cases | declared gap opens; embedded/same-conductor path rejects |
| Persistence | Detached CPU round trips with type and schema checks | port/network/result payload tests pass |

The targeted Phase 1 suite includes a CUDA profiler gate over the port hot loop:
it observes no scalar synchronization and no host-to-device or device-to-host
copy. Setup and final result construction may perform control-plane validation
or explicit persistence transfers. Port-only Gaussian/Ricker runs resolve to a
full pulse window beginning at step zero, and early field shutoff restores the
planned port DFT normalizer.

## Deliberate boundaries

- Automatic multi-port sweeps and complete N-port assembly belong to Phase 2.
- The FDTD port/RLC adjoint replay belongs to the final single-device closure
  phase. Until then, trainable scenes combined with a lumped port or R/L/C
  object fail explicitly before entering the adjoint bridge.
- Arbitrary callable source waveforms are not evaluated during device stepping;
  port excitation currently accepts device-native CW, Gaussian-pulse, and
  Ricker forms.
- Time-domain source impedance is real and positive. Complex reference
  impedance remains supported by the frequency-domain power-wave result model.
- Conductive, electrically dispersive, nonlinear, time-modulated, complex-field,
  and full off-diagonal anisotropic coupling at a lumped update remains guarded
  until its joint circuit equation is implemented.
