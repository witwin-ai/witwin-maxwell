# RF Port And Network Conventions

Status: Phase 0 contract
Scope: single-device FDTD RF ports and torch-native network postprocessing

## Field And Phasor Convention

RF port data uses peak complex phasors with physical time dependence

`x(t) = Re(X exp(-i omega t))`.

The matching running DFT is

`X = 2 / sum(w_n) * sum(w_n x_n exp(+i omega t_n))`.

The sample time is part of the accumulator contract. Electric voltage samples
and magnetic current samples are evaluated at their actual Yee leapfrog times;
the magnetic half-step offset is included while accumulating the DFT. It must not
be repaired later with an empirical phase rotation.

All RF result objects retain an explicit frequency dimension, including a
single-frequency result.

## Port Orientation

Voltage is the signed line integral from the negative terminal to the positive
terminal:

`V = integral_negative^positive E . dl`.

Current follows the right-hand-rule circulation around the positive conductor.
The positive current direction and the voltage path therefore describe one
oriented port, rather than two independently inferred signs.

Reversing a port reverses both voltage and current. Complex wave amplitudes gain
the same common minus sign, while impedance, incident/reflected/accepted power,
return loss, VSWR, and S-parameters remain physically invariant.

## Peak-Phasor Power And Power Waves

The average accepted power is

`P = 0.5 Re(V I*)`.

For a reference impedance `Z0` with `R0 = Re(Z0) > 0`, the peak-phasor Kurokawa
power waves are

`a = (V + Z0 I) / (2 sqrt(2 R0))`,

`b = (V - Z0* I) / (2 sqrt(2 R0))`.

Consequently,

`P_incident = |a|^2`,

`P_reflected = |b|^2`,

`P_accepted = |a|^2 - |b|^2 = 0.5 Re(V I*)`.

`available_power` is defined only when an excitation carries an explicit
generator impedance model. A port reference impedance alone does not define it.

## Tensor Shapes And Ordering

`PortData` uses an explicit trailing frequency axis for `voltage`, `current`,
`a`, and `b`. Its `frequencies` tensor has shape `[F]`.

`NetworkData` uses:

- `frequencies`: real tensor `[F]`
- `s`: complex tensor `[F, N, N]`
- `z0`: complex tensor `[F, N]`
- `valid_columns`: boolean tensor `[N]`
- `port_names`: unique names in the same order as both matrix port axes

The matrix order is `[frequency, output_port, input_port]`, so `b = S a`.
An excitation column that was not run is invalid, not zero. Full-matrix
operations and file export must reject incomplete columns.

## Differentiation And Persistence

Wave transforms and network algebra use PyTorch operations and retain the active
autograd graph. File I/O is an explicit detached-inference boundary: tensors are
stored on CPU and a live autograd graph is not serialized.

Discrete topology is not differentiable. Port path membership, current-loop
membership, port count, terminal ordering, and mode identity are preparation-time
structure. Supported electrical values such as reference impedance and later RLC
parameters may remain tensors.

## Phase 0 Golden Data

The Phase 0 analytic fixtures are intentionally solver-independent:

- constant electric field integrated over an axis-aligned Yee path;
- constant magnetic circulation integrated around an axis-aligned rectangular
  current surface;
- known complex `V`, `I`, and `Z0` values round-tripped through `a/b`;
- known well-conditioned S/Z/Y matrices round-tripped through network algebra;
- a port-orientation reversal proving common-sign wave reversal and power
  invariance;
- peak-versus-RMS checks proving the factor of `0.5` in average power.

These fixtures are the entry gate for FDTD source/load coupling. Passing them
does not by itself claim a production LumpedPort, TerminalPort, or RF WavePort.
