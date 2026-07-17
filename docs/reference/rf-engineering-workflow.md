# RF Engineering Workflow

Status: experimental single-device FDTD API

The RF workflow uses the same public architecture as the rest of Maxwell:

```text
Scene -> Simulation -> Result
```

Port orientation, peak-phasor normalization, power-wave definitions, and tensor
ordering are specified in `rf-port-conventions.md`. All live result tensors stay
on the Scene device. Saving a result or exporting a network is an explicit
detached CPU boundary.

## Lumped and Terminal Ports

Use `LumpedPort` when the voltage path and current surface are known directly.
Use `TerminalPort` when the two terminals are named axis-aligned PEC structures.
Both compile to sparse Yee voltage and current weights and use the same FDTD
runtime.

```python
import witwin.maxwell as mw

feed = mw.LumpedPort(
    name="feed",
    positive=(0.0, 0.0, 0.5e-3),
    negative=(0.0, 0.0, -0.5e-3),
    voltage_path=mw.AxisPath("z"),
    current_surface=mw.Box(
        position=(0.0, 0.0, -0.25e-3),
        size=(1.0e-3, 1.0e-3, 0.0),
    ),
    reference_impedance=50.0,
)

scene = mw.Scene(
    domain=mw.Domain(bounds=((-2e-3, 2e-3),) * 3),
    grid=mw.GridSpec.uniform(0.25e-3),
    boundary=mw.BoundarySpec.pml(num_layers=4),
    ports=(feed,),
)

result = mw.Simulation.fdtd(
    scene,
    frequency=10e9,
    excitations=mw.PortExcitation("feed", source_time=mw.CW(10e9)),
    run_time=mw.TimeConfig(time_steps=1000),
).run()

port = result.port("feed")
print(port.voltage, port.current, port.accepted_power)
```

A port may carry a passive `SeriesRLC` or `ParallelRLC` termination. Independent
`Resistor`, `Capacitor`, and `Inductor` objects are added with
`Scene.add_lumped_element(...)`. Active and passive termination semantics are
mutually exclusive in one run.

## Linear circuit co-simulation

Use `Circuit` when a load has internal nodes, controlled sources, scheduled
switches, transient PULSE/SIN/PWL sources, or connections across multiple EM
ports. The circuit binds existing lumped/terminal port names and runs as a
same-step GPU MNA/DAE solve inside FDTD; results are available through
`result.circuit(name)`. The restricted SPICE import, persistence schemas, and
forward checkpoint/resume workflow are documented in
[SPICE/MNA circuit co-simulation](spice-mna-cosimulation.md).

## Network Sweeps

`PortSweep` performs one deterministic run per selected input and returns a
complete `NetworkData` only when every requested column is valid:

```python
result = mw.Simulation.fdtd(
    scene,
    frequencies=(8e9, 10e9, 12e9),
    excitations=mw.PortSweep(),
    run_time=mw.TimeConfig(time_steps=2000),
).run()

network = result.network
z = network.to_z()
y = network.to_y()
renormalized = network.renormalize(75.0)
shifted = network.shift_reference_planes(
    (0.0, 1.0e-3),
    propagation_constants=beta,
)
```

The scattering tensor order is `[frequency, output_port, input_port]`.
Reference-plane shifting, S/Z/Y conversion, renormalization, and mixed-mode
conversion are torch-native and preserve an active autograd graph. Touchstone
export requires a complete detached network.

## RF Wave Ports

`WavePort` describes an axis-aligned modal aperture. Direct excitation and
`PortSweep` solve, normalize, and track the declared modes across frequency.

```python
wave_port = mw.WavePort(
    "input",
    position=(-5e-3, 0.0, 0.0),
    size=(0.0, 10e-3, 5e-3),
    direction="+",
    reference_plane=-5e-3,
    modes=(mw.WaveModeSpec("te", polarization="Ez"),),
)
```

Modal `PortData` retains explicit mode and frequency axes, propagation
constants, characteristic impedance, tracking IDs, and confidence. Reference
planes are shifted with the tracked propagation constants rather than an
empirical phase correction.

## Antenna Results

Declare a closed frequency-domain Huygens surface and name the driven port:

```python
scene.add_monitor(
    mw.ClosedSurfaceMonitor.box(
        "nf2ff",
        position=(0.0, 0.0, 0.0),
        size=(10e-3, 10e-3, 10e-3),
        frequencies=(10e9,),
    )
)

result = mw.Simulation.fdtd(
    scene,
    frequency=10e9,
    excitations=mw.PortExcitation("feed", source_time=mw.CW(10e9)),
    run_time=mw.TimeConfig(time_steps=2000),
).run()
antenna = result.antenna(surface="nf2ff", driven_port="feed")
```

`AntennaData` keeps `[frequency, theta, phi]` fields and reports radiation
intensity, radiated/accepted/incident power, directivity, gain, realized gain,
efficiencies, EIRP, Ludwig-3 co/cross polarization, axial ratio, phase-center
and frame provenance, and equivalent Huygens surface currents.

The exterior surrounding all six monitor faces must currently be homogeneous,
isotropic, and lossless. Surface currents are equivalent currents on the
monitor, not reconstructed subcell conductor currents. One result represents
one excitation column.

## Power Loss

`PowerLossMonitor` is the shared typed loss contract:

```python
scene.add_monitor(
    mw.PowerLossMonitor(
        "device_loss",
        position=(0.0, 0.0, 0.0),
        size=(2e-3, 2e-3, 2e-3),
        frequencies=(10e9,),
        channels=("conduction",),
    )
)

loss = result.power_loss("device_loss")
```

Static bulk electric conduction is computed from full-field FDTD spectra as
`0.5 * sigma_e * |E|^2` with sparse nonuniform Yee control volumes. Other
physics must explicitly supply its volume, surface, line, or integrated channel;
an unavailable channel is never fabricated as zero. `PowerLossData` always
retains its explicit frequency axis.

## Differentiation Boundary

The supported direct lumped adjoint covers real-field, finite positive series
R/L/C branches on `LumpedPort`, `TerminalPort`, and independent R/C/L elements.
It differentiates port V/I, accepted/incident/reflected power,
`available_power`, source amplitude, material tensors, and smooth
`MaterialRegion` density/geometry parameters. The implicit circuit state is
checkpointed and replayed with the Maxwell fields.

Fixed single-mode WavePort direct and sweep results are differentiable with
respect to material or smooth geometry design regions that do not affect the
port aperture or adjacent launch plane. `NetworkData` algebra and
`AntennaData` derived metrics are torch-native.

The following combinations fail explicitly rather than detaching:

- `ParallelRLC` in a differentiable circuit run;
- trainable source impedance or port reference impedance;
- differentiable lumped `PortSweep`;
- observer-only lumped contour ports without an active source or series load;
- trainable WavePort amplitude, multimode WavePort differentiation, or a design
  that affects the modal aperture or adjacent launch plane;
- a differentiable WavePort workflow mixed with lumped ports or independent
  R/L/C elements;
- conductive, dispersive, nonlinear, or complex-field combinations rejected by
  the local lumped-coupling capability guard.
