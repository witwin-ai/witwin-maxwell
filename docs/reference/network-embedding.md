# Embedded network FDTD

`TouchstoneNetwork` connects a complete passive multiport network file to named
FDTD terminal ports. The connection stays inside the normal
`Scene -> Simulation -> Result` workflow and feeds the network current back into
the Yee fields at every step.

## Two-port file example

```python
from pathlib import Path

import witwin.maxwell as mw


ports = (
    mw.LumpedPort(
        name="input",
        positive=(-0.01, 0.0, 0.005),
        negative=(-0.01, 0.0, -0.005),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(position=(-0.01, 0.0, -0.0025), size=(0.005, 0.005, 0.0)),
        reference_impedance=50.0,
    ),
    mw.LumpedPort(
        name="output",
        positive=(0.01, 0.0, 0.005),
        negative=(0.01, 0.0, -0.005),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(position=(0.01, 0.0, -0.0025), size=(0.005, 0.005, 0.0)),
        reference_impedance=50.0,
    ),
)

network = mw.TouchstoneNetwork(
    name="filter",
    path=Path("filter.s2p"),
    connections={1: "input", 2: "output"},
    fit=mw.RationalFitConfig(order=16),
    device="cuda",
)

scene = mw.Scene(
    domain=mw.Domain(bounds=((-0.03, 0.03), (-0.02, 0.02), (-0.02, 0.02))),
    grid=mw.GridSpec.uniform(0.005),
    boundary=mw.BoundarySpec.pml(num_layers=4),
    sources=(
        mw.PointDipole(
            position=(-0.01, 0.0, 0.0),
            polarization="Ez",
            source_time=mw.GaussianPulse(frequency=2.5e9, fwidth=0.5e9),
        ),
    ),
    ports=ports,
    networks=(network,),
    device="cuda",
)

result = mw.Simulation.fdtd(
    scene,
    frequencies=(2.0e9, 2.5e9, 3.0e9),
    run_time=mw.TimeConfig(time_steps=2000),
    spectral_sampler=mw.SpectralSampler(window="none"),
    cuda_graph=True,
).run()

input_data = result.port("input")
filter_data = result.embedded_network("filter")
result.save("embedded-result.pt")
```

The file must cover the requested output and source-effective frequency bands.
Network port order is the file's declared order; `connections` maps that order
to unique `LumpedPort` or resolved `TerminalPort` names. A connected terminal
cannot also carry an excitation or termination.

## Differentiable models

File parsing, automatic fitting, passivity enforcement, and explicit delay are
prepare-time operations and are not differentiable. For gradients, provide a
pre-fitted ordinary-Y `RationalModel` with `fit=False`; its `residues` and
`direct` tensors may require gradients. FDTD checkpoint/replay keeps memory
sublinear in time and also preserves gradients for supported material regions
near the connected ports. `EmbeddedNetworkData` voltage, current, and frequency
power are derived from the differentiable port outputs; `state_norm` is a
detached runtime diagnostic and is not an optimization objective.

Trainable poles or proportional terms, direct trainable state-space matrices,
explicit delay state, and spatial multi-GPU RF embedding fail explicitly. The
last item awaits the shared distributed port ownership and scalar transport
contract; no fallback or inferred owner is used.
