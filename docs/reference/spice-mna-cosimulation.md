# SPICE/MNA Circuit Co-simulation

Status: experimental single-device FDTD API

Linear circuits participate in the ordinary Maxwell workflow:

```text
Scene -> Simulation.fdtd -> Result
```

The runtime is native PyTorch/CUDA. It does not launch an external circuit or
reference solver while FDTD is stepping. The restricted SPICE parser is a
topology and parameter import boundary; the compiled MNA/DAE system, companion
history, source vectors, factorization, and same-step field coupling remain on
the Scene device.

Port voltage, current, polarity, peak-phasor, and power-wave definitions are the
existing contracts in [RF port conventions](rf-port-conventions.md). Circuit
bindings reuse those terminal definitions and do not introduce a second
terminal or execution API.

## Python circuit workflow

Create circuit nodes and shared linear device objects, bind one or more existing
`LumpedPort` or `TerminalPort` names, then add the circuit to the Scene:

```python
import witwin.maxwell as mw

circuit = mw.Circuit("matching_network")
input_node = circuit.node("input")
middle = circuit.node("middle")
output_node = circuit.node("output")

circuit.add(mw.Inductor("L1", input_node, middle, 2.65e-9))
circuit.add(mw.Capacitor("C1", middle, circuit.ground, 2.12e-12))
circuit.add(mw.Inductor("L2", middle, output_node, 2.65e-9))
circuit.bind_port("feed", positive=input_node, negative=circuit.ground)
circuit.bind_port("antenna", positive=output_node, negative=circuit.ground)

scene.add_circuit(circuit)
result = mw.Simulation.fdtd(
    scene,
    frequency=3e9,
    run_time=mw.TimeConfig(time_steps=1000),
).run()

data = result.circuit("matching_network")
print(data.node_voltage("middle"))
print(data.branch_current("L2"))
```

One circuit may bind multiple EM ports. Controlled E/G/F/H sources may reference
nodes or sense branches across those bindings. Independent sources accept DC
values or `PulseWaveform`, `SineWaveform`, and `PiecewiseLinearWaveform`; the
runtime inserts local backward-Euler steps at discontinuities and otherwise uses
the circuit's configured trapezoidal or backward-Euler companion model.

The current coupled execution scope accepts one circuit per Scene. A bound port
cannot also be directly excited or used in `PortSweep`. The initial DC solution
must draw zero current from the initially zero Yee field; ramp active sources
from zero when necessary.

## Restricted SPICE import

`Circuit.from_spice(...)` reads a file; `parse_spice(...)` reads text:

```python
circuit = mw.Circuit.from_spice(
    "matching.cir",
    name="matching_network",
    parameters={"Cmatch": 0.8e-12},
)
circuit.bind_port("feed", positive="input", negative="0")
```

The production subset includes R/L/C/K, independent V/I sources, E/G/F/H
controlled sources, safe parameter arithmetic, sandboxed includes, flattened
subcircuits, `.ic`, and PULSE/SIN/PWL sources. Unknown devices or directives,
executable expressions, unsafe include paths, duplicate devices, unsupported
topologies, and singular source constraints are hard errors. `Circuit.to_spice()`
provides deterministic canonical serialization for supported declarations.

## Result and CircuitData persistence

`CircuitData` contains device-resident sample times, node voltages, physical
branch currents, per-device powers, energy balance, and solver diagnostics. Its
standalone snapshot and the enclosing `Result` snapshot are detached CPU data:

```python
data.save("matching-circuit.pt")
restored_data = mw.CircuitData.load("matching-circuit.pt")

result.save("matching-result.pt")
restored_result = mw.Result.load("matching-result.pt", scene=scene)
```

`CircuitData` uses schema v1. `Result` files use schema v2 and always include a
`circuits` payload; the superseded v1 layout is rejected rather than carried as a
backward-support path. Sharded Result storage writes circuit data once in
coordinator metadata rather than duplicating it in rank field shards. Loaded tensors are detached and no solver, Scene,
autograd graph, or companion history is reconstructed.

## Differentiable coupled execution

On one CUDA device, a trainable circuit-coupled FDTD run checkpoints the circuit
companion state with the Maxwell state and replays the same strongly coupled MNA
step during backward. PyTorch's linear-solve VJP supplies the transpose solve.
Gradients propagate through direct, restricted-SPICE-expression, and
`SceneModule`-derived R/L/C values and independent-source waveform parameters,
as well as supported bound-port material or geometry inputs. Live port outputs
and all live `CircuitData` tensor outputs participate in autograd: node voltages,
physical branch currents (including independent current-source samples),
per-device powers, cumulative energy balance, port power, field-energy change,
and DC/transient condition diagnostics.

Direct `Circuit` construction and `parse_spice(...)` use ordinary eager
PyTorch expression semantics: a derived value such as `2 * base` is
materialized when the circuit is built. Rebuild or reparse the circuit for each
finite-difference or optimization evaluation so the expression sees the
current leaf value and owns a fresh autograd graph. For iterative optimization,
put that construction in `SceneModule.to_scene()`. Mutating `base` does not
retroactively update an already materialized device value, and retaining a
consumed graph is not part of the contract.

This adjoint starts from a fixed exactly-zero coupled operating point and
companion history. Consequently, an objective must slice `CircuitData` time
series from sample 1 onward; a tensor seed at sample 0 is an explicit error.
Trainable DC source values remain unsupported even when the source also has a
waveform, and trainable circuit initial conditions are unsupported. Fixed
nonzero DC/initial state, trainable port reference impedance, and distributed
adjoint execution also fail explicitly. Differentiable circuit parameters use
the generic forward circuit path rather than the fixed-parameter R/C or CUDA
Graph fast paths, including when a prepared dtype cast would otherwise erase a
runtime tensor's `requires_grad` flag. Saved `CircuitData` and `Result` snapshots
remain detached and do not preserve this live graph.

## Resumable forward execution

Forward checkpointing is separate from Result persistence. Advance a fresh
prepared simulation to an absolute step, optionally save the detached state,
and resume into another fresh preparation of the same simulation:

```python
simulation = mw.Simulation.fdtd(
    scene,
    frequency=3e9,
    run_time=mw.TimeConfig(time_steps=1000),
)

checkpoint = simulation.prepare().run_until(400)
checkpoint.save("fdtd-circuit-checkpoint.pt")

checkpoint = mw.FDTDResumeCheckpoint.load(
    "fdtd-circuit-checkpoint.pt",
    map_location="cpu",
)
result = simulation.prepare().run(resume_from=checkpoint)
```

The checkpoint records the absolute step, physical FDTD/CPML state, port DFT
accumulators and Yee times, complete circuit companion state, sampled result
prefixes, power/energy history, source and switch schedules, and an execution
fingerprint. Total steps, time step, grid/material coefficients, sources, port
geometry/impedance/frequencies, circuit topology, parameters, and bindings must
match before restoration. A prepared simulation is single-use.

The current resumable path is a detached single-GPU forward workflow for
circuit/port results. Full-field DFT, field/time observers, spatial multi-GPU
owner state, and adjoint replay checkpoints remain separate capability scopes
and fail explicitly.

## Numerical model

Each bound EM port contributes its discrete Yee Norton relation to the circuit
matrix. The GPU solve obtains all circuit node voltages and bound-port currents
at the same FDTD step, scatters those currents into the port edges, and only then
runs the remaining field corrections and observers. There is no one-step
`V[n] -> circuit -> I[n+1]` feedback path.

Fixed topology and integration/switch states reuse LU factors and preallocated
matrix, right-hand-side, and solution buffers. Topology parsing and validation
are CPU control-plane operations; per-step stamps, histories, source values,
solves, terminal scatter, samples, and diagnostics are CUDA tensors.

With `cuda_graph=True`, fixed built-in PULSE/SIN/PWL and scheduled-switch inputs
are precomputed on the device. The runtime captures one circuit replay per
integration/switch factor class and selects among those static replays without
freezing the source sample index. Fixed R/C networks additionally use batched
incidence-matrix updates for histories, currents, powers, and stored energy.
The internal `compile_batched_mna_factors(...)` primitive caches fixed-shape
`[batch, unknown, unknown]` GPU factors for task-level batches without widening
the one-circuit-per-Scene execution contract.
