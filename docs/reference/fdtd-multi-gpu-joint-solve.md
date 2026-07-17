# FDTD Multi-GPU Joint Solve

- Status: engineering preview
- Execution model: one Python process, homogeneous NVIDIA GPUs, CUDA peer-to-peer
- Validated hardware scope: two NVIDIA RTX A6000 GPUs only

This document is the maintained user and acceptance guide for one FDTD solve that
spans multiple GPUs. It implements the joint-solve direction from the
[detailed implementation plan](../plans/fdtd-multi-gpu-implementation-plan.md) and
the [multi-GPU execution roadmap](../plans/next-functional-2026-07/02-multi-gpu-execution.md).
It is not the independent-simulation ensemble scheduler described in the roadmap.

## Quick start

The public entrypoint remains `Scene -> Simulation -> Result`. Add an
`FDTDParallelConfig` to an ordinary FDTD simulation:

```python
import witwin.maxwell as mw

# Construct the scene on a participating CUDA device. PointDipole sources must use
# profile="ideal" in the current multi-GPU runtime.
scene = mw.Scene(
    domain=mw.Domain(bounds=((-1.0, 1.0), (-0.5, 0.5), (-0.5, 0.5))),
    grid=mw.GridSpec.uniform(0.02),
    boundary=mw.BoundarySpec.pml(num_layers=8),
    device="cuda:0",
)
scene.add_source(
    mw.PointDipole(
        position=(0.0, 0.0, 0.0),
        polarization="Ez",
        profile="ideal",
        source_time=mw.CW(frequency=1.0e9),
        name="source",
    )
)
scene.add_monitor(
    mw.PointMonitor("probe", (0.2, 0.0, 0.0), fields=("Ez",))
)

parallel = mw.FDTDParallelConfig(
    devices=("cuda:0", "cuda:1"),
    decomposition_axis="x",
    transport="auto",       # "auto" and "cuda_p2p" use the same-process P2P path
    overlap=True,
    gather_fields=False,     # monitor-first; do not rebuild a global field by default
    result_device="cuda:0",
)

simulation = mw.Simulation.fdtd(
    scene,
    frequencies=(1.0e9,),
    run_time=mw.TimeConfig(time_steps=1000),
    parallel=parallel,
    cuda_graph=False,
)

# prepare() performs topology, homogeneity, and partition checks. run() performs the
# DFT/output-capacity preflight before advancing the first time step.
prepared = simulation.prepare()
result = prepared.run()

probe = result.monitor("probe")
parallel_stats = result.solver_stats["parallel_stats"]
print(parallel_stats["partitions"])
print(parallel_stats["peak_memory_bytes"])
```

`devices` is ordered and must contain at least two unique, explicitly indexed CUDA
devices. `result_device` defaults to the first device and must be one of the
participants. Only `decomposition_axis="x"` is accepted. Passing `parallel=None`
continues to use the unchanged single-GPU solver.

`transport="auto"` currently selects CUDA P2P. `transport="nccl"` still raises
`RuntimeError` from the single-process `DistributedFDTD` coordinator: torch 2.13
binds one device per rank, so NCCL requires a one-process-per-GPU `torchrun`
launch, and the coordinator that would drive it from a rank-local shard engine is
not landed yet. It never silently switches to P2P or stages through host memory.
The rank-local NCCL transport primitive itself
(`fdtd/distributed/nccl_transport.py`) is implemented and conformance-tested: it
performs the forward electric/magnetic Yee x-plane halos via
`torch.distributed.batch_isend_irecv`, the transposed reverse-halo accumulations
for the adjoint, a scalar all-reduce, a cross-rank homogeneity check, and
deterministic `destroy_process_group` teardown, verified by a two-rank `torchrun`
worker (`tests/fdtd/multi_gpu/test_nccl_transport.py`).

The single-process guard is now pinned by explicit no-silent-fallback tests:
`transport="nccl"` raises the `torchrun` `RuntimeError` both from
`DistributedFDTD` directly and from the public `Simulation` build path, and the
in-process `CudaP2PHaloTransport` is never constructed as a fallback (asserted by
monkeypatching it to fail if reached). `preflight()` also validates an *adopted*
process group: if `torch.distributed` is already initialised, the transport
verifies the live group's world size, rank, and backend match this rank's
expectations before adopting it, rather than binding to a mismatched group; the
matching-group accept path is exercised inside the two-rank worker and the three
mismatch legs (world size, rank, non-NCCL backend) are covered by host tests.

### Operational env contract (`torchrun`)

The one-process-per-GPU launch is `torchrun --standalone --nnodes=1
--nproc-per-node=<gpus> <entrypoint>`, which populates the `RANK`, `WORLD_SIZE`,
and `LOCAL_RANK` variables the transport reads via
`NcclHaloTransport.from_env(...)`; a missing variable raises an actionable error
naming the required launch. Set `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` so a stalled
or dead peer surfaces as a `ProcessGroupNCCL` watchdog abort within the collective
timeout (`timeout_s`, default 1800 s) instead of an indefinite hang; the two-rank
conformance harness sets it by default. Teardown always calls
`destroy_process_group()` so the group is released deterministically even on the
error path.

## Result placement and gathering

The default is monitor-first because gathering a large global field can erase the
memory-capacity benefit of domain decomposition.

| Setting | Public result behavior |
| --- | --- |
| `gather_fields=False` | Supported point/time and assembled plane/flux/mode monitor payloads and metadata are moved to `result_device`; `result.fields` is empty. Rank-local mutable fields remain solver internals and are not a public distributed-field object. |
| `gather_fields=True`, one frequency, `full_field_dft=False` | The last-step global `Ex`, `Ey`, and `Ez` fields are assembled on `result_device`. |
| `gather_fields=True`, full-field DFT or multiple frequencies | Global electric DFT fields are assembled on `result_device`; multi-frequency tensors retain the leading frequency axis. |

Magnetic full fields are not emitted by the public `Result` field contract in this
preview. Request magnetic components through supported point monitors when they are
needed.

Before a gathered run starts, the solver checks the destination's free memory for the
global electric output plus pending local DFT work. It raises `MemoryError` rather than
starting a solve that cannot complete the requested gather. The check is reported in
`parallel_stats["gather_preflight"]`.

### Result persistence

`Result.save(path)` and `Result.load(path, scene=...)` keep their ordinary gathered
snapshot semantics: fields, monitors, metadata, and solver statistics are serialized,
but live solvers, CUDA streams, peer transport state, and prepared-scene runtime state
are not.

For a distributed result, `Result.save_sharded(directory)` asks the coordinator for
owned electric field shards and writes a manifest plus one detached CPU tensor payload
per rank. It does not assemble a global field first. The manifest records component
global x intervals, tensor shapes/dtypes, frequency metadata, and rank files; monitor
payloads, result metadata, and solver statistics remain in the result-level metadata.

`Result.load_sharded(directory, scene=..., gather_fields=False)` is lazy with respect
to rank tensor deserialization. It validates the manifest and required rank paths,
returns `fields == {}`, and exposes the sharded manifest and rank paths. Passing
`gather_fields=True` loads and validates the rank payloads, stitches each component's
contiguous owned intervals in global x order, and returns conventional gathered
fields on `map_location`. These APIs restore detached inference results, not a
resumable distributed solve.

## Architecture and ownership

This is spatial domain decomposition, not replicated data parallelism:

1. `FDTDPartitionPlan` balances the global physical x cells into contiguous half-open
   slabs. PML cells at the global x faces stay with the first and last slabs.
2. `FDTDShardLayout` gives every staggered component explicit owned, allocated, halo,
   and global-coordinate slices. `Ex`, `Hy`, and `Hz` follow cell ownership;
   `Ey`, `Ez`, and `Hx` follow low-node ownership, with the terminal node assigned to
   the final shard. An interface value therefore has exactly one owner.
3. Python is the control plane in both modes, but the time-loop ownership differs:
   the single-GPU solver owns its ordinary loop, while `DistributedFDTD` is an
   independent distributed coordinator with its own shard orchestration, halo exchange,
   event ordering, and distributed time loop. This is not a claim that the Python loops
   are the same.
4. The numerical core is shared. Every shard calls the same native CUDA Yee, boundary,
   CPML, conductivity, and ADE kernels as the single-GPU solver. The standard real-field
   path exposes six bounded native operations for `Hx/Hy/Hz` and `Ex/Ey/Ez`; the
   legacy full-domain standard wrappers invoke those same implementations with the full
   x range. There is no second Python or Torch field-update implementation.
5. One-plane asymmetric halos carry only the tangential components required by the
   second-order stencil. `Ey/Ez` move from the right shard to the left shard before
   the magnetic update. `Hy/Hz` move from the left shard to the right shard before
   the electric update.
6. A dedicated communication stream and CUDA events order each peer copy against both
   producer and destination use. With `overlap=True`, eligible standard real-field
   runs compute interior regions while halos are in flight and wait only for the
   boundary strip. CPML and modulation currently use the serialized-safe schedule;
   inspect `overlap_active` instead of assuming a request was activated.
7. Only the first and last slabs apply physical x-face boundary behavior. Internal
   interfaces never apply PML, PEC, PMC, Mur, periodic, or symmetry boundary rules.
   Every shard still owns the global y/z faces.
8. For a circuit-coupled run, each bound lumped/terminal port has one Yee-edge
   owner. The circuit owner is the shard owning the lexicographically minimum
   bound voltage edge, with `Ex`, `Ey`, `Ez` as the deterministic component
   tie-break. A remote port sends one voltage scalar to that owner and receives
   one solved current scalar per step; ports already on the circuit owner take a
   zero-P2P fast path. MNA state, circuit time series, port spectra, and live
   circuit checkpoint tensors remain on the circuit owner until final results are
   moved once to `result_device`. Each remote stream acknowledges completion of
   its current-scalar peer read, and the next owner solve waits for that event
   before reusing the source buffer; the acknowledgement does not wait for the
   subsequent shard-local field correction.
9. An embedded-network-coupled run uses the identical ownership rule and scalar
   contract as the circuit path, applied to the network's connected terminals: the
   network owner is the shard owning the lexicographically minimum connected
   voltage edge with the same `Ex`, `Ey`, `Ez` component tie-break, so a scene
   carrying both a circuit and a network would place them consistently. Each shard
   samples one voltage scalar per local connected port; the owner gathers the
   remote scalars, runs the sole authoritative state-space recurrence and the
   prepare-time pivoted-LU same-step feedthrough solve, accumulates every port's
   V/I DFT, and scatters one solved branch-current scalar back per remote port
   under the same copy-acknowledgement discipline. Network state and the owner
   `PortData`/`EmbeddedNetworkData` remain on the owner until moved once to
   `result_device`. The distributed apply is multi-stream and event-driven and is
   never CUDA-Graph captured.

The steady-state fields and field-sized CPML/ADE/DFT state are shard local. The
coordinator prepares global grid metadata and a common time step, but does not
initialize a complete duplicate six-field simulation on every GPU.

## Current support matrix

“Accepted” means the current public runtime has an implementation path. “A6000 parity”
means a two-GPU case is in the hardware acceptance suite and is compared with the
single-GPU native solver. Features accepted by the runtime but not named in that suite
must not be described as independently hardware-qualified.

| Area | Current joint-solve status | Evidence / notes |
| --- | --- | --- |
| Devices and topology | Accepted for homogeneous, explicitly indexed NVIDIA CUDA devices with bidirectional direct peer access | Two RTX A6000 GPUs validated. No CPU/host-staged fallback. |
| Decomposition | x slabs, one-cell halo, balanced or uneven physical-cell counts | A6000 parity covers uniform and uneven nonuniform x grids, including an interface source. |
| Grid | Uniform, anisotropic-spacing, custom/nonuniform, and prepared auto grids through shard-local custom coordinates | Uniform and uneven custom-grid parity are explicit acceptance cases. |
| Linear materials | Vacuum, scalar dielectric, diagonal electric/magnetic tensors, scalar/diagonal conductivity | Scalar dielectric plus CPML and `sigma_e` are in A6000 parity. Full off-diagonal electric tensors fail fast. |
| Dispersion | Electric and magnetic Debye/Drude/Lorentz ADE state is shard local | Conductive electric Lorentz and an interface-crossing magnetic Lorentz case have A6000 parity. Other pole combinations retain runtime support but are not separately hardware-qualified. |
| Modulation | Forward serialized-safe path | Communication/compute overlap is disabled for this path; no final A6000 acceptance claim yet. |
| Nonlinearity | Rejected | Additional collocation halos and bounded nonlinear kernels are required. |
| Absorbers and boundaries | Global-face CPML/PML/StablePML/absorber behavior and mixed `none`/PEC/PMC/Mur; y/z periodic and y/z symmetry are accepted | XYZ CPML plus dielectric, mixed PEC/PMC/Mur, y periodic, and y symmetry have two-GPU parity coverage. x periodic, all Bloch, and x symmetry fail fast. |
| Sources | `PointDipole(profile="ideal")` and `UniformCurrentSource` | Point sources on/near interfaces and a current volume crossing the interface have A6000 parity coverage. Plane-wave/TFSF, beam, mode, and other surface sources fail fast. |
| Circuit co-simulation | One linear circuit with one or more bound `LumpedPort`/`TerminalPort` objects; GPU-native owner MNA and P2P scalar exchange | Each individual port must be wholly owned by one x slab. Multiple ports may reside on different shards. Communication is two scalars per remote bound port per step, and no external circuit process is used. The physical two-GPU parity gate passes on 2x RTX A6000 with exact (bitwise, 0.000e+00) single-versus-two-GPU parity on all six fields, port V/I, and circuit node/branch data. |
| Embedded-network co-simulation | One `NetworkBlock`/`TouchstoneNetwork` connecting one or more `LumpedPort`/resolved `TerminalPort` terminals; GPU-native owner state-space + same-step LU feedthrough solve and P2P scalar exchange | Each connected terminal must be wholly owned by one x slab. Terminals may reside on different shards. Communication is two scalars per remote connected port per step; the owner alone advances the authoritative network state recurrence and accumulates every port V/I DFT. The physical two-GPU parity gate passes on 2x RTX A6000 with exact (bitwise, 0.000e+00) single-versus-two-GPU parity on all six fields, port V/I, network V/I, and network state. A scene mixing an embedded network with a bound circuit, more than one embedded network, and delayed (`delay_seconds=`) network cores are rejected on the distributed path. |
| Thin-wire co-simulation (forward) | One PEC `ThinWire` network on the shared real standard path with a non-absorbing boundary; distributed fragment ownership with owner-resident `I/q` and monitors | Every cell-local sampling/deposition Yee edge is applied by the shard that owns it; a physical segment may span any number of shards and its fragments split at each partition boundary. Per-segment EMF partials are reduced to the state owner (the shard owning the minimum global sampling edge, declaration-order independent) with a deterministic rank-ordered sum; `I/q` advance owner-only; the advanced current is broadcast back so each owned edge deposits exactly once. The shared joint Maxwell/wire dt is taken from the full-domain single-GPU wire runtime. Two-GPU A6000 gate: a straight wire crossing the x split is bitwise (0.000e+00) single-versus-two-GPU on all six fields and wire current/charge at a short horizon while carrying ~36 A across the split; a closed lossless loop crossing the split preserves the discrete energy invariant to ~2e-9 relative even where individual field phases drift by chaotic float32 halo-curl accumulation over long runs. A trainable wire, a distributed CPML boundary, and a wire mixed with an embedded network or lumped circuit fail closed at prepare. Finite-conductor loss and distributed wire reverse/gradient communication are deferred. |
| Monitors | Point spectral monitor, point `FieldTimeMonitor`, a valid `DipoleEmissionMonitor`, and spectral `PlaneMonitor`/`FinitePlaneMonitor`/`FluxMonitor`/`ModeMonitor` | y/z-normal planes are tiled across x and stitched from owned component intervals on `result_device`; x-normal planes have exactly one shard owner. Five Plane/Flux/Mode A6000 numerical cases observed exact single/two-GPU parity. Closed-surface, diffraction, flux-time, non-point field-time, and material monitors remain rejected. |
| Spectral output | One or many requested frequencies; supported point/plane monitor assembly; optional gathered electric fields | Multiple frequencies preserve frequency metadata and leading frequency dimension. |
| Persistence | Gathered `save`/`load` and distributed `save_sharded`/`load_sharded` | Lazy sharded load leaves `fields` empty and rank tensors unopened; explicit gather validates and stitches owned intervals. Persistence does not restore live solver/transport state. |
| Source normalization | Accepted only when exactly one logical source is present | Multiple logical sources fail fast when normalization is requested. |
| Auto shutoff | Global owned-electric-energy reduction on `result_device` | `steps_run`, halo totals, and normalization reflect early termination. |
| CUDA Graph | Rejected | Peer communication is not graph captured. |
| Plotting | Rejected during solve | Run with `gather_fields=True`, then consume the gathered result explicitly. |
| Trainable scenes / adjoint (Box density, standard path) | Accepted with A6000 parity | A trainable `Box` `MaterialRegion` density on the pure real standard (open/PEC boundary) path routes through the distributed joint-solve adjoint bridge: per-shard forward checkpoints, a transposed reverse step per forward step (Phase 1 → transposed magnetic halo → Phase 2 → transposed electric halo → Phase 3 → per-shard source-term eps grad), and a global grad_eps gather feeding the existing single-GPU material pullback once. Two-GPU acceptance: 1-vs-2-GPU objective parity (bit-identical forward) and gradient parity (rtol 1e-4, atol tied to grad magnitude); central finite differences (~1e-5 relative) on density texels on the x-split and interior to each shard, with the source and point-monitor objective on the interface node; 1-vs-2-GPU full-field-DFT objective gradient parity (right-half `EZ` power loss isolating the non-result shard's cotangent scatter leg); a checkpoint-capture ordering contract (every mid-loop capture preceded by a stream sync); and bitwise-reproducible gathered grad_eps. Point-monitor spectra and full-field DFT objectives only. |
| Trainable scenes / adjoint (other channels) | Rejected before distributed allocation | Trainable geometry, material perturbation, circuit, RF/port, and embedded-network (residue/direct-term) parameters have no verified distributed reverse core; any absorbing (PML) boundary (CPML/stable-PML and the legacy graded-sigma `pml`/`absorber` absorbers), dispersive/conductive/nonlinear/anisotropic/modulated media, field shutoff, and tiled plane/flux/mode-seeded objectives are rejected at prepare before any distributed allocation. Absorbing-boundary-trainable and tiled-monitor seed scatter are the remaining follow-ups (S4/S5). |
| NCCL transport primitive | Rank-local primitive implemented and conformance-tested | `fdtd/distributed/nccl_transport.py` provides the one-process-per-GPU (`torchrun`) transport surface: forward electric/magnetic Yee x-plane halos via `batch_isend_irecv`, transposed reverse-halo adjoint accumulations, a scalar all-reduce, an env/world-size/non-Linux failure matrix, an adopted-process-group world-size/rank/backend validation, a cross-rank homogeneity check, and deterministic teardown. A two-rank `torchrun` worker (`tests/fdtd/multi_gpu/test_nccl_transport.py`) asserts bitwise halo round-trips (including endpoint-ghost negative invariants and the matching-group adopt path) and adjoint accumulation. |
| NCCL no-silent-fallback guard | Pinned | `transport="nccl"` on the single-process runtime raises the `torchrun` `RuntimeError` from both `DistributedFDTD` and the public `Simulation` build path, and never constructs the in-process `CudaP2PHaloTransport` as a fallback (asserted by monkeypatching P2P to fail if reached). |
| End-to-end NCCL solve / multi-process / multi-node | Not landed | `transport="nccl"` still raises explicitly from the single-process `DistributedFDTD`; the rank-local shard-engine coordinator that would drive the NCCL transport across processes is the remaining Phase 8 work. Cross-node is out of scope. |
| Three or four GPUs | Structurally representable by the partition and neighbor transport | Not qualified in the current hardware acceptance record; no validation claim is made here. |

## Fail-fast conditions

Configuration, preparation, or the solve preflight raises a specific error for the
following cases rather than silently changing the physics or execution model:

- fewer than two devices, duplicate/unindexed/non-CUDA devices, a non-x axis, or a
  `result_device` outside the participant list;
- unavailable devices, heterogeneous device model/compute capability, any missing
  bidirectional neighbor P2P link, or missing bidirectional P2P between
  `result_device` and a shard;
- too few physical x cells for the shard count/halo, or insufficient device memory
  for local DFT state or an explicitly requested gather;
- trainable geometry, material-perturbation, circuit, or RF/port parameters (a
  trainable `Box` `MaterialRegion` density on the standard path is accepted and
  routed to the distributed adjoint bridge; a non-Box density region still fails);
- for a trainable distributed run: any absorbing (PML) boundary (CPML/stable-PML or
  the legacy graded-sigma `pml`/`absorber` absorbers), dispersive/conductive/nonlinear/
  anisotropic/modulated media, field shutoff, or a tiled plane/flux/mode-seeded
  objective (point-monitor and full-field-DFT objectives only);
- nonlinear media, full off-diagonal electric anisotropy, or lossy-metal/SIBC ownership;
- Bloch boundaries, periodic x, or x-axis symmetry;
- mode ports, unbound lumped/terminal ports, one bound port whose voltage edges
  cross an x-slab split, unsupported source classes, non-ideal point-dipole
  profiles, or an invalid dipole-emission source reference;
- `ClosedSurfaceMonitor`, `DiffractionMonitor`, `FluxTimeMonitor`, non-point
  `FieldTimeMonitor`, `PermittivityMonitor`, and `MediumMonitor`;
- an ordinary non-flux x-normal `PlaneMonitor` or `FinitePlaneMonitor` that requests
  `Ex` exactly on an internal x partition node. The owner lacks a current
  monitor-only low-`Ex` halo; move the plane off the split, remove `Ex`, or use a
  tangential-field `FluxMonitor`/`ModeMonitor`;
- `cuda_graph=True`, solve-time plotting, NCCL transport, or source normalization with
  more than one logical source.

This list is intentionally narrower than the single-GPU FDTD feature list. Removing a
guard requires interface-aware numerical parity and hardware evidence; accepting a
scene locally is not enough.

## Diagnostics and solver statistics

Use either `result.solver_stats["parallel_stats"]` or
`result.stats()["parallel_stats"]`. The nested record includes:

- `devices`, `decomposition_axis`, `transport`, and the peer `topology` snapshot;
- `overlap_requested` and the truthful `overlap_active` result;
- `gather_fields`, `result_device`, `gather_preflight`, and per-device `dft_preflight`;
- per-rank physical/global cell and node extents in `partitions`;
- `halo_bytes_per_step` and `halo_bytes_total`;
- per-device `peak_memory_bytes` before gathering and
  `peak_memory_bytes_including_gather` after output collection;
- `wall_time_s`.

Circuit-coupled runs additionally expose `parallel_stats["circuit"]`, including
the circuit and per-port owner ranks, same-shard and remote-port counts, scalar
transfers, owner copy acknowledgements, bytes per step/total, and the
circuit-checkpoint owner. Embedded-network-coupled runs expose the same shape
under `parallel_stats["network"]` (network and per-port owner ranks, connected
and remote-port counts, scalar transfers, copy acknowledgements, bytes per
step/total, and `communication_order == "O(connected_ports)"`). These
statistics count only circuit/network scalar traffic; halo bytes remain in the
top-level halo counters.

`compute_time_s`, `communication_time_s`, and
`exposed_communication_time_s` are currently `None`. Per-phase timing would require
synchronization that is deliberately absent from the production loop; use a CUDA
profiler and retain the accompanying `timing_note` instead of interpreting these keys
as zero communication cost.

## Dual-A6000 acceptance

Run the tests and benchmark from the repository root with both A6000 devices visible.
The examples below use the repository's `maxwell` environment; an equivalent Python
environment with the current native extension is acceptable.

```bash
nvidia-smi topo -m

/home/xingyu/miniconda3/envs/maxwell/bin/python -m pytest \
  tests/fdtd/multi_gpu -q --basetemp=/tmp/witwin-mgpu-pytest

/home/xingyu/miniconda3/envs/maxwell/bin/python \
  scripts/dev/fdtd/multi_gpu/bench_joint.py \
  --devices cuda:0 cuda:1 \
  --nodes-x 257 --nodes-y 257 --nodes-z 257 \
  --steps 100 --warmups 1 --repeats 5 \
  --weak-scaling --p2p-size-mib 256 \
  --assert-gates \
  --json /tmp/fdtd-multi-gpu-a6000.json
```

Run `--assert-gates` only on an otherwise idle, thermally stable host. The harness
checks a six-component one-versus-two-GPU diagnostic, both P2P directions, strong and
weak scaling, and peak memory. Its default gates are:

- diagnostic maximum absolute error at most `2e-6`, maximum relative error at most
  `2e-5`, finite fields, and nonzero reference signal;
- median P2P bandwidth at least 40 GB/s;
- strong-scaling speedup at least 1.0;
- weak-scaling efficiency at least 0.70;
- maximum per-GPU peak at most 1.15 times the ideal half-single-GPU allocation, plus
  any explicitly supplied fixed-overhead allowance.

The benchmark's default `gather_fields=False` measures the scalable monitor-first
solve. Run an additional `--gather-fields` pass when validating result assembly, but
do not mix gather allocation/time into the core scaling number.

### Physical two-GPU circuit gate

The circuit-owner acceptance case is a real single-versus-two-GPU solve, not a
mock transport test. It places two bound ports on different x slabs, exercises the
same-shard fast path and one bidirectional remote scalar exchange, and compares
gathered fields, port phasors, circuit time series, result placement, communication
accounting, and checkpoint ownership:

```bash
python -m pytest \
  tests/fdtd/multi_gpu/test_circuit_owner.py::test_physical_two_gpu_circuit_matches_single_gpu_and_reports_scalar_contract \
  -q
```

The gate requires two homogeneous GPUs with bidirectional CUDA peer access and uses
`rtol=2e-5` for field/circuit parity. It skips whenever the run sees fewer than two
peer-accessible devices, in which case the file reports `6 passed, 1 skipped` and the
skipped item is exactly this gate. On 2x RTX A6000 with both devices visible the file
reports `7 passed`, and this gate passes by an exact (bitwise, 0.000e+00) margin on all
six fields, both bound-port voltages/currents, and the circuit node/branch series. The
full deviation table and the scalar-transfer contract are recorded in
`docs/assessments/spice-mna-phase-4-acceptance.md`.

### Physical two-GPU embedded-network gate

The network-owner acceptance case mirrors the circuit gate for a two-port embedded
network split across x slabs. A `PointDipole(profile="ideal")` excites a scene whose
two connected `LumpedPort` terminals sit on different shards; the owner runs the sole
state-space recurrence and same-step LU solve, and the run compares gathered fields,
port phasors, network V/I, network state, result placement, the scalar-communication
contract, and checkpoint ownership:

```bash
python -m pytest \
  tests/fdtd/multi_gpu/test_network_owner.py::test_physical_two_gpu_network_matches_single_gpu_and_reports_scalar_contract \
  -q
```

The gate requires two homogeneous GPUs with bidirectional CUDA peer access and uses
`rtol=2e-5` for field/port/network parity. It skips whenever fewer than two
peer-accessible devices are visible. On 2x RTX A6000 with both devices visible this
gate passes by an exact (bitwise, 0.000e+00) margin on all six fields, both connected
port voltages/currents, the network V/I, and the network state, and asserts the
`parallel_stats["network"]` scalar contract (owner rank, per-port owner ranks, two
scalar transfers per step, one owner copy-acknowledgement, `O(connected_ports)`). The
file's ownership-unit and fail-closed cases (deterministic owner independent of
connection order, one terminal spanning a split, overlapping edges, network+circuit
rejection, and trainable+parallel rejection) run without GPUs.

### Clean-build test record

The final bounded-operator source was rebuilt cleanly with the CUDA 13 toolchain before
this acceptance record. Scoped checkpoints from that build are listed separately
because their membership can evolve; no aggregate total is implied:

- the multi-GPU suite passed 109 tests, then the five dual-A6000 numerical
  Plane/Flux/Mode cases raised that suite checkpoint to 114 passed;
- the CUDA FDTD suite passed 65 tests;
- the public API suite passed 31 tests;
- the focused numerical/persistence matrix passed 7 tests.

### Final benchmark measurements

These are final measurements for this engineering-preview implementation on the named
two-A6000 host, not estimates from the pre-bounded-kernel prototype. The memory factor
is the reported maximum per-GPU peak divided by ideal half of the one-GPU peak.

| Workload | One GPU | Two GPUs | Strong speedup | Weak efficiency | Memory factor | Numerical observation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Vacuum `257 x 257 x 257`, 100 steps | 3.4659 ms/step | 1.7559 ms/step | 1.97385 | 0.99274 | 1.01058 | Six-field diagnostic observed exact single/two-GPU agreement |
| CPML dielectric `257 x 129 x 129`, slab CPML | 1.0040 ms/step | 0.8116 ms/step | 1.23701 | — | 1.03698 | Diagnostic observed exact agreement |
| Two-frequency full-field DFT `257 x 129 x 129` | 1.7001 ms/step | 1.1110 ms/step | 1.53026 | — | 0.76273 | Multi-frequency DFT numerical acceptance passed |
| Two-frequency full-field DFT `129 x 65 x 65` | 0.3012 ms/step | 1.1284 ms/step | 0.26692 | — | passed | Numerically exact and memory gate passed; below the two-GPU break-even size |

The same host measured 52.65-52.69 GB/s direct P2P bandwidth across the two directions.
The small case is intentionally retained: it shows that correct domain decomposition
does not imply a speedup when launch and halo overhead dominate.

### Evidence boundary

The acceptance host contained only two RTX A6000 GPUs connected through NV4. No
three- or four-GPU system and no PCIe-only peer topology were available, so this record
does not qualify either case. Nsight Systems and Nsight Compute were not available on
the host; the benchmark therefore supplies no profiler trace or per-kernel roofline
claim.

This remains an engineering preview. The distributed joint-solve adjoint is qualified
only for a trainable `Box` `MaterialRegion` density on the pure real standard path with
point-monitor/full-field-DFT objectives (1-vs-2-GPU objective/gradient parity and
interface finite differences; timing/speedup of the reverse pass is not measured here).
The measurements and tests above do not qualify
end-to-end NCCL, multi-process or multi-node execution, distributed CPML-trainable
adjoint or
tiled-monitor adjoint seed scatter,
peer-aware CUDA Graph capture, advanced plane/beam/mode/TFSF sources, ports, x-periodic
or Bloch decomposition, x symmetry, nonlinear media, full off-diagonal media, SIBC, or
any other guarded feature. The rank-local NCCL transport primitive is separately
conformance-tested (two-rank `torchrun` halo round-trip), but no end-to-end NCCL
solve is qualified because the shard-engine coordinator is not landed.
