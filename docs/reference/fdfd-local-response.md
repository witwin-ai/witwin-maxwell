# Native GPU FDFD local responses

The frequency-domain runtime supports uniform, unequal-axis, custom and automatic
nonuniform Cartesian Yee grids, absorbing layers, fixed staircase PEC structures
and PEC domain faces. Local responses add precomputation, small linear observations,
finite dielectric or exact binary PEC candidate updates, and native PyTorch
material gradients. All matrix assembly and numerical solve/update work stays on
CUDA. Sparse symbolic planning and scalar control/diagnostic metadata use the host.
The local workflow explicitly disables cuDSS hybrid host factor storage.

```python
import torch
import witwin.maxwell as mw

config = mw.LocalResponseConfig(
    design=mw.LocalDesignRegion(bounds=((-0.02, 0.02),) * 3),
    observations=(mw.FieldProjection("probe", component="Ez", position=(0.05, 0, 0)),),
    rhs_batch_size=32,
    max_response_bytes=2**30,
    tolerance=1e-7,
)
simulation = mw.Simulation.fdfd(
    scene, frequency=1e9,
    solver=mw.GMRES(solver_type="direct", precision="double"),
    local_response=config,
)
with simulation.prepare() as prepared:
    result = prepared.run(scene=candidate_scene)
    value = result.monitor("probe")["value"]
    loss = value.abs().square()
    loss.backward()  # Candidate MaterialRegion/SceneModule parameters.
```

The supplied Scene is the reference operator. Candidate material changes must be
confined to the declared electric support; bounds include edge endpoint averaging
halos. `prepared.design_indices` returns an owned int64 CUDA vector with global Ex,
Ey, Ez indices. `delta=` alternatively accepts the actual local operator diagonal
change, `k0**2 * delta_epsilon_edge`, not relative permittivity itself. `pec=` is a
boolean vector and requires explicit `delta=`. Reference PEC must be outside the
design support; a candidate Scene can add/remove conductor constraints inside it.

For a fixed source, grid and background, the precomputation stores only the local
Green matrix and small observation projections. RHS batches reuse GPU sparse
factors, including partial last batches. It does not retain full-domain inverse
columns. Estimated response bytes are a preflight estimate excluding sparse
factors and library workspaces, not an allocator hard quota. Frequency sweeps
retain separate contexts and require enough memory for all of them.

## Trial and accepted designs

```python
trial = prepared.trial(delta=next_diagonal, pec=next_pec)
score = objective(trial.projections)
if score < best_score:
    result = prepared.accept(trial)
```

Trials do not modify accepted state. Sparse row changes use a small Woodbury
system. Accepted updates certify the current solution and an inverse action;
periodic direct reduced refactorization limits accumulated drift. The trial's
projections are owned CUDA tensors. Stale or cross-context acceptance fails.
`trial.result()` explicitly reconstructs all local fields, costing a dense Green
matvec; only `trial.projections` preserves the small-update evaluation cost.

`prepared.run(...)` is an independent differentiable evaluation and does not alter
the accepted design. `full_fields=True` recovers full Yee E through an additional
background solve and uses a checked implicit background adjoint during backward.
Material autograd passes through the existing scene compiler, including density
filters and interpolation. Binary PEC geometry, trainable sources/ports and edits
outside the fixed support are rejected. Local responses do not differentiate a
history of in-place candidate updates.

## Observations and excitation

- `FieldProjection(name, component="Ex"/"Hx"/..., position=(...))` interpolates
  native Yee fields. H is obtained with the same nonuniform discrete curl and
  exp(-i omega t) convention. Magnetic samples in PML are rejected.
- `FieldProjection(name, indices=..., weights=...)` defines a fixed linear electric
  functional, with no implicit conjugation. Weights/indices must be on CUDA.
- `FieldProjection.voltage(name, start, end)` computes -integral E dot dl along
  an axis-aligned node-snapped path.
- `LumpedCurrentSource(start, end, current=..., name=...)` is an ideal phasor
  current in amperes, normalized by the transverse dual Yee area. The matching
  voltage/current ratio gives input impedance; `(Z-Z0)/(Z+Z0)` gives reflection
  for real positive Z0. This is a prescribed current gap, not a terminated
  circuit/waveport model.
- `UniformCurrentSource` supports CW electric volume currents in full FDFD.
- `FarFieldProjection(name, bounds=..., direction=..., polarization=...)` compiles
  a closed vacuum Huygens surface into a linear functional. The output is
  `r exp(-ikr) E` in V (electric far-field amplitude). Direction and transverse
  polarization must be unit vectors. A local context verifies that the box
  encloses all discretized source/scatterer/design support. Non-vacuum exterior
  Green functions are not implemented by this observation compiler.
- Ordinary full FDFD accepts the same projections through `observations=...`.
  Do not also pass that argument when using `local_response`; put observations
  in LocalResponseConfig instead.

## Multiple frequencies and binary design

`Simulation.fdfd(..., frequencies=(...))` prepares one operator per frequency and
returns a Result with stacked frequency outputs. A local sweep supports the same
run/trial/accept operations. Explicit delta can have shape `(n_frequencies,n_local)`;
a one-dimensional contrast applies the same operator delta to all frequencies.
For physical nondispersive epsilon changes prefer candidate Scenes, or supply the
proper k0-squared scaling separately per frequency. Sweep acceptance rolls back
all contexts if any frequency fails.

`prepared.optimize_binary(groups, objective, initial=..., mode="pec", max_passes=...,
seed=...)` minimizes a finite scalar CUDA objective of projected fields. Groups are
CUDA vectors of *local* design indices. Shared edges use union occupancy. The
`dielectric` mode takes a physical `epsilon_contrast` and applies the frequency
scaling. The result contains an ordinary Result, final tiles, monotonically
improving objective history, counters and a local-convergence flag. Exhausting
max_passes is not convergence; a complete unsuccessful pass certifies only a
single-tile local optimum. RNG state is private to the optimizer.

See `examples/fdfd/local_response_antenna.py` for a compact multifrequency graded-grid
substrate/ground/feed/far-field workflow. It is not a reproduced published device.

## Qualification limits

The response is exact relative to its underlying discretization and solve tolerance;
this does not eliminate mesh, PML, or far-field quadrature error. Report cold
precomputation and initialization separately from warm trial and accepted-update
cost. Rejected trial complexity is O(n r^2+r^3), acceptance is O(r n^2), and dense
storage is O(n^2). Dense whole-design changes do not retain low-rank speedups.
The backend uses blocked selected inverse actions rather than CPU APF. Nonlinear
media, gyromagnetic response and full off-diagonal material tensors remain outside
the current FDFD physical model; AMS still requires a uniform auxiliary grid.
Fixed scalar/diagonal permeability is supported, including effective complex
magnetic loss/dispersion. Local updates and material gradients require unchanged
permeability; changing magnetic parameters or moving magnetic geometry fails
explicitly. H observations include the corresponding local permeability.

## Transverse mode helper

A single-frequency ordinary FDFD preparation also provides
`prepared.solve_modes(axis="z", position=0, bounds=((xmin,xmax),(ymin,ymax)), num_modes=1)`.
The aperture imposes rectangular PEC walls independently of the full-domain
absorbing boundary. Bounds follow cyclic transverse axes: x/y for z, y/z for x,
z/x for y. `result.monitor("modes")["beta"]` contains propagation constants;
`result.raw_output["profiles"]` holds six staggered E/H arrays with a leading mode
dimension, normalized to peak transverse E = 1 V/m. This CUDA first-order Maxwell
elimination supports nonuniform cells and diagonal complex epsilon. It uses a
bounded dense eigensolve (`max_dofs=2048` by default). It does not support interior
PEC geometry, nonvacuum permeability, modal material gradients, or automatic modal source injection.

FDFD phasors consistently multiply `exp(-i omega t)`: CW waveform phase maps to
`amplitude * exp(-i phase)`, and passive conductivity contributes positive
imaginary epsilon/mu. The FDFD material boundary corrects the shared static electric
conductivity sampler's opposite sign and adds magnetic conductivity exactly once;
existing pole susceptibilities are preserved. This changes the previous FDFD
static-loss and nonzero-CW-phase behavior. Explicit operator `delta` already uses
the FDFD convention and is not conjugated.

## Separate precomputation and optimization processes

```python
with simulation.prepare() as prepared:
    prepared.save_cache("background.pt")

# In a later process, construct the same Scene and Simulation.
with simulation.prepare(response_cache="background.pt") as prepared:
    trial = prepared.trial(delta=next_diagonal)
```

For a frequency sweep, pass a directory to both methods. Files hold background
responses only; accepted design state starts from the reference again. The cache
fingerprints the actual sparse operator, reciprocal scale, source, grid, design
indices, reference materials and projection arrays. Changed physics or precision,
corrupted tensor checksums and insufficient stored residual accuracy fail before
use. Load uses the restricted tensor deserializer. Sparse matrices are reassembled
for validation, but projection/update/gradient workflows skip factorization;
full-field recovery factorizes lazily if requested. CUDA tensors are copied through
CPU memory only for explicit persistence and checksums. This native format does
not import the reference package's binary Gmat files.

The antenna example accepts `--cache DIR` to save and `--cache DIR --reuse-cache`
to reuse all frequency responses in a separate process.

Performance: `prepared.trial_updates(indices, delta=..., pec=...)` supplies absolute
values on selected local rows. Indices and values are CUDA tensors; unspecified
rows retain their accepted values. Selected unchanged rows are allowed. Candidate
outputs remain owned and accepting a candidate invalidates earlier generations.
The default maximum inverse rebuild interval is 128; current residual and two
inverse-action probes are checked on every acceptance and can trigger earlier
reconstruction. Public `accept` still materializes a standard Result.

`prepared.trial_updates_batch(indices, delta=..., pec=...)` returns a tuple of
same-generation trials. `indices` has shape `(batch, rank)` and must contain unique
local indices within each row; values have the same shape. The fused path handles
rank <= 32, with the general CUDA path for larger ranks. A sweep also accepts
`delta` shaped `(frequency, batch, rank)`. Any invalid batch leaves accepted state
unchanged. Accepting one trial makes all other trials from that generation stale.

`optimize_binary(..., batch_size=16)` batches proposals sharing an accepted state.
It consumes them in the existing random order, accepts the first improvement and
recomputes the remaining proposals at the new generation. This preserves sequential
first-improvement semantics; `batch_size=1` is the serial control. Material-gradient
`run` calls retain the differentiable PyTorch solve and do not use these no-grad
proposal kernels. Batch scratch estimates are bounded by `max_response_bytes`.

The public trial/accept path still includes validation and synchronization. CUDA
Graph capture is not enabled: dynamic public index discovery, error reporting and
trial ownership require host decisions. Fixed-rank kernel fusion and batching are
used without weakening those contracts.

For subsequent-candidate accuracy, accepted-state checks reserve a 16x margin
relative to the configured tolerance, and projected inverse actions are refreshed
every eight accepts. Full inverse reconstruction eliminates inactive identity
rows exactly and factors only active material/PEC rows. The all-active case still
requires a full local solve; dense response/inverse storage remains quadratic.
