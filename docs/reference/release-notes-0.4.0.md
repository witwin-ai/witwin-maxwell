# witwin-maxwell 0.4.0 — Release Notes

Previous release: `0.3.0` (tag `witwin-maxwell-v0.3.0`, 2026-07-13).

---

## 1. Summary

0.4.0 turns the FDTD runtime from a general-purpose full-wave solver into a solver
with first-class engineering workflows on top of it. The release adds RF port and
network engineering (wave ports, port sweeps, S-parameters, a rebuilt full-vector
mode operator, antenna far-field metrics), strongly coupled circuit co-simulation
and Touchstone network embedding, a subgrid thin-wire conductor model with finite
conductivity, non-reciprocal gyromagnetic ferrite and generalized surface-impedance
metals, phased-array basis / codebook / MIMO workflows, a dedicated electrostatic
(Laplace/Poisson) runtime with Maxwell capacitance-matrix extraction, SAR and
exposure analysis, and ESD excitation with deterministic dielectric breakdown. It
also extends differentiability well past the single-GPU adjoint: CPML-trainable
distributed adjoints over two transports, a per-rank NCCL reverse driver, array
scene-gradient VJPs, electrostatic implicit differentiation, and a wire-conductivity
adjoint. Everything remains reachable only through the stable `Scene -> Simulation
-> Result` contract, and every capability that is not fully supported fails closed
with a physics-worded error rather than returning a silently wrong number.

This release also changes numerical behavior users will notice — read section 3
before upgrading a pinned regression suite.

---

## 2. New capabilities

Capability labels below are the project's own honest grades and are carried
verbatim. Where a label says *not certified*, it means exactly that: no
standards body has assessed this software, and a standard's name appearing in an
API or profile identifier is a description of the modeled waveform or averaging
recipe, not a certification.

### 2.1 RF ports, modes, and networks

| Entry point | What it gives you |
|---|---|
| `mw.WavePort`, `mw.WaveModeSpec`, `mw.LumpedPort`, `mw.TerminalPort`, `mw.PortExcitation` | Declarative modal and lumped RF ports with frozen apertures, propagation direction, reference planes, and deterministic mode identities. |
| `mw.PortSweep`, `Result.port(name)`, `mw.PortData` | Deterministic one-run-per-drive N-port sweeps producing S-parameters with the declared port order preserved; live torch-native `PortData`. |
| `Result.antenna(surface=..., driven_port=...)` | Far field, directivity, gain, radiation efficiency and equivalent surface currents from a closed Huygens monitor plus one driven port. |
| `mw.NetworkData.cascade(other, port_map=...)`, `.terminate(port, gamma=... \| impedance=...)` | First-principles multiport star-connection and single-port termination, batched over frequency and differentiable through the connection algebra; operations are recorded in the network transform history. |
| `mw.NetworkBlock`, `mw.TouchstoneNetwork`, `mw.RationalModel`, `mw.StateSpaceNetwork` | Passive rational fitting and time-domain embedding of measured/synthesized N-ports into an FDTD run through named port terminals. |
| `mw.Circuit`, `Circuit.from_spice(...)`, `Scene.add_circuit(...)`, `Result.circuit(name)` | GPU-native linear MNA transient co-simulation strongly coupled to the Yee grid through a shared trapezoidal port interface, plus a standalone nonlinear transient runtime. |

**Mode-operator redesign.** The transverse full-vector mode eigensolver was rebuilt
on a genuinely Yee-staggered operator: each transverse electric component keeps its
own Yee location, the longitudinal fields are eliminated analytically, and metallic
walls are treated symmetrically by construction. The previous centered branch
suffered odd/even sublattice decoupling that capped the TE10 mode-shape correlation
at 0.51–0.59; the staggered operator returns a clean full-grid `sin(pi y/a)` profile
at correlation `>= 0.9999`. This path is used automatically by `WavePort` and
`ModeSource` for homogeneous non-magnetic apertures. It also gained interior-PEC
masking (for guided non-TEM modes inside conductors) and a companion quasi-static
electrostatic line-mode engine (`eps_eff = C/C0`) that serves TEM / quasi-TEM
transmission-line families — coax, microstrip, differential pair.

**Wave-level validated RF scene set.** Five scenes are validated at the wave level
(a physical propagation observable checked against an analytic or independent
reference, not a self-consistency identity): `coax_thru`, `rectangular_waveguide`,
`lumped_open_short_match`, `series_parallel_rlc`, `half_wave_dipole`. The
rectangular waveguide is additionally cross-checked against one authorized external
reference-solver run. Details and per-scene numbers:
`docs/reference/rf-wave-validation-2026-07-18.md`.

*Capability note:* RF differentiation is deliberately narrower than RF forward
execution — see section 4.

### 2.2 Subgrid thin wires, including finite conductivity

`mw.ThinWire`, `Scene.add_thin_wire(...)`, `mw.WireConductor.finite(conductivity,
permeability=...)`, `mw.WireMonitor`, `Result` → `mw.WireData`.

Straight, bent, branched, junctioned and closed-loop PEC centerlines run without
thin-cylinder voxelization, bind to standard `LumpedPort` / `TerminalPort`
declarations, and take part in checkpoint/replay. New in this release: a
finite-conductor material law with the exact analytic solid-round-wire skin-effect
series impedance, realized as a passive rational ADE companion in the current
recurrence (its PEC limit is byte-identical to the lossless leapfrog), a real
`ohmic_loss` channel (`0.5 Re(Z'(f)) * length * |I(f)|^2`, exactly zero on PEC
segments), and a closed-form conductivity adjoint of that dissipation channel.

*Capability note:* the analytic AC-resistance gate closes at 8%, fit-limited, not
the 2% originally targeted. The field-coupled `dI/dsigma` sensitivity and the
distributed lossy reverse path fail closed.

### 2.3 Materials

- **Arbitrary-bias gyromagnetic ferrite** — `mw.GyromagneticFerrite(...)`, with
  `from_cgs(...)` and `from_resonance(...)` constructors and a differentiable
  `permeability_tensor_at_freq(f)`. An arbitrary uniform bias direction and scenes
  mixing bias directions (mixed axes, opposed signs, differing magnitudes) run
  directly through `Scene -> Simulation -> Result`, on a per-cell rotation of the
  implicit-midpoint Cayley update. Discretely passive; the physics contract,
  derivation and frozen acceptance budget are in
  `docs/reference/ferrite-physics-contract.md`.
  *Capability note:* **forward-only.** The adjoint, the frequency-domain runtime,
  the distributed runtime and Bloch-periodic ferrite runs all fail closed.
- **All-orientation staircased surface-impedance metals** — `mw.LossyMetalMedium`,
  `mw.SurfaceImpedanceMedium`, `mw.RationalSurfaceImpedance`. A good conductor on
  any non-`Box` geometry (a `Cylinder`, a `Sphere`, mixed orientations in one scene)
  is staircased from its node occupancy and every exposed axis-aligned voxel face
  becomes a Leontovich surface. A flat plate assembled entirely from voxel faces
  reproduces the analytic Leontovich reflection to `<1%`.
  *Capability note:* the **staircase** is what shipped. True oblique / conformal
  (curvature-corrected) surfaces, rotated `Box` geometry and Bloch runs still fail
  closed, and staircased curved conductors carry a documented ~18% systematic in
  absorbed power against a resolved conductor.
- **Edge-native per-Yee-component material sampling** — see section 3.1. This is a
  numerical improvement that moves results.

### 2.4 Phased arrays

`Result.array_basis(...)` → `mw.ArrayBasisData`; `mw.BeamCodebook`,
`BeamCodebook.from_scan_angles(...)`, `mw.MultipathEnvironment`, `mw.MIMOData`.

New in this release, and the reason arrays become an optimization target rather
than a reporting tool:

- `ArrayBasisData.scene_gradient_vjp(columns=..., weights=..., parameters=...,
  objective=...)` aggregates per-column adjoints of the linear beam combine
  `E = sum_n w_n e_n` back onto **trainable scene parameters** (a design `Box` /
  `MaterialRegion` density) — not just onto the beam weights. The per-column seed is
  the exact complex-product backward `conj(w_n) * dL/dE`, summed in a
  caller-controlled deterministic reduction order.
- `mw.aggregate_scene_gradient_vjp(...)` and `mw.ensemble_scene_gradient_vjp(...,
  execution=mw.MultiGPUExecution.ensemble(devices=...))` distribute the per-column
  forwards over a device pool and reduce deterministically. 1-GPU vs 2-GPU
  aggregated gradients are **bitwise identical** on homogeneous GPUs.

*Capability note:* no throughput or scaling number is claimed for the ensemble
aggregation. The NCCL joint-solve adjoint is out of scope for this path
(independent `Simulation`s only).

### 2.5 Multi-GPU execution and distributed differentiability

| Entry point | Support |
|---|---|
| `mw.MultiGPUExecution.ensemble(devices=...)`, `mw.run_many(simulations, execution=...)` | N independent `Simulation`s over a device pool. Bitwise identical to serial. Measured 1.98–2.00x on 2 GPUs at 96³/160³. |
| `mw.FDTDParallelConfig`, `Simulation.fdtd(..., parallel=...)` | Domain-decomposed joint forward solve. **Grid-conditional**: measured 0.544x at 128³ (communication-bound) rising to 1.726x at 192³ — payoff only at large grids. |
| `Simulation.fdtd(..., parallel=...)` with a trainable `Box` `MaterialRegion` density, `transport="cuda_p2p"` | In-process distributed adjoint, now including the **CPML absorbing update** (psi-active). 1-vs-2-GPU parity ~5.94e-7. |
| `witwin.maxwell.fdtd.distributed.adjoint.run_nccl_distributed_reverse(simulation, objective=...)` under `torchrun --nproc-per-node=N`, `transport="nccl"` (driver-level entry, not a top-level `mw.*` export) | Per-rank collective end-to-end reverse driver: per-rank checkpoints, NCCL forward-replay halos, separable local objective seeds, transposed reverse halos, gathered `grad_eps` on rank 0. Point objectives and separable y/z-plane objectives, standard and x-CPML. Parity ~2e-7 vs single-GPU. |

Both distributed adjoint transports are **load-race-free**: parity holds at the same
tolerances while both boards are saturated by a co-tenant, gated by committed stress
tests with falsifications (see section 3.5 and section 5).

### 2.6 Electrostatics and capacitance extraction

`Simulation.electrostatic(scene, boundary=..., solver=...).run()` →
`Result(method="electrostatic")` → `result.electrostatic`;
`Simulation.capacitance(scene, terminals=..., reference=...).run()` →
`Result(method="capacitance")` → `result.capacitance`.

Supporting objects: `mw.ElectrostaticTerminal` (fixed potential, grounded, or
floating with prescribed charge), `mw.ChargeDensity`, `mw.ElectrostaticBoundarySpec`
(`grounded_box()`, `dirichlet(v)`, `neumann()`), `mw.ElectrostaticSolverConfig`
(float64 by default), `mw.TruncationEstimate`.

- A matrix-free cell-centred FVM `-div(eps grad phi)` operator with harmonic-mean
  face permittivity, solved by a float64 GPU Jacobi-PCG. It is an **independent DC
  PDE runtime**, not a low-frequency full-wave approximation.
- Floating conductors with a prescribed total charge are resolved by exact linear
  superposition (one base solve plus one unit solve per floating conductor).
- N-terminal Maxwell capacitance matrix with `reciprocity_error` and `row_sum_error`
  reported and **no silent symmetrization**, plus derived accessors
  (`mutual_capacitance`, `capacitance_to_reference`, `two_terminal_capacitance`).
- **Tensor permittivity**: a `Material(epsilon_tensor=mw.Tensor3x3(...))` compiles
  into a per-cell SPD tensor field and solves through both entry points. The
  operator adds a symmetric cross-derivative coupling derived as the gradient of a
  discrete quadratic energy, so symmetry, positive-definiteness, the
  `0.5 integral(E.D)` identity and discrete Gauss closure all still hold.
- **Open boundaries** are handled explicitly rather than approximated:
  `Simulation.capacitance(..., truncation_estimate=mw.TruncationEstimate(
  padding_cells=N))` runs one extra enlarged grounded-box solve and reports the
  finite-enclosure truncation error, including a 1/L Richardson extrapolation.
- **Differentiable** by implicit differentiation: the reduced solve is wrapped in a
  `torch.autograd.Function` whose backward solves the adjoint system on the same SPD
  operator. `energy`, `terminal_charge`, the fields and every `CapacitanceData.matrix`
  entry backpropagate to the compiled permittivity, free-charge and fixed-potential
  tensors.

*Capability note:* **experimental.** Tensor-permittivity solves are **forward-only**
(a trainable input under a tensor dielectric fails closed rather than detaching), and
floating-conductor superposition gradients are not implemented.

### 2.7 SAR and exposure analysis

`Material(mass_density=...)`, `Scene.compile_mass_density()`,
`Result.sar(monitor, averaging=..., normalization=...)` → `mw.SARResult`;
`mw.SARAveraging`, `mw.SARPeak`, `mw.PowerNormalization`,
`mw.combine_coherent_sar(...)`, `mw.combine_incoherent_sar(...)`,
`mw.IncidentPowerDensityMonitor`, `Result.incident_power_density(...)` →
`mw.IncidentPowerDensity`.

- Point SAR is a pure result-domain reduction over a `PowerLossMonitor`'s electric
  volumetric loss density, colocated from Yee edges to cell centers by a
  power-conserving half-weight scatter so the region volume integral closes exactly
  against the edge-integrated channel power. Reported per channel and total, in
  W/kg, with NaN (never zero-fill) where tissue fill is below the occupancy epsilon.
- Mass-averaged SAR at 1 g / 10 g under the versioned profile `"cubical-prefix-v1"`,
  computed in O(1) per candidate half-width from 3D inclusive prefix sums.
  `SARResult.peak(mass)` returns the peak with its actual enclosed mass and physical
  cube extent; `SARResult.soft_peak(temperature, mass=...)` is a differentiable
  surrogate for optimization.
- Power normalization (`PowerNormalization.source(...)`,
  `.accepted_power(port, watts)`), coherent and incoherent multi-source combination.
- Incident power density: `IncidentPowerDensityMonitor` carries the tangential `E/H`
  needed for the time-averaged normal Poynting component; plane-wave gates verify
  `|S| = |E|^2 / (2 eta)` exactly. An optional `spatial_average` area applies a
  versioned `spatial-average-v1` moving-window average.

*Capability labels, binding:* `"cubical-prefix-v1"` is a **versioned but not
certified** averaging profile. It is IEEE/IEC-inspired and its documented
differences from IEC 62704-1 are real: a symmetric index-space cube with no
cube-face expansion asymmetry, and no tissue-connectivity flood fill. The
`spatial-average-v1` exposure window is an engineering convenience and records
`certified: False` in its own provenance. **Nothing here is a certified standards
profile, and no third party has assessed it.** `soft_peak` is additionally labeled
non-regulatory.

### 2.8 ESD excitation, stress monitoring, and dielectric breakdown

| Entry point | Capability label |
|---|---|
| `mw.ESDWaveform.iec_61000_4_2(level_voltage, discharge="contact")`, `mw.MeasuredWaveform(...)`, `waveform.diagnostics()`, `waveform.resample_to_grid(dt)` | **stress-only** |
| `mw.ESDCurrentSource(name, port=..., waveform=...)`, `Result.esd_waveform(name)` | **stress-only** (ideal prescribed current injection) |
| `mw.ESDVoltageSource(name, port=..., waveform=..., discharge_resistance=330.0, storage_capacitance=150e-12)`, `.build_circuit(...)`, `Result.esd_generator(name)` | **stress-only**, circuit-driven through the standard source network |
| `mw.BreakdownMonitor(...)`, `Result.breakdown(name)` → `mw.BreakdownStressData`; `mw.ComponentRating`, `mw.ComponentStressMonitor`, `Result.component_stress(name)` | **stress-only, non-feedback** |
| `mw.DielectricBreakdown(...)` composed into `Material(breakdown=...)`; `Result.breakdown_data`, `Result.breakdown_events` | **deterministic-breakdown, uncalibrated** |
| `mw.SmoothBreakdownRisk(...)`, `mw.SmoothBreakdownRiskData` | **non-physical, non-regulatory differentiable surrogate** |
| `mw.ElectrostaticInitialCondition.from_result(dc_result)` → `Simulation.fdtd(..., initial_condition=...)`; `Result.electrostatic_prebias` | DC-seeded FDTD pre-bias |

- ESD waveforms are four-parameter two-term Heidler sums with numerically computed
  peak-normalization factors and a `standard_revision` recorded in provenance;
  resampling to the run grid is charge-conserving binned integration (per-bin mean
  current), not point sampling, so `charge_ratio` is 1 by construction.
- The circuit-driven path assembles the standard 330 ohm / 150 pF generator as a real
  `Circuit`, binds it to a named port, and runs the ordinary `Scene ->
  Simulation.fdtd -> Result` flow — the ESD source is not a special-cased side path.
  Element defaults are exposed as `witwin.maxwell.esd.ESD_STANDARD_DISCHARGE_RESISTANCE`
  / `ESD_STANDARD_STORAGE_CAPACITANCE`.
- Dynamic breakdown is a **deterministic field-duration latching state machine**: a
  cell flips from intact to conducting once `|E|` stays at or above `critical_field`
  for a contiguous `minimum_duration`, then ramps conductivity toward
  `post_breakdown_conductivity` through the standard semi-implicit lossy update. The
  typed event log is deterministic, ordered by `(step, cell_index)`, collected in a
  bounded preallocated GPU buffer, and a capacity overflow is a hard error rather
  than a silent drop.
- `SmoothBreakdownRisk` reads the same colocated `|E|` the physical stress
  accumulator reads, only softened through a sigmoid margin. Its class name,
  `capability_level` and provenance (`non_physical=True`, `non_regulatory=True`) all
  say so.

*Capability labels, binding:* "stress-only" means these objects reproduce standard
current waveforms, inject them, and report local field stress, port V/I, charge and
action integral. They do **not** model discharge-gun geometry, arc channels, or
device-failure probability. "uncalibrated" means the breakdown state machine has no
calibration against measured breakdown data. A standard's name in a constructor is
not certification.

---

## 3. Behavior changes users will notice

This section is deliberately blunt. If you pin absolute numbers from 0.3.0, read all
of it.

### 3.1 Edge-native material sampling replaced node-centered sampling — geometry-resolved results move

FDTD material coefficients (diagonal background `eps`/`mu` and the static
`sigma_e`/`sigma_m`) are now sampled **edge-native**: the polarized (Kottke) or
arithmetic subpixel blend is evaluated directly at each Yee component's own
staggered location, with the SDF occupancy, the interface normal and any
`MaterialRegion` density sampled there. Previously the blend was formed at grid
nodes and then arithmetically averaged onto Yee edges — applying the interface
operator at the wrong place and then interpolating it.

**Geometry-resolved scenes shift numerically.** Scored against the *same*
pre-existing external reference caches, the geometry cluster's median `field_l2`
improved from **0.2072 to 0.0836 (−59.6%)**, with **11 scenes improved and 0
regressed** (5 flat). Artifacts: `docs/assessments/f4-geometry-cluster-before.json`,
`-after.json`, `-delta.json`; acceptance record
`docs/assessments/f4-subpixel-lever-acceptance-2026-07-21.md`.

The move is toward the references, not away — but it is still a move. **If you pin
absolute field values, fluxes, S-parameters or resonances from a geometry-resolved
0.3.0 run, those baselines must be regenerated.**

Scope, so you know what did *not* change:

- The node-centered model is still produced as the canonical representation for
  summaries, monitors, the mode solver and the SAR / mass models. `Result.material(...)`
  and SAR mass models read the node grid.
- The edge-native path covers isotropic, axis-aligned diagonal-anisotropic and
  `PerturbationMedium` families, with conductivity / dispersion / nonlinearity /
  modulation layered on top. Full off-diagonal anisotropy, 2D sheets and
  surface-impedance metals stay on the node→edge path (unchanged capability scope).
- The material VJP follows the forward, so geometry, region-density and
  diagonal-anisotropy gradients stay consistent by construction.
- The benchmark harness default stays `pec="staircase"`. F4 briefly switched it to
  `pec="conformal"`; K1 reverted that after the conformal PEC edge fill was given
  compact support and its residual spurious absorption on cut edges was measured
  (see `docs/assessments/k1-conformal-pec-fix-2026-07-22.md`). Scenes opt into
  conformal per scene. Dielectric scenes are unaffected either way.

### 3.2 CUDA-graph stepping is now the public default

`Simulation.fdtd(..., cuda_graph=True)` is the default. The per-step field-update
core is captured into a CUDA graph, with the pre-existing graceful eager fallback
when capture is not possible. Pass `cuda_graph=False` to force eager stepping.

**What the default actually buys you.** Measured on the shipped tree (2× RTX A6000,
driver 595.71.05, plain vacuum dipole scene, CPML 8 layers, steady-state ms/step from
a two-point slope so graph capture is excluded, 7 paired ABBA rounds per point):

| grid | graph ms/step | eager ms/step | throughput gain | one-sided CI95 | A/A floor | peak memory |
|---|---:|---:|---:|---:|---:|---:|
| 48³  | 0.0792 | 0.1643 | **+106.8%** | +105.1 … +108.4% | 1.253% | +21.1% |
| 64³  | 0.1618 | 0.1663 | +2.70% | +2.67 … +2.74% | 0.688% | +20.9% |
| 96³  | 0.2774 | 0.2818 | +1.60% | +1.53 … +1.68% | 0.051% | +9.4% |
| 128³ | 0.5681 | 0.5735 | +0.94% | +0.89 … +0.99% | 0.036% | +9.0% |
| 160³ | 0.9699 | 0.9752 | +0.55% | +0.53 … +0.56% | 0.024% | +8.5% |
| 288³ | 4.5106 | 4.5159 | +0.12% | +0.10 … +0.13% | 0.009% | +8.1% |

Every point is above its A/A resolution floor, so the graph default is never slower
on this host — but the win is concentrated entirely at the small-grid end. The
mechanism is visible in the eager column: eager stepping bottoms out at a **CPU
launch-bound floor of ~0.165 ms/step** (48³ and 64³ cost eager the same despite a
2.4× cell-count difference). Below that floor the graph replay is the only way to go
faster, hence +107% at 48³; from 64³ up the step is GPU-bound and the gain decays
monotonically from +2.7% to +0.1%. Graph capture itself costs 2.2–16.6 ms once per
solve.

Peak allocated memory is **+8.1% to +21.1%** for the graph path, largest at the
smallest grids where the fixed graph-side buffers are not amortized.

Artifact and reproduction command:
`docs/assessments/cuda-graph-throughput-2026-07-22.json`; driver
`docs/assessments/cuda-graph-throughput-probes/cuda_graph_throughput.py`.

*This supersedes the "+29% at 96³" figure quoted from the commit that flipped the
default.* That number was never backed by a tracked measurement and does not
reproduce on the shipped tree: 96³ measures **+1.6%**. The most likely reason is the
0.4.0 kernel work itself — the uniform-coefficient scalar fast path and the
compressed-CPML interior kernels cut the eager path's per-step launch count, which
moved the launch-bound crossover down to roughly 64³ and left much less for the graph
to recover. "Neutral at 288³" does hold (+0.12%).

Three consequences:

1. **New device requirement.** Graph capture is now bound to the solver's device: the
   graph runner takes a mandatory device, runs warmup/capture/synchronize inside that
   device context, and uses an explicit per-device capture stream. This fixes a real
   silent-corruption failure mode — a solver whose tensors lived on `cuda:1` while the
   calling thread still had `cuda:0` current used to record an *empty* graph and
   silently stop integrating after warmup (40-step vacuum dipole: peak `|Ez|`
   `2.126512e+04` eager vs `7.333388e+04` graphed on `cuda:1`). If you construct
   solver internals directly, you must now supply the device.
2. **An empty capture is a hard error.** A capture that records no CUDA work raises
   instead of installing a no-op replay, so the caller's fallback degrades to correct
   eager stepping rather than freezing the fields.
3. **Concurrent ensemble plans step eagerly.** CUDA-graph capture is process-global,
   and PyTorch captures in `cudaStreamCaptureModeGlobal`, so an open capture aborts
   any synchronizing call in any other thread. Any executor plan that can run more
   than one task at a time (`mw.run_many`, the ensemble network sweep, the
   array-gradient plan) now suspends capture for the whole plan. Those tasks give up
   the small-grid graph speedup. Serial plans (`max_concurrency == 1`) keep it.
   Distributed (`parallel=...`) configs force graphs off as before.

Note that consequence 3 gives up a gain that is only material below ~64³ — see the
throughput table above.

Sources: `docs/assessments/j1-perf-regression-fixes-2026-07-22.md`,
`docs/assessments/cuda-graph-throughput-2026-07-22.json`.

### 3.3 `ETA_0` restored to the CODATA literal

During development, the vacuum constants were centralized into
`witwin/maxwell/constants.py` and the vacuum wave impedance was briefly redefined as
the derived product `MU_0 * C_0`. Because `MU_0` and `EPSILON_0` are themselves
twelve-significant-digit CODATA literals, that product lands at `376.7303136668…`
instead of the recommended `376.730313668` — a **−3.043e-12 relative** offset.

`ETA_0` is the CODATA 2018 recommended literal again, matching every call site's
pre-existing value. The repository convention is now explicit and pinned by test:
CODATA recommended literals per constant, never one constant derived from the
others.

**User impact:** absolute power, impedance and far-field results move by ~3e-12
relative *between the intermediate development commits*. Relative to the 0.3.0
release the shipped value is unchanged, so absolute results are continuous across
the release boundary. If you tracked an intermediate snapshot, expect that ~3e-12
shift in the soft `PlaneWave`/`GaussianBeam` unit-power scales, TEM and vector-mode
impedances, far-field constants and the default array wave impedance.

### 3.4 `Result.breakdown(name)` and `Result.breakdown_data` are different things

Both accessors are new in 0.4.0, and the names are close enough to be worth stating
outright:

- `Result.breakdown(name)` → `BreakdownStressData` — the **non-feedback stress
  monitor** reading of a named `BreakdownMonitor` (peak field, exceedance duration,
  per-cell maps). Listed by `Result.breakdown_names()`.
- `Result.breakdown_data` → `BreakdownResultData` — the **dynamic breakdown runtime**
  output for a scene containing `Material(breakdown=DielectricBreakdown(...))` (event
  log, final-state mask, dissipated energy). `Result.breakdown_events` exposes the
  event tuple directly.

The dynamic accessor was renamed to `breakdown_data` specifically so the two can
coexist. If you followed a pre-release snapshot that used `Result.breakdown` for the
dynamic runtime, update the call.

### 3.5 Distributed adjoint gradients are now correct under concurrent GPU load

Both distributed adjoint transports previously drifted at the partition seam when
the GPUs were shared with another workload, while the forward output stayed bitwise
clean — the worst possible failure shape, because nothing looked wrong.

- The NCCL path had a caching-allocator cross-stream reuse hazard: reverse/replay
  halos ran on a non-default stream while their per-step adjoint planes were
  allocated on the default stream. The halos now run on the current (default) stream
  so allocation-stream equals use-stream.
- The in-process `transport="cuda_p2p"` path had a **distinct** hazard — a
  checkpoint-capture happens-before race, not the allocator class. The mid-forward
  checkpoint cloned the persistent field storage on the device default stream while
  forward updates run on each shard's compute stream, so under load the next update
  tore the snapshot and the replayed seam gradient drifted to ~8.09e-2. The clone now
  runs on the shard's compute stream.

Both are fixed with committed stressed parity gates (which spawn their own co-tenant
load) and falsifications that revert the fix and show the drift return. Post-fix
parity is ~2e-7 under saturating load, at unchanged tolerances. If you ran
distributed gradients on shared GPUs in a pre-release snapshot, **re-run them**.
Sources: `docs/assessments/i1-p2p-race-acceptance-2026-07-21.md`,
`docs/assessments/h1-nccl-driver-acceptance-2026-07-21.md`.

### 3.6 Other user-visible changes

- **Spectral DFT time stagger.** Point / plane / flux / closed-surface running-DFT
  observers sample with the physical Yee time stagger: electric observers at the
  plain step phase, magnetic observers carrying the `dt/2` half-step colocation.
  Consumers reading complex field spectra must honour it.
- **NF2FF quadrature.** Near-to-far-field surface quadrature integrates each
  equivalent-current sample over its full primal Yee control volume, and the
  quadrature is clipped to the sampled Huygens face.
- **Unified trapezoidal port interface.** Every model that couples a lumped device,
  an MNA circuit or a fitted network to the Yee grid now drives the coupling from the
  same trapezoidal half-step `(V, I)` interface (pinned at `rtol ~ 2e-6`). Mixed
  conventions from earlier snapshots are gone.
- **Resumed delayed networks.** An embedded network with explicit port delay now
  checkpoints and resumes its reference-plane rings and fractional-delay filter
  memory. Previously a resumed delayed network silently restarted those from zero.
- **`GridSpec.uniform(dl)` semantics.** `dl` is a *maximum requested* step: each axis
  takes `ceil(span / dl)` cells and redistributes them uniformly as `span / count`,
  and the resolved node array is endpoint-inclusive.
- **`TimeConfig.auto` with periodic boundaries has a recorded nondeterminism trap.**
  Pin an explicit `TimeConfig` for reproducible periodic runs.
- **Frequency vocabulary.** Public APIs use `frequency=` for scalar selection and
  `frequencies=` for one-or-many targets. `freqs` is not a public spelling.

---

## 4. Known limitations and unsupported configurations

The authoritative machine-checked statement of where support ends is the set of
**176 fail-closed capability guards** inventoried in
`docs/reference/fdtd-capability-guard-census.md` and enforced by
`tests/api/public/test_guard_census.py`. Every `NotImplementedError` capability path
is counted there, and an unlisted new guard fails the census test. The list below is
a human-readable index over that guard set; the guards, not this list, are binding.

### 4.1 FDFD — unsupported in this release

State this plainly:

- The frequency-domain sparse solver is **present in the package but user-deferred**.
  It is not part of this release's supported surface, it is not graded, and it should
  not be treated as production.
- **16 tests in the FDFD family fail in this tree.** The failures are environment
  dependent: the observed error is `GPU solve failed: No module named 'nvmath'` — the
  optional direct-solver dependency declared as the `direct` / `direct-cu13` extras in
  `pyproject.toml` is not installed here. These 16 failures are expected and are not
  a regression introduced by this release.
- What *is* supported for FDFD: nothing is claimed. What is *not*: nonuniform
  (`GridSpec.custom` / `auto`) grids, in-domain PEC materials and conformal PEC,
  domain symmetry, magnetic media and magnetic dispersion (electric anisotropy only),
  per-face boundary mixing beyond `none`/`pml`, ferrite, surface-impedance media,
  nonlinear media, and breakdown — each fails closed explicitly.

If you need frequency-domain results, use the FDTD runtime with spectral monitors.

### 4.2 Differentiability boundaries

Supported: single-GPU FDTD adjoint (material density, fixed-stencil geometry,
CPML psi-active); distributed CPML-trainable adjoint over `cuda_p2p`; NCCL per-rank
reverse for point and separable y/z-plane objectives; array scene-gradient VJP;
electrostatic implicit differentiation for scalar/diagonal permittivity; network and
circuit coefficient gradients; wire conductivity adjoint on the dissipation channel;
`SmoothBreakdownRisk` (a surrogate, not physics).

Fail-closed: hard dielectric breakdown (non-differentiable trigger); explicit-delay
network adjoint; floating-conductor superposition gradients; trainable tensor
permittivity; `WavePort` network embedding (no scalar `(V,I)` terminal contract);
field-coupled wire `dI/dsigma` and distributed lossy reverse; ferrite adjoint; SIBC
adjoint; trainable `RationalModel` poles / state-space `A,B,C,D`; flux / mode /
x-normal / finite-plane NCCL adjoint objectives; distributed trainable density beyond
CPML/stable-PML, ports, and non-`Box` density regions.

RF differentiation is narrower than RF forward execution: differentiable lumped runs
reject `ParallelRLC`, observer-only contour ports, open internal resistance, trainable
source/reference impedance, lumped `PortSweep`, and conductive / dispersive /
nonlinear / modulated / full-anisotropy / Bloch coupling. Differentiable `WavePort`
runs require one fixed mode and fixed amplitude and cannot mix with lumped ports or
standalone R/C/L elements.

### 4.3 Multi-GPU boundaries

- Ensemble execution: supported, bitwise identical to serial.
- Joint-solve forward: supported, but **grid-conditional** — measured 0.544x at 128³
  (communication-bound) and 1.726x at 192³. Do not expect a win on small grids.
  Composing ensemble with joint solve is rejected.
- Monitor gather: **forward path only**, under an owned-exclusive seam-ownership
  rule. Collective per-monitor gather beyond the forward path is not built.
- Coupled-runtime joint solve (circuit / network / thin-wire under multi-GPU):
  **not supported**, fails closed at prepare.
- Thin wires run distributed **forward only**. A trainable wire under multi-GPU, a
  distributed CPML or Mur absorbing boundary on a wire scene, and a wire mixed with
  an embedded network or a lumped circuit all fail closed at prepare. The distributed
  path also rejects any ferrite and any surface-impedance medium outright.
- No timing number is published for the NCCL reverse driver — it was qualified for
  correctness on shared GPUs, and an exclusive measurement window is still pending.

### 4.4 Physics and material boundaries

- **Ferrite**: forward-only, single-device, non-Bloch. No adjoint, no frequency-domain
  runtime, no distributed runtime, no `PerturbationMedium` over ferrite.
- **Surface impedance / SIBC**: staircased good conductors only. True oblique /
  conformal surfaces, rational (broadband) models on curved conductors, rotated
  `Box` geometry and Bloch runs fail closed. Staircased curved conductors carry a
  documented ~18% absorbed-power systematic.
- **Nonlinear circuit devices**: Phase-0 contract plus a **standalone** transient
  runtime only. FDTD field-path coupling, the transient companion into the Yee
  update, the adjoint, and BJT/MOSFET models all fail closed.
- **Anisotropy**: `Material.orientation` is unsupported. Full off-diagonal
  `Tensor3x3` permittivity is FDTD-only and cannot combine with Bloch, Kerr, or
  polarized subpixel averaging.
- **Breakdown / ESD composition**: dispersive, ferrite and SIBC media composed with
  breakdown fail closed. ESD phases beyond stress (surface/random/thermal feedback,
  calibrated gun/system workflows) are excluded, as is the conductive-media
  breakdown-feedback port.
- **SAR**: certified phantom profiles and an external-reference cross-check are
  deferred; the `antenna_near_phantom` benchmark scene is blocked upstream because
  the port machinery fails closed on a conductive host medium; VOP and multi-GPU SAR
  are not implemented; `PowerNormalization.input_power` fails closed because this
  build exposes no total injected source-power diagnostic.
- **Electrostatics**: grid-extending scene boundaries (PML/periodic), PEC-material
  dielectrics (use a terminal), dispersive/complex permittivity, gauge-singular
  pure-Neumann problems with no conductor, incompatible floating-charge constraints,
  and capacitance requests with no charge return path are all rejected.

### 4.5 Known accuracy gaps that are recorded rather than hidden

- Microstrip / differential-pair absolute `eps_eff` is ~24% low at `dx = 5 mm`; the
  quasi-static engine itself converges to the Hammerstad–Jensen closed form, so this
  is a resolution gap, not an engine defect.
- The probe-fed patch antenna does not reach broadside `TM010` with `D >= 5 dBi` at
  feasible resolution; that gate is a deliberate strict xfail (feed reactance plus a
  small finite ground), not a silently relaxed tolerance.
- The differential-pair four-port records max singular value ~1.18 against the shared
  1.10 passivity precedent — recorded as a fail, not forced to pass.
- The lossy-wire analytic AC-resistance gate closes at 8%, fit-limited.

---

## 5. Validation status

**What the evidence is.** The overwhelming majority of this release is validated
against analytic solutions, golden references, convergence studies, and conservation
or reciprocity identities — enforced as committed pytest gates that run on every
change. A small number of scenes are additionally cross-checked against an **external
reference solver**. Nothing in this release carries third-party certification of any
kind, and no standards body has assessed any profile shipped here.

**Gate vocabulary.** Every headline gate self-labels with the taxonomy in
`docs/reference/gate-classification.md`: `analytic-identity`, `tautology`,
`symmetric`, `postprocess-only`, `wave-level`, plus the `perf` label family. A
`wave-level` gate checks a physical propagation observable against an independent
reference; a `tautology` or `symmetric` gate does not, and is labeled so you can tell
them apart. This distinction is enforced in the records, not left to the reader.

**Validation tiers in this release.**

| Tier | What it covers |
|---|---|
| Cross-checked against an external reference solver | `rf/rectangular_waveguide` (propagation constant vs one authorized reference run, median 1.21% / max 2.74% against the analytic TE10 dispersion); the geometry benchmark cluster re-scored against the identical pre-existing reference caches for the edge-native sampling change (median `field_l2` −59.6%). |
| Analytic / golden only (`wave-level` head, no external reference) | `coax_thru`; the real NF2FF dipole; SAR layered-slab power-conservation closure; electrostatic MMS / analytic capacitance; ESD and breakdown energy closure; SIBC `alpha_c` against the textbook attenuation constant; the network raw-sample S-cascade cross-check. |
| Consistency class (annotated, never promoted) | Circuit reactive energy accounting; memoryless-network power balance. |

**Capability boundary.** The single authoritative statement of what is supported, at
what evidence grade, and where every boundary lies, is the section *"Capability
boundary (stable-release reference, 2026-07-21)"* in
`docs/plans/next-functional-2026-07/00-status-and-gaps-2026-07-19.md`, including its
2026-07-22 update. Every row there cites a tracked artifact. That section, plus the
176-guard census, supersedes any summary in this document if they ever disagree.

**Benchmark results.** `benchmark/RESULTS.md` is the generated comparison summary
(`python -m benchmark`). Regenerated summary for this release:

- Regenerated 2026-07-22 against the shipped code, cached external references only
  (all 102 FDTD scenarios validated their reference cache strictly; no cloud run, no
  trust-hook override).
- **35 scenarios improved, 11 regressed, 56 unchanged** on `field_l2`. Median
  `field_l2` 9.892e-02 -> 8.041e-02 (-18.7%); median `field_corr` 0.9959 -> 0.9972;
  scenarios meeting the `< 1e-1` target 52 -> 61. All 102 also got faster (median
  ms/step -6.92%, none slower).
- The improvement is dominated by the edge-native material sampling of section 3.1:
  the 16 `grid_geometry` scenarios reproduce their recorded post-change values exactly
  (cluster median `field_l2` 0.083642).
- Of the 11 regressions, 9 are <= 4.5% and cross no target. The two PEC scenarios that
  regressed materially in an intermediate build (`pec_box`, `rcs_pec_sphere`) were
  traced to a conformal-PEC defect, fixed, and now match their pre-change values
  exactly (section 3.7); no PEC scenario ships regressed.
- The generated table now carries a `Registered scenarios with no measured row`
  section, so a family that produced no measurement (FDFD here) is disclosed as
  absence of evidence rather than silently omitted.

**Test battery.** Full-battery counts for the release tag:

- `python -m pytest tests` on the release tree: **`16 failed, 3098 passed, 65 skipped, 3 xfailed, 1 xpassed` (2026-07-22, 2x RTX A6000, CUDA 13 / torch 2.13)**
- Every failure is in the deferred FDFD family; no other suite fails.

Note when reading the battery: the FDFD family contributes 16 expected failures from
the missing optional `nvmath` dependency (section 4.1). They are not regressions and
must not be triaged away by installing that dependency.

**A note on process.** No phase of this program is marked `completed` in the
project's own vocabulary — that bar requires both a non-author review and an external
reference cross-check, and it is unmet across the board. Every delivery was
adversarially audited internally. Where a gate failed, it is recorded as failing.
