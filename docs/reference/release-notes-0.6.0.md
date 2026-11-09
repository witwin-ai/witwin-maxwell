# WiTwin Maxwell 0.6.0

Previous published release: 0.5.1 (2026-09-08). Release date: 2026-11-08.

The [feature inventory](https://github.com/witwin-ai/witwin-maxwell/blob/witwin-maxwell-v0.6.0/FEATURE_LIST.md) remains the authoritative capability list, and the [changelog](https://github.com/witwin-ai/witwin-maxwell/blob/witwin-maxwell-v0.6.0/CHANGELOG.md) has the short form of these notes.

0.6.0 changes several numerical defaults. Results computed with 0.5.1, cached spectra and checkpoints should be recomputed for any scene that uses subpixel material sampling, additive electric current sources, pulsed multi-frequency spectra, pulsed TFSF, soft plane waves, complex fixed-k Bloch, FDFD PML or FDFD dipoles, or structures that touch a PML interface. Raw phasors also rotate by one known phase factor (physical Yee time labels, below).

FDTD scenes now run on the CUDA core path by default when they are eligible. On the release census this changes no result bit, but it does change execution, warnings and result metadata (see the first section below).

The package requires `witwin>=0.4.6,<0.5`, `torch>=2.10` and Python `>=3.10,<3.15`.

## Breaking changes

- **The surface-integral solver is renamed from SIE to MoM.** The old names are removed; there are no deprecated aliases.

  | 0.5.1 | 0.6.0 |
  |---|---|
  | `Simulation.sie(...)` | `Simulation.mom(...)` |
  | `SimulationMethod.SIE` (value `"sie"`) | `SimulationMethod.MOM` (value `"mom"`) |
  | `SIEConfig` | `MoMConfig` |
  | `SIEPeriodicConfig` | `MoMPeriodicConfig` |
  | `SIEDyadicPeriodicConfig` | `MoMDyadicPeriodicConfig` |
  | `witwin.maxwell.sie.*` (including `SIERuntime`) | `witwin.maxwell.mom.*` (`MoMRuntime`) |
  | `Result.method == "sie"` | `Result.method == "mom"` |
  | native op namespace `witwin_maxwell_sie_layered` | `witwin_maxwell_mom_layered` |
  | `WITWIN_MAXWELL_SIE_LAYERED_BUILD_DIR` | `WITWIN_MAXWELL_MOM_LAYERED_BUILD_DIR` |
  | periodic payload ids `sie_bravais_ewald_config_v1`, `sie_bravais_dyadic_config_v1` | `mom_bravais_ewald_config_v1`, `mom_bravais_dyadic_config_v1`; periodic results saved under the old ids do not restore |

  - Result snapshots and metadata written with the old method name, layered ABI tag or other renamed schema strings are not restored. The surface-integral formulation identifiers (`pmchwt_scalar_sie`, `anisotropic_dgf_sie`, `coupled_sie_vie_*`) are unchanged.
  - Downstream consumers importing `witwin.maxwell.sie` must move to `witwin.maxwell.mom`.

## Default and behavior changes

### FDTD execution defaults to the core path (`"auto"`)

`Simulation.fdtd(...)` and `FDTDConfig` default to `fdtd_core_path="auto"` and `accelerated_setup="auto"`; in 0.5.1 both were off.

- **What runs where.** Eligible scenes run on the fused CUDA core path, with CUDA material and coefficient setup. These are single-GPU, float32, forward and initializing runs. Accelerated setup is dispatched piece by piece; unsupported pieces keep the PyTorch setup.
- **Fallback.** These runs keep the existing stepping:
  - scenes with an admission blocker (see Known limitations);
  - resume and non-initializing preparation;
  - material-cache runs.

  The first automatic fallback in a `Simulation` warns once with the public `FDTDCoreFallbackWarning`. If core activation or whole-step graph capture fails under `"auto"`, the run is rebuilt with a fresh existing-path solver.
- **Reporting.** `Result.stats` and `Result.metadata` report `execution_path` (`"core"` or `"existing"`) and `fallback_reason` (`None` on the core path).
- **Explicit values.**
  - `fdtd_core_path=True` stays fail-closed: it raises for an ineligible scene and for resume.
  - `False` keeps the existing path and does not warn.
  - `accelerated_setup=True` or `False` controls setup independently.
- **PMC and closed-transverse CPML.** PMC boundaries, including PMC planes produced by `Scene.symmetry`, keep the existing path. So does closed-transverse CPML (explicit `absorber="cpml"` or stable PML), where the passivity guard runs. The CPML case is refused before preparation, so the scene is built once.
- **Result equivalence.** The release census covers 121 benchmark and data-generation scenes, 96 of which take the core path. Every FDTD scene's monitors are bitwise equal to the same run with both controls `False`.
- **Architecture.** Validated on A6000 (sm_86) only. Other architectures run the same kernels but are not qualified, and may differ at rounding level.
- **Extensions.** The core and setup CUDA extensions are compiled on first use, so the first eligible run pays a one-time build. Without a CUDA toolkit (no `nvcc` or `CUDA_HOME`), the build attempts fail in a few milliseconds, and `"auto"` runs complete on the existing path with one `FDTDCoreFallbackWarning`.
- **Migration.**
  - To reproduce the 0.5.1 execution path, pass `fdtd_core_path=False, accelerated_setup=False`.
  - To silence or escalate the fallback warning, use `warnings.filterwarnings(..., category=mw.FDTDCoreFallbackWarning)`.
  - Code that inspects `prepared.solver` internals should check `execution_path` first, because core-path solvers record different graph and runtime keys.

### Structures continued through PML interfaces

At preparation (FDTD and FDFD), a structure that reaches a PML interface and is prismatic there is continued through the PML along the face normal.

- **How.** Axis-aligned boxes are enlarged. Other prismatic geometry uses a normal-clamped signed distance. `MaterialRegion` densities are edge-replicated, and their gradient pullback transposes that sampling.
- **Not modified.** The public `Scene` is not modified, and structure order is preserved. The records appear as `pml_extrusions` in the prepared metadata and solver statistics.
- **Not extruded.** These warn instead:
  - mesh volumes;
  - non-prismatic contacts;
  - surface-impedance conductors (`LossyMetalMedium`, `SurfaceImpedanceMedium`, `CopperFoil`).

  Zero-volume sheets are only recorded.
- **Surface ownership.** A PEC structure and a surface-impedance conductor declared on the same face still fail closed. The ownership check compares the declared geometry, not the extruded copies.
- **Effect.** Results move only for scenes with such contacts; all other scenes are bitwise unchanged. Nonlinear structures flush with the PML now behave as the infinite structures they model. For example, a Kerr slab's nonlinear phase shift went from 2.63e-3, inflated by a truncated-edge hot spot, to 8.56e-4. That is within 1.4% of the |E|²-scaled infinite-slab prediction. In the benchmark suite, 10 rows moved and none got worse. Field L2 against the reference went from 0.569 to 0.041 for `dielectric_slab`, from 0.315 to 0.025 for `anisotropic_slab` and from 0.190 to 0.029 for `debye_slab`.

### FDTD phasors use physical Yee time labels

Post-step E is labelled (n+1)Δt and H (n+1/2)Δt, with the existing magnetic half-step stagger. Source references keep their provider compensation.

- **What rotates.** Compared with 0.5.1:
  - raw field and port phasors rotate by exp(iωΔt);
  - source-normalised additive-current and direct-drive phasors rotate by exp(iωΔt/2);
  - uncompensated magnetic, sheet and analytic-pulse phasors rotate by exp(iωΔt);
  - retained auxiliary-line and CW references are unchanged to rounding.
- **Unchanged.** Same-frequency S/Z/Γ, flux, matched field/port powers and directivity.
- **Field-time monitors.** The returned `t` advances by one Δt; the values are unchanged.
- **Checkpoints.** Resume checkpoints use format 6 and record the origin and provider reference. Checkpoints with the earlier spectral convention need a fresh run.

### Geometry smoothing at micrometre scale (witwin-core 0.4.6)

Earlier witwin-core releases clamped the geometry smoothing width to 1.19e-7 m, whatever the cell size. Micrometre-scale geometries were therefore over-smoothed by about 3 cells.

- Optimisation results and forward fluxes change materially for µm-scale scenes. In the bend example, the initial flux rises by 37.7% and the gradient cosine against the old result is 0.85.
- Scenes whose smallest cell is at least about 2.4 µm are unchanged bit for bit.
- Point-like occupancy queries without coordinate spacing must now pass an explicit `beta` in metres; otherwise they raise `ValueError`.
- witwin-core 0.4.6 is published.

### Subpixel material sampling defaults to `kottke_edge`

The default is now `SubpixelSpec(samples=4, averaging="kottke_edge", pec="staircase")`; 0.5.1 sampled one arithmetic point per Yee component.

- Interfaces are resolved with an edge-wise Kottke average, which converges at second order on smooth interfaces.
- Gradients fail closed. Trainable geometry requires `averaging="kottke_smooth"` (experimental) or `"polarized"`. Trainable `MaterialRegion` densities require `samples=1, averaging="arithmetic"`.
- Material preparation takes longer on curved bodies. The opt-in accelerated setup (see Performance) is the intended remedy.
- To reproduce 0.5.1 sampling, pass `SubpixelSpec(samples=1, averaging="arithmetic", pec="staircase")`.

Evidence: [plan](https://github.com/witwin-ai/witwin-maxwell/blob/witwin-maxwell-v0.6.0/docs/plans/subpixel-default-kottke-edge.md).

### Additive electric currents are injected at (n+1/2)Δt

`PointDipole`, `UniformCurrentSource` and electric `CustomCurrentSource` currents enter the electric update at the half step. Each source is normalised at its own injection instant.

- Raw port voltages and currents, impedances and reflection coefficients are unchanged.
- Source-normalised spectra carry the half-step phase exp(-iωΔt/2).
- Checkpoints that contain additive electric currents and were written before this change are refused on resume. Other checkpoints resume.

### Pulsed spectra are scaled per reference period

Raw pulsed spectra are computed as a Fourier integral of the executed record, with no planned-horizon scaling. This applies whenever source normalisation is off.

- **Per-period scale, including single-frequency runs.** Raw pulsed absolute spectra are scaled per reference period instead of averaged over the record length.
  - This holds for single-frequency runs too. For a single-frequency `GaussianPulse(f, fwidth=f)` run, every absolute power term moves by one uniform factor.
- **Per output frequency.** With `reference_frequency=None`, the reference period is 1/f for each output frequency. Raw multi-frequency amplitudes therefore also move by f/f_c, and powers by its square, away from the centre frequency.
- **Unchanged:** source-normalised results and all ratios, including S, Z, Γ, transmission and power closure.
- **Migration.** Absolute pulsed powers computed with 0.5.x differ by (f·T_record)²-type factors; for example, a 6 ns record at 1 GHz differs by 36×. Compare ratios, or recompute.
- **Commits.** The change landed in two steps:
  - `c971ac8a` changed the record-length average to a per-period scale, affecting single-frequency runs too.
  - `16139614` made the period per output frequency, which affects multi-frequency runs only.
- Runs that mix ongoing and transient drives without source normalisation refuse absolute spectral reads; ratios remain available.
- To keep the fixed-period convention, pass an explicit `reference_frequency`.

Evidence: [migration plan](https://github.com/witwin-ai/witwin-maxwell/blob/witwin-maxwell-v0.6.0/docs/plans/fdtd-pulse-dft-normalisation-migration.md).

### Other FDTD default changes

- **`full_field_dft`:** an explicit `full_field_dft=False` is honoured for multi-frequency runs. Volume-field reads and automatic electric power-loss channels then raise; pass `None` or `True`, or give explicit fields to `Result.power_loss`.
- **Pulsed TFSF:** the incident wave is referenced at the physical upstream entry plane, so its absolute phase no longer depends on absorbing-layer padding. Older pulsed-TFSF checkpoints are refused on resume.
- **Raw TFSF timing:** the auxiliary-line entry field uses the waveform at (n+1)Δt and analytic/discrete CW faces keep their physical J/M instants, so raw incident timing is exact on the Yee clock; normalisation references follow each provider. Finite-record source-normalised values shift by up to about 1e-3 for pulses and about 1e-2 for real CW on short records. TFSF checkpoints written before this change are refused on resume.
- **Soft equivalent-sheet plane waves:** they launch +E_inc. Absolute fields of soft plane-wave scenes change sign relative to 0.5.1, which was wrong. Evidence: [soft-sheet sign note](https://github.com/witwin-ai/witwin-maxwell/blob/witwin-maxwell-v0.6.0/docs/memory/soft-sheet-field-sign-2026-11-02.md).
- **Complex fixed-k Bloch CPML:**
  - These runs use a complex-frequency-shifted CPML, so that bound and evanescent modes do not grow in the absorber.
  - `cpml_config={"alpha_max": 0.0}` pins the legacy coefficients. This is not passive and is only for short runs.
  - Real-field, k=0 and non-Bloch runs are unchanged.
  - See the [CPML reference](https://github.com/witwin-ai/witwin-maxwell/blob/witwin-maxwell-v0.6.0/docs/reference/fdtd-cpml.md).
- **Fixed-k Bloch pulsed sources:** fixed-k Bloch runs accept pulsed TFSF-slab plane waves (see New features). They still refuse pulsed beams, mode and custom-field sources and soft plane waves; use one CW source per frequency for those. Point and current sources may be pulsed.
- **TFSF auxiliary-line termination:** the pulsed 1D incident line keeps its whole sampled extent lossless and ends in a 320-cell quadratic absorber appended beyond it. Previously its last 20 sampled cells formed a graded ramp, which reflected near-band-edge content and trapped a slowly decaying mode. The line now matches a reflection-free lossless reference to within max(1e-6·peak, ε32·√N) over the full record, and its in-band reflection is no larger than before. In-band benchmark results agree with the old termination to 3–4 digits. Checkpoints with an auxiliary TFSF line written before this change are refused on resume.
- **CPML grading:** magnetic grading is sampled at physical half points on both boundary sides, including on graded grids.
- **CopperFoil:**
  - Finite-thickness free-edge impedance is on by default; `edge_model="none"` restores the bare rim.
  - Diffusion uses compressed passive paired-pole fits. In band, these agree with the full model to about 1e-4.
- **Material and source discretisation:**
  - Staircase PEC shorts an edge only between conductor nodes.
  - Multi-sample averaging is second order on resolved Yee spacing.
  - Narrow Gaussian dipoles inject their full unit moment.
  - Periodic dipole images fold onto unique Yee samples.
  - Closed-surface monitors snap watertight to tangential-E nodes.
  - Bulk conduction loss uses the native Yee-edge conductivity.
- **`TwoPhotonAbsorption`:** the conductivity factor is 2/3, the linear-polarization cycle average. It was 4/3.

### FDFD

- The PML absorbs. It is the frequency-domain form of the FDTD CPML, so one `BoundarySpec.pml(...)` means the same physical layer in both solvers.
- `PointDipole` uses the FDTD unit control-volume current moment (A·m), and CW results are peak phasors under exp(-iωt).
- Material components sit on Yee edges, the loss sign matches FDTD, and dense coarse solves use `torch.linalg`.
- FDFD refuses scene monitors it cannot populate, and solver exceptions propagate and clear stale results.

### Rejected inputs

- Trainable fixed controls are rejected before scalar conversion. These are frequencies, time, grid and sampling controls, monitor geometry, and topology.
- Port probe gaps fully covered by PEC are rejected.
- Dynamic dielectric-breakdown feedback with coupled RF ports is rejected.

### Native extension

The CUDA extension ABI is `stable_abi_v13`. Prebuilt binaries for an older ABI are rejected and rebuilt on first use. With `WITWIN_MAXWELL_FDTD_CUDA_PREBUILT=1`, a matching ABI marker is now required; run one normal build first. The core-path extension (`core_abi_v11`) and the accelerated-setup extension (`setup_abi_v6`) are separate and compiled on first use.

### Absorber default and passivity guard

- `Simulation.fdtd(absorber="auto")` is the default. Real-field cells whose transverse walls are closed (periodic, PEC or PMC) and that have PML on the normal axis select the graded `absorber`. It needs at least 40 layers on every PML face; thinner layers raise a `ValueError` that names the closed transverse axes, the PML faces with too few layers and both remedies. Open cells and complex fixed-k cells keep CPML. Geometry is never changed.
- Explicit `absorber="cpml"` (or stable PML) in a real-field closed-transverse cell runs under a fail-closed passivity guard. It fits the late energy envelope and raises `FDTDPassivityError` when growth exceeds 4e-7 per step. Results record `absorber`, `absorber_resolution_reason` and `passivity_guard`.
- In complex fixed-k (oblique Bloch) cells the same check only warns: it emits one `RuntimeWarning` per run with the measured slope, standard error and window, records them in the `passivity_guard` and `passivity_guard_warning` metadata, and lets the run finish. Complex fixed-k CPML has known growth (below), so a fail-closed guard would stop otherwise valid long records. Non-finite energy fails closed in every topology.
- Closed-transverse CPML stays on the existing execution path, where the guard runs: `"auto"` keeps it there, and `fdtd_core_path=True` refuses it.
- FP64 native forward resolves `"auto"` the way preparation does. An `"auto"` that resolves to CPML runs. An `"auto"` that resolves to the graded absorber, and any explicit non-CPML absorber, keep the `FP64 native forward requires the CPML absorber.` refusal.
- **Migration.** Scenes that relied on the CPML default in closed-transverse real-field cells have two remedies. Pass `absorber="cpml"`: the numerics are unchanged and the run is now guarded. Or configure at least 40 PML layers: the results then come from the graded absorber. The repository's own tests and benchmark rows that verify CPML physics pin `absorber="cpml"`; tests that verify default behaviour use 40 layers.

## Physics fixes

- **Soft equivalent-sheet plane-wave sign.** See above.
- **Soft launches at transverse walls.** Soft plane-wave and beam launch currents are no longer written into inactive electric wall rows (transverse none/PML/PEC walls); this removes a linear late-time growth of the corner H field.
- **Scale-invariant power and normalisation checks.**
  - The `ModeSource` zero-power check compared the integrated power in watts with machine epsilon. A unit-amplitude mode on a micrometre aperture carries about 1e-16 W, so every micrometre-scale `ModeSource`/`ModePort` was refused at prepare. The check now refuses only power that cancels relative to the same Yee-quadrature integral of |E_t||H_t|.
  - The same class of absolute floors on dimensional quantities is now relative to a same-unit reference: modal power fractions; scattering, diffraction, antenna and near-to-far normalisation; network, port and terminal-wave impedance normalisation; observers; TFSF power; and the coupled-circuit DC port-current admission.
  - Roundoff pruning of source components uses float64 epsilon relative to the owning vector or profile family. User current and field datasets are pruned only at exact zero.
  - Valid results at ordinary scales are unchanged bitwise.
- **Mode-source launch-plane gradients.**
  - The defect: the adjoint launched the full-vector mode but differentiated a scalar mode with the opposite magnetic sign. The replay also rebuilt the 1/ε injection coefficients without a gradient graph. The plane-material gradient was wrong by up to about 7.6×: on a reference guide, the adjoint gave −18334 against a central finite difference of −2417 on that tree.
  - The fix: when a launch plane's material is trainable, lossless and PEC-free, the forward launch and the adjoint replay share one torch dense vector eigen-solve (complex128, explicit phase gauge, selected-eigenpair adjoint). The two profiles are bitwise identical. The same guide now gives −2418 against FD −2425 at metre scale (the two trees' FD references, −2417 and −2425, agree within the FD error bound of about ±30), and 1037.8 against 1039.1 at micrometre scale.
  - Limits: the dense operator is capped at n ≤ 4096, which is about 32×32 interior plane nodes. Above-cap, near-degenerate (relative eigen-gap ≤ √ε64·‖A‖∞/|λ|) and lossy or conductive trainable planes refuse with a named error. Non-trainable planes keep the existing solver, bitwise unchanged.
  - Resume: the launch-profile phase gauge restarts on resume. A run resumed mid-optimisation can flip the launched profile's global sign relative to an uninterrupted run. That doesn't matter for |S|²-type objectives. For phase-sensitive objectives, restart from the checkpointed design instead of resuming mid-run.
- **Conformal thin-sheet rims.** Faces split only when both chord endpoints are inside the outline. This removes a 4-7% low resonance bias and gives first-order convergence.
- **Custom J/M sources.** They deposit over the Yee control-cell overlap with the declared dataset volume; the compiled moment was previously more than twice too large. Evidence: [plan](https://github.com/witwin-ai/witwin-maxwell/blob/witwin-maxwell-v0.6.0/docs/plans/custom-current-source-support.md).
- **Custom-field sources.** They deposit equivalent currents on the bracketing Yee sheets with fractional aperture overlap. This removes an over-coverage gain and a spurious backward wave.
- **Time modulation.** It uses a device integer step counter with FP64 phase, which removes long-run clock drift, and an effective-medium partial-cell coefficient. Evidence: [plan](https://github.com/witwin-ai/witwin-maxwell/blob/witwin-maxwell-v0.6.0/docs/plans/modulation-partial-cell-coefficient.md).
- **Fixed-k Bloch:**
  - Floquet TE/TM readout with power-normalised reflection and transmission.
  - Complex CW conventions consistent with the adjoint.
  - Periodic TFSF faces deposited once.
  - PEC clamping of imaginary fields.

  Evidence: [broadband injection note](https://github.com/witwin-ai/witwin-maxwell/blob/witwin-maxwell-v0.6.0/docs/memory/bloch-broadband-tfsf-injection-2026-10-31.md).
- **Thin wires.** Finite thin wires are stabilised with a passive RL impedance. Evidence: [note](https://github.com/witwin-ai/witwin-maxwell/blob/witwin-maxwell-v0.6.0/docs/memory/thinwire-finite-instability.md).
- **RF clocks and observers:**
  - Long RF clocks, full-field DFT clocks and TEM phases use float64.
  - Lumped and terminal magnetic observers use current into the field.
  - Same-step polarization current enters direct lumped and network sampling.
- **Deterministic forward source injection.** Oblique and interpolated TFSF faces, periodic and Bloch wraps, batched lanes and the core-path periodic wrap used to accumulate shared cells with float32 atomics, so repeated runs could differ at about 1e-6 relative. Injection is now run-to-run deterministic. No pinned result moved, and oblique TFSF scenes step about 4% slower.
- **Deterministic adjoint gradients.** Gradients for TFSF auxiliary samples, collocated-monitor objectives and coincident point seeds are now run-to-run deterministic. Fixed-order gathers replace the native float atomics.
  - **Density and parameter-field gradients.** The trilinear sampling of `MaterialRegion` densities and Box parameter fields, the main inverse-design gradient path, uses native kernels.
    - Their forward and coordinate gradient are bitwise equal to the previous path.
    - The texture gradient is an ordered gather with per-term rounding, run-to-run deterministic, and equal to the previous result wherever a voxel has at most two contributions.
    - Backward cost on an inverse-design density scene rose by about 2.5%.
    - Double backward through this sampling raises.
- **Distributed periodic source crop.** In multi-GPU distributed FDTD with periodic or Bloch boundaries, a source term cropped to one GPU's owned x range kept the gather built for the uncropped patch. That had two effects:
  - the injection read past the cropped data, so results depended on leftover device memory;
  - the remaining planes were routed one cell low.

  The gathers are now rebuilt after the crop. For example, a uniform current source straddling the seam between GPUs was off by 4.84, and now matches the single-device result. Only the distributed path is affected; the single-GPU default path is not.
- **Dispersive interfaces.**
  - Under the `kottke_edge` default, each Yee component edge takes one constituent's complete pole set (binary ownership) for piecewise-constant pole media, including exactly uniform custom pole fields.
  - Spatially graded pole media, and the gradient-oriented subpixel modes, keep fractional pole weights.
  - Box faces declared in float32 snap to grid planes within their declaration rounding.
  - This removes the fractional-ADE interface error in dispersive scatterers. Drude and Lorentz sphere cross sections now converge toward the analytic Mie solution under grid refinement (observed order about 1.3), better than the old fractional weights; they are not exact on coarse grids.

## New features

| Feature | Status |
| --- | --- |
| Lumped loads and ports in complex fixed-k Bloch runs (R/L/C, series/parallel RLC terminations, checkpoint/resume) | Driven ports, copper terminals, MNA networks and adjoints fail closed |
| Conformal two-sided thin-sheet runtime for tilted and faceted `SurfaceMesh` CopperFoil or PEC sheets | Native CUDA |
| Conformal-sheet lumped gap feeds | **Experimental**; tilted/curved driven accuracy not qualified |
| Port hooks for external field kernels: `mw.port_voltage`, `mw.port_phasors`, `Result.port_drive`, `mw.trapezoidal_impedance` ([plan](https://github.com/witwin-ai/witwin-maxwell/blob/witwin-maxwell-v0.6.0/docs/plans/port-vi-hooks.md), [conventions](https://github.com/witwin-ai/witwin-maxwell/blob/witwin-maxwell-v0.6.0/docs/reference/rf-port-conventions.md)) | — |
| Consumer APIs: `Result.from_fields`, public `Result.metadata`, `mw.postprocess.plane_payload`, FDFD solver memory limits, `Result.release_solver()` | — |
| `Scene.to_dict()` / `Scene.from_dict()` with the JSON-safe `witwin.scene/1` format ([plan](https://github.com/witwin-ai/witwin-maxwell/blob/witwin-maxwell-v0.6.0/docs/plans/scene_dict.md)) | — |
| `averaging="kottke_smooth"`, a differentiable Kottke mode for geometry and material-value optimisation | **Experimental, opt-in**; density gradients not qualified |
| `averaging="polarized_tensor"` | Opt-in, FDTD only |
| `BendCoordinates` fixed-curvature orthogonal bends | **Experimental** |
| Inverse-design toolkit `witwin.maxwell.invdes`: filters, projections, length-scale constraints, symmetry, density parameterisation, design loop | — |
| PCB and printed-line workflows ([support matrix](https://github.com/witwin-ai/witwin-maxwell/blob/witwin-maxwell-v0.6.0/docs/reference/pcb-fdtd-support-matrix.md)): two-sided finite-thickness `CopperFoil`, `PlanarPEC`, named planar `TerminalPort` contacts, `LumpedPort.voltage_aperture`, PCB scene import, good-conductor and quasi-TEM modes, selected copper field queries | — |
| RF: coaxial lumped ports and resistors, `PECFrame`, terminal-wave ports with custom V/I integrals, modal port sweeps | **Experimental** |
| `NetworkData(reference_matrix=...)` | — |
| Boundaries: second-order Mur, `ModeABC` (including broadband), `InternalAbsorber`, generalized Bloch and CPML layouts | — |
| Materials: `Material.pmc()`, dispersion fitting and CSV loading, a material library, `Material.custom_sellmeier`, perturbation pole sensitivities | — |
| Opt-in surface roughness (Hammerstad, Huray, Groisse) | Opt-in |
| Geometry: boolean `Union`, `Intersection`, `Difference`; `Mesh.from_stl` | — |
| Grid refinement: `LayerRefinementSpec`, `CornerFinderSpec`, `GridRefinement` | Opt-in |
| Sources, monitors and postprocessing: broadband and table source times on surface sources, projection and Gaussian-overlap monitors, surface and auxiliary field time monitors, `FieldProjector`, a harmonic-inversion resonance finder, array-factor, taper and lobe metrics | — |
| FDFD: multifrequency solves, graded grids, in-domain staircase PEC, diagonal magnetic media, GPU local responses | — |
| Batch material reuse: density templates reuse a fixed material bank and recompile only changed patches | — |
| Surface integral equations: `Simulation.mom(...)`, scalar PMCHWT with dense, hierarchical, FMM and pFFT backends, open-sheet PEC EFIE, Calderón and near-field sparse-LU preconditioners | **Experimental**; development is paused |
| Layered-media MPIE MoM network solver for planar RF/IC passives: `Scene(background=LayerStack(...))` with `PlanarPEC` metal sheets and `LumpedPort` terminals, `Simulation.mom(scene, frequencies=[...], config=MoMConfig(layered=...))` returns `Result.network`. Native CUDA layered Green functions, deterministic Galerkin assembly, vias and vertical ports, thin-sheet conductor loss, a loop-tree low-frequency solve, GDS import (`process.scene_from_gds`) and an adaptive rational frequency sweep ([plan](https://github.com/witwin-ai/witwin-maxwell/blob/witwin-maxwell-v0.6.0/docs/plans/rfic_mom_development_plan.md)) | **Experimental**; inductance and Q-peak accuracy targets are strict expected failures; no closed thick-metal model, sources, monitors or derivatives |
| Broadband fixed-k TFSF: pulsed oblique TFSF-slab plane waves under fixed-k Bloch boundaries, driven by a complex 1D Yee incident line with the grid's discrete transverse wavenumber, normal spacing and time step (CPML at both ends; checkpoint/resume). `Result.metadata["tfsf_incident_reference"]` exports the incident electric spectrum at the lower TFSF plane, so `reflection_transmission(result, ...)` needs no separate incident run | Beams, mode and custom-field sources and soft plane waves stay CW-only under fixed k; phase is relative to the incident-line reference (see Known limitations) |
| Prepare-time incident-line spectrum warnings: the share of a non-CW waveform near the TFSF or fixed-k incident line's Yee band edge, and every source's DC content, each warned once above max(1e-6, ε32·√N). Fixed-k lines use their own band edge, asin(S·√(1+(k_t·Δs/2)²))/(πΔt), and integer-step drive clock | Warn-only; waveforms and compiled sources are never altered |

## Performance

All timings are on an A6000 (sm_86). Results on the default path are bitwise equal to the existing path on the release census (see the first section).

### Default core path

Under `fdtd_core_path="auto"`, 96 of the 121 release-census scenes run on the core path.

- **Core scenes.** Whole-call time (prepare and run, cold processes) fell from 670.5 s to 318.9 s (−52.4%). The run-only median speedup is 1.37×.
- **Existing-path scenes.** The FDTD scenes that keep the existing path fell from 296.4 s to 236.6 s, through accelerated setup.

The core path runs each step as one fused magnetic and one fused electric launch, with device-indexed source and DFT tables, and captures whole steps in 32-step chunks. Material and coefficient setup also runs on the GPU: a 256³ material compile went from 7.0 s to 0.21 s.

**Admitted.** Every admitted feature has an explicit bitwise gate scene:
- boundaries: `cpml`, stable PML, simple `pml`, graded `absorber`, none, PEC (including PEC symmetry) and periodic;
- sources: point and current sources, plane waves (soft and TFSF, including oblique TFSF), Gaussian and astigmatic beams, mode sources and custom field/current sources;
- monitors: point, plane, flux, finite-plane, closed-surface, mode, diffraction, permittivity, field-time, flux-time and breakdown monitors;
- RF: ports, lumped elements, thin wires, networks, CopperFoil and thin sheets, inside whole-step capture;
- media: dispersive (ADE), nonlinear, time-modulated and dielectric-breakdown media.

**Per-step loop.** Circuits, circuit-bound ports, SIBC, cut surfaces and host-driven wire monitors run the fused Yee kernels in the per-step loop, not in the multi-step graph. So do TFSF, magnetic sources and modulation when combined with ports or conductors.

**Step time.** On slab-CPML and no-PML cubes from 128³ to 384³, step time is within 1.025–1.107× of the Yee memory-traffic floor (128³ slab 1.107×, 384³ slab 1.085×, 384³ no-PML 1.025×).

Plans: [single-path plan](https://github.com/witwin-ai/witwin-maxwell/blob/witwin-maxwell-v0.6.0/docs/plans/fdtd-single-path-cuda.md); an earlier default-path A/B of the opt-in stage found no difference on the 31 default-path bit-identity nodes. The harnesses are [`benchmark/core_path/gates.py`](https://github.com/witwin-ai/witwin-maxwell/blob/witwin-maxwell-v0.6.0/benchmark/core_path/gates.py) and [`benchmark/core_path/setup_gates.py`](https://github.com/witwin-ai/witwin-maxwell/blob/witwin-maxwell-v0.6.0/benchmark/core_path/setup_gates.py).

### Core-path kernel cuts in this release

- **TFSF auxiliary-line fusion** (core ABI v11). For axis-aligned auxiliary-line TFSF, each field correction runs in one launch with its opposite auxiliary-line update. The two-cell incident boundary runs in one separately rounded launch.
  - On the 256³ TFSF row, launches per step fall from 42 to 8.
  - Median step time falls by 39.8% at 64³ (99.5 to 59.9 µs), 11.5% at 128³ (334.5 to 296.0 µs) and 1.9% at 256³ (2162 to 2121 µs).
  - Non-TFSF controls are unchanged to ±0.08%.
  - Oblique providers and wider ghost stencils keep the unfused route.
- **PEC clamp omission.** PEC clamp launches that cannot change a value are omitted: −7.2% at 128³ and −4.2% at 256³ on PEC rows.
- **Observer gather.** Grouped point/plane spectra are gathered in one launch per batch of up to 16 groups.
- **L2-aware Yee block order.** The core Yee launch uses 32-block j-strips when the plane reuse footprint exceeds two thirds of the device L2; otherwise it keeps the stock order. The 384³ slab row went from 1.148× to 1.085× of the byte floor, and the 384³ no-PML row from 1.121× to 1.025×. The threshold is calibrated on sm_86 only.
- **Auxiliary TFSF setup.** Each auxiliary TFSF step is evaluated once during setup. On the F2 data-generation samples, the setup share of the run fell from 10–12% to about 8.5%.

### Setup and observers

- **Point-monitor setup and readout.** Point-monitor setup is batched per Yee component, and observer setup and readout run with cyclic garbage collection paused. Outputs are bitwise unchanged.
  - On a 16,384-point sample, warm end-to-end core-path time fell from 2.95 s to 1.55 s.
  - The GC pause then cut a further 25–28% (1.89 to 1.42 s, and 1.81 to 1.30 s, in two rounds).
- **Accelerated setup.** It now admits fixed float64 Box, Sphere, Cylinder, Pyramid and PolySlab declarations and boolean trees, at the reference query precision.
- **What it builds.** `accelerated_setup` builds these with CUDA:
  - material and coefficient tables;
  - mesh distances, including a GPU LBVH;
  - cylinder, pyramid and polyslab distances;
  - PEC/PMC sampling;
  - lossy Kottke reductions;
  - periodic images.

  Pieces it does not support keep the PyTorch setup.

### Default-on kernels

- **Fused CopperFoil steps.** The copper-plus-port step drops from 0.92 to 0.29 ms, and from 438 to 54 kernels. Results agree with the torch reference, which remains the reference and adjoint path, within declared tolerances.
- **Fused conformal thin-sheet steps.** These drop from 263 to 69 kernels per step. `WITWIN_MAXWELL_THIN_SHEET_KERNEL=torch` selects the reference path. See the [plan](https://github.com/witwin-ai/witwin-maxwell/blob/witwin-maxwell-v0.6.0/docs/plans/thin-sheet-fused-kernels.md).
- **Thin-wire compilation.** It is vectorised and cached once per scene and device, and is bitwise unchanged. See the [plan](https://github.com/witwin-ai/witwin-maxwell/blob/witwin-maxwell-v0.6.0/docs/plans/thin-wire-compile-speedup.md).
- **Grouped spectral point and plane monitors.** These run inside CUDA graphs. A 31-frequency antenna case dropped from 247 s to 4 s.

## Deprecated / planned

0.6.0 is intended to be the last release that keeps the slower runtime paths as execution options ([single-path plan](https://github.com/witwin-ai/witwin-maxwell/blob/witwin-maxwell-v0.6.0/docs/plans/fdtd-single-path-cuda.md)). In 0.6.0 the existing stepping path is the fallback for `"auto"` and the path for explicit `fdtd_core_path=False`.
- After 0.6.0, the CUDA core path and accelerated setup are intended to become the only execution path.
- The CPU backend and the torch reference kernels are intended to remain only as test oracles.
- Torch autograd paths remain until the CUDA gradients are qualified.

## Known limitations

Known defects do not block this release. Each entry gives its magnitude, its scope (which features or paths, and whether they are on by default) and a workaround.

- **Architecture coverage of the core default.**
  - Magnitude: the automatic core default, its bitwise census and every performance figure in these notes were measured on A6000 (sm_86) only. The L2 threshold for the core Yee block order is calibrated only there.
  - Scope: all eligible FDTD scenes on other GPU architectures. The kernels run, but they are not qualified and may differ at rounding level.
  - Workaround: pass `fdtd_core_path=False` to use the existing path.
- **Thin-wire port impedance does not converge under refinement.**
  - Magnitude: thin-wire port input impedance and S11 do not converge under grid refinement. The median Richardson ratio across the refinement ladder is 0.75; a converging quantity gives about 2 to 2.9. The values also drift away from an analytic dipole estimate.
  - Scope: ports on thin-wire antennas (opt-in thin-wire feature). Field solutions, other port kinds and lumped ports are not affected.
  - Workaround: do not use thin-wire port Z_in/S11 for quantitative results. A fix is targeted for 0.6.1.
- **Curved and tilted copper.** Two separate effects:
  - Finite-edge response: on tilted or curved thin copper sheets the edge response is low by about 4–7%. The root cause is still under investigation. Scope: tilted and curved thin-copper antennas only; planar, grid-aligned copper foils are not affected. Workaround: none qualified. Validate tilted or curved thin-copper results against a refined grid.
  - Rim staircase: on driven conformal-sheet antennas on tilted or curved sheets, the resonance follows the rim staircase, about 2% at λ/40.
- **Oblique TE accuracy under fixed k.**
  - Magnitude: under the default complex-frequency-shifted CPML profile (α = 1.0), 35° oblique TE copper-foil transmission is within 4.43%, against a 3% gate. The case is a strict expected failure in the test suite. No single α meets the TE gate and stays passive at steep angles: α = 0.5 meets the TE number but grows energy at 60°, through a bound LC mode at +7.7e-7 per step against the 4e-7 bound.
  - Scope: complex fixed-k (oblique periodic) cells with TE incidence. Normal incidence and TM are not affected.
  - Workaround: none qualified. A smaller α is not recommended, because it is non-passive at steep angles.
- **Fixed-k source coverage.** Pulsed TFSF-slab plane waves run under fixed k in 0.6.0. Beams, mode and custom-field sources and soft plane waves stay CW only. The pulsed `bloch_oblique` and `bloch_oblique_te` benchmark rows stay quarantined in this release's validation suite until their limits are re-derived for the broadband provider.
- **Fixed-k TFSF phase.** Fixed-k TFSF reports incident phase relative to its incident-line reference; magnitudes and r/t ratios are unaffected. Absolute phase is planned for 0.6.1.
- **Benchmark residuals.**
  - **Reference-convention rows.** These rows fail against the external reference because of a convention difference, not a wrong result:
    - Nine rows have field L2 near 2 with shape L2 ≤ 0.15, a sign or absolute-phase convention difference: ring_resonator_s21, waveguide_s_matrix, mode_source_wg, mode_monitor_straight_wg, mode_monitor_two_planes, tfsf_vacuum, rcs_dielectric_box, rcs_pec_sphere and sphere_rcs.
    - Three rows keep a source-gauge difference after re-scoring: custom_current_source, magnetic_current_vacuum and uniform_current.
    - custom_grid_slab sits at L2 0.120 against a 0.1 target.
    - ricker_axis_x_anisotropic keeps an amplitude offset from the reference's Ricker waveform export. Maxwell's own result is independent of the waveform to 0.1%.
  - Ten benchmark scenarios keep an unresolved residual against the external reference solver. They keep the global limits and are reported on every run: custom_field_source, dispersive_kerr_slab, kerr_slab, kerr_slab_strong, lorentz_resonator, metal_sphere, mode_port_straight_wg, modulated_slab, modulated_slab_phase, sellmeier_sphere.
  - **custom_field_source:** the library defect, an aperture overcount and E/H components not staggered by half a cell, is fixed. The row stays open because the external reference itself carries a coarse-grid source artefact: transmission +16.7% and a 35× backward wave. The source is qualified by convergence against a continuum radiator instead.
  - **modulated_slab and modulated_slab_phase:** partial-cell coefficients now use the effective-medium derivative of the absolute time-varying permittivity. Interior cells and unmodulated scenes are bitwise unchanged. The residual sideband L2 against the external reference is 0.19/0.22 for modulated_slab (previously 0.29/0.39) and 0.10/0.27 for modulated_slab_phase (previously 0.15/0.40). The remaining cause is an interface-convention difference from the reference, which is not fully resolved. Scope: time-modulated media. Workaround: none qualified.
  - **mode_port_straight_wg:** the library result is correct, with a native phase error of 0.012 rad. The open row comes from a stale reference sampling plane and from a metric that is sensitive to the phase convention. No user workaround is needed.
- **Passivity guard coverage.**
  - Real-field periodic and closed-transverse cells can show late-time CPML growth through a grazing mode. Use a graded `absorber` for long runs ([note](https://github.com/witwin-ai/witwin-maxwell/blob/witwin-maxwell-v0.6.0/docs/memory/fdtd-real-periodic-cpml-growth-2026-11-01.md)).
  - Complex fixed-k cells with dielectric layers are not guaranteed passive.
  - **Fixed-k growth rates.** The CPML passivity guard does not detect growth slower than 4e-7 per step, which is about 1.5× energy over 1e6 steps. Complex fixed-k (oblique Bloch) CPML can grow both inside and well above that band. All measurements below are at 160 cells per wavelength, with f0 = 1 GHz and k0 = 2π f0 / c.
    - Below the band: 2.1e-7 per step, about 1.23× energy per 1e6 steps, in a point-dipole εr = 4 slab cell.
    - Near the band:
      - 3.5e-7 to 5.0e-7 per step in εr = 4 TM CW records at 1.1 f0 and at 20° incidence;
      - 7.0e-7 to 7.5e-7 per step in the vacuum TM CW arm at k_par = 0.25 k0, 0.9 f0.
    - Slab-guided mode: 8.2e-7 per step late in a CW record of an εr = 4 TM slab at k_par = 0.25 k0, 0.9 f0, where a slab-guided TM mode below the light line grows in the CPML. The energy is flat for the first 68k steps, and over half that 137k-step record r/t move by less than 3e-6.
    - About 10× larger:
      - 7.7e-6 per step in the vacuum TM CW arm at k_par = 0.55 k0, 0.9 f0, fitted over steps 104,672–135,392 of a 136,854-step record.
      - 8.2e-6 per step in a vacuum TM CW record at 20° incidence and 0.9 f0, run with 600 transient cycles (steps 207,328–238,048 of 239,494). The same case with 300 cycles fits 5.0e-7 per step, so the rate rises late in long CW records.
      - 7.7e-6 ± 1.9e-6 per step in an early window (steps 12,000–42,720) of a 320,000-step pulsed vacuum TM record at k_par = 0.55 k0. The late window of the same record family fits −7.8e-8 ± 6.2e-7, which is a high-variance false positive.
    - The reported acceptance results are unaffected at their existing bounds: pulse/CW parity ≤ 2.5e-4, CW r/t continuity ≤ 1.2e-4 and cutoff κ ≤ 8.6e-4.
    - Workaround: check the energy of long fixed-k runs, especially TM runs at larger k_par. Prefer CW per frequency with the default transient, or short pulsed records.
    - For complex fixed-k cells the guard warns once, records the measured slope, standard error and window in `passivity_guard_warning`, and lets the run finish. It is fail-closed only for real-field closed-transverse cells, and non-finite energy fails closed everywhere.
  - `absorber="auto"` selects the graded absorber in these cells and requires at least 40 layers. Explicit `absorber="cpml"` in the affected topologies is guarded by the passivity check.
  - The guard's energy does not include the energy stored in dispersive (ADE) media, gyromagnetic media, lumped networks, SIBC, thin wires or coaxial elements. Such scenes skip the guard with a `RuntimeWarning` and are marked in the result metadata. Workaround: for long runs of these scenes, use a graded `absorber` and check late-time energy yourself.
  - **CopperFoil and thin sheets.** The guard does not run on closed-transverse CPML scenes that contain CopperFoil or thin sheets.
    - They report `passivity_guard = "skipped: energy storage not counted (CopperFoil, thin sheets)"` and warn, at zero sampling cost.
    - Sampling their counted energy every 32 steps costs 39–133% of a step (planar foil, patch, board), and sparser sampling false-trips the guard.
    - The counted functional is qualified but not sampled by default in 0.6.0. Counting these scenes is planned for 0.6.1.
    - Workaround: as for the other skipped scenes, use a graded `absorber` for long runs and check late-time energy yourself.
    - See the [CPML reference](https://github.com/witwin-ai/witwin-maxwell/blob/witwin-maxwell-v0.6.0/docs/reference/fdtd-cpml.md).
- **Conformal PEC.** `pec="conformal"` keeps a curved-body bias and warns at construction ([note](https://github.com/witwin-ai/witwin-maxwell/blob/witwin-maxwell-v0.6.0/docs/memory/conformal-pec-bias.md)). `kottke_edge` is not geometry-differentiable.
- **Validation gates affected by the new defaults.** Each is listed by its resolution state:
  - Resolved: the batched-source, custom-current reader, batch gradient, ESD and lumped-adjoint tests were stale against the new half-step source timing or the new subpixel default. They have been updated.
  - Resolved, scope note: the modal port sweep phase gate states the scope it qualifies.
  - Resolved, noise floor: the PEC-cavity ring-down gate accounts for the FP32 noise floor.
  - Requalified under the `kottke_edge` default: the one-cell Rayleigh sphere `sigma_max_ratio` gate now runs under the default policy, with band 1.512–1.693. The resolved-conductor convergence test stays on its qualified one-sample arithmetic policy (see the resolved-conductor entry below).
- **Resolved volumetric conductors under `kottke_edge`.**
  - Magnitude: at 2–4 cells per skin depth, absorbed power shows ±4–8% non-monotone sensitivity to the grid tier. For a resolved cylinder at 0.6/0.5/0.42/0.35/0.3 mm cells, the absorbed power is 7.922/7.324/7.381/7.064/7.124 MW. The SIBC model of the same cylinder is grid-independent to 2.26%, inside its documented 25% first-order systematic.
  - Scope: good conductors resolved as volumetric lossy media under the default `kottke_edge` material policy.
  - Workaround: model good conductors with SIBC (`LossyMetalMedium`).
- **Second-order nonlinear media on the core path.**
  - Magnitude: unknown. Second-order nonlinear (chi2) media run on the core path under `"auto"` and pass their physics tests there (second-harmonic scaling and phase matching), but no core-vs-existing equality check covers them. Every other admitted medium class has bitwise core-vs-existing evidence: electric and magnetic dispersive media, Kerr, two-photon absorption, time modulation, their compositions and deterministic breakdown.
  - Scope: chi2 scenes on the default path.
  - Workaround: pass `fdtd_core_path=False` to use the existing path.
- **Fallback reason without a CUDA toolkit.**
  - Magnitude: cosmetic. When no CUDA toolkit is installed, the first `FDTDCoreFallbackWarning` in a process names the missing toolkit. Later `Simulation`s in the same process, and the recorded `fallback_reason`, instead report a missing setup-extension shared object (`.so`). Results and execution path are correct.
  - Scope: installs without `nvcc`/`CUDA_HOME`, under `"auto"`.
  - Workaround: none needed. Clearer reporting is planned for 0.6.1.
- **MoM dual-budget check.**
  - Magnitude: hierarchical and FMM MoM configurations whose `stopping.rtol` exceeds 0.1 × `solve_dual_tolerance` raise `ValueError`. This affects 41 cases in the MoM test suite, and the behaviour is the same as at the previous release base.
  - Scope: the experimental MoM solver, whose development is paused.
  - Workaround: set `stopping.rtol` at or below 0.1 × `solve_dual_tolerance`.
- **MoM run-to-run reproducibility.**
  - Magnitude: MoM FMM and dense-reference solves are not yet bitwise run-to-run reproducible; repeated runs differ by up to about 1e-14 relative in dense-reference currents and about 1e-10 relative in FMM currents, so a tolerance-sensitive test can fail intermittently.
  - Scope: the experimental MoM solver; unchanged from the previous release base.
  - Workaround: compare MoM results with tolerances rather than bitwise.
- **Core path scope.**
  - The core path covers single-GPU float32 forward runs only.
  - Under `"auto"`, these keep the existing path with `FDTDCoreFallbackWarning`, and explicit `fdtd_core_path=True` raises for them:
    - Bloch or complex fields;
    - PMC (including symmetry planes), Mur and mode-ABC boundaries;
    - closed-transverse CPML;
    - gyromagnetic media, full or off-diagonal anisotropy, magnetic permeability and magnetic conductivity;
    - coordinate charts;
    - gradient-requiring scenes;
    - batch lanes, multi-GPU and non-float32 fields.
  - These scenes are not slower than in 0.5.1, but they do not get the core-path speedup.

## Validation

### Integration batch on the release candidate

The final integration batch ran on the release-candidate head `b6d9e1aa` against the accepted previous batch head `3dfbda86`. It used both local A6000 GPUs (sm_86).

- **Coverage.**
  - All 11 components ran, MoM included.
  - All 1291 of 1291 units completed: no missing units or shards, 0 infrastructure failures and 0 collection errors. Every merge in the range carries a declaration.
  - About 33.9 GPU-hours.
- **Units.** 1146 PASS, 71 FAIL and 74 INCOMPLETE. Cases: 13,268 passed, 166 failed, 269 skipped and 5 expected failures.
- **INCOMPLETE units (74), all by design:**
  - 28 benchmark rows excluded as unconverged (their field or flux changes by more than the limit when the runtime is doubled);
  - 2 quarantined `bloch_oblique` rows;
  - 29 MoM opt-in or pinned-artifact skips;
  - 8 opt-in CUDA extension-build tests;
  - 3 scheduled-window skips;
  - 1 two-device skip;
  - 3 other opt-in or CPU-profile skips.

  Every non-MoM INCOMPLETE unit that also ran in the previous batch had the same status there.
- **Failures, by class:**

| Class | Cases | Units |
|---|---|---|
| New and unexplained | **0** | **0** |
| Fixed after the candidate head (below) | 19 | 12 |
| Pre-existing at `3dfbda86`, same signature | 111 | 52 (28 MoM, 24 benchmark) |
| Environment or test harness | 36 | 8 |

  One MoM unit has one pre-existing case and one environment case.

- **Fixed after the candidate head.** All 19 cases pass on the fixed head. Only one of them was a wrong result on the default path:
  - **Surface ownership guard.** After the PML-interface extrusion, a scene that declared a PEC box and a surface-impedance conductor on the same face compiled, and the PEC declaration was silently dropped. The ownership check now compares the declared geometry, so this scene fails closed again.
  - **FP64 native forward with `absorber="auto"`.** It refused the new default absorber with an error; it now resolves `"auto"` the way preparation does.
  - **Tests only.** The rest are test changes, with no library change:
    - stale tests after the core default and the physical Yee labels;
    - a Kerr-slab phase test, which now uses the infinite-slab prediction that the PML extrusion makes correct.
  - On the affected lumped-port scene, the default path is bitwise equal to `fdtd_core_path=False`.
- **Pre-existing at `3dfbda86`.**
  - **MoM: 51 cases in 28 units.** Each reproduces at the base with an identical first error, and the MoM sources and tests are unchanged in the range. 41 cases are the dual-budget check (see Known limitations). The others are the free-space-only FMM restrictions and stale configuration-version expectations.
  - **Benchmark: 24 rows.** All are listed under Known limitations:
    - the ten residual rows (custom_field_source, dispersive_kerr_slab, kerr_slab, kerr_slab_strong, lorentz_resonator, metal_sphere, modulated_slab, modulated_slab_phase, sellmeier_sphere, mode_port_straight_wg);
    - nine sign/absolute-phase convention rows;
    - three source-gauge rows;
    - custom_grid_slab;
    - ricker_axis_x_anisotropic.

    Of these, only ricker_axis_x_anisotropic did not also fail at the base. The PML-interface extrusion fixed its field shape (shape L2 0.488 to 0.043), so the row is now scored. Its remaining amplitude offset is a reference waveform convention: Maxwell's result is independent of the waveform to 0.1%.
- **Environment and harness: 36 cases.**
  - An unset environment variable and an uninstalled optional package (MoM).
  - A deleted temporary checkout that a provenance test expects.
  - Two harness tests that time out under batch load but pass 40 of 40 in isolation.
  - One CPU-bound MoM test stopped after 3.6 h.
- **Bit identity.** Every pinned bit-identity baseline is T0, with no T1 movers and no mover-declaration errors. The pinned field reference is `fc72f2c3` and the raw-phase reference is `b27cac68`.
- **Determinism release gates.** Forward source-patch determinism passed 7 of 7, adjoint determinism 19 of 19 and nonlinear CUDA determinism 4 of 4.
- **Suites that pass.** These all pass: FDTD core path, FDTD material setup, FDTD baselines, the PCB suite, fused CopperFoil, the 112-file thin-sheet suite, the MoM public smoke suite and prepared examples.

### Not covered by the integration batch

- **Two-device tier, both RTX A6000s.** It runs separately from the integration batch, which excludes the 193 two-device nodes and the milestone-only nodes.
  - The full multi_gpu suite gave 401 passed and 1 failed on the release candidate. The failure was an order-dependent distributed periodic-source parity case, the distributed source-crop bug described under Physics fixes.
  - With the fix, the full suite passes 405 of 405 (402 existing tests plus 3 new regression tests). The failing case also passes in the suite order that used to trigger it.
  - Bit-identity pin files are unchanged (33 of 33 in each of 8 files).
  - Determinism gates pass.
- **Layered RFIC MoM.** It merged after the batch, and the batch does not cover it. It was validated by its own gates on the merged tree: the layered MoM suite passed 180 tests, plus 2 strict expected failures for the inductance and Q-peak targets. Consumer metadata (23 of 23), guard census and integration batch (90 passed) also pass, and all eight bit-identity files give results identical to master (33 of 33).

### Validation tooling

The integration runner has no per-unit timeout. Three CPU-bound MoM units took 1.6–3.6 h in this batch. This is documented and does not block the release.

### Other release-lineage records

- **Default-path census.** The census covers 121 benchmark and data-generation scenes, 96 of which take the core path. Under the automatic default, every FDTD scene's monitors are bitwise equal to the run with `fdtd_core_path=False, accelerated_setup=False` (A6000, sm_86). Two admission defects that the census found before the default changed are fixed: PMC symmetry planes, and a double build of closed-transverse CPML scenes.
- **PCB application matrix.** All 385 of 385 declared-band targets pass on 3dfbda86, with a maximum power-closure error of 0.456%. For every target, the scene built directly from its physical declarations and the same scene imported through the PCB import adapter produce bitwise-identical port V/I/Γ and power observations, with identical compiled coefficients and port geometry.
