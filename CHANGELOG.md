# Changelog

All notable user-visible changes to WiTwin Maxwell are documented here. The project follows semantic versioning within the `0.x` development series.

## [0.6.0] - 2026-11-08

Full details, evidence and migration notes are in the [0.6.0 release notes](docs/reference/release-notes-0.6.0.md).

### Breaking changes

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

### Changed (defaults and behavior; recompute affected cached results)

- **FDTD execution defaults to `fdtd_core_path="auto"` and `accelerated_setup="auto"`.** In 0.5.1 both were off.
  - Eligible single-GPU float32 forward scenes now run on the fused CUDA core path, with CUDA material and coefficient setup.
  - Scenes with an admission blocker keep the existing stepping. So do resume, non-initializing preparation and material-cache runs. The first automatic fallback in a `Simulation` warns once with the public `FDTDCoreFallbackWarning`.
  - If core activation or graph capture fails under `"auto"`, the run is rebuilt on the existing path.
  - `Result.stats` and `Result.metadata` report `execution_path` (`"core"` or `"existing"`) and `fallback_reason`.
  - Explicit `True` stays fail-closed: it raises for an ineligible scene. Explicit `False` keeps the existing path and does not warn.
  - PMC boundaries keep the existing path, including PMC symmetry planes. So does closed-transverse CPML, which runs under the passivity guard.
  - On the 121-scene release census, every FDTD scene's monitors are bitwise equal to the run with both controls `False`.
  - Validated on A6000 (sm_86) only. Other architectures may differ at rounding level.
  - Migration: pass `fdtd_core_path=False, accelerated_setup=False` to reproduce the 0.5.1 execution path. Use the `warnings` filters to silence or escalate `FDTDCoreFallbackWarning`. Code that inspects `prepared.solver` internals should check `execution_path` first.
- Default material sampling is `SubpixelSpec(samples=4, averaging="kottke_edge", pec="staircase")` (was one arithmetic sample). Trainable geometry requires `averaging="kottke_smooth"` (experimental) or `"polarized"`; trainable `MaterialRegion` densities require `samples=1, averaging="arithmetic"`; both fail closed otherwise. Pass `SubpixelSpec(samples=1, averaging="arithmetic", pec="staircase")` to recover 0.5.1 sampling.
- Structures that reach a PML interface and are prismatic there are continued through the PML along the face normal at preparation (FDTD and FDFD). The public `Scene` is not modified, and the extrusions are recorded as `pml_extrusions` in the prepared metadata and solver statistics. Mesh volumes, non-prismatic contacts and surface-impedance conductors (`LossyMetalMedium`, `SurfaceImpedanceMedium`, `CopperFoil`) are not extruded and warn; zero-volume sheets are only recorded. Results move only for scenes with such contacts. In the benchmark suite, 10 rows moved and none got worse; for example, `dielectric_slab` field L2 against the reference fell from 0.569 to 0.041. All other scenes are bitwise unchanged. Nonlinear structures flush with the PML now behave as infinite structures: for example, a Kerr slab's nonlinear phase shift went from 2.63e-3, an edge artefact, to 8.56e-4, which matches the infinite-slab prediction. A PEC structure and a surface-impedance conductor declared on the same face still fail closed.
- FDTD phasors use physical Yee time labels: post-step E at (n+1)Δt and H at (n+1/2)Δt.
  - Compared with 0.5.1, raw field and port phasors rotate by exp(iωΔt). Source-normalised additive-current and direct-drive phasors rotate by exp(iωΔt/2), and uncompensated magnetic, sheet and analytic-pulse phasors by exp(iωΔt). Auxiliary-line and CW references are unchanged to rounding.
  - Same-frequency S/Z/Γ, flux, matched field/port powers and directivity are unchanged.
  - Field-time monitor `t` advances by one Δt; the values are unchanged.
  - Resume checkpoints use format 6. Checkpoints with the earlier spectral convention need a fresh run.
- Additive electric currents (`PointDipole`, `UniformCurrentSource`, electric `CustomCurrentSource`) are injected at (n+1/2)Δt and normalised at their own injection instant. Raw port V/I and Z/Γ are unchanged; source-normalised spectra carry the half-step phase. Checkpoints with additive electric currents written before this change are refused on resume.
- Raw (non-source-normalised) pulsed absolute spectra are scaled per reference period instead of averaged over the record length, including single-frequency runs; with `reference_frequency=None` the period is taken per output frequency, so multi-frequency raw outputs also move off the centre frequency. Source-normalised results and all ratios (S, Z, Γ, power closure) are unchanged. Migration: absolute pulsed powers from 0.5.x differ by (f·T_record)²-type factors; compare ratios or recompute. Pass an explicit `reference_frequency` for the fixed-period convention. Absolute spectral reads that mix ongoing and transient drives without source normalisation are refused.
- Explicit `full_field_dft=False` is honoured for multi-frequency runs; volume-field reads and automatic electric power-loss channels then raise.
- Pulsed TFSF incident waves are referenced at the physical upstream entry plane, so absolute incident phase no longer depends on absorbing-layer padding. Older pulsed-TFSF checkpoints are refused on resume.
- Raw TFSF incident timing is exact on the Yee clock: auxiliary-line entry E uses the waveform at (n+1)Δt and analytic/discrete CW faces keep their physical J/M instants; normalisation references follow each provider. Migration: finite-record source-normalised TFSF values shift by up to ~1e-3 for pulses and ~1e-2 for real CW on short records; recompute affected cached results. TFSF checkpoints written before this change are refused on resume.
- The pulsed TFSF auxiliary line keeps its whole sampled extent lossless and ends in a 320-cell absorber appended beyond it. Previously its last 20 sampled cells formed a graded ramp, which reflected near-band-edge content and trapped a slowly decaying mode. In-band benchmark results agree with the old termination to 3–4 digits. Checkpoints with an auxiliary TFSF line written before this change are refused on resume.
- Complex fixed-k Bloch runs use a complex-frequency-shifted CPML by default; `cpml_config={"alpha_max": 0.0}` pins legacy coefficients for short runs. Real-field, k=0 and non-Bloch paths are unchanged.
- Fixed-k Bloch runs accept pulsed TFSF-slab plane waves (see Added). They still refuse pulsed beams, mode and custom-field sources and soft plane waves; use one CW source per frequency for those. Point and current sources may be pulsed.
- FDFD: PML now absorbs (frequency-domain form of the FDTD CPML); `PointDipole` uses the FDTD unit control-volume current moment; Yee-edge material components and exp(-iωt) loss sign are aligned with FDTD; FDFD refuses scene monitors it cannot populate and propagates solver failures.
- Other default-path numerics: high-side magnetic CPML grading at physical half points; CopperFoil free-edge impedance on by default (`edge_model="none"` for the legacy rim) and compressed passive diffusion fits; node-consistent staircase PEC; second-order multi-sample averaging on resolved Yee spacing; full-moment narrow Gaussian dipoles; periodic dipole image folding; watertight closed-surface monitors; `TwoPhotonAbsorption` conductivity factor 2/3; Yee-edge bulk conduction loss.
- Trainable fixed controls (frequencies, time/grid/sampling controls, monitor geometry, topology) and port probe gaps fully covered by PEC are rejected up front.
- Native CUDA extension ABI is `stable_abi_v13`; older prebuilt binaries are rejected and rebuilt. `WITWIN_MAXWELL_FDTD_CUDA_PREBUILT=1` now requires a matching ABI marker. The core-path extension (`core_abi_v11`) and the setup extension (`setup_abi_v6`) are compiled on first use.
- Requires `witwin>=0.4.6,<0.5`. In earlier witwin-core releases, micrometre-scale geometries were over-smoothed by about 3 cells, because the geometry smoothing width was clamped to 1.19e-7 m whatever the cell size. Optimisation results and forward fluxes change materially for µm-scale scenes (bend example: +37.7% initial flux, gradient cosine 0.85). Scenes whose smallest cell is ≥ ~2.4 µm are unchanged bit for bit. Point-like occupancy queries without coordinate spacing must now pass an explicit beta in metres (ValueError otherwise).
- `Simulation.fdtd(absorber="auto")` is the default.
  - Real-field cells with closed transverse walls (periodic, PEC or PMC) and PML on the normal axis select the graded `absorber`. It needs at least 40 layers on every PML face; thinner layers raise a `ValueError` that names the closed axes, the deficient faces and both remedies.
  - Open scenes and complex fixed-k scenes keep CPML.
  - Explicit `absorber="cpml"` in a closed-transverse real-field cell runs under a fail-closed passivity guard (`FDTDPassivityError`). In a complex fixed-k cell the guard only warns: it emits a `RuntimeWarning` and records the measured slope in `passivity_guard` metadata. Non-finite energy fails closed everywhere.
  - Results record `absorber`, `absorber_resolution_reason` and `passivity_guard`.
  - FP64 native forward resolves `"auto"` the way preparation does. It runs when `"auto"` resolves to CPML, and keeps its CPML-only refusal otherwise.
  - Closed-transverse CPML stays on the existing execution path, where the guard runs.
  - Migration: scenes that relied on the CPML default in such cells have two options. Pass `absorber="cpml"`: the numerics are unchanged and the run is now guarded. Or configure at least 40 layers: the results then come from the graded absorber.

### Fixed

- Soft equivalent-sheet plane waves launch +E_inc (absolute fields of soft plane-wave scenes previously had the wrong sign).
- Soft plane-wave and beam launch currents are no longer written into inactive electric wall rows (transverse none/PML/PEC walls), which removes a linear late-time growth of the corner H field.
- Conformal thin-sheet rims split faces only for chords inside the outline (removes a 4-7% low resonance bias).
- Custom J/M sources deposit over the Yee control-cell overlap with the declared dataset volume; custom-field sources deposit on bracketing Yee sheets with fractional aperture.
- Time modulation uses a device integer step counter with FP64 phase and an effective-medium partial-cell coefficient.
- Fixed-k Bloch: Floquet TE/TM readout with power-normalised r/t, consistent complex CW conventions with the adjoint, single deposition of periodic TFSF faces, PEC-face clamping of imaginary fields.
- Finite thin wires are stable through a passive RL impedance; long RF clocks and TEM phases use float64; lumped/terminal magnetic observers use current into the field; same-step polarization current enters direct lumped/network sampling.
- Forward source injection is run-to-run deterministic for oblique and interpolated TFSF faces, periodic and Bloch wraps, batched lanes and the core-path periodic wrap (shared cells were previously accumulated with float32 atomics, so repeated runs could differ at about 1e-6 relative). No pinned result moved; oblique TFSF scenes step about 4% slower.
- Adjoint gradients are run-to-run deterministic for TFSF auxiliary samples, collocated-monitor objectives and coincident point seeds (fixed-order gathers replace float atomics). Density and Box parameter-field material gradients are run-to-run deterministic too: their trilinear sampling uses native kernels whose forward and coordinate gradient are bitwise equal to the previous path, with an ordered texture pullback.
- Multi-GPU distributed FDTD with periodic or Bloch boundaries: a source term cropped to one GPU's owned x range kept the gather built for the uncropped patch. The injection read past the cropped data, so results depended on leftover device memory, and it routed the remaining planes one cell low. The gathers are now rebuilt after the crop. For example, a uniform current source straddling the seam between GPUs was off by 4.84 and now matches the single-device result. Distributed path only; the single-GPU default path is unaffected.
- Micrometre-scale `ModeSource`/`ModePort` launches were refused at prepare ("zero integrated Poynting power"): the zero-power check compared watts with machine epsilon. It and other absolute floors on dimensional quantities (modal power fractions, scattering/diffraction/antenna normalisation, network and port impedance normalisation, observers, TFSF power, source-term pruning, DC port-current admission) are now relative to the same-unit reference, so results do not depend on the length unit.
- Gradients through trainable material on a `ModeSource`/`ModePort` launch plane are now correct. The adjoint previously differentiated a different (scalar) mode than the one launched and lost the 1/ε injection-coefficient derivative, giving gradients wrong by up to ~7.6× (reference guide, central finite differences on each tree: before, adjoint −18334 vs FD −2417; now, adjoint −2418 vs FD −2425). Trainable lossless launch planes now differentiate via a torch dense vector eigen-solve shared by the forward launch and the adjoint replay (operator n ≤ 4096). Lossy or conductive, near-degenerate and above-cap trainable planes refuse with a named `NotImplementedError`.
- Dispersive interfaces under the `kottke_edge` default: each Yee component edge takes one constituent's complete pole set (binary ownership) for piecewise-constant pole media, including exactly uniform custom pole fields; spatially graded pole media and the gradient subpixel modes keep fractional pole weights; float32-declared box faces snap to grid planes within their declaration rounding. Replaces the fractional-ADE interface error in dispersive scatterers; sphere cross sections converge toward the analytic Mie solution under refinement.

### Added

- Broadband fixed-k TFSF: pulsed oblique TFSF-slab plane waves under fixed-k Bloch boundaries. They are driven by a complex 1D Yee incident line with the grid's discrete transverse wavenumber, normal spacing and time step. `Result.metadata["tfsf_incident_reference"]` exports the incident electric spectrum at the lower TFSF plane, so `reflection_transmission(result, ...)` needs no separate incident run. Checkpoint/resume is supported.
- Prepare-time warnings for TFSF and fixed-k incident lines and for source waveforms. They report the share of a non-CW waveform near the incident line's Yee band edge and its DC content, once per hazard above max(1e-6, ε32·√N). Fixed-k lines use their own band edge and drive clock. Waveforms and compiled sources are never altered.
- Complex fixed-k Bloch lumped loads and ports (R/L/C, series/parallel RLC terminations, checkpoint/resume).
- Conformal two-sided thin-sheet runtime for tilted and faceted `SurfaceMesh` sheets; experimental conformal-sheet gap feeds.
- `mw.port_voltage`, `mw.port_phasors`, `Result.port_drive`, `mw.trapezoidal_impedance`; `Result.from_fields`, public `Result.metadata`, `mw.postprocess.plane_payload`, FDFD solver memory limits and `Result.release_solver()`.
- `Scene.to_dict()` / `Scene.from_dict()` (`witwin.scene/1`).
- Opt-in `averaging="kottke_smooth"` differentiable subpixel mode (experimental) and FDTD-only `averaging="polarized_tensor"`.
- Experimental `BendCoordinates`, inverse-design toolkit `witwin.maxwell.invdes`, PCB/printed-line workflows (`CopperFoil`, `PlanarPEC`, `TerminalPort`, PCB scene import, good-conductor and quasi-TEM modes), experimental coaxial/terminal-wave ports and modal port sweeps, Mur/`ModeABC`/`InternalAbsorber`, PMC, dispersion fitting and a material library, opt-in surface roughness, boolean geometry and STL meshes, grid refinement specs, broadband/table source times, projection and time monitors, `FieldProjector`, resonance finder and array-factor tools, FDFD multifrequency/graded grids/GPU local responses.
- Batch material reuse: density templates reuse a fixed material bank and recompile only changed patches.
- Surface integral equations (`Simulation.mom(...)`, PMCHWT with dense/hierarchical/FMM/pFFT backends, open-sheet PEC EFIE, preconditioners). Experimental; development is paused.
- Experimental layered-media MPIE MoM network solver for planar RF/IC passives: `Scene(background=LayerStack(...))` with `PlanarPEC` sheets and `LumpedPort` terminals, `Simulation.mom(scene, frequencies=[...], config=MoMConfig(layered=...))` returns `Result.network`. It includes vias and vertical ports, thin-sheet conductor loss, a low-frequency loop-tree solve, GDS import and an adaptive rational sweep. Its inductance and Q-peak accuracy targets are strict expected failures.

### Performance

All timings are on an A6000 (sm_86).

- **Default core path.** 96 of the 121 release-census scenes take the core path under `"auto"`.
  - On those scenes, whole-call time (prepare and run, cold processes) fell from 670.5 s to 318.9 s (−52.4%). The run-only median speedup is 1.37×.
  - On the FDTD scenes that keep the existing path, whole-call time fell from 296.4 s to 236.6 s through accelerated setup.
  - Fused single-GPU float32 stepping has one magnetic and one electric launch per step and graph capture in 32-step chunks. It covers absorbing, PEC and periodic boundaries, generic sources and monitors, RF ports, lumped elements, wires, networks, CopperFoil, thin sheets, SIBC, and dispersive, nonlinear and modulated media.
  - On slab-CPML and no-PML cubes from 128³ to 384³, step time is within 1.025–1.107× of the Yee memory-traffic floor.
- Core TFSF auxiliary-line corrections are fused with their auxiliary updates (core ABI v11). On the 256³ TFSF row this cuts launches from 42 to 8 per step. Step time falls by 39.8% at 64³, 11.5% at 128³ and 1.9% at 256³.
- Further core-path cuts:
  - Redundant PEC clamp launches are omitted: −7.2% at 128³ and −4.2% at 256³ on PEC rows.
  - Observer gathers use one launch per batch of up to 16 groups.
  - The Yee block order is L2-aware: the 384³ slab row went from 1.148× to 1.085× of the byte floor.
  - Each auxiliary TFSF step is evaluated once during setup.
- Point-monitor setup and readout are batched, and run with cyclic garbage collection paused. On a 16,384-point sample, warm end-to-end core-path time fell from 2.95 s to 1.55 s, then by a further 25–28% with the pause.
- Accelerated setup admits float64 primitive declarations (Box, Sphere, Cylinder, Pyramid, PolySlab and boolean trees).
- Fused CopperFoil and conformal thin-sheet CUDA kernels on by default (validated against the torch reference within declared tolerances; the torch path remains the reference and adjoint path); cached, vectorised thin-wire compilation; graph-captured grouped spectral monitors.

### Validation

- Final integration batch on the release candidate `b6d9e1aa`: all 1291 of 1291 units, 0 infrastructure failures, 33.9 GPU-hours on A6000.
  - 1146 units pass and 74 are incomplete by design.
  - 0 failures are new and unexplained. 19 cases are fixed after the candidate head (one was a default-path wrong result: the surface ownership guard). 111 are pre-existing at the base and 36 are environment or harness cases.
  - Bit identity is T0, and the determinism gates pass: source patch 7 of 7, adjoint 19 of 19, nonlinear 4 of 4.
  - Details are in the release notes.
- Two-device tier, both RTX A6000s:
  - The full multi_gpu suite gave 401 passed and 1 failed on the release candidate. The failure was an order-dependent distributed periodic-source parity case, the distributed source-crop bug fixed above.
  - With the fix, the full suite passes 405 of 405 (402 existing tests plus 3 new regression tests). The failing case also passes in the suite order that used to trigger it.
  - Bit-identity pin files are unchanged (33 of 33 in each of 8 files).
  - Determinism gates pass.
- The layered RFIC MoM solver merged after the batch. It was validated by its own gates, not by the integration batch.

### Deprecated / planned

- 0.6.0 is intended to be the last release that keeps the slower runtime paths as execution options. In 0.6.0, the existing stepping path is the fallback for `"auto"` and the path for explicit `False`.
  - After 0.6.0, the CUDA core path and accelerated setup are intended to become the only execution path.
  - The CPU backend and the torch reference kernels are intended to remain only as test oracles.
  - Torch autograd paths remain until CUDA gradients are qualified.

### Limitations

Known defects are documented with magnitude, scope and workaround in the release notes; they do not block the release.

- The automatic core default and its performance figures are validated on A6000 (sm_86) only. The L2-based Yee block-order threshold is calibrated only there. Pass `fdtd_core_path=False` on other architectures if you need the existing path.
- Second-order nonlinear (chi2) media run on the core path under `"auto"` and pass their physics tests there (second-harmonic scaling and phase matching), but have no core-vs-existing equality check. Dispersive (electric and magnetic), Kerr, two-photon absorption, time modulation, their compositions and deterministic breakdown are bitwise equal to the existing path. Pass `fdtd_core_path=False` for chi2 scenes that need the qualified existing path.
- Without a CUDA toolkit (a pip install with no `nvcc`/`CUDA_HOME`), `"auto"` runs complete on the existing path with one `FDTDCoreFallbackWarning`. The first warning names the missing toolkit. Later `Simulation`s in the same process, and the recorded `fallback_reason`, instead report a missing setup-extension shared object. This is cosmetic and is fixed in 0.6.1.
- The core path does not run these, which keep the existing path under `"auto"`:
  - Bloch or complex fields;
  - PMC (including symmetry planes), Mur and mode-ABC boundaries;
  - closed-transverse CPML;
  - gyromagnetic media, full anisotropy, magnetic permeability or magnetic conductivity;
  - coordinate charts;
  - gradient-requiring scenes;
  - batch lanes, multi-GPU and non-float32 fields.
- Thin-wire port Z_in/S11 does not converge under refinement (Richardson ratio median 0.75); do not use it for quantitative results. Fix targeted for 0.6.1.
- Curved or tilted thin copper: finite-edge response low by about 4–7% (no qualified workaround; validate against a refined grid); driven conformal-sheet antennas on tilted/curved sheets follow the rim staircase (about 2% at λ/40).
- Oblique TE under fixed k: 35° transmission within 4.43% against a 3% gate with the default CFS-CPML profile; no qualified workaround (a smaller α is not passive at steep angles).
- Under fixed k, beams, mode and custom-field sources and soft plane waves are CW only. The pulsed `bloch_oblique` benchmark rows stay quarantined in this release's validation suite.
- Fixed-k TFSF reports incident phase relative to its incident-line reference; absolute phase is planned for 0.6.1.
- MoM: hierarchical and FMM configurations with `stopping.rtol` above 0.1 × `solve_dual_tolerance` raise `ValueError` (unchanged from the previous release base); set `stopping.rtol` at or below that bound.
- MoM FMM and dense-reference solves are not yet bitwise run-to-run reproducible (repeated runs differ by up to about 1e-14 relative in dense-reference currents and about 1e-10 relative in FMM currents); a tolerance-sensitive test can fail intermittently.
- Benchmark rows with a reference-convention difference also fail against the external reference: nine sign or absolute-phase rows (field L2 near 2, shape L2 ≤ 0.15), three source-gauge rows, custom_grid_slab (L2 0.120 against 0.1) and ricker_axis_x_anisotropic (reference waveform export). See the release notes.
- Ten benchmark scenarios keep an unresolved residual against the external reference solver and are reported on every run, including custom_field_source, modulated_slab(_phase) and mode_port_straight_wg (see release notes).
- Real-field periodic or closed-transverse cells can show late-time CPML growth; `absorber="auto"` therefore selects the graded absorber there (at least 40 layers), and explicit `absorber="cpml"` in these topologies runs under the passivity guard.
- The CPML passivity guard does not detect growth slower than 4e-7 per step (about 1.5× energy over 1e6 steps). Complex fixed-k (oblique Bloch) CPML can grow both inside and well above that band. Measured at 160 cells per wavelength, f0 = 1 GHz, k0 = 2π f0 / c:
  - Below the band: 2.1e-7 per step (about 1.23× per 1e6 steps) in a point-dipole εr = 4 slab cell.
  - Near the band: 3.5e-7 to 5.0e-7 per step in εr = 4 TM CW records at 1.1 f0 and at 20° incidence. 7.0e-7 to 7.5e-7 per step in the vacuum TM CW arm at k_par = 0.25 k0, 0.9 f0.
  - Slab-guided mode: 8.2e-7 per step late in a CW record of an εr = 4 TM slab at k_par = 0.25 k0, 0.9 f0. Here a slab-guided TM mode below the light line grows in the CPML.
  - About 10× larger, in vacuum TM records:
    - 7.7e-6 per step in the CW arm at k_par = 0.55 k0, 0.9 f0, late in a 136,854-step record.
    - 8.2e-6 per step in a CW record at 20° incidence, 0.9 f0, run with 600 transient cycles. The same case with 300 cycles fits 5.0e-7 per step, so the rate rises late in long CW records.
    - 7.7e-6 ± 1.9e-6 per step in an early window of a 320,000-step pulsed record at k_par = 0.55 k0.
  - The reported acceptance results are unaffected at their existing bounds: pulse/CW parity ≤ 2.5e-4, CW r/t continuity ≤ 1.2e-4, cutoff κ ≤ 8.6e-4.
  - Long fixed-k runs, especially TM runs at larger k_par, should check their energy. Prefer CW per frequency with the default transient, or short pulsed records.
  - For complex fixed-k cells the guard warns once, records the measured slope, standard error and window in `passivity_guard_warning`, and lets the run finish. It is fail-closed only for real-field closed-transverse cells; non-finite energy fails closed everywhere.
- The passivity guard does not run on closed-transverse CPML scenes that contain CopperFoil or thin sheets.
  - These scenes report `passivity_guard = "skipped: energy storage not counted (CopperFoil, thin sheets)"` and warn.
  - Sampling their counted energy every 32 steps costs 39–133% of a step, and sparser sampling false-trips.
  - Counting is planned for 0.6.1. See `docs/reference/fdtd-cpml.md`.
  - Scenes with dispersive (ADE) or gyromagnetic media, lumped networks, SIBC, thin wires or coaxial elements are also skipped with a `RuntimeWarning`.
- Conformal PEC (`pec="conformal"`) keeps a curved-body bias and warns at construction; `kottke_edge` is not geometry-differentiable.
- Resolved volumetric conductors at 2–4 cells per skin depth under the default `kottke_edge` material policy show ±4–8% non-monotone tier sensitivity in absorbed power; use SIBC (`LossyMetalMedium`) for good conductors.

## [0.5.1] - 2026-09-08

### Added

- Experimental `Simulation.batch(...)` execution with reusable capacity-limited CUDA slots, batched field/CPML/source/observer updates, and ordinary owned `Result` objects.
- Local material compilation reuse, automatic resident capacity, private per-device CUDA allocation budgets, multi-GPU and dual-slot scheduling, and opt-in measured throughput tuning.
- PyTorch-native Batch checkpoint adjoints for supported density, dielectric geometry, and named source-scale inputs, including delayed backward after slot reuse or close.
- Reproducible Batch performance, memory, material-compilation, and device-scaling audits with numerical controls and process-contention records.

### Fixed

- Material-aware time-step stability, source timing, complex CPML and Mur boundary histories, and capture/resume state consistency in FDTD.
- Batch parameter cache identity, iterator ownership, result scene snapshots, selected electric spectra, and redundant spectral allocations in retained adjoint state.
- Automatic worker capacity distribution and isolation of tuning RNG, module state, and cross-device tensor copies.
- Corrected the base dependency range to `witwin>=0.4,<0.5`, matching the published `PhysicalMaterial` contract instead of an unavailable version range.
- Preserved local compilation reuse when declared design replacements acquire new base-package structure/material identity tokens.
- Declared plotting/progress/notebook display import dependencies and extended release checks to installed-package imports and native Batch allocator exports.

### Limitations

- Batch remains experimental with a narrower capability set than ordinary FDTD. Fused batch reverse kernels, exact preplanning of all retained outputs, and exclusive multi-GPU throughput acceptance remain pending.
- See [0.5.1 release notes](docs/reference/release-notes-0.5.1.md) for validation evidence and the supported execution contract.

## [0.5.0] - Unpublished

- The version tag built CUDA wheels but failed the installed-package import gate because `IPython` was undeclared. No distributions were uploaded to PyPI.

## [0.4.0] - 2026-07-23

### Added

- RF port, network, antenna, phased-array, thin-wire, electrostatic, capacitance, SAR, exposure, ESD, dielectric-breakdown, gyromagnetic, and generalized surface-impedance workflows.
- Distributed forward and adjoint execution, including CPML-trainable reverse paths and deterministic multi-device scene-gradient aggregation.

### Changed

- Moved material sampling to native Yee-component locations, improving the geometry validation cluster while changing geometry-resolved numerical baselines.
- Published one PyTorch Stable ABI CUDA binary per platform for PyTorch 2.10 and newer, load-tested through PyTorch 2.13 and CPython 3.10-3.14.
- Built Linux wheels against `manylinux_2_28`; Linux and Windows CUDA fat binaries include native SM87 SASS.
- Made version tags drive the gated package publication and release-artifact workflow.

### Fixed

- Closed release-blocking CUDA graph, distributed checkpoint, conformal PEC, observer staggering, near-to-far quadrature, modal selection, and surface-impedance correctness gaps.

Complete capability boundaries, validation evidence, behavior changes, and known limitations are documented in the [0.4.0 release notes](docs/reference/release-notes-0.4.0.md).

## [0.3.0] - 2026-07-13

### Added

- Native CUDA FDTD adjoint reverse paths for standard and CPML fields, complex Bloch fields, full electric anisotropy, electric conductivity and dispersion, Kerr nonlinearity, TFSF sources, and multi-source workflows.
- Broadband and arbitrary-normal-axis grating `TFSF.slab(...)`, including pulsed Bloch injection and mixed Bloch/CPML axis permutations.
- `Medium2D`, `Graphene`, `LossyMetalMedium`, `PerturbationMedium`, custom spatial dispersion, Sellmeier and gain media, nonlinear susceptibility, two-photon absorption, space-time modulation, magnetic conductivity/dispersion, and full `Tensor3x3` electric anisotropy.
- Composition of supported anisotropic, conductive, dispersive, nonlinear, modulated, and CPML material effects in the same compiled FDTD scene.
- Broadband, lossy, diagonal-anisotropic, and bent-waveguide mode-source/port workflows.
- Background-aware and curved-surface equivalent-current near-to-far propagation, directivity, bistatic RCS, and expanded S-parameter/modal-overlap processing.
- Tidy3D export for nonuniform grids, mixed boundaries, expanded geometry, material, source, and monitor families.
- CUDA graph capture coverage for standard, Bloch, dispersive, nonlinear, and TFSF stepping paths.
- Native CUDA SIBC updates, full-anisotropic conductive-current capture, and a device-resident modulation clock, all compatible with CUDA Graph replay.
- Python 3.10-3.14 and PyTorch 2.10-2.12 validation for the packaged LibTorch Stable ABI CUDA extension.

### Changed

- Consolidated the public architecture around the declarative `Scene -> Simulation -> Result` workflow. FDTD is the primary full-featured runtime; FDFD uses the same public model but remains a limited/experimental CUDA-only single-frequency path.
- Moved FDTD adjoint hot paths from Python/Torch replay to native CUDA kernels while keeping PyTorch autograd as the public differentiation interface.
- Generalized automatic/custom nonuniform grids, subpixel averaging, conformal PEC placement, and source phase correction to use local physical grid spacing.
- Standardized physical-domain handling so PML cells are appended outside `Domain.bounds`; material/source/monitor validation and cross-solver comparisons use the unchanged physical bounds.
- Adopted the WiTwin dual-license model and release wheels with CUDA 12.8 fat binaries for Linux and Windows.

### Fixed

- Aligned Maxwell and Tidy3D plane-wave and TFSF amplitudes using their physical power/field definitions instead of fitted dimensionless factors.
- Converted explicit Bloch vectors from Maxwell radians/metre to Tidy3D's dimensionless `k * period / (2*pi)` convention.
- Preserved requested field-monitor components during Tidy3D export and validation-cache loading, preventing undeclared near-zero components from being selected as scalar observables.
- Corrected external-PML SIBC half-space detection and low-side Yee surface-node placement for `LossyMetalMedium`.
- Corrected Gaussian-pulse phase/reference handling, CW export ramp bandwidth, source-spectrum normalization, flux-aperture integration, and coordinate-aligned complex-field comparison.
- Replaced invalid validation scenarios with physically supported TFSF, diffraction-monitor, cavity-probe, and closed-surface near-to-far workflows.
- Added visual slice/phase diagnostics and solver-independent scalar comparisons for resonances, diffraction orders, directivity/beamwidth, RCS, and S-parameters.
- Corrected symmetry-face PML removal, high-face grid anchoring, and Yee-control-volume point-dipole image scaling.
- Corrected singleton custom-current datasets on staggered Yee components so magnetic current samples compile to the nearest component location.
- Kept the stable-ABI CUDA extension source portable across MSVC and Linux NVCC by using explicit typed input tables in native anisotropic-current validation.

Detailed numerical conventions, pitfalls, final comparison values, and intentionally unresolved operator-level residuals are documented in [the 0.3.0 Tidy3D numerical-alignment notes](docs/validation/tidy3d-numerical-alignment-0.3.0.md).

## [0.2.0] - 2026-07-08

- Previous public release.

[0.6.0]: https://github.com/witwin-ai/witwin-maxwell/compare/witwin-maxwell-v0.5.1...witwin-maxwell-v0.6.0
[0.5.1]: https://github.com/witwin-ai/witwin-maxwell/compare/witwin-maxwell-v0.4.0...witwin-maxwell-v0.5.1
[0.4.0]: https://github.com/witwin-ai/witwin-maxwell/compare/witwin-maxwell-v0.3.0...witwin-maxwell-v0.4.0
[0.3.0]: https://github.com/witwin-ai/witwin-maxwell/compare/witwin-maxwell-v0.2.0...witwin-maxwell-v0.3.0
[0.2.0]: https://github.com/witwin-ai/witwin-maxwell/releases/tag/witwin-maxwell-v0.2.0
