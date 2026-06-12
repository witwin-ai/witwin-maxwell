# Feature List

This document tracks the current user-visible capabilities of the `maxwell` package.

## Maintenance Rule

- Update this file whenever a new user-visible feature, public API capability, or meaningful workflow is added, removed, or materially changed.
- Keep entries focused on capabilities that a user can discover through the public package, examples, or standard solver workflows.

## Public API Model

- Declarative simulation workflow: `Scene -> Simulation -> Result`
- PyTorch-native scene-module workflow: `SceneModule -> Simulation -> Result`, with automatic `to_scene()` normalization inside `Simulation`
- Top-level package exports for Maxwell-specific public objects under `maxwell`, including the full-featured public `Material`
- Public simulation entrypoints: `Simulation.fdfd(...)`, `Simulation.fdtd(...)`, and `run(...)`
- Typed simulation configuration records and enums exposed as `FDFDConfig`, `FDTDConfig`, `TimeConfig`, `SpectralSampler`, `SimulationMethod`, `SpectralWindowKind`, and `AbsorberKind`
- Public frequency vocabulary uses `frequency=` for scalar selection and `frequencies=` for one-or-many target frequencies; `freqs` is not exposed in public Maxwell APIs
- Public result container with structured field/material access (`result.E.x`, `result.materials.eps.scalar`), frequency-scoped selection through `Result.at(...)`, monitor access, stats, plotting, and save support

## Scene Construction

- `Domain` with explicit 3D bounds and `from_domain_range(...)`
- `GridSpec.uniform(...)` for isotropic grids
- `GridSpec.anisotropic(...)` for nonuniform `dx`, `dy`, `dz`
- `GridSpec.spacing` always returns `(dx, dy, dz)`, with `GridSpec.is_uniform` for uniform-grid checks
- `BoundarySpec.none()`, `BoundarySpec.pml(...)`, `BoundarySpec.periodic()`, `BoundarySpec.bloch(...)`, `BoundarySpec.pec()`, and `BoundarySpec.pmc()`
- Per-face boundary configuration through `BoundarySpec.faces(...)` and direct `BoundarySpec(kind=..., x=..., x_low=..., ...)` overrides, including global defaults plus per-axis or per-face specialization
- Public `BoundaryKind` literal type for boundary-mode selection across declarative scene APIs
- Shared `witwin.core.Structure` records pairing geometry and material, with `priority`, `enabled`, and `tags`
- `Geometry.with_material(...)` convenience path returns a shared `Structure`
- `MaterialRegion` for density-based PyTorch-native material overlays on the scene grid
- Scene assembly with `Scene.add_structure(...)`, `Scene.add_source(...)`, `Scene.add_monitor(...)`, and `Scene.add_material_region(...)`
- Scene-level modal port assembly with `Scene.add_port(...)`, plus `Scene.resolved_sources()` / `Scene.resolved_monitors()` to materialize first-class `ModePort` objects into the standard `Scene -> Simulation -> Result` workflow
- `Scene.clone(...)` for scene-preserving validation, benchmarking, and device-transfer workflows
- `Scene(...)` defaults to `device="cuda"` and requires an explicit `device="cpu"` override for scene-only CPU workflows
- Optional low-face symmetry specification on `Scene(symmetry=(..., ..., ...))` with per-axis `None` / `"PEC"` / `"PMC"`
- Optional subpixel material averaging via `Scene(subpixel_samples=...)`, accepting only `int` or `(sx, sy, sz)`
- Solver-side compiled scene inspection via `Simulation.prepare()`, where `prepared.solver.scene` materializes Yee-grid dimensions, lazy meshgrid allocation, material compilation, and orthogonal material cross sections without storing that state on the public `Scene`

## Geometry

- Analytic geometry primitives:
  `Box`, `Sphere`, `Cylinder`, `Ellipsoid`, `Cone`, `Pyramid`, `Prism`, `Torus`, `HollowBox`
- Shared core geometry constructors use `position=...` consistently across analytic primitives, `Mesh`, and `SMPLBody`
- Shared core geometry constructors default to `device=None`, while `Scene(...)` owns device placement and defaults to CUDA
- Rotation support for applicable analytic geometries
- Axis selection for directional geometries such as `Cylinder`, `Cone`, `Pyramid`, `Prism`, and `Torus`
- Triangle mesh geometry via `Mesh(vertices, faces, ...)`
- OBJ mesh loading via `Mesh.from_obj(...)`
- `Geometry.to_mesh(...)` returns torch-native vertex and face tensors, with faces standardized to `torch.int64`
- Mesh transforms: recentering, scaling, rotation, translation
- Mesh topology inspection: vertex count, face count, boundary edge count, non-manifold edge count, degenerate face count, inconsistent edge-orientation count, watertight flag
- Mesh fill modes: `auto`, `solid`, and `surface`

## Materials

- Public `Material(eps_r, mu_r, sigma_e, name=None)` in `witwin.maxwell`, extending the shared `witwin.core.Material` contract with Maxwell-specific constitutive behavior
- Optional electric and magnetic dispersive pole models on `Material`: `DebyePole`, `DrudePole`, and `LorentzPole`
- Dispersive poles expose `susceptibility(angular_frequency)` and `susceptibility_at_freq(frequency)` for explicit frequency-domain evaluation
- Axis-aligned diagonal anisotropy on `Material` through `DiagonalTensor3` for `epsilon_tensor`, `mu_tensor`, and `sigma_e_tensor`
- `Material.relative_permeability(frequency)` for isotropic magnetic dispersion evaluation
- Instantaneous isotropic Kerr nonlinearity on `Material(kerr_chi3=...)`
- Optional anisotropic Maxwell material descriptors: `DiagonalTensor3` and `Tensor3x3`
- Convenience constructors `Material.debye(...)`, `Material.drude(...)`, and `Material.lorentz(...)`
- Frequency-dependent material evaluation through the shared material compiler for single-frequency workflows, including isotropic `sigma_e`
- Component-aware material compilation keeps scalar summary tensors for visualization / compatibility while exposing explicit per-axis material grids for solver backends and `Result.material(...)`
- Density-based `MaterialRegion` compilation using native PyTorch tensor interpolation, optional box filtering, and optional projection
- Vacuum background by default (`eps_r = 1`, `mu_r = 1`)
- Multi-structure material composition on the scene grid
- Structure overlap resolution uses `Structure.priority` first, then append order among equal-priority structures
- Occupancy-based material blending on the scene grid, with SDF-driven soft occupancy for shared `Box`, `Sphere`, `Cylinder`, `Torus`, and `HollowBox`
- Phase-1.5 primitive SDF coverage for shared `Ellipsoid`, `Cone`, `Pyramid`, and `Prism`
- Differentiable mesh occupancy compilation through shared mesh signed-distance evaluation, including Slang-accelerated CUDA forward/backward distance-sign queries for watertight solid fill, geometry-state-aware static mesh SDF caching, cached BVH acceleration for larger static CUDA meshes, and shared surface-band modes
- Supersampled voxel averaging for smoother material interfaces on partial cells

## Sources

- `CW`, `GaussianPulse`, and `RickerWavelet` source-time definitions
- `PointDipole` source definition with `source_time` and selectable `profile="gaussian"|"ideal"`
- `PointDipole`, `ModeSource`, `ModeMonitor`, and `ModePort` use `position=...` as the public spatial-location argument
- `PointDipole` width compensation keeps narrow-source injected strength consistent with the default regularization width in FDTD/FDFD workflows
- `PointDipole(profile="ideal")` collapses injection to the nearest Yee sample with a calibrated discrete strength shared by FDTD and FDFD
- `PlaneWave` soft source for analytical plane-wave injection on an auto-placed source plane
- `PlaneWave` soft source uses a single-plane directional `E/H` equivalent-current injector with calibrated Tidy3D-style incident-power scaling
- `GaussianBeam` soft source for analytical Gaussian-beam injection with configurable waist and focus
- Experimental `ModeSource` soft source for axis-aligned FDTD waveguide launching, using a full-vector generalized 2D eigenmode solve for forward source-plane assembly on real isotropic apertures, dense and sparse forward backends for that generalized solve, and a retained experimental torch-differentiable scalar eigensolve path for current trainable FDTD scenes
- Experimental `TFSF(bounds=...)` injection descriptor for `PlaneWave` and `GaussianBeam`, with validated axis-aligned `PlaneWave` support for `CW` and `GaussianPulse`, and validated CW oblique `PlaneWave` support
- CUDA `PlaneWave` TFSF forward stepping uses Slang auxiliary-line updates and fused patch-application kernels to reduce per-step launch overhead
- `GaussianBeam` `TFSF` remains experimental and currently uses the analytical profile provider rather than the future angular-spectrum / discrete-face engine
- Polarization specified by field name (`"Ex"`, `"Ey"`, `"Ez"`) or explicit 3-vector
- Source amplitude and phase carried by `source_time`; spatial sources keep width / beam parameters and optional name
- Multiple sources per scene
- Source compilation uses `compile_fdfd_sources(...)` / `compile_fdtd_sources(...)` list-based interfaces across both solvers

## Monitors

- `PointMonitor` for point sampling of selected electric or magnetic field components
- `PlaneMonitor` for orthogonal plane sampling of selected electric or magnetic field components
- `FinitePlaneMonitor` for first-class finite rectangular plane sampling with explicit `position=(x, y, z)` and `size=(sx, sy, sz)` on a zero-thickness axis
- `ClosedSurfaceMonitor` for first-class finite closed Huygens-surface workflows, including `ClosedSurfaceMonitor.box(...)` and custom multi-face axis-aligned surfaces built from `FinitePlaneMonitor` faces
- `FluxMonitor` for plane-integrated power / flux extraction from tangential `E/H` fields
- Experimental `ModeMonitor` for first-class modal decomposition on an axis-aligned port plane, reusing the current `ModeSource` mode specification and returning forward / backward modal amplitudes and power through `Result.monitor(...)`
- Optional per-monitor `frequencies=` on plane and modal monitors / ports
- Named monitor results returned through the unified `Result` object
- Multi-component plane monitors with aggregated `Result.monitor(...)` payloads and collocated tangential grids for postprocessing workflows
- `Scene.resolved_monitors()` expands `ClosedSurfaceMonitor` into its underlying finite face monitors while preserving the public `Scene -> Simulation -> Result` workflow
- Multi-frequency monitor output with `Result.monitor(name, frequency=...)` / `freq_index=...` selection
- Optional `compute_flux=True` on `PlaneMonitor` to emit integrated Poynting flux / power per frequency
- `Result.raw_monitor(...)` to access the underlying point / plane payload directly when a first-class modal monitor resolves to a higher-level modal result

## Ports

- Experimental `ModePort` scene object that declaratively couples an optional `ModeSource` excitation with a first-class `ModeMonitor`, so modal ports still flow through the same `Scene -> Simulation -> Result` public architecture
- `ModePort(source_time=...)` materializes a named `ModeSource` plus a named modal monitor; `ModePort(source_time=None)` acts as a monitor-only modal port
- `ModePort(monitor_offset=...)` can separate the launch plane from the sampled monitor plane along the port normal without introducing a second public solver entrypoint

## FDFD Solver Workflow

- Frequency-domain simulation through `Simulation.fdfd(...)`
- CUDA-only solver execution; `Simulation.fdfd(...)` requires `Scene(device="cuda")`
- Single-frequency public wrapper with `frequency=`
- Dispersive-material support via effective complex `epsilon_r(omega)` at the simulation frequency
- Isotropic conductive-material support via `sigma_e` folded into effective complex `epsilon_r(omega)`
- Axis-aligned diagonal electric anisotropy and diagonal `sigma_e_tensor` support in the Yee-grid operator
- Per-face `none` / `pml` boundary selection, including one-sided and mixed-axis PML layouts
- Explicit fast-fail validation for unsupported magnetic response and Kerr media
- Configurable GMRES settings via `GMRES(max_iter, tol, restart, solver_type)`
- Typed FDFD solver configuration through `FDFDConfig(solver=..., enable_plot=..., verbose=...)`
- Prepared execution via `Simulation.prepare()` before running
- Unified `Result` output containing `Ex`, `Ey`, and `Ez`
- Solver stats including convergence flag, solver info, residual, and solver configuration

## FDTD Solver Workflow

- Time-domain simulation through `Simulation.fdtd(...)`
- CUDA-only solver execution; `Simulation.fdtd(...)` requires `Scene(device="cuda")`
- Single- or multi-frequency DFT extraction through `frequency=` or `frequencies=[...]`
- Source temporal frequency (`source_time.frequency`) remains distinct from simulation / monitor extraction frequencies; Maxwell does not infer extraction frequencies implicitly
- ADE-based electric and magnetic dispersive-material updates for Debye, Drude, and Lorentz media
- Axis-aligned diagonal anisotropy support for electric and magnetic material tensors on the Yee grid
- Instantaneous isotropic electric Kerr nonlinearity with GPU-resident dynamic update coefficients
- Automatic run length estimation with `TimeConfig.auto(...)`
- Explicit run-step control with `TimeConfig(time_steps=...)`
- Automatic `dt` tightening for broadband source-time objects such as `GaussianPulse` and `RickerWavelet`, and for electric or magnetic dispersive material poles such as `Drude`, `Debye`, and `Lorentz` media
- CPML absorber configuration through `BoundarySpec.pml(...)` plus typed `Simulation.fdtd(...)` config (`absorber=...`, `cpml_config=...`)
- CPML auxiliary `psi` storage auto-selects between a dense fast path and slab-allocated low-memory storage, with `cpml_config={"memory_mode": "auto"|"dense"|"slab"}` and optional `dense_memory_limit_mib` tuning
- Non-absorbing FDTD boundary conditions: periodic, Bloch phase-shifted periodic, PEC, and PMC
- Per-face FDTD boundary selection across `pml`, `periodic`, `pec`, `pmc`, and `none`, including mixed-axis combinations such as periodic-in-`y` plus PML-in-`x/z`
- Mixed low-face symmetry plus high-face absorber workflows through `Scene(symmetry=...)` with `BoundarySpec.pml(...)`
- Spectral window and normalization configuration through `SpectralSampler(window=..., normalize_source=...)`
- Pulse-driven spectral extraction starts at the transient without steady-state apodization for `GaussianPulse` and `RickerWavelet`, while CW extraction still skips startup transients
- Optional prepared execution via `Simulation.prepare()`
- Optional full-field DFT output, including simultaneous accumulation of multiple target frequencies in one run
- Single-frequency FDTD runs that do not request full-field DFT now return the last-step Yee fields instead of auto-enabling full-domain DFT work
- Full-field FDTD DFT results stay in native PyTorch tensors through `Simulation` / `Result`, with NumPy conversion deferred to plotting, export, and validation boundaries
- Selective monitor/observer extraction for point and plane monitors
- Simultaneous multi-frequency point / plane observer accumulation in one run
- Point, plane, and flux monitor payloads stay torch-native through `Result.monitor(...)`, including multi-frequency selection
- First-class modal monitor and port results remain torch-native through `Result.monitor(...)`, while the underlying raw plane payload remains available through `Result.raw_monitor(...)`
- GPU-accelerated soft source injection for `PointDipole`, `PlaneWave`, `GaussianBeam`, and experimental `ModeSource`
- Reverse-time FDTD adjoint for trainable scene inputs that contribute to prepared-scene material tensors, including analytic structure-geometry parameters and density-based `MaterialRegion` design tensors, through the existing `Scene` / `SceneModule -> Simulation -> Result` API, including source pullback for `PointDipole`, soft `PlaneWave`, soft `GaussianBeam`, experimental `ModeSource` replay on the structured reverse backends through an explicit source-term VJP, `TFSF` `PlaneWave` / `GaussianBeam` replay, static ADE dispersive media (`Debye`, `Drude`, `Lorentz`), and Bloch-boundary complex-field replay for nondispersive scenes
- Internal checkpoint/replay support for adjoint-enabled FDTD runs, including CPML auxiliary state replay, TFSF auxiliary-line replay, and Bloch real/imag field checkpoints
- Adjoint gradient pullback from Yee-grid electric permittivity coefficients back through `Scene.compile_material_tensors()` into trainable material-graph inputs
- Solver stats including time steps, `dt`, absorber, requested frequencies, per-frequency DFT sample counts, elapsed time, milliseconds per step, and steps per second
- Native CUDA extension builds on Windows can discover and load the Visual Studio x64 build environment automatically for accelerated FDTD kernels
- Native CUDA extension builds resolve conda-distributed torch import libraries automatically, and `WITWIN_MAXWELL_FDTD_CUDA_PREBUILT=1` loads the already-built extension without invoking the build toolchain (required under profilers such as Nsight Systems)
- Native CUDA CPML field updates skip full-volume coefficient reads when the decay/curl coefficient tensors are spatially uniform, detected automatically once per solve (about 1.4x faster forward stepping on homogeneous scenes)
- The native CUDA module surface accepts strided tensor views, so the reverse-time adjoint with `WITWIN_MAXWELL_FDTD_ADJOINT_BACKEND=slang` runs entirely on native CUDA reverse kernels for standard, CPML, TFSF, and Bloch scenes (about 1.7x faster backward than the default python-reference reverse path)
- FDTD result stats include CPML auxiliary-memory mode and allocated-versus-dense `psi` byte counts
- Support for odd grid sizes in Yee-component field outputs

## Result Handling

- Structured field lookup through `result.E.x`, `result.E.y`, `result.H.z`, and frequency/symmetry-scoped result views from `Result.at(...)`
- Monitor lookup through `Result.monitor(...)`
- Raw monitor lookup through `Result.raw_monitor(...)`
- `Result.monitor(...)` and `Result.raw_monitor(...)` reassemble first-class `ClosedSurfaceMonitor` payloads from the resolved finite face monitors, and finite-plane payloads expose cropped coordinates plus face metadata
- Frequency selection on structured field and monitor views through `Result.at(frequency=...)` or `Result.at(freq_index=...)`
- Structured material lookup through `result.materials.eps.scalar`, `result.materials.eps.x`, and `result.materials.mu.z`
- Low-level tensor escape hatches through `Result.tensor(...)` and `Result.material(...)`
- Optional symmetry-expanded field and material access via `Result.at(expand_symmetry=True)`
- Summary metadata through `Result.stats()`
- Field and material plotting through `Result.plot.field(...)` and `Result.plot.material(...)`
- Serialized save via `Result.save(...)`
- Access to raw backend output through `Result.raw_output`

## Postprocessing

- `maxwell.postprocess.PlanarEquivalentCurrents` for planar Huygens-surface current data
- `maxwell.postprocess.EquivalentCurrentsSurface` for multi-plane or closed-surface Huygens current collections
- `maxwell.postprocess.equivalent_surface_currents_from_fields(...)` to build equivalent currents from tangential `E/H` plane fields
- `maxwell.postprocess.equivalent_surface_currents_from_monitor(...)` to bridge a plane or first-class `ClosedSurfaceMonitor` on `Result` directly into planar or multi-surface equivalent currents
- `maxwell.postprocess.equivalent_surface_currents_from_monitors(...)` to assemble multi-plane or closed-surface Huygens currents directly from multiple `PlaneMonitor` outputs, with shared or per-monitor tangential cropping for finite closed surfaces
- `maxwell.postprocess.StrattonChuPropagator` for exact near-field propagation from planar equivalent currents
- `StrattonChuPropagator` defaults to CUDA and requires an explicit `device="cpu"` override for CPU postprocessing
- `maxwell.postprocess.NearFieldFarFieldTransformer` for planar near-field to far-field projection
- Postprocessing outputs stay torch-native end-to-end: equivalent currents, Stratton-Chu propagation, NFFT far fields, directivity, bistatic RCS, and flux-derived S-parameters preserve `torch.Tensor` results and autograd instead of detaching through NumPy
- `maxwell.postprocess.compute_bistatic_rcs(...)` and `transform_to_bistatic_rcs(...)` for bistatic RCS derived from NFFT far fields and PlaneWave incident amplitude
- `maxwell.postprocess.compute_directivity(...)` for radiation intensity, total radiated power, directivity, gain, radiation efficiency, and 3 dB beamwidth extraction from NFFT far fields
- `maxwell.postprocess.compute_s_parameters(...)` for broadband `S11` / `S21` extraction from flux-monitor results using reference-run or explicit incident-power normalization
- Experimental `maxwell.postprocess.compute_mode_overlap(...)` for plane-monitor or first-class `ModeMonitor` / `ModePort` modal decomposition against an axis-aligned scalar `ModeSource` reference, returning forward/backward modal amplitudes and power fractions
- End-to-end validation coverage for first-class finite closed-surface Huygens workflows in homogeneous free space: rectangular-box dipole near-field reconstruction/directivity, non-rectangular orthogonal-surface dipole directivity, and TFSF Rayleigh-sphere bistatic RCS are regression-tested against analytic or direct-FDTD references under `tests/validation/physics/test_postprocess_end_to_end_validation.py`
- First-class closed-surface postprocessing now rejects layered or otherwise inhomogeneous exterior media explicitly instead of silently applying a free-space Green's function outside its validity range

## Tidy3D Adapter

- `Scene.to_tidy3d(frequencies=..., run_time=..., **kwargs)` converts a maxwell Scene to a `tidy3d.Simulation` for cloud execution and cross-validation
- Adapter module at `maxwell/adapters/tidy3d.py` with optional `tidy3d` import
- `GaussianPulse` export carries Maxwell's carrier delay into the Tidy3D source phase so single-frequency pulse benchmarks compare against the same source spectrum
- `PlaneWave` export places the source inside the physical interior and uses Tidy3D's infinite-plane aperture convention instead of a finite rectangular source patch
- Supported mappings: Domain, GridSpec, BoundarySpec (uniform or per-face PML/periodic/PEC/PMC/Bloch export), Structure (Box/Sphere/Cylinder/Cone), Material (simple conductive / Drude / Lorentz / Debye / mixed PoleResidue with `mu_r = 1`), PointDipole, PlaneWave, GaussianBeam, PointMonitor, PlaneMonitor, FinitePlaneMonitor, FluxMonitor, symmetry

## Tidy3D Benchmarking

- `benchmark/` package as the unified Maxwell-vs-Tidy3D benchmarking entrypoint
- One-file-per-scenario definitions organized under `benchmark/scenes/dipole/` and `benchmark/scenes/planewave/`
- Predefined ~128^3-grid benchmark scenarios covering vacuum dipoles, plane-wave slab/sphere scattering, dispersive resonators, multi-dielectric scenes, and `dipole_dielectric_sphere`
- HDF5-based Tidy3D reference caching under `benchmark/cache/`
- Benchmark cache validation keyed to the exported Tidy3D scene configuration so stale reference data is regenerated automatically
- Benchmark-side Tidy3D exports preserve the full simulation domain and crop field/flux comparisons back to Maxwell's physical interior during analysis
- Shared benchmark scene helpers derive safe flux-monitor positions from the current PML-trimmed physical interior
- Error metrics: relative L2 error, relative L-infinity error, normalized cross-correlation, and flux relative error
- Coordinate-aware plane-field alignment and interpolation onto the Tidy3D reference grid for cross-solver field comparison
- Flux benchmark comparison re-integrates Maxwell monitor fields over the same PML-trimmed physical aperture used by the Tidy3D reference export
- Material and source comparison plots align slices by physical coordinates, use the same soft PlaneWave injection-plane placement as the runtime/Tidy3D export, and geometry voxelization uses boundary tolerances to avoid one-pixel drift on benchmark domains
- Auto-generated Maxwell-vs-Tidy3D permittivity/source comparison plots plus `Ex/Ey/Ez` field comparisons on `x/y/z` cut planes under `benchmark/plots/`
- Auto-updated benchmark summary in `benchmark/RESULTS.md`, grouped by benchmark scene folder with per-metric better-direction and target-range guidance
- Plane-wave benchmark runs enable source-spectrum normalization on the Maxwell side so pulsed `PlaneWave` scenarios compare against the same frequency-domain reference convention as the Tidy3D export

## Validation and Reference Utilities

- Shared material compiler used by scene construction and backend preparation
- Reference-alignment helpers for FDTD field comparisons
- Error metrics for comparison against reference fields
- Direct physics validation suites under `tests/validation/physics/` for vacuum plane-wave and dipole correctness, dielectric slab energy balance, boundary-condition validation (`periodic`, `Bloch`, `PEC`, `PMC`, mixed `periodic + PML`, `CPML`), and TFSF leakage/scatter validation for both axis-aligned and oblique CW plane waves
- Validation workflows emit representative electric-field and centerline plots under `tests/test_output/validation/` in addition to pass/fail assertions
- Test coverage for public API, scene construction, mesh geometry, material compilation, CPML, observer extraction, and FDFD/FDTD consistency

## Known Limitations

- Current subpixel material smoothing uses scalar arithmetic averaging on the scene grid
- Yee-edge tangential versus normal permittivity averaging is not yet implemented
- High-contrast interfaces can therefore show larger normal-field error than a true edge-aware subpixel scheme
- Maxwell supports axis-aligned diagonal anisotropy through `DiagonalTensor3`, but `Material.orientation` and full `Tensor3x3` rotation / off-diagonal tensors remain unsupported
- Kerr media cannot be combined with dispersive or anisotropic materials in the same scene in v1
- FDFD supports electric anisotropy only; static magnetic media and magnetic dispersion still fail explicitly
- FDTD rejects static conductive `sigma_e` materials explicitly; use FDFD for frequency-domain conductivity or Maxwell dispersive poles for time-domain media
- Mixed Bloch boundary configurations currently fail explicitly in FDTD; use homogeneous `BoundarySpec.bloch(...)` or avoid Bloch on mixed boundary layouts
- FDFD currently supports per-face boundary mixing only for `none` and `pml`
- Tidy3D export and the FDTD adjoint bridge reject anisotropy, magnetic dispersion, and Kerr media explicitly in v1
- Experimental `ModeSource`, `ModeMonitor`, and `ModePort` are currently limited to FDTD `CW` soft injection / monitoring on an axis-aligned plane and no Tidy3D mode-source export support; forward mode solving now uses a full-vector generalized eigenproblem on real isotropic source apertures and can handle non-unit `mu_r`, but the differentiable path still inherits the older scalar restrictions, and the broader modal stack is still missing complex/lossy, anisotropic, bent, angled, and RF-style wave-port coverage
- Experimental `compute_mode_overlap(...)` currently expects aligned tangential `E/H` fields on the target port plane and inherits the same axis-aligned modal-source assumptions as the current `ModeSource` implementation
- Raw single-plane postprocessing from `PlaneMonitor` data is still not a general 3D radiator/scatterer workflow; use `FinitePlaneMonitor` / `ClosedSurfaceMonitor` for the validated near/far/RCS/directivity path
- First-class closed-surface postprocessing currently requires a homogeneous exterior medium immediately outside every face; layered or otherwise inhomogeneous exteriors are rejected explicitly
- Closed-surface validation currently covers axis-aligned planar faces, including rectangular Huygens boxes and non-rectangular orthogonal polyhedra; curved/non-planar closed surfaces still need a separate workflow and validation path
- Benchmark soft-plane-wave absolute calibration still needs separate validation/debugging and remains a known open correctness issue
