# Changelog

All notable user-visible changes to WiTwin Maxwell are documented here. The project follows semantic versioning within the `0.x` development series.

## [Unreleased]

## [0.5.0] - 2026-09-08

### Added

- Experimental `Simulation.batch(...)` execution with reusable capacity-limited CUDA slots, batched field/CPML/source/observer updates, and ordinary owned `Result` objects.
- Local material compilation reuse, automatic resident capacity, private per-device CUDA allocation budgets, multi-GPU and dual-slot scheduling, and opt-in measured throughput tuning.
- PyTorch-native Batch checkpoint adjoints for supported density, dielectric geometry, and named source-scale inputs, including delayed backward after slot reuse or close.
- Reproducible Batch performance, memory, material-compilation, and device-scaling audits with numerical controls and process-contention records.

### Fixed

- Material-aware time-step stability, source timing, complex CPML and Mur boundary histories, and capture/resume state consistency in FDTD.
- Batch parameter cache identity, iterator ownership, result scene snapshots, selected electric spectra, and redundant spectral allocations in retained adjoint state.
- Automatic worker capacity distribution and isolation of tuning RNG, module state, and cross-device tensor copies.

### Limitations

- Batch remains experimental with a narrower capability set than ordinary FDTD. Fused batch reverse kernels, exact preplanning of all retained outputs, and exclusive multi-GPU throughput acceptance remain pending.
- See [0.5.0 release notes](docs/reference/release-notes-0.5.0.md) for validation evidence and the supported execution contract.

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

[Unreleased]: https://github.com/witwin-ai/witwin-maxwell/compare/witwin-maxwell-v0.5.0...HEAD
[0.5.0]: https://github.com/witwin-ai/witwin-maxwell/compare/witwin-maxwell-v0.4.0...witwin-maxwell-v0.5.0
[0.4.0]: https://github.com/witwin-ai/witwin-maxwell/compare/witwin-maxwell-v0.3.0...witwin-maxwell-v0.4.0
[0.3.0]: https://github.com/witwin-ai/witwin-maxwell/compare/witwin-maxwell-v0.2.0...witwin-maxwell-v0.3.0
[0.2.0]: https://github.com/witwin-ai/witwin-maxwell/releases/tag/witwin-maxwell-v0.2.0
