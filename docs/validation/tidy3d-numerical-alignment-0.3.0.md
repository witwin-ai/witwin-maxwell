# Maxwell-Tidy3D Numerical Alignment for 0.3.0

This note records the numerical improvements, diagnostic mistakes, and general fixes made during the Maxwell 0.3.0 cross-solver validation campaign. It is intentionally more detailed than the release changelog so future work does not repeat the same false starts.

## Validation principles

The campaign used the following rules:

1. Compare the same physical problem, not merely similar array shapes.
2. Keep `Domain.bounds` as the physical domain and append PML outside it in both solvers.
3. Preserve Maxwell's requested `0.025 m` maximum main-grid step. For the `1.28 m` comparison domains, Tidy3D resolves 52 equal physical cells, giving approximately `0.024615 m` per cell.
4. Compare fields after coordinate alignment and cropping to the shared physical domain.
5. Remove at most one unit-magnitude global source-reference phase. Never fit amplitude, local phase, or a scenario-specific correction factor.
6. Inspect field magnitude, real part, raw/aligned phase, phase difference, and center lines before attributing an error to a boundary or material model.
7. Pair field norms with a physical scalar observable such as resonance frequency, diffraction efficiency, directivity, RCS, flux, or an S-parameter.

FDFD was deliberately excluded from this campaign. It is not feature-complete and was neither used as a numerical reference nor treated as a 0.3.0 release gate; all cross-solver statements in this note mean FDTD versus Tidy3D.

## General improvements

### Physical domain and PML placement

Maxwell now keeps the declared physical domain unchanged while the prepared FDTD scene appends absorber cells outside it. Cross-solver analysis crops external PML samples before interpolation and comparison.

This convention also fixed `LossyMetalMedium` compilation. A metal slab flush with a physical domain face must remain a one-face half-space even after preparation adds PML nodes beyond that face. SIBC coverage is therefore tested against `scene.domain.bounds`, not the prepared array extents. The low-side surface uses the actual upper geometry-face Yee node rather than the last interior metal-cell index.

### Source normalization and phase

The source paths were aligned from physical definitions:

- Soft `PlaneWave` normalization derives the absolute power scale from the equivalent-current forward gain, the Yee numerical impedance, aperture area, incidence angle, and the E/H half-cell Poynting correction.
- Gaussian-pulse export preserves physical amplitude and phase while mapping envelope delay to Tidy3D's source offset. Source-spectrum normalization removes the pulse-envelope reference, not propagation phase.
- Maxwell TFSF amplitude is an electric field in V/m. Tidy3D TFSF uses a fixed `1 W/um^2` incident normalization, whose unit electric field is `sqrt(2 / (c * eps0)) V/um`. The adapter now converts between these definitions explicitly.
- TFSF validation removes only one global unit phasor. An earlier amplitude rescaling path effectively normalized twice and was removed.
- Tidy3D continuous-wave export uses a deterministic `fwidth = 0.1 * frequency` turn-on bandwidth, and the benchmark cache contract tracks that export behavior.

### Bloch and periodic boundaries

Maxwell stores explicit Bloch wavevectors in radians/metre. Tidy3D expects a dimensionless phase-per-period value. Each exported component is now converted independently as

`bloch_vec = k_axis * physical_period_axis / (2 * pi)`.

Automatic Bloch metadata is still rejected during direct export unless solver preparation has resolved it. This is safer than silently guessing a period or frequency.

Zero-phase periodic boundaries are accepted by soft surface sources only on axes transverse to propagation. Propagation-axis periodicity, a nonzero tangential phase advance, and Bloch illumination use the grating TFSF path.

### Monitor component contracts

Tidy3D field monitors can expose all six E/H components by default. Maxwell monitors have an explicit public `fields=(...)` contract. The adapter now passes that exact component list, and cached reference payloads are filtered again to the requested public fields before scalar or field postprocessing.

This fixed the most misleading Group 7 failure: the PMC cavity requested `Hz`, but scalar selection saw a near-zero undeclared `Ez` payload first and reported a false 164 MHz resonance. Selecting the declared `Hz` component moved the Tidy3D peak to approximately 210.5 MHz and exposed the real, much smaller coarse-grid resonance shift.

### Physical scalar comparisons

The validation framework now computes scalar observables from the same solver data used for field plots:

- normalized point-probe spectra and interpolated resonance peaks for PEC/PMC cavities;
- normalized transmitted diffraction power by order;
- closed-surface equivalent currents followed by the same SI-unit near-to-far implementation for both solvers;
- directivity, peak angle, and beamwidth for antenna cases;
- bistatic RCS cuts for sphere scattering;
- magnitude and phase spectra for complex S-parameters.

Using a shared postprocessor prevents differences in a vendor-specific far-field implementation from being mistaken for solver error.

## Important pitfalls and their solutions

| Pitfall | Misleading symptom | Diagnosis | General solution |
| --- | --- | --- | --- |
| PML counted as part of the physical domain | Geometry, source, or monitor appears shifted by several cells | Plot material/source slices with physical coordinates and mark the PML interface | Preserve `Domain.bounds`; append PML externally and crop it before comparison |
| Matching nominal step instead of mesh semantics | One solver has a one-cell drift across the domain | Compare node coordinates and physical cell count, not only `dl` | Keep Maxwell's `0.025 m` maximum step and allow Tidy3D to divide the physical span uniformly (`52` cells, about `0.024615 m`) |
| Treating TFSF amplitudes as dimensionless | Field shape and correlation look good but amplitude/flux is globally wrong | Check incident Poynting flux and source API units | Convert V/m to Tidy3D's fixed `1 W/um^2` normalization analytically |
| Passing Bloch `k` directly | Oblique wavefront has the wrong transverse phase slope | Plot raw phase across one unit cell and inspect the phase advance | Convert radians/metre to `kL/(2*pi)` per axis |
| Selecting the first available field component | Cavity resonance or polarization is completely wrong despite plausible data arrays | Inspect monitor declarations and per-component energy | Export and filter exactly the requested public monitor fields |
| Using only relative L2 near a sharp resonance | Cavity field error exceeds 1000% even though mode topology agrees | Plot normalized spectra and compare peak frequencies | Report resonance shift separately; interpret the worst-frequency field norm as peak sensitivity |
| Applying arbitrary phase correction | One line plot improves while spatial phase/reflections become inconsistent | Inspect 2D raw phase, aligned phase difference, and unwrapped center lines | Remove only a single global unit phasor associated with source reference time |
| Integrating flux on the wrong TFSF side | Empty-cell transmitted power is nearly zero or reflection has a large floor | Plot the TFSF faces, scatter/total-field regions, and monitor location | Measure transmitted total field inside the TFSF region and scattered reflection on the intended exterior side |
| Judging weak diffraction/RCS channels only by relative error | Tiny orders/cuts report 50-100% error while total scattering agrees | Plot absolute normalized power alongside relative error | Prioritize dominant orders/cuts and report weak-channel magnitude explicitly |
| Trying to tune a scalar for a curved-interface residual | One geometry improves while another degrades | Compare error localization against the voxelized material boundary | Treat it as an interface-operator/convergence problem; do not add geometry-specific factors |
| Inferring boundary errors from a single number | PEC/PMC or phase sign is blamed without evidence | Compare magnitude lobes, nodal lines, phase jumps, and boundary-local residuals | Require visual slice diagnostics before changing boundary equations |
| Reconstructing coordinates as `origin + index * requested_dl` | Slab phase and attenuation appear wrong after external PML or uniform cell redistribution | Compare the reconstructed coordinates with `PreparedScene.*_nodes64` | Use the prepared solver coordinate arrays; field tensors include external PML and `dl` is a maximum, not necessarily the realized spacing |
| Comparing an unresolved absolute RCS as if it were a pattern error | The L2 error is large although all lobes and nulls overlap visually | Plot normalized angular cuts and inspect the absolute scale separately | Validate normalized direction pattern independently from convergence of a one-cell curved scatterer |
| Subtracting flux powers to isolate reflection | Apparent reflectance plus transmittance violates unity | Expand the Poynting flux of the total field and identify the incident/reflected interference term | Subtract complex fields or use a modal decomposition; never obtain reflected power by subtracting total-field powers |

## Group 7 final results

The final repaired Group 7 scenarios produced the following representative values:

| Scenario | Relative L2 | Correlation | Physical scalar result | Interpretation |
| --- | ---: | ---: | --- | --- |
| `lossy_metal_slab` | 0.0229 | 0.9999 | flux error 3.09% | SIBC pass; residual is localized near the source/PML edge |
| `periodic_grating` | 0.1129 | 0.9938 | flux error 10.30% | Same transmitted topology; small TFSF/flux residual |
| `bloch_oblique` | 0.2076 | 0.9782 | flux error 7.53% | Smooth oblique phase/discretization residual |
| `pec_cavity` | peak-sensitive | 0.3942 at worst frequency | 219.406 vs 210.810 MHz, 3.92% | Same cavity mode with a coarse-grid resonance shift |
| `pmc_cavity` | peak-sensitive | 0.3308 at worst frequency | 219.819 vs 210.484 MHz, 4.25% | Magnetic dual shows the same dispersion shift |
| `grating_diffraction` | 0.1129 | 0.9938 | dominant orders 1.91%-8.60% | Large relative errors occur only in tiny +/-2 orders |
| `antenna_directivity` | 0.00874 | 1.0000 | Dmax 7.15%, beamwidth 9.17% | Field topology and peak direction agree |
| `sphere_rcs` | 0.1678 | 0.9867 | forward 17.9%, H-plane 2.69% | Remaining error follows the coarse curved interface; weak cuts amplify relative error |

The cavity L2 values are intentionally not shown as ordinary broadband pass/fail numbers: the campaign metric takes the worst frequency, so a 4% shift of a narrow peak produces a very large amplitude mismatch at the nominal sample even when the normalized spectrum and eigenmode shape agree.

## Visual findings

- PEC and PMC cavity slices have matching lobes, nodal lines, and phase-jump topology. The residual is a spectral peak shift, not a PEC/PMC sign or boundary-type error.
- The lossy-metal magnitude and phase-aligned real-field center lines overlap; the remaining difference is concentrated near the lower source/PML edge.
- Periodic and Bloch fields retain the same transmitted wavefront topology. The oblique residual changes smoothly in space, consistent with grid dispersion rather than a discontinuous phase-wrap bug.
- Antenna directivity has the same peak direction and lobe topology. The remaining Dmax/beamwidth differences are scalar discretization effects.
- Sphere RCS error follows the staircased/subpixel curved interface. A global material or amplitude correction would be non-general and was deliberately not introduced.
- The symmetry debug slice showed matching amplitude and phase after removing PML from the image face, anchoring high-face samples to the image plane, and applying Yee-aware image-source scaling. The remaining global norm was dominated by phase noise in very weak-field pixels rather than a boundary sign error.

## Deliberately unresolved work

The campaign did not hide the following operator-level limitations with fitted scales or per-scenario patches:

- a higher-fidelity full-tensor/subpixel curved-interface operator and convergence study;
- nonlinear Kerr E/H power collocation refinement;
- runtime mode-source and S-parameter absolute calibration for the remaining modal cases;
- broadband/generalized surface impedance for curved, oblique, or laterally finite conductors.

These items require dedicated numerical implementations. They are not boundary-condition toggles or phase constants.

## Regression coverage

The final alignment changes are covered by adapter, SIBC, source, TFSF, boundary, gradient, and validation-framework tests. The final `witwin2` release gates included 340 boundary/material tests, 535 core FDTD/API/monitor/postprocess tests (plus expected skips/xfail), 125 gradient tests, and 128 FDTD validation tests. FDFD performance and FDFD-vs-FDTD comparison tests were intentionally excluded.

## CUDA hot-path and performance audit

The 0.3.0 review searched every newly added package-code line for `torch.*`, NumPy operations, `.cpu()`, and `.item()`. The only new Torch allocation associated with time stepping is the one-time setup of the persistent modulation clock. The per-step functional changes execute through native CUDA kernels: SIBC uses one surface kernel, anisotropic conductive current capture uses one kernel, and modulation advances its device clock without a host scalar. Standard scenes that do not enable these features add no launches.

On the release workstation (RTX 5080, warmed native extension, best of eight forward runs), the representative forward timings were 0.483 ms/step for `dipole_vacuum` and 0.425 ms/step for `planewave_vacuum`, versus the recorded 0.499 and 0.454 ms/step baselines. Native adjoint backward measured 4.232 ms/step versus 4.33 ms/step. No performance regression was observed. Nsight Systems and Nsight Compute were unavailable, so this gate used source-level launch-path inspection, CUDA-graph coverage, and end-to-end timing rather than hardware-counter attribution.
