# FDTD vs Tidy3D Validation Report

Date: 2026-07-12

## Scope and method

This report is the low-cost numerical baseline for the campaign in
`docs/dev/validation-vs-tidy3d-plan.md`. FDFD is excluded by user decision.

- 48 FDTD cases were prepared; all 48 Tidy3D references were generated and cache-key verified. This includes
  34 S1-S5 campaign cases and 14 historical dipole/plane-wave cases.
- The common cloud grid uses `dx=0.025 m`, approximately 52 cells per axis, 8 PML cells, and a normal
  run-time factor of 8. This is a bug-location grid, not a convergence-study grid.
- Each Tidy3D task was estimated before submission and guarded by a 2 FlexCredit per-case ceiling.
- Fields are cropped to the non-PML interior, interpolated to common plane coordinates, and compared with
  relative L2, relative Linf, and complex correlation.
- Flux error is the maximum absolute monitor mismatch divided by one shared incident-power scale. It does not
  divide each monitor by its own near-zero reference.
- Multi-frequency fields are compared one frequency plane at a time; the scenario headline reports worst L2,
  worst Linf, and minimum correlation.

Detailed numerical rows and plots are in `benchmark/RESULTS.md` and `benchmark/plots/`.

## Campaign outcome

| Outcome | Count |
| --- | ---: |
| Prepared FDTD cases | 48 |
| Valid Tidy3D caches | 48 |
| Completed Maxwell/Tidy3D field comparisons | 41 |
| Passed the plan tolerance | 0 |
| Framework or scene failures | 7 |
| FDFD cases run | 0 |

No completed case meets the plan's joint field/scalar acceptance criteria. Every completed case has field
L2 near or above 1, flux error above 0.05, or both.

## Numerical difference summary

| Family | Cases compared | Observed range | Diagnosis |
| --- | ---: | --- | --- |
| Campaign sources | 5 | L2 1.000-1.002; corr 0.639-0.976; flux err about 1 | Maxwell amplitude/power is nearly absent relative to Tidy3D in four cases; uniform-current/custom-field retain shape but not scale. |
| Materials | 13 | L2 1.114-2.192; corr 0.137-0.921; flux err 0.101-2.901 | Both normalization and physics/phase differences. `sigma_e_slab` has the best scalar error (0.101), still outside tolerance. |
| Boundaries | 2 | L2 2.054-2.742; corr 0.971-0.977; flux err 0.376-0.866 | Strong shape agreement with large global amplitude/phase mismatch; investigate plane-wave spectral normalization first. |
| Grid/geometry | 4 | L2 1.647-2.112; corr 0.441-0.904; flux err 0.126-0.728 | Geometry cases show real shape differences in addition to amplitude mismatch. Torus mesh export warnings make those rows lower-confidence. |
| Postprocess | 3 | L2 1.000-1.679; corr 0.241-0.701; flux err 0.348-1.000 | Field-level baselines only. Dedicated S21/RCS scalar cross-validation is not yet wired into the cache/report schema. |
| Historical dipole/plane-wave | 14 | L2 0.899-8.376; corr 0.310-0.998; flux err 0.300-1.185e5 | No historical case passes. Dipole power scaling and broadband normalization are the dominant failures. |

The highest-confidence normalization signal is `pml_only`: corr=0.9765 while L2=2.7415. `symmetry_center`
shows the same pattern (corr=0.9713, L2=2.0544). These should be repaired before interpreting material or
geometry amplitudes. Source cases with L2 and flux error almost exactly 1 need a direct source-spectrum and
monitor-power audit.

The historical cases reinforce this diagnosis. `dipole_offcenter` retains corr=0.9983 but has L2=3.2644 and
flux error=14.655. The two broadband dipole cases are much worse: `multi_dielectric` has L2=8.0864 and
`dipole_two_freq` has L2=8.3763. `lorentz_resonator` reports flux error about 1.185e5; although the harness no
longer divides by each near-zero monitor independently, a shared incident scale is not physically robust for a
dipole-driven resonator and needs a source-power reference observable.

## Framework and scene failures

| Case | Failure | Required test repair |
| --- | --- | --- |
| `lossy_metal_slab` | SIBC rejects a mid-domain two-sided slab. | Place a full-transverse slab flush with one domain boundary, or test volumetric `sigma_e` separately. |
| `periodic_grating` | Soft `PlaneWave` does not support periodic boundaries. | Use a supported TFSF periodic injection. |
| `bloch_oblique` | Soft `PlaneWave` does not support Bloch boundaries. | Use explicit-wavevector TFSF/Bloch injection. |
| `pec_cavity` | Display monitor payload lacks `Ez`. | Use a cavity-compatible spectral monitor/observable and extract resonance frequency. |
| `pmc_cavity` | Display monitor payload lacks `Ez`. | Same as PEC cavity. |
| `grating_diffraction` | `ModeSource` solve is undefined under periodic/Bloch transverse boundaries. | Use the grating TFSF source expected by the diffraction workflow. |
| `antenna_directivity` | Display monitor payload lacks `Ez`. | Use a closed-surface monitor and compare projection/directivity scalars. |

## Test-framework changes completed

- Complete 38-case S1-S6 inventory with registered/cache status (`python -m benchmark --inventory`).
- Reference-only generation (`--references-only`) and campaign/solver selection (`--campaign-only`,
  `--solver fdtd|fdfd`).
- Per-task Tidy3D cost estimate and hard ceiling.
- Incident-power-normalized flux error.
- Per-frequency field metrics.
- FDTD/FDFD scenario dispatch (FDFD prepared but not exercised in this campaign).
- Nonuniform-grid step estimation and physical-interior trimming based on `GridSpec.min_spacing`.

## Recommended repair order

1. Fix the homogeneous plane-wave/source-spectrum amplitude convention using `pml_only` and
   `symmetry_center`; require corr to remain high while L2 and flux converge.
2. Audit UniformCurrent/CustomField/CustomCurrent/ModeSource normalization because their errors cluster at 1.
3. Re-run simple linear materials (`sigma_e_slab`, anisotropic slab) before nonlinear/modulated/Graphene cases.
4. Repair the seven invalid scenario definitions above, then add named scalar cache fields for S-parameters,
   diffraction efficiency, RCS, and directivity.
5. Only after the low-cost grid passes, run convergence checks on the failing high-contrast/resonant cases.

## Grouped repair worklist

The 48 FDTD scenarios are partitioned below without overlap. Process the groups in this dependency order:

`Group 1 -> Group 2 -> Group 3 -> Group 4 -> Group 7 -> Group 5 -> Group 6`.

### Group 1: PlaneWave baseline normalization

- `planewave_vacuum`
- `pml_only`
- `symmetry_center`

This is the highest-priority group because PlaneWave normalization affects most slab, scattering, material,
and boundary scenarios. The current high-correlation/high-L2 pattern points to a global complex-amplitude,
phase, or source-spectrum convention mismatch. Acceptance: correlation >= 0.995, field L2 <= 0.10, and flux
error <= 0.05.

### Group 2: Dipole, custom, beam, and modal source normalization

- `dipole_vacuum`
- `dipole_ey`
- `dipole_offcenter`
- `astigmatic_beam`
- `uniform_current`
- `custom_field_source`
- `custom_current_source`
- `mode_source_wg`

Recommended internal order: the three dipole cases, `uniform_current`, the two custom sources, then beam and
mode source. Audit source spectra, discrete-source mass, absolute current/field amplitudes, and monitor power
normalization. `dipole_offcenter` is the cleanest amplitude diagnostic because its correlation is 0.9983 while
L2 is 3.2644.

### Group 3: Linear, conductive, anisotropic, and dispersive materials

- `dielectric_slab`
- `sigma_e_slab`
- `anisotropic_slab`
- `full_tensor_slab`
- `perturbation_uniform_slab`
- `sellmeier_slab`
- `debye_slab`
- `sigma_e_drude_slab`
- `custom_pole_uniform_slab`
- `pec_box`

Start this group only after Group 1 passes. Check frequency-domain sign conventions, conductivity units,
Debye/Drude/Lorentz pole lowering, tensor component mapping, and reflection/transmission direction conventions.

### Group 4: Grid, geometry, and scattering

- `planewave_dielectric_sphere`
- `dielectric_sphere`
- `metal_sphere`
- `custom_grid_slab`
- `autogrid_ring`
- `polyslab_wg`
- `mesh_primitive_scatter`
- `sphere_rcs`

Use the two dielectric-sphere rows as an internal consistency pair. Verify material voxel slices before field
comparisons. Torus/TriangleMesh references carry watertight/normal warnings and should be treated as
lower-confidence until mesh validity is fixed. `sphere_rcs` still needs a named RCS scalar comparison.

### Group 5: Coupled, broadband, resonant, and nonlinear physics

- `dipole_dielectric_box`
- `dipole_dielectric_sphere`
- `high_eps_box`
- `multi_dielectric`
- `dipole_two_freq`
- `lorentz_resonator`
- `kerr_slab`
- `tpa_slab`
- `modulated_slab`
- `graphene_sheet`

Run this group after the source, material, and geometry baselines are stable. Broadband cases currently have
L2 near 8. Nonlinear comparisons require the same absolute incident intensity in both solvers. Dipole-driven
resonators need a source-power reference; a shared monitor-flux scale is not physically robust.

### Group 6: Modal and postprocessing scalar framework

- `ring_resonator_s21`
- `waveguide_s_matrix`

The current rows are field/flux baselines only. Extend the cache and report schema with complex S11/S21,
magnitude/phase, mode-power fraction, and resonance frequency before treating these as postprocessing
cross-validation.

### Group 7: Invalid scenario definitions or unsupported source/monitor combinations

- `lossy_metal_slab`
- `periodic_grating`
- `bloch_oblique`
- `pec_cavity`
- `pmc_cavity`
- `grating_diffraction`
- `antenna_directivity`

These are framework/scenario repairs rather than numerical solver fixes:

- Move `lossy_metal_slab` flush with one domain boundary to satisfy the single-face SIBC contract.
- Replace the periodic/Bloch soft PlaneWave cases with supported explicit-wavevector TFSF injection.
- Give the PEC/PMC cavities a cavity-compatible spectral observable and compare resonance frequency.
- Use the grating TFSF workflow for diffraction instead of a ModeSource under periodic transverse boundaries.
- Use a closed-surface projection monitor for antenna directivity and compare directivity/gain/beamwidth scalars.
