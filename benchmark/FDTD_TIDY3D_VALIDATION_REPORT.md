# FDTD vs Tidy3D Validation Report

Date: 2026-07-13

## Scope and method

This report is the low-cost numerical baseline for the campaign in
`docs/dev/validation-vs-tidy3d-plan.md`. FDFD is excluded by user decision.

- 48 FDTD cases were prepared; all 48 Tidy3D references were generated and cache-key verified. This includes
  34 S1-S5 campaign cases and 14 historical dipole/plane-wave cases.
- The common cloud grid requests `dl=0.025 m`; both solvers use 52 uniformly redistributed physical cells
  (`1.28 / 52 = 0.02461538... m`) and append PML outside the physical `Domain.bounds` (68 total grid samples
  for the usual 8-layer case). The normal run-time factor is 8. This is a bug-location grid, not a
  convergence-study grid.
- Each Tidy3D task was estimated before submission and guarded by a 2 FlexCredit per-case ceiling.
- Fields are cropped to the common physical domain, interpolated to common plane coordinates, and compared
  with relative L2, relative Linf, and complex correlation. Comparisons for one directional soft-surface
  source remove one unit-modulus global source-reference phase and use source geometry/direction to exclude
  only the discrete source sheet and upstream half-space; amplitude is never fitted.
- Flux error is the maximum absolute monitor mismatch divided by one shared incident-power scale. It does not
  divide each monitor by its own near-zero reference.
- Multi-frequency fields are compared one frequency plane at a time; the scenario headline reports worst L2,
  worst Linf, and minimum correlation.

Detailed numerical rows and plots are in `benchmark/RESULTS.md` and `benchmark/plots/`.

## Group 1 alignment update

Group 1 is numerically aligned after the common grid/source/PML/monitor repairs below. All three rows meet the
acceptance target (correlation >= 0.995, field L2 <= 0.10, flux error <= 0.05):

| Case | Field L2 | Field Linf | Correlation | Flux error |
| --- | ---: | ---: | ---: | ---: |
| `planewave_vacuum` | 1.4908e-02 | 1.7343e-02 | 0.9999 | 1.0515e-02 |
| `pml_only` | 2.2492e-03 | 4.0896e-03 | 1.0000 | 1.7607e-03 |
| `symmetry_center` | 1.4908e-02 | 1.7343e-02 | 0.9999 | 1.0515e-02 |

The fixes are general rather than case-fitted:

- `Domain.bounds` is physical; Maxwell now appends PML outside it, matching Tidy3D.
- Uniform `GridSpec` treats `dl` as a maximum and redistributes `ceil(span / dl)` cells uniformly.
- Tidy3D receives Maxwell's actual Courant step, and Maxwell run length is computed from its prepared `dt`.
- The default thin-layer CPML uses an impedance-matched cubic conductivity ramp. Forward/backward-wave fits
  reduced the Maxwell reflected amplitude from 8.15% to 0.30% (Tidy3D: 1.51%).
- Infinite soft plane waves normalize over the full computational aperture and include the derived Yee
  half-cell Poynting factor; no empirical amplitude multiplier or delay was introduced.
- Spectral flux planes use a shared Yee plane and cell-centred control-volume integration instead of
  independently interpolating E/H phasors and dropping half-width endpoint cells.
- Complex diagnostic plots show raw source-reference phase and phase-aligned field/center-line differences.

`symmetry_center` currently builds the same unfurled vacuum scene as `planewave_vacuum`; its numerical row is
therefore a duplicate baseline and does **not** validate symmetry folding. A genuine half-domain symmetry
scenario remains a scenario-definition task and should not be counted as symmetry-feature coverage.

Broader regression runs also exposed that the old AutoGrid/nonuniform slab tests manually embedded PML cells
inside `GridSpec.custom` and reused prepared grids without removing PML. Those fixtures now pass only physical
nodes and let `PreparedScene` append PML once. Their two-plane reflected-power estimator has a 5-16% residual
after flux planes snap to physical Yee nodes; the independent graded-slab Fresnel checks remain within their
existing 8% tier. A denser forward/backward modal estimator and convergence sweep belong to the later grid group,
not to this uniform-vacuum Group 1 acceptance.

## Group 2 alignment update

Group 2 has been fully rerun with refreshed Tidy3D references and inspected through complex-field slice plots.
Two of eight cases meet every campaign target; the three vacuum dipoles and custom field replay are close in field
shape or power, while the two volume-current cases retain material amplitude differences. The mode case now uses a
two-candidate polarization-filtered Tidy3D reference, but exact square-guide degeneracy and the finite-run propagation
envelope remain visibly different. This is a completed diagnostic/fix pass, not a claim that all eight rows align.

| Case | Field L2 | Field Linf | Correlation | Flux error | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `dipole_vacuum` | 5.4550e-02 | 8.0581e-02 | 0.9999 | 6.8288e-02 | Field pass; flux marginal |
| `dipole_ey` | 5.8028e-02 | 6.5710e-02 | 0.9999 | 1.4369e-02 | Pass |
| `dipole_offcenter` | 5.3690e-02 | 6.4194e-02 | 1.0000 | 7.4575e-02 | Field pass; flux marginal |
| `astigmatic_beam` | 1.5049e-02 | 1.9512e-02 | 0.9999 | 1.2199e-03 | Pass |
| `uniform_current` | 7.7862e-01 | 1.0054e+00 | 0.9092 | 6.6514e-01 | Residual volume-current amplitude |
| `custom_field_source` | 1.9275e-01 | 2.2148e-01 | 0.9896 | 3.2920e-02 | Power pass; field near target |
| `custom_current_source` | 8.0058e-01 | 8.2125e-01 | 0.9594 | 7.6660e-01 | Residual reverse-interpolation amplitude |
| `mode_source_wg` | 7.7478e-01 | 5.0843e-01 | 0.6367 | 1.4171e-01 | Degenerate-basis and propagation-envelope residual |

The retained repairs are general and contain no per-scenario fitted amplitude:

- FDTD point-dipole profiles now preserve one SI current moment under Yee control-volume integration; the
  Tidy3D adapter converts A m to A um. The three vacuum field slices then agree at L2 0.054-0.058 and
  correlation 0.9999-1.0000. The remaining 6.8-7.5% flux offsets for two dipoles are consistent with comparing
  Maxwell's finite Gaussian regularization against Tidy3D's ideal point source on this coarse grid.
- Source-spectrum normalization removes only the arbitrary waveform envelope and preserves user amplitude and
  phase. Tidy3D Gaussian-pulse export keeps physical phase separate from envelope offset.
- Gaussian/astigmatic beams are normalized by their analytic transverse power. Their pulsed time origin is the
  launch plane rather than a remote waist; this removed the pre-`t=0` pulse truncation. The downstream beam
  magnitude, real part, and unwrapped phase now overlap visually, with differences confined to the source sheet.
- Uniform current is deposited by source-box/Yee-control-volume overlap, exactly preserving physical current
  volume. This changes the slice correlation from 0.2887 to 0.9092, but exposes a common unresolved amplitude
  difference instead of hiding it behind an over-wide `7 x 7 x 7` source patch.
- Custom current uses component-aware staggered windows and Tidy3D-compatible endpoint extension. The wavefront
  correlation improves from 0.7831 to 0.9594; the remaining approximately uniform amplitude/power factor requires
  a faithful implementation of Tidy3D's private reverse-interpolation weights, so no empirical 0.76 multiplier was
  introduced.
- Custom field replay now uses one discrete TFSF face and converts physical H to the update-equation sign. Its
  forward power agrees within 3.3%; the residual field error is concentrated near the launch sheet.
- Full-vector modes are rotated deterministically inside exactly degenerate eigenspaces toward the requested
  polarization, normalized by integrated Poynting power, and time-aligned across staggered E/H launch faces.
  Tidy3D `ModeSource` and `ModeMonitor` now request both polarization candidates and receive the corresponding
  `ModeSortSpec` polarization-fraction filter, and the mode-export contract invalidates stale one-candidate caches.
  The refreshed transverse slice still has Tidy3D `||Ey|| / ||Ez|| = 0.956`, versus Maxwell's deterministic
  requested-`Ez` basis at 0.217. Tidy3D's filter orders the exact square-guide degenerate eigenvectors but does not
  rotate their eigenspace, so component-wise `Ez` remains basis-dependent. The complex longitudinal slice also shows
  a smooth Tidy3D packet envelope versus Maxwell's oscillatory finite-run envelope. Flux error is reduced from 5.51
  to 0.142, but no fitted polarization rotation or amplitude factor is introduced to hide these two residuals.

The diagnostic plots used for these conclusions are the per-case
`benchmark/plots/<case>/complex_field_diagnostic.png` files. They include raw/phase-aligned complex slices and
center-line magnitude, real-field, and unwrapped-phase traces; the source-boundary and propagation conclusions above
come from those plots, not from scalar metrics alone.

## Group 3 alignment update

Group 3 has been fully rerun with refreshed material/source-export references and inspected through magnitude,
real-field, and phase slices. Four of ten rows meet every campaign target, with Debye also meeting the field targets
but missing the flux target by 2.0 percentage points. Full-tensor source polarization is now exported correctly, but
that row exposes a remaining full-anisotropic Yee interface/collocation error rather than reaching alignment.

| Case | Field L2 | Field Linf | Correlation | Flux error | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `dielectric_slab` | 2.5646e-01 | 4.8777e-01 | 0.9666 | 1.1694e-01 | High-contrast interface residual |
| `sigma_e_slab` | 6.6279e-02 | 1.0810e-01 | 0.9978 | 4.1609e-02 | Pass |
| `anisotropic_slab` | 9.2068e-02 | 3.2959e-01 | 0.9958 | 2.8209e-02 | Pass |
| `full_tensor_slab` | 6.8997e-01 | 8.2745e-01 | 0.7655 | 6.1410e-01 | Full-tensor Yee interface residual |
| `perturbation_uniform_slab` | 7.5831e-02 | 1.9567e-01 | 0.9971 | 3.2132e-02 | Pass |
| `sellmeier_slab` | 1.3406e-01 | 1.5388e-01 | 0.9911 | 1.0289e-02 | Discrete phase residual |
| `debye_slab` | 6.5057e-02 | 8.3659e-02 | 0.9979 | 7.0168e-02 | Field pass; flux marginal |
| `sigma_e_drude_slab` | 2.0775e-01 | 2.1743e-01 | 0.9783 | 1.4805e-01 | Metal-interface/ADE residual |
| `custom_pole_uniform_slab` | 1.3974e-01 | 2.1811e-01 | 0.9902 | 8.4051e-02 | ADE phase residual |
| `pec_box` | 2.7226e-02 | 2.8768e-02 | 0.9997 | 4.6812e-02 | Pass |

The retained repairs are convention-level and shared by all exported materials:

- The public complex-permittivity evaluator now uses the same `exp(-i omega t)` conductivity sign as the FDTD
  runtime and Tidy3D.
- Debye export converts Maxwell's seconds-based `tau` to Tidy3D's angular-time convention by `2 pi`; Lorentz export
  converts Maxwell's `gamma` to Tidy3D's half-width parameter. Mixed dispersive media are lowered through the exact
  one-pole Tidy3D `pole_residue` identities rather than duplicating their private algebra. Direct `eps_model`
  regression checks cover Debye, Drude, Lorentz, conductivity, and mixed-pole media, and a material-export contract
  revision invalidates stale cloud caches.
- Reflective flux rows now use the matching empty-scene incident power instead of normalizing by a small net
  reflected/transmitted scene flux. This changes the PEC flux error from 0.631 to 0.0468 without changing its field.
- The soft-source comparison mask removes the full downstream Yee source stencil in addition to the upstream
  half-space. It does not remove any material-interface cells.
- Directional source export now derives Tidy3D's `pol_angle`, `angle_theta`, and `angle_phi` from both the Maxwell
  propagation vector and requested transverse polarization for every injection axis/sign. Real-Tidy3D vector tests
  cover normal and oblique negative propagation, and a source-export contract revision invalidates old fixed-P caches.
  Vacuum incident-power signatures intentionally ignore unit transverse orientation, so reflective flux errors for
  arbitrary plane-wave polarization use the same physical incident-power scale.
- Three-point polarized Kottke averaging replaces one-point material sampling in the common low-cost scenes. A
  controlled 3-to-5 point sweep improves six of seven planar material rows, but worsens the Drude-metal row and most
  curved/mesh rows; five and seven points also plateau for the dielectric sphere. The common three-point setting is
  therefore retained rather than selecting a per-case quadrature depth from the comparison metric.

The slices distinguish the remaining causes. Debye magnitude and real-field traces overlap almost completely, with
small residuals at the slab/source lines. PEC fields overlap outside the conductor; phase inside the zero-field PEC
region is undefined and is not evidence of propagation error. The custom Lorentz pole shows a downstream phase-slope
drift despite its exact exported `eps_model`, consistent with coarse-grid ADE numerical dispersion. The combined
Drude/conductivity row differs mainly inside and immediately after the metal slab, where Tidy3D's regular-metal
staircasing policy and Maxwell's polarized material smoothing are not equivalent. No fitted phase slope, material
coefficient, or metal-only amplitude scale was introduced.

The refreshed full-tensor reference uses the requested eigenpolarization rather than the old implicit Ex source.
Maxwell and Tidy3D then launch nearly the same polarization: the measured `||Ey||/||Ex||` ratios are 0.340 and 0.348,
respectively, versus the requested 0.351, and the material/source slices are coincident. The propagated Ex/Ey
correlations nevertheless fall to 0.765/0.731. A controlled Maxwell-only check fills the whole domain with the tensor
and compares that eigenmode with its scalar eigenvalue; Ex is within L2 0.092, while the finite slab comparison grows
to Ex/Ey L2 0.280/0.379. Snapping the slab thickness to the redistributed grid improves Ex only to 0.178 and leaves Ey
at 0.356. This isolates the residual to off-diagonal tensor coupling at a material interface and staggered-component
collocation. Maxwell currently rejects polarized Kottke averaging for off-diagonal tensors, so a correct fix requires
a full tensor-interface averaging operator and matching adjoint, not a source rotation or scenario-specific factor.
That larger numerical implementation is explicitly left unresolved in this pass.

## Group 4 alignment update

Group 4 has been fully rerun after the mesh and subpixel repairs. `custom_grid_slab` meets every campaign target. The
other seven rows retain high complex correlation (0.9422-0.9889), but their field L2 remains above 0.10 on this
approximately 0.0246 m grid.

| Case | Field L2 | Field Linf | Correlation | Flux error | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `planewave_dielectric_sphere` | 1.5007e-01 | 4.1854e-01 | 0.9889 | 1.0379e-02 | Curved-interface residual |
| `dielectric_sphere` | 1.5007e-01 | 4.1854e-01 | 0.9889 | 1.5193e-02 | Curved-interface residual |
| `metal_sphere` | 2.4529e-01 | 1.1873e+00 | 0.9695 | 2.3390e-02 | Metal/curved-interface residual |
| `custom_grid_slab` | 5.7926e-02 | 1.0130e-01 | 0.9983 | 4.6123e-02 | Pass |
| `autogrid_ring` | 1.6721e-01 | 3.2442e-01 | 0.9860 | 8.4676e-03 | Curved-interface residual |
| `polyslab_wg` | 1.6286e-01 | 2.5507e-01 | 0.9867 | 2.0226e-02 | Edge/interface residual |
| `mesh_primitive_scatter` | 1.6188e-01 | 5.0388e-01 | 0.9871 | 9.3592e-03 | Curved-interface residual |
| `sphere_rcs` | 3.4039e-01 | 6.0075e-01 | 0.9422 | 1.0084e-02 | Field baseline only; RCS scalar pending |

The generic geometry repairs and visual checks are:

- The shared `Torus.to_mesh()` major-ring x coordinate now uses `cos(phi)` rather than multiplying by an extra
  `cos(theta)`. Analytic volume/bounds tests cover the mesh. Tidy3D no longer reports the previous
  non-watertight/inconsistent-normal warnings.
- The Torus material/source comparison shows Maxwell and Tidy3D matching on all three central slices to about
  `1e-12`; the ring hole, outer radius, source plane, physical domain, and external PML placement are therefore
  aligned before field comparison.
- Complex difference slices localize the strongest sphere, ring, PolySlab, and mesh residuals to object boundaries
  and their outgoing scattered wavefronts. They do not show a one-sided PML reflection, a rigid spatial translation,
  or a global phase-sign reversal. PolySlab and custom-grid center-line phases nearly overlap outside the object.
- Uniform five- and seven-point sampling does not reduce the dielectric-sphere residual, and three points is best for
  `sphere_rcs`; raising the quadrature depth is therefore not a general cure. The remaining error is the difference
  between Maxwell's coarse Yee/Kottke interface operator and Tidy3D's subpixel operator. Resolving it requires an
  operator/convergence study, not a per-geometry material or amplitude correction.
- `sphere_rcs` still compares its near-field plane and flux only. A named far-field RCS scalar is not yet present in
  the benchmark cache/report schema, so this row must not be presented as completed RCS validation.

## Campaign outcome

| Outcome | Count |
| --- | ---: |
| Prepared FDTD cases | 48 |
| Valid Tidy3D caches | 48 |
| Completed Maxwell/Tidy3D field comparisons | 41 |
| Passed the plan tolerance in the original full-campaign baseline | 0 |
| Group 1 cases passing after repair | 3 |
| Group 2 cases passing every target after repair | 2 of 8 |
| Group 3 cases passing every target after repair | 4 of 10 |
| Group 4 cases passing every target after repair | 1 of 8 |
| Framework or scene failures | 7 |
| FDFD cases run | 0 |

The total prepared/completed/failure counts describe the original full-campaign inventory. Groups 1-4 have since been
regenerated as shown in their update sections; Groups 5-7 still represent the original diagnostic baseline.

## Numerical difference summary

| Family | Cases compared | Observed range | Diagnosis |
| --- | ---: | --- | --- |
| Campaign sources | 5 | L2 1.000-1.002; corr 0.639-0.976; flux err about 1 | Maxwell amplitude/power is nearly absent relative to Tidy3D in four cases; uniform-current/custom-field retain shape but not scale. |
| Materials | Updated Group 3: 10 | L2 0.027-0.690; corr 0.765-1.000; flux err 0.010-0.614 | Four pass; Debye is field-aligned. The main residuals are full-tensor interface collocation, regular-metal smoothing, and coarse-grid dispersive phase. |
| Boundaries | 2 | Original: L2 2.054-2.742; repaired Group 1: L2 0.002-0.015, corr 0.9999-1.0000, flux err 0.002-0.011 | External PML geometry, thin-layer CPML entrance mismatch, source power, and Yee flux integration were repaired. |
| Grid/geometry | Updated Group 4: 7 | L2 0.058-0.245; corr 0.970-0.998; flux err 0.008-0.046 | Torus geometry/material slices now match exactly; remaining differences localize to coarse curved/edge interface operators. |
| Postprocess | 3 | L2 1.000-1.679; corr 0.241-0.701; flux err 0.348-1.000 | Field-level baselines only. Dedicated S21/RCS scalar cross-validation is not yet wired into the cache/report schema. |
| Historical dipole/plane-wave | 14 | L2 0.899-8.376; corr 0.310-0.998; flux err 0.300-1.185e5 | No historical case passes. Dipole power scaling and broadband normalization are the dominant failures. |

The highest-confidence normalization signal was `pml_only`: corr=0.9765 while L2=2.7415. It now reaches
corr=1.0000 and L2=0.00225. Source cases outside Group 1 with L2 and flux error almost exactly 1 still need a
direct source-spectrum and monitor-power audit.

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

1. **Completed:** fix the homogeneous plane-wave/source-spectrum amplitude convention using
   `planewave_vacuum`, `pml_only`, and the duplicate `symmetry_center` baseline.
2. **Completed diagnostic pass:** source normalization was repaired where a public physical mapping was available;
   retain the documented volume-current reverse-interpolation and modal-envelope residuals without fitted scales.
3. **Completed diagnostic/fix pass:** material export conventions, reflective flux scaling, and low-cost subpixel
   sampling are repaired; retain the documented full-tensor/metal/ADE interface residuals without fitted factors.
4. **Completed diagnostic/fix pass:** mesh validity and material/source voxel alignment are repaired; use a true
   interface-operator convergence study for remaining curved-grid residuals.
5. Repair the seven invalid scenario definitions above, then add named scalar cache fields for S-parameters,
   diffraction efficiency, RCS, and directivity.
6. Only after the low-cost grid passes, run convergence checks on the failing high-contrast/resonant cases.

## Grouped repair worklist

The 48 FDTD scenarios are partitioned below without overlap. Process the groups in this dependency order:

`Group 1 -> Group 2 -> Group 3 -> Group 4 -> Group 7 -> Group 5 -> Group 6`.

### Group 1: PlaneWave baseline normalization

- `planewave_vacuum`
- `pml_only`
- `symmetry_center`

This group now passes. PlaneWave normalization affects most slab, scattering, material, and boundary
scenarios, so downstream groups should be regenerated from this repaired baseline. Acceptance: correlation
>= 0.995, field L2 <= 0.10, and flux error <= 0.05. The `symmetry_center` caveat above remains.

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

This group has now been rerun; use the Group 2 alignment update above rather than the original baseline values in
this worklist. Two cases pass every target, the three dipole fields pass with two marginal flux rows, custom-field
power passes, and the volume-current plus modal-degenerate-basis/envelope residuals remain explicitly open.

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

This group has now been rerun; use the Group 3 alignment update above. Four cases pass every target and Debye passes
the field targets. Full-tensor interface collocation, regular-metal smoothing, and coarse-grid ADE phase remain
explicitly open.

### Group 4: Grid, geometry, and scattering

- `planewave_dielectric_sphere`
- `dielectric_sphere`
- `metal_sphere`
- `custom_grid_slab`
- `autogrid_ring`
- `polyslab_wg`
- `mesh_primitive_scatter`
- `sphere_rcs`

This group has now been rerun; use the Group 4 alignment update above. The dielectric-sphere pair is internally
identical, Torus/TriangleMesh validity is fixed, and all inspected material/source slices align. The remaining
coarse-interface residuals and the missing named `sphere_rcs` scalar are explicitly open.

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
