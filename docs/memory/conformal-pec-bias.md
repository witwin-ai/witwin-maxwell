# Conformal volume PEC bias (2026-10-30)

## Verdict and current behavior

The PCB hypothesis is confirmed by the compiled coefficients and the native update:
volume `SubpixelSpec(pec="conformal")` has E-edge fractions but **no PEC open-face
areas or area-weighted H update**. More precisely, the implementation does not
merely shorten the electric circulation lengths: it damps the electric field
itself on every fractional edge, including its previous-step value. This adds
artificial loss and breaks the consistent electric/magnetic cut geometry.
Adding `1/A_open` to H alone would leave the electric damping defect in place.

The defect remains unfixed. Use `pec="staircase"`, the default. Construction of
`SubpixelSpec(pec="conformal")` now emits a `UserWarning` explaining the known
systematic bias, the roughly 30–40% non-converging curved-PEC scattering errors
in the source qualification, the staircase recommendation, and this note's path.
Construction still succeeds, including when a spec is created before a scene
contains PEC. The warning is deliberately attached to the policy, not a GPU run.

Numerical evidence below was collected on local **GPU 1 only**, at commit
`769b15168b2b84ea7644105123190fcd7202c852` (warning-only changes relative to its
parent). No numerical implementation was modified for these measurements.

## Root cause in the code

Line numbers refer to the measurement commit above.

| Location | What it actually does |
|---|---|
| `witwin/maxwell/compiler/materials.py:1799`, `_pec_signed_distance` | Samples the union signed distance of volume PEC, including periodic images; preserves the aligned-face tolerance. |
| `witwin/maxwell/compiler/materials.py:1826`, `_edge_line_coverage`; `:1850`, `_pec_edge_fill` | Interpolates endpoint signed distances into per-E-edge coverage. The compiler builds edge lengths, not open H-face areas. `_pec_edge_fill` retains the `torch.maximum(volume_fill, planar_sheet_edges)` at line 1863. |
| `witwin/maxwell/compiler/materials.py:2657` and `:2796`, `compile_material_model` | Both material compilation paths store `pec_edge_fill` when conformal is selected. Neither stores PEC face areas. |
| `witwin/maxwell/fdtd/runtime/materials.py:1069`, `_pec_edge_open_fractions` | Conformal consumes the fractional edge fills. Staircase uses the minimum of the two binary conductor-node classifications at lines 1113–1114. The separate sheet-edge maximum remains at line 1116. |
| `witwin/maxwell/fdtd/runtime/materials.py:1140`, `_apply_pec_edge_suppression` | Lines 1157–1162 multiply **both** E decay and E curl coefficients by `p = 1 - fill`; nonlinear external decay and anisotropic corrections receive the same suppression. |
| `witwin/maxwell/fdtd/cuda/kernels/electric.cu:326`, `update_electric_standard_kernel` | Consumes those coefficients as `E_new = decay*E_old + curl_coeff*curl(H)`. Thus vacuum cut edges obey `E_new = p*E_old + p*(dt/eps0)*curl(H)`. The CPML electric kernels consume the same scaled coefficients. |
| `witwin/maxwell/fdtd/runtime/materials.py:1059`, `_magnetic_update_coefficients`; `:1722`, `build_update_coefficients` | H curl coefficients are `dt/mu` with material/absorber factors only. PEC suppression is electric-only; the H mask at `_apply_pmc_face_suppression` is for **PMC**, not PEC. |
| `witwin/maxwell/fdtd/runtime/stepping.py:120`, `:180`, `:278` | Dense CPML, compressed CPML and standard magnetic dispatch pass H decay/curl and ordinary inverse spacings. No PEC edge-length or face-area tensors enter. |
| `witwin/maxwell/fdtd/cuda/kernels/magnetic.cu:58`, `:70`, `update_magnetic_standard_kernel` | Forms full-grid differences of E and multiplies by the H curl coefficient. `update_magnetic_cpml_kernel` (line 75) and `update_magnetic_cpml_compressed_kernel` (line 172) similarly use unweighted E differences, then stretch them for CPML. |
| `witwin/maxwell/fdtd/runtime/timestep.py:119`, `compile_fdtd_time_step` | The curl bound uses grid spacings and material bounds, with no cut-area dependence. It could not safely support a newly introduced unbounded `1/A_open` without modification. |

For a cut edge with no curl drive, the exact decay rate is
`gamma = -log(p)/dt`. This is a dissipative penalty, not a lossless PEC geometric
integral. Refinement also reduces dt; it does not remove the per-step penalty on
an edge with the same fractional cut. The old material comments' `eps*fill/dt`
conductivity analogy is a small-fill approximation; the recurrence above is the
precise statement used here.

The staircase rule from `aa7e14fd` is present and works in the controls below.
This diagnosis does not implicate its node rule or planar-sheet ownership.

## GPU evidence

### Round coax: speed and artificial attenuation

Reused `tests/validation/physics/test_pec_curved_coax_tem.py`'s cylinder fixture:
vacuum, inner radius 1 mm, shield-hole radius 2.3 mm, axial length 30 mm, four
radial pulse dipoles, z-PML, 13 Ex/Ey measurement planes from 12 to 24 mm.
The shield has 96 polygon vertices at 0.25 mm and 256 at 0.125 mm, matching the
existing fixture. There are no ports or fitted impedance references.
Runs use 6000/12000 steps, respectively; dt is 0.433312461/0.216656230 ps.

First, the fixture's original real-k forward/backward fit gives:

| h (mm) | PEC policy | v/c, 5 GHz | v/c, 8 GHz | v/c, 11 GHz | Maximum relative fit residual |
|---|---|---:|---:|---:|---:|
| 0.25 | staircase | 0.999978058 | 0.999947006 | 0.999898814 | 4.57e-7 |
| 0.25 | conformal | 0.854592643 | 0.855509229 | 0.856641089 | 1.32e-2 |
| 0.125 | staircase | 0.999992245 | 0.999987629 | 0.999974648 | 8.20e-7 |
| 0.125 | conformal | 0.916688238 | 0.917204646 | 0.917766830 | 4.28e-3 |

The conformal real-k fit does not satisfy the test helper's clean-line residual
assertion (`<1e-4`). The scratch diagnostic records this failure instead of
changing the test. To separate loss from phase, repeat just the conformal runs
and fit `F(z) = a exp((i beta - alpha)(z-zbar)) + b exp(-(i beta - alpha)(z-zbar))`.
For each beta/alpha candidate, solve a/b by linear least squares; optimize the two
real propagation parameters against the normalized complex residual. Results:

| h (mm) | Frequency (GHz) | v/c = k0/beta | alpha (Np/m, amplitude) | Relative complex-k fit residual |
|---|---:|---:|---:|---:|
| 0.25 | 5 | 0.854567019 | 0.8597356 | 2.02e-7 |
| 0.25 | 8 | 0.855437722 | 2.1368854 | 3.72e-7 |
| 0.25 | 11 | 0.856539671 | 3.8908581 | 2.47e-7 |
| 0.125 | 5 | 0.916684052 | 0.3161133 | 3.44e-7 |
| 0.125 | 8 | 0.917193689 | 0.7584862 | 5.02e-7 |
| 0.125 | 11 | 0.917750286 | 1.3337686 | 3.38e-7 |

The original residual was explained by attenuation: this is one clean attenuating
mode, with a large phase-speed error, in a physically lossless line. The 5 GHz
speed deficit shrinks by a factor 0.573 when h halves, far from the staircase's
small Yee dispersion. These two coax levels do **not** establish an asymptotically
nonzero speed deficit; the scattering plateau is a separate observation.

### Quantitative check of the missing geometry

In both policies, for all three H components over the entire compiled coax grid,
`ch*_curl * mu_H* / dt` has minimum **1.0** and maximum **1.0**. There is no hidden
PEC area factor in the coefficients, including inside the axial PML.

In an extruded cross-section, a cut Hx face spans y and z, so its open area fraction
is the open Ey length fraction at that x; similarly Hy uses Ex. Consequently the
compiled transverse fractions directly quantify the missing face-area correction
for this fixture's piecewise-linear cut geometry:

| h (mm) | Transverse cut-edge open p: min / median / max | Missing `1/alpha_face`: median / max | Median `-log(p)/dt` (s^-1) |
|---|---|---|---:|
| 0.25 | 0.128124 / 0.544824 / 0.891549 | 1.83546 / 7.80491 | 1.40151e12 |
| 0.125 | 0.0201465 / 0.397880 / 0.944659 | 2.51332 / 49.6364 | 4.25377e12 |

The inverse-area numbers apply to the **weighted circulation**, not to the
existing unweighted H curl. They must not be installed as an isolated multiplier.
At 5 GHz these median electric decay rates are 44.6/135.4 times omega.

A radius inferred from velocity alone cannot validate a vacuum-coax geometry:
every consistent pair of coax radii still gives `L'C' = mu0*eps0` and `v = c`.
The measured complex-fit speeds instead imply an effective `L'C'/(mu0*eps0)` of
1.36933/1.19004. As a **conditional toy inversion**, holding capacitance at its
nominal value and assigning all excess to inductance gives
`ln(b/a_L) = ln(b/a)/(v/c)^2`, hence `a_L = 0.73520/0.85361 mm` for fixed b=2.3 mm.
The apparent inward shifts are 1.06h/1.17h. This is consistent in scale with a
one-cell surface defect, but is not an independent prediction of the measured
speed or a unique effective radius. The direct coefficient/area comparison and
the measured loss are stronger evidence than that inversion.

### PEC sphere against Mie

Copied the harness at `4c69d038:benchmark/label_qualification/analytic_convergence.py`
into scratch. Only its repository-root assignment was changed to `Path.cwd()`
and the PEC parameter list restricted to x=1. The Mie functions are the existing
`scripts/dev/fdtd/subpixel_analytic.py` entries `mie_sphere_efficiencies` and
`mie_sphere_amplitudes`. No fit of amplitude, phase or radius is applied.

Wavelength is 1 m, radius `1/(2*pi)` m, Mie `C_sca = 0.162008930029084 m^2`.
The first seeded offset (seed 1729) is
`(-0.4692579704, -0.3315425202, -0.2060495530)` cells, reused at both levels and
for both policies. All other settings retain the source harness defaults:
80 transient + 24 steady cycles, Hanning sampling, 12 PML layers, 0.5 wavelength
padding, 32 Gauss-Legendre theta nodes plus 181 cut angles, 32 azimuth angles,
same-grid vacuum incident calibration. Computational shapes are 68^3/80^3,
with 4003/6005 steps per solve.

| Cells/lambda | Policy | C_sca, Huygens flux (m^2) | Flux error | C_sca, far-field quadrature (m^2) | Quadrature error | Complex far-field L2 error |
|---:|---|---:|---:|---:|---:|---:|
| 20 | staircase | 0.139543340 | 13.8669% | 0.138483261 | 14.5212% | 13.1155% |
| 20 | conformal | 0.237025880 | 46.3042% | 0.235177389 | 45.1632% | 64.9501% |
| 30 | staircase | 0.145430748 | 10.2329% | 0.145359193 | 10.2770% | 9.4161% |
| 30 | conformal | 0.227266026 | 40.2799% | 0.227165049 | 40.2176% | 45.5901% |

This cheaply reproduces the large conformal bias and substantially better
staircase controls. The source assessment's roughly 30–40% plateau is a
**two-offset mean over 10–40 cells/lambda**, whereas these are one-offset values
at 20/30; they need not agree numerically. These two levels alone do not prove a
nonzero continuum-limit error. The earlier wider sweep supplies the observed
non-convergence claim; neither study establishes an infinite-resolution theorem.

Scratch records: `<scratch>/witwin_cov_conf/coax.jsonl`, `coax_loss.jsonl`,
`sphere.jsonl`; scripts: `coax_diagnosis.py`, `coax_loss_diagnosis.py`,
`sphere_diagnosis.py`. The complex-fit run also retains the 13-plane functionals
in `coax_functional_0.00025000.npz` and `coax_functional_0.00012500.npz`.
Sphere runs took 5.4–10.3 s per method/level including vacuum calibration;
coax runs took about 5–27 s, apart from first-run setup. No cloud resources were used.

## Proposed correction, not implemented

Use physical E fields on open edge segments, keeping the current public field
normalization. Compile `l_open` on E edges and `A_open` on H faces from one
consistent volume-PEC cut geometry. Faraday's update must be

```text
H_f(new) = H_f(old) - dt/(mu_f * A_open_f) * sum_e sign(f,e) * l_open_e * E_e
```

Use the existing Yee metric on nonuniform grids. Ampere's update on a partially
open edge must retain its ordinary material decay and curl coefficients; only
fully closed edges are hard-zeroed. This removes the repeated fractional E
decay. Apply length factors to each E sample **before** forming H differences
and CPML histories; apply inverse face fractions to the H curl coefficient.
Simply changing E curl coefficients or dividing the current H curl by area is
insufficient. This is the edge-length/face-area structure of the
[Dey–Mittra operator, section 2](https://arxiv.org/html/1301.3794v1#S2).

For the first correctness implementation, retain every positive-area face and
treat small cells by reducing the global timestep using a certified upper bound
on the geometry-weighted Maxwell operator's largest eigenvalue. For uniform
vacuum, this operator is similar to the symmetric positive operator
`K = (mu*A)^(-1/2) C (L/eps) C^T (mu*A)^(-1/2)`, with diagonal fractional
edge lengths L and face areas A. A conservative absolute-row-sum/Gershgorin
upper bound U gives `dt <= 0.9 * 2/sqrt(U)` before the existing joint material
restrictions. Use the corresponding Yee volume metrics on nonuniform grids.
Fully closed faces carry zero H and are excluded from division. Compute geometry
and bounds on device during prepare; freeze dt for differentiation. An unchecked
power-iteration estimate is not a certified bound.

This small-cell policy costs timesteps but avoids an arbitrary, fixed face-area
floor or face deletion changing the convergence target. A later conservative
area-borrowing/face-merging method could recover efficiency, with its own energy
and stability derivation. Face exclusion can degrade convergence, as measured in
[Bauer, Werner and Cary, section 3](https://arxiv.org/html/1301.3794v1#S3).
The interior Maxwell bound alone does not qualify every absorber/material
composition; the coax requires testing conductors through CPML as well.

### Exact implementation touch list for the proposed first forward correction

No entries below are changes in this diagnosis commit. New helper names are
explicitly marked **new**. Paths are relative to `witwin/maxwell/`.

| File | Functions to change/add and purpose |
|---|---|
| `compiler/materials.py` | `compile_material_model` in both compilation branches; **new** `_pec_face_open_fractions` and its face-clipping helper. Reuse `_pec_signed_distance` and `_edge_line_coverage`; derive faces from a consistent reconstruction using the same edge intersections, including ambiguous four-corner cuts. Preserve `_pec_edge_fill`'s planar maximum and aligned-face tolerance. Store volume-only face areas separately from sheet ownership. |
| `fdtd/runtime/materials.py` | `build_materials` binds geometry before timestep selection; `_pec_edge_open_fractions` supplies lengths while preserving the staircase branch and sheet maximum; `_apply_pec_edge_suppression` uses a binary closed-edge mask in the corrected conformal branch, including nonlinear/aniso masks; `build_update_coefficients` invokes **new** `_apply_conformal_pec_face_weights` to divide H curl by positive face fractions and hard-zero closed H faces. Keep `_magnetic_update_coefficients`' base material formula. |
| `fdtd/runtime/timestep.py` | `compile_fdtd_time_step`; **new** `_conformal_pec_curl_rate_bound`. Include the cut geometry in the stability bound before source, absorber and material recurrences bind dt. |
| `fdtd/runtime/stepping.py` | `update_magnetic_fields_standard`, `update_magnetic_fields_cpml_dense`, `update_magnetic_fields_cpml_compressed` pass per-edge lengths to the native magnetic updates. Both real and imaginary fields use the same real geometry. |
| `fdtd/cuda/backend.py` | `_magnetic_hx_standard`, `_magnetic_hy_standard`, `_magnetic_hz_standard`; the corresponding `_standard_bounded`, `_cpml`, `_cpml_compressed` wrappers pass length tensors. Keep the existing no-conformal launch path efficient. |
| `fdtd/cuda/extension.cpp` | Matching `update_magnetic_h{x,y,z}_{standard,standard_bounded,cpml,cpml_compressed}_cuda` declarations and operator schemas in `WITWIN_CUDA_OPS`; update batch schemas alongside their callers, not independently. |
| `fdtd/cuda/kernels/magnetic.cu` | `update_magnetic_standard_kernel`, `update_magnetic_cpml_kernel`, `update_magnetic_cpml_compressed_kernel` and their matching Hx/Hy/Hz launchers: multiply each E stencil sample by its open length fraction, and use the area-scaled H curl coefficients. CPML psi consumes those weighted differences. No extra full-grid E copy is needed. |

The seven files above define the single-device forward numerical correction.
Production integration must also update the following callers/transposes in the
same implementation before claiming support for their workflows:

| File | Functions and required integration |
|---|---|
| `fdtd/runtime/batch.py` | `NativeFieldSlot.bind`, `NativeFieldSlot._bind_operations`: bind per-lane length tensors with the existing coefficient sharing rules and pass them to `update_magnetic_batch_cuda` / `update_magnetic_cpml_batch_cuda` in `magnetic.cu` and their schemas. |
| `fdtd/distributed/solver.py` | `_launch_magnetic_hx`, `_launch_magnetic_hy_hz`: pass local lengths with the bounded standard launches; local/global timestep selection must agree on the cut-geometry bound. |
| `fdtd/adjoint/core.py` | `_forward_magnetic_fields`, `_forward_magnetic_fields_cpml`, `_forward_magnetic_fields_complex`, `_step_state`: replay the same weighted E differences, including CPML memory. |
| `fdtd/adjoint/native.py` | `reverse_phase2_magnetic_to_e`, `reverse_cpml_phase_magnetic`, `_reverse_cpml_magnetic_phase_native`, `_reverse_step_bloch_native_core`: transpose the length-weighted magnetic curl, including seam phases. |
| `fdtd/cuda/kernels/adjoint.cu` | `reverse_magnetic_to_e_standard_kernel`, `reverse_magnetic_to_e_bloch_kernel` and their launchers: multiply the transposed curl contribution by the receiving E edge's length fraction. Carry the CPML length transpose through the derivative pullback. Matching reverse wrappers/schemas in `cuda/backend.py` and `cuda/extension.cpp` also change. |

Geometry derivatives of new face areas require a native PyTorch compilation
graph and a matching coefficient pullback; a field-adjoint transpose alone does
not qualify shape optimization. That extension needs its own scoped design and
must not be advertised by the forward correction.

PCB review invariants: preserve `_pec_edge_open_fractions`' staircase two-node
minimum; preserve **both** planar-sheet maxima (`compiler/materials.py:1863`,
`fdtd/runtime/materials.py:1116`); continue to obtain volume geometry through
`compiler/planar_pec.py::volume_pec_structures`. `compile_planar_pec` and
`attach_planar_pec` need no algorithm change. A zero-thickness `PlanarPEC` sheet
must not create a volume face fraction. The separate two-sided `SurfaceMesh`
sheet runtime is outside this volume-PEC correction.

Qualification for that future change should rerun the existing aligned-face and
node-consistency checks, verify lossless energy in a closed PEC cavity, repeat
this coax speed/attenuation sweep and offset-varied Mie convergence, and exercise
tiny positive faces, CPML, nonuniform grids and forward/adjoint parity. No such
numerical fix or new physics tests are included here.

## Warning validation and reproduction

One CPU warning test was added:
`tests/materials/compiler/test_subpixel_spec.py::test_conformal_pec_warns_about_systematic_bias`.
It checks the message, warning category, and continued construction. Running the
whole file in the maxwell environment gives **8 passed, 2 expected warnings**;
no existing test failed because of the warning, so none was changed.

GPU commands used (one per invocation, no environment assignments after `env`):

```bash
env CUDA_VISIBLE_DEVICES=1 python <scratch>/witwin_cov_conf/run_gpu.py .worktrees/cov-conf <scratch>/witwin_cov_conf/coax_diagnosis.py
env CUDA_VISIBLE_DEVICES=1 python <scratch>/witwin_cov_conf/run_gpu.py .worktrees/cov-conf <scratch>/witwin_cov_conf/sphere_diagnosis.py --case pec --methods staircase conformal --levels 20 30 --offsets 1 --output <scratch>/witwin_cov_conf/sphere.jsonl
env CUDA_VISIBLE_DEVICES=1 python <scratch>/witwin_cov_conf/run_gpu.py .worktrees/cov-conf <scratch>/witwin_cov_conf/coax_loss_diagnosis.py
```

The coax scratch scripts import the existing fixture by file path, run
`Scene -> Simulation.fdtd(...).prepare() -> prepared.run()`, and perform exactly
the annular Ex/Ey functional and two-wave fitting described above. They do not
alter solver coefficients. To recreate the sphere script from the retained
source commit, extract the harness with `git show`, replace
`ROOT = Path(__file__).resolve().parents[2]` by `ROOT = Path.cwd()`, and restrict
only `_parameters("pec")`'s x tuple from `(1., 2.5)` to `(1.,)`.

Source qualification:
`4c69d038:docs/assessments/lq1-analytic-convergence-2026-10-30.md`.
Earlier coax context: `docs/memory/rf-coax-curved-pec-slow-wave.md` and
`docs/memory/rf-coax-port-gates-staircase-line.md`.
