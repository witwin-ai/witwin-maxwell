# Same-grid FDFD/FDTD consistency comparisons

Use the same scene, grid and compiled Yee materials. Run FDTD at the physical
carrier `f0`, and read `dt = result.stats()['dt']` from that run. Its leapfrog
time derivative has the symbol

```text
w0 = 2*pi*f0
w_eff = (2/dt)*sin(w0*dt/2)
f_eff = w_eff/(2*pi) = sin(pi*f0*dt)/(pi*dt)
```

Solve FDFD at `f_eff`, not at `f0`. The expression tends to `f0` as `dt -> 0`;
there is no additional factor of two in `f_eff`. Keep the FDTD excitation and
DFT at `f0`. Check that the frequency change has not altered the intended
spatial source profile. For dispersive or conductive media, the discrete ADE
or conduction time symbols also need consideration; changing only the carrier
does not prove exact operator equivalence for those media or for PML.

The electric phasors also carry a half-step offset. With the present recorded
fields, fit `E_FDFD = a * E_FDTD` on the common sampling points. Report the
residual `||a E_FDTD - E_FDFD|| / ||E_FDFD||`, `arg(a)`, and its wrapped
difference from `w0*dt/2`. Equivalently, multiplying FDFD by
`exp(-i*w0*dt/2)` aligns the expected phase with FDTD. Reversing the direction
of the fitted scale reverses its phase sign. One global complex scale removes
source normalization and phasor reference; it must be shared by all components
and points, rather than fitted separately to each component or plane region.

At about 16 cells per free-space wavelength, ignoring these time symbols can
create percent-level false disagreement. The retained p16s evidence reports
9.85% disagreement with master's node-averaged edge materials and 0.15% for Part A sample
0 with shared Yee materials and the time correction. This demonstrates
consistency, not accuracy against a physical reference at that resolution.

Reanalysis of `<scratch>/witwin_cov_sub3/diag/{branchA_fdfd_0_w,master_fdtd_0}.pt`
gives `f0 = 17,634,952,302.877415 Hz`, `dt = 1.8415673237493016e-12 s`,
`f_eff = 17,604,373,496.789318 Hz`, residual **0.148835%**, fitted phase
**0.102325508 rad**, expected **0.102026212 rad**, difference
**0.000299296 rad** (0.01715 degrees). These are historical artifacts from
different source trees, not fresh same-branch acceptance of the speed change.
Fresh Part A + compile-reuse acceptance passes p16s 0/168 and the LQ-1 eps_r 4,
x=1 sphere at 20 cells per wavelength, polarized samples=3. Residuals are
0.148830% / 0.006847% / 0.103990%, respectively; phase differences from
`w0*dt/2` are +0.000299726 / -0.000000134 / -0.000059833 rad. All pass the
<=0.5% consistency gate. The sphere uses its LQ-1 geometry/grid with a common
CW point-current excitation; full scene/sampling details and fresh records are
in [the compile qualification](../memory/yee-edge-compile-perf.md).

## Existing comparisons omitting the rule

Read-only source audit at `cf9ab951`, 2026-10-31. No listed comparison or
tolerance is changed. There is one file under `tests/validation/cross_solver/`.
The eight field comparisons below score magnitudes, so a global half-step phase changes
their current metrics by **zero**. Frequency mismatch can still change their
magnitude patterns. The estimates below are sensitivity estimates, not
measured reductions of their current errors; that requires paired GPU runs.

For the nondispersive scenes here, the 30-step-per-cycle source bound is tighter
than the spatial CFL bound: `dt = 1/(30*f0)`. Thus `w0*dt/2 = pi/30`,
`1 - f_eff/f0 = 0.18267%`. An unaligned complex comparison would contain
10.467% phase-only disagreement. That number must not be attributed to these
magnitude metrics. A propagation-path estimate is `k0*L*(1-f_eff/f0)`; it bounds
neither resonant sensitivity nor cancellation near field nulls.

| Comparison | Current tolerance | Estimated removable contribution to its present metric |
|---|---|---|
| `tests/validation/cross_solver/test_fdfd_vs_fdtd.py::test_cw_dipole_amplitude_agreement` | Ex-magnitude plane relative L2 <5%; FDFD solve residual <=1e-7 | About 0.2–1 percentage point from carrier/amplitude/near-field sensitivity; phase removal contributes zero. A 0.24 m path has 0.00919 rad carrier phase mismatch before taking magnitudes. |
| `TestFDFDvsFDTD::test_point_source_free_space` in the same file | Total-magnitude correlation >=0.70; peak offset <=12 cells; FDFD residual <=1e-6 | Typically <=1e-3 correlation deficit and less than one cell; the free-space radiation magnitude weakly depends on carrier. A 1.28 m path has 0.0490 rad complex phase mismatch, which is not itself a magnitude error. |
| `TestFDFDvsFDTD::test_point_source_with_cube` | Same correlation, peak and solve limits | Away from resonances, about 0–a few percentage points of magnitude-pattern error, corresponding roughly to <=1e-3 correlation deficit; the 0.3 m eps_r 4 path adds about 0.023 rad carrier phase mismatch. Peak movement cannot be predicted without the fields. |
| `TestFDFDvsFDTD::test_polarized_dielectric_consistency` | Same correlation, peak and solve limits; polarized samples=2 | Same time-symbol estimate as the cube above. Material consistency is now shared; neither estimate is a measured post-Part-A error. The test's old node-versus-edge docstring is stale. |
| `benchmark/scenes/planned.py`: `fdfd_dielectric_slab` | No field pass/fail tolerance; reports Ex-magnitude L2, fitted-shape L2, Linf and correlation; flux disabled | Roughly 0–a few percentage points of magnitude error away from resonances. The 0.8 m vacuum path gives 0.0613 rad and the extra optical path of the 0.12 m eps_r 4 slab gives 0.00919 rad. Spatial errors at only 3.75 cells/lambda0 can dominate. |
| `fdfd_sigma_e_slab` in the same file | Same reporting-only field metrics | Order 0.2–a few percentage points away from nulls/resonances; conductivity time-symbol differences remain after carrier correction. No fixed fraction of the current error is established. |
| `fdfd_diag_aniso_slab` in the same file | Same reporting-only field metrics | Order 0–a few percentage points in magnitude away from resonances; direction-dependent optical path changes the sensitivity. Phase removal contributes zero. |
| `fdfd_drude_sphere` in the same file | Same reporting-only field metrics | Unquantified near dispersive resonances; away from them, a sub-percent carrier change suggests order 0.2–a few percentage points in magnitude. Its material-rate timestep bound may reduce dt below the 30-step value; use the actual result dt and the ADE symbol. |
| `benchmark/fdfd_vs_fdtd.py::run_case`, default sizes 32/48/64/96/128 | No field-agreement tolerance; iterative FDFD solve tolerance defaults to 1e-6 | Reports memory and time only, not field error. There is no reported disagreement for the rule to remove, so an error-removal estimate is inapplicable. Correcting the frequency can change iteration counts/time; its effect on performance is unmeasured. |

`benchmark/runner.py::run_benchmarks` sends the same uncorrected
`scenario.frequencies` to both runtimes. Its phase alignment and fitted shape
scale do not correct the frequency. The 1% constants in `benchmark/metrics.py`
are runtime-convergence limits, not FDFD/FDTD agreement tolerances; that
convergence branch runs only for primary FDTD scenarios. The four FDFD rows
are unmeasured in the checked-in `benchmark/RESULTS.md`, so there is no recorded
current field error to partition numerically.
