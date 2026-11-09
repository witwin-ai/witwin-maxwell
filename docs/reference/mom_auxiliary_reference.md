# Independent auxiliary-source reference

`benchmark.mom_auxiliary_reference` is a benchmark-only independent Maxwell
reference. It represents scattered vacuum fields by electric dipoles on an
interior auxiliary shell and internal dielectric fields by dipoles on an exterior
shell. It uses no MoM/FDTD kernels, RWG basis, surface-current assembly, or field
decoder. It is a fundamental-solution boundary collocation method, not a volume
dipole approximation, finite-element solver, or PMCHWT discretization.

The qualified campaign is a centered, axis-aligned ellipsoid with semiaxes
(0.1,0.08,0.065) m, eps_r=2.25, mu_r=1, zero conductivity, and vacuum exterior at
300 MHz. The incident field is a unit Ex plane wave along +z with phase 0.23,
meaning `E_inc,x = exp(i*k0*z - i*0.23)` under `exp(-i*omega*t)`. The implementation
can fit other positive passive scalar dielectrics, but that does not confer
reference accuracy outside the measured campaign.

For `g=exp(i*k*r)/(4*pi*r)`, each basis field is
`E=(I+grad grad/k^2)g p` and `H=curl(E)/(i*k*eta)`.
The moment coefficient `p` has units V: the physical electric-dipole prefactor is
absorbed into this expansion coefficient. Thus decoded E/H use V/m and A/m.
The outgoing far amplitude is
`(I-dd^T) sum_s p_s exp(-i*k*d.s)/(4*pi)` in V. Both fields satisfy homogeneous
Maxwell equations away from their auxiliary sources. CPU SciPy is used only by
the separate analytic Mie oracle; the reference kernels and solve use CUDA
complex128. Independently written vacuum constants use `mu0=4*pi*1e-7`; its
relative difference from the shared permeability is `5.44e-10`, far below the
`5e-4` empirical reference budget.

At exact analytic ellipsoid boundary points, an overdetermined system enforces
`E_out,t - E_in,t = -E_inc,t` and the corresponding H relation. H rows are scaled
by vacuum impedance; positive ellipsoid area weights and column normalization
precede CUDA SVD least squares. The report retains all singular values, cutoff,
retained rank/condition, coefficient norm, and true weighted collocation residual.
The final source rule has order 16 (512 points on each shell), shell factors
0.6 and 2.25, collocation order 33, and cutoff `1e-12`. Its matrix is 8712 by 3072;
peak allocation was about 3.37 GB. The matrix is ill-conditioned and truncated:
conditioning and cutoff sensitivity must remain visible.

Qualification requires sphere comparisons against independent Mie E/H and complex
far fields, two accepted final orders, independently shifted holdout rules for
E_t/H_t and normal D/B, additional denser holdout, shell perturbations, and SVD
cutoff perturbations. Far-integrated scattering and forward-optical-theorem
extinction changes each have their own `5e-4` budget at the last order transition
and every source-shell/SVD control; the field budget does not imply a power budget.
Normal conditions use eps_r E inside versus E outside, and
H continuity because mu_r=1. These controls provide empirical sampled accuracy,
not a rigorous error bound. Early orders and failed source placements remain
archived. No physical gate is certified merely by solving or archiving.

`solve_reference(...)` returns reusable fitted coefficients with `fields(...)`,
`far_field(...)`, and `holdout(...)`; queries must lie strictly in their declared
physical interior or exterior. One fit supplies every field, far-field and power
observable. MoM comparison reads immutable report arrays at identical points,
separating three fixed exterior probes from the 72 weighted control-surface
samples. Comparison requires a complete `physical_contract` matching the fixed
scene (including material, origin/orientation, source amplitude/phase/direction/
polarization and phasor convention). Legacy reports require `--mom-contract` with
a `mom-reviewed-physical-contract-v1` sidecar: its `reports` map keys are exact
original report SHA256 values, and each entry records `physical_contract`, a
`review` identifying reviewer and evidence, `source_sha256` for the frozen driver
and scene definition, and `source_snapshot_directory` relative to the sidecar.
The loader verifies those source hashes against both the immutable source bytes
and the original report provenance. This binds an explicit reviewer assertion; it does
not infer missing physics or authenticate an arbitrary reviewer. Missing, stale,
conflicting, or mismatched contracts fail closed. Original report bytes remain
unchanged. Without MoM inputs, an independent control sphere still measures power
closure; an empty input list cannot silently pass an unmeasured power check.

Run `python -m benchmark mom auxiliary --output report.json` for the qualification
campaign. Add `--mom-report small.json --mom-report fine.json` for the frozen
ellipsoid MoM comparisons, and `--mom-contract review.json` when those reports
need reviewed legacy metadata. Each run writes repository-relative source hashes and
an immutable source snapshot alongside its output, and stores physical samples,
weights, moments, and raw complex results. The report includes independent far
angular scattering, forward optical-theorem extinction, control-surface Poynting
closure, RCS, and MoM errors. Lossless absorption is reported absolutely or against
incident projected power, never divided by an analytic zero.
