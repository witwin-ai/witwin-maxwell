# CPML passive closure: B0 CPU investigation, 2026-11-02

**Result: no replacement selected. B0 is incomplete and B1 must not start from
these candidates.** Native reproduction passes, but time centering, an equivalent
constitutive ADE realization, and the tested one-memory orderings fail passivity.
This is a rejection of these implementations, not a nonexistence proof for every
energy-stable PML or every one-memory modification.

All code and raw results are in `scripts/dev/fdtd/cpml_passive/`. No runtime,
kernel, test, ABI, or default was changed. All calculations used the `maxwell`
environment on CPU. NumPy/SciPy perform the stepping and linear algebra; the
repository material compiler and `_cpml_profile_1d` run on CPU to supply inputs.

## 1. Validity and model scope

`model.native_matrix` applies the full y/z Yee stencil to the native probe's
real basis states, then performs its complex projection. It retains the compiled
polarized subpixel edges, float32 coefficients, node duplicates, physical memory
scales, zero exterior tangential E nodes, and H-before-E ordering. Float64
arithmetic gives:

| Native case | Reference max oscillatory \(|\lambda|\) | Float64 replay | Absolute error |
|---|---:|---:|---:|
| q1 TM | 1.0001352599793165 | 1.0001352583369676 | 1.65e-9 |
| q1 TE | 0.9999636620828670 | 0.9999636621692127 | 8.64e-11 |
| q0 TE | 0.9999480673476152 | 0.9999480677174210 | 3.70e-10 |

The TM frequency is 0.2317945846 GHz. The optional binary32/FMA replay differs
entrywise from the measured matrices by at most 9.12e-9, 9.11e-9, and 1.12e-16,
respectively. `validate_model.py` checks the float64 entries too, exits nonzero
on a tolerance violation, and records input hashes. The four additional native
variant radii (alpha scales 1/4, PEC/PMC q1) agree within 3.12e-12.

`model.Block` is a separate, float64 semi-discrete harmonic Galerkin reduction:
\(D_z^+=F\), \(D_z^-=-F^T=B\), with the exact Yee transverse symbol
\(k_h=2\sin(k_yh_y/2)/h_y\). PEC/PMC use half-wave sin/cos bases; periodic
uses full-wave harmonics. Its positive diagonal material metric uses projected
inverse permittivities, with half endpoint quadrature for wall nodes. It accepts
arbitrary staggered `(kappa,b,c)` vector triples through `coefficients=`.

**Scope limitation:** the finite box's subpixel corners weakly mix transverse
harmonics. The Galerkin block uses transverse means for periodic fields; it is
not entrywise identical to the native real-cosine projection. Its default TM
radius is 1.00013523626568 (2.37e-8 below the native radius). Sweep and large-cell
numbers below describe this harmonic model, not a qualified full-y native
operator. The full native-basis candidate check independently confirms rejection:
c1 gives TM 1.00015689318052 and TE 1.00009312996617.

## 2. Exact discrete work identity

Write the bare, mass-weighted curl pair as

\[
 M_h(H^+-H)=-\Delta t\,CE+Q_h,\qquad
 M_e(E^+-E)=\Delta t\,C^*H^++Q_e.
\]

Spatial quadrature is included in the inner products. The usual CFL-positive
staggered storage, expressed entirely in the current state, is

\[
 W(E,H)=\tfrac12\|E\|_{M_e}^2+\tfrac12\|H\|_{M_h}^2
            -\tfrac{\Delta t}{2}\Re\langle H,CE\rangle.
\]

Expanding differences and using the adjoint curl pair cancels all bare-curl
terms exactly, including the cross term in W:

\[
 W^+-W=\Re\langle (E^++E)/2,Q_e\rangle
       +\Re\langle (H^++H)/2,Q_h\rangle.
\]

The present closure uses
\(\psi^+=b\psi+c\,d\), \(d_{\rm corrected}=d/\kappa+\psi^+\).
At default kappa=1 in TM, \(Q_{e,y}=\Delta t\,\psi_{ey,z}^+\) and
\(Q_{h,x}=\Delta t\,\psi_{hx,z}^+\). For the leading harmonic eigenmode:

| Work / initial W | Float64 model | Native evidence |
|---|---:|---:|
| Electric psi | -4.38413665e-4 | -4.38421757e-4 |
| Magnetic psi | +7.08904485e-4 | +7.08913256e-4 |
| Net increment | +2.70490820e-4 | +2.70491833e-4 |

These are dimensionless per-step energy fractions, not watts. The model identity
normalized residual is -2.48e-16; the largest normalized residual across 32 random
valid states is 4.61e-16. The largest positive magnetic
contribution is at z=-0.923076923 m, inside the low-side layer; the mode has the
symmetric high-side contribution too. `energy.py` computes these values directly.

## 3. Candidates and storage obstruction

For supplied coefficients let \(d_0=1/\kappa\),
\(r=-\log b/\Delta t\), \(w=-c/(1-b)\) in active cells. Zero-strength slots
have r=w=0. This recovers the held-input ADE for the real default. For the
moment-matched CFS inputs it defines an **effective** ADE from the supplied
pole, DC residue, and direct coefficient; it does not assert recovery of the
pre-discretization physical alpha/kappa. This interpretation is shared by c1/c2.

**c1: time-centered ADE, existing memory count.** For each derivative:

\[
 \beta={1-r\Delta t/2\over1+r\Delta t/2},\quad
 \gamma={-rw\Delta t\over1+r\Delta t/2},\quad
 \psi^+=\beta\psi+\gamma d,\quad
 \widetilde d=d_0d+(\psi^++\psi)/2.
\]

The H memory goes from n-1/2 to n+1/2 with E at n; the E memory goes from n
to n+1 with H at n+1/2. Thus memory output is centered at the curl's time level;
interior Yee steps do not change. Default harmonic TM/TE radii are
1.00015687080539 / 1.00009312035044. Full simultaneous Maxwell/ADE midpoint
integration, available as `global_midpoint`, also retains the unstable TM mode;
it additionally changes the interior time integrator and is inadmissible.

**c2: constitutive stretched-coordinate ADE candidate.** Define
\(\alpha=r(1-w/d_0)\), \(G(s)=d_0(s+\alpha)/(s+r)\), and let f be the
bare mass-scaled curl of the opposite field. Tangential and normal updates are
realized as

\[
 \dot p_t=f-rp_t,\quad \dot x_t=d_0f-d_0(r-\alpha)p_t;\qquad
 \dot p_n=f-\alpha p_n,\quad
 \dot x_n=f/d_0+(r-\alpha)p_n/d_0.
\]

Local trapezoidal ADE kicks, H then E, keep the interior Yee update. This adds
normal-field polarization memory, but has the **same oscillatory spectrum** as
c1 in the harmonic block. It is an unsplit realization of coordinate stretching,
not a successfully transplanted energy-stable SBP/SAT discretization. Therefore
the requested energy-stable c2 candidate remains unimplemented/unselected.

An energy estimate in the literature is not automatically a passive storage
identity. For example, [Duru, sections 4.3 and 5](https://arxiv.org/pdf/1405.0536)
derives growth bounds and treats auxiliary boundary penalties; those theorems
do not supply non-increasing storage for this heterogeneous Yee truncation.

There is also a local constitutive obstruction: the normal permittivity is
\(\epsilon_n(s)=\epsilon G(s)\). At \(s=i\omega\),
\(\Re[s\epsilon_n(s)]=-\epsilon d_0(r-\alpha)\omega^2/(r^2+\omega^2)<0\)
when conductivity is nonzero. An ordinary positive local material-storage
argument cannot certify this response. This does not rule out a different
coupled/nonlocal Maxwell storage or a changed spatial/boundary realization.

**c3: ordering search, no passive variant found.** Keep native b,c,kappa and use
\(\widetilde d=d_0d+(1-f)\psi+f\psi^+\), independently for E/H. Both old memory
(f=0) and averaged memory (f=1/2) fail. All 81 pairs from
f in {-1,0,0.5,1,1.5,2,3,4,8} fail periodic q1 TE/TM at dt, dt/2, dt/4.
The best pair is (3,3), still with worst radius 1.00006523475612. This bounded
search does not exclude other coefficients or layouts.

**Storage certificate or countercertificate.** For any positive definite P and
eigenvector v, \(v^*(A^*PA-P)v=(|\lambda|^2-1)v^*Pv\). Each rejected candidate
has an oscillatory eigenvalue outside the unit disk, so **no positive definite
quadratic storage can satisfy the required inequality for that block**. This
countercertificate is stronger than testing random states. `energy.json` also
reports positive random increments for physical field storage plus unit-weight
auxiliaries, after removing inactive dummy psi states. These trial storages are
not presented as valid certificates.

As a control, scalar matched conductivity plus global midpoint has
\(P A_0+A_0^*P=0\) and
\(W^+-W=-\Delta t\,\bar x^*PD\bar x\le0\), D>=0. Its computed maximum
increment eigenvalue is 4.57e-16. It passes the spectral sweep, but is not a
stretched-coordinate closure and changes the interior time integrator; it is
not eligible for selection.

## 4. Sweeps and reflection

The 1,458-case Cartesian sweep covers periodic/PEC/PMC orders 1-3, TE/TM, three
time steps, base/y-half/z-half grids, 6/12/24 layers, and real/CFS/explicit-alpha0
profiles. CFS here has f_min=0.3 GHz and profile angle 35 degrees, including the
repository's grazing cap. Real and explicit-alpha0 are intentionally identical.

| Method | Cases above 1+1e-12 | Worst oscillatory radius |
|---|---:|---:|
| Current | 927 | 1.00016298613802 |
| c1 | 1031 | 1.00022236141708 |
| c2 realization | 1031 | 1.00022236141708 |
| c3 old | 1030 | 1.97520803880137 |
| c3 average | 1021 | 1.00025275865298 |
| Passive conductivity control | 0 | 0.999999998957589 |

The supplementary 288-case fixed-k sweep uses carrier angles 0/10/30/45/60/75,
f_min=0.3/2 GHz, Floquet orders 0-3, TE/TM and three time steps, on the baseline
12-layer grid. c1/c2 each fail 96 cases (worst 1.0000893465772).

The 4x-period, 100,000-step matrix-free check advances all 103 resolved Fourier
harmonics together. Current TE selects q3, fitted radius 1.00018574332921;
c1 TE selects q3, 1.00020354793202. Current/c1 TM select q4, with radii
1.00013268100381 / 1.00015641776973. This is the harmonic-model large-cell check;
it does not replace a full-y native qualification or test arbitrary slab seams.

Reflection is an exact frequency-domain elimination of the discrete map,
using a one-sided PEC-backed 12-layer vacuum stack, h=c/(20*3 GHz),
dt=h/(2*c*sqrt(2)), and discrete incoming/outgoing waves at the entrance.
With lambda=exp(-i omega dt), the current transfer is
\(g_j=1/\kappa_j+c_j\lambda/(\lambda-b_j)\). For c1/c2 substitute
\(g_j=G_j(2(\lambda-1)/(\Delta t(\lambda+1)))\). Eliminating fields gives

\[
 [\operatorname{diag}(g_e) B\operatorname{diag}(g_h)F
   +(\Omega_h^2-k_h^2)I]u=0,\quad
 \Omega_h^2=-{(\lambda-1)^2\over c^2\Delta t^2\lambda}.
\]

TE/TM have the same vacuum electric tangential-field reflection. All five
frequencies 0.3/0.6/1/2/3 GHz, four angles 0/35/60/80, and a below-cutoff
harmonic ky=1.25*omega/c are recorded for each profile. Evanescent R is an
amplitude ratio of decaying/growing solutions, not reflected power.

| Real profile, 0.3 GHz | Current amplitude R | c1 = c2 amplitude R |
|---|---:|---:|
| Normal | 1.05272903e-5 | 1.24601792e-5 |
| 35 degrees | 9.96538739e-6 | 2.87736478e-5 |
| 60 degrees | 2.07499209e-4 | 1.17142566e-3 |
| 80 degrees | 4.72587735e-2 | 9.18563015e-2 |
| Evanescent | 0.505937227 | 0.548661472 |

c1/c2 fail the no-worse-reflection test on 64/75 profile/frequency/angle rows;
c3 old/average fail 71/65. This includes explicit-alpha0 duplicate rows.
`verify_operators.py` independently checks the bare curl energy, staggered Yee
storage, and the frequency-domain wave elimination against actual amplification
eigenvectors, for both polarizations and real/CFS profiles.

## 5. Recommendation and core-path mirror status

**Do not implement these candidates in CUDA.** Rank first by passivity: all
eligible candidates are rejected before state cost can decide. Among the rejected
methods c1 has better reflection than c3 old, but it still fails both acceptance
criteria. c2 adds state without improving the spectrum or reflection.

There is **no chosen-closure mirror contract**. In particular, there is no
approved new state count, coefficient ABI, FMA grouping, or PML E/H update to
freeze. Interior cells must keep today's Yee step. The exact, *rejected* c1
arithmetic is specified here only to make this experiment reviewable:

1. Compute beta/gamma above in float64, then round coefficient vectors once.
2. `tmp = beta * psi_old`; `psi_new = fma(gamma, derivative, tmp)`.
3. `sum = psi_old + psi_new`; `mid = 0.5f * sum` (two rounded operations).
4. `corrected = fma(inv_kappa, derivative, mid)`; preserve the existing grouped
   corrected-derivative curl and final field FMA.

c1 would retain one float per stretched derivative, dense on the target Yee
field or in the existing low/high slab compression, with beta/gamma vectors on
the same node/half-point axis indices. c2 instead needs one polarization per
active field: three auxiliaries versus two in this 2D block, six versus four
for a single 3D PML face; extra normal-field storage has no existing psi slot.
Both candidates leave interior Yee cells unchanged. Neither is recommended.

Remaining B0 work is to construct and certify a genuinely different coupled
spatial/boundary or constitutive closure, including a valid c2 energy-stable
discretization; run it in the calibrated full stencil; and pass reflection before
issuing a B1 contract. No inference of impossibility for that remaining work is
warranted by this search.

## Reproduction

Activate `maxwell`; set `CUDA_VISIBLE_DEVICES=''`, `OPENBLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1`, and a writable `MPLCONFIGDIR`. From the repository root:

```bash
python scripts/dev/fdtd/cpml_passive/validate_model.py --variants --candidate-check --output scripts/dev/fdtd/cpml_passive/results/validation.json
python scripts/dev/fdtd/cpml_passive/verify_operators.py
python scripts/dev/fdtd/cpml_passive/energy.py --output scripts/dev/fdtd/cpml_passive/results/energy.json
python scripts/dev/fdtd/cpml_passive/sweep.py --workers 3
python scripts/dev/fdtd/cpml_passive/bloch_sweep.py --output scripts/dev/fdtd/cpml_passive/results/bloch_sweep.json
python scripts/dev/fdtd/cpml_passive/reflection.py --output scripts/dev/fdtd/cpml_passive/results/reflection.json
python scripts/dev/fdtd/cpml_passive/ordering_search.py --output scripts/dev/fdtd/cpml_passive/results/ordering_search.json
python scripts/dev/fdtd/cpml_passive/power.py --output scripts/dev/fdtd/cpml_passive/results/power.json
```

The spectral sweep intentionally exits 1 when no candidate passes; native
validation and operator verification must exit 0. JSON/JSONL retains the failed
cases rather than clipping unstable eigenvalues or relabeling a bound as a
passivity certificate.

## B0b-1: multi-axial PML

This experiment uses the same CPU model, grids, material projections, native
basis injection, and oscillatory-frequency mask (>1 MHz) as B0. It changes no
runtime, kernel, test, ABI, or default. `mpml.py` supplies the candidate profiles,
the full sweeps, exact discrete reflection, native-basis projections, and the
matrix-free large-cell check. `model.py` accepts optional tangential profiles in
its existing stencil and harmonic block; omitting them preserves B0 arithmetic.

### Candidate and state cost

At each target Yee location in a layer normal to z, the normal physical profile
is unchanged. For either tangential derivative, the exact choice is

\[
 \sigma_t(z)=p\sigma_z(z),\qquad
 \alpha_t(z)=p\alpha_z(z),\qquad
 \kappa_t(z)=1+p(\kappa_z(z)-1).
\]

Here sigma includes the normal builder's CFS conductivity compensation. That
compensation is retained, rather than recomputed at the smaller alpha: the
builder receives both alpha_max and shift_rate multiplied by p. Its ratio
alpha_max/(vacuum_constant*shift_rate), and therefore the sigma compensation,
stays fixed. Conductivity scaling is implemented by R_t=exp(p log R_z), with
the same physical layer depth and grading order. The normal profile is built by
the unchanged B0 call. Magnetic sigma and alpha retain their matched mu0/eps0
scaling. Physical kappa is distinguished from the builder's effective returned
kappa in CFS mode.

Tangential coefficients are sampled at the **target field's z position**:
TE Ex/Hz use z nodes; TM Ez/Hx use z half-points. In particular, an electric
tangential derivative does not always use a z-node profile. The same physical
position must be used for sigma, alpha and kappa before discretization.

This is **not implementable by replacing today's axis-only vectors alone**.
The y derivative currently reads `cpml_*_y[j]`, independent of z. A z-face
tangential profile requires z-dependent loads, and different z staggering for
different owning fields. Scalar field-update coefficients cannot represent a
general CFS tangential transfer pole. A field receiving both normal and
tangential derivatives generally has two distinct driven poles, so its existing
single normal memory cannot carry both. At alpha=0, kappa=1 a special
constitutive rewrite can trade a tangential pole for direct field damping; that
does not realize the full real/CFS family with today's coefficients and update
form, nor is it the discrete candidate tested here.

With the physical Yee fields kept as the state variables and independent curl
drives, generic CFS TE requires two poles at Ex (y/z), one at Hy (z), and one at
Hz (y); TM requires two at Hx (y/z), one at Ey (z), and one at Ez (y). Distinct
normal/tangential poles with nonzero residues cannot be combined into one local
scalar memory. Four auxiliary scalars, versus B0's two, are therefore necessary
for this generic local derivative realization. No optimization of a special
alpha0 transfer or change of physical field variables is used to claim a lower
state count for the CFS sweep.

The implemented minimal extension within the one-memory-per-driven-derivative
layout activates existing slot names `psi_ex_y` / `psi_hz_y` (TE), or `psi_ez_y` /
`psi_hx_y` (TM). The reduced block has four memories instead of two: **two
additional real scalars per corresponding PML cell**, or 8 bytes in float32
(16 bytes for separate complex real/imag banks). At 12 layers the active
harmonic state adds 44 scalars for TE and 48 for TM, because exterior tangential
nodes are excluded. Dense runtime storage already reserves those slot names;
the current compressed y-slabs are empty on a periodic/PEC/PMC y axis and would
need additional z-slab support. No extra slot names or non-psi state are used.
This state count is a minimal generic realization in this derivative layout,
not a proof of global minimality under arbitrary changes of field variables.

For a general isolated 3D z face, activating every tangential derivative uses
eight additional existing psi slots (twelve active versus four). The equal-pole
x/y contributions to each normal component could be combined into one memory,
reducing that count to six additional active scalars; that 3D compression is
outside this reduced experiment and is not implemented or qualified here.

### Exact coefficients and experimental mirror description

Write v=eps0 for electric memory or mu0 for magnetic memory, with the matched
physical sigma/alpha for that family. For the real/explicit-alpha0 builder,

\[
 u=(\sigma/\kappa+\alpha)\Delta t/v,\quad
 b=e^{-u},\quad d=1/\kappa,\quad
 c={\sigma(b-1)\over\kappa(\sigma+\alpha\kappa)}.
\]

For the existing moment-matched CFS builder, use

\[
 A=1-u/2+u^2/12,\quad B=1+u/2+u^2/12,\quad
 w={\sigma\over\kappa(\sigma+\alpha\kappa)},
\]
\[
 b=A/B,\quad c=-{wu\over AB},\quad
 d=1/\kappa-w+w/A,\quad \kappa_{\rm returned}=1/d.
\]

The builder's denominator mask is sigma+alpha*kappa > 1e-12; zero-strength
positions use c=0 and the builder's ordinary b/d values. All profile construction
uses float64 before rounding the returned vectors once to float32 for the
native-equivalent replay; inverse-kappa is then formed from that rounded
returned kappa, exactly as in B0. Each normal coefficient vector is unchanged.
Only the tangential derivative's b/c/inverse-kappa values become active. Their
index must follow the owning field's z node or half-point rather than the
derivative's y coordinate. The existing material field-update coefficients
ce/ch do not change. Thus there is no valid proposal consisting solely of new
values for the existing one-dimensional `cpml_*_e_y` / `cpml_*_h_y` vectors.

The experimental float32 order for **both** normal and tangential derivative
slots is today's recurrence, with no time centering:

1. Form the existing Yee derivative, preserving its subtraction and grid-scale
   multiplication order.
2. `tmp = b * psi_old` (one rounded multiply).
3. `psi_new = fma(c, derivative, tmp)`.
4. `corrected = fma(inv_kappa, derivative, psi_new)`.
5. Subtract the two corrected derivatives in the existing curl order, then
   apply the existing final field FMA with ce/ch and the old field.

H updates precede E updates. For the reduced TM stencil the curls are
`corrected_z - corrected_y` for Hx, `corrected_z` for Ey, and `-corrected_y`
for Ez. For TE they are `-corrected_z` for Hy, `corrected_y` for Hz, and
`corrected_y - corrected_z` for Ex. Uncorrected zero x derivatives stay zero.
There is no additional field decay, mixed-time field term, or changed E/H
integrator. PML cells acquire the tangential corrected derivatives; interior
cells must retain the exact existing code path. This describes a rejected CPU
experiment, **not an approved B1/core-path contract**.

### Reflection elimination

For every normal or tangential coefficient triple, use the actual fresh-memory
transfer \(g(\lambda)=d+c\lambda/(\lambda-b)\). Let
\(\Omega_h^2=-(\lambda-1)^2/(c_0^2\Delta t^2\lambda)\) and
\(K_j^2=\Omega_h^2-k_h^2g_{t,e,j}g_{t,h,j}\). The vacuum equations are

\[
 [\operatorname{diag}(g_{z,e})B\operatorname{diag}(g_{z,h})F
     +\operatorname{diag}(K^2)]u=0\quad\text{(TE)},
\]
\[
 [\operatorname{diag}(g_{z,e})B
       \operatorname{diag}(g_{z,h}/K^2)F+I]u=0\quad\text{(TM)}.
\]

TE K is sampled on interior z nodes, TM K on half-points. Spatially variable
tangential damping makes the two polarizations different. Reusing B0's scalar
K outside the layer, or assuming TE=TM inside the layer, would test a different
candidate. At p=0 K is constant and both reduce to B0's equation. The stack,
PEC backing, frequency list, grid, time step, incident/outgoing decomposition,
and evanescent entrance normalization are unchanged from `reflection.py`.
The independent eigenvector elimination check covers TE/TM, real/CFS, and
p=0/0.01/0.1/0.3; its maximum relative residual is 3.70e-15.

### Results and decision

**No tested p passes. No replacement is selected, and this candidate must not
start B1.** The 28 ratios cover 48,888 logical spectral cases (1,458 Cartesian
plus 288 fixed-k per ratio), 3,360 reflection rows, and 56 matrix-free large-cell
runs of 100,000 steps each. Eighteen ratios pass both finite spectral sweeps;
none passes the fixed large-cell flatness rule, and none passes reflection.
The acceptance limits were not changed.

The closest rejected ratio is **p=0.01349**, defined here as the smallest
*evaluated* ratio with zero failures in both finite eigenvalue sweeps. It is not
called passive: it fails rule 1's large-cell check and rule 2's reflection check.
The finite-sweep threshold is bracketed by 0.01348 (two failures) and 0.01349
(zero), a width of 1e-5. This is an evaluated bracket, not a monotonicity proof.
The limiting block is periodic TE q3, dt/4, base grid, six layers, real and
explicit-alpha0 profiles. Its frequency is 0.5358340409 GHz. The bisection used
to select the refined grid is only a heuristic: all four added ratios were then
run through the complete 1,458+288 cases.

For p=0.01349:

- Cartesian worst oscillatory radius: 0.999999999999948; 0/1,458 failures.
- Fixed-k worst oscillatory radius: 1.0000000000000118; 0/288 failures against
  the fixed 1+1e-12 bound.
- Large-cell fitted radius: TE 1.0000000399597992, TM 1.0000000901047699;
  both fail the fixed flatness limit.
- Worst reflection ratios to current: normal 1, 60 degrees 20837.0741,
  80 degrees 293.218946, evanescent 52.5227047. The limits are respectively
  2, 2, 1, and strictly below 1.
- Restricting reflection to the real profile alone still fails: the respective
  worst ratios are 1, 821.457444, 11.9457752, and 2.40317024. Rejection therefore
  does not depend on whether CFS reflection is included in the gate.

**p=0 reproduces B0 exactly**, with zero numerical difference in every stored
rho, frequency, and all-eigenvalue radius across 1,458 Cartesian and 288 fixed-k
cases, all 120 corresponding reflection rows, both 100,000-step fitted radii,
and the three native-basis float64/binary32 replays. The B0 Cartesian baseline
remains 927 failures, worst radius 1.000162986138022; fixed-k remains 98 failures,
worst 1.0000683933921046. Native validation including four profile/wall variants
passes; the existing bare-operator/reflection check's maximum relative residual
is 4.14e-15.

### Full finite-sweep p-curve

Radii below are oscillatory radii with B0's unchanged >1 MHz mask. Counts include
real/explicit-alpha0 duplicate cases, as required by the Cartesian definition.
Every row uses the complete sweep; the fixed-k column always has 288 cases.

| p | Worst Cartesian radius | Failures / 1458 | Worst fixed-k radius | Failures / 288 |
|---:|---:|---:|---:|---:|
| 0 | 1.000162986138022 | 927 | 1.000068393392105 | 98 |
| 0.001 | 1.000058267950664 | 334 | 1.000000000000013 | 0 |
| 0.002 | 1.000024525750050 | 228 | 1.000000000000011 | 0 |
| 0.003 | 1.000013276184263 | 168 | 1.000000000000010 | 0 |
| 0.004 | 1.000004762801417 | 134 | 1.000000000000011 | 0 |
| 0.005 | 1.000002426741248 | 102 | 1.000000000000012 | 0 |
| 0.0075 | 1.000000680400430 | 24 | 1.000000000000011 | 0 |
| 0.01 | 1.000000209812623 | 18 | 1.000000000000011 | 0 |
| 0.01347 | 1.000000000003086 | 2 | 1.000000000000010 | 0 |
| 0.01348 | 1.000000000001517 | 2 | 1.000000000000015 | 0 |
| 0.01349 | 0.999999999999948 | 0 | 1.000000000000012 | 0 |
| 0.0135 | 0.999999999998381 | 0 | 1.000000000000012 | 0 |
| 0.015 | 0.999999999982162 | 0 | 1.000000000000016 | 0 |
| 0.02 | 0.999999999976043 | 0 | 1.000000000000018 | 0 |
| 0.025 | 0.999999999969898 | 0 | 1.000000000000012 | 0 |
| 0.03 | 0.999999999963749 | 0 | 1.000000000000013 | 0 |
| 0.04 | 0.999999999951406 | 0 | 1.000000000000015 | 0 |
| 0.05 | 0.999999999939015 | 0 | 1.000000000000016 | 0 |
| 0.06 | 0.999999999926560 | 0 | 1.000000000000015 | 0 |
| 0.07 | 0.999999999914041 | 0 | 1.000000000000017 | 0 |
| 0.085 | 0.999999999895184 | 0 | 1.000000000000014 | 0 |
| 0.1 | 0.999999999876248 | 0 | 1.000000000000017 | 0 |
| 0.125 | 0.999999999844605 | 0 | 1.000000000000016 | 0 |
| 0.15 | 0.999999999813014 | 0 | 1.000000000000014 | 0 |
| 0.175 | 0.999999999781616 | 0 | 1.000000000000013 | 0 |
| 0.2 | 0.999999999750510 | 0 | 1.000000000000017 | 0 |
| 0.25 | 0.999999999689409 | 0 | 1.000000000000015 | 0 |
| 0.3 | 0.999999999630012 | 0 | 1.000000000000017 | 0 |

### Full reflection p-curve

Each entry is the **maximum pointwise amplitude ratio R(p)/R(0)** over all five
frequencies, both polarizations, and real/CFS/explicit-alpha0 profiles. It is not
a ratio of independently maximized amplitudes. Normal is analytically the same
stack as B0 because k_y=0; the implementation reuses B0's solve and band scaling
there. Every individual amplitude, baseline, frequency, polarization and
pass/fail result is retained in `results/mpml_reflection.json`. The evanescent
quantity remains the entrance amplitude ratio, not reflected power.

| p | Normal (<=2) | 60 degrees (<=2) | 80 degrees (<=1) | Evanescent (strictly <1) |
|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 1 | 1 |
| 0.001 | 1 | 2033.24515 | 53.6193556 | 4.40060453 |
| 0.002 | 1 | 4105.39421 | 86.6773081 | 8.68321787 |
| 0.003 | 1 | 6131.00147 | 116.749078 | 13.1296609 |
| 0.004 | 1 | 8055.7261 | 144.83147 | 17.5862802 |
| 0.005 | 1 | 9856.91144 | 168.967324 | 21.9495877 |
| 0.0075 | 1 | 13815.8633 | 217.153717 | 32.1912203 |
| 0.01 | 1 | 17101.8775 | 253.802978 | 41.3546003 |
| 0.01347 | 1 | 20817.9665 | 293.021672 | 52.4634065 |
| 0.01348 | 1 | 20827.5232 | 293.120344 | 52.4930618 |
| 0.01349 | 1 | 20837.0741 | 293.218946 | 52.5227047 |
| 0.0135 | 1 | 20846.619 | 293.317478 | 52.5523352 |
| 0.015 | 1 | 22214.9791 | 307.35388 | 56.8617855 |
| 0.02 | 1 | 26025.2735 | 345.747833 | 69.5696409 |
| 0.025 | 1 | 28992.1618 | 375.298315 | 80.3217755 |
| 0.03 | 1 | 31377.5142 | 399.095812 | 89.6497995 |
| 0.04 | 1 | 34988.1298 | 435.696889 | 105.289043 |
| 0.05 | 1 | 37593.2735 | 463.06 | 118.116901 |
| 0.06 | 1 | 39554.6326 | 484.626205 | 128.972234 |
| 0.07 | 1 | 41074.6736 | 502.245615 | 138.34368 |
| 0.085 | 1 | 42786.1747 | 523.574116 | 150.274782 |
| 0.1 | 1 | 44029.3864 | 540.643411 | 160.234684 |
| 0.125 | 1 | 45437.4361 | 562.854943 | 173.586838 |
| 0.15 | 1 | 46325.3783 | 579.907598 | 183.955286 |
| 0.175 | 1 | 46895.1822 | 593.531484 | 192.164996 |
| 0.2 | 1 | 47261.4312 | 604.731245 | 198.771841 |
| 0.25 | 1 | 47634.0486 | 622.17178 | 208.619357 |
| 0.3 | 1 | 47746.2801 | 635.209862 | 215.491795 |

At p=0 the evanescent ratio equals one, so the strict-improvement requirement
fails even though the three propagating tests pass. At p=0.01349 the following
rows attain the worst pointwise ratios (normal has ties):

| Metric | Profile / polarization / GHz | Current R | M-PML R | Ratio |
|---|---|---:|---:|---:|
| normal | real / TE / 0.3 | 1.052729026e-05 | 1.052729026e-05 | 1 |
| 60deg | cfs / TE / 0.6 | 7.404660845e-06 | 0.1542914664 | 20837.07406 |
| 80deg | cfs / TE / 3 | 0.001018998723 | 0.2987897315 | 293.2189461 |
| evanescent | cfs / TE / 0.3 | 0.003816191097 | 0.2004366779 | 52.52270467 |

### Full 4x-period, 100,000-step p-curve

This is precisely B0's 5.12 m, ny=205, 103-harmonic matrix-free Fourier stack,
with the same field seed (431), zero initial memories, norm rescaling every
100 steps, and a log-norm fit over steps 50,000-100,000. Each row below represents
both complete 100,000-step runs. **Every ratio fails** the fitted-radius bound
1+1e-12. A positive finite-window norm slope alone does not distinguish a slowly
converging static component from exponential modal growth; it is still a failure
of the prespecified flatness rule. No eigenvalue radius was substituted for that
fit and no transient window was discarded beyond B0's existing fit window.

| p | TE fitted radius | TM fitted radius | Flatness gate |
|---:|---:|---:|---|
| 0 | 1.000185743329212 | 1.000132681003809 | fail |
| 0.001 | 1.000084216805853 | 1.000000174834646 | fail |
| 0.002 | 1.000000166628845 | 1.000000121925367 | fail |
| 0.003 | 1.000000051422848 | 1.000000108333041 | fail |
| 0.004 | 1.000000048019202 | 1.000000099934692 | fail |
| 0.005 | 1.000000044830643 | 1.000000093694742 | fail |
| 0.0075 | 1.000000039523544 | 1.000000085289967 | fail |
| 0.01 | 1.000000038024642 | 1.000000084638089 | fail |
| 0.01347 | 1.000000039941259 | 1.000000090064590 | fail |
| 0.01348 | 1.000000039950522 | 1.000000090084675 | fail |
| 0.01349 | 1.000000039959799 | 1.000000090104770 | fail |
| 0.0135 | 1.000000039969090 | 1.000000090124875 | fail |
| 0.015 | 1.000000041487990 | 1.000000093204101 | fail |
| 0.02 | 1.000000047214851 | 1.000000102299952 | fail |
| 0.025 | 1.000000052642246 | 1.000000107480067 | fail |
| 0.03 | 1.000000057446568 | 1.000000108974659 | fail |
| 0.04 | 1.000000065629448 | 1.000000105805724 | fail |
| 0.05 | 1.000000072733348 | 1.000000100771111 | fail |
| 0.06 | 1.000000079471384 | 1.000000097211855 | fail |
| 0.07 | 1.000000086201901 | 1.000000095443354 | fail |
| 0.085 | 1.000000096389569 | 1.000000094829709 | fail |
| 0.1 | 1.000000106377131 | 1.000000095265420 | fail |
| 0.125 | 1.000000121432024 | 1.000000096926259 | fail |
| 0.15 | 1.000000133404527 | 1.000000099544441 | fail |
| 0.175 | 1.000000141843458 | 1.000000103016172 | fail |
| 0.2 | 1.000000146913083 | 1.000000107062314 | fail |
| 0.25 | 1.000000149025083 | 1.000000115729629 | fail |
| 0.3 | 1.000000144261032 | 1.000000123953503 | fail |

### Artifacts and reproduction

- `mpml.py`: profile construction, finite sweeps and threshold refinement,
  exact TE/TM reflection, native-basis replay, matrix-free stepping, and summary.
- `model.py`: optional tangential coefficient triples in the primary validated
  stencil and harmonic block; the normal-only path remains exact B0.
- `results/mpml_summary.json`: decision, full p-curve, exact p=0 comparisons,
  case counts, threshold bracket, input-file list and SHA-256 hashes.
- `results/mpml_sweep.json`, `results/mpml_refine.json`: every Cartesian and
  fixed-k result, including failures and the refinement's grid-selection probes.
- `results/mpml_reflection.json`: all 3,360 individual reflection rows.
- `results/mpml_power_initial.json`, `results/mpml_power_coarse.json`,
  `results/mpml_power_curve.json`, `results/mpml_power_refined.json`: all 56
  large-cell runs. The refined report also retains every fit checkpoint.
- `results/mpml_verify.json`: native-basis spectra at every evaluated p,
  exact B0 native replay, and independent TE/TM wave-elimination residuals.
- `results/mpml_native_regression.json`, `results/mpml_operator_regression.json`:
  the original native-artifact and bare-operator controls, both passing.

Activate `maxwell` and use the CPU-only environment below. A compact rerun may
consolidate the four saved power files into `mpml_power_all.json`; the fit and
gate numbers are the same. The original four report names and source hashes are
recorded in the saved summary. All work remains uncommitted.

```bash
conda activate maxwell
export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/cpml-mpml-mpl
mpml_dir=scripts/dev/fdtd/cpml_passive
mpml_ratios='0 0.001 0.002 0.003 0.004 0.005 0.0075 0.01 0.01347 0.01348 0.01349 0.0135 0.015 0.02 0.025 0.03 0.04 0.05 0.06 0.07 0.085 0.1 0.125 0.15 0.175 0.2 0.25 0.3'
python "$mpml_dir/mpml.py" sweep --workers 3
python "$mpml_dir/mpml.py" refine --workers 3
python "$mpml_dir/mpml.py" verify --p $mpml_ratios
python "$mpml_dir/validate_model.py" --variants --candidate-check --output "$mpml_dir/results/mpml_native_regression.json"
python "$mpml_dir/verify_operators.py" > "$mpml_dir/results/mpml_operator_regression.json"
python "$mpml_dir/mpml.py" power --p $mpml_ratios --workers 3 --output "$mpml_dir/results/mpml_power_all.json"
python "$mpml_dir/mpml.py" summarize \
  --sweeps "$mpml_dir/results/mpml_sweep.json" "$mpml_dir/results/mpml_refine.json" \
  --powers "$mpml_dir/results/mpml_power_all.json" \
  --output "$mpml_dir/results/mpml_summary.json"
```

The summary command requires complete 1,458/288 sweeps, both polarizations of
the 100,000-step run at every p, and the full native-basis p-curve. It asserts
exact B0 equality before writing the decision. This experiment rejects the
specified M-PML realization; it does not prove that every multi-axial absorber
or different spatial/state realization must fail. No CUDA implementation or
approved coefficient/state ABI follows from these results.
