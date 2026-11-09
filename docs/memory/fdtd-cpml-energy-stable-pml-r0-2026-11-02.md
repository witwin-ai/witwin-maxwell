# Energy-stable PML R0: transverse-memory diffusion on the B0 Yee block

Date: 2026-11-02. Branch: `cpml/energy-stable-pml-r0`.

**Result: NO-GO for the single discretization investigated here. No candidate
passes R0 or is selected for R2.** An exact rational piecewise-constant
counterexample has two amplification roots outside the unit disk. Consequently
there is no positive definite quadratic storage with non-positive source-free
increment for this map. The complete requested R1 measurements are retained as
rejection diagnostics, not as qualification of a candidate that passed R0.
This does not establish nonexistence for the entire published family.

## Selection and literature mapping

One family was selected for examination: the Appelö–Hagstrom–Kreiss (AHK)
Maxwell auxiliary-variable formulation. Its distinguishing addition here is
**transverse diffusion of the normal-derivative memories**. Equation (49) and
Lemma 11 of the [primary preprint reproduced in Appelö's thesis, paper 3,
pp. 10–13](https://kth.diva-portal.org/smash/get/diva2:12464/FULLTEXT01#page=95)
use positive constant sigma, alpha and diffusion, and `1 + eta*sigma > 0`.
For eta=0, their Fourier argument uses tau=alpha+nu*ky^2. On a Yee grid the
corresponding quantity is alpha+nu*kh^2, where

\[
 k_h={2\over h_y}\sin(k_yh_y/2),\qquad
 k_y=k_{\mathrm{Bloch}}+2\pi q/L_y.
\]

There is no Maxwell condition in that result requiring a fitted scalar-alpha
threshold relative to sigma. Positivity of the constant-coefficient continuous
symbol does not certify a heterogeneous, fully discrete map. Its derivative
energies also require care at static modes.

The experiment sets eta=0 and fixes nu=c0*hy/2 before any measurements. There
is no parameter scan. The existing real/CFS/explicit-alpha0 profiles are kept;
no scalar shift is strengthened. This is an IMEX transplant of the **parabolic
auxiliary term**, not a claim to reproduce a published fully discrete method.

No second family is carried. In particular, the [Duru SBP/SAT
analysis](https://arxiv.org/pdf/1405.0536), Theorems 4.5/5.3 and (5.17), includes
bounds permitting energy growth and uses a different spatial/time treatment.
The B0 pair already satisfies SBP. An interface SAT needs a new coupling and
auxiliary-boundary proof; attaching that citation to the previous constitutive
ADE would repeat the rejected c2 argument. No such reuse is made here.

## Exact update and state

Use B0's normalized fields and memory scalings. Set t=c*dt, h=hz, F=Dz+,
B=-F*, and positive material inverses a,av. The coefficient arrays are the
unchanged B0 `(kappa,b,c_memory)` triples, including their CFS interpretation.
Write p=h*psi_e and q=eta0*h*psi_h, with B0's polarization-specific normalization.
Define active-memory resolvents

\[
 R_e=(I+\nu\Delta t L_{y,e})^{-1},\quad
 R_h=(I+\nu\Delta t L_{y,h})^{-1},\qquad L_y=-D_y^-D_y^+\succeq0.
\]

Use the owning field's transverse parity and quadrature: periodic duplicates
are eliminated; PEC/PMC sine nodes have prescribed zero endpoints, cosine nodes
have half-weight Neumann endpoints, and half nodes use the matching ghost
parity. Complex Bloch wraps use exp(i*ky*Ly), including the duplicate node.
The harmonic reduction is exactly R=1/(1+nu*dt*kh^2). Diffusion is applied only
to driven normal-memory z slots. Inactive memories remain constrained to zero.

For TM, `(u,v,w)=(Ey,Ez,eta0*Hx)`:

\[
\begin{aligned}
 q^+&=R_h(b_hq+c_hhFu),\\
 w^+&=w+t(Fu/\kappa_h+q^+/h+k_hv),\\
 p^+&=R_e(b_ep+c_ehBw^+),\\
 u^+&=u+ta(Bw^+/\kappa_e+p^+/h),\\
 v^+&=v-t a_v k_hw^+.
\end{aligned}
\]

For TE, `(u,v,w)=(Ex,eta0*Hy,eta0*Hz)`:

\[
\begin{aligned}
 q^+&=R_h(b_hq+c_hhFu),\\
 v^+&=v-t(Fu/\kappa_h+q^+/h),\\
 w^+&=w+t k_hu,\\
 p^+&=R_e(b_ep+c_ehBv^+),\\
 u^+&=u-ta(Bv^+/\kappa_e+p^+/h+k_hw^+).
\end{aligned}
\]

Old E and p are at n, old H and q at n-1/2. New H and q are at n+1/2;
new E and p at n+1. The diffusion solve is part of each fresh memory update,
before that memory enters its field kick. There is no memory time centering.
The full stencil applies the y resolvent; `Block` applies its harmonic symbol.
Both preserve the original H-before-E field order.

State cost is unchanged: three Yee field scalars and two normal memories per
polarized PML cell, counting staggered ownership; complex fields double the
real scalar count. There are no tangential memories or additional dynamic
states. A full stencil needs a transverse line solve in PML memory slots. The
additional scalar nu and the line operator are new computation, so this is not
implementable solely by editing the existing axis coefficient vectors. At a
single harmonic only, effective b and c are divided by `1+nu*dt*kh^2`.
For a driven supplied triple, its held-input ADE interpretation is
`d0=1/kappa`, `r=-log(b)/dt`,
`sigma=-c_memory*r/(d0*(1-b))`, and `alpha=r-sigma`.
The zero-strength limit sets both rates to zero. Adding transverse diffusion
corresponds to the continuous pole `r+nu*kh^2`; the tested discrete resolvent
is a first-order splitting of that extra term. For CFS this is an extension
of the supplied effective discrete pole, not a rederivation of the physical
profile-builder parameters or a continuous-theorem certificate.
Interior field cells keep the existing Yee equations, coefficients and ordering;
no global-midpoint field solve is introduced. No CUDA arithmetic or ABI contract
is proposed.

## Fully discrete storage identity, piecewise constant first

Let the field kick residuals relative to the bare Yee curl be Qe,Qh. With
material/quadrature masses Me,Mh, use the same B0 storage

\[
 W_Y=\tfrac12\|E^n\|_{M_e}^2+
       \tfrac12\|H^{n-1/2}\|_{M_h}^2
       -{\Delta t\over2}\Re\langle H^{n-1/2},CE^n\rangle.
\]

The exact H-before-E identity, for complex states as well as real states, is

\[
 \Delta W_Y=\Re\langle(E^++E)/2,Q_e\rangle+
             \Re\langle(H^++H)/2,Q_h\rangle.                 \tag{1}
\]

For example in TM, Qh=t*((1/kappa_h-1)*Fu+q+/h) and
Qe=t*a*((1/kappa_e-1)*B*w_new+p_new/h). The TE signs follow the update above.

Within one constant layer, let xi denote either normalized memory, d its
normalized derivative drive, delta=nu*dt, and use the transverse SBP norm.
Since `(I+delta*Ly)*xi+ = b*xi+c*d`, exact expansion gives

\[
\begin{aligned}
 \|\xi^+\|^2-\|\xi\|^2
 ={}&(b^2-1)\|\xi\|^2+2bc\Re\langle\xi,d\rangle+c^2\|d\|^2\\
    &-2\delta\langle\xi^+,L_y\xi^+\rangle
     -\delta^2\|L_y\xi^+\|^2.                              \tag{2}
\end{aligned}
\]

Thus for any positive z-local auxiliary weights S, the storage
`W_Y + (||p||_S^2+||q||_S^2)/2` has increment (1) plus one half of the
S-weighted (2) for each memory. Diffusion contributes negative terms, but the
remaining field/memory/drive work has no established sign. Dropping it is not
an energy proof.

Equations (1)–(2) also hold for graded b(z),c(z),kappa(z),S(z): these
coefficients commute with Ly, which acts only in y. They do not commute with
F/B across z interfaces. A homogeneous-symbol storage therefore cannot be
summed across a piecewise or graded profile without additional interface and
commutator terms.

### Exact homogeneous-symbol identity and its limit

`exact_uniform()` supplies a genuine, limited discrete storage certificate.
It uses rational normalized `t=1/4`, normal and transverse symbols `1/2`,
`b=4/5`, `c_memory=-1/10`, and resolvent denominator `17/16`. This can use
`hy=2`, normalized c=1/4, dt=1 and nu=c*hy/2. The five-state rational map A,
full rational P,Q and positive leading principal minors are in
`results/es_verify.json`. They satisfy **exactly**

\[
 P\succ0,\quad Q=I-rr^T/(r^Tr)\succeq0,\quad
 A^TPA-P=-Q,\quad (A-I)r=0.
\]

Consequently W=x*Px/2 has increment -x*Qx/2. SymPy verifies the matrix equality,
all five positive minors, and the scalar form of (2) without floating-point
rounding. This is a single homogeneous symbol, not the required layer proof.

### Exact piecewise-constant countercertificate

`exact_piecewise()` uses the **primary Block stepping algebra with rational
operands**, not an independently coded time integrator. Its normalized data are:

- dt=1, c=1/4, hz=hy=1/2, kh=1/2 and nu=c*hy/2=1/16;
- four z intervals with zero exterior tangential E and one PML interval per side;
- all three interior tangential E nodes have eps=4; normal E has eps=(1,4,4,1);
- the two interface E memories and two exterior H memories have b=4/5,
  c_memory=-1/10, kappa=1; all interior memories are inactive;
- these constant memory coefficients correspond to positive
  sigma=alpha=log(5/4)/2, not negative damping; the field CFL is admissible.

After inactive memories are removed, A has **15 states**. Its characteristic
polynomial factors into `(z-1)^3` and two degree-six polynomials, all saved
exactly. Transform each nonstatic factor with `z=(1+s)/(1-s)` and clear its
denominator. The degree-12 product has Routh first-column signs

```
+ + + + + + + + + + + - +
```

There are exactly **two right-half-plane roots**, hence exactly two roots of A
outside the unit disk. This integer sign-count is an exact arithmetic
countercertificate, independent of eigenvalue tolerance. The numerical worst
radius is **1.0024382578970061**, eigen-residual about **1.43e-15**.
For an unstable eigenvector and any P>0,

\[
 v^*(A^*PA-P)v=(|\lambda|^2-1)v^*Pv>0.
\]

The required non-increasing positive storage is therefore impossible for this
piecewise-constant transplant. Its R0 exit gate fails before grading. It is
not carried as a certified candidate into R1 or R2.

## Graded diagnostics and qualification limits

The complete rejection diagnostics are in
`docs/assessments/fdtd-cpml-energy-stable-pml-r1-2026-11-02.md` and the compact
`results/es_*.json` files. They cover the prescribed 1,458 Cartesian and 288
fixed-k cases, both 4x-period 100,000-step runs, the explicit eps=4 TM slab
at k_parallel=0.55*k0 for 320,000 steps, and all requested reflection rows.
The G2b model uses f0=2 GHz (the existing upper fixed-k carrier), thickness
0.1 m and the standard B0 domain/grid; these choices are recorded explicitly.
It is a reduced-model check, not a claim to reproduce an unspecified 3D scene.

The native-equivalent stencil check includes real periodic/PEC/PMC and complex
fixed-k Bloch fields. The latter uses complex injection/projection and twisted
y differences and memory diffusion. B0's material seam projection limitation
remains: the harmonic transverse-mean slab does not reproduce every full-y
subpixel corner. No GPU execution, runtime edit, test edit, ABI change,
commit, default change, or feature-inventory addition is part of this work.

## Reproduction

Use the `maxwell` environment, CPU only, from the worktree root:

```bash
conda activate maxwell
export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/cpml-es-mpl
es_dir=scripts/dev/fdtd/cpml_passive
python "$es_dir/energy_stable.py" verify
python "$es_dir/energy_stable.py" sweep --workers 3
python "$es_dir/energy_stable.py" power --workers 2
python "$es_dir/energy_stable.py" g2b
python "$es_dir/energy_stable.py" reflection
python "$es_dir/validate_model.py" --output "$es_dir/results/es_native_regression.json"
python "$es_dir/energy_stable.py" summary
```

Commands exit successfully when diagnostic computation/identities are valid;
`es_summary.json` separately records every failed qualification gate and the
NO-GO recommendation. Raw sweep cases are in `/tmp/cpml-es-sweep-cases.json`,
with their SHA-256 in the compact sweep report. Preserve that external file if
per-case archival is needed; it is not added to git. Source and report hashes
are retained. All repository changes are left uncommitted.
The repository ignores new `docs/` files; both this derivation and the R1
assessment exist on disk and need explicit force-add when preparing a commit.
