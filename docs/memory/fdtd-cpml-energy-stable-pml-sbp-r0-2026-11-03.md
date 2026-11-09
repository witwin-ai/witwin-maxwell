# SBP/SAT PML interface R0, 2026-11-03

**NO-GO for this round.** Neither of the two specified interface fluxes admits
the required positive definite quadratic storage on the full reduced block.
Exact rational arithmetic certifies **two amplification roots outside the unit
disk** for each flux, already with piecewise-constant PML coefficients. The
central flux has radius **1.0018592946627674**; the impedance-upwind flux has
radius **1.0018262466460217**. Grading and R1 were not run because the first R0
gate fails. This rejects these discrete couplings, not every possible SBP/SAT
PML formulation or storage construction.

This is the additional, CPU-only, timeboxed R0 authorized on 2026-11-03, on
`cpml/energy-stable-pml-sbp`, starting at `4ead00aa`. The prior
[plan](../plans/fdtd-cpml-energy-stable-pml-2026-11-02.md),
[AHK R0 rejection](fdtd-cpml-energy-stable-pml-r0-2026-11-02.md),
[R1 rejection](../assessments/fdtd-cpml-energy-stable-pml-r1-2026-11-02.md), and
[passive-closure derivation](fdtd-cpml-passive-closure-derivation-2026-11-02.md)
were read. No previously rejected bulk closure was revived. There is no CUDA,
runtime, test, ABI, public feature, default, or git-history change.

## Selection and limits of the literature mapping

The candidate splits the interior and the two PML regions into three staggered
SBP blocks, duplicates their tangential electric interface traces, and couples
them with central or impedance-upwind SAT fluxes. The normal PML remains B0's
fresh-memory derivative realization. The SAT-corrected derivative also drives
its auxiliary recurrence; applying a field SAT while omitting its auxiliary
contribution is not the candidate examined here. There is no transverse-memory
diffusion, tangential stretching, stronger shift, or modified interior integrator.

[Duru, sections 5.1–5.3](https://arxiv.org/pdf/1405.0536) motivates including
boundary penalties in the auxiliary equations. Its semidiscrete estimates and
Runge–Kutta experiments do not certify this staggered-time transplant. Equation
(5.17) has a positive volume term. The fully discrete identities and exact
countercertificates below are independent of that published estimate.

The two choices are central flux, tau=0, and the normalized vacuum-impedance
upwind flux, tau=1/2. Both interfaces are in vacuum. These choices are fixed
flux definitions, not a fitted damping search. Midpoint evaluation applies only
to the local SAT self-penalty within each H or E kick; the fields retain Yee
H-before-E time levels. No global-midpoint solve is introduced.

## Grid, operators, and exact updates

Use TM phase coordinates `(u,v,w)=(Ey,Ez,eta0*Hx)`, with the transverse phase
absorbed into v, and the exact transverse Yee symbol k. Let h be the z spacing
and t=c*dt. Each subdomain uses electric nodes and magnetic half nodes. The
outer electric endpoints are prescribed zero. At each interior/PML interface
there are independent left/right electric endpoint traces. Their electric
quadrature weights are h/2, while ordinary nodes and all half nodes have weight
h. Denote the diagonal quadratures by He and Hh.

F0 contains the ordinary within-subdomain forward differences, including the
independent endpoint traces. Je takes right-minus-left electric trace jumps.
Jh takes right-minus-left magnetic trace jumps, using the nearest half node
as each subdomain's lowest-order boundary trace. S puts a +1 on both adjacent
magnetic half nodes at each interface. Define

\[
 F=F_0+\tfrac12 H_h^{-1}S J_e,\qquad
 B=-H_e^{-1}F^T H_h,\qquad H_e B+F^T H_h=0.                 \tag{1}
\]

This is a new split-grid coupling, rather than relabeling the existing conforming
B0 pair. With no PML and conforming electric traces, its central map is exactly
the primary B0 map, including its Yee time ordering; the script checks that
matrix intertwining identity using `model.Block.step` with rational operands.

For the selected flux set

\[
 D_e=\tau J_e^T J_e,\quad D_h=\tau J_h^T J_h,\quad
 L_e=H_e^{-1}D_e,\quad L_h=H_h^{-1}D_h.                    \tag{2}
\]

These matrices are confined to interface traces and their adjacent half nodes.
The electric interface duplicate and the two adjacent magnetic half nodes on
either side form the interface closure. The normal electric update everywhere
and the strictly interior volume stencils retain the original Yee equations.
In particular, the first/last magnetic half node in the interior subdomain is
in the SAT support: this is an interface boundary correction, not an unchanged
boundary stencil. No penalty or altered coefficient reaches a deeper interior
cell.

Let Eu and Ev be positive diagonal relative permittivities on the two electric
staggerings. Normalize memories as p=h*psi_e and q=eta0*h*psi_h. Old u,v,p are
at n; old w,q at n-1/2. New w,q are at n+1/2, and new u,v,p at n+1. For
kappa=1, with diagonal be,ce,bh,ch and inactive memories constrained to zero,
the complete candidate is

\[
\begin{aligned}
 d_h&=Fu^n-L_h(w^{n+1/2}+w^{n-1/2})/2,\\
 q^{n+1/2}&=b_h q^{n-1/2}+c_h h d_h,\\
 w^{n+1/2}&=w^{n-1/2}+t(d_h+q^{n+1/2}/h+kv^n),\\
 d_e&=Bw^{n+1/2}-L_e(u^{n+1}+u^n)/2,\\
 p^{n+1}&=b_e p^n+c_e h d_e,\\
 u^{n+1}&=u^n+t E_u^{-1}(d_e+p^{n+1}/h),\\
 v^{n+1}&=v^n-t E_v^{-1}kw^{n+1/2}.                       \tag{3}
\end{aligned}
\]

The SAT self-penalty makes the corresponding trace kick implicit. Eliminating
the fresh memory gives the exact, directly implementable small interface solves
used in the script. With ge=I+ce and gh=I+ch,

\[
\begin{aligned}
 (I+\tfrac t2 g_hL_h)w^+
 &= (I-\tfrac t2 g_hL_h)w
       +t(g_hFu+b_hq/h+kv),\\
 (I+\tfrac t2 E_u^{-1}g_eL_e)u^+
 &= (I-\tfrac t2 E_u^{-1}g_eL_e)u
       +tE_u^{-1}(g_eBw^++b_ep/h).                         \tag{4}
\end{aligned}
\]

Then evaluate dh,de and the two memories from (3). Noncommuting diagonal
profile factors stay on the indicated side of the SAT matrices. Equations
(3) and (4) are checked against each other exactly, without recovering rational
coefficients from floating-point output. No choice of float32 grouping is
proposed by this rejected CPU experiment.

## Discrete storage calculation, including the interface

Write E=(u,v), H=w, and

\[
 M_e=\operatorname{diag}(E_uH_e,E_vH_h),\quad M_h=H_h,\quad
 K=H_h[\,F\ \ kI\,].
\]

The staggered storage at the actual state time levels is

\[
 W_Y^n=\tfrac12(E^n)^*M_eE^n
       +\tfrac12(H^{n-1/2})^*M_hH^{n-1/2}
       +\tfrac t2\Re[(H^{n-1/2})^*KE^n].                 \tag{5}
\]

The positive cross-term sign follows the TM convention H+=H+t(Fu+kv).
It is consistent with B=-He^-1 F^T Hh. The symmetric matrix of (5) is
`P_Y=[[Me,t*K.T/2],[t*K/2,Mh]]`. For the exact block below, all **19 LDL
pivots are strictly positive**, checked rationally. Thus this is a positive
storage for the bare fields; CFL positivity has not been assumed from a
continuum estimate.

With arithmetic averages ubar,wbar between the old/new time levels, expanding
(3) gives the exact identity

\[
\begin{aligned}
 W_Y^{n+1}-W_Y^n={}&
 -t\,\overline u^*D_e\overline u
 -t\,\overline w^*D_h\overline w\\
 &+\frac t h\Re\big[
      \overline u^*H_ep^{n+1}
     +\overline w^*H_hq^{n+1/2}\big].                     \tag{6}
\end{aligned}
\]

The script verifies (6) as a full symmetric matrix equality for **all** states,
including arbitrary active memories and complex harmonic phases. With PML
removed, the second line vanishes and (6) is an exact non-increasing discrete
storage certificate for either SAT flux. This isolates the interface control
from the PML failure.

For a trial full storage WY+(p*Sp p+q*Sq q)/2 with positive diagonal auxiliary
weights, each memory's increment additionally contains

\[
 \|\xi^+\|_S^2-\|\xi\|_S^2
 =(b^2-1)\|\xi\|_S^2
   +2bch\Re\langle\xi,d\rangle_S+c^2h^2\|d\|_S^2.      \tag{7}
\]

The interface losses in (6) have the desired sign, but the fresh-memory work
and the drive terms in (7) have no such sign. They cannot be discarded or
bounded by a positive growth rate under the requested gate. The counterexample
below rules out repairing this by **any** positive definite quadratic full-state
matrix, including dense field/auxiliary/interface cross terms. It is stronger
than failure of this diagonal-weight trial.

## Exact piecewise-constant countercertificate

Use rational normalized dt=1, c=1/4, h=hy=1/2, and k=1/2. This transverse
symbol is realizable by a fixed Bloch wavenumber `ky=4*asin(1/8)`.
The z nodes before duplication are
`(-3/2,-1,-1/2,0,1/2,1,3/2)`; interfaces are at -1 and +1. There is one PML
edge per side and four interior edges. A centered dielectric slab is separated
from each PML by vacuum. In state order,

- u has 7 entries (two copies at each interface), with eps=(1,1,1,4,1,1,1);
- v has 6 half-node entries, with eps=(1,1,4,4,1,1);
- w has 6 half-node entries;
- p has 2 active entries, the PML-owned electric interface endpoints;
- q has 2 active entries, the exterior PML half nodes.

The full matrix is therefore **23 by 23**, including interior, interface,
PML fields and every active auxiliary. No dynamic variable is eliminated from
the characteristic polynomial. Electric and magnetic active memory coefficients
are constant in the layer: b=4/5, c_memory=-1/10, kappa=1. Interior memories
are absent. The held-input interpretation has positive effective
sigma=alpha=log(5/4)/2, so the counterexample does not use negative damping.

All matrices are built over the rationals. Factor the exact characteristic
polynomial in z, remove only its explicitly recorded `(z-1)` factors, and map
each remaining degree-d factor f by

\[
 \widetilde f(s)=(1-s)^d f((1+s)/(1-s)).                  \tag{8}
\]

This maps the outside of the unit disk to Re(s)>0. Every Routh first-column
entry is nonzero, and no all-zero row occurs. The sign counts therefore need
no epsilon substitution or floating-point root classification.

| Coupling | Exact factors by degree | Routh signs for unstable factor | Outside roots | Worst radius | Eigen-residual |
|---|---|---|---:|---:|---:|
| Central, tau=0 | (z-1)^7, degree 8, degree 8 | + + + + + + + - + | **2** | **1.0018592946627674** | 2.02e-15 |
| Impedance upwind, tau=1/2 | (z-1)^5, degree 9, degree 9 | + + + + + + + + - + | **2** | **1.0018262466460217** | 1.08e-15 |

The second nonstatic factor in each row has an entirely positive Routh first
column and zero outside roots. The JSON stores both full rational matrices,
all characteristic-factor coefficients, all transformed polynomial coefficients,
and the exact Routh first columns, not just the displayed signs. Floating-point
eigenvalues only supply the radius and residual annotations.

For an unstable eigenpair Av=lambda*v and **every** P>0,

\[
 v^*(A^*PA-P)v=(|\lambda|^2-1)v^*Pv>0.                   \tag{9}
\]

The minimum contradiction is already a growing full-block oscillatory mode.
Its storage multiplication per step is **1.0037220463021777** for central flux
or **1.0036558284688555** for upwind flux, independent of the choice of P.
Thus no positive definite quadratic storage with non-positive increment exists
for either tested piecewise-constant map. The same expanding orbit also
excludes a non-increasing storage bounded above and below by positive multiples
of the full-state squared norm. A degenerate functional that hides the growing
mode would not certify the requested full-state passivity.

The precise obstruction is the residual full field/auxiliary PML dynamics after
the SBP interface balance has been closed. The SAT control is itself discretely
dissipative and includes the auxiliary forcing, yet the complete map still has
an exact unstable pair. This is not an assertion that interface fluxes can never
stabilize a differently realized PML.

## State and coefficient changes of the rejected proposal

The TM proposal keeps two driven normal-memory families and adds **one electric
trace scalar per interface**, or **two additional scalars for this two-sided
harmonic block**. Complex fields double real storage. With N total half-node
intervals and L PML intervals per side, the tested layout has 3N+1 field scalars
and 4L active memory scalars, totaling 3N+1+4L; here N=6,L=1 gives 23. The
conforming B0 layout for the same data has 21 states. No separate SAT history or
new temporal auxiliary is introduced. TE and 3D state layouts were not developed
after the TM obstruction.

The original volume be,bh,ce,ch,kappa and material update factors retain their
values. New data are the split endpoint quadratures, trace jump operators,
central flux lifts, and the fixed interface impedance penalty. Interface curl
drives and their memory drives change together, and tau=1/2 requires the local
coupled trace solves (4). These changes cannot be represented solely by new
values of today's axis profile vectors. The unmodified interior volume kick
remains H-before-E leapfrog. No approved coefficient, kernel, or ABI contract
follows from a rejected R0.

## Exit decision, retained criteria, and artifacts

**Stop at R0: NO-GO.** A graded proof is not attempted after failure of its
piecewise-constant prerequisite. R1 is not run, and there is no SBP R1 assessment
or replacement selected for R2. The existing defaults and guard are untouched.

The unexecuted R1 criteria remain exactly the user's criteria: zero failures
above 1+1e-12 in 1458 Cartesian and 288 fixed-k cases; flat 4x-period 100k-step
large-cell evolution; non-growing G2b TM eps=4 at fixed k_parallel=0.55*k0;
12-layer reflection over 0.3–3 GHz no worse at normal/60 degrees, at most 2x
current at 80 degrees, and strictly lower evanescent return. None of these
measurements is inferred from the small-block radii.

Artifacts:

- `scripts/dev/fdtd/cpml_passive/sbp_sat_r0.py`: exact construction, full work
  identities, positive bare-storage control, primary B0 replay, and Routh checks.
- `scripts/dev/fdtd/cpml_passive/results/sbp_sat_r0.json`: **27,866 bytes**, two
  exact small maps and compact certificates, with input source hashes. No large
  per-case files are produced.
- This derivation and decision record. New docs are ignored by the repository's
  existing ignore rules; the file is present on disk and is left uncommitted.

Reproduce from the worktree root in the maxwell environment, CPU only:

```bash
conda activate maxwell
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/dev/fdtd/cpml_passive/sbp_sat_r0.py
```

The command succeeds when the exact identities, positive bare-control pivots,
primary B0 replay, and outside-root certificates verify. Its success means the
**NO-GO evidence is verified**, not that R0 passed. No GPU was accessed and no
commit was made.
