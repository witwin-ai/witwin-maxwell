# Candidate real symmetric derivative bound for scalar Helmholtz FMM

Status: independently reviewed ideal-kernel derivation. The opt-in native
implementation is under targeted validation; this proof supplies no floating-point,
quadrature, implemented-operator or production certificate. All gates are unchanged.

## Kernel and derivative norm

Let g(x)=exp(i k |x|)/(4 pi |x|), Im(k)>=0, and d be a nonzero real center
separation with D=|d|. T_n=derivative^n g(d) is a symmetric real-multilinear map
from (R^3)^n to C. For any real unit u, use s=t/D and mu=(d dot u)/D:

h(s)=sqrt(1+2 mu s+s^2), h(0)=1; F(s)=h(s)-1; z=i k D.

g(d+D s u)=exp(i k D)/(4 pi D) exp(z F(s))/h(s).

The roots of h(s)^2 have modulus1. For |s|<1, h(s)^2 cannot lie on the nonpositive real axis: writing s=x+i y,
its imaginary part is2y(mu+x). If y=0, 1+2mu x+x^2>0; if x=-mu,
its real part is1-mu^2-y^2>0 in the open disk. The analytic branch therefore
has positive real part and extends continuously to boundary roots. F maps the unit disk into itself: on s=exp(i theta),
h(s)^2=2 exp(i theta)(mu+cos(theta)). If mu+cos(theta)>=0,
|h-1|<=1 is equivalent to mu<=1; with negative sign it is equivalent to
-mu<=1. The limiting roots give h=0 and |h-1|=1. Maximum modulus and F(0)=0
therefore yield |F(s)|<=|s|. Consequently phi(s)=F(s)/s is analytic and
|phi|<=1 in the disk, including its limit at0.

1/h(s)=sum_l (-1)^l P_l(mu) s^l, and |P_l(mu)|<=1 for real |mu|<=1.
For Q_{n,j}=[s^n] F(s)^j/h(s), Q_{n,j}=0 if j>n. Q_{n,0} is a Legendre
coefficient and has modulus<=1. For j>=1, N=n-j>=0 and
Q_{n,j}=[s^N] phi(s)^j/h(s). The first N+1 coefficients of1/h can be
replaced by its degree-N polynomial L_N without changing this coefficient.
On any radius r<1, Parseval and |phi|<=1 imply

|Q_{n,j}|^2 r^(2N) <= mean |phi(r exp(i theta))^j L_N(r exp(i theta))|^2
  <= sum_(l=0)^N |P_l(mu)|^2 r^(2l) <= N+1.

Taking r to1 gives |Q_{n,j}|<=sqrt(n-j+1)<=sqrt(n)<=(n+1)/2 for n>=1.
The directional Taylor coefficient is thus bounded by

|T_n[u,...,u]|/n! <= (B0+(n+1) B1/2)/(4 pi D^(n+1)),
B0=exp(-Im(k) D), B1=exp((|k|-Im(k)) D)-exp(-Im(k) D)>=0.

For n=0 the actual value is B0/(4 pi D), so the displayed larger bound remains
valid. The finite polynomial sum in j is bounded by exp(|k|D)-1; the B0/B1
form avoids an unnecessary intermediate exp(|k|D) times small attenuation.

Banach's equality for symmetric multilinear forms on real Hilbert spaces
states that the operator norm equals the maximum diagonal norm. See equation2
of Carando and Rodriguez, [primary paper](https://arxiv.org/pdf/1810.09373).
For complex-valued T_n on a real domain, take the real part after a unit phase
chosen for any mixed-vector value. This is a real-valued symmetric form; its
mixed value is bounded by the supremum of |T_n[u,...,u]|. Thus the same bound
holds for the mixed real-vector operator norm, without assuming complex input
vectors or a complex Hilbert-space diagonal equality.

This use of the theorem and the Helmholtz coefficient derivation above are
our inference/derivation, not claims stated by that paper.

## Two separately truncated total-degree expansions

Let rt,rs>=0 be complete real support radii, a=rt/D,b=rs/D, q=a+b<1, and p>=0.
The degree-(n,m) grouped Cartesian Taylor term is
T_(n+m)[target^n,(-source)^m]/(n! m!). The derivative norm above bounds it by
binomial(n+m,n) a^n b^m [B0+(n+m+1)B1/2]/(4 pi D).

Since q<1, the positive generating majorants and their degree-weighted
variants converge absolutely. They justify grouping the double series and
termwise coordinate differentiation on the real support domain.
The implemented double-total-degree M2L omits n>p OR m>p. Bound the union by
summing the two nonnegative tails, allowing their intersection to be counted
twice. Define

Fs=(b/(1-a))^(p+1)/(1-q), Ft=(a/(1-b))^(p+1)/(1-q), F=Fs+Ft.
Us=(p+1)/(1-a)+1/(1-q), Ut=(p+1)/(1-b)+1/(1-q),
Vs=Us(Us+1)+(p+1)a/(1-a)^2+q/(1-q)^2,
Vt=Ut(Ut+1)+(p+1)b/(1-b)^2+q/(1-q)^2.
D1=Fs Us+Ft Ut; D2=Fs Vs+Ft Vt.

Here D1 sums (n+m+1) times the tail terms and D2 sums
(n+m+1)(n+m+2) times the tail terms. This follows by applying the Euler
derivative a d/da+b d/db plus1 and plus2 to the generating tails.

Candidate full point-kernel and each gradient-component absolute bounds are

Green <= (B0 F+B1 D1/2)/(4 pi D),
Gradient_component <= (B0 D1+B1 D2/2)/(4 pi D^2).

The gradient follows from T_(n+m+1) with one additional real unit coordinate
vector. These are ideal exact-arithmetic truncation estimates. They include
neither coefficient generation/integration, floating point nor solve error.
When rt=rs=0 the Taylor kernel is exact and both truncation bounds are0.
Other positive mathematical bounds must not become numeric0 by underflow.
q>=1, invalid geometry, wave numbers and finite representation failures must
fail closed. Valid independent existing Cauchy bounds may be combined by min;
no new bound may replace admission without proof and targeted verification.

## Required independent review and subsequent evidence

Attack the disk-map argument, Legendre/Parseval coefficient estimate, the
complex-valued real-domain Banach extension, the grouped Cartesian Taylor
interpretation and both tail/gradient generating identities. A numerical
search is not a proof. Then require CUDA scalar/vector kernel and gradient
comparisons with actual independent full Green evaluation on real support
points, low/high orders, separated/near geometric limits and passive complex
wave numbers, zero supports, precision, underflow/overflow and mutation rules.
Check unchanged original3840-edge budgets via saved geometry/receipt identity.
Any public selection must own a versioned numerical-policy identity and be
propagated consistently to planner/couplings/periodic applicable components;
unsupported combinations reject. No physical/complexity/production status
is inferred merely from tightening an ideal bound.


## Implemented policy and stable evaluation

`FMMConfig(kernel_bound_rule="real_support_min_v1")` owns the strict
`cartesian_fmm_real_support_config_v1` payload. Default Cauchy payloads remain
unchanged. Planner, optional ordered RWG aggregation and actual free-space
M2L couplings use the same policy; each owner rejects policy mutation. Public
periodic configuration rejects this rule rather than applying this proof to
a screened or summed kernel.

For B1 use log(B1)=delta D+log(1-exp(-|k|D)), delta=Re(k)^2/(|k|+Im(k)).
Scaled fixed wave constants and log(delta D) avoid cancellation, intermediate
squares and denominator overflow. `expm1` supplies the exponential remainder;
for log(|k|D)<-36 use its logarithmic small-argument limit within native
roundoff. All distance/radius/tail arithmetic remains on CUDA. A positive
finite mathematical tail has a native positive floor; unavailable values are
masked before the independent Cauchy/real minimum. This native evaluation is
not directed rounding or a floating-point enclosure. Neither an available
bound nor preallocation admission guarantees that high-order coefficient
tables can be represented in the selected precision.
