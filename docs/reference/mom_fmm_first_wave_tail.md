# Candidate exact-first-wave coefficient majorant

Status: implemented ideal-arithmetic majorant; targeted native controls pass; original growing-ka acceptance fails the unchanged gradient gate. Follows actual positive_tail_min_v1
growing-ka gradient FAIL1.2735512327556758e-5>1e-5. No threshold change.
The previous finite-wave positive generating proof and real-phase Banach norm
remain in force; no screened Ewald, floating-point or production certificate.

In that proof F=h-1, Q_(N,j)=[s^N]F^j/h. For j=1,
F/h=1-1/h, hence Q_(N,1)=-[s^N](1/h) for N>=1 and has modulus<=1
by the reviewed Legendre coefficient bound. Q_(0,1)=0.
This replaces the earlier bound (N+1)/2 on only the first-wave term.
For j>=2 retain |Q_(N,j)|<=(N-j+2)/2. Therefore

B_N*=1+K 1_(N>=1)+(1/2) sum_(j=2)^N (N-j+2)K^j/j!, K=|k|D.
Phi*(t)=(1+Kt)/(1-t)+C(t)(exp(Kt)-1-Kt),
C(t)=(2-t)/(2(1-t)^2), C'(t)=(3-t)/(2(1-t)^3).
Phi*'(t)=(1+K)/(1-t)^2
       +C(t)K(exp(Kt)-1)+C'(t)(exp(Kt)-1-Kt).

All coefficients are nonnegative and bound the actual real mixed derivative
norm. The previous whole-positive-tail argument is unchanged: source-tail
<=Phi*(a+r)(b/r)^(p+1) at b<r<1-a, and gradient uses Phi*'. Independently
minimize each side/component's valid contours, then sum union tails.

The native implementation evaluates exp(-kappa D)(exp(x)-1-x), x=Kt,
stably, without exponential subtraction or loss of a small positive second
remainder. For small x use its positive series starting x^2/2; for moderate
x use exp(x) times1-(1+x)exp(-x), with expm1/log1p, combined with attenuation
before overflow. A finite native series is not a float enclosure and cannot
substitute for the exact mathematical expression in the proof. Invalid
contours/representations reject, zero sides remain exact0, and any selected
new bound owns a distinct strict policy identity. Actual M2L/fullkernel,
old/default/rule regression and unchanged original-size admission still
required. No exact-first-wave or growing-ka admission claim is made yet.

## Native representation and strict identity

The explicit free-space `first_wave_min_v1` policy owns canonical
`cartesian_fmm_first_wave_config_v1`. It takes availability-safe minima with
the existing bivariate, real-support and finite-wave positive bounds; all old
policies and their configuration payloads remain unchanged. A zero support
side has an exact zero omitted-degree tail. Nonfinite or cancellation-ambiguous
contours are rejected. The rule does not cover the screened Ewald kernel.

For x<=.1, evaluate E2(x)=exp(x)-1-x as x^2 R2(x)/2. The retained
R2 polynomial through degree8 is sum(l=0..8) 2*x^l/(l+2)!. Its omitted
positive tail is bounded above by 2*x^9/11!/(1-x/12), since all subsequent
term ratios are at most x/12. Thus the series is not truncated downward.
Evaluate log(E2/exp(x)) as 2log(x)-log(2)-x+log(R2_upper), retaining
log(x) even when x or x^2 underflows. For .1<x<=50 use
log(-expm1(log1p(x)-x)); for x>50 use the conservative factor1.
Attenuation and growth are combined with signed logarithmic scaling before
exponentiation. These analytic expressions do not enclose native roundoff.

The derivative uses (1+K)/(1-t)^2 + C(t)K(exp(Kt)-1)
+ C'(t)(exp(Kt)-1-Kt), with the same positive second remainder upper bound.
Source and target contours independently minimize each component, then their
missing-degree union bounds are summed. The whole-positive-tail argument
uses Phi*(a+r)(b/r)^(p+1), without a geometric denominator.

Fixed-geometry current-vector gradients remain the supported differentiable
path. No material-parameter gradient, directional high-frequency complexity,
full original operator-error, peak-resource or physical qualification follows
from these ideal kernel tails.
