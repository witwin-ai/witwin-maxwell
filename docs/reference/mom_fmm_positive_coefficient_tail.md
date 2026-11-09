# Candidate positive-coefficient one-sided tails for Helmholtz FMM

Status: independently audited mathematical derivation, following the actual
real_support_min_v1 growing-ka3840-edge FAIL. Native controls are recorded
separately; no floating-point, physical or production certificate is supplied.
Uses the already independently reviewed real-domain symmetric derivative
proof in docs/reference/mom_fmm_real_support_bound.md. All original gates stay.

Let D>0, K=|k|D, Im(k)=kappa>=0, a=rt/D,b=rs/D, a+b<1.
The reviewed directional coefficient estimate gives Q_N0<=1 and for j>=1,
|Q_Nj|<=sqrt(N-j+1)<=(N-j+2)/2 (j<=N). Real-phase Banach then bounds
||derivative^N g||/N! <= exp(-kappa D) B_N/(4 pi D^(N+1)),
where B_N=1+(1/2) sum_(j=1)^N (N-j+2) K^j/j!, B_N>=0.

Keep the finite sum j<=N rather than majorizing it by exp(K). Its exact
positive generating majorant for |t|<1 is

Phi(t)=sum_N B_N t^N
      =1/(1-t)+(exp(Kt)-1)(2-t)/(2(1-t)^2).
Phi'(t)=1/(1-t)^2
      +K exp(Kt)(2-t)/(2(1-t)^2)
      +(exp(Kt)-1)(3-t)/(2(1-t)^3).

After grouped mixed target/source degrees, the nonnegative majorant is
Phi(a+b): coefficient of a^n b^m is binomial(n+m,n)B_(n+m).
For the gradient use Phi'(a+b), since coefficients weight
(n+m+1)B_(n+m+1). This matches the actual separately truncated M2L.
The prefactors are exp(-kappa D)/(4 pi D) and
exp(-kappa D)/(4 pi D^2), respectively.

For the source-tail m>p, hold the ACTUAL a fixed, and choose b<r<1-a.
Write Phi(a+z)=sum_m C_m(a) z^m with nonnegative C_m(a).
Coefficient positivity bounds the entire tail directly: with rho=b/r<1,

sum_(m>p) C_m(a) b^m =sum_(m>p) C_m(a)r^m rho^m
 <=rho^(p+1) sum_(m>p) C_m(a)r^m <=Phi(a+r)(b/r)^(p+1).

No extra geometric-series denominator is needed. Independent mathematical
review supplied this strict improvement and verified its positive-sum proof.

Use the same inequality with Phi' for the gradient. The target-tail uses
Phi(s+b), a<s<1-b and ratio a/s. The union of the two missing-degree tails
is bounded by their sum, with nonnegative intersection counted twice.
Minimize each of the four independent source/target Green/gradient contour
bounds over valid r/s, then sum the source/target minima. Each is valid on the
whole other-side support; independently choosing contours is legal.
A side with zero support has no missing-degree tail. Both zero is exact0.
All positive coefficient sums and derivatives converge absolutely because
a+b<1; evaluated a+r and s+b must stay strictly below1.

This retains the finite wave-degree information and enlarges only the omitted
side, rather than scaling both sides with one contour. It may help the actual
growing-ka failure without changing geometry, p6..12, material, kernel budgets
or retained budgets. It does NOT supply a directional FMM or high-frequency
complexity qualification. Numerical testing is not a proof.

Required independent proof review: finite coefficient generating identity,
complex real-domain mixed-derivative scope, positivity/series rearrangement,
source/target union tails and the Phi' gradient shift. Subsequent native
implementation would require stable scaled/log expm1 handling of
exp(-kappa D)Phi and exp(-kappa D)Phi', availability-safe minima, explicit
new policy identity, actual M2L/full-kernel comparisons and one meaningful
changed-policy original-size admission run. No Ewald screened-kernel claim.

Independent mathematical review completed with no P1/P2, including the
stronger whole-positive-tail inequality above. Native implementation, actual
M2L comparisons and unchanged-gate growing-ka acceptance remain required.


## Native policy

`positive_tail_min_v1` owns `cartesian_fmm_positive_tail_config_v1`. It takes
the availability-safe minimum of the reviewed positive-tail estimate and the
previous valid Cauchy/real-support estimates. Each actual scalar free-space
planner/coupling owner binds the same rule; periodic/Ewald remain rejected.
The seventeen adaptive contours enlarge only the omitted side, strictly
inside the open interval; rounded endpoints are excluded. Signed logarithmic
growth/attenuation arithmetic avoids intermediate overflow; unresolved
near-cancellation contours reject conservatively. Native evaluation remains
distinct from a directed floating-point enclosure or implemented-operator
certificate. Actual-kernel/free-action controls, public persistence and consolidated
affected regression pass. Original growing-ka Greenpasses but its gradient
1.2735512327556758e-5 still exceeds1e-5; full resource/physics remain open.
