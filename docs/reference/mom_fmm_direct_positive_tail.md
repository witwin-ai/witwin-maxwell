# Direct positive rational-exponential Helmholtz tails

Implemented ideal-arithmetic majorant after first_wave_min_v1 original growing-ka terminal FAIL.
The exact first-wave change leaves the worst G/H maxima unchanged. Do not
repeat small wave-coefficient refinements without resolving the actual tail.
No thresholds or geometry/order ranges change; no numerical qualification.

Use the already reviewed coefficientwise majorant Phi(t)=R1+C(exp(Kt)-1),
R_m(t)=(1-t)^(-m), C=(R1+R2)/2, K=|k|D>=0. Since
R1-C=(R1-R2)/2 has nonpositive coefficients, Phi is coefficientwise bounded
by exp(Kt)*(R1+R2)/2. Likewise Phi'=K C exp(Kt)+C'(exp(Kt)-1)+R2,
C'=R2/2+R3. R2-C'=R2/2-R3 has nonpositive coefficients. Therefore
Phi' is coefficientwise bounded by
exp(Kt)*(K R1/2+(K+1)R2/2+R3).
All direct majorant coefficients are positive. Real Banach derivative norms,
Legendre/Parseval proof and distance factors remain those already reviewed.
This is an upper majorant; native arithmetic is not a floating-point enclosure.

For held support a and omitted support b, a+b<1, let L=1-a, q=b/L,
P=p+1, E_P(x)=sum(l=P..infinity) x^l/l!. The exact omitted-degree tail
T_m=[b-degree>=P] exp(Kb)/(L-b)^m, for m=1,2,3, is the positive expression

T_m = q^P L^(-m) sum(l=0..p) (K L)^l/l! A_m(P-l,q)
      + L^(-m)(1-q)^(-m) E_P(Kb),

A_1(h,q)=1/(1-q),
A_2(h,q)=(h+1)/(1-q)+q/(1-q)^2,
A_3(h,q)=(h+1)(h+2)/(2(1-q))+(h+2)q/(1-q)^2+q^2/(1-q)^3.

Derivation: expand exp(Kb) and the rational denominator in nonnegative
powers. Split wave degree l<=p (geometric rational tail starts at n=P)
and l>=P (whole rational series starts at n=l). The binomial coefficient
of rational degree n-l gives the three shifted geometric sums above.

The source-side Green tail is bounded by
exp(-Im(k)D+K a)*(T1+T2)/(8*pi*D).
The source-side gradient-component tail is bounded by
exp(-Im(k)D+K a)*(K*T1/2+(K+1)*T2/2+T3)/(4*pi*D^2).
Swap a,b for target-side omitted degree; sum the union, including its
possibly double-counted intersection. Zero omitted support is exact zero.
This computes the positive tail directly, without an enlarged contour or
charging retained-degree terms to the omitted-degree bound.

Native proposal: all polynomial/geometric sums in CUDA float64 logs, limited
prefix workspace p<=64 (otherwise direct bound unavailable; old policies
remain usable). For E_P(x) retain l=P..P+96 in log-sum-exp and add
 t_(P+96)*x/(P+97)/(1-x/(P+98)), for x<=32, an analytic positive tail upper
bound since subsequent ratios are <=x/(P+98)<1. x above the cap makes this
bound unavailable, not zero. Log x retains underflow. No exp(x)-polynomial
subtraction or downward finite truncation. Signed-scaled attenuation/growth
combine before overflow; ambiguous cancellation rejects except exact zero
held support with lossless wave. All old available bounds may enter minima;
new public direct_positive_tail_min_v1 owns strict canonical identity,
free-space only. Actual M2L/fullkernel, affected old-rule/default regression,
public exact persistence/Bench archive and original growing-ka strict
readmission are required. Save worst-pair witnesses during that admission
without another planner pass, to diagnose any remaining failure.

Implementation status: targeted native controls and public workflow pass; no production qualification.

Original growing-ka p12 remains gradientFAIL; direct majorant alone did not improve the saved worst support pair. Targeted controls do not confer full-system production qualification.
