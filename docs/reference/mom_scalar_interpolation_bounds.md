# Scalar kernel interpolation truncation bound

This bound applies only to exact Chebyshev-Lobatto interpolation of the exact
free-space scalar kernel and its gradient, for fixed separated real boxes and a
passive outgoing wave number. It does not cover native floating-point rounding,
quadrature, moment accumulation/transfers, or a relative system operator error.

For one coordinate of half-width h, choose a Bernstein ellipse rho>1 while all
other coordinates remain real. Every ellipse point is within complex distance
h(rho-1) of some point in its real interval (bound real and imaginary excursions
separately). Let d>0 be the real box distance and set delta=s*d, rho=1+delta/h,
where 0<s<sqrt(2)-1. A complex displacement z=v+e has |v|=r>=d and ||e||<=delta.
Thus Re(z dot z)>=r^2(1-2s-s^2)>0, so its principal square root R is analytic
throughout this ellipse. Write alpha=sqrt(1-2s-s^2). Then

- Re R >= alpha*r, and |R| >= alpha*r;
- |R-r| <= delta*(2+s)/(1+alpha), from (R-r)(R+r)=2 v dot e+e dot e;
- |z_j/R| <= (1+s)/alpha for each component.

With k_i>=0, define B=delta*(2+s)/(1+alpha). Uniform ellipse bounds are

M_G = exp(|Re k| B - Im k * alpha*d)/(4*pi*alpha*d),
M_grad = M_G * (|k|+1/(alpha*d)) * (1+s)/alpha.

For p nodes (degree n=p-1), the univariate interpolation error is at most
4 M rho^(-n)/(rho-1). The degree-zero midpoint case follows directly by bounding
the Chebyshev series tail twice; it obeys the same conservative expression.
A collapsed axis has no interpolation error. The six-variable tensor error
f-I1...I6 f is the sum of I1...I(j-1)(f-Ij f). Bound each preceding interpolation
operator by Lambda_j <= 1+(2/pi)log(p_j), using the slightly looser node-count
form of the standard Chebyshev-Lobatto Lebesgue bound. This proves the implemented
sum of univariate errors times preceding Lebesgue factors; no joint six-dimensional
complex ellipse is assumed. A finite list of valid s values yields several
valid bounds; their minimum is also valid, independently for G and grad components.

The univariate interpolation estimate is stated by Trefethen in
[Convergence bounds for entire functions](https://www.chebfun.org/examples/approx/EntireBound.html).
The Lebesgue bound is given in Battles and Trefethen,
[The Chebfun System](https://www.chebfun.org/publications/chebfun_paper.pdf), section 2.
The Helmholtz ellipse estimates and tensor application above are derived here.

Implementation uses CUDA float64 geometry arithmetic and finite candidate checks,
not directed interval arithmetic. Underflowed positive results are floored to
float64 tiny. Consequently the receipt is an analytic truncation estimate
implemented numerically, not a certified floating-point bound. It must not be
reported as a complete RWG operator, physical error or production certificate.

Exponentials, inverse distances, wave magnitudes, interpolation decay and tensor
Lebesgue amplification are combined in log space before the final exponential.
This is required even for a bound: premature exp underflow can otherwise be
followed by large geometric factors that would have restored a representable
result. A degree-zero passive-wave counterexample at distance 1e-100 is retained
in the focused tests and the phase plan.

Prospective orders can be evaluated from raw CUDA bounds before interpolation
nodes are allocated. Uniform-order planning tests every regional far block at
every candidate order; no monotonicity of the computed bound is presumed. A
bound outside finite native representation marks that candidate unavailable,
rather than proving all higher orders unavailable. Retained table counts are
monotone in uniform order, so that separate capacity failure may stop the search.
