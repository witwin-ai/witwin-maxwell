# Kernel-to-RWG interpolation error

This component bounds only the ideal exactly integrated interpolant, not the
implemented floating-point moments, quadrature, transfers or solved field error.
All balancing uses B=diag(eta on J,1 on M) A diag(1 on J,eta on M), eta>0.

For a triangle, f=s(r-v0), div f=2s. With a=v1-v0 and b=v2-v0,
the mean of |r-v0|^2 is (|a|^2+|b|^2+a dot b)/6. Cauchy-Schwarz gives
integral |f| <= |s| area sqrt(mean square). Add the two support faces to
obtain u_i. Also q_i=integral |div f|=sum_faces 2|s|area exactly.
Log-space face-to-edge reductions avoid squaring tiny/large moments.

For uniform scalar kernel error g and per-component gradient error h,

|delta D_ij| <= g(u_i u_j + q_i q_j/|k|^2),
|delta K_ij| <= sqrt(3) h u_i u_j.

Let U=(sum_test u_i^2)(sum_source u_j^2),
Q=(sum_test q_i^2)(sum_source q_j^2),
C=(sum_test u_i q_i)(sum_source u_j q_j). Then

E_D^2 = g^2 (U+2C/|k|^2+Q/|k|^4),
E_K^2 = 3h^2 U.

For the regional block [[K,eD],[mD,K]], e=i omega epsilon, m=-i omega mu,
the balanced Frobenius error is bounded by the square root of
2 eta^2 E_K^2 + (eta^4 |e|^2+|m|^2) E_D^2.
Each node needs only three log sums (u^2,q^2,u*q), aggregated through the tree.
No expanded basis or far edge-pair matrix is required.

Multiple media on one block overlap and their bounds add by the triangle
inequality. Distinct terminal blocks partition coefficient pairs, so their
squared bounds add. In a global system the body-interior blocks are disjoint;
their joint bound is the root sum of squares. That bound adds to the exterior
bound because the exterior overlaps every interior diagonal block.

Every reduction is evaluated in CUDA float64/log arithmetic. This is not directed
interval arithmetic. The kernel bound already excludes floating-point error;
the RWG conversion additionally assumes exact support integration and exact
nested polynomial transfer. A relative implemented-operator certificate must
also bound those components and establish a valid reference-norm lower bound.

## Exact polynomial moment quadrature

Let S=sum_axes(node_count-1) for one interpolation box. The tensor cardinal
polynomial restricted to a triangle has Cartesian total degree at most S.
Multiplication by a linear RWG gives degree S+1. Under triangle_rule's map
r=v0+u((1-v)(v1-v0)+v(v2-v0)), each monomial has u/v degree at most S+1;
the Duffy Jacobian u raises the first degree to S+2. Tensor Gauss order q is
therefore sufficient when 2q-1 >= S+2, or q=ceil((S+3)/2)=(S+4)//2.
Divergence moments have lower degree and are covered by the same rule.
Collapsed axes contribute zero to S. This is a sufficient, not minimal,
triangle-specific rule; no geometric cancellation is assumed.

RWGInterpolationMoments, mixed-body leaves, NestedRWGMoments and DirectSystemActions
now default to this automatic choice (quadrature_order=None). Explicit low orders
are allowed for experimental controls but receive polynomial_exact_in_exact_arithmetic=false.
The ideal interpolation error report includes the actual moment quadrature receipt.
When the degree condition holds, ideal quadrature truncation of these polynomial
moments vanishes. Native Gauss nodes/weights, barycentric evaluation, accumulation
and transfer still incur floating-point error; exact near-field integration error
and the remaining operator/physical qualification are separate.

## Preallocation use

Geometry-only per-edge logs and three node statistics can be formed before any
interpolation nodes or moments. The uniform-order planner accepts these statistics
for region zero (common exterior) and the following disjoint body interiors,
recomputes every far-block bound for every candidate, and requires a declared
absolute RWG budget in addition to the two kernel bounds and retained table budget.
The direct system builds the statistics from its validated actual provider.
Explicit moment quadrature below the selected polynomial degree is rejected for
this admission path; automatic quadrature meets the sufficient degree condition.
No reference matrix norm is used, so this is not a relative error guarantee.
