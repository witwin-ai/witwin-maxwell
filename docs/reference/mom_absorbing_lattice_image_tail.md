# Absorbing lattice image tail over support spheres

Let A contain D independent real lattice rows, beta be real, Im(k)=h>0,
R be the target-source center displacement, and a,b their complete-support
radii. An image is weighted by exp(+i beta.A*n) and evaluated at R-A*n.
The phase has modulus one. For any support offsets x,y, |x|<=a, |y|<=b,

    |R+x-y-A*n| >= sigma*|n|_2 - B >= sigma*|n|_infinity - B,
    sigma = minimum singular value of A, B = |R|+a+b.

Truncate the integer cube at |n|_infinity<=S. Write M=S+1 and
r0=sigma*M-B. The bound is available only if r0>0. For shell m>=M,
the exact integer counts are C1(m)=2, C2(m)=8m, C3(m)=24m^2+2.
Every omitted Green value is bounded by

    exp(-h*r0)/(4*pi*r0) * q^(m-M), q=exp(-h*sigma).

Thus the complete absolute tail is bounded by that prefactor times
sum_{m=M}^infinity CD(m)*q^(m-M). Put d=1-q. The three needed sums are

    T0 = 1/d,
    T1 = M/d + q/d^2,
    T2 = M^2/d + 2*M*q/d^2 + q*(1+q)/d^3.

The dimension-specific factors are 2*T0, 8*T1 and 24*T2+2*T0.
For the free-space radial Helmholtz kernel, each Cartesian gradient component
is bounded by (|k|+1/r0) times the scalar bound. This bound is uniform over
both complete support spheres and independent of real Bloch phase.

Implementation uses CUDA float64 singular values and geometry, expm1 for d,
and logarithmic sums. Positive bounds below the smallest normal float64 value
are raised to that positive floor, never rounded into a zero certificate.
Unavailable geometry or overflowing bounds reject. This is not directed
rounding: numerical geometry and floating-point evaluation are not certified.

The tail bound excludes retained-image M2L truncation, near quadrature,
RWG assembly, solver and physical errors. Admitting this budget alone does
not qualify an infinite-periodic operator. In particular, h=0 is outside this
bound and requires a separate lossless periodic construction.

Independent read-only mathematical review found no P1/P2. GPU controls compare
omitted finite increments at support-domain points in skew 1D/2D/3D lattices,
check decreasing bounds with shell count, unavailable support separation,
strong-decay positive floors and rejection before coefficient construction.
Finite increments are numerical controls; the infinite-tail conclusion comes
from the above complete shell sum, not from a finite numerical comparison.

## Combining retained-image multipole truncation

For every retained remote image n, the Cartesian bivariate Cauchy bound
controls its Green and gradient truncations over the same complete support
spheres. Since the real Bloch phase has modulus one, triangle inequality
bounds the finite remote sum by the sum of these per-image bounds. Add the
infinite omitted-image bound above to obtain a complete ideal kernel bound
for remote multipoles plus exact retained-near kernels. Near quadrature is
not included. An unavailable per-image Cauchy contour rejects the combined
plan; no partial remote sum may be presented as complete. Planning and
construction share the same streamed image partition, and construction checks
near/far image counts against the plan.
