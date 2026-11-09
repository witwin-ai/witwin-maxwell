# Cartesian Helmholtz multipole and local expansions

The GSP-09 primitive uses total-degree Cartesian Taylor expansions. Related
Taylor-expansion FMM literature includes
[Wang, Chen and Cai, 2019](https://arxiv.org/abs/1902.05875).
The formulas below specify this implementation's normalization and recurrence;
the cited layered-medium algorithm is not implemented here.

For fixed source center c_s, M_alpha = sum_j q_j (y_j-c_s)^alpha.
Local coefficients represent u(c_t+x) = sum_beta L_beta x^beta.
Both spaces retain total degree at most p; no factorial is included in M.
With d equal to child center minus parent center,

S[a,b] = binom(a,b) d^(a-b) for b <= a, and zero otherwise.

Then M_parent = S M_child and L_child = S^T L_parent. These translations
are exact polynomial identities within their respective finite spaces.

Write R=c_t-c_s, r=|R|, t=delta/r and kappa=k*r. The Green function
exp(i*k*|R+delta|)/(4*pi*|R+delta|) is expanded by introducing
s=1+2*(R/r).t+t.t. Its radial function F(s) satisfies

4*s*F''(s) + 6*F'(s) + kappa^2*F(s) = 0.

For F(s)=sum_n a_n*(s-1)^n:
a_0=exp(i*kappa)/(4*pi*r), a_1=a_0*(i*kappa-1)/2, and

a_(n+2) = -[(n+1)*(4*n+6)*a_(n+1)+kappa^2*a_n] / [4*(n+2)*(n+1)].

Sparse multiplication by 2*(R/r).t+t.t constructs multivariate coefficients
on CUDA. Rescaling each coefficient of degree m by r^(-m) yields g_alpha
for G(R+delta)=sum_alpha g_alpha*delta^alpha. Normalizing by r keeps the
radial recurrence dimensionless; it does not establish numerical stability
for arbitrary order, electrical size, loss or geometric scale.

The source-target binomial expansion gives

L_beta = sum_alpha (-1)^|alpha| binom(alpha+beta,alpha)
         g_(alpha+beta) M_alpha.

The i-th target-gradient kernel uses (alpha_i+beta_i+1)*g_(alpha+beta+e_i)
in the same formula. Consequently the Green table needs degree 2p and the
separately truncated gradient table needs degree 2p+1. This gradient table
approximates the derivative kernel; it is not the derivative of the already
truncated degree-p local potential polynomial.

All numerical arrays and evaluations use native CUDA PyTorch. Integer
multi-index/binomial layout is host control metadata. Complex charge gradients
follow exact adjoints of the finite expansion maps. Geometry/material gradients
are not part of the primitive contract.

These are fixed-order low-electrical-size primitives, without certified
truncation or floating-point error. M2L storage is dense in expansion indices
(O(p^6)); this is not yet a fast setup or high-frequency method. A table-byte
gate does not cover derivative-generation or host/control workspace. Source
supports, separation admission, tolerance/ka order selection, hierarchical
passes, RWG projection, exact near correction, periodic remote images and
physical/resource/complexity qualification remain required.

## Bivariate Cauchy truncation envelope

For real offsets x,y in full-support spheres of radii a,b, put
F(s,t)=G(R+s*x-t*y), d=|R|. On |s|,|t|<=rho choose
delta=rho*(a+b)=u*d with u<sqrt(2)-1. For z=s*x-t*y,
|z|<=delta and |(R+z).(R+z)-d^2|<=(2u+u^2)*d^2.
Writing alpha=sqrt(1-2u-u^2), the real part of the squared complex radius
is positive. Its principal square root r is analytic, Re(r)>=alpha*d,
|r|>=alpha*d and |r-d|<=B=d*u*(2+u)/(1+alpha). Consequently

M_G=exp(|Re(k)|*B-Im(k)*alpha*d)/(4*pi*alpha*d)

bounds the passive outgoing kernel. Each Cartesian gradient component is
bounded by M_grad=M_G*(|k|+1/(alpha*d))*(1+u)/alpha. The numerator bound
uses |R_j+z_j|<=(1+u)*d, and the derivative formula
grad_j G=G*(i*k-1/r)*(R_j+z_j)/r.

Bivariate Cauchy coefficient bounds are |c_mn|<=M*rho^(-m-n).
Coefficients of s^m and t^n aggregate precisely target degree m and source
degree n, so retaining m,n<=p matches the implemented M2L double total-degree
truncation. For zeta=1/rho<1, the discarded tail is bounded by

M*(2*zeta^(p+1)-zeta^(2p+2))/(1-zeta)^2.

A zero-radius side removes that series variable, giving the one-dimensional
tail M*zeta^(p+1)/(1-zeta). Both radii zero imply exactly zero truncation.
The gradient kernel uses its own Cauchy expansion and the same tail factor,
consistent with the separately truncated derivative M2L table.

The implementation minimizes each bound over u=.05,.1,.15,.2,.25,.3,.35,.4
with u>(a+b)/d. If no candidate admits rho>1, the bound is explicitly
unavailable, even if the looser geometric separation predicate passed. This
conservative sufficient domain is not a necessary expansion convergence test.
All positive factors and the tail use log arithmetic until final conversion;
positive bounds below normal float64 range are raised to the smallest normal
positive value. No directed rounding enclosure is claimed. This establishes
an ideal exact-arithmetic kernel envelope only, excluding coefficient
recurrence rounding, RWG quadrature/projection, near rules and solve error.
Independent mathematical review found no P1/P2 in this derivation.
