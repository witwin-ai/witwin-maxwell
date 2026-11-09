# Centered directional Helmholtz coefficient FFT

For fixed R≠0, r=|R|, n=R/r, a fixed real unit carrier u and δ=x−y, define
G(R+δ)=exp(ik*u·δ)h_u(δ). There is no extra exp(-ik*u·R) constant.
With t=δ/r, s=n·t and q=t·t−s²,

h_n/G(R) = (1+s)^(-1) F(q/(1+s)², ikr(1+s)),
F(v,z) = (1+v)^(-1/2) exp(z*(sqrt(1+v)−1)).

F=Σ_j A_j(z)v^j, A_0=1 and
2(j+1)A_{j+1}=(z−1−2j)A_j+z Σ_{m=1}^j binom(1/2,m)A_{j−m}.
The exact rational constants a_{jl} of A_j(z)=Σ_l a_{jl}z^l are configuration
constants. Through degree D=2p+1, native CUDA evaluates
Σ_j q^j Σ_b s^b Σ_l a_{jl}(ikr)^l (-1)^b binom(2j−l+b,b), 2j+b≤D.
Then h_u=exp(ik(n−u)·δ)h_n is three separable coefficient convolutions.
No host numerical kernel fallback is introduced. Small quadratic coefficients
are not clamped; only bit-exact residual zero skips its exponential convolution.
Physical-degree powers and complex amplitudes reject nonrepresentable scaling.

CartesianDirectionalFFTTranslation accepts moments already containing the
source phase exp(-ik*u·y); it returns four physical-gradient local channels
before the receiver phase exp(ik*u·x). The caller applies each phase exactly
once. ∂G=exp(ik*u·δ)(∂h_u+ik*u*h_u), so both the Fourier derivative and
carrier gradient are required. Parent FFT Hermitian transpose conjugates the
complete physical-gradient grid; callers must conjugate their complex phases.
Native backward retains fixed-current actions rather than generated grids.

Geometry, carrier and explicit normalization are fixed CUDA float64; coefficients
and currents are complex128. Passive finite scalar k, p0..12, positive budgets
and unit carrier are mandatory. Inputs are cloned, all nine retained arrays are
independently owned and guarded against mutation. Batch setup uses≤32centers.
Retained scalar FFT bytes plus24carrier bytes are counted. Visible setup allowance
includes coefficient workspace and gradient mixing, excluding library/allocator,
caller state and system/tree resources.

This transformed finite truncation cannot inherit the old unmodulated radial
error certificate. Finite coefficients are not an accuracy certificate. Selected
independent exact G/H and same-order unmodulated controls establish only the
assessment's sample scope. Stable source/receiver projections, electromagnetic
RWG integration, cone selection/tree transition operators, native kernel admission,
public identity, periodic composition and whole-system physics/resources/scaling
are still required before directional FMM admission or production qualification.
