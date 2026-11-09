# FFT translation of the actual screened Ewald root polynomial

Internal scalar/EM/Ewald coefficient primitives are implemented; public periodic FFT
configuration/system integration and production qualification are still open.
The current free-space fft_v1 policy continues to reject all periodic entries.

Let c_gamma be ANY fixed scalar Taylor polynomial through total degree2p+1 at
a fixed expansion center, in the original multi_indices order. The original
compact scalar G and derivative matrix entries are exactly
G_(beta,alpha)=(-1)^|alpha| binom(alpha+beta,alpha)c_(alpha+beta),
Hj_(beta,alpha)=(-1)^|alpha| binom(alpha+beta,alpha)
               (alpha_j+beta_j+1)c_(alpha+beta+ej).
No radial free-space assumption is used in their factorial scaling identity.
For any strictly positive fixed finite real length s, the existing FFT fields
h_gamma=gamma!s^|gamma|c_gamma and
hj_gamma=(gamma+ej)!s^|gamma|c_(gamma+ej), totaldegree<=2p, together with source
(-1)^|alpha|/(alpha!s^|alpha|) and target1/(beta!s^|beta|), reproduce those matrices.
Cube L>2p, negative source embedding, and the same conjugate FFT adjoint apply.
Green coefficient c0 may be exactly zero (Bloch/symmetry cancellation), while
higher coefficients/derivatives remain nonzero; unlike a single radial
Helmholtz kernel, c0>0 is NOT a legitimate general polynomial admission gate.
Still reject nonfinite values, weights/factors and nonzero coefficient components
scaled to zero. No coefficient gradients; copied immutable fixed coefficients,
current-vector derivative only. Zero polynomial is valid. No numerical certificate.

For EwaldRootM2L, the ACTUAL existing coefficient vector equals
C=Taylor(G_configured_periodic - G_primary) aboutR=0
  -sum_declared_near_images_nonprimary exp(+i beta.A.n)*Taylor(G_free(R-A.n)).
The primary image was already removed by ewald_regular_taylor. Declared near
images remain owned exactly once by the caller. Reuse THIS actual coefficient
vector, kernel/lattice/medium/near index receipts, cuts/pole/preflight guards,
and derivative channels; do not substitute Helmholtz Taylor at fictitiousR or
sum unscreened translations in place of Ewald regular coefficients.
A coincident root has physical displacement0. Explicit normalization length
s=sigma_min(lattice vectors)/(2p+2)>0 is a conditioning policy, not a geometric
center displacement or analytic/cutoff/error admission. Independent real scalar
scale input is guarded/version-bound. Nativefloat64/complex128/p0..12 unchanged.

The internal owned arbitrary-coefficient scalar FFT factory shares existing
layout/field/action/adjoint math and composes with the original PMCHWT EM wrapper,
without duplicated curl/charge algebra.
The internal Ewald FFT wrapper copies already-guarded actual EwaldRootM2L
coefficients; after construction it discards compact index/binomial duplicates,
retaining kernel/lattice/medium/near bindings and owned FFT layout/coefficient
copies. No silent cache/fallback/new publicpolicy yet. Retained inventory must
count actual coefficients once, scale, zero physical center metadata, index/
weight/scatter tensors and near indices. Shared kernel/lattice excluded explicitly.
Temporary old compact construction is additional setup workspace and must be
reported; no unchanged-peak/construction speedup claim.

Meaningful controls: independent arbitrary polynomial dense binomial loops
including c0=0/zero polynomial, complex parity and true Hermitian/currentgrad;
actual1D/2D/3D Ewald roots, skew lattices, nonzeroBloch, primary-only and additional
near-image exclusions; compare to original compact action and actual regular
kernel/fullgradient sampled at physicalsource/targetoffsets. Physically scaled
moments/duals avoid pretending raw unweighted norms certify floating algebra.
All actions/adj/dot/currentgrad original1e-11 and actualkernel tolerances preserved.
Storageexact/independentownership/mutation/pole/normalization/prealloc guards.
After passed primitives, a separate explicit periodicFFT configuration version
and regional/system/capacity/public/Bench integration phase is required. Old
fft_v1 configuration cannot silently acquire different periodic identity.
CompleteGoal including generaltensor/directional/scaling/physics remains intact.
