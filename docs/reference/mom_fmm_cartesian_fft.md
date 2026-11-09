# Exact Cartesian M2L as a circular convolution

Implemented internal scalar primitive for genuine multipole/local algebra on
native CUDA FFTs. It is not a public EM branch, native error certificate,
high-frequency directional branch or periodic/system resource qualification.

Let A_p={alpha in N^3:|alpha|<=p}, g_gamma be the unchanged Taylor coefficients
of the Helmholtz kernel about fixed R. The existing dense Cartesian translation
for scalar moments and separately expanded gradient is
 L_beta = sum_alpha (-1)^|alpha| binom(alpha+beta,alpha) g_(alpha+beta) M_alpha,
 D_j L_beta = sum_alpha (-1)^|alpha| binom(alpha+beta,alpha)
                  (alpha_j+beta_j+1) g_(alpha+beta+e_j) M_alpha.
Existing coefficients through totaldegree2p+1 suffice. No altered truncation.

Choose any real positive fixed scale s, provisionally s=|R|/(2p+2). Define
 w_alpha=(-1)^|alpha|/(alpha! s^|alpha|),
 v_beta=1/(beta! s^|beta|),
 h_gamma=gamma! s^|gamma| g_gamma,
 h^(j)_gamma=(gamma+e_j)! s^|gamma| g_(gamma+e_j), |gamma|<=2p.
Outside that total-degree set all h are zero. Then
 L_beta=v_beta sum_alpha h_(alpha+beta) w_alpha M_alpha,
 D_j L_beta=v_beta sum_alpha h^(j)_(alpha+beta) w_alpha M_alpha.
The algebra follows by cancelling factorials and scale powers. Scale is a
conditioning choice, not a mathematical accuracy or floating point enclosure.

Let integer cube side L>2p (use next power of two >=2p+1, L=1 when p=0).
Embed Q[(-alpha) mod L]=w_alpha M_alpha and H[gamma]=h_gamma (or h^j).
Circular convolution C=IFFT(FFT(H)*FFT(Q)) satisfies at beta in A_p
 C[beta]=sum_alpha H[(beta+alpha) mod L] w_alpha M_alpha.
Since each beta_i+alpha_i<=2p<L, none of these retained output entries wraps;
the circular kernel equals the required linear correlation on all A_p.
No dense P-by-P table is needed. Nonretained cube entries are zeroed explicitly.
Cube h gradients also use only coefficients of totaldegree<=2p+1.

Exact adjoint of the same fixed discrete operator:
 Embed Y[beta]=v_beta value_beta (for each of four fields).
 Z=IFFT(FFT(Y)*conj(FFT(H))).
 Return w_alpha sum_fields Z[(-alpha) mod L].
Both w/v are real, including source parity. This is the Hermitian adjoint of
embedding, unit-normalized circular convolution and extraction. The usual
unnormalized-forward/normalized-inverse FFT convention is used consistently.
The four-field adjoint sums contributions and handles arbitrary leading RHS
axes. A future EM wrapper must derive/verify the original J/M charge/vector/curl
signs independently; this scalar primitive alone does not qualify that wrapper.

Native design:
- Internal standalone scalar translation primitive only first, fixed real CUDA
  float64 coordinates/complex128 moments and integer order (bounded initial
  supported native workspace). CPU config/oracles allowed, no CPU numerics.
- Original Taylor coefficients, factorial/scale in native float64 logarithms,
  reject nonfinite normalization/grids/output; no implicit numerical fallback.
- Bound grid workspace with declared budget and chunked RHS. Do not persist full
  four-field FFT grids per coupling: original6738pairs p12 would exceed table
  budget with retained4x32^3complex grids. Temporary per-action grids only in
  future compact integration; original compact retained inventory unchanged.
- No claims that normalization makes float roundoff bounded. FFT may lose small
  coefficients despite exact arithmetic equivalence. Before public integration,
  test ideal total-degree/cyclic alias identities independently, actual dense
  M2L Green/gradient action and adjoint/current gradients, SI-scaled physical
  source/test monomials and weak all-column operators with unchanged thresholds.
- Specifically test p0/3/6/11/12, passive complex waves, varying supports and
  saved actual original worst geometry. Record real failures and unsupported
  representations; do not weaken gates to obtain passing FFT results.
- FFT operation count per RHS/coupling O(p^3 log p) and grid workspace O(p^3)
  at fixed chunk size; actual system N scaling, setup/solve/crossover, resources,
  floating error and physical qualification require separate original cases.
  This does not satisfy the Goal's directional/high-frequency requirement.

The admitted native primitive rejects nonfinite/underflowed positive normalization
and nonzero coefficient components scaled to zero. Its temporary tensor budget
is an explicit conservative structural allowance, excluding backend FFT library
workspace, allocator cache and caller tensors. Actual single-process CUDA peaks
for a33RHS p12 scalar case are recorded separately, not a complete FMM peak gate.
