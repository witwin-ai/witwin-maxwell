# Directional RWG maps and physical electromagnetic coupling

For each complete edge support, S[e,a,c]=integral f_e(y)y^a exp(-ik*u.y)
and R[e,b,c]=integral f_e(x)x^b exp(+ik*u.x), with offsets about separate source
and receiver centers. c0..2 are vector RWG components; c3 is div(f_e). Both support
faces, signs and declared selected-edge order are retained. At passive complex k,
R is not S.conj(): store and integrate both signs independently.

Source current contraction produces M[a,c]=sum_e S[e,a,c]I[e]. Receiver evaluation
is V[e]=sum_{b,c}R[e,b,c]L[b,c]. Their adjoints respectively conjugate S and R.
The complete physical weak action is R A S and its Hermitian transpose is
S^H A^H R^H. Source/receiver derivatives retain native fixed-current autograd.
The scalar directional kernel restores physical gradients grad(h)+ik*u*h; do not
add carrier derivatives to physical RWG divergence or duplicate that curl term.
FFTElectromagneticM2L supplies original PMCHWT material, charge and curl channels.

The old polynomial quadrature enters a shared weighted helper with phase=None,
retaining the original real accumulation. New signed carrier integrations are
CUDAcomplex128. Geometry, direction and wave number are fixed, quadrature is
explicit and at least sufficient for the polynomial factor. Nonzero-k carrier
integration cannot inherit polynomial exactness; native finite output alone does
not certify quadrature convergence.

DirectionalRWGProjection owns selected edges, center, direction and four moment
arrays. Default-created PreparedRWGGeometry additionally contributes all four
geometry arrays to exact retained/preflight inventory; explicitly caller-provided
geometry is excluded from owned bytes and validated on every map. Receipts bind
these distinct ownership scopes. Conservative visible setup allowances exclude
allocator/library/caller/system resources and do not establish actual full peaks.

The implementation is an internal scalar-medium electromagnetic coupling. It has
no cone selection, direction-changing tree transitions, automatic truncation/native
admission, public table tag, periodic composition or complete directional FMM
resource/scaling/physics qualification. See the accompanying phase assessment
for selected original weak-action gates and preserved source reconciliation.
