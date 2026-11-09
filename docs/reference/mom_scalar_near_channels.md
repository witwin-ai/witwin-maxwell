# Original scalar near D/K channels

For one finite passive scalar medium, the selected PMCHWT matrix is
`A=[K,aD;bD,K]`, with `a=i*omega*epsilon` and `b=-i*omega*mu`.
The new internal helper integrates original D/K using the existing touching,
adaptive-near and far quadrature and full RWG supports. It does not divide an
assembled matrix by material factors or infer kernels from canceled entries.
The default four-block helper retains its original expressions and scatter order.

`ScalarNearChannels` owns two independent complex128 arrays and two cloned
ordered edge maps on fixed CUDA float64 geometry. Retained storage is
`32*nr*nc+8*(nr+nc)` bytes; this budget is checked before integration.
Explicit caller-provided prepared geometry is shared and version-guarded.
The default does not retain a newly prepared geometry. Assembly temporaries,
allocator cache, input basis and shared geometry are excluded from this budget.

For row-vector current batches, forward computes
`(J K^T + a M D^T, b J D^T + M K^T)` and Hermitian action computes
`(J conj(K) + conj(b) M conj(D), conj(a) J conj(D) + M conj(K))`.
Four-RHS chunks support rectangular shapes and noncontiguous leading dimensions.
Native current backward re-enters the guarded opposite action without saving
current/kernel action graphs. Material and geometry gradients remain unsupported.

Basis inputs, owned arrays, medium values, material factors, complete integration
configuration, geometry and metadata are bound against later changes. Nonfinite
kernels/actions reject. Accumulating D/K before material multiplication changes
floating-point operation order; no bit equivalence with four-block assembly is
claimed. Numerical gates retain their original thresholds.

This is an internal representation. Packed FMM storage, capacity/configuration,
public result identities, Bench replay, whole-system resources and memory
crossover require later integration and actual acceptance. No production,
physical, complexity, peak-memory or final Goal qualification is granted.
