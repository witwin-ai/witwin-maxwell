# Explicit scalar D/K FMM near storage

`FMMConfig(near_storage="scalar_dk_v1")` selects experimental free-space scalar
FMM near storage. Its canonical identity is `cartesian_fmm_scalar_dk_config_v1`,
with all table, shift, kernel-bound and near-storage fields explicit. Existing
`full_matrix` configurations retain their payload bytes, versions and construction
arguments; the default near field does not appear in those historical identities.

Each region has one scalar medium: the global exterior and separate body-local
interiors. Original D/K are integrated directly with existing weak quadrature;
global selections route every body pair in the caller's selected order. No
four-block assembly or material division occurs in the D/K near factory.
The common grouped CSR engine retains full J/M coefficient maps, shape groups,
row/column nodes and locations, while each numeric group stores `(blocks,2,nr,nc)`.
Blocks stream individually into final storage. Rectangular regional embedding
retains disjoint test/source namespaces and current spaces.

For `A=[K,aD;bD,K]`, `a=i*omega*epsilon`, `b=-i*omega*mu`, Hermitian action uses
conjugate(b) in the first output and conjugate(a) in the second. Native current
backward uses the opposite guarded action without saving generated action graphs.
Packed gathers/products use the configured block batch and at most four RHS.
Native CUDA float64 geometry and complex128 fixed kernels/currents are required.
Material factors, complete quadrature policy, maps and storage are mutation-bound;
nonfinite product or accumulated output rejects.

Retained planning counts `32*nr*nc` bytes per numeric D/K block instead of
`64*nr*nc`, preserving all full J/M maps and CSR overhead. The complete action
capacity still includes trees, centers, leaf maps/moments, shifts, M2L and body
indices. Actual unique CUDA storages must equal that plan. Budgets exclude
provider geometry/system, Riesz/solver state, setup/action/backward workspace and
allocator cache. Reduced retained bytes do not establish whole-system memory
crossover or a peak/workspace bound.

Public solve, Result identity, exact J/M replay and Bench native archive use the
new strict configuration version. Selected controls preserve original weak,
complex adjoint and sampled residual thresholds. Periodic/Ewald composition and
complex64 reject before public setup. Material/geometry gradients, directional
FMM, populations, whole resources, crossover/N-scaling and physical/production
qualification remain separate and OPEN.
