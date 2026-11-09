# Fixed-carrier directional tree-center algebra

Let d=c_child−c_parent, t=y−c_child, with one fixed real unit direction u.
Source moments integrate t^alpha exp(-ik*u.t). Parent offsets are t+d, hence
M_parent=exp(-ik*u.d) S(d) M_child, with real Cartesian binomial S.
This is exact at each finite retained degree in exact arithmetic.

Receiver fields are exp(+ik*u.(x−c_parent)) times a parent polynomial. Reexpressing
at the child center gives L_child=exp(+ik*u.d) S(d)^T L_parent. Their Hermitian
maps separately conjugate their own phases and transpose S. Complex passive
carrier phases are not mutual conjugates; both are evaluated and validated.

DirectionalFixedCarrierShift reuses SeparableMultipoleShift. Explicit source
and receiver maps provide source(...,adjoint=True) and receiver(...,adjoint=True)
without confusing their mathematical roles. Fixed-current custom backward reenters
these complete guarded Hermitian maps, retaining no generated gather/weight graph.
CUDAfloat64 fixed geometry, complex128 four physical vector/divergence channels,
p0..12, chunks4 and positive budgets are required. Source and target directions
must match every byte, including signed zero; direction-changing transitions
cannot use this finite-polynomial identity without another phase truncation bound.

Owned layouts contribute18arrays to actual retained/preflight counts, while
explicitly caller-shared layouts are excluded and version-validated on every
map. Direction/displacement clones and both phase scalars are always owned.
Each phase must be finite/nonzero; actions reject nonfinite outputs or silent
whole-complex phase-product underflow. Visible workspace excludes allocator,
caller state and whole tree/system resources; it is not actual peak qualification.

Finite native coefficients and exact-arithmetic identities do not certify all
floating-point inputs, directional cones, phase quadrature, physical kernels,
full-system resources or N-scaling. The phase assessment binds the selected
actual point/adjoint/current and storage/range controls. Tree/cone and changing
carrier integration, truncation/native admission, public/periodic storage identity
and complete Goal qualification remain required.
