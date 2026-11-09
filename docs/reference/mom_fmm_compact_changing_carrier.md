# Factorized changing-carrier transfer contract

Use the existing `DirectionalChangingCarrierShift` with explicit
`phase_storage='compact_phase_v1'`. Defaults retain dense storage. For parent
polynomial gamma and phase eta, `indices[gamma,eta]` selects child beta=gamma+eta;
Pminus[gamma,beta]=(-ik(v-u))^eta/eta!, with |eta|<=L. Pplus uses independent
positive signs, not conjugation at passive complex k. Source order must cover
parent order plus L, and every native order lies within 0..12.

With d=child-parent and S the real separable parent center shift, source is
exp(-ik v.d) S Pminus. Its full Hermitian reverses the order. Physical receiver
is exp(+ik v.d) Pplus.T S.T; its Hermitian uses S Pplus.conj(). Four physical
vector/divergence channels remain distinct. Receiver/scatter maps accumulate
all repeated indices. Neither atomically accumulated outputs nor current
backward are promised bit-exact between calls.

`ChangingCarrierPhaseLayout` shares fixed CUDA integer indices, powers and
float64 inverse factorials for (child order,parent order,L). Supply it through
`phase_layout`; supply a matching `SeparableMultipoleShift` through
`center_layout`. Each is independently identity/version guarded, including saved
current backward. Owned layouts are counted; explicit caller-shared layouts are
excluded from edge ownership. Default ownership is
128 + 32*nphase + (8*nparent*nphase + 32*nphase) + 144*nparent*(pparent+1) bytes.
Shared ownership is 128 + 32*nphase. There are no dense Q allocations.

Configuration and conservative visible tensor-workspace budgets reject before
layout construction. Native coefficient products, phases and current products
must remain finite and nonzero when structurally nonzero. These factorized
checks/evaluation order have a different admission range from dense storage.
Complex zero powers preserve polynomial 0**0=1 without evaluating that native
branch. Actions use bounded RHS chunks (default four), complete custom current
Hermitian maps and higher-current derivatives without saved generated graphs.
Strict center actions release each axis gather/product before the next axis.

The relative phase bound is ideal arithmetic and assumes the caller's complete
support lies within child_radius. It certifies neither roundoff nor derivatives,
cancellation-relative currents or EM kernels. Receipt version is
`experimental_directional_compact_changing_carrier_shift_v1`; all tree/cone,
global error, whole-peak, physics, complexity and production qualification flags
remain false. Compact support-tree shared inventory and full preflight are the
next required implementation step.
