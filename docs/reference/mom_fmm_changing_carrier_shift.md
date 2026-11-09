# Padded direction-changing directional shifts

`DirectionalChangingCarrierShift` takes d=childcenter−parentcenter, child unit
carrieru and parent unit carrierv, passive fixed scalar k, source/target/phase
orders, and a positive fixed CUDAfloat64 child support radius. All directions and
geometry are cloned. Native orders satisfy p_source>=p_target+L within0..12.
Full child support |y|<=radius is assumed here; future tree composition must
check it. No same-order phase approximation is silently introduced.

For beta=gamma+eta, gamma<=alpha, |eta|<=L, the source matrix accumulates
exp(-ik*v.d)*binom(alpha,gamma)*d^(alpha−gamma)*(-ik*(v−u))^eta/eta!.
Source action is Qminus; its Hermitian is Qminus.H. Receiver table Qplus uses
independently positive signs. Physical receiver action is Qplus.T and its
Hermitian is Qplus.conj(). All four vector/divergence channels use the same fixed
phase/polynomial transfer; physical carrier derivatives are not approximated or
certified by this contract.

An ideal relative phase Taylor bound exp(2x)*x^(L+1)/(L+1)!, x=|k|*||v−u||*radius,
is evaluated logarithmically against explicit phase_relative_budget. Exact zero
wave/equal directions distinguish mathematical zero from native underflow.
Finite/nonzero component/phase/structural term checks reject unrepresentable
coefficient construction. Native0**0 is avoided to preserve polynomial semantics.
The ideal bound is not a certified floating-point/current/kernel/global error.

`source` maps(...,n_source,4) to(...,n_target,4); `receiver` reverses the sizes.
Both accept a boolean adjoint flag, matching fixed CUDA complex128 values and
arbitrary leading RHS axes. Original custom current actions reenter complete
guarded Hermitian maps, supporting higher-current derivatives without saved
table/current graphs. Finite checks/matmul use RHS chunks4 by default.

All10 owned tensors comprise cloned d/u/v/radius, x/logbound, two center phases
and two independent tables:128+32*n_source*n_target bytes. Conservative visible
workspace preflight precedes table generation, which streams<=4096 contributions.
Requested caller outputs, allocator cache/workspace and whole systems are
excluded. Every scalar policy, tensor version and map identity is guarded during
action, receipt and saved backward. Peak/resource and physical qualification are
separate. Receipt experimental_directional_changing_carrier_shift_v1 records
ideal-phase scope and false cone/tree/kernel/FP/physics/complexity/peak status.
