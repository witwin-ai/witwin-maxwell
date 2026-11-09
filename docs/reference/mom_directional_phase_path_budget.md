# Nested ideal phase-path and prescribed-node budget contract

NestedDirectionalRWGProjection.ideal_phase_path_bound(node) computes an ideal
worst descendant leaf-to-node phase product. Each padded changing edge satisfies
p_child>=p_parent+phase_order. Source and receiver phase Taylor signs are independent;
both have ideal relative bounddelta_e. Exact edges contribute0 based on exact_phase;
their storedlog_phase_bound=0 is a sentinel and must not mean delta=1. CUDA leaf
logproducts are0; internal L_node=max_child[L_child+log1p(delta_child)]. Thus
Delta_node=expm1(L_node)>=max_path[product_e(1+delta_e)-1]. Strictly positive native
nonexact defects are floored before exp; unavailable/overflow rejects. Fullnode
workspace nodes*8*16+4096 is preflighted; no persistent arrays/receipt/action changes.

For two actual prescribed nodes, from_nested_nodes requires current matching
native device/order/carrier/wave and positive finite explicit kernel, balanced
RWG absolute and impedance budgets. Asymmetric orders reject. Fullprovider faces,
edges/nodes/support/path/statistic temporaries enter conservative visible tensor
preflight before allocation. Recheck all complete selected range support corners
about stored centers; radius is max(existingradius,inflatedactualcornerbound),
never shrunk. This is not outward-rounded certification. Caller maps/trees excluded.

With d=norm(R)-r_test-r_source>0, passive real-support majorants are
M_G=exp(-Im(k)*d)/(4*pi*d) and each M_gradient<=M_G*(abs(k)+1/d).
Delta=expm1(L_test+L_source). For ideal direct polynomial errorE, the complete
prescribed phase product error is E+Delta*(M+E). Evaluate each physical-gradient
component separately in CUDA log space; do not differentiate approximate phases
or restore the carrier a second time. Delta=0 preserves E exactly. Enforce the
supplied kernel budget on COMBINED bounds, then reuse actual selected-node integral
statistics and balanced PMCHWT block_error_log/_receipt for absolute RWG admission.
All admission precedes translation coefficients. Source_upward[selected_node] and
receiver_downward with only this node injected prescribe the supported composition.

The same existing EM moment action/H/current backward binds both tree owners,
selected node ids in flat ideal metadata, budgets/impedance/receipt, tree/geometry/
transfers/sharedlayouts and tensors. Owned arrays remain the configured scalar
translation only. Scalar assumed-support receipt remains unchanged for arbitrary
use; weighted node receipt supplies verification for the prescribed composition.
No additional truncation occurs in ideal padded phase/center transfers. The path
bound is not relative-current certification for cancellation-prone fields; RWG
absolute integral majorants provide the subsequent cancellation-safe bound.

Finite leaf carrier quadrature, coefficient/FFT/transfer numerical error, material/
geometry or phase derivatives, global far partition and relative normalization,
cones/public/periodic, full resources/scaling/physics and production qualification
remain separate. Numerical testrelative gates must explicitly disable pytest's
implicit absolute comparison tolerance when required (abs=0).
