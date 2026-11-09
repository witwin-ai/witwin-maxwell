# Prescribed changing-carrier DAG contract

`DirectionalRWGDAGProjection(basis, tree, carriers, orders, requests, links, wave_number, quadrature_order=q, ...)` builds active contexts on a complete fixed CUDA float64 physical support tree. A request is(physical node,carrier/order family). A link is(parent key,child key,phase order L). Carriers are canonical fixed unit vectors; equal carrier/order families cannot be duplicated. Explicit links cover every direct physical child exactly once and every active context must be reachable from requests. Child polynomial order must cover parent+L within0..12.

A shared child is one integrated/propagated context with several parent edges. Native `source_upward`, `source_downward`, `receiver_upward`, `receiver_downward` reuse the original distinct four maps and guarded custom current/higher-current protocol. Arbitrary coefficients at internal contexts are supported. Downward sums incoming parents once before visiting descendants. Native table/layout/capacity calculations remain primary shared implementations.

Use `ideal_phase_path_bound(context)` only as an ideal maximum multiplicative path defect. It does not establish derivative, quadrature, native, material/kernel or complete-system error. `receipt()` reports exact component tensor ownership with caller geometry/tree/Python metadata and whole inputs/outputs excluded. It grants no whole peak, crossover or complexity claim.

The DAG is explicitly distinct from `SparseDirectionalRWGProjection` and old full-node trees. Existing fixed-phase sparse M2L refuses it. Actual Maxwell DAG coupling requires new phase-inclusive material admission; do not bypass the type guard or reinterpret context IDs as physical node IDs. All numerical tensors/configurations remain fixed and guarded, including saved-output backward mutation checks. Geometry/material gradients remain outside this contract.

Evidence: `docs/assessments/sie_directional_rwg_dag.json` and `/tmp/gsp-directional-rwg-dag-evidence-v1/summary.json`. The complete Goal, all original failures and original final acceptance gates remain active. No production qualification.
