# Per-node directional RWG carrier trees

`NestedDirectionalRWGProjection` optionally accepts `node_directions`,
`node_orders`, `phase_orders` together and `phase_relative_budget` (default1e-12).
Directions are a fixed CUDAfloat64(nodes,3) unit tensor; integer order/phase
tuples have exactly one entry per node, root phase0, root order/direction matching
the original arguments. Each edge requires child order>=parent order+its phase
order within0..12. Fixed-mode configuration, receipts and inventory are unchanged.
Variable mode rejects shared fixed layouts and owns independent phase tables.

Each node uses an explicit own count C(node)=len(multi_indices(order(node))).
All four source/receiver/upward/downward methods retain the original contract,
with node tuple entries(...,C(node),4) and currents(...,total_edges). Missing
cotangents materialize zeros only per chunk at the appropriate node shape.
Current backward and higher derivatives reenter the paired complete Hermitian
map; no generated graph is saved. Every node's arbitrary input participates.

Topology/ranges/permutation and full-support axis enclosure are checked. In
variable mode, max(abs(supportlower−center),abs(supportupper−center)) bounds each
complete RWG support corner. Maximum corner norm with a conservative native
epsilon enlargement determines a finite positive node radius; explicit corner
enclosure is verified before leaf quadrature. Global body routes include mixed
leaves. Only leaves integrate independent physical source/receiver carrier maps.
Per-edge transfers use child/parent carriers and own orders with fullsource/H
and independently positive receiver/H maps. Polynomial quadrature minimum covers
the highest actual leaf order, without certifying oscillatory quadrature.

Variable owned retained inventory is32*nodes+24+24*nodes+8*edges+48*leaf_body_groups
+128*sum(leaf_edges*C(leaf))+sum_edges(128+32*C(parent)*C(child)), plus default
single-body geometry. Caller provider geometry/tree is excluded but guarded.
Setup/action workspace estimates include variable phase matrix construction and
bounded RHS/conjugation work; requested caller outputs and allocator/system
state are excluded. Every policy, node-order/count tuple, own carrier/radius
tensor and transfer/projection identity is guarded on receipt/action/backward.

Receipt experimental_changing_carrier_nested_rwg_v1 reports explicit changing
tree integration and actual support-radius checks. Per-edge ideal relative phase
admission is not a full-tree current/derivative/FP/quadrature/kernel guarantee.
Cone planning, global error/resources/scaling/physics and production qualification
remain false. See assessment for actual selected controls and dense transition
storage limitations; support depth/order padding can reject unsupported schedules.
