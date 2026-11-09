# Fixed-carrier nested directional RWG projections

`NestedDirectionalRWGProjection` accepts one fixed RWG basis or global
`ScalarSystemBlocks`, a complete CUDAfloat64 `SupportOctree`, a fixed unit
direction, a finite passive scalar wave number and order0..12. Carrier quadrature
is explicit and must resolve the polynomial factor; it is not certified by this
minimum. Geometry, direction, topology and operator configuration remain fixed.

For d=child center−parent center, source moments shift with exp(-ik*u.d)S(d).
Physical receiver locals shift with exp(+ik*u.d)S(d).T. The four methods are:

| Method | Map | Input/output |
| --- | --- | --- |
| source_upward | U_s | currents to every node |
| source_downward | U_s^H | every node to currents |
| receiver_downward | R_r | every node to currents |
| receiver_upward | R_r^H | currents to every node |

Upward traversal sums descendants; downward traversal adds arbitrary coefficients
at every node before propagating. Leaves alone integrate actual RWG vector and
physical divergence with independent exp(-ik*u.x) and exp(+ik*u.x) carriers.
No differentiated-carrier divergence is substituted. Global leaves route body
edge groups and offsets, including leaves containing more than one body.

Native custom current backward reenters the opposite guarded full Hermitian map
without saving node/current graphs. Unused output cotangents remain missing and
materialize zeros only inside RHS chunks. Higher-current derivatives are supported.
Fixed geometry/material gradients are outside this contract. Action finite checks
and temporary work use bounded RHS chunks; caller full-node results scale with
their requested shape and are not constant-workspace claims.

Owned retained bytes for n nodes, e edges, c coefficients and g leaf-body groups:
32*n+24+8*e+128*e*c+48*g+104*(n−1), plus default-created geometry and layout.
Default geometry is128B per triangle. One default layout is144*c*(p+1)B.
Caller-shared tree/geometry/layout are excluded but guarded. Setup preflight also
budgets visible geometry/quadrature/shift/action tensors conservatively. Whole
allocator peaks and system solver resources require separate actual measurement.

Receipt `experimental_fixed_carrier_nested_rwg_v1` declares fixed-carrier tree
integration and keeps cone transitions, truncation, quadrature convergence,
physical/complexity/peak qualification false. Changing carriers require additional
phase approximation and independent admission; same-order fixed shifts cannot
silently supply that contract. See the phase assessment for selected actual
all-node and nonroot high-ka weak controls and unresolved full Goal requirements.
