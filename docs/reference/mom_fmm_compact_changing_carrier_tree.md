# Compact changing-carrier support-tree contract

`NestedDirectionalRWGProjection` accepts explicit `phase_storage='compact_phase_v1'`
only with complete node_directions, node_orders and phase_orders. The default
remains dense changing transfers or the original fixed carrier. Root configuration,
unit/passive/native policy, polynomial padding and the ideal per-edge phase budget
are validated before preparation. Complete plus/minus RWG support boxes and
conservative actual vertex-corner radii remain checked before leaf quadrature.

The tree owns a `ChangingCarrierPhaseLayout` for each unique
(child order,parent order,phase order) and a `SeparableMultipoleShift` for each
unique parent order. Every matching edge receives the exact shared objects.
An edge retains only its 128-byte geometry/phase metadata and two independently
signed complex phase vectors, 32*nphase bytes. Its layouts are excluded from edge
ownership; the owning tree counts all unique phase/center layouts once. Layout
bytes are 8*nparent*nphase+32*nphase and 144*nparent*(pparent+1), respectively.
Leaf maps, centers/radii, node directions, owned geometry and edge maps continue
to use their original inventories. Caller-shared tree/provider geometry remains
excluded and guarded. Single-node trees allocate no phase or center layout.

Complete retained and conservative tensor-workspace planning precedes geometry,
layouts, moments or transfers. The shared compact primitive workspace is fully
included even though its retained layouts are tree-owned. Tree setup and full
all-node/RHS output buffers are separately bounded. Requested caller all-node
inputs/outputs, allocator/cache and whole-system resources are excluded; this
preflight is not a measured allocator peak or complexity qualification.

All four map loops preserve variable coefficient counts: source upward and its
complete Hermitian downward; physical receiver downward and its complete
Hermitian upward. Integrate leaves only, then apply factorized padded phase and
center transfers. Custom current backward saves no generated node graphs and
accepts absent internal-node cotangents. Layout cache keys/identities, policies,
orders and every owned tensor version are guarded on action, receipt and saved
backward. Shared tensors appear once in tree `tensors()` and no edge duplicates
their ownership. Repeated-index reductions do not promise bit-exact action replay.

Receipt `experimental_compact_changing_carrier_nested_rwg_v1` records exact shared
layout counts/bytes and tree ownership. It does not claim identical native
operation order/range to dense storage. Ideal per-edge phase admission is not a
whole-tree relative-current, derivative, floating-point, quadrature or EM/kernel
error bound. Automatic cone selection, global kernel admission, public/periodic
integration, actual whole-system resources/crossover/scaling/physics/populations
and production qualification remain required.
