# Sparse requested directional RWG context forest

`SparseDirectionalRWGProjection` in primary fmm_directional_moments.py constructs
only the descendant closure of prescribed (physical node, carrier/order slot)
requests. Input carriers are fixed native unit vectors, with a polynomial order
per slot and at most32 slots. Requests are unique and use every slot; duplicate
bit-identical carrier/order slots reject to require a canonical upstream catalog.
Within a family, requested ancestor/descendant closures share every context.
Contexts sort by carrier slot then original topological node index, forming a
forest with complete descendants and no allocated absent cone/tree slots.

The original complete physical tree and prepared geometry remain caller-owned.
Validate its full connected topology/permutation, native boxes and actual active
RWG support enclosure. Node centers use original box midpoints; enclosing radii
use actual selected complete support-box corners with512 float64 eps inflation.
Global mixed-body leaves route original edge indices to each actual body basis
and prepared geometry. Fixed geometry is mandatory at construction and action;
changes to mesh/prepared/carrier/map/tree autograd modes invalidate saved output
backward as well as direct actions/receipts.

Each active leaf integrates independent source and receiver carrier maps with
existing direct RWG projection. Each active parent/child context uses the
existing fixed-carrier center shift; unique polynomial-order layouts are shared
across families and counted once. Same-family descendant phase truncation is
mathematically exactly zero, while native phase/integration/transfer errors
remain uncertified. No direction-changing cone transition is implied.

All four native maps and their guarded current/higher-current backward share
the actual method objects from the primary full-tree implementation. The old
shared upward/downward code now counts `len(node_counts)`; for original full
trees this is identical to physical tree.node_count. A sparse forest has its
own context indices, so it deliberately does not inherit the complete-tree
type: existing complete-tree M2L constructors reject it before support/coefficients.
A future explicit sparse-coupling binding must establish these distinct indices,
physical medium, kernel/weighted/global budgets and system partitions.

Source/receiver upward maps expose active-context tuples; downward accepts one
coefficient slot per active context and returns the full physical current
space, summing overlaps across families. Omitted physical edges return exact
zero. Four-RHS chunking and the existing graph-free custom current protocol
cover arbitrary leading dimensions, missing cotangent slots and empty currents.
Caller input/output storage is excluded from component workspace. Internal
four-RHS active-context arrays, physical accumulation buffers, projection setup,
shift and support-validation allowances are included, without measured peak or
whole-system/crossover/scaling qualification.

Persistent native inventory is48 bytes per active context (two int64 IDs,
center and radius),24 bytes per carrier, unique shared separable layouts,
104 bytes per active transfer and each independently owned leaf/body projection.
No full-tree projection owners per carrier are created. Caller trees/geometries
are guarded and excluded. Retained and conservative workspace budgets preflight
before moment/transfer numerical factories, with exact storage checked afterward.
Host topology/control is integer metadata; physical calculations stay on CUDA.

Evidence: `/tmp/gsp-directional-sparse-contexts-evidence-v1/summary.json`.
Independent q16 direct carrier quadrature covers selected single/global/mixed-leaf
forests and all four source/receiver/Hermitian maps. Maximum relative error
1.907498768330214e-15 passes1e-11; native current/higher errors0 and no saved
wrapper tensors.15 noncontiguous RHS, shared descendants/layouts, missing slots,
empty current backward, exactzero20 omitted edges, capacity and saved-output
mutation/mode guards pass. Context counts12/11 replace hypothetical full-tree
slot counts78/102 in these selected controls; this is not N-scaling/crossover.
Only one required affected original single-space map check runs once. Additional
configuration/mode/type checks do not repeat covered numerical checks.

GSP01/GSP09/Goal remain OPEN/ACTIVE. Automatic global cone/order allocation,
actual sparse kernel/coupling/phase/weighted/global/finite quadrature/native
admission, complete interior near/FAR physics, public/periodic integration,
whole resources/crossover/N scaling/populations, current four-repository FINAL
acceptance and at least two FINAL independent audits remain required. This
phase grants no production, physics or complexity qualification.
