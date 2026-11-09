# Explicit disjoint directional FAR actions

`DirectionalNestedFarActions` in `mom/fmm_directional_rwg.py` composes an
explicit FAR component in full physical J/M row and column ordering. It takes
current complete test/source `NestedDirectionalRWGProjection` prototypes,
one finite-frequency scalar medium, admitted `from_nested_nodes` couplings,
an explicit positive ideal absolute RWG error budget, and bounded RHS/workspace
settings. The deterministic default balancing impedance is `abs(medium.eta)`,
including an empty coupling list. This is an internal component.

Every caller coupling must use that same medium object and balancing impedance.
Its prescribed maps must share their respective prototype's physical basis and
support-tree identities. Different carrier/projection owners on those trees are
allowed. The node admission and all external basis, geometry, tree, transfer,
projection and coefficient guards remain active on forward, Hermitian and saved
current backward. Arbitrary moments or unadmitted/direct-projection couplings do
not establish prescribed-node provenance.

Each complete node selects a half-open interval in the shared tree permutation.
The component checks disjoint physical test/source rectangles with an integer
row-event sweep and sorted active column intervals. End events precede starts at
the same row boundary. Adjacent rectangles are valid; duplicates and partial
overlap reject before new log-bound or action buffers. The check stores O(P)
integer metadata for P blocks and creates no physical N-by-N coefficient mask.

For the common balancing impedance eta, define the error norm as
`Frobenius(diag(eta_J,1_M) error diag(1_J,eta_M))`. Disjoint rectangles satisfy
`B_far^2 = sum_i B_i^2`. The implementation aggregates admitted ideal block log
bounds with CUDA logsumexp and the existing positive-floor/zero receipt rules,
then checks the supplied absolute budget before current actions. An empty
component has exact zero bound and action. This admission follows construction
of caller couplings; it does not preflight their coefficients or moments. Media
contributions that overlap one physical rectangle must first combine by the
triangle inequality in a subsequent system composition.

For each physical RHS chunk, each source owner runs source upward once.
Couplings translate the prescribed source nodes into target node-local buffers;
each target owner runs receiver downward once, and physical outputs are added.
The complete Hermitian uses receiver upward, coupling Hermitian, and source
downward. The native fixed-current custom backward re-enters these guarded
actions, saving no generated projection graph and preserving higher current
derivatives. Noncontiguous leading axes and empty RHS batches use the existing
bounded native chunk path. Nonfinite input or accumulated output rejects.

All full-node source caches and target local buffers can coexist. The planning
allowance charges those buffers, physical current stacks/output temporaries,
the maximum active map workspace and maximum coupling workspace. Generated
locals and moments are released at the end of each chunk and action. The
component owns no newly retained CUDA arrays: caller couplings, maps, geometry
and trees are referenced and guarded, and their storage is explicitly excluded.
Caller input/output tensors, allocator cache and backend FFT library workspaces
are also excluded from this tensor-workspace allowance. It is neither a whole
process peak measurement nor a complete system capacity/crossover receipt.

Untouched near coefficient slots remain zero. The bound covers only ideal
exactly integrated prescribed-node kernel truncation and phase products. Leaf
carrier quadrature, native coefficient/FFT/transfer/accumulation errors, complete
near plus FAR partition coverage, actual automatic cones, public/periodic
integration, resources, complexity and physical/production qualification remain
required. Geometry/material gradients are not introduced by this component.
