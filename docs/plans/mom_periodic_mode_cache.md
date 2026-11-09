# Bounded periodic outgoing modal reuse

The 1D/2D periodic kernels reuse SpectralModeCache, already used by free-space
spatial integration, with a configurable retained-storage budget (default256MiB).
Each kernel owns one immutable material/lattice pair. Keys distinguish dimension,
radiation sign, spectral shell/grid block, transverse mapping/range/order,
explicit 1D normal and pole tolerance. Observation coordinates do not enter the
key: modes are independent of query height and only phase integration is repeated.

Retained storage uses the existing shared-storage-aware LRU accounting and
mutation checks; material autograd graphs bypass retained entries. The bound
covers retained tensors, not build workspaces or active caller references.
Entries larger than the budget bypass it. Zero budget disables retention.
Dataclass replacement constructs a fresh cache. Existing primary/contact,
cutoff, pole, precision and physical-qualification policies are unchanged.

The isolated tree <scratch>/witwin-maxwell-periodic-mode-cache leaves both already
running physics controls frozen. Five targeted nodes check exact cached/no-cache
values, reuse across heights, different grids/normals/maps, native bias gradients,
small-budget bypass and owned-state mutation rejection. Runner
/tmp/gsp-periodic-mode-cache-v1.py/session45068 is initially live; terminal
validation and independent audit are required before acceptance. No timing or
complexity claim is based on concurrently shared GPU1 execution.

## Terminal checks and gradient correction

Initial v1 passed5nodes in27.233 runner seconds but independent audit found an
uncovered P2: a detached warm entry could mask newly differentiable inputs.
The fix adds explicit retain=False to the shared cache API, bypassing both lookup
and storage while retaining counters. Periodic calls decide before lookup using
the owned material and current local tensors, including no_grad warmups of
trainable materials. Three regressions cover same-value trainable normal after
detached warmup and material no_grad-to-grad transitions in1D/2D.

Final v2 passed8nodes, no skips/errors/failures,44.279 runner seconds; shared
cache regression passed4nodes,7.347 runner seconds. Each597source/test hashes
remained unchanged during its run; neither timing is isolated performance
evidence. Initialv1 is preserved but is not acceptance of the gradient fix.
gsp-periodic-mode-cache-v2 receipt SHA256 12a5005bf6621b5f26825256fcc34c1be659ca94fb2f1d3c6ecbc6a1d04a9053
gsp-periodic-mode-cache-shared-v1 receipt SHA256 eadc688d03d676f07af8701533c2faf1a3aa53d4a005242016bfb54869b0d924
