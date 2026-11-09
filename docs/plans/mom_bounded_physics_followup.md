# Bounded scalar physics follow-up

GSP-01 remains open despite the 521-test engineering pass. An independent
execution-order review recommends finite dense controls before advancing fast
backends solely to obtain larger physical-reference meshes.

Two prospective exploratory runs are started on the unchanged production
operator source at documentation HEAD 7efde3d:

- GPU0: original total-trace method, resonance peak, subdivision 4. Preserve
  independent Mie far/interior/exterior samples, coefficients, actual true
  residual and peak allocated memory. Wall limit 3600 s; allocator fraction .70.
- GPU1: analytic-lifting prototype, epsilon=1.01, subdivision 3. Same far-field
  samples as prototype v1, q6 RHS; exact self-panel magnetic zeroing and larger
  bounded batches. Wall limit 3600 s; allocator fraction .70.

A timeout or capacity rejection is a retained failed/incomplete observation,
not a qualification. No error threshold is relaxed. Based on these results,
consider lifting n4 and frequency meshes 20/24 under fresh explicit budgets.
The existing frequency mesh allows intermediate sizes: f20 has 8000 faces,
12000 edges and a 9.216 GB complex128 matrix; f24 has 11520 faces,17280 edges
and a 19.110 GB matrix. Compact Gram/factor and resident LU add substantial
storage, as do global pair classification and quadrature workspaces. The
existing sealed host restore mode may avoid resident LU but needs actual host
capacity checks. These are planning lower bounds, not fit guarantees.

Only after finite intermediate dense measurements demonstrate a resource
obstruction should execution order be revised to borrow minimal GSP-02/free-
space resource work. GSP-01 would remain open, and that dependency work would
have its own commit and dense equivalence controls. Multi-body, magnetic,
periodic and unrelated feature work cannot be justified by this dependency.

The weak-scattering n3 probe completed: far relative error 0.00833716, still
above 0.005. Next bounded experiment on GPU1 checks q4 far-panel RHS against
the retained q6 n2 result at relative budget 1e-4. Only if this gate passes does
it run n4 with q4 far/q6 touching-near rules. The prospective script records
both the reference NPZ hash and measured gate; timeout remains 3600 seconds and
allocator fraction .70. The original n4 resonance probe is still running.

Legal sliver controls have become two actual CUDA regressions. Each verifies
closed valid geometry, nonzero near-pair participation, adaptive quadrature
limits, true residual, A/A* and exterior queries under dense/compact storage.
The final strengthened tests passed 2/2. Independent read-only review passed;
next complete source catalog contains 523 nodes, so the old 521-node engineering
receipt is not retroactively extended to these tests.
