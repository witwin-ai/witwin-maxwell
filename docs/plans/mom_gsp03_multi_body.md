# GSP-03 multi-body implementation candidate

GSP-01 stays open; its retained physical failures do not block development.
This stage is an implementation candidate, not a production qualification.
Full regression, cross-repository acceptance and independent review are deferred
to the combined stage merge point and final acceptance, per user instruction.

## Implemented contracts

- SurfaceBody / SurfaceSystem retain independently validated connected, closed,
  oriented meshes; CUDA triangle-pair distances reject intersections, touching
  and zero gaps, and containment checks reject nesting. Face and edge offsets,
  unique aliases, enabled Scene ordering and mesh mutation guards are explicit.
- Global J-all/M-all assembly separates test/source bases. Diagonal blocks use
  exterior plus own interior; cross blocks use exterior alone. Cross categories
  never compare local vertex IDs and use actual distances for near adaptation.
  Single-body assembly directly returns the existing matrix without a second
  quadratic allocation. Dense scaling remains quadratic, not a fast backend.
- Compact Riesz storage keeps separate body Gram/Cholesky factors, exact global
  slices, batched inverse and coefficient autograd. Mutation invalidates factors
  and persistence identity; direct sealed storage rejects differentiable loans.
- The public runtime compiles every enabled structure and binds the complete
  Scene signature. SurfaceData exposes J_scattered/M_scattered, body views,
  explicit aliases, per-body physical total-current evaluation and region queries.
  Historical total J/M coefficient labels are deliberately not reintroduced.
- Analytic lifting uses one common immutable incident carrier. Each RHS block
  is the local medium-difference action; exterior equivalent incident traces
  vanish outside their own body. The system matrix is unchanged by this lift.
- Exterior fields sum all scattered potentials and add the incident field once.
  Only the owning body receives a trace jump. Interior fields use only their
  own interior representation. Adaptive exterior estimates compare coherently
  summed complete fields, including cancellations. These estimates are not
  rigorous error certificates. Fixed-order queries remain explicitly unestimated.
- Far fields sum complex amplitudes before power integration. Absorbed powers
  integrate each body's supplied physical volume rule; totals require every body.
  PointMonitor metadata records region_id, structure_index and structure_name.
- Executable inner snapshot schema 3 binds ordered bodies, media, offsets,
  slices, geometry, common carrier, numerical and decoder identity. Restore
  compiles and validates geometry without assembling an operator. Historical
  schema 1 archive migration retains its own original signature semantics.
- Regular-mode weak RHS and control spheres support the global body layout;
  modal calibration applies body mass factors without a dense global zero block.
  These low-level modal probes explicitly return unlifted total-RWG columns,
  not physical lifted SurfaceData. A generalized analytic modal carrier remains
  an acceptance/integration obligation; no probe is relabeled as a lifted solve.

## Targeted validation and open acceptance

CUDA tests already passed: 10 geometry/layout/negative tests; 2 independent
cross-block and single-body reduction tests; 2 block Riesz batch/gradient/cache/
sealed-storage tests. One initial block fixture used face 100 of an 80-face
source and failed with a CUDA assertion; it was corrected to valid face 50 and
both affected tests passed (the failure log is retained).

Public integration run `/tmp/gsp03-public-v2.log`: 20 passed in 70.78 seconds.
It covers invisible two-body fields and owning traces, coherent nonzero random
fields/far fields and gradients, nonzero lifted physical RHS and exact restored
queries, corruption rejection, complete-field adaptation, per-body absorption,
global modal RHS, body permutation of A/b/x, single-body physical persistence,
and historical archive rejection. Later changes tighten snapshot body/kernel
validation and avoid a second single-body matrix allocation; their small
follow-up validation is recorded below when complete.

Open obligations are retained: independent two-sphere multiple-scattering
reference, near-gap convergence and energy sweeps, far-separation independent
limit, precision/direct/GMRES/block/device matrix, lifted modal carrier, Bench
schema-3 integration, complete regression and independent merge-point audit.
No old GSP-01 physical failure or threshold has been removed or relaxed.

Final targeted follow-up `/tmp/gsp03-final-targeted.log`: 6 passed in 28.81 s,
covering the tightened restore and single-body allocation path. A distinct
nonzero trace ownership/interior isolation check passed in 8.84 s
(`/tmp/gsp03-trace-owner.log`): foreign currents changed by factors of one million
leave the target body's interior representation bit-identical; its exterior
trace agrees with owning jump plus the foreign off-surface potential.
