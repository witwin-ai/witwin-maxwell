# Internal prescribed directional scalar solver

`DirectionalMultipoleOperator(provider, actions, solve_config=DirectSolveConfig())`
in `fmm_operator.py` requires an actual complete `DirectionalSystemActions`, its
exact admitted provider identity and current immutable bindings. It returns the
caller-owned actions directly through `_build_actions` and delegates native Riesz
construction, forward/Hermitian/current actions, bounded state, residual loss and
Krylov solve to `DirectHierarchicalOperator`. No system tables or maps are rebuilt.

The shared LocalRWGRiesz uses the exact face-local RWG Gram L2 dual proxy with the
actual exterior impedance. It is not a certified trace-space norm. The actions'
common-eta ideal Frobenius truncation receipt remains a separately labeled ideal
quantity; it is not converted into a Riesz or finite implemented-operator residual
certificate, including when the prescribed common eta differs from exterior eta.

Use shared `StoppingRule` iterative configuration and `none`/`rwg_diagonal` scales.
The original fixed-system custom backward supports native current derivatives;
geometry/material parameter derivatives and solve-through parameter derivatives
are not established here. GMRES and block GMRES retain their existing convergence,
actual residual checks and no dense-fallback policy for this matrix-free operator.
An inner-converged result is accepted only if compressed dual and selected original
rows satisfy `DirectSolveConfig`'s finite positive tolerances, defaulting to1e-8.
Original rows are a sample by default and do not certify a global original residual.
Callers can request every edge for small independent controls without changing the
runtime qualification flags.

The adapter guards its own provider/actions/tree/Riesz bindings through shared
read access; complete actions and metric guard underlying versions in saved current
backward. `persistence_receipt` describes configuration and construction only, not
an executable snapshot. `operator_stats` labels the backend directional_fmm_internal
and all physical/resource/complexity/production qualifications remain false.
Component action ownership and native Riesz storage are not a total solve/setup/
process-peak resource accounting. Automatic cones, public/periodic configuration,
finite/native budgets, positive interior FAR physics and final acceptance remain
open. See the matching assessment for selected frozen evidence.
