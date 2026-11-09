# CUDA block GMRES

`StoppingRule(method="block_gmres")` selects a shared block Krylov solve through
the existing MoM `correct`/`solve_reference` interfaces and `MoMConfig.stopping`.
Default scalar GMRES and cached direct solves retain their previous behavior.
Vectors keep the existing last-dimension coefficient convention; leading RHS
batch dimensions are flattened into one block and restored on return. A single
RHS uses the same algorithm with initial block width one. No CPU numerical solve
or new public operator protocol is introduced.

The algorithm uses variable-width block Arnoldi for `L A R`, twice modified
Gram-Schmidt, rank-revealing CUDA SVD and a shared block Hessenberg least-squares
problem. The corrections are applied as `x0 + R V Y`. Every column minimizes its
own scaled residual in the common space; this is a true block method, not a loop
of independent GMRES calls. Existing row and row/column scaling are supported.
The original unscaled residual `A x-b` is recomputed after every block step.
Each RHS independently converges at `max(atol, rtol*norm(b))`, stagnates, or fails;
finished columns are locked while the remaining columns continue.

Initial scaled residual columns are normalized before rank detection, so a weak
RHS is not removed merely because an unrelated RHS has much larger amplitude.
Dependent directions are removed with a machine-precision/shape-scaled SVD
cutoff. Zero RHS can converge immediately from zero initial state, or be corrected
from a nonzero guess under the absolute floor. Rank-deficient seed blocks and
more RHS than unknowns are supported. When an exhausted full-rank Krylov image
leaves cancellation roundoff from a large initial guess, the method restarts from
the actual residual within the same budget. An inconsistent singular projected
system reports breakdown. SVD failures and nonfinite residuals are explicit
unsuccessful states, never convergence based only on a small projected residual.

`maxiter` counts block Arnoldi steps and `restart` counts block steps per cycle.
These are not scalar-GMRES iteration budgets: each block product can act on
multiple vectors. The basis is bounded by the unknown dimension and each cycle's
accumulated block rank. Callers should chunk very large RHS populations according
to their available memory. There is no claimed speed advantage or asymptotic
complexity improvement over the existing dense backend.

The parent solve log reports actual `block_operator_products`,
`block_residual_products`, `arnoldi_vector_actions`, `residual_vector_actions`, and
`matvecs` (their vector-equivalent sum), plus block ranks, peak basis size and
shared setup/solve times. Per-RHS logs contain their own residual histories and
checks; their `matvecs` count only those checks. Shared Arnoldi work is recorded
once in the parent log and must not be multiplied by RHS count. Iterations in a
column history are global block-step indices, not extra per-column products.
Timing/memory retain the existing CUDA synchronization and allocator-scope rules.

An explicitly requested `fallback="direct"` preserves the unsuccessful block
attempt in `gmres_attempt`, solves through cached CUDA LU, and includes both
phases' costs. The block-specific cost fields describe the primary attempt;
`fallback_log` describes the direct phase. Optional initial-subspace recycling
remains available for scalar GMRES and is not combined with the new block method.

The private `_single_gmres(..., action=...)` hook lets benchmark operator
prototypes reuse the same scalar residual-certified algorithm. Its default is
unchanged `apply_A(handle, ...)`; it does not broaden `DenseOperator`'s public
contract or create another solver implementation.

Run `python -m benchmark mom block --rtol 1e-10 --output report.json` for the actual 480-edge,
eight-source lossy PMCHWT campaign. It records zero/random initial guesses,
independent CUDA-LU coefficient comparisons, per-column true residuals, all costs,
raw initial/RHS/solution arrays, environment, source hashes and exact source bytes.
The test suite additionally includes a one-step case where a shared block solves
an invariant source subspace while independent scalar one-step solves fail.

The driver defaults to the original `1e-8` residual target. Its separate `1e-6`
impedance-balanced coefficient-error check can require a tighter residual on an
ill-conditioned operator. Both thresholds and failures remain explicit in the
report; tightening `--rtol` does not relabel an earlier failed coefficient check.
