"""E4b fixed-cost reduction gates for the connected network coupling path.

Non-numerical op-count gate (``perf-opcount``) plus a numerical no-regression
gate. The op-count gate asserts the composite same-step solve cuts the per-step
kernel-launch count of the delay-free 8-port/order-32 feedback block by at least
the pre-registered target relative to the legacy sequential-LU substitution
(measured 78 -> 27 launches, 65%). The no-regression gate asserts the composite
branch current reproduces the legacy pivoted-LU solve to within float64
tolerance on the same prepared inputs.

Both schedules run on the same prepared solver via the shared measurement
module, so the launch delta isolates the removed triangular-substitution
kernels and cannot drift from the artifact generator.
"""

from __future__ import annotations

import pytest
import torch

from tests.support.network_coupling_op_stream import (
    build_connected_solver,
    measure,
    numerical_equivalence,
)

# Pre-registered launch-reduction target for the delay-free connected coupling
# path (justified by the profiled 78 -> 27 launch drop: the sequential 8x8
# triangular substitution collapses to two dense matvecs against the constant
# loop operator).
_LAUNCH_REDUCTION_TARGET = 0.30


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_composite_solve_cuts_coupling_launch_count() -> None:
    solver = build_connected_solver(grid_cells=48)
    tallies = measure(solver, warmup_steps=8, profiled_steps=40)
    before = tallies["before"]["launches"]
    after = tallies["after"]["launches"]
    assert before > 0
    reduction = 1.0 - after / before
    assert reduction >= _LAUNCH_REDUCTION_TARGET, (
        f"launch reduction {reduction:.3f} below target {_LAUNCH_REDUCTION_TARGET}"
    )
    # The composite path allocates nothing per step and adds no host-side copy.
    assert tallies["after"]["allocs"] == 0
    assert tallies["after"]["memcpy_hostside"] == 0
    assert tallies["after"]["scalar_sync"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_composite_solve_matches_legacy_lu_no_regression() -> None:
    solver = build_connected_solver(grid_cells=48)
    ratio = numerical_equivalence(solver, steps=64)
    # The composite and sequential-LU solves are mathematically identical, so
    # even on adversarial random unit state (which cancels the ill-scaled
    # C@state matvec) their difference must stay within the matvec's own
    # floating-point roundoff bound. A ratio < 1 means the change is pure
    # rounding; a real algebra regression rides orders of magnitude past it.
    assert ratio < 1.0, f"composite vs legacy LU residual/bound ratio {ratio:.3e}"
