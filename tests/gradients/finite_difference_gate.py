"""Shared finite-difference acceptance gate for the thin-wire gradient suites.

Both ``test_fdtd_thin_wire_adjoint.py`` and ``test_fdtd_thin_wire_port_adjoint.py``
gate their central-difference sweeps with the same rule; keeping it in one place
stops the two copies from drifting apart.
"""

from __future__ import annotations

from witwin.maxwell.fdtd.thin_wire_reference import ACCEPTANCE_BUDGET


def assert_finite_difference_agrees(errors, *, context=""):
    """Gate a central-difference sweep without best-of-three cherry-picking.

    The finest step decides the verdict, so a single lucky step in the middle of
    the sweep can no longer carry the test. Coarser steps may exceed the budget
    only while the sweep is truncation-dominated, in which case refinement must
    measurably reduce the error. Once every step is inside the budget the sweep
    has reached the roundoff floor, where ordering carries no information.

    The second clause is deliberately conditional rather than an unconditional
    ``errors[0] > errors[-1]`` monotonicity assertion: some registered sweeps
    (e.g. the continuous oblique port coordinate adjoint) sit at the roundoff
    floor across all three steps and are mildly non-monotone there, yet every
    step is inside the budget. Requiring monotonicity unconditionally would make
    those flaky without adding signal.
    """

    budget = ACCEPTANCE_BUDGET.gradient_relative_error
    assert errors[-1] < budget, f"{context}errors={errors}"
    assert max(errors) < budget or errors[0] > errors[-1], f"{context}errors={errors}"
