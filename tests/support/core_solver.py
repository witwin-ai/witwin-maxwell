"""Bind the core launches to a directly constructed FDTD solver.

``Simulation.prepare()`` activates the core path; a test that builds
``witwin.maxwell.fdtd.solver.FDTD(...)`` itself and calls ``init_field()`` has to bind
the fused core Yee launches the same way before stepping (the per-component runner
was deleted in W7 row 16e). Without source tables the core runtime steps in the
per-step loop.
"""
from __future__ import annotations

from witwin.maxwell.fdtd.runtime.core_path import CorePathRuntime


def bind_core(solver):
    solver._core_runtime = CorePathRuntime(solver)
    return solver
