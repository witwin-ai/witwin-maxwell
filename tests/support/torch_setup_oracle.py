"""Torch material and coefficient setup as a test oracle for the accelerated setup.

Accelerated setup pieces dispatch only inside the FDTD preparation scope
(``compiler.accel.dispatch.setup_enabled``). This oracle builds the same solver
without initialization and runs its ``init_field`` outside that scope, so the
material compile, sample accumulation, pole edges and coefficient composition all
take their torch implementations, then finishes the remaining preparation steps
of ``Simulation`` unchanged, including core-path activation for an admitted scene.
It never changes how ``Simulation.prepare`` executes.
"""
from __future__ import annotations

import witwin.maxwell as mw
from witwin.maxwell.compiler.accel.dispatch import requires_grad_scope, setup_enabled
from witwin.maxwell.fdtd.runtime.core_path import activate_core_path
from witwin.maxwell.simulation import PreparedSimulation
from witwin.maxwell.simulation import MODE_SOURCE_GAUGES


def torch_setup_solver(simulation: mw.Simulation):
    """Return ``simulation``'s initialized FDTD solver built by the torch setup."""
    with requires_grad_scope():
        simulation._refresh_scene()
        solver = simulation._build_fdtd_solver(initialize=False)
        token = setup_enabled.set(False)
        try:
            solver.init_field()
        finally:
            setup_enabled.reset(token)
        owner = simulation.scene_module if simulation.scene_module is not None else simulation.scene
        MODE_SOURCE_GAUGES[owner] = {name: state["gauge"]
                                     for name, state in solver._mode_source_vector_state.items() if "gauge" in state}
        simulation._remember_fdtd_solver_time(solver)
        simulation._apply_initial_condition(solver)
        simulation._resolve_fdtd_time_steps(solver, simulation.scene)
        activate_core_path(solver)
    return solver


def torch_setup_prepare(simulation: mw.Simulation) -> PreparedSimulation:
    """A prepared simulation around :func:`torch_setup_solver` (for setup-dependent resume checks)."""
    return PreparedSimulation(simulation, torch_setup_solver(simulation))
