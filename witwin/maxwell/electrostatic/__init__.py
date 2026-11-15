from .api import (
    ChargeDensity,
    ElectrostaticBoundarySpec,
    ElectrostaticConfig,
    ElectrostaticSolverConfig,
    ElectrostaticTerminal,
)
from .capacitance import (
    CapacitanceConfig,
    CapacitanceData,
    TruncationEstimate,
    TruncationReport,
    extract_capacitance,
)
from .runtime import (
    ElectrostaticResultData,
    solve_electrostatics,
)
from .initial_condition import (
    DEFAULT_GAUSS_TOLERANCE,
    ElectrostaticInitialCondition,
)

from .._stability import experimental


@experimental("electrostatics")
def electrostatic(result) -> ElectrostaticResultData:
    """The typed solver output of a ``Simulation.electrostatic(...)`` ``result``."""
    return result._electrostatic()


@experimental("electrostatics")
def capacitance(result) -> CapacitanceData:
    """The typed capacitance matrix of a ``Simulation.capacitance(...)`` ``result``."""
    return result._capacitance()


@experimental("electrostatics")
def electrostatic_prebias(result):
    """Provenance of the electrostatic pre-bias initial condition of an FDTD ``result``, or ``None``.

    Set when the run was seeded with an :class:`ElectrostaticInitialCondition`
    (``FDTDConfig(initial_condition=...)``): the DC solve diagnostics, the mapped-field
    discrete-Gauss residual and the tolerance.
    """
    return (result.metadata or {}).get("electrostatic_prebias")


# Public namespace ``mw.electrostatic``; the solver entry is ``Simulation.electrostatic`` /
# ``Simulation.capacitance``.
__all__ = [
    "CapacitanceConfig",
    "CapacitanceData",
    "ChargeDensity",
    "ElectrostaticBoundarySpec",
    "ElectrostaticConfig",
    "ElectrostaticInitialCondition",
    "ElectrostaticResultData",
    "ElectrostaticSolverConfig",
    "ElectrostaticTerminal",
    "TruncationEstimate",
    "TruncationReport",
    "capacitance",
    "electrostatic",
    "electrostatic_prebias",
]
