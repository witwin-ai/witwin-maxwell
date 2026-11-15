"""``Simulation.capacitance(...)``: N-terminal capacitance extraction.

Every row of the electrostatic table applies. Extraction drives each conductor
to a set potential and measures the pure conductor response, so it also
refuses floating terminals and free charge.
"""
from __future__ import annotations

from ..model import inherent, override, unsupported
from .electrostatic import TABLE as ELECTROSTATIC

_TEST = "tests/electrostatic/test_silent_ignore_refusals.py::test_capacitance_refuses_floating_terminals_and_free_charge"

TABLE = override(
    ELECTROSTATIC,
    unsupported("electrostatic_terminal.floating",
                "floating (charge=) conductor; capacitance extraction would ground it at 0 V",
                "declare it grounded=True or potential=, or solve it with Simulation.electrostatic(...)",
                tests=(_TEST,)),
    inherent("charge_density", "capacitance measures the pure conductor response and excludes free charge",
             "use Simulation.electrostatic(...)", tests=(_TEST,)),
)
