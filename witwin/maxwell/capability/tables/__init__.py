"""One capability table per method: feature key -> ``Capability`` row."""
from __future__ import annotations

from .capacitance import TABLE as CAPACITANCE
from .electrostatic import TABLE as ELECTROSTATIC
from .fdfd import TABLE as FDFD
from .fdtd import TABLE as FDTD
from .mom import TABLE as MOM
from .mom_layered import TABLE as MOM_LAYERED

TABLES = {
    "fdtd": FDTD,
    "fdfd": FDFD,
    "mom": MOM,
    "mom.layered": MOM_LAYERED,
    "electrostatic": ELECTROSTATIC,
    "capacitance": CAPACITANCE,
}
# The public entry point each method's refusal names.
ENTRY_POINTS = {
    "fdtd": "fdtd", "fdfd": "fdfd", "mom": "mom", "mom.layered": "mom",
    "electrostatic": "electrostatic", "capacitance": "capacitance",
}
