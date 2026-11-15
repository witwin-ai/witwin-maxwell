"""``Simulation.mom(...)`` with a LayerStack background: the layered MPIE network runtime.

The layered runtime solves PlanarPEC sheets and Box vias in the LayerStack as
an N-port network driven by its LumpedPorts, and differentiates the
conductor-loss fields (conductivity, roughness, sheet thickness) through its
native adjoint. The rows mirror the homogeneous MoM table except where the
layered runtime differs; the refusals below are the scene inputs its compiler
(``mom.layered.runtime.unsupported_layered_inputs``) names, so a refused input
is reported at construction. That compiler still checks every input at
``prepare()``, including the ones without a registry key (material regions and
the LumpedPort reference plane and current surface). A LumpedPort voltage
aperture is read: it makes the layered edge port (``mom.layered.port_system``).
"""
from __future__ import annotations

from ..keys import BOUNDARY_KINDS, PORT_TYPES
from ..model import override, supported, unsupported
from ._shared import REGISTRY_TEST
from .mom import TABLE as MOM

LAYERED_REFUSALS_TEST = ("tests/mom/layered/test_layered_refusals.py::"
                         "test_public_construction_refuses_through_the_registry")
_TESTS = (LAYERED_REFUSALS_TEST, REGISTRY_TEST)
_UNBOUNDED = "the layered Green function is laterally unbounded"


# Collections the layered table refuses whole; their member rows (source types, monitor types, folded
# symmetry monitors, terminal flags) are decided by the refused collection and are not inherited.
_REFUSED_COLLECTIONS = ("source", "monitor", "symmetry", "electrostatic_terminal")


def _refused(key, reason, alternative=None):
    return unsupported(key, reason, alternative, tests=_TESTS)


TABLE = override(
    {key: row for key, row in MOM.items() if key.partition(".")[0] not in _REFUSED_COLLECTIONS or "." not in key},
    unsupported("background.homogeneous", "A homogeneous background cannot accompany the LayerStack background",
                tests=(REGISTRY_TEST,)),
    supported("background.layer_stack", "geometry.planar_pec", "structure.trainable_conductor_loss", "material.pmc",
              "material.core_surface_fields", "material.trainable_mu_r"),
    _refused("source", "the layered network is driven by its LumpedPorts only",
             "read the response from Result.network"),
    _refused("monitor", "the layered runtime records no monitors",
             "results are Result.network and mw.mom.surface_currents(result)"),
    _refused("lumped_element", "the layered runtime does not apply lumped elements",
             "load ports with Result.network.terminate(port, impedance=load)"),
    _refused("circuit", "the layered runtime does not apply circuits",
             "connect circuits to Result.network with NetworkData algebra"),
    _refused("electrostatic_terminal", "the layered runtime does not apply electrostatic terminals",
             "use Simulation.electrostatic(...) or Simulation.capacitance(...)"),
    _refused("charge_density", "the layered runtime does not apply charge densities",
             "use Simulation.electrostatic(...)"),
    _refused("internal_absorber", _UNBOUNDED + "; it has no grid absorber"),
    _refused("symmetry", "the layered solve has no symmetry reduction"),
    tuple(_refused(f"boundary.{kind}", _UNBOUNDED, "use BoundarySpec.none()")
          for kind in BOUNDARY_KINDS if kind != "none"),
    tuple(_refused(f"port.{name}", "layered ports are LumpedPort delta gaps")
          for name in PORT_TYPES if name != "LumpedPort"),
    _refused("port.termination", "the layered delta-gap port does not apply a termination",
             "Result.network is the open N-port; close it with Result.network.terminate(port, impedance=load) "
             "(exp(-i omega t): an inductor load is -1j*omega*L)"),
    _refused("port.wire_binding", "the layered delta-gap port has no wire binding",
             "layered ports need positive/negative point terminals on PlanarPEC metal"),
)
