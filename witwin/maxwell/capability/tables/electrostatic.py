"""``Simulation.electrostatic(...)``: the Laplace/Poisson finite-volume solve.

The solve reads the bulk ``Structure`` permittivities, the
``ElectrostaticTerminal`` conductors, ``Scene.charge_densities`` and the
``ElectrostaticBoundarySpec`` of its configuration; every full-wave declaration,
a full-wave ``Scene.boundary`` face included, is refused.
"""
from __future__ import annotations

from ..keys import BOUNDARY_KINDS
from ..model import inherent, supported, table, unsupported
from ._shared import breakdown_requires_fdtd, conductor_loss_gradients_require_layered_mom, optical_assignment

_ES = "tests/electrostatic/test_silent_ignore_refusals.py::"
_READS = ("the electrostatic solve reads only Structure permittivities, ElectrostaticTerminal conductors, "
          "Scene.charge_densities and its ElectrostaticBoundarySpec")
_COLLECTION_TESTS = {
    "source": "test_full_wave_collections_are_refused", "monitor": "test_full_wave_collections_are_refused",
    "port": "test_full_wave_collections_are_refused", "lumped_element": "test_full_wave_collections_are_refused",
    "internal_absorber": "test_full_wave_collections_are_refused",
    "network": "test_network_and_wire_collections_are_refused",
    "thin_wire": "test_network_and_wire_collections_are_refused",
    "circuit": "test_circuits_symmetry_and_coordinates_are_refused",
}
_TERMINAL = "declare the conductor as an ElectrostaticTerminal"
_DROPPED = _ES + "test_dropped_material_fields_are_refused"


def _conductor(key):
    return unsupported(key, "{material_type} is a conductor model", _TERMINAL,
                       tests=(_ES + "test_conductor_media_are_refused",))


TABLE = table(
    # Scene.background is not rasterized by the electrostatic compiler (admitted, not read).
    supported("background.homogeneous", "background.layer_stack", "subpixel.pec_conformal", "boundary.none",
              "geometry.surface_mesh", "geometry.planar_pec"),
    tuple(inherent(f"boundary.{kind}", "{owner} of Scene.boundary: the electrostatic solve takes its faces from "
                   "ElectrostaticBoundarySpec and needs Scene.boundary = BoundarySpec.none() (absorbing or periodic "
                   "padding would extend the grid)",
                   "set Scene.boundary = BoundarySpec.none() and the faces with ElectrostaticConfig(boundary=...) "
                   "or CapacitanceConfig(boundary=...)",
                   tests=(_ES + "test_full_wave_boundary_is_refused",))
          for kind in BOUNDARY_KINDS if kind != "none"),
    unsupported("coordinates", "the electrostatic solve has no mapped-coordinate operator",
                tests=(_ES + "test_circuits_symmetry_and_coordinates_are_refused",)),
    unsupported("symmetry", "the electrostatic solve takes its symmetry faces from ElectrostaticBoundarySpec",
                "use ElectrostaticBoundarySpec symmetry faces",
                tests=(_ES + "test_circuits_symmetry_and_coordinates_are_refused",)),
    optical_assignment(),
    conductor_loss_gradients_require_layered_mom(),
    _conductor("material.lossy_metal"),
    _conductor("material.surface_impedance"),
    _conductor("material.copper_foil"),
    _conductor("material.medium2d"),
    inherent("material.sigma_e", "sigma_e (electric conductivity has no electrostatic meaning)",
             "declare a conductor as an ElectrostaticTerminal", tests=(_ES + "test_conductivity_is_refused",)),
    unsupported("material.core_surface_fields", "core material field(s) {fields} are not modelled by the "
                "electrostatic solve", tests=(_ES + "test_core_surface_fields_are_refused",)),
    unsupported("material.nonlinear",
                "nonlinear susceptibility (kerr_chi3 or nonlinearity); the linear static solve would drop the "
                "field-dependent permittivity", "set kerr_chi3=None and nonlinearity=() for the static solve",
                tests=(_DROPPED,)),
    breakdown_requires_fdtd(tests=(_DROPPED,)),
    unsupported("material.perturbation",
                "PerturbationMedium; the static solve evaluates only the base permittivity and would drop the "
                "perturbation shift", "declare the perturbed permittivity as Structure materials",
                tests=(_DROPPED,)),
    supported("material.gyromagnetic", "material.full_tensor", "material.modulation", "material.pmc",
              "material.trainable_mu_r"),
    tuple(inherent(key, _READS, tests=(_ES + test,)) for key, test in _COLLECTION_TESTS.items()),
    supported("electrostatic_terminal", "charge_density", "electrostatic_terminal.floating"),
    unsupported("electrostatic_terminal.trainable_geometry",
                "trainable geometry (terminals are rasterized as a hard cell mask, so the gradient would be dropped)",
                "give the terminal geometry fixed values",
                tests=(_ES + "test_trainable_terminal_geometry_is_refused",)),
)
