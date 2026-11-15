"""``Simulation.mom(...)`` with homogeneous bodies: the surface-integral runtimes.

The surface runtimes read the linear constitutive fields of each body, the
analytic incident field of each source and the unbounded exterior. The rows
below refuse what they would drop silently; the runtimes' own scene-scope
checks (bodies, boundaries, ports, source kinds) stay runtime guards.
"""
from __future__ import annotations

from ..keys import BOUNDARY_KINDS, COLLECTIONS, IN_SIMULATION_MONITOR_TYPES, MONITOR_TYPES
from ..model import inherent, supported, table, unsupported
from ._shared import (
    ALL_PORTS, ALL_SOURCES, breakdown_requires_fdtd, conductor_loss_gradients_require_layered_mom,
    coordinates_require_fdtd, in_simulation_monitors, layer_stack_requires_layered_mom, networks_require_fdtd,
    optical_assignment, planar_pec_requires_sheet_runtime, runtime_selectors, thin_wires_require_fdtd,
)

_MOM = "tests/mom/test_mom_silent_ignore_refusals.py::"

TABLE = table(
    supported("background.homogeneous"),
    layer_stack_requires_layered_mom(),
    coordinates_require_fdtd(),
    optical_assignment(),
    conductor_loss_gradients_require_layered_mom(),
    supported("symmetry", "symmetry.projection_monitor", "symmetry.closed_surface_auto_grid",
              "symmetry.closed_surface_image_plane", "symmetry.diffraction_folded_axis", "symmetry.finite_plane_image_plane", "symmetry.mode_plane_image_plane",
              "subpixel.pec_conformal",
              *(f"boundary.{kind}" for kind in BOUNDARY_KINDS), "geometry.surface_mesh"),
    planar_pec_requires_sheet_runtime(),
    supported("material.lossy_metal", "material.surface_impedance", "material.copper_foil", "material.medium2d",
              "material.gyromagnetic", "material.nonlinear", "material.full_tensor", "material.modulation",
              "material.sigma_e", "material.perturbation"),
    breakdown_requires_fdtd(tests=(_MOM + "test_breakdown_is_refused",)),
    unsupported("material.pmc", "Material(pmc=True) has no surface-integral model (it would be solved as vacuum)",
                tests=(_MOM + "test_pmc_body_is_refused_instead_of_solved_as_vacuum",)),
    unsupported("material.core_surface_fields", "core material field(s) {fields} are not modelled by MoM",
                tests=(_MOM + "test_core_surface_fields_are_refused",)),
    unsupported("material.trainable_mu_r",
                "trainable mu_r; MoM material derivatives are not supported and the solve would detach it",
                "give mu_r a fixed value", tests=(_MOM + "test_trainable_mu_r_is_refused_instead_of_detached",)),
    supported(*(key for _, key in COLLECTIONS if key not in ("network", "thin_wire", "internal_absorber"))),
    networks_require_fdtd(),
    thin_wires_require_fdtd(),
    inherent("internal_absorber", "the MoM exterior is unbounded and has no grid absorber",
             tests=(_MOM + "test_internal_absorbers_are_refused",)),
    supported(*ALL_SOURCES, "source.cw_frequency_mismatch"),
    inherent("source.injection_tfsf", "injection=TFSF(...) is an FDTD grid injection; MoM uses the analytic "
             "incident field everywhere", tests=(_MOM + "test_tfsf_and_injection_axis_are_refused",)),
    inherent("source.injection_axis", "injection_axis is an FDTD grid-injection setting",
             tests=(_MOM + "test_tfsf_and_injection_axis_are_refused",)),
    unsupported("source.MaxwellPlaneModes.trainable",
                "trainable MaxwellPlaneModes tensors; the MoM solve would detach them", "give them fixed values",
                tests=(_MOM + "test_trainable_maxwell_plane_modes_are_refused",)),
    supported(*(f"monitor.{name}" for name in MONITOR_TYPES if name not in IN_SIMULATION_MONITOR_TYPES)),
    in_simulation_monitors(IN_SIMULATION_MONITOR_TYPES),
    supported(*ALL_PORTS, "port.termination", "port.wire_binding"),
    supported("electrostatic_terminal.trainable_geometry", "electrostatic_terminal.floating"),
    runtime_selectors(),
)
