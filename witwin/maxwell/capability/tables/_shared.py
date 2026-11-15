"""Rows and test references several method tables share."""
from __future__ import annotations

from ..keys import BOUNDARY_KINDS, MONITOR_TYPES, PORT_TYPES, SOURCE_TYPES
from ..model import inherent, unsupported

REGISTRY_TEST = "tests/api/public/test_capability_registry.py::test_refused_rows_are_reported"
OPTICAL_TEST = "tests/core/scene/test_construction_fixed_inputs.py::test_optical_structure_assignment_is_refused_by_every_solver"
LAYERED_TEST = "tests/mom/layered/test_layered_public.py::test_public_layered_guards"
CORE_SURFACE_TESTS = ("tests/api/public/test_field_manifest.py::test_grid_solvers_refuse_core_surface_fields",
                      REGISTRY_TEST)
PLANAR_PEC_TEST = "tests/mom/layered/test_layered_guards.py::test_planar_pec_requires_fdtd_or_layered_mom"
CONDUCTOR_LOSS_TEST = ("tests/gradients/test_layered_only_gradient_fields.py::"
                       "test_non_layered_paths_refuse_trainable_conductor_loss_fields")

USE_FDTD = "use Simulation.fdtd(...)"

ALL_BOUNDARIES = tuple(f"boundary.{kind}" for kind in BOUNDARY_KINDS)
ALL_SOURCES = tuple(f"source.{name}" for name in SOURCE_TYPES)
ALL_MONITORS = tuple(f"monitor.{name}" for name in MONITOR_TYPES)
ALL_PORTS = tuple(f"port.{name}" for name in PORT_TYPES)


def optical_assignment():
    return unsupported(
        "structure.optical_assignment",
        "{owner} declares an optical phase_screen or uv/face_uv assignment; no Maxwell solver "
        "(FDTD, FDFD, MoM, electrostatics) consumes it",
        "set phase_screen=None and uv=face_uv=None",
        tests=(OPTICAL_TEST, "tests/core/scene/test_construction_fixed_inputs.py::"
                             "test_optical_assignment_added_by_update_is_refused"))


def homogeneous_background_or_plane_modes(key: str):
    return unsupported(
        key, "Homogeneous background declarations and physical Maxwell modes require the dedicated MoM dyadic runtime",
        "use Simulation.mom(...)", tests=(REGISTRY_TEST,))


def layer_stack_requires_layered_mom():
    return unsupported("background.layer_stack",
                       "A LayerStack background requires Simulation.mom(...) with MoMConfig.layered",
                       tests=(LAYERED_TEST, REGISTRY_TEST))


def planar_pec_requires_sheet_runtime():
    return unsupported("geometry.planar_pec",
                       "PlanarPEC requires the FDTD sheet runtime or MoM with a LayerStack background",
                       tests=(PLANAR_PEC_TEST, REGISTRY_TEST))


def conductor_loss_gradients_require_layered_mom():
    return unsupported("structure.trainable_conductor_loss",
                       "Gradients for {fields} are supported on layered MoM only (Simulation.mom with a LayerStack "
                       "background); this method reads these fields as fixed values",
                       "pass a float or a detached tensor", tests=(CONDUCTOR_LOSS_TEST, REGISTRY_TEST))


def coordinates_require_fdtd():
    return unsupported("coordinates", "BendCoordinates currently requires native FDTD", USE_FDTD,
                       tests=(REGISTRY_TEST,))


def core_surface_fields(solver: str, *, tests: tuple[str, ...]):
    return unsupported("material.core_surface_fields",
                       f"core material field(s) {{fields}} are not modelled by {solver}",
                       "use a Maxwell material model (Material, LossyMetalMedium, CopperFoil, ...)", tests=tests)


def breakdown_requires_fdtd(*, tests: tuple[str, ...]):
    return inherent("material.breakdown",
                    "DielectricBreakdown is a time-domain feedback model (a latching conductivity update during "
                    "time stepping) defined only for FDTD", "set breakdown=None, or " + USE_FDTD, tests=tests)


def thin_wires_require_fdtd():
    return unsupported("thin_wire", "ThinWire is an FDTD-only subgrid model", USE_FDTD, tests=(REGISTRY_TEST,))


def networks_require_fdtd():
    return unsupported("network", "Embedded network feedback is defined only for the time-domain FDTD update; "
                       "frequency-domain solvers cannot ignore Scene.networks", USE_FDTD, tests=(REGISTRY_TEST,))


def in_simulation_monitors(names):
    return tuple(
        unsupported(f"monitor.{name}", "{owner}: projection, Gaussian-overlap, surface field-time and auxiliary "
                    "field-time monitors are recorded by the FDTD runtime", USE_FDTD, tests=(REGISTRY_TEST,))
        for name in names)


def runtime_selectors():
    return unsupported(
        "mom_config.runtime_selectors",
        "{owner} selects more than one MoM runtime ({selected}); prepare() would run only the {first} runtime "
        "and ignore the others",
        "set exactly one of MoMConfig.layered, linear_coupled and harmonic_balance",
        tests=("tests/api/public/test_capability_registry.py::test_mom_config_refuses_two_runtime_selectors",))
