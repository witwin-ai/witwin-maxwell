"""``Simulation.fdtd(...)``: the time-domain Yee runtime."""
from __future__ import annotations

from ..keys import COLLECTIONS, MATERIAL_KEYS
from ..model import supported, table, unsupported
from ._shared import (
    ALL_BOUNDARIES, ALL_MONITORS, ALL_PORTS, ALL_SOURCES, CORE_SURFACE_TESTS,
    conductor_loss_gradients_require_layered_mom, core_surface_fields,
    homogeneous_background_or_plane_modes, layer_stack_requires_layered_mom, optical_assignment,
)

_SYMMETRY = "tests/boundaries/symmetry/test_symmetry_generalization.py::"
_FULL_DOMAIN = "simulate the full domain without Scene.symmetry"

TABLE = table(
    homogeneous_background_or_plane_modes("background.homogeneous"),
    layer_stack_requires_layered_mom(),
    optical_assignment(),
    conductor_loss_gradients_require_layered_mom(),
    # Mapped (BendCoordinates) scenes are admitted here and checked by the coordinate preflight.
    supported("coordinates", "symmetry", "subpixel.pec_conformal", *ALL_BOUNDARIES,
              "geometry.surface_mesh", "geometry.planar_pec", *(key for key in MATERIAL_KEYS
                                                                if key != "material.core_surface_fields"),
              *(key for _, key in COLLECTIONS)),
    core_surface_fields("FDTD", tests=CORE_SURFACE_TESTS),
    unsupported("symmetry.projection_monitor",
                "{owner} cannot project from a Scene.symmetry domain: the folded domain holds only part of the "
                "Huygens surface and image currents are not expanded", _FULL_DOMAIN,
                tests=(_SYMMETRY + "test_projection_monitor_under_symmetry_is_refused_at_construction",)),
    unsupported("symmetry.closed_surface_auto_grid",
                "{owner} under Scene.symmetry needs a uniform or custom grid: on GridSpec.auto the face positions "
                "relative to the image plane are only known after meshing",
                tests=(_SYMMETRY + "test_closed_surface_under_symmetry_on_an_auto_grid_is_refused",)),
    unsupported("symmetry.closed_surface_image_plane",
                "{owner} touches or crosses the Scene.symmetry image plane on axis {axis!r}; a face on the image "
                "plane would sample tangential H without its image, and image-plane face integration is not "
                "available",
                "keep the surface inside the folded domain, more than half a cell from the image plane, or "
                + _FULL_DOMAIN,
                tests=(_SYMMETRY + "test_closed_surface_reaching_the_image_plane_is_refused",
                       _SYMMETRY + "test_closed_surface_reaching_a_high_face_image_plane_is_refused")),
    unsupported("symmetry.diffraction_folded_axis",
                "{owner} needs the full transverse period, but Scene.symmetry folds its tangential axis {axis!r}",
                "simulate the full unit cell without symmetry on that axis",
                tests=(_SYMMETRY + "test_diffraction_monitor_with_folded_tangential_axis_is_refused",)),
    unsupported("symmetry.finite_plane_image_plane",
                "{owner} crosses the Scene.symmetry image plane on axis {axis!r} without being symmetric about "
                "it, so its flux in the mirrored domain is not a multiple of the folded-domain flux",
                "center the rectangle on the image plane, keep it on the kept side, or " + _FULL_DOMAIN,
                tests=(_SYMMETRY + "test_finite_flux_plane_crossing_the_image_plane_asymmetrically_is_refused",)),
    unsupported("symmetry.mode_plane_image_plane",
                "{owner} has a mode plane that touches the Scene.symmetry image plane on axis {axis!r}; the transverse "
                "mode solver has no PEC/PMC symmetry edge and normalizes on the folded aperture, so the mode is not "
                "that of the mirrored domain",
                "keep the mode plane more than half a cell from every image plane, or " + _FULL_DOMAIN,
                tests=(_SYMMETRY + "test_mode_planes_touching_an_image_plane_are_refused",)),
    supported(*(key for key in ALL_SOURCES if key != "source.MaxwellPlaneModes")),
    homogeneous_background_or_plane_modes("source.MaxwellPlaneModes"),
    supported("source.injection_tfsf", "source.injection_axis", "source.cw_frequency_mismatch"),
    supported(*ALL_MONITORS, *ALL_PORTS, "port.termination", "port.wire_binding"),
    supported("electrostatic_terminal.trainable_geometry", "electrostatic_terminal.floating"),
)
