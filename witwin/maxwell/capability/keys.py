"""Feature keys of a scene: one pass over its declarations, with no compilation.

A key names one fact the method tables decide on. Collection keys (``"source"``,
``"lumped_element"``) are emitted once per non-empty Scene collection; dotted
keys refine them (``"source.PlaneWave"``, ``"source.injection_tfsf"``) or
describe one structure, material, boundary kind or configuration. Type keys
carry their class lineage, so a subclass inherits its base class's row.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import torch

SOURCE_TYPES = (
    "PointDipole", "PlaneWave", "GaussianBeam", "AstigmaticGaussianBeam", "ModeSource", "UniformCurrentSource",
    "CustomCurrentSource", "CustomFieldSource", "MaxwellPlaneModes", "LumpedCurrentSource", "ESDCurrentSource",
    "ESDVoltageSource",
)
MONITOR_TYPES = (
    "PointMonitor", "WireMonitor", "PlaneMonitor", "FluxMonitor", "IncidentPowerDensityMonitor",
    "FinitePlaneMonitor", "ClosedSurfaceMonitor", "PermittivityMonitor", "MediumMonitor", "PowerLossMonitor",
    "BreakdownMonitor", "ComponentStressMonitor", "DipoleEmissionMonitor", "FieldTimeMonitor", "FluxTimeMonitor",
    "ModeMonitor", "DiffractionMonitor", "AngleProjectionMonitor", "CartesianProjectionMonitor",
    "KSpaceProjectionMonitor", "GaussianOverlapMonitor", "AstigmaticGaussianOverlapMonitor",
    "SurfaceFieldTimeMonitor", "AuxFieldTimeMonitor",
)
# Monitors recorded inside the FDTD time loop (monitors.IN_SIMULATION_MONITOR_TYPES).
IN_SIMULATION_MONITOR_TYPES = (
    "AngleProjectionMonitor", "CartesianProjectionMonitor", "KSpaceProjectionMonitor", "GaussianOverlapMonitor",
    "AstigmaticGaussianOverlapMonitor", "SurfaceFieldTimeMonitor", "AuxFieldTimeMonitor",
)
PORT_TYPES = ("LumpedPort", "TerminalPort", "WavePort", "TerminalWavePort", "ModePort", "CoaxialLumpedPort")
BOUNDARY_KINDS = ("none", "pml", "periodic", "bloch", "pec", "pmc", "mur", "mode_abc")

# Scene attribute -> collection key, in report order.
COLLECTIONS = (
    ("sources", "source"), ("monitors", "monitor"), ("ports", "port"), ("lumped_elements", "lumped_element"),
    ("networks", "network"), ("circuits", "circuit"), ("thin_wires", "thin_wire"),
    ("internal_absorbers", "internal_absorber"), ("electrostatic_terminals", "electrostatic_terminal"),
    ("charge_densities", "charge_density"),
)
SYMMETRY_MONITOR_KEYS = (
    "symmetry.projection_monitor", "symmetry.closed_surface_auto_grid", "symmetry.closed_surface_image_plane",
    "symmetry.diffraction_folded_axis", "symmetry.finite_plane_image_plane", "symmetry.mode_plane_image_plane",
)
MATERIAL_KEYS = (
    "material.lossy_metal", "material.surface_impedance", "material.copper_foil", "material.medium2d",
    "material.gyromagnetic", "material.nonlinear", "material.full_tensor", "material.modulation", "material.pmc",
    "material.sigma_e", "material.core_surface_fields", "material.trainable_mu_r", "material.breakdown",
    "material.perturbation",
)
# Every scene key a method table must decide (directly, or through a refused prefix).
SCENE_KEYS = (
    "background.homogeneous", "background.layer_stack", "coordinates", "symmetry", *SYMMETRY_MONITOR_KEYS,
    "subpixel.pec_conformal", *(f"boundary.{kind}" for kind in BOUNDARY_KINDS),
    "structure.optical_assignment", "structure.trainable_conductor_loss", "geometry.surface_mesh",
    "geometry.planar_pec", *MATERIAL_KEYS,
    *(key for _, key in COLLECTIONS),
    *(f"source.{name}" for name in SOURCE_TYPES), "source.injection_tfsf", "source.injection_axis",
    "source.cw_frequency_mismatch", "source.MaxwellPlaneModes.trainable",
    *(f"monitor.{name}" for name in MONITOR_TYPES),
    *(f"port.{name}" for name in PORT_TYPES), "port.termination", "port.wire_binding",
    "electrostatic_terminal.trainable_geometry", "electrostatic_terminal.floating",
)
# MoMConfig sub-configurations that each select a different runtime in Simulation.prepare().
MOM_RUNTIME_SELECTORS = ("layered", "linear_coupled", "harmonic_balance")
MOM_CONFIG_KEYS = ("mom_config.runtime_selectors",)


@dataclass(frozen=True)
class Feature:
    """One declared fact: ``key`` (and, for typed objects, base-class ``lineage`` keys)."""

    key: str
    owner: str
    details: dict = field(default_factory=dict)
    lineage: tuple[str, ...] = ()

    @property
    def lookup(self) -> tuple[str, ...]:
        return (self.key, *self.lineage)


def _typed(category: str, value, owner: str) -> Feature:
    names = [cls.__name__ for cls in type(value).__mro__ if cls is not object]
    return Feature(f"{category}.{names[0]}", owner, lineage=tuple(f"{category}.{name}" for name in names[1:]))


def _label(kind: str, index: int, value) -> str:
    name = getattr(value, "name", None)
    return f"{kind} {name!r}" if name else f"{kind} #{index}"


def _closed_surface_image_plane_axis(scene, monitor) -> str | None:
    """Axis whose image plane a closed surface touches, crosses, or snaps onto."""
    for index, (axis, entry) in enumerate(zip("xyz", scene.symmetry)):
        if entry is None:
            continue
        _mode, face = entry
        lower, upper = (float(value) for value in scene.domain.bounds[index])
        coords = scene.grid.axis_coords(axis)
        if coords is None:
            cell = float(getattr(scene.grid, f"d{axis}"))
        else:
            cell = float(coords[1] - coords[0]) if face == "low" else float(coords[-1] - coords[-2])
        # Faces snap to the nearest tangential-E Yee node, so a face within half a
        # cell of the image plane lands on it.
        surface_lower, surface_upper = monitor.bounds[index]
        if face == "low" and surface_lower <= lower + 0.5 * cell:
            return axis
        if face == "high" and surface_upper >= upper - 0.5 * cell:
            return axis
    return None


def _mode_plane_image_plane_axis(scene, aperture) -> str | None:
    """Axis whose image plane a mode-solver aperture touches, crosses, or snaps onto.

    The transverse mode solve has a zero-field edge and normalizes to unit power on
    the folded aperture, so an aperture bounded by an image plane is neither the PEC
    nor the PMC mirrored-domain mode. A mode plane lying on an image plane is refused
    as well.
    """
    normal = "xyz".index(aperture.normal_axis)
    for index, (axis, entry) in enumerate(zip("xyz", scene.symmetry)):
        if entry is None:
            continue
        face = entry[1]
        lower, upper = (float(value) for value in scene.domain.bounds[index])
        plane = lower if face == "low" else upper
        coords = None if scene.grid.is_auto else scene.grid.axis_coords(axis)
        if coords is not None:
            cell = float(coords[1] - coords[0]) if face == "low" else float(coords[-1] - coords[-2])
        else:
            cell = 0.0 if scene.grid.is_auto else float(getattr(scene.grid, f"d{axis}"))
        reach = 0.5 * cell + 1.0e-12 * max(upper - lower, 1.0)
        center, half = float(aperture.position[index]), 0.5 * float(aperture.size[index])
        if index == normal:
            if abs(center - plane) <= reach:
                return axis
        elif (center - half <= plane + reach) if face == "low" else (center + half >= plane - reach):
            return axis
    return None


def _material_features(material, owner: str) -> Iterator[Feature]:
    from .._core_material_fields import declared_core_surface_fields

    conductor = next((key for flag, key in (("is_lossy_metal", "material.lossy_metal"),
                                            ("is_surface_impedance", "material.surface_impedance"),
                                            ("is_copper_foil", "material.copper_foil"),
                                            ("is_medium2d", "material.medium2d"))
                      if bool(getattr(material, flag, False))), None)
    if conductor is not None:
        yield Feature(conductor, owner, {"material_type": type(material).__name__})
    for flag, key in (("is_gyromagnetic", "material.gyromagnetic"), ("is_nonlinear", "material.nonlinear"),
                      ("has_full_epsilon_tensor", "material.full_tensor")):
        if bool(getattr(material, flag, False)):
            yield Feature(key, owner)
    if getattr(material, "modulation", None) is not None:
        yield Feature("material.modulation", owner)
    if getattr(material, "pmc", False):
        yield Feature("material.pmc", owner)
    if getattr(material, "breakdown", None) is not None:
        yield Feature("material.breakdown", owner)
    if getattr(material, "perturbation", None) is not None:
        yield Feature("material.perturbation", owner)
    # A conductor model carries its conductivity by definition; sigma_e is a fact of the other media.
    sigma_e = getattr(material, "sigma_e", 0.0)
    if conductor is None and (getattr(material, "sigma_e_tensor", None) is not None
                              or bool(torch.as_tensor(sigma_e).detach() != 0)):
        yield Feature("material.sigma_e", owner)
    fields = declared_core_surface_fields(material)
    if fields:
        yield Feature("material.core_surface_fields", owner, {"fields": ", ".join(fields)})
    mu_r = getattr(material, "mu_r", None)
    if isinstance(mu_r, torch.Tensor) and mu_r.requires_grad:
        yield Feature("material.trainable_mu_r", owner)


def _trainable_conductor_loss_fields(structure) -> tuple[str, ...]:
    """Conductor-loss fields of ``structure`` that hold a requires_grad tensor.

    ``LossyMetalMedium.conductivity``, every ``SurfaceRoughness`` field and
    ``PlanarPEC.physical_thickness`` keep a trainable tensor for the layered MoM
    native adjoint; every other method reads them as fixed floats.
    """
    from ..geometry import PlanarPEC
    from ..media import LossyMetalMedium

    def trainable(value):
        return isinstance(value, torch.Tensor) and value.requires_grad

    fields = []
    material, geometry = structure.material, structure.geometry
    if isinstance(material, LossyMetalMedium):
        if trainable(material.conductivity):
            fields.append("LossyMetalMedium.conductivity")
        if material.roughness is not None:
            fields.extend(f"{type(material.roughness).__name__}.{name}"
                          for name, value in vars(material.roughness).items() if trainable(value))
    if isinstance(geometry, PlanarPEC) and trainable(geometry.physical_thickness):
        fields.append("PlanarPEC.physical_thickness")
    return tuple(dict.fromkeys(fields))


def _symmetry_monitor_feature(scene, monitor, owner: str) -> Feature | None:
    from ..fdtd.quadrature import finite_plane_image_planes, symmetry_image_planes
    from ..monitors import (
        PROJECTION_MONITOR_TYPES, ClosedSurfaceMonitor, DiffractionMonitor, FinitePlaneMonitor, ModeMonitor,
    )

    if isinstance(monitor, PROJECTION_MONITOR_TYPES):
        return Feature("symmetry.projection_monitor", owner)
    if isinstance(monitor, ClosedSurfaceMonitor):
        if scene.grid.is_auto:
            return Feature("symmetry.closed_surface_auto_grid", owner)
        axis = _closed_surface_image_plane_axis(scene, monitor)
        return None if axis is None else Feature("symmetry.closed_surface_image_plane", owner, {"axis": axis})
    if isinstance(monitor, DiffractionMonitor):
        folded = {axis for axis, entry in zip("xyz", scene.symmetry) if entry is not None}
        tangential = sorted(folded.intersection(monitor.tangential_axes))
        if tangential:
            return Feature("symmetry.diffraction_folded_axis", owner, {"axis": tangential[0]})
    if isinstance(monitor, ModeMonitor):
        axis = _mode_plane_image_plane_axis(scene, monitor)
        if axis is not None:
            return Feature("symmetry.mode_plane_image_plane", owner, {"axis": axis})
    if isinstance(monitor, FinitePlaneMonitor) and monitor.compute_flux:
        _, straddled = finite_plane_image_planes(symmetry_image_planes(scene, monitor.axis), monitor.tangential_bounds)
        if straddled:
            return Feature("symmetry.finite_plane_image_plane", owner, {"axis": straddled[0]})
    return None


def scene_features(scene, *, frequencies=(), config=None) -> Iterator[Feature]:
    """Every feature key ``scene`` (and ``config``) declares, in report order."""
    from witwin.core import PhysicalMaterial

    from ..compiler.sources import fdfd_cw_frequency_mismatch
    from ..layers import LayerStack
    from ..ports import ModePort, TerminalWavePort, WavePort
    from ..sources import TFSF, MaxwellPlaneModes, ModeSource, PointDipole, UniformCurrentSource

    for index, structure in enumerate(scene.structures):
        if structure.phase_screen is not None or structure.uv is not None:
            yield Feature("structure.optical_assignment", _label("Structure", index, structure))
    background = scene.background
    if isinstance(background, LayerStack):
        yield Feature("background.layer_stack", "Scene.background")
    elif isinstance(background, PhysicalMaterial):
        yield Feature("background.homogeneous", "Scene.background")
        yield from _material_features(background, "Scene.background")
    if getattr(scene, "coordinates", None) is not None:
        yield Feature("coordinates", "Scene.coordinates")
    symmetric = any(entry is not None for entry in scene.symmetry)
    if symmetric:
        yield Feature("symmetry", "Scene.symmetry")
    if scene.subpixel.pec == "conformal":
        yield Feature("subpixel.pec_conformal", 'SubpixelSpec(pec="conformal")')
    for kind in dict.fromkeys(scene.boundary.face_kind(axis, side) for axis in "xyz" for side in ("low", "high")):
        yield Feature(f"boundary.{kind}", f"Boundary faces [{kind!r}]", {"kind": kind})

    for index, structure in enumerate(scene.structures):
        if not structure.enabled:
            continue
        owner = _label("Structure", index, structure)
        kind = getattr(structure.geometry, "kind", None)
        if kind in ("surface_mesh", "planar_pec"):
            yield Feature(f"geometry.{kind}", owner)
        fields = _trainable_conductor_loss_fields(structure)
        if fields:
            yield Feature("structure.trainable_conductor_loss", owner, {"fields": ", ".join(fields)})
        if structure.material is not None:
            yield from _material_features(structure.material, owner)

    for attribute, key in COLLECTIONS:
        values = tuple(getattr(scene, attribute, ()))
        if values:
            names = ", ".join(repr(getattr(value, "name", None)) for value in values)
            yield Feature(key, f"Scene.{attribute} ({len(values)})", {"count": len(values), "names": names})

    for index, source in enumerate(scene.sources):
        owner = _label("Source", index, source)
        yield _typed("source", source, owner)
        if isinstance(getattr(source, "injection", None), TFSF):
            yield Feature("source.injection_tfsf", owner)
        if getattr(source, "injection_axis", None) is not None:
            yield Feature("source.injection_axis", owner)
        if isinstance(source, MaxwellPlaneModes) and any(
                value.requires_grad for value in (source.wavevectors, source.electric, source.magnetic, source.origin)):
            yield Feature("source.MaxwellPlaneModes.trainable", owner)
        if symmetric and isinstance(source, ModeSource):
            axis = _mode_plane_image_plane_axis(scene, source)
            if axis is not None:
                yield Feature("symmetry.mode_plane_image_plane", owner, {"axis": axis})
        if isinstance(source, (PointDipole, UniformCurrentSource)) and any(
                fdfd_cw_frequency_mismatch(source, frequency) for frequency in frequencies):
            yield Feature("source.cw_frequency_mismatch", owner, {
                "cw_frequency": float(source.source_time.frequency),
                "frequencies": tuple(float(frequency) for frequency in frequencies)})

    for monitor in scene.monitors:
        owner = f"Monitor {getattr(monitor, 'name', None)!r} ({type(monitor).__name__})"
        yield _typed("monitor", monitor, owner)
        if symmetric:
            folded = _symmetry_monitor_feature(scene, monitor, owner)
            if folded is not None:
                yield folded

    for port in scene.ports:
        owner = f"Port {port.name!r} ({type(port).__name__})"
        yield _typed("port", port, owner)
        if symmetric and isinstance(port, (ModePort, WavePort, TerminalWavePort)):
            axis = _mode_plane_image_plane_axis(scene, port)
            if axis is not None:
                yield Feature("symmetry.mode_plane_image_plane", owner, {"axis": axis})
        if getattr(port, "termination", None) is not None:
            yield Feature("port.termination", owner)
        if getattr(port, "wire_binding", None) is not None:
            yield Feature("port.wire_binding", owner)

    for terminal in getattr(scene, "electrostatic_terminals", ()):
        owner = f"Terminal {terminal.name!r}"
        geometry = getattr(terminal, "geometry", None)
        if any(isinstance(value, torch.Tensor) and value.requires_grad
               for value in getattr(geometry, "__dict__", {}).values()):
            yield Feature("electrostatic_terminal.trainable_geometry", owner)
        if getattr(terminal, "is_floating", False):
            yield Feature("electrostatic_terminal.floating", owner)

    from ..mom.runtime import MoMConfig
    if isinstance(config, MoMConfig):
        selected = tuple(name for name in MOM_RUNTIME_SELECTORS if getattr(config, name) is not None)
        if len(selected) > 1:
            yield Feature("mom_config.runtime_selectors",
                          "MoMConfig(" + ", ".join(f"{name}=..." for name in selected) + ")",
                          {"selected": ", ".join(selected), "first": selected[0]})
