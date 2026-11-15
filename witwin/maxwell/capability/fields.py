"""Field-consumption manifest: how each method treats every public scene field.

Every public class a ``Scene`` can hold has one entry, and the entry declares
each constructor field once. The test suite fails on an undeclared field, so a
new field must state how every method treats it before it can ship.

A field is declared as one of:

* owned: part of the object's definition; a method that admits the object reads
  it (or refuses a configuration of it in a runtime guard, never silently).
* keyed: a non-default value emits one of the registry ``keys``; each method
  admits or refuses it through its table row.
* ``when``: further registry keys that some values emit (a folded symmetry
  monitor, a full tensor, a trainable tensor); refused ones are listed per cell.
* not read: the method admits the object and deliberately does not read the
  field, because it has no meaning there (``INHERENT``, with a reason).
* gap: the method admits the object and drops the field although it has a
  meaning there. Gaps are open defects; the test pins the exact list.
* neutral: identity, provenance or bookkeeping (names, tags, metadata, device
  placement). Renaming or changing it never changes a solve.

``owner`` is the registry key that admits a declaration of the class (a source
type, a material model, a collection). ``within`` names the fields that hold a
nested class; a method that reads none of them never sees the nested value.
``consumption(cls)`` resolves the per-method cells in the registry vocabulary:
``SUPPORTED`` (read), ``UNSUPPORTED``/``INHERENT`` (refused through a registry
row, or not read when no key is named) and ``None`` for a neutral field.
"""
from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import witwin.maxwell as mw
from witwin.core import PhysicalMaterial

from . import METHODS
from .model import Capability, Support
from .tables import TABLES

FDTD, FDFD, MOM, MOM_LAYERED, ELECTROSTATIC, CAPACITANCE = (
    "fdtd", "fdfd", "mom", "mom.layered", "electrostatic", "capacitance")
SURFACE = (MOM, MOM_LAYERED)
STATIC = (ELECTROSTATIC, CAPACITANCE)
GRID_FULL_WAVE = (FDTD, FDFD)


@dataclass(frozen=True)
class FieldUse:
    """How the methods treat one field (see the module docstring)."""

    keys: tuple[str, ...] = ()
    when: tuple[str, ...] = ()
    not_read: Mapping[str, str] = field(default_factory=dict)
    gaps: Mapping[str, str] = field(default_factory=dict)
    neutral: bool = False


@dataclass(frozen=True)
class SceneClass:
    """One public scene class: its admitting key, containers and fields."""

    cls: type
    owner: str | None
    within: tuple[str, ...]
    fields: Mapping[str, FieldUse]


@dataclass(frozen=True)
class Cell:
    """How one method treats one field: ``support`` (``None`` when neutral),
    the registry ``key`` that decides it (``None`` for a not-read field or a
    gap), the reason, and the keys that refuse particular values."""

    support: Support | None
    key: str | None = None
    reason: str = ""
    conditional: tuple[str, ...] = ()
    gap: bool = False


_OWNED = FieldUse()
_NEUTRAL = FieldUse(neutral=True)


def _for(methods, reason: str) -> dict[str, str]:
    return {method: reason for method in ((methods,) if isinstance(methods, str) else methods)}


def owned(*names: str) -> dict[str, FieldUse]:
    return {name: _OWNED for name in names}


def neutral(*names: str) -> dict[str, FieldUse]:
    return {name: _NEUTRAL for name in names}


def use(name: str, *, keys=(), when=(), not_read=None, gaps=None) -> dict[str, FieldUse]:
    return {name: FieldUse(tuple(keys), tuple(when), dict(not_read or {}), dict(gaps or {}))}


def declared_fields(cls) -> tuple[str, ...]:
    """The public fields of ``cls``: its constructor parameters (``*args`` included, ``**kwargs`` not)."""
    parameters = tuple(inspect.signature(cls).parameters.values())
    if dataclasses.is_dataclass(cls) and all(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in parameters):
        # A generic (*args, **kwargs) constructor: the dataclass fields are the declaration.
        return tuple(f.name for f in dataclasses.fields(cls) if f.init and not f.name.startswith("_"))
    return tuple(p.name for p in parameters if p.kind is not p.VAR_KEYWORD)


def _loss_gradient(*names: str) -> dict[str, FieldUse]:
    """Conductor-loss fields whose trainable tensors emit ``structure.trainable_conductor_loss``."""
    return {name: FieldUse(when=("structure.trainable_conductor_loss",)) for name in names}


def _entry(cls, *groups, owner=None, within=()) -> SceneClass:
    fields: dict[str, FieldUse] = {}
    for group in groups:
        for name, value in group.items():
            if name in fields:
                raise ValueError(f"{cls.__name__}.{name} is declared twice")
            fields[name] = value
    return SceneClass(cls, owner, tuple(within), MappingProxyType(fields))


_SOURCE_TIMES = tuple(f"{name}.source_time" for name in (
    "PointDipole", "PlaneWave", "GaussianBeam", "AstigmaticGaussianBeam", "ModeSource", "UniformCurrentSource",
    "CustomCurrentSource", "CustomFieldSource", "ModePort"))
_GEOMETRY_HOLDERS = ("Structure.geometry", "MaterialRegion.geometry", "MeshOverrideStructure.geometry",
                     "ElectrostaticTerminal.geometry", "ChargeDensity.geometry", "Union.geometries",
                     "Intersection.geometries", "Difference.base", "Difference.tools")
_MATERIAL_HOLDERS = ("Structure.material", "Scene.background", "PerturbationMedium.base")
_TENSOR_HOLDERS = ("Material.epsilon_tensor", "Material.mu_tensor", "Material.sigma_e_tensor")
_NO_GRID = "the surface-integral solve meshes the bodies and has an unbounded exterior; the volume grid has no meaning"
_NO_MAGNETIC = "permeability and magnetic loss have no electrostatic meaning"
_NO_SAR = "mass density feeds the SAR reducer, which integrates volume E-field samples on a grid"
# Layered MoM refuses every source (its network is driven by LumpedPorts), so only homogeneous MoM sees them.
_INJECTION_NOT_READ = _for(MOM, "the surface solve uses the analytic incident field everywhere")



def _source(cls, *groups):
    return _entry(cls, neutral("name"), *groups, owner=f"source.{cls.__name__}", within=("Scene.sources",))


def _monitor(cls, *names, labels=()):
    return _entry(cls, neutral("name", *labels), owned(*names), owner=f"monitor.{cls.__name__}",
                  within=("Scene.monitors",))


def _port(cls, *groups):
    return _entry(cls, neutral("name"), *groups, owner=f"port.{cls.__name__}", within=("Scene.ports",))


def _beam(*extra):
    return (owned("direction", "polarization", "beam_waist", "focus", *extra, "source_time"),
            use("injection", when=("source.injection_tfsf",), not_read=_INJECTION_NOT_READ),
            use("injection_axis", keys=("source.injection_axis",)))


def _projection(*names):
    return ("position", "size", *names, "frequencies", "origin", "normal_direction", "exclude_surfaces",
            "window_size", "sample_spacing")


def _lumped(cls, value):
    return _entry(cls, neutral("name"), owned("positive", "negative", value), owner="lumped_element",
                  within=("Scene.lumped_elements",))


def _device(cls, *names):
    return _entry(cls, neutral("name"), owned("positive", "negative", *names), within=("Circuit",))


_BOUNDARY_KEYS = tuple(f"boundary.{kind}" for kind in ("none", "pml", "periodic", "bloch", "pec", "pmc", "mur",
                                                       "mode_abc"))
_ABSORBER_NOT_READ = {**_for(SURFACE, "the surface solve's exterior is unbounded; absorber settings have no meaning"),
                      **_for(STATIC, "the electrostatic solve requires BoundarySpec.none() and takes its faces from "
                                     "ElectrostaticBoundarySpec")}
_FULL_WAVE_TERMINALS = _for(GRID_FULL_WAVE, "an electrostatic declaration: full-wave solves take conductors from "
                                            "Structures, and a scene may carry terminals for an electrostatic "
                                            "pre-bias solve")
_MATERIAL_SURFACE_KEYS = ("material.core_surface_fields",)

ENTRIES = (
    # Scene and its grid.
    _entry(
        mw.Scene,
        use("domain", not_read=_for(SURFACE, _NO_GRID)),
        use("grid", not_read=_for(SURFACE, _NO_GRID)),
        owned("boundary"),
        # Grid preparation refuses every background, so the grid solvers refuse it at runtime.
        use("background", keys=("background.homogeneous", "background.layer_stack")),
        owned("structures"),
        use("thin_wires", keys=("thin_wire",)),
        use("sources", keys=("source",)),
        use("monitors", keys=("monitor",)),
        use("ports", keys=("port",)),
        use("networks", keys=("network",)),
        use("lumped_elements", keys=("lumped_element",)),
        use("circuits", keys=("circuit",)),
        owned("material_regions"),
        use("electrostatic_terminals", keys=("electrostatic_terminal",), not_read=_FULL_WAVE_TERMINALS),
        use("charge_densities", keys=("charge_density",), not_read=_FULL_WAVE_TERMINALS),
        neutral("metadata"),
        use("subpixel", not_read={
            **_for(SURFACE, _NO_GRID),
            **_for(STATIC, "the electrostatic rasterizer blends soft geometry occupancy and has no subpixel "
                           "averaging")}),
        use("symmetry", keys=("symmetry",), when=("symmetry.projection_monitor", "symmetry.closed_surface_auto_grid",
                                                  "symmetry.closed_surface_image_plane",
                                                  "symmetry.diffraction_folded_axis",
                                                  "symmetry.finite_plane_image_plane",
                                                  "symmetry.mode_plane_image_plane")),
        use("coordinates", keys=("coordinates",)),
        use("internal_absorbers", keys=("internal_absorber",)),
    ),
    _entry(mw.Domain, owned("bounds"), within=("Scene.domain",)),
    _entry(mw.GridSpec, owned("dx", "dy", "dz", "x_coords", "y_coords", "z_coords", "min_steps_per_wavelength",
                              "wavelength", "max_ratio", "override_structures", "layer_refinement"),
           within=("Scene.grid",)),
    _entry(mw.grid.GridRefinement, owned("refinement_factor", "dl", "num_cells"),
           within=("LayerRefinementSpec.bounds_refinement", "LayerRefinementSpec.corner_refinement")),
    _entry(mw.grid.LayerRefinementSpec, owned(
        "min_cells", "axes", "axis", "bounds", "min_steps_along_axis", "bounds_snapping", "bounds_refinement",
        "corner_finder", "corner_snapping", "corner_refinement", "refinement_inside_sim_only",
        "dl_min_from_gap_width"), within=("GridSpec.layer_refinement",)),
    _entry(mw.grid.CornerFinderSpec, owned("medium", "angle_threshold", "distance_threshold", "concave_resolution",
                                      "convex_resolution", "mixed_resolution"),
           within=("LayerRefinementSpec.corner_finder",)),
    _entry(mw.grid.MeshOverrideStructure, owned("geometry", "dl"), within=("GridSpec.override_structures",)),
    _entry(
        mw.BoundarySpec,
        *(use(name, when=_BOUNDARY_KEYS) for name in ("kind", "x", "y", "z", "x_low", "x_high", "y_low", "y_high",
                                                       "z_low", "z_high")),
        use("num_layers", not_read=_ABSORBER_NOT_READ),
        use("strength", not_read=_ABSORBER_NOT_READ),
        use("bloch_wavevector", when=("boundary.bloch",)),
        use("mur_abc", when=("boundary.mur",)),
        use("mode_abc", when=("boundary.mode_abc",)),
        within=("Scene.boundary",)),
    _entry(mw.fdtd.MurABC, owned("order", "permittivity"), within=("BoundarySpec.mur_abc",)),
    _entry(mw.fdtd.ModeABC, owned("mode_index", "polarization", "frequency", "frequency_range", "order"),
           within=("BoundarySpec.mode_abc",)),
    _entry(mw.SubpixelSpec, owned("samples", "averaging"), use("pec", when=("subpixel.pec_conformal",)),
           within=("Scene.subpixel",)),
    _entry(mw.coordinates.BendCoordinates, owned("curvature", "origin", "frame"), neutral("device"), owner="coordinates",
           within=("Scene.coordinates",)),
    _entry(mw.mom.LayerStack, owned("layers", "bottom", "top"), owner="background.layer_stack",
           within=("Scene.background",)),
    _entry(mw.mom.Layer, neutral("name"), owned("thickness", "eps_r", "sigma", "mu_r"), within=("LayerStack.layers",)),
    _entry(mw.mom.HalfSpace, owned("eps_r", "sigma", "mu_r"), within=("LayerStack.bottom", "LayerStack.top")),
    # Frequency-dependent layer permittivity models (layered MoM evaluates eps(f) per table frequency).
    _entry(mw.media.WidebandDebye, owned("eps_inf", "delta_eps", "f1", "f2"), within=("Layer.eps_r", "HalfSpace.eps_r")),
    _entry(mw.media.PolePermittivity, owned("eps_inf", "poles"), within=("Layer.eps_r", "HalfSpace.eps_r")),

    # Structures and geometry.
    _entry(
        mw.Structure,
        owned("geometry", "material", "enabled"),
        use("priority", not_read=_for(SURFACE, "the surface solve requires disjoint bodies, so overlap priority has "
                                               "no meaning")),
        neutral("name", "tags", "metadata", "structure_id", "material_id", "assignment_id", "surface_id",
                "primitive_ids"),
        *(use(name, keys=("structure.optical_assignment",)) for name in ("phase_screen", "uv", "face_uv")),
        within=("Scene.structures",)),
    _entry(mw.MaterialRegion, neutral("name"),
           owned("geometry", "density", "basis", "bounds", "eps_bounds", "mu_bounds", "filter_radius",
                 "projection_beta", "symmetry"), within=("Scene.material_regions",)),
    *(_entry(cls, owned("position", *names, "rotation"), neutral("device"), within=_GEOMETRY_HOLDERS)
      for cls, names in ((mw.Box, ("size",)), (mw.Sphere, ("radius",)), (mw.Cylinder, ("radius", "height", "axis")),
                         (mw.geometry.Cone, ("radius", "height", "axis")), (mw.geometry.Ellipsoid, ("radii",)),
                         (mw.geometry.HollowBox, ("outer_size", "inner_size")),
                         (mw.geometry.Prism, ("radius", "height", "num_sides", "axis")),
                         (mw.geometry.ExtrudedPolygon, ("vertices", "height", "axis")),
                         (mw.geometry.Pyramid, ("base_size", "height", "axis")),
                         (mw.geometry.Torus, ("major_radius", "minor_radius", "axis")))),
    _entry(mw.Mesh, owned("vertices", "faces", "position", "scale", "rotation", "recenter", "fill_mode",
                          "surface_thickness"),
           neutral("source_path", "topology_diagnostics", "device"), within=_GEOMETRY_HOLDERS),
    _entry(mw.geometry.Union, owned("geometries", "smoothness", "position", "rotation"), neutral("device"),
           within=_GEOMETRY_HOLDERS),
    _entry(mw.geometry.Intersection, owned("geometries", "smoothness", "position", "rotation"), neutral("device"),
           within=_GEOMETRY_HOLDERS),
    _entry(mw.geometry.Difference, owned("base", "tools", "smoothness", "position", "rotation"), neutral("device"),
           within=_GEOMETRY_HOLDERS),
    _entry(mw.PolySlab, owned("vertices", "bounds", "axis", "sidewall_angle", "reference_plane", "position",
                              "rotation"), neutral("device"), within=_GEOMETRY_HOLDERS),
    _entry(mw.geometry.ComplexPolySlab, owned("loops", "bounds", "axis", "sidewall_angle", "reference_plane", "position",
                                     "rotation"), neutral("device"), within=_GEOMETRY_HOLDERS),
    # physical_thickness sets the layered MoM sheet impedance and closed-conductor extrusion; FDTD and the
    # electrostatic solves treat the sheet as infinitely thin (documented provenance there).
    _entry(mw.geometry.PlanarPEC, owned("vertices", "holes", "normal_axis", "coordinate"), neutral("device"),
           use("physical_thickness", when=("structure.trainable_conductor_loss",), not_read={
               **_for((FDTD, *STATIC), "provenance only: this method treats the PlanarPEC sheet as infinitely "
                                       "thin")}),
           owner="geometry.planar_pec", within=_GEOMETRY_HOLDERS),
    _entry(mw.geometry.SurfaceMesh, owned("vertices", "triangles", "position", "rotation", "sheet_model"),
           neutral("physical_thickness", "device"), owner="geometry.surface_mesh", within=_GEOMETRY_HOLDERS),

    # Materials.
    _entry(
        PhysicalMaterial,
        owned("eps_r"),
        use("mu_r", not_read=_for(STATIC, _NO_MAGNETIC)),
        use("sigma_e", keys=("material.sigma_e",)),
        neutral("name", "material_id", "version"),
        *(use(name, keys=_MATERIAL_SURFACE_KEYS) for name in (
            "thickness_m", "layers", "geometry_mode", "roughness_front", "roughness_back", "dispersion", "gain",
            "scattering_coefficient", "xpd_coefficient", "conductor_model")),
        within=_MATERIAL_HOLDERS),
    _entry(
        mw.Material,
        owned("eps_r", "debye_poles", "drude_poles", "lorentz_poles", "mu_debye_poles", "mu_drude_poles",
              "mu_lorentz_poles", "pec"),
        use("mu_r", when=("material.trainable_mu_r",), not_read=_for(STATIC, _NO_MAGNETIC)),
        use("sigma_e", keys=("material.sigma_e",)),
        neutral("name"),
        use("sigma_m", not_read=_for(STATIC, _NO_MAGNETIC)),
        use("epsilon_tensor", when=("material.full_tensor",)),
        use("mu_tensor", not_read=_for(STATIC, _NO_MAGNETIC)),
        use("sigma_e_tensor", keys=("material.sigma_e",)),
        use("orientation", when=("material.full_tensor",)),
        *(use(name, keys=("material.nonlinear",)) for name in ("kerr_chi3", "nonlinearity")),
        use("modulation", keys=("material.modulation",),
            not_read=_for(STATIC, "the static response is the time average, which is the unmodulated permittivity")),
        use("pmc", keys=("material.pmc",)),
        use("breakdown", keys=("material.breakdown",)),
        use("mass_density", not_read={**_for(SURFACE, _NO_SAR), **_for(STATIC, _NO_SAR)}),
        within=_MATERIAL_HOLDERS),
    _entry(mw.media.LossyMetalMedium, use("conductivity", when=("structure.trainable_conductor_loss",)), owned("roughness"),
           neutral("name"), owner="material.lossy_metal", within=_MATERIAL_HOLDERS),
    _entry(mw.media.SurfaceImpedanceMedium, owned("impedance"), neutral("name"), owner="material.surface_impedance",
           within=_MATERIAL_HOLDERS),
    _entry(mw.media.Medium2D, owned("sigma_s"), neutral("name"), owner="material.medium2d", within=_MATERIAL_HOLDERS),
    _entry(mw.media.Graphene, owned("chemical_potential", "scattering_time", "temperature", "include_interband"),
           neutral("name"), owner="material.medium2d", within=_MATERIAL_HOLDERS),
    _entry(
        mw.media.GyromagneticFerrite,
        owned("eps_r"),
        *(use(name, not_read=_for(STATIC, "magnetization dynamics have no electrostatic meaning"))
          for name in ("saturation_magnetization", "bias_field", "gilbert_damping", "gyromagnetic_ratio",
                       "mu_infinity", "cgs_conversion")),
        use("sigma_e", keys=("material.sigma_e",)),
        neutral("name"),
        owner="material.gyromagnetic", within=_MATERIAL_HOLDERS),
    _entry(mw.media.PerturbationMedium, owned("base", "perturbation", "eps_sensitivity", "debye_sensitivities",
                                        "drude_sensitivities", "lorentz_sensitivities"),
           neutral("name"), owner="material.perturbation", within=_MATERIAL_HOLDERS),
    _entry(mw.media.CopperFoil, owned("thickness", "frequency_range", "conductivity", "mode_count", "edge_model"),
           neutral("name"), owner="material.copper_foil", within=_MATERIAL_HOLDERS),
    _entry(mw.DebyePole, owned("delta_eps", "tau"),
           within=("Material.debye_poles", "Material.mu_debye_poles", "PolePermittivity.poles")),
    _entry(mw.DrudePole, owned("plasma_frequency", "gamma"),
           within=("Material.drude_poles", "Material.mu_drude_poles", "PolePermittivity.poles")),
    # allow_gain only permits a negative delta_eps at construction.
    _entry(mw.LorentzPole, owned("delta_eps", "resonance_frequency", "gamma"), neutral("allow_gain"),
           within=("Material.lorentz_poles", "Material.mu_lorentz_poles", "PolePermittivity.poles")),
    _entry(mw.media.CustomDebyePole, owned("delta_eps", "tau"), within=("Material.debye_poles",)),
    _entry(mw.media.CustomDrudePole, owned("plasma_frequency", "gamma"), within=("Material.drude_poles",)),
    _entry(mw.media.CustomLorentzPole, owned("delta_eps", "resonance_frequency", "gamma"),
           within=("Material.lorentz_poles",)),
    _entry(mw.DiagonalTensor3, owned("xx", "yy", "zz"), within=_TENSOR_HOLDERS),
    _entry(mw.media.Tensor3x3, owned("rows"), within=_TENSOR_HOLDERS),
    _entry(mw.media.ComplexTensor3x3, owned("rows", "reciprocal"), within=("Material.epsilon_tensor", "Material.mu_tensor")),
    _entry(mw.media.ModulationSpec, owned("frequency", "amplitude", "phase"), within=("Material.modulation",)),
    _entry(mw.media.NonlinearSusceptibility, owned("chi2", "chi3"), within=("Material.nonlinearity",)),
    _entry(mw.media.TensorNonlinearSusceptibility, owned("chi2", "chi3", "rotation"), within=("Material.nonlinearity",)),
    _entry(mw.media.TwoPhotonAbsorption, owned("beta", "n0"), within=("Material.nonlinearity",)),
    _entry(mw.breakdown.DielectricBreakdown, owned("critical_field", "post_breakdown_conductivity", "minimum_duration", "model",
                                         "state", "recovery", "damage_parameters", "ramp_time", "default_ramp_steps"),
           within=("Material.breakdown",)),
    # The fit report and certification inputs are checked at construction.
    _entry(mw.media.RationalSurfaceImpedance, owned("poles", "residues", "direct", "frequency_range", "representation"),
           neutral("fit_report", "sample_frequencies", "passivity_tolerance", "certificate_samples"),
           within=("SurfaceImpedanceMedium.impedance",)),
    _entry(mw.media.GroisseRoughness, _loss_gradient("rms_height"), within=("LossyMetalMedium.roughness",)),
    _entry(mw.media.HammerstadRoughness, _loss_gradient("rms_height", "roughness_factor"),
           within=("LossyMetalMedium.roughness",)),
    _entry(mw.media.HurayRoughness, _loss_gradient("sphere_radius", "sphere_density", "relative_area"),
           within=("LossyMetalMedium.roughness",)),

    # Sources, waveforms and injections.
    _source(mw.PointDipole, owned("position", "polarization", "profile", "source_time"),
            use("width", not_read=_for(MOM, "MoM radiates an ideal point moment (it refuses other profiles), "
                                                "which has no width"))),
    _source(mw.PlaneWave, owned("direction", "polarization", "source_time"),
            use("injection", when=("source.injection_tfsf",), not_read=_INJECTION_NOT_READ),
            use("injection_axis", keys=("source.injection_axis",))),
    _source(mw.GaussianBeam, *_beam()),
    _source(mw.sources.AstigmaticGaussianBeam, *_beam("focus_u", "focus_v")),
    _source(mw.ModeSource, owned("position", "size", "mode_index", "direction", "polarization", "source_time",
                                 "bend_radius", "bend_axis"),
            use("injection", when=("source.injection_tfsf",), not_read=_INJECTION_NOT_READ)),
    _source(mw.UniformCurrentSource, owned("size", "polarization", "source_time", "center")),
    _source(mw.sources.CustomCurrentSource, owned("current_dataset", "source_time")),
    _source(mw.sources.CustomFieldSource, owned("field_dataset", "source_time")),
    _source(mw.mom.MaxwellPlaneModes, owned("frequency"), neutral("kind"),
            *(use(name, when=("source.MaxwellPlaneModes.trainable",))
              for name in ("wavevectors", "electric", "magnetic", "origin"))),
    _source(mw.fdfd.LumpedCurrentSource, owned("start", "end", "current")),
    _source(mw.esd.ESDCurrentSource, owned("port", "waveform", "direction")),
    _source(mw.esd.ESDVoltageSource, owned("port", "waveform", "discharge_resistance", "storage_capacitance")),
    _entry(mw.CW, use("frequency", when=("source.cw_frequency_mismatch",)), owned("amplitude", "phase"),
           neutral("kind"), within=_SOURCE_TIMES),
    _entry(mw.GaussianPulse, owned("frequency", "fwidth", "amplitude", "phase", "delay"), neutral("kind"),
           within=_SOURCE_TIMES),
    _entry(mw.sources.RickerWavelet, owned("frequency", "amplitude", "delay"), neutral("kind"), within=_SOURCE_TIMES),
    _entry(mw.sources.CustomSourceTime, owned("times", "amplitudes", "amplitude", "characteristic_frequency", "frequency",
                                      "interpolation"), within=_SOURCE_TIMES),
    _entry(mw.sources.BroadbandPulse, owned("freq_range", "minimum_amplitude", "amplitude", "phase", "offset"),
           neutral("kind"), within=_SOURCE_TIMES),
    _entry(mw.sources.BasebandGaussianPulse, owned("twidth", "offset", "amplitude"), within=_SOURCE_TIMES),
    _entry(mw.sources.BasebandRectangularPulse, owned("rise_time", "twidth", "offset", "amplitude"), within=_SOURCE_TIMES),
    _entry(mw.TFSF, owned("bounds"), within=("PlaneWave.injection", "GaussianBeam.injection",
                                             "AstigmaticGaussianBeam.injection", "ModeSource.injection")),
    _entry(mw.sources.FieldDataset, owned("coords", "components"), within=("CustomFieldSource.field_dataset",)),
    _entry(mw.sources.CurrentDataset, owned("coords", "components"), within=("CustomCurrentSource.current_dataset",)),
    _entry(mw.esd.ESDWaveform, owned("level_voltage", "discharge", "standard_revision"),
           within=("ESDCurrentSource.waveform", "ESDVoltageSource.waveform")),
    _entry(mw.esd.MeasuredWaveform, owned("time", "values", "units", "bandwidth"), neutral("provenance"),
           within=("ESDCurrentSource.waveform", "ESDVoltageSource.waveform")),

    # Monitors.
    _monitor(mw.PointMonitor, "position", "fields"),
    _monitor(mw.rf.WireMonitor, "wire", "frequencies", "quantities"),
    _monitor(mw.PlaneMonitor, "axis", "position", "fields", "frequencies", "compute_flux", "normal_direction"),
    _monitor(mw.FluxMonitor, "axis", "position", "frequencies", "normal_direction"),
    _monitor(mw.sar.IncidentPowerDensityMonitor, "axis", "position", "frequencies", "normal_direction",
             "spatial_average"),
    _monitor(mw.FinitePlaneMonitor, "position", "size", "fields", "frequencies", "compute_flux", "normal_direction",
             "sample_spacing", "closure_planes", labels=("face_label", "surface_name")),
    _monitor(mw.monitors.ClosedSurfaceMonitor, "faces", "frequencies"),
    _monitor(mw.monitors.PermittivityMonitor, "position", "size", "frequencies"),
    _monitor(mw.monitors.MediumMonitor, "position", "size", "frequencies"),
    _monitor(mw.monitors.PowerLossMonitor, "position", "size", "frequencies", "channels", "quadrature"),
    _monitor(mw.esd.BreakdownMonitor, "region", "position", "size", "quantities", "critical_field", "minimum_duration",
             "damage_exponent"),
    _monitor(mw.esd.ComponentStressMonitor, "port", "rating", "voltage_series", "current_series"),
    _monitor(mw.monitors.DipoleEmissionMonitor, "source_name", "frequencies"),
    _monitor(mw.monitors.FieldTimeMonitor, "components", "position", "size", "start", "stop", "interval"),
    _monitor(mw.monitors.FluxTimeMonitor, "axis", "position", "start", "stop", "interval", "normal_direction"),
    _monitor(mw.ModeMonitor, "position", "size", "mode_index", "direction", "polarization", "frequencies",
             "normal_direction", "bend_radius", "bend_axis"),
    _monitor(mw.monitors.DiffractionMonitor, "position", "size", "frequencies", "normal_direction", "orders"),
    _monitor(mw.monitors.AngleProjectionMonitor, *_projection("theta", "phi", "radius", "far_field_approx")),
    _monitor(mw.monitors.CartesianProjectionMonitor, *_projection("x", "y", "distance", "axis", "far_field_approx")),
    _monitor(mw.monitors.KSpaceProjectionMonitor, *_projection("ux", "uy", "axis", "hemisphere", "radius")),
    _monitor(mw.monitors.GaussianOverlapMonitor, "position", "size", "beam_waist", "frequencies", "direction", "polarization",
             "focus"),
    _monitor(mw.monitors.AstigmaticGaussianOverlapMonitor, "position", "size", "beam_waist", "frequencies", "direction",
             "polarization", "focus", "focus_u", "focus_v"),
    _monitor(mw.monitors.SurfaceFieldTimeMonitor, "position", "size", "components", "start", "stop", "interval",
             "exclude_surfaces"),
    _monitor(mw.monitors.AuxFieldTimeMonitor, "components", "position", "size", "start", "stop", "interval"),
    _entry(mw.esd.ComponentRating, owned("voltage", "current", "energy", "pulse_width", "model"),
           within=("ComponentStressMonitor.rating",)),

    # Ports and their parts.
    _port(mw.LumpedPort, owned("positive", "negative", "voltage_path", "reference_impedance", "current_surface",
                               "reference_plane", "voltage_aperture"),
          use("termination", keys=("port.termination",)),
          use("wire_binding", keys=("port.wire_binding",))),
    _port(mw.rf.TerminalPort, owned("positive_terminal", "negative_terminal", "integration_path", "reference_plane",
                                 "reference_impedance", "current_surface", "voltage_aperture"),
          use("termination", keys=("port.termination",)), use("wire_binding", keys=("port.wire_binding",))),
    _port(mw.rf.WavePort, owned("position", "size", "direction", "reference_plane", "modes")),
    _port(mw.rf.TerminalWavePort, owned("position", "size", "direction", "reference_plane", "terminals", "reference",
                                     "voltage_integrals", "current_integrals", "basis", "reference_impedance",
                                     "impedance_definition", "conductor_model")),
    _port(mw.ModePort, owned("position", "size", "mode_index", "direction", "polarization", "frequencies",
                             "source_time", "monitor_offset", "normal_direction", "bend_radius", "bend_axis")),
    _port(mw.rf.CoaxialLumpedPort, owned("center", "inner_diameter", "outer_diameter", "normal_axis", "direction",
                                      "reference_impedance", "reference_plane"),
          use("termination", keys=("port.termination",))),
    _entry(mw.AxisPath, owned("axis"), within=("LumpedPort.voltage_path",)),
    _entry(mw.rf.WirePortBinding, owned("kind", "negative", "positive"),
           within=("LumpedPort.wire_binding", "TerminalPort.wire_binding")),
    _entry(mw.rf.WireNodeRef, owned("wire", "point"), within=("WirePortBinding.negative", "WirePortBinding.positive")),
    _entry(mw.rf.TerminalRef, owned("structure_name"),
           within=("TerminalPort.positive_terminal", "TerminalPort.negative_terminal", "TerminalWavePort.terminals",
                   "TerminalWavePort.reference")),
    _entry(mw.rf.WaveModeSpec, neutral("name"), owned("family", "mode_index", "polarization", "impedance_definition",
                                                   "voltage_path", "current_contour", "conductor_model"),
           within=("WavePort.modes",)),
    *(_entry(cls, owned(*names), within=("TerminalWavePort.voltage_integrals", "TerminalWavePort.current_integrals",
                                         "WaveModeSpec.voltage_path", "WaveModeSpec.current_contour",
                                         "TerminalPort.integration_path"))
      for cls, names in ((mw.rf.AxisSegment, ("start", "end")), (mw.rf.CircleLoop, ("center", "radius", "axis", "segments")),
                         (mw.rf.CompositeCurrentIntegral, ("terms",)), (mw.rf.CompositeVoltageIntegral, ("terms",)),
                         (mw.rf.CustomCurrentIntegral, ("loop",)), (mw.rf.CustomVoltageIntegral, ("path",)),
                         (mw.rf.PolygonLoop, ("points",)), (mw.rf.PolylinePath, ("points",)),
                         (mw.rf.RectangleLoop, ("center", "size")))),
    *(_entry(cls, owned("r", "l", "c"), within=("LumpedPort.termination", "TerminalPort.termination",
                                                 "CoaxialLumpedPort.termination"))
      for cls in (mw.rf.SeriesRLC, mw.rf.ParallelRLC)),

    # Lumped elements, circuits and networks.
    _lumped(mw.rf.Resistor, "resistance"),
    _lumped(mw.rf.Capacitor, "capacitance"),
    _lumped(mw.rf.Inductor, "inductance"),
    _entry(mw.rf.CoaxialLumpedResistor, neutral("name"),
           owned("center", "inner_diameter", "outer_diameter", "normal_axis", "resistance"), owner="lumped_element",
           within=("Scene.lumped_elements",)),
    _entry(mw.circuits.Circuit, neutral("name", "metadata"), owned("parameters", "config"), owner="circuit",
           within=("Scene.circuits",)),
    _entry(mw.circuits.CircuitNode, neutral("name"), within=("Circuit",)),
    _device(mw.circuits.VoltageSource, "voltage", "waveform"),
    _device(mw.circuits.CurrentSource, "current", "waveform"),
    _device(mw.circuits.VoltageControlledVoltageSource, "control_positive", "control_negative", "gain"),
    _device(mw.circuits.VoltageControlledCurrentSource, "control_positive", "control_negative", "transconductance"),
    _device(mw.circuits.CurrentControlledCurrentSource, "control_source", "gain"),
    _device(mw.circuits.CurrentControlledVoltageSource, "control_source", "transresistance"),
    _entry(mw.circuits.MutualInductor, neutral("name"), owned("first_inductor", "second_inductor", "coupling"),
           within=("Circuit",)),
    _device(mw.circuits.TimedSwitch, "transition_times", "initially_closed", "on_resistance", "off_resistance"),
    _device(mw.circuits.Diode, "saturation_current", "ideality", "series_resistance", "junction_capacitance", "temperature"),
    _device(mw.circuits.PolynomialIV, "coefficients"),
    _device(mw.circuits.PiecewiseLinearIV, "voltages", "currents"),
    _device(mw.circuits.VoltageDependentCapacitor, "q_coefficients"),
    _entry(mw.circuits.PortBinding, owned("port_name", "positive", "negative"), within=("Circuit",)),
    _entry(mw.circuits.PiecewiseLinearWaveform, owned("times", "values"),
           within=("VoltageSource.waveform", "CurrentSource.waveform")),
    _entry(mw.circuits.PulseWaveform, owned("initial", "pulsed", "delay", "rise", "fall", "width", "period"),
           within=("VoltageSource.waveform", "CurrentSource.waveform")),
    _entry(mw.circuits.SineWaveform, owned("offset", "amplitude", "frequency", "delay", "damping", "phase_degrees"),
           within=("VoltageSource.waveform", "CurrentSource.waveform")),
    _entry(mw.circuits.MNAConfig, owned("integration", "pivot_tolerance", "regularization", "initialization",
                               "dense_unknown_limit"), neutral("diagnostics"), within=("Circuit.config",)),
    _entry(mw.rf.TouchstoneNetwork, neutral("name", "device"),
           owned("connections", "network", "path", "fit", "model", "extrapolation", "delay_seconds",
                 "max_delay_steps", "dtype"), owner="network", within=("Scene.networks",)),
    _entry(mw.rf.NetworkBlock, neutral("name"),
           owned("network", "connections", "fit", "model", "extrapolation", "delay_seconds", "max_delay_steps"),
           owner="network", within=("Scene.networks",)),
    _entry(mw.rf.rational.StateSpaceNetwork, owned("A", "B", "C", "D", "representation", "port_order", "passivity_margin"),
           neutral("report"), within=("NetworkBlock.model", "TouchstoneNetwork.model")),
    _entry(mw.rf.rational.RationalModel, owned("poles", "residues", "direct", "proportional", "representation"),
           neutral("report"), within=("NetworkBlock.model", "TouchstoneNetwork.model")),

    # Thin wires, absorbers and electrostatic declarations.
    _entry(mw.rf.ThinWire, neutral("name"), owned("points", "radius", "conductor", "endpoints", "snap"), owner="thin_wire",
           within=("Scene.thin_wires",)),
    _entry(mw.rf.WireConductor, owned("kind", "conductivity", "permeability"), within=("ThinWire.conductor",)),
    _entry(mw.rf.WireEnd, owned("kind", "structure"), neutral("node_name"), within=("ThinWire.endpoints",)),
    _entry(mw.fdtd.InternalAbsorber, neutral("name"), owned("position", "size", "direction", "boundary", "grid_shift"),
           owner="internal_absorber", within=("Scene.internal_absorbers",)),
    _entry(mw.electrostatic.ElectrostaticTerminal, neutral("name"), owned("structure", "potential", "grounded"),
           use("geometry", when=("electrostatic_terminal.trainable_geometry",)),
           use("charge", keys=("electrostatic_terminal.floating",)),
           owner="electrostatic_terminal", within=("Scene.electrostatic_terminals",)),
    _entry(mw.electrostatic.ChargeDensity, owned("geometry", "density"), owner="charge_density", within=("Scene.charge_densities",)),
)

MANIFEST: Mapping[type, SceneClass] = MappingProxyType({entry.cls: entry for entry in ENTRIES})
_BY_NAME = {entry.cls.__name__: entry for entry in ENTRIES}

# Every other public class, by role. A new public class must join MANIFEST or one of these.
NOT_SCENE = MappingProxyType({
    "simulation configuration": (
        "FDTDConfig", "FDFDConfig", "FDFDSolverConfig", "TimeConfig", "SpectralSampler", "FDTDParallelConfig", "CPMLConfig",
        "FDTDMaterialCache", "LocalResponseConfig", "LocalDesignRegion", "FieldProjection", "FarFieldProjection",
        "ElectrostaticConfig", "CapacitanceConfig", "ElectrostaticBoundarySpec", "ElectrostaticSolverConfig",
        "ElectrostaticInitialCondition", "TruncationEstimate", "NonlinearSolveConfig", "MultiGPUExecution",
        "BatchExecution", "BatchParameters", "BatchVariation", "FDTDResumeCheckpoint"),
    "simulation excitation": ("PortExcitation", "PortSweep", "ModalPortSweep", "PortDrive"),
    "MoM configuration": (
        "MoMConfig", "MoMPeriodicConfig", "MoMDyadicPeriodicConfig", "HMatrixConfig", "H2Config", "DirectH2Config",
        "FMMConfig", "DirectionalFMMConfig", "PFFTConfig", "IdentityRightPreconditioner",
        "LeafBlockRightPreconditioner", "FunctionRightPreconditioner", "NearFieldRightPreconditioner",
        "PreconditionerContext", "CalderonPreconditioner", "KernelErrorBudget", "HarmonicBalanceConfig",
        "HarmonicCutoffConfig", "LinearCoupledConfig", "TetrahedralVolumeMesh", "OpenSurfaceConfig",
        "LayeredSurfaceConfig", "AdaptiveSweep", "FieldQueryConfig", "DyadicIntegrationConfig", "SurfaceMeshConfig",
        "EdgeGrading"),
    "scene builder": ("SceneModule", "PECFrame", "ProcessStack", "LayerMap", "ViaMap", "MaterialLibraryEntry",
                      "DispersionFit", "RationalFitConfig"),
    "abstract base": ("CircuitDevice", "SurfaceImpedanceModel", "SurfaceRoughness"),
    "postprocess specification": (
        "SARAveraging", "PowerNormalization", "Ludwig3", "LowFrequencySmoothingSpec", "SmoothBreakdownRisk",
        "BeamCodebook", "BeamWeights", "MultipathEnvironment", "MaxHoldComposite"),
    "execution": ("Simulation", "SimulationMethod", "BatchSimulation", "PreparedBatch", "BatchResultIterator"),
    "option Enum": (
        "BoundaryKind", "FieldComponent", "Direction", "ModePolarization", "DipoleProfile", "SubpixelAveraging",
        "PECTreatment", "FDFDSolverType", "FDFDPreconditioner", "Precision", "AbsorberKind", "SpectralWindowKind",
        "CPMLMemoryMode", "AveragingProfile", "MoMOperatorBackend", "RieszStorage", "LayeredOperatorBackend",
        "LayeredConductorModel", "PlanarMesher", "ACAPivoting", "FMMTableStorage", "FMMShiftStorage", "FMMNearStorage",
        "FMMKernelBoundRule", "FMMExpansionCenter", "LeafFactorKernel", "PFFTGridPrecision", "PFFTExteriorGrids",
        "OpenEdgeRule", "HarmonicBalanceMethod", "TensorFaceRule", "HarmonicImplicitSolver", "CutoffFailure",
        "TransverseMap", "DyadicRadialMap", "DyadicSpatialRule"),
    "result": (
        "Result", "ResultSequence", "BatchResult", "BatchItem", "BatchOutput", "BatchPlanReport", "ExecutionRecord",
        "DistributedFailure", "FailureKind", "FieldData", "FluxData", "FarFieldData", "ModeAmplitudeData",
        "SParameterData", "FieldFrame", "FramedFieldData", "AntennaData", "ArrayBasisData", "BeamData",
        "EmbeddedElementPatternData", "MIMOData", "AggregatedSceneGradient", "BreakdownEvent", "BreakdownResultData",
        "BreakdownStressData", "ComponentStressData", "CapacitanceData", "TruncationReport", "CircuitData",
        "ElectrostaticResultData", "EmbeddedNetworkData", "NetworkData", "PortData", "NetworkPhysicalityReport",
        "PowerLossData", "IncidentPowerDensity", "SARPeak", "SARResult", "SmoothBreakdownRiskData", "WireData",
        "ESDDiagnostics", "ESDPortRecord", "ESDResampledWaveform", "AdaptiveSweepResult", "DispersionFitReport",
        "FitReport", "NetworkFitReport", "PassivityReport", "DiscreteStateSpaceNetwork",
        "PassiveMacromodel", "MacromodelReport", "HamiltonianReport"),
    "error or warning": (
        "MaxwellError", "UnsupportedFeatureError", "SetupError", "ConvergenceError", "SchemaError", "CapacityError",
        "MaxwellWarning", "ExperimentalWarning", "FDTDPassivityError",
        "TerminalWavePortApertureWarning", "TouchstoneParseError"),
})


def row(method: str, key: str) -> Capability:
    """The row that decides ``key`` for ``method``, as ``check_scene`` would.

    A refused dotted prefix (a refused collection, a refused ``Scene.symmetry``)
    decides its members; a key without a row is refused as not declared.
    """
    rows = TABLES[method]
    parts = key.split(".")
    for end in range(1, len(parts)):
        prefix = rows.get(".".join(parts[:end]))
        if prefix is not None and prefix.support is not Support.SUPPORTED:
            return prefix
    return rows.get(key) or Capability(key, Support.UNSUPPORTED, f"{key} is not declared for {method}")


def _refusal(capability: Capability) -> Cell:
    return Cell(capability.support, capability.key, capability.reason)


def _class_cell(entry: SceneClass, method: str) -> Cell:
    """Whether ``method`` sees a declaration of this class at all."""
    if entry.owner is not None:
        owner = row(method, entry.owner)
        if owner.support is not Support.SUPPORTED:
            return _refusal(owner)
    if not entry.within:
        return Cell(Support.SUPPORTED, entry.owner)
    containers = []
    for holder in entry.within:
        name, _, field_name = holder.partition(".")
        holder_entry = _BY_NAME[name]
        cell = _field_cell(holder_entry, field_name, method) if field_name else _class_cell(holder_entry, method)
        if cell.support is Support.SUPPORTED:
            return Cell(Support.SUPPORTED, entry.owner)
        containers.append(cell)
    return containers[0]


def _field_cell(entry: SceneClass, name: str, method: str) -> Cell:
    use_ = entry.fields[name]
    if use_.neutral:
        return Cell(None, reason="neutral")
    seen = _class_cell(entry, method)
    if seen.support is not Support.SUPPORTED:
        return seen
    conditional = tuple(key for key in (*use_.keys, *use_.when) if row(method, key).support is not Support.SUPPORTED)
    if method in use_.gaps:
        return Cell(Support.UNSUPPORTED, None, use_.gaps[method], conditional, gap=True)
    if method in use_.not_read:
        return Cell(Support.INHERENT, None, use_.not_read[method], conditional)
    if use_.keys and all(key in conditional for key in use_.keys):
        refused = _refusal(row(method, use_.keys[0]))
        return dataclasses.replace(refused, conditional=conditional[1:])
    admitted = next((key for key in use_.keys if key not in conditional), None)
    return Cell(Support.SUPPORTED, admitted, "", conditional)


def consumption(cls) -> Mapping[str, Mapping[str, Cell]]:
    """``{field: {method: Cell}}`` for one manifest class."""
    entry = MANIFEST[cls]
    return MappingProxyType({name: MappingProxyType({method: _field_cell(entry, name, method) for method in METHODS})
                             for name in entry.fields})


def gaps() -> tuple[tuple[str, str, str], ...]:
    """Every declared open gap as ``(class name, field, method)``."""
    return tuple((entry.cls.__name__, name, method) for entry in ENTRIES for name, use_ in entry.fields.items()
                 for method in use_.gaps)
