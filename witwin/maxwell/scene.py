from __future__ import annotations

from ._fixed_inputs import require_fixed_input

import math
import copy
import warnings
from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

import numpy as np
import torch

from witwin.core import Box, Structure
from witwin.core.geometry.base import GeometryBase
from .circuits import Circuit, CircuitNode
from .lumped import Capacitor, CoaxialLumpedResistor, Inductor, Resistor
from .network import NetworkBlock
from .ports import (
    CoaxialLumpedPort,
    LumpedPort,
    ModePort,
    TerminalPort,
    TerminalWavePort,
    WavePort,
    _resolve_terminal_port,
    _resolve_terminal_wave_port,
)
from .thin_wire import ThinWire
from .absorbers import InternalAbsorber, ModeABC, MurABC
from ._enums import StrEnum, parse_enum
from .errors import SetupError, UnsupportedFeatureError


class BoundaryKind(StrEnum):
    """The boundary condition of a domain face.

    ``MIXED`` is never declared: it is the summary ``BoundarySpec.kind`` and
    ``BoundarySpec.axis_kind`` report when the faces differ.
    """

    NONE = "none"
    PML = "pml"
    PERIODIC = "periodic"
    BLOCH = "bloch"
    PEC = "pec"
    PMC = "pmc"
    MUR = "mur"
    MODE_ABC = "mode_abc"
    MIXED = "mixed"


BoundaryAxisOverride: TypeAlias = BoundaryKind | tuple[BoundaryKind, BoundaryKind]
BlochWavevector: TypeAlias = tuple[float, float, float] | Literal["auto"]

_BOUNDARY_AXES = ("x", "y", "z")
_BOUNDARY_AXIS_TO_INDEX = {axis: index for index, axis in enumerate(_BOUNDARY_AXES)}
_BOUNDARY_SIDE_TO_INDEX = {"low": 0, "high": 1}
_PAIR_ONLY_BOUNDARY_KINDS = {BoundaryKind.PERIODIC, BoundaryKind.BLOCH}


def _normalize_boundary_kind(value) -> BoundaryKind:
    try:
        kind = BoundaryKind(value)
    except ValueError:
        kind = BoundaryKind.MIXED
    if kind is BoundaryKind.MIXED:
        choices = ", ".join(repr(item.value) for item in BoundaryKind if item is not BoundaryKind.MIXED)
        raise SetupError(f"Boundary kind must be one of {choices}, got {value!r}.")
    return kind

def _normalize_boundary_axis_override(name: str, value) -> tuple[BoundaryKind, BoundaryKind] | None:
    if value is None:
        return None
    if isinstance(value, str):
        kind = _normalize_boundary_kind(value)
        return (kind, kind)
    if len(value) != 2:
        raise ValueError(f"{name} must be a boundary kind or a (low, high) pair.")
    return tuple(_normalize_boundary_kind(component) for component in value)


def _normalize_boundary_face_override(name: str, value) -> BoundaryKind | None:
    if value is None:
        return None
    return _normalize_boundary_kind(value)


def _summarize_boundary_kind(face_kinds) -> BoundaryKind:
    first_kind = face_kinds[0][0]
    if all(low_kind is first_kind and high_kind is first_kind for low_kind, high_kind in face_kinds):
        return first_kind
    return BoundaryKind.MIXED


@dataclass(frozen=True)
class Domain:
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]

    def __init__(self, bounds):
        require_fixed_input(bounds, name="Domain.bounds")
        if len(bounds) != 3:
            raise ValueError("bounds must contain x, y, z ranges.")
        normalized = []
        for axis_bounds in bounds:
            if len(axis_bounds) != 2:
                raise ValueError("Each axis bound must contain exactly two values.")
            start, end = float(axis_bounds[0]), float(axis_bounds[1])
            if end <= start:
                raise ValueError("Each axis bound must have end > start.")
            normalized.append((start, end))
        object.__setattr__(self, "bounds", tuple(normalized))

    @classmethod
    def from_domain_range(cls, domain_range):
        if len(domain_range) != 6:
            raise ValueError("domain_range must contain six values.")
        return cls(
            bounds=(
                (domain_range[0], domain_range[1]),
                (domain_range[2], domain_range[3]),
                (domain_range[4], domain_range[5]),
            )
        )

    @property
    def domain_range(self):
        return (
            self.bounds[0][0],
            self.bounds[0][1],
            self.bounds[1][0],
            self.bounds[1][1],
            self.bounds[2][0],
            self.bounds[2][1],
        )


@dataclass(frozen=True)
class MeshOverrideStructure:
    """Maximum mesh step enforced inside a geometry's axis-aligned bounding box."""

    geometry: Any
    dl: tuple[float, float, float]

    def __init__(self, geometry, dl):
        if geometry is None:
            raise ValueError("MeshOverrideStructure requires geometry=.")
        if isinstance(dl, (int, float)):
            values = (float(dl),) * 3
        else:
            values = tuple(float(value) for value in dl)
            if len(values) != 3:
                raise ValueError("MeshOverrideStructure dl must be a scalar or (dx, dy, dz).")
        if any(value <= 0.0 for value in values):
            raise ValueError("MeshOverrideStructure dl values must be > 0.")
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "dl", values)


_CORNER_MEDIA = ("metal", "dielectric", "all")
_LAYER_BOUNDS_SNAPPING = ("bounds", "lower", "upper", "center")


def _optional_positive_int(value, name: str):
    if value is None:
        return None
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer or None.")
    return int(value)


def _optional_positive_float(value, name: str):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number or None.")
    return value


@dataclass(frozen=True)
class CornerFinderSpec:
    """Corner detection on the 2D cross-section of a layer.

    A polygon vertex is a corner when the turn between its two edges is at least
    ``angle_threshold`` radians (the spanned angle is at most
    ``pi - angle_threshold``), so collinear and gently faceted vertices are
    skipped. ``medium`` selects metal (PEC, planar PEC, lossy-metal, surface
    impedance, copper foil), non-metal, or all structures. With
    ``distance_threshold`` the outline is first simplified with Douglas-Peucker
    at that tolerance and surviving corners closer than it are merged.

    ``convex_resolution`` / ``concave_resolution`` / ``mixed_resolution``
    request that many cells across each outline edge whose two end corners are
    both convex, both concave, or mixed (strip widths, notch widths, steps).
    """

    medium: Literal["metal", "dielectric", "all"] = "metal"
    angle_threshold: float = 0.25 * math.pi
    distance_threshold: float | None = None
    concave_resolution: int | None = None
    convex_resolution: int | None = None
    mixed_resolution: int | None = None

    def __post_init__(self):
        require_fixed_input((self.angle_threshold, self.distance_threshold), name="CornerFinderSpec controls")
        if self.medium not in _CORNER_MEDIA:
            raise ValueError("CornerFinderSpec.medium must be 'metal', 'dielectric', or 'all'.")
        angle = float(self.angle_threshold)
        if not 0.0 <= angle < math.pi:
            raise ValueError("CornerFinderSpec.angle_threshold must lie in [0, pi).")
        object.__setattr__(self, "angle_threshold", angle)
        object.__setattr__(
            self, "distance_threshold",
            _optional_positive_float(self.distance_threshold, "CornerFinderSpec.distance_threshold"),
        )
        for name in ("concave_resolution", "convex_resolution", "mixed_resolution"):
            object.__setattr__(self, name, _optional_positive_int(getattr(self, name), f"CornerFinderSpec.{name}"))


@dataclass(frozen=True)
class GridRefinement:
    """Local refinement around a point: ``num_cells`` cells of step ``dl``.

    The step is ``vacuum_dl / refinement_factor`` and/or ``dl`` (the smaller one
    when both are given; ``refinement_factor=2`` when neither is). The refined
    window is ``num_cells * step`` wide and centered on the point; the
    surrounding mesh grades into it within ``GridSpec.auto(max_ratio=...)``.
    """

    refinement_factor: float | None = None
    dl: float | None = None
    num_cells: int = 3

    def __post_init__(self):
        require_fixed_input((self.refinement_factor, self.dl), name="GridRefinement controls")
        object.__setattr__(
            self, "refinement_factor",
            _optional_positive_float(self.refinement_factor, "GridRefinement.refinement_factor"),
        )
        object.__setattr__(self, "dl", _optional_positive_float(self.dl, "GridRefinement.dl"))
        object.__setattr__(self, "num_cells", _optional_positive_int(self.num_cells, "GridRefinement.num_cells"))
        if self.num_cells is None:
            raise ValueError("GridRefinement.num_cells must be a positive integer.")

    def grid_size(self, vacuum_dl: float) -> float:
        """Refined step for a given vacuum step of the auto mesh."""
        factor = self.refinement_factor
        if factor is None and self.dl is None:
            factor = 2.0
        step = math.inf
        if factor is not None:
            step = float(vacuum_dl) / factor
        if self.dl is not None:
            step = min(step, self.dl)
        return step


@dataclass(frozen=True)
class LayerRefinementSpec:
    """Thin-interval and layer-local refinement for ``GridSpec.auto``.

    ``min_cells`` (default 2) is the global rule: every structure-bounded
    interval on the ``axes`` (``None`` = all axes) gets at least that many cells;
    ``min_cells=None`` disables it.

    Setting ``axis`` and ``bounds`` (layer extent along ``axis``) locates a layer
    and enables the opt-in layer features, all off by default:

    * ``min_steps_along_axis``: at least this many steps across the layer.
    * ``bounds_snapping``: force nodes at the ``"lower"``, ``"upper"``,
      ``"center"`` or both (``"bounds"``) layer planes.
    * ``bounds_refinement``: a :class:`GridRefinement` window around each layer
      plane along ``axis`` (dropped when not finer than ``min_steps_along_axis``).
    * ``corner_finder``: detect corners of the structure cross-sections on the
      layer mid-plane; with ``corner_snapping`` (default on once a finder is set)
      in-plane nodes pass through every corner, and ``corner_refinement`` adds a
      :class:`GridRefinement` window around each corner on both in-plane axes.
    * ``dl_min_from_gap_width``: resolve each in-plane gap between metal
      cross-sections that is narrower than two vacuum steps with at least two
      cells, with nodes on the gap edges.
    * ``refinement_inside_sim_only``: ignore corners outside the domain.
    """

    min_cells: int | None = 2
    axes: tuple[str, ...] | None = None
    axis: str | None = None
    bounds: tuple[float, float] | None = None
    min_steps_along_axis: float | None = None
    bounds_snapping: Literal["bounds", "lower", "upper", "center"] | None = None
    bounds_refinement: GridRefinement | None = None
    corner_finder: CornerFinderSpec | None = None
    corner_snapping: bool = True
    corner_refinement: GridRefinement | None = None
    refinement_inside_sim_only: bool = True
    dl_min_from_gap_width: bool = False

    def __post_init__(self):
        if self.min_cells is not None:
            min_cells = int(self.min_cells)
            if min_cells < 1:
                raise ValueError("LayerRefinementSpec.min_cells must be >= 1.")
            object.__setattr__(self, "min_cells", min_cells)
        if self.axes is not None:
            axes = tuple(str(axis).lower() for axis in self.axes)
            if any(axis not in _BOUNDARY_AXIS_TO_INDEX for axis in axes):
                raise ValueError("LayerRefinementSpec.axes entries must be 'x', 'y', or 'z'.")
            object.__setattr__(self, "axes", axes)
        self._validate_layer()

    def _validate_layer(self):
        require_fixed_input((self.bounds, self.min_steps_along_axis), name="LayerRefinementSpec layer controls")
        features = (
            self.min_steps_along_axis is not None
            or self.bounds_snapping is not None
            or self.bounds_refinement is not None
            or self.corner_finder is not None
            or self.corner_refinement is not None
            or self.dl_min_from_gap_width
        )
        if self.axis is None and self.bounds is None:
            if features:
                raise ValueError(
                    "LayerRefinementSpec layer features (min_steps_along_axis, bounds_*, corner_*, "
                    "dl_min_from_gap_width) require axis= and bounds=."
                )
            return
        if self.axis is None or self.bounds is None:
            raise ValueError("LayerRefinementSpec.axis and LayerRefinementSpec.bounds must be given together.")
        axis = {0: "x", 1: "y", 2: "z"}.get(self.axis, self.axis) if isinstance(self.axis, int) else str(self.axis).lower()
        if axis not in _BOUNDARY_AXIS_TO_INDEX:
            raise ValueError("LayerRefinementSpec.axis must be 'x', 'y', 'z' or 0, 1, 2.")
        object.__setattr__(self, "axis", axis)
        if isinstance(self.bounds, torch.Tensor):
            bounds = tuple(float(v) for v in self.bounds.detach().cpu().reshape(-1))
        else:
            bounds = tuple(float(v) for v in self.bounds)
        if len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or bounds[1] < bounds[0]:
            raise ValueError("LayerRefinementSpec.bounds must be finite (lower, upper) with lower <= upper.")
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(
            self, "min_steps_along_axis",
            _optional_positive_float(self.min_steps_along_axis, "LayerRefinementSpec.min_steps_along_axis"),
        )
        if self.bounds_snapping is not None and self.bounds_snapping not in _LAYER_BOUNDS_SNAPPING:
            raise ValueError("LayerRefinementSpec.bounds_snapping must be 'bounds', 'lower', 'upper', 'center' or None.")
        for name in ("bounds_refinement", "corner_refinement"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, GridRefinement):
                raise TypeError(f"LayerRefinementSpec.{name} must be a GridRefinement or None.")
        if self.corner_finder is not None and not isinstance(self.corner_finder, CornerFinderSpec):
            raise TypeError("LayerRefinementSpec.corner_finder must be a CornerFinderSpec or None.")
        if self.corner_refinement is not None and self.corner_finder is None:
            raise ValueError("LayerRefinementSpec.corner_refinement requires corner_finder=.")
        object.__setattr__(self, "corner_snapping", bool(self.corner_snapping))
        object.__setattr__(self, "refinement_inside_sim_only", bool(self.refinement_inside_sim_only))
        object.__setattr__(self, "dl_min_from_gap_width", bool(self.dl_min_from_gap_width))

    @classmethod
    def from_structures(cls, structures, axis, **kwargs) -> "LayerRefinementSpec":
        """Layer spanning the axis extent of ``structures`` (Structure or geometry)."""
        from .fdtd.meshing.autogrid import _geometry_aabb

        items = list(structures) if isinstance(structures, (list, tuple)) else [structures]
        if not items:
            raise ValueError("LayerRefinementSpec.from_structures requires at least one structure.")
        axis_name = str(axis).lower() if not isinstance(axis, int) else "xyz"[axis]
        if axis_name not in _BOUNDARY_AXIS_TO_INDEX:
            raise ValueError("LayerRefinementSpec.from_structures axis must be 'x', 'y', 'z' or 0, 1, 2.")
        index = _BOUNDARY_AXIS_TO_INDEX[axis_name]
        lows, highs = [], []
        for item in items:
            lo, hi = _geometry_aabb(getattr(item, "geometry", item))
            lows.append(float(lo[index]))
            highs.append(float(hi[index]))
        return cls(axis=axis_name, bounds=(min(lows), max(highs)), **kwargs)

    @property
    def is_layer(self) -> bool:
        return self.axis is not None

    def covers(self, axis: str) -> bool:
        return self.min_cells is not None and (self.axes is None or str(axis).lower() in self.axes)


def _validate_custom_axis_coords(values, axis: str) -> np.ndarray:
    require_fixed_input(values, name=f"GridSpec.{axis}_coords")
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    coords = np.array(np.asarray(values, dtype=np.float64), copy=True)
    if coords.ndim != 1:
        raise ValueError(f"GridSpec.custom {axis}_coords must be a 1D array of node coordinates.")
    if coords.size < 2:
        raise ValueError(f"GridSpec.custom {axis}_coords must contain at least two node coordinates.")
    if not np.all(np.isfinite(coords)):
        raise ValueError(f"GridSpec.custom {axis}_coords must contain only finite values.")
    if not np.all(np.diff(coords) > 0.0):
        raise ValueError(f"GridSpec.custom {axis}_coords must be strictly increasing.")
    coords.setflags(write=False)
    return coords


@dataclass(frozen=True, eq=False)
class GridSpec:
    dx: float | None
    dy: float | None
    dz: float | None
    x_coords: np.ndarray | None = None
    y_coords: np.ndarray | None = None
    z_coords: np.ndarray | None = None
    min_steps_per_wavelength: float | None = None
    wavelength: float | None = None
    max_ratio: float | None = None
    override_structures: tuple[MeshOverrideStructure, ...] = ()
    layer_refinement: LayerRefinementSpec | tuple[LayerRefinementSpec, ...] | None = None

    @classmethod
    def uniform(cls, dl: float) -> "GridSpec":
        require_fixed_input(dl, name="GridSpec.uniform spacing")
        step = float(dl)
        return cls(step, step, step)

    @classmethod
    def anisotropic(cls, dx: float, dy: float, dz: float) -> "GridSpec":
        require_fixed_input((dx, dy, dz), name="GridSpec.anisotropic spacing")
        return cls(float(dx), float(dy), float(dz))

    @classmethod
    def custom(cls, x_coords, y_coords, z_coords) -> "GridSpec":
        return cls(
            None,
            None,
            None,
            x_coords=_validate_custom_axis_coords(x_coords, "x"),
            y_coords=_validate_custom_axis_coords(y_coords, "y"),
            z_coords=_validate_custom_axis_coords(z_coords, "z"),
        )

    @classmethod
    def auto(
        cls,
        min_steps_per_wavelength: float = 10,
        wavelength: float | None = None,
        max_ratio: float = 1.4,
        override_structures=(),
        layer_refinement: LayerRefinementSpec | tuple[LayerRefinementSpec, ...] | None = None,
    ) -> "GridSpec":
        require_fixed_input((min_steps_per_wavelength, wavelength, max_ratio), name="GridSpec.auto controls")
        steps = float(min_steps_per_wavelength)
        if steps <= 0.0:
            raise ValueError("GridSpec.auto min_steps_per_wavelength must be > 0.")
        if wavelength is not None:
            wavelength = float(wavelength)
            if wavelength <= 0.0:
                raise ValueError("GridSpec.auto wavelength must be > 0 when provided.")
        ratio = float(max_ratio)
        if ratio <= 1.0:
            raise ValueError("GridSpec.auto max_ratio must be > 1.")
        overrides = tuple(override_structures)
        for override in overrides:
            if not isinstance(override, MeshOverrideStructure):
                raise TypeError(
                    "GridSpec.auto override_structures must contain MeshOverrideStructure entries."
                )
        if isinstance(layer_refinement, (list, tuple)):
            # Several specs (e.g. one per PCB layer) are kept as a tuple; a single
            # spec is stored unchanged.
            layer_refinement = tuple(layer_refinement)
            if not layer_refinement or not all(isinstance(spec, LayerRefinementSpec) for spec in layer_refinement):
                raise TypeError("GridSpec.auto layer_refinement must be a LayerRefinementSpec or a non-empty sequence of them.")
        elif layer_refinement is not None and not isinstance(layer_refinement, LayerRefinementSpec):
            raise TypeError("GridSpec.auto layer_refinement must be a LayerRefinementSpec.")
        return cls(
            None,
            None,
            None,
            min_steps_per_wavelength=steps,
            wavelength=wavelength,
            max_ratio=ratio,
            override_structures=overrides,
            layer_refinement=layer_refinement,
        )

    @property
    def is_custom(self) -> bool:
        return self.x_coords is not None or self.y_coords is not None or self.z_coords is not None

    @property
    def is_auto(self) -> bool:
        return self.min_steps_per_wavelength is not None

    def _require_scalar_spacing(self):
        if self.is_custom:
            raise ValueError("GridSpec.custom has no scalar spacing; use axis_coords()/min_spacing.")
        if self.is_auto:
            raise ValueError(
                "GridSpec.auto has no scalar spacing; it resolves to node coordinates at prepare time."
            )

    @property
    def spacing(self) -> tuple[float, float, float]:
        self._require_scalar_spacing()
        return (self.dx, self.dy, self.dz)

    @property
    def is_uniform(self) -> bool:
        self._require_scalar_spacing()
        return bool(np.isclose(self.dx, self.dy) and np.isclose(self.dy, self.dz))

    @property
    def min_spacing(self) -> tuple[float, float, float]:
        if self.is_auto:
            raise ValueError(
                "GridSpec.auto has no spacing before prepare; it resolves to node coordinates at prepare time."
            )
        if self.is_custom:
            return (
                float(np.diff(self.x_coords).min()),
                float(np.diff(self.y_coords).min()),
                float(np.diff(self.z_coords).min()),
            )
        return (self.dx, self.dy, self.dz)

    def axis_coords(self, axis: str) -> np.ndarray | None:
        axis_name = str(axis).lower()
        if axis_name not in _BOUNDARY_AXIS_TO_INDEX:
            raise ValueError("axis must be 'x', 'y', or 'z'.")
        return (self.x_coords, self.y_coords, self.z_coords)[_BOUNDARY_AXIS_TO_INDEX[axis_name]]


@dataclass(frozen=True)
class BoundarySpec:
    """Outer boundary of the scene: one kind per face, plus PML depth and strength.

    ``strength`` has two readings, chosen by the FDTD absorber (``FDTDConfig.absorber``):

    - ``"cpml"``, ``"stablepml"``, ``"absorber"`` and the FDFD stretched-coordinate PML read it
      as a *multiplier* of the reflection-calibrated peak conductivity: a value in (0, 100]
      scales it, while 0 and any value above 100 (including the ``BoundarySpec.pml`` default
      1e6) leave the nominal profile unchanged.
    - ``"pml"`` (the simple quartic PML) reads it as an *absolute* scale: the stored peak
      conductivity of its quartic ramp is ``2 * strength / (dt * min(dx, dy, dz))``.

    Both readings are kept as they are in 0.7.0; unifying them would change CPML numerics.
    """

    kind: BoundaryKind = BoundaryKind.NONE
    num_layers: int = 0
    strength: float = 0.0
    bloch_wavevector: BlochWavevector = (0.0, 0.0, 0.0)
    x: BoundaryAxisOverride | None = None
    y: BoundaryAxisOverride | None = None
    z: BoundaryAxisOverride | None = None
    x_low: BoundaryKind | None = None
    x_high: BoundaryKind | None = None
    y_low: BoundaryKind | None = None
    y_high: BoundaryKind | None = None
    z_low: BoundaryKind | None = None
    z_high: BoundaryKind | None = None
    mur_abc: MurABC = MurABC()
    mode_abc: ModeABC | None = None
    _face_kinds: tuple[tuple[BoundaryKind, BoundaryKind], tuple[BoundaryKind, BoundaryKind], tuple[BoundaryKind, BoundaryKind]] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self):
        require_fixed_input((self.num_layers, self.strength, self.bloch_wavevector), name="BoundarySpec physical controls")
        kind = _normalize_boundary_kind(self.kind)

        if self.num_layers < 0:
            raise ValueError("BoundarySpec.num_layers must be >= 0.")
        if self.strength < 0:
            raise ValueError("BoundarySpec.strength must be >= 0.")

        if isinstance(self.bloch_wavevector, str):
            if self.bloch_wavevector != "auto":
                raise ValueError("BoundarySpec.bloch_wavevector must be a length-3 tuple or 'auto'.")
            bloch_wavevector: BlochWavevector = "auto"
        else:
            bloch_wavevector = tuple(float(value) for value in self.bloch_wavevector)
            if len(bloch_wavevector) != 3:
                raise ValueError("BoundarySpec.bloch_wavevector must contain exactly three values.")

        axis_overrides = {
            "x": _normalize_boundary_axis_override("x", self.x),
            "y": _normalize_boundary_axis_override("y", self.y),
            "z": _normalize_boundary_axis_override("z", self.z),
        }
        face_overrides = {
            ("x", "low"): _normalize_boundary_face_override("x_low", self.x_low),
            ("x", "high"): _normalize_boundary_face_override("x_high", self.x_high),
            ("y", "low"): _normalize_boundary_face_override("y_low", self.y_low),
            ("y", "high"): _normalize_boundary_face_override("y_high", self.y_high),
            ("z", "low"): _normalize_boundary_face_override("z_low", self.z_low),
            ("z", "high"): _normalize_boundary_face_override("z_high", self.z_high),
        }
        face_kinds = {axis: [kind, kind] for axis in _BOUNDARY_AXES}
        for axis, axis_override in axis_overrides.items():
            if axis_override is None:
                continue
            face_kinds[axis][0] = axis_override[0]
            face_kinds[axis][1] = axis_override[1]
        for (axis, side), face_override in face_overrides.items():
            if face_override is None:
                continue
            face_kinds[axis][_BOUNDARY_SIDE_TO_INDEX[side]] = face_override

        normalized_face_kinds = tuple(
            (face_kinds[axis][0], face_kinds[axis][1])
            for axis in _BOUNDARY_AXES
        )
        resolved_kind = _summarize_boundary_kind(normalized_face_kinds)

        uses_pml = any(
            face_kind is BoundaryKind.PML
            for axis_face_kinds in normalized_face_kinds
            for face_kind in axis_face_kinds
        )
        uses_bloch = any(
            face_kind is BoundaryKind.BLOCH
            for axis_face_kinds in normalized_face_kinds
            for face_kind in axis_face_kinds
        )

        if uses_pml and self.num_layers <= 0:
            raise ValueError("PML boundaries require num_layers > 0.")
        if bloch_wavevector == "auto" and not uses_bloch:
            raise ValueError("BoundarySpec.bloch_wavevector='auto' requires at least one Bloch axis.")
        if uses_bloch:
            for axis, axis_index in _BOUNDARY_AXIS_TO_INDEX.items():
                axis_face_kinds = normalized_face_kinds[axis_index]
                if BoundaryKind.BLOCH not in axis_face_kinds:
                    continue
                if axis_face_kinds[0] is not BoundaryKind.BLOCH or axis_face_kinds[1] is not BoundaryKind.BLOCH:
                    raise ValueError(f"{axis}-axis Bloch boundaries must be configured on both faces.")
                # Zero translation phase is the periodic limit of Bloch data.
                # It is valid for explicit vectors, including mixed PML faces,
                # just as it is for a normally incident auto-resolved source.

        for axis, axis_face_kinds in zip(_BOUNDARY_AXES, normalized_face_kinds):
            axis_face_set = set(axis_face_kinds)
            if _PAIR_ONLY_BOUNDARY_KINDS.intersection(axis_face_set) and axis_face_kinds[0] != axis_face_kinds[1]:
                raise ValueError(
                    f"{axis}-axis periodic and Bloch boundaries must use the same kind on both faces."
                )

        if not isinstance(self.mur_abc, MurABC):
            raise TypeError("BoundarySpec.mur_abc must be a MurABC.")
        if self.mode_abc is not None and not isinstance(self.mode_abc, ModeABC):
            raise TypeError("BoundarySpec.mode_abc must be a ModeABC or None.")
        uses_face_kind = lambda kind: any(kind in pair for pair in normalized_face_kinds)
        if not uses_face_kind(BoundaryKind.MUR):
            object.__setattr__(self, "mur_abc", MurABC())
        if uses_face_kind(BoundaryKind.MODE_ABC):
            if self.mode_abc is None:
                object.__setattr__(self, "mode_abc", ModeABC())
        else:
            object.__setattr__(self, "mode_abc", None)

        if not uses_bloch:
            bloch_wavevector = (0.0, 0.0, 0.0)
        if not uses_pml:
            object.__setattr__(self, "num_layers", 0)
            object.__setattr__(self, "strength", 0.0)

        object.__setattr__(self, "kind", resolved_kind)
        object.__setattr__(self, "bloch_wavevector", bloch_wavevector)
        object.__setattr__(self, "x", axis_overrides["x"])
        object.__setattr__(self, "y", axis_overrides["y"])
        object.__setattr__(self, "z", axis_overrides["z"])
        object.__setattr__(self, "x_low", face_overrides[("x", "low")])
        object.__setattr__(self, "x_high", face_overrides[("x", "high")])
        object.__setattr__(self, "y_low", face_overrides[("y", "low")])
        object.__setattr__(self, "y_high", face_overrides[("y", "high")])
        object.__setattr__(self, "z_low", face_overrides[("z", "low")])
        object.__setattr__(self, "z_high", face_overrides[("z", "high")])
        object.__setattr__(self, "_face_kinds", normalized_face_kinds)

    @classmethod
    def pml(cls, num_layers: int = 15, strength: float = 1e6) -> "BoundarySpec":
        require_fixed_input((num_layers, strength), name="BoundarySpec.pml controls")
        return cls(kind=BoundaryKind.PML, num_layers=int(num_layers), strength=float(strength))

    @classmethod
    def none(cls) -> "BoundarySpec":
        return cls(kind=BoundaryKind.NONE, num_layers=0, strength=0.0)

    @classmethod
    def periodic(cls) -> "BoundarySpec":
        return cls(kind=BoundaryKind.PERIODIC)

    @classmethod
    def bloch(cls, wavevector) -> "BoundarySpec":
        return cls(kind=BoundaryKind.BLOCH, bloch_wavevector=wavevector)

    @classmethod
    def pec(cls) -> "BoundarySpec":
        return cls(kind=BoundaryKind.PEC)

    @classmethod
    def pmc(cls) -> "BoundarySpec":
        return cls(kind=BoundaryKind.PMC)

    @classmethod
    def mur(cls, order: int = 1, permittivity: float | None = None) -> "BoundarySpec":
        return cls(kind=BoundaryKind.MUR, mur_abc=MurABC(order=order, permittivity=permittivity))

    @classmethod
    def faces(
        cls,
        *,
        default: BoundaryKind = BoundaryKind.NONE,
        num_layers: int = 0,
        strength: float = 0.0,
        bloch_wavevector: BlochWavevector = (0.0, 0.0, 0.0),
        x: BoundaryAxisOverride | None = None,
        y: BoundaryAxisOverride | None = None,
        z: BoundaryAxisOverride | None = None,
        x_low: BoundaryKind | None = None,
        x_high: BoundaryKind | None = None,
        y_low: BoundaryKind | None = None,
        y_high: BoundaryKind | None = None,
        z_low: BoundaryKind | None = None,
        z_high: BoundaryKind | None = None,
        mur_abc: MurABC | None = None,
        mode_abc: ModeABC | None = None,
    ) -> "BoundarySpec":
        require_fixed_input((num_layers, strength, bloch_wavevector), name="BoundarySpec.faces controls")
        return cls(
            kind=default,
            num_layers=int(num_layers),
            strength=float(strength),
            bloch_wavevector=bloch_wavevector,
            x=x,
            y=y,
            z=z,
            x_low=x_low,
            x_high=x_high,
            y_low=y_low,
            y_high=y_high,
            z_low=z_low,
            z_high=z_high,
            mur_abc=MurABC() if mur_abc is None else mur_abc,
            mode_abc=mode_abc,
        )

    @property
    def face_kinds(
        self,
    ) -> tuple[tuple[BoundaryKind, BoundaryKind], tuple[BoundaryKind, BoundaryKind], tuple[BoundaryKind, BoundaryKind]]:
        return self._face_kinds

    def axis_face_kinds(self, axis: str) -> tuple[BoundaryKind, BoundaryKind]:
        axis_name = str(axis).lower()
        if axis_name not in _BOUNDARY_AXIS_TO_INDEX:
            raise ValueError("axis must be 'x', 'y', or 'z'.")
        return self._face_kinds[_BOUNDARY_AXIS_TO_INDEX[axis_name]]

    def axis_kind(self, axis: str) -> BoundaryKind:
        low_kind, high_kind = self.axis_face_kinds(axis)
        if low_kind is high_kind:
            return low_kind
        return BoundaryKind.MIXED

    def face_kind(self, axis: str, side: str) -> BoundaryKind:
        axis_name = str(axis).lower()
        side_name = str(side).lower()
        if axis_name not in _BOUNDARY_AXIS_TO_INDEX:
            raise ValueError("axis must be 'x', 'y', or 'z'.")
        if side_name not in _BOUNDARY_SIDE_TO_INDEX:
            raise ValueError("side must be 'low' or 'high'.")
        return self._face_kinds[_BOUNDARY_AXIS_TO_INDEX[axis_name]][_BOUNDARY_SIDE_TO_INDEX[side_name]]

    def uses_kind(self, kind: BoundaryKind) -> bool:
        normalized_kind = _normalize_boundary_kind(kind)
        return any(
            face_kind is normalized_kind
            for axis_face_kinds in self._face_kinds
            for face_kind in axis_face_kinds
        )

    def pml_layers_for_face(self, axis: str, side: str) -> int:
        return int(self.num_layers) if self.face_kind(axis, side) is BoundaryKind.PML else 0

    def bloch_phase_factors(
        self,
        domain_range,
    ) -> tuple[complex, complex, complex]:
        if self.bloch_wavevector == "auto":
            raise ValueError(
                "Automatic Bloch phase factors require Simulation.prepare() to resolve the incident wavevector."
            )
        lengths = (
            float(domain_range[1] - domain_range[0]),
            float(domain_range[3] - domain_range[2]),
            float(domain_range[5] - domain_range[4]),
        )
        phases = []
        for axis, length, wave_number in zip(_BOUNDARY_AXES, lengths, self.bloch_wavevector):
            if self.axis_kind(axis) is BoundaryKind.BLOCH:
                phases.append(np.exp(1j * wave_number * length))
            else:
                phases.append(1.0 + 0.0j)
        return tuple(phases)

    def with_faces(
        self,
        *,
        x: BoundaryAxisOverride | None = None,
        y: BoundaryAxisOverride | None = None,
        z: BoundaryAxisOverride | None = None,
        x_low: BoundaryKind | None = None,
        x_high: BoundaryKind | None = None,
        y_low: BoundaryKind | None = None,
        y_high: BoundaryKind | None = None,
        z_low: BoundaryKind | None = None,
        z_high: BoundaryKind | None = None,
    ) -> "BoundarySpec":
        return type(self).faces(
            default=BoundaryKind.NONE,
            num_layers=self.num_layers,
            strength=self.strength,
            bloch_wavevector=self.bloch_wavevector,
            x=self.axis_face_kinds("x") if x is None else x,
            y=self.axis_face_kinds("y") if y is None else y,
            z=self.axis_face_kinds("z") if z is None else z,
            x_low=x_low,
            x_high=x_high,
            y_low=y_low,
            y_high=y_high,
            z_low=z_low,
            z_high=z_high,
            mur_abc=self.mur_abc,
            mode_abc=self.mode_abc,
        )

@dataclass(frozen=True)
class MaterialRegion:
    name: str
    geometry: Any
    density: torch.Tensor
    basis: str = "density"
    bounds: tuple[float, float] = (0.0, 1.0)
    eps_bounds: tuple[float, float] = (1.0, 1.0)
    mu_bounds: tuple[float, float] = (1.0, 1.0)
    filter_radius: float | None = None
    projection_beta: float | None = None
    symmetry: str | None = None

    def __post_init__(self):
        if str(self.basis).lower() != "density":
            raise ValueError("MaterialRegion currently supports basis='density' only.")
        if not isinstance(self.density, torch.Tensor):
            raise TypeError("MaterialRegion.density must be a torch.Tensor.")
        if self.density.ndim != 3:
            raise ValueError("MaterialRegion.density must have shape (Nx, Ny, Nz).")
        if self.symmetry is not None:
            raise UnsupportedFeatureError(
                "MaterialRegion.symmetry is not consumed by any solver; impose design symmetry on the "
                "density with witwin.maxwell.invdes.symmetrize(...) or DensityParameterization and "
                "set symmetry=None.",
                feature="guard:38381bae", method=None
            )
        for field_name in ("bounds", "eps_bounds", "mu_bounds", "filter_radius", "projection_beta"):
            require_fixed_input(getattr(self, field_name), name=f"MaterialRegion.{field_name}")

        density_bounds = (float(self.bounds[0]), float(self.bounds[1]))
        if density_bounds[1] <= density_bounds[0]:
            raise ValueError("MaterialRegion.bounds must satisfy upper > lower.")
        eps_bounds = (float(self.eps_bounds[0]), float(self.eps_bounds[1]))
        mu_bounds = (float(self.mu_bounds[0]), float(self.mu_bounds[1]))
        if self.filter_radius is not None and float(self.filter_radius) <= 0.0:
            raise ValueError("MaterialRegion.filter_radius must be > 0 when provided.")
        if self.projection_beta is not None and float(self.projection_beta) <= 0.0:
            raise ValueError("MaterialRegion.projection_beta must be > 0 when provided.")

        object.__setattr__(self, "basis", "density")
        object.__setattr__(self, "bounds", density_bounds)
        object.__setattr__(self, "eps_bounds", eps_bounds)
        object.__setattr__(self, "mu_bounds", mu_bounds)
        object.__setattr__(
            self,
            "filter_radius",
            None if self.filter_radius is None else float(self.filter_radius),
        )
        object.__setattr__(
            self,
            "projection_beta",
            None if self.projection_beta is None else float(self.projection_beta),
        )


class SceneModule(torch.nn.Module):
    """A PyTorch module whose ``to_scene()`` materializes a standard ``Scene``.

    Recommended pattern: build the static part once and replace only what the parameters drive::

        class Design(mw.SceneModule):
            def __init__(self):
                super().__init__()
                self.density = torch.nn.Parameter(torch.full((32, 32, 1), 0.5))
                self._base = mw.Scene(domain=..., grid=..., boundary=..., structures=(...), sources=(...))
                self._design_box = mw.Box(position=(0, 0, 0), size=(...))

            def to_scene(self):
                return self._base.updated(material_regions=(
                    mw.MaterialRegion(name="design", geometry=self._design_box, density=self.density),))

    ``updated`` shares every static object (structural sharing, about 0.02 ms for 1,000 structures),
    and preparation reuses the compiled material prefix and PML census of the static structures while
    the same objects reappear on the same grid, device and dtype
    (``witwin.maxwell.compiler.static_cache``). Rebuilding the static structures inside ``to_scene()``
    stays correct but compiles them on every forward.
    """

    def to_scene(self) -> "Scene":
        raise NotImplementedError("SceneModule subclasses must implement to_scene().")


def _normalize_subpixel_samples(value) -> tuple[int, int, int]:
    require_fixed_input(value, name="SubpixelSpec.samples")
    if isinstance(value, bool):
        raise TypeError("subpixel_samples must be an int or a length-3 iterable, not bool.")
    if isinstance(value, int):
        if value < 1:
            raise ValueError("subpixel_samples must be >= 1.")
        return (value, value, value)
    if len(value) != 3:
        raise ValueError("subpixel_samples must be an int or a length-3 iterable.")
    normalized = tuple(int(v) for v in value)
    if any(v < 1 for v in normalized):
        raise ValueError("subpixel_samples values must be >= 1.")
    return normalized


class SubpixelAveraging(StrEnum):
    """The per-Yee-component material blend of ``SubpixelSpec.averaging``."""

    ARITHMETIC = "arithmetic"
    POLARIZED = "polarized"
    POLARIZED_TENSOR = "polarized_tensor"
    KOTTKE_EDGE = "kottke_edge"
    KOTTKE_SMOOTH = "kottke_smooth"


class PECTreatment(StrEnum):
    """The FDTD handling of PEC material edges, ``SubpixelSpec.pec``."""

    STAIRCASE = "staircase"
    CONFORMAL = "conformal"


@dataclass(frozen=True)
class SubpixelSpec:
    """Subpixel material-averaging policy.

    ``averaging`` selects the per-Yee-component permittivity/permeability blend:
    ``"arithmetic"`` (scalar volume-fraction mixing) or ``"polarized"``
    (Kottke normal-projection mixing that uses the harmonic mean along the interface
    normal and the arithmetic mean tangentially). ``"polarized_tensor"`` also
    reconstructs the full symmetric electric tensor at isotropic interfaces for
    FDTD only (opt-in); FDFD preparation rejects it. Anisotropic constituents
    retain local diagonal treatment. Both polarized modes require more than one
    sample. ``"kottke_edge"`` uses binary per-edge samples, an in-window first
    moment normal and the physical tensor diagonal (the FDTD and FDFD default).
    Geometry and MaterialRegion tensor fields must be fixed; Structure material
    values may be trainable. Use samples=4 for its reference quadrature.
    ``"kottke_smooth"`` shares that window and blend with curvature-corrected
    fractional coverage, allowing geometry and Structure material-value gradients.
    MaterialRegion density gradients require explicit arithmetic or polarized
    averaging; kottke_smooth is not qualified for them. Use samples=4.
    It is recommended for shape differentiation, with larger errors than kottke_edge
    at sharp high-contrast edges and corners.
    ``pec`` selects how PEC-material
    edges are handled by FDTD: ``"staircase"`` (hard edge) or ``"conformal"``
    (fractional partial-fill edge suppression). The default uses four samples per
    axis and staircase PEC. Geometry gradients require an explicit differentiable mode.
    """

    samples: tuple[int, int, int] = (4, 4, 4)
    averaging: SubpixelAveraging = SubpixelAveraging.KOTTKE_EDGE
    pec: PECTreatment = PECTreatment.STAIRCASE

    def __post_init__(self):
        object.__setattr__(self, "samples", _normalize_subpixel_samples(self.samples))
        averaging = parse_enum(SubpixelAveraging, self.averaging, name="SubpixelSpec.averaging")
        pec = parse_enum(PECTreatment, self.pec, name="SubpixelSpec.pec")
        if averaging in (SubpixelAveraging.POLARIZED, SubpixelAveraging.POLARIZED_TENSOR) and self.samples == (1, 1, 1):
            raise ValueError(
                f"SubpixelSpec(averaging='{averaging}') needs more than one subpixel sample: a single "
                "point sample has a binary fill, so the normal projection would reproduce the "
                "staircase. Pass samples >= 2 (for example samples=4)."
            )
        object.__setattr__(self, "averaging", averaging)
        object.__setattr__(self, "pec", pec)
        if pec is PECTreatment.CONFORMAL:
            warnings.warn(
                'SubpixelSpec(pec="conformal") has a known systematic bias: '
                'non-converging scattering error of about 30-40% on curved PEC. '
                'Use pec="staircase" instead. See docs/memory/conformal-pec-bias.md.',
                UserWarning,
                stacklevel=2,
            )


def _coerce_subpixel(value) -> SubpixelSpec:
    if isinstance(value, SubpixelSpec):
        return value
    return SubpixelSpec(samples=_normalize_subpixel_samples(value), averaging=SubpixelAveraging.ARITHMETIC)


def _normalize_symmetry_entry(axis_value) -> tuple[str, str] | None:
    """Normalize a single axis symmetry declaration to ``(mode, face)`` or ``None``.

    Accepted forms per axis:
      - ``None``: no symmetry on this axis.
      - ``"PEC"`` / ``"PMC"``: image plane on the domain low face (default).
      - ``("PEC", "low")`` / ``("PMC", "high")``: explicit symmetry-plane face.

    ``face`` selects which domain face carries the symmetry plane, i.e. which half
    of the full domain was folded away. ``"low"`` keeps ``[plane, high]`` and mirrors
    downward; ``"high"`` keeps ``[low, plane]`` and mirrors upward.
    """
    if axis_value is None:
        return None
    face = "low"
    mode = axis_value
    if isinstance(axis_value, (tuple, list)):
        if len(axis_value) != 2:
            raise ValueError("symmetry entry tuples must be (mode, face).")
        mode, face = axis_value
        face = str(face).lower()
        if face not in {"low", "high"}:
            raise ValueError("symmetry face must be 'low' or 'high'.")
    label = str(mode).upper()
    if label not in {"PEC", "PMC"}:
        raise ValueError("symmetry entries must be None, 'PEC', 'PMC', or (mode, face).")
    return (label, face)


def _normalize_symmetry(value) -> tuple[tuple[str, str] | None, ...]:
    if value is None:
        return (None, None, None)
    if len(value) != 3:
        raise ValueError("symmetry must contain exactly three axis entries.")
    return tuple(_normalize_symmetry_entry(axis_value) for axis_value in value)


def _component_summary(components: dict[str, torch.Tensor]) -> torch.Tensor:
    return (components["x"] + components["y"] + components["z"]) / 3.0


def _clone_component_map(components: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {axis: tensor.clone() for axis, tensor in components.items()}


def _place_scene_value(value, device, memo):
    """Move nested declarative tensors without mutating the source scene."""
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, torch.Tensor):
        moved = value.to(device=device)
        memo[id(value)] = moved
        return moved
    if isinstance(value, list):
        moved = []
        memo[id(value)] = moved
        moved.extend(_place_scene_value(v, device, memo) for v in value)
        return moved
    if isinstance(value, tuple):
        moved = tuple(_place_scene_value(v, device, memo) for v in value)
        memo[id(value)] = moved
        return moved
    if isinstance(value, dict):
        moved = {}
        memo[id(value)] = moved
        moved.update((k, _place_scene_value(v, device, memo)) for k, v in value.items())
        return moved
    if isinstance(value, MappingProxyType):
        moved = MappingProxyType({k: _place_scene_value(v, device, memo) for k, v in value.items()})
        memo[id(value)] = moved
        return moved
    if type(value).__module__.startswith("witwin.") and hasattr(value, "__dict__") and not isinstance(value, type):
        moved = copy.copy(value)
        memo[id(value)] = moved
        for name, child in vars(value).items():
            object.__setattr__(moved, name, _place_scene_value(child, device, memo))
        return moved
    return value


def place_scene(scene: "Scene", device) -> "Scene":
    """A copy of ``scene`` whose declarative tensors live on ``device``; ``scene`` is untouched."""
    return _place_scene_value(scene, str(torch.device(device)), {})


def _resolve_device(device) -> str:
    requested = "cuda" if device is None else device
    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Scene preparation targets CUDA by default, but torch.cuda.is_available() is False. "
            "Set device='cpu' on the solver config for scene-only or CPU postprocessing workflows."
        )
    return str(resolved)


def _domain_range_from_bounds(
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> tuple[float, float, float, float, float, float]:
    return (
        bounds[0][0],
        bounds[0][1],
        bounds[1][0],
        bounds[1][1],
        bounds[2][0],
        bounds[2][1],
    )


def _require_unique_name(items, item, label):
    if any(existing.name == item.name for existing in items):
        raise ValueError(f"{label} name {item.name!r} is already present in the scene.")


def _require_terminals_inside(domain, terminals, describe):
    for terminal_name, terminal in terminals:
        for axis, coordinate, (lower, upper) in zip(_BOUNDARY_AXES, terminal, domain.bounds):
            if not math.isfinite(coordinate) or coordinate < lower or coordinate > upper:
                raise ValueError(
                    f"{describe} {terminal_name} terminal {axis}={coordinate:g} lies outside "
                    f"the Scene domain [{lower:g}, {upper:g}]."
                )


def _require_coaxial_inside(domain, item):
    outer = 0.5 * item.outer_diameter
    normal = "xyz".index(item.normal_axis)
    for index, (axis, coordinate, (lower, upper)) in enumerate(zip(_BOUNDARY_AXES, item.center, domain.bounds)):
        reach = 0.0 if index == normal else outer
        if coordinate - reach < lower or coordinate + reach > upper:
            raise ValueError(f"{type(item).__name__} {item.name!r} annulus leaves the Scene domain along {axis}.")


def _check_background(scene, value):
    from witwin.core import PhysicalMaterial
    from .layers import LayerStack

    if value is not None and not isinstance(value, (PhysicalMaterial, LayerStack)):
        raise TypeError("Scene background must be a homogeneous material, a planar LayerStack or None")
    return value


def _check_structures(scene, value):
    structures = tuple(value or ())
    if any(not isinstance(structure, Structure) for structure in structures):
        raise TypeError("Maxwell Scene structures must be witwin.core.Structure instances.")
    return structures


def _check_thin_wires(scene, value):
    wires = []
    for wire in value or ():
        if not isinstance(wire, ThinWire):
            raise TypeError("Maxwell Scene thin wires must be ThinWire instances.")
        _require_unique_name(wires, wire, "Thin-wire")
        wires.append(wire)
    return tuple(wires)


def _check_ports(scene, value):
    ports = []
    for port in value or ():
        if isinstance(port, CoaxialLumpedPort):
            _require_unique_name(ports, port, "Port")
            _require_coaxial_inside(scene.domain, port)
        elif isinstance(port, TerminalWavePort):
            _require_unique_name(ports, port, "Port")
            port = _resolve_terminal_wave_port(port, scene.structures)
        elif not isinstance(port, (ModePort, LumpedPort, TerminalPort, WavePort)):
            raise TypeError(
                "Maxwell Scene ports must be ModePort, LumpedPort, TerminalPort, or WavePort instances."
            )
        else:
            _require_unique_name(ports, port, "Port")
            if isinstance(port, TerminalPort) and port.wire_binding is None:
                port = _resolve_terminal_port(port, scene.structures, scene.domain.bounds)
            if isinstance(port, LumpedPort) and port.wire_binding is None:
                _require_terminals_inside(
                    scene.domain, (("positive", port.positive), ("negative", port.negative)),
                    f"LumpedPort {port.name!r}",
                )
        ports.append(port)
    return tuple(ports)


def _check_networks(scene, value):
    networks = []
    ports_by_name = {port.name: port for port in scene.ports}
    connected = set()
    for network in value or ():
        if not isinstance(network, NetworkBlock):
            raise TypeError("Maxwell Scene networks must be NetworkBlock instances.")
        _require_unique_name(networks, network, "Network")
        for port_name in network.connected_port_names:
            if port_name not in ports_by_name:
                raise ValueError(f"Network {network.name!r} references unknown Scene port {port_name!r}.")
            port = ports_by_name[port_name]
            if isinstance(port, WavePort):
                raise ValueError(
                    f"Network {network.name!r} cannot connect to WavePort {port_name!r}: "
                    "an embedded state-space network couples through a scalar "
                    "voltage/current terminal on a single lumped Yee edge "
                    "(LumpedPort or resolved TerminalPort), but a WavePort is a "
                    "modal port defined by a cross-sectional mode-overlap field "
                    "pattern with no scalar time-domain terminal (V, I) contract. "
                    "This is a missing design contract, not a bug; use a LumpedPort "
                    "or TerminalPort terminal for embedded-network connections."
                )
            if not isinstance(port, (LumpedPort, TerminalPort)):
                raise TypeError(
                    f"Network {network.name!r} connections require LumpedPort or TerminalPort "
                    f"targets; {port_name!r} is {type(port).__name__}."
                )
            if port_name in connected:
                raise ValueError(f"Scene port {port_name!r} is already connected to another network.")
        connected.update(network.connected_port_names)
        networks.append(network)
    return tuple(networks)


def _check_lumped_elements(scene, value):
    elements = []
    for element in value or ():
        if isinstance(element, CoaxialLumpedResistor):
            _require_unique_name(elements, element, "Lumped element")
            _require_coaxial_inside(scene.domain, element)
            elements.append(element)
            continue
        if not isinstance(element, (Resistor, Capacitor, Inductor)):
            raise TypeError("Maxwell Scene lumped elements must be Resistor, Capacitor, or Inductor instances.")
        _require_unique_name(elements, element, "Lumped element")
        if isinstance(element.positive, CircuitNode) or isinstance(element.negative, CircuitNode):
            raise TypeError(
                "Scene-local lumped elements require 3D coordinate terminals; "
                "add circuit-node elements through Circuit.add(...)."
            )
        _require_terminals_inside(
            scene.domain, (("positive", element.positive), ("negative", element.negative)),
            f"Lumped element {element.name!r}",
        )
        differing_axes = sum(
            not math.isclose(positive, negative, rel_tol=0.0, abs_tol=1.0e-12)
            for positive, negative in zip(element.positive, element.negative)
        )
        if differing_axes != 1:
            raise ValueError(f"Lumped element {element.name!r} terminals must define an axis-aligned path.")
        elements.append(element)
    return tuple(elements)


def _check_circuits(scene, value):
    circuits = []
    for circuit in value or ():
        if not isinstance(circuit, Circuit):
            raise TypeError("Maxwell Scene circuits must be Circuit instances.")
        if any(existing.name.casefold() == circuit.name.casefold() for existing in circuits):
            raise ValueError(f"Circuit name {circuit.name!r} is already present in the scene.")
        circuits.append(circuit)
    return tuple(circuits)


def _check_internal_absorbers(scene, value):
    absorbers = []
    for absorber in value or ():
        if not isinstance(absorber, InternalAbsorber):
            raise TypeError("Scene.internal_absorbers entries must be InternalAbsorber instances.")
        if absorber.name is not None:
            _require_unique_name(absorbers, absorber, "Internal absorber")
        absorbers.append(absorber)
    return tuple(absorbers)


def _check_electrostatic_terminals(scene, value):
    from .electrostatic.api import ElectrostaticTerminal

    terminals = []
    for terminal in value or ():
        # Terminals never enter Scene.ports, so the RF port compiler cannot reinterpret
        # an equipotential constraint as an impedance/power-wave port.
        if not isinstance(terminal, ElectrostaticTerminal):
            raise TypeError("Scene.electrostatic_terminals entries must be ElectrostaticTerminal instances.")
        _require_unique_name(terminals, terminal, "Electrostatic terminal")
        terminals.append(terminal)
    return tuple(terminals)


def _check_charge_densities(scene, value):
    from .electrostatic.api import ChargeDensity

    densities = tuple(value or ())
    if any(not isinstance(density, ChargeDensity) for density in densities):
        raise TypeError("Scene.charge_densities entries must be ChargeDensity instances.")
    return densities


def _check_coordinates(scene, value):
    if value is None:
        return None
    from .coordinates import BendCoordinates

    if not isinstance(value, BendCoordinates):
        raise TypeError("Scene.coordinates must be BendCoordinates or None.")
    value.validate_bounds(scene.domain.bounds)
    return value


def _check_metadata(scene, value):
    return MappingProxyType(dict(value or {}))


_SCENE_COLLECTIONS = frozenset((  # thin_wires stays a tuple, as it always was
    "structures", "sources", "monitors", "ports", "networks", "lumped_elements", "circuits",
    "material_regions", "electrostatic_terminals", "charge_densities", "internal_absorbers"))
# Field normalizers in dependency order: a normalizer may read the fields above it.
_SCENE_CHECKS = {
    "domain": lambda scene, value: (
        Domain.from_domain_range((-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)) if value is None else value),
    "grid": lambda scene, value: GridSpec.uniform(0.05) if value is None else value,
    "boundary": lambda scene, value: BoundarySpec.none() if value is None else value,
    "background": _check_background,
    "structures": _check_structures,
    "thin_wires": _check_thin_wires,
    "sources": lambda scene, value: tuple(value or ()),
    "monitors": lambda scene, value: tuple(value or ()),
    "ports": _check_ports,
    "networks": _check_networks,
    "lumped_elements": _check_lumped_elements,
    "circuits": _check_circuits,
    "material_regions": lambda scene, value: tuple(value or ()),
    "electrostatic_terminals": _check_electrostatic_terminals,
    "charge_densities": _check_charge_densities,
    "internal_absorbers": _check_internal_absorbers,
    "subpixel": lambda scene, value: _coerce_subpixel(value),
    "symmetry": lambda scene, value: _normalize_symmetry(value),
    "coordinates": _check_coordinates,
    "metadata": _check_metadata,
}
# Cross-object rules: a field is re-checked by ``Scene.updated`` when one it reads changes.
_SCENE_READS = {
    "ports": ("domain", "structures"),
    "networks": ("ports",),
    "lumped_elements": ("domain",),
    "coordinates": ("domain",),
}


@dataclass(frozen=True, eq=False, repr=False)
class Scene:
    """Immutable declarative 3D scene definition for simulation backends.

    Collections are tuples. Build a scene with the constructor and derive variants with
    :meth:`updated`, which returns a new ``Scene`` and shares every unchanged object.
    Equality and hashing are by identity.
    """

    domain: Domain | None = None
    grid: GridSpec | None = None
    boundary: BoundarySpec | None = None
    _: KW_ONLY
    background: Any = None
    structures: tuple[Structure, ...] = ()
    thin_wires: tuple[ThinWire, ...] = ()
    sources: tuple = ()
    monitors: tuple = ()
    ports: tuple = ()
    networks: tuple = ()
    lumped_elements: tuple = ()
    circuits: tuple = ()
    material_regions: tuple[MaterialRegion, ...] = ()
    electrostatic_terminals: tuple = ()
    charge_densities: tuple = ()
    internal_absorbers: tuple = ()
    subpixel: SubpixelSpec = SubpixelSpec()
    symmetry: tuple = (None, None, None)
    coordinates: Any = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self):
        self._check(_SCENE_CHECKS)

    # The read-only metadata view does not pickle; copies and pickles carry a plain dict.
    def __getstate__(self):
        return {**self.__dict__, "metadata": dict(self.metadata)}

    def __setstate__(self, state):
        for name, value in state.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "metadata", MappingProxyType(self.metadata))

    def _check(self, names):
        for name, check in _SCENE_CHECKS.items():
            if name in names:
                object.__setattr__(self, name, check(self, getattr(self, name)))

    def updated(self, **fields) -> "Scene":
        """A new ``Scene`` with ``fields`` replaced; unchanged objects are shared, not copied.

        Only the replaced fields and the cross-object rules that read them are validated again.
        """
        unknown = sorted(set(fields).difference(_SCENE_CHECKS))
        if unknown:
            raise TypeError(f"Scene.updated() got unknown field(s) {unknown}; Scene fields are {list(_SCENE_CHECKS)}.")
        scene = object.__new__(Scene)
        for name in _SCENE_CHECKS:
            object.__setattr__(scene, name, fields[name] if name in fields else getattr(self, name))
        dirty = set(fields)
        for name, reads in _SCENE_READS.items():
            if dirty.intersection(reads):
                dirty.add(name)
        scene._check(dirty)
        return scene

    def to_dict(self, arrays: dict | None = None) -> dict:
        """Serialize the scene as a strict-JSON ``witwin.scene/2`` dictionary.

        Every field and every registered scene object is written; tensors keep their dtype and
        trainable ones record ``requires_grad``. With ``arrays``, tensors over 4096 elements are
        stored there by document path instead of inline. An object outside the schema (a
        callable, an unregistered type) raises :class:`~witwin.maxwell.SchemaError`.
        """
        from .io.scene_dict import scene_to_dict

        return scene_to_dict(self, arrays=arrays)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], arrays: Mapping[str, Any] | None = None) -> "Scene":
        """Rebuild a ``witwin.scene/2`` dictionary through the public constructors.

        Tensors come back on the CPU (the solver config places them); a trainable tensor comes
        back as a fresh leaf with ``requires_grad``.
        """
        from .io.scene_dict import scene_from_dict

        return scene_from_dict(d, arrays=arrays)

    def boundary_face_kind(self, axis: str, side: str) -> BoundaryKind:
        return self.boundary.face_kind(axis, side)

    def pml_thickness_for_face(self, axis: str, side: str) -> int:
        axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        symmetry = self.symmetry[axis_index]
        if symmetry is not None and symmetry[1] == side:
            return 0
        return self.boundary.pml_layers_for_face(axis, side)

    @property
    def has_symmetry(self) -> bool:
        return any(mode is not None for mode in self.symmetry)

    def resolved_sources(self):
        resolved = []
        for source in self.sources:
            # Declarative sources that lower to a core source (e.g. an ESD
            # terminal injection) expose ``resolve(scene)``; expand them here so
            # downstream compilers only see core source primitives.
            if hasattr(source, "resolve") and callable(getattr(source, "resolve")):
                resolved.append(source.resolve(self))
            else:
                resolved.append(source)
        for port in self.ports:
            if isinstance(port, ModePort):
                source = port.to_mode_source()
                if source is not None:
                    resolved.append(source)
        return resolved

    def resolved_monitors(self):
        resolved = []
        for monitor in self.monitors:
            if hasattr(monitor, "resolved_monitors"):
                resolved.extend(monitor.resolved_monitors())
            else:
                resolved.append(monitor)
        for port in self.ports:
            if isinstance(port, ModePort):
                resolved.append(port.to_mode_monitor())
        return resolved


def _axis_nodes64(
    grid: GridSpec,
    axis: str,
    axis_lo: float,
    axis_hi: float,
) -> np.ndarray:
    """Yee node (cell-boundary) coordinates spanning ``[axis_lo, axis_hi]`` inclusive.

    ``count`` cells always require ``count + 1`` nodes, matching the
    ``GridSpec.custom`` convention below. This holds for periodic/bloch axes
    too: the runtime keeps the wrap node explicitly (the last node duplicates
    the first, see ``_axis_dual_spacing64`` and the update kernels), so the
    node array is endpoint-inclusive for every boundary kind.
    """
    coords = grid.axis_coords(axis)
    if coords is None:
        step = float(getattr(grid, f"d{axis}"))
        count = max(1, int(math.ceil((float(axis_hi) - float(axis_lo)) / step)))
        return np.linspace(float(axis_lo), float(axis_hi), count + 1, endpoint=True, dtype=np.float64)
    span = float(axis_hi) - float(axis_lo)
    tolerance = 1e-9 * span
    if abs(float(coords[0]) - float(axis_lo)) > tolerance or abs(float(coords[-1]) - float(axis_hi)) > tolerance:
        raise ValueError(
            f"GridSpec.custom {axis}_coords must span the domain exactly along axis '{axis}'; "
            f"build Domain from ({float(coords[0]):g}, {float(coords[-1]):g})."
        )
    return coords


def _axis_dual_spacing64(primal: np.ndarray, boundary_kind: BoundaryKind) -> np.ndarray:
    dual = np.empty(primal.size + 1, dtype=np.float64)
    dual[1:-1] = 0.5 * (primal[:-1] + primal[1:])
    if boundary_kind in (BoundaryKind.PERIODIC, BoundaryKind.BLOCH):
        # Distance from the wrap image of the last half-point to the first half-point.
        wrap = 0.5 * (primal[0] + primal[-1])
        dual[0] = wrap
        dual[-1] = wrap
    else:
        # PMC mirror distance across the boundary node; placeholder for PEC/none/PML.
        dual[0] = primal[0]
        dual[-1] = primal[-1]
    return dual


def _build_axis_grid64(
    grid: GridSpec,
    axis: str,
    axis_lo: float,
    axis_hi: float,
    boundary_kind: str,
    pml_layers: tuple[int, int] = (0, 0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nodes = _axis_nodes64(grid, axis, axis_lo, axis_hi)
    low_layers, high_layers = (int(value) for value in pml_layers)
    if low_layers or high_layers:
        if nodes.size < 2:
            raise ValueError(f"Cannot extend the {axis}-axis PML from fewer than two physical nodes.")
        low_step = float(nodes[1] - nodes[0])
        high_step = float(nodes[-1] - nodes[-2])
        low_nodes = nodes[0] - low_step * np.arange(low_layers, 0, -1, dtype=np.float64)
        high_nodes = nodes[-1] + high_step * np.arange(1, high_layers + 1, dtype=np.float64)
        nodes = np.concatenate((low_nodes, nodes, high_nodes))
    primal = np.diff(nodes)
    half = nodes[:-1] + 0.5 * primal
    dual = _axis_dual_spacing64(primal, boundary_kind)
    return nodes, primal, half, dual


class _PMLExtrudedGeometry(GeometryBase):
    """Compile-only normal continuation of a prismatic interface section."""

    kind = "pml_extrusion"

    def __init__(self, base, faces):
        # Keep tensor leaves visible to the compiler's trainable-geometry gate.
        self.__dict__.update({k: v for k, v in vars(base).items() if torch.is_tensor(v)})
        self._pml_base = base
        self._pml_faces = faces

    def signed_distance(self, xx, yy, zz):
        coords = [xx, yy, zz]
        for axis, side, interface, _outer in self._pml_faces:
            coords[axis] = (coords[axis].clamp(min=interface) if side == "low"
                            else coords[axis].clamp(max=interface))
        return self._pml_base.signed_distance(*coords)

    def to_mesh(self, segments=16):
        # Periodic-image bounds only need transverse extents: PML and periodic
        # faces cannot share an axis. This wrapper is never a public mesh.
        return self._pml_base.to_mesh(segments=segments)


def _pml_plane_coords(scene, axis, value):
    others = [i for i in range(3) if i != axis]
    a, b = torch.meshgrid(*(getattr(scene, "xyz"[i]) for i in others), indexing="ij")
    coords = [None, None, None]
    coords[axis] = torch.ones_like(a) * value
    coords[others[0]], coords[others[1]] = a, b
    return tuple(coords)


def _pml_conservative_bounds(geometry, reference):
    """Cheap enclosing bounds only; unknown geometry keeps the exact census."""
    from witwin.core import Mesh, Sphere

    if isinstance(geometry, Box):
        # A circumsphere encloses every rotation without tessellation.
        radius = torch.linalg.vector_norm(geometry.size.to(reference) / 2)
    elif isinstance(geometry, Sphere):
        radius = geometry.radius.to(reference).abs()
    elif isinstance(geometry, Mesh):
        if geometry.fill_mode == "surface" or (geometry.fill_mode == "auto" and not geometry.is_watertight):
            # Surface bands extend beyond the vertex AABB.
            return None
        vertices, _ = geometry.to_mesh()
        return vertices.to(reference).amin(0), vertices.to(reference).amax(0)
    else:
        return None
    center = geometry.position.to(reference)
    return center - radius, center + radius


def _pml_contact_faces(scene, geometry):
    """Census flagged pairs, with an exact adjacent-plane prism decision."""
    from .geometry.surface_mesh import SurfaceMesh

    def occupied(value, axis):
        coords = _pml_plane_coords(scene, axis, value)
        if isinstance(geometry, SurfaceMesh):
            # Zero-volume surfaces expose exact contact, not occupancy SDFs.
            points = torch.stack(coords, dim=-1)
            _, contact = geometry.classify_query_points(points.reshape(-1, 3))
            return contact.reshape(points.shape[:-1])
        return geometry.signed_distance(*coords) <= 0

    flagged, rejected = [], []
    with torch.no_grad():
        bounds = _pml_conservative_bounds(geometry, scene.x)
        if bounds is not None:
            # Retain near-boundary cases for the authoritative SDF predicate,
            # including rounding in transforms and the node-plane coordinate.
            slack = 64 * torch.finfo(scene.x.dtype).eps * torch.stack(bounds).abs().amax().clamp(min=1)
        for axis, name in enumerate("xyz"):
            nodes = getattr(scene, name)
            for side, bound, outer in (("low", scene.physical_domain_range[2*axis], nodes[0]),
                                       ("high", scene.physical_domain_range[2*axis+1], nodes[-1])):
                if scene.scene.boundary.face_kind(name, side) != "pml":
                    continue
                index = int((nodes - bound).abs().argmin())
                interface = nodes[index]
                if bounds is not None and bool(
                    (interface < bounds[0][axis] - slack) | (interface > bounds[1][axis] + slack)
                ):
                    continue
                contact = occupied(interface, axis)
                if not bool(contact.any()):
                    continue
                outside = ~occupied(outer, axis)
                if not bool((contact & outside).any()):
                    continue
                inner = nodes[index + (1 if side == "low" else -1)]
                inside = occupied(inner, axis)
                face = (axis, side, interface, outer)
                flagged.append(face)
                if not torch.equal(contact, inside):
                    rejected.append(face)
    return flagged, rejected


def _pml_enlarged_box(scene, geometry, faces):
    lower, upper = geometry.position - geometry.size / 2, geometry.position + geometry.size / 2
    lows, highs = list(lower.unbind()), list(upper.unbind())
    for axis, side, _interface, outer in faces:
        nodes = getattr(scene, "xyz"[axis])
        # One full boundary cell beyond the last node also covers Yee samples.
        margin = (nodes[1] - nodes[0]) if side == "low" else (nodes[-1] - nodes[-2])
        target = (outer - margin if side == "low" else outer + margin).to(geometry.position)
        (lows if side == "low" else highs)[axis] = target
    lo, hi = torch.stack(lows), torch.stack(highs)
    changed = {face[0] for face in faces}
    position = torch.stack([(lo[i] + hi[i]) / 2 if i in changed else geometry.position[i] for i in range(3)])
    size = torch.stack([hi[i] - lo[i] if i in changed else geometry.size[i] for i in range(3)])
    return Box(position=position, size=size, rotation=geometry.rotation, device=geometry.device)


def _pml_density_is_prismatic(scene, region, density, face):
    axis, side, interface, _ = face
    nodes = getattr(scene, "xyz"[axis])
    index = int((nodes - interface).abs().argmin())
    inner = nodes[index + (1 if side == "low" else -1)]
    geometry = region.geometry
    lower = geometry.position[axis] - geometry.size[axis] / 2
    planes = torch.stack((interface, inner)).to(lower)
    # align_corners=False puts native density texels at cell centres. Compare
    # every native slice supporting the two planes, before interpolation can
    # introduce rounding differences between identical density columns.
    texels = ((planes - lower) / geometry.size[axis] * density.shape[axis] - .5).clamp(
        0, density.shape[axis] - 1)
    first, last = int(texels.min().floor()), int(texels.max().ceil())
    slices = density.narrow(axis, first, last - first + 1)
    return torch.equal(slices, slices.narrow(axis, 0, 1).expand_as(slices))


def _pml_geometry_exclusion(geometry):
    """Preserve geometry types consumed by sheet and mesh-specific compilers."""
    from witwin.core import BooleanGeometry, Mesh
    from .geometry.planar_pec import PlanarPEC
    from .geometry.surface_mesh import SurfaceMesh

    if isinstance(geometry, (PlanarPEC, SurfaceMesh)) or (
        isinstance(geometry, Mesh) and geometry.fill_mode == "surface"
    ):
        return "sheet"
    if isinstance(geometry, Mesh):
        return "mesh geometry"
    if isinstance(geometry, BooleanGeometry):
        reasons = [_pml_geometry_exclusion(child) for child in geometry.geometries]
        # A sheet leaf may not implement an occupancy SDF at all.
        if "sheet" in reasons:
            return "sheet"
        if "mesh geometry" in reasons:
            return "mesh geometry"
    return None


def _pml_surface_impedance_material(material):
    """Use the compiler's conductor types and material capability flags."""
    from .copper_foil import CopperFoil
    from .media import LossyMetalMedium, SurfaceImpedanceMedium

    # PEC/PMC have no surface-impedance layout and remain extrudable.
    return isinstance(material, (LossyMetalMedium, SurfaceImpedanceMedium, CopperFoil)) or any(
        bool(getattr(material, flag, False))
        for flag in ("is_surface_impedance", "is_lossy_metal", "is_copper_foil")
    )


def _extrude_pml_contacts(scene):
    """Replace only census-flagged prepared assignments, preserving their order; returns the records."""
    from witwin.core import BooleanGeometry

    from .compiler.static_cache import GEOMETRY_MEMO

    records = []
    structures, material_regions = list(scene.scene.structures), list(scene.scene.material_regions)
    # Everything _pml_contact_faces reads besides the geometry: the census of a static geometry is
    # reused across preparations of the same grid (SceneModule structural sharing).
    grid_key = (str(scene.x.device), scene.x.dtype, scene.x_nodes64.tobytes(), scene.y_nodes64.tobytes(),
                scene.z_nodes64.tobytes(), tuple(float(v) for v in scene.physical_domain_range),
                tuple(scene.scene.boundary.face_kind(a, s) for a in "xyz" for s in ("low", "high")))
    for collection, density_region in ((structures, False), (material_regions, True)):
        for index, owner in enumerate(collection):
            geometry = owner.geometry
            exclusion = _pml_geometry_exclusion(geometry)
            record = {"material_region_index" if density_region else "structure_index": index,
                      "name": owner.name}
            if exclusion == "sheet" and isinstance(geometry, BooleanGeometry):
                # A Boolean sheet leaf may not implement an occupancy SDF.
                continue
            faces, rejected = (list(found) for found in GEOMETRY_MEMO.get(
                "pml_contact_faces", geometry, grid_key, lambda: _pml_contact_faces(scene, geometry)))
            if not faces:
                continue
            if not density_region and _pml_surface_impedance_material(owner.material):
                names = ["xyz"[a] + "-" + s for a, s, *_ in faces]
                warnings.warn(
                    f"PML contact for structure {index} ({owner.name!r}) on faces {', '.join(names)}: "
                    "surface-impedance conductor reaches a PML interface; not extruded",
                    RuntimeWarning, stacklevel=3,
                )
                records.append(dict(record, faces=names, kind="skipped_surface_impedance"))
                continue
            if exclusion == "sheet":
                records.append(dict(record, faces=["xyz"[a] + "-" + s for a, s, *_ in faces],
                                    kind="skipped_sheet"))
                continue
            if exclusion == "mesh geometry":
                warnings.warn(
                    f"PML contact for structure {index} ({owner.name!r}) uses mesh geometry; "
                    "not extruding this assignment.", RuntimeWarning, stacklevel=3,
                )
                continue
            if density_region and not rejected:
                from .compiler.materials import _prepare_material_region_density

                with torch.no_grad():
                    density = _prepare_material_region_density(scene, owner)
                    for face in faces:
                        if not _pml_density_is_prismatic(scene, owner, density, face):
                            rejected.append(face)
            if rejected:
                names = ", ".join("xyz"[a] + "-" + s for a, s, *_ in rejected)
                warnings.warn(
                    f"PML contact for {'MaterialRegion' if density_region else 'structure'} "
                    f"{index} ({owner.name!r}) is non-prismatic on faces {names}; not extruding this assignment.",
                    RuntimeWarning, stacklevel=3,
                )
                continue
            rotation = getattr(geometry, "rotation", None)
            from witwin.core.geometry.base import _rotation_quaternion

            is_box = isinstance(geometry, Box) and (rotation is None or torch.equal(
                _rotation_quaternion(rotation).abs(), rotation.new_tensor((1., 0., 0., 0.))))
            replacement = copy.copy(owner)
            object.__setattr__(replacement, "geometry", _pml_enlarged_box(scene, geometry, faces)
                               if is_box else _PMLExtrudedGeometry(geometry, faces))
            if density_region:
                # Border texture sampling replicates the edge slice without
                # rescaling the interior design grid or its filter stencil.
                object.__setattr__(replacement, "_pml_density_geometry", geometry)
            collection[index] = replacement
            records.append({"structure_index" if not density_region else "material_region_index": index,
                            "name": owner.name, "faces": ["xyz"[a] + "-" + s for a, s, *_ in faces],
                            "kind": "density" if density_region else "box" if is_box else "wrapper"})
    scene.scene = _prepared_declarations(scene.scene, structures=tuple(structures),
                                         material_regions=tuple(material_regions))
    return records


def _prepared_declarations(scene: Scene, **fields) -> Scene:
    """``scene`` with the preparation outputs in ``fields`` replaced.

    The outputs (resolved grid, PML extrusions, placed coordinates) derive from validated declarations, so
    this skips ``Scene.updated``: ports keep their resolution against the declared structures instead of
    being resolved again against the extruded ones.
    """
    derived = object.__new__(Scene)
    for name in _SCENE_CHECKS:
        object.__setattr__(derived, name, fields[name] if name in fields else getattr(scene, name))
    return derived


class PreparedScene:
    """Internal solver-ready form of a ``Scene``: prepared declarations, grid state and compile caches.

    ``scene`` holds the declarations the runtimes read: the input scene with the resolved grid, the PML
    extrusions and the coordinates on ``device``. ``_public_scene`` is the input scene. The grid arrays, cell
    counts and material caches belong to the prepared scene; the ``compile_*`` functions of
    ``witwin.maxwell.compiler`` take it as their first argument.
    """

    def __init__(self, scene: Scene, *, device=None, lazy_meshgrid: bool = True, material_dtype=torch.float32):
        if material_dtype not in (torch.float32, torch.float64):
            raise ValueError("Prepared material_dtype must be float32 or float64.")
        self._material_dtype = material_dtype
        from .sources import MaxwellPlaneModes
        if scene.background is not None or any(isinstance(s, MaxwellPlaneModes) for s in scene.sources):
            raise UnsupportedFeatureError('Grid preparation does not yet support homogeneous background declarations or physical Maxwell mode sources',
                                          feature="background.homogeneous", method=None)
        self.device = _resolve_device(device)
        self.lazy_meshgrid = bool(lazy_meshgrid)
        self._public_scene = scene
        declared = _prepared_declarations(
            scene, coordinates=None if scene.coordinates is None else scene.coordinates.to(self.device))
        self.physical_domain_range = _domain_range_from_bounds(declared.domain.bounds)

        if declared.grid.is_auto:
            from .fdtd.meshing import resolve_auto_grid

            declared = _prepared_declarations(declared, grid=GridSpec.custom(*resolve_auto_grid(declared)))
        self.scene = declared

        x_start, x_end, y_start, y_end, z_start, z_end = self.physical_domain_range
        self.x_nodes64, self.dx_primal64, self.x_half64, self.dx_dual64 = _build_axis_grid64(
            declared.grid,
            "x",
            x_start,
            x_end,
            declared.boundary.axis_kind("x"),
            (declared.pml_thickness_for_face("x", "low"), declared.pml_thickness_for_face("x", "high")),
        )
        self.y_nodes64, self.dy_primal64, self.y_half64, self.dy_dual64 = _build_axis_grid64(
            declared.grid,
            "y",
            y_start,
            y_end,
            declared.boundary.axis_kind("y"),
            (declared.pml_thickness_for_face("y", "low"), declared.pml_thickness_for_face("y", "high")),
        )
        self.z_nodes64, self.dz_primal64, self.z_half64, self.dz_dual64 = _build_axis_grid64(
            declared.grid,
            "z",
            z_start,
            z_end,
            declared.boundary.axis_kind("z"),
            (declared.pml_thickness_for_face("z", "low"), declared.pml_thickness_for_face("z", "high")),
        )
        computational_bounds = []
        for axis_index, (axis, nodes, primal) in enumerate(
            (
                ("x", self.x_nodes64, self.dx_primal64),
                ("y", self.y_nodes64, self.dy_primal64),
                ("z", self.z_nodes64, self.dz_primal64),
            )
        ):
            low_layers = declared.pml_thickness_for_face(axis, "low")
            high_layers = declared.pml_thickness_for_face(axis, "high")
            physical_lo, physical_hi = declared.domain.bounds[axis_index]
            low_step = float(primal[low_layers]) if low_layers < primal.size else float(primal[0])
            high_index = primal.size - high_layers - 1
            high_step = float(primal[max(high_index, 0)])
            computational_bounds.append(
                (
                    float(physical_lo) - low_layers * low_step,
                    float(physical_hi) + high_layers * high_step,
                )
            )
        self.domain_range = _domain_range_from_bounds(computational_bounds)
        self.x = torch.tensor(self.x_nodes64, dtype=material_dtype, device=self.device)
        self.y = torch.tensor(self.y_nodes64, dtype=material_dtype, device=self.device)
        self.z = torch.tensor(self.z_nodes64, dtype=material_dtype, device=self.device)
        self.x_half = torch.tensor(self.x_half64, dtype=material_dtype, device=self.device)
        self.y_half = torch.tensor(self.y_half64, dtype=material_dtype, device=self.device)
        self.z_half = torch.tensor(self.z_half64, dtype=material_dtype, device=self.device)
        self.pml_extrusions = _extrude_pml_contacts(self)
        self.scene = _prepared_declarations(
            self.scene, metadata=MappingProxyType({**declared.metadata, "pml_extrusions": self.pml_extrusions}))

        self._xx = None
        self._yy = None
        self._zz = None
        self._material_model_cache = {}
        self._gyromagnetic_layout_cache = {}
        self._permittivity_components = None
        self._permeability_components = None
        self._permittivity = None
        self._permeability = None
        if not self.lazy_meshgrid:
            self._ensure_meshgrid()

        self.Nx = len(self.x)
        self.Ny = len(self.y)
        self.Nz = len(self.z)
        self.N_total = self.Nx * self.Ny * self.Nz
        self.N_total_nodes = self.N_total
        self.Nx_ex, self.Ny_ex, self.Nz_ex = self.Nx - 1, self.Ny, self.Nz
        self.Nx_ey, self.Ny_ey, self.Nz_ey = self.Nx, self.Ny - 1, self.Nz
        self.Nx_ez, self.Ny_ez, self.Nz_ez = self.Nx, self.Ny, self.Nz - 1
        self.N_ex = self.Nx_ex * self.Ny_ex * self.Nz_ex
        self.N_ey = self.Nx_ey * self.Ny_ey * self.Nz_ey
        self.N_ez = self.Nx_ez * self.Ny_ez * self.Nz_ez
        self.N_vector_total = self.N_ex + self.N_ey + self.N_ez

    def _ensure_meshgrid(self):
        if self._xx is None or self._yy is None or self._zz is None:
            self._xx, self._yy, self._zz = torch.meshgrid(self.x, self.y, self.z, indexing="ij")

    def _invalidate_material_cache(self):
        """Drop the compiled material tables; ``Result.release_solver`` frees them and material access recompiles."""
        self._material_model_cache = {}
        self._gyromagnetic_layout_cache = {}
        self._permittivity_components = None
        self._permeability_components = None
        self._permittivity = None
        self._permeability = None

    def _has_dynamic_materials(self) -> bool:
        if self.scene.material_regions:
            return True
        for structure in self.scene.structures:
            material = getattr(structure, "material", None)
            perturbation = getattr(material, "perturbation", None)
            if isinstance(perturbation, torch.Tensor) and perturbation.requires_grad:
                return True
        return False

    def _has_trainable_geometry(self) -> bool:
        for structure in self.scene.structures:
            geometry = getattr(structure, "geometry", None)
            if geometry is None:
                continue
            for value in vars(geometry).values():
                if isinstance(value, torch.Tensor) and value.requires_grad:
                    return True
        return False

    @property
    def xx(self):
        self._ensure_meshgrid()
        return self._xx

    @property
    def yy(self):
        self._ensure_meshgrid()
        return self._yy

    @property
    def zz(self):
        self._ensure_meshgrid()
        return self._zz

    @property
    def X(self):
        return self.xx

    @property
    def Y(self):
        return self.yy

    @property
    def Z(self):
        return self.zz

    def release_meshgrid(self):
        self._xx = None
        self._yy = None
        self._zz = None

    def _scalar_grid_step(self, value: float | None) -> float:
        if self.scene.grid.is_custom or self.scene.grid.is_auto:
            raise ValueError(
                "scalar spacing undefined on a nonuniform grid; use scene.x/y/z, x_half, or grid.min_spacing."
            )
        return float(value)

    @property
    def dx(self) -> float:
        return self._scalar_grid_step(self.scene.grid.dx)

    @property
    def dy(self) -> float:
        return self._scalar_grid_step(self.scene.grid.dy)

    @property
    def dz(self) -> float:
        return self._scalar_grid_step(self.scene.grid.dz)

    @property
    def grid_spacing(self) -> tuple[float, float, float]:
        return self.scene.grid.spacing

    @property
    def pml_thickness(self):
        return int(self.scene.boundary.num_layers) if self.scene.boundary.uses_kind("pml") else 0

    @property
    def pml_strength(self):
        return float(self.scene.boundary.strength) if self.scene.boundary.uses_kind("pml") else 0.0

    @property
    def bloch_wavevector(self):
        if not self.scene.boundary.uses_kind("bloch"):
            return (0.0, 0.0, 0.0)
        if self.scene.boundary.bloch_wavevector == "auto":
            raise ValueError(
                "Automatic Bloch wavevectors require Simulation.prepare() to resolve the incident wavevector."
            )
        return self.scene.boundary.bloch_wavevector

    @property
    def bloch_phase_factors(self):
        return self.scene.boundary.bloch_phase_factors(self.physical_domain_range)

    def refresh_material_grids(self, subpixel=None):
        from .compiler.materials import compile_material_components

        eps_components, mu_components = compile_material_components(self, subpixel=subpixel)
        self._permittivity_components = _clone_component_map(eps_components)
        self._permeability_components = _clone_component_map(mu_components)
        self._permittivity = _component_summary(self._permittivity_components)
        self._permeability = _component_summary(self._permeability_components)
        return self._permittivity, self._permeability

    def _ensure_material_grids(self):
        if self._permittivity is None or self._permeability is None:
            self.refresh_material_grids()

    @property
    def permittivity_components(self):
        self._ensure_material_grids()
        return self._permittivity_components

    @property
    def permeability_components(self):
        self._ensure_material_grids()
        return self._permeability_components

    @property
    def permittivity(self):
        self._ensure_material_grids()
        return self._permittivity

    @property
    def permeability(self):
        self._ensure_material_grids()
        return self._permeability

    def get_cross_section(self, axis="z", position=0.0, field=None):
        field = self.permittivity if field is None else field
        if axis == "z":
            idx = torch.argmin(torch.abs(self.z - position))
            return field[:, :, idx]
        if axis == "y":
            idx = torch.argmin(torch.abs(self.y - position))
            return field[:, idx, :]
        if axis == "x":
            idx = torch.argmin(torch.abs(self.x - position))
            return field[idx, :, :]
        raise ValueError("axis must be 'x', 'y', or 'z'")


def scene_declarations(scene: Scene | PreparedScene) -> Scene:
    """The declarative ``Scene`` of ``scene``: a prepared scene's ``scene``, or ``scene`` itself.

    The port-sweep and wave-port runs carry the scene prepared once in ``Simulation.prepare()`` so their
    columns reuse its grid and compile caches; they read declarations through this.
    """
    return scene.scene if isinstance(scene, PreparedScene) else scene


def prepare_scene(
    scene: Scene | PreparedScene,
    *,
    device=None,
    lazy_meshgrid: bool | None = None,
    material_dtype=None,
) -> PreparedScene:
    """The solver-ready form of ``scene`` with its tensors on ``device`` (default CUDA).

    ``device`` places a declarative Scene; an already prepared scene keeps its device.
    """
    if isinstance(scene, PreparedScene):
        dtype = scene._material_dtype if material_dtype is None else material_dtype
        lazy = scene.lazy_meshgrid if lazy_meshgrid is None else bool(lazy_meshgrid)
        if (lazy, dtype) == (scene.lazy_meshgrid, scene._material_dtype):
            return scene
        return PreparedScene(scene._public_scene, device=scene.device, lazy_meshgrid=lazy, material_dtype=dtype)
    if not isinstance(scene, Scene):
        raise TypeError("prepare_scene() expects a maxwell.Scene instance.")
    return PreparedScene(
        scene,
        device=device,
        lazy_meshgrid=True if lazy_meshgrid is None else lazy_meshgrid,
        material_dtype=torch.float32 if material_dtype is None else material_dtype,
    )
