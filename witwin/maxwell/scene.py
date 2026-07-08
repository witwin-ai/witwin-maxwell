from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import numpy as np
import torch

from witwin.core import SceneBase, Structure
from .ports import ModePort
from .compiler.materials import (
    _validate_scene_material_combinations,
    compile_material_model,
    evaluate_material_components,
    evaluate_material_permeability,
    evaluate_material_permittivity,
)


BoundaryKind: TypeAlias = Literal["none", "pml", "periodic", "bloch", "pec", "pmc", "mur"]
BoundaryAxisOverride: TypeAlias = BoundaryKind | tuple[BoundaryKind, BoundaryKind]
BlochWavevector: TypeAlias = tuple[float, float, float] | Literal["auto"]

_ResolvedBoundaryKind: TypeAlias = BoundaryKind | Literal["mixed"]
_BOUNDARY_AXES = ("x", "y", "z")
_BOUNDARY_AXIS_TO_INDEX = {axis: index for index, axis in enumerate(_BOUNDARY_AXES)}
_BOUNDARY_SIDE_TO_INDEX = {"low": 0, "high": 1}
_VALID_BOUNDARY_KINDS = {"none", "pml", "periodic", "bloch", "pec", "pmc", "mur"}
_PAIR_ONLY_BOUNDARY_KINDS = {"periodic", "bloch"}


def _normalize_boundary_kind(value, *, allow_mixed: bool = False) -> str:
    kind = str(value).lower()
    if kind == "mixed" and allow_mixed:
        return kind
    if kind not in _VALID_BOUNDARY_KINDS:
        raise ValueError(
            "Boundary kind must be one of 'none', 'pml', 'periodic', 'bloch', 'pec', 'pmc', or 'mur'."
        )
    return kind


def _normalize_boundary_axis_override(name: str, value) -> tuple[str, str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        kind = _normalize_boundary_kind(value)
        return (kind, kind)
    if len(value) != 2:
        raise ValueError(f"{name} must be a boundary kind or a (low, high) pair.")
    return tuple(_normalize_boundary_kind(component) for component in value)


def _normalize_boundary_face_override(name: str, value) -> str | None:
    if value is None:
        return None
    return _normalize_boundary_kind(value)


def _summarize_boundary_kind(face_kinds: tuple[tuple[str, str], tuple[str, str], tuple[str, str]]) -> _ResolvedBoundaryKind:
    first_kind = face_kinds[0][0]
    if all(low_kind == first_kind and high_kind == first_kind for low_kind, high_kind in face_kinds):
        return first_kind
    return "mixed"


@dataclass(frozen=True)
class Domain:
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]

    def __init__(self, bounds):
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


@dataclass(frozen=True)
class LayerRefinementSpec:
    """Minimum cell count across thin structure-bounded intervals.

    ``axes=None`` covers all axes; otherwise only the listed axes are refined.
    """

    min_cells: int = 2
    axes: tuple[str, ...] | None = None

    def __post_init__(self):
        min_cells = int(self.min_cells)
        if min_cells < 1:
            raise ValueError("LayerRefinementSpec.min_cells must be >= 1.")
        object.__setattr__(self, "min_cells", min_cells)
        if self.axes is not None:
            axes = tuple(str(axis).lower() for axis in self.axes)
            if any(axis not in _BOUNDARY_AXIS_TO_INDEX for axis in axes):
                raise ValueError("LayerRefinementSpec.axes entries must be 'x', 'y', or 'z'.")
            object.__setattr__(self, "axes", axes)

    def covers(self, axis: str) -> bool:
        return self.axes is None or str(axis).lower() in self.axes


def _validate_custom_axis_coords(values, axis: str) -> np.ndarray:
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
    layer_refinement: LayerRefinementSpec | None = None

    @classmethod
    def uniform(cls, dl: float) -> "GridSpec":
        step = float(dl)
        return cls(step, step, step)

    @classmethod
    def anisotropic(cls, dx: float, dy: float, dz: float) -> "GridSpec":
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
        layer_refinement: LayerRefinementSpec | None = None,
    ) -> "GridSpec":
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
        if layer_refinement is not None and not isinstance(layer_refinement, LayerRefinementSpec):
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
    kind: BoundaryKind | Literal["mixed"] = "none"
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
    _face_kinds: tuple[tuple[BoundaryKind, BoundaryKind], tuple[BoundaryKind, BoundaryKind], tuple[BoundaryKind, BoundaryKind]] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self):
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
            face_kind == "pml"
            for axis_face_kinds in normalized_face_kinds
            for face_kind in axis_face_kinds
        )
        uses_bloch = any(
            face_kind == "bloch"
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
                if "bloch" not in axis_face_kinds:
                    continue
                if axis_face_kinds[0] != "bloch" or axis_face_kinds[1] != "bloch":
                    raise ValueError(f"{axis}-axis Bloch boundaries must be configured on both faces.")
                if (
                    bloch_wavevector != "auto"
                    and resolved_kind != "bloch"
                    and np.isclose(bloch_wavevector[axis_index], 0.0)
                ):
                    raise ValueError(
                        f"{axis}-axis Bloch boundaries require a non-zero bloch_wavevector component."
                    )

        for axis, axis_face_kinds in zip(_BOUNDARY_AXES, normalized_face_kinds):
            axis_face_set = set(axis_face_kinds)
            if _PAIR_ONLY_BOUNDARY_KINDS.intersection(axis_face_set) and axis_face_kinds[0] != axis_face_kinds[1]:
                raise ValueError(
                    f"{axis}-axis periodic and Bloch boundaries must use the same kind on both faces."
                )

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
        return cls(kind="pml", num_layers=int(num_layers), strength=float(strength))

    @classmethod
    def none(cls) -> "BoundarySpec":
        return cls(kind="none", num_layers=0, strength=0.0)

    @classmethod
    def periodic(cls) -> "BoundarySpec":
        return cls(kind="periodic")

    @classmethod
    def bloch(cls, wavevector) -> "BoundarySpec":
        return cls(kind="bloch", bloch_wavevector=wavevector)

    @classmethod
    def pec(cls) -> "BoundarySpec":
        return cls(kind="pec")

    @classmethod
    def pmc(cls) -> "BoundarySpec":
        return cls(kind="pmc")

    @classmethod
    def mur(cls) -> "BoundarySpec":
        return cls(kind="mur")

    @classmethod
    def faces(
        cls,
        *,
        default: BoundaryKind = "none",
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
    ) -> "BoundarySpec":
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

    def axis_kind(self, axis: str) -> _ResolvedBoundaryKind:
        low_kind, high_kind = self.axis_face_kinds(axis)
        if low_kind == high_kind:
            return low_kind
        return "mixed"

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
            face_kind == normalized_kind
            for axis_face_kinds in self._face_kinds
            for face_kind in axis_face_kinds
        )

    def pml_layers_for_face(self, axis: str, side: str) -> int:
        return int(self.num_layers) if self.face_kind(axis, side) == "pml" else 0

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
            if self.axis_kind(axis) == "bloch":
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
            default="none",
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
    def to_scene(self) -> "Scene":
        raise NotImplementedError("SceneModule subclasses must implement to_scene().")


_USE_SCENE_SUBPIXEL_SAMPLES = object()


def _normalize_subpixel_samples(value) -> tuple[int, int, int]:
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


def _to_like_tensor(value, *, reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def _component_summary(components: dict[str, torch.Tensor]) -> torch.Tensor:
    return (components["x"] + components["y"] + components["z"]) / 3.0


def _clone_component_map(components: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {axis: tensor.clone() for axis, tensor in components.items()}


def _resolve_scene_device(device) -> str:
    requested = "cuda" if device is None else device
    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Scene requires CUDA by default, but torch.cuda.is_available() is False. "
            "Pass device='cpu' for scene-only or CPU postprocessing workflows."
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


class Scene(SceneBase):
    """Declarative 3D scene definition for simulation backends."""

    def __init__(
        self,
        domain: Domain | None = None,
        grid: GridSpec | None = None,
        boundary: BoundarySpec | None = None,
        *,
        structures=None,
        sources=None,
        monitors=None,
        ports=None,
        material_regions=None,
        metadata=None,
        device="cuda",
        verbose=False,
        lazy_meshgrid=True,
        subpixel_samples=1,
        symmetry=None,
    ):
        resolved_device = _resolve_scene_device(device)
        super().__init__(
            structures=structures,
            sources=sources,
            monitors=monitors,
            metadata=metadata,
            device=resolved_device,
            verbose=verbose,
        )

        if domain is None:
            domain = Domain.from_domain_range((-1.0, 1.0, -1.0, 1.0, -1.0, 1.0))
        if grid is None:
            grid = GridSpec.uniform(0.05)
        if boundary is None:
            boundary = BoundarySpec.none()

        self.domain = domain
        self.grid = grid
        self.boundary = boundary
        self.subpixel_samples = _normalize_subpixel_samples(subpixel_samples)
        self.symmetry = _normalize_symmetry(symmetry)
        self.material_regions = list(material_regions or [])
        self.ports = list(ports or [])
        self.lazy_meshgrid = bool(lazy_meshgrid)

    def _scalar_grid_step(self, value: float | None) -> float:
        if self.grid.is_custom or self.grid.is_auto:
            raise ValueError(
                "scalar spacing undefined on a nonuniform grid; use scene.x/y/z, x_half, or grid.min_spacing."
            )
        return float(value)

    @property
    def dx(self) -> float:
        return self._scalar_grid_step(self.grid.dx)

    @property
    def dy(self) -> float:
        return self._scalar_grid_step(self.grid.dy)

    @property
    def dz(self) -> float:
        return self._scalar_grid_step(self.grid.dz)

    @property
    def grid_spacing(self) -> tuple[float, float, float]:
        return self.grid.spacing

    @property
    def domain_size(self) -> float:
        bounds = self.domain.bounds
        return max(
            float(bounds[0][1] - bounds[0][0]),
            float(bounds[1][1] - bounds[1][0]),
            float(bounds[2][1] - bounds[2][0]),
        )

    @property
    def boundary_type(self):
        labels = {
            "none": "None",
            "pml": "PML",
            "periodic": "Periodic",
            "bloch": "Bloch",
            "pec": "PEC",
            "pmc": "PMC",
            "mixed": "Mixed",
        }
        return labels[self.boundary.kind]

    @property
    def bc_type(self):
        return self.boundary_type

    @property
    def pml_thickness(self):
        return int(self.boundary.num_layers) if self.boundary.uses_kind("pml") else 0

    @property
    def pml_strength(self):
        return float(self.boundary.strength) if self.boundary.uses_kind("pml") else 0.0

    @property
    def bloch_wavevector(self):
        if not self.boundary.uses_kind("bloch"):
            return (0.0, 0.0, 0.0)
        if self.boundary.bloch_wavevector == "auto":
            raise ValueError(
                "Automatic Bloch wavevectors require Simulation.prepare() to resolve the incident wavevector."
            )
        return self.boundary.bloch_wavevector

    def boundary_face_kind(self, axis: str, side: str) -> BoundaryKind:
        return self.boundary.face_kind(axis, side)

    def pml_thickness_for_face(self, axis: str, side: str) -> int:
        return self.boundary.pml_layers_for_face(axis, side)

    @property
    def bloch_phase_factors(self):
        domain_range = _domain_range_from_bounds(self.domain.bounds)
        return self.boundary.bloch_phase_factors(domain_range)

    @property
    def has_symmetry(self) -> bool:
        return any(mode is not None for mode in self.symmetry)

    def add_structure(self, structure: Structure):
        self.structures.append(structure)
        _validate_scene_material_combinations(self)
        return self

    def add_source(self, source):
        self.sources.append(source)
        return self

    def add_monitor(self, monitor):
        self.monitors.append(monitor)
        return self

    def add_port(self, port: ModePort):
        self.ports.append(port)
        return self

    def add_material_region(self, material_region: MaterialRegion):
        self.material_regions.append(material_region)
        return self

    def clone(self, **overrides):
        params = {
            "domain": self.domain,
            "grid": self.grid,
            "boundary": self.boundary,
            "structures": list(self.structures),
            "sources": list(self.sources),
            "monitors": list(self.monitors),
            "ports": list(self.ports),
            "material_regions": list(self.material_regions),
            "metadata": dict(self.metadata),
            "device": self.device,
            "verbose": self.verbose,
            "lazy_meshgrid": self.lazy_meshgrid,
            "subpixel_samples": self.subpixel_samples,
            "symmetry": self.symmetry,
        }
        params.update(overrides)
        return Scene(**params)

    def resolved_sources(self):
        resolved = list(self.sources)
        for port in self.ports:
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
            resolved.append(port.to_mode_monitor())
        return resolved

    def to_tidy3d(self, *, frequencies=None, run_time=None, **kwargs):
        """Convert this scene to a ``tidy3d.Simulation``.

        Parameters
        ----------
        frequencies : float or sequence of float, optional
            Monitoring / source frequencies in Hz.
        run_time : float, optional
            Simulation run time in seconds.  Estimated from domain size if omitted.
        **kwargs
            Extra keyword arguments forwarded to ``tidy3d.Simulation``.

        Returns
        -------
        tidy3d.Simulation
        """
        from .adapters.tidy3d import scene_to_tidy3d
        return scene_to_tidy3d(self, frequencies=frequencies, run_time=run_time, **kwargs)


def _axis_nodes64(grid: GridSpec, axis: str, axis_lo: float, axis_hi: float) -> np.ndarray:
    coords = grid.axis_coords(axis)
    if coords is None:
        step = float(getattr(grid, f"d{axis}"))
        count = max(0, int(math.ceil((float(axis_hi) - float(axis_lo)) / step)))
        return float(axis_lo) + np.arange(count, dtype=np.float64) * step
    span = float(axis_hi) - float(axis_lo)
    tolerance = 1e-9 * span
    if abs(float(coords[0]) - float(axis_lo)) > tolerance or abs(float(coords[-1]) - float(axis_hi)) > tolerance:
        raise ValueError(
            f"GridSpec.custom {axis}_coords must span the domain exactly along axis '{axis}'; "
            f"build Domain from ({float(coords[0]):g}, {float(coords[-1]):g})."
        )
    return coords


def _axis_dual_spacing64(primal: np.ndarray, boundary_kind: str) -> np.ndarray:
    dual = np.empty(primal.size + 1, dtype=np.float64)
    dual[1:-1] = 0.5 * (primal[:-1] + primal[1:])
    if boundary_kind in ("periodic", "bloch"):
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
    grid: GridSpec, axis: str, axis_lo: float, axis_hi: float, boundary_kind: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nodes = _axis_nodes64(grid, axis, axis_lo, axis_hi)
    primal = np.diff(nodes)
    half = nodes[:-1] + 0.5 * primal
    dual = _axis_dual_spacing64(primal, boundary_kind)
    return nodes, primal, half, dual


class PreparedScene(Scene):
    """Internal solver-ready scene with compiled grid state."""

    def __init__(self, scene: Scene, *, lazy_meshgrid: bool | None = None):
        resolved_lazy_meshgrid = scene.lazy_meshgrid if lazy_meshgrid is None else bool(lazy_meshgrid)
        super().__init__(
            domain=scene.domain,
            grid=scene.grid,
            boundary=scene.boundary,
            structures=list(scene.structures),
            sources=list(scene.sources),
            monitors=list(scene.monitors),
            ports=list(scene.ports),
            material_regions=list(scene.material_regions),
            metadata=dict(scene.metadata),
            device=scene.device,
            verbose=scene.verbose,
            lazy_meshgrid=resolved_lazy_meshgrid,
            subpixel_samples=scene.subpixel_samples,
            symmetry=scene.symmetry,
        )
        self._public_scene = scene
        self.domain_range = _domain_range_from_bounds(self.domain.bounds)

        if self.grid.is_auto:
            from .fdtd.meshing import resolve_auto_grid

            self.grid = GridSpec.custom(*resolve_auto_grid(self))

        x_start, x_end, y_start, y_end, z_start, z_end = self.domain_range
        self.x_nodes64, self.dx_primal64, self.x_half64, self.dx_dual64 = _build_axis_grid64(
            self.grid, "x", x_start, x_end, self.boundary.axis_kind("x")
        )
        self.y_nodes64, self.dy_primal64, self.y_half64, self.dy_dual64 = _build_axis_grid64(
            self.grid, "y", y_start, y_end, self.boundary.axis_kind("y")
        )
        self.z_nodes64, self.dz_primal64, self.z_half64, self.dz_dual64 = _build_axis_grid64(
            self.grid, "z", z_start, z_end, self.boundary.axis_kind("z")
        )
        self.x = torch.tensor(self.x_nodes64, dtype=torch.float32, device=self.device)
        self.y = torch.tensor(self.y_nodes64, dtype=torch.float32, device=self.device)
        self.z = torch.tensor(self.z_nodes64, dtype=torch.float32, device=self.device)
        self.x_half = torch.tensor(self.x_half64, dtype=torch.float32, device=self.device)
        self.y_half = torch.tensor(self.y_half64, dtype=torch.float32, device=self.device)
        self.z_half = torch.tensor(self.z_half64, dtype=torch.float32, device=self.device)

        self._xx = None
        self._yy = None
        self._zz = None
        self._material_model_cache = {}
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
        self._material_model_cache = {}
        self._permittivity_components = None
        self._permeability_components = None
        self._permittivity = None
        self._permeability = None

    def _has_dynamic_materials(self) -> bool:
        return bool(self.material_regions)

    def _has_trainable_geometry(self) -> bool:
        for structure in self.structures:
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

    def add_structure(self, structure: Structure):
        super().add_structure(structure)
        _validate_scene_material_combinations(self)
        self._invalidate_material_cache()
        return self

    def add_material_region(self, material_region: MaterialRegion):
        super().add_material_region(material_region)
        self._invalidate_material_cache()
        return self

    def compile_materials(self, subpixel_samples=_USE_SCENE_SUBPIXEL_SAMPLES):
        resolved_subpixel = (
            self.subpixel_samples
            if subpixel_samples is _USE_SCENE_SUBPIXEL_SAMPLES
            else _normalize_subpixel_samples(subpixel_samples)
        )
        key = resolved_subpixel
        if self._has_dynamic_materials() or self._has_trainable_geometry():
            model = compile_material_model(
                self,
                eps_background=1.0,
                mu_background=1.0,
                subpixel_samples=resolved_subpixel,
            )
            self.release_meshgrid()
            return model

        cached = self._material_model_cache.get(key)
        if cached is None:
            cached = compile_material_model(
                self,
                eps_background=1.0,
                mu_background=1.0,
                subpixel_samples=resolved_subpixel,
            )
            self._material_model_cache[key] = cached
            self.release_meshgrid()
        return cached

    def compile_relative_materials(
        self,
        subpixel_samples=_USE_SCENE_SUBPIXEL_SAMPLES,
        frequency=None,
    ):
        eps_components, mu_components = self.compile_material_components(
            subpixel_samples=subpixel_samples,
            frequency=frequency,
        )
        return _component_summary(eps_components), _component_summary(mu_components)

    def compile_material_components(
        self,
        subpixel_samples=_USE_SCENE_SUBPIXEL_SAMPLES,
        frequency=None,
    ):
        model = self.compile_materials(subpixel_samples=subpixel_samples)
        return evaluate_material_components(model, frequency)

    def compile_material_tensors(
        self,
        eps_background=1.0,
        mu_background=1.0,
        subpixel_samples=_USE_SCENE_SUBPIXEL_SAMPLES,
        frequency=None,
    ):
        eps_r, mu_r = self.compile_relative_materials(
            subpixel_samples=subpixel_samples,
            frequency=frequency,
        )
        eps_background_tensor = _to_like_tensor(eps_background, reference=eps_r)
        mu_background_tensor = _to_like_tensor(mu_background, reference=mu_r)
        if torch.equal(eps_background_tensor, torch.ones_like(eps_background_tensor)) and torch.equal(
            mu_background_tensor,
            torch.ones_like(mu_background_tensor),
        ):
            return eps_r.clone(), mu_r.clone()
        return eps_r * eps_background_tensor, mu_r * mu_background_tensor

    def refresh_material_grids(self, subpixel_samples=_USE_SCENE_SUBPIXEL_SAMPLES):
        eps_components, mu_components = self.compile_material_components(
            subpixel_samples=subpixel_samples,
        )
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


def prepare_scene(scene: Scene | PreparedScene, *, lazy_meshgrid: bool | None = None) -> PreparedScene:
    if isinstance(scene, PreparedScene):
        if lazy_meshgrid is None or scene.lazy_meshgrid == bool(lazy_meshgrid):
            return scene
        return PreparedScene(scene._public_scene, lazy_meshgrid=lazy_meshgrid)
    if not isinstance(scene, Scene):
        raise TypeError("prepare_scene() expects a maxwell.Scene instance.")
    return PreparedScene(scene, lazy_meshgrid=lazy_meshgrid)
