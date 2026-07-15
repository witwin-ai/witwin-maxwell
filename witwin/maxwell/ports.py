from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
from witwin.core import Box

from .lumped import ParallelRLC, SeriesRLC
from .monitors import ModeMonitor, _normalize_frequencies, _normalize_normal_direction
from .sources import (
    ModeSource,
    SourceTime,
    _normalize_bend,
    _normalize_mode_direction,
    _require_length3,
    _require_mode_source_size,
    _resolve_mode_source_normal_axis,
    _resolve_mode_source_polarization_axis,
    polarization_vector,
)


@dataclass(frozen=True)
class AxisPath:
    """Axis-aligned voltage integration path.

    The path orientation is supplied by the owning port's ``negative`` and
    ``positive`` terminals; this object only fixes the Cartesian axis.
    """

    axis: str

    def __init__(self, axis):
        if not isinstance(axis, str):
            raise TypeError("AxisPath axis must be one of 'x', 'y', or 'z'.")
        resolved_axis = axis.strip().lower()
        if resolved_axis not in {"x", "y", "z"}:
            raise ValueError("AxisPath axis must be one of 'x', 'y', or 'z'.")
        object.__setattr__(self, "axis", resolved_axis)


def _normalize_reference_impedance(reference_impedance):
    if torch.is_tensor(reference_impedance):
        if reference_impedance.numel() != 1:
            raise ValueError("reference_impedance tensor must contain one value.")
        real = torch.real(reference_impedance)
        imag = torch.imag(reference_impedance) if reference_impedance.is_complex() else torch.zeros_like(real)
        if not bool(torch.isfinite(real)) or not bool(torch.isfinite(imag)):
            raise ValueError("reference_impedance must be finite.")
        if not bool(real > 0.0):
            raise ValueError("reference_impedance real part must be positive.")
        return reference_impedance

    resolved_impedance = complex(reference_impedance)
    if not math.isfinite(resolved_impedance.real) or not math.isfinite(resolved_impedance.imag):
        raise ValueError("reference_impedance must be finite.")
    if resolved_impedance.real <= 0.0:
        raise ValueError("reference_impedance real part must be positive.")
    return (
        float(resolved_impedance.real)
        if resolved_impedance.imag == 0.0
        else resolved_impedance
    )


def _validate_termination(termination) -> None:
    if termination is not None and not isinstance(termination, (SeriesRLC, ParallelRLC)):
        raise TypeError("termination must be a SeriesRLC, ParallelRLC, or None.")


@dataclass(frozen=True)
class LumpedPort:
    """Declare a two-terminal lumped port with explicit discrete V/I geometry.

    Voltage is the line integral of E from ``negative`` to ``positive``.
    Current is the right-hand circulation of H around ``current_surface`` with
    the same orientation. Phasors are peak-valued, so average power is
    ``0.5 * Re(V * conj(I))``.
    """

    name: str
    positive: tuple[float, float, float]
    negative: tuple[float, float, float]
    voltage_path: AxisPath
    current_surface: Box
    reference_impedance: complex | float | torch.Tensor
    reference_plane: float | None = None
    termination: SeriesRLC | ParallelRLC | None = None
    kind: str = "lumped_port"
    phasor_convention: str = "peak"
    power_convention: str = "0.5*Re(V*conj(I))"

    def __init__(
        self,
        name,
        *,
        positive,
        negative,
        voltage_path,
        current_surface,
        reference_impedance=50.0,
        reference_plane=None,
        termination=None,
    ):
        if not isinstance(voltage_path, AxisPath):
            raise TypeError("voltage_path must be an AxisPath.")
        if not isinstance(current_surface, Box):
            raise TypeError("current_surface must be a witwin.core.Box.")

        resolved_positive = _require_length3("positive", positive)
        resolved_negative = _require_length3("negative", negative)
        if resolved_positive == resolved_negative:
            raise ValueError("LumpedPort positive and negative terminals must be distinct.")

        axis_index = "xyz".index(voltage_path.axis)
        transverse_indices = tuple(index for index in range(3) if index != axis_index)
        if any(
            not math.isclose(
                resolved_positive[index],
                resolved_negative[index],
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            for index in transverse_indices
        ):
            raise ValueError(
                "LumpedPort positive and negative terminals must define an axis-aligned voltage path."
            )
        if math.isclose(
            resolved_positive[axis_index],
            resolved_negative[axis_index],
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("LumpedPort positive and negative terminals must be distinct along voltage_path.axis.")

        stored_impedance = _normalize_reference_impedance(reference_impedance)

        resolved_name = str(name)
        if not resolved_name:
            raise ValueError("LumpedPort name must not be empty.")
        _validate_termination(termination)

        object.__setattr__(self, "name", resolved_name)
        object.__setattr__(self, "positive", resolved_positive)
        object.__setattr__(self, "negative", resolved_negative)
        object.__setattr__(self, "voltage_path", voltage_path)
        object.__setattr__(self, "current_surface", current_surface)
        object.__setattr__(self, "reference_impedance", stored_impedance)
        object.__setattr__(
            self,
            "reference_plane",
            None if reference_plane is None else float(reference_plane),
        )
        object.__setattr__(self, "termination", termination)
        object.__setattr__(self, "kind", "lumped_port")
        object.__setattr__(self, "phasor_convention", "peak")
        object.__setattr__(self, "power_convention", "0.5*Re(V*conj(I))")


@dataclass(frozen=True)
class TerminalRef:
    """Reference one uniquely named conductor structure in a scene."""

    structure_name: str

    def __init__(self, structure_name):
        if not isinstance(structure_name, str):
            raise TypeError("TerminalRef structure_name must be a string.")
        resolved_name = structure_name.strip()
        if not resolved_name:
            raise ValueError("TerminalRef structure_name must not be empty.")
        object.__setattr__(self, "structure_name", resolved_name)


@dataclass(frozen=True)
class TerminalPort:
    """Declare a two-conductor port resolved from named PEC Box terminals.

    Version 1 resolves the line path through the center of the terminals'
    transverse footprint overlap. The current contour is that overlap placed
    on ``reference_plane``. Once accepted by a :class:`Scene`, the port exposes
    the same resolved geometry properties used by the lumped-port runtime.
    """

    name: str
    positive_terminal: TerminalRef
    negative_terminal: TerminalRef
    integration_path: AxisPath
    reference_plane: float
    reference_impedance: complex | float | torch.Tensor
    termination: SeriesRLC | ParallelRLC | None = None
    kind: str = "terminal_port"
    phasor_convention: str = "peak"
    power_convention: str = "0.5*Re(V*conj(I))"
    _positive: tuple[float, float, float] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _negative: tuple[float, float, float] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _current_surface: Box | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        name,
        positive_terminal,
        negative_terminal,
        integration_path,
        reference_plane,
        reference_impedance=50.0,
        termination=None,
    ):
        resolved_name = str(name)
        if not resolved_name:
            raise ValueError("TerminalPort name must not be empty.")
        if not isinstance(positive_terminal, TerminalRef):
            raise TypeError(
                f"TerminalPort {resolved_name!r} positive_terminal must be a TerminalRef."
            )
        if not isinstance(negative_terminal, TerminalRef):
            raise TypeError(
                f"TerminalPort {resolved_name!r} negative_terminal must be a TerminalRef."
            )
        if positive_terminal == negative_terminal:
            raise ValueError(
                f"TerminalPort {resolved_name!r} positive and negative terminals must reference distinct structures."
            )
        if not isinstance(integration_path, AxisPath):
            raise TypeError(
                f"TerminalPort {resolved_name!r} integration_path must be an AxisPath."
            )
        try:
            resolved_reference_plane = float(reference_plane)
        except (TypeError, ValueError) as error:
            raise type(error)(
                f"TerminalPort {resolved_name!r} reference_plane must be a real scalar."
            ) from error
        if not math.isfinite(resolved_reference_plane):
            raise ValueError(
                f"TerminalPort {resolved_name!r} reference_plane must be finite."
            )
        try:
            stored_impedance = _normalize_reference_impedance(reference_impedance)
            _validate_termination(termination)
        except (TypeError, ValueError) as error:
            raise type(error)(f"TerminalPort {resolved_name!r}: {error}") from error

        object.__setattr__(self, "name", resolved_name)
        object.__setattr__(self, "positive_terminal", positive_terminal)
        object.__setattr__(self, "negative_terminal", negative_terminal)
        object.__setattr__(self, "integration_path", integration_path)
        object.__setattr__(self, "reference_plane", resolved_reference_plane)
        object.__setattr__(self, "reference_impedance", stored_impedance)
        object.__setattr__(self, "termination", termination)
        object.__setattr__(self, "kind", "terminal_port")
        object.__setattr__(self, "phasor_convention", "peak")
        object.__setattr__(self, "power_convention", "0.5*Re(V*conj(I))")
        object.__setattr__(self, "_positive", None)
        object.__setattr__(self, "_negative", None)
        object.__setattr__(self, "_current_surface", None)

    @property
    def voltage_path(self) -> AxisPath:
        return self.integration_path

    @property
    def positive(self) -> tuple[float, float, float]:
        if self._positive is None:
            raise RuntimeError(
                f"TerminalPort {self.name!r} terminal geometry is unresolved; add it to a Scene first."
            )
        return self._positive

    @property
    def negative(self) -> tuple[float, float, float]:
        if self._negative is None:
            raise RuntimeError(
                f"TerminalPort {self.name!r} terminal geometry is unresolved; add it to a Scene first."
            )
        return self._negative

    @property
    def current_surface(self) -> Box:
        if self._current_surface is None:
            raise RuntimeError(
                f"TerminalPort {self.name!r} terminal geometry is unresolved; add it to a Scene first."
            )
        return self._current_surface

    def _with_resolved_geometry(
        self,
        *,
        positive: tuple[float, float, float],
        negative: tuple[float, float, float],
        current_surface: Box,
    ) -> "TerminalPort":
        resolved = TerminalPort(
            self.name,
            positive_terminal=self.positive_terminal,
            negative_terminal=self.negative_terminal,
            integration_path=self.integration_path,
            reference_plane=self.reference_plane,
            reference_impedance=self.reference_impedance,
            termination=self.termination,
        )
        object.__setattr__(resolved, "_positive", positive)
        object.__setattr__(resolved, "_negative", negative)
        object.__setattr__(resolved, "_current_surface", current_surface)
        return resolved


def _resolve_terminal_port(port: TerminalPort, structures, domain_bounds) -> TerminalPort:
    """Resolve the v1 terminal geometry without performing Yee-grid snapping."""

    def fail(message: str):
        raise ValueError(f"TerminalPort {port.name!r}: {message}")

    def find_structure(terminal: TerminalRef, label: str):
        matches = [
            structure
            for structure in structures
            if getattr(structure, "name", None) == terminal.structure_name
        ]
        if not matches:
            fail(f"{label} terminal structure {terminal.structure_name!r} does not exist.")
        if len(matches) != 1:
            fail(
                f"{label} terminal structure name {terminal.structure_name!r} is not unique."
            )
        structure = matches[0]
        if not bool(getattr(structure, "enabled", True)):
            fail(f"{label} terminal structure {terminal.structure_name!r} is disabled.")
        if not isinstance(getattr(structure, "geometry", None), Box):
            fail(f"{label} terminal structure {terminal.structure_name!r} must use a Box geometry.")
        if not bool(getattr(getattr(structure, "material", None), "is_pec", False)):
            fail(f"{label} terminal structure {terminal.structure_name!r} must be PEC.")
        return structure

    positive_structure = find_structure(port.positive_terminal, "positive")
    negative_structure = find_structure(port.negative_terminal, "negative")

    positions = []
    sizes = []
    domain_scale = max(
        1.0,
        *(abs(float(value)) for bounds in domain_bounds for value in bounds),
    )
    tolerance = 1.0e-12 * domain_scale
    identity = torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float64)
    for label, structure in (
        ("positive", positive_structure),
        ("negative", negative_structure),
    ):
        geometry = structure.geometry
        rotation = torch.as_tensor(geometry.rotation).detach().cpu().to(dtype=torch.float64)
        if not torch.allclose(rotation, identity, rtol=0.0, atol=1.0e-12):
            fail(
                f"{label} terminal structure {structure.name!r} must be an unrotated axis-aligned Box."
            )
        position = torch.as_tensor(geometry.position).detach().cpu().to(dtype=torch.float64)
        size = torch.as_tensor(geometry.size).detach().cpu().to(dtype=torch.float64)
        if not bool(torch.all(torch.isfinite(position))) or not bool(torch.all(torch.isfinite(size))):
            fail(f"{label} terminal structure {structure.name!r} must have finite Box geometry.")
        if not bool(torch.all(size > 0.0)):
            fail(f"{label} terminal structure {structure.name!r} must have positive Box extents.")
        lower = position - 0.5 * size
        upper = position + 0.5 * size
        for axis, axis_lower, axis_upper, bounds in zip("xyz", lower, upper, domain_bounds):
            domain_lower, domain_upper = (float(value) for value in bounds)
            if float(axis_lower) < domain_lower - tolerance or float(axis_upper) > domain_upper + tolerance:
                fail(
                    f"{label} terminal structure {structure.name!r} lies outside the Scene domain along {axis}."
                )
        positions.append(position)
        sizes.append(size)

    positive_lower = positions[0] - 0.5 * sizes[0]
    positive_upper = positions[0] + 0.5 * sizes[0]
    negative_lower = positions[1] - 0.5 * sizes[1]
    negative_upper = positions[1] + 0.5 * sizes[1]
    axis_index = "xyz".index(port.integration_path.axis)

    overlap_lower = torch.maximum(positive_lower, negative_lower)
    overlap_upper = torch.minimum(positive_upper, negative_upper)
    for transverse_index in range(3):
        if transverse_index == axis_index:
            continue
        if float(overlap_upper[transverse_index] - overlap_lower[transverse_index]) <= tolerance:
            fail("terminal transverse footprints do not overlap.")

    positive_axis_lower = float(positive_lower[axis_index])
    positive_axis_upper = float(positive_upper[axis_index])
    negative_axis_lower = float(negative_lower[axis_index])
    negative_axis_upper = float(negative_upper[axis_index])
    if negative_axis_upper < positive_axis_lower - tolerance:
        negative_axis_coordinate = negative_axis_upper
        positive_axis_coordinate = positive_axis_lower
    elif positive_axis_upper < negative_axis_lower - tolerance:
        positive_axis_coordinate = positive_axis_upper
        negative_axis_coordinate = negative_axis_lower
    else:
        fail(
            f"terminal surfaces are not distinct facing faces along AxisPath({port.integration_path.axis!r})."
        )

    path_lower, path_upper = sorted((negative_axis_coordinate, positive_axis_coordinate))
    if not path_lower + tolerance < port.reference_plane < path_upper - tolerance:
        fail("reference_plane must lie strictly between the two facing terminal surfaces.")

    overlap_center = 0.5 * (overlap_lower + overlap_upper)
    positive = overlap_center.clone()
    negative = overlap_center.clone()
    positive[axis_index] = positive_axis_coordinate
    negative[axis_index] = negative_axis_coordinate
    surface_position = overlap_center.clone()
    surface_position[axis_index] = port.reference_plane
    surface_size = overlap_upper - overlap_lower
    surface_size[axis_index] = 0.0
    return port._with_resolved_geometry(
        positive=tuple(float(value) for value in positive),
        negative=tuple(float(value) for value in negative),
        current_surface=Box(
            position=tuple(float(value) for value in surface_position),
            size=tuple(float(value) for value in surface_size),
        ),
    )


@dataclass(frozen=True)
class ModePort:
    """Couple a source and monitor using a polarization-family mode index."""

    name: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    mode_index: int
    direction: str
    polarization: tuple[float, float, float]
    frequencies: tuple[float, ...] | None = None
    source_time: SourceTime | None = None
    monitor_offset: float = 0.0
    normal_direction: str = "+"
    normal_axis: str = "z"
    polarization_axis: str = "x"
    bend_radius: float | None = None
    bend_axis: str | None = None
    kind: str = "mode_port"

    def __init__(
        self,
        name,
        position=(0.0, 0.0, 0.0),
        size=(1.0, 1.0, 0.0),
        mode_index=0,
        direction="+",
        polarization="auto",
        frequencies=None,
        source_time=None,
        monitor_offset=0.0,
        normal_direction=None,
        bend_radius=None,
        bend_axis=None,
    ):
        resolved_position = _require_length3("position", position)
        resolved_size = _require_mode_source_size(size)
        normal_axis = _resolve_mode_source_normal_axis(resolved_size)
        polarization_axis = _resolve_mode_source_polarization_axis(normal_axis, resolved_size, polarization)
        resolved_bend_radius, resolved_bend_axis = _normalize_bend(bend_radius, bend_axis, normal_axis)
        resolved_direction = _normalize_mode_direction(direction)
        resolved_normal_direction = (
            resolved_direction if normal_direction is None else _normalize_normal_direction(normal_direction)
        )
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "position", resolved_position)
        object.__setattr__(self, "size", resolved_size)
        object.__setattr__(self, "mode_index", int(mode_index))
        object.__setattr__(self, "direction", resolved_direction)
        object.__setattr__(self, "polarization", polarization_vector(f"E{polarization_axis}"))
        object.__setattr__(self, "frequencies", _normalize_frequencies(frequencies))
        object.__setattr__(self, "source_time", source_time)
        object.__setattr__(self, "monitor_offset", float(monitor_offset))
        object.__setattr__(self, "normal_direction", resolved_normal_direction)
        object.__setattr__(self, "normal_axis", normal_axis)
        object.__setattr__(self, "polarization_axis", polarization_axis)
        object.__setattr__(self, "bend_radius", resolved_bend_radius)
        object.__setattr__(self, "bend_axis", resolved_bend_axis)
        object.__setattr__(self, "kind", "mode_port")
        if self.mode_index < 0:
            raise ValueError("mode_index must be >= 0.")

    @property
    def source_name(self) -> str:
        return f"{self.name}::source"

    def to_mode_source(self) -> ModeSource | None:
        if self.source_time is None:
            return None
        return ModeSource(
            position=self.position,
            size=self.size,
            mode_index=self.mode_index,
            direction=self.direction,
            polarization=f"E{self.polarization_axis}",
            source_time=self.source_time,
            name=self.source_name,
            bend_radius=self.bend_radius,
            bend_axis=self.bend_axis,
        )

    def to_mode_monitor(self) -> ModeMonitor:
        position = list(self.position)
        axis_index = "xyz".index(self.normal_axis)
        position[axis_index] += float(self.monitor_offset)
        return ModeMonitor(
            name=self.name,
            position=tuple(position),
            size=self.size,
            mode_index=self.mode_index,
            direction=self.direction,
            polarization=f"E{self.polarization_axis}",
            frequencies=self.frequencies,
            normal_direction=self.normal_direction,
            bend_radius=self.bend_radius,
            bend_axis=self.bend_axis,
        )
