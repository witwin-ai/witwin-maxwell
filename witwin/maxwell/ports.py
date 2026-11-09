from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math

import numpy as np
import torch
from witwin.core import Box

from ._fixed_inputs import require_fixed_input
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


@dataclass(frozen=True)
class WireNodeRef:
    """Stable reference to one declared point of a thin-wire polyline."""

    wire: str
    point: int

    def __init__(self, wire, point):
        resolved_wire = str(wire).strip()
        if not resolved_wire:
            raise ValueError("WireNodeRef wire must not be empty.")
        if isinstance(point, bool) or not isinstance(point, int):
            raise TypeError("WireNodeRef point must be an integer source-point index.")
        object.__setattr__(self, "wire", resolved_wire)
        object.__setattr__(self, "point", int(point))


@dataclass(frozen=True)
class WirePortBinding:
    """Bind a standard RF port to thin-wire nodes or a removed feed gap."""

    kind: str
    negative: WireNodeRef
    positive: WireNodeRef

    def __init__(
        self,
        kind,
        *,
        negative=None,
        positive=None,
    ):
        resolved = str(kind).strip().lower()
        if resolved not in {"nodes", "gap"}:
            raise ValueError("WirePortBinding kind must be 'nodes' or 'gap'.")
        if not isinstance(negative, WireNodeRef) or not isinstance(
            positive, WireNodeRef
        ):
            raise TypeError(
                f"WirePortBinding.{resolved}() requires WireNodeRef negative and positive terminals."
            )
        if negative == positive:
            raise ValueError("WirePortBinding terminals must be distinct.")
        if resolved == "gap" and negative.wire != positive.wire:
            raise ValueError("WirePortBinding gap terminals must reference one wire.")
        object.__setattr__(self, "kind", resolved)
        object.__setattr__(self, "negative", negative)
        object.__setattr__(self, "positive", positive)

    @classmethod
    def nodes(
        cls,
        *,
        negative: WireNodeRef,
        positive: WireNodeRef,
    ) -> "WirePortBinding":
        return cls("nodes", negative=negative, positive=positive)

    @classmethod
    def gap(
        cls,
        *,
        negative: WireNodeRef,
        positive: WireNodeRef,
    ) -> "WirePortBinding":
        return cls("gap", negative=negative, positive=positive)


def _fixed_reference_plane(value):
    require_fixed_input(value, name="Port.reference_plane")
    return float(value)


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


def _coordinate_roundoff(value) -> tuple[float, ...]:
    """Half an input ULP, retained before coordinates become Python floats."""
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.is_floating_point():
            magnitude = tensor.abs()
            ulp = torch.nextafter(magnitude, torch.full_like(magnitude, float("inf"))) - magnitude
            return tuple(float(item) * 0.5 for item in ulp.to(torch.float64).flatten())
        return (0.0,) * tensor.numel()
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.floating):
        return (0.0,) * array.size
    return tuple(float(item) * 0.5 for item in np.spacing(np.abs(array)).flatten())


@dataclass(frozen=True)
class LumpedPort:
    """Declare a two-terminal lumped port with explicit discrete V/I geometry.

    Voltage is the line integral of E from ``negative`` to ``positive``.
    Current into the field is minus the right-hand circulation of H around
    ``current_surface``, oriented along that terminal arrow. Phasors are
    peak-valued, so average power is
    ``0.5 * Re(V * conj(I))``.

    An optional ``voltage_aperture`` Box distributes the voltage row over
    parallel paths using normalized transverse dual-cell quadrature. Current
    deposition uses the same row, preserving field/circuit power duality.
    """

    name: str
    positive: tuple[float, float, float] | None
    negative: tuple[float, float, float] | None
    voltage_path: AxisPath | None
    current_surface: Box | None
    reference_impedance: complex | float | torch.Tensor
    reference_plane: float | None = None
    termination: SeriesRLC | ParallelRLC | None = None
    wire_binding: WirePortBinding | None = None
    kind: str = "lumped_port"
    phasor_convention: str = "peak"
    power_convention: str = "0.5*Re(V*conj(I))"
    _positive_roundoff: tuple[float, ...] = field(default=(), init=False, repr=False, compare=False)
    _negative_roundoff: tuple[float, ...] = field(default=(), init=False, repr=False, compare=False)
    voltage_aperture: Box | None = None

    def __init__(
        self,
        name,
        *,
        positive=None,
        negative=None,
        voltage_path=None,
        current_surface=None,
        reference_impedance=50.0,
        reference_plane=None,
        termination=None,
        wire_binding=None,
        voltage_aperture=None,
    ):
        if wire_binding is not None and not isinstance(wire_binding, WirePortBinding):
            raise TypeError("wire_binding must be a WirePortBinding or None.")
        has_geometry = any(
            value is not None
            for value in (positive, negative, voltage_path, current_surface, voltage_aperture)
        )
        if wire_binding is not None:
            if has_geometry:
                raise ValueError(
                    "A wire-bound LumpedPort derives its voltage/current coordinates from "
                    "wire_binding and must not also specify field geometry."
                )
            resolved_positive = None
            resolved_negative = None
        else:
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
        if voltage_aperture is not None:
            if not isinstance(voltage_aperture, Box):
                raise TypeError("voltage_aperture must be an axis-aligned Box.")
            if any(torch.is_tensor(v) and v.requires_grad for v in
                   (voltage_aperture.position, voltage_aperture.size, voltage_aperture.rotation)):
                raise NotImplementedError("Trainable aperture geometry changes discrete voltage support.")

        resolved_name = str(name)
        if not resolved_name:
            raise ValueError("LumpedPort name must not be empty.")
        _validate_termination(termination)

        object.__setattr__(self, "name", resolved_name)
        object.__setattr__(self, "positive", resolved_positive)
        object.__setattr__(self, "negative", resolved_negative)
        object.__setattr__(self, "_positive_roundoff", () if positive is None else _coordinate_roundoff(positive))
        object.__setattr__(self, "_negative_roundoff", () if negative is None else _coordinate_roundoff(negative))
        object.__setattr__(self, "voltage_path", voltage_path)
        object.__setattr__(self, "current_surface", current_surface)
        object.__setattr__(self, "reference_impedance", stored_impedance)
        object.__setattr__(
            self,
            "reference_plane",
            None if reference_plane is None else _fixed_reference_plane(reference_plane),
        )
        object.__setattr__(self, "termination", termination)
        object.__setattr__(self, "wire_binding", wire_binding)
        object.__setattr__(self, "voltage_aperture", voltage_aperture)
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
    """Declare a port between named PEC volumes or planar PEC/CopperFoil supports.

    Version 1 resolves the line path through the center of the terminals'
    transverse footprint overlap. Optional ``current_surface`` selects the path
    center and its current loop, including coplanar sheet gaps. An optional
    ``voltage_aperture`` uses the same power-dual row as LumpedPort. Polygon
    holes, connected return components and every native gap path are audited. Once accepted by a :class:`Scene`, the port exposes
    the same resolved geometry properties used by the lumped-port runtime.
    """

    name: str
    positive_terminal: TerminalRef | None
    negative_terminal: TerminalRef | None
    integration_path: AxisPath | None
    reference_plane: float | None
    reference_impedance: complex | float | torch.Tensor
    termination: SeriesRLC | ParallelRLC | None = None
    wire_binding: WirePortBinding | None = None
    voltage_aperture: Box | None = None
    _declared_current_surface: Box | None = field(default=None,init=False,repr=False,compare=False)
    _terminal_audit: dict = field(default_factory=dict,init=False,repr=False,compare=False)
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
    _current_bound_roundoff: tuple[float, ...] = field(default=(), init=False, repr=False, compare=False)
    _positive_roundoff: tuple[float, ...] = field(default=(), init=False, repr=False, compare=False)
    _negative_roundoff: tuple[float, ...] = field(default=(), init=False, repr=False, compare=False)

    def __init__(
        self,
        name,
        positive_terminal=None,
        negative_terminal=None,
        integration_path=None,
        reference_plane=None,
        reference_impedance=50.0,
        termination=None,
        wire_binding=None,
        *,
        current_surface=None,
        voltage_aperture=None,
    ):
        for label,box in (("current_surface",current_surface),("voltage_aperture",voltage_aperture)):
            if box is not None and not isinstance(box,Box):
                raise TypeError(f"TerminalPort {label} must be a Box or None.")
            if box is not None and any(torch.is_tensor(v) and v.requires_grad for v in (box.position,box.size,box.rotation)):
                raise NotImplementedError("Trainable terminal integration geometry changes discrete support.")
        resolved_name = str(name)
        if not resolved_name:
            raise ValueError("TerminalPort name must not be empty.")
        if wire_binding is not None and not isinstance(wire_binding, WirePortBinding):
            raise TypeError(
                f"TerminalPort {resolved_name!r} wire_binding must be a WirePortBinding or None."
            )
        if wire_binding is None:
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
                resolved_reference_plane = _fixed_reference_plane(reference_plane)
            except (TypeError, ValueError) as error:
                raise type(error)(
                    f"TerminalPort {resolved_name!r} reference_plane must be a real scalar."
                ) from error
            if not math.isfinite(resolved_reference_plane):
                raise ValueError(
                    f"TerminalPort {resolved_name!r} reference_plane must be finite."
                )
        else:
            if any(
                value is not None
                for value in (positive_terminal, negative_terminal, integration_path, current_surface, voltage_aperture)
            ):
                raise ValueError(
                    f"TerminalPort {resolved_name!r} wire binding must not also specify "
                    "structure terminal geometry."
                )
            if reference_plane is None:
                resolved_reference_plane = None
            else:
                resolved_reference_plane = _fixed_reference_plane(reference_plane)
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
        object.__setattr__(self, "wire_binding", wire_binding)
        object.__setattr__(self, "kind", "terminal_port")
        object.__setattr__(self, "phasor_convention", "peak")
        object.__setattr__(self, "power_convention", "0.5*Re(V*conj(I))")
        object.__setattr__(self, "_positive", None)
        object.__setattr__(self, "_negative", None)
        object.__setattr__(self, "_current_surface", None)
        object.__setattr__(self, "_declared_current_surface", current_surface)
        object.__setattr__(self, "voltage_aperture", voltage_aperture)
        object.__setattr__(self, "_terminal_audit", {})
        object.__setattr__(self, "_positive_roundoff", ())
        object.__setattr__(self, "_negative_roundoff", ())

    @property
    def voltage_path(self) -> AxisPath | None:
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
        positive_roundoff=(),
        negative_roundoff=(),
        current_bound_roundoff=(),
        terminal_audit=None,
    ) -> "TerminalPort":
        resolved = TerminalPort(
            self.name,
            positive_terminal=self.positive_terminal,
            negative_terminal=self.negative_terminal,
            integration_path=self.integration_path,
            reference_plane=self.reference_plane,
            reference_impedance=self.reference_impedance,
            termination=self.termination,
            wire_binding=self.wire_binding,
            current_surface=self._declared_current_surface,
            voltage_aperture=self.voltage_aperture,
        )
        object.__setattr__(resolved, "_terminal_audit", dict(terminal_audit or {}))
        object.__setattr__(resolved, "_positive", positive)
        object.__setattr__(resolved, "_negative", negative)
        object.__setattr__(resolved, "_current_surface", current_surface)
        object.__setattr__(resolved, "_current_bound_roundoff", current_bound_roundoff)
        object.__setattr__(resolved, "_positive_roundoff", positive_roundoff)
        object.__setattr__(resolved, "_negative_roundoff", negative_roundoff)
        return resolved


def _resolve_terminal_port(port: TerminalPort, structures, domain_bounds) -> TerminalPort:
    """Resolve fixed conductor support, including real polygon holes and contacts."""
    from .geometry.terminal_support import resolve_named_terminals
    try:
        return port._with_resolved_geometry(**resolve_named_terminals(port,structures,domain_bounds))
    except (TypeError,ValueError,NotImplementedError) as error:
        raise type(error)(f"TerminalPort {port.name!r}: {error}") from error


_WAVE_MODE_FAMILIES = {"tem", "te", "tm", "hybrid"}
_WAVE_IMPEDANCE_FORMULAS = {
    "voltage_current": "V/I",
    "te_wave": "omega*mu/beta",
    "tm_wave": "beta/(omega*epsilon)",
    "power_voltage": "abs(V)**2/(2*P)",
    "power_current": "2*P/abs(I)**2",
}


def _normalize_wave_voltage_path(value):
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError(
            "WaveModeSpec voltage_path must contain (negative_point, positive_point)."
        )
    negative = _require_length3("voltage_path negative point", value[0])
    positive = _require_length3("voltage_path positive point", value[1])
    if any(not math.isfinite(component) for point in (negative, positive) for component in point):
        raise ValueError("WaveModeSpec voltage_path points must be finite.")
    varying_axes = [
        axis
        for axis, (negative_value, positive_value) in enumerate(zip(negative, positive))
        if not math.isclose(negative_value, positive_value, rel_tol=0.0, abs_tol=1.0e-12)
    ]
    if len(varying_axes) != 1:
        raise ValueError(
            "WaveModeSpec voltage_path must be a nonzero axis-aligned negative-to-positive path."
        )
    return negative, positive


def _normalize_wave_polarization(value):
    if value is None:
        return "auto"
    if isinstance(value, str):
        token = value.strip().lower()
        if token not in {"auto", "ex", "ey", "ez", "x", "y", "z"}:
            raise ValueError(
                "WaveModeSpec polarization must be 'auto', Ex/Ey/Ez, x/y/z, "
                "or an axis-aligned vector."
            )
        return token
    vector = polarization_vector(value)
    active = [component for component in vector if abs(float(component)) > 1.0e-9]
    if len(active) != 1:
        raise ValueError("WaveModeSpec polarization vector must be axis-aligned.")
    return vector


@dataclass(frozen=True)
class WaveModeSpec:
    """Select one RF waveguide mode and freeze its impedance definition.

    TEM and hybrid voltage/current definitions use peak phasors. The voltage
    path is ordered ``(start_point, end_point)`` and voltage is the signed
    integral of E along that arrow. Choose its direction consistently with
    the right-hand current circulation about the propagation axis, so a
    passive forward mode has positive real V/I (typically signal to ground).
    The current contour is a planar, unrotated ``Box``. ``conductor_model``
    defaults to exact ideal-conductor geometry. Explicit ``"good_conductor"``
    enables a controlled first-order finite-foil expansion with PEC-limit launch
    fields; its applicability and approximation indicators ride in run metadata.
    """

    family: str
    conductor_model: str | None
    mode_index: int
    name: str
    impedance_definition: str
    impedance_formula: str
    polarization: str | tuple[float, float, float]
    voltage_path: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ] | None = None
    current_contour: Box | None = None
    _voltage_roundoff: tuple = field(default=(), init=False, repr=False, compare=False)

    def __init__(
        self,
        family,
        mode_index=0,
        *,
        name=None,
        polarization="auto",
        impedance_definition=None,
        voltage_path=None,
        current_contour=None,
        conductor_model=None,
    ):
        resolved_family = str(family).strip().lower()
        if resolved_family not in _WAVE_MODE_FAMILIES:
            raise ValueError("WaveModeSpec family must be 'tem', 'te', 'tm', or 'hybrid'.")
        resolved_index = int(mode_index)
        if resolved_index < 0:
            raise ValueError("WaveModeSpec mode_index must be >= 0.")
        resolved_name = (
            f"{resolved_family.upper()}{resolved_index}"
            if name is None
            else str(name).strip()
        )
        if not resolved_name:
            raise ValueError("WaveModeSpec name must not be empty.")
        if "::" in resolved_name:
            raise ValueError("WaveModeSpec name must not contain the reserved '::' separator.")

        resolved_voltage_path = _normalize_wave_voltage_path(voltage_path)
        resolved_polarization = _normalize_wave_polarization(polarization)
        if current_contour is not None and not isinstance(current_contour, Box):
            raise TypeError("WaveModeSpec current_contour must be a witwin.core.Box or None.")

        default_definition = {
            "tem": "voltage_current",
            "te": "te_wave",
            "tm": "tm_wave",
        }.get(resolved_family)
        resolved_definition = (
            default_definition
            if impedance_definition is None
            else str(impedance_definition).strip().lower()
        )
        if resolved_family == "tem":
            if resolved_definition != "voltage_current":
                raise ValueError(
                    "WaveModeSpec TEM impedance_definition must be 'voltage_current'."
                )
            if resolved_voltage_path is None or current_contour is None:
                raise ValueError(
                    "WaveModeSpec TEM requires both voltage_path and current_contour."
                )
        elif resolved_family == "te":
            if resolved_definition != "te_wave":
                raise ValueError("WaveModeSpec TE impedance_definition must be 'te_wave'.")
            if resolved_voltage_path is not None or current_contour is not None:
                raise ValueError("WaveModeSpec TE does not accept TEM V/I geometry.")
        elif resolved_family == "tm":
            if resolved_definition != "tm_wave":
                raise ValueError("WaveModeSpec TM impedance_definition must be 'tm_wave'.")
            if resolved_voltage_path is not None or current_contour is not None:
                raise ValueError("WaveModeSpec TM does not accept TEM V/I geometry.")
        else:
            supported = {"voltage_current", "power_voltage", "power_current"}
            if resolved_definition not in supported:
                raise ValueError(
                    "WaveModeSpec hybrid impedance_definition must be explicitly set to "
                    "'voltage_current', 'power_voltage', or 'power_current'."
                )
            if resolved_definition in {"voltage_current", "power_voltage"} and resolved_voltage_path is None:
                raise ValueError(
                    f"WaveModeSpec hybrid {resolved_definition!r} requires voltage_path."
                )
            if resolved_definition in {"voltage_current", "power_current"} and current_contour is None:
                raise ValueError(
                    f"WaveModeSpec hybrid {resolved_definition!r} requires current_contour."
                )

        if conductor_model not in (None,"good_conductor"):
            raise ValueError("WaveModeSpec conductor_model must be None or 'good_conductor'.")
        if conductor_model is not None and resolved_family != "tem":
            raise ValueError("WaveModeSpec good_conductor requires family='tem'.")
        object.__setattr__(self,"conductor_model",conductor_model)
        object.__setattr__(self, "family", resolved_family)
        object.__setattr__(self, "mode_index", resolved_index)
        object.__setattr__(self, "name", resolved_name)
        object.__setattr__(self, "impedance_definition", resolved_definition)
        object.__setattr__(
            self,
            "impedance_formula",
            _WAVE_IMPEDANCE_FORMULAS[resolved_definition],
        )
        object.__setattr__(self, "polarization", resolved_polarization)
        object.__setattr__(self, "voltage_path", resolved_voltage_path)
        object.__setattr__(self, "_voltage_roundoff", () if voltage_path is None else tuple(_coordinate_roundoff(point) for point in voltage_path))
        object.__setattr__(self, "current_contour", current_contour)


@dataclass(frozen=True)
class WavePort:
    """Declare an axis-aligned RF modal aperture without invoking a solver."""

    name: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    direction: str
    reference_plane: float
    modes: tuple[WaveModeSpec, ...]
    normal_axis: str
    direction_sign: int
    _position_roundoff: tuple = field(default=(), init=False, repr=False, compare=False)
    _size_roundoff: tuple = field(default=(), init=False, repr=False, compare=False)
    _reference_roundoff: float = field(default=0., init=False, repr=False, compare=False)
    kind: str = "wave_port"
    phasor_convention: str = "peak"
    power_convention: str = "0.5*Re(V*conj(I))"

    def __init__(self, name, position, size, direction, reference_plane, modes):
        resolved_name = str(name).strip()
        if not resolved_name:
            raise ValueError("WavePort name must not be empty.")
        if "::" in resolved_name:
            raise ValueError("WavePort name must not contain the reserved '::' separator.")
        resolved_position = _require_length3("position", position)
        resolved_size = _require_length3("size", size)
        if any(not math.isfinite(value) for value in (*resolved_position, *resolved_size)):
            raise ValueError(f"WavePort {resolved_name!r} position and size must be finite.")
        if any(value < 0.0 for value in resolved_size):
            raise ValueError(f"WavePort {resolved_name!r} size components must be >= 0.")
        zero_indices = [
            index
            for index, value in enumerate(resolved_size)
            if math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1.0e-12)
        ]
        if len(zero_indices) != 1:
            raise ValueError(
                f"WavePort {resolved_name!r} size must contain exactly one zero component."
            )
        normal_index = zero_indices[0]
        normalized_size = list(resolved_size)
        normalized_size[normal_index] = 0.0
        resolved_size = tuple(normalized_size)
        if any(
            value <= 0.0
            for index, value in enumerate(resolved_size)
            if index != normal_index
        ):
            raise ValueError(
                f"WavePort {resolved_name!r} tangential size components must be positive."
            )
        try:
            resolved_direction = _normalize_mode_direction(direction)
        except ValueError as error:
            raise ValueError(f"WavePort {resolved_name!r} direction must be '+' or '-'.") from error
        try:
            resolved_reference_plane = _fixed_reference_plane(reference_plane)
        except (TypeError, ValueError) as error:
            raise type(error)(
                f"WavePort {resolved_name!r} reference_plane must be a real scalar."
            ) from error
        if not math.isfinite(resolved_reference_plane):
            raise ValueError(f"WavePort {resolved_name!r} reference_plane must be finite.")
        normal_position = resolved_position[normal_index]
        tolerance = (1.0e-12 * max(1.0, abs(normal_position), abs(resolved_reference_plane))
                     + _coordinate_roundoff(position)[normal_index] + _coordinate_roundoff((reference_plane,))[0])
        if not math.isclose(
            normal_position,
            resolved_reference_plane,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ValueError(
                f"WavePort {resolved_name!r} reference_plane must equal the aperture plane."
            )

        try:
            resolved_modes = tuple(modes)
        except TypeError as error:
            raise TypeError(
                f"WavePort {resolved_name!r} modes must be an iterable of WaveModeSpec instances."
            ) from error
        if not resolved_modes:
            raise ValueError(f"WavePort {resolved_name!r} modes must not be empty.")
        if any(not isinstance(mode, WaveModeSpec) for mode in resolved_modes):
            raise TypeError(
                f"WavePort {resolved_name!r} modes must contain only WaveModeSpec instances."
            )
        mode_names = tuple(mode.name for mode in resolved_modes)
        if len(set(mode_names)) != len(mode_names):
            raise ValueError(f"WavePort {resolved_name!r} mode names must be unique.")

        normal_axis = "xyz"[normal_index]
        for mode in resolved_modes:
            try:
                _resolve_mode_source_polarization_axis(
                    normal_axis,
                    resolved_size,
                    mode.polarization,
                )
            except ValueError as error:
                raise ValueError(
                    f"WavePort {resolved_name!r} mode {mode.name!r} polarization: {error}"
                ) from error
        object.__setattr__(self, "name", resolved_name)
        object.__setattr__(self, "_position_roundoff", _coordinate_roundoff(position))
        object.__setattr__(self, "_size_roundoff", _coordinate_roundoff(size))
        object.__setattr__(self, "_reference_roundoff", _coordinate_roundoff((reference_plane,))[0])
        object.__setattr__(self, "position", resolved_position)
        object.__setattr__(self, "size", resolved_size)
        object.__setattr__(self, "direction", resolved_direction)
        object.__setattr__(self, "reference_plane", resolved_reference_plane)
        object.__setattr__(self, "modes", resolved_modes)
        object.__setattr__(self, "normal_axis", normal_axis)
        object.__setattr__(self, "direction_sign", 1 if resolved_direction == "+" else -1)
        object.__setattr__(self, "kind", "wave_port")
        object.__setattr__(self, "phasor_convention", "peak")
        object.__setattr__(self, "power_convention", "0.5*Re(V*conj(I))")

    def mode_name(self, mode: WaveModeSpec) -> str:
        if mode not in self.modes:
            raise ValueError(f"WaveModeSpec {mode.name!r} is not declared by WavePort {self.name!r}.")
        return f"{self.name}::{mode.name}"


_TERMINAL_IMPEDANCE_DEFINITIONS = {
    "power_voltage": "Zc = V V^H / (2 P)",
    "power_current": "Zc = 2 P (I I^H)^-1",
    "voltage_current": "Zc = V I^-1",
}


def _normalize_wave_aperture(label, name, position, size, direction, reference_plane):
    """Validate one planar axis-aligned modal aperture (shared by terminal wave ports)."""
    resolved_position = _require_length3("position", position)
    resolved_size = _require_length3("size", size)
    if any(not math.isfinite(value) for value in (*resolved_position, *resolved_size)):
        raise ValueError(f"{label} {name!r} position and size must be finite.")
    if any(value < 0.0 for value in resolved_size):
        raise ValueError(f"{label} {name!r} size components must be >= 0.")
    zero_indices = [
        index for index, value in enumerate(resolved_size)
        if math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1.0e-12)
    ]
    if len(zero_indices) != 1:
        raise ValueError(f"{label} {name!r} size must contain exactly one zero component.")
    normal_index = zero_indices[0]
    normalized_size = list(resolved_size)
    normalized_size[normal_index] = 0.0
    if any(value <= 0.0 for index, value in enumerate(normalized_size) if index != normal_index):
        raise ValueError(f"{label} {name!r} tangential size components must be positive.")
    try:
        resolved_direction = _normalize_mode_direction(direction)
    except ValueError as error:
        raise ValueError(f"{label} {name!r} direction must be '+' or '-'.") from error
    try:
        resolved_reference_plane = _fixed_reference_plane(reference_plane)
    except (TypeError, ValueError) as error:
        raise type(error)(f"{label} {name!r} reference_plane must be a real scalar.") from error
    if not math.isfinite(resolved_reference_plane):
        raise ValueError(f"{label} {name!r} reference_plane must be finite.")
    normal_position = resolved_position[normal_index]
    tolerance = (1.0e-12 * max(1.0, abs(normal_position), abs(resolved_reference_plane))
                 + _coordinate_roundoff(position)[normal_index] + _coordinate_roundoff((reference_plane,))[0])
    if not math.isclose(normal_position, resolved_reference_plane, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{label} {name!r} reference_plane must equal the aperture plane.")
    return resolved_position, tuple(normalized_size), resolved_direction, resolved_reference_plane, normal_index


def _normalize_terminal_refs(label, name, value, *, role):
    refs = (value,) if isinstance(value, TerminalRef) else value
    try:
        refs = tuple(refs)
    except TypeError as error:
        raise TypeError(f"{label} {name!r} {role} must be TerminalRef entries.") from error
    if not refs or any(not isinstance(ref, TerminalRef) for ref in refs):
        raise TypeError(f"{label} {name!r} {role} must be one or more TerminalRef entries.")
    if len(set(refs)) != len(refs):
        raise ValueError(f"{label} {name!r} {role} must reference distinct structures.")
    return refs


def _normalize_reference_impedance_matrix(label, name, value, count):
    """Return an ``N x N`` reference-impedance matrix as nested Python numbers.

    A scalar or length-N value is the diagonal special case. A full matrix couples
    terminals (for example the terminal-basis image of differential/common
    references) and must then be real, symmetric and positive definite so the
    generalized power waves conserve power.
    """
    require_fixed_input(value, name=f"{label} reference_impedance")
    raw = value.detach().cpu().numpy() if torch.is_tensor(value) else value
    array = np.asarray(raw, dtype=np.complex128)
    if array.ndim == 0:
        matrix = np.diag(np.full(count, complex(array)))
    elif array.shape == (count,):
        matrix = np.diag(array)
    elif array.shape == (count, count):
        matrix = array.copy()
    else:
        raise ValueError(
            f"{label} {name!r} reference_impedance must be a scalar, a length-{count} "
            f"vector or a {count}x{count} matrix."
        )
    if not bool(np.all(np.isfinite(matrix))):
        raise ValueError(f"{label} {name!r} reference_impedance must be finite.")
    if not bool(np.all(np.real(np.diag(matrix)) > 0.0)):
        raise ValueError(f"{label} {name!r} reference_impedance diagonal real parts must be positive.")
    off_diagonal = matrix - np.diag(np.diag(matrix))
    if bool(np.any(off_diagonal != 0.0)):
        if bool(np.any(np.imag(matrix) != 0.0)):
            raise ValueError(
                f"{label} {name!r} a coupled (non-diagonal) reference_impedance matrix must be real."
            )
        real = np.real(matrix)
        if not np.allclose(real, real.T, rtol=1.0e-12, atol=0.0):
            raise ValueError(f"{label} {name!r} reference_impedance matrix must be symmetric.")
        if not bool(np.all(np.linalg.eigvalsh(.5 * (real + real.T)) > 0.0)):
            raise ValueError(f"{label} {name!r} reference_impedance matrix must be positive definite.")
    return tuple(
        tuple(float(item.real) if item.imag == 0.0 else complex(item) for item in row)
        for row in matrix
    )


@dataclass(frozen=True)
class TerminalWavePort:
    """Declare a multi-conductor quasi-TEM wave port between named conductors.

    ``terminals`` are the N signal conductors and ``reference`` the return
    conductor(s); every conductor crossing the aperture must be one of them, and
    the aperture boundary is grounded. The port solves the quasi-static terminal
    capacitance matrices ``C`` / ``C0`` of the cross-section, launches the N
    quasi-TEM modes of ``C v = eps_eff C0 v`` (``L = mu0 eps0 C0^-1``) as modal
    fields, and recovers terminal voltages/currents from the modal amplitudes.

    ``voltage_integrals[k]`` (``CustomVoltageIntegral`` / ``CompositeVoltageIntegral``)
    defines terminal ``k``'s voltage on the actual mode field; declare each path
    from the terminal to the reference. A default path is resolved only for one
    signal over one reference conductor that are separated along exactly one
    tangential axis and overlap along the other (microstrip, stripline half): the
    straight segment from the signal surface to the reference surface through the
    centre of the footprint overlap, snapped to the nearest Yee node. CPW, multiple
    references and multiple terminals require explicit paths.

    ``impedance_definition`` fixes the characteristic matrix from the actual mode
    field: ``"power_voltage"`` (default, ``Zc = V V^H / 2`` per unit modal power),
    ``"power_current"`` (needs ``current_integrals``) or ``"voltage_current"``
    (needs both). The quasi-TEM ``C``/``C0`` impedance is reported as a cross-check.

    ``basis="terminal"`` exposes channels ``"<port>::T<k>"`` (1-based) normalized
    to ``reference_impedance`` (scalar, per-terminal vector or a full real
    symmetric positive-definite matrix). ``basis="modal"`` exposes channels
    ``"<port>::M<m>"`` as tracked one-watt modal power waves.
    """

    name: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    direction: str
    reference_plane: float
    terminals: tuple[TerminalRef, ...]
    reference: tuple[TerminalRef, ...]
    voltage_integrals: tuple | None
    current_integrals: tuple | None
    basis: str
    reference_impedance: tuple | None
    impedance_definition: str
    conductor_model: str | None
    normal_axis: str
    direction_sign: int
    _position_roundoff: tuple = field(default=(), init=False, repr=False, compare=False)
    _size_roundoff: tuple = field(default=(), init=False, repr=False, compare=False)
    _reference_roundoff: float = field(default=0., init=False, repr=False, compare=False)
    _default_voltage_path: dict | None = field(default=None, init=False, repr=False, compare=False)
    kind: str = "terminal_wave_port"
    phasor_convention: str = "peak"
    power_convention: str = "0.5*Re(V*conj(I))"

    def __init__(
        self,
        name,
        position,
        size,
        direction,
        reference_plane,
        terminals,
        reference,
        *,
        voltage_integrals=None,
        current_integrals=None,
        basis="terminal",
        reference_impedance=None,
        impedance_definition="power_voltage",
        conductor_model=None,
    ):
        from .path_integrals import CURRENT_INTEGRAL_TYPES, VOLTAGE_INTEGRAL_TYPES

        label = "TerminalWavePort"
        resolved_name = str(name).strip()
        if not resolved_name:
            raise ValueError("TerminalWavePort name must not be empty.")
        if "::" in resolved_name:
            raise ValueError("TerminalWavePort name must not contain the reserved '::' separator.")
        position_value, size_value, direction_value, plane_value, normal_index = _normalize_wave_aperture(
            label, resolved_name, position, size, direction, reference_plane)
        terminal_refs = _normalize_terminal_refs(label, resolved_name, terminals, role="terminals")
        reference_refs = _normalize_terminal_refs(label, resolved_name, reference, role="reference")
        if set(terminal_refs) & set(reference_refs):
            raise ValueError(f"{label} {resolved_name!r} terminals and reference must be distinct structures.")
        count = len(terminal_refs)

        def integrals(value, allowed, role):
            if value is None:
                return None
            try:
                resolved = tuple(value)
            except TypeError as error:
                raise TypeError(f"{label} {resolved_name!r} {role} must be a sequence.") from error
            if len(resolved) != count:
                raise ValueError(f"{label} {resolved_name!r} {role} must contain one entry per terminal.")
            if any(not isinstance(item, allowed) for item in resolved):
                raise TypeError(
                    f"{label} {resolved_name!r} {role} entries must be "
                    f"{' or '.join(kind.__name__ for kind in allowed)}."
                )
            return resolved

        voltage_value = integrals(voltage_integrals, VOLTAGE_INTEGRAL_TYPES, "voltage_integrals")
        current_value = integrals(current_integrals, CURRENT_INTEGRAL_TYPES, "current_integrals")
        definition = str(impedance_definition).strip().lower()
        if definition not in _TERMINAL_IMPEDANCE_DEFINITIONS:
            raise ValueError(
                f"{label} {resolved_name!r} impedance_definition must be 'power_voltage', "
                "'power_current' or 'voltage_current'."
            )
        if definition in {"power_current", "voltage_current"} and current_value is None:
            raise ValueError(f"{label} {resolved_name!r} {definition!r} requires current_integrals.")
        if voltage_value is None and (count != 1 or len(reference_refs) != 1):
            raise ValueError(
                f"{label} {resolved_name!r} has {count} terminal(s) and {len(reference_refs)} "
                "reference conductor(s); a default voltage path is defined only for one signal "
                "over one reference. Supply explicit voltage_integrals (a polyline per terminal "
                "for CPW and multi-ground lines)."
            )
        if definition == "voltage_current" and voltage_value is None:
            raise ValueError(f"{label} {resolved_name!r} 'voltage_current' requires voltage_integrals.")
        resolved_basis = str(basis).strip().lower()
        if resolved_basis not in {"terminal", "modal"}:
            raise ValueError(f"{label} {resolved_name!r} basis must be 'terminal' or 'modal'.")
        if resolved_basis == "modal":
            if reference_impedance is not None:
                raise ValueError(
                    f"{label} {resolved_name!r} modal basis channels are one-watt modal power "
                    "waves; reference_impedance applies to the terminal basis only."
                )
            impedance_value = None
        elif isinstance(reference_impedance, str):
            if reference_impedance.strip().lower() != "characteristic":
                raise ValueError(
                    f"{label} {resolved_name!r} reference_impedance string must be 'characteristic'."
                )
            impedance_value = "characteristic"
        else:
            impedance_value = _normalize_reference_impedance_matrix(
                label, resolved_name, 50.0 if reference_impedance is None else reference_impedance, count)
        if conductor_model not in (None, "good_conductor"):
            raise ValueError(f"{label} {resolved_name!r} conductor_model must be None or 'good_conductor'.")

        object.__setattr__(self, "name", resolved_name)
        object.__setattr__(self, "position", position_value)
        object.__setattr__(self, "size", size_value)
        object.__setattr__(self, "direction", direction_value)
        object.__setattr__(self, "reference_plane", plane_value)
        object.__setattr__(self, "terminals", terminal_refs)
        object.__setattr__(self, "reference", reference_refs)
        object.__setattr__(self, "voltage_integrals", voltage_value)
        object.__setattr__(self, "current_integrals", current_value)
        object.__setattr__(self, "basis", resolved_basis)
        object.__setattr__(self, "reference_impedance", impedance_value)
        object.__setattr__(self, "impedance_definition", definition)
        object.__setattr__(self, "conductor_model", conductor_model)
        object.__setattr__(self, "normal_axis", "xyz"[normal_index])
        object.__setattr__(self, "direction_sign", 1 if direction_value == "+" else -1)
        object.__setattr__(self, "_position_roundoff", _coordinate_roundoff(position))
        object.__setattr__(self, "_size_roundoff", _coordinate_roundoff(size))
        object.__setattr__(self, "_reference_roundoff", _coordinate_roundoff((reference_plane,))[0])
        object.__setattr__(self, "_default_voltage_path", None)
        object.__setattr__(self, "kind", "terminal_wave_port")
        object.__setattr__(self, "phasor_convention", "peak")
        object.__setattr__(self, "power_convention", "0.5*Re(V*conj(I))")

    @property
    def terminal_count(self) -> int:
        return len(self.terminals)

    @property
    def channel_names(self) -> tuple[str, ...]:
        prefix = "T" if self.basis == "terminal" else "M"
        return tuple(f"{self.name}::{prefix}{index + 1}" for index in range(self.terminal_count))

    @property
    def mode_names(self) -> tuple[str, ...]:
        return tuple(f"M{index + 1}" for index in range(self.terminal_count))

    @property
    def terminal_names(self) -> tuple[str, ...]:
        return tuple(f"T{index + 1}" for index in range(self.terminal_count))

    def _with_default_voltage_path(self, record) -> "TerminalWavePort":
        resolved = copy.copy(self)
        object.__setattr__(resolved, "_default_voltage_path", dict(record))
        return resolved


def _resolve_terminal_wave_port(port: TerminalWavePort, structures) -> TerminalWavePort:
    """Check conductor support and resolve the documented default voltage path."""
    from .geometry.terminal_support import resolve_wave_terminal_default_path, wave_terminal_supports
    try:
        wave_terminal_supports(port, structures)
        if port.voltage_integrals is not None:
            return port
        return port._with_default_voltage_path(resolve_wave_terminal_default_path(port, structures))
    except (TypeError, ValueError, NotImplementedError) as error:
        raise type(error)(f"TerminalWavePort {port.name!r}: {error}") from error


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


@dataclass(frozen=True)
class CoaxialLumpedPort:
    """Annular lumped port between two concentric circles on a Yee node plane.

    The port surface is the annulus ``inner_diameter / 2 < r < outer_diameter / 2``
    in the plane normal to ``normal_axis`` through ``center``; that plane must be
    a primal Yee node plane. The excitation is one radial current ``I`` with
    surface density ``I / (2 pi r)``. The power-dual voltage ``V`` is the
    azimuthal mean of the inner-to-outer radial line integral of E, which equals
    the radial line voltage of a TEM coaxial field. Driven or terminated ports
    report the coupled branch current entering the field network; a passive
    observer port reports the H circulation around the inner conductor on the
    ``direction`` side of the plane. Phasors are peak-valued and average power
    is ``0.5 * Re(V * conj(I))``. Only series terminations are supported.
    """

    name: str
    center: tuple[float, float, float]
    inner_diameter: float
    outer_diameter: float
    normal_axis: str
    direction: str
    reference_impedance: complex | float | torch.Tensor
    reference_plane: float | None = None
    termination: SeriesRLC | None = None
    kind: str = "coaxial_lumped_port"
    phasor_convention: str = "peak"
    power_convention: str = "0.5*Re(V*conj(I))"
    _center_roundoff: tuple[float, ...] = field(default=(), init=False, repr=False, compare=False)

    def __init__(
        self,
        name,
        *,
        center,
        inner_diameter,
        outer_diameter,
        normal_axis,
        direction="+",
        reference_impedance=50.0,
        reference_plane=None,
        termination=None,
    ):
        from .lumped import _coaxial_annulus

        resolved = _coaxial_annulus(
            "CoaxialLumpedPort", name, center, inner_diameter, outer_diameter, normal_axis
        )
        try:
            resolved_direction = _normalize_mode_direction(direction)
        except ValueError as error:
            raise ValueError(f"CoaxialLumpedPort {resolved[0]!r} direction must be '+' or '-'.") from error
        if termination is not None and not isinstance(termination, SeriesRLC):
            raise TypeError(
                f"CoaxialLumpedPort {resolved[0]!r} termination must be a SeriesRLC or None."
            )
        object.__setattr__(self, "name", resolved[0])
        object.__setattr__(self, "center", resolved[1])
        object.__setattr__(self, "_center_roundoff", _coordinate_roundoff(center))
        object.__setattr__(self, "inner_diameter", resolved[2])
        object.__setattr__(self, "outer_diameter", resolved[3])
        object.__setattr__(self, "normal_axis", resolved[4])
        object.__setattr__(self, "direction", resolved_direction)
        object.__setattr__(self, "reference_impedance", _normalize_reference_impedance(reference_impedance))
        object.__setattr__(
            self,
            "reference_plane",
            None if reference_plane is None else _fixed_reference_plane(reference_plane),
        )
        object.__setattr__(self, "termination", termination)
        object.__setattr__(self, "kind", "coaxial_lumped_port")
        object.__setattr__(self, "phasor_convention", "peak")
        object.__setattr__(self, "power_convention", "0.5*Re(V*conj(I))")

    @property
    def direction_sign(self) -> int:
        return 1 if self.direction == "+" else -1


@dataclass(frozen=True)
class PECFrame:
    """Thin PEC frame around a modal injection aperture.

    The frame is the four tangential side walls of the snapped aperture
    rectangle, each an ideal zero-thickness :class:`PlanarPEC` sheet on a Yee
    node plane, extending ``length`` primal cells from the aperture plane in the
    injection direction. It matches a cross-section mode solved with PEC
    aperture walls, so the launched field has no unsupported tangential E at the
    aperture edge. ``structures`` materializes the frame for a declared
    :class:`WavePort`, :class:`ModePort` or :class:`ModeSource`; add the returned
    structures to the scene with ``Scene.add_structure``.
    """

    length: int = 2
    kind: str = "pec_frame"

    def __init__(self, length=2):
        if isinstance(length, bool) or not isinstance(length, int):
            raise TypeError("PECFrame length must be a positive integer number of cells.")
        if length <= 0:
            raise ValueError("PECFrame length must be a positive integer number of cells.")
        object.__setattr__(self, "length", int(length))
        object.__setattr__(self, "kind", "pec_frame")

    def structures(self, port, scene) -> tuple:
        """Return the named PlanarPEC wall structures for ``port`` on ``scene``'s grid."""
        from witwin.core import Structure

        from .compiler.ports import _snap_index
        from .geometry.planar_pec import PlanarPEC
        from .media import Material
        from .scene import prepare_scene

        name = getattr(port, "name", None)
        position = getattr(port, "position", None)
        size = getattr(port, "size", None)
        direction = getattr(port, "direction", None)
        if name is None or position is None or size is None or direction not in ("+", "-"):
            raise TypeError("PECFrame requires a WavePort, ModePort or ModeSource with position, size and direction.")
        position = _require_length3("position", position)
        size = _require_length3("size", size)
        zero = [index for index, value in enumerate(size) if value == 0.0]
        if len(zero) != 1:
            raise ValueError(f"PECFrame target {name!r} must be a planar aperture with one zero size component.")
        normal = zero[0]
        prepared = prepare_scene(scene)
        nodes = [np.asarray(getattr(prepared, f"{axis}_nodes64"), dtype=np.float64) for axis in "xyz"]
        plane = _snap_index(nodes[normal], position[normal], label=f"{'xyz'[normal]} PECFrame aperture plane")
        sign = 1 if direction == "+" else -1
        far = plane + sign * self.length
        if not 0 <= far < nodes[normal].size:
            raise ValueError(f"PECFrame of length {self.length} around {name!r} leaves the grid along {'xyz'[normal]}.")
        normal_bounds = sorted((float(nodes[normal][plane]), float(nodes[normal][far])))
        bounds = {}
        for axis in range(3):
            if axis == normal:
                continue
            lower = _snap_index(nodes[axis], position[axis] - 0.5 * size[axis],
                                label=f"{'xyz'[axis]} PECFrame aperture lower bound")
            upper = _snap_index(nodes[axis], position[axis] + 0.5 * size[axis],
                                label=f"{'xyz'[axis]} PECFrame aperture upper bound")
            bounds[axis] = (float(nodes[axis][lower]), float(nodes[axis][upper]))
        result = []
        for wall_axis in bounds:
            other = next(axis for axis in bounds if axis != wall_axis)
            extents = {other: bounds[other], normal: tuple(normal_bounds)}
            first, second = sorted(extents)
            (a0, a1), (b0, b1) = extents[first], extents[second]
            vertices = ((a0, b0), (a1, b0), (a1, b1), (a0, b1))
            for side, coordinate in zip(("-", "+"), bounds[wall_axis]):
                result.append(Structure(
                    PlanarPEC(vertices, normal_axis="xyz"[wall_axis], coordinate=coordinate),
                    Material.pec(),
                    name=f"{name}::pec_frame::{'xyz'[wall_axis]}{side}",
                    tags=("pec_frame",),
                    metadata={"pec_frame": {"port": name, "length": self.length, "direction": direction}},
                ))
        return tuple(result)
