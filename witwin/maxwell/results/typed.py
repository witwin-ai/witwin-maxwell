"""Physics-typed result data.

Each type fixes the dimension order of its values; any order-preserving subset
is admitted, so point, plane and selected views keep their type. Frequency-domain
types use peak phasors with ``exp(-i*omega*t)`` time dependence; producers record
it in ``attrs["phasor_convention"]`` and the spectral scaling in
``attrs["normalization"]`` (``"raw"`` or ``"per_unit_source_spectrum"``).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Iterable

import torch

from .tensor_data import DERIVING, TensorData

FIELD_KIND_UNITS = {"E": "V/m", "H": "A/m"}
DIRECTION_ATTRS = ("ux", "uy", "uz")
# |u| = 1 within 64 float32 epsilons: unit vectors normalised in float32 pass.
UNIT_VECTOR_TOLERANCE = 64 * torch.finfo(torch.float32).eps


def field_component_units(components: Iterable[str]) -> tuple[str, dict[str, str]]:
    """``(units, component_units)`` for field component labels such as ``("Ex", "Hy")``.

    Labels starting with ``E`` are in V/m and with ``H`` in A/m. ``units`` is that
    unit when every component shares it and ``"mixed"`` otherwise; producers
    store ``component_units`` in ``attrs["component_units"]`` in both cases.
    """

    labels = tuple(components)
    if not labels or any(not isinstance(label, str) or label[:1] not in FIELD_KIND_UNITS for label in labels):
        raise ValueError(f"Field components must be non-empty E*/H* labels, got {labels!r}.")
    component_units = {label: FIELD_KIND_UNITS[label[0]] for label in labels}
    kinds = set(component_units.values())
    return (kinds.pop() if len(kinds) == 1 else "mixed"), component_units


@dataclass(frozen=True, eq=False, repr=False)
class FieldData(TensorData):
    """Electric or magnetic field samples over frequency (phasors) or time; ``component``
    labels the vector components.

    ``attrs["yee_location"]`` records where the samples live on the Yee grid and
    ``attrs["component_units"]`` the unit of each component (see ``field_component_units``);
    data mixing E and H components has ``units="mixed"``. Selecting components
    recomputes ``units`` from ``component_units``, and selecting a single
    component (dropping the dimension) names it in ``attrs["component"]``, the
    form of a one-component ``FieldData`` such as an FDFD ``Result.fields`` entry.
    """

    LAYOUTS: ClassVar[tuple[tuple[str, ...], ...]] = (
        ("component", "f", "x", "y", "z"),
        ("component", "t", "x", "y", "z"),
    )

    def __post_init__(self):
        super().__post_init__()
        if self.units == "mixed" and "component_units" not in self.attrs:
            raise ValueError("FieldData with units='mixed' needs attrs['component_units']; build both with field_component_units().")

    def isel(self, **indexers) -> "FieldData":
        selected = super().isel(**indexers)
        labels = self.coords.get("component")
        component_units = self.attrs.get("component_units")
        if "component" not in indexers or labels is None or component_units is None:
            return selected
        attrs = dict(selected.attrs)
        if "component" in selected.dims:
            kept = selected.coords["component"]
        else:
            kept = (labels[int(indexers["component"])],)
            attrs["component"] = kept[0]
        attrs["component_units"] = {label: component_units[label] for label in kept}
        kinds = set(attrs["component_units"].values())
        return selected._derive(units=kinds.pop() if len(kinds) == 1 else "mixed", attrs=attrs)


@dataclass(frozen=True, eq=False, repr=False)
class FluxData(TensorData):
    """Power flux through a monitor surface per frequency, per diffraction order and
    frequency, or over time (instantaneous flux).

    In the ``(order, f)`` layout the ``order`` coordinate is the order index and
    the per-order attrs (``m``, ``n``, the wave vector, angles,
    ``propagating``, ``efficiency`` and the order amplitudes) are selected with it.
    """

    units: str = "W"
    LAYOUTS: ClassVar[tuple[tuple[str, ...], ...]] = (("order", "f"), ("t",))
    INDEXED_ATTRS: ClassVar = MappingProxyType({"order": (
        "m", "n", "kx", "ky", "kz", "theta", "phi", "propagating", "efficiency", "amplitude_E", "amplitude_H")})


@dataclass(frozen=True, eq=False, repr=False)
class ModeAmplitudeData(TensorData):
    """Modal decomposition amplitudes per mode, propagation sense and frequency.

    The ``propagation`` labels are ``"+"`` and ``"-"`` (``mw.monitors.Direction``
    values) along the monitor normal; ``direction`` is reserved for unit vectors
    (``FarFieldData``).

    ``attrs["power"]`` (W) and ``attrs["power_fraction"]`` (of the net power
    through the monitor), when present, have the shape of the values and are
    selected with them; per-frequency attrs (``mode_power``, ``total_power``,
    ``effective_index``, ``beta``, ...) are selected with ``f``.
    """

    LAYOUTS: ClassVar[tuple[tuple[str, ...], ...]] = (("mode", "propagation", "f"),)
    VALUE_ATTRS: ClassVar[tuple[str, ...]] = ("power", "power_fraction")
    INDEXED_ATTRS: ClassVar = MappingProxyType({"f": (
        "mode_power", "total_power", "effective_index", "beta", "beta_complex", "background_index")})


@dataclass(frozen=True, eq=False, repr=False)
class SParameterData(TensorData):
    """Scattering parameters in ``[f, out_port, in_port]`` order; ``attrs`` carry ``z0`` and ``port_names``."""

    LAYOUTS: ClassVar[tuple[tuple[str, ...], ...]] = (("f", "out_port", "in_port"),)


@dataclass(frozen=True, eq=False, repr=False)
class FarFieldData(TensorData):
    """Far-field samples on an angular ``(theta, phi)`` grid, a direction-cosine ``(ux, uy)`` grid
    or a list of unit directions.

    In the ``(f, direction, component)`` layout the ``direction`` coordinate is
    the direction index and ``attrs["ux"]``, ``attrs["uy"]``, ``attrs["uz"]``
    hold the unit vectors, one entry per direction, on the values' device.
    Selecting along ``direction`` selects them too (an integer leaves 0-d
    tensors). Build it with ``FarFieldData.from_directions``.
    """

    LAYOUTS: ClassVar[tuple[tuple[str, ...], ...]] = (
        ("f", "theta", "phi", "component"),
        ("f", "ux", "uy", "component"),
        ("f", "direction", "component"),
    )
    INDEXED_ATTRS: ClassVar = MappingProxyType({"direction": DIRECTION_ATTRS})

    def __post_init__(self):
        super().__post_init__()
        present = [key for key in DIRECTION_ATTRS if key in self.attrs]
        if "direction" not in self.dims and not present:
            return
        if len(present) != len(DIRECTION_ATTRS):
            raise ValueError("FarFieldData over directions needs attrs 'ux', 'uy' and 'uz'; build it with from_directions().")
        grid = [dim for dim in ("theta", "phi", "ux", "uy") if dim in self.dims]
        if grid:
            raise ValueError(f"Direction vectors in attrs belong to the (f, direction, component) layout, not a {grid!r} grid.")
        shape = (self.sizes["direction"],) if "direction" in self.dims else ()
        attrs = dict(self.attrs)
        for key in DIRECTION_ATTRS:
            vector = attrs[key]
            if not isinstance(vector, torch.Tensor) or not vector.is_floating_point():
                raise TypeError(f"attrs[{key!r}] must be a real floating torch.Tensor.")
            if tuple(vector.shape) != shape:
                raise ValueError(f"attrs[{key!r}] must have shape {shape} (one entry per direction), got {tuple(vector.shape)}.")
            attrs[key] = vector.to(self.values.device)
        if not DERIVING.get():
            # Selections and transfers of validated data skip this host-synchronising check.
            _check_unit_vectors(*(attrs[key] for key in DIRECTION_ATTRS))
        object.__setattr__(self, "attrs", MappingProxyType(attrs))

    @classmethod
    def from_directions(cls, values, directions, f, components, units: str, attrs=None) -> "FarFieldData":
        """Far field over ``D`` unit directions: ``values`` ``[f, D, component]``, ``directions`` ``[D, 3]``.

        ``components=None`` builds a scalar far field (``values`` ``[f, D]``, dims ``(f, direction)``),
        such as a projection onto one polarization.
        """

        directions = torch.as_tensor(directions)
        if directions.ndim != 2 or directions.shape[1] != 3:
            raise ValueError(f"directions must have shape [D, 3], got {tuple(directions.shape)}.")
        attrs = dict(attrs or {})
        clash = [key for key in DIRECTION_ATTRS if key in attrs]
        if clash:
            raise ValueError(f"from_directions() sets attrs {clash!r} from directions; do not pass them.")
        attrs.update(zip(DIRECTION_ATTRS, directions.unbind(1)))
        coords = {"f": f, "direction": torch.arange(directions.shape[0])}
        if components is not None:
            coords["component"] = tuple(components)
        return cls(
            values,
            ("f", "direction") if components is None else ("f", "direction", "component"),
            coords=coords,
            units=units,
            attrs=attrs,
        )


def _check_unit_vectors(ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor) -> None:
    ux, uy, uz = (component.detach().double() for component in (ux, uy, uz))
    error = ((ux.square() + uy.square() + uz.square()).sqrt() - 1.0).abs()
    if not bool((error <= UNIT_VECTOR_TOLERANCE).all()):
        raise ValueError(f"Directions must be unit vectors; |u| deviates from 1 by up to {float(error.max()):.3g}.")
