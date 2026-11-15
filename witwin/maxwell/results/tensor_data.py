"""Dimension-labelled result tensors.

``TensorData`` wraps one torch tensor with named dimensions, per-dimension
coordinates, units and free-form attributes. The values stay a torch tensor on
their device: construction does not copy, and ``sel``/``isel``/``to`` are torch
indexing and transfer operations, so autograd flows through them. ``to_xarray``
is the only conversion that leaves torch; it returns a detached host copy.
"""

from __future__ import annotations

import contextvars
import numbers
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

import numpy as np
import torch

NORMALIZATIONS = ("raw", "per_unit_source_spectrum")

# True while isel/sel/to rebuild data from an already validated instance, so
# subclasses skip value checks that would synchronise with the device.
DERIVING = contextvars.ContextVar("witwin_maxwell_results_deriving", default=False)


@dataclass(frozen=True, eq=False, repr=False)
class TensorData:
    """A torch tensor with named dimensions, coordinates and units.

    ``coords`` maps a dimension to a 1-D numeric tensor in SI units (moved to the
    values' device) or to a tuple of unique string labels such as field
    components or port names. A dimension without coordinates is selectable by
    position only. Subclasses fix the admitted dimension orders in ``LAYOUTS``;
    any order-preserving subset of one layout is valid, so selections that drop
    a dimension keep the type. ``INDEXED_ATTRS`` names tensor attrs that run
    along a dimension (non-dimension coordinates): ``isel``/``sel`` on that
    dimension index them with it, and ``to_xarray`` exports them as coordinates.
    ``VALUE_ATTRS`` names tensor attrs shaped like the values (derived
    quantities such as mode powers); every selection indexes them like the
    values.
    """

    values: torch.Tensor
    dims: tuple[str, ...]
    coords: Mapping[str, torch.Tensor | tuple[str, ...]] = field(default_factory=dict)
    units: str = "1"
    attrs: Mapping[str, Any] = field(default_factory=dict)

    LAYOUTS: ClassVar[tuple[tuple[str, ...], ...]] = ()
    INDEXED_ATTRS: ClassVar[Mapping[str, tuple[str, ...]]] = MappingProxyType({})
    VALUE_ATTRS: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self):
        if not isinstance(self.values, torch.Tensor):
            raise TypeError(f"{type(self).__name__}.values must be a torch.Tensor, got {type(self.values).__name__}.")
        dims = tuple(self.dims)
        if any(not isinstance(dim, str) or not dim for dim in dims) or len(set(dims)) != len(dims):
            raise ValueError(f"{type(self).__name__}.dims must be unique non-empty strings, got {dims!r}.")
        if len(dims) != self.values.ndim:
            raise ValueError(
                f"{type(self).__name__}.dims {dims!r} name {len(dims)} dimensions but values have "
                f"{self.values.ndim} (shape {tuple(self.values.shape)})."
            )
        if self.LAYOUTS and not any(_is_subsequence(dims, layout) for layout in self.LAYOUTS):
            raise ValueError(
                f"{type(self).__name__}.dims {dims!r} must keep the order of a subset of "
                + " or ".join(repr(layout) for layout in self.LAYOUTS)
                + "."
            )
        unknown = [name for name in self.coords if name not in dims]
        if unknown:
            raise ValueError(f"{type(self).__name__}.coords name dimensions {unknown!r} that are not in dims {dims!r}.")
        coords = {
            dim: _coordinate(dim, self.coords[dim], self.values.shape[axis], self.values.device)
            for axis, dim in enumerate(dims)
            if dim in self.coords
        }
        if not isinstance(self.units, str):
            raise TypeError(f"{type(self).__name__}.units must be a string, got {type(self.units).__name__}.")
        attrs = dict(self.attrs)
        if "units" in attrs:
            raise ValueError(f"{type(self).__name__} keeps units in .units, not in attrs.")
        if "normalization" in attrs and attrs["normalization"] not in NORMALIZATIONS:
            raise ValueError(f"attrs['normalization'] must be one of {NORMALIZATIONS}, got {attrs['normalization']!r}.")
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "coords", MappingProxyType(coords))
        object.__setattr__(self, "attrs", MappingProxyType(attrs))

    @property
    def shape(self) -> torch.Size:
        return self.values.shape

    @property
    def sizes(self) -> dict[str, int]:
        """Dimension name to length, in ``dims`` order."""
        return dict(zip(self.dims, self.values.shape))

    @property
    def device(self) -> torch.device:
        return self.values.device

    @property
    def dtype(self) -> torch.dtype:
        return self.values.dtype

    def isel(self, **indexers) -> "TensorData":
        """Select by position along named dimensions.

        An integer drops the dimension; a slice or a 1-D integer sequence keeps it
        with the matching coordinates. Attrs indexed along the dimension
        (``INDEXED_ATTRS``) and value-shaped attrs (``VALUE_ATTRS``) are selected with it.
        """

        values = self.values
        dims = list(self.dims)
        coords = dict(self.coords)
        attrs = dict(self.attrs)
        shaped = {key: attrs[key] for key in self.VALUE_ATTRS if isinstance(attrs.get(key), torch.Tensor)}
        for dim, index in indexers.items():
            self._axis(dim)
            axis = dims.index(dim)
            positions = None
            if not isinstance(index, slice) and not _is_integer(index):
                positions = _positions(dim, index, values.shape[axis])
            values = _index_axis(values, axis, index, positions)
            shaped = {key: _index_axis(tensor, axis, index, positions) for key, tensor in shaped.items()}
            if _is_integer(index):
                del dims[axis]
                coords.pop(dim, None)
            if dim in coords:
                coords[dim] = _take(coords[dim], index, positions)
            for key in self.INDEXED_ATTRS.get(dim, ()):
                if key in attrs:
                    attrs[key] = _take(attrs[key], index, positions)
        attrs.update(shaped)
        return self._derive(values=values, dims=tuple(dims), coords=coords, attrs=attrs)

    def sel(self, method: str | None = None, **indexers) -> "TensorData":
        """Select by coordinate value along named dimensions.

        A scalar value or label drops the dimension; a sequence keeps it. Numeric
        values must equal a coordinate exactly (after rounding to the coordinate
        dtype) unless ``method="nearest"``.
        """

        if method not in (None, "nearest"):
            raise ValueError(f"method must be None or 'nearest', got {method!r}.")
        return self.isel(**{dim: self._coordinate_index(dim, value, method) for dim, value in indexers.items()})

    def to(self, device) -> "TensorData":
        """Move values, coordinates and tensor attributes to ``device``."""

        device = torch.device(device)
        return self._derive(
            values=self.values.to(device),
            coords={dim: coord.to(device) if isinstance(coord, torch.Tensor) else coord for dim, coord in self.coords.items()},
            attrs={key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in self.attrs.items()},
        )

    def to_xarray(self):
        """Detached host copy as an ``xarray.DataArray`` (needs the optional ``xarray`` package)."""

        try:
            import xarray
        except ImportError as error:
            raise ImportError(
                f"{type(self).__name__}.to_xarray() needs the optional dependency xarray; "
                "install it with `pip install xarray` or `pip install witwin-maxwell[xarray]`."
            ) from error
        coords = {dim: _host(coord) if isinstance(coord, torch.Tensor) else list(coord) for dim, coord in self.coords.items()}
        attrs = dict(self.attrs)
        for key in self.VALUE_ATTRS:
            if isinstance(attrs.get(key), torch.Tensor):
                coords[key] = (self.dims, _host(attrs.pop(key)))
        for dim, keys in self.INDEXED_ATTRS.items():
            for key in keys:
                if key in attrs:
                    value = _host(attrs.pop(key))
                    coords[key] = (dim, value) if dim in self.dims else value
        return xarray.DataArray(
            _host(self.values),
            dims=self.dims,
            coords=coords,
            attrs={
                **{key: _host(value) if isinstance(value, torch.Tensor) else value for key, value in attrs.items()},
                "units": self.units,
            },
        )

    def __repr__(self) -> str:
        sizes = ", ".join(f"{dim}: {size}" for dim, size in self.sizes.items())
        return (
            f"{type(self).__name__}({sizes}; units={self.units!r}, "
            f"dtype={self.values.dtype}, device={self.values.device})"
        )

    def _derive(self, **changes) -> "TensorData":
        token = DERIVING.set(True)
        try:
            return replace(self, **changes)
        finally:
            DERIVING.reset(token)

    def _axis(self, dim: str) -> int:
        if dim not in self.dims:
            raise ValueError(f"{type(self).__name__} has no dimension {dim!r}; dims are {self.dims!r}.")
        return self.dims.index(dim)

    def _coordinate_index(self, dim: str, value, method: str | None):
        self._axis(dim)
        coord = self.coords.get(dim)
        if coord is None:
            raise ValueError(f"Dimension {dim!r} has no coordinates; select it by position with isel().")
        if isinstance(value, slice):
            raise TypeError(f"sel() takes coordinate values, not slices; use isel({dim}=slice(...)).")
        if isinstance(coord, tuple):
            if method is not None:
                raise ValueError(f"method={method!r} needs numeric coordinates; {dim!r} is labelled.")
            if isinstance(value, str):
                return _label_position(dim, coord, value)
            return [_label_position(dim, coord, label) for label in value]
        scalar = isinstance(value, numbers.Number) or (isinstance(value, torch.Tensor) and value.ndim == 0)
        target = torch.as_tensor(np.asarray(value.detach().cpu() if isinstance(value, torch.Tensor) else value, dtype=np.float64))
        target = target.reshape(-1).to(coord.device)
        if coord.is_floating_point():
            target = target.to(coord.dtype)
        if method == "nearest":
            positions = (coord[None, :] - target[:, None]).abs().argmin(dim=1)
        else:
            hits = coord[None, :] == target[:, None]
            missing = ~hits.any(dim=1)
            if bool(missing.any()):
                raise KeyError(f"{dim}={target[missing].tolist()} not found in the coordinates; use method='nearest'.")
            positions = hits.to(torch.int32).argmax(dim=1)
        return int(positions[0]) if scalar else positions


def _is_subsequence(dims: tuple[str, ...], layout: tuple[str, ...]) -> bool:
    remaining = iter(layout)
    return all(dim in remaining for dim in dims)


def _coordinate(dim: str, coord, size: int, device: torch.device):
    if isinstance(coord, (list, tuple)) and coord and all(isinstance(label, str) for label in coord):
        labels = tuple(coord)
        if len(set(labels)) != len(labels):
            raise ValueError(f"Coordinate labels of {dim!r} must be unique, got {labels!r}.")
        if len(labels) != size:
            raise ValueError(f"Coordinate {dim!r} has {len(labels)} labels for a dimension of length {size}.")
        return labels
    if not isinstance(coord, torch.Tensor):
        array = np.asarray(coord)
        if array.dtype.kind not in "iuf":
            raise TypeError(f"Coordinate {dim!r} must be numeric or a sequence of string labels, got {array.dtype}.")
        coord = torch.as_tensor(array)
    if coord.ndim != 1 or coord.shape[0] != size:
        raise ValueError(f"Coordinate {dim!r} must be 1-D of length {size}, got shape {tuple(coord.shape)}.")
    if coord.is_complex() or coord.dtype == torch.bool:
        raise TypeError(f"Coordinate {dim!r} must be real numeric, got {coord.dtype}.")
    return coord.to(device)


def _is_integer(index) -> bool:
    if isinstance(index, bool):
        return False
    if isinstance(index, numbers.Integral):
        return True
    return isinstance(index, torch.Tensor) and index.ndim == 0 and not index.is_floating_point() and not index.is_complex() and index.dtype != torch.bool


def _index_axis(tensor: torch.Tensor, axis: int, index, positions: torch.Tensor | None) -> torch.Tensor:
    """Apply one ``isel`` index to ``axis`` of a tensor: a slice keeps it, an integer drops it."""

    if isinstance(index, slice):
        return tensor[(slice(None),) * axis + (index,)]
    if positions is None:
        return tensor.select(axis, int(index))
    return tensor.index_select(axis, positions.to(tensor.device))


def _take(item, index, positions: torch.Tensor | None):
    """Apply one ``isel`` index to a 1-D coordinate (tensor or labels) or indexed attr."""

    if isinstance(index, slice):
        return item[index]
    if positions is None:
        return item[int(index)]
    if isinstance(item, torch.Tensor):
        return item.index_select(0, positions.to(item.device))
    return tuple(item[position] for position in positions.tolist())


def _positions(dim: str, index, size: int) -> torch.Tensor:
    positions = index.detach().cpu() if isinstance(index, torch.Tensor) else torch.as_tensor(index)
    if positions.numel() == 0:
        positions = positions.to(torch.long)
    if positions.ndim != 1 or positions.is_floating_point() or positions.is_complex() or positions.dtype == torch.bool:
        raise TypeError(f"isel({dim}=...) takes an int, a slice or a 1-D integer sequence.")
    positions = positions.to(torch.long)
    if bool(((positions < -size) | (positions >= size)).any()):
        raise IndexError(f"isel({dim}={positions.tolist()}) is out of range for length {size}.")
    return torch.where(positions < 0, positions + size, positions)


def _label_position(dim: str, labels: tuple[str, ...], label) -> int:
    try:
        return labels.index(label)
    except ValueError:
        raise KeyError(f"{dim}={label!r} not found; labels are {labels!r}.") from None


def _host(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().resolve_conj().resolve_neg().numpy()
