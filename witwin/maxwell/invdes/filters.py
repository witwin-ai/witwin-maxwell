"""Physical-radius density filters on a regular design grid.

A filter maps a raw design field ``x`` defined on a regular design grid with
per-axis cell spacing ``spacing`` (metres) to a smoothed field
``x_tilde = (w * x) / sum(w)``. The kernel ``w`` depends on the physical distance
``r = sqrt(sum_i (k_i * h_i)^2)`` between cells, so the same ``radius`` produces
the same physical smoothing length on anisotropic or refined design grids.

Kernels
-------
``"conic"``
    Linear hat ``w = max(0, 1 - r / radius)`` (Bruns-Tortorelli / Bourdin cone).
``"disk"``
    Uniform ``w = 1`` for ``r <= radius`` (circle in 2D, ball in 3D).
``"gaussian"``
    ``w = exp(-r^2 / (2 sigma^2))`` with ``sigma = radius / 2`` (``radius`` is the
    two-sigma width), truncated at ``r <= 3 sigma = 1.5 radius``.

All kernels are normalized to unit sum, so a constant field is preserved by every
padding mode except ``"constant"`` with a different ``pad_value``.

Boundary handling
-----------------
``padding`` selects how the field is extended past the design-grid boundary before
the correlation. It may be a single mode or one mode per axis:

``"edge"``
    Replicate the boundary cell (zero normal derivative). Default; keeps features
    that touch the design boundary, such as waveguide stubs, from eroding.
``"reflect"``
    Mirror about the boundary cell centre (the boundary cell is not repeated).
    Requires the kernel half-width to be smaller than the axis length.
``"periodic"``
    Wrap around; use for periodic unit cells. Requires the kernel half-width to
    be at most the axis length.
``"constant"``
    Extend with ``pad_value`` (``0`` = void outside, ``1`` = solid outside).
``"normalized"``
    Only in-domain cells contribute and the kernel is renormalized over the
    truncated support (the classic density-filter boundary treatment).

Per-axis modes are applied axis by axis in axis order; only the corner regions of
mixed ``"constant"`` combinations depend on that order. ``"constant"`` and
``"normalized"`` cannot be mixed in one call.

Axes of length one are treated as extruded (invariant) and are never filtered, so
a ``(Nx, Ny, 1)`` slab density is filtered as a 2D design.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

_KERNELS = ("conic", "disk", "gaussian")
_PADDING = ("edge", "reflect", "periodic", "constant", "normalized")
_TORCH_PAD_MODE = {"edge": "replicate", "reflect": "reflect", "periodic": "circular"}
_GAUSSIAN_TRUNCATE = 1.5  # support radius in units of ``radius`` (= 3 sigma)


def design_spacing(size: Sequence[float], shape: Sequence[int]) -> tuple[float, ...]:
    """Return the per-axis cell spacing of a design grid spanning ``size`` metres.

    For a ``MaterialRegion`` with ``geometry=Box(size=...)`` and a density of shape
    ``(Nx, Ny, Nz)`` this is ``(sx / Nx, sy / Ny, sz / Nz)``.
    """
    if len(size) != len(shape):
        raise ValueError("design_spacing requires size and shape of equal length.")
    spacing = tuple(float(s) / int(n) for s, n in zip(size, shape))
    if any(not math.isfinite(h) or h <= 0.0 for h in spacing):
        raise ValueError("design_spacing requires positive finite size and positive shape entries.")
    return spacing


def _normalize_spacing(spacing, ndim: int) -> tuple[float, ...]:
    if isinstance(spacing, (int, float)):
        spacing = (float(spacing),) * ndim
    spacing = tuple(float(h) for h in spacing)
    if len(spacing) != ndim:
        raise ValueError(f"spacing must be a scalar or have {ndim} entries, got {len(spacing)}.")
    if any(not math.isfinite(h) or h <= 0.0 for h in spacing):
        raise ValueError("spacing entries must be positive and finite.")
    return spacing


def _normalize_padding(padding, ndim: int) -> tuple[str, ...]:
    modes = (padding,) * ndim if isinstance(padding, str) else tuple(padding)
    if len(modes) != ndim:
        raise ValueError(f"padding must be a mode name or have {ndim} entries.")
    modes = tuple(str(mode).lower() for mode in modes)
    for mode in modes:
        if mode not in _PADDING:
            raise ValueError(f"padding mode must be one of {_PADDING}, got {mode!r}.")
    if "constant" in modes and "normalized" in modes:
        raise ValueError("padding modes 'constant' and 'normalized' cannot be mixed.")
    return modes


def filter_kernel(
    radius: float,
    spacing,
    *,
    kind: str = "conic",
    shape: Sequence[int] | None = None,
    ndim: int | None = None,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Build a unit-sum filter kernel for a physical ``radius`` on a design grid.

    ``spacing`` is the per-axis design-cell size in metres (scalar or sequence).
    ``shape`` (the design-grid shape) marks length-one axes as extruded; it also
    fixes ``ndim`` when given. The kernel half-width along axis ``i`` is
    ``floor(support / spacing[i])``, so a radius below one cell yields the
    identity kernel along that axis.
    """
    kind = str(kind).lower()
    if kind not in _KERNELS:
        raise ValueError(f"kind must be one of {_KERNELS}, got {kind!r}.")
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("filter radius must be positive and finite.")
    if shape is not None:
        ndim = len(shape)
    elif ndim is None:
        if isinstance(spacing, (int, float)):
            raise ValueError("filter_kernel needs shape or ndim when spacing is a scalar.")
        ndim = len(spacing)
    if ndim not in (2, 3):
        raise ValueError("density filters support 2D and 3D design grids only.")
    spacing = _normalize_spacing(spacing, ndim)

    support = radius * (_GAUSSIAN_TRUNCATE if kind == "gaussian" else 1.0)
    axes = []
    for axis, h in enumerate(spacing):
        extruded = shape is not None and int(shape[axis]) == 1
        half = 0 if extruded else int(math.floor(support / h + 1e-9))
        axes.append(torch.arange(-half, half + 1, device=device, dtype=dtype) * h)
    grids = torch.meshgrid(*axes, indexing="ij")
    r = torch.sqrt(sum(g * g for g in grids))

    if kind == "conic":
        weights = (1.0 - r / radius).clamp_min(0.0)
    elif kind == "disk":
        weights = (r <= radius * (1.0 + 1e-9)).to(dtype)
    else:
        sigma = radius / 2.0
        weights = torch.exp(-(r * r) / (2.0 * sigma * sigma)) * (r <= support * (1.0 + 1e-9)).to(dtype)
    return weights / weights.sum()


def _pad_axis(x: torch.Tensor, axis: int, width: int, mode: str, value: float) -> torch.Tensor:
    """Pad one spatial axis of an ``(1, 1, *spatial)`` tensor."""
    if width == 0:
        return x
    nspatial = x.ndim - 2
    pads = [0, 0] * nspatial
    # F.pad orders pairs from the last dimension backwards.
    slot = nspatial - 1 - axis
    pads[2 * slot] = width
    pads[2 * slot + 1] = width
    if mode in ("constant", "normalized"):
        return F.pad(x, pads, mode="constant", value=value)
    return F.pad(x, pads, mode=_TORCH_PAD_MODE[mode])


def apply_filter(
    x: torch.Tensor,
    kernel: torch.Tensor,
    *,
    padding="edge",
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Correlate ``x`` with a unit-sum ``kernel`` using the requested boundary padding.

    ``x`` and ``kernel`` must have the same number of dimensions (2 or 3). The
    output has the shape of ``x`` and stays differentiable with respect to ``x``.
    """
    if x.ndim != kernel.ndim or x.ndim not in (2, 3):
        raise ValueError("apply_filter requires 2D or 3D x and a kernel of matching rank.")
    modes = _normalize_padding(padding, x.ndim)
    halves = tuple(int(size) // 2 for size in kernel.shape)
    for axis, (half, mode) in enumerate(zip(halves, modes)):
        n = int(x.shape[axis])
        if mode == "reflect" and half >= n:
            raise ValueError(f"'reflect' padding needs kernel half-width < axis length on axis {axis}.")
        if mode == "periodic" and half > n:
            raise ValueError(f"'periodic' padding needs kernel half-width <= axis length on axis {axis}.")
    if all(half == 0 for half in halves):
        return x

    weight = kernel.to(device=x.device, dtype=x.dtype)[None, None]
    conv = F.conv2d if x.ndim == 2 else F.conv3d
    padded = x[None, None]
    for axis, (half, mode) in enumerate(zip(halves, modes)):
        fill = 0.0 if mode == "normalized" else float(pad_value)
        padded = _pad_axis(padded, axis, half, mode, fill)
    out = conv(padded, weight)[0, 0]
    if "normalized" not in modes:
        return out

    mask = torch.ones((1, 1, *x.shape), device=x.device, dtype=x.dtype)
    for axis, (half, mode) in enumerate(zip(halves, modes)):
        mask = _pad_axis(mask, axis, half, "normalized" if mode == "normalized" else "edge", 0.0)
    return out / conv(mask, weight)[0, 0]


class DensityFilter(torch.nn.Module):
    """``torch.nn.Module`` density filter with a physical radius.

    The kernel is rebuilt for each distinct input shape, dtype and device and
    cached; it carries no trainable state.
    """

    def __init__(self, radius: float, spacing, *, kind: str = "conic", padding="edge", pad_value: float = 0.0):
        super().__init__()
        self.radius = float(radius)
        self.spacing = spacing
        self.kind = str(kind).lower()
        self.padding = padding
        self.pad_value = float(pad_value)
        self._kernels: dict[tuple, torch.Tensor] = {}
        if self.kind not in _KERNELS:
            raise ValueError(f"kind must be one of {_KERNELS}, got {self.kind!r}.")
        if not math.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("filter radius must be positive and finite.")

    def kernel(self, shape, *, device=None, dtype=torch.float32) -> torch.Tensor:
        key = (tuple(int(n) for n in shape), torch.device(device) if device is not None else None, dtype)
        cached = self._kernels.get(key)
        if cached is None:
            cached = filter_kernel(self.radius, self.spacing, kind=self.kind, shape=key[0], device=device, dtype=dtype)
            self._kernels[key] = cached
        return cached

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.kernel(x.shape, device=x.device, dtype=x.dtype)
        return apply_filter(x, kernel, padding=self.padding, pad_value=self.pad_value)

    def extra_repr(self) -> str:
        return f"radius={self.radius:g}, kind={self.kind!r}, padding={self.padding!r}"


def conic_filter(x: torch.Tensor, radius: float, spacing, *, padding="edge", pad_value: float = 0.0) -> torch.Tensor:
    """Filter ``x`` with a conic (linear-hat) kernel of physical ``radius``."""
    kernel = filter_kernel(radius, spacing, kind="conic", shape=x.shape, device=x.device, dtype=x.dtype)
    return apply_filter(x, kernel, padding=padding, pad_value=pad_value)


def gaussian_filter(x: torch.Tensor, radius: float, spacing, *, padding="edge", pad_value: float = 0.0) -> torch.Tensor:
    """Filter ``x`` with a truncated Gaussian kernel (``sigma = radius / 2``)."""
    kernel = filter_kernel(radius, spacing, kind="gaussian", shape=x.shape, device=x.device, dtype=x.dtype)
    return apply_filter(x, kernel, padding=padding, pad_value=pad_value)


def disk_filter(x: torch.Tensor, radius: float, spacing, *, padding="edge", pad_value: float = 0.0) -> torch.Tensor:
    """Filter ``x`` with a uniform circular (2D) or spherical (3D) kernel."""
    kernel = filter_kernel(radius, spacing, kind="disk", shape=x.shape, device=x.device, dtype=x.dtype)
    return apply_filter(x, kernel, padding=padding, pad_value=pad_value)
