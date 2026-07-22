from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from ..fdtd.observers import plane_normal_poynting
from ..monitors import INCIDENT_SPATIAL_AVERAGE_VERSION

POWER_DENSITY_UNIT = "W/m^2"
FLUX_UNIT = "W"


def _plane_coord_names(axis: str) -> tuple[str, str]:
    if axis == "x":
        return ("y", "z")
    if axis == "y":
        return ("x", "z")
    return ("x", "y")


@dataclass(frozen=True)
class IncidentPowerDensity:
    """Time-averaged incident power density on a monitor plane (W/m^2).

    ``normal_poynting`` is the signed per-cell normal Poynting component
    ``S.n = 0.5 * Re((E x conj(H)).n_hat)`` oriented by the monitor normal;
    ``power_density`` is its magnitude ``|S.n|`` (the exposure incident power
    density). ``flux`` is the plane-integrated ``sum(S.n * cell_area)`` (W), which
    is identically the :class:`FluxMonitor` integral over the same plane.
    ``spatial_average`` (present only when a window area was requested) is the
    area-weighted moving-window average of ``|S.n|`` under the versioned
    ``spatial-average-v1`` window. All large arrays keep their device and dtype.
    """

    monitor: str
    axis: str
    normal_direction: str
    frequencies: tuple[float, ...]
    coordinate_names: tuple[str, str]
    coordinates: tuple[torch.Tensor, torch.Tensor]
    normal_poynting: torch.Tensor
    power_density: torch.Tensor
    flux: torch.Tensor
    spatial_average: torch.Tensor | None
    provenance: Mapping[str, Any]

    @property
    def power_density_unit(self) -> str:
        return POWER_DENSITY_UNIT

    @property
    def flux_unit(self) -> str:
        return FLUX_UNIT


def _inclusive_prefix2d(values: torch.Tensor) -> torch.Tensor:
    """Zero-padded inclusive 2D prefix sum over the last two dims."""

    prefix = torch.cumsum(values, dim=-2)
    prefix = torch.cumsum(prefix, dim=-1)
    return F.pad(prefix, (1, 0, 1, 0))


def _box_sum(prefix: torch.Tensor, lo_u, hi_u, lo_v, hi_v) -> torch.Tensor:
    def gather(iu, iv):
        return prefix.index_select(-2, iu).index_select(-1, iv)

    return gather(hi_u, hi_v) - gather(lo_u, hi_v) - gather(hi_u, lo_v) + gather(lo_u, lo_v)


def _window_index_bounds(coord: torch.Tensor, half: float) -> tuple[torch.Tensor, torch.Tensor]:
    lower = torch.searchsorted(coord, coord - half, right=False)
    upper = torch.searchsorted(coord, coord + half, right=True)
    return lower.to(torch.long), upper.to(torch.long)


def _spatial_average(
    power_density: torch.Tensor,
    coord_u: torch.Tensor,
    coord_v: torch.Tensor,
    weight_u: torch.Tensor,
    weight_v: torch.Tensor,
    area: float,
) -> torch.Tensor:
    """Area-weighted moving-window average of ``|S.n|`` (spatial-average-v1).

    The window is an axis-aligned square of side ``sqrt(area)`` centred on each
    cell. Near the plane edge it is truncated to the in-domain cells; the average
    is ``sum(|S.n| * cell_area) / sum(cell_area)`` over the window cells.
    """

    half = 0.5 * math.sqrt(area)
    weights2d = weight_u[:, None] * weight_v[None, :]
    numerator = power_density * weights2d
    num_prefix = _inclusive_prefix2d(numerator)
    den_prefix = _inclusive_prefix2d(weights2d)

    lo_u, hi_u = _window_index_bounds(coord_u, half)
    lo_v, hi_v = _window_index_bounds(coord_v, half)

    num_box = _box_sum(num_prefix, lo_u, hi_u, lo_v, hi_v)
    den_box = _box_sum(den_prefix, lo_u, hi_u, lo_v, hi_v)
    return num_box / den_box


def compute_incident_power_density(
    payload: Mapping[str, Any],
    *,
    monitor_name: str,
    spatial_average_area: float | None = None,
) -> IncidentPowerDensity:
    """Reduce a flux-enabled plane payload to incident power density.

    ``payload`` is the plane monitor payload (aligned tangential fields cropped to
    the physical plane, cell widths, axis, normal direction). The normal Poynting
    component is formed by the shared :func:`plane_normal_poynting` helper so the
    integrated flux here matches the plane-flux integral exactly.
    """

    axis = str(payload["axis"]).lower()
    normal_direction = str(payload.get("normal_direction", "+"))
    coord_names = _plane_coord_names(axis)
    frequencies = tuple(
        float(freq) for freq in payload.get("frequencies", (payload.get("frequency"),))
    )

    poynting, weights = plane_normal_poynting(payload)
    # Real FDTD payloads are torch tensors (GPU-first); a NumPy payload (only
    # produced by hand-built or serialized dicts) is lifted to a CPU tensor so the
    # reducer stays a single torch path with no separate CPU code branch.
    if not isinstance(poynting, torch.Tensor):
        poynting = torch.as_tensor(poynting)
        weights = torch.as_tensor(weights)

    power_density = torch.abs(poynting)
    flux = torch.sum(poynting * weights, dim=(-2, -1))

    coord_u = payload[coord_names[0]]
    coord_v = payload[coord_names[1]]
    if not isinstance(coord_u, torch.Tensor):
        coord_u = torch.as_tensor(coord_u, device=poynting.device, dtype=poynting.dtype)
    if not isinstance(coord_v, torch.Tensor):
        coord_v = torch.as_tensor(coord_v, device=poynting.device, dtype=poynting.dtype)

    cell_widths = payload.get("cell_widths")
    if cell_widths is not None:
        weight_u = torch.as_tensor(
            cell_widths[coord_names[0]], device=poynting.device, dtype=poynting.dtype
        )
        weight_v = torch.as_tensor(
            cell_widths[coord_names[1]], device=poynting.device, dtype=poynting.dtype
        )
    else:
        weight_u = weights.sum(dim=1) / weights.shape[1] if weights.ndim == 2 else weights
        weight_v = weights.sum(dim=0) / weights.shape[0] if weights.ndim == 2 else weights

    provenance: dict[str, Any] = {
        "monitor": monitor_name,
        "axis": axis,
        "normal_direction": normal_direction,
        "field_convention": "peak-phasor",
        "poynting_definition": "0.5*Re((E x conj(H)).n_hat)",
        "power_density_quantity": "abs(normal_poynting)",
    }

    spatial_average = None
    if spatial_average_area is not None:
        area = float(spatial_average_area)
        if not math.isfinite(area) or area <= 0.0:
            raise ValueError("spatial_average area must be a positive window area in m^2.")
        spatial_average = _spatial_average(
            power_density, coord_u, coord_v, weight_u, weight_v, area
        )
        provenance["spatial_average"] = {
            "version": INCIDENT_SPATIAL_AVERAGE_VERSION,
            "area_m2": area,
            "window_side_m": math.sqrt(area),
            "window_shape": "square",
            "edge_policy": "truncate",
            "averaged_quantity": "abs(normal_poynting)",
            "certified": False,
        }

    return IncidentPowerDensity(
        monitor=monitor_name,
        axis=axis,
        normal_direction=normal_direction,
        frequencies=frequencies,
        coordinate_names=coord_names,
        coordinates=(coord_u, coord_v),
        normal_poynting=poynting,
        power_density=power_density,
        flux=flux,
        spatial_average=spatial_average,
        provenance=provenance,
    )
