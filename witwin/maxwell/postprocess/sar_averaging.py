from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from ..sar import AVERAGING_PROFILE, SARAveraging, SARPeak

# The cubical-prefix-v1 mass-averaging algorithm.
#
# For a target averaging mass ``m0``, at every candidate center cell we grow an
# axis-aligned cube of half-width ``h`` cells (edge ``2h+1`` cells) and take the
# smallest ``h`` whose enclosed tissue mass reaches ``m0``; the averaged SAR is
# ``enclosed_power / enclosed_mass`` recorded with the ACTUAL enclosed mass.
#
# The enclosed mass / power / tissue-volume of any cube are read in O(1) from 3D
# inclusive prefix sums (integral images), so the per-center search costs one
# prefix lookup per half-width instead of an O(k^3) window sum. A brute-force
# O(N*k^3) reference reproduces this exactly (see tests).
#
# Validity rules (all recorded in provenance; NaN + mask, never silently padded):
#   * boundary_policy "strict-interior": a cube may not extend past the monitor
#     region, so the largest admissible half-width at a center is its Chebyshev
#     distance to the region boundary. If ``m0`` is unreachable within that
#     interior cube the center is invalid.
#   * the tissue fill fraction of the chosen cube must be >= min_tissue_fraction
#     (the "no air-mass makeup" rule).
#   * the center cell itself must carry tissue (a valid point-SAR cell); the
#     regulatory average is reported at tissue voxels.
#
# Differences from IEEE/IEC 62704-1 (documented, not certified): the cube is a
# symmetric index-space cube with no cube-face expansion asymmetry, and v1 has no
# tissue-connectivity flood fill (connectivity="cube" only). On a nonuniform grid
# the cube is symmetric in index space, not a physical cube.


def _padded_prefix(field: torch.Tensor) -> torch.Tensor:
    """Inclusive 3D prefix sum over the last three (spatial) dims, zero-padded.

    Returns a tensor whose last three dims are each one longer; entry ``S[..., a,
    b, c]`` is the sum of all cells with index ``< (a, b, c)`` along each axis, so
    an inclusive cube ``[x0..x1, y0..y1, z0..z1]`` is the standard 8-term
    inclusion-exclusion of ``S`` at ``x0 / x1+1`` (etc.).
    """
    cumulative = field.cumsum(-3).cumsum(-2).cumsum(-1)
    return F.pad(cumulative, (1, 0, 1, 0, 1, 0))


def _fixed_h_bounds(n: int, h: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Padded-prefix lower/upper index vectors for a half-width ``h`` cube at every center."""
    i = torch.arange(n, device=device)
    lo = (i - h).clamp(0, n - 1)
    hi = (i + h).clamp(0, n - 1)
    return lo, hi + 1


def _box_sum(prefix: torch.Tensor, bounds) -> torch.Tensor:
    """8-term inclusion-exclusion cube sum from a padded prefix.

    ``bounds`` = ``(xa, xb, ya, yb, za, zb)`` of per-axis padded index vectors
    (lengths ``nx, ny, nz``). Works with any number of leading (frequency) dims.
    """
    xa, xb, ya, yb, za, zb = bounds
    ax = xa.view(-1, 1, 1)
    bx = xb.view(-1, 1, 1)
    ay = ya.view(1, -1, 1)
    by = yb.view(1, -1, 1)
    az = za.view(1, 1, -1)
    bz = zb.view(1, 1, -1)

    def gather(ix, iy, iz):
        return prefix[..., ix, iy, iz]

    return (
        gather(bx, by, bz)
        - gather(ax, by, bz)
        - gather(bx, ay, bz)
        - gather(bx, by, az)
        + gather(ax, ay, bz)
        + gather(ax, by, az)
        + gather(bx, ay, az)
        - gather(ax, ay, az)
    )


def _center_max_halfwidth(shape: tuple[int, int, int], device) -> torch.Tensor:
    """Chebyshev distance from each center to the region boundary (interior cube limit)."""
    nx, ny, nz = shape
    ix = torch.arange(nx, device=device)
    iy = torch.arange(ny, device=device)
    iz = torch.arange(nz, device=device)
    hx = torch.minimum(ix, (nx - 1) - ix)[:, None, None]
    hy = torch.minimum(iy, (ny - 1) - iy)[None, :, None]
    hz = torch.minimum(iz, (nz - 1) - iz)[None, None, :]
    return torch.minimum(torch.minimum(hx, hy), hz)


def _peak_from_field(
    *,
    averaged_sar: torch.Tensor,
    final_valid: torch.Tensor,
    half_width: torch.Tensor,
    enclosed_mass: torch.Tensor,
    coordinates: Mapping[str, torch.Tensor],
    cell_sizes: Sequence[torch.Tensor],
    frequencies: torch.Tensor,
    mass_target: float,
) -> SARPeak:
    """Extract the per-frequency peak (max over valid centers) with position and cube facts."""
    freq_count = averaged_sar.shape[0]
    nx, ny, nz = final_valid.shape
    flat = averaged_sar.reshape(freq_count, -1)
    neg_inf = torch.full((), float("-inf"), device=flat.device, dtype=flat.dtype)
    masked = torch.where(torch.isnan(flat), neg_inf, flat)
    max_value, argmax = masked.max(dim=1)
    any_valid = bool(torch.any(final_valid))

    x, y, z = coordinates["x"], coordinates["y"], coordinates["z"]
    dx, dy, dz = cell_sizes

    sar_values = []
    indices = []
    positions = []
    masses = []
    half_widths = []
    cube_sizes = []
    nan = torch.full((), float("nan"), device=x.device, dtype=x.dtype)
    for f in range(freq_count):
        if not any_valid or not torch.isfinite(max_value[f]):
            sar_values.append(torch.full((), float("nan"), device=flat.device, dtype=flat.dtype))
            indices.append(torch.full((3,), -1, device=flat.device, dtype=torch.int64))
            positions.append(torch.full((3,), float("nan"), device=x.device, dtype=x.dtype))
            masses.append(nan.clone())
            half_widths.append(torch.full((), -1, device=flat.device, dtype=torch.int64))
            cube_sizes.append(torch.full((3,), float("nan"), device=x.device, dtype=x.dtype))
            continue
        idx = int(argmax[f])
        i = idx // (ny * nz)
        j = (idx // nz) % ny
        k = idx % nz
        h = int(half_width[i, j, k])
        # keep the graph on the peak value / mass (the max element is differentiable)
        sar_values.append(averaged_sar[f, i, j, k])
        indices.append(torch.tensor([i, j, k], device=flat.device, dtype=torch.int64))
        positions.append(torch.stack((x[i], y[j], z[k])))
        masses.append(enclosed_mass[i, j, k])
        half_widths.append(torch.full((), h, device=flat.device, dtype=torch.int64))
        lo_i, hi_i = max(i - h, 0), min(i + h, nx - 1)
        lo_j, hi_j = max(j - h, 0), min(j + h, ny - 1)
        lo_k, hi_k = max(k - h, 0), min(k + h, nz - 1)
        cube_sizes.append(
            torch.stack(
                (
                    dx[lo_i : hi_i + 1].sum(),
                    dy[lo_j : hi_j + 1].sum(),
                    dz[lo_k : hi_k + 1].sum(),
                )
            )
        )

    return SARPeak(
        mass_target=float(mass_target),
        sar=torch.stack(sar_values),
        frequencies=frequencies,
        index=torch.stack(indices),
        position=torch.stack(positions),
        mass_kg=torch.stack(masses),
        cube_half_width=torch.stack(half_widths),
        cube_size_m=torch.stack(cube_sizes),
        profile=AVERAGING_PROFILE,
    )


def compute_mass_averaged_sar(
    averaging: SARAveraging,
    *,
    power_total: torch.Tensor,
    rho_cell: torch.Tensor,
    cell_volume: torch.Tensor,
    occupancy: torch.Tensor,
    valid: torch.Tensor,
    coordinates: Mapping[str, torch.Tensor],
    cell_sizes: Sequence[torch.Tensor],
    frequencies: torch.Tensor,
) -> tuple[dict[float, Mapping], dict[float, SARPeak]]:
    """cubical-prefix-v1 mass-averaged SAR fields and peaks for each target mass.

    ``power_total`` is the colocated total absorbed-power density ``[F, nx, ny, nz]``
    (W/m^3); ``rho_cell`` is the occupancy-weighted effective density so the cell
    tissue mass is ``rho_cell * cell_volume`` and the enclosed mass is exact.
    Enclosed power / mass at the chosen cube keep their autograd graph (fixed-window
    averaged SAR is differentiable); the discrete half-width search is stop-grad.
    """
    device = rho_cell.device
    shape = tuple(rho_cell.shape)
    nx, ny, nz = shape

    mass_cell = rho_cell * cell_volume                 # [nx,ny,nz] (grad through rho)
    power_cell = power_total * cell_volume[None]        # [F,nx,ny,nz] (grad through fields)
    tissue_vol_cell = occupancy * cell_volume           # tissue volume per cell
    total_vol_cell = cell_volume                        # cube total volume per cell

    prefix_mass = _padded_prefix(mass_cell)
    prefix_power = _padded_prefix(power_cell)
    prefix_tissue_vol = _padded_prefix(tissue_vol_cell).detach()
    prefix_total_vol = _padded_prefix(total_vol_cell).detach()
    prefix_mass_scan = prefix_mass.detach()

    max_halfwidth = _center_max_halfwidth(shape, device)
    global_hmax = int(max_halfwidth.max()) if max_halfwidth.numel() else 0
    min_fraction = float(averaging.min_tissue_fraction)

    averaged: dict[float, Mapping] = {}
    peaks: dict[float, SARPeak] = {}

    for m0 in averaging.mass:
        found = torch.zeros(shape, dtype=torch.bool, device=device)
        half_width = torch.full(shape, -1, dtype=torch.int64, device=device)
        for h in range(global_hmax + 1):
            bounds = (
                *_fixed_h_bounds(nx, h, device),
                *_fixed_h_bounds(ny, h, device),
                *_fixed_h_bounds(nz, h, device),
            )
            box_mass = _box_sum(prefix_mass_scan, bounds)
            within_interior = h <= max_halfwidth
            reached = (box_mass >= m0) & within_interior & (~found)
            half_width = torch.where(reached, torch.full_like(half_width, h), half_width)
            found = found | reached

        center_valid = found & valid

        enclosed_power = torch.zeros_like(power_cell)
        enclosed_mass = torch.zeros_like(mass_cell)
        tissue_fraction = torch.zeros_like(mass_cell)
        if bool(torch.any(center_valid)):
            unique_h = torch.unique(half_width[center_valid]).tolist()
            for h in unique_h:
                bounds = (
                    *_fixed_h_bounds(nx, int(h), device),
                    *_fixed_h_bounds(ny, int(h), device),
                    *_fixed_h_bounds(nz, int(h), device),
                )
                selection = (half_width == h) & center_valid
                box_power = _box_sum(prefix_power, bounds)
                box_mass = _box_sum(prefix_mass, bounds)
                box_tissue_vol = _box_sum(prefix_tissue_vol, bounds)
                box_total_vol = _box_sum(prefix_total_vol, bounds)
                enclosed_power = torch.where(selection[None], box_power, enclosed_power)
                enclosed_mass = torch.where(selection, box_mass, enclosed_mass)
                tissue_fraction = torch.where(
                    selection, box_tissue_vol / box_total_vol, tissue_fraction
                )

        fraction_ok = tissue_fraction >= min_fraction
        final_valid = center_valid & fraction_ok

        nan = torch.full((), float("nan"), device=device, dtype=power_cell.dtype)
        safe_mass = torch.where(final_valid, enclosed_mass, torch.ones_like(enclosed_mass))
        averaged_sar = torch.where(
            final_valid[None], enclosed_power / safe_mass[None], nan
        )
        mass_field = torch.where(
            final_valid, enclosed_mass, torch.full_like(enclosed_mass, float("nan"))
        )
        half_field = torch.where(final_valid, half_width, torch.full_like(half_width, -1))

        averaged[float(m0)] = MappingProxyType(
            {
                "sar": averaged_sar,
                "mass_kg": mass_field,
                "cube_half_width": half_field,
                "tissue_fraction": torch.where(
                    final_valid, tissue_fraction, torch.full_like(tissue_fraction, float("nan"))
                ),
                "valid": final_valid,
            }
        )
        peaks[float(m0)] = _peak_from_field(
            averaged_sar=averaged_sar,
            final_valid=final_valid,
            half_width=half_field,
            enclosed_mass=enclosed_mass,
            coordinates=coordinates,
            cell_sizes=cell_sizes,
            frequencies=frequencies,
            mass_target=float(m0),
        )

    return averaged, peaks


__all__ = ["compute_mass_averaged_sar"]
