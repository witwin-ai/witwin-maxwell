from __future__ import annotations

import hashlib
from types import MappingProxyType

import numpy as np
import torch

from ..compiler.mass_density import (
    BACKGROUND_TISSUE_ID,
    OCCUPANCY_EPSILON,
    CompiledMassDensity,
)
from ..compiler.materials import _bulk_structures, _structure_material
from ..compiler.power_loss import CompiledPowerLossMonitor
from ..power_loss import PowerLossData
from ..sar import PowerNormalization, SARAveraging, SARResult

# Volumetric electric-loss channels that dissipate power into tissue and are the
# regulatory SAR numerator. Magnetic dispersion, surface, wire, and circuit
# channels are not electric tissue loss and are excluded from SAR.
ELECTRIC_SAR_CHANNELS = ("conduction", "electric_dispersion", "nonlinear")

_AXIS_REDUCED_DIM = {"Ex": 0, "Ey": 1, "Ez": 2}


def _monitor_node_mask(scene, bounds) -> torch.Tensor:
    selections = []
    for coordinate, (lower, upper) in zip((scene.x, scene.y, scene.z), bounds):
        selections.append((coordinate >= lower) & (coordinate <= upper))
    return (
        selections[0][:, None, None]
        & selections[1][None, :, None]
        & selections[2][None, None, :]
    )


def _reject_lossy_material_without_density(scene, monitor):
    """Fail closed when the SAR region covers electric loss with no mass density."""
    region_mask = _monitor_node_mask(scene, monitor.bounds)
    for structure in _bulk_structures(scene):
        material = _structure_material(structure)
        if material is None:
            continue
        if not getattr(material, "is_electrically_lossy", False):
            continue
        if getattr(material, "mass_density", None) is not None:
            continue
        from ..compiler.materials import _geometry_occupancy, _static_periodic_shift_options

        occupancy = _geometry_occupancy(
            scene,
            structure.geometry,
            periodic_shift_options=_static_periodic_shift_options(scene, structure.geometry),
        )
        if bool(torch.any((occupancy > OCCUPANCY_EPSILON) & region_mask)):
            name = getattr(material, "name", None) or type(material).__name__
            raise ValueError(
                f"SAR request covers electrically lossy material {name!r} without a "
                "mass_density; assign mass_density to every lossy material in the monitor "
                "region, or shrink the PowerLossMonitor to exclude it."
            )


def _scatter_edge_power_to_nodes(edge_full: torch.Tensor, component: str) -> torch.Tensor:
    """Colocate a staggered electric-edge power grid to the node grid, conserving power.

    Each edge distributes half of its power to each of the two nodes it spans (the
    power-conserving transpose of the node-to-edge midpoint average the EM/loss
    compiler uses). Total power is preserved exactly, so the volume integral of the
    colocated density closes against the edge-integrated channel power.
    """
    dim = _AXIS_REDUCED_DIM[component] + 1  # +1 for the leading frequency axis
    half = 0.5 * edge_full
    zero_shape = list(half.shape)
    zero_shape[dim] = 1
    zero = torch.zeros(zero_shape, device=half.device, dtype=half.dtype)
    lower = torch.cat((half, zero), dim=dim)
    higher = torch.cat((zero, half), dim=dim)
    return lower + higher


def _channel_node_power(
    power_loss: PowerLossData,
    compiled: CompiledPowerLossMonitor,
    channel: str,
    node_shape: tuple[int, int, int],
    frequency_count: int,
) -> torch.Tensor:
    node_power = torch.zeros(
        (frequency_count, *node_shape),
        device=power_loss.device,
        dtype=torch.float32,
    )
    component_map = power_loss.volume_density[channel]
    for component, density in component_map.items():
        mask = compiled.component_masks[component]
        volumes = compiled.component_volumes[component]
        edge_power = density * volumes[None, :]
        edge_full = torch.zeros(
            (frequency_count, *mask.shape),
            device=power_loss.device,
            dtype=edge_power.dtype,
        )
        edge_full[:, mask] = edge_power
        node_power = node_power + _scatter_edge_power_to_nodes(edge_full, component)
    return node_power.to(torch.float32)


def _region_bounds(active: torch.Tensor) -> tuple[slice, slice, slice]:
    if not bool(torch.any(active)):
        raise ValueError(
            "SAR region is empty: no cell in the monitor carries absorbed power or tissue."
        )
    indices = torch.nonzero(active, as_tuple=False)
    lows = indices.amin(dim=0)
    highs = indices.amax(dim=0)
    return tuple(
        slice(int(lows[axis]), int(highs[axis]) + 1) for axis in range(3)
    )


def _grid_hash(scene, region: tuple[slice, slice, slice]) -> str:
    hasher = hashlib.sha256()
    for name in ("x_nodes64", "y_nodes64", "z_nodes64"):
        hasher.update(np.ascontiguousarray(getattr(scene, name), dtype=np.float64).tobytes())
    hasher.update(
        np.asarray(
            [(sl.start, sl.stop) for sl in region], dtype=np.int64
        ).tobytes()
    )
    return hasher.hexdigest()


def _broadcast_power_scale(power_scale, device) -> torch.Tensor | float:
    """Shape a scalar or ``[F]`` power scale to multiply a ``[F, nx, ny, nz]`` field."""
    if torch.is_tensor(power_scale):
        scale = power_scale.to(device)
        if scale.ndim == 0:
            return scale
        return scale.reshape(-1, 1, 1, 1)
    return float(power_scale)


def compute_tissue_statistics(
    *,
    tissue_id: torch.Tensor,
    valid: torch.Tensor,
    rho_cell: torch.Tensor,
    cell_volume: torch.Tensor,
    total_power: torch.Tensor,
    total_sar: torch.Tensor,
    tissue_names,
) -> dict[int, MappingProxyType]:
    """Per-tissue absorbed power and SAR summaries over the valid cells of a region.

    ``total_power`` is ``[F, nx, ny, nz]`` colocated absorbed power (W) and
    ``total_sar`` the matching point SAR (W/kg); background/invalid cells are
    excluded. Shared by the point-SAR reducer and the incoherent combiner so both
    report identical statistics conventions.
    """
    statistics: dict[int, MappingProxyType] = {}
    for tid in torch.unique(tissue_id).tolist():
        if int(tid) == BACKGROUND_TISSUE_ID:
            continue
        tissue_mask = (tissue_id == tid) & valid
        if not bool(torch.any(tissue_mask)):
            continue
        selected_power = total_power[:, tissue_mask]
        selected_sar = total_sar[:, tissue_mask]
        tissue_mass = (rho_cell * cell_volume)[tissue_mask].sum()
        statistics[int(tid)] = MappingProxyType(
            {
                "name": tissue_names.get(int(tid), f"tissue_{int(tid)}"),
                "cell_count": int(tissue_mask.sum()),
                "mass_kg": tissue_mass,
                "total_absorbed_power": selected_power.sum(dim=1),
                "mean_sar": selected_sar.mean(dim=1),
                "max_sar": selected_sar.amax(dim=1),
            }
        )
    return statistics


def compute_sar(
    *,
    prepared_scene,
    monitor,
    power_loss: PowerLossData,
    mass: CompiledMassDensity,
    compiled_loss: CompiledPowerLossMonitor,
    normalization: PowerNormalization,
    averaging: SARAveraging | None = None,
    power_scale=None,
) -> SARResult:
    """Reduce absorbed-power density and a tissue mass model to point SAR.

    Pure result-domain reduction: no solver is run. Fails closed if the region
    carries electric loss in cells with no mass density, or if no electric
    volumetric loss channel is available. ``power_scale`` is the resolved
    multiplicative power scale (scalar or per-frequency ``[F]`` tensor). When
    ``None`` it is resolved from ``normalization`` (source-amplitude only); the
    port-accepted-power scale is resolved by ``Result.sar`` which carries the port.
    """
    if not isinstance(power_loss, PowerLossData):
        raise TypeError("power_loss must be a PowerLossData instance.")
    if not isinstance(mass, CompiledMassDensity):
        raise TypeError("mass must be a CompiledMassDensity instance.")
    if not isinstance(normalization, PowerNormalization):
        raise TypeError("normalization must be a PowerNormalization instance.")

    _reject_lossy_material_without_density(prepared_scene, monitor)

    channels = tuple(
        channel
        for channel in ELECTRIC_SAR_CHANNELS
        if channel in power_loss.volume_density
    )
    if not channels:
        raise ValueError(
            "SAR requires a volumetric electric-loss channel (conduction / "
            f"electric_dispersion / nonlinear); the monitor supplied {tuple(power_loss.volume_density)}."
        )

    node_shape = mass.shape
    frequency_count = power_loss.frequencies.numel()
    if power_scale is None:
        power_scale = normalization.resolve_scale(result=None)
    scale = _broadcast_power_scale(power_scale, power_loss.device)

    node_power_by_channel = {}
    for channel in channels:
        node_power = _channel_node_power(
            power_loss, compiled_loss, channel, node_shape, frequency_count
        )
        node_power_by_channel[channel] = scale * node_power
    node_power_total = sum(node_power_by_channel.values())

    occupancy_full = mass.occupancy
    # The SAR region is the monitor's node coverage, widened to the one-cell
    # collocation skirt where interface-edge power lands. It is NOT the full tissue
    # extent: tissue outside the monitor has no measured absorbed power.
    active = _monitor_node_mask(prepared_scene, monitor.bounds) | (
        node_power_total.sum(dim=0) > 0
    )
    region = _region_bounds(active)

    def _crop_spatial(tensor):
        return tensor[region[0], region[1], region[2]]

    def _crop_field(tensor):
        return tensor[:, region[0], region[1], region[2]]

    rho_cell = _crop_spatial(mass.rho_cell)
    occupancy = _crop_spatial(occupancy_full)
    tissue_id = _crop_spatial(mass.tissue_id)
    cell_volume = _crop_spatial(mass.cell_volume)

    valid = (occupancy >= OCCUPANCY_EPSILON) & (rho_cell > 0)
    safe_rho = torch.where(valid, rho_cell, torch.ones_like(rho_cell))
    nan = torch.full((), float("nan"), device=rho_cell.device, dtype=torch.float32)

    point = {}
    absorbed_power_density = {}
    total_q = None
    for channel, node_power in node_power_by_channel.items():
        q = _crop_field(node_power) / cell_volume[None]
        absorbed_power_density[channel] = q
        sar = torch.where(valid[None], q / safe_rho[None], nan)
        point[channel] = sar
        total_q = q if total_q is None else total_q + q
    absorbed_power_density["total"] = total_q
    point["total"] = torch.where(valid[None], total_q / safe_rho[None], nan)

    node_power_total_region = _crop_field(node_power_total)
    statistics = compute_tissue_statistics(
        tissue_id=tissue_id,
        valid=valid,
        rho_cell=rho_cell,
        cell_volume=cell_volume,
        total_power=node_power_total_region,
        total_sar=point["total"],
        tissue_names=mass.tissue_names,
    )

    x = prepared_scene.x[region[0]]
    y = prepared_scene.y[region[1]]
    z = prepared_scene.z[region[2]]
    coordinates = MappingProxyType({"x": x, "y": y, "z": z})

    device = rho_cell.device
    dx = torch.as_tensor(prepared_scene.dx_dual64, device=device, dtype=torch.float32)[region[0]]
    dy = torch.as_tensor(prepared_scene.dy_dual64, device=device, dtype=torch.float32)[region[1]]
    dz = torch.as_tensor(prepared_scene.dz_dual64, device=device, dtype=torch.float32)[region[2]]
    cell_sizes = (dx, dy, dz)

    averaged = {}
    peaks = {}
    if averaging is not None:
        from .sar_averaging import compute_mass_averaged_sar

        averaged, peaks = compute_mass_averaged_sar(
            averaging,
            power_total=absorbed_power_density["total"],
            rho_cell=rho_cell,
            cell_volume=cell_volume,
            occupancy=occupancy,
            valid=valid,
            coordinates=coordinates,
            cell_sizes=cell_sizes,
            frequencies=power_loss.frequencies,
        )

    electric_total = torch.stack(
        [power_loss.channel_power[channel] for channel in channels], dim=0
    ).sum(dim=0)

    if torch.is_tensor(power_scale) and power_scale.ndim > 0:
        power_scale_record = power_scale.detach().to("cpu").tolist()
    else:
        power_scale_record = float(power_scale)

    provenance = MappingProxyType(
        {
            "sar_unit": "W/kg",
            "power_unit": "W",
            "field_convention": power_loss.phasor_convention,
            "loss_normalization": power_loss.normalization,
            "channels": channels,
            "collocation": (
                "electric edge loss density colocated to material cell centers by "
                "power-conserving half-weight scatter; effective mass density is "
                "occupancy-weighted so the occupancy cancels in point SAR"
            ),
            "occupancy_epsilon": OCCUPANCY_EPSILON,
            "power_scale": power_scale_record,
            "cell_sizes_dual": tuple(size.detach() for size in cell_sizes),
            "monitor": monitor.name,
            "monitor_bounds": monitor.bounds,
            "region_index_bounds": tuple((sl.start, sl.stop) for sl in region),
            "grid_hash": _grid_hash(prepared_scene, region),
            "averaging_profile": None if averaging is None else averaging.payload(),
            "electric_channel_power": electric_total.detach().to("cpu"),
            "boundary_note": (
                "at internal tissue/air interfaces a fraction of interface-edge power "
                "colocates to air-side cells and is excluded from per-tissue statistics; "
                "the region volume integral still closes against the edge-integrated power"
            ),
        }
    )

    return SARResult(
        point=MappingProxyType(point),
        statistics=MappingProxyType(statistics),
        normalization=normalization,
        provenance=provenance,
        frequencies=power_loss.frequencies,
        coordinates=coordinates,
        valid=valid,
        occupancy=occupancy,
        rho_cell=rho_cell,
        cell_volume=cell_volume,
        tissue_id=tissue_id,
        tissue_names=mass.tissue_names,
        absorbed_power_density=MappingProxyType(absorbed_power_density),
        averaged=MappingProxyType(averaged),
        peaks=MappingProxyType(peaks),
    )


__all__ = ["compute_sar", "compute_tissue_statistics", "ELECTRIC_SAR_CHANNELS"]
