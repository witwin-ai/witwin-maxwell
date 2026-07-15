from __future__ import annotations

from typing import Mapping

import torch

from ..compiler.power_loss import CompiledPowerLossMonitor
from ..power_loss import PowerLossData


_ELECTRIC_COMPONENTS = ("Ex", "Ey", "Ez")


def _validate_frequencies(
    frequencies: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(frequencies, torch.Tensor):
        raise TypeError("frequencies must be a torch.Tensor.")
    if frequencies.device != device:
        raise ValueError(f"frequencies must be on device {device}.")
    if frequencies.is_complex() or not frequencies.dtype.is_floating_point:
        raise TypeError("frequencies must be real floating-point data.")
    if frequencies.ndim != 1 or frequencies.numel() == 0:
        raise ValueError("frequencies must have non-empty shape [F].")
    if not bool(torch.all(torch.isfinite(frequencies))) or not bool(
        torch.all(frequencies > 0.0)
    ):
        raise ValueError("frequencies must be finite and strictly positive.")
    return frequencies


def _selected_spatial_tensor(
    value,
    *,
    name: str,
    component: str,
    compiled: CompiledPowerLossMonitor,
    allow_occupancy: bool,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name}[{component!r}] must be a torch.Tensor.")
    if value.device != compiled.device:
        raise ValueError(f"{name} tensors must be on device {compiled.device}.")
    mask = compiled.component_masks[component]
    selected_count = compiled.component_volumes[component].numel()
    if value.shape == mask.shape:
        selected = value[mask]
    elif value.shape == (selected_count,):
        selected = value
    else:
        raise ValueError(
            f"{name}[{component!r}] must match the full Yee shape or selected shape [N]."
        )
    if allow_occupancy:
        if selected.is_complex() or not selected.dtype.is_floating_point:
            raise TypeError(f"{name}[{component!r}] must be real floating-point data.")
    elif (
        selected.dtype == torch.bool
        or selected.dtype.is_floating_point
        or selected.is_complex()
    ):
        raise TypeError(f"{name}[{component!r}] must contain integer identifiers.")
    return selected


def _selected_component_metadata(
    values,
    *,
    name: str,
    compiled: CompiledPowerLossMonitor,
    allow_occupancy: bool,
):
    if values is None:
        return None
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a component mapping or None.")
    resolved = {}
    for component, value in values.items():
        if component not in _ELECTRIC_COMPONENTS:
            raise ValueError(f"{name} contains unknown component {component!r}.")
        resolved[component] = _selected_spatial_tensor(
            value,
            name=name,
            component=component,
            compiled=compiled,
            allow_occupancy=allow_occupancy,
        )
    return resolved


def _validate_electric_fields(
    electric_fields,
    *,
    frequency_count: int,
    compiled: CompiledPowerLossMonitor,
) -> dict[str, torch.Tensor]:
    if not isinstance(electric_fields, Mapping):
        raise TypeError("electric_fields must map Ex, Ey, and Ez to complex tensors.")
    if set(electric_fields) != set(_ELECTRIC_COMPONENTS):
        raise ValueError("electric_fields must contain exactly Ex, Ey, and Ez.")
    resolved = {}
    for component in _ELECTRIC_COMPONENTS:
        field = electric_fields[component]
        if not isinstance(field, torch.Tensor):
            raise TypeError(f"electric_fields[{component!r}] must be a torch.Tensor.")
        if not field.is_complex():
            raise TypeError(f"electric_fields[{component!r}] must be a complex tensor.")
        expected_shape = (frequency_count, *compiled.full_component_shapes[component])
        if field.shape != expected_shape:
            raise ValueError(
                f"electric_fields[{component!r}] must have shape [F, ...]."
            )
        if field.device != compiled.device:
            raise ValueError(f"electric_fields must be on device {compiled.device}.")
        if not bool(torch.all(torch.isfinite(field.real))) or not bool(
            torch.all(torch.isfinite(field.imag))
        ):
            raise ValueError(
                f"electric_fields[{component!r}] must contain only finite values."
            )
        resolved[component] = field
    return resolved


def _select_volume_density(
    density,
    *,
    channel: str,
    component: str,
    frequency_count: int,
    compiled: CompiledPowerLossMonitor,
) -> torch.Tensor:
    if not isinstance(density, torch.Tensor):
        raise TypeError(
            f"volume_channels[{channel!r}][{component!r}] must be a torch.Tensor."
        )
    if density.device != compiled.device:
        raise ValueError(f"volume channel tensors must be on device {compiled.device}.")
    if density.is_complex() or not density.dtype.is_floating_point:
        raise TypeError("Volume loss density must be real floating-point data.")
    mask = compiled.component_masks[component]
    selected_count = compiled.component_volumes[component].numel()
    if density.shape == (frequency_count, *mask.shape):
        selected = density[:, mask]
    elif density.shape == (frequency_count, selected_count):
        selected = density
    else:
        raise ValueError(
            f"volume_channels[{channel!r}][{component!r}] must have shape [F, ...] "
            "on the full grid or [F, N] after selection."
        )
    if not bool(torch.all(torch.isfinite(selected))) or not bool(
        torch.all(selected >= 0.0)
    ):
        raise ValueError("Volume loss density must be finite and non-negative.")
    return selected


def _resolve_volume_channels(
    channels,
    *,
    frequency_count: int,
    compiled: CompiledPowerLossMonitor,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    if channels is None:
        return {}, {}
    if not isinstance(channels, Mapping):
        raise TypeError("volume_channels must be a channel mapping or None.")
    densities = {}
    powers = {}
    for channel, component_map in channels.items():
        channel_name = str(channel)
        if channel_name == "conduction":
            raise ValueError(
                "conduction is computed from electric_fields and compiled conductivity."
            )
        if not isinstance(component_map, Mapping) or not component_map:
            raise ValueError(
                f"volume_channels[{channel_name!r}] must contain component densities."
            )
        selected_components = {}
        component_powers = []
        for component, density in component_map.items():
            if component not in _ELECTRIC_COMPONENTS:
                raise ValueError(
                    f"volume_channels[{channel_name!r}] contains unknown component {component!r}."
                )
            selected = _select_volume_density(
                density,
                channel=channel_name,
                component=component,
                frequency_count=frequency_count,
                compiled=compiled,
            )
            selected_components[component] = selected
            component_powers.append(
                torch.sum(
                    selected * compiled.component_volumes[component][None, :],
                    dim=1,
                )
            )
        densities[channel_name] = selected_components
        powers[channel_name] = torch.stack(component_powers, dim=0).sum(dim=0)
    return densities, powers


def _resolve_integrated_channels(
    channels,
    *,
    frequency_count: int,
    compiled: CompiledPowerLossMonitor,
) -> dict[str, torch.Tensor]:
    if channels is None:
        return {}
    if not isinstance(channels, Mapping):
        raise TypeError("integrated_channels must be a channel mapping or None.")
    resolved = {}
    for channel, power in channels.items():
        channel_name = str(channel)
        if channel_name == "conduction":
            raise ValueError(
                "conduction is computed from electric_fields and compiled conductivity."
            )
        if not isinstance(power, torch.Tensor):
            raise TypeError(
                f"integrated_channels[{channel_name!r}] must be a torch.Tensor."
            )
        if power.shape != (frequency_count,):
            raise ValueError(
                f"integrated_channels[{channel_name!r}] must have shape [F]."
            )
        if power.device != compiled.device:
            raise ValueError(
                f"integrated channel tensors must be on device {compiled.device}."
            )
        if power.is_complex() or not power.dtype.is_floating_point:
            raise TypeError(
                "Integrated loss channels must be real floating-point data."
            )
        if not bool(torch.all(torch.isfinite(power))) or not bool(
            torch.all(power >= 0.0)
        ):
            raise ValueError(
                "Integrated loss channels must be finite and non-negative."
            )
        resolved[channel_name] = power
    return resolved


def _resolve_measure_channels(
    channels,
    measures,
    *,
    channels_name: str,
    measures_name: str,
    frequency_count: int,
    compiled: CompiledPowerLossMonitor,
):
    if channels is None and measures is None:
        return {}, {}, {}
    if not isinstance(channels, Mapping) or not isinstance(measures, Mapping):
        raise TypeError(
            f"{channels_name} and {measures_name} must be supplied together as mappings."
        )
    resolved_measures = {}
    for entity, measure in measures.items():
        entity_name = str(entity)
        if not entity_name:
            raise ValueError(f"{measures_name} keys must not be empty.")
        if not isinstance(measure, torch.Tensor):
            raise TypeError(f"{measures_name}[{entity_name!r}] must be a torch.Tensor.")
        if measure.device != compiled.device:
            raise ValueError(f"{measures_name} tensors must be on device {compiled.device}.")
        if measure.ndim != 1 or measure.numel() == 0:
            raise ValueError(f"{measures_name}[{entity_name!r}] must have shape [N].")
        if measure.is_complex() or not measure.dtype.is_floating_point:
            raise TypeError(f"{measures_name} tensors must be real floating-point data.")
        if not bool(torch.all(torch.isfinite(measure))) or not bool(torch.all(measure > 0.0)):
            raise ValueError(f"{measures_name} tensors must be finite and positive.")
        resolved_measures[entity_name] = measure

    resolved_density = {}
    resolved_power = {}
    for channel, entity_map in channels.items():
        channel_name = str(channel)
        if not isinstance(entity_map, Mapping) or not entity_map:
            raise ValueError(f"{channels_name}[{channel_name!r}] must contain entity densities.")
        entity_density = {}
        entity_powers = []
        for entity, density in entity_map.items():
            entity_name = str(entity)
            if entity_name not in resolved_measures:
                raise ValueError(
                    f"{channels_name}[{channel_name!r}] references unknown entity {entity_name!r}."
                )
            if not isinstance(density, torch.Tensor):
                raise TypeError(
                    f"{channels_name}[{channel_name!r}][{entity_name!r}] must be a torch.Tensor."
                )
            expected_shape = (frequency_count, resolved_measures[entity_name].numel())
            if density.shape != expected_shape:
                raise ValueError(
                    f"{channels_name}[{channel_name!r}][{entity_name!r}] must have shape [F, N]."
                )
            if density.device != compiled.device:
                raise ValueError(f"{channels_name} tensors must be on device {compiled.device}.")
            if density.is_complex() or not density.dtype.is_floating_point:
                raise TypeError(f"{channels_name} tensors must be real floating-point data.")
            if not bool(torch.all(torch.isfinite(density))) or not bool(torch.all(density >= 0.0)):
                raise ValueError(f"{channels_name} tensors must be finite and non-negative.")
            entity_density[entity_name] = density
            entity_powers.append(
                torch.sum(density * resolved_measures[entity_name][None, :], dim=1)
            )
        resolved_density[channel_name] = entity_density
        resolved_power[channel_name] = torch.stack(entity_powers, dim=0).sum(dim=0)
    return resolved_density, resolved_power, resolved_measures


def compute_power_loss_data(
    compiled: CompiledPowerLossMonitor,
    frequencies: torch.Tensor,
    *,
    electric_fields: Mapping[str, torch.Tensor] | None = None,
    volume_channels: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    integrated_channels: Mapping[str, torch.Tensor] | None = None,
    surface_channels: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    face_areas: Mapping[str, torch.Tensor] | None = None,
    line_channels: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    line_lengths: Mapping[str, torch.Tensor] | None = None,
    occupancy: Mapping[str, torch.Tensor] | None = None,
    material_ids: Mapping[str, torch.Tensor] | None = None,
    geometry_ids: Mapping[str, torch.Tensor] | None = None,
    source_result_fingerprint: str | None = None,
    autograd_provenance: str = "live graph is preserved from supplied torch tensors; persisted data is detached",
) -> PowerLossData:
    """Compute available loss channels without inventing unsupported components.

    ``conduction`` is derived from compiled static bulk electric conductivity and
    peak-phasor electric fields. Other requested channels must be supplied as
    volume density in W/m3, surface density in W/m2, line density in W/m, or
    already integrated power in W. All tensors stay on the monitor's single
    device.
    """

    if not isinstance(compiled, CompiledPowerLossMonitor):
        raise TypeError("compiled must be a CompiledPowerLossMonitor instance.")
    frequencies = _validate_frequencies(frequencies, device=compiled.device)
    frequency_count = frequencies.numel()
    monitor_frequencies = compiled.monitor.frequencies
    if monitor_frequencies is not None:
        requested = torch.as_tensor(
            monitor_frequencies,
            device=compiled.device,
            dtype=frequencies.dtype,
        )
        if requested.shape != frequencies.shape or not torch.equal(
            requested, frequencies
        ):
            raise ValueError(
                "frequencies must match the compiled PowerLossMonitor frequencies."
            )

    volume_density, channel_power = _resolve_volume_channels(
        volume_channels,
        frequency_count=frequency_count,
        compiled=compiled,
    )
    integrated = _resolve_integrated_channels(
        integrated_channels,
        frequency_count=frequency_count,
        compiled=compiled,
    )
    surface_density, surface_power, resolved_face_areas = _resolve_measure_channels(
        surface_channels,
        face_areas,
        channels_name="surface_channels",
        measures_name="face_areas",
        frequency_count=frequency_count,
        compiled=compiled,
    )
    line_density, line_power, resolved_line_lengths = _resolve_measure_channels(
        line_channels,
        line_lengths,
        channels_name="line_channels",
        measures_name="line_lengths",
        frequency_count=frequency_count,
        compiled=compiled,
    )
    supplied_groups = (channel_power, integrated, surface_power, line_power)
    duplicates = set()
    for group_index, group in enumerate(supplied_groups):
        for other in supplied_groups[group_index + 1 :]:
            duplicates.update(set(group) & set(other))
    duplicate_channels = duplicates
    if duplicate_channels:
        raise ValueError(
            "Loss channels must use exactly one density/integrated representation: "
            f"{tuple(sorted(duplicate_channels))}."
        )
    channel_power.update(integrated)
    channel_power.update(surface_power)
    channel_power.update(line_power)

    requests_conduction = "conduction" in compiled.monitor.channels
    if requests_conduction:
        if compiled.conductivity is None:
            raise RuntimeError("Compiled conduction conductivity is unavailable.")
        fields = _validate_electric_fields(
            electric_fields,
            frequency_count=frequency_count,
            compiled=compiled,
        )
        conduction_density = {}
        component_powers = []
        for component in _ELECTRIC_COMPONENTS:
            selected_field = fields[component][:, compiled.component_masks[component]]
            density = (
                0.5
                * compiled.conductivity[component][None, :]
                * torch.abs(selected_field).square()
            )
            conduction_density[component] = density
            component_powers.append(
                torch.sum(
                    density * compiled.component_volumes[component][None, :],
                    dim=1,
                )
            )
        volume_density["conduction"] = conduction_density
        channel_power["conduction"] = torch.stack(component_powers, dim=0).sum(dim=0)
    elif electric_fields is not None:
        raise ValueError(
            "electric_fields were supplied but the monitor does not request conduction."
        )

    requested_channels = tuple(compiled.monitor.channels)
    supplied_channels = set(channel_power)
    missing = tuple(
        channel for channel in requested_channels if channel not in supplied_channels
    )
    extra = tuple(sorted(supplied_channels - set(requested_channels)))
    if missing:
        raise ValueError(
            f"No physical input was supplied for requested power-loss channels {missing}."
        )
    if extra:
        raise ValueError(f"Power-loss inputs contain unrequested channels {extra}.")

    ordered_power = {channel: channel_power[channel] for channel in requested_channels}
    ordered_density = {
        channel: volume_density[channel]
        for channel in requested_channels
        if channel in volume_density
    }
    total = torch.stack(tuple(ordered_power.values()), dim=0).sum(dim=0)
    selected_occupancy = _selected_component_metadata(
        occupancy,
        name="occupancy",
        compiled=compiled,
        allow_occupancy=True,
    )
    selected_material_ids = _selected_component_metadata(
        material_ids,
        name="material_ids",
        compiled=compiled,
        allow_occupancy=False,
    )
    selected_geometry_ids = _selected_component_metadata(
        geometry_ids,
        name="geometry_ids",
        compiled=compiled,
        allow_occupancy=False,
    )
    return PowerLossData(
        frequencies=frequencies,
        channel_power=ordered_power,
        volume_density=ordered_density,
        total=total,
        component_volumes=compiled.component_volumes,
        global_ids=compiled.global_ids,
        monitor_name=compiled.monitor.name,
        bounds=compiled.monitor.bounds,
        occupancy=selected_occupancy,
        material_ids=selected_material_ids,
        geometry_ids=selected_geometry_ids,
        surface_density=surface_density,
        face_areas=resolved_face_areas,
        line_density=line_density,
        line_lengths=resolved_line_lengths,
        source_result_fingerprint=source_result_fingerprint,
        autograd_provenance=autograd_provenance,
    )


__all__ = ["compute_power_loss_data"]
